// port-lint: source llama.cpp/src/llama-batch.h llama.cpp/src/llama-batch.cpp
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import kotlinx.cinterop.toKString

// =============================================================================
// LlamaUBatchInternal – the micro-batch unit used by the batch allocator
// Ported from: llama-batch.h  struct llama_ubatch
// =============================================================================

/**
 * Internal micro-batch used by the batch allocator during ubatch splitting.
 *
 * This is the **internal** counterpart of the public [LlamaUBatch] declared in
 * `LlamaAttention.kt`. It carries the additional bookkeeping arrays (`nSeqId`,
 * `seqIdUnq`, `seqIdx`, mutable data ownership) that the splitting logic needs
 * but the external API does not expose.
 *
 * C++ analogue: `struct llama_ubatch` in `llama-batch.h`.
 *
 * @property equalSeqs  `true` when every sequence set in this ubatch has
 *                      the same number of tokens — enables optimised paths.
 * @property nTokens    Total number of tokens (`nSeqTokens * nSeqs`).
 * @property nSeqTokens Tokens per sequence set.
 * @property nSeqs      Number of sequence sets in the ubatch.
 * @property nSeqsUnq   Number of **unique** sequence ids.
 * @property nPos       Number of position inputs per token/embedding (≥1;
 *                      M-RoPE may use 3+).
 */
data class LlamaUBatchInternal(
    var equalSeqs: Boolean = false,
    var nTokens: Int = 0,
    var nSeqTokens: Int = 0,
    var nSeqs: Int = 0,
    var nSeqsUnq: Int = 0,
    var nPos: Int = 1,

    // -- payload pointers (backed by `data` when owning) --
    var token: IntArray? = null,
    var embd: FloatArray? = null,
    var pos: IntArray? = null,
    var nSeqId: IntArray? = null,
    var seqId: Array<IntArray>? = null,
    var seqIdUnq: IntArray? = null,
    var seqIdx: IntArray? = null,
    var output: ByteArray? = null,

    /** Owned data storage. When non-null the arrays above point into these. */
    var data: UBatchData? = null,
) {
    /** `true` when multi-position (M-RoPE) mode is active. */
    fun isPos2d(): Boolean = nPos >= 3
}

/**
 * Owned storage for a [LlamaUBatchInternal].
 *
 * Port of `llama_ubatch::data_t`.
 */
data class UBatchData(
    val token: MutableList<Int> = mutableListOf(),
    val embd: MutableList<Float> = mutableListOf(),
    val pos: MutableList<Int> = mutableListOf(),
    val nSeqId: MutableList<Int> = mutableListOf(),
    val seqIdUnq: MutableList<Int> = mutableListOf(),
    val seqIdx: MutableList<Int> = mutableListOf(),
    val output: MutableList<Byte> = mutableListOf(),
    /** Flat backing storage that `seqId` array entries point into. */
    val seqIdData: MutableList<Int> = mutableListOf(),
)

// =============================================================================
// LlamaBatchAllocr – sanitisation, fulfillment, and splitting
// Ported from: llama-batch.h / llama-batch.cpp  class llama_batch_allocr
// =============================================================================

/**
 * Helper that **sanitises**, **auto-fills**, and **splits** a user-supplied
 * [LlamaBatch] into a sequence of [LlamaUBatchInternal] micro-batches.
 *
 * This is a faithful port of `class llama_batch_allocr` from `llama-batch.h`.
 *
 * Typical usage:
 * ```kotlin
 * val allocr = LlamaBatchAllocr(nPosPerEmbd = 1)
 * if (!allocr.init(batch, vocabSize, memory = null, nEmbd, nSeqMax, outputAll = false)) {
 *     error("batch validation failed")
 * }
 * allocr.splitReset()
 * while (true) {
 *     val ub = allocr.splitSimple(nUbatch)
 *     if (ub.nTokens == 0) break
 *     // … process ub …
 * }
 * ```
 *
 * @param nPosPerEmbd Number of position values stored per token/embedding.
 *                    Usually 1; M-RoPE models use 3+.
 */
class LlamaBatchAllocr(
    private val nPosPerEmbd: Int = 1,
) {
    // -- validated / augmented copy of the input batch --
    private var batch: LlamaBatch = LlamaBatch(tokens = intArrayOf())

    /** Only for debugging. */
    private var vocabSize: Int = 0

    private var nEmbd: Int = 0
    private var nSeqMax: Int = 0
    private var nOutputs: Int = 0

    // Auto-generated helper arrays (allocated once, grown as needed)
    private var pos = mutableListOf<Int>()
    private var nSeqId = mutableListOf<Int>()
    private var seqId = mutableListOf<IntArray>()
    private var seqIdUnq = mutableListOf<Int>()
    private var seqIdx = IntArray(LLAMA_MAX_SEQ) { -1 }
    private var output = mutableListOf<Byte>()

    /** Whether any coupled sequences exist in the batch. */
    private var hasCpl = false

    /** `seqPos[s]` – the set of positions in sequence `s`. */
    private val seqPos: Array<MutableSet<Int>> = Array(LLAMA_MAX_SEQ) { mutableSetOf() }

    /** `seqCpl[s0][s1]` – whether sequence `s0` is coupled to `s1`. */
    private val seqCpl: Array<BooleanArray> = Array(LLAMA_MAX_SEQ) { BooleanArray(LLAMA_MAX_SEQ) }

    /** `seqSet[i]` – bitset of sequence membership for token `i`. */
    private var seqSet = mutableListOf<Long>()  // bitfield over LLAMA_MAX_SEQ bits

    /**
     * Map from sequence-set bitfield → list of token indices sharing that set.
     * Port of `std::unordered_map<seq_set_t, idx_vec_t>`.
     */
    private val seqSetMap = mutableMapOf<Long, MutableList<Int>>()

    /** Batch indices of output tokens, in the order they were encountered. */
    private val outIds = mutableListOf<Int>()

    private var nUsed: Int = 0
    private var used = mutableListOf<Boolean>()

    private var debug: Int = 0

    init {
        // C++ reads LLAMA_BATCH_DEBUG from the environment.
        val envDebug = try { getEnvVar("LLAMA_BATCH_DEBUG")?.toIntOrNull() } catch (_: Exception) { null }
        debug = envDebug ?: 0
    }

    // -- public API --------------------------------------------------------

    fun getBatch(): LlamaBatch = batch

    fun getNTokens(): Int = batch.nTokens
    fun getNOutputs(): Int = nOutputs
    fun getNUsed(): Int = nUsed

    fun getOutIds(): MutableList<Int> = outIds

    /** Minimum position for [seqId] in the current batch, or `Int.MAX_VALUE`. */
    fun seqPosMin(seqId: LlamaSeqId): Int =
        seqPos.getOrNull(seqId)?.minOrNull() ?: Int.MAX_VALUE

    /** Maximum position for [seqId] in the current batch, or `Int.MIN_VALUE`. */
    fun seqPosMax(seqId: LlamaSeqId): Int =
        seqPos.getOrNull(seqId)?.maxOrNull() ?: Int.MIN_VALUE

    // -- init --------------------------------------------------------------

    /**
     * Validate and auto-fill the input batch.
     *
     * Port of `llama_batch_allocr::init()`.
     *
     * @return `true` on success; `false` when the batch is invalid.
     */
    fun init(
        batchInp: LlamaBatch,
        vocabSize: Int,
        nEmbd: Int,
        nSeqMax: Int,
        outputAll: Boolean,
    ): Boolean {
        clear()
        batch = batchInp
        this.vocabSize = vocabSize
        this.nEmbd = nEmbd
        this.nSeqMax = nSeqMax

        require(batch.nTokens > 0) { "batch.nTokens must be > 0" }

        if (nSeqMax > LLAMA_MAX_SEQ) {
            llamaLogError("init: nSeqMax = $nSeqMax > $LLAMA_MAX_SEQ\n")
            return false
        }

        // Validate tokens
        batch.tokens?.let { toks ->
            for (i in toks.indices) {
                if (toks[i] < 0 || toks[i] >= vocabSize) {
                    llamaLogError("init: invalid token[$i] = ${toks[i]}\n")
                    return false
                }
            }
        }

        // TODO: full init logic — auto-generate pos, seq_id, output arrays
        //       when the input batch leaves them null (mirrors C++ init body)
        nOutputs = if (outputAll) batch.nTokens else 0
        return true
    }

    // -- split API ---------------------------------------------------------

    /** Reset internal splitting state before iterating ubatches. */
    fun splitReset() {
        nUsed = 0
        used = MutableList(batch.nTokens) { false }
        outIds.clear()
    }

    /**
     * Produce the next simple (unequal-length) ubatch of up to [nUbatch] tokens.
     *
     * Returns a ubatch with `nTokens == 0` when the entire batch is consumed.
     *
     * Port of `llama_batch_allocr::split_simple()`.
     */
    fun splitSimple(nUbatch: Int): LlamaUBatchInternal {
        TODO("Port llama_batch_allocr::split_simple — iterate unprocessed tokens, respecting nUbatch budget")
    }

    /**
     * Produce the next equal-length ubatch.
     *
     * Port of `llama_batch_allocr::split_equal()`.
     *
     * @param sequential When `true`, tokens carry incrementing sequence ids.
     */
    fun splitEqual(nUbatch: Int, sequential: Boolean): LlamaUBatchInternal {
        TODO("Port llama_batch_allocr::split_equal")
    }

    /**
     * Produce the next single-sequence-set ubatch.
     *
     * Port of `llama_batch_allocr::split_seq()`.
     */
    fun splitSeq(nUbatch: Int): LlamaUBatchInternal {
        TODO("Port llama_batch_allocr::split_seq")
    }

    /**
     * Reserve space for a ubatch of given dimensions (without filling data).
     *
     * Port of `llama_batch_allocr::ubatch_reserve()`.
     */
    fun ubatchReserve(nSeqTokens: Int, nSeqs: Int): LlamaUBatchInternal {
        TODO("Port llama_batch_allocr::ubatch_reserve")
    }

    // -- private -----------------------------------------------------------

    private fun clear() {
        pos.clear()
        nSeqId.clear()
        seqId.clear()
        seqIdUnq.clear()
        seqIdx.fill(-1)
        output.clear()
        seqPos.forEach { it.clear() }
        seqCpl.forEach { it.fill(false) }
        seqSet.clear()
        seqSetMap.clear()
        outIds.clear()
        hasCpl = false
        nOutputs = 0
    }

    /**
     * Build and return a ubatch from the given token indices.
     *
     * Port of `llama_batch_allocr::ubatch_add()`.
     */
    private fun ubatchAdd(idxs: List<Int>, nSeqs: Int, equalSeqs: Boolean): LlamaUBatchInternal {
        TODO("Port llama_batch_allocr::ubatch_add")
    }
}

// =============================================================================
// Utility: platform env-var reader (nativeMain only)
// =============================================================================

/**
 * Read an environment variable on the native platform.
 *
 * Returns `null` when the variable is not set.
 */
@OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
internal fun getEnvVar(name: String): String? {
    val ptr = platform.posix.getenv(name) ?: return null
    return ptr.toKString()
}
