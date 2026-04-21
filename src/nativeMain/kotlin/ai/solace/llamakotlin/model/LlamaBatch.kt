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
    private var output = mutableListOf<Boolean>()

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
        memory: LlamaMemory? = null,
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

        // Validate seq_id ranges
        batch.seqId?.let { batchSeqId ->
            val batchNSeqId = batch.nSeqId
            if (batchNSeqId != null) {
                for (i in 0 until batch.nTokens) {
                    for (s in 0 until batchNSeqId[i]) {
                        val sid = batchSeqId[i][s]
                        if (sid < 0 || sid >= nSeqMax) {
                            llamaLogError("init: invalid seqId[$i][$s] = $sid >= $nSeqMax\n")
                            return false
                        }
                    }
                }
            }
        }

        // Auto-generate nSeqId if missing
        if (batch.nSeqId == null) {
            nSeqId.clear()
            for (i in 0 until batch.nTokens) {
                nSeqId.add(1) // default: 1 sequence per token
            }
            batch = batch.copy(nSeqId = this.nSeqId.toIntArray())
        }

        // Auto-generate seqId if missing
        if (batch.seqId == null) {
            seqId.clear()
            for (i in 0 until batch.nTokens) {
                seqId.add(intArrayOf(0)) // default seq id 0
            }
            batch = batch.copy(seqId = seqId.toTypedArray())
        }

        // Auto-generate pos if missing
        if (batch.pos == null) {
            pos.clear()
            val p0 = IntArray(nSeqMax) { s ->
                if (memory == null) 0 else (memory.seqPosMax(s) + 1)
            }
            val batchSeqId = batch.seqId!!
            for (i in 0 until batch.nTokens) {
                val sid = batchSeqId[i][0]
                pos.add(p0[sid])
                val batchNSeqId = batch.nSeqId!!
                for (s in 0 until batchNSeqId[i]) {
                    p0[batchSeqId[i][s]] = pos[i] + 1
                }
            }
            batch = batch.copy(pos = pos.toIntArray())
        }

        // Auto-generate output if missing
        if (batch.logits == null) {
            if (outputAll) {
                output.clear()
                output.addAll(List(batch.nTokens) { true })
            } else {
                output.clear()
                output.addAll(List(batch.nTokens) { false })
                if (output.isNotEmpty()) output[output.size - 1] = true
            }
            batch = batch.copy(logits = output.toBooleanArray())
        } else if (outputAll) {
            val logits = batch.logits!!
            if (logits.any { !it }) {
                llamaLogWarn("init: embeddings required but some input tokens not marked as outputs -> overriding\n")
                output.clear()
                output.addAll(List(batch.nTokens) { true })
                batch = batch.copy(logits = output.toBooleanArray())
            }
        }

        // Compute stats
        val batchLogits = batch.logits!!
        for (i in 0 until batch.nTokens) {
            if (batchLogits[i]) nOutputs++
        }

        hasCpl = false

        val batchSeqId = batch.seqId!!
        val batchNSeqId = batch.nSeqId!!
        val batchPos = batch.pos!!

        // Determine coupled sequences
        for (i in 0 until batch.nTokens) {
            val s0 = batchSeqId[i][0]
            for (s in 0 until batchNSeqId[i]) {
                val s1 = batchSeqId[i][s]
                seqPos[s1].add(batchPos[i])
                if (s > 0) {
                    seqCpl[s1][s0] = true
                    hasCpl = true
                }
            }
        }

        // Precompute sequence sets and unique sequence ids
        var seqSetUnq = 0L
        for (i in 0 until batch.nTokens) {
            var cur = 0L
            for (s in 0 until batchNSeqId[i]) {
                val sid = batchSeqId[i][s]
                cur = cur or (1L shl sid)
                seqSetUnq = seqSetUnq or (1L shl sid)
            }
            seqSet.add(cur)
            seqSetMap.getOrPut(cur) { mutableListOf() }.add(i)
        }

        for (s in 0 until nSeqMax) {
            if ((seqSetUnq and (1L shl s)) != 0L) {
                seqIdx[s] = seqIdUnq.size
                seqIdUnq.add(s)
            }
        }

        splitReset()
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
        // Find the first unused token
        var curIdx = 0
        while (curIdx < used.size && used[curIdx]) curIdx++

        // We are done
        if (curIdx >= used.size) return LlamaUBatchInternal()

        val idxs = mutableListOf<Int>()
        while (true) {
            idxs.add(curIdx)
            used[curIdx] = true
            nUsed++
            curIdx++
            if (curIdx >= used.size) break
            if (idxs.size >= nUbatch) break
        }
        return ubatchAdd(idxs, idxs.size, false)
    }

    /**
     * Produce the next equal-length ubatch.
     *
     * Port of `llama_batch_allocr::split_equal()`.
     *
     * @param sequential When `true`, tokens carry incrementing sequence ids.
     */
    fun splitEqual(nUbatch: Int, sequential: Boolean): LlamaUBatchInternal {
        if (sequential && hasCpl) {
            llamaLogError("splitEqual: sequential split not supported with coupled sequences\n")
            return LlamaUBatchInternal()
        }

        val curSeqSet = mutableListOf<Long>()
        var lastSeqId = -1

        // Determine non-overlapping sequence sets
        val batchSeqId = batch.seqId!!
        for (i in 0 until batch.nTokens) {
            if (used[i]) continue

            var add = true
            for (s in curSeqSet.indices) {
                if ((curSeqSet[s] and seqSet[i]) != 0L) {
                    add = false
                    break
                }
            }

            if (sequential) {
                add = add && (curSeqSet.isEmpty() || batchSeqId[i][0] == lastSeqId + 1)
            }

            if (add) {
                curSeqSet.add(seqSet[i])
                lastSeqId = batchSeqId[i][0]
                if (curSeqSet.size > nUbatch) break
            }
        }

        val nSeqs = curSeqSet.size
        if (nSeqs == 0) return LlamaUBatchInternal()

        // Current batch index for each sequence set
        val curIdx = IntArray(nSeqs) { 0 }
        for (s in 0 until nSeqs) {
            val mapping = seqSetMap[curSeqSet[s]]!!
            while (used[mapping[curIdx[s]]]) curIdx[s]++
        }

        val idxsPerSeq = Array(nSeqs) { mutableListOf<Int>() }

        while (true) {
            var canExpand = true
            for (s in 0 until nSeqs) {
                if (curIdx[s] >= seqSetMap[curSeqSet[s]]!!.size) {
                    canExpand = false
                    break
                }
            }
            if (!canExpand) break

            for (s in 0 until nSeqs) {
                val idx = seqSetMap[curSeqSet[s]]!![curIdx[s]]
                idxsPerSeq[s].add(idx)
                used[idx] = true
                nUsed++
                curIdx[s]++
            }

            if ((idxsPerSeq[0].size + 1) * nSeqs > nUbatch) break
        }

        // Concat per-sequence-set lists
        val idxs = mutableListOf<Int>()
        for (s in 0 until nSeqs) {
            idxs.addAll(idxsPerSeq[s])
        }

        return ubatchAdd(idxs, nSeqs, true)
    }

    /**
     * Produce the next single-sequence-set ubatch.
     *
     * Port of `llama_batch_allocr::split_seq()`.
     */
    fun splitSeq(nUbatch: Int): LlamaUBatchInternal {
        // Find the first unused token
        var curIdx = 0
        while (curIdx < used.size && used[curIdx]) curIdx++

        if (curIdx >= used.size) return LlamaUBatchInternal()

        // Starting sequence set — only allow subsets
        var curSeqSetBits = seqSet[curIdx]
        val idxs = mutableListOf<Int>()

        while (true) {
            idxs.add(curIdx)
            used[curIdx] = true
            nUsed++

            if (idxs.size >= nUbatch) break

            do {
                curIdx++
            } while (curIdx < getNTokens() && (used[curIdx] || (curSeqSetBits and seqSet[curIdx]) != seqSet[curIdx]))

            if (curIdx == getNTokens()) break

            curSeqSetBits = seqSet[curIdx]
        }

        return ubatchAdd(idxs, 1, true)
    }

    /**
     * Reserve space for a ubatch of given dimensions (without filling data).
     *
     * Port of `llama_batch_allocr::ubatch_reserve()`.
     */
    fun ubatchReserve(nSeqTokens: Int, nSeqs: Int): LlamaUBatchInternal {
        val nTokens = nSeqTokens * nSeqs
        clear()
        splitReset()

        val nPosAll = nTokens * nPosPerEmbd

        val udata = UBatchData(
            token = MutableList(nTokens) { 0 },
            pos = MutableList(nPosAll) { 0 },
            nSeqId = MutableList(nTokens) { 0 },
            seqIdUnq = mutableListOf(),
            seqIdx = MutableList(LLAMA_MAX_SEQ) { -1 },
            output = MutableList(nTokens) { 0.toByte() },
        )

        for (s in 0 until nSeqs) {
            udata.seqIdx[s] = s
            udata.seqIdUnq.add(s)
        }

        // Build seqId arrays (each token maps to its sequence)
        val seqIdArrays = Array(nTokens) { IntArray(0) }

        return LlamaUBatchInternal(
            equalSeqs = true,
            nTokens = nTokens,
            nSeqTokens = nSeqTokens,
            nSeqs = nSeqs,
            nSeqsUnq = nSeqs,
            nPos = nPosPerEmbd,
            token = udata.token.toIntArray(),
            pos = udata.pos.toIntArray(),
            nSeqId = udata.nSeqId.toIntArray(),
            seqId = seqIdArrays,
            seqIdUnq = udata.seqIdUnq.toIntArray(),
            seqIdx = udata.seqIdx.toIntArray(),
            output = udata.output.toByteArray(),
            data = udata,
        )
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
        val nTokens = idxs.size
        require(nTokens % nSeqs == 0)

        val batchTokens = batch.tokens
        val batchEmbd = batch.embeddings
        val batchPos = batch.pos!!
        val batchNSeqId = batch.nSeqId!!
        val batchSeqId = batch.seqId!!
        val batchLogits = batch.logits!!

        val hasEmbd = batchEmbd != null
        val nEmbdAll = if (hasEmbd) nTokens * nEmbd else 0
        val nPosAll = nTokens * nPosPerEmbd

        val udata = UBatchData(
            token = MutableList(nTokens) { 0 },
            embd = MutableList(nEmbdAll) { 0f },
            pos = MutableList(nPosAll) { 0 },
            nSeqId = MutableList(nTokens) { 0 },
            seqIdUnq = mutableListOf(),
            seqIdx = MutableList(LLAMA_MAX_SEQ) { -1 },
            output = MutableList(nTokens) { 0.toByte() },
        )

        var seqSetUnq = 0L

        for ((i, batchIdx) in idxs.withIndex()) {
            if (batchTokens != null) {
                udata.token[i] = batchTokens[batchIdx]
            }
            if (hasEmbd) {
                for (e in 0 until nEmbd) {
                    udata.embd[i * nEmbd + e] = batchEmbd!![batchIdx * nEmbd + e]
                }
            }

            for (j in 0 until nPosPerEmbd) {
                val srcOff = if (batchTokens != null) 0 else j * batch.nTokens
                udata.pos[j * nTokens + i] = batchPos[srcOff + batchIdx]
            }

            udata.nSeqId[i] = batchNSeqId[batchIdx]
            udata.output[i] = if (batchLogits[batchIdx]) 1.toByte() else 0.toByte()

            for (s in 0 until udata.nSeqId[i]) {
                val sid = batchSeqId[batchIdx][s]
                udata.seqIdData.add(sid)
                seqSetUnq = seqSetUnq or (1L shl sid)
            }

            if (batchLogits[batchIdx]) {
                outIds.add(batchIdx)
            }
        }

        // Build seqId arrays from flat seqIdData
        val seqIdArrays = Array(nTokens) { IntArray(0) }
        var dataIdx = 0
        for (i in 0 until nTokens) {
            val cnt = udata.nSeqId[i]
            seqIdArrays[i] = IntArray(cnt) { udata.seqIdData[dataIdx + it] }
            dataIdx += cnt
        }

        // Build seqIdUnq and seqIdx
        for (s in 0 until nSeqMax) {
            if ((seqSetUnq and (1L shl s)) != 0L) {
                udata.seqIdx[s] = udata.seqIdUnq.size
                udata.seqIdUnq.add(s)
            }
        }

        return LlamaUBatchInternal(
            equalSeqs = equalSeqs,
            nTokens = nTokens,
            nSeqTokens = nTokens / nSeqs,
            nSeqs = nSeqs,
            nSeqsUnq = udata.seqIdUnq.size,
            nPos = nPosPerEmbd,
            token = if (batchTokens != null) udata.token.toIntArray() else null,
            embd = if (hasEmbd) udata.embd.toFloatArray() else null,
            pos = udata.pos.toIntArray(),
            nSeqId = udata.nSeqId.toIntArray(),
            seqId = seqIdArrays,
            seqIdUnq = udata.seqIdUnq.toIntArray(),
            seqIdx = udata.seqIdx.toIntArray(),
            output = udata.output.toByteArray(),
            data = udata,
        )
    }

    // =========================================================================
    // ubatch debug printing
    // Ported from: llama-batch.cpp  llama_batch_allocr::ubatch_print
    // =========================================================================

    /**
     * Debug-print the contents of a [LlamaUBatchInternal].
     *
     * Port of `llama_batch_allocr::ubatch_print()`.
     *
     * @param ubatch The micro-batch to print.
     * @param debug  Verbosity level (1 = summary, 2 = per-token detail).
     */
    fun ubatchPrint(ubatch: LlamaUBatchInternal, debug: Int) {
        if (debug <= 0) return

        llamaLogDebug("ubatchPrint:   equal_seqs   = ${ubatch.equalSeqs}\n")
        llamaLogDebug("ubatchPrint:   n_tokens     = ${ubatch.nTokens}\n")
        llamaLogDebug("ubatchPrint:   n_seq_tokens = ${ubatch.nSeqTokens}\n")
        llamaLogDebug("ubatchPrint:   n_seqs       = ${ubatch.nSeqs}\n")
        llamaLogDebug("ubatchPrint:   n_seqs_unq   = ${ubatch.nSeqsUnq}\n")

        val seqIdUnqStr = ubatch.seqIdUnq?.joinToString(" ", "[ ", " ]") ?: "[]"
        val seqIdxStr = buildString {
            append("[")
            val seqIdx = ubatch.seqIdx
            for (s in 0 until LLAMA_MAX_SEQ) {
                val idx = seqIdx?.getOrElse(s) { -1 } ?: -1
                if (idx >= 0) append(idx % 10) else append(".")
            }
            append("]")
        }

        llamaLogDebug("ubatchPrint:   seq_id_unq = $seqIdUnqStr\n")
        llamaLogDebug("ubatchPrint:   seq_idx    = $seqIdxStr\n")
        llamaLogDebug("ubatchPrint:   n_outputs  = $nOutputs\n")

        if (debug > 1) {
            val nSeqIdArr = ubatch.nSeqId
            val seqIdArr = ubatch.seqId
            val tokenArr = ubatch.token
            val posArr = ubatch.pos
            val outputArr = ubatch.output

            var seqIdMax = 0
            for (i in 0 until ubatch.nTokens) {
                val nsi = nSeqIdArr?.getOrElse(i) { 0 } ?: 0
                val sids = seqIdArr?.getOrElse(i) { intArrayOf() }
                for (s in 0 until nsi) {
                    seqIdMax = maxOf(seqIdMax, sids?.getOrElse(s) { 0 } ?: 0)
                }
            }
            seqIdMax++

            llamaLogDebug("ubatchPrint:   token     = [\n")
            for (i in 0 until ubatch.nTokens) {
                val seqBits = IntArray(seqIdMax)
                val nsi = nSeqIdArr?.getOrElse(i) { 0 } ?: 0
                val sids = seqIdArr?.getOrElse(i) { intArrayOf() }
                for (s in 0 until nsi) {
                    val sid = sids?.getOrElse(s) { 0 } ?: 0
                    if (sid in 0 until seqIdMax) seqBits[sid] = 1
                }
                val seqStr = seqBits.joinToString("") { if (it != 0) (it % 10).toString() else "." }

                val token = tokenArr?.getOrElse(i) { -1 } ?: -1
                val pos = posArr?.getOrElse(i) { -1 } ?: -1
                val output = outputArr?.getOrElse(i) { 0 } ?: 0

                if (tokenArr != null && tokenArr.isNotEmpty()) {
                    llamaLogDebug("ubatchPrint:  ${i.toString().padStart(4)}: id = ${token.toString().padStart(6)}, pos = ${pos.toString().padStart(4)}, n_seq_id = ${nsi.toString().padStart(2)}, seq_id = [$seqStr], output = $output\n")
                } else {
                    llamaLogDebug("ubatchPrint:  ${i.toString().padStart(4)}: [embd], pos = ${pos.toString().padStart(4)}, n_seq_id = ${nsi.toString().padStart(2)}, seq_id = [$seqStr], output = $output\n")
                }
            }
            llamaLogDebug("ubatchPrint:   ]\n")
        }
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
