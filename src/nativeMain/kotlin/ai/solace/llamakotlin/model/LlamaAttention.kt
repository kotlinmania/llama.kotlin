// port-lint: source llama.cpp/src/llama-context.cpp
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import kotlin.math.*
import kotlin.time.TimeSource

// ---------------------------------------------------------------------------
// Enums ported from llama.h / llama-cparams.h
// ---------------------------------------------------------------------------

/** Pooling strategy for output embeddings. Maps to `llama_pooling_type`. */
enum class LlamaPoolingType {
    UNSPECIFIED,
    NONE,
    MEAN,
    CLS,
    LAST,
    RANK;
}

/** Attention mode selector. Maps to `llama_attention_type`. */
enum class LlamaAttentionType {
    UNSPECIFIED,
    CAUSAL,
    NON_CAUSAL;
}

/** RoPE scaling strategy. Uses [LlamaRopeScalingType] from LlamaModel.kt. */

/** Flash-attention mode selector. Maps to `llama_flash_attn_type`. */
enum class LlamaFlashAttnType {
    DISABLED,
    ENABLED,
    AUTO;
}

/** Type of graph to build for a micro-batch. Maps to `llm_graph_type`. */
// LlmGraphType defined in LlmGraphTypes.kt

typealias LlamaToken = Int
// LlamaSeqId defined in KVCache.kt; LLAMA_MAX_SEQ defined in KVCache.kt

// ---------------------------------------------------------------------------
// Context parameters  (maps to llama_context_params)
// ---------------------------------------------------------------------------

/**
 * Parameters supplied when creating a [LlamaContext].
 *
 * Every field mirrors a member of `llama_context_params` from the C++ codebase.
 * The Kotlin default values match the C++ defaults (0/false means "use model default").
 */
data class LlamaContextParams(
    val nCtx: Int = 0,
    val nBatch: Int = 2048,
    val nUbatch: Int = 512,
    val nSeqMax: Int = 1,
    val nThreads: Int = 4,
    val nThreadsBatch: Int = 4,
    val ropeScalingType: LlamaRopeScalingType = LlamaRopeScalingType.UNSPECIFIED,
    val ropeFreqBase: Float = 0.0f,
    val ropeFreqScale: Float = 0.0f,
    val yarnExtFactor: Float = -1.0f,
    val yarnAttnFactor: Float = 1.0f,
    val yarnBetaFast: Float = 32.0f,
    val yarnBetaSlow: Float = 1.0f,
    val yarnOrigCtx: Int = 0,
    val embeddings: Boolean = false,
    val offloadKqv: Boolean = true,
    val flashAttnType: LlamaFlashAttnType = LlamaFlashAttnType.DISABLED,
    val noPerf: Boolean = false,
    val attentionType: LlamaAttentionType = LlamaAttentionType.UNSPECIFIED,
    val poolingType: LlamaPoolingType = LlamaPoolingType.UNSPECIFIED,
    val typeK: GGMLType = GGMLType.F16,
    val typeV: GGMLType = GGMLType.F16,
    val swaFull: Boolean = false,
    val opOffload: Boolean = true,
    val kvUnified: Boolean = true,
)

// ---------------------------------------------------------------------------
// Computed context parameters  (maps to llama_cparams)
// ---------------------------------------------------------------------------

/**
 * Derived / computed context parameters that are resolved from [LlamaContextParams]
 * and the model's hyper-parameters during [LlamaContext] construction.
 *
 * Maps to `struct llama_cparams` in `llama-cparams.h`.
 */
data class LlamaCParams(
    var nCtx: Int = 0,
    var nCtxSeq: Int = 0,
    var nBatch: Int = 0,
    var nUbatch: Int = 0,
    var nSeqMax: Int = 1,
    var nThreads: Int = 4,
    var nThreadsBatch: Int = 4,
    var ropeFreqBase: Float = 10000.0f,
    var ropeFreqScale: Float = 1.0f,
    var nCtxOrigYarn: Int = 0,
    var yarnExtFactor: Float = 0.0f,
    var yarnAttnFactor: Float = 1.0f,
    var yarnBetaFast: Float = 32.0f,
    var yarnBetaSlow: Float = 1.0f,
    var embeddings: Boolean = false,
    var causalAttn: Boolean = true,
    var offloadKqv: Boolean = true,
    var flashAttn: Boolean = false,
    var autoFa: Boolean = false,
    var fusedGdnAr: Boolean = true,
    var fusedGdnCh: Boolean = true,
    var autoFgdn: Boolean = true,
    var noPerf: Boolean = false,
    var warmup: Boolean = false,
    var opOffload: Boolean = true,
    var kvUnified: Boolean = true,
    var pipelineParallel: Boolean = false,
    var poolingType: LlamaPoolingType = LlamaPoolingType.NONE,
)

// ---------------------------------------------------------------------------
// Performance counters  (maps to llama_perf_context_data)
// ---------------------------------------------------------------------------

/** Performance counters returned by [LlamaContext.perfGetData]. */
data class LlamaPerfContextData(
    val tStartMs: Double = 0.0,
    val tLoadMs: Double = 0.0,
    val tPEvalMs: Double = 0.0,
    val tEvalMs: Double = 0.0,
    val nPEval: Int = 1,
    val nEval: Int = 1,
    val nReused: Int = 0,
)

// ---------------------------------------------------------------------------
// Memory breakdown  (maps to llama_memory_breakdown_data)
// ---------------------------------------------------------------------------

/** Per-buffer-type memory usage breakdown. */
data class LlamaMemoryBreakdownData(
    var model: Long = 0,
    var context: Long = 0,
    var compute: Long = 0,
) {
    fun total(): Long = model + context + compute
}

// ---------------------------------------------------------------------------
// Batch  (maps to llama_batch)
// ---------------------------------------------------------------------------

/**
 * A batch of tokens (or embeddings) to process.
 *
 * Exactly one of [tokens] or [embeddings] must be non-null.
 * Mirrors `struct llama_batch` from `llama.h`.
 */
data class LlamaBatch(
    val nTokens: Int = 0,
    val tokens: IntArray? = null,
    val embeddings: FloatArray? = null,
    val nEmbeddings: Int = 0,
    val pos: IntArray? = null,
    val nSeqId: IntArray? = null,
    val seqId: Array<IntArray>? = null,
    val logits: BooleanArray? = null,
) {
    init {
        require((tokens != null) xor (embeddings != null)) {
            "Exactly one of tokens or embeddings must be provided"
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is LlamaBatch) return false
        return nTokens == other.nTokens &&
            tokens.contentEquals(other.tokens) &&
            embeddings.contentEquals(other.embeddings)
    }

    override fun hashCode(): Int {
        var result = nTokens
        result = 31 * result + (tokens?.contentHashCode() ?: 0)
        result = 31 * result + (embeddings?.contentHashCode() ?: 0)
        return result
    }
}

// ---------------------------------------------------------------------------
// Micro-batch  (maps to llama_ubatch — internal scheduling unit)
// ---------------------------------------------------------------------------

/**
 * A contiguous sub-range of a [LlamaBatch] that fits within the micro-batch
 * budget (`nUbatch`). This is the unit actually submitted to graph execution.
 */
data class LlamaUBatch(
    val nTokens: Int = 0,
    val nSeqTokens: Int = 0,
    val nSeqs: Int = 0,
    /** Number of unique sequence IDs in this ubatch. */
    val nSeqsUnq: Int = 0,
    /** Number of position dimensions per token/embedding (1 for normal, 4 for M-RoPE). */
    val nPos: Int = 1,
    val tokens: IntArray? = null,
    val embeddings: FloatArray? = null,
    /** Positions array of size [nTokens * nPos]. */
    val pos: IntArray? = null,
    val nSeqId: IntArray? = null,
    val seqId: Array<IntArray>? = null,
    /** Unique sequence IDs present in this ubatch. Size = [nSeqsUnq]. */
    val seqIdUnq: IntArray? = null,
    /**
     * Mapping from sequence ID → index in [0, nSeqsUnq).
     * Size = LLAMA_MAX_SEQ (or large enough for all seq IDs).
     * Used for pooled embedding extraction.
     */
    val seqIdx: IntArray? = null,
    val output: BooleanArray? = null,
    val equalSeqs: Boolean = false,
) {
    /** True when positions are multi-dimensional (M-RoPE with ≥3 dimensions). */
    fun isPos2d(): Boolean = nPos >= 3

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is LlamaUBatch) return false
        return nTokens == other.nTokens && equalSeqs == other.equalSeqs
    }

    override fun hashCode(): Int = nTokens * 31 + equalSeqs.hashCode()
}

// ---------------------------------------------------------------------------
// Swap info for output reordering
// ---------------------------------------------------------------------------

/** Pair of indices swapped during output reordering. */
data class SwapInfo(val i0: Int, val i1: Int)

// ---------------------------------------------------------------------------
// LlamaContext  (maps to struct llama_context)
// ---------------------------------------------------------------------------

/**
 * The primary inference context — owns the KV cache, backend scheduler,
 * compute buffers, and output (logits / embeddings) storage.
 *
 * This class is the Kotlin port of `struct llama_context` from
 * `llama-context.h` / `llama-context.cpp`. It is the entry-point for
 * [encode] and [decode] operations.
 *
 * Construction mirrors `llama_context::llama_context(model, params)`:
 * it resolves the computed context parameters ([cparams]) from the raw
 * [LlamaContextParams], initialises backends, creates the output buffer,
 * and reserves worst-case graphs.
 */
class LlamaContext(
    /** Reference to the loaded model (weights + hyper-parameters). */
    val model: LlamaModel,
    params: LlamaContextParams = LlamaContextParams(),
) {
    // -- resolved / computed context parameters --
    val cparams: LlamaCParams = resolveCParams(model.config, params)

    // -- memory (KV cache or equivalent abstract memory) --
    /** Per-layer KV caches; initialised lazily via [initMemory]. */
    var kvCaches: Array<KVCache>? = null
        private set

    /** Abstract memory interface (may be KV cache, recurrent, or hybrid). */
    var memory: LlamaMemory? = null
        internal set

    // -- threadpools (opaque handles, mirrors C++ ggml_threadpool_t) --
    var threadpool: Any? = null
        internal set
    var threadpoolBatch: Any? = null
        internal set

    // -- abort callback --
    var abortCallback: (() -> Boolean)? = null
        internal set

    // -- LoRA adapters: adapter → scale --
    val loras: MutableMap<Any, Float> = mutableMapOf()

    // -- batch allocator --
    var balloc: LlamaBatchAllocr? = null
        internal set

    // -- optimization context (for fine-tuning) --
    var optCtx: Any? = null
        internal set

    // -- graph results (previous and reserved) --
    var gfResPrev: Any? = null
        internal set
    var gfResReserve: Any? = null
        internal set

    // -- decode output (2-d array: [nOutputs][nVocab]) --
    var logits: FloatArray? = null
        internal set
    var logitsSize: Int = 0
        internal set

    // -- embeddings output (2-d array: [nOutputs][nEmbd]) --
    var embd: FloatArray? = null
        internal set
    var embdSize: Int = 0
        internal set

    // -- sequence-level embeddings (populated when pooling != NONE) --
    val embdSeq: MutableMap<LlamaSeqId, FloatArray> = mutableMapOf()

    // -- output bookkeeping --
    var nOutputs: Int = 0
        internal set

    /** Maps batch token positions → ids inside the logits / embd buffers. */
    var outputIds: IntArray = IntArray(0)
        internal set

    val outputSwaps: MutableList<SwapInfo> = mutableListOf()

    // -- scheduler state --
    var schedNeedReserve: Boolean = true
        internal set

    // -- backend sampling state --
    val sampling: SamplingInfo = SamplingInfo()

    // -- cross-attention state (for encoder-decoder models) --
    val cross: LlamaCross = LlamaCross()

    // -- graph reuse control --
    var graphReuseDisable: Boolean = false
        private set

    // -- perf counters (microseconds) --
    var tStartUs: Long = 0L
        internal set
    var tLoadUs: Long = 0L
        internal set
    var tPEvalUs: Long = 0L
        internal set
    var tEvalUs: Long = 0L
        internal set
    var tComputeStartUs: Long = 0L
        internal set
    var nQueuedTokens: Long = 0L
        internal set
    var nPEval: Int = 0
        internal set
    var nEval: Int = 0
        internal set
    var nReused: Int = 0
        internal set

    var hasEvaluatedOnce: Boolean = false
        internal set

    // -- graph --
    /** Graph allocator for compute operations. */
    val graphAllocator: GGMLGraphAllocator = GGMLGraphAllocator()

    /** Shared GGML context for tensor metadata. */
    val ggmlContext: GGMLContext = GGMLContext()

    init {
        tStartUs = currentTimeMicros()
        initMemory()
        initSamplingVocab()
    }

    // -----------------------------------------------------------------------
    // Accessors  (mirrors C++ getter methods)
    // -----------------------------------------------------------------------

    fun nCtx(): Int = cparams.nCtx
    fun nCtxSeq(): Int = cparams.nCtxSeq
    fun nBatch(): Int = cparams.nBatch
    fun nUbatch(): Int = cparams.nUbatch
    fun nSeqMax(): Int = cparams.nSeqMax
    fun nThreads(): Int = cparams.nThreads
    fun nThreadsBatch(): Int = cparams.nThreadsBatch
    fun poolingType(): LlamaPoolingType = cparams.poolingType

    // -----------------------------------------------------------------------
    // Logits / embeddings access
    // -----------------------------------------------------------------------

    /**
     * Return the full logits buffer. Triggers output reordering first.
     * Mirrors `llama_context::get_logits()`.
     */
    fun getLogits(): FloatArray? {
        outputReorder()
        return logits
    }

    /**
     * Return a view of the logits for the i-th output token.
     * Supports negative indices (last output row).
     * Mirrors `llama_context::get_logits_ith(i)`.
     */
    fun getLogitsIth(i: Int): FloatArray? {
        outputReorder()
        val buf = logits ?: return null
        val j = outputResolveRow(i)
        val nVocab = model.config.vocabSize
        val start = (j * nVocab).toInt()
        return buf.copyOfRange(start, start + nVocab)
    }

    /**
     * Return the full embeddings buffer.
     * Mirrors `llama_context::get_embeddings()`.
     */
    fun getEmbeddings(): FloatArray? {
        outputReorder()
        return embd
    }

    /**
     * Return a view of the embeddings for the i-th output token.
     * Mirrors `llama_context::get_embeddings_ith(i)`.
     */
    fun getEmbeddingsIth(i: Int): FloatArray? {
        outputReorder()
        val buf = embd ?: return null
        val j = outputResolveRow(i)
        val nEmbd = model.config.hiddenSize
        val start = (j * nEmbd).toInt()
        return buf.copyOfRange(start, start + nEmbd)
    }

    /**
     * Return sequence-level embeddings for the given sequence id.
     * Mirrors `llama_context::get_embeddings_seq(seq_id)`.
     */
    fun getEmbeddingsSeq(seqId: LlamaSeqId): FloatArray? = embdSeq[seqId]

    /** Return the sampled tokens array. Mirrors `llama_context::get_sampled_tokens()`. */
    fun getSampledTokens(): IntArray? = sampling.sampled

    // -----------------------------------------------------------------------
    // Model / cparams accessors
    // -----------------------------------------------------------------------

    fun getModel(): LlamaModel = model
    fun getCparams(): LlamaCParams = cparams

    /** Return the abstract memory interface (KV cache). */
    fun getMemory(): LlamaMemory? = memory

    // -----------------------------------------------------------------------
    // Thread configuration
    // -----------------------------------------------------------------------

    fun setNThreads(nThreads: Int, nThreadsBatch: Int) {
        cparams.nThreads = nThreads
        cparams.nThreadsBatch = nThreadsBatch
    }

    fun attachThreadpool(threadpool: Any?, threadpoolBatch: Any?) {
        this.threadpool = threadpool
        this.threadpoolBatch = threadpoolBatch ?: threadpool
    }

    fun detachThreadpool() {
        this.threadpool = null
        this.threadpoolBatch = null
    }

    // -----------------------------------------------------------------------
    // Attention / embedding mode setters
    // -----------------------------------------------------------------------

    fun setEmbeddings(value: Boolean) {
        cparams.embeddings = value
    }

    fun setCausalAttn(value: Boolean) {
        if (cparams.causalAttn == value) return
        cparams.causalAttn = value
        schedNeedReserve = true
    }

    fun setWarmup(value: Boolean) {
        if (cparams.warmup == value) return
        cparams.warmup = value
    }

    fun setAbortCallback(callback: (() -> Boolean)?) {
        this.abortCallback = callback
    }

    // -----------------------------------------------------------------------
    // Memory (KV cache) management
    // -----------------------------------------------------------------------

    /**
     * Initialise the KV cache layers based on the resolved context parameters.
     * Called during construction; can be called again after a context reset.
     */
    private fun initMemory() {
        val nLayers = model.config.numHiddenLayers
        val nHeads = model.config.numKeyValueHeads
        val headDim = model.config.headDim
        val maxSeqLen = cparams.nCtx

        kvCaches = Array(nLayers) {
            KVCache(
                maxSequenceLength = maxSeqLen,
                numHeads = nHeads,
                headDim = headDim,
            ).also { it.initialize(graphAllocator) }
        }
    }

    /**
     * Attempt to update (compact / defragment) the memory module.
     * Returns `true` if anything was actually changed.
     * Mirrors `llama_context::memory_update(optimize)`.
     */
    fun memoryUpdate(optimize: Boolean): Boolean = memoryUpdateImpl(optimize)

    // -----------------------------------------------------------------------
    // Scheduler
    // -----------------------------------------------------------------------

    /**
     * Reserve a new backend scheduler if needed (e.g. after changing LoRA
     * adapters, samplers, or attention type).
     * Mirrors `llama_context::sched_reserve()`.
     */
    fun schedReserve() = schedReserveImpl()

    /**
     * Wait for all pending backend computations to finish.
     * Mirrors `llama_context::synchronize()`.
     */
    fun synchronize() = synchronizeImpl()

    // -----------------------------------------------------------------------
    // Core inference  (encode / decode)
    // -----------------------------------------------------------------------

    /**
     * Process a single micro-batch through the graph.
     *
     * If a [memoryContext] is provided it is applied first (e.g. to set up the
     * KV cache view for the current ubatch).
     *
     * Returns a pair of (graphResult, status). The graph result is `null` when
     * the computation fails.
     *
     * Mirrors `llama_context::process_ubatch(ubatch, gtype, mctx, ret)`.
     */
    fun processUBatch(
        ubatch: LlamaUBatch,
        gtype: LlmGraphType,
        @Suppress("UNUSED_PARAMETER") memoryContext: LlamaMemoryContext? = null,
    ): Pair<GGMLCGraph?, Int> = processUBatchImpl(ubatch, gtype, memoryContext)

    /**
     * Encode a batch of tokens (non-causal / encoder path).
     *
     * Returns 0 on success, negative on failure (matches C++ convention).
     * Mirrors `llama_context::encode(batch_inp)`.
     */
    fun encode(batchInp: LlamaBatch): Int = encodeImpl(batchInp)

    /**
     * Decode a batch of tokens (causal / decoder path).
     *
     * Returns 0 on success, negative on failure (matches C++ convention).
     * Mirrors `llama_context::decode(batch_inp)`.
     */
    fun decode(batchInp: LlamaBatch): Int = decodeImpl(batchInp)

    // -----------------------------------------------------------------------
    // Output management
    // -----------------------------------------------------------------------

    /**
     * Reserve space for at least [nOutputsRequested] output rows.
     * Returns the actual number of rows reserved.
     * Mirrors `llama_context::output_reserve(n_outputs)`.
     */
    fun outputReserve(nOutputsRequested: Int): Int = outputReserveImpl(nOutputsRequested)

    /**
     * Reorder the output buffers according to [outputSwaps] so that
     * logits / embeddings are in the expected order.
     * Mirrors `llama_context::output_reorder()`.
     */
    private fun outputReorder() = outputReorderImpl()

    /**
     * Map a batch-token index [i] to the actual row inside the output buffer.
     * Supports negative indices (last output row).
     * Mirrors `llama_context::output_resolve_row(i)`.
     */
    private fun outputResolveRow(i: Int): Long {
        val j: Long = if (i < 0) {
            val resolved = nOutputs + i
            require(resolved >= 0) { "Negative index out of range [0, $nOutputs)" }
            resolved.toLong()
        } else {
            require(i < outputIds.size) { "Index out of range [0, ${outputIds.size})" }
            val mapped = outputIds[i]
            require(mapped >= 0) { "batch.logits[$i] != true" }
            mapped.toLong()
        }
        require(j < nOutputs) { "Corrupt output buffer (j=$j, nOutputs=$nOutputs)" }
        return j
    }

    // -----------------------------------------------------------------------
    // Graph helpers
    // -----------------------------------------------------------------------

    /**
     * Maximum number of nodes a graph may contain for the given token count.
     * Mirrors `llama_context::graph_max_nodes(n_tokens)`.
     */
    fun graphMaxNodes(nTokens: Int): Int = graphMaxNodesImpl(nTokens)

    /**
     * Submit a compute graph for asynchronous execution.
     * Returns the GGML status code.
     * Mirrors `llama_context::graph_compute(gf, batched)`.
     */
    fun graphCompute(
        graph: GGMLCGraph,
        batched: Boolean,
    ): Int = graphComputeImpl(graph, batched)

    /**
     * Reserve a graph with a dummy ubatch of the specified size.
     * Used during initialisation to pre-allocate backend buffers.
     * Mirrors `llama_context::graph_reserve(...)`.
     */
    fun graphReserve(
        nTokens: Int,
        nSeqs: Int,
        nOutputs: Int,
    ): GGMLCGraph? = graphReserveImpl(nTokens, nSeqs, nOutputs)

    // -----------------------------------------------------------------------
    // State save / load
    // -----------------------------------------------------------------------

    fun stateGetSize(): Long = stateGetSizeImpl()

    fun stateGetData(dst: ByteArray, size: Long): Long = stateGetDataImpl(dst, size)

    fun stateSetData(src: ByteArray, size: Long): Long = stateSetDataImpl(src, size)

    fun stateSeqGetSize(seqId: LlamaSeqId): Long = stateSeqGetSizeImpl(seqId)

    fun stateSeqGetData(seqId: LlamaSeqId, dst: ByteArray, size: Long): Long =
        stateSeqGetDataImpl(seqId, dst, size)

    fun stateSeqSetData(seqId: LlamaSeqId, src: ByteArray, size: Long): Long =
        stateSeqSetDataImpl(seqId, src, size)

    /**
     * Save full context state to a file. Mirrors `llama_context::state_save_file()`.
     * Writes magic, version, token prompt, then state data via [stateWriteDataImpl].
     */
    fun stateSaveFile(filepath: String, tokens: IntArray, nTokenCount: Int): Boolean {
        // LATER: implement file I/O when LlamaFile is available
        return false
    }

    /**
     * Load full context state from a file. Mirrors `llama_context::state_load_file()`.
     * Reads magic/version, prompt tokens, then restores state via [stateReadDataImpl].
     */
    fun stateLoadFile(
        filepath: String,
        tokensOut: IntArray,
        nTokenCapacity: Int,
    ): Pair<Boolean, Int> {
        // LATER: implement file I/O when LlamaFile is available
        return Pair(false, 0)
    }

    /**
     * Save a single sequence state to a file. Mirrors `llama_context::state_seq_save_file()`.
     */
    fun stateSeqSaveFile(
        seqId: LlamaSeqId,
        filepath: String,
        tokens: IntArray,
        nTokenCount: Int,
    ): Long {
        // LATER: implement file I/O when LlamaFile is available
        return 0L
    }

    /**
     * Load a single sequence state from a file. Mirrors `llama_context::state_seq_load_file()`.
     */
    fun stateSeqLoadFile(
        seqId: LlamaSeqId,
        filepath: String,
        tokensOut: IntArray,
        nTokenCapacity: Int,
    ): Pair<Long, Int> {
        // LATER: implement file I/O when LlamaFile is available
        return Pair(0L, 0)
    }

    // -----------------------------------------------------------------------
    // LoRA adapter management
    // -----------------------------------------------------------------------

    /**
     * Set LoRA adapters with their scales. Mirrors `llama_context::set_adapters_lora()`.
     */
    fun setAdaptersLora(adapters: List<Any>, scales: FloatArray) {
        loras.clear()
        for (i in adapters.indices) {
            if (scales[i] != 0.0f) {
                loras[adapters[i]] = scales[i]
            }
        }
        schedNeedReserve = true
    }

    /**
     * Check if the provided adapters+scales match what's currently loaded.
     * Mirrors `llama_context::adapters_lora_are_same()`.
     */
    fun adaptersLoraAreSame(adapters: List<Any>, scales: FloatArray): Boolean {
        var nNonZero = 0
        for (i in adapters.indices) {
            if (scales[i] == 0.0f) continue
            nNonZero++
            val existing = loras[adapters[i]]
            if (existing == null || existing != scales[i]) return false
        }
        return nNonZero == loras.size
    }

    // -----------------------------------------------------------------------
    // Graph callback
    // -----------------------------------------------------------------------

    /**
     * Return a naming/backend-assignment callback for graph building.
     * Mirrors `llama_context::graph_get_cb()`.
     */
    fun graphGetCb(): (GGMLTensor, String, Int) -> Unit = { cur, name, il ->
        if (il >= 0) {
            cur.name = "$name-$il"
        } else {
            cur.name = name
        }
    }

    // -----------------------------------------------------------------------
    // Performance
    // -----------------------------------------------------------------------

    /** Collect timing / throughput statistics. Mirrors `llama_context::perf_get_data()`. */
    fun perfGetData(): LlamaPerfContextData = LlamaPerfContextData(
        tStartMs = tStartUs * 1e-3,
        tLoadMs = tLoadUs * 1e-3,
        tPEvalMs = tPEvalUs * 1e-3,
        tEvalMs = tEvalUs * 1e-3,
        nPEval = maxOf(1, nPEval),
        nEval = maxOf(1, nEval),
        nReused = maxOf(0, nReused),
    )

    /** Reset the performance counters. Mirrors `llama_context::perf_reset()`. */
    fun perfReset() {
        tStartUs = currentTimeMicros()
        tEvalUs = 0L; nEval = 0
        tPEvalUs = 0L; nPEval = 0
        nReused = 0
    }

    // -----------------------------------------------------------------------
    // Internal helpers for extension functions (LlamaContextImpl.kt)
    // -----------------------------------------------------------------------

    /** Initialise the full-vocabulary token id array for backend samplers. */
    private fun initSamplingVocab() {
        val nVocab = model.config.vocabSize
        sampling.tokenIdsFullVocab = IntArray(nVocab) { it }
    }

    internal fun allocateLogits(size: Int) {
        logits = FloatArray(size)
        logitsSize = size
    }

    internal fun allocateEmbd(size: Int) {
        embd = FloatArray(size)
        embdSize = size
    }

    internal fun allocateOutputIds(size: Int) {
        outputIds = IntArray(size) { -1 }
    }

    internal fun resetOutputIds() {
        outputIds.fill(-1)
    }

    internal fun resetNOutputs() {
        nOutputs = 0
    }

    internal fun setNOutputsValue(value: Int) {
        nOutputs = value
    }

    internal fun setOutputId(index: Int, value: Int) {
        outputIds[index] = value
    }

    internal fun incrementNEval() {
        nEval++
    }

    internal fun incrementNPEval(count: Int) {
        nPEval += count
    }

    internal fun markEvaluated() {
        hasEvaluatedOnce = true
    }

    internal fun resetQueuedTokens() {
        nQueuedTokens = 0
        tComputeStartUs = 0
    }

    internal fun addQueuedTokens(n: Long) {
        nQueuedTokens += n
    }

    internal fun setComputeStart(us: Long) {
        tComputeStartUs = us
    }

    internal fun clearSchedNeedReserve() {
        schedNeedReserve = false
    }

    internal fun requestSchedReserve() {
        schedNeedReserve = true
    }

    /**
     * Public-facing wrapper around [outputResolveRow] for use by extension
     * functions that cannot access private members.
     */
    internal fun outputResolveRowPublic(i: Int): Long = outputResolveRow(i)

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    companion object {
        /**
         * Resolve raw [LlamaContextParams] + model [LlamaConfig] into the
         * computed [LlamaCParams] used throughout the context.
         *
         * This mirrors the first ~200 lines of the `llama_context` constructor
         * in `llama-context.cpp`.
         */
        internal fun resolveCParams(
            config: LlamaConfig,
            params: LlamaContextParams,
        ): LlamaCParams {
            val cp = LlamaCParams()

            cp.nSeqMax = maxOf(1, params.nSeqMax)
            require(cp.nSeqMax <= LLAMA_MAX_SEQ) { "nSeqMax must be <= $LLAMA_MAX_SEQ" }

            cp.nThreads = params.nThreads
            cp.nThreadsBatch = params.nThreadsBatch

            cp.yarnExtFactor = if (params.yarnExtFactor >= 0.0f) params.yarnExtFactor else 0.0f
            cp.yarnAttnFactor = if (params.yarnAttnFactor >= 0.0f) params.yarnAttnFactor else 1.0f
            cp.yarnBetaFast = if (params.yarnBetaFast >= 0.0f) params.yarnBetaFast else 32.0f
            cp.yarnBetaSlow = if (params.yarnBetaSlow >= 0.0f) params.yarnBetaSlow else 1.0f

            cp.embeddings = params.embeddings
            cp.offloadKqv = params.offloadKqv
            cp.noPerf = params.noPerf
            cp.warmup = false

            cp.nCtx = if (params.nCtx == 0) config.maxPositionEmbeddings else params.nCtx
            cp.ropeFreqBase = if (params.ropeFreqBase == 0.0f) config.ropeTheta else params.ropeFreqBase
            cp.ropeFreqScale = if (params.ropeFreqScale == 0.0f) 1.0f else params.ropeFreqScale

            cp.nCtxOrigYarn = when {
                params.yarnOrigCtx != 0 -> params.yarnOrigCtx
                else -> config.maxPositionEmbeddings
            }

            // Resolve attention type
            when (params.attentionType) {
                LlamaAttentionType.UNSPECIFIED -> cp.causalAttn = true
                LlamaAttentionType.CAUSAL -> cp.causalAttn = true
                LlamaAttentionType.NON_CAUSAL -> cp.causalAttn = false
            }

            cp.flashAttn = params.flashAttnType != LlamaFlashAttnType.DISABLED
            cp.autoFa = params.flashAttnType == LlamaFlashAttnType.AUTO

            // With causal attention the batch size is capped by context size
            cp.nBatch = if (cp.causalAttn) minOf(cp.nCtx, params.nBatch) else params.nBatch
            cp.nUbatch = minOf(cp.nBatch, if (params.nUbatch == 0) params.nBatch else params.nUbatch)

            cp.opOffload = params.opOffload
            cp.kvUnified = params.kvUnified

            // Pad n_ctx to a multiple of 256 (matches C++ GGML_PAD)
            cp.nCtx = ((cp.nCtx + 255) / 256) * 256

            if (cp.kvUnified) {
                cp.nCtxSeq = cp.nCtx
            } else {
                cp.nCtxSeq = cp.nCtx / cp.nSeqMax
                cp.nCtxSeq = ((cp.nCtxSeq + 255) / 256) * 256
                require(cp.nCtxSeq != 0) { "nCtxSeq == 0" }
                if (cp.nCtx != cp.nCtxSeq * cp.nSeqMax) {
                    cp.nCtx = cp.nCtxSeq * cp.nSeqMax
                }
            }

            // Resolve pooling type
            cp.poolingType = when (params.poolingType) {
                LlamaPoolingType.UNSPECIFIED -> LlamaPoolingType.NONE
                else -> params.poolingType
            }

            // Resolve rope scaling
            val ropeScaling = params.ropeScalingType
            if (ropeScaling == LlamaRopeScalingType.NONE) {
                cp.ropeFreqScale = 1.0f
            }
            if (cp.yarnExtFactor < 0.0f) {
                cp.yarnExtFactor =
                    if (ropeScaling == LlamaRopeScalingType.YARN) 1.0f else 0.0f
            }

            return cp
        }

        /** Microsecond clock — uses monotonic time source. */
        private val timeOrigin = TimeSource.Monotonic.markNow()
        private fun currentTimeMicros(): Long =
            timeOrigin.elapsedNow().inWholeMicroseconds
    }
}

// ---------------------------------------------------------------------------
// LlamaAttention  (multi-head attention mechanism)
// ---------------------------------------------------------------------------

/**
 * Multi-head attention mechanism for the LLaMA model.
 * Implements scaled dot-product attention with rotary position encoding (RoPE).
 */
class LlamaAttention(
    val hiddenSize: Int,
    val numHeads: Int,
    val headDim: Int = hiddenSize / numHeads,
    val maxPositionEmbeddings: Int = 2048,
    val ropeTheta: Float = 10000.0f
) {
    init {
        require(hiddenSize % numHeads == 0) {
            "Hidden size ($hiddenSize) must be divisible by number of heads ($numHeads)"
        }
        require(headDim * numHeads == hiddenSize) {
            "Head dimension ($headDim) * number of heads ($numHeads) must equal hidden size ($hiddenSize)"
        }
    }

    /**
     * Apply rotary position encoding to query and key tensors.
     */
    fun applyRoPE(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        tensor: GGMLTensor,
        position: Int
    ): GGMLTensor {
        val result = GGMLTensor(type = tensor.type)
        result.ne = tensor.ne.copyOf()
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        val batchSize = tensor.ne[2].toInt()
        val seqLen = tensor.ne[1].toInt()
        val dim = tensor.ne[0].toInt()
        
        // Allocate result tensor
        graphAllocator.allocateTensor(result)
        
        // Apply RoPE for each batch and sequence position
        for (b in 0 until batchSize) {
            for (s in 0 until seqLen) {
                for (head in 0 until numHeads) {
                    val headOffset = head * headDim
                    
                    // Apply rotation for each pair of dimensions in the head
                    for (i in 0 until headDim step 2) {
                        val pos = position + s
                        val exponent = (i.toDouble() / 2.0) / (headDim.toDouble() / 2.0)
                        val invFreq = 1.0 / ropeTheta.toDouble().pow(exponent)
                        val angle = pos * invFreq
                        val cos = cos(angle).toFloat()
                        val sin = sin(angle).toFloat()
                        
                        val x = tensor.getFloat(graphAllocator, headOffset + i, s, b)
                        val y = tensor.getFloat(graphAllocator, headOffset + i + 1, s, b)
                        
                        result.setFloat(graphAllocator, x * cos - y * sin, headOffset + i, s, b)
                        result.setFloat(graphAllocator, x * sin + y * cos, headOffset + i + 1, s, b)
                    }
                }
            }
        }
        
        return result
    }

    /**
     * Compute scaled dot-product attention.
     */
    fun computeAttention(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        query: GGMLTensor,
        key: GGMLTensor,
        value: GGMLTensor,
        attentionMask: GGMLTensor? = null,
        kvCache: KVCache? = null
    ): GGMLTensor {
        // Apply RoPE to query and key
        val qRope = applyRoPE(context, graphAllocator, query, 0)
        val kRope = applyRoPE(context, graphAllocator, key, 0)
        
        // If using KV cache, concatenate with cached keys/values
        val finalKey = if (kvCache != null) {
            kvCache.updateKey(graphAllocator, kRope)
        } else {
            kRope
        }
        
        val finalValue = if (kvCache != null) {
            kvCache.updateValue(graphAllocator, value)
        } else {
            value
        }
        
        // Compute attention scores: Q @ K^T
        val keyTransposed = transpose(context, graphAllocator, finalKey)
        val scores = matmul(context, graphAllocator, qRope, keyTransposed)
        
        // Scale by sqrt(head_dim)
        val scaleFactor = 1.0f / sqrt(headDim.toFloat())
        val scaledScores = scale(context, graphAllocator, scores, scaleFactor)
        
        // Apply attention mask if provided
        val maskedScores = if (attentionMask != null) {
            add(context, graphAllocator, scaledScores, attentionMask)
        } else {
            scaledScores
        }

        // Apply softmax using destination-based implementation
        val attentionWeights = GGMLTensor(type = maskedScores.type).apply {
            ne = maskedScores.ne.copyOf()
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
            op = GGMLOp.SOFT_MAX
            src[0] = maskedScores
            graphAllocator.allocateTensor(this)
        }
        computeSoftMax(graphAllocator, context, maskedScores, attentionWeights)
        
        // Apply attention to values: attention_weights @ V
        val output = matmul(context, graphAllocator, attentionWeights, finalValue)
        
        return output
    }

    /**
     * Helper function to transpose a tensor (swap last two dimensions).
     */
    private fun transpose(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        tensor: GGMLTensor
    ): GGMLTensor {
        val shape = tensor.ne.copyOf()
        val tmp = shape[0]
        shape[0] = tensor.ne[1]
        shape[1] = tmp

        val result = GGMLTensor(type = tensor.type).apply {
            ne = shape
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
            op = GGMLOp.TRANSPOSE
            src[0] = tensor
            graphAllocator.allocateTensor(this)
        }

        computeTranspose(graphAllocator, context, tensor, result)
        return result
    }

    /**
     * Helper function for matrix multiplication using existing infrastructure.
     */
    private fun matmul(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        a: GGMLTensor,
        b: GGMLTensor
    ): GGMLTensor {
        val result = GGMLTensor(type = GGMLType.F32)
        
        // Set result dimensions for matrix multiplication: [M, N, batch_dims...]
        // a is [M, K], b is [K, N] -> result is [M, N]
        result.ne[0] = a.ne[0]  // M
        result.ne[1] = b.ne[1]  // N
        result.ne[2] = maxOf(a.ne[2], b.ne[2])  // batch size
        result.ne[3] = maxOf(a.ne[3], b.ne[3])  // additional batch dims
        
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        // Set up operation
        result.op = GGMLOp.MUL_MAT
        result.src[0] = a
        result.src[1] = b
        
        // Allocate result tensor
        graphAllocator.allocateTensor(result)
        
        // Use existing matrix multiplication implementation
        computeMatMul(graphAllocator, context, a, b, result)
        
        return result
    }

    /**
     * Helper function for scaling tensor values.
     */
    private fun scale(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        tensor: GGMLTensor,
        factor: Float
    ): GGMLTensor {
        val result = GGMLTensor(type = tensor.type)
        result.ne = tensor.ne.copyOf()
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        // Allocate result tensor
        graphAllocator.allocateTensor(result)
        
        // Scale each element
        val numElements = tensor.numElements().toInt()
        for (i in 0 until numElements) {
            // Convert flat index to multi-dimensional indices
            val indices = IntArray(GGML_MAX_DIMS) { 0 }
            var temp = i
            for (d in 0 until GGML_MAX_DIMS) {
                indices[d] = temp % tensor.ne[d].toInt()
                temp /= tensor.ne[d].toInt()
            }
            
            val value = tensor.getFloat(graphAllocator, *indices)
            result.setFloat(graphAllocator, value * factor, *indices)
        }
        
        return result
    }

    /**
     * Helper function for element-wise addition.
     */
    private fun add(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        a: GGMLTensor,
        b: GGMLTensor
    ): GGMLTensor {
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne = a.ne.copyOf()
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        // Set up operation
        result.op = GGMLOp.ADD
        result.src[0] = a
        result.src[1] = b
        
        // Allocate result tensor
        graphAllocator.allocateTensor(result)
        
        // Use existing ADD operation from GGMLComputeOps
        computeAdd(graphAllocator, context, a, b, result)
        
        return result
    }

    /**
     * Placeholder for graph computation - will be implemented with proper graph execution.
     */
    private fun computeGraph(context: GGMLContext, graphAllocator: GGMLGraphAllocator, graph: GGMLCGraph) {
        // This would typically use the existing graph computation infrastructure
        // For now, this is a skeleton that would be filled in with proper implementation
        // The actual computation would happen through the existing GGMLComputeOps infrastructure
    }
}
