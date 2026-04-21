// port-lint: source llama.cpp/src/llama-context.cpp
/**
 * Implementation methods for [LlamaContext] — encode, decode, graph building,
 * output management, state save/load, and optimization.
 *
 * Split from `LlamaAttention.kt` to keep file sizes manageable. All methods
 * here are either extension functions on [LlamaContext] or free-standing helpers
 * that mirror their C++ counterparts from `llama-context.cpp`.
 */
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import kotlin.math.*

// Constants LLAMA_SESSION_MAGIC, LLAMA_SESSION_VERSION, LLAMA_STATE_SEQ_MAGIC,
// LLAMA_STATE_SEQ_VERSION are defined in LlamaHTypes.kt



// =============================================================================
// GGML status codes  (mirrors ggml_status enum)
// =============================================================================

/** Status codes returned by graph compute operations. */
object GGMLStatus {
    const val SUCCESS = 0
    const val FAILED = 1
    const val ABORTED = 2
    const val ALLOC_FAILED = 3
}

// =============================================================================
// SamplingInfo — nested data class for backend sampling state
// =============================================================================

/**
 * Per-context sampling bookkeeping.
 *
 * Port of `llama_context::sampling_info`.
 */
class SamplingInfo {
    /** Active backend samplers keyed by sequence id. */
    val samplers: MutableMap<LlamaSeqId, Any> = mutableMapOf()

    var logits: FloatArray? = null
    var logitsSize: Int = 0
    var sampled: IntArray? = null
    var sampledSize: Int = 0
    var probs: FloatArray? = null
    var probsSize: Int = 0
    var candidates: IntArray? = null
    var candidatesSize: Int = 0

    val logitsCount: MutableList<Int> = mutableListOf()
    val probsCount: MutableList<Int> = mutableListOf()
    val candidatesCount: MutableList<Int> = mutableListOf()

    /** Full vocabulary token ids for fallback candidate list. */
    var tokenIdsFullVocab: IntArray = IntArray(0)
}

// =============================================================================
// Cross-attention state reference
// =============================================================================
// LlamaCross is defined in LlamaGraph.kt

// =============================================================================
// I/O interfaces for state serialization
// =============================================================================

/**
 * Write interface for state serialization.
 * Port of `llama_io_write_i`.
 */
interface LlamaIoWriter {
    fun write(data: ByteArray)
    fun writeTensor(tensor: GGMLTensor, offset: Long, size: Long)
    fun writeString(str: String) {
        val bytes = str.encodeToByteArray()
        writeInt(bytes.size)
        write(bytes)
    }
    fun writeInt(value: Int) {
        val buf = ByteArray(4)
        buf[0] = (value and 0xFF).toByte()
        buf[1] = ((value shr 8) and 0xFF).toByte()
        buf[2] = ((value shr 16) and 0xFF).toByte()
        buf[3] = ((value shr 24) and 0xFF).toByte()
        write(buf)
    }
    fun nBytes(): Long
}

/**
 * Read interface for state deserialization.
 * Port of `llama_io_read_i`.
 */
interface LlamaIoReader {
    fun read(size: Int): ByteArray
    fun readTo(dst: ByteArray, size: Int) {
        val src = read(size)
        src.copyInto(dst, 0, 0, size)
    }
    fun readString(): String {
        val len = readInt()
        val bytes = read(len)
        return bytes.decodeToString()
    }
    fun readInt(): Int {
        val buf = read(4)
        return (buf[0].toInt() and 0xFF) or
            ((buf[1].toInt() and 0xFF) shl 8) or
            ((buf[2].toInt() and 0xFF) shl 16) or
            ((buf[3].toInt() and 0xFF) shl 24)
    }
    fun nBytes(): Long
}

// =============================================================================
// Dummy I/O writer (for computing state size without writing)
// =============================================================================

/** Measures how much data *would* be written. Port of `llama_io_write_dummy`. */
class LlamaIoWriteDummy : LlamaIoWriter {
    private var sizeWritten: Long = 0L
    override fun write(data: ByteArray) { sizeWritten += data.size }
    override fun writeTensor(tensor: GGMLTensor, offset: Long, size: Long) { sizeWritten += size }
    override fun nBytes(): Long = sizeWritten
}

// =============================================================================
// Buffer I/O writer
// =============================================================================

/** Writes state into a pre-allocated [ByteArray]. Port of `llama_io_write_buffer`. */
class LlamaIoWriteBuffer(private val dst: ByteArray) : LlamaIoWriter {
    private var pos: Int = 0
    override fun write(data: ByteArray) {
        require(pos + data.size <= dst.size) { "unexpectedly reached end of buffer" }
        data.copyInto(dst, pos)
        pos += data.size
    }
    override fun writeTensor(tensor: GGMLTensor, offset: Long, size: Long) {
        // TODO: full tensor read-back from backend
        require(pos + size.toInt() <= dst.size) { "unexpectedly reached end of buffer" }
        pos += size.toInt()
    }
    override fun nBytes(): Long = pos.toLong()
}

// =============================================================================
// Buffer I/O reader
// =============================================================================

/** Reads state from a [ByteArray]. Port of `llama_io_read_buffer`. */
class LlamaIoReadBuffer(private val src: ByteArray) : LlamaIoReader {
    private var pos: Int = 0
    override fun read(size: Int): ByteArray {
        require(pos + size <= src.size) { "unexpectedly reached end of buffer" }
        val result = src.copyOfRange(pos, pos + size)
        pos += size
        return result
    }
    override fun nBytes(): Long = pos.toLong()
}

// =============================================================================
// Extension: LlamaContext.synchronize()
// =============================================================================

/**
 * Full implementation of synchronize — updates perf counters.
 * Replaces the TODO stub in [LlamaContext].
 */
fun LlamaContext.synchronizeImpl() {
    // Add the evaluation to the stats
    if (nQueuedTokens == 1L) {
        if (!cparams.noPerf) {
            tEvalUs += currentTimeMicrosCtx() - tComputeStartUs
        }
        incrementNEval()
    } else if (nQueuedTokens > 1L) {
        if (!cparams.noPerf) {
            tPEvalUs += currentTimeMicrosCtx() - tComputeStartUs
        }
        incrementNPEval(nQueuedTokens.toInt())
    }

    // Get a more accurate load time upon first eval
    if (nQueuedTokens > 0 && !hasEvaluatedOnce) {
        tLoadUs = currentTimeMicrosCtx() - tStartUs
        markEvaluated()
    }

    resetQueuedTokens()
}

// =============================================================================
// Extension: LlamaContext.outputReserveImpl()
// =============================================================================

/**
 * Full implementation of output_reserve — allocates logits / embeddings buffers.
 *
 * Port of `llama_context::output_reserve(int32_t n_outputs)`.
 */
fun LlamaContext.outputReserveImpl(nOutputsRequested: Int): Int {
    val nOutputsMax = maxOf(nOutputsRequested.toLong(), nSeqMax().toLong()).toInt()
    val nVocab = model.config.vocabSize
    val nEmbd = model.config.hiddenSize
    val hasLogits = true
    val hasEmbd = cparams.embeddings

    val logitsNeeded = if (hasLogits) nVocab.toLong() * nOutputsMax else 0L
    val embdNeeded = if (hasEmbd) nEmbd.toLong() * nOutputsMax else 0L

    // Allocate or grow buffers as needed
    val currentLogitsSize = logitsSize.toLong()
    if (logitsNeeded > currentLogitsSize) {
        allocateLogits(logitsNeeded.toInt())
    }
    val currentEmbdSize = embdSize.toLong()
    if (embdNeeded > currentEmbdSize) {
        allocateEmbd(embdNeeded.toInt())
    }

    // Init output_ids if not yet done
    if (outputIds.isEmpty()) {
        allocateOutputIds(cparams.nBatch)
    }

    // Reset all ids as invalid (negative)
    resetOutputIds()
    resetNOutputs()

    return nOutputsMax
}

// =============================================================================
// Extension: LlamaContext.outputReorderImpl()
// =============================================================================

/**
 * Full implementation of output_reorder — applies pending [outputSwaps] to the
 * logits / embeddings buffers so they appear in the batch-order expected by
 * callers.
 *
 * Port of `llama_context::output_reorder()`.
 */
fun LlamaContext.outputReorderImpl() {
    val nVocab = model.config.vocabSize
    val nEmbd = model.config.hiddenSize
    val logitsBuf = logits
    val embdBuf = embd

    for (swap in outputSwaps) {
        val i0 = swap.i0
        val i1 = swap.i1

        if (logitsBuf != null && logitsSize > 0) {
            for (k in 0 until nVocab) {
                val a = i0 * nVocab + k
                val b = i1 * nVocab + k
                val tmp = logitsBuf[a]
                logitsBuf[a] = logitsBuf[b]
                logitsBuf[b] = tmp
            }
        }

        if (embdBuf != null && embdSize > 0) {
            for (k in 0 until nEmbd) {
                val a = i0 * nEmbd + k
                val b = i1 * nEmbd + k
                val tmp = embdBuf[a]
                embdBuf[a] = embdBuf[b]
                embdBuf[b] = tmp
            }
        }
    }

    outputSwaps.clear()
}

// =============================================================================
// Extension: LlamaContext.graphMaxNodesImpl()
// =============================================================================

/**
 * Maximum number of nodes a graph may contain for the given token count.
 *
 * Port of `llama_context::graph_max_nodes(uint32_t n_tokens)`.
 */
fun LlamaContext.graphMaxNodesImpl(@Suppress("UNUSED_PARAMETER") nTokens: Int): Int {
    // Conservative estimate: 8× number of model tensors, minimum 1024
    val nTensors = model.config.numHiddenLayers * 16L // approximate tensors per layer
    return maxOf(1024, (8L * nTensors).toInt())
}

// =============================================================================
// Extension: LlamaContext.graphComputeImpl()
// =============================================================================

/**
 * Submit a compute graph for execution.
 *
 * Port of `llama_context::graph_compute(ggml_cgraph * gf, bool batched)`.
 * In the full implementation this dispatches to the backend scheduler.
 * For now returns SUCCESS as a stub.
 */
fun LlamaContext.graphComputeImpl(graph: GGMLCGraph, batched: Boolean): Int {
    val nThreadsToUse = if (batched) cparams.nThreadsBatch else cparams.nThreads
    // TODO: full backend scheduler integration
    //   - Set threadpool on CPU backend
    //   - Set n_threads on all backends
    //   - Call ggml_backend_sched_graph_compute_async
    @Suppress("UNUSED_VARIABLE") val unused1 = nThreadsToUse
    return GGMLStatus.SUCCESS
}

// =============================================================================
// Extension: LlamaContext.graphReserveImpl()
// =============================================================================

/**
 * Reserve a graph with a dummy ubatch of the specified size.
 *
 * Port of `llama_context::graph_reserve(...)`.
 */
fun LlamaContext.graphReserveImpl(nTokens: Int, nSeqs: Int, nOutputsReq: Int): GGMLCGraph? {
    require(nOutputsReq >= 1) { "nOutputs must be >= 1" }

    var adjustedTokens = nTokens
    var adjustedOutputs = nOutputsReq

    if (adjustedTokens % nSeqs != 0) {
        adjustedTokens = ((adjustedTokens + (nSeqs - 1)) / nSeqs) * nSeqs
        adjustedOutputs = maxOf(adjustedOutputs, adjustedTokens)
    }

    // TODO: full implementation
    //   - Reset scheduler
    //   - Create dummy ubatch via batch allocator
    //   - Build graph via model.buildGraph(graphParams)
    //   - Schedule / allocate the graph
    //   - Return the compute graph

    return null // stub — full backend integration pending
}

// =============================================================================
// Extension: LlamaContext.graphParamsImpl()
// =============================================================================

/**
 * Construct [LlmGraphParams] for a graph build invocation.
 *
 * Port of `llama_context::graph_params(...)`.
 */
fun LlamaContext.graphParamsImpl(
    res: LlmGraphResult,
    ubatch: LlamaUBatch,
    gtype: LlmGraphType,
): LlmGraphParams {
    return LlmGraphParams(
        arch = LlamaModelArch.LLAMA,
        hparams = LlamaHparams(
            nEmbd = model.config.hiddenSize,
            nLayer = model.config.numHiddenLayers,
        ),
        cparams = cparams,
        ubatch = ubatch,
        gtype = gtype,
        nOutputs = nOutputs,
        res = res,
    )
}

// =============================================================================
// Extension: LlamaContext.processUBatchImpl()
// =============================================================================

/**
 * Process a single micro-batch through the graph (internal version).
 *
 * Port of `llama_context::process_ubatch(ubatch, gtype, mctx, ret)`.
 *
 * @return Pair of (LlmGraphResult or null, status code)
 */
private fun LlamaContext.processUBatchInternal(
    ubatch: LlamaUBatch,
    gtype: LlmGraphType,
    memoryContext: LlamaMemoryContext? = null,
): Pair<LlmGraphResult?, Int> {
    // Apply memory context (e.g. set up KV cache view for current ubatch)
    if (memoryContext != null && !memoryContext.apply()) {
        llamaLogError("processUBatch: failed to apply memory context\n")
        return null to GGMLStatus.FAILED
    }

    val res = gfResPrev as? LlmGraphResult
    if (res == null) {
        llamaLogError("processUBatch: no previous graph result available\n")
        return null to GGMLStatus.FAILED
    }

    val gparams = graphParamsImpl(res, ubatch, gtype).copy(mctx = memoryContext)

    if (!graphReuseDisable && res.canReuse(gparams)) {
        // Graph can be reused — synchronize if pipeline parallel to avoid
        // overwriting inputs that the previous compute is still reading.
        if (cparams.pipelineParallel) {
            synchronize()
        }
        nReused++
    } else {
        // Build a new graph
        res.reset()

        // In the full implementation this would call:
        //   ggml_backend_sched_reset(sched)
        //   gf = model.build_graph(gparams)
        //   ggml_backend_sched_alloc_graph(sched, gf)
        // For now, we build a minimal graph placeholder.
        val gf = GGMLCGraph(size = graphMaxNodesImpl(ubatch.nTokens))
        res.gf = gf

        if (res.getGf() == null) {
            llamaLogError("processUBatch: failed to build graph\n")
            return null to GGMLStatus.FAILED
        }
    }

    // Set the input data for the input tensors
    res.setInputs(ubatch)

    // Execute graph compute
    val gf = res.getGf()
    if (gf == null) {
        llamaLogError("processUBatch: graph is null after setup\n")
        return null to GGMLStatus.FAILED
    }

    val status = graphComputeImpl(gf, ubatch.nTokens > 1)
    if (status != GGMLStatus.SUCCESS) {
        llamaLogError("processUBatch: failed to compute graph, status: $status\n")
        return null to status
    }

    return res to GGMLStatus.SUCCESS
}

/**
 * Process a single micro-batch through the graph.
 *
 * Public-facing version that returns the compute graph (compatible with
 * [LlamaContext.processUBatch] signature).
 *
 * @return Pair of (compute graph or null, status code)
 */
fun LlamaContext.processUBatchImpl(
    ubatch: LlamaUBatch,
    gtype: LlmGraphType,
    memoryContext: LlamaMemoryContext? = null,
): Pair<GGMLCGraph?, Int> {
    val (res, status) = processUBatchInternal(ubatch, gtype, memoryContext)
    return (res?.getGf()) to status
}

// =============================================================================
// Extension: LlamaContext.encodeImpl()
// =============================================================================

/**
 * Encode a batch of tokens (non-causal / encoder path).
 *
 * Returns 0 on success, negative on failure.
 * Port of `llama_context::encode(const llama_batch & batch_inp)`.
 */
fun LlamaContext.encodeImpl(batchInp: LlamaBatch): Int {
    require((batchInp.tokens != null) xor (batchInp.embeddings != null)) {
        "Exactly one of tokens or embeddings must be set"
    }

    if (batchInp.nTokens == 0) {
        llamaLogError("encode: nTokens == 0\n")
        return -1
    }

    val nVocab = model.config.vocabSize
    val nEmbd = model.config.hiddenSize

    // Initialize the batch allocator
    val ba = balloc
    if (ba == null) {
        llamaLogError("encode: batch allocator not initialized\n")
        return -1
    }

    val nSeqMax = if (cparams.kvUnified) LLAMA_MAX_SEQ else cparams.nSeqMax

    if (!ba.init(batchInp, nVocab, nEmbd, nSeqMax, true)) {
        llamaLogError("encode: failed to initialize batch\n")
        return -1
    }

    val nTokens = ba.getNTokens()

    // micro-batching is not possible for non-causal encoding — process full batch
    require(cparams.nUbatch >= nTokens) { "encoder requires nUbatch >= nTokens" }

    if (tComputeStartUs == 0L) {
        setComputeStart(currentTimeMicrosCtx())
    }

    // Clear sequence embeddings
    embdSeq.clear()

    schedReserveImpl()

    addQueuedTokens(nTokens.toLong())

    // Reserve output buffer
    val reserved = outputReserveImpl(nTokens)
    if (reserved < nTokens) {
        llamaLogError("encode: could not reserve space for batch with $nTokens outputs\n")
        return -2
    }

    // Map all tokens to their output indices
    for (i in 0 until nTokens) {
        setOutputId(i, i)
    }
    setNOutputsValue(nTokens)

    // Save and temporarily override causal attention
    val causalAttnOrig = cparams.causalAttn
    cparams.causalAttn = false

    // Build the ubatch from the full batch via splitSimple (processes everything at once)
    val ubatchInternal = ba.splitSimple(nTokens)
    val ubatch = ubatchInternal.toLlamaUBatch()

    // Process the ubatch
    val (res, status) = processUBatchInternal(ubatch, LlmGraphType.ENCODER, null)

    // Restore causal attention
    cparams.causalAttn = causalAttnOrig

    if (res == null) {
        return when (status) {
            GGMLStatus.ABORTED -> 2
            GGMLStatus.ALLOC_FAILED -> -2
            else -> -3
        }
    }

    // Extract logits
    val tLogits = res.tLogits
    val tEmbd = res.tEmbdPooled ?: res.tEmbd

    if (logits != null && tLogits != null) {
        // In the full implementation this copies from the backend tensor to the host buffer.
        // For now the data is assumed to already be accessible via the tensor's data field.
        val logitsBuf = logits
        val srcData = tLogits.data as? ByteArray
        if (logitsBuf != null && srcData != null) {
            val count = minOf(nTokens * nVocab, logitsSize)
            copyF32FromBytes(srcData, logitsBuf, 0, count)
        }
    }

    // Extract embeddings
    if (embd != null && tEmbd != null) {
        val embdBuf = embd!!
        val nEmbdOut = nEmbd // In deepstack models this may differ
        when (cparams.poolingType) {
            LlamaPoolingType.NONE -> {
                val srcData = tEmbd.data as? ByteArray
                if (srcData != null) {
                    val count = minOf(nTokens * nEmbdOut, embdSize)
                    copyF32FromBytes(srcData, embdBuf, 0, count)
                }
            }
            LlamaPoolingType.MEAN, LlamaPoolingType.CLS, LlamaPoolingType.LAST -> {
                val srcData = tEmbd.data as? ByteArray
                if (srcData != null) {
                    val seqIdUnqArr = ubatch.seqIdUnq
                    val seqIdxArr = ubatch.seqIdx
                    if (seqIdUnqArr != null && seqIdxArr != null) {
                        for (s in 0 until ubatch.nSeqsUnq) {
                            val seqId = seqIdUnqArr[s]
                            val seqIdx = seqIdxArr[seqId]
                            val seqEmbd = FloatArray(nEmbdOut)
                            copyF32FromBytes(srcData, seqEmbd, nEmbdOut * seqIdx, nEmbdOut)
                            embdSeq[seqId] = seqEmbd
                        }
                    }
                }
            }
            LlamaPoolingType.RANK -> {
                // Extract rerank scores (n_cls_out floats per sequence)
                val srcData = tEmbd.data as? ByteArray
                if (srcData != null) {
                    val seqIdUnqArr = ubatch.seqIdUnq
                    val seqIdxArr = ubatch.seqIdx
                    if (seqIdUnqArr != null && seqIdxArr != null) {
                        for (s in 0 until ubatch.nSeqsUnq) {
                            val seqId = seqIdUnqArr[s]
                            val seqIdx = seqIdxArr[seqId]
                            val rankEmbd = FloatArray(1) // nClsOut typically 1
                            copyF32FromBytes(srcData, rankEmbd, seqIdx, 1)
                            embdSeq[seqId] = rankEmbd
                        }
                    }
                }
            }
            LlamaPoolingType.UNSPECIFIED -> {
                error("unknown pooling type")
            }
        }
    }

    // T5 cross-attention: copy encoder output for later use by decoder
    if (tEmbd != null) {
        val tEmbdNe0 = tEmbd.ne[0]
        val tEmbdNe1 = tEmbd.ne[1]
        if (tEmbdNe0 > 0 && tEmbdNe1 > 0) {
            cross.nEmbd = tEmbdNe0
            cross.nEnc = tEmbdNe1
            val nFloats = (tEmbdNe0 * tEmbdNe1).toInt()
            cross.vEmbd.clear()
            val embdBuf = embd
            if (embdBuf != null && nFloats <= embdSize) {
                for (k in 0 until nFloats) {
                    cross.vEmbd.add(embdBuf[k])
                }
            }
            // Remember sequence IDs used during encoding
            val batchAfterInit = ba.getBatch()
            cross.seqIdsEnc.clear()
            val batchSeqId = batchAfterInit.seqId
            val batchNSeqId = batchAfterInit.nSeqId
            if (batchSeqId != null && batchNSeqId != null) {
                for (i in 0 until nTokens) {
                    val seqSet = mutableSetOf<LlamaSeqId>()
                    for (s in 0 until batchNSeqId[i]) {
                        seqSet.add(batchSeqId[i][s])
                    }
                    cross.seqIdsEnc.add(seqSet)
                }
            }
        }
    }

    return 0
}

// =============================================================================
// Extension: LlamaContext.decodeImpl()
// =============================================================================

/**
 * Decode a batch of tokens (causal / decoder path).
 *
 * Returns 0 on success, 1 when no KV slot available, negative on failure.
 * Port of `llama_context::decode(const llama_batch & batch_inp)`.
 */
fun LlamaContext.decodeImpl(batchInp: LlamaBatch): Int {
    require((batchInp.tokens != null) xor (batchInp.embeddings != null)) {
        "Exactly one of tokens or embeddings must be set"
    }

    // If no memory module, fall back to encode (encoder-only model)
    if (memory == null) {
        llamaLogDebug("decode: cannot decode without memory (calling encode instead)\n")
        return encodeImpl(batchInp)
    }

    if (batchInp.nTokens == 0) {
        llamaLogError("decode: nTokens == 0\n")
        return -1
    }

    val nVocab = model.config.vocabSize
    val nEmbd = model.config.hiddenSize

    // When computing embeddings, all tokens are output
    val outputAll = cparams.embeddings
    val hasSamplers = sampling.samplers.isNotEmpty()

    val nSeqMax = if (cparams.kvUnified) LLAMA_MAX_SEQ else cparams.nSeqMax

    // Validate: backend sampling requires at most one output token per sequence
    if (hasSamplers && batchInp.logits != null) {
        val seqOutputCount = IntArray(nSeqMax)
        for (i in 0 until batchInp.nTokens) {
            if (batchInp.logits!![i] == false) continue
            val ns = batchInp.nSeqId?.get(i) ?: 1
            for (s in 0 until ns) {
                val seqId = batchInp.seqId?.get(i)?.get(s) ?: 0
                seqOutputCount[seqId]++
                if (seqOutputCount[seqId] > 1) {
                    llamaLogError("decode: backend sampling requires at most one output token per sequence (seq_id $seqId had ${seqOutputCount[seqId]})\n")
                    return -1
                }
            }
        }
    }

    // Initialize the batch allocator
    val ba = balloc
    if (ba == null) {
        llamaLogError("decode: batch allocator not initialized\n")
        return -1
    }

    if (!ba.init(batchInp, nVocab, nEmbd, nSeqMax, outputAll, memory)) {
        llamaLogError("decode: failed to initialize batch\n")
        return -1
    }

    val nTokensAll = ba.getNTokens()
    val nOutputsAll = ba.getNOutputs()

    if (outputAll) {
        if (nOutputsAll != nTokensAll) {
            llamaLogError("decode: pooled embedding requires that all tokens are output (nOutputsAll=$nOutputsAll, nTokensAll=$nTokensAll)\n")
            return -1
        }
    }

    check(nTokensAll <= cparams.nBatch) { "batch size $nTokensAll exceeds n_batch ${cparams.nBatch}" }
    check(cparams.causalAttn || cparams.nUbatch >= nTokensAll) {
        "non-causal attention requires nUbatch >= nTokens"
    }

    if (tComputeStartUs == 0L) {
        setComputeStart(currentTimeMicrosCtx())
    }
    addQueuedTokens(nTokensAll.toLong())

    // Clear sequence embeddings and output swaps
    embdSeq.clear()
    outputSwaps.clear()

    schedReserveImpl()

    var didOptimize = false

    // Handle any pending shifts/copies
    memoryUpdateImpl(false)

    // Initialize memory context — retry after optimization if needed
    val mem = memory!!
    var mctx: LlamaMemoryContext
    while (true) {
        mctx = mem.initBatch(ba, cparams.nUbatch, outputAll)

        when (mctx.getStatus()) {
            LlamaMemoryStatus.SUCCESS -> break
            LlamaMemoryStatus.NO_UPDATE -> {
                llamaLogError("decode: unexpected memory context status: NO_UPDATE\n")
                return -2
            }
            LlamaMemoryStatus.FAILED_PREPARE -> {
                if (!didOptimize) {
                    didOptimize = true
                    if (memoryUpdateImpl(true)) {
                        llamaLogDebug("decode: retrying batch size ${ba.getNTokens()} after cache optimization\n")
                        continue
                    }
                }
                llamaLogWarn("decode: failed to find a memory slot for batch of size ${ba.getNTokens()}\n")
                return 1
            }
            LlamaMemoryStatus.FAILED_COMPUTE -> {
                llamaLogError("decode: compute failed while preparing batch of size ${ba.getNTokens()}\n")
                return -2
            }
        }
    }

    // Reserve output buffer
    val reserved = outputReserveImpl(nOutputsAll)
    if (reserved < nOutputsAll) {
        llamaLogError("decode: could not reserve space for batch with $nOutputsAll outputs\n")
        return -2
    }

    var nOutputsPrev = 0

    // Process ubatches from the memory context
    do {
        val ubatch = mctx.getUbatch()

        // Count the outputs in this ubatch
        val nOutputsNew = if (nOutputsAll == nTokensAll) {
            ubatch.nTokens
        } else {
            val outArr = ubatch.output
            if (outArr != null) {
                var count = 0
                for (i in 0 until ubatch.nTokens) {
                    if (outArr[i]) count++
                }
                count
            } else {
                ubatch.nTokens
            }
        }

        // Set before graph build
        setNOutputsValue(nOutputsNew)

        val (res, status) = processUBatchInternal(ubatch, LlmGraphType.DECODER, mctx)

        if (res == null) {
            // Failed — remove all positions of this ubatch from the memory module
            val posMin = IntArray(LLAMA_MAX_SEQ) { Int.MAX_VALUE }

            val ubatchSeqId = ubatch.seqId
            val ubatchPos = ubatch.pos
            if (ubatchSeqId != null && ubatchPos != null) {
                for (i in 0 until ubatch.nTokens) {
                    val seqId = ubatchSeqId[i][0]
                    if (ubatchPos[i] < posMin[seqId]) {
                        posMin[seqId] = ubatchPos[i]
                    }
                }
            }

            for (s in 0 until LLAMA_MAX_SEQ) {
                if (posMin[s] == Int.MAX_VALUE) continue
                llamaLogWarn("decode: removing memory entries for seq_id=$s, pos=[${posMin[s]}, +inf)\n")
                mem.seqRm(s, posMin[s], -1)
            }

            return when (status) {
                GGMLStatus.ABORTED -> 2
                GGMLStatus.ALLOC_FAILED -> -2
                else -> -3
            }
        }

        // Extract logits
        val tLogits = res.tLogits
        if (logits != null && tLogits != null && nOutputsNew > 0 &&
            needsRawLogits(ubatch, sampling.samplers)
        ) {
            val logitsBuf = logits!!
            val srcData = tLogits.data as? ByteArray
            if (srcData != null) {
                check(nOutputsPrev + nOutputsNew <= nOutputsAll)
                val offset = nOutputsPrev * nVocab
                val count = minOf(nOutputsNew * nVocab, logitsSize - offset)
                if (count > 0) {
                    copyF32FromBytes(srcData, logitsBuf, offset, count)
                }
            }
        }

        // Extract embeddings
        val tEmbd = if (cparams.embeddings) {
            val pooled = res.tEmbdPooled
            pooled ?: res.tEmbd
        } else null

        if (embd != null && tEmbd != null && nOutputsNew > 0) {
            val embdBuf = embd!!
            val nEmbdOut = nEmbd
            val srcData = tEmbd.data as? ByteArray

            when (cparams.poolingType) {
                LlamaPoolingType.NONE -> {
                    if (srcData != null) {
                        val offset = nOutputsPrev * nEmbdOut
                        val count = minOf(nOutputsNew * nEmbdOut, embdSize - offset)
                        if (count > 0) {
                            copyF32FromBytes(srcData, embdBuf, offset, count)
                        }
                    }
                }
                LlamaPoolingType.MEAN, LlamaPoolingType.CLS, LlamaPoolingType.LAST -> {
                    if (srcData != null) {
                        val seqIdUnqArr = ubatch.seqIdUnq
                        val seqIdxArr = ubatch.seqIdx
                        if (seqIdUnqArr != null && seqIdxArr != null) {
                            for (s in 0 until ubatch.nSeqsUnq) {
                                val seqId = seqIdUnqArr[s]
                                val seqIdx = seqIdxArr[seqId]
                                val seqEmbd = FloatArray(nEmbdOut)
                                copyF32FromBytes(srcData, seqEmbd, nEmbdOut * seqIdx, nEmbdOut)
                                embdSeq[seqId] = seqEmbd
                            }
                        }
                    }
                }
                LlamaPoolingType.RANK -> {
                    if (srcData != null) {
                        val seqIdUnqArr = ubatch.seqIdUnq
                        val seqIdxArr = ubatch.seqIdx
                        if (seqIdUnqArr != null && seqIdxArr != null) {
                            for (s in 0 until ubatch.nSeqsUnq) {
                                val seqId = seqIdUnqArr[s]
                                val seqIdx = seqIdxArr[seqId]
                                val rankEmbd = FloatArray(1)
                                copyF32FromBytes(srcData, rankEmbd, seqIdx, 1)
                                embdSeq[seqId] = rankEmbd
                            }
                        }
                    }
                }
                LlamaPoolingType.UNSPECIFIED -> error("unknown pooling type")
            }
        }

        nOutputsPrev += nOutputsNew
    } while (mctx.next())

    // Set to total number of outputs in the batch
    setNOutputsValue(nOutputsAll)

    // Set output mappings
    if (nOutputsAll > 0) {
        var sortedOutput = true
        val outIds = ba.getOutIds()

        check(outIds.size == nOutputsAll) {
            "outIds.size=${outIds.size} != nOutputsAll=$nOutputsAll"
        }

        for (i in 0 until nOutputsAll) {
            val outId = outIds[i]
            setOutputId(outId, i)
            if (outId != i) {
                sortedOutput = false
            }
        }

        // Reorder outputs if not already sorted (relevant for recurrent models)
        if (!sortedOutput && nOutputsAll > 1) {
            // Selection sort to minimize swaps
            for (i in 0 until nOutputsAll - 1) {
                var jMin = i
                for (j in (i + 1) until nOutputsAll) {
                    if (outIds[j] < outIds[jMin]) {
                        jMin = j
                    }
                }
                if (jMin == i) continue

                // Swap in outIds
                val tmp = outIds[i]
                outIds[i] = outIds[jMin]
                outIds[jMin] = tmp

                // Remember the swap for lazy logits/embeddings reordering
                outputSwaps.add(SwapInfo(i, jMin))
            }

            // Rebuild output_ids mapping
            resetOutputIds()
            for (i in 0 until nOutputsAll) {
                setOutputId(outIds[i], i)
            }
        }
    }

    return 0
}

// =============================================================================
// Extension: LlamaContext.memoryUpdateImpl()
// =============================================================================

/**
 * Update the memory module (compact / defragment).
 * Returns `true` if anything was actually changed.
 *
 * Port of `llama_context::memory_update(bool optimize)`.
 */
fun LlamaContext.memoryUpdateImpl(@Suppress("UNUSED_PARAMETER") optimize: Boolean): Boolean {
    // TODO: full implementation
    //   1. Call memory.initUpdate(this, optimize)
    //   2. Check status — return false for NO_UPDATE or failures
    //   3. Reset previous graph result
    //   4. Call mctx.apply()
    //   5. Re-reserve worst-case graph
    return false
}

// =============================================================================
// Extension: LlamaContext.schedReserveImpl()
// =============================================================================

/**
 * Reserve a new backend scheduler if needed.
 *
 * Port of `llama_context::sched_reserve()`.
 */
fun LlamaContext.schedReserveImpl() {
    if (!schedNeedReserve) return
    clearSchedNeedReserve()

    synchronizeImpl()

    val nSeqs = cparams.nSeqMax
    val nTokens = minOf(cparams.nCtx, cparams.nUbatch)
    val maxNodes = graphMaxNodesImpl(nTokens)

    // TODO: full implementation
    //   1. Create new scheduler with backend_ptrs/buft
    //   2. Initialize memory context (full)
    //   3. Resolve auto flash-attention
    //   4. Reserve pp and tg graphs
    //   5. Log buffer sizes and graph info
    @Suppress("UNUSED_VARIABLE") val unused2 = nSeqs
    @Suppress("UNUSED_VARIABLE") val unused3 = maxNodes
}

// =============================================================================
// Extension: State save/load
// =============================================================================

/**
 * Compute the total serialized state size.
 * Port of `llama_context::state_get_size()`.
 */
fun LlamaContext.stateGetSizeImpl(): Long {
    val io = LlamaIoWriteDummy()
    return try {
        stateWriteDataImpl(io)
    } catch (_: Exception) {
        0L
    }
}

/**
 * Serialize the context state into [dst].
 * Port of `llama_context::state_get_data(dst, size)`.
 */
fun LlamaContext.stateGetDataImpl(dst: ByteArray, size: Long): Long {
    val io = LlamaIoWriteBuffer(dst)
    return try {
        stateWriteDataImpl(io)
    } catch (_: Exception) {
        0L
    }
}

/**
 * Deserialize context state from [src].
 * Port of `llama_context::state_set_data(src, size)`.
 */
fun LlamaContext.stateSetDataImpl(src: ByteArray, size: Long): Long {
    val io = LlamaIoReadBuffer(src)
    return try {
        stateReadDataImpl(io)
    } catch (_: Exception) {
        0L
    }
}

/**
 * Write full context state to an I/O writer.
 * Port of `llama_context::state_write_data(io)`.
 */
fun LlamaContext.stateWriteDataImpl(io: LlamaIoWriter): Long {
    // Write model arch string for validation on load
    val archStr = "llama"
    io.writeString(archStr)

    // Delegate to memory module via adapter
    val mem = memory
    if (mem != null) {
        val adapter = object : LlamaIoWrite {
            override fun write(src: ByteArray, size: Int) { io.write(src.copyOf(size)) }
            override fun writeTensor(tensor: GGMLTensor, offset: Int, size: Int) {
                io.writeTensor(tensor, offset.toLong(), size.toLong())
            }
            override fun nBytes(): Long = io.nBytes()
        }
        mem.stateWrite(adapter)
    }

    return io.nBytes()
}

/**
 * Read full context state from an I/O reader.
 * Port of `llama_context::state_read_data(io)`.
 */
fun LlamaContext.stateReadDataImpl(io: LlamaIoReader): Long {
    // Read and validate model info
    val archStr = io.readString()
    val expectedArch = "llama"
    require(archStr == expectedArch) {
        "wrong model arch: '$archStr' instead of '$expectedArch'"
    }

    // Delegate to memory module via adapter
    val mem = memory
    if (mem != null) {
        val adapter = object : LlamaIoRead {
            override fun read(size: Int): ByteArray = io.read(size)
            override fun readTo(dst: ByteArray, size: Int) { io.readTo(dst, size) }
            override fun nBytes(): Long = io.nBytes()
        }
        mem.stateRead(adapter)
    }

    return io.nBytes()
}

/**
 * Compute serialized size for a single sequence.
 * Port of `llama_context::state_seq_get_size(seq_id, flags)`.
 */
fun LlamaContext.stateSeqGetSizeImpl(seqId: LlamaSeqId, flags: Int = 0): Long {
    val io = LlamaIoWriteDummy()
    return try {
        stateSeqWriteDataImpl(io, seqId, flags)
    } catch (_: Exception) {
        0L
    }
}

/**
 * Serialize a single sequence's state.
 * Port of `llama_context::state_seq_get_data(seq_id, dst, size, flags)`.
 */
fun LlamaContext.stateSeqGetDataImpl(seqId: LlamaSeqId, dst: ByteArray, size: Long, flags: Int = 0): Long {
    val io = LlamaIoWriteBuffer(dst)
    return try {
        stateSeqWriteDataImpl(io, seqId, flags)
    } catch (_: Exception) {
        0L
    }
}

/**
 * Deserialize a single sequence's state.
 * Port of `llama_context::state_seq_set_data(seq_id, src, size, flags)`.
 */
fun LlamaContext.stateSeqSetDataImpl(seqId: LlamaSeqId, src: ByteArray, size: Long, flags: Int = 0): Long {
    val io = LlamaIoReadBuffer(src)
    return try {
        stateSeqReadDataImpl(io, seqId, flags)
    } catch (_: Exception) {
        0L
    }
}

/**
 * Write per-sequence state.
 * Port of `llama_context::state_seq_write_data(io, seq_id, flags)`.
 */
fun LlamaContext.stateSeqWriteDataImpl(io: LlamaIoWriter, seqId: LlamaSeqId, flags: Int): Long {
    val mem = memory
    if (mem != null) {
        val adapter = object : LlamaIoWrite {
            override fun write(src: ByteArray, size: Int) { io.write(src.copyOf(size)) }
            override fun writeTensor(tensor: GGMLTensor, offset: Int, size: Int) {
                io.writeTensor(tensor, offset.toLong(), size.toLong())
            }
            override fun nBytes(): Long = io.nBytes()
        }
        mem.stateWrite(adapter, seqId, flags)
    }
    return io.nBytes()
}

/**
 * Read per-sequence state.
 * Port of `llama_context::state_seq_read_data(io, seq_id, flags)`.
 */
fun LlamaContext.stateSeqReadDataImpl(io: LlamaIoReader, seqId: LlamaSeqId, flags: Int): Long {
    val mem = memory
    if (mem != null) {
        val adapter = object : LlamaIoRead {
            override fun read(size: Int): ByteArray = io.read(size)
            override fun readTo(dst: ByteArray, size: Int) { io.readTo(dst, size) }
            override fun nBytes(): Long = io.nBytes()
        }
        mem.stateRead(adapter, seqId, flags)
    }
    return io.nBytes()
}

// =============================================================================
// Extension: LlamaContext.setSampler()
// =============================================================================

/**
 * Register a backend sampler for a given sequence id.
 *
 * Port of `llama_context::set_sampler(seq_id, sampler)`.
 */
fun LlamaContext.setSamplerImpl(seqId: LlamaSeqId, sampler: Any?): Boolean {
    if (sampler == null && seqId !in sampling.samplers) {
        return true
    }

    if (sampler != null) {
        // TODO: check can_offload and call backend_init
        sampling.samplers[seqId] = sampler
        requestSchedReserve()
        return true
    }

    // Remove sampler for this sequence
    sampling.samplers.remove(seqId)
    requestSchedReserve()
    return true
}

// =============================================================================
// Extension: Sampled token/logits/probs access
// =============================================================================

/**
 * Get the sampled token for the i-th output position.
 * Port of `llama_context::get_sampled_token_ith(idx)`.
 */
fun LlamaContext.getSampledTokenIth(idx: Int): LlamaToken {
    outputReorderImpl()
    val buf = sampling.sampled ?: return LLAMA_TOKEN_NULL
    if (buf.isEmpty()) return LLAMA_TOKEN_NULL
    return try {
        val row = outputResolveRowPublic(idx)
        if (row < 0 || row >= buf.size) LLAMA_TOKEN_NULL else buf[row.toInt()]
    } catch (_: Exception) {
        LLAMA_TOKEN_NULL
    }
}

/**
 * Get the sampled logits for the i-th output position.
 * Port of `llama_context::get_sampled_logits_ith(idx)`.
 */
fun LlamaContext.getSampledLogitsIth(idx: Int): FloatArray? {
    outputReorderImpl()
    val buf = sampling.logits ?: return null
    if (buf.isEmpty()) return null
    return try {
        val row = outputResolveRowPublic(idx)
        val nVocab = model.config.vocabSize
        if (row < 0 || row.toInt() >= sampling.logitsCount.size || sampling.logitsCount[row.toInt()] == 0) {
            null
        } else {
            val start = (row * nVocab).toInt()
            buf.copyOfRange(start, start + nVocab)
        }
    } catch (_: Exception) {
        null
    }
}

/**
 * Count of sampled logits for the i-th output.
 * Port of `llama_context::get_sampled_logits_count(idx)`.
 */
fun LlamaContext.getSampledLogitsCount(idx: Int): Int {
    outputReorderImpl()
    val buf = sampling.logits ?: return model.config.vocabSize
    if (buf.isEmpty()) return model.config.vocabSize
    return try {
        val row = outputResolveRowPublic(idx)
        if (row.toInt() >= sampling.logitsCount.size) 0 else sampling.logitsCount[row.toInt()]
    } catch (_: Exception) {
        0
    }
}

/**
 * Get the sampled probabilities for the i-th output position.
 * Port of `llama_context::get_sampled_probs_ith(idx)`.
 */
fun LlamaContext.getSampledProbsIth(idx: Int): FloatArray? {
    outputReorderImpl()
    val buf = sampling.probs ?: return null
    if (buf.isEmpty()) return null
    return try {
        val row = outputResolveRowPublic(idx)
        val nVocab = model.config.vocabSize
        if (row < 0 || row.toInt() >= sampling.probsCount.size || sampling.probsCount[row.toInt()] == 0) {
            null
        } else {
            val start = (row * nVocab).toInt()
            buf.copyOfRange(start, start + nVocab)
        }
    } catch (_: Exception) {
        null
    }
}

/**
 * Count of sampled probs for the i-th output.
 * Port of `llama_context::get_sampled_probs_count(idx)`.
 */
fun LlamaContext.getSampledProbsCount(idx: Int): Int {
    outputReorderImpl()
    val buf = sampling.probs ?: return 0
    if (buf.isEmpty()) return 0
    return try {
        val row = outputResolveRowPublic(idx)
        if (row.toInt() >= sampling.probsCount.size) 0 else sampling.probsCount[row.toInt()]
    } catch (_: Exception) {
        0
    }
}

/**
 * Get the candidate tokens for the i-th output position.
 * Port of `llama_context::get_sampled_candidates_ith(idx)`.
 */
fun LlamaContext.getSampledCandidatesIth(idx: Int): IntArray {
    outputReorderImpl()
    val buf = sampling.candidates
    return try {
        val row = outputResolveRowPublic(idx)
        if (buf != null && buf.isNotEmpty() &&
            row.toInt() < sampling.candidatesCount.size &&
            sampling.candidatesCount[row.toInt()] > 0
        ) {
            val nVocab = model.config.vocabSize
            val start = (row * nVocab).toInt()
            buf.copyOfRange(start, start + nVocab)
        } else {
            sampling.tokenIdsFullVocab
        }
    } catch (_: Exception) {
        sampling.tokenIdsFullVocab
    }
}

/**
 * Count of candidates for the i-th output.
 * Port of `llama_context::get_sampled_candidates_count(idx)`.
 */
fun LlamaContext.getSampledCandidatesCount(idx: Int): Int {
    outputReorderImpl()
    val buf = sampling.candidates ?: return 0
    if (buf.isEmpty()) return 0
    return try {
        val row = outputResolveRowPublic(idx)
        if (row.toInt() >= sampling.candidatesCount.size) 0 else sampling.candidatesCount[row.toInt()]
    } catch (_: Exception) {
        0
    }
}

// =============================================================================
// Extension: LlamaContext.setAdapterCvec()
// =============================================================================

/**
 * Apply a control vector to the model.
 * Port of `llama_context::set_adapter_cvec(data, len, n_embd, il_start, il_end)`.
 */
fun LlamaContext.setAdapterCvecImpl(
    data: FloatArray,
    len: Int,
    nEmbd: Int,
    ilStart: Int,
    ilEnd: Int,
): Boolean {
    // TODO: full implementation — apply cvec to relevant layers
    requestSchedReserve()
    return true
}

// =============================================================================
// Extension: LlamaContext.optInit()
// =============================================================================

/**
 * Initialize the optimization context for training.
 * Port of `llama_context::opt_init(model, lopt_params)`.
 */
fun LlamaContext.optInitImpl() {
    // TODO: full implementation
    //   - Set up ggml_opt context
    //   - Mark model parameters for gradient computation
    //   - Configure optimizer type and hyperparameters
}

// =============================================================================
// Extension: LlamaContext.optEpoch()
// =============================================================================

/**
 * Run one training epoch over a dataset.
 * Port of `llama_context::opt_epoch(...)`.
 */
fun LlamaContext.optEpochImpl(
    @Suppress("UNUSED_PARAMETER") idataSplit: Long,
    @Suppress("UNUSED_PARAMETER") train: Boolean,
) {
    // TODO: full implementation
    //   - Iterate over dataset entries
    //   - Build graphs for each batch
    //   - Compute forward + backward
    //   - Apply optimizer step
}

// =============================================================================
// Extension: memory breakdown
// =============================================================================

/**
 * Collect per-backend memory usage.
 * Port of `llama_context::memory_breakdown()`.
 */
fun LlamaContext.memoryBreakdownImpl(): Map<String, LlamaMemoryBreakdownData> {
    val result = mutableMapOf<String, LlamaMemoryBreakdownData>()
    // TODO: full implementation
    //   - Aggregate model buffer sizes
    //   - Aggregate memory (KV cache) buffer sizes
    //   - Aggregate compute buffer sizes from scheduler
    result["cpu"] = LlamaMemoryBreakdownData()
    return result
}

// =============================================================================
// Helper: build_seq_to_output_row (free function)
// =============================================================================

/**
 * Build a mapping from sequence id to output row.
 * Port of static `build_seq_to_output_row()` from `llama-context.cpp`.
 */
fun buildSeqToOutputRow(ubatch: LlamaUBatch, rowOffset: Int): Map<LlamaSeqId, Int> {
    val seqToRow = mutableMapOf<LlamaSeqId, Int>()
    var local = 0
    val output = ubatch.output ?: return seqToRow
    val seqIds = ubatch.seqId ?: return seqToRow

    for (i in 0 until ubatch.nTokens) {
        if (!output[i]) continue
        val seqId = seqIds[i][0]
        seqToRow[seqId] = rowOffset + local
        local++
    }
    return seqToRow
}

// =============================================================================
// Helper: needsRawLogits (free function)
// =============================================================================

/**
 * Check whether any output token needs raw logits (no backend sampler for its sequence).
 * Port of static `needs_raw_logits()` from `llama-context.cpp`.
 */
fun needsRawLogits(ubatch: LlamaUBatch, samplers: Map<LlamaSeqId, Any>): Boolean {
    val output = ubatch.output ?: return true
    val seqIds = ubatch.seqId ?: return true

    for (i in 0 until ubatch.nTokens) {
        if (!output[i]) continue
        // Check if any sequence for this token lacks a backend sampler
        for (sid in seqIds[i]) {
            if (sid !in samplers) return true
        }
    }
    return false
}

// =============================================================================
// Utility: LlamaUBatchInternal → LlamaUBatch conversion
// =============================================================================

/**
 * Convert an internal micro-batch (from the batch allocator) to the public
 * [LlamaUBatch] type used by graph building and memory contexts.
 */
internal fun LlamaUBatchInternal.toLlamaUBatch(): LlamaUBatch {
    return LlamaUBatch(
        nTokens = nTokens,
        nSeqTokens = nSeqTokens,
        nSeqs = nSeqs,
        nSeqsUnq = nSeqsUnq,
        nPos = nPos,
        tokens = token,
        embeddings = embd,
        pos = pos,
        nSeqId = nSeqId,
        seqId = seqId,
        seqIdUnq = seqIdUnq,
        seqIdx = seqIdx,
        output = output?.let { bytes ->
            BooleanArray(bytes.size) { i -> bytes[i] != 0.toByte() }
        },
        equalSeqs = equalSeqs,
    )
}

// =============================================================================
// Utility: copyF32FromBytes — extract Float values from a ByteArray
// =============================================================================

/**
 * Copy [count] floats from a little-endian [src] ByteArray into [dst] starting
 * at index [dstOffset]. Each float is 4 bytes in the source.
 *
 * Used to simulate `ggml_backend_tensor_get_async` for host-accessible tensors.
 */
internal fun copyF32FromBytes(src: ByteArray, dst: FloatArray, dstOffset: Int, count: Int) {
    val effectiveCount = minOf(count, dst.size - dstOffset)
    for (i in 0 until effectiveCount) {
        val byteIdx = i * 4
        if (byteIdx + 3 >= src.size) break
        val bits = (src[byteIdx].toInt() and 0xFF) or
            ((src[byteIdx + 1].toInt() and 0xFF) shl 8) or
            ((src[byteIdx + 2].toInt() and 0xFF) shl 16) or
            ((src[byteIdx + 3].toInt() and 0xFF) shl 24)
        dst[dstOffset + i] = Float.fromBits(bits)
    }
}

// =============================================================================
// Utility: currentTimeMicrosCtx (extension-accessible clock)
// =============================================================================

/**
 * Monotonic microsecond clock accessible from extension functions.
 * Wraps the same TimeSource used by [LlamaContext.Companion].
 */
private val ctxTimeOrigin = kotlin.time.TimeSource.Monotonic.markNow()
internal fun currentTimeMicrosCtx(): Long = ctxTimeOrigin.elapsedNow().inWholeMicroseconds
