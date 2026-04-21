// port-lint: source llama.cpp/src/llama-graph.h llama.cpp/src/llama-graph.cpp
/**
 * Core graph types ported from `llama-graph.h` / `llama-graph.cpp`.
 *
 * These types are referenced by both [InferencePipeline] and [LlamaGraph] for
 * building and executing computation graphs during inference.  Enums, the
 * [LlmGraphInput] interface, concrete input implementations, result / parameter
 * containers, and the base [LlmGraphContext] builder class all live here.
 */
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import kotlin.math.*

// ---------------------------------------------------------------------------
// Tensor data-writing helpers (mirror ggml_backend_tensor_set)
// ---------------------------------------------------------------------------

/** Write an IntArray into a tensor's backing ByteArray at byte [offset]. */
private fun tensorSetI32(tensor: GGMLTensor, src: IntArray, offset: Int, count: Int) {
    val data = tensor.data as? ByteArray ?: return
    for (i in 0 until count) {
        val v = src[i]
        val pos = offset + i * 4
        data[pos]     = (v and 0xFF).toByte()
        data[pos + 1] = ((v shr 8) and 0xFF).toByte()
        data[pos + 2] = ((v shr 16) and 0xFF).toByte()
        data[pos + 3] = ((v shr 24) and 0xFF).toByte()
    }
}

/** Write a FloatArray into a tensor's backing ByteArray at byte [offset]. */
private fun tensorSetF32(tensor: GGMLTensor, src: FloatArray, offset: Int, count: Int) {
    val data = tensor.data as? ByteArray ?: return
    for (i in 0 until count) {
        val bits = src[i].toRawBits()
        val pos = offset + i * 4
        data[pos]     = (bits and 0xFF).toByte()
        data[pos + 1] = ((bits shr 8) and 0xFF).toByte()
        data[pos + 2] = ((bits shr 16) and 0xFF).toByte()
        data[pos + 3] = ((bits shr 24) and 0xFF).toByte()
    }
}

/** Write a single float into a tensor at byte [offset]. */
private fun tensorSetF32Single(tensor: GGMLTensor, value: Float, offset: Int) {
    val data = tensor.data as? ByteArray ?: return
    val bits = value.toRawBits()
    data[offset]     = (bits and 0xFF).toByte()
    data[offset + 1] = ((bits shr 8) and 0xFF).toByte()
    data[offset + 2] = ((bits shr 16) and 0xFF).toByte()
    data[offset + 3] = ((bits shr 24) and 0xFF).toByte()
}

/** Write a single I32 into a tensor at byte [offset]. */
private fun tensorSetI32Single(tensor: GGMLTensor, value: Int, offset: Int) {
    val data = tensor.data as? ByteArray ?: return
    data[offset]     = (value and 0xFF).toByte()
    data[offset + 1] = ((value shr 8) and 0xFF).toByte()
    data[offset + 2] = ((value shr 16) and 0xFF).toByte()
    data[offset + 3] = ((value shr 24) and 0xFF).toByte()
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/** Top-level graph variant used during inference. */
enum class LlmGraphType {
    DEFAULT,
    ENCODER,
    DECODER,
}

/** Activation / gating function applied inside feed-forward blocks. */
enum class LlmFfnOpType {
    SILU,
    GELU,
    RELU,
    RELU_SQR,
    SWIGLU,
    GEGLU,
    REGLU,
    SWIGLU_OAI_MOE,
}

/** How the FFN gate tensor is combined with the up-projection. */
enum class LlmFfnGateType {
    /** Gate applied sequentially. */
    SEQ,
    /** Gate applied in parallel. */
    PAR,
}

/** Normalisation variant applied to hidden states. */
enum class LlmNormType {
    NORM,
    RMS,
    GROUP,
}

// ---------------------------------------------------------------------------
// Graph callback type
// ---------------------------------------------------------------------------

/**
 * Callback that allows custom logic to be applied to each tensor during graph
 * construction (e.g. naming, offloading, ggml-alloc tagging).
 *
 * Port of `llm_graph_cb`.
 */
typealias LlmGraphCb = (ubatch: LlamaUBatch, cur: GGMLTensor, name: String, il: Int) -> Unit

// ---------------------------------------------------------------------------
// LlmGraphInput – interface + concrete implementations
// ---------------------------------------------------------------------------

/**
 * A single named input that must be populated before graph evaluation.
 *
 * Each implementation owns one or more [GGMLTensor] fields that are filled by
 * [setInput] from the current micro-batch and can optionally report whether a
 * previously built graph is still valid via [canReuse].
 */
interface LlmGraphInput {
    /** Populate owned tensors from the supplied micro-batch. */
    fun setInput(ubatch: LlamaUBatch)

    /** Return `true` when the graph built with the previous params can be reused. */
    fun canReuse(params: LlmGraphParams): Boolean = false
}

// -- embedding tokens -------------------------------------------------------

/** Input: token ids or pre-computed embeddings. */
class LlmGraphInputEmbd(val nEmbd: Long) : LlmGraphInput {
    /** I32 [nBatch] – token indices. */
    var tokens: GGMLTensor? = null
    /** F32 [nEmbd, nBatch] – pre-computed token embeddings. */
    var embd: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        // Set token ids into the tokens tensor
        val tok = ubatch.tokens
        val tokTensor = tokens
        if (tok != null && tokTensor != null) {
            tensorSetI32(tokTensor, tok, 0, ubatch.nTokens)
        }

        // Set pre-computed embeddings
        val emb = ubatch.embeddings
        val embTensor = embd
        if (emb != null && embTensor != null) {
            tensorSetF32(embTensor, emb, 0, ubatch.nTokens * nEmbd.toInt())
        }
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        var ok = true
        ok = ok && (params.ubatch.tokens == null || (tokens != null && tokens!!.ne[0] == params.ubatch.nTokens.toLong()))
        ok = ok && (params.ubatch.embeddings == null || (embd != null && embd!!.ne[1] == params.ubatch.nTokens.toLong()))
        return ok
    }
}

// -- positions --------------------------------------------------------------

/** Input: per-token position ids. */
class LlmGraphInputPos(val nPosPerEmbd: Int = 1) : LlmGraphInput {
    /** I32 [nBatch] – absolute position for each token. */
    var pos: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        val p = ubatch.pos ?: return
        val t = pos ?: return
        val nTokens = ubatch.nTokens

        if (ubatch.tokens != null && nPosPerEmbd == 4) {
            // M-RoPE with text tokens: convert 1D positions to 4D
            val posData = IntArray(nTokens * 4)
            for (i in 0 until nTokens) {
                posData[i] = p[i]
                posData[nTokens + i] = p[i]
                posData[2 * nTokens + i] = p[i]
                posData[3 * nTokens + i] = 0
            }
            tensorSetI32(t, posData, 0, posData.size)
        } else {
            tensorSetI32(t, p, 0, nTokens * nPosPerEmbd)
        }
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        val p = pos ?: return false
        return p.ne[0] == params.ubatch.nTokens.toLong() * nPosPerEmbd
    }
}

// -- output ids -------------------------------------------------------------

/** Input: indices of tokens whose logits are actually needed. */
class LlmGraphInputOutIds(
    val hparams: LlamaHparams,
    val cparams: LlamaCParams,
    val nOutputs: Int,
) : LlmGraphInput {
    /** I32 [nOutputs] – indices into the batch that require output. */
    var outIds: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        val t = outIds ?: return
        val nTokens = ubatch.nTokens

        if (nOutputs == nTokens) {
            val ids = IntArray(nTokens) { it }
            tensorSetI32(t, ids, 0, nTokens)
            return
        }

        val output = ubatch.output ?: return
        val ids = mutableListOf<Int>()
        for (i in 0 until nTokens) {
            if (output[i]) ids.add(i)
        }
        tensorSetI32(t, ids.toIntArray(), 0, ids.size)
    }

    override fun canReuse(params: LlmGraphParams): Boolean = nOutputs == params.nOutputs
}

// -- mean pooling -----------------------------------------------------------

/** Input: mean-pooling weight matrix. */
class LlmGraphInputMean(val cparams: LlamaCParams) : LlmGraphInput {
    /** F32 [nBatch, nBatch] – mean-pooling weights. */
    var mean: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        if (!cparams.embeddings) return
        if (cparams.poolingType != LlamaPoolingType.MEAN &&
            cparams.poolingType != LlamaPoolingType.RANK) return

        val t = mean ?: return
        val nTokens = ubatch.nTokens
        val nSeqTokens = ubatch.nSeqTokens
        val nSeqsUnq = ubatch.nSeqsUnq

        // Zero-fill the weight matrix
        val data = FloatArray(nTokens * nSeqsUnq)

        // Sum up how many tokens belong to each unique sequence
        val sums = LongArray(nSeqsUnq)
        val seqIdArr = ubatch.seqId ?: return
        val nSeqIdArr = ubatch.nSeqId ?: return
        val seqIdxArr = ubatch.seqIdx ?: return

        var i = 0
        while (i < nTokens) {
            for (s in 0 until nSeqIdArr[i]) {
                val seqId = seqIdArr[i][s]
                val seqIdx = seqIdxArr[seqId]
                sums[seqIdx] += nSeqTokens.toLong()
            }
            i += nSeqTokens
        }

        // Compute reciprocals
        val div = FloatArray(nSeqsUnq) { s ->
            val sum = sums[s]
            if (sum > 0) 1.0f / sum.toFloat() else 0.0f
        }

        // Fill the weight matrix: each token row gets 1/count for its sequence
        i = 0
        while (i < nTokens) {
            for (s in 0 until nSeqIdArr[i]) {
                val seqId = seqIdArr[i][s]
                val seqIdx = seqIdxArr[seqId]
                for (j in 0 until nSeqTokens) {
                    data[seqIdx * nTokens + i + j] = div[seqIdx]
                }
            }
            i += nSeqTokens
        }

        tensorSetF32(t, data, 0, data.size)
    }
}

// -- CLS token selection ----------------------------------------------------

/** Input: CLS token indices for classification pooling. */
class LlmGraphInputCls(
    val cparams: LlamaCParams,
    val arch: LlamaModelArch,
) : LlmGraphInput {
    /** I32 [nBatch] – CLS token index per sequence. */
    var cls: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        val nTokens = ubatch.nTokens
        val nSeqsUnq = ubatch.nSeqsUnq

        if (!cparams.embeddings) return
        if (cparams.poolingType != LlamaPoolingType.CLS &&
            cparams.poolingType != LlamaPoolingType.RANK &&
            cparams.poolingType != LlamaPoolingType.LAST) return

        val t = cls ?: return
        val posArr = ubatch.pos ?: return
        val seqIdArr = ubatch.seqId ?: return
        val nSeqIdArr = ubatch.nSeqId ?: return
        val seqIdxArr = ubatch.seqIdx ?: return

        val targetPos = IntArray(nSeqsUnq) { -1 }
        val targetRow = IntArray(nSeqsUnq) { -1 }

        // Qwen3 reranking/embedding models use last token for RANK pooling
        val last = (cparams.poolingType == LlamaPoolingType.LAST ||
            (cparams.poolingType == LlamaPoolingType.RANK &&
                (arch == LlamaModelArch.QWEN3 || arch == LlamaModelArch.QWEN3VL)))

        for (i in 0 until nTokens) {
            val pos = posArr[i]

            for (s in 0 until nSeqIdArr[i]) {
                val seqId = seqIdArr[i][s]
                val seqIdx = seqIdxArr[seqId]

                if (targetPos[seqIdx] == -1 ||
                    (last && pos >= targetPos[seqIdx]) ||
                    (!last && pos < targetPos[seqIdx])) {
                    targetPos[seqIdx] = pos
                    targetRow[seqIdx] = i
                }
            }
        }

        // Write indices as I32 (uint32_t in C++)
        val data = IntArray(nSeqsUnq) { s ->
            if (targetRow[s] >= 0) targetRow[s] else 0
        }
        tensorSetI32(t, data, 0, nSeqsUnq)
    }
}

// -- recurrent state --------------------------------------------------------

/** Input: state-copy bookkeeping for recurrent (SSM / RWKV) models. */
class LlmGraphInputRs(
    var mctx: LlamaMemoryRecurrentContext?,
) : LlmGraphInput {
    /** I32 [nRs] – full state-copy source indices. */
    var sCopy: GGMLTensor? = null
    /** I32 [nSeqs] – state-copy indices for the main sequences. */
    var sCopyMain: GGMLTensor? = null
    /** I32 [nRs - nSeqs] – state-copy indices for extra sequences. */
    var sCopyExtra: GGMLTensor? = null
    var head: Int = 0
    var rsZ: Int = 0

    override fun setInput(ubatch: LlamaUBatch) {
        val ctx = mctx ?: return
        val nRs = ctx.getNRs()

        val t = sCopy ?: return
        val data = IntArray(nRs)
        // Assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (i in 0 until nRs) {
            data[i] = ctx.sCopy(i)
        }
        tensorSetI32(t, data, 0, nRs)
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        val ctx = params.mctx as? LlamaMemoryRecurrentContext ?: return false
        this.mctx = ctx

        val sc = sCopy ?: return false
        val scMain = sCopyMain ?: return false
        val scExtra = sCopyExtra ?: return false

        var res = true
        res = res && sc.ne[0] == ctx.getNRs().toLong()
        res = res && scMain.ne[0] == params.ubatch.nSeqs.toLong()
        res = res && scExtra.ne[0] == (ctx.getNRs() - params.ubatch.nSeqs).toLong()
        res = res && head == ctx.getHead()
        res = res && rsZ == ctx.getRsZ()
        return res
    }
}

// -- self-attention (no KV cache) -------------------------------------------

/** Input: KQ masks for self-attention without a persistent cache. */
class LlmGraphInputAttnNoCache(
    val hparams: LlamaHparams,
    val cparams: LlamaCParams,
) : LlmGraphInput {
    var selfKqMask: GGMLTensor? = null
    var selfKqMaskCnv: GGMLTensor? = null
    var selfKqMaskSwa: GGMLTensor? = null
    var selfKqMaskSwaCnv: GGMLTensor? = null

    /** Converted KQ mask ready for use by attention kernels. */
    fun getKqMask(): GGMLTensor? = selfKqMaskCnv

    /** Converted sliding-window KQ mask. */
    fun getKqMaskSwa(): GGMLTensor? = selfKqMaskSwaCnv

    override fun setInput(ubatch: LlamaUBatch) {
        val nKv = ubatch.nTokens
        val nTokens = ubatch.nTokens
        val posArr = ubatch.pos ?: return
        val seqIdArr = ubatch.seqId ?: return

        // Lambda to fill a mask tensor
        fun fillMask(tensor: GGMLTensor, nSwa: Int, swaType: LlamaSwaType) {
            val data = FloatArray(nTokens * nKv) { Float.NEGATIVE_INFINITY }

            for (i1 in 0 until nTokens) {
                val s1 = seqIdArr[i1][0]
                val p1 = posArr[i1]
                val idst = i1 * nKv

                for (i0 in 0 until nTokens) {
                    val s0 = seqIdArr[i0][0]
                    val p0 = posArr[i0]

                    // Mask different sequences
                    if (s0 != s1) continue
                    // Mask future tokens (causal)
                    if (cparams.causalAttn && p0 > p1) continue
                    // Apply SWA if any
                    if (LlamaHparams.isMaskedSwa(nSwa, swaType, p0, p1)) continue

                    data[idst + i0] = if (hparams.useAlibi) {
                        -(kotlin.math.abs(p0 - p1)).toFloat()
                    } else {
                        0.0f
                    }
                }
            }
            tensorSetF32(tensor, data, 0, data.size)
        }

        // Fill the main self-attention mask
        val mask = selfKqMask
        if (mask != null) {
            fillMask(mask, 0, LlamaSwaType.NONE)
        }

        // Fill the SWA mask if sliding-window attention is configured
        if (hparams.swaType != LlamaSwaType.NONE) {
            val swaMask = selfKqMaskSwa
            if (swaMask != null) {
                fillMask(swaMask, hparams.nSwa, hparams.swaType)
            }
        }
    }
}

// -- cross-attention --------------------------------------------------------

/** Input: KQ mask for encoder–decoder cross-attention. */
class LlmGraphInputAttnCross(val cross: LlamaCross?) : LlmGraphInput {
    var crossKqMask: GGMLTensor? = null
    var crossKqMaskCnv: GGMLTensor? = null

    /** Converted cross-attention KQ mask. */
    fun getKqMaskCross(): GGMLTensor? = crossKqMaskCnv

    override fun setInput(ubatch: LlamaUBatch) {
        val mask = crossKqMask ?: return
        val cr = cross ?: return

        val nEnc = mask.ne[0].toInt()
        val nTokens = ubatch.nTokens
        val seqIdArr = ubatch.seqId ?: return
        val nSeqIdArr = ubatch.nSeqId ?: return

        require(cr.seqIdsEnc.isNotEmpty()) { "llamaEncode must be called first" }
        require(!ubatch.equalSeqs) // TODO: use ubatch.nSeqs instead of failing

        val data = FloatArray(nTokens * nEnc)
        for (i in 0 until nTokens) {
            for (j in 0 until nEnc) {
                var f = Float.NEGATIVE_INFINITY

                for (s in 0 until nSeqIdArr[i]) {
                    val seqId = seqIdArr[i][s]
                    if (j < cr.seqIdsEnc.size && seqId in cr.seqIdsEnc[j]) {
                        f = 0.0f
                    }
                }

                data[i * nEnc + j] = f
            }
        }
        tensorSetF32(mask, data, 0, data.size)
    }
}

// ---------------------------------------------------------------------------
// LlmGraphQkv – packed Q/K/V triplet produced by build_qkv
// ---------------------------------------------------------------------------

/** Packed query, key, and value tensors ready for attention. */
data class LlmGraphQkv(
    /** [nEmbdHead, nHead, nTokens] */
    val q: GGMLTensor,
    /** [nEmbdHead, nHeadKv, nTokens] */
    val k: GGMLTensor,
    /** [nEmbdHead, nHeadKv, nTokens] */
    val v: GGMLTensor,
)

// ---------------------------------------------------------------------------
// LlmGraphResult – tensors produced by a single graph evaluation
// ---------------------------------------------------------------------------

/**
 * Collects every output tensor produced during graph construction and owns the
 * list of [LlmGraphInput]s that must be populated before evaluation.
 */
class LlmGraphResult(val maxNodes: Long = 8192) {
    var tInpTokens: GGMLTensor? = null
    /** [nEmbdInp, nTokens] */
    var tInpEmbd: GGMLTensor? = null
    var tLogits: GGMLTensor? = null
    var tEmbd: GGMLTensor? = null
    var tEmbdPooled: GGMLTensor? = null

    val tSampledLogits: MutableMap<LlamaSeqId, GGMLTensor> = mutableMapOf()
    val tCandidates: MutableMap<LlamaSeqId, GGMLTensor> = mutableMapOf()
    val tSampled: MutableMap<LlamaSeqId, GGMLTensor> = mutableMapOf()
    val tSampledProbs: MutableMap<LlamaSeqId, GGMLTensor> = mutableMapOf()

    val inputs: MutableList<LlmGraphInput> = mutableListOf()

    /** The underlying ggml context for tensor allocation. */
    var ctxCompute: GGMLContext? = null

    /** The compute graph. */
    var gf: GGMLCGraph? = null

    /** Register an input and return it (convenience for chaining). */
    fun addInput(input: LlmGraphInput): LlmGraphInput {
        inputs.add(input)
        return input
    }

    /** Typed variant that returns the concrete input type (avoids casts at call sites). */
    @Suppress("UNCHECKED_CAST")
    fun <T : LlmGraphInput> addInputTyped(input: T): T {
        inputs.add(input)
        return input
    }

    /** Populate every registered input from [ubatch]. */
    fun setInputs(ubatch: LlamaUBatch) {
        for (input in inputs) {
            input.setInput(ubatch)
        }
    }

    /** Mark output tensors so backends know to preserve them. */
    fun setOutputs() {
        tLogits?.let { ggmlSetOutput(it) }
        tEmbd?.let { ggmlSetOutput(it) }
        tEmbdPooled?.let { ggmlSetOutput(it) }
        for ((_, t) in tSampled) ggmlSetOutput(t)
        for ((_, t) in tSampledProbs) ggmlSetOutput(t)
        for ((_, t) in tSampledLogits) ggmlSetOutput(t)
        for ((_, t) in tCandidates) ggmlSetOutput(t)
    }

    /** `true` when all inputs report the graph can be reused with [params]. */
    fun canReuse(params: LlmGraphParams): Boolean {
        return inputs.all { it.canReuse(params) }
    }

    /** Clear all stored tensors and inputs so the result can be rebuilt. */
    fun reset() {
        tInpTokens = null
        tInpEmbd = null
        tLogits = null
        tEmbd = null
        tEmbdPooled = null
        tSampledLogits.clear()
        tCandidates.clear()
        tSampled.clear()
        tSampledProbs.clear()
        inputs.clear()
    }

    fun getCtx(): GGMLContext? = ctxCompute
    fun getGf(): GGMLCGraph? = gf
}

// ---------------------------------------------------------------------------
// LlmGraphParams – immutable snapshot of everything needed to build a graph
// ---------------------------------------------------------------------------

/** Parameters that fully describe a single graph-build invocation. */
data class LlmGraphParams(
    val arch: LlamaModelArch = LlamaModelArch.UNKNOWN,
    val hparams: LlamaHparams = LlamaHparams(),
    val cparams: LlamaCParams = LlamaCParams(),
    val ubatch: LlamaUBatch = LlamaUBatch(),
    val gtype: LlmGraphType = LlmGraphType.DEFAULT,
    val nOutputs: Int = 0,
    val res: LlmGraphResult? = null,

    // Adapter references
    val cvec: LlamaAdapterCvec? = null,
    val loras: LlamaAdapterLoras? = null,
    /** Memory context for KV cache / recurrent state. */
    val mctx: LlamaMemoryContext? = null,
    val cross: LlamaCross? = null,

    // Graph callback for tensor naming / offloading
    val cb: LlmGraphCb? = null,
)

// ---------------------------------------------------------------------------
// LlmGraphContext – base builder class for model-specific graph construction
// ---------------------------------------------------------------------------

/**
 * Base class for building inference graphs.
 *
 * Model-specific subclasses (e.g. `LlmBuildLlama`) override selected builder
 * methods to wire architecture-specific attention, FFN, and normalisation
 * patterns.  Shared utilities (LoRA application, KQ-mask construction, …) are
 * implemented once here.
 *
 * Port of `llm_graph_context` from `llama-graph.h` / `llama-graph.cpp`.
 */
open class LlmGraphContext(val params: LlmGraphParams) {

    // -- convenience accessors from params ----------------------------------

    val arch: LlamaModelArch get() = params.arch
    val hparams: LlamaHparams get() = params.hparams
    val cparams: LlamaCParams get() = params.cparams
    val ubatch: LlamaUBatch get() = params.ubatch

    val nEmbd: Long get() = hparams.nEmbd.toLong()
    val nLayer: Long get() = hparams.nLayer.toLong()
    val nRot: Long get() = hparams.nRot().toLong()
    val nCtx: Long get() = cparams.nCtx.toLong()
    val nHead: Long get() = hparams.nHead().toLong()
    val nHeadKv: Long get() = hparams.nHeadKv().toLong()
    val nEmbdHeadK: Long get() = hparams.nEmbdHeadK().toLong()
    val nEmbdKGqa: Long get() = hparams.nEmbdKGqa().toLong()
    val nEmbdHeadV: Long get() = hparams.nEmbdHeadV().toLong()
    val nEmbdVGqa: Long get() = hparams.nEmbdVGqa().toLong()
    val nExpert: Long get() = hparams.nExpert.toLong()
    val nExpertUsed: Long
        get() = if (cparams.warmup) hparams.nExpert.toLong() else hparams.nExpertUsed.toLong()

    val freqBase: Float get() = cparams.ropeFreqBase
    val freqScale: Float get() = cparams.ropeFreqScale
    val extFactor: Float get() = cparams.yarnExtFactor
    val attnFactor: Float get() = cparams.yarnAttnFactor
    val betaFast: Float get() = cparams.yarnBetaFast
    val betaSlow: Float get() = cparams.yarnBetaSlow
    val normEps: Float get() = hparams.fNormEps
    val normRmsEps: Float get() = hparams.fNormRmsEps

    val nTokens: Long get() = ubatch.nTokens.toLong()
    val nOutputs: Long get() = params.nOutputs.toLong()
    val nCtxOrig: Int get() = cparams.nCtxOrigYarn

    val poolingType: LlamaPoolingType get() = cparams.poolingType
    val ropeType: LlamaRopeType get() = hparams.ropeType

    val cvec: LlamaAdapterCvec? get() = params.cvec
    val loras: LlamaAdapterLoras? get() = params.loras
    val cross: LlamaCross? get() = params.cross
    val cbFunc: LlmGraphCb? get() = params.cb

    var res: LlmGraphResult? = params.res
    var ctx0: GGMLContext? = res?.getCtx()
    var gf: GGMLCGraph? = res?.getGf()

    // -- tensor callback helper ---------------------------------------------

    /** Name a tensor and invoke the user callback. */
    fun cb(cur: GGMLTensor, name: String, il: Int) {
        cbFunc?.invoke(ubatch, cur, name, il)
    }

}
