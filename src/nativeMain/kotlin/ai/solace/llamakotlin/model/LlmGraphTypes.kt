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
        TODO("Port llm_graph_input_embd::set_input — backend tensor set")
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
        TODO("Port llm_graph_input_pos::set_input — backend tensor set")
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        val p = pos ?: return false
        return p.ne[0] == params.ubatch.nTokens.toLong() * nPosPerEmbd
    }
}

// -- output ids -------------------------------------------------------------

/** Input: indices of tokens whose logits are actually needed. */
class LlmGraphInputOutIds(
    val hparams: LlamaModelHParams,
    val cparams: LlamaCParams,
    val nOutputs: Int,
) : LlmGraphInput {
    /** I32 [nOutputs] – indices into the batch that require output. */
    var outIds: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_out_ids::set_input — backend tensor set")
    }

    override fun canReuse(params: LlmGraphParams): Boolean = nOutputs == params.nOutputs
}

// -- mean pooling -----------------------------------------------------------

/** Input: mean-pooling weight matrix. */
class LlmGraphInputMean(val cparams: LlamaCParams) : LlmGraphInput {
    /** F32 [nBatch, nBatch] – mean-pooling weights. */
    var mean: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_mean::set_input — backend tensor set")
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
        TODO("Port llm_graph_input_cls::set_input — backend tensor set")
    }
}

// -- recurrent state --------------------------------------------------------

/** Input: state-copy bookkeeping for recurrent (SSM / RWKV) models. */
class LlmGraphInputRs : LlmGraphInput {
    /** I32 [nRs] – full state-copy source indices. */
    var sCopy: GGMLTensor? = null
    /** I32 [nSeqs] – state-copy indices for the main sequences. */
    var sCopyMain: GGMLTensor? = null
    /** I32 [nRs - nSeqs] – state-copy indices for extra sequences. */
    var sCopyExtra: GGMLTensor? = null
    var head: Int = 0
    var rsZ: Int = 0

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_rs::set_input — backend tensor set")
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        TODO("Port llm_graph_input_rs::can_reuse — requires recurrent memory context")
    }
}

// -- self-attention (no KV cache) -------------------------------------------

/** Input: KQ masks for self-attention without a persistent cache. */
class LlmGraphInputAttnNoCache(
    val hparams: LlamaModelHParams,
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
        TODO("Port llm_graph_input_attn_no_cache::set_input — fill mask data")
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
        TODO("Port llm_graph_input_attn_cross::set_input — fill cross mask")
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
    val hparams: LlamaModelHParams = LlamaModelHParams(),
    val cparams: LlamaCParams = LlamaCParams(),
    val ubatch: LlamaUBatch = LlamaUBatch(),
    val gtype: LlmGraphType = LlmGraphType.DEFAULT,
    val nOutputs: Int = 0,
    val res: LlmGraphResult? = null,

    // Adapter references
    val cvec: LlamaAdapterCvec? = null,
    val loras: LlamaAdapterLoras? = null,
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
    val hparams: LlamaModelHParams get() = params.hparams
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

    // =======================================================================
    // Common builders
    // =======================================================================

    /** Apply control vector to the hidden state at layer [il]. */
    open fun buildCvec(cur: GGMLTensor, il: Int): GGMLTensor {
        val cv = cvec ?: return cur
        return cv.applyTo(cur, il)
    }

    /**
     * Matrix-multiply [cur] by weight [w] with optional LoRA deltas and
     * per-tensor scale [wS].
     *
     * Port of `llm_graph_context::build_lora_mm`.
     */
    open fun buildLoraMm(w: GGMLTensor, cur: GGMLTensor, wS: GGMLTensor? = null): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        var result = matMul(c, w, cur)

        loras?.let { loraMap ->
            for ((adapter, adapterScale) in loraMap) {
                val lw = adapter.getWeight(w) ?: continue
                val scale = lw.getScale(adapter.alpha, adapterScale)

                var abCur = matMul(c, lw.b!!, matMul(c, lw.a!!, cur))
                abCur = ggmlScale(c, abCur, scale)
                result = add(c, result, abCur)
            }
        }

        if (wS != null) {
            result = mul(c, result, wS)
        }
        return result
    }

    /**
     * Expert-routed LoRA matrix-multiply.
     *
     * Port of `llm_graph_context::build_lora_mm_id`.
     */
    open fun buildLoraMmId(w: GGMLTensor, cur: GGMLTensor, ids: GGMLTensor): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        var result = ggmlMulMatId(c, w, cur, ids)

        loras?.let { loraMap ->
            for ((adapter, adapterScale) in loraMap) {
                val lw = adapter.getWeight(w) ?: continue
                val alpha = adapter.alpha
                val rank = lw.b!!.ne[0].toFloat()
                val scale = if (alpha != 0f) adapterScale * alpha / rank else adapterScale

                var abCur = ggmlMulMatId(c, lw.b!!, ggmlMulMatId(c, lw.a!!, cur, ids), ids)
                abCur = ggmlScale(c, abCur, scale)
                result = add(c, result, abCur)
            }
        }
        return result
    }

    /**
     * Apply normalisation of the given [type] with optional weight [mw] and bias [mb].
     *
     * Port of `llm_graph_context::build_norm`.
     */
    open fun buildNorm(
        cur: GGMLTensor,
        mw: GGMLTensor?,
        mb: GGMLTensor?,
        type: LlmNormType,
        il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        var out = cur
        when (type) {
            LlmNormType.NORM -> {
                out = ggmlNorm(c, out, hparams.fNormEps)
            }
            LlmNormType.RMS -> {
                out = ggmlRmsNorm(c, out, hparams.fNormRmsEps)
            }
            LlmNormType.GROUP -> {
                out = ggmlReshape3d(c, out, out.ne[0], 1, out.ne[1])
                out = ggmlGroupNorm(c, out, hparams.nNormGroups, hparams.fNormGroupEps)
                out = ggmlReshape2d(c, out, out.ne[0], out.ne[2])
            }
        }

        if (mw != null || mb != null) {
            cb(out, "norm", il)
        }

        if (mw != null) {
            out = mul(c, out, mw)
            if (mb != null) {
                cb(out, "norm_w", il)
            }
        }

        if (mb != null) {
            out = add(c, out, mb)
        }

        return out
    }

    /**
     * Compute Q, K, V projections with optional bias and reshape.
     * Supports both fused wqkv and separate wq/wk/wv paths.
     *
     * Port of `llm_graph_context::build_qkv`.
     */
    open fun buildQkv(
        layer: LlamaModelLayer,
        cur: GGMLTensor,
        nEmbdHead: Long,
        nHead: Long,
        nHeadKv: Long,
        il: Int,
    ): LlmGraphQkv {
        val c = ctx0 ?: error("ctx0 not initialised")
        val nEmbdQ = nEmbdHead * nHead
        val nEmbdKv = nEmbdHead * nHeadKv

        var qCur: GGMLTensor
        var kCur: GGMLTensor
        var vCur: GGMLTensor

        if (layer.wqkv != null) {
            // fused QKV path
            var qkv = buildLoraMm(layer.wqkv!!, cur)
            cb(qkv, "wqkv", il)
            if (layer.bqkv != null) {
                qkv = add(c, qkv, layer.bqkv!!)
                cb(qkv, "bqkv", il)
            }
            if (hparams.fClampKqv > 0.0f) {
                qkv = ggmlClamp(c, qkv, -hparams.fClampKqv, hparams.fClampKqv)
                cb(qkv, "wqkv_clamped", il)
            }
            val rowSize = ggmlRowSize(qkv.type, nEmbdHead)
            qCur = ggmlView3d(c, qkv, nEmbdHead, nHead, nTokens, rowSize, qkv.nb[1], 0u)
            kCur = ggmlView3d(c, qkv, nEmbdHead, nHeadKv, nTokens,
                rowSize, qkv.nb[1], ggmlRowSize(qkv.type, nEmbdQ))
            vCur = ggmlView3d(c, qkv, nEmbdHead, nHeadKv, nTokens,
                rowSize, qkv.nb[1], ggmlRowSize(qkv.type, nEmbdQ + nEmbdKv))
        } else {
            // separate Q/K/V path
            qCur = buildLoraMm(layer.wq!!, cur)
            cb(qCur, "Qcur", il)
            if (layer.bq != null) {
                qCur = add(c, qCur, layer.bq!!)
                cb(qCur, "Qcur", il)
            }
            if (hparams.fClampKqv > 0.0f) {
                qCur = ggmlClamp(c, qCur, -hparams.fClampKqv, hparams.fClampKqv)
                cb(qCur, "Qcur_clamped", il)
            }

            kCur = buildLoraMm(layer.wk!!, cur)
            cb(kCur, "Kcur", il)
            if (layer.bk != null) {
                kCur = add(c, kCur, layer.bk!!)
                cb(kCur, "Kcur", il)
            }
            if (hparams.fClampKqv > 0.0f) {
                kCur = ggmlClamp(c, kCur, -hparams.fClampKqv, hparams.fClampKqv)
                cb(kCur, "Kcur_clamped", il)
            }

            vCur = buildLoraMm(layer.wv!!, cur)
            cb(vCur, "Vcur", il)
            if (layer.bv != null) {
                vCur = add(c, vCur, layer.bv!!)
                cb(vCur, "Vcur", il)
            }
            if (hparams.fClampKqv > 0.0f) {
                vCur = ggmlClamp(c, vCur, -hparams.fClampKqv, hparams.fClampKqv)
                cb(vCur, "Vcur_clamped", il)
            }

            qCur = ggmlReshape3d(c, qCur, nEmbdHead, nHead, nTokens)
            kCur = ggmlReshape3d(c, kCur, nEmbdHead, nHeadKv, nTokens)
            vCur = ggmlReshape3d(c, vCur, nEmbdHead, nHeadKv, nTokens)
        }

        cb(qCur, "Qcur", il)
        cb(kCur, "Kcur", il)
        cb(vCur, "Vcur", il)

        return LlmGraphQkv(qCur, kCur, vCur)
    }

    /**
     * Construct a feed-forward network sub-graph at layer [il].
     *
     * Port of `llm_graph_context::build_ffn`.
     */
    open fun buildFfn(
        cur: GGMLTensor,
        up: GGMLTensor?, upB: GGMLTensor?, upS: GGMLTensor?,
        gate: GGMLTensor?, gateB: GGMLTensor?, gateS: GGMLTensor?,
        down: GGMLTensor?, downB: GGMLTensor?, downS: GGMLTensor?,
        actScales: GGMLTensor?,
        typeOp: LlmFfnOpType,
        typeGate: LlmFfnGateType,
        il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        var typeGateMut = typeGate

        var tmp = if (up != null) buildLoraMm(up, cur) else cur
        cb(tmp, "ffn_up", il)

        if (upB != null) { tmp = add(c, tmp, upB); cb(tmp, "ffn_up_b", il) }
        if (upS != null) { tmp = mul(c, tmp, upS); cb(tmp, "ffn_up_s", il) }

        var out: GGMLTensor
        if (gate != null) {
            out = when (typeGateMut) {
                LlmFfnGateType.SEQ -> buildLoraMm(gate, tmp)
                LlmFfnGateType.PAR -> buildLoraMm(gate, cur)
            }
            cb(out, "ffn_gate", il)

            if (gateB != null) { out = add(c, out, gateB); cb(out, "ffn_gate_b", il) }
            if (gateS != null) { out = mul(c, out, gateS); cb(out, "ffn_gate_s", il) }
        } else {
            out = tmp
        }

        when (typeOp) {
            LlmFfnOpType.SILU -> {
                if (gate != null && typeGateMut == LlmFfnGateType.PAR) {
                    out = ggmlSwigluSplit(c, out, tmp)
                    cb(out, "ffn_swiglu", il)
                    typeGateMut = LlmFfnGateType.SEQ
                } else {
                    out = ggmlSilu(c, out)
                    cb(out, "ffn_silu", il)
                }
            }
            LlmFfnOpType.GELU -> {
                if (gate != null && typeGateMut == LlmFfnGateType.PAR) {
                    out = ggmlGegluSplit(c, out, tmp)
                    cb(out, "ffn_geglu", il)
                    typeGateMut = LlmFfnGateType.SEQ
                } else {
                    out = gelu(c, out)
                    cb(out, "ffn_gelu", il)
                    if (actScales != null) {
                        out = div(c, out, actScales)
                        cb(out, "ffn_act", il)
                    }
                }
            }
            LlmFfnOpType.RELU -> {
                if (gate != null && typeGateMut == LlmFfnGateType.PAR) {
                    out = ggmlRegluSplit(c, out, tmp)
                    cb(out, "ffn_reglu", il)
                    typeGateMut = LlmFfnGateType.SEQ
                } else {
                    out = relu(c, out)
                    cb(out, "ffn_relu", il)
                }
            }
            LlmFfnOpType.RELU_SQR -> {
                out = relu(c, out)
                cb(out, "ffn_relu", il)
                out = ggmlSqr(c, out)
                cb(out, "ffn_sqr(relu)", il)
            }
            LlmFfnOpType.SWIGLU -> {
                out = ggmlSwiglu(c, out)
                cb(out, "ffn_swiglu", il)
            }
            LlmFfnOpType.GEGLU -> {
                out = ggmlGeglu(c, out)
                cb(out, "ffn_geglu", il)
            }
            LlmFfnOpType.REGLU -> {
                out = ggmlReglu(c, out)
                cb(out, "ffn_reglu", il)
            }
            else -> error("Unsupported FFN op type: $typeOp")
        }

        if (gate != null && typeGateMut == LlmFfnGateType.PAR) {
            out = mul(c, out, tmp)
            cb(out, "ffn_gate_par", il)
        }

        if (down != null) {
            out = buildLoraMm(down, out)
        }
        if (downB != null) {
            cb(out, "ffn_down", il)
            out = add(c, out, downB)
        }
        if (downS != null) {
            out = mul(c, out, downS)
            cb(out, "ffn_down_s", il)
        }

        return out
    }

    /**
     * Construct a Mixture-of-Experts FFN sub-graph at layer [il].
     *
     * Port of `llm_graph_context::build_moe_ffn` (no-bias overload delegates
     * to the full overload with null bias tensors).
     */
    open fun buildMoeFfn(
        cur: GGMLTensor,
        gateInp: GGMLTensor,
        upExps: GGMLTensor,
        gateExps: GGMLTensor?,
        downExps: GGMLTensor,
        expProbsB: GGMLTensor?,
        nExpert: Long,
        nExpertUsed: Long,
        typeOp: LlmFfnOpType,
        normW: Boolean,
        wScale: Float,
        gatingOp: LlamaExpertGatingFuncType,
        il: Int,
    ): GGMLTensor {
        return buildMoeFfnFull(
            cur, gateInp, null, upExps, null, gateExps, null, downExps, null,
            expProbsB, nExpert, nExpertUsed, typeOp, normW, wScale, gatingOp, il,
        )
    }

    /**
     * Full MoE FFN with optional bias tensors on every expert weight.
     *
     * Port of the second `llm_graph_context::build_moe_ffn` overload.
     */
    open fun buildMoeFfnFull(
        cur: GGMLTensor,
        gateInp: GGMLTensor,
        gateInpB: GGMLTensor?,
        upExps: GGMLTensor,
        upExpsB: GGMLTensor?,
        gateExps: GGMLTensor?,
        gateExpsB: GGMLTensor?,
        downExps: GGMLTensor,
        downExpsB: GGMLTensor?,
        expProbsB: GGMLTensor?,
        nExpert: Long,
        nExpertUsed: Long,
        typeOp: LlmFfnOpType,
        normW: Boolean,
        wScale: Float,
        gatingOp: LlamaExpertGatingFuncType,
        il: Int,
        probsIn: GGMLTensor? = null,
        gateUpExps: GGMLTensor? = null,
        gateUpExpsB: GGMLTensor? = null,
        upExpsS: GGMLTensor? = null,
        gateExpsS: GGMLTensor? = null,
        downExpsS: GGMLTensor? = null,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")
        val nEmbdCur = cur.ne[0]
        val nTokensCur = cur.ne[1]

        // -- gating logits --------------------------------------------------
        var logits = if (probsIn == null) {
            val l = buildLoraMm(gateInp, cur)
            cb(l, "ffn_moe_logits", il)
            l
        } else {
            probsIn
        }

        if (gateInpB != null) {
            logits = add(c, logits, gateInpB)
            cb(logits, "ffn_moe_logits_biased", il)
        }

        var probs: GGMLTensor = when (gatingOp) {
            LlamaExpertGatingFuncType.SOFTMAX -> ggmlSoftMax(c, logits)
            LlamaExpertGatingFuncType.SIGMOID -> ggmlSigmoid(c, logits)
            LlamaExpertGatingFuncType.SOFTMAX_WEIGHT -> logits
            else -> error("Unsupported gating op: $gatingOp")
        }
        cb(probs, "ffn_moe_probs", il)

        var selectionProbs = probs
        if (expProbsB != null) {
            selectionProbs = add(c, probs, expProbsB)
            cb(selectionProbs, "ffn_moe_probs_biased", il)
        }

        if (arch == LlamaModelArch.LLAMA4) {
            selectionProbs = logits
        }

        // -- top-k expert selection -----------------------------------------
        val selectedExperts = ggmlArgsortTopK(c, selectionProbs, nExpertUsed)
        cb(selectedExperts, "ffn_moe_topk", il)

        probs = ggmlReshape3d(c, probs, 1, nExpert, nTokensCur)

        var weights = ggmlGetRows(c, probs, selectedExperts) // [1, nExpertUsed, nTokens]
        cb(weights, "ffn_moe_weights", il)

        if (gatingOp == LlamaExpertGatingFuncType.SOFTMAX_WEIGHT) {
            weights = ggmlReshape2d(c, weights, nExpertUsed, nTokensCur)
            weights = ggmlSoftMax(c, weights)
            weights = ggmlReshape3d(c, weights, 1, nExpertUsed, nTokensCur)
            cb(weights, "ffn_moe_weights_softmax", il)
        }

        if (normW) {
            weights = ggmlReshape2d(c, weights, nExpertUsed, nTokensCur)
            var weightsSum = ggmlSumRows(c, weights)
            cb(weightsSum, "ffn_moe_weights_sum", il)
            weightsSum = ggmlClamp(c, weightsSum, 6.103515625e-5f, Float.POSITIVE_INFINITY)
            cb(weightsSum, "ffn_moe_weights_sum_clamped", il)
            weights = div(c, weights, weightsSum)
            cb(weights, "ffn_moe_weights_norm", il)
            weights = ggmlReshape3d(c, weights, 1, nExpertUsed, nTokensCur)
        }
        if (wScale != 0.0f && wScale != 1.0f) {
            weights = ggmlScale(c, weights, wScale)
            cb(weights, "ffn_moe_weights_scaled", il)
        }

        ggmlBuildForwardExpand(graph, weights)

        var moeIn = ggmlReshape3d(c, cur, nEmbdCur, 1, nTokensCur)

        // -- up / gate projections ------------------------------------------
        var up: GGMLTensor
        if (gateUpExps != null) {
            var gateUp = buildLoraMmId(gateUpExps, moeIn, selectedExperts)
            cb(gateUp, "ffn_moe_gate_up", il)
            if (gateUpExpsB != null) {
                gateUp = ggmlAddId(c, gateUp, gateUpExpsB, selectedExperts)
                cb(gateUp, "ffn_moe_gate_up_biased", il)
            }
            val nFf = gateUp.ne[0] / 2
            moeIn = ggmlView3d(c, gateUp, nFf, gateUp.ne[1], gateUp.ne[2],
                gateUp.nb[1], gateUp.nb[2], 0u)
            cb(moeIn, "ffn_moe_gate", il)
            up = ggmlView3d(c, gateUp, nFf, gateUp.ne[1], gateUp.ne[2],
                gateUp.nb[1], gateUp.nb[2], (nFf.toULong() * gateUp.nb[0]))
            cb(up, "ffn_moe_up", il)
        } else {
            up = buildLoraMmId(upExps, moeIn, selectedExperts)
            cb(up, "ffn_moe_up", il)
            if (upExpsB != null) {
                up = ggmlAddId(c, up, upExpsB, selectedExperts)
                cb(up, "ffn_moe_up_biased", il)
            }
            if (gateExps != null) {
                moeIn = buildLoraMmId(gateExps, moeIn, selectedExperts)
                cb(moeIn, "ffn_moe_gate", il)
            } else {
                moeIn = up
            }
            if (gateExpsB != null) {
                moeIn = ggmlAddId(c, moeIn, gateExpsB, selectedExperts)
                cb(moeIn, "ffn_moe_gate_biased", il)
            }
        }

        val hasGate = gateExps != null || gateUpExps != null

        // -- activation -----------------------------------------------------
        var activated: GGMLTensor = when (typeOp) {
            LlmFfnOpType.SILU -> {
                if (hasGate) {
                    val r = ggmlSwigluSplit(c, moeIn, up); cb(r, "ffn_moe_swiglu", il); r
                } else {
                    val r = ggmlSilu(c, moeIn); cb(r, "ffn_moe_silu", il); r
                }
            }
            LlmFfnOpType.GELU -> {
                if (hasGate) {
                    val r = ggmlGegluSplit(c, moeIn, up); cb(r, "ffn_moe_geglu", il); r
                } else {
                    val r = gelu(c, moeIn); cb(r, "ffn_moe_gelu", il); r
                }
            }
            LlmFfnOpType.SWIGLU_OAI_MOE -> {
                val r = ggmlSwigluOai(c, moeIn, up, 1.702f, 7.0f)
                cb(r, "ffn_moe_swiglu_oai", il); r
            }
            LlmFfnOpType.RELU -> {
                if (hasGate) {
                    val r = ggmlRegluSplit(c, moeIn, up); cb(r, "ffn_moe_reglu", il); r
                } else {
                    val r = relu(c, moeIn); cb(r, "ffn_moe_relu", il); r
                }
            }
            LlmFfnOpType.RELU_SQR -> {
                var r = relu(c, moeIn); r = ggmlSqr(c, r)
                cb(r, "ffn_moe_relu_sqr", il); r
            }
            else -> error("Unsupported MoE FFN op type: $typeOp")
        }

        // -- down projection ------------------------------------------------
        var experts = buildLoraMmId(downExps, activated, selectedExperts)
        cb(experts, "ffn_moe_down", il)

        if (downExpsB != null) {
            experts = ggmlAddId(c, experts, downExpsB, selectedExperts)
            cb(experts, "ffn_moe_down_biased", il)
        }

        experts = mul(c, experts, weights)
        cb(experts, "ffn_moe_weighted", il)

        ggmlBuildForwardExpand(graph, experts)

        // aggregate expert views
        val nExpertUsedInt = hparams.nExpertUsed
        val curExperts = Array<GGMLTensor?>(nExpertUsedInt) { null }
        for (i in 0 until nExpertUsedInt) {
            curExperts[i] = ggmlView2d(c, experts, nEmbdCur, nTokensCur,
                experts.nb[2], (i.toLong().toULong() * experts.nb[1]))
            ggmlBuildForwardExpand(graph, curExperts[i]!!)
        }

        var moeOut = curExperts[0]!!
        for (i in 1 until nExpertUsedInt) {
            moeOut = add(c, moeOut, curExperts[i]!!)
            ggmlBuildForwardExpand(graph, moeOut)
        }

        if (nExpertUsedInt == 1) {
            moeOut = ggmlCont(c, moeOut)
        }

        cb(moeOut, "ffn_moe_out", il)
        return moeOut
    }

    // =======================================================================
    // Input builders
    // =======================================================================

    /**
     * Create input tensors for token-id / embedding injection.
     *
     * Port of `llm_graph_context::build_inp_embd`.
     */
    open fun buildInpEmbd(tokEmbd: GGMLTensor): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val nEmbdInp = hparams.nEmbdInp().toLong()

        val inp = LlmGraphInputEmbd(nEmbdInp)
        inp.tokens = ggmlNewTensor1d(c, GGMLType.I32, nTokens)
        cb(inp.tokens!!, "inp_tokens", -1)
        ggmlSetInput(inp.tokens!!)
        r.tInpTokens = inp.tokens

        inp.embd = ggmlNewTensor2d(c, GGMLType.F32, nEmbdInp, nTokens)
        cb(inp.embd!!, "inp_embd", -1)
        ggmlSetInput(inp.embd!!)

        // Token embeddings path: look up rows in tok_embd
        var tokPath = ggmlGetRows(c, tokEmbd, inp.tokens!!)

        // For Granite architecture
        if (hparams.fEmbeddingScale != 0.0f) {
            tokPath = ggmlScale(c, tokPath, hparams.fEmbeddingScale)
        }

        cb(tokPath, "embd", -1)

        r.tInpEmbd = tokPath
        r.addInput(inp)

        val graph = gf ?: error("gf not initialised")
        ggmlBuildForwardExpand(graph, tokPath)

        return tokPath
    }

    /** Create the position-id input tensor. Port of `build_inp_pos`. */
    open fun buildInpPos(): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputPos(hparams.nPosPerEmbd())
        inp.pos = ggmlNewTensor1d(c, GGMLType.I32, nTokens * hparams.nPosPerEmbd())
        ggmlSetInput(inp.pos!!)
        r.addInput(inp)
        return inp.pos!!
    }

    /** Create the output-id selection tensor. Port of `build_inp_out_ids`. */
    open fun buildInpOutIds(): GGMLTensor? {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputOutIds(hparams, cparams, nOutputs.toInt())
        inp.outIds = ggmlNewTensor1d(c, GGMLType.I32, nOutputs)
        ggmlSetInput(inp.outIds!!)
        r.addInput(inp)
        return inp.outIds
    }

    // =======================================================================
    // Attention builders
    // =======================================================================

    /**
     * Core multi-head attention sub-graph.
     *
     * Port of `llm_graph_context::build_attn_mha`.
     */
    open fun buildAttnMha(
        q: GGMLTensor, k: GGMLTensor, v: GGMLTensor,
        kqB: GGMLTensor?, kqMask: GGMLTensor?, sinks: GGMLTensor?,
        vMla: GGMLTensor?, kqScale: Float, il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")
        val nStream = k.ne[3]

        val q4 = ggmlView4d(c, q, q.ne[0], q.ne[1], q.ne[2] / nStream, nStream,
            q.nb[1], q.nb[2], q.nb[3] / nStream.toULong(), 0u)
        val qPerm = ggmlPermute(c, q4, 0, 2, 1, 3)
        val kPerm = ggmlPermute(c, k, 0, 2, 1, 3)
        val vPerm = ggmlPermute(c, v, 0, 2, 1, 3)

        val useFlashAttn = cparams.flashAttn && kqB == null
        val cur: GGMLTensor

        if (useFlashAttn) {
            var kFa = kPerm
            var vFa = vPerm
            if (kFa.type == GGMLType.F32) kFa = ggmlCast(c, kFa, GGMLType.F16)
            if (vFa.type == GGMLType.F32) vFa = ggmlCast(c, vFa, GGMLType.F16)

            var fa = ggmlFlashAttnExt(c, qPerm, kFa, vFa, kqMask, kqScale,
                hparams.fMaxAlibiBias,
                if (hparams.attnSoftCap) hparams.fAttnLogitSoftcapping else 0.0f)
            cb(fa, "fattn", il)
            ggmlFlashAttnExtAddSinks(fa, sinks)
            ggmlFlashAttnExtSetPrec(fa, GGMLPrec.F32)

            if (vMla != null) {
                fa = ggmlPermute(c, fa, 0, 2, 1, 3)
                fa = matMul(c, vMla, fa)
                cb(fa, "fattn_mla", il)
                fa = ggmlPermute(c, fa, 0, 2, 1, 3)
                fa = ggmlCont(c, fa)
            }

            cur = ggmlReshape2d(c, fa, fa.ne[0] * fa.ne[1], fa.ne[2] * fa.ne[3])
        } else {
            var kq = matMul(c, kPerm, qPerm)
            cb(kq, "kq", il)
            ggmlMulMatSetPrec(kq, GGMLPrec.F32)

            if (hparams.attnSoftCap) {
                kq = ggmlScale(c, kq, 1.0f / hparams.fAttnLogitSoftcapping)
                cb(kq, "kq_scaled_1", il)
                kq = ggmlTanh(c, kq)
                cb(kq, "kq_tanh", il)
                kq = ggmlScale(c, kq, hparams.fAttnLogitSoftcapping)
                cb(kq, "kq_scaled_2", il)
            }

            if (kqB != null) {
                kq = add(c, kq, kqB)
                cb(kq, "kq_plus_kq_b", il)
            }

            kq = ggmlSoftMaxExt(c, kq, kqMask, kqScale, hparams.fMaxAlibiBias)
            ggmlSoftMaxAddSinks(kq, sinks)
            cb(kq, "kq_soft_max", il)

            var vT = ggmlCont(c, ggmlTranspose(c, vPerm))
            cb(vT, "v_cont", il)

            var kqv = matMul(c, vT, kq)
            cb(kqv, "kqv", il)

            if (vMla != null) {
                kqv = matMul(c, vMla, kqv)
                cb(kqv, "kqv_mla", il)
            }

            cur = ggmlCont2d(c, ggmlPermute(c, kqv, 0, 2, 1, 3),
                kqv.ne[0] * kqv.ne[1], kqv.ne[2] * kqv.ne[3])
        }

        ggmlBuildForwardExpand(graph, cur)
        return cur
    }

    /** Build attention inputs for the no-cache path. Port of `build_attn_inp_no_cache`. */
    open fun buildAttnInpNoCache(): LlmGraphInputAttnNoCache {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputAttnNoCache(hparams, cparams)

        inp.selfKqMask = ggmlNewTensor4d(c, GGMLType.F32, nTokens, nTokens, 1, 1)
        ggmlSetInput(inp.selfKqMask!!)
        inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMask!!, GGMLType.F16)
            else inp.selfKqMask

        if (hparams.swaType != LlamaSwaType.NONE) {
            inp.selfKqMaskSwa = ggmlNewTensor4d(c, GGMLType.F32, nTokens, nTokens, 1, 1)
            ggmlSetInput(inp.selfKqMaskSwa!!)
            inp.selfKqMaskSwaCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMaskSwa!!, GGMLType.F16)
                else inp.selfKqMaskSwa
        }

        return r.addInputTyped(inp)
    }

    /**
     * Self-attention without a persistent KV cache.
     *
     * Port of `llm_graph_context::build_attn(llm_graph_input_attn_no_cache *)`.
     */
    open fun buildAttn(
        inp: LlmGraphInputAttnNoCache,
        wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
        qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
        kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
        kqScale: Float, il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")

        ggmlBuildForwardExpand(graph, qCur)
        ggmlBuildForwardExpand(graph, kCur)
        ggmlBuildForwardExpand(graph, vCur)

        val isSwa = hparams.isSwa(il)
        val kqMask = if (isSwa) inp.getKqMaskSwa() else inp.getKqMask()

        var cur = buildAttnMha(qCur, kCur, vCur, kqB, kqMask, sinks, vMla, kqScale, il)
        cb(cur, "kqv_out", il)

        if (wo != null) {
            cur = buildLoraMm(wo, cur, woS)
        }
        if (woB != null) {
            cur = add(c, cur, woB)
        }
        return cur
    }

    /** Build attention inputs for the KV-cache path. Port of `build_attn_inp_kv`. */
    open fun buildAttnInpKv(): LlmGraphInputAttnKv {
        TODO("Port llm_graph_context::build_attn_inp_kv — requires KV cache context")
    }

    /**
     * Self-attention with a persistent KV cache.
     *
     * Port of `llm_graph_context::build_attn(llm_graph_input_attn_kv *)`.
     */
    open fun buildAttn(
        inp: LlmGraphInputAttnKv,
        wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
        qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
        kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
        kqScale: Float, il: Int,
    ): GGMLTensor {
        TODO("Port llm_graph_context::build_attn(kv) — requires KV cache memory ops")
    }
}
