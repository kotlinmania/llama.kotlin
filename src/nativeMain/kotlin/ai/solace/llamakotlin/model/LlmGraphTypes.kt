// port-lint: source llama.cpp/src/llama-graph.h
/**
 * Core graph types ported from `llama-graph.h`.
 *
 * These types are referenced by both [InferencePipeline] and [LlamaGraph] for
 * building and executing computation graphs during inference.  Enums, the
 * [LlmGraphInput] interface, concrete input implementations, result / parameter
 * containers, and the base [LlmGraphContext] builder class all live here.
 */
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

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
        TODO("Port llm_graph_input_embd::set_input")
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        TODO("Port llm_graph_input_embd::can_reuse")
    }
}

// -- positions --------------------------------------------------------------

/** Input: per-token position ids. */
class LlmGraphInputPos(val nPosPerEmbd: Int = 1) : LlmGraphInput {
    /** I32 [nBatch] – absolute position for each token. */
    var pos: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_pos::set_input")
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        TODO("Port llm_graph_input_pos::can_reuse")
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
        TODO("Port llm_graph_input_out_ids::set_input")
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        TODO("Port llm_graph_input_out_ids::can_reuse")
    }
}

// -- mean pooling -----------------------------------------------------------

/** Input: mean-pooling weight matrix. */
class LlmGraphInputMean(val cparams: LlamaCParams) : LlmGraphInput {
    /** F32 [nBatch, nBatch] – mean-pooling weights. */
    var mean: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_mean::set_input")
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
        TODO("Port llm_graph_input_cls::set_input")
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
        TODO("Port llm_graph_input_rs::set_input")
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        TODO("Port llm_graph_input_rs::can_reuse")
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
        TODO("Port llm_graph_input_attn_no_cache::set_input")
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
        TODO("Port llm_graph_input_attn_cross::set_input")
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

    /** Register an input and return it (convenience for chaining). */
    fun addInput(input: LlmGraphInput): LlmGraphInput {
        inputs.add(input)
        return input
    }

    /** Populate every registered input from [ubatch]. */
    fun setInputs(ubatch: LlamaUBatch) {
        for (input in inputs) {
            input.setInput(ubatch)
        }
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
 */
open class LlmGraphContext(val params: LlmGraphParams) {

    // -- convenience accessors from params ----------------------------------

    val hparams: LlamaModelHParams get() = params.hparams
    val cparams: LlamaCParams get() = params.cparams

    val nEmbd: Long get() = hparams.nEmbd.toLong()
    val nLayer: Long get() = hparams.nLayer.toLong()
    val nHead: Long get() = hparams.nHead().toLong()
    val nHeadKv: Long get() = hparams.nHeadKv().toLong()
    val nExpert: Long get() = hparams.nExpert.toLong()
    val nExpertUsed: Long get() = hparams.nExpertUsed.toLong()

    var res: LlmGraphResult? = null

    // -- layer-level builders -----------------------------------------------

    /** Apply control vector to the hidden state at layer [il]. */
    open fun buildCvec(cur: GGMLTensor, il: Int): GGMLTensor {
        TODO("Port llm_graph_context::build_cvec")
    }

    /** Multiply [cur] by weight [w] with optional LoRA side-weights [wS]. */
    open fun buildLoraMm(w: GGMLTensor, cur: GGMLTensor, wS: GGMLTensor? = null): GGMLTensor {
        TODO("Port llm_graph_context::build_lora_mm")
    }

    /** LoRA matmul routed through expert [ids]. */
    open fun buildLoraMmId(w: GGMLTensor, cur: GGMLTensor, ids: GGMLTensor): GGMLTensor {
        TODO("Port llm_graph_context::build_lora_mm_id")
    }

    /** Apply normalisation of the given [type] with optional weight [mw] and bias [mb]. */
    open fun buildNorm(
        cur: GGMLTensor,
        mw: GGMLTensor?,
        mb: GGMLTensor?,
        type: LlmNormType,
        il: Int,
    ): GGMLTensor {
        TODO("Port llm_graph_context::build_norm")
    }

    /** Build Q/K/V projections for a single layer. */
    open fun buildQkv(
        layer: LlamaModelLayer,
        cur: GGMLTensor,
        nEmbdHead: Long,
        nHead: Long,
        nHeadKv: Long,
        il: Int,
    ): LlmGraphQkv {
        TODO("Port llm_graph_context::build_qkv")
    }

    /** Construct a feed-forward network sub-graph at layer [il]. */
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
        TODO("Port llm_graph_context::build_ffn")
    }

    /** Construct a Mixture-of-Experts FFN sub-graph at layer [il]. */
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
        TODO("Port llm_graph_context::build_moe_ffn")
    }

    // -- input builders -----------------------------------------------------

    /** Create input tensors for token-id / embedding injection. */
    open fun buildInpEmbd(tokEmbd: GGMLTensor): GGMLTensor {
        TODO("Port llm_graph_context::build_inp_embd")
    }

    /** Create the position-id input tensor. */
    open fun buildInpPos(): GGMLTensor {
        TODO("Port llm_graph_context::build_inp_pos")
    }

    /** Create the output-id selection tensor (may return `null` if all outputs are needed). */
    open fun buildInpOutIds(): GGMLTensor? {
        TODO("Port llm_graph_context::build_inp_out_ids")
    }

    // -- attention builders -------------------------------------------------

    /** Build attention inputs for the no-cache path. */
    open fun buildAttnInpNoCache(): LlmGraphInputAttnNoCache {
        TODO("Port llm_graph_context::build_attn_inp_no_cache")
    }

    /** Self-attention without a persistent KV cache. */
    open fun buildAttn(
        inp: LlmGraphInputAttnNoCache,
        wo: GGMLTensor, woB: GGMLTensor?, woS: GGMLTensor?,
        qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
        kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
        kqScale: Float, il: Int,
    ): GGMLTensor {
        TODO("Port llm_graph_context::build_attn (no_cache)")
    }

    /** Build attention inputs for the KV-cache path. */
    open fun buildAttnInpKv(): LlmGraphInputAttnKv {
        TODO("Port llm_graph_context::build_attn_inp_kv")
    }

    /** Self-attention with a persistent KV cache. */
    open fun buildAttn(
        inp: LlmGraphInputAttnKv,
        wo: GGMLTensor, woB: GGMLTensor?, woS: GGMLTensor?,
        qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
        kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
        kqScale: Float, il: Int,
    ): GGMLTensor {
        TODO("Port llm_graph_context::build_attn (kv)")
    }
}
