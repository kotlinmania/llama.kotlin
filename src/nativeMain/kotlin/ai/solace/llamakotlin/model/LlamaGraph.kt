// port-lint: source llama.cpp/src/llama-graph.h llama.cpp/src/llama-graph.cpp
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

// =============================================================================
// Additional graph types from llama-graph.h NOT already in InferencePipeline.kt
//
// The core graph types (LlmFfnOpType, LlmFfnGateType, LlmNormType,
// LlmGraphInput, LlmGraphInputEmbd/Pos/OutIds/Mean/Cls/AttnKv/AttnNoCache/Rs/
// AttnCross, LlmGraphQkv, LlmGraphResult, LlmGraphParams, LlmGraphContext)
// are defined in InferencePipeline.kt. This file ports the remaining types
// from the full C++ llama-graph.h header.
// =============================================================================

// ---------------------------------------------------------------------------
// LlamaCross – encoder → decoder embedding shuttle
// Ported from: llama-graph.h  struct llama_cross
// ---------------------------------------------------------------------------

/**
 * Temporary container that carries encoder output embeddings into a decoder's
 * cross-attention layers.
 *
 * Port of `struct llama_cross` from `llama-graph.h`.
 */
data class LlamaCross(
    var nEmbd: Long = 0,
    var nEnc: Long = 0,

    /** Encoder output embeddings (host-side copy). */
    val vEmbd: MutableList<Float> = mutableListOf(),

    /**
     * Sequence IDs assigned to each encoder output position.
     * Needed for constructing the cross-attention mask in the decoder.
     */
    val seqIdsEnc: MutableList<MutableSet<LlamaSeqId>> = mutableListOf(),
)

// ---------------------------------------------------------------------------
// Additional graph-input types not yet in InferencePipeline.kt
// ---------------------------------------------------------------------------

/**
 * Temperature-tuning input (used by Llama4).
 *
 * Port of `llm_graph_input_attn_temp`.
 */
class LlmGraphInputAttnTemp(
    val nAttnTempFloorScale: Int,
    val fAttnTempScale: Float,
    val fAttnTempOffset: Float,
) : LlmGraphInput {
    var attnScale: GGMLTensor? = null  // F32 [n_batch]

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_attn_temp::set_input")
    }
}

/**
 * Relative-position bucket input (encoder self-attention, e.g. T5).
 *
 * Port of `llm_graph_input_pos_bucket`.
 */
class LlmGraphInputPosBucket(
    val hparams: LlamaModelHParams,
) : LlmGraphInput {
    var posBucket: GGMLTensor? = null  // I32 [n_batch, n_batch]

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_pos_bucket::set_input")
    }
}

/**
 * Relative-position bucket input for KV-cached decoder (T5-style).
 *
 * Port of `llm_graph_input_pos_bucket_kv`.
 */
class LlmGraphInputPosBucketKv(
    val hparams: LlamaModelHParams,
) : LlmGraphInput {
    var posBucket: GGMLTensor? = null  // I32 [n_kv, n_batch]

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_pos_bucket_kv::set_input")
    }
}

/**
 * Cross-attention encoder embeddings input.
 *
 * Port of `llm_graph_input_cross_embd`.
 */
class LlmGraphInputCrossEmbd(
    val cross: LlamaCross?,
) : LlmGraphInput {
    var crossEmbd: GGMLTensor? = null  // F32 [n_embd, n_outputs_enc]

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_cross_embd::set_input")
    }
}

/**
 * V-less KV cache input (stores only K, no V).
 *
 * Port of `llm_graph_input_attn_k`.
 */
class LlmGraphInputAttnK(
    val hparams: LlamaModelHParams,
    val cparams: LlamaCParams,
) : LlmGraphInput {
    var selfKIdxs: GGMLTensor? = null       // I64 [n_batch]
    var selfKqMask: GGMLTensor? = null      // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    var selfKqMaskCnv: GGMLTensor? = null

    fun getKIdxs(): GGMLTensor? = selfKIdxs
    fun getKqMask(): GGMLTensor? = selfKqMaskCnv

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_attn_k::set_input")
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        TODO("Port llm_graph_input_attn_k::can_reuse")
    }
}

/**
 * ISWA (interleaved sliding-window attention) KV cache input.
 *
 * Port of `llm_graph_input_attn_kv_iswa`.
 */
class LlmGraphInputAttnKvIswa(
    val hparams: LlamaModelHParams,
    val cparams: LlamaCParams,
) : LlmGraphInput {
    // Full-attention stream
    var selfKIdxs: GGMLTensor? = null
    var selfVIdxs: GGMLTensor? = null
    var selfKqMask: GGMLTensor? = null
    var selfKqMaskCnv: GGMLTensor? = null

    // SWA stream
    var selfKIdxsSwa: GGMLTensor? = null
    var selfVIdxsSwa: GGMLTensor? = null
    var selfKqMaskSwa: GGMLTensor? = null
    var selfKqMaskSwaCnv: GGMLTensor? = null

    // RoPE rotation tensors
    var selfKRot: GGMLTensor? = null
    var selfVRot: GGMLTensor? = null
    var selfKRotSwa: GGMLTensor? = null
    var selfVRotSwa: GGMLTensor? = null

    fun getKIdxs(): GGMLTensor? = selfKIdxs
    fun getVIdxs(): GGMLTensor? = selfVIdxs
    fun getKIdxsSwa(): GGMLTensor? = selfKIdxsSwa
    fun getVIdxsSwa(): GGMLTensor? = selfVIdxsSwa
    fun getKqMask(): GGMLTensor? = selfKqMaskCnv
    fun getKqMaskSwa(): GGMLTensor? = selfKqMaskSwaCnv

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_attn_kv_iswa::set_input")
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        TODO("Port llm_graph_input_attn_kv_iswa::can_reuse")
    }
}

/**
 * Hybrid memory input (attention KV + recurrent state).
 *
 * Port of `llm_graph_input_mem_hybrid`.
 */
class LlmGraphInputMemHybrid(
    val cparams: LlamaCParams,
    val inpAttn: LlmGraphInputAttnKv,
    val inpRs: LlmGraphInputRs,
) : LlmGraphInput {

    fun getAttn(): LlmGraphInputAttnKv = inpAttn
    fun getRecr(): LlmGraphInputRs = inpRs

    override fun setInput(ubatch: LlamaUBatch) {
        inpAttn.setInput(ubatch)
        inpRs.setInput(ubatch)
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        return inpAttn.canReuse(params) && inpRs.canReuse(params)
    }
}

/**
 * Hybrid memory input (K-only attention + recurrent state).
 *
 * Port of `llm_graph_input_mem_hybrid_k`.
 */
class LlmGraphInputMemHybridK(
    val cparams: LlamaCParams,
    val inpAttn: LlmGraphInputAttnK,
    val inpRs: LlmGraphInputRs,
) : LlmGraphInput {

    fun getAttn(): LlmGraphInputAttnK = inpAttn
    fun getRecr(): LlmGraphInputRs = inpRs

    override fun setInput(ubatch: LlamaUBatch) {
        inpAttn.setInput(ubatch)
        inpRs.setInput(ubatch)
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        return inpAttn.canReuse(params) && inpRs.canReuse(params)
    }
}

/**
 * Hybrid memory input (ISWA attention + recurrent state).
 *
 * Port of `llm_graph_input_mem_hybrid_iswa`.
 */
class LlmGraphInputMemHybridIswa(
    val cparams: LlamaCParams,
    val inpAttn: LlmGraphInputAttnKvIswa,
    val inpRs: LlmGraphInputRs,
) : LlmGraphInput {

    fun getAttn(): LlmGraphInputAttnKvIswa = inpAttn
    fun getRecr(): LlmGraphInputRs = inpRs

    override fun setInput(ubatch: LlamaUBatch) {
        inpAttn.setInput(ubatch)
        inpRs.setInput(ubatch)
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        return inpAttn.canReuse(params) && inpRs.canReuse(params)
    }
}

/**
 * Backend-sampling input (carries per-sequence sampler references).
 *
 * Port of `llm_graph_input_sampling`.
 */
class LlmGraphInputSampling(
    val samplers: Map<LlamaSeqId, Any>,  // placeholder for LlamaSampler*
) : LlmGraphInput {

    override fun setInput(ubatch: LlamaUBatch) {
        TODO("Port llm_graph_input_sampling::set_input")
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        TODO("Port llm_graph_input_sampling::can_reuse")
    }
}

// ---------------------------------------------------------------------------
// Graph-context extensions for the additional input/attention types
// ---------------------------------------------------------------------------

/**
 * Extension functions on [LlmGraphContext] for the graph-building helpers
 * that depend on the additional input types declared in this file.
 *
 * These correspond to the remaining `build_*` methods in the C++ header that
 * are not yet covered by the base [LlmGraphContext] in InferencePipeline.kt.
 */

/** Build attention-temperature scaling input. */
fun LlmGraphContext.buildInpAttnScale(): GGMLTensor {
    TODO("Port llm_graph_context::build_inp_attn_scale")
}

/** Build mean-pooling input. */
fun LlmGraphContext.buildInpMean(): GGMLTensor {
    TODO("Port llm_graph_context::build_inp_mean")
}

/** Build CLS-token selector input. */
fun LlmGraphContext.buildInpCls(): GGMLTensor {
    TODO("Port llm_graph_context::build_inp_cls")
}

/** Build cross-attention embedding input. */
fun LlmGraphContext.buildInpCrossEmbd(): GGMLTensor {
    TODO("Port llm_graph_context::build_inp_cross_embd")
}

/** Build encoder-side position-bucket input. */
fun LlmGraphContext.buildInpPosBucketEnc(): GGMLTensor {
    TODO("Port llm_graph_context::build_inp_pos_bucket_enc")
}

/** Build decoder-side position-bucket input. */
fun LlmGraphContext.buildInpPosBucketDec(): GGMLTensor {
    TODO("Port llm_graph_context::build_inp_pos_bucket_dec")
}

/** Compute position bias from a bucket tensor and rel-attn bias weights. */
fun LlmGraphContext.buildPosBias(posBucket: GGMLTensor, attnRelB: GGMLTensor): GGMLTensor {
    TODO("Port llm_graph_context::build_pos_bias")
}

/**
 * Core multi-head attention sub-graph.
 *
 * Port of `llm_graph_context::build_attn_mha()`.
 */
fun LlmGraphContext.buildAttnMha(
    q: GGMLTensor,
    k: GGMLTensor,
    v: GGMLTensor,
    kqB: GGMLTensor?,
    kqMask: GGMLTensor?,
    sinks: GGMLTensor?,
    vMla: GGMLTensor?,
    kqScale: Float,
    il: Int,
): GGMLTensor {
    TODO("Port llm_graph_context::build_attn_mha")
}

/** Build V-less (K-only) attention inputs. */
fun LlmGraphContext.buildAttnInpK(): LlmGraphInputAttnK {
    TODO("Port llm_graph_context::build_attn_inp_k")
}

/** Build ISWA attention inputs. */
fun LlmGraphContext.buildAttnInpKvIswa(): LlmGraphInputAttnKvIswa {
    TODO("Port llm_graph_context::build_attn_inp_kv_iswa")
}

/** Build self-attention with V-less (K-only) cache. */
fun LlmGraphContext.buildAttn(
    inp: LlmGraphInputAttnK,
    wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
    qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
    kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
    kqScale: Float, il: Int,
): GGMLTensor {
    TODO("Port llm_graph_context::build_attn (attn_k)")
}

/** Build self-attention with ISWA cache. */
fun LlmGraphContext.buildAttn(
    inp: LlmGraphInputAttnKvIswa,
    wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
    qCur: GGMLTensor, kCur: GGMLTensor?, vCur: GGMLTensor?,
    kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
    kqScale: Float, il: Int,
): GGMLTensor {
    TODO("Port llm_graph_context::build_attn (iswa)")
}

/** Build cross-attention inputs. */
fun LlmGraphContext.buildAttnInpCross(): LlmGraphInputAttnCross {
    TODO("Port llm_graph_context::build_attn_inp_cross")
}

/** Build cross-attention sub-graph. */
fun LlmGraphContext.buildAttn(
    inp: LlmGraphInputAttnCross,
    wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
    qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
    kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
    kqScale: Float, il: Int,
): GGMLTensor {
    TODO("Port llm_graph_context::build_attn (cross)")
}

// -- recurrent-state builders -----------------------------------------------

/** Build recurrent-state copy input. */
fun LlmGraphContext.buildRsInp(): LlmGraphInputRs {
    TODO("Port llm_graph_context::build_rs_inp")
}

/**
 * Build recurrent-state (SSM / RWKV) sub-graph.
 *
 * Port of `llm_graph_context::build_rs()`.
 */
fun LlmGraphContext.buildRs(
    inp: LlmGraphInputRs,
    s: GGMLTensor,
    stateSize: Int,
    nSeqs: Int,
): GGMLTensor {
    TODO("Port llm_graph_context::build_rs")
}

/** Load RWKV token-shift state. */
fun LlmGraphContext.buildRwkvTokenShiftLoad(
    inp: LlmGraphInputRs,
    ubatch: LlamaUBatch,
    il: Int,
): GGMLTensor {
    TODO("Port llm_graph_context::build_rwkv_token_shift_load")
}

/** Store RWKV token-shift state. */
fun LlmGraphContext.buildRwkvTokenShiftStore(
    tokenShift: GGMLTensor,
    ubatch: LlamaUBatch,
    il: Int,
): GGMLTensor {
    TODO("Port llm_graph_context::build_rwkv_token_shift_store")
}

// -- hybrid memory ----------------------------------------------------------

/** Build hybrid (KV + recurrent) memory inputs. */
fun LlmGraphContext.buildInpMemHybrid(): LlmGraphInputMemHybrid {
    TODO("Port llm_graph_context::build_inp_mem_hybrid")
}

/** Build hybrid (K-only + recurrent) memory inputs. */
fun LlmGraphContext.buildInpMemHybridK(): LlmGraphInputMemHybridK {
    TODO("Port llm_graph_context::build_inp_mem_hybrid_k")
}

/** Build hybrid (ISWA + recurrent) memory inputs. */
fun LlmGraphContext.buildInpMemHybridIswa(): LlmGraphInputMemHybridIswa {
    TODO("Port llm_graph_context::build_inp_mem_hybrid_iswa")
}

// -- pooling / sampling / dense-out -----------------------------------------

/** Build pooling sub-graph. */
fun LlmGraphContext.buildPooling(
    cls: GGMLTensor?, clsB: GGMLTensor?,
    clsOut: GGMLTensor?, clsOutB: GGMLTensor?,
    clsNorm: GGMLTensor?,
) {
    TODO("Port llm_graph_context::build_pooling")
}

/** Build backend-sampling sub-graph. */
fun LlmGraphContext.buildSampling() {
    TODO("Port llm_graph_context::build_sampling")
}

/** Build dense-out (final linear projections). */
fun LlmGraphContext.buildDenseOut(
    dense2: GGMLTensor?, dense2B: GGMLTensor?,
    dense3: GGMLTensor?,
) {
    TODO("Port llm_graph_context::build_dense_out")
}

// =============================================================================
// Standalone utility
// Ported from: llama-graph.h  llama_relative_position_bucket
// =============================================================================

/**
 * Compute a relative-position bucket index (used by T5-style position bias).
 *
 * Port of `llama_relative_position_bucket()`.
 */
fun llamaRelativePositionBucket(
    x: LlamaPos,
    y: LlamaPos,
    nBuckets: Long,
    bidirectional: Boolean,
): Int {
    TODO("Port llama_relative_position_bucket")
}
