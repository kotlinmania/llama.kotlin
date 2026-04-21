// port-lint: source llama.cpp/src/llama-graph.h llama.cpp/src/llama-graph.cpp
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.min

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
        val scale = attnScale ?: return
        val positions = ubatch.pos ?: return
        val data = scale.data as? ByteArray ?: return
        val nTokens = ubatch.nTokens

        require(fAttnTempScale != 0.0f) { "fAttnTempScale must not be zero" }
        require(nAttnTempFloorScale != 0) { "nAttnTempFloorScale must not be zero" }

        for (i in 0 until nTokens) {
            val pos = positions[i].toFloat()
            val value = ln(
                floor((pos + fAttnTempOffset) / nAttnTempFloorScale) + 1.0
            ).toFloat() * fAttnTempScale + 1.0f
            val bits = value.toRawBits()
            val offset = i * 4
            data[offset + 0] = (bits and 0xFF).toByte()
            data[offset + 1] = ((bits shr 8) and 0xFF).toByte()
            data[offset + 2] = ((bits shr 16) and 0xFF).toByte()
            data[offset + 3] = ((bits shr 24) and 0xFF).toByte()
        }
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
        val bucket = posBucket ?: return
        val positions = ubatch.pos ?: return
        val data = bucket.data as? ByteArray ?: return
        val nTokens = ubatch.nTokens

        require(!ubatch.equalSeqs) { "pos_bucket requires non-equal sequences" }

        for (j in 0 until nTokens) {
            for (i in 0 until nTokens) {
                val value = llamaRelativePositionBucket(
                    positions[i], positions[j],
                    hparams.nRelAttnBkts.toLong(), bidirectional = true
                )
                val offset = (j * nTokens + i) * 4
                data[offset + 0] = (value and 0xFF).toByte()
                data[offset + 1] = ((value shr 8) and 0xFF).toByte()
                data[offset + 2] = ((value shr 16) and 0xFF).toByte()
                data[offset + 3] = ((value shr 24) and 0xFF).toByte()
            }
        }
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
        // In C++, this delegates to mctx->set_input_pos_bucket(pos_bucket, ubatch).
        // The KV cache memory context computes relative-position buckets between
        // query positions (from ubatch) and cached key positions (from the KV store).
        // Requires LlamaKvCacheContext.setInputPosBucket() integration.
        @Suppress("UNUSED_VARIABLE")
        val bucket = posBucket ?: return
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
        @Suppress("UNUSED_PARAMETER", "UNUSED_VARIABLE")
        val unused = ubatch // ubatch is unused in C++ implementation

        val embd = crossEmbd ?: return
        val cr = cross ?: return
        if (cr.vEmbd.isEmpty()) return

        val data = embd.data as? ByteArray ?: return
        // Copy encoder embeddings (F32) into tensor data
        val nBytes = minOf(cr.vEmbd.size * 4, data.size)
        for (i in 0 until nBytes / 4) {
            val bits = cr.vEmbd[i].toRawBits()
            val offset = i * 4
            data[offset + 0] = (bits and 0xFF).toByte()
            data[offset + 1] = ((bits shr 8) and 0xFF).toByte()
            data[offset + 2] = ((bits shr 16) and 0xFF).toByte()
            data[offset + 3] = ((bits shr 24) and 0xFF).toByte()
        }
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
        // In C++, delegates to:
        //   mctx->set_input_k_idxs(self_k_idxs, ubatch)
        //   mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn)
        // Requires LlamaKvCacheContext integration for index and mask computation.
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        val kIdxs = selfKIdxs ?: return false
        val kqMask = selfKqMask ?: return false

        var res = true
        // K indices must match the number of tokens in the new ubatch
        res = res && (kIdxs.ne[0] == params.ubatch.nTokens.toLong())

        // KQ mask dimensions: [n_kv, n_tokens/n_stream, 1, n_stream]
        // Full validation requires mctx.getNKv() which isn't available yet,
        // so we check what we can: the token dimension
        val nStream = if (params.cparams.kvUnified) 1L else params.ubatch.nSeqsUnq.toLong()
        res = res && (kqMask.ne[1] == params.ubatch.nTokens.toLong() / nStream)
        res = res && (kqMask.ne[2] == 1L)
        res = res && (kqMask.ne[3] == nStream)

        return res
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
        // In C++, this delegates to mctx->get_base() and mctx->get_swa() for:
        //   - set_input_k_idxs / set_input_v_idxs for both base and SWA streams
        //   - set_input_kq_mask for both base and SWA streams
        //   - set_input_k_rot / set_input_v_rot (optional) for both streams
        // Requires LlamaKvCacheIswaContext integration.
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        val kIdxs = selfKIdxs ?: return false
        val kIdxsSwa = selfKIdxsSwa ?: return false
        val kqMask = selfKqMask ?: return false
        val kqMaskSwa = selfKqMaskSwa ?: return false

        var res = true

        // Both base and SWA K-index tensors must match token count
        res = res && (kIdxs.ne[0] == params.ubatch.nTokens.toLong())
        res = res && (kIdxsSwa.ne[0] == params.ubatch.nTokens.toLong())

        // KQ mask dimension checks for base stream
        val nStream = if (params.cparams.kvUnified) 1L else params.ubatch.nSeqsUnq.toLong()
        res = res && (kqMask.ne[1] == params.ubatch.nTokens.toLong() / nStream)
        res = res && (kqMask.ne[2] == 1L)
        res = res && (kqMask.ne[3] == nStream)

        // KQ mask dimension checks for SWA stream
        res = res && (kqMaskSwa.ne[1] == params.ubatch.nTokens.toLong() / nStream)
        res = res && (kqMaskSwa.ne[2] == 1L)
        res = res && (kqMaskSwa.ne[3] == nStream)

        return res
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
        // Collect active seq_ids: those with output[i] == true
        val output = ubatch.output ?: return
        val seqIds = ubatch.seqId ?: return
        val activeSamplers = mutableSetOf<LlamaSeqId>()

        for (i in 0 until ubatch.nTokens) {
            if (output[i]) {
                val seqId = seqIds[i][0]
                activeSamplers.add(seqId)
            }
        }

        // For each active sampler, invoke its backend_set_input if available.
        // Currently samplers are typed as Any (placeholder); when the sampler
        // interface is fully ported, this will call sampler.iface.backendSetInput().
        for (seqId in activeSamplers) {
            val sampler = samplers[seqId] ?: continue
            // sampler.iface.backendSetInput(sampler) — requires sampler interface port
            @Suppress("UNUSED_VARIABLE")
            val unused = sampler // sampler.iface.backendSetInput(sampler) — requires sampler interface port
        }
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        // In C++, compares samplers map: sizes must match and all entries identical.
        // LlmGraphParams doesn't carry samplers yet, so conservatively return false
        // to force rebuild until the sampler infrastructure is integrated.
        return false
    }
}


// ---------------------------------------------------------------------------
// The builder extension functions that were previously here have been moved
// into LlmGraphContext as member functions in LlmGraphTypes.kt.
// Member functions shadow extensions with the same signature.
// ---------------------------------------------------------------------------


// =============================================================================
// Standalone utility
// Ported from: llama-graph.cpp  llama_relative_position_bucket
// =============================================================================

/**
 * Compute a relative-position bucket index (used by T5-style position bias).
 *
 * The bucketing scheme uses exact positions for small offsets and logarithmic
 * buckets for larger distances. With `bidirectional = true` half the buckets
 * are reserved for negative (backward) relative positions.
 *
 * Port of `llama_relative_position_bucket()` from `llama-graph.cpp`.
 */
fun llamaRelativePositionBucket(
    x: LlamaPos,
    y: LlamaPos,
    nBuckets: Long,
    bidirectional: Boolean,
): Int {
    var relativePosition = (x - y).toLong()
    var numBuckets = nBuckets.toInt()
    var ret = 0

    if (bidirectional) {
        numBuckets /= 2
        if (relativePosition > 0) {
            ret += numBuckets
        } else {
            relativePosition = -relativePosition
        }
    } else {
        if (relativePosition > 0) {
            relativePosition = 0
        } else {
            relativePosition = -relativePosition
        }
    }

    val maxExact = numBuckets / 2
    // Small relative positions use exact bucket mapping
    if (relativePosition < maxExact) {
        return ret + relativePosition.toInt()
    }

    // Larger relative positions use logarithmic buckets
    val valIf = (maxExact +
        (kotlin.math.ln(relativePosition.toDouble() / maxExact) /
            kotlin.math.ln(((nBuckets * 4).toDouble()) / maxExact) *
            (numBuckets - maxExact)).toInt())
    val bucket = minOf(valIf, numBuckets - 1)
    return ret + bucket
}
