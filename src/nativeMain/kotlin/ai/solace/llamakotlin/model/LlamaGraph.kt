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
    val hparams: LlamaHparams,
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
    val hparams: LlamaHparams,
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
    val hparams: LlamaHparams,
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
    val hparams: LlamaHparams,
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
// =======================================================================
// Common builders
// =======================================================================

/** Apply control vector to the hidden state at layer [il]. */
fun LlmGraphContext.buildCvec(cur: GGMLTensor, il: Int): GGMLTensor {
    val cv = cvec ?: return cur
    return cv.applyTo(cur, il)
}

/**
 * Matrix-multiply [cur] by weight [w] with optional LoRA deltas and
 * per-tensor scale [wS].
 *
 * Port of `llm_graph_context::build_lora_mm`.
 */
fun LlmGraphContext.buildLoraMm(w: GGMLTensor, cur: GGMLTensor, wS: GGMLTensor? = null): GGMLTensor {
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
fun LlmGraphContext.buildLoraMmId(w: GGMLTensor, cur: GGMLTensor, ids: GGMLTensor): GGMLTensor {
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
fun LlmGraphContext.buildNorm(
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
fun LlmGraphContext.buildQkv(
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
fun LlmGraphContext.buildFfn(
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
fun LlmGraphContext.buildMoeFfn(
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
fun LlmGraphContext.buildMoeFfnFull(
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
fun LlmGraphContext.buildInpEmbd(tokEmbd: GGMLTensor): GGMLTensor {
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
fun LlmGraphContext.buildInpPos(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputPos(hparams.nPosPerEmbd())
    inp.pos = ggmlNewTensor1d(c, GGMLType.I32, nTokens * hparams.nPosPerEmbd())
    ggmlSetInput(inp.pos!!)
    r.addInput(inp)
    return inp.pos!!
}

/** Create the output-id selection tensor. Port of `build_inp_out_ids`. */
fun LlmGraphContext.buildInpOutIds(): GGMLTensor? {
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
fun LlmGraphContext.buildAttnMha(
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
fun LlmGraphContext.buildAttnInpNoCache(): LlmGraphInputAttnNoCache {
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
fun LlmGraphContext.buildAttn(
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

// port-lint: source llama.cpp/src/llama-graph.cpp  llm_graph_context::build_attn_inp_kv

/**
 * Build attention inputs for the KV-cache path.
 *
 * Port of `llm_graph_context::build_attn_inp_kv`.  Creates K/V index
 * tensors, the causal KQ mask, and optional rotation matrices from the
 * [KVCacheContext] stored in [params].mctx.
 */
fun LlmGraphContext.buildAttnInpKv(): LlmGraphInputAttnKv {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val mctx = params.mctx ?: error("mctx required for KV attention")
    val mctxCur = mctx as KVCacheContext

    val inp = LlmGraphInputAttnKv(hparams, cparams)

    // Build K/V index tensors from the cache context
    inp.selfKIdxs = mctxCur.buildInputKIdxs(c, ubatch)
    inp.selfVIdxs = mctxCur.buildInputVIdxs(c, ubatch)

    // Build causal KQ mask: [nKv, nTokens, 1, 1]
    val nKv = mctxCur.getNKv().toLong()
    inp.selfKqMask = ggmlNewTensor4d(c, GGMLType.F32, nKv, nTokens, 1, 1)
    ggmlSetInput(inp.selfKqMask!!)
    inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMask!!, GGMLType.F16)
        else inp.selfKqMask

    // Optional rotation matrices (for k-shift)
    inp.selfKRot = mctxCur.buildInputKRot(c)
    inp.selfVRot = mctxCur.buildInputVRot(c)

    return r.addInputTyped(inp)
}

// port-lint: source llama.cpp/src/llama-graph.cpp  llm_graph_context::build_attn(llm_graph_input_attn_kv *)

/**
 * Self-attention with a persistent KV cache.
 *
 * Port of `llm_graph_context::build_attn(llm_graph_input_attn_kv *)`.
 * Applies optional rotation, stores K/V into the cache, retrieves cached
 * K/V, runs multi-head attention, and applies the output projection.
 */
fun LlmGraphContext.buildAttn(
    inp: LlmGraphInputAttnKv,
    wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
    qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
    kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
    kqScale: Float, il: Int,
): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val graph = gf ?: error("gf not initialised")
    val mctxCur = params.mctx as? KVCacheContext ?: error("mctx required for KV attention")

    var q = qCur
    var k = kCur
    var v = vCur

    // Apply optional rotation matrices
    if (inp.selfKRot != null) {
        q = ggmlMulMatAux(c, q, inp.selfKRot!!)
        k = ggmlMulMatAux(c, k, inp.selfKRot!!)
    }
    if (inp.selfVRot != null) {
        v = ggmlMulMatAux(c, v, inp.selfVRot!!)
    }

    // Ensure Q/K/V nodes are materialized before cache operations
    ggmlBuildForwardExpand(graph, q)
    ggmlBuildForwardExpand(graph, v)
    ggmlBuildForwardExpand(graph, k)

    // Store K and V into the cache
    val kIdxs = inp.getKIdxs() ?: error("K indices not initialised")
    val vIdxs = inp.getVIdxs() ?: error("V indices not initialised")

    ggmlBuildForwardExpand(graph, mctxCur.cpyK(c, k, kIdxs, il))
    ggmlBuildForwardExpand(graph, mctxCur.cpyV(c, v, vIdxs, il))

    // Retrieve cached K/V for the full attention window
    val kqMask = inp.getKqMask()
    val kCached = mctxCur.getK(c, il)
    val vCached = mctxCur.getV(c, il)

    var cur = buildAttnMha(q, kCached, vCached, kqB, kqMask, sinks, vMla, kqScale, il)
    cb(cur, "kqv_out", il)

    // Apply optional V rotation to the output
    if (inp.selfVRot != null) {
        cur = ggmlMulMatAux(c, cur, inp.selfVRot!!)
    }

    // Output projection
    if (wo != null) {
        if (arch == LlamaModelArch.GLM4 || arch == LlamaModelArch.GLM4_MOE || arch == LlamaModelArch.JAIS2) {
            // GLM4/JAIS2 need F32 precision accumulators
            cur = buildLoraMm(wo, cur)
            ggmlMulMatSetPrec(cur, GGMLPrec.F32)
            if (woS != null) {
                cur = mul(c, cur, woS)
            }
        } else {
            cur = buildLoraMm(wo, cur, woS)
        }
    }
    if (woB != null) {
        cur = add(c, cur, woB)
    }

    return cur
}

// =========================================================================
// Builder methods ported from llama-graph.cpp (lines 1801–2897)
// =========================================================================

/** Build attention-temperature scaling input. Port of `build_inp_attn_scale`. */
fun LlmGraphContext.buildInpAttnScale(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputAttnTemp(
        nAttnTempFloorScale = hparams.nAttnTempFloorScale,
        fAttnTempScale = hparams.fAttnTempScale,
        fAttnTempOffset = hparams.fAttnTempOffset,
    )
    // Must be 1x1xN for broadcasting across attention heads
    inp.attnScale = ggmlNewTensor3d(c, GGMLType.F32, 1, 1, nTokens)
    ggmlSetInput(inp.attnScale!!)
    ggmlSetName(inp.attnScale!!, "attn_scale")
    r.addInput(inp)
    return inp.attnScale!!
}

/** Build mean-pooling input. Port of `build_inp_mean`. */
fun LlmGraphContext.buildInpMean(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputMean(cparams)
    inp.mean = ggmlNewTensor2d(c, GGMLType.F32, nTokens, ubatch.nSeqsUnq.toLong())
    ggmlSetInput(inp.mean!!)
    r.addInput(inp)
    return inp.mean!!
}

/** Build CLS-token selector input. Port of `build_inp_cls`. */
fun LlmGraphContext.buildInpCls(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputCls(cparams, arch)
    inp.cls = ggmlNewTensor1d(c, GGMLType.I32, ubatch.nSeqsUnq.toLong())
    ggmlSetInput(inp.cls!!)
    r.addInput(inp)
    return inp.cls!!
}

/** Build cross-attention embedding input. Port of `build_inp_cross_embd`. */
fun LlmGraphContext.buildInpCrossEmbd(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val cr = cross
    val inp = LlmGraphInputCrossEmbd(cr)
    val nEmbdCross = if (cr != null && cr.vEmbd.isNotEmpty()) cr.nEmbd else hparams.nEmbdInp().toLong()
    val nEnc = if (cr != null && cr.vEmbd.isNotEmpty()) cr.nEnc else hparams.nCtxTrain.toLong()
    inp.crossEmbd = ggmlNewTensor2d(c, GGMLType.F32, nEmbdCross, nEnc)
    ggmlSetInput(inp.crossEmbd!!)
    r.addInput(inp)
    return inp.crossEmbd!!
}

/** Build encoder-side position-bucket input. Port of `build_inp_pos_bucket_enc`. */
fun LlmGraphContext.buildInpPosBucketEnc(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputPosBucket(hparams)
    inp.posBucket = ggmlNewTensor2d(c, GGMLType.I32, nTokens, nTokens)
    ggmlSetInput(inp.posBucket!!)
    r.addInput(inp)
    return inp.posBucket!!
}

/** Build decoder-side position-bucket input. Port of `build_inp_pos_bucket_dec`. */
fun LlmGraphContext.buildInpPosBucketDec(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val mctxCur = params.mctx as? KVCacheContext
        ?: error("mctx must be KVCacheContext for pos_bucket_dec")
    val inp = LlmGraphInputPosBucketKv(hparams)
    val nKv = mctxCur.getNKv().toLong()
    inp.posBucket = ggmlNewTensor2d(c, GGMLType.I32, nKv, nTokens)
    ggmlSetInput(inp.posBucket!!)
    r.addInput(inp)
    return inp.posBucket!!
}

/**
 * Compute position bias from a bucket tensor and relative-attention bias weights.
 * Port of `build_pos_bias`.
 */
fun LlmGraphContext.buildPosBias(posBucket: GGMLTensor, attnRelB: GGMLTensor): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    // Flatten the 2D bucket to 1D for ggml_get_rows
    val posBucket1d = ggmlReshape1d(c, posBucket, posBucket.ne[0] * posBucket.ne[1])
    cb(posBucket1d, "pos_bucket_1d", -1)

    var posBias = ggmlGetRows(c, attnRelB, posBucket1d)
    // Reshape to 3D and permute to [nHead, ne0, ne1] layout
    posBias = ggmlReshape3d(c, posBias, posBias.ne[0], posBucket.ne[0], posBucket.ne[1])
    posBias = ggmlPermute(c, posBias, 2, 0, 1, 3)
    posBias = ggmlCont(c, posBias)
    cb(posBias, "pos_bias", -1)
    return posBias
}

/** Build cross-attention inputs. Port of `build_attn_inp_cross`. */
fun LlmGraphContext.buildAttnInpCross(): LlmGraphInputAttnCross {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val cr = cross
    val inp = LlmGraphInputAttnCross(cr)
    val nEnc = if (cr != null && cr.vEmbd.isNotEmpty()) cr.nEnc else hparams.nCtxTrain.toLong()
    inp.crossKqMask = ggmlNewTensor4d(c, GGMLType.F32, nEnc, nTokens, 1, 1)
    ggmlSetInput(inp.crossKqMask!!)
    inp.crossKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.crossKqMask!!, GGMLType.F16)
        else inp.crossKqMask
    return r.addInputTyped(inp)
}

/** Build V-less (K-only) attention inputs. Port of `build_attn_inp_k`. */
fun LlmGraphContext.buildAttnInpK(): LlmGraphInputAttnK {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val mctxCur = params.mctx as? KVCacheContext
        ?: error("mctx must be KVCacheContext for attn_inp_k")
    val inp = LlmGraphInputAttnK(hparams, cparams)
    inp.selfKIdxs = mctxCur.buildInputKIdxs(c, ubatch)
    val nKv = mctxCur.getNKv().toLong()
    inp.selfKqMask = ggmlNewTensor4d(c, GGMLType.F32, nKv, nTokens, 1, 1)
    ggmlSetInput(inp.selfKqMask!!)
    inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMask!!, GGMLType.F16)
        else inp.selfKqMask
    return r.addInputTyped(inp)
}

/** Build ISWA attention inputs. Port of `build_attn_inp_kv_iswa`. */
fun LlmGraphContext.buildAttnInpKvIswa(): LlmGraphInputAttnKvIswa {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val mctxCur = params.mctx as? LlamaKvCacheIswaContext
        ?: error("mctx must be LlamaKvCacheIswaContext for attn_inp_kv_iswa")
    val inp = LlmGraphInputAttnKvIswa(hparams, cparams)

    // Base stream
    val baseCtx = mctxCur.getBase() as? KVCacheContext
        ?: error("ISWA base context must be KVCacheContext")
    inp.selfKIdxs = baseCtx.buildInputKIdxs(c, ubatch)
    inp.selfVIdxs = baseCtx.buildInputVIdxs(c, ubatch)
    inp.selfKqMask = buildAttnInpKqMask(c, baseCtx, ubatch, cparams)
    inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMask!!, GGMLType.F16)
        else inp.selfKqMask

    // SWA stream
    val swaCtx = mctxCur.getSwa() as? KVCacheContext
        ?: error("ISWA SWA context must be KVCacheContext")
    inp.selfKIdxsSwa = swaCtx.buildInputKIdxs(c, ubatch)
    inp.selfVIdxsSwa = swaCtx.buildInputVIdxs(c, ubatch)
    inp.selfKqMaskSwa = buildAttnInpKqMask(c, swaCtx, ubatch, cparams)
    inp.selfKqMaskSwaCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMaskSwa!!, GGMLType.F16)
        else inp.selfKqMaskSwa

    // RoPE rotation tensors
    inp.selfKRot = baseCtx.buildInputKRot(c)
    inp.selfVRot = baseCtx.buildInputVRot(c)
    inp.selfKRotSwa = swaCtx.buildInputKRot(c)
    inp.selfVRotSwa = swaCtx.buildInputVRot(c)

    return r.addInputTyped(inp)
}

/** Build self-attention with V-less (K-only) cache. Port of `build_attn(attn_k)`. */
fun LlmGraphContext.buildAttn(
    inp: LlmGraphInputAttnK,
    wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
    qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
    kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
    kqScale: Float, il: Int,
): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val graph = gf ?: error("gf not initialised")
    // Expand to prevent reordering
    ggmlBuildForwardExpand(graph, qCur)
    ggmlBuildForwardExpand(graph, vCur)
    ggmlBuildForwardExpand(graph, kCur)

    // Store K to cache
    val kIdxs = inp.getKIdxs()
    if (kIdxs != null) {
        val mctxCur = params.mctx as? KVCacheContext
        if (mctxCur != null) {
            ggmlBuildForwardExpand(graph, mctxCur.cpyK(c, kCur, kIdxs, il))
        }
    }

    val kqMask = inp.getKqMask()
    val q = qCur
    val k = (params.mctx as? KVCacheContext)?.getK(c, il) ?: kCur
    // V-less: derive v from k view
    val v = ggmlView4d(c, k, vCur.ne[0], k.ne[1], k.ne[2], k.ne[3],
        k.nb[1], k.nb[2], k.nb[3], 0u)

    var cur = buildAttnMha(q, k, v, kqB, kqMask, sinks, vMla, kqScale, il)
    cb(cur, "kqv_out", il)

    if (wo != null) {
        if (arch == LlamaModelArch.GLM4 || arch == LlamaModelArch.GLM4_MOE) {
            cur = buildLoraMm(wo, cur)
            ggmlMulMatSetPrec(cur, GGMLPrec.F32)
            if (woS != null) cur = mul(c, cur, woS)
        } else {
            cur = buildLoraMm(wo, cur, woS)
        }
    }
    if (woB != null) {
        cur = add(c, cur, woB)
    }
    return cur
}

/** Build self-attention with ISWA cache. Port of `build_attn(iswa)`. */
fun LlmGraphContext.buildAttn(
    inp: LlmGraphInputAttnKvIswa,
    wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
    qCur: GGMLTensor, kCur: GGMLTensor?, vCur: GGMLTensor?,
    kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
    kqScale: Float, il: Int,
): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val graph = gf ?: error("gf not initialised")
    val isSwa = hparams.isSwa(il)

    val kRot = if (isSwa) inp.selfKRotSwa else inp.selfKRot
    val vRot = if (isSwa) inp.selfVRotSwa else inp.selfVRot

    var qWork = qCur
    var kWork = kCur
    var vWork = vCur

    if (kRot != null) {
        qWork = ggmlMulMatAux(c, qWork, kRot)
        if (kWork != null) kWork = ggmlMulMatAux(c, kWork, kRot)
    }
    if (vRot != null) {
        if (vWork != null) vWork = ggmlMulMatAux(c, vWork, vRot)
    }

    ggmlBuildForwardExpand(graph, qWork)
    kWork?.let { ggmlBuildForwardExpand(graph, it) }
    vWork?.let { ggmlBuildForwardExpand(graph, it) }

    val mctxIswa = params.mctx as? LlamaKvCacheIswaContext
        ?: error("mctx must be LlamaKvCacheIswaContext for ISWA attention")
    val mctxCur = (if (isSwa) mctxIswa.getSwa() else mctxIswa.getBase()) as? KVCacheContext
        ?: error("ISWA sub-context must be KVCacheContext")

    // Optionally store K to cache
    if (kWork != null) {
        val kIdxs = if (isSwa) inp.getKIdxsSwa() else inp.getKIdxs()
        if (kIdxs != null) {
            ggmlBuildForwardExpand(graph, mctxCur.cpyK(c, kWork, kIdxs, il))
        }
    }

    // Optionally store V to cache
    if (vWork != null) {
        val vIdxs = if (isSwa) inp.getVIdxsSwa() else inp.getVIdxs()
        if (vIdxs != null) {
            ggmlBuildForwardExpand(graph, mctxCur.cpyV(c, vWork, vIdxs, il))
        }
    }

    val kqMask = if (isSwa) inp.getKqMaskSwa() else inp.getKqMask()
    val q = qWork
    val k = mctxCur.getK(c, il)
    val v = mctxCur.getV(c, il)

    var cur = buildAttnMha(q, k, v, kqB, kqMask, sinks, vMla, kqScale, il)
    cb(cur, "kqv_out", il)

    if (vRot != null) {
        cur = ggmlMulMatAux(c, cur, vRot)
    }

    if (wo != null) {
        cur = buildLoraMm(wo, cur, woS)
    }
    if (woB != null) {
        cur = add(c, cur, woB)
    }
    return cur
}

/** Build cross-attention sub-graph. Port of `build_attn(cross)`. */
fun LlmGraphContext.buildAttn(
    inp: LlmGraphInputAttnCross,
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

    val kqMask = inp.getKqMaskCross()
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

// -- recurrent-state builders -----------------------------------------------

/** Build recurrent-state copy input. Port of `build_rs_inp`. */
fun LlmGraphContext.buildRsInp(): LlmGraphInputRs {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val mctxCur = params.mctx as? LlamaMemoryRecurrentContext
        ?: error("mctx must be LlamaMemoryRecurrentContext for rs_inp")
    val inp = LlmGraphInputRs(mctx = mctxCur)

    val nRs = mctxCur.getNRs().toLong()
    val nSeqs = ubatch.nSeqs.toLong()

    inp.sCopy = ggmlNewTensor1d(c, GGMLType.I32, nRs)
    ggmlSetInput(inp.sCopy!!)
    inp.sCopyMain = ggmlView1d(c, inp.sCopy!!, nSeqs, 0u)
    inp.sCopyExtra = ggmlView1d(c, inp.sCopy!!, nRs - nSeqs,
        (nSeqs * ggmlElementSize(inp.sCopy!!).toLong()).toULong())
    inp.head = mctxCur.getHead()
    inp.rsZ = mctxCur.getRsZ()

    return r.addInputTyped(inp)
}

/**
 * Build recurrent-state (SSM / RWKV) sub-graph (low-level overload).
 * Port of `build_rs(s, state_copy_main, state_copy_extra, ...)`.
 */
fun LlmGraphContext.buildRs(
    s: GGMLTensor,
    stateCopyMain: GGMLTensor,
    stateCopyExtra: GGMLTensor,
    stateSize: Int,
    nSeqs: Int,
    nRs: Int,
    rsHead: Int,
    rsSize: Int,
    rsZero: Int,
    getStateRows: (GGMLContext, GGMLTensor, GGMLTensor) -> GGMLTensor,
): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val graph = gf ?: error("gf not initialised")

    val states = ggmlReshape2d(c, s, stateSize.toLong(), rsSize.toLong())

    // Clear a single state which will be copied to other cleared states
    val zeroMultiplier = if (rsZero >= 0) 1L else 0L
    val stateZero = ggmlView1d(c, states,
        stateSize.toLong() * zeroMultiplier,
        (rsZero.toLong() * states.nb[1].toLong() * zeroMultiplier).toULong())
    ggmlBuildForwardExpand(graph, ggmlScaleInplace(c, stateZero, 0.0f))

    // Copy states: {state_size, rs_size} -> {state_size, n_seqs}
    val outputStates = getStateRows(c, states, stateCopyMain)
    ggmlBuildForwardExpand(graph, outputStates)

    // Copy extra states (between n_seqs and n_rs)
    val statesExtra = ggmlGetRows(c, states, stateCopyExtra)
    ggmlBuildForwardExpand(graph,
        ggmlCpy(c, statesExtra,
            ggmlView2d(c, s, stateSize.toLong(), (nRs - nSeqs).toLong(),
                s.nb[1], ((rsHead + nSeqs).toLong() * s.nb[1].toLong()).toULong())))

    return outputStates
}

/**
 * Build recurrent-state sub-graph (high-level overload using input).
 * Port of `build_rs(inp, s, state_size, n_seqs, get_state_rows)`.
 */
fun LlmGraphContext.buildRs(
    inp: LlmGraphInputRs,
    s: GGMLTensor,
    stateSize: Int,
    nSeqs: Int,
    getStateRows: (GGMLContext, GGMLTensor, GGMLTensor) -> GGMLTensor = ::ggmlGetRows,
): GGMLTensor {
    val kvState = inp.mctx ?: error("LlmGraphInputRs.mctx not initialised")
    val sCopyMain = inp.sCopyMain ?: error("sCopyMain not initialised")
    val sCopyExtra = inp.sCopyExtra ?: error("sCopyExtra not initialised")
    return buildRs(
        s, sCopyMain, sCopyExtra,
        stateSize, nSeqs,
        kvState.getNRs(), kvState.getHead(), kvState.getSize(), kvState.getRsZ(),
        getStateRows,
    )
}

/** Load RWKV token-shift state. Port of `build_rwkv_token_shift_load`. */
fun LlmGraphContext.buildRwkvTokenShiftLoad(
    inp: LlmGraphInputRs,
    ubatchArg: LlamaUBatch,
    il: Int,
): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val mctxCur = params.mctx as? LlamaMemoryRecurrentContext
        ?: error("mctx must be LlamaMemoryRecurrentContext for rwkv_token_shift_load")

    val tokenShiftCount = hparams.tokenShiftCount
    val nSeqs = ubatchArg.nSeqs

    val tokenShiftAll = mctxCur.getRL(il) ?: error("missing rL[$il]")
    var tokenShift = buildRs(inp, tokenShiftAll, hparams.nEmbdR(), nSeqs)
    tokenShift = ggmlReshape3d(c, tokenShift, hparams.nEmbd.toLong(), tokenShiftCount.toLong(), nSeqs.toLong())

    return tokenShift
}

/** Store RWKV token-shift state. Port of `build_rwkv_token_shift_store`. */
fun LlmGraphContext.buildRwkvTokenShiftStore(
    tokenShift: GGMLTensor,
    ubatchArg: LlamaUBatch,
    il: Int,
): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val mctxCur = params.mctx as? LlamaMemoryRecurrentContext
        ?: error("mctx must be LlamaMemoryRecurrentContext for rwkv_token_shift_store")

    val tokenShiftCount = hparams.tokenShiftCount
    val nEmbdVal = hparams.nEmbd
    val nSeqs = ubatchArg.nSeqs
    val kvHead = mctxCur.getHead()
    val rL = mctxCur.getRL(il) ?: error("missing rL[$il]")

    return ggmlCpy(c,
        ggmlView1d(c, tokenShift, (nEmbdVal * nSeqs * tokenShiftCount).toLong(), 0u),
        ggmlView1d(c, rL,
            hparams.nEmbdR().toLong() * nSeqs,
            (hparams.nEmbdR().toLong() * kvHead * ggmlElementSize(rL).toLong()).toULong()))
}

// -- hybrid memory ----------------------------------------------------------

/** Build hybrid (KV + recurrent) memory inputs. Port of `build_inp_mem_hybrid`. */
fun LlmGraphContext.buildInpMemHybrid(): LlmGraphInputMemHybrid {
    val r = res ?: error("res not initialised")
    val inpRs = buildRsInp()
    val inpAttn = buildAttnInpKv()
    val inp = LlmGraphInputMemHybrid(cparams, inpAttn, inpRs)
    return r.addInputTyped(inp)
}

/** Build hybrid (K-only + recurrent) memory inputs. Port of `build_inp_mem_hybrid_k`. */
fun LlmGraphContext.buildInpMemHybridK(): LlmGraphInputMemHybridK {
    val r = res ?: error("res not initialised")
    val inpRs = buildRsInp()
    val inpAttn = buildAttnInpK()
    val inp = LlmGraphInputMemHybridK(cparams, inpAttn, inpRs)
    return r.addInputTyped(inp)
}

/** Build hybrid (ISWA + recurrent) memory inputs. Port of `build_inp_mem_hybrid_iswa`. */
fun LlmGraphContext.buildInpMemHybridIswa(): LlmGraphInputMemHybridIswa {
    val r = res ?: error("res not initialised")
    val inpRs = buildRsInp()
    val inpAttn = buildAttnInpKvIswa()
    val inp = LlmGraphInputMemHybridIswa(cparams, inpAttn, inpRs)
    return r.addInputTyped(inp)
}

// -- pooling / sampling / dense-out -----------------------------------------

/**
 * Build dense-out (final linear projections after pooling).
 * Port of `build_dense_out`.
 */
fun LlmGraphContext.buildDenseOut(
    dense2: GGMLTensor?, dense2B: GGMLTensor?,
    dense3: GGMLTensor?,
) {
    if (!cparams.embeddings || !(dense2 != null || dense2B != null || dense3 != null)) return
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val graph = gf ?: error("gf not initialised")
    var cur = r.tEmbdPooled ?: r.tEmbd ?: error("missing t_embd_pooled/t_embd")

    if (dense2 != null) {
        cur = matMul(c, dense2, cur)
    }
    if (dense2B != null) {
        cur = add(c, cur, dense2B)
    }
    if (dense3 != null) {
        cur = matMul(c, dense3, cur)
    }
    cb(cur, "result_embd_pooled", -1)
    r.tEmbdPooled = cur
    ggmlBuildForwardExpand(graph, cur)
}

/**
 * Build pooling sub-graph.
 * Port of `build_pooling`.
 */
fun LlmGraphContext.buildPooling(
    cls: GGMLTensor?, clsB: GGMLTensor?,
    clsOut: GGMLTensor?, clsOutB: GGMLTensor?,
    clsNorm: GGMLTensor?,
) {
    if (!cparams.embeddings) return
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val graph = gf ?: error("gf not initialised")
    val inp = r.tEmbd ?: error("missing result_norm/result_embd tensor")

    var cur: GGMLTensor

    when (poolingType) {
        LlamaPoolingType.NONE -> {
            cur = inp
        }
        LlamaPoolingType.MEAN -> {
            val inpMean = buildInpMean()
            cur = matMul(c, ggmlCont(c, ggmlTranspose(c, inp)), inpMean)
        }
        LlamaPoolingType.CLS, LlamaPoolingType.LAST -> {
            val inpCls = buildInpCls()
            cur = ggmlGetRows(c, inp, inpCls)
        }
        LlamaPoolingType.RANK -> {
            if (arch == LlamaModelArch.MODERN_BERT) {
                val inpMean = buildInpMean()
                cur = matMul(c, ggmlCont(c, ggmlTranspose(c, inp)), inpMean)
            } else {
                val inpCls = buildInpCls()
                cur = ggmlGetRows(c, inp, inpCls)
            }
            // Classification head
            if (cls != null) {
                cur = matMul(c, cls, cur)
                if (clsB != null) cur = add(c, cur, clsB)
                if (arch == LlamaModelArch.MODERN_BERT) {
                    cur = gelu(c, cur)
                } else {
                    cur = ggmlTanh(c, cur)
                }
                if (clsNorm != null) {
                    cur = buildNorm(cur, clsNorm, null, LlmNormType.NORM, -1)
                }
            }
            if (clsOut != null) {
                cur = matMul(c, clsOut, cur)
                if (clsOutB != null) cur = add(c, cur, clsOutB)
            }
            // Softmax for Qwen3 reranker
            if (arch == LlamaModelArch.QWEN3 || arch == LlamaModelArch.QWEN3VL) {
                cur = ggmlSoftMax(c, cur)
            }
        }
        else -> error("unknown pooling type: $poolingType")
    }

    cb(cur, "result_embd_pooled", -1)
    r.tEmbdPooled = cur
    ggmlBuildForwardExpand(graph, cur)
}

/** Build backend-sampling sub-graph. Port of `build_sampling`. */
fun LlmGraphContext.buildSampling() {
    val r = res ?: return
    val logits = r.tLogits ?: return
    val c = ctx0 ?: return
    val graph = gf ?: return

    // Backend sampling builds a sub-graph that applies per-sequence samplers.
    // The sampler backend infrastructure is not yet ported; once available:
    //   1. Pad logits with a dummy row (ggmlPad(ctx0, logits, 0, 1, 0, 0))
    //   2. For each (seqId, sampler) pair, extract logit row and call backendApply
    //   3. Wire data.sampled/probs/logits/candidates via ggmlBuildForwardSelect
    // Currently a no-op placeholder.
}

// -- private helpers --------------------------------------------------------

/**
 * Placeholder for `ggml_mul_mat_aux` – used for RoPE rotation in ISWA attention.
 * In C++ this is a specialized matrix multiply; here we approximate with matMul
 * until the dedicated op is ported.
 */
internal fun ggmlMulMatAux(c: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    return matMul(c, b, a)
}

/**
 * Build KQ mask for a given KV cache context. Placeholder that creates
 * a 4D mask tensor shaped [nKv, nTokens, 1, 1].
 */
fun LlmGraphContext.buildAttnInpKqMask(
    c: GGMLContext, cacheCtx: KVCacheContext,
    ubatchArg: LlamaUBatch, cparamsArg: LlamaCParams
): GGMLTensor {
    val nKv = cacheCtx.getNKv().toLong()
    val nTok = ubatchArg.nTokens.toLong()
    val mask = ggmlNewTensor4d(c, GGMLType.F32, nKv, nTok, 1, 1)
    ggmlSetInput(mask)
    return mask
}
