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
// Graph-context extensions for the additional input/attention types
// ---------------------------------------------------------------------------

/**
 * Extension functions on [LlmGraphContext] for the graph-building helpers
 * that depend on the additional input types declared in this file.
 *
 * These correspond to the remaining `build_*` methods in the C++ header that
 * are not yet covered by the base [LlmGraphContext] in LlmGraphTypes.kt.
 */

/** Build attention-temperature scaling input. Port of `build_inp_attn_scale`. */
fun LlmGraphContext.buildInpAttnScale(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputAttnTemp(
        nAttnTempFloorScale = hparams.nAttnTempFloorScale,
        fAttnTempScale = hparams.fAttnTempScale,
        fAttnTempOffset = hparams.fAttnTempOffset,
    )
    inp.attnScale = ggmlNewTensor1d(c, GGMLType.F32, nTokens)
    ggmlSetInput(inp.attnScale!!)
    r.addInput(inp)
    return inp.attnScale!!
}

/** Build mean-pooling input. Port of `build_inp_mean`. */
fun LlmGraphContext.buildInpMean(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputMean(cparams)
    inp.mean = ggmlNewTensor2d(c, GGMLType.F32, nTokens, nTokens)
    ggmlSetInput(inp.mean!!)
    r.addInput(inp)
    return inp.mean!!
}

/** Build CLS-token selector input. Port of `build_inp_cls`. */
fun LlmGraphContext.buildInpCls(): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputCls(cparams, arch)
    inp.cls = ggmlNewTensor1d(c, GGMLType.I32, nTokens)
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
    val nEnc = cr?.nEnc ?: 0L
    inp.crossEmbd = ggmlNewTensor2d(c, GGMLType.F32, nEmbd, nEnc)
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
    val inp = LlmGraphInputPosBucketKv(hparams)
    // In the decoder case, the bucket relates current tokens to all KV positions.
    // For now we allocate [nTokens, nTokens] as a placeholder; a real implementation
    // would use [nKv, nTokens] once KV cache context is available.
    inp.posBucket = ggmlNewTensor2d(c, GGMLType.I32, nTokens, nTokens)
    ggmlSetInput(inp.posBucket!!)
    r.addInput(inp)
    return inp.posBucket!!
}

/**
 * Compute position bias from a bucket tensor and relative-attention bias weights.
 *
 * Port of `llm_graph_context::build_pos_bias`.
 */
fun LlmGraphContext.buildPosBias(posBucket: GGMLTensor, attnRelB: GGMLTensor): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    var posBias = ggmlGetRows(c, attnRelB, posBucket)
    // Reshape to [nHead, nQ, nK, 1] pattern used by attention
    posBias = ggmlPermute(c, posBias, 2, 0, 1, 3)
    return posBias
}

/** Build V-less (K-only) attention inputs. Port of `build_attn_inp_k`. */
fun LlmGraphContext.buildAttnInpK(): LlmGraphInputAttnK {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputAttnK(hparams, cparams)
    // K-only attention still needs a KQ mask
    inp.selfKqMask = ggmlNewTensor4d(c, GGMLType.F32, nTokens, nTokens, 1, 1)
    ggmlSetInput(inp.selfKqMask!!)
    inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMask!!, GGMLType.F16)
        else inp.selfKqMask
    // K indices
    inp.selfKIdxs = ggmlNewTensor1d(c, GGMLType.I64, nTokens)
    ggmlSetInput(inp.selfKIdxs!!)
    return r.addInputTyped(inp)
}

/** Build ISWA attention inputs. Port of `build_attn_inp_kv_iswa`. */
fun LlmGraphContext.buildAttnInpKvIswa(): LlmGraphInputAttnKvIswa {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputAttnKvIswa(hparams, cparams)

    // Full-attention masks
    inp.selfKqMask = ggmlNewTensor4d(c, GGMLType.F32, nCtx, nTokens, 1, 1)
    ggmlSetInput(inp.selfKqMask!!)
    inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMask!!, GGMLType.F16)
        else inp.selfKqMask

    // SWA masks
    inp.selfKqMaskSwa = ggmlNewTensor4d(c, GGMLType.F32, nCtx, nTokens, 1, 1)
    ggmlSetInput(inp.selfKqMaskSwa!!)
    inp.selfKqMaskSwaCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMaskSwa!!, GGMLType.F16)
        else inp.selfKqMaskSwa

    // Index tensors
    inp.selfKIdxs = ggmlNewTensor1d(c, GGMLType.I64, nTokens)
    ggmlSetInput(inp.selfKIdxs!!)
    inp.selfVIdxs = ggmlNewTensor1d(c, GGMLType.I64, nTokens)
    ggmlSetInput(inp.selfVIdxs!!)
    inp.selfKIdxsSwa = ggmlNewTensor1d(c, GGMLType.I64, nTokens)
    ggmlSetInput(inp.selfKIdxsSwa!!)
    inp.selfVIdxsSwa = ggmlNewTensor1d(c, GGMLType.I64, nTokens)
    ggmlSetInput(inp.selfVIdxsSwa!!)

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
    ggmlBuildForwardExpand(graph, qCur)
    ggmlBuildForwardExpand(graph, kCur)
    ggmlBuildForwardExpand(graph, vCur)

    val kqMask = inp.getKqMask()
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
    ggmlBuildForwardExpand(graph, qCur)
    kCur?.let { ggmlBuildForwardExpand(graph, it) }
    vCur?.let { ggmlBuildForwardExpand(graph, it) }

    // Select appropriate mask depending on layer type
    val isSwa = hparams.isSwa(il)
    val kqMask = if (isSwa) inp.getKqMaskSwa() else inp.getKqMask()

    // For ISWA, K and V should be non-null (caller ensures this for the active stream)
    val k = kCur ?: error("kCur required for ISWA attention at layer $il")
    val v = vCur ?: error("vCur required for ISWA attention at layer $il")

    var cur = buildAttnMha(qCur, k, v, kqB, kqMask, sinks, vMla, kqScale, il)
    cb(cur, "kqv_out", il)

    if (wo != null) {
        cur = buildLoraMm(wo, cur, woS)
    }
    if (woB != null) {
        cur = add(c, cur, woB)
    }
    return cur
}

/** Build cross-attention inputs. Port of `build_attn_inp_cross`. */
fun LlmGraphContext.buildAttnInpCross(): LlmGraphInputAttnCross {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val inp = LlmGraphInputAttnCross(cross)
    // Cross-attention mask: [nEnc, nTokens]
    val nEnc = cross?.nEnc ?: nTokens
    inp.crossKqMask = ggmlNewTensor2d(c, GGMLType.F32, nEnc, nTokens)
    ggmlSetInput(inp.crossKqMask!!)
    inp.crossKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.crossKqMask!!, GGMLType.F16)
        else inp.crossKqMask
    return r.addInputTyped(inp)
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
    val inp = LlmGraphInputRs(mctx = null)
    // State-copy index tensor
    inp.sCopy = ggmlNewTensor1d(c, GGMLType.I32, nTokens)
    ggmlSetInput(inp.sCopy!!)
    return r.addInputTyped(inp)
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
    val c = ctx0 ?: error("ctx0 not initialised")
    val graph = gf ?: error("gf not initialised")
    // Gather states: use sCopy to scatter/gather from the state tensor
    val sIdxs = inp.sCopy ?: error("sCopy not initialised in LlmGraphInputRs")
    val gathered = ggmlGetRows(c, s, sIdxs)
    ggmlBuildForwardExpand(graph, gathered)
    return gathered
}

/** Load RWKV token-shift state. Port of `build_rwkv_token_shift_load`. */
fun LlmGraphContext.buildRwkvTokenShiftLoad(
    inp: LlmGraphInputRs,
    ubatch: LlamaUBatch,
    il: Int,
): GGMLTensor {
    val c = ctx0 ?: error("ctx0 not initialised")
    val graph = gf ?: error("gf not initialised")
    // Create a placeholder view into the recurrent state for RWKV token shifting.
    // Full implementation requires the recurrent memory context.
    val sIdxs = inp.sCopy ?: error("sCopy not initialised")
    val shift = ggmlNewTensor2d(c, GGMLType.F32, nEmbd, nTokens)
    ggmlSetInput(shift)
    ggmlBuildForwardExpand(graph, shift)
    cb(shift, "rwkv_token_shift", il)
    return shift
}

/** Store RWKV token-shift state. Port of `build_rwkv_token_shift_store`. */
fun LlmGraphContext.buildRwkvTokenShiftStore(
    tokenShift: GGMLTensor,
    ubatch: LlamaUBatch,
    il: Int,
): GGMLTensor {
    val graph = gf ?: error("gf not initialised")
    // Mark the token-shift for output so it persists for the next step
    ggmlSetOutput(tokenShift)
    ggmlBuildForwardExpand(graph, tokenShift)
    cb(tokenShift, "rwkv_token_shift_store", il)
    return tokenShift
}

// -- hybrid memory ----------------------------------------------------------

/** Build hybrid (KV + recurrent) memory inputs. Port of `build_inp_mem_hybrid`. */
fun LlmGraphContext.buildInpMemHybrid(): LlmGraphInputMemHybrid {
    val r = res ?: error("res not initialised")
    val inpAttn = buildAttnInpKv()
    val inpRs = buildRsInp()
    val inp = LlmGraphInputMemHybrid(cparams, inpAttn, inpRs)
    return r.addInputTyped(inp)
}

/** Build hybrid (K-only + recurrent) memory inputs. Port of `build_inp_mem_hybrid_k`. */
fun LlmGraphContext.buildInpMemHybridK(): LlmGraphInputMemHybridK {
    val r = res ?: error("res not initialised")
    val inpAttn = buildAttnInpK()
    val inpRs = buildRsInp()
    val inp = LlmGraphInputMemHybridK(cparams, inpAttn, inpRs)
    return r.addInputTyped(inp)
}

/** Build hybrid (ISWA + recurrent) memory inputs. Port of `build_inp_mem_hybrid_iswa`. */
fun LlmGraphContext.buildInpMemHybridIswa(): LlmGraphInputMemHybridIswa {
    val r = res ?: error("res not initialised")
    val inpAttn = buildAttnInpKvIswa()
    val inpRs = buildRsInp()
    val inp = LlmGraphInputMemHybridIswa(cparams, inpAttn, inpRs)
    return r.addInputTyped(inp)
}

// -- pooling / sampling / dense-out -----------------------------------------

/**
 * Build pooling sub-graph.
 *
 * Port of `llm_graph_context::build_pooling`.
 */
fun LlmGraphContext.buildPooling(
    cls: GGMLTensor?, clsB: GGMLTensor?,
    clsOut: GGMLTensor?, clsOutB: GGMLTensor?,
    clsNorm: GGMLTensor?,
) {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val graph = gf ?: error("gf not initialised")
    val curEmbd = r.tEmbd ?: return

    when (poolingType) {
        LlamaPoolingType.MEAN -> {
            val mean = buildInpMean()
            var pooled = matMul(c, curEmbd, mean)
            cb(pooled, "result_embd_pooled", -1)
            r.tEmbdPooled = pooled
            ggmlBuildForwardExpand(graph, pooled)
        }
        LlamaPoolingType.CLS -> {
            val clsIdx = buildInpCls()
            var pooled = ggmlGetRows(c, curEmbd, clsIdx)
            cb(pooled, "result_cls", -1)
            // Optional classification head layers
            if (cls != null) {
                pooled = buildLoraMm(cls, pooled)
                if (clsB != null) pooled = add(c, pooled, clsB)
                cb(pooled, "cls", -1)
            }
            if (clsNorm != null) {
                pooled = buildNorm(pooled, clsNorm, null, LlmNormType.NORM, -1)
                cb(pooled, "cls_norm", -1)
            }
            pooled = ggmlTanh(c, pooled)
            cb(pooled, "cls_tanh", -1)
            if (clsOut != null) {
                pooled = buildLoraMm(clsOut, pooled)
                if (clsOutB != null) pooled = add(c, pooled, clsOutB)
                cb(pooled, "cls_out", -1)
            }
            r.tEmbdPooled = pooled
            ggmlBuildForwardExpand(graph, pooled)
        }
        LlamaPoolingType.LAST -> {
            val clsIdx = buildInpCls()
            var pooled = ggmlGetRows(c, curEmbd, clsIdx)
            cb(pooled, "result_embd_pooled", -1)
            r.tEmbdPooled = pooled
            ggmlBuildForwardExpand(graph, pooled)
        }
        else -> {
            // NONE or unsupported – do nothing
        }
    }
}

/** Build backend-sampling sub-graph. Port of `build_sampling`. */
fun LlmGraphContext.buildSampling() {
    // Backend sampling builds a sub-graph that applies per-sequence samplers
    // to the logits tensor. The C++ implementation:
    //   1. Collects active samplers from tokens with output=true
    //   2. Pads logits with a dummy row (keeps graph static across activations)
    //   3. For each (seq_id, sampler) pair:
    //      - Extracts a logit row via ggml_view_1d
    //      - Calls sampler->iface->backend_apply(sampler, ctx0, gf, &data)
    //      - Wires data.sampled / data.probs / data.logits / data.candidates
    //        into res->t_sampled[seq_id] etc. using ggml_build_forward_select
    //
    // This requires:
    //   - LlmGraphContext.samplers: Map<LlamaSeqId, LlamaSampler>
    //   - LlamaSampler interface with backendApply(ctx, graph, data) method
    //   - ggmlPad, ggmlView1d, ggmlBuildForwardSelect graph-building ops
    //   - LlmGraphResult.tSampled / tSampledProbs / tSampledLogits / tCandidates maps
    //
    // Once the sampler backend infrastructure is ported, implement the logic here.
    val r = res ?: return
    if (r.tLogits == null) return
    // No samplers available on LlmGraphContext yet — nothing to build.
}

/**
 * Build dense-out (final linear projections after pooling).
 *
 * Port of `llm_graph_context::build_dense_out`.
 */
fun LlmGraphContext.buildDenseOut(
    dense2: GGMLTensor?, dense2B: GGMLTensor?,
    dense3: GGMLTensor?,
) {
    val c = ctx0 ?: error("ctx0 not initialised")
    val r = res ?: error("res not initialised")
    val graph = gf ?: error("gf not initialised")
    var cur = r.tEmbdPooled ?: return

    if (dense2 != null) {
        cur = buildLoraMm(dense2, cur)
        if (dense2B != null) cur = add(c, cur, dense2B)
        cur = relu(c, cur)
        cb(cur, "dense2", -1)
    }
    if (dense3 != null) {
        cur = buildLoraMm(dense3, cur)
        cb(cur, "dense3", -1)
    }

    r.tEmbdPooled = cur
    ggmlBuildForwardExpand(graph, cur)
}

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
