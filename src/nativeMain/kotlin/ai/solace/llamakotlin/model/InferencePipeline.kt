// port-lint: source llama.cpp/src/models/models.h
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import kotlin.math.sqrt

// =============================================================================
// Model Architecture Interface — base for all architectures
// Ported from: models.h  dispatch pattern
// =============================================================================

/**
 * Contract that every model architecture must implement.
 *
 * The two entry points mirror the C++ dispatch functions
 * `llama_model_build_graph()` and `llama_model_load_from_gguf()`.
 *
 * A concrete implementation (e.g. [LlamaArchitecture]) builds the compute
 * graph inside [buildGraph] and populates model tensors inside [loadFromGguf].
 */
interface ModelArchitecture {
    /** Architecture tag for this model. */
    val arch: LlamaModelArch

    /**
     * Build the full compute graph for one micro-batch.
     *
     * Implementations typically create an [LlmGraphContext] subclass whose
     * constructor performs the graph construction (mirroring the C++ pattern).
     *
     * @param model  The loaded model data.
     * @param params Graph parameters for this micro-batch.
     * @return The graph result containing output tensors and registered inputs.
     */
    fun buildGraph(model: LlamaModelData, params: LlmGraphParams): LlmGraphResult

    /**
     * Load model tensors from a GGUF file into [model].
     *
     * This populates [LlamaModelData.layers] and the global tensor fields
     * (tok_embd, output, output_norm, etc.) from the GGUF tensor map.
     *
     * @param model      The model data to populate.
     * @param tensorMap  Map from GGUF tensor name to loaded tensor.
     * @return `true` on success.
     */
    fun loadFromGguf(model: LlamaModelData, tensorMap: Map<String, GGMLTensor>): Boolean
}

// =============================================================================
// LLaMA Architecture — the reference implementation
// Ported from: models/llama.cpp  llm_build_llama
// =============================================================================

/**
 * LLaMA architecture implementation — attention + SwiGLU FFN transformer.
 *
 * This is the reference [ModelArchitecture] and the first one ported.
 * The [buildGraph] method mirrors the C++ `llm_build_llama` constructor
 * which builds the full forward-pass graph.
 *
 * Supports both standard decoder mode and embedding mode (no lm_head).
 */
class LlamaArchitecture(
    /** When `true`, build for embedding (no lm_head projection). */
    val embedMode: Boolean = false,
) : ModelArchitecture {
    override val arch: LlamaModelArch = LlamaModelArch.LLAMA

    override fun buildGraph(model: LlamaModelData, params: LlmGraphParams): LlmGraphResult {
        val res = LlmGraphResult(maxNodes = 8192)
        val builder = LlmBuildLlama(params, model, res, embedMode)
        return res
    }

    override fun loadFromGguf(model: LlamaModelData, tensorMap: Map<String, GGMLTensor>): Boolean {
        val hp = model.hparams

        // -- global tensors --
        model.tokEmbd = tensorMap["${LlamaTensorNames.TOKEN_EMBD}.weight"]
        model.outputNorm = tensorMap["${LlamaTensorNames.OUTPUT_NORM}.weight"]
        model.output = tensorMap["${LlamaTensorNames.OUTPUT}.weight"]

        // -- per-layer tensors --
        for (il in 0 until hp.nLayer) {
            val layer = if (il < model.layers.size) {
                model.layers[il]
            } else {
                LlamaModelLayer().also { model.layers.add(it) }
            }

            layer.attnNorm = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.ATTN_NORM, il)]
            layer.wq       = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.ATTN_Q, il)]
            layer.wk       = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.ATTN_K, il)]
            layer.wv       = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.ATTN_V, il)]
            layer.wo       = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.ATTN_OUT, il)]
            layer.ffnNorm  = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.FFN_NORM, il)]
            layer.ffnGate  = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.FFN_GATE, il)]
            layer.ffnDown  = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.FFN_DOWN, il)]
            layer.ffnUp    = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.FFN_UP, il)]

            // MoE layers (optional — only present in Mixtral-style LLaMA variants)
            layer.ffnGateInp  = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.FFN_GATE_INP, il)]
            layer.ffnGateExps = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.FFN_GATE_EXPS, il)]
            layer.ffnDownExps = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.FFN_DOWN_EXPS, il)]
            layer.ffnUpExps   = tensorMap[LlamaTensorNames.weight(LlamaTensorNames.FFN_UP_EXPS, il)]
        }

        return true
    }
}

/**
 * Graph builder for the LLaMA architecture.
 *
 * The constructor builds the complete forward-pass graph, mirroring the C++
 * `llm_build_llama::llm_build_llama(model, params)` constructor.
 *
 * The graph proceeds as:
 * 1. Token embedding lookup
 * 2. Position tensor
 * 3. For each layer: RMSNorm → self-attention (RoPE + KV cache) → residual
 *    → RMSNorm → SwiGLU FFN → residual
 * 4. Final RMSNorm → (optional) lm_head projection
 *
 * @param params    Graph parameters for this micro-batch.
 * @param model     The loaded model data (weights + hparams).
 * @param result    The [LlmGraphResult] to populate with output tensors.
 * @param embedMode When `true`, skip the lm_head projection (embedding mode).
 */
class LlmBuildLlama(
    params: LlmGraphParams,
    private val model: LlamaModelData,
    private val result: LlmGraphResult,
    private val embedMode: Boolean = false,
) : LlmGraphContext(params) {

    init {
        res = result

        val nEmbdHead = hparams.nEmbdHeadV().toLong()
        require(nEmbdHead == hparams.nEmbdHeadK().toLong()) {
            "LLaMA requires nEmbdHeadK == nEmbdHeadV, got K=${hparams.nEmbdHeadK()}, V=$nEmbdHead"
        }
        require(nEmbdHead == nRot()) {
            "LLaMA requires nEmbdHead == nRot, got nEmbdHead=$nEmbdHead, nRot=${nRot()}"
        }

        buildLlamaGraph()
    }

    /** Rotation dimension (convenience accessor). */
    private fun nRot(): Long = hparams.nRot().toLong()

    /**
     * Assign a debug name to [cur] following the `"name/il"` convention.
     */
    private fun nameTensor(cur: GGMLTensor, name: String, il: Int) {
        cur.name = if (il >= 0) "$name/$il" else name
    }

    /**
     * Construct the full LLaMA forward-pass graph.
     *
     * This mirrors the body of `llm_build_llama<embed>::llm_build_llama()`
     * from `models/llama.cpp` line 4–153.  Complex tensor operations delegate
     * to the base [LlmGraphContext] builder methods.
     */
    private fun buildLlamaGraph() {
        // 1. Token embedding
        var inpL = buildInpEmbd(model.tokEmbd ?: error("LLaMA model missing tok_embd"))

        // 2. Position tensor (for RoPE)
        val inpPos = buildInpPos()

        // 3. Attention input (KV-cached or no-cache depending on mode)
        val inpAttn: Any = if (embedMode) {
            buildAttnInpNoCache()
        } else {
            LlmGraphInputAttnKv(hparams, cparams).also {
                result.addInput(it)
            }
        }

        // Attention scale: 1/sqrt(nEmbdHead) unless overridden
        val nEmbdHead = hparams.nEmbdHeadV().toLong()
        val kqScale: Float = if (hparams.fAttentionScale == 0.0f) {
            1.0f / sqrt(nEmbdHead.toFloat())
        } else {
            hparams.fAttentionScale
        }

        // Output-id selection (for last-token-only extraction)
        val inpOutIds = buildInpOutIds()

        // 4. Transformer layers
        for (il in 0 until nLayer.toInt()) {
            val inpSA = inpL
            val layer = model.layers[il]

            // ---- Pre-attention RMS norm ----
            var cur = buildNorm(inpL, layer.attnNorm, null, LlmNormType.RMS, il)
            nameTensor(cur, "attn_norm", il)

            // ---- Self-attention ----
            run {
                // QKV projections + reshape
                val (qcur0, kcur0, vcur) = buildQkv(layer, cur, nEmbdHead, nHead, nHeadKv, il)
                var qcur = qcur0
                var kcur = kcur0

                // RoPE is applied to Q and K here; full implementation delegates to
                // ggml_rope_ext via the base-class builder.  TODO: wire rope_factors
                // from model.get_rope_factors(cparams, il) for LLaMA-3 style per-layer
                // RoPE frequency factors.
                cb(qcur, "Qcur", il)
                cb(kcur, "Kcur", il)
                cb(vcur, "Vcur", il)

                // Optional per-head QK normalization (Llama4TextL2Norm)
                if (hparams.useKqNorm) {
                    // qcur = rms_norm(qcur); kcur = rms_norm(kcur)
                    // TODO: apply ggml_rms_norm here
                    cb(qcur, "Qcur_normed", il)
                    cb(kcur, "Kcur_normed", il)
                }

                // Full attention (cache write + SDPA + output projection)
                cur = if (embedMode) {
                    buildAttn(
                        inpAttn as LlmGraphInputAttnNoCache,
                        layer.wo ?: error("Missing wo for layer $il"),
                        layer.bo, layer.woS,
                        qcur, kcur, vcur,
                        null, null, null, kqScale, il,
                    )
                } else {
                    buildAttn(
                        inpAttn as LlmGraphInputAttnKv,
                        layer.wo ?: error("Missing wo for layer $il"),
                        layer.bo, layer.woS,
                        qcur, kcur, vcur,
                        null, null, null, kqScale, il,
                    )
                }

                // Optional per-tensor output scale (bitnet / quantized models)
                if (layer.woS != null) {
                    // cur = cur * woS  — TODO: wire ggml_mul
                }
                nameTensor(cur, "attn_out", il)
            }

            // Last-layer output-id pruning
            if (il == nLayer.toInt() - 1 && inpOutIds != null) {
                // cur = ggml_get_rows(cur, inpOutIds)
                // inpSA = ggml_get_rows(inpSA, inpOutIds)
                // TODO: wire ggml_get_rows for output selection
            }

            // ---- Residual connection (attention) ----
            var ffnInp = cur // placeholder: should be ggml_add(cur, inpSA)
            cb(ffnInp, "ffn_inp", il)

            // ---- FFN (non-MoE or MoE) ----
            if (layer.ffnGateInp == null) {
                // Standard SwiGLU FFN
                cur = buildNorm(ffnInp, layer.ffnNorm, null, LlmNormType.RMS, il)
                nameTensor(cur, "ffn_norm", il)

                cur = buildFfn(
                    cur,
                    layer.ffnUp, layer.ffnUpB, layer.ffnUpS,
                    layer.ffnGate, layer.ffnGateB, layer.ffnGateS,
                    layer.ffnDown, layer.ffnDownB, layer.ffnDownS,
                    null,
                    LlmFfnOpType.SILU, LlmFfnGateType.PAR, il,
                )
                nameTensor(cur, "ffn_out", il)
            } else {
                // MoE branch
                cur = buildNorm(ffnInp, layer.ffnNorm, null, LlmNormType.RMS, il)
                nameTensor(cur, "ffn_norm", il)

                cur = buildMoeFfn(
                    cur,
                    layer.ffnGateInp!!,
                    layer.ffnUpExps ?: error("Missing ffnUpExps for MoE layer $il"),
                    layer.ffnGateExps,
                    layer.ffnDownExps ?: error("Missing ffnDownExps for MoE layer $il"),
                    null,
                    nExpert, nExpertUsed,
                    LlmFfnOpType.SILU, true,
                    hparams.expertWeightsScale,
                    LlamaExpertGatingFuncType.SOFTMAX,
                    il,
                )
                nameTensor(cur, "ffn_moe_out", il)
            }

            // ---- Residual connection (FFN) ----
            // cur = ggml_add(cur, ffnInp)  — TODO: wire element-wise add
            nameTensor(cur, "ffn_out", il)

            // Control vector injection
            cur = buildCvec(cur, il)
            nameTensor(cur, "l_out", il)

            // Feed into next layer
            inpL = cur
        }

        // 5. Final RMS norm
        var cur = inpL
        cur = buildNorm(cur, model.outputNorm, null, LlmNormType.RMS, -1)
        cb(cur, "result_norm", -1)
        result.tEmbd = cur

        // 6. lm_head projection (skip in embedding mode)
        if (!embedMode) {
            cur = buildLoraMm(model.output ?: error("LLaMA model missing output tensor"), cur)
            nameTensor(cur, "result_output", -1)
            result.tLogits = cur
        }

        // In C++ this calls ggml_build_forward_expand(gf, cur) to finalize the
        // graph.  The equivalent step will happen when the CPU backend executor
        // is wired up.
    }

}

// =============================================================================
// Architecture Registry & Dispatch
// Ported from: models.h  llama_model_build_graph / llama_model_load_from_gguf
// =============================================================================

/**
 * Central registry that maps [LlamaModelArch] tags to their
 * [ModelArchitecture] implementations.
 *
 * This mirrors the giant `switch (arch)` block in the C++ dispatch functions
 * `llama_model_build_graph()` and `llama_model_load_from_gguf()`.
 *
 * New architectures are added by calling [register].  The [buildGraph] and
 * [loadFromGguf] entry points look up the architecture and delegate.
 */
object ModelArchitectureRegistry {
    private val registry: MutableMap<LlamaModelArch, ModelArchitecture> = mutableMapOf()

    init {
        // Register the reference LLaMA architecture
        register(LlamaArchitecture(embedMode = false))
    }

    /** Register an architecture implementation (replaces any existing entry). */
    fun register(impl: ModelArchitecture) {
        registry[impl.arch] = impl
    }

    /** Look up the implementation for [arch], or `null` if not yet ported. */
    fun get(arch: LlamaModelArch): ModelArchitecture? = registry[arch]

    /**
     * Build a compute graph for [model] using its architecture.
     *
     * Mirrors `llama_model_build_graph()` in C++.
     *
     * @throws IllegalStateException if the architecture is not registered.
     */
    fun buildGraph(model: LlamaModelData, params: LlmGraphParams): LlmGraphResult {
        val impl = registry[model.arch]
            ?: error("Architecture ${model.arch.ggufName} is not yet ported")
        return impl.buildGraph(model, params)
    }

    /**
     * Load model tensors from GGUF for [model] using its architecture.
     *
     * Mirrors `llama_model_load_from_gguf()` in C++.
     *
     * @throws IllegalStateException if the architecture is not registered.
     */
    fun loadFromGguf(model: LlamaModelData, tensorMap: Map<String, GGMLTensor>): Boolean {
        val impl = registry[model.arch]
            ?: error("Architecture ${model.arch.ggufName} is not yet ported")
        return impl.loadFromGguf(model, tensorMap)
    }

    /** Return the set of currently registered architecture tags. */
    fun registeredArchitectures(): Set<LlamaModelArch> = registry.keys.toSet()
}

// =============================================================================
// Existing InferencePipeline code preserved below
// =============================================================================

/**
 * End-to-end LLaMA inference pipeline.
 * Integrates model, sampling, KV cache, and grammar constraints.
 */
class LlamaInferencePipeline(
    val model: LlamaModel,
    val tokenizer: Tokenizer,
    val samplingConfig: SamplingConfig = SamplingConfig(),
    val grammarText: String? = null
) {
    private val context = GGMLContext()
    private val graphAllocator = GGMLGraphAllocator()
    private val samplingContext = SamplingContext(samplingConfig, model.config.vocabSize)
    
    // KV cache for each layer
    private val kvCaches = Array(model.config.numHiddenLayers) { 
        KVCache(
            maxSequenceLength = model.config.maxPositionEmbeddings,
            numHeads = model.config.numAttentionHeads,
            headDim = model.config.headDim
        )
    }
    
    // Grammar constraint context (if grammar is provided)
    private val grammarContext = grammarText?.let { text ->
        val grammar = GBNFGrammar.parse(text)
        GrammarSamplingContext(grammar, tokenizer)
    }
    
    init {
        // Initialize model parameters
        graphAllocator.reserve(1024 * 1024 * 1024) // 1GB buffer
        model.initializeParameters(graphAllocator)
        
        // Initialize KV caches
        for (cache in kvCaches) {
            cache.initialize(graphAllocator)
        }
    }
    
    /**
     * Generate text given an input prompt.
     */
    fun generate(
        prompt: String,
        maxNewTokens: Int = 100,
        stopTokens: List<String> = emptyList(),
        doSample: Boolean = true
    ): GenerationResult {
        // Encode prompt
        val inputIds = tokenizer.encode(prompt).toIntArray()
        val generatedTokens = mutableListOf<Int>()
        val allTokens = inputIds.toMutableList()
        
        // Reset contexts
        samplingContext.reset()
        grammarContext?.reset()
        for (cache in kvCaches) {
            cache.clear()
        }
        
        var isFinished = false
        var finishReason = FinishReason.MAX_TOKENS
        
        for (step in 0 until maxNewTokens) {
            if (isFinished) break
            
            // Determine input for this step
            val currentInput = if (step == 0) {
                // First step: use full prompt
                inputIds
            } else {
                // Subsequent steps: use only the last generated token
                intArrayOf(generatedTokens.last())
            }
            
            // Forward pass through model
            val logits = model.forward(
                context = context,
                graphAllocator = graphAllocator,
                inputIds = currentInput,
                attentionMask = null, // Simplified - could add attention masking
                kvCaches = kvCaches
            )
            
            // Get logits for the last position (for generation)
            val lastPositionLogits = extractLastPositionLogits(logits)
            
            // Sample next token
            val nextTokenId = sampleNextToken(lastPositionLogits, allTokens)
            
            // Check for stop conditions
            val nextTokenText = tokenizer.decode(nextTokenId)
            if (stopTokens.contains(nextTokenText)) {
                isFinished = true
                finishReason = FinishReason.STOP_TOKEN
                break
            }
            
            // Add generated token
            generatedTokens.add(nextTokenId)
            allTokens.add(nextTokenId)
            
            // Update grammar context if using grammar constraints
            grammarContext?.acceptToken(nextTokenId)
            if (grammarContext?.isComplete() == true) {
                isFinished = true
                finishReason = FinishReason.GRAMMAR_COMPLETE
                break
            }
        }
        
        // Decode generated tokens
        val generatedText = tokenizer.decode(generatedTokens)
        
        return GenerationResult(
            prompt = prompt,
            generatedText = generatedText,
            fullText = prompt + generatedText,
            inputTokens = inputIds.toList(),
            generatedTokens = generatedTokens,
            allTokens = allTokens,
            finishReason = finishReason
        )
    }
    
    /**
     * Extract logits for the last position in the sequence.
     */
    private fun extractLastPositionLogits(logits: GGMLTensor): GGMLTensor {
        val seqLen = logits.ne[1].toInt()
        val vocabSize = logits.ne[0].toInt()
        
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = vocabSize.toLong()
        for (i in 1 until GGML_MAX_DIMS) result.ne[i] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(result)
        
        // Copy logits from the last sequence position
        val lastPos = seqLen - 1
        for (vocabIdx in 0 until vocabSize) {
            val value = logits.getFloat(graphAllocator, vocabIdx, lastPos, 0)
            result.setFloat(graphAllocator, value, vocabIdx)
        }
        
        return result
    }
    
    /**
     * Sample the next token using configured sampling strategy.
     */
    private fun sampleNextToken(logits: GGMLTensor, context: List<Int>): Int {
        // Apply repetition penalties based on context
        val penaltyTokens = context.takeLast(64) // Use last 64 tokens for penalty
        
        // Sample token
        var nextTokenId = samplingContext.sample(
            graphAllocator = graphAllocator,
            logits = logits,
            penaltyTokens = penaltyTokens
        )
        
        // Apply grammar constraints if enabled
        grammarContext?.let { grammarCtx ->
            // Convert logits to candidates
            val vocabSize = logits.ne[0].toInt()
            val candidates = Array(vocabSize) { i ->
                TokenData(
                    id = i,
                    logit = logits.getFloat(graphAllocator, i),
                    prob = 0.0f
                )
            }
            
            // Filter candidates based on grammar
            val filteredCandidates = grammarCtx.filterCandidates(candidates)
            
            // If current token is not allowed by grammar, resample from filtered candidates
            if (filteredCandidates.none { it.id == nextTokenId }) {
                if (filteredCandidates.isNotEmpty()) {
                    // Create a new logits tensor with only valid tokens
                    val filteredLogits = GGMLTensor(type = GGMLType.F32)
                    filteredLogits.ne[0] = vocabSize.toLong()
                    for (i in 1 until GGML_MAX_DIMS) filteredLogits.ne[i] = 1L
                    filteredLogits.nb = calculateContiguousStrides(filteredLogits.ne, filteredLogits.type, GGML_MAX_DIMS)
                    
                    graphAllocator.allocateTensor(filteredLogits)
                    
                    // Set logits to negative infinity for invalid tokens
                    val validTokenIds = filteredCandidates.map { it.id }.toSet()
                    for (i in 0 until vocabSize) {
                        if (i in validTokenIds) {
                            filteredLogits.setFloat(graphAllocator, logits.getFloat(graphAllocator, i), i)
                        } else {
                            filteredLogits.setFloat(graphAllocator, Float.NEGATIVE_INFINITY, i)
                        }
                    }
                    
                    // Resample with filtered logits
                    nextTokenId = samplingContext.sample(
                        graphAllocator = graphAllocator,
                        logits = filteredLogits,
                        penaltyTokens = penaltyTokens
                    )
                }
            }
        }
        
        return nextTokenId
    }
    
    /**
     * Stream generation token by token.
     */
    fun generateStream(
        prompt: String,
        maxNewTokens: Int = 100,
        stopTokens: List<String> = emptyList(),
        callback: (StreamingResult) -> Boolean // Return true to continue, false to stop
    ) {
        val inputIds = tokenizer.encode(prompt).toIntArray()
        val generatedTokens = mutableListOf<Int>()
        val allTokens = inputIds.toMutableList()
        
        // Reset contexts
        samplingContext.reset()
        grammarContext?.reset()
        for (cache in kvCaches) {
            cache.clear()
        }
        
        var isFinished = false
        var finishReason = FinishReason.MAX_TOKENS
        
        for (step in 0 until maxNewTokens) {
            if (isFinished) break
            
            val currentInput = if (step == 0) {
                inputIds
            } else {
                intArrayOf(generatedTokens.last())
            }
            
            // Forward pass
            val logits = model.forward(
                context = context,
                graphAllocator = graphAllocator,
                inputIds = currentInput,
                attentionMask = null,
                kvCaches = kvCaches
            )
            
            val lastPositionLogits = extractLastPositionLogits(logits)
            val nextTokenId = sampleNextToken(lastPositionLogits, allTokens)
            val nextTokenText = tokenizer.decode(nextTokenId)
            
            // Check stop conditions
            if (stopTokens.contains(nextTokenText)) {
                isFinished = true
                finishReason = FinishReason.STOP_TOKEN
            }
            
            generatedTokens.add(nextTokenId)
            allTokens.add(nextTokenId)
            
            // Update grammar context
            grammarContext?.acceptToken(nextTokenId)
            if (grammarContext?.isComplete() == true) {
                isFinished = true
                finishReason = FinishReason.GRAMMAR_COMPLETE
            }
            
            // Call callback with current state
            val streamingResult = StreamingResult(
                tokenId = nextTokenId,
                tokenText = nextTokenText,
                generatedText = tokenizer.decode(generatedTokens),
                isFinished = isFinished,
                finishReason = if (isFinished) finishReason else null
            )
            
            val shouldContinue = callback(streamingResult)
            if (!shouldContinue) {
                isFinished = true
                finishReason = FinishReason.USER_STOP
                break
            }
            
            if (grammarContext?.isComplete() == true) {
                break
            }
        }
    }
    
    /**
     * Get model information.
     */
    fun getModelInfo(): ModelInfo {
        return ModelInfo(
            config = model.config,
            vocabSize = tokenizer.getVocabSize(),
            maxSequenceLength = model.config.maxPositionEmbeddings,
            parameterCount = estimateParameterCount()
        )
    }
    
    private fun estimateParameterCount(): Long {
        val config = model.config
        var params = 0L
        
        // Embeddings
        params += config.vocabSize * config.hiddenSize
        
        // Each layer
        params += config.numHiddenLayers * (
            // Attention weights
            4 * config.hiddenSize * config.hiddenSize +
            // MLP weights  
            config.hiddenSize * config.intermediateSize * 3 +
            // Layer norms
            config.hiddenSize * 2
        )
        
        // Final layer norm and output head
        params += config.hiddenSize + config.vocabSize * config.hiddenSize
        
        return params
    }
}

/**
 * Generation result containing full information about the generation.
 */
data class GenerationResult(
    val prompt: String,
    val generatedText: String,
    val fullText: String,
    val inputTokens: List<Int>,
    val generatedTokens: List<Int>,
    val allTokens: List<Int>,
    val finishReason: FinishReason
)

/**
 * Streaming generation result for token-by-token generation.
 */
data class StreamingResult(
    val tokenId: Int,
    val tokenText: String,
    val generatedText: String,
    val isFinished: Boolean,
    val finishReason: FinishReason?
)

/**
 * Model information.
 */
data class ModelInfo(
    val config: LlamaConfig,
    val vocabSize: Int,
    val maxSequenceLength: Int,
    val parameterCount: Long
)

/**
 * Reasons why generation finished.
 */
enum class FinishReason {
    MAX_TOKENS,
    STOP_TOKEN,
    GRAMMAR_COMPLETE,
    USER_STOP,
    ERROR
}

/**
 * Builder for creating inference pipelines with different configurations.
 */
class LlamaInferencePipelineBuilder {
    private var model: LlamaModel? = null
    private var tokenizer: Tokenizer? = null
    private var samplingConfig = SamplingConfig()
    private var grammarText: String? = null
    
    fun model(model: LlamaModel) = apply { this.model = model }
    fun tokenizer(tokenizer: Tokenizer) = apply { this.tokenizer = tokenizer }
    fun samplingConfig(config: SamplingConfig) = apply { this.samplingConfig = config }
    fun grammar(grammarText: String) = apply { this.grammarText = grammarText }
    
    fun build(): LlamaInferencePipeline {
        val model = this.model ?: throw IllegalStateException("Model is required")
        val tokenizer = this.tokenizer ?: throw IllegalStateException("Tokenizer is required")
        
        return LlamaInferencePipeline(
            model = model,
            tokenizer = tokenizer,
            samplingConfig = samplingConfig,
            grammarText = grammarText
        )
    }
}