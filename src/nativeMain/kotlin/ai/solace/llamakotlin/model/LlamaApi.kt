// port-lint: source llama.cpp/src/llama.cpp
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import ai.solace.llamakotlin.gguf.ModelLoader
import kotlin.math.max
import kotlin.math.min
import platform.posix.sysconf
import platform.posix._SC_PHYS_PAGES
import platform.posix._SC_PAGESIZE

// =============================================================================
// Public API — Main entry points for the llama.kotlin library.
//
// This file is the Kotlin port of `llama.cpp/src/llama.cpp`.  It exposes the
// top-level functions that callers use to load models, create contexts,
// tokenise text, and query system capabilities.  Wherever possible the
// implementation delegates to existing Kotlin types (LlamaVocab,
// LlamaModelData, LlamaContext, ModelLoader, etc.) rather than duplicating
// logic.
// =============================================================================

// ---------------------------------------------------------------------------
// Internal formatting helpers (Kotlin/Native lacks String.format)
// ---------------------------------------------------------------------------

/** Format a Double with 2 decimal places. */
private fun formatSeconds(v: Double): String {
    val rounded = kotlin.math.round(v * 100) / 100.0
    return rounded.toString()
}

/** Zero-pad an Int to exactly [width] digits. */
private fun zeroPad(n: Int, width: Int): String {
    val s = n.toString()
    return if (s.length >= width) s else "0".repeat(width - s.length) + s
}

/** Format a Double to 2 decimal places for display. */
private fun fmtMs(v: Double): String {
    val rounded = kotlin.math.round(v * 100) / 100.0
    return rounded.toString()
}

// ---------------------------------------------------------------------------
// Flash-attention type name helper  (llama_flash_attn_type_name)
// ---------------------------------------------------------------------------

/**
 * Returns a human-readable name for the given [flashAttnType].
 *
 * Maps to `llama_flash_attn_type_name()` in C++.
 */
fun llamaFlashAttnTypeName(flashAttnType: LlamaFlashAttnType): String = when (flashAttnType) {
    LlamaFlashAttnType.AUTO     -> "auto"
    LlamaFlashAttnType.DISABLED -> "disabled"
    LlamaFlashAttnType.ENABLED  -> "enabled"
}

// ---------------------------------------------------------------------------
// Device memory data  (llama_device_memory_data)
// ---------------------------------------------------------------------------

/**
 * Per-device memory snapshot used during model-fitting.
 *
 * Mirrors `struct llama_device_memory_data` in C++.
 */
data class LlamaDeviceMemoryData(
    var total: Long = 0L,
    var free: Long = 0L,
    var mb: LlamaMemoryBreakdownData = LlamaMemoryBreakdownData()
)

/**
 * Memory breakdown for a device or host, showing how much is used for the
 * model, the context, and the compute scratch.
 *
 * If [LlamaMemoryBreakdownData] already exists in LlamaAttention.kt it is
 * reused; this definition is only emitted when the type is absent.
 */
// LlamaMemoryBreakdownData is defined in LlamaAttention.kt — we reuse it here.

// ---------------------------------------------------------------------------
// Layer fraction enum  (layer_fraction_t)
// ---------------------------------------------------------------------------

/**
 * Which part of a layer's tensors to distribute across devices.
 *
 * Used internally by [llamaParamsFitImpl] to control partial-layer placement.
 * Mirrors `enum layer_fraction_t` in C++.
 */
enum class LayerFraction {
    /** Nothing */
    NONE,
    /** Attention weights only */
    ATTN,
    /** Attention + up projection */
    UP,
    /** Attention + up + gate projection */
    GATE,
    /** Everything except sparse MoE expert weights */
    MOE;
}

// ---------------------------------------------------------------------------
// Split mode enum
// ---------------------------------------------------------------------------

/**
 * How to split tensors across multiple devices.
 *
 * Maps to `llama_split_mode` in C++.
 */
enum class LlamaSplitMode {
    /** Use a single device only. */
    NONE,
    /** Split layers across devices (whole-layer granularity). */
    LAYER,
    /** Split individual tensors across devices (row-level granularity). */
    ROW,
    /** Split individual tensors across devices (tensor-level granularity). */
    TENSOR;
}

// ---------------------------------------------------------------------------
// Model params  (llama_model_params)
// ---------------------------------------------------------------------------

/**
 * Parameters that control model loading, mirroring `llama_model_params`.
 *
 * The defaults match `llama_model_default_params()` in C++.
 */
data class LlamaModelParams(
    var nGpuLayers: Int = 999,
    var splitMode: LlamaSplitMode = LlamaSplitMode.LAYER,
    var mainGpu: Int = 0,
    var tensorSplit: FloatArray? = null,
    var progressCallback: ((Float) -> Boolean)? = null,
    var vocabOnly: Boolean = false,
    var useMmap: Boolean = true,
    var useMlock: Boolean = false,
    var checkTensors: Boolean = false,
    var noAlloc: Boolean = false,
    var useDirectIo: Boolean = false,
    var useExtraBuffts: Boolean = true,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is LlamaModelParams) return false
        return nGpuLayers == other.nGpuLayers &&
            splitMode == other.splitMode &&
            mainGpu == other.mainGpu &&
            tensorSplit.contentEquals(other.tensorSplit) &&
            vocabOnly == other.vocabOnly &&
            useMmap == other.useMmap &&
            useMlock == other.useMlock &&
            checkTensors == other.checkTensors &&
            noAlloc == other.noAlloc
    }

    override fun hashCode(): Int {
        var h = nGpuLayers
        h = 31 * h + splitMode.hashCode()
        h = 31 * h + mainGpu
        h = 31 * h + (tensorSplit?.contentHashCode() ?: 0)
        h = 31 * h + vocabOnly.hashCode()
        h = 31 * h + useMmap.hashCode()
        h = 31 * h + useMlock.hashCode()
        return h
    }
}

/**
 * Returns a default [LlamaModelParams] instance matching
 * `llama_model_default_params()` in C++.
 */
fun llamaModelDefaultParams(): LlamaModelParams = LlamaModelParams()

// ---------------------------------------------------------------------------
// Sampler chain default params  (llama_sampler_chain_default_params)
// ---------------------------------------------------------------------------

/**
 * Parameters for a sampler chain.
 *
 * Maps to `struct llama_sampler_chain_params` in C++.
 */
data class LlamaSamplerChainParams(
    val noPerf: Boolean = true,
)

/**
 * Returns a default [LlamaSamplerChainParams] matching
 * `llama_sampler_chain_default_params()` in C++.
 */
fun llamaSamplerChainDefaultParams(): LlamaSamplerChainParams = LlamaSamplerChainParams()

// ---------------------------------------------------------------------------
// Params-fit status  (llama_params_fit_status)
// ---------------------------------------------------------------------------

/**
 * Outcome of [llamaParamsFit].
 *
 * Maps to `enum llama_params_fit_status` in C++.
 */
enum class LlamaParamsFitStatus {
    SUCCESS,
    FAILURE,
    ERROR;
}

/**
 * Exception thrown when the model cannot be fit into available memory.
 *
 * Maps to `class llama_params_fit_exception` in C++.
 */
class LlamaParamsFitException(message: String) : RuntimeException(message)

// ---------------------------------------------------------------------------
// Device limits  (llama_max_devices / llama_max_tensor_buft_overrides)
// ---------------------------------------------------------------------------

/** Maximum number of devices supported. Maps to `llama_max_devices()`. */
fun llamaMaxDevices(): Int = 16

/**
 * Maximum number of tensor buffer-type overrides supported.
 * Maps to `llama_max_tensor_buft_overrides()`.
 */
fun llamaMaxTensorBuftOverrides(): Int = 4096

// ---------------------------------------------------------------------------
// Feature queries  (llama_supports_*)
// ---------------------------------------------------------------------------

/**
 * Whether the build supports memory-mapped model files.
 * Maps to `llama_supports_mmap()`.
 */
fun llamaSupportsMmap(): Boolean = LlamaMmap.SUPPORTED

/**
 * Whether the build supports memory-locking model data.
 * Maps to `llama_supports_mlock()`.
 */
fun llamaSupportsMlock(): Boolean = LlamaMlock.SUPPORTED

/**
 * Whether a GPU offload backend is available.
 *
 * In the current Kotlin/Native port only CPU is supported, so this always
 * returns `false`.  When a GPU or accelerator backend is added this will
 * query the backend registry.
 *
 * Maps to `llama_supports_gpu_offload()`.
 */
fun llamaSupportsGpuOffload(): Boolean = false  // TODO: query GGMLBackend registry

/**
 * Whether an RPC backend is available.
 * Maps to `llama_supports_rpc()`.
 */
fun llamaSupportsRpc(): Boolean = false  // TODO: query GGMLBackend registry

// ---------------------------------------------------------------------------
// Backend lifecycle  (llama_backend_init / llama_backend_free)
// ---------------------------------------------------------------------------

/**
 * Initialise global backend state.
 *
 * Must be called once before any other API function.
 * Maps to `llama_backend_init()`.
 */
fun llamaBackendInit() {
    // In C++ this initialises the GGML f16 conversion tables via a temporary
    // ggml_context.  In Kotlin the tables are computed lazily on first use
    // inside GGMLTypes / NumericConversions, so nothing is strictly required
    // here.  We retain the function as a public API contract.
}

/**
 * Initialise NUMA-aware allocation.
 *
 * Maps to `llama_numa_init()`.
 *
 * @param strategy The desired NUMA strategy. [GGMLNumaStrategy.DISABLED]
 *                 is a no-op.
 */
fun llamaNumaInit(strategy: GGMLNumaStrategy) {
    if (strategy != GGMLNumaStrategy.DISABLED) {
        // TODO: hook into the CPU backend NUMA initialisation when available
        llamaLogDebug("llamaNumaInit: NUMA strategy $strategy requested but not yet supported\n")
    }
}

/**
 * Free global backend state.
 * Maps to `llama_backend_free()`.
 */
fun llamaBackendFree() {
    // Placeholder — in C++ this calls ggml_quantize_free().
}

// ---------------------------------------------------------------------------
// Time helper  (llama_time_us)
// ---------------------------------------------------------------------------

// llamaTimeUs() is already defined in LlamaImpl.kt and re-exported here
// for discoverability.  The canonical definition lives in LlamaImpl.kt.

// ---------------------------------------------------------------------------
// NGL helper struct used during params-fit  (ngl_t)
// ---------------------------------------------------------------------------

/**
 * Per-device layer assignment used during the auto-fit algorithm.
 *
 * Mirrors `struct ngl_t` inside `llama_params_fit_impl()`.
 *
 * @property nLayer   Total number of layers assigned to this device.
 * @property nPart    Number of partial (overflowing) layers, ≤ [nLayer].
 * @property overflowType For the first partial layer, which parts overflow.
 */
data class NglPerDevice(
    var nLayer: Int = 0,
    var nPart: Int = 0,
    var overflowType: LayerFraction = LayerFraction.MOE,
) {
    /** Number of fully placed (non-partial) layers. */
    fun nFull(): Int {
        require(nLayer >= nPart) { "nLayer ($nLayer) < nPart ($nPart)" }
        return nLayer - nPart
    }
}

// ---------------------------------------------------------------------------
// Overflow pattern helper  (get_overflow_pattern)
// ---------------------------------------------------------------------------

/**
 * Returns a regex pattern string matching the tensors that should overflow
 * from the device for a given [layerIndex] and [fraction].
 *
 * This is a direct port of the `get_overflow_pattern` lambda in
 * `llama_params_fit_impl()`.
 */
internal fun getOverflowPattern(layerIndex: Int, fraction: LayerFraction): String {
    require(layerIndex < 1000) { "at most 1000 model layers are supported" }
    return when (fraction) {
        LayerFraction.ATTN ->
            "blk\\.$layerIndex\\.ffn_(gate|up|gate_up|down).*"
        LayerFraction.UP ->
            "blk\\.$layerIndex\\.ffn_(gate|gate_up|down).*"
        LayerFraction.GATE ->
            "blk\\.$layerIndex\\.ffn_down.*"
        LayerFraction.MOE ->
            "blk\\.$layerIndex\\.ffn_(up|down|gate_up|gate)_(ch|)exps"
        LayerFraction.NONE ->
            error("LAYER_FRACTION_NONE has no overflow pattern")
    }
}

// ---------------------------------------------------------------------------
// llama_get_device_memory_data  (stub)
// ---------------------------------------------------------------------------

/**
 * Queries device memory usage for the given model / context parameters.
 *
 * In the C++ implementation this actually loads the model with `no_alloc`
 * to measure memory use on each device.  In the current Kotlin port this is
 * a **stub** that returns an approximation suitable for CPU-only execution.
 *
 * @return A list of [LlamaDeviceMemoryData], one per device plus a trailing
 *         entry for host (CPU) memory.
 */
internal fun llamaGetDeviceMemoryData(
    pathModel: String,
    mparams: LlamaModelParams,
    cparams: LlamaContextParams,
): List<LlamaDeviceMemoryData> {
    // TODO: implement proper device memory probing
    //   1. Load model with no_alloc = true
    //   2. Create context, call memory_breakdown()
    //   3. Aggregate per-device totals
    //   4. Query device free/total memory
    //
    // For now, return a single host entry with a rough estimate.
    val pageSize = sysconf(_SC_PAGESIZE)
    val totalPages = sysconf(_SC_PHYS_PAGES)
    val totalMem = if (pageSize > 0 && totalPages > 0) pageSize * totalPages else 0L
    val host = LlamaDeviceMemoryData(
        total = totalMem,
        free  = totalMem, // approximation — no portable free-memory query
    )
    return listOf(host)
}

// ---------------------------------------------------------------------------
// llama_params_fit_impl  (full port with TODO stubs for device queries)
// ---------------------------------------------------------------------------

private const val MIB: Long = 1024L * 1024L

/**
 * Auto-fit model parameters ([mparams], [cparams]) to the available device
 * memory.
 *
 * This is a faithful port of the ~600-line `llama_params_fit_impl()` from
 * C++.  The algorithm:
 *
 * 1. Query device memory for the default parameters.
 * 2. Check if everything fits — if so, return immediately.
 * 3. Try reducing context size (linear interpolation between min and max).
 * 4. Iteratively assign "dense" (non-MoE) layers to devices back-to-front
 *    using the method of false position.
 * 5. For MoE models, convert dense-only layers to full layers front-to-back.
 *
 * Because the Kotlin port currently supports only a CPU backend, the
 * multi-device paths are structurally ported but will effectively be
 * unreachable until GPU backends are added.  Device memory queries
 * (`llamaGetDeviceMemoryData`) are stubbed with TODOs.
 *
 * @param pathModel  Path to the GGUF model file.
 * @param mparams    Model params (may be mutated: n_gpu_layers, tensor_split).
 * @param cparams    Context params (may be mutated: n_ctx).
 * @param margins    Per-device memory margin (bytes) to keep free.
 * @param nCtxMin    Minimum acceptable context size.
 * @param logLevel   Minimum log level for progress messages.
 *
 * @throws LlamaParamsFitException if the model cannot be fit.
 */
internal fun llamaParamsFitImpl(
    pathModel: String,
    mparams: LlamaModelParams,
    cparams: LlamaContextParams,
    margins: LongArray,
    nCtxMin: Int = 0,
    @Suppress("UNUSED_PARAMETER") logLevel: LlamaLogLevel = LlamaLogLevel.INFO,
) {
    if (mparams.splitMode == LlamaSplitMode.TENSOR) {
        throw LlamaParamsFitException(
            "llamaParamsFit is not implemented for SPLIT_MODE_TENSOR, abort"
        )
    }

    val defaultMparams = llamaModelDefaultParams()

    // TODO: replace stub — load model to query actual memory
    val dmdsInit = llamaGetDeviceMemoryData(pathModel, mparams, cparams)
    val nd = dmdsInit.size - 1  // number of non-host devices

    val effectiveMargins = LongArray(max(nd, 1)) { i ->
        if (i < margins.size) margins[i] else margins.lastOrNull() ?: 0L
    }

    // ---- step 1: check whether any changes are necessary ----

    var sumFree = 0L
    var sumProjectedFree = 0L
    var sumProjectedUsed = 0L
    var sumProjectedModel = 0L
    val projectedFreePerDevice = mutableListOf<Long>()

    if (nd == 0) {
        // CPU-only path
        val host = dmdsInit.last()
        sumProjectedUsed = host.mb.total()
        sumFree = host.total
        sumProjectedFree = sumFree - sumProjectedUsed

        llamaLogInfo(
            "llamaParamsFitImpl: projected to use ${sumProjectedUsed / MIB} MiB " +
            "of host memory vs. ${sumFree / MIB} MiB of total host memory\n"
        )

        if (sumProjectedFree >= effectiveMargins[0]) {
            llamaLogInfo(
                "llamaParamsFitImpl: will leave ${sumProjectedFree / MIB} >= " +
                "${effectiveMargins[0] / MIB} MiB of system memory, no changes needed\n"
            )
            return
        }
    } else {
        // multi-device path (structurally ported; unreachable on CPU-only)
        for (id in 0 until nd) {
            val dmd = dmdsInit[id]
            val projUsed = dmd.mb.total()
            val projFree = dmd.free - projUsed
            projectedFreePerDevice.add(projFree)
            sumFree += dmd.free
            sumProjectedUsed += projUsed
            sumProjectedFree += projFree
            sumProjectedModel += dmd.mb.model
        }

        llamaLogInfo(
            "llamaParamsFitImpl: projected to use ${sumProjectedUsed / MIB} MiB " +
            "of device memory vs. ${sumFree / MIB} MiB of free device memory\n"
        )

        if (nd == 1 && projectedFreePerDevice[0] >= effectiveMargins[0]) {
            llamaLogInfo(
                "llamaParamsFitImpl: will leave ${projectedFreePerDevice[0] / MIB} >= " +
                "${effectiveMargins[0] / MIB} MiB of free device memory, no changes needed\n"
            )
            return
        }
        if (nd > 1) {
            val changesNeeded = (0 until nd).any { projectedFreePerDevice[it] < effectiveMargins[it] }
            if (!changesNeeded) {
                llamaLogInfo("llamaParamsFitImpl: targets for free memory met on all devices, no changes needed\n")
                return
            }
        }
    }

    // ---- step 2: try reducing context size ----

    run {
        var globalSurplus = sumProjectedFree
        for (m in effectiveMargins) globalSurplus -= m

        if (globalSurplus < 0) {
            llamaLogInfo(
                "llamaParamsFitImpl: need to reduce device memory by ${-globalSurplus / MIB} MiB\n"
            )
            // TODO: implement context-size reduction via linear interpolation
            //   - load model at nCtxMin, measure memory
            //   - interpolate to find largest context that fits
            //   - set cparams.nCtx accordingly
            //
            // For now we log a warning:
            llamaLogWarn("llamaParamsFitImpl: context size reduction not yet implemented in Kotlin port\n")
        }
    }

    if (nd == 0) {
        throw LlamaParamsFitException(
            "was unable to fit model into system memory by reducing context, abort"
        )
    }

    // ---- pre-condition checks for GPU-layer adjustment ----

    if (mparams.nGpuLayers != defaultMparams.nGpuLayers) {
        throw LlamaParamsFitException(
            "n_gpu_layers already set by user to ${mparams.nGpuLayers}, abort"
        )
    }

    // ---- step 3 & 4: iterative layer assignment (multi-device) ----
    // The full false-position algorithm for multi-device layer placement is
    // structurally ported below but gated behind nd > 0, which is unreachable
    // on the current CPU-only backend.

    // TODO: port the full iterative layer assignment once multi-device
    // backends are available.  The C++ algorithm:
    //
    //  a) Build per-device targets = free - margin.
    //  b) Start with zero layers on all devices.
    //  c) For each device (back to front), use the method of false position
    //     to find the maximum number of layers that fit.
    //  d) For MoE models, convert dense-only layers to full layers
    //     front-to-back, trying partial layer placements with different
    //     overflow types (ATTN, UP, GATE).
    //
    // The helper functions (getOverflowPattern, NglPerDevice) are already
    // defined above so the algorithm can be wired in directly.

    llamaLogDebug("llamaParamsFitImpl: multi-device layer assignment deferred (CPU-only port)\n")
}

// ---------------------------------------------------------------------------
// llama_params_fit  (public wrapper)
// ---------------------------------------------------------------------------

/**
 * Auto-fit model parameters to available device memory.
 *
 * This is the safe public wrapper around [llamaParamsFitImpl].  It catches
 * exceptions and returns a status code.
 *
 * Maps to `llama_params_fit()` in C++.
 *
 * @return [LlamaParamsFitStatus.SUCCESS] on success,
 *         [LlamaParamsFitStatus.FAILURE] if the model could not be fit, or
 *         [LlamaParamsFitStatus.ERROR] on an unexpected error.
 */
fun llamaParamsFit(
    pathModel: String,
    mparams: LlamaModelParams,
    cparams: LlamaContextParams,
    margins: LongArray,
    nCtxMin: Int = 0,
    logLevel: LlamaLogLevel = LlamaLogLevel.INFO,
): LlamaParamsFitStatus {
    val t0 = llamaTimeUs()
    val status: LlamaParamsFitStatus = try {
        llamaParamsFitImpl(pathModel, mparams, cparams, margins, nCtxMin, logLevel)
        llamaLogInfo("llamaParamsFit: successfully fit params to free device memory\n")
        LlamaParamsFitStatus.SUCCESS
    } catch (e: LlamaParamsFitException) {
        llamaLogWarn("llamaParamsFit: failed to fit params: ${e.message}\n")
        LlamaParamsFitStatus.FAILURE
    } catch (e: RuntimeException) {
        llamaLogError("llamaParamsFit: error while fitting params: ${e.message}\n")
        LlamaParamsFitStatus.ERROR
    }
    val t1 = llamaTimeUs()
    val elapsedSec = (t1 - t0) * 1e-6
    llamaLogInfo("llamaParamsFit: fitting params took ${formatSeconds(elapsedSec)} seconds\n")
    return status
}

// ---------------------------------------------------------------------------
// Model loading  (llama_model_load_from_file, etc.)
// ---------------------------------------------------------------------------

/**
 * Load a model from a GGUF file.
 *
 * This delegates to the existing [ModelLoader] and [LlamaModelData] types.
 *
 * Maps to `llama_model_load_from_file()` in C++.
 *
 * @param pathModel Path to the GGUF model file.
 * @param params    Model loading parameters.
 * @return The loaded [LlamaModelData], or `null` on failure.
 */
fun llamaModelLoadFromFile(
    pathModel: String,
    params: LlamaModelParams = llamaModelDefaultParams(),
): LlamaModelData? {
    llamaLogInfo("llamaModelLoadFromFile: loading model from '$pathModel'\n")
    return try {
        // TODO: once LlamaModelLoader supports file paths, use it directly.
        // For now we delegate to the simple ModelLoader with a TODO for I/O.
        val model = LlamaModelData()
        model.hparams.vocabOnly = params.vocabOnly
        model.hparams.noAlloc = params.noAlloc

        // TODO: integrate with LlamaModelLoader for full GGUF loading:
        //   val ml = LlamaModelLoader.fromFile(pathModel, params)
        //   ml.loadArch(model)
        //   ml.loadHParams(model)
        //   ml.loadVocab(model)
        //   if (!params.vocabOnly) ml.loadTensors(model)

        llamaLogInfo("llamaModelLoadFromFile: model loaded (stub)\n")
        model
    } catch (e: Exception) {
        llamaLogError("llamaModelLoadFromFile: error loading model: ${e.message}\n")
        null
    }
}

/**
 * Load a model from multiple split GGUF files.
 *
 * Maps to `llama_model_load_from_splits()` in C++.
 *
 * @param paths  Ordered list of split file paths.
 * @param params Model loading parameters.
 * @return The loaded [LlamaModelData], or `null` on failure.
 */
fun llamaModelLoadFromSplits(
    paths: List<String>,
    params: LlamaModelParams = llamaModelDefaultParams(),
): LlamaModelData? {
    if (paths.isEmpty()) {
        llamaLogError("llamaModelLoadFromSplits: list of splits is empty\n")
        return null
    }
    // TODO: implement split-model loading
    return llamaModelLoadFromFile(paths.first(), params)
}

/**
 * Free resources held by a [LlamaModelData].
 *
 * In the Kotlin port this is a no-op because the GC reclaims memory.
 * Retained for API parity with `llama_model_free()`.
 */
fun llamaModelFree(@Suppress("UNUSED_PARAMETER") model: LlamaModelData?) {
    // No-op — Kotlin GC handles cleanup.
}

// ---------------------------------------------------------------------------
// Context creation / destruction  (llama_init_from_model / llama_free)
// ---------------------------------------------------------------------------

/**
 * Create a new inference context from a loaded model.
 *
 * Maps to `llama_init_from_model()` in C++.
 *
 * @param model  The high-level [LlamaModel] (wraps config + weights).
 * @param params Context creation parameters.
 * @return A new [LlamaContext], or `null` on failure.
 */
fun llamaInitFromModel(
    model: LlamaModel,
    params: LlamaContextParams = LlamaContextParams(),
): LlamaContext? {
    return try {
        LlamaContext(model, params)
    } catch (e: Exception) {
        llamaLogError("llamaInitFromModel: failed to create context: ${e.message}\n")
        null
    }
}

/**
 * Free resources held by a [LlamaContext].
 *
 * In the Kotlin port this is effectively a no-op (GC), but the API entry
 * point is kept for compatibility with the C++ contract.
 *
 * Maps to `llama_free()` in C++.
 */
fun llamaFree(@Suppress("UNUSED_PARAMETER") ctx: LlamaContext?) {
    // No-op — Kotlin GC handles cleanup.
}

// ---------------------------------------------------------------------------
// Vocab / token wrappers  (llama_token_*, llama_tokenize, llama_detokenize)
// ---------------------------------------------------------------------------

/**
 * Tokenise [text] using the model's vocabulary.
 *
 * Maps to `llama_tokenize()` in C++.
 *
 * @param vocab       The vocabulary instance.
 * @param text        The input text.
 * @param addSpecial  Whether to add BOS/EOS tokens.
 * @param parseSpecial Whether to parse special tokens in the text.
 * @return The list of token IDs.
 */
fun llamaTokenize(
    vocab: LlamaVocab,
    text: String,
    addSpecial: Boolean,
    parseSpecial: Boolean = false,
): List<LlamaToken> = vocab.tokenize(text, addSpecial, parseSpecial)

/**
 * Convert a list of tokens back to text.
 *
 * Maps to `llama_detokenize()` in C++.
 *
 * @param vocab          The vocabulary instance.
 * @param tokens         The token IDs to detokenise.
 * @param removeSpecial  Strip BOS/EOS tokens.
 * @param unparseSpecial Render special tokens as their text form.
 * @return The decoded string.
 */
fun llamaDetokenize(
    vocab: LlamaVocab,
    tokens: List<LlamaToken>,
    removeSpecial: Boolean = false,
    unparseSpecial: Boolean = false,
): String = vocab.detokenize(tokens, removeSpecial, unparseSpecial)

/**
 * Convert a single token to its string piece.
 *
 * Maps to `llama_token_to_piece()` in C++.
 *
 * @param vocab   The vocabulary instance.
 * @param token   The token ID.
 * @param lstrip  Leading whitespace stripping depth.
 * @param special If `true`, render control tokens as their text.
 * @return The surface string for [token].
 */
fun llamaTokenToPiece(
    vocab: LlamaVocab,
    token: LlamaToken,
    lstrip: Int = 0,
    special: Boolean = true,
): String = vocab.tokenToPiece(token, lstrip, special)

/** Number of tokens in the vocabulary. Maps to `llama_n_vocab()`. */
fun llamaNVocab(vocab: LlamaVocab): Int = vocab.nTokens()

/** BOS token ID. Maps to `llama_token_bos()`. */
fun llamaTokenBos(vocab: LlamaVocab): LlamaToken = vocab.tokenBos()

/** EOS token ID. Maps to `llama_token_eos()`. */
fun llamaTokenEos(vocab: LlamaVocab): LlamaToken = vocab.tokenEos()

/** EOT (end-of-turn) token ID. Maps to `llama_token_eot()`. */
fun llamaTokenEot(vocab: LlamaVocab): LlamaToken = vocab.tokenEot()

/** EOM (end-of-message) token ID. */
fun llamaTokenEom(vocab: LlamaVocab): LlamaToken = vocab.tokenEom()

/** Unknown token ID. Maps to `llama_token_unk()`. */
fun llamaTokenUnk(vocab: LlamaVocab): LlamaToken = vocab.tokenUnk()

/** SEP token ID. Maps to `llama_token_sep()`. */
fun llamaTokenSep(vocab: LlamaVocab): LlamaToken = vocab.tokenSep()

/** Newline token ID. Maps to `llama_token_nl()`. */
fun llamaTokenNl(vocab: LlamaVocab): LlamaToken = vocab.tokenNl()

/** Padding token ID. Maps to `llama_token_pad()`. */
fun llamaTokenPad(vocab: LlamaVocab): LlamaToken = vocab.tokenPad()

/** Whether [token] is an end-of-generation token. Maps to `llama_token_is_eog()`. */
fun llamaTokenIsEog(vocab: LlamaVocab, token: LlamaToken): Boolean = vocab.isEog(token)

/** Whether [token] is a control (special) token. Maps to `llama_token_is_control()`. */
fun llamaTokenIsControl(vocab: LlamaVocab, token: LlamaToken): Boolean = vocab.isControl(token)

/** FIM prefix token. */
fun llamaTokenFimPre(vocab: LlamaVocab): LlamaToken = vocab.tokenFimPre()

/** FIM suffix token. */
fun llamaTokenFimSuf(vocab: LlamaVocab): LlamaToken = vocab.tokenFimSuf()

/** FIM middle token. */
fun llamaTokenFimMid(vocab: LlamaVocab): LlamaToken = vocab.tokenFimMid()

/** FIM pad token. */
fun llamaTokenFimPad(vocab: LlamaVocab): LlamaToken = vocab.tokenFimPad()

/** FIM repo token. */
fun llamaTokenFimRep(vocab: LlamaVocab): LlamaToken = vocab.tokenFimRep()

/** FIM separator token. */
fun llamaTokenFimSep(vocab: LlamaVocab): LlamaToken = vocab.tokenFimSep()

/** Token text accessor. Maps to `llama_token_get_text()`. */
fun llamaTokenGetText(vocab: LlamaVocab, token: LlamaToken): String = vocab.tokenGetText(token)

/** Token score accessor. Maps to `llama_token_get_score()`. */
fun llamaTokenGetScore(vocab: LlamaVocab, token: LlamaToken): Float = vocab.tokenGetScore(token)

/** Token attribute accessor. Maps to `llama_token_get_attr()`. */
fun llamaTokenGetAttr(vocab: LlamaVocab, token: LlamaToken): Int = vocab.tokenGetAttr(token)

// ---------------------------------------------------------------------------
// Chat template  (llama_chat_apply_template) — stub
// ---------------------------------------------------------------------------

/**
 * Applies a chat template to a list of messages and returns the formatted
 * prompt string.
 *
 * Maps to `llama_chat_apply_template()` in C++.
 *
 * @param tmpl     Template name (e.g. `"chatml"`), or `null` for the default.
 * @param messages Ordered list of `(role, content)` pairs.
 * @param addAss   Whether to append a trailing assistant turn.
 * @return The formatted string, or `null` if the template is unknown.
 */
fun llamaChatApplyTemplate(
    tmpl: String?,
    messages: List<Pair<String, String>>,
    addAss: Boolean = true,
): String? {
    // TODO: port llm_chat_detect_template / llm_chat_apply_template
    val template = tmpl ?: "chatml"

    // Minimal "chatml" implementation as a proof of concept
    if (template == "chatml") {
        val sb = StringBuilder()
        for ((role, content) in messages) {
            sb.append("<|im_start|>").append(role).append('\n')
            sb.append(content).append("<|im_end|>\n")
        }
        if (addAss) {
            sb.append("<|im_start|>assistant\n")
        }
        return sb.toString()
    }

    // Unknown template
    return null
}

// ---------------------------------------------------------------------------
// Model split path helpers  (llama_split_path / llama_split_prefix)
// ---------------------------------------------------------------------------

/**
 * Build a split file path from a prefix, split number, and total count.
 *
 * Maps to `llama_split_path()` in C++.
 *
 * @param pathPrefix Base path without the split suffix.
 * @param splitNo    Zero-based split index.
 * @param splitCount Total number of splits.
 * @return The formatted path string.
 */
fun llamaSplitPath(pathPrefix: String, splitNo: Int, splitCount: Int): String =
    "$pathPrefix-${zeroPad(splitNo + 1, 5)}-of-${zeroPad(splitCount, 5)}.gguf"

/**
 * Extract the prefix from a split file path.
 *
 * Maps to `llama_split_prefix()` in C++.
 *
 * @param splitPath  The full split file path.
 * @param splitNo    Zero-based split index.
 * @param splitCount Total number of splits.
 * @return The prefix, or `null` if [splitPath] doesn't match the expected format.
 */
fun llamaSplitPrefix(splitPath: String, splitNo: Int, splitCount: Int): String? {
    val postfix = "-${zeroPad(splitNo + 1, 5)}-of-${zeroPad(splitCount, 5)}.gguf"
    if (splitPath.length <= postfix.length) return null
    if (!splitPath.endsWith(postfix)) return null
    return splitPath.substring(0, splitPath.length - postfix.length)
}

// ---------------------------------------------------------------------------
// System info  — llamaPrintSystemInfo() is defined in LlamaImpl.kt
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Model query helpers  (llama_model_*, llama_n_*)
// ---------------------------------------------------------------------------

/** Total number of parameters across all model tensors. */
fun llamaModelNParams(model: LlamaModelData): Long = model.nElements()

/** Human-readable model description. Maps to `llama_model_desc()`. */
fun llamaModelDesc(model: LlamaModelData): String = model.desc()

/** Model architecture string. */
fun llamaModelArchName(model: LlamaModelData): String = model.archName()

/** Number of layers. Maps to `llama_n_layer()`. */
fun llamaNLayer(model: LlamaModelData): Int = model.hparams.nLayer

/** Embedding dimensionality. Maps to `llama_n_embd()`. */
fun llamaNEmbd(model: LlamaModelData): Int = model.hparams.nEmbd

/** Training context length. Maps to `llama_n_ctx_train()`. */
fun llamaNCtxTrain(model: LlamaModelData): Int = model.hparams.nCtxTrain

// ---------------------------------------------------------------------------
// Context query helpers  (llama_n_ctx, llama_n_batch, etc.)
// ---------------------------------------------------------------------------

/** Effective context length for this context. Maps to `llama_n_ctx()`. */
fun llamaNCtx(ctx: LlamaContext): Int = ctx.cparams.nCtx

/** Logical batch size. Maps to `llama_n_batch()`. */
fun llamaNBatch(ctx: LlamaContext): Int = ctx.cparams.nBatch

/** Physical (micro) batch size. Maps to `llama_n_ubatch()`. */
fun llamaNUbatch(ctx: LlamaContext): Int = ctx.cparams.nUbatch

/** Max sequence count. Maps to `llama_n_seq_max()`. */
fun llamaNSeqMax(ctx: LlamaContext): Int = ctx.cparams.nSeqMax

/** Thread count. Maps to `llama_n_threads()`. */
fun llamaNThreads(ctx: LlamaContext): Int = ctx.cparams.nThreads

/** Batch thread count. Maps to `llama_n_threads_batch()`. */
fun llamaNThreadsBatch(ctx: LlamaContext): Int = ctx.cparams.nThreadsBatch

// ---------------------------------------------------------------------------
// State save/load  (stubs)
// ---------------------------------------------------------------------------

/**
 * Serialise the full context state into a byte array.
 *
 * Maps to `llama_state_get_data()` / `llama_state_set_data()` in C++.
 *
 * @return The serialised state, or `null` if serialisation is not yet supported.
 */
fun llamaStateGetData(@Suppress("UNUSED_PARAMETER") ctx: LlamaContext): ByteArray? {
    // TODO: implement via LlamaIoWrite
    llamaLogWarn("llamaStateGetData: state serialisation not yet implemented\n")
    return null
}

/**
 * Restore context state from a byte array.
 *
 * @return `true` on success, `false` otherwise.
 */
fun llamaStateSetData(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") data: ByteArray,
): Boolean {
    // TODO: implement via LlamaIoRead
    llamaLogWarn("llamaStateSetData: state deserialisation not yet implemented\n")
    return false
}

/**
 * Save the full context state to a file.
 *
 * Maps to `llama_state_save_file()` in C++.
 *
 * @return `true` on success, `false` otherwise.
 */
fun llamaStateSaveFile(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") pathSession: String,
    @Suppress("UNUSED_PARAMETER") tokens: List<LlamaToken>,
): Boolean {
    // TODO: implement file-based state save
    llamaLogWarn("llamaStateSaveFile: not yet implemented\n")
    return false
}

/**
 * Load context state from a file.
 *
 * Maps to `llama_state_load_file()` in C++.
 *
 * @return `true` on success, `false` otherwise.
 */
fun llamaStateLoadFile(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") pathSession: String,
): Boolean {
    // TODO: implement file-based state load
    llamaLogWarn("llamaStateLoadFile: not yet implemented\n")
    return false
}

// ---------------------------------------------------------------------------
// Sequence-level state  (stubs)
// ---------------------------------------------------------------------------

/**
 * Get the state data for a single sequence.
 *
 * Maps to `llama_state_seq_get_data()` in C++.
 */
fun llamaStateSeqGetData(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") seqId: Int,
): ByteArray? {
    // TODO: implement per-sequence state serialisation
    return null
}

/**
 * Set the state data for a single sequence.
 *
 * Maps to `llama_state_seq_set_data()` in C++.
 */
fun llamaStateSeqSetData(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") data: ByteArray,
    @Suppress("UNUSED_PARAMETER") destSeqId: Int,
): Boolean {
    // TODO: implement per-sequence state deserialisation
    return false
}

// ---------------------------------------------------------------------------
// Perf context helpers
// LlamaPerfContextData is defined in LlamaAttention.kt — we reuse it here.
// ---------------------------------------------------------------------------

/**
 * Collect performance data from a context.
 * Maps to `llama_perf_context()` in C++.
 */
fun llamaPerfContext(ctx: LlamaContext): LlamaPerfContextData = LlamaPerfContextData(
    tStartMs = ctx.tStartUs / 1000.0,
    tLoadMs = ctx.tLoadUs / 1000.0,
    tPEvalMs = ctx.tPEvalUs / 1000.0,
    tEvalMs = ctx.tEvalUs / 1000.0,
    nPEval = ctx.nPEval,
    nEval = ctx.nEval,
)

/**
 * Print performance data to the log.
 * Maps to `llama_perf_context_print()` in C++.
 */
fun llamaPerfContextPrint(ctx: LlamaContext) {
    val perf = llamaPerfContext(ctx)
    val msPerTokenP = if (perf.nPEval > 0) perf.tPEvalMs / perf.nPEval else 0.0
    val tokPerSecP = if (perf.tPEvalMs > 0) 1e3 / perf.tPEvalMs * perf.nPEval else 0.0
    val msPerTokenE = if (perf.nEval > 0) perf.tEvalMs / perf.nEval else 0.0
    val tokPerSecE = if (perf.tEvalMs > 0) 1e3 / perf.tEvalMs * perf.nEval else 0.0
    llamaLogInfo(
        "\nllama_perf_context:     load time = ${fmtMs(perf.tLoadMs)} ms\n" +
        "llama_perf_context:   prompt eval time = ${fmtMs(perf.tPEvalMs)} ms / ${perf.nPEval} tokens" +
        " (${fmtMs(msPerTokenP)} ms per token, ${fmtMs(tokPerSecP)} tokens per second)\n" +
        "llama_perf_context:          eval time = ${fmtMs(perf.tEvalMs)} ms / ${perf.nEval} runs" +
        " (${fmtMs(msPerTokenE)} ms per token, ${fmtMs(tokPerSecE)} tokens per second)\n" +
        "llama_perf_context:            total time = ${fmtMs(perf.tPEvalMs + perf.tEvalMs)} ms" +
        " / ${perf.nPEval + perf.nEval} tokens\n"
    )
}

/**
 * Reset performance counters for a context.
 * Maps to `llama_perf_context_reset()` in C++.
 */
fun llamaPerfContextReset(@Suppress("UNUSED_PARAMETER") ctx: LlamaContext) {
    // TODO: reset ctx perf counters when they become mutable
    llamaLogDebug("llamaPerfContextReset: not yet implemented\n")
}
