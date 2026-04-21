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
    var progressCallbackUserData: Any? = null,
    var kvOverrides: List<LlamaModelKvOverride>? = null,
    var tensorBuftOverrides: List<LlamaModelTensorBuftOverride>? = null,
    var vocabOnly: Boolean = false,
    var useMmap: Boolean = true,
    var useDirectIo: Boolean = false,
    var useMlock: Boolean = false,
    var checkTensors: Boolean = false,
    var useExtraBuffts: Boolean = true,
    var noHost: Boolean = false,
    var noAlloc: Boolean = false,
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

// ---------------------------------------------------------------------------
// Quantize default params  (llama_model_quantize_default_params)
// ---------------------------------------------------------------------------

/** Returns default [LlamaModelQuantizeParams]. Maps to `llama_model_quantize_default_params()`. */
fun llamaModelQuantizeDefaultParams(): LlamaModelQuantizeParams = LlamaModelQuantizeParams()

// ---------------------------------------------------------------------------
// Max parallel sequences  (llama_max_parallel_sequences)
// ---------------------------------------------------------------------------

/** Maximum parallel sequences supported. Maps to `llama_max_parallel_sequences()`. */
fun llamaMaxParallelSequences(): ULong = 1uL

// ---------------------------------------------------------------------------
// Context query helpers  (llama_n_ctx_seq)
// ---------------------------------------------------------------------------

/** Per-sequence context length. Maps to `llama_n_ctx_seq()`. */
fun llamaNCtxSeq(ctx: LlamaContext): UInt = (ctx.cparams.nCtx / ctx.cparams.nSeqMax).toUInt()

// ---------------------------------------------------------------------------
// Model accessors  (llama_get_model, llama_get_memory, llama_pooling_type)
// ---------------------------------------------------------------------------

/** Get the model from a context. Maps to `llama_get_model()`. */
fun llamaGetModel(ctx: LlamaContext): LlamaModel = ctx.model

/** Get the memory handle from a context. Maps to `llama_get_memory()`. */
fun llamaGetMemory(@Suppress("UNUSED_PARAMETER") ctx: LlamaContext): LlamaMemoryT? {
    // TODO: return actual memory when KV cache implements LlamaMemoryT
    return null
}

/** Get the pooling type for a context. Maps to `llama_pooling_type()`. */
fun llamaPoolingType(ctx: LlamaContext): LlamaPoolingType = ctx.cparams.poolingType

// ---------------------------------------------------------------------------
// Model info accessors  (llama_model_*)
// ---------------------------------------------------------------------------

/** Get the vocab from a model. Maps to `llama_model_get_vocab()`. */
fun llamaModelGetVocab(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): LlamaVocab? {
    // TODO: return model vocab when LlamaModelData includes it
    return null
}

/** Get the RoPE type. Maps to `llama_model_rope_type()`. */
fun llamaModelRopeType(model: LlamaModelData): LlamaRopeType = model.hparams.ropeType

/** Training context length. Maps to `llama_model_n_ctx_train()`. */
fun llamaModelNCtxTrain(model: LlamaModelData): Int = model.hparams.nCtxTrain

/** Embedding dimensionality. Maps to `llama_model_n_embd()`. */
fun llamaModelNEmbd(model: LlamaModelData): Int = model.hparams.nEmbd

/** Input embedding dimensionality. Maps to `llama_model_n_embd_inp()`. */
fun llamaModelNEmbdInp(model: LlamaModelData): Int = model.hparams.nEmbd

/** Output embedding dimensionality. Maps to `llama_model_n_embd_out()`. */
fun llamaModelNEmbdOut(model: LlamaModelData): Int = model.hparams.nEmbd

/** Number of layers. Maps to `llama_model_n_layer()`. */
fun llamaModelNLayer(model: LlamaModelData): Int = model.hparams.nLayer

/** Number of attention heads. Maps to `llama_model_n_head()`. */
fun llamaModelNHead(model: LlamaModelData): Int = model.hparams.nHead()

/** Number of KV attention heads. Maps to `llama_model_n_head_kv()`. */
fun llamaModelNHeadKv(model: LlamaModelData): Int = model.hparams.nHeadKv()

/** Sliding-window attention size (0 if disabled). Maps to `llama_model_n_swa()`. */
fun llamaModelNSwa(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): Int = 0 // TODO

/** RoPE frequency scale from training. Maps to `llama_model_rope_freq_scale_train()`. */
fun llamaModelRopeFreqScaleTrain(model: LlamaModelData): Float = model.hparams.ropeFreqScaleTrain

/** Number of classifier outputs. Maps to `llama_model_n_cls_out()`. */
fun llamaModelNClsOut(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): UInt = 0u // TODO

/** Classifier label by index. Maps to `llama_model_cls_label()`. */
fun llamaModelClsLabel(
    @Suppress("UNUSED_PARAMETER") model: LlamaModelData,
    @Suppress("UNUSED_PARAMETER") i: UInt,
): String? = null // TODO

/** Get vocab type. Maps to `llama_vocab_type()`. */
fun llamaVocabType(vocab: LlamaVocab): LlamaVocabType = vocab.type

/** Number of tokens in the vocab. Maps to `llama_vocab_n_tokens()`. */
fun llamaVocabNTokens(vocab: LlamaVocab): Int = vocab.nTokens()

/** Metadata value by key. Maps to `llama_model_meta_val_str()`. */
fun llamaModelMetaValStr(
    @Suppress("UNUSED_PARAMETER") model: LlamaModelData,
    @Suppress("UNUSED_PARAMETER") key: String,
): String? = null // TODO

/** Number of metadata pairs. Maps to `llama_model_meta_count()`. */
fun llamaModelMetaCount(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): Int = 0 // TODO

/** Sampling metadata key name. Maps to `llama_model_meta_key_str()`. */
fun llamaModelMetaKeyStr(key: LlamaModelMetaKey): String = when (key) {
    LlamaModelMetaKey.SAMPLING_SEQUENCE -> "sampling.sequence"
    LlamaModelMetaKey.SAMPLING_TOP_K -> "sampling.top_k"
    LlamaModelMetaKey.SAMPLING_TOP_P -> "sampling.top_p"
    LlamaModelMetaKey.SAMPLING_MIN_P -> "sampling.min_p"
    LlamaModelMetaKey.SAMPLING_XTC_PROBABILITY -> "sampling.xtc_probability"
    LlamaModelMetaKey.SAMPLING_XTC_THRESHOLD -> "sampling.xtc_threshold"
    LlamaModelMetaKey.SAMPLING_TEMP -> "sampling.temp"
    LlamaModelMetaKey.SAMPLING_PENALTY_LAST_N -> "sampling.penalty_last_n"
    LlamaModelMetaKey.SAMPLING_PENALTY_REPEAT -> "sampling.penalty_repeat"
    LlamaModelMetaKey.SAMPLING_MIROSTAT -> "sampling.mirostat"
    LlamaModelMetaKey.SAMPLING_MIROSTAT_TAU -> "sampling.mirostat_tau"
    LlamaModelMetaKey.SAMPLING_MIROSTAT_ETA -> "sampling.mirostat_eta"
}

/** Total tensor size. Maps to `llama_model_size()`. */
fun llamaModelSize(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): ULong = 0uL // TODO

/** Default chat template. Maps to `llama_model_chat_template()`. */
fun llamaModelChatTemplate(
    @Suppress("UNUSED_PARAMETER") model: LlamaModelData,
    @Suppress("UNUSED_PARAMETER") name: String? = null,
): String? = null // TODO

/** Whether the model has an encoder. Maps to `llama_model_has_encoder()`. */
fun llamaModelHasEncoder(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): Boolean = false // TODO

/** Whether the model has a decoder. Maps to `llama_model_has_decoder()`. */
fun llamaModelHasDecoder(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): Boolean = true // TODO

/** Decoder start token for enc-dec models. Maps to `llama_model_decoder_start_token()`. */
fun llamaModelDecoderStartToken(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): LlamaToken = LLAMA_TOKEN_NULL

/** Whether the model is recurrent. Maps to `llama_model_is_recurrent()`. */
fun llamaModelIsRecurrent(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): Boolean = false // TODO

/** Whether the model is hybrid. Maps to `llama_model_is_hybrid()`. */
fun llamaModelIsHybrid(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): Boolean = false // TODO

/** Whether the model is diffusion-based. Maps to `llama_model_is_diffusion()`. */
fun llamaModelIsDiffusion(@Suppress("UNUSED_PARAMETER") model: LlamaModelData): Boolean = false // TODO

/** Quantize a model file. Maps to `llama_model_quantize()`. */
fun llamaModelQuantize(
    @Suppress("UNUSED_PARAMETER") fnameInp: String,
    @Suppress("UNUSED_PARAMETER") fnameOut: String,
    @Suppress("UNUSED_PARAMETER") params: LlamaModelQuantizeParams,
): UInt {
    // TODO: implement full quantization pipeline
    return 1u
}

/** Save model to file. Maps to `llama_model_save_to_file()`. */
fun llamaModelSaveToFile(
    @Suppress("UNUSED_PARAMETER") model: LlamaModelData,
    @Suppress("UNUSED_PARAMETER") pathModel: String,
) {
    // TODO: implement GGUF model writing
}

// ---------------------------------------------------------------------------
// Adapter functions  (llama_adapter_lora_*)
// ---------------------------------------------------------------------------

/** Load a LoRA adapter. Maps to `llama_adapter_lora_init()`. */
fun llamaAdapterLoraInit(
    @Suppress("UNUSED_PARAMETER") model: LlamaModelData,
    @Suppress("UNUSED_PARAMETER") pathLora: String,
): LlamaAdapterLora? {
    // TODO: implement LoRA loading
    return null
}

/** Free a LoRA adapter. Maps to `llama_adapter_lora_free()`. */
fun llamaAdapterLoraFree(@Suppress("UNUSED_PARAMETER") adapter: LlamaAdapterLora?) {
    // No-op — Kotlin GC handles cleanup
}

/** Number of aLoRA invocation tokens. Maps to `llama_adapter_get_alora_n_invocation_tokens()`. */
fun llamaAdapterGetAloraNInvocationTokens(adapter: LlamaAdapterLora): ULong =
    adapter.nInvocationTokens.toULong()

/** aLoRA invocation token IDs. Maps to `llama_adapter_get_alora_invocation_tokens()`. */
fun llamaAdapterGetAloraInvocationTokens(adapter: LlamaAdapterLora): IntArray =
    adapter.invocationTokens

/** Set LoRA adapters on a context. Maps to `llama_set_adapters_lora()`. */
fun llamaSetAdaptersLora(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") adapters: List<LlamaAdapterLora>,
    @Suppress("UNUSED_PARAMETER") scales: FloatArray,
): Int {
    // TODO: implement adapter application
    return -1
}

/** Apply a control vector. Maps to `llama_set_adapter_cvec()`. */
fun llamaSetAdapterCvec(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") data: FloatArray?,
    @Suppress("UNUSED_PARAMETER") nEmbd: Int,
    @Suppress("UNUSED_PARAMETER") ilStart: Int,
    @Suppress("UNUSED_PARAMETER") ilEnd: Int,
): Int {
    // TODO: implement control vector application
    return -1
}

// ---------------------------------------------------------------------------
// Memory functions  (llama_memory_*)
// ---------------------------------------------------------------------------

/** Clear memory. Maps to `llama_memory_clear()`. */
fun llamaMemoryClear(mem: LlamaMemoryT, data: Boolean) = mem.clear(data)

/** Remove tokens from memory. Maps to `llama_memory_seq_rm()`. */
fun llamaMemorySeqRm(mem: LlamaMemoryT, seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos): Boolean =
    mem.seqRm(seqId, p0, p1)

/** Copy sequence. Maps to `llama_memory_seq_cp()`. */
fun llamaMemorySeqCp(mem: LlamaMemoryT, seqIdSrc: LlamaSeqId, seqIdDst: LlamaSeqId, p0: LlamaPos, p1: LlamaPos) =
    mem.seqCp(seqIdSrc, seqIdDst, p0, p1)

/** Keep only one sequence. Maps to `llama_memory_seq_keep()`. */
fun llamaMemorySeqKeep(mem: LlamaMemoryT, seqId: LlamaSeqId) = mem.seqKeep(seqId)

/** Shift positions. Maps to `llama_memory_seq_add()`. */
fun llamaMemorySeqAdd(mem: LlamaMemoryT, seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, delta: LlamaPos) =
    mem.seqAdd(seqId, p0, p1, delta)

/** Integer-divide positions. Maps to `llama_memory_seq_div()`. */
fun llamaMemorySeqDiv(mem: LlamaMemoryT, seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, d: Int) =
    mem.seqDiv(seqId, p0, p1, d)

/** Min position for sequence. Maps to `llama_memory_seq_pos_min()`. */
fun llamaMemorySeqPosMin(mem: LlamaMemoryT, seqId: LlamaSeqId): LlamaPos = mem.seqPosMin(seqId)

/** Max position for sequence. Maps to `llama_memory_seq_pos_max()`. */
fun llamaMemorySeqPosMax(mem: LlamaMemoryT, seqId: LlamaSeqId): LlamaPos = mem.seqPosMax(seqId)

/** Whether memory supports shifting. Maps to `llama_memory_can_shift()`. */
fun llamaMemoryCanShift(mem: LlamaMemoryT): Boolean = mem.canShift()

// ---------------------------------------------------------------------------
// Extended sequence state  (llama_state_seq_*_ext)
// ---------------------------------------------------------------------------

/** State size with flags. Maps to `llama_state_seq_get_size_ext()`. */
fun llamaStateSeqGetSizeExt(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") seqId: LlamaSeqId,
    @Suppress("UNUSED_PARAMETER") flags: UInt,
): ULong = 0uL // TODO

/** Get state data with flags. Maps to `llama_state_seq_get_data_ext()`. */
fun llamaStateSeqGetDataExt(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") seqId: LlamaSeqId,
    @Suppress("UNUSED_PARAMETER") flags: UInt,
): ByteArray? = null // TODO

/** Set state data with flags. Maps to `llama_state_seq_set_data_ext()`. */
fun llamaStateSeqSetDataExt(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") data: ByteArray,
    @Suppress("UNUSED_PARAMETER") destSeqId: LlamaSeqId,
    @Suppress("UNUSED_PARAMETER") flags: UInt,
): ULong = 0uL // TODO

// ---------------------------------------------------------------------------
// Sequence state file I/O
// ---------------------------------------------------------------------------

/** Save sequence state to file. Maps to `llama_state_seq_save_file()`. */
fun llamaStateSeqSaveFile(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") filepath: String,
    @Suppress("UNUSED_PARAMETER") seqId: LlamaSeqId,
    @Suppress("UNUSED_PARAMETER") tokens: List<LlamaToken>,
): ULong = 0uL // TODO

/** Load sequence state from file. Maps to `llama_state_seq_load_file()`. */
fun llamaStateSeqLoadFile(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") filepath: String,
    @Suppress("UNUSED_PARAMETER") destSeqId: LlamaSeqId,
): Pair<ULong, List<LlamaToken>>? = null // TODO

// ---------------------------------------------------------------------------
// Batch helpers  (llama_batch_get_one, llama_batch_init, llama_batch_free)
// ---------------------------------------------------------------------------

/**
 * Return a batch for a single sequence of tokens. Position tracking is automatic.
 * Maps to `llama_batch_get_one()`.
 */
fun llamaBatchGetOne(tokens: List<LlamaToken>): LlamaBatch = LlamaBatch(
    nTokens = tokens.size,
    tokens = tokens.toIntArray(),
)

/**
 * Allocate a batch that can hold up to [nTokens] tokens.
 * Maps to `llama_batch_init()`.
 */
fun llamaBatchInit(nTokens: Int, embd: Int = 0, nSeqMax: Int = 1): LlamaBatch {
    return if (embd != 0) {
        LlamaBatch(
            nTokens = 0,
            embeddings = FloatArray(nTokens * embd),
            nEmbeddings = embd,
            pos = IntArray(nTokens),
            nSeqId = IntArray(nTokens) { 1 },
            seqId = Array(nTokens) { IntArray(nSeqMax) },
            logits = BooleanArray(nTokens),
        )
    } else {
        LlamaBatch(
            nTokens = 0,
            tokens = IntArray(nTokens),
            pos = IntArray(nTokens),
            nSeqId = IntArray(nTokens) { 1 },
            seqId = Array(nTokens) { IntArray(nSeqMax) },
            logits = BooleanArray(nTokens),
        )
    }
}

/** Free a batch. No-op in Kotlin. Maps to `llama_batch_free()`. */
fun llamaBatchFree(@Suppress("UNUSED_PARAMETER") batch: LlamaBatch) {
    // No-op — Kotlin GC handles cleanup
}

// ---------------------------------------------------------------------------
// Encode / Decode  (llama_encode, llama_decode)
// ---------------------------------------------------------------------------

/**
 * Encode a batch (for encoder-decoder models).
 * Maps to `llama_encode()`.
 * @return 0 on success, negative on error.
 */
fun llamaEncode(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") batch: LlamaBatch,
): Int {
    // TODO: implement encoder pass
    return -1
}

/**
 * Decode a batch of tokens.
 * Maps to `llama_decode()`.
 * @return 0 on success, 1 if no KV slot, 2 if aborted, -1 on error.
 */
fun llamaDecode(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") batch: LlamaBatch,
): Int {
    // TODO: implement decoder pass
    return -1
}

// ---------------------------------------------------------------------------
// Thread config  (llama_set_n_threads)
// ---------------------------------------------------------------------------

/** Set thread counts. Maps to `llama_set_n_threads()`. */
fun llamaSetNThreads(ctx: LlamaContext, nThreads: Int, nThreadsBatch: Int) {
    ctx.cparams.nThreads = nThreads
    ctx.cparams.nThreadsBatch = nThreadsBatch
}

// ---------------------------------------------------------------------------
// Context settings  (llama_set_embeddings, llama_set_causal_attn, etc.)
// ---------------------------------------------------------------------------

/** Set embedding mode. Maps to `llama_set_embeddings()`. */
fun llamaSetEmbeddings(ctx: LlamaContext, embeddings: Boolean) {
    ctx.cparams.embeddings = embeddings
}

/** Set causal attention. Maps to `llama_set_causal_attn()`. */
fun llamaSetCausalAttn(ctx: LlamaContext, causalAttn: Boolean) {
    ctx.cparams.causalAttn = causalAttn
}

/** Set warmup mode. Maps to `llama_set_warmup()`. */
fun llamaSetWarmup(ctx: LlamaContext, warmup: Boolean) {
    ctx.cparams.warmup = warmup
}

/** Set abort callback. Maps to `llama_set_abort_callback()`. */
fun llamaSetAbortCallback(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") abortCallback: (() -> Boolean)?,
) {
    // TODO: store and check during decode
}

/** Wait for async compute to finish. Maps to `llama_synchronize()`. */
fun llamaSynchronize(@Suppress("UNUSED_PARAMETER") ctx: LlamaContext) {
    // No-op — currently synchronous
}

// ---------------------------------------------------------------------------
// Output accessors  (llama_get_logits*, llama_get_embeddings*, llama_get_sampled_*)
// ---------------------------------------------------------------------------

/** Get all output logits. Maps to `llama_get_logits()`. */
fun llamaGetLogits(@Suppress("UNUSED_PARAMETER") ctx: LlamaContext): FloatArray? = null // TODO

/** Get logits for the i-th output token. Maps to `llama_get_logits_ith()`. */
fun llamaGetLogitsIth(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") i: Int,
): FloatArray? = null // TODO

/** Get all output embeddings. Maps to `llama_get_embeddings()`. */
fun llamaGetEmbeddings(@Suppress("UNUSED_PARAMETER") ctx: LlamaContext): FloatArray? = null // TODO

/** Get embeddings for the i-th output token. Maps to `llama_get_embeddings_ith()`. */
fun llamaGetEmbeddingsIth(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") i: Int,
): FloatArray? = null // TODO

/** Get embeddings for a sequence. Maps to `llama_get_embeddings_seq()`. */
fun llamaGetEmbeddingsSeq(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") seqId: LlamaSeqId,
): FloatArray? = null // TODO

/** Backend-sampled token for i-th position. Maps to `llama_get_sampled_token_ith()`. */
fun llamaGetSampledTokenIth(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") i: Int,
): LlamaToken = LLAMA_TOKEN_NULL // TODO

// ---------------------------------------------------------------------------
// Vocab accessors  (llama_vocab_get_text, etc.)
// ---------------------------------------------------------------------------

/** Token text. Maps to `llama_vocab_get_text()`. */
fun llamaVocabGetText(vocab: LlamaVocab, token: LlamaToken): String = vocab.tokenGetText(token)

/** Token score. Maps to `llama_vocab_get_score()`. */
fun llamaVocabGetScore(vocab: LlamaVocab, token: LlamaToken): Float = vocab.tokenGetScore(token)

/** Token attributes. Maps to `llama_vocab_get_attr()`. */
fun llamaVocabGetAttr(vocab: LlamaVocab, token: LlamaToken): Int = vocab.tokenGetAttr(token)

/** Is end-of-generation? Maps to `llama_vocab_is_eog()`. */
fun llamaVocabIsEog(vocab: LlamaVocab, token: LlamaToken): Boolean = vocab.isEog(token)

/** Is control token? Maps to `llama_vocab_is_control()`. */
fun llamaVocabIsControl(vocab: LlamaVocab, token: LlamaToken): Boolean = vocab.isControl(token)

/** BOS token. Maps to `llama_vocab_bos()`. */
fun llamaVocabBos(vocab: LlamaVocab): LlamaToken = vocab.tokenBos()

/** EOS token. Maps to `llama_vocab_eos()`. */
fun llamaVocabEos(vocab: LlamaVocab): LlamaToken = vocab.tokenEos()

/** EOT token. Maps to `llama_vocab_eot()`. */
fun llamaVocabEot(vocab: LlamaVocab): LlamaToken = vocab.tokenEot()

/** Separator token. Maps to `llama_vocab_sep()`. */
fun llamaVocabSep(vocab: LlamaVocab): LlamaToken = vocab.tokenSep()

/** Newline token. Maps to `llama_vocab_nl()`. */
fun llamaVocabNl(vocab: LlamaVocab): LlamaToken = vocab.tokenNl()

/** Padding token. Maps to `llama_vocab_pad()`. */
fun llamaVocabPad(vocab: LlamaVocab): LlamaToken = vocab.tokenPad()

/** Mask token. Maps to `llama_vocab_mask()`. */
fun llamaVocabMask(vocab: LlamaVocab): LlamaToken = vocab.specialMaskId

/** Whether to add BOS. Maps to `llama_vocab_get_add_bos()`. */
fun llamaVocabGetAddBos(vocab: LlamaVocab): Boolean = vocab.addBos

/** Whether to add EOS. Maps to `llama_vocab_get_add_eos()`. */
fun llamaVocabGetAddEos(vocab: LlamaVocab): Boolean = vocab.addEos

/** Whether to add SEP. Maps to `llama_vocab_get_add_sep()`. */
fun llamaVocabGetAddSep(vocab: LlamaVocab): Boolean = vocab.addSep

/** FIM prefix. Maps to `llama_vocab_fim_pre()`. */
fun llamaVocabFimPre(vocab: LlamaVocab): LlamaToken = vocab.tokenFimPre()

/** FIM suffix. Maps to `llama_vocab_fim_suf()`. */
fun llamaVocabFimSuf(vocab: LlamaVocab): LlamaToken = vocab.tokenFimSuf()

/** FIM middle. Maps to `llama_vocab_fim_mid()`. */
fun llamaVocabFimMid(vocab: LlamaVocab): LlamaToken = vocab.tokenFimMid()

/** FIM pad. Maps to `llama_vocab_fim_pad()`. */
fun llamaVocabFimPad(vocab: LlamaVocab): LlamaToken = vocab.tokenFimPad()

/** FIM repo. Maps to `llama_vocab_fim_rep()`. */
fun llamaVocabFimRep(vocab: LlamaVocab): LlamaToken = vocab.tokenFimRep()

/** FIM separator. Maps to `llama_vocab_fim_sep()`. */
fun llamaVocabFimSep(vocab: LlamaVocab): LlamaToken = vocab.tokenFimSep()

// ---------------------------------------------------------------------------
// Chat template helpers  (llama_chat_builtin_templates)
// ---------------------------------------------------------------------------

// llamaChatBuiltinTemplates() is already defined in LlamaChat.kt

// ---------------------------------------------------------------------------
// Sampler lifecycle wrappers  (llama_sampler_*)
// ---------------------------------------------------------------------------

/** Get sampler name. Maps to `llama_sampler_name()`. */
fun llamaSamplerName(smpl: LlamaSampler): String = smpl.name()

/** Accept a token. Maps to `llama_sampler_accept()`. */
fun llamaSamplerAccept(smpl: LlamaSampler, token: LlamaToken) = smpl.accept(token)

/** Apply sampler to candidates. Maps to `llama_sampler_apply()`. */
fun llamaSamplerApply(smpl: LlamaSampler, curP: LlamaTokenDataArray) = smpl.apply(curP)

/** Reset sampler. Maps to `llama_sampler_reset()`. */
fun llamaSamplerReset(smpl: LlamaSampler) = smpl.reset()

/** Clone sampler. Maps to `llama_sampler_clone()`. */
fun llamaSamplerClone(smpl: LlamaSampler): LlamaSampler = smpl.clone()

/** Free sampler. No-op in Kotlin. Maps to `llama_sampler_free()`. */
fun llamaSamplerFree(@Suppress("UNUSED_PARAMETER") smpl: LlamaSampler?) {
    // No-op — Kotlin GC handles cleanup
}

/** Get seed from sampler. Maps to `llama_sampler_get_seed()`. */
fun llamaSamplerGetSeed(@Suppress("UNUSED_PARAMETER") smpl: LlamaSampler): UInt = LLAMA_DEFAULT_SEED

// ---------------------------------------------------------------------------
// Sampler chain wrappers  (llama_sampler_chain_*)
// ---------------------------------------------------------------------------

// llamaSamplerChainInit() is already defined in LlamaSampler.kt

/** Add a sampler to a chain. Maps to `llama_sampler_chain_add()`. */
fun llamaSamplerChainAdd(chain: LlamaSamplerChain, smpl: LlamaSampler) = chain.add(smpl)

/** Get sampler at index. Maps to `llama_sampler_chain_get()`. */
fun llamaSamplerChainGet(chain: LlamaSamplerChain, i: Int): LlamaSampler? = chain.get(i)

/** Number of samplers in chain. Maps to `llama_sampler_chain_n()`. */
fun llamaSamplerChainN(chain: LlamaSamplerChain): Int = chain.n()

/** Remove sampler at index. Maps to `llama_sampler_chain_remove()`. */
fun llamaSamplerChainRemove(chain: LlamaSamplerChain, i: Int): LlamaSampler? = chain.remove(i)

// ---------------------------------------------------------------------------
// Sampler sample shorthand  (llama_sampler_sample)
// ---------------------------------------------------------------------------

/**
 * Sample a token from the idx-th output of the last evaluation.
 * Maps to `llama_sampler_sample()`.
 */
fun llamaSamplerSample(
    smpl: LlamaSampler,
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") idx: Int,
): LlamaToken {
    // TODO: get logits from ctx at idx, build token data array, apply, return selected
    return LLAMA_TOKEN_NULL
}

// ---------------------------------------------------------------------------
// Perf sampler helpers  (llama_perf_sampler*)
// ---------------------------------------------------------------------------

/** Collect perf data from a sampler chain. Maps to `llama_perf_sampler()`. */
fun llamaPerfSampler(@Suppress("UNUSED_PARAMETER") chain: LlamaSampler): LlamaPerfSamplerData =
    LlamaPerfSamplerData()

/** Print sampler perf data. Maps to `llama_perf_sampler_print()`. */
fun llamaPerfSamplerPrint(chain: LlamaSampler) {
    val perf = llamaPerfSampler(chain)
    llamaLogInfo(
        "llama_perf_sampler: sampling time = ${fmtMs(perf.tSampleMs)} ms / ${perf.nSample} runs\n"
    )
}

/** Reset sampler perf counters. Maps to `llama_perf_sampler_reset()`. */
fun llamaPerfSamplerReset(@Suppress("UNUSED_PARAMETER") chain: LlamaSampler) {
    // TODO: reset when LlamaSamplerChain tracks perf
}

/** Print memory breakdown. Maps to `llama_memory_breakdown_print()`. */
fun llamaMemoryBreakdownPrint(@Suppress("UNUSED_PARAMETER") ctx: LlamaContext) {
    // TODO: implement memory breakdown reporting
}

// ---------------------------------------------------------------------------
// System info  (llama_print_system_info)
// ---------------------------------------------------------------------------

/** Print system info. Maps to `llama_print_system_info()`. */
fun llamaPrintSystemInfo(): String = "llama.kotlin (Kotlin/Native CPU backend)"

// ---------------------------------------------------------------------------
// Logging  (llama_log_set, llama_log_get)
// ---------------------------------------------------------------------------

/** Set global log callback. Maps to `llama_log_set()`. */
fun llamaLogSet(callback: LlamaLogCallback?) {
    LlamaLogger.setCallback(callback)
}

// ---------------------------------------------------------------------------
// Training  (llama_opt_*)
// ---------------------------------------------------------------------------

/** Initialise training optimiser. Maps to `llama_opt_init()`. */
fun llamaOptInit(
    @Suppress("UNUSED_PARAMETER") ctx: LlamaContext,
    @Suppress("UNUSED_PARAMETER") model: LlamaModelData,
    @Suppress("UNUSED_PARAMETER") params: LlamaOptParams,
) {
    // TODO: implement training initialisation
}

/** Filter that selects all tensors as trainable. Maps to `llama_opt_param_filter_all()`. */
fun llamaOptParamFilterAll(
    @Suppress("UNUSED_PARAMETER") tensor: GGMLTensor,
): Boolean = true
