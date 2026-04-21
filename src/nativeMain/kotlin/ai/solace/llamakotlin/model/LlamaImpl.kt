// port-lint: source llama.cpp/src/llama-impl.h llama.cpp/src/llama-impl.cpp
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import kotlin.time.TimeSource

// =============================================================================
// Logging infrastructure
// Ported from: llama-impl.h / llama-impl.cpp — logging macros, callback,
//              internal log dispatcher, and the global logger singleton.
// =============================================================================

/**
 * Log-level constants that mirror the GGML / llama.cpp convention.
 *
 * These extend the core [GGMLLogLevel] enum with the additional llama-specific
 * levels (`NONE` and `CONT`) that llama.cpp adds on top of GGML.
 */
enum class LlamaLogLevel {
    NONE,
    INFO,
    WARN,
    ERROR,
    DEBUG,
    /** Continuation — appends to the previous log line without a prefix. */
    CONT;
}

/**
 * Signature for a user-supplied log callback.
 *
 * Maps to `ggml_log_callback` (`void (*)(ggml_log_level, const char*, void*)`).
 * In Kotlin we drop the `void* user_data` pointer — callers can capture state
 * via a closure instead.
 */
typealias LlamaLogCallback = (level: LlamaLogLevel, text: String) -> Unit

/**
 * Global logger singleton.
 *
 * Port of the file-scoped `g_logger_state` in `llama-impl.cpp`.
 * Default behaviour writes to stderr (via [defaultLogCallback]).
 */
object LlamaLogger {
    var callback: LlamaLogCallback = ::defaultLogCallback
        private set

    /**
     * Install a custom log callback, or pass `null` to restore the default.
     *
     * Maps to `llama_log_set()` in the C++ API.
     */
    fun setCallback(cb: LlamaLogCallback?) {
        callback = cb ?: ::defaultLogCallback
    }
}

/**
 * Default log callback — prints to stderr, matching `llama_log_callback_default`.
 */
fun defaultLogCallback(level: LlamaLogLevel, text: String) {
    // C++ version writes to stderr via fputs.  In Kotlin/Native we use the
    // platform print functions.
    // Note: GGML_LOG_LEVEL_NONE is printed too (consistent with C++).
    print(text) // stderr redirection can be done at platform level
}

// ---------------------------------------------------------------------------
// Convenience log helpers (replace C++ LLAMA_LOG_* macros)
// ---------------------------------------------------------------------------

/** General (no-level) log. Equivalent to `LLAMA_LOG(...)`. */
fun llamaLog(msg: String) = LlamaLogger.callback(LlamaLogLevel.NONE, msg)

/** Info-level log. Equivalent to `LLAMA_LOG_INFO(...)`. */
fun llamaLogInfo(msg: String) = LlamaLogger.callback(LlamaLogLevel.INFO, msg)

/** Warning-level log. Equivalent to `LLAMA_LOG_WARN(...)`. */
fun llamaLogWarn(msg: String) = LlamaLogger.callback(LlamaLogLevel.WARN, msg)

/** Error-level log. Equivalent to `LLAMA_LOG_ERROR(...)`. */
fun llamaLogError(msg: String) = LlamaLogger.callback(LlamaLogLevel.ERROR, msg)

/** Debug-level log. Equivalent to `LLAMA_LOG_DEBUG(...)`. */
fun llamaLogDebug(msg: String) = LlamaLogger.callback(LlamaLogLevel.DEBUG, msg)

/** Continuation-level log. Equivalent to `LLAMA_LOG_CONT(...)`. */
fun llamaLogCont(msg: String) = LlamaLogger.callback(LlamaLogLevel.CONT, msg)

// =============================================================================
// TimeMeas – scoped time accumulator
// Ported from: llama-impl.h / llama-impl.cpp  struct time_meas
// =============================================================================

/**
 * Scoped wall-clock timer that accumulates elapsed microseconds into an
 * external counter.
 *
 * Usage mirrors the RAII `time_meas` in C++. Because Kotlin has no destructors
 * on regular classes, callers must invoke [stop] explicitly (or use [use]).
 *
 * ```kotlin
 * val tm = TimeMeas(myAccumulator)
 * // … do work …
 * tm.stop()
 * ```
 *
 * @param accumulator A lambda that receives the elapsed µs to add.
 * @param disabled    When `true`, the timer is a no-op (matches `disable` param in C++).
 */
class TimeMeas(
    private val accumulator: (Long) -> Unit,
    disabled: Boolean = false,
) {
    private val mark = TimeSource.Monotonic.markNow()
    private val active = !disabled
    private var stopped = false

    /** Stop the timer and add elapsed µs to the accumulator. Idempotent. */
    fun stop() {
        if (active && !stopped) {
            stopped = true
            val elapsedUs = mark.elapsedNow().inWholeMicroseconds
            accumulator(elapsedUs)
        }
    }

    /** Convenience: execute [block], then [stop]. */
    inline fun <R> use(block: () -> R): R {
        try {
            return block()
        } finally {
            stop()
        }
    }
}

// =============================================================================
// BufferView – lightweight typed view over a memory region
// Ported from: llama-impl.h  template <typename T> struct buffer_view
// =============================================================================

/**
 * Non-owning view over a contiguous region of data.
 *
 * The C++ original is a raw-pointer + size pair.  In Kotlin we store a
 * [ByteArray] slice (offset + length) — the most common representation
 * used elsewhere in the codebase.
 *
 * @property data   Backing byte array (may be shared).
 * @property offset Start position within [data].
 * @property length Number of bytes in the view.
 */
data class BufferView(
    val data: ByteArray?,
    val offset: Int = 0,
    val length: Int = 0,
) {
    /** `true` when the view references valid data. */
    fun hasData(): Boolean = data != null && length > 0

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BufferView) return false
        return offset == other.offset && length == other.length &&
            data.contentEquals(other.data)
    }

    override fun hashCode(): Int {
        var h = offset
        h = 31 * h + length
        h = 31 * h + (data?.contentHashCode() ?: 0)
        return h
    }
}

// =============================================================================
// String helpers
// Ported from: llama-impl.cpp  replace_all, format, llama_format_tensor_shape
// =============================================================================

/**
 * Replace every occurrence of [search] with [replace] in [s].
 *
 * Port of `replace_all()` in `llama-impl.cpp`.
 */
fun replaceAll(s: String, search: String, replace: String): String {
    if (search.isEmpty()) return s
    return s.replace(search, replace)
}

/**
 * Format a tensor shape as a comma-separated dimension list, e.g. `"  4096,   128"`.
 *
 * Port of `llama_format_tensor_shape(const std::vector<int64_t>&)`.
 *
 * @param ne Dimension extents (e.g. `[4096, 128, 1, 1]`).
 */
fun llamaFormatTensorShape(ne: LongArray): String {
    return ne.joinToString(", ") { it.toString().padStart(6) }
}

/**
 * Overload that reads dimensions from a [GGMLTensor].
 *
 * Port of `llama_format_tensor_shape(const ggml_tensor*)`.
 */
fun llamaFormatTensorShape(t: GGMLTensor): String {
    return t.ne.joinToString(", ") { it.toString().padStart(6) }
}

// =============================================================================
// Well-known internal tensor names
// Ported from: llama-impl.h  LLAMA_TENSOR_NAME_* macros
// =============================================================================

/** Sentinel tensor name used for flash-attention intermediates. */
const val LLAMA_TENSOR_NAME_FATTN = "__fattn__"

/** Sentinel tensor name for fused GDN autoregressive intermediates. */
const val LLAMA_TENSOR_NAME_FGDN_AR = "__fgdn_ar__"

/** Sentinel tensor name for fused GDN channel intermediates. */
const val LLAMA_TENSOR_NAME_FGDN_CH = "__fgdn_ch__"
