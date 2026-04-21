// port-lint: source tmp/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h
package ai.solace.llamakotlin.core

/**
 * Kotlin port of ggml-cpu-impl.h — CPU backend implementation internals.
 *
 * This file contains:
 * - [GGMLComputeParams]: per-thread compute context passed to every CPU kernel.
 * - Threadpool barrier and chunk-scheduling function stubs.
 * - Platform-agnostic scalar fallbacks for FP16 ↔ FP32 conversion.
 *
 * SIMD intrinsic types (NEON, SSE/AVX, POWER9, WASM, LoongArch, VXE) are
 * intentionally omitted — they have no Kotlin/Native equivalent. When a future
 * CPU backend needs vectorised paths, platform-specific `actual` declarations
 * or C-interop wrappers should be added.
 */

// ============================================================================
// ggml_threadpool — opaque handle (defined in ggml-cpu internals)
// ============================================================================

/**
 * Opaque threadpool handle.
 * The actual implementation lives in the CPU backend; this is just the type
 * so other files can reference it.
 */
class GGMLThreadpool {
    var nThreads: Int = 1
    // Implementation details will be filled during CPU backend port
}

// ============================================================================
// ggml_compute_params
// ============================================================================

/**
 * Per-thread parameters handed to every CPU compute kernel.
 *
 * Each worker thread receives its own instance so it can determine which
 * slice of the workload it owns via [ith] / [nth].
 *
 * @param ith   Zero-based index of the current thread.
 * @param nth   Total number of threads participating in this computation.
 * @param wsize Size (in bytes) of the shared work buffer.
 * @param wdata Shared work buffer (one allocation, all threads index into it).
 *              `null` when no scratch space is required.
 * @param threadpool Handle to the owning threadpool (nullable during single-threaded runs).
 * @param useRef When `true`, the kernel must use the reference (non-optimised) implementation.
 */
data class GGMLComputeParams(
    val ith: Int,
    val nth: Int,
    val wsize: ULong = 0uL,
    val wdata: ByteArray? = null,
    val threadpool: GGMLThreadpool? = null,
    val useRef: Boolean = false
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GGMLComputeParams) return false
        return ith == other.ith && nth == other.nth && wsize == other.wsize && useRef == other.useRef
    }

    override fun hashCode(): Int {
        var result = ith
        result = 31 * result + nth
        result = 31 * result + wsize.hashCode()
        result = 31 * result + useRef.hashCode()
        return result
    }
}

// ============================================================================
// Threadpool helpers (stubs — will be implemented with real threading)
// ============================================================================

/**
 * Full barrier across all threads in [tp].
 *
 * Every thread must call this before any of them may proceed past it.
 */
fun ggmlBarrier(tp: GGMLThreadpool) {
    error("not yet ported: port from ggml-cpu-impl.h — ggml_barrier: requires platform threading primitives")
}

/**
 * Atomically set the threadpool's shared chunk counter to [value].
 */
fun ggmlThreadpoolChunkSet(tp: GGMLThreadpool, value: Int) {
    error("not yet ported: port from ggml-cpu-impl.h — ggml_threadpool_chunk_set: requires atomic chunk counter on GGMLThreadpool")
}

/**
 * Atomically add [value] to the threadpool's shared chunk counter and
 * return the *previous* value (fetch-and-add semantics).
 */
fun ggmlThreadpoolChunkAdd(tp: GGMLThreadpool, value: Int): Int {
    error("not yet ported: port from ggml-cpu-impl.h — ggml_threadpool_chunk_add: requires atomic chunk counter on GGMLThreadpool")
}

// ============================================================================
// Scalar FP16 helpers (fallback / reference path)
// ============================================================================

/**
 * CPU-level FP32 → FP16 conversion.
 *
 * On platforms with hardware F16C support the C++ original uses `_cvtss_sh`;
 * here we delegate to the software conversion already present in
 * [ggmlFp32ToFp16].
 */
inline fun ggmlCpuFp32ToFp16(x: Float): UShort = ggmlFp32ToFp16(x)

/**
 * CPU-level FP16 → FP32 conversion.
 *
 * Mirrors `GGML_CPU_FP16_TO_FP32` in the C++ source.
 */
inline fun ggmlCpuFp16ToFp32(x: UShort): Float = ggmlFp16ToFp32(x)
