// port-lint: source tmp/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h
package ai.solace.llamakotlin.core

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/**
 * Kotlin port of ggml-cpu-impl.h — CPU backend implementation internals.
 *
 * This file contains:
 * - [GGMLComputeParams]: per-thread compute context passed to every CPU kernel.
 * - Threadpool barrier and chunk-scheduling functions.
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
 * Port of `struct ggml_threadpool` internals from ggml-cpu.c.
 */
@OptIn(ExperimentalAtomicApi::class)
class GGMLThreadpool {
    var nThreads: Int = 1

    /** Atomic chunk counter for work-stealing. Port of `tp->current_chunk`. */
    private val currentChunk = AtomicInt(0)

    /** Atomic barrier counter. Port of `tp->n_barrier_passed`. */
    private val nBarrierPassed = AtomicInt(0)

    /** Fetch current chunk value. */
    fun chunkGet(): Int = currentChunk.load()

    /** Set chunk counter. Port of `ggml_threadpool_chunk_set`. */
    fun chunkSet(value: Int) { currentChunk.store(value) }

    /** Fetch-and-add on chunk counter. Port of `ggml_threadpool_chunk_add`. */
    fun chunkAdd(value: Int): Int = currentChunk.fetchAndAdd(value)
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
// Threadpool helpers (single-threaded for now, real threading to be added)
// ============================================================================

/**
 * Full barrier across all threads in [tp].
 * Port of `ggml_barrier` from ggml-cpu.c.
 *
 * Every thread must call this before any of them may proceed past it.
 * Single-threaded case returns immediately.
 */
fun ggmlBarrier(tp: GGMLThreadpool) {
    if (tp.nThreads == 1) return
    // FIXME - implement multi-threaded spin barrier when threading is added
}

/**
 * Atomically set the threadpool's shared chunk counter to [value].
 * Port of `ggml_threadpool_chunk_set` from ggml-cpu.c.
 */
fun ggmlThreadpoolChunkSet(tp: GGMLThreadpool, value: Int) {
    tp.chunkSet(value)
}

/**
 * Atomically add [value] to the threadpool's shared chunk counter and
 * return the *previous* value (fetch-and-add semantics).
 * Port of `ggml_threadpool_chunk_add` from ggml-cpu.c.
 */
fun ggmlThreadpoolChunkAdd(tp: GGMLThreadpool, value: Int): Int {
    return tp.chunkAdd(value)
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
