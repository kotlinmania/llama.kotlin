// port-lint: source ggml/include/ggml-cpu.h
package ai.solace.llamakotlin.core

// ============================================================================
// ggml-cpu.h  –  Public CPU-backend API
// ============================================================================
// The declarations below mirror the C header in source-file order.
// Existing executor code is preserved at the bottom of this file.
// ============================================================================

// ---------------------------------------------------------------------------
// Compute plan  (ggml_cplan)
// ---------------------------------------------------------------------------

/**
 * Compute plan produced by [ggmlGraphPlan] and consumed by [ggmlGraphCompute].
 * `workSize` is calculated by the planner; the caller must allocate `workData`
 * before calling compute.
 *
 * Mirrors `struct ggml_cplan` from ggml-cpu.h.
 */
data class GGMLCPlan(
    /** Size of the work buffer in bytes, calculated by [ggmlGraphPlan]. */
    var workSize: ULong = 0uL,
    /** Work buffer supplied by the caller before [ggmlGraphCompute]. */
    var workData: ByteArray? = null,

    /** Number of threads to use during compute. */
    var nThreads: Int = GGML_DEFAULT_N_THREADS,
    /** Optional threadpool handle. */
    var threadpool: GGMLThreadpool? = null,

    /** Callback checked between compute steps; returning `true` aborts the run. */
    var abortCallback: ((data: Any?) -> Boolean)? = null,
    /** Opaque data forwarded to [abortCallback]. */
    var abortCallbackData: Any? = null,

    /** When `true`, only reference (non-optimised) implementations are used. */
    var useRef: Boolean = false
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GGMLCPlan) return false
        return workSize == other.workSize && nThreads == other.nThreads && useRef == other.useRef
    }
    override fun hashCode(): Int = (workSize xor nThreads.toULong()).toInt()
}

// ---------------------------------------------------------------------------
// NUMA strategy  (ggml_numa_strategy)
// ---------------------------------------------------------------------------

/**
 * NUMA placement strategy. Call [ggmlNumaInit] once at startup for better
 * performance on multi-socket machines.
 *
 * Mirrors `enum ggml_numa_strategy` from ggml-cpu.h.
 */
enum class GGMLNumaStrategy(val value: Int) {
    DISABLED(0),
    DISTRIBUTE(1),
    ISOLATE(2),
    NUMACTL(3),
    MIRROR(4);

    companion object {
        val COUNT = entries.size
    }
}

/** Initialise NUMA awareness; call once at startup. */
fun ggmlNumaInit(strategy: GGMLNumaStrategy) {
    TODO("port from ggml-cpu.h – ggml_numa_init")
}

/** Returns `true` when the runtime detected more than one NUMA node. */
fun ggmlIsNuma(): Boolean {
    TODO("port from ggml-cpu.h – ggml_is_numa")
}

// ---------------------------------------------------------------------------
// Scalar tensor helpers
// ---------------------------------------------------------------------------

/** Create a 1-element I32 tensor. */
fun ggmlNewI32(ctx: GGMLContext, value: Int): GGMLTensor {
    TODO("port from ggml-cpu.h – ggml_new_i32")
}

/** Create a 1-element F32 tensor. */
fun ggmlNewF32(ctx: GGMLContext, value: Float): GGMLTensor {
    TODO("port from ggml-cpu.h – ggml_new_f32")
}

/** Fill every element of [tensor] with [value]. */
fun ggmlSetI32(tensor: GGMLTensor, value: Int): GGMLTensor {
    TODO("port from ggml-cpu.h – ggml_set_i32")
}

/** Fill every element of [tensor] with [value]. */
fun ggmlSetF32(tensor: GGMLTensor, value: Float): GGMLTensor {
    TODO("port from ggml-cpu.h – ggml_set_f32")
}

// --- 1-D indexed access ---

fun ggmlGetI32_1d(tensor: GGMLTensor, i: Int): Int {
    TODO("port from ggml-cpu.h – ggml_get_i32_1d")
}

fun ggmlSetI32_1d(tensor: GGMLTensor, i: Int, value: Int) {
    TODO("port from ggml-cpu.h – ggml_set_i32_1d")
}

fun ggmlGetF32_1d(tensor: GGMLTensor, i: Int): Float {
    TODO("port from ggml-cpu.h – ggml_get_f32_1d")
}

fun ggmlSetF32_1d(tensor: GGMLTensor, i: Int, value: Float) {
    TODO("port from ggml-cpu.h – ggml_set_f32_1d")
}

// --- N-D indexed access ---

fun ggmlGetI32_nd(tensor: GGMLTensor, i0: Int, i1: Int, i2: Int, i3: Int): Int {
    TODO("port from ggml-cpu.h – ggml_get_i32_nd")
}

fun ggmlSetI32_nd(tensor: GGMLTensor, i0: Int, i1: Int, i2: Int, i3: Int, value: Int) {
    TODO("port from ggml-cpu.h – ggml_set_i32_nd")
}

fun ggmlGetF32_nd(tensor: GGMLTensor, i0: Int, i1: Int, i2: Int, i3: Int): Float {
    TODO("port from ggml-cpu.h – ggml_get_f32_nd")
}

fun ggmlSetF32_nd(tensor: GGMLTensor, i0: Int, i1: Int, i2: Int, i3: Int, value: Float) {
    TODO("port from ggml-cpu.h – ggml_set_f32_nd")
}

// ---------------------------------------------------------------------------
// Threadpool  (ggml_threadpool / ggml_threadpool_t)
// ---------------------------------------------------------------------------
// The primary GGMLThreadpool class lives in GGMLCpuImpl.kt (it also holds
// the atomic chunkCounter needed by the work-scheduling kernels).
// The free-function API below wraps its lifecycle as declared in ggml-cpu.h.

/** Create a new threadpool from the given parameters. */
fun ggmlThreadpoolNew(params: GGMLThreadpoolParams): GGMLThreadpool {
    return GGMLThreadpool().also { it.nThreads = params.nThreads }
}

/** Release resources associated with the threadpool. */
fun ggmlThreadpoolFree(threadpool: GGMLThreadpool) {
    TODO("port from ggml-cpu.h – ggml_threadpool_free")
}

/** Return the number of threads in the pool. */
fun ggmlThreadpoolGetNThreads(threadpool: GGMLThreadpool): Int = threadpool.nThreads

/** Pause the threadpool (workers go to sleep). */
fun ggmlThreadpoolPause(threadpool: GGMLThreadpool) {
    TODO("port from ggml-cpu.h – ggml_threadpool_pause")
}

/** Resume a paused threadpool. */
fun ggmlThreadpoolResume(threadpool: GGMLThreadpool) {
    TODO("port from ggml-cpu.h – ggml_threadpool_resume")
}

// ---------------------------------------------------------------------------
// Graph plan / compute
// ---------------------------------------------------------------------------

/**
 * Plan memory and threading for graph computation.
 * When `plan.workSize > 0` the caller must allocate `plan.workData` before
 * calling [ggmlGraphCompute].
 */
fun ggmlGraphPlan(
    graph: GGMLCGraph,
    nThreads: Int = GGML_DEFAULT_N_THREADS,
    threadpool: GGMLThreadpool? = null
): GGMLCPlan {
    TODO("port from ggml-cpu.h – ggml_graph_plan")
}

/** Execute a previously planned graph computation. */
fun ggmlGraphCompute(graph: GGMLCGraph, plan: GGMLCPlan): GGMLStatus {
    TODO("port from ggml-cpu.h – ggml_graph_compute")
}

/**
 * Convenience wrapper: allocate work data from the context and compute the
 * graph in one call. The context must have sufficient memory for the work
 * buffer.
 */
fun ggmlGraphComputeWithCtx(ctx: GGMLContext, graph: GGMLCGraph, nThreads: Int): GGMLStatus {
    TODO("port from ggml-cpu.h – ggml_graph_compute_with_ctx")
}

// ---------------------------------------------------------------------------
// System-info helpers  (CPU feature detection)
// ---------------------------------------------------------------------------

/**
 * Aggregated CPU feature-detection results. Each field mirrors the
 * corresponding `ggml_cpu_has_*` probe from ggml-cpu.h.
 */
data class GGMLCpuFeatures(
    // x86
    val hasSse3: Boolean = false,
    val hasSsse3: Boolean = false,
    val hasAvx: Boolean = false,
    val hasAvxVnni: Boolean = false,
    val hasAvx2: Boolean = false,
    val hasBmi2: Boolean = false,
    val hasF16c: Boolean = false,
    val hasFma: Boolean = false,
    val hasAvx512: Boolean = false,
    val hasAvx512Vbmi: Boolean = false,
    val hasAvx512Vnni: Boolean = false,
    val hasAvx512Bf16: Boolean = false,
    val hasAmxInt8: Boolean = false,
    // ARM
    val hasNeon: Boolean = false,
    val hasArmFma: Boolean = false,
    val hasFp16Va: Boolean = false,
    val hasDotprod: Boolean = false,
    val hasMatmulInt8: Boolean = false,
    val hasSve: Boolean = false,
    val sveCnt: Int = 0,
    val hasSme: Boolean = false,
    // RISC-V
    val hasRiscvV: Boolean = false,
    val rvvVlen: Int = 0,
    // Other
    val hasVsx: Boolean = false,
    val hasVxe: Boolean = false,
    val hasWasmSimd: Boolean = false,
    val hasLlamafile: Boolean = false,
)

/** Detect CPU features at runtime. */
fun ggmlCpuDetectFeatures(): GGMLCpuFeatures {
    TODO("port from ggml-cpu.h – ggml_cpu_has_* family")
}

// ---------------------------------------------------------------------------
// Type traits (CPU-specific)  (ggml_type_traits_cpu)
// ---------------------------------------------------------------------------

/**
 * Vectorised dot-product function signature.
 * Mirrors `ggml_vec_dot_t` from ggml-cpu.h.
 *
 * @param n number of elements
 * @param s output accumulator (single-element FloatArray)
 * @param bs byte stride for s (unused in Kotlin, kept for API parity)
 * @param x first operand buffer
 * @param bx byte stride for x
 * @param y second operand buffer
 * @param by byte stride for y
 * @param nrc number of rows to compute simultaneously
 */
typealias GGMLVecDotFn = (n: Int, s: FloatArray, bs: ULong,
                          x: ByteArray, bx: ULong,
                          y: ByteArray, by_: ULong,
                          nrc: Int) -> Unit

/**
 * Per-type conversion function from F32.
 * Mirrors `ggml_from_float_t`.
 */
typealias GGMLFromFloatFn = (src: FloatArray, dst: ByteArray, count: Long) -> Unit

/**
 * CPU-specific type traits.
 * Mirrors `struct ggml_type_traits_cpu` from ggml-cpu.h.
 */
data class GGMLTypeTraitsCpu(
    val fromFloat: GGMLFromFloatFn? = null,
    val vecDot: GGMLVecDotFn? = null,
    val vecDotType: GGMLType = GGMLType.COUNT,
    /** Number of rows the vec_dot kernel processes simultaneously. */
    val nRows: Long = 1
)

/** Look up CPU-specific traits for [type]. */
fun ggmlGetTypeTraitsCpu(type: GGMLType): GGMLTypeTraitsCpu {
    TODO("port from ggml-cpu.h – ggml_get_type_traits_cpu")
}

/** One-time CPU subsystem initialisation. */
fun ggmlCpuInit() {
    TODO("port from ggml-cpu.h – ggml_cpu_init")
}

// ---------------------------------------------------------------------------
// CPU backend creation & configuration
// ---------------------------------------------------------------------------

/** Create and return a new CPU backend instance. */
fun ggmlBackendCpuInit(): GGMLBackend {
    return GGMLCpuBackend()
}

/** Returns `true` when [backend] is the CPU backend. */
fun ggmlBackendIsCpu(backend: GGMLBackend): Boolean {
    return backend is GGMLCpuBackend
}

/** Set the number of compute threads for [backendCpu]. */
fun ggmlBackendCpuSetNThreads(backendCpu: GGMLBackend, nThreads: Int) {
    require(backendCpu is GGMLCpuBackend) { "Expected CPU backend" }
    backendCpu.setThreadCount(nThreads)
}

/** Assign a threadpool to [backendCpu]. */
fun ggmlBackendCpuSetThreadpool(backendCpu: GGMLBackend, threadpool: GGMLThreadpool) {
    require(backendCpu is GGMLCpuBackend) { "Expected CPU backend" }
    TODO("port from ggml-cpu.h – ggml_backend_cpu_set_threadpool")
}

/** Install an abort callback on [backendCpu]. */
fun ggmlBackendCpuSetAbortCallback(
    backendCpu: GGMLBackend,
    abortCallback: ((data: Any?) -> Boolean)?,
    abortCallbackData: Any? = null
) {
    require(backendCpu is GGMLCpuBackend) { "Expected CPU backend" }
    TODO("port from ggml-cpu.h – ggml_backend_cpu_set_abort_callback")
}

/** Force the backend to use reference implementations only. */
fun ggmlBackendCpuSetUseRef(backendCpu: GGMLBackend, useRef: Boolean) {
    require(backendCpu is GGMLCpuBackend) { "Expected CPU backend" }
    TODO("port from ggml-cpu.h – ggml_backend_cpu_set_use_ref")
}

/** Return the global CPU backend registration singleton. */
fun ggmlBackendCpuReg(): GGMLBackendReg {
    TODO("port from ggml-cpu.h – ggml_backend_cpu_reg")
}

// ---------------------------------------------------------------------------
// Precision-conversion utilities
// ---------------------------------------------------------------------------

/** F32 → F32 copy (baseline / reference). */
fun ggmlCpuFp32ToFp32(src: FloatArray, dst: FloatArray, n: Long) {
    src.copyInto(dst, 0, 0, n.toInt())
}

/** F32 → I32 truncation conversion. */
fun ggmlCpuFp32ToI32(src: FloatArray, dst: IntArray, n: Long) {
    for (i in 0 until n.toInt()) { dst[i] = src[i].toInt() }
}

/** F32 → F16 conversion. */
fun ggmlCpuFp32ToFp16(src: FloatArray, dst: ShortArray, n: Long) {
    TODO("port from ggml-cpu.h – ggml_cpu_fp32_to_fp16")
}

/** F16 → F32 conversion. */
fun ggmlCpuFp16ToFp32(src: ShortArray, dst: FloatArray, n: Long) {
    TODO("port from ggml-cpu.h – ggml_cpu_fp16_to_fp32")
}

/** F32 → BF16 conversion. */
fun ggmlCpuFp32ToBf16(src: FloatArray, dst: ShortArray, n: Long) {
    TODO("port from ggml-cpu.h – ggml_cpu_fp32_to_bf16")
}

/** BF16 → F32 conversion. */
fun ggmlCpuBf16ToFp32(src: ShortArray, dst: FloatArray, n: Long) {
    TODO("port from ggml-cpu.h – ggml_cpu_bf16_to_fp32")
}

// ============================================================================
// Existing executor implementation (preserved)
// ============================================================================

/**
 * Runtime configuration for the CPU backend. Developers can tweak threading and scheduling
 * without rebuilding the backend itself, enabling experimentation with worker counts and
 * dispatch policies from higher layers.
 */
internal data class GGMLCpuRuntimeConfig(
    var threadCount: Int = 1,
    var schedulingStrategy: GGMLSchedulingStrategy = GGMLSchedulingStrategy.SEQUENTIAL,
    var useDedicatedDispatcher: Boolean = false
) {
    fun normalizedThreadCount(): Int = threadCount.coerceAtLeast(1)
    fun prefersParallel(): Boolean = schedulingStrategy != GGMLSchedulingStrategy.SEQUENTIAL
}

/**
 * Helper responsible for executing computation graphs on the CPU backend. It bridges the
 * backend layer with the destination-based compute operations and provides a lightweight
 * work-stealing-style scheduler using Kotlin/Native workers. Dependency tracking ensures only
 * independent nodes are processed in parallel; if the runtime is configured for sequential
 * execution, the helper falls back to `GGMLComputeOps.computeGraph`.
 */
internal class GGMLCpuExecutor {

    fun compute(graph: GGMLCGraph, config: GGMLCpuRuntimeConfig): GGMLStatus {
        val allocator = graph.allocator ?: return GGMLStatus.FAILED
        val threads = config.normalizedThreadCount()
        return if (threads == 1 || !config.prefersParallel()) {
            executeSequential(graph)
        } else {
            executeWithDependencies(graph, allocator, threads, config)
        }
    }

    private fun executeSequential(graph: GGMLCGraph): GGMLStatus {
        return try {
            GGMLComputeOps.computeGraph(graph)
            GGMLStatus.SUCCESS
        } catch (t: Throwable) {
            println("GGMLCpuExecutor: sequential compute failed - ${t.message}")
            GGMLStatus.FAILED
        }
    }

    private fun executeWithDependencies(
        graph: GGMLCGraph,
        allocator: GGMLGraphAllocator,
        maxBatchSize: Int,
        config: GGMLCpuRuntimeConfig
    ): GGMLStatus {
        val tracker = GGMLDependencyTracker()
        tracker.buildDependencies(graph)

        return try {
            blockingCompute(tracker, allocator, maxBatchSize, config)
            GGMLStatus.SUCCESS
        } catch (t: Throwable) {
            println("GGMLCpuExecutor: dependency scheduled compute failed - ${t.message}")
            GGMLStatus.FAILED
        }
    }

    private fun blockingCompute(
        tracker: GGMLDependencyTracker,
        allocator: GGMLGraphAllocator,
        maxBatchSize: Int,
        config: GGMLCpuRuntimeConfig
    ) {
        while (!tracker.isComplete()) {
            val ready = tracker.getReadyNodes()
            if (ready.isEmpty()) {
                throw IllegalStateException("No ready nodes but graph not complete (possible cycle)")
            }

            val batches = if (ready.size <= maxBatchSize) {
                listOf(ready)
            } else {
                ready.chunked(maxBatchSize)
            }

            batches.forEach { batch ->
                processBatchSync(allocator, batch)
                batch.forEach { tracker.markCompleted(it) }
            }
        }
    }

    private fun processBatchSync(
        allocator: GGMLGraphAllocator,
        batch: List<GGMLTensor>
    ) {
        batch.forEach { tensor ->
            GGMLComputeOps.computeNode(allocator, tensor)
        }
    }
}
