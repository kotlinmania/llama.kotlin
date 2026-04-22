// port-lint: source ggml/include/ggml-cpu.h  ggml/src/ggml-cpu/ggml-cpu.c
package ai.solace.llamakotlin.core

import ai.solace.llamakotlin.core.ByteArrayExtensions.getFloatLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getIntLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getShortLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setFloatLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setIntLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setShortLe

// ============================================================================
// ggml-cpu.h  –  Public CPU-backend API
// ============================================================================
// Transliterated 1:1 from the C header and implementation.
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
 * NUMA placement strategy. Call [ggmlNumaInit] once at startup.
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

// Global NUMA state — mirrors g_state.numa in ggml-cpu.c
private var g_numa_n_nodes = 0

/**
 * `ggml_numa_init` — C: ggml-cpu.c lines 623-709.
 *
 * On non-Linux platforms this is a no-op (matches C #else branch).
 * Kotlin/Native doesn't have direct /sys access from common code.
 */
fun ggmlNumaInit(strategy: GGMLNumaStrategy) {
    if (g_numa_n_nodes > 0) {
        println("ggmlNumaInit: NUMA already initialized")
        return
    }
    // Non-Linux: no-op (mirrors the C #else branch which is also a no-op)
}

/**
 * `ggml_is_numa` — C: ggml-cpu.c line 711.
 * Returns `true` when the runtime detected more than one NUMA node.
 */
fun ggmlIsNuma(): Boolean = g_numa_n_nodes > 1

// ---------------------------------------------------------------------------
// Scalar tensor helpers  (ggml-cpu.c lines 735-1143)
// ---------------------------------------------------------------------------

/**
 * `ggml_new_i32` — C: ggml-cpu.c lines 735-743.
 * Create a 1-element I32 tensor and set its value.
 */
fun ggmlNewI32(ctx: GGMLContext, value: Int): GGMLTensor {
    val result = ggmlNewTensor1d(ctx, GGMLType.I32, 1)
    ggmlSetI32(result, value)
    return result
}

/**
 * `ggml_new_f32` — C: ggml-cpu.c lines 745-753.
 * Create a 1-element F32 tensor and set its value.
 */
fun ggmlNewF32(ctx: GGMLContext, value: Float): GGMLTensor {
    val result = ggmlNewTensor1d(ctx, GGMLType.F32, 1)
    ggmlSetF32(result, value)
    return result
}

// Helper to get byte data from a tensor, with null check.
private fun tensorData(tensor: GGMLTensor): ByteArray {
    val d = tensor.data
    require(d is ByteArray) { "tensor data must be ByteArray" }
    return d
}

/**
 * `ggml_set_i32` — C: ggml-cpu.c lines 755-812.
 * Fill every element of [tensor] with [value].
 */
fun ggmlSetI32(tensor: GGMLTensor, value: Int): GGMLTensor {
    val n = ggmlNrows(tensor).toInt()
    val nc = tensor.ne[0].toInt()
    val n1 = tensor.nb[1].toInt()
    val data = tensorData(tensor)

    when (tensor.type) {
        GGMLType.I8 -> {
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data[base + j] = value.toByte()
            }
        }
        GGMLType.I16 -> {
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setShortLe(base + j * 2, value.toShort())
            }
        }
        GGMLType.I32 -> {
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setIntLe(base + j * 4, value)
            }
        }
        GGMLType.F16 -> {
            val fp16 = GGML_FP32_TO_FP16(value.toFloat())
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setShortLe(base + j * 2, fp16)
            }
        }
        GGMLType.BF16 -> {
            val bf16 = GGML_FP32_TO_BF16(value.toFloat())
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setShortLe(base + j * 2, bf16.bits.toShort())
            }
        }
        GGMLType.F32 -> {
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setFloatLe(base + j * 4, value.toFloat())
            }
        }
        else -> error("ggmlSetI32: unsupported type ${tensor.type}")
    }
    return tensor
}

/**
 * `ggml_set_f32` — C: ggml-cpu.c lines 814-871.
 * Fill every element of [tensor] with [value].
 */
fun ggmlSetF32(tensor: GGMLTensor, value: Float): GGMLTensor {
    val n = ggmlNrows(tensor).toInt()
    val nc = tensor.ne[0].toInt()
    val n1 = tensor.nb[1].toInt()
    val data = tensorData(tensor)

    when (tensor.type) {
        GGMLType.I8 -> {
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data[base + j] = value.toInt().toByte()
            }
        }
        GGMLType.I16 -> {
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setShortLe(base + j * 2, value.toInt().toShort())
            }
        }
        GGMLType.I32 -> {
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setIntLe(base + j * 4, value.toInt())
            }
        }
        GGMLType.F16 -> {
            val fp16 = GGML_FP32_TO_FP16(value)
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setShortLe(base + j * 2, fp16)
            }
        }
        GGMLType.BF16 -> {
            val bf16 = GGML_FP32_TO_BF16(value)
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setShortLe(base + j * 2, bf16.bits.toShort())
            }
        }
        GGMLType.F32 -> {
            for (i in 0 until n) {
                val base = i * n1
                for (j in 0 until nc) data.setFloatLe(base + j * 4, value)
            }
        }
        else -> error("ggmlSetF32: unsupported type ${tensor.type}")
    }
    return tensor
}

// --- 1-D indexed access (C lines 873-1091) ---

/** `ggml_get_i32_1d` — C: ggml-cpu.c lines 873-915. */
fun ggmlGetI32_1d(tensor: GGMLTensor, i: Int): Int {
    if (!ggmlIsContiguous(tensor)) {
        val id = ggmlUnravelIndex(tensor, i.toLong())
        return ggmlGetI32_nd(tensor, id[0].toInt(), id[1].toInt(), id[2].toInt(), id[3].toInt())
    }
    val data = tensorData(tensor)
    return when (tensor.type) {
        GGMLType.I8  -> data[i].toInt()
        GGMLType.I16 -> data.getShortLe(i * 2).toInt()
        GGMLType.I32 -> data.getIntLe(i * 4)
        GGMLType.F16 -> GGML_FP16_TO_FP32(data.getShortLe(i * 2)).toInt()
        GGMLType.BF16 -> GGML_BF16_TO_FP32(GGMLBF16(data.getShortLe(i * 2).toUShort())).toInt()
        GGMLType.F32 -> data.getFloatLe(i * 4).toInt()
        else -> error("ggmlGetI32_1d: unsupported type ${tensor.type}")
    }
}

/** `ggml_set_i32_1d` — C: ggml-cpu.c lines 917-960. */
fun ggmlSetI32_1d(tensor: GGMLTensor, i: Int, value: Int) {
    if (!ggmlIsContiguous(tensor)) {
        val id = ggmlUnravelIndex(tensor, i.toLong())
        ggmlSetI32_nd(tensor, id[0].toInt(), id[1].toInt(), id[2].toInt(), id[3].toInt(), value)
        return
    }
    val data = tensorData(tensor)
    when (tensor.type) {
        GGMLType.I8  -> data[i] = value.toByte()
        GGMLType.I16 -> data.setShortLe(i * 2, value.toShort())
        GGMLType.I32 -> data.setIntLe(i * 4, value)
        GGMLType.F16 -> data.setShortLe(i * 2, GGML_FP32_TO_FP16(value.toFloat()))
        GGMLType.BF16 -> data.setShortLe(i * 2, GGML_FP32_TO_BF16(value.toFloat()).bits.toShort())
        GGMLType.F32 -> data.setFloatLe(i * 4, value.toFloat())
        else -> error("ggmlSetI32_1d: unsupported type ${tensor.type}")
    }
}

/** `ggml_get_f32_1d` — C: ggml-cpu.c lines 1016-1052. */
fun ggmlGetF32_1d(tensor: GGMLTensor, i: Int): Float {
    if (!ggmlIsContiguous(tensor)) {
        val id = ggmlUnravelIndex(tensor, i.toLong())
        return ggmlGetF32_nd(tensor, id[0].toInt(), id[1].toInt(), id[2].toInt(), id[3].toInt())
    }
    val data = tensorData(tensor)
    return when (tensor.type) {
        GGMLType.I8  -> data[i].toFloat()
        GGMLType.I16 -> data.getShortLe(i * 2).toFloat()
        GGMLType.I32 -> data.getIntLe(i * 4).toFloat()
        GGMLType.F16 -> GGML_FP16_TO_FP32(data.getShortLe(i * 2))
        GGMLType.BF16 -> GGML_BF16_TO_FP32(GGMLBF16(data.getShortLe(i * 2).toUShort()))
        GGMLType.F32 -> data.getFloatLe(i * 4)
        else -> error("ggmlGetF32_1d: unsupported type ${tensor.type}")
    }
}

/** `ggml_set_f32_1d` — C: ggml-cpu.c lines 1054-1091. */
fun ggmlSetF32_1d(tensor: GGMLTensor, i: Int, value: Float) {
    if (!ggmlIsContiguous(tensor)) {
        val id = ggmlUnravelIndex(tensor, i.toLong())
        ggmlSetF32_nd(tensor, id[0].toInt(), id[1].toInt(), id[2].toInt(), id[3].toInt(), value)
        return
    }
    val data = tensorData(tensor)
    when (tensor.type) {
        GGMLType.I8  -> data[i] = value.toInt().toByte()
        GGMLType.I16 -> data.setShortLe(i * 2, value.toInt().toShort())
        GGMLType.I32 -> data.setIntLe(i * 4, value.toInt())
        GGMLType.F16 -> data.setShortLe(i * 2, GGML_FP32_TO_FP16(value))
        GGMLType.BF16 -> data.setShortLe(i * 2, GGML_FP32_TO_BF16(value).bits.toShort())
        GGMLType.F32 -> data.setFloatLe(i * 4, value)
        else -> error("ggmlSetF32_1d: unsupported type ${tensor.type}")
    }
}

// --- N-D indexed access (C lines 962-1143) ---

/** `ggml_get_i32_nd` — C: ggml-cpu.c lines 962-980. */
fun ggmlGetI32_nd(tensor: GGMLTensor, i0: Int, i1: Int, i2: Int, i3: Int): Int {
    val offset = i0 * tensor.nb[0].toInt() + i1 * tensor.nb[1].toInt() +
                 i2 * tensor.nb[2].toInt() + i3 * tensor.nb[3].toInt()
    val data = tensorData(tensor)
    return when (tensor.type) {
        GGMLType.I8  -> data[offset].toInt()
        GGMLType.I16 -> data.getShortLe(offset).toInt()
        GGMLType.I32 -> data.getIntLe(offset)
        GGMLType.F16 -> GGML_FP16_TO_FP32(data.getShortLe(offset)).toInt()
        GGMLType.BF16 -> GGML_BF16_TO_FP32(GGMLBF16(data.getShortLe(offset).toUShort())).toInt()
        GGMLType.F32 -> data.getFloatLe(offset).toInt()
        else -> error("ggmlGetI32_nd: unsupported type ${tensor.type}")
    }
}

/** `ggml_set_i32_nd` — C: ggml-cpu.c lines 982-1014. */
fun ggmlSetI32_nd(tensor: GGMLTensor, i0: Int, i1: Int, i2: Int, i3: Int, value: Int) {
    val offset = i0 * tensor.nb[0].toInt() + i1 * tensor.nb[1].toInt() +
                 i2 * tensor.nb[2].toInt() + i3 * tensor.nb[3].toInt()
    val data = tensorData(tensor)
    when (tensor.type) {
        GGMLType.I8  -> data[offset] = value.toByte()
        GGMLType.I16 -> data.setShortLe(offset, value.toShort())
        GGMLType.I32 -> data.setIntLe(offset, value)
        GGMLType.F16 -> data.setShortLe(offset, GGML_FP32_TO_FP16(value.toFloat()))
        GGMLType.BF16 -> data.setShortLe(offset, GGML_FP32_TO_BF16(value.toFloat()).bits.toShort())
        GGMLType.F32 -> data.setFloatLe(offset, value.toFloat())
        else -> error("ggmlSetI32_nd: unsupported type ${tensor.type}")
    }
}

/** `ggml_get_f32_nd` — C: ggml-cpu.c lines 1093-1111. */
fun ggmlGetF32_nd(tensor: GGMLTensor, i0: Int, i1: Int, i2: Int, i3: Int): Float {
    val offset = i0 * tensor.nb[0].toInt() + i1 * tensor.nb[1].toInt() +
                 i2 * tensor.nb[2].toInt() + i3 * tensor.nb[3].toInt()
    val data = tensorData(tensor)
    return when (tensor.type) {
        GGMLType.I8  -> data[offset].toFloat()
        GGMLType.I16 -> data.getShortLe(offset).toFloat()
        GGMLType.I32 -> data.getIntLe(offset).toFloat()
        GGMLType.F16 -> GGML_FP16_TO_FP32(data.getShortLe(offset))
        GGMLType.BF16 -> GGML_BF16_TO_FP32(GGMLBF16(data.getShortLe(offset).toUShort()))
        GGMLType.F32 -> data.getFloatLe(offset)
        else -> error("ggmlGetF32_nd: unsupported type ${tensor.type}")
    }
}

/** `ggml_set_f32_nd` — C: ggml-cpu.c lines 1113-1143. */
fun ggmlSetF32_nd(tensor: GGMLTensor, i0: Int, i1: Int, i2: Int, i3: Int, value: Float) {
    val offset = i0 * tensor.nb[0].toInt() + i1 * tensor.nb[1].toInt() +
                 i2 * tensor.nb[2].toInt() + i3 * tensor.nb[3].toInt()
    val data = tensorData(tensor)
    when (tensor.type) {
        GGMLType.I8  -> data[offset] = value.toInt().toByte()
        GGMLType.I16 -> data.setShortLe(offset, value.toInt().toShort())
        GGMLType.I32 -> data.setIntLe(offset, value.toInt())
        GGMLType.F16 -> data.setShortLe(offset, GGML_FP32_TO_FP16(value))
        GGMLType.BF16 -> data.setShortLe(offset, GGML_FP32_TO_BF16(value).bits.toShort())
        GGMLType.F32 -> data.setFloatLe(offset, value)
        else -> error("ggmlSetF32_nd: unsupported type ${tensor.type}")
    }
}

// ---------------------------------------------------------------------------
// Threadpool  (ggml_threadpool / ggml_threadpool_t)
// ---------------------------------------------------------------------------

/**
 * `ggml_threadpool_new` — C: ggml-cpu.c (deferred to threading layer).
 * Creates a new threadpool from the given parameters.
 */
fun ggmlThreadpoolNew(params: GGMLThreadpoolParams): GGMLThreadpool {
    return GGMLThreadpool().also { it.nThreads = params.nThreads }
}

/**
 * `ggml_threadpool_free` — C: ggml-cpu.c.
 * Release resources. In Kotlin the threadpool is GC'd; this matches the
 * C API shape.
 */
fun ggmlThreadpoolFree(threadpool: GGMLThreadpool) {
    threadpool.nThreads = 0
}

/** `ggml_threadpool_get_n_threads` */
fun ggmlThreadpoolGetNThreads(threadpool: GGMLThreadpool): Int = threadpool.nThreads

/**
 * `ggml_threadpool_pause` — suspend worker threads.
 * In Kotlin/Native without a real thread-pool this is a no-op.
 */
fun ggmlThreadpoolPause(threadpool: GGMLThreadpool) {
    // no-op: Kotlin/Native threadpool management not yet implemented
}

/**
 * `ggml_threadpool_resume` — wake worker threads.
 * In Kotlin/Native without a real thread-pool this is a no-op.
 */
fun ggmlThreadpoolResume(threadpool: GGMLThreadpool) {
    // no-op: Kotlin/Native threadpool management not yet implemented
}

// ---------------------------------------------------------------------------
// Graph plan / compute  (ggml-cpu.c lines 2737+)
// ---------------------------------------------------------------------------

/**
 * `ggml_graph_plan` — C: ggml-cpu.c lines 2737+.
 *
 * Plan memory and threading for graph computation. When `plan.workSize > 0`
 * the caller must allocate `plan.workData` before calling [ggmlGraphCompute].
 *
 * The C implementation walks every node to compute the maximum work-buffer
 * size needed. This transliteration currently returns a plan with workSize=0
 * (single-threaded, no work buffer) which is valid for nThreads=1.
 * The full multi-threaded planner requires the per-op work-size calculation
 * from ggml-cpu.c lines 2737-3200.
 */
fun ggmlGraphPlan(
    graph: GGMLCGraph,
    nThreads: Int = GGML_DEFAULT_N_THREADS,
    threadpool: GGMLThreadpool? = null
): GGMLCPlan {
    // C: struct ggml_cplan cplan = { 0 };
    val cplan = GGMLCPlan()
    cplan.nThreads = nThreads.coerceAtLeast(1)
    cplan.threadpool = threadpool

    // The full implementation walks graph nodes to calculate work_size.
    // For single-threaded execution, work_size = 0 is correct for most ops.
    // Multi-threaded work-size estimation requires per-op analysis from
    // ggml-cpu.c lines 2800-3200.
    var workSize = 0UL
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        // Most ops need no work buffer for single-threaded execution.
        // MUL_MAT and similar may need scratch space — this will be
        // expanded as the compute ops require it.
    }
    cplan.workSize = workSize

    return cplan
}

/**
 * `ggml_graph_compute` — C: ggml-cpu.c lines 3202+.
 *
 * Execute a previously planned graph computation. The C implementation
 * dispatches work across threads; this single-threaded transliteration
 * walks nodes sequentially and calls the compute-ops layer.
 */
fun ggmlGraphCompute(graph: GGMLCGraph, plan: GGMLCPlan): GGMLStatus {
    if (plan.workSize > 0uL && plan.workData == null) {
        return GGMLStatus.ALLOC_FAILED
    }

    // Check abort callback before starting
    plan.abortCallback?.let { cb ->
        if (cb(plan.abortCallbackData)) return GGMLStatus.ABORTED
    }

    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue

        // Check abort callback between nodes
        plan.abortCallback?.let { cb ->
            if (cb(plan.abortCallbackData)) return GGMLStatus.ABORTED
        }

        // Skip view ops — they don't need computation
        if (ggmlIsViewOp(node.op)) continue

        val allocator = graph.allocator
        if (allocator != null) {
            GGMLComputeOps.computeNode(allocator, node)
        }
    }

    return GGMLStatus.SUCCESS
}

/**
 * `ggml_graph_compute_with_ctx` — C: ggml-cpu.c line 74.
 *
 * Convenience wrapper: plan and compute in one call. The context must have
 * sufficient memory for the work buffer.
 */
fun ggmlGraphComputeWithCtx(ctx: GGMLContext, graph: GGMLCGraph, nThreads: Int): GGMLStatus {
    val plan = ggmlGraphPlan(graph, nThreads)
    if (plan.workSize > 0uL) {
        plan.workData = ByteArray(plan.workSize.toInt())
    }
    return ggmlGraphCompute(graph, plan)
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

/**
 * `ggml_cpu_has_*` — C: ggml-cpu.c lines 3492-3695.
 *
 * Detect CPU features at runtime. This requires platform-specific code
 * (native expect/actual or cinterop). Returns conservative defaults here;
 * a nativeMain actual can provide real detection.
 */
fun ggmlCpuDetectFeatures(): GGMLCpuFeatures {
    // Conservative: report no SIMD features from common code.
    // nativeMain expect/actual can override with real cpuid / mrs checks.
    return GGMLCpuFeatures()
}

// ---------------------------------------------------------------------------
// Type traits (CPU-specific)  (ggml_type_traits_cpu)
// ---------------------------------------------------------------------------

/**
 * Vectorised dot-product function signature.
 * Mirrors `ggml_vec_dot_t` from ggml-cpu.h.
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
    val nRows: Long = 1
)

// Type-traits table — populated by ggmlCpuInit
private val typeTraitsCpu = Array(GGMLType.COUNT.ordinal) { GGMLTypeTraitsCpu() }

/**
 * `ggml_get_type_traits_cpu` — C: ggml-cpu.h line 123.
 * Look up CPU-specific traits for [type].
 */
fun ggmlGetTypeTraitsCpu(type: GGMLType): GGMLTypeTraitsCpu {
    val idx = type.ordinal
    if (idx < 0 || idx >= typeTraitsCpu.size) return GGMLTypeTraitsCpu()
    return typeTraitsCpu[idx]
}

// CPU init state
private var cpuInitDone = false

/**
 * `ggml_cpu_init` — C: ggml-cpu.c lines 3697-3764.
 *
 * One-time CPU subsystem initialisation. In C this builds lookup tables for
 * GELU/SILU/EXP and detects ARM/RISC-V features. In Kotlin we populate the
 * type-traits table and mark init as done.
 */
fun ggmlCpuInit() {
    if (cpuInitDone) return
    cpuInitDone = true

    // Populate vec_dot_type for quantized types.
    // In C, the type_traits_cpu table is filled with function pointers;
    // here we record the metadata. Actual vec_dot functions are in GGMLCpuQuants.
    typeTraitsCpu[GGMLType.Q4_0.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_0, nRows = 1)
    typeTraitsCpu[GGMLType.Q4_1.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_1, nRows = 1)
    typeTraitsCpu[GGMLType.Q5_0.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_0, nRows = 1)
    typeTraitsCpu[GGMLType.Q5_1.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_1, nRows = 1)
    typeTraitsCpu[GGMLType.Q8_0.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_0, nRows = 1)
    typeTraitsCpu[GGMLType.Q8_1.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_1, nRows = 1)
    typeTraitsCpu[GGMLType.Q2_K.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_K, nRows = 1)
    typeTraitsCpu[GGMLType.Q3_K.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_K, nRows = 1)
    typeTraitsCpu[GGMLType.Q4_K.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_K, nRows = 1)
    typeTraitsCpu[GGMLType.Q5_K.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_K, nRows = 1)
    typeTraitsCpu[GGMLType.Q6_K.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_K, nRows = 1)
    typeTraitsCpu[GGMLType.Q8_K.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.Q8_K, nRows = 1)
    typeTraitsCpu[GGMLType.F16.ordinal]  = GGMLTypeTraitsCpu(vecDotType = GGMLType.F32, nRows = 1)
    typeTraitsCpu[GGMLType.BF16.ordinal] = GGMLTypeTraitsCpu(vecDotType = GGMLType.BF16, nRows = 1)
    typeTraitsCpu[GGMLType.F32.ordinal]  = GGMLTypeTraitsCpu(vecDotType = GGMLType.F32, nRows = 1)
}

// ---------------------------------------------------------------------------
// CPU backend creation & configuration — top-level convenience wrappers
// ---------------------------------------------------------------------------
// The actual classes (GGMLCpuBackend, GGMLCpuBackendReg, GGMLCpuDevice) live
// in GGMLCpuBackend.kt, which is the port of ggml-cpu.cpp. These wrappers
// mirror the C free-function API that delegates to the backend objects.

/** `ggml_backend_cpu_init` — C: ggml-cpu.cpp lines 217-247. */
fun ggmlBackendCpuInit(): GGMLBackend {
    ggmlCpuInit()
    return GGMLCpuBackend()
}

/** `ggml_backend_is_cpu` — C: ggml-cpu.cpp lines 249-251. */
fun ggmlBackendIsCpu(backend: GGMLBackend): Boolean {
    return backend is GGMLCpuBackend
}

/** `ggml_backend_cpu_set_n_threads` — C: ggml-cpu.cpp lines 253-258. */
fun ggmlBackendCpuSetNThreads(backendCpu: GGMLBackend, nThreads: Int) {
    require(backendCpu is GGMLCpuBackend) { "Expected CPU backend" }
    backendCpu.setThreadCount(nThreads)
}

/**
 * `ggml_backend_cpu_set_threadpool` — C: ggml-cpu.cpp lines 260-270.
 */
fun ggmlBackendCpuSetThreadpool(backendCpu: GGMLBackend, threadpool: GGMLThreadpool) {
    require(backendCpu is GGMLCpuBackend) { "Expected CPU backend" }
    val ctx = backendCpu.cpuCtx
    if (ctx.threadpool != null && ctx.threadpool !== threadpool) {
        ggmlThreadpoolPause(ctx.threadpool!!)
    }
    ctx.threadpool = threadpool
}

/**
 * `ggml_backend_cpu_set_abort_callback` — C: ggml-cpu.cpp lines 272-278.
 */
fun ggmlBackendCpuSetAbortCallback(
    backendCpu: GGMLBackend,
    abortCallback: ((data: Any?) -> Boolean)?,
    abortCallbackData: Any? = null
) {
    require(backendCpu is GGMLCpuBackend) { "Expected CPU backend" }
    backendCpu.cpuCtx.abortCallback = abortCallback
    backendCpu.cpuCtx.abortCallbackData = abortCallbackData
}

/** `ggml_backend_cpu_set_use_ref` — C: ggml-cpu.cpp lines 280-285. */
fun ggmlBackendCpuSetUseRef(backendCpu: GGMLBackend, useRef: Boolean) {
    require(backendCpu is GGMLCpuBackend) { "Expected CPU backend" }
    backendCpu.cpuCtx.useRef = useRef
}

// ---------------------------------------------------------------------------
// Precision-conversion utilities  (ggml-cpu.c lines 3322-3460)
// ---------------------------------------------------------------------------

/** `ggml_cpu_fp32_to_fp32` — C: line 3322. */
fun ggmlCpuFp32ToFp32(src: FloatArray, dst: FloatArray, n: Long) {
    src.copyInto(dst, 0, 0, n.toInt())
}

/** `ggml_cpu_fp32_to_i32` — C: line 3420. */
fun ggmlCpuFp32ToI32(src: FloatArray, dst: IntArray, n: Long) {
    for (i in 0 until n.toInt()) { dst[i] = src[i].toInt() }
}

/**
 * `ggml_cpu_fp32_to_fp16` — C: line 3326.
 *
 * The C version has SIMD fast-paths (F16C, AVX512, RISC-V zvfh).
 * This is the scalar fallback loop.
 */
fun ggmlCpuFp32ToFp16(src: FloatArray, dst: ShortArray, n: Long) {
    for (i in 0 until n.toInt()) {
        dst[i] = GGML_FP32_TO_FP16(src[i])
    }
}

/**
 * `ggml_cpu_fp16_to_fp32` — C: line 3359.
 *
 * The C version has SIMD fast-paths. This is the scalar fallback.
 */
fun ggmlCpuFp16ToFp32(src: ShortArray, dst: FloatArray, n: Long) {
    for (i in 0 until n.toInt()) {
        dst[i] = GGML_FP16_TO_FP32(src[i])
    }
}

/**
 * `ggml_cpu_fp32_to_bf16` — C: line 3413.
 * Scalar loop (no SIMD fast-path in C either for this direction).
 */
fun ggmlCpuFp32ToBf16(src: FloatArray, dst: ShortArray, n: Long) {
    for (i in 0 until n.toInt()) {
        dst[i] = GGML_FP32_TO_BF16(src[i]).bits.toShort()
    }
}

/**
 * `ggml_cpu_bf16_to_fp32` — C: line 3427.
 *
 * The C version has AVX2/AVX512 fast-paths. This is the scalar fallback.
 */
fun ggmlCpuBf16ToFp32(src: ShortArray, dst: FloatArray, n: Long) {
    for (i in 0 until n.toInt()) {
        dst[i] = GGML_BF16_TO_FP32(GGMLBF16(src[i].toUShort()))
    }
}
