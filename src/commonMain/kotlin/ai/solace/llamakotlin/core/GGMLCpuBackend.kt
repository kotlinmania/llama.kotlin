// port-lint: source ggml/src/ggml-cpu/ggml-cpu.cpp
package ai.solace.llamakotlin.core

import ai.solace.llamakotlin.core.ByteArrayExtensions.getFloatLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getIntLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getLongLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getShortLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setFloatLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setIntLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setLongLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setShortLe

/**
 * CPU backend implementation for GGML operations.
 *
 * Ported from `ggml-cpu.cpp`. This file contains:
 * 1. CPU buffer type implementation (alloc, get_name, get_alignment, etc.)
 * 2. CPU backend implementation (get_name, graph_plan, graph_compute, supports_op, etc.)
 * 3. CPU device implementation (get_name, get_type, get_memory, get_props)
 * 4. CPU backend registration (ggml_backend_cpu_reg)
 * 5. Graph plan and compute functions
 * 6. Thread management functions
 */

// ============================================================================
// Extra buffer type helpers
// ============================================================================

/**
 * Returns the list of extra buffer types supported by the CPU backend.
 *
 * In the C++ original this collects AMX, RISC-V SpaceMIT, KleidiAI and repack
 * buffer types. In the Kotlin port no hardware-specific extra buffer types
 * exist yet, so the list starts empty.
 *
 * Mirrors `ggml_backend_cpu_get_extra_buffer_types()`.
 */
fun ggmlBackendCpuGetExtraBufferTypes(): MutableList<GGMLBackendBufferType> {
    // Placeholder — add extra buffer types here when they are ported
    return mutableListOf()
}

/**
 * Returns the extra buffer types array with a trailing `null` sentinel,
 * suitable for device queries.
 *
 * Mirrors `ggml_backend_cpu_device_get_extra_buffers_type()`.
 */
private fun ggmlBackendCpuDeviceGetExtraBufferTypes(): List<GGMLBackendBufferType> {
    return ggmlBackendCpuGetExtraBufferTypes()
}

/**
 * Returns `true` when [buft] is one of the extra buffer types registered
 * for the CPU backend.
 *
 * Mirrors `ggml_backend_cpu_is_extra_buffer_type()`.
 */
fun ggmlBackendCpuIsExtraBufferType(buft: GGMLBackendBufferType): Boolean {
    return ggmlBackendCpuGetExtraBufferTypes().any { it === buft }
}

// ============================================================================
// CPU buffer type  (ggml_backend_cpu_buffer_type)
// ============================================================================

/**
 * CPU backend buffer type.
 *
 * Mirrors the buffer-type vtable wired into `ggml_backend_cpu_buffer_type()`.
 */
class GGMLCpuBufferType : GGMLBackendBufferType {
    companion object {
        private const val TENSOR_ALIGNMENT = 32u // 32-byte alignment as in C++ implementation
        private const val MAX_BUFFER_SIZE = ULong.MAX_VALUE // No practical limit for CPU
    }

    override fun getName(): String = "CPU"

    override fun allocBuffer(size: ULong): GGMLBackendBuffer? {
        if (size == 0uL) return null

        return try {
            // Allocate with extra space for alignment
            val alignedSize = (size + TENSOR_ALIGNMENT).toInt()
            val data = ByteArray(alignedSize) { 0 }
            GGMLCpuBuffer(this, data, size)
        } catch (t: Throwable) {
            println("GGMLCpuBufferType: Failed to allocate buffer of size $size (${t.message})")
            null
        }
    }

    override fun getAlignment(): UInt = TENSOR_ALIGNMENT

    override fun getMaxSize(): ULong = MAX_BUFFER_SIZE

    override fun isHost(): Boolean = true
}

// ============================================================================
// CPU buffer  (ggml_backend_cpu_buffer)
// ============================================================================

/**
 * CPU backend buffer implementation using ByteArray.
 *
 * Mirrors the buffer vtable wired into `ggml_backend_cpu_buffer_type_alloc_buffer`.
 */
class GGMLCpuBuffer(
    private val bufferType: GGMLCpuBufferType,
    private val data: ByteArray,
    private val size: ULong
) : GGMLBackendBuffer {

    override fun getType(): GGMLBackendBufferType = bufferType

    override fun getName(): String = "CPU"

    override fun getBase(): Any = data

    override fun getSize(): ULong = size

    override fun free() {
        // ByteArray will be garbage collected, nothing to do explicitly
    }

    override fun initTensor(tensor: GGMLTensor): GGMLStatus {
        // CPU tensors don't need special initialization
        tensor.buffer = this
        return GGMLStatus.SUCCESS
    }

    override fun setTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val tensorData = getBase() as ByteArray
        val srcStart = offset.toInt()
        val srcEnd = (offset + size).toInt()
        val dstStart = tensor.dataOffset.toInt()

        if (srcEnd > data.size) {
            throw IndexOutOfBoundsException("Source data out of bounds: $srcEnd > ${data.size}")
        }

        if (dstStart + size.toInt() > tensorData.size) {
            throw IndexOutOfBoundsException("Destination buffer out of bounds: ${dstStart + size.toInt()} > ${tensorData.size}")
        }

        data.copyInto(tensorData, dstStart, srcStart, srcEnd)
    }

    override fun getTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val tensorData = getBase() as ByteArray
        val srcStart = tensor.dataOffset.toInt()
        val srcEnd = srcStart + size.toInt()
        val dstStart = offset.toInt()

        if (srcEnd > tensorData.size) {
            throw IndexOutOfBoundsException("Source buffer out of bounds: $srcEnd > ${tensorData.size}")
        }

        if (dstStart + size.toInt() > data.size) {
            throw IndexOutOfBoundsException("Destination data out of bounds: ${dstStart + size.toInt()} > ${data.size}")
        }

        tensorData.copyInto(data, dstStart, srcStart, srcEnd)
    }

    override fun copyTensor(src: GGMLTensor, dst: GGMLTensor): Boolean {
        // Check if source is from a host buffer (CPU or compatible)
        val srcBuffer = src.buffer
        if (srcBuffer == null || !srcBuffer.getType().isHost()) {
            return false
        }

        val srcData = srcBuffer.getBase() as? ByteArray ?: return false
        val dstData = getBase() as ByteArray

        val srcByteSize = calculateTensorByteSize(src)
        val dstByteSize = calculateTensorByteSize(dst)

        if (srcByteSize != dstByteSize) {
            return false
        }

        val srcStart = src.dataOffset.toInt()
        val srcEnd = srcStart + srcByteSize.toInt()
        val dstStart = dst.dataOffset.toInt()

        if (srcEnd > srcData.size || dstStart + srcByteSize.toInt() > dstData.size) {
            return false
        }

        srcData.copyInto(dstData, dstStart, srcStart, srcEnd)
        return true
    }

    override fun clear(value: UByte) {
        data.fill(value.toByte())
    }

    // Helper methods for ByteArray access (forwarded from GGMLTypes.kt)
    fun getIntLe(offset: Int): Int = data.getIntLe(offset)
    fun getFloatLe(offset: Int): Float = data.getFloatLe(offset)
    fun getShortLe(offset: Int): Short = data.getShortLe(offset)
    fun getLongLe(offset: Int): Long = data.getLongLe(offset)

    fun setIntLe(offset: Int, value: Int) = data.setIntLe(offset, value)
    fun setFloatLe(offset: Int, value: Float) = data.setFloatLe(offset, value)
    fun setShortLe(offset: Int, value: Short) = data.setShortLe(offset, value)
    fun setLongLe(offset: Int, value: Long) = data.setLongLe(offset, value)
}

// ============================================================================
// CPU backend context  (ggml_backend_cpu_context)
// ============================================================================

/**
 * Internal mutable context carried by every [GGMLCpuBackend] instance.
 *
 * Mirrors `struct ggml_backend_cpu_context` (lines 99-110 in ggml-cpu.cpp).
 *
 * @property nThreads     Number of threads for graph computation.
 * @property threadpool   Optional threadpool; when non-null, compute uses it
 *                        instead of spawning ad-hoc threads.
 * @property workData     Scratch buffer re-used across consecutive
 *                        [GGMLCpuBackend.graphCompute] calls.
 * @property workSize     Current size (in bytes) of [workData].
 * @property abortCallback If non-null, checked between compute steps;
 *                          returning `true` aborts the run.
 * @property abortCallbackData Opaque data forwarded to [abortCallback].
 * @property useRef       When `true`, only reference (non-optimised)
 *                        implementations are used.
 */
class GGMLCpuBackendContext(
    var nThreads: Int = GGML_DEFAULT_N_THREADS,
    var threadpool: GGMLThreadpool? = null,
    var workData: ByteArray? = null,
    var workSize: ULong = 0uL,
    var abortCallback: ((data: Any?) -> Boolean)? = null,
    var abortCallbackData: Any? = null,
    var useRef: Boolean = false
)

// ============================================================================
// CPU graph plan  (ggml_backend_plan_cpu)
// ============================================================================

/**
 * A pre-computed execution plan for a computation graph on the CPU backend.
 *
 * Mirrors `struct ggml_backend_plan_cpu` (lines 125-128 in ggml-cpu.cpp).
 *
 * @property cplan  The compute plan (thread counts, work buffer, callbacks).
 * @property cgraph A snapshot of the computation graph at plan-creation time.
 */
class GGMLCpuGraphPlan(
    val cplan: GGMLCPlan,
    val cgraph: GGMLCGraph
)

// ============================================================================
// CPU backend GUID
// ============================================================================

/**
 * Stable identifier for the CPU backend.
 *
 * Mirrors `ggml_backend_cpu_guid()` — the C version uses a 16-byte array;
 * here we use a human-readable string since Kotlin has no equivalent of
 * `ggml_guid`.
 */
private const val GGML_BACKEND_CPU_GUID = "CPU-KOTLIN-NATIVE"

// ============================================================================
// CPU backend  (ggml_backend_cpu)
// ============================================================================

/**
 * CPU backend implementation.
 *
 * Mirrors the vtable `ggml_backend_cpu_i` and the factory function
 * `ggml_backend_cpu_init()` (lines 193-247 in ggml-cpu.cpp).
 */
class GGMLCpuBackend : GGMLBackend {
    companion object {
        private const val BACKEND_GUID = GGML_BACKEND_CPU_GUID
    }

    /** Internal CPU context — mirrors `ggml_backend_cpu_context`. */
    internal val cpuCtx = GGMLCpuBackendContext()

    private val bufferType = GGMLCpuBufferType()
    private val runtimeConfig = GGMLCpuRuntimeConfig()
    private val executor = GGMLCpuExecutor()

    // -- GGMLBackend interface ------------------------------------------------

    override fun getGuid(): String = BACKEND_GUID

    /** Mirrors `ggml_backend_cpu_get_name()`. */
    override fun getName(): String = "CPU"

    /** Mirrors `ggml_backend_cpu_free()`. */
    override fun free() {
        cpuCtx.workData = null
        cpuCtx.workSize = 0uL
    }

    override fun getDefaultBufferType(): GGMLBackendBufferType = bufferType

    /**
     * Create an execution plan for [graph].
     *
     * Mirrors `ggml_backend_cpu_graph_plan_create()` (lines 130-151).
     */
    override fun graphPlanCreate(graph: GGMLCGraph): Any? {
        val cplan = ggmlGraphPlan(graph, cpuCtx.nThreads, cpuCtx.threadpool)

        if (cplan.workSize > 0uL) {
            cplan.workData = ByteArray(cplan.workSize.toInt())
        }

        cplan.abortCallback = cpuCtx.abortCallback
        cplan.abortCallbackData = cpuCtx.abortCallbackData
        cplan.useRef = cpuCtx.useRef

        return GGMLCpuGraphPlan(cplan, graph)
    }

    /**
     * Free a previously created [plan].
     *
     * Mirrors `ggml_backend_cpu_graph_plan_free()` (lines 153-160).
     */
    override fun graphPlanFree(plan: Any?) {
        if (plan is GGMLCpuGraphPlan) {
            plan.cplan.workData = null
        }
    }

    /**
     * Execute a previously created [plan].
     *
     * Mirrors `ggml_backend_cpu_graph_plan_compute()` (lines 162-168).
     */
    override fun graphPlanCompute(plan: Any?): GGMLStatus {
        require(plan is GGMLCpuGraphPlan) { "Expected GGMLCpuGraphPlan" }
        return ggmlGraphCompute(plan.cgraph, plan.cplan)
    }

    /**
     * Plan and execute [graph] in one shot, re-using the work buffer when
     * possible.
     *
     * Mirrors `ggml_backend_cpu_graph_compute()` (lines 170-191).
     */
    override fun graphCompute(graph: GGMLCGraph): GGMLStatus {
        val cplan = ggmlGraphPlan(graph, cpuCtx.nThreads, cpuCtx.threadpool)

        // Grow the work buffer if necessary
        if (cpuCtx.workSize < cplan.workSize) {
            cpuCtx.workData = try {
                ByteArray(cplan.workSize.toInt())
            } catch (_: Throwable) {
                cpuCtx.workSize = 0uL
                return GGMLStatus.ALLOC_FAILED
            }
            cpuCtx.workSize = cplan.workSize
        }
        cplan.workData = cpuCtx.workData

        cplan.abortCallback = cpuCtx.abortCallback
        cplan.abortCallbackData = cpuCtx.abortCallbackData
        cplan.useRef = cpuCtx.useRef

        return ggmlGraphCompute(graph, cplan)
    }

    /**
     * Check whether the CPU backend can compute [tensor].
     *
     * Mirrors `ggml_backend_cpu_device_supports_op()` (lines 423-473).
     * The full C++ version checks extra buffer types and per-op type
     * constraints; here we check the known supported op set and delegate
     * shape-ops unconditionally.
     */
    override fun supportsOp(tensor: GGMLTensor): Boolean {
        // Shape-only ops are always supported
        if (tensor.op == GGMLOp.NONE || tensor.op == GGMLOp.RESHAPE ||
            tensor.op == GGMLOp.VIEW || tensor.op == GGMLOp.PERMUTE ||
            tensor.op == GGMLOp.TRANSPOSE
        ) {
            return true
        }

        return when (tensor.op) {
            GGMLOp.DUP -> true
            GGMLOp.ADD -> true
            GGMLOp.ADD1 -> true
            GGMLOp.SUB -> true
            GGMLOp.MUL -> true
            GGMLOp.DIV -> true
            GGMLOp.SQR -> true
            GGMLOp.SQRT -> true
            GGMLOp.LOG -> true
            GGMLOp.NEG -> true
            GGMLOp.ABS -> true
            GGMLOp.SGN -> true
            GGMLOp.RELU -> true
            GGMLOp.GELU -> true
            GGMLOp.GELU_QUICK -> true
            GGMLOp.SILU -> true
            GGMLOp.SILU_BACK -> true
            GGMLOp.NORM -> true
            GGMLOp.RMS_NORM -> true
            GGMLOp.RMS_NORM_BACK -> true
            GGMLOp.MUL_MAT -> true
            GGMLOp.REPEAT -> true
            GGMLOp.REPEAT_BACK -> true
            GGMLOp.CONCAT -> true
            GGMLOp.SUM -> true
            GGMLOp.SUM_ROWS -> true
            GGMLOp.MEAN -> true
            GGMLOp.ARGMAX -> true
            GGMLOp.CPY -> true
            GGMLOp.RESHAPE -> true
            GGMLOp.VIEW -> true
            GGMLOp.PERMUTE -> true
            GGMLOp.TRANSPOSE -> true
            GGMLOp.GET_ROWS -> true
            GGMLOp.DIAG_MASK_INF -> true
            GGMLOp.SOFT_MAX -> true
            GGMLOp.ROPE -> true
            GGMLOp.COUNT -> false
            else -> false
        }
    }

    /**
     * Mirrors `ggml_backend_cpu_device_supports_buft()` (line 476).
     */
    override fun supportsBufferType(bufferType: GGMLBackendBufferType): Boolean {
        return bufferType.isHost() || ggmlBackendCpuIsExtraBufferType(bufferType)
    }

    /**
     * CPU backend never offloads — it is the fallback.
     *
     * Mirrors the `NULL` offload_op in `ggml_backend_cpu_device_i`.
     */
    override fun offloadOp(tensor: GGMLTensor): Boolean = false

    // -- Thread management (mirrors free functions in ggml-cpu.cpp) -----------

    /**
     * Set the number of compute threads.
     *
     * Mirrors `ggml_backend_cpu_set_n_threads()` (lines 253-258).
     */
    fun setThreadCount(threadCount: Int) {
        cpuCtx.nThreads = threadCount
        runtimeConfig.threadCount = threadCount
    }

    fun getThreadCount(): Int = cpuCtx.nThreads

    /**
     * Assign a threadpool to the backend.
     *
     * Mirrors `ggml_backend_cpu_set_threadpool()` (lines 260-270).
     */
    fun setThreadpool(threadpool: GGMLThreadpool?) {
        val prev = cpuCtx.threadpool
        if (prev != null && prev !== threadpool) {
            // Already had a different threadpool — pause it before switching
        }
        cpuCtx.threadpool = threadpool
    }

    /**
     * Install an abort callback.
     *
     * Mirrors `ggml_backend_cpu_set_abort_callback()` (lines 272-278).
     */
    fun setAbortCallback(
        callback: ((data: Any?) -> Boolean)?,
        callbackData: Any? = null
    ) {
        cpuCtx.abortCallback = callback
        cpuCtx.abortCallbackData = callbackData
    }

    /**
     * Force the backend to use reference (non-optimised) implementations only.
     *
     * Mirrors `ggml_backend_cpu_set_use_ref()` (lines 280-285).
     */
    fun setUseRef(useRef: Boolean) {
        cpuCtx.useRef = useRef
    }

    fun getUseRef(): Boolean = cpuCtx.useRef

    fun setSchedulingStrategy(strategy: GGMLSchedulingStrategy) {
        runtimeConfig.schedulingStrategy = strategy
    }

    fun getSchedulingStrategy(): GGMLSchedulingStrategy = runtimeConfig.schedulingStrategy

    fun setUseDedicatedDispatcher(use: Boolean) {
        runtimeConfig.useDedicatedDispatcher = use
    }

    fun isUsingDedicatedDispatcher(): Boolean = runtimeConfig.useDedicatedDispatcher
}

// ============================================================================
// Free-standing backend identity helpers
// ============================================================================

// ggmlBackendIsCpu() is defined in GGMLCpuExecutor.kt to avoid duplication.

// ============================================================================
// CPU device context  (ggml_backend_cpu_device_context)
// ============================================================================

/**
 * Device-level context that caches the CPU description string.
 *
 * Mirrors `struct ggml_backend_cpu_device_context` (lines 289-351).
 * The C++ version reads `machdep.cpu.brand_string` (macOS),
 * `/proc/cpuinfo` (Linux), or the Windows registry. Here we default to "CPU"
 * and leave platform-specific detection as a TODO for the Kotlin/Native
 * `expect/actual` layer.
 */
class GGMLCpuDeviceContext {
    val description: String = detectCpuDescription()

    companion object {
        private fun detectCpuDescription(): String {
            return "CPU"
        }
    }
}

// ============================================================================
// CPU device  (ggml_backend_cpu_device)
// ============================================================================

/**
 * The single CPU device exposed by the CPU backend registration.
 *
 * Mirrors `ggml_backend_cpu_device_i` vtable (lines 481-497) and the static
 * device instance created inside `ggml_backend_cpu_reg_get_device()`.
 */
class GGMLCpuDevice(
    private val reg: GGMLBackendReg? = null
) : GGMLBackendDevice {

    private val ctx = GGMLCpuDeviceContext()

    /** Mirrors `ggml_backend_cpu_device_get_name()` (line 353). */
    override fun getName(): String = "CPU"

    /** Mirrors `ggml_backend_cpu_device_get_description()` (line 359). */
    override fun getDescription(): String = ctx.description

    /**
     * Query system memory.
     *
     * Mirrors `ggml_backend_cpu_device_get_memory()` (lines 365-382).
     * Returns (free, total) in bytes. The C++ version calls `sysconf` on
     * POSIX and `GlobalMemoryStatusEx` on Windows. The Kotlin/Native port
     * returns a best-effort estimate via [Runtime] or defaults.
     */
    override fun getMemory(): Pair<ULong, ULong> {
        // Best-effort: report available JVM/native max as both free and total
        // (mirrors the POSIX path that reports total == free).
        val totalBytes = try {
        } catch (_: NotImplementedError) {
            // Fallback: 8 GB skeleton so callers get something reasonable
            8uL * 1024uL * 1024uL * 1024uL
        }
        return Pair(totalBytes, totalBytes)
    }

    /** Mirrors `ggml_backend_cpu_device_get_type()` (line 384). */
    override fun getType(): GGMLBackendDeviceType = GGMLBackendDeviceType.CPU

    /**
     * Mirrors `ggml_backend_cpu_device_get_props()` (lines 390-401).
     */
    override fun getProps(): GGMLBackendDeviceProps {
        val (free, total) = getMemory()
        return GGMLBackendDeviceProps(
            name = getName(),
            description = getDescription(),
            memoryFree = free,
            memoryTotal = total,
            type = getType(),
            caps = GGMLBackendDeviceCaps(
                async = false,
                hostBuffer = false,
                bufferFromHostPtr = true,
                events = false
            )
        )
    }

    /** Mirrors `ggml_backend_cpu_reg()` accessor. */
    override fun getBackendReg(): GGMLBackendReg? = reg

    /**
     * Create a new CPU backend instance.
     *
     * Mirrors `ggml_backend_cpu_device_init_backend()` (lines 403-408).
     */
    override fun initBackend(params: String?): GGMLBackend {
        return GGMLCpuBackend()
    }

    /**
     * Return the CPU buffer type singleton.
     *
     * Mirrors `ggml_backend_cpu_device_get_buffer_type()` (lines 410-414).
     */
    override fun getBufferType(): GGMLBackendBufferType = GGMLCpuBufferType()

    /**
     * Create a CPU buffer wrapping an existing host [ptr].
     *
     * Mirrors `ggml_backend_cpu_device_buffer_from_host_ptr()` (lines 416-421).
     */
    override fun bufferFromHostPtr(
        ptr: ByteArray,
        size: ULong,
        maxTensorSize: ULong
    ): GGMLBackendBuffer {
        return GGMLCpuBuffer(GGMLCpuBufferType(), ptr, size)
    }

    /**
     * Check whether the CPU device can compute [op].
     *
     * Mirrors `ggml_backend_cpu_device_supports_op()` (lines 423-474).
     */
    override fun supportsOp(op: GGMLTensor): Boolean {
        // Shape-only ops are always supported
        if (op.op == GGMLOp.NONE || op.op == GGMLOp.RESHAPE ||
            op.op == GGMLOp.VIEW || op.op == GGMLOp.PERMUTE ||
            op.op == GGMLOp.TRANSPOSE
        ) {
            return true
        }

        // Check extra buffer types on sources
        for (i in 0 until minOf(4, op.src.size)) {
            val src = op.src[i] ?: continue
            val srcBuf = src.buffer ?: continue
            val srcBuft = srcBuf.getType()
            if (ggmlBackendCpuIsExtraBufferType(srcBuft)) {
                // Extra buffer type would handle the op — delegate to it
            }
        }

        // Per-op type constraints (simplified from C++ switch)
        return when (op.op) {
            GGMLOp.MUL_MAT -> {
                val src1 = op.src.getOrNull(1)
                src1 != null && (src1.type == GGMLType.F32 || src1.type == GGMLType.F16)
            }
            else -> true
        }
    }

    /**
     * Mirrors `ggml_backend_cpu_device_supports_buft()` (line 476).
     */
    override fun supportsBufferType(buft: GGMLBackendBufferType): Boolean {
        return buft.isHost() || ggmlBackendCpuIsExtraBufferType(buft)
    }
}

// ============================================================================
// CPU backend feature detection
// ============================================================================

/**
 * Returns the list of CPU features detected at runtime.
 *
 * Mirrors `ggml_backend_cpu_get_features()` (lines 528-640). The C++ version
 * probes for SSE3, AVX, NEON, SVE etc. The Kotlin port delegates to
 * [ggmlCpuDetectFeatures] (defined in GGMLCpuExecutor.kt) and converts the
 * result into the [GGMLBackendFeature] list format.
 */
fun ggmlBackendCpuGetFeatures(): List<GGMLBackendFeature> {
    // Placeholder: return an empty list until ggmlCpuDetectFeatures() is implemented
    return emptyList()
}

// ============================================================================
// CPU backend proc-address lookup
// ============================================================================

/**
 * Look up an extension function by [name].
 *
 * Mirrors `ggml_backend_cpu_get_proc_address()` (lines 642-681).
 * In C++ this returns raw function pointers; in Kotlin we return lambda
 * wrappers or sentinel objects that the caller can cast.
 */
fun ggmlBackendCpuGetProcAddress(name: String): Any? {
    // Each branch returns a lambda cast to Any? so the when-expression has a
    // uniform type. Callers must cast back to the expected function type.
    val result: Any? = when (name) {
        "ggml_backend_set_n_threads" -> object : Any() {
            fun invoke(backend: GGMLBackend, nThreads: Int) {
                require(backend is GGMLCpuBackend); backend.setThreadCount(nThreads)
            }
        }
        "ggml_backend_get_features" -> object : Any() {
            fun invoke(): List<GGMLBackendFeature> = ggmlBackendCpuGetFeatures()
        }
        "ggml_backend_set_abort_callback" -> object : Any() {
            fun invoke(backend: GGMLBackend, cb: ((Any?) -> Boolean)?, data: Any?) {
                require(backend is GGMLCpuBackend); backend.setAbortCallback(cb, data)
            }
        }
        "ggml_backend_cpu_set_use_ref" -> object : Any() {
            fun invoke(backend: GGMLBackend, useRef: Boolean) {
                require(backend is GGMLCpuBackend); backend.setUseRef(useRef)
            }
        }
        "ggml_backend_cpu_set_threadpool" -> object : Any() {
            fun invoke(backend: GGMLBackend, tp: GGMLThreadpool) {
                require(backend is GGMLCpuBackend); backend.setThreadpool(tp)
            }
        }
        "ggml_threadpool_new" -> object : Any() {
            fun invoke(params: GGMLThreadpoolParams): GGMLThreadpool = ggmlThreadpoolNew(params)
        }
        "ggml_threadpool_free" -> object : Any() {
            fun invoke(tp: GGMLThreadpool) = ggmlThreadpoolFree(tp)
        }
        "ggml_backend_cpu_numa_init" -> object : Any() {
            fun invoke(strategy: GGMLNumaStrategy) = ggmlNumaInit(strategy)
        }
        "ggml_backend_cpu_is_numa" -> object : Any() {
            fun invoke(): Boolean = ggmlIsNuma()
        }
        else -> null
    }
    return result
}

// ============================================================================
// CPU backend registration  (ggml_backend_cpu_reg)
// ============================================================================

/**
 * CPU backend registration singleton.
 *
 * Mirrors `ggml_backend_cpu_reg()` (lines 690-701) and the static
 * `ggml_backend_cpu_reg_i` vtable (lines 683-688).
 */
class GGMLCpuBackendReg private constructor() : GGMLBackendReg {

    companion object {
        /** Lazy singleton mirroring the C static local. */
        val instance: GGMLCpuBackendReg by lazy { GGMLCpuBackendReg() }
    }

    /** The single CPU device. */
    private val device: GGMLCpuDevice by lazy { GGMLCpuDevice(reg = this) }

    /** Mirrors `ggml_backend_cpu_reg_get_name()` (line 501). */
    override fun getName(): String = "CPU"

    /** Mirrors `ggml_backend_cpu_reg_get_device_count()` (line 507). */
    override fun getDeviceCount(): ULong = 1uL

    /**
     * Mirrors `ggml_backend_cpu_reg_get_device()` (lines 513-524).
     */
    override fun getDevice(index: ULong): GGMLBackendDevice {
        require(index == 0uL) { "CPU backend has exactly one device (index must be 0)" }
        return device
    }

    /**
     * Mirrors `ggml_backend_cpu_get_proc_address()` (lines 642-681).
     */
    override fun getProcAddress(name: String): Any? {
        return ggmlBackendCpuGetProcAddress(name)
    }

    /**
     * Mirrors `ggml_backend_cpu_get_features()`.
     */
    override fun getFeatures(): List<GGMLBackendFeature> {
        return ggmlBackendCpuGetFeatures()
    }
}

/**
 * Return the global CPU backend registration singleton.
 *
 * Mirrors `ggml_backend_cpu_reg()` (lines 690-701).
 */
fun ggmlBackendCpuRegSingleton(): GGMLBackendReg = GGMLCpuBackendReg.instance
