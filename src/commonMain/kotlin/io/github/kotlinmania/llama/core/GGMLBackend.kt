// port-lint: source ggml/include/ggml-backend.h
package io.github.kotlinmania.llama.core

/**
 * Kotlin Native port of the GGML public backend API (`ggml-backend.h`).
 *
 * This file defines the backend interfaces that abstract over different compute backends
 * (currently CPU; additional accelerators can plug into the same interface in the future).
 *
 * Opaque C pointer types are represented by Kotlin interfaces / classes:
 *   ggml_backend_buffer_type_t → GGMLBackendBufferType
 *   ggml_backend_buffer_t      → GGMLBackendBuffer
 *   ggml_backend_t             → GGMLBackend
 *   ggml_backend_event_t       → GGMLBackendEvent
 *   ggml_backend_reg_t         → GGMLBackendReg
 *   ggml_backend_dev_t         → GGMLBackendDevice
 *   ggml_backend_sched_t       → GGMLBackendSched
 *   ggml_backend_graph_plan_t  → Any? (opaque)
 */

// ---------------------------------------------------------------------------
// Backend buffer usage  (ggml_backend_buffer_usage)
// ---------------------------------------------------------------------------

/**
 * How a backend buffer will be used.
 * Mirrors `enum ggml_backend_buffer_usage`.
 */
enum class GGMLBackendBufferUsage {
    /** General-purpose allocation */
    ANY,
    /** Buffer holds model weights */
    WEIGHTS,
    /** Buffer is used for intermediate compute results */
    COMPUTE
}

// ---------------------------------------------------------------------------
// Backend buffer type  (ggml_backend_buffer_type_t)
// ---------------------------------------------------------------------------

/**
 * Interface for backend buffer types – describes the properties of a class of buffers.
 * Mirrors the C `ggml_backend_buffer_type` opaque struct and the free functions that
 * operate on `ggml_backend_buffer_type_t`.
 */
interface GGMLBackendBufferType {

    /** `ggml_backend_buft_name` */
    fun getName(): String

    /** `ggml_backend_buft_alloc_buffer` */
    fun allocBuffer(size: ULong): io.github.kotlinmania.llama.core.GGMLBackendBuffer?

    /** `ggml_backend_buft_get_alignment` */
    fun getAlignment(): UInt

    /** `ggml_backend_buft_get_max_size` */
    fun getMaxSize(): ULong

    /** `ggml_backend_buft_get_alloc_size` – allocation size required for a tensor */
    fun getAllocSize(tensor: io.github.kotlinmania.llama.core.GGMLTensor): ULong {
        return _root_ide_package_.io.github.kotlinmania.llama.core.calculateTensorByteSize(tensor)
    }

    /** `ggml_backend_buft_is_host` */
    fun isHost(): Boolean

    /** `ggml_backend_buft_get_device` */
    fun getDevice(): io.github.kotlinmania.llama.core.GGMLBackendDevice? {
        return null
    }

    /** C field: `void * context` — opaque pointer for extra buffer type data. */
    fun getContext(): Any? {
        return null
    }
}

// ---------------------------------------------------------------------------
// Backend buffer  (ggml_backend_buffer_t)
// ---------------------------------------------------------------------------

/**
 * Interface for backend buffers – represents a block of allocated memory.
 * Mirrors the C `ggml_backend_buffer` opaque struct and associated free functions.
 */
interface GGMLBackendBuffer {

    /** `ggml_backend_buffer_get_type` */
    fun getType(): io.github.kotlinmania.llama.core.GGMLBackendBufferType

    /** `ggml_backend_buffer_name` */
    fun getName(): String

    /** `ggml_backend_buffer_get_base` */
    fun getBase(): Any?

    /** `ggml_backend_buffer_get_size` */
    fun getSize(): ULong

    /** `ggml_backend_buffer_free` */
    fun free()

    /** `ggml_backend_buffer_init_tensor` */
    fun initTensor(tensor: io.github.kotlinmania.llama.core.GGMLTensor): io.github.kotlinmania.llama.core.GGMLStatus {
        return _root_ide_package_.io.github.kotlinmania.llama.core.GGMLStatus.SUCCESS
    }

    /** `ggml_backend_buffer_set_tensor` – set tensor data from host memory */
    fun setTensor(tensor: io.github.kotlinmania.llama.core.GGMLTensor, data: ByteArray, offset: ULong, size: ULong)

    /** `ggml_backend_buffer_get_tensor` – get tensor data into host memory */
    fun getTensor(tensor: io.github.kotlinmania.llama.core.GGMLTensor, data: ByteArray, offset: ULong, size: ULong)

    /** Copy tensor data between buffers – returns true if the copy was handled */
    fun copyTensor(src: io.github.kotlinmania.llama.core.GGMLTensor, dst: io.github.kotlinmania.llama.core.GGMLTensor): Boolean

    /** `ggml_backend_buffer_clear` */
    fun clear(value: UByte)

    /** `ggml_backend_buffer_get_alignment` */
    fun getAlignment(): ULong {
        return getType().getAlignment().toULong()
    }

    /** `ggml_backend_buffer_get_max_size` */
    fun getMaxSize(): ULong {
        return getType().getMaxSize()
    }

    /** `ggml_backend_buffer_get_alloc_size` */
    fun getAllocSize(tensor: io.github.kotlinmania.llama.core.GGMLTensor): ULong {
        return getType().getAllocSize(tensor)
    }

    /** `ggml_backend_buffer_is_host` */
    fun isHost(): Boolean {
        return getType().isHost()
    }

    /** `ggml_backend_buffer_set_usage` */
    fun setUsage(usage: io.github.kotlinmania.llama.core.GGMLBackendBufferUsage) {
        // Default: identity; implementations may track usage
    }

    /** `ggml_backend_buffer_get_usage` */
    fun getUsage(): io.github.kotlinmania.llama.core.GGMLBackendBufferUsage {
        return _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendBufferUsage.ANY
    }

    /** `ggml_backend_buffer_reset` – reset the buffer (clear allocations) */
    fun reset() {
        // Default: identity
    }
}

// ---------------------------------------------------------------------------
// Backend computation status  (ggml_status)
// ---------------------------------------------------------------------------

/**
 * Computation status returned by graph-compute and related operations.
 * Mirrors `enum ggml_status`.
 */
enum class GGMLStatus {
    SUCCESS,
    FAILED,
    ALLOC_FAILED,
    ABORTED
}

// ---------------------------------------------------------------------------
// Backend (stream)  (ggml_backend_t)
// ---------------------------------------------------------------------------

/**
 * Main backend interface – represents a compute backend (stream).
 * Mirrors the C `ggml_backend` opaque struct and associated free functions.
 */
interface GGMLBackend {

    /** `ggml_backend_guid` */
    fun getGuid(): String

    /** `ggml_backend_name` */
    fun getName(): String

    /** `ggml_backend_free` */
    fun free()

    /** `ggml_backend_get_default_buffer_type` */
    fun getDefaultBufferType(): io.github.kotlinmania.llama.core.GGMLBackendBufferType

    /** `ggml_backend_alloc_buffer` */
    fun allocBuffer(size: ULong): io.github.kotlinmania.llama.core.GGMLBackendBuffer? {
        return getDefaultBufferType().allocBuffer(size)
    }

    /** `ggml_backend_get_alignment` */
    fun getAlignment(): UInt {
        return getDefaultBufferType().getAlignment()
    }

    /** `ggml_backend_get_max_size` */
    fun getMaxSize(): ULong {
        return getDefaultBufferType().getMaxSize()
    }

    /** `ggml_backend_tensor_set_async` */
    fun setTensorAsync(tensor: io.github.kotlinmania.llama.core.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val buffer = tensor.buffer
        buffer?.setTensor(tensor, data, offset, size)
    }

    /** `ggml_backend_tensor_get_async` */
    fun getTensorAsync(tensor: io.github.kotlinmania.llama.core.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val buffer = tensor.buffer
        buffer?.getTensor(tensor, data, offset, size)
    }

    /**
     * `ggml_backend_tensor_set_2d_async`
     * Set 2-D strided tensor data asynchronously.
     */
    fun setTensor2dAsync(
        tensor: io.github.kotlinmania.llama.core.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong,
        nCopies: ULong,
        strideTensor: ULong,
        strideData: ULong
    ) {
    }

    /**
     * `ggml_backend_tensor_get_2d_async`
     * Get 2-D strided tensor data asynchronously.
     */
    fun getTensor2dAsync(
        tensor: io.github.kotlinmania.llama.core.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong,
        nCopies: ULong,
        strideTensor: ULong,
        strideData: ULong
    ) {
    }

    /**
     * `ggml_backend_tensor_copy_async`
     * Asynchronous tensor copy between backends. Returns false if unsupported.
     */
    fun copyTensorAsync(backendSrc: GGMLBackend, src: io.github.kotlinmania.llama.core.GGMLTensor, dst: io.github.kotlinmania.llama.core.GGMLTensor): Boolean {
        return false
    }

    /** `ggml_backend_synchronize` */
    fun synchronize() {}

    /** `ggml_backend_graph_plan_create` */
    fun graphPlanCreate(graph: io.github.kotlinmania.llama.core.GGMLCGraph): Any? { return null }

    /** `ggml_backend_graph_plan_free` */
    fun graphPlanFree(plan: Any?) {}

    /** `ggml_backend_graph_plan_compute` */
    fun graphPlanCompute(plan: Any?): io.github.kotlinmania.llama.core.GGMLStatus { return _root_ide_package_.io.github.kotlinmania.llama.core.GGMLStatus.FAILED }

    /** `ggml_backend_graph_compute` */
    fun graphCompute(graph: io.github.kotlinmania.llama.core.GGMLCGraph): io.github.kotlinmania.llama.core.GGMLStatus

    /** `ggml_backend_graph_compute_async` */
    fun graphComputeAsync(graph: io.github.kotlinmania.llama.core.GGMLCGraph): io.github.kotlinmania.llama.core.GGMLStatus {
        return graphCompute(graph)
    }

    /** `ggml_backend_supports_op` – NOTE: will be removed; use device version instead */
    fun supportsOp(tensor: io.github.kotlinmania.llama.core.GGMLTensor): Boolean

    /** `ggml_backend_supports_buft` */
    fun supportsBufferType(bufferType: io.github.kotlinmania.llama.core.GGMLBackendBufferType): Boolean

    /** `ggml_backend_offload_op` */
    fun offloadOp(tensor: io.github.kotlinmania.llama.core.GGMLTensor): Boolean {
        return supportsOp(tensor)
    }

    /** `ggml_backend_get_device` */
    fun getDevice(): io.github.kotlinmania.llama.core.GGMLBackendDevice? {
        return null
    }
}

// ---------------------------------------------------------------------------
// Free-standing tensor helpers are implemented in GGMLBackendUtils.kt:
// ggmlBackendTensorCopy, ggmlBackendTensorSet, ggmlBackendTensorGet,
// ggmlBackendTensorSet2d, ggmlBackendTensorGet2d, ggmlBackendTensorMemset,
// ggmlBackendTensorAlloc, ggmlBackendViewInit

// ---------------------------------------------------------------------------
// Events  (ggml_backend_event_t)
// ---------------------------------------------------------------------------

/**
 * Backend event for synchronisation between backends.
 * Mirrors the C `ggml_backend_event` opaque struct.
 */
interface GGMLBackendEvent {

    /** `ggml_backend_event_free` */
    fun free()

    /** `ggml_backend_event_record` */
    fun record(backend: io.github.kotlinmania.llama.core.GGMLBackend)

    /** `ggml_backend_event_synchronize` */
    fun synchronize()

    /** `ggml_backend_event_wait` */
    fun wait(backend: io.github.kotlinmania.llama.core.GGMLBackend)
}

// ---------------------------------------------------------------------------
// Backend device  (ggml_backend_dev_t)
// ---------------------------------------------------------------------------

/**
 * Device type classification.
 * Mirrors `enum ggml_backend_dev_type`.
 */
enum class GGMLBackendDeviceType {
    /** CPU device using system memory */
    CPU,
    /** GPU device using dedicated memory */
    GPU,
    /** Integrated GPU using host memory */
    IGPU,
    /** Accelerator device used alongside CPU (e.g. BLAS, AMX) */
    ACCEL,
    /** Meta device wrapping multiple devices for tensor parallelism */
    META
}

/**
 * Capabilities supported by a backend device.
 * Mirrors `struct ggml_backend_dev_caps`.
 */
data class GGMLBackendDeviceCaps(
    /** Supports asynchronous operations */
    val async: Boolean = false,
    /** Supports pinned host buffer */
    val hostBuffer: Boolean = false,
    /** Can create buffers from a host pointer */
    val bufferFromHostPtr: Boolean = false,
    /** Supports event synchronisation */
    val events: Boolean = false
)

/**
 * All device properties.
 * Mirrors `struct ggml_backend_dev_props`.
 */
data class GGMLBackendDeviceProps(
    val name: String,
    val description: String,
    /** Free memory in bytes */
    val memoryFree: ULong,
    /** Total memory in bytes */
    val memoryTotal: ULong,
    val type: io.github.kotlinmania.llama.core.GGMLBackendDeviceType,
    /** PCI bus id or null if unknown */
    val deviceId: String? = null,
    val caps: io.github.kotlinmania.llama.core.GGMLBackendDeviceCaps = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendDeviceCaps()
)

/**
 * Interface representing a backend device.
 * Mirrors the C `ggml_backend_device` opaque struct and associated free functions.
 */
interface GGMLBackendDevice {

    /** `ggml_backend_dev_name` */
    fun getName(): String

    /** `ggml_backend_dev_description` */
    fun getDescription(): String

    /** `ggml_backend_dev_memory` – returns (free, total) in bytes */
    fun getMemory(): Pair<ULong, ULong>

    /** `ggml_backend_dev_type` */
    fun getType(): io.github.kotlinmania.llama.core.GGMLBackendDeviceType

    /** `ggml_backend_dev_get_props` */
    fun getProps(): io.github.kotlinmania.llama.core.GGMLBackendDeviceProps

    /** `ggml_backend_dev_backend_reg` */
    fun getBackendReg(): io.github.kotlinmania.llama.core.GGMLBackendReg?

    /** `ggml_backend_dev_init` */
    fun initBackend(params: String? = null): io.github.kotlinmania.llama.core.GGMLBackend?

    /** `ggml_backend_dev_buffer_type` */
    fun getBufferType(): io.github.kotlinmania.llama.core.GGMLBackendBufferType

    /** `ggml_backend_dev_host_buffer_type` */
    fun getHostBufferType(): io.github.kotlinmania.llama.core.GGMLBackendBufferType? {
        return null
    }

    /** `ggml_backend_dev_buffer_from_host_ptr` */
    fun bufferFromHostPtr(ptr: ByteArray, size: ULong, maxTensorSize: ULong): io.github.kotlinmania.llama.core.GGMLBackendBuffer? {
        return null
    }

    /** `ggml_backend_dev_supports_op` */
    fun supportsOp(op: io.github.kotlinmania.llama.core.GGMLTensor): Boolean

    /** `ggml_backend_dev_supports_buft` */
    fun supportsBufferType(buft: io.github.kotlinmania.llama.core.GGMLBackendBufferType): Boolean

    /** `ggml_backend_dev_offload_op` */
    fun offloadOp(op: io.github.kotlinmania.llama.core.GGMLTensor): Boolean {
        return supportsOp(op)
    }

    /** `ggml_backend_event_new` – create an event on this device */
    fun newEvent(): io.github.kotlinmania.llama.core.GGMLBackendEvent? {
        return null
    }
}

// ---------------------------------------------------------------------------
// Backend registration  (ggml_backend_reg_t)
// ---------------------------------------------------------------------------

/**
 * A feature flag exposed by a backend registration.
 * Mirrors `struct ggml_backend_feature`.
 */
data class GGMLBackendFeature(
    val name: String,
    val value: String
)

/**
 * Interface for a backend registration entry.
 * Mirrors the C `ggml_backend_reg` opaque struct and associated free functions.
 */
interface GGMLBackendReg {

    /** `ggml_backend_reg_name` */
    fun getName(): String

    /** `ggml_backend_reg_dev_count` */
    fun getDeviceCount(): ULong

    /** `ggml_backend_reg_dev_get` */
    fun getDevice(index: ULong): io.github.kotlinmania.llama.core.GGMLBackendDevice?

    /** `ggml_backend_reg_get_proc_address` – retrieve an extension function by name */
    fun getProcAddress(name: String): Any? {
        return null
    }

    /** `ggml_backend_get_features` – optional list of feature flags */
    fun getFeatures(): List<io.github.kotlinmania.llama.core.GGMLBackendFeature> {
        return emptyList()
    }
}

// ---------------------------------------------------------------------------
// Backend registry entry (legacy convenience wrapper)
// ---------------------------------------------------------------------------

/**
 * Simple data holder used by [io.github.kotlinmania.llama.core.GGMLBackendRegistry] to track registered backends.
 * This is a Kotlin-side convenience; the C API uses `ggml_backend_reg_t` directly.
 */
data class GGMLBackendRegistration(
    val name: String,
    val initFunction: (String?) -> io.github.kotlinmania.llama.core.GGMLBackend?,
    val defaultBufferType: io.github.kotlinmania.llama.core.GGMLBackendBufferType,
    val userData: Any? = null
)

// ---------------------------------------------------------------------------
// Global backend & device registry
// ---------------------------------------------------------------------------

/**
 * Global backend registry for managing available backends and devices.
 * Combines the C functions:
 *   ggml_backend_register, ggml_backend_device_register,
 *   ggml_backend_reg_count / _get / _by_name,
 *   ggml_backend_dev_count / _get / _by_name / _by_type,
 *   ggml_backend_init_by_name / _by_type / _best,
 *   ggml_backend_load / _unload / _load_all / _load_all_from_path.
 */
object GGMLBackendRegistry {
    private val backends = mutableListOf<io.github.kotlinmania.llama.core.GGMLBackendRegistration>()
    private val regs = mutableListOf<io.github.kotlinmania.llama.core.GGMLBackendReg>()
    private val devices = mutableListOf<io.github.kotlinmania.llama.core.GGMLBackendDevice>()
    private var initialized = false

    // -- legacy registration helpers ----------------------------------------

    /**
     * Register a backend with the legacy registration wrapper.
     */
    fun register(registration: io.github.kotlinmania.llama.core.GGMLBackendRegistration) {
        backends.add(registration)
    }

    /**
     * Initialise the registry with built-in backends.
     */
    fun init() {
        if (initialized) return
        initialized = true

        register(
            _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendRegistration(
                name = "CPU",
                initFunction = { _ -> _root_ide_package_.io.github.kotlinmania.llama.core.GGMLCpuBackend() },
                defaultBufferType = _root_ide_package_.io.github.kotlinmania.llama.core.createDefaultCpuBufferType()
            )
        )
    }

    /**
     * Get the number of registered (legacy) backends.
     */
    fun getCount(): Int {
        init()
        return backends.size
    }

    /**
     * Find a legacy backend by name (case-insensitive). Returns index or null.
     */
    fun findByName(name: String): Int? {
        init()
        return backends.indexOfFirst { it.name.equals(name, ignoreCase = true) }.takeIf { it >= 0 }
    }

    /**
     * Get legacy backend name by index.
     */
    fun getName(index: Int): String? {
        init()
        return backends.getOrNull(index)?.name
    }

    /**
     * Initialise a backend by legacy index.
     */
    fun initBackend(index: Int, params: String? = null): io.github.kotlinmania.llama.core.GGMLBackend? {
        init()
        return backends.getOrNull(index)?.initFunction?.invoke(params)
    }

    /**
     * Get default buffer type by legacy index.
     */
    fun getDefaultBufferType(index: Int): io.github.kotlinmania.llama.core.GGMLBackendBufferType? {
        init()
        return backends.getOrNull(index)?.defaultBufferType
    }

    /**
     * Allocate a buffer via the legacy index.
     */
    fun allocBuffer(index: Int, size: ULong): io.github.kotlinmania.llama.core.GGMLBackendBuffer? {
        return getDefaultBufferType(index)?.allocBuffer(size)
    }

    // -- ggml_backend_reg_t registration ------------------------------------

    /** `ggml_backend_register` */
    fun registerReg(reg: io.github.kotlinmania.llama.core.GGMLBackendReg) {
        regs.add(reg)
    }

    /** `ggml_backend_reg_count` */
    fun regCount(): ULong {
        init()
        return regs.size.toULong()
    }

    /** `ggml_backend_reg_get` */
    fun regGet(index: ULong): io.github.kotlinmania.llama.core.GGMLBackendReg? {
        init()
        return regs.getOrNull(index.toInt())
    }

    /** `ggml_backend_reg_by_name` */
    fun regByName(name: String): io.github.kotlinmania.llama.core.GGMLBackendReg? {
        init()
        return regs.firstOrNull { it.getName().equals(name, ignoreCase = true) }
    }

    // -- device registration ------------------------------------------------

    /** `ggml_backend_device_register` */
    fun registerDevice(device: io.github.kotlinmania.llama.core.GGMLBackendDevice) {
        devices.add(device)
    }

    /** `ggml_backend_dev_count` */
    fun deviceCount(): ULong {
        init()
        return devices.size.toULong()
    }

    /** `ggml_backend_dev_get` */
    fun deviceGet(index: ULong): io.github.kotlinmania.llama.core.GGMLBackendDevice? {
        init()
        return devices.getOrNull(index.toInt())
    }

    /** `ggml_backend_dev_by_name` */
    fun deviceByName(name: String): io.github.kotlinmania.llama.core.GGMLBackendDevice? {
        init()
        return devices.firstOrNull { it.getName().equals(name, ignoreCase = true) }
    }

    /** `ggml_backend_dev_by_type` */
    fun deviceByType(type: io.github.kotlinmania.llama.core.GGMLBackendDeviceType): io.github.kotlinmania.llama.core.GGMLBackendDevice? {
        init()
        return devices.firstOrNull { it.getType() == type }
    }

    // -- direct backend initialisation --------------------------------------

    /** `ggml_backend_init_by_name` */
    fun initByName(name: String, params: String? = null): io.github.kotlinmania.llama.core.GGMLBackend? {
        return deviceByName(name)?.initBackend(params)
    }

    /** `ggml_backend_init_by_type` */
    fun initByType(type: io.github.kotlinmania.llama.core.GGMLBackendDeviceType, params: String? = null): io.github.kotlinmania.llama.core.GGMLBackend? {
        return deviceByType(type)?.initBackend(params)
    }

    /** `ggml_backend_init_best` */
    fun initBest(): io.github.kotlinmania.llama.core.GGMLBackend? {
        return (deviceByType(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendDeviceType.GPU)
            ?: deviceByType(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendDeviceType.CPU))?.initBackend()
    }

    // -- dynamic loading (no-op on native, used for dynamic backend plugins) --

    /** `ggml_backend_load` */
    fun load(path: String): io.github.kotlinmania.llama.core.GGMLBackendReg? { return null }

    /** `ggml_backend_unload` */
    fun unload(reg: io.github.kotlinmania.llama.core.GGMLBackendReg) {}

    /** `ggml_backend_load_all` */
    fun loadAll() {
    }

    /** `ggml_backend_load_all_from_path` */
    fun loadAllFromPath(dirPath: String) {
    }
}

// Registry free functions moved to GGMLBackendUtils.kt (ggml-backend-reg.cpp)

// ---------------------------------------------------------------------------
// Backend scheduler  (ggml_backend_sched_t)
// ---------------------------------------------------------------------------

/**
 * Evaluation callback for the scheduler.
 * Mirrors `ggml_backend_sched_eval_callback`.
 *
 * When [ask] is true the scheduler wants to know whether the user wishes to observe
 * the tensor [t]. When [ask] is false the tensor is being presented for observation;
 * returning false cancels the graph compute.
 */
/**
 * Eval callback for the scheduler.
 * C: `ggml_backend_sched_eval_callback`
 * @param t the tensor being evaluated
 * @param ask true = asking if data is needed, false = data is ready
 * @param userData opaque user data
 * @return true to continue, false to stop
 */
typealias GGMLBackendSchedEvalCallback = (t: io.github.kotlinmania.llama.core.GGMLTensor, ask: Boolean, userData: Any?) -> Boolean

/**
 * Backend scheduler – coordinates multiple backends, handles buffer allocation,
 * tensor assignment, and cross-backend copies.
 *
 * Faithful transliteration of `struct ggml_backend_sched` from ggml-backend.cpp lines 774-828.
 * All internal fields mirror the C struct. Public API methods mirror the C free functions.
 */
class GGMLBackendSched private constructor(
    val nBackends: Int,
    val nCopies: Int,
    val opOffload: Boolean,
    val debug: Int,
    val debugRealloc: Int,
    graphSize: Int
) {
    // --- state flags ---
    var isReset: Boolean = true
    var isAlloc: Boolean = false

    // --- backends and buffer types ---
    val backends: Array<io.github.kotlinmania.llama.core.GGMLBackend?> = arrayOfNulls(_root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_BACKENDS)
    val bufts: Array<io.github.kotlinmania.llama.core.GGMLBackendBufferType?> = arrayOfNulls(_root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_BACKENDS)

    // --- graph allocator (ggml_gallocr_t) ---
    // Will be typed as GGMLGAllocr when ggml-alloc.c gallocr is fully ported.
    // For now, Any? matches the opaque pointer pattern until gallocr is fully ported.
    var galloc: Any? = null

    // --- hash map of the nodes in the graph ---
    val hashSet: io.github.kotlinmania.llama.core.GGMLHashSet =
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLHashSet(
            _root_ide_package_.io.github.kotlinmania.llama.core.GGMLHashSet.ggml_hash_size(graphSize)
        )
    val hvTensorBackendIds: IntArray = IntArray(hashSet.size) { -1 }
    val hvTensorCopies: Array<io.github.kotlinmania.llama.core.GGMLTensor?> = arrayOfNulls(hashSet.size * _root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_BACKENDS * _root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_COPIES)

    // --- per-node/leaf backend assignments ---
    private val nodesSize: Int = graphSize + graphSize * _root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_SPLIT_INPUTS * 2
    var nodeBackendIds: IntArray = IntArray(nodesSize)
    var leafBackendIds: IntArray = IntArray(nodesSize)
    var prevNodeBackendIds: IntArray = IntArray(nodesSize)
    var prevLeafBackendIds: IntArray = IntArray(nodesSize)

    // --- copy of the graph with modified inputs ---
    val graph: io.github.kotlinmania.llama.core.GGMLCGraph =
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLCGraph()

    // --- graph splits ---
    var splits: MutableList<io.github.kotlinmania.llama.core.GGMLBackendSchedSplit> = MutableList(16) { _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendSchedSplit() }
    var nSplits: Int = 0
    var splitsCapacity: Int = 16

    // --- pipeline parallelism support ---
    var curCopy: Int = 0
    var nextCopy: Int = 0
    val events: Array<Array<io.github.kotlinmania.llama.core.GGMLBackendEvent?>> = Array(_root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_BACKENDS) { arrayOfNulls(
        _root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_COPIES
    ) }
    val graphInputs: Array<io.github.kotlinmania.llama.core.GGMLTensor?> = arrayOfNulls(_root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_SPLIT_INPUTS)
    var nGraphInputs: Int = 0

    // --- context for temporary tensor allocations during split_graph ---
    var ctx: io.github.kotlinmania.llama.core.GGMLContext? = null

    // --- eval callback ---
    var callbackEval: io.github.kotlinmania.llama.core.GGMLBackendSchedEvalCallback? = null
    var callbackEvalUserData: Any? = null

    // --- context buffer (Kotlin heap-allocated, no manual memory management needed) ---
    val contextBufferSize: Int = graphSize * _root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_SPLIT_INPUTS * 2
    var contextBuffer: ByteArray = ByteArray(contextBufferSize)

    // --- debug ---
    var debugGraphSize: Int = 0
    var debugPrevGraphSize: Int = 0

    // =====================================================================
    // C macro equivalents: hash_id, tensor_backend_id, tensor_id_copy, tensor_copy
    // =====================================================================

    /** C: `hash_id(tensor)` → `ggml_hash_find_or_insert(&sched->hash_set, tensor)` */
    fun hashId(tensor: io.github.kotlinmania.llama.core.GGMLTensor): Int =
        _root_ide_package_.io.github.kotlinmania.llama.core.ggml_hash_find_or_insert(hashSet, tensor)

    /** C: `tensor_backend_id(tensor)` → `sched->hv_tensor_backend_ids[hash_id(tensor)]` */
    fun tensorBackendId(tensor: io.github.kotlinmania.llama.core.GGMLTensor): Int = hvTensorBackendIds[hashId(tensor)]

    /** Set the backend id for a tensor in the hash map. */
    fun setTensorBackendId(tensor: io.github.kotlinmania.llama.core.GGMLTensor, id: Int) {
        hvTensorBackendIds[hashId(tensor)] = id
    }

    /**
     * C: `tensor_id_copy(id, backend_id, copy_id)`
     * → `sched->hv_tensor_copies[(id) * n_backends * n_copies + (backend_id) * n_copies + (copy_id)]`
     */
    fun tensorIdCopy(id: Int, backendId: Int, copyId: Int): io.github.kotlinmania.llama.core.GGMLTensor? =
        hvTensorCopies[id * nBackends * nCopies + backendId * nCopies + copyId]

    fun setTensorIdCopy(id: Int, backendId: Int, copyId: Int, value: io.github.kotlinmania.llama.core.GGMLTensor?) {
        hvTensorCopies[id * nBackends * nCopies + backendId * nCopies + copyId] = value
    }

    /** C: `tensor_copy(tensor, backend_id, copy_id)` → `tensor_id_copy(hash_id(tensor), ...)` */
    fun tensorCopy(tensor: io.github.kotlinmania.llama.core.GGMLTensor, backendId: Int, copyId: Int): io.github.kotlinmania.llama.core.GGMLTensor? =
        tensorIdCopy(hashId(tensor), backendId, copyId)

    fun setTensorCopy(tensor: io.github.kotlinmania.llama.core.GGMLTensor, backendId: Int, copyId: Int, value: io.github.kotlinmania.llama.core.GGMLTensor?) {
        setTensorIdCopy(hashId(tensor), backendId, copyId, value)
    }

    // =====================================================================
    // Factory (companion object)
    // =====================================================================

    companion object {
        /**
         * `ggml_backend_sched_new` — C: ggml-backend.cpp lines 1727-1793.
         *
         * Create a new backend scheduler. Backends with a lower index are given
         * higher priority. The last backend MUST be CPU.
         */
        fun new(
            backends: List<io.github.kotlinmania.llama.core.GGMLBackend>,
            bufts: List<io.github.kotlinmania.llama.core.GGMLBackendBufferType?>?,
            graphSize: Int,
            parallel: Boolean,
            opOffload: Boolean
        ): GGMLBackendSched {
            require(backends.isNotEmpty())
            require(backends.size <= _root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_BACKENDS)

            val nCopies = if (parallel) _root_ide_package_.io.github.kotlinmania.llama.core.GGML_SCHED_MAX_COPIES else 1

            val sched = GGMLBackendSched(
                nBackends = backends.size,
                nCopies = nCopies,
                opOffload = opOffload,
                debug = 0,
                debugRealloc = 0,
                graphSize = graphSize
            )

            for (b in backends.indices) {
                sched.backends[b] = backends[b]
                sched.bufts[b] = bufts?.getOrNull(b)
                    ?: _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendGetDefaultBufferType(backends[b])
                require(
                    _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSupportsBuft(
                        backends[b],
                        sched.bufts[b]!!
                    )
                )

                if (nCopies > 1) {
                    for (c in 0 until nCopies) {
                        sched.events[b][c] =
                            _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendEventNew(backends[b].getDevice())
                    }
                }
            }

            // galloc = ggmlGallocrNewN(sched.bufts, backends.size) — when gallocr is ported
            sched.reset()

            return sched
        }
    }

    // =====================================================================
    // Public API methods — mirror C free functions
    // =====================================================================

    /** `ggml_backend_sched_free` — C: lines 1796-1818 */
    fun free() {
        for (b in 0 until nBackends) {
            for (c in 0 until nCopies) {
                _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendEventFree(events[b][c])
            }
        }
        // galloc?.free() — when gallocr is ported
        ctx = null
        // In Kotlin, GC handles the rest (no manual free needed for arrays)
    }

    /** `ggml_backend_sched_reset` — C: lines 1821-1831 */
    fun reset() {
        if (!isReset) {
            hashSet.reset()
            hvTensorBackendIds.fill(-1)
            hvTensorCopies.fill(null)
            isReset = true
        }
        isAlloc = false
    }

    /**
     * `ggml_backend_sched_reserve_size` — C: lines 1833-1844.
     * Reserve with explicit per-backend sizes.
     */
    fun reserveSize(measureGraph: io.github.kotlinmania.llama.core.GGMLCGraph, sizes: ULongArray) {
        require(hashSet.size >= measureGraph.nNodes + measureGraph.nLeafs)
        reset()
        synchronize()
        _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSchedSplitGraph(this, measureGraph)
        // ggmlGallocrReserveNSize(galloc, graph, nodeBackendIds, leafBackendIds, sizes)
    }

    /** `ggml_backend_sched_reserve` — C: lines 1847-1861. Returns true on success. */
    fun reserve(measureGraph: io.github.kotlinmania.llama.core.GGMLCGraph): Boolean {
        require(hashSet.size >= measureGraph.nNodes + measureGraph.nLeafs)
        synchronize()
        _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSchedSplitGraph(this, measureGraph)
        // if (!ggmlGallocrReserveN(galloc, graph, nodeBackendIds, leafBackendIds)) return false
        reset()
        return true
    }

    /** `ggml_backend_sched_alloc_graph` — C: lines 1864-1881. Returns true on success. */
    fun allocGraph(graph: io.github.kotlinmania.llama.core.GGMLCGraph): Boolean {
        require(hashSet.size >= graph.nNodes + graph.nLeafs)
        require(!isAlloc)

        curCopy = nextCopy
        nextCopy = (nextCopy + 1) % nCopies

        _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSchedSplitGraph(this, graph)

        if (!_root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSchedAllocSplits(this)) {
            return false
        }

        isAlloc = true
        return true
    }

    /** `ggml_backend_sched_graph_compute` — C: lines 1883-1887 */
    fun graphCompute(graph: io.github.kotlinmania.llama.core.GGMLCGraph): io.github.kotlinmania.llama.core.GGMLStatus {
        val err = graphComputeAsync(graph)
        synchronize()
        return err
    }

    /** `ggml_backend_sched_graph_compute_async` — C: lines 1889-1902 */
    fun graphComputeAsync(graph: io.github.kotlinmania.llama.core.GGMLCGraph): io.github.kotlinmania.llama.core.GGMLStatus {
        if (!isReset && !isAlloc) {
            reset()
        }
        if (!isAlloc) {
            if (!allocGraph(graph)) {
                return _root_ide_package_.io.github.kotlinmania.llama.core.GGMLStatus.ALLOC_FAILED
            }
        }
        return _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSchedComputeSplits(this)
    }

    /** `ggml_backend_sched_synchronize` — C: lines 1904-1915 */
    fun synchronize() {
        for (i in 0 until nBackends) {
            _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSynchronize(backends[i]!!)
        }
        if (!isAlloc) {
            nextCopy = 0
        }
    }

    /** `ggml_backend_sched_set_eval_callback` — C: lines 1917-1921 */
    fun setEvalCallback(callback: io.github.kotlinmania.llama.core.GGMLBackendSchedEvalCallback?, userData: Any? = null) {
        callbackEval = callback
        callbackEvalUserData = userData
    }

    /** `ggml_backend_sched_get_n_splits` */
    fun getNumSplits(): Int = nSplits

    /** `ggml_backend_sched_get_n_copies` */
    fun getNumCopies(): Int = nCopies

    /** `ggml_backend_sched_get_n_backends` */
    fun getNumBackends(): Int = nBackends

    /** `ggml_backend_sched_get_backend` */
    fun getBackend(index: Int): io.github.kotlinmania.llama.core.GGMLBackend? {
        require(index in 0 until nBackends)
        return backends[index]
    }

    /** `ggml_backend_sched_get_buffer_type` — C: lines 1944-1949 */
    fun getBufferType(backend: io.github.kotlinmania.llama.core.GGMLBackend): io.github.kotlinmania.llama.core.GGMLBackendBufferType? {
        val idx = _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSchedBackendId(this, backend)
        require(idx in 0 until nBackends)
        return bufts[idx]
    }

    /** `ggml_backend_sched_get_buffer_size` — C: lines 1952-1957 */
    fun getBufferSize(backend: io.github.kotlinmania.llama.core.GGMLBackend): ULong {
        val idx = _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSchedBackendId(this, backend)
        require(idx in 0 until nBackends)
        // return ggmlGallocrGetBufferSize(galloc, idx) — when gallocr is ported
        return 0UL
    }

    /** `ggml_backend_sched_set_tensor_backend` — C: lines 1960-1967 */
    fun setTensorBackend(node: io.github.kotlinmania.llama.core.GGMLTensor, backend: io.github.kotlinmania.llama.core.GGMLBackend) {
        val idx = _root_ide_package_.io.github.kotlinmania.llama.core.ggmlBackendSchedBackendId(this, backend)
        require(idx in 0 until nBackends)
        setTensorBackendId(node, idx)
        isReset = false
    }

    /** `ggml_backend_sched_get_tensor_backend` — C: lines 1969-1976 */
    fun getTensorBackend(node: io.github.kotlinmania.llama.core.GGMLTensor): io.github.kotlinmania.llama.core.GGMLBackend? {
        val idx = tensorBackendId(node)
        if (idx == -1) return null
        return backends[idx]
    }
}

// ---------------------------------------------------------------------------
// Meta backend – split axis & split state
// ---------------------------------------------------------------------------

/** Maximum devices in a meta backend. Mirrors `GGML_BACKEND_META_MAX_DEVICES`. */
const val GGML_BACKEND_META_MAX_DEVICES: Int = 16

/**
 * Axis along which a tensor is split for tensor parallelism.
 * Mirrors `enum ggml_backend_meta_split_axis`.
 */
enum class GGMLBackendMetaSplitAxis(val value: Int) {
    AXIS_0(0),
    AXIS_1(1),
    AXIS_2(2),
    AXIS_3(3),
    /** All values mirrored on every backend */
    MIRRORED(10),
    /** Each backend holds a partial sum */
    PARTIAL(11),
    /** Internal: no split */
    NONE(98),
    /** Internal: split state not yet determined */
    UNKNOWN(99);

    companion object {
        fun fromValue(v: Int): GGMLBackendMetaSplitAxis =
            entries.firstOrNull { it.value == v } ?: UNKNOWN
    }
}

// ggmlBackendMetaSplitAxisName moved to GGMLBackendUtils.kt

/**
 * Split state for a tensor in a meta-backend context.
 * Mirrors `struct ggml_backend_meta_split_state`.
 */
data class GGMLBackendMetaSplitState(
    val axis: io.github.kotlinmania.llama.core.GGMLBackendMetaSplitAxis,
    /**
     * Per-segment, per-device element counts. Outer loop = segments, inner = devices.
     * Size: nSegments * nDevices. Mirrors the C `ne[16*GGML_BACKEND_META_MAX_DEVICES]`.
     */
    val ne: LongArray = LongArray(16 * _root_ide_package_.io.github.kotlinmania.llama.core.GGML_BACKEND_META_MAX_DEVICES),
    val nSegments: UInt = 1u
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GGMLBackendMetaSplitState) return false
        return axis == other.axis && ne.contentEquals(other.ne) && nSegments == other.nSegments
    }

    override fun hashCode(): Int {
        var result = axis.hashCode()
        result = 31 * result + ne.contentHashCode()
        result = 31 * result + nSegments.hashCode()
        return result
    }
}

/**
 * Callback type for assigning split states to statically-allocated tensors.
 * Mirrors `ggml_backend_meta_get_split_state_t`.
 */
typealias GGMLBackendMetaGetSplitState = (tensor: io.github.kotlinmania.llama.core.GGMLTensor) -> io.github.kotlinmania.llama.core.GGMLBackendMetaSplitState

// ggmlBackendMetaDevice moved to GGMLBackendUtils.kt

// Graph copy utilities, GGMLBackendGraphCopy, ggmlBackendGraphCopy,
// ggmlBackendGraphCopyFree, GGMLBackendEvalCallback,
// ggmlBackendCompareGraphBackend all moved to GGMLBackendUtils.kt

// =========================================================================
// Free-standing C-style API functions have been moved to GGMLBackendUtils.kt
// (the port-lint target file for ggml-backend.cpp).
// =========================================================================

