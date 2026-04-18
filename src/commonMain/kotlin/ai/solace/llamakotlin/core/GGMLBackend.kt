// port-lint: source ggml/include/ggml-backend.h
package ai.solace.llamakotlin.core

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
    fun allocBuffer(size: ULong): GGMLBackendBuffer?

    /** `ggml_backend_buft_get_alignment` */
    fun getAlignment(): UInt

    /** `ggml_backend_buft_get_max_size` */
    fun getMaxSize(): ULong

    /** `ggml_backend_buft_get_alloc_size` – allocation size required for a tensor */
    fun getAllocSize(tensor: GGMLTensor): ULong {
        return calculateTensorByteSize(tensor)
    }

    /** `ggml_backend_buft_is_host` */
    fun isHost(): Boolean

    /** `ggml_backend_buft_get_device` */
    fun getDevice(): GGMLBackendDevice? {
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
    fun getType(): GGMLBackendBufferType

    /** `ggml_backend_buffer_name` */
    fun getName(): String

    /** `ggml_backend_buffer_get_base` */
    fun getBase(): Any?

    /** `ggml_backend_buffer_get_size` */
    fun getSize(): ULong

    /** `ggml_backend_buffer_free` */
    fun free()

    /** `ggml_backend_buffer_init_tensor` */
    fun initTensor(tensor: GGMLTensor): GGMLStatus {
        return GGMLStatus.SUCCESS
    }

    /** `ggml_backend_buffer_set_tensor` – set tensor data from host memory */
    fun setTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong)

    /** `ggml_backend_buffer_get_tensor` – get tensor data into host memory */
    fun getTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong)

    /** Copy tensor data between buffers – returns true if the copy was handled */
    fun copyTensor(src: GGMLTensor, dst: GGMLTensor): Boolean

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
    fun getAllocSize(tensor: GGMLTensor): ULong {
        return getType().getAllocSize(tensor)
    }

    /** `ggml_backend_buffer_is_host` */
    fun isHost(): Boolean {
        return getType().isHost()
    }

    /** `ggml_backend_buffer_set_usage` */
    fun setUsage(usage: GGMLBackendBufferUsage) {
        // Default: no-op; implementations may track usage
    }

    /** `ggml_backend_buffer_get_usage` */
    fun getUsage(): GGMLBackendBufferUsage {
        return GGMLBackendBufferUsage.ANY
    }

    /** `ggml_backend_buffer_reset` – reset the buffer (clear allocations) */
    fun reset() {
        // Default: no-op
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
    fun getDefaultBufferType(): GGMLBackendBufferType

    /** `ggml_backend_alloc_buffer` */
    fun allocBuffer(size: ULong): GGMLBackendBuffer? {
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
    fun setTensorAsync(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val buffer = tensor.buffer
        buffer?.setTensor(tensor, data, offset, size)
    }

    /** `ggml_backend_tensor_get_async` */
    fun getTensorAsync(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val buffer = tensor.buffer
        buffer?.getTensor(tensor, data, offset, size)
    }

    /**
     * `ggml_backend_tensor_set_2d_async`
     * Set 2-D strided tensor data asynchronously.
     */
    fun setTensor2dAsync(
        tensor: GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong,
        nCopies: ULong,
        strideTensor: ULong,
        strideData: ULong
    ) {
        TODO("port from ggml-backend.h")
    }

    /**
     * `ggml_backend_tensor_get_2d_async`
     * Get 2-D strided tensor data asynchronously.
     */
    fun getTensor2dAsync(
        tensor: GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong,
        nCopies: ULong,
        strideTensor: ULong,
        strideData: ULong
    ) {
        TODO("port from ggml-backend.h")
    }

    /**
     * `ggml_backend_tensor_copy_async`
     * Asynchronous tensor copy between backends. Returns false if unsupported.
     */
    fun copyTensorAsync(backendSrc: GGMLBackend, src: GGMLTensor, dst: GGMLTensor): Boolean {
        return false
    }

    /** `ggml_backend_synchronize` */
    fun synchronize() {}

    /** `ggml_backend_graph_plan_create` */
    fun graphPlanCreate(graph: GGMLCGraph): Any? {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_graph_plan_free` */
    fun graphPlanFree(plan: Any?) {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_graph_plan_compute` */
    fun graphPlanCompute(plan: Any?): GGMLStatus {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_graph_compute` */
    fun graphCompute(graph: GGMLCGraph): GGMLStatus

    /** `ggml_backend_graph_compute_async` */
    fun graphComputeAsync(graph: GGMLCGraph): GGMLStatus {
        return graphCompute(graph)
    }

    /** `ggml_backend_supports_op` – NOTE: will be removed; use device version instead */
    fun supportsOp(tensor: GGMLTensor): Boolean

    /** `ggml_backend_supports_buft` */
    fun supportsBufferType(bufferType: GGMLBackendBufferType): Boolean

    /** `ggml_backend_offload_op` */
    fun offloadOp(tensor: GGMLTensor): Boolean {
        return supportsOp(tensor)
    }

    /** `ggml_backend_get_device` */
    fun getDevice(): GGMLBackendDevice? {
        return null
    }
}

// ---------------------------------------------------------------------------
// Free-standing tensor helpers (not tied to a backend instance)
// ---------------------------------------------------------------------------

/**
 * `ggml_backend_tensor_copy` – copy tensor data between different backends.
 */
fun ggmlBackendTensorCopy(src: GGMLTensor, dst: GGMLTensor) {
    TODO("port from ggml-backend.h")
}

/**
 * `ggml_backend_tensor_set` – synchronous set from host memory.
 */
fun ggmlBackendTensorSet(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    val buffer = tensor.buffer ?: error("tensor has no buffer")
    buffer.setTensor(tensor, data, offset, size)
}

/**
 * `ggml_backend_tensor_get` – synchronous get into host memory.
 */
fun ggmlBackendTensorGet(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    val buffer = tensor.buffer ?: error("tensor has no buffer")
    buffer.getTensor(tensor, data, offset, size)
}

/**
 * `ggml_backend_tensor_set_2d` – synchronous 2-D strided set.
 */
fun ggmlBackendTensorSet2d(
    tensor: GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong,
    nCopies: ULong,
    strideTensor: ULong,
    strideData: ULong
) {
    TODO("port from ggml-backend.h")
}

/**
 * `ggml_backend_tensor_get_2d` – synchronous 2-D strided get.
 */
fun ggmlBackendTensorGet2d(
    tensor: GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong,
    nCopies: ULong,
    strideTensor: ULong,
    strideData: ULong
) {
    TODO("port from ggml-backend.h")
}

/**
 * `ggml_backend_tensor_memset` – fill a region of a tensor with a byte value.
 */
fun ggmlBackendTensorMemset(tensor: GGMLTensor, value: UByte, offset: ULong, size: ULong) {
    TODO("port from ggml-backend.h")
}

/**
 * `ggml_backend_tensor_alloc` – allocate a tensor within a buffer at a given address.
 */
fun ggmlBackendTensorAlloc(buffer: GGMLBackendBuffer, tensor: GGMLTensor, addr: Any?): GGMLStatus {
    TODO("port from ggml-backend.h")
}

/**
 * `ggml_backend_view_init` – initialise a tensor view.
 */
fun ggmlBackendViewInit(tensor: GGMLTensor): GGMLStatus {
    TODO("port from ggml-backend.h")
}

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
    fun record(backend: GGMLBackend)

    /** `ggml_backend_event_synchronize` */
    fun synchronize()

    /** `ggml_backend_event_wait` */
    fun wait(backend: GGMLBackend)
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
    val type: GGMLBackendDeviceType,
    /** PCI bus id or null if unknown */
    val deviceId: String? = null,
    val caps: GGMLBackendDeviceCaps = GGMLBackendDeviceCaps()
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
    fun getType(): GGMLBackendDeviceType

    /** `ggml_backend_dev_get_props` */
    fun getProps(): GGMLBackendDeviceProps

    /** `ggml_backend_dev_backend_reg` */
    fun getBackendReg(): GGMLBackendReg?

    /** `ggml_backend_dev_init` */
    fun initBackend(params: String? = null): GGMLBackend?

    /** `ggml_backend_dev_buffer_type` */
    fun getBufferType(): GGMLBackendBufferType

    /** `ggml_backend_dev_host_buffer_type` */
    fun getHostBufferType(): GGMLBackendBufferType? {
        return null
    }

    /** `ggml_backend_dev_buffer_from_host_ptr` */
    fun bufferFromHostPtr(ptr: ByteArray, size: ULong, maxTensorSize: ULong): GGMLBackendBuffer? {
        return null
    }

    /** `ggml_backend_dev_supports_op` */
    fun supportsOp(op: GGMLTensor): Boolean

    /** `ggml_backend_dev_supports_buft` */
    fun supportsBufferType(buft: GGMLBackendBufferType): Boolean

    /** `ggml_backend_dev_offload_op` */
    fun offloadOp(op: GGMLTensor): Boolean {
        return supportsOp(op)
    }

    /** `ggml_backend_event_new` – create an event on this device */
    fun newEvent(): GGMLBackendEvent? {
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
    fun getDevice(index: ULong): GGMLBackendDevice?

    /** `ggml_backend_reg_get_proc_address` – retrieve an extension function by name */
    fun getProcAddress(name: String): Any? {
        return null
    }

    /** `ggml_backend_get_features` – optional list of feature flags */
    fun getFeatures(): List<GGMLBackendFeature> {
        return emptyList()
    }
}

// ---------------------------------------------------------------------------
// Backend registry entry (legacy convenience wrapper)
// ---------------------------------------------------------------------------

/**
 * Simple data holder used by [GGMLBackendRegistry] to track registered backends.
 * This is a Kotlin-side convenience; the C API uses `ggml_backend_reg_t` directly.
 */
data class GGMLBackendRegistration(
    val name: String,
    val initFunction: (String?) -> GGMLBackend?,
    val defaultBufferType: GGMLBackendBufferType,
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
    private val backends = mutableListOf<GGMLBackendRegistration>()
    private val regs = mutableListOf<GGMLBackendReg>()
    private val devices = mutableListOf<GGMLBackendDevice>()
    private var initialized = false

    // -- legacy registration helpers ----------------------------------------

    /**
     * Register a backend with the legacy registration wrapper.
     */
    fun register(registration: GGMLBackendRegistration) {
        backends.add(registration)
    }

    /**
     * Initialise the registry with built-in backends.
     */
    fun init() {
        if (initialized) return
        initialized = true

        register(GGMLBackendRegistration(
            name = "CPU",
            initFunction = { _ -> GGMLCpuBackend() },
            defaultBufferType = GGMLCpuBufferType()
        ))
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
    fun initBackend(index: Int, params: String? = null): GGMLBackend? {
        init()
        return backends.getOrNull(index)?.initFunction?.invoke(params)
    }

    /**
     * Get default buffer type by legacy index.
     */
    fun getDefaultBufferType(index: Int): GGMLBackendBufferType? {
        init()
        return backends.getOrNull(index)?.defaultBufferType
    }

    /**
     * Allocate a buffer via the legacy index.
     */
    fun allocBuffer(index: Int, size: ULong): GGMLBackendBuffer? {
        return getDefaultBufferType(index)?.allocBuffer(size)
    }

    // -- ggml_backend_reg_t registration ------------------------------------

    /** `ggml_backend_register` */
    fun registerReg(reg: GGMLBackendReg) {
        regs.add(reg)
    }

    /** `ggml_backend_reg_count` */
    fun regCount(): ULong {
        init()
        return regs.size.toULong()
    }

    /** `ggml_backend_reg_get` */
    fun regGet(index: ULong): GGMLBackendReg? {
        init()
        return regs.getOrNull(index.toInt())
    }

    /** `ggml_backend_reg_by_name` */
    fun regByName(name: String): GGMLBackendReg? {
        init()
        return regs.firstOrNull { it.getName().equals(name, ignoreCase = true) }
    }

    // -- device registration ------------------------------------------------

    /** `ggml_backend_device_register` */
    fun registerDevice(device: GGMLBackendDevice) {
        devices.add(device)
    }

    /** `ggml_backend_dev_count` */
    fun deviceCount(): ULong {
        init()
        return devices.size.toULong()
    }

    /** `ggml_backend_dev_get` */
    fun deviceGet(index: ULong): GGMLBackendDevice? {
        init()
        return devices.getOrNull(index.toInt())
    }

    /** `ggml_backend_dev_by_name` */
    fun deviceByName(name: String): GGMLBackendDevice? {
        init()
        return devices.firstOrNull { it.getName().equals(name, ignoreCase = true) }
    }

    /** `ggml_backend_dev_by_type` */
    fun deviceByType(type: GGMLBackendDeviceType): GGMLBackendDevice? {
        init()
        return devices.firstOrNull { it.getType() == type }
    }

    // -- direct backend initialisation --------------------------------------

    /** `ggml_backend_init_by_name` */
    fun initByName(name: String, params: String? = null): GGMLBackend? {
        return deviceByName(name)?.initBackend(params)
    }

    /** `ggml_backend_init_by_type` */
    fun initByType(type: GGMLBackendDeviceType, params: String? = null): GGMLBackend? {
        return deviceByType(type)?.initBackend(params)
    }

    /** `ggml_backend_init_best` */
    fun initBest(): GGMLBackend? {
        return (deviceByType(GGMLBackendDeviceType.GPU)
            ?: deviceByType(GGMLBackendDeviceType.CPU))?.initBackend()
    }

    // -- dynamic loading stubs ----------------------------------------------

    /** `ggml_backend_load` */
    fun load(path: String): GGMLBackendReg? {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_unload` */
    fun unload(reg: GGMLBackendReg) {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_load_all` */
    fun loadAll() {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_load_all_from_path` */
    fun loadAllFromPath(dirPath: String) {
        TODO("port from ggml-backend.h")
    }
}

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
typealias GGMLBackendSchedEvalCallback = (t: GGMLTensor, ask: Boolean) -> Boolean

/**
 * Backend scheduler – coordinates multiple backends, handles buffer allocation,
 * tensor assignment, and cross-backend copies.
 * Mirrors the C `ggml_backend_sched` opaque struct and associated free functions.
 */
class GGMLBackendSched private constructor(
    private val backends: List<GGMLBackend>,
    private val bufferTypes: List<GGMLBackendBufferType?>,
    private val graphSize: ULong,
    private val parallel: Boolean,
    private val opOffload: Boolean
) {
    private var evalCallback: GGMLBackendSchedEvalCallback? = null

    companion object {
        /**
         * `ggml_backend_sched_new`
         *
         * Create a new backend scheduler. Backends with a lower index are given
         * higher priority.
         */
        fun new(
            backends: List<GGMLBackend>,
            bufferTypes: List<GGMLBackendBufferType?>?,
            graphSize: ULong,
            parallel: Boolean,
            opOffload: Boolean
        ): GGMLBackendSched {
            return GGMLBackendSched(
                backends = backends,
                bufferTypes = bufferTypes ?: List(backends.size) { null },
                graphSize = graphSize,
                parallel = parallel,
                opOffload = opOffload
            )
        }
    }

    /** `ggml_backend_sched_free` */
    fun free() {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_reserve_size` – reserve with explicit per-backend sizes */
    fun reserveSize(measureGraph: GGMLCGraph, sizes: ULongArray) {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_reserve` – returns true on success */
    fun reserve(measureGraph: GGMLCGraph): Boolean {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_get_n_backends` */
    fun getNumBackends(): Int = backends.size

    /** `ggml_backend_sched_get_backend` */
    fun getBackend(index: Int): GGMLBackend? = backends.getOrNull(index)

    /** `ggml_backend_sched_get_n_splits` */
    fun getNumSplits(): Int {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_get_n_copies` */
    fun getNumCopies(): Int {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_get_buffer_type` */
    fun getBufferType(backend: GGMLBackend): GGMLBackendBufferType? {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_get_buffer_size` */
    fun getBufferSize(backend: GGMLBackend): ULong {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_set_tensor_backend` */
    fun setTensorBackend(node: GGMLTensor, backend: GGMLBackend) {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_get_tensor_backend` */
    fun getTensorBackend(node: GGMLTensor): GGMLBackend? {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_split_graph` */
    fun splitGraph(graph: GGMLCGraph) {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_alloc_graph` – returns true on success */
    fun allocGraph(graph: GGMLCGraph): Boolean {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_graph_compute` */
    fun graphCompute(graph: GGMLCGraph): GGMLStatus {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_graph_compute_async` */
    fun graphComputeAsync(graph: GGMLCGraph): GGMLStatus {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_synchronize` */
    fun synchronize() {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_reset` */
    fun reset() {
        TODO("port from ggml-backend.h")
    }

    /** `ggml_backend_sched_set_eval_callback` */
    fun setEvalCallback(callback: GGMLBackendSchedEvalCallback?) {
        this.evalCallback = callback
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

/**
 * `ggml_backend_meta_split_axis_name`
 */
fun ggmlBackendMetaSplitAxisName(axis: GGMLBackendMetaSplitAxis): String = axis.name

/**
 * Split state for a tensor in a meta-backend context.
 * Mirrors `struct ggml_backend_meta_split_state`.
 */
data class GGMLBackendMetaSplitState(
    val axis: GGMLBackendMetaSplitAxis,
    /**
     * Per-segment, per-device element counts. Outer loop = segments, inner = devices.
     * Size: nSegments * nDevices. Mirrors the C `ne[16*GGML_BACKEND_META_MAX_DEVICES]`.
     */
    val ne: LongArray = LongArray(16 * GGML_BACKEND_META_MAX_DEVICES),
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
typealias GGMLBackendMetaGetSplitState = (tensor: GGMLTensor) -> GGMLBackendMetaSplitState

/**
 * `ggml_backend_meta_device` – create a meta device from constituent devices.
 */
fun ggmlBackendMetaDevice(
    devices: List<GGMLBackendDevice>,
    getSplitState: GGMLBackendMetaGetSplitState
): GGMLBackendDevice {
    TODO("port from ggml-backend.h")
}

// ---------------------------------------------------------------------------
// Graph copy utilities
// ---------------------------------------------------------------------------

/**
 * Result of copying a graph to a different backend.
 * Mirrors `struct ggml_backend_graph_copy`.
 */
data class GGMLBackendGraphCopy(
    val buffer: GGMLBackendBuffer?,
    val ctxAllocated: GGMLContext?,
    val ctxUnallocated: GGMLContext?,
    val graph: GGMLCGraph?
)

/**
 * `ggml_backend_graph_copy` – copy a graph to a different backend.
 */
fun ggmlBackendGraphCopy(backend: GGMLBackend, graph: GGMLCGraph): GGMLBackendGraphCopy {
    TODO("port from ggml-backend.h")
}

/**
 * `ggml_backend_graph_copy_free` – free a previously copied graph.
 */
fun ggmlBackendGraphCopyFree(copy: GGMLBackendGraphCopy) {
    TODO("port from ggml-backend.h")
}

/**
 * Evaluation callback used by `ggml_backend_compare_graph_backend`.
 * Mirrors `ggml_backend_eval_callback`.
 */
typealias GGMLBackendEvalCallback = (nodeIndex: Int, t1: GGMLTensor, t2: GGMLTensor) -> Boolean

/**
 * `ggml_backend_compare_graph_backend` – compare the output of two backends.
 */
fun ggmlBackendCompareGraphBackend(
    backend1: GGMLBackend,
    backend2: GGMLBackend,
    graph: GGMLCGraph,
    callback: GGMLBackendEvalCallback,
    testNodes: List<GGMLTensor>
): Boolean {
    TODO("port from ggml-backend.h")
}

// ---------------------------------------------------------------------------
// CPU convenience helpers
// ---------------------------------------------------------------------------

/**
 * `ggml_backend_cpu_buffer_from_ptr` – wrap an existing byte array as a CPU buffer.
 */
fun ggmlBackendCpuBufferFromPtr(ptr: ByteArray, size: ULong): GGMLBackendBuffer {
    TODO("port from ggml-backend.h")
}

/**
 * `ggml_backend_cpu_buffer_type` – get the CPU buffer type singleton.
 */
fun ggmlBackendCpuBufferType(): GGMLBackendBufferType {
    return GGMLCpuBufferType()
}
