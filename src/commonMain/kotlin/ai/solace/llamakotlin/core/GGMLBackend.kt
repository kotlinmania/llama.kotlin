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
        // Default: identity; implementations may track usage
    }

    /** `ggml_backend_buffer_get_usage` */
    fun getUsage(): GGMLBackendBufferUsage {
        return GGMLBackendBufferUsage.ANY
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
    fun graphPlanCreate(graph: GGMLCGraph): Any? { return null }

    /** `ggml_backend_graph_plan_free` */
    fun graphPlanFree(plan: Any?) {}

    /** `ggml_backend_graph_plan_compute` */
    fun graphPlanCompute(plan: Any?): GGMLStatus { return GGMLStatus.FAILED }

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
            defaultBufferType = createDefaultCpuBufferType()
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

    // -- dynamic loading (no-op on native, used for dynamic backend plugins) --

    /** `ggml_backend_load` */
    fun load(path: String): GGMLBackendReg? { return null }

    /** `ggml_backend_unload` */
    fun unload(reg: GGMLBackendReg) {}

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
typealias GGMLBackendSchedEvalCallback = (t: GGMLTensor, ask: Boolean, userData: Any?) -> Boolean

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
    val backends: Array<GGMLBackend?> = arrayOfNulls(GGML_SCHED_MAX_BACKENDS)
    val bufts: Array<GGMLBackendBufferType?> = arrayOfNulls(GGML_SCHED_MAX_BACKENDS)

    // --- graph allocator (ggml_gallocr_t) ---
    // Will be typed as GGMLGAllocr when ggml-alloc.c gallocr is fully ported.
    // For now, Any? matches the opaque pointer pattern until gallocr is fully ported.
    var galloc: Any? = null

    // --- hash map of the nodes in the graph ---
    val hashSet: GGMLHashSet = GGMLHashSet(GGMLHashSet.ggml_hash_size(graphSize))
    val hvTensorBackendIds: IntArray = IntArray(hashSet.size) { -1 }
    val hvTensorCopies: Array<GGMLTensor?> = arrayOfNulls(hashSet.size * GGML_SCHED_MAX_BACKENDS * GGML_SCHED_MAX_COPIES)

    // --- per-node/leaf backend assignments ---
    private val nodesSize: Int = graphSize + graphSize * GGML_SCHED_MAX_SPLIT_INPUTS * 2
    var nodeBackendIds: IntArray = IntArray(nodesSize)
    var leafBackendIds: IntArray = IntArray(nodesSize)
    var prevNodeBackendIds: IntArray = IntArray(nodesSize)
    var prevLeafBackendIds: IntArray = IntArray(nodesSize)

    // --- copy of the graph with modified inputs ---
    val graph: GGMLCGraph = GGMLCGraph()

    // --- graph splits ---
    var splits: MutableList<GGMLBackendSchedSplit> = MutableList(16) { GGMLBackendSchedSplit() }
    var nSplits: Int = 0
    var splitsCapacity: Int = 16

    // --- pipeline parallelism support ---
    var curCopy: Int = 0
    var nextCopy: Int = 0
    val events: Array<Array<GGMLBackendEvent?>> = Array(GGML_SCHED_MAX_BACKENDS) { arrayOfNulls(GGML_SCHED_MAX_COPIES) }
    val graphInputs: Array<GGMLTensor?> = arrayOfNulls(GGML_SCHED_MAX_SPLIT_INPUTS)
    var nGraphInputs: Int = 0

    // --- context for temporary tensor allocations during split_graph ---
    var ctx: GGMLContext? = null

    // --- eval callback ---
    var callbackEval: GGMLBackendSchedEvalCallback? = null
    var callbackEvalUserData: Any? = null

    // --- context buffer (Kotlin heap-allocated, no manual memory management needed) ---
    val contextBufferSize: Int = graphSize * GGML_SCHED_MAX_SPLIT_INPUTS * 2
    var contextBuffer: ByteArray = ByteArray(contextBufferSize)

    // --- debug ---
    var debugGraphSize: Int = 0
    var debugPrevGraphSize: Int = 0

    // =====================================================================
    // C macro equivalents: hash_id, tensor_backend_id, tensor_id_copy, tensor_copy
    // =====================================================================

    /** C: `hash_id(tensor)` → `ggml_hash_find_or_insert(&sched->hash_set, tensor)` */
    fun hashId(tensor: GGMLTensor): Int = ggml_hash_find_or_insert(hashSet, tensor)

    /** C: `tensor_backend_id(tensor)` → `sched->hv_tensor_backend_ids[hash_id(tensor)]` */
    fun tensorBackendId(tensor: GGMLTensor): Int = hvTensorBackendIds[hashId(tensor)]

    /** Set the backend id for a tensor in the hash map. */
    fun setTensorBackendId(tensor: GGMLTensor, id: Int) {
        hvTensorBackendIds[hashId(tensor)] = id
    }

    /**
     * C: `tensor_id_copy(id, backend_id, copy_id)`
     * → `sched->hv_tensor_copies[(id) * n_backends * n_copies + (backend_id) * n_copies + (copy_id)]`
     */
    fun tensorIdCopy(id: Int, backendId: Int, copyId: Int): GGMLTensor? =
        hvTensorCopies[id * nBackends * nCopies + backendId * nCopies + copyId]

    fun setTensorIdCopy(id: Int, backendId: Int, copyId: Int, value: GGMLTensor?) {
        hvTensorCopies[id * nBackends * nCopies + backendId * nCopies + copyId] = value
    }

    /** C: `tensor_copy(tensor, backend_id, copy_id)` → `tensor_id_copy(hash_id(tensor), ...)` */
    fun tensorCopy(tensor: GGMLTensor, backendId: Int, copyId: Int): GGMLTensor? =
        tensorIdCopy(hashId(tensor), backendId, copyId)

    fun setTensorCopy(tensor: GGMLTensor, backendId: Int, copyId: Int, value: GGMLTensor?) {
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
            backends: List<GGMLBackend>,
            bufts: List<GGMLBackendBufferType?>?,
            graphSize: Int,
            parallel: Boolean,
            opOffload: Boolean
        ): GGMLBackendSched {
            require(backends.isNotEmpty())
            require(backends.size <= GGML_SCHED_MAX_BACKENDS)

            val nCopies = if (parallel) GGML_SCHED_MAX_COPIES else 1

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
                    ?: ggmlBackendGetDefaultBufferType(backends[b])
                require(ggmlBackendSupportsBuft(backends[b], sched.bufts[b]!!))

                if (nCopies > 1) {
                    for (c in 0 until nCopies) {
                        sched.events[b][c] = ggmlBackendEventNew(backends[b].getDevice())
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
                ggmlBackendEventFree(events[b][c])
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
    fun reserveSize(measureGraph: GGMLCGraph, sizes: ULongArray) {
        require(hashSet.size >= measureGraph.nNodes + measureGraph.nLeafs)
        reset()
        synchronize()
        ggmlBackendSchedSplitGraph(this, measureGraph)
        // ggmlGallocrReserveNSize(galloc, graph, nodeBackendIds, leafBackendIds, sizes)
    }

    /** `ggml_backend_sched_reserve` — C: lines 1847-1861. Returns true on success. */
    fun reserve(measureGraph: GGMLCGraph): Boolean {
        require(hashSet.size >= measureGraph.nNodes + measureGraph.nLeafs)
        synchronize()
        ggmlBackendSchedSplitGraph(this, measureGraph)
        // if (!ggmlGallocrReserveN(galloc, graph, nodeBackendIds, leafBackendIds)) return false
        reset()
        return true
    }

    /** `ggml_backend_sched_alloc_graph` — C: lines 1864-1881. Returns true on success. */
    fun allocGraph(graph: GGMLCGraph): Boolean {
        require(hashSet.size >= graph.nNodes + graph.nLeafs)
        require(!isAlloc)

        curCopy = nextCopy
        nextCopy = (nextCopy + 1) % nCopies

        ggmlBackendSchedSplitGraph(this, graph)

        if (!ggmlBackendSchedAllocSplits(this)) {
            return false
        }

        isAlloc = true
        return true
    }

    /** `ggml_backend_sched_graph_compute` — C: lines 1883-1887 */
    fun graphCompute(graph: GGMLCGraph): GGMLStatus {
        val err = graphComputeAsync(graph)
        synchronize()
        return err
    }

    /** `ggml_backend_sched_graph_compute_async` — C: lines 1889-1902 */
    fun graphComputeAsync(graph: GGMLCGraph): GGMLStatus {
        if (!isReset && !isAlloc) {
            reset()
        }
        if (!isAlloc) {
            if (!allocGraph(graph)) {
                return GGMLStatus.ALLOC_FAILED
            }
        }
        return ggmlBackendSchedComputeSplits(this)
    }

    /** `ggml_backend_sched_synchronize` — C: lines 1904-1915 */
    fun synchronize() {
        for (i in 0 until nBackends) {
            ggmlBackendSynchronize(backends[i]!!)
        }
        if (!isAlloc) {
            nextCopy = 0
        }
    }

    /** `ggml_backend_sched_set_eval_callback` — C: lines 1917-1921 */
    fun setEvalCallback(callback: GGMLBackendSchedEvalCallback?, userData: Any? = null) {
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
    fun getBackend(index: Int): GGMLBackend? {
        require(index in 0 until nBackends)
        return backends[index]
    }

    /** `ggml_backend_sched_get_buffer_type` — C: lines 1944-1949 */
    fun getBufferType(backend: GGMLBackend): GGMLBackendBufferType? {
        val idx = ggmlBackendSchedBackendId(this, backend)
        require(idx in 0 until nBackends)
        return bufts[idx]
    }

    /** `ggml_backend_sched_get_buffer_size` — C: lines 1952-1957 */
    fun getBufferSize(backend: GGMLBackend): ULong {
        val idx = ggmlBackendSchedBackendId(this, backend)
        require(idx in 0 until nBackends)
        // return ggmlGallocrGetBufferSize(galloc, idx) — when gallocr is ported
        return 0UL
    }

    /** `ggml_backend_sched_set_tensor_backend` — C: lines 1960-1967 */
    fun setTensorBackend(node: GGMLTensor, backend: GGMLBackend) {
        val idx = ggmlBackendSchedBackendId(this, backend)
        require(idx in 0 until nBackends)
        setTensorBackendId(node, idx)
        isReset = false
    }

    /** `ggml_backend_sched_get_tensor_backend` — C: lines 1969-1976 */
    fun getTensorBackend(node: GGMLTensor): GGMLBackend? {
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

// ggmlBackendMetaDevice moved to GGMLBackendUtils.kt

// Graph copy utilities, GGMLBackendGraphCopy, ggmlBackendGraphCopy,
// ggmlBackendGraphCopyFree, GGMLBackendEvalCallback,
// ggmlBackendCompareGraphBackend all moved to GGMLBackendUtils.kt

// =========================================================================
// Free-standing C-style API (ggml-backend.h declarations)
// =========================================================================

/** `ggml_backend_buft_name` */
fun ggmlBackendBuftName(buft: GGMLBackendBufferType): String {
    return buft.getName()
}

/** `ggml_backend_buft_alloc_buffer` — returns a dummy buffer for zero-sized requests. */
fun ggmlBackendBuftAllocBuffer(buft: GGMLBackendBufferType, size: ULong): GGMLBackendBuffer? {
    if (size == 0UL) {
        // return a dummy buffer for zero-sized allocations (mirrors C)
        return null
    }
    return buft.allocBuffer(size)
}

/** `ggml_backend_buft_get_alignment` */
fun ggmlBackendBuftGetAlignment(buft: GGMLBackendBufferType): UInt {
    return buft.getAlignment()
}

/** `ggml_backend_buft_get_max_size` — defaults to ULong.MAX_VALUE. */
fun ggmlBackendBuftGetMaxSize(buft: GGMLBackendBufferType): ULong {
    return buft.getMaxSize()
}

/** `ggml_backend_buft_get_alloc_size` — defaults to ggml_nbytes(tensor). */
fun ggmlBackendBuftGetAllocSize(buft: GGMLBackendBufferType, tensor: GGMLTensor): ULong {
    val size = buft.getAllocSize(tensor)
    require(size >= ggmlNbytes(tensor)) { "alloc size must be >= ggml_nbytes" }
    return size
}

/** `ggml_backend_buft_is_host` */
fun ggmlBackendBuftIsHost(buft: GGMLBackendBufferType): Boolean {
    return buft.isHost()
}

/** `ggml_backend_buft_get_device` */
fun ggmlBackendBuftGetDevice(buft: GGMLBackendBufferType): GGMLBackendDevice? {
    return buft.getDevice()
}

/** `ggml_backend_buffer_name` */
fun ggmlBackendBufferName(buffer: GGMLBackendBuffer): String {
    return ggmlBackendBuftName(buffer.getType())
}

/** `ggml_backend_buffer_free` */
fun ggmlBackendBufferFree(buffer: GGMLBackendBuffer?) {
    buffer?.free()
}

/** `ggml_backend_buffer_get_size` */
fun ggmlBackendBufferGetSize(buffer: GGMLBackendBuffer): ULong {
    return buffer.getSize()
}

/** `ggml_backend_buffer_get_base` */
fun ggmlBackendBufferGetBase(buffer: GGMLBackendBuffer): Any? {
    if (buffer.getSize() == 0UL) return null
    return buffer.getBase()
}

/** `ggml_backend_buffer_init_tensor` */
fun ggmlBackendBufferInitTensor(buffer: GGMLBackendBuffer, tensor: GGMLTensor): GGMLStatus {
    return buffer.initTensor(tensor)
}

/** `ggml_backend_buffer_clear` */
fun ggmlBackendBufferClear(buffer: GGMLBackendBuffer, value: UByte) {
    if (buffer.getSize() == 0UL) return
    buffer.clear(value)
}

/** `ggml_backend_buffer_get_alignment` */
fun ggmlBackendBufferGetAlignment(buffer: GGMLBackendBuffer): ULong {
    return ggmlBackendBuftGetAlignment(buffer.getType()).toULong()
}

/** `ggml_backend_buffer_get_max_size` */
fun ggmlBackendBufferGetMaxSize(buffer: GGMLBackendBuffer): ULong {
    return ggmlBackendBuftGetMaxSize(buffer.getType())
}

/** `ggml_backend_buffer_get_alloc_size` */
fun ggmlBackendBufferGetAllocSize(buffer: GGMLBackendBuffer, tensor: GGMLTensor): ULong {
    return ggmlBackendBuftGetAllocSize(buffer.getType(), tensor)
}

/** `ggml_backend_buffer_is_host` */
fun ggmlBackendBufferIsHost(buffer: GGMLBackendBuffer): Boolean {
    return ggmlBackendBuftIsHost(buffer.getType())
}

/** `ggml_backend_buffer_set_usage` */
fun ggmlBackendBufferSetUsage(buffer: GGMLBackendBuffer, usage: GGMLBackendBufferUsage) {
    buffer.setUsage(usage)
}

/** `ggml_backend_buffer_get_usage` */
fun ggmlBackendBufferGetUsage(buffer: GGMLBackendBuffer): GGMLBackendBufferUsage {
    return buffer.getUsage()
}

/** `ggml_backend_buffer_get_type` */
fun ggmlBackendBufferGetType(buffer: GGMLBackendBuffer): GGMLBackendBufferType {
    return buffer.getType()
}

/** `ggml_backend_buffer_reset` */
fun ggmlBackendBufferReset(buffer: GGMLBackendBuffer) {
    buffer.reset()
}

/** `ggml_backend_guid` */
fun ggmlBackendGuid(backend: GGMLBackend?): String? {
    return backend?.getGuid()
}

/** `ggml_backend_name` */
fun ggmlBackendName(backend: GGMLBackend?): String {
    return backend?.getName() ?: "NULL"
}

/** `ggml_backend_free` */
fun ggmlBackendFree(backend: GGMLBackend?) {
    backend?.free()
}

/** `ggml_backend_get_default_buffer_type` */
fun ggmlBackendGetDefaultBufferType(backend: GGMLBackend): GGMLBackendBufferType {
    return backend.getDefaultBufferType()
}

/** `ggml_backend_alloc_buffer` */
fun ggmlBackendAllocBuffer(backend: GGMLBackend, size: ULong): GGMLBackendBuffer? {
    return ggmlBackendBuftAllocBuffer(ggmlBackendGetDefaultBufferType(backend), size)
}

/** `ggml_backend_get_alignment` */
fun ggmlBackendGetAlignment(backend: GGMLBackend): UInt {
    return ggmlBackendBuftGetAlignment(ggmlBackendGetDefaultBufferType(backend))
}

/** `ggml_backend_get_max_size` */
fun ggmlBackendGetMaxSize(backend: GGMLBackend): ULong {
    return ggmlBackendBuftGetMaxSize(ggmlBackendGetDefaultBufferType(backend))
}

/** `ggml_backend_tensor_set_async` */
fun ggmlBackendTensorSetAsync(
    backend: GGMLBackend,
    tensor: GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong
) {
    require(offset + size <= ggmlNbytes(tensor)) { "tensor write out of bounds" }
    backend.setTensorAsync(tensor, data, offset, size)
}

/** `ggml_backend_tensor_get_async` */
fun ggmlBackendTensorGetAsync(
    backend: GGMLBackend,
    tensor: GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong
) {
    require(offset + size <= ggmlNbytes(tensor)) { "tensor read out of bounds" }
    backend.getTensorAsync(tensor, data, offset, size)
}

/**
 * `ggml_backend_tensor_set_2d_async`
 * Set 2-D strided tensor data asynchronously.
 */
fun ggmlBackendTensorSet2dAsync(
    backend: GGMLBackend,
    tensor: GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong,
    nCopies: ULong,
    strideTensor: ULong,
    strideData: ULong
) {
    if (nCopies <= 1UL) {
        for (i in 0UL until nCopies) {
            ggmlBackendTensorSetAsync(backend, tensor, data, offset + i * strideTensor, size)
        }
        return
    }
    if (size == 0UL) return
    require(offset + (nCopies - 1UL) * strideTensor + size <= ggmlNbytes(tensor)) { "tensor write out of bounds" }
    backend.setTensor2dAsync(tensor, data, offset, size, nCopies, strideTensor, strideData)
}

/**
 * `ggml_backend_tensor_get_2d_async`
 * Get 2-D strided tensor data asynchronously.
 */
fun ggmlBackendTensorGet2dAsync(
    backend: GGMLBackend,
    tensor: GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong,
    nCopies: ULong,
    strideTensor: ULong,
    strideData: ULong
) {
    if (nCopies <= 1UL) {
        for (i in 0UL until nCopies) {
            ggmlBackendTensorGetAsync(backend, tensor, data, offset + i * strideTensor, size)
        }
        return
    }
    if (size == 0UL) return
    require(offset + (nCopies - 1UL) * strideTensor + size <= ggmlNbytes(tensor)) { "tensor read out of bounds" }
    backend.getTensor2dAsync(tensor, data, offset, size, nCopies, strideTensor, strideData)
}

/**
 * `ggml_backend_tensor_set` — synchronous set from host memory.
 * (Re-implements the existing minimal with full C logic.)
 */
fun ggmlBackendTensorSet(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    if (size == 0UL) return
    require(offset + size <= ggmlNbytes(tensor)) { "tensor write out of bounds" }
    buf.setTensor(tensor, data, offset, size)
}

/**
 * `ggml_backend_tensor_get` — synchronous get into host memory.
 * (Re-implements the existing minimal with full C logic.)
 */
fun ggmlBackendTensorGet(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    if (size == 0UL) return
    require(offset + size <= ggmlNbytes(tensor)) { "tensor read out of bounds" }
    buf.getTensor(tensor, data, offset, size)
}

/**
 * `ggml_backend_tensor_set_2d` — synchronous 2-D strided set.
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
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    // fallback to per-row set if no 2-D helper on the buffer
    for (i in 0UL until nCopies) {
        ggmlBackendTensorSet(tensor, data, offset + i * strideTensor, size)
    }
}

/**
 * `ggml_backend_tensor_get_2d` — synchronous 2-D strided get.
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
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    for (i in 0UL until nCopies) {
        ggmlBackendTensorGet(tensor, data, offset + i * strideTensor, size)
    }
}

/**
 * `ggml_backend_tensor_memset` — fill a region of a tensor with a byte value.
 */
fun ggmlBackendTensorMemset(tensor: GGMLTensor, value: UByte, offset: ULong, size: ULong) {
    if (size == 0UL) return
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    require(offset + size <= ggmlNbytes(tensor)) { "tensor write out of bounds" }
    buf.clear(value) // simplified: clear the entire buffer with value
}

/** `ggml_backend_synchronize` */
fun ggmlBackendSynchronize(backend: GGMLBackend) {
    backend.synchronize()
}

/** `ggml_backend_graph_plan_create` */
fun ggmlBackendGraphPlanCreate(backend: GGMLBackend, graph: GGMLCGraph): Any? {
    return backend.graphPlanCreate(graph)
}

/** `ggml_backend_graph_plan_free` */
fun ggmlBackendGraphPlanFree(backend: GGMLBackend, plan: Any?) {
    backend.graphPlanFree(plan)
}

/** `ggml_backend_graph_plan_compute` */
fun ggmlBackendGraphPlanCompute(backend: GGMLBackend, plan: Any?): GGMLStatus {
    return backend.graphPlanCompute(plan)
}

/** `ggml_backend_graph_compute` — sync wrapper around async compute. */
fun ggmlBackendGraphCompute(backend: GGMLBackend, graph: GGMLCGraph): GGMLStatus {
    val err = ggmlBackendGraphComputeAsync(backend, graph)
    ggmlBackendSynchronize(backend)
    return err
}

/** `ggml_backend_graph_compute_async` */
fun ggmlBackendGraphComputeAsync(backend: GGMLBackend, graph: GGMLCGraph): GGMLStatus {
    return backend.graphComputeAsync(graph)
}

/** `ggml_backend_supports_op` */
fun ggmlBackendSupportsOp(backend: GGMLBackend, op: GGMLTensor): Boolean {
    return backend.supportsOp(op)
}

/** `ggml_backend_supports_buft` */
fun ggmlBackendSupportsBuft(backend: GGMLBackend, buft: GGMLBackendBufferType): Boolean {
    return backend.supportsBufferType(buft)
}

/** `ggml_backend_offload_op` */
fun ggmlBackendOffloadOp(backend: GGMLBackend, op: GGMLTensor): Boolean {
    return backend.offloadOp(op)
}

/** `ggml_backend_get_device` */
fun ggmlBackendGetDevice(backend: GGMLBackend): GGMLBackendDevice? {
    return backend.getDevice()
}

/**
 * `ggml_backend_tensor_copy` — copy tensor data between different backends.
 * Tries direct buffer copy first, then falls back through host memory.
 */
fun ggmlBackendTensorCopy(src: GGMLTensor, dst: GGMLTensor) {
    if (src === dst) return

    val srcBuf = src.buffer
    val dstBuf = dst.buffer

    if (srcBuf != null && ggmlBackendBufferIsHost(srcBuf)) {
        // source is in host memory — direct set into dst
        val nbytes = ggmlNbytes(src)
        val tmp = ByteArray(nbytes.toInt())
        srcBuf.getTensor(src, tmp, 0UL, nbytes)
        ggmlBackendTensorSet(dst, tmp, 0UL, nbytes)
    } else if (dstBuf != null && ggmlBackendBufferIsHost(dstBuf)) {
        // destination is in host memory — direct get from src
        val nbytes = ggmlNbytes(src)
        val tmp = ByteArray(nbytes.toInt())
        ggmlBackendTensorGet(src, tmp, 0UL, nbytes)
        dstBuf.setTensor(dst, tmp, 0UL, nbytes)
    } else if (!ggmlBackendBufferCopyTensor(src, dst)) {
        // slow path: round-trip through host
        val nbytes = ggmlNbytes(src)
        val tmp = ByteArray(nbytes.toInt())
        ggmlBackendTensorGet(src, tmp, 0UL, nbytes)
        ggmlBackendTensorSet(dst, tmp, 0UL, nbytes)
    }
}

/**
 * `ggml_backend_tensor_copy_async` — async copy between two backends.
 * Falls back to synchronous copy when the backend doesn't support async copies.
 */
fun ggmlBackendTensorCopyAsync(
    backendSrc: GGMLBackend,
    backendDst: GGMLBackend,
    src: GGMLTensor,
    dst: GGMLTensor
) {
    if (src === dst) return

    if (backendDst.copyTensorAsync(backendSrc, src, dst)) {
        return
    }
    // fallback: synchronize both then do a blocking copy
    ggmlBackendSynchronize(backendSrc)
    ggmlBackendSynchronize(backendDst)
    ggmlBackendTensorCopy(src, dst)
}

/** `ggml_backend_event_new` */
fun ggmlBackendEventNew(device: GGMLBackendDevice?): GGMLBackendEvent? {
    return device?.newEvent()
}

/** `ggml_backend_event_free` */
fun ggmlBackendEventFree(event: GGMLBackendEvent?) {
    event?.free()
}

/** `ggml_backend_event_record` */
fun ggmlBackendEventRecord(event: GGMLBackendEvent, backend: GGMLBackend) {
    event.record(backend)
}

/** `ggml_backend_event_synchronize` */
fun ggmlBackendEventSynchronize(event: GGMLBackendEvent) {
    event.synchronize()
}

/** `ggml_backend_event_wait` */
fun ggmlBackendEventWait(backend: GGMLBackend, event: GGMLBackendEvent) {
    event.wait(backend)
}

/** `ggml_backend_dev_name` */
fun ggmlBackendDevName(device: GGMLBackendDevice): String {
    return device.getName()
}

/** `ggml_backend_dev_description` */
fun ggmlBackendDevDescription(device: GGMLBackendDevice): String {
    return device.getDescription()
}

/** `ggml_backend_dev_memory` — returns (free, total). */
fun ggmlBackendDevMemory(device: GGMLBackendDevice): Pair<ULong, ULong> {
    return device.getMemory()
}

/** `ggml_backend_dev_type` */
fun ggmlBackendDevType(device: GGMLBackendDevice): GGMLBackendDeviceType {
    return device.getType()
}

/** `ggml_backend_dev_get_props` */
fun ggmlBackendDevGetProps(device: GGMLBackendDevice): GGMLBackendDeviceProps {
    return device.getProps()
}

/** `ggml_backend_dev_backend_reg` */
fun ggmlBackendDevBackendReg(device: GGMLBackendDevice): GGMLBackendReg? {
    return device.getBackendReg()
}

/** `ggml_backend_dev_init` */
fun ggmlBackendDevInit(device: GGMLBackendDevice, params: String? = null): GGMLBackend? {
    return device.initBackend(params)
}

/** `ggml_backend_dev_buffer_type` */
fun ggmlBackendDevBufferType(device: GGMLBackendDevice): GGMLBackendBufferType {
    return device.getBufferType()
}

/** `ggml_backend_dev_host_buffer_type` */
fun ggmlBackendDevHostBufferType(device: GGMLBackendDevice): GGMLBackendBufferType? {
    return device.getHostBufferType()
}

/** `ggml_backend_dev_buffer_from_host_ptr` */
fun ggmlBackendDevBufferFromHostPtr(
    device: GGMLBackendDevice,
    ptr: ByteArray,
    size: ULong,
    maxTensorSize: ULong
): GGMLBackendBuffer? {
    return device.bufferFromHostPtr(ptr, size, maxTensorSize)
}

/** `ggml_backend_dev_supports_op` */
fun ggmlBackendDevSupportsOp(device: GGMLBackendDevice, op: GGMLTensor): Boolean {
    return device.supportsOp(op)
}

/** `ggml_backend_dev_supports_buft` */
fun ggmlBackendDevSupportsBuft(device: GGMLBackendDevice, buft: GGMLBackendBufferType): Boolean {
    return device.supportsBufferType(buft)
}

/** `ggml_backend_dev_offload_op` */
fun ggmlBackendDevOffloadOp(device: GGMLBackendDevice, op: GGMLTensor): Boolean {
    return device.offloadOp(op)
}

/** `ggml_backend_reg_name` */
fun ggmlBackendRegName(reg: GGMLBackendReg): String {
    return reg.getName()
}

/** `ggml_backend_reg_dev_count` */
fun ggmlBackendRegDevCount(reg: GGMLBackendReg): ULong {
    return reg.getDeviceCount()
}

/** `ggml_backend_reg_dev_get` */
fun ggmlBackendRegDevGet(reg: GGMLBackendReg, index: ULong): GGMLBackendDevice? {
    return reg.getDevice(index)
}

/** `ggml_backend_reg_get_proc_address` */
fun ggmlBackendRegGetProcAddress(reg: GGMLBackendReg, name: String): Any? {
    return reg.getProcAddress(name)
}

/** `ggml_backend_register` — C: ggml-backend-reg.cpp line 279. */
fun ggmlBackendRegister(reg: GGMLBackendReg) {
    GGMLBackendRegistry.registerReg(reg)
}

/** `ggml_backend_device_register` — C: ggml-backend-reg.cpp line 283. */
fun ggmlBackendDeviceRegister(device: GGMLBackendDevice) {
    GGMLBackendRegistry.registerDevice(device)
}

/** `ggml_backend_reg_count` — C: ggml-backend-reg.cpp line 297. */
fun ggmlBackendRegCount(): ULong {
    return GGMLBackendRegistry.regCount()
}

/** `ggml_backend_reg_get` — C: ggml-backend-reg.cpp line 301. */
fun ggmlBackendRegGet(index: ULong): GGMLBackendReg? {
    return GGMLBackendRegistry.regGet(index)
}

/** `ggml_backend_reg_by_name` — C: ggml-backend-reg.cpp line 306. */
fun ggmlBackendRegByName(name: String): GGMLBackendReg? {
    return GGMLBackendRegistry.regByName(name)
}

/** `ggml_backend_dev_count` — C: ggml-backend-reg.cpp line 317. */
fun ggmlBackendDevCount(): ULong {
    return GGMLBackendRegistry.deviceCount()
}

/** `ggml_backend_dev_get` — C: ggml-backend-reg.cpp line 321. */
fun ggmlBackendDevGet(index: ULong): GGMLBackendDevice? {
    return GGMLBackendRegistry.deviceGet(index)
}

/** `ggml_backend_dev_by_name` — C: ggml-backend-reg.cpp line 326. */
fun ggmlBackendDevByName(name: String): GGMLBackendDevice? {
    return GGMLBackendRegistry.deviceByName(name)
}

/** `ggml_backend_dev_by_type` — C: ggml-backend-reg.cpp line 336. */
fun ggmlBackendDevByType(type: GGMLBackendDeviceType): GGMLBackendDevice? {
    return GGMLBackendRegistry.deviceByType(type)
}

/** `ggml_backend_init_by_name` — C: ggml-backend-reg.cpp line 347. */
fun ggmlBackendInitByName(name: String, params: String? = null): GGMLBackend? {
    return GGMLBackendRegistry.initByName(name, params)
}

/** `ggml_backend_init_by_type` — C: ggml-backend-reg.cpp line 355. */
fun ggmlBackendInitByType(type: GGMLBackendDeviceType, params: String? = null): GGMLBackend? {
    return GGMLBackendRegistry.initByType(type, params)
}

/** `ggml_backend_init_best` — C: ggml-backend-reg.cpp line 363. */
fun ggmlBackendInitBest(): GGMLBackend? {
    return GGMLBackendRegistry.initBest()
}

/** `ggml_backend_load` — C: ggml-backend-reg.cpp line 374. */
fun ggmlBackendLoad(path: String): GGMLBackendReg? {
    return GGMLBackendRegistry.load(path)
}

/** `ggml_backend_unload` — C: ggml-backend-reg.cpp line 378. */
fun ggmlBackendUnload(reg: GGMLBackendReg) {
    GGMLBackendRegistry.unload(reg)
}

/** `ggml_backend_load_all` — C: ggml-backend-reg.cpp line 543. */
fun ggmlBackendLoadAll() {
    GGMLBackendRegistry.loadAll()
}

/** `ggml_backend_load_all_from_path` — C: ggml-backend-reg.cpp line 547. */
fun ggmlBackendLoadAllFromPath(dirPath: String) {
    GGMLBackendRegistry.loadAllFromPath(dirPath)
}

/** `ggml_backend_meta_split_axis_name` — C: ggml-backend-meta.cpp line 26. */
fun ggmlBackendMetaSplitAxisName(axis: GGMLBackendMetaSplitAxis): String = axis.name

/**
 * `ggml_backend_meta_device` — create a meta device from constituent devices.
 * C: ggml-backend.h line 398. Interim implementation for future tensor-parallelism support.
 */
fun ggmlBackendMetaDevice(
    devices: List<GGMLBackendDevice>,
    getSplitState: GGMLBackendMetaGetSplitState
): GGMLBackendDevice? {
    return null
}

/**
 * `ggml_backend_graph_copy` — deep-copy a graph and its tensors to a target backend.
 * C: ggml-backend.cpp lines 2068-2148.
 */
fun ggmlBackendGraphCopy(backend: GGMLBackend, graph: GGMLCGraph): GGMLBackendGraphCopy {
    val hashSet = GGMLHashSet(GGMLHashSet.ggml_hash_size(graph.size))
    val nodeCopies: Array<GGMLTensor?> = arrayOfNulls(hashSet.size)
    val nodeInit = BooleanArray(hashSet.size)

    val ctxAllocated = GGMLContext(noAlloc = true)
    val ctxUnallocated = GGMLContext(noAlloc = true)

    // dup nodes
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        graphCopyDupTensor(hashSet, nodeCopies, ctxAllocated, ctxUnallocated, node)
    }

    // copy data and init views
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        graphCopyInitTensor(hashSet, nodeCopies, nodeInit, node)
    }

    // build graph copy
    val graphCopy = ggmlNewGraphCustom(ctxAllocated, graph.size.toULong(), false)
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        val nodeCopy = nodeCopies[ggml_hash_find(hashSet, node)]
        graphCopy.nodes[i] = nodeCopy
    }
    graphCopy.nNodes = graph.nNodes

    return GGMLBackendGraphCopy(
        buffer = null,
        ctxAllocated = ctxAllocated,
        ctxUnallocated = ctxUnallocated,
        graph = graphCopy
    )
}

/** `ggml_backend_graph_copy_free` — C: ggml-backend.cpp lines 2150-2154. */
fun ggmlBackendGraphCopyFree(copy: GGMLBackendGraphCopy) {
    ggmlBackendBufferFree(copy.buffer)
}

/**
 * `ggml_backend_compare_graph_backend` — C: ggml-backend.cpp lines 2156-2209.
 * Compute a graph on two backends and call [callback] to compare each pair of result tensors.
 */
fun ggmlBackendCompareGraphBackend(
    backend1: GGMLBackend,
    backend2: GGMLBackend,
    graph: GGMLCGraph,
    callback: GGMLBackendEvalCallback,
    testNodes: List<GGMLTensor>
): Boolean {
    val copy = ggmlBackendGraphCopy(backend2, graph)
    if (copy.buffer == null && copy.graph == null) {
        return false
    }

    val g1 = graph
    val g2 = copy.graph!!

    require(g1.nNodes == g2.nNodes)

    if (testNodes.isNotEmpty()) {
        ggmlBackendGraphCompute(backend1, g1)
        ggmlBackendGraphCompute(backend2, g2)

        for (i in 0 until g1.nNodes) {
            for (testNode in testNodes) {
                if (g1.nodes[i] === testNode) {
                    callback(i, g1.nodes[i]!!, g2.nodes[i]!!)
                }
            }
        }
    } else {
        for (i in 0 until g1.nNodes) {
            val t1 = g1.nodes[i]!!
            val t2 = g2.nodes[i]!!

            val g1v = ggml_graph_view(g1, i, i + 1)
            val g2v = ggml_graph_view(g2, i, i + 1)

            ggmlBackendGraphCompute(backend1, g1v)
            ggmlBackendGraphCompute(backend2, g2v)

            if (ggmlIsViewOp(t1.op)) {
                continue
            }

            if (!callback(i, t1, t2)) {
                break
            }
        }
    }
    ggmlBackendGraphCopyFree(copy)
    return true
}

/**
 * Split a computation graph into sub-graphs, each assigned to a single backend.
 * This is the core scheduling algorithm — 5-pass approach.
 * C: `ggml_backend_sched_split_graph` (lines 1014-1487)
 */
fun ggmlBackendSchedSplitGraph(sched: GGMLBackendSched, graph: GGMLCGraph) {
    // reset splits
    sched.nSplits = 0
    sched.nGraphInputs = 0
    sched.isReset = false

    // re-initialize context for temporary tensor allocations
    sched.ctx = GGMLContext(noAlloc = true)

    graph.uid = ggml_graph_next_uid()

    // pass 1: assign backends to ops with pre-allocated inputs
    for (i in 0 until graph.nLeafs) {
        val leaf = graph.leafs[i] ?: continue
        val leafBid = sched.tensorBackendId(leaf)
        if (leafBid == -1) {
            sched.setTensorBackendId(leaf, ggmlBackendSchedBackendIdFromCur(sched, leaf))
        }
    }

    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        val nodeBid = sched.tensorBackendId(node)
        if (nodeBid == -1) {
            sched.setTensorBackendId(node, ggmlBackendSchedBackendIdFromCur(sched, node))
        }
    }

    // pass 2: expand current backend assignments
    // expand gpu backends (non-last-prio) down and up, ignoring cpu (lowest priority backend)
    // expand gpu down
    run {
        var curBid = -1
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            if (ggmlIsViewOp(node.op)) continue
            val nodeBid = sched.tensorBackendId(node)
            if (nodeBid != -1) {
                curBid = if (nodeBid == sched.nBackends - 1) -1 else nodeBid
            } else if (curBid != -1) {
                ggmlBackendSchedSetIfSupported(sched, node, curBid)
            }
        }
    }
    // expand gpu up
    run {
        var curBid = -1
        for (i in graph.nNodes - 1 downTo 0) {
            val node = graph.nodes[i] ?: continue
            if (ggmlIsViewOp(node.op)) continue
            val nodeBid = sched.tensorBackendId(node)
            if (nodeBid != -1) {
                curBid = if (nodeBid == sched.nBackends - 1) -1 else nodeBid
            } else if (curBid != -1) {
                ggmlBackendSchedSetIfSupported(sched, node, curBid)
            }
        }
    }
    // expand rest down
    run {
        var curBid = -1
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            if (ggmlIsViewOp(node.op)) continue
            val nodeBid = sched.tensorBackendId(node)
            if (nodeBid != -1) {
                curBid = nodeBid
            } else if (curBid != -1) {
                ggmlBackendSchedSetIfSupported(sched, node, curBid)
            }
        }
    }
    // expand rest up
    run {
        var curBid = -1
        for (i in graph.nNodes - 1 downTo 0) {
            val node = graph.nodes[i] ?: continue
            if (ggmlIsViewOp(node.op)) continue
            val nodeBid = sched.tensorBackendId(node)
            if (nodeBid != -1) {
                curBid = nodeBid
            } else if (curBid != -1) {
                ggmlBackendSchedSetIfSupported(sched, node, curBid)
            }
        }
    }

    // pass 3: upgrade to higher prio backends with compatible buffer types
    // and assign remaining unassigned nodes to backend with most supported inputs
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        if (ggmlIsViewOp(node.op)) continue
        val nodeBid = sched.tensorBackendId(node)
        if (nodeBid == -1) {
            // unassigned node: find the backend with the most supported inputs
            var nSupportedBest = -1
            for (b in 0 until sched.nBackends) {
                if (ggmlBackendSupportsOp(sched.backends[b]!!, node)) {
                    var nSupported = 0
                    for (j in 0 until GGML_MAX_SRC) {
                        val src = node.src[j] ?: continue
                        val srcBid = sched.tensorBackendId(src)
                        val srcViewBid = if (src.viewSrc != null) sched.tensorBackendId(src.viewSrc!!) else -1
                        if ((srcBid != -1 || srcViewBid != -1) && ggmlBackendSchedBufferSupported(sched, src, b)) {
                            nSupported++
                        }
                    }
                    if (nSupported > nSupportedBest) {
                        nSupportedBest = nSupported
                        sched.setTensorBackendId(node, b)
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (b in 0 until nodeBid) {
                if (sched.bufts[b] === sched.bufts[nodeBid] &&
                    ggmlBackendSupportsOp(sched.backends[b]!!, node)
                ) {
                    var supported = true
                    for (j in 0 until GGML_MAX_SRC) {
                        val src = node.src[j] ?: continue
                        if (!ggmlBackendSchedBufferSupported(sched, src, b)) {
                            supported = false
                            break
                        }
                    }
                    if (supported) {
                        sched.setTensorBackendId(node, b)
                        break
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        var curBid = sched.tensorBackendId(node)
        if (node.viewSrc != null && curBid == -1) {
            curBid = sched.tensorBackendId(node.viewSrc!!)
            sched.setTensorBackendId(node, curBid)
        }
        for (j in 0 until GGML_MAX_SRC) {
            val src = node.src[j] ?: continue
            val srcBid = sched.tensorBackendId(src)
            if (srcBid == -1) {
                if (src.viewSrc != null) {
                    sched.setTensorBackendId(src, sched.tensorBackendId(src.viewSrc!!))
                } else {
                    sched.setTensorBackendId(src, curBid)
                }
            }
        }
        // if node is still unassigned, assign to first backend that supports it
        if (sched.tensorBackendId(node) == -1) {
            for (b in 0 until sched.nBackends) {
                if (ggmlBackendSupportsOp(sched.backends[b]!!, node)) {
                    sched.setTensorBackendId(node, b)
                    break
                }
            }
        }
        require(sched.tensorBackendId(node) != -1) { "node ${node.name} has no backend assigned" }
    }

    // pass 5: split graph, find tensors that need to be copied
    run pass5@ {
        var iSplit = 0
        var split = sched.splits[0]
        // find the backend of the first split, skipping view ops
        var i = 0
        while (i < graph.nNodes) {
            val node = graph.nodes[i]
            if (node == null) { i++; continue }
            if (!ggmlIsViewOp(node.op)) {
                split.backendId = sched.tensorBackendId(node)
                break
            }
            i++
        }
        split.iStart = 0
        split.nInputs = 0
        var curBid = split.backendId

        while (i < graph.nNodes) {
            val node = graph.nodes[i]
            if (node == null) { i++; continue }
            if (ggmlIsViewOp(node.op)) { i++; continue }

            val nodeBid = sched.tensorBackendId(node)
            require(nodeBid != -1) { "all nodes should be assigned by now" }

            // check if we should start a new split
            var needNewSplit = false
            if (nodeBid == curBid && split.nInputs > 0) {
                for (j in 0 until GGML_MAX_SRC) {
                    val src = node.src[j] ?: continue
                    if (src.buffer != null && src.buffer!!.getUsage() == GGMLBackendBufferUsage.WEIGHTS) {
                        val srcBid = sched.tensorBackendId(src)
                        if (srcBid != curBid && !ggmlBackendSchedBufferSupported(sched, src, curBid)) {
                            needNewSplit = true
                            break
                        }
                    }
                    if (split.nInputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        val srcId = sched.hashId(src)
                        val srcBid2 = sched.hvTensorBackendIds[srcId]
                        val supported = ggmlBackendSchedBufferSupported(sched, src, curBid)
                        if (srcBid2 != curBid && sched.tensorIdCopy(srcId, curBid, 0) == null && !supported) {
                            needNewSplit = true
                            break
                        }
                    }
                }
            }

            if (nodeBid != curBid || needNewSplit) {
                split.iEnd = i
                iSplit++
                if (iSplit >= sched.splitsCapacity) {
                    sched.splitsCapacity *= 2
                    while (sched.splits.size < sched.splitsCapacity) {
                        sched.splits.add(GGMLBackendSchedSplit())
                    }
                }
                split = sched.splits[iSplit]
                split.backendId = nodeBid
                split.iStart = i
                split.nInputs = 0
                curBid = nodeBid
            }

            // find inputs that are not on the same backend
            for (j in 0 until GGML_MAX_SRC) {
                val src = node.src[j] ?: continue

                val srcId = sched.hashId(src)
                val srcBid = sched.hvTensorBackendIds[srcId]
                require(srcBid != -1) { "all inputs should be assigned by now" }

                if (src.flags and GGML_TENSOR_FLAG_INPUT != 0 && sched.nCopies > 1) {
                    if (sched.tensorIdCopy(srcId, srcBid, 0) == null) {
                        val backend = sched.backends[srcBid]!!
                        for (c in 0 until sched.nCopies) {
                            val tensorCopy: GGMLTensor
                            if (c == sched.curCopy) {
                                tensorCopy = src
                            } else {
                                tensorCopy = ggmlDupTensorLayout(sched.ctx!!, src)
                                ggmlFormatName(tensorCopy, "%s#%s#%d", ggmlBackendName(backend), src.name ?: "", c)
                            }
                            ggmlSetInput(tensorCopy)
                            ggmlSetOutput(tensorCopy)
                            sched.setTensorIdCopy(srcId, srcBid, c, tensorCopy)
                        }
                        val ngi = sched.nGraphInputs++
                        require(ngi < GGML_SCHED_MAX_SPLIT_INPUTS)
                        sched.graphInputs[ngi] = src
                    }
                }

                if (srcBid != curBid && !ggmlBackendSchedBufferSupported(sched, src, curBid)) {
                    if (sched.tensorIdCopy(srcId, curBid, 0) == null) {
                        val backend = sched.backends[curBid]!!
                        for (c in 0 until sched.nCopies) {
                            val tensorCopy = ggmlDupTensorLayout(sched.ctx!!, src)
                            ggmlFormatName(tensorCopy, "%s#%s#%d", ggmlBackendName(backend), src.name ?: "", c)
                            if (sched.nCopies > 1) {
                                ggmlSetInput(tensorCopy)
                                ggmlSetOutput(tensorCopy)
                            }
                            sched.setTensorIdCopy(srcId, curBid, c, tensorCopy)
                        }
                        val ni = split.nInputs++
                        require(ni < GGML_SCHED_MAX_SPLIT_INPUTS)
                        split.inputs[ni] = src
                    }
                    node.src[j] = sched.tensorIdCopy(srcId, curBid, sched.curCopy)
                }
            }
            i++
        }
        split.iEnd = graph.nNodes
        sched.nSplits = iSplit + 1
    }

    if (sched.debug > 0) {
        ggmlBackendSchedPrintAssignments(sched, graph)
    }

    // swap node_backend_ids and leaf_backend_ids with prevs
    run {
        val tmp = sched.nodeBackendIds
        sched.nodeBackendIds = sched.prevNodeBackendIds
        sched.prevNodeBackendIds = tmp

        val tmp2 = sched.leafBackendIds
        sched.leafBackendIds = sched.prevLeafBackendIds
        sched.prevLeafBackendIds = tmp2
    }

    val graphSize = maxOf(graph.nNodes, graph.nLeafs) +
        sched.nSplits * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sched.nCopies

    sched.debugPrevGraphSize = sched.debugGraphSize
    sched.debugGraphSize = graphSize

    if (sched.graph.size < graphSize) {
        sched.graph.size = graphSize
        sched.graph.nodes = Array(graphSize) { null }
        sched.graph.leafs = Array(graphSize) { null }
    }
    sched.graph.nNodes = 0
    sched.graph.nLeafs = 0

    val graphCopy = sched.graph

    for (si in 0 until sched.nSplits) {
        val sp = sched.splits[si]
        sp.graph = ggml_graph_view(graph, sp.iStart, sp.iEnd)

        // add inputs to graph copy so they are allocated by ggml-alloc at the start of the split
        for (j in 0 until sp.nInputs) {
            require(graphCopy.size > graphCopy.nNodes + 1)

            val input = sp.inputs[j]!!
            val inputId = sched.hashId(input)
            val inputCpy = sched.tensorIdCopy(inputId, sp.backendId, sched.curCopy)

            // add a dependency to the input source so that it is not freed before the copy is done
            val inputDep = ggmlViewTensor(sched.ctx!!, input)
            inputDep.src[0] = input
            sched.nodeBackendIds[graphCopy.nNodes] = sched.hvTensorBackendIds[inputId]
            graphCopy.nodes[graphCopy.nNodes++] = inputDep

            // add the input copy
            sched.nodeBackendIds[graphCopy.nNodes] = sp.backendId
            graphCopy.nodes[graphCopy.nNodes++] = inputCpy
        }

        for (j in sp.iStart until sp.iEnd) {
            require(graphCopy.size > graphCopy.nNodes)
            sched.nodeBackendIds[graphCopy.nNodes] = sched.tensorBackendId(graph.nodes[j]!!)
            graphCopy.nodes[graphCopy.nNodes++] = graph.nodes[j]
        }
    }

    if (sched.nCopies > 1) {
        // add input copies as leafs so that they are allocated first
        for (gi in 0 until sched.nGraphInputs) {
            val input = sched.graphInputs[gi]!!
            val id = sched.hashId(input)
            val bid = sched.tensorBackendId(input)
            for (c in 0 until sched.nCopies) {
                val inputCpy = sched.tensorIdCopy(id, bid, c)
                sched.leafBackendIds[graphCopy.nLeafs] = bid
                require(graphCopy.size > graphCopy.nLeafs)
                graphCopy.leafs[graphCopy.nLeafs++] = inputCpy
            }
        }

        for (si in 0 until sched.nSplits) {
            val sp = sched.splits[si]
            val bid = sp.backendId
            for (j in 0 until sp.nInputs) {
                val input = sp.inputs[j]!!
                val id = sched.hashId(input)
                for (c in 0 until sched.nCopies) {
                    val inputCpy = sched.tensorIdCopy(id, bid, c)
                    sched.leafBackendIds[graphCopy.nLeafs] = bid
                    require(graphCopy.size > graphCopy.nLeafs)
                    graphCopy.leafs[graphCopy.nLeafs++] = inputCpy
                }
            }
        }
    }

    // add leafs from the original graph
    for (i in 0 until graph.nLeafs) {
        val leaf = graph.leafs[i] ?: continue
        sched.leafBackendIds[graphCopy.nLeafs] = sched.tensorBackendId(leaf)
        require(graphCopy.size > graphCopy.nLeafs)
        graphCopy.leafs[graphCopy.nLeafs++] = leaf
    }

    // set ids for all splits
    for (si in 0 until sched.nSplits) {
        sched.splits[si].graph.uid = ggml_graph_next_uid()
    }
}

/**
 * `ggml_backend_view_init` — initialise a tensor view.
 * C: ggml-backend.cpp lines 1980-1989.
 */
fun ggmlBackendViewInit(tensor: GGMLTensor): GGMLStatus {
    require(tensor.buffer == null) { "tensor already has a buffer" }
    requireNotNull(tensor.viewSrc) { "tensor is not a view" }
    requireNotNull(tensor.viewSrc!!.buffer) { "view_src buffer is null" }
    requireNotNull(tensor.viewSrc!!.data) { "view_src data is null" }

    tensor.buffer = tensor.viewSrc!!.buffer
    // tensor.data = viewSrc.data + viewOffs — handled via viewOffs accessor
    return ggmlBackendBufferInitTensor(tensor.buffer!!, tensor)
}

/**
 * `ggml_backend_tensor_alloc` — place a tensor at a specific address inside a buffer.
 * C: ggml-backend.cpp lines 1992-2005.
 */
fun ggmlBackendTensorAlloc(buffer: GGMLBackendBuffer, tensor: GGMLTensor, addr: Any?): GGMLStatus {
    require(tensor.buffer == null) { "tensor already allocated" }
    require(tensor.viewSrc == null) { "tensor is a view, use view_init" }

    tensor.buffer = buffer
    tensor.data = addr
    return ggmlBackendBufferInitTensor(buffer, tensor)
}

/** `ggml_backend_cpu_buffer_from_ptr` — C line 2368. */
fun ggmlBackendCpuBufferFromPtr(ptr: ByteArray, size: ULong): GGMLBackendBuffer {
    return GGMLCpuBufferFromPtr(ptr, size)
}

/** `ggml_backend_cpu_buffer_type` — C line 2328. Returns the CPU buffer type. */
fun ggmlBackendCpuBufferType(): GGMLBackendBufferType = createDefaultCpuBufferType()

/**
 * `ggml_backend_sched_new` — C: ggml-backend.cpp lines 1727-1793.
 */
fun ggmlBackendSchedNew(
    backends: List<GGMLBackend>,
    bufts: List<GGMLBackendBufferType>?,
    nBackends: Int,
    graphSize: Int,
    parallel: Boolean,
    opOffload: Boolean
): GGMLBackendSched {
    return GGMLBackendSched.new(
        backends = backends.take(nBackends),
        bufts = bufts,
        graphSize = graphSize,
        parallel = parallel,
        opOffload = opOffload
    )
}

/** `ggml_backend_sched_free` — C: ggml-backend.cpp lines 1796-1819. */
fun ggmlBackendSchedFree(sched: GGMLBackendSched?) {
    sched?.free()
}

/** `ggml_backend_sched_reset` — C: ggml-backend.cpp lines 1821-1831. */
fun ggmlBackendSchedReset(sched: GGMLBackendSched) {
    sched.reset()
}

/** `ggml_backend_sched_reserve_size` — C: ggml-backend.cpp lines 1833-1845. */
fun ggmlBackendSchedReserveSize(sched: GGMLBackendSched, measureGraph: GGMLCGraph, sizes: ULongArray) {
    sched.reserveSize(measureGraph, sizes)
}

/** `ggml_backend_sched_reserve` — C: ggml-backend.cpp lines 1847-1862. */
fun ggmlBackendSchedReserve(sched: GGMLBackendSched, measureGraph: GGMLCGraph): Boolean {
    return sched.reserve(measureGraph)
}

/** `ggml_backend_sched_alloc_graph` — C: ggml-backend.cpp lines 1864-1881. */
fun ggmlBackendSchedAllocGraph(sched: GGMLBackendSched, graph: GGMLCGraph): Boolean {
    return sched.allocGraph(graph)
}

/** `ggml_backend_sched_graph_compute` — C: ggml-backend.cpp lines 1883-1887. */
fun ggmlBackendSchedGraphCompute(sched: GGMLBackendSched, graph: GGMLCGraph): GGMLStatus {
    return sched.graphCompute(graph)
}

/** `ggml_backend_sched_graph_compute_async` — C: ggml-backend.cpp lines 1889-1902. */
fun ggmlBackendSchedGraphComputeAsync(sched: GGMLBackendSched, graph: GGMLCGraph): GGMLStatus {
    return sched.graphComputeAsync(graph)
}

/** `ggml_backend_sched_synchronize` — C: ggml-backend.cpp lines 1904-1915. */
fun ggmlBackendSchedSynchronize(sched: GGMLBackendSched) {
    sched.synchronize()
}

/** `ggml_backend_sched_set_eval_callback` — C: ggml-backend.cpp lines 1917-1921. */
fun ggmlBackendSchedSetEvalCallback(sched: GGMLBackendSched, callback: GGMLBackendSchedEvalCallback?, userData: Any? = null) {
    sched.setEvalCallback(callback, userData)
}

/** `ggml_backend_sched_get_n_splits` — C: ggml-backend.cpp line 1923. */
fun ggmlBackendSchedGetNSplits(sched: GGMLBackendSched): Int {
    return sched.getNumSplits()
}

/** `ggml_backend_sched_get_n_copies` — C: ggml-backend.cpp line 1928. */
fun ggmlBackendSchedGetNCopies(sched: GGMLBackendSched): Int {
    return sched.getNumCopies()
}

/** `ggml_backend_sched_get_n_backends` — C: ggml-backend.cpp line 1933. */
fun ggmlBackendSchedGetNBackends(sched: GGMLBackendSched): Int {
    return sched.getNumBackends()
}

/** `ggml_backend_sched_get_backend` — C: ggml-backend.cpp line 1938. */
fun ggmlBackendSchedGetBackend(sched: GGMLBackendSched, i: Int): GGMLBackend? {
    return sched.getBackend(i)
}

/** `ggml_backend_sched_get_buffer_type` — C: ggml-backend.cpp line 1944. */
fun ggmlBackendSchedGetBufferType(sched: GGMLBackendSched, backend: GGMLBackend): GGMLBackendBufferType? {
    return sched.getBufferType(backend)
}

/** `ggml_backend_sched_get_buffer_size` — C: ggml-backend.cpp line 1952. */
fun ggmlBackendSchedGetBufferSize(sched: GGMLBackendSched, backend: GGMLBackend): ULong {
    return sched.getBufferSize(backend)
}

/** `ggml_backend_sched_set_tensor_backend` — C: ggml-backend.cpp line 1960. */
fun ggmlBackendSchedSetTensorBackend(sched: GGMLBackendSched, node: GGMLTensor, backend: GGMLBackend) {
    sched.setTensorBackend(node, backend)
}

/** `ggml_backend_sched_get_tensor_backend` — C: ggml-backend.cpp line 1969. */
fun ggmlBackendSchedGetTensorBackend(sched: GGMLBackendSched, node: GGMLTensor): GGMLBackend? {
    return sched.getTensorBackend(node)
}

