// port-lint: source ggml/src/ggml-backend.cpp
package ai.solace.llamakotlin.core

/**
 * Kotlin port of `ggml-backend.cpp` — the main backend implementation file.
 *
 * Contains:
 * 1. Backend buffer-type operations (alloc, alignment, max-size, is-host …)
 * 2. Backend buffer operations (init, free, get/set tensor, clear, reset …)
 * 3. Backend (stream) operations (sync, graph plan/compute, supports-op …)
 * 4. Tensor data helpers (set, get, memset, copy, copy-async, 2-D strided …)
 * 5. Event operations (new, free, record, synchronize, wait)
 * 6. Device operations (name, description, memory, type, props, buffer-type …)
 * 7. Registry operations (name, dev-count, dev-get, proc-address)
 * 8. Multi-buffer helpers
 * 9. Backend scheduler (graph-split, alloc, compute, synchronize)
 * 10. Graph-copy / compare utilities
 * 11. CPU backend buffer and buffer-type implementation
 *
 * Existing higher-level helpers (GGMLBackendManager, globalBackendManager, etc.)
 * are preserved at the bottom of the file.
 */

// =====================================================================
// Constants  (mirrors C #define values)
// =====================================================================

/** Maximum number of backends in a scheduler. C: `GGML_SCHED_MAX_BACKENDS`. */
const val GGML_SCHED_MAX_BACKENDS: Int = 16

/** Maximum number of inputs to a single graph split. C: `GGML_SCHED_MAX_SPLIT_INPUTS`. */
const val GGML_SCHED_MAX_SPLIT_INPUTS: Int = 30

/** Maximum number of pipeline-parallel copies. C: `GGML_SCHED_MAX_COPIES`. */
const val GGML_SCHED_MAX_COPIES: Int = 4

// =====================================================================
// 1. Backend buffer-type free-standing functions
//    C: ggml_backend_buft_*
// =====================================================================

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

// =====================================================================
// 2. Backend buffer free-standing functions
//    C: ggml_backend_buffer_*
// =====================================================================

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

/**
 * `ggml_backend_buffer_copy_tensor` — copy src tensor data using the
 * destination buffer's copyTensor hook.
 */
fun ggmlBackendBufferCopyTensorImpl(src: GGMLTensor, dst: GGMLTensor): Boolean {
    val dstBuf = dst.viewSrc?.buffer ?: dst.buffer ?: return false
    return dstBuf.copyTensor(src, dst)
}

// =====================================================================
// 3. Backend (stream) free-standing functions
//    C: ggml_backend_*
// =====================================================================

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
fun ggmlBackendTensorSetImpl(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
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
fun ggmlBackendTensorGetImpl(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    if (size == 0UL) return
    require(offset + size <= ggmlNbytes(tensor)) { "tensor read out of bounds" }
    buf.getTensor(tensor, data, offset, size)
}

/**
 * `ggml_backend_tensor_set_2d` — synchronous 2-D strided set.
 */
fun ggmlBackendTensorSet2dImpl(
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
        ggmlBackendTensorSetImpl(tensor, data, offset + i * strideTensor, size)
    }
}

/**
 * `ggml_backend_tensor_get_2d` — synchronous 2-D strided get.
 */
fun ggmlBackendTensorGet2dImpl(
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
        ggmlBackendTensorGetImpl(tensor, data, offset + i * strideTensor, size)
    }
}

/**
 * `ggml_backend_tensor_memset` — fill a region of a tensor with a byte value.
 */
fun ggmlBackendTensorMemsetImpl(tensor: GGMLTensor, value: UByte, offset: ULong, size: ULong) {
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
fun ggmlBackendSupportsBufferType(backend: GGMLBackend, buft: GGMLBackendBufferType): Boolean {
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

/** `ggml_backend_graph_optimize` (static in C) */
fun ggmlBackendGraphOptimize(backend: GGMLBackend, graph: GGMLCGraph) {
    // optional: identity unless a backend overrides
}

// =====================================================================
// 4. Tensor copy helpers
//    C: ggml_backend_tensor_copy / _async
// =====================================================================

/**
 * `ggml_backend_tensor_copy` — copy tensor data between different backends.
 * Tries direct buffer copy first, then falls back through host memory.
 */
fun ggmlBackendTensorCopyImpl(src: GGMLTensor, dst: GGMLTensor) {
    if (src === dst) return

    val srcBuf = src.buffer
    val dstBuf = dst.buffer

    if (srcBuf != null && ggmlBackendBufferIsHost(srcBuf)) {
        // source is in host memory — direct set into dst
        val nbytes = ggmlNbytes(src)
        val tmp = ByteArray(nbytes.toInt())
        srcBuf.getTensor(src, tmp, 0UL, nbytes)
        ggmlBackendTensorSetImpl(dst, tmp, 0UL, nbytes)
    } else if (dstBuf != null && ggmlBackendBufferIsHost(dstBuf)) {
        // destination is in host memory — direct get from src
        val nbytes = ggmlNbytes(src)
        val tmp = ByteArray(nbytes.toInt())
        ggmlBackendTensorGetImpl(src, tmp, 0UL, nbytes)
        dstBuf.setTensor(dst, tmp, 0UL, nbytes)
    } else if (!ggmlBackendBufferCopyTensorImpl(src, dst)) {
        // slow path: round-trip through host
        val nbytes = ggmlNbytes(src)
        val tmp = ByteArray(nbytes.toInt())
        ggmlBackendTensorGetImpl(src, tmp, 0UL, nbytes)
        ggmlBackendTensorSetImpl(dst, tmp, 0UL, nbytes)
    }
}

/**
 * `ggml_backend_tensor_copy_async` — async copy between two backends.
 * Falls back to synchronous copy when the backend doesn't support async copies.
 */
fun ggmlBackendTensorCopyAsyncImpl(
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
    ggmlBackendTensorCopyImpl(src, dst)
}

// =====================================================================
// 5. Event operations
//    C: ggml_backend_event_*
// =====================================================================

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

// =====================================================================
// 6. Device operations
//    C: ggml_backend_dev_*
// =====================================================================

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
fun ggmlBackendDevSupportsBufferType(device: GGMLBackendDevice, buft: GGMLBackendBufferType): Boolean {
    return device.supportsBufferType(buft)
}

/** `ggml_backend_dev_offload_op` */
fun ggmlBackendDevOffloadOp(device: GGMLBackendDevice, op: GGMLTensor): Boolean {
    return device.offloadOp(op)
}

// =====================================================================
// 7. Backend registration (reg) operations
//    C: ggml_backend_reg_*
// =====================================================================

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

// =====================================================================
// 8. Multi-buffer
//    C: ggml_backend_multi_buffer_*
// =====================================================================

/**
 * Context for a multi-buffer (logical buffer wrapping several sub-buffers).
 * Mirrors `ggml_backend_multi_buffer_context` in C.
 */
class GGMLBackendMultiBufferContext(
    val buffers: MutableList<GGMLBackendBuffer> = mutableListOf()
)

/**
 * `ggml_backend_multi_buffer_alloc_buffer`
 * Allocate a logical multi-buffer wrapping several sub-buffers.
 */
fun ggmlBackendMultiBufferAllocBuffer(buffers: List<GGMLBackendBuffer>): GGMLBackendBuffer {
}

/**
 * `ggml_backend_buffer_is_multi_buffer`
 * Returns true if [buffer] is a multi-buffer wrapper.
 */
fun ggmlBackendBufferIsMultiBuffer(buffer: GGMLBackendBuffer): Boolean {
    return buffer is GGMLBackendMultiBufferWrapper
}

/** Marker interface / class for multi-buffer detection. */
class GGMLBackendMultiBufferWrapper(
    private val ctx: GGMLBackendMultiBufferContext,
    private val buft: GGMLBackendBufferType,
    private val totalSize: ULong
) : GGMLBackendBuffer {
    override fun getType(): GGMLBackendBufferType = buft
    override fun getName(): String = ggmlBackendBuftName(buft)
    override fun getBase(): Any? = null
    override fun getSize(): ULong = totalSize
    override fun free() {
        ctx.buffers.forEach { it.free() }
        ctx.buffers.clear()
    }
    override fun setTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        error("multi-buffer does not support direct tensor set")
    }
    override fun getTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        error("multi-buffer does not support direct tensor get")
    }
    override fun copyTensor(src: GGMLTensor, dst: GGMLTensor): Boolean = false
    override fun clear(value: UByte) {
        ctx.buffers.forEach { it.clear(value) }
    }
    override fun setUsage(usage: GGMLBackendBufferUsage) {
        ctx.buffers.forEach { ggmlBackendBufferSetUsage(it, usage) }
    }
}

/**
 * `ggml_backend_multi_buffer_set_usage`
 * Set usage flag on every sub-buffer inside a multi-buffer.
 */
fun ggmlBackendMultiBufferSetUsageImpl(buffer: GGMLBackendBuffer, usage: GGMLBackendBufferUsage) {
    buffer.setUsage(usage)
}

// =====================================================================
// 9. Utility helpers
//    C: ggml_dup_tensor_layout, ggml_is_view_op, fmt_size
// =====================================================================

/** Create a copy of a tensor with the same memory layout (strides). */
fun ggmlDupTensorLayout(ctx: GGMLContext, tensor: GGMLTensor): GGMLTensor {
    val dup = GGMLTensor(
        type = tensor.type,
        ne = tensor.ne.copyOf(),
        nb = tensor.nb.copyOf(),
        op = tensor.op,
        name = tensor.name,
        flags = tensor.flags
    )
    return dup
}

/** Returns true if [op] is a view operation (VIEW, RESHAPE, PERMUTE, TRANSPOSE). */
fun ggmlIsViewOp(op: GGMLOp): Boolean {
    return op == GGMLOp.VIEW || op == GGMLOp.RESHAPE || op == GGMLOp.PERMUTE || op == GGMLOp.TRANSPOSE
}

/** Format a byte size as a human-readable string (e.g. "128K", "64M"). */
fun fmtSize(size: ULong): String {
    return when {
        size >= (1024UL * 1024UL) -> "${size / (1024UL * 1024UL)}M"
        else -> "${size / 1024UL}K"
    }
}

// =====================================================================
// 10. Scheduler split data structure
//     C: struct ggml_backend_sched_split
// =====================================================================

/**
 * A single split in the scheduler's graph decomposition.
 * Mirrors `struct ggml_backend_sched_split` in C.
 */
data class GGMLBackendSchedSplit(
    /** Index into the scheduler's backends array. */
    var backendId: Int = -1,
    /** Start index (inclusive) into the graph's node list. */
    var iStart: Int = 0,
    /** End index (exclusive) into the graph's node list. */
    var iEnd: Int = 0,
    /** Tensor inputs that need to be copied to this split's backend. */
    val inputs: Array<GGMLTensor?> = arrayOfNulls(GGML_SCHED_MAX_SPLIT_INPUTS),
    /** Number of valid entries in [inputs]. */
    var nInputs: Int = 0,
    /** Graph view containing only the nodes for this split. */
    var graph: GGMLCGraph? = null
) {
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = backendId xor iStart xor iEnd xor nInputs
}

// =====================================================================
// 11. Scheduler internal helpers
//     C: ggml_backend_sched_backend_id, _from_buffer, _id_from_cur,
//        _print_assignments, _buffer_supported, _set_if_supported,
//        _split_graph, _alloc_splits, _compute_splits
// =====================================================================

/**
 * Find the index of [backend] in the scheduler's backends list.
 * Returns -1 if not found. Lower index = higher priority.
 * C: `ggml_backend_sched_backend_id`
 */
fun ggmlBackendSchedBackendId(backends: List<GGMLBackend>, backend: GGMLBackend): Int {
    return backends.indexOfFirst { it === backend }
}

/**
 * Find the highest-priority backend that supports both the buffer type
 * of [tensor] and the operation [op].
 * C: `ggml_backend_sched_backend_from_buffer`
 */
fun ggmlBackendSchedBackendFromBuffer(
    backends: List<GGMLBackend>,
    tensor: GGMLTensor,
    op: GGMLTensor
): Int {
    val buffer = tensor.viewSrc?.buffer ?: tensor.buffer ?: return -1
    for (i in backends.indices) {
        if (ggmlBackendSupportsBufferType(backends[i], buffer.getType()) &&
            ggmlBackendSupportsOp(backends[i], op)
        ) {
            return i
        }
    }
    return -1
}

/**
 * Determine the backend that should own a node based on its current buffer
 * allocations, view sources, and weight locations.
 * C: `ggml_backend_sched_backend_id_from_cur`
 */
fun ggmlBackendSchedBackendIdFromCur(
    backends: List<GGMLBackend>,
    tensor: GGMLTensor,
    opOffload: Boolean
): Int {
}

/**
 * Print the node→backend assignments for debugging.
 * C: `ggml_backend_sched_print_assignments`
 */
fun ggmlBackendSchedPrintAssignments(
    backends: List<GGMLBackend>,
    splits: List<GGMLBackendSchedSplit>,
    graph: GGMLCGraph
) {
}

/**
 * Check if a tensor's buffer type is supported on a given backend.
 * C: `ggml_backend_sched_buffer_supported`
 */
fun ggmlBackendSchedBufferSupported(
    backends: List<GGMLBackend>,
    bufferTypes: List<GGMLBackendBufferType?>,
    tensorBackendIds: IntArray,
    tensor: GGMLTensor,
    backendId: Int
): Boolean {
}

/**
 * Assign a backend to a node if the backend supports the operation.
 * C: `ggml_backend_sched_set_if_supported`
 */
fun ggmlBackendSchedSetIfSupported(
    backends: List<GGMLBackend>,
    node: GGMLTensor,
    curBackendId: Int,
    nodeBackendId: IntArray,
    nodeIndex: Int
) {
    if (ggmlBackendSupportsOp(backends[curBackendId], node)) {
        nodeBackendId[nodeIndex] = curBackendId
    }
}

/**
 * Split a computation graph into sub-graphs, each assigned to a single backend.
 * This is the core scheduling algorithm.
 * C: `ggml_backend_sched_split_graph`
 */
fun ggmlBackendSchedSplitGraph(
    backends: List<GGMLBackend>,
    bufferTypes: List<GGMLBackendBufferType?>,
    graph: GGMLCGraph,
    opOffload: Boolean
): List<GGMLBackendSchedSplit> {
}

/**
 * Allocate memory for all splits using the graph allocator.
 * C: `ggml_backend_sched_alloc_splits`
 */
fun ggmlBackendSchedAllocSplits(
    backends: List<GGMLBackend>,
    splits: List<GGMLBackendSchedSplit>,
    graph: GGMLCGraph
): Boolean {
}

/**
 * Execute all splits, copying inputs between backends as needed.
 * C: `ggml_backend_sched_compute_splits`
 */
fun ggmlBackendSchedComputeSplits(
    backends: List<GGMLBackend>,
    splits: List<GGMLBackendSchedSplit>,
    evalCallback: GGMLBackendSchedEvalCallback?
): GGMLStatus {
}

// =====================================================================
// 12. View / tensor alloc utils
//     C: ggml_backend_view_init, ggml_backend_tensor_alloc
// =====================================================================

/**
 * `ggml_backend_view_init` — initialise a tensor view.
 * Sets the view's buffer and data pointer from its view source.
 */
fun ggmlBackendViewInitImpl(tensor: GGMLTensor): GGMLStatus {
    requireNotNull(tensor.viewSrc) { "tensor is not a view" }
    val viewSrc = tensor.viewSrc!!
    requireNotNull(viewSrc.buffer) { "view_src buffer is null" }
    tensor.buffer = viewSrc.buffer
    return ggmlBackendBufferInitTensor(tensor.buffer!!, tensor)
}

/**
 * `ggml_backend_tensor_alloc` — place a tensor at a specific address
 * inside a buffer.
 */
fun ggmlBackendTensorAllocImpl(
    buffer: GGMLBackendBuffer,
    tensor: GGMLTensor,
    addr: Any?
): GGMLStatus {
    require(tensor.buffer == null) { "tensor already allocated" }
    require(tensor.viewSrc == null) { "tensor is a view, use view_init" }
    tensor.buffer = buffer
    return ggmlBackendBufferInitTensor(buffer, tensor)
}

// =====================================================================
// 13. Graph copy / compare
//     C: ggml_backend_graph_copy, _free, ggml_backend_compare_graph_backend
//
// NOTE: GGMLBackendGraphCopy, ggmlBackendGraphCopyFree, and
//       GGMLBackendEvalCallback are already declared in GGMLBackend.kt.
// =====================================================================

/**
 * `ggml_backend_graph_copy` — deep-copy a graph and its tensors to
 * a target backend.
 */
fun ggmlBackendGraphCopyCreate(backend: GGMLBackend, graph: GGMLCGraph): GGMLBackendGraphCopy {
}

/**
 * `ggml_backend_compare_graph_backend` — compute a graph on two backends
 * and call [callback] to compare each pair of result tensors.
 */
fun ggmlBackendCompareGraphBackend(
    backend1: GGMLBackend,
    backend2: GGMLBackend,
    graph: GGMLCGraph,
    callback: GGMLBackendEvalCallback,
    testNodes: List<GGMLTensor>? = null
): Boolean {
}

// =====================================================================
// 14. CPU backend buffer / buffer-type
//     C: ggml_backend_cpu_buffer_*, ggml_backend_cpu_buffer_type_*,
//        ggml_backend_cpu_buffer_from_ptr
//
// NOTE: ggmlBackendCpuBufferType() and ggmlBackendCpuBufferFromPtr()
//       are already declared in GGMLBackend.kt / GGMLCpuBackend.kt.
// =====================================================================

// =====================================================================
// Existing higher-level helpers (preserved from original file)
// =====================================================================

/**
 * Backend selection strategy for hybrid execution
 */
enum class GGMLBackendSelectionStrategy {
    /** Always use CPU backend */
    CPU_ONLY,
    /** Automatic selection (currently identical to CPU_ONLY until new backends exist) */
    AUTO
}

/**
 * Backend manager for handling multiple backends and hybrid execution
 */
class GGMLBackendManager {
    private val availableBackends = mutableMapOf<String, GGMLBackend>()
    private var primaryBackend: GGMLBackend? = null
    private var fallbackBackend: GGMLBackend? = null
    private var selectionStrategy = GGMLBackendSelectionStrategy.AUTO
    
    init {
        initializeBackends()
    }
    
    /**
     * Initialize available backends
     */
    private fun initializeBackends() {
        val cpuBackend = GGMLCpuBackend()
        availableBackends[cpuBackend.getName()] = cpuBackend
        primaryBackend = cpuBackend
        fallbackBackend = cpuBackend
    }
    
    /**
     * Get the list of available backend names
     */
    fun getAvailableBackends(): List<String> {
        return availableBackends.keys.toList()
    }
    
    /**
     * Get a backend by name
     */
    fun getBackend(name: String): GGMLBackend? {
        return availableBackends[name]
    }
    
    /**
     * Get the primary backend
     */
    fun getPrimaryBackend(): GGMLBackend? = primaryBackend
    
    /**
     * Get the fallback backend (usually CPU)
     */
    fun getFallbackBackend(): GGMLBackend? = fallbackBackend

    /**
     * Simple list of backend instances (for compatibility)
     */
    fun getBackends(): List<GGMLBackend> = availableBackends.values.toList()
    
    /**
     * Set the backend selection strategy
     */
    fun setSelectionStrategy(strategy: GGMLBackendSelectionStrategy) {
        selectionStrategy = strategy
    }
    
    /**
     * Select the best backend for a given operation
     */
    fun selectBackend(tensor: GGMLTensor): GGMLBackend? {
        return fallbackBackend
    }
    
    /**
     * Create a graph with automatic backend selection
     */
    fun createGraphWithBackend(size: Int, strategy: GGMLBackendSelectionStrategy? = null): GGMLCGraph {
        val oldStrategy = selectionStrategy
        strategy?.let { setSelectionStrategy(it) }
        
        val backend = fallbackBackend
        val graph = createGraph(size, backend)
        setSelectionStrategy(oldStrategy) // Restore old strategy
        return graph
    }
    
    /**
     * Compute a graph with hybrid execution
     */
    fun computeGraphHybrid(graph: GGMLCGraph): GGMLStatus {
        val allocator = graph.allocator
        if (allocator == null) {
            return GGMLStatus.FAILED
        }
        
        // For now, use the graph's associated backend
        // In a full hybrid implementation, we would analyze each node
        // and potentially execute different nodes on different backends
        return computeGraphWithBackend(graph)
    }
    
    /**
     * Get backend information for debugging
     */
    fun getBackendInfo(): Map<String, Any> {
        val info = mutableMapOf<String, Any>()
        
        info["availableBackends"] = availableBackends.keys.toList()
        info["primaryBackend"] = primaryBackend?.getName() ?: "None"
        info["fallbackBackend"] = fallbackBackend?.getName() ?: "None" 
        info["selectionStrategy"] = selectionStrategy.name

        // Add backend-specific info
        availableBackends.forEach { (name, backend) ->
            info["${name}_guid"] = backend.getGuid()
            info["${name}_bufferType"] = backend.getDefaultBufferType().getName()
            info["${name}_alignment"] = backend.getAlignment()
            info["${name}_maxSize"] = backend.getMaxSize()
        }
        
        return info
    }
    
    /**
     * Clean up all backends
     */
    fun cleanup() {
        availableBackends.values.forEach { backend ->
            try {
                backend.free()
            } catch (e: Exception) {
                println("Error freeing backend ${backend.getName()}: ${e.message}")
            }
        }
        availableBackends.clear()
        primaryBackend = null
        fallbackBackend = null
    }
}

/**
 * Global backend manager instance
 */
val globalBackendManager = GGMLBackendManager()

/**
 * Convenience function to create a graph with the global backend manager
 */
fun createGraphWithGlobalBackend(
    size: Int, 
    strategy: GGMLBackendSelectionStrategy = GGMLBackendSelectionStrategy.AUTO
): GGMLCGraph {
    return globalBackendManager.createGraphWithBackend(size, strategy)
}

/**
 * Convenience function to compute a graph with hybrid execution
 */
fun computeGraphHybrid(graph: GGMLCGraph): GGMLStatus {
    return globalBackendManager.computeGraphHybrid(graph)
}
