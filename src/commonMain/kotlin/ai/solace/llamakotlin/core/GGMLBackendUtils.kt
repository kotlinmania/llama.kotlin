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
 * Scheduler, graph-copy, and CPU buffer implementations follow the
 * transliterated C++ code from ggml-backend.cpp.
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


// Functions matching ggml-backend.h declarations moved to GGMLBackend.kt

// =====================================================================
// 2. Backend buffer free-standing functions
//    C: ggml_backend_buffer_*
// =====================================================================

// ggmlBackendBufferCopyTensor lives in GGMLBackendImpl.kt (declared in ggml-backend-impl.h)

// =====================================================================
// 3. Backend (stream) free-standing functions
//    C: ggml_backend_*
// =====================================================================

/** `ggml_backend_graph_optimize` (static in C) */
fun ggmlBackendGraphOptimize(backend: GGMLBackend, graph: GGMLCGraph) {
    // optional: identity unless a backend overrides
}

// =====================================================================
// 4. Tensor copy helpers
//    C: ggml_backend_tensor_copy / _async
// =====================================================================

// =====================================================================
// 5. Event operations
//    C: ggml_backend_event_*
// =====================================================================

// =====================================================================
// 6. Device operations
//    C: ggml_backend_dev_*
// =====================================================================

// =====================================================================
// 7. Backend registration (reg) operations
//    C: ggml_backend_reg_*
// =====================================================================

// =====================================================================
// 7b. Multi-buffer
//     C: ggml-backend.cpp lines 667-735
// =====================================================================

/**
 * Context for a multi-buffer (logical buffer wrapping several sub-buffers).
 * Mirrors `ggml_backend_multi_buffer_context` in C.
 */
class GGMLBackendMultiBufferContext(
    val buffers: MutableList<GGMLBackendBuffer> = mutableListOf()
)

/** Marker wrapper class for multi-buffer detection. C: ggml-backend.cpp lines 637-663. */
class GGMLBackendMultiBufferWrapper(
    private val ctx: GGMLBackendMultiBufferContext,
    private val buft: GGMLBackendBufferType,
    private val totalSize: ULong
) : GGMLBackendBuffer {
    override fun getType(): GGMLBackendBufferType = buft
    override fun getName(): String = buft.getName()
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
        ctx.buffers.forEach { it.setUsage(usage) }
    }
}

/**
 * `ggml_backend_multi_buffer_alloc_buffer`
 * Allocate a logical multi-buffer wrapping several sub-buffers.
 * C: ggml-backend.cpp line 707.
 */
fun ggmlBackendMultiBufferAllocBuffer(buffers: List<GGMLBackendBuffer>): GGMLBackendBuffer {
    require(buffers.isNotEmpty()) { "multi-buffer requires at least one sub-buffer" }
    val ctx = GGMLBackendMultiBufferContext(buffers.toMutableList())
    var totalSize = 0UL
    for (buf in buffers) {
        totalSize += buf.getSize()
    }
    return GGMLBackendMultiBufferWrapper(ctx, buffers[0].getType(), totalSize)
}

// ggmlBackendBufferIsMultiBuffer and ggmlBackendMultiBufferSetUsage live in GGMLBackendImpl.kt

/**
 * `ggml_backend_multi_buffer_free_buffer` — static vtable entry in C.
 * Frees all sub-buffers in a multi-buffer wrapper.
 * C: ggml-backend.cpp line 674.
 */
fun ggmlBackendMultiBufferFreeBuffer(buffer: GGMLBackendBuffer) {
    buffer.free()
}

/**
 * `ggml_backend_multi_buffer_clear` — static vtable entry in C.
 * Clears all sub-buffers in a multi-buffer.
 * C: ggml-backend.cpp line 685.
 */
fun ggmlBackendMultiBufferClear(buffer: GGMLBackendBuffer, value: UByte) {
    buffer.clear(value)
}

// =====================================================================
// 8. Backend registry free functions (ggml-backend-reg.cpp)
// =====================================================================

// =====================================================================
// 8b. Meta backend functions (ggml-backend-meta.cpp)
// =====================================================================

// =====================================================================
// 8c. Graph copy utilities (ggml-backend.cpp lines 2007-2209)
// =====================================================================

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

/** Evaluation callback used by `ggml_backend_compare_graph_backend`. */
typealias GGMLBackendEvalCallback = (nodeIndex: Int, t1: GGMLTensor, t2: GGMLTensor) -> Boolean

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
//     C: struct ggml_backend_sched_split (ggml-backend.cpp lines 764-772)
// =====================================================================

/**
 * A single split in the scheduler's graph decomposition.
 * Mirrors `struct ggml_backend_sched_split` in C.
 */
data class GGMLBackendSchedSplit(
    var backendId: Int = -1,
    var iStart: Int = 0,
    var iEnd: Int = 0,
    val inputs: Array<GGMLTensor?> = arrayOfNulls(GGML_SCHED_MAX_SPLIT_INPUTS),
    var nInputs: Int = 0,
    var graph: GGMLCGraph = GGMLCGraph()
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
 * Find the index of [backend] in the scheduler's backends array.
 * Returns -1 if not found. Lower index = higher priority.
 * C: `ggml_backend_sched_backend_id` (lines 836-843)
 */
fun ggmlBackendSchedBackendId(sched: GGMLBackendSched, backend: GGMLBackend): Int {
    for (i in 0 until sched.nBackends) {
        if (sched.backends[i] === backend) {
            return i
        }
    }
    return -1
}

/**
 * Find the highest-priority backend that supports the buffer type of [tensor] and the operation [op].
 * C: `ggml_backend_sched_backend_from_buffer` (lines 845-865)
 */
fun ggmlBackendSchedBackendFromBuffer(sched: GGMLBackendSched, tensor: GGMLTensor, op: GGMLTensor): Int {
    val buffer = tensor.viewSrc?.buffer ?: tensor.buffer ?: return -1

    for (i in 0 until sched.nBackends) {
        if (ggmlBackendSupportsBuft(sched.backends[i]!!, buffer.getType()) &&
            ggmlBackendSupportsOp(sched.backends[i]!!, op)
        ) {
            return i
        }
    }

    return -1
}

/**
 * Determine the backend that should own a node based on its current buffer
 * allocations, view sources, and weight locations.
 * C: `ggml_backend_sched_backend_id_from_cur` (lines 878-933)
 */
fun ggmlBackendSchedBackendIdFromCur(sched: GGMLBackendSched, tensor: GGMLTensor): Int {
    // assign pre-allocated nodes to their backend
    var curBackendId = ggmlBackendSchedBackendFromBuffer(sched, tensor, tensor)
    if (curBackendId != -1) {
        return curBackendId
    }

    // view_src
    if (tensor.viewSrc != null) {
        curBackendId = ggmlBackendSchedBackendFromBuffer(sched, tensor.viewSrc!!, tensor)
        if (curBackendId != -1) {
            return curBackendId
        }
    }

    if (tensor.buffer != null || (tensor.viewSrc != null && tensor.viewSrc!!.buffer != null)) {
        val buffer = if (tensor.viewSrc != null) tensor.viewSrc!!.buffer else tensor.buffer
        error("pre-allocated tensor (${tensor.name}) in a buffer (${ggmlBackendBufferName(buffer!!)}) that cannot run the operation (${tensor.op.name})")
    }

    // graph input
    if (tensor.flags and GGML_TENSOR_FLAG_INPUT != 0) {
        curBackendId = sched.nBackends - 1 // last backend (assumed CPU)
        return curBackendId
    }

    // operations with weights are preferably run on the same backend as the weights
    for (i in 0 until GGML_MAX_SRC) {
        val src = tensor.src[i] ?: continue
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        if (tensor.op != GGMLOp.ROPE && src.buffer != null &&
            src.buffer!!.getUsage() == GGMLBackendBufferUsage.WEIGHTS
        ) {
            val srcBackendId = ggmlBackendSchedBackendFromBuffer(sched, src, tensor)
            // check if a backend with higher prio wants to offload the op
            if (sched.opOffload && srcBackendId == sched.nBackends - 1 &&
                ggmlBackendBufferIsHost(src.buffer!!)
            ) {
                for (b in 0 until srcBackendId) {
                    if (ggmlBackendSupportsOp(sched.backends[b]!!, tensor) &&
                        ggmlBackendOffloadOp(sched.backends[b]!!, tensor)
                    ) {
                        return b
                    }
                }
            }
            return srcBackendId
        }
    }

    return -1
}

/**
 * Print the node→backend assignments for debugging.
 * C: `ggml_backend_sched_print_assignments` (lines 945-983)
 */
fun ggmlBackendSchedPrintAssignments(sched: GGMLBackendSched, graph: GGMLCGraph) {
    var curSplit = 0
    for (i in 0 until graph.nNodes) {
        if (curSplit < sched.nSplits && i == sched.splits[curSplit].iStart) {
            val splitBackend = sched.backends[sched.splits[curSplit].backendId]
            val sb = StringBuilder()
            sb.append("\n## SPLIT #$curSplit: ${ggmlBackendName(splitBackend)} # ${sched.splits[curSplit].nInputs} inputs")
            for (j in 0 until sched.splits[curSplit].nInputs) {
                if (j == 0) sb.append(": ")
                val inp = sched.splits[curSplit].inputs[j]!!
                sb.append("[${inp.name} (${fmtSize(ggmlNbytes(inp))})] ")
            }
            println(sb.toString())
            curSplit++
        }
        val node = graph.nodes[i] ?: continue
        if (ggmlIsViewOp(node.op)) continue

        if (sched.debug > 1) {
            val tensorBackend = sched.getTensorBackend(node)
            val sb = StringBuilder()
            sb.append("node #${i}: ${node.op.name} ${node.name} (${fmtSize(ggmlNbytes(node))}) [${tensorBackend?.getName() ?: "NULL"}]:")
            for (j in 0 until GGML_MAX_SRC) {
                val src = node.src[j] ?: continue
                val srcBackend = sched.getTensorBackend(src)
                sb.append(" ${src.name} (${fmtSize(ggmlNbytes(src))}) [${srcBackend?.getName() ?: "NULL"}]")
            }
            println(sb.toString())
        }
    }
}

/**
 * Check if a tensor's buffer type is supported on a given backend.
 * C: `ggml_backend_sched_buffer_supported` (lines 985-1004)
 */
fun ggmlBackendSchedBufferSupported(sched: GGMLBackendSched, t: GGMLTensor, backendId: Int): Boolean {
    val buf = t.viewSrc?.buffer ?: t.buffer
    var buft: GGMLBackendBufferType? = null

    if (buf != null) {
        buft = buf.getType()
    } else {
        var tensorBid = sched.tensorBackendId(t)
        if (tensorBid == -1 && t.viewSrc != null) {
            tensorBid = sched.tensorBackendId(t.viewSrc!!)
        }
        if (tensorBid != -1) {
            buft = sched.bufts[tensorBid]
        }
    }

    return buft != null && ggmlBackendSupportsBuft(sched.backends[backendId]!!, buft)
}

/**
 * Assign a backend to a node if the backend supports the operation.
 * C: `ggml_backend_sched_set_if_supported` (lines 1006-1011)
 */
fun ggmlBackendSchedSetIfSupported(sched: GGMLBackendSched, node: GGMLTensor, curBackendId: Int) {
    if (ggmlBackendSupportsOp(sched.backends[curBackendId]!!, node)) {
        sched.setTensorBackendId(node, curBackendId)
    }
}

/**
 * Allocate memory for all splits using the graph allocator.
 * C: `ggml_backend_sched_alloc_splits` (lines 1489-1539)
 */
fun ggmlBackendSchedAllocSplits(sched: GGMLBackendSched): Boolean {
    var backendIdsChanged = false
    for (i in 0 until sched.graph.nNodes) {
        if (sched.nodeBackendIds[i] != sched.prevNodeBackendIds[i] &&
            sched.bufts[sched.nodeBackendIds[i]] !== sched.bufts[sched.prevNodeBackendIds[i]]
        ) {
            backendIdsChanged = true
            break
        }
    }
    if (!backendIdsChanged) {
        for (i in 0 until sched.graph.nLeafs) {
            if (sched.leafBackendIds[i] != sched.prevLeafBackendIds[i] &&
                sched.bufts[sched.leafBackendIds[i]] !== sched.bufts[sched.prevLeafBackendIds[i]]
            ) {
                backendIdsChanged = true
                break
            }
        }
    }

    // allocate graph via gallocr
    // if (backendIdsChanged || !ggmlGallocrAllocGraph(sched.galloc, sched.graph)) {
    if (backendIdsChanged) {
        if (sched.debugRealloc > 0) {
            val unexpected = !backendIdsChanged && sched.debugPrevGraphSize == sched.debugGraphSize
            if (unexpected || sched.debugRealloc > 1) {
                error("unexpected graph reallocation (graph size = ${sched.debugGraphSize})")
            }
        }

        // synchronize all backends before re-allocation
        for (i in 0 until sched.nBackends) {
            ggmlBackendSynchronize(sched.backends[i]!!)
        }

        // ggmlGallocrReserveN(sched.galloc, sched.graph, sched.nodeBackendIds, sched.leafBackendIds)
        // if (!ggmlGallocrAllocGraph(sched.galloc, sched.graph)) {
        //     return false
        // }
    }

    return true
}

/**
 * Execute all splits, copying inputs between backends as needed.
 * C: `ggml_backend_sched_compute_splits` (lines 1541-1725)
 */
fun ggmlBackendSchedComputeSplits(sched: GGMLBackendSched): GGMLStatus {
    for (splitId in 0 until sched.nSplits) {
        val split = sched.splits[splitId]
        val splitBid = split.backendId
        val splitBackend = sched.backends[splitBid]!!

        // copy the input tensors to the split backend
        for (inputId in 0 until split.nInputs) {
            val input = split.inputs[inputId]!!
            val inputCpy = sched.tensorCopy(input, splitBid, sched.curCopy)!!

            if (input.flags and GGML_TENSOR_FLAG_INPUT != 0) {
                // inputs from the user must be copied immediately
                if (sched.events[splitBid][sched.curCopy] != null) {
                    ggmlBackendEventSynchronize(sched.events[splitBid][sched.curCopy]!!)
                } else {
                    ggmlBackendSynchronize(splitBackend)
                }
                ggmlBackendTensorCopy(input, inputCpy)
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched.events[splitBid][sched.curCopy] != null) {
                    ggmlBackendEventWait(splitBackend, sched.events[splitBid][sched.curCopy]!!)
                } else {
                    ggmlBackendSynchronize(splitBackend)
                }

                // MoE expert optimization: check if we can copy only used experts
                val firstNode = if (split.graph.nNodes > 0) split.graph.nodes[0] else null
                if (firstNode != null &&
                    input.buffer != null &&
                    ggmlBackendBufferGetUsage(input.buffer!!) == GGMLBackendBufferUsage.WEIGHTS &&
                    ggmlBackendBufferIsHost(input.buffer!!) &&
                    firstNode.src[0] === inputCpy && firstNode.op == GGMLOp.MUL_MAT_ID
                ) {
                    // MoE weight copy optimization — copy full tensor for now
                    // Full expert-level copy optimization requires bitset and ids tensor inspection
                    ggmlBackendTensorCopy(input, inputCpy)
                } else {
                    // try async copy, fallback to sync
                    val inputBackend = sched.getTensorBackend(input)
                    if (inputBackend != null) {
                        if (!splitBackend.copyTensorAsync(inputBackend, input, inputCpy)) {
                            ggmlBackendSynchronize(inputBackend)
                            if (sched.events[splitBid][sched.curCopy] != null) {
                                ggmlBackendEventSynchronize(sched.events[splitBid][sched.curCopy]!!)
                            } else {
                                ggmlBackendSynchronize(splitBackend)
                            }
                            ggmlBackendTensorCopy(input, inputCpy)
                        }
                    } else {
                        ggmlBackendTensorCopy(input, inputCpy)
                    }
                }
            }
        }

        if (sched.callbackEval == null) {
            val ec = ggmlBackendGraphComputeAsync(splitBackend, split.graph)
            if (ec != GGMLStatus.SUCCESS) {
                return ec
            }
        } else {
            // compute with eval callback — similar to compare_graph_backend
            var j0 = 0
            while (j0 < split.graph.nNodes) {
                var t = split.graph.nodes[j0]!!
                var need = sched.callbackEval!!(t, true, sched.callbackEvalUserData)
                var j1 = j0

                while (!need && j1 < split.graph.nNodes - 1) {
                    t = split.graph.nodes[++j1]!!
                    need = sched.callbackEval!!(t, true, sched.callbackEvalUserData)
                }

                val gv = ggml_graph_view(split.graph, j0, j1 + 1)
                val ec = ggmlBackendGraphComputeAsync(splitBackend, gv)
                if (ec != GGMLStatus.SUCCESS) {
                    return ec
                }

                ggmlBackendSynchronize(splitBackend)

                if (need && !sched.callbackEval!!(t, false, sched.callbackEvalUserData)) {
                    break
                }

                j0 = j1 + 1
            }
        }

        // record the event of this copy
        if (split.nInputs > 0) {
            if (sched.events[splitBid][sched.curCopy] != null) {
                ggmlBackendEventRecord(sched.events[splitBid][sched.curCopy]!!, splitBackend)
            }
        }
    }

    return GGMLStatus.SUCCESS
}

// =====================================================================
// 12. View / tensor alloc utils
//     C: ggml_backend_view_init, ggml_backend_tensor_alloc
// =====================================================================

// =====================================================================
// 13. Graph copy / compare helpers
//     C: graph_copy_dup_tensor, graph_copy_init_tensor
//     (ggml-backend.cpp lines 2007-2066)
// =====================================================================

/**
 * Recursively duplicate a tensor and all its sources for graph copy.
 * C: `graph_copy_dup_tensor` (lines 2007-2038)
 */
fun graphCopyDupTensor(
    hashSet: GGMLHashSet,
    nodeCopies: Array<GGMLTensor?>,
    ctxAllocated: GGMLContext,
    ctxUnallocated: GGMLContext,
    src: GGMLTensor
): GGMLTensor {
    val id = ggml_hash_insert(hashSet, src)
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return nodeCopies[ggml_hash_find(hashSet, src)]!!
    }

    val ctx = if (src.data != null && src.viewSrc == null) ctxAllocated else ctxUnallocated
    val dst = ggmlDupTensorLayout(ctx, src)
    if (src.viewSrc != null) {
        dst.viewSrc = graphCopyDupTensor(hashSet, nodeCopies, ctxAllocated, ctxUnallocated, src.viewSrc!!)
        dst.viewOffs = src.viewOffs
    }
    dst.op = src.op
    dst.flags = src.flags
    src.opParams?.let { dst.opParams = it.copyOf() }
    dst.name = src.name

    for (i in 0 until GGML_MAX_SRC) {
        val s = src.src[i] ?: continue
        dst.src[i] = graphCopyDupTensor(hashSet, nodeCopies, ctxAllocated, ctxUnallocated, s)
    }

    nodeCopies[id] = dst
    return dst
}

/**
 * Initialize a copied tensor — copy data or init view.
 * C: `graph_copy_init_tensor` (lines 2041-2066)
 */
fun graphCopyInitTensor(
    hashSet: GGMLHashSet,
    nodeCopies: Array<GGMLTensor?>,
    nodeInit: BooleanArray,
    src: GGMLTensor
) {
    val id = ggml_hash_find(hashSet, src)
    if (nodeInit[id]) return
    nodeInit[id] = true

    val dst = nodeCopies[id]!!
    if (dst.viewSrc != null) {
        graphCopyInitTensor(hashSet, nodeCopies, nodeInit, src.viewSrc!!)
        val status = ggmlBackendViewInit(dst)
        require(status == GGMLStatus.SUCCESS)
    } else {
        ggmlBackendTensorCopy(src, dst)
    }

    for (i in 0 until GGML_MAX_SRC) {
        val s = src.src[i] ?: continue
        graphCopyInitTensor(hashSet, nodeCopies, nodeInit, s)
    }
}

// =====================================================================
// 14. CPU backend buffer / buffer-type
//     C: ggml_backend_cpu_buffer_* (lines 2211-2371)
//
//     GGMLCpuBuffer and GGMLCpuBufferType are implemented in GGMLCpuBackend.kt.
//     GGMLCpuBufferFromPtr and GGMLCpuBufferFromPtrType are unique to this file
//     (they wrap external byte arrays without owning them).
// =====================================================================

/**
 * CPU buffer type for buffers created from external pointers (not owned).
 * C: `ggml_backend_cpu_buffer_from_ptr_type` (lines 2351-2366)
 */
class GGMLCpuBufferFromPtrType : GGMLBackendBufferType {
    override fun getName(): String = "CPU_Mapped"
    override fun allocBuffer(size: ULong): GGMLBackendBuffer {
        return createDefaultCpuBufferType().allocBuffer(size)
            ?: throw IllegalStateException("Failed to allocate CPU buffer of size $size")
    }
    override fun getAlignment(): UInt = TENSOR_ALIGNMENT.toUInt()
    override fun getMaxSize(): ULong = ULong.MAX_VALUE
    override fun getAllocSize(tensor: GGMLTensor): ULong = ggmlNbytes(tensor)
    override fun isHost(): Boolean = true
    override fun getDevice(): GGMLBackendDevice? = null
}

/**
 * Create a CPU buffer wrapping an existing byte array (not owned by the buffer).
 * C: `ggml_backend_cpu_buffer_from_ptr` (lines 2368-2371)
 */
class GGMLCpuBufferFromPtr(
    private val ptr: ByteArray,
    private val size: ULong
) : GGMLBackendBuffer {
    private val buft = GGMLCpuBufferFromPtrType()
    private var usage = GGMLBackendBufferUsage.COMPUTE

    override fun getType(): GGMLBackendBufferType = buft
    override fun getName(): String = "CPU_Mapped"
    override fun getBase(): Any = ptr
    override fun getSize(): ULong = size
    override fun free() {}
    override fun initTensor(tensor: GGMLTensor): GGMLStatus = GGMLStatus.SUCCESS
    override fun setTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val dst = tensor.data
        if (dst is ByteArray) data.copyInto(dst, offset.toInt(), 0, size.toInt())
    }
    override fun getTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val src = tensor.data
        if (src is ByteArray) src.copyInto(data, 0, offset.toInt(), (offset + size).toInt())
    }
    override fun copyTensor(src: GGMLTensor, dst: GGMLTensor): Boolean {
        val srcBuf = src.buffer ?: return false
        if (ggmlBackendBufferIsHost(srcBuf)) {
            val srcData = src.data
            val dstData = dst.data
            if (srcData is ByteArray && dstData is ByteArray) {
                srcData.copyInto(dstData, 0, 0, ggmlNbytes(src).toInt())
                return true
            }
        }
        return false
    }
    override fun clear(value: UByte) { ptr.fill(value.toByte()) }
    override fun setUsage(usage: GGMLBackendBufferUsage) { this.usage = usage }
    override fun getUsage(): GGMLBackendBufferUsage = usage
    override fun reset() { clear(0u) }
}

// =====================================================================
// 15. CPU buffer vtable functions (top-level wrappers)
//     C: static ggml_backend_cpu_buffer_* functions (lines 2213-2327)
//     These are static vtable entries in C. In Kotlin the logic is in
//     GGMLCpuBuffer / GGMLCpuBufferType classes — these wrappers exist
//     for naming parity with the C source.
// =====================================================================

/** `ggml_backend_cpu_buffer_get_base` — C line 2213 (static vtable entry). */
fun ggmlBackendCpuBufferGetBase(buffer: GGMLBackendBuffer): Any? = buffer.getBase()

/** `ggml_backend_cpu_buffer_free_buffer` — C line 2225 (static vtable entry). */
fun ggmlBackendCpuBufferFreeBuffer(buffer: GGMLBackendBuffer) = buffer.free()

/** `ggml_backend_cpu_buffer_memset_tensor` — C line 2230 (static vtable entry). */
fun ggmlBackendCpuBufferMemsetTensor(buffer: GGMLBackendBuffer, tensor: GGMLTensor, value: UByte, offset: ULong, size: ULong) {
    val data = tensor.data
    if (data is ByteArray) {
        data.fill(value.toByte(), offset.toInt(), (offset + size).toInt())
    }
}

/** `ggml_backend_cpu_buffer_set_tensor` — C line 2237 (static vtable entry). */
fun ggmlBackendCpuBufferSetTensor(buffer: GGMLBackendBuffer, tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    buffer.setTensor(tensor, data, offset, size)
}

/** `ggml_backend_cpu_buffer_get_tensor` — C line 2244 (static vtable entry). */
fun ggmlBackendCpuBufferGetTensor(buffer: GGMLBackendBuffer, tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    buffer.getTensor(tensor, data, offset, size)
}

/** `ggml_backend_cpu_buffer_cpy_tensor` — C line 2251 (static vtable entry). */
fun ggmlBackendCpuBufferCpyTensor(buffer: GGMLBackendBuffer, src: GGMLTensor, dst: GGMLTensor): Boolean {
    return buffer.copyTensor(src, dst)
}

/** `ggml_backend_cpu_buffer_clear` — C line 2262 (static vtable entry). */
fun ggmlBackendCpuBufferClear(buffer: GGMLBackendBuffer, value: UByte) {
    buffer.clear(value)
}

/** `ggml_backend_cpu_buffer_type_get_name` — C line 2299 (static vtable entry). */
fun ggmlBackendCpuBufferTypeGetName(buft: GGMLBackendBufferType): String = buft.getName()

/** `ggml_backend_cpu_buffer_type_alloc_buffer` — C line 2305 (static vtable entry). */
fun ggmlBackendCpuBufferTypeAllocBuffer(buft: GGMLBackendBufferType, size: ULong): GGMLBackendBuffer? {
    return buft.allocBuffer(size)
}

/** `ggml_backend_cpu_buffer_type_get_alignment` — C line 2316 (static vtable entry). */
fun ggmlBackendCpuBufferTypeGetAlignment(buft: GGMLBackendBufferType): UInt = buft.getAlignment()

/** `ggml_backend_cpu_buffer_type_is_host` — C line 2322 (static vtable entry). */
fun ggmlBackendCpuBufferTypeIsHost(buft: GGMLBackendBufferType): Boolean = buft.isHost()

/** `ggml_backend_cpu_buffer_from_ptr_type_get_name` — C line 2345. */
fun ggmlBackendCpuBufferFromPtrTypeGetName(buft: GGMLBackendBufferType): String = buft.getName()

/** `ggml_backend_cpu_buffer_from_ptr_type` — C line 2351. Returns singleton buft. */
fun ggmlBackendCpuBufferFromPtrType(): GGMLBackendBufferType = GGMLCpuBufferFromPtrType()

// Multi-buffer functions in GGMLBackendImpl.kt (declared in ggml-backend-impl.h)

// ---------------------------------------------------------------------------
// Scheduler public API  (ggml-backend.cpp lines 1727-1976)
// Top-level functions matching C naming, delegating to GGMLBackendSched class.
// ---------------------------------------------------------------------------

