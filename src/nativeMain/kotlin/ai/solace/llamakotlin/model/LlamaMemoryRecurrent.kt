// port-lint: source llama.cpp/src/llama-memory-recurrent.h llama.cpp/src/llama-memory-recurrent.cpp
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

// =============================================================================
// LlamaMemoryRecurrent – recurrent state cache for SSM / RWKV models
// Ported from: llama-memory-recurrent.h / llama-memory-recurrent.cpp
// =============================================================================

// LayerFilterCallback is defined in LlamaMemory.kt

// ---------------------------------------------------------------------------
// MemCell – per-slot metadata
// ---------------------------------------------------------------------------

/**
 * Metadata for a single recurrent-memory cell.
 *
 * Port of `llama_memory_recurrent::mem_cell`.
 *
 * @property pos   Token position, or -1 if empty.
 * @property src   Source cell index for state copying (general).
 * @property src0  Source cell index used only during input setting.
 * @property tail  Index of the tail cell in the sequence (for ring-buffer logic).
 */
data class RecurrentMemCell(
    var pos: LlamaPos = -1,
    var src: Int = -1,
    var src0: Int = -1,
    var tail: Int = -1,
) {
    /** Sequence ids that own this cell. */
    val seqId: MutableSet<LlamaSeqId> = mutableSetOf()

    /** `true` if [id] is among the occupying sequences. */
    fun hasSeqId(id: LlamaSeqId): Boolean = id in seqId

    /** `true` when the cell is unoccupied. */
    fun isEmpty(): Boolean = seqId.isEmpty()

    /** `true` when this cell has exactly the same sequence membership as [other]. */
    fun isSameSeq(other: RecurrentMemCell): Boolean = seqId == other.seqId
}

// ---------------------------------------------------------------------------
// LlamaMemoryRecurrent
// ---------------------------------------------------------------------------

/**
 * Recurrent-state memory manager for SSM and RWKV model families.
 *
 * This class is the Kotlin port of `class llama_memory_recurrent` from
 * `llama-memory-recurrent.h`. It manages a fixed pool of [RecurrentMemCell]
 * entries and per-layer r/s state tensors that are re-used across inference
 * steps.
 *
 * @param hparams    Model hyper-parameters (layer count, state sizes, etc.).
 * @param typeR      GGML data type for the "r" (recurrent) state tensors.
 * @param typeS      GGML data type for the "s" (SSM hidden) state tensors.
 * @param offload    Whether to offload state tensors to accelerator memory.
 * @param memSize    Total number of recurrent-memory cells to allocate.
 * @param nSeqMax    Maximum number of concurrent sequences.
 * @param filter     Per-layer filter deciding which layers get state buffers.
 */
class LlamaMemoryRecurrent(
    private val hparams: LlamaModelHParams,
    private val typeR: GGMLType = GGMLType.F32,
    private val typeS: GGMLType = GGMLType.F32,
    private val offload: Boolean = false,
    memSize: Int,
    private val nSeqMax: Int = 1,
    private val filter: LayerFilterCallback = { true },
) {
    /** Location where the next batch will be placed (see [findSlot]). */
    var head: Int = 0

    /** Total number of cells. */
    var size: Int = memSize

    /** Number of currently used cells (≥ 1 seq_id). */
    var used: Int = 0

    /** Number of active cells — computed before each graph build. */
    var n: Int = 0

    /** Index of the first zero-initialised state, or -1 if none. */
    var rsZ: Int = -1

    /** Per-slot metadata. */
    val cells: MutableList<RecurrentMemCell> =
        MutableList(memSize) { RecurrentMemCell() }

    /** Per-layer "r" state tensors (recurrent hidden state). */
    val rL: MutableList<GGMLTensor?> = MutableList(hparams.nLayer) { null }

    /** Per-layer "s" state tensors (SSM hidden state). */
    val sL: MutableList<GGMLTensor?> = MutableList(hparams.nLayer) { null }

    // -----------------------------------------------------------------------
    // Memory interface methods (mirror llama_memory_i)
    // -----------------------------------------------------------------------

    /**
     * Initialise a batch-processing context.
     *
     * Port of `llama_memory_recurrent::init_batch()`.
     */
    fun initBatch(
        balloc: LlamaBatchAllocr,
        nUbatch: Int,
        embdAll: Boolean,
    ): LlamaMemoryRecurrentContext {
        TODO("Port llama_memory_recurrent::init_batch")
    }

    /**
     * Initialise a full-cache context (used for state save/load).
     *
     * Port of `llama_memory_recurrent::init_full()`.
     */
    fun initFull(): LlamaMemoryRecurrentContext {
        TODO("Port llama_memory_recurrent::init_full")
    }

    /**
     * Initialise an update context (defrag / optimise).
     *
     * Port of `llama_memory_recurrent::init_update()`.
     */
    fun initUpdate(
        lctx: LlamaContext,
        optimize: Boolean,
    ): LlamaMemoryRecurrentContext {
        TODO("Port llama_memory_recurrent::init_update")
    }

    /**
     * Clear all cells (and optionally the underlying tensor data).
     *
     * Port of `llama_memory_recurrent::clear()`.
     */
    fun clear(data: Boolean) {
        for (cell in cells) {
            cell.pos = -1
            cell.src = -1
            cell.src0 = -1
            cell.tail = -1
            cell.seqId.clear()
        }
        head = 0
        used = 0
        n = 0
        rsZ = -1
        // TODO: if `data`, zero-fill the r_l / s_l tensor backing buffers
    }

    // -- sequence operations ------------------------------------------------

    /**
     * Remove positions `[p0, p1)` for [seqId]. Pass `p0 < 0` to remove all.
     *
     * Port of `llama_memory_recurrent::seq_rm()`.
     *
     * @return `true` on success.
     */
    fun seqRm(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos): Boolean {
        TODO("Port llama_memory_recurrent::seq_rm")
    }

    /**
     * Copy sequence [seqIdSrc] to [seqIdDst] for positions `[p0, p1)`.
     *
     * Port of `llama_memory_recurrent::seq_cp()`.
     */
    fun seqCp(seqIdSrc: LlamaSeqId, seqIdDst: LlamaSeqId, p0: LlamaPos, p1: LlamaPos) {
        TODO("Port llama_memory_recurrent::seq_cp")
    }

    /**
     * Keep only [seqId]; remove all other sequences from every cell.
     *
     * Port of `llama_memory_recurrent::seq_keep()`.
     */
    fun seqKeep(seqId: LlamaSeqId) {
        TODO("Port llama_memory_recurrent::seq_keep")
    }

    /**
     * Shift positions of [seqId] in range `[p0, p1)` by [shift].
     *
     * Port of `llama_memory_recurrent::seq_add()`.
     */
    fun seqAdd(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, shift: LlamaPos) {
        TODO("Port llama_memory_recurrent::seq_add")
    }

    /**
     * Integer-divide positions of [seqId] in range `[p0, p1)` by [d].
     *
     * Port of `llama_memory_recurrent::seq_div()`.
     */
    fun seqDiv(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, d: Int) {
        TODO("Port llama_memory_recurrent::seq_div")
    }

    /** Minimum position among cells belonging to [seqId], or -1 if none. */
    fun seqPosMin(seqId: LlamaSeqId): LlamaPos {
        var minPos = Int.MAX_VALUE
        for (cell in cells) {
            if (seqId in cell.seqId && cell.pos >= 0) {
                if (cell.pos < minPos) minPos = cell.pos
            }
        }
        return if (minPos == Int.MAX_VALUE) -1 else minPos
    }

    /** Maximum position among cells belonging to [seqId], or -1 if none. */
    fun seqPosMax(seqId: LlamaSeqId): LlamaPos {
        var maxPos = Int.MIN_VALUE
        for (cell in cells) {
            if (seqId in cell.seqId && cell.pos >= 0) {
                if (cell.pos > maxPos) maxPos = cell.pos
            }
        }
        return if (maxPos == Int.MIN_VALUE) -1 else maxPos
    }

    // -- slot management ---------------------------------------------------

    /**
     * Prepare the memory for the given sequence of ubatches.
     *
     * Port of `llama_memory_recurrent::prepare()`.
     *
     * @return `true` when all ubatches can be accommodated.
     */
    fun prepare(ubatches: List<LlamaUBatch>): Boolean {
        TODO("Port llama_memory_recurrent::prepare")
    }

    /**
     * Find a contiguous slot and emplace the ubatch there.
     *
     * Port of `llama_memory_recurrent::find_slot()`.
     *
     * @return `true` on success; `false` when no contiguous slot is available.
     */
    fun findSlot(ubatch: LlamaUBatch): Boolean {
        TODO("Port llama_memory_recurrent::find_slot")
    }

    /**
     * Whether this memory implementation supports K-shift (it does not).
     *
     * Port of `llama_memory_recurrent::get_can_shift()`.
     */
    fun getCanShift(): Boolean = false

    // -- state serialisation -----------------------------------------------

    /**
     * Serialise memory state for the given sequence.
     *
     * Port of `llama_memory_recurrent::state_write()`.
     *
     * @param seqId Sequence to serialise, or -1 for all.
     */
    fun stateWrite(seqId: LlamaSeqId = -1) {
        TODO("Port llama_memory_recurrent::state_write")
    }

    /**
     * Deserialise memory state for the given sequence.
     *
     * Port of `llama_memory_recurrent::state_read()`.
     *
     * @param seqId Destination sequence, or -1 for all.
     */
    fun stateRead(seqId: LlamaSeqId = -1) {
        TODO("Port llama_memory_recurrent::state_read")
    }

    // -- private helpers ---------------------------------------------------

    /** Total bytes consumed by r + s state tensors. */
    private fun totalSize(): Long {
        return sizeRBytes() + sizeSBytes()
    }

    private fun sizeRBytes(): Long {
        TODO("Port llama_memory_recurrent::size_r_bytes")
    }

    private fun sizeSBytes(): Long {
        TODO("Port llama_memory_recurrent::size_s_bytes")
    }
}

// =============================================================================
// LlamaMemoryRecurrentContext – batch/full/update context
// Ported from: llama-memory-recurrent.h  class llama_memory_recurrent_context
// =============================================================================

/**
 * Context object handed out by [LlamaMemoryRecurrent] to manage a batch
 * processing session, a full-state snapshot, or a defrag/optimise pass.
 *
 * Port of `class llama_memory_recurrent_context`.
 */
class LlamaMemoryRecurrentContext private constructor(
    val status: LlamaMemoryStatus,
    private val mem: LlamaMemoryRecurrent?,
    private val ubatches: List<LlamaUBatch>,
    private val isFull: Boolean,
) {
    private var iNext: Int = 0

    // -- factory constructors matching C++ overloads -----------------------

    companion object {
        /** Error context. */
        fun error(status: LlamaMemoryStatus) =
            LlamaMemoryRecurrentContext(status, mem = null, ubatches = emptyList(), isFull = false)

        /** Full-cache or update context. */
        fun full(mem: LlamaMemoryRecurrent) =
            LlamaMemoryRecurrentContext(LlamaMemoryStatus.SUCCESS, mem, ubatches = emptyList(), isFull = true)

        /** Batch-processing context. */
        fun batch(mem: LlamaMemoryRecurrent, ubatches: List<LlamaUBatch>) =
            LlamaMemoryRecurrentContext(LlamaMemoryStatus.SUCCESS, mem, ubatches, isFull = false)
    }

    // -- iteration ---------------------------------------------------------

    /**
     * Advance to the next ubatch.
     *
     * Port of `llama_memory_recurrent_context::next()`.
     *
     * @return `true` while there is a ubatch to process; `false` when done.
     */
    fun next(): Boolean {
        if (iNext >= ubatches.size) return false
        iNext++
        return true
    }

    /**
     * Apply state changes for the current ubatch (find slot, etc.).
     *
     * Port of `llama_memory_recurrent_context::apply()`.
     */
    fun apply(): Boolean {
        TODO("Port llama_memory_recurrent_context::apply")
    }

    /** Status of this context. */
    fun getStatus(): LlamaMemoryStatus = status

    /** Currently active ubatch (valid after a successful [next] call). */
    fun getUbatch(): LlamaUBatch {
        require(iNext > 0 && iNext <= ubatches.size) { "No current ubatch — call next() first" }
        return ubatches[iNext - 1]
    }

    // -- recurrent-specific accessors --------------------------------------

    /** Number of active recurrent states. */
    fun getNRs(): Int = mem?.n ?: 0

    /** Head position in the cell ring. */
    fun getHead(): Int = mem?.head ?: 0

    /** Index of the first zero state. */
    fun getRsZ(): Int = mem?.rsZ ?: -1

    /** Total cell count. */
    fun getSize(): Int = mem?.size ?: 0

    /** Per-layer "r" state tensor, or `null`. */
    fun getRL(il: Int): GGMLTensor? = mem?.rL?.getOrNull(il)

    /** Per-layer "s" state tensor, or `null`. */
    fun getSL(il: Int): GGMLTensor? = mem?.sL?.getOrNull(il)

    /**
     * Source index for the i-th state copy.
     *
     * Port of `llama_memory_recurrent_context::s_copy()`.
     */
    fun sCopy(i: Int): Int {
        val cell = mem?.cells?.getOrNull(i) ?: return -1
        return cell.src
    }
}
