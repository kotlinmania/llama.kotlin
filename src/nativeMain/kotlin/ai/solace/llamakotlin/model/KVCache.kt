// port-lint: source llama.cpp/src/llama-kv-cache.cpp
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

// ---------------------------------------------------------------------------
// Type aliases matching C++ originals
// ---------------------------------------------------------------------------

/** Sequence ID type (maps to llama_seq_id in C++). */
typealias LlamaSeqId = Int

/** Token position type (maps to llama_pos in C++). */
typealias LlamaPos = Int

/** Maximum number of sequences supported. */
const val LLAMA_MAX_SEQ = 64

// ---------------------------------------------------------------------------
// Sliding-window attention type – mirrors llama_swa_type in llama-hparams.h
// ---------------------------------------------------------------------------

/**
 * Sliding-window attention variants.
 *
 * Maps 1-to-1 to the C++ `llama_swa_type` enum declared in
 * `llama.cpp/src/llama-hparams.h`.
 */
enum class LlamaSwaType {
    NONE,
    STANDARD,
    CHUNKED,
    SYMMETRIC,
}

// ---------------------------------------------------------------------------
// Memory status – mirrors llama_memory_status
// ---------------------------------------------------------------------------

/**
 * Status codes returned by memory-context operations.
 *
 * These map 1-to-1 to the C++ `llama_memory_status` enum.
 */
enum class LlamaMemoryStatus {
    SUCCESS,
    NO_UPDATE,
    FAILED_PREPARE,
    FAILED_COMPUTE,
}

// ---------------------------------------------------------------------------
// KvCell – per-slot metadata (port of llama_kv_cell)
// ---------------------------------------------------------------------------

/**
 * Metadata for a single KV cache cell.
 *
 * Each cell records its position in the sequence, which sequence IDs occupy it,
 * and an accumulated shift (used by K-shift / RoPE updates).
 *
 * @property pos  The token position stored in this cell, or -1 if empty.
 * @property delta  Accumulated position shift applied to this cell (for K-shift).
 */
data class KvCell(
    var pos: LlamaPos = -1,
    var delta: LlamaPos = 0
) {
    /** Set of sequence IDs that share this cell. */
    val seqIds: MutableSet<LlamaSeqId> = mutableSetOf()

    /** `true` when the cell contains valid cached data. */
    val hasPos: Boolean get() = pos >= 0

    /** `true` when no sequence occupies this cell. */
    val isEmpty: Boolean get() = seqIds.isEmpty()
}

// ---------------------------------------------------------------------------
// KvCells – a resizable array of KvCell with helper queries
// ---------------------------------------------------------------------------

/**
 * Container for an array of [KvCell] entries that backs one KV cache stream.
 *
 * Provides the same query surface as `llama_kv_cells` in C++ (position range
 * queries, sequence membership, shift tracking, etc.).
 */
class KvCells(size: Int = 0) {
    private val cells = ArrayList<KvCell>(size).apply {
        repeat(size) { add(KvCell()) }
    }

    private var usedCount = 0

    fun size(): Int = cells.size

    fun resize(n: Int) {
        cells.clear()
        repeat(n) { cells.add(KvCell()) }
        usedCount = 0
    }

    fun getUsed(): Int = usedCount

    /** Reset every cell to its default (empty) state. */
    fun reset() {
        for (cell in cells) {
            cell.pos = -1
            cell.delta = 0
            cell.seqIds.clear()
        }
        usedCount = 0
    }

    fun isEmpty(i: Int): Boolean = cells[i].isEmpty

    fun posGet(i: Int): LlamaPos = cells[i].pos

    fun posSet(i: Int, pos: LlamaPos) {
        val wasEmpty = cells[i].isEmpty
        cells[i].pos = pos
        if (wasEmpty && pos >= 0) usedCount++
    }

    /** Add `shift` to the position; returns `true` if the cell becomes empty. */
    fun posAdd(i: Int, shift: LlamaPos): Boolean {
        cells[i].pos += shift
        cells[i].delta += shift
        if (cells[i].pos < 0) {
            rm(i)
            return true
        }
        return false
    }

    fun posDiv(i: Int, d: Int) {
        val oldPos = cells[i].pos
        cells[i].pos /= d
        cells[i].delta += cells[i].pos - oldPos
    }

    fun posIn(i: Int, p0: LlamaPos, p1: LlamaPos): Boolean {
        val p = cells[i].pos
        return p >= p0 && p < p1
    }

    fun seqHas(i: Int, seqId: LlamaSeqId): Boolean = seqId in cells[i].seqIds

    fun seqAdd(i: Int, seqId: LlamaSeqId) {
        val wasEmpty = cells[i].isEmpty
        cells[i].seqIds.add(seqId)
        if (wasEmpty) usedCount++
    }

    /** Remove a single sequence from cell `i`. Returns `true` if the cell became empty. */
    fun seqRm(i: Int, seqId: LlamaSeqId): Boolean {
        cells[i].seqIds.remove(seqId)
        if (cells[i].isEmpty) {
            cells[i].pos = -1
            cells[i].delta = 0
            usedCount--
            return true
        }
        return false
    }

    /** Keep only `seqId` in cell `i`; returns `true` if cell was freed. */
    fun seqKeep(i: Int, seqId: LlamaSeqId): Boolean {
        if (!cells[i].seqIds.contains(seqId)) {
            if (cells[i].seqIds.isNotEmpty()) {
                rm(i)
                return true
            }
        } else {
            cells[i].seqIds.retainAll(setOf(seqId))
        }
        return false
    }

    fun seqCount(i: Int): Int = cells[i].seqIds.size

    fun seqGet(i: Int): LlamaSeqId = cells[i].seqIds.first()

    /** Remove all sequences and position data from cell `i`. */
    fun rm(i: Int) {
        if (cells[i].seqIds.isNotEmpty()) usedCount--
        cells[i].pos = -1
        cells[i].delta = 0
        cells[i].seqIds.clear()
    }

    fun getShift(i: Int): LlamaPos = cells[i].delta

    fun resetShift() {
        for (cell in cells) cell.delta = 0
    }

    /** Minimum position among cells belonging to [seqId], or -1 if none. */
    fun seqPosMin(seqId: LlamaSeqId): LlamaPos {
        var min = Int.MAX_VALUE
        for (cell in cells) {
            if (seqId in cell.seqIds && cell.pos in 0 until min) {
                min = cell.pos
            }
        }
        return if (min == Int.MAX_VALUE) -1 else min
    }

    /** Maximum position among cells belonging to [seqId], or -1 if none. */
    fun seqPosMax(seqId: LlamaSeqId): LlamaPos {
        var max = -1
        for (cell in cells) {
            if (seqId in cell.seqIds && cell.pos > max) {
                max = cell.pos
            }
        }
        return max
    }

    /** One past the highest used cell index. */
    fun usedMaxP1(): Int {
        for (i in cells.indices.reversed()) {
            if (!cells[i].isEmpty) return i + 1
        }
        return 0
    }
}

// ---------------------------------------------------------------------------
// KvLayer – per-layer K/V tensor references
// ---------------------------------------------------------------------------

/**
 * Holds the K and V tensors for a single transformer layer inside the cache.
 *
 * When the cache uses multiple streams (one per unique sequence), [kStream]
 * and [vStream] provide 2-D views into the base [k]/[v] tensors – one per
 * stream.
 *
 * @property il  The model layer index this entry corresponds to.
 * @property k   The key tensor for this layer (shape: [n_embd_k_gqa, kv_size, n_stream]).
 * @property v   The value tensor for this layer (shape depends on v_trans setting).
 * @property kStream  Per-stream 2-D views into [k].
 * @property vStream  Per-stream 2-D views into [v].
 */
data class KvLayer(
    val il: Int,
    var k: GGMLTensor? = null,
    var v: GGMLTensor? = null,
    val kStream: MutableList<GGMLTensor?> = mutableListOf(),
    val vStream: MutableList<GGMLTensor?> = mutableListOf()
)

// ---------------------------------------------------------------------------
// StreamCopyInfo – pending cross-stream buffer copies
// ---------------------------------------------------------------------------

/**
 * Records pending stream-to-stream buffer copies that will be executed
 * during the next [KVCache.update] call.
 *
 * Cross-stream sequence copies (`seq_cp` between different streams)
 * require actual buffer data to be copied, which is deferred until the
 * next graph computation.
 */
data class StreamCopyInfo(
    val ssrc: MutableList<Int> = mutableListOf(),
    val sdst: MutableList<Int> = mutableListOf()
) {
    fun empty(): Boolean {
        require(ssrc.size == sdst.size)
        return ssrc.isEmpty()
    }

    fun clear() {
        ssrc.clear()
        sdst.clear()
    }
}

// ---------------------------------------------------------------------------
// SlotInfo – describes where a micro-batch lands in the cache
// ---------------------------------------------------------------------------

/**
 * For each micro-batch, [SlotInfo] describes where its tokens should be
 * placed inside the KV cache cells.
 *
 * In the single-stream (unified) case there is exactly one entry in [idxs];
 * with per-sequence streams there is one entry per stream spanned by the
 * micro-batch.
 *
 * Mirrors `llama_kv_cache::slot_info` in C++.
 *
 * @property s0   Lowest stream index used by this slot.
 * @property s1   Highest stream index used by this slot.
 * @property strm Per-stream sequence IDs, length = s1 − s0 + 1.
 * @property idxs Per-stream cell index vectors (one vector per stream).
 */
data class SlotInfo(
    var s0: Int = LLAMA_MAX_SEQ,
    var s1: Int = 0,
    val strm: MutableList<LlamaSeqId> = mutableListOf(),
    val idxs: MutableList<MutableList<Int>> = mutableListOf()
) {
    fun head(): Int {
        require(idxs.size == 1)
        require(idxs[0].isNotEmpty())
        return idxs[0][0]
    }

    fun resize(n: Int) {
        strm.clear()
        repeat(n) { strm.add(0) }
        idxs.clear()
        repeat(n) { idxs.add(mutableListOf()) }
    }

    fun size(): Int {
        require(idxs.size == strm.size)
        require(idxs.isNotEmpty())
        return idxs[0].size
    }

    fun nStream(): Int = strm.size

    fun empty(): Boolean = idxs.isEmpty()

    fun clear() { idxs.clear() }

    /** Check if indices are contiguous starting from [head]. */
    fun isContiguous(): Boolean {
        if (idxs.isEmpty() || idxs[0].isEmpty()) return true
        if (idxs.size > 1) return false
        val h = idxs[0][0]
        for (i in idxs[0].indices) {
            if (idxs[0][i] != h + i) return false
        }
        return true
    }
}

// ---------------------------------------------------------------------------
// KVCache – main cache structure (port of llama_kv_cache)
// ---------------------------------------------------------------------------

/**
 * Key-Value cache for efficient transformer inference.
 *
 * This is the Kotlin port of `llama_kv_cache` from llama.cpp.  It stores
 * per-layer K and V tensors, manages cell metadata for sequence tracking,
 * and supports operations like slot finding, sequence removal/copy/keep,
 * position shifting, and defragmentation.
 *
 * The cache can operate in two modes:
 * - **Unified** (`nStream == 1`): all sequences share a single ring buffer.
 * - **Per-sequence** (`nStream == nSeqMax`): each sequence gets its own stream.
 *
 * ## Original (preserved) simple API
 * The constructor that takes `(maxSequenceLength, numHeads, headDim)` retains
 * the earlier allocator-first interface for basic single-batch usage.
 *
 * @property kvSize   Total number of KV cells in each stream.
 * @property nSeqMax  Maximum number of concurrent sequences.
 * @property nPad     Required padding for the cache size.
 * @property nSwa     Sliding-window size (0 = full context).
 * @property swaType  Sliding-window attention type.
 * @property typeK    Data type of the K cache tensors.
 * @property typeV    Data type of the V cache tensors.
 * @property vTrans   Whether the V cache is stored transposed.
 */
class KVCache private constructor(
    val kvSize: Int,
    val nSeqMax: Int,
    private val nStream: Int,
    private val nPad: Int,
    val nSwa: Int,
    val swaType: LlamaSwaType,
    val typeK: GGMLType,
    val typeV: GGMLType,
    val vTrans: Boolean
) {
    // ── per-stream cell arrays & heads ──────────────────────────────────
    private val vHeads = IntArray(nStream)
    private val vCells = Array(nStream) { KvCells(kvSize) }
    private val seqToStream = IntArray(if (nStream > 1) nStream else LLAMA_MAX_SEQ)

    // ── per-layer tensor storage ────────────────────────────────────────
    private val layers = mutableListOf<KvLayer>()
    private val mapLayerIds = mutableMapOf<Int, Int>()

    // ── pending stream copies ───────────────────────────────────────────
    private val scInfo = StreamCopyInfo()

    // ── attention rotation flags (populated during init) ────────────────
    private var attnRotK = false
    private var attnRotV = false
    private var nEmbdHeadKAll: Int = 0
    private var nEmbdHeadVAll: Int = 0

    // ── debug level (env LLAMA_KV_CACHE_DEBUG) ──────────────────────────
    private var debug = 0

    init {
        require(kvSize % nPad == 0) { "kvSize ($kvSize) must be divisible by nPad ($nPad)" }
        require(nStream == 1 || nStream == nSeqMax) { "nStream must be 1 or nSeqMax" }

        // initialise stream mapping
        if (nStream > 1) {
            for (s in 0 until nStream) seqToStream[s] = s
        }
        // else: all sequences map to stream 0 (default zero-filled)
    }

    // ── simple constructor preserved from original KVCache ──────────────

    /**
     * Simplified constructor for single-batch usage (legacy API).
     *
     * Creates a unified (single-stream) cache with F32 key/value types,
     * no sliding window, and padding of 1.
     */
    constructor(
        maxSequenceLength: Int,
        numHeads: Int,
        headDim: Int
    ) : this(
        kvSize = maxSequenceLength,
        nSeqMax = 1,
        nStream = 1,
        nPad = 1,
        nSwa = 0,
        swaType = LlamaSwaType.NONE,
        typeK = GGMLType.F32,
        typeV = GGMLType.F32,
        vTrans = true
    ) {
        _legacyNumHeads = numHeads
        _legacyHeadDim = headDim
    }

    // ── legacy API fields & methods (preserved from original file) ──────

    private var _legacyNumHeads: Int = 0
    private var _legacyHeadDim: Int = 0
    private var _legacyKeys: GGMLTensor? = null
    private var _legacyValues: GGMLTensor? = null
    private var _legacyCurrentLength = 0

    /** Legacy: maximum sequence length (alias for [kvSize]). */
    val maxSequenceLength: Int get() = kvSize
    /** Legacy: number of attention heads. */
    val numHeads: Int get() = _legacyNumHeads
    /** Legacy: per-head embedding dimension. */
    val headDim: Int get() = _legacyHeadDim

    fun initialize(graphAllocator: GGMLGraphAllocator) {
        _legacyKeys = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = _legacyHeadDim.toLong()
            ne[1] = _legacyNumHeads.toLong()
            ne[2] = kvSize.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }.also { graphAllocator.allocateTensor(it) }

        _legacyValues = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = _legacyHeadDim.toLong()
            ne[1] = _legacyNumHeads.toLong()
            ne[2] = kvSize.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }.also { graphAllocator.allocateTensor(it) }

        _legacyCurrentLength = 0
    }

    fun updateKey(graphAllocator: GGMLGraphAllocator, newKey: GGMLTensor): GGMLTensor {
        val cacheKey = _legacyKeys ?: throw IllegalStateException("KVCache not initialized")
        val newSeqLen = newKey.ne[2].toInt()
        require(_legacyCurrentLength + newSeqLen <= kvSize) { "Sequence length would exceed maximum" }

        for (h in 0 until _legacyNumHeads) {
            for (s in 0 until newSeqLen) {
                for (d in 0 until _legacyHeadDim) {
                    val v = newKey.getFloat(graphAllocator, d, h, s, 0)
                    cacheKey.setFloat(graphAllocator, v, d, h, _legacyCurrentLength + s, 0)
                }
            }
        }
        val out = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = _legacyHeadDim.toLong()
            ne[1] = _legacyNumHeads.toLong()
            ne[2] = (_legacyCurrentLength + newSeqLen).toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        graphAllocator.allocateTensor(out)
        for (h in 0 until _legacyNumHeads) {
            for (s in 0 until (_legacyCurrentLength + newSeqLen)) {
                for (d in 0 until _legacyHeadDim) {
                    val v = cacheKey.getFloat(graphAllocator, d, h, s, 0)
                    out.setFloat(graphAllocator, v, d, h, s, 0)
                }
            }
        }
        _legacyCurrentLength += newSeqLen
        return out
    }

    fun updateValue(graphAllocator: GGMLGraphAllocator, newValue: GGMLTensor): GGMLTensor {
        val cacheVal = _legacyValues ?: throw IllegalStateException("KVCache not initialized")
        val newSeqLen = newValue.ne[2].toInt()
        require(_legacyCurrentLength >= newSeqLen) { "updateKey must be called before updateValue in a step" }

        val start = _legacyCurrentLength - newSeqLen
        for (h in 0 until _legacyNumHeads) {
            for (s in 0 until newSeqLen) {
                for (d in 0 until _legacyHeadDim) {
                    val v = newValue.getFloat(graphAllocator, d, h, s, 0)
                    cacheVal.setFloat(graphAllocator, v, d, h, start + s, 0)
                }
            }
        }

        val out = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = _legacyHeadDim.toLong()
            ne[1] = _legacyNumHeads.toLong()
            ne[2] = _legacyCurrentLength.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        graphAllocator.allocateTensor(out)
        for (h in 0 until _legacyNumHeads) {
            for (s in 0 until _legacyCurrentLength) {
                for (d in 0 until _legacyHeadDim) {
                    val v = cacheVal.getFloat(graphAllocator, d, h, s, 0)
                    out.setFloat(graphAllocator, v, d, h, s, 0)
                }
            }
        }
        return out
    }

    fun getCurrentLength(): Int = _legacyCurrentLength

    // ====================================================================
    // Full llama_kv_cache API (ported from C++)
    // ====================================================================

    companion object {
        /**
         * Create a fully-configured KV cache matching the C++ `llama_kv_cache` constructor.
         *
         * @param typeK     Data type for key tensors.
         * @param typeV     Data type for value tensors.
         * @param vTrans    Whether value tensors are stored transposed.
         * @param unified   If `true`, all sequences share one stream; otherwise one stream per sequence.
         * @param kvSize    Number of cache cells per stream.
         * @param nSeqMax   Maximum number of concurrent sequences.
         * @param nPad      Padding requirement (kvSize must be divisible by this).
         * @param nSwa      Sliding-window size (0 = disabled).
         * @param swaType   Type of sliding-window attention.
         */
        fun create(
            typeK: GGMLType = GGMLType.F32,
            typeV: GGMLType = GGMLType.F32,
            vTrans: Boolean = true,
            unified: Boolean = true,
            kvSize: Int = 512,
            nSeqMax: Int = 1,
            nPad: Int = 1,
            nSwa: Int = 0,
            swaType: LlamaSwaType = LlamaSwaType.NONE
        ): KVCache {
            return KVCache(
                kvSize = kvSize,
                nSeqMax = nSeqMax,
                nStream = if (unified) 1 else nSeqMax,
                nPad = nPad,
                nSwa = nSwa,
                swaType = swaType,
                typeK = typeK,
                typeV = typeV,
                vTrans = vTrans
            )
        }
    }

    // ── size / query accessors ──────────────────────────────────────────

    fun getSize(): Int = kvSize
    fun getNStream(): Int = nStream

    /** `true` when any cell has a non-zero delta (pending K-shift). */
    fun getHasShift(): Boolean {
        for (s in 0 until nStream) {
            for (i in 0 until vCells[s].size()) {
                if (vCells[s].getShift(i) != 0) return true
            }
        }
        return false
    }

    /** Whether K-shift (RoPE rotation of cached keys) is supported. */
    fun getCanShift(): Boolean = !attnRotK

    // ── clear ───────────────────────────────────────────────────────────

    /**
     * Reset all cell metadata (and optionally zero the backing data).
     *
     * @param data If `true`, also zero the tensor buffer contents.
     */
    fun clear(data: Boolean = false) {
        for (s in 0 until nStream) {
            vCells[s].reset()
            vHeads[s] = 0
        }
        _legacyCurrentLength = 0
        // NOTE: buffer zeroing requires backend buffer access – deferred
        if (data) {
            TODO("port from llama-kv-cache.cpp: clear buffer data via backend_buffer_clear")
        }
    }

    // ── sequence management ─────────────────────────────────────────────

    /**
     * Remove positions in range `[p0, p1)` from the given sequence (or all
     * sequences if [seqId] is -1).
     *
     * @return `true` always (matching C++ signature).
     */
    fun seqRm(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos): Boolean {
        var pp0 = if (p0 < 0) 0 else p0
        var pp1 = if (p1 < 0) Int.MAX_VALUE else p1

        if (seqId >= 0) {
            require(seqId < seqToStream.size) { "seqId out of range" }
            val streamIdx = seqToStream[seqId]
            val cells = vCells[streamIdx]
            var newHead = cells.size()

            for (i in 0 until cells.size()) {
                if (!cells.posIn(i, pp0, pp1)) continue
                if (cells.seqHas(i, seqId) && cells.seqRm(i, seqId)) {
                    if (newHead == cells.size()) newHead = i
                }
            }
            if (newHead != cells.size() && newHead < vHeads[streamIdx]) {
                vHeads[streamIdx] = newHead
            }
        } else {
            for (s in 0 until nStream) {
                val cells = vCells[s]
                var newHead = cells.size()
                for (i in 0 until cells.size()) {
                    if (!cells.posIn(i, pp0, pp1)) continue
                    cells.rm(i)
                    if (newHead == cells.size()) newHead = i
                }
                if (newHead != cells.size() && newHead < vHeads[s]) {
                    vHeads[s] = newHead
                }
            }
        }
        return true
    }

    /**
     * Copy sequence [seqIdSrc] to [seqIdDst] for positions in `[p0, p1)`.
     *
     * Within the same stream this only updates cell metadata.  Across
     * streams it enqueues a buffer copy that executes during the next
     * [update] call.
     */
    fun seqCp(seqIdSrc: LlamaSeqId, seqIdDst: LlamaSeqId, p0: LlamaPos, p1: LlamaPos) {
        require(seqIdSrc >= 0 && seqIdSrc < seqToStream.size) { "seqIdSrc out of range" }
        require(seqIdDst >= 0 && seqIdDst < seqToStream.size) { "seqIdDst out of range" }

        val s0 = seqToStream[seqIdSrc]
        val s1 = seqToStream[seqIdDst]

        if (s0 == s1) {
            if (seqIdSrc == seqIdDst) return
            val cells = vCells[s0]
            val pp0 = if (p0 < 0) 0 else p0
            val pp1 = if (p1 < 0) Int.MAX_VALUE else p1
            for (i in 0 until cells.size()) {
                if (!cells.posIn(i, pp0, pp1)) continue
                if (cells.seqHas(i, seqIdSrc)) {
                    cells.seqAdd(i, seqIdDst)
                }
            }
            return
        }

        // Cross-stream copy – enqueue for next update()
        scInfo.ssrc.add(s0)
        scInfo.sdst.add(s1)

        vCells[s1].reset()
        for (i in 0 until vCells[s0].size()) {
            if (vCells[s0].seqHas(i, seqIdSrc)) {
                val pos = vCells[s0].posGet(i)
                vCells[s1].posSet(i, pos)
                vCells[s1].seqAdd(i, seqIdDst)
            }
        }
        vHeads[s1] = vHeads[s0]
    }

    /**
     * Keep only [seqId] in every cell; remove all other sequences.
     */
    fun seqKeep(seqId: LlamaSeqId) {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        val streamIdx = seqToStream[seqId]
        val cells = vCells[streamIdx]
        var newHead = cells.size()
        for (i in 0 until cells.size()) {
            if (cells.seqKeep(i, seqId)) {
                if (newHead == cells.size()) newHead = i
            }
        }
        if (newHead != cells.size() && newHead < vHeads[streamIdx]) {
            vHeads[streamIdx] = newHead
        }
    }

    /**
     * Add [shift] to the position of every cell belonging to [seqId]
     * within `[p0, p1)`.
     */
    fun seqAdd(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, shift: LlamaPos) {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        if (shift == 0) return

        val streamIdx = seqToStream[seqId]
        val cells = vCells[streamIdx]
        val pp0 = if (p0 < 0) 0 else p0
        val pp1 = if (p1 < 0) Int.MAX_VALUE else p1
        if (pp0 == pp1) return

        var newHead = cells.size()
        for (i in 0 until cells.size()) {
            if (!cells.posIn(i, pp0, pp1)) continue
            if (cells.seqHas(i, seqId)) {
                if (cells.posAdd(i, shift)) {
                    if (newHead == cells.size()) newHead = i
                }
            }
        }
        vHeads[streamIdx] = if (newHead != cells.size()) newHead else 0
    }

    /**
     * Divide the position of cells belonging to [seqId] within `[p0, p1)` by [d].
     */
    fun seqDiv(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, d: Int) {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        if (d == 1) return

        val streamIdx = seqToStream[seqId]
        val cells = vCells[streamIdx]
        val pp0 = if (p0 < 0) 0 else p0
        val pp1 = if (p1 < 0) Int.MAX_VALUE else p1
        if (pp0 == pp1) return

        for (i in 0 until cells.size()) {
            if (!cells.posIn(i, pp0, pp1)) continue
            if (cells.seqHas(i, seqId)) {
                cells.posDiv(i, d)
            }
        }
    }

    /** Minimum position for [seqId], or -1 if no cells belong to it. */
    fun seqPosMin(seqId: LlamaSeqId): LlamaPos {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        return vCells[seqToStream[seqId]].seqPosMin(seqId)
    }

    /** Maximum position for [seqId], or -1 if no cells belong to it. */
    fun seqPosMax(seqId: LlamaSeqId): LlamaPos {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        return vCells[seqToStream[seqId]].seqPosMax(seqId)
    }

    // ── slot finding (port of llama_kv_cache::find_slot) ────────────────

    /**
     * Find a contiguous (or non-contiguous) run of free cells that can
     * hold `nTokens` tokens.
     *
     * @param nTokens  Number of tokens to place.
     * @param cont     If `true`, require a contiguous block.
     * @return A [SlotInfo] describing the placement, or an empty [SlotInfo] on failure.
     */
    fun findSlot(nTokens: Int, cont: Boolean = false): SlotInfo {
        TODO("port from llama-kv-cache.cpp: find_slot – ring-buffer scan with SWA awareness")
    }

    // ── update (apply pending shifts and stream copies) ─────────────────

    /**
     * Apply pending K-shift and stream copies.
     *
     * @param doShift   Whether to perform K-shift (RoPE rotation update).
     * @param scInfoArg Pending stream copy info (consumed).
     * @return `true` if any work was done.
     */
    fun update(doShift: Boolean, scInfoArg: StreamCopyInfo = StreamCopyInfo()): Boolean {
        TODO("port from llama-kv-cache.cpp: update – buffer copy + K-shift graph build/compute")
    }

    // ── defrag ──────────────────────────────────────────────────────────

    /**
     * Defragment the cache by moving occupied cells toward the front,
     * closing gaps left by removed sequences.
     */
    fun defrag() {
        TODO("port from llama-kv-cache.cpp: defrag – compact cells and update tensor data")
    }

    // ── state serialisation ─────────────────────────────────────────────

    /**
     * Write cache state to the given writer.
     *
     * @param seqId  If >= 0, write only cells belonging to this sequence;
     *               if -1, write all cells.
     */
    fun stateWrite(seqId: LlamaSeqId = -1) {
        TODO("port from llama-kv-cache.cpp: state_write – serialise cell metadata + tensor data")
    }

    /**
     * Read cache state from the given reader.
     *
     * @param seqId  If >= 0, load only into cells for this sequence;
     *               if -1, load all cells.
     */
    fun stateRead(seqId: LlamaSeqId = -1) {
        TODO("port from llama-kv-cache.cpp: state_read – deserialise cell metadata + tensor data")
    }

    // ── internal helpers ────────────────────────────────────────────────

    /** Total bytes used by K cache tensors across all layers. */
    fun sizeKBytes(): Long {
        TODO("port from llama-kv-cache.cpp: size_k_bytes")
    }

    /** Total bytes used by V cache tensors across all layers. */
    fun sizeVBytes(): Long {
        TODO("port from llama-kv-cache.cpp: size_v_bytes")
    }

    /** Sum of K + V byte sizes. */
    fun totalSize(): Long = sizeKBytes() + sizeVBytes()
}

// ---------------------------------------------------------------------------
// KVCacheContext – mirrors llama_kv_cache_context
// ---------------------------------------------------------------------------

/**
 * Context object created by [KVCache] for a single batch / update operation.
 *
 * Mirrors `llama_kv_cache_context` in C++.  Holds the slot info and
 * micro-batches needed to drive graph construction and input-tensor filling.
 *
 * @property status   Current status of this context.
 * @property kvCache  The parent KV cache (may be null for error contexts).
 */
class KVCacheContext(
    val status: LlamaMemoryStatus,
    private val kvCache: KVCache? = null
) {
    private var iCur: Int = 0
    private val sinfos = mutableListOf<SlotInfo>()
    private var doShift: Boolean = false
    private var scInfo: StreamCopyInfo = StreamCopyInfo()

    /** A heuristic: how many KV cells to attend to in the current step. */
    var nKv: Int = 0
        private set

    /** Advance to the next micro-batch. Returns `false` when done. */
    fun next(): Boolean {
        if (iCur >= sinfos.size) return false
        iCur++
        return iCur < sinfos.size
    }

    /** Apply the current micro-batch's slot info to the cache. */
    fun apply(): Boolean {
        TODO("port from llama-kv-cache.cpp: llama_kv_cache_context::apply")
    }

    fun getNKv(): Int = nKv
    fun typeK(): GGMLType = kvCache?.typeK ?: GGMLType.F32
    fun typeV(): GGMLType = kvCache?.typeV ?: GGMLType.F32
}