package ai.solace.llamakotlin.backend.klangnative

import ai.solace.klangnative.fp.CFloat32
import ai.solace.klangnative.mem.CIntVar
import ai.solace.klangnative.mem.GlobalHeap
import ai.solace.klangnative.mem.KAligned
import kotlin.math.max

/**
 * Tensor storage using KLangNative's GlobalHeap for C-aligned buffers with bit-exact
 * IEEE-754 semantics. This provides the foundation for porting GGML quantization code
 * to pure Kotlin while maintaining exact C parity.
 *
 * Based on the reference implementation from ember-ml-kotlin with adaptations for
 * GGML's quantization requirements.
 */
object KLangNativeHeapTensorStorage {

    data class Buffer(val ptr: Int, val sizeBytes: Int) {
        fun free() = KAligned.alignedFree(ptr)
    }

    /**
     * Allocate an aligned buffer for `count` float32 elements (4 bytes each).
     * Alignment defaults to 32 bytes for SIMD compatibility.
     */
    fun mallocFloat32(count: Int, alignment: Int = 32): Buffer {
        require(count >= 0) { "count must be non-negative" }
        val bytes = count * 4
        val ptr = KAligned.alignedCalloc(alignment, max(bytes, 1))
        return Buffer(ptr, bytes)
    }

    /**
     * Store Kotlin floats into the buffer as CFloat32 bit patterns.
     * Element-by-element operation for correctness.
     */
    fun writeFloat32(buffer: Buffer, data: FloatArray) {
        require(data.size * 4 <= buffer.sizeBytes) { "buffer too small" }
        var offset = buffer.ptr
        data.forEach { f ->
            val bits = CFloat32.fromFloat(f).toBits()
            CIntVar(offset).value = bits
            offset += 4
        }
    }

    /**
     * Read back into a Kotlin FloatArray.
     */
    fun readFloat32(buffer: Buffer, count: Int): FloatArray {
        require(count * 4 <= buffer.sizeBytes) { "buffer too small" }
        val out = FloatArray(count)
        var offset = buffer.ptr
        for (i in 0 until count) {
            val bits = CIntVar(offset).value
            out[i] = Float.fromBits(bits)
            offset += 4
        }
        return out
    }

    /**
     * Bulk copy variant using IntArray packing for better performance on large tensors.
     * Use this for initial tensor loading and final results extraction.
     */
    fun writeFloat32Bulk(buffer: Buffer, data: FloatArray) {
        require(data.size * 4 <= buffer.sizeBytes) { "buffer too small" }
        val packed = IntArray(data.size) { idx -> CFloat32.fromFloat(data[idx]).toBits() }
        GlobalHeap.copyFromIntArray(buffer.ptr, packed, packed.size)
    }

    /**
     * Bulk read variant matching [writeFloat32Bulk].
     */
    fun readFloat32Bulk(buffer: Buffer, count: Int): FloatArray {
        require(count * 4 <= buffer.sizeBytes) { "buffer too small" }
        val ints = IntArray(count)
        GlobalHeap.copyToIntArray(buffer.ptr, ints, count)
        return FloatArray(count) { i -> Float.fromBits(ints[i]) }
    }

    /**
     * Fast path when callers already have packed IEEE754 bits (zero-copy).
     * Use this for pre-quantized data or when working directly with bit patterns.
     */
    fun writeFloat32Packed(buffer: Buffer, packedBits: IntArray) {
        require(packedBits.size * 4 <= buffer.sizeBytes) { "buffer too small" }
        GlobalHeap.copyFromIntArray(buffer.ptr, packedBits, packedBits.size)
    }

    /**
     * Read raw bits into an IntArray without decoding to Float.
     */
    fun readFloat32Packed(buffer: Buffer, count: Int): IntArray {
        require(count * 4 <= buffer.sizeBytes) { "buffer too small" }
        val ints = IntArray(count)
        GlobalHeap.copyToIntArray(buffer.ptr, ints, count)
        return ints
    }

    /**
     * Allocate buffer for bytes (used for quantized block storage).
     */
    fun mallocBytes(count: Int, alignment: Int = 32): Buffer {
        require(count >= 0) { "count must be non-negative" }
        val ptr = KAligned.alignedCalloc(alignment, max(count, 1))
        return Buffer(ptr, count)
    }

    /**
     * Write bytes directly to buffer.
     */
    fun writeBytes(buffer: Buffer, data: ByteArray, offset: Int = 0) {
        require(data.size - offset <= buffer.sizeBytes) { "buffer too small" }
        var dstOffset = buffer.ptr
        for (i in offset until data.size) {
            GlobalHeap.sb(dstOffset, data[i])
            dstOffset += 1
        }
    }

    /**
     * Read bytes from buffer.
     */
    fun readBytes(buffer: Buffer, count: Int): ByteArray {
        require(count <= buffer.sizeBytes) { "buffer too small" }
        val out = ByteArray(count)
        var srcOffset = buffer.ptr
        for (i in 0 until count) {
            out[i] = GlobalHeap.lb(srcOffset)
            srcOffset += 1
        }
        return out
    }
}
