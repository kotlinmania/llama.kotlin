// port-lint: source ggml/src/ggml-cpu/ops.cpp
package ai.solace.llamakotlin.core

import ai.solace.llamakotlin.core.ByteArrayExtensions.getFloatLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getIntLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getLongLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getShortLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setFloatLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setIntLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setLongLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.setShortLe
import kotlin.math.min

/**
 * Line-by-line transliteration of ops.cpp lines 576–2096 (ADD through CONCAT).
 *
 * Function names use snake_case to match the C++ originals and maximise
 * ast_distance parity.  Public dispatchers are `fun`; type-specific helpers
 * are `private fun`.
 */

// ============================================================================
// Constants
// ============================================================================

/** Cache-line size in floats (64 bytes / 4 bytes per float). */
private const val CACHE_LINE_SIZE_F32 = 16

// ============================================================================
// Vector helper functions — scalar implementations matching C++ vec.h
// ============================================================================

/** dst[i] += src[i]  for i in 0 until n */
private fun ggml_vec_acc_f32(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] += src[srcOff + i]
    }
}

/** dst[i] += v  for i in 0 until n */
private fun ggml_vec_acc1_f32(n: Long, dst: FloatArray, dstOff: Int, v: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] += v
    }
}

/** dst[i] = src0[i] + src1[i]  for i in 0 until n */
private fun ggml_vec_add_f32(n: Long, dst: FloatArray, dstOff: Int, src0: FloatArray, src0Off: Int, src1: FloatArray, src1Off: Int) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = src0[src0Off + i] + src1[src1Off + i]
    }
}

/** dst[i] = src[i] + v  for i in 0 until n */
private fun ggml_vec_add1_f32(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int, v: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = src[srcOff + i] + v
    }
}

/** dst[i] = src[i]  for i in 0 until n */
private fun ggml_vec_cpy_f32(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int) {
    src.copyInto(dst, dstOff, srcOff, srcOff + n.toInt())
}

/** dst[i] = v  for i in 0 until n */
private fun ggml_vec_set_f32(n: Long, dst: FloatArray, dstOff: Int, v: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = v
    }
}

/** *result = sum(src[i])  for i in 0 until n  (double accumulator for ggf variant) */
private fun ggml_vec_sum_f32_ggf(n: Long, src: FloatArray, srcOff: Int): Double {
    var sum = 0.0
    for (i in 0 until n.toInt()) {
        sum += src[srcOff + i].toDouble()
    }
    return sum
}

/** *result = sum(src[i])  for i in 0 until n  (float accumulator) */
private fun ggml_vec_sum_f32(n: Long, src: FloatArray, srcOff: Int): Float {
    var sum = 0f
    for (i in 0 until n.toInt()) {
        sum += src[srcOff + i]
    }
    return sum
}

/** *result = sum(fp16_to_f32(src[i]))  for i in 0 until n */
private fun ggml_vec_sum_f16_ggf(n: Long, data: ByteArray, dataOff: Int): Float {
    var sum = 0f
    for (i in 0 until n.toInt()) {
        val bits = data.getShortLe(dataOff + i * 2)
        sum += ggmlFp16ToFp32(bits.toUShort())
    }
    return sum
}

/** *result = sum(bf16_to_f32(src[i]))  for i in 0 until n */
private fun ggml_vec_sum_bf16_ggf(n: Long, data: ByteArray, dataOff: Int): Float {
    var sum = 0f
    for (i in 0 until n.toInt()) {
        val bits = data.getShortLe(dataOff + i * 2)
        sum += bf16ToF32(GGMLBF16(bits.toUShort()))
    }
    return sum
}

/** dst[i] = prefix_sum(src[0..i])  for i in 0 until n */
private fun ggml_vec_cumsum_f32(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int) {
    if (n <= 0) return
    dst[dstOff] = src[srcOff]
    for (i in 1 until n.toInt()) {
        dst[dstOff + i] = dst[dstOff + i - 1] + src[srcOff + i]
    }
}

/** *result = argmax(src[0..n)) */
private fun ggml_vec_argmax_f32(n: Long, src: FloatArray, srcOff: Int): Int {
    var maxIdx = 0
    var maxVal = src[srcOff]
    for (i in 1 until n.toInt()) {
        if (src[srcOff + i] > maxVal) {
            maxVal = src[srcOff + i]
            maxIdx = i
        }
    }
    return maxIdx
}

// ============================================================================
// ByteArray data access helpers
// ============================================================================

/** Read a float from a ByteArray at byte offset. */
private fun readFloat(data: ByteArray, off: Int): Float = data.getFloatLe(off)

/** Write a float to a ByteArray at byte offset. */
private fun writeFloat(data: ByteArray, off: Int, v: Float) = data.setFloatLe(off, v)

/** Read an int from a ByteArray at byte offset. */
private fun readInt(data: ByteArray, off: Int): Int = data.getIntLe(off)

/** Write an int to a ByteArray at byte offset. */
private fun writeInt(data: ByteArray, off: Int, v: Int) = data.setIntLe(off, v)

/** Read a long from a ByteArray at byte offset. */
private fun readLong(data: ByteArray, off: Int): Long = data.getLongLe(off)

/** Write a long to a ByteArray at byte offset. */
private fun writeLong(data: ByteArray, off: Int, v: Long) = data.setLongLe(off, v)

/** Read a short from a ByteArray at byte offset. */
private fun readShort(data: ByteArray, off: Int): Short = data.getShortLe(off)

/** Write a short to a ByteArray at byte offset. */
private fun writeShort(data: ByteArray, off: Int, v: Short) = data.setShortLe(off, v)

/** Read ne00..ne03 and nb00..nb03 from src0 plus ne10..ne13, nb10..nb13 from src1
 *  plus ne0..ne3, nb0..nb3 from dst.  Returns a data class for clarity. */
private data class BinaryOpLocals(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne10: Long, val ne11: Long, val ne12: Long, val ne13: Long,
    val nb10: Long, val nb11: Long, val nb12: Long, val nb13: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun binaryOpLocals(dst: GGMLTensor): BinaryOpLocals {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    return BinaryOpLocals(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne10 = src1.ne[0], ne11 = src1.ne[1], ne12 = src1.ne[2], ne13 = src1.ne[3],
        nb10 = src1.nb[0].toLong(), nb11 = src1.nb[1].toLong(), nb12 = src1.nb[2].toLong(), nb13 = src1.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

private data class UnaryOpLocals(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun unaryOpLocals(dst: GGMLTensor): UnaryOpLocals {
    val src0 = dst.src[0]!!
    return UnaryOpLocals(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

private data class TernaryOpLocals(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne10: Long, val ne11: Long, val ne12: Long, val ne13: Long,
    val nb10: Long, val nb11: Long, val nb12: Long, val nb13: Long,
    val ne20: Long, val ne21: Long, val ne22: Long, val ne23: Long,
    val nb20: Long, val nb21: Long, val nb22: Long, val nb23: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun ternaryOpLocals(dst: GGMLTensor): TernaryOpLocals {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    val src2 = dst.src[2]!!
    return TernaryOpLocals(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne10 = src1.ne[0], ne11 = src1.ne[1], ne12 = src1.ne[2], ne13 = src1.ne[3],
        nb10 = src1.nb[0].toLong(), nb11 = src1.nb[1].toLong(), nb12 = src1.nb[2].toLong(), nb13 = src1.nb[3].toLong(),
        ne20 = src2.ne[0], ne21 = src2.ne[1], ne22 = src2.ne[2], ne23 = src2.ne[3],
        nb20 = src2.nb[0].toLong(), nb21 = src2.nb[1].toLong(), nb22 = src2.nb[2].toLong(), nb23 = src2.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

// ============================================================================
// Dequantize / quantize dispatch helpers
// ============================================================================

/** Dequantize a row of quantized data to floats (matches ggml_to_float_t). */
private fun dequantize_row(type: GGMLType, src: ByteArray, srcOff: Int, dst: FloatArray, dstOff: Int, k: Long) {
    when (type) {
        GGMLType.Q1_0  -> dequantize_row_q1_0(src, srcOff, dst, dstOff, k)
        GGMLType.Q4_0  -> dequantize_row_q4_0(src, srcOff, dst, dstOff, k)
        GGMLType.Q4_1  -> dequantize_row_q4_1(src, srcOff, dst, dstOff, k)
        GGMLType.Q5_0  -> dequantize_row_q5_0(src, srcOff, dst, dstOff, k)
        GGMLType.Q5_1  -> dequantize_row_q5_1(src, srcOff, dst, dstOff, k)
        GGMLType.Q8_0  -> dequantize_row_q8_0(src, srcOff, dst, dstOff, k)
        GGMLType.Q2_K  -> dequantize_row_q2_K(src, srcOff, dst, dstOff, k)
        GGMLType.Q3_K  -> dequantize_row_q3_K(src, srcOff, dst, dstOff, k)
        GGMLType.Q4_K  -> dequantize_row_q4_K(src, srcOff, dst, dstOff, k)
        GGMLType.Q5_K  -> dequantize_row_q5_K(src, srcOff, dst, dstOff, k)
        GGMLType.Q6_K  -> dequantize_row_q6_K(src, srcOff, dst, dstOff, k)
        GGMLType.TQ1_0 -> dequantize_row_tq1_0(src, srcOff, dst, dstOff, k)
        GGMLType.TQ2_0 -> dequantize_row_tq2_0(src, srcOff, dst, dstOff, k)
        GGMLType.IQ2_XXS -> dequantize_row_iq2_xxs(src, srcOff, dst, dstOff, k)
        GGMLType.IQ2_XS  -> dequantize_row_iq2_xs(src, srcOff, dst, dstOff, k)
        GGMLType.IQ3_XXS -> dequantize_row_iq3_xxs(src, srcOff, dst, dstOff, k)
        GGMLType.IQ4_NL  -> dequantize_row_iq4_nl(src, srcOff, dst, dstOff, k)
        GGMLType.IQ4_XS  -> dequantize_row_iq4_xs(src, srcOff, dst, dstOff, k)
        GGMLType.MXFP4   -> dequantize_row_mxfp4(src, srcOff, dst, dstOff, k)
        GGMLType.NVFP4   -> dequantize_row_nvfp4(src, srcOff, dst, dstOff, k)
        else -> error("dequantize_row: unsupported type $type")
    }
}

/** Quantize a row of floats to quantized data (matches ggml_from_float_t). */
private fun quantize_row(type: GGMLType, src: FloatArray, dst: ByteArray, k: Long) {
    when (type) {
        GGMLType.Q1_0  -> quantize_row_q1_0(src, dst, k)
        GGMLType.Q4_0  -> quantize_row_q4_0(src, dst, k)
        GGMLType.Q4_1  -> quantize_row_q4_1(src, dst, k)
        GGMLType.Q5_0  -> quantize_row_q5_0(src, dst, k)
        GGMLType.Q5_1  -> quantize_row_q5_1(src, dst, k)
        GGMLType.Q8_0  -> quantize_row_q8_0_generic(src, dst, k)
        GGMLType.Q2_K  -> quantize_row_q2_K(src, dst, k)
        GGMLType.Q3_K  -> quantize_row_q3_K(src, dst, k)
        GGMLType.Q4_K  -> quantize_row_q4_K(src, dst, k)
        GGMLType.Q5_K  -> quantize_row_q5_K(src, dst, k)
        GGMLType.Q6_K  -> quantize_row_q6_K(src, dst, k)
        GGMLType.TQ1_0 -> quantize_row_tq1_0(src, dst, k)
        GGMLType.TQ2_0 -> quantize_row_tq2_0(src, dst, k)
        GGMLType.MXFP4 -> quantize_row_mxfp4(src, dst, k)
        GGMLType.NVFP4 -> quantize_row_nvfp4(src, dst, k)
        GGMLType.IQ4_NL -> quantize_row_iq4_nl(src, dst, k)
        // F32 "quantize" is just a memcpy — handled by caller
        GGMLType.F32 -> {
            // no quantize needed; caller should memcpy
        }
        else -> error("quantize_row: unsupported type $type")
    }
}

/** Whether the given type has a from_float quantize function. */
private fun hasQuantizeRow(type: GGMLType): Boolean = when (type) {
    GGMLType.Q1_0, GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1,
    GGMLType.Q8_0, GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K,
    GGMLType.Q6_K, GGMLType.TQ1_0, GGMLType.TQ2_0, GGMLType.MXFP4, GGMLType.NVFP4,
    GGMLType.IQ4_NL -> true
    else -> false
}

// ============================================================================
// Float-array views into ByteArray (read/write rows as FloatArray)
// ============================================================================

/** Read a row of [n] floats from [data] at byte offset [off]. */
private fun readFloatRow(data: ByteArray, off: Int, n: Int): FloatArray {
    val row = FloatArray(n)
    for (i in 0 until n) {
        row[i] = readFloat(data, off + i * 4)
    }
    return row
}

/** Write a row of [n] floats into [data] at byte offset [off]. */
private fun writeFloatRow(data: ByteArray, off: Int, src: FloatArray, srcOff: Int, n: Int) {
    for (i in 0 until n) {
        writeFloat(data, off + i * 4, src[srcOff + i])
    }
}

// ============================================================================
// ggml_compute_forward_add  — ops.cpp line 576
// ============================================================================

// ops.cpp line 578
private fun ggml_compute_forward_add_q_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst))

    val nr = ggmlNrows(src0).toInt()

    val l = binaryOpLocals(dst)

    val ith = params.ith
    val nth = params.nth

    val type = src0.type
    val dtype = dst.type

    // we don't support permuted src0 or src1
    check(l.nb00 == ggmlTypeSize(type).toLong())
    check(l.nb10 == 4L) // sizeof(float)

    // dst cannot be transposed or permuted
    check(l.nb0 <= l.nb1)
    check(l.nb1 <= l.nb2)
    check(l.nb2 <= l.nb3)

    check(ggmlIsQuantized(src0.type))
    check(src1.type == GGMLType.F32)

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val wdata = FloatArray((l.ne00.toInt() + CACHE_LINE_SIZE_F32))

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        // src0 indices
        val i03 = ir / (l.ne02 * l.ne01).toInt()
        val i02 = (ir - i03 * (l.ne02 * l.ne01).toInt()) / l.ne01.toInt()
        val i01 = ir - i03 * (l.ne02 * l.ne01).toInt() - i02 * l.ne01.toInt()

        // src1 and dst are same shape as src0 => same indices
        val i13 = i03; val i12 = i02; val i11 = i01
        val i3 = i03; val i2 = i02; val i1 = i01

        val src0RowOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
        val src1RowOff = (i11 * l.nb11 + i12 * l.nb12 + i13 * l.nb13).toInt()
        val dstRowOff  = (i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()

        check(l.ne00 % 32 == 0L)

        // unquantize row from src0 to temp buffer
        dequantize_row(type, src0Data, src0RowOff, wdata, 0, l.ne00)
        // add src1
        val src1Row = readFloatRow(src1Data, src1RowOff, l.ne00.toInt())
        ggml_vec_acc_f32(l.ne00, wdata, 0, src1Row, 0)
        // quantize row to dst
        if (hasQuantizeRow(dtype)) {
            val tmpDst = ByteArray(ggmlRowSize(dtype, l.ne00).toInt())
            quantize_row(dtype, wdata, tmpDst, l.ne00)
            tmpDst.copyInto(dstData, dstRowOff, 0, tmpDst.size)
        } else {
            // memcpy(dst_row, wdata, ne0*nb0)
            writeFloatRow(dstData, dstRowOff, wdata, 0, l.ne0.toInt())
        }
    }
}

// ops.cpp line 654
fun ggml_compute_forward_add(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F32,
        GGMLType.F16,
        GGMLType.BF16 -> {
            ggml_compute_forward_add_non_quantized(params, dst)
        }
        GGMLType.Q1_0,
        GGMLType.Q4_0, GGMLType.Q4_1,
        GGMLType.Q5_0, GGMLType.Q5_1,
        GGMLType.Q8_0,
        GGMLType.MXFP4, GGMLType.NVFP4,
        GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
        GGMLType.TQ1_0, GGMLType.TQ2_0,
        GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ3_XXS,
        GGMLType.IQ1_S, GGMLType.IQ1_M,
        GGMLType.IQ4_NL, GGMLType.IQ4_XS,
        GGMLType.IQ3_S, GGMLType.IQ2_S -> {
            ggml_compute_forward_add_q_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_add_non_quantized — binary-ops.cpp line 140
// ============================================================================

// Transliteration of binary_op<op_add> from binary-ops.cpp.
// This handles the non-quantized cases: F32/F32, F16/F16, BF16/BF16, mixed.
fun ggml_compute_forward_add_non_quantized(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggmlCanRepeat(src1, src0) && ggml_are_same_shape(src0, dst))

    val l = binaryOpLocals(dst)

    val (ir0, ir1) = getThreadRange(params, src0)
    val isContiguousRows = ggmlIsContiguousRows(src1)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i03 = ir / (l.ne02 * l.ne01)
        val i02 = (ir - i03 * l.ne02 * l.ne01) / l.ne01
        val i01 = ir - i03 * l.ne02 * l.ne01 - i02 * l.ne01

        val i13 = i03 % l.ne13
        val i12 = i02 % l.ne12
        val i11 = i01 % l.ne11

        val dstOff  = (i03 * l.nb3 + i02 * l.nb2 + i01 * l.nb1).toInt()
        val src0Off = (i03 * l.nb03 + i02 * l.nb02 + i01 * l.nb01).toInt()
        val src1Off = (i13 * l.nb13 + i12 * l.nb12 + i11 * l.nb11).toInt()

        if (isContiguousRows) {
            val nr0 = l.ne00 / l.ne10
            for (r in 0 until nr0) {
                val rOff = (r * l.ne10).toInt()
                // Apply add element-wise based on types
                addRowTyped(src0.type, src1.type, dst.type,
                    dstData, dstOff + rOff * dst.type.sizeBytes,
                    src0Data, src0Off + rOff * src0.type.sizeBytes,
                    src1Data, src1Off,
                    l.ne10)
            }
        } else {
            addRowNonContiguous(src0.type, src1.type, dst.type,
                dstData, dstOff,
                src0Data, src0Off,
                src1Data, src1Off,
                l.ne0, l.ne10, l.nb10)
        }
    }
}

/** Element-wise add for contiguous rows with type dispatch. */
private fun addRowTyped(
    src0Type: GGMLType, src1Type: GGMLType, dstType: GGMLType,
    dstData: ByteArray, dstOff: Int,
    src0Data: ByteArray, src0Off: Int,
    src1Data: ByteArray, src1Off: Int,
    n: Long
) {
    val ni = n.toInt()
    // All F32
    if (src0Type == GGMLType.F32 && src1Type == GGMLType.F32 && dstType == GGMLType.F32) {
        for (i in 0 until ni) {
            val a = readFloat(src0Data, src0Off + i * 4)
            val b = readFloat(src1Data, src1Off + i * 4)
            writeFloat(dstData, dstOff + i * 4, a + b)
        }
        return
    }
    // All F16
    if (src0Type == GGMLType.F16 && src1Type == GGMLType.F16 && dstType == GGMLType.F16) {
        for (i in 0 until ni) {
            val a = ggmlFp16ToFp32(readShort(src0Data, src0Off + i * 2).toUShort())
            val b = ggmlFp16ToFp32(readShort(src1Data, src1Off + i * 2).toUShort())
            writeShort(dstData, dstOff + i * 2, ggmlFp32ToFp16(a + b).toShort())
        }
        return
    }
    // All BF16
    if (src0Type == GGMLType.BF16 && src1Type == GGMLType.BF16 && dstType == GGMLType.BF16) {
        for (i in 0 until ni) {
            val a = bf16ToF32(GGMLBF16(readShort(src0Data, src0Off + i * 2).toUShort()))
            val b = bf16ToF32(GGMLBF16(readShort(src1Data, src1Off + i * 2).toUShort()))
            val r = f32ToBf16(a + b)
            writeShort(dstData, dstOff + i * 2, r.bits.toShort())
        }
        return
    }
    // BF16 src0, F32 src1, BF16 dst
    if (src0Type == GGMLType.BF16 && src1Type == GGMLType.F32 && dstType == GGMLType.BF16) {
        for (i in 0 until ni) {
            val a = bf16ToF32(GGMLBF16(readShort(src0Data, src0Off + i * 2).toUShort()))
            val b = readFloat(src1Data, src1Off + i * 4)
            val r = f32ToBf16(a + b)
            writeShort(dstData, dstOff + i * 2, r.bits.toShort())
        }
        return
    }
    // BF16 src0, F32 src1, F32 dst
    if (src0Type == GGMLType.BF16 && src1Type == GGMLType.F32 && dstType == GGMLType.F32) {
        for (i in 0 until ni) {
            val a = bf16ToF32(GGMLBF16(readShort(src0Data, src0Off + i * 2).toUShort()))
            val b = readFloat(src1Data, src1Off + i * 4)
            writeFloat(dstData, dstOff + i * 4, a + b)
        }
        return
    }
    // F16 src0, F32 src1, F16 dst
    if (src0Type == GGMLType.F16 && src1Type == GGMLType.F32 && dstType == GGMLType.F16) {
        for (i in 0 until ni) {
            val a = ggmlFp16ToFp32(readShort(src0Data, src0Off + i * 2).toUShort())
            val b = readFloat(src1Data, src1Off + i * 4)
            writeShort(dstData, dstOff + i * 2, ggmlFp32ToFp16(a + b).toShort())
        }
        return
    }
    // F16 src0, F32 src1, F32 dst
    if (src0Type == GGMLType.F16 && src1Type == GGMLType.F32 && dstType == GGMLType.F32) {
        for (i in 0 until ni) {
            val a = ggmlFp16ToFp32(readShort(src0Data, src0Off + i * 2).toUShort())
            val b = readFloat(src1Data, src1Off + i * 4)
            writeFloat(dstData, dstOff + i * 4, a + b)
        }
        return
    }
    error("ggml_compute_forward_add_non_quantized: unsupported types: dst=$dstType, src0=$src0Type, src1=$src1Type")
}

/** Element-wise add for non-contiguous rows. */
private fun addRowNonContiguous(
    src0Type: GGMLType, src1Type: GGMLType, dstType: GGMLType,
    dstData: ByteArray, dstOff: Int,
    src0Data: ByteArray, src0Off: Int,
    src1Data: ByteArray, src1Off: Int,
    ne0: Long, ne10: Long, nb10: Long
) {
    val ni = ne0.toInt()
    for (i in 0 until ni) {
        val i10 = (i % ne10.toInt())
        val src0Val = readTypedAsF32(src0Type, src0Data, src0Off + i * typeSizeOf(src0Type))
        val src1Val = readTypedAsF32(src1Type, src1Data, src1Off + (i10 * nb10.toInt()))
        writeF32AsTyped(dstType, dstData, dstOff + i * typeSizeOf(dstType), src0Val + src1Val)
    }
}

/** Read one element as F32 from the given type. */
private fun readTypedAsF32(type: GGMLType, data: ByteArray, off: Int): Float = when (type) {
    GGMLType.F32 -> readFloat(data, off)
    GGMLType.F16 -> ggmlFp16ToFp32(readShort(data, off).toUShort())
    GGMLType.BF16 -> bf16ToF32(GGMLBF16(readShort(data, off).toUShort()))
    else -> error("readTypedAsF32: unsupported type $type")
}

/** Write one F32 value as the given type. */
private fun writeF32AsTyped(type: GGMLType, data: ByteArray, off: Int, v: Float) {
    when (type) {
        GGMLType.F32 -> writeFloat(data, off, v)
        GGMLType.F16 -> writeShort(data, off, ggmlFp32ToFp16(v).toShort())
        GGMLType.BF16 -> writeShort(data, off, f32ToBf16(v).bits.toShort())
        else -> error("writeF32AsTyped: unsupported type $type")
    }
}

/** Element byte size for simple non-quantized types. */
private fun typeSizeOf(type: GGMLType): Int = when (type) {
    GGMLType.F32 -> 4
    GGMLType.F16, GGMLType.BF16, GGMLType.I16 -> 2
    GGMLType.I8 -> 1
    GGMLType.I32 -> 4
    GGMLType.I64 -> 8
    else -> type.sizeBytes
}

// ============================================================================
// ggml_compute_forward_add_id  — ops.cpp line 701
// ============================================================================

// ops.cpp line 703
private fun ggml_compute_forward_add_id_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    val src2 = dst.src[2]!!

    check(dst.type == GGMLType.F32)
    check(src0.type == GGMLType.F32)
    check(src1.type == GGMLType.F32)
    check(src2.type == GGMLType.I32)

    check(src0.nb[0] == 4uL) // sizeof(float)
    check(src1.nb[0] == 4uL)

    val ith = params.ith
    val nth = params.nth

    val nr = ggmlNrows(src0).toInt()

    val l = ternaryOpLocals(dst)

    check(l.nb0 == 4L) // sizeof(float)
    check(l.nb10 == 4L)

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val src2Data = src2.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        // src0 indices
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        // src1 indices
        val i11 = readInt(src2Data, (i1 * l.nb20 + i2 * l.nb21).toInt())

        check(i11 >= 0 && i11 < l.ne11.toInt())

        val dstRowOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0RowOff = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()
        val src1RowOff = (i11.toLong() * l.nb11).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val a = readFloat(src0Data, src0RowOff + i * 4)
            val b = readFloat(src1Data, src1RowOff + i * 4)
            writeFloat(dstData, dstRowOff + i * 4, a + b)
        }
    }
}

// ops.cpp line 754
fun ggml_compute_forward_add_id(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F32 -> {
            ggml_compute_forward_add_id_f32(params, dst)
        }
        else -> {
            error("unsupported type for ggml_compute_forward_add_id: ${ggmlTypeName(src0.type)}")
        }
    }
}

// ============================================================================
// ggml_compute_forward_add1  — ops.cpp line 772
// ============================================================================

// ops.cpp line 774
private fun ggml_compute_forward_add1_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggml_are_same_shape(src0, dst))
    check(ggmlIsScalar(src1))

    val ith = params.ith
    val nth = params.nth

    val nr = ggmlNrows(src0).toInt()

    val l = unaryOpLocals(dst)

    check(l.nb0 == 4L) // sizeof(float)
    check(l.nb00 == 4L)

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    val v = readFloat(src1Data, 0)

    for (ir in ir0 until ir1) {
        // src0 and dst are same shape => same indices
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        val src0Row = readFloatRow(src0Data, src0Off, l.ne0.toInt())
        for (i in 0 until l.ne0.toInt()) {
            writeFloat(dstData, dstOff + i * 4, src0Row[i] + v)
        }
    }
}

// ops.cpp line 824
private fun ggml_compute_forward_add1_f16_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggml_are_same_shape(src0, dst))
    check(ggmlIsScalar(src1))

    // scalar to add
    val v = readFloat(src1.data as ByteArray, 0)

    val ith = params.ith
    val nth = params.nth

    val nr = ggmlNrows(src0).toInt()

    val l = unaryOpLocals(dst)

    check(src0.type == GGMLType.F16)
    check(src1.type == GGMLType.F32)
    check(dst.type == GGMLType.F16)

    check(l.nb0 == 2L) // sizeof(ggml_fp16_t)
    check(l.nb00 == 2L)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val src0Val = ggmlFp16ToFp32(readShort(src0Data, src0Off + i * 2).toUShort())
            writeShort(dstData, dstOff + i * 2, ggmlFp32ToFp16(src0Val + v).toShort())
        }
    }
}

// ops.cpp line 872
private fun ggml_compute_forward_add1_f16_f16(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggml_are_same_shape(src0, dst))
    check(ggmlIsScalar(src1))

    // scalar to add
    val v = ggmlFp16ToFp32(readShort(src1.data as ByteArray, 0).toUShort())

    val ith = params.ith
    val nth = params.nth

    val nr = ggmlNrows(src0).toInt()

    val l = unaryOpLocals(dst)

    check(src0.type == GGMLType.F16)
    check(src1.type == GGMLType.F16)
    check(dst.type == GGMLType.F16)

    check(l.nb0 == 2L)
    check(l.nb00 == 2L)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val src0Val = ggmlFp16ToFp32(readShort(src0Data, src0Off + i * 2).toUShort())
            writeShort(dstData, dstOff + i * 2, ggmlFp32ToFp16(src0Val + v).toShort())
        }
    }
}

// ops.cpp line 920
private fun ggml_compute_forward_add1_q_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggml_are_same_shape(src0, dst))
    check(ggmlIsScalar(src1))

    // scalar to add
    val v = readFloat(src1.data as ByteArray, 0)

    val ith = params.ith
    val nth = params.nth

    val nr = ggmlNrows(src0).toInt()

    val l = unaryOpLocals(dst)

    val type = src0.type

    // we don't support permuted src0
    check(l.nb00 == ggmlTypeSize(type).toLong())

    // dst cannot be transposed or permuted
    check(l.nb0 <= l.nb1)
    check(l.nb1 <= l.nb2)
    check(l.nb2 <= l.nb3)

    check(ggmlIsQuantized(src0.type))
    check(dst.type == src0.type)
    check(src1.type == GGMLType.F32)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val wdata = FloatArray(l.ne0.toInt() + CACHE_LINE_SIZE_F32)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val src0RowOff = (i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()
        val dstRowOff  = (i1 * l.nb1  + i2 * l.nb2  + i3 * l.nb3).toInt()

        check(l.ne0 % 32 == 0L)

        // unquantize row from src0 to temp buffer
        dequantize_row(type, src0Data, src0RowOff, wdata, 0, l.ne0)
        // add src1
        ggml_vec_acc1_f32(l.ne0, wdata, 0, v)
        // quantize row to dst
        val tmpDst = ByteArray(ggmlRowSize(type, l.ne0).toInt())
        quantize_row(type, wdata, tmpDst, l.ne0)
        tmpDst.copyInto(dstData, dstRowOff, 0, tmpDst.size)
    }
}

// ops.cpp line 985
private fun ggml_compute_forward_add1_bf16_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggml_are_same_shape(src0, dst))
    check(ggmlIsScalar(src1))

    // scalar to add
    val v = readFloat(src1.data as ByteArray, 0)

    val ith = params.ith
    val nth = params.nth

    val nr = ggmlNrows(src0).toInt()

    val l = unaryOpLocals(dst)

    check(src0.type == GGMLType.BF16)
    check(src1.type == GGMLType.F32)
    check(dst.type == GGMLType.BF16)

    check(l.nb0 == 2L)
    check(l.nb00 == 2L)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val src0Val = bf16ToF32(GGMLBF16(readShort(src0Data, src0Off + i * 2).toUShort()))
            writeShort(dstData, dstOff + i * 2, f32ToBf16(src0Val + v).bits.toShort())
        }
    }
}

// ops.cpp line 1033
private fun ggml_compute_forward_add1_bf16_bf16(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggml_are_same_shape(src0, dst))
    check(ggmlIsScalar(src1))

    // scalar to add
    val v = bf16ToF32(GGMLBF16(readShort(src1.data as ByteArray, 0).toUShort()))

    val ith = params.ith
    val nth = params.nth

    val nr = ggmlNrows(src0).toInt()

    val l = unaryOpLocals(dst)

    check(src0.type == GGMLType.BF16)
    check(src1.type == GGMLType.BF16)
    check(dst.type == GGMLType.BF16)

    check(l.nb0 == 2L)
    check(l.nb00 == 2L)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val src0Val = bf16ToF32(GGMLBF16(readShort(src0Data, src0Off + i * 2).toUShort()))
            writeShort(dstData, dstOff + i * 2, f32ToBf16(src0Val + v).bits.toShort())
        }
    }
}

// ops.cpp line 1081
fun ggml_compute_forward_add1(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    when (src0.type) {
        GGMLType.F32 -> {
            ggml_compute_forward_add1_f32(params, dst)
        }
        GGMLType.F16 -> {
            if (src1.type == GGMLType.F16) {
                ggml_compute_forward_add1_f16_f16(params, dst)
            } else if (src1.type == GGMLType.F32) {
                ggml_compute_forward_add1_f16_f32(params, dst)
            } else {
                error("fatal error")
            }
        }
        GGMLType.BF16 -> {
            if (src1.type == GGMLType.BF16) {
                ggml_compute_forward_add1_bf16_bf16(params, dst)
            } else if (src1.type == GGMLType.F32) {
                ggml_compute_forward_add1_bf16_f32(params, dst)
            } else {
                error("fatal error")
            }
        }
        GGMLType.Q1_0,
        GGMLType.Q4_0, GGMLType.Q4_1,
        GGMLType.Q5_0, GGMLType.Q5_1,
        GGMLType.Q8_0, GGMLType.Q8_1,
        GGMLType.MXFP4, GGMLType.NVFP4,
        GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
        GGMLType.TQ1_0, GGMLType.TQ2_0,
        GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ3_XXS,
        GGMLType.IQ1_S, GGMLType.IQ1_M,
        GGMLType.IQ4_NL, GGMLType.IQ4_XS,
        GGMLType.IQ3_S, GGMLType.IQ2_S -> {
            ggml_compute_forward_add1_q_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_acc  — ops.cpp line 1152
// ============================================================================

// ops.cpp line 1154
private fun ggml_compute_forward_acc_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggml_are_same_shape(src0, dst))
    check(ggmlIsContiguous(dst) && ggmlIsContiguous(src0))

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    val nb1     = dst.opParams[0].toLong()
    val nb2     = dst.opParams[1].toLong()
    val nb3     = dst.opParams[2].toLong()
    val offset  = dst.opParams[3].toLong()
    val inplace = dst.opParams[4] != 0

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    if (!inplace) {
        if (params.ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            src0Data.copyInto(dstData, 0, 0, ggmlNbytes(dst).toInt())
        }
        ggmlBarrier(params.threadpool ?: GGMLThreadpool())
    }

    val ith = params.ith
    val nth = params.nth

    val nr = ggmlNrows(src1).toInt()
    val nc = src1.ne[0]

    // GGML_TENSOR_LOCALS for src1
    val ne10 = src1.ne[0]; val ne11 = src1.ne[1]; val ne12 = src1.ne[2]; val ne13 = src1.ne[3]
    val nb10 = src1.nb[0].toLong(); val nb11 = src1.nb[1].toLong()
    val nb12 = src1.nb[2].toLong(); val nb13 = src1.nb[3].toLong()

    // src0 and dst as viewed during acc
    val nb0 = ggmlElementSize(src0).toLong()

    val nb00 = nb0
    val nb01 = nb1
    val nb02 = nb2
    val nb03 = nb3

    check(offset + (if (ne10 == 0L) 0L else ne10 - 1) * nb0 + (if (ne11 == 0L) 0L else ne11 - 1) * nb1 + (if (ne12 == 0L) 0L else ne12 - 1) * nb2 + (if (ne13 == 0L) 0L else ne13 - 1) * nb3 < ggmlNbytes(dst).toLong())
    check(offset + (if (ne10 == 0L) 0L else ne10 - 1) * nb00 + (if (ne11 == 0L) 0L else ne11 - 1) * nb01 + (if (ne12 == 0L) 0L else ne12 - 1) * nb02 + (if (ne13 == 0L) 0L else ne13 - 1) * nb03 < ggmlNbytes(src0).toLong())

    check(nb10 == 4L) // sizeof(float)

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    for (ir in ir0 until ir1) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        val i3 = ir / (ne12 * ne11).toInt()
        val i2 = (ir - i3 * (ne12 * ne11).toInt()) / ne11.toInt()
        val i1 = ir - i3 * (ne12 * ne11).toInt() - i2 * ne11.toInt()

        val dstRowOff  = (i3 * nb3 + i2 * nb2 + i1 * nb1 + offset).toInt()
        val src0RowOff = (i3 * nb03 + i2 * nb02 + i1 * nb01 + offset).toInt()
        val src1RowOff = (i3 * nb13 + i2 * nb12 + i1 * nb11).toInt()

        for (i in 0 until nc.toInt()) {
            val a = readFloat(src0Data, src0RowOff + i * 4)
            val b = readFloat(src1Data, src1RowOff + i * 4)
            writeFloat(dstData, dstRowOff + i * 4, a + b)
        }
    }
}

// ops.cpp line 1234
fun ggml_compute_forward_acc(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F32 -> {
            ggml_compute_forward_acc_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_sum  — ops.cpp line 1279
// ============================================================================

// ops.cpp line 1281
private fun ggml_compute_forward_sum_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(ggmlIsScalar(dst))
    check(src0.nb[0] == 4uL) // sizeof(float)

    val ne00 = src0.ne[0]; val ne01 = src0.ne[1]; val ne02 = src0.ne[2]; val ne03 = src0.ne[3]
    val nb01 = src0.nb[1].toLong(); val nb02 = src0.nb[2].toLong(); val nb03 = src0.nb[3].toLong()

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    var sum = 0.0

    for (i03 in 0 until ne03) {
        for (i02 in 0 until ne02) {
            for (i01 in 0 until ne01) {
                val rowOff = (i01 * nb01 + i02 * nb02 + i03 * nb03).toInt()
                val rowSum = ggml_vec_sum_f32_ggf(ne00, src0Data, rowOff)
                sum += rowSum
            }
        }
    }
    writeFloat(dstData, 0, sum.toFloat())
}

/** ggml_vec_sum_f32_ggf for ByteArray (reads floats at byte offset). */
private fun ggml_vec_sum_f32_ggf(n: Long, data: ByteArray, off: Int): Double {
    var sum = 0.0
    for (i in 0 until n.toInt()) {
        sum += readFloat(data, off + i * 4).toDouble()
    }
    return sum
}

// ops.cpp line 1313
private fun ggml_compute_forward_sum_f16(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(ggmlIsScalar(dst))
    check(src0.nb[0] == 2uL) // sizeof(ggml_fp16_t)

    val ne00 = src0.ne[0]; val ne01 = src0.ne[1]; val ne02 = src0.ne[2]; val ne03 = src0.ne[3]
    val nb01 = src0.nb[1].toLong(); val nb02 = src0.nb[2].toLong(); val nb03 = src0.nb[3].toLong()

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    var sum = 0f

    for (i03 in 0 until ne03) {
        for (i02 in 0 until ne02) {
            for (i01 in 0 until ne01) {
                val rowOff = (i01 * nb01 + i02 * nb02 + i03 * nb03).toInt()
                val rowSum = ggml_vec_sum_f16_ggf(ne00, src0Data, rowOff)
                sum += rowSum
            }
        }
    }
    writeShort(dstData, 0, ggmlFp32ToFp16(sum).toShort())
}

// ops.cpp line 1346
private fun ggml_compute_forward_sum_bf16(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(ggmlIsScalar(dst))
    check(src0.nb[0] == 2uL) // sizeof(ggml_bf16_t)

    val ne00 = src0.ne[0]; val ne01 = src0.ne[1]; val ne02 = src0.ne[2]; val ne03 = src0.ne[3]
    val nb01 = src0.nb[1].toLong(); val nb02 = src0.nb[2].toLong(); val nb03 = src0.nb[3].toLong()

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    var sum = 0f

    for (i03 in 0 until ne03) {
        for (i02 in 0 until ne02) {
            for (i01 in 0 until ne01) {
                val rowOff = (i01 * nb01 + i02 * nb02 + i03 * nb03).toInt()
                val rowSum = ggml_vec_sum_bf16_ggf(ne00, src0Data, rowOff)
                sum += rowSum
            }
        }
    }
    val bf16Val = f32ToBf16(sum)
    writeShort(dstData, 0, bf16Val.bits.toShort())
}

// ops.cpp line 1379
fun ggml_compute_forward_sum(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F32 -> {
            ggml_compute_forward_sum_f32(params, dst)
        }
        GGMLType.F16 -> {
            ggml_compute_forward_sum_f16(params, dst)
        }
        GGMLType.BF16 -> {
            ggml_compute_forward_sum_bf16(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_cumsum  — ops.cpp line 1405
// ============================================================================

// ops.cpp line 1407
private fun ggml_compute_forward_cumsum_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    check(src0.nb[0] == 4uL) // sizeof(float)
    check(dst.nb[0] == 4uL)

    val l = unaryOpLocals(dst)

    check(l.ne0 == l.ne00)
    check(l.ne1 == l.ne01)
    check(l.ne2 == l.ne02)
    check(l.ne3 == l.ne03)

    val (ir0, ir1) = getThreadRange(params, src0)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i03 = ir / (l.ne02 * l.ne01)
        val i02 = (ir - i03 * l.ne02 * l.ne01) / l.ne01
        val i01 = ir - i03 * l.ne02 * l.ne01 - i02 * l.ne01

        val srcRowOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
        val dstRowOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()

        val srcRow = readFloatRow(src0Data, srcRowOff, l.ne00.toInt())
        val dstRow = FloatArray(l.ne00.toInt())
        ggml_vec_cumsum_f32(l.ne00, dstRow, 0, srcRow, 0)
        writeFloatRow(dstData, dstRowOff, dstRow, 0, l.ne00.toInt())
    }
}

// ops.cpp line 1437
fun ggml_compute_forward_cumsum(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F32 -> {
            ggml_compute_forward_cumsum_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_sum_rows  — ops.cpp line 1455
// ============================================================================

// ops.cpp line 1457
private fun ggml_compute_forward_sum_rows_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(src0.nb[0] == 4uL) // sizeof(float)
    check(dst.nb[0] == 4uL)

    val l = unaryOpLocals(dst)

    check(l.ne0 == 1L)
    check(l.ne1 == l.ne01)
    check(l.ne2 == l.ne02)
    check(l.ne3 == l.ne03)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i3 in 0 until l.ne03) {
        for (i2 in 0 until l.ne02) {
            for (i1 in 0 until l.ne01) {
                val srcRowOff = (i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()
                val dstRowOff = (i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()
                val srcRow = readFloatRow(src0Data, srcRowOff, l.ne00.toInt())
                val rowSum = ggml_vec_sum_f32(l.ne00, srcRow, 0)
                writeFloat(dstData, dstRowOff, rowSum)
            }
        }
    }
}

// ops.cpp line 1490
fun ggml_compute_forward_sum_rows(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F32 -> {
            ggml_compute_forward_sum_rows_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_mean  — ops.cpp line 1508
// ============================================================================

// ops.cpp line 1510
private fun ggml_compute_forward_mean_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(src0.nb[0] == 4uL) // sizeof(float)

    val l = unaryOpLocals(dst)

    check(l.ne0 == 1L)
    check(l.ne1 == l.ne01)
    check(l.ne2 == l.ne02)
    check(l.ne3 == l.ne03)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i03 in 0 until l.ne03) {
        for (i02 in 0 until l.ne02) {
            for (i01 in 0 until l.ne01) {
                val srcRowOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                val dstOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()
                val srcRow = readFloatRow(src0Data, srcRowOff, l.ne00.toInt())
                val rowSum = ggml_vec_sum_f32(l.ne00, srcRow, 0)
                writeFloat(dstData, dstOff, rowSum / l.ne00.toFloat())
            }
        }
    }
}

// ops.cpp line 1547
fun ggml_compute_forward_mean(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F32 -> {
            ggml_compute_forward_mean_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_argmax  — ops.cpp line 1565
// ============================================================================

// ops.cpp line 1567
private fun ggml_compute_forward_argmax_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(src0.nb[0] == 4uL) // sizeof(float)
    check(dst.nb[0] == 4uL)  // sizeof(float) — dst stores int32 as float-sized

    val ne00 = src0.ne[0]
    val ne01 = src0.ne[1]

    val nb01 = src0.nb[1].toLong()
    val nb0 = dst.nb[0].toLong()

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i1 in 0 until ne01) {
        val srcRow = readFloatRow(src0Data, (i1 * nb01).toInt(), ne00.toInt())
        val v = ggml_vec_argmax_f32(ne00, srcRow, 0)
        writeInt(dstData, (i1 * nb0).toInt(), v)
    }
}

// ops.cpp line 1595
fun ggml_compute_forward_argmax(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F32 -> {
            ggml_compute_forward_argmax_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_count_equal  — ops.cpp line 1613
// ============================================================================

// ops.cpp line 1615
private fun ggml_compute_forward_count_equal_i32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = binaryOpLocals(dst)

    check(src0.type == GGMLType.I32)
    check(src1.type == GGMLType.I32)
    check(ggml_are_same_shape(src0, src1))
    check(ggmlIsScalar(dst))
    check(dst.type == GGMLType.I64)

    val nr = ggmlNrows(src0)

    val ith = params.ith
    val nth = params.nth

    // NOTE: In C++, sums is params->wdata.  We use a shared array approach.
    // For single-threaded, we just accumulate directly.
    var sumThread = 0L

    // rows per thread
    val dr = ((nr + nth - 1) / nth).toInt()

    // row range for this thread
    val ir0 = (dr * ith).toLong()
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i03 = ir / (l.ne02 * l.ne01)
        val i02 = (ir - i03 * l.ne03) / l.ne01
        val i01 = ir - i03 * l.ne03 - i02 * l.ne02

        val data0Off = (i03 * l.nb03 + i02 * l.nb02 + i01 * l.nb01).toInt()
        val data1Off = (i03 * l.nb13 + i02 * l.nb12 + i01 * l.nb11).toInt()

        for (i00 in 0 until l.ne00) {
            val val0 = readInt(src0Data, data0Off + (i00 * l.nb00).toInt())
            val val1 = readInt(src1Data, data1Off + (i00 * l.nb10).toInt())

            if (val0 == val1) sumThread++
        }
    }

    // NOTE: Multi-threaded accumulation via wdata is simplified here.
    // For single-threaded (nth==1), write directly.
    // For multi-threaded, a proper barrier + accumulation would be needed.
    if (ith == 0) {
        writeLong(dstData, 0, sumThread)
    }
}

// ops.cpp line 1675
fun ggml_compute_forward_count_equal(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.I32 -> {
            ggml_compute_forward_count_equal_i32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_repeat  — ops.cpp line 1693
// ============================================================================

// ops.cpp line 1695
private fun ggml_compute_forward_repeat_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(ggmlCanRepeat(src0, dst))

    val l = unaryOpLocals(dst)

    // guaranteed to be an integer due to the check in ggml_can_repeat
    val nr0 = (l.ne0 / l.ne00).toInt()
    val nr1 = (l.ne1 / l.ne01).toInt()
    val nr2 = (l.ne2 / l.ne02).toInt()
    val nr3 = (l.ne3 / l.ne03).toInt()

    // TODO: support for transposed / permuted tensors
    check(l.nb0 == 4L) // sizeof(float)
    check(l.nb00 == 4L)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i3 in 0 until nr3) {
        for (k3 in 0 until l.ne03.toInt()) {
            for (i2 in 0 until nr2) {
                for (k2 in 0 until l.ne02.toInt()) {
                    for (i1 in 0 until nr1) {
                        for (k1 in 0 until l.ne01.toInt()) {
                            for (i0 in 0 until nr0) {
                                val dstOff = ((i3 * l.ne03 + k3) * l.nb3 + (i2 * l.ne02 + k2) * l.nb2 + (i1 * l.ne01 + k1) * l.nb1 + (i0 * l.ne00) * l.nb0).toInt()
                                val srcOff = (k3 * l.nb03 + k2 * l.nb02 + k1 * l.nb01).toInt()
                                // ggml_vec_cpy_f32
                                for (idx in 0 until l.ne00.toInt()) {
                                    writeFloat(dstData, dstOff + idx * 4, readFloat(src0Data, srcOff + idx * 4))
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ops.cpp line 1739
private fun ggml_compute_forward_repeat_f16(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(ggmlCanRepeat(src0, dst))

    val l = unaryOpLocals(dst)

    // guaranteed to be an integer due to the check in ggml_can_repeat
    val nr0 = (l.ne0 / l.ne00).toInt()
    val nr1 = (l.ne1 / l.ne01).toInt()
    val nr2 = (l.ne2 / l.ne02).toInt()
    val nr3 = (l.ne3 / l.ne03).toInt()

    // TODO: support for transposed / permuted tensors
    check(l.nb0 == 2L) // sizeof(ggml_fp16_t)
    check(l.nb00 == 2L)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i3 in 0 until nr3) {
        for (k3 in 0 until l.ne03.toInt()) {
            for (i2 in 0 until nr2) {
                for (k2 in 0 until l.ne02.toInt()) {
                    for (i1 in 0 until nr1) {
                        for (k1 in 0 until l.ne01.toInt()) {
                            for (i0 in 0 until nr0) {
                                val dstOff = ((i3 * l.ne03 + k3) * l.nb3 + (i2 * l.ne02 + k2) * l.nb2 + (i1 * l.ne01 + k1) * l.nb1 + (i0 * l.ne00) * l.nb0).toInt()
                                val srcOff = (k3 * l.nb03 + k2 * l.nb02 + k1 * l.nb01).toInt()
                                // ggml_vec_cpy_f16 — copy raw bytes
                                for (idx in 0 until l.ne00.toInt()) {
                                    val v = readShort(src0Data, srcOff + idx * 2)
                                    writeShort(dstData, dstOff + idx * 2, v)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ops.cpp line 1786
fun ggml_compute_forward_repeat(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F16, GGMLType.BF16, GGMLType.I16 -> {
            ggml_compute_forward_repeat_f16(params, dst)
        }
        GGMLType.F32, GGMLType.I32 -> {
            ggml_compute_forward_repeat_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_repeat_back  — ops.cpp line 1817
// ============================================================================

// ops.cpp line 1819
private fun ggml_compute_forward_repeat_back_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(ggmlCanRepeat(dst, src0))

    val l = unaryOpLocals(dst)

    // guaranteed to be an integer due to the check in ggml_can_repeat
    val nr0 = (l.ne00 / l.ne0).toInt()
    val nr1 = (l.ne01 / l.ne1).toInt()
    val nr2 = (l.ne02 / l.ne2).toInt()
    val nr3 = (l.ne03 / l.ne3).toInt()

    // TODO: support for transposed / permuted tensors
    check(l.nb0 == 4L) // sizeof(float)
    check(l.nb00 == 4L)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    if (ggmlIsContiguous(dst)) {
        val totalN = l.ne0 * l.ne1 * l.ne2 * l.ne3
        for (i in 0 until totalN.toInt()) {
            writeFloat(dstData, i * 4, 0f)
        }
    } else {
        for (k3 in 0 until l.ne3.toInt()) {
            for (k2 in 0 until l.ne2.toInt()) {
                for (k1 in 0 until l.ne1.toInt()) {
                    val rowOff = (k1 * l.nb1 + k2 * l.nb2 + k3 * l.nb3).toInt()
                    for (i in 0 until l.ne0.toInt()) {
                        writeFloat(dstData, rowOff + i * 4, 0f)
                    }
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    for (i3 in 0 until nr3) {
        for (k3 in 0 until l.ne3.toInt()) {
            for (i2 in 0 until nr2) {
                for (k2 in 0 until l.ne2.toInt()) {
                    for (i1 in 0 until nr1) {
                        for (k1 in 0 until l.ne1.toInt()) {
                            for (i0 in 0 until nr0) {
                                val dstOff = (k3 * l.nb3 + k2 * l.nb2 + k1 * l.nb1).toInt()
                                val srcOff = ((i3 * l.ne3 + k3) * l.nb03 + (i2 * l.ne2 + k2) * l.nb02 + (i1 * l.ne1 + k1) * l.nb01 + (i0 * l.ne0) * l.nb00).toInt()
                                // ggml_vec_acc_f32
                                for (idx in 0 until l.ne0.toInt()) {
                                    val existing = readFloat(dstData, dstOff + idx * 4)
                                    val toAdd = readFloat(src0Data, srcOff + idx * 4)
                                    writeFloat(dstData, dstOff + idx * 4, existing + toAdd)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ops.cpp line 1877
fun ggml_compute_forward_repeat_back(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F32 -> {
            ggml_compute_forward_repeat_back_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_concat  — ops.cpp line 1895
// ============================================================================

// ops.cpp line 1897
private fun ggml_compute_forward_concat_any(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val len = ggmlTypeSize(src0.type).toInt()

    val ith = params.ith
    val nth = params.nth

    val l = binaryOpLocals(dst)

    val dim = ggml_get_op_params_i32(dst, 0)

    check(dim in 0..3)

    val o = longArrayOf(0L, 0L, 0L, 0L)
    o[dim] = src0.ne[dim]

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    // TODO: smarter multi-threading
    for (i3 in 0 until l.ne3.toInt()) {
        var i2 = ith
        while (i2 < l.ne2.toInt()) {
            for (i1 in 0 until l.ne1.toInt()) {
                for (i0 in 0 until l.ne0.toInt()) {
                    val xOff: Int
                    val xData: ByteArray
                    if (i0 < l.ne00.toInt() && i1 < l.ne01.toInt() && i2 < l.ne02.toInt() && i3 < l.ne03.toInt()) {
                        xData = src0Data
                        xOff = (i0 * l.nb00 + i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()
                    } else {
                        xData = src1Data
                        xOff = ((i0 - o[0]) * l.nb10 + (i1 - o[1]) * l.nb11 + (i2 - o[2]) * l.nb12 + (i3 - o[3]) * l.nb13).toInt()
                    }

                    val yOff = (i0 * l.nb0 + i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()

                    xData.copyInto(dstData, yOff, xOff, xOff + len)
                }
            }
            i2 += nth
        }
    }
}

// ops.cpp line 1940
private fun ggml_compute_forward_concat_i8(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggmlTypeSize(src0.type) == 1uL) // sizeof(int8_t)

    val ith = params.ith
    val nth = params.nth

    val l = binaryOpLocals(dst)

    val dim = ggml_get_op_params_i32(dst, 0)

    check(dim in 0..3)

    val o = longArrayOf(0L, 0L, 0L, 0L)
    o[dim] = src0.ne[dim]

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    // TODO: smarter multi-threading
    for (i3 in 0 until l.ne3.toInt()) {
        var i2 = ith
        while (i2 < l.ne2.toInt()) {
            for (i1 in 0 until l.ne1.toInt()) {
                for (i0 in 0 until l.ne0.toInt()) {
                    val x: Byte
                    if (i0 < l.ne00.toInt() && i1 < l.ne01.toInt() && i2 < l.ne02.toInt() && i3 < l.ne03.toInt()) {
                        x = src0Data[(i0 * l.nb00 + i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()]
                    } else {
                        x = src1Data[((i0 - o[0]) * l.nb10 + (i1 - o[1]) * l.nb11 + (i2 - o[2]) * l.nb12 + (i3 - o[3]) * l.nb13).toInt()]
                    }

                    val yOff = (i0 * l.nb0 + i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()
                    dstData[yOff] = x
                }
            }
            i2 += nth
        }
    }
}

// ops.cpp line 1983
private fun ggml_compute_forward_concat_f16(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggmlTypeSize(src0.type) == 2uL) // sizeof(ggml_fp16_t)

    val ith = params.ith
    val nth = params.nth

    val l = binaryOpLocals(dst)

    val dim = ggml_get_op_params_i32(dst, 0)

    check(dim in 0..3)

    val o = longArrayOf(0L, 0L, 0L, 0L)
    o[dim] = src0.ne[dim]

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    // TODO: smarter multi-threading
    for (i3 in 0 until l.ne3.toInt()) {
        var i2 = ith
        while (i2 < l.ne2.toInt()) {
            for (i1 in 0 until l.ne1.toInt()) {
                for (i0 in 0 until l.ne0.toInt()) {
                    val xVal: Short
                    if (i0 < l.ne00.toInt() && i1 < l.ne01.toInt() && i2 < l.ne02.toInt() && i3 < l.ne03.toInt()) {
                        xVal = readShort(src0Data, (i0 * l.nb00 + i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt())
                    } else {
                        xVal = readShort(src1Data, ((i0 - o[0]) * l.nb10 + (i1 - o[1]) * l.nb11 + (i2 - o[2]) * l.nb12 + (i3 - o[3]) * l.nb13).toInt())
                    }

                    val yOff = (i0 * l.nb0 + i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()
                    writeShort(dstData, yOff, xVal)
                }
            }
            i2 += nth
        }
    }
}

// ops.cpp line 2026
private fun ggml_compute_forward_concat_f32(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(ggmlTypeSize(src0.type) == 4uL) // sizeof(float)

    val ith = params.ith
    val nth = params.nth

    val l = binaryOpLocals(dst)

    val dim = ggml_get_op_params_i32(dst, 0)

    check(dim in 0..3)

    val o = longArrayOf(0L, 0L, 0L, 0L)
    o[dim] = src0.ne[dim]

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    // TODO: smarter multi-threading
    for (i3 in 0 until l.ne3.toInt()) {
        var i2 = ith
        while (i2 < l.ne2.toInt()) {
            for (i1 in 0 until l.ne1.toInt()) {
                for (i0 in 0 until l.ne0.toInt()) {
                    val xVal: Float
                    if (i0 < l.ne00.toInt() && i1 < l.ne01.toInt() && i2 < l.ne02.toInt() && i3 < l.ne03.toInt()) {
                        xVal = readFloat(src0Data, (i0 * l.nb00 + i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt())
                    } else {
                        xVal = readFloat(src1Data, ((i0 - o[0]) * l.nb10 + (i1 - o[1]) * l.nb11 + (i2 - o[2]) * l.nb12 + (i3 - o[3]) * l.nb13).toInt())
                    }

                    val yOff = (i0 * l.nb0 + i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()
                    writeFloat(dstData, yOff, xVal)
                }
            }
            i2 += nth
        }
    }
}

// ops.cpp line 2069
fun ggml_compute_forward_concat(
    params: GGMLComputeParams,
    dst: GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        GGMLType.F16, GGMLType.BF16, GGMLType.I16 -> {
            ggml_compute_forward_concat_f16(params, dst)
        }
        GGMLType.I8 -> {
            ggml_compute_forward_concat_i8(params, dst)
        }
        GGMLType.F32, GGMLType.I32 -> {
            ggml_compute_forward_concat_f32(params, dst)
        }
        else -> {
            ggml_compute_forward_concat_any(params, dst)
        }
    }
}
