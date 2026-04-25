// port-lint: source ggml/src/ggml-quants.c

package ai.solace.llamakotlin.core

// QK constants are top-level in GGMLTypes.kt and GGMLCommon.kt — same package, no import needed.

/**
 * Port of `ggml-quants.c` — dequantization, quantization-row-ref, and
 * importance-matrix quantization functions.
 *
 * Block data is stored as packed [ByteArray]. Each block type has a known
 * SIZE_BYTES constant in [GGMLCommon]. Functions here operate on raw byte
 * buffers with explicit block offsets, matching the C memory layout.
 *
 * Dequantize functions read packed blocks and write F32 output.
 * Quantize-row-ref functions read F32 input and write packed blocks.
 *
 * IQ dequantize functions that depend on grid lookup tables (iq2xxs_grid,
 * iq1s_grid, etc.) are deferred until those tables are ported.
 */

private const val GROUP_MAX_EPS = 1e-15f
// IQ1S_DELTA is defined in GGMLCommon.kt

// ════════════════════════════════════════════════════════════════════════════════
//  Helper: get_scale_min_k4 (used by Q4_K, Q5_K dequantize)
// ════════════════════════════════════════════════════════════════════════════════

private data class ScaleMin(val d: Int, val m: Int)

/** Port of `get_scale_min_k4` from ggml-quants.c. */
private fun get_scale_min_k4(j: Int, q: ByteArray, qOff: Int): ScaleMin {
    return if (j < 4) {
        ScaleMin(
            d = (q[qOff + j].toInt() and 0xFF) and 63,
            m = (q[qOff + j + 4].toInt() and 0xFF) and 63
        )
    } else {
        ScaleMin(
            d = ((q[qOff + j + 4].toInt() and 0xFF) and 0xF) or ((((q[qOff + j - 4].toInt() and 0xFF) shr 6) and 3) shl 4),
            m = ((q[qOff + j + 4].toInt() and 0xFF) shr 4) or ((((q[qOff + j].toInt() and 0xFF) shr 6) and 3) shl 4)
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Quantization helper functions (ggml-quants.c lines 559–1080)
// ════════════════════════════════════════════════════════════════════════════════

/** Port of `nearest_int` (C line 559). Uses the float bit trick for fast rounding. */
private fun nearest_int(fval: Float): Int {
    val v = fval + 12582912f
    val i = v.toRawBits()
    return (i and 0x007fffff) - 0x00400000
}

/** Port of `make_qx_quants` (C line 566). Signed quantization helper. */
private fun make_qx_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    L: ByteArray, lOff: Int, rmseType: Int,
    qw: FloatArray?, qwOff: Int
): Float {
    var max = 0f
    var amax = 0f
    for (i in 0 until n) {
        val ax = kotlin.math.abs(x[xOff + i])
        if (ax > amax) { amax = ax; max = x[xOff + i] }
    }
    if (amax < GROUP_MAX_EPS) {
        for (i in 0 until n) L[lOff + i] = 0
        return 0f
    }
    var iscale = -nmax.toFloat() / max
    var rmseTypeVar = rmseType
    if (rmseTypeVar == 0) {
        for (i in 0 until n) {
            val l = nearest_int(iscale * x[xOff + i])
            L[lOff + i] = (nmax + maxOf(-nmax, minOf(nmax - 1, l))).toByte()
        }
        return 1f / iscale
    }
    var returnEarly = false
    if (rmseTypeVar < 0) {
        rmseTypeVar = -rmseTypeVar
        returnEarly = true
    }
    var sumlx = 0f
    var suml2 = 0f
    for (i in 0 until n) {
        var l = nearest_int(iscale * x[xOff + i])
        l = maxOf(-nmax, minOf(nmax - 1, l))
        L[lOff + i] = (l + nmax).toByte()
        val w = if (qw != null) qw[qwOff + i]
                else if (rmseTypeVar == 1) x[xOff + i] * x[xOff + i]
                else if (rmseTypeVar == 2) 1f
                else if (rmseTypeVar == 3) kotlin.math.abs(x[xOff + i])
                else kotlin.math.sqrt(kotlin.math.abs(x[xOff + i]))
        sumlx += w * x[xOff + i] * l
        suml2 += w * l * l
    }
    var scale = if (suml2 != 0f) sumlx / suml2 else 0f
    if (returnEarly) return if (suml2 > 0f) 0.5f * (scale + 1f / iscale) else 1f / iscale
    var best = scale * sumlx
    for (is_ in -9..9) {
        if (is_ == 0) continue
        iscale = -(nmax + 0.1f * is_) / max
        sumlx = 0f; suml2 = 0f
        for (i in 0 until n) {
            var l = nearest_int(iscale * x[xOff + i])
            l = maxOf(-nmax, minOf(nmax - 1, l))
            val w = if (qw != null) qw[qwOff + i]
                    else if (rmseTypeVar == 1) x[xOff + i] * x[xOff + i]
                    else if (rmseTypeVar == 2) 1f
                    else if (rmseTypeVar == 3) kotlin.math.abs(x[xOff + i])
                    else kotlin.math.sqrt(kotlin.math.abs(x[xOff + i]))
            sumlx += w * x[xOff + i] * l
            suml2 += w * l * l
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (i in 0 until n) {
                val l = nearest_int(iscale * x[xOff + i])
                L[lOff + i] = (nmax + maxOf(-nmax, minOf(nmax - 1, l))).toByte()
            }
            scale = sumlx / suml2; best = scale * sumlx
        }
    }
    return scale
}

/** Port of `make_q3_quants` (C line 635). Signed quantization with optional RMSE refinement. */
private fun make_q3_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    L: ByteArray, lOff: Int, doRmse: Boolean
): Float {
    var max = 0f
    var amax = 0f
    for (i in 0 until n) {
        val ax = kotlin.math.abs(x[xOff + i])
        if (ax > amax) { amax = ax; max = x[xOff + i] }
    }
    if (amax < GROUP_MAX_EPS) {
        for (i in 0 until n) L[lOff + i] = 0
        return 0f
    }
    val iscale = -nmax.toFloat() / max
    if (doRmse) {
        var sumlx = 0f
        var suml2 = 0f
        for (i in 0 until n) {
            var l = nearest_int(iscale * x[xOff + i])
            l = maxOf(-nmax, minOf(nmax - 1, l))
            L[lOff + i] = l.toByte()
            val w = x[xOff + i] * x[xOff + i]
            sumlx += w * x[xOff + i] * l
            suml2 += w * l * l
        }
        for (itry in 0 until 5) {
            var nChanged = 0
            for (i in 0 until n) {
                val w = x[xOff + i] * x[xOff + i]
                val slx0 = sumlx - w * x[xOff + i] * L[lOff + i]
                if (slx0 > 0) {
                    var sl2 = suml2 - w * L[lOff + i] * L[lOff + i]
                    var newL = nearest_int(x[xOff + i] * sl2 / slx0)
                    newL = maxOf(-nmax, minOf(nmax - 1, newL))
                    if (newL != L[lOff + i].toInt()) {
                        val slx = slx0 + w * x[xOff + i] * newL
                        sl2 += w * newL * newL
                        if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[lOff + i] = newL.toByte(); sumlx = slx; suml2 = sl2
                            ++nChanged
                        }
                    }
                }
            }
            if (nChanged == 0) break
        }
        for (i in 0 until n) {
            L[lOff + i] = (L[lOff + i] + nmax).toByte()
        }
        return if (suml2 > 0f) sumlx / suml2 else 0f
    }
    for (i in 0 until n) {
        var l = nearest_int(iscale * x[xOff + i])
        l = maxOf(-nmax, minOf(nmax - 1, l))
        L[lOff + i] = (l + nmax).toByte()
    }
    return 1f / iscale
}

/** Port of `make_qkx1_quants` (C line 694). Unsigned quantization with min subtraction. */
private fun make_qkx1_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    L: UByteArray, lOff: Int, theMin: FloatArray, minIdx: Int,
    ntry: Int, alpha: Float
): Float {
    var min = x[xOff]
    var max = x[xOff]
    for (i in 1 until n) {
        if (x[xOff + i] < min) min = x[xOff + i]
        if (x[xOff + i] > max) max = x[xOff + i]
    }
    if (max == min) {
        for (i in 0 until n) L[lOff + i] = 0u
        theMin[minIdx] = 0f
        return 0f
    }
    if (min > 0) min = 0f
    var iscale = nmax.toFloat() / (max - min)
    var scale = 1f / iscale
    for (itry in 0 until ntry) {
        var sumlx = 0f; var suml2 = 0
        var didChange = false
        for (i in 0 until n) {
            var l = nearest_int(iscale * (x[xOff + i] - min))
            l = maxOf(0, minOf(nmax, l))
            if (l.toUByte() != L[lOff + i]) {
                L[lOff + i] = l.toUByte()
                didChange = true
            }
            sumlx += (x[xOff + i] - min) * l
            suml2 += l * l
        }
        scale = sumlx / suml2
        var sum = 0f
        for (i in 0 until n) {
            sum += x[xOff + i] - scale * L[lOff + i].toInt()
        }
        min = alpha * min + (1 - alpha) * sum / n
        if (min > 0) min = 0f
        iscale = 1f / scale
        if (!didChange) break
    }
    theMin[minIdx] = -min
    return scale
}

/** Port of `make_qkx2_quants` (C line 737). Weighted quantization with search. */
private fun make_qkx2_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    weights: FloatArray, wOff: Int, L: UByteArray, lOff: Int,
    theMin: FloatArray, minIdx: Int, Laux: UByteArray, lauxOff: Int,
    rmin: Float, rdelta: Float, nstep: Int, useMad: Boolean
): Float {
    var min = x[xOff]
    var max = x[xOff]
    var sumW = weights[wOff]
    var sumX = sumW * x[xOff]
    for (i in 1 until n) {
        if (x[xOff + i] < min) min = x[xOff + i]
        if (x[xOff + i] > max) max = x[xOff + i]
        val w = weights[wOff + i]
        sumW += w
        sumX += w * x[xOff + i]
    }
    if (min > 0) min = 0f
    if (max == min) {
        for (i in 0 until n) L[lOff + i] = 0u
        theMin[minIdx] = -min
        return 0f
    }
    var iscale = nmax.toFloat() / (max - min)
    var scale = 1f / iscale
    var bestError = 0f
    for (i in 0 until n) {
        var l = nearest_int(iscale * (x[xOff + i] - min))
        L[lOff + i] = maxOf(0, minOf(nmax, l)).toUByte()
        var diff = scale * L[lOff + i].toInt() + min - x[xOff + i]
        diff = if (useMad) kotlin.math.abs(diff) else diff * diff
        val w = weights[wOff + i]
        bestError += w * diff
    }
    if (nstep < 1) {
        theMin[minIdx] = -min
        return scale
    }
    for (is_ in 0..nstep) {
        iscale = (rmin + rdelta * is_ + nmax) / (max - min)
        var sumL = 0f; var sumL2 = 0f; var sumXL = 0f
        for (i in 0 until n) {
            var l = nearest_int(iscale * (x[xOff + i] - min))
            l = maxOf(0, minOf(nmax, l))
            Laux[lauxOff + i] = l.toUByte()
            val w = weights[wOff + i]
            sumL += w * l
            sumL2 += w * l * l
            sumXL += w * l * x[xOff + i]
        }
        val D = sumW * sumL2 - sumL * sumL
        if (D > 0) {
            var thisScale = (sumW * sumXL - sumX * sumL) / D
            var thisMin = (sumL2 * sumX - sumL * sumXL) / D
            if (thisMin > 0) {
                thisMin = 0f
                thisScale = sumXL / sumL2
            }
            var curError = 0f
            for (i in 0 until n) {
                var diff = thisScale * Laux[lauxOff + i].toInt() + thisMin - x[xOff + i]
                diff = if (useMad) kotlin.math.abs(diff) else diff * diff
                val w = weights[wOff + i]
                curError += w * diff
            }
            if (curError < bestError) {
                for (i in 0 until n) L[lOff + i] = Laux[lauxOff + i]
                bestError = curError
                scale = thisScale
                min = thisMin
            }
        }
    }
    theMin[minIdx] = -min
    return scale
}

/** Port of `make_qkx3_quants` (C line 931). Like qkx2 but weights can be null. */
private fun make_qkx3_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    weights: FloatArray?, wOff: Int, L: UByteArray, lOff: Int,
    theMin: FloatArray, minIdx: Int, Laux: UByteArray, lauxOff: Int,
    rmin: Float, rdelta: Float, nstep: Int, useMad: Boolean
): Float {
    var min = x[xOff]
    var max = x[xOff]
    var sumW = if (weights != null) weights[wOff] else x[xOff] * x[xOff]
    var sumX = sumW * x[xOff]
    for (i in 1 until n) {
        if (x[xOff + i] < min) min = x[xOff + i]
        if (x[xOff + i] > max) max = x[xOff + i]
        val w = if (weights != null) weights[wOff + i] else x[xOff + i] * x[xOff + i]
        sumW += w
        sumX += w * x[xOff + i]
    }
    if (min > 0) min = 0f
    if (max <= min) {
        for (i in 0 until n) L[lOff + i] = 0u
        theMin[minIdx] = -min
        return 0f
    }
    var iscale = nmax.toFloat() / (max - min)
    var scale = 1f / iscale
    var bestMad = 0f
    for (i in 0 until n) {
        var l = nearest_int(iscale * (x[xOff + i] - min))
        L[lOff + i] = maxOf(0, minOf(nmax, l)).toUByte()
        var diff = scale * L[lOff + i].toInt() + min - x[xOff + i]
        diff = if (useMad) kotlin.math.abs(diff) else diff * diff
        val w = if (weights != null) weights[wOff + i] else x[xOff + i] * x[xOff + i]
        bestMad += w * diff
    }
    if (nstep < 1) {
        theMin[minIdx] = -min
        return scale
    }
    for (is_ in 0..nstep) {
        iscale = (rmin + rdelta * is_ + nmax) / (max - min)
        var sumL = 0f; var sumL2 = 0f; var sumXL = 0f
        for (i in 0 until n) {
            var l = nearest_int(iscale * (x[xOff + i] - min))
            l = maxOf(0, minOf(nmax, l))
            Laux[lauxOff + i] = l.toUByte()
            val w = if (weights != null) weights[wOff + i] else x[xOff + i] * x[xOff + i]
            sumL += w * l
            sumL2 += w * l * l
            sumXL += w * l * x[xOff + i]
        }
        val D = sumW * sumL2 - sumL * sumL
        if (D > 0) {
            var thisScale = (sumW * sumXL - sumX * sumL) / D
            var thisMin = (sumL2 * sumX - sumL * sumXL) / D
            if (thisMin > 0) {
                thisMin = 0f
                thisScale = sumXL / sumL2
            }
            var mad = 0f
            for (i in 0 until n) {
                var diff = thisScale * Laux[lauxOff + i].toInt() + thisMin - x[xOff + i]
                diff = if (useMad) kotlin.math.abs(diff) else diff * diff
                val w = if (weights != null) weights[wOff + i] else x[xOff + i] * x[xOff + i]
                mad += w * diff
            }
            if (mad < bestMad) {
                for (i in 0 until n) L[lOff + i] = Laux[lauxOff + i]
                bestMad = mad
                scale = thisScale
                min = thisMin
            }
        }
    }
    theMin[minIdx] = -min
    return scale
}

/** Port of `make_qp_quants` (C line 1014). Positive-only quantization with iterative refinement. */
private fun make_qp_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    L: UByteArray, lOff: Int, quantWeights: FloatArray, qwOff: Int
): Float {
    var max = 0f
    for (i in 0 until n) {
        max = maxOf(max, x[xOff + i])
    }
    if (max < GROUP_MAX_EPS) {
        for (i in 0 until n) L[lOff + i] = 0u
        return 0f
    }
    var iscale = nmax.toFloat() / max
    for (i in 0 until n) {
        L[lOff + i] = nearest_int(iscale * x[xOff + i]).toUByte()
    }
    var scale = 1f / iscale
    var bestMse = 0f
    for (i in 0 until n) {
        val diff = x[xOff + i] - scale * L[lOff + i].toInt()
        val w = quantWeights[qwOff + i]
        bestMse += w * diff * diff
    }
    for (is_ in -4..4) {
        if (is_ == 0) continue
        val iscaleIs = (0.1f * is_ + nmax) / max
        val scaleIs = 1f / iscaleIs
        var mse = 0f
        for (i in 0 until n) {
            var l = nearest_int(iscaleIs * x[xOff + i])
            l = minOf(nmax, l)
            val diff = x[xOff + i] - scaleIs * l
            val w = quantWeights[qwOff + i]
            mse += w * diff * diff
        }
        if (mse < bestMse) {
            bestMse = mse
            iscale = iscaleIs
        }
    }
    var sumlx = 0f
    var suml2 = 0f
    for (i in 0 until n) {
        var l = nearest_int(iscale * x[xOff + i])
        l = minOf(nmax, l)
        L[lOff + i] = l.toUByte()
        val w = quantWeights[qwOff + i]
        sumlx += w * x[xOff + i] * l
        suml2 += w * l * l
    }
    for (itry in 0 until 5) {
        var nChanged = 0
        for (i in 0 until n) {
            val w = quantWeights[qwOff + i]
            val slx = sumlx - w * x[xOff + i] * L[lOff + i].toInt()
            val sl2 = suml2 - w * L[lOff + i].toInt() * L[lOff + i].toInt()
            if (slx > 0 && sl2 > 0) {
                var newL = nearest_int(x[xOff + i] * sl2 / slx)
                newL = minOf(nmax, newL)
                if (newL != L[lOff + i].toInt()) {
                    val slxNew = slx + w * x[xOff + i] * newL
                    val sl2New = sl2 + w * newL * newL
                    if (slxNew * slxNew * suml2 > sumlx * sumlx * sl2New) {
                        L[lOff + i] = newL.toUByte(); sumlx = slxNew; suml2 = sl2New
                        ++nChanged
                    }
                }
            }
        }
        if (nChanged == 0) break
    }
    return if (suml2 > 0f) sumlx / suml2 else 0f
}

// ════════════════════════════════════════════════════════════════════════════════
//  ByteArray block readers — read fields at known offsets
// ════════════════════════════════════════════════════════════════════════════════

private fun readShortLE(data: ByteArray, off: Int): Short =
    ((data[off].toInt() and 0xFF) or ((data[off + 1].toInt() and 0xFF) shl 8)).toShort()

private fun readIntLE(data: ByteArray, off: Int): Int =
    (data[off].toInt() and 0xFF) or
    ((data[off + 1].toInt() and 0xFF) shl 8) or
    ((data[off + 2].toInt() and 0xFF) shl 16) or
    ((data[off + 3].toInt() and 0xFF) shl 24)

private fun readUShortLE(data: ByteArray, off: Int): Int =
    (data[off].toInt() and 0xFF) or ((data[off + 1].toInt() and 0xFF) shl 8)

private fun fp16ToF32(data: ByteArray, off: Int): Float =
    GGML_FP16_TO_FP32(readShortLE(data, off))

// ════════════════════════════════════════════════════════════════════════════════
//  Dequantize functions
// ════════════════════════════════════════════════════════════════════════════════

// --- block_q1_0: d(2) + qs(QK1_0/8) ---
// QK1_0 = 64, block size = 2 + 8 = 10 bytes (but actual block size from GGMLCommon)

fun dequantize_row_q1_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = QK1_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk / 8  // d(2) + qs(qk/8)

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = fp16ToF32(x, bOff)
        val negD = -d
        val qsOff = bOff + 2
        for (j in 0 until qk) {
            val byteIndex = j / 8
            val bitOffset = j % 8
            val bit = ((x[qsOff + byteIndex].toInt() and 0xFF) shr bitOffset) and 1
            y[yOff + i * qk + j] = if (bit != 0) d else negD
        }
    }
}

// --- block_q4_0: d(2) + qs(QK4_0/2) = 18 bytes ---

fun dequantize_row_q4_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = QK4_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = fp16ToF32(x, bOff)
        val qsOff = bOff + 2
        for (j in 0 until qk / 2) {
            val x0 = (x[qsOff + j].toInt() and 0x0F) - 8
            val x1 = ((x[qsOff + j].toInt() and 0xFF) shr 4) - 8
            y[yOff + i * qk + j] = x0 * d
            y[yOff + i * qk + j + qk / 2] = x1 * d
        }
    }
}

// --- block_q4_1: d(2) + m(2) + qs(QK4_1/2) = 20 bytes ---

fun dequantize_row_q4_1(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = QK4_1
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 4 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = fp16ToF32(x, bOff)
        val m = fp16ToF32(x, bOff + 2)
        val qsOff = bOff + 4
        for (j in 0 until qk / 2) {
            val x0 = x[qsOff + j].toInt() and 0x0F
            val x1 = (x[qsOff + j].toInt() and 0xFF) shr 4
            y[yOff + i * qk + j] = x0 * d + m
            y[yOff + i * qk + j + qk / 2] = x1 * d + m
        }
    }
}

// --- block_q5_0: d(2) + qh(4) + qs(QK5_0/2) = 22 bytes ---

fun dequantize_row_q5_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = QK5_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + 4 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = fp16ToF32(x, bOff)
        val qh = readIntLE(x, bOff + 2).toLong() and 0xFFFFFFFFL
        val qsOff = bOff + 6
        for (j in 0 until qk / 2) {
            val xh0 = (((qh shr (j + 0)) shl 4) and 0x10L).toInt()
            val xh1 = (((qh shr (j + 12))) and 0x10L).toInt()
            val x0 = ((x[qsOff + j].toInt() and 0x0F) or xh0) - 16
            val x1 = (((x[qsOff + j].toInt() and 0xFF) shr 4) or xh1) - 16
            y[yOff + i * qk + j] = x0 * d
            y[yOff + i * qk + j + qk / 2] = x1 * d
        }
    }
}

// --- block_q5_1: d(2) + m(2) + qh(4) + qs(QK5_1/2) = 24 bytes ---

fun dequantize_row_q5_1(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = QK5_1
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 4 + 4 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = fp16ToF32(x, bOff)
        val m = fp16ToF32(x, bOff + 2)
        val qh = readIntLE(x, bOff + 4).toLong() and 0xFFFFFFFFL
        val qsOff = bOff + 8
        for (j in 0 until qk / 2) {
            val xh0 = (((qh shr (j + 0)) shl 4) and 0x10L).toInt()
            val xh1 = (((qh shr (j + 12))) and 0x10L).toInt()
            val x0 = (x[qsOff + j].toInt() and 0x0F) or xh0
            val x1 = ((x[qsOff + j].toInt() and 0xFF) shr 4) or xh1
            y[yOff + i * qk + j] = x0 * d + m
            y[yOff + i * qk + j + qk / 2] = x1 * d + m
        }
    }
}

// --- block_q8_0: d(2) + qs(QK8_0) = 34 bytes ---

fun dequantize_row_q8_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = QK8_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = fp16ToF32(x, bOff)
        val qsOff = bOff + 2
        for (j in 0 until qk) {
            y[yOff + i * qk + j] = x[qsOff + j].toInt() * d
        }
    }
}

// --- block_mxfp4: e(1) + qs(QK_MXFP4/2) ---

fun dequantize_row_mxfp4(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = QK_MXFP4
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 1 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = GGML_E8M0_TO_FP32_HALF((x[bOff].toInt() and 0xFF).toUByte())
        val qsOff = bOff + 1
        for (j in 0 until qk / 2) {
            val x0 = kvalues_mxfp4[(x[qsOff + j].toInt() and 0xFF) and 0x0F]
            val x1 = kvalues_mxfp4[(x[qsOff + j].toInt() and 0xFF) shr 4]
            y[yOff + i * qk + j] = x0 * d
            y[yOff + i * qk + j + qk / 2] = x1 * d
        }
    }
}

// --- block_nvfp4: d(QK_NVFP4/QK_NVFP4_SUB) + qs(QK_NVFP4/2) ---

fun dequantize_row_nvfp4(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = QK_NVFP4
    val qkSub = QK_NVFP4_SUB
    val nSub = qk / qkSub
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = nSub + qk / 2  // d bytes (nSub UE4M3) + qs bytes

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        for (s in 0 until nSub) {
            val d = ggml_ue4m3_to_fp32((x[bOff + s].toInt() and 0xFF).toUByte())
            val yBase = yOff + i * qk + s * qkSub
            val qsBase = bOff + nSub + s * (qkSub / 2)
            for (j in 0 until qkSub / 2) {
                val v0 = kvalues_mxfp4[(x[qsBase + j].toInt() and 0xFF) and 0x0F]
                val v1 = kvalues_mxfp4[(x[qsBase + j].toInt() and 0xFF) shr 4]
                y[yBase + j] = v0 * d
                y[yBase + j + qkSub / 2] = v1 * d
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  K-Quant dequantize
// ════════════════════════════════════════════════════════════════════════════════

// --- block_q2_K: scales(16) + qs(64) + d(2) + dmin(2) = 84 bytes ---
// Layout: qs[QK_K/4]=64, scales[QK_K/16]=16, d(2), dmin(2)
// Actual layout from ggml-common.h: scales(16), qs(64), d(2), dmin(2)

fun dequantize_row_q2_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockQ2K.SIZE_BYTES

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: scales(16) at 0, qs(64) at 16, d(2) at 80, dmin(2) at 82
        val scalesOff = bOff + 0
        val qsOff = bOff + 16
        val d = fp16ToF32(x, bOff + 80)
        val min = fp16ToF32(x, bOff + 82)

        var qIdx = qsOff
        var is_ = 0
        for (n in 0 until QK_K step 128) {
            var shift = 0
            for (j in 0 until 4) {
                val sc = x[scalesOff + is_].toInt() and 0xFF; is_++
                val dl = d * (sc and 0xF)
                val ml = min * (sc shr 4)
                for (l in 0 until 16) {
                    y[yIdx++] = dl * (((x[qIdx + l].toInt() and 0xFF) shr shift) and 3) - ml
                }

                val sc2 = x[scalesOff + is_].toInt() and 0xFF; is_++
                val dl2 = d * (sc2 and 0xF)
                val ml2 = min * (sc2 shr 4)
                for (l in 0 until 16) {
                    y[yIdx++] = dl2 * (((x[qIdx + 16 + l].toInt() and 0xFF) shr shift) and 3) - ml2
                }
                shift += 2
            }
            qIdx += 32
        }
    }
}

// --- block_q3_K: hmask(32) + qs(64) + scales(12) + d(2) = 110 bytes ---

fun dequantize_row_q3_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockQ3K.SIZE_BYTES

    val kmask1 = 0x03030303
    val kmask2 = 0x0f0f0f0f

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: hmask(32) at 0, qs(64) at 32, scales(12) at 96, d(2) at 108
        val hmOff = bOff + 0
        val qsOff = bOff + 32
        val scOff = bOff + 96
        val dAll = fp16ToF32(x, bOff + 108)

        // Decode scales from 12 bytes into 16 int8 values via aux[4]
        val aux = IntArray(4)
        aux[0] = readIntLE(x, scOff + 0)
        aux[1] = readIntLE(x, scOff + 4)
        aux[2] = readIntLE(x, scOff + 8)
        val tmp = aux[2]
        aux[2] = ((aux[0] shr 4) and kmask2) or (((tmp shr 4) and kmask1) shl 4)
        aux[3] = ((aux[1] shr 4) and kmask2) or (((tmp shr 6) and kmask1) shl 4)
        aux[0] = (aux[0] and kmask2) or (((tmp shr 0) and kmask1) shl 4)
        aux[1] = (aux[1] and kmask2) or (((tmp shr 2) and kmask1) shl 4)

        // Convert aux[4] to scales[16] (int8 values)
        val scales = ByteArray(16)
        for (si in 0 until 4) {
            scales[si * 4 + 0] = (aux[si] and 0xFF).toByte()
            scales[si * 4 + 1] = ((aux[si] shr 8) and 0xFF).toByte()
            scales[si * 4 + 2] = ((aux[si] shr 16) and 0xFF).toByte()
            scales[si * 4 + 3] = ((aux[si] shr 24) and 0xFF).toByte()
        }

        var yIdx = yOff + i * QK_K
        var qIdx = qsOff
        var m = 1
        var is_ = 0
        for (n in 0 until QK_K step 128) {
            var shift = 0
            for (j in 0 until 4) {
                val dl = dAll * (scales[is_].toInt() - 32); is_++
                for (l in 0 until 16) {
                    y[yIdx++] = dl * (((x[qIdx + l].toInt() and 0xFF) shr shift) and 3).let { v ->
                        v - (if ((x[hmOff + l].toInt() and 0xFF) and m != 0) 0 else 4)
                    }
                }
                val dl2 = dAll * (scales[is_].toInt() - 32); is_++
                for (l in 0 until 16) {
                    y[yIdx++] = dl2 * (((x[qIdx + 16 + l].toInt() and 0xFF) shr shift) and 3).let { v ->
                        v - (if ((x[hmOff + 16 + l].toInt() and 0xFF) and m != 0) 0 else 4)
                    }
                }
                shift += 2
                m = m shl 1
            }
            qIdx += 32
        }
    }
}

// --- block_q4_K: d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes ---

fun dequantize_row_q4_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockQ4K.SIZE_BYTES

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: d(2) at 0, dmin(2) at 2, scales(12) at 4, qs(128) at 16
        val d = fp16ToF32(x, bOff)
        val min = fp16ToF32(x, bOff + 2)
        val scOff = bOff + 4
        var qIdx = bOff + 16
        var yIdx = yOff + i * QK_K

        var is_ = 0
        for (j in 0 until QK_K step 64) {
            val sm1 = get_scale_min_k4(is_ + 0, x, scOff)
            val d1 = d * sm1.d; val m1 = min * sm1.m
            val sm2 = get_scale_min_k4(is_ + 1, x, scOff)
            val d2 = d * sm2.d; val m2 = min * sm2.m
            for (l in 0 until 32) {
                y[yIdx++] = d1 * ((x[qIdx + l].toInt() and 0xFF) and 0xF) - m1
            }
            for (l in 0 until 32) {
                y[yIdx++] = d2 * ((x[qIdx + l].toInt() and 0xFF) shr 4) - m2
            }
            qIdx += 32; is_ += 2
        }
    }
}

// --- block_q5_K: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 bytes ---

fun dequantize_row_q5_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockQ5K.SIZE_BYTES

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: d(2) at 0, dmin(2) at 2, scales(12) at 4, qh(32) at 16, qs(128) at 48
        val d = fp16ToF32(x, bOff)
        val min = fp16ToF32(x, bOff + 2)
        val scOff = bOff + 4
        val qhOff = bOff + 16
        var qlIdx = bOff + 48
        var yIdx = yOff + i * QK_K

        var is_ = 0
        var u1 = 1; var u2 = 2
        for (j in 0 until QK_K step 64) {
            val sm1 = get_scale_min_k4(is_ + 0, x, scOff)
            val d1 = d * sm1.d; val m1 = min * sm1.m
            val sm2 = get_scale_min_k4(is_ + 1, x, scOff)
            val d2 = d * sm2.d; val m2 = min * sm2.m
            for (l in 0 until 32) {
                val ql = x[qlIdx + l].toInt() and 0xFF
                val qh = x[qhOff + l].toInt() and 0xFF
                y[yIdx++] = d1 * ((ql and 0xF) + (if (qh and u1 != 0) 16 else 0)) - m1
            }
            for (l in 0 until 32) {
                val ql = x[qlIdx + l].toInt() and 0xFF
                val qh = x[qhOff + l].toInt() and 0xFF
                y[yIdx++] = d2 * ((ql shr 4) + (if (qh and u2 != 0) 16 else 0)) - m2
            }
            qlIdx += 32; is_ += 2
            u1 = u1 shl 2; u2 = u2 shl 2
        }
    }
}

// --- block_q6_K: ql(128) + qh(64) + scales(16) + d(2) = 210 bytes ---

fun dequantize_row_q6_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockQ6K.SIZE_BYTES

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: ql(128) at 0, qh(64) at 128, scales(16) at 192, d(2) at 208
        val d = fp16ToF32(x, bOff + 208)
        var qlIdx = bOff + 0
        var qhIdx = bOff + 128
        var scIdx = bOff + 192
        var yIdx = yOff + i * QK_K

        for (n in 0 until QK_K step 128) {
            for (l in 0 until 32) {
                val isv = l / 16
                val qlLo = x[qlIdx + l].toInt() and 0xFF
                val qlHi = x[qlIdx + l + 32].toInt() and 0xFF
                val qhv = x[qhIdx + l].toInt() and 0xFF
                val q1 = ((qlLo and 0xF) or (((qhv shr 0) and 3) shl 4)) - 32
                val q2 = ((qlHi and 0xF) or (((qhv shr 2) and 3) shl 4)) - 32
                val q3 = ((qlLo shr 4) or (((qhv shr 4) and 3) shl 4)) - 32
                val q4 = ((qlHi shr 4) or (((qhv shr 6) and 3) shl 4)) - 32
                y[yIdx + l + 0]  = d * x[scIdx + isv + 0].toInt() * q1
                y[yIdx + l + 32] = d * x[scIdx + isv + 2].toInt() * q2
                y[yIdx + l + 64] = d * x[scIdx + isv + 4].toInt() * q3
                y[yIdx + l + 96] = d * x[scIdx + isv + 6].toInt() * q4
            }
            yIdx += 128; qlIdx += 64; qhIdx += 32; scIdx += 8
        }
    }
}

// --- block_q8_K: d(4 float) + qs(256) + bsums(32) = 292 bytes ---

fun dequantize_row_q8_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockQ8K.SIZE_BYTES

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: d(4 bytes float) at 0, qs(256) at 4
        val d = Float.fromBits(readIntLE(x, bOff))
        val qsOff = bOff + 4
        for (j in 0 until QK_K) {
            y[yIdx++] = d * x[qsOff + j].toInt()
        }
    }
}

// --- block_tq1_0 ---

fun dequantize_row_tq1_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockTQ1_0.SIZE_BYTES
    val pow3 = intArrayOf(1, 3, 9, 27, 81, 243)

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // block_tq1_0: qs[(QK_K-4*QK_K/64)/5] + qh[QK_K/64] + d(2)
        val qsSize = (QK_K - 4 * QK_K / 64) / 5
        val qhSize = QK_K / 64
        val dOff = bOff + qsSize + qhSize
        val d = fp16ToF32(x, dOff)

        // Process qs in 32-byte chunks
        val fullChunks = qsSize - qsSize % 32
        var qsIdx = bOff
        for (j in 0 until fullChunks step 32) {
            for (n in 0 until 5) {
                for (m in 0 until 32) {
                    val q = ((x[qsIdx + m].toInt() and 0xFF) * pow3[n]) and 0xFF
                    val xi = ((q * 3) shr 8)
                    y[yIdx++] = (xi - 1).toFloat() * d
                }
            }
            qsIdx += 32
        }
        for (j in fullChunks until qsSize step 16) {
            for (n in 0 until 5) {
                for (m in 0 until 16) {
                    val q = ((x[qsIdx + m].toInt() and 0xFF) * pow3[n]) and 0xFF
                    val xi = ((q * 3) shr 8)
                    y[yIdx++] = (xi - 1).toFloat() * d
                }
            }
            qsIdx += 16
        }

        // Process qh
        val qhOff = bOff + qsSize
        for (n in 0 until 4) {
            for (j in 0 until qhSize) {
                val q = ((x[qhOff + j].toInt() and 0xFF) * pow3[n]) and 0xFF
                val xi = ((q * 3) shr 8)
                y[yIdx++] = (xi - 1).toFloat() * d
            }
        }
    }
}

// --- block_tq2_0 ---

fun dequantize_row_tq2_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockTQ2_0.SIZE_BYTES
    val qsSize = QK_K / 4

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val dOff = bOff + qsSize
        val d = fp16ToF32(x, dOff)

        var qsIdx = bOff
        for (j in 0 until qsSize step 32) {
            for (l in 0 until 4) {
                for (m in 0 until 32) {
                    val q = ((x[qsIdx + m].toInt() and 0xFF) shr (l * 2)) and 3
                    y[yIdx++] = (q - 1).toFloat() * d
                }
            }
            qsIdx += 32
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  IQ Dequantize — require grid lookup tables
// ════════════════════════════════════════════════════════════════════════════════

fun dequantize_row_iq4_nl(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK4_NL == 0L)
    val nb = (k / QK4_NL).toInt()
    val blockSize = 2 + QK4_NL / 2  // d(2) + qs(QK4_NL/2)

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = fp16ToF32(x, bOff)
        val qsOff = bOff + 2
        for (j in 0 until QK4_NL / 2) {
            val qByte = x[qsOff + j].toInt() and 0xFF
            y[yIdx + j] = d * kvalues_iq4nl[qByte and 0xf]
            y[yIdx + j + QK4_NL / 2] = d * kvalues_iq4nl[qByte shr 4]
        }
        yIdx += QK4_NL
    }
}

fun dequantize_row_iq4_xs(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockIQ4XS.SIZE_BYTES

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: d(2) at 0, scales_h(2) at 2, scales_l(QK_K/64) at 4, qs(QK_K/2) at 4+QK_K/64
        val d = fp16ToF32(x, bOff)
        val scalesH = readUShortLE(x, bOff + 2)
        val slOff = bOff + 4
        val qsOff = slOff + QK_K / 64

        for (ib in 0 until QK_K / 32) {
            val slByte = x[slOff + ib / 2].toInt() and 0xFF
            val ls = ((slByte shr (4 * (ib % 2))) and 0xf) or (((scalesH shr (2 * ib)) and 3) shl 4)
            val dl = d * (ls - 32)
            val qBase = qsOff + ib * 16
            for (j in 0 until 16) {
                val qByte = x[qBase + j].toInt() and 0xFF
                y[yIdx + j + 0]  = dl * kvalues_iq4nl[qByte and 0xf]
                y[yIdx + j + 16] = dl * kvalues_iq4nl[qByte shr 4]
            }
            yIdx += 32
        }
    }
}

// IQ dequantize functions requiring grid tables — deferred until tables are ported

fun dequantize_row_iq2_xxs(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    TODO("requires iq2xxs_grid table")
}

fun dequantize_row_iq2_xs(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    TODO("requires iq2xs_grid table")
}

fun dequantize_row_iq2_s(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    TODO("requires iq2s_grid table")
}

fun dequantize_row_iq3_xxs(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    TODO("requires iq3xxs_grid table")
}

fun dequantize_row_iq3_s(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    TODO("requires iq3s_grid table")
}

fun dequantize_row_iq1_s(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    TODO("requires iq1s_grid table")
}

fun dequantize_row_iq1_m(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    TODO("requires iq1s_grid table")
}

// ════════════════════════════════════════════════════════════════════════════════
//  Quantize-row-ref functions (reference implementations)
// ════════════════════════════════════════════════════════════════════════════════

fun quantize_row_q4_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = QK4_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk / 2

    for (i in 0 until nb) {
        var amax = 0f
        var max = 0f
        val base = xOff + i * qk
        for (j in 0 until qk) {
            val v = x[base + j]
            if (amax < kotlin.math.abs(v)) { amax = kotlin.math.abs(v); max = v }
        }
        val d = max / -8f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        // Write d as fp16
        val dFp16 = GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val qsOff = bOff + 2
        for (j in 0 until qk / 2) {
            val x0 = x[base + j] * id
            val x1 = x[base + j + qk / 2] * id
            val xi0 = minOf(15, (x0 + 8.5f).toInt())
            val xi1 = minOf(15, (x1 + 8.5f).toInt())
            y[qsOff + j] = (xi0 or (xi1 shl 4)).toByte()
        }
    }
}

fun quantize_row_q8_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = QK8_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk

    for (i in 0 until nb) {
        var amax = 0f
        val base = xOff + i * qk
        for (j in 0 until qk) {
            amax = maxOf(amax, kotlin.math.abs(x[base + j]))
        }
        val d = amax / 127f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val qsOff = bOff + 2
        for (j in 0 until qk) {
            y[qsOff + j] = (x[base + j] * id + 0.5f).toInt().coerceIn(-128, 127).toByte()
        }
    }
}

fun quantize_row_q4_1_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = QK4_1
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 4 + qk / 2

    for (i in 0 until nb) {
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        val base = xOff + i * qk
        for (j in 0 until qk) {
            if (x[base + j] < min) min = x[base + j]
            if (x[base + j] > max) max = x[base + j]
        }
        val d = (max - min) / 15f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val mFp16 = GGML_FP32_TO_FP16(min)
        y[bOff + 2] = (mFp16.toInt() and 0xFF).toByte()
        y[bOff + 3] = ((mFp16.toInt() shr 8) and 0xFF).toByte()
        val qsOff = bOff + 4
        for (j in 0 until qk / 2) {
            val x0 = (x[base + j] - min) * id
            val x1 = (x[base + j + qk / 2] - min) * id
            val xi0 = minOf(15, (x0 + 0.5f).toInt())
            val xi1 = minOf(15, (x1 + 0.5f).toInt())
            y[qsOff + j] = (xi0 or (xi1 shl 4)).toByte()
        }
    }
}

fun quantize_row_q5_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = QK5_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + 4 + qk / 2

    for (i in 0 until nb) {
        var amax = 0f; var max = 0f
        val base = xOff + i * qk
        for (j in 0 until qk) {
            val v = x[base + j]
            if (amax < kotlin.math.abs(v)) { amax = kotlin.math.abs(v); max = v }
        }
        val d = max / -16f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        // qh at bOff+2..bOff+5
        var qh = 0
        val qsOff = bOff + 6
        for (j in 0 until qk / 2) {
            val x0 = x[base + j] * id
            val x1 = x[base + j + qk / 2] * id
            val xi0 = minOf(31, (x0 + 16.5f).toInt())
            val xi1 = minOf(31, (x1 + 16.5f).toInt())
            y[qsOff + j] = ((xi0 and 0x0F) or ((xi1 and 0x0F) shl 4)).toByte()
            qh = qh or (((xi0 and 0x10) shr 4) shl (j + 0))
            qh = qh or (((xi1 and 0x10) shr 4) shl (j + qk / 2))
        }
        y[bOff + 2] = (qh and 0xFF).toByte()
        y[bOff + 3] = ((qh shr 8) and 0xFF).toByte()
        y[bOff + 4] = ((qh shr 16) and 0xFF).toByte()
        y[bOff + 5] = ((qh shr 24) and 0xFF).toByte()
    }
}

fun quantize_row_q5_1_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = QK5_1
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 4 + 4 + qk / 2

    for (i in 0 until nb) {
        var min = Float.MAX_VALUE; var max = -Float.MAX_VALUE
        val base = xOff + i * qk
        for (j in 0 until qk) {
            if (x[base + j] < min) min = x[base + j]
            if (x[base + j] > max) max = x[base + j]
        }
        val d = (max - min) / 31f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val mFp16 = GGML_FP32_TO_FP16(min)
        y[bOff + 2] = (mFp16.toInt() and 0xFF).toByte()
        y[bOff + 3] = ((mFp16.toInt() shr 8) and 0xFF).toByte()
        var qh = 0
        val qsOff = bOff + 8
        for (j in 0 until qk / 2) {
            val x0 = (x[base + j] - min) * id
            val x1 = (x[base + j + qk / 2] - min) * id
            val xi0 = minOf(31, (x0 + 0.5f).toInt())
            val xi1 = minOf(31, (x1 + 0.5f).toInt())
            y[qsOff + j] = ((xi0 and 0x0F) or ((xi1 and 0x0F) shl 4)).toByte()
            qh = qh or (((xi0 and 0x10) shr 4) shl (j + 0))
            qh = qh or (((xi1 and 0x10) shr 4) shl (j + qk / 2))
        }
        y[bOff + 4] = (qh and 0xFF).toByte()
        y[bOff + 5] = ((qh shr 8) and 0xFF).toByte()
        y[bOff + 6] = ((qh shr 16) and 0xFF).toByte()
        y[bOff + 7] = ((qh shr 24) and 0xFF).toByte()
    }
}

fun quantize_row_q8_1_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = QK8_0  // Q8_1 uses same QK as Q8_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    // block_q8_1: d(2) + s(2) + qs(32) = 36 bytes
    val blockSize = 4 + qk

    for (i in 0 until nb) {
        var amax = 0f
        val base = xOff + i * qk
        for (j in 0 until qk) {
            amax = maxOf(amax, kotlin.math.abs(x[base + j]))
        }
        val d = amax / 127f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        var sum = 0
        val qsOff = bOff + 4
        for (j in 0 until qk) {
            val v = (x[base + j] * id + 0.5f).toInt().coerceIn(-128, 127)
            y[qsOff + j] = v.toByte()
            sum += v
        }
        // s = d * sum
        val sFp16 = GGML_FP32_TO_FP16(d * sum)
        y[bOff + 2] = (sFp16.toInt() and 0xFF).toByte()
        y[bOff + 3] = ((sFp16.toInt() shr 8) and 0xFF).toByte()
    }
}

fun quantize_row_q1_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = QK1_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk / 8

    for (i in 0 until nb) {
        val base = xOff + i * qk
        // Find scale: d = max(abs(x))
        var amax = 0f
        for (j in 0 until qk) {
            amax = maxOf(amax, kotlin.math.abs(x[base + j]))
        }
        val d = amax
        val bOff = yOff + i * blockSize
        val dFp16 = GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val qsOff = bOff + 2
        for (j in 0 until qk / 8) y[qsOff + j] = 0
        for (j in 0 until qk) {
            if (x[base + j] > 0f) {
                val byteIdx = j / 8
                val bitOff = j % 8
                y[qsOff + byteIdx] = (y[qsOff + byteIdx].toInt() or (1 shl bitOff)).toByte()
            }
        }
    }
}

// K-quant quantize_row_*_ref — complex implementations, deferred for now

fun quantize_row_q2_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_q2_K_ref: complex K-quant quantization not yet ported")
}

fun quantize_row_q3_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_q3_K_ref: complex K-quant quantization not yet ported")
}

fun quantize_row_q4_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_q4_K_ref: complex K-quant quantization not yet ported")
}

fun quantize_row_q5_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_q5_K_ref: complex K-quant quantization not yet ported")
}

fun quantize_row_q6_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_q6_K_ref: complex K-quant quantization not yet ported")
}

fun quantize_row_q8_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % QK_K == 0L)
    val nb = (k / QK_K).toInt()
    val blockSize = BlockQ8K.SIZE_BYTES

    var xIdx = xOff
    for (i in 0 until nb) {
        var max = 0f
        var amax = 0f
        for (j in 0 until QK_K) {
            val ax = kotlin.math.abs(x[xIdx + j])
            if (ax > amax) { amax = ax; max = ax }
        }
        val bOff = yOff + i * blockSize
        if (amax == 0f) {
            // Write d = 0 as float32 LE
            val dBits = 0f.toRawBits()
            y[bOff + 0] = (dBits and 0xFF).toByte()
            y[bOff + 1] = ((dBits shr 8) and 0xFF).toByte()
            y[bOff + 2] = ((dBits shr 16) and 0xFF).toByte()
            y[bOff + 3] = ((dBits shr 24) and 0xFF).toByte()
            // Zero out qs and bsums
            for (j in 0 until QK_K) y[bOff + 4 + j] = 0
            for (j in 0 until (QK_K / 16) * 2) y[bOff + 4 + QK_K + j] = 0
            xIdx += QK_K
            continue
        }
        val iscale = -128f / max
        val qsOff = bOff + 4
        for (j in 0 until QK_K) {
            val v = nearest_int(iscale * x[xIdx + j])
            y[qsOff + j] = minOf(127, v).toByte()
        }
        // Compute bsums (QK_K/16 int16_t values)
        val bsumsOff = bOff + 4 + QK_K
        for (j in 0 until QK_K / 16) {
            var sum = 0
            for (ii in 0 until 16) {
                sum += y[qsOff + j * 16 + ii].toInt()
            }
            // Write int16 LE
            y[bsumsOff + j * 2 + 0] = (sum and 0xFF).toByte()
            y[bsumsOff + j * 2 + 1] = ((sum shr 8) and 0xFF).toByte()
        }
        // Write d = 1/iscale as float32 LE
        val d = 1f / iscale
        val dBits = d.toRawBits()
        y[bOff + 0] = (dBits and 0xFF).toByte()
        y[bOff + 1] = ((dBits shr 8) and 0xFF).toByte()
        y[bOff + 2] = ((dBits shr 16) and 0xFF).toByte()
        y[bOff + 3] = ((dBits shr 24) and 0xFF).toByte()
        xIdx += QK_K
    }
}

fun quantize_row_tq1_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_tq1_0_ref: ternary quantization not yet ported")
}

fun quantize_row_tq2_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_tq2_0_ref: ternary quantization not yet ported")
}

fun quantize_row_mxfp4_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_mxfp4_ref: MXFP4 quantization not yet ported")
}

fun quantize_row_nvfp4_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_nvfp4_ref: NVFP4 quantization not yet ported")
}

// IQ quantize_row_ref — all require grid tables

fun quantize_row_iq3_xxs_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_iq3_xxs_ref: requires IQ grid tables")
}

fun quantize_row_iq4_nl_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_iq4_nl_ref: requires IQ grid tables")
}

fun quantize_row_iq4_xs_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_iq4_xs_ref: requires IQ grid tables")
}

fun quantize_row_iq3_s_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_iq3_s_ref: requires IQ grid tables")
}

fun quantize_row_iq2_s_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    TODO("quantize_row_iq2_s_ref: requires IQ grid tables")
}

// ════════════════════════════════════════════════════════════════════════════════
//  Importance-matrix quantize functions
// ════════════════════════════════════════════════════════════════════════════════

fun quantize_q1_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q1_0: importance-matrix quantization not yet ported")
}

fun quantize_q4_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q4_0: importance-matrix quantization not yet ported")
}

fun quantize_q4_1(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q4_1: importance-matrix quantization not yet ported")
}

fun quantize_q5_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q5_0: importance-matrix quantization not yet ported")
}

fun quantize_q5_1(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q5_1: importance-matrix quantization not yet ported")
}

fun quantize_q8_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q8_0: importance-matrix quantization not yet ported")
}

fun quantize_q2_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q2_K: importance-matrix quantization not yet ported")
}

fun quantize_q3_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q3_K: importance-matrix quantization not yet ported")
}

fun quantize_q4_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q4_K: importance-matrix quantization not yet ported")
}

fun quantize_q5_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q5_K: importance-matrix quantization not yet ported")
}

fun quantize_q6_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_q6_K: importance-matrix quantization not yet ported")
}

fun quantize_tq1_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_tq1_0: importance-matrix quantization not yet ported")
}

fun quantize_tq2_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_tq2_0: importance-matrix quantization not yet ported")
}

fun quantize_mxfp4(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_mxfp4: importance-matrix quantization not yet ported")
}

fun quantize_nvfp4(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_nvfp4: importance-matrix quantization not yet ported")
}

fun quantize_iq2_xxs(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_iq2_xxs: requires IQ grid tables")
}

fun quantize_iq2_xs(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_iq2_xs: requires IQ grid tables")
}

fun quantize_iq2_s(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_iq2_s: requires IQ grid tables")
}

fun quantize_iq3_xxs(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_iq3_xxs: requires IQ grid tables")
}

fun quantize_iq1_s(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_iq1_s: requires IQ grid tables")
}

fun quantize_iq1_m(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_iq1_m: requires IQ grid tables")
}

fun quantize_iq4_nl(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_iq4_nl: importance-matrix quantization not yet ported")
}

fun quantize_iq4_xs(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_iq4_xs: importance-matrix quantization not yet ported")
}

fun quantize_iq3_s(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    TODO("quantize_iq3_s: requires IQ grid tables")
}

// IQ init/free

fun iq2xs_init_impl(type: GGMLType) {
    TODO("iq2xs_init_impl: requires IQ grid table initialization")
}

fun iq2xs_free_impl(type: GGMLType) {
    TODO("iq2xs_free_impl: requires IQ grid table cleanup")
}

fun iq3xs_init_impl(gridSize: Int) {
    TODO("iq3xs_init_impl: requires IQ grid table initialization")
}

fun iq3xs_free_impl(gridSize: Int) {
    TODO("iq3xs_free_impl: requires IQ grid table cleanup")
}
