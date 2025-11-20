package ai.solace.llamakotlin.core

import ai.solace.klang.fp.CFloat32
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

internal data class QuantizationStats(val scale: Float, val min: Float)

private const val GROUP_MAX_EPS = 1e-15f

internal fun nearestIntFloat(value: Float): Int {
    if (!value.isFinite()) return 0
    if (value > 4_194_303.0f || value < -4_194_303.0f) {
        return value.roundToInt()
    }
    val adjusted = value + 12_582_912.0f
    val bits = adjusted.toRawBits()
    return (bits and 0x007F_FFFF) - 0x0040_0000
}

private inline fun FloatArray.weightAt(offset: Int, index: Int): Float = this[offset + index]

private fun qkx3Weight(
    values: FloatArray,
    valuesOffset: Int,
    weights: FloatArray?,
    weightsOffset: Int,
    index: Int
): Float {
    return weights?.weightAt(weightsOffset, index) ?: run {
        val xi = values[valuesOffset + index]
        xi * xi
    }
}

private fun computeRmseWeight(
    values: FloatArray,
    valuesOffset: Int,
    quantWeights: FloatArray?,
    weightsOffset: Int,
    index: Int,
    rmseType: Int
): Float {
    quantWeights?.let { return it[weightsOffset + index] }
    val xi = values[valuesOffset + index]
    return when (rmseType) {
        1 -> xi * xi
        2 -> 1f
        3 -> abs(xi)
        else -> sqrt(abs(xi))
    }
}

internal fun makeQ3Quants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    doRmse: Boolean
): Float {
    var maxValue = 0f
    var absMax = 0f
    for (i in 0 until n) {
        val xi = values[valuesOffset + i]
        val ax = abs(xi)
        if (ax > absMax) {
            absMax = ax
            maxValue = xi
        }
    }

    if (absMax < GROUP_MAX_EPS) {
        for (i in 0 until n) dest[destOffset + i] = 0
        return 0f
    }

    val codes = IntArray(n)
    var iscale = -nmax.toFloat() / maxValue

    if (doRmse) {
        var sumLx = 0f
        var sumL2 = 0f
        for (i in 0 until n) {
            val xi = values[valuesOffset + i]
            var l = nearestIntFloat(iscale * xi)
            l = l.coerceIn(-nmax, nmax - 1)
            codes[i] = l
            val w = xi * xi
            sumLx += w * xi * l
            sumL2 += w * l * l
        }

        repeat(5) {
            var changed = 0
            for (i in 0 until n) {
                val xi = values[valuesOffset + i]
                val w = xi * xi
                val current = codes[i]
                val slx = sumLx - w * xi * current
                if (slx > 0f) {
                    val sl2 = sumL2 - w * current * current
                    var newCode = nearestIntFloat(xi * sl2 / slx)
                    newCode = newCode.coerceIn(-nmax, nmax - 1)
                    if (newCode != current) {
                        val candidateSlx = slx + w * xi * newCode
                        val candidateSl2 = sl2 + w * newCode * newCode
                        if (candidateSl2 > 0f && candidateSlx * candidateSlx * sumL2 > sumLx * sumLx * candidateSl2) {
                            codes[i] = newCode
                            sumLx = candidateSlx
                            sumL2 = candidateSl2
                            changed++
                        }
                    }
                }
            }
            if (changed == 0) return@repeat
        }

        for (i in 0 until n) {
            dest[destOffset + i] = (codes[i] + nmax).toByte()
        }
        return if (sumL2 > 0f) sumLx / sumL2 else 0f
    }

    for (i in 0 until n) {
        val xi = values[valuesOffset + i]
        var l = nearestIntFloat(iscale * xi)
        l = l.coerceIn(-nmax, nmax - 1)
        dest[destOffset + i] = (l + nmax).toByte()
    }
    return 1f / iscale
}

internal fun makeQKX2Quants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    weights: FloatArray,
    weightsOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    mins: FloatArray,
    minsIndex: Int,
    aux: ByteArray,
    auxOffset: Int,
    rmin: Float,
    rdelta: Float,
    nstep: Int,
    useMad: Boolean
): QuantizationStats {
    var minVal = values[valuesOffset]
    var maxVal = minVal
    var sumW = CFloat32.fromFloat(weights.weightAt(weightsOffset, 0))
    var sumX = sumW * values[valuesOffset]
    for (i in 1 until n) {
        val xi = values[valuesOffset + i]
        if (xi < minVal) minVal = xi
        if (xi > maxVal) maxVal = xi
        val w = CFloat32.fromFloat(weights.weightAt(weightsOffset, i))
        sumW = sumW + w
        sumX = sumX + w * xi
    }
    if (minVal > 0f) minVal = 0f
    if (maxVal == minVal) {
        for (i in 0 until n) dest[destOffset + i] = 0
        mins[minsIndex] = -minVal
        return QuantizationStats(scale = 0f, min = -minVal)
    }

    val maxFixed = maxVal
    var currentMin = minVal
    var iscale = nmax.toFloat() / (maxFixed - currentMin)
    var scale = 1f / iscale
    var bestMetric = 0f
    for (i in 0 until n) {
        val xi = values[valuesOffset + i]
        val l = nearestIntFloat(iscale * (xi - currentMin)).coerceIn(0, nmax)
        dest[destOffset + i] = l.toByte()
        var diff = scale * l + currentMin - xi
        diff = if (useMad) abs(diff) else diff * diff
        val w = weights.weightAt(weightsOffset, i)
        bestMetric += w * diff
    }
    if (nstep < 1) {
        mins[minsIndex] = -currentMin
        return QuantizationStats(scale = scale, min = -currentMin)
    }

    for (step in 0..nstep) {
        val stepIscale = (rmin + rdelta * step + nmax) / (maxFixed - currentMin)
        var sumL = CFloat32.fromFloat(0f)
        var sumL2 = CFloat32.fromFloat(0f)
        var sumXL = CFloat32.fromFloat(0f)
        for (i in 0 until n) {
            val xi = values[valuesOffset + i]
            var l = nearestIntFloat(stepIscale * (xi - currentMin))
            l = l.coerceIn(0, nmax)
            aux[auxOffset + i] = l.toByte()
            val w = CFloat32.fromFloat(weights.weightAt(weightsOffset, i))
            val lf = CFloat32.fromFloat(l.toFloat())
            sumL = sumL + (w * lf)
            sumL2 = sumL2 + (w * lf * lf)
            sumXL = sumXL + (w * lf * xi)
        }
        val D = sumW * sumL2 - sumL * sumL
        if (D.toFloat() > 0f) {
            var scaleCand = (sumW * sumXL - sumX * sumL) / D
            var minCand = (sumL2 * sumX - sumL * sumXL) / D
            if (minCand.toFloat() > 0f) {
                minCand = CFloat32.fromFloat(0f)
                scaleCand = sumXL / sumL2
            }
            var metric = CFloat32.fromFloat(0f)
            for (i in 0 until n) {
                val xi = CFloat32.fromFloat(values[valuesOffset + i])
                val li = CFloat32.fromFloat((aux[auxOffset + i].toInt() and 0xFF).toFloat())
                var diff = scaleCand * li + minCand - xi
                diff = if (useMad) CFloat32.fromFloat(abs(diff.toFloat())) else diff * diff
                val w = CFloat32.fromFloat(weights.weightAt(weightsOffset, i))
                metric = metric + w * diff
            }
            if (metric.toFloat() < bestMetric) {
                for (i in 0 until n) dest[destOffset + i] = aux[auxOffset + i]
                bestMetric = metric.toFloat()
                scale = scaleCand.toFloat()
                currentMin = minCand.toFloat()
            }
        }
    }

    mins[minsIndex] = -currentMin
    return QuantizationStats(scale = scale, min = -currentMin)
}

internal fun makeQKX3Quants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    weights: FloatArray?,
    weightsOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    mins: FloatArray,
    minsIndex: Int,
    aux: ByteArray,
    auxOffset: Int,
    rmin: Float,
    rdelta: Float,
    nstep: Int,
    useMad: Boolean
): QuantizationStats {
    var minVal = values[valuesOffset]
    var maxVal = minVal
    var sumW = CFloat32.fromFloat(qkx3Weight(values, valuesOffset, weights, weightsOffset, 0))
    var sumX = sumW * values[valuesOffset]
    for (i in 1 until n) {
        val xi = values[valuesOffset + i]
        if (xi < minVal) minVal = xi
        if (xi > maxVal) maxVal = xi
        val w = CFloat32.fromFloat(qkx3Weight(values, valuesOffset, weights, weightsOffset, i))
        sumW = sumW + w
        sumX = sumX + w * xi
    }
    if (minVal > 0f) minVal = 0f
    if (maxVal <= minVal) {
        for (i in 0 until n) dest[destOffset + i] = 0
        mins[minsIndex] = -minVal
        return QuantizationStats(scale = 0f, min = -minVal)
    }

    val maxFixed = maxVal
    var currentMin = minVal
    var iscale = nmax.toFloat() / (maxFixed - currentMin)
    var scale = 1f / iscale
    var bestMetric = 0f
    for (i in 0 until n) {
        val xi = values[valuesOffset + i]
        val l = nearestIntFloat(iscale * (xi - currentMin)).coerceIn(0, nmax)
        dest[destOffset + i] = l.toByte()
        var diff = scale * l + currentMin - xi
        diff = if (useMad) abs(diff) else diff * diff
        val w = qkx3Weight(values, valuesOffset, weights, weightsOffset, i)
        bestMetric += w * diff
    }
    if (nstep < 1) {
        mins[minsIndex] = -currentMin
        return QuantizationStats(scale = scale, min = -currentMin)
    }

    for (step in 0..nstep) {
        val stepIscale = (rmin + rdelta * step + nmax) / (maxFixed - currentMin)
        var sumL = CFloat32.fromFloat(0f)
        var sumL2 = CFloat32.fromFloat(0f)
        var sumXL = CFloat32.fromFloat(0f)
        for (i in 0 until n) {
            val xi = values[valuesOffset + i]
            var l = nearestIntFloat(stepIscale * (xi - currentMin))
            l = l.coerceIn(0, nmax)
            aux[auxOffset + i] = l.toByte()
            val w = CFloat32.fromFloat(qkx3Weight(values, valuesOffset, weights, weightsOffset, i))
            val lf = CFloat32.fromFloat(l.toFloat())
            sumL = sumL + (w * lf)
            sumL2 = sumL2 + (w * lf * lf)
            sumXL = sumXL + (w * lf * xi)
        }
        val D = sumW * sumL2 - sumL * sumL
        if (D.toFloat() > 0f) {
            var scaleCand = (sumW * sumXL - sumX * sumL) / D
            var minCand = (sumL2 * sumX - sumL * sumXL) / D
            if (minCand.toFloat() > 0f) {
                minCand = CFloat32.fromFloat(0f)
                scaleCand = sumXL / sumL2
            }
            var metric = CFloat32.fromFloat(0f)
            for (i in 0 until n) {
                val xi = CFloat32.fromFloat(values[valuesOffset + i])
                val li = CFloat32.fromFloat((aux[auxOffset + i].toInt() and 0xFF).toFloat())
                var diff = scaleCand * li + minCand - xi
                diff = if (useMad) CFloat32.fromFloat(abs(diff.toFloat())) else diff * diff
                val w = CFloat32.fromFloat(qkx3Weight(values, valuesOffset, weights, weightsOffset, i))
                metric = metric + w * diff
            }
            if (metric.toFloat() < bestMetric) {
                for (i in 0 until n) dest[destOffset + i] = aux[auxOffset + i]
                bestMetric = metric.toFloat()
                scale = scaleCand.toFloat()
                currentMin = minCand.toFloat()
            }
        }
    }

    mins[minsIndex] = -currentMin
    return QuantizationStats(scale = scale, min = -currentMin)
}

internal fun makeQPQuants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    quantWeights: FloatArray,
    weightsOffset: Int
): Float {
    var maxVal = 0f
    for (i in 0 until n) {
        maxVal = max(maxVal, values[valuesOffset + i])
    }
    if (maxVal == 0f) {
        for (i in 0 until n) dest[destOffset + i] = 0
        return 0f
    }
    var iscale = nmax.toFloat() / maxVal
    val tempCodes = IntArray(n)
    for (i in 0 until n) {
        val l = nearestIntFloat(iscale * values[valuesOffset + i]).coerceIn(0, nmax)
        dest[destOffset + i] = l.toByte()
        tempCodes[i] = l
    }
    var scale = 1f / iscale
    var bestMse = 0f
    for (i in 0 until n) {
        val diff = values[valuesOffset + i] - scale * tempCodes[i]
        bestMse += quantWeights[weightsOffset + i] * diff * diff
    }
    for (adjust in -4..4) {
        if (adjust == 0) continue
        val candidateIscale = (0.1f * adjust + nmax) / maxVal
        val candidateScale = 1f / candidateIscale
        var mse = 0f
        for (i in 0 until n) {
            var l = nearestIntFloat(candidateIscale * values[valuesOffset + i])
            l = min(nmax, l)
            val diff = values[valuesOffset + i] - candidateScale * l
            mse += quantWeights[weightsOffset + i] * diff * diff
        }
        if (mse < bestMse) {
            bestMse = mse
            iscale = candidateIscale
        }
    }

    var sumLx = 0f
    var sumL2 = 0f
    for (i in 0 until n) {
        var l = nearestIntFloat(iscale * values[valuesOffset + i])
        l = min(nmax, l)
        dest[destOffset + i] = l.toByte()
        val w = quantWeights[weightsOffset + i]
        sumLx += w * values[valuesOffset + i] * l
        sumL2 += w * l * l
    }
    for (iteration in 0 until 5) {
        var changed = 0
        for (i in 0 until n) {
            val w = quantWeights[weightsOffset + i]
            val xi = values[valuesOffset + i]
            val current = dest[destOffset + i].toInt() and 0xFF
            var slx = sumLx - w * xi * current
            var sl2 = sumL2 - w * current * current
            if (slx > 0f && sl2 > 0f) {
                var newL = nearestIntFloat(xi * sl2 / slx)
                newL = min(nmax, newL)
                if (newL != current) {
                    slx += w * xi * newL
                    sl2 += w * newL * newL
                    if (slx * slx * sumL2 > sumLx * sumLx * sl2) {
                        dest[destOffset + i] = newL.toByte()
                        sumLx = slx
                        sumL2 = sl2
                        changed++
                    }
                }
            }
        }
        if (changed == 0) break
    }
    return if (sumL2 > 0f) sumLx / sumL2 else 0f
}

internal fun makeQXQuants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    rmseTypeInput: Int,
    quantWeights: FloatArray?,
    weightsOffset: Int
): Float {
    var maxValue = 0f
    var absMax = 0f
    for (i in 0 until n) {
        val ax = abs(values[valuesOffset + i])
        if (ax > absMax) {
            absMax = ax
            maxValue = values[valuesOffset + i]
        }
    }
    if (absMax < GROUP_MAX_EPS) {
        for (i in 0 until n) dest[destOffset + i] = 0
        return 0f
    }
    var iscale = -nmax.toFloat() / maxValue
    var rmseType = rmseTypeInput
    if (rmseType == 0) {
        for (i in 0 until n) {
            var l = nearestIntFloat(iscale * values[valuesOffset + i])
            l = max(-nmax, min(nmax - 1, l))
            dest[destOffset + i] = (nmax + l).toByte()
        }
        return 1f / iscale
    }
    var returnEarly = false
    if (rmseType < 0) {
        rmseType = -rmseType
        returnEarly = true
    }

    var sumLx = 0f
    var sumL2 = 0f
    for (i in 0 until n) {
        var l = nearestIntFloat(iscale * values[valuesOffset + i])
        l = max(-nmax, min(nmax - 1, l))
        dest[destOffset + i] = (nmax + l).toByte()
        val w = computeRmseWeight(values, valuesOffset, quantWeights, weightsOffset, i, rmseType)
        sumLx += w * values[valuesOffset + i] * l
        sumL2 += w * l * l
    }
    var scale = if (sumL2 > 0f) sumLx / sumL2 else 0f
    if (returnEarly) {
        return if (sumL2 > 0f) 0.5f * (scale + 1f / iscale) else 1f / iscale
    }
    var best = scale * sumLx

    for (adjust in -9..9) {
        if (adjust == 0) continue
        val candidateIscale = -(nmax + 0.1f * adjust) / maxValue
        var candidateSumLx = 0f
        var candidateSumL2 = 0f
        for (i in 0 until n) {
            val l = nearestIntFloat(candidateIscale * values[valuesOffset + i]).coerceIn(-nmax, nmax - 1)
            val w = computeRmseWeight(values, valuesOffset, quantWeights, weightsOffset, i, rmseType)
            candidateSumLx += w * values[valuesOffset + i] * l
            candidateSumL2 += w * l * l
        }
        if (candidateSumL2 > 0f && candidateSumLx * candidateSumLx > best * candidateSumL2) {
            iscale = candidateIscale
            scale = candidateSumLx / candidateSumL2
            best = scale * candidateSumLx
            for (i in 0 until n) {
                val l = nearestIntFloat(iscale * values[valuesOffset + i]).coerceIn(-nmax, nmax - 1)
                dest[destOffset + i] = (nmax + l).toByte()
            }
        }
    }

    return scale
}
