package ai.solace.llamakotlin.core.simd

import ai.solace.llamakotlin.core.ByteArrayExtensions.getFloatLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getShortLe
import ai.solace.llamakotlin.core.halfToFloat

internal object GGMLSimd {

    private const val UNROLL = 8

    fun dotF32(
        dataA: ByteArray,
        baseOffsetA: Int,
        strideABytes: Int,
        dataB: ByteArray,
        baseOffsetB: Int,
        strideBBytes: Int,
        length: Int
    ): Float {
        var sum = 0.0f
        var idx = 0
        val limit = length - length % UNROLL
        var offsetA = baseOffsetA
        var offsetB = baseOffsetB

        while (idx < limit) {
            sum += dataA.getFloatLe(offsetA) * dataB.getFloatLe(offsetB)
            sum += dataA.getFloatLe(offsetA + strideABytes) * dataB.getFloatLe(offsetB + strideBBytes)
            sum += dataA.getFloatLe(offsetA + 2 * strideABytes) * dataB.getFloatLe(offsetB + 2 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 3 * strideABytes) * dataB.getFloatLe(offsetB + 3 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 4 * strideABytes) * dataB.getFloatLe(offsetB + 4 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 5 * strideABytes) * dataB.getFloatLe(offsetB + 5 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 6 * strideABytes) * dataB.getFloatLe(offsetB + 6 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 7 * strideABytes) * dataB.getFloatLe(offsetB + 7 * strideBBytes)

            idx += UNROLL
            offsetA += UNROLL * strideABytes
            offsetB += UNROLL * strideBBytes
        }

        while (idx < length) {
            sum += dataA.getFloatLe(offsetA) * dataB.getFloatLe(offsetB)
            idx += 1
            offsetA += strideABytes
            offsetB += strideBBytes
        }

        return sum
    }

    fun dotF16(
        dataA: ByteArray,
        baseOffsetA: Int,
        strideABytes: Int,
        dataB: ByteArray,
        baseOffsetB: Int,
        strideBBytes: Int,
        length: Int
    ): Float {
        var sum = 0.0f
        var idx = 0
        val limit = length - length % UNROLL
        var offsetA = baseOffsetA
        var offsetB = baseOffsetB

        while (idx < limit) {
            sum += halfToFloat(dataA.getShortLe(offsetA)) * halfToFloat(dataB.getShortLe(offsetB))
            sum += halfToFloat(dataA.getShortLe(offsetA + strideABytes)) * halfToFloat(dataB.getShortLe(offsetB + strideBBytes))
            sum += halfToFloat(dataA.getShortLe(offsetA + 2 * strideABytes)) * halfToFloat(dataB.getShortLe(offsetB + 2 * strideBBytes))
            sum += halfToFloat(dataA.getShortLe(offsetA + 3 * strideABytes)) * halfToFloat(dataB.getShortLe(offsetB + 3 * strideBBytes))
            sum += halfToFloat(dataA.getShortLe(offsetA + 4 * strideABytes)) * halfToFloat(dataB.getShortLe(offsetB + 4 * strideBBytes))
            sum += halfToFloat(dataA.getShortLe(offsetA + 5 * strideABytes)) * halfToFloat(dataB.getShortLe(offsetB + 5 * strideBBytes))
            sum += halfToFloat(dataA.getShortLe(offsetA + 6 * strideABytes)) * halfToFloat(dataB.getShortLe(offsetB + 6 * strideBBytes))
            sum += halfToFloat(dataA.getShortLe(offsetA + 7 * strideABytes)) * halfToFloat(dataB.getShortLe(offsetB + 7 * strideBBytes))

            idx += UNROLL
            offsetA += UNROLL * strideABytes
            offsetB += UNROLL * strideBBytes
        }

        while (idx < length) {
            sum += halfToFloat(dataA.getShortLe(offsetA)) * halfToFloat(dataB.getShortLe(offsetB))
            idx += 1
            offsetA += strideABytes
            offsetB += strideBBytes
        }

        return sum
    }
}
