package ai.solace.llamakotlin.core.simd

import ai.solace.llamakotlin.core.ByteArrayExtensions.getFloatLe
import ai.solace.llamakotlin.core.ByteArrayExtensions.getShortLe
import ai.solace.llamakotlin.core.GGML_FP16_TO_FP32
import ai.solace.llamakotlin.core.GGML_FP32_TO_FP16
import ai.solace.llamakotlin.core.halfToFloat

// port-lint: source ggml/src/ggml-cpu/simd-mappings.h
// The C header defines SIMD macros per-architecture (ARM NEON, AVX, SVE, etc.).
// In Kotlin/Native we only have the scalar fallback path.

/** Scalar fallback for GGML_CPU_FP16_TO_FP32 — on x86 this is a lookup table, here we just convert. */
inline fun GGML_CPU_FP16_TO_FP32(x: Short): Float = GGML_FP16_TO_FP32(x)

/** Scalar fallback for GGML_CPU_FP32_TO_FP16. */
inline fun GGML_CPU_FP32_TO_FP16(x: Float): Short = GGML_FP32_TO_FP16(x)

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
