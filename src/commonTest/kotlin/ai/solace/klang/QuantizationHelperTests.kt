package ai.solace.klangnative

import ai.solace.llamakotlin.core.makeQKX2Quants
import ai.solace.llamakotlin.core.makeQKX3Quants
import ai.solace.llamakotlin.core.makeQPQuants
import ai.solace.llamakotlin.core.makeQXQuants
import ai.solace.llamakotlin.core.nearestIntFloat
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.math.max
import kotlin.math.min
import kotlin.test.assertEquals

class QuantizationHelperTests {
    @Test
    fun makeQKX2QuantsSimple() {
        val values = floatArrayOf(0f, 1f, 2f, 3f)
        val weights = FloatArray(values.size) { 1f }
        val dest = ByteArray(values.size)
        val mins = FloatArray(1)
        val aux = ByteArray(values.size)

        val stats = makeQKX2Quants(
            n = values.size,
            nmax = 3,
            values = values,
            valuesOffset = 0,
            weights = weights,
            weightsOffset = 0,
            dest = dest,
            destOffset = 0,
            mins = mins,
            minsIndex = 0,
            aux = aux,
            auxOffset = 0,
            rmin = -0.5f,
            rdelta = 0.1f,
            nstep = 0,
            useMad = false
        )

        assertContentEquals(byteArrayOf(0, 1, 2, 3), dest.copyOf(values.size))
        assertEquals(1f, stats.scale, 1e-6f)
        assertEquals(0f, stats.min, 1e-6f)
        assertEquals(0f, mins[0], 1e-6f)
    }

    @Test
    fun makeQKX3QuantsSimple() {
        val values = floatArrayOf(-1f, 0f, 1f, 2f)
        val dest = ByteArray(values.size)
        val mins = FloatArray(1)
        val aux = ByteArray(values.size)

        val stats = makeQKX3Quants(
            n = values.size,
            nmax = 7,
            values = values,
            valuesOffset = 0,
            weights = null,
            weightsOffset = 0,
            dest = dest,
            destOffset = 0,
            mins = mins,
            minsIndex = 0,
            aux = aux,
            auxOffset = 0,
            rmin = -0.9f,
            rdelta = 0.05f,
            nstep = 0,
            useMad = false
        )

        assertEquals(true, stats.min >= 0f)
        val codes = dest.map { it.toInt() and 0xFF }
        assertEquals(true, codes.all { it in 0..7 })
    }

    @Test
    fun makeQPQuantsSimple() {
        val values = floatArrayOf(0f, 1f, 2f, 3f)
        val weights = FloatArray(values.size) { 1f }
        val dest = ByteArray(values.size)

        val scale = makeQPQuants(
            n = values.size,
            nmax = 3,
            values = values,
            valuesOffset = 0,
            dest = dest,
            destOffset = 0,
            quantWeights = weights,
            weightsOffset = 0
        )

        assertEquals(1f, scale, 1e-6f)
        assertContentEquals(byteArrayOf(0, 1, 2, 3), dest)
    }

    @Test
    fun makeQXQuantsRmseZero() {
        val values = floatArrayOf(-3f, -1f, 0f, 1f, 3f)
        val dest = ByteArray(values.size)
        val scale = makeQXQuants(
            n = values.size,
            nmax = 4,
            values = values,
            valuesOffset = 0,
            dest = dest,
            destOffset = 0,
            rmseTypeInput = 0,
            quantWeights = null,
            weightsOffset = 0
        )
        assertEquals(0.75f, scale, 1e-6f)
        val expectedCodes = run {
            val maxValue = values.maxBy { kotlin.math.abs(it) }
            val iscale = -4f / maxValue
            values.map { value ->
                var l = nearestIntFloat(iscale * value)
                l = max(-4, min(3, l))
                4 + l
            }
        }
        val actualCodes = dest.map { it.toInt() and 0xFF }
        assertEquals(expectedCodes, actualCodes)
    }
}
