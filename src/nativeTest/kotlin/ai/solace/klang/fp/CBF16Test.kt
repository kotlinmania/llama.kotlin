package ai.solace.klang.fp

import ai.solace.klang.bitwise.Float32Math
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CBF16Test {
    private fun f2bf16bits(f: Float): Short {
        val bits = f.toRawBits()
        val roundBias = 0x7FFF + (((bits ushr 16) and 1))
        val rounded = bits + roundBias
        return (rounded ushr 16).toShort()
    }

    @Test
    fun convertEdges() {
        val zeros = arrayOf(0.0f, -0.0f)
        for (z in zeros) {
            val bf = CBF16.fromFloat(z)
            assertEquals(z.toRawBits(), bf.toFloat().toRawBits())
        }

        val infs = arrayOf(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)
        for (x in infs) {
            val bf = CBF16.fromFloat(x)
            assertTrue(bf.isInf())
            assertEquals(x.toRawBits(), bf.toFloat().toRawBits())
        }

        val nan = Float.NaN
        val bfNan = CBF16.fromFloat(nan)
        assertTrue(bfNan.isNaN())
    }

    @Test
    fun roundTripBasics() {
        val vals = floatArrayOf(1.0f, -1.0f, 0.5f, -0.5f, 3.1415926f, -123.456f)
        for (v in vals) {
            val bf = CBF16.fromFloat(v)
            val back = bf.toFloat()
            // Reference rounding:
            val refBits = f2bf16bits(v).toInt() and 0xFFFF
            val refFloat = Float.fromBits(refBits shl 16)
            assertEquals(refFloat.toRawBits(), back.toRawBits(), "v=$v")
        }
    }

    @Test
    fun arithmeticViaF32() {
        val a = CBF16.fromFloat(1.5f)
        val b = CBF16.fromFloat(2.75f)
        val sum = (a + b).toFloat()
        val refSum = Float32Math.add(1.5f, 2.75f)
        // compare as bf16-rounded
        assertEquals(f2bf16bits(refSum).toInt(), f2bf16bits(sum).toInt())

        val prod = (a * b).toFloat()
        val refProd = Float32Math.mul(1.5f, 2.75f)
        assertEquals(f2bf16bits(refProd).toInt(), f2bf16bits(prod).toInt())

        val div = (a / b).toFloat()
        val refDiv = Float32Math.div(1.5f, 2.75f)
        assertEquals(f2bf16bits(refDiv).toInt(), f2bf16bits(div).toInt())

        val s = a.sqrt().toFloat()
        val refS = Float.fromBits(Float32Math.sqrtBits(1.5f.toRawBits()))
        assertEquals(f2bf16bits(refS).toInt(), f2bf16bits(s).toInt())
    }
}

