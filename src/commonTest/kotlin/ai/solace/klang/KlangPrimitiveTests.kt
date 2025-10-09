package ai.solace.klang

import ai.solace.klang.bitwise.BitPrimitives
import ai.solace.klang.bitwise.Float32Math
import ai.solace.klang.bitwise.PackOps
import ai.solace.klang.buffer.MemoryOps
import ai.solace.klang.common.StatOps
import ai.solace.klang.fp.VectorOps
import ai.solace.klang.int.hpc.HPC16x8
import kotlin.ExperimentalUnsignedTypes
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@OptIn(ExperimentalUnsignedTypes::class)
class KlangPrimitiveTests {
    @Test
    fun bitPrimitiveBasics() {
        assertEquals(1, BitPrimitives.clz32(0x40000000))
        assertEquals(0, BitPrimitives.ctz32(0x1))
        assertEquals(32, BitPrimitives.clz32(0))
        assertEquals(4, BitPrimitives.popcount32(0b101101))
        val rotl = BitPrimitives.rotl32(0x00010001, 1)
        assertEquals(0x00020002, rotl)
        assertEquals(0x80008000.toInt(), BitPrimitives.rotr32(0x00010001, 1))
        val packed = BitPrimitives.bitFieldInsert32(0, 0b1011, 4, 4)
        assertEquals(0b10110000, packed)
        assertEquals(0b1011, BitPrimitives.bitFieldExtract32(packed, 4, 4))
    }

    @Test
    fun packOpsHelpers() {
        val packed = PackOps.packNibbles(0x3, 0xA)
        assertEquals(0xA3, packed)
        assertEquals(0x3, PackOps.unpackLowNibble(packed))
        assertEquals(0xA, PackOps.unpackHighNibble(packed))
        val quads = PackOps.packQuads(1, 2, 3, 0)
        assertEquals(1, PackOps.unpackQuad(quads, 0))
        assertEquals(2, PackOps.unpackQuad(quads, 1))
        assertEquals(3, PackOps.unpackQuad(quads, 2))
        assertEquals(0, PackOps.unpackQuad(quads, 3))
        val bitplane = PackOps.bitplaneWrite(0, 0b11, 2, 2)
        assertEquals(0b001100, bitplane)
        assertEquals(0b11, PackOps.bitplaneRead(bitplane, 2, 2))
    }

    @Test
    fun memoryOpsBehave() {
        val bytes = ByteArray(8)
        MemoryOps.memset(bytes, 0, bytes.size, 0x7F)
        bytes.forEach { assertEquals(0x7F, it.toInt() and 0xFF) }
        val other = ByteArray(8) { it.toByte() }
        MemoryOps.memcpy(bytes, 0, other, 0, other.size)
        bytes.forEachIndexed { idx, value -> assertEquals(idx, value.toInt() and 0xFF) }
        // overlapping copy moves backwards
        MemoryOps.memcpy(bytes, 2, bytes, 0, 6)
        val expected = byteArrayOf(0, 1, 0, 1, 2, 3, 4, 5)
        expected.forEachIndexed { index, b -> assertEquals(b.toInt() and 0xFF, bytes[index].toInt() and 0xFF) }
        val cmp = MemoryOps.memcmp(bytes, 0, expected, 0, expected.size)
        assertEquals(0, cmp)
    }

    @Test
    fun vectorOpsDotAccumulate() {
        val lhs = floatArrayOf(1f, 2f, 3f, 4f)
        val rhs = floatArrayOf(0.5f, -1f, 2f, 0f)
        val dot = VectorOps.dotAccumulate(lhs.size, lhs, 0, rhs, 0)
        assertEquals(1f * 0.5f + 2f * -1f + 3f * 2f + 4f * 0f, dot)
    }

    @Test
    fun statOpsBasics() {
        val values = floatArrayOf(-2f, -1f, 0f, 1f, 2f)
        val mean = StatOps.mean(values)
        assertEquals(0f, mean)
        val varVal = StatOps.variance(values, mean = mean)
        assertEquals(2f, varVal)
        val mad = StatOps.meanAbsoluteDeviation(values, mean = mean)
        assertEquals(1.2f, mad, 1e-6f)
        val random = Random(1234)
        val r = StatOps.randomUniform(random, -1f, 1f)
        assertTrue(r in -1f..1f)
    }

    @Test
    fun floatMathHelpers() {
        val fma = Float32Math.fma(1.5f, 4f, -6f)
        assertEquals(0f, fma)
        assertEquals(3L, Float32Math.lrint(2.8f))
        assertEquals(-2L, Float32Math.lrint(-2.4f))
        assertEquals(-2L, Float32Math.lrint(-1.5f))
        assertEquals(2f, Float32Math.nearbyint(2.4f))
        assertEquals(-2f, Float32Math.nearbyint(-2.4f))
    }

    @Test
    fun hpcWideArithmetic() {
        fun uint128Of(hi: ULong, lo: ULong): HPC16x8 {
            val limbs = UShortArray(8)
            var tmp = lo
            for (i in 0..3) {
                limbs[i] = (tmp and 0xFFFFu).toUShort()
                tmp = tmp shr 16
            }
            tmp = hi
            for (i in 4..7) {
                limbs[i] = (tmp and 0xFFFFu).toUShort()
                tmp = tmp shr 16
            }
            return HPC16x8.ofLimbsLE(limbs[0], limbs[1], limbs[2], limbs[3], limbs[4], limbs[5], limbs[6], limbs[7])
        }
        fun HPC16x8.toPair(): Pair<ULong, ULong> {
            var lo = 0uL
            for (i in 3 downTo 0) {
                lo = (lo shl 16) or limb(i).toULong()
            }
            var hi = 0uL
            for (i in 7 downTo 4) {
                hi = (hi shl 16) or limb(i).toULong()
            }
            return hi to lo
        }

        val a = uint128Of(1uL, 3uL)
        val b = uint128Of(0uL, 5uL)
        val (prodLo, prodHi) = HPC16x8.mul128(a, b)
        assertEquals(5uL to 15uL, prodLo.toPair())
        assertEquals(0uL to 0uL, prodHi.toPair())

        val (quot, rem) = HPC16x8.divmod128(uint128Of(5uL, 15uL), uint128Of(0uL, 5uL))
        assertEquals(1uL to 3uL, quot.toPair())
        assertEquals(0uL to 0uL, rem.toPair())

        val (hi, lo) = HPC16x8.umul128(0xFFFF_FFFF_FFFF_FFFEuL, 2uL)
        assertEquals(1uL, hi)
        assertEquals(0xFFFF_FFFF_FFFF_FFFCuL, lo)

        assertEquals(65, HPC16x8.bitLength(uint128Of(1uL, 0uL)))
    }

    @Test
    fun memoryFloatHelpers() {
        val floats = FloatArray(4)
        MemoryOps.memset(floats, 0, floats.size, 1.5f)
        floats.forEach { assertEquals(1.5f, it) }
        val other = floatArrayOf(0f, 1f, 2f, 3f)
        MemoryOps.memcpy(floats, 0, other, 0, other.size)
        other.forEachIndexed { idx, v -> assertEquals(v, floats[idx]) }
    }
}
