package ai.solace.klang.stringshift

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

private fun leftShiftBitwiseLimbs(aIn: IntArray, s: Int): IntArray {
    require(s >= 0)
    if (s == 0) return aIn.copyOf()
    val carryMask = if (s == 0) 0 else ((1 shl (s % 16)) - 1)
    val shiftFull = s / 16
    val shiftBits = s % 16
    var a = aIn.copyOf()
    if (shiftFull > 0) {
        val b = IntArray(a.size + shiftFull)
        for (i in a.indices) b[i + shiftFull] = a[i] and 0xFFFF
        a = b
    }
    if (shiftBits == 0) return a
    var carry = 0
    for (i in 0 until a.size) {
        val cur = a[i] and 0xFFFF
        val out = ((cur shl shiftBits) and 0xFFFF) or carry
        a[i] = out
        carry = (cur ushr (16 - shiftBits)) and carryMask
    }
    if (carry != 0) {
        val b = IntArray(a.size + 1)
        for (i in a.indices) b[i] = a[i]
        b[a.size] = carry
        a = b
    }
    return a
}

private fun rightShiftBitwiseLimbs(aIn: IntArray, s: Int): IntArray {
    require(s >= 0)
    if (s == 0) return aIn.copyOf()
    val shiftFull = s / 16
    val shiftBits = s % 16
    if (shiftFull >= aIn.size) return intArrayOf(0)
    var a = if (shiftFull > 0) {
        val b = IntArray(aIn.size - shiftFull)
        for (i in shiftFull until aIn.size) b[i - shiftFull] = aIn[i] and 0xFFFF
        b
    } else aIn.copyOf()
    if (shiftBits == 0) return normalizeZero(a)
    var carry = 0
    for (i in a.size - 1 downTo 0) {
        val cur = a[i] and 0xFFFF
        val out = (cur ushr shiftBits) or (carry shl (16 - shiftBits))
        a[i] = out and 0xFFFF
        carry = cur and ((1 shl shiftBits) - 1)
    }
    return normalizeZero(a)
}

private fun normalizeZero(a: IntArray): IntArray {
    // strip high zero limbs
    var n = a.size
    while (n > 1 && a[n - 1] == 0) n--
    return if (n == a.size) a else a.copyOf(n)
}

class HexShiftTest {
    @Test
    fun testLeftAndRightShiftHexParity() {
        val sizes = listOf(1, 3, 8, 64, 257)
        val shifts = (0..63)
        val rnd = Random(1234)
        repeat(50) {
            for (sz in sizes) {
                val limbs = IntArray(sz) { rnd.nextInt() and 0xFFFF }
                // avoid all-zero random case occasionally
                if (limbs.all { it == 0 }) limbs[0] = 1
                val hex0 = limbsToHex(limbs)
                for (s in shifts) {
                    // left shift
                    val viaHexL = leftShiftHexString(hex0, s)
                    val viaNumL = limbsToHex(leftShiftBitwiseLimbs(limbs, s))
                    assertEquals(viaNumL, viaHexL, "left s=$s size=$sz")
                    // right shift
                    val viaHexR = rightShiftHexString(hex0, s)
                    val viaNumR = limbsToHex(rightShiftBitwiseLimbs(limbs, s))
                    assertEquals(viaNumR, viaHexR, "right s=$s size=$sz")
                }
            }
        }
    }

    @Test
    fun testShiftComposition() {
        val rnd = Random(42)
        val limbs = IntArray(16) { rnd.nextInt() and 0xFFFF }
        val hex0 = limbsToHex(limbs)
        val s1 = 13
        val s2 = 27
        val a = leftShiftHexString(leftShiftHexString(hex0, s1), s2)
        val b = leftShiftHexString(hex0, s1 + s2)
        assertEquals(b, a)

        val c = rightShiftHexString(rightShiftHexString(hex0, s1), s2)
        val d = rightShiftHexString(hex0, s1 + s2)
        assertEquals(d, c)
    }
}

