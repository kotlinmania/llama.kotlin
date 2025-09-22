package ai.solace.klang.bitwise

/**
 * Array-wide bit shifts for limb arrays (little-endian) with optional sticky tracking.
 * These are scalar fallbacks designed to be allocation-free and branch-light.
 * Future: provide platform-optimized actuals (JVM Vector API / Kotlin/Native Vector128).
 */
object ArrayBitShifts {
    data class ShiftResult(val carryOut: Int, val sticky: Boolean)
    private val eng16Arith = BitShiftEngine(BitShiftMode.ARITHMETIC, 16)
    private val eng16 get() = eng16Arith
    private val a16 = ArithmeticBitwiseOps(16)
    private val a32 = ArithmeticBitwiseOps.BITS_32

    /**
     * In-place left shift of a little-endian IntArray of 16-bit limbs: a[from .. from+len-1].
     * Each element is treated as 0..0xFFFF. Returns carryOut (upper s bits from the last limb).
     */
    fun shl16LEInPlace(a: IntArray, from: Int, len: Int, s: Int, carryIn: Int = 0): ShiftResult {
        require(s in 0..15) { "s must be in 0..15" }
        if (len <= 0 || s == 0) return ShiftResult(carryIn and 0xFFFF, false)
        var carry = carryIn and 0xFFFF
        var sticky = false
        for (i in from until from + len) {
            val cur = (a[i] and 0xFFFF).toLong()
            val rs = eng16.leftShift(cur, s)
            val lowShifted = a16.normalize(rs.value).toInt()
            // mask = (1 << s) - 1 via arithmetic-only
            val pow2s = a32.leftShift(1L, s).toInt()
            val mask = (pow2s - 1).coerceAtLeast(0)
            val carryLow = if (mask == 0) 0 else (carry % (mask + 1))
            val combined = BitwiseOps.orArithmetic(lowShifted, carryLow)
            a[i] = (combined % 65536)
            // bits shifted out from cur become new carry
            val carryMask = mask
            carry = if (carryMask == 0) 0 else (rs.carry.toInt() % (carryMask + 1))
        }
        return ShiftResult(carry and 0xFFFF, sticky)
    }

    /**
     * In-place right shift of a little-endian IntArray of 16-bit limbs: a[from .. from+len-1].
     * Returns carryOut (low s bits shifted out from the first limb) and sticky (OR of all bits shifted out).
     */
    fun rsh16LEInPlace(a: IntArray, from: Int, len: Int, s: Int): ShiftResult {
        require(s in 0..15) { "s must be in 0..15" }
        if (len <= 0 || s == 0) return ShiftResult(0, false)
        var nextCarry = 0
        var sticky = false
        var carryOut = 0
        for (i in from + len - 1 downTo from) {
            val cur = (a[i] and 0xFFFF).toLong()
            val rs = eng16.unsignedRightShift(cur, s)
            val lowPart = a16.normalize(rs.value).toInt()
            // compute nextCarry * 2^(16-s) arithmetically
            val shiftHi = 16 - s
            val highPart = if (shiftHi == 0) nextCarry % 65536 else a32.leftShift(nextCarry.toLong(), shiftHi).toInt()
            val out = BitwiseOps.orArithmeticGeneral(lowPart, highPart)
            // dropped = cur % 2^s
            val pow2s = a32.leftShift(1L, s).toInt()
            val dropped = if (s == 0) 0 else (cur.toInt() % pow2s)
            if (i == from) carryOut = dropped
            sticky = sticky or (dropped != 0)
            a[i] = (out % 65536)
            nextCarry = dropped
        }
        return ShiftResult(carryOut and 0xFFFF, sticky)
    }

    /** Word-shift (multiple of 16 bits) left in-place for 16-bit limbs. */
    fun shl16LEWordsInPlace(a: IntArray, from: Int, len: Int, words: Int) {
        if (words <= 0) return
        for (i in (from + len - 1) downTo (from + words)) {
            a[i] = a[i - words] and 0xFFFF
        }
        for (i in from until from + words) a[i] = 0
    }

    /** Word-shift (multiple of 16 bits) right in-place for 16-bit limbs. */
    fun rsh16LEWordsInPlace(a: IntArray, from: Int, len: Int, words: Int) {
        if (words <= 0) return
        for (i in from until from + len - words) {
            a[i] = a[i + words] and 0xFFFF
        }
        for (i in (from + len - words) until (from + len)) a[i] = 0
    }
}
