package ai.solace.klang.bitwise

/**
 * Array-wide bit shifts for limb arrays (little-endian) with optional sticky tracking.
 * These are scalar fallbacks designed to be allocation-free and branch-light.
 * Future: provide platform-optimized actuals (JVM Vector API / Kotlin/Native Vector128).
 */
object ArrayBitShifts {
    data class ShiftResult(val carryOut: Int, val sticky: Boolean)
    private val eng16Native = BitShiftEngine(BitShiftMode.NATIVE, 16)
    private val eng16Arith = BitShiftEngine(BitShiftMode.ARITHMETIC, 16)
    private val eng16 get() = eng16Arith

    /**
     * In-place left shift of a little-endian IntArray of 16-bit limbs: a[from .. from+len-1].
     * Each element is treated as 0..0xFFFF. Returns carryOut (upper s bits from the last limb).
     */
    fun shl16LEInPlace(a: IntArray, from: Int, len: Int, s: Int, carryIn: Int = 0): ShiftResult {
        require(s in 0..15) { "s must be in 0..15" }
        if (len <= 0 || s == 0) return ShiftResult(carryIn and 0xFFFF, false)
        val before = if (ArrayShiftTrace.enabled) IntArray(len) { a[from + it] and 0xFFFF } else IntArray(0)
        var carry = carryIn and 0xFFFF
        var sticky = false
        for (i in from until from + len) {
            val cur = (a[i] and 0xFFFF).toLong()
            val rs = eng16.leftShift(cur, s)
            val low = (rs.value.toInt() and 0xFFFF) or (carry and ((1 shl s) - 1))
            a[i] = low and 0xFFFF
            // bits shifted out from cur become new carry
            carry = (rs.carry.toInt() and ((1 shl s) - 1))
        }
        val res = ShiftResult(carry and 0xFFFF, sticky)
        if (ArrayShiftTrace.enabled) {
            val after = IntArray(len) { a[from + it] and 0xFFFF }
            ArrayShiftTrace.record("shl16", s, from, len, before, after, res.carryOut, res.sticky)
        }
        return res
    }

    /**
     * In-place right shift of a little-endian IntArray of 16-bit limbs: a[from .. from+len-1].
     * Returns carryOut (low s bits shifted out from the first limb) and sticky (OR of all bits shifted out).
     */
    fun rsh16LEInPlace(a: IntArray, from: Int, len: Int, s: Int): ShiftResult {
        require(s in 0..15) { "s must be in 0..15" }
        if (len <= 0 || s == 0) return ShiftResult(0, false)
        val before = if (ArrayShiftTrace.enabled) IntArray(len) { a[from + it] and 0xFFFF } else IntArray(0)
        var nextCarry = 0
        var sticky = false
        var carryOut = 0
        for (i in from + len - 1 downTo from) {
            val cur = (a[i] and 0xFFFF).toLong()
            val rs = eng16.unsignedRightShift(cur, s)
            val out = (rs.value.toInt() and 0xFFFF) or (nextCarry shl (16 - s))
            val dropped = (cur.toInt() and ((1 shl s) - 1))
            if (i == from) carryOut = dropped
            sticky = sticky or (dropped != 0)
            a[i] = out and 0xFFFF
            nextCarry = dropped
        }
        val res = ShiftResult(carryOut and 0xFFFF, sticky)
        if (ArrayShiftTrace.enabled) {
            val after = IntArray(len) { a[from + it] and 0xFFFF }
            ArrayShiftTrace.record("rsh16", s, from, len, before, after, res.carryOut, res.sticky)
        }
        return res
    }

    /** Word-shift (multiple of 16 bits) left in-place for 16-bit limbs. */
    fun shl16LEWordsInPlace(a: IntArray, from: Int, len: Int, words: Int) {
        if (words <= 0) return
        val before = if (ArrayShiftTrace.enabled) IntArray(len) { a[from + it] and 0xFFFF } else IntArray(0)
        for (i in (from + len - 1) downTo (from + words)) {
            a[i] = a[i - words] and 0xFFFF
        }
        for (i in from until from + words) a[i] = 0
        if (ArrayShiftTrace.enabled) {
            val after = IntArray(len) { a[from + it] and 0xFFFF }
            ArrayShiftTrace.record("shl16w", words, from, len, before, after, 0, false)
        }
    }

    /** Word-shift (multiple of 16 bits) right in-place for 16-bit limbs. */
    fun rsh16LEWordsInPlace(a: IntArray, from: Int, len: Int, words: Int) {
        if (words <= 0) return
        val before = if (ArrayShiftTrace.enabled) IntArray(len) { a[from + it] and 0xFFFF } else IntArray(0)
        for (i in from until from + len - words) {
            a[i] = a[i + words] and 0xFFFF
        }
        for (i in (from + len - words) until (from + len)) a[i] = 0
        if (ArrayShiftTrace.enabled) {
            val after = IntArray(len) { a[from + it] and 0xFFFF }
            ArrayShiftTrace.record("rsh16w", words, from, len, before, after, 0, false)
        }
    }
}
