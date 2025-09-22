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
    private const val BASE16: Int = 65536
    private const val VECTOR_THRESHOLD: Int = 64

    /**
     * In-place left shift of a little-endian IntArray of 16-bit limbs: a[from .. from+len-1].
     * Each element is treated as 0..0xFFFF. Returns carryOut (upper s bits from the last limb).
     */
    fun shl16LEInPlace(a: IntArray, from: Int, len: Int, s: Int, carryIn: Int = 0): ShiftResult {
        require(s in 0..15) { "s must be in 0..15" }
        if (len <= 0 || s == 0) return ShiftResult(carryIn and 0xFFFF, false)
        if (len >= VECTOR_THRESHOLD) return shl16ThreePass(a, from, len, s, carryIn)
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
            a[i] = (combined % BASE16)
            // bits shifted out from cur become new carry
            val carryMask = mask
            carry = if (carryMask == 0) 0 else (rs.carry.toInt() % (carryMask + 1))
        }
        return ShiftResult(carry and 0xFFFF, sticky)
    }

    // 3-pass left shift for better auto-vectorization
    private fun shl16ThreePass(a: IntArray, from: Int, len: Int, s: Int, carryIn: Int): ShiftResult {
        val lo = IntArray(len)
        val hi = IntArray(len)
        val pow2s = a32.leftShift(1L, s).toInt()
        val mask16 = BASE16 - 1

        // Pass A: lo = (val * 2^s) mod 2^16
        var idx = 0
        while (idx < len) {
            val v = a[from + idx] and mask16
            val prod = (v.toLong() * pow2s.toLong())
            lo[idx] = (prod % BASE16).toInt()
            idx++
        }
        // Pass B: hi = floor(val / 2^(16-s)) (top s bits moved to low)
        idx = 0
        while (idx < len) {
            val v = a[from + idx] and mask16
            hi[idx] = a16.rightShift(v.toLong(), 16 - s).toInt() and mask16
            idx++
        }
        // Pass C: combine with neighbor carry; carryIn feeds element 0
        val maskLowS = (pow2s - 1).coerceAtLeast(0)
        var carry = carryIn and 0xFFFF
        idx = 0
        while (idx < len) {
            val neighbor = if (idx == 0) (if (maskLowS == 0) 0 else (carry % (maskLowS + 1))) else hi[idx - 1]
            val combined = BitwiseOps.orArithmeticGeneral(lo[idx], neighbor)
            a[from + idx] = combined % BASE16
            idx++
        }
        val carryOut = if (len > 0) (hi[len - 1] and maskLowS) else 0
        return ShiftResult(carryOut and 0xFFFF, false)
    }

    /**
     * In-place right shift of a little-endian IntArray of 16-bit limbs: a[from .. from+len-1].
     * Returns carryOut (low s bits shifted out from the first limb) and sticky (OR of all bits shifted out).
     */
    fun rsh16LEInPlace(a: IntArray, from: Int, len: Int, s: Int): ShiftResult {
        require(s in 0..15) { "s must be in 0..15" }
        if (len <= 0 || s == 0) return ShiftResult(0, false)
        if (len >= VECTOR_THRESHOLD) return rsh16ThreePass(a, from, len, s)
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
            a[i] = (out % BASE16)
            nextCarry = dropped
        }
        return ShiftResult(carryOut and 0xFFFF, sticky)
    }

    private fun rsh16ThreePass(a: IntArray, from: Int, len: Int, s: Int): ShiftResult {
        val hi = IntArray(len)
        val dropped = IntArray(len)
        val mask16 = BASE16 - 1
        val pow2s = a32.leftShift(1L, s).toInt()
        val pow2_16_minus_s = a32.leftShift(1L, 16 - s).toInt()

        // Pass A: hi = floor(val / 2^s)
        var idx = 0
        while (idx < len) {
            val v = a[from + idx] and mask16
            hi[idx] = a16.rightShift(v.toLong(), s).toInt() and mask16
            idx++
        }
        // Pass B: dropped = val % 2^s
        idx = 0
        while (idx < len) {
            val v = a[from + idx] and mask16
            dropped[idx] = if (s == 0) 0 else (v % pow2s)
            idx++
        }
        // Pass C: out[i] = hi[i] OR ((i+1<) dropped[i+1] * 2^(16-s))
        var sticky = false
        idx = 0
        while (idx < len) {
            val neighbor = if (idx + 1 < len) ((dropped[idx + 1].toLong() * pow2_16_minus_s) % BASE16).toInt() else 0
            val combined = BitwiseOps.orArithmeticGeneral(hi[idx], neighbor)
            a[from + idx] = combined % BASE16
            sticky = sticky or (dropped[idx] != 0)
            idx++
        }
        val carryOut = if (len > 0) dropped[0] else 0
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
