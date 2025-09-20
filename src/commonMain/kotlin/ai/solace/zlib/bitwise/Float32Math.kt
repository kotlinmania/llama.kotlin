package ai.solace.zlib.bitwise

import kotlin.math.abs

/**
 * Software IEEE-754 float32 multiply using integer bit manipulation and
 * round-to-nearest, ties-to-even. Handles zeros, subnormals, infinities, NaNs.
 */
object Float32Math {
    private const val SIGN_MASK = 0x80000000.toInt()
    private const val EXP_MASK  = 0x7F800000.toInt()
    private const val FRAC_MASK = 0x007FFFFF
    private const val EXP_BIAS  = 127

    private const val CANONICAL_NAN = 0x7FC00000.toInt()

    fun mul(a: Float, b: Float): Float = Float.fromBits(mulBits(a.toRawBits(), b.toRawBits()))

    fun mulBits(aBits: Int, bBits: Int): Int {
        val aExp = (aBits ushr 23) and 0xFF
        val bExp = (bBits ushr 23) and 0xFF
        val aFrac = aBits and FRAC_MASK
        val bFrac = bBits and FRAC_MASK

        val aSign = (aBits ushr 31) and 1
        val bSign = (bBits ushr 31) and 1
        val outSign = aSign xor bSign

        // NaNs
        val aNaN = aExp == 0xFF && aFrac != 0
        val bNaN = bExp == 0xFF && bFrac != 0
        if (aNaN || bNaN) return CANONICAL_NAN

        // Infinities and zeros
        val aInf = aExp == 0xFF && aFrac == 0
        val bInf = bExp == 0xFF && bFrac == 0
        val aZero = aExp == 0 && aFrac == 0
        val bZero = bExp == 0 && bFrac == 0

        if ((aInf && bZero) || (bInf && aZero)) return CANONICAL_NAN
        if (aInf || bInf) return (outSign shl 31) or EXP_MASK // infinity
        if (aZero || bZero) return (outSign shl 31) // signed zero

        // Build 24-bit significands and effective exponents
        var sigA: Long
        var sigB: Long
        var eA = aExp
        var eB = bExp

        if (aExp == 0) {
            // subnormal: exponent = -126, significand = frac, normalize
            sigA = aFrac.toLong()
            var shift = clz24(sigA)
            if (shift == 24) return (outSign shl 31) // shouldn't happen after zero check
            sigA = sigA shl shift
            eA = 1 - shift // will become: eA_eff = -126 - (shift-1); we encode below
        } else {
            sigA = (1L shl 23) or aFrac.toLong()
        }

        if (bExp == 0) {
            sigB = bFrac.toLong()
            var shift = clz24(sigB)
            if (shift == 24) return (outSign shl 31)
            sigB = sigB shl shift
            eB = 1 - shift
        } else {
            sigB = (1L shl 23) or bFrac.toLong()
        }

        // Effective unbiased exponents
        val eAeff = if (aExp == 0) (1 - EXP_BIAS) + (eA - 1) else (aExp - EXP_BIAS)
        val eBeff = if (bExp == 0) (1 - EXP_BIAS) + (eB - 1) else (bExp - EXP_BIAS)

        // 24x24 -> 48-bit product
        val prod: Long = sigA * sigB // up to 48 bits

        // Normalize: target 24-bit significand with leading 1 at bit 23
        val topBit47 = (prod ushr 47) and 1L
        val shift = if (topBit47 == 1L) 24 else 23
        var signif: Long = prod ushr shift // top 24 bits
        var remainder: Long = prod and ((1L shl shift) - 1)

        var exp = eAeff + eBeff + if (topBit47 == 1L) 1 else 0

        // Round to nearest, ties to even
        val guard = (remainder ushr (shift - 1)) and 1L
        val sticky = remainder and ((1L shl (shift - 1)) - 1)
        val lsb = signif and 1L
        if (guard == 1L && (sticky != 0L || lsb == 1L)) {
            signif += 1
            if (signif == (1L shl 24)) {
                // carry-out renormalization
                signif = 1L shl 23
                exp += 1
            }
        }

        // Handle overflow to infinity
        var biasedExp = exp + EXP_BIAS
        if (biasedExp >= 0xFF) {
            return (outSign shl 31) or EXP_MASK
        }

        // Handle subnormal/underflow
        if (biasedExp <= 0) {
            // shift right to create subnormal significand
            val rshift = (1 - biasedExp).coerceAtMost(31)
            // include the implicit 1 for normals
            var sig = signif
            // Compose a 24-bit value with rounding while shifting to subnormal
            var extra = 0L
            if (rshift >= 24) {
                extra = sig
                sig = 0
            } else {
                extra = sig and ((1L shl rshift) - 1)
                sig = sig ushr rshift
            }
            // Round when shifting to subnormal
            val extraGuard = (extra ushr (rshift - 1).coerceAtLeast(0)) and 1L
            val extraSticky = extra and ((1L shl (rshift - 1).coerceAtLeast(0)) - 1)
            val lsb2 = sig and 1L
            if (rshift > 0 && extraGuard == 1L && (extraSticky != 0L || lsb2 == 1L)) {
                sig += 1
            }
            val frac = (sig and FRAC_MASK.toLong()).toInt()
            return (outSign shl 31) or frac
        }

        val frac = ((signif and FRAC_MASK.toLong()).toInt())
        return (outSign shl 31) or (biasedExp shl 23) or frac
    }

    // Count leading zeros in a 24-bit value (bits in positions 23..0). Returns 24 if zero.
    private fun clz24(xIn: Long): Int {
        var x = xIn and 0xFFFFFF
        if (x == 0L) return 24
        var n = 0
        var v = x
        var bit = 1 shl 23
        while ((v and bit.toLong()) == 0L) {
            n++
            bit = bit ushr 1
        }
        return n
    }
}

