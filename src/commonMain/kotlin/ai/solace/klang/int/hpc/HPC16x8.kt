package ai.solace.klang.int.hpc

/**
 * HPC16x8: 128-bit unsigned integer stored as 8 little-endian 16-bit limbs.
 */
class HPC16x8 private constructor(private val limbs: UShortArray) {
    init { require(limbs.size == 8) }

    fun copy(): HPC16x8 = HPC16x8(limbs.copyOf())

    // Operators for convenience
    operator fun div(den: HPC16x4): HPC16x4 = divRem(den).first
    operator fun rem(den: HPC16x4): HPC16x4 = divRem(den).second
    fun divRem(den: HPC16x4): Pair<HPC16x4, HPC16x4> = div128by64(this, den)

    fun limb(i: Int): UShort = limbs[i]

    fun add(other: HPC16x8): Pair<HPC16x8, UShort> {
        val out = UShortArray(8)
        var carry = 0u
        for (i in 0..7) {
            val sum = limbs[i].toUInt() + other.limbs[i].toUInt() + carry
            out[i] = (sum and 0xFFFFu).toUShort()
            carry = sum shr 16
        }
        return HPC16x8(out) to carry.toUShort()
    }

    fun sub(other: HPC16x8): Pair<HPC16x8, UShort> {
        val out = UShortArray(8)
        var borrow = 0
        for (i in 0..7) {
            var diff = limbs[i].toInt() - other.limbs[i].toInt() - borrow
            if (diff < 0) { diff += 1 shl 16; borrow = 1 } else borrow = 0
            out[i] = (diff and 0xFFFF).toUShort()
        }
        return HPC16x8(out) to borrow.toUShort()
    }

    fun compareTo(other: HPC16x8): Int {
        for (i in 7 downTo 0) {
            val a = limbs[i].toUInt(); val b = other.limbs[i].toUInt()
            if (a != b) return if (a > b) 1 else -1
        }
        return 0
    }

    /** base-2^16 left shift by k bits, 0<=k<16 */
    fun shlBits(k: Int): Pair<HPC16x8, UShort> {
        require(k in 0..15)
        if (k == 0) return this.copy() to 0u
        val arr = IntArray(8) { limbs[it].toInt() and 0xFFFF }
        val res = ai.solace.klang.bitwise.ArrayBitShifts.shl16LEInPlace(arr, 0, 8, k)
        val out = UShortArray(8) { i -> (arr[i] and 0xFFFF).toUShort() }
        return HPC16x8(out) to (res.carryOut and 0xFFFF).toUShort()
    }

    fun shrBits(k: Int): Pair<HPC16x8, UShort> {
        require(k in 0..15)
        if (k == 0) return this.copy() to 0u
        val arr = IntArray(8) { limbs[it].toInt() and 0xFFFF }
        val res = ai.solace.klang.bitwise.ArrayBitShifts.rsh16LEInPlace(arr, 0, 8, k)
        val out = UShortArray(8) { i -> (arr[i] and 0xFFFF).toUShort() }
        return HPC16x8(out) to (res.carryOut and 0xFFFF).toUShort()
    }

    fun shlWords(words: Int): HPC16x8 {
        require(words in 0..8)
        if (words == 0) return copy()
        val out = UShortArray(8)
        for (i in 7 downTo words) out[i] = limbs[i - words]
        for (i in 0 until words) out[i] = 0u
        return HPC16x8(out)
    }

    fun shrWords(words: Int): HPC16x8 {
        require(words in 0..8)
        if (words == 0) return copy()
        val out = UShortArray(8)
        for (i in 0..7 - words) out[i] = limbs[i + words]
        for (i in 8 - words..7) out[i] = 0u
        return HPC16x8(out)
    }

    companion object {
        fun zero(): HPC16x8 = HPC16x8(UShortArray(8) { 0u })

        fun mul64x64To128(a: HPC16x4, b: HPC16x4): HPC16x8 {
            val res = UIntArray(8) { 0u }
            for (i in 0..3) {
                var carry = 0u
                val ai = a.limb(i).toUInt()
                for (j in 0..3) {
                    val idx = i + j
                    val sum = ai * b.limb(j).toUInt() + res[idx] + carry
                    res[idx] = sum and 0xFFFFu
                    carry = sum shr 16
                }
                // propagate remaining carry
                var k = i + 4
                while (carry != 0u && k < 8) {
                    val sum = res[k] + carry
                    res[k] = sum and 0xFFFFu
                    carry = sum shr 16
                    k++
                }
            }
            val out = UShortArray(8) { i -> res[i].toUShort() }
            return HPC16x8(out)
        }

        private fun toULong64(x: HPC16x4): ULong {
            var v = 0uL
            for (i in 3 downTo 0) {
                v = (v shl 16) or x.limb(i).toULong()
            }
            return v
        }

        fun ofLimbsLE(l0: UShort, l1: UShort, l2: UShort, l3: UShort, l4: UShort, l5: UShort, l6: UShort, l7: UShort): HPC16x8 =
            HPC16x8(ushortArrayOf(l0, l1, l2, l3, l4, l5, l6, l7))

        private fun hiLoULong(x: HPC16x8): Pair<ULong, ULong> {
            var lo = 0uL
            for (i in 3 downTo 0) lo = (lo shl 16) or x.limb(i).toULong()
            var hi = 0uL
            for (i in 7 downTo 4) hi = (hi shl 16) or x.limb(i).toULong()
            return hi to lo
        }

        private fun fromULong64(v: ULong): HPC16x4 = HPC16x4.fromULong(v)

        /** 128/64 division with bitwise long division under precondition hi < divisor. */
        fun div128by64(num: HPC16x8, den: HPC16x4): Pair<HPC16x4, HPC16x4> {
            val (hi, lo) = hiLoULong(num)
            val d = toULong64(den)
            require(d != 0uL) { "division by zero" }
            require(hi < d) { "128/64 quotient exceeds 64 bits (hi >= divisor)" }
            var rem = 0uL
            // accumulate hi bits
            for (i in 63 downTo 0) {
                val carry = (hi shr i) and 1uL
                rem = (rem shl 1) or carry
                if (rem >= d) rem -= d
            }
            var q = 0uL
            for (i in 63 downTo 0) {
                val carry = (lo shr i) and 1uL
                rem = (rem shl 1) or carry
                if (rem >= d) {
                    rem -= d
                    q = q or (1uL shl i)
                }
            }
            return fromULong64(q) to fromULong64(rem)
        }

        private fun clz16(xIn: Int): Int {
            var x = xIn and 0xFFFF
            if (x == 0) return 16
            var n = 0
            var bit = 1 shl 15
            while ((x and bit) == 0) { n++; bit = bit ushr 1 }
            return n
        }

        private fun lshiftBase(a: IntArray, len: Int, s: Int) {
            if (s == 0) return
            var carry = 0
            for (i in 0 until len) {
                val cur = a[i] and 0xFFFF
                val v = ((cur shl s) or carry) and 0x1FFFFF
                a[i] = v and 0xFFFF
                carry = (cur ushr (16 - s)) and ((1 shl s) - 1)
            }
            a[len] = carry and 0xFFFF
        }

        private fun rshiftBase(a: IntArray, len: Int, s: Int) {
            if (s == 0) return
            var carry = 0
            for (i in len - 1 downTo 0) {
                val cur = a[i] and 0xFFFF
                val v = ((cur ushr s) or (carry shl (16 - s))) and 0xFFFF
                carry = cur and ((1 shl s) - 1)
                a[i] = v
            }
        }
    }
}
