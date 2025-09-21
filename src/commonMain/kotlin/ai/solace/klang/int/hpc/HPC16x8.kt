package ai.solace.klang.int.hpc

/**
 * HPC16x8: 128-bit unsigned integer stored as 8 little-endian 16-bit limbs.
 */
class HPC16x8 private constructor(private val limbs: UShortArray) {
    init { require(limbs.size == 8) }

    fun copy(): HPC16x8 = HPC16x8(limbs.copyOf())

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
        val out = UShortArray(8)
        var carry = 0u
        for (i in 0..7) {
            val cur = limbs[i].toUInt()
            val v = (cur shl k) or carry
            out[i] = (v and 0xFFFFu).toUShort()
            carry = (cur shr (16 - k)) and ((1u shl k) - 1u)
        }
        return HPC16x8(out) to carry.toUShort()
    }

    fun shrBits(k: Int): Pair<HPC16x8, UShort> {
        require(k in 0..15)
        if (k == 0) return this.copy() to 0u
        val out = UShortArray(8)
        var carry = 0u
        for (i in 7 downTo 0) {
            val cur = limbs[i].toUInt()
            val v = (cur shr k) or (carry shl (16 - k))
            out[i] = (v and 0xFFFFu).toUShort()
            carry = cur and ((1u shl k) - 1u)
        }
        return HPC16x8(out) to carry.toUShort()
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
            val base = 65536u
            val acc = UIntArray(16) { 0u }
            for (i in 0..3) {
                val ai = a.limb(i).toUInt()
                var carry = 0u
                for (j in 0..3) {
                    val idx = i + j
                    val prod = ai * b.limb(j).toUInt() + acc[idx].toUInt() + carry
                    acc[idx] = prod and 0xFFFFu
                    carry = prod shr 16
                }
                acc[i + 4] = (acc[i + 4] + carry) and 0xFFFFu
            }
            val limbs = UShortArray(8)
            for (i in 0..7) limbs[i] = acc[i].toUShort()
            return HPC16x8(limbs)
        }
    }
}

