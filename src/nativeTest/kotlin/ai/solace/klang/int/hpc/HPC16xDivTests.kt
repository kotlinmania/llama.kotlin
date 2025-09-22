package ai.solace.klang.int.hpc

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class HPC16xDivTests {
    @Test
    fun div128by64_basic() {
        val hi = HPC16x4.fromULong(0x0000_0000_0000_0001uL)
        val lo = HPC16x4.fromULong(0x0000_0000_FFFF_FFFFuL)
        val num = HPC16x8.mul64x64To128(hi, HPC16x4.fromULong(0x0000_0000_0000_0001uL))
        val num2 = HPC16x8.ofLimbsLE(0xFFFFu, 0xFFFFu, 0u, 0u, 1u, 0u, 0u, 0u)
        val den = HPC16x4.fromULong(0x0000_0001_0000_0000uL)
        val (q, r) = HPC16x8.div128by64(num2, den)
        // For d = 2^32, quotient = (hi<<32) | (lo>>32), remainder = lo & (2^32-1)
        val hiU = 1uL
        val loU = 0x0000_0000_FFFF_FFFFuL
        val refQ = (hiU shl 32) or (loU shr 32)
        val refR = (loU and 0xFFFF_FFFFuL)
        assertEquals(refQ, q.toULong())
        assertEquals(refR, r.toULong())
    }

    @Test
    fun div128by64_selected_cases() {
        val cases = listOf(
            0x8000_0000_0000_0000uL to 0x0000_0000_0000_0001uL,
            0x8000_0000_0000_0000uL to 0xFFFF_FFFF_FFFF_FFFFuL
        )
        for ((denU, loU) in cases) {
            val hiU = denU - 1uL
            val num = HPC16x8.ofLimbsLE(
                (loU and 0xFFFFu).toUShort(),
                ((loU shr 16) and 0xFFFFu).toUShort(),
                ((loU shr 32) and 0xFFFFu).toUShort(),
                ((loU shr 48) and 0xFFFFu).toUShort(),
                (hiU and 0xFFFFu).toUShort(),
                ((hiU shr 16) and 0xFFFFu).toUShort(),
                ((hiU shr 32) and 0xFFFFu).toUShort(),
                ((hiU shr 48) and 0xFFFFu).toUShort(),
            )
            val den = HPC16x4.fromULong(denU)
            val (q, r) = HPC16x8.div128by64(num, den)
            val prod = HPC16x8.mul64x64To128(q, den)
            val recomposed = run {
                val arr = UShortArray(8)
                for (i in 0..7) arr[i] = prod.limb(i)
                var carry = 0u
                for (i in 0..3) {
                    val sum = arr[i].toUInt() + r.limb(i).toUInt() + carry
                    arr[i] = (sum and 0xFFFFu).toUShort()
                    carry = sum shr 16
                }
                var idx = 4
                while (carry != 0u && idx < 8) {
                    val sum = arr[idx].toUInt() + carry
                    arr[idx] = (sum and 0xFFFFu).toUShort()
                    carry = sum shr 16
                    idx++
                }
                HPC16x8.ofLimbsLE(arr[0],arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7])
            }
            var equal = true
            for (i in 0..7) if (recomposed.limb(i) != num.limb(i)) { equal = false; break }
            if (!equal) {
                fun h(u: ULong) = u.toString(16)
                fun limb4(a: HPC16x8, i: Int) = a.limb(i).toString(16).padStart(4, '0')
                error("seed=99 den=0x" + h(denU) + " hi=0x" + h(hiU) + " lo=0x" + h(loU) +
                        " q=0x" + h(q.toULong()) + " r=0x" + h(r.toULong()) +
                        " num=[${limb4(num,7)} ${limb4(num,6)} ${limb4(num,5)} ${limb4(num,4)} | ${limb4(num,3)} ${limb4(num,2)} ${limb4(num,1)} ${limb4(num,0)}]" +
                        " rec=[${limb4(recomposed,7)} ${limb4(recomposed,6)} ${limb4(recomposed,5)} ${limb4(recomposed,4)} | ${limb4(recomposed,3)} ${limb4(recomposed,2)} ${limb4(recomposed,1)} ${limb4(recomposed,0)}]")
            }
            assertTrue(equal)
            assertTrue(r.toULong() < denU)
        }
    }

    @Test
    fun div128by64_randomized_recomposition() {
        val rnd = Random(99)
        repeat(100) {
            val denU = (rnd.nextLong().toULong() or 1uL)
            val hiU = (rnd.nextLong().toULong()) % denU
            val loU = rnd.nextLong().toULong()
            val num = HPC16x8.ofLimbsLE(
                (loU and 0xFFFFu).toUShort(),
                ((loU shr 16) and 0xFFFFu).toUShort(),
                ((loU shr 32) and 0xFFFFu).toUShort(),
                ((loU shr 48) and 0xFFFFu).toUShort(),
                (hiU and 0xFFFFu).toUShort(),
                ((hiU shr 16) and 0xFFFFu).toUShort(),
                ((hiU shr 32) and 0xFFFFu).toUShort(),
                ((hiU shr 48) and 0xFFFFu).toUShort(),
            )
            val den = HPC16x4.fromULong(denU)
            val (q, r) = num.divRem(den)
            // Recompose and check
            val prod = HPC16x8.mul64x64To128(q, den)
            val recomposed = run {
                val arr = UShortArray(8)
                for (i in 0..7) arr[i] = prod.limb(i)
                var carry = 0u
                for (i in 0..3) {
                    val sum = arr[i].toUInt() + r.limb(i).toUInt() + carry
                    arr[i] = (sum and 0xFFFFu).toUShort()
                    carry = sum shr 16
                }
                var idx = 4
                while (carry != 0u && idx < 8) {
                    val sum = arr[idx].toUInt() + carry
                    arr[idx] = (sum and 0xFFFFu).toUShort()
                    carry = sum shr 16
                    idx++
                }
                HPC16x8.ofLimbsLE(arr[0],arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7])
            }
            var equal = true
            for (i in 0..7) if (recomposed.limb(i) != num.limb(i)) { equal = false; break }
            assertTrue(equal)
            assertTrue(r.toULong() < denU)
        }
    }
}
