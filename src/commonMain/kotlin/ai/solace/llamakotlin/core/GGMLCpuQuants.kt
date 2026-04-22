// port-lint: source ggml/src/ggml-cpu/quants.c

package ai.solace.llamakotlin.core

/**
 * CPU-specific quantization / dequantization kernels ported from `ggml-cpu/quants.c`.
 *
 * This file contains:
 * - `quantizeRow*` — row-level quantization for every quant type
 * - `vecDot*` — quantized dot-product routines (scalar fallbacks)
 * - Helper conversions (E8M0, UE4M3, etc.)
 *
 * SIMD-optimised paths are **not** ported; every function here corresponds to the
 * `*_generic` (scalar) path in the C source. Block type definitions live in
 * [GGMLCommon.kt]; fp16↔fp32 helpers live in [NumericConversions.kt].
 *
 * Legacy Q1.5_K stubs that pre-date this port are preserved at the bottom of the file.
 */

private const val GROUP_MAX_EPS = 1e-15f

// ════════════════════════════════════════════════════════════════════════════════
//  Quantize-row stubs (delegate to _ref once those are ported in GGMLQuantsRef.kt)
// ════════════════════════════════════════════════════════════════════════════════

/** Quantize [k] floats from [x] into Q1_0 blocks in [y]. */
fun quantizeRowQ1_0(x: FloatArray, y: Array<BlockQ1_0>, k: Long) {
}

/** Quantize [k] floats from [x] into Q4_0 blocks in [y]. */
fun quantizeRowQ4_0(x: FloatArray, y: Array<BlockQ4_0>, k: Long) {
}

/** Quantize [k] floats from [x] into Q4_1 blocks in [y]. */
fun quantizeRowQ4_1(x: FloatArray, y: Array<BlockQ4_1>, k: Long) {
}

/** Quantize [k] floats from [x] into Q5_0 blocks in [y]. */
fun quantizeRowQ5_0(x: FloatArray, y: Array<BlockQ5_0>, k: Long) {
}

/** Quantize [k] floats from [x] into Q5_1 blocks in [y]. */
fun quantizeRowQ5_1(x: FloatArray, y: Array<BlockQ5_1>, k: Long) {
}

/** Quantize [k] floats from [x] into Q8_0 blocks in [y] (scalar/generic path). */
fun quantizeRowQ8_0(x: FloatArray, y: Array<BlockQ8_0>, k: Long) {
}

/** Quantize [k] floats from [x] into Q8_1 blocks in [y] (scalar/generic path). */
fun quantizeRowQ8_1(x: FloatArray, y: Array<BlockQ8_1>, k: Long) {
}

/** Quantize [k] floats from [x] into MXFP4 blocks in [y]. */
fun quantizeRowMXFP4(x: FloatArray, y: Array<BlockMXFP4>, k: Long) {
}

/** Quantize [k] floats from [x] into NVFP4 blocks in [y]. */
fun quantizeRowNVFP4(x: FloatArray, y: Array<BlockNVFP4>, k: Long) {
}

// ── K-quant quantize-row stubs ──────────────────────────────────────────────────

fun quantizeRowQ2K(x: FloatArray, y: Array<BlockQ2K>, k: Long) {
}

fun quantizeRowQ3K(x: FloatArray, y: Array<BlockQ3K>, k: Long) {
}

fun quantizeRowQ4K(x: FloatArray, y: Array<BlockQ4K>, k: Long) {
    require(k % QK_K == 0L) { "k ($k) must be a multiple of QK_K ($QK_K)" }
}

fun quantizeRowQ5K(x: FloatArray, y: Array<BlockQ5K>, k: Long) {
    require(k % QK_K == 0L) { "k ($k) must be a multiple of QK_K ($QK_K)" }
}

fun quantizeRowQ6K(x: FloatArray, y: Array<BlockQ6K>, k: Long) {
    require(k % QK_K == 0L) { "k ($k) must be a multiple of QK_K ($QK_K)" }
}

fun quantizeRowQ8K(x: FloatArray, y: Array<BlockQ8K>, k: Long) {
}

// ── Ternary quantize-row stubs ──────────────────────────────────────────────────

fun quantizeRowTQ1_0(x: FloatArray, y: Array<BlockTQ1_0>, k: Long) {
    require(k % QK_K == 0L) { "k ($k) must be a multiple of QK_K ($QK_K)" }
}

fun quantizeRowTQ2_0(x: FloatArray, y: Array<BlockTQ2_0>, k: Long) {
    require(k % QK_K == 0L) { "k ($k) must be a multiple of QK_K ($QK_K)" }
}

// ── IQ quantize-row stubs ───────────────────────────────────────────────────────

fun quantizeRowIQ4NL(x: FloatArray, y: Array<BlockIQ4NL>, k: Long) {
    require(k % QK4_NL == 0L) { "k ($k) must be a multiple of QK4_NL ($QK4_NL)" }
}

fun quantizeRowIQ4XS(x: FloatArray, y: Array<BlockIQ4XS>, k: Long) {
    require(k % QK_K == 0L) { "k ($k) must be a multiple of QK_K ($QK_K)" }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Scalar dot-product kernels (generic / fallback paths)
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Q1_0 × Q8_0 dot product (scalar).
 *
 * Each Q1_0 block covers [QK1_0] = 128 elements, which spans 4 Q8_0 blocks
 * of 32 elements each. Each bit in `x[i].qs` maps to ±1.
 *
 * @param n   number of elements (must be divisible by [QK1_0])
 * @param vx  Q1_0 blocks
 * @param vy  Q8_0 blocks (4× as many as vx)
 * @return    the dot product
 */
fun vecDotQ1_0Q8_0(n: Int, vx: Array<BlockQ1_0>, vy: Array<BlockQ8_0>): Float {
    val qk = QK1_0
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (i in 0 until nb) {
        val d0 = halfToFloat(vx[i].d)
        var sumi = 0.0f

        for (k in 0 until 4) {
            val d1 = halfToFloat(vy[i * 4 + k].d)
            var sumiBlock = 0

            for (j in 0 until QK8_0) {
                val bitIndex = k * QK8_0 + j
                val byteIndex = bitIndex / 8
                val bitOffset = bitIndex % 8

                val xi = if (((vx[i].qs[byteIndex].toInt() ushr bitOffset) and 1) != 0) 1 else -1
                sumiBlock += xi * vy[i * 4 + k].qs[j].toInt()
            }
            sumi += d1 * sumiBlock
        }
        sumf += d0 * sumi
    }
    return sumf
}

/**
 * Q4_0 × Q8_0 dot product (scalar).
 *
 * Each Q4_0 block stores 32 four-bit weights packed into 16 bytes (two per byte,
 * offset by 8). The companion Q8_0 block has 32 signed-byte weights.
 *
 * @param n   number of elements (must be divisible by [QK8_0])
 * @param vx  Q4_0 blocks
 * @param vy  Q8_0 blocks
 * @return    the dot product
 */
fun vecDotQ4_0Q8_0(n: Int, vx: Array<BlockQ4_0>, vy: Array<BlockQ8_0>): Float {
    val qk = QK8_0
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        var sumi0 = 0
        var sumi1 = 0
        for (j in 0 until qk / 2) {
            val v0 = (vx[ib].qs[j].toInt() and 0x0F) - 8
            val v1 = ((vx[ib].qs[j].toInt() and 0xFF) ushr 4) - 8

            sumi0 += v0 * vy[ib].qs[j].toInt()
            sumi1 += v1 * vy[ib].qs[j + qk / 2].toInt()
        }
        val sumi = sumi0 + sumi1
        sumf += sumi * halfToFloat(vx[ib].d) * halfToFloat(vy[ib].d)
    }
    return sumf
}

/**
 * Q4_1 × Q8_1 dot product (scalar).
 *
 * Like Q4_0 but with an additional minimum (`m`) and sum (`s`) per block.
 * Nibbles are unsigned (no −8 offset).
 */
fun vecDotQ4_1Q8_1(n: Int, vx: Array<BlockQ4_1>, vy: Array<BlockQ8_1>): Float {
    val qk = QK8_1
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        var sumi0 = 0
        var sumi1 = 0
        for (j in 0 until qk / 2) {
            val v0 = vx[ib].qs[j].toInt() and 0x0F
            val v1 = (vx[ib].qs[j].toInt() and 0xFF) ushr 4

            sumi0 += v0 * vy[ib].qs[j].toInt()
            sumi1 += v1 * vy[ib].qs[j + qk / 2].toInt()
        }
        val sumi = sumi0 + sumi1
        sumf += (halfToFloat(vx[ib].d) * halfToFloat(vy[ib].d)) * sumi +
                halfToFloat(vx[ib].m) * halfToFloat(vy[ib].s)
    }
    return sumf
}

/**
 * Q5_0 × Q8_0 dot product (scalar).
 *
 * Q5_0 uses 4 low bits in `qs` and a 5th (high) bit in `qh` (packed as a uint32).
 */
fun vecDotQ5_0Q8_0(n: Int, vx: Array<BlockQ5_0>, vy: Array<BlockQ8_0>): Float {
    val qk = QK8_0
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        // Reconstruct uint32 qh from 4 bytes (little-endian)
        val qh: Int = (vx[ib].qh[0].toInt() and 0xFF) or
                ((vx[ib].qh[1].toInt() and 0xFF) shl 8) or
                ((vx[ib].qh[2].toInt() and 0xFF) shl 16) or
                ((vx[ib].qh[3].toInt() and 0xFF) shl 24)

        var sumi0 = 0
        var sumi1 = 0
        for (j in 0 until qk / 2) {
            val xh0 = ((qh and (1 shl (j + 0))) ushr (j + 0)) shl 4
            val xh1 = ((qh and (1 shl (j + 16))) ushr (j + 12))

            val x0 = (((vx[ib].qs[j].toInt() and 0x0F) or xh0) - 16).toByte().toInt()
            val x1 = ((((vx[ib].qs[j].toInt() and 0xFF) ushr 4) or xh1) - 16).toByte().toInt()

            sumi0 += x0 * vy[ib].qs[j].toInt()
            sumi1 += x1 * vy[ib].qs[j + qk / 2].toInt()
        }
        val sumi = sumi0 + sumi1
        sumf += (halfToFloat(vx[ib].d) * halfToFloat(vy[ib].d)) * sumi
    }
    return sumf
}

/**
 * Q5_1 × Q8_1 dot product (scalar).
 *
 * Like Q5_0 but with separate min/sum fields and unsigned nibbles.
 */
fun vecDotQ5_1Q8_1(n: Int, vx: Array<BlockQ5_1>, vy: Array<BlockQ8_1>): Float {
    val qk = QK8_1
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        val qh: Int = (vx[ib].qh[0].toInt() and 0xFF) or
                ((vx[ib].qh[1].toInt() and 0xFF) shl 8) or
                ((vx[ib].qh[2].toInt() and 0xFF) shl 16) or
                ((vx[ib].qh[3].toInt() and 0xFF) shl 24)

        var sumi0 = 0
        var sumi1 = 0
        for (j in 0 until qk / 2) {
            val xh0 = ((qh ushr (j + 0)) shl 4) and 0x10
            val xh1 = ((qh ushr (j + 12))) and 0x10

            val x0 = (vx[ib].qs[j].toInt() and 0x0F) or xh0
            val x1 = ((vx[ib].qs[j].toInt() and 0xFF) ushr 4) or xh1

            sumi0 += x0 * vy[ib].qs[j].toInt()
            sumi1 += x1 * vy[ib].qs[j + qk / 2].toInt()
        }
        val sumi = sumi0 + sumi1
        sumf += (halfToFloat(vx[ib].d) * halfToFloat(vy[ib].d)) * sumi +
                halfToFloat(vx[ib].m) * halfToFloat(vy[ib].s)
    }
    return sumf
}

/**
 * Q8_0 × Q8_0 dot product (scalar).
 *
 * Straightforward: multiply corresponding 8-bit quants and accumulate,
 * then scale by the product of both block deltas.
 */
fun vecDotQ8_0Q8_0(n: Int, vx: Array<BlockQ8_0>, vy: Array<BlockQ8_0>): Float {
    val qk = QK8_0
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        var sumi = 0
        for (j in 0 until qk) {
            sumi += vx[ib].qs[j].toInt() * vy[ib].qs[j].toInt()
        }
        sumf += sumi * (halfToFloat(vx[ib].d) * halfToFloat(vy[ib].d))
    }
    return sumf
}

/**
 * MXFP4 × Q8_0 dot product (scalar).
 *
 * Uses the [GGMLCommonTables.kvaluesMXFP4] lookup to dequantize 4-bit mantissa values.
 */
fun vecDotMXFP4Q8_0(n: Int, vx: Array<BlockMXFP4>, vy: Array<BlockQ8_0>): Float {
    require(n % QK_MXFP4 == 0) { "n ($n) must be divisible by $QK_MXFP4" }
    val nb = n / QK_MXFP4

    var sumf = 0.0f
    for (ib in 0 until nb) {
        val d = halfToFloat(vy[ib].d) * ggmlE8M0ToFp32Half(vx[ib].e)
        var sumi1 = 0
        var sumi2 = 0
        for (j in 0 until QK_MXFP4 / 2) {
            sumi1 += vy[ib].qs[j].toInt() *
                    GGMLCommonTables.kvaluesMXFP4[vx[ib].qs[j].toInt() and 0xF].toInt()
            sumi2 += vy[ib].qs[j + QK_MXFP4 / 2].toInt() *
                    GGMLCommonTables.kvaluesMXFP4[(vx[ib].qs[j].toInt() and 0xFF) ushr 4].toInt()
        }
        sumf += d * (sumi1 + sumi2)
    }
    return sumf
}

/**
 * NVFP4 × Q8_0 dot product (scalar).
 *
 * NVFP4 super-blocks have 64 elements = 4 sub-blocks of 16, spanning 2 Q8_0 blocks.
 */
fun vecDotNVFP4Q8_0(n: Int, vx: Array<BlockNVFP4>, vy: Array<BlockQ8_0>): Float {
    require(n % QK_NVFP4 == 0) { "n ($n) must be divisible by $QK_NVFP4" }
    val nb = n / QK_NVFP4

    var sumf = 0.0f
    for (ib in 0 until nb) {
        for (sIdx in 0 until 4) {
            val d = ggmlUE4M3ToFp32(vx[ib].d[sIdx])
            val q8Block = sIdx / 2
            val q8Off = (sIdx % 2) * QK_NVFP4_SUB
            val dy = halfToFloat(vy[2 * ib + q8Block].d)

            var sumiLo = 0
            var sumiHi = 0
            for (j in 0 until QK_NVFP4_SUB / 2) {
                val qv = vx[ib].qs[sIdx * (QK_NVFP4_SUB / 2) + j].toInt() and 0xFF
                sumiLo += vy[2 * ib + q8Block].qs[q8Off + j].toInt() *
                        GGMLCommonTables.kvaluesMXFP4[qv and 0xF].toInt()
                sumiHi += vy[2 * ib + q8Block].qs[q8Off + j + QK_NVFP4_SUB / 2].toInt() *
                        GGMLCommonTables.kvaluesMXFP4[qv ushr 4].toInt()
            }
            sumf += dy * d * (sumiLo + sumiHi)
        }
    }
    return sumf
}

// ── Ternary dot products ────────────────────────────────────────────────────────

/**
 * TQ1_0 × Q8_K dot product (scalar).
 *
 * Ternary base-3 packed quants. 5 ternary values per byte in `qs`,
 * 4 values per byte in `qh`.
 */
fun vecDotTQ1_0Q8K(n: Int, vx: Array<BlockTQ1_0>, vy: Array<BlockQ8K>): Float {
    val nb = n / QK_K
    val pow3 = intArrayOf(1, 3, 9, 27, 81, 243)

    var sumf = 0.0f
    for (i in 0 until nb) {
        var sum = 0
        val qsSize = vx[i].qs.size

        // Main loop over qs in chunks of 32
        val qsAligned = qsSize - (qsSize % 32)
        var j = 0
        while (j < qsAligned) {
            for (l in 0 until 5) {
                for (m in 0 until 32) {
                    val q = ((vx[i].qs[j + m].toInt() and 0xFF) * pow3[l]) and 0xFF
                    val xi = ((q * 3) ushr 8)
                    sum += (xi - 1) * vy[i].qs[j * 5 + l * 32 + m].toInt()
                }
            }
            j += 32
        }
        // Remainder (chunks of 16)
        while (j < qsSize) {
            for (l in 0 until 5) {
                for (m in 0 until 16) {
                    val q = ((vx[i].qs[j + m].toInt() and 0xFF) * pow3[l]) and 0xFF
                    val xi = ((q * 3) ushr 8)
                    sum += (xi - 1) * vy[i].qs[j * 5 + l * 16 + m].toInt()
                }
            }
            j += 16
        }

        // qh part
        val qhSize = vx[i].qh.size
        for (l in 0 until 4) {
            for (jj in 0 until qhSize) {
                val q = ((vx[i].qh[jj].toInt() and 0xFF) * pow3[l]) and 0xFF
                val xi = ((q * 3) ushr 8)
                sum += (xi - 1) * vy[i].qs[qsSize * 5 + l * qhSize + jj].toInt()
            }
        }

        sumf += sum.toFloat() * (halfToFloat(vx[i].d) * vy[i].d)
    }
    return sumf
}

/**
 * TQ2_0 × Q8_K dot product (scalar).
 *
 * 2-bit ternary packed: each byte holds 4 ternary values in 2-bit pairs.
 */
fun vecDotTQ2_0Q8K(n: Int, vx: Array<BlockTQ2_0>, vy: Array<BlockQ8K>): Float {
    val nb = n / QK_K
    var sumf = 0.0f

    for (i in 0 until nb) {
        var sumi = 0
        val qsSize = vx[i].qs.size

        var j = 0
        while (j < qsSize) {
            val chunkSize = minOf(32, qsSize - j)
            for (l in 0 until 4) {
                for (k in 0 until chunkSize) {
                    sumi += vy[i].qs[j * 4 + l * 32 + k].toInt() *
                            ((((vx[i].qs[j + k].toInt() and 0xFF) ushr (l * 2)) and 3) - 1)
                }
            }
            j += 32
        }

        val d = vy[i].d * halfToFloat(vx[i].d)
        sumf += sumi.toFloat() * d
    }
    return sumf
}

// ── K-quant dot products ────────────────────────────────────────────────────────

/**
 * Q2_K × Q8_K dot product (scalar).
 *
 * Super-block with 4-bit scales/mins in `scales`, 2-bit quants in `qs`.
 */
fun vecDotQ2KQ8K(n: Int, vx: Array<BlockQ2K>, vy: Array<BlockQ8K>): Float {
    val nb = n / QK_K
    var sumf = 0.0f

    for (i in 0 until nb) {
        val sc = vx[i].scales

        var summs = 0
        for (j in 0 until 16) {
            summs += vy[i].bsums[j].toInt() * ((sc[j].toInt() and 0xFF) ushr 4)
        }

        val dall = vy[i].d * halfToFloat(vx[i].d)
        val dmin = vy[i].d * halfToFloat(vx[i].dmin)

        var isum = 0
        var isIdx = 0
        var q2Off = 0
        var q8Off = 0
        for (k in 0 until QK_K / 128) {
            var shift = 0
            for (j in 0 until 4) {
                val d1 = sc[isIdx++].toInt() and 0xF
                var isuml = 0
                for (l in 0 until 16) {
                    isuml += vy[i].qs[q8Off + l].toInt() *
                            (((vx[i].qs[q2Off + l].toInt() and 0xFF) ushr shift) and 3)
                }
                isum += d1 * isuml

                val d2 = sc[isIdx++].toInt() and 0xF
                isuml = 0
                for (l in 16 until 32) {
                    isuml += vy[i].qs[q8Off + l].toInt() *
                            (((vx[i].qs[q2Off + l].toInt() and 0xFF) ushr shift) and 3)
                }
                isum += d2 * isuml
                shift += 2
                q8Off += 32
            }
            q2Off += 32
        }
        sumf += dall * isum - dmin * summs
    }
    return sumf
}

/** Q3_K × Q8_K dot product (scalar). */
fun vecDotQ3KQ8K(n: Int, vx: Array<BlockQ3K>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotQ3KQ8K not yet ported")
}

/** Q4_K × Q8_K dot product (scalar). */
fun vecDotQ4KQ8K(n: Int, vx: Array<BlockQ4K>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotQ4KQ8K not yet ported")
}

/** Q5_K × Q8_K dot product (scalar). */
fun vecDotQ5KQ8K(n: Int, vx: Array<BlockQ5K>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotQ5KQ8K not yet ported")
}

/** Q6_K × Q8_K dot product (scalar). */
fun vecDotQ6KQ8K(n: Int, vx: Array<BlockQ6K>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotQ6KQ8K not yet ported")
}

// ── IQ dot-product stubs ────────────────────────────────────────────────────────

fun vecDotIQ2XXSQ8K(n: Int, vx: Array<BlockIQ2XXS>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotIQ2XXSQ8K not yet ported")
}

fun vecDotIQ2XSQ8K(n: Int, vx: Array<BlockIQ2XS>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotIQ2XSQ8K not yet ported")
}

fun vecDotIQ2SQ8K(n: Int, vx: Array<BlockIQ2S>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotIQ2SQ8K not yet ported")
}

fun vecDotIQ3XXSQ8K(n: Int, vx: Array<BlockIQ3XXS>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotIQ3XXSQ8K not yet ported")
}

fun vecDotIQ3SQ8K(n: Int, vx: Array<BlockIQ3S>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotIQ3SQ8K not yet ported")
}

fun vecDotIQ1SQ8K(n: Int, vx: Array<BlockIQ1S>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotIQ1SQ8K not yet ported")
}

fun vecDotIQ1MQ8K(n: Int, vx: Array<BlockIQ1M>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotIQ1MQ8K not yet ported")
}

/**
 * IQ4_NL × Q8_0 dot product (scalar).
 *
 * Non-linear 4-bit quants use [GGMLCommonTables.kvaluesIQ4NL] for dequantization.
 */
fun vecDotIQ4NLQ8_0(n: Int, vx: Array<BlockIQ4NL>, vy: Array<BlockQ8_0>): Float {
    require(n % QK4_NL == 0) { "n ($n) must be divisible by $QK4_NL" }
    val nb = n / QK4_NL

    var sumf = 0.0f
    for (ib in 0 until nb) {
        val d = halfToFloat(vy[ib].d) * halfToFloat(vx[ib].d)
        var sumi1 = 0
        var sumi2 = 0
        for (j in 0 until QK4_NL / 2) {
            sumi1 += vy[ib].qs[j].toInt() *
                    GGMLCommonTables.kvaluesIQ4NL[vx[ib].qs[j].toInt() and 0xF].toInt()
            sumi2 += vy[ib].qs[j + QK4_NL / 2].toInt() *
                    GGMLCommonTables.kvaluesIQ4NL[(vx[ib].qs[j].toInt() and 0xFF) ushr 4].toInt()
        }
        sumf += d * (sumi1 + sumi2)
    }
    return sumf
}

fun vecDotIQ4XSQ8K(n: Int, vx: Array<BlockIQ4XS>, vy: Array<BlockQ8K>): Float {
    require(n % QK_K == 0) { "n ($n) must be divisible by $QK_K" }
        error("vecDotIQ4XSQ8K not yet ported")
}

// ════════════════════════════════════════════════════════════════════════════════
//  Numeric helpers (E8M0, UE4M3)
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Convert an E8M0 exponent byte to a half-precision scale (returned as Float).
 *
 * E8M0 encodes a power-of-two exponent with an 8-bit biased value (bias = 127).
 * The result is `2^(e - 127)`, but we halve it (multiply by 0.5) to match the
 * `GGML_E8M0_TO_FP32_HALF` macro in the C source.
 */
internal fun ggmlE8M0ToFp32Half(e: Byte): Float {
    val exp = e.toInt() and 0xFF
    if (exp == 0xFF) return Float.NaN
    // 2^(exp - 127) * 0.5 = 2^(exp - 128)
    return Float.fromBits((exp - 128 + 127) shl 23)
}

/**
 * Convert a UE4M3 byte (unsigned 4-bit exponent, 3-bit mantissa) to Float.
 *
 * Mirrors `ggml_ue4m3_to_fp32` from the C source.
 */
internal fun ggmlUE4M3ToFp32(b: Byte): Float {
    val v = b.toInt() and 0xFF
    val mantissa = v and 0x7
    val exponent = v ushr 3
    return if (exponent == 0) {
        // Subnormal: mantissa * 2^(-9)
        mantissa * (1.0f / 512.0f)
    } else {
        // Normal: (8 + mantissa) * 2^(exponent - 8)
        (8 + mantissa).toFloat() * Float.fromBits((exponent - 8 + 127) shl 23)
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Legacy Q1.5_K stubs (pre-existing — preserved verbatim)
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Quantizes a row of float values to Q1.5_K format (ternary quantization).
 *
 * Q1.5_K uses ternary values (-1, 0, 1) to represent weights with minimal precision loss.
 * Each block contains a scale factor and packed ternary values optimized for storage efficiency.
 *
 * @param source The input array of float values to quantize
 * @param dest The output byte array for quantized data storage
 * @param elements The number of float elements to process
 * @param scale The quantization scale factor for this block
 */
fun quantizeRowQ15K(source: FloatArray, dest: ByteArray, elements: Int, scale: Float) {
}

/**
 * Dequantizes a row of Q1.5_K data back to float values.
 *
 * @param source The input byte array containing Q1.5_K quantized data
 * @param dest The output float array for dequantized values
 * @param elements The number of elements to dequantize
 * @param scale The scale factor used during quantization
 */
fun dequantizeRowQ15K(source: ByteArray, dest: FloatArray, elements: Int, scale: Float) {
}

/**
 * Computes the dot product of two Q1.5_K quantized vectors.
 *
 * @param elements The number of elements in each vector
 * @param vx The first quantized vector in Q1.5_K format
 * @param scaleX The scaling factor for the first vector
 * @param vy The second quantized vector in Q1.5_K format
 * @param scaleY The scaling factor for the second vector
 * @return The computed dot product as a float
 */
fun dotQ15K(elements: Int, vx: ByteArray, scaleX: Float, vy: ByteArray, scaleY: Float): Float {
    error("dotQ15K not yet ported")
}
