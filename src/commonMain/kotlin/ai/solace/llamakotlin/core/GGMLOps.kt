// port-lint: source ggml/include/ggml.h
package ai.solace.llamakotlin.core

// ============================================================================
// Constants from ggml.h (those not already in GGMLTypes.kt)
// ============================================================================

/** File magic number "ggml" */
const val GGML_FILE_MAGIC: Int = 0x67676d6c

/** File version */
const val GGML_FILE_VERSION: Int = 2

/** Quantization format version – bump on quantization format changes. */
const val GGML_QNT_VERSION: Int = 2
const val GGML_QNT_VERSION_FACTOR: Int = 1000

/** Maximum number of scheduling parameters. */
const val GGML_MAX_PARAMS: Int = 2048

/** Maximum number of threads allowed. */
const val GGML_MAX_N_THREADS: Int = 512

/** Default number of threads for graph compute. */
const val GGML_DEFAULT_N_THREADS: Int = 4

/** Default computation graph size (number of nodes). */
const val GGML_DEFAULT_GRAPH_SIZE: Int = 2048

/** Memory alignment in bytes (matches 64-bit platforms). */
const val GGML_MEM_ALIGN: Int = 16

const val GGML_EXIT_SUCCESS: Int = 0
const val GGML_EXIT_ABORTED: Int = 1

// RoPE type constants
const val GGML_ROPE_TYPE_NORMAL: Int = 0
const val GGML_ROPE_TYPE_NEOX: Int = 2
const val GGML_ROPE_TYPE_MROPE: Int = 8
const val GGML_ROPE_TYPE_VISION: Int = 24
const val GGML_ROPE_TYPE_IMROPE: Int = 40

/** Number of sections for multi-dimensional RoPE. */
const val GGML_MROPE_SECTIONS: Int = 4

/** Sentinel: use maximum available tasks. */
const val GGML_N_TASKS_MAX: Int = -1

// Tensor flag constants (supplement those in GGMLTypes.kt and GGMLGraph.kt)
// GGML_TENSOR_FLAG_OUTPUT already defined in GGMLTypes.kt
// GGML_TENSOR_FLAG_PARAM already defined in GGMLGraph.kt
const val GGML_TENSOR_FLAG_INPUT: Int = 1
const val GGML_TENSOR_FLAG_LOSS: Int = 8
const val GGML_TENSOR_FLAG_COMPUTE: Int = 16

// ============================================================================
// Enums from ggml.h not yet in GGMLTypes.kt
// ============================================================================

/** Precision hint for matrix multiplication and flash-attention. */
enum class GGMLPrec(val value: Int) {
    DEFAULT(0),
    F32(10);
}

/**
 * Model file types – describes the predominant quantization used in a model
 * file so that loaders can pick the right de-quantisation path.
 */
enum class GGMLFType(val value: Int) {
    UNKNOWN(-1),
    ALL_F32(0),
    MOSTLY_F16(1),
    MOSTLY_Q4_0(2),
    MOSTLY_Q4_1(3),
    MOSTLY_Q4_1_SOME_F16(4),
    MOSTLY_Q8_0(7),
    MOSTLY_Q5_0(8),
    MOSTLY_Q5_1(9),
    MOSTLY_Q2_K(10),
    MOSTLY_Q3_K(11),
    MOSTLY_Q4_K(12),
    MOSTLY_Q5_K(13),
    MOSTLY_Q6_K(14),
    MOSTLY_IQ2_XXS(15),
    MOSTLY_IQ2_XS(16),
    MOSTLY_IQ3_XXS(17),
    MOSTLY_IQ1_S(18),
    MOSTLY_IQ4_NL(19),
    MOSTLY_IQ3_S(20),
    MOSTLY_IQ2_S(21),
    MOSTLY_IQ4_XS(22),
    MOSTLY_IQ1_M(23),
    MOSTLY_BF16(24),
    MOSTLY_MXFP4(25),
    MOSTLY_NVFP4(26),
    MOSTLY_Q1_0(27);
}

/** Unary operations dispatched via GGML_OP_UNARY. */
enum class GGMLUnaryOp {
    ABS, SGN, NEG, STEP, TANH, ELU, RELU, SIGMOID,
    GELU, GELU_QUICK, SILU, HARDSWISH, HARDSIGMOID,
    EXP, EXPM1, SOFTPLUS, GELU_ERF, XIELU,
    FLOOR, CEIL, ROUND, TRUNC,
    COUNT;
}

/** Gated Linear Unit operation variants. */
enum class GGMLGluOp {
    REGLU, GEGLU, SWIGLU, SWIGLU_OAI, GEGLU_ERF, GEGLU_QUICK,
    COUNT;
}

/** Object types stored in a GGML context. */
enum class GGMLObjectType {
    TENSOR, GRAPH, WORK_BUFFER;
}

// GGMLLogLevel is defined in NumericConversions.kt

/** Triangular matrix type for [ggmlTri]. */
enum class GGMLTriType(val value: Int) {
    UPPER_DIAG(0), UPPER(1), LOWER_DIAG(2), LOWER(3);
}

/** Pooling operation kind. */
enum class GGMLOpPool {
    MAX, AVG, COUNT;
}

/** Interpolation scale mode. */
enum class GGMLScaleMode(val value: Int) {
    NEAREST(0), BILINEAR(1), BICUBIC(2), COUNT(3);
}

/** Interpolation scale flags (bitwise-OR with [GGMLScaleMode]). */
object GGMLScaleFlag {
    const val ALIGN_CORNERS: Int = 1 shl 8
    const val ANTIALIAS: Int = 1 shl 9
}

/** Sort order for [ggmlArgsort]. */
enum class GGMLSortOrder {
    ASC, DESC;
}

/** Scheduling priority for threadpool. */
enum class GGMLSchedPriority(val value: Int) {
    LOW(-1), NORMAL(0), MEDIUM(1), HIGH(2), REALTIME(3);
}

// ============================================================================
// Utility inline functions (macros from ggml.h)
// ============================================================================

/** Rounds [x] up to the nearest multiple of [n]. [n] must be a power of two. */
inline fun ggmlPad(x: Int, n: Int): Int = (x + n - 1) and (n - 1).inv()

/** Rounds [x] up to the nearest multiple of [n]. [n] must be a power of two. */
inline fun ggmlPad(x: ULong, n: ULong): ULong = (x + n - 1uL) and (n - 1uL).inv()

// ============================================================================
// Tensor-local extraction helpers (GGML_TENSOR_LOCALS equivalent)
// ============================================================================

/**
 * Extracts the [ne] (number-of-elements) array from a nullable tensor,
 * returning zeros when the tensor is null.
 */
fun tensorNe(t: GGMLTensor?): LongArray =
    t?.ne?.copyOf() ?: LongArray(GGML_MAX_DIMS)

/**
 * Extracts the [nb] (stride) array from a nullable tensor,
 * returning zeros when the tensor is null.
 */
fun tensorNb(t: GGMLTensor?): ULongArray =
    t?.nb?.copyOf() ?: ULongArray(GGML_MAX_DIMS)

// ============================================================================
// FP16 / BF16 conversion helpers
// ============================================================================

/** Convert an IEEE-754 half-precision value (stored as [UShort]) to [Float]. */
fun ggmlFp16ToFp32(h: UShort): Float = halfToFloat(h.toShort())

/** Convert a [Float] to IEEE-754 half-precision (stored as [UShort]). */
fun ggmlFp32ToFp16(f: Float): UShort = floatToHalf(f).toUShort()

/** Convert a row of FP16 values to FP32. */
fun ggmlFp16ToFp32Row(src: ShortArray, dst: FloatArray, k: Long) {
    for (i in 0 until k.toInt()) {
        dst[i] = halfToFloat(src[i])
    }
}

/** Convert a row of FP32 values to FP16. */
fun ggmlFp32ToFp16Row(src: FloatArray, dst: ShortArray, k: Long) {
    for (i in 0 until k.toInt()) {
        dst[i] = floatToHalf(src[i])
    }
}

/** Convert a BF16 value to FP32 (shift left by 16). */
fun ggmlBf16ToFp32(v: GGMLBF16): Float =
    Float.fromBits(v.bits.toInt() shl 16)

/** Convert a FP32 value to BF16 (truncate lower 16 bits). */
fun ggmlFp32ToBf16(f: Float): GGMLBF16 =
    GGMLBF16((f.toRawBits() ushr 16).toUShort())

// ============================================================================
// Tensor query helpers (ggml_nelements, ggml_nrows, …)
// ============================================================================

/** Total number of elements in [tensor]. */
fun ggmlNelements(tensor: GGMLTensor): Long = tensor.numElements()

/** Number of rows – product of ne[1]×ne[2]×ne[3]. */
fun ggmlNrows(tensor: GGMLTensor): Long =
    tensor.ne[1] * tensor.ne[2] * tensor.ne[3]

/** Total byte size of the tensor data. */
fun ggmlNbytes(tensor: GGMLTensor): ULong = calculateTensorByteSize(tensor)

/** Byte size padded to [GGML_MEM_ALIGN]. */
fun ggmlNbytesPad(tensor: GGMLTensor): ULong =
    ggmlPad(ggmlNbytes(tensor), GGML_MEM_ALIGN.toULong())

/** Block size for the given [type]. Returns 1 for non-block types. */
fun ggmlBlckSize(type: GGMLType): Long = when (type) {
    GGMLType.Q4_0 -> QK4_0.toLong()
    GGMLType.Q4_1 -> QK4_1.toLong()
    GGMLType.Q5_0 -> 32L
    GGMLType.Q5_1 -> 32L
    GGMLType.Q8_0 -> QK8_0.toLong()
    GGMLType.Q8_1 -> 32L
    GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K,
    GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K -> QK_K.toLong()
    GGMLType.BITNET_1_58 -> QK_BITNET_1_58.toLong()
    else -> 1L
}

/** Size in bytes of one complete block for [type]. */
fun ggmlTypeSize(type: GGMLType): ULong = type.byteSize

/** Size in bytes of one row of [ne] elements of the given [type]. */
fun ggmlRowSize(type: GGMLType, ne: Long): ULong {
    val blck = ggmlBlckSize(type)
    return (ne / blck).toULong() * ggmlTypeSize(type)
}

/** Human-readable name for a [GGMLType]. */
fun ggmlTypeName(type: GGMLType): String = type.description

/** Human-readable name for a [GGMLOp]. */
fun ggmlOpName(op: GGMLOp): String = op.name

/** Human-readable symbol for an operation (e.g. "+" for ADD). */
fun ggmlOpSymbol(op: GGMLOp): String = when (op) {
    GGMLOp.ADD -> "+"
    GGMLOp.SUB -> "-"
    GGMLOp.MUL -> "*"
    GGMLOp.DIV -> "/"
    GGMLOp.NEG -> "-"
    GGMLOp.SQR -> "²"
    GGMLOp.SQRT -> "√"
    GGMLOp.MUL_MAT -> "×"
    else -> op.name
}

/** Name of a [GGMLUnaryOp]. */
fun ggmlUnaryOpName(op: GGMLUnaryOp): String = op.name

/** Name of a [GGMLGluOp]. */
fun ggmlGluOpName(op: GGMLGluOp): String = op.name

/** Description string for a tensor – returns the unary op name if applicable. */
fun ggmlOpDesc(t: GGMLTensor): String = ggmlOpName(t.op)

/** Element size in bytes for the tensor's type. */
fun ggmlElementSize(tensor: GGMLTensor): ULong = ggmlTypeSize(tensor.type)

/** Whether the type is block-quantized. */
fun ggmlIsQuantized(type: GGMLType): Boolean = when (type) {
    GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1,
    GGMLType.Q8_0, GGMLType.Q8_1,
    GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K,
    GGMLType.Q6_K, GGMLType.Q8_K,
    GGMLType.BITNET_1_58, GGMLType.Q1_5_K -> true
    else -> false
}

/** Whether the tensor's memory layout is transposed (nb[0] > nb[1]). */
fun ggmlIsTransposed(tensor: GGMLTensor): Boolean =
    tensor.nb[0] > tensor.nb[1]

/** Whether the tensor is permuted (strides not in ascending order). */
fun ggmlIsPermuted(tensor: GGMLTensor): Boolean =
    tensor.nb[0] > tensor.nb[1] ||
    tensor.nb[1] > tensor.nb[2] ||
    tensor.nb[2] > tensor.nb[3]

/** Whether the tensor has zero elements. */
fun ggmlIsEmpty(tensor: GGMLTensor): Boolean =
    tensor.ne[0] == 0L || tensor.ne[1] == 0L ||
    tensor.ne[2] == 0L || tensor.ne[3] == 0L

/** Whether the tensor is a view of another tensor. */
fun ggmlIsView(tensor: GGMLTensor): Boolean =
    tensor.viewSrc != null

/** Whether the tensor is a scalar (single element). */
fun ggmlIsScalar(tensor: GGMLTensor): Boolean =
    tensor.ne[0] == 1L && tensor.ne[1] == 1L &&
    tensor.ne[2] == 1L && tensor.ne[3] == 1L

/** Whether the tensor is a 1-D vector. */
fun ggmlIsVector(tensor: GGMLTensor): Boolean =
    tensor.ne[1] == 1L && tensor.ne[2] == 1L && tensor.ne[3] == 1L

/** Whether the tensor is a 2-D matrix. */
fun ggmlIsMatrix(tensor: GGMLTensor): Boolean =
    tensor.ne[2] == 1L && tensor.ne[3] == 1L

/** Whether the tensor is 3-D. */
fun ggmlIs3d(tensor: GGMLTensor): Boolean =
    tensor.ne[3] == 1L

/** Number of dimensions (returns 1 for scalars). */
fun ggmlNDims(tensor: GGMLTensor): Int {
    for (i in GGML_MAX_DIMS - 1 downTo 1) {
        if (tensor.ne[i] > 1L) return i + 1
    }
    return 1
}

/** Whether the tensor elements are contiguous in memory. */
fun ggmlIsContiguous(tensor: GGMLTensor): Boolean =
    ggmlIsContiguous0(tensor)

/** Contiguous check starting from dimension 0. */
fun ggmlIsContiguous0(tensor: GGMLTensor): Boolean {
    if (tensor.nb[0] != ggmlTypeSize(tensor.type)) return false
    val blck = ggmlBlckSize(tensor.type)
    if (tensor.nb[1] != tensor.nb[0] * (tensor.ne[0].toULong() / blck.toULong())) return false
    if (tensor.nb[2] != tensor.nb[1] * tensor.ne[1].toULong()) return false
    if (tensor.nb[3] != tensor.nb[2] * tensor.ne[2].toULong()) return false
    return true
}

/** Contiguous for dims ≥ 1. */
fun ggmlIsContiguous1(tensor: GGMLTensor): Boolean {
    val blck = ggmlBlckSize(tensor.type)
    if (tensor.nb[1] != ggmlTypeSize(tensor.type) * (tensor.ne[0].toULong() / blck.toULong())) return false
    if (tensor.nb[2] != tensor.nb[1] * tensor.ne[1].toULong()) return false
    if (tensor.nb[3] != tensor.nb[2] * tensor.ne[2].toULong()) return false
    return true
}

/** Contiguous for dims ≥ 2. */
fun ggmlIsContiguous2(tensor: GGMLTensor): Boolean {
    if (tensor.nb[2] != tensor.nb[1] * tensor.ne[1].toULong()) return false
    if (tensor.nb[3] != tensor.nb[2] * tensor.ne[2].toULong()) return false
    return true
}

/** Whether data is contiguously allocated (no gaps, permutation ok). */
fun ggmlIsContiguouslyAllocated(tensor: GGMLTensor): Boolean {
    val sorted = tensor.nb.copyOf().also { it.sort() }
    if (sorted[0] != ggmlTypeSize(tensor.type)) return false
    for (i in 1 until GGML_MAX_DIMS) {
        // find which dim has stride sorted[i-1]
        val prevDim = (0 until GGML_MAX_DIMS).firstOrNull { tensor.nb[it] == sorted[i - 1] } ?: return false
        if (sorted[i] != sorted[i - 1] * tensor.ne[prevDim].toULong()) return false
    }
    return true
}

/** Whether two tensors have the same shape. */
fun ggmlAreSameShape(t0: GGMLTensor, t1: GGMLTensor): Boolean {
    for (i in 0 until GGML_MAX_DIMS) {
        if (t0.ne[i] != t1.ne[i]) return false
    }
    return true
}

/** Whether two tensors have the same strides. */
fun ggmlAreSameStride(t0: GGMLTensor, t1: GGMLTensor): Boolean {
    for (i in 0 until GGML_MAX_DIMS) {
        if (t0.nb[i] != t1.nb[i]) return false
    }
    return true
}

/** Whether [t0] can be broadcast/repeated to match [t1]. */
fun ggmlCanRepeat(t0: GGMLTensor, t1: GGMLTensor): Boolean {
    for (i in 0 until GGML_MAX_DIMS) {
        if (t1.ne[i] % t0.ne[i] != 0L) return false
    }
    return true
}

/** Memory overhead per tensor object (metadata only). */
fun ggmlTensorOverhead(): ULong = 256uL // approximate; actual C struct size

// ============================================================================
// Existing functions — preserved verbatim
// ============================================================================

internal fun calculateContiguousStrides(ne: LongArray, type: GGMLType, rank: Int): ULongArray {
    val nb = ULongArray(GGML_MAX_DIMS) { 0uL }

    if (type.byteSize == 0uL) {
        if (type != GGMLType.COUNT && !type.name.startsWith("Q", ignoreCase = true) && !type.name.startsWith("q", ignoreCase = true) ) {
            println("Warning: GGMLType ${type.name} has byteSize 0. Strides will be all zeros.")
        }
        return nb
    }

    nb[0] = type.byteSize
    if (GGML_MAX_DIMS > 1) {
        for (d in 1 until GGML_MAX_DIMS) {
            val dimSize = ne.getOrElse(d - 1) { 1L }
            nb[d] = nb[d - 1] * (if (dimSize > 0L) dimSize.toULong() else 1uL)
        }
    }
    return nb
}

/** Lightweight view ops used by backward/graph code: reshape, permute, transpose. */
fun reshape(context: GGMLContext, a: GGMLTensor, vararg newShape: Long): GGMLTensor {
    val out = GGMLTensor(type = a.type)
    val r = newShape.copyOf(GGML_MAX_DIMS)
    for (i in 0 until GGML_MAX_DIMS) out.ne[i] = if (i < newShape.size) r[i] else 1L
    out.nb = calculateContiguousStrides(out.ne, out.type, out.rank())
    out.viewSrc = a
    out.op = GGMLOp.RESHAPE
    return if (context.computeImmediately) out else out
}

fun permute(context: GGMLContext, a: GGMLTensor, ax0: Int, ax1: Int, ax2: Int, ax3: Int): GGMLTensor {
    val axes = intArrayOf(ax0, ax1, ax2, ax3)
    val out = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) out.ne[i] = a.ne[axes.getOrElse(i) { i }]
    out.nb = calculateContiguousStrides(out.ne, out.type, out.rank())
    out.viewSrc = a
    out.op = GGMLOp.PERMUTE
    out.opParams = axes
    return if (context.computeImmediately) out else out
}

fun transpose(context: GGMLContext, a: GGMLTensor, ax0: Int, ax1: Int): GGMLTensor {
    val axes = intArrayOf(ax0, ax1, 2, 3)
    return permute(context, a, axes[0], axes[1], axes[2], axes[3]).also { it.op = GGMLOp.TRANSPOSE }
}

/**
 * Kotlin Native port of GGML tensor operations.
 * This file contains the implementation of basic tensor operations.
 *
 * This is a placeholder implementation that will be expanded in future versions.
 */

/**
 * Creates a new tensor with the specified dimensions and type.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @return The new tensor
 */
@Suppress("unused")
fun createTensor(context: GGMLContext, type: GGMLType): GGMLTensor {
    val tensor = GGMLTensor(type = type)

    tensor.ne[0] = 1
    for (i in 1 until GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    if (context.memBuffer != null && !context.noAlloc) {
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(1) { 0.0f }
            GGMLType.I32 -> tensor.data = IntArray(1) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(1) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

/**
 * Creates a new 1-dimensional tensor.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @return The new tensor
 */
@Suppress("unused")
fun createTensor1D(context: GGMLContext, type: GGMLType, ne0: Int): GGMLTensor {
    val tensor = GGMLTensor(type = type)

    tensor.ne[0] = ne0.toLong()
    for (i in 1 until GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    if (context.memBuffer != null && !context.noAlloc) {
        val totalSize = ne0
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

/**
 * Creates a new 2-dimensional tensor.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @param ne1 The number of elements in the second dimension
 * @return The new tensor
 */
fun createTensor2D(context: GGMLContext, type: GGMLType, ne0: Int, ne1: Int): GGMLTensor {
    val tensor = GGMLTensor(type = type)

    tensor.ne[0] = ne0.toLong()
    tensor.ne[1] = ne1.toLong()
    for (i in 2 until GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    if (context.memBuffer != null && !context.noAlloc) {
        val totalSize = ne0 * ne1
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

/**
 * Creates a new 3D tensor with the specified type and dimensions.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @param ne1 The number of elements in the second dimension
 * @param ne2 The number of elements in the third dimension
 * @return The new tensor
 */
fun createTensor3D(context: GGMLContext, type: GGMLType, ne0: Int, ne1: Int, ne2: Int): GGMLTensor {
    val tensor = GGMLTensor(type = type)

    tensor.ne[0] = ne0.toLong()
    tensor.ne[1] = ne1.toLong()
    tensor.ne[2] = ne2.toLong()
    for (i in 3 until GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    if (context.memBuffer != null && !context.noAlloc) {
        val totalSize = ne0 * ne1 * ne2
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

/**
 * Creates a new 4D tensor with the specified type and dimensions.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @param ne1 The number of elements in the second dimension
 * @param ne2 The number of elements in the third dimension
 * @param ne3 The number of elements in the fourth dimension
 * @return The new tensor
 */
fun createTensor4D(context: GGMLContext, type: GGMLType, ne0: Int, ne1: Int, ne2: Int, ne3: Int): GGMLTensor {
    val tensor = GGMLTensor(type = type)

    tensor.ne[0] = ne0.toLong()
    tensor.ne[1] = ne1.toLong()
    tensor.ne[2] = ne2.toLong()
    tensor.ne[3] = ne3.toLong()

    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    if (context.memBuffer != null && !context.noAlloc) {
        val totalSize = ne0 * ne1 * ne2 * ne3
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

// ============================================================================
// Context lifecycle
// ============================================================================

/** Initialize a new GGML context with the given parameters. */
fun ggmlInit(params: GGMLInitParams): GGMLContext {
    val ctx = GGMLContext()
    ctx.memSize = params.memSize
    ctx.noAlloc = params.noAlloc
    if (params.memBuffer != null) {
        ctx.memBuffer = params.memBuffer
        ctx.memBufferOwned = false
    } else if (params.memSize > 0uL) {
        ctx.memBuffer = ByteArray(params.memSize.toInt())
        ctx.memBufferOwned = true
    }
    return ctx
}

/** Reset the context – zeros out used memory but keeps the allocation. */
fun ggmlReset(ctx: GGMLContext) {
    ctx.nObjects = 0
    ctx.objectsBegin = null
    ctx.objectsEnd = null
}

/** Free the context and optionally its owned memory buffer. */
fun ggmlFree(ctx: GGMLContext) {
    if (ctx.memBufferOwned) {
        ctx.memBuffer = null
    }
}

/** Amount of memory used by the context (approximate). */
fun ggmlUsedMem(ctx: GGMLContext): ULong = TODO("port from ggml.h")

fun ggmlGetNoAlloc(ctx: GGMLContext): Boolean = ctx.noAlloc
fun ggmlSetNoAlloc(ctx: GGMLContext, noAlloc: Boolean) { ctx.noAlloc = noAlloc }

// ============================================================================
// N-D tensor creation (Long-based, matching ggml_new_tensor signatures)
// ============================================================================

/** Create a new tensor with the given type and dimensions array. */
fun ggmlNewTensor(ctx: GGMLContext, type: GGMLType, nDims: Int, ne: LongArray): GGMLTensor {
    val tensor = GGMLTensor(type = type)
    for (i in 0 until GGML_MAX_DIMS) {
        tensor.ne[i] = if (i < nDims) ne[i] else 1L
    }
    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())
    return tensor
}

fun ggmlNewTensor1d(ctx: GGMLContext, type: GGMLType, ne0: Long): GGMLTensor =
    ggmlNewTensor(ctx, type, 1, longArrayOf(ne0))

fun ggmlNewTensor2d(ctx: GGMLContext, type: GGMLType, ne0: Long, ne1: Long): GGMLTensor =
    ggmlNewTensor(ctx, type, 2, longArrayOf(ne0, ne1))

fun ggmlNewTensor3d(ctx: GGMLContext, type: GGMLType, ne0: Long, ne1: Long, ne2: Long): GGMLTensor =
    ggmlNewTensor(ctx, type, 3, longArrayOf(ne0, ne1, ne2))

fun ggmlNewTensor4d(ctx: GGMLContext, type: GGMLType, ne0: Long, ne1: Long, ne2: Long, ne3: Long): GGMLTensor =
    ggmlNewTensor(ctx, type, 4, longArrayOf(ne0, ne1, ne2, ne3))

/** Duplicate a tensor's metadata (creates a new tensor with the same shape/type). */
fun ggmlDupTensor(ctx: GGMLContext, src: GGMLTensor): GGMLTensor {
    val t = ggmlNewTensor(ctx, src.type, ggmlNDims(src), src.ne)
    t.name = src.name
    return t
}

/** Create a view of [src] (same data, separate metadata). */
fun ggmlViewTensor(ctx: GGMLContext, src: GGMLTensor): GGMLTensor {
    val t = GGMLTensor(type = src.type)
    src.ne.copyInto(t.ne)
    src.nb.copyInto(t.nb)
    t.viewSrc = src
    t.data = src.data
    t.op = GGMLOp.VIEW
    return t
}

// ============================================================================
// Context tensor enumeration
// ============================================================================

/** Get a tensor from the context by name. */
fun ggmlGetTensor(ctx: GGMLContext, name: String): GGMLTensor? {
    TODO("port from ggml.h – requires context tensor list")
}

// ============================================================================
// Tensor metadata helpers
// ============================================================================

/** Get the unary op stored in the tensor's op_params. */
fun ggmlGetUnaryOp(tensor: GGMLTensor): GGMLUnaryOp =
    GGMLUnaryOp.entries[tensor.opParams[0]]

/** Get the glu op stored in the tensor's op_params. */
fun ggmlGetGluOp(tensor: GGMLTensor): GGMLGluOp =
    GGMLGluOp.entries[tensor.opParams[0]]

/** Get the tensor name. */
fun ggmlGetName(tensor: GGMLTensor): String = tensor.name

/** Set the tensor name (returns the tensor for chaining). */
fun ggmlSetName(tensor: GGMLTensor, name: String): GGMLTensor {
    tensor.name = name.take(GGML_MAX_NAME - 1)
    return tensor
}

// Tensor flag setters
fun ggmlSetInput(tensor: GGMLTensor) { tensor.flags = tensor.flags or GGML_TENSOR_FLAG_INPUT }
fun ggmlSetOutput(tensor: GGMLTensor) { tensor.flags = tensor.flags or GGML_TENSOR_FLAG_OUTPUT }
fun ggmlSetParam(tensor: GGMLTensor) { tensor.flags = tensor.flags or GGML_TENSOR_FLAG_PARAM  }
fun ggmlSetLoss(tensor: GGMLTensor) { tensor.flags = tensor.flags or GGML_TENSOR_FLAG_LOSS }

/** Set all elements to zero. */
fun ggmlSetZero(tensor: GGMLTensor): GGMLTensor {
    // For in-memory data arrays
    when (val d = tensor.data) {
        is FloatArray -> d.fill(0.0f)
        is IntArray -> d.fill(0)
        is LongArray -> d.fill(0L)
        is ShortArray -> d.fill(0)
        is ByteArray -> d.fill(0)
    }
    return tensor
}

// ============================================================================
// Internal helpers for graph-building ops
// ============================================================================

/**
 * Creates a result tensor for a unary operation.
 * Copies shape/strides from [a] and records [op] with source [a].
 */
private fun buildUnary(a: GGMLTensor, op: GGMLOp, inplace: Boolean = false): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    a.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = op
    result.src[0] = a
    if (inplace) result.viewSrc = a
    return result
}

/**
 * Creates a result tensor for a binary element-wise operation.
 * Copies shape/strides from [a] and records [op] with sources [a] and [b].
 */
private fun buildBinary(a: GGMLTensor, b: GGMLTensor, op: GGMLOp, inplace: Boolean = false): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    a.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = op
    result.src[0] = a
    result.src[1] = b
    if (inplace) result.viewSrc = a
    return result
}

/**
 * Creates a result tensor for a unary op routed through GGML_OP_UNARY.
 * Stores the specific [unaryOp] in opParams[0].
 */
private fun buildUnaryOp(a: GGMLTensor, unaryOp: GGMLUnaryOp, inplace: Boolean = false): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY, inplace)
    result.opParams[0] = unaryOp.ordinal
    return result
}

// ============================================================================
// Operations on tensors with backpropagation (graph-building)
// ============================================================================

/** ggml_dup – duplicate tensor data. */
fun ggmlDup(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.DUP)

/** ggml_dup_inplace – in-place duplicate (returns view). */
fun ggmlDupInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.DUP, inplace = true)

/** ggml_add – element-wise addition. */
fun add(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    buildBinary(a, b, GGMLOp.ADD)

/** ggml_add_inplace */
fun ggmlAddInplace(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    buildBinary(a, b, GGMLOp.ADD, inplace = true)

/** ggml_add_cast – add with output cast to [type]. */
fun ggmlAddCast(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, type: GGMLType): GGMLTensor {
    val result = buildBinary(a, b, GGMLOp.ADD)
    result.type = type
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    return result
}

/** ggml_sub – element-wise subtraction. */
fun sub(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    buildBinary(a, b, GGMLOp.SUB)

fun ggmlSubInplace(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    buildBinary(a, b, GGMLOp.SUB, inplace = true)

/** ggml_mul – element-wise multiplication. */
fun mul(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    buildBinary(a, b, GGMLOp.MUL)

fun ggmlMulInplace(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    buildBinary(a, b, GGMLOp.MUL, inplace = true)

/** ggml_div – element-wise division. */
fun div(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    buildBinary(a, b, GGMLOp.DIV)

fun ggmlDivInplace(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    buildBinary(a, b, GGMLOp.DIV, inplace = true)

/** ggml_sqr – element-wise square. */
fun ggmlSqr(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.SQR)

fun ggmlSqrInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.SQR, inplace = true)

/** ggml_sqrt – element-wise square root. */
fun ggmlSqrt(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.SQRT)

fun ggmlSqrtInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.SQRT, inplace = true)

/** ggml_log – element-wise natural logarithm. */
fun ggmlLog(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.LOG)

fun ggmlLogInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.LOG, inplace = true)

/** ggml_sin – element-wise sine. */
fun ggmlSin(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.ABS).also {
        // sin doesn't have its own GGMLOp – it is unary
        // But in the C header it's GGML_OP_SIN -- if your GGMLOp enum has SIN, use that
        // For now, stub with TODO
        TODO("port sin: add SIN to GGMLOp or route through UNARY")
    }

/** ggml_cos – element-wise cosine. */
fun ggmlCos(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    TODO("port cos: add COS to GGMLOp or route through UNARY")

// --- reduction ops ---

/** ggml_sum – sum all elements, return scalar. */
fun ggmlSum(ctx: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = 1L
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.SUM
    result.src[0] = a
    return result
}

/** ggml_sum_rows – sum along rows: [a,b,c,d] → [1,b,c,d]. */
fun ggmlSumRows(ctx: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = 1L
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = a.ne[i]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.SUM_ROWS
    result.src[0] = a
    return result
}

/** ggml_mean – mean along rows. */
fun ggmlMean(ctx: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = 1L
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = a.ne[i]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MEAN
    result.src[0] = a
    return result
}

/** ggml_argmax – argmax along rows, returns I32. */
fun ggmlArgmax(ctx: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.I32)
    result.ne[0] = 1L
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = a.ne[i]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.ARGMAX
    result.src[0] = a
    return result
}

/** ggml_count_equal – count equal elements between a and b. */
fun ggmlCountEqual(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = 1L
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.COUNT // reusing COUNT; the C code uses COUNT_EQUAL
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_repeat – tile a to fit shape of b. */
fun ggmlRepeat(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    b.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.REPEAT
    result.src[0] = a
    return result
}

/** ggml_repeat_back – sum repetitions of a back to shape of b. */
fun repeatBack(context: GGMLContext, a: GGMLTensor, reference: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = reference.ne[i]
    }
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.REPEAT_BACK
    result.src[0] = a
    return result
}

/** ggml_concat – concatenate a and b along [dim]. */
fun ggmlConcat(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, dim: Int): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = if (i == dim) a.ne[i] + b.ne[i] else a.ne[i]
    }
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.CONCAT
    result.src[0] = a
    result.src[1] = b
    result.opParams[0] = dim
    return result
}

// --- unary activation / element-wise ops ---

fun ggmlAbs(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.ABS)

fun ggmlAbsInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.ABS, inplace = true)

fun ggmlSgn(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.SGN)

fun ggmlSgnInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.SGN, inplace = true)

fun neg(context: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.NEG)

fun ggmlNegInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.NEG, inplace = true)

fun ggmlStep(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.STEP)

fun ggmlStepInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.STEP, inplace = true)

fun ggmlTanh(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.TANH)

fun ggmlTanhInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.TANH, inplace = true)

fun ggmlElu(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.ELU)

fun ggmlEluInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.ELU, inplace = true)

fun relu(context: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.RELU)

fun ggmlReluInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.RELU, inplace = true)

/** ggml_leaky_relu */
fun ggmlLeakyRelu(ctx: GGMLContext, a: GGMLTensor, negativeSlope: Float, inplace: Boolean): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY, inplace)
    // Store negative slope as float bits in opParams
    result.opParams[0] = GGMLUnaryOp.RELU.ordinal // base op
    result.opParams[1] = negativeSlope.toRawBits()
    return result
}

fun ggmlSigmoid(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.SIGMOID)

fun ggmlSigmoidInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.SIGMOID, inplace = true)

fun gelu(context: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.GELU)

fun ggmlGeluInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.GELU, inplace = true)

fun ggmlGeluErf(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.GELU_ERF)

fun ggmlGeluErfInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.GELU_ERF, inplace = true)

fun ggmlGeluQuick(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.GELU_QUICK)

fun ggmlGeluQuickInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.GELU_QUICK, inplace = true)

fun ggmlSilu(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.SILU)

fun ggmlSiluInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.SILU, inplace = true)

/** ggml_silu_back – a=x, b=dy. */
fun ggmlSiluBack(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    buildBinary(a, b, GGMLOp.SILU_BACK)

fun ggmlHardswish(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.HARDSWISH)

fun ggmlHardsigmoid(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.HARDSIGMOID)

fun ggmlExp(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.EXP)

fun ggmlExpInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.EXP, inplace = true)

fun ggmlExpm1(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.EXPM1)

fun ggmlExpm1Inplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.EXPM1, inplace = true)

fun ggmlSoftplus(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.SOFTPLUS)

fun ggmlSoftplusInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.SOFTPLUS, inplace = true)

fun ggmlFloor(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.FLOOR)

fun ggmlFloorInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.FLOOR, inplace = true)

fun ggmlCeil(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.CEIL)

fun ggmlCeilInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.CEIL, inplace = true)

fun ggmlRound(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.ROUND)

fun ggmlRoundInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.ROUND, inplace = true)

/** ggml_trunc – truncate fractional part toward zero. */
fun ggmlTrunc(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.TRUNC)

fun ggmlTruncInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnaryOp(a, GGMLUnaryOp.TRUNC, inplace = true)

// --- GLU ops ---

/** ggml_glu – generic gated linear unit. */
fun ggmlGlu(ctx: GGMLContext, a: GGMLTensor, op: GGMLGluOp, swapped: Boolean): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0] / 2
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = a.ne[i]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY // GLU routes through GLU op in full C impl
    result.src[0] = a
    result.opParams[0] = op.ordinal
    result.opParams[1] = if (swapped) 1 else 0
    return result
}

fun ggmlReglu(ctx: GGMLContext, a: GGMLTensor): GGMLTensor = ggmlGlu(ctx, a, GGMLGluOp.REGLU, false)
fun ggmlRegluSwapped(ctx: GGMLContext, a: GGMLTensor): GGMLTensor = ggmlGlu(ctx, a, GGMLGluOp.REGLU, true)
fun ggmlGeglu(ctx: GGMLContext, a: GGMLTensor): GGMLTensor = ggmlGlu(ctx, a, GGMLGluOp.GEGLU, false)
fun ggmlGegluSwapped(ctx: GGMLContext, a: GGMLTensor): GGMLTensor = ggmlGlu(ctx, a, GGMLGluOp.GEGLU, true)
fun ggmlSwiglu(ctx: GGMLContext, a: GGMLTensor): GGMLTensor = ggmlGlu(ctx, a, GGMLGluOp.SWIGLU, false)
fun ggmlSwigluSwapped(ctx: GGMLContext, a: GGMLTensor): GGMLTensor = ggmlGlu(ctx, a, GGMLGluOp.SWIGLU, true)

// --- normalization ---

/** ggml_norm – layer normalization along rows. */
fun ggmlNorm(ctx: GGMLContext, a: GGMLTensor, eps: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.NORM)
    result.opParams[0] = eps.toRawBits()
    return result
}

fun ggmlNormInplace(ctx: GGMLContext, a: GGMLTensor, eps: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.NORM, inplace = true)
    result.opParams[0] = eps.toRawBits()
    return result
}

/** ggml_rms_norm – RMS normalization along rows. */
fun ggmlRmsNorm(ctx: GGMLContext, a: GGMLTensor, eps: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.RMS_NORM)
    result.opParams[0] = eps.toRawBits()
    return result
}

fun ggmlRmsNormInplace(ctx: GGMLContext, a: GGMLTensor, eps: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.RMS_NORM, inplace = true)
    result.opParams[0] = eps.toRawBits()
    return result
}

/** ggml_group_norm */
fun ggmlGroupNorm(ctx: GGMLContext, a: GGMLTensor, nGroups: Int, eps: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY) // GROUP_NORM not in existing enum
    result.opParams[0] = nGroups
    result.opParams[1] = eps.toRawBits()
    return result
}

fun ggmlGroupNormInplace(ctx: GGMLContext, a: GGMLTensor, nGroups: Int, eps: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY, inplace = true)
    result.opParams[0] = nGroups
    result.opParams[1] = eps.toRawBits()
    return result
}

/** ggml_rms_norm_back – a=x, b=dy */
fun ggmlRmsNormBack(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, eps: Float): GGMLTensor {
    val result = buildBinary(a, b, GGMLOp.RMS_NORM_BACK)
    result.opParams[0] = eps.toRawBits()
    return result
}

// --- matrix multiplication ---

/**
 * ggml_mul_mat – matrix multiplication.
 * A: [ne03, ne02, n, k], B: [ne03*x, ne02*y, m, k]  → result: [ne03*x, ne02*y, m, n]
 */
fun matMul(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = a.ne[1] // n
    result.ne[1] = b.ne[1] // m
    result.ne[2] = b.ne[2]
    result.ne[3] = b.ne[3]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MUL_MAT
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_mul_mat_set_prec – set precision for mul_mat. */
fun ggmlMulMatSetPrec(a: GGMLTensor, prec: GGMLPrec) {
    a.opParams[0] = prec.value
}

/** ggml_mul_mat_id – indirect matrix multiplication with index tensor. */
fun ggmlMulMatId(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, ids: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = a.ne[1]
    result.ne[1] = b.ne[1]
    result.ne[2] = b.ne[2]
    result.ne[3] = b.ne[3]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MUL_MAT
    result.src[0] = a
    result.src[1] = b
    result.src[2] = ids
    return result
}

/** ggml_out_prod – outer product: A[m,n] × B[p,n] → result[m,p]. */
fun ggmlOutProd(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = a.ne[0]
    result.ne[1] = b.ne[0]
    result.ne[2] = b.ne[2]
    result.ne[3] = b.ne[3]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_BINARY // OUT_PROD not in existing enum
    result.src[0] = a
    result.src[1] = b
    return result
}

// --- operations without backpropagation ---

/** ggml_scale – multiply all elements by scalar [s]. */
fun ggmlScale(ctx: GGMLContext, a: GGMLTensor, s: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.SCALE)
    result.opParams[0] = s.toRawBits()
    return result
}

fun ggmlScaleInplace(ctx: GGMLContext, a: GGMLTensor, s: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.SCALE, inplace = true)
    result.opParams[0] = s.toRawBits()
    return result
}

/** ggml_scale_bias – x = s*a + b */
fun ggmlScaleBias(ctx: GGMLContext, a: GGMLTensor, s: Float, b: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.SCALE)
    result.opParams[0] = s.toRawBits()
    result.opParams[1] = b.toRawBits()
    return result
}

fun ggmlScaleBiasInplace(ctx: GGMLContext, a: GGMLTensor, s: Float, b: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.SCALE, inplace = true)
    result.opParams[0] = s.toRawBits()
    result.opParams[1] = b.toRawBits()
    return result
}

/** ggml_set – copy b into a view of a at the given offset/strides. */
fun ggmlSet(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, nb1: ULong, nb2: ULong, nb3: ULong, offset: ULong): GGMLTensor {
    val result = buildBinary(a, b, GGMLOp.MAP_BINARY)
    result.opParams[0] = nb1.toInt()
    result.opParams[1] = nb2.toInt()
    result.opParams[2] = nb3.toInt()
    result.opParams[3] = offset.toInt()
    return result
}

/** ggml_cpy – copy a into b, return view(b). */
fun ggmlCpy(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = b.type)
    b.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.CPY
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_cast – type cast of a to [type]. */
fun ggmlCast(ctx: GGMLContext, a: GGMLTensor, type: GGMLType): GGMLTensor {
    val result = GGMLTensor(type = type)
    a.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.CPY
    result.src[0] = a
    return result
}

/** ggml_cont – make contiguous. */
fun ggmlCont(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.MAP_UNARY) // CONT not in existing enum; could add later

/** ggml_cont_1d – make contiguous with new 1-D shape. */
fun ggmlCont1d(ctx: GGMLContext, a: GGMLTensor, ne0: Long): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = ne0
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

/** ggml_cont_2d */
fun ggmlCont2d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1
    for (i in 2 until GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

/** ggml_cont_3d */
fun ggmlCont3d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long, ne2: Long): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

/** ggml_cont_4d */
fun ggmlCont4d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = ne3
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

// --- reshape (ggml.h style, using tensor ref or explicit dims) ---

/** ggml_reshape – reshape a to match the shape of b. */
fun ggmlReshape(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor =
    reshape(ctx, a, *b.ne)

/** ggml_reshape_1d */
fun ggmlReshape1d(ctx: GGMLContext, a: GGMLTensor, ne0: Long): GGMLTensor =
    reshape(ctx, a, ne0)

/** ggml_reshape_2d */
fun ggmlReshape2d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long): GGMLTensor =
    reshape(ctx, a, ne0, ne1)

/** ggml_reshape_3d */
fun ggmlReshape3d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long, ne2: Long): GGMLTensor =
    reshape(ctx, a, ne0, ne1, ne2)

/** ggml_reshape_4d */
fun ggmlReshape4d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long): GGMLTensor =
    reshape(ctx, a, ne0, ne1, ne2, ne3)

// --- views ---

/** ggml_view_1d – 1-D view with byte offset. */
fun ggmlView1d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, offset: ULong): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = ne0
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.VIEW
    result.viewSrc = a
    result.viewOffs = offset
    result.src[0] = a
    return result
}

/** ggml_view_2d */
fun ggmlView2d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long, nb1: ULong, offset: ULong): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1
    for (i in 2 until GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb[0] = a.type.byteSize
    result.nb[1] = nb1
    result.nb[2] = nb1 * ne1.toULong()
    result.nb[3] = result.nb[2]
    result.op = GGMLOp.VIEW
    result.viewSrc = a
    result.viewOffs = offset
    result.src[0] = a
    return result
}

/** ggml_view_3d */
fun ggmlView3d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long, ne2: Long, nb1: ULong, nb2: ULong, offset: ULong): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = 1L
    result.nb[0] = a.type.byteSize
    result.nb[1] = nb1
    result.nb[2] = nb2
    result.nb[3] = nb2 * ne2.toULong()
    result.op = GGMLOp.VIEW
    result.viewSrc = a
    result.viewOffs = offset
    result.src[0] = a
    return result
}

/** ggml_view_4d */
fun ggmlView4d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long, nb1: ULong, nb2: ULong, nb3: ULong, offset: ULong): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = ne3
    result.nb[0] = a.type.byteSize
    result.nb[1] = nb1
    result.nb[2] = nb2
    result.nb[3] = nb3
    result.op = GGMLOp.VIEW
    result.viewSrc = a
    result.viewOffs = offset
    result.src[0] = a
    return result
}

/** ggml_permute */
fun ggmlPermute(ctx: GGMLContext, a: GGMLTensor, axis0: Int, axis1: Int, axis2: Int, axis3: Int): GGMLTensor =
    permute(ctx, a, axis0, axis1, axis2, axis3)

/** ggml_transpose – alias for permute(1,0,2,3). */
fun ggmlTranspose(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    permute(ctx, a, 1, 0, 2, 3).also { it.op = GGMLOp.TRANSPOSE }

// --- get_rows / set_rows ---

/** ggml_get_rows – gather rows from a using indices in b. */
fun ggmlGetRows(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = a.ne[0]
    result.ne[1] = b.ne[0]
    result.ne[2] = b.ne[1]
    result.ne[3] = b.ne[2]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.GET_ROWS
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_get_rows_back */
fun ggmlGetRowsBack(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, c: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    c.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.GET_ROWS
    result.src[0] = a
    result.src[1] = b
    result.src[2] = c
    return result
}

// --- diag_mask ---

/** ggml_diag_mask_inf – set elements above the diagonal to -INF. */
fun ggmlDiagMaskInf(ctx: GGMLContext, a: GGMLTensor, nPast: Int): GGMLTensor {
    val result = buildUnary(a, GGMLOp.DIAG_MASK_INF)
    result.opParams[0] = nPast
    return result
}

fun ggmlDiagMaskInfInplace(ctx: GGMLContext, a: GGMLTensor, nPast: Int): GGMLTensor {
    val result = buildUnary(a, GGMLOp.DIAG_MASK_INF, inplace = true)
    result.opParams[0] = nPast
    return result
}

/** ggml_diag_mask_zero – set elements above the diagonal to 0. */
fun ggmlDiagMaskZero(ctx: GGMLContext, a: GGMLTensor, nPast: Int): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY)
    result.opParams[0] = nPast
    return result
}

fun ggmlDiagMaskZeroInplace(ctx: GGMLContext, a: GGMLTensor, nPast: Int): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY, inplace = true)
    result.opParams[0] = nPast
    return result
}

// --- softmax ---

/** ggml_soft_max */
fun ggmlSoftMax(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.SOFT_MAX)

fun ggmlSoftMaxInplace(ctx: GGMLContext, a: GGMLTensor): GGMLTensor =
    buildUnary(a, GGMLOp.SOFT_MAX, inplace = true)

/** ggml_soft_max_ext – fused soft_max(a*scale + mask*(ALiBi slope)). */
fun ggmlSoftMaxExt(ctx: GGMLContext, a: GGMLTensor, mask: GGMLTensor?, scale: Float, maxBias: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.SOFT_MAX)
    result.src[1] = mask
    result.opParams[0] = scale.toRawBits()
    result.opParams[1] = maxBias.toRawBits()
    return result
}

fun ggmlSoftMaxExtInplace(ctx: GGMLContext, a: GGMLTensor, mask: GGMLTensor?, scale: Float, maxBias: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.SOFT_MAX, inplace = true)
    result.src[1] = mask
    result.opParams[0] = scale.toRawBits()
    result.opParams[1] = maxBias.toRawBits()
    return result
}

// --- RoPE ---

/** ggml_rope – rotary position embedding. */
fun ggmlRope(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, nDims: Int, mode: Int): GGMLTensor {
    val result = buildUnary(a, GGMLOp.ROPE)
    result.src[1] = b
    result.opParams[0] = nDims
    result.opParams[1] = mode
    return result
}

fun ggmlRopeInplace(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, nDims: Int, mode: Int): GGMLTensor {
    val result = buildUnary(a, GGMLOp.ROPE, inplace = true)
    result.src[1] = b
    result.opParams[0] = nDims
    result.opParams[1] = mode
    return result
}

/**
 * ggml_rope_ext – extended RoPE with YaRN scaling parameters.
 * @param a input tensor
 * @param b positions (I32 vector)
 * @param c freq factors (optional)
 */
fun ggmlRopeExt(
    ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, c: GGMLTensor?,
    nDims: Int, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): GGMLTensor {
    val result = buildUnary(a, GGMLOp.ROPE)
    result.src[1] = b
    result.src[2] = c
    result.opParams[0] = nDims
    result.opParams[1] = mode
    result.opParams[2] = nCtxOrig
    result.opParams[3] = freqBase.toRawBits()
    result.opParams[4] = freqScale.toRawBits()
    result.opParams[5] = extFactor.toRawBits()
    result.opParams[6] = attnFactor.toRawBits()
    result.opParams[7] = betaFast.toRawBits()
    result.opParams[8] = betaSlow.toRawBits()
    return result
}

fun ggmlRopeExtInplace(
    ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, c: GGMLTensor?,
    nDims: Int, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): GGMLTensor {
    val result = buildUnary(a, GGMLOp.ROPE, inplace = true)
    result.src[1] = b
    result.src[2] = c
    result.opParams[0] = nDims
    result.opParams[1] = mode
    result.opParams[2] = nCtxOrig
    result.opParams[3] = freqBase.toRawBits()
    result.opParams[4] = freqScale.toRawBits()
    result.opParams[5] = extFactor.toRawBits()
    result.opParams[6] = attnFactor.toRawBits()
    result.opParams[7] = betaFast.toRawBits()
    result.opParams[8] = betaSlow.toRawBits()
    return result
}

/** ggml_rope_multi – multi-dimensional RoPE for Qwen-VL and similar. */
fun ggmlRopeMulti(
    ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, c: GGMLTensor?,
    nDims: Int, sections: IntArray, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): GGMLTensor {
    val result = ggmlRopeExt(ctx, a, b, c, nDims, mode, nCtxOrig, freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow)
    // Store sections in remaining opParams
    for (i in 0 until minOf(sections.size, GGML_MROPE_SECTIONS)) {
        result.opParams[9 + i] = sections[i]
    }
    return result
}

/** ggml_clamp – clamp values to [min, max]. */
fun ggmlClamp(ctx: GGMLContext, a: GGMLTensor, min: Float, max: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY, inplace = true)
    result.opParams[0] = min.toRawBits()
    result.opParams[1] = max.toRawBits()
    return result
}

// --- convolution ---

/** ggml_conv_1d */
fun ggmlConv1d(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, s0: Int, p0: Int, d0: Int): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    val ks = a.ne[0]
    val outLen = ((b.ne[0] + 2 * p0 - d0 * (ks - 1) - 1) / s0) + 1
    result.ne[0] = outLen
    result.ne[1] = a.ne[2]
    result.ne[2] = b.ne[2]
    result.ne[3] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.CONV_1D_1S // approximate mapping
    result.src[0] = a
    result.src[1] = b
    result.opParams[0] = s0; result.opParams[1] = p0; result.opParams[2] = d0
    return result
}

/** ggml_conv_2d */
fun ggmlConv2d(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, s0: Int, s1: Int, p0: Int, p1: Int, d0: Int, d1: Int): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = ((b.ne[0] + 2 * p0 - d0 * (a.ne[0] - 1) - 1) / s0) + 1
    result.ne[1] = ((b.ne[1] + 2 * p1 - d1 * (a.ne[1] - 1) - 1) / s1) + 1
    result.ne[2] = a.ne[3] // output channels
    result.ne[3] = b.ne[3] // batch
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.CONV_1D_2S // approximate mapping
    result.src[0] = a
    result.src[1] = b
    result.opParams[0] = s0; result.opParams[1] = s1
    result.opParams[2] = p0; result.opParams[3] = p1
    result.opParams[4] = d0; result.opParams[5] = d1
    return result
}

// --- pooling ---

/** ggml_pool_1d */
fun ggmlPool1d(ctx: GGMLContext, a: GGMLTensor, op: GGMLOpPool, k0: Int, s0: Int, p0: Int): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = (a.ne[0] + 2 * p0 - k0) / s0 + 1
    result.ne[1] = a.ne[1]
    result.ne[2] = a.ne[2]
    result.ne[3] = a.ne[3]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = op.ordinal; result.opParams[1] = k0; result.opParams[2] = s0; result.opParams[3] = p0
    return result
}

/** ggml_pool_2d */
fun ggmlPool2d(ctx: GGMLContext, a: GGMLTensor, op: GGMLOpPool, k0: Int, k1: Int, s0: Int, s1: Int, p0: Float, p1: Float): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = ((a.ne[0] + (2 * p0).toLong() - k0) / s0) + 1
    result.ne[1] = ((a.ne[1] + (2 * p1).toLong() - k1) / s1) + 1
    result.ne[2] = a.ne[2]
    result.ne[3] = a.ne[3]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

// --- upscale / pad / roll ---

/** ggml_upscale – multiply ne0 and ne1 by scaleFactor. */
fun ggmlUpscale(ctx: GGMLContext, a: GGMLTensor, scaleFactor: Int, mode: GGMLScaleMode): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0] * scaleFactor
    result.ne[1] = a.ne[1] * scaleFactor
    result.ne[2] = a.ne[2]
    result.ne[3] = a.ne[3]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = scaleFactor
    result.opParams[1] = mode.value
    return result
}

/** ggml_interpolate – resize to target dimensions. */
fun ggmlInterpolate(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long, mode: Int): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = ne3
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = mode
    return result
}

/** ggml_pad – zero-pad each dimension. */
fun ggmlPad(ctx: GGMLContext, a: GGMLTensor, p0: Int, p1: Int, p2: Int, p3: Int): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0] + p0
    result.ne[1] = a.ne[1] + p1
    result.ne[2] = a.ne[2] + p2
    result.ne[3] = a.ne[3] + p3
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = p0; result.opParams[1] = p1; result.opParams[2] = p2; result.opParams[3] = p3
    return result
}

/** ggml_roll – circular shift each dimension. */
fun ggmlRoll(ctx: GGMLContext, a: GGMLTensor, shift0: Int, shift1: Int, shift2: Int, shift3: Int): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY)
    result.opParams[0] = shift0; result.opParams[1] = shift1
    result.opParams[2] = shift2; result.opParams[3] = shift3
    return result
}

/** ggml_tri – triangular matrix mask. */
fun ggmlTri(ctx: GGMLContext, a: GGMLTensor, type: GGMLTriType): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY)
    result.opParams[0] = type.value
    return result
}

/** ggml_fill – fill with constant c. */
fun ggmlFill(ctx: GGMLContext, a: GGMLTensor, c: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY)
    result.opParams[0] = c.toRawBits()
    return result
}

fun ggmlFillInplace(ctx: GGMLContext, a: GGMLTensor, c: Float): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY, inplace = true)
    result.opParams[0] = c.toRawBits()
    return result
}

// --- sort / top-k ---

/** ggml_argsort */
fun ggmlArgsort(ctx: GGMLContext, a: GGMLTensor, order: GGMLSortOrder): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.I32)
    a.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.ARGMAX // reuse; actual dispatch uses opParams
    result.src[0] = a
    result.opParams[0] = order.ordinal
    return result
}

/** ggml_top_k – top k elements per row. */
fun ggmlTopK(ctx: GGMLContext, a: GGMLTensor, k: Int): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = k.toLong()
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = a.ne[i]
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = k
    return result
}

/** ggml_arange – creates a 1-D tensor [start, start+step, …, stop). */
fun ggmlArange(ctx: GGMLContext, start: Float, stop: Float, step: Float): GGMLTensor {
    val n = ((stop - start) / step).toLong()
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = n
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.opParams[0] = start.toRawBits()
    result.opParams[1] = stop.toRawBits()
    result.opParams[2] = step.toRawBits()
    return result
}

// --- flash attention ---

/**
 * ggml_flash_attn_ext
 * q:    [n_embd_k, n_batch, n_head,    ne3]
 * k:    [n_embd_k, n_kv,    n_head_kv, ne3]
 * v:    [n_embd_v, n_kv,    n_head_kv, ne3]  (not transposed)
 * mask: [n_kv,     n_batch, ne32,      ne33]  (optional)
 * result: [n_embd_v, n_head, n_batch, ne3]    (permuted)
 */
fun ggmlFlashAttnExt(
    ctx: GGMLContext,
    q: GGMLTensor, k: GGMLTensor, v: GGMLTensor, mask: GGMLTensor?,
    scale: Float, maxBias: Float, logitSoftcap: Float
): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = v.ne[0]  // n_embd_v
    result.ne[1] = q.ne[2]  // n_head
    result.ne[2] = q.ne[1]  // n_batch
    result.ne[3] = q.ne[3]  // ne3
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.FLASH_ATTN
    result.src[0] = q
    result.src[1] = k
    result.src[2] = v
    result.src[3] = mask
    result.opParams[0] = scale.toRawBits()
    result.opParams[1] = maxBias.toRawBits()
    result.opParams[2] = logitSoftcap.toRawBits()
    return result
}

/** ggml_flash_attn_ext_set_prec */
fun ggmlFlashAttnExtSetPrec(a: GGMLTensor, prec: GGMLPrec) {
    a.opParams[3] = prec.value
}

/** ggml_flash_attn_back */
fun ggmlFlashAttnBack(ctx: GGMLContext, q: GGMLTensor, k: GGMLTensor, v: GGMLTensor, d: GGMLTensor, masked: Boolean): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    q.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.FLASH_ATTN
    result.src[0] = q; result.src[1] = k; result.src[2] = v; result.src[3] = d
    result.opParams[0] = if (masked) 1 else 0
    return result
}

// --- SSM ops ---

/** ggml_ssm_conv */
fun ggmlSsmConv(ctx: GGMLContext, sx: GGMLTensor, c: GGMLTensor): GGMLTensor {
    val result = buildBinary(sx, c, GGMLOp.MAP_BINARY)
    return result
}

/** ggml_ssm_scan */
fun ggmlSsmScan(ctx: GGMLContext, s: GGMLTensor, x: GGMLTensor, dt: GGMLTensor, a: GGMLTensor, b: GGMLTensor, c: GGMLTensor, ids: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    s.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = s; result.src[1] = x; result.src[2] = dt
    result.src[3] = a; result.src[4] = b; result.src[5] = c; result.src[6] = ids
    return result
}

// --- window partition (SAM) ---

/** ggml_win_part */
fun ggmlWinPart(ctx: GGMLContext, a: GGMLTensor, w: Int): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0]
    result.ne[1] = w.toLong()
    result.ne[2] = w.toLong()
    val nw0 = (a.ne[1].toInt() + w - 1) / w
    val nw1 = (a.ne[2].toInt() + w - 1) / w
    result.ne[3] = (nw0 * nw1).toLong()
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = w
    return result
}

/** ggml_win_unpart */
fun ggmlWinUnpart(ctx: GGMLContext, a: GGMLTensor, w0: Int, h0: Int, w: Int): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0]
    result.ne[1] = w0.toLong()
    result.ne[2] = h0.toLong()
    result.ne[3] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = w
    return result
}

// --- unary dispatch ---

/** ggml_unary – dispatch a unary op. */
fun ggmlUnary(ctx: GGMLContext, a: GGMLTensor, op: GGMLUnaryOp): GGMLTensor =
    buildUnaryOp(a, op)

fun ggmlUnaryInplace(ctx: GGMLContext, a: GGMLTensor, op: GGMLUnaryOp): GGMLTensor =
    buildUnaryOp(a, op, inplace = true)

// --- relative position (SAM) ---

/** ggml_get_rel_pos */
fun ggmlGetRelPos(ctx: GGMLContext, a: GGMLTensor, qh: Int, kh: Int): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0]
    result.ne[1] = kh.toLong()
    result.ne[2] = qh.toLong()
    result.ne[3] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

/** ggml_add_rel_pos */
fun ggmlAddRelPos(ctx: GGMLContext, a: GGMLTensor, pw: GGMLTensor, ph: GGMLTensor): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY)
    result.src[1] = pw
    result.src[2] = ph
    return result
}

fun ggmlAddRelPosInplace(ctx: GGMLContext, a: GGMLTensor, pw: GGMLTensor, ph: GGMLTensor): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY, inplace = true)
    result.src[1] = pw
    result.src[2] = ph
    return result
}

// --- RWKV ---

/** ggml_rwkv_wkv6 */
fun ggmlRwkvWkv6(ctx: GGMLContext, k: GGMLTensor, v: GGMLTensor, r: GGMLTensor, tf: GGMLTensor, td: GGMLTensor, state: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    k.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = k; result.src[1] = v; result.src[2] = r
    result.src[3] = tf; result.src[4] = td; result.src[5] = state
    return result
}

/** ggml_rwkv_wkv7 */
fun ggmlRwkvWkv7(ctx: GGMLContext, r: GGMLTensor, w: GGMLTensor, k: GGMLTensor, v: GGMLTensor, a: GGMLTensor, b: GGMLTensor, state: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    r.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = r; result.src[1] = w; result.src[2] = k
    result.src[3] = v; result.src[4] = a; result.src[5] = b; result.src[6] = state
    return result
}

/** ggml_gated_linear_attn */
fun ggmlGatedLinearAttn(ctx: GGMLContext, k: GGMLTensor, v: GGMLTensor, q: GGMLTensor, g: GGMLTensor, state: GGMLTensor, scale: Float): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    k.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_UNARY
    result.src[0] = k; result.src[1] = v; result.src[2] = q
    result.src[3] = g; result.src[4] = state
    result.opParams[0] = scale.toRawBits()
    return result
}

// --- loss ---

/** ggml_cross_entropy_loss */
fun ggmlCrossEntropyLoss(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = 1L
    for (i in 1 until GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_BINARY
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_cross_entropy_loss_back */
fun ggmlCrossEntropyLossBack(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, c: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    a.ne.copyInto(result.ne)
    result.nb = calculateContiguousStrides(result.ne, result.type, result.rank())
    result.op = GGMLOp.MAP_BINARY
    result.src[0] = a; result.src[1] = b; result.src[2] = c
    return result
}

// --- optimizers ---

/** ggml_opt_step_adamw */
fun ggmlOptStepAdamw(ctx: GGMLContext, a: GGMLTensor, grad: GGMLTensor, m: GGMLTensor, v: GGMLTensor, adamwParams: GGMLTensor): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY)
    result.src[1] = grad; result.src[2] = m; result.src[3] = v; result.src[4] = adamwParams
    return result
}

/** ggml_opt_step_sgd */
fun ggmlOptStepSgd(ctx: GGMLContext, a: GGMLTensor, grad: GGMLTensor, sgdParams: GGMLTensor): GGMLTensor {
    val result = buildUnary(a, GGMLOp.MAP_UNARY)
    result.src[1] = grad; result.src[2] = sgdParams
    return result
}

// ============================================================================
// Graph construction
// ============================================================================

/** ggml_new_graph – allocate a graph with default size and no gradient tracking. */
fun ggmlNewGraph(ctx: GGMLContext): GGMLCGraph {
    return ggmlNewGraphCustom(ctx, GGML_DEFAULT_GRAPH_SIZE.toULong(), grads = false)
}

/** ggml_new_graph_custom */
fun ggmlNewGraphCustom(ctx: GGMLContext, size: ULong, grads: Boolean): GGMLCGraph {
    val s = size.toInt()
    return GGMLCGraph(
        size = s,
        nodes = arrayOfNulls(s),
        grads = if (grads) arrayOfNulls(s) else emptyArray(),
        leafs = arrayOfNulls(s)
    )
}

/** ggml_build_forward_expand – recursively add tensor and its sources to the graph. */
fun ggmlBuildForwardExpand(cgraph: GGMLCGraph, tensor: GGMLTensor) {
    // Avoid duplicates (simple linear scan; real impl uses hash set)
    for (i in 0 until cgraph.nNodes) {
        if (cgraph.nodes[i] === tensor) return
    }
    // Recurse into sources first
    for (src in tensor.src) {
        if (src != null) {
            ggmlBuildForwardExpand(cgraph, src)
        }
    }
    // Add this tensor as a node (or leaf if NONE op)
    if (tensor.op == GGMLOp.NONE) {
        if (cgraph.nLeafs < cgraph.leafs.size) {
            cgraph.leafs[cgraph.nLeafs++] = tensor
        }
    } else {
        if (cgraph.nNodes < cgraph.nodes.size) {
            cgraph.nodes[cgraph.nNodes++] = tensor
        }
    }
}

/** ggml_graph_clear – reset all nodes and leafs. */
fun ggmlGraphClear(cgraph: GGMLCGraph) {
    cgraph.nNodes = 0
    cgraph.nLeafs = 0
    for (i in cgraph.nodes.indices) cgraph.nodes[i] = null
    for (i in cgraph.leafs.indices) cgraph.leafs[i] = null
}

/** ggml_graph_size */
fun ggmlGraphSize(cgraph: GGMLCGraph): Int = cgraph.size

/** ggml_graph_node – get node at index i (negative indices count from end). */
fun ggmlGraphNode(cgraph: GGMLCGraph, i: Int): GGMLTensor? =
    if (i < 0) cgraph.nodes[cgraph.nNodes + i] else cgraph.nodes[i]

/** ggml_graph_n_nodes */
fun ggmlGraphNNodes(cgraph: GGMLCGraph): Int = cgraph.nNodes

/** ggml_graph_add_node – append a tensor to the graph. */
fun ggmlGraphAddNode(cgraph: GGMLCGraph, tensor: GGMLTensor) {
    if (cgraph.nNodes < cgraph.nodes.size) {
        cgraph.nodes[cgraph.nNodes++] = tensor
    }
}

/** ggml_graph_get_tensor – find a tensor by name. */
fun ggmlGraphGetTensor(cgraph: GGMLCGraph, name: String): GGMLTensor? {
    for (i in 0 until cgraph.nNodes) {
        if (cgraph.nodes[i]?.name == name) return cgraph.nodes[i]
    }
    for (i in 0 until cgraph.nLeafs) {
        if (cgraph.leafs[i]?.name == name) return cgraph.leafs[i]
    }
    return null
}

/** ggml_graph_overhead – estimated memory overhead for a graph. */
fun ggmlGraphOverhead(): ULong = ggmlGraphOverheadCustom(GGML_DEFAULT_GRAPH_SIZE.toULong(), false)

/** ggml_graph_overhead_custom */
fun ggmlGraphOverheadCustom(size: ULong, grads: Boolean): ULong {
    // Each pointer is ~8 bytes on 64-bit, plus bookkeeping
    val ptrSize = 8uL
    val nodesSize = size * ptrSize
    val gradsSize = if (grads) size * ptrSize else 0uL
    return nodesSize + gradsSize + 256uL // struct overhead
}

// ============================================================================
// Type traits (ggml_type_traits)
// ============================================================================

/**
 * ggml_type_traits – metadata about a tensor data type.
 */
data class GGMLTypeTraits(
    val typeName: String,
    val blckSize: Long,
    val typeSize: ULong,
    val isQuantized: Boolean
)

/** ggml_get_type_traits */
fun ggmlGetTypeTraits(type: GGMLType): GGMLTypeTraits = GGMLTypeTraits(
    typeName = ggmlTypeName(type),
    blckSize = ggmlBlckSize(type),
    typeSize = ggmlTypeSize(type),
    isQuantized = ggmlIsQuantized(type)
)

// ============================================================================
// Threadpool params (stub)
// ============================================================================

/**
 * ggml_threadpool_params – configuration for the compute threadpool.
 */
data class GGMLThreadpoolParams(
    val cpuMask: BooleanArray = BooleanArray(GGML_MAX_N_THREADS),
    var nThreads: Int = GGML_DEFAULT_N_THREADS,
    var priority: GGMLSchedPriority = GGMLSchedPriority.NORMAL,
    var poll: Int = 0,
    var strictCpu: Boolean = false,
    var paused: Boolean = false
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GGMLThreadpoolParams) return false
        return nThreads == other.nThreads && priority == other.priority
    }
    override fun hashCode(): Int = nThreads * 31 + priority.hashCode()
}

/** ggml_threadpool_params_default */
fun ggmlThreadpoolParamsDefault(nThreads: Int): GGMLThreadpoolParams =
    GGMLThreadpoolParams(nThreads = nThreads)

// =============================================================================
// Stub functions needed by llm_graph_context but not yet fully implemented
// =============================================================================

/** ggml_swiglu_split – SwiGLU with separate gate and up tensors. */
fun ggmlSwigluSplit(ctx: GGMLContext, gate: GGMLTensor, up: GGMLTensor): GGMLTensor {
    TODO("Port ggml_swiglu_split")
}

/** ggml_geglu_split – GeGLU with separate gate and up tensors. */
fun ggmlGegluSplit(ctx: GGMLContext, gate: GGMLTensor, up: GGMLTensor): GGMLTensor {
    TODO("Port ggml_geglu_split")
}

/** ggml_reglu_split – ReGLU with separate gate and up tensors. */
fun ggmlRegluSplit(ctx: GGMLContext, gate: GGMLTensor, up: GGMLTensor): GGMLTensor {
    TODO("Port ggml_reglu_split")
}

/** ggml_swiglu_oai – OpenAI-style SwiGLU with alpha and limit parameters. */
fun ggmlSwigluOai(ctx: GGMLContext, gate: GGMLTensor, up: GGMLTensor, alpha: Float, limit: Float): GGMLTensor {
    TODO("Port ggml_swiglu_oai")
}

/** ggml_argsort_top_k – Return indices of top-k elements per row. */
fun ggmlArgsortTopK(ctx: GGMLContext, a: GGMLTensor, k: Long): GGMLTensor {
    TODO("Port ggml_argsort_top_k")
}

/** ggml_set_rows – scatter rows from [src] into [dst] at positions given by [ids]. */
fun ggmlSetRows(ctx: GGMLContext, dst: GGMLTensor, src: GGMLTensor, ids: GGMLTensor): GGMLTensor {
    TODO("Port ggml_set_rows")
}

/** ggml_add_id – element-wise add with expert-id routing. */
fun ggmlAddId(ctx: GGMLContext, a: GGMLTensor, b: GGMLTensor, ids: GGMLTensor): GGMLTensor {
    TODO("Port ggml_add_id")
}

/** ggml_repeat_4d – repeat tensor to fill target shape. */
fun ggmlRepeat4d(ctx: GGMLContext, a: GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long): GGMLTensor {
    TODO("Port ggml_repeat_4d")
}

/** ggml_format_name – format a tensor name using printf-style syntax. */
fun ggmlFormatName(tensor: GGMLTensor, format: String, vararg args: Any) {
    tensor.name = if (args.isEmpty()) format else {
        var result = format
        for (arg in args) result = result.replaceFirst("%d", arg.toString()).replaceFirst("%s", arg.toString())
        result
    }
}

/**
 * ggml_build_forward_select – select one of multiple tensors for the forward pass.
 *
 * Returns the selected tensor and adds all alternatives to the graph.
 */
fun ggmlBuildForwardSelect(gf: GGMLCGraph, tensors: Array<GGMLTensor?>, count: Int, selected: Int): GGMLTensor {
    for (i in 0 until count) {
        val t = tensors[i] ?: continue
        ggmlBuildForwardExpand(gf, t)
    }
    return tensors[selected] ?: error("Selected tensor at index $selected is null")
}

/** ggml_flash_attn_ext_add_sinks – attach attention-sink vector to flash attention node. */
fun ggmlFlashAttnExtAddSinks(fattn: GGMLTensor, sinks: GGMLTensor?) {
    // In the C++ implementation this stores `sinks` as an extra src on the flash-attn node.
    // Stub: if sinks is non-null, store it.
    if (sinks != null) {
        // find first empty src slot after existing ones
        for (i in fattn.src.indices) {
            if (fattn.src[i] == null) {
                fattn.src[i] = sinks
                break
            }
        }
    }
}

/** ggml_soft_max_add_sinks – attach attention-sink vector to softmax node. */
fun ggmlSoftMaxAddSinks(softmax: GGMLTensor, sinks: GGMLTensor?) {
    if (sinks != null) {
        for (i in softmax.src.indices) {
            if (softmax.src[i] == null) {
                softmax.src[i] = sinks
                break
            }
        }
    }
}

// port-lint: source ggml/include/ggml.h  ggml_mul_mat_aux, ggml_new_tensor_like

/**
 * ggml_mul_mat_aux — auxiliary matrix multiply (used for rotation matrices in KV cache).
 *
 * Port of `ggml_mul_mat_aux` from `ggml.h`.  When the auxiliary tensor is null
 * this is equivalent to a regular mat-mul; otherwise it applies the auxiliary
 * transformation alongside the primary operand.
 */
fun ggmlMulMatAux(ctx: GGMLContext, a: GGMLTensor, aux: GGMLTensor): GGMLTensor {
    // Auxiliary mat-mul: result = a * aux  (rotation / projection)
    return matMul(ctx, a, aux)
}

/**
 * ggml_new_tensor_like — allocate a new tensor with the same shape/type as [src].
 *
 * Port of `ggml_new_tensor_like` from `ggml.h`.
 */
fun ggmlNewTensorLike(ctx: GGMLContext, src: GGMLTensor): GGMLTensor {
    // GGML tensors are always 4D; infer effective rank from trailing ne==1 dims
    return ggmlNewTensor4d(ctx, src.type, src.ne[0], src.ne[1], src.ne[2], src.ne[3])
}
