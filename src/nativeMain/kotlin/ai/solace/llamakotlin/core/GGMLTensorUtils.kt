package ai.solace.llamakotlin.core

/**
 * Centralized tensor utility functions to avoid code duplication.
 * This file consolidates common helper functions used across the GGML implementation.
 */
object GGMLTensorUtils {
    
    /**
     * Calculate total size (number of elements) of a tensor from its dimensions.
     * Used throughout the codebase for memory allocation and validation.
     */
    fun calculateTotalSize(ne: LongArray): Long {
        var totalSize = 1L
        for (i in 0 until GGML_MAX_DIMS) {
            if (i < ne.size && ne[i] > 0) {
                totalSize *= ne[i]
            } else if (i >= ne.size) {
                break // Only multiply valid dimensions
            }
        }
        return totalSize
    }
    
    /**
     * Legacy compatibility function that returns Int for backward compatibility.
     * @deprecated Use calculateTotalSize() that returns Long for better precision
     */
    @Deprecated("Use calculateTotalSize() for better precision", ReplaceWith("calculateTotalSize(ne).toInt()"))
    fun calculateTotalSizeInt(ne: LongArray): Int {
        return calculateTotalSize(ne).toInt()
    }
    
    /**
     * Calculate contiguous tensor strides for memory layout.
     * Consolidates the stride calculation logic from GGMLOps and GGMLTestUtils.
     */
    fun GGMLTensorUtils.calculateContiguousStrides(ne: LongArray, type: GGMLType, maxDims: Int = GGML_MAX_DIMS): ULongArray {
        val nb = ULongArray(maxDims) { 0uL }
        
        if (type.byteSize == 0uL) {
            // Handle types with zero byte size (e.g., COUNT or uninitialized quantized types)
            if (type != GGMLType.COUNT && !type.name.startsWith("Q", ignoreCase = true) && !type.name.startsWith("q", ignoreCase = true)) {
                println("Warning: GGMLType ${type.name} has byteSize 0. Strides will be all zeros.")
            }
            return nb // Return zeroed strides
        }
        
        nb[0] = type.byteSize
        if (maxDims > 1) {
            for (d in 1 until maxDims) {
                val prevDimSize = ne.getOrElse(d - 1) { 1L }
                nb[d] = nb[d - 1] * (if (prevDimSize > 0L) prevDimSize.toULong() else 1uL)
            }
        }
        return nb
    }
    
    /**
     * Calculate tensor byte size for memory allocation.
     * Handles both regular types and quantized types with block structures.
     */
    fun calculateTensorByteSize(type: GGMLType, ne: LongArray): ULong {
        if (type.byteSize == 0uL && type != GGMLType.COUNT && !type.name.startsWith("Q")) {
            return 0uL
        }
        
        var elements = 1UL
        var validDimFound = false
        
        for (i in ne.indices) {
            if (ne[i] > 0L) {
                elements *= ne[i].toULong()
                validDimFound = true
            } else if (ne[i] == 0L && elements != 0UL && validDimFound) {
                return 0UL
            }
        }
        
        if (!validDimFound && ne.isNotEmpty() && ne.all { it <= 1L }) {
            elements = 1UL // Scalar or effectively scalar
        } else if (!validDimFound && ne.isEmpty()) {
            elements = 1UL // Treat as scalar if ne is completely empty
        }

        // Handle quantized types with block structures
        return when (type) {
            GGMLType.Q8_0 -> {
                if (elements > 0uL) {
                    if (elements.toLong() % QK8_0 != 0L) {
                        println("Warning: Total elements $elements for Q8_0 is not divisible by block size $QK8_0.")
                    }
                    (elements.toLong() / QK8_0).toULong() * type.byteSize
                } else 0uL
            }
            GGMLType.Q4_0 -> {
                if (elements > 0uL) {
                    if (elements.toLong() % QK4_0 != 0L) {
                        println("Warning: Total elements $elements for Q4_0 is not divisible by block size $QK4_0.")
                    }
                    (elements.toLong() / QK4_0).toULong() * type.byteSize
                } else 0uL
            }
            GGMLType.Q4_1 -> {
                if (elements > 0uL) {
                    if (elements.toLong() % QK4_1 != 0L) {
                        println("Warning: Total elements $elements for Q4_1 is not divisible by block size $QK4_1.")
                    }
                    (elements.toLong() / QK4_1).toULong() * type.byteSize
                } else 0uL
            }
            else -> elements * type.byteSize
        }
    }
    
    /**
     * Initialize a tensor's dimensions and strides based on the provided shape.
     * Consolidates the repetitive tensor setup pattern.
     */
    fun initializeTensorDimensions(tensor: GGMLTensor, shape: LongArray) {
        // Ensure ne array is properly sized and initialized
        tensor.ne = LongArray(GGML_MAX_DIMS) { 1L }
        shape.forEachIndexed { index, dimSize ->
            if (index < GGML_MAX_DIMS) {
                tensor.ne[index] = dimSize
            }
        }
        
        // Calculate and set strides
        tensor.nb = GGMLTensorUtils.calculateContiguousStrides(tensor.ne, tensor.type, GGML_MAX_DIMS)
    }
    
    /**
     * Create a tensor with proper dimensions and memory layout.
     * Consolidates the tensor creation pattern used throughout the codebase.
     */
    fun createTensor(type: GGMLType, name: String = "", vararg shape: Long): GGMLTensor {
        val tensor = GGMLTensor(type = type, name = name)
        initializeTensorDimensions(tensor, shape)
        return tensor
    }
}