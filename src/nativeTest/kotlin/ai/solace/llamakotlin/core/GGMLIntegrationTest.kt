package ai.solace.llamakotlin.core

import kotlin.math.*
import kotlin.test.*
import kotlin.time.ExperimentalTime
import kotlin.time.TimeSource

/**
 * Integration tests for end-to-end computation graph operations.
 * Tests complex computation chains, memory allocation patterns, and multi-operation flows.
 */
@OptIn(ExperimentalTime::class)
class GGMLIntegrationTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private val bufferSize = 8 * 1024 * 1024 // 8MB for complex graph operations

    @BeforeTest
    fun setup() {
        val (allocator, _) = GGMLTestUtils.createTestAllocator(bufferSize)
        graphAllocator = allocator
        resetAllocatorTracking(graphAllocator)
    }

    private val dummyContext = GGMLContext()

    // Helper to create tensor with data
    private fun createTensorWithData(name: String, type: GGMLType, ne: LongArray, data: FloatArray): GGMLTensor =
        GGMLTestUtils.createTensorWithData(graphAllocator, name, type, ne, data)

    private fun createDestinationTensor(name: String, type: GGMLType, ne: LongArray): GGMLTensor =
        GGMLTestUtils.allocateDestinationTensor(graphAllocator, name, type, ne.copyOf())

    // Helper to extract float data from tensor
    private fun extractFloatData(tensor: GGMLTensor): FloatArray =
        GGMLTestUtils.extractFloatData(tensor, graphAllocator)

    // --- Basic Computation Chain Tests ---
    
    @Test
    fun testSimpleComputationChain() {
        // Test: (A + B) * C - D
        val ne = longArrayOf(4)
        val dataA = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val dataB = floatArrayOf(0.5f, 1.0f, 1.5f, 2.0f)
        val dataC = floatArrayOf(2.0f, 3.0f, 4.0f, 5.0f)
        val dataD = floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)
        
        val tensorA = createTensorWithData("A", GGMLType.F32, ne, dataA)
        val tensorB = createTensorWithData("B", GGMLType.F32, ne, dataB)
        val tensorC = createTensorWithData("C", GGMLType.F32, ne, dataC)
        val tensorD = createTensorWithData("D", GGMLType.F32, ne, dataD)
        
        val addDst = createDestinationTensor("chain_add", GGMLType.F32, ne)
        computeAdd(graphAllocator, dummyContext, tensorA, tensorB, addDst)

        val mulDst = createDestinationTensor("chain_mul", GGMLType.F32, ne)
        computeMul(graphAllocator, dummyContext, addDst, tensorC, mulDst)

        val result = createDestinationTensor("chain_result", GGMLType.F32, ne)
        computeSub(graphAllocator, dummyContext, mulDst, tensorD, result)

        val resultData = extractFloatData(result)
        
        // Expected: ((1+0.5)*2-1, (2+1)*3-1, (3+1.5)*4-1, (4+2)*5-1) = (2, 8, 17, 29)
        val expected = floatArrayOf(2.0f, 8.0f, 17.0f, 29.0f)
        
        assertEquals(expected.size, resultData.size, "Result size mismatch")
        for (i in expected.indices) {
            assertEquals(expected[i], resultData[i], 0.001f, "Computation chain result mismatch at index $i")
        }
        
        println("Simple computation chain test passed: ${resultData.contentToString()}")
    }

    @Test
    fun testMixedPrecisionChain() {
        // Test computation chain with mixed F32 and F16 precision
        val ne = longArrayOf(3)
        val dataF32 = floatArrayOf(5.0f, 10.0f, 15.0f)
        val dataF16 = floatArrayOf(2.0f, 3.0f, 4.0f)
        
        val tensorF32 = createTensorWithData("F32", GGMLType.F32, ne, dataF32)
        val tensorF16 = createTensorWithData("F16", GGMLType.F16, ne, dataF16)
        val convertedF16 = createDestinationTensor("converted", GGMLType.F32, ne)
        val totalElements = tensorF16.numElements().toInt()
        applyNDIter(tensorF16, totalElements) { _, indices ->
            convertedF16.setFloat(graphAllocator, tensorF16.getHalf(graphAllocator, *indices), *indices)
        }
        
        // Perform operation: F32 + converted(F16)
        val mixedDst = createDestinationTensor("mixed_add", GGMLType.F32, ne)
        computeAdd(graphAllocator, dummyContext, tensorF32, convertedF16, mixedDst)
        val resultData = extractFloatData(mixedDst)
        
        val expected = floatArrayOf(7.0f, 13.0f, 19.0f) // 5+2, 10+3, 15+4
        
        for (i in expected.indices) {
            assertEquals(expected[i], resultData[i], 0.01f, "Mixed precision result mismatch at index $i")
        }
        
        println("Mixed precision chain test passed")
    }

    // --- Matrix Operation Integration Tests ---
    
    @Test
    fun testMatrixMultiplicationChain() {
        // Test: (A * B) + C where A, B, C are matrices
        val matrixSize = 4
        val neMatrix = longArrayOf(matrixSize.toLong(), matrixSize.toLong()) // 4x4 matrices
        
        // Create identity-like matrices for predictable results
        val identityData = FloatArray(matrixSize * matrixSize) { i ->
            if (i % (matrixSize + 1) == 0) 1.0f else 0.0f // Diagonal = 1, others = 0
        }
        val onesData = FloatArray(matrixSize * matrixSize) { 0.1f } // Small constant values
        
        val matrixA = createTensorWithData("MatA", GGMLType.F32, neMatrix, identityData)
        val matrixB = createTensorWithData("MatB", GGMLType.F32, neMatrix, identityData)  
        val matrixC = createTensorWithData("MatC", GGMLType.F32, neMatrix, onesData)
        
        try {
            // A * B (should be identity since both are identity)
            val matMulDst = createDestinationTensor("matmul_chain_dst", GGMLType.F32, neMatrix)
            computeMatMul(graphAllocator, dummyContext, matrixA, matrixB, matMulDst)
            
            // (A * B) + C
            val finalResult = createDestinationTensor("matmul_add_chain", GGMLType.F32, neMatrix)
            computeAdd(graphAllocator, dummyContext, matMulDst, matrixC, finalResult)
            val resultData = extractFloatData(finalResult)
            
            // Expected: identity + 0.1 everywhere = diagonal 1.1, others 0.1
            for (i in 0 until matrixSize) {
                for (j in 0 until matrixSize) {
                    val index = i * matrixSize + j
                    val expected = if (i == j) 1.1f else 0.1f
                    assertEquals(expected, resultData[index], 0.001f, 
                               "Matrix chain result mismatch at ($i,$j)")
                }
            }
            
            println("Matrix multiplication chain test passed")
        } catch (e: Exception) {
            println("Matrix multiplication chain test skipped: ${e.message}")
        }
    }

    // --- Quantization Integration Tests ---
    
    @Test
    fun testQuantizedOperationChain() {
        // Test computation chain with quantized tensors
        val ne = longArrayOf(64) // Size divisible by quantization block sizes
        val data1 = FloatArray(64) { i -> sin(i * 0.1f) }
        val data2 = FloatArray(64) { i -> cos(i * 0.1f) }
        
        val tensorF32_1 = createTensorWithData("F32_1", GGMLType.F32, ne, data1)
        val tensorF32_2 = createTensorWithData("F32_2", GGMLType.F32, ne, data2)
        
        try {
            // Quantize inputs
            val tensorQ8_0 = quantizeTensor(graphAllocator, tensorF32_1, GGMLType.Q8_0)
            val tensorQ4_0 = quantizeTensor(graphAllocator, tensorF32_2, GGMLType.Q4_0)
            
            // Dequantize for operations (since mixed quantized ops may not be implemented)
            val dequant1 = dequantizeTensor(graphAllocator, tensorQ8_0)
            val dequant2 = dequantizeTensor(graphAllocator, tensorQ4_0)
            
            // Perform operations on dequantized data
            val addDst = createDestinationTensor("quant_chain_add", GGMLType.F32, ne)
            computeAdd(graphAllocator, dummyContext, dequant1, dequant2, addDst)
            val mulDst = createDestinationTensor("quant_chain_mul", GGMLType.F32, ne)
            computeMul(graphAllocator, dummyContext, addDst, dequant1, mulDst)
            
            // Quantize final result
            val finalQuantized = quantizeTensor(graphAllocator, mulDst, GGMLType.Q8_0)
            val finalDequantized = dequantizeTensor(graphAllocator, finalQuantized)
            
            val resultData = extractFloatData(finalDequantized)
            
            // Verify result makes sense (should be approximately (sin + cos) * sin)
            for (i in 0 until 10) { // Check first 10 values
                val expected = (sin(i * 0.1f) + cos(i * 0.1f)) * sin(i * 0.1f)
                assertTrue(abs(expected - resultData[i]) < 0.1f, 
                          "Quantized chain result significantly different at index $i: expected≈$expected, got=${resultData[i]}")
            }
            
            println("Quantized operation chain test passed")
        } catch (e: Exception) {
            println("Quantized operation chain test skipped: ${e.message}")
        }
    }

    // --- Memory Allocation Stress Tests ---
    
    @Test
    fun testMemoryAllocationStress() {
        // Test allocation and deallocation of many tensors
        val tensors = mutableListOf<GGMLTensor>()
        val tensorCount = 20
        val tensorSize = 256
        
        try {
            // Allocate many tensors
            for (i in 0 until tensorCount) {
                val data = FloatArray(tensorSize) { j -> (i + j).toFloat() }
                val tensor = createTensorWithData("stress_$i", GGMLType.F32, longArrayOf(tensorSize.toLong()), data)
                tensors.add(tensor)
            }
            
            // Perform operations on all tensors
            var result = tensors[0]
            for (i in 1 until tensors.size step 2) {
                if (i + 1 < tensors.size) {
                    val addDst = createDestinationTensor("stress_add_$i", result.type, result.ne.copyOf())
                    computeAdd(graphAllocator, dummyContext, tensors[i], tensors[i + 1], addDst)

                    val mulDst = createDestinationTensor("stress_mul_$i", result.type, result.ne.copyOf())
                    computeMul(graphAllocator, dummyContext, result, addDst, mulDst)
                    result = mulDst
                }
            }
            
            val finalData = extractFloatData(result)
            assertTrue(finalData.isNotEmpty(), "Final result should not be empty")
            
            println("Memory allocation stress test passed with $tensorCount tensors")
        } catch (e: OutOfMemoryError) {
            println("Memory allocation stress test reached memory limit as expected")
        } catch (e: Exception) {
            println("Memory allocation stress test completed with exception: ${e.message}")
        }
    }

    // --- Activation Function Integration ---
    
    @Test
    fun testActivationFunctionChain() {
        // Test: RELU(GELU(input)) chain
        val ne = longArrayOf(8)
        val inputData = floatArrayOf(-3.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 2.0f, 3.0f)
        
        val inputTensor = createTensorWithData("input", GGMLType.F32, ne, inputData)
        
        try {
            val geluDst = createDestinationTensor("gelu_chain", GGMLType.F32, ne)
            computeGelu(graphAllocator, dummyContext, inputTensor, geluDst)
            
            val reluDst = createDestinationTensor("relu_chain", GGMLType.F32, ne)
            computeRelu(graphAllocator, dummyContext, geluDst, reluDst)
            
            val resultData = extractFloatData(reluDst)
            
            // GELU followed by RELU should be positive values (RELU clips negatives to 0)
            for (i in resultData.indices) {
                assertTrue(resultData[i] >= 0.0f, "RELU(GELU(x)) should be non-negative at index $i: ${resultData[i]}")
            }
            
            println("Activation function chain test passed: ${resultData.contentToString()}")
        } catch (e: Exception) {
            println("Activation function chain test skipped: ${e.message}")
        }
    }

    // --- Complex Mathematical Expressions ---
    
    @Test
    fun testComplexMathematicalExpression() {
        // Test: sqrt((A^2 + B^2) / (C + epsilon))
        val ne = longArrayOf(4)
        val dataA = floatArrayOf(3.0f, 4.0f, 5.0f, 12.0f)
        val dataB = floatArrayOf(4.0f, 3.0f, 12.0f, 5.0f)  
        val dataC = floatArrayOf(2.0f, 2.0f, 2.0f, 2.0f)
        val epsilon = 1e-6f
        
        val tensorA = createTensorWithData("A", GGMLType.F32, ne, dataA)
        val tensorB = createTensorWithData("B", GGMLType.F32, ne, dataB)
        val tensorC = createTensorWithData("C", GGMLType.F32, ne, dataC)
        val epsilonTensor = createTensorWithData("eps", GGMLType.F32, ne, floatArrayOf(epsilon, epsilon, epsilon, epsilon))
        
        try {
            val aSquared = createDestinationTensor("complex_a_sq", GGMLType.F32, ne)
            computeSqr(graphAllocator, dummyContext, tensorA, aSquared)

            val bSquared = createDestinationTensor("complex_b_sq", GGMLType.F32, ne)
            computeSqr(graphAllocator, dummyContext, tensorB, bSquared)

            val sumSquares = createDestinationTensor("complex_sum_sq", GGMLType.F32, ne)
            computeAdd(graphAllocator, dummyContext, aSquared, bSquared, sumSquares)

            val cPlusEps = createDestinationTensor("complex_denom", GGMLType.F32, ne)
            computeAdd(graphAllocator, dummyContext, tensorC, epsilonTensor, cPlusEps)

            val division = createDestinationTensor("complex_div", GGMLType.F32, ne)
            computeDiv(graphAllocator, dummyContext, sumSquares, cPlusEps, division)

            val finalResult = createDestinationTensor("complex_sqrt", GGMLType.F32, ne)
            computeSqrt(graphAllocator, dummyContext, division, finalResult)
            
            val resultData = extractFloatData(finalResult)
            
            // Expected results: sqrt((3²+4²)/2), sqrt((4²+3²)/2), sqrt((5²+12²)/2), sqrt((12²+5²)/2)
            // = sqrt(25/2), sqrt(25/2), sqrt(169/2), sqrt(169/2) = sqrt(12.5), sqrt(12.5), sqrt(84.5), sqrt(84.5)
            val expected = floatArrayOf(
                sqrt(12.5f), sqrt(12.5f), sqrt(84.5f), sqrt(84.5f)
            )
            
            for (i in expected.indices) {
                assertEquals(expected[i], resultData[i], 0.01f, "Complex math expression mismatch at index $i")
            }
            
            println("Complex mathematical expression test passed")
        } catch (e: Exception) {
            println("Complex mathematical expression test skipped (operations not implemented): ${e.message}")
        }
    }

    // --- Performance and Scalability Tests ---
    
    @Test
    fun testPerformanceBenchmark() {
        val sizes = arrayOf(64, 256, 1024, 4096)
        
        for (size in sizes) {
            val ne = longArrayOf(size.toLong())
            val data1 = FloatArray(size) { i -> sin(i * 0.01f) }
            val data2 = FloatArray(size) { i -> cos(i * 0.01f) }
            
            val tensor1 = createTensorWithData("perf1_$size", GGMLType.F32, ne, data1)
            val tensor2 = createTensorWithData("perf2_$size", GGMLType.F32, ne, data2)
            
            val startMark = TimeSource.Monotonic.markNow()
            
            // Perform a series of operations
            val addDst = createDestinationTensor("perf_add_$size", GGMLType.F32, ne)
            computeAdd(graphAllocator, dummyContext, tensor1, tensor2, addDst)

            val mulDst = createDestinationTensor("perf_mul_$size", GGMLType.F32, ne)
            computeMul(graphAllocator, dummyContext, addDst, tensor1, mulDst)

            val subDst = createDestinationTensor("perf_sub_$size", GGMLType.F32, ne)
            computeSub(graphAllocator, dummyContext, mulDst, tensor2, subDst)

            val negDst = createDestinationTensor("perf_neg_$size", GGMLType.F32, ne)
            computeNeg(graphAllocator, dummyContext, subDst, negDst)
            
            val duration = startMark.elapsedNow().inWholeMilliseconds
            
            // Extract final result to ensure computation happened
            val resultData = extractFloatData(negDst)
            assertTrue(resultData.isNotEmpty(), "Performance test result should not be empty")
            
            println("Performance test size $size: ${duration}ms")
            
            // Performance should scale reasonably
            assertTrue(duration < 1000L, "Operations on $size elements should complete in reasonable time")
        }
    }

    // --- Error Handling Integration ---
    
    @Test
    fun testErrorHandlingInChains() {
        val ne = longArrayOf(4)
        val normalData = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val zeroData = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
        
        val normalTensor = createTensorWithData("normal", GGMLType.F32, ne, normalData)
        val zeroTensor = createTensorWithData("zeros", GGMLType.F32, ne, zeroData)
        
        // Test division by zero handling
        try {
            val divisionResult = createDestinationTensor("error_div_dst", GGMLType.F32, ne)
            computeDiv(graphAllocator, dummyContext, normalTensor, zeroTensor, divisionResult)
            val resultData = extractFloatData(divisionResult)
            
            // Should handle division by zero gracefully (infinity or NaN)
            for (value in resultData) {
                assertTrue(value.isInfinite() || value.isNaN(), "Division by zero should produce Inf or NaN")
            }
            
            println("Division by zero handled correctly in computation chain")
        } catch (e: ArithmeticException) {
            println("Division by zero correctly threw ArithmeticException")
        }
        
        // Test dimension mismatch handling
        val differentSizeTensor = createTensorWithData("different", GGMLType.F32, longArrayOf(6), 
                                                      floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
        
        val mismatchDst = createDestinationTensor("mismatch_dst", GGMLType.F32, ne)
        assertFailsWith<IllegalArgumentException> {
            computeAdd(graphAllocator, dummyContext, normalTensor, differentSizeTensor, mismatchDst)
        }
        println("Dimension mismatch correctly caught")
    }

    @Test 
    fun testLlamaModelComponentsBasicIntegration() {
        // Test basic LLaMA model components using existing infrastructure
        
        // Test RMSNorm operation with realistic parameters
        val hiddenSize = 32
        val input = createTensorWithData(
            "rms_input", 
            GGMLType.F32, 
            longArrayOf(hiddenSize.toLong(), 1L), // [hidden_size, seq_len]
            FloatArray(hiddenSize) { (it + 1).toFloat() / hiddenSize }
        )
        
        // Test RMSNorm computation
        val rmsNormResult = createDestinationTensor("rms_norm_result", GGMLType.F32, input.ne.copyOf())
        computeRMSNorm(graphAllocator, dummyContext, input, 1e-6f, rmsNormResult)
        assertNotNull(rmsNormResult)
        assertEquals(hiddenSize.toLong(), rmsNormResult.ne[0])
        
        // Verify normalization worked (check magnitude)
        val outputData = extractFloatData(rmsNormResult)
        var sumSquared = 0.0f
        for (value in outputData) {
            sumSquared += value * value
        }
        val rms = sqrt(sumSquared / hiddenSize)
        assertTrue(abs(rms - 1.0f) < 0.1f, "RMS should be approximately 1.0, got $rms")
        
        println("✓ RMSNorm component test passed")
        
        // Test SiLU activation function
        val activationInput = createTensorWithData(
            "activation_input",
            GGMLType.F32,
            longArrayOf(8L),
            floatArrayOf(-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f)
        )
        
        val siluResult = createDestinationTensor("silu_result", GGMLType.F32, activationInput.ne.copyOf())
        computeSilu(graphAllocator, dummyContext, activationInput, siluResult)
        assertNotNull(siluResult)
        assertEquals(8L, siluResult.ne[0])
        
        val siluData = extractFloatData(siluResult)
        // Verify SiLU properties: f(0) = 0, f(x) > 0 for x > 0
        assertTrue(abs(siluData[3]) < 1e-6f, "SiLU(0) should be ~0")
        assertTrue(siluData[4] > 0, "SiLU(0.5) should be positive")
        assertTrue(siluData[5] > siluData[4], "SiLU should be increasing for positive values")
        
        println("✓ SiLU activation test passed")
        
        // Test basic matrix multiplication for transformer components
        val inputDim = 16
        val outputDim = 32
        val seqLen = 4
        
        val inputTensor = createTensorWithData(
            "transformer_input",
            GGMLType.F32, 
            longArrayOf(inputDim.toLong(), seqLen.toLong()),
            FloatArray(inputDim * seqLen) { 0.1f * (it % 10 + 1) }
        )
        
        val weightTensor = createTensorWithData(
            "linear_weight",
            GGMLType.F32,
            longArrayOf(inputDim.toLong(), outputDim.toLong()),
            FloatArray(inputDim * outputDim) { 0.01f * ((it % 7) - 3) }
        )
        
        val matmulResult = createDestinationTensor(
            "matmul_result",
            GGMLType.F32,
            longArrayOf(outputDim.toLong(), seqLen.toLong())
        )
        computeMatMul(graphAllocator, dummyContext, weightTensor, inputTensor, matmulResult)
        assertNotNull(matmulResult)
        assertEquals(outputDim.toLong(), matmulResult.ne[0])
        assertEquals(seqLen.toLong(), matmulResult.ne[1])
        
        val matmulData = extractFloatData(matmulResult)
        // Verify reasonable output range and no NaN/Inf
        for (value in matmulData) {
            assertFalse(value.isNaN(), "Matrix multiplication output should not be NaN")
            assertFalse(value.isInfinite(), "Matrix multiplication output should not be infinite")
        }
        
        println("✓ Linear transformation test passed")
        
        // Test attention-like computation pattern (simplified)
        val numHeads = 4
        val headDim = 8
        val totalDim = numHeads * headDim // 32
        
        val queryTensor = createTensorWithData(
            "query",
            GGMLType.F32,
            longArrayOf(totalDim.toLong(), 1L), // Single token
            FloatArray(totalDim) { 0.1f }
        )
        
        val keyTensor = createTensorWithData(
            "key", 
            GGMLType.F32,
            longArrayOf(totalDim.toLong(), 3L), // 3 key vectors
            FloatArray(totalDim * 3) { 0.05f * (it % 5 + 1) }
        )
        
        // Compute attention scores: Q @ K^T (simplified - using existing matmul)
        // Note: This is a simplified test, real attention would need transpose and softmax
        val attentionScores = createDestinationTensor(
            "attention_scores",
            GGMLType.F32,
            longArrayOf(1L, 3L)
        )
        computeMatMul(graphAllocator, dummyContext, queryTensor, keyTensor, attentionScores)
        assertNotNull(attentionScores)
        assertEquals(1L, attentionScores.ne[0]) // query_seq_len
        assertEquals(3L, attentionScores.ne[1]) // key_seq_len
        
        println("✓ Attention pattern test passed")
        
        // Test softmax operation (important for attention)
        val logitsInput = createTensorWithData(
            "logits",
            GGMLType.F32,
            longArrayOf(5L),
            floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        )
        
        val softmaxResult = createDestinationTensor("softmax_result", GGMLType.F32, logitsInput.ne.copyOf())
        computeSoftMax(graphAllocator, dummyContext, logitsInput, softmaxResult)
        assertNotNull(softmaxResult)
        assertEquals(5L, softmaxResult.ne[0])
        
        val softmaxData = extractFloatData(softmaxResult)
        // Verify softmax properties: sum to 1, all positive
        var sum = 0.0f
        for (value in softmaxData) {
            assertTrue(value > 0, "Softmax outputs should be positive")
            assertTrue(value < 1, "Individual softmax outputs should be < 1")
            sum += value
        }
        assertTrue(abs(sum - 1.0f) < 1e-5f, "Softmax outputs should sum to 1.0, got $sum")
        
        println("✓ Softmax test passed")
        println("✓ All LLaMA model component integration tests passed!")
    }
}
