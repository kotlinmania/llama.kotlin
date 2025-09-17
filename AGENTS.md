# llama.kotlin Project Instructions

This file provides guidance for future agents working on the Kotlin port of `llama.cpp`.
It summarizes the current project state and lists recommended next steps. Before starting new work, read `KOTLIN_PORT_CHECKLIST.md` in the repository root for a detailed roadmap.

## Project Overview
- **Goal**: Create a Kotlin/Native implementation of llama.cpp, focusing on CPU and Apple Metal backends
- **Current Status**: Phase 3-4 (Advanced Core Implementation & Backend Development) with substantial infrastructure complete
- The repository is a work‐in‐progress port of `llama.cpp` to Kotlin/Native
- Kotlin sources live under `src/nativeMain/kotlin/ai/solace/llamakotlin`
- The original C/C++ sources remain under `src` while porting progresses
- Design notes and porting progress are documented in:
  - `KOTLIN_PORT_CHECKLIST.md` - Detailed development roadmap with current progress
  - `KOTLIN_PORT_STATUS.md` - Overall project status and completion overview  
  - `GGML_COMPUTE_OPS_DESIGN.md` - Technical design for computation operations
  - `TENSOR_OPERATIONS_DESIGN.md` - Design patterns for tensor operations
  - `CPP_CORE_ANALYSIS.md` - Analysis of original C++ codebase
  - `MATMUL_OPTIMIZATION_SUMMARY.md` - Matrix multiplication optimization achievements
  - `GGUF_IMPLEMENTATION.md` - GGUF file format support implementation
  - `GGML_TESTING_SUMMARY.md` - Comprehensive testing infrastructure documentation

## Current Implementation Status

### ✅ Completed Core Features
- **Advanced Memory Management**: Sophisticated tensor allocation with `GGMLGraphAllocator` and `GGMLDynTensorAllocator`
  - Primary ByteArray buffer with dynamic allocation within reserved space
  - Inplace tensor allocation and memory reuse logic for optimization
  - Tensor usage tracking and automatic memory freeing
  - Graph-level memory planning with comprehensive allocation strategies
- **Comprehensive Tensor Data Access**: Multi-dimensional tensor operations with stride support
  - F32, F16, I32, I16 data accessors with stride information
  - Efficient ByteArray-based data storage and retrieval
  - Multi-dimensional indexing with proper stride calculations
- **Advanced Quantization Ecosystem**: Comprehensive quantization support implemented
  - Q8_0: F16 scale + 32xI8 weights (34 bytes per block)
  - Q4_0: F16 scale + 32x4-bit packed weights (18 bytes per block)  
  - Q4_1: 2x F16 scale/min + 32x4-bit packed weights (20 bytes per block)
  - BitNet 1.58: Ternary quantization with F16 scale + base-3 packing (10 bytes per block)
  - Optimized dot product routines for all quantized operations
  - Direct quantized-to-quantized operations (Q×Q) avoiding expensive dequantization
- **Matrix Multiplication Optimization Framework**: Complete optimization coverage
  - Symmetric F32×Q_type optimizations providing 2-3x speedups
  - Direct Q_type×Q_type operations providing 3-5x speedups  
  - Mixed quantized operations (Q8_0×Q4_0, etc.)
  - Performance validation with comprehensive benchmarking suite
- **Core Tensor Operations**: Element-wise and matrix operations with multi-type support
  - ADD, MUL, SUB, DIV, NEG, SQR, SQRT, MatMul for F32/F16 and quantized types
  - Activation functions: RELU, GELU, SILU, RMSNorm
  - Broadcasting and dimension validation with proper error handling
- **GGUF File Format Support**: Complete model loading pipeline
  - Binary GGUF file parsing with metadata extraction  
  - Tensor loading and integration with existing tensor system
  - Model validation through forward pass testing
  - Support for all standard GGUF data types and structures
- **Automatic Differentiation**: Comprehensive backward pass implementation
  - ADD, SUB, MUL, NEG, DIV, SQR, SQRT operations
  - RELU, GELU activation functions
  - MUL_MAT (matrix multiplication)
  - SUM, MEAN, REPEAT operations
- **Compute Operations Architecture**: Major refactor to destination-based operations
  - All compute functions now write directly into pre-allocated destination tensors
  - Function signatures changed from `computeAdd(...): GGMLTensor` to `computeAdd(..., dst: GGMLTensor)`
  - Eliminated memory allocation within compute operations for improved efficiency
  - Aligned with GGML architecture patterns for memory reuse and graph optimization
- **Comprehensive Testing Infrastructure**: Multi-tier validation and benchmarking
  - Unit tests with complete coverage for all core operations  
  - Quantization accuracy testing with rigorous MSE, RMSE, MAD validation
  - Performance benchmarking suite with throughput and latency metrics
  - Integration tests for end-to-end workflows and mathematical expressions
  - Reference validation framework with analytical validation
  - Memory stress testing and allocation efficiency validation

### 🔄 In Progress  
- **Additional Quantization**: K-Quant types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
- **CPU Backend Formalization**: Structured CPU backend with multi-threading support
- **Model Architecture Implementation**: LLaMA model structures and inference pipeline
- **Metal Backend Foundation**: Basic Metal context and shader infrastructure

### 📋 Testing Infrastructure
- Comprehensive unit tests for core operations under `src/nativeTest/kotlin`
- Quantization accuracy tests with MSE and MAD validation
- Memory allocator tests for graph-level memory planning
- **Destination-based compute operations testing** with comprehensive validation
  - `GGMLComputeOpsDestinationTest.kt`: Tests for new in-place computation interface
  - Dimension and type mismatch validation
  - Direct integration testing with graph allocator memory management
- Tensor data accessor tests for all supported types

## Coding Guidelines
- **Kotlin Style**: Use idiomatic Kotlin with descriptive names and comprehensive KDoc comments
- **Modular Design**: Separate tensor creation logic from compute kernels (see `GGMLOps.kt` vs `GGMLComputeOps.kt`)
- **Immutability**: Prefer immutable data structures where practical
- **Documentation**: Document placeholders or incomplete implementations with `TODO` comments
- **Memory Efficiency**: Use ByteArray-based storage with accessor methods rather than individual arrays
- **Type Safety**: Leverage Kotlin's type system for compile-time safety in tensor operations
- **Performance**: Consider SIMD and multi-threading opportunities in CPU-intensive operations
- **DRY Principle**: Use centralized utilities in `GGMLUtilities.kt` and `GGMLTensorUtils.kt` to avoid code duplication
- **Consistent Formatting**: Use `GGMLUtilities.formatDouble()` and related functions for consistent display formatting
- **Centralized Extensions**: Use `ByteArrayExtensions` object for all ByteArray manipulation to ensure consistency

## Key Architecture Patterns
- **Memory Management**: Use `GGMLGraphAllocator` for graph-level memory planning with inplace optimization
- **Data Access**: Implement typed accessors (`getF32`, `setF32`, etc.) for ByteArray-backed tensors
- **Quantization**: Follow block-based quantization patterns with optimized dot product routines
- **Computation Separation**: Keep operation setup (graph building) separate from computation (execution)
- **Utility Consolidation**: Leverage centralized utilities for common operations:
  - `GGMLUtilities.kt`: Formatting, display helpers, and ByteArray extensions
  - `GGMLTensorUtils.kt`: Tensor dimension calculations and validation
  - `DemoTextUtilities`: Reusable components for demo and test output

## Build and Test
- **Build System**: The project uses Gradle with Kotlin Multiplatform. Build with `./gradlew build`
- **Network Dependencies**: Network access may be required to download dependencies; configure as needed
- **SSL Issues**: If Gradle fails with a `PKIX path building failed` SSL error, install Java and Gradle via SDKMAN:
  ```bash
  curl -s "https://get.sdkman.io" | bash
  source "$HOME/.sdkman/bin/sdkman-init.sh"
  sdk install java 17.0.9-tem
  sdk install gradle 8.13
  ```
  Alternatively set `GRADLE_OPTS="-Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts"` to use the system certificate store
- **Target Platforms**: Currently targets macOS (x64 and arm64) - configured in `build.gradle.kts`
- **Test Structure**: 
  - Unit tests: `src/nativeTest/kotlin/ai/solace/llamakotlin/core/`
  - Test categories: Core operations, memory allocation, quantization accuracy, data accessors
  - Run tests: `./gradlew allTests` (when build system is accessible)
- **C++ Legacy**: C++ tests under `tests/` are not required for the Kotlin port

### Current Test Coverage
- **GGMLComputeOpsTest.kt**: Core tensor operations (ADD, MUL, MatMul, activations)
- **GGMLComputeOpsDestinationTest.kt**: New destination-based compute operations interface
- **GGMLQuantizationAccuracyTest.kt**: Q8_0, Q4_0, Q4_1 quantization accuracy validation
- **GGMLAllocTest.kt**: Memory allocation and graph planning functionality  
- **GGMLTypesTest.kt**: Tensor data accessors and type handling

## Implementation Priorities (Based on Current Checklist Status)

### 🎯 Immediate Next Steps
1. **LLaMA Model Architecture Implementation**
   - Begin core model structure implementation (attention, feed-forward networks)
   - Integrate with existing tensor operations and memory management systems
   - Leverage GGUF loading capabilities for real model file support

2. **CPU Backend Formalization**
   - Integrate current `GGMLComputeOps.kt` logic into formal CPU backend structure
   - Implement multi-threading for graph computation using Kotlin coroutines
   - Explore SIMD optimizations within Kotlin/Native constraints

3. **Additional Quantization Support** 
   - Implement K-Quant types (Q4_K, Q2_K) with block structures and optimized operations
   - Extend matrix multiplication optimization framework to cover new quantization formats
   - Maintain comprehensive testing coverage for accuracy and performance

4. **Metal Backend Foundation**
   - Basic Metal context setup and shader compilation infrastructure
   - Simple Metal compute shader for proof-of-concept operations
   - Integration planning with existing tensor operations

### 🔄 Mid-term Goals
1. **Advanced Model Support** (Phase 5)
   - Complete LLaMA attention mechanism and transformer architecture
   - Implement sampling methods (top-k, top-p, temperature scaling)
   - Add grammar-constrained generation capabilities

2. **Enhanced Backend Development**
   - Metal backend implementation for Apple Silicon optimization
   - Advanced CPU optimizations and multi-threading improvements
   - Performance benchmarking against reference implementations

3. **Production Readiness** (Phase 7)
   - Command-line applications (llama-cli equivalent)
   - Server application and API implementations
   - Integration examples and usage documentation

### 📚 Implementation Resources
- **Memory Management Patterns**: See `GGMLAlloc.kt` for tensor and graph allocation strategies  
- **Quantization Implementation**: Reference `GGMLComputeOps.kt` for Q8_0/Q4_0/Q4_1 patterns
- **Data Access Patterns**: See `GGMLTensor` accessor methods for ByteArray-based storage
- **Testing Patterns**: Follow existing test structure in `src/nativeTest/kotlin/` 
- **Utility Patterns**: Use `GGMLUtilities.kt` for formatting, display, and ByteArray operations
- **DRY Implementation**: Reference centralized utilities to avoid code duplication:
  - `GGMLTensorUtils.calculateTotalSize()` for tensor size calculations
  - `ByteArrayExtensions` for consistent little-endian operations
  - `DemoTextUtilities` for reusable demo components 

## Technical Implementation Guidance

### Memory Management Best Practices
```kotlin
// Use GGMLGraphAllocator for efficient memory planning
val allocator = GGMLGraphAllocator()
allocator.reserve(graph, bufferSize)
val tensor = allocator.allocateTensor(type, dimensions) // Automatically uses inplace when possible
```

### Quantization Implementation Pattern  
```kotlin
// Follow block-based quantization pattern
// 1. Define block structure (scale + packed weights)
// 2. Implement block data accessors (getScale, getWeight)
// 3. Add quantize/dequantize functions
// 4. Create optimized dot product routine
// 5. Integrate with MatMul operation
```

### Adding New Tensor Operations
```kotlin
// 1. Add operation enum to GGMLOp
// 2. Implement computation in GGMLComputeOps.kt using destination-based approach:
fun computeNewOp(graphAllocator: GGMLGraphAllocator, context: GGMLContext, 
                 a: GGMLTensor, b: GGMLTensor, dst: GGMLTensor) {
    // Write directly to dst tensor using allocator-managed memory
    dst.setFloat(graphAllocator, result, *indices)
}
// 3. Add high-level interface in GGMLOps.kt  
// 4. Create unit tests in GGMLComputeOpsDestinationTest.kt
// 5. Add backward pass for automatic differentiation
// 6. Use GGMLUtilities for consistent formatting and display
// 7. Leverage GGMLTensorUtils for dimension calculations and validation
```

## Project Scope and Backend Support
- **Supported Backends**: CPU (primary focus) and Apple Metal for macOS/iOS
- **Archived Backends**: CUDA, hipBLAS, Vulkan, SYCL, MUSA, and CANN backends moved to `archive/` 
- **Platform Targets**: macOS (x64 and arm64) with future support for other Kotlin/Native targets
- **Memory Model**: Adapted for Kotlin/Native's memory management (different from C++ original)

## Advanced Research Goals (Future Work)
The project includes research into hybrid architectures beyond the core llama.cpp port:
- **Liquid Neural Networks Integration**: Actor-based computation with adaptive time constants (see `hybrid-llama-lnn-design.md`)
- **Quantum-Classical Hybrid Framework**: Meta-Cognitive Temporal Architecture (MCTA) research (see `quantum-classical-hybrid-mcta-design.md`)
- **Memory Cube Architecture**: Advanced caching and inference optimization strategies
- **Actor-Coroutine Integration**: Leveraging Kotlin's concurrency model for neural computation

*Note: These are research directions for future exploration after completing the core llama.cpp port.*

## Key Files and Modules
- **Core Types**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLTypes.kt`
- **Memory Management**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLAlloc.kt`
- **Tensor Operations**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLOps.kt`
- **Computation Logic**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLComputeOps.kt`
- **Graph Execution**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLGraph.kt`
- **Test Suite**: `src/nativeTest/kotlin/ai/solace/llamakotlin/core/`

## Challenges and Considerations
- **Memory Management**: Kotlin/Native memory model requires different patterns than C++
- **Performance**: Maintaining comparable performance to C++ original while leveraging Kotlin strengths
- **Interoperability**: Potential C interop for performance-critical sections (evaluate on case-by-case basis)
- **SIMD Limitations**: Kotlin/Native has limited SIMD support compared to C++ (explore alternatives)
- **Metal Integration**: Implementing Metal backend for Apple Silicon optimization
- **Quantization Precision**: Ensuring accuracy across all quantization formats

Follow this guide when extending the Kotlin port. Keep commits focused and include relevant tests whenever possible. For detailed implementation status, always reference `KOTLIN_PORT_CHECKLIST.md` for the most current progress information.
