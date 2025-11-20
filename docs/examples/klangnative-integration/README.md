# KLangNative Integration Examples

Reference implementations from [ember-ml-kotlin](https://github.com/KotlinMania/ember-ml-kotlin) demonstrating best practices for using KLangNative's heap, pointers, and C types for high-performance tensor storage.

## Files

### commonMain/KlangHeapTensorStorage.kt
Multiplatform tensor storage using KLangNative's GlobalHeap with aligned allocation. Demonstrates:
- Aligned memory allocation via `KAligned.alignedCalloc()`
- Element-by-element operations with `CFloat32` for bit-exact IEEE-754 semantics
- Bulk operations using `GlobalHeap.copyFromIntArray()` / `copyToIntArray()`
- Zero-copy packed bit operations for pre-converted data

### nativeMain/NativeHeapTensorStorage.kt
Kotlin/Native fast-path using `nativeHeap` and CInterop for zero-copy, word-sized access. Shows how to:
- Use `kotlinx.cinterop.nativeHeap.allocArray()` for native allocation
- Reinterpret byte pointers as typed pointers (`IntVar`, etc.)
- Avoid ByteArray bounds checks for maximum performance

### nativeMain/HeapBench.kt
Comprehensive benchmarking suite comparing:
- KlangHeapTensorStorage (scalar)
- KlangHeapTensorStorage (bulk)
- Packed operations
- In-place heap math (no array copies)
- NativeHeapTensorStorage (K/N CInterop)
- Plain Kotlin arrays (baseline)

Demonstrates performance characteristics and trade-offs of each approach.

### commonTest/KlangHeapTensorStorageTest.kt
Unit tests for KlangHeapTensorStorage covering:
- Round-trip correctness (element and bulk)
- NaN and Infinity handling
- Alignment verification
- Performance timing comparisons

### commonTest/ScalarTest.kt
Tests for scalar C types (`C_UInt128`, `C_Int128`, etc.)

## Usage

These files serve as **reference implementations** for migrating llama.kotlin's tensor storage to use KLangNative's GlobalHeap and CFloat32. Key patterns to adopt:

1. **Use CFloat32 for quantization math** - Eliminates platform-dependent float drift
2. **Prefer bulk operations** - `copyFromIntArray/copyToIntArray` are much faster than element-by-element
3. **Consider Native fast-paths** - For critical loops, K/N CInterop can provide significant speedup
4. **Use aligned allocation** - SIMD operations require proper alignment (typically 32 bytes)

## Migration Path

See `docs/kdocs/klangnative-heap-migration-plan.md` for the full strategy to integrate these patterns into llama.kotlin.
