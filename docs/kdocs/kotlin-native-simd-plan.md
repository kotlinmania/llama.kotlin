# Kotlin/Native SIMD Port Plan

Updated: 2025‑09‑21

This plan now coexists with the KLang numeric core (pure‑Kotlin soft‑float + 16‑bit limb engine). KLang guarantees bit‑exact IEEE‑754 semantics; SIMD is an opt‑in accelerator layered on top. All correctness must hold on the scalar path first, then SIMD paths are enabled behind the same API.

## Status Summary
- KLang added: `CFloat32` division matches compiler‑rt; HPC16x4/x8 limb types landed to support Float64 later.
- SIMD scaffolding present: `core/simd/GGMLSimd.kt` hosts initial F32 helpers and scalar fallbacks.
- Kotlin/Native exposes LLVM vector types via `kotlinx.cinterop.Vector128` (and related helpers), giving us direct access to SIMD loads, stores, and arithmetic in pure Kotlin code.[^1]
- Higher-level conveniences (`vectorOf`, typed load/store helpers) exist for building and manipulating these vectors, but reductions and shuffles still require manual composition.[^2]
- Coroutine dispatchers can already cap parallelism per graph execution using `Dispatchers.Default.limitedParallelism(...)`, and we can opt into a dedicated thread-pool dispatcher when we need to reserve CPU cores for SIMD-heavy kernels.[^3]
- The upstream C implementation relies on architecture-specific intrinsics inside ggml (e.g., `_mm512_dpbf16_ps`, `_mm256_mul_ps`, NEON equivalents) to implement `ggml_vec_dot_*` routines for every tensor type.

## Goals
1. Preserve SIMD parity with upstream ggml while keeping the Kotlin port “all Kotlin” (no C interop stubs).
2. Mirror ggml’s block-based quantized kernels so existing accuracy/performance characteristics carry over.
3. Keep the execution surface compatible with the coroutine-based CPU executor—SIMD kernels will be invoked from within coroutine batches without crossing language boundaries.

## Porting Strategy
1. **Introduce Vector Abstractions**
   - Create a `simd` package in `src/commonMain` that wraps `Vector128` (and any required future types) with descriptive helpers (`loadF32`, `fmaF32`, `horizontalSumF32`, etc.).
   - Provide architecture guards (`expect/actual` or runtime checks) to choose between SSE/AVX/NEON code paths while falling back to scalar loops when SIMD is unavailable.

2. **Port Core Dot Products**
   - Start with `ggml_vec_dot_f32` and `ggml_vec_dot_f16`, matching the loop unrolling and leftover handling from ggml’s C macros.
   - Add unit tests that compare Kotlin SIMD outputs against the current scalar Kotlin loops for randomized inputs to guarantee bitwise parity within acceptable tolerance.

3. **Extend to Quantized Kernels**
   - Implement Kotlin equivalents for the packed block decoders (`dequantize_q4_0`, etc.) using vector helpers to unpack and multiply in parallel.
   - Stage work per quant format (Q4_0/Q8_0 → Q4_1/Q5_* → K-quant families) so we can land incremental improvements without blocking on the entire matrix.

4. **Integrate with Compute Ops**
   - Update `GGMLComputeOps` dot-product call sites to dispatch to the SIMD helpers when strides, alignment, and block size requirements are satisfied; otherwise use the existing scalar fallback.
   - Expose simple feature flags (via `GGMLCpuRuntimeConfig`) so benchmarks can toggle SIMD usage at runtime.

5. **Benchmark & Tune**
   - Reuse the existing matmul and quantization accuracy tests; add microbenchmarks that isolate dot products to confirm throughput gains per architecture.
   - Adjust coroutine dispatcher selection if SIMD kernels become memory-bound or benefit from pinning to a dedicated pool.

## Deliverables & Milestones
1. **Phase 1 – Infrastructure (1–2 PRs)**
   - SIMD helper module with F32/F16 vector ops.
   - Unit tests verifying helper correctness on x86_64 and arm64 targets.
2. **Phase 2 – Float Kernels**
   - Kotlin-native `vec_dot_f32` and `vec_dot_f16` implementations wired into compute ops.
   - Benchmark comparison vs. scalar fallbacks.
3. **Phase 3 – Quantized Kernels**
   - Sequentially port Q4/Q5/Q8 dot products, validating against existing tests after each format.
4. **Phase 4 – Coverage & Cleanup**
   - Optional support for BF16/AVX512 paths (guarded by platform checks).
   - Documentation updates summarizing benchmark gains and architecture support matrix.

Current timeline adjustments (reflecting KLang work):
- Float SIMD enablement follows after Float32 `mulBits`/`sqrtBits` are exact; quant SIMD follows Q2_K parity.


## Risks & Mitigations
- **API Stability:** `newFixedThreadPoolContext` is marked delicate; we confine its usage behind a configuration flag and keep `Dispatchers.Default` as the default path.[^3]
- **Platform Variance:** Some intrinsics (e.g., AVX512 BF16) are not yet expressible in Kotlin. We will target SSE/AVX2/NEON first and retain scalar fallbacks for unsupported instructions.
- **Testing Coverage:** SIMD logic is notoriously prone to off-by-one errors; property-based tests comparing against scalar implementations will accompany each ported kernel.

[^1]: Kotlin Standard Library – Native `Vector128` reference (accessed Sep 19 2025). citeturn0search5
[^2]: Kotlin Multiplatform `vectorOf` helper discussion (accessed Sep 19 2025). citeturn2search0turn2search5
[^3]: Kotlin `Dispatchers.Default.limitedParallelism`/`newFixedThreadPoolContext` documentation (accessed Sep 19 2025). citeturn0search0
