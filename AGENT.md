# AGENT NOTES

These notes capture the state of the llama.kotlin port and provide quick pointers when handing work between agents.

## Project Snapshot (2025-10-07)
- Goal: idiomatic Kotlin/Native port of llama.cpp with an optimized CPU backend (additional accelerators deferred).
- Primary roadmap documents:
  - `KOTLIN_PORT_CHECKLIST.md` (authoritative task list and phase tracking)
  - `KOTLIN_PORT_STATUS.md` (high-level progress & milestones)
  - `GGML_COMPUTE_OPS_DESIGN.md`, `GGML_ALLOC_DESIGN.md`, `GGUF_IMPLEMENTATION.md` (technical design notes)
- Kotlin sources live under `src/nativeMain/kotlin/ai/solace/llamakotlin`.
- Legacy C/C++ sits under `src/` for reference only.

## Testing Infrastructure
- Framework: **kotlin.test** (JetBrains’ standard multiplatform test library). No Kotest dependency.
- Shared tests live in `src/commonTest/kotlin`. Platform-specific suites can extend them under `*/Test` as needed.
- JVM tests are currently disabled (see `build.gradle.kts`) to keep the focus on Kotlin/Native; prefer `macosArm64Test` once fixtures are regenerated.
- Useful commands:
  - `./gradlew build -x macosArm64Test` – primary sanity check (macOS fixtures still pending regeneration).
  - `./gradlew jsTest`, `linuxX64Test`, `macosArm64Test` – targeted native/JS runs when fixtures/assets are ready.

## Klang Primitives (bitwise & numeric core)
- Location: `src/commonMain/kotlin/ai/solace/klang/**`.
- Key modules:
  - `bitwise/BitPrimitives.kt` – clz/ctz/popcount, rotations, bit-field insert/extract.
  - `bitwise/PackOps.kt` – nibble & bit-plane packing helpers aligned with merge-dev quantizers.
  - `buffer/MemoryOps.kt` – memcpy/memset/memcmp equivalents for Byte/Short/Int/Float arrays.
  - `fp/VectorOps.kt` – deterministic dot/axpy helpers built on soft-float operations.
  - `common/StatOps.kt` – mean, variance, MAD, uniform RNG.
  - `int/hpc/HPC16x8.kt` – 128-bit arithmetic (umul/divmod/mulHi) for BitNet LUT work.
  - `bitwise/Float32Math.kt` – soft-float add/mul/div/fma, lrint/nearbyint, conversions.
- All bit operations route through `BitShiftEngine`/`ArithmeticBitwiseOps` for consistent semantics across Native targets.
- Tests for these helpers live in `src/commonTest/kotlin/ai/solace/klang/KlangPrimitiveTests.kt`.

## Quantization & BitNet Status
- `QuantizationHelper.kt` now includes ports of merge-dev `make_qkx2/3`, `make_qp`, `make_qx`, and the new `make_q3_quants`. Q2_K/Q4_K/Q5_K/Q6_K continue to call the helper layer; Q3_K has been refactored to share the same machinery and scratch buffers.
- `GGMLComputeOps.kt` uses the shared helpers for Q2_Q3_Q4_Q5_Q6; Q3_K quantization was re-written for merge-dev parity, and helper utilities (`PackOps`, `BitPrimitives`) replaced ad-hoc shifts.
- `GGMLTypes.kt` accessor/setter routines for K-Quant blocks now route through the same helper layer (no raw `shl`/`and`).
- BitNet LUT + bias transform from `ggml_bitnet_transform_tensor` is still pending; HPC16x8 primitives are ready for it.
- Fixtures for `macosArm64Test` need regeneration once parity is achieved.

## Recommended Workflow for New Tasks
1. Read `KOTLIN_PORT_CHECKLIST.md` for current priorities (kept up to date with every handoff).
2. Review `AGENT.md` (this file) and `KOTLIN_PORT_STATUS.md` before coding.
3. When touching quantizers, use `BitPrimitives`/`PackOps`/`MemoryOps` helpers instead of ad-hoc bit twiddling.
4. Regenerate fixtures with the merge-dev C reference located at `/Volumes/emberstuff/Projects/llama.cpp-bitnet` once parity work is done.
5. Document any new helpers or conventions here and in the status/checklist docs.

## Useful Commands & Locations
- Build: `./gradlew build -x macosArm64Test`
- Tests: `./gradlew jsTest`, `./gradlew linuxX64Test`, `./gradlew macosArm64Test`
- Backend and quantization test suites now reside in `src/commonTest/...`; everything compiles across targets with CPU-only assumptions. Platform-specific experiments (kcoro ping-pong) remain under `src/macosArm64Test`.
- kcoro ping-pong benchmarks/tests live under `src/macosArm64Test/...` and require running `macosArm64Test` (or `kcoroBench`) on Apple Silicon.
- Benchmarks: `./gradlew kcoroBench`
- Merge-dev reference repo: `/Volumes/emberstuff/Projects/llama.cpp-bitnet`

## Handoff Notes
- Keep an eye on the `macosArm64Test` fixtures; they are currently disabled from CI via `-x` and must be restored once regenerated.
- When introducing new primitives, prefer adding shared tests in `src/commonTest` so coverage runs everywhere.
- Document every substantial change in `KOTLIN_PORT_CHECKLIST.md` and update this file if workflow guidance changes.
