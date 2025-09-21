# Test Triage (2025‑09‑21)

This page captures a snapshot of test status and priorities after the KLang/HPC16 introduction and destination‑based API unification.

## Green (targeted)
- KLang bitwise/float:
  - `Float32DivTest` (CFloat32 division semantics)
  - `FloatKlangExtensionsTest` (Float on LHS ops)
- Limb arithmetic:
  - `HPC16xTests` (add/sub, basic 64×64→128 mul check)

## In Progress / Next
- Float32 soft‑float
  - Tighten `mulBits` to compiler‑rt `fp_mul_impl.inc`
  - Implement `sqrtBits` (`fp_sqrt_impl.inc`)
  - Add conversions tests
- HPC16
  - Complete carry propagation and add 128/64 division tests (Knuth D)
  - JVM BigInteger oracle for randomized limb validation (tests only)

## Red / Pending Migration
- Quantization snapshot:
  - `Q2KSnapshotTest` — parity blocked by refinement drift; will re‑check once Float32 mul/sqrt/conversions are exact.
- Legacy optimization/model tests referencing removed return‑alloc ops:
  - Update to destination‑based helpers and snapshot inputs before assertions.

## Guidance
- Prefer allocator‑backed `dst` via shared helpers; snapshot operand data before compute
- Compare raw bits where applicable (float ops); or both NaN handling per IEEE‑754
- Quantization tests: keep byte‑for‑byte diagnostics only while red; remove once parity holds

## Next Review
- After Float32 mul/sqrt land and HPC16 128/64 division is implemented, re‑run full `macosArm64Test` and update this page.
