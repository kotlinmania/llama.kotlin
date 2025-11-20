# Destination-Based API Migration (Status & Guidance)

Updated: 2025‑09‑21

## Summary
All compute ops now use destination tensors (no internal allocation). Legacy return‑allocating helpers (e.g., `computeAddRet`) were removed. Graph execution and tests must allocate `dst` explicitly via allocator helpers, enabling aggressive inplace reuse and predictable memory planning.

## Why
- Eliminates hidden allocations and GC pressure
- Aligns with GGML’s architecture for reuse/inplace
- Makes graph‑level memory planning deterministic and testable

## What Changed
- Old: `computeAdd(a,b): GGMLTensor`
- New: `computeAdd(a,b,dst)` — writes directly into `dst`
- Tests and benchmarks updated to allocate `dst` with shared helpers
- Some ops gained convenience wrappers (e.g., `NEG/RELU/GELU`) during migration but those wrappers no longer allocate

## Test Guidance
- Snapshot inputs before invoking compute ops — operands may be mutated or share buffers due to inplace planning
- Use `allocateLike(allocator, src, type?)` (or equivalent) to allocate `dst`
- Prefer shared test utilities (`GGMLTestUtils`) for shapes/strides and allocator setup

## Known Follow‑ups
- Sweep remaining tests that compute expectations after mutation; ensure they snapshot inputs first
- Keep one API path: no v1/v2, no env flags — tests should reflect the new patterns

## Related Docs
- KOTLIN_PORT_CHECKLIST.md (Immediate Next Steps)
- docs/kdocs/klangnative-soft-float-and-hpc16.md (numeric determinism)
- docs/kdocs/kotlin-native-simd-plan.md (SIMD on top of the unified API)
