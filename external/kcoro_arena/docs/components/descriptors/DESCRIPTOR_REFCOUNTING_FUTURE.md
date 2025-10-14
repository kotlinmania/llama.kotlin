# Descriptor Reference Counting — Future Ideas

_Collect experiments here; keep shipping behaviour in `DESCRIPTOR_REFCOUNTING_VERIFIED.md`._

## Potential improvements

- **Shared slab allocator:** Introduce a pooled allocator for copy descriptors to reduce fragmentation for typical payload sizes (small, medium, large slabs).
- **Descriptor compression:** Investigate compacting descriptor metadata for high-throughput channels (e.g., 16-byte struct with packed fields).
- **Deferred release:** Explore batching releases to amortize hash bucket locking overhead under heavy churn.

## Observability

- Expose refcount statistics (histograms of live descriptors, alias hit/miss counts) through the runtime metrics surface.
- Track alias LRU efficiency and suggest capacity tuning based on workload.

## Open questions

1. Should we support cross-arena descriptors (single descriptor referencing buffers from multiple arenas)?
2. Can we safely expose descriptor cloning to user code, or should clones remain an internal tool for broadcast paths?
3. Would a scoped descriptor guard (RAII-style wrapper) simplify user code without leaking C semantics into higher-level APIs?
