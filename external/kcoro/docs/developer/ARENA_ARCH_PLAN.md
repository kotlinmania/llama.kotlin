# Unified Token Queue + Arena Architecture Tasks

This note captures the concrete architectural work required to finish the "one true path" channel stack with the MLX-inspired arena.

## 1. Arena-Backed Descriptor Layer
- Implement `kc_rv_arena` with fixed-size pages, free lists, and refcounted descriptors.
- Define 64-bit tickets (`arena_id | page_index | length | flags`) and expose retain/release helpers.
- Replace `kc_chan_ptrmsg` copies with descriptor issuance in rendezvous send/recv.
- Add checksum/generation fields so stale tickets are rejected during hydrate.

## 2. Channel Integration (All Kinds)
- Route buffered, unlimited, conflated, and zero-copy variants through the arena descriptors and token queues.
- Remove the remaining `-ENOTSUP` stubs in `kc_chan_send/_recv` and pointer wrappers.
- Update select registration/cancel paths to append arena-backed pending nodes exclusively.
- Ensure cancellation and close scrub descriptor refs via `release_ticket()`.

## 3. Token Kernel Worker Loop
- Introduce a dedicated worker thread (or pool) that drains token events instead of invoking callbacks inline.
- Provide a lock-free queue between channel wake sites and the worker to keep rendezvous hot paths short.
- Extend the worker to handle timeouts, cancellations, and arena compaction triggers.

## 4. Instrumentation & Tooling
- Extend `kc_chan_snapshot()` to expose arena depth, ticket churn, spill events, and worker backlog.
- Teach chanmon/bench harness to read the new fields and emit BizTalk-style queue diagnostics.
- Add an admin dump (`kc_arena_dump()` or chanmon command) that lists live tickets, page chains, and refcounts.

## 5. Persistence & Spill Hooks
- Layer an optional write-ahead log behind the arena allocator for crash recovery experiments.
- Support pluggable spill targets (shared memory, file-backed mmap) when RAM pressure exceeds thresholds.
- Document durability guarantees and how tooling replays the log to rehydrate pending tickets.

## 6. Bench & Test Coverage
- Rebuild buffered/unlimited latency + throughput benches on the unified path (no legacy waiters).
- Add unit tests that assert descriptor retain/release on send/recv, select win, cancel, close, and timeout.
- Stress-test arena compaction with mixed small/large payloads to watch for fragmentation regressions.

---
**Status:** Rendezvous pointer path already rides the token queues; all other items remain outstanding.
