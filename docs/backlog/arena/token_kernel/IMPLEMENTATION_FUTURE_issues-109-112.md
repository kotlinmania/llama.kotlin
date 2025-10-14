<!-- Related issues: #109, #112 -->
# Token Kernel Implementation — Future Notes

_Use this document to track experiments and improvements for the token kernel._

## Potential optimizations

- **Lock-free buckets:** Explore CAS-based bucket updates to reduce mutex contention under high concurrency.
- **Batch dequeue:** Dequeue multiple tokens at once before releasing the ready-queue lock to cut down on wakeups when the system is busy.
- **Multi-worker model:** Prototype a small pool of worker threads (one per NUMA node) to scale callback throughput on multi-core systems.

## Diagnostics & telemetry

- Emit per-ticket metrics (match count, busy retries, average wait time) surfaced through the runtime metrics API.
- Add optional tracing hooks around `kc_token_post_ready` and worker callbacks to integrate with external profilers.

## API experiments

- Provide a non-blocking “try match” helper for specialized channels that can handle retries more gracefully.
- Consider exposing a cancellation API that can remove pending tokens before they match (useful for timeouts).

Add further ideas here; keep the verified implementation doc focused on code that exists today.
