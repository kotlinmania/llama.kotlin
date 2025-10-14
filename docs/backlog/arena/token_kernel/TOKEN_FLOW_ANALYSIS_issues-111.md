_Backlog: related issues #111_

<!-- Related issues: #111 -->
# Token Kernel Send/Receive Flow — Future Ideas

_Use this space to capture flow-level improvements or experiments._

## Enhancements to explore

- Visual tooling: generate runtime diagrams (sequence/flow) from live traces to aid debugging.
- Batch wakeups: examine whether coalescing multiple ready tokens before signalling the scheduler reduces overhead for bursty workloads.
- Timeout integration: detail how deadlines/timeouts should insert themselves into the flow without disturbing zero-spin guarantees.

## Open questions

1. Should the token kernel expose hooks for custom instrumentation (e.g., per-match callbacks before enqueuing)?
2. How would a multi-worker token kernel affect the strict FIFO property—do we need additional ordering guarantees?
3. Can we provide better backoff guidance to callers experiencing frequent `-EBUSY` responses?
