# Channel Design Ideas (Future)

_Concepts captured here are exploratory and not implemented._

## Potential enhancements

- **Fair queueing:** Introduce priority-aware queues so high-priority fibers do not starve when sharing a channel with bulk traffic.
- **Back-pressure metrics:** Emit structured telemetry (queue depth, wait time percentiles) to help upstream systems adapt their send rates.
- **Batch wakeups:** Allow the token kernel to coalesce multiple ready continuations when the scheduler thread is idle, improving throughput under bursty loads.
- **Zero-copy slices:** Investigate passing arena-backed sliced ranges directly without requiring a full descriptor retain/release cycle each time.

## API questions

1. Should we expose “peek” APIs for conflated channels to observe the latest value without consuming it?
2. Would an optional timeout parameter on send/recv reduce boilerplate, or should timeouts stay explicit via the token kernel?
3. How should we model multi-consumer broadcast semantics: dedicated broadcast channels, or combinators that fan out to multiple rendezvous channels?

Add new experiments or spike notes here so the verified document stays focused on shipping behaviour.
