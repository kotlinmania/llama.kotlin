# Stackless Scheduler — Future Ideas

_Things we’d like to explore once the current scheduler is rock-solid._

## Queueing & Stealing

- **Work-stealing extensions:** Prototype a per-thread deque + stealing model to allow multiple scheduler threads without losing the intrusive ready-queue design.
- **Batch dispatch:** Evaluate grouping several ready continuations before yielding back to the event loop to reduce condition-variable chatter under bursty traffic.

## Telemetry

- Structured metrics: expose queue depth, wait time percentiles, and callback lag so operator tooling can spot slow producers/consumers.
- Tracing hooks: optional callbacks around `koro_run` iterations for integration with low-cost tracing systems.

## API surface

- Higher-level async helpers: consider providing futures/promises built on continuations for teams that prefer promise-style flows.
- Scheduler domains: allow more than one logical scheduler so subsystems can isolate their continuations without spinning up new threads.

## Open questions

1. Should we offer an opt-in preemption checkpoint (e.g., cooperative slice accounting) for long-running compute loops?
2. How do we keep the event loop portable if we add platform-specific optimizations (e.g., kqueue vs. epoll)?
3. Is it worth integrating the scheduler with coroutine cancellation tokens directly, or should cancellation continue to be managed at the token-kernel layer?

Add new experiments here; keep `SCHEDULER_VERIFIED.md` focused on shipping behaviour.
