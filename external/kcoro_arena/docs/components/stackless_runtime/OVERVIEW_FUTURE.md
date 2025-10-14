# Stackless Runtime Future Notes

_This document captures design ideas and open questions. Nothing here is implemented yet._

## Stretch goals

- **Work-stealing scheduler:** Evaluate a cross-thread work-stealing queue to reduce contention under heavy load.
- **Timer wheel integration:** Replace the current timer condition-variable loop with a more precise, low-overhead wheel.
- **Back-pressure hooks:** Add optional callbacks when coroutines remain parked beyond a configurable threshold (useful for observability).
- **Structured cancellation:** Layer a lightweight job/tree API over continuations so higher-level code can cancel a subtree without visiting each continuation manually.

## Research topics

- **Automatic CPS transforms:** Investigate compiler plugins or codegen helpers that can transform sequential-looking Kotlin/Native code into continuation form automatically.
- **SIMD-aware suspend points:** Explore emitting prefetch hints before suspension/resumption for data-heavy kernels.
- **Scheduler metrics:** Design a stable telemetry surface (histograms, queue depths) that can be consumed by sym/HLL monitoring.

## Open questions

1. Should we expose a unified “task” abstraction once the structured cancellation story is ready, or keep continuations as the only low-level primitive?
2. Can we deliver a portable fiber-like API on top of continuations for teams who prefer stackful semantics?
3. What is the minimum viable API for integrating external event sources (e.g., networking) without reimplementing the token kernel?

Contributions welcome—add to this list with references or prototype branches as experiments evolve.
