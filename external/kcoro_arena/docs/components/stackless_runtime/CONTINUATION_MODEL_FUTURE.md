# Continuation Model — Future Ideas

_Track structured concurrency and ergonomic improvements here._

## Ideas under consideration

- **Structured concurrency:** Add parent/child relationships so cancellation and error propagation flow through continuation trees.
- **Reference counting:** Introduce refcounts for continuations that are shared across schedulers or held by external subsystems.
- **Debug introspection:** Provide APIs to snapshot continuation fields (state, next_step) for debugging tools.

## Questions

1. Should we surface a higher-level "task" abstraction over continuations once structured concurrency is ready?
2. Can we auto-generate the switch-based state machine from sequential Kotlin/Native code?
3. How can we integrate cancellation tokens cleanly without complicating the core struct?
