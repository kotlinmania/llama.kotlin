# Stackless Continuation Ideas (Future)

_Use this file to park experiments and questions about improving the developer experience._

## Potential improvements

- **Macro generator CLI:** Provide a small tool that can translate a sequential Kotlin/Native function into the macro-heavy CPS form automatically.
- **Inspector tooling:** Expose a dbg command that prints the next_step/state/user_data snapshot for a continuation while the scheduler is paused.
- **Type-safe state machine:** Explore generating a sealed interface around continuation steps to catch missing suspend branches at compile time.

## Open questions

1. Should we surface higher-level helpers (e.g., async/await lookalikes) on top of the macros, or keep the API low-level to encourage explicit CPS thinking?
2. Can we integrate code-gen with the token kernel so that common rendezvous patterns (send/recv pairs) are emitted automatically?
3. Would a debugging hook to visualize the continuation queue (similar to the Merlin Gantt view) be worth the complexity?

Add experiment notes here without worrying about breaking the verified document.
