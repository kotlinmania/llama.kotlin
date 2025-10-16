_Backlog: related issues #107, 117_

<!-- Related issues: #107, #117 -->
# Continuation Model — Future Ideas

_Track structured concurrency and ergonomic improvements here._

## Status Update (2025-10-16)

Issue #117 has been **partially addressed** with an experimental task abstraction implementation:

- ✅ Task API header and implementation added (`koro_task.h`, `koro_task.c`)
- ✅ Parent-child relationships for structured concurrency
- ✅ Reference counting for shared task ownership
- ✅ Cancellation propagation API (cooperative)
- ✅ Documentation in [TASK_API.md](../../../external/kcoro_arena/docs/components/stackless_runtime/TASK_API.md)
- ⚠️ Task completion tracking is best-effort (not fully integrated with scheduler)
- ⚠️ Join operations defined but require additional scheduler work

The task abstraction provides the core API surface for higher-level task management while maintaining compatibility with raw continuations. Further scheduler integration would improve completion tracking and enable full join semantics.

## Ideas under consideration

- **Structured concurrency:** ✅ Basic parent/child relationships implemented
- **Reference counting:** ✅ Implemented via `koro_task_retain()`/`release()`
- **Debug introspection:** Provide APIs to snapshot continuation fields (state, next_step) for debugging tools.
- **Tighter scheduler integration:** Automatic task state updates on continuation lifecycle events
- **Task-local storage:** Per-task data that's automatically cleaned up
- **Priority scheduling:** Task priorities for weighted scheduling

## Questions

1. ~~Should we surface a higher-level "task" abstraction over continuations once structured concurrency is ready?~~ → **Answered**: Yes, experimental implementation delivered (issue #117)
2. Can we auto-generate the switch-based state machine from sequential Kotlin/Native code?
3. How can we integrate cancellation tokens cleanly without complicating the core struct? → **Partial answer**: Task wrapper provides cancellation API without modifying core continuation struct
