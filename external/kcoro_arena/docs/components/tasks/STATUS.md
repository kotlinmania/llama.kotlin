# Task Component Status

## Current State: REMOVED/OBSOLETE

The `kc_task` component (`kc_task.c`, `kcoro_task.h`) is intentionally stubbed out and marked with `#error`.

## Reason for Removal

The experimental task abstraction layer was found to be unnecessary given the stackless coroutine model. The token kernel + stackless continuations provide all the necessary primitives for concurrent task execution without an additional abstraction layer.

## What It Was

`kc_task` was an early experiment in creating a "lightweight task" abstraction that sat between coroutines and the scheduler. It provided:
- Task context (`kc_task_ctx_t`)
- Task state machine (RUNNING/BLOCKED/READY)
- Task yield/block/wake primitives

## Why It Was Removed

1. **Redundancy**: Stackless continuations (`koro_cont_t`) already provide task-like semantics
2. **Complexity**: Added an unnecessary layer between user code and the scheduler
3. **Golden Path**: Simplified architecture eliminates this component entirely

## Migration Path

Code that referenced `kc_task` should use:
- `koro_cont_t` for task representation
- `koro_park_cps()` for suspension
- Token kernel callbacks for resumption
- Scheduler ready queue for dispatch

## File Status

- `core/src/kc_task.c`: Contains `#error` directive to catch stale references
- `include/kcoro_task.h`: Header exists but should not be used
- Future: These files will be fully removed in next cleanup pass

## See Also

- `docs/components/stackless_runtime/` - The replacement architecture
- `docs/GOLDEN_PATH_COMPLETE.md` - Simplified design rationale
