# Continuation Model — Verified

_Last reviewed: 2025-10-16_

## `koro_cont` structure

```c
struct koro_cont {
    koro_step_fn next_step;
    int          state;
    void        *user_data;
    struct koro_cont *parent; // reserved for structured concurrency
    int          flags;       // RUNNING, SUSPENDED, COMPLETED, etc.
    // scheduler metadata follows
};
```

Key fields:
- `next_step`: function pointer invoked on resume (`k->next_step(k)`).
- `state`: integer state machine slot used by switch-based CPS macros.
- `user_data`: heap-allocated locals struct (persists across suspensions).
- `parent`: reserved for structured concurrency (unused in current implementation).
- `flags`: scheduler status bits (`KORO_RUNNING`, `KORO_SUSPENDED`, `KORO_COMPLETED`, `KORO_CANCELLED`).

## Lifecycle

1. **Allocation**
   ```c
   k = malloc(sizeof *k);
   k->next_step = entry_step;
   k->state = 0;
   k->user_data = malloc(locals_size);
   k->flags = KORO_SUSPENDED;
   ```
2. **Enqueue**: `koro_sched_enqueue_ready(k)`.
3. **Dispatch**: scheduler dequeues, sets `KORO_RUNNING`, calls `k->next_step(k)`.
4. **Suspension**: step function returns without `KORO_COMPLETED`; scheduler re-enqueues when appropriate.
5. **Completion**: step sets `KORO_COMPLETED`; scheduler frees continuation and `user_data`.

## Memory considerations

- Continuation struct: ~200 bytes.
- `user_data`: user-defined; freed when coroutine completes.
- Scheduler cleans up automatically for managed continuations (`koro_go`). Manual continuations must be freed/reused by the caller.

## Comparison to traditional stacks

| Aspect | Traditional stack | Continuation record |
|--------|-------------------|---------------------|
| Allocation | `mmap` (~64 KB) | `malloc` (~200 B + locals) |
| Locals | Automatic | Stored in `user_data` |
| Control flow | CPU call/ret | Explicit `next_step` / `state` switch |
| Suspension | Block/yield | Return to scheduler |

## Example

```c
struct producer_locals { int i; int ticket; };

static void producer_step(koro_cont_t* k) {
    struct producer_locals* st = k->user_data;
    switch (k->state) {
        case 0:
            st->i = 0;
            k->state = 1;
            // fall through
        case 1:
            if (st->i >= 5) {
                k->flags |= KORO_COMPLETED;
                return;
            }
            koro_send(k, st->ticket, (void*)(intptr_t)st->i);
            st->i++;
            k->state = 1;
            return;
    }
}
```

## Task Abstraction (Optional Higher-Level Layer)

The task abstraction provides structured concurrency features on top of raw continuations. This is an optional layer—raw continuations remain available for low-level control.

### `koro_task` structure

```c
struct koro_task {
    koro_cont_t* cont;              // Underlying continuation
    koro_task_t* parent;            // Parent task (for tree structure)
    koro_task_t* first_child;       // First child in linked list
    koro_task_t* next_sibling;      // Next sibling
    atomic_int refcount;            // Reference counting
    atomic_int state;               // Task state flags
    atomic_int cancel_requested;    // Cancellation flag
    void* result;                   // Completion result
    koro_task_completion_fn completion_cb;  // Callback on completion
    // ... join support fields ...
};
```

### Key Features

1. **Structured Concurrency**: Tasks form trees via parent/child relationships
2. **Reference Counting**: Shared ownership via `koro_task_retain()`/`koro_task_release()`
3. **Cancellation Propagation**: `koro_task_cancel()` cascades to all children
4. **Join Semantics**: Tasks can wait for other tasks via `KORO_TASK_JOIN()` macro
5. **Completion Callbacks**: Optional callbacks on task completion

### Task Lifecycle

1. **Creation**: `koro_task_create()` or `koro_task_spawn()` (create + schedule)
2. **Execution**: Scheduler runs underlying continuation normally
3. **State Tracking**: Task wrapper updates state flags (RUNNING, SUSPENDED, COMPLETED)
4. **Cancellation**: Cooperative via `koro_task_is_cancelled()` checks
5. **Completion**: Notifies joiners and calls completion callback
6. **Cleanup**: Reference counted destruction via `koro_task_release()`

### Example: Spawning Child Tasks

```c
static void* parent_step(koro_cont_t* k) {
    KORO_BEGIN(k);
    
    koro_task_t* self = koro_task_current();
    
    // Spawn child tasks
    koro_task_t* child1 = koro_task_spawn(worker_step, data1, size, self);
    koro_task_t* child2 = koro_task_spawn(worker_step, data2, size, self);
    
    // Parent continues execution...
    KORO_YIELD(k);
    
    // Check cancellation
    if (koro_task_is_cancelled(self)) {
        // Cleanup and exit
        KORO_END(k);
    }
    
    // More work...
    
    KORO_END(k);
}
```

### Example: Task Cancellation

```c
// Cancel a task and all its children
koro_task_cancel(parent_task);

// Within task code, check for cancellation
if (koro_task_is_cancelled(koro_task_current())) {
    // Graceful shutdown
    cleanup();
    KORO_END(k);
}
```

### API Summary

| Function | Purpose |
|----------|---------|
| `koro_task_create()` | Create task without scheduling |
| `koro_task_spawn()` | Create and schedule task |
| `koro_task_retain()`/`release()` | Reference counting |
| `koro_task_cancel()` | Request cancellation (cooperative) |
| `koro_task_is_cancelled()` | Check cancellation status |
| `koro_task_get_state()` | Query task state flags |
| `koro_task_current()` | Get current executing task |
| `koro_task_count_children()` | Count direct children |
| `KORO_TASK_JOIN()` | Wait for task completion (macro) |

### Design Principles

1. **Thin wrapper**: Tasks add minimal overhead to continuations
2. **Opt-in**: Raw continuations still available for performance-critical code
3. **Composable**: Works seamlessly with existing scheduler and macros
4. **Thread-safe**: Task tree modifications protected by mutex
5. **Ergonomic**: Higher-level spawn/cancel/join API for common patterns

### Memory Overhead

- Task structure: ~128 bytes (includes refcount, pointers, state)
- Plus continuation: ~200 bytes
- Total per task: ~330 bytes (vs ~200 bytes for raw continuation)

The overhead is acceptable for most use cases requiring structured concurrency features.

## Related docs

- [Continuation guide (verified)](./CONTINUATION_GUIDE_VERIFIED.md)
- [Scheduler (verified)](./SCHEDULER_VERIFIED.md)
- Task API header: `external/kcoro_arena/include/koro_task.h`
