# Continuation Model — Verified

_Last reviewed: 2025-10-14_

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

## Related docs

- [Continuation guide (verified)](./CONTINUATION_GUIDE_VERIFIED.md)
- [Scheduler (verified)](./SCHEDULER_VERIFIED.md)
