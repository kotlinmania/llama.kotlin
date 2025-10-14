# Token Kernel Send/Receive Flow — Verified

_Last reviewed: 2025-10-14_

This walk-through reflects the current code path from stackless send/recv through the token kernel and back to the scheduler.

## Architecture at a glance

```
User coroutine  ─┐
                 │ publish token
Token kernel  ───┼─ stores & matches tokens
                 │ callback
Scheduler     ───┘ enqueue continuation
```

1. Stackless coroutine calls `koro_send_stackless` or `koro_recv_stackless`.
2. The channel layer interacts with the token kernel (`kc_token_kernel_publish_*`).
3. On match, the token kernel enqueues the continuation via its resume callback.
4. The scheduler dequeues and resumes the continuation (next step function).

## Step-by-step (send → receive)

1. **User coroutine (send):**
   ```c
   void* result = koro_send_stackless(k, ch, data, len);
   if (result == NULL) return NULL;  // suspended
   ```
2. **Publish token:** send helper registers the token via `kc_token_kernel_publish_send`.
3. **Receiver coroutine:** analogous flow via `koro_recv_stackless` → `kc_token_kernel_publish_recv`.
4. **Channel match:** channel detects match, calls `kc_token_kernel_callback` with payload.
5. **Token kernel:** `kc_token_post_ready` enqueues the token and signals the worker.
6. **Worker thread:** waits on condition variable, dequeues ready token, invokes resume callback (`resume_pc`).
7. **Resume callback:** updates continuation state/payload and enqueues it in scheduler (`koro_sched_enqueue_ready`).
8. **Scheduler:** picks continuation from ready queue, calls `next_step` to resume the coroutine at the suspension point.

## Properties ensured today

- Zero-spin operation (worker blocks on `pthread_cond_wait`).
- FIFO order in ready queue ensures fairness for matched tokens.
- Payload or descriptor IDs are handed off through the token structure (`kc_payload`).

## Related docs

- [Token kernel overview (verified)](./OVERVIEW_VERIFIED.md)
- [Token kernel implementation (verified)](./IMPLEMENTATION_VERIFIED.md)
- [Stackless scheduler (verified)](../stackless_runtime/SCHEDULER_VERIFIED.md)
