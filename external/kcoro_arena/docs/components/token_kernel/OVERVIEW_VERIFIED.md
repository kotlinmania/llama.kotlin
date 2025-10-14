# Token Kernel Overview (Verified)

_Last reviewed: 2025-10-14_

## Component Summary

- **Location:** `external/kcoro_arena/core/src/kc_token_kernel.c`
- **Role:** zero-spin rendezvous mediator between senders and receivers
- **Key idea:** each ticket maps to an atomic rendezvous cell; callbacks fire immediately on match, and continuations are enqueued without spinning

## Why Token Kernel?

Traditional channel implementations rely on polling or OS-level blocking. The token kernel removes both by using:
- Lock-free atomic transitions (no mutexes in hot paths)
- Callback-based wakeups (no busy waiting)
- Tight integration with the scheduler (continuations are enqueued as soon as callbacks finish)

## Core concepts

### Rendezvous cell structure

```c
typedef struct {
    _Atomic(int) state;      // EMPTY, SENDER_READY, RECEIVER_READY, MATCHED
    void* payload;
    size_t payload_len;
    void* sender_cb_arg;
    void* receiver_cb_arg;
    send_callback_t send_cb;
    recv_callback_t recv_cb;
    kcoro_t* sender_coro;
    kcoro_t* receiver_coro;
} rendezvous_cell_t;
```

### State machine

```
     EMPTY
       |
       | sender arrives
       v
  SENDER_READY ─────────┐
       |                │
       | receiver       │ receiver arrives first
       | arrives        │
       v                v
    MATCHED <───── RECEIVER_READY
       |
       | callbacks dispatched
       v
     EMPTY (reset)
```

Only one sender and one receiver win per cycle. CAS operations ensure exclusivity.

### Callback flow

1. Sender/receiver arrive and attempt to match.
2. On success, both callbacks fire immediately (receiver first, then sender).
3. The token kernel enqueues both continuations via `koro_sched_enqueue_ready`.
4. The cell resets to `EMPTY` for the next rendezvous.

## API behaviour

### `kc_token_send`
- Immediate match if a receiver is waiting (`0` return code).
- Registers sender token and returns `KORO_SUSPEND` if cell is empty.
- Returns `-EBUSY` if another sender is registered.

### `kc_token_recv`
- Symmetric to send: matches waiting sender, otherwise registers receiver token.

## Pseudocode snippets

Sender registration:
```c
if (atomic_compare_exchange_strong(&cell->state, &expected, MATCHED)) {
    // matched receiver already present
    ... // invoke callbacks, enqueue continuations, reset state
} else if (atomic_compare_exchange_strong(&cell->state, &expected, SENDER_READY)) {
    // register sender token
    ...
    return KORO_SUSPEND;
} else {
    return -EBUSY;
}
```

Receiver path mirrors the sender.

## Integration points

- Channels push senders/receivers into the token kernel when a rendezvous is needed.
- Scheduler wakes parked continuations once the token kernel enqueues them.

## Related docs

- [Scheduler (verified)](../stackless_runtime/SCHEDULER_VERIFIED.md)
- [Channel design (verified)](../channels/CHANNEL_DESIGN_VERIFIED.md)
- [Token kernel implementation](./IMPLEMENTATION.md)
