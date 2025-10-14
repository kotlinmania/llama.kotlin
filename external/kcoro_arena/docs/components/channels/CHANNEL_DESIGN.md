# Channel Design Overview (kcoro_arena)

This note explains how channels behave in the stackless arena runtime. Read it when you want a refresher on what happens inside `kc_chan.c` without wading through every line of C code.

## Channel flavors at a glance

| Kind | What it means | When to use it |
|------|----------------|----------------|
| `KC_RENDEZVOUS` | Sender and receiver meet in the middle. No buffering. | Handshake-style protocols where the producer must wait for the consumer. |
| `KC_BUFFERED` | Bounded queue of fixed-size elements. | High-throughput pipelines where occasional bursts should not back up the producer immediately. |
| `KC_UNLIMITED` | Segmented, growable queue. | Convenience for tooling/tests when you do not want to think about capacity. |
| `KC_CONFLATED` | Single slot keeps only the most recent value. | State replication (“latest config wins”). |

All variants share the same core scaffolding: a mutex-protected struct, per-kind storage, and two wait queues (send and receive) implemented as linked lists of `kc_waiter` records.

## Life of a rendezvous

1. **Fast path (no waiters):**
   - Receiver arrives first → it pushes a waiter onto the receive queue and parks.
   - Sender arrives → it sees the waiting receiver, copies the payload directly into the receiver’s buffer, and schedules the receiver’s continuation.
2. **Both sides waiting:**
   - The token kernel keeps track of who is parked. Only one sender+receiver pair wins the match. Others remain queued.
3. **Resumption:**
   - When the winning match completes, the token kernel calls back into the scheduler (`koro_sched_enqueue_ready`) so the parked continuation resumes on the next scheduler tick.

Because kcoro_arena is stackless, suspend/resume boils down to “store the next step and return.” Channels never switch stacks—they only manage continuations.

## Buffered channels

Buffered channels add a ring buffer in front of the rendezvous mechanics:

- Producers write into the ring when it is not full. If it fills up, they fall back to the rendezvous path and park in the send queue.
- Consumers read from the ring while it has data. If it empties, they look for parked senders or park themselves.
- Resizing is automatic for the unlimited flavor; the buffer grows in chunks so we do not reallocate on every push.

## Pointer payloads and zero copy

The “ptr” variants (e.g., `kc_chan_send_ptr`) wrap arena descriptors. The important points are:

- The alias-LRU cache remembers recently seen descriptors so repeat sends of the same pointer avoid new lookups.
- Descriptors carry their own reference counts. The channel retains references while a message is in-flight and releases them once the consumer finishes.

## Cancellation and close

- Cancelling a waiter removes it from the queue immediately and signals the coroutine with `KC_ECANCELED`.
- Closing a channel wakes both senders and receivers. Rendezvous pairs finish their current transfer first; buffered data is drained before close reports `KC_EPIPE`.

## Further reading

- [Token kernel overview](../token_kernel/OVERVIEW.md) for the callback machinery.
- [Descriptor refcounting](../descriptors/DESCRIPTOR_REFCOUNTING.md) to understand the zero-copy path.
- `external/kcoro_arena/core/src/kc_chan.c` for the exact implementation (search for the block matching the channel kind you care about).

If you update the channel code, add a short note here so the mental model and reality stay aligned.
