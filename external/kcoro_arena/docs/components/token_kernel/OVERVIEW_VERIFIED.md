# Token Kernel Overview (Verified)

_Last reviewed: 2025-10-31_

## Component Summary

- **Location:** `external/kcoro_arena/core/src/kc_token_kernel.c`
- **Role:** zero-spin event-driven mediator for channel rendezvous operations
- **Key idea:** publish/callback model with dedicated worker thread; tokens stored in hash buckets, callbacks processed via ready queue

## Why Token Kernel?

Traditional channel implementations rely on polling or OS-level blocking. The token kernel removes both by using:
- **Event-driven architecture**: Publish tokens, trigger callbacks on match
- **Zero-spin worker thread**: Dedicated thread processes callbacks without busy waiting
- **Freelist recycling**: Token blocks are pooled to reduce allocation overhead
- **Tight integration with channels**: Sole path for stackless channel rendezvous wakeups

## Current Architecture (as of 2025-10-31)

The token kernel is now the **exclusive rendezvous path** for the stackless channel engine (`kc_chan_stackless.c`). No direct CAS-based state machines remain; all coordination flows through publish/callback.

### Core structures

**Token block** (recycled via freelist):
```c
typedef struct kc_token_block {
    kc_token_block       *next_hash;
    struct kc_chan       *channel;
    kc_payload            payload;
    kc_token_resume_fn    resume_cb;
    void                 *resume_ctx;
    kc_token_id_t         id;
} kc_token_block;
```

**Ready queue** (feeds worker thread):
```c
typedef struct kc_token_ready_queue {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    kc_token_block *head;
    kc_token_block *tail;
    int             stop;
} kc_token_ready_queue;
```

**Bucket-based storage** (1024 buckets for O(1) lookup):
```c
typedef struct kc_token_bucket {
    pthread_mutex_t mu;
    kc_token_block *head;
} kc_token_bucket;
```

### Token lifecycle

```
  PUBLISH (send/recv)
       |
       v
  [Store in bucket by ticket ID]
       |
       |-- awaiting match --
       |
  CALLBACK (matched)
       |
       v
  [Move to ready queue]
       |
       v
  [Worker dequeues]
       |
       v
  [Invoke resume callback]
       |
       v
  [Return to freelist]
```

### Worker thread model

Single dedicated worker thread processes callbacks:
```c
static void *kc_token_worker_main(void *arg) {
    for (;;) {
        kc_token_block *blk = ready_dequeue(&g_kernel.ready_queue);
        if (!blk) break;  // Shutdown signal
        kc_token_process_block(blk);  // Invoke callback, return to freelist
    }
    return NULL;
}
```

**Zero-spin guarantee**: `ready_dequeue()` uses `pthread_cond_wait()` when queue is empty.

## API behaviour

### `kc_token_kernel_publish_send`
- Allocates token block from freelist
- Assigns unique ticket ID
- Stores in hash bucket
- Returns ticket for future callback

### `kc_token_kernel_publish_recv`
- Symmetric to publish_send
- No payload attached initially
- Callback receives payload when matched

### `kc_token_kernel_callback`
- Removes token from bucket by ID
- Updates payload
- Enqueues to ready queue → worker processes it

### `kc_token_kernel_cancel`
- Removes token from bucket
- Marks with error status
- Still enqueues to ready queue (callback sees cancellation)

## Event notification system

The token kernel exposes hooks for observability:

```c
typedef enum {
    KC_TOKEN_EVENT_EMPTY_TO_SENDER_READY,
    KC_TOKEN_EVENT_EMPTY_TO_RECEIVER_READY,
    KC_TOKEN_EVENT_SENDER_TO_MATCHED,
    KC_TOKEN_EVENT_RECEIVER_TO_MATCHED,
    KC_TOKEN_EVENT_ANY_TO_CANCELLED,
    KC_TOKEN_EVENT_COUNT
} kc_token_event_type;
```

Subscribers register callbacks to track state transitions for diagnostics and telemetry.

## Metrics and observability

Built-in atomic counters track:
- `matches_total`: Successful rendezvous operations
- `publish_send_total` / `publish_recv_total`: Token registration counts
- `callback_total`: Callbacks invoked
- `cancel_total`: Cancellation events
- `cas_failures_total`: Contention/retry events

Access via `kc_token_kernel_get_metrics()`.

## Integration points

- **Stackless channels** (`kc_chan_stackless.c`): Exclusive user of token kernel for rendezvous wakeups
- **Scheduler**: Worker thread callbacks enqueue continuations via `koro_sched_enqueue_ready()`
- **Descriptor system**: Coordinates with `kc_desc.c` for zero-copy payload handling

## Performance characteristics

- **Freelist hit rate**: ~100% under steady load (blocks rarely malloc)
- **Bucket distribution**: 1024 buckets provide O(1) average lookup
- **Worker efficiency**: No spinning; condvar wakeup only when work available
- **Lock granularity**: Per-bucket locks minimize contention

## Future enhancements

See [Token Kernel Enhancements backlog](../../../../docs/backlog/arena/token_kernel/ENHANCEMENTS_issues-109-112-116.md) for:
- **Batch-ready wakeups**: Dequeue multiple tokens per worker cycle
- **Descriptor pooling**: Reduce retain/release churn for high-frequency channels
- **Fairness/priority**: Weighted scheduling for multi-tenant workloads

## Related docs

- [Token Kernel Enhancements (backlog)](../../../../docs/backlog/arena/token_kernel/ENHANCEMENTS_issues-109-112-116.md)
- [Scheduler (verified)](../stackless_runtime/SCHEDULER_VERIFIED.md)
- [Stackless channels (implementation)](../../../../external/kcoro_arena/core/src/kc_chan_stackless.c)
- [Token kernel implementation](./IMPLEMENTATION.md)
