# Token Kernel Implementation (Verified)

_Last reviewed: 2025-10-14_

## Overview

The token kernel (`kc_token_kernel.c`) implements the zero-spin rendezvous dispatcher in kcoro_arena. It routes send/receive matches via callbacks and enqueues continuations without polling.

## Core data structures

### Token block
```c
struct kc_token_block {
    kc_token_block    *next_hash;
    struct kc_chan    *channel;
    kcoro_t           *owner_co;
    kc_payload         payload;
    void              (*resume_pc)(void);
    kc_token_id_t      id;
};
```

### Global state
```c
static struct {
    atomic_uint_fast64_t next_id;
    kc_token_bucket     *buckets;
    size_t               bucket_count;
    kc_token_freelist    freelist;
    kc_token_ready_queue ready_queue;
    pthread_t            worker;
    int                  worker_started;
    atomic_int           initialized;
} g_kernel;
```

## Key operations

- **kc_token_create**: obtains a token block from the freelist (fast path) or malloc, assigns a unique ID, and zeroes fields.
- **kc_token_register**: inserts the token into a hash bucket protected by a mutex.
- **kc_token_post_ready**: appends a matched token to the ready queue and signals the worker thread.
- **Worker loop**: waits on the condition variable, dequeues ready tokens, invokes the resume callback, unregisters, and recycles the token.

```c
while (!stop) {
    pthread_mutex_lock(&ready_queue.mu);
    while (queue empty && !stop) {
        pthread_cond_wait(&ready_queue.cv, &ready_queue.mu);
    }
    token = dequeue_head();
    pthread_mutex_unlock(&ready_queue.mu);

    if (token) {
        token->resume_pc();
        kc_token_unregister(token->id);
        kc_token_destroy(token);
    }
}
```

## Callback contract

- When `resume_pc()` executes, a match has occurred and payload (if any) is stored in `token->payload`.
- The callback is responsible for enqueuing the corresponding continuation (`koro_sched_enqueue_ready`).
- After the callback returns, the token is unregistered and returned to the freelist.

Example (from stackless runtime):
```c
static void koro_send_resume_callback(void* user_data)
{
    koro_cont_t* k = (koro_cont_t*)user_data;
    if (!k) return;

    k->completed = 1;
    k->last_park_result = 0;
    koro_sched_enqueue_ready(k);
}
```

## Hash table details

- Buckets: default 1024; each bucket has its own mutex.
- Freelist: unbounded cache of token blocks to reduce malloc/free churn.
- Ready queue: protected by mutex + condition variable to provide zero-spin wakeups.

## Performance characteristics

- Token creation/registration: O(1) average.
- Post-ready: O(1) enqueue + signal.
- Worker idle CPU usage: zero (blocked on condition variable).

## Known limitations

- Buckets are fixed-size (no shrinking/growth).
- Freelist can grow unbounded under extremely bursty workloads.
- Worker thread is single-threaded; scaling to multiple cores would require additional work.

## Related docs

- [Token kernel overview (verified)](./OVERVIEW_VERIFIED.md)
- [Scheduler (verified)](../stackless_runtime/SCHEDULER_VERIFIED.md)
- [Channel design (verified)](../channels/CHANNEL_DESIGN_VERIFIED.md)
