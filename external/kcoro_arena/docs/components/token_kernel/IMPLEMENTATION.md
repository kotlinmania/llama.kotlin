# Token Kernel Implementation Details

## Overview

The token kernel (`kc_token_kernel.c`) is the zero-spin event dispatcher for kcoro_arena. It implements a complete, production-ready system that eliminates all spin loops through callback-based resumption.

## Core Data Structures

### Token Block
```c
struct kc_token_block {
    kc_token_block    *next_hash;      // Hash collision chain
    struct kc_chan    *channel;        // Associated channel (if any)
    kcoro_t           *owner_co;       // Owning coroutine
    kc_payload         payload;        // Data payload
    void              (*resume_pc)(void); // Callback to invoke on match
    kc_token_id_t      id;            // Unique token ID
};
```

### Global Kernel State
```c
static struct {
    atomic_uint_fast64_t next_id;      // Atomic ID generator
    kc_token_bucket     *buckets;      // Hash table buckets
    size_t               bucket_count; // Number of buckets
    kc_token_freelist    freelist;     // Recycled tokens
    kc_token_ready_queue ready_queue;  // Matched tokens awaiting dispatch
    pthread_t            worker;        // Background worker thread
    int                  worker_started;
    atomic_int           initialized;
} g_kernel;
```

## Key Operations

### Token Creation and Registration

**kc_token_create()**
```
1. Try to allocate from freelist (fast path)
2. If empty, malloc new token block
3. Generate unique ID atomically
4. Zero-initialize fields
5. Return token block
```

**kc_token_register()**
```
1. Compute hash bucket: id % bucket_count
2. Lock bucket
3. Insert at head of collision chain
4. Store channel, owner, callback
5. Unlock bucket
```

### Zero-Spin Matching

**kc_token_post_ready()**
```
1. Lock ready queue
2. Append token to tail (FIFO order)
3. Signal condition variable
4. Unlock queue
5. Background worker wakes immediately
```

This is the critical "push" operation that eliminates spin. When a send/receive operation completes, it calls `post_ready`, which wakes the worker thread via `pthread_cond_signal`. The worker then invokes the registered callback on its own thread, making the coroutine runnable.

### Background Worker Thread

**token_kernel_worker()**
```
while (!stop) {
    lock(ready_queue.mu);
    while (queue empty && !stop) {
        pthread_cond_wait(&ready_queue.cv, &ready_queue.mu);  // Zero CPU here
    }
    token = dequeue_head();
    unlock(ready_queue.mu);
    
    if (token) {
        token->resume_pc();  // Invoke callback (e.g., koro_send_resume_callback)
        kc_token_unregister(token->id);
        kc_token_destroy(token);
    }
}
```

The worker spends zero CPU when idle because it blocks on `pthread_cond_wait`. This is the core of the zero-spin design.

## Callback Contract

When `resume_pc()` is invoked:
- The token has been matched
- The payload (if any) is available in `token->payload`
- The callback must enqueue the coroutine into the scheduler's ready queue
- After the callback returns, the token is unregistered and destroyed

Example callback (from `kcoro_stackless.c`):
```c
static void koro_send_resume_callback(void* user_data)
{
    koro_cont_t* k = (koro_cont_t*)user_data;
    if (k) {
        k->completed = 1;
        k->last_park_result = 0;
        koro_sched_enqueue_ready(k);  // Make coroutine runnable
    }
}
```

## Hash Table Details

### Sizing
- Default: 1024 buckets
- Load factor: unbounded (collision chains grow as needed)
- Lookup: O(1) average, O(n) worst case per bucket

### Thread Safety
- Each bucket has its own mutex (fine-grained locking)
- Freelist has separate mutex
- Ready queue has separate mutex + condition variable
- No global locks for token operations

## Memory Management

### Freelist Strategy
- Destroyed tokens return to freelist instead of being freed
- Reduces malloc/free overhead for high-churn workloads
- Freelist is unbounded (no size limit)

### Token Lifecycle
```
malloc/freelist → register → [waiting] → post_ready → callback → unregister → freelist
```

## Performance Characteristics

- **Token creation**: O(1) with freelist, fallback to malloc
- **Token registration**: O(1) average (hash table insert)
- **Token lookup**: O(1) average (hash table search)
- **Post ready**: O(1) (enqueue + signal)
- **Callback invocation**: O(1) (dequeue + function call)

**Zero CPU when idle**: The worker thread blocks on condition variable, using no CPU cycles until an event arrives.

## Integration with Stackless Coroutines

The token kernel bridges the gap between arena send/receive primitives and the stackless scheduler:

1. Coroutine calls `koro_send_cps(ticket, payload, callback)`
2. Send logic creates a token, registers callback
3. When a receiver arrives, token kernel calls `post_ready(token)`
4. Worker thread invokes `callback(user_data)`
5. Callback enqueues coroutine into scheduler's ready queue
6. Scheduler calls `k->next_step(k)` on next iteration

This is the BizTalk-inspired "instance subscription" pattern: a specific token (subscription) is created for a specific coroutine instance, and the message (callback invocation) is routed precisely to that instance via the token ID.

## Future Optimizations

- Lock-free hash table (CAS-based bucket updates)
- Per-thread ready queues (reduce contention)
- Batch callback invocation (amortize locking overhead)

## Known Limitations

- Hash table cannot shrink (buckets are fixed at initialization)
- Freelist grows unbounded (no high-water mark eviction)
- Single worker thread (could be pool for multi-core scaling)

These are acceptable trade-offs for the current design goals: simplicity, correctness, and zero-spin operation.
