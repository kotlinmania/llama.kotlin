# Token Kernel Overview

**Component:** `kc_token_kernel.c`  
**Status:** ✅ Golden Path Implementation  
**Architecture:** Zero-Spin, Event-Driven Rendezvous

## Purpose

The **Token Kernel** is kcoro_arena's core synchronization primitive. It provides **lock-free, zero-spin rendezvous** between producers and consumers using atomic state machines and callback-driven wakeups.

## Why Token Kernel?

Traditional channel implementations poll (spin-wait) or use condition variables with OS-level blocking. The Token Kernel eliminates both:

- **No spinning:** Workers don't poll for work; they suspend until notified
- **No OS blocking:** All wakeups happen via lightweight callback dispatch
- **No locks:** All state transitions use atomic compare-and-swap (CAS)

This architecture is inspired by **BizTalk MessageBox** patterns:
- Each ticket is a "subscription slot"
- Senders/receivers register "waiter tokens" (like BizTalk's durable subscriptions)
- Matches trigger callback invocations (like BizTalk's message routing)

## Core Concepts

### The Rendezvous Cell

Each communication "ticket" maps to a **RendezvousCell** with an atomic state machine:

```c
typedef struct {
    _Atomic(int) state;       // EMPTY, SENDER_READY, RECEIVER_READY, MATCHED
    void* payload;            // Data pointer (copy or zero-copy descriptor)
    size_t payload_len;       // Bytes
    void* sender_cb_arg;      // Opaque callback context (sender side)
    void* receiver_cb_arg;    // Opaque callback context (receiver side)
    send_callback_t send_cb;  // Function to invoke on match (sender)
    recv_callback_t recv_cb;  // Function to invoke on match (receiver)
} rendezvous_cell_t;
```

### State Machine

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

**Key Property:** Only **one** sender and **one** receiver can match per cycle. Atomic CAS operations enforce this.

### Callback Model

When a match occurs (sender ↔ receiver rendezvous), the token kernel invokes **both callbacks immediately** without resuming coroutines first. This allows:

- Data copying/transfer to happen in-place
- Zero-copy descriptor handoff without extra scheduling overhead
- Metrics updates (match counters, latency tracking)

After callbacks complete, the coroutines are enqueued as **ready** in the scheduler.

## API

### Send

```c
int kc_token_send(
    int ticket,                    // Which cell to send on
    void* payload,                 // Data to send
    size_t len,                    // Payload size
    send_callback_t callback,      // Called on match
    void* cb_arg                   // Opaque context for callback
);
```

**Behavior:**

- If a receiver is waiting → instant match, callbacks fire, returns `0`
- If cell empty → register sender token, suspend (returns `KORO_SUSPEND`)
- If another sender present → returns `-EBUSY` (caller retries)

### Receive

```c
int kc_token_recv(
    int ticket,
    void** out_payload,            // Where to store received data pointer
    size_t* out_len,               // Where to store received length
    recv_callback_t callback,
    void* cb_arg
);
```

**Behavior:**

- If a sender is waiting → instant match, callbacks fire, returns `0`
- If cell empty → register receiver token, suspend (returns `KORO_SUSPEND`)
- If another receiver present → returns `-EBUSY` (caller retries)

## Internal Implementation

### Registration (Sender Example)

```c
int kc_token_send(int ticket, void* payload, size_t len, send_callback_t cb, void* arg) {
    rendezvous_cell_t* cell = &arena->cells[ticket];
    
    // Try immediate match with waiting receiver
    int expected = RECEIVER_READY;
    if (atomic_compare_exchange_strong(&cell->state, &expected, MATCHED)) {
        // Success! We matched with a waiting receiver
        
        // Transfer payload
        cell->payload = payload;
        cell->payload_len = len;
        
        // Invoke callbacks (receiver first, then sender)
        if (cell->recv_cb) {
            cell->recv_cb(cell->receiver_cb_arg, payload, len);
        }
        if (cb) {
            cb(arg, 0);  // 0 = success
        }
        
        // Enqueue both coroutines as ready
        koro_sched_enqueue_ready(cell->receiver_coro);
        koro_sched_enqueue_ready(cell->sender_coro);
        
        // Reset cell
        atomic_store(&cell->state, EMPTY);
        return 0;
    }
    
    // No receiver waiting; register ourselves
    expected = EMPTY;
    if (atomic_compare_exchange_strong(&cell->state, &expected, SENDER_READY)) {
        // Successfully registered
        cell->payload = payload;
        cell->payload_len = len;
        cell->send_cb = cb;
        cell->sender_cb_arg = arg;
        cell->sender_coro = current_coro();
        
        return KORO_SUSPEND;  // Caller must suspend
    }
    
    // Cell busy (another sender); caller must retry
    return -EBUSY;
}
```

### Match (Receiver Arrives)

When a receiver arrives and finds a sender:

```c
int kc_token_recv(int ticket, void** out, size_t* out_len, recv_callback_t cb, void* arg) {
    rendezvous_cell_t* cell = &arena->cells[ticket];
    
    int expected = SENDER_READY;
    if (atomic_compare_exchange_strong(&cell->state, &expected, MATCHED)) {
        // Matched!
        
        // Extract payload
        *out = cell->payload;
        *out_len = cell->payload_len;
        
        // Invoke callbacks
        if (cb) {
            cb(arg, cell->payload, cell->payload_len);
        }
        if (cell->send_cb) {
            cell->send_cb(cell->sender_cb_arg, 0);
        }
        
        // Enqueue coroutines
        koro_sched_enqueue_ready(cell->sender_coro);
        koro_sched_enqueue_ready(cell->receiver_coro);
        
        atomic_store(&cell->state, EMPTY);
        return 0;
    }
    
    // No sender; register ourselves
    expected = EMPTY;
    if (atomic_compare_exchange_strong(&cell->state, &expected, RECEIVER_READY)) {
        cell->recv_cb = cb;
        cell->receiver_cb_arg = arg;
        cell->receiver_coro = current_coro();
        return KORO_SUSPEND;
    }
    
    return -EBUSY;
}
```

## Zero-Spin Property

**Critical Insight:** There are no polling loops.

- Senders/receivers that suspend remain suspended until a match occurs
- When a match happens, the token kernel **immediately** enqueues both coroutines
- The scheduler only runs coroutines that are **ready** (no idle spinning)

This is achieved via the callback model:

```
Sender arrives → suspends → receiver arrives → callbacks fire → both enqueued
```

No worker threads scanning cells. No condition variable waits. Pure event-driven dispatch.

## Comparison to Other Primitives

### Go Channels

- **Similarities:** Rendezvous semantics, blocking send/recv
- **Differences:** Go uses OS-level goroutine parking (futex/condvar); kcoro_arena uses pure callbacks

### Kotlin Channels

- **Similarities:** Suspending functions, continuation-based
- **Differences:** Kotlin's channels use internal queues and locks; kcoro_arena is lock-free

### Traditional Condition Variables

```c
// Traditional approach
pthread_mutex_lock(&mu);
while (!data_ready) {
    pthread_cond_wait(&cv, &mu);  // OS blocks
}
process(data);
pthread_mutex_unlock(&mu);
```

**Token Kernel equivalent:**

```c
KORO_RECV(k, ticket, &data);  // Suspend (no OS block)
// Later: callback fires, coroutine enqueued
process(data);  // Resume here
```

**Advantages:**
- No syscalls
- No lock contention
- Cache-friendly (single atomic per operation)

## Performance

- **Match latency:** ~50-100ns (two atomic CAS + callbacks)
- **Suspend overhead:** ~50ns (store continuation, return `NULL`)
- **Resume overhead:** ~50ns (scheduler dequeue + callback invoke)

**Total send→recv roundtrip:** ~200ns

Compare to:
- pthread condition variable: ~2-5µs (syscall overhead)
- Kotlin Channel (JVM): ~500ns (lock + queue operations)
- Go channel: ~1µs (goroutine park/unpark)

## Limitations

1. **Single slot per ticket:** No buffering; always rendezvous
2. **Fixed ticket count:** Arena size determined at initialization
3. **No priority:** FIFO within state (sender-arrival order)

## Files

- **`kc_token_kernel.c`**: Core rendezvous logic
- **`kc_token_kernel.h`**: Public API
- **`kc_token_kernel_internal.h`**: Cell state machine definitions

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Detailed state machine diagrams
- [PERFORMANCE.md](./PERFORMANCE.md) - Latency benchmarks
- [../channels/OVERVIEW.md](../channels/OVERVIEW.md) - Higher-level channel API
- [../arena/OVERVIEW.md](../arena/OVERVIEW.md) - Memory management
