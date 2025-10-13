# Token Kernel Send/Receive Flow Analysis

## Overview

The token kernel implements a zero-spin, event-driven rendezvous mechanism for kcoro_arena channels. This document traces the complete flow of a send and receive operation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Stackless Code                       │
│  koro_send_stackless() → suspends → koro_send_resume_callback│
└────────────────┬───────────────────────────▲────────────────┘
                 │                            │
                 │ publish                    │ callback
                 ▼                            │
┌─────────────────────────────────────────────────────────────┐
│                     Token Kernel                             │
│  • Hash buckets (pending tokens)                            │
│  • Ready queue (matched tokens)                             │
│  • Worker thread (processes ready queue)                    │
└────────────────┬───────────────────────────▲────────────────┘
                 │                            │
                 │ register                   │ match notification
                 ▼                            │
┌─────────────────────────────────────────────────────────────┐
│                    Channel Layer                             │
│  • Pending send/recv lists                                  │
│  • Match logic                                              │
│  • Calls kc_token_kernel_callback() on match               │
└─────────────────────────────────────────────────────────────┘
```

## Complete Send/Receive Flow

### Step 1: User Initiates Send

```c
// In user's stackless coroutine function
void* result = koro_send_stackless(continuation, channel, data, data_len);
if (result == NULL) {
    return NULL; // Suspended, scheduler will resume us later
}
```

### Step 2: Publish Send Token

`koro_send_stackless()` → `kc_token_kernel_publish_send()`

At this point:
- Token is stored in hash bucket, waiting for match
- Coroutine checks for immediate completion (fast path)
- If not immediate, coroutine returns NULL (suspends)
- Scheduler moves to next runnable continuation

### Step 3: Receiver Arrives (in another coroutine)

```c
void* result = koro_recv_stackless(continuation, channel);
// Similar publish flow, but with kc_token_kernel_publish_recv()
```

### Step 4: Channel Match Logic

The channel code detects the match and calls:

```c
// In kc_chan.c fulfill_coroutine_recv()
kc_token_kernel_callback(recv_node->ticket, payload);
```

### Step 5: Token Kernel Callback Processing

`kc_token_kernel_callback()` → enqueue to ready queue → worker processes

### Step 6: Worker Thread Processes Token

The worker thread dequeues the ready token and invokes its resume callback.

### Step 7: Resume Callback Re-schedules Continuation

```c
static void koro_recv_resume_callback(void* user_data)
{
    koro_cont_t* k = (koro_cont_t*)user_data;
    
    // 1. Consume payload from token kernel
    kc_payload result;
    int rc = kc_token_kernel_consume_payload(&result);
    k->last_park_result = rc;
    
    if (rc == 0) {
        // 2. Store received data in continuation
        k->arena_payload = result.ptr;
        k->arena_payload_len = result.len;
        k->arena_desc_id = result.desc_id;
    }
    
    // 3. Re-enqueue continuation in scheduler's ready queue
    koro_sched_enqueue_ready(k);
}
```

### Step 8: Scheduler Resumes Continuation

The stackless scheduler picks up the continuation from its ready queue and calls its next_step function. The continuation resumes right after the `koro_recv_stackless()` call that suspended it.

## Key Design Properties

### Zero-Spin Operation

- **No polling loops**: Worker thread blocks on `pthread_cond_wait` when queue empty
- **No active scanning**: Tokens sit passively in hash buckets until matched
- **Event-driven wakeup**: Worker only runs when notified

### Memory Efficiency

- **Freelist allocation**: Token blocks recycled, not constantly malloc'd
- **Hash bucket organization**: O(1) token lookup by ID
- **Single worker thread**: Minimal context switching

### Fast Path Optimization

Both send/recv check for immediate completion before suspending.

## Comparison to BizTalk MessageBox

| BizTalk Component | kcoro_arena Equivalent |
|-------------------|------------------------|
| Subscription table (SQL) | Hash bucket array (in-memory) |
| Message queue (per host) | Ready queue (single worker) |
| Orchestration hibernation (disk) | Continuation suspend (heap) |
| bts_FindSubscriptions (stored proc) | bucket_remove() (O(1) hash lookup) |
| Rehydration (deserialize from DB) | Resume callback (invoke function pointer) |

The fundamental pattern is identical: publish → wait in durable store → match → resume.

## Current Implementation Status

✅ **Complete**:
- Token publish (send/recv)
- Hash bucket storage
- Ready queue + worker thread
- Callback invocation
- Freelist memory management
- Cancellation support
- Channel integration
- Stackless coroutine integration

🔲 **Not yet implemented**:
- Timeout support (deadline tokens)
- Advanced metrics/observability
- Multi-worker scaling

## Testing Requirements

To validate the token kernel:

1. **Basic send/receive match**
2. **Cancellation before match**
3. **Close during pending operation**
4. **Fast path (immediate completion)**
5. **Memory leak validation**
6. **Worker thread shutdown**
