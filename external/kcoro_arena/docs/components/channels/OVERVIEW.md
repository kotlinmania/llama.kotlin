# Channels Overview

**Component:** `kc_chan.c`  
**Status:** ✅ Golden Path Implementation  
**Built On:** Token Kernel + Stackless Runtime

## Purpose

Channels provide the high-level, user-facing API for asynchronous communication in kcoro_arena. They wrap the low-level token kernel with ergonomic send/receive semantics that integrate seamlessly with stackless coroutines.

## Design Philosophy

Channels in kcoro_arena follow these principles:

1. **Token-kernel backed**: Every channel is a thin wrapper around a token kernel ticket
2. **Zero-copy capable**: Channels support both copy and zero-copy (descriptor-based) data transfer
3. **Stackless-native**: All operations are non-blocking suspension points compatible with CPS
4. **BizTalk-inspired**: Channels act like correlation subscriptions, not traditional queues

## Channel Types

### Copy Channel

```c
kc_chan_t* ch = kc_chan_create_copy(sizeof(int));
```

- **Data transfer**: Payload is copied into token kernel cell
- **Use case**: Small messages (<1KB), simple types
- **Performance**: Single memcpy, no ref-counting overhead

### Zero-Copy Channel

```c
kc_chan_t* ch = kc_chan_create_zdesc();
```

- **Data transfer**: Only descriptor ID is transferred
- **Use case**: Large messages, shared buffers, avoiding copies
- **Performance**: O(1) transfer, ref-counted descriptors

## Core API

### Send

```c
// Copy mode
int kc_chan_send(kc_chan_t* ch, const void* data, size_t len);

// Zero-copy mode
int kc_chan_send_zdesc(kc_chan_t* ch, kc_desc_id_t desc_id, size_t len);
```

**Semantics:**
- **Non-blocking**: Returns immediately if receiver is waiting
- **Suspending**: If no receiver, suspends via callback registration
- **Zero-spin**: Uses token kernel event callbacks, not polling

**Integration with Stackless:**

```c
koro_cont_t* producer(koro_cont_t* k) {
    KORO_BEGIN(k);
    
    KORO_SEND(k, channel_ticket, &data);  // Macro wraps kc_chan_send_cps
    
    KORO_END(k);
}
```

### Receive

```c
// Copy mode
int kc_chan_recv(kc_chan_t* ch, void* out_buf, size_t buf_len);

// Zero-copy mode
int kc_chan_recv_zdesc(kc_chan_t* ch, kc_desc_id_t* out_desc_id, size_t* out_len);
```

**Semantics:**
- **Non-blocking**: Returns immediately if sender is ready
- **Suspending**: If no sender, suspends via callback
- **Direct handoff**: Receiver extracts payload directly from matched sender's cell

**Integration with Stackless:**

```c
koro_cont_t* consumer(koro_cont_t* k) {
    struct my_state* state = k->user_data;
    
    KORO_BEGIN(k);
    
    KORO_RECV(k, channel_ticket, &state->received_data);
    
    KORO_END(k);
}
```

### Close

```c
int kc_chan_close(kc_chan_t* ch);
```

**Semantics:**
- Wakes all pending senders/receivers with `EPIPE` error
- Prevents new send/recv operations
- Releases arena resources (descriptors, tickets)

## Internal Architecture

### Token Kernel Integration

Each channel wraps a **ticket** from the token kernel:

```
Channel send/recv
      ↓
Token kernel cell
      ↓
Atomic CAS state machine (EMPTY → SENDER_READY → MATCHED)
      ↓
Callback invoked
      ↓
Continuation re-enqueued in scheduler
```

The channel API provides:
- Type safety (copy vs zdesc)
- Error handling (EPIPE, EAGAIN)
- Metrics (bytes transferred, matches)
- Integration with high-level constructs (select, actors)

### Memory Safety

**Copy channels:**
- Payloads are copied into token kernel cells
- Lifetime managed by token kernel (freed on match/cancel)

**Zero-copy channels:**
- Only descriptor IDs are transferred
- Descriptors are ref-counted
- Receiver must call `kc_desc_release` when done

### Error Handling

| Error Code | Meaning |
|------------|---------|
| `0` | Success |
| `-EAGAIN` | Would block (non-blocking call) |
| `-EPIPE` | Channel closed |
| `-EINVAL` | Invalid argument |
| `-ENOMEM` | Out of memory |

## Comparison to Traditional Channels

### Go Channels

```go
ch := make(chan int)
ch <- 42        // May block (stackful)
val := <-ch     // May block (stackful)
```

**Differences:**
- Go uses stack-switching for blocking
- kcoro_arena uses explicit suspension points
- Go channels buffer; kcoro_arena channels are rendezvous-only

### Kotlin Channels

```kotlin
val ch = Channel<Int>()
ch.send(42)     // suspend fun
val v = ch.receive()  // suspend fun
```

**Similarities:**
- Both use continuation-passing under the hood
- Both support zero-copy via platform-specific optimizations
- Both are cancel-safe

**Differences:**
- Kotlin compiles `suspend` to CPS automatically
- kcoro_arena requires explicit `KORO_SEND`/`KORO_RECV` macros
- Kotlin has buffered variants; kcoro_arena is rendezvous-only

## Performance Characteristics

- **Send latency**: ~50-100ns (hot path, receiver ready)
- **Recv latency**: ~50-100ns (hot path, sender ready)
- **Suspended send**: ~500ns (callback registration + token kernel CAS)
- **Rehydration**: ~200ns (callback invoke + scheduler enqueue)

**Zero-spin guarantee:** No busy-waiting loops. All waits are event-driven.

## Files

- **`kc_chan.c`**: Core send/receive implementation
- **`kc_chan.h`**: Public channel API
- **`kc_chan_internal.h`**: Private structures (cell wrappers, metrics)

## See Also

- [API.md](./API.md) - Detailed API reference
- [INTEGRATION.md](./INTEGRATION.md) - How channels use token kernel
- [../token_kernel/OVERVIEW.md](../token_kernel/OVERVIEW.md) - Underlying rendezvous mechanism
- [../stackless_runtime/OVERVIEW.md](../stackless_runtime/OVERVIEW.md) - Suspension/resumption model
