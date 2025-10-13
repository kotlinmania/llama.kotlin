# Select Overview

**Component:** `kc_select.c`  
**Status:** ✅ Golden Path Implementation  
**Built On:** Token Kernel + Channel primitives

## Purpose

Select provides **multi-way synchronization** over multiple channels, enabling a coroutine to wait on several communication operations simultaneously and proceed with whichever completes first.

## Why Select?

Without select, a coroutine can only wait on a single channel at a time:

```c
// Deadlock risk: what if msg arrives on ch2 first?
KORO_RECV(k, ch1, &data);
process(data);
KORO_RECV(k, ch2, &data);
process(data);
```

With select, you can wait on both:

```c
KORO_SELECT_BEGIN(k, &sel);
    KORO_CASE_RECV(sel, ch1, &data) {
        process_ch1(data);
    }
    KORO_CASE_RECV(sel, ch2, &data) {
        process_ch2(data);
    }
KORO_SELECT_END(sel);
```

**First match wins**—the coroutine proceeds as soon as any clause is ready.

## Core Semantics

### Single-Winner Guarantee

Only **one clause** in a select can complete, even if multiple channels are ready simultaneously. This is enforced via an atomic claim operation:

```c
// Token kernel level
if (atomic_cas(&sel->claimed, 0, 1)) {
    // This clause won
    invoke_callback();
} else {
    // Another clause already won; skip this one
    unregister_and_cleanup();
}
```

### Deterministic Dispatch

When multiple clauses are ready:
- **Biased select** (default): First clause in source order wins
- **Fair select** (optional): Randomized selection

kcoro_arena uses **biased select** for deterministic debugging.

### Cancel-Safe

If the select operation is cancelled (coroutine cancelled, timeout):
- All registered clauses are atomically unregistered
- Pending payloads are released (zero-copy descriptors decremented)
- No ghost waiters remain in token kernel

## API

### Structure

```c
typedef struct kc_select {
    koro_cont_t* owner;        // Coroutine performing select
    int num_clauses;           // Number of CASE statements
    _Atomic(int) claimed;      // 0=open, 1=claimed by a clause
    int winner_index;          // Which clause won (-1 if none)
    void* winner_data;         // Received data (if any)
    size_t winner_len;         // Received data length
} kc_select_t;
```

### Usage Pattern

```c
koro_cont_t* handler(koro_cont_t* k) {
    struct handler_state* state = k->user_data;
    
    KORO_BEGIN(k);
    
    // Initialize select
    kc_select_t sel;
    kc_select_init(&sel, k);
    
    KORO_SELECT_BEGIN(k, &sel);
        KORO_CASE_RECV(sel, state->ch1_ticket, &state->buf1) {
            printf("Received on ch1: %d\n", state->buf1);
        }
        KORO_CASE_RECV(sel, state->ch2_ticket, &state->buf2) {
            printf("Received on ch2: %d\n", state->buf2);
        }
        KORO_CASE_SEND(sel, state->ch3_ticket, &state->out_data) {
            printf("Sent on ch3\n");
        }
    KORO_SELECT_END(sel);
    
    KORO_END(k);
}
```

### Timeout Support

```c
KORO_SELECT_BEGIN(k, &sel);
    KORO_CASE_RECV(sel, ch1, &data) { ... }
    KORO_CASE_TIMEOUT(sel, 1000) {  // 1 second
        printf("Timed out\n");
    }
KORO_SELECT_END(sel);
```

## Internal Implementation

### Registration Phase

When `KORO_SELECT_BEGIN` executes:

1. For each `KORO_CASE_RECV` or `KORO_CASE_SEND`:
   - Register with the corresponding channel's token kernel ticket
   - Store a callback that atomically claims the select

2. Suspend the coroutine (return `NULL`)

### Match Phase (in Token Kernel)

When a sender/receiver arrives at a token kernel cell:

```c
// In token kernel match logic
if (cell->waiter_is_select) {
    kc_select_t* sel = cell->select_ptr;
    
    if (atomic_cas(&sel->claimed, 0, 1)) {
        // This channel won the select
        sel->winner_index = cell->clause_index;
        sel->winner_data = extract_payload();
        
        // Wake the select's owner coroutine
        koro_sched_enqueue_ready(sel->owner);
    } else {
        // Select already claimed by another clause
        // Clean up this match attempt
        unregister_waiter(cell);
    }
}
```

### Resume Phase (in Scheduler)

When the coroutine resumes:

```c
KORO_SELECT_BEGIN(k, &sel);
    KORO_CASE_RECV(sel, ch1, &data) {
        // switch (sel.winner_index) case 0:
        if (sel->winner_index == 0) {
            data = sel->winner_data;
            // User code here
        }
    }
    KORO_CASE_RECV(sel, ch2, &data) {
        // case 1:
        if (sel->winner_index == 1) { ... }
    }
KORO_SELECT_END(sel);
```

The winning clause's block executes; others are skipped.

## Comparison to Other Systems

### Go Select

```go
select {
case v := <-ch1:
    process(v)
case ch2 <- data:
    // sent
default:
    // non-blocking
}
```

**Similarities:**
- Multi-way synchronization
- First-ready wins
- Cancel-safe

**Differences:**
- Go's select is a language primitive (compiled to runtime calls)
- kcoro_arena's select is macro-based (pure C)
- Go's select can have default (non-blocking); kcoro_arena requires explicit timeout

### Kotlin `select` Expression

```kotlin
select<Unit> {
    ch1.onReceive { value -> process(value) }
    ch2.onSend(data) { /* sent */ }
}
```

**Similarities:**
- Continuation-based (stackless)
- Atomic clause claiming
- Works with arbitrary suspending functions (not just channels)

**Differences:**
- Kotlin's select is type-safe and composable
- kcoro_arena's select is channel-specific

## Performance

- **Registration overhead**: ~100-200ns per clause
- **Match latency**: ~50ns (atomic CAS check)
- **Cancel overhead**: ~100ns per clause (unregister all)

**Zero-spin:** Select waits are event-driven. No polling loops.

## Limitations

1. **Fixed clause count**: Must be known at registration (no dynamic clause addition)
2. **Channel-only**: Cannot select on arbitrary coroutines (actor model requires separate abstraction)
3. **Biased by default**: Clause order matters (first-listed clauses have priority)

## Files

- **`kc_select.c`**: Core select implementation
- **`kc_select.h`**: Public select API
- **`kc_select_internal.h`**: Clause registration, claim logic

## See Also

- [API.md](./API.md) - Detailed API and examples
- [IMPLEMENTATION.md](./IMPLEMENTATION.md) - Single-winner logic internals
- [../channels/OVERVIEW.md](../channels/OVERVIEW.md) - Channel primitives
- [../token_kernel/OVERVIEW.md](../token_kernel/OVERVIEW.md) - Atomic match mechanism
