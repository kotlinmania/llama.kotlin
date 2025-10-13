# FFI Go Dispatch: C Function to Coroutine Bridge

## Purpose

The FFI Go Dispatch component provides a clean C-compatible API that allows standard C functions to participate in the kcoro_arena coroutine runtime. It bridges the gap between traditional procedural C code and the suspend/resume semantics of coroutines.

## Core Concept

Since C has no native coroutine syntax, we treat standard C function pointers as coroutine **bodies**. The dispatch system:

1. Wraps the C function in a coroutine context
2. Allocates a dedicated stack for it
3. Creates a continuation that can suspend/resume the function
4. Schedules it on the kcoro runtime

When the C function calls suspension points like `koro_send()` or `koro_recv()`, the library:
- Captures the current execution state
- Parks the coroutine (like BizTalk hibernation)
- Stores the continuation in a RendezvousCell
- Resumes execution later when matched

## Architecture

```
┌─────────────────────┐
│   C Application     │
│   (main.c)          │
│                     │
│   koro_init()       │
│   koro_go(func)     │──┐
│   koro_shutdown()   │  │
└─────────────────────┘  │
                         │
                         ▼
┌─────────────────────────────────────┐
│  FFI Go Dispatch (kcoro_c.h)        │
│                                     │
│  - Allocate stack                   │
│  - Wrap C func in continuation      │
│  - Schedule on runtime              │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  kcoro Runtime & Arena              │
│                                     │
│  - Worker coroutines                │
│  - RendezvousCells                  │
│  - Suspension/resumption            │
└─────────────────────────────────────┘
```

## API Surface

### Core Functions

```c
// Initialize the kcoro_arena runtime and workers
void koro_init(void);

// Launch a C function as a new coroutine
// func: standard C function pointer (void (*)(void*))
// arg: opaque context pointer passed to func
void koro_go(void (*func)(void*), void* arg);

// Suspend until receiver matches (sender side)
// ticket: RendezvousCell index
// payload: data to transfer (cast to void*)
void koro_send(int ticket, void* payload);

// Suspend until sender matches (receiver side)
// ticket: RendezvousCell index
// returns: payload sent by matched sender
void* koro_recv(int ticket);

// Gracefully shut down runtime and join workers
void koro_shutdown(void);
```

### Optional Advanced API

```c
// Try non-blocking send
// returns: 1 on success, 0 if no receiver ready
int koro_try_send(int ticket, void* payload);

// Try non-blocking recv
// returns: payload pointer or NULL if no sender ready
void* koro_try_recv(int ticket);

// Check if current execution is in a coroutine context
int koro_in_coroutine(void);

// Get current coroutine ID (for debugging)
uint64_t koro_current_id(void);
```

## Implementation Strategy

### Stack Allocation

Each dispatched C function needs its own stack:

```c
#define CORO_STACK_SIZE (64 * 1024) // 64 KB default

typedef struct {
    void* stack_base;
    size_t stack_size;
    void (*entry_func)(void*);
    void* entry_arg;
    kcoro_t* coro;           // kcoro runtime handle
    int state;               // INIT, RUNNING, SUSPENDED, DONE
} c_coroutine_ctx;
```

### Wrapping the C Function

The dispatch system creates a wrapper coroutine that:
1. Sets up the C function's stack frame
2. Calls the function pointer
3. Handles cleanup when function returns

Pseudo-implementation:

```c
static void c_func_wrapper(void* ctx_ptr) {
    c_coroutine_ctx* ctx = (c_coroutine_ctx*)ctx_ptr;
    
    // Mark as running
    ctx->state = RUNNING;
    
    // Invoke the user's C function with their arg
    ctx->entry_func(ctx->entry_arg);
    
    // Mark as done
    ctx->state = DONE;
    
    // Cleanup (deallocate stack, remove from runtime)
    cleanup_c_coroutine(ctx);
}

void koro_go(void (*func)(void*), void* arg) {
    c_coroutine_ctx* ctx = malloc(sizeof(c_coroutine_ctx));
    ctx->stack_base = malloc(CORO_STACK_SIZE);
    ctx->stack_size = CORO_STACK_SIZE;
    ctx->entry_func = func;
    ctx->entry_arg = arg;
    ctx->state = INIT;
    
    // Create kcoro coroutine with the wrapper
    ctx->coro = kcoro_new(c_func_wrapper, ctx, ctx->stack_base, ctx->stack_size);
    
    // Schedule on runtime ready queue
    kc_sched_spawn(global_scheduler, &ctx->coro, NULL);
}
```

### Suspension Points

When a C function calls `koro_send()` or `koro_recv()`, we:

1. Obtain the current coroutine context via TLS (thread-local storage)
2. Create a WaiterToken containing the continuation
3. Place token in the RendezvousCell
4. Call `kcoro_park()` to suspend

Example for `koro_send`:

```c
void koro_send(int ticket, void* payload) {
    // Get current coroutine (stored in TLS by scheduler)
    kcoro_t* current = kcoro_current();
    if (!current) {
        // Not in coroutine context; error
        abort();
    }
    
    // Use existing kc_chan_send logic but adapted for arena
    kc_arena_send(global_arena, ticket, payload, -1); // infinite wait
    // When resumed, control returns here and function continues
}
```

## Integration with Existing kcoro

### Changes Required in Core

1. **TLS for current coroutine**:
   ```c
   // In kcoro_core.c
   _Thread_local kcoro_t* tls_current_coro = NULL;
   
   kcoro_t* kcoro_current(void) {
       return tls_current_coro;
   }
   ```

2. **Set TLS in scheduler before dispatch**:
   ```c
   // In kc_sched.c worker loop
   static void worker_loop(sched_worker_t* w) {
       while (...) {
           kcoro_t* co = rq_pop_locked(w);
           if (co) {
               tls_current_coro = co;  // ← NEW
               kcoro_switch(w->main_co, co);
               tls_current_coro = NULL; // ← NEW
           }
       }
   }
   ```

3. **Adapt arena send/recv to work with coroutines**:
   - Current `kc_chan_send` already suspends via `kcoro_park()`
   - Arena version uses same suspension logic, but operates on RendezvousCells instead of a slot

### No Breaking Changes

- Existing coroutine-based code (e.g., orchestrations in kcoro) is unaffected
- FFI dispatch is purely additive
- C functions and native coroutines can interoperate seamlessly via the same arena

## BizTalk Parallel

This FFI dispatch is conceptually identical to how BizTalk allows external .NET components to participate in orchestrations:

| BizTalk Concept                     | kcoro_arena Equivalent               |
|-------------------------------------|--------------------------------------|
| `.NET` class instantiated in orch  | C function wrapped as coroutine      |
| Class marked `[Serializable]`       | C function state on dedicated stack  |
| Orchestration calls method          | Coroutine calls `koro_send/recv`     |
| Method blocks → orch dehydrates     | C func blocks → coroutine parks      |
| MessageBox stores state blob        | WaiterToken in RendezvousCell        |
| Message arrives → rehydrate         | Match found → resume continuation    |

In both systems, the external code doesn't manage the async lifecycle—it just calls blocking APIs and the runtime handles suspension/resumption transparently.

## Example Usage

```c
#include "kcoro_c.h"
#include <stdio.h>

void producer(void* arg) {
    int ticket = (int)(intptr_t)arg;
    for (int i = 0; i < 5; i++) {
        printf("Producer sending: %d\n", i);
        koro_send(ticket, (void*)(intptr_t)i);
    }
    printf("Producer done\n");
}

void consumer(void* arg) {
    int ticket = (int)(intptr_t)arg;
    for (int i = 0; i < 5; i++) {
        void* val = koro_recv(ticket);
        printf("Consumer received: %d\n", (int)(intptr_t)val);
    }
    printf("Consumer done\n");
}

int main() {
    koro_init();
    
    int ticket = 42;
    koro_go(producer, (void*)(intptr_t)ticket);
    koro_go(consumer, (void*)(intptr_t)ticket);
    
    // Let them run (in production, this would be an event loop)
    sleep(2);
    
    koro_shutdown();
    return 0;
}
```

Output:
```
Producer sending: 0
Consumer received: 0
Producer sending: 1
Consumer received: 1
...
Producer done
Consumer done
```

## Status

**Not yet implemented.** This document describes the design. Current gaps:

1. No `koro_go()` function in kcoro_c.h
2. No TLS tracking of current coroutine
3. Arena send/recv exist but need C-compatible wrappers
4. No stack allocator for C function contexts

Next steps:
1. Add TLS support to kcoro_core
2. Implement c_coroutine_ctx wrapper and stack allocation
3. Create kcoro_c.h with FFI API
4. Write unit tests using pure C code
5. Document performance characteristics (stack overhead, context switch cost)

## See Also

- [Arena Architecture](../arena/OVERVIEW.md)
- [WaiterToken State Machine](../tokens/STATE_MACHINE.md)
- [Continuation Mechanics](../callbacks/CONTINUATION.md)
