# Stackless Runtime Overview

## Purpose

The stackless runtime provides **zero-stack** coroutines using **Continuation-Passing Style (CPS)** transformation. Unlike traditional stackful coroutines (which allocate separate stacks), stackless coroutines store their execution state in heap-allocated continuation records and execute entirely on the scheduler's thread stack.

## Why Stackless?

### Memory Efficiency

- **Stackful**: Each coroutine needs 64 KB–1 MB of stack space (Linux default: 64 KB per stack)
  - 1,000 coroutines = 64 MB minimum
  - 100,000 coroutines = 6.4 GB (impossible)

- **Stackless**: Each coroutine is a ~200-byte continuation record
  - 1,000 coroutines = 200 KB
  - 100,000 coroutines = 20 MB (trivial)

### Portability

- No assembly stack-switching required (beyond a simple trampoline)
- Works on any platform with standard C11
- No guard pages, no mmap, no platform-specific memory management

### Cache Locality

- All coroutine state fits in L1/L2 cache
- Scheduler dispatch is a simple function call (not context switch)
- Better CPU utilization due to fewer cache misses

## Core Concepts

### 1. Continuation Record

A continuation is a heap-allocated struct containing:

```c
struct koro_cont {
    koro_step_fn next_step;  // Function pointer to next "step"
    int state;                // Current execution state (for switch)
    void* user_data;          // User's local variables
    // ... metadata ...
};
```

Think of it as a manually managed stack frame.

### 2. Step Functions

A coroutine's logic is broken into "steps" (functions). Each step:

1. Executes some work
2. If it needs to suspend: updates `state`, returns `NULL`
3. If it can continue: calls the next step function directly

This is **Continuation-Passing Style**: instead of using the C call stack to track progress, we explicitly pass a continuation record forward.

### 3. The Scheduler Loop

The scheduler is a simple event loop:

```c
while (has_runnable_coroutines()) {
    struct koro_cont* k = dequeue_next_runnable();
    k->next_step(k);  // Just a function call!
}
```

No context switching, no assembly, no stack manipulation.

### 4. Suspension

When a coroutine wants to suspend (e.g., waiting for a channel):

```c
void my_step(struct koro_cont* k) {
    // ... do some work ...
    
    // Need to wait for data:
    k->next_step = my_resume_step;  // Set where to resume
    k->state = WAITING_FOR_DATA;
    return;  // <-- Suspend! Control returns to scheduler
}
```

The scheduler sees the `return` and moves to the next coroutine.

### 5. Resumption

When the data arrives, another coroutine (or I/O callback) enqueues this continuation:

```c
// Data is ready!
enqueue_runnable(k);
```

The scheduler will call `k->next_step(k)` again, and execution resumes exactly where it left off.

## Comparison to Stackful Model

| Aspect | Stackful (kcoro) | Stackless (kcoro_arena) |
|--------|------------------|-------------------------|
| Memory per coroutine | 64 KB – 1 MB | 200 bytes |
| Max practical coroutines | ~10,000 | 1,000,000+ |
| Suspension mechanism | Assembly context switch | C `return` statement |
| Portability | Assembly per arch | Pure C11 |
| Can call blocking functions | Yes | No (must be async) |
| Cache efficiency | Poor (stack scattered) | Excellent (all in L1) |

## Trade-offs

### Advantage: Memory + Speed

- Orders of magnitude more coroutines
- Faster suspension/resumption (no context switch)
- Better cache utilization

### Limitation: "Function Color"

- A stackless coroutine **cannot** call a blocking function (e.g., `read()`, `sleep()`)
- All I/O must be asynchronous
- Any function that suspends must be written in CPS

This is acceptable for `kcoro_arena` because:
- All I/O is via the arena (already async)
- The token kernel is explicitly designed for async handoff
- User code is wrapped in macros that handle CPS transformation

## Next steps

- Start with [CONTINUATION_GUIDE.md](./CONTINUATION_GUIDE.md) for a gentle walkthrough of writing a continuation.
- Dive deeper with [CONTINUATION_MODEL.md](./CONTINUATION_MODEL.md) if you need the exact struct layout.
- Read [SCHEDULER.md](./SCHEDULER.md) for the event-driven dispatch loop.
- Check [MACROS.md](./MACROS.md) to see the helper wrappers most user code relies on.

## References

- [Protothreads](http://dunkels.com/adam/pt/): Classic stackless coroutine pattern in C
- [CPS in Wikipedia](https://en.wikipedia.org/wiki/Continuation-passing_style)
- [Why Stackless Matters](https://without.boats/blog/why-async-rust/): Rust async/await rationale (applies here)
