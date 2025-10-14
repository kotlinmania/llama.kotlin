# Stackless Scheduler

## Purpose

The scheduler is an **event-driven dispatch loop** that executes stackless coroutines (continuations) on a single thread. It replaces the traditional OS thread scheduler with a lightweight, cooperative scheduler optimized for I/O-bound tasks.

## Core Loop

```c
void koro_run() {
    while (1) {
        struct koro_cont* k = dequeue_next_runnable();
        if (!k) {
            // No runnable coroutines; check for pending events
            if (no_pending_events()) break;  // Exit
            wait_for_event();  // Block until I/O ready
            continue;
        }

        // Dispatch the continuation
        k->flags |= KORO_RUNNING;
        k->next_step(k);  // Execute one step
        k->flags &= ~KORO_RUNNING;

        // If still runnable and not completed, re-enqueue
        if (!(k->flags & KORO_COMPLETED) && !(k->flags & KORO_SUSPENDED)) {
            enqueue_runnable(k);
        } else if (k->flags & KORO_COMPLETED) {
            free_continuation(k);
        }
    }
}
```

## Key Concepts

### 1. Single Thread

Unlike traditional OS threads, all coroutines execute on **one thread**. There is no parallelism—only concurrency (interleaving). This eliminates:

- Race conditions (no need for locks on coroutine state)
- Context switch overhead (just a function call)
- Cache invalidation (all state stays in L1/L2)

### 2. Cooperative Scheduling

Coroutines voluntarily yield control via explicit suspension points:

```c
KORO_SEND(k, ticket, data);  // Suspends here
```

There is no preemption. A coroutine runs until it explicitly returns to the scheduler. This requires well-behaved code that doesn't hog the CPU.

### 3. Event-Driven

The scheduler doesn't spin or poll. It blocks on an event notification mechanism (e.g., `epoll`, `kqueue`, or in our case, the arena's callback system).

When a coroutine suspends waiting for a channel operation:

1. The token kernel places a waiter in a cell
2. The scheduler moves to the next coroutine
3. When the match completes, a **callback** enqueues the continuation back into the ready queue

Zero CPU usage while waiting.

## Ready Queue

The ready queue is a simple FIFO queue of continuations:

```c
struct koro_queue {
    struct koro_cont* head;
    struct koro_cont* tail;
};

void enqueue_runnable(struct koro_cont* k) {
    k->next = NULL;
    if (queue.tail) {
        queue.tail->next = k;
    } else {
        queue.head = k;
    }
    queue.tail = k;
}

struct koro_cont* dequeue_next_runnable() {
    if (!queue.head) return NULL;
    struct koro_cont* k = queue.head;
    queue.head = k->next;
    if (!queue.head) queue.tail = NULL;
    return k;
}
```

In practice the arena runtime must accept wakeups from the token kernel's
background worker thread, so the production implementation wraps this queue
with a `pthread_mutex_t`/`pthread_cond_t` pair. Callbacks safely append ready
continuations, signal the condition variable, and the main `koro_run()` loop
dequeues under the same lock. The cooperative semantics stay the same—the
mutex only protects the queue data structure.

### Managed vs. manual continuations

The scheduler now recognises two kinds of continuations:

| Launch path                        | `managed` flag | Lifetime owner        | Typical use                            |
|------------------------------------|---------------|-----------------------|----------------------------------------|
| `koro_go(step, arg, local_size)`   | `1`           | Scheduler             | Fire-and-forget tasks                  |
| `koro_cont_create(...)` + enqueue | `0`           | Caller                | Reusable continuations / custom pools  |

- `koro_go()` calls `koro_cont_create`, marks the resulting continuation as **managed**, and enqueues it. When the continuation reports completion, the scheduler automatically calls `koro_cont_destroy()` on your behalf.
- If you need to recycle continuations or add custom metadata, create them yourself with `koro_cont_create(...)`, enqueue via `koro_sched_enqueue_ready(...)`, and destroy (or reuse) them explicitly once they finish.

Example:

```c
/* Managed coroutine: scheduler owns lifetime */
static void* worker_step(koro_cont_t* k) {
    KORO_BEGIN(k);
    do_work();
    KORO_END(k);
}

void launch_worker(void) {
    koro_sched_init();
    koro_go(worker_step, NULL, 0);  /* auto destroyed when done */
    koro_run();
}

/* Manual coroutine: caller recycles object */
static void* manual_step(koro_cont_t* k) {
    struct state* st = k->user_data;
    KORO_BEGIN(k);
    process(st);
    KORO_END(k);
}

void reuse_manual(struct state_pool* pool) {
    koro_cont_t* c = state_pool_checkout(pool);
    if (!c) {
        c = koro_cont_create(manual_step, NULL, sizeof(struct state));
    }
    init_state((struct state*)c->user_data);
    koro_sched_enqueue_ready(c);
    koro_run();
    /* manual: return to pool instead of destroying */
    state_pool_return(pool, c);
}
```

## Event Notification (The Zero-Spin Magic)

Instead of having worker threads continuously scan the arena cells, the stackless scheduler uses **event callbacks**.

### Old Model (Spinning Workers)

```c
// Worker thread (constant CPU usage)
while (1) {
    for (int ticket = start; ticket < end; ticket++) {
        if (cell_is_matchable(ticket)) {
            try_match(ticket);
        }
    }
    yield();  // Spin forever, even with no work
}
```

### New Model (Event-Driven)

```c
// Token kernel (no spinning)
int koro_send(struct koro_cont* k, int ticket, void* data) {
    if (try_immediate_match(ticket, data)) {
        return 0;  // Success without suspension
    }

    // No match; suspend
    place_sender_waiter(ticket, k);
    k->flags |= KORO_SUSPENDED;
    return SUSPEND;  // Scheduler moves to next coroutine
}

// Later, when a receiver arrives:
void on_receiver_arrives(int ticket) {
    struct koro_cont* sender = pop_sender_waiter(ticket);
    if (sender) {
        sender->flags &= ~KORO_SUSPENDED;
        enqueue_runnable(sender);  // Wake it up!
    }
}
```

The key: **no CPU usage between events**. The scheduler blocks on the ready
condition variable—`koro_run()` sleeps when the queue is empty and no
continuations are active, and callbacks wake it up by signalling the condvar.
`koro_go()` marks continuations as managed so the scheduler can automatically
destroy them once they report completion, leaving manually managed continuations
to callers that still want direct control.

## Example Execution Trace

```
Time | Event                                    | Ready Queue
-----|------------------------------------------|------------------
0    | koro_run() starts                        | []
1    | Producer spawned                         | [producer]
2    | Consumer spawned                         | [producer, consumer]
3    | Dispatch producer                        | [consumer]
4    | Producer: koro_send(ticket=1, data=0)   | [consumer]
5    |   -> No receiver yet, suspend            | [consumer]
6    | Dispatch consumer                        | []
7    | Consumer: koro_recv(ticket=1)            | []
8    |   -> Finds sender! Match!                | []
9    |   -> on_sender_matched() enqueues both   | [consumer, producer]
10   | Dispatch consumer (resumes with data=0)  | [producer]
11   | Consumer: prints "received 0"            | [producer]
12   | Dispatch producer                        | []
13   | Producer: koro_send(ticket=1, data=1)    | []
     | ... and so on ...
```

Notice:
- No spinning loops
- Coroutines only run when they have actual work
- The scheduler is idle (blocked in `wait_for_event`) when all coroutines are suspended

## Comparison to BizTalk

| BizTalk MessageBox | kcoro_arena Scheduler |
|--------------------|-----------------------|
| SQL Server database | In-memory arena cells |
| Subscription table | Waiter tokens in cells |
| Message arrival triggers `bts_FindSubscriptions` | Match completion triggers callback → enqueue |
| Orchestration hibernation | Continuation suspension (`KORO_SUSPENDED`) |
| Rehydration from DB | Continuation enqueue → dispatch |
| SQL Agent jobs (polling) | **Replaced by event callbacks** |

The key innovation: BizTalk still uses polling (SQL Agent jobs). We use pure push notifications.

## Scheduler API

### User-Facing

```c
void koro_run();              // Main event loop (blocks until all done)
void koro_go(void (*fn)(struct koro_cont*, void*), void* arg);  // Spawn a coroutine
void koro_yield();            // Voluntarily yield (re-enqueue self)
void koro_exit();             // Mark self as completed
```

### Internal (used by token kernel)

```c
void enqueue_runnable(struct koro_cont* k);  // Wake a suspended coroutine
void mark_suspended(struct koro_cont* k);    // Remove from ready queue
```

## Performance Characteristics

### Dispatch Cost

- **Stackful context switch**: ~100-200 CPU cycles (save/restore registers, TLB flush)
- **Stackless dispatch**: ~5-10 cycles (function pointer call)

~20x faster per dispatch.

### Memory Overhead

- **Stackful**: 64 KB per coroutine (stack)
- **Stackless**: 200 bytes per continuation

~300x more memory efficient.

### Idle CPU Usage

- **Spinning workers**: 100% CPU (even with no work)
- **Event-driven scheduler**: 0% CPU (blocks in `wait_for_event`)

Infinite improvement when idle.

## Limitations

### Single-Threaded

All coroutines execute on one thread. For CPU-bound tasks, this is a bottleneck. Solution: use a thread pool with multiple schedulers (future work).

### No Preemption

A misbehaving coroutine can starve others:

```c
void bad_coroutine(struct koro_cont* k) {
    while (1) {
        compute_forever();  // Never suspends!
    }
}
```

Mitigation: enforce suspension points in loops (via macros).

### Function Color

Any function called from a stackless coroutine must be:
- Non-blocking (no `read()`, `sleep()`, etc.)
- Or written in CPS (transformed into steps)

This is acceptable for arena I/O (all async).

## Next Steps

- Read [MACROS.md](./MACROS.md) for how users write coroutines without manual state machines
- Read [../token_kernel/SPECIFICATION.md](../token_kernel/SPECIFICATION.md) for how matches trigger callbacks
- Read [../channels/INTEGRATION.md](../channels/INTEGRATION.md) for how high-level channel API uses the scheduler

## References

- [libuv event loop](https://docs.libuv.org/en/v1.x/design.html): Similar event-driven model
- [Node.js event loop](https://nodejs.org/en/docs/guides/event-loop-timers-and-nexttick/): User-space cooperative scheduler
- [Erlang scheduler](https://www.erlang.org/doc/reference_manual/processes.html): Preemptive, but conceptually similar (lightweight process dispatch)
