# kcoro_arena Completion Roadmap

## Executive Summary

kcoro_arena is a **stackless, zero-spin, event-driven coroutine system** with an arena-backed token kernel for zero-copy messaging. This document details the precise technical tasks required to complete the implementation and achieve production readiness.

## Current Architecture Status

### ✅ Complete Components

1. **Token Kernel (`kc_token_kernel.c`)**
   - Hash-bucketed token storage (1024 buckets, empirically optimized)
   - Worker thread with event-driven ready queue
   - Send/receive token publication and matching
   - Callback-based resumption (zero-spin design)
   - Thread-safe freelist for token block recycling

2. **Stackless Primitives (`kcoro_stackless.c`)**
   - `koro_cont_t` continuation structure with state machine
   - CPS-transformed send/receive operations
   - Resume callbacks (`koro_send_resume_callback`, `koro_recv_resume_callback`)
   - Immediate-completion fast path (no suspension when match is ready)

3. **Scheduler (`koro_sched_stackless.c`)**
   - Ready queue for runnable continuations
   - Single-threaded execution loop (all coroutines share scheduler stack)
   - Continuation dispatch and state machine execution

4. **Arena Memory (`kc_arena.c`, `kc_desc.c`)**
   - Zero-copy descriptor system with reference counting
   - Alias-LRU cache for frequently used payloads
   - Page-aligned memory allocation for efficient DMA

### 🚧 Incomplete Components

The following tasks remain to achieve full end-to-end functionality:

---

## Task 1: Complete Event Callback Registration System

### Current State
- Token kernel has callback infrastructure (`resume_pc` field in `kc_token_block`)
- Stackless layer registers callbacks (`koro_send_resume_callback`, `koro_recv_resume_callback`)
- Worker thread invokes callbacks from ready queue

### Missing Pieces

#### 1.1 Payload Delivery Mechanism
**File:** `kc_token_kernel.c`

**Issue:** `kc_token_kernel_consume_payload()` is declared but not fully implemented.

**Required Implementation:**
```c
/* Thread-local storage for delivering payload from worker to continuation */
static _Thread_local kc_payload g_tls_payload;
static _Thread_local int g_tls_payload_valid = 0;

int kc_token_kernel_consume_payload(kc_payload* out)
{
    if (!g_tls_payload_valid) {
        return -EAGAIN; /* No payload available */
    }
    
    if (out) {
        *out = g_tls_payload;
    }
    
    g_tls_payload_valid = 0;
    return 0; /* Success */
}

/* Called by worker before invoking resume callback */
static void set_tls_payload(kc_payload payload)
{
    g_tls_payload = payload;
    g_tls_payload_valid = 1;
}
```

**Modification to Worker Loop:**
```c
static void* kc_token_kernel_worker(void* arg)
{
    while (!g_kernel.ready_queue.stop) {
        kc_token_block* blk = ready_dequeue(&g_kernel.ready_queue);
        if (!blk) continue;
        
        /* NEW: Set payload in TLS before callback */
        set_tls_payload(blk->payload);
        
        if (blk->resume_pc) {
            blk->resume_pc(); /* Invoke continuation's resume callback */
        }
        
        freelist_push(&g_kernel.freelist, blk);
    }
    return NULL;
}
```

**Success Criteria:**
- Continuations can retrieve delivered payload via `kc_token_kernel_consume_payload()`
- TLS ensures thread-safe delivery in single-threaded scheduler
- Worker clears TLS after each callback invocation

---

#### 1.2 Matching Logic Completion
**File:** `kc_token_kernel.c`

**Current State:** Token blocks are published but matching logic may not be complete.

**Required Implementation:**

Ensure `kc_token_kernel_publish_send()` and `kc_token_kernel_publish_receive()` implement the following matching state machine:

```c
kc_ticket kc_token_kernel_publish_send(struct kc_chan* ch, 
                                        void* data, 
                                        size_t len,
                                        void (*resume_pc)(void))
{
    /* 1. Allocate token block */
    kc_token_block* send_blk = allocate_token();
    send_blk->channel = ch;
    send_blk->payload.data = data;
    send_blk->payload.len = len;
    send_blk->resume_pc = resume_pc;
    send_blk->id = allocate_ticket_id();
    
    /* 2. Insert into hash bucket */
    insert_into_bucket(send_blk);
    
    /* 3. Check for immediate match with waiting receiver */
    kc_token_block* recv_blk = find_receiver_for_channel(ch);
    if (recv_blk) {
        /* Immediate match! */
        deliver_payload(recv_blk, send_blk->payload);
        ready_enqueue(&g_kernel.ready_queue, recv_blk); /* Wake receiver */
        ready_enqueue(&g_kernel.ready_queue, send_blk); /* Wake sender */
        return send_blk->id;
    }
    
    /* 4. No match; sender will wait until receiver arrives */
    return send_blk->id;
}
```

**Success Criteria:**
- Sends and receives match correctly across channels
- Immediate matches enqueue both tokens to ready queue
- Deferred matches occur when opposite token arrives later

---

## Task 2: Wire Up Full Send/Receive Test

### Objective
Create a complete end-to-end test that exercises:
- Token kernel publication
- Matching logic
- Callback invocation
- Scheduler execution
- Payload delivery

### Test File
**Path:** `external/kcoro_arena/tests/test_token_kernel_send_recv.c`

### Test Structure

```c
#include "kcoro_stackless.h"
#include "koro_sched_stackless.h"
#include "kcoro_token_kernel.h"
#include <assert.h>
#include <stdio.h>

/* User state for producer coroutine */
struct producer_state {
    int counter;
    struct kc_chan* channel;
};

/* Producer continuation step function */
void* producer_step(koro_cont_t* k)
{
    struct producer_state* s = (struct producer_state*)k->user_data;
    
    /* Protothreads-style state machine */
    switch (k->state) {
        case 0: /* Initial state */
            printf("Producer starting\n");
            s->counter = 0;
            k->state = 1;
            return (void*)1; /* Continue */
            
        case 1: /* Send loop */
            if (s->counter >= 5) {
                k->state = 2; /* Done */
                return (void*)1;
            }
            
            printf("Producer sending: %d\n", s->counter);
            
            /* Suspend if send blocks */
            void* result = koro_send_stackless(k, s->channel, 
                                               &s->counter, 
                                               sizeof(s->counter));
            if (!result) {
                /* Suspended; will be resumed by callback */
                return NULL;
            }
            
            /* Send completed immediately */
            s->counter++;
            k->state = 1; /* Loop */
            return (void*)1;
            
        case 2: /* Completion */
            printf("Producer done\n");
            k->completed = 1;
            return (void*)1;
    }
    
    return (void*)1;
}

/* Consumer state */
struct consumer_state {
    int received_count;
    struct kc_chan* channel;
};

/* Consumer continuation step function */
void* consumer_step(koro_cont_t* k)
{
    struct consumer_state* s = (struct consumer_state*)k->user_data;
    
    switch (k->state) {
        case 0: /* Initial */
            printf("Consumer starting\n");
            s->received_count = 0;
            k->state = 1;
            return (void*)1;
            
        case 1: /* Receive loop */
            if (s->received_count >= 5) {
                k->state = 2; /* Done */
                return (void*)1;
            }
            
            printf("Consumer waiting for data...\n");
            
            /* Suspend if receive blocks */
            void* result = koro_recv_stackless(k, s->channel);
            if (!result) {
                /* Suspended */
                return NULL;
            }
            
            /* Receive completed */
            int value = *(int*)k->arena_payload;
            printf("Consumer received: %d\n", value);
            s->received_count++;
            k->state = 1; /* Loop */
            return (void*)1;
            
        case 2: /* Completion */
            printf("Consumer done\n");
            k->completed = 1;
            return (void*)1;
    }
    
    return (void*)1;
}

int main()
{
    /* 1. Initialize token kernel */
    kc_token_kernel_init();
    
    /* 2. Create a channel (stub for now; will be arena-backed) */
    struct kc_chan* ch = create_test_channel();
    
    /* 3. Create producer continuation */
    struct producer_state* prod_state = calloc(1, sizeof(struct producer_state));
    prod_state->channel = ch;
    
    koro_cont_t* producer = koro_cont_create(producer_step, 
                                             prod_state, 
                                             sizeof(struct producer_state));
    
    /* 4. Create consumer continuation */
    struct consumer_state* cons_state = calloc(1, sizeof(struct consumer_state));
    cons_state->channel = ch;
    
    koro_cont_t* consumer = koro_cont_create(consumer_step, 
                                             cons_state, 
                                             sizeof(struct consumer_state));
    
    /* 5. Enqueue both in scheduler */
    koro_sched_enqueue_ready(producer);
    koro_sched_enqueue_ready(consumer);
    
    /* 6. Run scheduler until both complete */
    printf("Starting scheduler...\n");
    koro_sched_run();
    printf("Scheduler finished\n");
    
    /* 7. Verify results */
    assert(producer->completed);
    assert(consumer->completed);
    assert(cons_state->received_count == 5);
    
    /* 8. Cleanup */
    koro_cont_destroy(producer);
    koro_cont_destroy(consumer);
    kc_token_kernel_shutdown();
    
    printf("Test PASSED\n");
    return 0;
}
```

### Success Criteria
- Test compiles without errors
- Producer sends 5 messages
- Consumer receives all 5 messages
- Scheduler alternates execution correctly
- No spin loops (verified with CPU profiling)
- Test completes in < 1ms (excluding I/O)

---

## Task 3: Measure and Verify Zero-Spin Operation

### Objective
Prove that kcoro_arena achieves true zero-spin operation where:
- No coroutine burns CPU while waiting
- All suspensions yield to scheduler
- Worker thread only wakes on events (not polling)

### Measurement Tools

#### 3.1 CPU Profiling
**Tool:** `perf` on Linux / `Instruments` on macOS

**Command:**
```bash
# Linux
perf record -e cpu-clock ./test_token_kernel_send_recv
perf report

# macOS
instruments -t "Time Profiler" ./test_token_kernel_send_recv
```

**Expected Result:**
- Worker thread shows zero samples in busy-wait loops
- All CPU time attributed to:
  - Token kernel matching logic
  - Callback invocations
  - Scheduler dispatch

#### 3.2 Instrumentation Points
Add cycle counters to measure suspension latency:

**File:** `koro_sched_stackless.c`

```c
#include <time.h>

static uint64_t get_nanos()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void koro_sched_run()
{
    uint64_t total_suspensions = 0;
    uint64_t total_suspension_nanos = 0;
    
    while (has_work()) {
        koro_cont_t* k = dequeue_ready();
        
        uint64_t start = get_nanos();
        void* result = k->next_step(k); /* Execute step */
        uint64_t end = get_nanos();
        
        if (!result) {
            /* Suspension occurred */
            total_suspensions++;
            total_suspension_nanos += (end - start);
        }
    }
    
    printf("Average suspension latency: %lu ns\n", 
           total_suspensions ? total_suspension_nanos / total_suspensions : 0);
}
```

**Expected Result:**
- Average suspension latency < 500 ns
- Zero time spent in spin loops

#### 3.3 Verification Test
**File:** `external/kcoro_arena/tests/test_zero_spin_verification.c`

```c
#include <pthread.h>
#include <unistd.h>
#include <assert.h>

/* Monitor CPU usage of worker thread */
void* monitor_cpu(void* arg)
{
    pid_t worker_tid = *(pid_t*)arg;
    
    for (int i = 0; i < 10; i++) {
        /* Read /proc/<pid>/stat on Linux */
        /* Parse utime + stime fields */
        /* Assert CPU usage < 1% when no work available */
        usleep(100000); /* 100ms sample interval */
    }
    
    return NULL;
}

int main()
{
    kc_token_kernel_init();
    
    /* Get worker thread ID */
    pid_t worker_tid = get_worker_tid();
    
    /* Start monitoring */
    pthread_t monitor;
    pthread_create(&monitor, NULL, monitor_cpu, &worker_tid);
    
    /* Let worker idle for 1 second with no work */
    sleep(1);
    
    pthread_join(monitor, NULL);
    
    kc_token_kernel_shutdown();
    
    printf("Zero-spin verification PASSED\n");
    return 0;
}
```

---

## Task 4: Documentation Updates

### Files to Update

#### 4.1 Architecture Overview
**File:** `external/kcoro_arena/docs/components/architecture/OVERVIEW.md`

**Required Sections:**
1. Token Kernel Design
   - Hash-bucketed storage
   - Event-driven worker
   - Callback-based resumption
2. Stackless Model
   - Continuation-passing style
   - State machine execution
   - Payload delivery via TLS
3. Zero-Spin Guarantee
   - Worker blocks on condition variable
   - No polling loops
   - Immediate wakeup on token match

#### 4.2 API Documentation
**File:** `external/kcoro_arena/docs/components/ffi_go/API.md`

**Add Examples:**
- Complete producer/consumer with state machines
- Error handling patterns
- Cancellation support

#### 4.3 Performance Benchmarks
**File:** `external/kcoro_arena/docs/BENCHMARKS.md`

**Required Metrics:**
- Throughput (messages/sec)
- Latency (suspension → resumption time)
- Memory overhead per coroutine
- CPU usage under load vs. idle

---

## Task 5: Integration Testing

### Test Suite

#### 5.1 Basic Functionality
- [x] Token allocation/deallocation
- [ ] Send/receive matching
- [ ] Callback invocation
- [ ] Payload delivery

#### 5.2 Concurrency
- [ ] Multiple senders, one receiver
- [ ] One sender, multiple receivers
- [ ] Many-to-many communication
- [ ] Channel creation/destruction under load

#### 5.3 Edge Cases
- [ ] Send to closed channel
- [ ] Receive from closed channel
- [ ] Cancellation during suspension
- [ ] Token ID wraparound (after 2^64 ops)

#### 5.4 Performance
- [ ] 1M messages/sec sustained throughput
- [ ] < 100ns average suspension latency
- [ ] Zero CPU usage when idle
- [ ] < 1KB memory per suspended coroutine

---

## Success Metrics

### Completion Criteria
- ✅ All tests pass
- ✅ Zero-spin verified (< 0.1% CPU when idle)
- ✅ Throughput > 1M msg/sec on commodity hardware
- ✅ Memory < 1KB per coroutine
- ✅ Documentation complete
- ✅ No TODOs/FIXMEs in production code

### Comparison to kcoro (stackful)
| Metric | kcoro (stackful) | kcoro_arena (stackless) |
|--------|------------------|-------------------------|
| Stack per coroutine | 64 KB | 0 bytes |
| Max concurrent | ~10K | > 1M |
| Suspension latency | Variable (context switch) | < 100ns (callback) |
| Spin loops | Yes (in base) | No (event-driven) |
| Portability | Requires assembly | Pure C |

---

## Timeline

### Phase 1 (Current Session)
- [ ] Complete payload delivery (Task 1.1)
- [ ] Finish matching logic (Task 1.2)
- [ ] Wire up send/receive test (Task 2)

### Phase 2 (Next Session)
- [ ] Measure zero-spin (Task 3)
- [ ] Full test suite (Task 5)
- [ ] Documentation (Task 4)

### Phase 3 (Polish)
- [ ] Benchmarking
- [ ] Optimization
- [ ] Production hardening

---

## Next Immediate Actions

1. **Implement `kc_token_kernel_consume_payload()` with TLS**
2. **Add `set_tls_payload()` to worker loop**
3. **Write `test_token_kernel_send_recv.c`**
4. **Build and run test**
5. **Verify zero-spin with profiler**

Ready to proceed?
