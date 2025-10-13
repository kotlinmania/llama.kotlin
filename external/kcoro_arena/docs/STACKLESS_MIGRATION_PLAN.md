# Stackless Coroutines Migration Plan

## Current Status

The kcoro_arena implementation uses **stack-based coroutines**:
- Each `kcoro_t` allocates 64KB stack via `mmap`
- Assembly code (`kc_ctx_switch.S`) saves/restores registers and stack pointer
- Trampoline function (`kcoro_trampoline`) bootstraps new coroutines on their stack
- Maximum practical scale: ~100K coroutines due to address space limits

## Target Architecture

**Stackless coroutines** using Continuation-Passing Style (CPS):
- Zero per-coroutine memory overhead (only ~256 byte continuation record)
- No assembly code required (pure C implementation)
- Scales to millions of coroutines
- Better cache locality and reduced TLB pressure

## Implementation Phases

### Phase 1: Core Stackless Infrastructure (Week 1)

**Files to create:**
- `external/kcoro_arena/core/src/kc_continuation.c`
- `external/kcoro_arena/core/src/kc_continuation.h`
- `external/kcoro_arena/core/src/kc_cps_dispatch.c`

**Data structures:**
```c
/* Core continuation record */
typedef struct kc_continuation {
    int state_id;                        /* Which suspension point */
    void (*resume_fn)(struct kc_continuation*, void* result);
    void* captured[KCoro_MAX_CAPTURED];  /* Captured local vars */
    void* result;                        /* Result from suspension */
    struct kc_continuation* caller;      /* Return continuation */
    void* user_data;                     /* Opaque per-coroutine data */
} kc_continuation_t;

/* Continuation allocator (bump allocator for performance) */
typedef struct kc_cont_arena {
    void* base;
    size_t size;
    atomic_size_t offset;
} kc_cont_arena_t;
```

**Key functions:**
```c
kc_continuation_t* kc_cont_create(void (*resume_fn)(kc_continuation_t*, void*));
void kc_cont_resume(kc_continuation_t* k, void* result);
void kc_cont_destroy(kc_continuation_t* k);
```

**Tests:**
- `test_continuation_lifecycle.c`: Create → suspend → resume → destroy
- `test_continuation_chain.c`: Nested continuations (caller/callee)

### Phase 2: Async Arena API (Week 2)

**Modify:**
- `external/kcoro_arena/core/src/kc_arena.c`
- `external/kcoro_arena/core/include/kcoro_arena.h`

**New async API:**
```c
/* Old (blocking) */
void* kc_arena_recv(kc_arena_t* arena, int ticket);

/* New (async + continuation) */
void kc_arena_recv_async(kc_arena_t* arena, int ticket, kc_continuation_t* k);
void kc_arena_send_async(kc_arena_t* arena, int ticket, void* payload, kc_continuation_t* k);
```

**`RendezvousCell` modification:**
```c
struct RendezvousCell {
    atomic_int state;
    kc_continuation_t* sender_cont;    /* Changed from WaiterToken* */
    kc_continuation_t* receiver_cont;
    void* payload;
    // ... metrics unchanged
};
```

**Match logic:**
```c
bool try_match_stackless(RendezvousCell* cell) {
    if (atomic_compare_exchange_strong(&cell->state, &expected_ready, MATCHED)) {
        kc_continuation_t* recv = atomic_exchange(&cell->receiver_cont, NULL);
        kc_continuation_t* send = atomic_exchange(&cell->sender_cont, NULL);
        
        if (recv) {
            kc_cont_resume(recv, cell->payload);  /* Direct function call */
        }
        if (send) {
            kc_cont_resume(send, NULL);  /* Ack sender */
        }
        return true;
    }
    return false;
}
```

**Tests:**
- `test_arena_async_send_recv.c`: Async send/recv with continuations
- `test_arena_async_many.c`: 10K+ concurrent coroutines

### Phase 3: Event-Driven Workers (Week 3)

**Replace polling workers with event-driven:**

**Before (polling):**
```c
void arena_worker_loop() {
    while (true) {
        for (int ticket = start; ticket < end; ticket++) {
            if (try_match(&cells[ticket])) { /* ... */ }
        }
        koro_yield();  // Spin-yield
    }
}
```

**After (event-driven):**
```c
void arena_worker_loop() {
    while (true) {
        int ticket = channel_recv(event_queue);  // Blocks until event
        if (try_match_stackless(&cells[ticket])) {
            // Continuations resumed inline
        }
    }
}
```

**Event notification:**
```c
void kc_arena_send_async(...) {
    // ... place continuation in cell ...
    if (CAS_success) {
        channel_send(event_queue, ticket);  // Wake worker
    }
}
```

**Tests:**
- `test_event_driven_workers.c`: Verify zero CPU when idle
- `bench_event_latency.c`: Measure wake-to-match latency

### Phase 4: User API Helpers (Week 4)

**Create macro-based DSL for CPS transformation:**

**File:** `external/kcoro_arena/core/include/kc_async.h`

```c
#define KC_ASYNC_FN(name, arg_type) \
    void name##_resume(kc_continuation_t* __k, void* __r); \
    kc_continuation_t* name(arg_type arg)

#define KC_ASYNC_BEGIN(state_struct) \
    state_struct* __s = (state_struct*)__k->captured[0]; \
    switch (__s->state) { case 0:

#define KC_AWAIT(async_call, state_id) \
    do { \
        __s->state = state_id; \
        return (async_call); \
    } while(0); \
    case state_id:

#define KC_ASYNC_RETURN(value) \
    do { \
        if (__k->caller) { \
            kc_cont_resume(__k->caller, (void*)(value)); \
        } \
        kc_cont_destroy(__k); \
        return; \
    } while(0)

#define KC_ASYNC_END }
```

**Example usage:**
```c
struct my_state {
    int state;
    int x;
};

KC_ASYNC_FN(my_coroutine, void*) {
    KC_ASYNC_BEGIN(struct my_state)
        __s->x = compute();
        KC_AWAIT(kc_arena_recv_async(arena, 42, __k), 1)
        process(__s->x, (int)(intptr_t)__r);
        KC_ASYNC_RETURN(NULL);
    KC_ASYNC_END
}
```

**Tests:**
- `test_async_macros.c`: Verify macro expansion correctness
- `example_async_pipeline.c`: Real-world pipeline with multiple suspension points

### Phase 5: Deprecate Stack-Based Code (Week 5)

**Conditional compilation:**
```c
#ifdef KC_USE_STACKS
    // Keep old stack-based implementation
    #include "kcoro_core_stacks.c"
#else
    // Default to stackless
    #include "kc_continuation.c"
    #include "kc_cps_dispatch.c"
#endif
```

**Update build system:**
```makefile
# Default build (stackless)
CFLAGS += -DKC_STACKLESS=1

# Legacy build (stacks)
# CFLAGS += -DKC_USE_STACKS=1
```

**Migration guide:**
- Document: `docs/STACK_TO_STACKLESS_MIGRATION.md`
- Provide automated refactoring script: `tools/convert_to_async.sh`

**Tests:**
- Run full test suite with both `KC_USE_STACKS` and `KC_STACKLESS`
- Verify identical behavior

### Phase 6: Benchmark & Optimize (Week 6)

**Benchmarks:**
1. **Memory usage**: 1M coroutines (stack vs stackless)
2. **Context switch latency**: Measure resume overhead
3. **Throughput**: Messages/sec in ping-pong test
4. **Cache performance**: perf stat (L1/L2 miss rates)

**Expected results:**
- Memory: 250× reduction (64KB → 256 bytes per coroutine)
- Latency: 5-10× faster context switch
- Throughput: 2-3× higher messages/sec
- Cache: 50% fewer L2 misses

**Optimize hot paths:**
- Inline `kc_cont_resume` for common cases
- Pool continuation allocations (slab allocator)
- SIMD-optimize continuation state copying

## Migration Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Core stackless infrastructure | `kc_continuation.c`, unit tests |
| 2 | Async arena API | Modified arena with async send/recv |
| 3 | Event-driven workers | Zero-spin worker loop |
| 4 | User API helpers | Macro DSL for CPS |
| 5 | Deprecation of stacks | Conditional compilation, docs |
| 6 | Benchmark & optimize | Performance report, tuning |

## Risk Mitigation

**Risk: Breaking existing code**
- Mitigation: Keep stack-based code under `KC_USE_STACKS` flag
- Mitigation: Comprehensive test suite runs both modes

**Risk: Performance regression in simple cases**
- Mitigation: Inline fast path; profile with perf/Instruments
- Mitigation: Benchmark before/after on representative workloads

**Risk: Complex debugging of CPS state machines**
- Mitigation: Enhanced logging (state_id → source line mapping)
- Mitigation: GDB pretty-printers for continuation structs

**Risk: User adoption friction**
- Mitigation: Provide macro DSL to hide CPS details
- Mitigation: Incremental migration: allow mixing stack/stackless

## Success Criteria

- ✅ 1M+ concurrent coroutines without address space exhaustion
- ✅ Zero CPU usage when all coroutines idle (no spin loops)
- ✅ Context switch latency < 10ns (down from ~50ns stack-based)
- ✅ Throughput ≥ 2× current stack-based implementation
- ✅ All existing tests pass in stackless mode
- ✅ Memory usage < 500MB for 1M coroutines

## Next Steps

1. Review and approve this migration plan
2. Create GitHub milestone: "Stackless Coroutines"
3. Break down into individual issues/PRs
4. Begin Phase 1 implementation

## References

- Protothreads: http://dunkels.com/adam/pt/
- Boost.Context documentation
- BizTalk orchestration engine internals
- Duff's Device pattern in C

---

**Status**: Draft proposal  
**Author**: AI Assistant  
**Date**: 2025-01-13  
**Approved by**: [Pending review]
