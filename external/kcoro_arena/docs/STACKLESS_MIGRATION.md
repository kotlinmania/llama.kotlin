# Stackless Coroutine Migration Plan

## Executive Summary

We are transitioning kcoro_arena from a **stackful** to a **stackless** coroutine architecture. This document outlines the rationale, design, implementation phases, and migration path.

## Why Stackless?

### Memory Efficiency

Stackful coroutines allocate full stacks per coroutine:
- Typical stack: 64 KB per coroutine
- 10,000 coroutines = 640 MB of memory
- Most of this is wasted (stack rarely fills)

Stackless coroutines use heap-allocated state:
- Typical continuation: ~100-200 bytes
- 10,000 coroutines = 1-2 MB of memory
- **99% memory savings**

### Portability

Stackful requires platform-specific assembly:
- `kc_ctx_switch_arm64.S` - ARM64 context switching
- `kc_ctx_switch_x86_64.S` - x86-64 context switching  
- Must maintain for every target architecture
- Subject to ABI changes and calling convention shifts

Stackless is pure C:
- Single implementation works everywhere
- No assembly knowledge required
- Compiler handles all ABI details

### Cache Locality

Stackful spreads state across 64KB stacks:
- Poor cache utilization
- Memory wandering between coroutines

Stackless packs state in small records:
- Excellent cache locality
- State fits in L1/L2 cache
- Scheduler loop stays hot

## Architectural Comparison

### Stackful (Current)

```
┌─────────────────────────────────────────────┐
│  Coroutine A Stack (64 KB)                  │
│  ┌───────────────────────────────────────┐  │
│  │ Local variables                       │  │
│  │ Function frames                       │  │
│  │ Return addresses                      │  │
│  │ Saved registers                       │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Requires kc_ctx_switch.S assembly          │
│  to swap SP, LR, callee-saved registers     │
└─────────────────────────────────────────────┘

Suspension:  
1. Save all registers to stack
2. Save SP to coroutine record
3. Load target SP
4. Restore registers from target stack
5. Return to target code

Memory: 64 KB × N coroutines
Lines of assembly: ~50-100 per architecture
```

### Stackless (New Design)

```
┌─────────────────────────────────────────────┐
│  Continuation Record (100 bytes)            │
│  ┌───────────────────────────────────────┐  │
│  │ state = 42         (resumption line)  │  │
│  │ next_step = func   (function pointer) │  │
│  │ user_data → { i=5, ticket=10, ... }   │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Pure C, no assembly required               │
│  All coroutines run on scheduler's stack    │
└─────────────────────────────────────────────┘

Suspension:
1. Update state = __LINE__
2. return NULL;

Resumption:
1. switch(state) { case 42: ... }
2. Continue from saved line

Memory: ~100 bytes × N coroutines  
Lines of assembly: 0
```

## Design: Continuation-Passing Style (CPS)

### Core Concept

User writes sequential-looking code:
```c
void my_coroutine(koro_cont_t* k) {
    struct my_state* s = k->user_data;
    
    KORO_BEGIN(k);
    
    for (s->i = 0; s->i < 10; s->i++) {
        KORO_SEND(k, ticket, &s->data, len);
        // Suspends here if blocked
        // Resumes here when ready
        printf("Sent %d\n", s->i);
    }
    
    KORO_END(k);
}
```

Macro expansion transforms into state machine:
```c
void my_coroutine(koro_cont_t* k) {
    struct my_state* s = k->user_data;
    
    switch (k->state) {
        case 0:  // KORO_BEGIN
            
        for (s->i = 0; s->i < 10; s->i++) {
            k->state = __LINE__;  // KORO_SEND
            // ... attempt send ...
            if (would_block) return NULL;
            case __LINE__:;  // Resume point
            
            printf("Sent %d\n", s->i);
        }
    }  // KORO_END
    k->completed = 1;
    return (void*)1;
}
```

### Key Mechanisms

**Resumption Points**: `__LINE__` macro provides unique integer per suspension point

**State Persistence**: User-defined struct holds all locals that must survive suspension

**Function Pointers**: Scheduler calls `k->next_step(k)` repeatedly

**Suspension Protocol**: Return `NULL` = suspended, non-`NULL` = complete

## Implementation Phases

### Phase 1: Core Structures ✅ COMPLETE

Created:
- `kcoro_stackless.h` - Public API and macros
- `kcoro_stackless.c` - Core implementation
- `stackless_example.c` - Usage demonstration

Defines:
- `koro_cont_t` - Continuation record structure
- `KORO_BEGIN` / `KORO_END` - Function framing macros
- `KORO_YIELD` / `KORO_WAIT_UNTIL` - Suspension macros
- `KORO_SEND` / `KORO_RECV` - Arena integration macros

### Phase 2: Arena Integration (IN PROGRESS)

Must modify:

**`kc_arena.c`**: Add stackless send/receive primitives
```c
void* kc_arena_try_send_cont(kc_arena_t* arena, int ticket, 
                              koro_cont_t* k, void* data, size_t len);
void* kc_arena_try_recv_cont(kc_arena_t* arena, int ticket,
                              koro_cont_t* k);
```

**`kc_chan.c`**: Support continuation-based waiters
- Add `koro_cont_t*` field to `WaiterToken`
- Resume via `k->next_step(k)` instead of context switch

### Phase 3: Scheduler Rewrite (NEXT)

Transform `kc_sched.c` from stack-switching to function-calling:

**Old (Stackful)**:
```c
while (has_work) {
    kcoro_t* co = dequeue_ready();
    kc_ctx_switch(current_co, co);  // Assembly!
}
```

**New (Stackless)**:
```c
while (has_work) {
    koro_cont_t* k = dequeue_ready();
    void* result = k->next_step(k);  // Pure C function call
    if (!result) {
        // Suspended, leave in queue or wait
    } else {
        // Completed, clean up
        koro_cont_destroy(k);
    }
}
```

### Phase 4: Assembly Removal

Delete obsolete files:
- `kc_ctx_switch_arm64.S`
- `kc_ctx_switch_x86_64.S`
- `kc_vm_apply.S`

Update build:
- Remove assembly sources from Makefile
- Simplify platform detection

### Phase 5: FFI and Public API

Create `koro_go` launcher:
```c
/* Launch a C function as a stackless coroutine.
 * - fn: user's step function  
 * - arg: argument to pass
 * - state_size: bytes for user_data
 * Returns coroutine ID. */
uint64_t koro_go(koro_step_fn fn, void* arg, size_t state_size);

/* Block until all coroutines complete. */
void koro_run(void);

/* Shutdown scheduler and cleanup. */
void koro_shutdown(void);
```

Implementation:
1. Allocate continuation record
2. Add to scheduler's ready queue
3. Return immediately (non-blocking)

## Migration Strategy

### Coexistence Period

Both models will coexist temporarily:
- New code uses stackless API (`kcoro_stackless.h`)
- Legacy code continues using stackful (`kcoro_core.h`)
- Tests exercise both paths

### Deprecation Path

1. **Week 1**: Phase 1+2 (structures + arena integration)
2. **Week 2**: Phase 3 (scheduler rewrite)
3. **Week 3**: Phase 4+5 (assembly removal + FFI)
4. **Week 4**: Convert existing tests to stackless
5. **Week 5**: Remove stackful code and old API

## Performance Expectations

### Memory

| Metric | Stackful | Stackless | Improvement |
|--------|----------|-----------|-------------|
| Per-coroutine overhead | 64 KB | 100 bytes | 99.8% reduction |
| 10K coroutines | 640 MB | 1 MB | 640× less |
| Cache footprint | Poor | Excellent | ~10× better locality |

### Latency

| Operation | Stackful | Stackless | Notes |
|-----------|----------|-----------|-------|
| Suspend/resume | ~50-100ns | ~5-10ns | No register save/restore |
| Scheduler loop | ~100ns | ~20ns | Hot function pointers |
| Arena send/recv | ~100ns | ~100ns | Same (dominated by atomics) |

**Net effect**: 2-5× faster scheduling overhead, same arena performance

### Throughput

Expected improvement: **30-50% higher message throughput** due to:
- Better cache utilization
- Faster scheduler dispatch
- Reduced memory pressure

## Testing Plan

### Unit Tests

- `test_stackless_basic.c` - Simple suspend/resume
- `test_stackless_loops.c` - Iteration and conditionals
- `test_stackless_arena.c` - Arena send/recv integration

### Integration Tests

- `test_stackless_prodcons.c` - Classic producer/consumer
- `test_stackless_pingpong.c` - Bidirectional communication
- `test_stackless_stress.c` - 10K+ coroutines

### Benchmark Comparison

Run identical workloads on both implementations:
```
./bench_stackful_rv_metrics --producers=100 --messages=1000
./bench_stackless_rv_metrics --producers=100 --messages=1000
```

Measure:
- Throughput (msg/sec)
- Memory usage (RSS, heap)
- CPU utilization
- Cache miss rate

## Open Questions & Decisions

### 1. User API Style

**Decision**: Use Protothreads-style macros (`KORO_BEGIN`, `KORO_SEND`, etc.)

**Rationale**:
- Familiar to embedded developers
- Clean, readable user code
- Standard C (no compiler extensions)
- Proven in production (Contiki, lwIP)

### 2. State Struct Management

**Decision**: User allocates and manages their state struct

**Rationale**:
- User knows exact size needed
- Type-safe casts in user code
- Library doesn't need reflection/introspection

### 3. Compatibility with `async`/`await`

**Question**: Should we add syntax sugar to mimic other languages?

**Answer**: No. Keep it simple C. Macros are explicit and debuggable.

### 4. Integration with Existing Arena Tests

**Decision**: Run both implementations in parallel during transition

**Plan**:
- Add `--stackless` flag to tests
- Compare results for equivalence
- Gradually migrate test suite

## Conclusion

Stackless coroutines offer dramatic improvements in memory efficiency, portability, and performance. The Protothreads-style macro API provides an ergonomic interface while maintaining full compatibility with standard C.

This migration positions kcoro_arena as a best-in-class, zero-overhead coroutine library suitable for resource-constrained environments and high-performance applications alike.

---

**Status**: Phase 1 complete, Phase 2 in progress  
**Target**: Full migration by end of month  
**Risk**: Low (coexistence strategy allows gradual rollout)
