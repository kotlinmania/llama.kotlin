# Backporting Stackless Architecture from kcoro_arena to Base kcoro

## Executive Summary

The stackless coroutine architecture developed for `kcoro_arena` offers substantial benefits that should be backported to base `kcoro`. This document outlines the strategy, trade-offs, and implementation plan.

## Key Benefits of Stackless Architecture

### Memory Efficiency
- **Current**: ~64KB-1MB per coroutine (stack allocation)
- **Stackless**: ~100-500 bytes per coroutine (continuation record)
- **Impact**: Scale from ~10K to potentially **millions** of concurrent coroutines

### Code Simplicity
- **Eliminate** `kc_ctx_switch.S` (ARM64/x86_64 assembly)
- **Eliminate** platform-specific stack allocation (`mmap` with guard pages)
- **Portable**: Pure C implementation works everywhere
- **Debuggable**: Standard debuggers work perfectly (no stack switching)

### Performance
- **Faster** context switches: function calls vs register save/restore
- **Better** cache locality: all execution on one stack
- **Predictable**: No mysterious stack overflow issues

## Trade-offs and Constraints

### Function Color Problem
Stackless coroutines introduce the "function color" constraint:

- **Stackful** (current): Can call any C function, including blocking operations
- **Stackless** (proposed): Can ONLY call non-blocking, CPS-transformed functions

**Impact**: Any function that needs to suspend must be written in continuation-passing style.

**Example** of the constraint:
```c
// STACKFUL (current) - works fine
void my_coro(void* arg) {
    FILE* f = fopen("data.txt", "r");  // Blocking call - OK
    kc_chan_send(ch, data);            // Suspend - OK
    fread(buf, 1, 100, f);             // Blocking call - OK
}

// STACKLESS (proposed) - blocking calls don't work
void* my_coro(koro_cont_t* k) {
    struct my_state { int i; } *s = k->user_data;
    KORO_BEGIN(k);
    
    // CANNOT do: fread() - would block the entire scheduler
    // MUST use: koro_file_read_async() - which returns immediately
    KORO_SEND(k, ch, data, len);  // OK - uses CPS internally
    
    KORO_END(k);
}
```

**Mitigation**: For base `kcoro`, this is acceptable because:
1. All channel/arena operations are already async
2. File I/O can be delegated to a thread pool
3. Network I/O is naturally async (epoll/kqueue integration)

### API Surface Changes

Users must adapt to Protothread-style macros:
```c
// OLD (stackful)
void producer_func(void* arg) {
    for (int i = 0; i < 100; i++) {
        kc_chan_send(ch, &i, sizeof(i));
    }
}

// NEW (stackless)
void* producer_func(koro_cont_t* k) {
    struct producer_state { int i; } *state = k->user_data;
    
    KORO_BEGIN(k);
    for (state->i = 0; state->i < 100; state->i++) {
        KORO_SEND(k, ch, &state->i, sizeof(state->i));
    }
    KORO_END(k);
}
```

**Mitigation**: Provide migration guide and maintain stackful API in parallel during transition.

## Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Port stackless primitives without breaking existing code

1. **Add** `external/kcoro/include/kcoro_stackless.h`
   - Copy from `kcoro_arena/include/kcoro_stackless.h`
   - Adapt to base kcoro scheduler
   
2. **Add** `external/kcoro/core/src/kcoro_stackless.c`
   - Implement `koro_cont_create/destroy`
   - Adapt `koro_go` to accept stackless functions
   
3. **Extend** `kc_sched.c` to support both models:
   ```c
   typedef enum {
       KCORO_TYPE_STACKFUL,
       KCORO_TYPE_STACKLESS
   } kcoro_execution_model_t;
   
   struct kcoro {
       kcoro_execution_model_t model;
       union {
           struct {
               void* reg[32];
               void* stack_ptr;
               size_t stack_size;
           } stackful;
           struct {
               koro_cont_t* continuation;
           } stackless;
       };
       // ... rest of fields
   };
   ```

4. **Test** stackless execution in isolation (new test suite)

### Phase 2: Channel Integration (Week 2)
**Goal**: Make channels work with stackless coroutines

1. **Port** `koro_send_stackless` and `koro_recv_stackless` from arena
2. **Adapt** to base kcoro's channel implementation (`kc_chan.c`)
3. **Add** `KORO_SEND/KORO_RECV` macros
4. **Test** producer/consumer scenarios

### Phase 3: Scheduler Unification (Week 3)
**Goal**: Optimize scheduler for stackless-first execution

1. **Modify** `kc_sched_drain`:
   ```c
   while (has_work) {
       kcoro_t* co = dequeue_ready();
       if (co->model == KCORO_TYPE_STACKFUL) {
           // Old path: context switch
           kcoro_switch(main_co, co);
       } else {
           // New path: function call
           void* result = koro_cont_step(co->stackless.continuation);
           if (result != NULL) {
               mark_completed(co);
           }
       }
   }
   ```

2. **Add** runtime flag to prefer stackless model:
   ```c
   kc_sched_opts opts = {
       .workers = 4,
       .prefer_stackless = true  // NEW
   };
   ```

3. **Benchmark** mixed workloads (stackful + stackless)

### Phase 4: Migration Path (Week 4)
**Goal**: Provide smooth transition for users

1. **Document** migration patterns:
   - Converting stackful functions to stackless
   - Handling local variables
   - Dealing with nested function calls

2. **Provide** wrapper library for common patterns:
   ```c
   // Helper for iterating
   #define KORO_FOR(k, init, cond, incr) \
       for (init; cond; incr) { KORO_YIELD(k); }
   ```

3. **Create** compatibility shim:
   ```c
   // Old API (stackful) continues to work
   kc_chan_send(ch, data, len);
   
   // New API (stackless) is opt-in
   KORO_SEND(k, ch, data, len);
   ```

### Phase 5: Deprecation (Week 5+)
**Goal**: Gradually sunset stackful implementation

1. **Mark** stackful API as deprecated
2. **Provide** automated migration tool (source-to-source transform)
3. **Remove** assembly code in future major version

## Architectural Decision Records

### ADR-001: Keep Both Models Initially
**Decision**: Support both stackful and stackless in parallel

**Rationale**:
- Allows gradual migration
- Users can choose based on their needs
- Validates stackless design before full commitment

**Consequences**:
- Increased code complexity temporarily
- More test coverage needed
- Clear migration path for users

### ADR-002: Stackless as Default for New Code
**Decision**: New `koro_go` calls use stackless by default

**Rationale**:
- Memory efficiency is critical for scalability
- Portability is increasingly important
- Performance is better in stackless model

**Consequences**:
- New users learn stackless patterns
- Old code continues to work
- Clear direction for the future

### ADR-003: Assembly Removal Timeline
**Decision**: Remove assembly in kcoro 2.0

**Rationale**:
- Once stackless is proven, assembly adds no value
- Maintenance burden is eliminated
- Portability becomes complete

**Consequences**:
- Must ensure stackless is battle-tested
- Migration guide must be comprehensive
- Clear versioning strategy needed

## Success Criteria

### Performance Benchmarks
- [ ] Stackless context switch ≤ 10ns (vs 30ns stackful)
- [ ] Memory per coroutine ≤ 512 bytes (vs 64KB stackful)
- [ ] Scalability: 1M concurrent coroutines on 16GB RAM

### Functional Tests
- [ ] All existing tests pass with `prefer_stackless=true`
- [ ] No regressions in channel throughput
- [ ] Correct cancellation behavior

### User Experience
- [ ] Migration guide covers 95% of use cases
- [ ] Documentation includes before/after examples
- [ ] FAQ addresses common pitfalls

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Foundation | Stackless primitives compile and link |
| 2 | Channel Integration | Producer/consumer test passes |
| 3 | Scheduler Unification | Mixed workload benchmarks |
| 4 | Migration Path | Documentation complete |
| 5+ | Deprecation | Stackful marked deprecated |

## Comparison: Current vs Future

| Aspect | Current (Stackful) | Future (Stackless) |
|--------|-------------------|-------------------|
| Memory/coroutine | 64KB-1MB | 100-500 bytes |
| Max concurrency | ~10K | ~1M+ |
| Context switch | 30ns | 10ns |
| Assembly needed | Yes (ARM64/x86_64) | No |
| Portability | 2 architectures | All platforms |
| Debugging | Complex (stack switching) | Standard (single stack) |
| API style | C functions | Protothread macros |
| Blocking calls | Allowed | Not allowed |

## Conclusion

Backporting the stackless architecture from `kcoro_arena` to base `kcoro` is a strategic investment that will:

1. **Dramatically** improve memory efficiency (100x reduction per coroutine)
2. **Eliminate** platform-specific assembly code
3. **Simplify** debugging and maintenance
4. **Enable** scaling to millions of concurrent coroutines

The trade-off (function color constraint) is acceptable for `kcoro`'s use case, where all operations are already asynchronous by design.

**Recommendation**: Proceed with Phase 1 immediately after completing current kcoro_arena stabilization work.

## References

- `external/kcoro_arena/include/kcoro_stackless.h` - Reference implementation
- `external/kcoro_arena/docs/components/stackless/` - Design documentation
- Protothreads paper: [http://dunkels.com/adam/pt/](http://dunkels.com/adam/pt/)
- Kotlin coroutines (inspiration for original design)
