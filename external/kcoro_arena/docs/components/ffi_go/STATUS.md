# FFI Go Dispatch: Current Implementation Status

## Summary

The FFI Go dispatch component is **not yet implemented**, but the underlying infrastructure is largely ready. This document tracks what exists, what's missing, and the implementation path forward.

## Infrastructure Already in Place ✅

### 1. TLS for Current Coroutine
**Location**: `external/kcoro_arena/core/src/kcoro_core.c`

```c
static __thread kcoro_t* current_kcoro = NULL;

kcoro_t* kcoro_current(void) {
    return current_kcoro;
}
```

- Uses C11 `__thread` for thread-local storage
- Already set/cleared around `kcoro_switch()` calls
- Accessible via public `kcoro_current()` API

**Status**: ✅ Complete — no changes needed

### 2. Coroutine State Machine
**Location**: `external/kcoro_arena/include/kcoro_core.h`

```c
typedef enum {
    KCORO_INIT = 0,
    KCORO_READY,
    KCORO_RUNNING,
    KCORO_SUSPENDED,
    KCORO_DEAD
} kcoro_state;
```

**Status**: ✅ Complete — compatible with C coroutine state tracking

### 3. Stack Allocation & Context Switching
**Location**: 
- Assembly: `external/kcoro_arena/core/src/kc_vm_apply.S` (ARM64)
- Wrapper: `external/kcoro_arena/core/src/kc_ctx_switch_vm.c`

- Fast context switch via assembly (10-20 ns)
- Supports custom stack allocation
- Properly saves/restores registers

**Status**: ✅ Complete — ready for C function stacks

### 4. Arena Infrastructure
**Location**: `external/kcoro_arena/core/src/kc_arena.c`

- RendezvousCell state machine
- Atomic ticket management
- Worker coroutines for matching

**Status**: ✅ Complete — needs C wrapper API

### 5. Scheduler Runtime
**Location**: `external/kcoro_arena/core/src/kc_sched.c`

- Multi-worker thread pool
- Ready queue and coroutine dispatch
- `kc_sched_spawn()` for adding new coroutines

**Status**: ✅ Complete — can spawn C-wrapped coroutines

## Missing Components ❌

### 1. C API Header (`kcoro_c.h`)
**Status**: ❌ Not created

Need to create `external/kcoro_arena/include/kcoro_c.h` with:

```c
// Global init/shutdown
void koro_init(void);
void koro_shutdown(void);

// Spawn C function as coroutine
void koro_go(void (*func)(void*), void* arg);

// Suspension points
void  koro_send(int ticket, void* payload);
void* koro_recv(int ticket);

// Non-blocking variants
int   koro_try_send(int ticket, void* payload);
void* koro_try_recv(int ticket);
```

### 2. C Coroutine Context Wrapper
**Status**: ❌ Not implemented

Need to create `c_coroutine_ctx` structure and management:

```c
typedef struct c_coroutine_ctx {
    void*          stack_base;
    size_t         stack_size;
    void          (*entry_func)(void*);
    void*          entry_arg;
    kcoro_t*       coro;
    uint64_t       id;
    c_coro_state   state;
} c_coroutine_ctx;
```

Functions needed:
- `c_func_wrapper()` — internal trampoline that invokes user function
- `c_coroutine_ctx_new()` — allocate context + stack
- `c_coroutine_ctx_free()` — cleanup when done

### 3. Global Registry
**Status**: ❌ Not implemented

Need a thread-safe registry of active C coroutines for:
- Introspection (`koro_dump_active()`)
- Cleanup on shutdown
- Debugging support

Minimal implementation:

```c
typedef struct {
    c_coroutine_ctx**  contexts;
    size_t             count;
    size_t             capacity;
    pthread_mutex_t    lock;
} c_coro_registry;

static c_coro_registry global_registry;
```

### 4. Arena C Wrappers
**Status**: ❌ Not implemented

Current arena API uses kcoro-specific types. Need C-friendly wrappers:

```c
// Map to existing kc_arena_send/recv but handle types
void koro_send(int ticket, void* payload) {
    kcoro_t* co = kcoro_current();
    // ... validation ...
    kc_arena_send(global_arena, ticket, payload, -1);
}
```

### 5. Error Handling Strategy
**Status**: ❌ Not defined

Need to decide:
- `abort()` on misuse (e.g., calling `koro_send` outside coroutine)?
- Return error codes?
- Logging/telemetry for failures?

Current thinking: abort for programmer errors, return codes for expected failures.

### 6. Unit Tests
**Status**: ❌ Not written

Need test suite in `external/kcoro_arena/tests/`:
- `test_ffi_basic.c` — spawn, send, recv, complete
- `test_ffi_multi.c` — multiple coroutines, different tickets
- `test_ffi_stress.c` — 10k coroutines, validate no leaks
- `test_ffi_errors.c` — misuse detection

### 7. Example Programs
**Status**: ❌ Not written

Need demonstrative examples:
- `examples/ffi_hello.c` — minimal hello-world
- `examples/ffi_producer_consumer.c` — classic pattern
- `examples/ffi_pipeline.c` — chain of processing stages

## Implementation Roadmap

### Phase 1: Core API (1-2 days)
1. Create `include/kcoro_c.h` with function declarations
2. Implement `c_coroutine_ctx` structure and lifecycle
3. Write `koro_go()` function (stack alloc + spawn)
4. Write `c_func_wrapper()` trampoline

**Validation**: Can spawn a C function that prints and returns

### Phase 2: Communication (1 day)
1. Implement `koro_send()` wrapper around `kc_arena_send`
2. Implement `koro_recv()` wrapper around `kc_arena_recv`
3. Add state tracking (suspend counts, times)

**Validation**: Producer/consumer pair exchanges messages

### Phase 3: Lifecycle (1 day)
1. Implement `koro_init()` (scheduler + arena setup)
2. Implement `koro_shutdown()` (drain + cleanup)
3. Add global registry for active contexts
4. Add cleanup on coroutine completion

**Validation**: No memory leaks after shutdown

### Phase 4: Testing (1-2 days)
1. Write unit tests
2. Write stress tests
3. Run under valgrind/ASAN
4. Benchmark vs. native kcoro

**Validation**: All tests pass, <10% overhead

### Phase 5: Documentation & Examples (1 day)
1. Write example programs
2. Add API reference documentation
3. Document limitations and best practices

**Validation**: External developers can use API without consulting source

**Total Estimated Effort**: 5-7 days for one developer

## Dependencies

### External
- None — uses existing POSIX APIs (pthread, malloc)

### Internal
- kcoro_core (already stable)
- kc_sched (already stable)
- kc_arena (already stable)
- Context switch assembly (already stable)

All dependencies are mature; no blocking issues.

## Risk Assessment

### Low Risk
- TLS and context switching are proven
- Scheduler and arena are battle-tested in native kcoro
- C FFI patterns are well-understood

### Medium Risk
- Stack size tuning (64KB may be insufficient for some workloads)
- Error propagation from C code (need clear abort/return strategy)
- Interaction with existing kcoro features (select, cancellation, scopes)

### Mitigation
- Make stack size configurable via env var
- Document clear guidelines for C function design
- Test interop scenarios explicitly

## Success Criteria

FFI Go is considered **complete** when:

1. ✅ Pure C program can spawn coroutines via `koro_go()`
2. ✅ C functions can send/receive via `koro_send/recv()`
3. ✅ Multiple C coroutines can communicate concurrently
4. ✅ Clean shutdown with no memory leaks
5. ✅ <10% overhead vs. native kcoro performance
6. ✅ All unit and stress tests pass
7. ✅ Example programs demonstrate common patterns
8. ✅ API documentation is clear and complete

## Next Steps

**Before starting implementation**:
1. Review this status document with team
2. Confirm API design (function names, signatures)
3. Agree on error handling strategy
4. Plan test coverage

**First implementation task**:
Create `include/kcoro_c.h` with API declarations and placeholder implementations that compile but return errors. This establishes the API contract before implementation.

---

**Last Updated**: 2025-01-13  
**Maintainer**: kcoro_arena team
