# Stackful vs Stackless Architecture Audit

## Executive Summary

The kcoro_arena codebase currently contains BOTH stackful and stackless implementations running side-by-side. This audit identifies which files belong to which architecture so we can complete the migration to pure stackless.

## Definitions

**Stackful Architecture:**
- Allocates separate memory stacks per coroutine (via mmap/malloc)
- Requires assembly code for context switching (saving/restoring CPU registers, stack pointers)
- Uses `kcoro_t`, `kcoro_create()`, `kcoro_resume()`, `kcoro_yield()`
- Each coroutine has `stack_ptr` and `stack_size` fields

**Stackless Architecture:**
- Uses heap-allocated continuation records (`koro_cont_t`)
- Implements CPS (Continuation-Passing Style) with explicit state machines
- No assembly required - pure C
- All coroutines run on scheduler's single stack
- Uses callbacks/events for resumption

---

## STACKLESS Implementation (Keep/Complete These)

### Core Stackless Runtime
- `core/src/kcoro_stackless.c` ✓ Pure stackless
- `core/src/kc_token_kernel.c` ✓ Event-driven, no stacks
- `include/kcoro_stackless.h` ✓ Stackless API
- `include/kcoro_token_kernel.h` ✓ Token-based coordination

### Stackless Tests
- `tests/test_stubs_stackless.c` ✓ Tests stackless primitives
- `tests/test_token_kernel_basic.c` ✓ Tests token kernel
- `tests/example_stackless.c` ✓ Demonstrates stackless patterns

---

## STACKFUL Implementation (DELETE These)

### Assembly Context Switching
- `arch/aarch64/kc_ctx_switch.S` ❌ ARM64 stack switching
- `arch/x86_64/kc_ctx_switch.S` ❌ x86-64 stack switching
- `core/src/kc_vm_apply.S` ❌ VM assembly glue
- `core/src/kc_ctx_switch_vm.c` ❌ VM wrapper

### Core Stackful Runtime
- `core/src/kcoro_core.c` ❌ Contains `kcoro_create()` with mmap stack allocation
- `core/src/kc_sched.c` ❌ Scheduler for stackful coroutines
- `core/src/kc_dispatch.c` ❌ Dispatcher with stack_size parameters
- `core/src/kc_scope.c` ❌ Scope management with stackful spawning
- `core/src/kc_select.c` ❌ Select using stackful primitives

### Stackful Headers
- `include/kcoro_core.h` ❌ Defines `kcoro_t` with stack fields
- `include/kcoro_sched.h` ❌ Stackful scheduler API
- `include/kcoro_dispatch.h` ❌ Dispatcher with stack_size
- `include/kcoro.h` ❌ Master header including stackful APIs
- `include/kc_job.h` ❌ Job system with stack_size

### Stackful Tests
- `tests/test_sched_basic.c` ❌ Tests stackful scheduler
- `tests/test_chan_ptr_buffered_basic.c` ❌ Uses kcoro_create
- `tests/test_chan_ptr_rendezvous_basic.c` ❌ Uses stackful APIs
- `tests/test_chan_ptr_rendezvous_stress.c` ❌ NEW stackful stress test
- `tests/test_chan_rv_zref_basic.c` ❌ Stackful zero-ref test
- `tests/test_chan_rv_zref_timeout.c` ❌ Stackful timeout test
- `tests/test_chan_zref_stress.c` ❌ Stackful stress test
- `tests/test_desc_alias_lru.c` ❌ Uses stackful runtime
- `tests/test_ptr_rv_cache_stress.c` ❌ Stackful stress test

---

## HYBRID Files (Need Refactoring)

These files contain channel/descriptor logic that should work with EITHER architecture, but currently have stackful dependencies:

### Channel Implementation
- `core/src/kc_chan.c` - Channel logic with stackful waiter queues
- `core/src/kc_chan_internal.h` - Waiter structures reference kcoro_t

**Action:** Refactor waiters to use `koro_cont_t*` instead of `kcoro_t*`

### Zero-Copy System
- `core/src/kc_zcopy.c` - Descriptor management (architecture-agnostic)
- `include/kcoro_desc.h` - Descriptor API (should be architecture-agnostic)

**Action:** Audit and ensure no stackful dependencies

### Utilities
- `core/src/kc_actor.c` - Actor system (could work with either)
- `core/src/kc_bench.c` - Benchmarking (could work with either)

**Action:** Port to use stackless APIs only

---

## UNAFFECTED (Pure Platform/Utility Code)

- `core/src/kcoro_port.c` - Platform abstractions
- `include/kcoro_port.h` - Platform headers
- `tests/test_atomic_operations.c` - Tests atomics, not coroutines

---

## Migration Plan

### Phase 1: Complete Stackless Runtime
1. ✅ Implement `koro_send_cps()` and `koro_recv_cps()` in token kernel
2. ✅ Wire up event callbacks for zero-spin operation
3. ⏳ Build end-to-end stackless producer/consumer test

### Phase 2: Port Hybrid Components
1. Refactor `kc_chan.c` waiters to use `koro_cont_t*`
2. Update channel internal headers
3. Port actor and benchmark code

### Phase 3: Delete Stackful Code
1. Remove all assembly files (`*.S`)
2. Delete `kcoro_core.c` stackful implementation
3. Remove stackful scheduler, dispatcher, scope
4. Delete all stackful tests
5. Remove stackful headers

### Phase 4: Clean Build System
1. Update Makefiles to only build stackless code
2. Remove `-DKCORO_USE_STACKFUL` and related flags
3. Verify no stackful references remain

---

## Why This Matters

Keeping stackful code around:
- **Security:** More attack surface (stack overflows, ROP gadgets)
- **Portability:** Assembly ties us to specific architectures
- **Maintenance:** Double the code to maintain and test
- **Performance:** Stack allocation overhead vs CPS efficiency
- **Complexity:** Two execution models confuse developers

The stackless architecture gives us:
- **Zero spin:** Event-driven, no polling loops
- **Portability:** Pure C, runs anywhere
- **Memory efficiency:** No per-coroutine stack allocation
- **Determinism:** Explicit state machines, easy to reason about

---

## Current Status

- ✅ Stackless runtime core implemented
- ✅ Token kernel with event callbacks working
- ⏳ Hybrid components need refactoring
- ❌ Stackful code still present and buildable
- ❌ No tests exercise pure stackless end-to-end

**Next Step:** Complete Phase 1 by running first pure-stackless producer/consumer test, then proceed to Phase 2 refactoring.
