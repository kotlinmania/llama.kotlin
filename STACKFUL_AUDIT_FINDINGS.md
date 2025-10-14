# Stackful Code Audit Findings

## Executive Summary

After analyzing the codebase, stackful coroutine code remains in both `external/kcoro` and `external/kcoro_arena`. However, **the presence of stackful code does not mean it was "resurrected"** - it's legacy code that was never fully removed during the stackless migration.

## Key Findings

### 1. kcoro_arena Status

**What's Currently Built:**
- The Makefile (line 17) compiles: `kc_chan.c`, `kc_actor.c`, `kc_cancel.c`, `kc_sched.c`, `kcoro_core.c`, `kc_scope.c`, `kc_select.c`, `kc_zcopy.c`, `kc_runtime_config.c`, `kc_bench.c`, `kc_dispatch.c`, `kc_token_kernel.c`, `kc_desc.c`, `kc_arena.c`
- Assembly: `kc_ctx_switch.S` (x86_64 or aarch64)

**Stackful Usage Analysis:**
```
kc_actor.c:     2 references
kc_bench.c:     7 references  
kc_chan.c:     12 references
kc_sched.c:     4 references
kc_select.c:    1 reference
kcoro_core.c:   5 references
```

**Stackless Components (Clean):**
- `kc_token_kernel.c` - Pure stackless implementation ✓
- `kc_arena.c` - Arena management ✓
- `kc_desc.c` - Descriptor management ✓
- `kc_dispatch.c` - Dispatcher (0 stackful refs) ✓
- `kc_scope.c` - Scope management (0 stackful refs) ✓
- `kc_cancel.c` - Cancellation ✓
- `kc_runtime_config.c` - Config ✓

**Test Files Using Stackful:**
- `test_chan_buffered_basic.c`
- `test_chan_close_semantics.c`
- `test_chan_failure_counters.c`
- `test_chan_ptr_rendezvous_basic.c`
- `test_chan_rv_zref_basic.c`
- `test_chan_rv_zref_timeout.c`
- `test_chan_zref_stress.c`

**Test Files Using Stackless:**
- `test_token_kernel_basic.c` ✓
- `test_stackless_primitives.c` ✓
- `test_stubs_stackless.c` ✓
- `test_atomic_operations.c` ✓

### 2. external/kcoro Status

This entire directory is the **original stackful implementation**. It has NOT been migrated to stackless and serves as the "Kotlin-aligned" coroutine library.

## What Needs To Happen

### Option A: Full Stackless Migration (Aggressive)

Remove ALL stackful code from both projects:

1. **kcoro_arena**: Remove stackful code from:
   - `kc_chan.c` - rewrite channel ops to use token kernel
   - `kc_actor.c` - rewrite actor model
   - `kc_bench.c` - rewrite benchmarks
   - `kc_sched.c` - rewrite scheduler 
   - `kc_select.c` - rewrite select
   - `kcoro_core.c` - remove stack management
   - Delete `kc_ctx_switch.S` assembly files
   - Rewrite or delete the 7 stackful tests

2. **external/kcoro**: Either:
   - Delete entirely, OR
   - Keep as separate "Kotlin-aligned stackful library" with clear separation

### Option B: Hybrid Approach (Pragmatic)

Keep two separate implementations:

1. **kcoro_arena** → Complete stackless migration:
   - Finish token kernel integration
   - Remove all stackful primitives
   - Keep only stackless tests

2. **external/kcoro** → Keep as stackful library:
   - Document as "Kotlin-aligned stackful coroutines"
   - Separate from arena project entirely
   - No cross-contamination

### Option C: Coexistence (Current State)

Accept that both models can coexist in the same binary:
- Stackless for new code (token kernel, arena)
- Stackful for legacy APIs (channels, actors, select)
- Document the boundary clearly

## Recommendation

**Option B (Hybrid)** is the cleanest path forward:

1. Complete the stackless migration in `kcoro_arena`
2. Keep `external/kcoro` as a separate, standalone stackful library
3. Clear documentation that they are two different coroutine implementations
4. Users choose which to link against

This preserves both implementations without confusion and allows kcoro_arena to be fully stackless while kcoro remains a Kotlin-aligned stackful implementation.

## Action Items

### Immediate (kcoro_arena stackless completion):

1. Finish token kernel event callback system
2. Rewrite `kc_chan.c` to use token kernel instead of direct kcoro calls
3. Rewrite `kc_actor.c` for stackless model  
4. Rewrite `kc_sched.c` for stackless dispatching
5. Rewrite `kc_bench.c` tests for stackless
6. Remove `kc_ctx_switch.S` assembly
7. Rewrite or remove 7 stackful tests
8. Update all headers to remove stackful API exports

### Long-term (kcoro cleanup):

1. Document `external/kcoro` as "Kotlin-aligned stackful coroutine library"
2. Add README explaining the difference between kcoro and kcoro_arena
3. Consider moving kcoro to a separate repository if desired
4. Ensure build systems keep them separate

## Conclusion

**No code was "resurrected"** - the stackful code was simply never fully removed during the stackless migration. The presence of stackful constructs in `kcoro_arena` is legacy code that still needs to be converted or removed.

The path forward is clear: complete the stackless migration in kcoro_arena while keeping kcoro as a separate stackful library.
