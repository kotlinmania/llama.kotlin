# Stackful Code Audit Summary

## Executive Summary

After thorough code review, the status is:

### ✅ external/kcoro - STACKLESS (Complete)
This implementation is **already stackless** and uses the CPS (Continuation Passing Style) model correctly.

**Evidence:**
- `kcoro_core.c` uses `calloc` to allocate coroutine objects on heap
- No `mmap` calls for stack allocation
- `kcoro_t` struct contains `next_step` function pointer and `user_data` (replaces stack)
- Comment in code: "Stackless: no stack to free" (line 99)
- `kcoro_resume` calls stored step functions, no context switching
- `kcoro_yield`/`kcoro_park` only flip state bits, no stack manipulation

**Verdict:** ✅ No work needed - this is the reference stackless implementation

---

### ❌ external/kcoro_arena - STACKFUL (Needs Conversion)
This implementation **still uses stackful coroutines** and must be converted.

**Evidence of stackful code:**

1. **Assembly context switchers exist:**
   - `arch/aarch64/kc_ctx_switch.S`
   - `arch/x86_64/kc_ctx_switch.S`  
   - `core/src/kc_ctx_switch_vm.c`

2. **Stack allocation in kcoro_core.c:**
   - Line 89-138: `kcoro_create()` allocates stacks with `mmap`
   - Line 35: `#define KCORO_DEFAULT_STACK_SIZE (64 * 1024)`
   - Line 97-109: Stack memory allocation with guard pages
   - Line 118-119: `co->stack_ptr` and `co->stack_size` fields
   - Line 130-136: Stack pointer setup and alignment
   - Line 144-145: `munmap(co->stack_ptr, co->stack_size)` in cleanup

3. **Stackful API calls throughout:**
   - `kc_spawn_co()` accepts `stack_size` parameter
   - `kcoro_create(fn, arg, stack_size)` signature
   - All scheduler/channel/scope code uses stackful primitives

**Verdict:** ❌ Must be converted to stackless (follow kcoro model)

---

## Recommended Action

**Copy the stackless patterns from external/kcoro to external/kcoro_arena:**

1. Delete assembly files (.S files)
2. Replace stackful `kcoro_create()` with stackless `kcoro_create_cps()`
3. Update `kcoro_t` struct to remove `stack_ptr`/`stack_size`, add `next_step`
4. Eliminate all `mmap`/`munmap` calls
5. Replace context switching with simple function pointer calls
6. Update all tests to use CPS style

The good news: **We already have a working stackless reference** in external/kcoro to copy from!
