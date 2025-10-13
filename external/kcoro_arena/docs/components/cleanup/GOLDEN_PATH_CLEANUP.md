# Golden Path Cleanup Plan for kcoro_arena

## Objective
Remove all non-essential macros, feature flags, and alternate code paths to create a single, secure, maintainable "golden path" implementation.

## Principles
1. **Zero API macros** - User-facing API should be fixed and simple
2. **Zero tuning parameters** - Optimal defaults baked in, no knobs
3. **Zero feature flags** - One implementation, fully tested
4. **Platform guards only** - Keep only essential OS/arch compatibility

## Items to KEEP (Essential Platform Compatibility)

### Architecture Guards
- `#if defined(__aarch64__)` in kc_ctx_switch_vm.c (will be removed when stackless complete)
- `#ifdef __linux__` for platform-specific includes

### OS Portability
- `#ifdef CLOCK_MONOTONIC` - fallback timing (essential for BSD/macOS)
- `#ifndef MAP_ANON` / `MAP_ANONYMOUS` - mmap portability
- `#ifndef _GNU_SOURCE` - feature test macro for Linux

### Header Guards
- All `#ifndef HEADER_H` / `#define HEADER_H` / `#endif` guards (standard practice)
- All `#ifdef __cplusplus` extern "C" guards (C++ compatibility)

## Items to REMOVE (Non-Essential)

### 1. Tuning Parameters in kcoro_config.h
**Status: REMOVE ALL**
```c
KCORO_CANCEL_SLICE_MS      // Remove - use fixed optimal value (10ms)
KCORO_UNLIMITED_INIT_CAP   // Remove - use fixed optimal value (256)
KCORO_IPC_BACKLOG          // Remove - not used by core
KCORO_IPC_MAX_TLV_ELEM     // Remove - not used by core
```

**Action:** Replace with hardcoded constants in implementation files.

### 2. Token Kernel Tuning
**File:** kc_token_kernel.c
```c
#ifndef KC_TOKEN_KERNEL_BUCKETS
#define KC_TOKEN_KERNEL_BUCKETS 1024u  // Remove - use fixed value
#endif
```

**Action:** Hardcode to 1024 or optimal value based on testing.

### 3. Scheduler Tuning
**File:** kc_sched.c
```c
#ifndef KC_SCHED_STEAL_SCAN_MAX
// Remove - use fixed optimal value
#endif
```

**Action:** Hardcode optimal work-stealing scan limit.

### 4. Stackful Code Paths
**File:** kc_token_kernel.c
```c
#ifndef KCORO_STACKLESS_BUILD
// All stackful fallback code
#endif
```

**Action:** Remove all `#ifndef KCORO_STACKLESS_BUILD` blocks entirely. Stackless is the golden path.

### 5. Zero-Copy Feature Flag
**File:** include/kcoro_zcopy.h
```c
#ifndef KC_CHAN_CAP_ZERO_COPY
// ...
#endif
```

**Action:** Zero-copy is mandatory in arena implementation. Remove guards and always enable.

## Implementation Steps

### Phase 1: Remove Tuning Parameters
1. Replace all `#ifndef PARAM` with hardcoded optimal values
2. Remove kcoro_config.h entirely or reduce to empty file with comment
3. Update documentation to reflect fixed configuration

### Phase 2: Remove Stackful Code
1. Remove all `#ifndef KCORO_STACKLESS_BUILD` conditional blocks
2. Remove kc_ctx_switch.S (old stackful assembly)
3. Keep only kc_ctx_switch_vm.c for VM-based stackless implementation
4. Remove legacy kcoro_core.c stack allocation code

### Phase 3: Clean Up Headers
1. Remove KC_CHAN_CAP_ZERO_COPY - always enabled
2. Simplify kcoro_zcopy.h to unconditional API
3. Remove IPC-specific tunables from kcoro_config.h

### Phase 4: Verify Golden Path
1. Compile with zero non-essential macros defined
2. Run full test suite
3. Verify no conditional compilation outside platform guards
4. Security audit - ensure no attack surface from removed flags

## Expected Benefits

### Security
- **Reduced attack surface** - No alternate code paths to exploit
- **No CVE exposure** - Can't trigger bugs in disabled features
- **Simpler auditing** - Single code path to review

### Maintainability
- **No configuration matrix** - Test one path, not 2^N combinations
- **Clear semantics** - No "if you enable X, then Y behaves..."
- **Faster builds** - No macro expansion overhead

### Performance
- **Optimal defaults** - Based on real testing, not user guesswork
- **Better optimization** - Compiler can inline/optimize fixed paths
- **Predictable behavior** - No performance cliffs from misconfig

## Testing Strategy

After each phase:
1. **Compile test** - Ensure code builds with no warnings
2. **Unit tests** - All atomic operation tests pass
3. **Integration tests** - Producer/consumer stress tests pass
4. **Benchmark** - Verify performance matches or exceeds before cleanup

## Final State

After cleanup, kcoro_arena will have:
- **Zero user-configurable macros**
- **Zero feature flags**
- **One tested, optimal implementation**
- **Platform guards only** (arch, OS compatibility)

This creates a "pit of success" where users can't misconfigure the library, and maintainers test exactly one code path.
