# kcoro_arena Macro and Configuration Audit

This document catalogs all conditional compilation paths, configuration macros, and tuning parameters in `kcoro_arena`. The goal: eliminate all paths except the "golden" production path.

## Current Conditional Compilation Directives

### Platform Detection (Keep - Required for Portability)

| Macro | File | Purpose | Action |
|-------|------|---------|--------|
| `__aarch64__` | kc_ctx_switch_vm.c | ARM64 architecture | Keep (platform-specific) |
| `__linux__` | kc_sched.c | Linux-specific pthread features | Keep (platform-specific) |
| `__APPLE__`, `__MACH__` | port/posix.h | macOS/Darwin detection | Keep (platform-specific) |
| `_GNU_SOURCE` | kcoro_core.c | GNU extensions | Keep (platform-specific) |
| `MAP_ANON`, `MAP_ANONYMOUS` | kcoro_core.c | mmap flags | Keep (portability) |
| `CLOCK_MONOTONIC` | Multiple files | POSIX clock source | Keep (portability) |

### API Protection (Keep - Standard C/C++ Hygiene)

| Macro | Files | Purpose | Action |
|-------|-------|---------|--------|
| `#ifndef FOO_H` + `#define FOO_H` | All .h files | Include guards | Keep (standard practice) |
| `#ifdef __cplusplus` + `extern "C"` | All public .h | C++ compatibility | Keep (standard practice) |

### Feature Flags (REMOVE - Attack Surface)

| Macro | File | Purpose | Golden Path | Action |
|-------|------|---------|-------------|--------|
| `KCORO_STACKLESS_BUILD` | kc_token_kernel.c | Toggle stackless vs stackful | Stackless | **REMOVE** - always stackless |
| `KC_SCHED` | kc_actor_bench.h | Enable scheduler integration | Enabled | **REMOVE** - always enabled |
| `HAS_KORO_SCHEDULER` | test_stackless_primitives.c | Test-time scheduler flag | Enabled | **REMOVE** - always enabled |

### Tuning Parameters (REMOVE - Use Hardcoded Defaults)

| Macro | File | Default | Golden Value | Action |
|-------|------|---------|--------------|--------|
| `KC_TOKEN_KERNEL_BUCKETS` | kc_token_kernel.c | 256 | 256 | **REMOVE** - hardcode 256 |
| `KC_SCHED_STEAL_SCAN_MAX` | kc_sched.c, kcoro_sched.h | 8 | 8 | **REMOVE** - hardcode 8 |
| `KCORO_CANCEL_SLICE_MS` | kcoro_config.h | 100 | 100 | **REMOVE** - hardcode 100 |
| `KCORO_UNLIMITED_INIT_CAP` | kcoro_config.h | 128 | 128 | **REMOVE** - hardcode 128 |
| `KCORO_IPC_BACKLOG` | kcoro_config.h | 64 | 64 | **REMOVE** - hardcode 64 |
| `KCORO_IPC_MAX_TLV_ELEM` | kcoro_config.h | 16 | 16 | **REMOVE** - hardcode 16 |
| `KC_CHAN_CAP_ZERO_COPY` | kcoro_zcopy.h | Varies | Enabled | **REMOVE** - always enabled |

### Runtime Environment Variables (NONE FOUND)

No `getenv()` calls found in core implementation. Configuration is compile-time only.

## Simplification Plan

### Phase 1: Remove Feature Flags

**Affected files:**
- `kc_token_kernel.c` - Remove all `#ifndef KCORO_STACKLESS_BUILD` guards, keep only stackless code
- `kc_actor_bench.h` - Remove `#ifdef KC_SCHED` guard, always include scheduler
- `test_stackless_primitives.c` - Remove `#ifdef HAS_KORO_SCHEDULER`, always test with scheduler

**Validation:** Rebuild all tests, ensure no dead code paths remain.

### Phase 2: Hardcode Tuning Parameters

**Target files:**
- `kc_token_kernel.c` - Replace `#ifndef KC_TOKEN_KERNEL_BUCKETS` with `#define KC_TOKEN_KERNEL_BUCKETS 256`
- `kc_sched.c`, `kcoro_sched.h` - Hardcode `KC_SCHED_STEAL_SCAN_MAX = 8`
- `kcoro_config.h` - Convert all `#ifndef` guards to plain `#define` statements

**Validation:** Confirm performance unchanged from defaults, document in code comments.

### Phase 3: Zero-Copy Always Enabled

**Target files:**
- `kcoro_zcopy.h` - Remove `#ifndef KC_CHAN_CAP_ZERO_COPY`, always define it

**Validation:** Run zero-copy stress tests.

### Phase 4: Final Audit

**Actions:**
1. Grep for remaining `#ifdef` not in keep categories
2. Verify no dead code branches
3. Run full test suite with all simplifications
4. Update documentation to reflect single golden path

## Security Rationale

**Why remove tuning knobs?**
- Each `#ifdef` is a potential attack vector (buffer size bypass, feature downgrade)
- Untested code paths harbor bugs
- CVE reviewers flag configurable security-critical parameters
- Production systems should run exactly one tested configuration

**What we preserve:**
- Platform portability (OS/arch detection)
- Standard C/C++ hygiene (include guards, extern "C")
- No loss of functionality - we pick the best path and lock it in

## Expected Outcome

- **Before:** ~70 conditional compilation points
- **After:** ~30 (all platform portability + standard guards)
- **Security gain:** 40 fewer code paths to audit and maintain
- **Performance:** Unchanged (defaults become constants)
- **Maintenance:** Simpler codebase, one path to optimize and secure
