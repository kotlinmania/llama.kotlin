# kcoro_arena Configuration Options

This document catalogs all preprocessor macros and environment variables that control code paths, features, and behavior in kcoro_arena.

## Build-Time Configuration Macros

### Platform Detection

| Macro | Purpose | Values | Default |
|-------|---------|--------|---------|
| `__aarch64__` | Detect ARM64 architecture | defined/undefined | auto-detected |
| `__linux__` | Detect Linux platform | defined/undefined | auto-detected |
| `__APPLE__`, `__MACH__` | Detect macOS platform | defined/undefined | auto-detected |

### Core Feature Flags

| Macro | Purpose | Values | Default | Location |
|-------|---------|--------|---------|----------|
| `KCORO_STACKLESS_BUILD` | Enable stackless coroutine mode (CPS-based, zero stack allocation) | defined/undefined | undefined | core/include/kcoro_stackless.h |
| `KC_SCHED` | Enable the full cooperative scheduler (not just arena) | defined/undefined | undefined | core/src/kc_sched.c |
| `HAS_KORO_SCHEDULER` | Indicates scheduler presence for dependent code | defined/undefined | auto (set if KC_SCHED) | multiple |

### Channel and Arena Configuration

| Macro | Purpose | Values | Default | Location |
|-------|---------|--------|---------|----------|
| `KC_CHAN_CAP_ZERO_COPY` | Enable zero-copy descriptor mode for channels | `1u<<0` | defined | core/include/kcoro.h |
| `KC_CHAN_CAP_PTR` | Enable pointer-passing mode for channels | `1u<<1` | defined | core/include/kcoro.h |
| `KC_ARENA_MAX` | Maximum number of arenas per process | integer | 4 | core/src/kc_arena.c |
| `KC_DESC_BUCKETS` | Hash table size for descriptor tracking | integer | 256 | core/src/kc_desc.c |
| `KC_TOKEN_KERNEL_BUCKETS` | Hash table size for token kernel | integer | 1024 | core/include/kcoro_token_kernel.h |
| `KC_REGION_F_NONE` | Flags for memory regions | bitfield | 0 | core/src/kc_arena.c |

### Scheduler Tuning

| Macro | Purpose | Values | Default | Location |
|-------|---------|--------|---------|----------|
| `KC_SCHED_STEAL_SCAN_MAX` | Max workers to scan for work-stealing | integer | 4 | core/src/kc_sched.c |
| `KCORO_CANCEL_SLICE_MS` | Cancellation check interval (ms) | integer | 50 | core/src/kc_cancel.c |
| `KCORO_DEFAULT_STACK_SIZE` | Stack size for stackful coroutines (bytes) | integer | 65536 (64KB) | core/include/kcoro_core.h |
| `KCORO_UNLIMITED_INIT_CAP` | Initial capacity for unlimited channels | integer | 64 | core/src/kc_chan.c |

### IPC/Serialization

| Macro | Purpose | Values | Default | Location |
|-------|---------|--------|---------|----------|
| `KCORO_IPC_BACKLOG` | IPC connection backlog | integer | 8 | core/src/kc_ipc.c |
| `KCORO_IPC_MAX_TLV_ELEM` | Max TLV element size | integer | 65535 | core/src/kc_ipc.c |
| `KC_CFG_MAX_FILE_SIZE` | Max config file size | bytes | 1048576 (1MB) | core/src/kc_cfg.c |

### Threading/Synchronization Abstractions

| Macro | Purpose | Platform | Notes |
|-------|---------|----------|-------|
| `KC_MUTEX_*` | Mutex operations | POSIX | pthread_mutex wrappers |
| `KC_COND_*` | Condition variable ops | POSIX | pthread_cond wrappers with monotonic clock |
| `kc_get_nprocs` | CPU count query | Linux/glibc | Aliased to `get_nprocs` |

### Memory Management

| Macro | Purpose | Values | Default |
|-------|---------|--------|---------|
| `KC_ALLOC(n)` | Allocate memory | function-like | `malloc(n)` |
| `KC_FREE(p)` | Free memory | function-like | `free(p)` |
| `MAP_ANON`, `MAP_ANONYMOUS` | Anonymous mapping flag | flag | auto-detect for mmap |

### Error Codes (Negated errno)

| Macro | POSIX Equivalent | Value |
|-------|------------------|-------|
| `KC_EAGAIN` | EAGAIN | -11 |
| `KC_ETIME` | ETIME | -62 |
| `KC_EPIPE` | EPIPE | -32 |
| `KC_ECANCELED` | ECANCELED | -42 (custom) |

### Job/Scope Flags

| Macro | Purpose | Bit |
|-------|---------|-----|
| `KC_JOB_F_SUPERVISOR` | Job is a supervisor | 1<<0 |
| `KC_JOB_F_DETACHED` | Job is detached | 1<<1 |
| `KC_JOB_F_TIMEOUT_WRP` | Job has timeout wrapper | 1<<2 |

### Architecture-Specific

| Macro | Purpose | Value | Platform |
|-------|---------|-------|----------|
| `KC_REG_LR` | Link register index in context | 13 | ARM64 |
| `KC_REG_LR` | Placeholder LR index | 10 | non-ARM |

### Stackless Coroutine Macros (User API)

When `KCORO_STACKLESS_BUILD` is defined, these macros provide the CPS transformation API:

| Macro | Purpose | Signature |
|-------|---------|-----------|
| `KORO_BEGIN(k)` | Start coroutine state machine | `(struct koro_cont* k)` |
| `KORO_END(k)` | End coroutine state machine | `(struct koro_cont* k)` |
| `KORO_YIELD(k)` | Suspend and return to scheduler | `(struct koro_cont* k)` |
| `KORO_SEND(k,ch,data,len)` | Send with suspension point | `(k, channel, data, len)` |
| `KORO_RECV(k,ch)` | Receive with suspension point | `(k, channel)` |
| `KORO_WAIT_UNTIL(k,condition)` | Wait for condition | `(k, cond)` |

## Environment Variables (Runtime)

### Debugging/Observability

| Variable | Purpose | Values | Default | Read By |
|----------|---------|--------|---------|---------|
| `KCORO_TRACE` | Path to trace log file | filepath | none | kc_trace.c |
| `KCORO_DEBUG` | Enable debug output | 0/1 | 0 | kc_debug.c (if exists) |

### Performance Tuning

| Variable | Purpose | Values | Default | Read By |
|----------|---------|--------|---------|---------|
| `KCORO_WORKERS` | Override CPU count for scheduler | integer | auto-detect | kc_sched.c |
| `KCORO_ARENA_SIZE` | Arena allocation size (bytes) | integer | platform-dependent | kc_arena.c |

## Build Modes

### Stackless Mode (`KCORO_STACKLESS_BUILD`)

**When enabled:**
- Eliminates `kc_ctx_switch.S` assembly dependency
- Coroutines execute as state machines using CPS transformation
- Zero per-coroutine stack allocation
- Function color constraint: blocking calls not allowed in coroutine body
- User code must use `KORO_BEGIN` / `KORO_END` macros

**When disabled (default):**
- Uses stackful coroutines with assembly context switching
- Each coroutine gets `KCORO_DEFAULT_STACK_SIZE` stack
- User code can call any C function (including blocking I/O)
- Requires platform-specific `kc_ctx_switch.S`

### Scheduler Mode (`KC_SCHED`)

**When enabled:**
- Full cooperative scheduler with work queues and work-stealing
- Required for multi-threaded arena operation
- Provides `kc_sched_spawn`, `kc_sched_drain` API

**When disabled:**
- Arena operates in single-threaded or externally-scheduled mode
- User must provide event loop integration

## Test Configuration Constants

These are *not* user-configurable; they are hardcoded in test files:

| Constant | Purpose | Typical Value | File |
|----------|---------|---------------|------|
| `PRODUCERS`, `CONSUMERS` | Test workload counts | 4 each | test_*.c |
| `PER_PRODUCER`, `MSGS_PER_PRODUCER` | Messages per producer | 100-20000 | test_*.c |
| `NUM_THREADS`, `COROS_PER_THREAD` | Concurrency stress levels | 4-100 | test_*.c |
| `STRESS_COUNT`, `MAX_ITEMS` | Iteration/capacity limits | 1000-2000 | test_*.c |

## Cross-Reference: Key Files

- **Platform abstractions**: `core/src/kc_platform.c`, `core/include/kcoro_platform.h`
- **Stackless API**: `core/include/kcoro_stackless.h`, `core/src/kc_stackless.c`
- **Scheduler config**: `core/src/kc_sched.c`
- **Channel capabilities**: `core/src/kc_chan.c`, line ~40-50 (capability flags)
- **Token kernel**: `core/src/kc_token_kernel.c`, `core/include/kcoro_token_kernel.h`

## Best Practices

1. **Always define `_GNU_SOURCE` and `_POSIX_C_SOURCE`** at the top of every `.c` file for consistent POSIX/Linux API availability.
2. **Use `KC_ALLOC`/`KC_FREE` macros** instead of raw malloc/free to allow future instrumentation or custom allocators.
3. **Guard platform-specific code** with `#ifdef __linux__` or `#if defined(__aarch64__)` as needed.
4. **Set `KCORO_STACKLESS_BUILD` only in embedded/memory-constrained targets** where 64KB per coroutine is prohibitive.
5. **Enable `KC_SCHED` for production** unless integrating with an external event loop.

## Future Additions

- `KCORO_VALGRIND_BUILD`: Annotate stack switches for Valgrind/Helgrind
- `KCORO_TSAN_BUILD`: ThreadSanitizer-compatible annotations
- `KCORO_CUSTOM_ALLOCATOR`: User-provided allocator hooks
- `KCORO_ZERO_SPIN_WORKERS`: Event-driven worker model (planned)

---

**Last Updated**: 2025-10-13  
**Maintainer**: llama.kotlin project
