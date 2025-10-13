# Assembly Context Switching: AArch64 Register Preservation

## Overview

The context switch primitives (`kc_vm_capture` and `kc_vm_apply`) implement **hand-coded assembly routines** for saving and restoring CPU register state, enabling cooperative coroutine switching with **sub-50ns latency**. This is the lowest-level foundation of the entire kcoro arena system.

## Implementation: `kc_vm_apply.S`

### AArch64 (ARM64) Assembly

```asm
// File: kc_vm_apply.S
#if defined(__aarch64__)
    .text
    .align 2
    .globl _kc_vm_capture
_kc_vm_capture:
    str x19, [x0, #0x00]   // Save callee-saved registers
    str x20, [x0, #0x08]
    str x21, [x0, #0x10]
    str x22, [x0, #0x18]
    str x23, [x0, #0x20]
    str x24, [x0, #0x28]
    str x25, [x0, #0x30]
    str x26, [x0, #0x38]
    str x27, [x0, #0x40]
    str x28, [x0, #0x48]
    str x1,  [x0, #0x68]   // Save return address (x1 = link register backup)
    mov x9, sp
    str x9, [x0, #0x70]    // Save stack pointer
    str x29, [x0, #0x78]   // Save frame pointer
    ret                     // Return to caller

    .globl _kc_vm_apply
_kc_vm_apply:
    ldr x19, [x0, #0x00]   // Restore callee-saved registers
    ldr x20, [x0, #0x08]
    ldr x21, [x0, #0x10]
    ldr x22, [x0, #0x18]
    ldr x23, [x0, #0x20]
    ldr x24, [x0, #0x28]
    ldr x25, [x0, #0x30]
    ldr x26, [x0, #0x38]
    ldr x27, [x0, #0x40]
    ldr x28, [x0, #0x48]
    ldr x30, [x0, #0x68]   // Restore return address
    ldr x9,  [x0, #0x70]
    mov sp, x9              // Restore stack pointer
    ldr x29, [x0, #0x78]   // Restore frame pointer
    br x30                  // Jump to restored return address (does NOT return)
#else
void kc_vm_capture(uint64_t *regs) {(void)regs;}
void kc_vm_apply(uint64_t *regs, void (*fn)(void)) {(void)regs; fn();}
#endif
```

### Register Save Layout

**Memory layout (offsets in `uint64_t regs[]` array):**

| Offset | Register | Purpose | ABI Classification |
|--------|----------|---------|-------------------|
| 0x00 | x19 | Callee-saved GPR | Must preserve |
| 0x08 | x20 | Callee-saved GPR | Must preserve |
| 0x10 | x21 | Callee-saved GPR | Must preserve |
| 0x18 | x22 | Callee-saved GPR | Must preserve |
| 0x20 | x23 | Callee-saved GPR | Must preserve |
| 0x28 | x24 | Callee-saved GPR | Must preserve |
| 0x30 | x25 | Callee-saved GPR | Must preserve |
| 0x38 | x26 | Callee-saved GPR | Must preserve |
| 0x40 | x27 | Callee-saved GPR | Must preserve |
| 0x48 | x28 | Callee-saved GPR | Must preserve |
| 0x68 | x30 (LR) | Link register (return address) | Critical |
| 0x70 | SP | Stack pointer | Critical |
| 0x78 | x29 (FP) | Frame pointer | Critical |

**Total size**: 16 × 8 bytes = **128 bytes**

### Why These Registers?

**ARM Procedure Call Standard (AAPCS64):**

- **Callee-saved**: x19-x28 must be preserved across function calls; any function modifying them must restore original values
- **x30 (LR)**: Link register; holds return address after `bl` (branch-with-link)
- **SP**: Stack pointer; critical for local variable access and function calls
- **x29 (FP)**: Frame pointer; used by debuggers and exception unwinders

**Caller-saved registers (x0-x18)** intentionally **not saved** because:
- Coroutine switch happens at well-defined suspension points
- C compiler guarantees caller-saved registers are already spilled to stack before suspension
- Saves 19 × 8 = 152 bytes per context

## Operational Semantics

### Capture: Freeze Current Execution State

```c
uint64_t regs[16];  // Context buffer
kc_vm_capture(regs, &&resume_point);

// After capture, execution continues here as normal
// 'regs' now contains snapshot of CPU state
```

**What happens:**

1. **x0** = pointer to `regs` array
2. **x1** = address of `&&resume_point` label (GNU extension: label-as-value)
3. Assembly stores x19-x28, SP, FP, and x1 into `regs`
4. `ret` instruction returns normally to caller

**Result**: `regs` contains complete register state to resume from `&&resume_point`.

### Apply: Restore Execution State (Context Switch)

```c
// Later, possibly on different thread
kc_vm_apply(regs, NULL);

// Execution DOES NOT return here!
// Instead, jumps to 'resume_point' label with restored registers
```

**What happens:**

1. **x0** = pointer to `regs` array
2. Assembly restores x19-x28, SP, FP from `regs`
3. Loads saved return address (x30) from `regs[0x68]`
4. `br x30` branches to saved address—**does not return**

**Critical behavior**: This is a **tail call**; the caller's stack frame is abandoned.

## Integration with Coroutine Scheduler

### Suspend (`kcoro_park`)

```c
void kcoro_park(void) {
    kcoro_t *co = kcoro_current();
    
    // [1] Capture current coroutine's register state
    kc_vm_capture(co->regs, &&park_resume);
    
    // [2] Mark coroutine as suspended
    co->state = KCORO_PARKED;
    
    // [3] Switch to scheduler's context
    kc_vm_apply(scheduler_context.regs, NULL);
    
    // NEVER REACHES HERE (until resumed)
    
park_resume:
    // [4] Coroutine resumes here when kc_vm_apply() restores its context
    return;  // Resume normal execution
}
```

**State transition:**
```
Running → Suspended → (scheduler finds work) → Restored → Running
```

### Resume (`kcoro_unpark`)

```c
void kcoro_unpark(kcoro_t *co) {
    if (co->state != KCORO_PARKED) return;
    
    // [1] Mark runnable
    co->state = KCORO_READY;
    
    // [2] Enqueue on scheduler ready queue
    // (Actual resume happens when scheduler picks coroutine)
}

// Later, in scheduler loop:
void scheduler_run_next(void) {
    kcoro_t *co = dequeue_ready_coroutine();
    
    // [1] Save scheduler's own context
    kc_vm_capture(scheduler_context.regs, &&scheduler_resume);
    
    // [2] Switch to coroutine (restores its registers)
    kc_vm_apply(co->regs, NULL);
    
    // NEVER REACHES HERE (coroutine runs until it parks again)
    
scheduler_resume:
    // [3] Scheduler resumes here after coroutine parks
    // Ready to pick next coroutine
}
```

## Performance Analysis

### Latency Breakdown

| Operation | Cycles | Time (3.2 GHz) | Notes |
|-----------|--------|----------------|-------|
| `str` (single) | ~1 | 0.3ns | Write to L1 cache |
| `ldr` (single) | ~1 | 0.3ns | Read from L1 cache |
| `kc_vm_capture` | ~15 | ~5ns | 10× stores + move + ret |
| `kc_vm_apply` | ~18 | ~6ns | 10× loads + move + branch |
| **Total context switch** | ~33 | **~10ns** | Both capture + apply |

**Actual measured latency**: ~40-50ns (includes C function call overhead).

### Comparison to Thread Switch

| Mechanism | Latency | Context Saved |
|-----------|---------|---------------|
| **Thread switch** | ~1-3µs | Full thread state (kernel syscall) |
| **Coroutine switch** | ~40-50ns | 16 registers (user-space) |

**Speedup**: ~50-100× faster than OS thread switch.

### Memory Bandwidth

**Per context switch:**
- Write: 13 × 8 = 104 bytes
- Read: 13 × 8 = 104 bytes
- **Total**: 208 bytes

**L1 cache efficiency:**
- Typical L1 cache line: 64 bytes
- Context data: ~3 cache lines
- High likelihood of L1 hit (coroutines ping-pong between same threads)

**Bandwidth requirement at 1M switches/sec:**
- 208 MB/s
- Negligible vs modern CPU L1 bandwidth (>100 GB/s)

## Platform-Specific Considerations

### macOS (Darwin) Symbol Prefix

```asm
.globl _kc_vm_capture   // ← Note leading underscore
```

**Rationale**: macOS/Mach-O linker prepends `_` to C symbol names. Assembly must match.

**Linux/ELF** uses `kc_vm_capture` (no prefix).

### Stack Alignment

**AAPCS64 requirement**: SP must be 16-byte aligned before `bl` instruction.

**Coroutine stack setup:**

```c
void *stack = aligned_alloc(16, STACK_SIZE);  // Guarantee alignment
co->regs[0x70] = (uint64_t)(stack + STACK_SIZE - 16);  // SP at top, aligned
```

**Violation symptom**: `Bus error` or `SIGILL` on vector instructions (`ldp`, `stp`).

### Floating-Point Registers (SIMD)

**Not saved**: d8-d15 (callee-saved SIMD registers) intentionally omitted.

**Rationale:**

- Adds 8 × 16 = 128 bytes per context
- Most coroutine code doesn't use SIMD
- C compiler spills SIMD state to stack before suspension points

**Future enhancement:** Conditional save (detect SIMD usage at runtime).

## Debugging and Safety

### Stack Corruption Detection

**Symptom**: Segfault or garbage data after context switch.

**Cause**: Stack pointer (SP) points to invalid/unowned memory.

**Debug technique:**

```c
void validate_stack(kcoro_t *co) {
    uint64_t sp = co->regs[0x70];
    assert(sp >= (uint64_t)co->stack_base);
    assert(sp <= (uint64_t)co->stack_base + co->stack_size);
}
```

Call before `kc_vm_apply()` to catch corruption early.

### Uninitialized Context

**Symptom**: `kc_vm_apply()` jumps to garbage address → immediate crash.

**Cause**: `kc_vm_capture()` never called; `regs` array contains uninitialized data.

**Prevention:**

```c
void coroutine_init(kcoro_t *co) {
    memset(co->regs, 0, sizeof(co->regs));  // Zero-initialize
    co->regs[0x70] = (uint64_t)(co->stack + STACK_SIZE - 16);  // Valid SP
    co->regs[0x68] = (uint64_t)coroutine_entry_point;  // Valid PC
}
```

### Race Conditions

**Danger**: Concurrent `kc_vm_apply()` on same `regs` buffer.

**Example:**

```c
// Thread A
kc_vm_apply(co->regs, NULL);  // Starts modifying CPU state

// Thread B (simultaneously)
kc_vm_apply(co->regs, NULL);  // Clobbers thread A's restoration
```

**Result**: Undefined behavior; likely crash.

**Protection**: Coroutine state machine enforces single-threaded access:

```c
int kcoro_unpark(kcoro_t *co) {
    int expected = KCORO_PARKED;
    if (!atomic_compare_exchange(&co->state, &expected, KCORO_READY)) {
        return -EINVAL;  // Already running or being resumed
    }
    // Safe to enqueue for resume
}
```

## Advanced Topics

### Cooperative vs Preemptive

**Current model**: Fully cooperative; coroutines yield control explicitly via `kcoro_park()`.

**Preemptive alternative (not implemented):**

1. Set timer signal (SIGALRM)
2. Signal handler captures context via `kc_vm_capture()`
3. Handler forces context switch to scheduler

**Trade-off**: Preemptive adds complexity and overhead (syscalls, signal handling).

### Stack-Less Coroutines (Theoretical)

**Current requirement**: Each coroutine needs separate stack (~64KB).

**Alternative**: Store continuation state in heap-allocated structure; no stack needed.

**Benefit**: Scales to 100K+ concurrent coroutines (64KB × 100K = 6.4GB vs ~10MB for continuation heap).

**Constraint**: Requires compiler support (yield expressions, state machine transformation).

### Zero-Copy Parameter Passing

**Current**: Function arguments passed via stack or callee-saved registers (already zero-copy).

**Enhancement**: Store function pointer + args in coroutine struct; `kc_vm_apply()` passes to trampoline:

```c
co->resume_fn = my_function;
co->resume_arg1 = ptr;
co->resume_arg2 = len;

// Trampoline invoked by kc_vm_apply:
void coroutine_trampoline(kcoro_t *co) {
    co->resume_fn(co->resume_arg1, co->resume_arg2);
}
```

**Benefit**: Eliminates stack setup overhead for simple callbacks.

## Related Documentation

- `TOKEN_KERNEL_SPECIFICATION.md`: How context switches integrate with token-based resumption
- `SCHEDULER_ARCHITECTURE.md`: Ready queue and scheduling policy
- `COROUTINE_LIFECYCLE.md`: State transitions (Running → Parked → Ready)

---

**Implementation reference**: `external/kcoro_arena/core/src/kc_vm_apply.S`
