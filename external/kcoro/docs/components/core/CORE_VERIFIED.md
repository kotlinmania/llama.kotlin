# Coroutine Core — Verified Snapshot

_Last reviewed: 2025-10-14_

This document reflects the stackful kcoro implementation shipped under `external/kcoro/core/src`. Update it alongside code or test changes.

## 1) Coroutine Core (kcoro_core.c, kcoro_core.h)

### Structure (see `include/kcoro_core.h`)
- `reg[32]`: register save area used by the ARM64 switcher.
  - Indices: x19..x28 → `reg[0..9]`, LR(x30) → `reg[13]`, SP → `reg[14]`, FP(x29) → `reg[15]`.
- `state`: `KCORO_CREATED | KCORO_READY | KCORO_RUNNING | KCORO_SUSPENDED | KCORO_PARKED | KCORO_FINISHED`.
- `fn`, `arg`: entry function and argument.
- `id`: unique 64-bit id (increments via `__sync_fetch_and_add`).
- `main_co`: coroutine to yield back to (the worker’s main coroutine).
- `scheduler`: owning `kc_sched_t*` when scheduled.
- `stack_ptr` / `stack_size`: private stack allocated with `mmap`; 16-byte aligned SP.
- `next` / `prev`: ready-queue links; `name`: optional string for debugging.

### Lifecycle at a glance
- `kcoro_create(fn,arg,stack)`: allocates struct, mmaps a private stack (default 64 KiB if 0), aligns SP, and seeds `reg[13]=kcoro_trampoline`, `reg[14]=SP`, `reg[15]=FP`. State = CREATED.
- `kcoro_resume(co)`: switches from current to `co` via `kcoro_switch`. Caller transitions RUNNING→SUSPENDED; target becomes RUNNING. Returning restores the previous coroutine and marks it RUNNING.
- `kcoro_yield()`: switches back to `main_co`; current becomes SUSPENDED, main becomes RUNNING, then resumes later.
- `kcoro_yield_to(target)`: direct yield to another coroutine (rare; scheduling typically uses queues).
- `kcoro_park()`: mark current PARKED and switch to main; later `kcoro_unpark` sets READY and enqueues if a scheduler is active.
- `kcoro_destroy(co)`: unmaps the private stack and frees the struct. Scheduler-owned coroutines are destroyed by the scheduler after completion.
- `kcoro_current()`: TLS pointer to current coroutine; `kcoro_create_main()`: constructs the special main coroutine per worker thread.

### Trampoline & protector
- `kcoro_trampoline`: internal entry that calls `fn(arg)`, marks FINISHED, and switches back to main. A protector aborts if user code returns past the trampoline unexpectedly.
- `kcoro_funcp_protector_asm`: assembly stub that preserves LR while calling `kcoro_funcp_protector` (diagnostics).

## 2) Refcount / retire queue (lifecycle hardening)

Goal: avoid use-after-free races observed in stress runs when one thread frees a coroutine while another still references it.

Plan:
1. Add an atomic refcount to `kcoro_t`. Owners (scheduler enqueue, worker resume, retire queue) increment while holding a reference and decrement when done.
2. Replace immediate frees with a retire queue: instead of destroying inside the worker return path, push `kcoro_t*` onto a per-scheduler retire list protected by a mutex.
3. A cleanup stage drains the retire list and calls `kcoro_destroy` only when the refcount reaches zero. This guarantees no worker holds a pointer to freed memory during concurrent resumes.
4. Special-case main coroutines to maintain a permanent reference while their hosting thread is alive.

**Implementation notes**
- Fast resume/yield paths remain lightweight (only atomic refcount ops); heavier reclamation work is deferred to the cleanup path.
- In debug builds, assertions ensure `refcount >= 1` for any coroutine reachable from queues; negative refcounts or double-frees assert.

**Diagnostic validation notes**
- Repro: headless monitor + ASan harness once exposed UAF in the ready-queue path.
- Validation: after converting the ready queue to intrusive links and adding the retire/refcount scheme, repeated runs produced no ASan reports; addresses remained stable and reuse patterns were clean.

## 3) Context Switch Integration (`kc_ctx_switch.S` quick view)

- `kcoro_switch(from,to)` saves callee-saved regs and SP into `from->reg[]`, restores `to->reg[]`, and branches to `to->reg[13]` (continuation).
- Only callee-saved GPRs are touched; FP/SIMD state is not saved in the current ARM64 path.
- Stack alignment and calling convention are respected; SP is restored exactly as saved.

## 4) State Machine & invariants

States: CREATED → READY → RUNNING → (SUSPENDED | PARKED) → RUNNING → FINISHED.

Invariants:
1. A parked coroutine is enqueued only after state is set to READY.
2. Coroutines are resumed by the thread that owns their `main_co`.
3. Destroy happens after completion and only when no other owner holds a reference (retire queue policy).

## 5) API surface (selected)

- `kcoro_create` / `kcoro_destroy`
- `kcoro_resume` / `kcoro_yield` / `kcoro_yield_to`
- `kcoro_park` / `kcoro_unpark`
- `kcoro_current` / `kcoro_create_main`

## 6) Portability & security notes

- Control logic lives in C; only the register swap is arch-specific. Other architectures can reuse the core by providing the switch primitive.
- Debug builds may enable stack guard pages to detect overflow early.

## Performance targets

- Coroutine spawn (empty body): ≤ 2.5 µs (stretch 1.2 µs).
- Memory per parked coroutine (including stack): ≤ 8 KB (stretch 4 KB).
