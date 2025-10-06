# Token Kernel Integration Journal — 2025-10-06

Tonight was all about proof, not polish. We set out to thread the BizTalk-style correlation engine into kcoro and, in the process, we touched nearly every sharp edge of the runtime. Here’s the unvarnished story so far.

## Early sketches and false starts

1. **Token VM everywhere (first attempt).**
   - The plan was seductive: replace `kc_ctx_switch.S` with the token interpreter, measure the 11-cycle resumes, and never look back.
   - It “worked” in microbench; our 5M-iteration loop lit up the profiler with numbers we wanted to believe.
   - Reality: the moment we parked a real coroutine (`kcoro_park()`), we returned to garbage. `lab_simple_park_demo` crashed before printing “after park.” We were saving the caller’s LR from inside `kcoro_switch` and then returning to ourselves. Instant infinite tail recursion.
   - Conclusion: microbench throughput is meaningless if basic semantics fail. We parked the token VM switch for later and rolled back to the known-good assembly until we could prove correctness.

2. **Token VM apply stub edits.**
   - Repeated edits to `_kc_vm_apply` (move `sp`, load registers, `blr x1`, etc.). Every small tweak improved something but not enough. The crash persisted because the continuation pointer was still wrong.

3. **Breakpoint archaeology with lldb.**
   - Break on `_kc_vm_apply`, dump registers, realize LR and SP were being restored, but the return address kept pointing at `kcoro_switch` instead of the caller.
   - Attempted to read back `to_co->reg[13]` (LR) from LLDB and even simple `memory read -f A` commands failed—because we had momentarily “captured” a pointer that didn’t exist.

4. **`__builtin_return_address(1)` experiment.**
   - Tried to ask the compiler for the caller’s caller. Clang flagged it as unsafe (`-Wframe-address`). Makes sense: the optimizer may inline or tail-call-away the frames. Dropped the idea.

5. **Manual LR capture via inline asm.**
   - New idea: read LR in `kc_ctx_switch_vm.c` with `asm("mov %0, lr")`. That gave us a deterministic pointer to the call site. Still, the real issue remained: parking without a scheduler meant we had to deal with “returning to main” ourselves.

## Token kernel scaffolding

6. **Freelist + correlation table.**
   - Introduced `kc_token_kernel.c` with: (a) atomic counter for IDs, (b) bucketed hash table, (c) freelist guarded by a mutex for now. State machine: `NEW -> PARKED -> READY -> COMPLETED`.
   - Added coroutine payload fields (`token_payload_ptr`, `token_payload_len`, `status`) and a flag `token_payload_ready` so resumed coroutines can pull the event data.

7. **Callback path.**
   - `kc_token_kernel_callback(ticket, payload)` stores payload into the parked coroutine, sets the flag, and calls `kcoro_unpark(co)`; on cancel we mark status and also `kcoro_unpark`.

8. **Demo harness.**
   - `lab_token_kernel_demo`: coroutine publishes a ticket, parks, we simulate a callback, and the coroutine prints the payload and status. Great confirmation that the kernel control plane works, even before changing channels.
   - `lab_simple_park_demo`: sanity test to make sure `kcoro_resume`/`kcoro_park` still work. After reverting to assembly for the switch, this demo now prints all expected lines.

9. **Token kernel numbers (for the record).**
   - Token resume microbench (with interpreter) still hit ~109M resumes/sec (~9.1 cycles). Once the switch is stable, those numbers will be meaningful.

## Context switch post-mortem

10. **Why the crashes happened.**
    - `kc_vm_capture` saved LR into `reg[13]`, but `kcoro_switch` never populated it with the caller’s LR—only with the internal helper. Restoring those values spiraled back into `kcoro_switch` forever.
    - After reverting to the original assembly, determinism returned. To resurrect the token switch, we’ll need to capture LR before the helper call so that `reg[13]` points to the call site in `kcoro_resume`.

11. **Rolling back for stability.**
    - Removed `KC_VM_SWITCH` flag from the mirror build and resumed using `arch/aarch64/kc_ctx_switch.S`. Token-based switch files remain in the tree for future iteration once the functional work is done.

## What’s next (strictly without token switch)

12. **Integrate callback path into channels.**
    - Hook `kc_chan_send_ptr`, `kc_chan_recv_ptr`, and select clause helpers to publish tickets and park immediately instead of enqueuing waiters that require scheduler threads.

13. **Resume from callbacks.**
    - Teach channel wake paths to invoke `kc_token_kernel_callback(ticket, payload)` so parked coroutines wake via `kcoro_unpark` exactly like the demo.

14. **Benchmark harness.**
    - Update `bench_chan_metrics` to use the async API and re-run the throughput tests. Expect CPU near idle during waits—no spins, no worker threads.

15. **Reintroduce token switch (when ready).**
    - Once the channel flow is stable, revisit the token-based `kcoro_switch`. Capture LR via inline asm, write it to the coroutine’s register file, and have `kc_vm_apply` branch to it. Gate the experimental path behind a feature flag so we can compare both.

## Notes and observations

16. **LLDB tips gathered along the way:**
    - `register read x1` on AArch64 to check continuations.
    - `bt` after each `process continue` to confirm stack trails.
    - `expr *(void**)((uint64_t)$x0 + 0x68)` fails if the pointer is invalid—use `memory read` carefully.

17. **Plain assembly works (again).**
    - After the rollback, `lab_simple_park_demo` prints:
      ```
      [main] resuming
      [co] before park
      [main] first resume return
      [co] after park
      [main] second resume return
      ```
      That confirms the default behavior is intact.

18. **Token kernel demo output now:**
      ```
      [main] initializing token kernel...
      [main] first resume
      [coroutine] published ticket id=1, parking...
      [main] returned from first resume? (should not happen)
      [main] coroutine parked with ticket id=1
      [main] resuming coroutine after callback
      [coroutine] resumed rc=0 payload_ptr=0x... len=2 status=0
      [main] returned from resume
      ```
      The “returned from first resume?” line is expected in the absence of a scheduler we haven’t written yet. Once callbacks go through the channel code proper, this will become the new handshake point.

19. **Here’s what matters:**
    - Microbench wins mean nothing unless functional tests pass.
    - Running crash repros in LLDB (`b kc_vm_apply`, `register read`, `expr`) is invaluable.
    - Going deep (like tonight) fixes core issues permanently; the eventual token switch will be all the better for it.

---

That’s tonight’s brain dump. Next entry should capture the channel callback rotary, minimal scheduler removal, and the first meaningful benchmarks under the zero-spin design. Stay tuned.
