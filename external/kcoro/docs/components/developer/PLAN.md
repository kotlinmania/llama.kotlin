# kcoro Plan

## Current Focus (2025-10-05)

- **Match upstream rendezvous lifecycle:** Mirror the retain/release ordering observed in the reference native coroutine runtime (`BufferedChannel.send/receive`, `CancellableContinuationImpl`) so coroutines remain owned until dequeue completes.
- **Waiter cancellation parity:** Install per-cell cancellation handlers that perform the same `Segment.onCancellation` flow (CAS to `INTERRUPTED_*`, wait for expand completion, invoke undelivered hooks).
- **Scheduler TLS hand-off:** Align worker TLS updates with the reference dispatcher pattern (`main_co` swap, continuation restore before release) to stop racey releases.
- **Documentation alignment:** Capture the runtime-derived invariants in the architecture docs (channels + scheduler) so future changes stay in lock-step with the reference implementation.

## Next Actions

1. **Channel wake/retain patch** (in-flight): clarified and verified: channel wake paths retain under lock before waiter disposal; scheduler enqueue takes ownership and a single balancing release happens in `kc_chan_schedule_wake`. Ready queue owns exactly one reference until dequeue.
2. **Cancellation hook wiring:** scaffolding added to `kc_waiter` for an optional `cancel` token (best-effort). Next step: provide `kc_waiter_install_cancel()` and poll in retry loops; consider callback registration once token supports listeners.
3. **Scheduler TLS audit:** documented actual behavior in `docs/components/scheduler/ARCHITECTURE.md` and validated worker path (`worker_main` sets `tls_current_sched`, installs `main_co`, resumes, then re-enqueues or releases).
4. **ASan/TSan regression suite:** SAN switch added to common.mk (`SAN=asan|tsan`). ASan builds run; RV tests still reproduce stalls (see issue #81). Keep enabled for further debugging.
5. **Docs & code comments:** keep `CHANNELS_ALGORITHM.md` and `scheduler/ARCHITECTURE.md` in sync with the findings (already started in this iteration). Reference the native sample/disassembly path (`tools/kotlin-native-samples/chan`).

## Watch List / Follow Ups

- Evaluate whether additional assembly changes are required after we align C code. Current data shows the ISA emitted by the upstream toolchain is informative, but we expect to stay in C for fixes.
- Track upstream native runtime releases for allocator or coroutine lifecycle changes that could impact parity checks.
- Consider scripting the sample build/disassembly so we can diff upstream output as part of CI when that toolchain bumps.
