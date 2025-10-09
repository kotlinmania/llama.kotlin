# Test Case 05 — Kotlin Interop Channel Soak Test

## Purpose
Demonstrate that the Kotlin expect/actual bindings to kcoro channels remain stable under prolonged load, catching deadlocks observed during prior macOS interop runs.

## Preconditions
- Kotlin/Native binding (`KcoroInterop`) compiled and linked against kcoro core.
- macOS arm64 hardware or simulator with kcoro enabled.
- Kotlin test harness capable of launching coroutines with timeout guards.

## Steps
1. Initialize a kcoro rendezvous channel via `KcoroInterop.createChannel`.
2. Launch 64 Kotlin coroutines that repeatedly send 1 KiB payloads (some zero-copy, some copy) for 60 seconds.
3. Launch matching receiver coroutines consuming messages with `withTimeout` to detect stalls.
4. Track kcoro metrics via interop API after each minute (matches, cancels, arena utilization).
5. After soak period, tear down the channel and verify arena cells return to `EMPTY`.
6. Repeat with progressive load (256 coroutines) and random cancellation/resume to stress cancellation paths.

## Expected Results
- No coroutine hits the timeout (i.e., no hang). If timeout occurs, capture dump and fail test.
- kcoro metrics equal total sends/receives, with zero leaked arena cells.
- Kotlin memory usage remains bounded; no runaway native allocations.
- Channel destruction completes without dangling tokens or crashes.
