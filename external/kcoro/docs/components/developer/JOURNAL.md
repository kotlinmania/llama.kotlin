# kcoro Engineering Journal

Chronological notes capturing design decisions, implementation steps, and validation results. This journal complements PLAN.md and the design docs.

## 2025‑09‑22 — zref-only backend, tests green, docs polish

- Removed the separate ptr backend; unified on a single “zref” backend that implements rendezvous and queued descriptor paths.
- Canonicalized on descriptor APIs (`kc_chan_send_desc/_recv_desc` + `_c` variants); `*_ptr` / `*_zref` functions are thin wrappers that compose `kc_zdesc`.
- Purged compile‑time debug forks (`KCORO_DEBUG_SCHED`, `ZREF_DBG*`) and fossil `#if 0` blocks; kept runtime `KCORO_DEBUG` logging.
- Fixed pointer rendezvous lifecycle in tests via deterministic drain/join; default test suite is green across buffered/rv/zref/timeout/scheduler.
- Docstring pass across headers and kc_zcopy.c (2–3 line explanations for constants; clear file‑level prologues). Added OS‑neutral prologue to `kcoro_port.h`.
- Updated PLAN.md with a status section and focused N1/N2 next steps.

## 2025‑09‑21 — Zero‑Copy Surface Unified; Monitor Uses Real Stats (update)

- Centralized zero‑copy under a neutral descriptor (`kc_zdesc`) and backend vtable in `kcoro_zcopy.h`.
- Implemented backend factory and a single built‑in backend in `kc_zcopy.c`:
  - `zref` unified backend:
    - Rendezvous: hand‑to‑hand
    - Buffered/Unlimited/Conflated: queued descriptor path (former ptr path)
- `kc_chan.c` pointer‑first ops update bytes/ops via shared helpers; rendezvous routes through backend.
- kcoro_chanmon now reads `kc_chan_snapshot()` and uses `kc_chan_compute_rate()` to show true PPS/GBPS and failure deltas. Headless NDJSON output added.
- Added “pre‑kill” hook before builds/tests to eliminate lockups from lingering test binaries; wrapped each test with a timeout; heavy stress kept under `test-full` with a longer timeout.
- Tests passing in default suite: buffered pointer basic; zref basic; zref timeout; failure counters; scheduler basic.
- Ptr rendezvous test: still failing on length assertion; root cause narrowed to rendezvous wake + capability bit; fixed recv wake (pop from `wq_recv_head`) and set ZERO_COPY bit on backend bind; remaining issue under investigation.
- Performance spot‑check (local): headless chanmon with P=1/C=1, N=20000, size=512 shows ~78 Gbps sustained (single‑process synthetic scenario).

Next:
- Fix pointer rendezvous waiter lifecycle (single free owner in wake helpers); re‑enable pointer RV test by default.
- Document adapter guidelines and region API expectations.
- Optional: CI guard against non‑POSIX or kernel headers in core; sanitizer run for waiter misuse.

## 2025‑09‑20 — Metrics & Monitor Migration

- Refactored monitor to use snapshot + rate helper instead of ad‑hoc loops.
- Ensured per‑op counters update on every success (ops and bytes) so deltas are precise.
- Added failure counters to the snapshot; monitor displays deltas alongside PPS/GBPS.

---
Notes
- Dates reflect repository local time. Entries summarize the primary user‑visible changes plus engineering guardrails added for stability.
