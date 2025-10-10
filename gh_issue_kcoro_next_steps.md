kcoro rendezvous: remaining work — direct handoff, ptr tracing, cancellation parity, SAN sweep

Context
- Follow‑up to #81 (rendezvous deadlock). We landed instrumentation and initial handshake tweaks in commits 3d3eaefe and 3f7ee75e, but RV tests still expose stalls in pointer rendezvous and rv_metrics stress.
- This issue tracks the scoped remaining work to finish stabilizing base kcoro’s rendezvous path before we mirror the invariants into the arena/token design.

Status snapshot (2025‑10‑09)
- Cross‑wake after enqueue and pop‑first bias on recv are in.
- Sender‑first publish for infinite waits (copy + ptr) is in.
- Wake/retain discipline + scheduler TLS hand‑off documented and verified.
- SAN build toggles added (common.mk: `SAN=asan|tsan`).
- Still observed under tests:
  - `test_chan_ptr_rendezvous_basic`: progress halts early (e.g., sends≈3, recvs≈2), with `has_value=1` and no further `recv_match` observed.
  - `test_chan_rv_metrics`: early mismatch (e.g., sends≈21, recvs≈20) and producer `KC_EPIPE`.

Proposed remaining tasks (checklist)
- [ ] Pointer‑path tracing: add KCORO_TRACE events in `kc_chan_send_ptr` / `kc_chan_recv_ptr` mirroring copy path (enter/enqueue/publish/match/wake).*Owner: @me*
- [ ] Direct handoff on recv (pop‑first, copy + ptr):
  - When `has_value==0` and a sender is queued, pop the sender and complete the transfer entirely inside `kc_chan_recv{,_ptr}` under the lock (publish into slot if needed, then immediately consume; wake the popped sender). Keep cross‑wake after enqueue as a fallback.*Owner: @me*
- [ ] Re‑run RV tests with trace:
  - Commands: `cd external/kcoro && make tests && cd tests && KCORO_TRACE=$(pwd)/kcoro_trace.log ./build/test_chan_ptr_rendezvous_basic` and `./build/test_chan_rv_metrics`.
  - Attach before/after trace snippets showing no "both queues populated" stalls.*Owner: @me*
- [ ] Cancellation parity (phase 1): introduce `kc_waiter_install_cancel()`; poll token in bounded loops to return `KC_ECANCELED`. Leave listener callback wiring for phase 2.*Owner: @me*
- [ ] Debug guards: dev‑only assert/log if `has_value==0 && wq_send_head && wq_recv_head` persists for N cycles; add counters for publish→consume progress.*Owner: @me*
- [ ] ASan/TSan sweep: ensure tests link with sanitizer flags; run both RV tests under SAN and fix any surfacing issues.*Owner: @me*
- [ ] Docs + closure: finalize CHANNELS_ALGORITHM pop‑first/hand‑off section; post traces and green runs; close #81 with a pointer here or close this with a pointer to #81 (whichever you prefer).*Owner: @me*

References
- #81 — Rendezvous channel deadlock when receiver enqueues while sender waiting
- Commits: 3d3eaefe (kcoro RV tracing + handshake tweaks), 3f7ee75e (staged remaining local edits)

Notes
- The immediate goal is to get both `test_chan_ptr_rendezvous_basic` and `test_chan_rv_metrics` green consistently under normal and SAN builds, then promote them to default in external/kcoro/tests/Makefile.
