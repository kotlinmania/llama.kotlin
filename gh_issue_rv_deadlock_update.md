Status update: not closed yet — cross‑wake reduces but does not eliminate the rendezvous stall

Summary of work since opening
- Added channel tracing behind KCORO_TRACE to log every rendezvous state change (send/recv enter, enqueue, has_value toggles, wakes, and queue heads) without polluting stdout.
- Implemented a minimal handshake tweak: after enqueueing on the rendezvous path, cross‑wake the opposite side (sender wakes receiver; receiver wakes sender) if the slot is empty. This is intended to prevent both queues being populated with has_value==0 and no future wake.
- Rebuilt and stress‑tested `external/kcoro/tests/test_chan_rv_metrics` and `test_chan_ptr_rendezvous_basic` with Koru (macOS arm64).

Results
- The original stall signature is visible in traces prior to the patch: both `wq_send_head` and `wq_recv_head` non‑NULL while `has_value==0`, leading to scheduler drain timeout.
- With the cross‑wake tweak, the exact “both queues + empty slot” snapshot is rarer, but the tests still fail:
  - `test_chan_rv_metrics`: timeouts with mismatched send/recv totals and occasional early `closed=1` state; producers observe `KC_EPIPE` before all sends complete.
  - `test_chan_ptr_rendezvous_basic`: hangs part‑way (e.g., sends=5 recvs=4) and aborts after a fixed wait loop.
- Trace snippets show wakes being scheduled, but the system can still converge to a non‑progressing state (no one sets `has_value`); in other runs the channel flips to `closed=1` early due to teardown while parked waiters remain.

Representative outputs
- rv metrics example (after patch):
```
[rv-metrics] mismatch sends=17 recvs=16 expected=800
[rv-metrics] producer send failed rc=-32 at i=2
```
- pointer rendezvous example (after patch):
```
[ptr rv][debug] sends=5 recvs=4 ready=0 has_value=1 send_wait=0x... recv_wait=0x...
(then loop stalls and aborts)
```

Interpretation
- The cross‑wake-after‑enqueue step narrows one deadlock window but is not sufficient. There are still interleavings where both sides end up parked or the intended counterpart does not resume to publish/consume.
- PLAN.md items under “Current Focus” likely apply:
  1) Wake/retain ordering: ensure the waiter/co is retained through resume and only released after removal from the queue; otherwise the wake can be lost or a resumed coroutine may race with disposal.
  2) Waiter cancellation parity: install per‑waiter cancellation handlers mirroring upstream patterns so parked waiters are reliably removed and signaled.
  3) Scheduler TLS hand‑off alignment: avoid racey releases around resume/restore that can lead to early close or missed progress.
- A more robust handshake option is a pop‑first policy in rendezvous: when receiver finds a waiting sender (or vice‑versa) it avoids enqueue and prioritizes a direct handoff path. Our current slot‑based publish requires the sender to run to set `has_value`; we should guarantee that wake is scheduled under the same lock and that the woken sender cannot be preempted before publishing.

Next steps (proposed)
- Implement wake/retain fix: in `kc_chan_schedule_wake` and the rendezvous wake paths, retain the target until resume is confirmed; release only after dequeue completes (PLAN.md step 1).
- Add per‑waiter cancellation hooks matching upstream semantics (PLAN.md step 2).
- Strengthen rendezvous pairing: avoid enqueue if the opposite queue is non‑empty; instead, schedule a deterministic wake under lock and loop until `has_value` is observed or closed.
- Re‑run `test_chan_rv_metrics` and `test_chan_ptr_rendezvous_basic` under ASan/TSan and attach traces.

Conclusion
- Leaving this issue OPEN. The deadlock frequency is reduced, but the rendezvous path still fails under stress and pointer RV basic tests. I will follow up with the wake/retain + cancellation parity changes and a pop‑first rendezvous policy patch, then post green test results.
