Rendezvous channel deadlock when receiver enqueues while sender waiting

Summary
kcoro rendezvous channels can hang when the receiver path enqueues itself even though a sender is already waiting (or arrives concurrently). In this state `has_value == 0` while both wait queues are populated, so neither side advances and the scheduler never drains.

Repro
```
cd external/kcoro && make tests
cd external/kcoro/tests
# optional tracing (added locally): KCORO_TRACE=$(pwd)/kcoro_trace.log
./build/test_chan_rv_metrics
```

Observed
- The test times out in `kc_sched_drain`.
- Thread snapshots show producers/consumers blocked inside `kc_chan_send`/`kc_chan_recv`.
- Channel snapshot at timeout:
  - `has_value == 0`
  - both `wq_send_head` and `wq_recv_head` are non-null (wait-queue counters are hints; the queue pointers tell the story)

Example trace line near the stall (KCORO_TRACE):
```
[chan] ... has_value=0 ... send_head=0x600000c70000 recv_head=0x600000c70800 closed=0
```

Root cause hypothesis
- In the rendezvous path, `kc_chan_recv` can enqueue the receiver even when a sender is already waiting (or races in), leaving:
  - both queues non-empty, and
  - `has_value == 0` (slot empty),
  which deadlocks because no subsequent wake is scheduled and every participant is parked.

Minimal fix idea (handshake tweak)
- After enqueuing on a rendezvous channel, cross‑wake the opposite side if present and the slot is empty. This ensures the last enqueuer nudges a counterpart to perform the handoff.
  - In `kc_chan_send` (RV, blocking): after appending to `wq_send`, if `wq_recv_head && !has_value`, call `kc_chan_wake_recv_locked` and schedule that wake before yielding.
  - In `kc_chan_recv` (RV, blocking): after appending to `wq_recv`, if `wq_send_head && !has_value`, call `kc_chan_wake_send_locked` and schedule before yielding.
- Alternative: pop‑first or keep the receiver “hot” until sender runs, but the cross‑wake-after‑enqueue route is minimal and keeps semantics.

Notes
- Environment: macOS (arm64), but the logic appears platform‑independent.
- The behavior reproduces reliably with 4 producers + 4 consumers in `test_chan_rv_metrics.c` and also in a small pointer‑RV test.

What success looks like
- `./build/test_chan_rv_metrics` completes without timeout; final snapshot shows `waiters_send == 0`, `waiters_recv == 0`, and `kc_sched_drain` returns 0.

Happy to attach the trace or open a PR with the cross‑wake change if helpful.
