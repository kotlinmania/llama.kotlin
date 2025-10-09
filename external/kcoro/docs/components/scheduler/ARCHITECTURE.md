# Scheduler — Architecture Notes (2025‑10‑09)

This note captures current invariants relevant to channels and rendezvous handshakes.

## TLS / `main_co` hand‑off
- Each worker thread owns a dedicated `main_co` created at start. Before resuming a task coroutine `co`, the worker sets:
  - `co->main_co = worker->main_co`
  - thread‑local `kc_sched_current()` to the owning scheduler
  - thread‑local `kcoro_set_thread_main(worker->main_co)`
- After `kcoro_resume(co)` returns, the worker restores bookkeeping and either re‑enqueues `co` or releases it depending on its state.

## Ready queue reference discipline
- Enqueue (`kc_sched_enqueue_ready`): retains `co` and marks `ready_enqueued=true` while pushing to the intrusive FIFO. The caller is responsible for one balancing release after enqueue.
- Dequeue: worker pops from the FIFO (clears `ready_enqueued`), then runs/reschedules/releases. Exactly one reference is held by the ready queue while `co` is runnable.

## Channel wake integration
- Channel wake paths (`kc_chan_wake_{send,recv}_locked`) retain `co` under the channel lock before disposing the waiter. The wake is then scheduled via `kc_chan_schedule_wake`, which performs the enqueue and a single balancing release. This keeps ownership intact from waiter pop until the scheduler takes over.

## Don’ts
- Do not auto‑create a default scheduler from `kcoro_unpark`; only enqueue if a scheduler is active on the current thread. Channels call into the scheduler explicitly when appropriate.

