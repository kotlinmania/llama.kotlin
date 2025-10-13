# kcoro Channel Implementation Gaps

**Status**: Analysis of what standard coroutine channels provide that we don't yet implement

## What We Have (Aligned)

### Core Operations
- ✅ `send(timeout)` / `recv(timeout)` – blocking with timeout
- ✅ `try_send` / `try_recv` – non-blocking (timeout=0)
- ✅ `send_c` / `recv_c` – cancellable variants
- ✅ Channel kinds: RENDEZVOUS, BUFFERED, UNLIMITED, CONFLATED
- ✅ Waiter state machine (INIT → ENQ → CLAIMED → CANCELLED)
- ✅ Single-winner commit protocol (claim + commit before wake)
- ✅ onUndelivered hooks (copy + ptr variants)
- ✅ Prompt cancellation filtering at wake
- ✅ Close semantics: allow staged value to flow after close
- ✅ Zero-copy pointer channels (`send_ptr` / `recv_ptr`)
- ✅ Zero-ref arena channels (`send_zref` / `recv_zref`)
- ✅ Metrics and snapshots

## What We're Missing (Gaps)

### 1. Buffer Overflow Strategies
Standard channels support three overflow modes when buffer is full:
- **SUSPEND** (default): block sender until space (we have this)
- **DROP_OLDEST**: discard oldest element and add new (we don't have)
- **DROP_LATEST**: discard incoming element (we don't have)

**Our status**: We only support SUSPEND. CONFLATED is close to DROP_OLDEST but not exactly the same.

**Impact**: Medium. Some use cases need fire-and-forget with bounded memory.

---

### 2. Explicit Result Types
Standard API distinguishes:
- `send()` throws on close → **ClosedSendChannelException**
- `receive()` throws on close → **ClosedReceiveChannelException**
- `trySend()` returns **ChannelResult<Unit>** (success | closed | failure)
- `tryReceive()` returns **ChannelResult<E>** (success(value) | closed | failure)
- `receiveCatching()` returns **ChannelResult<E>** without throwing

**Our status**: We return errno codes (EPIPE, EAGAIN, ETIME). No discriminated union type for "closed with cause" vs "empty/full".

**Impact**: Low for C API. If we add higher-level bindings, we'd want richer result types.

---

### 3. Iterator API
Standard channels provide:
```kotlin
for (x in channel) { ... }
```
via `iterator()` and `hasNext()`.

**Our status**: No iterator. Users must loop `recv()` until EPIPE.

**Impact**: Low. Ergonomic sugar; not critical for C.

---

### 4. Broadcast Channels (Deprecated)
Standard library had `BroadcastChannel` (now deprecated in favor of `SharedFlow`).

**Our status**: Not implemented. We'd need pub-sub semantics, not 1:1 channels.

**Impact**: Out of scope for now. Different primitive.

---

### 5. Close with Cause
Standard channels can `close(cause: Throwable?)` and propagate the exception to waiters.

**Our status**: `kc_chan_close()` takes no cause. We return EPIPE generically.

**Impact**: Medium. Useful for debugging: "why was this closed?"

**Plan**: Add optional `int kc_chan_close_ex(ch, int reason_code)` and store `close_reason` in the channel struct. Wake waiters with that code instead of generic EPIPE.

---

### 6. Select with Multiple Channels
Standard `select` can wait on multiple channel clauses (send/recv) atomically.

**Our status**: We have `kc_select` infrastructure and single-winner commit (`kc_select_try_complete`). We register send/recv clauses and use claim/commit on the winner.

**Gaps**:
- No convenience API for "select first ready channel from a set"
- No `onSend`/`onReceive` builder syntax (C doesn't have lambdas, so we'd expose a lower-level registration API)

**Impact**: Medium. Users can build multi-channel select on top of our primitives, but it's manual.

**Plan**: Add `kc_select_register_send(sel, ch, msg)` and `kc_select_register_recv(sel, ch, buf)` for N channels, then `kc_select_wait(sel)` returns the winning clause index.

---

### 7. Offer/Poll (Deprecated Names)
Standard library had `offer()`/`poll()` (now `trySend`/`tryReceive`).

**Our status**: We have `kc_chan_try_send` / `kc_chan_try_recv` (inline wrappers for timeout=0). Naming aligned with modern API.

**Impact**: None. We're good.

---

### 8. Atomics and Memory Ordering
Standard implementation uses lock-free structures with careful memory ordering (FAA indices, CAS on cell states).

**Our status**: Mutex-based. All channel ops under `KC_MUTEX_LOCK(&ch->mu)`.

**Impact**: Performance. Lock contention under high concurrency.

**Plan**: Future. Our commit protocol is sound; we can swap in lock-free queues later without changing the API.

---

### 9. Dispatcher Integration
Standard channels integrate with coroutine dispatchers (resume on specific thread pool).

**Our status**: `kc_sched` is a basic multi-worker scheduler. `kc_chan_schedule_wake(wake)` enqueues on any worker. No affinity or priority.

**Impact**: Medium for real-world use. E.g., "resume UI coroutines on main thread."

**Plan**: Add `kc_sched_submit_to(sched, worker_id, co)` for affinity. Expose worker count and let users pin.

---

### 10. ReceiveCatching and Closed Detection
Standard `receiveCatching()` returns `ChannelResult.closed` or `ChannelResult.success(value)`.

**Our status**: `recv()` returns 0 on success, -EPIPE on closed. No way to distinguish "closed normally" vs "closed with error" at the API level.

**Impact**: Low for C. If we want it: add `kc_chan_recv_result(ch, out, result_t*)` where `result_t` is `{ int rc; int close_reason; }`.

---

## Summary Table

| Feature | Status | Priority | Effort |
|---------|--------|----------|--------|
| Buffer overflow (DROP_*) | Missing | Medium | Medium |
| Close with cause | Missing | Medium | Low |
| Select multi-channel API | Partial (infra exists) | Medium | Medium |
| ChannelResult types | Missing (use errno) | Low | Low |
| Iterator API | Missing | Low | Low |
| Lock-free queues | Missing (use mutex) | Low (perf) | High |
| Dispatcher affinity | Missing | Medium | Medium |
| Broadcast/SharedFlow | Out of scope | N/A | High |

---

## Recommended Next Steps

### Immediate (Closes Known Bugs)
1. ✅ **Rendezvous correctness**: Single-winner commit, no wake-and-hope → **DONE** (recent commits)
2. ✅ **Drain semantics**: Wait for alive==0 → **DONE**
3. ✅ **Close flow**: Allow staged value to be consumed → **DONE**
4. 🔲 **Stress testing**: Run `test_chan_rv_metrics` 100x; verify 100/100 pass

### Short-term (API Completeness)
5. 🔲 **Close with reason**: `kc_chan_close_ex(ch, reason_code)` + store in `ch->close_reason`
6. 🔲 **Select multi-channel**: Add `kc_select_register_{send,recv}(sel, ch, ...)` and `kc_select_wait(sel, out_clause_index)`
7. 🔲 **Buffer overflow modes**: Add `KC_DROP_OLDEST` and `KC_DROP_LATEST` flags; adjust enqueue logic

### Medium-term (Performance & Robustness)
8. 🔲 **Lock-free channel**: Implement FAA-based BufferedChannel (research from 2022 paper patterns)
9. 🔲 **Dispatcher affinity**: `kc_sched_submit_to(sched, worker_id, co)`
10. 🔲 **Metrics export**: JSON/Prometheus format for production observability

---

**Last updated**: 2025-10-13  
**Related issues**: See GitHub issues for kcoro rendezvous parity, select improvements, and arena zero-copy integration
