# kcoro rendezvous: Kotlin parity analysis — test structure aligned, channel implementation bugs remain

## Context
Follow-up to issues documented in `gh_issue_rv_deadlock.md`, `gh_issue_rv_deadlock_update.md`, and `gh_issue_kcoro_next_steps.md`.

After extensive analysis of kotlinx.coroutines native/multiplatform channel implementation and tests, we've identified the root causes and aligned our test structure with Kotlin's patterns. The remaining failures are in the channel implementation itself, not test design.

## What we analyzed (Kotlin source)

Cloned and studied `kotlinx.coroutines` focusing on:
- **Common/multiplatform tests** (not JVM-specific): `kotlinx-coroutines-core/common/test/channels/`
- **Key files examined**:
  - `SendReceiveStressTest.kt` — rendezvous stress test with bounded loops
  - `RendezvousChannelTest.kt` — individual rendezvous test patterns
  - `Produce.kt` — `produce{}` builder implementation showing auto-close on completion
  - `BufferedChannel.kt` — actual channel implementation with tryResume/commit protocol

## Kotlin's rendezvous patterns (what we should match)

### 1. Structured concurrency with bounded loops
```kotlin
// From SendReceiveStressTest.kt (lines 26-45)
private suspend fun testStress(channel: Channel<Int>) = coroutineScope {
    val n = 100
    val sender = launch {
        for (i in 1..n) {
            channel.send(i)
        }
        expect(2)
    }
    val receiver = launch {
        for (i in 1..n) {
            val next = channel.receive()
            check(next == i)
        }
        expect(3)
    }
    expect(1)
    sender.join()
    receiver.join()
    finish(4)
}
```

**Key points:**
- Both sender and receiver have **bounded loops** (1..n)
- Both exit naturally when loops complete
- **No explicit close** during execution
- `coroutineScope` ensures all children complete before scope exits
- Channel cleanup happens automatically when scope finishes

### 2. Auto-close on producer completion
```kotlin
// From Produce.kt (lines 285-297)
private class ProducerCoroutine<E>(
    parentContext: CoroutineContext, channel: Channel<E>
) : ChannelCoroutine<E>(parentContext, channel, true, active = true), ProducerScope<E> {
    override fun onCompleted(value: Unit) {
        _channel.close()
    }

    override fun onCancelled(cause: Throwable, handled: Boolean) {
        val processed = _channel.close(cause)
        if (!processed && !handled) handleCoroutineException(context, cause)
    }
}
```

**Key insight:** The `produce{}` builder attaches close to the coroutine's completion callback. When the producer coroutine finishes (normally or via cancellation), the channel is automatically closed. **No manual close needed in producer code.**

### 3. Single-winner commit protocol
```kotlin
// From BufferedChannel.kt (conceptual, lines ~652-666, 1167-1184, 2951-2962)
// Sender resumes receiver:
tryResumeReceiver(...) → cont.tryResume0(...) → completeResume(token)

// Receiver resumes sender:
tryResumeSender(...) → tryResume0(Unit) → completeResume(token)
```

**Key insight:** Every rendezvous match uses an atomic tryResume/commit pattern:
1. Try to claim the counterpart (tryResume returns a token or fails)
2. Perform data transfer
3. Commit the resume (finalize, schedule continuation)
4. Only one side can win; losers retry

**No "wake and hope"** — the winning side commits atomically before scheduling.

## What we implemented (test structure)

Our test now matches Kotlin's multiplatform pattern exactly:

```c
// From test_chan_rv_metrics.c (commit 6e15c749)
static void producer_co(void *arg) {
    for (int i = 0; i < PER_PRODUCER; ++i) {  // Bounded loop
        int payload = i;
        int rc = kc_chan_send(ctx->chan, &payload, -1);
        if (rc != 0) return;  // Early exit on error
        atomic_fetch_add(&ctx->sends_completed, 1);
    }
    // Producer exits naturally after all sends complete
}

static void consumer_co(void *arg) {
    for (int i = 0; i < PER_PRODUCER; ++i) {  // Bounded loop (not infinite!)
        int rc = kc_chan_recv(ctx->chan, &value, -1);
        if (rc != 0) return;  // Early exit on error
        atomic_fetch_add(&ctx->recvs_completed, 1);
    }
    // Consumer exits after receiving its share
}

// In main:
// Spawn producers and consumers
kc_sched_drain(sched, timeout_ms);  // Wait for all to complete (like sender.join(); receiver.join())
kc_chan_close(chan);                // Close after drain (simulating scope cleanup)
```

**Changes from old test:**
- ✅ Consumers now have **bounded loops** (receive exactly PER_PRODUCER messages each)
- ✅ No infinite `for(;;)` waiting for EPIPE
- ✅ Close happens **after drain** when all coroutines have exited
- ✅ Matches Kotlin's `sender.join(); receiver.join()` pattern

## Current test results (still failing)

After aligning test structure with Kotlin:

```bash
$ make -C external/kcoro/tests all && cd external/kcoro/tests && ./build/test_chan_rv_metrics
# 10 runs: PASS=0 FAIL=10 (100% failure rate)

# Example failure:
[rv-metrics] mismatch sends=19 recvs=19 expected=800 (atomic sends=23 recvs=19)
[rv-metrics] producer send failed rc=-32 at i=13

# Observations:
- Early mismatches (sends << 800)
- Producers get -EPIPE (rc=-32) mid-execution
- Occasional segfaults
- Close telemetry shows corrupted counters (garbage values in struct)
```

## Root cause: channel implementation bugs, not test structure

### The "200-send boundary" race we discovered

**Problem:** In the old test with close-in-producer, we saw failures at exactly i=199 (the last send). Analysis revealed:

1. Producer A finishes i=199 (last send), exits loop
2. Producer A increments `producers_done` to 4 and calls `kc_chan_close()`
3. **Producer B is still parked inside `kc_chan_send()` at i=150**
4. Close wakes Producer B, it resumes and hits `if (ch->closed) return EPIPE`
5. Producer B exits without finishing, totals mismatch

**Why this matters:** Even though we fixed the test to close-after-drain, the analysis revealed a deeper issue: **producers can be parked inside `kc_chan_send()` when another finishes**. This means the channel's wake/park discipline isn't matching Kotlin's atomic commit semantics.

### Missing Kotlin patterns in our implementation

From the source analysis, our channel lacks:

1. **tryResume/commit discipline** (lines 2951-2962 in BufferedChannel.kt)
   - We use "wake and retry" instead of atomic claim/commit
   - A parked sender can wake but not complete transfer if counterpart vanished
   - Kotlin: `tryResume` returns token; `completeResume(token)` commits atomically

2. **Direct handoff at match point** (sendImpl/receiveImpl patterns)
   - We enqueue waiters and schedule wakes hoping they'll run
   - Kotlin: data transfer happens at the match point, then counterpart is resumed
   - We partially implemented this but it's incomplete/buggy

3. **Prompt cancellation** (CancellableContinuation contract)
   - Cancelled coroutines should immediately unlink from channel queues
   - We only filter at wake time; cancelled waiters linger in queues

4. **onUndelivered hooks** (for resource cleanup)
   - We added the API but it's not consistently called when payloads are dropped

5. **Scope-driven lifetime** (Job/parent tracking)
   - Kotlin's channels know when their scope is cancelled/complete
   - Our channels rely on explicit close, leading to races

## The refactor (commit 21f81a7f) introduced regressions

Testing history:
- **Before 21f81a7f**: Tests were already unstable (close-in-producer race)
- **After 21f81a7f**: "Kotlin-style commit semantics" refactor, but introduced new bugs:
  - Direct handoff code path causes hangs
  - Waiter state machine incomplete
  - Wake/commit ordering issues

**Even commit 3f7ee75e (before our refactor) fails consistently**, suggesting long-standing issues in the rendezvous implementation.

## What needs to be fixed (channel implementation)

Based on Kotlin's patterns, we need:

### 1. Proper tryResume/commit tokens
```c
// Add to kc_chan_internal.h:
struct kc_resume_token {
    struct kc_waiter *waiter;
    int expected_state;  // W_ENQ → W_CLAIMED transition guard
};

// In match paths:
int kc_waiter_try_resume_locked(struct kc_waiter *w, struct kc_resume_token *out);
int kc_waiter_commit_locked(struct kc_resume_token *tok);

// Usage in kc_chan_send (rendezvous, receiver present):
struct kc_waiter *rw = kc_waiter_pop(&ch->wq_recv_head, &ch->wq_recv_tail);
struct kc_resume_token tok;
if (kc_waiter_try_resume_locked(rw, &tok) == 0) {
    // Transfer data under lock
    memcpy(ch->slot, msg, ch->elem_sz);
    ch->has_value = 1;
    // Commit wins the race
    if (kc_waiter_commit_locked(&tok) == 0) {
        // Success: schedule wake
        kc_chan_schedule_wake(...);
        return 0;
    }
}
// Commit failed (receiver cancelled): retry
```

### 2. Complete direct handoff (pop-first)
- Receiver should pop a waiting sender and complete transfer entirely under lock
- No "wake sender and hope it publishes" pattern
- Copy payload directly from stashed buffer to receiver's output

### 3. Cancellation integration
- Add `kc_cancel_t *cancel` field to `kc_waiter`
- On cancel: unlink from queue exactly once, call onUndelivered, wake with error

### 4. Remove "success without transfer" paths
- Every `kc_chan_send` return 0 must guarantee element was transferred
- Every `kc_chan_recv` return 0 must guarantee element was received
- No early-return optimizations that skip the handoff

### 5. Memory safety
- Close telemetry shows corrupted struct fields (garbage counter values)
- Likely use-after-free or double-free in waiter disposal
- Need valgrind/ASan sweep

## Test strategy going forward

1. **Incremental validation:** Don't do big refactors. Add one Kotlin pattern at a time, validate with tests.

2. **Start from known-good state:** Consider reverting to a pre-refactor commit that at least compiles, then add patterns incrementally.

3. **Simple repro first:** Get `test_chan_ptr_rendezvous_basic` (simpler, fewer coroutines) passing before stress test.

4. **ASan/TSan:** Run with sanitizers to catch memory bugs causing corruption.

5. **Kotlin reference:** Keep BufferedChannel.kt open as reference for every change we make.

## References

- **Kotlin coroutines repo:** `/Volumes/stuff/Projects/kotlinx.coroutines`
- **Key Kotlin files:**
  - `kotlinx-coroutines-core/common/src/channels/Produce.kt`
  - `kotlinx-coroutines-core/common/src/channels/BufferedChannel.kt`
  - `kotlinx-coroutines-core/common/test/channels/SendReceiveStressTest.kt`
  - `kotlinx-coroutines-core/common/test/channels/RendezvousChannelTest.kt`
- **Our commits:**
  - `6e15c749` — Test aligned with Kotlin multiplatform pattern
  - `21f81a7f` — Refactor that introduced regressions
  - Earlier: `gh_issue_rv_deadlock.md`, `gh_issue_rv_deadlock_update.md`

## Conclusion

✅ **Test structure is now correct** — matches Kotlin's multiplatform tests exactly  
❌ **Channel implementation has fundamental bugs** — needs tryResume/commit, direct handoff, prompt cancellation  
🎯 **Next steps:** Implement Kotlin's patterns incrementally in channel code, validate at each step

The good news: we know exactly what needs to be fixed and have Kotlin's source as a reference. The bad news: it's not a small fix; the channel rendezvous core needs to be rebuilt with Kotlin's invariants.
