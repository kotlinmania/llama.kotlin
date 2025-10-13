# Rendezvous Implementation Gaps (vs Industry Best Practices)

## Date: 2025-10-13

## Summary

After comparing our kcoro rendezvous channel implementation against established coroutine channel patterns, several gaps emerged that explain why our stress tests fail at exactly 200 messages.

---

## Key Differences

### 1. **Cell State Machine Atomicity**

**Industry pattern:**
- Uses explicit cell states: `NULL` → `WAITER` → `RESUMING_BY_RCV` → `DONE_RCV`
- Also has: `BUFFERED`, `INTERRUPTED_SEND`, `INTERRUPTED_RCV`, `POISONED`
- State transitions are atomic CAS operations
- Intermediate `RESUMING_*` states prevent double-resume races

**Our implementation:**
- Uses `has_value` flag + waiter queues
- State transitions rely on lock-protected code paths
- No intermediate "resuming" state to prevent races
- Multiple threads can observe inconsistent state between lock acquisitions

**Impact:**  
At high message counts (200+), concurrent send/recv can create a window where:
- Receiver sees `has_value=1` and consumes
- Before receiver publishes follow-on value, another receiver arrives
- Both receivers might think they can proceed
- System stalls with staged value but no committed consumer

---

### 2. **tryResume + Commit Protocol**

**Industry pattern:**
```
if (sender.tryResumeSender(segment, index)) {
    segment.setState(index, DONE_RCV)  // commit
    expandBuffer()
    segment.retrieveElement(index)
} else {
    segment.setState(index, INTERRUPTED_SEND)  // rollback
    segment.onCancelledRequest(index, false)
    FAILED
}
```
- `tryResume` is atomic; returns token
- Only on success does state move to `DONE_*`
- Failure explicitly marks `INTERRUPTED_*` and cleans up

**Our implementation:**
```c
int cl = kc_waiter_claim_prepare_wake_locked(rw, 0, &wake_r);
if (cl == 0) {
    // publish and schedule
}
// if claim failed, push back and retry
```
- Claim can fail silently
- Push-back creates re-enqueue which can violate FIFO
- No explicit "interrupted" marker; waiter just re-enters queue

**Impact:**  
Failed claims create phantom waiters that confuse progress tracking. After 200 iterations, accumulated re-enqueues can cause:
- Scheduler to see "idle" when waiters exist
- Counters to mismatch (sends != recvs)

---

### 3. **Poisoned / Failed Cell Handling**

**Industry pattern:**
- If a cell is "covered by sender" (global index < senders) but state is `NULL`, mark it `POISONED`
- `POISONED` cells are explicitly skipped and cleaned
- `expandBuffer()` is called even on poisoned cells to maintain invariants

**Our implementation:**
- No poisoning concept
- If match fails, we just `goto again_recv` or push waiter back
- No explicit "this cell is unusable" marker

**Impact:**  
Rendezvous at exactly the producer count boundary (200) can create cells that are logically "should have a sender" but actually don't, leading to infinite retry loops.

---

### 4. **Intermediate RESUMING States**

**Industry pattern:**
- `RESUMING_BY_RCV`: receiver is attempting to resume sender
- `RESUMING_BY_EB`: expandBuffer is attempting to resume sender
- These prevent double-resume races

**Our implementation:**
- Transition directly from waiter to committed/scheduled
- No intermediate state visible to other threads
- Race window: claim succeeds, but before scheduling, another thread sees old state

**Impact:**  
At message boundaries (like 200), multiple threads racing to claim the same waiter can both think they succeeded, leading to:
- Double-wake (both schedule the same coroutine)
- Or worse: both think they own the transfer, counters diverge

---

### 5. **expandBuffer() Discipline**

**Industry pattern:**
- Called on every state transition (even suspension, cancellation, poisoning)
- Maintains invariant: logical buffer end always advances when operations complete
- Rendezvous channels skip actual buffer expand but still call it for side-effects

**Our implementation:**
- No equivalent of `expandBuffer()`
- Progress is implicit via wake scheduling

**Impact:**  
Without a unifying "advance progress" call, we rely entirely on wake scheduling being perfect. Any missed wake = permanent stall.

---

## Why Exactly 200 Messages?

**Hypothesis:**

- Test uses 4 producers × 200 messages/producer = 800 total
- At message 200 from the first producer, that producer finishes and tries to close
- Simultaneously:
  - Other producers are at various points (50-150 messages)
  - Consumers are draining
- **Critical race:** the 200th message from producer #1 lands in a cell where:
  - A receiver is arriving (state should transition to rendezvous)
  - But producer #1 also initiates close
  - Cell state is ambiguous: not clearly "sender waiting" or "closed"
  - Our code doesn't have `POISONED` or `INTERRUPTED_*` states to handle this
  - Result: cell is stuck, scheduler sees no progress, drain succeeds prematurely

---

## Recommended Fixes (Priority Order)

### 1. **Add Cell State Enum** (HIGH)
Define explicit states for rendezvous cells:
```c
enum kc_cell_state {
    KC_CELL_EMPTY,
    KC_CELL_SENDER,        // waiter present, not yet claimed
    KC_CELL_RESUMING_SEND, // receiver is claiming sender
    KC_CELL_DONE,          // transfer complete
    KC_CELL_INTERRUPTED,   // waiter was cancelled
    KC_CELL_POISONED       // cell is unusable, skip it
};
```
Store this in waiter or a per-slot state field.

### 2. **Implement Proper tryResume Token** (HIGH)
Change `kc_waiter_claim_prepare_wake_locked` to return a token that must be committed:
```c
struct kc_resume_token {
    struct kc_waiter *w;
    int expected_state;
};

int kc_waiter_try_resume_locked(struct kc_waiter *w, struct kc_resume_token *out);
int kc_waiter_commit_locked(struct kc_resume_token *tok);
```
- `try_resume`: CAS waiter state from SENDER → RESUMING_SEND, return token
- `commit`: CAS RESUMING_SEND → DONE, finalize transfer
- If either fails, explicitly mark INTERRUPTED

### 3. **Add Progress Callback** (MEDIUM)
Equivalent to `expandBuffer()`:
```c
static void kc_chan_advance_progress_locked(struct kc_chan *ch);
```
Call this after every send/recv operation (success, suspension, cancellation, or failure). Even on rendezvous (no actual buffer), this updates metrics and wakes blocked operations.

### 4. **Poison Cells on Rendezvous Mismatch** (MEDIUM)
When receiver arrives at a cell index that *should* have a sender (based on global counters) but state is EMPTY:
- Mark cell POISONED
- Call advance_progress
- Return FAILED and restart operation

### 5. **Fix Close Semantics** (MEDIUM)
- Close must transition all waiting cells to INTERRUPTED or CLOSED state atomically
- Staged value is explicitly marked "consumable once" with a state flag
- After that flag is consumed, further recvs get EPIPE

### 6. **Audit Re-Enqueue Logic** (LOW)
- Push-back on claim failure violates FIFO
- Better: dispose waiter and let caller create a fresh one
- Or: mark waiter as "retry" and ensure it goes to tail, not head

---

## Next Steps

1. File GitHub issue: "Rendezvous stalls at message boundary (200+) due to missing cell state atomicity"
2. Implement fix #1 (cell state enum) and #2 (tryResume token) first
3. Re-run stress tests to confirm 800/800 success
4. Then layer in #3-#6 for robustness

---

## References

- Industry implementations use lock-free segment-based queues with atomic cell states
- Paper: "Fast and Scalable Channels in Kotlin Coroutines" (Elizarov et al. 2022)
- Our clean-room implementation intentionally differs (mutex-based, simpler), but must adopt the *semantic* invariants (atomic state transitions, explicit failure modes)

