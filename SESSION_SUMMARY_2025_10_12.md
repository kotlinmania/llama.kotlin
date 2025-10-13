# Session Summary: kcoro Rendezvous Kotlin Parity Analysis
**Date:** 2025-10-12  
**Duration:** ~6 hours  
**Focus:** Analyze kotlinx.coroutines to understand proper rendezvous channel patterns and align kcoro tests

---

## What We Accomplished

### 1. Deep Kotlin Source Analysis ✅

**Cloned and studied kotlinx.coroutines:**
- Location: `/Volumes/stuff/Projects/kotlinx.coroutines`
- Focused on **common/multiplatform** tests (not JVM-specific)
- Key files analyzed:
  - `SendReceiveStressTest.kt` — rendezvous stress test pattern
  - `RendezvousChannelTest.kt` — individual test cases
  - `Produce.kt` — auto-close implementation via onCompleted callback
  - `BufferedChannel.kt` — tryResume/commit protocol internals

**Key learnings documented in:** `gh_issue_kotlin_parity_analysis.md`

### 2. Test Structure Alignment ✅

**Transformed test from:** Infinite consumer loop waiting for EPIPE  
**To:** Bounded loops matching Kotlin's pattern

**Before:**
```c
// Consumers loop forever waiting for close
for (;;) {
    rc = kc_chan_recv(chan, &value, -1);
    if (rc == -EPIPE) break;  // Wait for manual close
    ...
}
```

**After (commit 6e15c749):**
```c
// Consumers receive exactly PER_PRODUCER messages (bounded loop)
for (int i = 0; i < PER_PRODUCER; ++i) {
    rc = kc_chan_recv(chan, &value, -1);
    if (rc != 0) return;  // Early exit on error
    ...
}
// Consumer exits naturally, no EPIPE needed
```

**Pattern now matches:**
- Kotlin's `SendReceiveStressTest.kt` (lines 26-45)
- Both sender and receiver have bounded loops
- Both exit naturally when complete
- Close happens after `drain` (like `sender.join(); receiver.join()`)

### 3. Root Cause Identification ✅

**Discovered the "200-send boundary" race:**

When one producer finishes and closes the channel, **other producers are still parked inside `kc_chan_send()`**. They wake, see `closed==true`, return EPIPE, and exit without finishing — causing mismatched totals.

**Why it matters:** Revealed that our channel implementation doesn't match Kotlin's atomic commit semantics. In Kotlin, a parked operation either completes or is explicitly cancelled; it never "half-completes."

### 4. Kotlin vs kcoro Pattern Comparison ✅

| Pattern | Kotlin (BufferedChannel.kt) | kcoro (current) | Status |
|---------|----------------------------|-----------------|--------|
| **Match protocol** | tryResume → commit → schedule | wake → retry loop | ❌ Missing |
| **Direct handoff** | Transfer at match point | Wake and hope | ⚠️ Partial |
| **Cancellation** | Immediate unlink via CancellableContinuation | Filter at wake time | ❌ Missing |
| **Undelivered cleanup** | onUndeliveredElement hook | API added, not fully wired | ⚠️ Partial |
| **Scope lifetime** | Auto-close via Job completion | Manual close | ❌ Missing |
| **Success == transfer** | Atomic guarantee | Best effort | ❌ Broken |

### 5. Test Results (Current State) ⚠️

**After aligning test structure:**
```bash
$ make -C external/kcoro/tests all && cd external/kcoro/tests && ./build/test_chan_rv_metrics
# 10 runs: PASS=0 FAIL=10 (100% failure rate)

Example failures:
- [rv-metrics] mismatch sends=19 recvs=19 expected=800
- [rv-metrics] producer send failed rc=-32 at i=13
- Segmentation fault: 11 (occasional)
- Close telemetry shows corrupted struct fields (garbage values)
```

**Conclusion:** Test structure is correct; failures are in **channel implementation**.

---

## Commits Made This Session

1. **21f81a7f** — Attempted Kotlin-style commit semantics (introduced regressions)
2. **2d08c6e8** — Wake receiver after select-send; direct handoff attempt (caused hangs)
3. **da69cf06** — Added close monitoring counters for debugging
4. **bc6c684d** — Coordinator pattern to defer close (discovered 200-boundary race)
5. **6e15c749** — Final test alignment with Kotlin multiplatform pattern ✅
6. **508a25f3** — Comprehensive analysis document (this summary references it)

---

## Key Documents Created

1. **`gh_issue_kotlin_parity_analysis.md`** (278 lines)
   - Complete analysis of Kotlin's patterns
   - Side-by-side code comparisons
   - Detailed root cause breakdown
   - Implementation roadmap

2. **Session traces and findings:**
   - Located in commit messages
   - Test output snapshots in analysis doc

---

## What Still Needs to Be Done

### Immediate (Channel Implementation Fixes)

1. **Add tryResume/commit tokens** (highest priority)
   ```c
   struct kc_resume_token { struct kc_waiter *waiter; int expected_state; };
   int kc_waiter_try_resume_locked(...);
   int kc_waiter_commit_locked(...);
   ```

2. **Complete direct handoff** (pop-first policy)
   - Receiver pops sender and transfers data under lock
   - No "wake and hope" pattern

3. **Prompt cancellation**
   - Add `kc_cancel_t *cancel` to waiters
   - Unlink immediately on cancel, not at next wake

4. **Memory safety fixes**
   - ASan/valgrind sweep to find use-after-free
   - Fix corrupted struct fields (close counters showing garbage)

5. **Remove "success without transfer" paths**
   - Audit all return-0 paths to guarantee transfer happened

### Next Steps (Strategy)

1. **Revert to stable base** — Consider resetting to pre-21f81a7f
2. **Incremental pattern adoption** — Add one Kotlin pattern at a time, validate
3. **Simple test first** — Get `test_chan_ptr_rendezvous_basic` passing
4. **Sanitizer sweep** — Run with ASan/TSan to catch memory bugs
5. **Kotlin reference always open** — Use BufferedChannel.kt as guide for every change

---

## Testing Against Kotlin Patterns (Checklist)

- [x] Analyzed Kotlin common/multiplatform tests
- [x] Identified proper test structure (bounded loops, no explicit close)
- [x] Aligned our test to match Kotlin's SendReceiveStressTest.kt
- [x] Documented all missing patterns in channel implementation
- [x] Created comprehensive analysis document
- [ ] Implement tryResume/commit tokens
- [ ] Implement direct handoff (pop-first)
- [ ] Add prompt cancellation
- [ ] Pass test_chan_ptr_rendezvous_basic
- [ ] Pass test_chan_rv_metrics (10/10 clean runs)
- [ ] ASan/TSan clean

---

## References

- **Kotlin repo:** `/Volumes/stuff/Projects/kotlinx.coroutines`
- **Analysis doc:** `gh_issue_kotlin_parity_analysis.md`
- **Prior issues:** `gh_issue_rv_deadlock.md`, `gh_issue_rv_deadlock_update.md`, `gh_issue_kcoro_next_steps.md`

---

## Key Insight

> The test was never the problem. Our test structure now **perfectly matches** Kotlin's multiplatform stress tests. The issue is that our channel implementation doesn't have Kotlin's atomic commit protocol, direct handoff semantics, or prompt cancellation. We know exactly what needs to be fixed because we've read their source.

**Bottom line:** We have a clear roadmap. The channel needs to be rebuilt with Kotlin's invariants, but at least we now know exactly what those are.
