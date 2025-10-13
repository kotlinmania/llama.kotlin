# Rendezvous Channel Debugging Session

**Date:** 2025-10-13  
**Issues:** #81 (deadlock), #84 (commit protocol), #87 (memory corruption)

## Work Completed

### 1. Fixed Memory Corruption (Issue #87) ✅
**Commit:** f4ec47d4

**Problem:** Double-free bugs in `kc_chan_wake_send_locked` and `kc_chan_wake_recv_locked`
- Wake helpers were manually disposing waiters without using claim protocol
- Could dispose same waiter twice in different code paths
- Caused segfaults and garbage telemetry values

**Solution:**
- Both wake helpers now use `kc_waiter_claim_prepare_wake_locked()`
- Single-winner semantics enforced
- Disabled unused `kc_chan_select_deliver_*` functions that bypassed claim
- No more segfaults or memory corruption

### 2. Fixed Push-Back Antipattern ⏸️
**Commit:** 7e6d82be

**Key Learning from Kotlin BufferedChannel.kt (lines 550-650):**
```kotlin
return if (receiver.tryResumeReceiver(element)) {
    segment.setState(index, DONE_RCV)
    RESULT_RENDEZVOUS
} else {
    // The resumption has failed
    segment.getAndSetState(index, INTERRUPTED_RCV)
    RESULT_FAILED  // <-- Outer loop retries, DON'T push back!
}
```

**Kotlin's Pattern:**
When `tryResumeReceiver` fails:
1. Mark waiter as `INTERRUPTED_RCV`
2. Return `RESULT_FAILED`
3. **Outer `while(true)` loop continues to next cell**
4. **NEVER push waiter back onto queue** (creates circular waits)

**Our Fix:**
- Send path: when receiver claim fails → dispose + goto retry
- Recv path: when sender claim fails or no payload → dispose + goto retry
- Added `kc_waiter_claim_prepare_wake_locked()` to direct handoff
- Wake sender after successful direct handoff in recv

**Status:** Test still hangs (deeper issue remains)

## Current Problem

Test `test_chan_rv_metrics` still deadlocks despite fixes above.

**Hypothesis:** Need complete audit of all rendezvous paths to ensure:
- Proper claim → transfer → commit → wake sequence
- No remaining circular waits
- All branches follow Kotlin's "dispose and retry" pattern

## Next Steps

1. Full audit of all RV send/recv code paths
2. Verify claim/commit discipline in every branch
3. Check for any remaining "push back" patterns
4. Ensure proper wake scheduling after every successful transfer
5. Add detailed trace logging to identify exact hang point

## Key Files

- `external/kcoro/core/src/kc_chan.c` - Channel implementation
- `external/kcoro/tests/test_chan_rv_metrics.c` - Stress test
- `/Volumes/stuff/Projects/kotlinx.coroutines/kotlinx-coroutines-core/common/src/channels/BufferedChannel.kt` - Reference

## Commits This Session

- f4ec47d4 - Fix double-free and memory corruption in wake helpers
- 7e6d82be - Fix claim-failed push-back antipattern (partial)
