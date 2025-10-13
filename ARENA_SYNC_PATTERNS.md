# Arena Version - Syncable Patterns from Base kcoro Fixes

**Date:** 2025-10-13  
**Base commits:** f4ec47d4, 7e6d82be

This doc identifies patterns from our base kcoro fixes that can be applied to the arena version in `external/kcoro/lab/mirror/`.

---

## ✅ Pattern 1: Scheduler Alive Tracking

**Base kcoro:** Added `_Atomic(int) alive` counter (commit 21f81a7f)
- Increment on spawn
- Decrement on finish  
- `kc_sched_drain` waits for `alive==0` instead of just "approx_idle"

**Arena status:** Has `tasks_submitted`/`tasks_completed` counters but NO alive tracking

**Sync action:**
```c
// In struct kc_sched:
_Atomic(int) alive;

// In spawn path (add after enqueue):
atomic_fetch_add_explicit(&s->alive, 1, memory_order_acq_rel);

// In finish path (add before release):
atomic_fetch_sub_explicit(&s->alive, 1, memory_order_acq_rel);

// In kc_sched_drain:
if (approx_idle(s) && atomic_load_explicit(&s->alive, memory_order_acquire) == 0)
    return 0;
```

**Why:** Prevents false "idle" detection when coroutines are parked but work remains.

---

## ❓ Pattern 2: Double-Callback Protection (NEEDS REVIEW)

**Base kcoro issue:** Wake helpers could dispose same waiter twice → segfault

**Arena equivalent:** `pending_X_dequeue()` + `kc_token_kernel_callback()` + `free(pending)`

**Locations to audit:**
- Line 410-416: `pending_recv_dequeue` → callback → `free(pending)`
- Line 503-508: Similar pattern
- Line 585-597: `pending_send_dequeue` → callback → `free(pending)`
- Line 857-869: Similar pattern

**Question:** Can `kc_token_kernel_callback()` be called twice on same ticket?
- Check if token kernel has single-fire protection
- Check if dequeue can race with close/cancel paths

**Sync action IF needed:**
Add claim/consumed flag to pending nodes:
```c
struct kc_pending_send {
    ...
    int consumed;  // Set by winner before callback
};

// Before callback:
if (pending->consumed) {
    free(pending);
    continue;  // Already handled
}
pending->consumed = 1;
kc_token_kernel_callback(...);
free(pending);
```

**Why:** Our base bug was calling wake twice. Arena might have similar issue with callbacks.

---

## ✅ Pattern 3: Don't Push Back Failed Operations (NOT APPLICABLE)

**Base kcoro issue:** When claim failed, we pushed waiter back → circular wait

**Arena status:** Uses ticket system, not waiter push-back pattern

**Sync action:** ❌ **Not applicable** - Arena doesn't have this pattern

---

## ✅ Pattern 4: Close With Staged Value (APPLICABLE)

**Base kcoro:** If `has_value==1` at close, wake one receiver first (commit 2d08c6e8)

**Arena equivalent:** Unknown - need to check how arena handles close with pending descriptors

**Check locations:**
- `kc_chan_close()` implementation in arena
- Does it drain pending sends/recvs?
- Are in-flight descriptors consumed or dropped?

**Sync action TBD** after reviewing arena close logic.

---

## 🔍 Pattern 5: Cancellation Cleanup (NEEDS AUDIT)

**Base kcoro:** Added prompt cancellation - drop cancelled waiters from queue, call onUndelivered

**Arena equivalent:** Lines 113, 131, 299-300, 310-311 show `kc_token_kernel_cancel()`

**Check:**
- Does token kernel remove ticket from hash table on cancel?
- Are pending descriptors released on cancel?
- Can cancelled ticket still be called back?

**Sync action TBD** after token kernel audit.

---

## Summary

**Can sync immediately:**
1. ✅ Scheduler alive tracking (straightforward port)

**Need to investigate first:**
2. ❓ Double-callback protection (check token kernel guarantees)
3. ❓ Close with staged value (audit arena close logic)
4. ❓ Cancellation cleanup (audit token kernel cancel path)

**Not applicable:**
- Push-back pattern (arena doesn't use this)

---

## Next Steps

1. Port scheduler alive tracking (low risk, high value)
2. Audit token kernel for single-fire callback guarantee
3. Review arena close logic for descriptor handling
4. Test arena version after each sync

