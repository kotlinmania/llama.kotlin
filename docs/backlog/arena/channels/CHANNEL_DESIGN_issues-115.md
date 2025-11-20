_Backlog: related issues #115_

<!-- Related issues: #115 -->
# Channel Design Ideas (Future)

_Concepts captured here are exploratory and not implemented._

## Design Proposals

### 1. Fairness Knobs for Priority Handling

**Problem Statement:**
Currently, channel wait queues (`token_send_head/tail` and `token_recv_head/tail`) are simple FIFO linked lists. In scenarios with mixed workloads—high-priority control messages and bulk data transfers—FIFO ordering can lead to starvation of latency-sensitive operations.

**Design Options:**

#### Option A: Priority-Aware Wait Queues (Recommended)
Add an optional priority field to `kc_pending_send` and `kc_pending_recv` structures:

```c
struct kc_pending_send {
    struct kc_pending_send *next;
    enum kc_pending_kind    kind;
    enum kc_pending_role    role;
    kc_ticket               ticket;
    kc_select_t            *sel;
    int                     clause_index;
    kc_desc_id              desc_id;
    uint8_t                 priority;  // NEW: 0 (default/low) to 255 (high)
};
```

**Implementation approach:**
- Maintain multiple wait queues per priority level (e.g., 3 levels: high, normal, low)
- Or maintain sorted insertion within a single queue (trade-off: O(n) insertion vs O(1) pop)
- Service higher-priority waiters first when matching rendezvous pairs
- Provide APIs: `kc_chan_send_priority()`, `kc_chan_recv_priority()`
- Default priority = 128 (middle) for backward compatibility

**Trade-offs:**
- Pro: Prevents starvation of latency-sensitive operations
- Pro: Minimal API surface change (priority parameter is optional)
- Con: Slightly more complex queue management
- Con: Need to define fairness policy (strict priority vs weighted round-robin)

#### Option B: Time-Based Aging
Instead of explicit priorities, track wait time and boost priority of long-waiting operations:

```c
struct kc_pending_send {
    // ... existing fields ...
    long enqueue_time_ns;  // NEW: timestamp when enqueued
};
```

**Implementation approach:**
- Periodically scan wait queues and boost priority of aged entries
- Effective priority = base_priority + (wait_time / aging_threshold)
- Prevents indefinite starvation while maintaining FIFO semantics for same-priority operations

**Trade-offs:**
- Pro: Automatic fairness without explicit priority management
- Pro: Simpler API (no priority parameters needed)
- Con: Requires periodic queue scans (overhead)
- Con: Less predictable latency for high-priority operations

**Recommendation:**
Start with **Option A** (priority-aware queues) using 3 priority levels and strict priority scheduling. Add aging (Option B) later if starvation is observed in production workloads.

---

### 2. Timeout Helpers for Send/Recv APIs

**Problem Statement:**
Currently, timeouts must be implemented at the token kernel level or via external timer mechanisms. Many users need simple timeout semantics without complex integration.

**Current State:**
- Token kernel supports callback-based timeouts via ticket management
- No high-level channel API exposes timeout parameters
- Users must manually integrate with token kernel or use select-with-timeout patterns

**Design Options:**

#### Option A: Inline Timeout Parameters (Simple)
Add timeout variants to channel APIs:

```c
// New timeout-aware APIs
int kc_chan_send_timeout(kc_chan_t *ch, const void *data, size_t len,
                         long timeout_ns);
int kc_chan_recv_timeout(kc_chan_t *ch, void *buf, size_t len,
                         long timeout_ns);

// Return codes:
// 0 = success
// -ETIME = timeout expired before operation completed
// -EAGAIN = would block (for zero timeout)
// existing error codes...
```

**Implementation approach:**
- Integrate with token kernel timeout machinery
- Register timeout callback that cancels the pending operation
- Clean up wait queue entry on timeout
- Return `-ETIME` to caller

**Trade-offs:**
- Pro: Simple, familiar API pattern (similar to POSIX `recv`/`send` with `SO_RCVTIMEO`)
- Pro: No need for users to understand token kernel directly
- Con: Adds API surface area
- Con: Requires careful cancellation handling to avoid races

#### Option B: Context-Based Deadline (Advanced)
Pass a deadline context that applies to multiple operations:

```c
typedef struct {
    long deadline_ns;  // absolute deadline
    int canceled;      // external cancellation flag
} kc_deadline_ctx;

int kc_chan_send_ctx(kc_chan_t *ch, const void *data, size_t len,
                     kc_deadline_ctx *ctx);
```

**Trade-offs:**
- Pro: Supports complex timeout patterns (cascade timeouts, shared deadlines)
- Pro: Enables cancellation from external sources
- Con: More complex API requiring explicit context management
- Con: Requires careful lifetime management of context objects

#### Option C: Select-Style Timeout (Status Quo+)
Improve existing select mechanism with better timeout support:

```c
// Already exists but could be better documented/ergonomic
kc_select_t sel;
kc_select_init(&sel);
kc_select_case_send(&sel, ch, data, len);
kc_select_case_timeout(&sel, timeout_ns);
int result = kc_select_wait(&sel);
```

**Trade-offs:**
- Pro: No new API surface
- Pro: Already supports multiple channels + timeout
- Con: Verbose for simple single-channel timeout case
- Con: Requires understanding select mechanism

**Recommendation:**
Implement **Option A** (inline timeout parameters) for common use cases. This provides the simplest user experience for the 80% case (single channel operation with timeout). Document how to use select for more complex patterns.

**Implementation Notes:**
- Coordinate with token kernel timeout experiments (see related issues #109, #112, #116)
- Ensure timeout cleanup doesn't interfere with descriptor refcounting
- Add tests for timeout races (timeout fires exactly as rendezvous completes)

---

### 3. Multi-Recipient Broadcast Semantics

**Problem Statement:**
Current channels are point-to-point (one sender matches one receiver). Many use cases need broadcast semantics where one sender delivers to multiple receivers.

**Use Cases:**
- Configuration updates propagated to all workers
- Event notifications to multiple subscribers
- Telemetry/logging fan-out patterns

**Design Options:**

#### Option A: Dedicated Broadcast Channel Type
Introduce a new `KC_BROADCAST` channel kind:

```c
kc_chan_t *ch = kc_chan_new(elem_sz, KC_BROADCAST);

// Sender API (unchanged)
kc_chan_send(ch, data, len);

// Receiver registration (explicit subscription)
kc_chan_subscribe(ch, receiver_id);
kc_chan_recv(ch, buf, len);  // blocks until broadcast available
```

**Implementation approach:**
- Maintain list of subscribed receivers
- On send, copy payload to all subscriber queues
- Receivers drain their individual queues (FIFO per-receiver)
- Handle late subscribers (miss messages or buffer N most recent?)

**Trade-offs:**
- Pro: Native broadcast semantics, efficient implementation
- Pro: Clear subscription model
- Con: New channel type increases complexity
- Con: Memory overhead (N copies of each message)
- Con: Need to define buffer behavior (bounded vs unbounded per-receiver queue)

#### Option B: Fan-Out Combinator Pattern
Provide utility that fans out to multiple rendezvous channels:

```c
// User code creates N channels
kc_chan_t *receivers[N];
for (int i = 0; i < N; i++) {
    receivers[i] = kc_chan_new(elem_sz, KC_RENDEZVOUS);
}

// Utility function fans out
int kc_chan_fanout(const void *data, size_t len,
                   kc_chan_t **channels, int n_channels);
// Returns: number of successful sends (may be partial)
```

**Implementation approach:**
- Application-level primitive, not core channel feature
- Iterate through receiver channels and attempt send on each
- Non-blocking sends with configurable failure policy (all-or-nothing vs best-effort)
- Can be implemented as a library function, not core runtime

**Trade-offs:**
- Pro: Reuses existing channel primitives
- Pro: No new channel types or complex runtime changes
- Con: User must manage N channels explicitly
- Con: Less efficient (N separate rendezvous operations)
- Con: Hard to guarantee atomic broadcast (all-or-nothing delivery)

#### Option C: Pub-Sub Layer (Future)
Build a higher-level pub-sub abstraction on top of channels:

```c
typedef struct kc_pubsub kc_pubsub_t;

kc_pubsub_t *ps = kc_pubsub_new(elem_sz);
kc_subscription_t *sub = kc_pubsub_subscribe(ps, "topic");

// Publisher
kc_pubsub_publish(ps, "topic", data, len);

// Subscriber
kc_chan_t *ch = kc_subscription_channel(sub);
kc_chan_recv(ch, buf, len);
```

**Trade-offs:**
- Pro: Full-featured pub-sub with topics, filtering, QoS
- Pro: Natural fit for event-driven architectures
- Con: Significant implementation effort
- Con: Out of scope for core channel primitives
- Con: May overlap with higher-level frameworks

**Recommendation:**
Start with **Option B** (fan-out combinator) as a library utility. This provides immediate value with minimal runtime complexity. Evaluate **Option A** (dedicated broadcast channel) if usage patterns show high demand and performance becomes a bottleneck. **Option C** is deferred to future higher-level abstractions.

**Example Implementation Sketch (Option B):**

```c
int kc_chan_fanout_best_effort(const void *data, size_t len,
                                kc_chan_t **channels, int n) {
    int successful = 0;
    for (int i = 0; i < n; i++) {
        int rc = kc_chan_send_nowait(channels[i], data, len);
        if (rc == 0) successful++;
    }
    return successful;
}

int kc_chan_fanout_all_or_nothing(const void *data, size_t len,
                                   kc_chan_t **channels, int n) {
    // Phase 1: Check if all channels are ready
    for (int i = 0; i < n; i++) {
        if (!kc_chan_send_ready(channels[i])) {
            return -EAGAIN;  // At least one channel not ready
        }
    }

    // Phase 2: Send to all (races possible, but best effort)
    int successful = 0;
    for (int i = 0; i < n; i++) {
        if (kc_chan_send_nowait(channels[i], data, len) == 0) {
            successful++;
        }
    }

    return (successful == n) ? 0 : -EAGAIN;
}
```

---

## Additional Enhancements (Lower Priority)

- **Back-pressure metrics:** Emit structured telemetry (queue depth, wait time percentiles) to help upstream systems adapt their send rates.
- **Batch wakeups:** Allow the token kernel to coalesce multiple ready continuations when the scheduler thread is idle, improving throughput under bursty loads.
- **Zero-copy slices:** Investigate passing arena-backed sliced ranges directly without requiring a full descriptor retain/release cycle each time.
- **Peek APIs:** Expose "peek" APIs for conflated channels to observe the latest value without consuming it.

---

## Next Steps

1. **Prototype fairness implementation:** Add priority field to pending structures and implement priority queue logic in a feature branch
2. **Timeout API RFC:** Create detailed API proposal for timeout helpers with code examples and error handling semantics
3. **Benchmark fan-out pattern:** Implement and test Option B for broadcast semantics, measure performance vs dedicated broadcast channel
4. **Integration testing:** Coordinate with token kernel timeout experiments (issues #109, #112, #116)
5. **Documentation:** Update `CHANNEL_DESIGN_VERIFIED.md` if any proposals ship

---

## References

- Token kernel backlog: [OVERVIEW_issues-109-112-116.md](../../token_kernel/OVERVIEW_issues-109-112-116.md)
- Channel implementation: `external/arena/core/src/kc_chan.c`
- Internal structures: `external/arena/core/src/kc_chan_internal.h`
