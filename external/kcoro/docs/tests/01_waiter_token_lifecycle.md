# Test Case 01 — Waiter Token Lifecycle Integrity

## Purpose
Verify that `kc_waiter_token_t` instances move through the expected lifecycle states (INIT → ENQUEUED → CLAIMED → MATCHED or CANCELLED) without skipping or repeating states under single-threaded and multi-threaded execution.

## Preconditions
- kcoro core library built with assertions enabled.
- Arena allocator initialized with at least one rendezvous cell.
- Test harness provides direct access to `kc_waiter_token_t` instrumentation hooks or exposes a debug callback.

## Steps
1. Allocate a rendezvous cell and create a sender waiter token.
2. Invoke the enqueue routine (`rv_sender_arrive`) and capture the token state after enqueue.
3. Drive the matching logic with a receiver arrival (`rv_receiver_arrive`).
4. Observe the state transitions recorded for both sender and receiver tokens.
5. Repeat steps 1–4 with cancellation injected between ENQUEUED and CLAIMED for both sender and receiver.
6. Run steps 1–5 inside a loop of 10⁵ iterations with concurrent producer/consumer threads to surface race conditions.
7. Capture metrics for total transitions, cancellations, and invalid transitions (if any).

## Expected Results
- Every token must traverse the state graph without skipping states.
- No token transitions back to INIT or ENQUEUED once CLAIMED.
- Cancellation path produces MATCHED=0 and CANCELLED=1 in the rendezvous metrics.
- No assertions or undefined behaviour triggered during the stress loop.
