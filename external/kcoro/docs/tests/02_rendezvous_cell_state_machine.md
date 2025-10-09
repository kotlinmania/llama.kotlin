# Test Case 02 — Rendezvous Cell State Machine Validation

## Purpose
Ensure `kc_rv_cell_t` transitions correctly between `EMPTY`, `SENDER_READY`, `RECEIVER_READY`, `MATCHED`, and `CANCELLED` under the presence of multiple concurrent senders/receivers, verifying lock-free CAS loops behave as specified.

## Preconditions
- Arena seeded with N ≥ 16 rendezvous cells.
- Instrumentation to snapshot cell state after each CAS.
- Ability to interleave sender/receiver threads deterministically (e.g., via a scheduler shim).

## Steps
1. For each cell, enqueue a sender and verify state moves from `EMPTY` → `SENDER_READY`.
2. Interleave receiver arrival; confirm state becomes `MATCHED` and both waiters detach.
3. Repeat with reversed order (receiver first, sender second) and ensure final state is `MATCHED`.
4. Inject a cancellation after `SENDER_READY` but before receiver arrival; verify final state is `CANCELLED` and cell returns to `EMPTY` after cleanup.
5. Run a randomized interleaving of sender/receiver arrivals for 1,000,000 operations, recording any illegal state transitions.

## Expected Results
- No illegal transitions (e.g., direct `EMPTY` → `MATCHED`).
- After every match or cancellation, the cell resets to `EMPTY` with zeroed payload pointers.
- Metrics counters (`matches`, `cancels`) reflect the exact number of completed operations.
- Stress test completes without cell starvation or ABA detection failures.
