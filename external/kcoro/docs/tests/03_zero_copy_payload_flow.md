# Test Case 03 — Zero-Copy Payload Flow

## Purpose
Validate that zero-copy descriptors (`KC_PAYLOAD_ZDESC`) travel from sender to receiver without buffer duplication, and that reference counts are released when match or cancel paths complete.

## Preconditions
- Zero-copy support compiled in (payload descriptor allocator available).
- Debug hooks to count retain/release events on descriptors.
- Ability to inspect payload pointer and length at both sender and receiver ends.

## Steps
1. Allocate a zero-copy descriptor referencing a 4 KiB buffer.
2. Sender enqueues descriptor; receiver dequeues and validates pointer/length equality.
3. After MATCHED state, release descriptor and confirm retain/release counts net to zero.
4. Repeat with immediate cancellation to ensure descriptors are released on failure path.
5. Execute 100,000 iterations with alternating match/cancel to surface leaks.

## Expected Results
- Receiver observes the exact pointer and length that sender published; no memcpy occurs.
- Descriptor retain count returns to baseline after each iteration.
- Cancellation path releases descriptor even if receiver never arrives.
- Stress loop reports zero leaked descriptors and stable heap usage.
