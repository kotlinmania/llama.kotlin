# Test Case 04 — Token VM Scheduler Resume Correctness

## Purpose
Confirm that the token VM captures/restores coroutine state correctly across suspends, covering register spill/fill, stack pointer handling, and status transitions (`READY`, `RUNNING`, `BLOCKED`, `DONE`).

## Preconditions
- Access to `mirror_scheduler` hooks or equivalent scheduler API.
- Ability to snapshot register contents pre/post resume (e.g., test coroutine writes sentinel values).
- Timer or watchdog to detect lockups.

## Steps
1. Create a coroutine that stores sentinel values in callee-saved registers and on the stack, then suspends via channel send/receive.
2. Let the scheduler park the coroutine (status → `BLOCKED`).
3. Resume coroutine and verify sentinels are intact, status transitions `BLOCKED` → `READY` → `RUNNING` → `DONE`.
4. Repeat with multiple concurrent coroutines to ensure scheduler fairness.
5. Stress loop: spawn 1,000 coroutines performing suspend/resume cycles 10,000 times, recording any mismatched sentinel or deadlock.

## Expected Results
- No register corruption; all sentinel values match expected after resume.
- Scheduler statuses follow documented order with no skipped states.
- Stress loop finishes without watchdog timeouts or unhandled deadlocks.
