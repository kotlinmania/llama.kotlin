# Kotlin Shared-Memory Channel Specification (BizTalk-Inspired)

## Background

The existing Kotlin SWAR channel implementation delivers nanosecond-level latency (≈22 GB/s) with segmented buffers and coroutine waiters. For the upcoming ticket-based shared-memory queue, we align with kcoro lab experiments (token VM, waiter tokens) and BizTalk-style correlation tables:

- **Waiter tokens**: deterministic state machine (INIT → ENQUEUED → CLAIMED → CANCELLED).
- **Rendezvous cell**: single shared record per slot tracking sender/receiver/payload/match counters.
- **Arena ledger**: ram-backed ticket table for zero-copy payloads, akin to BizTalk MessageBox fragments.

## Goals

1. Implement a coroutine-friendly shared arena so producers/consumers coordinate via a ticket table instead of raw buffer indices.
2. Match Kotlin’s channel semantics (suspend send/receive, cancellation) while enabling zero-copy payloads and multi-subscriber throughput.
3. Expose observability hooks (matches, cancels, backlog) so tools can inspect the queue like BizTalk’s MessageBox.
4. Preserve compatibility with SWAR optimizations and high-level coroutine APIs.

## Architecture Overview

```
+-----------------------+           +-----------------------+
|  Producer Coroutine   |           |  Consumer Coroutine   |
|  - suspending send()  |           |  - suspending receive()|
|  - obtains ticket     |           |  - claims ticket      |
+-----------+-----------+           +-----------+-----------+
            \                           /
             \                         /
         +-------------------------------+
         |   Arena-backed Ticket Table   |
         |   (Array<RendezvousCell>)     |
         +-------------------------------+
                   |             |
                   |             |
            +---------------+ +---------------+
            | Metrics/      | | SWAR Worker   |
            | Inspection    | | Coroutines    |
            +---------------+ +---------------+
```

### Core Components

1. **RendezvousCell** (per ticket)
   - State: `AtomicInt` (values: EMPTY, SENDER_READY, RECEIVER_READY, MATCHED, CANCELLED).
   - Fields:
     ```kotlin
     data class RendezvousCell(
         val state: AtomicInt = AtomicInt(State.EMPTY.id),
         val senderToken: AtomicRef<WaiterToken?> = AtomicRef(null),
         val receiverToken: AtomicRef<WaiterToken?> = AtomicRef(null),
         val payload: AtomicRef<Payload> = AtomicRef(Payload.None),
         val metrics: CellMetrics = CellMetrics()
     )
     ```
   - Metrics: `matches`, `cancels`, `zdescMatches` (atomic increments).

2. **WaiterToken**
   - State machine: Kotlin sealed class or enum plus atomics.
   - Stores coroutine `Continuation` and cancellation handler.
   - Methods: `publish()`, `claim()`, `cancel()` returning boolean to reflect state transitions.

3. **Payload**
   - Union of `ByteArray` reference or zero-copy descriptor (`Pointer + length`).
   - Zero-copy path ensures pointer lifetime is managed by the arena allocator.

4. **Arena**
   - Allocates fixed-size pages (e.g., 4 KB). Each page holds a number of `RendezvousCell`s and optional external payload descriptors.
   - Maintains a free list/ticket table for quick ticket assignment.
   - Supports telemetry: backlog depth, free vs used tickets, cancellation counts.

5. **SWAR Workers**
   - Set of coroutines (possibly pinned to threads) scanning assigned ticket ranges for state transitions.
   - On match: resume corresponding producer/consumer via stored `Continuation`.
   - On cancellation: rotate through backlog to reclaim tickets.

6. **Public API**
   ```kotlin
   interface SharedArenaChannel<T> {
       suspend fun send(value: T)
       suspend fun receive(): T
       fun trySend(value: T): Boolean
       fun tryReceive(): T?
       fun close()
       fun dumpMetrics(): ArenaMetrics
   }
   ```

## State Machine Details

### States

| State             | Description                                     |
|-------------------|-------------------------------------------------|
| EMPTY             | Slot free; no send/recv published.              |
| SENDER_READY      | Sender token enqueued with payload.             |
| RECEIVER_READY    | Receiver token waiting for payload.             |
| MATCHED           | Payload delivered; both tokens claimed.         |
| CANCELLED         | One side cancelled; payload cleared.            |

### State Transitions (Pseudo-code)

```kotlin
fun senderArrive(cell: RendezvousCell, token: WaiterToken, payload: Payload): Boolean {
    when (cell.state.value) {
        EMPTY -> if (token.publish()) {
                    cell.senderToken.value = token
                    cell.payload.value = payload
                    return cell.state.compareAndSet(EMPTY, SENDER_READY)
                 }
        RECEIVER_READY -> if (cell.receiverToken.value?.claim() == true) {
                              cell.payload.value = payload
                              cell.state.value = MATCHED
                              cell.metrics.matches.incrementAndGet()
                              return true
                          }
    }
    return false
}
```

```kotlin
fun receiverArrive(cell: RendezvousCell, token: WaiterToken, outPayload: AtomicRef<Payload>): Boolean {
    when (cell.state.value) {
        EMPTY -> if (token.publish()) {
                    cell.receiverToken.value = token
                    return cell.state.compareAndSet(EMPTY, RECEIVER_READY)
                 }
        SENDER_READY -> if (cell.senderToken.value?.claim() == true) {
                            outPayload.value = cell.payload.value
                            cell.state.value = MATCHED
                            cell.metrics.matches.incrementAndGet()
                            return true
                        }
    }
    return false
}
```

Cancellation turns token status to CANCELLED and sets `cell.state = CANCELLED`, clearing payload and incrementing cancel metrics.

## Coroutine Integration

- `send`/`receive` suspending functions register a `WaiterToken` with the current `Continuation`.
- If the state machine fails to match immediately, suspend the coroutine and park the token.
- SWAR worker coroutine scans cells, matching tokens and resuming `Continuation`s.
- Cancellation handler invokes `WaiterToken.cancel()` ensuring slot reclamation.

## Arena Allocation Strategy

- Pre-allocate pages: e.g., 64 MB arena → 16 KB pages containing 128 cells each.
- Maintain `AtomicInt` free index per page; when `send`/`receive` needs a ticket, fetch next free cell.
- Optional journaling: append events (ticket allocated, matched, cancelled) to a lightweight log for recovery/debug.

## Observability & Metrics

Expose structured metrics:

```kotlin
data class ArenaMetrics(
    val totalTickets: Int,
    val freeTickets: Int,
    val matches: Long,
    val cancels: Long,
    val zeroCopyMatches: Long,
    val busiestWorkerId: Int,
    val perWorkerQueueDepth: IntArray
)
```

- `dumpMetrics()` walks pages to compute totals.
- Offer hooks for monitoring tools (e.g., channel inspector) akin to BizTalk MessageBox view.

## Compatibility with Existing Kotlin Channels

- Implement as a new channel variant or backend under the existing API (`BufferedChannel`).
- Fallback to standard Kotlin implementation when environment lacks shared memory or zero-copy features.
- Provide `SharedArenaChannelFactory(capacity, payloadMode)` returning a `SendChannel`/`ReceiveChannel` facade.

## Testing Strategy

1. **Unit Tests**: deterministic state transitions, cancellation, zero-copy payload delivery.
2. **Concurrency Tests**: multiple producers/consumers with random delays verifying no blocking or ticket leaks.
3. **Stress Tests**: millions of send/receive cycles, metrics validation.
4. **Performance Benchmarks**: compare throughput/latency against Kotlin’s existing channel implementation; monitor memory usage.

## Implementation Plan

1. Build core data structures in Kotlin/Native (RendezvousCell, WaiterToken, Arena).
2. Integrate into `SharedArenaChannel` with suspending APIs; use atomic state machine.
3. Implement SWAR worker coroutines scanning assigned cell ranges.
4. Add observability (metrics, dumps); bridge to existing channel inspector tools.
5. Validate zero-copy payload support (ByteArray vs native pointer) with tests.
6. Benchmark and iterate; decide later if token VM mirror is needed for further optimization.

## Open Questions

- How many worker coroutines/threads per arena? (Configurable; start with core count.)
- How to expose zero-copy descriptors to Kotlin safely? (Use `Pinned` memory or `CPointer` with lifecycle management.)
- Do we need journaling for durability? (Optional; can be a separate feature flag.)
- Interaction with Kotlin structured concurrency and cancellation? (Ensure tokens register with parent job/scopes.)

---

This specification captures the planned integration path, ensuring the BizTalk-inspired ticketed shared-memory design remains documented and actionable for future work.
