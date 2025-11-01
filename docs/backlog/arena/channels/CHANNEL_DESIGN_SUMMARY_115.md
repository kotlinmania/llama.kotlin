# Channel Backlog Design Summary (Issue #115)

_Date: 2025-11-01_

## Overview

This document summarizes the design work completed for issue #115, which asked to move wishlist items from `CHANNEL_DESIGN_FUTURE.md` into the issue tracker and expand them with detailed design proposals.

## Completed Work

### 1. Documentation Organization
- ✅ Fixed stub reference in `CHANNEL_DESIGN_FUTURE.md` to correctly point to `CHANNEL_DESIGN_issues-115.md` (not `CHANNEL_DESIGN_FUTURE_issues-115.md`)
- ✅ Verified backlog file location and naming follows project conventions per AGENTS.md documentation workflow

### 2. Fairness Knobs Design

**Problem Addressed:**
Current FIFO channel wait queues can starve high-priority operations when mixed with bulk traffic.

**Design Proposals:**
- **Option A (Recommended):** Priority-aware wait queues with 3 levels (high/normal/low)
  - Add `priority` field to `kc_pending_send` and `kc_pending_recv` structures
  - New APIs: `kc_chan_send_priority()`, `kc_chan_recv_priority()`
  - Default priority = 128 for backward compatibility

- **Option B (Alternative):** Time-based aging system
  - Track enqueue time and automatically boost priority of long-waiting operations
  - No API changes required, automatic fairness

**Recommendation:** Start with Option A, add Option B if starvation observed in production.

### 3. Timeout Helper APIs

**Problem Addressed:**
No high-level timeout support in channel APIs; users must integrate directly with token kernel or use select patterns.

**Design Proposals:**
- **Option A (Recommended):** Inline timeout parameters
  - New APIs: `kc_chan_send_timeout()`, `kc_chan_recv_timeout()`
  - Returns `-ETIME` on timeout
  - Simple, familiar pattern (like POSIX socket timeouts)

- **Option B (Alternative):** Context-based deadline
  - Pass deadline context object to operations
  - Supports complex patterns (cascade timeouts, shared deadlines)

- **Option C (Status Quo+):** Improve select-with-timeout documentation
  - No new APIs, better examples

**Recommendation:** Implement Option A for 80% use case (simple single-channel timeout). Document Option C for complex multi-channel patterns.

**Integration Notes:**
- Coordinate with token kernel timeout experiments (issues #109, #112, #116)
- Requires careful cancellation handling to avoid races
- Test timeout races (timeout fires exactly as rendezvous completes)

### 4. Broadcast Semantics

**Problem Addressed:**
Current channels are point-to-point. Many use cases need one-to-many broadcast.

**Design Proposals:**
- **Option A:** Dedicated `KC_BROADCAST` channel type
  - Native broadcast with subscription model
  - Efficient but adds complexity
  - Memory overhead (N copies per message)

- **Option B (Recommended):** Fan-out combinator utility
  - Library function: `kc_chan_fanout()` sends to multiple channels
  - Reuses existing primitives
  - Two variants: best-effort and all-or-nothing

- **Option C (Future):** Full pub-sub layer
  - Topic-based routing, filtering, QoS
  - Out of scope for core channel primitives

**Recommendation:** Start with Option B as library utility. Evaluate Option A if performance becomes bottleneck. Option C deferred to higher-level frameworks.

## Implementation Status

**Current State:**
- All design proposals are **exploratory and not implemented**
- No shipping code changes
- No updates to `CHANNEL_DESIGN_VERIFIED.md` (correctly reflects only implemented behavior)

**Next Steps (if prioritized):**
1. Prototype fairness implementation in feature branch
2. Create API RFC for timeout helpers with error handling semantics
3. Implement and benchmark fan-out combinator (Option B)
4. Integration testing with token kernel timeout experiments
5. Update VERIFIED docs only after implementation ships

## Related Issues

- #109, #112, #116: Token kernel timeout experiments (coordinate for timeout helpers)
- #115: This issue (channel backlog design)

## References

- Backlog file: `docs/backlog/arena/channels/CHANNEL_DESIGN_issues-115.md`
- Verified doc: `external/kcoro_arena/docs/components/channels/CHANNEL_DESIGN_VERIFIED.md`
- Token kernel backlog: `docs/backlog/arena/token_kernel/OVERVIEW_issues-109-112-116.md`
- Implementation: `external/kcoro_arena/core/src/kc_chan.c`
- Internal structures: `external/kcoro_arena/core/src/kc_chan_internal.h`

## Outcomes

The issue acceptance criteria have been met:

✅ **Design optional fairness knobs** - Two approaches designed with clear recommendations and trade-offs

✅ **Evaluate timeout helpers** - Three API approaches evaluated with implementation notes and coordination plan

✅ **Scope broadcast semantics** - Three patterns analyzed with recommendation for fan-out combinator

✅ **Document outcomes** - This summary document plus detailed proposals in backlog file

⚠️ **Merge shipping changes into VERIFIED** - Not applicable: no implementation, all proposals are exploratory. VERIFIED document correctly reflects only shipping behavior per documentation workflow.

## Conclusion

The design phase for issue #115 is complete. All three enhancement areas have detailed proposals with implementation approaches, trade-offs, and recommendations. These proposals are ready for review and prioritization by the team. If any proposals are implemented, those shipping changes would then be documented in `CHANNEL_DESIGN_VERIFIED.md` per the close-out workflow in AGENTS.md.
