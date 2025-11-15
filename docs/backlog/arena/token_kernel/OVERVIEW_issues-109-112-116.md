_Backlog: related issues #109, 112, 116_

<!-- Related issues: #109, #112, #116 -->
# Token Kernel — Future Directions

_This file has been superseded by a comprehensive analysis document._

## Status Update (2025-10-31)

The three main enhancement areas requested in issues #109, #112, and #116 have been **analyzed and documented** in detail:

**See: [ENHANCEMENTS_issues-109-112-116.md](./ENHANCEMENTS_issues-109-112-116.md)**

That document provides:
- **Batch-ready wakeups**: Design, expected impact, implementation strategy
- **Descriptor metadata pooling**: Coordination with existing subsystem, slab allocator design
- **Fairness/priority hints**: Weighted round-robin scheduling, API extensions

Current verified behavior is documented in:
- [OVERVIEW_VERIFIED.md](../../../external/kcoro_arena/docs/components/token_kernel/OVERVIEW_VERIFIED.md)

## Recommended priorities

1. **Batching** (highest ROI, low risk)
2. **Descriptor pooling** (medium ROI, coordinate with descriptor subsystem)  
3. **Fairness** (workload-dependent, higher complexity)

## Other future directions

### Integration experiments

- Expose hooks so external event sources (network drivers, custom I/O) can plug into the token kernel without going through channels first.
- ✅ **Metrics counters already implemented**: matches, retries, CAS failures, publish/callback counts

### Open questions

1. Should we support multi-recipient broadcasts natively, or leave that to channel combinators?
2. Can we safely extend the token kernel to capture timeouts without complicating callback semantics?
3. Is there value in exposing a programmable backoff when bucket contention is high?

## Revision history

- **2025-10-31**: Superseded by comprehensive ENHANCEMENTS document
- **2025-10-14**: Original backlog items captured from OVERVIEW_FUTURE.md
