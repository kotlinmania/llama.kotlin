_Backlog: related issues #109, 112, 116_

<!-- Related issues: #109, #112, #116 -->
# Token Kernel — Future Directions

_Explore ideas here; keep shipping behaviour in `OVERVIEW_VERIFIED.md`._

## Possible enhancements

- **Batch matching:** Investigate batching multiple ready continuations before enqueuing to reduce scheduler wakeups during bursts.
- **Descriptor pooling:** Explore pooling zero-copy descriptor metadata inside the token kernel to reduce retain/release churn for high-frequency channels.
- **Fairness knobs:** Allow optional priority hints so long-lived streams do not starve bursty traffic (and vice versa).

## Integration experiments

- Expose hooks so external event sources (network drivers, custom I/O) can plug into the token kernel without going through channels first.
- Add metrics counters (matches, retries, CAS failures) for integration with runtime observability.

## Open questions

1. Should we support multi-recipient broadcasts natively, or leave that to channel combinators?
2. Can we safely extend the state machine to capture timeouts without complicating the CAS logic?
3. Is there value in exposing a programmable backoff when `-EBUSY` happens frequently?
