<!-- Related issues: #118 -->
# Coroutine Core — Future Ideas

_This file is exploratory; none of the items below are implemented._

## Potential upgrades

- **Cross-platform switchers:** Provide portable implementations for x86_64 and RISC-V so kcoro can run stackfully beyond ARM64.
- **Optional shadow stacks:** Investigate a shadow-stack model to detect misuse of blocking APIs while keeping stackful ergonomics.
- **Pluggable retire queue:** Allow embedders to supply their own coroutine reclamation strategy (e.g., custom allocators, tracing hooks).

## Open design questions

1. Should kcoro expose hooks for integrating with the stackless scheduler, or remain a completely separate reference runtime?
2. Is there value in surfacing a higher-level “task” layer over kcoro once structured cancellation lands in kcoro_arena?
3. Can we share more code between kcoro and kcoro_arena (e.g., metrics, logging) without introducing stackful assumptions into the stackless runtime?

Contribute experiments and rough notes here so the verified document remains a description of shipping behaviour.
