# Architectural Index

This is a single place to navigate the architecture docs by component. It links to the ground‑truth VERIFIED documents and to their corresponding backlog items stored under `docs/backlog/`.

Note on convention:
- `*_VERIFIED.md` reflects implemented behavior and tests.
- Backlog files live in `docs/backlog/.../*_issues-<ids>.md` and capture aspirational work tied to GitHub issues.

**Arena / Stackless Runtime**
- Verified
  - [Overview](external/kcoro_arena/docs/components/stackless_runtime/OVERVIEW_VERIFIED.md)
  - [Continuation Guide](external/kcoro_arena/docs/components/stackless_runtime/CONTINUATION_GUIDE_VERIFIED.md)
  - [Continuation Model](external/kcoro_arena/docs/components/stackless_runtime/CONTINUATION_MODEL_VERIFIED.md)
  - [Scheduler](external/kcoro_arena/docs/components/stackless_runtime/SCHEDULER_VERIFIED.md)
- Backlog
  - [Overview — issues 107, 117](docs/backlog/arena/stackless_runtime/OVERVIEW_issues-107-117.md)
  - [Continuation Guide — issue 114](docs/backlog/arena/stackless_runtime/CONTINUATION_GUIDE_issues-114.md)
  - [Continuation Model — issues 107, 117](docs/backlog/arena/stackless_runtime/CONTINUATION_MODEL_issues-107-117.md)
  - [Scheduler — issues 108, 113](docs/backlog/arena/stackless_runtime/SCHEDULER_issues-108-113.md)

**Arena / Token Kernel**
- Verified
  - [Overview](external/kcoro_arena/docs/components/token_kernel/OVERVIEW_VERIFIED.md)
  - [Implementation](external/kcoro_arena/docs/components/token_kernel/IMPLEMENTATION_VERIFIED.md)
  - [Send/Receive Flow](external/kcoro_arena/docs/components/token_kernel/TOKEN_FLOW_ANALYSIS_VERIFIED.md)
- Backlog
  - [Overview — issues 109, 112, 116](docs/backlog/arena/token_kernel/OVERVIEW_issues-109-112-116.md)
  - [Implementation — issues 109, 112](docs/backlog/arena/token_kernel/IMPLEMENTATION_issues-109-112.md)
  - [Flow Analysis — issue 111](docs/backlog/arena/token_kernel/TOKEN_FLOW_ANALYSIS_issues-111.md)

**Arena / Channels**
- Verified
  - [Design](external/kcoro_arena/docs/components/channels/CHANNEL_DESIGN_VERIFIED.md)
- Backlog
  - [Design — issue 115](docs/backlog/arena/channels/CHANNEL_DESIGN_issues-115.md)

**Arena / Descriptors**
- Verified
  - [Reference Counting](external/kcoro_arena/docs/components/descriptors/DESCRIPTOR_REFCOUNTING_VERIFIED.md)
- Backlog
  - [Reference Counting — issues 110, 116](docs/backlog/arena/descriptors/DESCRIPTOR_REFCOUNTING_issues-110-116.md)

**Legacy kcoro / Core**
- Verified
  - [Core](external/kcoro/docs/components/core/CORE_VERIFIED.md)
- Backlog
  - [Core — issue 118](docs/backlog/kcoro/core/CORE_issues-118.md)

