# Documentation Guide

Welcome! This project hosts a stackless, zero-spin coroutine engine we build new features on.

To keep things approachable, we’ve grouped the docs by subsystem. Start with the quick map below, then dive into the component folders for deeper design notes, APIs, and examples.

  Quick map (pick the doc that answers your question first)

  | Area | Why you’d read it | Starting point |
  |------|--------------------|----------------|
  | Getting oriented | Understand what “stackless” buys us | [`external/arena/docs/components/stackless_runtime/OVERVIEW.md`](../external/arena/docs/components/stackless_runtime/OVERVIEW.md) |
  | Coroutine core | Learn how continuation records, schedulers, and callbacks fit together | `external/arena/docs/components/stackless_runtime/` |
  | Channels and tokens | See how rendezvous, buffering, and the token kernel cooperate | `external/arena/docs/components/channels/` and `external/arena/docs/components/token_kernel/` |
  | Zero-copy descriptors | Work with arena-backed payloads safely and efficiently | `external/arena/docs/components/descriptors/` |
  | Migration & roadmap | Check what’s next or how to port older code | `external/docs/components/roadmap/` and the issue tracker |

  _Verified vs. future docs_: when you see `*_VERIFIED.md`, it matches shipping code. The sibling `*_FUTURE.md` files capture ideas and open questions so we can brainstorm without muddying the ground truth.

  Connectedness (stackless pieces at a glance)

  ```mermaid
  flowchart LR
      subgraph Stackless Runtime
          Scheduler -->|resume/park| Continuation
          Continuation -->|suspend| Scheduler
          Channel --> TokenKernel
          TokenKernel --> Scheduler
          Channel --> Descriptor
      end
      Descriptor --> ArenaStorage
  ```

## Writing Style

When you add or update documentation, aim for a “teach the next engineer” voice:

1. Start with a plain-language description before diving into precise details.
2. Define jargon the first time it appears.
3. Add a short example, checklist, or diagram whenever you describe a process.
4. Keep technical facts exact—even when the tone is friendly.

That balance helps future contributors pick things up quickly while keeping the engineering story accurate.

## Still Looking for Something?

If you can’t find the answer you need:

- Skim the component README files (each folder has a short “What lives here?” section).
- Search the `external/arena/docs` for keywords.
- If you discover a gap, please open an issue or add a note directly—future you will thank you for it.

Happy hacking!
