# Descriptor Reference Counting — Verified

_Last reviewed: 2025-10-14_

This document describes the descriptor subsystem as implemented in `external/kcoro_arena/core/src/kc_desc.c`. Update it when behaviour changes.

## Overview

Descriptors provide reference-counted handles for message payloads. They enable zero-copy handoffs between senders and receivers while guarding against use-after-free.

## Core architecture

### Descriptor entry

```c
struct kc_desc_entry {
    kc_desc_entry *next;
    kc_desc_id     id;
    void          *data;
    size_t         len;
    unsigned       flags;           // KC_DESC_FLAG_ALIAS, etc.
    unsigned       arena_id;
    size_t         arena_len;
    int            owns_allocation;
    atomic_uint    refcount;
};
```

Key points:
- Intrusive hash chaining (no extra allocation for buckets).
- Atomic refcount for concurrent retain/release.
- Ownership flags distinguish arena-owned copies from external aliases.

## Descriptor types

### Copy (`kc_desc_make_copy`)

- Allocates a fresh buffer from the arena, copies payload, owns allocation.
- Safe for producers to reuse original buffer immediately.

### Alias (`kc_desc_make_alias`)

- Wraps an existing buffer without copying.
- Producer must keep the buffer alive until the descriptor refcount drops to zero.

## Reference counting protocol

- `kc_desc_retain(id)`: increments refcount; used when LRU caches or multi-subscriber paths need additional references.
- `kc_desc_release(id)`: decrements refcount; when it reaches zero, removes the entry from the hash table and frees memory if owned.

Cleanup flow:
```c
if (entry->owns_allocation) kc_arena_free(entry->arena_id, entry->data, entry->arena_len);
else if (!(entry->flags & KC_DESC_FLAG_ALIAS)) free(entry->data);
free(entry);
```

## Alias LRU cache

- Optional per-channel cache to reuse descriptor IDs for the same pointer/length pair.
- Enabled by default; cache size ~32 entries.
- On hit: retain cached descriptor and reuse ID.
- On miss: create new descriptor, insert into LRU, retaining an extra reference for the cache.

## Integration points

- Channels call `kc_desc_make_copy`/`kc_desc_make_alias` depending on payload mode.
- Select combinators retain descriptors when broadcasting to multiple consumers.
- Token kernel releases descriptors once both sender and receiver callbacks complete.

## Related docs

- [CHANNEL_DESIGN_VERIFIED.md](../channels/CHANNEL_DESIGN_VERIFIED.md)
- [OVERVIEW_VERIFIED.md](../stackless_runtime/OVERVIEW_VERIFIED.md)
