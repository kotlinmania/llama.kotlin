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

## Observability and metrics

The descriptor subsystem exposes telemetry for understanding memory usage and cache efficiency.

### Metrics structure

```c
typedef struct kc_desc_metrics {
    uint64_t alias_created_total;   // Total alias descriptors created
    uint64_t copy_created_total;    // Total copy descriptors created
    uint64_t retain_total;          // Total retain operations
    uint64_t release_total;         // Total release operations
    uint64_t descriptor_evicts;     // Descriptors evicted (refcount reached 0)
    uint64_t lookup_hits;           // Successful descriptor lookups
    uint64_t lookup_misses;         // Failed descriptor lookups
} kc_desc_metrics;
```

### API functions

```c
/* Get current metrics snapshot (thread-safe) */
int kc_desc_get_metrics(kc_desc_metrics *out);

/* Reset all metrics to zero (thread-safe) */
void kc_desc_reset_metrics(void);
```

### Enabling metrics

Metrics collection uses the same runtime configuration as the token kernel:

```json
{
  "channel": {
    "metrics": {
      "auto_enable": true
    }
  }
}
```

When disabled, metrics have zero runtime overhead.

### Interpretation guide

- **alias_created_total / copy_created_total**: Track allocation patterns. High alias rates indicate efficient zero-copy paths; high copy rates suggest safety-first strategies or incompatible lifetime constraints.
- **retain_total / release_total**: Indicate reference counting activity. Imbalance (retain > release) may reveal leaks; properly balanced workloads show equal totals over time.
- **descriptor_evicts**: Count final releases (refcount → 0). Should roughly match creation totals in steady state.
- **lookup_hits / lookup_misses**: Measure hash table efficiency. High hit rates (>95%) indicate good descriptor locality; increasing misses may suggest ID reuse bugs or premature eviction.

### LRU cache efficiency

The alias LRU cache hit rate can be estimated from descriptor metrics:

```c
kc_desc_metrics m;
kc_desc_get_metrics(&m);

double total_lookups = (double)(m.lookup_hits + m.lookup_misses);
double hit_rate = total_lookups > 0 ? (m.lookup_hits / total_lookups) * 100.0 : 0.0;

printf("Descriptor cache hit rate: %.2f%%\n", hit_rate);
printf("Total evictions: %llu (avg refcount lifetime)\n", 
       (unsigned long long)m.descriptor_evicts);
```

### Usage example

```c
kc_desc_metrics metrics;
if (kc_desc_get_metrics(&metrics) == 0) {
    printf("Descriptor stats:\n");
    printf("  Created: alias=%llu, copy=%llu\n",
           (unsigned long long)metrics.alias_created_total,
           (unsigned long long)metrics.copy_created_total);
    printf("  Refcounting: retain=%llu, release=%llu, evicts=%llu\n",
           (unsigned long long)metrics.retain_total,
           (unsigned long long)metrics.release_total,
           (unsigned long long)metrics.descriptor_evicts);
    printf("  Lookups: hits=%llu, misses=%llu\n",
           (unsigned long long)metrics.lookup_hits,
           (unsigned long long)metrics.lookup_misses);
    
    if (metrics.retain_total != metrics.release_total) {
        printf("  WARNING: Refcount imbalance detected!\n");
    }
}
```

### Tuning guidance

- **High miss rates**: Consider increasing hash bucket count (requires code change) or reviewing descriptor ID generation strategy.
- **Excessive evictions**: May indicate short-lived descriptors or small LRU cache. Monitor eviction rate vs. creation rate.
- **Copy vs. alias ratio**: Optimize for more aliases if payload lifetime permits; use copies when safety is paramount.

## Related docs

- [CHANNEL_DESIGN_VERIFIED.md](../channels/CHANNEL_DESIGN_VERIFIED.md)
- [OVERVIEW_VERIFIED.md](../stackless_runtime/OVERVIEW_VERIFIED.md)
- [Token kernel metrics](../token_kernel/IMPLEMENTATION_VERIFIED.md#observability-and-metrics)
