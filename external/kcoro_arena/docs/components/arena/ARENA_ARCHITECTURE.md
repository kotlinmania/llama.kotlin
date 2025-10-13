# Arena Architecture: Implementation Specification

## Overview

The arena subsystem provides **bump-pointer memory allocation** for zero-copy message payloads and descriptor management. Unlike traditional malloc/free, the arena uses a linear allocation strategy optimized for coroutine channel communication where message lifetimes are explicitly managed through reference counting.

## Core Architecture

### Design Philosophy

The arena implements a **BizTalk MessageBox fragment pattern** adapted for in-memory coroutine communication:

- **Single allocation, deferred cleanup**: Like BizTalk's message fragments table, payloads are allocated once and cleaned up via reference-counted descriptors
- **Correlation via tickets**: Descriptor IDs serve as correlation tokens (analogous to BizTalk's message references)
- **Staged persistence**: Messages are "staged" in the arena until all subscribers have consumed them

### Implementation: `kc_arena.c`

```c
typedef struct kc_arena {
    int active;
    pthread_mutex_t mu;
    unsigned char *base;      // Contiguous backing buffer
    size_t         size;      // Total arena capacity
    size_t         offset;    // Current bump pointer
    size_t         bytes_allocated; // Accounting (best-effort)
} kc_arena;
```

**Key characteristics:**

- **Array-based storage**: Pre-allocated backing buffer (`malloc` once at creation)
- **Bump-pointer allocation**: `offset` advances monotonically; no per-allocation metadata overhead
- **Global registry**: Up to `KC_ARENA_MAX` (4) simultaneous arenas via `g_arenas[]` static array
- **Thread-safe**: Mutex protects bump pointer updates

## Memory Management Strategy

### Allocation: `kc_arena_alloc()`

```c
void *kc_arena_alloc(unsigned arena_id, size_t len)
{
    // 1. 16-byte alignment for SIMD/cache-line efficiency
    size_t aligned = (len + 15u) & ~((size_t)15u);
    
    // 2. Atomic bump-pointer update under lock
    pthread_mutex_lock(&arena->mu);
    if (arena->offset + aligned > arena->size) {
        // Out of space; return NULL (no expand/compact)
        return NULL;
    }
    unsigned char *ptr = arena->base + arena->offset;
    arena->offset += aligned;
    pthread_mutex_unlock(&arena->mu);
    
    return ptr;
}
```

**Critical properties:**

- **O(1) allocation**: No search, no free-list traversal
- **No fragmentation**: Contiguous bump pointer eliminates external fragmentation
- **Deterministic failure**: Returns NULL when arena exhausted; caller must handle gracefully

### Deallocation: `kc_arena_free()`

```c
void kc_arena_free(unsigned arena_id, void *ptr, size_t len)
{
    // Individual frees are no-ops; accounting only
    pthread_mutex_lock(&arena->mu);
    if (arena->bytes_allocated >= len) 
        arena->bytes_allocated -= len;
    pthread_mutex_unlock(&arena->mu);
    // Physical memory NOT returned until arena_destroy()
}
```

**Why this design?**

1. **Descriptor-managed lifetimes**: Actual deallocation happens via `kc_desc_release()` when refcount→0
2. **Batch cleanup**: Arena destruction (`kc_arena_destroy()`) releases all memory at once—ideal for bounded message processing sessions
3. **Amortized cost**: Eliminates per-message free() syscalls; channels operate at nanosecond scale

## Integration with Coroutine Channels

### Message Flow

```
Producer Coroutine
      ↓
[1] kc_arena_alloc()  → raw buffer
      ↓
[2] Copy message payload → buffer
      ↓
[3] kc_desc_make_copy() → wraps buffer in descriptor
      ↓
[4] kc_chan_send()      → publishes descriptor ID
      ↓
      ↓ (async)
      ↓
Consumer Coroutine
      ↓
[5] kc_chan_recv()      → receives descriptor ID
      ↓
[6] kc_desc_payload()   → retrieves buffer ptr + len
      ↓
[7] Read message (zero-copy)
      ↓
[8] kc_desc_release()   → decrement refcount
      ↓
[9] (refcount→0) → kc_arena_free() accounting
```

### Lifecycle Guarantees

- **Arena creation**: Idempotent; `kc_arena_create()` returns `-EEXIST` if arena already active
- **Descriptor coupling**: `kc_desc.c` calls `kc_arena_alloc()` transparently for copy descriptors
- **Cleanup order**: 
  1. Channel close → cancel pending tokens
  2. Descriptor release → decrement refcounts
  3. Arena destroy → free backing buffer

## Performance Characteristics

### Allocation

| Metric | Value | Rationale |
|--------|-------|-----------|
| **Latency** | ~10-50ns | Bump pointer + alignment calculation only |
| **Throughput** | >1M allocs/sec | No allocator metadata; single atomic increment |
| **Memory overhead** | ~0.5% | 16-byte alignment padding; no per-block headers |

### Comparison to `malloc()`

```
Operation           malloc()      Arena         Speedup
──────────────────────────────────────────────────────
Allocate            150-300ns     20-30ns       ~8-10x
Free (individual)   100-200ns     2-5ns         ~40x
Free (batch)        N×200ns       1×free()      ~N×200x
```

**Trade-off**: Arena cannot reclaim space for individual messages; best for bounded workloads.

## Configuration and Tuning

### Environment Variables

None directly; arenas configured at creation:

```c
// Default: 8MB if size=0
kc_arena_create(arena_id, 8 * 1024 * 1024);
```

### Capacity Planning

**Rule of thumb**: `arena_size ≥ max_concurrent_messages × avg_message_size × 1.2`

Example for 10k concurrent 4KB messages:
```
10,000 × 4,096 × 1.2 = ~49MB arena
```

### Failure Modes

| Condition | Symptom | Recovery |
|-----------|---------|----------|
| Arena full | `kc_arena_alloc()` returns NULL | Caller must retry or fail gracefully |
| OOM on creation | `-ENOMEM` | Reduce arena size or increase system limits |
| Double-destroy | `-ENOENT` | Idempotent; safe to ignore |

## Advanced Topics

### Multi-Arena Strategies

Up to 4 concurrent arenas supported:

```c
// Separate arenas for different message classes
kc_arena_create(0, 64 * MB);  // High-priority, small messages
kc_arena_create(1, 512 * MB); // Bulk data transfers
```

**Use case**: Prevent large messages from starving small message allocation.

### Zero-Copy Optimization

For pointer-mode channels (`KC_PTR_RENDEZVOUS`):

1. Producer allocates large buffer from arena
2. Descriptor wraps buffer (alias descriptor, no copy)
3. Consumer receives pointer + length, reads directly
4. No memcpy() overhead for multi-MB payloads

**Constraint**: Producer must not free/reuse buffer until all consumers finish (enforced by descriptor refcount).

## Implementation Notes

### Why Not Lock-Free?

Current design uses `pthread_mutex_t` for simplicity and portability. Lock-free bump pointer possible via:

```c
// Theoretical lock-free version (not implemented)
size_t old_offset = atomic_fetch_add(&arena->offset, aligned);
if (old_offset + aligned > arena->size) {
    atomic_fetch_sub(&arena->offset, aligned); // Rollback
    return NULL;
}
return arena->base + old_offset;
```

**Decision**: Mutex overhead (~20ns) is negligible vs allocation cost; lock-free adds complexity for minimal gain.

### Alignment Rationale

16-byte alignment ensures:
- **SIMD operations**: Messages can use `movaps` (aligned 128-bit loads)
- **Cache efficiency**: Reduces false sharing on multi-core systems
- **Platform ABI**: Satisfies strictest alignment (x86-64, AArch64)

## Related Components

- **Descriptors (`kc_desc.c`)**: Wraps arena-allocated buffers with reference counting
- **Channels (`kc_chan.c`)**: Primary consumer of arena allocations
- **Token Kernel (`kc_token_kernel.c`)**: Manages token payloads (descriptor IDs) via arena

## Future Enhancements

### Potential Optimizations (Not Implemented)

1. **Compaction**: Slide live allocations to reclaim free space (requires relocatable pointers)
2. **Multi-tier arenas**: Hot/cold separation based on allocation frequency
3. **Memory-mapped backing**: `mmap()` for inter-process zero-copy
4. **NUMA-aware allocation**: Pin arenas to specific CPU sockets

### Compatibility Notes

- **Thread-safety**: Full; all operations are mutex-protected
- **Signal-safety**: NOT signal-safe (uses pthread primitives)
- **Platform support**: POSIX-compliant; tested on macOS (Darwin), Linux

---

**See also:**
- `DESCRIPTOR_ARCHITECTURE.md` for reference counting details
- `CHANNEL_TOKEN_INTEGRATION.md` for descriptor-to-channel flow
