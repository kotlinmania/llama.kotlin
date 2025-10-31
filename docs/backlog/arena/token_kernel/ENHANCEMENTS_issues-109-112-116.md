_Backlog: related issues #109, 112, 116_

<!-- Related issues: #109, #112, #116 -->
# Token Kernel Enhancements: Batching, Descriptor Pooling, and Fairness

_Analysis and design for three key performance enhancements to the token kernel._

## Executive Summary

This document evaluates three proposed enhancements to the token kernel to improve throughput, reduce memory overhead, and provide fairness guarantees:

1. **Batch-ready wakeups**: Dequeue multiple ready tokens before scheduler wakeup
2. **Descriptor metadata pooling**: Pool descriptor structures inside token kernel
3. **Fairness/priority hints**: Balance long-lived streams vs. bursty traffic

## Current Architecture

The token kernel (as of 2025-10-31) implements:

- **Bucket-based hash table** with 1024 buckets for token storage
- **Freelist recycling** for `kc_token_block` structures (basic pooling already exists)
- **Ready queue with dedicated worker thread** for callback processing
- **Single-token dequeue model**: Worker processes one token at a time via `ready_dequeue()`
- **Mutex-based synchronization** for buckets, freelist, and ready queue
- **Event notification hooks** for observability

Key structures:
```c
typedef struct kc_token_block {
    kc_token_block       *next_hash;
    struct kc_chan       *channel;
    kc_payload            payload;
    kc_token_resume_fn    resume_cb;
    void                 *resume_ctx;
    kc_token_id_t         id;
} kc_token_block;

typedef struct kc_token_ready_queue {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    kc_token_block *head;
    kc_token_block *tail;
    int             stop;
} kc_token_ready_queue;
```

## Enhancement 1: Batch-Ready Wakeups

### Problem Statement

The current worker thread processes tokens one at a time:
```c
static void *kc_token_worker_main(void *arg) {
    for (;;) {
        kc_token_block *blk = ready_dequeue(&g_kernel.ready_queue);
        if (!blk) break;
        kc_token_process_block(blk);  // Single token
    }
    return NULL;
}
```

During high-throughput bursts, this creates:
- **Lock contention**: Acquiring `ready_queue.mu` for each token
- **Scheduler overhead**: Each callback may trigger scheduler wakeup individually
- **Cache thrashing**: Frequent mutex lock/unlock cycles

### Proposed Solution

Implement batch dequeue with configurable batch size:

```c
#define KC_TOKEN_BATCH_SIZE 16  // Tunable parameter

typedef struct kc_token_batch {
    kc_token_block *blocks[KC_TOKEN_BATCH_SIZE];
    size_t count;
} kc_token_batch;

static size_t ready_dequeue_batch(kc_token_ready_queue *q, 
                                  kc_token_batch *batch,
                                  size_t max_count) {
    pthread_mutex_lock(&q->mu);
    size_t dequeued = 0;
    
    while (dequeued < max_count && q->head) {
        batch->blocks[dequeued++] = q->head;
        q->head = q->head->next_hash;
    }
    
    if (!q->head) q->tail = NULL;
    pthread_mutex_unlock(&q->mu);
    
    batch->count = dequeued;
    return dequeued;
}

static void *kc_token_worker_main_batched(void *arg) {
    kc_token_batch batch;
    for (;;) {
        size_t count = ready_dequeue_batch(&g_kernel.ready_queue, 
                                          &batch, 
                                          KC_TOKEN_BATCH_SIZE);
        if (count == 0) break;
        
        // Process entire batch before next lock acquisition
        for (size_t i = 0; i < count; i++) {
            kc_token_process_block(batch.blocks[i]);
        }
    }
    return NULL;
}
```

### Expected Impact

**Throughput improvements:**
- **Reduced lock acquisitions**: 16x fewer for full batches
- **Better cache locality**: Process contiguous blocks
- **Amortized scheduler overhead**: Batch callback execution

**Latency considerations:**
- **Potential head-of-line blocking**: First token in batch waits for batch to fill
- **Mitigation**: Use adaptive batch sizing or timeout-based flush

**Metrics to track:**
- Average batch size under load
- Lock contention time (before/after)
- P50, P95, P99 callback latency
- Throughput (callbacks/sec) under various loads

### Implementation Strategy

1. **Phase 1**: Add batch dequeue infrastructure with opt-in flag
2. **Phase 2**: Implement adaptive batching (size based on queue depth)
3. **Phase 3**: Add timeout-based flush for low-latency scenarios
4. **Phase 4**: Performance benchmarking and tuning

### Configuration Options

```c
typedef struct kc_token_kernel_config {
    size_t batch_size;           // 1=no batching, 16=default, 64=max
    int adaptive_batching;       // 0=fixed, 1=queue-depth-based
    uint64_t flush_timeout_ns;   // Max wait for partial batch
} kc_token_kernel_config;
```

## Enhancement 2: Descriptor Metadata Pooling

### Problem Statement

Currently, the token kernel has basic pooling for `kc_token_block` via freelist, but descriptor metadata (`kc_payload`) may still cause:
- **Retain/release churn**: High-frequency channels constantly allocate/free descriptors
- **Memory fragmentation**: Variable-sized payloads fragment the heap
- **Cache misses**: Scattered descriptor metadata

The descriptor subsystem (`kc_desc.c`) already provides:
```c
kc_desc_id kc_desc_make_copy(const void *src, size_t len);
kc_desc_id kc_desc_make_alias(void *ptr, size_t len);
void kc_desc_retain(kc_desc_id id);
void kc_desc_release(kc_desc_id id);
```

### Coordination with Existing Descriptor System

**Key principle**: Avoid double-pooling. The token kernel should:
1. Pool `kc_token_block` structures (already done)
2. Reuse payload buffers when possible (NEW)
3. Coordinate with descriptor system for zero-copy paths

### Proposed Solution

Add payload buffer pooling with size classes:

```c
#define KC_PAYLOAD_POOL_SMALL 64      // 0-64 bytes
#define KC_PAYLOAD_POOL_MEDIUM 512    // 65-512 bytes  
#define KC_PAYLOAD_POOL_LARGE 4096    // 513-4096 bytes

typedef struct kc_payload_slab {
    void *buffer;
    size_t capacity;
    int in_use;
    struct kc_payload_slab *next;
} kc_payload_slab;

typedef struct kc_payload_pool {
    pthread_mutex_t mu;
    kc_payload_slab *small_head;
    kc_payload_slab *medium_head;
    kc_payload_slab *large_head;
    size_t small_count;
    size_t medium_count;
    size_t large_count;
} kc_payload_pool;

static void *payload_pool_alloc(kc_payload_pool *pool, size_t len);
static void payload_pool_free(kc_payload_pool *pool, void *ptr, size_t len);
```

### Integration Points

**With existing descriptor system:**
- Use descriptor aliasing for large payloads (avoid copies)
- Pool only small/medium copy buffers in token kernel
- Maintain descriptor refcounts when using pooled buffers

**With token kernel:**
- Allocate from pool in `kc_token_kernel_publish_send()`
- Return to pool in `kc_token_process_block()` after callback
- Track pool utilization in metrics

### Expected Impact

**Memory efficiency:**
- **Reduced allocations**: Reuse buffers for typical payload sizes
- **Less fragmentation**: Fixed-size slabs prevent heap fragmentation
- **Better cache locality**: Hot buffers stay in cache

**Performance:**
- **Faster alloc/free**: Pool access vs. malloc/free
- **Reduced retain/release overhead**: Fewer descriptor operations

**Metrics to track:**
- Pool hit rate (successful reuse)
- Average pool utilization
- Allocation latency (pooled vs. malloc)

### Implementation Strategy

1. **Phase 1**: Implement basic slab pool for small payloads
2. **Phase 2**: Add medium/large size classes
3. **Phase 3**: Tune pool sizes based on workload profiling
4. **Phase 4**: Coordinate with descriptor alias LRU cache

### Open Questions

1. Should pool capacity be fixed or grow dynamically?
2. How to handle pathological cases (all large payloads)?
3. Integration with descriptor compression (backlog issue #116)?

## Enhancement 3: Fairness/Priority Hints

### Problem Statement

The current token kernel processes tokens in FIFO order from the ready queue. This can cause:
- **Starvation**: Long-lived streams may monopolize worker thread
- **Unfairness**: Bursty traffic gets delayed behind steady streams
- **No prioritization**: All channels treated equally regardless of importance

### Proposed Solution

Add optional priority levels and fairness scheduling:

```c
typedef enum {
    KC_TOKEN_PRIORITY_LOW    = 0,
    KC_TOKEN_PRIORITY_NORMAL = 1,
    KC_TOKEN_PRIORITY_HIGH   = 2,
    KC_TOKEN_PRIORITY_COUNT  = 3
} kc_token_priority_t;

typedef struct kc_token_block_v2 {
    kc_token_block       *next_hash;
    struct kc_chan       *channel;
    kc_payload            payload;
    kc_token_resume_fn    resume_cb;
    void                 *resume_ctx;
    kc_token_id_t         id;
    kc_token_priority_t   priority;      // NEW
    uint64_t              enqueue_time;  // NEW: for fairness tracking
} kc_token_block_v2;

typedef struct kc_priority_ready_queue {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    kc_token_block *heads[KC_TOKEN_PRIORITY_COUNT];
    kc_token_block *tails[KC_TOKEN_PRIORITY_COUNT];
    size_t counts[KC_TOKEN_PRIORITY_COUNT];
    int stop;
} kc_priority_ready_queue;
```

### Scheduling Algorithms

**Option 1: Strict Priority**
- Always dequeue highest-priority non-empty queue
- Risk: Low-priority starvation

**Option 2: Weighted Round-Robin**
- Process N high-priority, M normal, 1 low-priority per cycle
- Example: 4:2:1 ratio
- Guarantees progress for all priorities

**Option 3: Deficit Round-Robin**
- Track "quantum" for each priority level
- Accumulate deficit when queue is empty
- Prevents starvation while respecting priorities

**Recommended: Weighted Round-Robin** for simplicity and fairness

### API Extensions

```c
// Extended publish API with priority hint
kc_ticket kc_token_kernel_publish_send_pri(struct kc_chan *ch,
                                           void *ptr,
                                           size_t len,
                                           kc_token_resume_fn resume_cb,
                                           void *user_ctx,
                                           kc_token_priority_t priority);

// Channel-level default priority
void kc_chan_set_priority(struct kc_chan *ch, kc_token_priority_t priority);
```

### Expected Impact

**Fairness improvements:**
- **Controlled starvation prevention**: Guaranteed progress for all priorities
- **Latency-sensitive paths**: Mark critical channels as high-priority
- **Backpressure management**: Reduce low-priority traffic under load

**Complexity costs:**
- **Additional state**: Priority queues and scheduling logic
- **Lock complexity**: May need per-priority locks for scalability

**Metrics to track:**
- Per-priority queue depths
- Wait time distribution by priority
- Starvation events (tokens waiting > threshold)

### Implementation Strategy

1. **Phase 1**: Design priority queue structure and API
2. **Phase 2**: Implement weighted round-robin scheduler
3. **Phase 3**: Add channel-level priority configuration
4. **Phase 4**: Expose fairness metrics for monitoring

### Configuration Options

```c
typedef struct kc_token_fairness_config {
    int enabled;                        // 0=FIFO, 1=priority
    size_t high_weight;                 // Default: 4
    size_t normal_weight;               // Default: 2
    size_t low_weight;                  // Default: 1
    uint64_t starvation_threshold_ms;   // Alert threshold
} kc_token_fairness_config;
```

## Integration and Testing Plan

### Testing Strategy

**Batching tests:**
1. Benchmark single-token vs. batch processing throughput
2. Measure latency distribution under various batch sizes
3. Stress test with bursty traffic patterns

**Descriptor pooling tests:**
1. Memory profiling with/without pooling
2. Measure allocation latency for various payload sizes
3. Pool exhaustion scenarios

**Fairness tests:**
1. Multi-priority workload simulations
2. Starvation detection and prevention
3. Latency consistency across priority levels

### Metrics Dashboard

Add to `kc_token_kernel_metrics`:
```c
typedef struct kc_token_kernel_metrics_v2 {
    // Existing metrics
    uint64_t matches_total;
    uint64_t retries_total;
    uint64_t cas_failures_total;
    uint64_t publish_send_total;
    uint64_t publish_recv_total;
    uint64_t callback_total;
    uint64_t cancel_total;
    
    // Batching metrics
    uint64_t batch_dequeues_total;
    uint64_t tokens_per_batch_sum;     // For average calculation
    uint64_t batches_partial;          // Batches < max size
    
    // Pooling metrics
    uint64_t pool_allocs_total;
    uint64_t pool_hits_total;
    uint64_t pool_misses_total;
    uint64_t pool_current_size;
    
    // Fairness metrics
    uint64_t high_priority_tokens;
    uint64_t normal_priority_tokens;
    uint64_t low_priority_tokens;
    uint64_t starvation_events;
} kc_token_kernel_metrics_v2;
```

## Recommendations

### Priority 1: Batching (Highest ROI)
- **Complexity**: Low (mostly dequeue logic change)
- **Impact**: High (significant throughput improvement)
- **Risk**: Low (easy to feature-flag and revert)

**Action**: Implement batch dequeue with adaptive sizing

### Priority 2: Descriptor Pooling (Medium ROI)
- **Complexity**: Medium (coordinate with descriptor subsystem)
- **Impact**: Medium (memory efficiency, reduced fragmentation)
- **Risk**: Medium (must not conflict with existing pooling)

**Action**: Start with small payload pool, measure impact

### Priority 3: Fairness (Lower ROI)
- **Complexity**: High (requires scheduling logic redesign)
- **Impact**: Medium (workload-dependent)
- **Risk**: Medium (complexity may introduce bugs)

**Action**: Design phase first, implement only if workload demonstrates need

## Zero-Spin and Callback Semantics Preservation

All enhancements must maintain:

1. **Zero-spin property**: No busy waiting in worker thread
   - Batch dequeue uses `pthread_cond_wait()` when queue empty
   - Timeout-based flush uses `pthread_cond_timedwait()`

2. **Callback semantics**: Callbacks fire in dedicated worker context
   - Batch processing preserves callback ordering
   - Priority scheduling may reorder, but within documented rules

3. **Event notification**: All state transitions still trigger events
   - Batching does not suppress per-token events
   - Metrics accurately reflect individual operations

## Related Work

- [Token Kernel Implementation (verified)](../../../external/kcoro_arena/docs/components/token_kernel/IMPLEMENTATION_VERIFIED.md)
- [Descriptor Refcounting backlog](./DESCRIPTOR_REFCOUNTING_issues-110-116.md)
- [Scheduler (verified)](../../../external/kcoro_arena/docs/components/stackless_runtime/SCHEDULER_VERIFIED.md)

## Revision History

- **2025-10-31**: Initial analysis document created for issues #109, #112, #116
