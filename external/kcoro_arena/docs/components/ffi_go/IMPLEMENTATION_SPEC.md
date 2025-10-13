# FFI Go Dispatch: Implementation Specification

## Detailed Design

### Data Structures

#### C Coroutine Context

```c
typedef enum {
    C_CORO_INIT = 0,
    C_CORO_RUNNING,
    C_CORO_SUSPENDED,
    C_CORO_DONE,
    C_CORO_CANCELLED
} c_coro_state;

typedef struct c_coroutine_ctx {
    // Stack management
    void*          stack_base;        // malloc'd stack memory
    size_t         stack_size;        // typically 64KB
    void*          stack_top;         // current stack pointer (for debugging)
    
    // User function
    void          (*entry_func)(void*);
    void*          entry_arg;
    
    // Runtime linkage
    kcoro_t*       coro;              // underlying kcoro handle
    uint64_t       id;                // unique coroutine ID
    c_coro_state   state;             // lifecycle state
    
    // Cleanup
    void          (*cleanup_cb)(void*);  // optional cleanup callback
    void*          cleanup_arg;
    
    // Introspection
    const char*    name;              // optional debug name
    uint64_t       spawn_time_ns;     // when created
    uint64_t       total_suspend_ns;  // cumulative time suspended
    uint32_t       suspend_count;     // number of times suspended
} c_coroutine_ctx;
```

#### Global Registry

To support `kcoro_current()` and cleanup, maintain a global registry:

```c
typedef struct {
    c_coroutine_ctx**  contexts;      // array of active contexts
    size_t             capacity;      // allocated size
    size_t             count;         // current count
    pthread_mutex_t    lock;          // protects access
} c_coro_registry;

static c_coro_registry global_registry;
```

### API Implementation

#### koro_init()

```c
void koro_init(void) {
    // 1. Initialize global scheduler if not already running
    if (!global_scheduler) {
        kc_sched_options opts = {
            .workers = 4,  // default; could be env-configurable
            .stack_size = 64 * 1024,
            .use_affinity = 1
        };
        global_scheduler = kc_sched_new(&opts);
    }
    
    // 2. Initialize arena
    if (!global_arena) {
        kc_arena_opts arena_opts = {
            .num_cells = 1024,
            .workers = 2,  // arena-specific workers
            .mode = ARENA_MODE_RENDEZVOUS
        };
        global_arena = kc_arena_new(&arena_opts);
    }
    
    // 3. Initialize C coroutine registry
    pthread_mutex_init(&global_registry.lock, NULL);
    global_registry.capacity = 128;
    global_registry.contexts = calloc(128, sizeof(c_coroutine_ctx*));
    global_registry.count = 0;
    
    // 4. Set up TLS key for current coroutine
    pthread_key_create(&tls_current_coro_key, NULL);
}
```

#### koro_go()

```c
void koro_go(void (*func)(void*), void* arg) {
    // 1. Allocate context
    c_coroutine_ctx* ctx = calloc(1, sizeof(c_coroutine_ctx));
    ctx->entry_func = func;
    ctx->entry_arg = arg;
    ctx->state = C_CORO_INIT;
    ctx->spawn_time_ns = kc_now_ns();
    ctx->id = next_coro_id_atomic();
    
    // 2. Allocate stack
    ctx->stack_size = CORO_STACK_SIZE;
    ctx->stack_base = aligned_alloc(16, ctx->stack_size);
    if (!ctx->stack_base) {
        free(ctx);
        abort(); // or return error code
    }
    ctx->stack_top = (char*)ctx->stack_base + ctx->stack_size;
    
    // 3. Create kcoro coroutine with wrapper function
    ctx->coro = kcoro_new(c_func_wrapper, ctx, ctx->stack_base, ctx->stack_size);
    if (!ctx->coro) {
        free(ctx->stack_base);
        free(ctx);
        abort();
    }
    
    // 4. Register in global registry
    pthread_mutex_lock(&global_registry.lock);
    if (global_registry.count >= global_registry.capacity) {
        // Grow registry
        size_t new_cap = global_registry.capacity * 2;
        global_registry.contexts = realloc(global_registry.contexts, 
                                           new_cap * sizeof(c_coroutine_ctx*));
        global_registry.capacity = new_cap;
    }
    global_registry.contexts[global_registry.count++] = ctx;
    pthread_mutex_unlock(&global_registry.lock);
    
    // 5. Schedule on runtime
    kcoro_t* out_co = NULL;
    kc_sched_spawn(global_scheduler, &out_co, NULL);
    // Note: ctx->coro is now managed by scheduler
}
```

#### c_func_wrapper (internal)

```c
static void c_func_wrapper(void* ctx_ptr) {
    c_coroutine_ctx* ctx = (c_coroutine_ctx*)ctx_ptr;
    
    // Set TLS so koro_send/recv can find this context
    pthread_setspecific(tls_current_coro_key, ctx);
    
    // Transition to running
    ctx->state = C_CORO_RUNNING;
    
    // Invoke user function
    ctx->entry_func(ctx->entry_arg);
    
    // Mark done
    ctx->state = C_CORO_DONE;
    
    // Cleanup callback if registered
    if (ctx->cleanup_cb) {
        ctx->cleanup_cb(ctx->cleanup_arg);
    }
    
    // Unregister from global registry
    unregister_c_coroutine(ctx);
    
    // Free stack and context (scheduler will free the kcoro_t)
    free(ctx->stack_base);
    free(ctx);
    
    pthread_setspecific(tls_current_coro_key, NULL);
}
```

#### koro_send()

```c
void koro_send(int ticket, void* payload) {
    // 1. Get current context from TLS
    c_coroutine_ctx* ctx = pthread_getspecific(tls_current_coro_key);
    if (!ctx) {
        fprintf(stderr, "koro_send called outside coroutine context\n");
        abort();
    }
    
    // 2. Mark as suspended
    ctx->state = C_CORO_SUSPENDED;
    uint64_t suspend_start = kc_now_ns();
    
    // 3. Use arena send (which internally does kcoro_park)
    kc_arena_send(global_arena, ticket, payload, -1); // infinite wait
    
    // 4. Resumed! Update metrics and state
    uint64_t suspend_end = kc_now_ns();
    ctx->total_suspend_ns += (suspend_end - suspend_start);
    ctx->suspend_count++;
    ctx->state = C_CORO_RUNNING;
}
```

#### koro_recv()

```c
void* koro_recv(int ticket) {
    c_coroutine_ctx* ctx = pthread_getspecific(tls_current_coro_key);
    if (!ctx) {
        fprintf(stderr, "koro_recv called outside coroutine context\n");
        abort();
    }
    
    ctx->state = C_CORO_SUSPENDED;
    uint64_t suspend_start = kc_now_ns();
    
    void* result = kc_arena_recv(global_arena, ticket, -1);
    
    uint64_t suspend_end = kc_now_ns();
    ctx->total_suspend_ns += (suspend_end - suspend_start);
    ctx->suspend_count++;
    ctx->state = C_CORO_RUNNING;
    
    return result;
}
```

#### koro_shutdown()

```c
void koro_shutdown(void) {
    // 1. Signal arena workers to stop
    if (global_arena) {
        kc_arena_shutdown(global_arena);
    }
    
    // 2. Drain scheduler (wait for running coroutines to finish)
    if (global_scheduler) {
        kc_sched_drain(global_scheduler, 10000); // 10 sec timeout
        kc_sched_shutdown(global_scheduler);
    }
    
    // 3. Clean up any leaked contexts (shouldn't happen)
    pthread_mutex_lock(&global_registry.lock);
    for (size_t i = 0; i < global_registry.count; i++) {
        c_coroutine_ctx* ctx = global_registry.contexts[i];
        if (ctx->state != C_CORO_DONE) {
            fprintf(stderr, "Warning: coroutine %lu still active at shutdown\n", ctx->id);
        }
        free(ctx->stack_base);
        free(ctx);
    }
    free(global_registry.contexts);
    pthread_mutex_unlock(&global_registry.lock);
    pthread_mutex_destroy(&global_registry.lock);
    
    // 4. Destroy TLS key
    pthread_key_delete(tls_current_coro_key);
}
```

### Thread-Local Storage Setup

In `kcoro_core.c`:

```c
static pthread_key_t tls_current_coro_key;

void kcoro_init_tls(void) {
    pthread_key_create(&tls_current_coro_key, NULL);
}

kcoro_t* kcoro_current(void) {
    return pthread_getspecific(tls_current_coro_key);
}
```

In scheduler worker loop:

```c
static void worker_loop(sched_worker_t* w) {
    while (!atomic_load(&w->sched->stop)) {
        kcoro_t* co = rq_pop_locked(w);
        if (co) {
            pthread_setspecific(tls_current_coro_key, co); // ← SET TLS
            kcoro_switch(w->main_co, co);
            pthread_setspecific(tls_current_coro_key, NULL); // ← CLEAR TLS
            // ...
        }
    }
}
```

## Memory Management

### Stack Lifetime

- **Allocation**: `koro_go()` allocates stack via `aligned_alloc()`
- **Ownership**: Context owns the stack; scheduler owns the kcoro_t
- **Deallocation**: `c_func_wrapper()` frees stack when function returns

### Context Lifetime

1. Created in `koro_go()`
2. Registered in global registry
3. Stays alive until user function returns
4. Unregistered and freed in `c_func_wrapper()`

### Registry Cleanup

- Normal case: contexts self-unregister on completion
- Shutdown: `koro_shutdown()` scans registry for leaks
- No memory leaks if user functions complete normally

## Error Handling

### Panic Conditions

| Condition | Action | Recovery |
|-----------|--------|----------|
| `koro_send/recv` outside coroutine | `abort()` | None; programming error |
| Stack allocation failure | `abort()` | None; OOM |
| Context allocation failure | `abort()` | None; OOM |

### Graceful Failures

| Condition | Return Value | Notes |
|-----------|--------------|-------|
| `koro_try_send` when no receiver | `0` | Non-blocking; expected |
| `koro_try_recv` when no sender | `NULL` | Non-blocking; expected |

## Performance Characteristics

### Dispatch Overhead

- Stack allocation: ~50-100 ns (one-time)
- Context creation: ~20-30 ns (one-time)
- Registry insert: ~10-20 ns (one-time, amortized)
- **Total `koro_go()` cost: ~100-200 ns**

### Suspension Cost

- TLS lookup: ~5 ns
- State transition: ~2 ns
- Arena operation: ~50-200 ns (depends on match)
- **Total suspend/resume cycle: ~60-250 ns**

### Context Switch

Same as native kcoro: ~10-20 ns (using assembly fast-path)

## Debugging Support

### Introspection Functions

```c
// Get metrics for a running coroutine (by ID)
c_coro_metrics koro_get_metrics(uint64_t id);

// Dump all active coroutines to stderr
void koro_dump_active(void);

// Set debug name for current coroutine
void koro_set_name(const char* name);
```

Example dump output:

```
Active C Coroutines:
ID=1234 name=producer state=SUSPENDED suspends=42 total_suspend_time=1.2ms
ID=1235 name=consumer state=RUNNING suspends=41 total_suspend_time=1.1ms
```

## Testing Strategy

### Unit Tests

1. **Basic dispatch**: `koro_go()` → function runs → completes
2. **Send/recv**: producer/consumer pair exchanges 1000 messages
3. **Multiple tickets**: 4 coroutines using different tickets simultaneously
4. **Shutdown**: verify no leaks after `koro_shutdown()`
5. **Error cases**: call `koro_send` from main thread (expect abort)

### Stress Tests

1. Spawn 10,000 coroutines, each does 1 send/recv pair
2. Validate all complete successfully
3. Check memory usage (no leaks)
4. Compare throughput to native kcoro orchestrations

### Benchmarks

Compare:
- FFI dispatch latency vs. native kcoro spawn
- Send/recv cost vs. native channel operations
- Memory overhead (stack + context) per coroutine

Target: <10% overhead vs. native kcoro

## Integration Checklist

Before implementing:

- [ ] Audit existing kcoro_core for TLS support
- [ ] Verify scheduler can set/clear TLS on context switch
- [ ] Check if arena send/recv can be called from any kcoro context
- [ ] Design error propagation strategy (abort vs. return codes)
- [ ] Plan for graceful shutdown semantics
- [ ] Write test suite skeleton
- [ ] Document any ABI requirements for C functions

After implementing:

- [ ] Run all existing kcoro tests (ensure no regression)
- [ ] Run new FFI-specific tests
- [ ] Benchmark against native kcoro
- [ ] Profile memory usage
- [ ] Document known limitations
- [ ] Write example programs (producer/consumer, pipeline, etc.)

## Known Limitations (Future Work)

1. **No cancellation API**: C functions can't be cancelled mid-execution (would need longjmp or similar)
2. **Fixed stack size**: 64KB for all; no dynamic adjustment
3. **No exception handling**: C functions that abort() take down the runtime
4. **Single global arena**: can't have multiple independent arenas per koro_init()
5. **No priority scheduling**: all C coroutines have equal priority

These are acceptable for v1; can be addressed in future iterations if needed.
