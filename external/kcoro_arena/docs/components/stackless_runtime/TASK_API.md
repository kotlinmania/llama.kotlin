# Task API — Higher-Level Abstraction over Continuations

_Status: Experimental (Issue #117)_

## Overview

The task API provides an optional structured concurrency layer on top of the stackless continuation primitives. It adds:

- **Parent-child relationships** for task trees
- **Reference counting** for shared ownership
- **Cancellation propagation** through task hierarchies
- **Join semantics** to wait for task completion

## Design Philosophy

The task abstraction is a **thin wrapper** over continuations:

- Tasks wrap `koro_cont_t` structures
- All task operations compose with existing scheduler and macros
- Optional layer - raw continuations remain available for performance-critical code
- Minimal memory overhead (~330 bytes per task vs ~200 bytes for raw continuation)

## API Functions

### Task Creation and Lifecycle

```c
/* Create a task without scheduling it */
koro_task_t* koro_task_create(void* (*func)(koro_cont_t*),
                               void* arg,
                               size_t local_size,
                               koro_task_t* parent);

/* Create and immediately schedule a task */
koro_task_t* koro_task_spawn(void* (*func)(koro_cont_t*),
                              void* arg,
                              size_t local_size,
                              koro_task_t* parent);

/* Reference counting */
void koro_task_retain(koro_task_t* task);
void koro_task_release(koro_task_t* task);

/* Mark task as completed (optional - for explicit state tracking) */
void koro_task_complete(koro_task_t* task, void* result);
```

### Task Control

```c
/* Request cancellation (cooperative - task must check status) */
int koro_task_cancel(koro_task_t* task);

/* Check if cancellation was requested */
int koro_task_is_cancelled(koro_task_t* task);

/* Get task state flags */
int koro_task_get_state(koro_task_t* task);

/* Set completion callback */
int koro_task_set_callback(koro_task_t* task,
                            koro_task_completion_fn callback,
                            void* user_arg);
```

### Task Introspection

```c
/* Get task from continuation */
koro_task_t* koro_task_from_cont(koro_cont_t* cont);

/* Get currently executing task (if any) */
koro_task_t* koro_task_current(void);

/* Get task result */
void* koro_task_get_result(koro_task_t* task);

/* Count child tasks */
int koro_task_count_children(koro_task_t* task);
```

## Usage Patterns

### Basic Task Spawning

```c
struct worker_locals {
    int id;
    int items_processed;
};

static void* worker_step(koro_cont_t* k) {
    struct worker_locals* local = k->user_data;
    int* id = (int*)k->user_arg;
    
    KORO_BEGIN(k);
    
    local->id = *id;
    local->items_processed = 0;
    
    while (local->items_processed < 10) {
        /* Do work */
        local->items_processed++;
        KORO_YIELD(k);
    }
    
    KORO_END(k);
}

void spawn_workers(void) {
    koro_sched_init();
    
    static int worker1_id = 1;
    static int worker2_id = 2;
    
    koro_task_t* t1 = koro_task_spawn(worker_step, &worker1_id, 
                                       sizeof(struct worker_locals), NULL);
    koro_task_t* t2 = koro_task_spawn(worker_step, &worker2_id,
                                       sizeof(struct worker_locals), NULL);
    
    koro_run();
    
    koro_task_release(t1);
    koro_task_release(t2);
}
```

### Parent-Child Task Relationships

```c
static void* parent_step(koro_cont_t* k) {
    KORO_BEGIN(k);
    
    /* Get self to use as parent for children */
    koro_task_t* self = koro_task_from_cont(k);
    
    /* Spawn child tasks */
    koro_task_t* child1 = koro_task_spawn(child_step, data1, size, self);
    koro_task_t* child2 = koro_task_spawn(child_step, data2, size, self);
    
    /* Parent continues execution */
    KORO_YIELD(k);
    
    /* Children are tracked */
    int child_count = koro_task_count_children(self);
    printf("Have %d children\n", child_count);
    
    KORO_END(k);
}
```

### Cooperative Cancellation

```c
static void* cancellable_step(koro_cont_t* k) {
    struct locals* local = k->user_data;
    
    KORO_BEGIN(k);
    
    local->i = 0;
    while (local->i < 100) {
        /* Check for cancellation */
        koro_task_t* self = koro_task_from_cont(k);
        if (self && koro_task_is_cancelled(self)) {
            printf("Cancelled at iteration %d\n", local->i);
            break;
        }
        
        /* Do work */
        local->i++;
        KORO_YIELD(k);
    }
    
    KORO_END(k);
}

/* From another task or external code */
koro_task_cancel(task);  /* Request cancellation */
```

## Current Limitations

1. **Task Completion Tracking**: Task state updates are best-effort. The scheduler manages continuation lifecycle independently, so task state may not always perfectly reflect continuation state.

2. **Current Task Context**: `koro_task_current()` returns NULL because the scheduler doesn't currently track which continuation is executing. Use `koro_task_from_cont(k)` within continuation functions instead.

3. **Join Operations**: The `KORO_TASK_JOIN()` macro is defined but join support requires additional scheduler integration.

4. **No Automatic Cleanup**: When a continuation completes, the scheduler cleans up the continuation but not necessarily the task wrapper. Use reference counting carefully.

## Recommended Usage

For most use cases, the task API provides:

✅ **Structured task hierarchies** via parent/child relationships  
✅ **Cancellation propagation** for coordinated shutdown  
✅ **Reference counting** for shared ownership  
✅ **Introspection** of task trees

For performance-critical code or when completion timing is critical, continue using raw continuations.

## Future Enhancements

- Tighter scheduler integration for automatic task completion tracking
- Join operation support with proper suspend/resume
- Task-local storage
- Priority-based scheduling
- Task groups and barriers

## Related Documentation

- [Continuation Model (verified)](./CONTINUATION_MODEL_VERIFIED.md)
- [Scheduler (verified)](./SCHEDULER_VERIFIED.md)
- [Continuation Guide (verified)](./CONTINUATION_GUIDE_VERIFIED.md)
- Source: `external/kcoro_arena/include/koro_task.h`
- Source: `external/kcoro_arena/core/src/koro_task.c`
