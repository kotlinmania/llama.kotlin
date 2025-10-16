// SPDX-License-Identifier: BSD-3-Clause
/* koro_task.h - Higher-level task abstraction over stackless continuations
 *
 * This provides an optional structured concurrency layer on top of the
 * raw continuation primitives. Tasks support:
 * - Structured concurrency with parent/child relationships
 * - Reference counting for shared task ownership
 * - Cancellation propagation through task trees
 * - Join semantics to wait for task completion
 *
 * Design principles:
 * - Tasks are thin wrappers around continuations
 * - All operations compose cleanly with existing scheduler
 * - Optional layer - raw continuations still available for low-level use
 */
#ifndef KORO_TASK_H
#define KORO_TASK_H

#include "kcoro_stackless.h"
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct koro_task;
typedef struct koro_task koro_task_t;

/* Task state flags */
typedef enum {
    KORO_TASK_CREATED    = 0x01,  /* Task created but not started */
    KORO_TASK_RUNNING    = 0x02,  /* Task currently executing */
    KORO_TASK_SUSPENDED  = 0x04,  /* Task suspended, waiting for event */
    KORO_TASK_COMPLETED  = 0x08,  /* Task completed successfully */
    KORO_TASK_CANCELLED  = 0x10,  /* Task was cancelled */
    KORO_TASK_FAILED     = 0x20,  /* Task failed with error */
} koro_task_state_t;

/* Task completion callback.
 * Called when task completes, is cancelled, or fails.
 * - task: The task that completed
 * - result: User-defined result value (or NULL)
 * - user_arg: User argument passed to koro_task_set_callback */
typedef void (*koro_task_completion_fn)(koro_task_t* task, void* result, void* user_arg);

/* Task structure.
 * Wraps a continuation with structured concurrency features. */
struct koro_task {
    /* Underlying continuation */
    koro_cont_t* cont;
    
    /* Structured concurrency */
    koro_task_t* parent;          /* Parent task (NULL for root tasks) */
    koro_task_t* first_child;     /* First child in linked list */
    koro_task_t* next_sibling;    /* Next sibling in parent's child list */
    
    /* Lifecycle management */
    atomic_int refcount;          /* Reference count for shared ownership */
    atomic_int state;             /* Current task state (koro_task_state_t flags) */
    
    /* Cancellation */
    atomic_int cancel_requested;  /* True if cancellation was requested */
    void* cancel_token;           /* User-defined cancellation token */
    
    /* Completion */
    void* result;                 /* User-defined result value */
    koro_task_completion_fn completion_cb;  /* Optional completion callback */
    void* completion_arg;         /* User argument for completion callback */
    
    /* Join support */
    koro_task_t** joiners;        /* Array of tasks waiting on this task */
    int joiner_count;             /* Number of waiting tasks */
    int joiner_capacity;          /* Capacity of joiners array */
};

/* ============================================================================
 * Task Lifecycle API
 * ============================================================================ */

/* Create a new task wrapping a continuation.
 * - func: Continuation step function
 * - arg: User argument passed to continuation
 * - local_size: Bytes to allocate for continuation local variables
 * - parent: Optional parent task (NULL for root task)
 * Returns task handle or NULL on failure. */
koro_task_t* koro_task_create(void* (*func)(koro_cont_t*),
                               void* arg,
                               size_t local_size,
                               koro_task_t* parent);

/* Increment task reference count.
 * Use when sharing task ownership across multiple contexts. */
void koro_task_retain(koro_task_t* task);

/* Decrement task reference count.
 * Destroys task when refcount reaches zero. */
void koro_task_release(koro_task_t* task);

/* Mark task as completed with optional result.
 * Call this when task's continuation completes (after KORO_END).
 * Usually not needed as managed tasks are cleaned up automatically. */
void koro_task_complete(koro_task_t* task, void* result);

/* Spawn a task and schedule it for execution.
 * Equivalent to koro_task_create() + schedule.
 * Returns task handle or NULL on failure. */
koro_task_t* koro_task_spawn(void* (*func)(koro_cont_t*),
                              void* arg,
                              size_t local_size,
                              koro_task_t* parent);

/* ============================================================================
 * Task Control API
 * ============================================================================ */

/* Request cancellation of a task and all its children.
 * Cancellation is cooperative - task must check cancel status.
 * Returns 0 on success, negative on error. */
int koro_task_cancel(koro_task_t* task);

/* Check if cancellation was requested for a task.
 * Call this from within task code to check for cancellation.
 * Returns non-zero if cancelled. */
int koro_task_is_cancelled(koro_task_t* task);

/* Get current task state flags.
 * Returns bitmask of koro_task_state_t values. */
int koro_task_get_state(koro_task_t* task);

/* Set completion callback for a task.
 * Callback is invoked when task completes, is cancelled, or fails.
 * Returns 0 on success, negative on error. */
int koro_task_set_callback(koro_task_t* task,
                            koro_task_completion_fn callback,
                            void* user_arg);

/* ============================================================================
 * Task Join API
 * ============================================================================ */

/* Wait for a task to complete.
 * Suspends current task until target task finishes.
 * Returns 0 on success, negative on error.
 * 
 * NOTE: Must be called from within a task's continuation function.
 * Use KORO_TASK_JOIN() macro for proper CPS transformation. */
int koro_task_join_impl(koro_cont_t* current_cont, koro_task_t* target_task);

/* Macro for joining a task within a continuation.
 * Suspends current task until target completes. */
#define KORO_TASK_JOIN(k, task) \
    do { \
        (k)->state = __LINE__; \
        case __LINE__: { \
            koro_task_t* _self_task = koro_task_from_cont(k); \
            int _join_res = koro_task_join_impl((k), (task)); \
            if (_join_res < 0) { \
                /* Join failed - continue immediately */ \
            } else if (!koro_task_get_state(task) & KORO_TASK_COMPLETED) { \
                /* Task not complete yet - suspend */ \
                return NULL; \
            } \
        } \
    } while (0)

/* ============================================================================
 * Task Introspection API
 * ============================================================================ */

/* Get task associated with a continuation.
 * Returns NULL if continuation is not wrapped in a task. */
koro_task_t* koro_task_from_cont(koro_cont_t* cont);

/* Get the currently executing task (if any).
 * Returns NULL if current continuation is not wrapped in a task. */
koro_task_t* koro_task_current(void);

/* Get task result value.
 * Valid after task completes. Returns NULL if no result set. */
void* koro_task_get_result(koro_task_t* task);

/* Count child tasks.
 * Returns number of direct children. */
int koro_task_count_children(koro_task_t* task);

#ifdef __cplusplus
}
#endif

#endif /* KORO_TASK_H */
