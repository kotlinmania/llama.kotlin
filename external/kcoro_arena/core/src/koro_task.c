// SPDX-License-Identifier: BSD-3-Clause
/* koro_task.c - Task abstraction implementation */

#include "koro_task.h"
#include "koro_sched_stackless.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* Thread-local current task pointer */
static _Thread_local koro_task_t* g_current_task = NULL;

/* Mutex for protecting task tree modifications */
static pthread_mutex_t g_task_tree_lock = PTHREAD_MUTEX_INITIALIZER;

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/* Add a child task to parent's child list.
 * Must be called with g_task_tree_lock held. */
static void task_add_child_locked(koro_task_t* parent, koro_task_t* child)
{
    if (!parent || !child) return;
    
    child->next_sibling = parent->first_child;
    parent->first_child = child;
    child->parent = parent;
}

/* Remove a child task from parent's child list.
 * Must be called with g_task_tree_lock held. */
static void task_remove_child_locked(koro_task_t* parent, koro_task_t* child)
{
    if (!parent || !child) return;
    
    koro_task_t** pp = &parent->first_child;
    while (*pp) {
        if (*pp == child) {
            *pp = child->next_sibling;
            child->next_sibling = NULL;
            child->parent = NULL;
            return;
        }
        pp = &(*pp)->next_sibling;
    }
}

/* Notify all joiners that a task has completed.
 * Must be called with g_task_tree_lock held. */
static void task_notify_joiners_locked(koro_task_t* task)
{
    if (!task) return;
    
    for (int i = 0; i < task->joiner_count; i++) {
        koro_task_t* joiner = task->joiners[i];
        if (joiner && joiner->cont) {
            /* Re-enqueue the joiner task */
            koro_sched_enqueue_ready(joiner->cont);
        }
    }
    
    /* Clear joiners list */
    free(task->joiners);
    task->joiners = NULL;
    task->joiner_count = 0;
    task->joiner_capacity = 0;
}

/* Cancel all child tasks recursively.
 * Must be called with g_task_tree_lock held. */
static void task_cancel_children_locked(koro_task_t* task)
{
    if (!task) return;
    
    koro_task_t* child = task->first_child;
    while (child) {
        atomic_store(&child->cancel_requested, 1);
        task_cancel_children_locked(child);
        child = child->next_sibling;
    }
}

/* Wrapper continuation step function that updates task state.
 * This wraps the user's step function to maintain task bookkeeping. */
static void* task_cont_wrapper(koro_cont_t* k)
{
    koro_task_t* task = (koro_task_t*)k->user_arg;
    if (!task) {
        /* Shouldn't happen, but handle gracefully */
        return (void*)1;
    }
    
    /* Set current task for thread-local access */
    koro_task_t* prev_task = g_current_task;
    g_current_task = task;
    
    /* Update task state to running */
    int prev_state = atomic_load(&task->state);
    if (!(prev_state & KORO_TASK_COMPLETED) && 
        !(prev_state & KORO_TASK_CANCELLED) &&
        !(prev_state & KORO_TASK_FAILED)) {
        atomic_store(&task->state, KORO_TASK_RUNNING);
    }
    
    /* Call the actual continuation step */
    void* result = koro_cont_step(k);
    
    /* Check if task completed */
    if (result != NULL || k->completed) {
        pthread_mutex_lock(&g_task_tree_lock);
        
        /* Update task state */
        if (atomic_load(&task->cancel_requested)) {
            atomic_store(&task->state, KORO_TASK_CANCELLED);
        } else {
            atomic_store(&task->state, KORO_TASK_COMPLETED);
        }
        
        /* Store result if provided */
        if (result != NULL) {
            task->result = result;
        }
        
        /* Notify joiners */
        task_notify_joiners_locked(task);
        
        /* Call completion callback if registered */
        if (task->completion_cb) {
            task->completion_cb(task, task->result, task->completion_arg);
        }
        
        /* Remove from parent's child list */
        if (task->parent) {
            task_remove_child_locked(task->parent, task);
        }
        
        pthread_mutex_unlock(&g_task_tree_lock);
    } else {
        /* Task suspended */
        atomic_store(&task->state, KORO_TASK_SUSPENDED);
    }
    
    /* Restore previous current task */
    g_current_task = prev_task;
    
    return result;
}

/* ============================================================================
 * Task Lifecycle Implementation
 * ============================================================================ */

koro_task_t* koro_task_create(void* (*func)(koro_cont_t*),
                               void* arg,
                               size_t local_size,
                               koro_task_t* parent)
{
    if (!func) {
        return NULL;
    }
    
    /* Allocate task structure */
    koro_task_t* task = (koro_task_t*)calloc(1, sizeof(koro_task_t));
    if (!task) {
        return NULL;
    }
    
    /* Create underlying continuation
     * We pass the task as user_arg so wrapper can access it */
    task->cont = koro_cont_create((koro_step_fn)func, arg, local_size);
    if (!task->cont) {
        free(task);
        return NULL;
    }
    
    /* Store original user_arg in continuation's user_data if needed */
    /* (User step function will access via k->user_arg) */
    
    /* Initialize task fields */
    atomic_init(&task->refcount, 1);
    atomic_init(&task->state, KORO_TASK_CREATED);
    atomic_init(&task->cancel_requested, 0);
    task->parent = NULL;
    task->first_child = NULL;
    task->next_sibling = NULL;
    task->cancel_token = NULL;
    task->result = NULL;
    task->completion_cb = NULL;
    task->completion_arg = NULL;
    task->joiners = NULL;
    task->joiner_count = 0;
    task->joiner_capacity = 0;
    
    /* Add to parent's child list if parent provided */
    if (parent) {
        pthread_mutex_lock(&g_task_tree_lock);
        task_add_child_locked(parent, task);
        koro_task_retain(parent);  /* Parent holds reference to child */
        pthread_mutex_unlock(&g_task_tree_lock);
    }
    
    return task;
}

void koro_task_retain(koro_task_t* task)
{
    if (!task) return;
    atomic_fetch_add(&task->refcount, 1);
}

void koro_task_release(koro_task_t* task)
{
    if (!task) return;
    
    int old_count = atomic_fetch_sub(&task->refcount, 1);
    if (old_count <= 1) {
        /* Last reference released - destroy task */
        
        pthread_mutex_lock(&g_task_tree_lock);
        
        /* Remove from parent if still attached */
        if (task->parent) {
            task_remove_child_locked(task->parent, task);
            koro_task_release(task->parent);
        }
        
        /* Cancel all children */
        task_cancel_children_locked(task);
        
        /* Release all children */
        koro_task_t* child = task->first_child;
        while (child) {
            koro_task_t* next = child->next_sibling;
            koro_task_release(child);
            child = next;
        }
        
        /* Clear joiners */
        free(task->joiners);
        
        pthread_mutex_unlock(&g_task_tree_lock);
        
        /* Destroy continuation */
        if (task->cont) {
            koro_cont_destroy(task->cont);
        }
        
        /* Free task structure */
        free(task);
    }
}

koro_task_t* koro_task_spawn(void* (*func)(koro_cont_t*),
                              void* arg,
                              size_t local_size,
                              koro_task_t* parent)
{
    koro_task_t* task = koro_task_create(func, arg, local_size, parent);
    if (!task) {
        return NULL;
    }
    
    /* Mark continuation as managed so scheduler will clean it up */
    task->cont->managed = 1;
    
    /* Store task pointer in continuation for wrapper access */
    task->cont->user_arg = (void*)task;
    
    /* Schedule task for execution */
    koro_sched_enqueue_ready(task->cont);
    
    return task;
}

/* ============================================================================
 * Task Control Implementation
 * ============================================================================ */

int koro_task_cancel(koro_task_t* task)
{
    if (!task) {
        return -1;
    }
    
    pthread_mutex_lock(&g_task_tree_lock);
    
    /* Set cancel flag */
    atomic_store(&task->cancel_requested, 1);
    
    /* Cancel all children recursively */
    task_cancel_children_locked(task);
    
    pthread_mutex_unlock(&g_task_tree_lock);
    
    /* Mark continuation as completed to stop execution */
    if (task->cont) {
        task->cont->completed = 1;
    }
    
    return 0;
}

int koro_task_is_cancelled(koro_task_t* task)
{
    if (!task) {
        return 0;
    }
    return atomic_load(&task->cancel_requested);
}

int koro_task_get_state(koro_task_t* task)
{
    if (!task) {
        return 0;
    }
    return atomic_load(&task->state);
}

int koro_task_set_callback(koro_task_t* task,
                            koro_task_completion_fn callback,
                            void* user_arg)
{
    if (!task) {
        return -1;
    }
    
    task->completion_cb = callback;
    task->completion_arg = user_arg;
    return 0;
}

/* ============================================================================
 * Task Join Implementation
 * ============================================================================ */

int koro_task_join_impl(koro_cont_t* current_cont, koro_task_t* target_task)
{
    if (!current_cont || !target_task) {
        return -1;
    }
    
    /* Check if target already completed */
    int state = atomic_load(&target_task->state);
    if (state & (KORO_TASK_COMPLETED | KORO_TASK_CANCELLED | KORO_TASK_FAILED)) {
        return 0;  /* Already done */
    }
    
    /* Get current task */
    koro_task_t* current_task = koro_task_from_cont(current_cont);
    if (!current_task) {
        return -1;  /* Can only join from within a task */
    }
    
    pthread_mutex_lock(&g_task_tree_lock);
    
    /* Add current task to target's joiners list */
    if (target_task->joiner_count >= target_task->joiner_capacity) {
        int new_capacity = target_task->joiner_capacity == 0 ? 4 : target_task->joiner_capacity * 2;
        koro_task_t** new_joiners = (koro_task_t**)realloc(
            target_task->joiners,
            new_capacity * sizeof(koro_task_t*)
        );
        if (!new_joiners) {
            pthread_mutex_unlock(&g_task_tree_lock);
            return -1;
        }
        target_task->joiners = new_joiners;
        target_task->joiner_capacity = new_capacity;
    }
    
    target_task->joiners[target_task->joiner_count++] = current_task;
    
    pthread_mutex_unlock(&g_task_tree_lock);
    
    return 0;  /* Caller should suspend */
}

/* ============================================================================
 * Task Introspection Implementation
 * ============================================================================ */

koro_task_t* koro_task_from_cont(koro_cont_t* cont)
{
    if (!cont) {
        return NULL;
    }
    /* Task pointer stored in continuation's user_arg by spawn */
    return (koro_task_t*)cont->user_arg;
}

koro_task_t* koro_task_current(void)
{
    return g_current_task;
}

void* koro_task_get_result(koro_task_t* task)
{
    if (!task) {
        return NULL;
    }
    return task->result;
}

int koro_task_count_children(koro_task_t* task)
{
    if (!task) {
        return 0;
    }
    
    int count = 0;
    pthread_mutex_lock(&g_task_tree_lock);
    
    koro_task_t* child = task->first_child;
    while (child) {
        count++;
        child = child->next_sibling;
    }
    
    pthread_mutex_unlock(&g_task_tree_lock);
    
    return count;
}
