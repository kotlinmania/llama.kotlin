// SPDX-License-Identifier: BSD-3-Clause
/* example_task.c - Practical example of task abstraction
 *
 * Demonstrates a parallel tree processing scenario with:
 * - Parent task that spawns multiple child workers
 * - Graceful cancellation propagation
 * - Task completion tracking
 */
#include "koro_task.h"
#include "koro_sched_stackless.h"
#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
 * Worker Task: Processes a slice of data
 * ============================================================================ */

struct worker_locals {
    int worker_id;
    int items_processed;
    int max_items;
};

static void* worker_task_step(koro_cont_t* k)
{
    struct worker_locals* local = (struct worker_locals*)k->user_data;
    int* worker_id = (int*)k->user_arg;
    
    KORO_BEGIN(k);
    
    /* Initialize */
    local->worker_id = *worker_id;
    local->items_processed = 0;
    local->max_items = 5;
    
    printf("  [Worker %d] Starting work\n", local->worker_id);
    
    /* Process items, checking for cancellation */
    while (local->items_processed < local->max_items) {
        /* Check if cancellation was requested */
        koro_task_t* self = koro_task_current();
        if (self && koro_task_is_cancelled(self)) {
            printf("  [Worker %d] Cancelled after %d items\n",
                   local->worker_id, local->items_processed);
            break;
        }
        
        /* Simulate work */
        printf("  [Worker %d] Processing item %d/%d\n",
               local->worker_id, local->items_processed + 1, local->max_items);
        
        local->items_processed++;
        
        /* Yield to allow other tasks to run */
        KORO_YIELD(k);
    }
    
    if (local->items_processed == local->max_items) {
        printf("  [Worker %d] Completed all %d items\n",
               local->worker_id, local->items_processed);
    }
    
    KORO_END(k);
}

/* ============================================================================
 * Coordinator Task: Spawns workers and handles cancellation
 * ============================================================================ */

struct coordinator_locals {
    int step;
    int num_workers;
    koro_task_t** workers;
    int should_cancel;
};

static void* coordinator_task_step(koro_cont_t* k)
{
    struct coordinator_locals* local = (struct coordinator_locals*)k->user_data;
    
    KORO_BEGIN(k);
    
    /* Initialize */
    local->step = 0;
    local->num_workers = 3;
    local->should_cancel = 0;
    
    printf("[Coordinator] Starting with %d workers\n", local->num_workers);
    
    /* Allocate worker array */
    local->workers = (koro_task_t**)calloc(local->num_workers, sizeof(koro_task_t*));
    if (!local->workers) {
        printf("[Coordinator] Failed to allocate worker array\n");
        KORO_END(k);
    }
    
    /* Get self for parent link */
    koro_task_t* self = koro_task_current();
    
    /* Spawn worker tasks */
    printf("[Coordinator] Spawning workers...\n");
    static int worker_ids[10];  // Static to persist across yields
    for (int i = 0; i < local->num_workers; i++) {
        worker_ids[i] = i + 1;
        local->workers[i] = koro_task_spawn(
            worker_task_step,
            &worker_ids[i],
            sizeof(struct worker_locals),
            self  /* This coordinator is parent */
        );
        
        if (!local->workers[i]) {
            printf("[Coordinator] Failed to spawn worker %d\n", i + 1);
        }
    }
    
    /* Verify children were spawned */
    int child_count = koro_task_count_children(self);
    printf("[Coordinator] Spawned %d child tasks\n", child_count);
    
    /* Let workers run for a while */
    KORO_YIELD(k);
    KORO_YIELD(k);
    
    /* Simulate a condition that triggers cancellation */
    printf("[Coordinator] Simulating cancellation condition...\n");
    local->should_cancel = 1;
    
    if (local->should_cancel) {
        printf("[Coordinator] Cancelling all workers\n");
        
        /* Cancel all workers - could also just cancel self to propagate */
        for (int i = 0; i < local->num_workers; i++) {
            if (local->workers[i]) {
                koro_task_cancel(local->workers[i]);
            }
        }
    }
    
    /* Continue for a bit to see cancellation take effect */
    KORO_YIELD(k);
    KORO_YIELD(k);
    
    /* Check worker states */
    printf("[Coordinator] Checking worker states:\n");
    for (int i = 0; i < local->num_workers; i++) {
        if (local->workers[i]) {
            int state = koro_task_get_state(local->workers[i]);
            printf("  Worker %d: state = %d %s\n",
                   i + 1, state,
                   (state & KORO_TASK_CANCELLED) ? "(CANCELLED)" :
                   (state & KORO_TASK_COMPLETED) ? "(COMPLETED)" : "");
        }
    }
    
    /* Cleanup */
    free(local->workers);
    
    printf("[Coordinator] Done\n");
    
    KORO_END(k);
}

/* ============================================================================
 * Main Program
 * ============================================================================ */

int main(void)
{
    printf("=== Task Abstraction Example: Parallel Workers ===\n\n");
    
    /* Initialize scheduler */
    if (koro_sched_init() != 0) {
        fprintf(stderr, "Failed to initialize scheduler\n");
        return 1;
    }
    
    /* Spawn coordinator task (which will spawn workers) */
    printf("Main: Spawning coordinator task...\n\n");
    koro_task_t* coordinator = koro_task_spawn(
        coordinator_task_step,
        NULL,
        sizeof(struct coordinator_locals),
        NULL  /* Root task */
    );
    
    if (!coordinator) {
        fprintf(stderr, "Failed to spawn coordinator\n");
        return 1;
    }
    
    /* Run scheduler to execute all tasks */
    printf("Main: Running scheduler...\n\n");
    koro_run();
    
    printf("\nMain: Scheduler finished\n");
    
    /* Cleanup */
    koro_task_release(coordinator);
    
    printf("\n=== Example Complete ===\n");
    return 0;
}
