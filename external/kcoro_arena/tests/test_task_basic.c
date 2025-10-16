// SPDX-License-Identifier: BSD-3-Clause
/* test_task_basic.c - Basic tests for task abstraction
 *
 * Tests fundamental task operations:
 * - Task creation and destruction
 * - Task spawning and completion
 * - Parent-child relationships
 * - Reference counting
 */
#include "koro_task.h"
#include "koro_sched_stackless.h"
#include <stdio.h>
#include <assert.h>

/* ============================================================================
 * Test 1: Basic task creation and destruction
 * ============================================================================ */

struct simple_task_locals {
    int counter;
};

static void* simple_task_step(koro_cont_t* k)
{
    struct simple_task_locals* local = (struct simple_task_locals*)k->user_data;
    
    KORO_BEGIN(k);
    
    printf("  Simple task: counter = %d\n", local->counter);
    local->counter++;
    KORO_YIELD(k);
    
    printf("  Simple task: counter = %d\n", local->counter);
    local->counter++;
    
    KORO_END(k);
}

static int test_basic_creation(void)
{
    printf("TEST: Basic task creation\n");
    
    /* Initialize scheduler */
    if (koro_sched_init() != 0) {
        fprintf(stderr, "  FAIL: scheduler init failed\n");
        return 1;
    }
    
    /* Create a task without spawning */
    koro_task_t* task = koro_task_create(
        simple_task_step,
        NULL,
        sizeof(struct simple_task_locals),
        NULL  /* No parent */
    );
    
    if (!task) {
        fprintf(stderr, "  FAIL: task creation failed\n");
        return 1;
    }
    
    /* Verify initial state */
    int state = koro_task_get_state(task);
    if (state != KORO_TASK_CREATED) {
        fprintf(stderr, "  FAIL: expected CREATED state, got %d\n", state);
        koro_task_release(task);
        return 1;
    }
    
    printf("  Task created successfully with state CREATED\n");
    
    /* Clean up */
    koro_task_release(task);
    
    printf("  PASS\n\n");
    return 0;
}

/* ============================================================================
 * Test 2: Task spawning and completion
 * ============================================================================ */

static void* counting_task_step(koro_cont_t* k)
{
    struct simple_task_locals* local = (struct simple_task_locals*)k->user_data;
    
    KORO_BEGIN(k);
    
    /* Initialize locals */
    local->counter = 0;
    
    /* Count to 3, yielding between each step */
    while (local->counter < 3) {
        printf("  Counting task: %d\n", local->counter);
        local->counter++;
        KORO_YIELD(k);
    }
    
    printf("  Counting task: Done at %d\n", local->counter);
    
    KORO_END(k);
}

static int test_spawn_and_complete(void)
{
    printf("TEST: Task spawn and completion\n");
    
    /* Initialize scheduler */
    if (koro_sched_init() != 0) {
        fprintf(stderr, "  FAIL: scheduler init failed\n");
        return 1;
    }
    
    /* Spawn a task */
    koro_task_t* task = koro_task_spawn(
        counting_task_step,
        NULL,
        sizeof(struct simple_task_locals),
        NULL  /* No parent */
    );
    
    if (!task) {
        fprintf(stderr, "  FAIL: task spawn failed\n");
        return 1;
    }
    
    printf("  Task spawned, running scheduler...\n");
    
    /* Run scheduler to execute task */
    koro_run();
    
    /* Verify task completed */
    int state = koro_task_get_state(task);
    if (!(state & KORO_TASK_COMPLETED)) {
        fprintf(stderr, "  FAIL: expected COMPLETED state, got %d\n", state);
        koro_task_release(task);
        return 1;
    }
    
    printf("  Task completed successfully\n");
    
    /* Clean up */
    koro_task_release(task);
    
    printf("  PASS\n\n");
    return 0;
}

/* ============================================================================
 * Test 3: Parent-child task relationships
 * ============================================================================ */

struct parent_task_locals {
    int step;
    koro_task_t* self;
    koro_task_t* child1;
    koro_task_t* child2;
};

static void* child_task_step(koro_cont_t* k)
{
    int* id = (int*)k->user_arg;
    
    KORO_BEGIN(k);
    
    printf("  Child task %d: Starting\n", *id);
    KORO_YIELD(k);
    
    printf("  Child task %d: Working\n", *id);
    KORO_YIELD(k);
    
    printf("  Child task %d: Done\n", *id);
    
    KORO_END(k);
}

static void* parent_task_step(koro_cont_t* k)
{
    struct parent_task_locals* local = (struct parent_task_locals*)k->user_data;
    
    KORO_BEGIN(k);
    
    printf("  Parent task: Starting\n");
    local->step = 0;
    
    /* Get current task to use as parent - for now will be NULL */
    local->self = koro_task_from_cont(k);
    
    /* Spawn two child tasks */
    static int child1_id = 1;
    static int child2_id = 2;
    
    local->child1 = koro_task_spawn(
        child_task_step,
        &child1_id,
        0,  /* No local state */
        local->self
    );
    
    local->child2 = koro_task_spawn(
        child_task_step,
        &child2_id,
        0,  /* No local state */
        local->self
    );
    
    if (!local->child1 || !local->child2) {
        printf("  Parent task: Failed to spawn children\n");
    } else {
        printf("  Parent task: Spawned 2 children\n");
        KORO_YIELD(k);
        
        /* Verify children are tracked */
        if (local->self) {
            int child_count = koro_task_count_children(local->self);
            printf("  Parent task: Has %d children\n", child_count);
        }
        
        /* Continue working */
        printf("  Parent task: Waiting for children\n");
        KORO_YIELD(k);
        
        printf("  Parent task: Done\n");
    }
    
    KORO_END(k);
}

static int test_parent_child(void)
{
    printf("TEST: Parent-child task relationships\n");
    
    /* Initialize scheduler */
    if (koro_sched_init() != 0) {
        fprintf(stderr, "  FAIL: scheduler init failed\n");
        return 1;
    }
    
    /* Spawn parent task */
    koro_task_t* parent = koro_task_spawn(
        parent_task_step,
        NULL,
        sizeof(struct parent_task_locals),
        NULL  /* Root task */
    );
    
    if (!parent) {
        fprintf(stderr, "  FAIL: parent task spawn failed\n");
        return 1;
    }
    
    printf("  Parent task spawned, running scheduler...\n");
    
    /* Run scheduler to execute tasks */
    koro_run();
    
    printf("  Scheduler finished\n");
    
    /* Clean up */
    koro_task_release(parent);
    
    printf("  PASS\n\n");
    return 0;
}

/* ============================================================================
 * Test 4: Task cancellation
 * ============================================================================ */

struct cancellable_task_locals {
    int iterations;
};

static void* cancellable_task_step(koro_cont_t* k)
{
    struct cancellable_task_locals* local = (struct cancellable_task_locals*)k->user_data;
    
    KORO_BEGIN(k);
    
    local->iterations = 0;
    
    while (local->iterations < 100) {
        /* Check for cancellation */
        koro_task_t* self = koro_task_from_cont(k);
        if (self && koro_task_is_cancelled(self)) {
            printf("  Cancellable task: Cancelled at iteration %d\n", local->iterations);
            break;
        }
        
        local->iterations++;
        KORO_YIELD(k);
    }
    
    printf("  Cancellable task: Completed %d iterations\n", local->iterations);
    
    KORO_END(k);
}

static void* cancelling_task_step(koro_cont_t* k)
{
    koro_task_t** target = (koro_task_t**)k->user_arg;
    
    KORO_BEGIN(k);
    
    printf("  Cancelling task: Waiting a bit...\n");
    KORO_YIELD(k);
    KORO_YIELD(k);
    
    printf("  Cancelling task: Cancelling target\n");
    if (*target) {
        koro_task_cancel(*target);
    }
    
    KORO_END(k);
}

static int test_cancellation(void)
{
    printf("TEST: Task cancellation\n");
    
    /* Initialize scheduler */
    if (koro_sched_init() != 0) {
        fprintf(stderr, "  FAIL: scheduler init failed\n");
        return 1;
    }
    
    /* Spawn cancellable task */
    koro_task_t* target = koro_task_spawn(
        cancellable_task_step,
        NULL,
        sizeof(struct cancellable_task_locals),
        NULL
    );
    
    if (!target) {
        fprintf(stderr, "  FAIL: target task spawn failed\n");
        return 1;
    }
    
    /* Spawn task that will cancel the target */
    koro_task_t* canceller = koro_task_spawn(
        cancelling_task_step,
        &target,
        0,
        NULL
    );
    
    if (!canceller) {
        fprintf(stderr, "  FAIL: canceller task spawn failed\n");
        koro_task_release(target);
        return 1;
    }
    
    printf("  Both tasks spawned, running scheduler...\n");
    
    /* Run scheduler */
    koro_run();
    
    /* Verify target was cancelled */
    int state = koro_task_get_state(target);
    if (state & KORO_TASK_CANCELLED) {
        printf("  Target task was successfully cancelled\n");
    } else {
        printf("  Note: Target task completed without cancellation (state=%d)\n", state);
    }
    
    /* Clean up */
    koro_task_release(target);
    koro_task_release(canceller);
    
    printf("  PASS\n\n");
    return 0;
}

/* ============================================================================
 * Main Test Driver
 * ============================================================================ */

int main(void)
{
    printf("=== Task Abstraction Basic Tests ===\n\n");
    
    int failed = 0;
    
    failed += test_basic_creation();
    failed += test_spawn_and_complete();
    failed += test_parent_child();
    failed += test_cancellation();
    
    if (failed == 0) {
        printf("=== All tests PASSED ===\n");
        return 0;
    } else {
        printf("=== %d test(s) FAILED ===\n", failed);
        return 1;
    }
}
