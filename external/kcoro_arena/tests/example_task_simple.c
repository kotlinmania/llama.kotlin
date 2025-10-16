// SPDX-License-Identifier: BSD-3-Clause
/* example_task_simple.c - Simple demonstration of task API features
 *
 * This example demonstrates:
 * - Task creation and reference counting
 * - Parent-child relationships
 * - Task introspection (counting children)
 * - Cancellation requests
 *
 * NOTE: This example doesn't run the scheduler to avoid completion
 * tracking issues. It demonstrates the API surface only.
 */
#include "koro_task.h"
#include "koro_sched_stackless.h"
#include <stdio.h>
#include <stdlib.h>

/* Simple worker continuation */
static void* worker_step(koro_cont_t* k)
{
    (void)k;  /* Unused in this demo */
    return NULL;  /* Would normally suspend/resume */
}

int main(void)
{
    printf("=== Task API Simple Example ===\n\n");
    
    /* Initialize scheduler (required for task creation) */
    if (koro_sched_init() != 0) {
        fprintf(stderr, "Failed to initialize scheduler\n");
        return 1;
    }
    
    printf("1. Creating root task\n");
    koro_task_t* root = koro_task_create(
        worker_step,
        NULL,
        0,  /* No local state */
        NULL  /* No parent - this is root */
    );
    
    if (!root) {
        fprintf(stderr, "   Failed to create root task\n");
        return 1;
    }
    printf("   ✓ Root task created (refcount=1)\n");
    
    /* Demonstrate reference counting */
    printf("\n2. Testing reference counting\n");
    koro_task_retain(root);
    printf("   ✓ Retained root task (refcount=2)\n");
    
    /* Create child tasks */
    printf("\n3. Creating child tasks\n");
    koro_task_t* child1 = koro_task_create(
        worker_step,
        NULL,
        0,
        root  /* Parent is root */
    );
    
    koro_task_t* child2 = koro_task_create(
        worker_step,
        NULL,
        0,
        root  /* Parent is root */
    );
    
    if (!child1 || !child2) {
        fprintf(stderr, "   Failed to create child tasks\n");
        koro_task_release(root);
        koro_task_release(root);
        return 1;
    }
    printf("   ✓ Created 2 child tasks\n");
    
    /* Count children */
    int child_count = koro_task_count_children(root);
    printf("   ✓ Root task has %d children\n", child_count);
    
    /* Demonstrate cancellation API */
    printf("\n4. Testing cancellation\n");
    if (koro_task_cancel(child1) == 0) {
        printf("   ✓ Requested cancellation of child1\n");
    }
    
    if (koro_task_is_cancelled(child1)) {
        printf("   ✓ child1 shows cancelled status\n");
    }
    
    if (!koro_task_is_cancelled(child2)) {
        printf("   ✓ child2 is not cancelled\n");
    }
    
    /* Demonstrate state inspection */
    printf("\n5. Inspecting task states\n");
    int root_state = koro_task_get_state(root);
    int child1_state = koro_task_get_state(child1);
    int child2_state = koro_task_get_state(child2);
    
    printf("   Root state: %d (%s)\n", root_state,
           (root_state & KORO_TASK_CREATED) ? "CREATED" : "OTHER");
    printf("   Child1 state: %d (%s)\n", child1_state,
           (child1_state & KORO_TASK_CANCELLED) ? "CANCELLED" : "OTHER");
    printf("   Child2 state: %d (%s)\n", child2_state,
           (child2_state & KORO_TASK_CREATED) ? "CREATED" : "OTHER");
    
    /* Cleanup with reference counting */
    printf("\n6. Cleanup\n");
    printf("   Releasing child1...\n");
    koro_task_release(child1);
    
    printf("   Releasing child2...\n");
    koro_task_release(child2);
    
    printf("   Child count after releasing children: %d\n",
           koro_task_count_children(root));
    
    printf("   Releasing root (first ref)...\n");
    koro_task_release(root);
    
    printf("   Releasing root (second ref - should destroy)...\n");
    koro_task_release(root);
    
    printf("\n=== Task API features demonstrated successfully ===\n");
    printf("\nKey features shown:\n");
    printf("  ✓ Task creation with parent-child relationships\n");
    printf("  ✓ Reference counting (retain/release)\n");
    printf("  ✓ Cancellation requests (cooperative)\n");
    printf("  ✓ Task introspection (state, child count)\n");
    printf("  ✓ Proper cleanup via refcounting\n");
    
    return 0;
}
