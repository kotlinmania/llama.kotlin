// SPDX-License-Identifier: BSD-3-Clause
/* koro_sched_stackless.c - Stackless coroutine scheduler
 *
 * This implements a simple, assembly-free scheduler for stackless coroutines.
 * All coroutines run on the main thread's stack—no context switching required.
 */
#include "kcoro_stackless.h"
#include <stdlib.h>
#include <stdio.h>

/* Simple queue for ready coroutines */
typedef struct koro_queue {
    koro_cont_t* head;
    koro_cont_t* tail;
} koro_queue_t;

/* Global scheduler state */
static struct {
    koro_queue_t ready;
    int running;
    int initialized;
} sched = {0};

/* Initialize the stackless scheduler */
int koro_sched_init(void)
{
    if (sched.initialized) return 0;
    
    sched.ready.head = NULL;
    sched.ready.tail = NULL;
    sched.running = 0;
    sched.initialized = 1;
    
    return 0;
}

/* Enqueue a continuation to the ready queue */
void koro_sched_enqueue_ready(koro_cont_t* k)
{
    if (!k) return;
    if (k->ready_enqueued) return; /* Already in queue */
    
    k->next = NULL;
    k->ready_enqueued = 1;
    
    if (sched.ready.tail) {
        sched.ready.tail->next = k;
        sched.ready.tail = k;
    } else {
        sched.ready.head = k;
        sched.ready.tail = k;
    }
}

/* Dequeue a continuation from the ready queue */
static koro_cont_t* koro_sched_dequeue_ready(void)
{
    koro_cont_t* k = sched.ready.head;
    if (!k) return NULL;
    
    sched.ready.head = k->next;
    if (!sched.ready.head) {
        sched.ready.tail = NULL;
    }
    
    k->next = NULL;
    k->ready_enqueued = 0;
    return k;
}

/* Run the scheduler main loop until all coroutines complete.
 * This is the event loop that replaces stack-based context switching. */
int koro_run(void)
{
    if (!sched.initialized) {
        fprintf(stderr, "koro_sched_init() must be called first\n");
        return -1;
    }
    
    sched.running = 1;
    
    /* Main scheduler loop: repeatedly dequeue and step coroutines */
    while (sched.running) {
        koro_cont_t* k = koro_sched_dequeue_ready();
        
        if (!k) {
            /* No ready coroutines; all must be suspended or complete.
             * In a real implementation, we'd check if any are still alive
             * (e.g., waiting on I/O). For now, we exit. */
            break;
        }
        
        /* Execute one step of the coroutine.
         * This is just a normal C function call on our stack. */
        void* result = koro_cont_step(k);
        
        if (result != NULL) {
            /* Coroutine completed; clean it up */
            koro_cont_destroy(k);
        } else {
            /* Coroutine suspended; it will be re-enqueued by a callback
             * when its I/O operation completes. */
        }
    }
    
    sched.running = 0;
    return 0;
}

/* Stop the scheduler */
void koro_stop(void)
{
    sched.running = 0;
}

/* Spawn a new stackless coroutine.
 * This is the C-friendly API for launching a coroutine from a function pointer. */
int koro_go(void* (*func)(koro_cont_t*), void* arg, size_t local_size)
{
    if (!sched.initialized) {
        fprintf(stderr, "koro_sched_init() must be called first\n");
        return -1;
    }
    
    koro_cont_t* k = koro_cont_create((koro_step_fn)func, arg, local_size);
    if (!k) return -1;
    
    /* Enqueue the new coroutine to start execution */
    koro_sched_enqueue_ready(k);
    
    return 0;
}
