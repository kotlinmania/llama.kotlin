// SPDX-License-Identifier: BSD-3-Clause
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>

#include "kcoro_core.h"
#include "kcoro_sched.h"

/* Thread-local current coroutine */
static __thread kcoro_t* current_kcoro = NULL;
/* Thread-local main coroutine (yield target) */
static __thread kcoro_t* main_kcoro = NULL;

/* Coroutine ID counter */
static uint64_t next_kcoro_id = 1;

kcoro_t* kcoro_create_main(void)
{
    kcoro_t* main_co = (kcoro_t*)calloc(1, sizeof(kcoro_t));
    if (!main_co) return NULL;

    /* Initialize main coroutine */
    main_co->next_step = NULL;
    main_co->state = KCORO_RUNNING;
    main_co->user_data = NULL;
    main_co->id = 0;     /* Special ID for main */
    main_co->main_co = NULL;  /* Main has no parent */
    main_co->name = "main";
    main_co->ready_enqueued = false;
    atomic_init(&main_co->running_flag, 0);
    atomic_init(&main_co->refcount, 1);
    main_co->last_send_delivered = 0;
    main_co->last_recv_delivered = 0;
    main_co->last_park_result = 0;
    
    /* Set as current */
    current_kcoro = main_co;
    main_kcoro = main_co;

    return main_co;
}

void kcoro_set_thread_main(kcoro_t* main_co)
{
    main_kcoro = main_co;
    current_kcoro = main_co;
}

kcoro_t* kcoro_create_cps(kcoro_step_fn_t initial_step, void* user_data)
{
    if (!initial_step) return NULL;
    
    kcoro_t* co = (kcoro_t*)calloc(1, sizeof(kcoro_t));
    if (!co) return NULL;
    
    /* Initialize stackless coroutine */
    co->next_step = initial_step;
    co->state = KCORO_CREATED;
    co->user_data = user_data;
    co->id = __atomic_fetch_add(&next_kcoro_id, 1, __ATOMIC_RELAXED);
    co->main_co = main_kcoro;
    co->scheduler = NULL;
    co->name = NULL;
    co->ready_enqueued = false;
    atomic_init(&co->running_flag, 0);
    atomic_init(&co->refcount, 1);
    co->last_send_delivered = 0;
    co->last_recv_delivered = 0;
    co->last_park_result = 0;
    co->next = NULL;
    co->prev = NULL;

    return co;
}

static void kcoro_free(kcoro_t* co)
{
    if (!co) return;

    /* Stackless: no stack to free, just the coroutine object itself */
    if (current_kcoro == co) {
        current_kcoro = NULL;
    }
    if (main_kcoro == co) {
        main_kcoro = NULL;
    }
    free(co);
}

void kcoro_destroy(kcoro_t* co)
{
    kcoro_release(co);
}

void kcoro_set_name(kcoro_t* co, const char* name)
{
    if (co) {
        co->name = name;
    }
}

kcoro_t* kcoro_current(void)
{
    return current_kcoro;
}

kcoro_t* kcoro_thread_main(void)
{
    return main_kcoro;
}

void kcoro_retain(kcoro_t* co)
{
    if (!co) return;
    atomic_fetch_add_explicit(&co->refcount, 1, memory_order_relaxed);
}

void kcoro_release(kcoro_t* co)
{
    if (!co) return;
    int prev = atomic_fetch_sub_explicit(&co->refcount, 1, memory_order_acq_rel);
    if (prev == 1) {
        kcoro_free(co);
    }
}

void kcoro_resume(kcoro_t* co)
{
    if (!co || co->state == KCORO_FINISHED) return;
    if (!co->next_step) return;
    
    /* Stackless: just run one step of the continuation */
    kcoro_t* prev_current = current_kcoro;
    
    co->state = KCORO_RUNNING;
    current_kcoro = co;
    
    /* Execute the coroutine's next step function */
    co->next_step = co->next_step(co);
    
    /* If next_step is NULL, the coroutine has finished */
    if (!co->next_step) {
        co->state = KCORO_FINISHED;
    } else {
        co->state = KCORO_SUSPENDED;
    }
    
    current_kcoro = prev_current;
}

void kcoro_yield(void)
{
    /* Stackless: yield is just a state change. The actual control return
     * happens when the current CPS step function returns NULL.
     * Scheduler will call kcoro_resume to invoke the next_step later. */
    kcoro_t* current = current_kcoro;
    if (!current) return;
    
    current->state = KCORO_SUSPENDED;
    /* No stack switch needed in stackless model.
     * When this function returns to the CPS step function, that function
     * will return NULL, giving control back to the scheduler. */
}

void kcoro_yield_to(kcoro_t* target_co)
{
    /* Stackless: just update current coroutine pointer and states.
     * No actual stack switching. Control flow is managed by scheduler. */
    if (!target_co) return;
    
    kcoro_t* current = current_kcoro;
    
    /* Update states */
    if (current) {
        current->state = KCORO_SUSPENDED;
    }
    target_co->state = KCORO_RUNNING;
    current_kcoro = target_co;
    
    /* Stackless: no context switch needed. When the current CPS step
     * returns, scheduler will see the new current_kcoro and continue there. */
}

/* Park current coroutine: transitions to KCORO_PARKED and suspends execution */
void kcoro_park(void)
{
    /* Stackless park: mark as parked and return control to scheduler.
     * Scheduler will skip this coroutine until something explicitly unparks it. */
    kcoro_t* current = current_kcoro;
    if (!current) return;
    if (current->state == KCORO_FINISHED) return;
    
    current->state = KCORO_PARKED;
    /* No stack switch. When CPS step returns NULL, scheduler sees PARKED state. */
}

void kcoro_unpark(kcoro_t* co)
{
    if (!co) return;
    if (co->state != KCORO_PARKED) return; /* Only parked coroutines */
    co->state = KCORO_READY;
    kc_sched_t* s = kc_sched_current();
    /* Do NOT auto-create a default scheduler here; tests that drive coroutines
     * manually (single-threaded) expect unparks to enqueue only when running
     * under an explicit worker scheduler. Auto-initializing would introduce a
     * background thread concurrently resuming coroutines, leading to races and
     * duplicate consumption. */
    if (s) {
        /* Unified scheduler path: legacy and v2 merged, so enqueue is always work-stealing aware now. */
        kc_sched_enqueue_ready(s, co);
    }
}

int kcoro_is_parked(const kcoro_t* co)
{
    return co && co->state == KCORO_PARKED;
}
