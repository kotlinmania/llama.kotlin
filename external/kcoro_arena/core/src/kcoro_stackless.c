// SPDX-License-Identifier: BSD-3-Clause
/* kcoro_stackless.c - Stackless coroutine implementation
 *
 * This implements the stackless coroutine primitives for kcoro_arena.
 * All coroutines execute on the scheduler's stack with explicit state machines.
 */
#include "kcoro_stackless.h"
#include "kcoro_token_kernel.h"
#include "kc_chan_api.h"
#include "kcoro_port.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

/* Global coroutine ID counter */
static uint64_t next_koro_id = 1;

/* Create a new stackless coroutine continuation. */
koro_cont_t* koro_cont_create(koro_step_fn initial_step, 
                               void* user_arg,
                               size_t user_data_size)
{
    if (!initial_step) return NULL;
    
    koro_cont_t* k = (koro_cont_t*)calloc(1, sizeof(koro_cont_t));
    if (!k) return NULL;
    
    /* Allocate user data space if requested */
    if (user_data_size > 0) {
        k->user_data = calloc(1, user_data_size);
        if (!k->user_data) {
            free(k);
            return NULL;
        }
    }
    
    /* Initialize continuation */
    k->state = 0;           /* Start at beginning */
    k->next_step = initial_step;
    k->user_arg = user_arg;
    k->next = NULL;
    k->ready_enqueued = 0;
    k->id = __atomic_fetch_add(&next_koro_id, 1, __ATOMIC_SEQ_CST);
    k->name = "stackless";
    k->completed = 0;
    k->managed = 0;
    k->tracked = 0;
    k->last_park_result = 0;
    k->arena_payload = NULL;
    k->arena_payload_len = 0;
    k->arena_desc_id = 0;
    k->arena_ticket = NULL; /* No pending ticket yet */
    
    return k;
}

/* Destroy a continuation and free its resources. */
void koro_cont_destroy(koro_cont_t* k)
{
    if (!k) return;
    
    /* Free user data if allocated */
    if (k->user_data) {
        free(k->user_data);
        k->user_data = NULL;
    }

    k->managed = 0;
    k->tracked = 0;
    
    /* Free the continuation itself */
    free(k);
}

/* Forward declaration of scheduler enqueue (implemented in scheduler) */
extern void koro_sched_enqueue_ready(koro_cont_t* k);

/* Arena integration: Stackless send operation.
 * This is a CPS transformation of the blocking arena send.
 * Returns NULL if suspended, non-NULL if complete. */
void* koro_send_stackless(koro_cont_t* k, struct kc_chan* ch, void* data, size_t len)
{
    if (!k || !ch) {
        if (k) k->last_park_result = -EINVAL;
        return (void*)1; /* Complete with error */
    }
    
    /* Use new stackless channel API */
    int result = kc_chan_send_stackless(k, ch, data, len);
    
    if (result == 0) {
        /* Completed immediately */
        return (void*)1;
    } else if (result == 1) {
        /* Suspended - will be resumed by channel callback */
        return NULL;
    } else {
        /* Error */
        k->last_park_result = result;
        return (void*)1;
    }
}

/* Arena integration: Stackless receive operation. */
void* koro_recv_stackless(koro_cont_t* k, struct kc_chan* ch)
{
    if (!k || !ch) {
        if (k) k->last_park_result = -EINVAL;
        return (void*)1; /* Complete with error */
    }
    
    /* Use new stackless channel API */
    int result = kc_chan_recv_stackless(k, ch);
    
    if (result == 0) {
        /* Completed immediately - data in k->arena_payload */
        return (void*)1;
    } else if (result == 1) {
        /* Suspended - will be resumed by channel callback */
        return NULL;
    } else {
        /* Error */
        k->last_park_result = result;
        return (void*)1;
    }
}
