// SPDX-License-Identifier: BSD-3-Clause
/* kcoro_stackless.c - Stackless coroutine implementation
 *
 * This implements the stackless coroutine primitives for kcoro_arena.
 * All coroutines execute on the scheduler's stack with explicit state machines.
 */
#include "kcoro_stackless.h"
#include "kcoro_token_kernel.h"
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
    k->last_park_result = 0;
    k->arena_payload = NULL;
    k->arena_payload_len = 0;
    k->arena_desc_id = 0;
    k->arena_ticket.id = 0;
    k->arena_ticket.channel = NULL;
    
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
    
    /* Free the continuation itself */
    free(k);
}

/* Forward declaration of scheduler enqueue */
extern void koro_sched_enqueue_ready(koro_cont_t* k);

/* Callback invoked by token kernel when send completes.
 * This is the "rehydration" event—it makes the coroutine runnable again. */
static void koro_send_resume_callback(void* user_data)
{
    koro_cont_t* k = (koro_cont_t*)user_data;
    if (!k) return;
    
    /* Consume the payload result from token kernel */
    kc_payload result;
    int rc = kc_token_kernel_consume_payload(&result);
    k->last_park_result = rc;
    
    /* Re-enqueue continuation in scheduler's ready queue */
    koro_sched_enqueue_ready(k);
}

/* Arena integration: Stackless send operation.
 * This is a CPS transformation of the blocking arena send.
 * Returns NULL if suspended, non-NULL if complete. */
void* koro_send_stackless(koro_cont_t* k, struct kc_chan* ch, void* data, size_t len)
{
    if (!k || !ch) {
        k->last_park_result = -EINVAL;
        return (void*)1; /* Complete with error */
    }
    
    /* Publish send token with our continuation as resume callback.
     * The token kernel will call koro_send_resume_callback when matched. */
    kc_ticket ticket = kc_token_kernel_publish_send(
        ch, 
        data, 
        len, 
        (void(*)(void))koro_send_resume_callback
    );
    
    /* Store ticket in continuation for potential cancellation */
    k->arena_ticket = ticket;
    
    /* Check if send completed immediately (fast path) */
    kc_payload immediate_result;
    int rc = kc_token_kernel_consume_payload(&immediate_result);
    if (rc == 0) {
        /* Immediate success—no suspension needed */
        k->last_park_result = 0;
        return (void*)1; /* Complete */
    }
    
    /* Send is pending; suspend by returning NULL.
     * The token kernel will invoke koro_send_resume_callback
     * when a receiver matches, which will re-enqueue this continuation. */
    return NULL; /* Suspended */
}

/* Callback invoked by token kernel when receive completes. */
static void koro_recv_resume_callback(void* user_data)
{
    koro_cont_t* k = (koro_cont_t*)user_data;
    if (!k) return;
    
    /* Consume the received payload from token kernel */
    kc_payload result;
    int rc = kc_token_kernel_consume_payload(&result);
    k->last_park_result = rc;
    
    if (rc == 0) {
        /* Store received data in continuation for user access */
        k->arena_payload = result.ptr;
        k->arena_payload_len = result.len;
        k->arena_desc_id = result.desc_id;
    }
    
    /* Re-enqueue continuation in scheduler's ready queue */
    koro_sched_enqueue_ready(k);
}

/* Arena integration: Stackless receive operation. */
void* koro_recv_stackless(koro_cont_t* k, struct kc_chan* ch)
{
    if (!k || !ch) {
        k->last_park_result = -EINVAL;
        return (void*)1; /* Complete with error */
    }
    
    /* Publish receive token with our continuation as resume callback */
    kc_ticket ticket = kc_token_kernel_publish_recv(
        ch,
        (void(*)(void))koro_recv_resume_callback
    );
    
    /* Store ticket in continuation for potential cancellation */
    k->arena_ticket = ticket;
    
    /* Check if data is immediately available (fast path) */
    kc_payload immediate_result;
    int rc = kc_token_kernel_consume_payload(&immediate_result);
    if (rc == 0) {
        /* Immediate success */
        k->arena_payload = immediate_result.ptr;
        k->arena_payload_len = immediate_result.len;
        k->arena_desc_id = immediate_result.desc_id;
        k->last_park_result = 0;
        return (void*)1; /* Complete */
    }
    
    /* Receive is pending; suspend.
     * Token kernel will invoke koro_recv_resume_callback when sender matches. */
    return NULL; /* Suspended */
}
