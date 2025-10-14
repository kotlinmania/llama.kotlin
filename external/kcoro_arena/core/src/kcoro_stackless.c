// SPDX-License-Identifier: BSD-3-Clause
/* kcoro_stackless.c - Stackless coroutine implementation
 *
 * This implements the stackless coroutine primitives for kcoro_arena.
 * All coroutines execute on the scheduler's stack with explicit state machines.
 */
#include "kcoro_stackless.h"
#include "kcoro_token_kernel.h"
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
    
    /* Free any pending arena ticket */
    if (k->arena_ticket) {
        free(k->arena_ticket);
        k->arena_ticket = NULL;
    }
    
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

static void koro_clear_ticket(koro_cont_t *k)
{
    if (k && k->arena_ticket) {
        free(k->arena_ticket);
        k->arena_ticket = NULL;
    }
}

static void koro_send_resume_callback(void *user_data, const kc_payload *payload)
{
    koro_cont_t *k = (koro_cont_t*)user_data;
    if (!k) return;

    if (payload) {
        k->arena_payload = payload->ptr;
        k->arena_payload_len = payload->len;
        k->arena_desc_id = payload->desc_id;
        k->last_park_result = payload->status;
    } else {
        k->arena_payload = NULL;
        k->arena_payload_len = 0;
        k->arena_desc_id = 0;
        k->last_park_result = KC_EPIPE;
    }

    koro_clear_ticket(k);
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
    
    kc_ticket ticket = kc_token_kernel_publish_send(
        ch,
        data,
        len,
        koro_send_resume_callback,
        k
    );
    
    /* Store ticket for cancellation support */
    k->arena_ticket = malloc(sizeof(kc_ticket));
    if (k->arena_ticket) {
        memcpy(k->arena_ticket, &ticket, sizeof(kc_ticket));
    }

    if (ticket.id == 0) {
        koro_clear_ticket(k);
        k->last_park_result = KC_EAGAIN;
        return (void*)1;
    }

    k->last_park_result = KC_EAGAIN;
    return NULL; /* Suspended */
}

static void koro_recv_resume_callback(void *user_data, const kc_payload *payload)
{
    koro_cont_t *k = (koro_cont_t*)user_data;
    if (!k) return;

    if (payload) {
        k->arena_payload = payload->ptr;
        k->arena_payload_len = payload->len;
        k->arena_desc_id = payload->desc_id;
        k->last_park_result = payload->status;
    } else {
        k->arena_payload = NULL;
        k->arena_payload_len = 0;
        k->arena_desc_id = 0;
        k->last_park_result = KC_EPIPE;
    }

    koro_clear_ticket(k);
    koro_sched_enqueue_ready(k);
}

/* Arena integration: Stackless receive operation. */
void* koro_recv_stackless(koro_cont_t* k, struct kc_chan* ch)
{
    if (!k || !ch) {
        k->last_park_result = -EINVAL;
        return (void*)1; /* Complete with error */
    }
    
    kc_ticket ticket = kc_token_kernel_publish_recv(
        ch,
        koro_recv_resume_callback,
        k
    );
    
    k->arena_ticket = malloc(sizeof(kc_ticket));
    if (k->arena_ticket) {
        memcpy(k->arena_ticket, &ticket, sizeof(kc_ticket));
    }

    if (ticket.id == 0) {
        koro_clear_ticket(k);
        k->last_park_result = KC_EAGAIN;
        return (void*)1;
    }

    k->last_park_result = KC_EAGAIN;
    return NULL; /* Suspended */
}
