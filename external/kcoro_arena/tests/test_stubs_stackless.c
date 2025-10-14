// SPDX-License-Identifier: BSD-3-Clause
/* test_stubs_stackless.c - Minimal stubs for testing stackless primitives
 *
 * These are mock implementations of the token kernel and scheduler
 * APIs that would normally be provided by the full kcoro_arena system.
 */

#include <stddef.h>
#include <stdio.h>
#include "kcoro_token_kernel.h"
#include "kcoro_stackless.h"

/* Stub: publish send token to arena */
kc_ticket kc_token_kernel_publish_send(
    struct kc_chan* ch,
    void* ptr,
    size_t len,
    kc_token_resume_fn resume_cb,
    void *user_ctx
) {
    (void)ptr; (void)len; (void)resume_cb; (void)user_ctx;
    kc_ticket t = { .id = 1, .channel = ch };
    printf("[stub] publish_send ticket=%lu\n", (unsigned long)t.id);
    return t;
}

/* Stub: publish receive token to arena */
kc_ticket kc_token_kernel_publish_recv(
    struct kc_chan* ch,
    kc_token_resume_fn resume_cb,
    void *user_ctx
) {
    (void)resume_cb; (void)user_ctx;
    kc_ticket t = { .id = 2, .channel = ch };
    printf("[stub] publish_recv ticket=%lu\n", (unsigned long)t.id);
    return t;
}

/* Stub: enqueue a ready continuation */
void koro_sched_enqueue_ready(struct koro_cont* k) {
    printf("[stub] sched_enqueue k=%p state=%d\n", (void*)k, k->state);
}

int kc_token_kernel_subscribe(kc_token_event_type event,
                              kc_token_event_cb cb,
                              void *user_ctx) {
    (void)event; (void)cb; (void)user_ctx; return 0;
}

int kc_token_kernel_notify_event(kc_token_event_type event,
                                 struct kc_chan *channel,
                                 const kc_payload *payload) {
    (void)event; (void)channel; (void)payload; return 0;
}
