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
    void (*resume_callback)(void)
) {
    (void)ptr; (void)len; (void)resume_callback;
    kc_ticket t = { .id = 1, .channel = ch };
    printf("[stub] publish_send ticket=%lu\n", (unsigned long)t.id);
    return t;
}

/* Stub: publish receive token to arena */
kc_ticket kc_token_kernel_publish_recv(
    struct kc_chan* ch,
    void (*resume_callback)(void)
) {
    (void)resume_callback;
    kc_ticket t = { .id = 2, .channel = ch };
    printf("[stub] publish_recv ticket=%lu\n", (unsigned long)t.id);
    return t;
}

/* Stub: try to consume payload from matched token */
int kc_token_kernel_consume_payload(kc_payload* out) {
    (void)out;
    /* Return -1 to indicate no payload yet (suspension required) */
    return -1;
}

/* Stub: enqueue a ready continuation */
void koro_sched_enqueue_ready(struct koro_cont* k) {
    printf("[stub] sched_enqueue k=%p state=%d\n", (void*)k, k->state);
}
