// SPDX-License-Identifier: BSD-3-Clause
// Validate rendezvous channel metrics under concurrent coroutine load.

#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/kcoro.h"
#include "../include/kcoro_core.h"
#include "../include/kcoro_sched.h"
#include "../core/src/kc_chan_internal.h"

#define PRODUCERS 4
#define CONSUMERS 4
#define PER_PRODUCER 200
#define STACK_SIZE (64 * 1024)

struct test_context {
    kc_chan_t *chan;
    atomic_int producers_done;
    unsigned long expected_msgs;
    atomic_ulong sends_completed;
    atomic_ulong recvs_completed;
};

static void producer_co(void *arg) {
    struct test_context *ctx = (struct test_context *)arg;
    for (int i = 0; i < PER_PRODUCER; ++i) {
        int payload = i;
        int rc = kc_chan_send(ctx->chan, &payload, -1);
        if (rc != 0) {
            fprintf(stderr, "[rv-metrics] producer send failed rc=%d at i=%d\n", rc, i);
            return;
        }
        atomic_fetch_add_explicit(&ctx->sends_completed, 1, memory_order_relaxed);
    }
    int finished = atomic_fetch_add_explicit(&ctx->producers_done, 1, memory_order_acq_rel) + 1;
    if (finished == PRODUCERS) {
        kc_chan_close(ctx->chan);
    }
}

static void consumer_co(void *arg) {
    struct test_context *ctx = (struct test_context *)arg;
    int value = 0;
    for (;;) {
        int rc = kc_chan_recv(ctx->chan, &value, -1);
        if (rc == -EPIPE) {
            break;
        }
        if (rc != 0) {
            fprintf(stderr, "[rv-metrics] consumer recv failed rc=%d\n", rc);
            break;
        }
        atomic_fetch_add_explicit(&ctx->recvs_completed, 1, memory_order_relaxed);
    }
    (void)ctx;
}

int main(void) {
    kcoro_t *main_co = kcoro_create_main();
    kcoro_set_thread_main(main_co);

    kc_sched_opts_t opts = {0};
    opts.workers = PRODUCERS + CONSUMERS;
    kc_sched_t *sched = kc_sched_init(&opts);
    assert(sched);

    kc_chan_t *chan = NULL;
    int rc = kc_chan_make(&chan, KC_RENDEZVOUS, sizeof(int), 0);
    assert(rc == 0 && chan);

    struct test_context ctx = {
        .chan = chan,
        .expected_msgs = (unsigned long)PRODUCERS * PER_PRODUCER,
    };
    atomic_init(&ctx.producers_done, 0);
    atomic_init(&ctx.sends_completed, 0);
    atomic_init(&ctx.recvs_completed, 0);

    for (int i = 0; i < PRODUCERS; ++i) {
        rc = kc_spawn_co(sched, producer_co, &ctx, STACK_SIZE, NULL);
        assert(rc == 0);
    }
    for (int i = 0; i < CONSUMERS; ++i) {
        rc = kc_spawn_co(sched, consumer_co, &ctx, STACK_SIZE, NULL);
        assert(rc == 0);
    }

    /* Wait for scheduler to drain (up to 10 seconds). */
    const long timeout_ms = 10000;
    rc = kc_sched_drain(sched, timeout_ms);
    if (rc != 0) {
        struct kc_chan_snapshot fail_snap = {0};
        kc_chan_snapshot(chan, &fail_snap);
        fprintf(stderr,
                "[rv-metrics] drain timeout (%d) sends=%lu (atomic=%lu) recvs=%lu (atomic=%lu) rv_matches=%lu rv_cancels=%lu waiters_send=%u waiters_recv=%u producers_done=%d\n",
                rc,
                fail_snap.total_sends,
                (unsigned long)atomic_load_explicit(&ctx.sends_completed, memory_order_relaxed),
                fail_snap.total_recvs,
                (unsigned long)atomic_load_explicit(&ctx.recvs_completed, memory_order_relaxed),
                fail_snap.rv_matches,
                fail_snap.rv_cancels,
                ((struct kc_chan *)chan)->waiters_send,
                ((struct kc_chan *)chan)->waiters_recv,
                atomic_load_explicit(&ctx.producers_done, memory_order_relaxed));
        kc_sched_shutdown(sched);
        kc_chan_destroy(chan);
        return 1;
    }

    struct kc_chan_snapshot snap = {0};
    rc = kc_chan_snapshot(chan, &snap);
    assert(rc == 0);

    if (snap.total_sends != ctx.expected_msgs || snap.total_recvs != ctx.expected_msgs) {
        fprintf(stderr,
                "[rv-metrics] mismatch sends=%lu recvs=%lu expected=%lu\n",
                snap.total_sends, snap.total_recvs, ctx.expected_msgs);
        kc_sched_shutdown(sched);
        kc_chan_destroy(chan);
        return 2;
    }

    struct kc_chan *internal = (struct kc_chan *)chan;
    assert(internal->waiters_send == 0);
    assert(internal->waiters_recv == 0);
    assert(internal->wq_send_head == NULL);
    assert(internal->wq_recv_head == NULL);
    assert(internal->rv_matches == ctx.expected_msgs);
    assert(internal->rv_cancels <= CONSUMERS);

    kc_sched_shutdown(sched);
    kc_chan_destroy(chan);
    return 0;
}
