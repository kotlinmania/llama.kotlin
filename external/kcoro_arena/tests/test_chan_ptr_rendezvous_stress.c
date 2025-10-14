// SPDX-License-Identifier: BSD-3-Clause
#define _GNU_SOURCE

#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kcoro.h"
#include "kcoro_sched.h"
#include "kcoro_token_kernel.h"
#include "kcoro_zcopy.h"

#define NUM_PRODUCERS 4
#define NUM_CONSUMERS 4
#define MSGS_PER_PRODUCER 200
#define TOTAL_MSGS (NUM_PRODUCERS * MSGS_PER_PRODUCER)

struct rendezvous_msg {
    int producer_id;
    int seq;
};

typedef struct {
    kc_chan_t *chan;
    int id;
    struct rendezvous_msg msgs[MSGS_PER_PRODUCER];
    atomic_int sent;
} producer_ctx_t;

typedef struct {
    kc_chan_t *chan;
    atomic_int received;
} consumer_ctx_t;

static atomic_int g_total_sent = 0;
static atomic_int g_total_received = 0;
static atomic_int g_duplicates = 0;
static atomic_int g_cancel_events = 0;
static atomic_int g_sender_match_events = 0;
static atomic_int g_receiver_match_events = 0;
static atomic_int g_sender_ready_events = 0;
static atomic_int g_receiver_ready_events = 0;

static atomic_int g_seen[NUM_PRODUCERS][MSGS_PER_PRODUCER];

static void token_event_logger(struct kc_chan *channel,
                               const kc_payload *payload,
                               void *user_ctx)
{
    (void)channel;
    (void)payload;
    uintptr_t tag = (uintptr_t)user_ctx;
    switch (tag) {
    case KC_TOKEN_EVENT_EMPTY_TO_SENDER_READY:
        atomic_fetch_add_explicit(&g_sender_ready_events, 1, memory_order_relaxed);
        break;
    case KC_TOKEN_EVENT_EMPTY_TO_RECEIVER_READY:
        atomic_fetch_add_explicit(&g_receiver_ready_events, 1, memory_order_relaxed);
        break;
    case KC_TOKEN_EVENT_SENDER_TO_MATCHED:
        atomic_fetch_add_explicit(&g_sender_match_events, 1, memory_order_relaxed);
        break;
    case KC_TOKEN_EVENT_RECEIVER_TO_MATCHED:
        atomic_fetch_add_explicit(&g_receiver_match_events, 1, memory_order_relaxed);
        break;
    case KC_TOKEN_EVENT_ANY_TO_CANCELLED:
        atomic_fetch_add_explicit(&g_cancel_events, 1, memory_order_relaxed);
        break;
    default:
        break;
    }
}

static void producer_fn(void *arg)
{
    producer_ctx_t *ctx = (producer_ctx_t*)arg;

    for (int i = 0; i < MSGS_PER_PRODUCER; ++i) {
        struct rendezvous_msg *msg = &ctx->msgs[i];
        msg->producer_id = ctx->id;
        msg->seq = i;

        int rc = kc_chan_send_ptr(ctx->chan, msg, sizeof(*msg), -1);
        assert(rc == 0);

        atomic_fetch_add_explicit(&ctx->sent, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_total_sent, 1, memory_order_relaxed);
    }
}

static void consumer_fn(void *arg)
{
    consumer_ctx_t *ctx = (consumer_ctx_t*)arg;

    const int expected = MSGS_PER_PRODUCER;
    for (int count = 0; count < expected; ++count) {
        void *ptr = NULL;
        size_t len = 0;
        int rc = kc_chan_recv_ptr(ctx->chan, &ptr, &len, -1);
        assert(rc == 0);
        assert(len == sizeof(struct rendezvous_msg));

        struct rendezvous_msg *msg = (struct rendezvous_msg*)ptr;
        assert(msg->producer_id >= 0 && msg->producer_id < NUM_PRODUCERS);
        assert(msg->seq >= 0 && msg->seq < MSGS_PER_PRODUCER);

        atomic_int *slot = &g_seen[msg->producer_id][msg->seq];
        int prev = atomic_fetch_add_explicit(slot, 1, memory_order_relaxed);
        if (prev != 0) {
            atomic_fetch_add_explicit(&g_duplicates, 1, memory_order_relaxed);
        }

        atomic_fetch_add_explicit(&ctx->received, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_total_received, 1, memory_order_relaxed);
    }
}

int main(void)
{
    printf("=== kcoro_arena Rendezvous Stress Test ===\n");

    int rc = kc_token_kernel_global_init();
    if (rc != 0) {
        fprintf(stderr, "token kernel init failed: %d\n", rc);
        return 1;
    }

    /* subscribe to token events for diagnostics */
    for (int evt = 0; evt < KC_TOKEN_EVENT_COUNT; ++evt) {
        kc_token_kernel_subscribe((kc_token_event_type)evt,
                                  token_event_logger,
                                  (void*)(uintptr_t)evt);
    }

    kc_sched_opts_t sched_opts = { .workers = NUM_PRODUCERS + NUM_CONSUMERS };
    kc_sched_t *sched = kc_sched_init(&sched_opts);
    if (!sched) {
        fprintf(stderr, "scheduler init failed\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }

    kc_chan_t *chan = NULL;
    rc = kc_chan_make_ptr(&chan, KC_RENDEZVOUS, 0);
    if (rc != 0) {
        fprintf(stderr, "kc_chan_make_ptr failed: %d\n", rc);
        kc_sched_shutdown(sched);
        kc_token_kernel_global_shutdown();
        return 1;
    }

    rc = kc_chan_enable_zero_copy(chan);
    assert(rc == 0);

    producer_ctx_t producers[NUM_PRODUCERS];
    consumer_ctx_t consumers[NUM_CONSUMERS];

    memset(producers, 0, sizeof(producers));
    memset(consumers, 0, sizeof(consumers));
    memset(g_seen, 0, sizeof(g_seen));

    for (int i = 0; i < NUM_PRODUCERS; ++i) {
        producers[i].chan = chan;
        producers[i].id = i;
        atomic_init(&producers[i].sent, 0);
        kc_spawn_co(sched, producer_fn, &producers[i], 0, NULL);
    }

    for (int i = 0; i < NUM_CONSUMERS; ++i) {
        consumers[i].chan = chan;
        atomic_init(&consumers[i].received, 0);
        kc_spawn_co(sched, consumer_fn, &consumers[i], 0, NULL);
    }

    rc = kc_sched_drain(sched, 10000);
    if (rc != 0) {
        fprintf(stderr, "kc_sched_drain timed out: %d\n", rc);
        kc_chan_close(chan);
        kc_sched_shutdown(sched);
        kc_token_kernel_global_shutdown();
        return 1;
    }

    printf("Total sent=%d received=%d duplicates=%d\n",
           atomic_load(&g_total_sent),
           atomic_load(&g_total_received),
           atomic_load(&g_duplicates));

    for (int p = 0; p < NUM_PRODUCERS; ++p) {
        if (atomic_load(&producers[p].sent) != MSGS_PER_PRODUCER) {
            fprintf(stderr, "producer %d sent only %d messages\n",
                    p, atomic_load(&producers[p].sent));
            return 1;
        }
    }

    for (int c = 0; c < NUM_CONSUMERS; ++c) {
        if (atomic_load(&consumers[c].received) != MSGS_PER_PRODUCER) {
            fprintf(stderr, "consumer %d received only %d messages\n",
                    c, atomic_load(&consumers[c].received));
            return 1;
        }
    }

    if (atomic_load(&g_total_sent) != TOTAL_MSGS ||
        atomic_load(&g_total_received) != TOTAL_MSGS) {
        fprintf(stderr, "mismatch totals: sent=%d recv=%d\n",
                atomic_load(&g_total_sent),
                atomic_load(&g_total_received));
        return 1;
    }

    if (atomic_load(&g_duplicates) != 0) {
        fprintf(stderr, "duplicate messages detected: %d\n",
                atomic_load(&g_duplicates));
        return 1;
    }

    kc_chan_snapshot chan_snap = {0};
    rc = kc_chan_snapshot(chan, &chan_snap);
    if (rc == 0) {
        printf("Channel stats: sends=%lu recvs=%lu rv_matches=%lu cancel=%lu\n",
               chan_snap.total_sends,
               chan_snap.total_recvs,
               chan_snap.rv_matches,
               chan_snap.rv_cancels);
    }

    printf("Token events: sender_ready=%d receiver_ready=%d sender_match=%d receiver_match=%d cancelled=%d\n",
           atomic_load(&g_sender_ready_events),
           atomic_load(&g_receiver_ready_events),
           atomic_load(&g_sender_match_events),
           atomic_load(&g_receiver_match_events),
           atomic_load(&g_cancel_events));

    kc_chan_close(chan);
    kc_chan_destroy(chan);
    kc_sched_shutdown(sched);
    kc_token_kernel_global_shutdown();

    printf("=== PASS kcoro_arena Rendezvous Stress Test ===\n");
    return 0;
}
