// SPDX-License-Identifier: BSD-3-Clause
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include "../include/kcoro.h"
#include "../include/kcoro_sched.h"
#include <unistd.h>
#include "../core/src/kc_chan_internal.h"

#define PRODUCERS 4
#define CONSUMERS 4
#define PER_PRODUCER 20000

typedef struct ctx {
    kc_chan_t *ch;
    const char **pool;
    int pool_size;
    atomic_ulong sends;
    atomic_ulong recvs;
} ctx_t;

static void producer(void *arg) {
    ctx_t *c = (ctx_t*)arg;
    for (int i = 0; i < PER_PRODUCER; ++i) {
        int k = i % c->pool_size;
        const char *p = c->pool[k];
        size_t l = strlen(p) + 1;
        int rc = kc_chan_send_ptr(c->ch, (void*)p, l, -1);
        if (rc != 0) { fprintf(stderr, "send rc=%d\n", rc); return; }
        atomic_fetch_add_explicit(&c->sends, 1, memory_order_relaxed);
    }
}

static void consumer(void *arg) {
    ctx_t *c = (ctx_t*)arg;
    for (;;) {
        void *p = NULL; size_t l = 0;
        int rc = kc_chan_recv_ptr(c->ch, &p, &l, -1);
        if (rc == -EPIPE) break;
        if (rc != 0) { fprintf(stderr, "recv rc=%d\n", rc); return; }
        if (!p || l == 0) { fprintf(stderr, "bad payload len=%zu\n", l); return; }
        atomic_fetch_add_explicit(&c->recvs, 1, memory_order_relaxed);
    }
}

int main(void) {
    setenv("KC_DESC_ALIAS_LRU", "1", 1);
    kc_sched_t *s = kc_sched_default();
    assert(s);
    kc_chan_t *ch = NULL;
    int rc = kc_chan_make_ptr(&ch, KC_RENDEZVOUS, 0);
    assert(rc == 0 && ch);
    struct kc_chan *internal = (struct kc_chan*)ch;

    static const char *pool[16] = {
        "p0","p1","p2","p3","p4","p5","p6","p7",
        "p8","p9","p10","p11","p12","p13","p14","p15"
    };
    ctx_t ctx = { .ch = ch, .pool = pool, .pool_size = 16 };
    atomic_init(&ctx.sends, 0);
    atomic_init(&ctx.recvs, 0);

    for (int i = 0; i < PRODUCERS; ++i) {
        rc = kc_spawn_co(s, producer, &ctx, 64*1024, NULL); assert(rc == 0);
    }
    for (int i = 0; i < CONSUMERS; ++i) {
        rc = kc_spawn_co(s, consumer, &ctx, 64*1024, NULL); assert(rc == 0);
    }

    /* Close channel after producers complete by scheduling a closer task */
    while (atomic_load_explicit(&ctx.sends, memory_order_relaxed) < (unsigned long)PRODUCERS*PER_PRODUCER) {
        usleep(1000);
    }
    kc_chan_close(ch);
    (void)kc_sched_drain(s, 10000);

    struct kc_chan_snapshot snap = {0};
    rc = kc_chan_snapshot(ch, &snap); assert(rc == 0);
    printf("[stress] sends=%lu recvs=%lu hits=%lu misses=%lu evicts=%lu\n",
           snap.total_sends, snap.total_recvs,
           internal->alias_lru_hits, internal->alias_lru_misses, internal->alias_lru_evicts);
    assert(snap.total_sends == (unsigned long)PRODUCERS*PER_PRODUCER);
    assert(snap.total_recvs == snap.total_sends);
    assert(internal->alias_lru_hits > 0);

    kc_chan_destroy(ch);
    return 0;
}
