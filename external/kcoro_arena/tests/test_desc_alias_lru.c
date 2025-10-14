// SPDX-License-Identifier: BSD-3-Clause
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/kcoro.h"
#include "../include/kcoro_sched.h"
#include "../core/src/kc_chan_internal.h" /* internal for inspection */

static kc_chan_t *g_ch = NULL;

static void producer(void *arg) {
    (void)arg;
    /* Hot pointer set */
    static char a[] = "alpha";
    static char b[] = "bravo";
    
    for (int i = 0; i < 1000; ++i) {
        void *p = (i % 2 == 0) ? (void*)a : (void*)b;
        size_t l = strlen((const char*)p) + 1;
        int rc = kc_chan_send_ptr(g_ch, p, l, -1);
        assert(rc == 0);
    }
}

static void consumer(void *arg) {
    (void)arg;
    static char a[] = "alpha";
    static char b[] = "bravo";
    
    for (int i = 0; i < 1000; ++i) {
        void *expected = (i % 2 == 0) ? (void*)a : (void*)b;
        size_t expected_len = strlen((const char*)expected) + 1;
        void *outp = NULL; 
        size_t outl = 0;
        int rc = kc_chan_recv_ptr(g_ch, &outp, &outl, -1);
        assert(rc == 0);
        /* Data should match, but pointer may be alias */
        assert(outl == expected_len);
        assert(memcmp(outp, expected, expected_len) == 0);
    }
}

int main(void) {
    kc_sched_t *sched = kc_sched_init(NULL);
    assert(sched);

    int rc = kc_chan_make_ptr(&g_ch, KC_RENDEZVOUS, 0);
    assert(rc == 0 && g_ch);
    struct kc_chan *internal = (struct kc_chan*)g_ch;
    assert(internal->ptr_mode == 1);

    /* Spawn producer and consumer */
    rc = kc_spawn_co(sched, producer, NULL, 0, NULL);
    assert(rc == 0);
    rc = kc_spawn_co(sched, consumer, NULL, 0, NULL);
    assert(rc == 0);

    /* Drain until done */
    rc = kc_sched_drain(sched, -1);
    assert(rc == 0);

    /* After warmup, LRU should show hits > 0 and no excessive evictions for 2 hot keys */
    printf("[alias-lru] hits=%lu misses=%lu evicts=%lu\n",
           internal->alias_lru_hits, internal->alias_lru_misses, internal->alias_lru_evicts);
    assert(internal->alias_lru_enabled);
    assert(internal->alias_lru_hits > 0);
    assert(internal->alias_lru_evicts == 0);

    kc_chan_destroy(g_ch);
    kc_sched_shutdown(sched);
    return 0;
}
