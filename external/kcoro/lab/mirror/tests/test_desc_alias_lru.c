// SPDX-License-Identifier: BSD-3-Clause
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/kcoro.h"
#include "../core/src/kc_chan_internal.h" /* internal for inspection */

int main(void) {
    /* Enable alias LRU */
    setenv("KC_DESC_ALIAS_LRU", "1", 1);

    kc_chan_t *ch = NULL;
    int rc = kc_chan_make_ptr(&ch, KC_RENDEZVOUS, 0);
    assert(rc == 0 && ch);
    struct kc_chan *internal = (struct kc_chan*)ch;
    assert(internal->ptr_mode == 1);

    /* Hot pointer set */
    static char a[] = "alpha";
    static char b[] = "bravo";

    /* Send/recv a few times to populate cache */
    for (int i = 0; i < 1000; ++i) {
        void *p = (i % 2 == 0) ? (void*)a : (void*)b;
        size_t l = strlen((const char*)p) + 1;
        rc = kc_chan_send_ptr(ch, p, l, -1);
        assert(rc == 0);
        void *outp = NULL; size_t outl = 0;
        rc = kc_chan_recv_ptr(ch, &outp, &outl, -1);
        assert(rc == 0);
        assert(outp == p && outl == l);
    }

    /* After warmup, LRU should show hits > 0 and no excessive evictions for 2 hot keys */
    printf("[alias-lru] hits=%lu misses=%lu evicts=%lu\n",
           internal->alias_lru_hits, internal->alias_lru_misses, internal->alias_lru_evicts);
    assert(internal->alias_lru_enabled);
    assert(internal->alias_lru_hits > 0);
    assert(internal->alias_lru_evicts == 0);

    kc_chan_destroy(ch);
    return 0;
}

