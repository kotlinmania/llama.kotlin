// SPDX-License-Identifier: BSD-3-Clause
/* kc_select_internal.h - Stub for legacy select infrastructure
 *
 * This is a minimal stub to allow kc_chan.c to compile during the
 * stackless cutover. The select functionality is deprecated and these
 * functions should not be called in stackless-only code.
 */
#ifndef KC_SELECT_INTERNAL_H
#define KC_SELECT_INTERNAL_H

#include "../../include/kcoro.h"

/* Forward declaration */
typedef struct kc_select kc_select_t;

/* Stub structure (not actually used in stackless mode) */
struct kc_select {
    int dummy;
};

/* Stub functions - these should never be called in stackless mode */
static inline int kc_select_try_complete(kc_select_t *sel, int clause, int result) {
    (void)sel; (void)clause; (void)result;
    return 0;  /* Always fail */
}

static inline kcoro_t* kc_select_waiter(kc_select_t *sel) {
    (void)sel;
    return NULL;
}

static inline void* kc_select_recv_buffer(kc_select_t *sel, int clause_index) {
    (void)sel; (void)clause_index;
    return NULL;
}

static inline const void* kc_select_send_buffer(kc_select_t *sel, int clause_index) {
    (void)sel; (void)clause_index;
    return NULL;
}

#endif /* KC_SELECT_INTERNAL_H */
