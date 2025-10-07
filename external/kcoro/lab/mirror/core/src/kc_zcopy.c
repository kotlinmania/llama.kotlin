// SPDX-License-Identifier: BSD-3-Clause
#include <errno.h>
#include <string.h>
#include "../../include/kcoro_zcopy.h"
#include "kc_chan_internal.h"

/* Minimal stub implementation while the arena-backed descriptors are under
 * construction. TODO(arena): replace these stubs once kc_rv_arena issues real
 * tickets (see ARENA_ARCH_PLAN.md). */

kc_zcopy_backend_id kc_zcopy_register(const char *name,
                                      const kc_zcopy_backend_ops_t *ops,
                                      uint32_t caps)
{
    (void)name; (void)ops; (void)caps;
    return -ENOTSUP;
}

kc_zcopy_backend_id kc_zcopy_resolve(const char *name)
{
    (void)name;
    return -ENOTSUP;
}

int kc_chan_enable_zero_copy_backend(kc_chan_t *ch,
                                     kc_zcopy_backend_id id,
                                     const void *opts)
{
    (void)ch; (void)id; (void)opts;
    return -ENOTSUP;
}

int kc_chan_send_desc(kc_chan_t *ch, const kc_zdesc_t *d, long timeout_ms)
{
    (void)ch; (void)d; (void)timeout_ms;
    return -ENOTSUP;
}

int kc_chan_recv_desc(kc_chan_t *ch, kc_zdesc_t *d, long timeout_ms)
{
    (void)ch; (void)d; (void)timeout_ms;
    return -ENOTSUP;
}

int kc_chan_send_desc_c(kc_chan_t *ch, const kc_zdesc_t *d, long timeout_ms, const kc_cancel_t *ct)
{
    if (ct && kc_cancel_is_set(ct)) return KC_ECANCELED;
    return kc_chan_send_desc(ch, d, timeout_ms);
}

int kc_chan_recv_desc_c(kc_chan_t *ch, kc_zdesc_t *d, long timeout_ms, const kc_cancel_t *ct)
{
    if (ct && kc_cancel_is_set(ct)) return KC_ECANCELED;
    return kc_chan_recv_desc(ch, d, timeout_ms);
}
