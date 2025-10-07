// SPDX-License-Identifier: BSD-3-Clause
#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <time.h>

#include "../../include/kcoro.h"
#include "../../include/kcoro_core.h"
#include "../../include/kcoro_sched.h"
#include "../../include/kcoro_token_kernel.h"
#include "../../include/kcoro_config.h"
#include "../../include/kcoro_config_runtime.h"
#include "../../include/kcoro_zcopy.h"
#include "kc_chan_internal.h"
#include "kc_select_internal.h"

/* ------------------------------------------------------------------------- */
/* Metrics helpers (unchanged from legacy path). */

static void kc_chan_note_op_locked(struct kc_chan *ch, int is_send, size_t len)
{
    if (is_send) {
        ch->total_sends++;
        ch->total_bytes_sent += len;
    } else {
        ch->total_recvs++;
        ch->total_bytes_recv += len;
    }
    long now = kc_now_ns();
    if (ch->first_op_time_ns == 0) ch->first_op_time_ns = now;
    ch->last_op_time_ns = now;
}

unsigned kc_chan_len(kc_chan_t *c)
{
    if (!c) return 0;
    struct kc_chan *ch = (struct kc_chan*)c;
    KC_MUTEX_LOCK(&ch->mu);
    unsigned len = ch->count;
    KC_MUTEX_UNLOCK(&ch->mu);
    return len;
}

/* ------------------------------------------------------------------------- */
/* Pending queue helpers */

static void pending_send_enqueue(struct kc_chan *ch, struct kc_pending_send *node)
{
    node->next = NULL;
    if (ch->token_send_tail) {
        ch->token_send_tail->next = node;
    } else {
        ch->token_send_head = node;
    }
    ch->token_send_tail = node;
}

static struct kc_pending_send *pending_send_dequeue(struct kc_chan *ch)
{
    struct kc_pending_send *node = ch->token_send_head;
    if (!node) return NULL;
    ch->token_send_head = node->next;
    if (!ch->token_send_head) ch->token_send_tail = NULL;
    node->next = NULL;
    return node;
}

static void pending_recv_enqueue(struct kc_chan *ch, struct kc_pending_recv *node)
{
    node->next = NULL;
    if (ch->token_recv_tail) {
        ch->token_recv_tail->next = node;
    } else {
        ch->token_recv_head = node;
    }
    ch->token_recv_tail = node;
}

static struct kc_pending_recv *pending_recv_dequeue(struct kc_chan *ch)
{
    struct kc_pending_recv *node = ch->token_recv_head;
    if (!node) return NULL;
    ch->token_recv_head = node->next;
    if (!ch->token_recv_head) ch->token_recv_tail = NULL;
    node->next = NULL;
    return node;
}

static void pending_send_remove_select(struct kc_chan *ch, kc_select_t *sel, int clause)
{
    struct kc_pending_send **cur = &ch->token_send_head;
    struct kc_pending_send *tail = ch->token_send_tail;
    while (*cur) {
        struct kc_pending_send *node = *cur;
        if (node->role == KC_PENDING_ROLE_SELECT && node->sel == sel && node->clause_index == clause) {
            if (node->ticket.id) kc_token_kernel_cancel(node->ticket, KC_ECANCELED);
            if (node->desc_id) kc_desc_release(node->desc_id);
            *cur = node->next;
            if (tail == node) ch->token_send_tail = NULL;
            free(node);
            return;
        }
        cur = &node->next;
    }
}

static void pending_recv_remove_select(struct kc_chan *ch, kc_select_t *sel, int clause)
{
    struct kc_pending_recv **cur = &ch->token_recv_head;
    struct kc_pending_recv *tail = ch->token_recv_tail;
    while (*cur) {
        struct kc_pending_recv *node = *cur;
        if (node->role == KC_PENDING_ROLE_SELECT && node->sel == sel && node->clause_index == clause) {
            if (node->ticket.id) kc_token_kernel_cancel(node->ticket, KC_ECANCELED);
            if (node->desc_id) kc_desc_release(node->desc_id);
            *cur = node->next;
            if (tail == node) ch->token_recv_tail = NULL;
            free(node);
            return;
        }
        cur = &node->next;
    }
}

/* ------------------------------------------------------------------------- */
/* Wake helpers */

struct kc_wake {
    kcoro_t     *co;
    kc_select_t *sel;
};

static void kc_schedule_wake(struct kc_wake wake)
{
    if (!wake.co) return;
    kcoro_t *co = wake.co;
    if (kcoro_is_parked(co)) kcoro_unpark(co);
    kc_sched_t *sched = kc_sched_current();
    if (!sched) sched = kc_sched_default();
    if (sched) kc_sched_enqueue_ready(sched, co);
    kcoro_release(co);
}

static void complete_select(kc_select_t *sel, int clause, int result)
{
    if (!sel) return;
    if (kc_select_try_complete(sel, clause, result)) {
        kcoro_t *co = kc_select_waiter(sel);
        if (co) {
            kcoro_retain(co);
            kc_schedule_wake((struct kc_wake){ .co = co, .sel = sel });
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Channel lifecycle */

int kc_chan_make(kc_chan_t **out, int kind, size_t elem_sz, size_t capacity)
{
    if (!out || elem_sz == 0) return -EINVAL;
    struct kc_chan *ch = calloc(1, sizeof(*ch));
    if (!ch) return -ENOMEM;
    KC_MUTEX_INIT(&ch->mu);
    KC_COND_INIT(&ch->cv_send);
    KC_COND_INIT(&ch->cv_recv);
    ch->kind = kind;
    ch->elem_sz = elem_sz;
    if (kind == KC_RENDEZVOUS || kind == KC_CONFLATED) {
        ch->slot = malloc(elem_sz);
        if (!ch->slot) { free(ch); return -ENOMEM; }
    } else {
        ch->capacity = capacity ? capacity : 64;
        ch->mask = (ch->capacity & (ch->capacity - 1)) == 0 ? ch->capacity - 1 : 0;
        ch->buf = malloc(ch->capacity * elem_sz);
        if (!ch->buf) { free(ch); return -ENOMEM; }
    }
    ch->emit_check_mask = 0x3FFUL;
    *out = ch;
    kc_desc_global_init();
    return 0;
}

void kc_chan_destroy(kc_chan_t *c)
{
    if (!c) return;
    struct kc_chan *ch = (struct kc_chan*)c;
    kc_chan_close(c);
    free(ch->buf);
    free(ch->slot);
    KC_MUTEX_DESTROY(&ch->mu);
    KC_COND_DESTROY(&ch->cv_send);
    KC_COND_DESTROY(&ch->cv_recv);
    free(ch);
}

void kc_chan_close(kc_chan_t *c)
{
    struct kc_chan *ch = (struct kc_chan*)c;
    KC_MUTEX_LOCK(&ch->mu);
    if (ch->closed) {
        KC_MUTEX_UNLOCK(&ch->mu);
        return;
    }
    ch->closed = 1;
    struct kc_pending_send *ps = ch->token_send_head;
    struct kc_pending_recv *pr = ch->token_recv_head;
    ch->token_send_head = ch->token_send_tail = NULL;
    ch->token_recv_head = ch->token_recv_tail = NULL;
    KC_MUTEX_UNLOCK(&ch->mu);

    while (ps) {
        struct kc_pending_send *next = ps->next;
        if (ps->role == KC_PENDING_ROLE_CORO && ps->ticket.id) {
            kc_token_kernel_cancel(ps->ticket, KC_EPIPE);
        } else if (ps->role == KC_PENDING_ROLE_SELECT) {
            complete_select(ps->sel, ps->clause_index, KC_EPIPE);
        }
        if (ps->desc_id) kc_desc_release(ps->desc_id);
        free(ps);
        ps = next;
    }
    while (pr) {
        struct kc_pending_recv *next = pr->next;
        if (pr->role == KC_PENDING_ROLE_CORO && pr->ticket.id) {
            kc_token_kernel_cancel(pr->ticket, KC_EPIPE);
        } else if (pr->role == KC_PENDING_ROLE_SELECT) {
            complete_select(pr->sel, pr->clause_index, KC_EPIPE);
        }
        if (pr->desc_id) kc_desc_release(pr->desc_id);
        free(pr);
        pr = next;
    }
    if (ch->rv_slot_desc) {
        kc_desc_release(ch->rv_slot_desc);
        ch->rv_slot_desc = 0;
    }
}

/* ------------------------------------------------------------------------- */
/* Rendezvous pointer helpers */

static void fulfill_coroutine_send(struct kc_pending_send *node, kc_desc_id desc)
{
    kc_payload payload = {0};
    if (kc_desc_payload(desc, &payload) != 0) {
        payload.status = KC_EPIPE;
        payload.ptr = NULL;
        payload.len = 0;
        payload.desc_id = 0;
        kc_token_kernel_callback(node->ticket, payload);
        kc_desc_release(desc);
        free(node);
        return;
    }
    kc_desc_retain(desc);
    payload.desc_id = desc;
    kc_token_kernel_callback(node->ticket, payload);
    kc_desc_release(desc);
    free(node);
}

static void fulfill_coroutine_recv(struct kc_pending_recv *node, kc_desc_id desc)
{
    kc_payload payload = {0};
    if (kc_desc_payload(desc, &payload) != 0) {
        payload.status = KC_EPIPE;
        payload.ptr = NULL;
        payload.len = 0;
        payload.desc_id = 0;
        kc_token_kernel_callback(node->ticket, payload);
        kc_desc_release(desc);
        free(node);
        return;
    }
    kc_desc_retain(desc);
    payload.desc_id = desc;
    kc_token_kernel_callback(node->ticket, payload);
    kc_desc_release(desc);
    free(node);
}

static void fulfill_select_send(struct kc_chan *ch, struct kc_pending_send *node, kc_desc_id desc)
{
    (void)ch;
    kc_payload payload = {0};
    int rc = kc_desc_payload(desc, &payload);
    struct kc_chan_ptrmsg msg = { .ptr = payload.ptr, .len = payload.len };
    void *dst = kc_select_recv_buffer(node->sel, node->clause_index);
    if (rc == 0 && dst) memcpy(dst, &msg, sizeof(msg));
    complete_select(node->sel, node->clause_index, (rc == 0 && dst) ? 0 : KC_ECANCELED);
    kc_desc_release(desc);
    free(node);
}

static void fulfill_select_recv(struct kc_chan *ch, struct kc_pending_recv *node, kc_desc_id desc)
{
    (void)ch;
    kc_payload payload = {0};
    int rc = kc_desc_payload(desc, &payload);
    struct kc_chan_ptrmsg msg = { .ptr = payload.ptr, .len = payload.len };
    void *dst = kc_select_recv_buffer(node->sel, node->clause_index);
    if (rc == 0 && dst) memcpy(dst, &msg, sizeof(msg));
    complete_select(node->sel, node->clause_index, (rc == 0 && dst) ? 0 : KC_ECANCELED);
    kc_desc_release(desc);
    free(node);
}

/* ------------------------------------------------------------------------- */
/* Pointer send/recv (rendezvous only for now). */

int kc_chan_send_ptr(kc_chan_t *c, void *ptr, size_t len, long timeout_ms)
{
    struct kc_chan *ch = (struct kc_chan*)c;
    if (!ch || !ptr || len == 0) return -EINVAL;
    if (ch->kind != KC_RENDEZVOUS) return -ENOTSUP;

    kc_desc_id desc = 0;
    long deadline_ns = (timeout_ms > 0) ? kc_now_ns() + timeout_ms * 1000000L : 0;

    for (;;) {
        KC_MUTEX_LOCK(&ch->mu);
        if (ch->closed) {
            ch->send_epipe++;
            KC_MUTEX_UNLOCK(&ch->mu);
            return KC_EPIPE;
        }
        struct kc_pending_recv *wait = pending_recv_dequeue(ch);
        if (wait) {
            if (!desc) {
                desc = kc_desc_make_alias(ptr, len);
                if (!desc) {
                    KC_MUTEX_UNLOCK(&ch->mu);
                    free(wait);
                    return -ENOMEM;
                }
            }
            kc_chan_note_op_locked(ch, 1, len);
            kc_desc_retain(desc);
            KC_MUTEX_UNLOCK(&ch->mu);
            if (wait->role == KC_PENDING_ROLE_CORO) {
                fulfill_coroutine_recv(wait, desc);
            } else {
                fulfill_select_recv(ch, wait, desc);
            }
            return 0;
        }
        if (timeout_ms == 0) {
            ch->send_eagain++;
            KC_MUTEX_UNLOCK(&ch->mu);
            return KC_EAGAIN;
        }
        if (timeout_ms > 0) {
            KC_MUTEX_UNLOCK(&ch->mu);
            if (kc_now_ns() >= deadline_ns) {
                ch->send_etime++;
                return KC_ETIME;
            }
            kcoro_yield();
            continue;
        }
        if (!desc) {
            desc = kc_desc_make_alias(ptr, len);
            if (!desc) {
                KC_MUTEX_UNLOCK(&ch->mu);
                return -ENOMEM;
            }
        }
        struct kc_pending_send *node = calloc(1, sizeof(*node));
        if (!node) {
            KC_MUTEX_UNLOCK(&ch->mu);
            kc_desc_release(desc);
            return -ENOMEM;
        }
        kc_payload payload = {0};
        if (kc_desc_payload(desc, &payload) != 0) {
            KC_MUTEX_UNLOCK(&ch->mu);
            kc_desc_release(desc);
            free(node);
            return KC_EPIPE;
        }
        kc_ticket ticket = kc_token_kernel_publish_send(ch, payload.ptr, payload.len, NULL);
        if (ticket.id == 0) {
            kc_desc_release(desc);
            KC_MUTEX_UNLOCK(&ch->mu);
            free(node);
            return KC_EAGAIN;
        }
        node->kind = KC_PENDING_KIND_PTR;
        node->role = KC_PENDING_ROLE_CORO;
        node->ticket = ticket;
        node->desc_id = desc;
        pending_send_enqueue(ch, node);
        desc = 0;
        KC_MUTEX_UNLOCK(&ch->mu);

        kcoro_park();
        kc_payload ack = {0};
        int rc = kc_token_kernel_consume_payload(&ack);
        if (ack.desc_id) kc_desc_release(ack.desc_id);
        if (rc < 0) return rc;
        return ack.status;
    }
}

int kc_chan_recv_ptr(kc_chan_t *c, void **out_ptr, size_t *out_len, long timeout_ms)
{
    struct kc_chan *ch = (struct kc_chan*)c;
    if (!ch || !out_ptr || !out_len) return -EINVAL;
    if (ch->kind != KC_RENDEZVOUS) return -ENOTSUP;

    long deadline_ns = (timeout_ms > 0) ? kc_now_ns() + timeout_ms * 1000000L : 0;

    for (;;) {
        KC_MUTEX_LOCK(&ch->mu);
        struct kc_pending_send *wait = pending_send_dequeue(ch);
        if (wait) {
            kc_payload payload = {0};
            int rc = kc_desc_payload(wait->desc_id, &payload);
            if (rc != 0) {
                KC_MUTEX_UNLOCK(&ch->mu);
                if (wait->role == KC_PENDING_ROLE_CORO) {
                    kc_token_kernel_callback(wait->ticket, (kc_payload){ .ptr = NULL, .len = 0, .status = KC_EPIPE, .desc_id = 0 });
                } else {
                    complete_select(wait->sel, wait->clause_index, KC_EPIPE);
                }
                kc_desc_release(wait->desc_id);
                free(wait);
                return KC_EPIPE;
            }
            kc_chan_note_op_locked(ch, 0, payload.len);
            *out_ptr = payload.ptr;
            *out_len = payload.len;
            KC_MUTEX_UNLOCK(&ch->mu);
            if (wait->role == KC_PENDING_ROLE_CORO) {
                fulfill_coroutine_send(wait, wait->desc_id);
            } else {
                fulfill_select_send(ch, wait, wait->desc_id);
            }
            return 0;
        }
        if (timeout_ms == 0) {
            ch->recv_eagain++;
            KC_MUTEX_UNLOCK(&ch->mu);
            return KC_EAGAIN;
        }
        if (ch->closed) {
            ch->recv_epipe++;
            KC_MUTEX_UNLOCK(&ch->mu);
            return KC_EPIPE;
        }
        if (timeout_ms > 0) {
            KC_MUTEX_UNLOCK(&ch->mu);
            if (kc_now_ns() >= deadline_ns) {
                ch->recv_etime++;
                return KC_ETIME;
            }
            kcoro_yield();
            continue;
        }
        struct kc_pending_recv *node = calloc(1, sizeof(*node));
        if (!node) {
            KC_MUTEX_UNLOCK(&ch->mu);
            return -ENOMEM;
        }
        node->kind = KC_PENDING_KIND_PTR;
        node->role = KC_PENDING_ROLE_CORO;
        kc_ticket ticket = kc_token_kernel_publish_recv(ch, NULL);
        if (ticket.id == 0) {
            free(node);
            KC_MUTEX_UNLOCK(&ch->mu);
            return KC_EAGAIN;
        }
        node->ticket = ticket;
        pending_recv_enqueue(ch, node);
        KC_MUTEX_UNLOCK(&ch->mu);

        kcoro_park();
        kc_payload payload = {0};
        int rc = kc_token_kernel_consume_payload(&payload);
        if (payload.desc_id) kc_desc_release(payload.desc_id);
        if (rc < 0) return rc;
        if (payload.status < 0) return payload.status;
        *out_ptr = payload.ptr;
        *out_len = payload.len;
        return 0;
    }
}

/* ------------------------------------------------------------------------- */
/* Generic send/recv stubs */

int kc_chan_send(kc_chan_t *ch, const void *msg, long timeout_ms)
{
    // TODO(arena): Route byte sends through the arena-backed descriptor path once
    // buffered/unlimited channels are ported off the legacy waiters (see
    // ARENA_ARCH_PLAN.md#2). For now we fail fast so the lab only exercises the
    // rendezvous pointer implementation.
    (void)ch; (void)msg; (void)timeout_ms;
    return -ENOTSUP;
}

int kc_chan_recv(kc_chan_t *ch, void *out, long timeout_ms)
{
    // TODO(arena): Implement arena-backed dequeue for byte channels once the
    // buffered/unlimited pending queues are wired up. Until then we surface the
    // stub so tests cannot silently fall back to the removed waiter code.
    (void)ch; (void)out; (void)timeout_ms;
    return -ENOTSUP;
}

int kc_chan_send_c(kc_chan_t *ch, const void *msg, long timeout_ms, const kc_cancel_t *cancel)
{
    if (cancel && kc_cancel_is_set(cancel)) return KC_ECANCELED;
    return kc_chan_send(ch, msg, timeout_ms);
}

int kc_chan_recv_c(kc_chan_t *ch, void *out, long timeout_ms, const kc_cancel_t *cancel)
{
    if (cancel && kc_cancel_is_set(cancel)) return KC_ECANCELED;
    return kc_chan_recv(ch, out, timeout_ms);
}

int kc_chan_send_ptr_c(kc_chan_t *c, void *ptr, size_t len, long timeout_ms, const kc_cancel_t *cancel)
{
    if (cancel && kc_cancel_is_set(cancel)) return KC_ECANCELED;
    return kc_chan_send_ptr(c, ptr, len, timeout_ms);
}

int kc_chan_recv_ptr_c(kc_chan_t *c, void **out_ptr, size_t *out_len, long timeout_ms, const kc_cancel_t *cancel)
{
    if (cancel && kc_cancel_is_set(cancel)) return KC_ECANCELED;
    return kc_chan_recv_ptr(c, out_ptr, out_len, timeout_ms);
}

/* ------------------------------------------------------------------------- */
/* Select registration (rendezvous pointer only). */

int kc_chan_select_register_recv(kc_chan_t *c, kc_select_t *sel, int clause_index)
{
    struct kc_chan *ch = (struct kc_chan*)c;
    if (!ch || !sel) return -EINVAL;
    if (ch->kind != KC_RENDEZVOUS) {
        // TODO(arena): Extend select registration to buffered/unlimited channels
        // once their pending queues speak arena descriptors. Today only
        // rendezvous pointer clauses are supported in the lab.
        return -ENOTSUP;
    }

    KC_MUTEX_LOCK(&ch->mu);
    struct kc_pending_send *pending = pending_send_dequeue(ch);
    if (pending) {
        kc_payload payload = {0};
        int rc = kc_desc_payload(pending->desc_id, &payload);
        if (rc != 0) {
            KC_MUTEX_UNLOCK(&ch->mu);
            complete_select(sel, clause_index, KC_EPIPE);
            kc_desc_release(pending->desc_id);
            free(pending);
            return KC_EPIPE;
        }
        kc_chan_note_op_locked(ch, 0, payload.len);
        KC_MUTEX_UNLOCK(&ch->mu);

        struct kc_chan_ptrmsg msg = { .ptr = payload.ptr, .len = payload.len };
        void *dst = kc_select_recv_buffer(sel, clause_index);
        if (dst) memcpy(dst, &msg, sizeof(msg));
        complete_select(sel, clause_index, dst ? 0 : KC_ECANCELED);

        if (pending->role == KC_PENDING_ROLE_CORO) {
            fulfill_coroutine_send(pending, pending->desc_id);
        } else {
            fulfill_select_send(ch, pending, pending->desc_id);
        }
        return 0;
    }
    struct kc_pending_recv *node = calloc(1, sizeof(*node));
    if (!node) {
        KC_MUTEX_UNLOCK(&ch->mu);
        return -ENOMEM;
    }
    node->kind = KC_PENDING_KIND_PTR;
    node->role = KC_PENDING_ROLE_SELECT;
    node->sel = sel;
    node->clause_index = clause_index;
    node->desc_id = 0;
    pending_recv_enqueue(ch, node);
    KC_MUTEX_UNLOCK(&ch->mu);
    return KC_EAGAIN;
}

int kc_chan_select_register_send(kc_chan_t *c, kc_select_t *sel, int clause_index)
{
    struct kc_chan *ch = (struct kc_chan*)c;
    if (!ch || !sel) return -EINVAL;
    if (ch->kind != KC_RENDEZVOUS) {
        // TODO(arena): Align send-side select support with buffered/unlimited
        // queues after they adopt arena descriptors. Stub keeps the lab honest.
        return -ENOTSUP;
    }

    KC_MUTEX_LOCK(&ch->mu);
    struct kc_pending_recv *pending = pending_recv_dequeue(ch);
    if (pending) {
        const void *src = kc_select_send_buffer(sel, clause_index);
        if (!src) {
            KC_MUTEX_UNLOCK(&ch->mu);
            complete_select(sel, clause_index, KC_ECANCELED);
            if (pending->desc_id) kc_desc_release(pending->desc_id);
            if (pending->role == KC_PENDING_ROLE_CORO) {
                kc_token_kernel_callback(pending->ticket, (kc_payload){ .ptr = NULL, .len = 0, .status = KC_ECANCELED, .desc_id = 0 });
            } else {
                complete_select(pending->sel, pending->clause_index, KC_ECANCELED);
            }
            free(pending);
            return KC_ECANCELED;
        }
        const struct kc_chan_ptrmsg *msg = src;
        kc_desc_id desc = kc_desc_make_alias(msg->ptr, msg->len);
        if (!desc) {
            KC_MUTEX_UNLOCK(&ch->mu);
            complete_select(sel, clause_index, KC_EPIPE);
            if (pending->desc_id) kc_desc_release(pending->desc_id);
            if (pending->role == KC_PENDING_ROLE_CORO) {
                kc_token_kernel_callback(pending->ticket, (kc_payload){ .ptr = NULL, .len = 0, .status = KC_EPIPE, .desc_id = 0 });
            } else {
                complete_select(pending->sel, pending->clause_index, KC_EPIPE);
            }
            free(pending);
            return KC_EPIPE;
        }
        kc_chan_note_op_locked(ch, 1, msg->len);
        KC_MUTEX_UNLOCK(&ch->mu);
        if (pending->role == KC_PENDING_ROLE_CORO) {
            fulfill_coroutine_recv(pending, desc);
        } else {
            fulfill_select_recv(ch, pending, desc);
        }
        complete_select(sel, clause_index, 0);
        return 0;
    }
    struct kc_pending_send *node = calloc(1, sizeof(*node));
    if (!node) {
        KC_MUTEX_UNLOCK(&ch->mu);
        return -ENOMEM;
    }
    node->kind = KC_PENDING_KIND_PTR;
    node->role = KC_PENDING_ROLE_SELECT;
    node->sel = sel;
    node->clause_index = clause_index;
    const void *src = kc_select_send_buffer(sel, clause_index);
    if (src) {
        const struct kc_chan_ptrmsg *msg = src;
        node->desc_id = kc_desc_make_alias(msg->ptr, msg->len);
        if (!node->desc_id) {
            KC_MUTEX_UNLOCK(&ch->mu);
            free(node);
            return -ENOMEM;
        }
    } else {
        node->desc_id = 0;
    }
    pending_send_enqueue(ch, node);
    KC_MUTEX_UNLOCK(&ch->mu);
    return KC_EAGAIN;
}

void kc_chan_select_cancel(kc_chan_t *c, kc_select_t *sel, int clause_index, enum kc_select_clause_kind kind)
{
    struct kc_chan *ch = (struct kc_chan*)c;
    if (!ch || !sel) return;
    KC_MUTEX_LOCK(&ch->mu);
    if (kind == KC_SELECT_CLAUSE_SEND) pending_send_remove_select(ch, sel, clause_index);
    else pending_recv_remove_select(ch, sel, clause_index);
    KC_MUTEX_UNLOCK(&ch->mu);
}

/* ------------------------------------------------------------------------- */
/* Pointer-channel creation */

int kc_chan_make_ptr(kc_chan_t **out, int kind, size_t capacity)
{
    int rc = kc_chan_make(out, kind, sizeof(struct kc_chan_ptrmsg), capacity);
    if (rc == 0) {
        struct kc_chan *ch = (struct kc_chan*)(*out);
        ch->ptr_mode = 1;
        ch->capabilities |= KC_CHAN_CAP_PTR;
    }
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Stats and metrics stubs */

int kc_chan_get_stats(kc_chan_t *c, struct kc_chan_stats *out)
{
    if (!c || !out) return -EINVAL;
    // TODO(metrics): Rehydrate stats once the unified path updates counters for
    // every channel kind. Returning zeros prevents stale waiter-era data from
    // leaking into tools.
    memset(out, 0, sizeof(*out));
    return 0;
}

int kc_chan_snapshot(kc_chan_t *c, struct kc_chan_snapshot *out)
{
    if (!c || !out) return -EINVAL;
    // TODO(metrics): Populate snapshots with arena/queue depth when the unified
    // buffered implementation lands (ARENA_ARCH_PLAN.md#4).
    memset(out, 0, sizeof(*out));
    return 0;
}

int kc_chan_compute_rate(const struct kc_chan_snapshot *prev,
                         const struct kc_chan_snapshot *curr,
                         struct kc_chan_rate_sample *out)
{
    (void)prev; (void)curr; (void)out;
    return -ENOTSUP;
}

int kc_chan_enable_metrics_pipe(kc_chan_t *ch, kc_chan_t **out_pipe, size_t capacity)
{
    (void)ch; (void)out_pipe; (void)capacity;
    // TODO(metrics): Restore metrics pipe emission once counters and arena stats
    // are accurate under the new token path.
    return -ENOTSUP;
}

int kc_chan_disable_metrics_pipe(kc_chan_t *ch)
{
    (void)ch;
    // TODO(metrics): Complement kc_chan_enable_metrics_pipe with a real tear-down
    // when metrics piping is reintroduced.
    return -ENOTSUP;
}

kc_chan_t *kc_chan_metrics_pipe(kc_chan_t *ch)
{
    (void)ch;
    return NULL;
}

/* ------------------------------------------------------------------------- */
/* Zero-copy stubs */

unsigned kc_chan_capabilities(kc_chan_t *ch)
{
    if (!ch) return 0;
    struct kc_chan *c = (struct kc_chan*)ch;
    return c->capabilities;
}

int kc_chan_enable_zero_copy(kc_chan_t *ch)
{
    (void)ch;
    // TODO(arena): Bind the arena-backed zref backend here once descriptors are
    // implemented.
    return -ENOTSUP;
}

int kc_chan_send_zref(kc_chan_t *ch, void *ptr, size_t len, long timeout_ms)
{
    (void)ch; (void)ptr; (void)len; (void)timeout_ms;
    // TODO(arena): Expose arena-issued descriptors via the zref API.
    return -ENOTSUP;
}

int kc_chan_send_zref_c(kc_chan_t *ch, void *ptr, size_t len, long timeout_ms, const kc_cancel_t *cancel)
{
    if (cancel && kc_cancel_is_set(cancel)) return KC_ECANCELED;
    return kc_chan_send_zref(ch, ptr, len, timeout_ms);
}

int kc_chan_recv_zref(kc_chan_t *ch, void **out_ptr, size_t *out_len, long timeout_ms)
{
    (void)ch; (void)out_ptr; (void)out_len; (void)timeout_ms;
    // TODO(arena): Complement kc_chan_send_zref once arena descriptors exist.
    return -ENOTSUP;
}

int kc_chan_recv_zref_c(kc_chan_t *ch, void **out_ptr, size_t *out_len, long timeout_ms, const kc_cancel_t *cancel)
{
    if (cancel && kc_cancel_is_set(cancel)) return KC_ECANCELED;
    return kc_chan_recv_zref(ch, out_ptr, out_len, timeout_ms);
}

int kc_chan_get_zstats(kc_chan_t *ch, struct kc_chan_zstats *out)
{
    (void)ch; (void)out;
    // TODO(arena): Publish arena/zref counters after the zero-copy path is wired.
    return -ENOTSUP;
}

/* ------------------------------------------------------------------------- */
