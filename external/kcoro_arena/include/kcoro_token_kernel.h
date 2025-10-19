// SPDX-License-Identifier: BSD-3-Clause
#ifndef KCORO_TOKEN_KERNEL_H
#define KCORO_TOKEN_KERNEL_H

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kc_chan;
struct kcoro;

typedef uint64_t kc_token_id_t;

typedef struct kc_payload {
    void   *ptr;
    size_t  len;
    int     status;   // 0=OK, negative = error (e.g. KC_ECANCELLED)
    uint64_t desc_id; // descriptor backing the payload (0 when not used)
} kc_payload;

typedef struct kc_ticket {
    kc_token_id_t      id;
    struct kc_chan *channel;
} kc_ticket;

typedef void (*kc_token_resume_fn)(void *user_ctx, const kc_payload *payload);

typedef enum {
    KC_TOKEN_EVENT_EMPTY_TO_SENDER_READY = 0,
    KC_TOKEN_EVENT_EMPTY_TO_RECEIVER_READY = 1,
    KC_TOKEN_EVENT_SENDER_TO_MATCHED = 2,
    KC_TOKEN_EVENT_RECEIVER_TO_MATCHED = 3,
    KC_TOKEN_EVENT_ANY_TO_CANCELLED = 4,
    KC_TOKEN_EVENT_COUNT
} kc_token_event_type;

typedef void (*kc_token_event_cb)(struct kc_chan *channel,
                                  const kc_payload *payload,
                                  void *user_ctx);

int kc_token_kernel_global_init(void);
void kc_token_kernel_global_shutdown(void);

kc_ticket kc_token_kernel_publish_send(struct kc_chan *ch,
                                       void *ptr,
                                       size_t len,
                                       kc_token_resume_fn resume_cb,
                                       void *user_ctx);

kc_ticket kc_token_kernel_publish_recv(struct kc_chan *ch,
                                       kc_token_resume_fn resume_cb,
                                       void *user_ctx);

void kc_token_kernel_callback(kc_ticket ticket, kc_payload payload);
void kc_token_kernel_cancel(kc_ticket ticket, int reason);

int kc_token_kernel_subscribe(kc_token_event_type event,
                              kc_token_event_cb cb,
                              void *user_ctx);

int kc_token_kernel_notify_event(kc_token_event_type event,
                                 struct kc_chan *channel,
                                 const kc_payload *payload);

pthread_t kc_token_kernel_get_worker_thread(void);

#ifdef __cplusplus
}
#endif

#endif // KCORO_TOKEN_KERNEL_H
