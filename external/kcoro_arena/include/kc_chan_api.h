// SPDX-License-Identifier: BSD-3-Clause
/* kc_chan_api.h - Stackless channel API
 *
 * Public API for stackless-only channels in kcoro_arena.
 * Provides rendezvous, buffered, and conflated channel types
 * with zero-spin operation via the token kernel.
 */
#ifndef KC_CHAN_API_H
#define KC_CHAN_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct koro_cont;
struct kc_chan;

/* Channel types */
typedef enum {
    KC_CHAN_RENDEZVOUS,  /* Unbuffered: sender waits for receiver */
    KC_CHAN_BUFFERED,    /* Buffered: N slots, blocks when full */
    KC_CHAN_CONFLATED    /* Latest-value: overwrites unread data */
} kc_chan_type_t;

/* Channel creation and destruction */

/* Create a stackless channel.
 * - type: Channel type (rendezvous, buffered, or conflated)
 * - capacity: Buffer size (0 for rendezvous, N for buffered, 1 for conflated)
 * Returns channel handle or NULL on error. */
struct kc_chan* kc_chan_make_stackless(kc_chan_type_t type, size_t capacity);

/* Close a stackless channel.
 * Wakes all waiting coroutines with error status.
 * Returns 0 on success, negative on error. */
int kc_chan_close_stackless(struct kc_chan* ch);

/* Destroy a stackless channel and free resources.
 * Channel must be closed first. */
void kc_chan_destroy_stackless(struct kc_chan* ch);

/* Stackless channel operations */

/* Send data to channel (stackless).
 * - k: Continuation record
 * - ch: Channel handle
 * - data: Data to send (copied)
 * - len: Data length in bytes
 * Returns 0 if completed immediately, 1 if suspended, negative on error. */
int kc_chan_send_stackless(struct koro_cont* k, struct kc_chan* ch, 
                           const void* data, size_t len);

/* Receive data from channel (stackless).
 * - k: Continuation record
 * - ch: Channel handle
 * Returns 0 if completed immediately, 1 if suspended, negative on error.
 * On success, data is in k->arena_payload with length k->arena_payload_len. */
int kc_chan_recv_stackless(struct koro_cont* k, struct kc_chan* ch);

/* Channel introspection */

/* Get number of items currently in stackless channel buffer.
 * Returns count or 0 if channel is empty/rendezvous. */
size_t kc_chan_len_stackless(struct kc_chan* ch);

/* Get channel capacity.
 * Returns 0 for rendezvous, N for buffered, 1 for conflated. */
size_t kc_chan_cap(struct kc_chan* ch);

/* Check if channel is closed.
 * Returns 1 if closed, 0 if open. */
int kc_chan_is_closed(struct kc_chan* ch);

#ifdef __cplusplus
}
#endif

#endif /* KC_CHAN_API_H */
