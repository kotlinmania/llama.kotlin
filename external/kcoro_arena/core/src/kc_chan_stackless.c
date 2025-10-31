// SPDX-License-Identifier: BSD-3-Clause
#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <errno.h>

#include "../../include/kc_chan_api.h"
#include "../../include/kcoro_stackless.h"
#include "../../include/kcoro_token_kernel.h"
#include "../../include/koro_sched_stackless.h"

/* Internal channel structure */
typedef struct kc_chan {
    kc_chan_type_t type;
    size_t capacity;
    size_t count;         /* Current items in buffer */
    int closed;
    
    /* Circular buffer for buffered/conflated channels */
    void** buffer;        /* Array of data pointers */
    size_t* lengths;      /* Array of data lengths */
    size_t head;          /* Read position */
    size_t tail;          /* Write position */
    
    /* Wait queues */
    struct waiter_list {
        struct waiter* head;
        struct waiter* tail;
    } send_waiters, recv_waiters;
    
    pthread_mutex_t lock;
} kc_chan_t;

/* Waiter for suspended coroutines */
typedef struct waiter {
    struct koro_cont* cont;
    void* send_data_copy;     /* Owned copy for senders */
    size_t send_len;
    struct waiter* next;
} waiter_t;

/* Helper: allocate and copy data */
static void* copy_data(const void* data, size_t len) {
    if (!data || len == 0) return NULL;
    void* copy = malloc(len);
    if (!copy) return NULL;
    memcpy(copy, data, len);
    return copy;
}

/* Helper: enqueue waiter */
static void enqueue_waiter(struct waiter_list* list, waiter_t* w) {
    w->next = NULL;
    if (list->tail) {
        list->tail->next = w;
        list->tail = w;
    } else {
        list->head = list->tail = w;
    }
}

/* Helper: dequeue waiter */
static waiter_t* dequeue_waiter(struct waiter_list* list) {
    waiter_t* w = list->head;
    if (w) {
        list->head = w->next;
        if (!list->head) list->tail = NULL;
        w->next = NULL;
    }
    return w;
}

/* Create a stackless channel */
struct kc_chan* kc_chan_make_stackless(kc_chan_type_t type, size_t capacity) {
    kc_chan_t* ch = calloc(1, sizeof(kc_chan_t));
    if (!ch) return NULL;
    
    ch->type = type;
    ch->capacity = capacity;
    ch->count = 0;
    ch->closed = 0;
    ch->head = 0;
    ch->tail = 0;
    
    if (capacity > 0) {
        ch->buffer = calloc(capacity, sizeof(void*));
        ch->lengths = calloc(capacity, sizeof(size_t));
        if (!ch->buffer || !ch->lengths) {
            free(ch->buffer);
            free(ch->lengths);
            free(ch);
            return NULL;
        }
    }
    
    pthread_mutex_init(&ch->lock, NULL);
    
    return ch;
}

/* Close a stackless channel */
int kc_chan_close_stackless(struct kc_chan* ch) {
    if (!ch) return -EINVAL;
    
    pthread_mutex_lock(&ch->lock);
    if (ch->closed) {
        pthread_mutex_unlock(&ch->lock);
        return -EALREADY;
    }
    
    ch->closed = 1;
    
    /* Wake all waiting coroutines with error */
    waiter_t* w;
    while ((w = dequeue_waiter(&ch->send_waiters))) {
        w->cont->last_park_result = -EPIPE;
        koro_sched_enqueue_ready(w->cont);
        free(w->send_data_copy);
        free(w);
    }
    while ((w = dequeue_waiter(&ch->recv_waiters))) {
        w->cont->last_park_result = -EPIPE;
        koro_sched_enqueue_ready(w->cont);
        free(w);
    }
    
    pthread_mutex_unlock(&ch->lock);
    return 0;
}

/* Destroy a stackless channel */
void kc_chan_destroy_stackless(struct kc_chan* ch) {
    if (!ch) return;
    
    pthread_mutex_lock(&ch->lock);
    
    /* Free buffered data */
    if (ch->buffer) {
        for (size_t i = 0; i < ch->count; i++) {
            size_t idx = (ch->head + i) % ch->capacity;
            free(ch->buffer[idx]);
        }
        free(ch->buffer);
    }
    free(ch->lengths);
    
    /* Ensure no waiters remain */
    assert(ch->send_waiters.head == NULL);
    assert(ch->recv_waiters.head == NULL);
    
    pthread_mutex_unlock(&ch->lock);
    pthread_mutex_destroy(&ch->lock);
    
    free(ch);
}

/* Send data to channel (stackless) */
int kc_chan_send_stackless(struct koro_cont* k, struct kc_chan* ch,
                           const void* data, size_t len) {
    if (!k || !ch || !data) return -EINVAL;
    
    pthread_mutex_lock(&ch->lock);
    
    if (ch->closed) {
        pthread_mutex_unlock(&ch->lock);
        k->last_park_result = -EPIPE;
        return -EPIPE;
    }
    
    /* Rendezvous channel: match with waiting receiver */
    if (ch->type == KC_CHAN_RENDEZVOUS) {
        waiter_t* recv = dequeue_waiter(&ch->recv_waiters);
        if (recv) {
            /* Direct handoff */
            recv->cont->arena_payload = copy_data(data, len);
            if (!recv->cont->arena_payload) {
                recv->cont->arena_payload_len = 0;
                recv->cont->last_park_result = -ENOMEM;
                k->last_park_result = -ENOMEM;
                free(recv);
                pthread_mutex_unlock(&ch->lock);
                return -ENOMEM;
            }
            recv->cont->arena_payload_len = len;
            recv->cont->last_park_result = 0;
            koro_sched_enqueue_ready(recv->cont);
            free(recv);
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = 0;
            return 0;  /* Completed immediately */
        }
        
        /* No receiver: suspend sender */
        waiter_t* w = malloc(sizeof(waiter_t));
        if (!w) {
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = -ENOMEM;
            return -ENOMEM;
        }
        
        w->cont = k;
        w->send_data_copy = copy_data(data, len);
        if (!w->send_data_copy) {
            free(w);
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = -ENOMEM;
            return -ENOMEM;
        }
        w->send_len = len;
        enqueue_waiter(&ch->send_waiters, w);
        
        pthread_mutex_unlock(&ch->lock);
        k->last_park_result = 0;  /* Will be set when matched */
        return 1;  /* Suspended */
    }
    
    /* Buffered channel */
    if (ch->type == KC_CHAN_BUFFERED) {
        if (ch->count < ch->capacity) {
            /* Space available */
            void* data_copy = copy_data(data, len);
            if (!data_copy) {
                pthread_mutex_unlock(&ch->lock);
                k->last_park_result = -ENOMEM;
                return -ENOMEM;
            }
            ch->buffer[ch->tail] = data_copy;
            ch->lengths[ch->tail] = len;
            ch->tail = (ch->tail + 1) % ch->capacity;
            ch->count++;
            
            /* Wake waiting receiver if any */
            waiter_t* recv = dequeue_waiter(&ch->recv_waiters);
            if (recv) {
                recv->cont->arena_payload = ch->buffer[ch->head];
                recv->cont->arena_payload_len = ch->lengths[ch->head];
                ch->buffer[ch->head] = NULL;
                ch->head = (ch->head + 1) % ch->capacity;
                ch->count--;
                recv->cont->last_park_result = 0;
                koro_sched_enqueue_ready(recv->cont);
                free(recv);
            }
            
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = 0;
            return 0;  /* Completed immediately */
        }
        
        /* Buffer full: suspend sender */
        waiter_t* w = malloc(sizeof(waiter_t));
        if (!w) {
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = -ENOMEM;
            return -ENOMEM;
        }
        
        w->cont = k;
        w->send_data_copy = copy_data(data, len);
        w->send_len = len;
        enqueue_waiter(&ch->send_waiters, w);
        
        pthread_mutex_unlock(&ch->lock);
        k->last_park_result = 0;  /* Will be set when matched */
        return 1;  /* Suspended */
    }
    
    /* Conflated channel: always overwrites */
    if (ch->type == KC_CHAN_CONFLATED) {
        if (ch->count > 0) {
            /* Replace existing value */
            free(ch->buffer[0]);
            ch->buffer[0] = copy_data(data, len);
            if (!ch->buffer[0]) {
                pthread_mutex_unlock(&ch->lock);
                k->last_park_result = -ENOMEM;
                return -ENOMEM;
            }
            ch->lengths[0] = len;
        } else {
            /* Store new value */
            ch->buffer[0] = copy_data(data, len);
            if (!ch->buffer[0]) {
                pthread_mutex_unlock(&ch->lock);
                k->last_park_result = -ENOMEM;
                return -ENOMEM;
            }
            ch->lengths[0] = len;
            ch->count = 1;
        }
        
        /* Wake waiting receiver if any */
        waiter_t* recv = dequeue_waiter(&ch->recv_waiters);
        if (recv) {
            recv->cont->arena_payload = ch->buffer[0];
            recv->cont->arena_payload_len = ch->lengths[0];
            ch->buffer[0] = NULL;
            ch->count = 0;
            recv->cont->last_park_result = 0;
            koro_sched_enqueue_ready(recv->cont);
            free(recv);
        }
        
        pthread_mutex_unlock(&ch->lock);
        k->last_park_result = 0;
        return 0;  /* Completed immediately */
    }
    
    pthread_mutex_unlock(&ch->lock);
    k->last_park_result = -EINVAL;
    return -EINVAL;
}

/* Receive data from channel (stackless) */
int kc_chan_recv_stackless(struct koro_cont* k, struct kc_chan* ch) {
    if (!k || !ch) return -EINVAL;
    
    pthread_mutex_lock(&ch->lock);
    
    if (ch->closed && ch->count == 0) {
        pthread_mutex_unlock(&ch->lock);
        k->last_park_result = -EPIPE;
        return -EPIPE;
    }
    
    /* Rendezvous channel: match with waiting sender */
    if (ch->type == KC_CHAN_RENDEZVOUS) {
        waiter_t* send = dequeue_waiter(&ch->send_waiters);
        if (send) {
            /* Direct handoff - transfer ownership of data */
            /*
             * Ownership of send->send_data_copy is transferred to the receiver's k->arena_payload.
             * The receiver (k) is now responsible for eventually freeing k->arena_payload,
             * or otherwise documenting its expected lifecycle.
             */
            k->arena_payload = send->send_data_copy;
            k->arena_payload_len = send->send_len;
            k->last_park_result = 0;
            send->cont->last_park_result = 0;
            koro_sched_enqueue_ready(send->cont);
            free(send);
            pthread_mutex_unlock(&ch->lock);
            return 0;  /* Completed immediately */
        }
        
        /* No sender: suspend receiver */
        waiter_t* w = malloc(sizeof(waiter_t));
        if (!w) {
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = -ENOMEM;
            return -ENOMEM;
        }
        
        w->cont = k;
        w->send_data_copy = NULL;
        w->send_len = 0;
        enqueue_waiter(&ch->recv_waiters, w);
        
        pthread_mutex_unlock(&ch->lock);
        k->last_park_result = 0;  /* Will be set when matched */
        return 1;  /* Suspended */
    }
    
    /* Buffered channel */
    if (ch->type == KC_CHAN_BUFFERED) {
        if (ch->count > 0) {
            /* Data available */
            k->arena_payload = ch->buffer[ch->head];
            k->arena_payload_len = ch->lengths[ch->head];
            ch->buffer[ch->head] = NULL;
            ch->head = (ch->head + 1) % ch->capacity;
            ch->count--;
            
            /* Wake waiting sender if any */
            waiter_t* send = dequeue_waiter(&ch->send_waiters);
            if (send) {
                ch->buffer[ch->tail] = send->send_data_copy;
                ch->lengths[ch->tail] = send->send_len;
                ch->tail = (ch->tail + 1) % ch->capacity;
                ch->count++;
                send->cont->last_park_result = 0;
                koro_sched_enqueue_ready(send->cont);
                free(send);
            }
            
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = 0;
            return 0;  /* Completed immediately */
        }
        
        /* Buffer empty: suspend receiver */
        waiter_t* w = malloc(sizeof(waiter_t));
        if (!w) {
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = -ENOMEM;
            return -ENOMEM;
        }
        
        w->cont = k;
        w->send_data_copy = NULL;
        w->send_len = 0;
        enqueue_waiter(&ch->recv_waiters, w);
        
        pthread_mutex_unlock(&ch->lock);
        k->last_park_result = 0;  /* Will be set when matched */
        return 1;  /* Suspended */
    }
    
    /* Conflated channel */
    if (ch->type == KC_CHAN_CONFLATED) {
        if (ch->count > 0) {
            /* Data available */
            k->arena_payload = ch->buffer[0];
            k->arena_payload_len = ch->lengths[0];
            ch->buffer[0] = NULL;
            ch->count = 0;
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = 0;
            return 0;  /* Completed immediately */
        }
        
        /* No data: suspend receiver */
        waiter_t* w = malloc(sizeof(waiter_t));
        if (!w) {
            pthread_mutex_unlock(&ch->lock);
            k->last_park_result = -ENOMEM;
            return -ENOMEM;
        }
        
        w->cont = k;
        w->send_data_copy = NULL;
        w->send_len = 0;
        enqueue_waiter(&ch->recv_waiters, w);
        
        pthread_mutex_unlock(&ch->lock);
        k->last_park_result = 0;  /* Will be set when matched */
        return 1;  /* Suspended */
    }
    
    pthread_mutex_unlock(&ch->lock);
    k->last_park_result = -EINVAL;
    return -EINVAL;
}

/* Get number of items in stackless channel */
size_t kc_chan_len_stackless(struct kc_chan* ch) {
    if (!ch) return 0;
    pthread_mutex_lock(&ch->lock);
    size_t count = ch->count;
    pthread_mutex_unlock(&ch->lock);
    return count;
}

/* Get channel capacity */
size_t kc_chan_cap(struct kc_chan* ch) {
    if (!ch) return 0;
    return ch->capacity;
}

/* Check if channel is closed */
int kc_chan_is_closed(struct kc_chan* ch) {
    if (!ch) return 1;
    pthread_mutex_lock(&ch->lock);
    int closed = ch->closed;
    pthread_mutex_unlock(&ch->lock);
    return closed;
}
