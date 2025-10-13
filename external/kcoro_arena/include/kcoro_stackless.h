// SPDX-License-Identifier: BSD-3-Clause
/* kcoro_stackless.h - Stackless coroutine primitives
 * 
 * This header defines the stackless coroutine model for kcoro_arena.
 * Unlike stackful coroutines that allocate separate stacks, stackless
 * coroutines maintain state in heap-allocated continuation records.
 *
 * Key benefits:
 * - Memory efficiency: ~100 bytes per coroutine vs 64KB+ stacks
 * - No assembly required: pure C implementation
 * - Portable: works on any architecture
 * - Cache-friendly: better locality
 *
 * Design inspired by Protothreads and continuation-passing style (CPS).
 */
#ifndef KCORO_STACKLESS_H
#define KCORO_STACKLESS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct koro_cont;
struct koro_scheduler;
struct kc_chan;
typedef uint64_t kc_token_id_t;

/* Ticket for arena operations (opaque to user) */
typedef struct {
    kc_token_id_t id;
    struct kc_chan* channel;
} kc_ticket;

/* Continuation function pointer type.
 * Returns NULL when suspended, non-NULL when complete.
 * The scheduler calls this repeatedly until it returns non-NULL. */
typedef void* (*koro_step_fn)(struct koro_cont* k);

/* Coroutine continuation record.
 * This is the heap-allocated "stack frame" for a stackless coroutine.
 * All state that must survive suspension is stored here. */
typedef struct koro_cont {
    /* Core state machine */
    int state;              /* Current resumption point (line number) */
    koro_step_fn next_step; /* Function to call on next resume */
    
    /* User state */
    void* user_data;        /* Points to user's local variables struct */
    void* user_arg;         /* Original argument passed to koro_go */
    
    /* Scheduler linkage */
    struct koro_cont* next; /* Next in ready queue */
    int ready_enqueued;     /* True if in scheduler's ready queue */
    
    /* Identity and lifecycle */
    uint64_t id;            /* Unique coroutine ID */
    const char* name;       /* Debug name */
    int completed;          /* True when coroutine has finished */
    
    /* Arena integration */
    int last_park_result;   /* Result from last suspension (arena status) */
    void* arena_payload;    /* Cached arena payload pointer */
    size_t arena_payload_len; /* Cached arena payload length */
    uint64_t arena_desc_id; /* Descriptor ID for zero-copy payloads */
    kc_ticket arena_ticket; /* Pending ticket for cancellation */
} koro_cont_t;

/* Protothread-style macros for user code.
 * These expand into a switch statement-based state machine. */

/* Begin a stackless coroutine function.
 * Place at the start of the function, before any locals. */
#define KORO_BEGIN(k) \
    switch ((k)->state) { \
        case 0:

/* End a stackless coroutine function.
 * Place at the end. Sets completed flag and returns. */
#define KORO_END(k) \
    } \
    (k)->state = 0; \
    (k)->completed = 1; \
    return (void*)1;

/* Suspend execution and yield to scheduler.
 * Saves current line as resumption point. */
#define KORO_YIELD(k) \
    do { \
        (k)->state = __LINE__; \
        return NULL; \
        case __LINE__:; \
    } while (0)

/* Suspend while waiting for a condition.
 * Yields repeatedly until condition becomes true. */
#define KORO_WAIT_UNTIL(k, condition) \
    do { \
        (k)->state = __LINE__; \
        case __LINE__: \
        if (!(condition)) return NULL; \
    } while (0)

/* Public API functions */

/* Create a new stackless coroutine.
 * - initial_step: first function to execute
 * - user_arg: argument passed to user code
 * - user_data_size: bytes to allocate for local variables
 * Returns continuation record or NULL on failure. */
koro_cont_t* koro_cont_create(koro_step_fn initial_step, 
                               void* user_arg,
                               size_t user_data_size);

/* Free a continuation record and its user_data.
 * Should only be called when coroutine is complete. */
void koro_cont_destroy(koro_cont_t* k);

/* Execute one step of a coroutine.
 * Returns NULL if suspended, non-NULL if complete.
 * This is what the scheduler calls. */
static inline void* koro_cont_step(koro_cont_t* k) {
    if (k->completed) return (void*)1;
    if (!k->next_step) return (void*)1;
    return k->next_step(k);
}

/* Check if a coroutine is complete. */
static inline int koro_cont_is_done(koro_cont_t* k) {
    return k->completed;
}

/* Stackless arena primitives.
 * These are CPS versions of the arena operations. */

/* Attempt to send to arena channel.
 * Returns immediately if successful.
 * Suspends and returns NULL if blocked. */
void* koro_send_stackless(koro_cont_t* k, struct kc_chan* ch, void* data, size_t len);

/* Attempt to receive from arena channel.
 * Returns immediately if data available (stores in k->arena_payload).
 * Suspends and returns NULL if blocked. */
void* koro_recv_stackless(koro_cont_t* k, struct kc_chan* ch);

/* Macros for arena operations in user code */

/* Send data through arena channel, suspending if necessary.
 * After resume, check k->last_park_result for status. */
#define KORO_SEND(k, ch, data, len) \
    do { \
        (k)->state = __LINE__; \
        case __LINE__: { \
            void* _res = koro_send_stackless((k), (ch), (data), (len)); \
            if (!_res) return NULL; \
        } \
    } while (0)

/* Receive data from arena channel, suspending if necessary.
 * After resume, data is in k->arena_payload with length k->arena_payload_len.
 * For zero-copy descriptors, k->arena_desc_id contains the descriptor ID. */
#define KORO_RECV(k, ch) \
    do { \
        (k)->state = __LINE__; \
        case __LINE__: { \
            void* _res = koro_recv_stackless((k), (ch)); \
            if (!_res) return NULL; \
        } \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* KCORO_STACKLESS_H */
