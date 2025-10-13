// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file kcoro_core.h
 * @brief Core coroutine primitives (create/destroy, resume/yield, park/unpark).
 *
 * -----------------------------------------------------------------------------
 * Header Surface & Optional Items
 * -----------------------------------------------------------------------------
 * Purpose
 *   Small, focused primitives for user‑space coroutines with an ARM64 switcher.
 *   Assembly keeps context hops fast; scheduling and policy remain in C.
 *
 * Optional items
 *   - This header intentionally has no optional tunables. The primitives here
 *     are foundational and stable.
 *
 * Install guidance
 *   - This header is part of the production public API and should be installed.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct kcoro kcoro_t;
typedef struct kcoro_sched kcoro_sched_t;

/* Stackless continuation step function type.
 * Returns NULL to signal suspension, or a pointer to the next step. */
typedef void* (*kcoro_step_fn_t)(struct kcoro* self);

/* Coroutine state */
typedef enum {
    KCORO_CREATED,           /* Created but not started */
    KCORO_READY,             /* Ready to run */
    KCORO_RUNNING,           /* Currently executing */
    KCORO_SUSPENDED,         /* Suspended (yielded) */
    KCORO_PARKED,            /* Parked (not runnable until explicitly unparked) */
    KCORO_FINISHED           /* Completed execution */
} kcoro_state_t;

/* Core coroutine structure - stackless CPS model */
struct kcoro {
    /* Stackless continuation: next step to execute */
    kcoro_step_fn_t next_step;   /* Function pointer to next continuation step */
    
    /* Coroutine metadata */
    kcoro_state_t state;         /* Current execution state */
    void* user_data;             /* User-defined state (replaces stack) */
    uint64_t id;                 /* Unique coroutine ID */

    /* Execution context */
    kcoro_t* main_co;            /* Main coroutine (yield target) */
    kcoro_sched_t* scheduler;    /* Owning scheduler */
    bool ready_enqueued;         /* Scheduler ready-queue flag */
    atomic_int running_flag;     /* 0 = idle, 1 = running */
    atomic_int refcount;         /* Reference count for lifetime management */
    
    /* Scheduler linkage */
    kcoro_t* next;               /* Next in queue */
    kcoro_t* prev;               /* Previous in queue */
    
    /* Debug info */
    const char* name;            /* Optional name for debugging */

    /* Channel handshake hints */
    int last_send_delivered;     /* 1 if last parked send was delivered by recv */
    int last_recv_delivered;     /* 1 if last parked recv had data delivered by send */
    int last_park_result;        /* result code after park: 0=success, KC_EPIPE=closed, KC_ETIME=timeout */
};

/**
 * @name Core coroutine API (Stackless CPS Model)
 * Create/destroy coroutines and control execution.
 * @{ */

/* Create a stackless coroutine with initial step function */
kcoro_t* kcoro_create_cps(kcoro_step_fn_t initial_step, void* user_data);
void kcoro_destroy(kcoro_t* co);

/* Set optional name for debugging */
void kcoro_set_name(kcoro_t* co, const char* name);

/** Execution control */
void kcoro_resume(kcoro_t* co);
kcoro_t* kcoro_current(void);
kcoro_t* kcoro_thread_main(void);
void kcoro_retain(kcoro_t* co);
void kcoro_release(kcoro_t* co);
void kcoro_yield(void);
void kcoro_yield_to(kcoro_t* target_co);

/** Parking (no fairness requeue) */
void kcoro_park(void);              /* Park current coroutine (must be running in scheduler context) */
void kcoro_unpark(kcoro_t* co);     /* Make a parked coroutine ready again */
int  kcoro_is_parked(const kcoro_t* co);

/** Main coroutine setup */
kcoro_t* kcoro_create_main(void);
void kcoro_set_thread_main(kcoro_t* main_co);
/** @} */

#ifdef __cplusplus
}
#endif
