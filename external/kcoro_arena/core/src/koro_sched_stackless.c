// SPDX-License-Identifier: BSD-3-Clause
/* koro_sched_stackless.c - Stackless coroutine scheduler
 *
 * Provides a thread-safe ready queue that can be driven from the token kernel
 * worker thread while the main thread executes continuations cooperatively.
 */
#include "kcoro_stackless.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    koro_cont_t    *head;
    koro_cont_t    *tail;
    size_t          active;      /* number of continuations tracked by scheduler */
    int             initialized;
    int             running;
    int             stop;
} koro_runtime_t;

static pthread_mutex_t g_sched_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_sched_cv = PTHREAD_COND_INITIALIZER;

static koro_runtime_t g_sched = {
    .head = NULL,
    .tail = NULL,
    .active = 0,
    .initialized = 0,
    .running = 0,
    .stop = 0,
};

static void koro_sched_ensure_init_locked(void)
{
    if (g_sched.initialized) {
        return;
    }
    g_sched.head = NULL;
    g_sched.tail = NULL;
    g_sched.active = 0;
    g_sched.running = 0;
    g_sched.stop = 0;
    g_sched.initialized = 1;
}

int koro_sched_init(void)
{
    pthread_mutex_lock(&g_sched_mu);
    koro_sched_ensure_init_locked();
    pthread_mutex_unlock(&g_sched_mu);
    return 0;
}

static koro_cont_t* dequeue_ready_locked(void)
{
    koro_cont_t *k = g_sched.head;
    if (!k) {
        return NULL;
    }
    g_sched.head = k->next;
    if (!g_sched.head) {
        g_sched.tail = NULL;
    }
    k->next = NULL;
    k->ready_enqueued = 0;
    return k;
}

void koro_sched_enqueue_ready(koro_cont_t* k)
{
    if (!k) return;

    pthread_mutex_lock(&g_sched_mu);
    koro_sched_ensure_init_locked();

    if (k->completed) {
        pthread_mutex_unlock(&g_sched_mu);
        return;
    }

    if (!k->tracked) {
        k->tracked = 1;
        g_sched.active++;
    }

    if (k->ready_enqueued) {
        pthread_mutex_unlock(&g_sched_mu);
        return;
    }

    k->next = NULL;
    k->ready_enqueued = 1;

    if (g_sched.tail) {
        g_sched.tail->next = k;
        g_sched.tail = k;
    } else {
        g_sched.head = g_sched.tail = k;
    }

    pthread_cond_signal(&g_sched_cv);
    pthread_mutex_unlock(&g_sched_mu);
}

static void mark_completed(koro_cont_t *k)
{
    pthread_mutex_lock(&g_sched_mu);
    if (k->tracked) {
        k->tracked = 0;
        if (g_sched.active > 0) {
            g_sched.active--;
        }
    }
    int managed = k->managed;
    pthread_mutex_unlock(&g_sched_mu);

    if (managed) {
        koro_cont_destroy(k);
    }
}

int koro_run(void)
{
    pthread_mutex_lock(&g_sched_mu);
    koro_sched_ensure_init_locked();

    g_sched.running++;

    for (;;) {
        while (!g_sched.head && !g_sched.stop && g_sched.active > 0) {
            pthread_cond_wait(&g_sched_cv, &g_sched_mu);
        }

        if (g_sched.stop && !g_sched.head) {
            break;
        }

        if (!g_sched.head) {
            /* No active continuations remain. */
            if (g_sched.active == 0) {
                break;
            }
            continue;
        }

        koro_cont_t *k = dequeue_ready_locked();
        pthread_mutex_unlock(&g_sched_mu);

        if (!k) {
            pthread_mutex_lock(&g_sched_mu);
            continue;
        }

        void *result = koro_cont_step(k);
        int completed = (result != NULL) || k->completed;
        if (completed) {
            mark_completed(k);
        }

        pthread_mutex_lock(&g_sched_mu);
    }

    g_sched.running--;
    g_sched.stop = 0;
    pthread_mutex_unlock(&g_sched_mu);
    return 0;
}

void koro_stop(void)
{
    pthread_mutex_lock(&g_sched_mu);
    if (g_sched.initialized) {
        g_sched.stop = 1;
        pthread_cond_broadcast(&g_sched_cv);
    }
    pthread_mutex_unlock(&g_sched_mu);
}

int koro_go(void* (*func)(koro_cont_t*), void* arg, size_t local_size)
{
    if (!func) {
        return -1;
    }

    koro_cont_t* k = koro_cont_create((koro_step_fn)func, arg, local_size);
    if (!k) {
        return -1;
    }

    k->managed = 1;
    koro_sched_enqueue_ready(k);
    return 0;
}
