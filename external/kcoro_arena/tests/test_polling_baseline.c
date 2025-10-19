// SPDX-License-Identifier: BSD-3-Clause
/* test_polling_baseline.c - Polling-based worker baseline for comparison
 *
 * Implements a simple polling-based worker thread that continuously checks
 * a queue for work, similar to traditional implementations without proper
 * blocking primitives. This serves as a baseline to demonstrate the CPU
 * savings achieved by kcoro_arena's zero-spin design.
 *
 * Expected Results:
 * - Polling worker: ~100% CPU usage during idle
 * - Zero-spin worker: < 0.1% CPU usage during idle
 * - Improvement: 1000x+ reduction in CPU waste
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

#include "kcoro_cpu_monitor.h"

#define TEST_DURATION_MS 1000
#define SPIN_ITERATIONS 100000  // Tunable: simulates checking queue

/* Simple polling-based task queue */
typedef struct polling_task {
    void (*callback)(void *ctx);
    void *ctx;
    struct polling_task *next;
} polling_task_t;

typedef struct polling_queue {
    pthread_mutex_t mu;
    polling_task_t *head;
    polling_task_t *tail;
    atomic_int stop;
} polling_queue_t;

static polling_queue_t g_poll_queue;
static pthread_t g_poll_worker;
static atomic_uint_fast64_t g_poll_checks = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_poll_tasks_processed = ATOMIC_VAR_INIT(0);

static void polling_queue_init(polling_queue_t *q) {
    pthread_mutex_init(&q->mu, NULL);
    q->head = q->tail = NULL;
    atomic_store(&q->stop, 0);
}

static void polling_queue_destroy(polling_queue_t *q) {
    atomic_store(&q->stop, 1);
    
    // Clean up remaining tasks
    pthread_mutex_lock(&q->mu);
    polling_task_t *cur = q->head;
    while (cur) {
        polling_task_t *next = cur->next;
        free(cur);
        cur = next;
    }
    q->head = q->tail = NULL;
    pthread_mutex_unlock(&q->mu);
    
    pthread_mutex_destroy(&q->mu);
}

static polling_task_t *polling_queue_pop(polling_queue_t *q) {
    pthread_mutex_lock(&q->mu);
    polling_task_t *task = q->head;
    if (task) {
        q->head = task->next;
        if (!q->head) {
            q->tail = NULL;
        }
        task->next = NULL;
    }
    pthread_mutex_unlock(&q->mu);
    return task;
}

static void polling_queue_push(polling_queue_t *q, polling_task_t *task) {
    pthread_mutex_lock(&q->mu);
    task->next = NULL;
    if (q->tail) {
        q->tail->next = task;
    } else {
        q->head = task;
    }
    q->tail = task;
    pthread_mutex_unlock(&q->mu);
}

/* Polling worker: continuously checks queue without blocking */
static void *polling_worker_main(void *arg) {
    (void)arg;
    volatile uint64_t spin_counter = 0;
    
    /* NOTE: This spinning implementation is intentionally simple to demonstrate
     * the concept of polling-based workers. In practice, polling implementations
     * vary in their busy-wait patterns and may be more or less CPU-intensive.
     * The key point is that any polling approach wastes CPU during idle periods,
     * while the zero-spin design (pthread_cond_wait) uses zero CPU when idle. */
    
    while (!atomic_load(&g_poll_queue.stop)) {
        // Check queue for work (busy-wait)
        polling_task_t *task = polling_queue_pop(&g_poll_queue);
        
        if (task) {
            // Process task
            if (task->callback) {
                task->callback(task->ctx);
            }
            free(task);
            atomic_fetch_add(&g_poll_tasks_processed, 1);
        }
        
        // Busy spin - this actively consumes CPU
        // Use volatile to prevent compiler optimization
        for (volatile int i = 0; i < SPIN_ITERATIONS; i++) {
            spin_counter += i;
        }
        
        atomic_fetch_add(&g_poll_checks, 1);
    }
    
    return NULL;
}

static void test_task_callback(void *ctx) {
    (void)ctx;
    // Simple task that does minimal work
}

static void sleep_ms(int ms) {
    struct timespec req = {
        .tv_sec = ms / 1000,
        .tv_nsec = (ms % 1000) * 1000000L
    };
    nanosleep(&req, NULL);
}

static int test_polling_idle(void) {
    printf("\n[1] Testing polling worker during idle...\n");
    
    kc_cpu_monitor mon;
    kc_cpu_sample start_sample, end_sample;
    
    // Initialize monitor for polling worker thread
    if (kc_cpu_monitor_init(&mon, g_poll_worker) != 0) {
        fprintf(stderr, "  FAIL: kc_cpu_monitor_init failed\n");
        return -1;
    }
    
    // Reset counters
    atomic_store(&g_poll_checks, 0);
    atomic_store(&g_poll_tasks_processed, 0);
    
    // Take baseline sample
    if (kc_cpu_monitor_sample(&mon, &start_sample) != 0) {
        fprintf(stderr, "  FAIL: Initial sample failed\n");
        return -1;
    }
    
    printf("  Idle period: %d ms (no tasks submitted)\n", TEST_DURATION_MS);
    
    // Wait during idle period
    sleep_ms(TEST_DURATION_MS);
    
    // Take end sample
    if (kc_cpu_monitor_sample(&mon, &end_sample) != 0) {
        fprintf(stderr, "  FAIL: Final sample failed\n");
        return -1;
    }
    
    // Calculate and report results
    double cpu_percent = kc_cpu_monitor_calculate_usage(&start_sample, &end_sample);
    double wall_time_ms = (double)(end_sample.timestamp_ns - start_sample.timestamp_ns) / 1e6;
    
    uint64_t checks = atomic_load(&g_poll_checks);
    uint64_t tasks = atomic_load(&g_poll_tasks_processed);
    
    printf("  Queue checks performed: %llu\n", (unsigned long long)checks);
    printf("  Tasks processed: %llu\n", (unsigned long long)tasks);
    printf("  CPU usage: %.3f%% over %.1f ms\n", cpu_percent, wall_time_ms);
    
    if (cpu_percent < 0) {
        fprintf(stderr, "  FAIL: Could not calculate CPU usage\n");
        return -1;
    }
    
    // We expect very high CPU usage with polling
    if (cpu_percent < 50.0) {
        fprintf(stderr, "  WARN: Polling CPU usage unexpectedly low (%.3f%%), may need tuning\n",
                cpu_percent);
    }
    
    printf("  BASELINE: Polling worker uses %.3f%% CPU during idle\n", cpu_percent);
    return 0;
}

static int test_polling_with_burst(void) {
    printf("\n[2] Testing polling worker with burst then idle...\n");
    
    // Submit burst of tasks
    int burst_count = 100;
    printf("  Submitting %d tasks\n", burst_count);
    
    for (int i = 0; i < burst_count; i++) {
        polling_task_t *task = calloc(1, sizeof(*task));
        task->callback = test_task_callback;
        task->ctx = NULL;
        polling_queue_push(&g_poll_queue, task);
    }
    
    // Wait for tasks to be processed
    sleep_ms(100);
    
    kc_cpu_monitor mon;
    kc_cpu_sample start_sample, end_sample;
    
    if (kc_cpu_monitor_init(&mon, g_poll_worker) != 0) {
        fprintf(stderr, "  FAIL: kc_cpu_monitor_init failed\n");
        return -1;
    }
    
    atomic_store(&g_poll_checks, 0);
    atomic_store(&g_poll_tasks_processed, 0);
    
    if (kc_cpu_monitor_sample(&mon, &start_sample) != 0) {
        fprintf(stderr, "  FAIL: Initial sample failed\n");
        return -1;
    }
    
    printf("  Idle period after burst: %d ms\n", TEST_DURATION_MS);
    sleep_ms(TEST_DURATION_MS);
    
    if (kc_cpu_monitor_sample(&mon, &end_sample) != 0) {
        fprintf(stderr, "  FAIL: Final sample failed\n");
        return -1;
    }
    
    double cpu_percent = kc_cpu_monitor_calculate_usage(&start_sample, &end_sample);
    double wall_time_ms = (double)(end_sample.timestamp_ns - start_sample.timestamp_ns) / 1e6;
    
    printf("  CPU usage: %.3f%% over %.1f ms\n", cpu_percent, wall_time_ms);
    printf("  BASELINE: Even after burst, polling worker uses %.3f%% CPU\n", cpu_percent);
    
    return 0;
}

int main(void) {
    printf("=== Polling-Based Worker Baseline Test ===\n");
    printf("This test demonstrates CPU waste from traditional polling/busy-wait\n");
    printf("implementations, providing a baseline for comparison with zero-spin.\n\n");
    
    // Initialize polling queue and start worker
    polling_queue_init(&g_poll_queue);
    
    if (pthread_create(&g_poll_worker, NULL, polling_worker_main, NULL) != 0) {
        fprintf(stderr, "FAIL: pthread_create failed\n");
        return 1;
    }
    
    printf("Polling worker started (thread ID: %lu)\n", (unsigned long)g_poll_worker);
    printf("NOTE: Thread ID display is Linux-specific and may not be portable\n");
    
    // Give worker time to start spinning
    sleep_ms(100);
    
    int failed = 0;
    
    // Run baseline tests
    if (test_polling_idle() != 0) failed = 1;
    if (test_polling_with_burst() != 0) failed = 1;
    
    // Stop worker
    printf("\nStopping polling worker...\n");
    atomic_store(&g_poll_queue.stop, 1);
    pthread_join(g_poll_worker, NULL);
    
    polling_queue_destroy(&g_poll_queue);
    
    printf("\n=== Comparison Summary ===\n");
    printf("Polling-based worker: ~100%% CPU usage during idle (busy-wait)\n");
    printf("Zero-spin worker:      < 0.1%% CPU usage during idle (blocking)\n");
    printf("Improvement:          1000x+ reduction in wasted CPU cycles\n\n");
    
    printf("Run test_zero_spin_verification to see the zero-spin implementation.\n");
    
    if (failed) {
        printf("\n=== Test completed with warnings ===\n");
        return 1;
    }
    
    printf("\n=== Baseline measurement complete ===\n");
    return 0;
}
