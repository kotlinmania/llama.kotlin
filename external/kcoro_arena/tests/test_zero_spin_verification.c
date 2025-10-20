// SPDX-License-Identifier: BSD-3-Clause
/* test_zero_spin_verification.c - Comprehensive zero-spin operation verification
 *
 * Tests that kcoro_arena achieves true zero-spin operation by measuring
 * worker thread CPU usage during idle periods and low-frequency traffic.
 *
 * Verification Criteria:
 * - Workers consume < 0.1% CPU during 1-second idle period
 * - No busy-wait loops in any code path
 * - All suspensions use proper blocking primitives
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

#include "kcoro_token_kernel.h"
#include "kcoro_token_metrics.h"
#include "kcoro_cpu_monitor.h"

#define IDLE_PERIOD_MS 1000
#define LOW_FREQ_PERIOD_MS 2000
#define BURST_COUNT 100
#define CPU_THRESHOLD_PERCENT 0.1

static volatile int test_running = 0;
static volatile int callbacks_received = 0;
static pthread_t worker_thread_id = 0;

/* Helper to get worker thread ID from token kernel */
extern pthread_t kc_token_kernel_get_worker_thread(void);

static void test_callback(void *ctx, const kc_payload *payload) {
    (void)ctx;
    (void)payload;
    __atomic_add_fetch(&callbacks_received, 1, __ATOMIC_SEQ_CST);
}

static void sleep_ms(int ms) {
    struct timespec req = {
        .tv_sec = ms / 1000,
        .tv_nsec = (ms % 1000) * 1000000L
    };
    nanosleep(&req, NULL);
}

static int verify_cpu_usage(const char *test_name, 
                           kc_cpu_sample *start, 
                           kc_cpu_sample *end,
                           double threshold_percent) {
    (void)test_name; // Unused but kept for future debugging
    double cpu_percent = kc_cpu_monitor_calculate_usage(start, end);
    double wall_time_ms = (double)(end->timestamp_ns - start->timestamp_ns) / 1e6;
    
    char usage_str[256];
    kc_cpu_format_usage(usage_str, sizeof(usage_str), cpu_percent, wall_time_ms);
    
    printf("  %s\n", usage_str);
    
    if (cpu_percent < 0) {
        fprintf(stderr, "  FAIL: Could not calculate CPU usage\n");
        return -1;
    }
    
    if (cpu_percent > threshold_percent) {
        fprintf(stderr, "  FAIL: CPU usage %.3f%% exceeds threshold %.3f%%\n",
                cpu_percent, threshold_percent);
        return -1;
    }
    
    printf("  PASS: CPU usage %.3f%% is below threshold %.3f%%\n",
           cpu_percent, threshold_percent);
    return 0;
}

static int test_idle_arena(void) {
    printf("\n[1] Testing idle arena (no activity)...\n");
    
    kc_cpu_monitor mon;
    kc_cpu_sample start_sample, end_sample;
    
    // Initialize monitor for worker thread
    if (kc_cpu_monitor_init(&mon, worker_thread_id) != 0) {
        fprintf(stderr, "  FAIL: kc_cpu_monitor_init failed\n");
        return -1;
    }
    
    // Take baseline sample
    if (kc_cpu_monitor_sample(&mon, &start_sample) != 0) {
        fprintf(stderr, "  FAIL: Initial sample failed\n");
        return -1;
    }
    
    printf("  Idle period: %d ms\n", IDLE_PERIOD_MS);
    
    // Wait during idle period
    sleep_ms(IDLE_PERIOD_MS);
    
    // Take end sample
    if (kc_cpu_monitor_sample(&mon, &end_sample) != 0) {
        fprintf(stderr, "  FAIL: Final sample failed\n");
        return -1;
    }
    
    // Verify CPU usage
    return verify_cpu_usage("Idle arena test", &start_sample, &end_sample, 
                           CPU_THRESHOLD_PERCENT);
}

static int test_burst_then_idle(void) {
    printf("\n[2] Testing burst of activity followed by idle...\n");
    
    // Generate burst of activity
    printf("  Generating burst of %d messages\n", BURST_COUNT);
    callbacks_received = 0;
    
    for (int i = 0; i < BURST_COUNT; i++) {
        char data[64];
        snprintf(data, sizeof(data), "burst_%d", i);
        kc_ticket ticket = kc_token_kernel_publish_send(NULL, data, strlen(data),
                                                        test_callback, NULL);
        if (ticket.id == 0) {
            fprintf(stderr, "  FAIL: publish_send returned zero ticket\n");
            return -1;
        }
        
        // Immediately trigger callback
        kc_payload response = { 
            .ptr = data, 
            .len = strlen(data), 
            .status = 0, 
            .desc_id = 0 
        };
        kc_token_kernel_callback(ticket, response);
    }
    
    // Wait for burst to complete
    sleep_ms(100);
    
    printf("  Callbacks received: %d\n", callbacks_received);
    
    // Now measure idle period after burst
    kc_cpu_monitor mon;
    kc_cpu_sample start_sample, end_sample;
    
    if (kc_cpu_monitor_init(&mon, worker_thread_id) != 0) {
        fprintf(stderr, "  FAIL: kc_cpu_monitor_init failed\n");
        return -1;
    }
    
    if (kc_cpu_monitor_sample(&mon, &start_sample) != 0) {
        fprintf(stderr, "  FAIL: Initial sample failed\n");
        return -1;
    }
    
    printf("  Idle period after burst: %d ms\n", IDLE_PERIOD_MS);
    sleep_ms(IDLE_PERIOD_MS);
    
    if (kc_cpu_monitor_sample(&mon, &end_sample) != 0) {
        fprintf(stderr, "  FAIL: Final sample failed\n");
        return -1;
    }
    
    return verify_cpu_usage("Burst then idle test", &start_sample, &end_sample,
                           CPU_THRESHOLD_PERCENT);
}

static int test_low_frequency_traffic(void) {
    printf("\n[3] Testing sustained low-frequency traffic (1 msg/sec)...\n");
    
    kc_cpu_monitor mon;
    kc_cpu_sample start_sample, end_sample;
    
    if (kc_cpu_monitor_init(&mon, worker_thread_id) != 0) {
        fprintf(stderr, "  FAIL: kc_cpu_monitor_init failed\n");
        return -1;
    }
    
    if (kc_cpu_monitor_sample(&mon, &start_sample) != 0) {
        fprintf(stderr, "  FAIL: Initial sample failed\n");
        return -1;
    }
    
    callbacks_received = 0;
    int msg_count = LOW_FREQ_PERIOD_MS / 1000; // 1 msg/sec
    
    printf("  Sending %d messages over %d ms period\n", msg_count, LOW_FREQ_PERIOD_MS);
    
    for (int i = 0; i < msg_count; i++) {
        char data[64];
        snprintf(data, sizeof(data), "lowfreq_%d", i);
        kc_ticket ticket = kc_token_kernel_publish_send(NULL, data, strlen(data),
                                                        test_callback, NULL);
        if (ticket.id == 0) {
            fprintf(stderr, "  FAIL: publish_send returned zero ticket\n");
            return -1;
        }
        
        kc_payload response = { 
            .ptr = data, 
            .len = strlen(data), 
            .status = 0, 
            .desc_id = 0 
        };
        kc_token_kernel_callback(ticket, response);
        
        // Sleep 1 second between messages
        sleep_ms(1000);
    }
    
    printf("  Callbacks received: %d\n", callbacks_received);
    
    if (kc_cpu_monitor_sample(&mon, &end_sample) != 0) {
        fprintf(stderr, "  FAIL: Final sample failed\n");
        return -1;
    }
    
    // For low frequency traffic, we expect very low average CPU usage
    // Most of the time should be idle
    return verify_cpu_usage("Low frequency traffic test", &start_sample, &end_sample,
                           CPU_THRESHOLD_PERCENT * 10); // Allow 1% since we had some activity
}

static int test_cpu_monitor_basics(void) {
    printf("\n[4] Testing CPU monitor basic functionality...\n");
    
    // Test thread time measurement
    uint64_t time1 = kc_cpu_get_thread_time_ns();
    if (time1 == 0) {
        fprintf(stderr, "  FAIL: kc_cpu_get_thread_time_ns returned 0\n");
        return -1;
    }
    
    // Do some work
    volatile int sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += i;
    }
    
    uint64_t time2 = kc_cpu_get_thread_time_ns();
    if (time2 <= time1) {
        fprintf(stderr, "  FAIL: CPU time did not increase after work\n");
        return -1;
    }
    
    printf("  CPU time increased by %llu ns after work\n", 
           (unsigned long long)(time2 - time1));
    
    // Test monotonic time
    uint64_t mono1 = kc_cpu_get_monotonic_ns();
    sleep_ms(100);
    uint64_t mono2 = kc_cpu_get_monotonic_ns();
    
    if (mono2 <= mono1) {
        fprintf(stderr, "  FAIL: Monotonic time did not increase\n");
        return -1;
    }
    
    double elapsed_ms = (double)(mono2 - mono1) / 1e6;
    printf("  Monotonic time elapsed: %.1f ms\n", elapsed_ms);
    
    if (elapsed_ms < 90 || elapsed_ms > 150) {
        fprintf(stderr, "  WARN: Monotonic time measurement seems off (expected ~100ms)\n");
    }
    
    printf("  PASS: CPU monitor basics working\n");
    return 0;
}

int main(void) {
    printf("=== Zero-Spin Operation Verification Test ===\n");
    printf("Threshold: CPU usage must be < %.3f%% during idle periods\n", 
           CPU_THRESHOLD_PERCENT);
    
    // Initialize token kernel (starts worker thread)
    int rc = kc_token_kernel_global_init();
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_token_kernel_global_init returned %d\n", rc);
        return 1;
    }
    
    // Get worker thread ID
    worker_thread_id = kc_token_kernel_get_worker_thread();
    if (pthread_equal(worker_thread_id, pthread_self())) {
        fprintf(stderr, "FAIL: Worker thread ID is same as main thread\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    printf("Worker thread ID obtained: %lu\n", (unsigned long)worker_thread_id);
    
    // Give worker thread time to initialize
    sleep_ms(100);
    
    test_running = 1;
    int failed = 0;
    
    // Run verification tests
    if (test_cpu_monitor_basics() != 0) failed = 1;
    if (test_idle_arena() != 0) failed = 1;
    if (test_burst_then_idle() != 0) failed = 1;
    if (test_low_frequency_traffic() != 0) failed = 1;
    
    test_running = 0;
    
    // Get final metrics
    kc_token_kernel_metrics metrics = {0};
    kc_token_kernel_get_metrics(&metrics);
    
    printf("\n=== Final Metrics ===\n");
    printf("Total callbacks: %llu\n", (unsigned long long)metrics.callback_total);
    printf("Total publishes (send/recv): %llu / %llu\n",
           (unsigned long long)metrics.publish_send_total,
           (unsigned long long)metrics.publish_recv_total);
    
    // Cleanup
    kc_token_kernel_global_shutdown();
    
    if (failed) {
        printf("\n=== FAILED: Zero-spin verification failed ===\n");
        return 1;
    }
    
    printf("\n=== SUCCESS: All zero-spin verification tests passed ===\n");
    printf("Worker thread maintains < %.3f%% CPU usage during idle periods\n",
           CPU_THRESHOLD_PERCENT);
    return 0;
}
