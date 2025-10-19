// SPDX-License-Identifier: BSD-3-Clause
#ifndef KCORO_CPU_MONITOR_H
#define KCORO_CPU_MONITOR_H

#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CPU Monitor - Thread CPU Time Measurement
 * 
 * Provides utilities to measure per-thread CPU usage for verifying
 * zero-spin operation. Uses clock_gettime(CLOCK_THREAD_CPUTIME_ID)
 * to measure actual CPU time consumed by threads.
 */

typedef struct kc_cpu_sample {
    uint64_t timestamp_ns;    // Wall-clock timestamp (monotonic)
    uint64_t cpu_time_ns;     // Thread CPU time consumed
    pthread_t thread_id;      // Thread being sampled
} kc_cpu_sample;

typedef struct kc_cpu_monitor {
    pthread_t thread_id;      // Thread to monitor
    uint64_t baseline_ns;     // Initial CPU time
    uint64_t baseline_wall_ns; // Initial wall time
    int initialized;
} kc_cpu_monitor;

/**
 * Initialize a CPU monitor for the specified thread.
 * Pass pthread_self() to monitor the calling thread.
 * 
 * Returns 0 on success, -1 on error.
 */
int kc_cpu_monitor_init(kc_cpu_monitor *mon, pthread_t thread_id);

/**
 * Take a CPU sample for the monitored thread.
 * Captures both wall-clock time and thread CPU time.
 * 
 * Returns 0 on success, -1 on error.
 */
int kc_cpu_monitor_sample(kc_cpu_monitor *mon, kc_cpu_sample *sample);

/**
 * Calculate CPU usage percentage between two samples.
 * 
 * Returns CPU usage as a percentage (0.0 to 100.0+).
 * Returns -1.0 on error (e.g., samples from different threads).
 */
double kc_cpu_monitor_calculate_usage(const kc_cpu_sample *start,
                                      const kc_cpu_sample *end);

/**
 * Get current thread CPU time in nanoseconds.
 * 
 * Returns CPU time in nanoseconds, or 0 on error.
 */
uint64_t kc_cpu_get_thread_time_ns(void);

/**
 * Get monotonic wall-clock time in nanoseconds.
 * 
 * Returns wall time in nanoseconds, or 0 on error.
 */
uint64_t kc_cpu_get_monotonic_ns(void);

/**
 * Format CPU usage for display.
 * 
 * buf: Output buffer
 * buf_size: Size of output buffer
 * cpu_percent: CPU usage as percentage
 * wall_time_ms: Wall time duration in milliseconds
 * 
 * Returns number of characters written (excluding null terminator).
 */
int kc_cpu_format_usage(char *buf, size_t buf_size,
                        double cpu_percent, double wall_time_ms);

#ifdef __cplusplus
}
#endif

#endif // KCORO_CPU_MONITOR_H
