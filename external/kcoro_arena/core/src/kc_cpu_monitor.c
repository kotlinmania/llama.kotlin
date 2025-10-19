// SPDX-License-Identifier: BSD-3-Clause
#include "kcoro_cpu_monitor.h"

#include <time.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

uint64_t kc_cpu_get_thread_time_ns(void)
{
    struct timespec ts;
#ifdef CLOCK_THREAD_CPUTIME_ID
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#else
    (void)ts;
    return 0;
#endif
}

uint64_t kc_cpu_get_monotonic_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int kc_cpu_monitor_init(kc_cpu_monitor *mon, pthread_t thread_id)
{
    if (!mon) {
        return -1;
    }
    
    memset(mon, 0, sizeof(*mon));
    mon->thread_id = thread_id;
    
    // Note: We can't get CPU time for arbitrary threads directly with POSIX.
    // The caller must be the thread being monitored, or we need platform-specific code.
    // For now, we'll store the baseline when first sample is taken.
    mon->initialized = 1;
    
    return 0;
}

int kc_cpu_monitor_sample(kc_cpu_monitor *mon, kc_cpu_sample *sample)
{
    if (!mon || !sample || !mon->initialized) {
        return -1;
    }
    
    uint64_t cpu_time = kc_cpu_get_thread_time_ns();
    uint64_t wall_time = kc_cpu_get_monotonic_ns();
    
    if (cpu_time == 0 || wall_time == 0) {
        return -1;
    }
    
    // Initialize baseline on first sample
    if (mon->baseline_ns == 0) {
        mon->baseline_ns = cpu_time;
        mon->baseline_wall_ns = wall_time;
    }
    
    sample->timestamp_ns = wall_time;
    sample->cpu_time_ns = cpu_time;
    sample->thread_id = mon->thread_id;
    
    return 0;
}

double kc_cpu_monitor_calculate_usage(const kc_cpu_sample *start,
                                      const kc_cpu_sample *end)
{
    if (!start || !end) {
        return -1.0;
    }
    
    // Verify samples are from the same thread
    if (!pthread_equal(start->thread_id, end->thread_id)) {
        return -1.0;
    }
    
    // Calculate deltas
    uint64_t wall_delta_ns = end->timestamp_ns - start->timestamp_ns;
    uint64_t cpu_delta_ns = end->cpu_time_ns - start->cpu_time_ns;
    
    if (wall_delta_ns == 0) {
        return 0.0;
    }
    
    // CPU usage as percentage
    double usage = (100.0 * (double)cpu_delta_ns) / (double)wall_delta_ns;
    
    return usage;
}

int kc_cpu_format_usage(char *buf, size_t buf_size,
                        double cpu_percent, double wall_time_ms)
{
    if (!buf || buf_size == 0) {
        return 0;
    }
    
    return snprintf(buf, buf_size,
                   "CPU usage: %.3f%% over %.1f ms",
                   cpu_percent, wall_time_ms);
}
