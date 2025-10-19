// SPDX-License-Identifier: BSD-3-Clause
#include "kcoro_cpu_monitor.h"

#include <time.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>

/* Get thread CPU time from /proc filesystem (Linux-specific)
 * Returns CPU time in clock ticks, or 0 on error */
static uint64_t get_thread_cpu_ticks(pid_t tid)
{
#ifdef __linux__
    char path[256];
    snprintf(path, sizeof(path), "/proc/self/task/%d/stat", (int)tid);
    
    FILE *f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    
    /* Format of /proc/<pid>/task/<tid>/stat:
     * pid comm state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt 
     * utime stime cutime cstime ...
     * We want utime (field 14) + stime (field 15) */
    
    unsigned long utime = 0, stime = 0;
    int matched = fscanf(f, "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu",
                        &utime, &stime);
    fclose(f);
    
    if (matched != 2) {
        return 0;
    }
    
    return utime + stime;
#else
    (void)tid;
    return 0;
#endif
}

/* Get clock ticks per second */
static long get_clock_ticks_per_sec(void)
{
    static long ticks = 0;
    if (ticks == 0) {
        ticks = sysconf(_SC_CLK_TCK);
        if (ticks <= 0) {
            ticks = 100; // fallback
        }
    }
    return ticks;
}

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
    
    uint64_t wall_time = kc_cpu_get_monotonic_ns();
    if (wall_time == 0) {
        return -1;
    }
    
    /* Get CPU time for the monitored thread using /proc
     * NOTE: This implementation casts pthread_t to pid_t which works on Linux
     * where pthread_t is typically the thread ID, but is not portable to other
     * platforms. For production use, consider using syscall(SYS_gettid) within
     * the monitored thread or storing the actual TID separately. */
    pid_t tid = (pid_t)mon->thread_id;
    uint64_t cpu_ticks = get_thread_cpu_ticks(tid);
    if (cpu_ticks == 0 && mon->baseline_ns != 0) {
        /* Could not read /proc, might not be Linux or permissions issue */
        return -1;
    }
    
    /* Convert ticks to nanoseconds */
    long ticks_per_sec = get_clock_ticks_per_sec();
    uint64_t cpu_time_ns = (cpu_ticks * 1000000000ULL) / (uint64_t)ticks_per_sec;
    
    /* Initialize baseline on first sample */
    if (mon->baseline_ns == 0) {
        mon->baseline_ns = cpu_time_ns;
        mon->baseline_wall_ns = wall_time;
    }
    
    sample->timestamp_ns = wall_time;
    sample->cpu_time_ns = cpu_time_ns;
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
