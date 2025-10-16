// SPDX-License-Identifier: BSD-3-Clause
#ifndef KCORO_DESC_METRICS_H
#define KCORO_DESC_METRICS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Descriptor LRU Metrics
 * 
 * Telemetry for descriptor alias caching and reference counting.
 * Useful for tuning cache sizes and understanding memory patterns.
 */

typedef struct kc_desc_metrics {
    uint64_t alias_created_total;   // Total alias descriptors created
    uint64_t copy_created_total;    // Total copy descriptors created
    uint64_t retain_total;          // Total retain operations
    uint64_t release_total;         // Total release operations
    uint64_t descriptor_evicts;     // Descriptors evicted (refcount reached 0)
    uint64_t lookup_hits;           // Successful descriptor lookups
    uint64_t lookup_misses;         // Failed descriptor lookups
} kc_desc_metrics;

/**
 * Get current descriptor metrics snapshot.
 * Thread-safe; can be called anytime.
 * Returns 0 on success, -EINVAL if out is NULL.
 */
int kc_desc_get_metrics(kc_desc_metrics *out);

/**
 * Reset all descriptor metrics to zero.
 * Thread-safe.
 */
void kc_desc_reset_metrics(void);

#ifdef __cplusplus
}
#endif

#endif // KCORO_DESC_METRICS_H
