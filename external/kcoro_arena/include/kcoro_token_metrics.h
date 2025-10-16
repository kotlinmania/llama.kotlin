// SPDX-License-Identifier: BSD-3-Clause
#ifndef KCORO_TOKEN_METRICS_H
#define KCORO_TOKEN_METRICS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Token Kernel Metrics
 * 
 * Observability hooks for the token kernel matching process.
 * Tracks matches, retries, and CAS failures for performance tuning.
 */

typedef struct kc_token_kernel_metrics {
    uint64_t matches_total;         // Successful rendezvous matches
    uint64_t retries_total;         // CAS retry attempts due to contention
    uint64_t cas_failures_total;    // CAS operations that failed
    uint64_t publish_send_total;    // Total send tokens published
    uint64_t publish_recv_total;    // Total receive tokens published
    uint64_t callback_total;        // Total callbacks invoked
    uint64_t cancel_total;          // Total cancellations
} kc_token_kernel_metrics;

/**
 * Get current token kernel metrics snapshot.
 * Thread-safe; can be called anytime.
 * Returns 0 on success, -EINVAL if out is NULL.
 */
int kc_token_kernel_get_metrics(kc_token_kernel_metrics *out);

/**
 * Reset all token kernel metrics to zero.
 * Thread-safe.
 */
void kc_token_kernel_reset_metrics(void);

#ifdef __cplusplus
}
#endif

#endif // KCORO_TOKEN_METRICS_H
