// SPDX-License-Identifier: BSD-3-Clause
/* test_metrics_basic.c - Basic metrics collection test
 *
 * Tests that metrics are properly collected for token kernel and descriptors.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "kcoro_token_kernel.h"
#include "kcoro_token_metrics.h"
#include "kcoro_desc.h"
#include "kcoro_desc_metrics.h"

static volatile int callback_invoked = 0;

static void test_resume_callback(void *ctx, const kc_payload *payload) {
    (void)ctx;
    (void)payload;
    callback_invoked = 1;
}

int main(void) {
    printf("=== Metrics Basic Test ===\n\n");
    
    /* Test 1: Token Kernel Metrics */
    printf("[1] Testing token kernel metrics...\n");
    
    int rc = kc_token_kernel_global_init();
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_token_kernel_global_init returned %d\n", rc);
        return 1;
    }
    
    kc_token_kernel_metrics tk_metrics = {0};
    rc = kc_token_kernel_get_metrics(&tk_metrics);
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_token_kernel_get_metrics returned %d\n", rc);
        return 1;
    }
    
    printf("  Initial metrics: matches=%llu, retries=%llu, cas_failures=%llu\n",
           (unsigned long long)tk_metrics.matches_total,
           (unsigned long long)tk_metrics.retries_total,
           (unsigned long long)tk_metrics.cas_failures_total);
    
    /* Publish some tokens */
    char test_data[] = "Test data";
    kc_ticket ticket1 = kc_token_kernel_publish_send(NULL, test_data, sizeof(test_data), 
                                                      test_resume_callback, NULL);
    kc_ticket ticket2 = kc_token_kernel_publish_recv(NULL, test_resume_callback, NULL);
    
    if (ticket1.id == 0 || ticket2.id == 0) {
        fprintf(stderr, "FAIL: ticket IDs are zero\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    /* Trigger callback */
    kc_payload response = { .ptr = test_data, .len = sizeof(test_data), .status = 0, .desc_id = 0 };
    kc_token_kernel_callback(ticket1, response);
    
    sleep(1); /* Let worker process */
    
    rc = kc_token_kernel_get_metrics(&tk_metrics);
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_token_kernel_get_metrics returned %d\n", rc);
        return 1;
    }
    
    printf("  After operations: publish_send=%llu, publish_recv=%llu, callback=%llu, matches=%llu\n",
           (unsigned long long)tk_metrics.publish_send_total,
           (unsigned long long)tk_metrics.publish_recv_total,
           (unsigned long long)tk_metrics.callback_total,
           (unsigned long long)tk_metrics.matches_total);
    
    if (!callback_invoked) {
        fprintf(stderr, "FAIL: callback was not invoked\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    printf("  PASS: Token kernel metrics collected\n");
    
    /* Test 2: Descriptor Metrics */
    printf("\n[2] Testing descriptor metrics...\n");
    
    rc = kc_desc_global_init();
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_desc_global_init returned %d\n", rc);
        return 1;
    }
    
    kc_desc_metrics desc_metrics = {0};
    rc = kc_desc_get_metrics(&desc_metrics);
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_desc_get_metrics returned %d\n", rc);
        return 1;
    }
    
    printf("  Initial metrics: alias_created=%llu, copy_created=%llu\n",
           (unsigned long long)desc_metrics.alias_created_total,
           (unsigned long long)desc_metrics.copy_created_total);
    
    /* Create descriptors */
    char data[] = "Descriptor test data";
    kc_desc_id alias_id = kc_desc_make_alias(data, sizeof(data));
    kc_desc_id copy_id = kc_desc_make_copy(data, sizeof(data));
    
    if (alias_id == 0 || copy_id == 0) {
        fprintf(stderr, "FAIL: descriptor IDs are zero\n");
        kc_desc_global_shutdown();
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    /* Retain and release */
    kc_desc_retain(alias_id);
    kc_desc_retain(copy_id);
    
    kc_payload payload_out = {0};
    rc = kc_desc_payload(alias_id, &payload_out);
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_desc_payload returned %d\n", rc);
        return 1;
    }
    
    kc_desc_release(alias_id);
    kc_desc_release(alias_id); /* Final release */
    kc_desc_release(copy_id);
    kc_desc_release(copy_id); /* Final release */
    
    rc = kc_desc_get_metrics(&desc_metrics);
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_desc_get_metrics returned %d\n", rc);
        return 1;
    }
    
    printf("  After operations:\n");
    printf("    alias_created=%llu, copy_created=%llu\n",
           (unsigned long long)desc_metrics.alias_created_total,
           (unsigned long long)desc_metrics.copy_created_total);
    printf("    retain=%llu, release=%llu, evicts=%llu\n",
           (unsigned long long)desc_metrics.retain_total,
           (unsigned long long)desc_metrics.release_total,
           (unsigned long long)desc_metrics.descriptor_evicts);
    printf("    lookup_hits=%llu, lookup_misses=%llu\n",
           (unsigned long long)desc_metrics.lookup_hits,
           (unsigned long long)desc_metrics.lookup_misses);
    
    printf("  PASS: Descriptor metrics collected\n");
    
    /* Test 3: Metrics Reset */
    printf("\n[3] Testing metrics reset...\n");
    
    kc_token_kernel_reset_metrics();
    kc_desc_reset_metrics();
    
    rc = kc_token_kernel_get_metrics(&tk_metrics);
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_token_kernel_get_metrics returned %d\n", rc);
        return 1;
    }
    
    rc = kc_desc_get_metrics(&desc_metrics);
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_desc_get_metrics returned %d\n", rc);
        return 1;
    }
    
    if (tk_metrics.matches_total != 0 || tk_metrics.publish_send_total != 0 ||
        desc_metrics.alias_created_total != 0 || desc_metrics.retain_total != 0) {
        fprintf(stderr, "FAIL: metrics not properly reset\n");
        return 1;
    }
    
    printf("  PASS: Metrics reset successfully\n");
    
    /* Cleanup */
    kc_desc_global_shutdown();
    kc_token_kernel_global_shutdown();
    
    printf("\n=== All Tests Passed ===\n");
    return 0;
}
