// SPDX-License-Identifier: BSD-3-Clause
/* test_metrics_enabled.c - Metrics collection test with metrics enabled
 *
 * Tests that metrics are properly collected when explicitly enabled.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "kcoro_token_kernel.h"
#include "kcoro_token_metrics.h"
#include "kcoro_desc.h"
#include "kcoro_desc_metrics.h"
#include "kcoro_config_runtime.h"

static volatile int callback_invoked = 0;

static void test_resume_callback(void *ctx, const kc_payload *payload) {
    (void)ctx;
    (void)payload;
    callback_invoked = 1;
}

int main(void) {
    printf("=== Metrics Enabled Test ===\n\n");
    
    /* Create a config file to enable metrics */
    const char *config_path = "/tmp/kcoro_test_config.json";
    FILE *f = fopen(config_path, "w");
    if (f) {
        fprintf(f, "{\n");
        fprintf(f, "  \"channel\": {\n");
        fprintf(f, "    \"metrics\": {\n");
        fprintf(f, "      \"auto_enable\": true,\n");
        fprintf(f, "      \"emit_min_ops\": 1,\n");
        fprintf(f, "      \"emit_min_ms\": 10\n");
        fprintf(f, "    }\n");
        fprintf(f, "  }\n");
        fprintf(f, "}\n");
        fclose(f);
        printf("[Init] Created config file with metrics enabled\n");
    }
    
    /* Load config before initializing subsystems */
    int rc = kc_runtime_config_init(config_path);
    if (rc != 0) {
        fprintf(stderr, "WARN: kc_runtime_config_init returned %d\n", rc);
    }
    
    const struct kc_runtime_config *cfg = kc_runtime_config_get();
    printf("[Config] auto_enable=%d, emit_min_ops=%lu\n", 
           cfg->chan_metrics_auto_enable, 
           cfg->chan_metrics_emit_min_ops);
    
    /* Test 1: Token Kernel Metrics */
    printf("\n[1] Testing token kernel metrics...\n");
    
    rc = kc_token_kernel_global_init();
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
    
    if (tk_metrics.publish_send_total == 0 || tk_metrics.publish_recv_total == 0) {
        fprintf(stderr, "FAIL: metrics not being tracked (are metrics enabled?)\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    if (!callback_invoked) {
        fprintf(stderr, "FAIL: callback was not invoked\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    printf("  PASS: Token kernel metrics collected correctly\n");
    
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
    
    if (desc_metrics.alias_created_total == 0 || desc_metrics.copy_created_total == 0) {
        fprintf(stderr, "FAIL: descriptor metrics not being tracked\n");
        return 1;
    }
    
    double hit_rate = (double)desc_metrics.lookup_hits / 
                     (desc_metrics.lookup_hits + desc_metrics.lookup_misses) * 100.0;
    printf("  Cache hit rate: %.2f%%\n", hit_rate);
    
    printf("  PASS: Descriptor metrics collected correctly\n");
    
    /* Cleanup */
    unlink(config_path);
    kc_desc_global_shutdown();
    kc_token_kernel_global_shutdown();
    
    printf("\n=== All Tests Passed ===\n");
    return 0;
}
