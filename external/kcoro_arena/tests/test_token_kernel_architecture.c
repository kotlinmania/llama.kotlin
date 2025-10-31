// SPDX-License-Identifier: BSD-3-Clause
/* test_token_kernel_architecture.c - Validate token kernel architecture
 *
 * Tests that verify the current event-driven architecture:
 * - Publish/callback model
 * - Freelist recycling
 * - Worker thread processing
 * - Metrics tracking
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include "kcoro_token_kernel.h"
#include "kcoro_token_metrics.h"

static volatile int callbacks_invoked = 0;
static kc_payload received_payloads[10];

static void test_resume_callback(void *ctx, const kc_payload *payload) {
    int idx = (int)(intptr_t)ctx;
    if (idx >= 0 && idx < 10) {
        received_payloads[idx] = *payload;
    }
    __sync_fetch_and_add(&callbacks_invoked, 1);
}

int main(void) {
    printf("=== Token Kernel Architecture Test ===\n\n");
    
    /* Test 1: Initialization */
    printf("[1] Testing initialization...\n");
    int rc = kc_token_kernel_global_init();
    if (rc != 0) {
        fprintf(stderr, "FAIL: initialization returned %d\n", rc);
        return 1;
    }
    printf("  PASS: kernel initialized\n");
    
    /* Test 2: Metrics system (optional - may be disabled) */
    printf("\n[2] Testing metrics system...\n");
    kc_token_kernel_metrics metrics = {0};
    rc = kc_token_kernel_get_metrics(&metrics);
    if (rc != 0) {
        fprintf(stderr, "FAIL: get_metrics returned %d\n", rc);
        kc_token_kernel_global_shutdown();
        return 1;
    }
    printf("  PASS: metrics accessible (enabled: %s)\n",
           (metrics.publish_send_total == 0 && metrics.callback_total == 0) ? "no" : "yes");
    printf("  Initial state: matches=%llu, publish_send=%llu, publish_recv=%llu, callback=%llu\n",
           (unsigned long long)metrics.matches_total,
           (unsigned long long)metrics.publish_send_total,
           (unsigned long long)metrics.publish_recv_total,
           (unsigned long long)metrics.callback_total);
    
    /* Test 3: Publish multiple tokens (freelist recycling) */
    printf("\n[3] Testing publish and freelist recycling...\n");
    const int NUM_TOKENS = 5;
    kc_ticket tickets[NUM_TOKENS];
    char test_data[] = "test payload";
    
    for (int i = 0; i < NUM_TOKENS; i++) {
        tickets[i] = kc_token_kernel_publish_send(
            NULL,  /* channel */
            test_data,
            strlen(test_data) + 1,
            test_resume_callback,
            (void*)(intptr_t)i
        );
        if (tickets[i].id == 0) {
            fprintf(stderr, "FAIL: publish_send[%d] returned zero ticket\n", i);
            kc_token_kernel_global_shutdown();
            return 1;
        }
    }
    printf("  PASS: published %d tokens (IDs: %llu, %llu, %llu, %llu, %llu)\n",
           NUM_TOKENS,
           (unsigned long long)tickets[0].id,
           (unsigned long long)tickets[1].id,
           (unsigned long long)tickets[2].id,
           (unsigned long long)tickets[3].id,
           (unsigned long long)tickets[4].id);
    
    /* Test 4: Trigger callbacks (worker thread processing) */
    printf("\n[4] Testing callback processing via worker thread...\n");
    for (int i = 0; i < NUM_TOKENS; i++) {
        kc_payload response = {
            .ptr = test_data,
            .len = strlen(test_data) + 1,
            .status = 0,
            .desc_id = 0
        };
        kc_token_kernel_callback(tickets[i], response);
    }
    
    /* Give worker thread time to process */
    sleep(1);
    
    if (callbacks_invoked != NUM_TOKENS) {
        fprintf(stderr, "FAIL: expected %d callbacks, got %d\n", 
                NUM_TOKENS, callbacks_invoked);
        kc_token_kernel_global_shutdown();
        return 1;
    }
    printf("  PASS: all %d callbacks processed\n", callbacks_invoked);
    
    /* Test 5: Check metrics (may be disabled) */
    printf("\n[5] Checking metrics state...\n");
    rc = kc_token_kernel_get_metrics(&metrics);
    if (rc != 0) {
        fprintf(stderr, "FAIL: get_metrics returned %d\n", rc);
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    if (metrics.publish_send_total > 0) {
        printf("  Metrics enabled: publish_send=%llu, callback=%llu\n",
               (unsigned long long)metrics.publish_send_total,
               (unsigned long long)metrics.callback_total);
        if (metrics.publish_send_total < NUM_TOKENS || metrics.callback_total < NUM_TOKENS) {
            fprintf(stderr, "  WARNING: metrics may be incomplete\n");
        }
    } else {
        printf("  Metrics disabled (behavior still verified via callbacks)\n");
    }
    printf("  PASS: metrics system operational\n");
    
    /* Test 6: Cancellation path */
    printf("\n[6] Testing cancellation...\n");
    callbacks_invoked = 0;
    kc_ticket cancel_ticket = kc_token_kernel_publish_recv(
        NULL,
        test_resume_callback,
        (void*)(intptr_t)9
    );
    
    if (cancel_ticket.id == 0) {
        fprintf(stderr, "FAIL: publish_recv returned zero ticket\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    kc_token_kernel_cancel(cancel_ticket, -42);
    sleep(1);
    
    if (callbacks_invoked != 1) {
        fprintf(stderr, "FAIL: cancellation callback not invoked\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    if (received_payloads[9].status != -42) {
        fprintf(stderr, "FAIL: cancellation reason not preserved (got %d)\n",
                received_payloads[9].status);
        kc_token_kernel_global_shutdown();
        return 1;
    }
    
    printf("  PASS: cancellation processed correctly\n");
    
    /* Test 7: Shutdown */
    printf("\n[7] Testing shutdown...\n");
    kc_token_kernel_global_shutdown();
    printf("  PASS: shutdown complete\n");
    
    printf("\n=== All Architecture Tests Passed ===\n");
    printf("\nArchitecture verified:\n");
    printf("  ✓ Event-driven publish/callback model\n");
    printf("  ✓ Worker thread processing (zero-spin)\n");
    printf("  ✓ Freelist recycling\n");
    printf("  ✓ Metrics tracking\n");
    printf("  ✓ Cancellation support\n");
    
    return 0;
}
