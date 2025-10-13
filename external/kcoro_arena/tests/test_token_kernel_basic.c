// SPDX-License-Identifier: BSD-3-Clause
/* test_token_kernel_basic.c - Basic token kernel operations test
 *
 * Tests the core send/receive flow through the token kernel.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "kcoro_token_kernel.h"

/* Simple test: publish a send, trigger callback, verify we get woken */
static volatile int callback_invoked = 0;
static kc_payload received_payload = {0};

void test_resume_callback(void) {
    callback_invoked = 1;
    printf("  Resume callback invoked!\n");
}

int main(void) {
    printf("=== Token Kernel Basic Test ===\n\n");
    
    /* Test 1: Init/Shutdown */
    printf("[1] Testing init/shutdown...\n");
    int rc = kc_token_kernel_global_init();
    if (rc != 0) {
        fprintf(stderr, "FAIL: kc_token_kernel_global_init returned %d\n", rc);
        return 1;
    }
    printf("  PASS: kernel initialized\n");
    
    /* Test 2: Publish send */
    printf("\n[2] Publishing send token...\n");
    char test_data[] = "Hello from token kernel!";
    kc_ticket ticket = kc_token_kernel_publish_send(
        NULL,  /* channel (not needed for basic test) */
        test_data,
        strlen(test_data) + 1,
        test_resume_callback
    );
    
    if (ticket.id == 0) {
        fprintf(stderr, "FAIL: publish_send returned zero ticket\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }
    printf("  PASS: ticket.id = %llu\n", (unsigned long long)ticket.id);
    
    /* Test 3: Trigger callback */
    printf("\n[3] Triggering callback with payload...\n");
    kc_payload response = {
        .ptr = test_data,
        .len = strlen(test_data) + 1,
        .status = 0,
        .desc_id = 0
    };
    kc_token_kernel_callback(ticket, response);
    
    /* Give worker thread time to process */
    sleep(1);
    
    if (!callback_invoked) {
        fprintf(stderr, "FAIL: callback was not invoked\n");
        kc_token_kernel_global_shutdown();
        return 1;
    }
    printf("  PASS: callback invoked successfully\n");
    
    /* Test 4: Shutdown */
    printf("\n[4] Shutting down kernel...\n");
    kc_token_kernel_global_shutdown();
    printf("  PASS: shutdown complete\n");
    
    printf("\n=== All Tests Passed ===\n");
    return 0;
}
