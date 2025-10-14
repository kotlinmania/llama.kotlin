// SPDX-License-Identifier: BSD-3-Clause
/* test_token_kernel_basic.c - Basic token kernel operations test
 *
 * Tests the core send/receive flow through the token kernel.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include "kcoro_token_kernel.h"

static volatile int callback_invoked = 0;
static kc_payload received_payload = {0};
static int event_hits[KC_TOKEN_EVENT_COUNT] = {0};

static void test_resume_callback(void *ctx, const kc_payload *payload) {
    (void)ctx;
    callback_invoked = 1;
    if (payload) {
        received_payload = *payload;
    } else {
        memset(&received_payload, 0, sizeof(received_payload));
        received_payload.status = -1;
    }
    printf("  Resume callback invoked!\n");
}

static void event_logger(struct kc_chan *channel, const kc_payload *payload, void *user_ctx)
{
    (void)channel;
    (void)payload;
    intptr_t idx = (intptr_t)user_ctx;
    if (idx >= 0 && idx < KC_TOKEN_EVENT_COUNT) {
        event_hits[idx]++;
    }
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

    /* Subscribe to events (diagnostic) */
    for (int evt = 0; evt < KC_TOKEN_EVENT_COUNT; ++evt) {
        kc_token_kernel_subscribe((kc_token_event_type)evt, event_logger, (void*)(intptr_t)evt);
    }
    
    /* Test 2: Publish send */
    printf("\n[2] Publishing send token...\n");
    char test_data[] = "Hello from token kernel!";
    kc_ticket ticket = kc_token_kernel_publish_send(
        NULL,  /* channel (not needed for basic test) */
        test_data,
        strlen(test_data) + 1,
        test_resume_callback,
        NULL
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
    printf("  Event counts: sender_ready=%d receiver_ready=%d sender_matched=%d receiver_matched=%d cancelled=%d\n",
           event_hits[KC_TOKEN_EVENT_EMPTY_TO_SENDER_READY],
           event_hits[KC_TOKEN_EVENT_EMPTY_TO_RECEIVER_READY],
           event_hits[KC_TOKEN_EVENT_SENDER_TO_MATCHED],
           event_hits[KC_TOKEN_EVENT_RECEIVER_TO_MATCHED],
           event_hits[KC_TOKEN_EVENT_ANY_TO_CANCELLED]);
    
    /* Test 4: Shutdown */
    printf("\n[4] Shutting down kernel...\n");
    kc_token_kernel_global_shutdown();
    printf("  PASS: shutdown complete\n");
    
    printf("\n=== All Tests Passed ===\n");
    return 0;
}
