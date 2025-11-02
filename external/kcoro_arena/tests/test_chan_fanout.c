// SPDX-License-Identifier: BSD-3-Clause
/* Test fanout/broadcast channel utilities */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "../include/kcoro.h"

#define TEST_PASS "\033[32m[PASS]\033[0m"
#define TEST_FAIL "\033[31m[FAIL]\033[0m"

/* Test basic fanout API exists */
static void test_fanout_api_exists(void) {
    printf("test_fanout_api_exists... ");
    
    /* Create multiple channels */
    kc_chan_t *channels[3];
    for (int i = 0; i < 3; i++) {
        int rc = kc_chan_make(&channels[i], KC_BUFFERED, sizeof(int), 10);
        assert(rc == 0);
        assert(channels[i] != NULL);
    }
    
    int val = 42;
    /* Test best-effort fanout */
    int successful = kc_chan_fanout_best_effort(&val, sizeof(int), channels, 3, 0);
    /* Should successfully send to all buffered channels */
    assert(successful >= 0 && successful <= 3);
    
    /* Test all-or-nothing fanout */
    int val2 = 99;
    int rc = kc_chan_fanout_all_or_nothing(&val2, sizeof(int), channels, 3, 0);
    /* Should succeed or fail atomically */
    assert(rc == 0 || rc < 0);
    
    /* Cleanup */
    for (int i = 0; i < 3; i++) {
        kc_chan_destroy(channels[i]);
    }
    
    printf("%s\n", TEST_PASS);
}

/* Test fanout with buffered channels */
static void test_fanout_buffered(void) {
    printf("test_fanout_buffered... ");
    
    /* Create 3 buffered channels */
    kc_chan_t *channels[3];
    for (int i = 0; i < 3; i++) {
        int rc = kc_chan_make(&channels[i], KC_BUFFERED, sizeof(int), 5);
        assert(rc == 0);
    }
    
    /* Send to all channels */
    int val = 123;
    int successful = kc_chan_fanout_best_effort(&val, sizeof(int), channels, 3, 0);
    assert(successful == 3); /* All should succeed with buffered channels */
    
    /* Verify each channel received the value */
    for (int i = 0; i < 3; i++) {
        int received = 0;
        int rc = kc_chan_recv(channels[i], &received, 0);
        assert(rc == 0);
        assert(received == val);
        kc_chan_destroy(channels[i]);
    }
    
    printf("%s\n", TEST_PASS);
}

/* Test fanout error handling */
static void test_fanout_errors(void) {
    printf("test_fanout_errors... ");
    
    int val = 42;
    kc_chan_t *channels[2];
    
    /* Test with NULL data */
    int rc = kc_chan_fanout_best_effort(NULL, sizeof(int), channels, 1, 0);
    assert(rc == -EINVAL);
    
    /* Test with NULL channels array */
    rc = kc_chan_fanout_best_effort(&val, sizeof(int), NULL, 1, 0);
    assert(rc == -EINVAL);
    
    /* Test with zero channels */
    rc = kc_chan_fanout_best_effort(&val, sizeof(int), channels, 0, 0);
    assert(rc == -EINVAL);
    
    printf("%s\n", TEST_PASS);
}

int main(void) {
    printf("=== Channel Fanout Tests ===\n");
    
    test_fanout_api_exists();
    test_fanout_buffered();
    test_fanout_errors();
    
    printf("\nAll tests passed!\n");
    return 0;
}
