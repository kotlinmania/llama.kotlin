// SPDX-License-Identifier: BSD-3-Clause
/* Test priority-aware channel operations */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "../include/kcoro.h"

#define TEST_PASS "\033[32m[PASS]\033[0m"
#define TEST_FAIL "\033[31m[FAIL]\033[0m"

/* Test basic priority API existence and compilation */
static void test_priority_api_exists(void) {
    printf("test_priority_api_exists... ");
    
    kc_chan_t *ch = NULL;
    int rc = kc_chan_make(&ch, KC_RENDEZVOUS, sizeof(int), 0);
    assert(rc == 0);
    assert(ch != NULL);
    
    int val = 42;
    /* Test that priority functions exist and can be called */
    /* Note: Implementation currently delegates to standard send/recv */
    rc = kc_chan_send_priority(ch, &val, 0, KC_CHAN_PRIORITY_HIGH);
    /* Non-blocking send on empty rendezvous channel should return -EAGAIN */
    assert(rc == -EAGAIN || rc == 0);
    
    kc_chan_destroy(ch);
    printf("%s\n", TEST_PASS);
}

/* Test priority constants are defined */
static void test_priority_constants(void) {
    printf("test_priority_constants... ");
    
    assert(KC_CHAN_PRIORITY_LOW == 64);
    assert(KC_CHAN_PRIORITY_NORMAL == 128);
    assert(KC_CHAN_PRIORITY_HIGH == 192);
    
    printf("%s\n", TEST_PASS);
}

int main(void) {
    printf("=== Channel Priority Tests ===\n");
    
    test_priority_constants();
    test_priority_api_exists();
    
    printf("\nAll tests passed!\n");
    return 0;
}
