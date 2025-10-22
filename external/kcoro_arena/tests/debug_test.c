#include "kcoro_stackless.h"
#include "koro_sched_stackless.h"
#include "kcoro_token_kernel.h"
#include "kc_chan_api.h"
#include <stdio.h>
#include <stdint.h>

int main(void) {
    printf("Creating channel...\n");
    struct kc_chan* ch = kc_chan_make_stackless(KC_CHAN_RENDEZVOUS, 0);
    if (!ch) {
        fprintf(stderr, "Failed to create channel\n");
        return 1;
    }
    printf("Channel created: %p\n", (void*)ch);
    
    printf("Creating continuation...\n");
    koro_cont_t* k = koro_cont_create(NULL, NULL, 0);
    if (!k) {
        fprintf(stderr, "Failed to create continuation\n");
        return 1;
    }
    printf("Continuation created: %p\n", (void*)k);
    
    int data = 42;
    printf("Calling send (should suspend)...\n");
    int result = kc_chan_send_stackless(k, ch, &data, sizeof(data));
    printf("Send returned: %d, last_park_result: %d\n", result, k->last_park_result);
    
    koro_cont_destroy(k);
    kc_chan_close_stackless(ch);
    kc_chan_destroy_stackless(ch);
    
    return 0;
}
