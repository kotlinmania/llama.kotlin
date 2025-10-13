// SPDX-License-Identifier: BSD-3-Clause
/* example_stackless.c - Demonstration of stackless arena coroutines
 *
 * This shows how to write producer/consumer coroutines using the
 * stackless API with BizTalk-style ticket-based coordination.
 */
#include "kcoro_stackless.h"
#include "koro_sched_stackless.h"
#include "kcoro_token_kernel.h"
#include <stdio.h>
#include <stdint.h>

/* User-defined local variables for producer coroutine */
struct producer_locals {
    int i;              /* Loop counter */
    int data;           /* Data to send */
    struct kc_chan* ch; /* Channel reference */
};

/* Producer coroutine function.
 * Sends 5 messages through the channel. */
void* producer_step(koro_cont_t* k)
{
    struct producer_locals* local = (struct producer_locals*)k->user_data;
    
    KORO_BEGIN(k);
    
    printf("Producer: Starting\n");
    
    for (local->i = 0; local->i < 5; local->i++) {
        local->data = local->i * 100;
        printf("Producer: Sending %d...\n", local->data);
        
        /* This suspends if no receiver is ready */
        KORO_SEND(k, local->ch, &local->data, sizeof(local->data));
        
        /* Resumed here after send completes */
        if (k->last_park_result != 0) {
            printf("Producer: Send failed with error %d\n", k->last_park_result);
            break;
        }
        printf("Producer: Sent %d successfully\n", local->data);
    }
    
    printf("Producer: Done\n");
    
    KORO_END(k);
}

/* User-defined local variables for consumer coroutine */
struct consumer_locals {
    int i;              /* Loop counter */
    int* recv_data;     /* Received data pointer */
    struct kc_chan* ch; /* Channel reference */
};

/* Consumer coroutine function.
 * Receives 5 messages from the channel. */
void* consumer_step(koro_cont_t* k)
{
    struct consumer_locals* local = (struct consumer_locals*)k->user_data;
    
    KORO_BEGIN(k);
    
    printf("Consumer: Starting\n");
    
    for (local->i = 0; local->i < 5; local->i++) {
        printf("Consumer: Waiting to receive...\n");
        
        /* This suspends if no sender is ready */
        KORO_RECV(k, local->ch);
        
        /* Resumed here with data in k->arena_payload */
        if (k->last_park_result != 0) {
            printf("Consumer: Receive failed with error %d\n", k->last_park_result);
            break;
        }
        
        local->recv_data = (int*)k->arena_payload;
        printf("Consumer: Received %d\n", *local->recv_data);
    }
    
    printf("Consumer: Done\n");
    
    KORO_END(k);
}

/* Main program */
int main(void)
{
    printf("=== Stackless Arena Coroutine Example ===\n\n");
    
    /* Initialize the token kernel (arena callback system) */
    if (kc_token_kernel_global_init() != 0) {
        fprintf(stderr, "Failed to initialize token kernel\n");
        return 1;
    }
    
    /* Initialize the stackless scheduler */
    if (koro_sched_init() != 0) {
        fprintf(stderr, "Failed to initialize scheduler\n");
        return 1;
    }
    
    /* Create a shared channel for producer/consumer coordination.
     * In a real implementation, this would use kc_chan_make().
     * For now, we'll pass NULL and document the integration point. */
    struct kc_chan* shared_channel = NULL; /* TODO: kc_chan_make() */
    
    /* Spawn producer coroutine */
    printf("Main: Spawning producer...\n");
    if (koro_go(producer_step, shared_channel, sizeof(struct producer_locals)) != 0) {
        fprintf(stderr, "Failed to spawn producer\n");
        return 1;
    }
    
    /* Spawn consumer coroutine */
    printf("Main: Spawning consumer...\n");
    if (koro_go(consumer_step, shared_channel, sizeof(struct consumer_locals)) != 0) {
        fprintf(stderr, "Failed to spawn consumer\n");
        return 1;
    }
    
    /* Run the scheduler event loop.
     * This executes both coroutines on our single thread's stack
     * until both complete. No assembly, no stack switching. */
    printf("Main: Running scheduler...\n\n");
    koro_run();
    
    printf("\n=== All coroutines completed ===\n");
    
    /* Cleanup */
    kc_token_kernel_global_shutdown();
    
    return 0;
}
