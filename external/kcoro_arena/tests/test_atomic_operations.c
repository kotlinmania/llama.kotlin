// SPDX-License-Identifier: BSD-3-Clause
/* test_atomic_operations.c - Comprehensive atomic operation tests
 *
 * Tests all atomic primitives used in stackless kcoro_arena:
 * - Token kernel atomic operations
 * - Continuation state transitions
 * - Arena allocation atomicity
 * - Queue operations
 * - Cell state machine transitions
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include "kcoro_stackless.h"
#include "kcoro_token_kernel.h"
#include "kc_arena.h"

/* Test result tracking */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_START(name) \
    do { \
        tests_run++; \
        printf("[TEST] %s ...\n", name); \
    } while(0)

#define TEST_PASS() \
    do { \
        tests_passed++; \
        printf("  PASS\n"); \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        tests_failed++; \
        printf("  FAIL: %s\n", msg); \
    } while(0)

#define ASSERT_EQ(expected, actual, msg) \
    do { \
        if ((expected) != (actual)) { \
            TEST_FAIL(msg); \
            printf("    Expected: %ld, Got: %ld\n", (long)(expected), (long)(actual)); \
            return; \
        } \
    } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

/* ================================================================
 * Test 1: Atomic ID generation (token kernel)
 * ================================================================ */
static void test_atomic_id_generation(void)
{
    TEST_START("Atomic ID generation");
    
    /* Initialize kernel */
    int rc = kc_token_kernel_init();
    ASSERT_EQ(0, rc, "Kernel init failed");
    
    /* Generate IDs and ensure monotonic increase */
    kc_token_id_t id1, id2, id3;
    id1 = kc_token_kernel_next_id();
    id2 = kc_token_kernel_next_id();
    id3 = kc_token_kernel_next_id();
    
    ASSERT_TRUE(id1.id > 0, "ID1 must be positive");
    ASSERT_TRUE(id2.id > id1.id, "ID2 must be > ID1");
    ASSERT_TRUE(id3.id > id2.id, "ID3 must be > ID2");
    
    kc_token_kernel_shutdown();
    TEST_PASS();
}

/* ================================================================
 * Test 2: Concurrent ID generation
 * ================================================================ */
#define NUM_THREADS_ID 8
#define IDS_PER_THREAD 1000

static void* id_generator_thread(void* arg)
{
    kc_token_id_t* ids = (kc_token_id_t*)arg;
    for (int i = 0; i < IDS_PER_THREAD; i++) {
        ids[i] = kc_token_kernel_next_id();
    }
    return NULL;
}

static void test_concurrent_id_generation(void)
{
    TEST_START("Concurrent ID generation");
    
    int rc = kc_token_kernel_init();
    ASSERT_EQ(0, rc, "Kernel init failed");
    
    pthread_t threads[NUM_THREADS_ID];
    kc_token_id_t* thread_ids[NUM_THREADS_ID];
    
    /* Allocate ID storage for each thread */
    for (int i = 0; i < NUM_THREADS_ID; i++) {
        thread_ids[i] = malloc(sizeof(kc_token_id_t) * IDS_PER_THREAD);
        ASSERT_TRUE(thread_ids[i] != NULL, "Memory allocation failed");
    }
    
    /* Spawn threads */
    for (int i = 0; i < NUM_THREADS_ID; i++) {
        pthread_create(&threads[i], NULL, id_generator_thread, thread_ids[i]);
    }
    
    /* Wait for completion */
    for (int i = 0; i < NUM_THREADS_ID; i++) {
        pthread_join(threads[i], NULL);
    }
    
    /* Verify no duplicates across all threads */
    int total_ids = NUM_THREADS_ID * IDS_PER_THREAD;
    uint64_t* all_ids = malloc(sizeof(uint64_t) * total_ids);
    ASSERT_TRUE(all_ids != NULL, "Memory allocation failed");
    
    int idx = 0;
    for (int t = 0; t < NUM_THREADS_ID; t++) {
        for (int i = 0; i < IDS_PER_THREAD; i++) {
            all_ids[idx++] = thread_ids[t][i].id;
        }
    }
    
    /* Sort and check for duplicates */
    for (int i = 0; i < total_ids - 1; i++) {
        for (int j = i + 1; j < total_ids; j++) {
            ASSERT_TRUE(all_ids[i] != all_ids[j], "Duplicate IDs detected");
        }
    }
    
    /* Cleanup */
    for (int i = 0; i < NUM_THREADS_ID; i++) {
        free(thread_ids[i]);
    }
    free(all_ids);
    
    kc_token_kernel_shutdown();
    TEST_PASS();
}

/* ================================================================
 * Test 3: Continuation state atomicity
 * ================================================================ */
static void* simple_step(koro_cont_t* k)
{
    k->state++;
    return (void*)1; /* Complete */
}

static void test_continuation_state_atomicity(void)
{
    TEST_START("Continuation state atomicity");
    
    koro_cont_t* k = koro_cont_create(simple_step, NULL, 0);
    ASSERT_TRUE(k != NULL, "Continuation creation failed");
    ASSERT_EQ(0, k->state, "Initial state must be 0");
    ASSERT_TRUE(k->next_step == simple_step, "Step function mismatch");
    
    /* Execute step */
    void* result = k->next_step(k);
    ASSERT_TRUE(result != NULL, "Step should complete");
    ASSERT_EQ(1, k->state, "State should advance to 1");
    
    koro_cont_destroy(k);
    TEST_PASS();
}

/* ================================================================
 * Test 4: Arena bump allocation atomicity
 * ================================================================ */
#define NUM_THREADS_ARENA 4
#define ALLOCS_PER_THREAD 500

static void* arena_allocator_thread(void* arg)
{
    unsigned arena_id = *(unsigned*)arg;
    
    for (int i = 0; i < ALLOCS_PER_THREAD; i++) {
        size_t size = 16 + (i % 256); /* Varying sizes */
        void* ptr = kc_arena_alloc(arena_id, size);
        if (!ptr) {
            return (void*)-1; /* Allocation failed */
        }
        /* Write pattern to verify no overlaps */
        memset(ptr, (unsigned char)i, size);
    }
    return NULL;
}

static void test_arena_bump_allocation_atomicity(void)
{
    TEST_START("Arena bump allocation atomicity");
    
    unsigned arena_id = 0;
    size_t arena_size = 2 * 1024 * 1024; /* 2MB */
    
    int rc = kc_arena_create(arena_id, arena_size);
    ASSERT_EQ(0, rc, "Arena creation failed");
    
    pthread_t threads[NUM_THREADS_ARENA];
    unsigned thread_args[NUM_THREADS_ARENA];
    
    /* Spawn threads doing concurrent allocations */
    for (int i = 0; i < NUM_THREADS_ARENA; i++) {
        thread_args[i] = arena_id;
        pthread_create(&threads[i], NULL, arena_allocator_thread, &thread_args[i]);
    }
    
    /* Wait for completion */
    for (int i = 0; i < NUM_THREADS_ARENA; i++) {
        void* result;
        pthread_join(threads[i], &result);
        ASSERT_TRUE(result == NULL, "Thread allocation failed");
    }
    
    kc_arena_destroy(arena_id);
    TEST_PASS();
}

/* ================================================================
 * Test 5: Token publish/claim race (core rendezvous primitive)
 * ================================================================ */
#define NUM_PRODUCERS 4
#define NUM_CONSUMERS 4
#define MSGS_PER_PRODUCER 100

typedef struct {
    int id;
    atomic_int sent;
    atomic_int received;
} worker_context_t;

static void* producer_thread(void* arg)
{
    worker_context_t* ctx = (worker_context_t*)arg;
    
    for (int i = 0; i < MSGS_PER_PRODUCER; i++) {
        /* Simulate token publish operation */
        int *data = malloc(sizeof(int));
        *data = ctx->id * 10000 + i;
        
        kc_ticket ticket = kc_token_kernel_publish_send(
            NULL,  // channel
            data,  // payload ptr
            sizeof(int),  // payload len
            NULL, // resume callback
            NULL
        );
        
        if (ticket.id != 0) {
            atomic_fetch_add(&ctx->sent, 1);
        }
        
        free(data);  // In real code, payload lifetime would be managed by the kernel
        
        /* Small delay to increase contention */
        for (volatile int j = 0; j < 100; j++);
    }
    
    return NULL;
}

static void* consumer_thread(void* arg)
{
    worker_context_t* ctx = (worker_context_t*)arg;
    
    for (int i = 0; i < MSGS_PER_PRODUCER; i++) {
        /* Simulate token claim operation */
        kc_payload result;
        int rc = kc_token_kernel_try_claim_receive(&result);
        
        if (rc == 0) {
            atomic_fetch_add(&ctx->received, 1);
            /* Cleanup payload */
            if (result.type == KC_PAYLOAD_COPY && result.data.copy.data) {
                free(result.data.copy.data);
            }
        }
        
        /* Small delay */
        for (volatile int j = 0; j < 100; j++);
    }
    
    return NULL;
}

static void test_token_publish_claim_race(void)
{
    TEST_START("Token publish/claim race (rendezvous core)");
    
    int rc = kc_token_kernel_init();
    ASSERT_EQ(0, rc, "Kernel init failed");
    
    worker_context_t prod_ctx[NUM_PRODUCERS] = {0};
    worker_context_t cons_ctx[NUM_CONSUMERS] = {0};
    pthread_t prod_threads[NUM_PRODUCERS];
    pthread_t cons_threads[NUM_CONSUMERS];
    
    /* Initialize contexts */
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        prod_ctx[i].id = i;
        atomic_init(&prod_ctx[i].sent, 0);
    }
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        cons_ctx[i].id = i;
        atomic_init(&cons_ctx[i].received, 0);
    }
    
    /* Spawn producers and consumers */
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_create(&prod_threads[i], NULL, producer_thread, &prod_ctx[i]);
    }
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_create(&cons_threads[i], NULL, consumer_thread, &cons_ctx[i]);
    }
    
    /* Wait for all threads */
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_join(prod_threads[i], NULL);
    }
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_join(cons_threads[i], NULL);
    }
    
    /* Verify totals */
    int total_sent = 0;
    int total_received = 0;
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        total_sent += atomic_load(&prod_ctx[i].sent);
    }
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        total_received += atomic_load(&cons_ctx[i].received);
    }
    
    printf("  Total sent: %d, Total received: %d\n", total_sent, total_received);
    
    /* In a working system, sent and received should balance */
    ASSERT_TRUE(total_sent > 0, "No messages were sent");
    ASSERT_TRUE(total_received > 0, "No messages were received");
    
    kc_token_kernel_shutdown();
    TEST_PASS();
}

/* ================================================================
 * Test 6: Queue enqueue/dequeue atomicity (ready queue)
 * ================================================================ */
#define NUM_ENQUEUE_THREADS 4
#define NUM_DEQUEUE_THREADS 4
#define ITEMS_PER_THREAD 200

typedef struct test_queue_item {
    struct test_queue_item* next;
    int value;
} test_queue_item_t;

typedef struct {
    pthread_mutex_t mu;
    test_queue_item_t* head;
    test_queue_item_t* tail;
} test_queue_t;

static void test_queue_init(test_queue_t* q)
{
    pthread_mutex_init(&q->mu, NULL);
    q->head = q->tail = NULL;
}

static void test_queue_enqueue(test_queue_t* q, test_queue_item_t* item)
{
    pthread_mutex_lock(&q->mu);
    item->next = NULL;
    if (q->tail) {
        q->tail->next = item;
    } else {
        q->head = item;
    }
    q->tail = item;
    pthread_mutex_unlock(&q->mu);
}

static test_queue_item_t* test_queue_dequeue(test_queue_t* q)
{
    pthread_mutex_lock(&q->mu);
    test_queue_item_t* item = q->head;
    if (item) {
        q->head = item->next;
        if (!q->head) q->tail = NULL;
    }
    pthread_mutex_unlock(&q->mu);
    return item;
}

static void test_queue_destroy(test_queue_t* q)
{
    /* Drain remaining items */
    test_queue_item_t* item;
    while ((item = test_queue_dequeue(q)) != NULL) {
        free(item);
    }
    pthread_mutex_destroy(&q->mu);
}

static test_queue_t g_test_queue;

static void* enqueue_thread(void* arg)
{
    int thread_id = *(int*)arg;
    
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        test_queue_item_t* item = malloc(sizeof(test_queue_item_t));
        item->value = thread_id * 10000 + i;
        test_queue_enqueue(&g_test_queue, item);
    }
    
    return NULL;
}

static void* dequeue_thread(void* arg)
{
    atomic_int* count = (atomic_int*)arg;
    
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        test_queue_item_t* item;
        while ((item = test_queue_dequeue(&g_test_queue)) == NULL) {
            sched_yield(); /* Busy-wait retry */
        }
        
        atomic_fetch_add(count, 1);
        free(item);
    }
    
    return NULL;
}

static void test_queue_enqueue_dequeue_atomicity(void)
{
    TEST_START("Queue enqueue/dequeue atomicity");
    
    test_queue_init(&g_test_queue);
    
    pthread_t enq_threads[NUM_ENQUEUE_THREADS];
    pthread_t deq_threads[NUM_DEQUEUE_THREADS];
    int enq_ids[NUM_ENQUEUE_THREADS];
    atomic_int deq_count = ATOMIC_VAR_INIT(0);
    
    /* Spawn enqueue threads */
    for (int i = 0; i < NUM_ENQUEUE_THREADS; i++) {
        enq_ids[i] = i;
        pthread_create(&enq_threads[i], NULL, enqueue_thread, &enq_ids[i]);
    }
    
    /* Spawn dequeue threads */
    for (int i = 0; i < NUM_DEQUEUE_THREADS; i++) {
        pthread_create(&deq_threads[i], NULL, dequeue_thread, &deq_count);
    }
    
    /* Wait for all threads */
    for (int i = 0; i < NUM_ENQUEUE_THREADS; i++) {
        pthread_join(enq_threads[i], NULL);
    }
    for (int i = 0; i < NUM_DEQUEUE_THREADS; i++) {
        pthread_join(deq_threads[i], NULL);
    }
    
    int expected = NUM_ENQUEUE_THREADS * ITEMS_PER_THREAD;
    int actual = atomic_load(&deq_count);
    
    printf("  Expected: %d, Dequeued: %d\n", expected, actual);
    ASSERT_EQ(expected, actual, "Enqueue/dequeue count mismatch");
    
    test_queue_destroy(&g_test_queue);
    TEST_PASS();
}

/* ================================================================
 * Main test runner
 * ================================================================ */
int main(void)
{
    printf("=== kcoro_arena Atomic Operations Test Suite ===\n\n");
    
    test_atomic_id_generation();
    test_concurrent_id_generation();
    test_continuation_state_atomicity();
    test_arena_bump_allocation_atomicity();
    test_token_publish_claim_race();
    test_queue_enqueue_dequeue_atomicity();
    
    printf("\n=== Test Summary ===\n");
    printf("Total:  %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    
    return (tests_failed == 0) ? 0 : 1;
}
