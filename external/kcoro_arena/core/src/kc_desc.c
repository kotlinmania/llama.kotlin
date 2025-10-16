// SPDX-License-Identifier: BSD-3-Clause
#include "kcoro_desc.h"
#include "kcoro_desc_metrics.h"
#include "kc_arena.h"
#include "kcoro_config_runtime.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <limits.h>

#define KC_DESC_BUCKETS 256u

typedef struct kc_desc_entry kc_desc_entry;

struct kc_desc_entry {
    kc_desc_entry *next;
    kc_desc_id     id;
    void          *data;
    size_t         len;
    unsigned       flags;
    unsigned       arena_id;
    size_t         arena_len;
    int            owns_allocation;
    atomic_uint    refcount;
};

typedef struct kc_desc_bucket {
    pthread_mutex_t mu;
    kc_desc_entry  *head;
} kc_desc_bucket;

static struct {
    kc_desc_bucket buckets[KC_DESC_BUCKETS];
    atomic_uint_fast64_t next_id;
    atomic_int initialized;
    
    /* Metrics (atomic for thread-safety) */
    atomic_uint_fast64_t metrics_alias_created;
    atomic_uint_fast64_t metrics_copy_created;
    atomic_uint_fast64_t metrics_retain;
    atomic_uint_fast64_t metrics_release;
    atomic_uint_fast64_t metrics_evicts;
    atomic_uint_fast64_t metrics_lookup_hits;
    atomic_uint_fast64_t metrics_lookup_misses;
    int                  metrics_enabled;
} g_desc = {
    .next_id = ATOMIC_VAR_INIT(1),
    .initialized = ATOMIC_VAR_INIT(0),
    .metrics_alias_created = ATOMIC_VAR_INIT(0),
    .metrics_copy_created = ATOMIC_VAR_INIT(0),
    .metrics_retain = ATOMIC_VAR_INIT(0),
    .metrics_release = ATOMIC_VAR_INIT(0),
    .metrics_evicts = ATOMIC_VAR_INIT(0),
    .metrics_lookup_hits = ATOMIC_VAR_INIT(0),
    .metrics_lookup_misses = ATOMIC_VAR_INIT(0),
    .metrics_enabled = 0,
};

static size_t bucket_index(kc_desc_id id)
{
    return (size_t)(id & (KC_DESC_BUCKETS - 1));
}

int kc_desc_global_init(void)
{
    int expected = 0;
    if (!atomic_compare_exchange_strong(&g_desc.initialized, &expected, 1)) {
        return 0; // already initialized
    }
    for (size_t i = 0; i < KC_DESC_BUCKETS; ++i) {
        pthread_mutex_init(&g_desc.buckets[i].mu, NULL);
        g_desc.buckets[i].head = NULL;
    }
    atomic_store(&g_desc.next_id, 1);
    
    /* Initialize metrics based on config */
    const struct kc_runtime_config *cfg = kc_runtime_config_get();
    g_desc.metrics_enabled = cfg ? cfg->chan_metrics_auto_enable : 0;
    atomic_store(&g_desc.metrics_alias_created, 0);
    atomic_store(&g_desc.metrics_copy_created, 0);
    atomic_store(&g_desc.metrics_retain, 0);
    atomic_store(&g_desc.metrics_release, 0);
    atomic_store(&g_desc.metrics_evicts, 0);
    atomic_store(&g_desc.metrics_lookup_hits, 0);
    atomic_store(&g_desc.metrics_lookup_misses, 0);
    
    atomic_store(&g_desc.initialized, 2);
    return 0;
}

static void entry_destroy(kc_desc_entry *entry)
{
    if (!entry) return;
    if (entry->owns_allocation && entry->data) {
        kc_arena_free(entry->arena_id, entry->data, entry->arena_len);
    } else if (!(entry->flags & KC_DESC_FLAG_ALIAS) && entry->data) {
        free(entry->data);
    }
    free(entry);
}

void kc_desc_global_shutdown(void)
{
    if (atomic_load(&g_desc.initialized) != 2) return;
    for (size_t i = 0; i < KC_DESC_BUCKETS; ++i) {
        pthread_mutex_lock(&g_desc.buckets[i].mu);
        kc_desc_entry *cur = g_desc.buckets[i].head;
        while (cur) {
            kc_desc_entry *next = cur->next;
            entry_destroy(cur);
            cur = next;
        }
        g_desc.buckets[i].head = NULL;
        pthread_mutex_unlock(&g_desc.buckets[i].mu);
        pthread_mutex_destroy(&g_desc.buckets[i].mu);
    }
    atomic_store(&g_desc.initialized, 0);
}

static kc_desc_entry *kc_desc_insert(void *data, size_t len, unsigned flags,
                                    unsigned arena_id, size_t arena_len, int owns)
{
    kc_desc_entry *entry = calloc(1, sizeof(*entry));
    if (!entry) return NULL;
    entry->data = data;
    entry->len = len;
    entry->flags = flags;
    entry->arena_id = arena_id;
    entry->arena_len = arena_len;
    entry->owns_allocation = owns;
    atomic_init(&entry->refcount, 1);
    entry->id = atomic_fetch_add(&g_desc.next_id, 1);

    size_t idx = bucket_index(entry->id);
    kc_desc_bucket *bucket = &g_desc.buckets[idx];
    pthread_mutex_lock(&bucket->mu);
    entry->next = bucket->head;
    bucket->head = entry;
    pthread_mutex_unlock(&bucket->mu);
    return entry;
}

kc_desc_id kc_desc_make_alias(void *ptr, size_t len)
{
    if (atomic_load(&g_desc.initialized) != 2) {
        if (kc_desc_global_init() != 0) return 0;
    }
    kc_desc_entry *entry = kc_desc_insert(ptr, len, KC_DESC_FLAG_ALIAS, UINT_MAX, 0, 0);
    if (entry && g_desc.metrics_enabled) {
        atomic_fetch_add(&g_desc.metrics_alias_created, 1);
    }
    return entry ? entry->id : 0;
}

kc_desc_id kc_desc_make_copy(const void *src, size_t len)
{
    if (atomic_load(&g_desc.initialized) != 2) {
        if (kc_desc_global_init() != 0) return 0;
    }
    if (len && kc_arena_create(0, 0) == -EEXIST) {
        /* already active */
    }
    void *copy = len ? kc_arena_alloc(0, len) : NULL;
    if (len && !copy) return 0;
    if (len && src) memcpy(copy, src, len);
    kc_desc_entry *entry = kc_desc_insert(copy, len, 0, 0, len, 1);
    if (!entry) {
        kc_arena_free(0, copy, len);
        return 0;
    }
    if (g_desc.metrics_enabled) {
        atomic_fetch_add(&g_desc.metrics_copy_created, 1);
    }
    return entry->id;
}

void kc_desc_retain(kc_desc_id id)
{
    if (!id) return;
    kc_desc_bucket *bucket = &g_desc.buckets[bucket_index(id)];
    pthread_mutex_lock(&bucket->mu);
    kc_desc_entry *cur = bucket->head;
    int found = 0;
    while (cur) {
        if (cur->id == id) {
            atomic_fetch_add_explicit(&cur->refcount, 1, memory_order_relaxed);
            found = 1;
            break;
        }
        cur = cur->next;
    }
    pthread_mutex_unlock(&bucket->mu);
    
    if (g_desc.metrics_enabled) {
        atomic_fetch_add(&g_desc.metrics_retain, 1);
        if (found) {
            atomic_fetch_add(&g_desc.metrics_lookup_hits, 1);
        } else {
            atomic_fetch_add(&g_desc.metrics_lookup_misses, 1);
        }
    }
}

static void kc_desc_remove_locked(kc_desc_bucket *bucket, kc_desc_entry *target)
{
    kc_desc_entry *prev = NULL;
    kc_desc_entry *cur = bucket->head;
    while (cur) {
        if (cur == target) {
            if (prev) prev->next = cur->next;
            else bucket->head = cur->next;
            cur->next = NULL;
            entry_destroy(cur);
            return;
        }
        prev = cur;
        cur = cur->next;
    }
}

void kc_desc_release(kc_desc_id id)
{
    if (!id) return;
    size_t idx = bucket_index(id);
    kc_desc_bucket *bucket = &g_desc.buckets[idx];
    pthread_mutex_lock(&bucket->mu);
    kc_desc_entry *cur = bucket->head;
    int found = 0;
    int evicted = 0;
    while (cur) {
        if (cur->id == id) {
            found = 1;
            unsigned prev = atomic_fetch_sub_explicit(&cur->refcount, 1, memory_order_acq_rel);
            if (prev == 1) {
                kc_desc_remove_locked(bucket, cur);
                evicted = 1;
            }
            pthread_mutex_unlock(&bucket->mu);
            
            if (g_desc.metrics_enabled) {
                atomic_fetch_add(&g_desc.metrics_release, 1);
                if (evicted) {
                    atomic_fetch_add(&g_desc.metrics_evicts, 1);
                }
                atomic_fetch_add(&g_desc.metrics_lookup_hits, 1);
            }
            return;
        }
        cur = cur->next;
    }
    pthread_mutex_unlock(&bucket->mu);
    
    if (g_desc.metrics_enabled) {
        atomic_fetch_add(&g_desc.metrics_release, 1);
        if (!found) {
            atomic_fetch_add(&g_desc.metrics_lookup_misses, 1);
        }
    }
}

int kc_desc_payload(kc_desc_id id, kc_payload *out_payload)
{
    if (!out_payload) return -EINVAL;
    kc_desc_bucket *bucket = &g_desc.buckets[bucket_index(id)];
    pthread_mutex_lock(&bucket->mu);
    kc_desc_entry *cur = bucket->head;
    while (cur) {
        if (cur->id == id) {
            out_payload->ptr = cur->data;
            out_payload->len = cur->len;
            out_payload->status = 0;
            out_payload->desc_id = id;
            pthread_mutex_unlock(&bucket->mu);
            
            if (g_desc.metrics_enabled) {
                atomic_fetch_add(&g_desc.metrics_lookup_hits, 1);
            }
            return 0;
        }
        cur = cur->next;
    }
    pthread_mutex_unlock(&bucket->mu);
    
    if (g_desc.metrics_enabled) {
        atomic_fetch_add(&g_desc.metrics_lookup_misses, 1);
    }
    return -ENOENT;
}

int kc_desc_get_metrics(kc_desc_metrics *out)
{
    if (!out) return -EINVAL;
    
    out->alias_created_total = atomic_load_explicit(&g_desc.metrics_alias_created, memory_order_relaxed);
    out->copy_created_total = atomic_load_explicit(&g_desc.metrics_copy_created, memory_order_relaxed);
    out->retain_total = atomic_load_explicit(&g_desc.metrics_retain, memory_order_relaxed);
    out->release_total = atomic_load_explicit(&g_desc.metrics_release, memory_order_relaxed);
    out->descriptor_evicts = atomic_load_explicit(&g_desc.metrics_evicts, memory_order_relaxed);
    out->lookup_hits = atomic_load_explicit(&g_desc.metrics_lookup_hits, memory_order_relaxed);
    out->lookup_misses = atomic_load_explicit(&g_desc.metrics_lookup_misses, memory_order_relaxed);
    
    return 0;
}

void kc_desc_reset_metrics(void)
{
    atomic_store_explicit(&g_desc.metrics_alias_created, 0, memory_order_relaxed);
    atomic_store_explicit(&g_desc.metrics_copy_created, 0, memory_order_relaxed);
    atomic_store_explicit(&g_desc.metrics_retain, 0, memory_order_relaxed);
    atomic_store_explicit(&g_desc.metrics_release, 0, memory_order_relaxed);
    atomic_store_explicit(&g_desc.metrics_evicts, 0, memory_order_relaxed);
    atomic_store_explicit(&g_desc.metrics_lookup_hits, 0, memory_order_relaxed);
    atomic_store_explicit(&g_desc.metrics_lookup_misses, 0, memory_order_relaxed);
}
