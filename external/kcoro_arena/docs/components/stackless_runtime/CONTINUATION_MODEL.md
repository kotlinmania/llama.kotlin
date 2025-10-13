# Continuation Model

## The `koro_cont` Structure

The continuation record is the heart of the stackless runtime. It's a heap-allocated struct that replaces the traditional call stack.

```c
struct koro_cont {
    koro_step_fn next_step;      // Function pointer to next execution step
    int state;                    // Current state in the coroutine's state machine
    void* user_data;              // Pointer to user's local variables
    struct koro_cont* parent;     // For structured concurrency (optional)
    int flags;                    // Status flags (RUNNING, SUSPENDED, COMPLETED)
    // ... additional scheduler metadata ...
};
```

## Fields Explained

### `next_step`

A function pointer with signature:

```c
typedef void (*koro_step_fn)(struct koro_cont* k);
```

This points to the next "chunk" of code to execute when this continuation is resumed. The scheduler simply calls `k->next_step(k)` to dispatch the coroutine.

### `state`

An integer representing the current execution state. Used in conjunction with a `switch` statement to jump to the correct resumption point:

```c
void my_coroutine_step(struct koro_cont* k) {
    switch (k->state) {
        case 0:  // Initial entry
            // ... code ...
            k->state = 1;
            return;  // Suspend

        case 1:  // Resume here after first suspension
            // ... more code ...
            k->state = 2;
            return;  // Suspend again

        case 2:  // Resume here after second suspension
            // ... final code ...
            k->state = DONE;
            return;  // Complete
    }
}
```

This is the "Duff's device" pattern used in Protothreads.

### `user_data`

A `void*` pointing to a user-defined struct containing the coroutine's local variables. Example:

```c
struct my_locals {
    int counter;
    char buffer[256];
    struct some_handle* resource;
};

// In koro_go():
struct my_locals* locals = malloc(sizeof(struct my_locals));
k->user_data = locals;
```

Any variable that needs to persist across suspension points must be stored here.

### `parent`

Optional pointer to a parent continuation for structured concurrency. If a child coroutine completes or fails, it can notify the parent. (Not currently used in the golden path.)

### `flags`

Bit flags for scheduler state:

```c
#define KORO_RUNNING    (1 << 0)  // Currently executing
#define KORO_SUSPENDED  (1 << 1)  // Waiting for an event
#define KORO_COMPLETED  (1 << 2)  // Finished execution
#define KORO_CANCELLED  (1 << 3)  // Cancelled by user/parent
```

## Lifecycle of a Continuation

### 1. Allocation

```c
struct koro_cont* k = malloc(sizeof(struct koro_cont));
k->next_step = my_coroutine_entry;
k->state = 0;
k->user_data = malloc(sizeof(struct my_locals));
k->flags = KORO_SUSPENDED;
```

### 2. Enqueueing

The scheduler maintains a ready queue:

```c
enqueue_runnable(k);  // Add to scheduler's work queue
```

### 3. Execution

Scheduler dequeues and dispatches:

```c
struct koro_cont* k = dequeue_next_runnable();
k->flags |= KORO_RUNNING;
k->next_step(k);  // Execute one "step"
k->flags &= ~KORO_RUNNING;
```

### 4. Suspension

If the step function `return`s without marking itself as `COMPLETED`, it's automatically re-enqueued (or left off the queue if it's waiting for an external event).

### 5. Completion

```c
void final_step(struct koro_cont* k) {
    // Cleanup
    free(k->user_data);
    k->flags |= KORO_COMPLETED;
    // Scheduler will free `k` itself
}
```

## Memory Management

### Allocation

- **Continuation struct**: Allocated once at coroutine spawn
- **User data**: Allocated once, freed at completion
- Total overhead: ~200-500 bytes per coroutine (depending on user locals)

### Deallocation

The scheduler automatically frees completed continuations after they return with `KORO_COMPLETED` flag set.

### Reference Counting (Future)

For more complex patterns (parent/child relationships), continuations could use reference counting:

```c
k->refcount = 1;  // Scheduler holds one reference
k->parent_ref = parent_cont;  // Child holds reference to parent
```

(Not implemented in current golden path.)

## Comparison to Traditional Stacks

| Aspect | Traditional Stack | Continuation Record |
|--------|-------------------|---------------------|
| Allocation | `mmap` (64 KB default) | `malloc` (~200 bytes) |
| Local variables | On stack (automatic) | In `user_data` (manual) |
| Function calls | Native call instruction | Function pointer dispatch |
| Return | `ret` instruction | Explicit `return` to scheduler |
| Context switch | Save/restore 20+ registers | Update `next_step` pointer |

## Example: Producer Coroutine

```c
struct producer_locals {
    int ticket;
    int i;
};

void producer_step(struct koro_cont* k) {
    struct producer_locals* locals = k->user_data;

    switch (k->state) {
        case 0:  // Initialize
            locals->i = 0;
            k->state = 1;
            // Fall through

        case 1:  // Loop body
            if (locals->i >= 5) {
                k->state = DONE;
                k->flags |= KORO_COMPLETED;
                return;
            }

            // Suspend at send
            koro_send(k, locals->ticket, (void*)(intptr_t)locals->i);
            locals->i++;
            k->state = 1;  // Loop back
            return;  // Scheduler will re-enqueue us
    }
}
```

Compare to the user-facing macro version (which expands to the above):

```c
void producer(struct koro_cont* k, int ticket) {
    struct producer_locals* locals = k->user_data;

    KORO_BEGIN(k);

    for (locals->i = 0; locals->i < 5; locals->i++) {
        KORO_SEND(k, ticket, (void*)(intptr_t)locals->i);
    }

    KORO_END(k);
}
```

The macros hide the state machine boilerplate.

## Next Steps

- Read [SCHEDULER.md](./SCHEDULER.md) for how continuations are dispatched
- Read [MACROS.md](./MACROS.md) for the user-facing API that generates continuation code
- Read [../ffi_go/IMPLEMENTATION_SPEC.md](../ffi_go/IMPLEMENTATION_SPEC.md) for the `koro_go()` function that creates continuations
