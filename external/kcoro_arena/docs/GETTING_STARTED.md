# Getting Started with kcoro_arena

This guide walks you through building your first kcoro_arena application, from basic setup to production-ready patterns.

## Prerequisites

- **C Compiler**: GCC 4.9+ or Clang 3.5+
- **Operating System**: Linux (x86_64 or ARM64), macOS, or BSD
- **Build System**: Make or CMake
- **Optional**: Valgrind for leak checking

## Installation

### Building the Library

```bash
cd external/kcoro_arena/core
make clean all
```

This produces `build/lib/libkcoro_arena.a` and installs headers to `../include/`.

### Verifying the Build

```bash
cd ../tests
make all
./build/test_token_kernel_basic
```

You should see:
```
[test] Token kernel basic: PASS
```

## Your First Program

### Step 1: Include the Header

```c
#include "kcoro.h"
#include <stdio.h>
```

### Step 2: Write Stackless Coroutines

Stackless coroutines use CPS (Continuation-Passing Style) with macros for suspend points:

```c
// Producer: sends 5 integers
void producer_func(struct koro_cont* k, int ticket) {
    // Define local state that persists across suspensions
    struct producer_state {
        int i;
    } *state = k->user_data;
    
    KORO_BEGIN(k);  // Required: initializes state machine
    
    for (state->i = 0; state->i < 5; state->i++) {
        printf("Producer: sending %d\n", state->i);
        KORO_SEND(k, ticket, (void*)(intptr_t)state->i);
        printf("Producer: sent %d\n", state->i);
    }
    
    KORO_END(k);  // Required: cleanup
}

// Consumer: receives 5 integers
void consumer_func(struct koro_cont* k, int ticket) {
    struct consumer_state {
        int i;
        void* received_data;
    } *state = k->user_data;
    
    KORO_BEGIN(k);
    
    for (state->i = 0; state->i < 5; state->i++) {
        printf("Consumer: waiting for data\n");
        KORO_RECV(k, ticket, &state->received_data);
        int val = (int)(intptr_t)state->received_data;
        printf("Consumer: received %d\n", val);
    }
    
    KORO_END(k);
}
```

### Step 3: Dispatch and Run

```c
int main(void) {
    // 1. Initialize the runtime
    koro_init();
    
    // 2. Define a rendezvous ticket ID
    int ticket = 42;
    
    // 3. Launch coroutines
    koro_go(producer_func, (void*)(intptr_t)ticket);
    koro_go(consumer_func, (void*)(intptr_t)ticket);
    
    // 4. Enter the event loop (blocks until all coroutines finish)
    koro_run();
    
    // 5. Clean up
    koro_shutdown();
    
    printf("Done!\n");
    return 0;
}
```

### Step 4: Build and Run

```bash
gcc -std=c11 -I../include -o my_first_app my_first_app.c \
    -L../core/build/lib -lkcoro_arena -lpthread

./my_first_app
```

**Expected Output:**
```
Producer: sending 0
Consumer: waiting for data
Producer: sent 0
Consumer: received 0
Producer: sending 1
Consumer: waiting for data
Producer: sent 1
Consumer: received 1
...
Done!
```

## Understanding the Code

### The State Machine (`KORO_BEGIN`/`KORO_END`)

```c
KORO_BEGIN(k);
// ... user code ...
KORO_END(k);
```

Expands to:
```c
switch (k->state) {
    case 0:  // Initial entry
        // ... user code ...
        k->state = 1;  // Mark completion
}
```

### Suspension Points (`KORO_SEND`/`KORO_RECV`)

```c
KORO_SEND(k, ticket, data);
```

Expands to:
```c
k->state = __LINE__;  // Save resume point
if (koro_send_cps(k, ticket, data) == KORO_SUSPENDED) {
    return NULL;  // Suspend to scheduler
}
case __LINE__:;  // Resume here later
```

The scheduler calls your function, it runs until `return NULL`, then moves to the next coroutine. When a match occurs, the scheduler calls your function again, and `switch(k->state)` jumps to the saved line.

### Persistent State (`k->user_data`)

Local variables in regular C functions live on the stack and disappear when you `return`. Stackless coroutines need heap-allocated state:

```c
struct my_state {
    int counter;
    char buffer[256];
} *state = k->user_data;
```

This struct persists between suspensions. Initialize it in `KORO_BEGIN` if needed.

## Common Patterns

### Pattern 1: Multiple Consumers

```c
int main(void) {
    koro_init();
    
    int ticket = 100;
    koro_go(producer_func, (void*)(intptr_t)ticket);
    koro_go(consumer_func, (void*)(intptr_t)ticket);  // Consumer 1
    koro_go(consumer_func, (void*)(intptr_t)ticket);  // Consumer 2
    
    koro_run();
    koro_shutdown();
    return 0;
}
```

Both consumers will receive messages alternately (rendezvous is 1:1, so producers and consumers alternate).

### Pattern 2: Request/Reply

```c
void client(struct koro_cont* k, int request_ticket, int reply_ticket) {
    struct { int req_id; void* reply_data; } *state = k->user_data;
    
    KORO_BEGIN(k);
    
    state->req_id = 123;
    KORO_SEND(k, request_ticket, (void*)(intptr_t)state->req_id);
    KORO_RECV(k, reply_ticket, &state->reply_data);
    printf("Got reply: %d\n", (int)(intptr_t)state->reply_data);
    
    KORO_END(k);
}

void server(struct koro_cont* k, int request_ticket, int reply_ticket) {
    struct { void* req; int result; } *state = k->user_data;
    
    KORO_BEGIN(k);
    
    KORO_RECV(k, request_ticket, &state->req);
    state->result = (int)(intptr_t)state->req * 2;  // Process
    KORO_SEND(k, reply_ticket, (void*)(intptr_t)state->result);
    
    KORO_END(k);
}

int main(void) {
    koro_init();
    koro_go(client, (void*)0);  // pass both tickets via struct
    koro_go(server, (void*)0);
    koro_run();
    koro_shutdown();
    return 0;
}
```

(Simplified; in practice, pass ticket struct via user_data.)

### Pattern 3: Pipeline

```c
// Stage 1: Read data
void stage1(struct koro_cont* k, int out_ticket) {
    struct { int i; } *state = k->user_data;
    KORO_BEGIN(k);
    for (state->i = 0; state->i < 10; state->i++) {
        KORO_SEND(k, out_ticket, (void*)(intptr_t)state->i);
    }
    KORO_END(k);
}

// Stage 2: Transform
void stage2(struct koro_cont* k, int in_ticket, int out_ticket) {
    struct { void* data; int val; } *state = k->user_data;
    KORO_BEGIN(k);
    while (1) {
        KORO_RECV(k, in_ticket, &state->data);
        state->val = (int)(intptr_t)state->data * 2;
        KORO_SEND(k, out_ticket, (void*)(intptr_t)state->val);
    }
    KORO_END(k);
}

// Stage 3: Write
void stage3(struct koro_cont* k, int in_ticket) {
    struct { void* data; } *state = k->user_data;
    KORO_BEGIN(k);
    while (1) {
        KORO_RECV(k, in_ticket, &state->data);
        printf("Result: %d\n", (int)(intptr_t)state->data);
    }
    KORO_END(k);
}
```

## Zero-Copy with Descriptors

For large payloads, avoid `memcpy` overhead:

```c
void send_large(struct koro_cont* k, int ticket) {
    struct { struct kc_desc* desc; } *state = k->user_data;
    
    KORO_BEGIN(k);
    
    // Allocate zero-copy descriptor
    state->desc = kc_desc_new(1024 * 1024);  // 1MB
    memcpy(state->desc->data, large_data, 1024 * 1024);
    
    // Send just the pointer + length
    KORO_SEND_PTR(k, ticket, state->desc, 1024 * 1024);
    
    KORO_END(k);
}

void recv_large(struct koro_cont* k, int ticket) {
    struct { void* ptr; size_t len; } *state = k->user_data;
    
    KORO_BEGIN(k);
    
    KORO_RECV_PTR(k, ticket, &state->ptr, &state->len);
    
    // Use data directly (no copy)
    process_data(state->ptr, state->len);
    
    // Release when done
    kc_desc_release((struct kc_desc*)state->ptr);
    
    KORO_END(k);
}
```

## Debugging Tips

### Enable Tracing

```bash
export KCORO_TRACE=/tmp/kcoro_trace.log
./my_app
tail -f /tmp/kcoro_trace.log
```

### Check for Leaks

```bash
valgrind --leak-check=full --show-leak-kinds=all ./my_app
```

### Common Mistakes

**Mistake 1: Forgetting `KORO_BEGIN`/`KORO_END`**
```c
void bad_func(struct koro_cont* k, int ticket) {
    KORO_SEND(k, ticket, data);  // ❌ No KORO_BEGIN
}
```
Fix: Always wrap in `KORO_BEGIN` / `KORO_END`.

**Mistake 2: Using Stack Variables**
```c
void bad_func(struct koro_cont* k, int ticket) {
    int local_var = 42;  // ❌ Lost on suspend!
    KORO_BEGIN(k);
    KORO_SEND(k, ticket, &local_var);  // ❌ Dangling pointer
    KORO_END(k);
}
```
Fix: Use `k->user_data` for persistent state.

**Mistake 3: Blocking Calls**
```c
void bad_func(struct koro_cont* k, int ticket) {
    KORO_BEGIN(k);
    sleep(1);  // ❌ Blocks entire scheduler!
    KORO_END(k);
}
```
Fix: Use `KORO_DELAY` (not yet implemented) or restructure logic.

## Next Steps

- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)**: Porting existing code
- **[Components Overview](components/)**: Deep dive into each subsystem
- **[API Reference](components/ffi_go/OVERVIEW.md)**: Complete API documentation
- **[Examples](../tests/)**: Production-ready examples in the test suite

## Performance Tips

1. **Minimize state size**: Smaller `user_data` structs = better cache locality
2. **Use zero-copy**: For >1KB payloads, always use descriptors
3. **Batch operations**: Send multiple items before suspending
4. **Profile with `perf`**: Identify hotspots in your logic, not kcoro

---

**Congratulations!** You've built your first stackless, zero-spin coroutine application. The rest of the documentation covers advanced topics and internal details.
