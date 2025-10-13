# Stack Architecture: Stackful vs. Stackless Coroutines

## Executive Summary

**kcoro_arena uses a stackful coroutine model**, allocating a dedicated execution stack for each coroutine. This differs fundamentally from Kotlin's stackless approach and has profound implications for memory usage, implementation complexity, and interoperability with C.

---

## The Two Coroutine Models

### Stackless Coroutines (Kotlin)

**How It Works:**
- Compiler transforms `suspend` functions into state machines at compile time
- Each suspended coroutine is a small heap object (typically 200-500 bytes)
- Contains: state label, local variables, continuation reference
- No separate execution stack needed

**Memory Profile:**
```
Per coroutine: ~300 bytes (heap object)
1 million coroutines: ~300 MB
```

**Advantages:**
- Extremely memory efficient
- Can spawn millions of coroutines
- Fast context switches (just object state)
- Compiler-enforced suspension points

**Disadvantages:**
- Requires compiler support for CPS transformation
- Cannot call arbitrary C functions that block
- Suspension points must be statically known
- Complex compiler machinery

### Stackful Coroutines (kcoro_arena)

**How It Works:**
- Each coroutine has its own execution stack allocated via `mmap`
- Stack size: **64 KB default** (`KCORO_DEFAULT_STACK_SIZE`)
- Assembly-level context switching (`kc_ctx_switch.S`)
- Can suspend at *any* function call depth

**Memory Profile:**
```
Per coroutine: 64 KB (private stack) + ~200 bytes (control structure)
1 million coroutines: ~64 GB
```

**Advantages:**
- No compiler support required (works with plain C)
- Can call blocking C APIs transparently
- Suspension can occur at any depth (even in nested calls)
- Simpler mental model for C developers

**Disadvantages:**
- Higher memory footprint per coroutine
- Limited scalability (realistically 10k-100k coroutines on typical systems)
- Context switches require saving/restoring full register state

---

## kcoro_arena Stack Implementation

### Allocation (kcoro_core.c)

```c
#define KCORO_DEFAULT_STACK_SIZE (64 * 1024)  /* 64KB */

kcoro_t* kcoro_create(kcoro_fn_t fn, void* arg, size_t stack_size) {
    if (stack_size == 0) stack_size = KCORO_DEFAULT_STACK_SIZE;
    
    // Align to page boundary for guard pages
    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t total_size = (stack_size + page_size - 1) & ~(page_size - 1);
    
    // Allocate with mmap for memory protection
    void* stack_mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    
    if (stack_mem == MAP_FAILED) return NULL;
    
    // Optional: Set guard page at bottom of stack
    mprotect(stack_mem, page_size, PROT_NONE);  // Guard page
    
    co->stack_ptr = stack_mem;
    co->stack_size = total_size;
    
    // Initialize stack pointer to top of usable stack
    uintptr_t stack_top = (uintptr_t)stack_mem + total_size;
    // ... set up initial context ...
}
```

### Key Features

**Guard Pages**: The first page is marked `PROT_NONE`, so stack overflow triggers a segfault rather than silent corruption.

**mmap Advantages**:
- Stack memory not part of process heap
- Can be unmapped cleanly on coroutine termination
- OS can provide copy-on-write optimization
- Explicit memory protection

**Stack Top Initialization**: ARM64 requires 16-byte stack alignment; initial `sp` is set to `(stack_base + total_size) & ~0xF`.

---

## Memory Layout Comparison

### Kotlin (Stackless) Process

```
Process Address Space (4 GB typical)
┌─────────────────────────────┐
│ Kernel Space (2 GB)         │
├─────────────────────────────┤
│ User Stack (8 MB)           │ ← Single thread stack
├─────────────────────────────┤
│ Heap                        │
│  ┌─────────────────┐        │
│  │ Coroutine 1     │ 300 B  │
│  │ Coroutine 2     │ 300 B  │
│  │ ...             │        │
│  │ Coroutine 1M    │ 300 B  │ ← All live in heap
│  └─────────────────┘        │
├─────────────────────────────┤
│ .bss / .data / .text        │
└─────────────────────────────┘

Total for 1M coroutines: ~300 MB
```

### kcoro_arena (Stackful) Process

```
Process Address Space (64-bit ARM64: 48-bit virtual = 256 TB)
┌─────────────────────────────┐
│ Kernel Space (128 TB)       │
├─────────────────────────────┤
│ mmap region (128 TB)        │
│  ┌─────────────────┐        │
│  │ Coroutine 1     │ 64 KB  │
│  │ Guard page      │  4 KB  │
│  ├─────────────────┤        │
│  │ Coroutine 2     │ 64 KB  │
│  │ Guard page      │  4 KB  │
│  ├─────────────────┤        │
│  │ ...             │        │
│  ├─────────────────┤        │
│  │ Coroutine 10000 │ 64 KB  │ ← Each has own stack
│  │ Guard page      │  4 KB  │
│  └─────────────────┘        │
├─────────────────────────────┤
│ Heap                        │
│  Arena structures (~MB)     │
├─────────────────────────────┤
│ Thread Stacks (8 MB each)  │
└─────────────────────────────┘

Total for 10k coroutines: ~680 MB (stacks) + arena overhead
Total for 1M coroutines: ~68 GB (not practical)
```

---

## ARM64 Linux Memory Layout (64-bit)

Modern ARM64 Linux uses a **48-bit virtual address space** (256 TB addressable):

```
0x0000_0000_0000_0000  ←  User space start
    ↓
    Text segment (.text)
    Data segment (.data, .bss)
    Heap (grows up) ← malloc, new
    ↓
    ...
    ↓
    mmap region (grows down from top) ← kcoro stacks allocated here
    ↓
    Thread stacks (8 MB default per thread)
    ↓
0x0000_7FFF_FFFF_FFFF  ←  User space end (128 TB)
─────────────────────────────────────
0xFFFF_8000_0000_0000  ←  Kernel space start (128 TB)
    Kernel code, data, modules
0xFFFF_FFFF_FFFF_FFFF  ←  Top of address space
```

**Key Points:**
- User space: 128 TB (0x0000_0000_0000_0000 - 0x0000_7FFF_FFFF_FFFF)
- Kernel space: 128 TB (0xFFFF_8000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF)
- Each process gets its own page tables (via TTBR0_EL1 for user, TTBR1_EL1 for kernel)
- Default thread stack: 8 MB (ulimit -s)
- mmap allocates from high addresses, growing downward

**Practical Limits:**
- Physical RAM on modern ARM64 systems: 8-128 GB typical
- OS overcommit allows virtual allocations > physical RAM
- Page fault brings in actual pages on first access
- 10,000 coroutines × 64 KB = 640 MB virtual (realistic)
- 1,000,000 coroutines × 64 KB = 64 GB virtual (would thrash)

---

## Design Trade-offs

### When Stackful Makes Sense

✅ **kcoro_arena stackful is optimal for:**
- Interoperating with blocking C libraries (zlib, openssl, etc.)
- Systems programming where suspension depth is unpredictable
- Scenarios with <10k concurrent coroutines
- Debugging (stack traces work naturally with gdb)

### When Stackless Would Be Better

❌ **Stackless would be needed for:**
- Hundreds of thousands of concurrent connections (async I/O server)
- Memory-constrained embedded systems
- Languages with compiler support (Rust async, Kotlin, C++ with co_await)

---

## Hybrid Optimization: Dynamic Stack Sizing

**Future Enhancement Possibility:**

Instead of fixed 64 KB, use **split stacks** or **segmented stacks**:

```c
// Allocate small initial stack (4 KB)
void* stack = mmap(NULL, 4096, ...);

// On stack overflow (guard page fault):
// 1. Allocate new larger segment (e.g., 16 KB)
// 2. Copy active frame to new segment
// 3. Update stack pointer
// 4. Resume execution

// Result: Most coroutines use 4 KB; deep-call coroutines grow as needed
```

This approach (used by Go until 1.3) can reduce average memory per coroutine from 64 KB to ~8 KB while preserving stackful benefits.

---

## Conclusion

Our **stackful** design is a deliberate architectural choice optimized for:

1. **C interoperability**: No compiler required; works with any C toolchain
2. **Simplicity**: Assembly-level context switching is well-understood
3. **Transparency**: Can suspend inside any function call depth
4. **Debugging**: Standard tools (gdb, lldb) work naturally

The memory cost (64 KB/coroutine) limits scalability to ~10k concurrent coroutines on typical systems, but this is sufficient for high-performance, low-latency use cases where arena-based zero-copy messaging provides the real performance win.

For workloads requiring >100k coroutines, a stackless transformation or hybrid approach would be necessary—but that would require compiler infrastructure we're deliberately avoiding to maintain the "plain C, any toolchain" design goal.

---

## References

- ARM64 Virtual Memory: https://www.kernel.org/doc/html/latest/arm64/memory.html
- mmap(2) man page: `MAP_STACK`, `MAP_GROWSDOWN` flags
- Kotlin Coroutines Internals: State machine transformation in kotlinc
- Go 1.3 Release Notes: Switch from segmented to contiguous stacks
- BizTalk Orchestration Engine: Serialized state blobs in SQL Server
