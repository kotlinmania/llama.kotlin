# kcoro_arena Documentation Status

## Last Updated: 2025-10-13  
## Status: Post-Code-Audit Update

This document tracks the current state of kcoro_arena documentation against actual implementation.

## Implementation Status Summary

### ✅ Completed Components

#### Core Runtime
- **Stackless Continuations** (`kcoro_stackless.c`): Fully implemented with CPS-style execution
  - State: Working, production-ready
  - Doc Location: `docs/components/stackless_runtime/`
  
- **Token Kernel** (`kc_token_kernel.c`): Complete implementation with hash table, freelist, ready queue
  - State: Working, zero-spin event-driven callbacks
  - Doc Location: `docs/components/token_kernel/`
  
- **Scheduler** (`koro_sched_stackless.c`): Stackless main loop with ready queue
  - State: Working, no stack switching
  - Doc Location: `docs/components/stackless_runtime/SCHEDULER.md`

#### Memory Management
- **Arena Allocator** (`kc_arena.c`): Bump allocator with pages
  - State: Working
  - Doc Location: `docs/components/arena/`
  
- **Descriptor System** (`kc_desc.c`): Zero-copy payload descriptors
  - State: Working
  - Doc Location: `docs/components/descriptors/`

#### Communication Primitives
- **Token-based Send/Receive**: Event-driven, callback-based
  - State: Working
  - Doc Location: Needs update in token_kernel docs

### ⚠️ Documentation Updates Needed

1. **Stackless Migration Complete**
   - Old docs mention stack switching (obsolete)
   - Need to remove references to `kc_ctx_switch.S`
   - Update all flow diagrams to show callback-based resumption

2. **Token Kernel Reality**
   - Current docs mention "stubs" but implementation is complete
   - Missing: callback registration patterns
   - Missing: ready queue mechanics

3. **FFI/koro_go Layer**
   - Documentation exists (`docs/components/ffi_go/`) but implementation incomplete
   - Needs status update: planned but not yet built

4. **Assembly Code**
   - `kc_vm_apply.S` still exists but unused in golden path
   - Documentation should clarify: legacy/deprecated

### 🔄 Component Documentation Reorganization Needed

#### Current Structure Issues
- Some docs in root (`STACKLESS_MIGRATION.md`) should be in `components/`
- Worker documentation (`docs/components/workers/ASSEMBLY_CONTEXT_SWITCH.md`) obsolete
- Missing component docs for:
  - `kc_task.c` (lightweight task abstraction)
  - `kc_dispatch.c` (current state vs planned koro_go)

## Priority Documentation Tasks

### 🚨 CRITICAL: Architecture Documentation
**All docs must reflect stackless CPS design as the golden path.**

### High Priority

1. ✅ **Main README** - CREATED with accurate architecture overview
2. ⚠️ **Update `token_kernel/OVERVIEW.md`** - Stub mentions need removal, worker thread details needed
3. ⚠️ **Mark assembly docs as deprecated** - Add notices to all worker/assembly docs
4. ❌ **Create callback system docs** - Missing documentation for callback patterns
5. ❌ **Document macro system** - `KORO_BEGIN/SEND/END` expansion needs examples

### Medium Priority  
1. ⚠️ **Update arena component docs** - Remove stack references, expand descriptor system
2. ❌ **Create scheduler integration doc** - How token kernel and scheduler interact
3. ❌ **Document zero-spin mechanics** - Prove absence of spin loops with code paths
4. ⚠️ **Flow diagrams** - Need event-driven resumption diagrams (no context switch)

### Low Priority
1. Clean up obsolete stackful migration docs
2. Consolidate duplicate information
3. Add performance characteristics documentation

## Documentation Standards

All component documentation should follow this structure:

```
docs/components/<component>/
  ├── OVERVIEW.md          # What it is, why it exists
  ├── API.md              # Public interface (if applicable)
  ├── IMPLEMENTATION.md    # How it works internally
  └── STATUS.md           # Current state, known issues
```

## Next Steps

1. Audit each component's actual source code
2. Update/create documentation to match reality
3. Remove obsolete content (stackful references)
4. Add missing components (task, dispatch current state)
5. Update top-level README with architectural overview
