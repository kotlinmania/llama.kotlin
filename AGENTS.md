# llama.kotlin Project Instructions

This file provides guidance for agents working on the Kotlin port of `llama.cpp`.
It summarizes the current project state and the porting methodology.

## Documentation Philosophy
- Write docs the way you'd brief the next engineer on shift: approachable voice first, precise details alongside.
- Define domain jargon the first time it appears so future readers never have to guess.
- When you describe a workflow, include a tiny example or checklist so it can be followed step by step.
- Keep accuracy non-negotiable—if something is nuanced, spell it out clearly without slipping into cryptic shorthand.

## Project Overview
- **Goal**: Direct line-by-line transliteration of llama.cpp from C++ to Kotlin/Native
- **Method**: Use `ast_distance --deep` to track porting progress, cinterop for C API access
- **Source**: `tmp/llama.cpp/` (cloned reference repo)
- **Target**: `src/nativeMain/kotlin/ai/solace/llamakotlin/` (Kotlin port)
- **Build**: Kotlin 2.3.20, Gradle Multiplatform, targets macOS (arm64/x64), Linux x64, Windows x64

## Porting Strategy

### Direct C++ → Kotlin Transliteration
This is a **line-by-line transliteration** — mirror the C++ structure as closely as possible in Kotlin.
Each C++ source file should map to a corresponding Kotlin file. Preserve function names, variable names,
and control flow. The goal is that someone reading the C++ and Kotlin side-by-side can verify correctness.

### cinterop for Validation
Kotlin/Native cinterop is configured to expose the ggml C API (`ggml.cinterop` package).
This allows ported Kotlin code to call into the reference C implementation for validation and
comparison during development. The cinterop def file is at `src/nativeInterop/cinterop/ggml.def`.

### ast_distance Tracking
The `tools/ast_distance` binary tracks porting progress. Key commands:

```bash
# Full analysis: AST + deps + TODOs + lint + line ratios
./tools/ast_distance --deep tmp/llama.cpp/ggml cpp src/nativeMain/kotlin/ai/solace/llamakotlin kotlin

# Show files missing from target, ranked by importance
./tools/ast_distance --missing tmp/llama.cpp/ggml cpp src/nativeMain/kotlin/ai/solace/llamakotlin kotlin

# Compare functions between specific files
./tools/ast_distance --compare-functions <cpp_file> cpp <kotlin_file> kotlin

# Initialize task queue for systematic porting
./tools/ast_distance --init-tasks tmp/llama.cpp/ggml cpp src/nativeMain/kotlin/ai/solace/llamakotlin kotlin tasks.json
```

### port-lint Headers
Every ported Kotlin file must include a port-lint header in the first 50 lines so ast_distance
can match it to the correct C++ source file:

```kotlin
// port-lint: source ggml/src/ggml-cpu/ops.cpp
package ai.solace.llamakotlin.core
```

This replaces heuristic name-matching with explicit source tracking.

## Porting Priority

Focus on core ggml files first, ordered by dependency count (from `ast_distance --deep`):

1. **ggml-impl.h** (58 dependents) → Core implementation internals
2. **ggml-backend-impl.h** (47 dependents) → Backend interface
3. **ggml-cpu/common.h** (47 dependents) → CPU backend common definitions
4. **ggml-backend.h** (38 dependents) → Public backend API
5. **ggml-common.h** (16 dependents) → Shared definitions
6. **ggml-cpu.h** (13 dependents) → CPU backend public API
7. **ggml-cpu/ggml-cpu-impl.h** (11 dependents) → CPU implementation details

Skip hardware-specific backends (SYCL, OpenVINO, CANN, Hexagon, Vulkan, CUDA, Metal)
until the CPU path is complete.

## Coding Guidelines
- **Transliteration First**: Mirror C++ structure. Don't redesign — translate.
- **Kotlin/Native Types**: Use Kotlin equivalents: `Int` for `int32_t`, `Long` for `int64_t`,
  `Float` for `float`, `Double` for `double`, `ByteArray` for raw buffers.
- **cinterop Types**: Use `ggml.cinterop.*` types when calling into the C reference for validation.
- **Naming**: Preserve C++ function and variable names as closely as Kotlin allows.
  Convert `snake_case` to `camelCase` only where Kotlin convention strongly demands it.
- **port-lint Headers**: Always include `// port-lint: source <relative-path>` in ported files.
- **Documentation**: Document any deviations from the C++ original with `// NOTE:` comments.
- **Tests**: Write tests that validate Kotlin output against the C reference via cinterop.

## Build and Test
- **Build**: `./gradlew build`
- **Test**: `./gradlew allTests`
- **Targets**: macOS arm64/x64, Linux x64, Windows x64, JS (IR), JVM
- **cinterop**: Configured for all native targets, exposes `ggml.cinterop` package
- **SSL Issues**: If Gradle fails with SSL errors, install Java/Gradle via SDKMAN:
  ```bash
  curl -s "https://get.sdkman.io" | bash
  source "$HOME/.sdkman/bin/sdkman-init.sh"
  sdk install java 17.0.9-tem
  sdk install gradle 8.13
  ```

## Key Files and Modules
- **Core Types**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLTypes.kt`
- **Memory Management**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLAlloc.kt`
- **Tensor Operations**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLOps.kt`
- **Computation Logic**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLComputeOps.kt`
- **Graph Execution**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLGraph.kt`
- **cinterop Def**: `src/nativeInterop/cinterop/ggml.def`
- **Test Suite**: `src/commonTest/kotlin/ai/solace/llamakotlin/core/`
- **C++ Reference**: `tmp/llama.cpp/` (shallow clone)
- **ast_distance Reports**: `port_status_report.md`, `high_priority_ports.md`, `NEXT_ACTIONS.md`

## Project Scope
- **Primary Focus**: CPU backend via direct transliteration
- **cinterop**: For validation against C reference, not as the runtime path
- **Deferred**: GPU backends (CUDA, Metal, Vulkan, etc.) — CPU first
- **Platform Targets**: macOS (arm64, x64), Linux x64, Windows x64

## Documentation Workflow: Verified vs. Backlog

This repository uses a clear, low-friction split between ground truth and aspirational ideas.

- Voice and tone: Write like you're briefing the next engineer on shift. Friendly first, precise alongside. Avoid compressed "note-to-self" shorthand.
- Navigation: Folder structure is the linkage. We avoid nested READMEs for navigation. One top-level index exists: `architectural_index.md`.

What goes where
- Component folders:
  - `*_VERIFIED.md` — shipping behavior and tests (ground truth).
  - `*_FUTURE.md` — a stub that stays in place and points to the centralized backlog (preserves local links and traceability).
- Central backlog (`docs/backlog/...`):
  - Files are named `NAME_issues-<id>[-<id>...].md` (example: `SCHEDULER_issues-108-113.md`).
  - Each backlog file starts with `_Backlog: related issues #…_` and also includes an HTML comment listing the issue IDs.

Adding new future work
1) Capture notes in the component's `*_FUTURE.md` or ensure its pointer exists.
2) Open GitHub issues for those items; record issue numbers in the backlog filename.
3) Write/update the backlog file under `docs/backlog/<area>/<component>/NAME_issues-<ids>.md`.
4) Update `architectural_index.md` to link the VERIFIED doc and the backlog file.

Renaming convention
- Backlog filenames intentionally drop the word "FUTURE".
- Keep the original `*_FUTURE.md` stubs in component folders for familiarity and link stability.

Close-out workflow
- When work ships, update the backlog file with the outcome and PR links.
- Reflect shipped changes in the sibling `*_VERIFIED.md`.
- In the `*_FUTURE.md` stub, replace the "Moved" note with: "Resolved via #<id> → incorporated into VERIFIED on <YYYY‑MM‑DD>", and remove the backlog pointer.
- If fully consumed, move the backlog file to `docs/backlog/_archive/` (same filename) for history.

Notes
- No pre-commit/CI checks enforce this. Keep edits small and consistent.
- Useful greps:
  - List backlog files: `rg --files docs/backlog -n`
  - Find active backlog headers: `rg '^_Backlog: related issues #' docs/backlog`
  - Find FUTURE stubs: `rg --files -g '*FUTURE.md' external -n`
