# Immediate Actions - High-Value Files

Based on AST analysis, here are the concrete next steps.

## Summary

- **Current Progress:** 6.1% (23/277 files)
- **Matched Files:** 17
- **Average Similarity:** 0.02
- **Critical Issues:** 17 files with <0.60 similarity

## Priority 1: Fix Incomplete High-Dependency Files

### 1. include.ggml-cpu
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 13
- **Priority Score:** 1097.9
- **Symbol Deficit:** 71 (functions: 62, types: 9)
- **Action:** Deep review - likely missing major functionality

### 2. ggml-impl
- **Similarity:** 0.34 (needs 51% improvement)
- **Dependencies:** 58
- **Priority Score:** 449.2
- **Symbol Deficit:** 24 (functions: 18, types: 6)
- **Action:** Deep review - likely missing major functionality

### 3. ggml-common
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 16
- **Priority Score:** 45.5
- **Symbol Deficit:** 2 (functions: 2, types: 0)
- **TODOs:** 6
- **Action:** Deep review - likely missing major functionality

### 4. include.ggml
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 68
- **Priority Score:** 21.2
- **Action:** Deep review - likely missing major functionality

### 5. ggml-backend-impl
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 47
- **Priority Score:** 19.4
- **Action:** Deep review - likely missing major functionality

### 6. include.ggml-backend
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 38
- **Priority Score:** 18.3
- **Action:** Deep review - likely missing major functionality

## Priority 2: Port Missing High-Value Files

Critical missing files (>10 dependencies):

1. **ggml-cpu.common** (47 deps)
   - Path: `src/ggml-cpu/common.h`
   - Essential for 47 other files

2. **ggml-sycl.fattn-vec** (37 deps)
   - Path: `src/ggml-sycl/fattn-vec.hpp`
   - Essential for 37 other files

3. **ggml-zdnn.utils** (23 deps)
   - Path: `src/ggml-zdnn/utils.hpp`
   - Essential for 23 other files

4. **openvino.node_context** (19 deps)
   - Path: `src/ggml-openvino/openvino/node_context.h`
   - Essential for 19 other files

5. **openvino.op_table** (18 deps)
   - Path: `src/ggml-openvino/openvino/op_table.h`
   - Essential for 18 other files

6. **ggml-sycl.convert** (17 deps)
   - Path: `src/ggml-sycl/convert.hpp`
   - Essential for 17 other files

7. **htp.hvx-base** (11 deps)
   - Path: `src/ggml-hexagon/htp/hvx-base.h`
   - Essential for 11 other files

8. **ggml-cpu.ggml-cpu-impl** (11 deps)
   - Path: `src/ggml-cpu/ggml-cpu-impl.h`
   - Essential for 11 other files

9. **ggml-sycl.fattn-tile** (11 deps)
   - Path: `src/ggml-sycl/fattn-tile.hpp`
   - Essential for 11 other files

10. **ggml-cpu.simd-mappings** (10 deps)
   - Path: `src/ggml-cpu/simd-mappings.h`
   - Essential for 10 other files

## Success Criteria

For each file to be considered "complete":
- **Similarity ≥ 0.85** (Excellent threshold)
- All public APIs ported
- All tests ported
- Documentation ported
- port-lint header present

## Next Commands

```bash
# Initialize task queue for systematic porting
cd tools/ast_distance
./ast_distance --init-tasks ../../tmp/llama.cpp/ggml cpp ../../src/commonMain/kotlin/ai/solace/llamakotlin kotlin tasks.json ../../AGENTS.md

# Get next high-priority task
./ast_distance --assign tasks.json <agent-id>
```
