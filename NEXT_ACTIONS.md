# Immediate Actions - High-Value Files

Based on AST analysis, here are the concrete next steps.

## Summary

- **Current Progress:** 14.8% (160/256 files)
- **Matched Files:** 38
- **Average Similarity:** 0.07
- **Critical Issues:** 37 files with <0.60 similarity

## Priority 1: Fix Incomplete High-Dependency Files

### 1. ggml-backend-impl
- **Similarity:** 0.36 (needs 49% improvement)
- **Dependencies:** 47
- **Priority Score:** 349.4
- **Symbol Deficit:** 19 (functions: 3, types: 16)
- **Action:** Deep review - likely missing major functionality

### 2. ggml-cpu.simd-mappings
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 10
- **Priority Score:** 307.9
- **Symbol Deficit:** 20 (functions: 20, types: 0)
- **Action:** Deep review - likely missing major functionality

### 3. ggml-backend
- **Similarity:** 0.72 (needs 13% improvement)
- **Dependencies:** 20
- **Priority Score:** 293.9
- **Symbol Deficit:** 18 (functions: 4, types: 14)
- **Action:** Review and complete missing sections

### 4. ggml-impl
- **Similarity:** 0.54 (needs 31% improvement)
- **Dependencies:** 58
- **Priority Score:** 154.6
- **Symbol Deficit:** 8 (functions: 2, types: 6)
- **Action:** Deep review - likely missing major functionality

### 5. ggml-zdnn.utils
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 23
- **Priority Score:** 81.3
- **Symbol Deficit:** 4 (functions: 4, types: 0)
- **Action:** Deep review - likely missing major functionality

### 6. ggml-common
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 16
- **Priority Score:** 45.5
- **Symbol Deficit:** 2 (functions: 2, types: 0)
- **Action:** Deep review - likely missing major functionality

### 7. ggml-cpu.common
- **Similarity:** 0.56 (needs 29% improvement)
- **Dependencies:** 47
- **Priority Score:** 43.9
- **Symbol Deficit:** 2 (functions: 0, types: 2)
- **Action:** Deep review - likely missing major functionality

### 8. ggml
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 50
- **Priority Score:** 37.5
- **Symbol Deficit:** 1 (functions: 1, types: 0)
- **Action:** Deep review - likely missing major functionality

### 9. ggml-cpu.ggml-cpu
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 12
- **Priority Score:** 12.8
- **Action:** Deep review - likely missing major functionality

### 10. ggml-cpu.ggml-cpu-impl
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 11
- **Priority Score:** 12.4
- **TODOs:** 1
- **Action:** Deep review - likely missing major functionality

## Priority 2: Port Missing High-Value Files

Critical missing files (>10 dependencies):

1. **ggml-sycl.fattn-vec** (37 deps)
   - Path: `ggml-sycl/fattn-vec.hpp`
   - Essential for 37 other files

2. **openvino.node_context** (19 deps)
   - Path: `ggml-openvino/openvino/node_context.h`
   - Essential for 19 other files

3. **openvino.op_table** (18 deps)
   - Path: `ggml-openvino/openvino/op_table.h`
   - Essential for 18 other files

4. **ggml-sycl.convert** (17 deps)
   - Path: `ggml-sycl/convert.hpp`
   - Essential for 17 other files

5. **htp.hvx-base** (11 deps)
   - Path: `ggml-hexagon/htp/hvx-base.h`
   - Essential for 11 other files

6. **ggml-sycl.fattn-tile** (11 deps)
   - Path: `ggml-sycl/fattn-tile.hpp`
   - Essential for 11 other files

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
./ast_distance --init-tasks ../../tmp/llama.cpp/ggml/src cpp ../../src kotlin tasks.json ../../AGENTS.md

# Get next high-priority task
./ast_distance --assign tasks.json <agent-id>
```
