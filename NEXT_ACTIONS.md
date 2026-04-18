# Immediate Actions - High-Value Files

Based on AST analysis, here are the concrete next steps.

## Summary

- **Current Progress:** 2.2% (19/277 files)
- **Matched Files:** 6
- **Average Similarity:** 0.00
- **Critical Issues:** 6 files with <0.60 similarity

## Priority 1: Fix Incomplete High-Dependency Files

### 1. include.ggml
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 68
- **Priority Score:** 21.2
- **Action:** Deep review - likely missing major functionality

## Priority 2: Port Missing High-Value Files

Critical missing files (>10 dependencies):

1. **ggml-impl** (58 deps)
   - Path: `src/ggml-impl.h`
   - Essential for 58 other files

2. **ggml-backend-impl** (47 deps)
   - Path: `src/ggml-backend-impl.h`
   - Essential for 47 other files

3. **ggml-cpu.common** (47 deps)
   - Path: `src/ggml-cpu/common.h`
   - Essential for 47 other files

4. **include.ggml-backend** (38 deps)
   - Path: `include/ggml-backend.h`
   - Essential for 38 other files

5. **ggml-sycl.fattn-vec** (37 deps)
   - Path: `src/ggml-sycl/fattn-vec.hpp`
   - Essential for 37 other files

6. **ggml-zdnn.utils** (23 deps)
   - Path: `src/ggml-zdnn/utils.hpp`
   - Essential for 23 other files

7. **openvino.node_context** (19 deps)
   - Path: `src/ggml-openvino/openvino/node_context.h`
   - Essential for 19 other files

8. **openvino.op_table** (18 deps)
   - Path: `src/ggml-openvino/openvino/op_table.h`
   - Essential for 18 other files

9. **ggml-sycl.convert** (17 deps)
   - Path: `src/ggml-sycl/convert.hpp`
   - Essential for 17 other files

10. **ggml-common** (16 deps)
   - Path: `src/ggml-common.h`
   - Essential for 16 other files

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
./ast_distance --init-tasks ../../tmp/llama.cpp/ggml cpp ../../src/nativeMain/kotlin/ai/solace/llamakotlin kotlin tasks.json ../../AGENTS.md

# Get next high-priority task
./ast_distance --assign tasks.json <agent-id>
```
