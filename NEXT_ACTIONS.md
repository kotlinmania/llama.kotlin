# Immediate Actions - High-Value Files

Based on AST analysis, here are the concrete next steps.

## Summary

- **Current Progress:** 5.6% (19/143 files)
- **Matched Files:** 8
- **Average Similarity:** 0.02
- **Critical Issues:** 8 files with <0.60 similarity

## Priority 1: Fix Incomplete High-Dependency Files

### 1. models.models
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 114
- **Priority Score:** 4993.6
- **Symbol Deficit:** 255 (functions: 142, types: 113)
- **Action:** Deep review - likely missing major functionality

### 2. llama-model
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 12
- **Priority Score:** 1404.8
- **Symbol Deficit:** 92 (functions: 77, types: 15)
- **Action:** Deep review - likely missing major functionality

## Priority 2: Port Missing High-Value Files

Critical missing files (>10 dependencies):

1. **llama-impl** (19 deps)
   - Path: `llama-impl.h`
   - Essential for 19 other files

2. **llama** (16 deps)
   - Path: `llama.cpp`
   - Essential for 16 other files

3. **llama-memory-recurrent** (12 deps)
   - Path: `llama-memory-recurrent.h`
   - Essential for 12 other files

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
./ast_distance --init-tasks ../../tmp/llama.cpp/src cpp ../../src/nativeMain/kotlin/ai/solace/llamakotlin kotlin tasks.json ../../AGENTS.md

# Get next high-priority task
./ast_distance --assign tasks.json <agent-id>
```
