# Immediate Actions - High-Value Files

Based on AST analysis, here are the concrete next steps.

## Summary

- **Current Progress:** 22.4% (166/143 files)
- **Matched Files:** 32
- **Average Similarity:** 0.14
- **Critical Issues:** 30 files with <0.60 similarity

## Priority 1: Fix Incomplete High-Dependency Files

### 1. llama
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 16
- **Priority Score:** 828.8
- **Symbol Deficit:** 52 (functions: 34, types: 18)
- **Action:** Deep review - likely missing major functionality

### 2. llama-impl
- **Similarity:** 0.13 (needs 72% improvement)
- **Dependencies:** 19
- **Priority Score:** 204.9
- **Symbol Deficit:** 12 (functions: 9, types: 3)
- **Action:** Deep review - likely missing major functionality

### 3. models.models
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 114
- **Priority Score:** 23.7
- **TODOs:** 2
- **Action:** Deep review - likely missing major functionality

### 4. llama-memory-recurrent
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 12
- **Priority Score:** 12.8
- **TODOs:** 1
- **Action:** Deep review - likely missing major functionality

### 5. llama-model
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 12
- **Priority Score:** 12.8
- **TODOs:** 1
- **Action:** Deep review - likely missing major functionality

## Priority 2: Port Missing High-Value Files

Critical missing files (>10 dependencies):

No missing high-value files detected.

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
./ast_distance --init-tasks ../../tmp/llama.cpp/src cpp ../../src kotlin tasks.json ../../AGENTS.md

# Get next high-priority task
./ast_distance --assign tasks.json <agent-id>
```
