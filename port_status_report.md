# Code Port - Progress Report

**Generated:** 2026-04-18
**Source:** tmp/llama.cpp/ggml
**Target:** src/nativeMain/kotlin/ai/solace/llamakotlin

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 277 | 100% |
| Target units (paired) | 19 | - |
| Target files (total) | 19 | - |
| Porting progress | 6 | 2.2% (matched) |
| Missing files | 271 | 97.8% |

## Port Quality Analysis

**Average Similarity:** 0.00

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 6 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `gguf` → `gguf.GGUFParser` (0.00)
- `include.gguf` → `gguf.GGUFTypes` (0.00, 3 deps)
- `openvino.input_model` → `model.Grammar` (0.00, 3 deps)
- `include.ggml` → `model.GGMLIntegration` (0.00, 68 deps)
- `op.cont` → `gguf.GGUFContext` (0.00)
- `ggml-cann.aclnn_ops` → `lnn.LNNActors` (0.00, 1 deps)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `gguf` | `gguf.GGUFParser` | 14/14 | `type_to_gguf_type`, `gguf_type`, `gguf_kv` … |
| `include.gguf` | `gguf.GGUFTypes` | 5/6 | `gguf_context`, `gguf_init_params`, `ggml_context` … |
| `openvino.input_model` | `model.Grammar` | 3/3 | `FrontEnd`, `GgmlDecoder`, `InputModel` |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `ggml-impl` | 58 | `src/ggml-impl.h` |
| 2 | `ggml-backend-impl` | 47 | `src/ggml-backend-impl.h` |
| 3 | `ggml-cpu.common` | 47 | `src/ggml-cpu/common.h` |
| 4 | `include.ggml-backend` | 38 | `include/ggml-backend.h` |
| 5 | `ggml-sycl.fattn-vec` | 37 | `src/ggml-sycl/fattn-vec.hpp` |
| 6 | `ggml-zdnn.utils` | 23 | `src/ggml-zdnn/utils.hpp` |
| 7 | `openvino.node_context` | 19 | `src/ggml-openvino/openvino/node_context.h` |
| 8 | `openvino.op_table` | 18 | `src/ggml-openvino/openvino/op_table.h` |
| 9 | `ggml-sycl.convert` | 17 | `src/ggml-sycl/convert.hpp` |
| 10 | `ggml-common` | 16 | `src/ggml-common.h` |
| 11 | `include.ggml-cpu` | 13 | `include/ggml-cpu.h` |
| 12 | `ggml-cpu.ggml-cpu-impl` | 11 | `src/ggml-cpu/ggml-cpu-impl.h` |
| 13 | `htp.hvx-base` | 11 | `src/ggml-hexagon/htp/hvx-base.h` |
| 14 | `ggml-sycl.fattn-tile` | 11 | `src/ggml-sycl/fattn-tile.hpp` |
| 15 | `ggml-cpu.simd-mappings` | 10 | `src/ggml-cpu/simd-mappings.h` |
| 16 | `op.reshape` | 9 | `src/ggml-openvino/openvino/op/reshape.cpp` |
| 17 | `ggml-sycl.concat` | 9 | `src/ggml-sycl/concat.hpp` |
| 18 | `ggml-cpu.traits` | 9 | `src/ggml-cpu/traits.h` |
| 19 | `dpct.helper` | 9 | `src/ggml-sycl/dpct/helper.hpp` |
| 20 | `ggml-virtgpu.ggml-remoting` | 8 | `src/ggml-virtgpu/ggml-remoting.h` |

... and 251 more missing files.

## Documentation Gaps

There is missing documentation that is hurting overall scoring.

**Documentation coverage:** 332 / 1418 lines (23%)

Top documentation gaps (>20%):

- `ggml-cann.aclnn_ops` - 89% gap (1413 → 157 lines)

