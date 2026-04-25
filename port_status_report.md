# Code Port - Progress Report

**Generated:** 2026-04-25
**Source:** tmp/llama.cpp/ggml
**Target:** src/nativeMain/kotlin/ai/solace/llamakotlin

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 277 | 100% |
| Target units (paired) | 48 | - |
| Target files (total) | 48 | - |
| Porting progress | 7 | 2.5% (matched) |
| Missing files | 270 | 97.5% |

## Port Quality Analysis

**Average Similarity:** 0.00

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 7 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `ggml-backend` → `core.GGMLCpuBufferNative` (0.00)
- `gguf` → `gguf.GGUFParser` (0.01)
- `include.gguf` → `gguf.GGUFTypes` (0.00, 3 deps)
- `openvino.input_model` → `model.LlamaInferencePipelineBuilder` (0.00, 3 deps)
- `ggml-sycl.type` → `model.LlamaHTypes` (0.00, 2 deps)
- `op.cont` → `gguf.GGUFContext` (0.00)
- `ggml-cann.aclnn_ops` → `lnn.LNNActors` (0.00, 1 deps)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `ggml-backend` | `core.GGMLCpuBufferNative` | 16/16 | `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_status` … |
| `gguf` | `gguf.GGUFParser` | 14/14 | `type_to_gguf_type`, `gguf_type`, `gguf_kv` … |
| `include.gguf` | `gguf.GGUFTypes` | 4/6 | `gguf_context`, `ggml_context`, `ggml_tensor` … |
| `openvino.input_model` | `model.LlamaInferencePipelineBuilder` | 3/3 | `FrontEnd`, `GgmlDecoder`, `InputModel` |
| `ggml-sycl.type` | `model.LlamaHTypes` | 1/1 | `__nv_fp8_e4m3` |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `include.ggml` | 68 | `include/ggml.h` |
| 2 | `ggml-impl` | 58 | `src/ggml-impl.h` |
| 3 | `ggml-backend-impl` | 47 | `src/ggml-backend-impl.h` |
| 4 | `ggml-cpu.common` | 47 | `src/ggml-cpu/common.h` |
| 5 | `include.ggml-backend` | 38 | `include/ggml-backend.h` |
| 6 | `ggml-sycl.fattn-vec` | 37 | `src/ggml-sycl/fattn-vec.hpp` |
| 7 | `ggml-zdnn.utils` | 23 | `src/ggml-zdnn/utils.hpp` |
| 8 | `openvino.node_context` | 19 | `src/ggml-openvino/openvino/node_context.h` |
| 9 | `openvino.op_table` | 18 | `src/ggml-openvino/openvino/op_table.h` |
| 10 | `ggml-sycl.convert` | 17 | `src/ggml-sycl/convert.hpp` |
| 11 | `ggml-common` | 16 | `src/ggml-common.h` |
| 12 | `include.ggml-cpu` | 13 | `include/ggml-cpu.h` |
| 13 | `htp.hvx-base` | 11 | `src/ggml-hexagon/htp/hvx-base.h` |
| 14 | `ggml-cpu.ggml-cpu-impl` | 11 | `src/ggml-cpu/ggml-cpu-impl.h` |
| 15 | `ggml-sycl.fattn-tile` | 11 | `src/ggml-sycl/fattn-tile.hpp` |
| 16 | `ggml-cpu.simd-mappings` | 10 | `src/ggml-cpu/simd-mappings.h` |
| 17 | `ggml-cpu.traits` | 9 | `src/ggml-cpu/traits.h` |
| 18 | `dpct.helper` | 9 | `src/ggml-sycl/dpct/helper.hpp` |
| 19 | `op.reshape` | 9 | `src/ggml-openvino/openvino/op/reshape.cpp` |
| 20 | `ggml-sycl.concat` | 9 | `src/ggml-sycl/concat.hpp` |

... and 250 more missing files.

## Documentation Gaps

There is missing documentation that is hurting overall scoring.

**Documentation coverage:** 639 / 1413 lines (45%)

Top documentation gaps (>20%):

- `ggml-cann.aclnn_ops` - 89% gap (1413 → 157 lines)

