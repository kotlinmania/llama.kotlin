# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `ggml-backend` | `core.GGMLCpuBufferNative` | 0.00 | 0 | 147 | 1470.0 |
| 2 | `gguf` | `gguf.GGUFParser` | 0.01 | 0 | 97 | 970.0 |
| 3 | `include.gguf` | `gguf.GGUFTypes` | 0.00 | 3 | 62 | 798.8 |
| 4 | `openvino.input_model` | `model.LlamaInferencePipelineBuilder` | 0.00 | 3 | 5 | 70.8 |
| 5 | `ggml-sycl.type` | `model.LlamaHTypes` | 0.00 | 2 | 4 | 54.3 |
| 6 | `op.cont` | `gguf.GGUFContext` | 0.00 | 0 | 1 | 10.0 |
| 7 | `ggml-cann.aclnn_ops` | `lnn.LNNActors` | 0.00 | 1 | 0 | 3.5 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **include.gguf** → `gguf.GGUFTypes`
  - Similarity: 0.00
  - Dependencies: 3

- **openvino.input_model** → `model.LlamaInferencePipelineBuilder`
  - Similarity: 0.00
  - Dependencies: 3

- **ggml-sycl.type** → `model.LlamaHTypes`
  - Similarity: 0.00
  - Dependencies: 2
  - Lint issues: 2

- **ggml-cann.aclnn_ops** → `lnn.LNNActors`
  - Similarity: 0.00
  - Dependencies: 1
  - TODOs: 9
  - Lint issues: 3

## Missing Files (Top by Dependents)

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

