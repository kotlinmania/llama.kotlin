# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `gguf` | `gguf.GGUFParser` | 0.00 | 0 | 98 | 980.0 |
| 2 | `include.gguf` | `gguf.GGUFTypes` | 0.00 | 3 | 63 | 811.6 |
| 3 | `openvino.input_model` | `model.Grammar` | 0.00 | 3 | 5 | 70.8 |
| 4 | `include.ggml` | `model.GGMLIntegration` | 0.00 | 68 | 0 | 21.2 |
| 5 | `op.cont` | `gguf.GGUFContext` | 0.00 | 0 | 1 | 10.0 |
| 6 | `ggml-cann.aclnn_ops` | `lnn.LNNActors` | 0.00 | 1 | 0 | 3.5 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **include.gguf** → `gguf.GGUFTypes`
  - Similarity: 0.00
  - Dependencies: 3

- **openvino.input_model** → `model.Grammar`
  - Similarity: 0.00
  - Dependencies: 3

- **include.ggml** → `model.GGMLIntegration`
  - Similarity: 0.00
  - Dependencies: 68
  - Lint issues: 10

- **ggml-cann.aclnn_ops** → `lnn.LNNActors`
  - Similarity: 0.00
  - Dependencies: 1
  - TODOs: 9
  - Lint issues: 3

## Missing Files (Top by Dependents)

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

