# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `include.ggml` | `core.GGMLOps` | 0.00 | 68 | 363 | 6725.1 |
| 2 | `dpct.helper` | `core.QuantizationHelper` | 0.00 | 9 | 192 | 2815.7 |
| 3 | `ggml-backend` | `core.GGMLBackend` | 0.00 | 0 | 144 | 1440.0 |
| 4 | `ggml-impl` | `core.NumericConversions` | 0.00 | 58 | 63 | 1164.2 |
| 5 | `include.ggml-cpu` | `core.GGMLCpuExecutor` | 0.00 | 13 | 71 | 1097.9 |
| 6 | `ggml-cpu.ggml-cpu` | `core.GGMLCpuBackend` | 0.00 | 0 | 46 | 460.0 |
| 7 | `include.ggml-alloc` | `core.GGMLAlloc` | 0.00 | 8 | 19 | 284.5 |
| 8 | `openvino.utils` | `core.GGMLTensorUtils` | 0.00 | 0 | 17 | 170.0 |
| 9 | `ggml-virtgpu.ggml-backend` | `core.GGMLBackendUtils` | 0.00 | 0 | 6 | 60.0 |
| 10 | `ggml-sycl.type` | `core.GGMLTypes` | 0.00 | 2 | 4 | 54.3 |
| 11 | `ggml-cpu.simd-gemm` | `simd.GGMLSimd` | 0.00 | 1 | 4 | 49.0 |
| 12 | `ggml-threading` | `core.GGMLScheduler` | 0.00 | 1 | 2 | 26.2 |
| 13 | `ggml-cpu.quants` | `core.GGMLQuants` | 0.00 | 4 | 0 | 8.0 |
| 14 | `ggml` | `core.GGMLGraph` | 0.00 | 0 | 0 | 0.0 |
| 15 | `ggml-cpu.ops` | `core.GGMLComputeOps` | 0.00 | 0 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **include.ggml** → `core.GGMLOps`
  - Similarity: 0.00
  - Dependencies: 68
  - Lint issues: 9

- **dpct.helper** → `core.QuantizationHelper`
  - Similarity: 0.00
  - Dependencies: 9

- **ggml-impl** → `core.NumericConversions`
  - Similarity: 0.00
  - Dependencies: 58

- **include.ggml-cpu** → `core.GGMLCpuExecutor`
  - Similarity: 0.00
  - Dependencies: 13
  - Lint issues: 1

- **include.ggml-alloc** → `core.GGMLAlloc`
  - Similarity: 0.00
  - Dependencies: 8

- **ggml-sycl.type** → `core.GGMLTypes`
  - Similarity: 0.00
  - Dependencies: 2

- **ggml-cpu.simd-gemm** → `simd.GGMLSimd`
  - Similarity: 0.00
  - Dependencies: 1

- **ggml-threading** → `core.GGMLScheduler`
  - Similarity: 0.00
  - Dependencies: 1
  - Lint issues: 3

- **ggml-cpu.quants** → `core.GGMLQuants`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 4

## Missing Files (Top by Dependents)

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `ggml-backend-impl` | 47 | `src/ggml-backend-impl.h` |
| 2 | `ggml-cpu.common` | 47 | `src/ggml-cpu/common.h` |
| 3 | `include.ggml-backend` | 38 | `include/ggml-backend.h` |
| 4 | `ggml-sycl.fattn-vec` | 37 | `src/ggml-sycl/fattn-vec.hpp` |
| 5 | `ggml-zdnn.utils` | 23 | `src/ggml-zdnn/utils.hpp` |
| 6 | `openvino.node_context` | 19 | `src/ggml-openvino/openvino/node_context.h` |
| 7 | `openvino.op_table` | 18 | `src/ggml-openvino/openvino/op_table.h` |
| 8 | `ggml-sycl.convert` | 17 | `src/ggml-sycl/convert.hpp` |
| 9 | `ggml-common` | 16 | `src/ggml-common.h` |
| 10 | `htp.hvx-base` | 11 | `src/ggml-hexagon/htp/hvx-base.h` |
| 11 | `ggml-cpu.ggml-cpu-impl` | 11 | `src/ggml-cpu/ggml-cpu-impl.h` |
| 12 | `ggml-sycl.fattn-tile` | 11 | `src/ggml-sycl/fattn-tile.hpp` |
| 13 | `ggml-cpu.simd-mappings` | 10 | `src/ggml-cpu/simd-mappings.h` |
| 14 | `op.reshape` | 9 | `src/ggml-openvino/openvino/op/reshape.cpp` |
| 15 | `ggml-sycl.concat` | 9 | `src/ggml-sycl/concat.hpp` |
| 16 | `ggml-cpu.traits` | 9 | `src/ggml-cpu/traits.h` |
| 17 | `htp.hex-utils` | 8 | `src/ggml-hexagon/htp/hex-utils.h` |
| 18 | `ggml-sycl.presets` | 8 | `src/ggml-sycl/presets.hpp` |
| 19 | `ggml-virtgpu.ggml-remoting` | 8 | `src/ggml-virtgpu/ggml-remoting.h` |
| 20 | `op.transpose` | 7 | `src/ggml-openvino/openvino/op/transpose.cpp` |

... and 242 more missing files.

