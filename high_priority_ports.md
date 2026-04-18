# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `dpct.helper` | `core.QuantizationHelper` | 0.00 | 9 | 192 | 2815.7 |
| 2 | `ggml-backend` | `core.GGMLBackendUtils` | 0.00 | 0 | 147 | 1470.0 |
| 3 | `include.ggml-cpu` | `core.GGMLCpuExecutor` | 0.00 | 13 | 71 | 1097.9 |
| 4 | `ggml-cpu.ggml-cpu` | `core.GGMLCpuBackend` | 0.00 | 0 | 46 | 460.0 |
| 5 | `ggml-impl` | `core.NumericConversions` | 0.34 | 58 | 24 | 449.2 |
| 6 | `include.ggml-alloc` | `core.GGMLAlloc` | 0.00 | 8 | 19 | 284.5 |
| 7 | `openvino.utils` | `core.GGMLTensorUtils` | 0.00 | 0 | 17 | 170.0 |
| 8 | `ggml-sycl.type` | `core.GGMLTypes` | 0.00 | 2 | 4 | 54.3 |
| 9 | `ggml-cpu.simd-gemm` | `simd.GGMLSimd` | 0.00 | 1 | 4 | 49.0 |
| 10 | `ggml-common` | `core.GGMLCommon` | 0.00 | 16 | 2 | 45.5 |
| 11 | `ggml-threading` | `core.GGMLScheduler` | 0.00 | 1 | 2 | 26.2 |
| 12 | `include.ggml` | `core.GGMLOps` | 0.00 | 68 | 0 | 21.2 |
| 13 | `ggml-backend-impl` | `core.GGMLBackendImpl` | 0.00 | 47 | 0 | 19.4 |
| 14 | `include.ggml-backend` | `core.GGMLBackend` | 0.00 | 38 | 0 | 18.3 |
| 15 | `ggml-cpu.quants` | `core.GGMLQuants` | 0.00 | 4 | 0 | 8.0 |
| 16 | `ggml` | `core.GGMLGraph` | 0.00 | 0 | 0 | 0.0 |
| 17 | `ggml-cpu.ops` | `core.GGMLComputeOps` | 0.00 | 0 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **dpct.helper** → `core.QuantizationHelper`
  - Similarity: 0.00
  - Dependencies: 9

- **include.ggml-cpu** → `core.GGMLCpuExecutor`
  - Similarity: 0.00
  - Dependencies: 13
  - Lint issues: 1

- **ggml-impl** → `core.NumericConversions`
  - Similarity: 0.34
  - Dependencies: 58

- **include.ggml-alloc** → `core.GGMLAlloc`
  - Similarity: 0.00
  - Dependencies: 8

- **ggml-sycl.type** → `core.GGMLTypes`
  - Similarity: 0.00
  - Dependencies: 2

- **ggml-cpu.simd-gemm** → `simd.GGMLSimd`
  - Similarity: 0.00
  - Dependencies: 1

- **ggml-common** → `core.GGMLCommon`
  - Similarity: 0.00
  - Dependencies: 16
  - TODOs: 6

- **ggml-threading** → `core.GGMLScheduler`
  - Similarity: 0.00
  - Dependencies: 1
  - Lint issues: 3

- **include.ggml** → `core.GGMLOps`
  - Similarity: 0.00
  - Dependencies: 68
  - Lint issues: 83

- **ggml-backend-impl** → `core.GGMLBackendImpl`
  - Similarity: 0.00
  - Dependencies: 47
  - Lint issues: 54

- **include.ggml-backend** → `core.GGMLBackend`
  - Similarity: 0.00
  - Dependencies: 38
  - Lint issues: 65

- **ggml-cpu.quants** → `core.GGMLQuants`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 4

## Missing Files (Top by Dependents)

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `ggml-cpu.common` | 47 | `src/ggml-cpu/common.h` |
| 2 | `ggml-sycl.fattn-vec` | 37 | `src/ggml-sycl/fattn-vec.hpp` |
| 3 | `ggml-zdnn.utils` | 23 | `src/ggml-zdnn/utils.hpp` |
| 4 | `openvino.node_context` | 19 | `src/ggml-openvino/openvino/node_context.h` |
| 5 | `openvino.op_table` | 18 | `src/ggml-openvino/openvino/op_table.h` |
| 6 | `ggml-sycl.convert` | 17 | `src/ggml-sycl/convert.hpp` |
| 7 | `htp.hvx-base` | 11 | `src/ggml-hexagon/htp/hvx-base.h` |
| 8 | `ggml-cpu.ggml-cpu-impl` | 11 | `src/ggml-cpu/ggml-cpu-impl.h` |
| 9 | `ggml-sycl.fattn-tile` | 11 | `src/ggml-sycl/fattn-tile.hpp` |
| 10 | `ggml-cpu.simd-mappings` | 10 | `src/ggml-cpu/simd-mappings.h` |
| 11 | `ggml-cpu.traits` | 9 | `src/ggml-cpu/traits.h` |
| 12 | `op.reshape` | 9 | `src/ggml-openvino/openvino/op/reshape.cpp` |
| 13 | `ggml-sycl.concat` | 9 | `src/ggml-sycl/concat.hpp` |
| 14 | `ggml-sycl.presets` | 8 | `src/ggml-sycl/presets.hpp` |
| 15 | `htp.hex-utils` | 8 | `src/ggml-hexagon/htp/hex-utils.h` |
| 16 | `ggml-virtgpu.ggml-remoting` | 8 | `src/ggml-virtgpu/ggml-remoting.h` |
| 17 | `op.transpose` | 7 | `src/ggml-openvino/openvino/op/transpose.cpp` |
| 18 | `ggml-quants` | 6 | `src/ggml-quants.h` |
| 19 | `shared.apir_backend` | 6 | `src/ggml-virtgpu/backend/shared/apir_backend.h` |
| 20 | `ggml-sycl.set` | 6 | `src/ggml-sycl/set.hpp` |

... and 240 more missing files.

