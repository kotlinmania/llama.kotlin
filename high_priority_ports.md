# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `dpct.helper` | `QuantizationHelper` | 0.00 | 9 | 192 | 2815.7 |
| 2 | `openvino.utils` | `GGMLTensorUtils` | 0.00 | 0 | 17 | 170.0 |
| 3 | `ggml-sycl.type` | `GGMLTypes` | 0.00 | 2 | 4 | 54.3 |
| 4 | `ggml-cpu.simd-gemm` | `simd.GGMLSimd` | 0.00 | 1 | 4 | 49.0 |
| 5 | `ggml-common` | `GGMLCommon` | 0.00 | 16 | 2 | 45.5 |
| 6 | `ggml-cpu.common` | `GGMLCpuCommon` | 0.56 | 47 | 2 | 43.9 |
| 7 | `ggml-impl` | `NumericConversions` | 0.00 | 58 | 0 | 20.4 |
| 8 | `ggml` | `GGMLOps` | 0.00 | 50 | 0 | 19.7 |
| 9 | `ggml-backend-impl` | `GGMLBackendImpl` | 0.00 | 47 | 0 | 19.4 |
| 10 | `ggml-backend` | `GGMLBackendUtils` | 0.00 | 20 | 0 | 15.2 |
| 11 | `ggml-cpu.ggml-cpu` | `GGMLCpuBackend` | 0.00 | 12 | 0 | 12.8 |
| 12 | `ggml-cpu.ggml-cpu-impl` | `GGMLCpuImpl` | 0.00 | 11 | 0 | 12.4 |
| 13 | `ggml-cpu.quants` | `GGMLQuants` | 0.00 | 4 | 0 | 8.0 |
| 14 | `ggml-threading` | `GGMLScheduler` | 0.09 | 1 | 0 | 3.1 |
| 15 | `ggml-virtgpu.ggml-backend` | `GGMLBackend` | 0.00 | 0 | 0 | 0.0 |
| 16 | `ggml-cpu.ops` | `GGMLComputeOps` | 0.00 | 0 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **dpct.helper** → `QuantizationHelper`
  - Similarity: 0.00
  - Dependencies: 9

- **ggml-sycl.type** → `GGMLTypes`
  - Similarity: 0.00
  - Dependencies: 2

- **ggml-cpu.simd-gemm** → `simd.GGMLSimd`
  - Similarity: 0.00
  - Dependencies: 1

- **ggml-common** → `GGMLCommon`
  - Similarity: 0.00
  - Dependencies: 16
  - TODOs: 6

- **ggml-cpu.common** → `GGMLCpuCommon`
  - Similarity: 0.56
  - Dependencies: 47

- **ggml-impl** → `NumericConversions`
  - Similarity: 0.00
  - Dependencies: 58
  - Lint issues: 6

- **ggml** → `GGMLOps`
  - Similarity: 0.00
  - Dependencies: 50
  - Lint issues: 83

- **ggml-backend-impl** → `GGMLBackendImpl`
  - Similarity: 0.00
  - Dependencies: 47
  - Lint issues: 54

- **ggml-backend** → `GGMLBackendUtils`
  - Similarity: 0.00
  - Dependencies: 20
  - Lint issues: 39

- **ggml-cpu.ggml-cpu** → `GGMLCpuBackend`
  - Similarity: 0.00
  - Dependencies: 12
  - Lint issues: 2

- **ggml-cpu.ggml-cpu-impl** → `GGMLCpuImpl`
  - Similarity: 0.00
  - Dependencies: 11
  - Lint issues: 5

- **ggml-cpu.quants** → `GGMLQuants`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 87

- **ggml-threading** → `GGMLScheduler`
  - Similarity: 0.09
  - Dependencies: 1
  - Lint issues: 3

## Missing Files (Top by Dependents)

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `ggml-sycl.fattn-vec` | 37 | `ggml-sycl/fattn-vec.hpp` |
| 2 | `ggml-zdnn.utils` | 23 | `ggml-zdnn/utils.hpp` |
| 3 | `openvino.node_context` | 19 | `ggml-openvino/openvino/node_context.h` |
| 4 | `openvino.op_table` | 18 | `ggml-openvino/openvino/op_table.h` |
| 5 | `ggml-sycl.convert` | 17 | `ggml-sycl/convert.hpp` |
| 6 | `htp.hvx-base` | 11 | `ggml-hexagon/htp/hvx-base.h` |
| 7 | `ggml-sycl.fattn-tile` | 11 | `ggml-sycl/fattn-tile.hpp` |
| 8 | `ggml-cpu.simd-mappings` | 10 | `ggml-cpu/simd-mappings.h` |
| 9 | `ggml-sycl.concat` | 9 | `ggml-sycl/concat.hpp` |
| 10 | `ggml-cpu.traits` | 9 | `ggml-cpu/traits.h` |
| 11 | `op.reshape` | 9 | `ggml-openvino/openvino/op/reshape.cpp` |
| 12 | `ggml-sycl.presets` | 8 | `ggml-sycl/presets.hpp` |
| 13 | `ggml-virtgpu.ggml-remoting` | 8 | `ggml-virtgpu/ggml-remoting.h` |
| 14 | `htp.hex-utils` | 8 | `ggml-hexagon/htp/hex-utils.h` |
| 15 | `op.transpose` | 7 | `ggml-openvino/openvino/op/transpose.cpp` |
| 16 | `ggml-quants` | 6 | `ggml-quants.h` |
| 17 | `backend.backend-virgl-apir` | 6 | `ggml-virtgpu/backend/backend-virgl-apir.h` |
| 18 | `shared.apir_backend` | 6 | `ggml-virtgpu/backend/shared/apir_backend.h` |
| 19 | `ggml-sycl.set` | 6 | `ggml-sycl/set.hpp` |
| 20 | `backend.backend-dispatched` | 5 | `ggml-virtgpu/backend/backend-dispatched.h` |

... and 220 more missing files.

