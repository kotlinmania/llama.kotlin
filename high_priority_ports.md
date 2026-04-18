# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `dpct.helper` | `core.QuantizationHelper` | 0.00 | 9 | 192 | 2815.7 |
| 2 | `openvino.utils` | `core.GGMLTensorUtils` | 0.00 | 0 | 17 | 170.0 |
| 3 | `ggml-cpu.simd-gemm` | `simd.GGMLSimd` | 0.00 | 1 | 4 | 49.0 |
| 4 | `ggml-common` | `core.GGMLCommon` | 0.00 | 16 | 2 | 45.5 |
| 5 | `ggml-cpu.common` | `core.GGMLCpuCommon` | 0.56 | 47 | 2 | 43.9 |
| 6 | `ggml-threading` | `core.GGMLScheduler` | 0.00 | 1 | 2 | 26.2 |
| 7 | `include.ggml` | `core.GGMLOps` | 0.00 | 68 | 0 | 21.2 |
| 8 | `ggml-impl` | `core.NumericConversions` | 0.00 | 58 | 0 | 20.4 |
| 9 | `ggml-backend-impl` | `core.GGMLBackendImpl` | 0.00 | 47 | 0 | 19.4 |
| 10 | `include.ggml-backend` | `core.GGMLBackend` | 0.00 | 38 | 0 | 18.3 |
| 11 | `include.ggml-cpu` | `core.GGMLCpuExecutor` | 0.00 | 13 | 0 | 13.2 |
| 12 | `ggml-cpu.ggml-cpu-impl` | `core.GGMLCpuImpl` | 0.00 | 11 | 0 | 12.4 |
| 13 | `include.ggml-alloc` | `core.GGMLAlloc` | 0.00 | 8 | 0 | 11.0 |
| 14 | `ggml` | `core.GGMLTypes` | 0.00 | 0 | 1 | 10.0 |
| 15 | `ggml-cpu.quants` | `core.GGMLQuants` | 0.00 | 4 | 0 | 8.0 |
| 16 | `ggml-cpu.ggml-cpu` | `core.GGMLCpuBackend` | 0.00 | 0 | 0 | 0.0 |
| 17 | `ggml-backend` | `core.GGMLBackendUtils` | 0.00 | 0 | 0 | 0.0 |
| 18 | `ggml-cpu.ops` | `core.GGMLComputeOps` | 0.00 | 0 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **dpct.helper** → `core.QuantizationHelper`
  - Similarity: 0.00
  - Dependencies: 9

- **ggml-cpu.simd-gemm** → `simd.GGMLSimd`
  - Similarity: 0.00
  - Dependencies: 1

- **ggml-common** → `core.GGMLCommon`
  - Similarity: 0.00
  - Dependencies: 16
  - TODOs: 6

- **ggml-cpu.common** → `core.GGMLCpuCommon`
  - Similarity: 0.56
  - Dependencies: 47

- **ggml-threading** → `core.GGMLScheduler`
  - Similarity: 0.00
  - Dependencies: 1
  - Lint issues: 3

- **include.ggml** → `core.GGMLOps`
  - Similarity: 0.00
  - Dependencies: 68
  - Lint issues: 83

- **ggml-impl** → `core.NumericConversions`
  - Similarity: 0.00
  - Dependencies: 58
  - Lint issues: 6

- **ggml-backend-impl** → `core.GGMLBackendImpl`
  - Similarity: 0.00
  - Dependencies: 47
  - Lint issues: 54

- **include.ggml-backend** → `core.GGMLBackend`
  - Similarity: 0.00
  - Dependencies: 38
  - Lint issues: 65

- **include.ggml-cpu** → `core.GGMLCpuExecutor`
  - Similarity: 0.00
  - Dependencies: 13
  - Lint issues: 68

- **ggml-cpu.ggml-cpu-impl** → `core.GGMLCpuImpl`
  - Similarity: 0.00
  - Dependencies: 11
  - Lint issues: 5

- **include.ggml-alloc** → `core.GGMLAlloc`
  - Similarity: 0.00
  - Dependencies: 8
  - Lint issues: 13

- **ggml-cpu.quants** → `core.GGMLQuants`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 87

## Missing Files (Top by Dependents)

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `ggml-sycl.fattn-vec` | 37 | `src/ggml-sycl/fattn-vec.hpp` |
| 2 | `ggml-zdnn.utils` | 23 | `src/ggml-zdnn/utils.hpp` |
| 3 | `openvino.node_context` | 19 | `src/ggml-openvino/openvino/node_context.h` |
| 4 | `openvino.op_table` | 18 | `src/ggml-openvino/openvino/op_table.h` |
| 5 | `ggml-sycl.convert` | 17 | `src/ggml-sycl/convert.hpp` |
| 6 | `htp.hvx-base` | 11 | `src/ggml-hexagon/htp/hvx-base.h` |
| 7 | `ggml-sycl.fattn-tile` | 11 | `src/ggml-sycl/fattn-tile.hpp` |
| 8 | `ggml-cpu.simd-mappings` | 10 | `src/ggml-cpu/simd-mappings.h` |
| 9 | `op.reshape` | 9 | `src/ggml-openvino/openvino/op/reshape.cpp` |
| 10 | `ggml-sycl.concat` | 9 | `src/ggml-sycl/concat.hpp` |
| 11 | `ggml-cpu.traits` | 9 | `src/ggml-cpu/traits.h` |
| 12 | `ggml-sycl.presets` | 8 | `src/ggml-sycl/presets.hpp` |
| 13 | `ggml-virtgpu.ggml-remoting` | 8 | `src/ggml-virtgpu/ggml-remoting.h` |
| 14 | `htp.hex-utils` | 8 | `src/ggml-hexagon/htp/hex-utils.h` |
| 15 | `op.transpose` | 7 | `src/ggml-openvino/openvino/op/transpose.cpp` |
| 16 | `ggml-sycl.set` | 6 | `src/ggml-sycl/set.hpp` |
| 17 | `backend.backend-virgl-apir` | 6 | `src/ggml-virtgpu/backend/backend-virgl-apir.h` |
| 18 | `shared.apir_backend` | 6 | `src/ggml-virtgpu/backend/shared/apir_backend.h` |
| 19 | `ggml-quants` | 6 | `src/ggml-quants.h` |
| 20 | `backend.backend-dispatched` | 5 | `src/ggml-virtgpu/backend/backend-dispatched.h` |

... and 239 more missing files.

