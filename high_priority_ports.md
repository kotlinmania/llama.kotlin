# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `ggml-backend-impl` | `GGMLBackendImpl` | 0.36 | 47 | 19 | 349.4 |
| 2 | `ggml-cpu.simd-mappings` | `simd.GGMLSimd` | 0.00 | 10 | 20 | 307.9 |
| 3 | `ggml-backend` | `GGMLBackendUtils` | 0.72 | 20 | 18 | 293.9 |
| 4 | `openvino.utils` | `GGMLTensorUtils` | 0.00 | 0 | 17 | 170.0 |
| 5 | `ggml-impl` | `NumericConversions` | 0.54 | 58 | 8 | 154.6 |
| 6 | `ggml-cpu.traits` | `GGMLCpuTraits` | 0.19 | 9 | 9 | 140.8 |
| 7 | `ggml-virtgpu.ggml-backend` | `GGMLBackend` | 0.00 | 0 | 6 | 60.0 |
| 8 | `ggml-sycl.type` | `GGMLTypes` | 0.00 | 2 | 4 | 54.3 |
| 9 | `ggml-sycl.quants` | `GGMLCpuQuants` | 0.00 | 0 | 5 | 50.0 |
| 10 | `ggml-common` | `GGMLCommon` | 0.00 | 16 | 2 | 45.5 |
| 11 | `ggml-cpu.common` | `GGMLCpuCommon` | 0.56 | 47 | 2 | 43.9 |
| 12 | `ggml` | `GGMLOps` | 0.00 | 50 | 1 | 37.5 |
| 13 | `ggml-cpu.ggml-cpu` | `GGMLCpuBackend` | 0.00 | 12 | 0 | 12.8 |
| 14 | `ggml-cpu.ggml-cpu-impl` | `GGMLCpuImpl` | 0.00 | 11 | 0 | 12.4 |
| 15 | `ggml-quants` | `GGMLQuantsRef` | 0.00 | 6 | 0 | 9.7 |
| 16 | `ggml-cpu.quants` | `GGMLQuants` | 0.00 | 4 | 0 | 8.0 |
| 17 | `ggml-threading` | `GGMLScheduler` | 0.10 | 1 | 0 | 3.1 |
| 18 | `ggml-cpu.ops` | `GGMLComputeOps` | 0.00 | 0 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **ggml-backend-impl** → `GGMLBackendImpl`
  - Similarity: 0.36
  - Dependencies: 47
  - Lint issues: 49

- **ggml-cpu.simd-mappings** → `simd.GGMLSimd`
  - Similarity: 0.00
  - Dependencies: 10

- **ggml-impl** → `NumericConversions`
  - Similarity: 0.54
  - Dependencies: 58
  - Lint issues: 10

- **ggml-cpu.traits** → `GGMLCpuTraits`
  - Similarity: 0.19
  - Dependencies: 9

- **ggml-sycl.type** → `GGMLTypes`
  - Similarity: 0.00
  - Dependencies: 2

- **ggml-common** → `GGMLCommon`
  - Similarity: 0.00
  - Dependencies: 16

- **ggml-cpu.common** → `GGMLCpuCommon`
  - Similarity: 0.56
  - Dependencies: 47

- **ggml** → `GGMLOps`
  - Similarity: 0.00
  - Dependencies: 50
  - Lint issues: 123

- **ggml-cpu.ggml-cpu** → `GGMLCpuBackend`
  - Similarity: 0.00
  - Dependencies: 12
  - Lint issues: 2

- **ggml-cpu.ggml-cpu-impl** → `GGMLCpuImpl`
  - Similarity: 0.00
  - Dependencies: 11
  - TODOs: 1

- **ggml-quants** → `GGMLQuantsRef`
  - Similarity: 0.00
  - Dependencies: 6
  - Lint issues: 234

- **ggml-cpu.quants** → `GGMLQuants`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 115

- **ggml-threading** → `GGMLScheduler`
  - Similarity: 0.10
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
| 8 | `op.reshape` | 9 | `ggml-openvino/openvino/op/reshape.cpp` |
| 9 | `dpct.helper` | 9 | `ggml-sycl/dpct/helper.hpp` |
| 10 | `ggml-sycl.concat` | 9 | `ggml-sycl/concat.hpp` |
| 11 | `ggml-virtgpu.ggml-remoting` | 8 | `ggml-virtgpu/ggml-remoting.h` |
| 12 | `htp.hex-utils` | 8 | `ggml-hexagon/htp/hex-utils.h` |
| 13 | `ggml-sycl.presets` | 8 | `ggml-sycl/presets.hpp` |
| 14 | `op.transpose` | 7 | `ggml-openvino/openvino/op/transpose.cpp` |
| 15 | `shared.apir_backend` | 6 | `ggml-virtgpu/backend/shared/apir_backend.h` |
| 16 | `backend.backend-virgl-apir` | 6 | `ggml-virtgpu/backend/backend-virgl-apir.h` |
| 17 | `ggml-sycl.set` | 6 | `ggml-sycl/set.hpp` |
| 18 | `backend.backend-dispatched` | 5 | `ggml-virtgpu/backend/backend-dispatched.h` |
| 19 | `ggml-sycl.dequantize` | 5 | `ggml-sycl/dequantize.hpp` |
| 20 | `shared.apir_cs` | 4 | `ggml-virtgpu/backend/shared/apir_cs.h` |

... and 218 more missing files.

