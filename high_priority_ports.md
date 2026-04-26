# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `include.ggml-backend` | `core.GGMLBackend` | 0.00 | 38 | 91 | 1595.1 |
| 2 | `include.ggml` | `core.GGMLOps` | 0.46 | 68 | 29 | 547.1 |
| 3 | `ggml-cpu.quants` | `core.GGMLQuants` | 0.10 | 4 | 27 | 364.2 |
| 4 | `ggml-backend-impl` | `core.GGMLBackendImpl` | 0.55 | 47 | 17 | 310.4 |
| 5 | `ggml-impl` | `core.NumericConversions` | 0.52 | 58 | 10 | 191.4 |
| 6 | `openvino.utils` | `core.GGMLTensorUtils` | 0.00 | 0 | 17 | 170.0 |
| 7 | `include.ggml-cpu` | `core.GGMLCpuExecutor` | 0.25 | 13 | 7 | 116.9 |
| 8 | `ggml-cpu.ggml-cpu-impl` | `core.GGMLCpuImpl` | 0.25 | 11 | 6 | 99.2 |
| 9 | `include.ggml-alloc` | `core.GGMLAlloc` | 0.22 | 8 | 6 | 94.9 |
| 10 | `ggml-cpu.traits` | `core.GGMLCpuTraits` | 0.64 | 9 | 5 | 77.2 |
| 11 | `ggml-sycl.quants` | `core.GGMLCpuQuants` | 0.00 | 0 | 5 | 50.0 |
| 12 | `ggml-common` | `core.GGMLCommon` | 0.00 | 16 | 2 | 45.5 |
| 13 | `ggml-cpu.common` | `core.GGMLCpuCommon` | 0.56 | 47 | 2 | 43.9 |
| 14 | `ggml-quants` | `core.GGMLQuantsRef` | 0.12 | 6 | 1 | 22.5 |
| 15 | `ggml` | `core.GGMLTypes` | 0.00 | 0 | 1 | 10.0 |
| 16 | `ggml-cpu.simd-mappings` | `simd.GGMLSimd` | 0.17 | 10 | 0 | 10.0 |
| 17 | `ggml-threading` | `core.GGMLScheduler` | 0.10 | 1 | 0 | 3.1 |
| 18 | `ggml-cpu.ggml-cpu` | `core.GGMLCpuBackend` | 0.00 | 0 | 0 | 0.0 |
| 19 | `ggml-backend` | `core.GGMLBackendUtils` | 0.00 | 0 | 0 | 0.0 |
| 20 | `ggml-cpu.ops` | `core.GGMLComputeOps` | 0.00 | 0 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **include.ggml-backend** → `core.GGMLBackend`
  - Similarity: 0.00
  - Dependencies: 38
  - Lint issues: 22

- **include.ggml** → `core.GGMLOps`
  - Similarity: 0.46
  - Dependencies: 68
  - Lint issues: 90

- **ggml-cpu.quants** → `core.GGMLQuants`
  - Similarity: 0.10
  - Dependencies: 4

- **ggml-backend-impl** → `core.GGMLBackendImpl`
  - Similarity: 0.55
  - Dependencies: 47
  - Lint issues: 49

- **ggml-impl** → `core.NumericConversions`
  - Similarity: 0.52
  - Dependencies: 58
  - Lint issues: 8

- **include.ggml-cpu** → `core.GGMLCpuExecutor`
  - Similarity: 0.25
  - Dependencies: 13
  - Lint issues: 2

- **ggml-cpu.ggml-cpu-impl** → `core.GGMLCpuImpl`
  - Similarity: 0.25
  - Dependencies: 11

- **include.ggml-alloc** → `core.GGMLAlloc`
  - Similarity: 0.22
  - Dependencies: 8
  - Lint issues: 15

- **ggml-common** → `core.GGMLCommon`
  - Similarity: 0.00
  - Dependencies: 16

- **ggml-cpu.common** → `core.GGMLCpuCommon`
  - Similarity: 0.56
  - Dependencies: 47

- **ggml-quants** → `core.GGMLQuantsRef`
  - Similarity: 0.12
  - Dependencies: 6
  - Lint issues: 10

- **ggml-cpu.simd-mappings** → `simd.GGMLSimd`
  - Similarity: 0.17
  - Dependencies: 10

- **ggml-threading** → `core.GGMLScheduler`
  - Similarity: 0.10
  - Dependencies: 1
  - Lint issues: 3

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
| 8 | `op.reshape` | 9 | `src/ggml-openvino/openvino/op/reshape.cpp` |
| 9 | `ggml-sycl.concat` | 9 | `src/ggml-sycl/concat.hpp` |
| 10 | `dpct.helper` | 9 | `src/ggml-sycl/dpct/helper.hpp` |
| 11 | `htp.hex-utils` | 8 | `src/ggml-hexagon/htp/hex-utils.h` |
| 12 | `ggml-virtgpu.ggml-remoting` | 8 | `src/ggml-virtgpu/ggml-remoting.h` |
| 13 | `ggml-sycl.presets` | 8 | `src/ggml-sycl/presets.hpp` |
| 14 | `op.transpose` | 7 | `src/ggml-openvino/openvino/op/transpose.cpp` |
| 15 | `ggml-sycl.set` | 6 | `src/ggml-sycl/set.hpp` |
| 16 | `shared.apir_backend` | 6 | `src/ggml-virtgpu/backend/shared/apir_backend.h` |
| 17 | `backend.backend-virgl-apir` | 6 | `src/ggml-virtgpu/backend/backend-virgl-apir.h` |
| 18 | `ggml-sycl.dequantize` | 5 | `src/ggml-sycl/dequantize.hpp` |
| 19 | `backend.backend-dispatched` | 5 | `src/ggml-virtgpu/backend/backend-dispatched.h` |
| 20 | `ggml-cpu.repack` | 4 | `src/ggml-cpu/repack.h` |

... and 237 more missing files.

