# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `gguf` | `gguf.GGUFParser` | 0.01 | 1 | 97 | 1107.9 |
| 2 | `ggml-cpu.quants` | `core.GGMLCpuQuants` | 0.07 | 4 | 51 | 681.7 |
| 3 | `ggml-sycl.common` | `common.StatOps` | 0.00 | 0 | 57 | 570.0 |
| 4 | `ggml-opt` | `core.GGMLOptimizationSchedulerTest` | 0.00 | 0 | 57 | 570.0 |
| 5 | `ggml-cann.common` | `nativeMain.kotlin.ai.solace.klang.common.ZlibLoggerNative` | 0.00 | 0 | 37 | 370.0 |
| 6 | `kleidiai.kernels` | `bench.ShiftKernels` | 0.00 | 1 | 31 | 356.4 |
| 7 | `ggml-sycl.fattn-common` | `commonMain.kotlin.ai.solace.klang.common.ZlibLogger` | 0.00 | 3 | 27 | 351.8 |
| 8 | `ggml-backend-impl` | `core.GGMLBackendImpl` | 0.36 | 47 | 19 | 349.4 |
| 9 | `ggml-cpu.simd-mappings` | `simd.GGMLSimd` | 0.00 | 10 | 20 | 307.9 |
| 10 | `ggml-backend` | `core.GGMLBackendUtils` | 0.72 | 20 | 18 | 293.9 |
| 11 | `ggml-openvino.utils` | `checksum.Adler32Utils` | 0.00 | 0 | 27 | 270.0 |
| 12 | `ggml-metal.ggml-metal-common` | `common.ZlibLogger` | 0.01 | 1 | 23 | 265.3 |
| 13 | `openvino.utils` | `core.GGMLTestUtils` | 0.00 | 0 | 17 | 170.0 |
| 14 | `ggml-impl` | `core.NumericConversions` | 0.54 | 58 | 8 | 154.6 |
| 15 | `ggml-cpu.traits` | `core.GGMLCpuTraits` | 0.19 | 9 | 9 | 140.8 |
| 16 | `ggml-cpu.vec` | `fp.VectorOps` | 0.21 | 1 | 11 | 128.0 |
| 17 | `ggml-zdnn.utils` | `util.BitUtils` | 0.00 | 23 | 4 | 81.3 |
| 18 | `ggml-sycl.set_rows` | `core.GGMLTensorUtils` | 0.00 | 2 | 6 | 78.7 |
| 19 | `openvino.input_model` | `model.IntegrationTest` | 0.00 | 3 | 5 | 70.8 |
| 20 | `backend.backend-dispatched-buffer` | `buffer.LimbBuffer` | 0.00 | 0 | 7 | 70.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **gguf** → `gguf.GGUFParser`
  - Similarity: 0.01
  - Dependencies: 1
  - Lint issues: 8

- **ggml-cpu.quants** → `core.GGMLCpuQuants`
  - Similarity: 0.07
  - Dependencies: 4
  - Lint issues: 87

- **kleidiai.kernels** → `bench.ShiftKernels`
  - Similarity: 0.00
  - Dependencies: 1

- **ggml-sycl.fattn-common** → `commonMain.kotlin.ai.solace.klang.common.ZlibLogger`
  - Similarity: 0.00
  - Dependencies: 3

- **ggml-backend-impl** → `core.GGMLBackendImpl`
  - Similarity: 0.36
  - Dependencies: 47
  - Lint issues: 49

- **ggml-cpu.simd-mappings** → `simd.GGMLSimd`
  - Similarity: 0.00
  - Dependencies: 10

- **ggml-metal.ggml-metal-common** → `common.ZlibLogger`
  - Similarity: 0.01
  - Dependencies: 1

- **ggml-impl** → `core.NumericConversions`
  - Similarity: 0.54
  - Dependencies: 58
  - Lint issues: 10

- **ggml-cpu.traits** → `core.GGMLCpuTraits`
  - Similarity: 0.19
  - Dependencies: 9

- **ggml-cpu.vec** → `fp.VectorOps`
  - Similarity: 0.21
  - Dependencies: 1
  - Lint issues: 9

- **ggml-zdnn.utils** → `util.BitUtils`
  - Similarity: 0.00
  - Dependencies: 23

- **ggml-sycl.set_rows** → `core.GGMLTensorUtils`
  - Similarity: 0.00
  - Dependencies: 2

- **openvino.input_model** → `model.IntegrationTest`
  - Similarity: 0.00
  - Dependencies: 3

- **ggml-sycl.type** → `core.GGMLTypes`
  - Similarity: 0.00
  - Dependencies: 2

- **ggml-common** → `core.GGMLCommon`
  - Similarity: 0.00
  - Dependencies: 16

- **ggml-cpu.common** → `core.GGMLCpuCommon`
  - Similarity: 0.56
  - Dependencies: 47

- **ggml** → `core.GGMLOps`
  - Similarity: 0.00
  - Dependencies: 50
  - Lint issues: 123

- **ggml-cpu.ggml-cpu** → `core.GGMLCpuBackend`
  - Similarity: 0.00
  - Dependencies: 12
  - Lint issues: 2

- **ggml-cpu.ggml-cpu-impl** → `core.GGMLCpuImpl`
  - Similarity: 0.00
  - Dependencies: 11
  - TODOs: 1

- **ggml-quants** → `core.GGMLQuantsRef`
  - Similarity: 0.00
  - Dependencies: 6
  - Lint issues: 234

- **ggml-sycl.backend** → `core.GGMLBackendIntegrationTest`
  - Similarity: 0.04
  - Dependencies: 2

- **ggml-cann.aclnn_ops** → `lnn.LNNActors`
  - Similarity: 0.00
  - Dependencies: 1
  - TODOs: 9
  - Lint issues: 3

- **ggml-threading** → `core.GGMLScheduler`
  - Similarity: 0.10
  - Dependencies: 1
  - Lint issues: 3

## Missing Files (Top by Dependents)

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `ggml-sycl.fattn-vec` | 37 | `ggml-sycl/fattn-vec.hpp` |
| 2 | `openvino.node_context` | 19 | `ggml-openvino/openvino/node_context.h` |
| 3 | `openvino.op_table` | 18 | `ggml-openvino/openvino/op_table.h` |
| 4 | `ggml-sycl.convert` | 17 | `ggml-sycl/convert.hpp` |
| 5 | `htp.hvx-base` | 11 | `ggml-hexagon/htp/hvx-base.h` |
| 6 | `ggml-sycl.fattn-tile` | 11 | `ggml-sycl/fattn-tile.hpp` |
| 7 | `op.reshape` | 9 | `ggml-openvino/openvino/op/reshape.cpp` |
| 8 | `ggml-sycl.concat` | 9 | `ggml-sycl/concat.hpp` |
| 9 | `dpct.helper` | 9 | `ggml-sycl/dpct/helper.hpp` |
| 10 | `ggml-sycl.presets` | 8 | `ggml-sycl/presets.hpp` |
| 11 | `ggml-virtgpu.ggml-remoting` | 8 | `ggml-virtgpu/ggml-remoting.h` |
| 12 | `htp.hex-utils` | 8 | `ggml-hexagon/htp/hex-utils.h` |
| 13 | `op.transpose` | 7 | `ggml-openvino/openvino/op/transpose.cpp` |
| 14 | `shared.apir_backend` | 6 | `ggml-virtgpu/backend/shared/apir_backend.h` |
| 15 | `backend.backend-virgl-apir` | 6 | `ggml-virtgpu/backend/backend-virgl-apir.h` |
| 16 | `ggml-sycl.set` | 6 | `ggml-sycl/set.hpp` |
| 17 | `backend.backend-dispatched` | 5 | `ggml-virtgpu/backend/backend-dispatched.h` |
| 18 | `ggml-sycl.dequantize` | 5 | `ggml-sycl/dequantize.hpp` |
| 19 | `ggml-openvino.ggml-openvino-extra` | 4 | `ggml-openvino/ggml-openvino-extra.h` |
| 20 | `ggml-virtgpu.virtgpu-forward-impl` | 4 | `ggml-virtgpu/virtgpu-forward-impl.h` |

... and 198 more missing files.

