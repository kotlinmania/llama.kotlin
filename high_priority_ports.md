# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `dpct.helper` | `core.QuantizationHelper` | 0.00 | 9 | 192 | 2815.7 |
| 2 | `ggml-cpu.vec` | `fp.VectorOps` | 0.00 | 1 | 106 | 1210.4 |
| 3 | `gguf` | `gguf.GGUFParser` | 0.01 | 1 | 97 | 1107.9 |
| 4 | `ggml-sycl.common` | `common.StatOps` | 0.00 | 0 | 57 | 570.0 |
| 5 | `ggml-opt` | `core.GGMLOptimizationSchedulerTest` | 0.00 | 0 | 57 | 570.0 |
| 6 | `ggml-cann.common` | `nativeMain.kotlin.ai.solace.klang.common.ZlibLoggerNative` | 0.00 | 0 | 37 | 370.0 |
| 7 | `kleidiai.kernels` | `bench.ShiftKernels` | 0.00 | 1 | 31 | 356.4 |
| 8 | `ggml-sycl.fattn-common` | `common.Constants` | 0.00 | 3 | 27 | 351.8 |
| 9 | `ggml-cpu.simd-mappings` | `simd.GGMLSimd` | 0.00 | 10 | 20 | 307.9 |
| 10 | `ggml-openvino.utils` | `checksum.Adler32Utils` | 0.00 | 0 | 27 | 270.0 |
| 11 | `ggml-metal.ggml-metal-common` | `commonMain.kotlin.ai.solace.klang.common.ZlibLogger` | 0.01 | 1 | 23 | 265.3 |
| 12 | `openvino.utils` | `core.GGMLTestUtils` | 0.00 | 0 | 17 | 170.0 |
| 13 | `ggml-zdnn.utils` | `util.BitUtils` | 0.00 | 23 | 4 | 81.3 |
| 14 | `ggml-sycl.set_rows` | `core.GGMLTensorUtils` | 0.00 | 2 | 6 | 78.7 |
| 15 | `openvino.input_model` | `model.Sampling` | 0.00 | 3 | 5 | 70.8 |
| 16 | `backend.backend-dispatched-buffer` | `buffer.LimbBuffer` | 0.00 | 0 | 7 | 70.0 |
| 17 | `backend.backend-dispatched-buffer-type` | `buffer.MemoryOps` | 0.00 | 0 | 6 | 60.0 |
| 18 | `amx.common` | `common.ZlibLoggerNative` | 0.00 | 0 | 6 | 60.0 |
| 19 | `backend.backend` | `core.GGMLBackendTest` | 0.00 | 0 | 6 | 60.0 |
| 20 | `ggml-sycl.type` | `gguf.GGUFTypes` | 0.00 | 2 | 4 | 54.3 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **dpct.helper** → `core.QuantizationHelper`
  - Similarity: 0.00
  - Dependencies: 9

- **ggml-cpu.vec** → `fp.VectorOps`
  - Similarity: 0.00
  - Dependencies: 1

- **gguf** → `gguf.GGUFParser`
  - Similarity: 0.01
  - Dependencies: 1
  - Lint issues: 7

- **kleidiai.kernels** → `bench.ShiftKernels`
  - Similarity: 0.00
  - Dependencies: 1

- **ggml-sycl.fattn-common** → `common.Constants`
  - Similarity: 0.00
  - Dependencies: 3

- **ggml-cpu.simd-mappings** → `simd.GGMLSimd`
  - Similarity: 0.00
  - Dependencies: 10

- **ggml-metal.ggml-metal-common** → `commonMain.kotlin.ai.solace.klang.common.ZlibLogger`
  - Similarity: 0.01
  - Dependencies: 1

- **ggml-zdnn.utils** → `util.BitUtils`
  - Similarity: 0.00
  - Dependencies: 23

- **ggml-sycl.set_rows** → `core.GGMLTensorUtils`
  - Similarity: 0.00
  - Dependencies: 2

- **openvino.input_model** → `model.Sampling`
  - Similarity: 0.00
  - Dependencies: 3

- **ggml-sycl.type** → `gguf.GGUFTypes`
  - Similarity: 0.00
  - Dependencies: 2

- **ggml-common** → `core.GGMLCommon`
  - Similarity: 0.00
  - Dependencies: 16
  - TODOs: 6

- **ggml-cpu.common** → `core.GGMLCpuCommon`
  - Similarity: 0.56
  - Dependencies: 47

- **ggml-impl** → `core.NumericConversions`
  - Similarity: 0.00
  - Dependencies: 58
  - Lint issues: 6

- **ggml** → `core.GGMLOps`
  - Similarity: 0.00
  - Dependencies: 50
  - Lint issues: 114

- **ggml-backend-impl** → `core.GGMLBackendImpl`
  - Similarity: 0.00
  - Dependencies: 47
  - Lint issues: 54

- **ggml-backend** → `core.GGMLBackendUtils`
  - Similarity: 0.00
  - Dependencies: 20
  - Lint issues: 39

- **ggml-cpu.ggml-cpu** → `core.GGMLCpuBackend`
  - Similarity: 0.00
  - Dependencies: 12
  - Lint issues: 2

- **ggml-cpu.ggml-cpu-impl** → `core.GGMLCpuImpl`
  - Similarity: 0.00
  - Dependencies: 11
  - Lint issues: 5

- **ggml-cpu.quants** → `core.GGMLQuants`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 87

- **ggml-sycl.backend** → `core.GGMLBackendIntegrationTest`
  - Similarity: 0.03
  - Dependencies: 2

- **ggml-cann.aclnn_ops** → `lnn.LNNCore`
  - Similarity: 0.00
  - Dependencies: 1
  - TODOs: 5
  - Lint issues: 2

- **ggml-threading** → `core.GGMLScheduler`
  - Similarity: 0.09
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
| 7 | `ggml-sycl.concat` | 9 | `ggml-sycl/concat.hpp` |
| 8 | `ggml-cpu.traits` | 9 | `ggml-cpu/traits.h` |
| 9 | `op.reshape` | 9 | `ggml-openvino/openvino/op/reshape.cpp` |
| 10 | `ggml-virtgpu.ggml-remoting` | 8 | `ggml-virtgpu/ggml-remoting.h` |
| 11 | `htp.hex-utils` | 8 | `ggml-hexagon/htp/hex-utils.h` |
| 12 | `ggml-sycl.presets` | 8 | `ggml-sycl/presets.hpp` |
| 13 | `op.transpose` | 7 | `ggml-openvino/openvino/op/transpose.cpp` |
| 14 | `shared.apir_backend` | 6 | `ggml-virtgpu/backend/shared/apir_backend.h` |
| 15 | `backend.backend-virgl-apir` | 6 | `ggml-virtgpu/backend/backend-virgl-apir.h` |
| 16 | `ggml-quants` | 6 | `ggml-quants.h` |
| 17 | `ggml-sycl.set` | 6 | `ggml-sycl/set.hpp` |
| 18 | `backend.backend-dispatched` | 5 | `ggml-virtgpu/backend/backend-dispatched.h` |
| 19 | `ggml-sycl.dequantize` | 5 | `ggml-sycl/dequantize.hpp` |
| 20 | `shared.apir_cs` | 4 | `ggml-virtgpu/backend/shared/apir_cs.h` |

... and 200 more missing files.

