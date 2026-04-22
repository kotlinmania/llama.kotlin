# Code Port - Progress Report

**Generated:** 2026-04-21
**Source:** tmp/llama.cpp/ggml/src
**Target:** src

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 256 | 100% |
| Target units (paired) | 158 | - |
| Target files (total) | 158 | - |
| Porting progress | 36 | 14.1% (matched) |
| Missing files | 220 | 85.9% |

## Port Quality Analysis

**Average Similarity:** 0.08

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 1 files (2.8% of matched)
- Critical (<0.60): 35 files (97.2% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `gguf` → `gguf.GGUFParser` (0.01, 1 deps)
- `ggml-cpu.quants` → `core.GGMLCpuQuants` (0.07, 4 deps)
- `ggml-sycl.common` → `common.StatOps` (0.00)
- `ggml-opt` → `core.GGMLOptimizationSchedulerTest` (0.00)
- `ggml-cann.common` → `nativeMain.kotlin.ai.solace.klang.common.ZlibLoggerNative` (0.00)
- `kleidiai.kernels` → `bench.ShiftKernels` (0.00, 1 deps)
- `ggml-sycl.fattn-common` → `commonMain.kotlin.ai.solace.klang.common.ZlibLogger` (0.00, 3 deps)
- `ggml-backend-impl` → `core.GGMLBackendImpl` (0.36, 47 deps)
- `ggml-cpu.simd-mappings` → `simd.GGMLSimd` (0.00, 10 deps)
- `ggml-metal.ggml-metal-common` → `common.Constants` (0.00, 1 deps)
- `ggml-openvino.utils` → `checksum.Adler32Utils` (0.00)
- `openvino.utils` → `core.GGMLTestUtils` (0.00)
- `ggml-impl` → `core.NumericConversions` (0.54, 58 deps)
- `ggml-cpu.vec` → `fp.VectorOps` (0.21, 1 deps)
- `ggml-zdnn.utils` → `util.BitUtils` (0.00, 23 deps)
- `ggml-sycl.set_rows` → `core.GGMLTensorUtils` (0.00, 2 deps)
- `openvino.input_model` → `model.IntegrationTest` (0.00, 3 deps)
- `backend.backend-dispatched-buffer` → `buffer.LimbBuffer` (0.00)
- `backend.backend` → `core.GGMLBackendTest` (0.00)
- `ggml-virtgpu.ggml-backend` → `core.GGMLBackend` (0.00)
- `amx.common` → `common.ZlibLoggerNative` (0.00)
- `backend.backend-dispatched-buffer-type` → `buffer.MemoryOps` (0.00)
- `ggml-sycl.type` → `gguf.GGUFTypes` (0.00, 2 deps)
- `ggml-common` → `core.GGMLCommon` (0.00, 16 deps)
- `ggml-cpu.common` → `core.GGMLCpuCommon` (0.56, 47 deps)
- `ggml-zdnn.common` → `common.ZlibLogger` (0.11)
- `ggml` → `core.GGMLOps` (0.00, 50 deps)
- `ggml-cpu.ggml-cpu` → `core.GGMLCpuBackend` (0.00, 12 deps)
- `ggml-cpu.ggml-cpu-impl` → `core.GGMLCpuImpl` (0.00, 11 deps)
- `op.cont` → `gguf.GGUFContext` (0.00)
- `ggml-sycl.backend` → `core.GGMLBackendIntegrationTest` (0.04, 2 deps)
- `ggml-cann.aclnn_ops` → `lnn.LNNActors` (0.00, 1 deps)
- `ggml-threading` → `core.GGMLScheduler` (0.10, 1 deps)
- `ggml-sycl.quants` → `core.GGMLQuants` (0.00)
- `ggml-cpu.ops` → `core.GGMLComputeOps` (0.00)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `gguf` | `gguf.GGUFParser` | 14/14 | `type_to_gguf_type`, `gguf_type`, `gguf_kv` … |
| `ggml-opt` | `core.GGMLOptimizationSchedulerTest` | 13/13 | `ggml_opt_dataset`, `ggml_context`, `ggml_tensor` … |
| `ggml-cann.common` | `nativeMain.kotlin.ai.solace.klang.common.ZlibLoggerNative` | 10/10 | `ggml_cann_device_info`, `cann_device_info`, `ggml_cann_pool` … |
| `kleidiai.kernels` | `bench.ShiftKernels` | 5/5 | `cpu_feature`, `kernel_info`, `lhs_packing_info` … |
| `ggml-backend-impl` | `core.GGMLBackendImpl` | 16/18 | `ggml_backend_buffer_type_i`, `ggml_tensor`, `ggml_backend_buffer_type` … |
| `ggml-backend` | `core.GGMLBackendUtils` | 14/16 | `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_status` … |
| `ggml-metal.ggml-metal-common` | `common.Constants` | 4/4 | `ggml_tensor`, `ggml_cgraph`, `ggml_mem_range_type` … |
| `ggml-openvino.utils` | `checksum.Adler32Utils` | 5/5 | `graph_key`, `graph_key_hash`, `ov_runtime_context` … |
| `ggml-impl` | `core.NumericConversions` | 6/12 | `ggml_tensor`, `ggml_op`, `ggml_cgraph_eval_order` … |
| `openvino.input_model` | `model.IntegrationTest` | 3/3 | `FrontEnd`, `GgmlDecoder`, `InputModel` |
| `backend.backend` | `core.GGMLBackendTest` | 2/2 | `ggml_log_level`, `virgl_apir_callbacks` |
| `amx.common` | `common.ZlibLoggerNative` | 1/1 | `ggml_type` |
| `ggml-sycl.type` | `gguf.GGUFTypes` | 1/1 | `__nv_fp8_e4m3` |
| `ggml-cpu.common` | `core.GGMLCpuCommon` | 2/4 | `ggml_compute_params`, `ggml_tensor` |
| `ggml-zdnn.common` | `common.ZlibLogger` | 4/4 | `ggml_backend_zdnn_device_context`, `ggml_backend_zdnn_context`, `ggml_backend_zdnn_buffer` … |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `ggml-sycl.fattn-vec` | 37 | `ggml-sycl/fattn-vec.hpp` |
| 2 | `openvino.node_context` | 19 | `ggml-openvino/openvino/node_context.h` |
| 3 | `openvino.op_table` | 18 | `ggml-openvino/openvino/op_table.h` |
| 4 | `ggml-sycl.convert` | 17 | `ggml-sycl/convert.hpp` |
| 5 | `ggml-sycl.fattn-tile` | 11 | `ggml-sycl/fattn-tile.hpp` |
| 6 | `htp.hvx-base` | 11 | `ggml-hexagon/htp/hvx-base.h` |
| 7 | `op.reshape` | 9 | `ggml-openvino/openvino/op/reshape.cpp` |
| 8 | `ggml-sycl.concat` | 9 | `ggml-sycl/concat.hpp` |
| 9 | `dpct.helper` | 9 | `ggml-sycl/dpct/helper.hpp` |
| 10 | `ggml-cpu.traits` | 9 | `ggml-cpu/traits.h` |
| 11 | `ggml-virtgpu.ggml-remoting` | 8 | `ggml-virtgpu/ggml-remoting.h` |
| 12 | `htp.hex-utils` | 8 | `ggml-hexagon/htp/hex-utils.h` |
| 13 | `ggml-sycl.presets` | 8 | `ggml-sycl/presets.hpp` |
| 14 | `op.transpose` | 7 | `ggml-openvino/openvino/op/transpose.cpp` |
| 15 | `ggml-sycl.set` | 6 | `ggml-sycl/set.hpp` |
| 16 | `ggml-quants` | 6 | `ggml-quants.h` |
| 17 | `shared.apir_backend` | 6 | `ggml-virtgpu/backend/shared/apir_backend.h` |
| 18 | `backend.backend-virgl-apir` | 6 | `ggml-virtgpu/backend/backend-virgl-apir.h` |
| 19 | `ggml-sycl.dequantize` | 5 | `ggml-sycl/dequantize.hpp` |
| 20 | `backend.backend-dispatched` | 5 | `ggml-virtgpu/backend/backend-dispatched.h` |

... and 200 more missing files.

## Documentation Gaps

**Documentation coverage:** 4142 / 1661 lines (249%)

Top documentation gaps (>20%):

- `ggml-cann.aclnn_ops` - 89% gap (1413 → 157 lines)
- `ggml-cann.common` - 100% gap (192 → 0 lines)

