# Code Port - Progress Report

**Generated:** 2026-04-25
**Source:** tmp/llama.cpp/ggml
**Target:** src/commonMain/kotlin/ai/solace/llamakotlin

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 277 | 100% |
| Target units (paired) | 22 | - |
| Target files (total) | 22 | - |
| Porting progress | 20 | 7.2% (matched) |
| Missing files | 257 | 92.8% |

## Port Quality Analysis

**Average Similarity:** 0.16

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 1 files (5.0% of matched)
- Critical (<0.60): 19 files (95.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `include.ggml-backend` → `core.GGMLBackend` (0.00, 38 deps)
- `include.ggml` → `core.GGMLOps` (0.45, 68 deps)
- `ggml-backend-impl` → `core.GGMLBackendImpl` (0.38, 47 deps)
- `ggml-cpu.simd-mappings` → `simd.GGMLSimd` (0.00, 10 deps)
- `include.ggml-alloc` → `core.GGMLAlloc` (0.05, 8 deps)
- `ggml-impl` → `core.NumericConversions` (0.52, 58 deps)
- `openvino.utils` → `core.GGMLTensorUtils` (0.00)
- `ggml-cpu.traits` → `core.GGMLCpuTraits` (0.21, 9 deps)
- `include.ggml-cpu` → `core.GGMLCpuExecutor` (0.25, 13 deps)
- `ggml-sycl.quants` → `core.GGMLCpuQuants` (0.00)
- `ggml-common` → `core.GGMLCommon` (0.00, 16 deps)
- `ggml-cpu.common` → `core.GGMLCpuCommon` (0.56, 47 deps)
- `ggml-cpu.ggml-cpu-impl` → `core.GGMLCpuImpl` (0.00, 11 deps)
- `ggml` → `core.GGMLTypes` (0.00)
- `ggml-quants` → `core.GGMLQuantsRef` (0.00, 6 deps)
- `ggml-cpu.quants` → `core.GGMLQuants` (0.00, 4 deps)
- `ggml-threading` → `core.GGMLScheduler` (0.10, 1 deps)
- `ggml-cpu.ggml-cpu` → `core.GGMLCpuBackend` (0.00)
- `ggml-cpu.ops` → `core.GGMLComputeOps` (0.00)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `include.ggml-backend` | `core.GGMLBackend` | 7/18 | `ggml_tensor`, `ggml_cgraph`, `ggml_backend_dev_type` … |
| `include.ggml` | `core.GGMLOps` | 12/25 | `ggml_status`, `ggml_object`, `ggml_context` … |
| `ggml-backend-impl` | `core.GGMLBackendImpl` | 16/18 | `ggml_backend_buffer_type_i`, `ggml_tensor`, `ggml_backend_buffer_type` … |
| `include.ggml-alloc` | `core.GGMLAlloc` | 6/8 | `ggml_backend_buffer_type`, `ggml_backend_buffer`, `ggml_backend` … |
| `ggml-impl` | `core.NumericConversions` | 6/12 | `ggml_tensor`, `ggml_op`, `ggml_cgraph_eval_order` … |
| `ggml-cpu.traits` | `core.GGMLCpuTraits` | 4/4 | `ggml_compute_params`, `ggml_tensor`, `tensor_traits` … |
| `ggml-backend` | `core.GGMLBackendUtils` | 13/16 | `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_status` … |
| `include.ggml-cpu` | `core.GGMLCpuExecutor` | 6/9 | `ggml_threadpool`, `ggml_context`, `ggml_tensor` … |
| `ggml-sycl.quants` | `core.GGMLCpuQuants` | 2/2 | `block_q_t`, `traits` |
| `ggml-cpu.common` | `core.GGMLCpuCommon` | 2/4 | `ggml_compute_params`, `ggml_tensor` |

## High Priority Missing Files

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

## Documentation Gaps

**Documentation coverage:** 4074 / 60 lines (6790%)

Top documentation gaps (>20%):

No significant documentation gaps found.

