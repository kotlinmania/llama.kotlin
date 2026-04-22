# Code Port - Progress Report

**Generated:** 2026-04-21
**Source:** tmp/llama.cpp/ggml/src
**Target:** src/commonMain/kotlin/ai/solace/llamakotlin/core

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 256 | 100% |
| Target units (paired) | 22 | - |
| Target files (total) | 22 | - |
| Porting progress | 18 | 7.0% (matched) |
| Missing files | 238 | 93.0% |

## Port Quality Analysis

**Average Similarity:** 0.14

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 1 files (5.6% of matched)
- Critical (<0.60): 17 files (94.4% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `ggml-backend-impl` → `GGMLBackendImpl` (0.36, 47 deps)
- `ggml-cpu.simd-mappings` → `simd.GGMLSimd` (0.00, 10 deps)
- `openvino.utils` → `GGMLTensorUtils` (0.00)
- `ggml-impl` → `NumericConversions` (0.54, 58 deps)
- `ggml-cpu.traits` → `GGMLCpuTraits` (0.19, 9 deps)
- `ggml-virtgpu.ggml-backend` → `GGMLBackend` (0.00)
- `ggml-sycl.type` → `GGMLTypes` (0.00, 2 deps)
- `ggml-sycl.quants` → `GGMLCpuQuants` (0.00)
- `ggml-common` → `GGMLCommon` (0.00, 16 deps)
- `ggml-cpu.common` → `GGMLCpuCommon` (0.56, 47 deps)
- `ggml` → `GGMLOps` (0.00, 50 deps)
- `ggml-cpu.ggml-cpu` → `GGMLCpuBackend` (0.00, 12 deps)
- `ggml-cpu.ggml-cpu-impl` → `GGMLCpuImpl` (0.00, 11 deps)
- `ggml-quants` → `GGMLQuantsRef` (0.00, 6 deps)
- `ggml-cpu.quants` → `GGMLQuants` (0.00, 4 deps)
- `ggml-threading` → `GGMLScheduler` (0.10, 1 deps)
- `ggml-cpu.ops` → `GGMLComputeOps` (0.00)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `ggml-backend-impl` | `GGMLBackendImpl` | 16/18 | `ggml_backend_buffer_type_i`, `ggml_tensor`, `ggml_backend_buffer_type` … |
| `ggml-backend` | `GGMLBackendUtils` | 14/16 | `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_status` … |
| `ggml-impl` | `NumericConversions` | 6/12 | `ggml_tensor`, `ggml_op`, `ggml_cgraph_eval_order` … |
| `ggml-cpu.traits` | `GGMLCpuTraits` | 4/4 | `ggml_compute_params`, `ggml_tensor`, `tensor_traits` … |
| `ggml-sycl.type` | `GGMLTypes` | 1/1 | `__nv_fp8_e4m3` |
| `ggml-sycl.quants` | `GGMLCpuQuants` | 2/2 | `block_q_t`, `traits` |
| `ggml-cpu.common` | `GGMLCpuCommon` | 2/4 | `ggml_compute_params`, `ggml_tensor` |

## High Priority Missing Files

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

## Documentation Gaps

**Documentation coverage:** 3708 / 55 lines (6742%)

Top documentation gaps (>20%):

No significant documentation gaps found.

