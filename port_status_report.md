# Code Port - Progress Report

**Generated:** 2026-04-21
**Source:** tmp/llama.cpp/ggml/src
**Target:** src/commonMain/kotlin/ai/solace/llamakotlin/core

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 256 | 100% |
| Target units (paired) | 25 | - |
| Target files (total) | 25 | - |
| Porting progress | 16 | 6.2% (matched) |
| Missing files | 240 | 93.8% |

## Port Quality Analysis

**Average Similarity:** 0.04

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 16 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `dpct.helper` → `QuantizationHelper` (0.00, 9 deps)
- `openvino.utils` → `GGMLTensorUtils` (0.00)
- `ggml-sycl.type` → `GGMLTypes` (0.00, 2 deps)
- `ggml-cpu.simd-gemm` → `simd.GGMLSimd` (0.00, 1 deps)
- `ggml-common` → `GGMLCommon` (0.00, 16 deps)
- `ggml-cpu.common` → `GGMLCpuCommon` (0.56, 47 deps)
- `ggml-impl` → `NumericConversions` (0.00, 58 deps)
- `ggml` → `GGMLOps` (0.00, 50 deps)
- `ggml-backend-impl` → `GGMLBackendImpl` (0.00, 47 deps)
- `ggml-backend` → `GGMLBackendUtils` (0.00, 20 deps)
- `ggml-cpu.ggml-cpu` → `GGMLCpuBackend` (0.00, 12 deps)
- `ggml-cpu.ggml-cpu-impl` → `GGMLCpuImpl` (0.00, 11 deps)
- `ggml-cpu.quants` → `GGMLQuants` (0.00, 4 deps)
- `ggml-threading` → `GGMLScheduler` (0.09, 1 deps)
- `ggml-virtgpu.ggml-backend` → `GGMLBackend` (0.00)
- `ggml-cpu.ops` → `GGMLComputeOps` (0.00)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `dpct.helper` | `QuantizationHelper` | 29/29 | `matrix_info_t`, `error_code`, `memcpy_direction` … |
| `ggml-sycl.type` | `GGMLTypes` | 1/1 | `__nv_fp8_e4m3` |
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

## Documentation Gaps

**Documentation coverage:** 3517 / 305 lines (1153%)

Top documentation gaps (>20%):

- `dpct.helper` - 100% gap (250 → 0 lines)

