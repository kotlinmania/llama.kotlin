# Code Port - Progress Report

**Generated:** 2026-04-18
**Source:** tmp/llama.cpp/ggml
**Target:** src/commonMain/kotlin/ai/solace/llamakotlin

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 277 | 100% |
| Target units (paired) | 23 | - |
| Target files (total) | 23 | - |
| Porting progress | 17 | 6.1% (matched) |
| Missing files | 260 | 93.9% |

## Port Quality Analysis

**Average Similarity:** 0.02

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 17 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `dpct.helper` → `core.QuantizationHelper` (0.00, 9 deps)
- `ggml-backend` → `core.GGMLBackendUtils` (0.00)
- `include.ggml-cpu` → `core.GGMLCpuExecutor` (0.00, 13 deps)
- `ggml-cpu.ggml-cpu` → `core.GGMLCpuBackend` (0.00)
- `ggml-impl` → `core.NumericConversions` (0.34, 58 deps)
- `include.ggml-alloc` → `core.GGMLAlloc` (0.00, 8 deps)
- `openvino.utils` → `core.GGMLTensorUtils` (0.00)
- `ggml-sycl.type` → `core.GGMLTypes` (0.00, 2 deps)
- `ggml-cpu.simd-gemm` → `simd.GGMLSimd` (0.00, 1 deps)
- `ggml-common` → `core.GGMLCommon` (0.00, 16 deps)
- `ggml-threading` → `core.GGMLScheduler` (0.00, 1 deps)
- `include.ggml` → `core.GGMLOps` (0.00, 68 deps)
- `ggml-backend-impl` → `core.GGMLBackendImpl` (0.00, 47 deps)
- `include.ggml-backend` → `core.GGMLBackend` (0.00, 38 deps)
- `ggml-cpu.quants` → `core.GGMLQuants` (0.00, 4 deps)
- `ggml` → `core.GGMLGraph` (0.00)
- `ggml-cpu.ops` → `core.GGMLComputeOps` (0.00)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `dpct.helper` | `core.QuantizationHelper` | 29/29 | `matrix_info_t`, `error_code`, `memcpy_direction` … |
| `ggml-backend` | `core.GGMLBackendUtils` | 16/16 | `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_status` … |
| `include.ggml-cpu` | `core.GGMLCpuExecutor` | 9/9 | `ggml_cplan`, `ggml_threadpool`, `ggml_numa_strategy` … |
| `ggml-cpu.ggml-cpu` | `core.GGMLCpuBackend` | 13/13 | `ggml_backend_cpu_context`, `ggml_backend_plan_cpu`, `ggml_cplan` … |
| `ggml-impl` | `core.NumericConversions` | 6/12 | `ggml_tensor`, `ggml_op`, `ggml_cgraph_eval_order` … |
| `include.ggml-alloc` | `core.GGMLAlloc` | 8/8 | `ggml_backend_buffer_type`, `ggml_backend_buffer`, `ggml_backend` … |
| `ggml-sycl.type` | `core.GGMLTypes` | 1/1 | `__nv_fp8_e4m3` |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `ggml-cpu.common` | 47 | `src/ggml-cpu/common.h` |
| 2 | `ggml-sycl.fattn-vec` | 37 | `src/ggml-sycl/fattn-vec.hpp` |
| 3 | `ggml-zdnn.utils` | 23 | `src/ggml-zdnn/utils.hpp` |
| 4 | `openvino.node_context` | 19 | `src/ggml-openvino/openvino/node_context.h` |
| 5 | `openvino.op_table` | 18 | `src/ggml-openvino/openvino/op_table.h` |
| 6 | `ggml-sycl.convert` | 17 | `src/ggml-sycl/convert.hpp` |
| 7 | `htp.hvx-base` | 11 | `src/ggml-hexagon/htp/hvx-base.h` |
| 8 | `ggml-cpu.ggml-cpu-impl` | 11 | `src/ggml-cpu/ggml-cpu-impl.h` |
| 9 | `ggml-sycl.fattn-tile` | 11 | `src/ggml-sycl/fattn-tile.hpp` |
| 10 | `ggml-cpu.simd-mappings` | 10 | `src/ggml-cpu/simd-mappings.h` |
| 11 | `ggml-cpu.traits` | 9 | `src/ggml-cpu/traits.h` |
| 12 | `op.reshape` | 9 | `src/ggml-openvino/openvino/op/reshape.cpp` |
| 13 | `ggml-sycl.concat` | 9 | `src/ggml-sycl/concat.hpp` |
| 14 | `ggml-sycl.presets` | 8 | `src/ggml-sycl/presets.hpp` |
| 15 | `htp.hex-utils` | 8 | `src/ggml-hexagon/htp/hex-utils.h` |
| 16 | `ggml-virtgpu.ggml-remoting` | 8 | `src/ggml-virtgpu/ggml-remoting.h` |
| 17 | `op.transpose` | 7 | `src/ggml-openvino/openvino/op/transpose.cpp` |
| 18 | `ggml-quants` | 6 | `src/ggml-quants.h` |
| 19 | `shared.apir_backend` | 6 | `src/ggml-virtgpu/backend/shared/apir_backend.h` |
| 20 | `ggml-sycl.set` | 6 | `src/ggml-sycl/set.hpp` |

... and 240 more missing files.

## Documentation Gaps

**Documentation coverage:** 2491 / 307 lines (811%)

Top documentation gaps (>20%):

- `dpct.helper` - 100% gap (250 → 0 lines)

