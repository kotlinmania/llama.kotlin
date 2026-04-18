# Code Port - Progress Report

**Generated:** 2026-04-18
**Source:** tmp/llama.cpp/ggml
**Target:** src/commonMain/kotlin/ai/solace/llamakotlin

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 277 | 100% |
| Target units (paired) | 21 | - |
| Target files (total) | 21 | - |
| Porting progress | 15 | 5.4% (matched) |
| Missing files | 262 | 94.6% |

## Port Quality Analysis

**Average Similarity:** 0.00

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 15 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `include.ggml` → `core.GGMLOps` (0.00, 68 deps)
- `dpct.helper` → `core.QuantizationHelper` (0.00, 9 deps)
- `ggml-backend` → `core.GGMLBackend` (0.00)
- `ggml-impl` → `core.NumericConversions` (0.00, 58 deps)
- `include.ggml-cpu` → `core.GGMLCpuExecutor` (0.00, 13 deps)
- `ggml-cpu.ggml-cpu` → `core.GGMLCpuBackend` (0.00)
- `include.ggml-alloc` → `core.GGMLAlloc` (0.00, 8 deps)
- `openvino.utils` → `core.GGMLTensorUtils` (0.00)
- `ggml-virtgpu.ggml-backend` → `core.GGMLBackendUtils` (0.00)
- `ggml-sycl.type` → `core.GGMLTypes` (0.00, 2 deps)
- `ggml-cpu.simd-gemm` → `simd.GGMLSimd` (0.00, 1 deps)
- `ggml-threading` → `core.GGMLScheduler` (0.00, 1 deps)
- `ggml-cpu.quants` → `core.GGMLQuants` (0.00, 4 deps)
- `ggml` → `core.GGMLGraph` (0.00)
- `ggml-cpu.ops` → `core.GGMLComputeOps` (0.00)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `include.ggml` | `core.GGMLOps` | 25/25 | `ggml_status`, `ggml_object`, `ggml_context` … |
| `dpct.helper` | `core.QuantizationHelper` | 29/29 | `matrix_info_t`, `error_code`, `memcpy_direction` … |
| `ggml-backend` | `core.GGMLBackend` | 13/16 | `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_cgraph` … |
| `ggml-impl` | `core.NumericConversions` | 12/12 | `ggml_tensor`, `ggml_op`, `ggml_log_level` … |
| `include.ggml-cpu` | `core.GGMLCpuExecutor` | 9/9 | `ggml_cplan`, `ggml_threadpool`, `ggml_numa_strategy` … |
| `ggml-cpu.ggml-cpu` | `core.GGMLCpuBackend` | 13/13 | `ggml_backend_cpu_context`, `ggml_backend_plan_cpu`, `ggml_cplan` … |
| `include.ggml-alloc` | `core.GGMLAlloc` | 8/8 | `ggml_backend_buffer_type`, `ggml_backend_buffer`, `ggml_backend` … |
| `ggml-sycl.type` | `core.GGMLTypes` | 1/1 | `__nv_fp8_e4m3` |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `ggml-backend-impl` | 47 | `src/ggml-backend-impl.h` |
| 2 | `ggml-cpu.common` | 47 | `src/ggml-cpu/common.h` |
| 3 | `include.ggml-backend` | 38 | `include/ggml-backend.h` |
| 4 | `ggml-sycl.fattn-vec` | 37 | `src/ggml-sycl/fattn-vec.hpp` |
| 5 | `ggml-zdnn.utils` | 23 | `src/ggml-zdnn/utils.hpp` |
| 6 | `openvino.node_context` | 19 | `src/ggml-openvino/openvino/node_context.h` |
| 7 | `openvino.op_table` | 18 | `src/ggml-openvino/openvino/op_table.h` |
| 8 | `ggml-sycl.convert` | 17 | `src/ggml-sycl/convert.hpp` |
| 9 | `ggml-common` | 16 | `src/ggml-common.h` |
| 10 | `htp.hvx-base` | 11 | `src/ggml-hexagon/htp/hvx-base.h` |
| 11 | `ggml-cpu.ggml-cpu-impl` | 11 | `src/ggml-cpu/ggml-cpu-impl.h` |
| 12 | `ggml-sycl.fattn-tile` | 11 | `src/ggml-sycl/fattn-tile.hpp` |
| 13 | `ggml-cpu.simd-mappings` | 10 | `src/ggml-cpu/simd-mappings.h` |
| 14 | `op.reshape` | 9 | `src/ggml-openvino/openvino/op/reshape.cpp` |
| 15 | `ggml-sycl.concat` | 9 | `src/ggml-sycl/concat.hpp` |
| 16 | `ggml-cpu.traits` | 9 | `src/ggml-cpu/traits.h` |
| 17 | `htp.hex-utils` | 8 | `src/ggml-hexagon/htp/hex-utils.h` |
| 18 | `ggml-sycl.presets` | 8 | `src/ggml-sycl/presets.hpp` |
| 19 | `ggml-virtgpu.ggml-remoting` | 8 | `src/ggml-virtgpu/ggml-remoting.h` |
| 20 | `op.transpose` | 7 | `src/ggml-openvino/openvino/op/transpose.cpp` |

... and 242 more missing files.

## Documentation Gaps

**Documentation coverage:** 1179 / 306 lines (385%)

Top documentation gaps (>20%):

- `dpct.helper` - 100% gap (250 → 0 lines)
- `ggml-impl` - 31% gap (48 → 33 lines)

