# Code Port - Progress Report

**Generated:** 2026-04-18
**Source:** tmp/llama.cpp/ggml
**Target:** src/commonMain/kotlin/ai/solace/llamakotlin

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 277 | 100% |
| Target units (paired) | 25 | - |
| Target files (total) | 25 | - |
| Porting progress | 18 | 6.5% (matched) |
| Missing files | 259 | 93.5% |

## Port Quality Analysis

**Average Similarity:** 0.03

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 18 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `dpct.helper` → `core.QuantizationHelper` (0.00, 9 deps)
- `openvino.utils` → `core.GGMLTensorUtils` (0.00)
- `ggml-cpu.simd-gemm` → `simd.GGMLSimd` (0.00, 1 deps)
- `ggml-common` → `core.GGMLCommon` (0.00, 16 deps)
- `ggml-cpu.common` → `core.GGMLCpuCommon` (0.56, 47 deps)
- `ggml-threading` → `core.GGMLScheduler` (0.00, 1 deps)
- `include.ggml` → `core.GGMLOps` (0.00, 68 deps)
- `ggml-impl` → `core.NumericConversions` (0.00, 58 deps)
- `ggml-backend-impl` → `core.GGMLBackendImpl` (0.00, 47 deps)
- `include.ggml-backend` → `core.GGMLBackend` (0.00, 38 deps)
- `include.ggml-cpu` → `core.GGMLCpuExecutor` (0.00, 13 deps)
- `ggml-cpu.ggml-cpu-impl` → `core.GGMLCpuImpl` (0.00, 11 deps)
- `include.ggml-alloc` → `core.GGMLAlloc` (0.00, 8 deps)
- `ggml` → `core.GGMLTypes` (0.00)
- `ggml-cpu.quants` → `core.GGMLQuants` (0.00, 4 deps)
- `ggml-cpu.ggml-cpu` → `core.GGMLCpuBackend` (0.00)
- `ggml-backend` → `core.GGMLBackendUtils` (0.00)
- `ggml-cpu.ops` → `core.GGMLComputeOps` (0.00)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `dpct.helper` | `core.QuantizationHelper` | 29/29 | `matrix_info_t`, `error_code`, `memcpy_direction` … |
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
| 8 | `ggml-cpu.simd-mappings` | 10 | `src/ggml-cpu/simd-mappings.h` |
| 9 | `op.reshape` | 9 | `src/ggml-openvino/openvino/op/reshape.cpp` |
| 10 | `ggml-sycl.concat` | 9 | `src/ggml-sycl/concat.hpp` |
| 11 | `ggml-cpu.traits` | 9 | `src/ggml-cpu/traits.h` |
| 12 | `ggml-sycl.presets` | 8 | `src/ggml-sycl/presets.hpp` |
| 13 | `ggml-virtgpu.ggml-remoting` | 8 | `src/ggml-virtgpu/ggml-remoting.h` |
| 14 | `htp.hex-utils` | 8 | `src/ggml-hexagon/htp/hex-utils.h` |
| 15 | `op.transpose` | 7 | `src/ggml-openvino/openvino/op/transpose.cpp` |
| 16 | `ggml-sycl.set` | 6 | `src/ggml-sycl/set.hpp` |
| 17 | `backend.backend-virgl-apir` | 6 | `src/ggml-virtgpu/backend/backend-virgl-apir.h` |
| 18 | `shared.apir_backend` | 6 | `src/ggml-virtgpu/backend/shared/apir_backend.h` |
| 19 | `ggml-quants` | 6 | `src/ggml-quants.h` |
| 20 | `backend.backend-dispatched` | 5 | `src/ggml-virtgpu/backend/backend-dispatched.h` |

... and 239 more missing files.

## Documentation Gaps

**Documentation coverage:** 3190 / 310 lines (1029%)

Top documentation gaps (>20%):

- `dpct.helper` - 100% gap (250 → 0 lines)

