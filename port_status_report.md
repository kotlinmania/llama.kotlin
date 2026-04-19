# Code Port - Progress Report

**Generated:** 2026-04-18
**Source:** tmp/llama.cpp/src
**Target:** src/nativeMain/kotlin/ai/solace/llamakotlin

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 143 | 100% |
| Target units (paired) | 19 | - |
| Target files (total) | 19 | - |
| Porting progress | 8 | 5.6% (matched) |
| Missing files | 135 | 94.4% |

## Port Quality Analysis

**Average Similarity:** 0.02

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 8 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `models.models` → `model.InferencePipeline` (0.00, 114 deps)
- `llama-context` → `model.LlamaAttention` (0.00, 4 deps)
- `llama-model` → `model.LlamaModel` (0.00, 12 deps)
- `llama-kv-cache` → `model.KVCache` (0.00, 4 deps)
- `llama-model-loader` → `gguf.ModelLoader` (0.00, 4 deps)
- `llama-grammar` → `model.Grammar` (0.00, 1 deps)
- `llama-model-saver` → `model.GGMLIntegration` (0.00, 1 deps)
- `models.maincoder` → `main` (0.16)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `models.models` | `model.InferencePipeline` | 113/113 | `llm_build_mamba_base`, `llm_build_delta_net_base`, `llm_build_rwkv6_base` … |
| `llama-context` | `model.LlamaAttention` | 12/12 | `llama_model`, `llama_batch_allocr`, `llama_io_read_i` … |
| `llama-model` | `model.LlamaModel` | 15/16 | `llama_cparams`, `llama_ubatch`, `llama_model_loader` … |
| `llama-kv-cache` | `model.KVCache` | 10/10 | `llama_cparams`, `llama_hparams`, `llama_model` … |
| `llama-model-loader` | `gguf.ModelLoader` | 16/16 | `llama_ftype`, `GKV_Base_Type`, `GKV_Base` … |
| `llama-grammar` | `model.Grammar` | 3/3 | `llama_gretype`, `llama_grammar`, `llama_vocab` |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `llama-impl` | 19 | `llama-impl.h` |
| 2 | `llama` | 16 | `llama.cpp` |
| 3 | `llama-memory-recurrent` | 12 | `llama-memory-recurrent.h` |
| 4 | `llama-batch` | 7 | `llama-batch.h` |
| 5 | `llama-graph` | 7 | `llama-graph.h` |
| 6 | `llama-memory` | 7 | `llama-memory.h` |
| 7 | `llama-vocab` | 6 | `llama-vocab.h` |
| 8 | `llama-mmap` | 5 | `llama-mmap.h` |
| 9 | `llama-arch` | 5 | `llama-arch.h` |
| 10 | `llama-cparams` | 5 | `llama-cparams.h` |
| 11 | `llama-hparams` | 4 | `llama-hparams.h` |
| 12 | `llama-io` | 3 | `llama-io.h` |
| 13 | `llama-kv-cache-iswa` | 3 | `llama-kv-cache-iswa.h` |
| 14 | `llama-memory-hybrid` | 3 | `llama-memory-hybrid.h` |
| 15 | `llama-memory-hybrid-iswa` | 3 | `llama-memory-hybrid-iswa.h` |
| 16 | `llama-adapter` | 2 | `llama-adapter.h` |
| 17 | `llama-ext` | 2 | `llama-ext.h` |
| 18 | `unicode-data` | 1 | `unicode-data.h` |
| 19 | `unicode` | 1 | `unicode.h` |
| 20 | `llama-sampler` | 1 | `llama-sampler.h` |

... and 115 more missing files.

## Documentation Gaps

**Documentation coverage:** 270 / 2 lines (13500%)

Top documentation gaps (>20%):

No significant documentation gaps found.

