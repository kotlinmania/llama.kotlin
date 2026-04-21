# Code Port - Progress Report

**Generated:** 2026-04-21
**Source:** tmp/llama.cpp/src
**Target:** src/nativeMain/kotlin/ai/solace/llamakotlin

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 143 | 100% |
| Target units (paired) | 30 | - |
| Target files (total) | 30 | - |
| Porting progress | 18 | 12.6% (matched) |
| Missing files | 125 | 87.4% |

## Port Quality Analysis

**Average Similarity:** 0.02

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 18 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `llama-model` → `model.LlamaModel` (0.02, 12 deps)
- `llama` → `model.Grammar` (0.00, 16 deps)
- `llama-memory` → `model.LlamaMemory` (0.03, 7 deps)
- `llama-impl` → `model.LlamaImpl` (0.07, 19 deps)
- `llama-arch` → `model.LlamaArch` (0.08, 5 deps)
- `llama-io` → `model.LlamaIO` (0.08, 3 deps)
- `models.models` → `model.InferencePipeline` (0.00, 114 deps)
- `llama-memory-recurrent` → `model.LlamaMemoryRecurrent` (0.00, 12 deps)
- `llama-graph` → `model.LlamaGraph` (0.00, 7 deps)
- `llama-batch` → `model.LlamaBatch` (0.00, 7 deps)
- `llama-vocab` → `model.LlamaVocab` (0.00, 6 deps)
- `llama-mmap` → `model.LlamaMmap` (0.00, 5 deps)
- `llama-kv-cache` → `model.KVCache` (0.00, 4 deps)
- `llama-model-loader` → `gguf.ModelLoader` (0.00, 4 deps)
- `llama-context` → `model.LlamaAttention` (0.00, 4 deps)
- `llama-adapter` → `model.LlamaAdapter` (0.00, 2 deps)
- `llama-model-saver` → `model.GGMLIntegration` (0.00, 1 deps)
- `models.maincoder` → `main` (0.16)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `llama-model` | `model.LlamaModel` | 11/16 | `llama_cparams`, `llama_ubatch`, `llama_model_loader` … |
| `llama` | `model.Grammar` | 18/18 | `llama_flash_attn_type`, `llama_device_memory_data`, `user_data_t` … |
| `llama-memory` | `model.LlamaMemory` | 7/8 | `llama_ubatch`, `llama_batch_allocr`, `llama_io_write_i` … |
| `llama-impl` | `model.LlamaImpl` | 3/5 | `no_init`, `ggml_tensor`, `gguf_context` |
| `llama-arch` | `model.LlamaArch` | 3/7 | `llm_arch`, `LLM_TN_IMPL`, `LLM_TN` |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `llama-cparams` | 5 | `llama-cparams.h` |
| 2 | `llama-hparams` | 4 | `llama-hparams.h` |
| 3 | `llama-memory-hybrid` | 3 | `llama-memory-hybrid.h` |
| 4 | `llama-memory-hybrid-iswa` | 3 | `llama-memory-hybrid-iswa.h` |
| 5 | `llama-kv-cache-iswa` | 3 | `llama-kv-cache-iswa.h` |
| 6 | `llama-ext` | 2 | `llama-ext.h` |
| 7 | `unicode-data` | 1 | `unicode-data.h` |
| 8 | `llama-chat` | 1 | `llama-chat.h` |
| 9 | `llama-grammar` | 1 | `llama-grammar.h` |
| 10 | `unicode` | 1 | `unicode.h` |
| 11 | `llama-sampler` | 1 | `llama-sampler.h` |
| 12 | `llama-kv-cells` | 1 | `llama-kv-cells.h` |
| 13 | `models.llada-moe` | 0 | `models/llada-moe.cpp` |
| 14 | `models.jamba` | 0 | `models/jamba.cpp` |
| 15 | `models.arctic` | 0 | `models/arctic.cpp` |
| 16 | `models.arwkv7` | 0 | `models/arwkv7.cpp` |
| 17 | `models.baichuan` | 0 | `models/baichuan.cpp` |
| 18 | `models.bailingmoe` | 0 | `models/bailingmoe.cpp` |
| 19 | `models.bailingmoe2` | 0 | `models/bailingmoe2.cpp` |
| 20 | `models.bert` | 0 | `models/bert.cpp` |

... and 105 more missing files.

## Documentation Gaps

**Documentation coverage:** 2205 / 2 lines (110250%)

Top documentation gaps (>20%):

No significant documentation gaps found.

