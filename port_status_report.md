# Code Port - Progress Report

**Generated:** 2026-04-21
**Source:** tmp/llama.cpp/src
**Target:** src/nativeMain/kotlin/ai/solace/llamakotlin

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 143 | 100% |
| Target units (paired) | 32 | - |
| Target files (total) | 32 | - |
| Porting progress | 20 | 14.0% (matched) |
| Missing files | 123 | 86.0% |

## Port Quality Analysis

**Average Similarity:** 0.05

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 0 files (0.0% of matched)
- Critical (<0.60): 20 files (100.0% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `llama-grammar` → `model.Grammar` (0.19, 1 deps)
- `llama-memory` → `model.LlamaMemory` (0.03, 7 deps)
- `llama-impl` → `model.LlamaImpl` (0.13, 19 deps)
- `llama-arch` → `model.LlamaArch` (0.08, 5 deps)
- `llama-io` → `model.LlamaIO` (0.08, 3 deps)
- `models.models` → `model.InferencePipeline` (0.00, 114 deps)
- `llama` → `model.KVCache` (0.00, 16 deps)
- `llama-chat` → `model.LlamaChat` (0.35, 1 deps)
- `llama-memory-recurrent` → `model.LlamaMemoryRecurrent` (0.00, 12 deps)
- `llama-model` → `model.LlamaModel` (0.00, 12 deps)
- `llama-batch` → `model.LlamaBatch` (0.00, 7 deps)
- `llama-graph` → `model.LlamaGraph` (0.00, 7 deps)
- `llama-vocab` → `model.LlamaVocab` (0.00, 6 deps)
- `llama-mmap` → `model.LlamaMmap` (0.00, 5 deps)
- `llama-model-loader` → `gguf.ModelLoader` (0.00, 4 deps)
- `llama-context` → `model.LlamaAttention` (0.00, 4 deps)
- `llama-adapter` → `model.LlamaAdapter` (0.00, 2 deps)
- `llama-model-saver` → `model.GGMLIntegration` (0.00, 1 deps)
- `models.llama` → `model.LlamaApi` (0.00)
- `models.maincoder` → `main` (0.16)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `llama-grammar` | `model.Grammar` | 1/3 | `llama_vocab` |
| `llama-memory` | `model.LlamaMemory` | 7/8 | `llama_ubatch`, `llama_batch_allocr`, `llama_io_write_i` … |
| `llama-impl` | `model.LlamaImpl` | 3/5 | `no_init`, `ggml_tensor`, `gguf_context` |
| `llama-arch` | `model.LlamaArch` | 3/7 | `llm_arch`, `LLM_TN_IMPL`, `LLM_TN` |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `llama-cparams` | 5 | `llama-cparams.h` |
| 2 | `llama-hparams` | 4 | `llama-hparams.h` |
| 3 | `llama-kv-cache` | 4 | `llama-kv-cache.h` |
| 4 | `llama-memory-hybrid-iswa` | 3 | `llama-memory-hybrid-iswa.h` |
| 5 | `llama-kv-cache-iswa` | 3 | `llama-kv-cache-iswa.h` |
| 6 | `llama-memory-hybrid` | 3 | `llama-memory-hybrid.h` |
| 7 | `llama-ext` | 2 | `llama-ext.h` |
| 8 | `unicode-data` | 1 | `unicode-data.h` |
| 9 | `llama-kv-cells` | 1 | `llama-kv-cells.h` |
| 10 | `llama-sampler` | 1 | `llama-sampler.h` |
| 11 | `unicode` | 1 | `unicode.h` |
| 12 | `llama-quant` | 0 | `llama-quant.h` |
| 13 | `models.jamba` | 0 | `models/jamba.cpp` |
| 14 | `models.arctic` | 0 | `models/arctic.cpp` |
| 15 | `models.arwkv7` | 0 | `models/arwkv7.cpp` |
| 16 | `models.baichuan` | 0 | `models/baichuan.cpp` |
| 17 | `models.bailingmoe` | 0 | `models/bailingmoe.cpp` |
| 18 | `models.bailingmoe2` | 0 | `models/bailingmoe2.cpp` |
| 19 | `models.bert` | 0 | `models/bert.cpp` |
| 20 | `models.bitnet` | 0 | `models/bitnet.cpp` |

... and 103 more missing files.

## Documentation Gaps

**Documentation coverage:** 2691 / 3 lines (89700%)

Top documentation gaps (>20%):

No significant documentation gaps found.

