# Code Port - Progress Report

**Generated:** 2026-04-21
**Source:** tmp/llama.cpp/src
**Target:** src

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 143 | 100% |
| Target units (paired) | 159 | - |
| Target files (total) | 159 | - |
| Porting progress | 28 | 19.6% (matched) |
| Missing files | 115 | 80.4% |

## Port Quality Analysis

**Average Similarity:** 0.09

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 2 files (7.1% of matched)
- Critical (<0.60): 26 files (92.9% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `llama` → `model.Grammar` (0.00, 16 deps)
- `llama-memory` → `model.LlamaMemory` (0.03, 7 deps)
- `llama-impl` → `model.LlamaImpl` (0.13, 19 deps)
- `llama-model-saver` → `model.IntegrationTest` (0.00, 1 deps)
- `llama-arch` → `model.LlamaArch` (0.08, 5 deps)
- `llama-io` → `model.LlamaIO` (0.08, 3 deps)
- `models.models` → `model.InferencePipeline` (0.00, 114 deps)
- `llama-chat` → `model.LlamaChat` (0.35, 1 deps)
- `llama-model` → `model.LlamaModel` (0.00, 12 deps)
- `llama-memory-recurrent` → `model.LlamaMemoryRecurrent` (0.00, 12 deps)
- `llama-batch` → `model.LlamaBatch` (0.00, 7 deps)
- `llama-graph` → `model.LlamaGraph` (0.00, 7 deps)
- `llama-vocab` → `model.LlamaVocab` (0.00, 6 deps)
- `llama-mmap` → `model.LlamaMmap` (0.00, 5 deps)
- `llama-kv-cache` → `model.KVCache` (0.00, 4 deps)
- `llama-model-loader` → `gguf.ModelLoader` (0.00, 4 deps)
- `llama-context` → `model.LlamaAttention` (0.00, 4 deps)
- `llama-memory-hybrid-iswa` → `model.LlamaMemoryHybridIswa` (0.00, 3 deps)
- `llama-memory-hybrid` → `model.LlamaMemoryHybrid` (0.00, 3 deps)
- `llama-adapter` → `model.LlamaAdapter` (0.00, 2 deps)
- `llama-ext` → `model.LlamaExt` (0.00, 2 deps)
- `llama-sampler` → `model.LlamaSampler` (0.00, 1 deps)
- `unicode-data` → `model.LlamaUnicodeData` (0.32, 1 deps)
- `models.llama` → `model.LlamaApi` (0.00)
- `models.maincoder` → `llamakotlin.main` (0.16)
- `models.bitnet` → `core.GGMLBitNet158Test` (0.16)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `llama` | `model.Grammar` | 18/18 | `llama_flash_attn_type`, `llama_device_memory_data`, `user_data_t` … |
| `llama-memory` | `model.LlamaMemory` | 7/8 | `llama_ubatch`, `llama_batch_allocr`, `llama_io_write_i` … |
| `llama-impl` | `model.LlamaImpl` | 3/5 | `no_init`, `ggml_tensor`, `gguf_context` |
| `llama-model-saver` | `model.IntegrationTest` | 6/6 | `llama_model_saver`, `gguf_context`, `llama_model` … |
| `llama-arch` | `model.LlamaArch` | 3/7 | `llm_arch`, `LLM_TN_IMPL`, `LLM_TN` |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `llama-cparams` | 5 | `llama-cparams.h` |
| 2 | `llama-hparams` | 4 | `llama-hparams.h` |
| 3 | `unicode` | 1 | `unicode.h` |
| 4 | `llama-grammar` | 1 | `llama-grammar.h` |
| 5 | `llama-quant` | 0 | `llama-quant.h` |
| 6 | `models.llada` | 0 | `models/llada.cpp` |
| 7 | `models.arcee` | 0 | `models/arcee.cpp` |
| 8 | `models.arctic` | 0 | `models/arctic.cpp` |
| 9 | `models.arwkv7` | 0 | `models/arwkv7.cpp` |
| 10 | `models.baichuan` | 0 | `models/baichuan.cpp` |
| 11 | `models.bailingmoe` | 0 | `models/bailingmoe.cpp` |
| 12 | `models.bailingmoe2` | 0 | `models/bailingmoe2.cpp` |
| 13 | `models.bert` | 0 | `models/bert.cpp` |
| 14 | `models.bloom` | 0 | `models/bloom.cpp` |
| 15 | `models.chameleon` | 0 | `models/chameleon.cpp` |
| 16 | `models.chatglm` | 0 | `models/chatglm.cpp` |
| 17 | `models.codeshell` | 0 | `models/codeshell.cpp` |
| 18 | `models.cogvlm` | 0 | `models/cogvlm.cpp` |
| 19 | `models.cohere2-iswa` | 0 | `models/cohere2-iswa.cpp` |
| 20 | `models.command-r` | 0 | `models/command-r.cpp` |

... and 95 more missing files.

## Documentation Gaps

**Documentation coverage:** 3299 / 2 lines (164950%)

Top documentation gaps (>20%):

No significant documentation gaps found.

