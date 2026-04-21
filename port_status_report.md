# Code Port - Progress Report

**Generated:** 2026-04-21
**Source:** tmp/llama.cpp/src
**Target:** src

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total source files | 143 | 100% |
| Target units (paired) | 166 | - |
| Target files (total) | 166 | - |
| Porting progress | 32 | 22.4% (matched) |
| Missing files | 111 | 77.6% |

## Port Quality Analysis

**Average Similarity:** 0.14

**Quality Distribution:**
- Excellent (≥0.85): 0 files (0.0% of matched)
- Good (0.60-0.84): 2 files (6.2% of matched)
- Critical (<0.60): 30 files (93.8% of matched)

### Excellent Ports (Similarity ≥ 0.85)

These files are well-ported and likely complete:


### Critical Ports (Similarity < 0.60)

These files need significant work:

- `llama` → `model.Grammar` (0.00, 16 deps)
- `llama-hparams` → `model.LlamaHparams` (0.30, 4 deps)
- `llama-graph` → `model.LlamaGraph` (0.42, 7 deps)
- `llama-memory` → `model.LlamaMemory` (0.07, 7 deps)
- `llama-mmap` → `model.LlamaMmap` (0.19, 5 deps)
- `llama-impl` → `model.LlamaImpl` (0.16, 19 deps)
- `llama-arch` → `model.LlamaArch` (0.09, 5 deps)
- `llama-model-saver` → `model.LlamaModelSaver` (0.28, 1 deps)
- `llama-batch` → `model.LlamaBatch` (0.56, 7 deps)
- `llama-io` → `model.LlamaIO` (0.19, 3 deps)
- `models.models` → `model.InferencePipeline` (0.00, 114 deps)
- `llama-cparams` → `model.LlamaCparams` (0.00, 5 deps)
- `llama-chat` → `model.LlamaChat` (0.35, 1 deps)
- `llama-memory-recurrent` → `model.LlamaMemoryRecurrent` (0.00, 12 deps)
- `llama-model` → `model.LlamaModel` (0.00, 12 deps)
- `llama-vocab` → `model.LlamaVocab` (0.00, 6 deps)
- `llama-context` → `model.LlamaAttention` (0.00, 4 deps)
- `llama-model-loader` → `gguf.ModelLoader` (0.00, 4 deps)
- `llama-kv-cache` → `model.KVCache` (0.00, 4 deps)
- `llama-memory-hybrid-iswa` → `model.LlamaMemoryHybridIswa` (0.00, 3 deps)
- `llama-memory-hybrid` → `model.LlamaMemoryHybrid` (0.00, 3 deps)
- `llama-adapter` → `model.LlamaAdapter` (0.00, 2 deps)
- `llama-ext` → `model.LlamaExt` (0.00, 2 deps)
- `llama-sampler` → `model.LlamaSampler` (0.00, 1 deps)
- `unicode` → `model.LlamaUnicode` (0.00, 1 deps)
- `unicode-data` → `model.LlamaUnicodeData` (0.32, 1 deps)
- `models.llama` → `model.LlamaApi` (0.00)
- `llama-quant` → `model.LlamaQuant` (0.00)
- `models.maincoder` → `llamakotlin.main` (0.16)
- `models.bitnet` → `core.GGMLBitNet158Test` (0.16)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `llama` | `model.Grammar` | 18/18 | `llama_flash_attn_type`, `llama_device_memory_data`, `user_data_t` … |
| `llama-graph` | `model.LlamaGraph` | 1/1 | `llama_sampler_data` |
| `llama-memory` | `model.LlamaMemory` | 7/8 | `llama_ubatch`, `llama_batch_allocr`, `llama_io_write_i` … |
| `llama-mmap` | `model.LlamaMmap` | 1/4 | `impl` |
| `llama-impl` | `model.LlamaImpl` | 3/5 | `no_init`, `ggml_tensor`, `gguf_context` |
| `llama-arch` | `model.LlamaArch` | 3/7 | `llm_arch`, `LLM_TN_IMPL`, `LLM_TN` |
| `llama-model-saver` | `model.LlamaModelSaver` | 5/6 | `gguf_context`, `llama_model`, `LLM_KV` … |
| `llama-batch` | `model.LlamaBatch` | 1/1 | `llama_batch` |

## High Priority Missing Files

| Rank | Source file | Deps | Path |
|------|------------|------|------|
| 1 | `llama-grammar` | 1 | `llama-grammar.h` |
| 2 | `models.llada-moe` | 0 | `models/llada-moe.cpp` |
| 3 | `models.xverse` | 0 | `models/xverse.cpp` |
| 4 | `models.arcee` | 0 | `models/arcee.cpp` |
| 5 | `models.arctic` | 0 | `models/arctic.cpp` |
| 6 | `models.arwkv7` | 0 | `models/arwkv7.cpp` |
| 7 | `models.baichuan` | 0 | `models/baichuan.cpp` |
| 8 | `models.bailingmoe` | 0 | `models/bailingmoe.cpp` |
| 9 | `models.bailingmoe2` | 0 | `models/bailingmoe2.cpp` |
| 10 | `models.bert` | 0 | `models/bert.cpp` |
| 11 | `models.bloom` | 0 | `models/bloom.cpp` |
| 12 | `models.chameleon` | 0 | `models/chameleon.cpp` |
| 13 | `models.chatglm` | 0 | `models/chatglm.cpp` |
| 14 | `models.codeshell` | 0 | `models/codeshell.cpp` |
| 15 | `models.cogvlm` | 0 | `models/cogvlm.cpp` |
| 16 | `models.cohere2-iswa` | 0 | `models/cohere2-iswa.cpp` |
| 17 | `models.command-r` | 0 | `models/command-r.cpp` |
| 18 | `models.dbrx` | 0 | `models/dbrx.cpp` |
| 19 | `models.deci` | 0 | `models/deci.cpp` |
| 20 | `models.deepseek` | 0 | `models/deepseek.cpp` |

... and 91 more missing files.

## Documentation Gaps

**Documentation coverage:** 4353 / 2 lines (217650%)

Top documentation gaps (>20%):

No significant documentation gaps found.

