# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `llama-grammar` | `model.Grammar` | 0.19 | 1 | 35 | 401.3 |
| 2 | `llama-memory` | `model.LlamaMemory` | 0.03 | 7 | 27 | 392.4 |
| 3 | `llama-impl` | `model.LlamaImpl` | 0.13 | 19 | 12 | 204.9 |
| 4 | `llama-arch` | `model.LlamaArch` | 0.08 | 5 | 9 | 130.5 |
| 5 | `llama-io` | `model.LlamaIO` | 0.08 | 3 | 7 | 95.8 |
| 6 | `models.models` | `model.InferencePipeline` | 0.00 | 114 | 0 | 23.7 |
| 7 | `llama` | `model.KVCache` | 0.00 | 16 | 0 | 14.2 |
| 8 | `llama-chat` | `model.LlamaChat` | 0.35 | 1 | 1 | 13.6 |
| 9 | `llama-memory-recurrent` | `model.LlamaMemoryRecurrent` | 0.00 | 12 | 0 | 12.8 |
| 10 | `llama-model` | `model.LlamaModel` | 0.00 | 12 | 0 | 12.8 |
| 11 | `llama-batch` | `model.LlamaBatch` | 0.00 | 7 | 0 | 10.4 |
| 12 | `llama-graph` | `model.LlamaGraph` | 0.00 | 7 | 0 | 10.4 |
| 13 | `llama-vocab` | `model.LlamaVocab` | 0.00 | 6 | 0 | 9.7 |
| 14 | `llama-mmap` | `model.LlamaMmap` | 0.00 | 5 | 0 | 9.0 |
| 15 | `llama-model-loader` | `gguf.ModelLoader` | 0.00 | 4 | 0 | 8.0 |
| 16 | `llama-context` | `model.LlamaAttention` | 0.00 | 4 | 0 | 8.0 |
| 17 | `llama-adapter` | `model.LlamaAdapter` | 0.00 | 2 | 0 | 5.5 |
| 18 | `llama-model-saver` | `model.GGMLIntegration` | 0.00 | 1 | 0 | 3.5 |
| 19 | `models.llama` | `model.LlamaApi` | 0.00 | 0 | 0 | 0.0 |
| 20 | `models.maincoder` | `main` | 0.16 | 0 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **llama-grammar** → `model.Grammar`
  - Similarity: 0.19
  - Dependencies: 1

- **llama-memory** → `model.LlamaMemory`
  - Similarity: 0.03
  - Dependencies: 7

- **llama-impl** → `model.LlamaImpl`
  - Similarity: 0.13
  - Dependencies: 19

- **llama-arch** → `model.LlamaArch`
  - Similarity: 0.08
  - Dependencies: 5

- **llama-io** → `model.LlamaIO`
  - Similarity: 0.08
  - Dependencies: 3

- **models.models** → `model.InferencePipeline`
  - Similarity: 0.00
  - Dependencies: 114
  - TODOs: 2
  - Lint issues: 2

- **llama** → `model.KVCache`
  - Similarity: 0.00
  - Dependencies: 16
  - Lint issues: 4

- **llama-chat** → `model.LlamaChat`
  - Similarity: 0.35
  - Dependencies: 1

- **llama-memory-recurrent** → `model.LlamaMemoryRecurrent`
  - Similarity: 0.00
  - Dependencies: 12
  - TODOs: 1
  - Lint issues: 25

- **llama-model** → `model.LlamaModel`
  - Similarity: 0.00
  - Dependencies: 12
  - TODOs: 1
  - Lint issues: 5

- **llama-batch** → `model.LlamaBatch`
  - Similarity: 0.00
  - Dependencies: 7
  - TODOs: 1
  - Lint issues: 8

- **llama-graph** → `model.LlamaGraph`
  - Similarity: 0.00
  - Dependencies: 7
  - Lint issues: 14

- **llama-vocab** → `model.LlamaVocab`
  - Similarity: 0.00
  - Dependencies: 6
  - TODOs: 2
  - Lint issues: 8

- **llama-mmap** → `model.LlamaMmap`
  - Similarity: 0.00
  - Dependencies: 5
  - Lint issues: 18

- **llama-model-loader** → `gguf.ModelLoader`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 1

- **llama-context** → `model.LlamaAttention`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 16

- **llama-adapter** → `model.LlamaAdapter`
  - Similarity: 0.00
  - Dependencies: 2
  - Lint issues: 5

- **llama-model-saver** → `model.GGMLIntegration`
  - Similarity: 0.00
  - Dependencies: 1
  - Lint issues: 10

## Missing Files (Top by Dependents)

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

