# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `llama-model` | `model.LlamaModel` | 0.02 | 12 | 80 | 1222.9 |
| 2 | `llama` | `model.Grammar` | 0.00 | 16 | 52 | 828.8 |
| 3 | `llama-memory` | `model.LlamaMemory` | 0.03 | 7 | 27 | 392.4 |
| 4 | `llama-impl` | `model.LlamaImpl` | 0.07 | 19 | 14 | 237.9 |
| 5 | `llama-arch` | `model.LlamaArch` | 0.08 | 5 | 9 | 130.5 |
| 6 | `llama-io` | `model.LlamaIO` | 0.08 | 3 | 7 | 95.8 |
| 7 | `models.models` | `model.InferencePipeline` | 0.00 | 114 | 0 | 23.7 |
| 8 | `llama-memory-recurrent` | `model.LlamaMemoryRecurrent` | 0.00 | 12 | 0 | 12.8 |
| 9 | `llama-graph` | `model.LlamaGraph` | 0.00 | 7 | 0 | 10.4 |
| 10 | `llama-batch` | `model.LlamaBatch` | 0.00 | 7 | 0 | 10.4 |
| 11 | `llama-vocab` | `model.LlamaVocab` | 0.00 | 6 | 0 | 9.7 |
| 12 | `llama-mmap` | `model.LlamaMmap` | 0.00 | 5 | 0 | 9.0 |
| 13 | `llama-kv-cache` | `model.KVCache` | 0.00 | 4 | 0 | 8.0 |
| 14 | `llama-model-loader` | `gguf.ModelLoader` | 0.00 | 4 | 0 | 8.0 |
| 15 | `llama-context` | `model.LlamaAttention` | 0.00 | 4 | 0 | 8.0 |
| 16 | `llama-adapter` | `model.LlamaAdapter` | 0.00 | 2 | 0 | 5.5 |
| 17 | `llama-model-saver` | `model.GGMLIntegration` | 0.00 | 1 | 0 | 3.5 |
| 18 | `models.maincoder` | `main` | 0.16 | 0 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **llama-model** → `model.LlamaModel`
  - Similarity: 0.02
  - Dependencies: 12
  - Lint issues: 5

- **llama** → `model.Grammar`
  - Similarity: 0.00
  - Dependencies: 16

- **llama-memory** → `model.LlamaMemory`
  - Similarity: 0.03
  - Dependencies: 7

- **llama-impl** → `model.LlamaImpl`
  - Similarity: 0.07
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

- **llama-memory-recurrent** → `model.LlamaMemoryRecurrent`
  - Similarity: 0.00
  - Dependencies: 12
  - TODOs: 1
  - Lint issues: 25

- **llama-graph** → `model.LlamaGraph`
  - Similarity: 0.00
  - Dependencies: 7
  - Lint issues: 14

- **llama-batch** → `model.LlamaBatch`
  - Similarity: 0.00
  - Dependencies: 7
  - TODOs: 1
  - Lint issues: 8

- **llama-vocab** → `model.LlamaVocab`
  - Similarity: 0.00
  - Dependencies: 6
  - TODOs: 2
  - Lint issues: 8

- **llama-mmap** → `model.LlamaMmap`
  - Similarity: 0.00
  - Dependencies: 5
  - Lint issues: 18

- **llama-kv-cache** → `model.KVCache`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 4

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

