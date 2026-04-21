# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `llama` | `model.Grammar` | 0.00 | 16 | 52 | 828.8 |
| 2 | `llama-memory` | `model.LlamaMemory` | 0.03 | 7 | 27 | 392.4 |
| 3 | `llama-impl` | `model.LlamaImpl` | 0.13 | 19 | 12 | 204.9 |
| 4 | `llama-model-saver` | `model.IntegrationTest` | 0.00 | 1 | 16 | 185.6 |
| 5 | `llama-arch` | `model.LlamaArch` | 0.08 | 5 | 9 | 130.5 |
| 6 | `llama-io` | `model.LlamaIO` | 0.08 | 3 | 7 | 95.8 |
| 7 | `llama-kv-cache-iswa` | `model.LlamaKvCacheIswa` | 0.60 | 3 | 3 | 41.1 |
| 8 | `models.models` | `model.InferencePipeline` | 0.00 | 114 | 0 | 23.7 |
| 9 | `llama-kv-cells` | `model.LlamaKvCells` | 0.73 | 1 | 2 | 23.7 |
| 10 | `llama-chat` | `model.LlamaChat` | 0.35 | 1 | 1 | 13.6 |
| 11 | `llama-model` | `model.LlamaModel` | 0.00 | 12 | 0 | 12.8 |
| 12 | `llama-memory-recurrent` | `model.LlamaMemoryRecurrent` | 0.00 | 12 | 0 | 12.8 |
| 13 | `llama-batch` | `model.LlamaBatch` | 0.00 | 7 | 0 | 10.4 |
| 14 | `llama-graph` | `model.LlamaGraph` | 0.00 | 7 | 0 | 10.4 |
| 15 | `llama-vocab` | `model.LlamaVocab` | 0.00 | 6 | 0 | 9.7 |
| 16 | `llama-mmap` | `model.LlamaMmap` | 0.00 | 5 | 0 | 9.0 |
| 17 | `llama-kv-cache` | `model.KVCache` | 0.00 | 4 | 0 | 8.0 |
| 18 | `llama-model-loader` | `gguf.ModelLoader` | 0.00 | 4 | 0 | 8.0 |
| 19 | `llama-context` | `model.LlamaAttention` | 0.00 | 4 | 0 | 8.0 |
| 20 | `llama-memory-hybrid-iswa` | `model.LlamaMemoryHybridIswa` | 0.00 | 3 | 0 | 6.9 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **llama** → `model.Grammar`
  - Similarity: 0.00
  - Dependencies: 16

- **llama-memory** → `model.LlamaMemory`
  - Similarity: 0.03
  - Dependencies: 7

- **llama-impl** → `model.LlamaImpl`
  - Similarity: 0.13
  - Dependencies: 19

- **llama-model-saver** → `model.IntegrationTest`
  - Similarity: 0.00
  - Dependencies: 1

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

- **llama-chat** → `model.LlamaChat`
  - Similarity: 0.35
  - Dependencies: 1

- **llama-model** → `model.LlamaModel`
  - Similarity: 0.00
  - Dependencies: 12
  - TODOs: 1
  - Lint issues: 5

- **llama-memory-recurrent** → `model.LlamaMemoryRecurrent`
  - Similarity: 0.00
  - Dependencies: 12
  - TODOs: 1
  - Lint issues: 25

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

- **llama-kv-cache** → `model.KVCache`
  - Similarity: 0.00
  - Dependencies: 4
  - TODOs: 3
  - Lint issues: 2

- **llama-model-loader** → `gguf.ModelLoader`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 1

- **llama-context** → `model.LlamaAttention`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 16

- **llama-memory-hybrid-iswa** → `model.LlamaMemoryHybridIswa`
  - Similarity: 0.00
  - Dependencies: 3
  - TODOs: 2
  - Lint issues: 7

- **llama-memory-hybrid** → `model.LlamaMemoryHybrid`
  - Similarity: 0.00
  - Dependencies: 3
  - TODOs: 2
  - Lint issues: 4

- **llama-adapter** → `model.LlamaAdapter`
  - Similarity: 0.00
  - Dependencies: 2
  - Lint issues: 5

- **llama-ext** → `model.LlamaExt`
  - Similarity: 0.00
  - Dependencies: 2
  - Lint issues: 4

- **llama-sampler** → `model.LlamaSampler`
  - Similarity: 0.00
  - Dependencies: 1
  - TODOs: 3
  - Lint issues: 2

- **unicode-data** → `model.LlamaUnicodeData`
  - Similarity: 0.32
  - Dependencies: 1
  - TODOs: 3

## Missing Files (Top by Dependents)

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

