# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `llama` | `model.Grammar` | 0.00 | 16 | 52 | 828.8 |
| 2 | `llama-hparams` | `model.LlamaHparams` | 0.30 | 4 | 50 | 666.6 |
| 3 | `llama-graph` | `model.LlamaGraph` | 0.42 | 7 | 41 | 586.5 |
| 4 | `llama-memory` | `model.LlamaMemory` | 0.07 | 7 | 26 | 377.8 |
| 5 | `llama-mmap` | `model.LlamaMmap` | 0.19 | 5 | 18 | 251.7 |
| 6 | `llama-impl` | `model.LlamaImpl` | 0.16 | 19 | 11 | 188.6 |
| 7 | `llama-arch` | `model.LlamaArch` | 0.09 | 5 | 8 | 116.8 |
| 8 | `llama-model-saver` | `model.LlamaModelSaver` | 0.28 | 1 | 9 | 105.0 |
| 9 | `llama-batch` | `model.LlamaBatch` | 0.56 | 7 | 5 | 75.4 |
| 10 | `llama-kv-cache-iswa` | `model.LlamaKvCacheIswa` | 0.61 | 3 | 3 | 41.0 |
| 11 | `llama-io` | `model.LlamaIO` | 0.19 | 3 | 2 | 31.2 |
| 12 | `models.models` | `model.InferencePipeline` | 0.00 | 114 | 0 | 23.7 |
| 13 | `llama-kv-cells` | `model.LlamaKvCells` | 0.73 | 1 | 2 | 23.7 |
| 14 | `llama-cparams` | `model.LlamaCparams` | 0.00 | 5 | 1 | 22.5 |
| 15 | `llama-chat` | `model.LlamaChat` | 0.35 | 1 | 1 | 13.6 |
| 16 | `llama-memory-recurrent` | `model.LlamaMemoryRecurrent` | 0.00 | 12 | 0 | 12.8 |
| 17 | `llama-model` | `model.LlamaModel` | 0.00 | 12 | 0 | 12.8 |
| 18 | `llama-vocab` | `model.LlamaVocab` | 0.00 | 6 | 0 | 9.7 |
| 19 | `llama-context` | `model.LlamaAttention` | 0.00 | 4 | 0 | 8.0 |
| 20 | `llama-model-loader` | `gguf.ModelLoader` | 0.00 | 4 | 0 | 8.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **llama** → `model.Grammar`
  - Similarity: 0.00
  - Dependencies: 16

- **llama-hparams** → `model.LlamaHparams`
  - Similarity: 0.30
  - Dependencies: 4

- **llama-graph** → `model.LlamaGraph`
  - Similarity: 0.42
  - Dependencies: 7
  - Lint issues: 8

- **llama-memory** → `model.LlamaMemory`
  - Similarity: 0.07
  - Dependencies: 7

- **llama-mmap** → `model.LlamaMmap`
  - Similarity: 0.19
  - Dependencies: 5
  - Lint issues: 2

- **llama-impl** → `model.LlamaImpl`
  - Similarity: 0.16
  - Dependencies: 19

- **llama-arch** → `model.LlamaArch`
  - Similarity: 0.09
  - Dependencies: 5

- **llama-model-saver** → `model.LlamaModelSaver`
  - Similarity: 0.28
  - Dependencies: 1

- **llama-batch** → `model.LlamaBatch`
  - Similarity: 0.56
  - Dependencies: 7
  - Lint issues: 1

- **llama-io** → `model.LlamaIO`
  - Similarity: 0.19
  - Dependencies: 3

- **models.models** → `model.InferencePipeline`
  - Similarity: 0.00
  - Dependencies: 114
  - TODOs: 2

- **llama-cparams** → `model.LlamaCparams`
  - Similarity: 0.00
  - Dependencies: 5

- **llama-chat** → `model.LlamaChat`
  - Similarity: 0.35
  - Dependencies: 1

- **llama-memory-recurrent** → `model.LlamaMemoryRecurrent`
  - Similarity: 0.00
  - Dependencies: 12
  - TODOs: 1
  - Lint issues: 4

- **llama-model** → `model.LlamaModel`
  - Similarity: 0.00
  - Dependencies: 12
  - TODOs: 1
  - Lint issues: 5

- **llama-vocab** → `model.LlamaVocab`
  - Similarity: 0.00
  - Dependencies: 6
  - TODOs: 2
  - Lint issues: 8

- **llama-context** → `model.LlamaAttention`
  - Similarity: 0.00
  - Dependencies: 4
  - TODOs: 4
  - Lint issues: 18

- **llama-model-loader** → `gguf.ModelLoader`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 1

- **llama-kv-cache** → `model.KVCache`
  - Similarity: 0.00
  - Dependencies: 4
  - TODOs: 5
  - Lint issues: 8

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

- **llama-ext** → `model.LlamaExt`
  - Similarity: 0.00
  - Dependencies: 2
  - Lint issues: 4

- **llama-sampler** → `model.LlamaSampler`
  - Similarity: 0.00
  - Dependencies: 1
  - TODOs: 3
  - Lint issues: 2

- **unicode** → `model.LlamaUnicode`
  - Similarity: 0.00
  - Dependencies: 1
  - TODOs: 1
  - Lint issues: 1

- **unicode-data** → `model.LlamaUnicodeData`
  - Similarity: 0.32
  - Dependencies: 1
  - TODOs: 3

## Missing Files (Top by Dependents)

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

