# High Priority Ports - Action Plan

## Top 20 Files by Impact

Priority = (missing functions + missing types) × (10 + log1p(deps) × 2) + log1p(deps) × (1 − similarity) × 5

| Rank | Source | Target | Similarity | Deps | SymDeficit | Priority |
|------|--------|--------|------------|------|-----------|----------|
| 1 | `models.models` | `model.InferencePipeline` | 0.00 | 114 | 255 | 4993.6 |
| 2 | `llama-context` | `model.LlamaAttention` | 0.00 | 4 | 195 | 2585.7 |
| 3 | `llama-model` | `model.LlamaModel` | 0.00 | 12 | 92 | 1404.8 |
| 4 | `llama-kv-cache` | `model.KVCache` | 0.00 | 4 | 81 | 1078.8 |
| 5 | `llama-model-loader` | `gguf.ModelLoader` | 0.00 | 4 | 58 | 774.7 |
| 6 | `llama-grammar` | `model.Grammar` | 0.00 | 1 | 53 | 606.9 |
| 7 | `llama-model-saver` | `model.GGMLIntegration` | 0.00 | 1 | 0 | 3.5 |
| 8 | `models.maincoder` | `main` | 0.16 | 0 | 0 | 0.0 |

## Critical Issues (Similarity < 0.60 with Dependencies)

These files need immediate attention:

- **models.models** → `model.InferencePipeline`
  - Similarity: 0.00
  - Dependencies: 114

- **llama-context** → `model.LlamaAttention`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 4

- **llama-model** → `model.LlamaModel`
  - Similarity: 0.00
  - Dependencies: 12
  - Lint issues: 5

- **llama-kv-cache** → `model.KVCache`
  - Similarity: 0.00
  - Dependencies: 4

- **llama-model-loader** → `gguf.ModelLoader`
  - Similarity: 0.00
  - Dependencies: 4
  - Lint issues: 1

- **llama-grammar** → `model.Grammar`
  - Similarity: 0.00
  - Dependencies: 1

- **llama-model-saver** → `model.GGMLIntegration`
  - Similarity: 0.00
  - Dependencies: 1
  - Lint issues: 10

## Missing Files (Top by Dependents)

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

