// port-lint: source llama.cpp/src/llama-hparams.h
package ai.solace.llamakotlin.model

// =============================================================================
// LlamaHparams – supplementary hyperparameter types
// Ported from: llama-hparams.h
//
// The main data class `LlamaModelHParams` (port of `struct llama_hparams`)
// lives in LlamaModel.kt.  This file defines the small sub-structs and any
// additional enums that the C++ header declares alongside the main struct.
// =============================================================================

// NOTE: LlamaExpertGatingFuncType and LlamaSwaType are already defined in
//       LlamaModel.kt and KVCache.kt respectively.
// NOTE: LLAMA_MAX_LAYERS and LLAMA_MAX_EXPERTS are defined in LlamaModel.kt.

/**
 * PosNet hyperparameters (WavTokenizer).
 *
 * Port of `struct llama_hparams_posnet` from `llama-hparams.h`.
 *
 * @param nEmbd  Embedding dimension used by PosNet layers.
 * @param nLayer Number of PosNet layers.
 */
data class LlamaHparamsPosnet(
    var nEmbd: Int = 0,
    var nLayer: Int = 0,
)

/**
 * ConvNext hyperparameters (WavTokenizer).
 *
 * Port of `struct llama_hparams_convnext` from `llama-hparams.h`.
 *
 * @param nEmbd  Embedding dimension used by ConvNext layers.
 * @param nLayer Number of ConvNext layers.
 */
data class LlamaHparamsConvnext(
    var nEmbd: Int = 0,
    var nLayer: Int = 0,
)
