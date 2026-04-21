// port-lint: source llama.cpp/src/llama-cparams.h
package ai.solace.llamakotlin.model

// =============================================================================
// LlamaCparams – context-parameter constants and callback types
// Ported from: llama-cparams.h
//
// The main data classes `LlamaCParams` (computed context parameters) and
// `LlamaContextParams` (user-facing context configuration) are defined in
// LlamaAttention.kt because they were ported alongside the context logic.
// This file contributes the constant `LLAMA_MAX_SEQ_CPARAMS` and the eval
// callback typealias that the C++ header declares alongside the struct.
// =============================================================================

/**
 * Maximum number of sequences supported per context.
 *
 * This is the value used in the C++ header (`#define LLAMA_MAX_SEQ 256`).
 * The project also defines [LLAMA_MAX_SEQ] in `KVCache.kt` (currently 64)
 * for the KV-cache layer.  This constant preserves the original C++ value
 * for any code that needs the header-level cap.
 */
const val LLAMA_MAX_SEQ_CPARAMS: Int = 256

/**
 * Callback invoked after each backend-scheduler evaluation step.
 *
 * Port of `ggml_backend_sched_eval_callback` used by `llama_cparams`.
 *
 * - **nodeIndex**: index of the tensor node just evaluated.
 * - **isLast**: `true` when this is the final node in the graph.
 * - **return**: `true` to continue evaluation, `false` to abort early.
 */
typealias GgmlBackendSchedEvalCallback = ((nodeIndex: Int, isLast: Boolean) -> Boolean)?
