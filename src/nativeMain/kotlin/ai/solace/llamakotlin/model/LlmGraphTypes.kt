/**
 * Core graph type enums and the [LlmGraphInput] interface.
 * Originally from `llama-graph.h`; primary port lives in [LlamaGraph.kt].
 *
 * Concrete input implementations, result / parameter containers, and the base
 * [LlmGraphContext] builder class live in [LlamaGraph.kt].
 */
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/** Top-level graph variant used during inference. */
enum class LlmGraphType {
    DEFAULT,
    ENCODER,
    DECODER,
}

/** Activation / gating function applied inside feed-forward blocks. */
enum class LlmFfnOpType {
    SILU,
    GELU,
    RELU,
    RELU_SQR,
    SWIGLU,
    GEGLU,
    REGLU,
    SWIGLU_OAI_MOE,
}

/** How the FFN gate tensor is combined with the up-projection. */
enum class LlmFfnGateType {
    /** Gate applied sequentially. */
    SEQ,
    /** Gate applied in parallel. */
    PAR,
}

/** Normalisation variant applied to hidden states. */
enum class LlmNormType {
    NORM,
    RMS,
    GROUP,
}

// ---------------------------------------------------------------------------
// Graph callback type
// ---------------------------------------------------------------------------

/**
 * Callback that allows custom logic to be applied to each tensor during graph
 * construction (e.g. naming, offloading, ggml-alloc tagging).
 *
 * Port of `llm_graph_cb`.
 */
typealias LlmGraphCb = (ubatch: LlamaUBatch, cur: GGMLTensor, name: String, il: Int) -> Unit

// ---------------------------------------------------------------------------
// LlmGraphInput – interface
// ---------------------------------------------------------------------------

/**
 * A single named input that must be populated before graph evaluation.
 *
 * Each implementation owns one or more [GGMLTensor] fields that are filled by
 * [setInput] from the current micro-batch and can optionally report whether a
 * previously built graph is still valid via [canReuse].
 */
interface LlmGraphInput {
    /** Populate owned tensors from the supplied micro-batch. */
    fun setInput(ubatch: LlamaUBatch)

    /** Return `true` when the graph built with the previous params can be reused. */
    fun canReuse(params: LlmGraphParams): Boolean = false
}
