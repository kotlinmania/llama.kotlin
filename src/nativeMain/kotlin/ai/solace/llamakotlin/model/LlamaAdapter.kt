// port-lint: source llama.cpp/src/llama-adapter.h
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

// =============================================================================
// LlamaAdapter – LoRA and control-vector adapter types
// Ported from: llama-adapter.h
// =============================================================================

// ---------------------------------------------------------------------------
// Control Vectors (llama_adapter_cvec)
// ---------------------------------------------------------------------------

/**
 * A set of per-layer control vectors that can be added to the hidden state
 * to steer model behaviour without fine-tuning.
 *
 * Port of `llama_adapter_cvec` from `llama-adapter.h`.
 *
 * @property layerStart  First layer index (inclusive) that receives the vector.
 * @property layerEnd    Last layer index (exclusive).
 * @property tensors     Per-layer control-vector tensors (indexed by layer number).
 */
class LlamaAdapterCvec(
    var layerStart: Int = -1,
    var layerEnd: Int = -1,
    val tensors: MutableList<GGMLTensor?> = mutableListOf(),
) {
    /**
     * Return the control-vector tensor for layer [il], or `null` if no
     * vector is assigned to that layer.
     */
    fun tensorFor(il: Int): GGMLTensor? {
        if (il < layerStart || il >= layerEnd) return null
        val idx = il - layerStart
        return tensors.getOrNull(idx)
    }

    /**
     * Build a graph node that applies the control vector to [cur] at layer [il].
     *
     * Returns [cur] unchanged if no vector exists for that layer.
     */
    fun applyTo(cur: GGMLTensor, il: Int): GGMLTensor {
        val t = tensorFor(il) ?: return cur
        TODO("Port llama_adapter_cvec::apply_to — add t to cur via ggml_add")
    }

    /**
     * Load control-vector data into this adapter.
     *
     * @param data     Raw float data for all layers.
     * @param len      Total number of floats.
     * @param nEmbd    Model embedding dimension.
     * @param ilStart  First layer to apply (inclusive).
     * @param ilEnd    Last layer to apply (exclusive).
     * @return `true` on success.
     */
    fun apply(data: FloatArray, len: Int, nEmbd: Int, ilStart: Int, ilEnd: Int): Boolean {
        TODO("Port llama_adapter_cvec::apply")
    }
}

// ---------------------------------------------------------------------------
// LoRA weight pair (llama_adapter_lora_weight)
// ---------------------------------------------------------------------------

/**
 * A pair of low-rank matrices (A and B) that form a single LoRA delta.
 *
 * The actual weight update is: `delta = (B × A) × scale`, where
 * `scale = alpha / rank * adapter_scale`.
 *
 * Port of `llama_adapter_lora_weight` from `llama-adapter.h`.
 *
 * @property a  The A (down-projection) tensor, or `null` if not yet loaded.
 * @property b  The B (up-projection) tensor, or `null` if not yet loaded.
 */
data class LlamaAdapterLoraWeight(
    var a: GGMLTensor? = null,
    var b: GGMLTensor? = null,
) {
    /**
     * Compute the effective scale for this weight.
     *
     * @param alpha         The LoRA alpha hyper-parameter (0 means use [adapterScale] directly).
     * @param adapterScale  Global adapter scaling factor.
     * @return The multiplier to apply to the `B × A` product.
     */
    fun getScale(alpha: Float, adapterScale: Float): Float {
        val bTensor = b ?: error("LoRA weight B tensor is null")
        val rank = bTensor.ne[0].toFloat()
        return if (alpha != 0f) adapterScale * alpha / rank else adapterScale
    }
}

// ---------------------------------------------------------------------------
// LoRA adapter (llama_adapter_lora)
// ---------------------------------------------------------------------------

/**
 * A complete LoRA adapter loaded from a GGUF file.
 *
 * Port of `llama_adapter_lora` from `llama-adapter.h`.
 *
 * @property alpha  The LoRA alpha hyper-parameter controlling scaling.
 * @property abMap  Map from base-model tensor name to the corresponding LoRA weight pair.
 * @property ggufKv  Metadata key-value pairs from the adapter's GGUF file.
 * @property aloraInvocationTokens  Tokens that trigger activated-LoRA (aLoRA) behaviour.
 */
class LlamaAdapterLora(
    var alpha: Float = 0f,
    val abMap: MutableMap<String, LlamaAdapterLoraWeight> = mutableMapOf(),
    val ggufKv: MutableMap<String, String> = mutableMapOf(),
    val aloraInvocationTokens: MutableList<Int> = mutableListOf(),
) {
    /**
     * Look up the LoRA delta for a given base-model weight tensor.
     *
     * @param w  The base-model tensor (looked up by [GGMLTensor.name]).
     * @return The matching [LlamaAdapterLoraWeight], or `null` if this
     *         adapter doesn't override that weight.
     */
    fun getWeight(w: GGMLTensor): LlamaAdapterLoraWeight? = abMap[w.name]

    /**
     * Estimated number of compute-graph nodes required for this adapter.
     *
     * Each weight pair contributes ~6 nodes: A matmul, B matmul, scale,
     * add, and two intermediate results.
     */
    fun getNumNodes(): Int = abMap.size * 6
}

/**
 * Active LoRA adapters mapped to their per-adapter scaling factor.
 *
 * Port of `llama_adapter_loras` — `std::unordered_map<llama_adapter_lora *, float>`.
 */
typealias LlamaAdapterLoras = MutableMap<LlamaAdapterLora, Float>
