package ai.solace.llamakotlin.core

import kotlin.js.ExperimentalJsExport
import kotlin.js.JsExport

@OptIn(ExperimentalJsExport::class)
@JsExport
class EmberTensor internal constructor(private val tensor: GGMLTensor) {
    val name: String get() = tensor.name
    val type: String get() = tensor.type.name
    val shape: Array<Long> get() = tensor.ne.copyOf().toTypedArray()
}

@OptIn(ExperimentalJsExport::class)
@JsExport
object Ember {
    fun tensorF32(name: String, vararg dims: Int, value: Float = 0f): EmberTensor {
        require(dims.isNotEmpty()) { "Tensor shape must have at least one dimension" }
        val shape = LongArray(GGML_MAX_DIMS) { 1L }
        dims.forEachIndexed { index, dim ->
            require(index < GGML_MAX_DIMS) { "Rank greater than $GGML_MAX_DIMS is not supported" }
            shape[index] = dim.toLong()
        }
        val tensor = GGMLTensor(type = GGMLType.F32, name = name, ne = shape)
        tensor.data = FloatArray(tensor.numElements().toInt()) { value }
        return EmberTensor(tensor)
    }
}
