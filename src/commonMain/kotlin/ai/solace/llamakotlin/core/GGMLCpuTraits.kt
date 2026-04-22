package ai.solace.llamakotlin.core

// port-lint: source ggml/src/ggml-cpu/traits.h

/**
 * Port of `ggml::cpu::tensor_traits` — interface registered in tensor->extra
 * to provide custom compute_forward / work_size for accelerated ops.
 */
interface GGMLCpuTensorTraits {
    fun workSize(nThreads: Int, op: GGMLTensor, size: LongArray): Boolean
    fun computeForward(params: GGMLComputeParams, op: GGMLTensor): Boolean
}

/**
 * Port of `ggml::cpu::extra_buffer_type` — interface for extra accelerator buffer types
 * that can intercept tensor ops on the CPU backend.
 */
interface GGMLCpuExtraBufferType {
    fun supportsOp(dev: GGMLBackendDevice, op: GGMLTensor): Boolean
    fun getTensorTraits(op: GGMLTensor): GGMLCpuTensorTraits?
}

/**
 * Port of `ggml_cpu_extra_compute_forward` from traits.cpp.
 * Returns true if the op was handled by an extra accelerator.
 */
fun ggmlCpuExtraComputeForward(params: GGMLComputeParams, op: GGMLTensor): Boolean {
    val extra = op.extra
    if (extra is GGMLCpuTensorTraits) {
        return extra.computeForward(params, op)
    }
    return false
}

/**
 * Port of `ggml_cpu_extra_work_size` from traits.cpp.
 * If the tensor has extra traits, queries them for required work buffer size.
 */
fun ggmlCpuExtraWorkSize(nThreads: Int, op: GGMLTensor, size: LongArray): Boolean {
    val extra = op.extra
    if (extra is GGMLCpuTensorTraits) {
        return extra.workSize(nThreads, op, size)
    }
    return false
}

/**
 * Port of `ggml_backend_cpu_get_extra_buffer_types` from ggml-cpu.cpp.
 * Returns the registered extra buffer types for the CPU backend.
 */
fun ggmlBackendCpuGetExtraBufferTypes(): MutableList<GGMLBackendBufferType> {
    return mutableListOf()
}
