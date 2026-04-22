package ai.solace.llamakotlin.core

// port-lint: source ggml/src/ggml-cpu/traits.h + traits.cpp

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
 * Iterates over all registered extra buffer types and delegates to the
 * first one whose tensor_traits can handle the op.
 */
fun ggmlCpuExtraComputeForward(params: GGMLComputeParams, op: GGMLTensor): Boolean {
    for (extra in ggmlBackendCpuGetExtraBufferTypes()) {
        val context = extra.getContext()
        if (context is GGMLCpuExtraBufferType) {
            val tensorTraits = context.getTensorTraits(op)
            if (tensorTraits != null && tensorTraits.computeForward(params, op)) {
                return true
            }
        }
    }
    return false
}

/**
 * Port of `ggml_cpu_extra_work_size` from traits.cpp.
 * Iterates over all registered extra buffer types and queries each for
 * the required work buffer size.
 */
fun ggmlCpuExtraWorkSize(nThreads: Int, op: GGMLTensor, size: LongArray): Boolean {
    for (extra in ggmlBackendCpuGetExtraBufferTypes()) {
        val context = extra.getContext()
        if (context is GGMLCpuExtraBufferType) {
            val tensorTraits = context.getTensorTraits(op)
            if (tensorTraits != null && tensorTraits.workSize(nThreads, op, size)) {
                return true
            }
        }
    }
    return false
}

/**
 * Port of `ggml_backend_cpu_get_extra_buffer_types` from ggml-cpu.cpp.
 * Returns the registered extra buffer types for the CPU backend.
 */
fun ggmlBackendCpuGetExtraBufferTypes(): List<GGMLBackendBufferType> {
    return emptyList()
}
