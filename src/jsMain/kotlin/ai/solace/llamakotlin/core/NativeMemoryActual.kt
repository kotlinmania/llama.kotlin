package ai.solace.llamakotlin.core

/**
 * JS actual for [ggml_aligned_malloc].
 *
 * Falls back to [ByteArray] — JavaScript does not expose aligned memory allocation.
 */
actual fun ggml_aligned_malloc(size: Long): Any? {
    if (size == 0L) return null
    return ByteArray(size.toInt())
}

/**
 * JS actual for [ggml_aligned_free].
 *
 * No-op: JavaScript garbage collector handles deallocation.
 */
actual fun ggml_aligned_free(ptr: Any?, size: Long) {
    // No-op: GC handles ByteArray deallocation
}

/**
 * JS actual for [createDefaultCpuBufferType].
 *
 * Returns the ByteArray-backed [GGMLCpuBufferType].
 */
actual fun createDefaultCpuBufferType(): GGMLBackendBufferType {
    return GGMLCpuBufferType()
}
