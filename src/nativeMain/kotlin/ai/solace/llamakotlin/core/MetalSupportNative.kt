package ai.solace.llamakotlin.core

internal actual fun metalBackendRegistration(): GGMLBackendRegistration? =
    GGMLBackendRegistration(
        name = "Metal",
        initFunction = { _ -> GGMLMetalBackend() },
        defaultBufferType = GGMLMetalBufferType()
    )

internal actual fun metalBackendInstance(): GGMLBackend? {
    val backend = GGMLMetalBackend()
    return if (backend.initialize()) backend else null
}

internal actual fun metalDeviceInfo(): Map<String, Any>? =
    metalBackendInstance()?.let { backend ->
        if (backend is GGMLMetalBackend) backend.getDeviceInfo() else null
    }
