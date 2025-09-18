package ai.solace.llamakotlin.core

/**
 * Returns the platform-specific Metal backend registration if available.
 */
internal expect fun metalBackendRegistration(): GGMLBackendRegistration?

/**
 * Returns an initialized Metal backend instance if the platform supports it.
 */
internal expect fun metalBackendInstance(): GGMLBackend?

/**
 * Returns descriptive Metal device information when available.
 */
internal expect fun metalDeviceInfo(): Map<String, Any>?
