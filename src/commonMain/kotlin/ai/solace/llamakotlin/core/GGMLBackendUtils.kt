// port-lint: source ggml/src/ggml-backend.cpp
package ai.solace.llamakotlin.core

/**
 * Backend utilities and integration helpers for GGML operations.
 * 
 * This file provides utility functions for working with backends,
 * hybrid execution, and backend selection logic.
 */

/**
 * Backend selection strategy for hybrid execution
 */
enum class GGMLBackendSelectionStrategy {
    /** Always use CPU backend */
    CPU_ONLY,
    /** Automatic selection (currently identical to CPU_ONLY until new backends exist) */
    AUTO
}

/**
 * Backend manager for handling multiple backends and hybrid execution
 */
class GGMLBackendManager {
    private val availableBackends = mutableMapOf<String, GGMLBackend>()
    private var primaryBackend: GGMLBackend? = null
    private var fallbackBackend: GGMLBackend? = null
    private var selectionStrategy = GGMLBackendSelectionStrategy.AUTO
    
    init {
        initializeBackends()
    }
    
    /**
     * Initialize available backends
     */
    private fun initializeBackends() {
        val cpuBackend = GGMLCpuBackend()
        availableBackends[cpuBackend.getName()] = cpuBackend
        primaryBackend = cpuBackend
        fallbackBackend = cpuBackend
    }
    
    /**
     * Get the list of available backend names
     */
    fun getAvailableBackends(): List<String> {
        return availableBackends.keys.toList()
    }
    
    /**
     * Get a backend by name
     */
    fun getBackend(name: String): GGMLBackend? {
        return availableBackends[name]
    }
    
    /**
     * Get the primary backend
     */
    fun getPrimaryBackend(): GGMLBackend? = primaryBackend
    
    /**
     * Get the fallback backend (usually CPU)
     */
    fun getFallbackBackend(): GGMLBackend? = fallbackBackend

    /**
     * Simple list of backend instances (for compatibility)
     */
    fun getBackends(): List<GGMLBackend> = availableBackends.values.toList()
    
    /**
     * Set the backend selection strategy
     */
    fun setSelectionStrategy(strategy: GGMLBackendSelectionStrategy) {
        selectionStrategy = strategy
    }
    
    /**
     * Select the best backend for a given operation
     */
    fun selectBackend(tensor: GGMLTensor): GGMLBackend? {
        return fallbackBackend
    }
    
    /**
     * Create a graph with automatic backend selection
     */
    fun createGraphWithBackend(size: Int, strategy: GGMLBackendSelectionStrategy? = null): GGMLCGraph {
        val oldStrategy = selectionStrategy
        strategy?.let { setSelectionStrategy(it) }
        
        val backend = fallbackBackend
        val graph = createGraph(size, backend)
        setSelectionStrategy(oldStrategy) // Restore old strategy
        return graph
    }
    
    /**
     * Compute a graph with hybrid execution
     */
    fun computeGraphHybrid(graph: GGMLCGraph): GGMLStatus {
        val allocator = graph.allocator
        if (allocator == null) {
            return GGMLStatus.FAILED
        }
        
        // For now, use the graph's associated backend
        // In a full hybrid implementation, we would analyze each node
        // and potentially execute different nodes on different backends
        return computeGraphWithBackend(graph)
    }
    
    /**
     * Get backend information for debugging
     */
    fun getBackendInfo(): Map<String, Any> {
        val info = mutableMapOf<String, Any>()
        
        info["availableBackends"] = availableBackends.keys.toList()
        info["primaryBackend"] = primaryBackend?.getName() ?: "None"
        info["fallbackBackend"] = fallbackBackend?.getName() ?: "None" 
        info["selectionStrategy"] = selectionStrategy.name

        // Add backend-specific info
        availableBackends.forEach { (name, backend) ->
            info["${name}_guid"] = backend.getGuid()
            info["${name}_bufferType"] = backend.getDefaultBufferType().getName()
            info["${name}_alignment"] = backend.getAlignment()
            info["${name}_maxSize"] = backend.getMaxSize()
        }
        
        return info
    }
    
    /**
     * Clean up all backends
     */
    fun cleanup() {
        availableBackends.values.forEach { backend ->
            try {
                backend.free()
            } catch (e: Exception) {
                println("Error freeing backend ${backend.getName()}: ${e.message}")
            }
        }
        availableBackends.clear()
        primaryBackend = null
        fallbackBackend = null
    }
}

/**
 * Global backend manager instance
 */
val globalBackendManager = GGMLBackendManager()

/**
 * Convenience function to create a graph with the global backend manager
 */
fun createGraphWithGlobalBackend(
    size: Int, 
    strategy: GGMLBackendSelectionStrategy = GGMLBackendSelectionStrategy.AUTO
): GGMLCGraph {
    return globalBackendManager.createGraphWithBackend(size, strategy)
}

/**
 * Convenience function to compute a graph with hybrid execution
 */
fun computeGraphHybrid(graph: GGMLCGraph): GGMLStatus {
    return globalBackendManager.computeGraphHybrid(graph)
}
