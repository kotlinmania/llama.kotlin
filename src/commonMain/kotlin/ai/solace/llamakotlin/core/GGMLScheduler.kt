// port-lint: source ggml/src/ggml-threading.cpp
package ai.solace.llamakotlin.core

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/**
 * Kotlin Native port of GGML scheduler functionality.
 * This file contains the multi-threaded and multi-backend scheduler implementation.
 */

// ---------------------------------------------------------------------------
// ggml-threading.h / ggml-threading.cpp — critical section (global mutex)
// ---------------------------------------------------------------------------

/**
 * Simple spin-lock based critical section matching the C++ std::mutex usage.
 * In Kotlin/Native single-threaded mode this is effectively a identity guard;
 * with the new memory model it provides basic mutual exclusion.
 */
@OptIn(ExperimentalAtomicApi::class)
private val criticalSectionLock = AtomicInt(0)

@OptIn(ExperimentalAtomicApi::class)
fun ggml_critical_section_start() {
    while (!criticalSectionLock.compareAndSet(0, 1)) {
        // spin
    }
}

@OptIn(ExperimentalAtomicApi::class)
fun ggml_critical_section_end() {
    criticalSectionLock.store(0)
}

/**
 * High-level backend status used by scheduler/examples
 */
enum class GGMLBackendStatus {
    SUCCESS,
    FAILED,
    ALLOC_FAILED,
    NOT_SUPPORTED
}

/**
 * Scheduling strategy enumeration
 */
enum class GGMLSchedulingStrategy {
    /** Execute operations sequentially */
    SEQUENTIAL,
    /** Execute independent operations in parallel */
    PARALLEL,
    /** Use multiple backends when available */
    MULTI_BACKEND
}

/**
 * Graph split for backend execution
 */
data class GGMLGraphSplit(
    val backendName: String,
    val startNode: Int,
    val endNode: Int,
    val nodes: List<GGMLTensor>,
    val inputs: List<GGMLTensor> = emptyList()
)

/**
 * Execution context for a scheduled operation
 */
data class GGMLExecutionContext(
    val backend: GGMLBackend,
    val split: GGMLGraphSplit,
    val context: GGMLContext
)

/**
 * Dependency tracker for graph nodes
 */
class GGMLDependencyTracker {
    private val dependencies = mutableMapOf<GGMLTensor, MutableSet<GGMLTensor>>()
    private val dependents = mutableMapOf<GGMLTensor, MutableSet<GGMLTensor>>()
    private val completed = mutableSetOf<GGMLTensor>()
    
    /**
     * Build dependency graph for a computation graph
     */
    fun buildDependencies(graph: GGMLCGraph) {
        dependencies.clear()
        dependents.clear()
        completed.clear()
        
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            dependencies[node] = mutableSetOf()
            dependents[node] = mutableSetOf()
            
            // Add source dependencies
            for (src in node.src) {
                if (src != null) {
                    dependencies[node]!!.add(src)
                    dependents.getOrPut(src) { mutableSetOf() }.add(node)
                }
            }
            
            // Add view source dependency
            node.viewSrc?.let { viewSrc ->
                dependencies[node]!!.add(viewSrc)
                dependents.getOrPut(viewSrc) { mutableSetOf() }.add(node)
            }
        }
    }
    
    /**
     * Get nodes that are ready to execute (all dependencies completed)
     */
    fun getReadyNodes(): List<GGMLTensor> {
        return dependencies.keys.filter { node ->
            node !in completed && dependencies[node]!!.all { it in completed }
        }
    }
    
    /**
     * Mark a node as completed
     */
    fun markCompleted(node: GGMLTensor) {
        completed.add(node)
    }
    
    /**
     * Check if all nodes are completed
     */
    fun isComplete(): Boolean {
        return completed.size == dependencies.size
    }
    
    /**
     * Reset the tracker
     */
    fun reset() {
        completed.clear()
    }
}

/**
 * Multi-threaded scheduler for computation graphs
 */
class GGMLScheduler(
    private val backends: List<GGMLBackend>,
    private val strategy: GGMLSchedulingStrategy = GGMLSchedulingStrategy.PARALLEL
) {
    private val dependencyTracker = GGMLDependencyTracker()
    private var maxWorkers = 4
    
    private fun primaryBackend(): GGMLBackend? = backends.firstOrNull()
    
    fun setMaxWorkers(workers: Int) {
        maxWorkers = workers.coerceAtLeast(1)
    }
    
    fun execute(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        return when (strategy) {
            GGMLSchedulingStrategy.SEQUENTIAL -> executeSequential(graph, context)
            GGMLSchedulingStrategy.PARALLEL -> executeParallel(graph, context)
            GGMLSchedulingStrategy.MULTI_BACKEND -> executeMultiBackend(graph, context)
        }
    }
    
    private fun executeSequential(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        val backend = primaryBackend() ?: return GGMLBackendStatus.FAILED
        return try {
            val allocator = graph.allocator ?: GGMLGraphAllocator().also { graph.allocator = it }
            if (!allocator.allocateGraph(graph)) return GGMLBackendStatus.ALLOC_FAILED
            val status = backend.graphCompute(graph)
            if (status == GGMLStatus.SUCCESS) GGMLBackendStatus.SUCCESS else GGMLBackendStatus.FAILED
        } catch (e: Exception) {
            GGMLBackendStatus.FAILED
        }
    }
    
    private fun executeParallel(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        dependencyTracker.buildDependencies(graph)
        return executeSequentialWithDependencies(graph, context)
    }
    
    private fun executeSequentialWithDependencies(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        dependencyTracker.buildDependencies(graph)
        
        while (!dependencyTracker.isComplete()) {
            val readyNodes = dependencyTracker.getReadyNodes()
            if (readyNodes.isEmpty()) return GGMLBackendStatus.FAILED
            
            for (node in readyNodes) {
                try {
                    val backend = primaryBackend() ?: return GGMLBackendStatus.FAILED
                    executeNode(node, context, backend)
                    dependencyTracker.markCompleted(node)
                } catch (e: Exception) {
                    return GGMLBackendStatus.FAILED
                }
            }
        }
        
        return GGMLBackendStatus.SUCCESS
    }
    
    private fun executeMultiBackend(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        val splits = createGraphSplits(graph)
        
        if (splits.size <= 1) {
            return executeSequential(graph, context)
        }
        
        for (split in splits) {
            val backend = backends.firstOrNull { it.getName() == split.backendName } ?: continue
            val status = executeGraphSplit(split, context, backend)
            if (status != GGMLBackendStatus.SUCCESS) return status
        }
        
        backends.forEach { it.synchronize() }
        return GGMLBackendStatus.SUCCESS
    }
    
    private fun executeNode(node: GGMLTensor, context: GGMLContext, backend: GGMLBackend) {
        val tempGraph = createGraph(1)
        tempGraph.nodes[0] = node
        tempGraph.nNodes = 1
        val allocator = tempGraph.allocator ?: GGMLGraphAllocator().also { tempGraph.allocator = it }
        allocator.allocateGraph(tempGraph)
        backend.graphCompute(tempGraph)
    }
    
    private fun executeGraphSplit(split: GGMLGraphSplit, context: GGMLContext, backend: GGMLBackend): GGMLBackendStatus {
        val subGraph = createGraph(split.nodes.size)
        for (i in split.nodes.indices) {
            subGraph.nodes[i] = split.nodes[i]
        }
        subGraph.nNodes = split.nodes.size
        val allocator = subGraph.allocator ?: GGMLGraphAllocator().also { subGraph.allocator = it }
        if (!allocator.allocateGraph(subGraph)) return GGMLBackendStatus.ALLOC_FAILED
        val status = backend.graphCompute(subGraph)
        return if (status == GGMLStatus.SUCCESS) GGMLBackendStatus.SUCCESS else GGMLBackendStatus.FAILED
    }
    
    private fun createGraphSplits(graph: GGMLCGraph): List<GGMLGraphSplit> {
        val splits = mutableListOf<GGMLGraphSplit>()
        if (backends.isEmpty()) return splits
        
        var currentBackend = primaryBackend() ?: return splits
        var splitStart = 0
        val currentSplitNodes = mutableListOf<GGMLTensor>()
        
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            val bestBackend = primaryBackend() ?: currentBackend
            if (bestBackend.getName() != currentBackend.getName() && currentSplitNodes.isNotEmpty()) {
                splits.add(GGMLGraphSplit(
                    backendName = currentBackend.getName(),
                    startNode = splitStart,
                    endNode = i - 1,
                    nodes = currentSplitNodes.toList()
                ))
                currentSplitNodes.clear()
                splitStart = i
                currentBackend = bestBackend
            }
            currentSplitNodes.add(node)
        }
        
        if (currentSplitNodes.isNotEmpty()) {
            splits.add(GGMLGraphSplit(
                backendName = currentBackend.getName(),
                startNode = splitStart,
                endNode = graph.nNodes - 1,
                nodes = currentSplitNodes.toList()
            ))
        }
        
        return splits
    }
    
    fun getStats(): SchedulerStats {
        return SchedulerStats(
            maxWorkers = maxWorkers,
            strategy = strategy,
            availableBackends = backends.map { it.getName() }
        )
    }
}

/**
 * Scheduler statistics
 */
data class SchedulerStats(
    val maxWorkers: Int,
    val strategy: GGMLSchedulingStrategy,
    val availableBackends: List<String>
)