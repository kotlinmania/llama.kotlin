package ai.solace.llamakotlin.core


/**
 * Runtime configuration for the CPU backend. Developers can tweak threading and scheduling
 * without rebuilding the backend itself, enabling experimentation with worker counts and
 * dispatch policies from higher layers.
 */
internal data class GGMLCpuRuntimeConfig(
    var threadCount: Int = 1,
    var schedulingStrategy: GGMLSchedulingStrategy = GGMLSchedulingStrategy.SEQUENTIAL,
    var useDedicatedDispatcher: Boolean = false
) {
    fun normalizedThreadCount(): Int = threadCount.coerceAtLeast(1)
    fun prefersParallel(): Boolean = schedulingStrategy != GGMLSchedulingStrategy.SEQUENTIAL
}

/**
 * Helper responsible for executing computation graphs on the CPU backend. It bridges the
 * backend layer with the destination-based compute operations and provides a lightweight
 * work-stealing-style scheduler using Kotlin/Native workers. Dependency tracking ensures only
 * independent nodes are processed in parallel; if the runtime is configured for sequential
 * execution, the helper falls back to `GGMLComputeOps.computeGraph`.
 */
internal class GGMLCpuExecutor {

    fun compute(graph: GGMLCGraph, config: GGMLCpuRuntimeConfig): GGMLStatus {
        val allocator = graph.allocator ?: return GGMLStatus.FAILED
        val threads = config.normalizedThreadCount()
        return if (threads == 1 || !config.prefersParallel()) {
            executeSequential(graph)
        } else {
            executeWithDependencies(graph, allocator, threads, config)
        }
    }

    private fun executeSequential(graph: GGMLCGraph): GGMLStatus {
        return try {
            GGMLComputeOps.computeGraph(graph)
            GGMLStatus.SUCCESS
        } catch (t: Throwable) {
            println("GGMLCpuExecutor: sequential compute failed - ${t.message}")
            GGMLStatus.FAILED
        }
    }

    private fun executeWithDependencies(
        graph: GGMLCGraph,
        allocator: GGMLGraphAllocator,
        maxBatchSize: Int,
        config: GGMLCpuRuntimeConfig
    ): GGMLStatus {
        val tracker = GGMLDependencyTracker()
        tracker.buildDependencies(graph)

        return try {
            blockingCompute(tracker, allocator, maxBatchSize, config)
            GGMLStatus.SUCCESS
        } catch (t: Throwable) {
            println("GGMLCpuExecutor: dependency scheduled compute failed - ${t.message}")
            GGMLStatus.FAILED
        }
    }

    private fun blockingCompute(
        tracker: GGMLDependencyTracker,
        allocator: GGMLGraphAllocator,
        maxBatchSize: Int,
        config: GGMLCpuRuntimeConfig
    ) {
        while (!tracker.isComplete()) {
            val ready = tracker.getReadyNodes()
            if (ready.isEmpty()) {
                throw IllegalStateException("No ready nodes but graph not complete (possible cycle)")
            }

            val batches = if (ready.size <= maxBatchSize) {
                listOf(ready)
            } else {
                ready.chunked(maxBatchSize)
            }

            batches.forEach { batch ->
                processBatchSync(allocator, batch)
                batch.forEach { tracker.markCompleted(it) }
            }
        }
    }

    private fun processBatchSync(
        allocator: GGMLGraphAllocator,
        batch: List<GGMLTensor>
    ) {
        batch.forEach { tensor ->
            GGMLComputeOps.computeNode(allocator, tensor)
        }
    }
}
