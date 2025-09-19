package ai.solace.llamakotlin.core

import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.newFixedThreadPoolContext
import kotlinx.coroutines.runBlocking

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
        val parallelism = maxBatchSize.coerceAtLeast(1)
        val dispatcher = selectDispatcher(parallelism, config.useDedicatedDispatcher)
        runBlocking(dispatcher) {
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
                    val completed = processBatch(allocator, batch)
                    completed.forEach { tracker.markCompleted(it) }
                }
            }
        }
    }

    private suspend fun processBatch(
        allocator: GGMLGraphAllocator,
        batch: List<GGMLTensor>
    ): List<GGMLTensor> = coroutineScope {
        val jobs = batch.map { tensor ->
            async {
                GGMLComputeOps.computeNode(allocator, tensor)
                tensor
            }
        }
        jobs.awaitAll()
    }

    private fun selectDispatcher(parallelism: Int, dedicated: Boolean): kotlinx.coroutines.CoroutineDispatcher {
        return if (dedicated) {
            dedicatedDispatcher(parallelism)
        } else {
            Dispatchers.Default.limitedParallelism(parallelism)
        }
    }

    @OptIn(DelicateCoroutinesApi::class)
    private fun dedicatedDispatcher(parallelism: Int): kotlinx.coroutines.CoroutineDispatcher {
        val poolSize = parallelism.coerceIn(1, MAX_DEDICATED_THREADS)
        return dispatcherCache.getOrPut(poolSize) {
            newFixedThreadPoolContext(poolSize, "ggml-cpu-$poolSize")
        }
    }

    companion object {
        private const val MAX_DEDICATED_THREADS = 32
        private val dispatcherCache = mutableMapOf<Int, kotlinx.coroutines.CoroutineDispatcher>()
    }
}
