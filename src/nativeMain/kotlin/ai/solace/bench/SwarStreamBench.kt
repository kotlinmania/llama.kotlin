package ai.solace.bench

import ai.solace.klang.bitwise.SwAR
import kotlin.math.min
import kotlin.time.TimeSource
import kotlinx.coroutines.Channel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

internal fun runSwarAvgStreamBench(workerCount: Int = Runtime.getRuntime().availableProcessors()) {
    println("variant,size,iters,ns,GBps,checksum,workers")
    val sizes = listOf(8, 64, 4096, 262144)
    fun itersFor(size: Int) = when {
        size <= 64 -> 200_000
        size <= 4096 -> 50_000
        else -> 2_000
    }
    for (sz in sizes) {
        val iters = itersFor(sz)
        val (a0, b0) = genPacks(sz)
        val out = IntArray(sz)
        val dispatcher = Dispatchers.Default
        val t0 = TimeSource.Monotonic.markNow()
        runBlocking {
            val tileChannel = Channel<Int>(capacity = min(sz, 1024))
            val workerChannels = List(workerCount) { Channel<Int>(capacity = min(sz, 1024)) }

            val workers: List<Job> = workerChannels.map { channel ->
                launch(dispatcher) {
                    for (idx in channel) {
                        out[idx] = SwAR.avgU8TruncArith(a0[idx], b0[idx])
                    }
                }
            }

            val router = launch(dispatcher) {
                var nextWorker = 0
                for (idx in tileChannel) {
                    workerChannels[nextWorker].send(idx)
                    nextWorker = (nextWorker + 1) % workerCount
                }
                workerChannels.forEach { it.close() }
            }

            val producer = launch(dispatcher) {
                repeat(iters) {
                    var idx = 0
                    while (idx < sz) {
                        tileChannel.send(idx)
                        idx++
                    }
                }
                tileChannel.close()
            }

            producer.join()
            router.join()
            workers.joinAll()
        }
        val ns = t0.elapsedNow().inWholeNanoseconds
        val gbps = (sz * q8LogicalBytesPerBlock * iters / 1e9) / (ns.toDouble() / 1e9)
        val checksum = checksumIntArray(out)
        val gbpsFmt = ((gbps * 1000.0).toLong() / 1000.0)
        println("swar-u8-trunc-stream,$sz,$iters,$ns,$gbpsFmt,$checksum,$workerCount")
    }
}
