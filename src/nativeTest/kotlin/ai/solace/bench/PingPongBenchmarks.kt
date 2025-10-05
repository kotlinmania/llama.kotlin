@file:OptIn(
    kotlinx.cinterop.ExperimentalForeignApi::class,
    kotlin.concurrent.atomics.ExperimentalAtomicApi::class
)

package ai.solace.bench

import ai.solace.bench.kcoro.KcoroInterop
import ai.solace.kcoro.KC_RENDEZVOUS
import ai.solace.kcoro.kc_chan_recv
import ai.solace.kcoro.kc_chan_send
import kotlinx.cinterop.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlin.concurrent.atomics.AtomicInt
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.time.TimeSource

private enum class Role { PING, PONG }

private class KcoroPingPongSharedContext(
    val forward: CPointer<ai.solace.kcoro.kc_chan_t>?,
    val backward: CPointer<ai.solace.kcoro.kc_chan_t>?,
    val iterations: Int
) {
    val readyCount = AtomicInt(0)
    val startFlag = AtomicInt(0)
    val doneCount = AtomicInt(0)
}

private class KcoroPingPongWorkerArgs(
    val shared: KcoroPingPongSharedContext,
    val role: Role,
    val sendBuf: CPointer<IntVar>,
    val recvBuf: CPointer<IntVar>
)

private val kcoroPingWorker = staticCFunction { raw: COpaquePointer? ->
    val args = raw!!.asStableRef<KcoroPingPongWorkerArgs>().get()
    val shared = args.shared
    shared.readyCount.addAndGet(1)
    while (shared.startFlag.load() == 0) {
        KcoroInterop.yield()
    }
    val iterations = shared.iterations
    when (args.role) {
        Role.PING -> {
            val sendPtr = args.sendBuf.reinterpret<CPointed>()
            val recvPtr = args.recvBuf.reinterpret<CPointed>()
            for (i in 0 until iterations) {
                args.sendBuf.pointed.value = i
                if (kc_chan_send(shared.forward, sendPtr, -1L) != 0) break
                if (kc_chan_recv(shared.backward, recvPtr, -1L) != 0) break
            }
        }
        Role.PONG -> {
            val sendPtr = args.sendBuf.reinterpret<CPointed>()
            val recvPtr = args.recvBuf.reinterpret<CPointed>()
            for (i in 0 until iterations) {
                if (kc_chan_recv(shared.forward, recvPtr, -1L) != 0) break
                val value = args.recvBuf.pointed.value
                args.sendBuf.pointed.value = value
                if (kc_chan_send(shared.backward, sendPtr, -1L) != 0) break
            }
        }
    }
    shared.doneCount.addAndGet(1)
    Unit
}

object PingPongBenchmarks {
    fun runKcoroPingPong(iterations: Int, workers: Int = 2): Long {
        val sched = KcoroInterop.createScheduler(workers) ?: error("kc_sched_init failed")
        val elemSize = sizeOf<IntVar>().toULong()
        val forward = KcoroInterop.createChannel(KC_RENDEZVOUS, elemSize, 0uL)
        val backward = KcoroInterop.createChannel(KC_RENDEZVOUS, elemSize, 0uL)

        val shared = KcoroPingPongSharedContext(forward, backward, iterations)

        val pingSendVar = nativeHeap.alloc<IntVar>()
        val pingRecvVar = nativeHeap.alloc<IntVar>()
        val pongSendVar = nativeHeap.alloc<IntVar>()
        val pongRecvVar = nativeHeap.alloc<IntVar>()

        val pingArgs = KcoroPingPongWorkerArgs(shared, Role.PING, pingSendVar.ptr, pingRecvVar.ptr)
        val pongArgs = KcoroPingPongWorkerArgs(shared, Role.PONG, pongSendVar.ptr, pongRecvVar.ptr)

        val pingRef = StableRef.create(pingArgs)
        val pongRef = StableRef.create(pongArgs)

        val pingSpawn = KcoroInterop.spawnCoroutine(sched, kcoroPingWorker, pingRef.asCPointer())
        val pongSpawn = KcoroInterop.spawnCoroutine(sched, kcoroPingWorker, pongRef.asCPointer())
        require(pingSpawn.first == 0 && pongSpawn.first == 0) { "kc_spawn_co failed" }

        while (shared.readyCount.load() < 2) {
            KcoroInterop.yield()
        }

        val mark = TimeSource.Monotonic.markNow()
        shared.startFlag.store(1)

        KcoroInterop.drainScheduler(sched, -1L)
        val total = mark.elapsedNow().inWholeNanoseconds

        KcoroInterop.shutdownScheduler(sched)

        KcoroInterop.destroyChannel(forward)
        KcoroInterop.destroyChannel(backward)

        pingRef.dispose()
        pongRef.dispose()
        nativeHeap.free(pingSendVar)
        nativeHeap.free(pingRecvVar)
        nativeHeap.free(pongSendVar)
        nativeHeap.free(pongRecvVar)

        return total
    }

    fun runKotlinPingPong(iterations: Int): Long = runBlocking {
        val forward = Channel<Int>(Channel.RENDEZVOUS)
        val backward = Channel<Int>(Channel.RENDEZVOUS)
        val worker = launch {
            repeat(iterations) {
                val value = forward.receive()
                backward.send(value)
            }
        }
        val mark = TimeSource.Monotonic.markNow()
        repeat(iterations) { i ->
            forward.send(i)
            backward.receive()
        }
        val total = mark.elapsedNow().inWholeNanoseconds
        worker.join()
        forward.close()
        backward.close()
        total
    }
}

class PingPongBenchmarkTest {
    @Test
    fun compareKcoroAndKotlin() {
        val iterations = 200_000
        val kcoroTime = PingPongBenchmarks.runKcoroPingPong(iterations)
        val kotlinTime = PingPongBenchmarks.runKotlinPingPong(iterations)
        println("kcoro ping-pong: total=${kcoroTime} ns (~${kcoroTime.toDouble() / iterations} ns/op)")
        println("kotlin coroutines ping-pong: total=${kotlinTime} ns (~${kotlinTime.toDouble() / iterations} ns/op)")
        assertTrue(kcoroTime > 0 && kotlinTime > 0)
    }
}
