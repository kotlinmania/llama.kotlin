@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package ai.solace.bench.kcoro

import ai.solace.kcoro.*
import kotlinx.cinterop.*

object KcoroInterop {
    fun createScheduler(workers: Int = 0): CPointer<kc_sched_t>? = memScoped {
        val opts = alloc<kc_sched_opts_t>().apply {
            this.workers = workers
            queue_capacity = 0
            inject_q_cap = 0
        }
        kc_sched_init(opts.ptr)
    }

    fun shutdownScheduler(sched: CPointer<kc_sched_t>?) {
        if (sched != null) {
            kc_sched_shutdown(sched)
        }
    }

    fun drainScheduler(sched: CPointer<kc_sched_t>?, timeoutMs: Long = -1L): Int {
        if (sched == null) return -95 // -ENOTSUP fallback
        return kc_sched_drain(sched, timeoutMs)
    }

    fun createChannel(kind: Int, elemSize: ULong, capacity: ULong = 0u): CPointer<kc_chan_t> = memScoped {
        val out = alloc<CPointerVar<kc_chan_t>>()
        val rc = kc_chan_make(out.ptr, kind, elemSize, capacity)
        require(rc == 0) { "kc_chan_make failed rc=$rc" }
        out.value ?: error("kc_chan_make returned null channel")
    }

    fun destroyChannel(ch: CPointer<kc_chan_t>?) {
        if (ch != null) kc_chan_destroy(ch)
    }

    fun spawnCoroutine(
        sched: CPointer<kc_sched_t>?,
        fn: CPointer<CFunction<(COpaquePointer?) -> Unit>>,
        arg: COpaquePointer?,
        stackBytes: ULong = (64uL * 1024uL)
    ): Pair<Int, CPointer<kcoro_t>?> = memScoped {
        val out = alloc<CPointerVar<kcoro_t>>()
        val rc = kc_spawn_co(sched, fn, arg, stackBytes, out.ptr)
        rc to out.value
    }

    fun spawnTask(
        sched: CPointer<kc_sched_t>?,
        fn: CPointer<CFunction<(COpaquePointer?) -> Unit>>,
        arg: COpaquePointer?
    ): Int = kc_spawn(sched, fn, arg)

    fun yield() = kc_yield()
    fun sleepMs(ms: Int) = kc_sleep_ms(ms)
}
