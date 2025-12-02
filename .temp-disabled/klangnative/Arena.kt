package ai.solace.llamakotlin.backend.klangnative

import kotlinx.atomicfu.atomic
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import ai.solace.klangnative.mem.GlobalHeap
import ai.solace.klangnative.mem.KAligned
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.math.max

/**
 * Per-coroutine bump allocator arena for KLangNative heap.
 *
 * Provides fast, lock-free allocation within a fixed backing buffer using
 * monotonic offset bumping. Modeled after kcoro_arena from llama.kotlin/external/kcoro_arena.
 *
 * Key properties:
 * - 16-byte aligned allocations by default
 * - Atomic offset for lock-free bumping
 * - No individual frees (bump allocator semantics)
 * - Fast reset() to reclaim entire arena
 * - Safe for single coroutine/thread exclusive use
 *
 * For multi-coroutine sharing, wrap with [HeapActor] to serialize access.
 *
 * @param basePtr Heap pointer to backing buffer start
 * @param capacityBytes Total size of backing buffer
 * @param alignment Alignment requirement (default 16 for SIMD)
 */
class Arena(
    val basePtr: Int,
    val capacityBytes: Int,
    private val alignment: Int = 16
) {
    private val offset = atomic(0)

    /**
     * Allocate [bytes] from this arena, returning heap pointer.
     * Throws [IllegalStateException] if arena is exhausted.
     */
    fun alloc(bytes: Int): Int {
        require(bytes >= 0) { "bytes must be non-negative" }
        val aligned = align(max(1, bytes))
        while (true) {
            val cur = offset.value
            val next = cur + aligned
            if (next > capacityBytes) {
                throw IllegalStateException("Arena exhausted: $cur + $aligned > $capacityBytes")
            }
            if (offset.compareAndSet(cur, next)) {
                return basePtr + cur
            }
        }
    }

    /**
     * Reset arena to empty state. All previous allocations are invalidated.
     * This is the only "free" operation - no individual frees in bump allocator.
     */
    fun reset() {
        offset.value = 0
    }

    /**
     * Current used bytes (for debugging/monitoring).
     */
    val used: Int get() = offset.value

    /**
     * Available bytes remaining.
     */
    val available: Int get() = capacityBytes - offset.value

    private fun align(n: Int): Int {
        // Align to next multiple of alignment
        return (n + (alignment - 1)) and (alignment - 1).inv()
    }

    companion object {
        /**
         * Create arena backed by KAligned allocation.
         * Caller must free backing buffer when done.
         */
        fun create(capacityBytes: Int, alignment: Int = 16): Arena {
            val ptr = KAligned.alignedCalloc(alignment, capacityBytes)
            return Arena(ptr, capacityBytes, alignment)
        }
    }
}

/**
 * Coroutine context element carrying the current arena.
 *
 * Use with `withContext(ArenaContext(arena))` or `ArenaProvider.withArena()`.
 */
class ArenaContext(val arena: Arena) : CoroutineContext.Element {
    override val key: CoroutineContext.Key<ArenaContext> = Key

    companion object Key : CoroutineContext.Key<ArenaContext>
}

/**
 * Manages per-coroutine arena access via coroutine context.
 *
 * Usage patterns:
 * ```kotlin
 * // Per-coroutine arena
 * val arena = Arena.create(1024 * 1024) // 1MB
 * ArenaProvider.withArena(arena) {
 *     val ptr = ArenaProvider.current()?.alloc(256)
 *     // ... use ptr
 * }
 *
 * // Or explicit context
 * withContext(ArenaContext(arena)) {
 *     val ptr = ArenaProvider.current(coroutineContext)?.alloc(256)
 * }
 * ```
 */
object ArenaProvider {
    /**
     * Get the current arena from coroutine context.
     * Returns null if no arena is bound.
     */
    fun current(context: CoroutineContext = EmptyCoroutineContext): Arena? {
        return context[ArenaContext]?.arena
    }

    /**
     * Run [block] with [arena] bound to coroutine context.
     * Arena is automatically unbound after block completes.
     */
    suspend fun <T> withArena(arena: Arena, block: suspend () -> T): T {
        val ctx = ArenaContext(arena)
        return kotlinx.coroutines.withContext(ctx) {
            block()
        }
    }
}

/**
 * Actor-based serialization for shared arena access across coroutines.
 *
 * When multiple coroutines need to share the same arena, HeapActor ensures
 * thread-safe access by serializing all operations through a single-threaded
 * actor with a channel-based mailbox.
 *
 * Usage:
 * ```kotlin
 * val arena = Arena.create(1024 * 1024)
 * val actor = HeapActor.launch(arena)
 *
 * // From multiple coroutines
 * launch {
 *     val ptr = actor.alloc(256)
 *     actor.writeInts(ptr, intArrayOf(1, 2, 3))
 * }
 * launch {
 *     val data = actor.readInts(ptr, 3)
 * }
 * ```
 *
 * @param arena The arena to manage (exclusive ownership)
 * @param scope CoroutineScope for actor lifetime
 */
class HeapActor private constructor(
    private val arena: Arena,
    private val scope: CoroutineScope
) {
    private val mailbox = Channel<Msg>(Channel.UNLIMITED)

    sealed interface Msg
    private data class Alloc(val bytes: Int, val reply: Channel<Int>) : Msg
    private data class Write(val addr: Int, val data: IntArray, val reply: Channel<Unit>) : Msg
    private data class Read(val addr: Int, val count: Int, val reply: Channel<IntArray>) : Msg
    private data class Reset(val reply: Channel<Unit>) : Msg
    private data class GetStats(val reply: Channel<Pair<Int, Int>>) : Msg

    init {
        scope.launch {
            for (msg in mailbox) {
                when (msg) {
                    is Alloc -> {
                        try {
                            val ptr = arena.alloc(msg.bytes)
                            msg.reply.send(ptr)
                        } catch (e: IllegalStateException) {
                            msg.reply.close(e)
                        } finally {
                            msg.reply.close()
                        }
                    }
                    is Write -> {
                        try {
                            var p = msg.addr
                            msg.data.forEach { word ->
                                GlobalHeap.sw(p, word)
                                p += 4
                            }
                            msg.reply.send(Unit)
                        } finally {
                            msg.reply.close()
                        }
                    }
                    is Read -> {
                        try {
                            val out = IntArray(msg.count)
                            var p = msg.addr
                            for (i in 0 until msg.count) {
                                out[i] = GlobalHeap.lw(p)
                                p += 4
                            }
                            msg.reply.send(out)
                        } finally {
                            msg.reply.close()
                        }
                    }
                    is Reset -> {
                        try {
                            arena.reset()
                            msg.reply.send(Unit)
                        } finally {
                            msg.reply.close()
                        }
                    }
                    is GetStats -> {
                        try {
                            msg.reply.send(arena.used to arena.available)
                        } finally {
                            msg.reply.close()
                        }
                    }
                }
            }
        }
    }

    /**
     * Allocate [bytes] from the managed arena.
     * Suspends until allocation completes.
     */
    suspend fun alloc(bytes: Int): Int {
        val reply = Channel<Int>(1)
        mailbox.send(Alloc(bytes, reply))
        return reply.receive()
    }

    /**
     * Write int array to heap address.
     * Each int is written as a 32-bit word via GlobalHeap.sw().
     */
    suspend fun writeInts(addr: Int, data: IntArray) {
        val reply = Channel<Unit>(1)
        mailbox.send(Write(addr, data, reply))
        reply.receive()
    }

    /**
     * Read [count] ints from heap address.
     * Each int is read as a 32-bit word via GlobalHeap.lw().
     */
    suspend fun readInts(addr: Int, count: Int): IntArray {
        val reply = Channel<IntArray>(1)
        mailbox.send(Read(addr, count, reply))
        return reply.receive()
    }

    /**
     * Reset arena to empty state.
     * All previous allocations are invalidated.
     */
    suspend fun reset() {
        val reply = Channel<Unit>(1)
        mailbox.send(Reset(reply))
        reply.receive()
    }

    /**
     * Get arena usage statistics: (used, available) bytes.
     */
    suspend fun getStats(): Pair<Int, Int> {
        val reply = Channel<Pair<Int, Int>>(1)
        mailbox.send(GetStats(reply))
        return reply.receive()
    }

    companion object {
        /**
         * Launch actor managing [arena] in [scope].
         * Actor lifetime is tied to scope - cancel scope to stop actor.
         */
        fun launch(
            arena: Arena,
            scope: CoroutineScope = CoroutineScope(SupervisorJob())
        ): HeapActor = HeapActor(arena, scope)
    }
}
