package ai.solace.llamakotlin.backend.klangnative

import ai.solace.klangnative.mem.KAligned
import kotlinx.coroutines.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for arena-based concurrency patterns.
 *
 * Demonstrates safe multi-coroutine usage via:
 * 1. Per-coroutine arenas (isolated, no sharing)
 * 2. Shared HeapActor (serialized access)
 */
class ArenaConcurrencyTest {

    @Test
    fun testArenaBasicAllocation() {
        val arena = Arena.create(1024)
        try {
            val ptr1 = arena.alloc(256)
            val ptr2 = arena.alloc(128)

            // Pointers should be distinct and properly aligned
            assertTrue(ptr2 > ptr1, "Second allocation should be after first")
            assertTrue(ptr1 % 16 == 0, "Pointer should be 16-byte aligned")
            assertTrue(ptr2 % 16 == 0, "Pointer should be 16-byte aligned")

            // Verify usage tracking
            assertTrue(arena.used >= 256 + 128, "Used bytes should account for allocations")
            assertTrue(arena.available < 1024, "Available should decrease")
        } finally {
            KAligned.alignedFree(arena.basePtr)
        }
    }

    @Test
    fun testArenaReset() {
        val arena = Arena.create(1024)
        try {
            arena.alloc(256)
            arena.alloc(128)
            val used = arena.used

            assertTrue(used > 0, "Should have used bytes after allocation")

            arena.reset()

            assertEquals(0, arena.used, "Used should be 0 after reset")
            assertEquals(1024, arena.available, "Available should be full capacity after reset")

            // Can allocate again after reset
            val ptr = arena.alloc(512)
            assertTrue(ptr == arena.basePtr, "First allocation after reset should be at base")
        } finally {
            KAligned.alignedFree(arena.basePtr)
        }
    }

    @Test
    fun testPerCoroutineArenas() = runTest {
        // Create separate arenas for two coroutines
        val arena1 = Arena.create(1024)
        val arena2 = Arena.create(1024)

        try {
            val job1 = async {
                ArenaProvider.withArena(arena1) {
                    val ptr = ArenaProvider.current()?.alloc(256)
                    assertTrue(ptr != null, "Should have current arena")
                    ptr
                }
            }

            val job2 = async {
                ArenaProvider.withArena(arena2) {
                    val ptr = ArenaProvider.current()?.alloc(256)
                    assertTrue(ptr != null, "Should have current arena")
                    ptr
                }
            }

            val ptr1 = job1.await()
            val ptr2 = job2.await()

            // Pointers should be from different arenas
            assertTrue(ptr1 != null && ptr2 != null, "Both should have allocated")
            assertTrue(ptr1 != ptr2, "Pointers from different arenas should differ")

            // Each arena should have its own usage
            assertTrue(arena1.used >= 256, "Arena 1 should have used bytes")
            assertTrue(arena2.used >= 256, "Arena 2 should have used bytes")
        } finally {
            KAligned.alignedFree(arena1.basePtr)
            KAligned.alignedFree(arena2.basePtr)
        }
    }

    @Test
    fun testHeapActorSharedAccess() = runTest {
        val arena = Arena.create(4096)
        val actor = HeapActor.launch(arena, this)

        try {
            // Multiple coroutines write to shared arena via actor
            val jobs = (0 until 10).map { id ->
                async {
                    val ptr = actor.alloc(128)
                    val data = IntArray(32) { id * 100 + it }
                    actor.writeInts(ptr, data)
                    ptr to id
                }
            }

            val results = jobs.awaitAll()

            // Verify all allocations succeeded and are distinct
            val pointers = results.map { it.first }.toSet()
            assertEquals(10, pointers.size, "All pointers should be unique")

            // Verify we can read back data
            for ((ptr, id) in results) {
                val data = actor.readInts(ptr, 32)
                assertEquals(id * 100, data[0], "First element should match ID * 100")
                assertEquals(id * 100 + 31, data[31], "Last element should match ID * 100 + 31")
            }

            // Verify stats
            val (used, available) = actor.getStats()
            assertTrue(used >= 10 * 128, "Should have used at least 10 * 128 bytes")
            assertTrue(available > 0, "Should have available space")

            // Reset should work
            actor.reset()
            val (usedAfter, availableAfter) = actor.getStats()
            assertEquals(0, usedAfter, "Used should be 0 after reset")
            assertEquals(4096, availableAfter, "Available should be full after reset")
        } finally {
            KAligned.alignedFree(arena.basePtr)
        }
    }

    @Test
    fun testKLangNativeHeapWithArena() {
        val arena = Arena.create(2048)

        try {
            // Use arena with KLangNativeHeapTensorStorage
            val buffer = KLangNativeHeapTensorStorage.mallocFloat32(
                count = 64,
                alignment = 32,
                arena = arena
            )

            // Write data
            val data = FloatArray(64) { it.toFloat() * 0.5f }
            KLangNativeHeapTensorStorage.writeFloat32(buffer, data)

            // Read back
            val readBack = KLangNativeHeapTensorStorage.readFloat32(buffer, 64)

            // Verify
            for (i in 0 until 64) {
                assertEquals(data[i], readBack[i], "Values should match at index $i")
            }

            // Verify arena was used (not KAligned)
            assertTrue(arena.used >= 64 * 4, "Arena should have used bytes")

            // Note: Don't call buffer.free() when using arena!
            // Arena owns the memory and will be freed separately
        } finally {
            KAligned.alignedFree(arena.basePtr)
        }
    }

    @Test
    fun testConcurrentProducerConsumer() = runTest {
        val arena = Arena.create(8192)
        val actor = HeapActor.launch(arena, this)

        try {
            val channel = kotlinx.coroutines.channels.Channel<Pair<Int, Int>>(10)

            // Producer: allocate and write data
            val producer = launch {
                repeat(20) { i ->
                    val ptr = actor.alloc(64)
                    val data = IntArray(16) { i * 1000 + it }
                    actor.writeInts(ptr, data)
                    channel.send(ptr to 16)
                    delay(1) // Simulate work
                }
                channel.close()
            }

            // Consumer: read and verify data
            val consumer = async {
                var count = 0
                for ((ptr, size) in channel) {
                    val data = actor.readInts(ptr, size)
                    assertTrue(data[0] % 1000 == 0, "First element should be multiple of 1000")
                    count++
                }
                count
            }

            producer.join()
            val consumed = consumer.await()

            assertEquals(20, consumed, "Should have consumed all produced items")
        } finally {
            KAligned.alignedFree(arena.basePtr)
        }
    }

    @Test
    fun testArenaExhaustion() {
        val arena = Arena.create(128) // Small arena

        try {
            // Fill arena
            arena.alloc(64)
            arena.alloc(48) // Should work, total = 112 (aligned)

            // This should throw OutOfMemoryError
            try {
                arena.alloc(64) // Would exceed capacity
                throw AssertionError("Should have thrown OutOfMemoryError")
            } catch (e: OutOfMemoryError) {
                // Expected
                assertTrue(e.message?.contains("exhausted") == true, "Should mention exhaustion")
            }

            // After reset, can allocate again
            arena.reset()
            val ptr = arena.alloc(64)
            assertTrue(ptr == arena.basePtr, "Should allocate from start after reset")
        } finally {
            KAligned.alignedFree(arena.basePtr)
        }
    }
}
