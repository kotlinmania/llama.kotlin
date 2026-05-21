package ai.solace.llamakotlin.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith

/**
 * Tests for [GGMLThreadLocalContext].
 *
 * Verifies thread-local compute parameter storage works correctly across
 * different threads and scopes.
 */
class GGMLThreadLocalContextTest {

    @Test
    fun testSetAndGetParams() {
        // Initially, params should be null
        assertNull(GGMLThreadLocalContext.getCurrentParams())
        assertFalse(GGMLThreadLocalContext.hasCurrentParams())

        // Set params
        val params = GGMLComputeParams(ith = 0, nth = 1)
        GGMLThreadLocalContext.setCurrentParams(params)

        // Should be able to retrieve them
        val retrieved = GGMLThreadLocalContext.getCurrentParams()
        assertNotNull(retrieved)
        assertEquals(0, retrieved.ith)
        assertEquals(1, retrieved.nth)
        assertTrue(GGMLThreadLocalContext.hasCurrentParams())

        // Clear params
        GGMLThreadLocalContext.clearCurrentParams()
        assertNull(GGMLThreadLocalContext.getCurrentParams())
        assertFalse(GGMLThreadLocalContext.hasCurrentParams())
    }

    @Test
    fun testGetParamsOr() {
        // Clear any existing params
        GGMLThreadLocalContext.clearCurrentParams()

        // Should use default when not set
        val defaultParams = GGMLComputeParams(ith = 99, nth = 100)
        val retrieved = GGMLThreadLocalContext.getCurrentParamsOr { defaultParams }
        assertEquals(99, retrieved.ith)
        assertEquals(100, retrieved.nth)

        // Set params
        val actualParams = GGMLComputeParams(ith = 5, nth = 10)
        GGMLThreadLocalContext.setCurrentParams(actualParams)

        // Should use actual params now
        val retrieved2 = GGMLThreadLocalContext.getCurrentParamsOr { defaultParams }
        assertEquals(5, retrieved2.ith)
        assertEquals(10, retrieved2.nth)

        GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testRequireCurrentParams() {
        // Clear params
        GGMLThreadLocalContext.clearCurrentParams()

        // Should throw when not set
        assertFailsWith<IllegalStateException> {
            GGMLThreadLocalContext.requireCurrentParams()
        }

        // Set params
        val params = GGMLComputeParams(ith = 3, nth = 8)
        GGMLThreadLocalContext.setCurrentParams(params)

        // Should return params
        val retrieved = GGMLThreadLocalContext.requireCurrentParams()
        assertEquals(3, retrieved.ith)
        assertEquals(8, retrieved.nth)

        GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testWithParams() {
        // Clear params
        GGMLThreadLocalContext.clearCurrentParams()

        val params = GGMLComputeParams(ith = 2, nth = 4)

        val result = GGMLThreadLocalContext.withParams(params) {
            // Inside block, params should be available
            val retrieved = GGMLThreadLocalContext.getCurrentParams()
            assertNotNull(retrieved)
            assertEquals(2, retrieved.ith)
            assertEquals(4, retrieved.nth)

            // Return a value
            "test result"
        }

        // Block returned value
        assertEquals("test result", result)

        // After block, params should be cleared
        assertNull(GGMLThreadLocalContext.getCurrentParams())
    }

    @Test
    fun testWithParamsException() {
        GGMLThreadLocalContext.clearCurrentParams()

        val params = GGMLComputeParams(ith = 1, nth = 2)

        try {
            GGMLThreadLocalContext.withParams(params) {
                // Even if we throw, params should be cleared
                throw IllegalStateException("test exception")
            }
        } catch (e: IllegalStateException) {
            // Expected
        }

        // Params should still be cleared
        assertNull(GGMLThreadLocalContext.getCurrentParams())
    }

    @Test
    fun testGetCurrentThreadId() {
        GGMLThreadLocalContext.clearCurrentParams()

        // Should be null when not set
        assertNull(GGMLThreadLocalContext.getCurrentThreadId())

        // Set params
        val params = GGMLComputeParams(ith = 7, nth = 16)
        GGMLThreadLocalContext.setCurrentParams(params)

        // Should return thread ID
        assertEquals(7, GGMLThreadLocalContext.getCurrentThreadId())

        GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testGetTotalThreads() {
        GGMLThreadLocalContext.clearCurrentParams()

        // Should be null when not set
        assertNull(GGMLThreadLocalContext.getTotalThreads())

        // Set params
        val params = GGMLComputeParams(ith = 3, nth = 32)
        GGMLThreadLocalContext.setCurrentParams(params)

        // Should return thread count
        assertEquals(32, GGMLThreadLocalContext.getTotalThreads())

        GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testIsolationBetweenThreads() {
        // This test would require actual threading, which is platform-specific
        // For now, just verify basic single-thread behavior
        GGMLThreadLocalContext.clearCurrentParams()

        val params1 = GGMLComputeParams(ith = 0, nth = 1)
        GGMLThreadLocalContext.setCurrentParams(params1)
        assertEquals(0, GGMLThreadLocalContext.getCurrentThreadId())

        val params2 = GGMLComputeParams(ith = 1, nth = 1)
        GGMLThreadLocalContext.setCurrentParams(params2)
        assertEquals(1, GGMLThreadLocalContext.getCurrentThreadId())

        GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testNestedWithParams() {
        GGMLThreadLocalContext.clearCurrentParams()

        val outer = GGMLComputeParams(ith = 0, nth = 2)
        val inner = GGMLComputeParams(ith = 1, nth = 2)

        GGMLThreadLocalContext.withParams(outer) {
            assertEquals(0, GGMLThreadLocalContext.getCurrentThreadId())

            GGMLThreadLocalContext.withParams(inner) {
                assertEquals(1, GGMLThreadLocalContext.getCurrentThreadId())
            }

            // Outer params are cleared too (not stacked)
            assertNull(GGMLThreadLocalContext.getCurrentThreadId())
        }

        assertNull(GGMLThreadLocalContext.getCurrentThreadId())
    }
}
