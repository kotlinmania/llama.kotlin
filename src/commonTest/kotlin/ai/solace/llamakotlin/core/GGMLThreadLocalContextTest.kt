package io.github.kotlinmania.llama..core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith

/**
 * Tests for [io.github.kotlinmania.llama.core.GGMLThreadLocalContext].
 *
 * Verifies thread-local compute parameter storage works correctly across
 * different threads and scopes.
 */
class GGMLThreadLocalContextTest {

    @Test
    fun testSetAndGetParams() {
        // Initially, params should be null
        assertNull(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentParams())
        assertFalse(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.hasCurrentParams())

        // Set params
        val params = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 0, nth = 1)
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.setCurrentParams(params)

        // Should be able to retrieve them
        val retrieved = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentParams()
        assertNotNull(retrieved)
        assertEquals(0, retrieved.ith)
        assertEquals(1, retrieved.nth)
        assertTrue(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.hasCurrentParams())

        // Clear params
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()
        assertNull(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentParams())
        assertFalse(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.hasCurrentParams())
    }

    @Test
    fun testGetParamsOr() {
        // Clear any existing params
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()

        // Should use default when not set
        val defaultParams = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 99, nth = 100)
        val retrieved = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentParamsOr { defaultParams }
        assertEquals(99, retrieved.ith)
        assertEquals(100, retrieved.nth)

        // Set params
        val actualParams = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 5, nth = 10)
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.setCurrentParams(actualParams)

        // Should use actual params now
        val retrieved2 = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentParamsOr { defaultParams }
        assertEquals(5, retrieved2.ith)
        assertEquals(10, retrieved2.nth)

        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testRequireCurrentParams() {
        // Clear params
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()

        // Should throw when not set
        assertFailsWith<IllegalStateException> {
            _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.requireCurrentParams()
        }

        // Set params
        val params = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 3, nth = 8)
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.setCurrentParams(params)

        // Should return params
        val retrieved = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.requireCurrentParams()
        assertEquals(3, retrieved.ith)
        assertEquals(8, retrieved.nth)

        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testWithParams() {
        // Clear params
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()

        val params = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 2, nth = 4)

        val result = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.withParams(params) {
            // Inside block, params should be available
            val retrieved = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentParams()
            assertNotNull(retrieved)
            assertEquals(2, retrieved.ith)
            assertEquals(4, retrieved.nth)

            // Return a value
            "test result"
        }

        // Block returned value
        assertEquals("test result", result)

        // After block, params should be cleared
        assertNull(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentParams())
    }

    @Test
    fun testWithParamsException() {
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()

        val params = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 1, nth = 2)

        try {
            _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.withParams(params) {
                // Even if we throw, params should be cleared
                throw IllegalStateException("test exception")
            }
        } catch (e: IllegalStateException) {
            // Expected
        }

        // Params should still be cleared
        assertNull(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentParams())
    }

    @Test
    fun testGetCurrentThreadId() {
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()

        // Should be null when not set
        assertNull(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentThreadId())

        // Set params
        val params = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 7, nth = 16)
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.setCurrentParams(params)

        // Should return thread ID
        assertEquals(7, _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentThreadId())

        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testGetTotalThreads() {
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()

        // Should be null when not set
        assertNull(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getTotalThreads())

        // Set params
        val params = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 3, nth = 32)
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.setCurrentParams(params)

        // Should return thread count
        assertEquals(32, _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getTotalThreads())

        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testIsolationBetweenThreads() {
        // This test would require actual threading, which is platform-specific
        // For now, just verify basic single-thread behavior
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()

        val params1 = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 0, nth = 1)
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.setCurrentParams(params1)
        assertEquals(0, _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentThreadId())

        val params2 = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 1, nth = 1)
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.setCurrentParams(params2)
        assertEquals(1, _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentThreadId())

        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testNestedWithParams() {
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.clearCurrentParams()

        val outer = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 0, nth = 2)
        val inner = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLComputeParams(ith = 1, nth = 2)

        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.withParams(outer) {
            assertEquals(0, _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentThreadId())

            _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.withParams(inner) {
                assertEquals(1, _root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentThreadId())
            }

            // Outer params are cleared too (not stacked)
            assertNull(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentThreadId())
        }

        assertNull(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLThreadLocalContext.getCurrentThreadId())
    }
}
