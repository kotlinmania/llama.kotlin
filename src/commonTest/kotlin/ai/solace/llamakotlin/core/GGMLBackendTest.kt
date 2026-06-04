package io.github.kotlinmania.llama..core

import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.assertContentEquals
import kotlin.test.assertFalse

class GGMLBackendTest {
    @BeforeTest
    fun ensureRegistryInitialized() {
        _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendRegistry.init()
    }

    @Test
    fun cpuBackendIsRegistered() {
        val index = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendRegistry.findByName("CPU")
        assertNotNull(index, "CPU backend should be registered")
        assertEquals("CPU", _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendRegistry.getName(index!!))
        assertEquals(1, _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendRegistry.getCount(), "Only CPU backend expected")
    }

    @Test
    fun cpuBackendCreationAndBufferType() {
        val cpuIndex = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendRegistry.findByName("CPU")
        assertNotNull(cpuIndex)

        val backend = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLBackendRegistry.initBackend(cpuIndex!!)
        assertTrue(backend is io.github.kotlinmania.llama.core.GGMLCpuBackend)
        assertEquals("CPU", backend.getName())
        assertTrue(backend.getGuid().isNotEmpty())

        val bufferType = backend.getDefaultBufferType()
        assertTrue(bufferType is io.github.kotlinmania.llama.core.GGMLCpuBufferType)
        assertTrue(bufferType.isHost())
        assertEquals("CPU", bufferType.getName())
        assertTrue(bufferType.getAlignment() > 0u)
    }

    @Test
    fun cpuBufferLifecycle() {
        val backend = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLCpuBackend()
        val bufferType = backend.getDefaultBufferType()

        val buffer = bufferType.allocBuffer(1024u)
        assertNotNull(buffer)
        assertTrue(buffer is io.github.kotlinmania.llama.core.GGMLCpuBuffer)
        assertEquals(1024u, buffer.getSize())
        assertTrue(buffer.getBase() is ByteArray)

        val zero = bufferType.allocBuffer(0u)
        assertNull(zero, "Zero-sized allocation should return null")

        buffer.free()
        backend.free()
    }

    @Test
    fun cpuBufferReadWriteRoundTrip() {
        val buffer = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLCpuBufferType()
            .allocBuffer(1024u) as io.github.kotlinmania.llama.core.GGMLCpuBuffer

        val tensor =
            _root_ide_package_.io.github.kotlinmania.llama.core.GGMLTensor(type = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLType.F32)
        tensor.ne[0] = 10L
        tensor.nb[0] = 4u
        tensor.dataOffset = 0u
        buffer.initTensor(tensor)

        val payload = ByteArray(40) { (it * 3).toByte() }
        buffer.setTensor(tensor, payload, 0u, payload.size.toULong())

        val roundTrip = ByteArray(payload.size)
        buffer.getTensor(tensor, roundTrip, 0u, roundTrip.size.toULong())
        assertContentEquals(payload, roundTrip)

        buffer.clear(0u)
        val cleared = buffer.getBase() as ByteArray
        assertTrue(cleared.all { it == 0.toByte() })

        buffer.free()
    }

    @Test
    fun cpuBackendSupportsExpectedOps() {
        val backend = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLCpuBackend()

        val supportedOps = listOf(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLOp.ADD, _root_ide_package_.io.github.kotlinmania.llama.core.GGMLOp.MUL, _root_ide_package_.io.github.kotlinmania.llama.core.GGMLOp.MUL_MAT)
        supportedOps.forEach { op ->
            val tensor = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLTensor(op = op)
            assertTrue(backend.supportsOp(tensor), "CPU should support $op")
        }

        val unsupported =
            _root_ide_package_.io.github.kotlinmania.llama.core.GGMLTensor(op = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLOp.NONE)
        assertFalse(backend.supportsOp(unsupported))

        val bufferType = backend.getDefaultBufferType()
        assertTrue(backend.supportsBufferType(bufferType))

        backend.free()
    }

    @Test
    fun graphAllocatorBindsCpuBackendBuffers() {
        val backend = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLCpuBackend()
        val allocator = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLGraphAllocator(backend)

        assertEquals(backend, allocator.backend)
        assertTrue(allocator.backendBuffers.first() is io.github.kotlinmania.llama.core.GGMLCpuBuffer)

        val graph = _root_ide_package_.io.github.kotlinmania.llama.core.GGMLCGraph(allocator = allocator)
        val status = backend.graphCompute(graph)
        assertEquals(_root_ide_package_.io.github.kotlinmania.llama.core.GGMLStatus.SUCCESS, status)

        backend.free()
    }
}
