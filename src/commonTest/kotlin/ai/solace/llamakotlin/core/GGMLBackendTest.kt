package ai.solace.llamakotlin.core

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
        GGMLBackendRegistry.init()
    }

    @Test
    fun cpuBackendIsRegistered() {
        val index = GGMLBackendRegistry.findByName("CPU")
        assertNotNull(index, "CPU backend should be registered")
        assertEquals("CPU", GGMLBackendRegistry.getName(index!!))
        assertEquals(1, GGMLBackendRegistry.getCount(), "Only CPU backend expected")
    }

    @Test
    fun cpuBackendCreationAndBufferType() {
        val cpuIndex = GGMLBackendRegistry.findByName("CPU")
        assertNotNull(cpuIndex)

        val backend = GGMLBackendRegistry.initBackend(cpuIndex!!)
        assertTrue(backend is GGMLCpuBackend)
        assertEquals("CPU", backend.getName())
        assertTrue(backend.getGuid().isNotEmpty())

        val bufferType = backend.getDefaultBufferType()
        assertTrue(bufferType is GGMLCpuBufferType)
        assertTrue(bufferType.isHost())
        assertEquals("CPU", bufferType.getName())
        assertTrue(bufferType.getAlignment() > 0u)
    }

    @Test
    fun cpuBufferLifecycle() {
        val backend = GGMLCpuBackend()
        val bufferType = backend.getDefaultBufferType()

        val buffer = bufferType.allocBuffer(1024u)
        assertNotNull(buffer)
        assertTrue(buffer is GGMLCpuBuffer)
        assertEquals(1024u, buffer.getSize())
        assertTrue(buffer.getBase() is ByteArray)

        val zero = bufferType.allocBuffer(0u)
        assertNull(zero, "Zero-sized allocation should return null")

        buffer.free()
        backend.free()
    }

    @Test
    fun cpuBufferReadWriteRoundTrip() {
        val buffer = GGMLCpuBufferType().allocBuffer(1024u) as GGMLCpuBuffer

        val tensor = GGMLTensor(type = GGMLType.F32)
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
        val backend = GGMLCpuBackend()

        val supportedOps = listOf(GGMLOp.ADD, GGMLOp.MUL, GGMLOp.MUL_MAT)
        supportedOps.forEach { op ->
            val tensor = GGMLTensor(op = op)
            assertTrue(backend.supportsOp(tensor), "CPU should support $op")
        }

        val unsupported = GGMLTensor(op = GGMLOp.NONE)
        assertFalse(backend.supportsOp(unsupported))

        val bufferType = backend.getDefaultBufferType()
        assertTrue(backend.supportsBufferType(bufferType))

        backend.free()
    }

    @Test
    fun graphAllocatorBindsCpuBackendBuffers() {
        val backend = GGMLCpuBackend()
        val allocator = GGMLGraphAllocator(backend)

        assertEquals(backend, allocator.backend)
        assertTrue(allocator.backendBuffers.first() is GGMLCpuBuffer)

        val graph = GGMLCGraph(allocator = allocator)
        val status = backend.graphCompute(graph)
        assertEquals(GGMLStatus.SUCCESS, status)

        backend.free()
    }
}
