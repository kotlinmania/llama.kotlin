package ai.solace.llamakotlin.core

import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GGMLBackendIntegrationTest {

    @BeforeTest
    fun setUp() {
        GGMLBackendRegistry.init()
    }

    @Test
    fun endToEndCpuExecution() {
        val backend = GGMLCpuBackend()
        val graph = createGraph(5, backend)
        val status = backend.graphCompute(graph)
        assertEquals(GGMLStatus.SUCCESS, status)
        backend.free()
    }

    @Test
    fun bufferDataIntegrity() {
        val backend = GGMLCpuBackend()
        val buffer = backend.allocBuffer(64u) as GGMLCpuBuffer

        val tensor = GGMLTensor(type = GGMLType.F32)
        tensor.ne[0] = 4L
        tensor.nb[0] = 4u
        tensor.dataOffset = 0u
        buffer.initTensor(tensor)

        val payload = ByteArray(16) { it.toByte() }
        buffer.setTensor(tensor, payload, 0u, payload.size.toULong())

        val roundTrip = ByteArray(payload.size)
        buffer.getTensor(tensor, roundTrip, 0u, roundTrip.size.toULong())
        assertContentEquals(payload, roundTrip)

        buffer.free()
        backend.free()
    }
}
