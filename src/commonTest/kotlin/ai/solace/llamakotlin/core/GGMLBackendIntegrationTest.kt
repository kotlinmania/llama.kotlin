package ai.solace.llamakotlin.core

import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GGMLBackendIntegrationTest {
    private lateinit var manager: GGMLBackendManager

    @BeforeTest
    fun setUp() {
        GGMLBackendRegistry.init()
        manager = GGMLBackendManager()
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
    fun backendSelectionIsCpuOnly() {
        val tensor = GGMLTensor(op = GGMLOp.MUL_MAT).apply {
            ne[0] = 1024
            ne[1] = 1024
        }
        val backend = manager.selectBackend(tensor)
        assertNotNull(backend)
        assertEquals("CPU", backend.getName())
    }

    @Test
    fun bufferDataIntegrityThroughManager() {
        val backend = manager.getPrimaryBackend() as GGMLCpuBackend
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

    @Test
    fun globalBackendManagerUsesCpu() {
        val graph = createGraphWithGlobalBackend(10)
        val status = computeGraphHybrid(graph)
        assertEquals(GGMLStatus.SUCCESS, status)

        val info = globalBackendManager.getBackendInfo()
        assertTrue((info["availableBackends"] as List<*>).contains("CPU"))
        assertEquals("CPU", info["primaryBackend"])
        assertEquals("CPU", info["fallbackBackend"])
    }
}
