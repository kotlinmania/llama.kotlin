@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package ai.solace.llamakotlin.core

import kotlin.math.cos
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.cstr
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.usePinned
import platform.posix.fclose
import platform.posix.fflush
import platform.posix.fopen
import platform.posix.fputs
import platform.posix.fwrite
import platform.posix.system

class Q2KSnapshotTest {
    private fun hexToByteArray(hexTable: String): ByteArray {
        return hexTable.trim().split(Regex("\\s+"))
            .filter { it.isNotEmpty() }
            .map { it.toInt(16).toByte() }
            .toByteArray()
    }

    private val referenceBlock: ByteArray = hexToByteArray(
        """
            0b 09 06 44 a3 e2 f0 e2 b3 54 05 09 0f 0b 09 06 ff ff bf 7f 6f 2b 2b 2b 1b 1b 1b 1b 07 47 47 87 3f 3f 2f 2f 2b 2b 1b 1b 57 57 47 46 46 82 82 82 e4 e4 e4 e4 e9 e9 e9 e9 ea ea ee ee af af af af f8 f8 f8 f9 f9 f9 f9 bd be be be be bf bf 7f 7f 43 2c 0b 30
        """
    )

    @Test
    fun q2KQuantizationMatchesReferenceBlock() {
        val graphAllocator = GGMLGraphAllocator()
        val buffer = ByteArray(4 * 1024 * 1024)
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())
        graphAllocator.buffers[0] = buffer
        graphAllocator.tensorAllocators[0].reset(buffer.size.toULong())
        GGMLTestAllocatorState.bind(graphAllocator, buffer.size.toULong())

        val values = FloatArray(QK_K) { idx ->
            0.1f + 2.0f * cos(idx.toFloat() * 0.03f)
        }

        val tensor = GGMLTensor(GGMLType.F32)
        tensor.ne = LongArray(GGML_MAX_DIMS) { 1L }
        tensor.ne[0] = QK_K.toLong()
        tensor.nb = ULongArray(GGML_MAX_DIMS) { 0uL }
        tensor.nb[0] = GGMLType.F32.byteSize
        val offset = graphAllocator.allocateTensorData(QK_K * Float.SIZE_BYTES)
        tensor.bufferId = 0
        tensor.dataOffset = offset
        for (i in values.indices) {
            tensor.setFloat(graphAllocator, values[i], i)
        }

        Q2KDiagnosticsRecorder.reset()
        Q2KDiagnosticsRecorder.enabled = true
        println("Diagnostics enabled? ${Q2KDiagnosticsRecorder.enabled}")
        val quantized = quantizeTensor(graphAllocator, tensor, GGMLType.Q2_K)
        Q2KDiagnosticsRecorder.enabled = false
        assertEquals(GGMLType.Q2_K, quantized.type)

        val data = quantized.data as? ByteArray ?: error("Expected ByteArray for quantized tensor data")
        system("mkdir -p build/q2k-diagnostics")

        writeBinary("build/q2k-diagnostics/q2k-kotlin.bin", data)
        dumpPerSubBlock("build/q2k-diagnostics/q2k-kotlin.txt", data)
        dumpRecordedFloats("build/q2k-diagnostics/q2k-kotlin-floats.txt")
        dumpTrace("build/q2k-diagnostics/q2k-kotlin-trace.txt")

        val actualHex = data.joinToString(separator = " ") { (it.toInt() and 0xFF).toString(16).padStart(2, '0') }
        val expectedHex = referenceBlock.joinToString(separator = " ") { (it.toInt() and 0xFF).toString(16).padStart(2, '0') }
        println("Actual Q2_K block: $actualHex")
        println("Expected Q2_K block: $expectedHex")
        val mismatched = referenceBlock.indices.filter { referenceBlock[it] != data[it] }
        println("Mismatched indices: $mismatched")

        assertEquals(referenceBlock.size, data.size)
        assertContentEquals(referenceBlock, data)
    }

    private fun writeBinary(path: String, bytes: ByteArray) {
        val file = fopen(path, "wb") ?: error("Failed to open $path")
        bytes.usePinned {
            val written = fwrite(it.addressOf(0), 1uL, bytes.size.toULong(), file)
            if (written.toInt() != bytes.size) {
                fclose(file)
                error("Failed to write full block to $path")
            }
        }
        fflush(file)
        fclose(file)
    }

    private fun dumpPerSubBlock(path: String, bytes: ByteArray) {
        fun Byte.toUHex(): String = (toInt() and 0xFF).toString(16).padStart(2, '0')
        val file = fopen(path, "w") ?: error("Failed to open $path")
        val scaleCount = QK_K / 16 // 16
        val quantsCount = QK_K / 4  // 64
        val dIndex = scaleCount + quantsCount
        val dHex = bytes[dIndex].toUHex() + bytes[dIndex + 1].toUHex()
        val dminHex = bytes[dIndex + 2].toUHex() + bytes[dIndex + 3].toUHex()
        fputs("d=$dHex dmin=$dminHex\n", file)

        val quantsBase = scaleCount
        val bytesPerSubBlock = quantsCount / (QK_K / 16)

        for (sub in 0 until QK_K / 16) {
            val scaleByte = bytes[sub]
            val start = quantsBase + sub * bytesPerSubBlock
            val quantsHex = bytes.slice(start until start + bytesPerSubBlock).joinToString(" ") { it.toUHex() }
            val line = "subBlock=$sub scale=${scaleByte.toUHex()} quants=$quantsHex\n"
            fputs(line, file)
        }
        fflush(file)
        fclose(file)
    }

    private fun dumpRecordedFloats(path: String) {
        val file = fopen(path, "w") ?: error("Failed to open $path")
        val headerLine = "d=${Q2KDiagnosticsRecorder.headerD} dmin=${Q2KDiagnosticsRecorder.headerDMin}\n"
        fputs(headerLine, file)
        for (entry in Q2KDiagnosticsRecorder.entries) {
            val quantsHex = entry.quants.joinToString(" ") { ((it.toInt() and 0xFF)).toString(16).padStart(2, '0') }
            val line = "subBlock=${entry.index} scale=${entry.scale} min=${entry.min} quants=$quantsHex\n"
            fputs(line, file)
        }
        fflush(file)
        fclose(file)
    }

    private fun dumpTrace(path: String) {
        val entries = Q2KDiagnosticsRecorder.traceEntries
        val file = fopen(path, "w") ?: error("Failed to open $path")
        if (entries.isEmpty()) {
            fputs("<empty>\n", file)
            fflush(file)
            fclose(file)
            return
        }
        for (entry in entries) {
            val lhsHex = entry.lhsBits.toUInt().toString(16).padStart(8, '0')
            val rhsHex = entry.rhsBits?.let { it.toUInt().toString(16).padStart(8, '0') } ?: "--------"
            val resultHex = entry.resultBits.toUInt().toString(16).padStart(8, '0')
            val line = "${entry.op}: lhs=$lhsHex (${entry.lhs}), rhs=$rhsHex (${entry.rhs ?: "null"}), result=$resultHex (${entry.result})\n"
            fputs(line, file)
        }
        fflush(file)
        fclose(file)
    }
}
