// port-lint: source llama.cpp/src/llama-io.h
package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

// =============================================================================
// LlamaIO – serialization / deserialization interfaces
// Ported from: llama-io.h
// =============================================================================

/**
 * Interface for writing serialized state data.
 *
 * Implementations may write to a byte buffer, a file, or a network stream.
 *
 * Port of `llama_io_write_i` from `llama-io.h`.
 */
interface LlamaIoWrite {

    /**
     * Write raw bytes from [src].
     *
     * @param src   Source byte array.
     * @param size  Number of bytes to write (starting from index 0).
     */
    fun write(src: ByteArray, size: Int = src.size)

    /**
     * Write the data of a [tensor], starting at byte [offset] for [size] bytes.
     *
     * @param tensor  The tensor whose backing data should be written.
     * @param offset  Byte offset into the tensor's data buffer.
     * @param size    Number of bytes to write.
     */
    fun writeTensor(tensor: GGMLTensor, offset: Int, size: Int)

    /** Total number of bytes written so far. */
    fun nBytes(): Long

    /**
     * Convenience: write a [String] as a length-prefixed sequence of bytes.
     *
     * Port of `llama_io_write_i::write_string()`.
     */
    fun writeString(str: String) {
        val bytes = str.encodeToByteArray()
        val lenBytes = ByteArray(4)
        lenBytes[0] = (bytes.size and 0xFF).toByte()
        lenBytes[1] = ((bytes.size shr 8) and 0xFF).toByte()
        lenBytes[2] = ((bytes.size shr 16) and 0xFF).toByte()
        lenBytes[3] = ((bytes.size shr 24) and 0xFF).toByte()
        write(lenBytes)
        write(bytes)
    }
}

/**
 * Interface for reading serialized state data.
 *
 * Port of `llama_io_read_i` from `llama-io.h`.
 */
interface LlamaIoRead {

    /**
     * Read [size] bytes and return them as a new [ByteArray].
     *
     * @param size  Number of bytes to read.
     * @return A byte array of exactly [size] bytes.
     */
    fun read(size: Int): ByteArray

    /**
     * Read [size] bytes into the provided [dst] buffer.
     *
     * @param dst   Destination byte array (must be at least [size] bytes).
     * @param size  Number of bytes to read.
     */
    fun readTo(dst: ByteArray, size: Int)

    /** Total number of bytes read so far. */
    fun nBytes(): Long

    /**
     * Convenience: read a length-prefixed [String].
     *
     * Port of `llama_io_read_i::read_string()`.
     */
    fun readString(): String {
        val lenBytes = read(4)
        val len = (lenBytes[0].toInt() and 0xFF) or
                ((lenBytes[1].toInt() and 0xFF) shl 8) or
                ((lenBytes[2].toInt() and 0xFF) shl 16) or
                ((lenBytes[3].toInt() and 0xFF) shl 24)
        val strBytes = read(len)
        return strBytes.decodeToString()
    }
}
