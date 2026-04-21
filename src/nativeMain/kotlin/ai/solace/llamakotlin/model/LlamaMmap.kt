// port-lint: source llama.cpp/src/llama-mmap.h
package ai.solace.llamakotlin.model

// =============================================================================
// LlamaMmap – memory-mapped file types for model loading
// Ported from: llama-mmap.h
// =============================================================================

/**
 * Abstraction over a file handle with positioned read/write operations.
 *
 * Port of `llama_file` from `llama-mmap.h`. On Kotlin/Native the actual
 * implementation will use platform-specific POSIX I/O.
 *
 * @property path  The filesystem path that was opened.
 */
class LlamaFile(val path: String) {

    private var position: Long = 0
    private var fileSize: Long = 0

    /** Current read/write position in bytes. */
    fun tell(): Long = position

    /** Total size of the file in bytes. */
    fun size(): Long = fileSize

    /** Seek to an absolute [offset]. */
    fun seek(offset: Long) {
        TODO("Port llama_file::seek — implement via POSIX lseek or Kotlin/Native file API")
    }

    /**
     * Read exactly [len] bytes into [dst] starting at [dstOffset].
     *
     * @throws IllegalStateException if fewer bytes are available.
     */
    fun readRaw(dst: ByteArray, len: Int, dstOffset: Int = 0) {
        TODO("Port llama_file::read_raw — implement via POSIX read")
    }

    /** Read a little-endian 32-bit unsigned integer. */
    fun readU32(): UInt {
        TODO("Port llama_file::read_u32")
    }

    /** Write [len] bytes from [src] starting at [srcOffset]. */
    fun writeRaw(src: ByteArray, len: Int, srcOffset: Int = 0) {
        TODO("Port llama_file::write_raw — implement via POSIX write")
    }

    /** Write a little-endian 32-bit unsigned integer. */
    fun writeU32(value: UInt) {
        TODO("Port llama_file::write_u32")
    }

    /** Alignment used for direct-I/O reads, or 1 for normal I/O. */
    fun readAlignment(): Long {
        TODO("Port llama_file::read_alignment")
    }

    /** Whether this file was opened with direct I/O (O_DIRECT). */
    fun hasDirectIo(): Boolean = false

    companion object {
        /**
         * Open a file at [path] with the given [mode] (e.g. "rb", "wb").
         *
         * @param useDirectIo  Attempt O_DIRECT for bypassing the page cache.
         */
        fun open(path: String, mode: String, useDirectIo: Boolean = false): LlamaFile {
            TODO("Port llama_file constructor — implement via POSIX open")
        }
    }
}

/**
 * Memory-mapped region of a [LlamaFile].
 *
 * Port of `llama_mmap` from `llama-mmap.h`. Uses platform mmap(2) underneath.
 *
 * @property file  The file being mapped.
 */
class LlamaMmap private constructor(
    val file: LlamaFile,
) {
    private var mappedSize: Long = 0
    private var mappedAddr: Long = 0  // raw pointer address (platform-specific)

    /** Size of the mapped region in bytes. */
    fun size(): Long = mappedSize

    /**
     * Unmap a fragment of the region between byte offsets [first] and [last].
     *
     * This can release physical pages back to the OS for memory savings.
     */
    fun unmapFragment(first: Long, last: Long) {
        TODO("Port llama_mmap::unmap_fragment — implement via munmap")
    }

    companion object {
        /** Whether memory-mapping is supported on the current platform. */
        val SUPPORTED: Boolean = true  // macOS and Linux always support mmap

        /**
         * Create a new memory-mapping of [file].
         *
         * @param file      The file to map.
         * @param prefetch  Number of bytes to prefetch (hint to the OS).
         * @param numa      Whether to use NUMA-aware allocation.
         */
        fun create(file: LlamaFile, prefetch: Long = -1, numa: Boolean = false): LlamaMmap {
            TODO("Port llama_mmap constructor — implement via mmap(2)")
        }
    }
}

/**
 * Pin physical pages in memory to prevent swapping.
 *
 * Port of `llama_mlock` from `llama-mmap.h`. Uses mlock(2) on supported platforms.
 */
class LlamaMlock {

    private var lockedAddr: Long = 0
    private var lockedSize: Long = 0

    /**
     * Set the base address to lock.
     *
     * Must be called before [growTo].
     */
    fun init(addr: Long) {
        TODO("Port llama_mlock::init")
    }

    /**
     * Grow the locked region to [targetSize] bytes.
     *
     * Pages between the current locked size and [targetSize] are locked.
     */
    fun growTo(targetSize: Long) {
        TODO("Port llama_mlock::grow_to — implement via mlock(2)")
    }

    companion object {
        /** Whether memory locking is supported on the current platform. */
        val SUPPORTED: Boolean = true  // macOS and Linux support mlock
    }
}

/**
 * Return the platform maximum path length.
 *
 * Port of `llama_path_max()` from `llama-mmap.h`.
 */
fun llamaPathMax(): Long = 4096  // PATH_MAX on most POSIX systems
