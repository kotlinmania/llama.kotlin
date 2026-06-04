package io.github.kotlinmania.llama.zlib.zip

import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem

internal val supportsZipFileIO: Boolean by lazy {
    return@lazy try {
        val tmpDir = Path("./build/tmp/zip-capability-check")
        SystemFileSystem.createDirectories(tmpDir, mustCreate = false)
        true
    } catch (_: Throwable) {
        false
    }
}
