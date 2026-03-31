package ai.solace.llamakotlin.core

import ai.solace.klangnative.bitwise.CFloatTrace

internal object Q2KDiagnosticsRecorder {
    data class SubBlockEntry(
        val index: Int,
        val scale: Float,
        val min: Float,
        val quants: ByteArray
    )

    var enabled: Boolean = false
    private val _entries = mutableListOf<SubBlockEntry>()
    val entries: List<SubBlockEntry> get() = _entries
    var headerD: Short = 0
    var headerDMin: Short = 0
    var traceEntries: List<CFloatTrace.Entry> = emptyList()

    fun reset() {
        _entries.clear()
        headerD = 0
        headerDMin = 0
        traceEntries = emptyList()
    }

    fun recordHeader(d: Short, dmin: Short) {
        if (!enabled) return
        headerD = d
        headerDMin = dmin
    }

    fun recordSubBlock(index: Int, scale: Float, min: Float, quants: ByteArray) {
        if (!enabled) return
        _entries += SubBlockEntry(index, scale, min, quants.copyOf())
    }

    fun recordTrace(entries: List<CFloatTrace.Entry>) {
        if (!enabled) return
        traceEntries = entries
    }
}
