package ai.solace.llamakotlin.core

internal object Q2KDiagnosticsRecorder {
    data class SubBlockEntry(
        val index: Int,
        val scale: Float,
        val min: Float,
        val quants: ByteArray
    )

    /** Diagnostic trace entry for float operations (op name, LHS/RHS/result bit patterns). */
    data class TraceEntry(
        val op: String,
        val lhsBits: Int,
        val rhsBits: Int?,
        val resultBits: Int
    ) {
        val lhs: Float get() = Float.fromBits(lhsBits)
        val rhs: Float? get() = rhsBits?.let { Float.fromBits(it) }
        val result: Float get() = Float.fromBits(resultBits)
    }

    var enabled: Boolean = false
    private val _entries = mutableListOf<SubBlockEntry>()
    val entries: List<SubBlockEntry> get() = _entries
    var headerD: Short = 0
    var headerDMin: Short = 0
    var traceEntries: List<TraceEntry> = emptyList()

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

    fun recordTrace(entries: List<TraceEntry>) {
        if (!enabled) return
        traceEntries = entries
    }
}
