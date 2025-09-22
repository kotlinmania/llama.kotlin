package ai.solace.klang.bitwise

/**
 * ArrayShiftTrace: opt-in tracing for ArrayBitShifts. Disabled by default.
 * Use ArrayShiftTrace.withTracing { ... } to capture entries during a block.
 */
object ArrayShiftTrace {
    data class Entry(
        val op: String,           // "shl16" | "rsh16" | "shl16w" | "rsh16w"
        val s: Int,               // bit count (or word count for *w ops)
        val from: Int,
        val len: Int,
        val before: IntArray,     // 16-bit limbs (LE) before
        val after: IntArray,      // 16-bit limbs (LE) after
        val carryOut: Int,
        val sticky: Boolean,
    )

    var enabled: Boolean = false

    private val _entries = mutableListOf<Entry>()
    val entries: List<Entry> get() = _entries.toList()

    fun <T> withTracing(block: () -> T): Pair<T, List<Entry>> {
        val prev = enabled
        enabled = true
        _entries.clear()
        return try {
            val result = block()
            result to entries
        } finally {
            enabled = prev
        }
    }

    fun record(op: String, s: Int, from: Int, len: Int, before: IntArray, after: IntArray, carryOut: Int, sticky: Boolean) {
        if (!enabled) return
        val b = before.copyOf()
        val a = after.copyOf()
        _entries += Entry(op, s, from, len, b, a, carryOut and 0xFFFF, sticky)
    }
}
