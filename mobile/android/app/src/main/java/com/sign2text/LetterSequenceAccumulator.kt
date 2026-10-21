package com.sign2text

/**
 * Builds a spelling buffer from debounced display letters: appends each **transition** to a new
 * stable A–Z (see [LetterDisplayDebouncer]); repeated frames of the same letter append once.
 */
class LetterSequenceAccumulator {
    private val buffer = StringBuilder()
    private var previousDebounced: String? = null

    val spelling: String
        get() = synchronized(this) { buffer.toString() }

    fun onDebouncedLetter(debounced: String) = synchronized(this) {
        val prev = previousDebounced
        previousDebounced = debounced
        if (debounced == prev) return@synchronized
        if (debounced.length == 1) {
            val c = debounced[0]
            if (c in 'A'..'Z') {
                buffer.append(c)
            }
        }
    }

    fun clear() = synchronized(this) {
        buffer.clear()
        previousDebounced = null
    }

    /** Removes the last letter so the user can re-sign a correction. */
    fun deleteLastLetter(): Boolean = synchronized(this) {
        if (buffer.isEmpty()) return@synchronized false
        buffer.deleteCharAt(buffer.lastIndex)
        previousDebounced = null
        true
    }

    /**
     * Atomically reads the buffer and clears it (for save).
     * Keeps [previousDebounced] so the same on-screen letter is not appended again until it changes.
     */
    fun takeSpellingAndClear(): String = synchronized(this) {
        val s = buffer.toString()
        buffer.clear()
        s
    }
}
