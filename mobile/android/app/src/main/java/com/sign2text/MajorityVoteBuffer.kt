package com.sign2text

/**
 * Sliding window majority vote on predicted class indices to reduce single-frame flips.
 * Call [push] with the current top-1 class; [stableClass] returns the mode if it meets [minVotes],
 * else null (caller may show "?" or keep previous).
 */
class MajorityVoteBuffer(
    private val windowSize: Int = 7,
    private val minVotes: Int = 4
) {
    private val buf = ArrayDeque<Int>(windowSize + 1)

    fun reset() {
        buf.clear()
    }

    fun push(classIndex: Int) {
        buf.addLast(classIndex)
        while (buf.size > windowSize) {
            buf.removeFirst()
        }
    }

    /**
     * Mode of current window if count >= [minVotes], else null.
     */
    fun stableClass(): Int? {
        if (buf.isEmpty()) return null
        val counts = mutableMapOf<Int, Int>()
        for (c in buf) {
            counts[c] = (counts[c] ?: 0) + 1
        }
        val best = counts.maxByOrNull { it.value } ?: return null
        return if (best.value >= minVotes) best.key else null
    }
}
