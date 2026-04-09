package com.sign2text

/**
 * Debounces the **displayed** letter:
 * - A new letter must appear as the candidate [framesToConfirm] times in a row before replacing [committed].
 * - Uncertain frames ([showLetter]=false) for [framesToClear] consecutive frames clear to "?".
 */
class LetterDisplayDebouncer(
    private val framesToConfirm: Int = 5,
    private val framesToClear: Int = 2
) {
    private var pending: String? = null
    private var streak = 0
    private var committed: String = "?"
    private var uncertainStreak = 0

    fun reset() {
        pending = null
        streak = 0
        committed = "?"
        uncertainStreak = 0
    }

    fun update(showLetter: Boolean, letter: String): String {
        if (!showLetter) {
            uncertainStreak++
            if (uncertainStreak >= framesToClear) {
                pending = null
                streak = 0
                committed = "?"
            }
            return committed
        }
        uncertainStreak = 0
        if (letter == pending) {
            streak++
        } else {
            pending = letter
            streak = 1
        }
        if (streak >= framesToConfirm) {
            committed = letter
        }
        return committed
    }
}
