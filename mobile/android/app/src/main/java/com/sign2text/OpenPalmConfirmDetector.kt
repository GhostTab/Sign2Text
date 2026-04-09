package com.sign2text

/**
 * Fires [true] once after [framesToConfirm] consecutive open-palm frames, then enters [cooldownFrames]
 * where no further confirms fire (avoids double triggers while the hand stays open).
 */
class OpenPalmConfirmDetector(
    private val framesToConfirm: Int = 12,
    private val cooldownFrames: Int = 20
) {
    private var openStreak = 0
    private var cooldownRemaining = 0

    fun reset() {
        openStreak = 0
        cooldownRemaining = 0
    }

    /**
     * @param isOpenPalm from [OpenPalmDetector.isOpenPalm]
     * @return true once when confirm gesture completes
     */
    fun update(isOpenPalm: Boolean): Boolean {
        if (cooldownRemaining > 0) {
            cooldownRemaining--
            if (!isOpenPalm) {
                openStreak = 0
            }
            return false
        }
        if (isOpenPalm) {
            openStreak++
            if (openStreak >= framesToConfirm) {
                openStreak = 0
                cooldownRemaining = cooldownFrames
                return true
            }
        } else {
            openStreak = 0
        }
        return false
    }
}
