package com.sign2text

/**
 * C, M, N, O, P, T, U, and Z can show a little sooner than the other letters, but only
 * when that letter is clearly the best guess. A resting hand stays blank.
 */
object LetterAcceptance {
    val lenient = setOf("C", "M", "N", "O", "P", "T", "U", "Z")

    private const val LENIENT_MIN_CONF = 0.55f
    private const val LENIENT_MIN_MARGIN = 0.12f

    fun choose(
        topLabel: String,
        topP: Float,
        secondLabel: String?,
        secondP: Float,
        entropy: Float,
        maxEntropy: Float,
        zDrawn: Boolean,
        minConfidence: Float = 0.60f,
        minMargin: Float = 0.18f,
        thumbIndexGap: Float = 1f,
    ): String? {
        if (zDrawn) return "Z"
        val margin = topP - secondP
        // A resting hand used to become C or O. Only split those two after the
        // model is already sure the shape is one of them.
        if ((topLabel == "C" || topLabel == "O") && topP >= LENIENT_MIN_CONF && margin >= LENIENT_MIN_MARGIN) {
            if (thumbIndexGap < 0.42f) return "O"
            if (thumbIndexGap > 0.48f) return "C"
        }
        if (topLabel in lenient && topP >= LENIENT_MIN_CONF && margin >= LENIENT_MIN_MARGIN) {
            return topLabel
        }
        if (topP >= minConfidence && margin >= minMargin && entropy <= maxEntropy) return topLabel
        return null
    }
}
