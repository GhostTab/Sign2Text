package com.sign2text

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.sqrt

/**
 * Heuristic "open hand" detector from MediaPipe hand landmarks (no extra model).
 * Tuned for finger-spelling context: extended fingers vs curled.
 */
object OpenPalmDetector {
    private const val WRIST = 0
    private const val THUMB_TIP = 4
    private const val THUMB_IP = 3
    private const val INDEX_PIP = 6
    private const val INDEX_TIP = 8
    private const val MIDDLE_PIP = 10
    private const val MIDDLE_TIP = 12
    private const val RING_PIP = 14
    private const val RING_TIP = 16
    private const val PINKY_PIP = 18
    private const val PINKY_TIP = 20

    /** Ratio wrist→tip must exceed wrist→PIP (thumb: wrist→IP) to count as extended. */
    private const val EXTENSION_RATIO = 1.06f

    /** Need this many of 5 checks to call the pose an open palm. */
    private const val MIN_EXTENDED_FINGERS = 4

    fun isOpenPalm(landmarks: List<NormalizedLandmark>): Boolean {
        if (landmarks.size < 21) return false
        val w = landmarks[WRIST]
        var extended = 0
        if (isExtended(w, landmarks[THUMB_TIP], landmarks[THUMB_IP])) extended++
        if (isExtended(w, landmarks[INDEX_TIP], landmarks[INDEX_PIP])) extended++
        if (isExtended(w, landmarks[MIDDLE_TIP], landmarks[MIDDLE_PIP])) extended++
        if (isExtended(w, landmarks[RING_TIP], landmarks[RING_PIP])) extended++
        if (isExtended(w, landmarks[PINKY_TIP], landmarks[PINKY_PIP])) extended++
        return extended >= MIN_EXTENDED_FINGERS
    }

    private fun isExtended(
        wrist: NormalizedLandmark,
        tip: NormalizedLandmark,
        joint: NormalizedLandmark
    ): Boolean {
        val dTip = dist(wrist, tip)
        val dJoint = dist(wrist, joint)
        return dTip > EXTENSION_RATIO * dJoint
    }

    private fun dist(a: NormalizedLandmark, b: NormalizedLandmark): Float {
        val dx = (a.x() - b.x()).toDouble()
        val dy = (a.y() - b.y()).toDouble()
        val dz = (a.z() - b.z()).toDouble()
        return sqrt(dx * dx + dy * dy + dz * dz).toFloat()
    }
}
