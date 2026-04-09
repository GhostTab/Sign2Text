package com.sign2text

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.sqrt

object HandPreprocessor {
    /** MediaPipe: middle finger MCP = landmark 9 */
    private const val MID_MCP_IDX = 9

    /**
     * Match Python `landmark_preprocessing.normalize_landmarks_xyz`:
     * - translate so wrist (0) is origin
     * - scale by max wrist-to-point distance in XY
     * - if middle MCP is left of wrist (x < 0 after center+scale), mirror X (canonical right-hand frame)
     */
    fun toFeatures63(landmarks: List<NormalizedLandmark>): FloatArray {
        require(landmarks.size >= 21)

        val wrist = landmarks[0]

        var maxD = 0f
        for (i in 0 until 21) {
            val p = landmarks[i]
            val dx = (p.x() - wrist.x()).toFloat()
            val dy = (p.y() - wrist.y()).toFloat()
            val d = sqrt(dx * dx + dy * dy)
            if (d > maxD) maxD = d
        }
        val scale = if (maxD > 1e-6f) maxD else 1f

        val feats = FloatArray(63)
        for (i in 0 until 21) {
            val p = landmarks[i]
            val base = i * 3
            feats[base] = ((p.x() - wrist.x()).toFloat()) / scale
            feats[base + 1] = ((p.y() - wrist.y()).toFloat()) / scale
            feats[base + 2] = ((p.z() - wrist.z()).toFloat()) / scale
        }

        val midX = feats[MID_MCP_IDX * 3]
        if (midX < 0f) {
            for (i in 0 until 21) {
                feats[i * 3] = -feats[i * 3]
            }
        }

        return feats
    }

    /**
     * L2 distance from wrist (origin) to landmarks 1..20 — matches [geometric_features.wrist_distances_from_flat63].
     */
    fun wristDistances20(feats63: FloatArray): FloatArray {
        require(feats63.size == 63)
        val out = FloatArray(20)
        for (i in 1 until 21) {
            val b = i * 3
            val x = feats63[b]
            val y = feats63[b + 1]
            val z = feats63[b + 2]
            out[i - 1] = sqrt(x * x + y * y + z * z)
        }
        return out
    }

    /**
     * Single-frame model input: 63-D from landmarks, optionally +20 wrist distances (83-D).
     */
    fun toModelFeatures(landmarks: List<NormalizedLandmark>, useGeometry: Boolean): FloatArray {
        val f63 = toFeatures63(landmarks)
        if (!useGeometry) return f63
        val d = wristDistances20(f63)
        return FloatArray(83) { i ->
            if (i < 63) f63[i] else d[i - 63]
        }
    }
}
