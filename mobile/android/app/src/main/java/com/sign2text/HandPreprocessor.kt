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
     * xyz of index, middle, ring, and pinky PIP minus the thumb tip.
     * Matches [geometric_features.thumb_relations_from_flat63]. This is the M-vs-N signal:
     * which fingers sit over the tucked thumb.
     */
    fun thumbRelations12(feats63: FloatArray): FloatArray {
        require(feats63.size == 63)
        val thumbTip = 4
        val pips = intArrayOf(6, 10, 14, 18)
        val tx = feats63[thumbTip * 3]
        val ty = feats63[thumbTip * 3 + 1]
        val tz = feats63[thumbTip * 3 + 2]
        val out = FloatArray(12)
        for (k in pips.indices) {
            val b = pips[k] * 3
            out[k * 3] = feats63[b] - tx
            out[k * 3 + 1] = feats63[b + 1] - ty
            out[k * 3 + 2] = feats63[b + 2] - tz
        }
        return out
    }

    /** Distance from thumb tip to index tip in the wrist-scaled landmark frame. O is small, C is large. */
    fun thumbIndexGap(features: FloatArray): Float {
        require(features.size >= 63)
        val tx = features[4 * 3]
        val ty = features[4 * 3 + 1]
        val tz = features[4 * 3 + 2]
        val ix = features[8 * 3]
        val iy = features[8 * 3 + 1]
        val iz = features[8 * 3 + 2]
        val dx = tx - ix
        val dy = ty - iy
        val dz = tz - iz
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    /** 63 landmarks + 20 wrist distances + 12 thumb relations. */
    const val MODEL_FEATURE_DIM = 95

    /**
     * Single-frame model input: 63-D from landmarks, or 95-D with wrist distances and thumb relations.
     */
    fun toModelFeatures(landmarks: List<NormalizedLandmark>, useGeometry: Boolean): FloatArray {
        val f63 = toFeatures63(landmarks)
        if (!useGeometry) return f63
        val d = wristDistances20(f63)
        val t = thumbRelations12(f63)
        return FloatArray(MODEL_FEATURE_DIM) { i ->
            when {
                i < 63 -> f63[i]
                i < 83 -> d[i - 63]
                else -> t[i - 83]
            }
        }
    }
}
