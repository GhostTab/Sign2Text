package com.sign2text

import android.util.Log
import com.sign2text.BuildConfig

/**
 * Compare live [FloatArray] (63) to training distribution — enable DEBUG builds to log.
 * Cross-check with: python scripts/dataset_feature_stats.py
 */
object FeatureDiagnostics {
    private const val TAG = "Sign2TextFeatures"
    private var lastLogMs = 0L
    private const val LOG_INTERVAL_MS = 800L

    fun maybeLog(features: FloatArray) {
        if (!BuildConfig.DEBUG || features.isEmpty()) return
        val now = System.currentTimeMillis()
        if (now - lastLogMs < LOG_INTERVAL_MS) return
        lastLogMs = now

        val head = minOf(63, features.size)
        var s = 0.0
        for (i in 0 until head) s += kotlin.math.abs(features[i].toDouble())
        val meanAbs = (s / head).toFloat()
        val checksum = features.fold(0L) { acc, v -> acc * 31L + (v.toBits().toLong() and 0xFFFFFFFFL) }

        Log.d(
            TAG,
            "live f (first $head): meanAbs=${"%.4f".format(meanAbs)} " +
                "min=${features.minOrNull()} max=${features.maxOrNull()} " +
                "f0..f5=[${features[0]},${features[1]},${features[2]},${features[3]},${features[4]},${features[5]}] " +
                "hash=${checksum}"
        )
    }
}
