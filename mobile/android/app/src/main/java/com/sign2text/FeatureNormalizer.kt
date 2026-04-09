package com.sign2text

import org.json.JSONObject

/**
 * Applies the same z-score as training: x' = (x - mean) / std.
 * Loaded from assets/feature_norm.json (written by train_mlp.py).
 * Supports input_dim 63 or 83 (with geometric wrist distances).
 */
class FeatureNormalizer(
    private val mean: FloatArray,
    private val std: FloatArray
) {
    val inputDim: Int get() = mean.size

    init {
        require(mean.size == std.size && mean.isNotEmpty())
    }

    fun normalizeInPlace(features: FloatArray) {
        require(features.size == mean.size)
        for (i in mean.indices) {
            features[i] = (features[i] - mean[i]) / std[i]
        }
    }

    companion object {
        fun fromJson(json: String): FeatureNormalizer {
            val obj = JSONObject(json)
            val m = obj.getJSONArray("mean")
            val s = obj.getJSONArray("std")
            val dim = obj.optInt("input_dim", m.length())
            require(dim == m.length() && dim == s.length())
            val mean = FloatArray(dim) { i -> m.getDouble(i).toFloat() }
            val std = FloatArray(dim) { i -> kotlin.math.max(s.getDouble(i).toFloat(), 1e-6f) }
            return FeatureNormalizer(mean, std)
        }
    }
}
