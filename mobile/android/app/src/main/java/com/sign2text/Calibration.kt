package com.sign2text

import org.json.JSONObject

/**
 * Temperature scaling for logits (matches models/calibration.json from train_mlp.py).
 */
data class Calibration(
    val temperature: Float,
    val outputIsLogits: Boolean
) {
    companion object {
        fun fromJson(json: String): Calibration {
            val o = JSONObject(json)
            return Calibration(
                o.optDouble("temperature", 1.0).toFloat(),
                o.optBoolean("output_is_logits", true)
            )
        }
    }
}
