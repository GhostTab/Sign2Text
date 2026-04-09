package com.sign2text

/**
 * EMA probability smoothing for stable real-time output.
 */
class Smoother(private val numClasses: Int, private val alpha: Float = 0.3f) {
    private var ema: FloatArray? = null

    fun reset() {
        ema = null
    }

    fun update(probs: FloatArray): FloatArray {
        require(probs.size == numClasses)
        val prev = ema
        if (prev == null) {
            ema = probs.copyOf()
            return probs
        }
        for (i in probs.indices) {
            prev[i] = alpha * probs[i] + (1f - alpha) * prev[i]
        }
        ema = prev
        return prev
    }

    fun argmax(probs: FloatArray): Pair<Int, Float> {
        var bestI = 0
        var bestV = Float.NEGATIVE_INFINITY
        for (i in probs.indices) {
            val v = probs[i]
            if (v > bestV) {
                bestV = v
                bestI = i
            }
        }
        return bestI to bestV
    }
}

