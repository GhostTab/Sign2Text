package com.sign2text

import kotlin.math.abs
import kotlin.math.hypot

/**
 * Recognizes the ASL letter Z from the index fingertip path: a horizontal stroke,
 * a diagonal down the other way, then a horizontal stroke in the first direction.
 * A still pointed finger does not count.
 */
class ZStrokeDetector(
    private val windowMs: Long = 1800L,
    private val holdMs: Long = 800L,
) {
    private data class Pt(val x: Float, val y: Float)
    private data class Sample(val t: Long, val p: Pt, val scale: Float)

    private val samples = ArrayDeque<Sample>()
    private var holdUntilMs = 0L

    fun reset() {
        samples.clear()
        holdUntilMs = 0L
    }

    /**
     * @param indexExtended index finger is out and the other fingers are curled
     * @param handScale wrist-to-middle-knuckle distance in the same units as x and y
     */
    fun update(
        timeMs: Long,
        x: Float,
        y: Float,
        indexExtended: Boolean,
        handScale: Float,
    ): Boolean {
        if (timeMs < holdUntilMs) return true

        val last = samples.lastOrNull()
        if (!indexExtended || handScale < 0.05f) {
            if (last != null && timeMs - last.t > 220L) samples.clear()
            return false
        }

        samples.addLast(Sample(timeMs, Pt(x, y), handScale))
        while (samples.isNotEmpty() && timeMs - samples.first().t > windowMs) {
            samples.removeFirst()
        }
        if (samples.size < 6) return false

        val scale = samples.map { it.scale }.sorted()[samples.size / 2]
        val points = samples.map { it.p }
        if (!looksLikeZ(points, scale)) return false

        samples.clear()
        holdUntilMs = timeMs + holdMs
        return true
    }

    private fun looksLikeZ(points: List<Pt>, handScale: Float): Boolean {
        var length = 0f
        for (i in 1 until points.size) {
            length += hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y)
        }
        if (length < handScale * 1.05f) return false

        val path = resample(points, 15)
        if (path.size < 15) return false
        val a = path[0]
        val b = path[5]
        val c = path[10]
        val d = path[14]
        val s1x = b.x - a.x
        val s1y = b.y - a.y
        val s2x = c.x - b.x
        val s2y = c.y - b.y
        val s3x = d.x - c.x
        val s3y = d.y - c.y

        if (!mostlyHorizontal(s1x, s1y) || !mostlyHorizontal(s3x, s3y)) return false
        if (s1x * s3x <= 0f) return false
        if (s2y < handScale * 0.22f) return false
        if (s2x * s1x >= 0f) return false
        if (abs(s1x) < handScale * 0.28f || abs(s3x) < handScale * 0.22f) return false
        return true
    }

    private fun mostlyHorizontal(dx: Float, dy: Float): Boolean {
        return abs(dx) > abs(dy) * 0.90f
    }

    private fun resample(points: List<Pt>, count: Int): List<Pt> {
        val lengths = FloatArray(points.size)
        var total = 0f
        for (i in 1 until points.size) {
            total += hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y)
            lengths[i] = total
        }
        if (total < 1e-4f) return emptyList()
        val out = ArrayList<Pt>(count)
        for (k in 0 until count) {
            val target = total * k / (count - 1).toFloat()
            var i = 1
            while (i < lengths.size - 1 && lengths[i] < target) i++
            val span = (lengths[i] - lengths[i - 1]).coerceAtLeast(1e-6f)
            val t = ((target - lengths[i - 1]) / span).coerceIn(0f, 1f)
            val p0 = points[i - 1]
            val p1 = points[i]
            out.add(Pt(p0.x + (p1.x - p0.x) * t, p0.y + (p1.y - p0.y) * t))
        }
        return out
    }
}
