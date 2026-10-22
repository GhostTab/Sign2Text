package com.sign2text

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ZStrokeDetectorTest {

    @Test
    fun drawnZIsRecognized() {
        val detector = ZStrokeDetector(holdMs = 0L)
        assertTrue(feed(detector, zPath(leftToRight = true)))
    }

    @Test
    fun mirroredZIsRecognized() {
        val detector = ZStrokeDetector(holdMs = 0L)
        assertTrue(feed(detector, zPath(leftToRight = false)))
    }

    @Test
    fun stillFingerIsNotZ() {
        val detector = ZStrokeDetector()
        val points = List(16) { 0.40f + (it % 2) * 0.004f to 0.45f }
        assertFalse(feed(detector, points))
    }

    @Test
    fun straightLineIsNotZ() {
        val detector = ZStrokeDetector()
        val points = List(16) { i -> 0.20f + i * 0.03f to 0.40f }
        assertFalse(feed(detector, points))
    }

    @Test
    fun curledHandDoesNotCount() {
        val detector = ZStrokeDetector(holdMs = 0L)
        assertFalse(feed(detector, zPath(leftToRight = true), indexExtended = false))
    }

    private fun feed(
        detector: ZStrokeDetector,
        points: List<Pair<Float, Float>>,
        indexExtended: Boolean = true,
    ): Boolean {
        var hit = false
        points.forEachIndexed { i, (x, y) ->
            if (detector.update(i * 40L, x, y, indexExtended, handScale = 0.18f)) hit = true
        }
        return hit
    }

    private fun zPath(leftToRight: Boolean): List<Pair<Float, Float>> {
        val pts = ArrayList<Pair<Float, Float>>()
        val x0 = if (leftToRight) 0.20f else 0.60f
        val x1 = if (leftToRight) 0.60f else 0.20f
        val step = if (leftToRight) 1 else -1
        for (i in 0..8) pts.add(x0 + step * i * 0.05f to 0.30f)
        for (i in 1..8) pts.add(x1 - step * i * 0.05f to 0.30f + i * 0.04f)
        for (i in 1..8) pts.add(x0 + step * i * 0.05f to 0.62f)
        return pts
    }
}
