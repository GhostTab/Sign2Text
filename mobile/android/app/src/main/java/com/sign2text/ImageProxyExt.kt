package com.sign2text

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import androidx.camera.core.ImageProxy

/**
 * Camera frame → Bitmap for MediaPipe. Prefer RGBA analysis output (see MainActivity) to avoid JPEG loss.
 */
fun ImageProxy.toBitmap(): Bitmap {
    // CameraX RGBA8888 analysis: single plane, 4 bytes/pixel (avoid YUV→JPEG path).
    if (planes.size == 1 && planes[0].pixelStride == 4) {
        return rgba8888ToBitmap(this)
    }
    require(format == ImageFormat.YUV_420_888)
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = android.graphics.YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = java.io.ByteArrayOutputStream()
    // Higher quality reduces JPEG artifacts that shift MediaPipe landmarks vs training (static RGB images).
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 95, out)
    val jpegBytes = out.toByteArray()
    var bmp = android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)

    val rot = imageInfo.rotationDegrees
    if (rot != 0) {
        val matrix = Matrix()
        matrix.postRotate(rot.toFloat())
        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
        bmp.recycle()
        bmp = rotated
    }

    return bmp
}

/**
 * Horizontal mirror so front-camera analysis matches the mirrored [PreviewView] selfie preview.
 */
fun Bitmap.mirrorHorizontal(): Bitmap {
    val m = Matrix().apply { postScale(-1f, 1f, width / 2f, height / 2f) }
    return Bitmap.createBitmap(this, 0, 0, width, height, m, true)
}

/**
 * Direct copy from CameraX RGBA analysis stream (no JPEG), then rotation to upright preview.
 * Respects [ImageProxy.getCropRect] when Preview + ImageAnalysis share a [ViewPort] so the bitmap
 * matches what the user sees in the preview (same sensor crop / FOV).
 */
private fun rgba8888ToBitmap(image: ImageProxy): Bitmap {
    val plane = image.planes[0]
    val buffer = plane.buffer
    val rowStride = plane.rowStride
    val pixelStride = plane.pixelStride
    require(pixelStride == 4) { "Expected RGBA8888 pixelStride=4, got $pixelStride" }

    val crop: Rect = image.cropRect.takeIf { !it.isEmpty && it.width() > 0 && it.height() > 0 }
        ?: Rect(0, 0, image.width, image.height)
    val cropW = crop.width().coerceAtLeast(1)
    val cropH = crop.height().coerceAtLeast(1)

    val bmp = Bitmap.createBitmap(cropW, cropH, Bitmap.Config.ARGB_8888)
    val pixels = IntArray(cropW * cropH)
    var dst = 0
    for (y in 0 until cropH) {
        val sy = crop.top + y
        for (x in 0 until cropW) {
            val sx = crop.left + x
            val offset = sy * rowStride + sx * pixelStride
            val r = buffer.get(offset).toInt() and 0xFF
            val g = buffer.get(offset + 1).toInt() and 0xFF
            val b = buffer.get(offset + 2).toInt() and 0xFF
            val a = buffer.get(offset + 3).toInt() and 0xFF
            pixels[dst++] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }
    }
    buffer.rewind()
    bmp.setPixels(pixels, 0, cropW, 0, 0, cropW, cropH)

    // With ImageAnalysis.setOutputImageRotationEnabled(true), pixels are usually already upright
    // (rotationDegrees == 0). Only rotate when the device still reports a non-zero correction.
    val rot = image.imageInfo.rotationDegrees
    if (rot == 0) {
        return bmp
    }
    val matrix = Matrix()
    matrix.postRotate(rot.toFloat())
    val out = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
    if (out !== bmp) {
        bmp.recycle()
    }
    return out
}

