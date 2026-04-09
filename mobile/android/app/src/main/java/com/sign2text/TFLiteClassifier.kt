package com.sign2text

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream

class TFLiteClassifier(
    context: Context,
    modelAssetName: String = "asl_mlp.tflite",
    numThreads: Int = 2
) {
    private val interpreter: Interpreter

    /** Logits length; must match label_map.json. */
    val numClasses: Int

    /** Feature width (63 or 83); must match feature_norm.json and HandPreprocessor. */
    val inputDim: Int

    init {
        val model = loadModelFile(context, modelAssetName)
        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
        }
        interpreter = Interpreter(model, options)
        val inShape = interpreter.getInputTensor(0).shape()
        inputDim = inShape[inShape.size - 1]
        numClasses = inferOutputClassCount(interpreter)
    }

    private fun inferOutputClassCount(interpreter: Interpreter): Int {
        val shape = interpreter.getOutputTensor(0).shape()
        return when (shape.size) {
            1 -> shape[0]
            2 -> shape[1]
            else -> error("Unexpected output rank ${shape.contentToString()}")
        }
    }

    fun close() {
        interpreter.close()
    }

    /**
     * Runs the model and returns **class probabilities** (softmax over logits / temperature).
     */
    fun predict(features: FloatArray, temperature: Float): FloatArray {
        require(features.size == inputDim) {
            "Expected $inputDim features, got ${features.size}"
        }
        val input = Array(1) { features }
        val logits = Array(1) { FloatArray(numClasses) }
        interpreter.run(input, logits)
        val raw = logits[0]
        // Legacy exports used softmax in-graph; new training outputs logits.
        if (raw.all { it >= 0f && it <= 1.01f } && kotlin.math.abs(raw.sum() - 1f) < 0.08f) {
            return raw
        }
        return softmax(raw, temperature)
    }

    private fun softmax(logits: FloatArray, temperature: Float): FloatArray {
        val t = temperature.coerceAtLeast(1e-6f)
        var mx = Float.NEGATIVE_INFINITY
        for (x in logits) if (x > mx) mx = x
        val out = FloatArray(logits.size)
        var sum = 0f
        for (i in logits.indices) {
            out[i] = kotlin.math.exp((logits[i] - mx) / t)
            sum += out[i]
        }
        for (i in out.indices) out[i] /= sum
        return out
    }

    private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
        val fd = context.assets.openFd(assetName)
        FileInputStream(fd.fileDescriptor).use { fis ->
            val channel = fis.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }
}
