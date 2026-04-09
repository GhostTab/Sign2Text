package com.sign2text

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Rational
import android.util.Size
import android.content.res.ColorStateList
import android.graphics.Color
import android.view.View
import android.widget.TextView
import androidx.activity.ComponentActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.HandLandmarkerOptions
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.framework.image.BitmapImageBuilder
import java.util.concurrent.Executors


class MainActivity : ComponentActivity() {
    private companion object {
        /** Front camera mirrors analysis to match selfie preview; back camera uses sensor orientation as-is. */
        const val USE_FRONT_CAMERA = false
        const val MIN_CONFIDENCE_TO_SHOW = 0.52f
        const val MIN_MARGIN_TOP1_TOP2 = 0.08f
    }

    private lateinit var previewView: PreviewView
    private lateinit var predLetter: TextView
    private lateinit var spellingText: TextView
    private lateinit var wordMatchText: TextView
    private lateinit var hintText: TextView
    private lateinit var wordFormationHint: TextView
    private lateinit var topMessage: TextView
    private lateinit var wordModePanel: MaterialCardView
    private lateinit var toggleWordModeButton: MaterialButton
    private lateinit var confirmWordButton: MaterialButton

    private val executor = Executors.newSingleThreadExecutor()

    @Volatile
    private var wordModeEnabled = false

    private var handLandmarker: HandLandmarker? = null
    private var classifier: TFLiteClassifier? = null
    private var labelMap: LabelMap? = null
    private var featureNormalizer: FeatureNormalizer? = null
    private var smoother: Smoother? = null
    private var maxEntropyThreshold = 2.75f
    private var calibrationTemperature = 1f
    private val majorityVote = MajorityVoteBuffer(windowSize = 7, minVotes = 4)
    private val displayDebouncer = LetterDisplayDebouncer(framesToConfirm = 5, framesToClear = 2)
    private val letterAccumulator = LetterSequenceAccumulator()
    private val openPalmConfirm = OpenPalmConfirmDetector()
    private var wordDictionary: WordDictionary? = null

    private var lastLogPredictionMs = 0L
    private var lastLoggedDisplayedLetter: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        predLetter = findViewById(R.id.predLetter)
        wordModePanel = findViewById(R.id.wordModePanel)
        spellingText = findViewById(R.id.spellingText)
        wordMatchText = findViewById(R.id.wordMatchText)
        hintText = findViewById(R.id.hintText)
        wordFormationHint = findViewById(R.id.wordFormationHint)
        topMessage = findViewById(R.id.topMessage)
        toggleWordModeButton = findViewById(R.id.toggleWordModeButton)
        confirmWordButton = findViewById(R.id.confirmWordButton)
        toggleWordModeButton.setOnClickListener {
            setWordModeEnabled(!wordModeEnabled)
        }
        confirmWordButton.setOnClickListener {
            executor.execute { finalizeSpelledWord() }
        }
        applyWordModeUi()

        if (!hasCameraPermission()) {
            setTopMessage(getString(R.string.msg_permission_prompt), true)
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                1001
            )
        } else {
            initializePipeline()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            initializePipeline()
        } else if (requestCode == 1001) {
            setTopMessage(getString(R.string.msg_camera_permission), true)
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun setTopMessage(text: CharSequence?, visible: Boolean) {
        if (visible && !text.isNullOrBlank()) {
            topMessage.text = text
            topMessage.visibility = View.VISIBLE
        } else {
            topMessage.visibility = View.GONE
        }
    }

    private fun initializePipeline() {
        setTopMessage(getString(R.string.msg_preparing), true)
        hintText.setText(R.string.hint_idle)

        try {
            labelMap = LabelMap.fromJson(AssetUtils.readAssetText(this, "label_map.json"))
        } catch (e: Exception) {
            setTopMessage(getString(R.string.err_setup), true)
            predLetter.text = "—"
            return
        }

        try {
            featureNormalizer = FeatureNormalizer.fromJson(AssetUtils.readAssetText(this, "feature_norm.json"))
        } catch (_: Exception) {
            featureNormalizer = null
            Log.w(
                "Sign2Text",
                "feature_norm.json missing — required for models trained with z-score; add to assets after train_mlp.py"
            )
        }

        try {
            classifier = TFLiteClassifier(
                this,
                modelAssetName = "asl_mlp_fp32.tflite",
                numThreads = 2
            )
        } catch (e: Exception) {
            Log.e("Sign2Text", "TFLite load failed", e)
            setTopMessage(getString(R.string.err_setup), true)
            predLetter.text = "—"
            return
        }

        val nClassModel = classifier!!.numClasses
        val nClassLabels = labelMap!!.numClasses
        if (nClassModel != nClassLabels) {
            Log.e(
                "Sign2Text",
                "Class count mismatch: model outputs $nClassModel, label_map.json has $nClassLabels"
            )
            setTopMessage(getString(R.string.err_setup), true)
            predLetter.text = "—"
            classifier?.close()
            classifier = null
            return
        }

        val cl = classifier!!
        val fn = featureNormalizer
        if (fn != null && fn.inputDim != cl.inputDim) {
            setTopMessage(getString(R.string.err_setup), true)
            predLetter.text = "—"
            classifier?.close()
            classifier = null
            return
        }
        if (cl.inputDim == 83 && fn == null) {
            setTopMessage(getString(R.string.err_setup), true)
            predLetter.text = "—"
            classifier?.close()
            classifier = null
            return
        }

        try {
            calibrationTemperature =
                Calibration.fromJson(AssetUtils.readAssetText(this, "calibration.json")).temperature
        } catch (_: Exception) {
            calibrationTemperature = 1f
            Log.w("Sign2Text", "calibration.json missing — using temperature=1.0")
        }

        val nClass = nClassModel
        smoother = Smoother(numClasses = nClass, alpha = 0.7f)
        maxEntropyThreshold =
            if (nClass <= 1) 0f
            else (kotlin.math.ln(nClass.toDouble()) * 0.92).toFloat()

        try {
            handLandmarker = createHandLandmarker()
        } catch (e: Exception) {
            setTopMessage(getString(R.string.err_setup), true)
            predLetter.text = "—"
            return
        }

        wordDictionary = try {
            WordDictionary.fromLines(AssetUtils.readAssetText(this, "words_en.txt"))
        } catch (e: Exception) {
            Log.w("Sign2Text", "words_en.txt missing or invalid — word matching disabled", e)
            null
        }
        runOnUiThread {
            if (wordModeEnabled) {
                wordMatchText.text = ""
            }
        }

        previewView.post { startCamera() }
    }

    private fun setWordModeEnabled(enabled: Boolean) {
        wordModeEnabled = enabled
        letterAccumulator.clear()
        openPalmConfirm.reset()
        applyWordModeUi()
    }

    private fun applyWordModeUi() {
        wordModePanel.visibility = if (wordModeEnabled) View.VISIBLE else View.GONE
        toggleWordModeButton.setText(
            if (wordModeEnabled) R.string.mode_word_disable else R.string.mode_word_enable
        )
        applyWordToggleAppearance()
        if (!wordModeEnabled) {
            wordMatchText.text = ""
            spellingText.setText(R.string.word_spelling_empty)
        }
    }

    private fun applyWordToggleAppearance() {
        val strokePx = (resources.displayMetrics.density * 1f).toInt().coerceAtLeast(1)
        if (wordModeEnabled) {
            toggleWordModeButton.strokeWidth = 0
            toggleWordModeButton.strokeColor = ColorStateList.valueOf(Color.TRANSPARENT)
            toggleWordModeButton.backgroundTintList =
                ColorStateList.valueOf(ContextCompat.getColor(this, R.color.primary))
            toggleWordModeButton.setTextColor(ContextCompat.getColor(this, R.color.on_primary))
        } else {
            toggleWordModeButton.strokeWidth = strokePx
            toggleWordModeButton.strokeColor =
                ColorStateList.valueOf(ContextCompat.getColor(this, R.color.word_toggle_outline))
            toggleWordModeButton.backgroundTintList = ColorStateList.valueOf(Color.TRANSPARENT)
            toggleWordModeButton.setTextColor(ContextCompat.getColor(this, R.color.primary))
        }
    }

    private fun createHandLandmarker(): HandLandmarker {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.35f)
            .setMinHandPresenceConfidence(0.35f)
            .setMinTrackingConfidence(0.35f)
            .setResultListener { result: HandLandmarkerResult, _: MPImage ->
                onHandResult(result)
            }
            .setErrorListener { e ->
                runOnUiThread {
                    setTopMessage(getString(R.string.err_tracking), true)
                    Log.e("Sign2Text", "HandLandmarker error", e)
                }
            }
            .build()

        return HandLandmarker.createFromOptions(this, options)
    }

    private fun startCamera() {
        val vw = previewView.width
        val vh = previewView.height
        if (vw <= 0 || vh <= 0) {
            previewView.post { startCamera() }
            return
        }

        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val rotation = previewView.display.rotation
                val viewPort = ViewPort.Builder(Rational(vw, vh), rotation)
                    .setScaleType(ViewPort.FILL_CENTER)
                    .build()

                val preview = Preview.Builder()
                    .setTargetRotation(rotation)
                    .build()
                val analysis = ImageAnalysis.Builder()
                    .setTargetRotation(rotation)
                    .setOutputImageRotationEnabled(true)
                    .setTargetResolution(Size(640, 480))
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { ia ->
                        ia.setAnalyzer(executor) { imageProxy ->
                            try {
                                var bmp = imageProxy.toBitmap()
                                if (USE_FRONT_CAMERA) {
                                    val mirrored = bmp.mirrorHorizontal()
                                    bmp.recycle()
                                    bmp = mirrored
                                }
                                val mpImage: MPImage = BitmapImageBuilder(bmp).build()
                                val tsMs = System.currentTimeMillis()
                                handLandmarker?.detectAsync(mpImage, tsMs)
                            } catch (e: Exception) {
                                Log.e("Sign2Text", "Frame analysis failed", e)
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                val cameraSelector = if (USE_FRONT_CAMERA) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }

                val useCaseGroup = UseCaseGroup.Builder()
                    .setViewPort(viewPort)
                    .addUseCase(preview)
                    .addUseCase(analysis)
                    .build()

                preview.setSurfaceProvider(previewView.surfaceProvider)
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, useCaseGroup)

                runOnUiThread {
                    setTopMessage(null, false)
                    hintText.setText(R.string.hint_idle)
                }
            } catch (e: Exception) {
                Log.e("Sign2Text", "Camera bind failed", e)
                runOnUiThread {
                    setTopMessage(getString(R.string.err_camera), true)
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onHandResult(result: HandLandmarkerResult) {
        try {
            if (result.landmarks().isEmpty()) {
                majorityVote.reset()
                displayDebouncer.reset()
                if (wordModeEnabled) {
                    openPalmConfirm.reset()
                }
                runOnUiThread {
                    predLetter.text = "—"
                    hintText.setText(R.string.hint_no_hand)
                }
                return
            }

            val landmarks = result.landmarks()[0]

            val useGeometry = classifier?.inputDim == 83
            val features = HandPreprocessor.toModelFeatures(landmarks, useGeometry)
            featureNormalizer?.normalizeInPlace(features)
            FeatureDiagnostics.maybeLog(features)

            val probs = classifier?.predict(features, calibrationTemperature) ?: return
            val sm = smoother ?: return
            val smoothed = sm.update(probs)
            val (idx, conf) = sm.argmax(smoothed)
            val letter = labelMap?.labelFor(idx) ?: "?"

            val sorted = smoothed.mapIndexed { i, v -> i to v }.sortedByDescending { it.second }
            val second = sorted.getOrNull(1)?.second ?: 0f
            val margin = conf - second
            val entropy = softmaxEntropy(smoothed)
            val top3Line = sorted.take(3).joinToString("  ") { (i, p) ->
                "${labelMap?.labelFor(i) ?: "?"}:${String.format("%.2f", p)}"
            }

            majorityVote.push(idx)
            val stableIdx = majorityVote.stableClass()
            val majorityAgrees = stableIdx != null && stableIdx == idx
            val showLetter = majorityAgrees &&
                conf >= MIN_CONFIDENCE_TO_SHOW &&
                margin >= MIN_MARGIN_TOP1_TOP2 &&
                entropy <= maxEntropyThreshold

            val debounced = displayDebouncer.update(showLetter, letter)

            val finalizedRaw: String? = if (wordModeEnabled) {
                letterAccumulator.onDebouncedLetter(debounced)
                val openPalmThisFrame = OpenPalmDetector.isOpenPalm(landmarks)
                val confirmedByOpenPalm = openPalmConfirm.update(openPalmThisFrame)
                if (confirmedByOpenPalm) letterAccumulator.takeSpellingAndClear() else null
            } else {
                null
            }

            val now = SystemClock.uptimeMillis()
            if (now - lastLogPredictionMs >= 250L) {
                lastLogPredictionMs = now
                Log.d(
                    "Sign2Text",
                    "prediction: $letter  conf=${String.format("%.2f", conf)}  margin=${String.format("%.2f", margin)}  top3=[$top3Line]"
                )
            }
            if (debounced != lastLoggedDisplayedLetter) {
                lastLoggedDisplayedLetter = debounced
                Log.i("Sign2Text", "displayed letter: $debounced  (raw top-1: $letter)")
            }

            runOnUiThread {
                predLetter.text = debounced
                if (wordModeEnabled) {
                    val spell = letterAccumulator.spelling
                    spellingText.text =
                        if (spell.isEmpty()) getString(R.string.word_spelling_empty) else spell
                    if (finalizedRaw != null) {
                        wordMatchText.text = wordResultMessage(finalizedRaw)
                    }
                }
                when (debounced) {
                    "?" -> hintText.setText(R.string.hint_uncertain)
                    else -> hintText.text = ""
                }
            }
        } catch (e: Exception) {
            Log.e("Sign2Text", "onHandResult failed", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handLandmarker?.close()
        classifier?.close()
        executor.shutdown()
    }

    /**
     * Called on [executor] (same thread as [onHandResult]) so it does not race [letterAccumulator].
     */
    private fun finalizeSpelledWord() {
        if (!wordModeEnabled) return
        val raw = letterAccumulator.takeSpellingAndClear()
        runOnUiThread {
            if (!wordModeEnabled) return@runOnUiThread
            wordMatchText.text = wordResultMessage(raw)
            val spell = letterAccumulator.spelling
            spellingText.text =
                if (spell.isEmpty()) getString(R.string.word_spelling_empty) else spell
        }
    }

    private fun wordResultMessage(raw: String): CharSequence {
        if (raw.isEmpty()) return getString(R.string.word_empty)
        val d = wordDictionary
        if (d == null) return getString(R.string.word_no_dict)
        val lower = raw.lowercase()
        return when {
            d.containsWord(lower) -> getString(R.string.word_match, lower)
            d.hasPrefix(lower) -> getString(R.string.word_prefix_ok)
            else -> getString(R.string.word_no_match, raw)
        }
    }

    private fun softmaxEntropy(p: FloatArray): Float {
        var e = 0.0
        for (x in p) {
            if (x > 1e-7f) {
                e -= x * kotlin.math.ln(x.toDouble())
            }
        }
        return e.toFloat()
    }
}
