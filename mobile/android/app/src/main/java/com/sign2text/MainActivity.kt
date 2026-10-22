package com.sign2text

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Rational
import android.util.Size
import android.content.res.ColorStateList
import android.graphics.Color
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.button.MaterialButton
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
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
import java.util.concurrent.TimeUnit


@androidx.camera.camera2.interop.ExperimentalCamera2Interop
class MainActivity : ComponentActivity() {
    private companion object {
        /** Below this, the frame is too unsure to show a letter. */
        const val MIN_CONFIDENCE_TO_SHOW = 0.60f
        /** Top letter must lead the runner-up by this much, or the screen stays blank. */
        const val MIN_MARGIN_TOP1_TOP2 = 0.18f
        const val INDEX_TIP = 8
    }

    private lateinit var previewView: PreviewView
    private lateinit var cameraTapTarget: View
    private lateinit var predLetter: TextView
    private lateinit var spellingText: TextView
    private lateinit var wordMatchText: TextView
    private lateinit var savedResultText: TextView
    private lateinit var hintText: TextView
    private lateinit var wordFormationHint: TextView
    private lateinit var topMessage: TextView
    private lateinit var predictionCard: LinearLayout
    private lateinit var wordPeekRow: LinearLayout
    private lateinit var sheetExpandedDetails: LinearLayout
    private lateinit var sheetSwipeHint: TextView
    private lateinit var lettersOnlyHint: TextView
    private lateinit var sheetHandleArea: LinearLayout
    private lateinit var toggleWordModeButton: MaterialButton
    private lateinit var saveResultButton: MaterialButton
    private lateinit var clearResultButton: MaterialButton
    private lateinit var backspaceButton: MaterialButton
    private lateinit var bottomSheetBehavior: BottomSheetBehavior<LinearLayout>

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
    private val displayDebouncer = LetterDisplayDebouncer(framesToConfirm = 2, framesToClear = 3)
    private val zStrokeDetector = ZStrokeDetector()
    private val letterAccumulator = LetterSequenceAccumulator(framesToRecord = 6)
    private val openPalmConfirm = OpenPalmConfirmDetector()
    private var wordDictionary: WordDictionary? = null

    private var lastLogPredictionMs = 0L
    private var lastLoggedDisplayedLetter: String? = null

    /** Front camera mirrors analysis to match the selfie preview. Back camera stays as the sensor sees it. */
    @Volatile
    private var useFrontCamera = false

    private var cameraSwitchInFlight = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        cameraTapTarget = findViewById(R.id.cameraTapTarget)
        predLetter = findViewById(R.id.predLetter)
        predictionCard = findViewById(R.id.predictionCard)
        wordPeekRow = findViewById(R.id.wordPeekRow)
        sheetExpandedDetails = findViewById(R.id.sheetExpandedDetails)
        sheetSwipeHint = findViewById(R.id.sheetSwipeHint)
        lettersOnlyHint = findViewById(R.id.lettersOnlyHint)
        sheetHandleArea = findViewById(R.id.sheetHandleArea)
        spellingText = findViewById(R.id.spellingText)
        wordMatchText = findViewById(R.id.wordMatchText)
        savedResultText = findViewById(R.id.savedResultText)
        hintText = findViewById(R.id.hintText)
        wordFormationHint = findViewById(R.id.wordFormationHint)
        topMessage = findViewById(R.id.topMessage)
        toggleWordModeButton = findViewById(R.id.toggleWordModeButton)
        saveResultButton = findViewById(R.id.saveResultButton)
        clearResultButton = findViewById(R.id.clearResultButton)
        backspaceButton = findViewById(R.id.backspaceButton)

        setupBottomSheet()
        setupCameraFlip()

        toggleWordModeButton.setOnClickListener {
            setWordModeEnabled(!wordModeEnabled)
        }
        saveResultButton.setOnClickListener {
            executor.execute { saveSpelledResult() }
        }
        clearResultButton.setOnClickListener {
            executor.execute { clearSpellingAndResult() }
        }
        backspaceButton.setOnClickListener {
            executor.execute { backspaceLastLetter() }
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
        setLiveHint(R.string.hint_idle)

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
        if (cl.inputDim != 63 && fn == null) {
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
        smoother = Smoother(numClasses = nClass, alpha = 0.70f)
        maxEntropyThreshold =
            if (nClass <= 1) 0f
            else (kotlin.math.ln(nClass.toDouble()) * 0.75).toFloat()

        try {
            handLandmarker = createHandLandmarker()
        } catch (t: Throwable) {
            Log.e("Sign2Text", "HandLandmarker load failed", t)
            setTopMessage(getString(R.string.err_setup), true)
            predLetter.text = "—"
            handLandmarker = null
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

    private fun setupBottomSheet() {
        bottomSheetBehavior = BottomSheetBehavior.from(predictionCard)
        bottomSheetBehavior.isFitToContents = true
        bottomSheetBehavior.isHideable = false
        bottomSheetBehavior.skipCollapsed = false
        bottomSheetBehavior.isDraggable = true
        bottomSheetBehavior.addBottomSheetCallback(object : BottomSheetBehavior.BottomSheetCallback() {
            override fun onStateChanged(bottomSheet: View, newState: Int) {
                updateSheetChromeForState(newState)
            }

            override fun onSlide(bottomSheet: View, slideOffset: Float) = Unit
        })
        sheetHandleArea.setOnClickListener {
            if (bottomSheetBehavior.state == BottomSheetBehavior.STATE_EXPANDED) {
                bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
            } else {
                bottomSheetBehavior.state = BottomSheetBehavior.STATE_EXPANDED
            }
        }
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
        updateSheetChromeForState(BottomSheetBehavior.STATE_COLLAPSED)
    }

    private fun updateSheetChromeForState(state: Int) {
        val expanded = state == BottomSheetBehavior.STATE_EXPANDED
        if (wordModeEnabled) {
            // Keep details laid out so peekHeight clips them; swipe up reveals Save/Clear.
            sheetExpandedDetails.visibility = View.VISIBLE
            sheetSwipeHint.visibility = View.VISIBLE
            sheetSwipeHint.setText(
                if (expanded) R.string.sheet_swipe_down else R.string.sheet_swipe_up
            )
        } else {
            sheetExpandedDetails.visibility = View.GONE
            sheetSwipeHint.visibility = View.GONE
        }
    }

    private fun setWordModeEnabled(enabled: Boolean) {
        wordModeEnabled = enabled
        letterAccumulator.clear()
        openPalmConfirm.reset()
        applyWordModeUi()
        // Spell mode starts collapsed so the camera stays large; swipe up for Save/Clear.
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
        updateSheetChromeForState(BottomSheetBehavior.STATE_COLLAPSED)
    }

    private fun applyWordModeUi() {
        val peek = resources.getDimensionPixelSize(
            if (wordModeEnabled) R.dimen.sheet_peek_words else R.dimen.sheet_peek_letters
        )
        bottomSheetBehavior.peekHeight = peek
        wordPeekRow.visibility = if (wordModeEnabled) View.VISIBLE else View.GONE
        lettersOnlyHint.visibility = if (wordModeEnabled) View.GONE else View.VISIBLE
        toggleWordModeButton.setText(
            if (wordModeEnabled) R.string.mode_word_disable else R.string.mode_word_enable
        )
        applyWordToggleAppearance()
        if (!wordModeEnabled) {
            wordMatchText.text = ""
            spellingText.setText(R.string.word_spelling_empty)
            savedResultText.setText(R.string.word_result_empty)
            sheetExpandedDetails.visibility = View.GONE
            sheetSwipeHint.visibility = View.GONE
        } else {
            updateSheetChromeForState(bottomSheetBehavior.state)
        }
    }

    /** Expand sheet so Save/Clear are reachable after a successful save via open palm. */
    private fun expandSheetForActions() {
        if (!wordModeEnabled) return
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_EXPANDED
        updateSheetChromeForState(BottomSheetBehavior.STATE_EXPANDED)
    }

    private fun applyWordToggleAppearance() {
        val strokePx = (resources.displayMetrics.density * 1.5f).toInt().coerceAtLeast(1)
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
            .setMinHandDetectionConfidence(0.20f)
            .setMinHandPresenceConfidence(0.20f)
            .setMinTrackingConfidence(0.20f)
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

    private fun setupCameraFlip() {
        val detector = GestureDetector(
            this,
            object : GestureDetector.SimpleOnGestureListener() {
                override fun onDown(e: MotionEvent): Boolean = true

                override fun onDoubleTap(e: MotionEvent): Boolean {
                    flipCamera()
                    return true
                }
            }
        )
        cameraTapTarget.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                Log.i("Sign2Text", "camera tap down")
            }
            detector.onTouchEvent(event)
            true
        }
    }

    private fun flipCamera() {
        if (cameraSwitchInFlight) return
        cameraSwitchInFlight = true
        useFrontCamera = !useFrontCamera
        Log.i("Sign2Text", "flipCamera front=$useFrontCamera")
        cameraTapTarget.contentDescription = getString(
            if (useFrontCamera) R.string.camera_front else R.string.camera_back
        )
        smoother?.reset()
        displayDebouncer.reset()
        zStrokeDetector.reset()
        predLetter.text = "—"
        Toast.makeText(
            this,
            if (useFrontCamera) R.string.camera_front else R.string.camera_back,
            Toast.LENGTH_SHORT
        ).show()
        startCamera()
    }

    private fun startCamera() {
        val vw = previewView.width
        val vh = previewView.height
        if (vw <= 0 || vh <= 0) {
            previewView.post { startCamera() }
            return
        }

        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        previewView.implementationMode = PreviewView.ImplementationMode.PERFORMANCE

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val rotation = previewView.display.rotation
                val viewPort = ViewPort.Builder(Rational(vw, vh), rotation)
                    .setScaleType(ViewPort.FILL_CENTER)
                    .build()

                val highPreview = ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(1920, 1080),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()
                val previewBuilder = Preview.Builder()
                    .setTargetRotation(rotation)
                    .setResolutionSelector(highPreview)
                Camera2Interop.Extender(previewBuilder)
                    .setCaptureRequestOption(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                    )
                    .setCaptureRequestOption(
                        CaptureRequest.CONTROL_AE_MODE,
                        CaptureRequest.CONTROL_AE_MODE_ON
                    )
                val preview = previewBuilder.build()
                val analysisBuilder = ImageAnalysis.Builder()
                    .setTargetRotation(rotation)
                    .setOutputImageRotationEnabled(true)
                    .setTargetResolution(Size(640, 480))
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                Camera2Interop.Extender(analysisBuilder)
                    .setCaptureRequestOption(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                    )
                val analysis = analysisBuilder.build()
                    .also { ia ->
                        ia.setAnalyzer(executor) { imageProxy ->
                            try {
                                var bmp = imageProxy.toBitmap()
                                if (useFrontCamera) {
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

                val cameraSelector = if (useFrontCamera) {
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
                val camera = cameraProvider.bindToLifecycle(this, cameraSelector, useCaseGroup)
                val focusFactory = SurfaceOrientedMeteringPointFactory(vw.toFloat(), vh.toFloat())
                val focusPoint = focusFactory.createPoint(vw / 2f, vh / 2f)
                val focusAction = FocusMeteringAction.Builder(
                    focusPoint,
                    FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
                )
                    .setAutoCancelDuration(2, TimeUnit.SECONDS)
                    .build()
                camera.cameraControl.startFocusAndMetering(focusAction)

                runOnUiThread {
                    setTopMessage(null, false)
                    setLiveHint(R.string.hint_idle)
                }
            } catch (e: Exception) {
                Log.e("Sign2Text", "Camera bind failed", e)
                runOnUiThread {
                    setTopMessage(getString(R.string.err_camera), true)
                }
            } finally {
                cameraSwitchInFlight = false
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onHandResult(result: HandLandmarkerResult) {
        try {
            if (result.landmarks().isEmpty()) {
                displayDebouncer.reset()
                zStrokeDetector.reset()
                if (wordModeEnabled) {
                    openPalmConfirm.reset()
                }
                runOnUiThread {
                    predLetter.text = "—"
                    setLiveHint(R.string.hint_no_hand)
                }
                return
            }

            val landmarks = result.landmarks()[0]
            if (landmarks.size < 21) return

            val modelDim = classifier?.inputDim ?: 63
            val useGeometry = modelDim == HandPreprocessor.MODEL_FEATURE_DIM
            val features = HandPreprocessor.toModelFeatures(landmarks, useGeometry)
            val thumbIndexGap = HandPreprocessor.thumbIndexGap(features)
            featureNormalizer?.normalizeInPlace(features)
            FeatureDiagnostics.maybeLog(features)

            val probs = classifier?.predict(features, calibrationTemperature) ?: return
            val sm = smoother ?: return
            val smoothed = sm.update(probs)
            val (idx, conf) = sm.argmax(smoothed)
            val letter = labelMap?.labelFor(idx) ?: "?"

            val sorted = smoothed.mapIndexed { i, v -> i to v }.sortedByDescending { it.second }
            val secondP = sorted.getOrNull(1)?.second ?: 0f
            val secondLabel = sorted.getOrNull(1)?.let { (i, _) -> labelMap?.labelFor(i) }
            val margin = conf - secondP
            val entropy = softmaxEntropy(smoothed)
            val top3Line = sorted.take(3).joinToString("  ") { (i, p) ->
                "${labelMap?.labelFor(i) ?: "?"}:${String.format("%.2f", p)}"
            }

            val now = SystemClock.uptimeMillis()
            val zDrawn = zStrokeDetector.update(
                now,
                landmarks[INDEX_TIP].x(),
                landmarks[INDEX_TIP].y(),
                indexFingerExtended(landmarks),
                wristToMiddleKnuckle(landmarks),
            )
            val chosen = LetterAcceptance.choose(
                topLabel = letter,
                topP = conf,
                secondLabel = secondLabel,
                secondP = secondP,
                entropy = entropy,
                maxEntropy = maxEntropyThreshold,
                zDrawn = zDrawn,
                minConfidence = MIN_CONFIDENCE_TO_SHOW,
                minMargin = MIN_MARGIN_TOP1_TOP2,
                thumbIndexGap = thumbIndexGap,
            )
            val displayCandidate = chosen ?: letter
            val debounced = displayDebouncer.update(
                showLetter = chosen != null,
                letter = displayCandidate,
                framesNeeded = 3,
            )

            // Spelling only accumulates here. Do NOT auto-save on open palm — many letter
            // poses look "open" and were clearing the word before the user finished.
            // User finishes a word with the Save button (swipe sheet up).
            if (wordModeEnabled) {
                letterAccumulator.onDebouncedLetter(debounced)
            }
            val finalizedRaw: String? = null

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
                        applySavedResult(finalizedRaw)
                        expandSheetForActions()
                    }
                }
                when (debounced) {
                    "?" -> setLiveHint(R.string.hint_uncertain)
                    else -> setLiveHint(null)
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
     * Save path (button or open palm): store spelling, show dictionary match, clear live buffer.
     * Called on [executor] so it does not race [letterAccumulator].
     */
    private fun saveSpelledResult() {
        if (!wordModeEnabled) return
        val raw = letterAccumulator.takeSpellingAndClear()
        openPalmConfirm.reset()
        runOnUiThread {
            if (!wordModeEnabled) return@runOnUiThread
            applySavedResult(raw)
            spellingText.setText(R.string.word_spelling_empty)
            if (raw.isNotEmpty()) expandSheetForActions()
        }
    }

    private fun setLiveHint(resId: Int?) {
        // Keep a short centered prompt under the letter chip in letters-only mode.
        val text = when (resId) {
            null -> getString(R.string.hint_idle)
            else -> getString(resId)
        }
        hintText.text = if (resId == null) "" else text
        lettersOnlyHint.text = text
        lettersOnlyHint.visibility = if (wordModeEnabled) View.GONE else View.VISIBLE
    }

    private fun applySavedResult(raw: String) {
        if (raw.isEmpty()) {
            wordMatchText.text = getString(R.string.word_empty)
            return
        }
        savedResultText.text = raw
        wordMatchText.text = wordResultMessage(raw)
    }

    private fun backspaceLastLetter() {
        if (!wordModeEnabled) return
        letterAccumulator.deleteLastLetter()
        openPalmConfirm.reset()
        runOnUiThread {
            if (!wordModeEnabled) return@runOnUiThread
            val spell = letterAccumulator.spelling
            spellingText.text =
                if (spell.isEmpty()) getString(R.string.word_spelling_empty) else spell
        }
    }

    private fun clearSpellingAndResult() {
        if (!wordModeEnabled) return
        letterAccumulator.clear()
        openPalmConfirm.reset()
        runOnUiThread {
            if (!wordModeEnabled) return@runOnUiThread
            spellingText.setText(R.string.word_spelling_empty)
            savedResultText.setText(R.string.word_result_empty)
            wordMatchText.text = getString(R.string.word_cleared)
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

    private fun indexFingerExtended(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): Boolean {
        val index = distance(landmarks[0], landmarks[INDEX_TIP])
        val middle = distance(landmarks[0], landmarks[12])
        val ring = distance(landmarks[0], landmarks[16])
        return index > middle * 1.15f && index > ring * 1.15f
    }

    private fun wristToMiddleKnuckle(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): Float {
        return distance(landmarks[0], landmarks[9])
    }

    private fun distance(
        a: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
        b: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
    ): Float {
        val dx = a.x() - b.x()
        val dy = a.y() - b.y()
        return kotlin.math.hypot(dx, dy)
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
