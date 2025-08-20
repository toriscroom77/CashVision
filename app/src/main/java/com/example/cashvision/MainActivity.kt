package com.example.cashvision

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import android.view.View
import android.view.animation.AnimationUtils
import android.graphics.Bitmap
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.cashvision.camera.ImageAnalyzer
import com.example.cashvision.ml.Detection
import com.example.cashvision.ml.YoloDetector
import com.example.cashvision.ui.DetectionOverlay
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.request.*
import io.ktor.client.request.forms.*
import io.ktor.http.*
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var btnFlash: Button
    private lateinit var btnDetection: Button
    private lateinit var btnCorrect: Button
    private lateinit var statusText: TextView
    private lateinit var statusIndicator: View
    private lateinit var detectionOverlay: DetectionOverlay

    private var camera: Camera? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var yoloDetector: YoloDetector? = null

    private var isFlashOn = false
    private var isDetectionActive = false

    private val httpClient by lazy {
        HttpClient(CIO)
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else finish()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initializeViews()
        setupClickListeners()
        initializeYoloDetector()
        requestCameraPermission()
    }

    private fun initializeViews() {
        previewView = findViewById(R.id.previewView)
        btnFlash = findViewById(R.id.btnFlash)
        btnDetection = findViewById(R.id.btnDetection)
        btnCorrect = findViewById(R.id.btnCorrect)
        statusText = findViewById(R.id.statusText)
        statusIndicator = findViewById(R.id.statusIndicator)
        detectionOverlay = findViewById(R.id.detectionOverlay)

        // Initialize UI state
        updateFlashButtonState(false)
        updateDetectionButtonState(false)
        updateStatusIndicator(false)
    }

    private fun initializeYoloDetector() {
        statusText.text = getString(R.string.status_model_loading)

        lifecycleScope.launch {
            try {
                yoloDetector = YoloDetector(this@MainActivity)
                val success = yoloDetector?.initialize() ?: false

                if (success) {
                    statusText.text = getString(R.string.status_camera_ready)
                } else {
                    statusText.text = getString(R.string.status_model_error)
                }
            } catch (e: Exception) {
                statusText.text = getString(R.string.status_model_error)
                e.printStackTrace()
            }
        }
    }

    private fun setupClickListeners() {
        btnFlash.setOnClickListener {
            animateButtonPress(btnFlash)
            toggleFlash()
        }

        btnDetection.setOnClickListener {
            animateButtonPress(btnDetection)
            toggleDetection()
        }

        btnCorrect.setOnClickListener {
            handleCorrection()
        }
    }

    private fun animateButtonPress(button: Button) {
        val pressAnim = AnimationUtils.loadAnimation(this, R.anim.button_press)
        val releaseAnim = AnimationUtils.loadAnimation(this, R.anim.button_release)

        button.startAnimation(pressAnim)
        button.postDelayed({
            button.startAnimation(releaseAnim)
        }, 150)
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                // Setup image analysis for YOLO detection
                setupImageAnalysis()

                val selector = CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    this, selector, preview, imageAnalyzer
                )

                // Update UI when camera is ready
                updateStatusIndicator(true)
                if (yoloDetector != null) {
                    statusText.text = getString(R.string.status_camera_ready)
                }

            } catch (exc: Exception) {
                statusText.text = getString(R.string.status_camera_error)
                updateStatusIndicator(false)
                exc.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun setupImageAnalysis() {
        val detector = yoloDetector ?: return

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(Size(1280, 720))  // Mejor resolución para detección
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { analyzer ->
                analyzer.setAnalyzer(
                    ContextCompat.getMainExecutor(this),
                    ImageAnalyzer(detector) { detections, imageWidth, imageHeight ->
                        onDetectionResult(detections, imageWidth, imageHeight)
                    }
                )
            }
    }

    private fun toggleFlash() {
        val info = camera?.cameraInfo ?: return
        val control = camera?.cameraControl ?: return

        if (info.hasFlashUnit()) {
            isFlashOn = !isFlashOn
            control.enableTorch(isFlashOn)
            updateFlashButtonState(isFlashOn)
            updateStatusText(isFlashOn)
        } else {
            statusText.text = getString(R.string.status_no_flashlight)
        }
    }

    private fun toggleDetection() {
        isDetectionActive = !isDetectionActive
        updateDetectionButtonState(isDetectionActive)

        if (isDetectionActive) {
            statusText.text = getString(R.string.status_detection_on)
        } else {
            statusText.text = getString(R.string.status_detection_off)
            detectionOverlay.clearDetections()
        }
    }

    private fun onDetectionResult(detections: List<Detection>, imageWidth: Int, imageHeight: Int) {
        if (isDetectionActive) {
            Log.d("MainActivity", "Received ${detections.size} detections from model")

            // Escalar las coordenadas de la imagen al tamaño de la vista
            val scaledDetections = scaleDetectionsToView(detections, imageWidth, imageHeight)

            // Log de todas las detecciones para debugging
            scaledDetections.forEachIndexed { index, detection ->
                Log.d("MainActivity", "Detection $index: ${detection.className} confidence=${detection.confidence} bbox=${detection.bbox}")
            }

            // Filtrar detecciones con confianza razonable
            val validDetections = scaledDetections.filter { it.confidence >= 0.4f }

            Log.d("MainActivity", "Valid detections (>=40%): ${validDetections.size}")

            // Si hay detecciones válidas, mostrar solo la mejor
            val detectionsToShow = if (validDetections.isNotEmpty()) {
                listOf(validDetections.maxByOrNull { it.confidence }!!)
            } else if (scaledDetections.isNotEmpty()) {
                // Mostrar la mejor detección aunque tenga baja confianza para debugging
                Log.d("MainActivity", "No high-confidence detections, showing best available")
                listOf(scaledDetections.maxByOrNull { it.confidence }!!)
            } else {
                Log.d("MainActivity", "No detections to show")
                emptyList()
            }

            detectionOverlay.updateDetections(detectionsToShow)

            if (detectionsToShow.isNotEmpty()) {
                val bestDetection = detectionsToShow.first()
                statusText.text = "Detectado: ${bestDetection.getFormattedDenomination()} (${(bestDetection.confidence * 100).toInt()}%)"
                btnCorrect.visibility = View.VISIBLE
                Log.d("MainActivity", "Showing detection: ${bestDetection.className} with ${(bestDetection.confidence * 100).toInt()}%")
            } else {
                statusText.text = getString(R.string.status_detecting)
                btnCorrect.visibility = View.GONE
            }
        } else {
            btnCorrect.visibility = View.GONE
        }
    }

    private fun scaleDetectionsToView(detections: List<Detection>, imageWidth: Int, imageHeight: Int): List<Detection> {
        val viewWidth = previewView.width.toFloat()
        val viewHeight = previewView.height.toFloat()

        if (viewWidth <= 0 || viewHeight <= 0) return detections

        val scaleX = viewWidth / imageWidth
        val scaleY = viewHeight / imageHeight

        return detections.map { detection ->
            val scaledBbox = android.graphics.RectF(
                detection.bbox.left * scaleX,
                detection.bbox.top * scaleY,
                detection.bbox.right * scaleX,
                detection.bbox.bottom * scaleY
            )

            detection.copy(bbox = scaledBbox)
        }
    }
    
        private fun handleCorrection() {
            val imageBitmap = previewView.bitmap
            if (imageBitmap == null) {
                Log.e("MainActivity", "No se pudo obtener el bitmap de la vista previa.")
                return
            }
    
            // Guardar la imagen y mostrar el diálogo de selección
            Log.d("MainActivity", "Bitmap capturado con éxito. Mostrando diálogo de corrección.")
            showCorrectionDialog(imageBitmap)
        }
    
        private fun showCorrectionDialog(image: Bitmap) {
            val denominations = arrayOf("1000 Pesos", "2000 Pesos", "5000 Pesos", "10000 Pesos", "20000 Pesos") // Idealmente, esto vendría de una fuente de datos
            val builder = AlertDialog.Builder(this)
            builder.setTitle("Selecciona la denominación correcta")
            builder.setItems(denominations) { dialog, which ->
                val selectedDenomination = denominations[which]
                Log.d("MainActivity", "Corrección: El usuario seleccionó '$selectedDenomination'")
    
                uploadCorrection(image, selectedDenomination)
                dialog.dismiss()
            }
            builder.setNegativeButton("Cancelar") { dialog, _ ->
                dialog.dismiss()
            }
            builder.create().show()
        }

    private fun uploadCorrection(image: Bitmap, label: String) {
        lifecycleScope.launch {
            val stream = ByteArrayOutputStream()
            image.compress(Bitmap.CompressFormat.JPEG, 90, stream)
            val byteArray = stream.toByteArray()

            try {
                httpClient.post("https://cashvision-backend.free.beeceptor.com/correct") {
                    setBody(MultiPartFormDataContent(
                        formData {
                            append("label", label)
                            append("image", byteArray, Headers.build {
                                append(HttpHeaders.ContentType, "image/jpeg")
                                append(HttpHeaders.ContentDisposition, "filename=\"correction.jpg\"")
                            })
                        }
                    ))
                }
                Log.d("MainActivity", "Corrección enviada al servidor para: $label")
                Toast.makeText(this@MainActivity, "Corrección enviada", Toast.LENGTH_SHORT).show()

            } catch (e: Exception) {
                Log.e("MainActivity", "Error al enviar la corrección", e)
                Toast.makeText(this@MainActivity, "Error de red", Toast.LENGTH_SHORT).show()
                e.printStackTrace()
            }
        }
    }

    private fun updateFlashButtonState(isOn: Boolean) {
        btnFlash.isSelected = isOn
        btnFlash.text = getString(if (isOn) R.string.btn_flashlight_off else R.string.btn_flashlight)
    }

    private fun updateDetectionButtonState(isOn: Boolean) {
        btnDetection.isSelected = isOn
        btnDetection.text = getString(if (isOn) R.string.btn_detection_off else R.string.btn_detection)
    }

    private fun updateStatusIndicator(isActive: Boolean) {
        statusIndicator.isSelected = isActive
    }

    private fun updateStatusText(flashOn: Boolean) {
        statusText.text = getString(
            if (flashOn) R.string.status_flashlight_on
            else R.string.status_flashlight_off
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        yoloDetector?.close()
        httpClient.close()
    }
}
