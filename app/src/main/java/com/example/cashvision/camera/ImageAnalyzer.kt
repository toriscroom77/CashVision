package com.example.cashvision.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.cashvision.ml.Detection
import com.example.cashvision.ml.YoloDetector
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class ImageAnalyzer(
    private val yoloDetector: YoloDetector,
    private val onDetectionResult: (List<Detection>, Int, Int) -> Unit
) : ImageAnalysis.Analyzer {

    private val scope = CoroutineScope(Dispatchers.Default)
    private var isProcessing = false
    private var frameCount = 0

    companion object {
        private const val TAG = "ImageAnalyzer"
        private const val PROCESS_EVERY_N_FRAMES = 10  // Procesar cada 10 frames para debugging
    }

    override fun analyze(image: ImageProxy) {
        frameCount++

        // Skip if already processing or not the right frame
        if (isProcessing || frameCount % PROCESS_EVERY_N_FRAMES != 0) {
            image.close()
            return
        }

        isProcessing = true
        Log.d(TAG, "Processing frame ${frameCount}, image size: ${image.width}x${image.height}")

        scope.launch {
            try {
                val startTime = System.currentTimeMillis()
                val bitmap = imageProxyToBitmap(image)

                if (bitmap != null) {
                    Log.d(TAG, "Bitmap created: ${bitmap.width}x${bitmap.height}")
                    val detections = yoloDetector.detect(bitmap)
                    val processingTime = System.currentTimeMillis() - startTime

                    Log.d(TAG, "Detection completed in ${processingTime}ms, found ${detections.size} detections")

                    // Post results to main thread
                    launch(Dispatchers.Main) {
                        onDetectionResult(detections, bitmap.width, bitmap.height)
                    }

                    bitmap.recycle()
                } else {
                    Log.w(TAG, "Failed to convert ImageProxy to Bitmap")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during image analysis", e)
            } finally {
                isProcessing = false
                image.close()
            }
        }
    }

    /**
     * Convert ImageProxy to Bitmap
     */
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val buffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            
            // For YUV format
            if (image.format == ImageFormat.YUV_420_888) {
                yuvToBitmap(image)
            } else {
                // For JPEG format
                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                rotateBitmap(bitmap, image.imageInfo.rotationDegrees)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * Convert YUV to Bitmap
     */
    private fun yuvToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
            val imageBytes = out.toByteArray()
            
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            rotateBitmap(bitmap, image.imageInfo.rotationDegrees)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * Rotate bitmap according to camera orientation
     */
    private fun rotateBitmap(bitmap: Bitmap?, rotationDegrees: Int): Bitmap? {
        if (bitmap == null || rotationDegrees == 0) return bitmap
        
        return try {
            val matrix = Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: Exception) {
            e.printStackTrace()
            bitmap
        }
    }
}
