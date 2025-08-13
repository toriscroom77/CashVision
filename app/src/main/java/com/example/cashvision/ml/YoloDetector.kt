package com.example.cashvision.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class YoloDetector(private val context: Context) {
    
    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment? = null
    
    // Model configuration
    private val inputSize = 640
    private val confidenceThreshold = 0.3f  // Reducido para detectar billetes reales
    private val iouThreshold = 0.4f  // Ajustado para mejor NMS
    
    // Class names - adjust according to your model (solo 5 clases)
    private val classNames = arrayOf(
        "billete_1000",
        "billete_2000",
        "billete_5000",
        "billete_10000",
        "billete_20000"
    )
    
    companion object {
        private const val TAG = "YoloDetector"
    }
    
    /**
     * Initialize the ONNX model
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            val modelBytes = context.assets.open("yolo_model.onnx").use { inputStream ->
                inputStream.readBytes()
            }
            
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.addCPU(false) // Use CPU
            
            ortSession = ortEnvironment?.createSession(modelBytes, sessionOptions)
            
            Log.d(TAG, "YOLO model initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize YOLO model", e)
            false
        }
    }
    
    /**
     * Run detection on a bitmap
     */
    suspend fun detect(bitmap: Bitmap): List<Detection> = withContext(Dispatchers.Default) {
        val session = ortSession ?: return@withContext emptyList()
        
        try {
            // Preprocess image
            val preprocessedData = preprocessImage(bitmap)
            
            // Create input tensor
            val inputName = session.inputNames.iterator().next()
            val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, preprocessedData, shape)
            
            // Run inference
            val inputs = mapOf(inputName to inputTensor)
            val outputs = session.run(inputs)

            // Debug output info
            Log.d(TAG, "Model outputs count: ${outputs.size()}")
            for (i in 0 until outputs.size()) {
                val output = outputs[i]
                Log.d(TAG, "Output $i info: ${output.info}")
                Log.d(TAG, "Output $i type: ${output.value::class.java.simpleName}")
            }
            
            // Process outputs - Primero vamos a inspeccionar la estructura
            val outputValue = outputs[0].value
            Log.d(TAG, "Output type: ${outputValue::class.java.simpleName}")

            // Intentar diferentes formatos de salida
            val detections = when (outputValue) {
                is Array<*> -> {
                    Log.d(TAG, "Output is Array, size: ${outputValue.size}")
                    if (outputValue.isNotEmpty()) {
                        val firstElement = outputValue[0]
                        Log.d(TAG, "First element type: ${firstElement?.javaClass?.simpleName ?: "null"}")
                        when (firstElement) {
                            is Array<*> -> {
                                Log.d(TAG, "Nested array, size: ${firstElement.size}")
                                if (firstElement.isNotEmpty()) {
                                    Log.d(TAG, "Second level type: ${firstElement[0]?.javaClass?.simpleName ?: "null"}")
                                }
                                postprocessOutput(outputValue as Array<Array<FloatArray>>, bitmap.width, bitmap.height)
                            }
                            is FloatArray -> {
                                Log.d(TAG, "FloatArray, size: ${firstElement.size}")
                                postprocessOutputFlat(outputValue as Array<FloatArray>, bitmap.width, bitmap.height)
                            }
                            else -> {
                                Log.e(TAG, "Unexpected output format")
                                emptyList()
                            }
                        }
                    } else {
                        emptyList()
                    }
                }
                is FloatArray -> {
                    Log.d(TAG, "Output is FloatArray, size: ${outputValue.size}")
                    postprocessOutputSingle(outputValue, bitmap.width, bitmap.height)
                }
                else -> {
                    Log.e(TAG, "Unknown output format: ${outputValue::class.java.simpleName}")
                    emptyList()
                }
            }
            
            // Clean up
            inputTensor.close()
            outputs.close()
            
            detections
        } catch (e: Exception) {
            Log.e(TAG, "Detection failed", e)
            emptyList()
        }
    }
    
    /**
     * Preprocess image for YOLO input
     */
    private fun preprocessImage(bitmap: Bitmap): FloatBuffer {
        Log.d(TAG, "Preprocessing image: ${bitmap.width}x${bitmap.height} -> ${inputSize}x${inputSize}")

        // Resize bitmap to model input size manteniendo aspect ratio
        val resizedBitmap = resizeBitmapWithPadding(bitmap, inputSize, inputSize)

        val buffer = FloatBuffer.allocate(3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Convert to CHW format (channels first) y normalizar correctamente
        // Orden: R channel completo, luego G channel completo, luego B channel completo
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            buffer.put(r)
        }

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            buffer.put(g)
        }

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val b = (pixel and 0xFF) / 255.0f
            buffer.put(b)
        }

        buffer.rewind()
        resizedBitmap.recycle()
        Log.d(TAG, "Image preprocessing completed")
        return buffer
    }

    /**
     * Resize bitmap - SIMPLIFICADO para evitar problemas de coordenadas
     */
    private fun resizeBitmapWithPadding(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        // Por ahora, usar resize simple sin padding para evitar problemas de coordenadas
        Log.d(TAG, "Resizing bitmap from ${bitmap.width}x${bitmap.height} to ${targetWidth}x${targetHeight}")
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }
    
    /**
     * Post-process YOLO output - formato original
     */
    private fun postprocessOutput(
        output: Array<Array<FloatArray>>,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()

        Log.d(TAG, "Processing nested array output")
        Log.d(TAG, "Output dimensions: ${output.size} x ${output[0].size}")
        if (output[0].isNotEmpty()) {
            Log.d(TAG, "Detection array size: ${output[0][0].size}")
        }

        // Process each detection
        for (detection in output[0]) {
            if (detection.size < 5 + classNames.size) {
                Log.d(TAG, "Skipping detection with insufficient data: ${detection.size}")
                continue
            }

            val centerX = detection[0]
            val centerY = detection[1]
            val width = detection[2]
            val height = detection[3]
            val objectness = detection[4]

            Log.d(TAG, "Raw detection: centerX=$centerX, centerY=$centerY, width=$width, height=$height, objectness=$objectness")

            if (objectness < confidenceThreshold) {
                Log.d(TAG, "Skipping detection with low objectness: $objectness")
                continue
            }

            // Find best class
            var bestClassId = -1
            var bestClassScore = 0f

            for (i in classNames.indices) {
                val classScore = detection[5 + i]
                if (classScore > bestClassScore) {
                    bestClassScore = classScore
                    bestClassId = i
                }
            }

            // Calcular confianza final y aplicar validaciones adicionales
            val finalConfidence = if (bestClassScore > 1.0f) {
                // Si los scores son > 1, probablemente están mal escalados
                Log.d(TAG, "Unusual class score > 1.0: $bestClassScore, using objectness only")
                objectness
            } else {
                objectness * bestClassScore
            }

            // Validaciones básicas para reducir falsos positivos
            if (objectness < 0.1f) {
                Log.d(TAG, "Rejecting detection with very low objectness: $objectness")
                continue
            }

            if (bestClassScore < 0.1f) {
                Log.d(TAG, "Rejecting detection with very low class score: $bestClassScore")
                continue
            }

            Log.d(TAG, "Best class: $bestClassId (${classNames.getOrNull(bestClassId)}), classScore=$bestClassScore, finalConfidence=$finalConfidence")

            if (finalConfidence < confidenceThreshold || bestClassId == -1) {
                Log.d(TAG, "Skipping detection with low final confidence: $finalConfidence")
                continue
            }

            // Determinar si las coordenadas están normalizadas (0-1) o en píxeles
            val isNormalized = centerX <= 1.0f && centerY <= 1.0f && width <= 1.0f && height <= 1.0f

            Log.d(TAG, "Coordinate analysis: centerX=$centerX, centerY=$centerY, width=$width, height=$height")
            Log.d(TAG, "Original image size: ${originalWidth}x${originalHeight}, Input size: ${inputSize}x${inputSize}")
            Log.d(TAG, "Is normalized: $isNormalized")

            val left: Float
            val top: Float
            val right: Float
            val bottom: Float

            if (isNormalized) {
                // Coordenadas normalizadas - convertir a píxeles
                Log.d(TAG, "Using normalized coordinates")
                left = (centerX - width / 2) * originalWidth
                top = (centerY - height / 2) * originalHeight
                right = (centerX + width / 2) * originalWidth
                bottom = (centerY + height / 2) * originalHeight
            } else {
                // Coordenadas ya en píxeles - escalar desde input size
                Log.d(TAG, "Using pixel coordinates, scaling from input size")
                val scaleX = originalWidth.toFloat() / inputSize
                val scaleY = originalHeight.toFloat() / inputSize
                left = (centerX - width / 2) * scaleX
                top = (centerY - height / 2) * scaleY
                right = (centerX + width / 2) * scaleX
                bottom = (centerY + height / 2) * scaleY
            }

            Log.d(TAG, "Calculated bbox: left=$left, top=$top, right=$right, bottom=$bottom")

            val bbox = RectF(
                max(0f, left),
                max(0f, top),
                min(originalWidth.toFloat(), right),
                min(originalHeight.toFloat(), bottom)
            )

            // Validaciones básicas de tamaño
            val bboxWidth = bbox.width()
            val bboxHeight = bbox.height()

            if (bboxWidth < 10f || bboxHeight < 10f) {
                Log.d(TAG, "Skipping detection with too small bbox: ${bboxWidth}x${bboxHeight}")
                continue
            }

            // Validación menos estricta del área
            val imageArea = originalWidth * originalHeight
            val bboxArea = bboxWidth * bboxHeight
            val areaRatio = bboxArea / imageArea

            if (areaRatio > 0.95f) {
                Log.d(TAG, "Skipping detection with too large bbox (${(areaRatio * 100).toInt()}% of image)")
                continue
            }

            Log.d(TAG, "Adding detection: bbox=$bbox, confidence=$finalConfidence")

            detections.add(
                Detection(
                    bbox = bbox,
                    confidence = finalConfidence,
                    classId = bestClassId,
                    className = classNames[bestClassId]
                )
            )
        }

        Log.d(TAG, "Found ${detections.size} valid detections before NMS")

        val finalDetections = applyNMS(detections)
        Log.d(TAG, "Final detections after NMS: ${finalDetections.size}")

        return finalDetections
    }

    /**
     * Post-process YOLO output for flat array format
     */
    private fun postprocessOutputFlat(
        output: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()

        Log.d(TAG, "Processing flat output with ${output.size} arrays")

        // Para formato YOLOv5/v8, cada detección es [x, y, w, h, conf, class_scores...]
        for (i in output.indices) {
            val detection = output[i]
            if (detection.size < 5 + classNames.size) {
                Log.d(TAG, "Skipping detection $i with insufficient data: ${detection.size}")
                continue
            }

            val centerX = detection[0]
            val centerY = detection[1]
            val width = detection[2]
            val height = detection[3]
            val objectness = detection[4]

            Log.d(TAG, "Detection $i: centerX=$centerX, centerY=$centerY, width=$width, height=$height, objectness=$objectness")

            if (objectness < confidenceThreshold) {
                Log.d(TAG, "Skipping detection $i with low objectness: $objectness")
                continue
            }

            // Find best class
            var bestClassId = -1
            var bestClassScore = 0f

            for (j in classNames.indices) {
                val classScore = detection[5 + j]
                if (classScore > bestClassScore) {
                    bestClassScore = classScore
                    bestClassId = j
                }
            }

            val finalConfidence = objectness * bestClassScore

            if (finalConfidence < confidenceThreshold || bestClassId == -1) {
                Log.d(TAG, "Skipping detection $i with low final confidence: $finalConfidence")
                continue
            }

            // Convert normalized coordinates to pixel coordinates
            val scaleX = originalWidth.toFloat()
            val scaleY = originalHeight.toFloat()

            val left = (centerX - width / 2) * scaleX
            val top = (centerY - height / 2) * scaleY
            val right = (centerX + width / 2) * scaleX
            val bottom = (centerY + height / 2) * scaleY

            val bbox = RectF(
                max(0f, left),
                max(0f, top),
                min(originalWidth.toFloat(), right),
                min(originalHeight.toFloat(), bottom)
            )

            if (bbox.width() < 10f || bbox.height() < 10f) {
                Log.d(TAG, "Skipping detection $i with too small bbox: ${bbox.width()}x${bbox.height()}")
                continue
            }

            detections.add(
                Detection(
                    bbox = bbox,
                    confidence = finalConfidence,
                    classId = bestClassId,
                    className = classNames[bestClassId]
                )
            )
        }

        Log.d(TAG, "Found ${detections.size} valid detections from flat format")
        return applyNMS(detections)
    }

    /**
     * Post-process YOLO output for single array format
     */
    private fun postprocessOutputSingle(
        output: FloatArray,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()

        Log.d(TAG, "Processing single array output with ${output.size} elements")

        // Calcular cuántas detecciones hay
        val elementsPerDetection = 5 + classNames.size
        val numDetections = output.size / elementsPerDetection

        Log.d(TAG, "Elements per detection: $elementsPerDetection, Number of detections: $numDetections")

        for (i in 0 until numDetections) {
            val startIdx = i * elementsPerDetection

            if (startIdx + elementsPerDetection > output.size) break

            val centerX = output[startIdx]
            val centerY = output[startIdx + 1]
            val width = output[startIdx + 2]
            val height = output[startIdx + 3]
            val objectness = output[startIdx + 4]

            Log.d(TAG, "Detection $i: centerX=$centerX, centerY=$centerY, width=$width, height=$height, objectness=$objectness")

            if (objectness < confidenceThreshold) {
                Log.d(TAG, "Skipping detection $i with low objectness: $objectness")
                continue
            }

            // Find best class
            var bestClassId = -1
            var bestClassScore = 0f

            for (j in classNames.indices) {
                val classScore = output[startIdx + 5 + j]
                if (classScore > bestClassScore) {
                    bestClassScore = classScore
                    bestClassId = j
                }
            }

            val finalConfidence = objectness * bestClassScore

            if (finalConfidence < confidenceThreshold || bestClassId == -1) {
                Log.d(TAG, "Skipping detection $i with low final confidence: $finalConfidence")
                continue
            }

            // Convert normalized coordinates to pixel coordinates
            val scaleX = originalWidth.toFloat()
            val scaleY = originalHeight.toFloat()

            val left = (centerX - width / 2) * scaleX
            val top = (centerY - height / 2) * scaleY
            val right = (centerX + width / 2) * scaleX
            val bottom = (centerY + height / 2) * scaleY

            val bbox = RectF(
                max(0f, left),
                max(0f, top),
                min(originalWidth.toFloat(), right),
                min(originalHeight.toFloat(), bottom)
            )

            if (bbox.width() < 10f || bbox.height() < 10f) {
                Log.d(TAG, "Skipping detection $i with too small bbox: ${bbox.width()}x${bbox.height()}")
                continue
            }

            detections.add(
                Detection(
                    bbox = bbox,
                    confidence = finalConfidence,
                    classId = bestClassId,
                    className = classNames[bestClassId]
                )
            )
        }

        Log.d(TAG, "Found ${detections.size} valid detections from single array format")
        return applyNMS(detections)
    }

    /**
     * Apply Non-Maximum Suppression - Mejorado para reducir detecciones múltiples
     */
    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        Log.d(TAG, "Applying NMS to ${detections.size} detections")

        // Ordenar por confianza descendente
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val result = mutableListOf<Detection>()

        // Aplicar NMS por clase para evitar detecciones múltiples del mismo tipo
        for (classId in classNames.indices) {
            val classDetections = sortedDetections.filter { it.classId == classId }
            if (classDetections.isEmpty()) continue

            Log.d(TAG, "Processing ${classDetections.size} detections for class ${classNames[classId]}")

            val classResult = mutableListOf<Detection>()

            for (detection in classDetections) {
                var shouldKeep = true

                // Verificar solapamiento con detecciones ya aceptadas de esta clase
                for (kept in classResult) {
                    val iou = calculateIoU(detection.bbox, kept.bbox)
                    if (iou > iouThreshold) {
                        Log.d(TAG, "Suppressing detection with IoU $iou > $iouThreshold")
                        shouldKeep = false
                        break
                    }
                }

                // También verificar solapamiento con otras clases (más permisivo)
                for (kept in result) {
                    val iou = calculateIoU(detection.bbox, kept.bbox)
                    if (iou > 0.5f) { // Umbral más alto para diferentes clases
                        Log.d(TAG, "Suppressing detection due to overlap with different class, IoU: $iou")
                        shouldKeep = false
                        break
                    }
                }

                if (shouldKeep) {
                    classResult.add(detection)
                    Log.d(TAG, "Keeping detection: ${detection.className} with confidence ${detection.confidence}")
                }
            }

            // Solo mantener la mejor detección por clase si hay múltiples
            if (classResult.size > 1) {
                Log.d(TAG, "Multiple detections for ${classNames[classId]}, keeping only the best one")
                result.add(classResult.first()) // Ya está ordenado por confianza
            } else {
                result.addAll(classResult)
            }
        }

        Log.d(TAG, "NMS result: ${result.size} detections kept")
        return result
    }
    
    /**
     * Calculate Intersection over Union
     */
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = max(box1.left, box2.left)
        val intersectionTop = max(box1.top, box2.top)
        val intersectionRight = min(box1.right, box2.right)
        val intersectionBottom = min(box1.bottom, box2.bottom)
        
        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0f
        }
        
        val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val unionArea = box1Area + box2Area - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
    }
}
