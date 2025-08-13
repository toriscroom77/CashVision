package com.example.cashvision.ml

import android.graphics.RectF

/**
 * Represents a detection result from YOLO model
 */
data class Detection(
    val bbox: RectF,           // Bounding box coordinates
    val confidence: Float,     // Confidence score (0-1)
    val classId: Int,         // Class ID
    val className: String     // Class name (e.g., "billete_10000", "billete_20000")
) {
    /**
     * Get the denomination value from class name
     */
    fun getDenomination(): Int {
        return when {
            className.contains("1000") -> 1000
            className.contains("2000") -> 2000
            className.contains("5000") -> 5000
            className.contains("10000") -> 10000
            className.contains("20000") -> 20000
            else -> 0
        }
    }

    /**
     * Get formatted denomination string
     */
    fun getFormattedDenomination(): String {
        val value = getDenomination()
        return if (value > 0) {
            "$${value}"
        } else {
            "Desconocido"
        }
    }

    /**
     * Get color for this denomination
     */
    fun getColor(): Int {
        return when (getDenomination()) {
            1000 -> 0xFF4CAF50.toInt()    // Green
            2000 -> 0xFF2196F3.toInt()    // Blue
            5000 -> 0xFFFF9800.toInt()    // Orange
            10000 -> 0xFFE91E63.toInt()   // Pink
            20000 -> 0xFF9C27B0.toInt()   // Purple
            else -> 0xFFFFFFFF.toInt()    // White
        }
    }
}
