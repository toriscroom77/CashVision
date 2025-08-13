package com.example.cashvision.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.example.cashvision.ml.Detection

class DetectionOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<Detection> = emptyList()
    private val paint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        textSize = 48f
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        style = Paint.Style.FILL
        textSize = 36f
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
        color = Color.WHITE
    }
    
    private val backgroundPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    /**
     * Update detections to display
     */
    fun updateDetections(newDetections: List<Detection>) {
        detections = newDetections
        invalidate() // Trigger redraw
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        for (detection in detections) {
            drawDetection(canvas, detection)
        }
    }

    /**
     * Draw a single detection
     */
    private fun drawDetection(canvas: Canvas, detection: Detection) {
        val bbox = detection.bbox
        val color = detection.getColor()
        
        // Set paint color
        paint.color = color
        backgroundPaint.color = Color.argb(180, Color.red(color), Color.green(color), Color.blue(color))
        
        // Draw bounding box
        canvas.drawRect(bbox, paint)
        
        // Prepare text
        val text = "${detection.getFormattedDenomination()} (${(detection.confidence * 100).toInt()}%)"
        val textBounds = Rect()
        textPaint.getTextBounds(text, 0, text.length, textBounds)
        
        // Calculate text position
        val textX = bbox.left + 8f
        val textY = bbox.top - 8f
        
        // Draw text background
        val backgroundRect = RectF(
            textX - 4f,
            textY - textBounds.height() - 4f,
            textX + textBounds.width() + 8f,
            textY + 4f
        )
        canvas.drawRoundRect(backgroundRect, 8f, 8f, backgroundPaint)
        
        // Draw text
        canvas.drawText(text, textX, textY - 4f, textPaint)
        
        // Draw confidence indicator
        drawConfidenceIndicator(canvas, detection)
    }

    /**
     * Draw confidence indicator
     */
    private fun drawConfidenceIndicator(canvas: Canvas, detection: Detection) {
        val bbox = detection.bbox
        val confidence = detection.confidence
        
        // Draw confidence bar
        val barWidth = 100f
        val barHeight = 8f
        val barX = bbox.right - barWidth - 8f
        val barY = bbox.bottom - barHeight - 8f
        
        // Background bar
        paint.color = Color.GRAY
        paint.style = Paint.Style.FILL
        canvas.drawRoundRect(
            barX, barY, barX + barWidth, barY + barHeight,
            4f, 4f, paint
        )
        
        // Confidence bar
        paint.color = detection.getColor()
        canvas.drawRoundRect(
            barX, barY, barX + (barWidth * confidence), barY + barHeight,
            4f, 4f, paint
        )
        
        // Reset paint style
        paint.style = Paint.Style.STROKE
    }

    /**
     * Clear all detections
     */
    fun clearDetections() {
        detections = emptyList()
        invalidate()
    }
}
