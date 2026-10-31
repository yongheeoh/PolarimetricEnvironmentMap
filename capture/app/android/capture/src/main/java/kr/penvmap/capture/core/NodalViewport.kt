package kr.penvmap.capture.core

import kotlin.math.min

/** Display-only transforms. Alignment translates the live image in preview pixels, never the reference. */
class NodalViewport {
    var imageWidth = 0
        private set
    var imageHeight = 0
        private set
    var viewWidth = 0
        private set
    var viewHeight = 0
        private set
    var zoom = 1f
        private set
    var panX = 0f
        private set
    var panY = 0f
        private set
    var liveX = 0f
        private set
    var liveY = 0f
        private set

    val scale: Float get() = if (imageWidth > 0 && imageHeight > 0 && viewWidth > 0 && viewHeight > 0)
        min(viewWidth.toFloat() / imageWidth, viewHeight.toFloat() / imageHeight) * zoom else 1f
    val left: Float get() = (viewWidth - imageWidth * scale) / 2f + panX
    val top: Float get() = (viewHeight - imageHeight * scale) / 2f + panY

    fun imageSize(w: Int, h: Int) {
        if (w == imageWidth && h == imageHeight) return
        imageWidth = w
        imageHeight = h
        fit()
        resetAlignment()
    }

    fun viewSize(w: Int, h: Int) {
        if (w == viewWidth && h == viewHeight) return
        viewWidth = w
        viewHeight = h
        fit()
    }

    fun fit() { zoom = 1f; panX = 0f; panY = 0f }
    fun resetAlignment() { liveX = 0f; liveY = 0f }
    fun pan(dx: Float, dy: Float) { panX += dx; panY += dy }
    fun alignLive(dx: Float, dy: Float) { liveX += dx / scale; liveY += dy / scale }

    /** Keep the image point under the two fingers stationary, including at the zoom limits. */
    fun zoomAt(factor: Float, x: Float, y: Float) {
        if (!factor.isFinite() || factor <= 0f) return
        val ix = (x - left) / scale
        val iy = (y - top) / scale
        zoom = (zoom * factor).coerceIn(1f, 12f)
        panX = x - (viewWidth - imageWidth * scale) / 2f - ix * scale
        panY = y - (viewHeight - imageHeight * scale) / 2f - iy * scale
    }
}
