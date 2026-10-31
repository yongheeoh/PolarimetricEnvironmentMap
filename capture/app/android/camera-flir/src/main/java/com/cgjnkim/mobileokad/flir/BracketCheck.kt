package com.cgjnkim.mobileokad.flir

import kotlin.math.abs
import kotlin.math.max

/**
 * Per-frame checks of a sequencer bracket, the same three the PC capture makes
 * (`hardware/sensor/flir.py`): consecutive FrameIDs (no frame lost), the exposure the frame was
 * actually taken with matches the one programmed for its set, and the frame came from the set
 * it should have.
 */
object BracketCheck {

    /** PC tolerance: max(2 µs, 0.1 %). */
    fun exposureTolerance(programmedUs: Double): Double = max(2.0, 1e-3 * programmedUs)

    /** Null when frame [index] is consistent, otherwise the reason (same wording as the PC). */
    fun check(
        index: Int,
        frameId: Long,
        previousFrameId: Long?,
        chunkExposureUs: Double,
        programmedUs: Double,
        chunkSet: Int,
    ): String? {
        if (previousFrameId != null && frameId != previousFrameId + 1) {
            return "frame loss: FrameID $previousFrameId -> $frameId"
        }
        if (abs(chunkExposureUs - programmedUs) > exposureTolerance(programmedUs)) {
            return "exposure mismatch at frame $index: programmed ${programmedUs} us, captured ${chunkExposureUs} us"
        }
        if (chunkSet != index) return "sequencer set mismatch at frame $index: captured set $chunkSet"
        return null
    }

    /** Splits a bracket into consecutive segments of at most [capacity] exposures (PC `segments`). */
    fun <T> segments(exposures: List<T>, capacity: Int): List<List<T>> {
        require(capacity >= 1) { "capacity must be at least 1" }
        return exposures.chunked(capacity)
    }
}
