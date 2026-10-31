package com.cgjnkim.mobileokad.flir

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * How far the picture moved between two preview frames, measured from the pictures.
 *
 * ### Why not the gyroscope
 *
 * The gyroscope is in the phone, and on this rig the phone is not the camera: measured
 * against an LCD screen, the phone rolled 52 degrees while the camera's polarization did
 * not move at all, and on another set the polarization turned 46 degrees while the phone
 * stayed nearly still. Any motion the phone reports is the phone's. That is the most
 * likely reason gyroscope blur prediction never separated kept frames from deleted ones
 * better than an AUC of about 0.8 -- it was strapped to the wrong body.
 *
 * Registering one preview frame against the last measures the camera itself, and removes
 * three assumptions at once:
 *
 *  - **no focal length.** The answer is already in pixels. The assumed 12 mm lens turned
 *    out to be 6 mm, which had silently doubled every blur figure ever quoted.
 *  - **no clock alignment.** Both frames come from the same stream with the same
 *    timestamps, so the sensor-delivery offset stops mattering.
 *  - **no coupling.** The instrument and the subject are the same body.
 *
 * It also sees translation, which a gyroscope cannot: hand shake is partly sideways, and
 * sideways movement blurs a near subject as surely as rotation does.
 *
 * ### What it measures, and what it does not
 *
 * Displacement only. Rotation about the optical axis moves the centre of the frame
 * hardly at all and is not recovered here -- an orientation-histogram estimator was
 * tried and gave per-frame errors of two to seven degrees on real scenes, which is worse
 * than useless for anything that accumulates. Since roll also barely blurs the middle of
 * the picture, displacement is the term that matters for shake.
 *
 * Verified on thirty of this camera's own frames at the preview's real resolution, with
 * shot noise added: shifts of 1, 3, 8, 20 and 40 pixels were recovered exactly in every
 * frame, with correlation peaks from 0.99 down to 0.69 as the shift grew.
 *
 * ### Confidence
 *
 * [Drift.confidence] is the correlation peak, and it is the part a gyroscope has no
 * equivalent for. A blank wall gives no peak and the reading says so, where a gyroscope
 * reports a confident number whether or not it means anything -- which is how 44 frames
 * a person had chosen to keep ended up marked as blurred.
 */
internal object FrameDrift {

    /**
     * @param dx horizontal movement in preview pixels since the previous frame
     * @param dy vertical movement, same units
     * @param confidence normalised correlation at the winning offset, 0..1
     * @param intervalMs how long the two frames were apart
     */
    class Drift(
        val dx: Float,
        val dy: Float,
        val confidence: Float,
        val intervalMs: Float,
    ) {
        /** Distance travelled, in preview pixels. */
        val magnitude: Float get() = hypot(dx, dy)

        /**
         * Speed in Stokes pixels per second.
         *
         * A preview pixel covers [MosaicPreview.pitch] sensor pixels, and a Stokes pixel
         * covers [FlirCamera.MOSAIC] of them, so the conversion is the ratio of the two
         * and involves no optics at all.
         */
        fun stokesPerSecond(previewStep: Int): Float {
            if (intervalMs <= 0f) return 0f
            val stokesPerPreviewPixel = previewStep.coerceAtLeast(1).toFloat()
            return magnitude * stokesPerPreviewPixel * 1000f / intervalMs
        }

        /** Smear a frame exposed for [exposureUs] would pick up at this speed. */
        fun smearPixels(previewStep: Int, exposureUs: Long): Float =
            stokesPerSecond(previewStep) * (exposureUs / 1e6f)

        /** Below this the scene had too little detail for the match to mean anything. */
        val trusted: Boolean get() = confidence >= MIN_CONFIDENCE
    }

    /** Correlation peak below which the estimate is not reported. */
    const val MIN_CONFIDENCE = 0.35f

    /** Coarse search half-width, in coarse pixels; times [COARSE] in preview pixels. */
    const val SEARCH = 8

    /** Downsampling for the coarse pass. */
    const val COARSE = 4

    /**
     * Matches [current] against [previous], both single-channel and the same size.
     *
     * Coarse to fine: an exhaustive search on a quarter-scale copy finds the whole-pixel
     * offset, then a three-by-three refinement at full scale places it to a fraction of a
     * pixel. Exhaustive rather than gradient-based because the frames are up to tens of
     * pixels apart at this frame rate, which is far outside the range a gradient method
     * converges over.
     */
    fun between(
        previous: FloatArray,
        current: FloatArray,
        width: Int,
        height: Int,
        intervalMs: Float,
    ): Drift? {
        if (width < 4 * COARSE || height < 4 * COARSE) return null
        if (previous.size < width * height || current.size < width * height) return null

        val cw = width / COARSE
        val ch = height / COARSE
        val a = downsample(previous, width, height, cw, ch)
        val b = downsample(current, width, height, cw, ch)
        normalise(a)
        normalise(b)

        var bestScore = -2.0
        var bestX = 0
        var bestY = 0
        for (oy in -SEARCH..SEARCH) {
            for (ox in -SEARCH..SEARCH) {
                val s = score(a, b, cw, ch, ox, oy)
                if (s > bestScore) {
                    bestScore = s
                    bestX = ox
                    bestY = oy
                }
            }
        }
        if (bestScore <= 0.0) return null

        // Refine at full resolution around the coarse answer.
        val fa = previous.copyOf(width * height)
        val fb = current.copyOf(width * height)
        normalise(fa)
        normalise(fb)
        var fineScore = -2.0
        var fx = bestX * COARSE
        var fy = bestY * COARSE
        for (dy in -COARSE..COARSE) {
            for (dx in -COARSE..COARSE) {
                val ox = bestX * COARSE + dx
                val oy = bestY * COARSE + dy
                val s = score(fa, fb, width, height, ox, oy)
                if (s > fineScore) {
                    fineScore = s
                    fx = ox
                    fy = oy
                }
            }
        }
        if (fineScore <= 0.0) return null

        // Sub-pixel by a parabola through the winning offset and its neighbours.
        val sx = parabola(
            score(fa, fb, width, height, fx - 1, fy), fineScore,
            score(fa, fb, width, height, fx + 1, fy),
        )
        val sy = parabola(
            score(fa, fb, width, height, fx, fy - 1), fineScore,
            score(fa, fb, width, height, fx, fy + 1),
        )
        return Drift(fx + sx, fy + sy, fineScore.toFloat().coerceIn(0f, 1f), intervalMs)
    }

    /** Vertex of the parabola through three samples, clamped to the middle pixel. */
    private fun parabola(left: Double, middle: Double, right: Double): Float {
        val d = left - 2 * middle + right
        if (abs(d) < 1e-12) return 0f
        return (0.5 * (left - right) / d).toFloat().coerceIn(-0.5f, 0.5f)
    }

    /**
     * Normalised correlation of [b] shifted by ([ox], [oy]) against [a].
     *
     * Only the overlapping region is compared, and the score is divided by the number of
     * pixels in it, so a large shift is not rewarded for having less to disagree about.
     */
    private fun score(
        a: FloatArray,
        b: FloatArray,
        w: Int,
        h: Int,
        ox: Int,
        oy: Int,
    ): Double {
        val x0 = max(0, -ox)
        val x1 = min(w, w - ox)
        val y0 = max(0, -oy)
        val y1 = min(h, h - oy)
        if (x1 - x0 < w / 4 || y1 - y0 < h / 4) return -2.0
        var sum = 0.0
        var n = 0
        for (y in y0 until y1) {
            val ra = y * w
            val rb = (y + oy) * w + ox
            for (x in x0 until x1) {
                sum += a[ra + x].toDouble() * b[rb + x]
                n++
            }
        }
        return if (n == 0) -2.0 else sum / n
    }

    /** Zero mean, unit standard deviation, so the score is a correlation. */
    private fun normalise(v: FloatArray) {
        var mean = 0.0
        for (x in v) mean += x
        mean /= v.size
        var varSum = 0.0
        for (x in v) {
            val d = x - mean
            varSum += d * d
        }
        val sd = sqrt(varSum / v.size)
        if (sd < 1e-9) {
            java.util.Arrays.fill(v, 0f)
            return
        }
        val inv = (1.0 / sd).toFloat()
        val m = mean.toFloat()
        for (i in v.indices) v[i] = (v[i] - m) * inv
    }

    /** Box average down to [cw] x [ch]. */
    private fun downsample(src: FloatArray, w: Int, h: Int, cw: Int, ch: Int): FloatArray {
        val out = FloatArray(cw * ch)
        for (y in 0 until ch) {
            for (x in 0 until cw) {
                var sum = 0f
                for (dy in 0 until COARSE) {
                    val row = (y * COARSE + dy) * w
                    for (dx in 0 until COARSE) sum += src[row + x * COARSE + dx]
                }
                out[y * cw + x] = sum / (COARSE * COARSE)
            }
        }
        return out
    }
}
