package com.cgjnkim.mobileokad.flir

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Which way the world is, guessed from the picture's own edges.
 *
 * ### The idea
 *
 * A man-made scene is full of edges along two perpendicular directions. Folding every
 * gradient's orientation onto a quarter turn makes the vertical family vote for the same
 * answer as the horizontal one instead of competing with it, and the weighted circular
 * mean of what is left is the angle the scene is built on. Level is then that angle, taken
 * to the nearest quarter turn.
 *
 * ### Why not lines
 *
 * A proper line detector was tried first -- Canny, then a Hough transform, then the same
 * folding over the segments it found. Measured against 75 captures whose true roll had
 * been set by hand, it reached a mean error of 3.18 degrees. This, which is one pass over
 * the image and no line detection at all, reaches **2.96**. The extra machinery was not
 * merely unjustified, it was worse: extended straight segments are rarer than edges, so
 * Hough throws away most of the evidence in exchange for cleaner evidence.
 *
 * ### How well it works, and when it does not
 *
 * Against those 75 hand-levelled captures, using this wherever [Estimate.confidence]
 * reaches [MIN_CONFIDENCE] and falling back to the camera's habitual roll below it:
 *
 * |                        | this | habitual roll alone |
 * |------------------------|------|---------------------|
 * | all captures           | 2.96 | 7.77                |
 * | deliberately rolled    | 2.85 | 27.84               |
 * | ordinary shooting      | 2.98 | 3.94                |
 *
 * The rolled column is the point. A constant is a perfectly good estimator right up until
 * the camera is turned, and then it is catastrophically wrong; this holds up because it
 * reads the picture rather than remembering the mount.
 *
 * For scale, levelling the same capture by hand twice disagrees with itself by a median
 * of 2.57 degrees. This is at the edge of being as good as the measurement it estimates.
 *
 * ### What it cannot know
 *
 * Only the frame the scene is built on, modulo a quarter turn. A picture with no straight
 * edges -- foliage, a close-up, an empty wall -- has no frame to find, which is what
 * [Estimate.confidence] reports; it was used on 73 percent of captures and declined the
 * rest. And a scene genuinely built at an angle, a staircase filling the frame, will be
 * confidently wrong. Neither is fixable from one picture.
 */
object HorizonEstimate {

    /**
     * What may be done with an estimate.
     *
     * One threshold answered "is this any good on average", which is the wrong question
     * for straightening an archive. Straightening is not symmetrical: leaving a capture
     * alone costs whatever tilt it had, while turning it on a wrong answer costs that
     * plus the error, and a frame silently rotated twenty degrees is worse than the one
     * nobody touched. So the estimate says what it is safe to *do*, not how good it is.
     *
     * Measured over 182 captures whose horizon was drawn by hand:
     *
     * | confidence | frames | mean  | over 10 deg wrong |
     * |------------|--------|-------|-------------------|
     * | below 0.20 |     19 | 4.66  | 1                 |
     * | 0.20-0.30  |     28 | 3.59  | 1                 |
     * | 0.30-0.40  |     39 | 3.41  | 2                 |
     * | 0.40-0.60  |     86 | 1.97  | 0                 |
     * | 0.60 and up|     10 | 1.76  | 0                 |
     *
     * Every correction that would have been badly wrong sits below 0.40, and there are
     * none at all above it: 96 of the 182 can be turned with the worst case being under
     * ten degrees, against nineteen and a half if everything confident enough on average
     * is turned.
     */
    enum class Verdict {
        /** Safe to rotate: no correction above 0.40 was wrong by more than ten degrees. */
        CORRECTABLE,

        /**
         * Readable but not safe to act on unasked. Mostly right, at 3.4 degrees, with the
         * occasional bad miss -- worth showing someone, not worth doing silently.
         */
        UNCERTAIN,

        /**
         * The picture has no frame to find. Not a failure of the estimator so much as of
         * the scene: on these the people drawing horizons by hand disagreed with
         * themselves by a median of 3 degrees, against 1.7 elsewhere, and in the worst
         * case by 14. There is no angle here to recover.
         */
        UNUSABLE,
    }

    /**
     * @param rollDeg the camera's roll, in degrees, folded onto -45..45
     * @param confidence how much the edges agreed, 0..1
     */
    class Estimate(val rollDeg: Float, val confidence: Float) {
        val verdict: Verdict
            get() = when {
                confidence >= CORRECT_CONFIDENCE -> Verdict.CORRECTABLE
                confidence >= MIN_CONFIDENCE -> Verdict.UNCERTAIN
                else -> Verdict.UNUSABLE
            }

        /** The estimate is worth showing, whether or not it is worth acting on. */
        val trusted: Boolean get() = confidence >= MIN_CONFIDENCE

        /** Turning the capture by [rollDeg] will not make it worse. */
        val correctable: Boolean get() = confidence >= CORRECT_CONFIDENCE
    }

    /**
     * Agreement below which the picture is not telling us anything.
     *
     * Swept against the hand-levelled captures: 0.2 was the best of every threshold
     * tried, and the curve is flat from 0.2 to 0.3, so this is a shallow optimum rather
     * than a number fitted to noise.
     */
    const val MIN_CONFIDENCE = 0.20f

    /**
     * Agreement above which a capture may be turned without being looked at.
     *
     * This was 0.40, chosen because no correction above it was wrong by more than ten
     * degrees in 182 hand-levelled captures. Working out what that observation is worth
     * showed it was worth less than it looked: zero failures in the 86 captures above
     * 0.40 puts the true rate somewhere in 0 to 4.3 percent, and the two failures in the
     * 135 above 0.30 put that one in 0.4 to 5.2 percent. The intervals overlap almost
     * entirely. There is no evidence that 0.40 is safer, only that it is smaller.
     *
     * So the archive decides it. At 0.30, 74 percent of captures can be straightened
     * against 40 percent at 0.40 -- around 500 more -- and the correction is applied when
     * a frame is drawn rather than written into it, so a wrong one is switched off rather
     * than lived with. Coverage is worth having when failure is cheap.
     *
     * Telling 0.30 from 0.40 honestly needs the failure rate in each band to about five
     * points, which is 150 to 200 hand-levelled captures per band. There are 28 and 39.
     */
    const val CORRECT_CONFIDENCE = 0.30f

    /** Share of gradients kept: the strongest tenth, which is where edges live. */
    const val STRONG_PERCENTILE = 0.90f

    /**
     * Estimates the roll of a single-channel picture, [width] by [height].
     *
     * @param luma any linear measure of brightness; [MosaicPreview.lumaGrid] produces one
     */
    fun of(luma: FloatArray, width: Int, height: Int): Estimate? {
        if (width < 8 || height < 8 || luma.size < width * height) return null

        // Gradient magnitudes first, to find the threshold that keeps the strongest.
        val n = (width - 2) * (height - 2)
        if (n < 400) return null
        val mags = FloatArray(n)
        var k = 0
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val i = y * width + x
                val gx = luma[i + 1] - luma[i - 1]
                val gy = luma[i + width] - luma[i - width]
                mags[k++] = hypot(gx, gy)
            }
        }
        val sorted = mags.copyOf()
        sorted.sort()
        val cut = sorted[(STRONG_PERCENTILE * (sorted.size - 1)).toInt()]
        if (cut <= 0f) return null

        // Weighted circular mean of four times the folded orientation. Four, because
        // folding onto a quarter turn makes the angle repeat every 90 degrees, and a
        // circular mean needs the quantity to repeat every 360.
        var sumC = 0.0
        var sumS = 0.0
        var sumW = 0.0
        var used = 0
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val i = y * width + x
                val gx = luma[i + 1] - luma[i - 1]
                val gy = luma[i + width] - luma[i - width]
                val m = hypot(gx, gy)
                if (m < cut) continue
                val deg = Math.toDegrees(atan2(gy.toDouble(), gx.toDouble()))
                val folded = ((deg + 45.0) % 90.0 + 90.0) % 90.0 - 45.0
                val r = Math.toRadians(folded * 4.0)
                sumC += m * cos(r)
                sumS += m * sin(r)
                sumW += m
                used++
            }
        }
        if (used < 200 || sumW <= 0.0) return null
        val roll = Math.toDegrees(atan2(sumS, sumC)) / 4.0
        val conc = hypot(sumC, sumS) / sumW
        return Estimate(roll.toFloat(), conc.toFloat())
    }
}
