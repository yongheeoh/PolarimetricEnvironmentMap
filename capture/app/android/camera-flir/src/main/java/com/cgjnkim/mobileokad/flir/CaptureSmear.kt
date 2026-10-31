package com.cgjnkim.mobileokad.flir

/**
 * How far the picture will move while the shutter is open, and what to do about it.
 *
 * ### The measurement
 *
 * The preview registers each frame against the last, which gives the camera's own
 * displacement in its own pixels -- no focal length, no assumption about how the phone is
 * attached, and not the phone's gyroscope, which has been shown to turn independently of
 * the camera on this rig. Multiplied by the exposure about to be used, that is the smear
 * the next frame will carry.
 *
 * ### What the archive says it is worth
 *
 * Measured over the two days the drift reading has been recorded, on 882 captures that
 * were kept and 35 that a person later deleted:
 *
 * | | predicted smear |
 * |---|---|
 * | kept | 1.13 px median |
 * | deleted | 7.55 px median |
 *
 * Ranking captures by it separates the two at an AUC of 0.728, against 0.678 for the
 * phone's gyroscope on the same captures. It is a genuine predictor and it is a weak one:
 * measuring the frame after the fact reaches 0.954. This is what can be known *before*
 * the shutter, which is the only moment at which anything can still be done.
 */
object CaptureSmear {

    /**
     * Pixels the picture moves during [exposureUs], or null when nothing was measured.
     *
     * @param driftPixelsPerSecond the preview's own registration, in Stokes pixels
     */
    fun pixels(driftPixelsPerSecond: Float?, exposureUs: Long): Float? {
        if (driftPixelsPerSecond == null || exposureUs <= 0) return null
        return driftPixelsPerSecond * (exposureUs / 1e6f)
    }

    /**
     * Smear past which taking several frames and keeping the sharpest stops helping.
     *
     * A burst works because shake varies between frames a fifth of a second apart, so one
     * of them catches a quiet moment. That stops being true once the movement is large:
     * every frame is smeared by the same sweep and there is no good one to find.
     *
     * The archive says exactly where. Across 857 captures carrying both a burst and a
     * drift reading, the sharpness gained by keeping the best of four rather than simply
     * taking the first:
     *
     * | predicted smear | n | gain | first frame won |
     * |---|---|---|---|
     * | under 0.5 px | 248 | 1.022x | 31% |
     * | 0.5 to 1 | 144 | 1.034x | 34% |
     * | 1 to 2 | 147 | 1.022x | 43% |
     * | 2 to 4 | 110 | 1.010x | 45% |
     * | 4 to 10 | 104 | 1.032x | 40% |
     * | **over 10** | **101** | **1.000x** | **57%** |
     *
     * The gain is flat wherever the burst is worth anything at all, and it is gone in the
     * last row -- the first frame wins more often than chance, which is what "no signal"
     * looks like. So there is no case for bursting *harder* when the camera is moving,
     * which is what this was originally going to do; there is only a case for not paying
     * for three extra frames that demonstrably buy nothing.
     */
    const val BURST_FUTILE_PIXELS = 10f

    /**
     * Smear worth waiting a moment to avoid.
     *
     * Set at the warning's own raise threshold, so the shutter waits for exactly the
     * condition the viewfinder is already marking rather than inventing a second opinion.
     */
    const val STEADY_PIXELS = 1.75f

    /**
     * Longest the shutter will wait for a hand to settle, in milliseconds.
     *
     * Long enough for a tremor to pass -- hand shake is a few hertz, so this covers a
     * cycle or two -- and short enough that a press which does not settle still takes the
     * picture rather than feeling broken. The wait is abandoned the moment the drift
     * drops, so a steady hand never pays it.
     */
    const val STEADY_WAIT_MS = 300L

    /**
     * Smear above which shortening the exposure is worth the noise it costs.
     *
     * Above the futile threshold on purpose: below it a burst is still finding good
     * frames, and shortening the exposure there would spend signal-to-noise on a problem
     * that is already being solved for free. Above it neither the burst nor waiting is
     * going to help, and blur is the one of the two that cannot be undone afterwards --
     * noise can be corrected for, and the app already does, from a calibrated model.
     */
    const val SHORTEN_ABOVE_PIXELS = BURST_FUTILE_PIXELS

    /** Smear a shortened exposure aims for. */
    const val SHORTEN_TARGET_PIXELS = 4f

    /**
     * Most the exposure may be shortened, as a factor.
     *
     * Four is two stops, which doubles the noise standard deviation and so doubles the
     * DoLP noise floor. Past that a dim scene stops being measurable at all, and an
     * unmeasurable sharp frame is no better than a blurred one.
     */
    const val SHORTEN_MAX_FACTOR = 4f

    /**
     * The exposure to use instead of [exposureUs], given what the preview is seeing.
     *
     * Returns [exposureUs] unchanged whenever there is nothing to gain -- no drift
     * reading, or a hand steady enough that the frame was going to be fine.
     */
    fun shortenedExposureUs(driftPixelsPerSecond: Float?, exposureUs: Long): Long {
        val smear = pixels(driftPixelsPerSecond, exposureUs) ?: return exposureUs
        if (smear <= SHORTEN_ABOVE_PIXELS) return exposureUs
        val wanted = (smear / SHORTEN_TARGET_PIXELS).coerceAtMost(SHORTEN_MAX_FACTOR)
        return (exposureUs / wanted).toLong().coerceAtLeast(1L)
    }
}
