package com.cgjnkim.mobileokad.flir

import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * The views a Stokes frame can be shown as, rendered to packed ARGB.
 *
 * The same set backs the live preview and the stored capture, so what the gallery shows
 * later is exactly what was on screen when the shutter was pressed.
 *
 * Every view but S0_RGB is single-channel and is computed from whichever band the
 * caller renders it for; the file kinds name the view, not the band, since the capture
 * pipeline writes the default one.
 */
enum class StokesView(val label: String, val fileKind: String) {
    /** Intensity in colour: S0 of R, G and B as an ordinary RGB image. */
    S0_RGB("S0 RGB", "pol_s0"),

    /** Angle of linear polarization, on a cyclic hue wheel. */
    AOLP("AoLP", "pol_aolp"),

    /** Degree of linear polarization, 0..1 as black to white. */
    DOLP("DoLP", "pol_dolp"),

    /** Normalized S1, diverging blue-white-red. */
    S1("S1", "pol_s1"),

    /** Normalized S2, diverging blue-white-red. */
    S2("S2", "pol_s2");

    companion object {
        val LABELS: List<String> = entries.map { it.label }
    }
}

object StokesRender {

    /**
     * Band the single-channel views are computed from when the caller does not say.
     *
     * Green, because it has two of the four colour cells and so the least noise; it is
     * the right default but not the only answer, since a surface can be far more
     * polarizing at one wavelength than another.
     */
    const val DEFAULT_BAND = StokesImage.G

    /**
     * Renders [view] into a packed ARGB array of `image.width * image.height` pixels.
     *
     * [out] is reused across preview frames when it is already the right size, so the
     * preview loop does not allocate a new 600 kB array several times a second.
     */
    @JvmOverloads
    fun render(
        image: StokesImage,
        view: StokesView,
        out: IntArray? = null,
        display: StokesDisplay = StokesDisplay.DEFAULT,
        band: Int = DEFAULT_BAND,
    ): IntArray {
        val n = image.width * image.height
        val pixels = if (out != null && out.size == n) out else IntArray(n)

        // Smoothing is a polarization correction, so S0 RGB never sees it and never pays
        // for it. Done once here rather than inside each view, so DoLP and the AoLP fade
        // are drawn from the same components.
        // The gradient leak is removed before smoothing, not after: smoothing is what
        // makes the leak visible, so correcting afterwards would be arguing with the
        // filter about an artefact that was in the frame before either ran.
        val corrected =
            if (view != StokesView.S0_RGB && display.correctGradientLeak) {
                image.gradientLeakCorrected(band)
            } else {
                image
            }
        val smoothed =
            if (view != StokesView.S0_RGB && display.smoothPolarization) {
                corrected.smoothedPolarization(band)
            } else {
                corrected
            }
        // Last, because it is a change of reference frame and nothing else: the leak
        // correction reads gradients in the sensor's own axes and the filter is guided by
        // S0, so both belong before the frame is turned.
        val source = display.horizonDeg
            ?.takeIf { view != StokesView.S0_RGB && it != 0f }
            ?.let { smoothed.horizonCorrected(band, it) }
            ?: smoothed

        // S0_RGB is the one view that is inherently all three bands, so it ignores the
        // band and renders the colour picture whatever is asked for.
        when (view) {
            StokesView.S0_RGB -> renderS0Rgb(image, pixels, display)
            StokesView.AOLP -> renderAolp(source, pixels, display, band)
            StokesView.DOLP -> renderDolp(source, pixels, display, band)
            StokesView.S1 -> renderDiverging(source.normalized(source.s1, band), pixels)
            StokesView.S2 -> renderDiverging(source.normalized(source.s2, band), pixels)
        }
        // Last of all, and on the picture rather than on the components: straightening is
        // a resampling of the frame, so doing it earlier would mean every view resampling
        // the measurement it is drawn from. This way the arrays stay exactly as measured
        // and only what reaches the screen is moved.
        if (!display.rectify) return pixels
        val straightened = LensCalibration.rectify(pixels, image.width, image.height)
        straightened.copyInto(pixels)
        return pixels
    }

    /**
     * S0 as an ordinary colour picture.
     *
     * S0 is the four analyser readings summed and halved, so dividing by the format's
     * full scale puts an unpolarized scene on 0..1. Everything after that is a viewing
     * choice -- exposure, white balance, transfer curve -- and none of it changes the
     * arrays underneath. [StokesDisplay.LINEAR] renders exactly what was measured.
     */
    private fun renderS0Rgb(image: StokesImage, out: IntArray, display: StokesDisplay) {
        val r = image.s0[StokesImage.R]
        val g = image.s0[StokesImage.G]
        val b = image.s0[StokesImage.B]
        val encoder = S0Encoder(display, image.maxValue, r, g, b)
        Parallel.over(out.size) { from, to ->
            for (i in from until to) out[i] = encoder.encode(r[i], g[i], b[i])
        }
    }

    /**
     * Turns linear S0 into screen pixels: exposure, balance, temperature, transfer curve.
     *
     * Shared with [Demosaic] rather than written twice. The demosaiced picture and the
     * quarter-resolution one are the same measurement at different sampling, and if their
     * colour rendering drifted apart, switching between them in the viewer would look like
     * the data had changed.
     *
     * The balance is decided once from the whole frame at construction, so every pixel
     * goes through the same gains -- which is what makes it a white balance rather than a
     * local correction.
     */
    internal class S0Encoder(
        display: StokesDisplay,
        maxValue: Float,
        red: FloatArray,
        green: FloatArray,
        blue: FloatArray,
    ) {
        // S0 is the four analyser readings summed and halved, so an unpolarized scene at
        // full scale arrives here as twice maxValue.
        private val norm = 0.5f / maxValue * display.exposureGain
        private val gainR: Float
        private val gainB: Float

        /**
         * The transfer curve, precomputed.
         *
         * It was a `pow` per channel per pixel, and at the sensor's own resolution that is
         * fifteen million of them: measured at 214 ms of the 331 ms a full-resolution
         * demosaic took, which is most of the wait the gallery was showing a spinner for.
         * The table costs 65536 of the same calls to fill, under a millisecond, and turns
         * the rest into an array read.
         *
         * Sized for the shadows rather than for the table. The curve is steepest near
         * black -- its slope there is what decides how finely the input must be sampled --
         * and 65536 steps keep every output within half a level of the exact value, where
         * the 1024 that would do for the highlights would band the darks visibly.
         */
        private val curve = IntArray(CURVE_STEPS)

        init {
            val invGamma =
                1f / display.gamma.coerceIn(StokesDisplay.MIN_GAMMA, StokesDisplay.MAX_GAMMA)
            for (i in 0 until CURVE_STEPS) {
                curve[i] = clamp8(255f * (i / (CURVE_STEPS - 1f)).pow(invGamma))
            }
            val balance = when (display.whiteBalance) {
                WhiteBalance.GREY_WORLD -> grayWorldGains(red, green, blue)
                // Divided by what the camera had already applied, so one pair of
                // constants describes a capture from either side of the day the ratios
                // were set to unity.
                WhiteBalance.FIXED -> floatArrayOf(
                    StokesDisplay.FIXED_WHITE_BALANCE_RED / display.sensorRedRatio.orOne(),
                    StokesDisplay.FIXED_WHITE_BALANCE_BLUE / display.sensorBlueRatio.orOne(),
                )
                WhiteBalance.NONE -> floatArrayOf(1f, 1f)
            }
            val temperature = display.temperatureGains()
            gainR = balance[0] * temperature[0]
            gainB = balance[1] * temperature[1]
        }

        fun encode(r: Float, g: Float, b: Float): Int = argb(
            transfer(r * norm * gainR),
            transfer(g * norm),
            transfer(b * norm * gainB),
        )

        // Rounded to the nearest entry, not truncated to the one below. The curve climbs
        // fastest at the bottom, so truncating there costs a whole output level -- it read
        // 4 where the exact value is 5 -- while rounding halves the worst error for the
        // price of an addition.
        private fun transfer(linear: Float): Int =
            curve[(linear.coerceIn(0f, 1f) * (CURVE_STEPS - 1) + 0.5f).toInt()]
    }

    /**
     * Steps in the transfer table.
     *
     * Enough that the darkest output levels are still one apart: the curve's slope at the
     * bottom is what sets this, and at 2.2 the first step of a 1024 entry table would jump
     * three levels at once.
     */
    private const val CURVE_STEPS = 65536

    /**
     * Red and blue gains that put the frame's channel averages onto green's.
     *
     * Green is the reference because it has two of the four colour cells and so the
     * least noise of the three.
     */
    /** A ratio that could not be read is a ratio that was not applied. */
    private fun Float.orOne(): Float = if (isFinite() && this > 0f) this else 1f

    private fun grayWorldGains(r: FloatArray, g: FloatArray, b: FloatArray): FloatArray {
        val totals = Parallel.sums(r.size, 3) { from, to ->
            var sr = 0.0
            var sg = 0.0
            var sb = 0.0
            for (i in from until to) {
                sr += r[i]
                sg += g[i]
                sb += b[i]
            }
            doubleArrayOf(sr, sg, sb)
        }
        val sumR = totals[0]
        val sumG = totals[1]
        val sumB = totals[2]
        if (sumR <= EPSILON || sumG <= EPSILON || sumB <= EPSILON) return floatArrayOf(1f, 1f)
        return floatArrayOf(
            (sumG / sumR).toFloat().coerceIn(WB_GAIN_MIN, WB_GAIN_MAX),
            (sumG / sumB).toFloat().coerceIn(WB_GAIN_MIN, WB_GAIN_MAX),
        )
    }

    /**
     * Hue carries the angle; how far the hue is pushed from grey carries how much
     * polarization there was to measure it from.
     *
     * Without that second term the view is unreadable in anything dim: the angle is
     * undefined where S1 and S2 are noise, and painting it at full strength anyway is
     * what makes dark parts of a frame look like confetti.
     *
     * The strength is the debiased DoLP when the frame carries a noise fit, so the fade
     * is driven by the polarization that was actually measured rather than by the floor
     * the noise puts under it. That floor is 0.07 to 0.09 across a dim frame, which is
     * enough on its own to keep noise hues clearly visible.
     */
    private fun renderAolp(image: StokesImage, out: IntArray, display: StokesDisplay, band: Int) {
        val s0 = image.s0[band]
        val s1 = image.s1[band]
        val s2 = image.s2[band]
        val model = if (display.debiasDolp) image.noise else null
        val modulate = display.modulateAolpByDolp

        // One pass, reading the arrays directly. Going through aolp() and dolp() meant
        // three passes over the frame and two 1.2 MB arrays allocated and thrown away
        // per render -- and the sliders re-render on every touch. Those two stay as they
        // are for anyone measuring; this is the drawing.
        Parallel.over(out.size) { from, to ->
            for (i in from until to) {
                val hue = HUE_LUT[(turnOf(s2[i], s1[i]) * HUE_STEPS).toInt() % HUE_STEPS]
                if (!modulate) {
                    out[i] = hue
                    continue
                }
                val denom = s0[i]
                val strength = if (denom <= EPSILON) {
                    0f
                } else {
                    val magnitude = model?.debiasedMagnitude(s1[i], s2[i], denom, band)
                        ?: sqrt(s1[i] * s1[i] + s2[i] * s2[i])
                    (magnitude / denom).coerceIn(0f, 1f)
                }
                out[i] = fadeToWhite(hue, strength)
            }
        }
    }

    /** Blends a colour toward white; [amount] 0 leaves white, 1 leaves the colour. */
    private fun fadeToWhite(color: Int, amount: Float): Int {
        val a = amount.coerceIn(0f, 1f)
        val r = (color shr 16) and 0xFF
        val g = (color shr 8) and 0xFF
        val b = color and 0xFF
        return argb(
            clamp8(255f - a * (255 - r)),
            clamp8(255f - a * (255 - g)),
            clamp8(255f - a * (255 - b)),
        )
    }

    private fun renderDolp(image: StokesImage, out: IntArray, display: StokesDisplay, band: Int) {
        val dolp = image.dolp(band, display.debiasDolp)
        Parallel.over(out.size) { from, to ->
            for (i in from until to) {
                val v = clamp8(dolp[i] * 255f)
                out[i] = argb(v, v, v)
            }
        }
    }

    /**
     * Blue for negative, white at zero, red for positive.
     *
     * A diverging map is what makes the sign readable at a glance; a sequential one
     * would hide where the component crosses zero.
     */
    private fun renderDiverging(values: FloatArray, out: IntArray) {
        Parallel.over(out.size) { from, to ->
            for (i in from until to) {
                val v = values[i].coerceIn(-1f, 1f)
                out[i] = if (v < 0f) {
                    val t = 1f + v  // -1 -> pure blue, 0 -> white
                    argb(clamp8(255f * t), clamp8(255f * t), 255)
                } else {
                    val t = 1f - v  // 0 -> white, 1 -> pure red
                    argb(255, clamp8(255f * t), clamp8(255f * t))
                }
                    }
        }
    }

    /** Fully saturated hue wheel; [hue] is a fraction of a full turn. */
    private fun hueToArgb(hue: Float): Int {
        val h = ((hue % 1f) + 1f) % 1f * 6f
        val sector = h.toInt()
        val f = h - sector
        val q = clamp8(255f * (1f - f))
        val t = clamp8(255f * f)
        return when (sector) {
            0 -> argb(255, t, 0)
            1 -> argb(q, 255, 0)
            2 -> argb(0, 255, t)
            3 -> argb(0, q, 255)
            4 -> argb(t, 0, 255)
            else -> argb(255, 0, q)
        }
    }

    private fun clamp8(value: Float): Int = value.roundToInt().coerceIn(0, 255)

    private fun argb(r: Int, g: Int, b: Int): Int = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    /**
     * The hue wheel, precomputed.
     *
     * 256 steps over half a turn of analyser angle is 0.7 degrees a step, which is well
     * inside the angle's own uncertainty anywhere the fade leaves it visible.
     */
    private const val HUE_STEPS = 256
    private val HUE_LUT = IntArray(HUE_STEPS) { hueToArgb(it.toFloat() / HUE_STEPS) }

    private const val EPSILON = 1e-6
    private const val WB_GAIN_MIN = 0.5f
    private const val WB_GAIN_MAX = 3f
}

/**
 * `atan2(y, x)` as a fraction of a full turn, in `[0, 1)`.
 *
 * Two things at once, both for drawing rather than measuring. The turn is what a hue
 * wants -- the angle of linear polarization is half the Stokes angle, so a full turn of
 * `atan2` is exactly the half turn of analyser angle the wheel spans, and no conversion
 * through degrees is needed. And the arc tangent is approximated rather than called:
 * a rational fit on `[0, 1]` folded into the right octant, accurate to about 1e-5 rad.
 *
 * That error is a thousandth of one of the 256 hue steps, so nothing drawn through this
 * differs from the exact answer. [StokesImage.aolp] remains exact, and is what anything
 * measuring the angle should read.
 */
internal fun turnOf(y: Float, x: Float): Float {
    if (x == 0f && y == 0f) return 0f
    val ax = kotlin.math.abs(x)
    val ay = kotlin.math.abs(y)
    val a = kotlin.math.min(ax, ay) / kotlin.math.max(ax, ay)
    val s = a * a
    var r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a
    if (ay > ax) r = HALF_PI - r
    if (x < 0f) r = PI_F - r
    if (y < 0f) r = -r
    val turn = r * INV_TWO_PI
    return if (turn < 0f) turn + 1f else turn
}

private const val PI_F = 3.14159265f
private const val HALF_PI = PI_F / 2f
private const val INV_TWO_PI = 1f / (2f * PI_F)
