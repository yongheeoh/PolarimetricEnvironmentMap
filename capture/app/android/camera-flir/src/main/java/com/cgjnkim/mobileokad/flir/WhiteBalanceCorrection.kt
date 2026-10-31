package com.cgjnkim.mobileokad.flir

/**
 * Taking the camera's own white balance back out of captures that were shot with it on.
 *
 * ### What happened
 *
 * The camera holds a ratio per colour against green. Setting `BalanceWhiteAuto` to Off,
 * which is what the app did from the start, stops the ratios being *updated* -- it does
 * not neutralise them. This body was holding 1.0986 for red and 1.8311 for blue, and
 * `IspEnable` reading zero was not enough to keep them off the payload.
 *
 * The archive says so plainly. A band multiplied by k can only land on one integer code in
 * k, so its histogram is a comb. Across well-exposed captures ten days apart, blue occupies
 * 0.546 of its range where green occupies 1.000 -- and 1/1.8311 is 0.546. Red comes out
 * near 0.90 against a predicted 0.910. The scaling reached the pixels.
 *
 * ### What it did and did not break
 *
 * Per-band DoLP and AoLP are untouched. A colour's four analyser readings sit in one 2x2
 * cell and are scaled alike, so the ratio cancels in a ratio.
 *
 * The noise model is not untouched. [StokesNoise] reads variance as a constant times the
 * signal in raw counts; a band multiplied by k has variance k^2 * g * (I/k) = k * g * I, so
 * its shot gain is k times what the model assumes. Everything resting on that -- the DoLP
 * debias, the noise floor, the valid-pixel fraction -- is wrong for red and blue by that
 * factor and right for green. Absolute radiometry and any comparison between bands are
 * wrong by the ratio itself.
 *
 * ### What this can and cannot give back
 *
 * Dividing restores the scale exactly. It does not restore the quantisation: blue reached
 * 54.6 percent of its codes and the rest were never distinguishable, so blue carries about
 * 11.2 bits rather than 12. That loss is permanent and small, and it is the reason the
 * camera is now set to unity at open rather than this being treated as a solved problem.
 *
 * ### Which captures need it
 *
 * The ones with no record of their ratios. Captures taken since the app started writing
 * `rawPreset.balanceRatios` carry their own, and those read one because the app now sets
 * them; anything older predates both the recording and the fix, and was shot under the
 * factory values in [LEGACY]. That rule is what keeps a corrected archive from being
 * corrected twice.
 */
object WhiteBalanceCorrection {

    /**
     * What the camera was holding before the app started setting it to one.
     *
     * Read from the registers its own description points at, and confirmed against the
     * archive's histograms independently.
     */
    val LEGACY = mapOf(StokesImage.R to 1.0986f, StokesImage.B to 1.8311f)

    /** Nothing to undo, for captures that were taken after the fix. */
    val NONE = emptyMap<Int, Float>()

    /**
     * The ratios to divide out of a capture, given what its metadata recorded.
     *
     * @param recorded what `rawPreset.balanceRatios` held, or null when the capture
     *   predates that field
     */
    fun ratiosFor(recorded: Map<Int, Float>?): Map<Int, Float> {
        // No record means it was shot before the app knew to keep one, which is also
        // before it set them to unity.
        if (recorded == null) return LEGACY
        // A record of unity is a capture that needs nothing, and a record of anything else
        // is that capture's own truth, which beats the constant.
        return recorded.filterValues { kotlin.math.abs(it - 1f) > SLACK }
    }

    /**
     * Divides [ratios] out of a decoded frame, in place, returning it.
     *
     * Works on the Stokes planes rather than the mosaic because that is where a band is a
     * band: every component of one colour carries the same factor, so all three are
     * divided and the polarization the ratio already cancelled out of stays where it was.
     */
    fun applyTo(image: StokesImage, ratios: Map<Int, Float>): StokesImage {
        if (ratios.isEmpty()) return image
        for ((band, ratio) in ratios) {
            if (ratio <= 0f || band !in image.s0.indices) continue
            val scale = 1f / ratio
            for (plane in arrayOf(image.s0[band], image.s1[band], image.s2[band])) {
                for (i in plane.indices) plane[i] *= scale
            }
        }
        return image
    }

    /** How far a recorded ratio may sit from one before it is worth undoing. */
    private const val SLACK = 0.001f

    /**
     * What a capture's own pixels say was done to them.
     *
     * @param red ratio implied by red's histogram, or null when it could not be read
     * @param green the control: the band white balance is measured against is never
     *   scaled, so this should read one and anything less means the frame was too dark to
     *   fill its histogram -- which makes the other two meaningless as well
     */
    class Implied(val red: Float?, val blue: Float?, val green: Float) {
        /**
         * Whether the frame was bright enough for the other two to mean anything.
         *
         * A dark capture leaves most of the range empty for every band, so all three
         * occupancies fall together and none of them is about scaling. Green reading near
         * one is what says the histogram was actually filled.
         */
        val trustworthy: Boolean get() = green >= MIN_GREEN_OCCUPANCY
    }

    /**
     * Reads the ratios back out of a capture's own histogram.
     *
     * A band multiplied by k can only land on one integer code in k, so the share of codes
     * it ever occupies is 1/k and the occupancy inverts to the ratio. This is what makes
     * the archive correctable without the camera: every capture carries the evidence of
     * what was done to it, whether or not anybody was recording it at the time.
     *
     * Confirmed against the registers on this body: blue occupied 0.546 of its range where
     * green occupied 1.000, and 1/1.8311 is 0.546.
     */
    fun measure(mosaic: Mosaic): Implied {
        val m = FlirCamera.MOSAIC
        if (mosaic.width % m != 0 || mosaic.height % m != 0) return Implied(null, null, 0f)
        // Bayer RGGB over the super-pixel: colour is the outer 2x2, analyser the inner.
        val red = occupancy(mosaic, 0, 0)
        val green = occupancy(mosaic, 2, 0)
        val blue = occupancy(mosaic, 2, 2)
        return Implied(
            red = red?.let { 1f / it },
            blue = blue?.let { 1f / it },
            green = green ?: 0f,
        )
    }

    /**
     * Share of the integer codes a band uses, over the part of its range it actually
     * populates.
     *
     * Measured between the band's own quantiles rather than between its extremes, and this
     * is what makes the reading about scaling instead of about exposure. A band that was
     * multiplied cannot reach one code in k **anywhere**, and the gaps are periodic; a band
     * that is merely dim reaches every code it lands near and leaves the rest of its span
     * empty at random. Taking the whole min-to-max range cannot tell those apart -- a dim
     * blue band in a bright frame reads as heavily scaled -- while the middle of the
     * distribution, where the samples are, separates them cleanly.
     */
    private fun occupancy(mosaic: Mosaic, offsetX: Int, offsetY: Int): Float? {
        val values = ArrayList<Int>()
        var y = offsetY
        while (y < mosaic.height) {
            var x = offsetX
            while (x < mosaic.width) {
                values += mosaic[x, y].toInt()
                x += FlirCamera.MOSAIC
            }
            y += FlirCamera.MOSAIC
        }
        if (values.size < MIN_SAMPLES) return null
        values.sort()
        val lo = values[(values.size * LOW_QUANTILE).toInt()]
        val hi = values[((values.size * HIGH_QUANTILE).toInt()).coerceAtMost(values.size - 1)]
        val span = hi - lo + 1
        // Too narrow a window says the band is nearly flat, which no occupancy describes.
        if (span < MIN_SPAN) return null
        val seen = HashSet<Int>()
        for (v in values) if (v in lo..hi) seen += v
        return seen.size.toFloat() / span.toFloat()
    }

    /** The middle of a band's distribution, where its samples actually are. */
    private const val LOW_QUANTILE = 0.10
    private const val HIGH_QUANTILE = 0.90

    /** Below these a band cannot support the reading at all. */
    private const val MIN_SAMPLES = 10_000
    private const val MIN_SPAN = 64

    /**
     * Green occupancy below which the reading is not about white balance.
     *
     * Green is never scaled -- it is the band the others are measured against -- so over a
     * window where its samples actually sit it occupies every code and reads one. It doing
     * anything else means the window found no distribution to measure, and the other two
     * bands are then describing the same absence rather than a ratio.
     *
     * Measured over the archive this reads exactly 1.000 on every capture tried, bright
     * and dim alike, which is what says the quantile window took exposure out of the
     * question. It is kept as a guard rather than dropped because a frame of a blank wall
     * or a lens cap has no distribution at all.
     */
    const val MIN_GREEN_OCCUPANCY = 0.95f
}
