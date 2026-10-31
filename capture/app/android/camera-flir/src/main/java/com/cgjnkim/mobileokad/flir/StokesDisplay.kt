package com.cgjnkim.mobileokad.flir

import kotlin.math.pow

/**
 * Display-only settings for turning a Stokes frame into a picture.
 *
 * None of this touches what is stored. The archive is the raw mosaic and the Stokes
 * arrays; these choices are how a viewer draws them, and two viewers with different
 * settings are looking at the same measurement.
 */
/**
 * Where a picture's colour balance comes from.
 */
enum class WhiteBalance {
    /**
     * From the frame itself: the channel means are put onto green's.
     *
     * Needs no calibration target and no knowledge of the sensor, and it is right
     * whenever the scene really does average to neutral. When it does not -- a wall, a
     * field, a sky -- it corrects the scene's own colour away, and it does it by a
     * different amount for every capture, so two frames of the same subject taken a
     * second apart come out two colours.
     */
    GREY_WORLD,

    /**
     * From a calibration of the sensor: [StokesDisplay.FIXED_WHITE_BALANCE_RED] and
     * [StokesDisplay.FIXED_WHITE_BALANCE_BLUE].
     *
     * The same correction for every capture, which is what makes an archive comparable
     * with itself -- and what This Polar Camera's gallery does. A coloured scene stays
     * coloured. It assumes the light is roughly what the calibration sample was shot
     * under, and the temperature control is there for when it is not.
     */
    FIXED,

    /** None at all: the bands as the sensor delivered them, which read green. */
    NONE,
}

class StokesDisplay(
    /**
     * Whether the angle of polarization is drawn at a strength set by how polarized the
     * light actually is.
     *
     * On by default, and it matters more than it sounds. AoLP is an angle derived from
     * S1 and S2; where those are small the angle is not merely noisy but undefined, and
     * drawing it at full strength fills dark parts of the frame with confident-looking
     * colour that is pure noise. Measured on this camera at 50 ms and 0 dB, the DoLP
     * expected from noise alone is 0.07-0.09 wherever S0 is under about 500, which is
     * the whole of a dim scene -- and at DoLP 0.05 the angle's own standard deviation is
     * around 34 degrees on a scale that wraps at 180.
     *
     * Fading toward grey with DoLP is what both polanalyser and polar_camera do; the
     * latter builds it into its render kernel as the saturation term. Turning it off
     * shows the raw angle everywhere, which is worth having when the question is what
     * the noise itself looks like.
     */
    val modulateAolpByDolp: Boolean = true,

    /**
     * Whether DoLP has the noise's own contribution subtracted before it is shown.
     *
     * DoLP is a magnitude, so noise can only lengthen it: an unpolarized dark patch still
     * reads 0.07 to 0.09 on this camera. [StokesNoise] measures how much of that is the
     * sensor -- from the two green cells of every super-pixel, on the frame itself -- and
     * the Wardle-Kronberg estimator takes it back out, reporting zero where the reading
     * cannot support a claim at all.
     *
     * On by default, because leaving it off means every dim frame shows polarization it
     * did not measure. Off is worth having when the question is how big that floor is.
     *
     * Applies to the DoLP view and to the strength [modulateAolpByDolp] fades AoLP by. It
     * deliberately does not touch S1 and S2: those are signed, their noise is symmetric,
     * and they are already unbiased. Only the magnitude built from them is not.
     */
    val debiasDolp: Boolean = true,

    /**
     * Whether S1 and S2 are smoothed against S0 before the polarization views are drawn.
     *
     * Where the light is thin the components DoLP is built from are mostly noise, and
     * averaging them over a neighbourhood is what takes that back. [PolarSmoothing] does
     * it with S0 as a guide so the averaging stops at the scene's own boundaries rather
     * than dragging one surface's polarization onto the next.
     *
     * Measured on seven well-exposed captures: the DoLP of the darkest quarter came down
     * by a median factor of 1.20 for a median 2.4 percent of the real polarization. It is
     * worth far more on some scenes than others -- 1.47x for 1.4 percent on a dim smooth
     * one, and 1.02x for 10.4 percent on a bright textured one, where it is simply a loss.
     * On by default because dim and smooth is where DoLP noise is the complaint, and worth
     * turning off when the scene is neither.
     *
     * Applies to DoLP, AoLP and the two component views, and to the DoLP the AoLP fade is
     * drawn at. S0 is never smoothed: it is the denominator and it has the signal.
     */
    /**
     * Whether the brightness gradient that leaks into S1 and S2 at edges is removed.
     *
     * Off by default, and on only in the gallery viewer. It subtracts real signal along
     * with the artefact wherever an edge is genuinely polarized, so it is a way of looking
     * at a capture rather than a correction that belongs in everything: what a capture
     * writes to a PNG, and what the live preview draws, are left alone.
     *
     * See [StokesImage.gradientLeakCorrected].
     */
    /**
     * Roll to take out of the polarization frame, in degrees, or null to leave it alone.
     *
     * The camera's own roll for this capture, which on this rig only a hand-drawn horizon
     * supplies. With it set, AoLP is reported against level instead of against the
     * sensor, which is what makes two captures comparable. See
     * [StokesImage.horizonCorrected].
     */
    val horizonDeg: Float? = null,

    /**
     * Whether the lens's barrel distortion is taken out of the picture before it is shown.
     *
     * Off by default, and a viewing choice like everything else here: the stored frame is
     * what the sensor recorded and stays that way. On, straight edges in the world are
     * drawn straight, which is what makes a frame measurable with a ruler and what makes
     * two frames of the same wall comparable.
     *
     * The middle of the frame does not move, so switching it on and off is a fair
     * comparison rather than a change of zoom. See [LensCalibration].
     */
    val rectify: Boolean = false,
    val correctGradientLeak: Boolean = false,
    val smoothPolarization: Boolean = true,

    /** Exposure adjustment in stops. Each stop doubles or halves the linear signal. */
    val exposureStops: Float = 0f,

    /**
     * Transfer curve applied on the way out. 1.0 leaves the data linear, which is what
     * the sensor measures and what a photograph never looks like; 2.2 is the usual
     * display gamma.
     */
    val gamma: Float = DEFAULT_GAMMA,

    /**
     * How the colour balance is decided, before any temperature is applied.
     *
     * The raw bands are nowhere near balanced -- green carries two of the four colour
     * cells -- so a frame straight off the sensor reads heavily green, and something has
     * to correct it. See [WhiteBalance] for the three answers and when each is right.
     */
    val whiteBalance: WhiteBalance = WhiteBalance.GREY_WORLD,

    /**
     * What the camera had already applied to this capture, red and blue against green.
     *
     * Not a viewing choice -- it is a property of the file -- but the balance cannot be
     * expressed in a single basis without it. Most of this archive was shot while the
     * body was holding [WhiteBalanceCorrection.LEGACY] and those ratios reached the
     * pixels, so a fixed gain applied to an old capture and to a new one would be two
     * different corrections wearing one number. [WhiteBalance.FIXED] divides these out
     * first, which is what makes one constant describe the whole archive.
     *
     * One for a capture that was shot with the ratios at unity, which is every capture
     * taken since the app started setting them.
     */
    val sensorRedRatio: Float = 1f,
    val sensorBlueRatio: Float = 1f,

    /**
     * Colour temperature of the light the picture is balanced for, in kelvin.
     *
     * The raw-editor convention, not the lamp-shopping one: the number describes the
     * illuminant, and the render corrects for it, so **raising it warms the picture**.
     * Telling the render the light was bluer than it was makes it add warmth back.
     *
     * **Not colorimetric.** Turning kelvin into channel gains needs the sensor's
     * spectral response, and this one has never been characterised, so the number is a
     * familiar handle on a warm-to-cool axis rather than a measurement. Anything that
     * needs real colour has to start from the raw mosaic and a calibration target.
     */
    val temperatureK: Float = NEUTRAL_TEMPERATURE_K,

    /** Green to magenta trim, -1 to 1, on top of [temperatureK]. */
    val tint: Float = 0f,
) {
    /**
     * A copy with some settings changed.
     *
     * Spelled out rather than generated: this is a plain class and not a data class,
     * because a `copy()` over eleven settings is exactly the place a new one gets
     * forgotten. Every property is here, and [signature] is the test that says so.
     */
    fun with(
        modulateAolpByDolp: Boolean = this.modulateAolpByDolp,
        debiasDolp: Boolean = this.debiasDolp,
        horizonDeg: Float? = this.horizonDeg,
        rectify: Boolean = this.rectify,
        correctGradientLeak: Boolean = this.correctGradientLeak,
        smoothPolarization: Boolean = this.smoothPolarization,
        exposureStops: Float = this.exposureStops,
        gamma: Float = this.gamma,
        whiteBalance: WhiteBalance = this.whiteBalance,
        sensorRedRatio: Float = this.sensorRedRatio,
        sensorBlueRatio: Float = this.sensorBlueRatio,
        temperatureK: Float = this.temperatureK,
        tint: Float = this.tint,
    ) = StokesDisplay(
        modulateAolpByDolp = modulateAolpByDolp,
        debiasDolp = debiasDolp,
        horizonDeg = horizonDeg,
        rectify = rectify,
        correctGradientLeak = correctGradientLeak,
        smoothPolarization = smoothPolarization,
        exposureStops = exposureStops,
        gamma = gamma,
        whiteBalance = whiteBalance,
        sensorRedRatio = sensorRedRatio,
        sensorBlueRatio = sensorBlueRatio,
        temperatureK = temperatureK,
        tint = tint,
    )

    /** Multiplies the linear signal before the transfer curve. */
    internal val exposureGain: Float get() = 2f.pow(exposureStops)

    /**
     * Red and blue gains for [temperatureK] and [tint], relative to green.
     *
     * The temperature term follows the usual daylight approximation of the Planckian
     * locus, normalised so that [NEUTRAL_TEMPERATURE_K] leaves the frame alone.
     */
    internal fun temperatureGains(): FloatArray {
        val here = planckianRgb(temperatureK)
        val neutral = planckianRgb(NEUTRAL_TEMPERATURE_K)
        // A warmer target means less blue and more red, which is the inverse of the
        // illuminant's own colour.
        val gainR = (neutral[0] / here[0]).coerceIn(GAIN_MIN, GAIN_MAX)
        val gainB = (neutral[2] / here[2]).coerceIn(GAIN_MIN, GAIN_MAX)

        // Tint trades green against red and blue together.
        val trim = 1f + tint.coerceIn(-1f, 1f) * TINT_RANGE
        return floatArrayOf(gainR * trim, gainB * trim)
    }

    /** Approximate linear RGB of a black body at [kelvin], normalised to green. */
    private fun planckianRgb(kelvin: Float): FloatArray {
        val t = (kelvin.coerceIn(MIN_TEMPERATURE_K, MAX_TEMPERATURE_K) / 100f).toDouble()

        val red = if (t <= 66.0) 255.0 else 329.698727446 * (t - 60).pow(-0.1332047592)
        val green = if (t <= 66.0) {
            99.4708025861 * kotlin.math.ln(t) - 161.1195681661
        } else {
            288.1221695283 * (t - 60).pow(-0.0755148492)
        }
        val blue = when {
            t >= 66.0 -> 255.0
            t <= 19.0 -> 0.0
            else -> 138.5177312231 * kotlin.math.ln(t - 10) - 305.0447927307
        }

        val g = green.coerceIn(1.0, 255.0)
        return floatArrayOf(
            (red.coerceIn(1.0, 255.0) / g).toFloat(),
            1f,
            (blue.coerceIn(1.0, 255.0) / g).toFloat(),
        )
    }

    /**
     * Every setting that changes the picture, as one string.
     *
     * For callers that cache a rendered frame and need to know whether the cached one is
     * still the right one. Written to cover the whole class rather than a chosen few
     * fields, because a hand-written list is a list somebody will forget to add to: this
     * is exactly how the viewer came to keep showing an unstraightened neighbour after
     * the lens correction was switched on, having listed eight of the settings and not
     * the ninth.
     */
    val signature: String get() = listOf(
        modulateAolpByDolp, debiasDolp, horizonDeg, rectify, correctGradientLeak,
        smoothPolarization, exposureStops, gamma, whiteBalance, sensorRedRatio,
        sensorBlueRatio, temperatureK, tint,
    ).joinToString("|")

    companion object {
        /** What the preview and the stored renders use. */
        val DEFAULT = StokesDisplay()

        /** Linear light, no balance: the data as measured. */
        val LINEAR = StokesDisplay(
            gamma = 1f,
            whiteBalance = WhiteBalance.NONE,
            modulateAolpByDolp = false,
            debiasDolp = false,
            smoothPolarization = false,
        )

        const val DEFAULT_GAMMA = 2.2f

        /**
         * The calibrated balance, red and blue against green, for a capture whose sensor
         * ratios were at unity.
         *
         * Measured the way This Polar Camera measured its own: the per-capture grey-world
         * gains of a uniform sample of this archive, taken at the median. 111 captures
         * spread evenly from the first to the last, 79 of them usable once frames too
         * dark to carry a balance and frames with more than 5 percent of their samples at
         * full scale were dropped. The camera's own [WhiteBalanceCorrection.LEGACY] ratios
         * were divided out first, so these are in the unity basis and describe the sensor
         * rather than a setting it happened to be holding.
         *
         * The scatter between captures is the argument for using them. Blue's per-capture
         * gain runs from 1.9 to 3.0 across that sample, which is what grey-world does on
         * scenes that are not grey: a wall of one colour is corrected until the wall is
         * neutral, and the next capture of the same wall from a step to the left is
         * corrected differently. A constant is the same colour every time.
         *
         * Independent support: This Polar Camera derived 1.2465 and 2.2827 from 95 frames
         * of its own gallery on the same sensor. Two rigs, two scene sets, within 7
         * percent -- which is a good deal more confidence than either number has alone.
         */
        const val FIXED_WHITE_BALANCE_RED = 1.2319f
        const val FIXED_WHITE_BALANCE_BLUE = 2.1216f

        /** Daylight. Leaves the frame untouched by the temperature term. */
        const val NEUTRAL_TEMPERATURE_K = 6500f
        const val MIN_TEMPERATURE_K = 2000f
        const val MAX_TEMPERATURE_K = 12000f

        const val MIN_EXPOSURE_STOPS = -4f
        const val MAX_EXPOSURE_STOPS = 4f
        const val MIN_GAMMA = 1f
        const val MAX_GAMMA = 3f

        private const val GAIN_MIN = 0.25f
        private const val GAIN_MAX = 4f
        private const val TINT_RANGE = 0.3f
    }
}
