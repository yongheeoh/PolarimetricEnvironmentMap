package com.cgjnkim.mobileokad.flir

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * How much of a frame's measured polarization is the sensor rather than the scene.
 *
 * DoLP is a magnitude, `sqrt(S1^2 + S2^2) / S0`, and a magnitude built from noisy
 * components is biased upward: S1 and S2 each scatter around their true value with equal
 * probability either way, but squaring throws the sign away, so noise can only ever add
 * length to the vector. Where the scene is genuinely unpolarized the measurement still
 * reads a positive DoLP -- on this camera at 50 ms and 0 dB, 0.07 to 0.09 across the whole
 * of a dim frame, which is the same size as a real signal from a wet leaf.
 *
 * Radio astronomy met this first and named the distribution (Rice) and the fix: Wardle and
 * Kronberg's 1974 estimator subtracts the variance the noise contributes before taking the
 * square root,
 *
 *     p = sqrt(S1^2 + S2^2 - sigma^2),   zero when that is negative
 *
 * ### What this does and does not remove
 *
 * It is unbiased once the real polarization is a couple of sigma clear of the noise, which is
 * the regime that matters for a signal anyone would want to read. On a genuinely unpolarized
 * patch it does not reach zero: the magnitude is Rayleigh there, and taking
 * `sqrt(r^2 - sigma^2)` over that distribution leaves a mean of 0.76 sigma against the 1.25
 * sigma the raw magnitude carries -- a 39 percent cut in the floor, not its removal. What it
 * does do sharply is silence the pixels with the least to say: 39 percent of pure-noise
 * pixels fall at or below sigma and are reported as exactly zero.
 *
 * It is also not a denoiser. It shifts each pixel by what the noise is expected to contribute
 * without ever looking at its neighbours, and the literature puts that at roughly a factor of
 * two in mean squared error against the factor of 36 a spatial method like PBM3D reaches.
 * What it buys instead is that a dark frame stops claiming polarization it never measured,
 * for the cost of one number per pixel and no assumption about what the scene looks like.
 *
 * ### Where sigma comes from
 *
 * Nowhere external. The 4x4 super-pixel carries two green cells two pixels apart, each with
 * its own full set of four analyser angles, so they see the same scene through the same
 * optics -- and the difference between their S1 (or S2) is therefore noise, with any real
 * structure appearing only at edges. [measure] reads that difference across the frame and
 * fits the sensor's own noise model to it. No dark frame, no calibration target, and it is
 * measured on the very frame it corrects, so it follows exposure and gain automatically.
 *
 * The model is the usual one for a CMOS sensor: a single analyser reading of `I` counts
 * has variance
 *
 *     shotGain * I + readVariance
 *
 * the first term photon shot noise (variance proportional to signal), the second everything
 * that does not care how much light arrived. Two numbers, fitted per frame, and they
 * generalise across the bands -- red and blue have one cell each rather than green's two,
 * which is a factor in [stokesVariance] rather than a separate measurement.
 *
 * Nothing here touches what is stored. The archive keeps the raw mosaic and the raw Stokes
 * arrays; this is a viewing correction, and the fitted numbers travel in the capture
 * metadata so the gallery can make the same correction later or decline to.
 */
class StokesNoise(
    /** Variance added per count of signal: photon shot noise. */
    val shotGain: Float,
    /** Variance present with no signal at all: read noise, in counts squared. */
    val readVariance: Float,
    /** Super-pixels the fit was measured from, for judging how much to trust it. */
    val samples: Int = 0,
) {

    /**
     * Variance of S1, and equally of S2, for [band] at intensity [s0].
     *
     * S0 is the four analyser readings summed and halved, so a mean reading is `S0 / 2`.
     * S1 is a difference of two such readings and carries twice one reading's variance;
     * green averages two cells and so halves it again.
     *
     * The doubling assumes the two readings are independent, and adjacent photodiodes are
     * not quite -- they share charge, which makes their difference noisier than the sum of
     * their variances. It cancels: [CALIBRATED_SHOT_GAIN] is measured by differencing the
     * same adjacent pair, so whatever the correlation is, it is already inside the number
     * being doubled here.
     */
    fun stokesVariance(s0: Float, band: Int): Float {
        val meanReading = 0.5f * s0
        val perReading = shotGain * meanReading + readVariance
        if (perReading <= 0f) return 0f
        val cells = if (band == StokesImage.G) 2 else 1
        return 2f * perReading / cells
    }

    /**
     * The Wardle-Kronberg magnitude: the polarized part of the signal with the noise's
     * expected contribution taken back out.
     *
     * Returns zero rather than a small number when the measurement is entirely consistent
     * with an unpolarized scene, which is the estimator's point -- a pixel that cannot
     * support a claim should not make a faint one. Roughly two in five pixels of pure noise
     * land there.
     */
    fun debiasedMagnitude(s1: Float, s2: Float, s0: Float, band: Int): Float {
        val corrected = s1 * s1 + s2 * s2 - stokesVariance(s0, band)
        return if (corrected > 0f) sqrt(corrected) else 0f
    }

    /** The fit as it is written into the capture metadata. */
    fun toJson(): String =
        """{"model": "variance of one analyser reading = shotGain * reading + readVariance, in counts", """ +
            """"shotGain": %.6f, "readVariance": %.3f, "samples": %d}"""
                .format(shotGain, readVariance, samples)

    override fun toString(): String =
        "StokesNoise(shotGain=%.4f, readVariance=%.1f, samples=%d)"
            .format(shotGain, readVariance, samples)

    companion object {

        /**
         * The sensor's own shot-noise constant, which is what the correction should use.
         *
         * Variance per count of signal, at 12 bits and 0 dB. It is a property of the
         * sensor, not of the picture: a photon either arrives or does not, and how many
         * counts that is worth does not depend on what is in front of the lens. Measuring
         * it once and keeping it is therefore not a shortcut but the correct thing --
         * which matters here because [measure] cannot help measuring the scene as well.
         *
         * ### Where 0.38 comes from
         *
         * From 555 captures taken over one morning, of which 527 carry [measure]'s own
         * fit. Scene structure can only *add* variance, so the truth is the lower envelope
         * rather than the middle: across the well-exposed frames the fit bottoms out at
         * **0.385** and has a median of 0.45. The dark frames bottom out lower still, at
         * 0.27, but that is an artefact -- at a few counts of signal the median of an
         * integer difference is quantised to 1 or 2 and stops measuring anything.
         *
         * ### Why not 0.29, which a second estimator insists on
         *
         * Differencing one analyser against its twin in the *other* green cell reads
         * 1.4 to 1.5 times lower, and repeatably so, on every frame at every level. The
         * two are not measuring the same thing. That estimator differences pixels 2.8 px
         * apart, while S1 is a difference of *adjacent* pixels, and adjacent photodiodes
         * share charge: an electron generated between them lands in one or the other, so
         * their noise is anti-correlated and `var(I0 - I90)` genuinely exceeds
         * `var(I0) + var(I90)`.
         *
         * The one that matters is the one whose assumption cancels. [measure] differences
         * exactly the pixels S1 differences, so `var(S1diff) / 4` fed through
         * [stokesVariance]'s doubling returns `var(I0 - I90)` whatever the correlation
         * between neighbours -- while the 2.8 px estimator would only be right if there
         * were none. So the higher figure is the correct one, and the earlier 0.30 was
         * under-correcting by a quarter.
         *
         * A quadratic term was looked for and is not there -- the fit returns a negative
         * `I^2` coefficient -- so there is no measurable fixed-pattern noise and the two
         * terms below are the whole model.
         */
        const val CALIBRATED_SHOT_GAIN = 0.38f

        /**
         * Read noise, in counts squared, at 12 bits and 0 dB.
         *
         * Zero because it is below what any of these methods can see. Over 527 captures
         * the fitted read variance had a median of 0.13 and a third quartile of 1.2, and
         * shot noise passes that by the time a reading reaches four counts. The IMX250's
         * specified read noise is around 2.3 electrons, which is under one count here.
         */
        const val CALIBRATED_READ_VARIANCE = 0f

        /**
         * The noise to correct a frame with, for a capture at [gainDb] binned [binning] x.
         *
         * Analog gain multiplies signal and its shot noise together, so the constant
         * scales with the gain's linear factor. Summing super-pixels scales signal and
         * variance by the same factor too, which leaves the shot term alone and only
         * lifts the floor.
         */
        @JvmStatic
        @JvmOverloads
        fun calibrated(
            gainDb: Double = 0.0,
            binning: Int = 1,
            fullScale: Float = TWELVE_BIT_FULL_SCALE,
        ): StokesNoise {
            val linear = Math.pow(10.0, gainDb / 20.0).toFloat()
            val cells = (binning * binning).coerceAtLeast(1)
            // The constants are counts at 12 bits. Read the same light at 8 and every
            // count is sixteen times as much signal, so the variance per count falls by
            // the same factor -- and the variance itself, being in counts squared, by its
            // square. Without this an eight-bit frame would be told it is sixteen times
            // noisier than it is, and the DoLP debias would subtract most of the
            // polarization in the picture.
            val depth = (fullScale / TWELVE_BIT_FULL_SCALE).coerceAtLeast(MIN_DEPTH)
            return StokesNoise(
                CALIBRATED_SHOT_GAIN * linear * depth,
                CALIBRATED_READ_VARIANCE * linear * linear * cells * depth * depth,
                samples = 0,
            )
        }

        /** Full scale of one analyser reading at 12 bits, which the constants are in. */
        const val TWELVE_BIT_FULL_SCALE = 4095f

        /** A guard against a nonsense full scale silently zeroing the model. */
        private const val MIN_DEPTH = 1e-4f

        /**
         * Fits the noise model to [data] from the difference between the two green cells.
         *
         * ### This measures the scene as well as the sensor
         *
         * The two green cells are 2.8 px apart, and anything in the picture that changes
         * over 2.8 px lands in the difference along with the noise. Measured on the same
         * camera the fit runs from 0.34 on a blank wall to 5.5 on a bright street, a
         * factor of sixteen driven entirely by what was being photographed -- and since
         * structure can only *add* variance, the answer is always an upper bound.
         *
         * Saturated cells are dropped and only the flattest quarter of the frame is used,
         * which takes the street scene from 5.5 to 0.6, but that is a reduction and not a
         * cure. So this is a **diagnostic**, recorded beside a capture and useful for
         * spotting a camera whose noise has changed; the correction itself uses
         * [calibrated]. Deciding how hard to debias from the frame's own contents would
         * mean erasing the most polarization exactly where there is the most to see.
         *
         * Returns null when the frame gives nothing to fit -- too few super-pixels, or a
         * frame so uniform that every sample lands in one brightness bin with no scatter.
         *
         * @param step super-pixels between samples on each axis, or 0 to pick one. The fit
         *   is a handful of medians and wants a few thousand super-pixels, not five million,
         *   so the automatic step aims at [TARGET_SAMPLES] whatever the frame size -- which
         *   matters, because the preview runs at half resolution and a step chosen for a
         *   full frame would starve its bins.
         */
        @JvmStatic
        @JvmOverloads
        fun measure(
            data: ByteArray,
            width: Int,
            height: Int,
            pixelFormat: Long,
            step: Int = 0,
        ): StokesNoise? {
            val packed12 = StokesDecoder.isPacked12(pixelFormat)
            val expected =
                if (packed12) width.toLong() * height * 3 / 2 else width.toLong() * height
            if (data.size < expected) return null
            return measureFrom(width, height, StokesDecoder.maxValueOf(pixelFormat), step) { x, y ->
                px(data, width, x, y, packed12)
            }
        }

        /**
         * Fits the model to an expanded frame, binned or not.
         *
         * A summed frame needs no special case: summing scales signal and variance alike,
         * so the same two numbers describe it -- `shotGain` comes out unchanged and only
         * `readVariance` grows with the block. Refitting on the binned frame is therefore
         * both correct and the simplest thing to do.
         */
        @JvmStatic
        @JvmOverloads
        fun measure(mosaic: Mosaic, step: Int = 0): StokesNoise? {
            if (mosaic.samples.size < mosaic.width.toLong() * mosaic.height) return null
            return measureFrom(mosaic.width, mosaic.height, mosaic.maxValue, step) { x, y ->
                mosaic[x, y]
            }
        }

        /**
         * The fit itself, over whatever [sample] reads.
         *
         * Inline so the packed-byte and expanded-sample paths share this loop without
         * paying for a call per pixel: the frame is walked once and the sampler is the
         * only thing that differs between them.
         */
        private inline fun measureFrom(
            width: Int,
            height: Int,
            maxValue: Float,
            step: Int,
            sample: (x: Int, y: Int) -> Float,
        ): StokesNoise? {
            if (width % FlirCamera.MOSAIC != 0 || height % FlirCamera.MOSAIC != 0) return null

            val w = width / FlirCamera.MOSAIC
            val h = height / FlirCamera.MOSAIC
            val stride = if (step > 0) step else autoStep(w, h)

            // Collected first and binned second, because the roughness threshold is a
            // percentile of the frame and cannot be known until the frame has been read.
            val capacity = ((w + stride - 1) / stride) * ((h + stride - 1) / stride)
            val levels = FloatArray(capacity)
            val diffS1 = FloatArray(capacity)
            val diffS2 = FloatArray(capacity)
            val roughness = FloatArray(capacity)
            var total = 0
            val ceiling = maxValue * SATURATION_HEADROOM

            var sy = 0
            while (sy < h) {
                val baseY = sy * FlirCamera.MOSAIC
                var sx = 0
                while (sx < w) {
                    val baseX = sx * FlirCamera.MOSAIC

                    // The green pair: cells 1 and 2 of the super-pixel.
                    val a = StokesDecoder.CELLS[1]
                    val b = StokesDecoder.CELLS[2]
                    val a90 = sample(baseX + a[0] + StokesDecoder.D90_X, baseY + a[1] + StokesDecoder.D90_Y)
                    val a45 = sample(baseX + a[0] + StokesDecoder.D45_X, baseY + a[1] + StokesDecoder.D45_Y)
                    val a135 = sample(baseX + a[0] + StokesDecoder.D135_X, baseY + a[1] + StokesDecoder.D135_Y)
                    val a0 = sample(baseX + a[0] + StokesDecoder.D0_X, baseY + a[1] + StokesDecoder.D0_Y)
                    val b90 = sample(baseX + b[0] + StokesDecoder.D90_X, baseY + b[1] + StokesDecoder.D90_Y)
                    val b45 = sample(baseX + b[0] + StokesDecoder.D45_X, baseY + b[1] + StokesDecoder.D45_Y)
                    val b135 = sample(baseX + b[0] + StokesDecoder.D135_X, baseY + b[1] + StokesDecoder.D135_Y)
                    val b0 = sample(baseX + b[0] + StokesDecoder.D0_X, baseY + b[1] + StokesDecoder.D0_Y)

                    // A saturated reading has had its noise clipped off, and a pair of
                    // them differ by nothing at all -- which would read as a sensor that
                    // grows quieter the brighter it gets. On the street frames a fifth of
                    // the pixels are there.
                    val brightest = maxOf(
                        maxOf(maxOf(a90, a45), maxOf(a135, a0)),
                        maxOf(maxOf(b90, b45), maxOf(b135, b0)),
                    )
                    if (brightest < ceiling) {
                        val s0a = (a90 + a45 + a135 + a0) * 0.5f
                        val s0b = (b90 + b45 + b135 + b0) * 0.5f
                        levels[total] = (s0a + s0b) * 0.25f
                        diffS1[total] = abs((a0 - a90) - (b0 - b90))
                        diffS2[total] = abs((a45 - a135) - (b45 - b135))
                        // How much the picture itself changes over the very baseline the
                        // noise is being measured across. Choosing on it does not bias the
                        // result: a reading enters S0 with weight one half and S1 with
                        // plus or minus one, so the two are uncorrelated.
                        roughness[total] = abs(s0a - s0b)
                        total++
                    }

                    sx += stride
                }
                sy += stride
            }
            if (total < MIN_TOTAL_SAMPLES) return null

            // Keep the flattest part of the frame, but never so little that the bins
            // empty: on a small frame the whole thing is the sample.
            val keep = maxOf(ROUGHNESS_KEEP, MIN_KEPT.toFloat() / total).coerceAtMost(1f)
            val threshold = if (keep >= 1f) Float.MAX_VALUE else {
                val sorted = roughness.copyOf(total)
                sorted.sort()
                sorted[((total - 1) * keep).toInt()]
            }

            val diffs = Array(BINS) { FloatArray(BIN_CAPACITY) }
            val filled = IntArray(BINS)
            val counts = IntArray(BINS)
            val readingSums = DoubleArray(BINS)
            for (i in 0 until total) {
                if (roughness[i] > threshold) continue
                val bin = binOf(levels[i], maxValue)
                val room = filled[bin]
                if (room + 1 < BIN_CAPACITY) {
                    diffs[bin][room] = diffS1[i]
                    diffs[bin][room + 1] = diffS2[i]
                    filled[bin] = room + 2
                }
                counts[bin]++
                readingSums[bin] += levels[i].toDouble()
            }

            // One point per usable bin: the mean reading there, and the variance of a
            // single analyser reading implied by how far the two green cells disagree.
            val xs = DoubleArray(BINS)
            val ys = DoubleArray(BINS)
            val ws = DoubleArray(BINS)
            var used = 0
            for (bin in 0 until BINS) {
                if (filled[bin] < MIN_BIN_SAMPLES) continue
                val spread = medianAbsolute(diffs[bin], filled[bin]) / NORMAL_MEDIAN
                // The difference of two cells' S1 carries four readings' worth of variance.
                val perReading = spread.toDouble() * spread.toDouble() / 4.0
                if (perReading <= 0.0) continue
                xs[used] = readingSums[bin] / counts[bin]
                ys[used] = perReading
                // Weighted by count over the square of the estimate, because a median's
                // error is proportional to what it measures. Weighting by count alone lets
                // the bright bins, whose variance is an order of magnitude larger, settle
                // the whole line, and the dark end -- the only place the correction changes
                // a picture -- is then reached by extrapolation. In simulation that one
                // change took the worst error in predicted variance from 73% to 21%.
                ws[used] = counts[bin] / (perReading * perReading)
                used++
            }
            if (used == 0) return null

            return fit(xs, ys, ws, used, total)
        }

        /**
         * Weighted least squares of variance against signal, falling back as the data
         * allows.
         *
         * A frame with brightness at only one level cannot separate the two terms, and a
         * fit that runs negative is describing measurement scatter rather than a sensor;
         * both cases collapse to the flatter model instead of reporting a slope that was
         * never seen. Reporting a constant is the conservative answer -- it under-corrects
         * the bright end, which is where the bias barely matters anyway.
         */
        @PublishedApi
        internal fun fit(
            xs: DoubleArray,
            ys: DoubleArray,
            ws: DoubleArray,
            n: Int,
            samples: Int,
        ): StokesNoise? {
            var sw = 0.0; var swx = 0.0; var swy = 0.0; var swxx = 0.0; var swxy = 0.0
            var lowest = Double.MAX_VALUE
            var highest = -Double.MAX_VALUE
            for (i in 0 until n) {
                val w = ws[i]; val x = xs[i]; val y = ys[i]
                sw += w; swx += w * x; swy += w * y; swxx += w * x * x; swxy += w * x * y
                if (x < lowest) lowest = x
                if (x > highest) highest = x
            }
            if (sw <= 0.0) return null

            val flat = { StokesNoise(0f, (swy / sw).toFloat().coerceAtLeast(0f), samples) }
            if (n < 2 || highest - lowest < MIN_SIGNAL_SPREAD) return flat()

            val denominator = sw * swxx - swx * swx
            if (denominator <= 0.0) return flat()

            var slope = (sw * swxy - swx * swy) / denominator
            var intercept = (swy - slope * swx) / sw
            if (slope < 0.0) return flat()
            if (intercept < 0.0) {
                // Re-fit through the origin rather than keeping a negative floor, which
                // would have the model promise less noise than darkness itself carries.
                if (swxx <= 0.0) return flat()
                slope = swxy / swxx
                intercept = 0.0
                if (slope < 0.0) return flat()
            }
            return StokesNoise(slope.toFloat(), intercept.toFloat(), samples)
        }

        /**
         * Median of [count] absolute differences, scaled by [NORMAL_MEDIAN] by the caller.
         *
         * A median rather than a standard deviation, because the two green cells are two
         * pixels apart and an edge between them contributes a genuinely large difference
         * that is not noise. Those land in the tail; the median does not move.
         */
        @PublishedApi
        internal fun medianAbsolute(values: FloatArray, count: Int): Float {
            val slice = values.copyOf(count)
            slice.sort()
            return if (count % 2 == 1) {
                slice[count / 2]
            } else {
                0.5f * (slice[count / 2 - 1] + slice[count / 2])
            }
        }

        /**
         * Bins by the square root of signal, so the dark end -- where the correction
         * actually matters -- gets most of the resolution.
         */
        @PublishedApi
        internal fun binOf(reading: Float, maxValue: Float): Int {
            val t = sqrt((reading / maxValue).coerceIn(0f, 1f))
            return (t * BINS).toInt().coerceIn(0, BINS - 1)
        }

        private fun px(data: ByteArray, stride: Int, x: Int, y: Int, packed12: Boolean): Float =
            StokesDecoder.px(data, stride, x, y, packed12)

        /** No noise at all: what a synthetic frame has, and what disables the correction. */
        @JvmField
        val NONE = StokesNoise(0f, 0f)

        /** median(|x|) for a zero-mean normal is 0.6745 sigma. */
        @PublishedApi internal const val NORMAL_MEDIAN = 0.674490f

        /**
         * Brightness bins the frame is split into before a median is taken.
         *
         * Enough that the signal barely varies inside one. A wide bin mixes readings whose
         * noise genuinely differs, and the median of that mixture sits below the noise at
         * the bin's mean brightness rather than on it: at 8 bins every bin came out around
         * 7 percent low, which the line then turned into an intercept off by a factor of
         * seven.
         */
        @PublishedApi internal const val BINS = 32

        /** A median does not improve past a few hundred samples, and this is allocated per frame. */
        @PublishedApi internal const val BIN_CAPACITY = 512
        @PublishedApi internal const val MIN_BIN_SAMPLES = 64
        @PublishedApi internal const val MIN_TOTAL_SAMPLES = 256

        /** Counts of signal between the dimmest and brightest bin before a slope is believed. */
        private const val MIN_SIGNAL_SPREAD = 8.0

        /** A reading this close to full scale has been clipped and carries no noise. */
        private const val SATURATION_HEADROOM = 0.95f

        /** Fraction of the frame kept, flattest first. */
        private const val ROUGHNESS_KEEP = 0.25f

        /** Below this the whole frame is used rather than a quarter of very little. */
        private const val MIN_KEPT = 2000

        /** Super-pixels the automatic step aims to read, whatever the frame's size. */
        internal const val TARGET_SAMPLES = 8000

        @PublishedApi
        internal fun autoStep(w: Int, h: Int): Int =
            sqrt(w.toDouble() * h / TARGET_SAMPLES).toInt().coerceAtLeast(1)
    }
}
