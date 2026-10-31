package com.cgjnkim.mobileokad.flir

/**
 * A colour-polarization frame with one unsigned sample per pixel, still in the mosaic.
 *
 * The camera hands over Mono12p, two pixels to three bytes, which is the wrong shape for
 * anything but streaming. This is the same frame expanded: the sensor's own numbers, one
 * per pixel, in the sensor's own scale. It is what gets archived as a TIFF and what the
 * Stokes decode reads.
 */
class Mosaic(
    val samples: ShortArray,
    val width: Int,
    val height: Int,
    /** Largest value a sample can hold: 255 at 8 bits, 4095 at 12, more once summed. */
    val maxValue: Float,
) {
    /**
     * How sharp this frame is: fine detail over the detail one scale up.
     *
     * The public way in to a measure that lives with the preview's other mosaic maths.
     * Dividing one scale by the next is what makes the number comparable between scenes,
     * so a stored capture can be scored against the same thresholds the viewfinder uses.
     * See the implementation for what it is calibrated against.
     *
     * @return null when the frame is not a whole number of super-pixels, or has no detail
     */
    fun blurRatio(): Float? = MosaicPreview.blurRatio(this)

    /** One sample, unsigned -- a summed frame can pass the signed short's range. */
    operator fun get(x: Int, y: Int): Float =
        (samples[y * width + x].toInt() and 0xFFFF).toFloat()

    /**
     * Sums each [factor] x [factor] block of super-pixels into one, for a frame that is
     * smaller and quieter.
     *
     * ### Why this is done on the host
     *
     * Not because the camera cannot bin. BFS-U3-51S5PC exposes `BinningHorizontal`,
     * `BinningVertical` and a `BinningHorizontalMode` of Sum, all writable, and using them
     * would additionally cut the USB transfer by four, which this cannot. It is done here
     * because what that firmware bins is undocumented for a polarization sensor, and the
     * failure mode is silent.
     *
     * The concern is concrete. Teledyne document that on a colour model binning turns
     * BayerRG16 into BayerGB16 -- the pattern survives and only its phase moves, so their
     * binning combines pixels *two apart* to keep each colour intact. On this sensor two
     * apart is exactly the analyser period: a bin that is careful about the Bayer grid and
     * unaware of the polarizer would add 0 to 90 degrees and 45 to 135, and hand back a
     * frame that still looks like a mosaic with the polarization averaged out of it.
     * Whether the P firmware knows better has not been tested here.
     *
     * So this takes the sum between super-pixels rather than inside them: sample (dx, dy)
     * of one super-pixel is added to sample (dx, dy) of its neighbours, which are the same
     * colour behind the same analyser angle looking at almost the same place. The result is
     * a valid mosaic of the same layout at half the width and height with four times the
     * counts, so nothing downstream changes -- and it is correct by construction, which the
     * camera's version is not yet known to be.
     *
     * ### What it buys
     *
     * Four readings summed have four times the signal and four times the variance, so the
     * noise grows by two while the signal grows by four: a factor of two in signal to
     * noise, which for DoLP means the floor `0.71 / sqrt(S0)` halves. Measured on a real
     * capture, the darkest tenth of the frame went from a DoLP of 0.073 to 0.046.
     *
     * Summing rather than averaging keeps it exact -- four integers add to an integer,
     * where a mean would round two bits away -- and leaves the noise model unchanged:
     * shot-noise variance is proportional to signal, so summing scales both by four and
     * [StokesNoise]'s `shotGain` comes out the same on a binned frame as on a full one.
     *
     * The cost is resolution, and it is a real cost: this is a low-pass on the polarization
     * field, so where the angle turns inside one block the true polarization partly cancels
     * along with the noise. It belongs behind a deliberate choice, not on by default.
     *
     * @throws U3vException when the frame does not divide evenly, or when the sum would
     *   not fit in the 16 bits the archive format stores
     */
    fun binnedSum(factor: Int): Mosaic {
        if (factor <= 1) return this

        val group = FlirCamera.MOSAIC * factor
        if (width % group != 0 || height % group != 0) {
            throw U3vException(
                "a ${factor}x bin needs ${width}x$height to be a multiple of $group"
            )
        }
        val peak = maxValue * factor * factor
        if (peak > UNSIGNED_16_MAX) {
            throw U3vException(
                "a ${factor}x bin would reach ${peak.toInt()}, past the ${UNSIGNED_16_MAX.toInt()} a sample holds"
            )
        }

        val outWidth = width / factor
        val outHeight = height / factor
        val out = ShortArray(outWidth * outHeight)

        for (oy in 0 until outHeight) {
            // A pixel's place in the mosaic is its offset inside the 4x4 super-pixel, and
            // that has to survive: the sample being summed must stay the same colour and
            // the same analyser angle.
            val superY = oy / FlirCamera.MOSAIC
            val cellY = oy % FlirCamera.MOSAIC
            val outRow = oy * outWidth
            for (ox in 0 until outWidth) {
                val superX = ox / FlirCamera.MOSAIC
                val cellX = ox % FlirCamera.MOSAIC

                var sum = 0
                for (j in 0 until factor) {
                    val y = (superY * factor + j) * FlirCamera.MOSAIC + cellY
                    val row = y * width
                    for (i in 0 until factor) {
                        val x = (superX * factor + i) * FlirCamera.MOSAIC + cellX
                        sum += samples[row + x].toInt() and 0xFFFF
                    }
                }
                out[outRow + ox] = sum.toShort()
            }
        }
        return Mosaic(out, outWidth, outHeight, peak)
    }

    companion object {
        private const val UNSIGNED_16_MAX = 65535f
    }
}
