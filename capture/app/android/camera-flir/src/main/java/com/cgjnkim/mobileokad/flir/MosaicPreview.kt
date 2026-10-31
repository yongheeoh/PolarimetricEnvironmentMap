package com.cgjnkim.mobileokad.flir

import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Renders one view straight from the raw mosaic, in a single pass.
 *
 * The capture path goes mosaic -> [StokesImage] -> view, which is right for a capture:
 * every band and component is wanted, and the arrays are saved. A preview wants one
 * view of one band, so that route computes nine planes to read two of them and walks
 * the frame twice with a few megabytes of short-lived arrays in between.
 *
 * This is the fused equivalent: it reads each sample once, keeps only what the chosen
 * view needs, and writes ARGB. The single-channel views need only the two green cells,
 * so they touch half the samples. The idea and the hue lookup table are taken from
 * polar_camera's `render_mosaic_polarization_rgb` kernels.
 *
 * The polarization views match the two-stage path exactly, except that AoLP hues are
 * quantised to the lookup table's resolution. S0 RGB deliberately does not: it is white
 * balanced and gamma encoded for display, so that the FLIR preview reads like the OAK's
 * beside it. Nothing that is stored goes through here -- the saved PNGs come from
 * [StokesRender] and the archive arrays from [StokesDecoder], both still linear and
 * unbalanced.
 */
internal object MosaicPreview {

    /**
     * Hue steps across the 180 degrees AoLP spans.
     *
     * A preview pixel is about a third of a degree wide in angle at this size, which is
     * finer than the noise on the measurement, and it removes the per-pixel branch and
     * arithmetic of an HSV conversion.
     */
    private const val HUE_STEPS = 256

    private val HUE_LUT = IntArray(HUE_STEPS) { hueToArgb(it.toFloat() / HUE_STEPS) }

    /**
     * @param out reused when it is already the right size, so a preview loop running at
     *   display rate does not allocate a new frame buffer each time
     * @param step read only every nth super-pixel in each direction. The output shrinks
     *   with it while the field of view does not, which is how a full-sensor preview
     *   stays affordable: the whole frame still crosses the wire, but the decode walks a
     *   quarter of it at step 2. The mosaic is sampled, never mixed, so every view stays
     *   exactly what it was, only coarser.
     */
    /**
     * Sensor pixels per output pixel at [step].
     *
     * Public because a caller has to size a bitmap to what [render] produces, and working
     * that out separately is how the two came apart: at step 2 the preview loop still
     * sized itself at one pixel per super-pixel, asked for four times the pixels the
     * buffer held, and took the UI thread down with it.
     */
    fun pitch(step: Int): Int = FlirCamera.MOSAIC * step.coerceAtLeast(1)

    fun render(
        data: ByteArray,
        width: Int,
        height: Int,
        pixelFormat: Long,
        view: StokesView,
        out: IntArray? = null,
        modulateAolpByDolp: Boolean = true,
        noise: StokesNoise? = null,
        step: Int = 1,
        sumWithinStep: Boolean = false,
        zebra: Boolean = false,
    ): IntArray {
        if (width % FlirCamera.MOSAIC != 0 || height % FlirCamera.MOSAIC != 0) {
            throw U3vException("frame ${width}x$height is not a multiple of ${FlirCamera.MOSAIC}")
        }
        val packed12 = StokesDecoder.isPacked12(pixelFormat)
        val expected = if (packed12) width.toLong() * height * 3 / 2 else width.toLong() * height
        if (data.size < expected) {
            throw U3vException("frame data ${data.size} B is short of $expected")
        }

        // Whole super-pixels, so a step never lands the reader on a different analyser
        // angle: skipping is only ever a coarser sampling of the same mosaic, never a
        // different one.
        val pitch = pitch(step)
        val w = width / pitch
        val h = height / pitch
        // Only meaningful when the step is skipping something.
        val blocks = if (sumWithinStep) step.coerceAtLeast(1) else 1
        val pixels = if (out != null && out.size == w * h) out else IntArray(w * h)

        if (view == StokesView.S0_RGB) {
            renderS0Rgb(
                data, width, w, h, pitch, packed12,
                StokesDecoder.maxValueOf(pixelFormat), pixels, blocks,
            )
        } else {
            renderGreenBand(
                data, width, w, h, pitch, packed12, view, pixels, modulateAolpByDolp, noise, blocks,
            )
        }
        if (zebra) {
            stripeClipped(data, width, w, h, pitch, packed12, StokesDecoder.maxValueOf(pixelFormat), pixels)
        }
        return pixels
    }

    /**
     * Intensity only, so each cell needs a sum rather than the full Stokes vector.
     *
     * Two display corrections are applied on the way out, and only here:
     *
     *  - grey-world white balance, because the sensor's raw bands are nothing like
     *    balanced (the green cells alone carry twice the sampling) and the frame comes
     *    out heavily green next to the OAK's ISP-corrected picture;
     *  - an sRGB transfer curve, because S0 is linear light while the OAK preview is
     *    already gamma encoded, which is most of what makes the raw version look dark
     *    and contrasty beside it.
     *
     * Both are display-only. What gets written to storage never passes through this
     * function.
     */
    private fun renderS0Rgb(
        data: ByteArray,
        stride: Int,
        w: Int,
        h: Int,
        pitch: Int,
        packed12: Boolean,
        maxValue: Float,
        out: IntArray,
        blocks: Int,
    ) {
        // S0 is the four analysers halved, so dividing by full scale puts an unpolarized
        // scene back on a 0..1 scale. Deliberately not divided by the number of
        // super-pixels summed: the extra signal is the point, and dividing it away would
        // leave a less noisy picture of the same brightness rather than a brighter one.
        // A bright scene therefore clips, which is why this is a switch and not a default.
        val norm = 0.5f / maxValue
        val gains = grayWorldGains(data, stride, w, h, pitch, packed12)
        val gainR = gains[0]
        val gainB = gains[1]

        Parallel.over(h, Parallel.MIN_ROWS_PER_SLICE) { from, to ->
            for (sy in from until to) {
                val baseY = sy * pitch
                val row = sy * w
                for (sx in 0 until w) {
                    val baseX = sx * pitch
                    var r = 0f
                    var g = 0f
                    var b = 0f
                    for (cell in StokesDecoder.CELLS.indices) {
                        val sum = cellValue(data, stride, baseX + StokesDecoder.CELLS[cell][0],
                            baseY + StokesDecoder.CELLS[cell][1], packed12, blocks)
                        when (cell) {
                            0 -> r = sum
                            3 -> b = sum
                            else -> g += 0.5f * sum   // two green cells, averaged
                        }
                    }
                    out[row + sx] = argb(
                        encode(r * norm * gainR),
                        encode(g * norm),
                        encode(b * norm * gainB),
                    )
                }
                    }
        }
    }

    /**
     * Mean S0 of the green cells, in sensor counts, over a coarse grid.
     *
     * The green cells because they are two of the four and the brightest, so they say
     * soonest whether there is enough light for the polarization to mean anything. A
     * scene at S0 = 100 counts carries a DoLP noise floor of about 0.05 -- five points of
     * polarization that are not there -- which is the number the darkness warning is set
     * against; [StokesNoise] is where that arithmetic lives.
     *
     * Sampled one super-pixel in [grid] each way: a mean over three hundred cells is
     * already far steadier than the thing it measures, and this runs inside the preview
     * loop where the whole frame decode is the budget.
     *
     * @return mean S0 in counts, or null when the frame cannot be read as a mosaic
     */
    fun meanS0(
        data: ByteArray,
        width: Int,
        height: Int,
        pixelFormat: Long,
        grid: Int = LEVEL_GRID,
    ): Float? {
        // Null rather than zero on a frame this cannot read. Zero is what a dark scene
        // measures, so returning it for a truncated transfer would put a "too dark"
        // warning over a bright one -- the failure would look like a reading.
        if (width % FlirCamera.MOSAIC != 0 || height % FlirCamera.MOSAIC != 0) return null
        val packed12 = StokesDecoder.isPacked12(pixelFormat)
        val expected = if (packed12) width.toLong() * height * 3 / 2 else width.toLong() * height
        if (data.size < expected) return null

        val pitch = pitch(grid)
        var sum = 0.0
        var n = 0
        var y = 0
        while (y + FlirCamera.MOSAIC <= height) {
            var x = 0
            while (x + FlirCamera.MOSAIC <= width) {
                // Both green cells, so the reading does not depend on which of the two
                // the grid happens to land on.
                sum += cellSum(data, width, x + StokesDecoder.CELLS[1][0],
                    y + StokesDecoder.CELLS[1][1], packed12)
                sum += cellSum(data, width, x + StokesDecoder.CELLS[2][0],
                    y + StokesDecoder.CELLS[2][1], packed12)
                n += 2
                x += pitch
            }
            y += pitch
        }
        return if (n == 0) null else (sum / n).toFloat()
    }

    /**
     * The brightest sample in a frame, and how much of the frame is already clipped.
     *
     * What the host meter reads. See [AutoExposure] for why the ceiling rather than the
     * average is the thing an auto exposure has to protect here.
     *
     * ### Why individual pixels rather than cells
     *
     * [meanS0] works in cell sums because brightness is a property of the super-pixel.
     * Saturation is not: it happens to one photosite, and a single analyser angle pinned
     * to full scale is exactly the failure being looked for -- the cell it belongs to
     * still sums to something unremarkable, which is how a clipped polarized highlight
     * hides from any cell-level statistic.
     *
     * ### Why every angle and colour, and only a quarter of the frame
     *
     * Whole super-pixels are visited, so all four analyser angles and all four Bayer
     * cells are represented equally and the reading cannot favour the angle the grid
     * happened to land on. Visiting every [grid]-th super-pixel in each axis then reads
     * one pixel in [grid] squared -- a quarter of the frame at the default, the same
     * budget the preview render already works to.
     *
     * A specular smaller than [grid] super-pixels can slip between the samples. It clips
     * too few pixels to pass [AutoExposure.CLIP_TOLERANCE] anyway, so the sampling and
     * the tolerance are refusing the same frames for the same reason.
     *
     * @return null when the frame cannot be read as a mosaic, never a zeroed reading:
     *   zero peak is what a black frame measures, and a truncated transfer reported as
     *   one would drive the exposure straight to the ceiling.
     */
    fun meter(
        data: ByteArray,
        width: Int,
        height: Int,
        pixelFormat: Long,
        grid: Int = HIGHLIGHT_GRID,
    ): AutoExposure.Reading? {
        if (width % FlirCamera.MOSAIC != 0 || height % FlirCamera.MOSAIC != 0) return null
        val packed12 = StokesDecoder.isPacked12(pixelFormat)
        val expected = if (packed12) width.toLong() * height * 3 / 2 else width.toLong() * height
        if (data.size < expected) return null

        val maxValue = StokesDecoder.maxValueOf(pixelFormat)
        if (maxValue <= 0f) return null
        val saturation = SATURATION * maxValue

        val m = FlirCamera.MOSAIC
        val pitch = pitch(grid)
        var peak = 0f
        var clipped = 0L
        var total = 0.0
        var n = 0L
        var y = 0
        while (y + m <= height) {
            var x = 0
            while (x + m <= width) {
                for (dy in 0 until m) {
                    for (dx in 0 until m) {
                        val value = px(data, width, x + dx, y + dy, packed12)
                        if (value > peak) peak = value
                        if (value >= saturation) clipped++
                        total += value.toDouble()
                        n++
                    }
                }
                x += pitch
            }
            y += pitch
        }
        if (n == 0L) return null
        return AutoExposure.Reading(
            peak = peak / maxValue,
            clipped = clipped.toFloat() / n,
            mean = (total / n / maxValue).toFloat(),
        )
    }

    /**
     * How much fine detail a frame holds, for choosing between frames of the same scene.
     *
     * Mean squared difference between horizontally and vertically adjacent green cells,
     * divided by the square of the mean level. Blur is the loss of exactly this: the
     * difference between neighbouring samples is what a smear averages away, and the
     * neighbours here are one super-pixel apart, which is the Stokes pixel grid -- the
     * same scale the motion estimate is expressed in.
     *
     * ### What it is and is not for
     *
     * Only comparable **between frames of the same scene**, which is why it exists for
     * bursts and nothing else. A textured wall scores higher than a blank one at any
     * sharpness, so this cannot say whether a single frame is blurred; across four frames
     * taken in under a second it says which of them is sharpest, and the scene cancels.
     *
     * Dividing by the mean squared removes exposure, so a burst that brackets or a scene
     * that changes brightness mid-burst does not tilt the comparison toward the brighter
     * frames. It does not remove everything: a frame that reframes onto a plainer part of
     * the scene scores lower for a reason that is not blur, which is the honest limit of
     * picking this way and the reason every frame's score is recorded rather than only
     * the winner's.
     */
    fun sharpness(data: ByteArray, width: Int, height: Int, pixelFormat: Long): Float {
        if (width % FlirCamera.MOSAIC != 0 || height % FlirCamera.MOSAIC != 0) return 0f
        val packed12 = StokesDecoder.isPacked12(pixelFormat)
        val expected = if (packed12) width.toLong() * height * 3 / 2 else width.toLong() * height
        if (data.size < expected) return 0f

        val m = FlirCamera.MOSAIC
        val cols = width / m
        val rows = height / m
        if (cols < 2 || rows < 2) return 0f

        // One row of green-cell values at a time, so the whole frame is walked once and
        // only two rows are held: a 5 MP frame is 313k cells and building them all first
        // would be another 1.2 MB for no reason.
        var previous = FloatArray(cols)
        var current = FloatArray(cols)
        var sum = 0.0
        var energy = 0.0
        var pairs = 0L

        for (ry in 0 until rows) {
            val baseY = ry * m
            for (rx in 0 until cols) {
                current[rx] = cellSum(
                    data, width,
                    rx * m + StokesDecoder.CELLS[1][0],
                    baseY + StokesDecoder.CELLS[1][1],
                    packed12,
                )
                sum += current[rx]
            }
            for (rx in 1 until cols) {
                val d = current[rx] - current[rx - 1]
                energy += d.toDouble() * d
                pairs++
            }
            if (ry > 0) {
                for (rx in 0 until cols) {
                    val d = current[rx] - previous[rx]
                    energy += d.toDouble() * d
                    pairs++
                }
            }
            val swap = previous
            previous = current
            current = swap
        }

        val mean = sum / (rows.toLong() * cols)
        if (mean <= 0.0 || pairs == 0L) return 0f
        return ((energy / pairs) / (mean * mean)).toFloat()
    }

    /**
     * A plain brightness grid at the preview's own pitch, for frame-to-frame matching.
     *
     * The two green cells summed, one value per output pixel, no white balance and no
     * transfer curve -- [FrameDrift] correlates these against the previous frame's and
     * only wants a stable, linear measure of how bright each patch is.
     *
     * @return null when the frame cannot be read as a mosaic
     */
    fun lumaGrid(
        data: ByteArray,
        width: Int,
        height: Int,
        pixelFormat: Long,
        step: Int,
    ): FloatArray? {
        if (width % FlirCamera.MOSAIC != 0 || height % FlirCamera.MOSAIC != 0) return null
        val packed12 = StokesDecoder.isPacked12(pixelFormat)
        val expected = if (packed12) width.toLong() * height * 3 / 2 else width.toLong() * height
        if (data.size < expected) return null
        val pitch = pitch(step)
        val w = width / pitch
        val h = height / pitch
        if (w < 1 || h < 1) return null
        val out = FloatArray(w * h)
        for (sy in 0 until h) {
            val baseY = sy * pitch
            for (sx in 0 until w) {
                val baseX = sx * pitch
                out[sy * w + sx] =
                    cellSum(data, width, baseX + StokesDecoder.CELLS[1][0], baseY + StokesDecoder.CELLS[1][1], packed12) +
                    cellSum(data, width, baseX + StokesDecoder.CELLS[2][0], baseY + StokesDecoder.CELLS[2][1], packed12)
            }
        }
        return out
    }

    /**
     * Whether a frame is blurred, from the frame alone.
     *
     * Returns the ratio of detail at the super-pixel scale to detail at twice that
     * scale. Blur removes the fine scale first, so a smeared frame has already lost most
     * of what this divides by and the ratio falls; a sharp frame keeps it and the ratio
     * is high. Low means blurred.
     *
     * ### Why a ratio, and not [sharpness]
     *
     * The obvious metric -- how much fine detail is there -- cannot tell a blurred frame
     * from a photograph of a blank wall, so it only works within a burst where the scene
     * is fixed. Dividing by the detail at the next scale up cancels the scene: a plain
     * wall has little of either and the ratio survives, while a smear takes the numerator
     * away and leaves the denominator. That is what makes this usable on a single frame
     * of an unknown subject, which is what a live warning needs.
     *
     * ### What it is calibrated against
     *
     * 172 captures reviewed frame by frame, 27 rejected as blurred and 145 kept, all with
     * a gyroscope record. Separating those two piles:
     *
     *  - this ratio, AUC 0.95
     *  - edge width, 0.92
     *  - predicted smear from the gyroscope, 0.81
     *  - plain fine-scale detail, 0.75
     *
     * At a matched 80 percent of rejected frames caught, the gyroscope raises 44 false
     * warnings on the 145 frames that were kept and this raises 11. The frames that made
     * the difference are the ones a person kept while the gyroscope called them blurred:
     * on this measure they sit with the rest of the kept frames, which is the evidence
     * that the gyroscope was wrong about them rather than the judgement.
     *
     * @return the ratio, or null when the frame cannot be read as a mosaic
     */
    fun blurRatio(data: ByteArray, width: Int, height: Int, pixelFormat: Long): Float? {
        if (width % FlirCamera.MOSAIC != 0 || height % FlirCamera.MOSAIC != 0) return null
        val packed12 = StokesDecoder.isPacked12(pixelFormat)
        val expected = if (packed12) width.toLong() * height * 3 / 2 else width.toLong() * height
        if (data.size < expected) return null
        val m = FlirCamera.MOSAIC
        return blurRatioOf(width / m, height / m) { row, out ->
            readGreenRow(data, width, row, out.size, packed12, out)
        }
    }

    /**
     * The same measure on a mosaic already unpacked from an archived TIFF.
     *
     * The gallery scores stored captures with it, so that what a frame is judged by hours
     * later is the same number the viewfinder judged it by at the shutter -- one
     * implementation rather than two that will eventually disagree.
     */
    fun blurRatio(mosaic: Mosaic): Float? {
        val m = FlirCamera.MOSAIC
        if (mosaic.width % m != 0 || mosaic.height % m != 0) return null
        val baseX = StokesDecoder.CELLS[1][0]
        val baseY = StokesDecoder.CELLS[1][1]
        return blurRatioOf(mosaic.width / m, mosaic.height / m) { row, out ->
            val y = row * m + baseY
            for (rx in out.indices) out[rx] = mosaic[rx * m + baseX, y]
        }
    }

    /**
     * Fine detail over the detail one scale up, given a way to read one row of green
     * cells at a time.
     *
     * @param readRow fills the array with the green cell of every super-pixel in that row
     */
    private inline fun blurRatioOf(cols: Int, rows: Int, readRow: (Int, FloatArray) -> Unit): Float? {
        if (cols < 4 || rows < 4) return null

        val top = FloatArray(cols)
        val bottom = FloatArray(cols)
        val halfCols = cols / 2
        var coarsePrev: FloatArray? = null
        val coarse = FloatArray(halfCols)

        var fine = 0.0
        var finePairs = 0L
        var coarseEnergy = 0.0
        var coarsePairs = 0L

        // Two super-pixel rows at a time: they give the fine differences directly and
        // average down into one row of the coarse grid, so the whole frame is walked once
        // and nothing bigger than a row is held.
        var ry = 0
        while (ry + 1 < rows) {
            readRow(ry, top)
            readRow(ry + 1, bottom)

            for (rx in 1 until cols) {
                val a1 = top[rx] - top[rx - 1]
                val b1 = bottom[rx] - bottom[rx - 1]
                fine += a1.toDouble() * a1 + b1.toDouble() * b1
                finePairs += 2
            }
            for (rx in 0 until cols) {
                val d = bottom[rx] - top[rx]
                fine += d.toDouble() * d
                finePairs++
            }

            for (hx in 0 until halfCols) {
                val x = hx * 2
                coarse[hx] = 0.25f * (top[x] + top[x + 1] + bottom[x] + bottom[x + 1])
            }
            for (hx in 1 until halfCols) {
                val d = coarse[hx] - coarse[hx - 1]
                coarseEnergy += d.toDouble() * d
                coarsePairs++
            }
            coarsePrev?.let { prev ->
                for (hx in 0 until halfCols) {
                    val d = coarse[hx] - prev[hx]
                    coarseEnergy += d.toDouble() * d
                    coarsePairs++
                }
            }
            coarsePrev = coarse.copyOf()
            ry += 2
        }

        if (finePairs == 0L || coarsePairs == 0L) return null
        val coarseMean = coarseEnergy / coarsePairs
        // A frame with no detail at either scale -- a lens cap, a blown white wall -- has
        // no ratio to report, and reporting one would be inventing a verdict.
        if (coarseMean <= 0.0) return null
        return ((fine / finePairs) / coarseMean).toFloat()
    }

    private fun readGreenRow(
        data: ByteArray,
        stride: Int,
        row: Int,
        cols: Int,
        packed12: Boolean,
        out: FloatArray,
    ) {
        val baseY = row * FlirCamera.MOSAIC + StokesDecoder.CELLS[1][1]
        for (rx in 0 until cols) {
            out[rx] = cellSum(data, stride, rx * FlirCamera.MOSAIC + StokesDecoder.CELLS[1][0], baseY, packed12)
        }
    }

    /** One super-pixel in eight each way: about 300 cells on this sensor. */
    private const val LEVEL_GRID = 8

    /**
     * Every second super-pixel, which reads a quarter of the frame's pixels.
     *
     * Denser than [LEVEL_GRID] because the two measure different things. A mean over a
     * coarse grid is already steadier than the quantity it estimates; a maximum is not
     * estimated by sampling at all, only bounded from below, so every sample skipped is
     * a highlight that might have been missed.
     */
    /** One super-pixel in two each way, so the meter reads a quarter of the frame. */
    internal const val HIGHLIGHT_GRID = 2

    /**
     * Where a sample is called saturated, as a fraction of full scale.
     *
     * Under full scale on purpose. A photosite's response flattens before it reaches the
     * top code, so a sample at 98 percent has already stopped being proportional to the
     * light -- and proportionality, not the code value, is what the Stokes arithmetic
     * needs from it.
     */
    const val SATURATION = 0.98f

    /**
     * Red and blue gains that put the frame's channel averages onto green's.
     *
     * Grey-world: it assumes the scene averages to neutral, which is wrong for a frame
     * filled with one colour, so the gains are clamped rather than trusted outright. It
     * needs no calibration target, which a fixed matrix would.
     *
     * Sampled on a coarse grid: a mean does not need every super-pixel, and reading one
     * in sixteen keeps this well under the cost of the render itself.
     */
    private fun grayWorldGains(
        data: ByteArray,
        stride: Int,
        w: Int,
        h: Int,
        pitch: Int,
        packed12: Boolean,
    ): FloatArray {
        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var n = 0

        var sy = 0
        while (sy < h) {
            val baseY = sy * pitch
            var sx = 0
            while (sx < w) {
                val baseX = sx * pitch
                sumR += cellSum(data, stride, baseX + StokesDecoder.CELLS[0][0],
                    baseY + StokesDecoder.CELLS[0][1], packed12)
                sumB += cellSum(data, stride, baseX + StokesDecoder.CELLS[3][0],
                    baseY + StokesDecoder.CELLS[3][1], packed12)
                sumG += 0.5f * (
                    cellSum(data, stride, baseX + StokesDecoder.CELLS[1][0],
                        baseY + StokesDecoder.CELLS[1][1], packed12) +
                        cellSum(data, stride, baseX + StokesDecoder.CELLS[2][0],
                            baseY + StokesDecoder.CELLS[2][1], packed12)
                    )
                n++
                sx += WB_SAMPLE_STEP
            }
            sy += WB_SAMPLE_STEP
        }

        if (n == 0 || sumR <= EPSILON || sumB <= EPSILON || sumG <= EPSILON) {
            return floatArrayOf(1f, 1f)
        }
        return floatArrayOf(
            (sumG / sumR).toFloat().coerceIn(WB_GAIN_MIN, WB_GAIN_MAX),
            (sumG / sumB).toFloat().coerceIn(WB_GAIN_MIN, WB_GAIN_MAX),
        )
    }

    /** Linear 0..1 to an 8 bit sRGB level, through a table to keep pow() off the loop. */
    private fun encode(linear: Float): Int =
        SRGB_LUT[(linear.coerceIn(0f, 1f) * (SRGB_LUT_SIZE - 1)).toInt()]

    /** The polarization views are all read from green, the band with two cells. */
    private fun renderGreenBand(
        data: ByteArray,
        stride: Int,
        w: Int,
        h: Int,
        pitch: Int,
        packed12: Boolean,
        view: StokesView,
        out: IntArray,
        modulateAolpByDolp: Boolean,
        noise: StokesNoise?,
        blocks: Int,
    ) {
        Parallel.over(h, Parallel.MIN_ROWS_PER_SLICE) { from, to ->
            for (sy in from until to) {
                val baseY = sy * pitch
                val row = sy * w
                for (sx in 0 until w) {
                    val baseX = sx * pitch
                    var s0 = 0f
                    var s1 = 0f
                    var s2 = 0f

                    // Cells 1 and 2 are the green pair; each contributes half the band.
                    for (cell in intArrayOf(1, 2)) {
                        val cx = baseX + StokesDecoder.CELLS[cell][0]
                        val cy = baseY + StokesDecoder.CELLS[cell][1]
                        val i90 = analyser(data, stride, cx, cy, StokesDecoder.D90_X, StokesDecoder.D90_Y, packed12, blocks)
                        val i45 = analyser(data, stride, cx, cy, StokesDecoder.D45_X, StokesDecoder.D45_Y, packed12, blocks)
                        val i135 = analyser(data, stride, cx, cy, StokesDecoder.D135_X, StokesDecoder.D135_Y, packed12, blocks)
                        val i0 = analyser(data, stride, cx, cy, StokesDecoder.D0_X, StokesDecoder.D0_Y, packed12, blocks)

                        s0 += 0.5f * (i0 + i45 + i90 + i135) * 0.5f
                        s1 += 0.5f * (i0 - i90)
                        s2 += 0.5f * (i45 - i135)
                    }

                    out[row + sx] = when (view) {
                        StokesView.AOLP -> {
                            // A full turn of atan2 is exactly the half turn of analyser
                            // angle the wheel spans, so the fraction indexes the table
                            // directly -- and it is approximated, because the exact call was
                            // the most expensive thing in the preview loop. See turnOf.
                            val hue = HUE_LUT[(turnOf(s2, s1) * HUE_STEPS).toInt() % HUE_STEPS]
                            // How far the hue is pushed from grey says how much polarization
                            // the angle was measured from. Where there is none the angle is
                            // undefined, and drawing it at full strength anyway is what makes
                            // a dim scene look like confetti.
                            if (!modulateAolpByDolp) hue else fadeToWhite(hue, dolp(s0, s1, s2, noise))
                        }
                        StokesView.DOLP -> {
                            val v = clamp8(dolp(s0, s1, s2, noise) * 255f)
                            argb(v, v, v)
                        }
                        StokesView.S1 -> diverging(normalize(s1, s0))
                        StokesView.S2 -> diverging(normalize(s2, s0))
                        StokesView.S0_RGB -> throw IllegalStateException("handled by renderS0Rgb")
                    }
                }
                    }
        }
    }

    /**
     * Degree of linear polarization, with the noise's own contribution removed when a fit
     * for this frame was supplied.
     *
     * Always the green band here: the fused kernel reads the two green cells and nothing
     * else, which is also why green's variance is the one [StokesNoise] is asked for.
     */
    private fun dolp(s0: Float, s1: Float, s2: Float, noise: StokesNoise?): Float {
        if (s0 <= EPSILON) return 0f
        val magnitude = noise?.debiasedMagnitude(s1, s2, s0, StokesImage.G)
            ?: sqrt(s1 * s1 + s2 * s2)
        return (magnitude / s0).coerceIn(0f, 1f)
    }

    /**
     * Paints diagonal stripes over every output pixel whose footprint contains a sample
     * at or past saturation.
     *
     * ### Why any sample, and not the pixel's value
     *
     * The test is on the **raw samples, before any averaging** -- This Polar Camera does
     * the same, and on a polarization sensor it is the only test that means anything. An
     * output pixel is an average over four analyser angles; one of them can be pinned to
     * full scale while the average comes out unremarkable. That single clipped angle has
     * already destroyed the polarization for that super-pixel, because S1 and S2 are
     * differences between angles and one of the terms is now a number the sensor made up.
     *
     * It is worse than losing the value: a saturated polarized highlight reads as *low*
     * polarization, since all four angles pin together and their differences collapse.
     * So this is most useful on exactly the views where the damage is invisible.
     *
     * ### Why stripes and not a colour
     *
     * A flat overlay colour is a colour the scene might have. Stripes move against the
     * picture and cannot be mistaken for part of it, and alternating black and white
     * means they stay visible whatever they are drawn over.
     */
    private fun stripeClipped(
        data: ByteArray,
        stride: Int,
        w: Int,
        h: Int,
        pitch: Int,
        packed12: Boolean,
        maxValue: Float,
        out: IntArray,
    ) {
        if (maxValue <= 0f) return
        val saturation = SATURATION * maxValue
        for (sy in 0 until h) {
            val baseY = sy * pitch
            val row = sy * w
            for (sx in 0 until w) {
                val baseX = sx * pitch
                var clipped = false
                // Every sample the output pixel stands for, including the super-pixels a
                // step skips: a highlight that clips between the samples is still a
                // highlight that clips.
                var dy = 0
                while (dy < pitch && !clipped) {
                    var dx = 0
                    while (dx < pitch) {
                        if (px(data, stride, baseX + dx, baseY + dy, packed12) >= saturation) {
                            clipped = true
                            break
                        }
                        dx++
                    }
                    dy++
                }
                if (!clipped) continue
                val black = (sx + sy) % (2 * ZEBRA_STRIPE) < ZEBRA_STRIPE
                out[row + sx] = if (black) ZEBRA_DARK else ZEBRA_LIGHT
            }
        }
    }

    /** Stripe width in output pixels. Eight, as the reference implementation uses. */
    const val ZEBRA_STRIPE = 8

    private const val ZEBRA_DARK = 0xFF000000.toInt()
    private const val ZEBRA_LIGHT = 0xFFFFFFFF.toInt()

    /**
     * One analyser reading, summed over the same footprint as [cellValue].
     *
     * Every super-pixel in the footprint carries the same four angles in the same places,
     * so summing across them adds photons of one angle to photons of that same angle.
     * The Stokes arithmetic downstream is unchanged: DoLP and AoLP are ratios, so the
     * common factor cancels and only the noise improves.
     */
    private inline fun analyser(
        data: ByteArray,
        stride: Int,
        cx: Int,
        cy: Int,
        dx: Int,
        dy: Int,
        packed12: Boolean,
        blocks: Int,
    ): Float {
        if (blocks <= 1) return px(data, stride, cx + dx, cy + dy, packed12)
        var total = 0f
        for (by in 0 until blocks) {
            for (bx in 0 until blocks) {
                total += px(
                    data, stride,
                    cx + dx + bx * FlirCamera.MOSAIC,
                    cy + dy + by * FlirCamera.MOSAIC,
                    packed12,
                )
            }
        }
        return total
    }

    /**
     * One cell of the output pixel, either sampled or summed over the step's footprint.
     *
     * At step 2 the preview reads one super-pixel in four and discards the rest. Those
     * three were transferred over the bus and paid for; summing them instead of throwing
     * them away is four times the signal at the same output size, the same bandwidth, and
     * without touching the exposure or the gain. Noise grows as the square root, so it is
     * also two stops of brightness at half the noise cost of two stops of gain.
     *
     * The sum stays within one output pixel's footprint, so every super-pixel it touches
     * has the same analyser layout and the same colour: this mixes photons, never angles.
     * That is what the camera's own binning cannot do on this sensor, where a 2x bin
     * would combine neighbouring pixels that are different analysers.
     *
     * @param blocks super-pixels per output pixel in each direction; 1 samples as before
     */
    private inline fun cellValue(
        data: ByteArray,
        stride: Int,
        cx: Int,
        cy: Int,
        packed12: Boolean,
        blocks: Int,
    ): Float {
        if (blocks <= 1) return cellSum(data, stride, cx, cy, packed12)
        var total = 0f
        for (by in 0 until blocks) {
            for (bx in 0 until blocks) {
                total += cellSum(
                    data, stride,
                    cx + bx * FlirCamera.MOSAIC,
                    cy + by * FlirCamera.MOSAIC,
                    packed12,
                )
            }
        }
        return total
    }

    private fun cellSum(data: ByteArray, stride: Int, cx: Int, cy: Int, packed12: Boolean): Float =
        (px(data, stride, cx + StokesDecoder.D90_X, cy + StokesDecoder.D90_Y, packed12) +
            px(data, stride, cx + StokesDecoder.D45_X, cy + StokesDecoder.D45_Y, packed12) +
            px(data, stride, cx + StokesDecoder.D135_X, cy + StokesDecoder.D135_Y, packed12) +
            px(data, stride, cx + StokesDecoder.D0_X, cy + StokesDecoder.D0_Y, packed12)) * 0.5f

    private fun px(data: ByteArray, stride: Int, x: Int, y: Int, packed12: Boolean): Float =
        StokesDecoder.px(data, stride, x, y, packed12)

    private fun normalize(value: Float, s0: Float): Float =
        if (s0 > EPSILON) (value / s0).coerceIn(-1f, 1f) else 0f

    /** Blue for negative, white at zero, red for positive. */
    private fun diverging(value: Float): Int = if (value < 0f) {
        val t = 1f + value
        argb(clamp8(255f * t), clamp8(255f * t), 255)
    } else {
        val t = 1f - value
        argb(255, clamp8(255f * t), clamp8(255f * t))
    }

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

    /** Blends a colour toward white; [amount] 0 leaves white, 1 leaves the colour. */
    private fun fadeToWhite(color: Int, amount: Float): Int {
        val a = amount.coerceIn(0f, 1f)
        return argb(
            clamp8(255f - a * (255 - ((color shr 16) and 0xFF))),
            clamp8(255f - a * (255 - ((color shr 8) and 0xFF))),
            clamp8(255f - a * (255 - (color and 0xFF))),
        )
    }

    private fun clamp8(value: Float): Int = value.roundToInt().coerceIn(0, 255)

    private fun argb(r: Int, g: Int, b: Int): Int = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    /** One super-pixel in sixteen is plenty for a channel average. */
    private const val WB_SAMPLE_STEP = 4

    /** Keeps a strongly coloured scene from being bleached by its own average. */
    private const val WB_GAIN_MIN = 0.5f
    private const val WB_GAIN_MAX = 3f

    private const val SRGB_LUT_SIZE = 1024
    private val SRGB_LUT = IntArray(SRGB_LUT_SIZE) { i ->
        val linear = i / (SRGB_LUT_SIZE - 1f)
        val encoded = if (linear <= 0.0031308f) {
            12.92f * linear
        } else {
            1.055f * Math.pow(linear.toDouble(), 1.0 / 2.4).toFloat() - 0.055f
        }
        clamp8(encoded * 255f)
    }

    private const val EPSILON = 1e-6f
}
