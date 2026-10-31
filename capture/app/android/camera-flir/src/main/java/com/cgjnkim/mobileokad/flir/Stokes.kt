package com.cgjnkim.mobileokad.flir

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Stokes parameters for one frame, per colour band.
 *
 * Each array is [BANDS] planes of [width] x [height], one quarter the sensor's
 * resolution in each axis because a 4x4 super-pixel yields one measurement per band.
 */
class StokesImage(
    val width: Int,
    val height: Int,
    /** Full-scale value of one analyser reading: 255 at 8 bits, 4095 at 12. */
    val maxValue: Float,
    /** S0, total intensity. Same 0..255 scale as the raw pixels. */
    val s0: Array<FloatArray>,
    /** S1, the 0-vs-90 degree difference. */
    val s1: Array<FloatArray>,
    /** S2, the 45-vs-135 degree difference. */
    val s2: Array<FloatArray>,
    /**
     * The sensor noise measured on the frame these arrays came from, when it was measured.
     *
     * Null means nobody looked, not that there was none, so a caller asking for a debiased
     * DoLP without it gets the plain one rather than a correction by assumption. It does
     * not survive [toNpy] -- the format carries an array and nothing else -- and travels
     * in the capture metadata instead.
     */
    val noise: StokesNoise? = null,
) {
    /**
     * Degree of linear polarization, 0..1.
     *
     * @param debias subtract the noise's expected contribution to the magnitude, as
     *   [StokesNoise] describes. Ignored when no noise was measured. It changes only what
     *   is returned here; the arrays are untouched, and both the preview and the gallery
     *   can be switched between the two views of the same measurement.
     */
    @JvmOverloads
    fun dolp(band: Int, debias: Boolean = false): FloatArray {
        val out = FloatArray(width * height)
        val a = s0[band]
        val b = s1[band]
        val c = s2[band]
        val model = if (debias) noise else null
        // A square root per pixel, and the noise model is a pure function of its
        // arguments, so the pixels can be shared out.
        Parallel.over(out.size) { from, to ->
            for (i in from until to) {
                val denom = a[i]
                if (denom <= EPSILON) continue
                val magnitude = if (model != null) {
                    model.debiasedMagnitude(b[i], c[i], denom, band)
                } else {
                    sqrt(b[i] * b[i] + c[i] * c[i])
                }
                out[i] = (magnitude / denom).coerceIn(0f, 1f)
            }
        }
        return out
    }

    /**
     * Angle of linear polarization in degrees, 0..180.
     *
     * The value is cyclic: 0 and 180 are the same orientation, which is why it wants a
     * cyclic colour map rather than a linear ramp.
     */
    fun aolp(band: Int): FloatArray {
        val out = FloatArray(width * height)
        val b = s1[band]
        val c = s2[band]
        // An atan2 per pixel, which is the most expensive thing any view asks for.
        Parallel.over(out.size) { from, to ->
            for (i in from until to) {
                val deg = 0.5f * atan2(c[i], b[i]) * DEGREES
                out[i] = ((deg % 180f) + 180f) % 180f
            }
        }
        return out
    }

    /**
     * The same measurement with [noise] attached.
     *
     * The arrays are shared, not copied: this says what is known about the frame, and
     * knowing it changes nothing about what was measured.
     */
    fun withNoise(noise: StokesNoise?): StokesImage =
        StokesImage(width, height, maxValue, s0, s1, s2, noise)

    /**
     * This measurement with [band]'s S1 and S2 smoothed against its S0.
     *
     * Only the band asked for: the viewer draws one at a time, and filtering the other two
     * would be work nobody looks at. The untouched planes are shared, not copied, and S0
     * is never filtered -- it is the denominator and it carries the signal.
     */
    /**
     * Removes the brightness gradient that leaks into S1 and S2 at every edge.
     *
     * ### The leak
     *
     * S1 is I(0 deg) minus I(90 deg), and on this sensor those two analysers are diagonal
     * neighbours inside the 2x2 block, offset by (+1,+1) sensor pixels. S2 is I(45) minus
     * I(135), which is the other diagonal, (+1,-1). So each difference subtracts two
     * different points of the scene, and wherever the scene is not flat, part of the
     * brightness gradient arrives as if it were polarization.
     *
     * Measured on three captures, S1 correlates with the gradient along its own diagonal
     * at +0.41, +0.22 and +0.21, and with the other diagonal at +0.19, +0.10 and -0.02;
     * S2 does the same the other way round. Each component follows the diagonal its own
     * analysers lie on, which is the signature of the offset rather than of anything in
     * the scene. In the strongest fiftieth of gradients, |S1| runs 91 counts against 5
     * elsewhere.
     *
     * Smoothing does not cause this -- it is in the raw frame -- but it makes it obvious,
     * because averaging removes the noise that was covering it and leaves the edge.
     *
     * ### The correction, and what it costs
     *
     * The leak is proportional to the gradient along the component's own diagonal, with a
     * coefficient set by the pixel pitch and the optics. Rather than deriving that, it is
     * fitted per frame by least squares -- one scalar per component -- which absorbs the
     * focus and the exposure without needing to know either.
     *
     * This is not free of risk, and it is why the correction is off by default and lives
     * in the viewer. A specular edge is *genuinely* polarized, and its polarization is
     * genuinely co-located with the brightness gradient, so a fit like this removes some
     * real signal along with the artefact. What it cannot do is invent polarization: it
     * only ever subtracts, and only along the gradient.
     */
    /**
     * Turns the polarization reference frame by [deg], so a rolled capture reads level.
     *
     * ### Why this is needed at all
     *
     * The angle of polarization is measured against the sensor's own axes. A capture
     * taken with the camera rolled reports every AoLP rotated by that roll, so two
     * captures of the same subject at different angles cannot be compared, averaged or
     * stacked until they are brought onto a common horizon. On this rig the camera sits a
     * mean 6.4 degrees off level with a spread of 5.3, measured from 34 hand-drawn
     * horizons, so this is not a small correction.
     *
     * ### What it does to the numbers
     *
     * Rotating the frame by an angle turns the Stokes pair by twice it, which is the
     * whole of the arithmetic: S1 and S2 are the components of a quantity that repeats
     * every 180 degrees, not every 360.
     *
     * DoLP is the length of that pair and is therefore untouched -- a useful check, since
     * a correction that changed how polarized something is would be doing something other
     * than what it claims.
     *
     * ### What it is not
     *
     * Display only, and the caller supplies the angle. Nothing here knows which way is
     * up: the phone's gravity sensor measures the phone, which on this rig turns
     * independently of the camera (34 labelled captures put the coupling slope at 0.06
     * where rigid attachment would give 1.0), and polarization gives no usable horizon on
     * ordinary scenes. The only angle worth passing in is one a person drew.
     */
    fun horizonCorrected(band: Int, deg: Float): StokesImage {
        if (deg == 0f) return this
        val twice = Math.toRadians(2.0 * deg)
        val c = kotlin.math.cos(twice).toFloat()
        val sn = kotlin.math.sin(twice).toFloat()
        val a = s1[band]
        val b = s2[band]
        val n = width * height
        val newA = FloatArray(n)
        val newB = FloatArray(n)
        for (i in 0 until n) {
            newA[i] = a[i] * c + b[i] * sn
            newB[i] = -a[i] * sn + b[i] * c
        }
        val outS1 = s1.copyOf()
        val outS2 = s2.copyOf()
        outS1[band] = newA
        outS2[band] = newB
        return StokesImage(width, height, maxValue, s0, outS1, outS2, noise)
    }

    fun gradientLeakCorrected(band: Int): StokesImage {
        val s0b = s0[band]
        val n = width * height
        if (n == 0 || width < 3 || height < 3) return this

        // Both diagonals of the S0 gradient, on the Stokes grid.
        val alongS1 = FloatArray(n)
        val alongS2 = FloatArray(n)
        for (y in 0 until height) {
            val up = if (y > 0) y - 1 else y
            val down = if (y < height - 1) y + 1 else y
            for (x in 0 until width) {
                val left = if (x > 0) x - 1 else x
                val right = if (x < width - 1) x + 1 else x
                val gx = 0.5f * (s0b[y * width + right] - s0b[y * width + left])
                val gy = 0.5f * (s0b[down * width + x] - s0b[up * width + x])
                alongS1[y * width + x] = gx + gy
                alongS2[y * width + x] = gx - gy
            }
        }

        val newS1 = s1.copyOf()
        val newS2 = s2.copyOf()
        newS1[band] = subtractFit(s1[band], alongS1)
        newS2[band] = subtractFit(s2[band], alongS2)
        return StokesImage(width, height, maxValue, s0, newS1, newS2, noise)
    }

    /**
     * Removes the part of [x] that varies with [basis], by least squares.
     *
     * Both sides are centred first, and only the varying part is subtracted. Fitting
     * through the origin instead was wrong in a way worth recording: wherever the basis
     * has a non-zero mean -- which it does on any scene brighter at one side than the
     * other -- the fit absorbs the whole average level of S1 and a uniformly polarized
     * frame comes back unpolarized. The artefact being removed is a term that follows the
     * gradient, and a constant is not that.
     *
     * The slope is clamped at zero when the basis carries no variation, which is a flat
     * frame: there is no gradient there to blame anything on.
     */
    private fun subtractFit(x: FloatArray, basis: FloatArray): FloatArray {
        val n = x.size
        if (n == 0) return x
        var meanX = 0.0
        var meanB = 0.0
        for (i in 0 until n) {
            meanX += x[i].toDouble()
            meanB += basis[i].toDouble()
        }
        meanX /= n
        meanB /= n

        var xy = 0.0
        var yy = 0.0
        for (i in 0 until n) {
            val dx = x[i] - meanX
            val db = basis[i] - meanB
            xy += dx * db
            yy += db * db
        }
        if (yy <= 0.0) return x
        val k = (xy / yy).toFloat()
        // Only ever removes gradient-aligned signal; a fit that came out the wrong sign
        // would be adding the artefact rather than taking it away.
        if (k <= 0f) return x
        val offset = meanB.toFloat()
        return FloatArray(n) { x[it] - k * (basis[it] - offset) }
    }

    @JvmOverloads
    fun smoothedPolarization(
        band: Int,
        radius: Int = PolarSmoothing.DEFAULT_RADIUS,
        epsilon: Float = PolarSmoothing.DEFAULT_EPSILON,
    ): StokesImage {
        val (a, b) = PolarSmoothing.guided(
            s0[band], s1[band], s2[band], width, height, radius, epsilon,
        )
        val newS1 = s1.copyOf()
        val newS2 = s2.copyOf()
        newS1[band] = a
        newS2[band] = b
        return StokesImage(width, height, maxValue, s0, newS1, newS2, noise)
    }

    /** S1 or S2 divided by S0, in -1..1, which is what the diverging views show. */
    fun normalized(component: Array<FloatArray>, band: Int): FloatArray {
        val out = FloatArray(width * height)
        val a = s0[band]
        val v = component[band]
        Parallel.over(out.size) { from, to ->
            for (i in from until to) {
                out[i] = if (a[i] > EPSILON) (v[i] / a[i]).coerceIn(-1f, 1f) else 0f
            }
        }
        return out
    }

    /**
     * NumPy .npy of shape (3 bands, 3 components, height, width), float32.
     *
     * Chosen so the capture can be reopened by anything that reads .npy without a
     * bespoke parser; band order is R, G, B and component order is S0, S1, S2.
     */
    fun toNpy(): ByteArray {
        val header = "{'descr': '<f4', 'fortran_order': False, " +
            "'shape': (${BANDS}, 3, $height, $width), }"
        // The spec wants the header padded so the data starts on a 64 byte boundary.
        val prefixLength = 10 + header.length + 1
        val padding = (64 - prefixLength % 64) % 64
        val padded = header + " ".repeat(padding) + "\n"

        val out = ByteArrayOutputStream(prefixLength + padding + BANDS * 3 * width * height * 4)
        out.write(byteArrayOf(0x93.toByte()) + "NUMPY".toByteArray(Charsets.US_ASCII))
        out.write(1)  // major version
        out.write(0)  // minor version
        out.write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(padded.length.toShort()).array())
        out.write(padded.toByteArray(Charsets.US_ASCII))

        val plane = ByteBuffer.allocate(width * height * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (band in 0 until BANDS) {
            for (component in listOf(s0, s1, s2)) {
                plane.clear()
                for (v in component[band]) plane.putFloat(v)
                out.write(plane.array())
            }
        }
        return out.toByteArray()
    }

    companion object {
        /**
         * Reads back what [toNpy] wrote.
         *
         * @param maxValue the full-scale reading the arrays were computed from, 255 or
         *   4095. It cannot travel in the file: numpy rejects a header carrying anything
         *   beyond descr, fortran_order and shape, and putting it there would make the
         *   .npy unreadable by the tool the format exists to be readable by. The capture
         *   metadata records it as `maxValue` for exactly this.
         *
         * @param noise the sensor noise fitted at capture time, likewise from the
         *   metadata. Optional: without it the arrays still read back exactly, and only
         *   the debiased DoLP is unavailable.
         *
         * @throws U3vException when the bytes are not the array this writes
         */
        @JvmOverloads
        fun fromNpy(bytes: ByteArray, maxValue: Float, noise: StokesNoise? = null): StokesImage {
            if (bytes.size < MIN_NPY_SIZE || bytes[0] != MAGIC_BYTE ||
                String(bytes, 1, 5, Charsets.US_ASCII) != "NUMPY"
            ) {
                throw U3vException("not a .npy file")
            }

            val major = bytes[6].toInt() and 0xFF
            // Version 1 stores the header length in two bytes, 2 and 3 in four.
            val lengthBytes = if (major >= 2) 4 else 2
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val headerLength =
                if (lengthBytes == 2) buffer.getShort(8).toInt() and 0xFFFF else buffer.getInt(8)
            val dataOffset = 8 + lengthBytes + headerLength
            if (headerLength <= 0 || dataOffset > bytes.size) {
                throw U3vException("npy header length $headerLength does not fit the file")
            }

            val header = String(bytes, 8 + lengthBytes, headerLength, Charsets.US_ASCII)
            if (!header.contains("\'descr\': \'<f4\'")) {
                throw U3vException("npy is not little-endian float32: $header")
            }
            if (header.contains("\'fortran_order\': True")) {
                throw U3vException("npy is in Fortran order, which toNpy never writes")
            }

            val shape = Regex("""\'shape\':\s*\(([^)]*)\)""").find(header)
                ?.groupValues?.get(1)
                ?.split(',')
                ?.mapNotNull { it.trim().toIntOrNull() }
                ?: throw U3vException("npy header has no shape: $header")
            if (shape.size != 4 || shape[0] != BANDS || shape[1] != COMPONENTS) {
                throw U3vException("expected a ($BANDS, $COMPONENTS, height, width) array, got $shape")
            }

            val height = shape[2]
            val width = shape[3]
            val plane = width * height
            val expected = dataOffset.toLong() + BANDS.toLong() * COMPONENTS * plane * 4
            if (bytes.size.toLong() < expected) {
                throw U3vException("npy is ${bytes.size} B, short of $expected")
            }

            val floats = ByteBuffer.wrap(bytes, dataOffset, bytes.size - dataOffset)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer()

            val s0 = Array(BANDS) { FloatArray(plane) }
            val s1 = Array(BANDS) { FloatArray(plane) }
            val s2 = Array(BANDS) { FloatArray(plane) }
            // Written band by band, each with its three components in S0, S1, S2 order.
            for (band in 0 until BANDS) {
                for (component in listOf(s0, s1, s2)) {
                    floats.get(component[band], 0, plane)
                }
            }
            return StokesImage(width, height, maxValue, s0, s1, s2, noise)
        }

        private const val MAGIC_BYTE = 0x93.toByte()
        private const val MIN_NPY_SIZE = 12
        private const val COMPONENTS = 3

        const val BANDS = 3
        const val R = 0
        const val G = 1
        const val B = 2

        val BAND_NAMES = listOf("R", "G", "B")

        private const val EPSILON = 1e-6f
        private const val DEGREES = (180.0 / Math.PI).toFloat()
    }
}

/**
 * Converts a BayerRGPolarized8 frame into per-band Stokes parameters.
 *
 * The IMX250MYR stacks an RGGB Bayer filter on a four-directional polarizer so that one
 * colour covers a whole 2x2 polarizer unit. A 4x4 super-pixel therefore holds four
 * colour cells, each carrying all four analyser angles:
 *
 *     R R | G G          within every 2x2 cell:   90  45
 *     R R | G G                                  135   0
 *     ----+----
 *     G G | B B
 *     G G | B B
 *
 * The two green cells are averaged into a single G band.
 */
object StokesDecoder {

    /**
     * Offsets of the four analyser angles inside a 2x2 polarizer cell, as (dx, dy).
     *
     * Matches the arrangement polanalyser documents for the IMX250MZR / MYR:
     * "(0, 0) is 90, (0, 1) is 45, (1, 0) is 135, (1, 1) is 0" in (row, column) terms.
     */
    internal const val D90_X = 0; internal const val D90_Y = 0
    internal const val D45_X = 1; internal const val D45_Y = 0
    internal const val D135_X = 0; internal const val D135_Y = 1
    internal const val D0_X = 1; internal const val D0_Y = 1

    /**
     * Top-left corner of each colour cell inside the 4x4 super-pixel, for BayerRG.
     *
     * The analyser repeats every two pixels and the colour every four, so each colour
     * covers one whole 2x2 polarizer cell. Equivalent to how polanalyser splits a colour
     * frame: sub-sampling by two per analyser position leaves an RGGB Bayer image.
     */
    internal val CELLS = arrayOf(
        intArrayOf(0, 0),  // R
        intArrayOf(2, 0),  // G, first green
        intArrayOf(0, 2),  // G, second green
        intArrayOf(2, 2),  // B
    )

    /**
     * @param data raw mosaic straight off the camera, 8bpp or Mono12p packed
     * @throws U3vException when the frame is not a whole number of super-pixels
     */
    fun decode(data: ByteArray, width: Int, height: Int, pixelFormat: Long): StokesImage =
        decode(Mosaic(unpack(data, width, height, pixelFormat), width, height, maxValueOf(pixelFormat)))

    internal fun isPacked12(pixelFormat: Long): Boolean = when (pixelFormat) {
        PixelFormats.BAYER_RG_POLARIZED_12P, PixelFormats.POLARIZED_12P -> true
        PixelFormats.BAYER_RG_POLARIZED_8, PixelFormats.POLARIZED_8 -> false
        else -> throw U3vException("unsupported pixel format 0x%08x".format(pixelFormat))
    }

    internal fun maxValueOf(pixelFormat: Long): Float =
        if (isPacked12(pixelFormat)) MAX_12BIT else MAX_8BIT

    /**
     * The real decode, from an expanded mosaic.
     *
     * Taking a [Mosaic] rather than the packed bytes is what lets a binned frame go
     * through the identical path: summing super-pixels leaves the layout alone and only
     * changes the numbers and the frame's size, both of which the mosaic already carries.
     *
     * @throws U3vException when the frame is not a whole number of super-pixels
     */
    fun decode(mosaic: Mosaic): StokesImage {
        val width = mosaic.width
        val height = mosaic.height
        if (width % FlirCamera.MOSAIC != 0 || height % FlirCamera.MOSAIC != 0) {
            throw U3vException("frame ${width}x$height is not a multiple of ${FlirCamera.MOSAIC}")
        }
        if (mosaic.samples.size < width.toLong() * height) {
            throw U3vException("frame has ${mosaic.samples.size} samples, short of ${width * height}")
        }

        val w = width / FlirCamera.MOSAIC
        val h = height / FlirCamera.MOSAIC
        val n = w * h

        val s0 = Array(StokesImage.BANDS) { FloatArray(n) }
        val s1 = Array(StokesImage.BANDS) { FloatArray(n) }
        val s2 = Array(StokesImage.BANDS) { FloatArray(n) }

        for (sy in 0 until h) {
            val baseY = sy * FlirCamera.MOSAIC
            val outRow = sy * w
            for (sx in 0 until w) {
                val baseX = sx * FlirCamera.MOSAIC
                val out = outRow + sx

                for (cell in CELLS.indices) {
                    val cx = baseX + CELLS[cell][0]
                    val cy = baseY + CELLS[cell][1]

                    val i90 = mosaic[cx + D90_X, cy + D90_Y]
                    val i45 = mosaic[cx + D45_X, cy + D45_Y]
                    val i135 = mosaic[cx + D135_X, cy + D135_Y]
                    val i0 = mosaic[cx + D0_X, cy + D0_Y]

                    // Cells 1 and 2 are both green, so each contributes half of band G.
                    val band = when (cell) {
                        0 -> StokesImage.R
                        3 -> StokesImage.B
                        else -> StokesImage.G
                    }
                    val weight = if (band == StokesImage.G) 0.5f else 1f

                    s0[band][out] += weight * (i0 + i45 + i90 + i135) * 0.5f
                    s1[band][out] += weight * (i0 - i90)
                    s2[band][out] += weight * (i45 - i135)
                }
            }
        }
        return StokesImage(w, h, mosaic.maxValue, s0, s1, s2)
    }

    /**
     * One analyser reading.
     *
     * Mono12p packs two pixels into three bytes as a little endian 24 bit word: the
     * first pixel is the low 12 bits, the second the high 12.
     */
    /**
     * Expands a raw frame to one unsigned sample per pixel, for storage.
     *
     * 12-bit values are left in their own scale (0..4095) rather than stretched to 16
     * bits: a reader should see the sensor's numbers, not a rescaling of them.
     */
    fun unpack(data: ByteArray, width: Int, height: Int, pixelFormat: Long): ShortArray {
        val packed12 = isPacked12(pixelFormat)
        val expected = if (packed12) width.toLong() * height * 3 / 2 else width.toLong() * height
        if (data.size < expected) {
            throw U3vException("frame data ${data.size} B is short of $expected")
        }
        val out = ShortArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                out[y * width + x] = px(data, width, x, y, packed12).toInt().toShort()
            }
        }
        return out
    }

    internal fun px(data: ByteArray, stride: Int, x: Int, y: Int, packed12: Boolean): Float {
        val index = y * stride + x
        if (!packed12) return (data[index].toInt() and 0xFF).toFloat()

        val group = index / 2 * 3
        val b1 = data[group + 1].toInt() and 0xFF
        return if (index and 1 == 0) {
            (((data[group].toInt() and 0xFF) or ((b1 and 0x0F) shl 8))).toFloat()
        } else {
            (((b1 shr 4) or ((data[group + 2].toInt() and 0xFF) shl 4))).toFloat()
        }
    }

    internal const val MAX_8BIT = 255f
    internal const val MAX_12BIT = 4095f
}
