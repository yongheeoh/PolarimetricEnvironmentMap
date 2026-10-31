package com.cgjnkim.mobileokad.flir

import kotlin.math.abs

/** A rendered picture: packed ARGB with the size it came out at. */
class RenderedImage(val width: Int, val height: Int, val argb: IntArray)

/**
 * S0 as a colour picture at four times the pixels, by demosaicing instead of collapsing.
 *
 * ### Why there are four times as many, and not sixteen
 *
 * The 4x4 super-pixel is four 2x2 analyser cells, one per colour: R, G, G, B. Every cell
 * measures all four analyser angles of its own colour, so **S0 is genuinely measured once
 * per cell** -- and the cells sit on a 2 px grid whose colours read R G / G B. That is an
 * ordinary Bayer mosaic at 1224 x 1024, half the sensor in each axis.
 *
 * [StokesRender] throws that away. It averages the two greens and reports one pixel per
 * super-pixel, which is 612 x 512: correct, because the polarization views need all three
 * bands co-located, and wasteful for a picture that only wants intensity. Demosaicing the
 * cell grid instead gives 1224 x 1024 with every pixel's own colour actually measured and
 * the other two interpolated, which is what every colour camera ever made does.
 *
 * [s0RgbFull] goes further, to the sensor's own 2448 x 2048, and it is a different claim.
 * Every pixel's photon count is real, but no pixel measured its own S0 -- only one of the
 * four angles -- so the rest comes from its cell. Real luminance detail, with the
 * separation from polarization interpolated. Both are offered because they answer
 * different questions, and they are named apart so neither is mistaken for the other.
 *
 * Nothing here is stored. It is a way of drawing the archive, and the archive is the raw
 * mosaic.
 */
object Demosaic {

    /**
     * S0 of every analyser cell, laid out as an RGGB Bayer plane.
     *
     * Same scale and definition as [StokesImage.s0]: the four readings summed and halved,
     * so an unpolarized scene at full scale comes to twice [Mosaic.maxValue].
     */
    fun s0Bayer(mosaic: Mosaic): FloatArray {
        val cell = CELL
        if (mosaic.width % cell != 0 || mosaic.height % cell != 0) {
            throw U3vException(
                "frame ${mosaic.width}x${mosaic.height} is not a whole number of analyser cells"
            )
        }
        val w = mosaic.width / cell
        val h = mosaic.height / cell
        val out = FloatArray(w * h)
        Parallel.over(h, Parallel.MIN_ROWS_PER_SLICE) { from, to ->
            for (v in from until to) {
                val y = v * cell
                val row = v * w
                for (u in 0 until w) {
                    val x = u * cell
                    out[row + u] = (
                        mosaic[x, y] + mosaic[x + 1, y] + mosaic[x, y + 1] + mosaic[x + 1, y + 1]
                        ) * 0.5f
                }
                    }
        }
        return out
    }

    /**
     * The colour picture, at half the sensor's width and height.
     *
     * Green is interpolated along whichever axis the picture is smoother on, which is what
     * keeps a fine edge from turning into a zip of colour fringes; red and blue are then
     * interpolated as differences from green, because the three channels track each other
     * far better than any of them varies on its own.
     */
    @JvmOverloads
    fun s0Rgb(
        mosaic: Mosaic,
        display: StokesDisplay = StokesDisplay.DEFAULT,
        out: IntArray? = null,
    ): RenderedImage {
        val p = planes(mosaic)
        val encoder = StokesRender.S0Encoder(display, mosaic.maxValue, p.red, p.green, p.blue)
        val n = p.width * p.height
        val pixels = if (out != null && out.size == n) out else IntArray(n)
        Parallel.over(n) { from, to ->
            for (i in from until to) pixels[i] = encoder.encode(p.red[i], p.green[i], p.blue[i])
        }
        return straightened(RenderedImage(p.width, p.height, pixels), display)
    }

    /**
     * S0 as a colour picture at the sensor's own 2448 x 2048.
     *
     * ### What is measured here and what is not
     *
     * Every pixel has its own photon count, and that is real: a dot one pixel wide is in
     * the data, and [s0Rgb] loses it by summing each analyser cell. What no pixel has is
     * its own *S0*, because it saw only one of the four angles.
     *
     * The obvious repair does not work. Predicting each pixel from its own cell's Stokes
     * vector and adding back the difference spends three of the cell's four measurements
     * on polarization, which leaves the detail exactly one degree of freedom per cell --
     * and that one is stuck in a single shape, a diagonal checkerboard. A dot comes back
     * smeared across the diagonal and a line does not come back at all.
     *
     * So the polarization is taken from a **wider neighbourhood** instead, where it is far
     * smoother than the picture is, and each pixel then keeps its own reading in full:
     *
     *     S0(pixel) = 2 * reading / (1 + n1*cos2a + n2*sin2a)
     *
     * with `n1`, `n2` the normalized Stokes averaged over the cell's neighbours and `a`
     * the pixel's own analyser angle. Divide the reading by how much its analyser was
     * expected to pass, and what is left is that pixel's share of the light. All four
     * measurements in a cell now carry intensity, which is where the resolution comes
     * from.
     *
     * ### What it cannot do, and the costs
     *
     * A structure aligned with the analyser grid is genuinely ambiguous. A one-pixel line
     * down every other column brightens the 0 and 45 degree positions and darkens the
     * others, which is *exactly* what an 85 percent polarized field at 22.5 degrees looks
     * like. No single frame can tell those apart, and this reads it as polarization.
     *
     * Noise: each pixel now stands alone rather than being averaged with three others, so
     * the shot noise per pixel is **twice** the cell picture's. It is a sharper picture of
     * a noisier measurement.
     *
     * Colour: interpolated from the cell grid and upsampled, with the detail added to all
     * three channels alike. Fine detail is therefore achromatic, which is what every
     * pan-sharpening method assumes and is nearly true of edges and texture.
     *
     * [s0Rgb] is the one to reach for when the question is what was measured. This is the
     * one to reach for when the question is what the scene looked like.
     */
    @JvmOverloads
    fun s0RgbFull(
        mosaic: Mosaic,
        display: StokesDisplay = StokesDisplay.DEFAULT,
        out: IntArray? = null,
    ): RenderedImage {
        val p = planes(mosaic)
        val width = mosaic.width
        val height = mosaic.height
        val detail = detailLayer(mosaic)

        // The balance is decided from the cell grid, where every channel is present; the
        // detail is achromatic and shifts nothing about the colour it is added to.
        val encoder = StokesRender.S0Encoder(display, mosaic.maxValue, p.red, p.green, p.blue)
        val pixels = if (out != null && out.size == width * height) out else IntArray(width * height)

        Parallel.over(height, Parallel.MIN_ROWS_PER_SLICE) { from, to ->
            for (y in from until to) {
                // Cell centres sit half a pixel off the pixel grid, so a doubling weights the
                // two nearest cells three to one rather than evenly; getting this wrong shows
                // up as the whole picture sliding half a pixel diagonally.
                val v = y / CELL
                val nearV = if (y % CELL == 0) v - 1 else v + 1
                for (x in 0 until width) {
                    val u = x / CELL
                    val nearU = if (x % CELL == 0) u - 1 else u + 1
                    val d = detail[y * width + x]
                    pixels[y * width + x] = encoder.encode(
                        upsample(p.red, p.width, p.height, u, v, nearU, nearV) + d,
                        upsample(p.green, p.width, p.height, u, v, nearU, nearV) + d,
                        upsample(p.blue, p.width, p.height, u, v, nearU, nearV) + d,
                    )
                }
                    }
        }
        return straightened(RenderedImage(width, height, pixels), display)
    }

    /**
     * The lens taken out of a finished S0 picture, when the display asks for it.
     *
     * S0 does not go through [StokesRender.render] -- intensity is measured once per
     * analyser cell, four times as often as the Stokes arrays keep it, so the colour
     * picture is demosaiced from the mosaic instead. That means the straightening applied
     * at the end of that render never reached this view, and S0 stayed bowed while every
     * polarization view was corrected. It belongs on both paths, at the same point in
     * each: after the picture exists and before anybody sees it.
     *
     * The correction is a fraction of the frame rather than a count of pixels, so it is
     * the same optics whether this ran at cell resolution or at the sensor's own.
     */
    private fun straightened(image: RenderedImage, display: StokesDisplay): RenderedImage {
        if (!display.rectify) return image
        val fixed = LensCalibration.rectify(image.argb, image.width, image.height)
        fixed.copyInto(image.argb)
        return image
    }

    /** Bilinear over the four nearest cells, weighted 9:3:3:1 for a doubling. */
    private fun upsample(
        plane: FloatArray, w: Int, h: Int, u: Int, v: Int, nearU: Int, nearV: Int,
    ): Float {
        val u2 = nearU.coerceIn(0, w - 1)
        val v2 = nearV.coerceIn(0, h - 1)
        return 0.5625f * plane[v * w + u] +
            0.1875f * plane[v * w + u2] +
            0.1875f * plane[v2 * w + u] +
            0.0625f * plane[v2 * w + u2]
    }

    /**
     * How much brighter each pixel is than its cell, once its analyser is divided out.
     *
     * The polarization used for the division is smoothed over the cell's neighbours: a
     * cell that judges itself has no measurements left over for detail, while polarization
     * that is allowed to vary only over several cells leaves all four.
     */
    private fun detailLayer(mosaic: Mosaic): FloatArray {
        val width = mosaic.width
        val height = mosaic.height
        val w = width / CELL
        val h = height / CELL

        // Normalized Stokes per cell, then a box average of them. Normalized first,
        // because it is the *fraction* of light each analyser passes that varies slowly;
        // S1 itself scales with however bright the cell happens to be.
        val s0 = FloatArray(w * h)
        val n1 = FloatArray(w * h)
        val n2 = FloatArray(w * h)
        Parallel.over(h, Parallel.MIN_ROWS_PER_SLICE) { from, to ->
            for (v in from until to) {
                for (u in 0 until w) {
                    val x = u * CELL
                    val y = v * CELL
                    val i90 = mosaic[x + StokesDecoder.D90_X, y + StokesDecoder.D90_Y]
                    val i45 = mosaic[x + StokesDecoder.D45_X, y + StokesDecoder.D45_Y]
                    val i135 = mosaic[x + StokesDecoder.D135_X, y + StokesDecoder.D135_Y]
                    val i0 = mosaic[x + StokesDecoder.D0_X, y + StokesDecoder.D0_Y]
                    val total = (i0 + i45 + i90 + i135) * 0.5f
                    val i = v * w + u
                    s0[i] = total
                    if (total > EPSILON) {
                        n1[i] = ((i0 - i90) / total).coerceIn(-1f, 1f)
                        n2[i] = ((i45 - i135) / total).coerceIn(-1f, 1f)
                    }
                }
                    }
        }
        val smooth1 = boxBlur(n1, w, h)
        val smooth2 = boxBlur(n2, w, h)

        val out = FloatArray(width * height)
        Parallel.over(h, Parallel.MIN_ROWS_PER_SLICE) { from, to ->
            for (v in from until to) {
                for (u in 0 until w) {
                    val i = v * w + u
                    val a = smooth1[i]
                    val b = smooth2[i]
                    val cellS0 = s0[i]
                    val x = u * CELL
                    val y = v * CELL
                    // cos2a and sin2a are 1, 0, -1, 0 and 0, 1, 0, -1 at these four angles,
                    // so the modulation is just plus or minus one of the two components.
                    put(out, width, x + StokesDecoder.D0_X, y + StokesDecoder.D0_Y,
                        mosaic[x + StokesDecoder.D0_X, y + StokesDecoder.D0_Y], 1f + a, cellS0)
                    put(out, width, x + StokesDecoder.D45_X, y + StokesDecoder.D45_Y,
                        mosaic[x + StokesDecoder.D45_X, y + StokesDecoder.D45_Y], 1f + b, cellS0)
                    put(out, width, x + StokesDecoder.D90_X, y + StokesDecoder.D90_Y,
                        mosaic[x + StokesDecoder.D90_X, y + StokesDecoder.D90_Y], 1f - a, cellS0)
                    put(out, width, x + StokesDecoder.D135_X, y + StokesDecoder.D135_Y,
                        mosaic[x + StokesDecoder.D135_X, y + StokesDecoder.D135_Y], 1f - b, cellS0)
                }
                    }
        }
        return out
    }

    /**
     * One pixel's detail. [pass] is how much of the light its analyser was expected to let
     * through, floored so that a nearly total polarization cannot divide by nothing and
     * turn a reading into a spike.
     */
    private fun put(
        out: FloatArray, width: Int, x: Int, y: Int, reading: Float, pass: Float, cellS0: Float,
    ) {
        out[y * width + x] = 2f * reading / pass.coerceAtLeast(MIN_TRANSMISSION) - cellS0
    }

    /** Three-by-three mean over the cell grid, edges mirrored. */
    private fun boxBlur(plane: FloatArray, w: Int, h: Int): FloatArray {
        val out = FloatArray(w * h)
        Parallel.over(h, Parallel.MIN_ROWS_PER_SLICE) { from, to ->
            for (v in from until to) {
                for (u in 0 until w) {
                    var sum = 0f
                    for (dv in -1..1) for (du in -1..1) sum += at(plane, w, h, u + du, v + dv)
                    out[v * w + u] = sum / 9f
                }
                    }
        }
        return out
    }

    /** The cell-grid colour picture in linear light, before any display choice. */
    private class Planes(
        val width: Int,
        val height: Int,
        val red: FloatArray,
        val green: FloatArray,
        val blue: FloatArray,
    )

    private fun planes(mosaic: Mosaic): Planes {
        val width = mosaic.width / CELL
        val height = mosaic.height / CELL
        val bayer = s0Bayer(mosaic)
        val green = interpolateGreen(bayer, width, height)
        val red = FloatArray(width * height)
        val blue = FloatArray(width * height)
        interpolateChroma(bayer, green, width, height, red, blue)
        return Planes(width, height, red, green, blue)
    }

    /** Colour at a site of the RGGB grid: R where both are even, B where both are odd. */
    private fun isGreen(u: Int, v: Int) = ((u xor v) and 1) == 1
    private fun isRed(u: Int, v: Int) = (u and 1) == 0 && (v and 1) == 0

    /**
     * Green everywhere, from the half of the grid that measured it.
     *
     * At a red or blue site the two candidate averages are compared by how much the
     * picture changes along each axis, and the smoother one wins -- Hamilton and Adams'
     * gradient test. The correction term is the second difference of the site's own colour,
     * which carries the detail green is missing at that point: interpolating green from
     * green alone blurs every edge that runs across the sampling grid.
     */
    private fun interpolateGreen(bayer: FloatArray, w: Int, h: Int): FloatArray {
        val g = FloatArray(w * h)
        Parallel.over(h, Parallel.MIN_ROWS_PER_SLICE) { from, to ->
            for (v in from until to) {
                for (u in 0 until w) {
                    val i = v * w + u
                    if (isGreen(u, v)) {
                        g[i] = bayer[i]
                        continue
                    }
                    val c = bayer[i]
                    val left = at(bayer, w, h, u - 1, v)
                    val right = at(bayer, w, h, u + 1, v)
                    val up = at(bayer, w, h, u, v - 1)
                    val down = at(bayer, w, h, u, v + 1)
                    val c2Left = at(bayer, w, h, u - 2, v)
                    val c2Right = at(bayer, w, h, u + 2, v)
                    val c2Up = at(bayer, w, h, u, v - 2)
                    val c2Down = at(bayer, w, h, u, v + 2)

                    val laplaceH = 2f * c - c2Left - c2Right
                    val laplaceV = 2f * c - c2Up - c2Down
                    val gradH = abs(left - right) + abs(laplaceH)
                    val gradV = abs(up - down) + abs(laplaceV)

                    g[i] = when {
                        gradH < gradV -> 0.5f * (left + right) + 0.25f * laplaceH
                        gradV < gradH -> 0.5f * (up + down) + 0.25f * laplaceV
                        else -> 0.25f * (left + right + up + down) + 0.125f * (laplaceH + laplaceV)
                    }
                }
                    }
        }
        return g
    }

    /**
     * Red and blue everywhere, interpolated as their difference from green.
     *
     * Interpolating red directly would smear it across every edge, because red is sampled
     * on a quarter of the grid. The difference `R - G` is nearly flat over an edge that
     * both channels see, so interpolating that and adding green back keeps the edge sharp
     * and only the colour smooth -- which is how a scene actually behaves.
     */
    private fun interpolateChroma(
        bayer: FloatArray,
        g: FloatArray,
        w: Int,
        h: Int,
        red: FloatArray,
        blue: FloatArray,
    ) {
        // The difference is known only where the colour was measured; elsewhere it is
        // averaged from the sites that have it.
        fun diffAt(u: Int, v: Int): Float {
            val i = mirror(v, h) * w + mirror(u, w)
            return bayer[i] - g[i]
        }

        Parallel.over(h, Parallel.MIN_ROWS_PER_SLICE) { from, to ->
            for (v in from until to) {
                for (u in 0 until w) {
                    val i = v * w + u
                    val here = g[i]
                    if (isGreen(u, v)) {
                        // A green site has red on one axis and blue on the other; which is
                        // which alternates row by row.
                        val redIsHorizontal = (v and 1) == 0
                        val dr = if (redIsHorizontal) {
                            0.5f * (diffAt(u - 1, v) + diffAt(u + 1, v))
                        } else {
                            0.5f * (diffAt(u, v - 1) + diffAt(u, v + 1))
                        }
                        val db = if (redIsHorizontal) {
                            0.5f * (diffAt(u, v - 1) + diffAt(u, v + 1))
                        } else {
                            0.5f * (diffAt(u - 1, v) + diffAt(u + 1, v))
                        }
                        red[i] = here + dr
                        blue[i] = here + db
                    } else {
                        // Red and blue sit on each other's diagonals.
                        val diagonal = 0.25f * (
                            diffAt(u - 1, v - 1) + diffAt(u + 1, v - 1) +
                                diffAt(u - 1, v + 1) + diffAt(u + 1, v + 1)
                            )
                        if (isRed(u, v)) {
                            red[i] = bayer[i]
                            blue[i] = here + diagonal
                        } else {
                            blue[i] = bayer[i]
                            red[i] = here + diagonal
                        }
                    }
                }
                    }
        }
    }

    private fun at(plane: FloatArray, w: Int, h: Int, u: Int, v: Int): Float =
        plane[mirror(v, h) * w + mirror(u, w)]

    /**
     * Folds an index back inside the picture, mirrored about the edge.
     *
     * Mirroring rather than clamping, because on a Bayer grid the colour of a site is its
     * coordinates' parity and clamping does not preserve it: at the top row, reading one
     * pixel up clamps to the row itself, which is the wrong colour, and the interpolation
     * then averages green into blue. Reflecting moves by two and keeps the parity, so a
     * flat field stays flat right to the border -- which it did not before.
     */
    private fun mirror(i: Int, n: Int): Int {
        var k = i
        if (k < 0) k = -k
        if (k >= n) k = 2 * (n - 1) - k
        return k.coerceIn(0, n - 1)
    }

    /** Pixels on a side of one analyser cell. */
    private const val CELL = 2

    /**
     * Least transmission the division will assume.
     *
     * A perfectly polarized field would have one analyser passing nothing, and dividing a
     * reading of noise by nothing makes a spike where the picture is in fact dark. Below
     * this the pixel is simply drawn as bright as the floor allows.
     */
    private const val MIN_TRANSMISSION = 0.25f

    private const val EPSILON = 1e-6f
}
