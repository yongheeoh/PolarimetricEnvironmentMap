package com.cgjnkim.mobileokad.flir

import kotlin.math.hypot
import kotlin.math.roundToInt

/**
 * What the lens does to straight lines, and how to put them back.
 *
 * ### Where these numbers came from
 *
 * Two methods that share no machinery, run on 199 frames of a furniture showroom shot on
 * 2026-09-09 -- a scene with real depth, real texture and a great many straight edges.
 *
 * **Structure from motion.** COLMAP over the whole session, matched with affine shape
 * estimation and domain-size pooling, which is what lets frames seconds apart on a walk
 * match at all. It built two independent reconstructions, of 23 and 13 images, and three
 * camera models were then fitted to the same matches so that the extra terms had to earn
 * their place against each other rather than against nothing:
 *
 * | model         | reprojection | k1     | k2     | tangential |
 * |---------------|--------------|--------|--------|------------|
 * | one radial    | 0.887 px     | -0.108 |        |            |
 * | two radial    | 0.838 px     | -0.155 | +0.065 |            |
 * | plus tangential | 0.796 px   | -0.158 | +0.067 | 0.0008     |
 *
 * The second reconstruction, sharing no images with the first, returned -0.161/+0.068 and
 * -0.168/+0.072. Tangential distortion came back at a thousandth and is ignored; the two
 * focal lengths came back within a percent of each other, so the pixels are square.
 *
 * A fourth reconstruction, over 225 frames of a street on the same day, registered 124 of
 * them into one model of 41926 points at 0.581 px -- fourteen times the size of anything
 * the showroom gave -- and returned -0.163/+0.069. Two smaller street models returned
 * -0.164/+0.066 and -0.157/+0.084. **The numbers below are the -0.164/+0.066 model**,
 * which was the one that straightened both scenes best when each candidate was scored on
 * chains from the place it did not come from. The choice hardly matters: a grid search over
 * the combined chains of both scenes beat it by 0.08 percent, so anything in this
 * neighbourhood is the same correction.
 *
 * **Plumb lines.** 18022 long edge chains from the same frames, corrected at each
 * candidate pair and scored by how bowed the ones that are genuinely straight remain.
 * Searched over a grid with no knowledge of the SfM result, the best cell is exactly
 * k1 = -0.16, k2 = +0.07. The median bow of a peripheral edge:
 *
 * | correction                | bow     |
 * |---------------------------|---------|
 * | none                      | 0.317 px |
 * | one term, k1 = -0.074     | 0.247 px |
 * | these two terms           | 0.224 px |
 *
 * Two methods with nothing in common landing on the same pair is the reason to believe
 * it. A single radial term was what shipped first, and it takes out three quarters of the
 * bow the second term also removes.
 *
 * ### What went wrong on the way, worth not repeating
 *
 * An earlier attempt on a different set measured nothing at all, and concluded the archive
 * could not calibrate the lens. Those frames were 2x-binned and eight stops underexposed:
 * a realistic distortion put less sag into an edge than the edge could be located to.
 * Unbinned and properly exposed is what made this possible.
 *
 * A plumb-line fit scored on the RMS bend of every chain also reads as no distortion at
 * all, because a showroom is full of genuinely curved things and they outvote the straight
 * edges. The median bow of chains that are straight once corrected is the estimator that
 * works. Either way, bend known-straight chains by a known coefficient and check it comes
 * back before believing any fit.
 *
 * ### What is still soft
 *
 * Nobody has shot a chessboard, which would settle this in minutes and independently. The
 * optical axis is assumed to be the middle of the array. Focal length and the radial terms
 * trade against each other, and only SfM constrains the focal at all: its estimates spread
 * about a percent across the reconstructions here, and [K1] and [K2] carry that through.
 */
object LensCalibration {

    /**
     * Focal length as a fraction of the frame's width.
     *
     * Stored this way because the app produces frames at more than one size -- a Stokes
     * grid is 612x512 unbinned and 306x256 when the camera bins -- and a focal length in
     * pixels belongs to one of them. Binning changes how the field is sampled, never how
     * wide it is, so this fraction is the same for all of them.
     *
     * 907.8 px on a 1224-wide export. Against the 3.45 um sensor pitch that is 6.26 mm,
     * for a lens sold as 6 mm.
     *
     * Five reconstructions put it between 899 and 933, and focal length trades against the
     * radial terms, so this is soft at about two percent. It does not matter for
     * straightening: every candidate pair below straightens the archive's edges to within
     * one percent of every other. It would matter for measuring a distance in a scene.
     */
    const val FOCAL_PER_WIDTH = 907.8f / 1224f

    /**
     * First radial term, in the usual normalized convention: a point that ideally sits at
     * radius r from the axis, measured in focal lengths, lands at
     * r * (1 + [K1]*r^2 + [K2]*r^4).
     *
     * Negative is barrel, which is what this lens has.
     */
    const val K1 = -0.1636f

    /**
     * Second radial term, which pulls the correction back at the very edge.
     *
     * Positive against a negative [K1]: the barrel grows less quickly than r^2 out at the
     * corners than a single term would have it. Fitting only the first term forces it to
     * compromise across the whole frame and it settles near -0.108, which is right nowhere
     * in particular -- too weak in the middle distance and too strong at the corner.
     */
    const val K2 = 0.0657f

    /**
     * Straightens one rendered frame, [width] by [height], returning it.
     *
     * Works on the picture rather than on the Stokes arrays, which is deliberate. The
     * archive is the raw mosaic and the components computed from it, and nothing on the
     * viewing path is allowed to change either; this produces a second picture from the
     * first and leaves the measurement alone.
     *
     * The output keeps the calibrated focal length, so the middle of the frame is
     * untouched and only the periphery moves. That makes the on-off comparison honest --
     * a difference on screen is the distortion and not a change of zoom -- at the cost of
     * the extreme corners of the sensor falling outside the straightened frame.
     *
     * @param pixels ARGB, row-major, [width] * [height] long
     */
    fun rectify(pixels: IntArray, width: Int, height: Int): IntArray {
        require(pixels.size >= width * height) { "pixels shorter than $width x $height" }
        val out = IntArray(width * height)
        val f = FOCAL_PER_WIDTH * width
        val cx = (width - 1) / 2f
        val cy = (height - 1) / 2f
        var i = 0
        for (y in 0 until height) {
            val ny = (y - cy) / f
            for (x in 0 until width) {
                val nx = (x - cx) / f
                // Every output pixel asks where on the sensor it came from, so only the
                // forward model is needed and no radial equation has to be inverted.
                val r2 = nx * nx + ny * ny
                val s = 1f + K1 * r2 + K2 * r2 * r2
                out[i++] = sample(pixels, width, height, cx + nx * s * f, cy + ny * s * f)
            }
        }
        return out
    }

    /**
     * Bilinear sample of an ARGB frame, per channel, clamped at the edges.
     *
     * Nearest-neighbour would leave the straightened edges visibly stepped, which is the
     * one thing this is for. Sampling off the frame returns the edge rather than nothing,
     * so a barrel correction cannot ring against a hard black border.
     */
    private fun sample(pixels: IntArray, width: Int, height: Int, fx: Float, fy: Float): Int {
        val cx = fx.coerceIn(0f, (width - 1).toFloat())
        val cy = fy.coerceIn(0f, (height - 1).toFloat())
        val x0 = cx.toInt().coerceAtMost(width - 1)
        val y0 = cy.toInt().coerceAtMost(height - 1)
        val x1 = (x0 + 1).coerceAtMost(width - 1)
        val y1 = (y0 + 1).coerceAtMost(height - 1)
        val tx = cx - x0
        val ty = cy - y0
        val p00 = pixels[y0 * width + x0]
        val p10 = pixels[y0 * width + x1]
        val p01 = pixels[y1 * width + x0]
        val p11 = pixels[y1 * width + x1]
        var argb = 0
        for (shift in intArrayOf(24, 16, 8, 0)) {
            val a = ((p00 ushr shift) and 0xFF) * (1 - tx) + ((p10 ushr shift) and 0xFF) * tx
            val b = ((p01 ushr shift) and 0xFF) * (1 - tx) + ((p11 ushr shift) and 0xFF) * tx
            val v = (a * (1 - ty) + b * ty).roundToInt().coerceIn(0, 255)
            argb = argb or (v shl shift)
        }
        return argb
    }

    /**
     * Where a point at [x], [y] in the straightened frame came from on the sensor.
     *
     * The model itself, for anything that needs the geometry rather than the pixels.
     */
    fun toSensor(x: Float, y: Float, width: Int, height: Int): FloatArray {
        val f = FOCAL_PER_WIDTH * width
        val cx = (width - 1) / 2f
        val cy = (height - 1) / 2f
        val nx = (x - cx) / f
        val ny = (y - cy) / f
        val r2 = nx * nx + ny * ny
        val s = 1f + K1 * r2 + K2 * r2 * r2
        return floatArrayOf(cx + nx * s * f, cy + ny * s * f)
    }

    /**
     * How far the corner of a [width] by [height] frame moves when it is straightened.
     *
     * For showing the size of the correction next to the control that applies it, so the
     * number a reader is trusting is on screen rather than in this file.
     */
    fun cornerShiftPixels(width: Int, height: Int): Float {
        val corner = toSensor((width - 1).toFloat(), (height - 1).toFloat(), width, height)
        return hypot((width - 1) / 2f, (height - 1) / 2f) -
            hypot(corner[0] - (width - 1) / 2f, corner[1] - (height - 1) / 2f)
    }
}
