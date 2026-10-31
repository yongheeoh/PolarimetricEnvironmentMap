package com.cgjnkim.mobileokad.flir

/**
 * Smooths S1 and S2 without smoothing the picture, by letting S0 say where the edges are.
 *
 * DoLP is a magnitude over S0, and where the light is thin the components it is built from
 * are mostly noise. Averaging them over a neighbourhood is the obvious cure and the wrong
 * one on its own: a plain blur crosses every boundary in the scene and drags one surface's
 * polarization onto the next.
 *
 * A guided filter avoids that by fitting S1 to S0 locally rather than averaging it. Inside
 * a region where S0 is flat the fit degenerates to the local mean -- full smoothing, which
 * is what a featureless patch wants. Across an edge, where S0 varies, the fit follows S0
 * and the output keeps the edge. The scene's own intensity is the only guide available and
 * it is a good one: a boundary in polarization is nearly always a boundary in brightness
 * too, because both come from a change of surface.
 *
 * ### What it is worth, measured
 *
 * On seven well-exposed frames from a morning of shooting, the DoLP of the darkest quarter
 * came down by a median factor of **1.20**, and the polarization in the brightest quarter
 * lost a median of **2.4 percent**. Both numbers vary a lot by scene, and the spread is
 * the important part:
 *
 *     dim and smooth   1.47x  for 1.4 percent
 *     bright, textured 1.02x  for 10.4 percent
 *
 * On the worst frame it is a straight loss -- nothing removed, a tenth of the signal
 * softened. It earns its place on dim, smooth scenes, which is also where DoLP noise is
 * the thing anyone complains about, and it should be switched off when it is not.
 *
 * ### And where the ceiling is
 *
 * A wider window buys almost nothing and costs steadily more: over the same frames, radius
 * 1 gives 1.20x for 2.4 percent, radius 3 gives 1.27x for 4.1 percent, and radius 5 gives
 * 1.23x for 4.3 percent. Two things cap it, and neither is fixed by filtering harder.
 *
 * The first is that most of what is left is not noise. The noise model predicts a floor of
 * 0.048 for those frames while the raw dark-quarter DoLP is 0.126 -- more than twice as
 * much -- so the majority of it was real polarization before any filter touched it.
 *
 * The second is that the frames arrive pre-smoothed. Over 526 captures the gyroscope put
 * the median hand movement at 3.3 Stokes pixels during the exposure, so the polarization
 * field has already been low-passed by more than this filter's own window before it
 * reached the file. A shorter exposure would give this something to do.
 *
 * Nothing here is stored. It is a viewing choice on the archive, which stays the raw
 * mosaic and the raw Stokes arrays.
 */
object PolarSmoothing {

    /**
     * Radius that gets the whole benefit.
     *
     * One pixel. Measured over seven captures, going wider adds at most 0.07 to the
     * factor while roughly doubling what it takes out of the real signal, and the frames
     * already carry three pixels of motion blur -- so a bigger window trades structure for
     * very little.
     */
    const val DEFAULT_RADIUS = 1

    /**
     * How much of a change in S0 counts as an edge, relative to the local level.
     *
     * Below it the filter averages freely; above it the output follows S0 instead. Two
     * percent sits well under any real boundary and well over the shot noise on S0, which
     * at these levels is a fraction of a percent.
     */
    const val DEFAULT_EPSILON = 0.02f

    /**
     * S1 and S2 fitted to [s0] over a `2 * radius + 1` window.
     *
     * @return two new planes, in that order. The inputs are untouched.
     */
    fun guided(
        s0: FloatArray,
        s1: FloatArray,
        s2: FloatArray,
        width: Int,
        height: Int,
        radius: Int = DEFAULT_RADIUS,
        epsilon: Float = DEFAULT_EPSILON,
    ): Array<FloatArray> {
        val n = width * height
        if (radius < 1 || n == 0 || s0.size < n || s1.size < n || s2.size < n) {
            return arrayOf(s1.copyOf(), s2.copyOf())
        }

        val integral = Integral(width, height, radius)
        val meanS0 = integral.box(s0)
        val meanS0Sq = integral.box(FloatArray(n) { s0[it] * s0[it] })

        // Variance of the guide, which is what decides between averaging and following.
        val varS0 = FloatArray(n) { (meanS0Sq[it] - meanS0[it] * meanS0[it]).coerceAtLeast(0f) }

        return arrayOf(
            fit(s0, s1, meanS0, varS0, integral, n, epsilon),
            fit(s0, s2, meanS0, varS0, integral, n, epsilon),
        )
    }

    /** One component, fitted as `a * s0 + b` over each window and the fits then averaged. */
    private fun fit(
        s0: FloatArray,
        x: FloatArray,
        meanS0: FloatArray,
        varS0: FloatArray,
        integral: Integral,
        n: Int,
        epsilon: Float,
    ): FloatArray {
        val meanX = integral.box(x)
        val meanS0X = integral.box(FloatArray(n) { s0[it] * x[it] })

        val a = FloatArray(n)
        val b = FloatArray(n)
        for (i in 0 until n) {
            // Scaled to the local level, so the same setting means the same thing in a
            // bright part of the frame and a dark one.
            val level = meanS0[i].coerceAtLeast(1f) * epsilon
            a[i] = (meanS0X[i] - meanS0[i] * meanX[i]) / (varS0[i] + level * level)
            b[i] = meanX[i] - a[i] * meanS0[i]
        }
        // Averaging the coefficients rather than applying them where they were computed is
        // what keeps the output continuous; every pixel is covered by many windows and
        // this is their consensus.
        val meanA = integral.box(a)
        val meanB = integral.box(b)
        return FloatArray(n) { meanA[it] * s0[it] + meanB[it] }
    }

    /**
     * Box means over a fixed window, in time that does not depend on the window's size.
     *
     * A summed-area table, with the border repeated so an edge pixel averages the picture
     * rather than a fringe of zeros. Kept in doubles: the table accumulates the whole
     * frame, which for S0 near full scale on half a megapixel passes what a float can hold
     * without losing counts.
     */
    private class Integral(val width: Int, val height: Int, val radius: Int) {
        private val stride = width + 2 * radius + 1
        private val table = DoubleArray(stride * (height + 2 * radius + 1))
        private val window = (2 * radius + 1) * (2 * radius + 1)

        fun box(plane: FloatArray): FloatArray {
            val r = radius
            java.util.Arrays.fill(table, 0.0)
            for (y in 0 until height + 2 * r) {
                val sy = (y - r).coerceIn(0, height - 1) * width
                var row = 0.0
                val out = (y + 1) * stride
                val above = y * stride
                for (x in 0 until width + 2 * r) {
                    row += plane[sy + (x - r).coerceIn(0, width - 1)]
                    table[out + x + 1] = table[above + x + 1] + row
                }
            }
            val n = 2 * r + 1
            val result = FloatArray(width * height)
            for (y in 0 until height) {
                val top = y * stride
                val bottom = (y + n) * stride
                val outRow = y * width
                for (x in 0 until width) {
                    val sum = table[bottom + x + n] - table[top + x + n] -
                        table[bottom + x] + table[top + x]
                    result[outRow + x] = (sum / window).toFloat()
                }
            }
            return result
        }
    }
}
