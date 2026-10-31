package kr.penvmap.capture.core

import kotlin.math.pow

/**
 * S0 preview from the raw mosaic without demosaicing (capture/utils.py `mosaic_s0`, `s0_to_u8`).
 *
 * The FLIR's 4x4 superpixel is 2x2 colour blocks [[R, G], [G, B]], each holding 2x2 analyser pixels
 * (0/45/90/135). Averaging a block's 4 analyser pixels gives a value proportional to that colour's S0:
 * enough for framing, exposure and the live envmap -- pemap does the real demosaicing from the HDR stack.
 */
object Mosaic {
    const val SUPERPIXEL = 4

    class S0(val width: Int, val height: Int, /** Interleaved BGR, linear, full scale = 1. */ val bgr: FloatArray)

    /** Superpixel S0 at 1/4 resolution. [samples] is the raw mosaic (width x height). */
    fun s0(samples: Exposure.Samples, width: Int, height: Int, fullScale: Double): S0 {
        val w4 = width / SUPERPIXEL
        val h4 = height / SUPERPIXEL
        val out = FloatArray(w4 * h4 * 3)
        val inv = (1.0 / fullScale).toFloat()
        for (by in 0 until h4) {
            val row0 = by * SUPERPIXEL
            for (bx in 0 until w4) {
                val col0 = bx * SUPERPIXEL
                // block (cy, cx): mean of its 2x2 analyser pixels
                fun block(cy: Int, cx: Int): Float {
                    val r = row0 + cy * 2
                    val c = col0 + cx * 2
                    val s = samples[r * width + c] + samples[r * width + c + 1] +
                        samples[(r + 1) * width + c] + samples[(r + 1) * width + c + 1]
                    return s / 4f
                }
                val red = block(0, 0)
                val green = 0.5f * (block(0, 1) + block(1, 0))
                val blue = block(1, 1)
                val o = (by * w4 + bx) * 3
                out[o] = blue * inv
                out[o + 1] = green * inv
                out[o + 2] = red * inv
            }
        }
        return S0(w4, h4, out)
    }

    /**
     * Raw preview: the 8-bit mosaic values as they are, grey (no colour, no polarizer averaging, no
     * normalisation, no gamma), so 255 = saturated shows as pure white. Each output pixel is the MAX of a
     * [pool] x [pool] block: half resolution is about the screen's, and one saturated pixel still shows.
     * Returns interleaved 3-channel grey (for [rotate]) and its size.
     */
    fun rawGray(pixels: ByteArray, width: Int, height: Int, pool: Int = 2): Triple<Int, Int, ByteArray> {
        val wp = width / pool
        val hp = height / pool
        val out = ByteArray(wp * hp * 3)
        for (py in 0 until hp) {
            val y0 = py * pool
            for (px in 0 until wp) {
                val x0 = px * pool
                var m = 0
                for (dy in 0 until pool) {
                    val row = (y0 + dy) * width + x0
                    for (dx in 0 until pool) {
                        val v = pixels[row + dx].toInt() and 0xFF
                        if (v > m) m = v
                    }
                }
                val o = (py * wp + px) * 3
                val b = m.toByte()
                out[o] = b
                out[o + 1] = b
                out[o + 2] = b
            }
        }
        return Triple(wp, hp, out)
    }

    /** numpy.percentile (linear interpolation) of [values]. */
    fun percentile(values: FloatArray, q: Double, count: Int = values.size): Double {
        if (count == 0) return Double.NaN
        val sorted = values.copyOf(count)
        sorted.sort()
        val pos = q / 100.0 * (count - 1)
        val lo = pos.toInt()
        val hi = minOf(lo + 1, count - 1)
        val frac = pos - lo
        return sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }

    /**
     * Linear S0 -> uint8 BGR, normalised at the 99th percentile with gamma 2.2 (`s0_to_u8`).
     * [subsample] > 1 estimates the percentile from every n-th value (phone preview speed).
     */
    fun toU8(bgr: FloatArray, gamma: Double = 2.2, subsample: Int = 1): ByteArray {
        val src = if (subsample <= 1) bgr else FloatArray((bgr.size + subsample - 1) / subsample) { bgr[it * subsample] }
        val ref = maxOf(percentile(src, 99.0), 1e-8)
        return encode(bgr, ref, gamma)
    }

    /** Global tonemap of a stitched map: percentile over the covered (> 0) values only (`tonemap`). */
    fun tonemap(bgr: FloatArray, percentile: Double = 99.0, gamma: Double = 2.2): ByteArray {
        val valid = bgr.filter { it > 0f }.toFloatArray()
        val ref = maxOf(if (valid.isNotEmpty()) percentile(valid, percentile) else 1.0, 1e-8)
        return encode(bgr, ref, gamma)
    }

    private fun encode(values: FloatArray, ref: Double, gamma: Double): ByteArray {
        val g = 1.0 / gamma
        val out = ByteArray(values.size)
        for (i in values.indices) {
            val x = (values[i] / ref).coerceIn(0.0, 1.0)
            out[i] = (x.pow(g) * 255.0).toInt().toByte()
        }
        return out
    }

    /**
     * Full-resolution mask (true = valid) -> mask at the S0 resolution: a superpixel is valid only if all
     * of its pixels are (capture/utils.py `downscale_mask`).
     */
    fun downscaleMask(mask: BooleanArray, width: Int, height: Int, factor: Int = SUPERPIXEL): BooleanArray {
        val w4 = width / factor
        val h4 = height / factor
        return BooleanArray(w4 * h4) { i ->
            val bx = i % w4
            val by = i / w4
            var ok = true
            loop@ for (y in by * factor until (by + 1) * factor) for (x in bx * factor until (bx + 1) * factor) {
                if (!mask[y * width + x]) {
                    ok = false
                    break@loop
                }
            }
            ok
        }
    }

    /** Rotates an interleaved 3-channel image for display ([Pose.Upright]); returns (width, height, data). */
    fun rotate(src: ByteArray, width: Int, height: Int, how: Pose.Upright): Triple<Int, Int, ByteArray> {
        if (how == Pose.Upright.NONE) return Triple(width, height, src)
        val out = ByteArray(src.size)
        val (w2, h2) = if (how == Pose.Upright.R180) width to height else height to width
        for (y in 0 until height) for (x in 0 until width) {
            val (nx, ny) = when (how) {
                Pose.Upright.CCW -> y to (width - 1 - x)       // cv2.ROTATE_90_COUNTERCLOCKWISE
                Pose.Upright.CW -> (height - 1 - y) to x       // cv2.ROTATE_90_CLOCKWISE
                Pose.Upright.R180 -> (width - 1 - x) to (height - 1 - y)
                Pose.Upright.NONE -> x to y
            }
            val s = (y * width + x) * 3
            val d = (ny * w2 + nx) * 3
            out[d] = src[s]
            out[d + 1] = src[s + 1]
            out[d + 2] = src[s + 2]
        }
        return Triple(w2, h2, out)
    }
}
