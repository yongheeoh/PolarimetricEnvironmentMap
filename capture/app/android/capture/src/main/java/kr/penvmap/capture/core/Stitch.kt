package kr.penvmap.capture.core

import java.io.File
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin

/** Pinhole + OpenCV 5-coefficient distortion (k1, k2, p1, p2, k3), as in the calibration JSON. */
class Intrinsics(val k: DoubleArray, val dist: DoubleArray) {

    /** Camera matrix for an image resized by (sx, sy), pixel-centre convention (capture/stitch.py). */
    fun scaled(sx: Double, sy: Double): Intrinsics {
        val s = k.copyOf()
        s[0] *= sx
        s[4] *= sy
        s[2] = sx * (k[2] + 0.5) - 0.5
        s[5] = sy * (k[5] + 0.5) - 0.5
        return Intrinsics(s, dist)
    }

    /** cv2.projectPoints of one camera-frame point (x, y, z); returns (u, v). */
    fun project(x: Double, y: Double, z: Double, out: DoubleArray) {
        val xp = x / z
        val yp = y / z
        val r2 = xp * xp + yp * yp
        val k1 = dist.getOrElse(0) { 0.0 }
        val k2 = dist.getOrElse(1) { 0.0 }
        val p1 = dist.getOrElse(2) { 0.0 }
        val p2 = dist.getOrElse(3) { 0.0 }
        val k3 = dist.getOrElse(4) { 0.0 }
        val radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        val xd = xp * radial + 2 * p1 * xp * yp + p2 * (r2 + 2 * xp * xp)
        val yd = yp * radial + p1 * (r2 + 2 * yp * yp) + 2 * p2 * xp * yp
        out[0] = k[0] * xd + k[1] * yd + k[2]
        out[1] = k[4] * yd + k[5]
    }

    companion object {
        /** The repo's calibration JSON (`camera_matrix.values`, `dist_coeffs.values`). */
        fun fromJson(text: String): Intrinsics {
            val root = Json.parse(text).asMap()
            val km = root["camera_matrix"].asMap()["values"].asList().flatMap { it.asList() }.map { it.asDouble() }
            val d = root["dist_coeffs"].asMap()["values"].asList().map { it.asDouble() }
            return Intrinsics(km.toDoubleArray(), d.toDoubleArray())
        }

        fun fromFile(file: File) = fromJson(file.readText())
    }
}

/**
 * Incremental average-blend equirectangular stitcher for on-site checking (capture/stitch.py
 * `LiveStitcher`): prior rotations only, plain average in the overlaps, low resolution. Same
 * convention as pemap.geometry.spherical_coordinates: directions (sinθ sinφ, cosθ, -sinθ cosφ), Y up, points behind the
 * camera dropped.
 *
 * [feather] (the app's live view, not the PC) weights each view by cos⁴ of the angle from its optical
 * axis, so the darker, vignetted rim of one fisheye view does not draw arcs over the centre of another.
 */
class LiveStitcher(private val intrinsics: Intrinsics, val width: Int, val height: Int, private val feather: Boolean = false) {

    private val dirs = directions(width, height)
    val acc = DoubleArray(width * height * 3)
    val count = IntArray(width * height)
    private val weights = if (feather) DoubleArray(width * height) else null

    /**
     * Warps one interleaved BGR float image (w x h, intrinsics must match that size) with its
     * world-to-camera rvec and accumulates it. [mask] (w*h, true = valid) excludes calibration-masked
     * pixels: a target is blended only where all four interpolation taps are valid.
     */
    fun add(bgr: FloatArray, w: Int, h: Int, rvec: DoubleArray, mask: BooleanArray? = null) {
        val r = Rot.fromRvec(rvec)
        val uv = DoubleArray(2)
        val maxCoord = maxOf(w, h).toDouble()
        for (i in 0 until width * height) {
            val dx = dirs[3 * i]
            val dy = dirs[3 * i + 1]
            val dz = dirs[3 * i + 2]
            val cx = r[0] * dx + r[1] * dy + r[2] * dz
            val cy = r[3] * dx + r[4] * dy + r[5] * dz
            val cz = r[6] * dx + r[7] * dy + r[8] * dz
            if (cz <= 0) continue
            intrinsics.project(cx, cy, cz, uv)
            if (!uv[0].isFinite() || !uv[1].isFinite()) continue
            // OpenCV clips the map to [-1, max(w, h)] before remapping; inside means [0, w-1] x [0, h-1].
            val u = uv[0].coerceIn(-1.0, maxCoord).toFloat().toDouble()
            val v = uv[1].coerceIn(-1.0, maxCoord).toFloat().toDouble()
            if (u < 0 || u > w - 1 || v < 0 || v > h - 1) continue
            val x0 = floor(u).toInt()
            val y0 = floor(v).toInt()
            val fx = u - x0
            val fy = v - y0
            val x1 = minOf(x0 + 1, w - 1)
            val y1 = minOf(y0 + 1, h - 1)
            val w00 = (1 - fx) * (1 - fy)
            val w01 = fx * (1 - fy)
            val w10 = (1 - fx) * fy
            val w11 = fx * fy
            if (mask != null) {
                // PC: remap the mask bilinearly and keep targets above 0.999 (all taps valid).
                fun m(yy: Int, xx: Int, inImage: Boolean) = if (inImage && mask[yy * w + xx]) 1.0 else 0.0
                val warped = w00 * m(y0, x0, true) + w01 * m(y0, x1, x0 + 1 < w) +
                    w10 * m(y1, x0, y0 + 1 < h) + w11 * m(y1, x1, x0 + 1 < w && y0 + 1 < h)
                if (warped <= 0.999) continue
            }
            val o = 3 * i
            // cos of the angle from the optical axis (rotation keeps the unit length): 1 centre .. ~0.6 rim
            val viewWeight = if (feather) cz * cz * cz * cz else 1.0
            for (ch in 0..2) {
                // Taps outside the image (x0+1 == w at the right edge) carry zero weight, like BORDER_CONSTANT.
                val a = bgr[(y0 * w + x0) * 3 + ch]
                val b = if (x0 + 1 < w) bgr[(y0 * w + x1) * 3 + ch] else 0f
                val c = if (y0 + 1 < h) bgr[(y1 * w + x0) * 3 + ch] else 0f
                val d = if (x0 + 1 < w && y0 + 1 < h) bgr[(y1 * w + x1) * 3 + ch] else 0f
                acc[o + ch] += viewWeight * (w00 * a + w01 * b + w10 * c + w11 * d)
            }
            count[i]++
            weights?.let { it[i] += viewWeight }
        }
    }

    /** Current map (interleaved BGR float), 0 where uncovered. */
    fun result(): FloatArray = FloatArray(width * height * 3) { j ->
        val n = weights?.get(j / 3) ?: count[j / 3].toDouble()
        if (n > 0) (acc[j] / n).toFloat() else 0f
    }

    fun uncoveredFraction(): Double = count.count { it == 0 }.toDouble() / count.size

    companion object {
        /** Y-up world directions per envmap pixel, poles excluded (float32 like the PC). */
        fun directions(width: Int, height: Int): DoubleArray {
            val out = DoubleArray(width * height * 3)
            for (row in 0 until height) {
                val theta = (PI * (row + 1) / (height + 1)).toFloat()
                val st = sin(theta.toDouble()).toFloat()
                val ct = cos(theta.toDouble()).toFloat()
                for (col in 0 until width) {
                    val phi = (2 * PI * col / width).toFloat()
                    val sp = sin(phi.toDouble()).toFloat()
                    val cp = cos(phi.toDouble()).toFloat()
                    val i = 3 * (row * width + col)
                    out[i] = (st * sp).toDouble()
                    out[i + 1] = ct.toDouble()
                    out[i + 2] = (-st * cp).toDouble()
                }
            }
            return out
        }
    }
}
