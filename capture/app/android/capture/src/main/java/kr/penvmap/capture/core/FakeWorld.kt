package kr.penvmap.capture.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * A synthetic environment seen through the real lens model, for testing the app without a camera:
 * the fake camera renders what a FLIR on the stage would see at the fake stage's current pose, so a
 * rotation stitches into a recognisable panorama (sky gradient, sun, a horizon, a 30-degree grid) and
 * every part of the pipeline -- bracket, files, live stitch -- is exercised end to end.
 */
class FakeWorld(private val intrinsics: Intrinsics, val width: Int = 2448, val height: Int = 2048) {

    @Volatile var yawDeg = 0.0
    @Volatile var pitchDeg = 0.0
    @Volatile var mountRollDeg = 90.0

    private val quarter = intrinsics.scaled(0.25, 0.25)

    /**
     * Non-repeating landmarks (random blobs of random size and colour) so feature matching -- main.py's
     * camera-rotation optimisation -- has something unambiguous to lock onto; a bare grid repeats every
     * 30 degrees and gets matched to the wrong cell.
     */
    private class Blob(val x: Double, val y: Double, val z: Double, val cosRadius: Double, val sharp: Double, val gain: DoubleArray)

    private val blobs: List<Blob> = run {
        val rnd = java.util.Random(20261008)
        List(140) {
            var x: Double
            var y: Double
            var z: Double
            do {
                x = rnd.nextGaussian(); y = rnd.nextGaussian(); z = rnd.nextGaussian()
            } while (x * x + y * y + z * z < 1e-6)
            val n = sqrt(x * x + y * y + z * z)
            val radius = Math.toRadians(1.5 + 7.0 * rnd.nextDouble())
            Blob(x / n, y / n, z / n, kotlin.math.cos(3 * radius), 1.0 / (radius * radius),
                DoubleArray(3) { 0.3 + 1.4 * rnd.nextDouble() })
        }
    }

    /** Radiance (BGR, ~0..1e4 like a daylight scene's relative range) towards world direction d (Y up). */
    fun radiance(dx: Double, dy: Double, dz: Double, out: DoubleArray) {
        val theta = acos(dy.coerceIn(-1.0, 1.0)) // 0 = zenith
        val phi = atan2(dx, -dz)
        val sun = doubleArrayOf(0.5, 0.6, -0.62).let { s -> val n = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]); DoubleArray(3) { s[it] / n } }
        val cosSun = dx * sun[0] + dy * sun[1] + dz * sun[2]
        val grid = run {
            val a = Math.toDegrees(phi) / 30.0
            val b = Math.toDegrees(theta) / 30.0
            val near = min(abs(a - Math.rint(a)), abs(b - Math.rint(b)))
            if (near < 0.03) 0.35 else 1.0
        }
        if (dy > 0) {
            val sky = 0.25 + 0.75 * (1 - dy)
            val sunGlow = 40.0 * exp(-(1 - cosSun) * 400.0) + 2.0 * exp(-(1 - cosSun) * 8.0)
            out[0] = (0.9 * sky + sunGlow) * grid
            out[1] = (0.6 * sky + sunGlow) * grid
            out[2] = (0.35 * sky + sunGlow) * grid
        } else {
            val ground = 0.12 + 0.05 * ((Math.toDegrees(phi) + 360) % 60 / 60.0)
            out[0] = 0.25 * ground * grid
            out[1] = 0.35 * ground * grid
            out[2] = 0.45 * ground * grid
        }
        for (b in blobs) {
            val c = dx * b.x + dy * b.y + dz * b.z
            if (c < b.cosRadius) continue
            val w = exp(-(2 * (1 - c)) * b.sharp) // ~ exp(-angle^2 / radius^2)
            for (ch in 0..2) out[ch] *= 1 + (b.gain[ch] - 1) * w
        }
    }

    /**
     * Linear BGR radiance per superpixel (width/4 x height/4) for the current pose: the inverse of
     * the stitcher's projection (pixel -> undistorted ray -> world).
     */
    fun superpixels(): FloatArray {
        val w4 = width / 4
        val h4 = height / 4
        val rWorldToCam = Rot.fromRvec(Pose.priorRvec(yawDeg, pitchDeg, mountRollDeg))
        val camToWorld = Rot.transpose(rWorldToCam)
        val out = FloatArray(w4 * h4 * 3)
        val rgb = DoubleArray(3)
        val k = quarter.k
        val dist = quarter.dist
        for (y in 0 until h4) for (x in 0 until w4) {
            // undistort (fixed-point iteration, as cv2.undistortPoints)
            val xd = (x - k[2]) / k[0]
            val yd = (y - k[5]) / k[4]
            var xu = xd
            var yu = yd
            repeat(8) {
                val r2 = xu * xu + yu * yu
                val radial = 1 + r2 * (dist.getOrElse(0) { 0.0 } + r2 * (dist.getOrElse(1) { 0.0 } + r2 * dist.getOrElse(4) { 0.0 }))
                xu = xd / radial
                yu = yd / radial
            }
            val n = sqrt(xu * xu + yu * yu + 1)
            val cx = xu / n
            val cy = yu / n
            val cz = 1 / n
            val wx = camToWorld[0] * cx + camToWorld[1] * cy + camToWorld[2] * cz
            val wy = camToWorld[3] * cx + camToWorld[4] * cy + camToWorld[5] * cz
            val wz = camToWorld[6] * cx + camToWorld[7] * cy + camToWorld[8] * cz
            radiance(wx, wy, wz, rgb)
            // fisheye-ish image circle: dark outside ~95 deg off axis, like the 4 mm lens
            val inCircle = if (acos(cz) > Math.toRadians(95.0)) 0.0 else 1.0
            val o = (y * w4 + x) * 3
            out[o] = (rgb[0] * inCircle).toFloat()
            out[o + 1] = (rgb[1] * inCircle).toFloat()
            out[o + 2] = (rgb[2] * inCircle).toFloat()
        }
        return out
    }

    /**
     * A raw mosaic at [exposureUs]: each superpixel's colour blocks filled with its radiance times the
     * exposure ([countsPerUs] per unit radiance), 12-bit, MSB aligned when [bits16] (BayerRG16), else
     * the 8-bit preview. A little linear polarization (DoLP ~ 0.2, sky-like) is put on the analysers.
     */
    fun mosaic(exposureUs: Double, bits16: Boolean, countsPerUs: Double = 0.6, radiance: FloatArray = superpixels()): ByteArray {
        val w4 = width / 4
        val h4 = height / 4
        val out = ByteArray(width * height * (if (bits16) 2 else 1))
        // analyser order in each 2x2 cell: (0,0)=90, (0,1)=45, (1,0)=135, (1,1)=0  (FLIR / polanalyser layout)
        val analyser = doubleArrayOf(90.0, 45.0, 135.0, 0.0)
        val aolp = Math.toRadians(30.0)
        val dolp = 0.2
        val gainBy = DoubleArray(4) { 1 + dolp * kotlin.math.cos(2 * (Math.toRadians(analyser[it]) - aolp)) }
        for (y in 0 until h4) for (x in 0 until w4) {
            val o = (y * w4 + x) * 3
            val bgr = floatArrayOf(radiance[o], radiance[o + 1], radiance[o + 2])
            for (cy in 0..1) for (cx in 0..1) {
                val colour = when {
                    cy == 0 && cx == 0 -> bgr[2]
                    cy == 1 && cx == 1 -> bgr[0]
                    else -> bgr[1]
                }
                for (py in 0..1) for (px in 0..1) {
                    val counts = colour * exposureUs * countsPerUs * gainBy[py * 2 + px]
                    val v12 = max(0.0, min(4095.0, counts)).toInt()
                    val row = y * 4 + cy * 2 + py
                    val col = x * 4 + cx * 2 + px
                    if (bits16) {
                        val v = v12 shl 4
                        val i = (row * width + col) * 2
                        out[i] = (v and 0xFF).toByte()
                        out[i + 1] = (v shr 8).toByte()
                    } else {
                        out[row * width + col] = (v12 shr 4).toByte()
                    }
                }
            }
        }
        return out
    }

    companion object {
        /** Degrees for PI, kept here so the class has no android dependency for tests. */
        const val DEG = 180.0 / PI
    }
}
