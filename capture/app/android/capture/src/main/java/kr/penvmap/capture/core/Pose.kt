package kr.penvmap.capture.core

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/** Row-major 3x3 rotation helpers and OpenCV-compatible Rodrigues conversion. */
object Rot {

    fun mul(a: DoubleArray, b: DoubleArray): DoubleArray {
        val c = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += a[3 * i + k] * b[3 * k + j]
            c[3 * i + j] = s
        }
        return c
    }

    fun transpose(a: DoubleArray): DoubleArray = DoubleArray(9) { a[3 * (it % 3) + it / 3] }

    /** Rotation vector -> matrix (cv2.Rodrigues). */
    fun fromRvec(r: DoubleArray): DoubleArray {
        val theta = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        if (theta < 1e-12) return doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        val x = r[0] / theta
        val y = r[1] / theta
        val z = r[2] / theta
        val c = cos(theta)
        val s = sin(theta)
        val c1 = 1 - c
        return doubleArrayOf(
            c + c1 * x * x, c1 * x * y - s * z, c1 * x * z + s * y,
            c1 * y * x + s * z, c + c1 * y * y, c1 * y * z - s * x,
            c1 * z * x - s * y, c1 * z * y + s * x, c + c1 * z * z,
        )
    }

    /**
     * Matrix -> rotation vector, following OpenCV's cvRodrigues2 (including its branch for angles
     * near pi), so the prior rvec written to `hdr/stack/{i}.json` matches the PC's cv2.Rodrigues output.
     */
    fun toRvec(m: DoubleArray): DoubleArray {
        val rx0 = m[7] - m[5]
        val ry0 = m[2] - m[6]
        val rz0 = m[3] - m[1]
        val s = sqrt((rx0 * rx0 + ry0 * ry0 + rz0 * rz0) * 0.25)
        var c = (m[0] + m[4] + m[8] - 1) * 0.5
        c = c.coerceIn(-1.0, 1.0)
        val theta = acos(c)
        if (s < 1e-5) {
            if (c > 0) return doubleArrayOf(0.0, 0.0, 0.0)
            var rx = sqrt(max((m[0] + 1) * 0.5, 0.0))
            val ry = sqrt(max((m[4] + 1) * 0.5, 0.0)) * (if (m[1] < 0) -1.0 else 1.0)
            var rz = sqrt(max((m[8] + 1) * 0.5, 0.0)) * (if (m[2] < 0) -1.0 else 1.0)
            if (abs(rx) < abs(ry) && abs(rx) < abs(rz) && (m[5] > 0) != (ry * rz > 0)) rz = -rz
            val n = sqrt(rx * rx + ry * ry + rz * rz)
            val k = theta / n
            rx *= k
            return doubleArrayOf(rx, ry * k, rz * k)
        }
        val v = 1.0 / (2 * s) * theta
        return doubleArrayOf(rx0 * v, ry0 * v, rz0 * v)
    }
}

object Pose {

    /**
     * Stage yaw / pitch (deg) -> world-to-camera Rodrigues vector, the stitching prior.
     * Port of capture/utils.py `prior_rvec` (stage angles negated for the equirectangular convention).
     */
    fun priorRvec(yawDeg: Double, pitchDeg: Double, mountRollDeg: Double): DoubleArray {
        val yaw = Math.toRadians(-yawDeg)
        val pitch = Math.toRadians(-pitchDeg)
        val roll = Math.toRadians(mountRollDeg)
        val rYaw = doubleArrayOf(cos(yaw), 0.0, sin(yaw), 0.0, 1.0, 0.0, -sin(yaw), 0.0, cos(yaw))
        val rPitch = doubleArrayOf(1.0, 0.0, 0.0, 0.0, cos(pitch), -sin(pitch), 0.0, sin(pitch), cos(pitch))
        val rRoll = doubleArrayOf(cos(roll), -sin(roll), 0.0, sin(roll), cos(roll), 0.0, 0.0, 0.0, 1.0)
        val camToWorld = Rot.mul(Rot.mul(rYaw, rPitch), rRoll)
        return Rot.toRvec(Rot.transpose(camToWorld))
    }

    enum class Upright { NONE, CCW, CW, R180 }

    /**
     * How to turn a sensor-oriented image upright (sky on top) for this mount -- capture/utils.py
     * `upright_rotate_code`. Display only; saved data stays in sensor orientation.
     */
    fun upright(mountRollDeg: Double): Upright {
        val r = Rot.fromRvec(priorRvec(0.0, 0.0, mountRollDeg))
        val x = r[1]
        val y = r[4] // R @ (0, 1, 0)
        return if (abs(x) >= abs(y)) {
            if (x > 0) Upright.CCW else Upright.CW
        } else {
            if (y > 0) Upright.R180 else Upright.NONE
        }
    }
}
