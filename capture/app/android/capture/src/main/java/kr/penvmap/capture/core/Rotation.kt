package kr.penvmap.capture.core

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The scene rotation (capture/capture.py `run_rotation`): pitch outer loop, yaw inner loop, view index
 * from 0 in that order; home (0, 0) before and after; per-view timing recorded like the PC.
 */
class RotationRunner(
    private val stage: Stage,
    private val yaws: List<Double>,
    private val pitches: List<Double>,
    private val captureDelayS: Double = 0.001,
) {
    class ViewTiming(val idx: Int, val yawDeg: Double, val pitchDeg: Double, val moveS: Double, val captureS: Double, val envmapS: Double) {
        fun toMap(): Map<String, Any?> = linkedMapOf(
            "idx" to idx, "yaw_deg" to yawDeg, "pitch_deg" to pitchDeg,
            "move_s" to round3(moveS), "capture_s" to round3(captureS), "envmap_s" to round3(envmapS),
        )
    }

    interface Callbacks {
        /** Capture and save view [idx]; returns when the files are written. */
        fun capture(idx: Int, yawDeg: Double, pitchDeg: Double)
        /** Update the live envmap with view [idx] (after [capture]). */
        fun stitch(idx: Int) {}
        fun progress(done: Int, total: Int, yawDeg: Double, pitchDeg: Double) {}
        /** Checked between views; true stops the rotation (the stage still goes home). */
        fun cancelled(): Boolean = false
    }

    val totalViews: Int get() = yaws.size * pitches.size

    /** (yaw, pitch) of every view in capture order: pitch outer loop, yaw inner loop. */
    val plan: List<Pair<Double, Double>> get() = pitches.flatMap { p -> yaws.map { y -> y to p } }

    /**
     * Runs the rotation; returns (views, rotation_timing) for capture_meta.json. Views before [skipBefore]
     * are not visited (resuming a rotation that stopped there), and their indices are kept.
     */
    fun run(cb: Callbacks, startIndex: Int = 0, skipBefore: Int = 0): Pair<List<Map<String, Any?>>, Map<String, Any?>> {
        val views = ArrayList<ViewTiming>()
        val startedAt = isoNow()
        val t0 = System.nanoTime()
        var cancelled = false
        stage.moveAbs(0.0, 0.0, large = true)
        try {
            var currentPitch: Double? = null
            var tMove = System.nanoTime()
            for ((i, view) in plan.withIndex()) {
                if (i < skipBefore) continue
                val (yaw, pitch) = view
                if (cb.cancelled()) {
                    cancelled = true
                    break
                }
                if (pitch != currentPitch) {
                    tMove = System.nanoTime()
                    stage.moveAbs(null, pitch, large = true)
                    currentPitch = pitch
                }
                stage.moveAbs(yaw, null, large = false)
                val idx = startIndex + i
                val tCapture = System.nanoTime()
                cb.capture(idx, yaw, pitch)
                val tDone = System.nanoTime()
                cb.stitch(idx)
                val tEnv = System.nanoTime()
                views += ViewTiming(idx, yaw, pitch, secs(tCapture - tMove), secs(tDone - tCapture), secs(tEnv - tDone))
                cb.progress(i + 1, totalViews, yaw, pitch)
                Thread.sleep((captureDelayS * 1000).toLong())
                tMove = System.nanoTime()
            }
        } finally {
            stage.moveAbs(0.0, 0.0, large = true)
        }
        val duration = secs(System.nanoTime() - t0)
        val timing = linkedMapOf<String, Any?>(
            "rotation_started_at" to startedAt,
            "rotation_finished_at" to isoNow(),
            "rotation_duration_s" to round1(duration),
            "total_move_s" to round1(views.sumOf { it.moveS }),
            "total_capture_s" to round1(views.sumOf { it.captureS }),
            "total_envmap_s" to round1(views.sumOf { it.envmapS }),
        )
        if (cancelled) timing["cancelled_after_views"] = views.size
        return views.map { it.toMap() } to timing
    }

    companion object {
        fun isoNow(): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date())
        private fun secs(ns: Long) = ns / 1e9
        fun round1(x: Double) = Math.round(x * 10) / 10.0
        fun round3(x: Double) = Math.round(x * 1000) / 1000.0
    }
}
