package kr.penvmap.capture.core

import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Scene layout. The HDR payload and sidecars keep the PC capture format; the groups use the
 * agreed envmap / floor / sun / color paths. PC processing creates the result files.
 *
 *     {base}/{experiment}/{YYYYMMDD}/{HHMMSS}/
 *         envmap/hdr/{i}.npy    (T, H, W) raw bracket stack, uint16 BayerRG16, sensor orientation
 *         envmap/hdr/{i}.json   exposure_times_s, rvec (3, 1), yaw/pitch/roll, camera, pixel_format, ...
 *         envmap/ldr/{i}.png    optional mid-exposure raw frame (16-bit PNG, Save LDR)
 *                              S0 / live panorama previews exist only in memory
 *         color/...            same hdr / ldr layout for scene color checker shots
 *         sun/...              same layout for sun shots (json adds the stage pose and the ND)
 *         floor/...            optional repositioned floor shots, separate from rotation views
 *         {group}/result/      PC output; envmap/result is the panorama BEFORE Floor/Sun
 *         envmap/mask/         PC per-view masks, {view index}.png (255 usable, 0 excluded)
 *         result/              PC final panorama including Floor/Sun
 *         meta/capture_meta.json
 */
class SessionDirs(val root: File, val calibrationKind: CalibrationKind? = null) {

    class Part(base: File) {
        val stack = File(base, "hdr")
        val ldrRaw = File(base, "ldr")
        /** Legacy file, only used when clearing a retake; new captures do not write it. */
        val stitchS0 = File(base, "preview.png")
        val result = File(base, "result")

        fun create() {
            for (d in listOf(stack, ldrRaw)) check(d.isDirectory || d.mkdirs()) { "cannot create $d" }
        }
    }

    val meta = File(root, "meta")
    val scene = Part(File(root, "envmap"))
    val colorChecker = Part(File(root, "color"))
    val sun = Part(File(root, "sun"))
    val floor = Part(File(root, "floor"))
    val viewMasks = File(root, "envmap/mask")
    val result = File(root, "result")
    /** Generic reference data: no color-checker, sun or envmap subfolder. */
    val calibration = calibrationKind?.let { Part(root) }

    fun create(): SessionDirs {
        check(meta.isDirectory || meta.mkdirs()) { "cannot create $meta" }
        if (calibration != null) calibration.create() else {
            for (part in listOf(scene, floor, sun, colorChecker)) {
                part.create()
                check(part.result.isDirectory || part.result.mkdirs()) { "cannot create ${part.result}" }
            }
            for (dir in listOf(viewMasks, result)) check(dir.isDirectory || dir.mkdirs()) { "cannot create $dir" }
        }
        return this
    }

    companion object {
        val RAW_STORAGE: Map<String, Any> = linkedMapOf(
            "version" to 3,
            "ldr_content" to "raw",
            "ldr_path" to "ldr/{i}.png",
            "ldr_frame" to "middle bracket exposure (zero-based T/2)",
            "ldr_optional" to true,
            "preview_saved" to false,
        )
        val SCENE_LAYOUT: Map<String, Any> = linkedMapOf(
            "groups" to linkedMapOf("envmap" to "envmap", "floor" to "floor", "sun" to "sun", "color" to "color"),
            "view_masks" to "envmap/mask",
            "group_result" to "{group}/result",
            "final_result" to "result",
            "envmap_result_scope" to "Panorama before Floor/Sun compositing",
        ) + RAW_STORAGE
        /** `YYYYMMDD/HHMMSS` of [time] (local time, like the PC's datetime.now()). */
        fun timestamp(time: Date = Date()): String = SimpleDateFormat("yyyyMMdd/HHmmss", Locale.US).format(time)

        fun prepare(base: File, experiment: String, timestamp: String = timestamp()): SessionDirs {
            val root = File(File(base, experiment), timestamp)
            check(!root.exists()) { "A scene already exists at $timestamp; try again in a moment" }
            return SessionDirs(root).create()
        }

        fun prepareCalibration(base: File, kind: CalibrationKind, timestamp: String = timestamp()): SessionDirs {
            val root = File(File(base, kind.experiment), timestamp)
            check(!root.exists()) { "A calibration session already exists at $timestamp; try again in a moment" }
            return SessionDirs(root, kind).create()
        }
    }
}

/** One captured view as written by [SessionWriter.saveView]. */
class SavedView(val index: Int, val s0: Mosaic.S0, val rvec: DoubleArray)

object SessionWriter {

    /**
     * `save_hdr_stack` + `save_ldr` of capture/session.py `capture_view`: the stack (as the camera sent
     * it), its json with the *measured* exposure times (chunk read-back, not the request), the
     * optional mid-exposure raw PNG. S0 is returned for display only, never saved.
     */
    fun saveView(
        part: SessionDirs.Part,
        index: Int,
        frames: List<ByteArray>,
        width: Int,
        height: Int,
        exposureUs: List<Double>,
        requestedUs: List<Long>,
        yawDeg: Double,
        pitchDeg: Double,
        mountRollDeg: Double,
        bracketInfo: Map<String, Any?>,
        camera: String = "flir",
        pixelFormat: String = "BayerRG16",
        saveRawPng: Boolean = true,
        /** More json fields (e.g. the sun shot's stage pose and ND). */
        extra: Map<String, Any?> = emptyMap(),
    ): SavedView {
        Npy.writeU16Stack(File(part.stack, "$index.npy"), frames, height, width)
        val rvec = Pose.priorRvec(yawDeg, pitchDeg, mountRollDeg)
        val meta = linkedMapOf<String, Any?>(
            "exposure_times_s" to exposureUs.map { it / 1e6 },
            "rvec" to rvec.map { listOf(it) },
            "yaw_deg" to yawDeg,
            "pitch_deg" to pitchDeg,
            "mount_roll_deg" to mountRollDeg,
            "camera" to camera,
            "pixel_format" to pixelFormat,
            "requested_exposure_us" to requestedUs.map { it.toDouble() },
            "bracket" to bracketInfo,
            "save_raw_png" to saveRawPng,
            "ldr" to linkedMapOf(
                "kind" to "raw", "path" to if (saveRawPng) "../ldr/$index.png" else null,
                "frame_index" to frames.size / 2, "dtype" to "uint16", "orientation" to "sensor",
                "preview_saved" to false,
            ),
        )
        meta.putAll(extra)
        File(part.stack, "$index.json").writeText(Json.write(meta), Charsets.UTF_8)

        val mid = frames[frames.size / 2]
        val s0 = Mosaic.s0(Exposure.u16le(mid), width, height, 65535.0)
        if (saveRawPng) {
            check(part.ldrRaw.isDirectory || part.ldrRaw.mkdirs()) { "cannot create ${part.ldrRaw}" }
            Png.writeGray16(File(part.ldrRaw, "$index.png"), mid, width, height)
        }
        return SavedView(index, s0, rvec)
    }

    /** `meta/capture_meta.json` (overwritten each time, like the PC's save_meta). */
    fun saveMeta(dirs: SessionDirs, meta: Map<String, Any?>): File {
        val f = File(dirs.meta, "capture_meta.json")
        val stored = meta + ("storage_layout" to if (dirs.calibrationKind == null) SessionDirs.SCENE_LAYOUT else SessionDirs.RAW_STORAGE)
        f.writeText(Json.write(stored), Charsets.UTF_8)
        return f
    }

    /** Next free index in a stack dir (continuing a session keeps earlier views). */
    fun nextIndex(part: SessionDirs.Part): Int =
        (part.stack.listFiles { f -> f.name.endsWith(".npy") }?.mapNotNull { it.name.removeSuffix(".npy").toIntOrNull() }?.maxOrNull() ?: -1) + 1
}
