package kr.penvmap.capture

import android.content.Context
import kr.penvmap.capture.core.Json
import kr.penvmap.capture.core.asList
import kr.penvmap.capture.core.asMap

/**
 * Capture settings, the phone's equivalent of capture/config/{base,flir}.yaml. Same names and
 * defaults as the PC FLIR preset so capture_meta.json reads the same.
 */
data class AppSettings(
    val experimentName: String = "FLIR/test",
    val cameraMode: String = CAMERA_FLIR, // flir | fake (basler: not supported yet)
    val stageMode: String = STAGE_HTTP, // http | fake
    val mechaUrl: String = "http://192.168.8.1",
    val stageSettleYawS: Double = 0.2,
    val stageSettlePitchS: Double = 0.2,
    val yawDegs: List<Double> = listOf(0.0, 60.0, 120.0, 180.0, 240.0, 300.0), // capture/config/base.yaml (from 0°, 2026-10-13)
    val pitchDegs: List<Double> = listOf(-45.0, 0.0, 45.0),
    val mountRollDeg: Double = 90.0,
    val bracketMinUs: Long = 21,
    val bracketMaxUs: Long = 344_064, // 21 us × 4^7: daylight, no ND, aperture ring at 11 (capture/config/flir.yaml, 2026-10-12)
    val numBrackets: Int = 8,
    /** true = camera factory default, pinned like the PC preset; null = leave the camera's value */
    val blackLevelClamping: Boolean? = true,
    val defectCorrectStatic: Boolean? = true,
    val captureDelayS: Double = 0.001,
    val stitchWidth: Int = 1024,
    val stitchHeight: Int = 512,
    val saveRawPng: Boolean = true,
    /** Preview frame-rate cap (FLIR AcquisitionFrameRate while previewing; captures are not limited). */
    val previewFps: Double = 4.0,
    /** Show the preview as the sensor reads it (the saved orientation) instead of sky up; PC preview_show_sensor. */
    val previewShowSensor: Boolean = false,
    /** Live view: [PREVIEW_RGB] (superpixel S0 colour, tone-mapped) or [PREVIEW_RAW] (raw values as grey, 255 = saturated). */
    val previewMode: String = PREVIEW_RGB,
    /** The base path: [Storage.DOCUMENTS] (the phone's Documents/penvmap) or [Storage.CUSTOM] ([customBasePath]). */
    val storageMode: String = Storage.DOCUMENTS,
    /** A folder picked as the base path (USB SSD, SD card, ...), absolute; kept when switching back to the phone. */
    val customBasePath: String? = null,
) {
    /**
     * What Settings > Reset to defaults gives: preview, camera and stage back to the defaults; the project
     * name, bracket and base path (set on the main screen) stay as they are.
     */
    fun withSettingsDefaults(): AppSettings {
        val d = AppSettings()
        return copy(
            saveRawPng = d.saveRawPng, previewShowSensor = d.previewShowSensor, previewFps = d.previewFps,
            previewMode = d.previewMode, cameraMode = d.cameraMode, blackLevelClamping = d.blackLevelClamping, defectCorrectStatic = d.defectCorrectStatic,
            stageMode = d.stageMode, mechaUrl = d.mechaUrl, yawDegs = d.yawDegs, pitchDegs = d.pitchDegs,
            stageSettleYawS = d.stageSettleYawS, stageSettlePitchS = d.stageSettlePitchS, mountRollDeg = d.mountRollDeg,
        )
    }

    /** The PC's `config` block of capture_meta.json, with the phone's values. */
    fun toConfigMap(baseDir: String, date: String): Map<String, Any?> = linkedMapOf(
        "base_dir" to baseDir,
        "experiment_name" to experimentName,
        "camera" to (if (cameraMode == CAMERA_FAKE) "fake" else "flir"),
        "pixel_format" to "BayerRG16",
        "camera_index" to 0,
        "gain" to 0,
        "raw_max" to null,
        "mount_roll_deg" to mountRollDeg,
        "preview_show_sensor" to previewShowSensor,
        "preview_fps" to previewFps,
        "bracket_min_us" to bracketMinUs,
        "bracket_max_us" to bracketMaxUs,
        "num_brackets" to numBrackets,
        "mask_dir" to null,
        "settle_s" to 0.2,
        "settle_frames" to 2,
        "capture_delay" to captureDelayS,
        "intrinsic_path" to INTRINSIC_ASSET,
        "stitch_size" to listOf(stitchWidth, stitchHeight),
        "flir_black_level_clamping" to blackLevelClamping,
        "flir_defect_correct_static" to defectCorrectStatic,
        "flir_bracket_mode" to "sequencer",
        "date" to date,
        "mecha_url" to mechaUrl,
        "stage_settle_yaw_s" to stageSettleYawS,
        "stage_settle_pitch_s" to stageSettlePitchS,
        "yaw_degs" to yawDegs,
        "pitch_degs" to pitchDegs,
        "stage_mode" to stageMode,
        "save_raw_png" to saveRawPng,
        "storage_mode" to storageMode,
    )

    fun toJson(): String = Json.write(
        linkedMapOf(
            "experimentName" to experimentName, "cameraMode" to cameraMode, "stageMode" to stageMode,
            "mechaUrl" to mechaUrl, "stageSettleYawS" to stageSettleYawS, "stageSettlePitchS" to stageSettlePitchS,
            "yawDegs" to yawDegs, "pitchDegs" to pitchDegs, "mountRollDeg" to mountRollDeg,
            "bracketMinUs" to bracketMinUs, "bracketMaxUs" to bracketMaxUs, "numBrackets" to numBrackets,
            "blackLevelClamping" to blackLevelClamping, "defectCorrectStatic" to defectCorrectStatic,
            "captureDelayS" to captureDelayS, "stitchWidth" to stitchWidth, "stitchHeight" to stitchHeight,
            "saveRawPng" to saveRawPng, "previewFps" to previewFps, "previewShowSensor" to previewShowSensor,
            "previewMode" to previewMode, "storageMode" to storageMode, "customBasePath" to customBasePath,
        ),
    )

    companion object {
        const val CAMERA_FLIR = "flir"
        const val CAMERA_FAKE = "fake"
        /** Basler full-Stokes camera: shown in Settings, disabled until the app can drive it. */
        const val CAMERA_BASLER = "basler"
        const val PREVIEW_RGB = "rgb"
        const val PREVIEW_RAW = "raw"
        const val STAGE_HTTP = "http"
        const val STAGE_FAKE = "fake"
        const val INTRINSIC_ASSET = "intrinsics/4mm_FLIR_20260316.json"
        private const val PREFS = "penvmap_capture"
        private const val KEY = "settings_json"

        fun load(context: Context): AppSettings {
            val text = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, null) ?: return AppSettings()
            return runCatching { fromJson(text) }.getOrDefault(AppSettings())
        }

        fun save(context: Context, s: AppSettings) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY, s.toJson()).apply()
        }

        fun fromJson(text: String): AppSettings {
            val m = Json.parse(text).asMap()
            val d = AppSettings()
            fun dbl(k: String, def: Double) = (m[k] as? Number)?.toDouble() ?: def
            fun lng(k: String, def: Long) = (m[k] as? Number)?.toLong() ?: def
            fun list(k: String, def: List<Double>) = (m[k] as? List<*>)?.asList()?.map { (it as Number).toDouble() } ?: def
            return AppSettings(
                experimentName = m["experimentName"] as? String ?: d.experimentName,
                cameraMode = m["cameraMode"] as? String ?: d.cameraMode,
                stageMode = m["stageMode"] as? String ?: d.stageMode,
                mechaUrl = m["mechaUrl"] as? String ?: d.mechaUrl,
                stageSettleYawS = dbl("stageSettleYawS", d.stageSettleYawS),
                stageSettlePitchS = dbl("stageSettlePitchS", d.stageSettlePitchS),
                yawDegs = list("yawDegs", d.yawDegs),
                pitchDegs = list("pitchDegs", d.pitchDegs),
                mountRollDeg = dbl("mountRollDeg", d.mountRollDeg),
                bracketMinUs = lng("bracketMinUs", d.bracketMinUs),
                bracketMaxUs = lng("bracketMaxUs", d.bracketMaxUs),
                numBrackets = lng("numBrackets", d.numBrackets.toLong()).toInt(),
                blackLevelClamping = if ("blackLevelClamping" in m) m["blackLevelClamping"] as? Boolean else d.blackLevelClamping,
                defectCorrectStatic = if ("defectCorrectStatic" in m) m["defectCorrectStatic"] as? Boolean else d.defectCorrectStatic,
                captureDelayS = dbl("captureDelayS", d.captureDelayS),
                stitchWidth = lng("stitchWidth", d.stitchWidth.toLong()).toInt(),
                stitchHeight = lng("stitchHeight", d.stitchHeight.toLong()).toInt(),
                saveRawPng = m["saveRawPng"] as? Boolean ?: d.saveRawPng,
                previewFps = dbl("previewFps", d.previewFps).coerceIn(0.5, 30.0),
                previewShowSensor = m["previewShowSensor"] as? Boolean ?: d.previewShowSensor,
                previewMode = (m["previewMode"] as? String)?.takeIf { it == PREVIEW_RGB || it == PREVIEW_RAW } ?: d.previewMode,
                customBasePath = (m["customBasePath"] as? String)?.let { Storage.appPath(it) },
                // The old app folder is only read now; a custom base path needs its folder.
                storageMode = if (m["storageMode"] == Storage.CUSTOM && m["customBasePath"] is String) Storage.CUSTOM else Storage.DOCUMENTS,
            )
        }
    }
}
