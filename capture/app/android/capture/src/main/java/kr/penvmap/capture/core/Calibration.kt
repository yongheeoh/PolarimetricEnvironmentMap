package kr.penvmap.capture.core

/** The purpose of reference captures, independent of how the camera acquires a frame/bracket. */
enum class CalibrationKind(val id: String, val label: String, val directory: String) {
    VIGNETTING("vignetting", "Vignetting", "photometric/vignetting"),
    INTRINSIC("intrinsic", "Intrinsic", "geometric/intrinsic"),
    HDR("hdr", "HDR / sensor", "photometric/hdr"),
    POLARIMETRIC("polarimetric", "Polarimetric", "polarimetric"),
    ND("nd", "ND transmission", "radiometric/nd"),
    WB_CC("wb_cc", "Color checker", "radiometric/wb_cc"),
    MASK("mask", "Mask", "photometric/mask"),
    NODAL("nodal", "Nodal", "geometric/nodal");

    val experiment: String get() = "FLIR/calibration/$directory/data"

    companion object {
        fun fromId(id: String?): CalibrationKind? = entries.firstOrNull { it.id == id }
    }
}

/** Snapshot taken when Capture is pressed; UI edits cannot relabel an in-flight shot. */
data class CalibrationShot(
    val kind: CalibrationKind,
    val note: String = "",
    val ndOd: Double? = null,
) {
    fun metadata(stagePoseKnown: Boolean): Map<String, Any?> = linkedMapOf<String, Any?>(
        "type" to kind.id,
        "scope" to "global",
        "exposure_mode" to "bracket",
        "note" to note.trim().takeIf { it.isNotEmpty() },
        "stage_pose_known" to stagePoseKnown,
    ).also { if (kind == CalibrationKind.ND) it["nd_od"] = ndOd }
}
