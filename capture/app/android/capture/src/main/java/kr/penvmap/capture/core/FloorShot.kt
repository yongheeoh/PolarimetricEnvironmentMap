package kr.penvmap.capture.core

/** A repositioned nadir reference, never a view of the original pure-rotation panorama. */
data class FloorShot(val note: String = "") {
    fun metadata(sourceScene: String, pose: Pair<Double, Double>?, capturedAt: String): Map<String, Any?> = linkedMapOf(
        "purpose" to "nadir_patch",
        "source_scene" to sourceScene,
        "exposure_mode" to "bracket",
        "captured_at" to capturedAt,
        "stage_yaw_deg" to pose?.first,
        "stage_pitch_deg" to pose?.second,
        "stage_pose_known" to (pose != null),
        "pose_reference" to "stage_local_after_repositioning",
        "scene_pose_known" to false,
        "translation_m" to null,
        "include_in_rotation" to false,
        "nd_od" to null,
        "note" to note.trim().takeIf { it.isNotEmpty() },
    )
}
