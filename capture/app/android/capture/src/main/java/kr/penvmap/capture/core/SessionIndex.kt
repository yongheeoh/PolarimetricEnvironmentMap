package kr.penvmap.capture.core

import java.io.File

/**
 * The sessions under one storage root (`{root}/{experiment}/{YYYYMMDD}/{HHMMSS}/meta/capture_meta.json`),
 * newest first. Pure JVM so the session manager's listing is unit-testable.
 */
object SessionIndex {

    class Info(
        val root: File,
        /** `experiment/YYYYMMDD/HHMMSS`, the same relative path the PC uses under base_dir. */
        val relPath: String,
        val experiment: String,
        val date: String,
        val time: String,
        val views: Int,
        val colorCheckers: Int,
        val sunShots: Int,
        val bytes: Long,
        val finished: Boolean,
        val camera: String?,
        val annotations: Map<String, String>,
        /** Legacy preview PNG, or a RAW stack to render in memory for current captures. */
        val thumbnail: File?,
        val calibrationType: String? = null,
        val calibrationShots: Int = 0,
        val floorShots: Int = 0,
    ) {
        val displayDate: String get() = if (date.length == 8 && time.length == 6) {
            "${date.substring(0, 4)}-${date.substring(4, 6)}-${date.substring(6)} ${time.substring(0, 2)}:${time.substring(2, 4)}:${time.substring(4)}"
        } else {
            "$date $time"
        }
    }

    fun list(base: File): List<Info> {
        if (!base.isDirectory) return emptyList()
        // Calibration paths can be deeper than eight levels. Once a session is found, only
        // descend into its meta folder, not the raw/results trees.
        val metas = base.walkTopDown().onEnter {
            it == base || it.name == "meta" || !File(it.parentFile, "meta/capture_meta.json").isFile
        }
            .filter { it.name == "capture_meta.json" && it.parentFile?.name == "meta" }
            .toList()
        return metas.mapNotNull { read(base, it.parentFile!!.parentFile!!) }
            .sortedByDescending { "${it.date}/${it.time}" }
    }

    fun read(base: File, root: File): Info? {
        val metaFile = File(root, "meta/capture_meta.json")
        if (!metaFile.isFile) return null
        val m = runCatching { Json.parse(metaFile.readText()).asMap() }.getOrNull() ?: emptyMap()
        val rel = root.relativeTo(base).path.replace(File.separatorChar, '/')
        val parts = rel.split('/')
        val time = parts.getOrElse(parts.size - 1) { "" }
        val date = parts.getOrElse(parts.size - 2) { "" }
        val experiment = parts.dropLast(2).joinToString("/")
        val stitch = File(root, "ldr/stitch_s0.png")
        val ccS0 = File(root, "color_checker/ldr/s0/0.png")
        val calibrationType = (m["calibration"] as? Map<*, *>)?.get("type") as? String
        val isCalibration = m["capture_kind"] == "calibration" || calibrationType != null
        val compactPreview = File(root, "ldr").listFiles { f -> f.extension == "png" }
            ?.sortedBy { it.nameWithoutExtension.toIntOrNull() ?: Int.MAX_VALUE }?.firstOrNull()
        val candidates = listOfNotNull(File(root, "envmap/preview.png"), stitch,
            File(root, "color/ldr/0.png"), File(root, "wb_cc/ldr/0.png"), ccS0, File(root, "envmap/ldr/0.png"),
            File(root, "sun/ldr/0.png"), File(root, "nd/ldr/0.png"), File(root, "ldr/s0/0.png"),
            File(root, "sun/ldr/s0/0.png"), File(root, "floor/ldr/0.png"), compactPreview)
        val rawOnly = (m["storage_layout"] as? Map<*, *>)?.get("ldr_content") == "raw"
        val rawThumbnail = listOf("envmap/hdr", "color/hdr", "sun/hdr", "floor/hdr", "hdr",
            "wb_cc/hdr", "nd/hdr", "hdr/stack", "color_checker/hdr/stack", "sun/hdr/stack")
            .asSequence().mapNotNull { dir ->
                File(root, dir).listFiles { f -> f.extension == "npy" && f.nameWithoutExtension.toIntOrNull() != null }
                    ?.minByOrNull { it.nameWithoutExtension.toInt() }
            }.firstOrNull()
        @Suppress("UNCHECKED_CAST")
        val notes = (m["annotations"] as? Map<String, Any?>)?.mapValues { it.value.toString() } ?: emptyMap()
        return Info(
            root = root,
            relPath = rel,
            experiment = experiment,
            date = date,
            time = time,
            views = if (isCalibration) 0 else countNpy(File(root, "envmap/hdr")) + countNpy(File(root, "hdr/stack")),
            colorCheckers = countNpy(File(root, "color/hdr")) + countNpy(File(root, "wb_cc/hdr")) + countNpy(File(root, "color_checker/hdr/stack")),
            sunShots = countNpy(File(root, "sun/hdr")) + countNpy(File(root, "nd/hdr")) + countNpy(File(root, "sun/hdr/stack")),
            floorShots = countNpy(File(root, "floor/hdr")),
            bytes = root.walkTopDown().filter { it.isFile }.sumOf { it.length() },
            finished = (m["session_timing"] as? Map<*, *>)?.get("finished") == true,
            camera = m["camera"] as? String,
            annotations = notes,
            thumbnail = listOf(File(root, "result/s0.png"), File(root, "result_s0/s0.png")).filter { it.isFile }
                .maxByOrNull { it.lastModified() }
                ?: if (rawOnly) rawThumbnail else candidates.firstOrNull { it.isFile } ?: rawThumbnail,
            calibrationType = calibrationType,
            calibrationShots = if (isCalibration) countNpy(File(root, "hdr")) else 0,
        )
    }

    private fun countNpy(dir: File) = dir.listFiles { f -> f.name.endsWith(".npy") }?.size ?: 0

    fun humanBytes(b: Long): String = when {
        b >= 10_000_000_000L -> "%.0f GB".format(b / 1e9)
        b >= 1_000_000_000L -> "%.1f GB".format(b / 1e9)
        b >= 1_000_000L -> "%.0f MB".format(b / 1e6)
        b >= 1_000L -> "%.0f kB".format(b / 1e3)
        else -> "$b B"
    }
}
