package kr.penvmap.capture.core

import java.io.File
import java.security.MessageDigest
import java.util.Locale

/** Read published PC/phone results without changing captures or assuming absolute PC paths still exist. */
object ResultIndex {
    data class Verification(val checked: Int, val problems: List<String>, val unverified: Int) {
        val message: String get() = buildString {
            append("$checked final result files match the saved record.")
            if (unverified > 0) append("\n$unverified files could not be checked from this record.")
            if (problems.isNotEmpty()) append("\n\n" + problems.joinToString("\n"))
            if (checked == 0 && problems.isEmpty()) append("\nNo usable file checksums were recorded.")
            append("\n\nThis checks saved file contents. Use the pictures to review masks and alignment.")
        }
    }
    data class Picture(val file: File, val title: String, val caption: String)
    data class Result(
        val mode: String, val directory: File, val report: Map<String, Any?>,
        val preview: File, val modified: Long, val pictures: Map<String, List<Picture>>,
    ) {
        val width = (report["width"] as? Number)?.toInt()
        val height = (report["height"] as? Number)?.toInt()
        val label get() = if (mode == "full") "Full Stokes" else "Quick S0"
        val coverage get() = report["coverage"].resultMap()
        val missing get() = (coverage["remaining_missing_pixels"] as? Number)?.toLong()
        val warnings get() = (report["warnings"] as? List<*>)?.map { it.toString() }.orEmpty()
        val limitations get() = (report["limitations"] as? List<*>)?.map { it.toString() }.orEmpty()
        val summary: String get() = buildString {
            append(label)
            if (width != null && height != null) append(" · $width × $height")
            appendLine(if (report["engine"] == "android") " · This phone" else if (report["engine"] == "desktop") " · PC" else "")
            val count = missing
            val total = (width?.toLong() ?: 0L) * (height?.toLong() ?: 0L)
            appendLine(when {
                count == null -> "Coverage · not recorded"
                count == 0L -> "Coverage · no missing pixels"
                total > 0 && count in 0..total -> "Coverage · %.2f%% · %,d missing pixels".format(Locale.US, 100.0 * (total - count) / total, count)
                else -> "Coverage · $count missing pixels"
            })
            val radiometric = report["radiometric"].resultMap()
            val color = radiometric["WB_CCM"].resultMap().ifEmpty { report["WB_CCM"].resultMap() }
            val scope = color["scope"] ?: color["source_scope"]
            appendLine("WB/CCM · " + when (scope) { "scene" -> "Scene ColorChecker"; "global" -> "Global profile"; else -> "not recorded" })
            appendLine(component("Sun", report["sun_fill"].resultMap().ifEmpty { report["sun"].resultMap() }))
            appendLine(component("Floor", report["floor"].resultMap()))
            appendLine("Vignetting · " + when (radiometric["vignetting_applied"] ?: report["vignetting_applied"]) { true -> "On"; false -> "Off"; else -> "not recorded" })
            report["created_at"]?.let { appendLine("Built · $it") }
            if (warnings.isNotEmpty()) appendLine("${warnings.size} processing notes · review masks and alignment")
        }.trim()

        private fun component(name: String, value: Map<String, Any?>): String {
            if (value["applied"] == true) {
                val ids = (value["shot_ids"] as? List<*>)?.joinToString(", ") ?: value["shot_id"]?.toString()
                return "$name · Applied" + if (ids != null) " · shot $ids" else ""
            }
            return "$name · " + (value["reason"] ?: if (value["applied"] == false) "Not applied" else "not recorded")
        }
    }

    /** A preview alone is not a finished scientific result. Do not present partial exports as complete. */
    fun list(scene: File): List<Result> = listOf("s0", "full").mapNotNull { mode ->
        runCatching {
            val folder = inside(scene, if (mode == "s0") "result_s0" else "result") ?: return@runCatching null
            val required = listOf("run.json", "s0.png", "coverage_mask.png") +
                if (mode == "s0") listOf("s0.exr") else listOf("s2l2.exr", "thetaphi.exr", "duff.exr")
            if (required.any { inside(scene, folder.relativeTo(scene.canonicalFile).invariantSeparatorsPath + "/" + it)?.let { f -> !f.isFile || f.length() == 0L } != false }) return@runCatching null
            val record = File(folder, "run.json")
            val report = Json.parse(record.readText().removePrefix("\uFEFF")).resultMap()
            if (report.isEmpty() || (report["mode"] != null && report["mode"] != mode)) return@runCatching null
            val preview = File(folder, "s0.png")
            Result(mode, folder, report, preview, record.lastModified(), pictures(scene, folder, report))
        }.getOrNull()
    }.sortedByDescending { it.modified }

    /** Stream the selected run's final outputs; never hashes other runs or follows external paths. */
    fun verify(result: Result): Verification {
        val root = result.directory.parentFile.canonicalFile
        fun sameRun() = runCatching {
            Json.parse(File(result.directory, "run.json").readText().removePrefix("\uFEFF")).resultMap() == result.report
        }.getOrDefault(false)
        fun changed() = Verification(0, listOf("This result was rebuilt or its record changed. Reopen the scene and check again."), 0)
        if (!sameRun()) return changed()
        var checked = 0
        var unverified = 0
        val problems = mutableListOf<String>()
        val recorded = mutableSetOf<String>()
        for (value in (result.report["outputs"] as? List<*>).orEmpty()) {
            val item = value.resultMap()
            val relative = item["relative_path"] as? String
            val legacy = (item["path"] as? String)?.replace('\\', '/')
            // Old PC records have absolute paths. Only their final folder/name can be relocated.
            val path = when {
                relative != null -> inside(root, relative)
                legacy != null && legacy.substringBeforeLast('/').substringAfterLast('/') == result.directory.name ->
                    inside(root, result.directory.name + "/" + legacy.substringAfterLast('/'))
                else -> null
            }
            val expected = item["sha256"] as? String
            if (path == null || !path.toPath().startsWith(result.directory.canonicalFile.toPath()) ||
                expected == null || !expected.matches(Regex("[0-9a-fA-F]{64}"))) { unverified++; continue }
            recorded.add(path.name)
            if (!path.isFile) { problems.add("Missing: ${path.name}"); continue }
            try {
                val digest = MessageDigest.getInstance("SHA-256")
                path.inputStream().buffered().use { stream ->
                    val buffer = ByteArray(1024 * 1024)
                    while (true) { val n = stream.read(buffer); if (n < 0) break; digest.update(buffer, 0, n) }
                }
                val actual = digest.digest().joinToString("") { "%02x".format(it) }
                if (actual.equals(expected, ignoreCase = true)) checked++ else problems.add("Changed: ${path.name}")
            } catch (_: Exception) { problems.add("Could not read: ${path.name}") }
        }
        if (!sameRun()) return changed()
        val required = listOf("s0.png", "coverage_mask.png") +
            if (result.mode == "s0") listOf("s0.exr") else listOf("s2l2.exr", "thetaphi.exr", "duff.exr")
        unverified += required.count { it !in recorded }
        return Verification(checked, problems, unverified)
    }

    /** Only files under this scene are displayed, including after a PC scene is copied to a phone. */
    internal fun inside(scene: File, relative: String): File? = runCatching {
        if (relative.startsWith('/') || File(relative).isAbsolute || relative.contains('\\') || relative.split('/').contains("..") || relative.contains(':')) return@runCatching null
        val root = scene.canonicalFile
        File(root, relative).canonicalFile.takeIf { it.toPath().startsWith(root.toPath()) }
    }.getOrNull()

    private fun pictures(scene: File, folder: File, report: Map<String, Any?>): Map<String, List<Picture>> {
        val result = linkedMapOf<String, List<Picture>>()
        result["Panorama"] = listOf(Picture(File(folder, "s0.png"), "Final S0", "Display preview; original linear HDR values remain in the EXR."))
        result["Coverage"] = listOf(Picture(File(folder, "coverage_mask.png"), "Coverage", "White = observed / filled. Black = missing. Coverage does not verify alignment or mask accuracy."))
        val layout = report["storage_layout"].resultMap()
        val dirs = layout["diagnostics"].resultMap()
        val runId = layout["run_id"] as? String
        fun base(group: String): File? {
            val relative = dirs[group] as? String ?: if (!runId.isNullOrEmpty()) "$group/analysis/$runId" else "$group/analysis"
            return inside(scene, relative)?.takeIf { it.isDirectory }
        }
        fun add(label: String, directory: File?, subdir: String, caption: String, accept: (File) -> Boolean = { true }) {
            if (directory == null) return
            val dir = inside(scene, directory.relativeTo(scene.canonicalFile).invariantSeparatorsPath + "/" + subdir) ?: return
            val files = dir.listFiles().orEmpty().filter { f ->
                f.isFile && f.extension.lowercase() in listOf("png", "jpg", "jpeg") && accept(f) &&
                    inside(scene, f.relativeTo(scene.canonicalFile).invariantSeparatorsPath) != null
            }.sortedWith(compareBy<File> { it.nameWithoutExtension.toIntOrNull() ?: Int.MAX_VALUE }.thenBy { it.name })
            if (files.isNotEmpty()) result[label] = files.map { Picture(it, "$label · ${it.name}", caption) }
        }
        add("View masks", base("envmap"), "mask_overlay", "Red = excluded from this view. Check equipment edges and real ground before trusting the result.")
        add("Floor masks", base("floor"), "mask_overlay", "Red = excluded from the Floor shot. Ground used for replacement must remain visible.")
        add("Color", base("color"), "", "Captured / corrected / target patches. A good chart fit does not guarantee every scene color.") { it.name.contains("patch") || it.name.contains("roi") }
        add("Floor", base("floor"), "", "Floor diagnostics; alpha is blend weight, mask is the replacement region.") { it.name.contains("nadir") || it.name in listOf("floor_alpha.png", "floor_mask.png") }
        add("Sun", base("sun"), "", "sun_mask = replacement region. saturation_mask = remaining capture saturation, not display clipping.") { it.name in listOf("sun_mask.png", "saturation_mask.png") }
        return result
    }
}

@Suppress("UNCHECKED_CAST")
private fun Any?.resultMap(): Map<String, Any?> = this as? Map<String, Any?> ?: emptyMap()
