package kr.penvmap.capture

import android.app.AlertDialog
import android.app.Dialog
import android.graphics.BitmapFactory
import android.graphics.Bitmap
import android.util.LruCache
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import kr.penvmap.capture.CaptureController.Level
import kr.penvmap.capture.core.SessionIndex
import kr.penvmap.capture.core.CalibrationKind
import kr.penvmap.capture.core.Npy
import kr.penvmap.capture.core.Mosaic
import kr.penvmap.capture.core.Exposure
import java.util.concurrent.Executors

/**
 * The gallery: the scenes under one base path -- the one scenes are saved to, or any other picked with the
 * folder button (viewing only; the capture's base path does not change) -- grouped by project and date,
 * newest first, with panorama, counts, size and whether it was finished. Open / Delete per scene.
 * To the PC: app/tools/pull_sessions.py (phone), or the SSD itself.
 */
class SessionsScreen(private val activity: MainActivity, private val controller: CaptureController) {
    private val kit = Kit(activity)
    private lateinit var dialog: Dialog
    private lateinit var list: LinearLayout
    private lateinit var folderButton: TextView
    /** The folder shown; starts at the capture's base path. */
    private var viewDir: java.io.File = controller.baseDir
    private val thumbnails = Executors.newSingleThreadExecutor()
    private val thumbnailCache = LruCache<String, Bitmap>(8)
    private var refreshGeneration = 0
    private var currentDetail: AlertDialog? = null

    fun show() {
        dialog = Dialog(activity, android.R.style.Theme_Material_NoActionBar)
        dialog.setOnDismissListener { currentDetail?.dismiss(); thumbnails.shutdownNow(); thumbnailCache.evictAll() }
        val root = kit.col().apply { setBackgroundColor(Palette.BG) }
        kit.padForSystemBars(root)
        val bar = kit.row().apply { setPadding(kit.dp(8), kit.dp(6), kit.dp(14), kit.dp(6)) }
        bar.addView(kit.button("Close", Kit.Style.GHOST, 14.5f) { dialog.dismiss() })
        bar.addView(kit.text("Gallery", 18f, bold = true).apply { gravity = Gravity.CENTER }, kit.lp(0, kit.wrap(), 1f))
        folderButton = kit.button("📁 ▾", Kit.Style.SECONDARY, 13f) {
            activity.pickBasePath("Gallery folder", viewDir, "Show", forCapture = false) { dir ->
                viewDir = dir
                refresh()
            }
        }.apply {
            minHeight = kit.dp(38)
            maxLines = 1
            setPadding(kit.dp(12), kit.dp(6), kit.dp(12), kit.dp(6))
        }
        bar.addView(folderButton)
        root.addView(bar)
        val body = kit.col().apply { setPadding(kit.dp(14), kit.dp(4), kit.dp(14), kit.dp(24)) }
        list = kit.col()
        body.addView(list)
        root.addView(ScrollView(activity).apply { addView(body) }, kit.lp(kit.match(), 0, 1f))
        dialog.setContentView(root)
        dialog.window?.apply {
            setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            statusBarColor = Palette.BG
            navigationBarColor = Palette.BG
        }
        refresh()
        dialog.show()
    }

    /** The scenes under [viewDir] with its kind (phone / drive), newest first. */
    private fun all(): List<Pair<String, SessionIndex.Info>> {
        val mode = if (Storage.isPhone(viewDir)) Storage.DOCUMENTS else Storage.CUSTOM
        return SessionIndex.list(viewDir).map { mode to it }.sortedByDescending { "${it.second.date}/${it.second.time}" }
    }

    private fun refresh() {
        list.removeAllViews()
        folderButton.text = (if (Storage.isPhone(viewDir)) "📁 " else "💾 ") + shortName(viewDir) + " ▾"
        if (!Storage.available(activity, viewDir)) {
            list.addView(kit.text("${Storage.readable(activity, viewDir.absolutePath)}\nnot connected", 14f, Palette.RED)
                .apply { gravity = Gravity.CENTER }, kit.lp(t = 30))
            return
        }
        val generation = ++refreshGeneration
        list.addView(kit.text("Loading scenes…", 14f, Palette.TEXT2), kit.lp(t = 20))
        thumbnails.execute {
            val scenes = runCatching { all() }.getOrDefault(emptyList())
            activity.runOnUiThread {
                if (dialog.isShowing && generation == refreshGeneration) showScenes(scenes)
            }
        }
    }

    private fun showScenes(scenes: List<Pair<String, SessionIndex.Info>>) {
        list.removeAllViews()
        if (scenes.isEmpty()) {
            list.addView(kit.text("No scenes in ${Storage.readable(activity, viewDir.absolutePath)}", 14f, Palette.TEXT3)
                .apply { gravity = Gravity.CENTER }, kit.lp(t = 30))
            return
        }
        // project, then date: a heading whenever either changes (the list is newest first)
        var group = ""
        for ((mode, info) in scenes) {
            val g = "${info.experiment.ifEmpty { "(no project)" }}  ·  ${info.displayDate.substringBefore(' ')}"
            if (g != group) {
                group = g
                list.addView(kit.label(g), kit.lp(t = if (list.childCount == 0) 4 else 14, b = 8, l = 4))
            }
            list.addView(card(mode, info), kit.lp(b = 12))
        }
    }

    /** "Phone/Documents/penvmap" -> "penvmap", "Samsung USB/penvmap/data/dataset_FLIR" -> "dataset_FLIR". */
    private fun shortName(dir: java.io.File): String = when {
        dir.absolutePath == Storage.baseDir(activity, Storage.DOCUMENTS).absolutePath -> "Phone"
        else -> dir.name
    }

    private fun card(mode: String, e: SessionIndex.Info): View {
        val c = kit.card(12)
        val open = e.root.absolutePath == controller.sessionPath
        e.thumbnail?.let { f ->
                val img = ImageView(activity).apply {
                    adjustViewBounds = true
                    scaleType = ImageView.ScaleType.FIT_CENTER
                    setBackgroundColor(0xFF000000.toInt())
                    setOnClickListener { detail(mode, e) }
                }
                kit.clipRounded(img, 12f)
                c.addView(img, kit.lp(b = 10))
                loadThumbnail(f, img)
        }
        val title = kit.row()
        // project and date are in the group heading above
        title.addView(kit.text(e.displayDate.substringAfter(' '), 15.5f, bold = true), kit.lp(0, kit.wrap(), 1f))
        title.addView(kit.text(SessionIndex.humanBytes(e.bytes), 13f, Palette.TEXT2))
        c.addView(title)
        c.addView(kit.text(captureSummary(e), 13f, Palette.TEXT2), kit.lp(t = 4))
        val chips = kit.row()
        val chipList = mutableListOf<TextView>()
        chipList += when {
            open -> kit.chip("open", Palette.BLUE)
            e.finished -> kit.chip("finished", Palette.GREEN)
            else -> kit.chip("incomplete", Palette.AMBER)
        }
        if (mode == Storage.CUSTOM) chipList += kit.chip(Storage.volumeName(activity, e.root), Palette.BLUE)
        if (e.camera == "fake") chipList += kit.chip("simulated", null)
        for (ch in chipList) chips.addView(ch, kit.lp(kit.wrap(), kit.wrap(), r = 6))
        c.addView(android.widget.HorizontalScrollView(activity).apply {
            isHorizontalScrollBarEnabled = false
            addView(chips)
        }, kit.lp(t = 8))
        val actions = kit.row()
        actions.addView(kit.button("Open", Kit.Style.GHOST, 14f) { detail(mode, e) }, kit.lp(0, kit.wrap(), 1f))
        val delete = kit.button("Delete", Kit.Style.GHOST, 14f) { confirmDelete(e) }.apply { setTextColor(Palette.RED) }
        if (open || ProcessingJobs.owns(activity, e.root)) kit.enable(delete, false)
        actions.addView(delete, kit.lp(0, kit.wrap(), 1f))
        c.addView(actions, kit.lp(t = 4))
        return c
    }

    private fun detail(mode: String, e: SessionIndex.Info) {
        val box = kit.col().apply { setPadding(kit.dp(16), kit.dp(8), kit.dp(16), kit.dp(8)) }
        e.thumbnail?.takeUnless { it.parentFile?.name in listOf("result", "result_s0") }?.let { f ->
            val img = ImageView(activity).apply {
                adjustViewBounds = true
            }
            kit.clipRounded(img, 10f)
            box.addView(img, kit.lp(b = 12))
            loadThumbnail(f, img)
        }
        val info = buildString {
            appendLine(e.relPath)
            appendLine("${e.displayDate} · ${captureSummary(e)} · ${SessionIndex.humanBytes(e.bytes)}")
            appendLine("${if (e.finished) "finished" else "not finished"} · camera ${e.camera ?: "?"} · ${Storage.label(activity, mode, e.root)}")
            if (e.annotations.isNotEmpty()) appendLine("Notes: " + e.annotations.entries.joinToString(" · ") { "${it.key}: ${it.value}" })
            append(e.root.absolutePath)
        }
        box.addView(kit.text("${e.displayDate}\n${captureSummary(e)}", 14f, bold = true), kit.lp(b = 4))
        val captureDetails = kit.text(info, 12.5f, Palette.TEXT2).apply { visibility = View.GONE }
        box.addView(kit.button("Capture details", Kit.Style.GHOST, 12.5f) {
            captureDetails.visibility = if (captureDetails.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        })
        box.addView(captureDetails)
        val review = ResultReview(activity, e.root, thumbnails, ::loadThumbnail)
        val status = kit.text("", 13f, Palette.TEXT2)
        val progress = android.widget.ProgressBar(activity, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
        val supportedCamera = e.camera?.lowercase() in listOf(null, "flir", "fake")
        fun start(mode: String) {
            if (!supportedCamera) {
                activity.message("Offline processing supports FLIR captures. PC results can still be viewed.", Level.WARN); return
            }
            if (controller.sessionPath == e.root.absolutePath || controller.busyWith != null) {
                activity.message("Finish the current capture before processing", Level.WARN); return
            }
            runCatching { ProcessingJobs.start(activity, e.root, mode) }
                .onFailure { activity.message(it.message ?: "Could not start", Level.ERROR) }
        }
        val quick = kit.button("Build S0 · 1024 × 512", Kit.Style.PRIMARY, 14f) { start("s0") }
        val full = kit.button("Build full Stokes · 2048 × 1024", Kit.Style.SECONDARY, 14f) { start("full") }
        val cancel = kit.button("Cancel processing", Kit.Style.GHOST, 13f) { ProcessingJobs.cancel(activity) }
        if (e.calibrationType == null && e.views > 0) {
            box.addView(kit.text("Process on this phone", 16f, bold = true), kit.lp(t = 18, b = 8))
            box.addView(kit.text("Offline processing with WB/CCM, multiband and available Sun/Floor captures. S0 is a quick preview; full mode includes polarization.", 12.5f, Palette.TEXT2), kit.lp(b = 10))
            box.addView(status); box.addView(progress); box.addView(cancel)
            box.addView(quick, kit.lp(t = 8, b = 8)); box.addView(full, kit.lp(b = 8))
            box.addView(kit.text("Quick and full results are kept separately. Earlier results are archived when rebuilt.", 12f, Palette.TEXT3), kit.lp(t = 8, b = 8))
        }
        box.addView(review)
        val detailDialog = AlertDialog.Builder(activity, android.R.style.Theme_Material_Dialog_Alert)
            .setView(ScrollView(activity).apply { addView(box) })
            .setPositiveButton("Close", null)
            .show()
        currentDetail = detailDialog
        val handler = android.os.Handler(android.os.Looper.getMainLooper())
        val poll = object : Runnable {
            override fun run() {
                if (!detailDialog.isShowing) return
                val s = ProcessingJobs.status(activity)
                val active = ProcessingJobs.active(activity, s)
                val own = s["scene"] == e.root.absolutePath
                val allowed = supportedCamera && e.finished && controller.sessionPath != e.root.absolutePath && controller.busyWith == null && !active
                kit.enable(quick, allowed); kit.enable(full, allowed)
                progress.visibility = if (active && own) View.VISIBLE else View.GONE
                progress.progress = (s["progress"] as? Number)?.toInt() ?: 0
                cancel.visibility = if (active && own) View.VISIBLE else View.GONE
                val stopping = active && own && ProcessingJobs.cancellationRequested(activity)
                cancel.text = if (stopping) "Stopping after current operation…" else "Cancel processing"
                kit.enable(cancel, !stopping)
                status.text = when {
                    !supportedCamera -> "This camera needs PC processing. Saved results can still be viewed here."
                    active && !own -> "Another scene is processing."
                    own && active -> {
                        val started = (s["started"] as? Number)?.toLong() ?: System.currentTimeMillis()
                        val elapsed = maxOf(0L, (System.currentTimeMillis() - started) / 1000)
                        val elapsedText = "%d:%02d".format(elapsed / 60, elapsed % 60)
                        "${s["progress"] ?: 0}% · ${s["stage"]} · $elapsedText"
                    }
                    own && s["state"] in listOf("queued", "running", "cancelling") -> "Processing was interrupted. Previous results are preserved; run again."
                    own -> "${s["stage"] ?: ""}"
                    !e.finished -> "Finish this scene before processing."
                    else -> "Ready"
                }
                review.refresh()
                handler.postDelayed(this, 1000)
            }
        }
        detailDialog.setOnDismissListener {
            handler.removeCallbacks(poll); review.close()
            if (currentDetail === detailDialog) currentDetail = null
            if (dialog.isShowing) refresh()
        }
        handler.post(poll)
    }

    /** New sessions have no saved preview; render only the middle RAW exposure, off the UI thread. */
    private fun loadThumbnail(file: java.io.File, image: ImageView) {
        val key = "${file.absolutePath}:${file.lastModified()}:${file.length()}"
        image.tag = key
        thumbnailCache.get(key)?.let { image.setImageBitmap(it); return }
        if (thumbnails.isShutdown) return
        thumbnails.execute {
            if (Thread.currentThread().isInterrupted) return@execute
            val bmp = runCatching {
                if (file.extension == "npy") {
                    val raw = Npy.readMiddleU16(file)
                    val s0 = Mosaic.s0(Exposure.u16le(raw.bytes), raw.width, raw.height, 65535.0)
                    CaptureController.bgrToBitmap(Mosaic.toU8(s0.bgr), s0.width, s0.height)
                } else BitmapFactory.decodeFile(file.absolutePath, BitmapFactory.Options().apply { inSampleSize = 2 })
            }.getOrNull()
            if (bmp != null && !Thread.currentThread().isInterrupted) {
                thumbnailCache.put(key, bmp)
                activity.runOnUiThread { if (dialog.isShowing && image.tag == key) image.setImageBitmap(bmp) }
            }
        }
    }

    private fun captureSummary(e: SessionIndex.Info): String = if (e.calibrationType != null) {
        "${CalibrationKind.fromId(e.calibrationType)?.label ?: e.calibrationType} · ${e.calibrationShots} captures"
    } else {
        "${e.views} views · ${e.colorCheckers} color checker" +
            (if (e.sunShots > 0) " · ${e.sunShots} sun" else "") +
            (if (e.floorShots > 0) " · ${e.floorShots} floor" else "")
    }

    private fun confirmDelete(e: SessionIndex.Info) {
        val warning = "Take it to the PC first (app/tools/pull_sessions.py, or the SSD)."
        AlertDialog.Builder(activity, android.R.style.Theme_Material_Dialog_Alert)
            .setTitle("Delete this scene?")
            .setMessage("${e.relPath}\n${SessionIndex.humanBytes(e.bytes)}\n\n$warning This cannot be undone.")
            .setPositiveButton("Delete") { _, _ ->
                if (ProcessingJobs.owns(activity, e.root) || e.root.absolutePath == controller.sessionPath) {
                    activity.message("This scene is in use", Level.WARN); return@setPositiveButton
                }
                e.root.deleteRecursively()
                // empty date / experiment folders are left behind; harmless and kept for the PC's layout
                activity.message("Deleted ${e.relPath}", Level.OK)
                controller.pushState()
                refresh()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}
