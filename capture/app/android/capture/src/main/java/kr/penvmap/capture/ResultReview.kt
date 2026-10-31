package kr.penvmap.capture

import android.app.AlertDialog
import android.app.Dialog
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import kr.penvmap.capture.core.ResultIndex
import java.io.File
import java.util.concurrent.ExecutorService

/** Published results remain inspectable even after another scene has been processed. */
class ResultReview(
    private val activity: MainActivity, private val scene: File,
    private val loader: ExecutorService, private val loadPreview: (File, ImageView) -> Unit,
) : LinearLayout(activity) {
    private val kit = Kit(activity)
    private var stamp = ""
    private var requested = ""
    private var selected: String? = null
    private var results = emptyList<ResultIndex.Result>()
    private var closed = false
    private var viewer: Dialog? = null

    init { orientation = VERTICAL }

    fun close() { closed = true; viewer?.dismiss() }

    fun refresh() {
        if (closed || loader.isShutdown) return
        val key = listOf("result", "result_s0").joinToString { name ->
            val dir = File(scene, name)
            listOf(dir, File(dir, "run.json"), File(dir, "s0.png")).joinToString { "${it.lastModified()}:${it.length()}" }
        }
        if (key == stamp || key == requested) return
        requested = key
        loader.execute {
            val found = ResultIndex.list(scene)
            activity.runOnUiThread {
                if (!closed && requested == key) {
                    stamp = key; requested = ""; results = found; render()
                }
            }
        }
    }

    private fun render() {
        removeAllViews()
        if (results.isEmpty()) return
        val result = results.firstOrNull { it.mode == selected } ?: results.first().also { selected = it.mode }
        addView(kit.text("Saved results", 16f, bold = true), kit.lp(t = 18, b = 8))
        val modes = kit.row()
        for (r in results.sortedBy { it.mode != "s0" }) modes.addView(
            kit.button(r.label, if (r.mode == result.mode) Kit.Style.PRIMARY else Kit.Style.SECONDARY, 13f) {
                selected = r.mode; render()
            }, kit.lp(0, kit.wrap(), 1f, r = 4))
        addView(modes)
        addView(kit.text(result.summary, 12.5f, Palette.TEXT2), kit.lp(t = 10, b = 8))
        // Reserve the ERP's aspect ratio before async decoding, so switching modes
        // does not shrink the ScrollView and jump away from the selected controls.
        val image = object : ImageView(activity) {
            override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
                val width = MeasureSpec.getSize(widthMeasureSpec)
                val height = if ((result.width ?: 0) > 0 && (result.height ?: 0) > 0)
                    (width.toLong() * result.height!! / result.width!!).toInt() else width / 2
                super.onMeasure(widthMeasureSpec, MeasureSpec.makeMeasureSpec(height, MeasureSpec.EXACTLY))
            }
        }.apply {
            adjustViewBounds = true
            setBackgroundColor(android.graphics.Color.BLACK)
            contentDescription = "${result.label} panorama. Tap to zoom."
            setOnClickListener { showPictures(result.pictures.getValue("Panorama")) }
        }
        kit.clipRounded(image, 10f); addView(image); loadPreview(result.preview, image)
        addView(kit.text("Tap a picture to zoom · pinch / drag", 12f, Palette.TEXT3), kit.lp(t = 6, b = 6))
        var row = kit.row()
        val groups = result.pictures.filterKeys { it != "Panorama" }.entries.toList()
        for ((i, entry) in groups.withIndex()) {
            row.addView(kit.button(entry.key, Kit.Style.SECONDARY, 12.5f) { showPictures(entry.value) }, kit.lp(0, kit.wrap(), 1f, r = 4))
            if (i % 2 == 1 || i == groups.lastIndex) { addView(row, kit.lp(b = 6)); row = kit.row() }
        }
        if (result.warnings.isNotEmpty() || result.limitations.isNotEmpty()) addView(
            kit.button("Processing notes", Kit.Style.GHOST, 13f) {
                val text = (result.warnings + result.limitations).distinct().joinToString("\n\n")
                AlertDialog.Builder(activity).setTitle(result.label + " · notes")
                    .setView(ScrollView(activity).apply { addView(kit.text(text, 13f).apply { setPadding(kit.dp(18), kit.dp(8), kit.dp(18), kit.dp(8)) }) })
                    .setPositiveButton("Close", null).show()
            })
        val verify = kit.button("Check saved files", Kit.Style.GHOST, 13f) {}
        verify.setOnClickListener {
            if (!loader.isShutdown) {
                kit.enable(verify, false); verify.text = "Checking saved files…"
                loader.execute {
                    val checked = runCatching { ResultIndex.verify(result).message }.getOrElse { "Could not check files: ${it.message}" }
                    activity.runOnUiThread {
                        if (!closed) {
                            verify.text = "Check saved files"; kit.enable(verify, true)
                            AlertDialog.Builder(activity).setTitle(result.label + " · saved files")
                                .setMessage(checked).setPositiveButton("Close", null).show()
                        }
                    }
                }
            }
        }
        addView(verify)
    }

    private fun showPictures(pictures: List<ResultIndex.Picture>) {
        if (pictures.isEmpty() || closed) return
        viewer?.dismiss()
        val dialog = Dialog(activity, android.R.style.Theme_Material_NoActionBar)
        viewer = dialog
        val root = kit.col().apply { setBackgroundColor(Palette.BG) }
        kit.padForSystemBars(root)
        val body = kit.col().apply { setPadding(kit.dp(12), 0, kit.dp(12), 0) }
        root.addView(body, kit.lp(kit.match(), kit.match()))
        val title = kit.text("", 15f, bold = true)
        val caption = kit.text("", 12f, Palette.TEXT2)
        val image = ZoomPicture(activity)
        var index = 0
        var generation = 0
        val bar = kit.row()
        val previous = kit.button("Previous", Kit.Style.SECONDARY, 13f) {}
        val next = kit.button("Next", Kit.Style.SECONDARY, 13f) {}
        fun show() {
            val item = pictures[index]; val token = ++generation
            title.text = "${index + 1}/${pictures.size} · ${item.title}"
            caption.text = item.caption + "\nPinch to zoom · drag to pan · double-tap to reset"
            image.setImage(null)
            kit.enable(previous, index > 0); kit.enable(next, index < pictures.lastIndex)
            if (loader.isShutdown) return
            loader.execute {
                // Diagnostics and ERP previews are small; cap unexpected legacy images at ~4096px.
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(item.file.absolutePath, bounds)
                var sample = 1
                while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 4096) sample *= 2
                val bitmap = runCatching { BitmapFactory.decodeFile(item.file.absolutePath, BitmapFactory.Options().apply { inSampleSize = sample }) }.getOrNull()
                activity.runOnUiThread {
                    if (dialog.isShowing && generation == token && !closed) {
                        image.setImage(bitmap)
                        if (bitmap == null) caption.text = "Could not read ${item.file.name}"
                    }
                }
            }
        }
        previous.setOnClickListener { if (index > 0) { index--; show() } }
        next.setOnClickListener { if (index < pictures.lastIndex) { index++; show() } }
        body.addView(title, kit.lp(t = 10, b = 8)); body.addView(caption, kit.lp(b = 8))
        body.addView(image, kit.lp(kit.match(), 0, 1f))
        bar.addView(previous, kit.lp(0, kit.wrap(), 1f, r = 4))
        bar.addView(kit.button("Reset", Kit.Style.GHOST, 13f) { image.reset() }, kit.lp(0, kit.wrap(), 1f))
        bar.addView(next, kit.lp(0, kit.wrap(), 1f, l = 4))
        body.addView(bar, kit.lp(t = 8))
        val footer = kit.row()
        footer.addView(kit.button("Rotate", Kit.Style.GHOST, 14f) { image.rotate() }, kit.lp(0, kit.wrap(), 1f))
        footer.addView(kit.button("Close", Kit.Style.GHOST, 14f) { dialog.dismiss() }, kit.lp(0, kit.wrap(), 1f))
        body.addView(footer, kit.lp(b = 8))
        dialog.setContentView(root)
        dialog.setOnDismissListener { generation++; image.setImage(null); if (viewer === dialog) viewer = null }
        dialog.show(); dialog.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        show()
    }
}

/** Read-only viewer: never changes exposure or the scientific files. */
private class ZoomPicture(context: android.content.Context) : View(context) {
    private var bitmap: Bitmap? = null
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private var zoom = 1f
    private var turns = 0
    private var x = 0f
    private var y = 0f
    private fun imageWidth() = bitmap?.let { if (turns % 2 == 0) it.width else it.height } ?: 1
    private fun imageHeight() = bitmap?.let { if (turns % 2 == 0) it.height else it.width } ?: 1
    private fun fit(): Float = minOf(width.toFloat() / imageWidth(), height.toFloat() / imageHeight())
    private val scale = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            val next = (zoom * detector.scaleFactor).coerceIn(1f, 16f)
            val ratio = next / zoom
            x = detector.focusX - (detector.focusX - x) * ratio
            y = detector.focusY - (detector.focusY - y) * ratio
            zoom = next; constrain(); invalidate(); return true
        }
    })
    private val gesture = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDown(e: MotionEvent) = true
        override fun onDoubleTap(e: MotionEvent): Boolean { reset(); return true }
        override fun onScroll(e1: MotionEvent?, e2: MotionEvent, dx: Float, dy: Float): Boolean {
            if (!scale.isInProgress) { x -= dx; y -= dy; constrain(); invalidate() }
            return true
        }
    })
    init { setBackgroundColor(android.graphics.Color.BLACK); contentDescription = "Zoomable result image" }
    fun setImage(value: Bitmap?) { bitmap = value; reset() }
    fun rotate() { turns = (turns + 1) % 4; reset() }
    fun reset() { zoom = 1f; constrain(); invalidate() }
    private fun constrain() {
        bitmap?.let {
            val w = imageWidth() * fit() * zoom; val h = imageHeight() * fit() * zoom
            x = if (w <= width) (width - w) / 2 else x.coerceIn(width - w, 0f)
            y = if (h <= height) (height - h) / 2 else y.coerceIn(height - h, 0f)
        }
    }
    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) { reset() }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        bitmap?.let {
            canvas.save(); canvas.translate(x, y); canvas.scale(fit() * zoom, fit() * zoom)
            when (turns) { 1 -> canvas.translate(it.height.toFloat(), 0f); 2 -> canvas.translate(it.width.toFloat(), it.height.toFloat()); 3 -> canvas.translate(0f, it.width.toFloat()) }
            canvas.rotate(turns * 90f); canvas.drawBitmap(it, 0f, 0f, paint); canvas.restore()
        }
    }
    override fun onTouchEvent(event: MotionEvent): Boolean {
        scale.onTouchEvent(event); gesture.onTouchEvent(event)
        if (event.actionMasked == MotionEvent.ACTION_UP) performClick()
        return true
    }
    override fun performClick(): Boolean { super.performClick(); return true }
}
