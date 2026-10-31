package kr.penvmap.capture

import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import kr.penvmap.capture.MainActivity.Companion.fmtExposure
import kr.penvmap.capture.core.NodalViewport
import java.util.Locale

/**
 * Manual, local near/far comparison, using the existing preview stream. Nothing is written to disk or
 * applied to capture data. Reference + alignment survive closing the window, but not leaving Nodal.
 * All methods run on the UI thread; preview bitmaps are newly allocated and never mutated by capture.
 */
class NodalCompare(private val context: Context, private val onJog: (Double, Double) -> Unit) {
    private val kit = Kit(context)
    private val handler = Handler(Looper.getMainLooper())
    private val viewport = NodalViewport()
    private data class Frame(val bitmap: Bitmap, val info: CaptureController.PreviewInfo,
                             val pose: Pair<Double, Double>?) {
        val key get() = "${info.formatKey}/${info.exposureUs}/${bitmap.width}/${bitmap.height}"
    }
    private enum class Mode(val label: String) { LIVE("Live"), OVERLAY("Overlay"), BLINK("Blink"), REF("Ref") }
    private var mode = Mode.LIVE
    private var align = false
    private var live: Frame? = null
    private var reference: Frame? = null
    private var state: CaptureController.State? = null
    private var work: String? = null
    private var acceptFramesAfter = 0L
    private var notice = "Set a reference before rotating."
    private var step = 5.0
    private var blinkRef = false
    private var dialog: Dialog? = null
    private var image: CompareView? = null
    private var status: TextView? = null
    private var hint: TextView? = null
    private var refButton: TextView? = null
    private val modeKeys = linkedMapOf<Mode, TextView>()
    private val gestureKeys = ArrayList<TextView>()
    private val stageKeys = ArrayList<TextView>()

    private val tick = object : Runnable {
        override fun run() {
            if (dialog == null) return
            blinkRef = !blinkRef
            render()
            handler.postDelayed(this, 650)
        }
    }

    fun updateState(s: CaptureController.State, busy: String?) {
        val disconnected = state?.cameraOnline == true && !s.cameraOnline
        state = s
        if (disconnected) {
            live = null
            clearReference("Camera disconnected. Set a new reference after reconnecting.")
        }
        if (work != busy) {
            work = busy
            // Frames processed during a move/capture must not be frozen as the settled reference.
            live = null
            acceptFramesAfter = SystemClock.elapsedRealtime()
        }
        render()
    }

    fun preview(bitmap: Bitmap, info: CaptureController.PreviewInfo) {
        if (work != null || info.createdAtMs < acceptFramesAfter) return
        val next = Frame(bitmap, info, state?.stagePose)
        val oldKey = reference?.key ?: live?.key
        if (oldKey != null && oldKey != next.key) {
            clearReference("Preview settings changed. Set a new reference.")
        }
        live = next
        viewport.imageSize(bitmap.width, bitmap.height)
        render()
    }

    private fun clearReference(reason: String) {
        reference = null
        mode = Mode.LIVE
        align = false
        viewport.resetAlignment()
        notice = reason
    }

    /** Called on Finish/type change/exit, so a different calibration never inherits an old comparison. */
    fun reset() {
        close()
        live = null
        state = null
        work = null
        acceptFramesAfter = SystemClock.elapsedRealtime()
        viewport.fit()
        clearReference("Set a reference before rotating.")
    }

    fun close() { dialog?.dismiss() }

    fun show() {
        if (dialog != null) return
        val d = Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen)
        val root = kit.col().apply { setBackgroundColor(Palette.BG) }
        kit.padForSystemBars(root)
        val header = kit.row().apply { setPadding(kit.dp(10), kit.dp(6), kit.dp(10), kit.dp(6)) }
        header.addView(kit.text("Nodal compare", 17f, bold = true), kit.lp(0, kit.wrap(), 1f))
        header.addView(key("Help") { help() }, kit.lp(kit.dp(58), kit.dp(44), r = 6))
        header.addView(key("Close") { d.dismiss() }, kit.lp(kit.dp(64), kit.dp(44)))
        root.addView(header)

        image = CompareView(context)
        root.addView(image, kit.lp(kit.match(), 0, 1f))
        val controls = kit.col().apply { setPadding(kit.dp(10), kit.dp(8), kit.dp(10), kit.dp(10)) }
        status = kit.text("", 11.5f, Palette.TEXT2, mono = true).apply { setLines(2) }
        controls.addView(status)
        val referenceRow = kit.row()
        refButton = key("Set ref") {
            val f = live?.takeIf { fresh() } ?: return@key
            reference = f
            viewport.resetAlignment()
            mode = Mode.OVERLAY
            align = false
            notice = "Rotate, then align a far edge. Compare the nearby edge."
            render()
        }
        referenceRow.addView(refButton, kit.lp(kit.dp(94), kit.dp(44), r = 8))
        hint = kit.text("", 12f, Palette.TEXT2).apply { setLines(3) }
        referenceRow.addView(hint, kit.lp(0, kit.wrap(), 1f))
        controls.addView(referenceRow, kit.lp(t = 6))

        val modes = kit.row()
        for (m in Mode.entries) {
            val button = key(m.label) { mode = m; if (m == Mode.REF) align = false; blinkRef = false; render() }
            modeKeys[m] = button
            modes.addView(button, kit.lp(0, kit.dp(44), 1f, r = if (m == Mode.REF) 0 else 4))
        }
        controls.addView(modes, kit.lp(t = 6))

        val gestures = kit.row()
        gestureKeys += key("Pan / zoom") { align = false; render() }
        gestureKeys += key("Align far") {
            align = true
            mode = Mode.OVERLAY
            notice = "Drag LIVE until the far edges overlap; inspect the near edge."
            render()
        }
        gestureKeys += key("Reset shift") { viewport.resetAlignment(); render() }
        gestureKeys.forEachIndexed { i, v -> gestures.addView(v, kit.lp(0, kit.dp(44), 1f, r = if (i == 2) 0 else 4)) }
        controls.addView(gestures, kit.lp(t = 6))

        val navigation = kit.row()
        navigation.addView(key("−") { viewport.zoomAt(0.5f, viewport.viewWidth / 2f, viewport.viewHeight / 2f); render() }, kit.lp(0, kit.dp(40), 1f, r = 4))
        navigation.addView(key("Fit view") { viewport.fit(); render() }, kit.lp(0, kit.dp(40), 2f, r = 4))
        navigation.addView(key("+") { viewport.zoomAt(2f, viewport.viewWidth / 2f, viewport.viewHeight / 2f); render() }, kit.lp(0, kit.dp(40), 1f))
        controls.addView(navigation, kit.lp(t = 6))

        val arrows = kit.row()
        fun jog(label: String, yaw: Double, pitch: Double) = key(label) {
            // Unknown pose is intentionally not homed by this comparison tool.
            if (state?.stageOk == true && state?.stagePose != null && work == null) onJog(yaw * step, pitch * step)
        }.also { stageKeys += it }
        val buttons = listOf(jog("Yaw ◀", -1.0, 0.0), jog("Yaw ▶", 1.0, 0.0),
            jog("Pitch ▼", 0.0, -1.0), jog("Pitch ▲", 0.0, 1.0))
        for (b in buttons) arrows.addView(b, kit.lp(0, kit.dp(44), 1f, r = 4))
        val stepKey = key("${step.toInt()}° ▾") {}
        stepKey.setOnClickListener {
            val steps = listOf(1.0, 5.0, 10.0)
            AlertDialog.Builder(context, android.R.style.Theme_Material_Dialog_Alert)
                .setTitle("Degrees per arrow tap")
                .setSingleChoiceItems(steps.map { "${it.toInt()}°" }.toTypedArray(), steps.indexOf(step)) { picker, i ->
                    step = steps[i]; stepKey.text = "${step.toInt()}° ▾"; picker.dismiss()
                }.setNegativeButton("Cancel", null).show()
        }
        arrows.addView(stepKey, kit.lp(kit.dp(48), kit.dp(44)))
        controls.addView(arrows, kit.lp(t = 6))
        root.addView(controls)
        d.setContentView(root)
        d.window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        d.setOnDismissListener {
            handler.removeCallbacks(tick)
            dialog = null
            image = null
            status = null
            hint = null
            refButton = null
            modeKeys.clear()
            gestureKeys.clear()
            stageKeys.clear()
        }
        dialog = d
        d.show()
        d.window?.setLayout(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT)
        render()
        handler.postDelayed(tick, 650)
    }

    private fun key(label: String, action: () -> Unit): TextView = kit.button(label, Kit.Style.SECONDARY, 12.5f, action).apply {
        minHeight = 0
        setPadding(kit.dp(4), 0, kit.dp(4), 0)
        setSingleLine(true)
        contentDescription = label
    }

    private fun help() {
        AlertDialog.Builder(context, android.R.style.Theme_Material_Dialog_Alert)
            .setTitle("Compare a near and a far edge")
            .setMessage("1. Keep focus and aperture fixed. Place a near edge close to a distant edge in the image. Choose a clear, unsaturated preview exposure.\n\n" +
                "2. At the starting angle, tap Set ref. Rotate a few degrees with the arrows. Angles shown are commands, not measured positions. If unknown, close this window and establish the stage origin first.\n\n" +
                "3. Pinch / + to enlarge. Pan / zoom moves both images. Align far drags only LIVE: overlap the distant edges, then use Blink to inspect the nearby edge. Alignment is local translation only; compare adjacent edges, not opposite image corners.\n\n" +
                "4. Adjust the physical rail, return to the starting angle and Set ref again. Repeat both rotation directions, then check pitch and another part of the image.\n\n" +
                "5. Close keeps the reference. Use Note to record rail positions and Capture to save an HDR bracket. Finish or leaving Nodal clears the reference.\n\n" +
                "This is a preview comparison, not a nodal measurement. RGB uses 4×4 superpixels; Raw uses 2×2 max pooling. Zoom adds no sensor detail. Changing preview format/exposure or reconnecting clears the reference.")
            .setPositiveButton("OK", null).show()
    }

    private fun fresh(): Boolean {
        val f = live ?: return false
        val limitMs = maxOf(3000L, f.info.exposureUs / 1000 * 2 + 1000)
        return state?.cameraOnline == true && work == null && SystemClock.elapsedRealtime() - f.info.createdAtMs <= limitMs
    }

    private fun pose(p: Pair<Double, Double>?) = p?.let {
        String.format(Locale.US, "Y%+.1f° P%+.1f°", it.first, it.second)
    } ?: "pose unknown"

    private fun render() {
        if (dialog == null) return
        val ref = reference
        val ready = fresh()
        status?.text = "REF  ${ref?.let { pose(it.pose) + " · " + fmtExposure(it.info.exposureUs) } ?: "not set"}\n" +
            "CMD  ${pose(state?.stagePose)} · ${work ?: if (ready) "live" else "waiting for preview"}"
        refButton?.let { kit.enable(it, ready); it.text = if (ref == null) "Set ref" else "New ref" }
        hint?.text = when {
            state?.cameraOnline != true -> "No camera. Reconnect to compare."
            work != null -> "Waiting for the stage / camera."
            state?.stagePose == null -> "Stage pose unknown. Set origin in the main screen to use arrows."
            align -> "Drag LIVE to align a far edge. Pinch zooms both images."
            else -> notice
        }
        for ((m, b) in modeKeys) {
            kit.enable(b, m == Mode.LIVE || ref != null)
            selected(b, m == mode)
        }
        gestureKeys.forEachIndexed { i, b ->
            kit.enable(b, i == 0 || ref != null)
            selected(b, (i == 0 && !align) || (i == 1 && align))
        }
        stageKeys.forEach { kit.enable(it, state?.stageOk == true && state?.stagePose != null && work == null) }
        image?.invalidate()
    }

    private fun selected(view: TextView, selected: Boolean) {
        if (view.isSelected == selected) return
        view.isSelected = selected
        view.background = kit.ripple(kit.rounded(if (selected) Palette.BLUE else Palette.SURFACE2, 10f), 10f)
    }

    private inner class CompareView(ctx: Context) : View(ctx) {
        private val paint = Paint().apply { isFilterBitmap = false }
        private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Palette.TEXT; textSize = kit.dp(12).toFloat(); typeface = kit.mono
        }
        private val labelBg = Paint().apply { color = 0xBB000000.toInt() }
        private var multiTouch = false
        private val scaleGesture = ScaleGestureDetector(ctx, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                viewport.zoomAt(detector.scaleFactor, detector.focusX, detector.focusY)
                invalidate()
                return true
            }
        })
        private val gesture = GestureDetector(ctx, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent) = true
            override fun onScroll(e1: MotionEvent?, e2: MotionEvent, distanceX: Float, distanceY: Float): Boolean {
                if (multiTouch || scaleGesture.isInProgress) return true
                if (align && reference != null && mode != Mode.REF) viewport.alignLive(-distanceX, -distanceY)
                else viewport.pan(-distanceX, -distanceY)
                invalidate()
                return true
            }
            override fun onDoubleTap(e: MotionEvent): Boolean {
                if (viewport.zoom >= 8f) viewport.fit() else viewport.zoomAt(2f, e.x, e.y)
                invalidate()
                return true
            }
            override fun onSingleTapUp(e: MotionEvent): Boolean { performClick(); return true }
        })

        init { contentDescription = "Nodal image comparison. Pinch to zoom; drag to pan or align live." }
        override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) { viewport.viewSize(w, h) }
        override fun performClick(): Boolean { super.performClick(); return true }
        override fun onTouchEvent(event: MotionEvent): Boolean {
            if (event.actionMasked == MotionEvent.ACTION_DOWN) multiTouch = false
            if (event.pointerCount > 1) multiTouch = true
            scaleGesture.onTouchEvent(event)
            gesture.onTouchEvent(event)
            return true
        }

        override fun onDraw(canvas: Canvas) {
            canvas.drawColor(0xFF000000.toInt())
            val r = reference
            val f = live?.takeIf { fresh() }
            val refOnly = r != null && (mode == Mode.REF || (mode == Mode.BLINK && blinkRef))
            val overlay = mode == Mode.OVERLAY && r != null && f != null
            fun draw(frame: Frame, current: Boolean, alpha: Int = 255) {
                val dx = if (current) viewport.liveX * viewport.scale else 0f
                val dy = if (current) viewport.liveY * viewport.scale else 0f
                val x = viewport.left + dx
                val y = viewport.top + dy
                paint.alpha = alpha
                canvas.drawBitmap(frame.bitmap, null, RectF(x, y, x + frame.bitmap.width * viewport.scale, y + frame.bitmap.height * viewport.scale), paint)
            }
            if (refOnly || overlay) r?.let { draw(it, false) }
            if (!refOnly) f?.let { draw(it, true, if (overlay) 128 else 255) }
            val shown = if (refOnly) "REFERENCE" else if (f == null) "WAITING FOR LIVE" else if (overlay) "50% OVERLAY" else "LIVE"
            val size = (f ?: r)?.bitmap?.let { "${it.width}×${it.height}" } ?: ""
            val label = String.format(Locale.US, "%s · %.1f× · %s", shown, viewport.zoom, size)
            canvas.drawRect(0f, 0f, width.toFloat(), kit.dp(30).toFloat(), labelBg)
            canvas.drawText(label, kit.dp(8).toFloat(), kit.dp(20).toFloat(), labelPaint)
        }
    }
}
