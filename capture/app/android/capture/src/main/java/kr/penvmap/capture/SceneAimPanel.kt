package kr.penvmap.capture

import android.app.AlertDialog
import android.content.Context
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import kr.penvmap.capture.core.CalibrationKind

enum class SceneAimKind(val label: String) { COLOR_CHECKER("Color checker"), SUN("Sun"), FLOOR("Floor") }

/** Aiming controls occupy exactly the scene card; the capture actions keep their bottom-row height. */
class SceneAimPanel(
    context: Context,
    onJog: (Double, Double) -> Unit,
    onHome: () -> Unit,
    onClose: () -> Unit,
    onCapture: () -> Unit,
    onCancel: () -> Unit,
    onNodalCompare: () -> Unit,
) {
    private val kit = Kit(context)
    private var noteScope: String? = null
    private var floorMode = false
    var note = ""
        private set
    private var busy = false
    private val steps = listOf(1.0, 5.0, 10.0, 30.0, 90.0)
    private var stepDeg = 1.0
    var ndOd: Double? = 4.0
        private set
    val view = kit.row().apply {
        id = R.id.aim_panel
        background = kit.rounded(Palette.SURFACE, 18f)
        setPadding(kit.dp(8), kit.dp(3), kit.dp(8), kit.dp(3))
    }
    val actions = kit.row().apply { id = R.id.aim_actions }
    private val count = kit.text("", 13.5f, Palette.TEXT, bold = true).apply { setSingleLine(true) }
    private val pose = kit.text("", 12.5f, Palette.TEXT2, mono = true).apply { setLines(2) }
    private val step = key("1° ▾", "Angle step") {
        AlertDialog.Builder(context, android.R.style.Theme_Material_Dialog_Alert)
            .setTitle("Degrees per arrow tap")
            .setSingleChoiceItems(steps.map { "${it.toInt()}°" }.toTypedArray(), steps.indexOf(stepDeg)) { d, i ->
                stepDeg = steps[i]
                updateStep()
                d.dismiss()
            }.setNegativeButton("Cancel", null).show()
    }
    private val nd = key("ND OD 4 ▾", "ND filter on the lens") {
        val values = listOf<Double?>(null, 2.0, 3.0, 4.0)
        AlertDialog.Builder(context, android.R.style.Theme_Material_Dialog_Alert)
            .setTitle("ND filter on the lens")
            .setSingleChoiceItems(arrayOf("None", "OD 2", "OD 3", "OD 4"), values.indexOf(ndOd)) { d, i ->
                ndOd = values[i]
                updateNd()
                d.dismiss()
            }.setNegativeButton("Cancel", null).show()
    }
    private val noteButton: TextView = key("Note", "Calibration condition or note") {
        val field = kit.field(note, hint = if (floorMode) "Rig shift / direction / conditions (optional)" else "Condition / note (optional)")
        val box = kit.col().apply {
            setPadding(kit.dp(18), kit.dp(8), kit.dp(18), 0)
            if (floorMode) addView(kit.text("After the scene capture, move the whole rig aside and aim at the floor it covered. Keep the focus and aperture; remove ND.", 14f), kit.lp(b = 10))
            addView(field)
        }
        AlertDialog.Builder(context, android.R.style.Theme_Material_Dialog_Alert)
            .setTitle(if (floorMode) "Floor shot note" else "Capture condition").setView(box)
            .setPositiveButton("Save") { _, _ ->
                note = field.text.toString().trim()
                noteButton.text = if (note.isEmpty()) "Note" else "Note •"
            }.setNegativeButton("Cancel", null).show()
    }
    private val options = kit.row()
    private val compare = key("Compare", "Compare nodal reference and live preview", onNodalCompare)
    private val stageKeys = ArrayList<View>()
    private val close = kit.button("Close", Kit.Style.SECONDARY, 16f) {
        if (busy) onCancel() else onClose()
    }
    private val capture = kit.button("Capture checker", Kit.Style.PRIMARY, 17f, onCapture)

    init {
        val info = kit.col().apply { gravity = Gravity.CENTER_VERTICAL }
        info.addView(count)
        info.addView(pose, kit.lp(t = 4))
        options.addView(nd, kit.lp(0, kit.match(), 1.4f))
        options.addView(noteButton, kit.lp(0, kit.match(), 1f, l = 4))
        options.addView(compare, kit.lp(0, kit.match(), 1.5f, l = 4))
        info.addView(options, kit.lp(kit.match(), kit.dp(30), t = 4))
        view.addView(info, kit.lp(0, kit.match(), 1f, r = 8))

        val pad = kit.col()
        fun arrow(text: String, description: String, yaw: Double, pitch: Double) = key(text, description) {
            onJog(yaw * stepDeg, pitch * stepDeg)
        }.also { it.textSize = 19f; stageKeys += it }
        val home = key("0,0", "Stage origin", onHome).also { stageKeys += it }
        val up = arrow("▲", "Pitch up", 0.0, 1.0)
        val left = arrow("◀", "Yaw left", -1.0, 0.0)
        val down = arrow("▼", "Pitch down", 0.0, -1.0)
        val right = arrow("▶", "Yaw right", 1.0, 0.0)
        for ((i, keys) in listOf(listOf(step, up, home), listOf(left, down, right)).withIndex()) {
            val row = kit.row()
            keys.forEachIndexed { j, v -> row.addView(v, kit.lp(0, kit.match(), 1f, r = if (j < 2) 4 else 0)) }
            pad.addView(row, kit.lp(kit.match(), 0, 1f, b = if (i == 0) 4 else 0))
        }
        view.addView(pad, kit.lp(kit.dp(140), kit.match()))
        actions.addView(close, kit.lp(0, kit.dp(64), 1f, r = 8))
        actions.addView(capture, kit.lp(0, kit.dp(64), 2f))
        nd.setTextColor(Palette.AMBER)
    }

    private fun key(text: String, description: String, click: () -> Unit): TextView =
        kit.button(text, Kit.Style.SECONDARY, 12.5f, click).apply {
            contentDescription = description
            minHeight = 0
            minimumHeight = 0
            setPadding(0, 0, 0, 0)
            setSingleLine(true)
            background = kit.ripple(kit.rounded(Palette.SURFACE2, 10f, Palette.STROKE), 10f)
        }

    private fun updateStep() { step.text = "${stepDeg.toInt()}° ▾" }
    private fun updateNd() {
        val prefix = if (noteButton.visibility == View.VISIBLE) "" else "ND "
        nd.text = ndOd?.let { "${prefix}OD ${it.toInt()} ▾" } ?: "${prefix}none ▾"
    }

    fun render(mode: SceneAimKind, s: CaptureController.State, work: String?) {
        floorMode = mode == SceneAimKind.FLOOR
        if (floorMode) setNoteScope("floor:${s.sessionRel}")
        val shots = when (mode) {
            SceneAimKind.COLOR_CHECKER -> s.colorCheckers
            SceneAimKind.SUN -> s.sunShots
            SceneAimKind.FLOOR -> s.floorShots
        }
        val label = when (mode) {
            SceneAimKind.COLOR_CHECKER -> "Capture checker"
            SceneAimKind.SUN -> "Capture sun"
            SceneAimKind.FLOOR -> "Capture floor"
        }
        val canCapture = !floorMode || (s.sessionRel != null && s.viewsDone > 0 && s.resumeFrom == null)
        renderControls(mode, mode.label, shots, label, "Close", false, canCapture, s, work)
    }

    private fun setNoteScope(scope: String) {
        if (scope != noteScope) {
            noteScope = scope
            note = ""
            noteButton.text = "Note"
        }
    }

    fun renderCalibration(s: CaptureController.State, work: String?) {
        floorMode = false
        val kind = s.calibrationKind
        setNoteScope("calibration:$kind")
        val style = when (kind) {
            CalibrationKind.WB_CC -> SceneAimKind.COLOR_CHECKER
            CalibrationKind.ND -> SceneAimKind.SUN
            else -> null
        }
        val label = when (kind) {
            CalibrationKind.ND -> "ND"
            CalibrationKind.HDR -> "HDR / sensor"
            else -> kind?.label ?: "Calibration"
        }
        renderControls(style, label, s.calibrationShots,
            when (kind) { CalibrationKind.WB_CC -> "Capture checker"; CalibrationKind.ND -> "Capture ND"; else -> "Capture" },
            "Finish", true, kind != null, s, work)
    }

    private fun renderControls(style: SceneAimKind?, label: String, shots: Int, captureLabel: String,
                               closeLabel: String, reference: Boolean, canCapture: Boolean,
                               s: CaptureController.State, work: String?) {
        val styleKey = style?.name ?: "reference"
        if (capture.tag != styleKey) {
            val colors = when (style) {
                SceneAimKind.SUN -> intArrayOf(0xFFFFC107.toInt(), 0xFFFF6F00.toInt())
                SceneAimKind.COLOR_CHECKER -> intArrayOf(0xFFE53935.toInt(), 0xFFF57C00.toInt(), 0xFFFBC02D.toInt(), 0xFF43A047.toInt(), 0xFF1E88E5.toInt(), 0xFF8E24AA.toInt())
                SceneAimKind.FLOOR, null -> intArrayOf(Palette.BLUE, Palette.BLUE)
            }
            val fill = GradientDrawable(GradientDrawable.Orientation.LEFT_RIGHT, colors).apply { cornerRadius = kit.dp(14).toFloat() }
            capture.background = kit.ripple(fill, 14f)
            capture.setShadowLayer(6f, 0f, 1.5f, 0xCC000000.toInt())
            capture.tag = styleKey
        }
        busy = work != null
        count.text = "$label · $shots"
        pose.text = s.stagePose?.let { (y, p) -> "Yaw   %6.1f°\nPitch %6.1f°".format(y, p) } ?: "Yaw     —\nPitch   —"
        nd.visibility = if (style == SceneAimKind.SUN) View.VISIBLE else View.GONE
        noteButton.visibility = if ((reference && canCapture) || style == SceneAimKind.FLOOR) View.VISIBLE else View.GONE
        noteButton.contentDescription = if (floorMode) "Floor shot note and instructions" else "Calibration condition or note"
        compare.visibility = if (reference && s.calibrationKind == CalibrationKind.NODAL) View.VISIBLE else View.GONE
        options.visibility = if (nd.visibility == View.VISIBLE || noteButton.visibility == View.VISIBLE || compare.visibility == View.VISIBLE) View.VISIBLE else View.GONE
        updateNd()
        for (v in stageKeys) kit.enable(v, !busy && s.stageOk)
        kit.enable(step, !busy)
        kit.enable(nd, !busy)
        kit.enable(noteButton, !busy)
        kit.enable(compare, !busy)
        kit.enable(capture, !busy && canCapture)
        kit.enable(close, busy || !reference || canCapture)
        close.text = if (busy) "Cancel" else closeLabel
        capture.text = if (work in listOf("color checker", "sun", "floor", "calibration capture")) "Capturing…" else captureLabel
    }
}

/** The original scene card determines the height, even while the aiming controls cover it. */
class SceneSlot(context: Context, private val scene: View, private val aiming: View) : ViewGroup(context) {
    init {
        id = R.id.scene_slot
        addView(scene, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(aiming, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        measureChild(scene, widthMeasureSpec, heightMeasureSpec)
        setMeasuredDimension(scene.measuredWidth, scene.measuredHeight)
        aiming.measure(MeasureSpec.makeMeasureSpec(measuredWidth, MeasureSpec.EXACTLY),
            MeasureSpec.makeMeasureSpec(measuredHeight, MeasureSpec.EXACTLY))
    }

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        scene.layout(0, 0, r - l, b - t)
        aiming.layout(0, 0, r - l, b - t)
    }
}
