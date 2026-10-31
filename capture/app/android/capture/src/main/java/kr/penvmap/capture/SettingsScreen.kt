package kr.penvmap.capture

import android.app.Dialog
import android.os.Build
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import com.cgjnkim.mobileokad.flir.PenvmapCamera

/**
 * Full-screen settings, in sections: preview, camera, rotation stage, fixed (shown, not editable), about. The project name and the base path
 * are set on the main screen. Nothing changes until Apply; then the camera and stage are reopened.
 * Reset to defaults refills the form with [AppSettings.withSettingsDefaults] (applied with Apply too).
 */
class SettingsScreen(private val activity: MainActivity, initial: AppSettings, private val onApply: (AppSettings) -> Unit) {
    private val kit = Kit(activity)
    private var s = initial
    private lateinit var url: TextView
    private lateinit var yaws: TextView
    private lateinit var pitches: TextView
    private lateinit var settleYaw: TextView
    private lateinit var settlePitch: TextView
    private lateinit var roll: TextView

    fun show() {
        val d = Dialog(activity, android.R.style.Theme_Material_NoActionBar)
        val root = kit.col().apply { setBackgroundColor(Palette.BG) }
        kit.padForSystemBars(root)

        val body = kit.col().apply { setPadding(kit.dp(14), kit.dp(4), kit.dp(14), kit.dp(24)) }
        val scroll = ScrollView(activity).apply { addView(body) }
        lateinit var reset: () -> Unit
        reset = {
            s = s.withSettingsDefaults()
            body.removeAllViews()
            fill(body, reset)
            scroll.scrollTo(0, 0)
            Toast.makeText(activity, "Defaults restored (preview, camera, stage) -- tap Apply", Toast.LENGTH_LONG).show()
        }
        fill(body, reset)

        // ---- top bar
        val bar = kit.row().apply { setPadding(kit.dp(8), kit.dp(6), kit.dp(14), kit.dp(6)) }
        bar.addView(kit.button("Cancel", Kit.Style.GHOST, 14.5f) { d.dismiss() })
        bar.addView(kit.text("Settings", 18f, bold = true).apply { gravity = Gravity.CENTER }, kit.lp(0, kit.wrap(), 1f))
        bar.addView(kit.button("Apply", Kit.Style.PRIMARY, 14.5f) {
            val ns = runCatching {
                s.copy(
                    mechaUrl = url.text.toString().trim(),
                    yawDegs = parseList(yaws.text.toString()),
                    pitchDegs = parseList(pitches.text.toString()),
                    stageSettleYawS = settleYaw.text.toString().toDouble(),
                    stageSettlePitchS = settlePitch.text.toString().toDouble(),
                    mountRollDeg = roll.text.toString().toDouble(),
                )
            }.getOrElse {
                Toast.makeText(activity, "Invalid value: ${it.message}", Toast.LENGTH_LONG).show()
                return@button
            }
            if (ns.yawDegs.isEmpty() || ns.pitchDegs.isEmpty()) {
                Toast.makeText(activity, "Yaw and pitch need at least one angle each", Toast.LENGTH_LONG).show()
                return@button
            }
            d.dismiss()
            onApply(ns)
        }.apply { minHeight = kit.dp(40) })
        root.addView(bar)
        root.addView(scroll, kit.lp(kit.match(), 0, 1f))

        d.setContentView(root)
        d.window?.apply {
            setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            statusBarColor = Palette.BG
            navigationBarColor = Palette.BG
        }
        d.show()
    }

    /** The form, from [s]; [onReset] is the Reset to defaults button. */
    private fun fill(body: LinearLayout, onReset: () -> Unit) {
        // ---- preview
        val preview = section(body, "Preview")
        val (ldrRow, _) = kit.switchRow("Save LDR", "Save the middle exposure as 16-bit RAW PNG", s.saveRawPng) { s = s.copy(saveRawPng = it) }
        preview.addView(ldrRow)
        preview.addView(kit.text("Display", 14f, Palette.TEXT2), kit.lp(t = 14))
        val modes = listOf(AppSettings.PREVIEW_RGB, AppSettings.PREVIEW_RAW)
        preview.addView(kit.segmented(listOf("RGB", "Raw"), modes.indexOf(s.previewMode).coerceAtLeast(0)) { s = s.copy(previewMode = modes[it]) }, kit.lp(t = 6))
        preview.addView(kit.text("Orientation", 14f, Palette.TEXT2), kit.lp(t = 12))
        preview.addView(kit.segmented(listOf("Sky up", "Sensor"), if (s.previewShowSensor) 1 else 0) { s = s.copy(previewShowSensor = it == 1) }, kit.lp(t = 6))
        preview.addView(kit.text("Frame rate", 14f, Palette.TEXT2), kit.lp(t = 12))
        val rates = listOf(2.0, 4.0, 8.0)
        val rateIndex = rates.indexOfFirst { it == s.previewFps }.takeIf { it >= 0 } ?: 1
        preview.addView(kit.segmented(rates.map { "${fmt(it)} fps" }, rateIndex) { s = s.copy(previewFps = rates[it]) }, kit.lp(t = 6))

        // ---- camera
        val camera = section(body, "Camera")
        val camModes = listOf(AppSettings.CAMERA_FLIR, AppSettings.CAMERA_BASLER, AppSettings.CAMERA_FAKE)
        camera.addView(kit.segmented(
            listOf("FLIR (USB)", "Basler", "Simulated"), camModes.indexOf(s.cameraMode).coerceAtLeast(0),
            disabled = setOf(camModes.indexOf(AppSettings.CAMERA_BASLER)),
            onDisabledTap = { Toast.makeText(activity, "Basler is not supported yet", Toast.LENGTH_SHORT).show() },
        ) { i -> s = s.copy(cameraMode = camModes[i]) })
        fun tri(v: Boolean?) = when (v) { null -> 0; true -> 1; false -> 2 }
        fun untri(i: Int) = when (i) { 1 -> true; 2 -> false; else -> null }
        camera.addView(kit.text("Black level clamping", 14f, Palette.TEXT2), kit.lp(t = 14))
        camera.addView(kit.segmented(listOf("Camera's", "On", "Off"), tri(s.blackLevelClamping)) { s = s.copy(blackLevelClamping = untri(it)) }, kit.lp(t = 6))
        camera.addView(kit.text("Static defect correction", 14f, Palette.TEXT2), kit.lp(t = 12))
        camera.addView(kit.segmented(listOf("Camera's", "On", "Off"), tri(s.defectCorrectStatic)) { s = s.copy(defectCorrectStatic = untri(it)) }, kit.lp(t = 6))

        // ---- stage
        val stage = section(body, "Rotation stage")
        val stageModes = listOf(AppSettings.STAGE_HTTP, AppSettings.STAGE_FAKE)
        stage.addView(kit.segmented(listOf("HTTP stage", "Simulated"), stageModes.indexOf(s.stageMode).coerceAtLeast(0)) { i ->
            s = s.copy(stageMode = stageModes[i])
        })
        stage.addView(kit.text("Address (mecha_url)", 14f, Palette.TEXT2), kit.lp(t = 14))
        url = kit.field(s.mechaUrl, hint = "http://192.168.8.1")
        stage.addView(url, kit.lp(t = 6))
        stage.addView(kit.text("Yaw angles (°)", 14f, Palette.TEXT2), kit.lp(t = 12))
        yaws = kit.field(s.yawDegs.joinToString(", ") { fmt(it) })
        stage.addView(yaws, kit.lp(t = 6))
        stage.addView(kit.text("Pitch angles (°)", 14f, Palette.TEXT2), kit.lp(t = 12))
        pitches = kit.field(s.pitchDegs.joinToString(", ") { fmt(it) })
        stage.addView(pitches, kit.lp(t = 6))
        val nums = kit.row()
        settleYaw = numberBox(nums, "Settle yaw (s)", fmt(s.stageSettleYawS))
        settlePitch = numberBox(nums, "Settle pitch (s)", fmt(s.stageSettlePitchS))
        roll = numberBox(nums, "Mount roll (°)", fmt(s.mountRollDeg))
        stage.addView(nums, kit.lp(t = 12))

        // ---- fixed: what PenvmapCamera.applyPreset sets every time the camera opens, as on the PC
        val fixed = section(body, "Fixed")
        val onOff = listOf("On", "Off")
        fun fixedRow(title: String, control: android.view.View) {
            val r = kit.row()
            r.addView(kit.text(title, 14f, Palette.TEXT2), kit.lp(0, kit.wrap(), 1f))
            r.addView(control, kit.lp(kit.dp(150), kit.wrap()))
            fixed.addView(r, kit.lp(t = if (fixed.childCount == 0) 0 else 10))
        }
        fixedRow("Gain (dB)", kit.fixedField("0"))
        fixedRow("Exposure auto", kit.fixedSegmented(onOff, 1))
        fixedRow("Gain auto", kit.fixedSegmented(onOff, 1))
        fixedRow("White balance auto", kit.fixedSegmented(onOff, 1))
        fixedRow("Balance ratio", kit.fixedField("1"))
        fixedRow("Gamma", kit.fixedSegmented(onOff, 1))
        fixedRow("ISP", kit.fixedSegmented(onOff, 1))
        fixedRow("Black level", kit.fixedField("0"))
        fixedRow("ADC bit depth", kit.fixedSegmented(listOf("10 bit", "12 bit"), 1))
        fixedRow("Pixel format", kit.fixedField(PenvmapCamera.CAPTURE_FORMAT))
        fixedRow("Reverse X", kit.fixedSegmented(onOff, 1))
        fixedRow("Reverse Y", kit.fixedSegmented(onOff, 1))

        // ---- about
        val about = section(body, "About")
        val version = runCatching { activity.packageManager.getPackageInfo(activity.packageName, 0).versionName }.getOrNull()
        about.addView(kit.text(
            "Penvmap Capture $version  ·  ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE}\n" +
                "Base path: ${Storage.readable(Storage.baseDir(activity, Storage.DOCUMENTS).absolutePath)} (project name and base path: main screen)\n" +
                "Masks: ${activity.getExternalFilesDir(null)}/masks/mask_{pitch}.png\n" +
                "Diagnostics: ${activity.getExternalFilesDir(null)}/diagnostics\n" +
                "PC: python app/tools/pull_sessions.py --pull --verify",
            12.5f, Palette.TEXT2,
        ))

        body.addView(kit.button("Reset to defaults", Kit.Style.SECONDARY, 14.5f) { onReset() }.apply { minHeight = kit.dp(46) }, kit.lp(t = 22))
    }

    private fun section(body: LinearLayout, title: String): LinearLayout {
        body.addView(kit.label(title), kit.lp(t = 16, b = 8, l = 4))
        val card = kit.card(14)
        body.addView(card)
        return card
    }

    private fun divider() = android.view.View(activity).apply { setBackgroundColor(Palette.STROKE) }

    private fun numberBox(row: LinearLayout, title: String, value: String): TextView {
        val c = kit.col()
        c.addView(kit.text(title, 12.5f, Palette.TEXT2))
        val f = kit.field(value, numeric = true)
        c.addView(f, kit.lp(t = 6))
        row.addView(c, kit.lp(0, kit.wrap(), 1f, r = if (row.childCount < 2) 8 else 0))
        return f
    }

    private fun parseList(t: String): List<Double> = t.split(',', ' ').filter { it.isNotBlank() }.map { it.trim().toDouble() }

    private fun fmt(d: Double) = if (d == Math.rint(d)) d.toLong().toString() else d.toString()
}
