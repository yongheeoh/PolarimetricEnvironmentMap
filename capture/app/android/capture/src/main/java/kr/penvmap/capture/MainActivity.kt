package kr.penvmap.capture

import android.app.Activity
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.drawable.GradientDrawable
import android.hardware.usb.UsbManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.cgjnkim.mobileokad.flir.FlirProbe
import kr.penvmap.capture.CaptureController.Level
import kr.penvmap.capture.core.SessionIndex
import kr.penvmap.capture.core.CalibrationKind
import kr.penvmap.capture.core.CalibrationShot

/**
 * One screen, laid out in code: the upright S0 preview (the live envmap during a rotation), the bracket,
 * the session, and the capture actions of the PC's keys.
 *
 * For scripted testing from the PC the same actions can be sent with
 * `adb shell am start -n kr.penvmap.capture/.MainActivity --es action <cc|rotate|resume|check|auto|finish|...>`
 * and settings with `--es settings '<json>'` (see [AppSettings.fromJson]).
 */
class MainActivity : Activity(), CaptureController.Ui {

    private lateinit var kit: Kit
    private lateinit var controller: CaptureController
    private val main = Handler(Looper.getMainLooper())

    private lateinit var experimentText: TextView
    private lateinit var camChip: TextView
    private lateinit var stageChip: TextView
    private lateinit var folderButton: TextView
    private lateinit var storageChip: TextView
    private lateinit var previewImage: ImageView
    private lateinit var panoImage: ImageView
    private lateinit var expandButton: ImageView
    private lateinit var modeTabs: LinearLayout
    private lateinit var rgbTab: TextView
    private lateinit var rawTab: TextView
    /** The full-screen view while it is open (it follows the stream / panorama). */
    private var fullImage: ImageView? = null
    private var fullText: TextView? = null
    private var fullPanorama = false
    private lateinit var overlayText: TextView
    private lateinit var emptyText: TextView
    private var haveFrame = false
    private lateinit var liveTab: TextView
    private lateinit var panoTab: TextView
    private lateinit var tabs: LinearLayout
    private lateinit var bracketSummary: TextView
    private lateinit var pillsRow: LinearLayout
    private lateinit var checkText: TextView
    /** Scene card: folder (date time), counts, what is running, Cancel, progress bar. */
    private lateinit var sceneIdText: TextView
    private lateinit var sessionText: TextView
    private lateinit var busySub: TextView
    private lateinit var busyLine: ProgressLine
    private lateinit var busyCancel: TextView
    private lateinit var messageText: TextView
    private lateinit var ccButton: TextView
    private lateinit var rotateButton: TextView
    /** Ends the open scene; enabled only when one is open and nothing runs. */
    private lateinit var finishButton: TextView
    private lateinit var sunButton: TextView
    private lateinit var floorButton: TextView
    private lateinit var calibrationButton: TextView
    private var aimKind: SceneAimKind? = null
    private val sunMode: Boolean get() = aimKind == SceneAimKind.SUN
    private lateinit var actionsRow: LinearLayout
    private lateinit var aimPanel: SceneAimPanel
    private val nodalCompare by lazy { NodalCompare(this) { yaw, pitch -> controller.stageJog(yaw, pitch) } }
    private var nodalActive = false
    private lateinit var crosshair: Crosshair
    private lateinit var sceneCardView: View
    private lateinit var sceneSlotView: View
    private val lockedWhileBusy = ArrayList<View>()

    private var state: CaptureController.State? = null
    private var showPanorama = false
    private var havePanorama = false
    private var lastOverlay = ""
    private var rotating = false
    private var exposureInfo: TextView? = null
    private var folderCallback: ((Uri?) -> Unit)? = null
    /** Continues (or cancels) choosing a base path once the user is back from the All files access screen. */
    private var afterAllFilesAccess: Pair<() -> Unit, () -> Unit>? = null

    private val usbDetached = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != UsbManager.ACTION_USB_DEVICE_DETACHED) return
            val device = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, android.hardware.usb.UsbDevice::class.java)
            } else {
                @Suppress("DEPRECATION") intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
            }
            // Only the FLIR: unplugging the SSD (or anything else on a hub) is not a camera drop.
            val flir = device?.vendorId == FlirProbe.VENDOR_ID || (device == null && FlirProbe.findDevice(context) == null)
            if (flir) controller.onUsbDetached()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.statusBarColor = Palette.BG
        window.navigationBarColor = Palette.BG
        kit = Kit(this)
        setContentView(buildUi())
        controller = CaptureController(applicationContext, AppSettings.load(this), this)
        setPreviewMode(controller.settings.previewMode, save = false)
        val filter = IntentFilter(UsbManager.ACTION_USB_DEVICE_DETACHED)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) registerReceiver(usbDetached, filter, Context.RECEIVER_EXPORTED)
        else registerReceiver(usbDetached, filter)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            onBackInvokedDispatcher.registerOnBackInvokedCallback(android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT) { handleBack() }
        }
        handleIntent(intent, startup = true)
        // Optional session location; capture works without it. Android 9 / 10 also need the storage permission
        // for the base path Documents/penvmap (Android 11+ needs none for the app's own files there).
        val wanted = mutableListOf<String>()
        if (checkSelfPermission(android.Manifest.permission.ACCESS_FINE_LOCATION) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            wanted += listOf(android.Manifest.permission.ACCESS_FINE_LOCATION, android.Manifest.permission.ACCESS_COARSE_LOCATION)
        }
        if (Storage.needsLegacyPermission(Storage.DOCUMENTS) &&
            checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE) != android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            wanted += android.Manifest.permission.WRITE_EXTERNAL_STORAGE
        }
        if (wanted.isNotEmpty()) requestPermissions(wanted.toTypedArray(), 1)
    }

    override fun onResume() {
        super.onResume()
        val (next, cancel) = afterAllFilesAccess ?: return
        afterAllFilesAccess = null
        if (Storage.hasAllFilesAccess()) {
            next()
        } else {
            message("All files access is off -- the base path stays where it was", Level.WARN)
            cancel()
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIntent(intent, startup = false)
    }

    /**
     * Leaving the app ends the session (and would abort a rotation), so never on a stray back press. Back
     * gestures reach [onBackPressed] up to Android 15 and the OnBackInvokedCallback from Android 16 on.
     * (Lint flags the override for API 36+; there the callback registered in onCreate handles back.)
     */
    @android.annotation.SuppressLint("GestureBackNavigation")
    @Deprecated("Deprecated in Java")
    override fun onBackPressed() = handleBack()

    private fun handleBack() {
        if (controller.busyWith != null) {
            message("Busy (${controller.busyWith}) -- use Cancel to stop", Level.WARN)
            return
        }
        if (controller.calibrationMode) {
            if (controller.calibrationKind != null) finishCalibration() else leaveCalibration()
            return
        }
        if (aimKind != null) {
            exitAimMode()
            return
        }
        dialog()
            .setTitle("Exit the capture app?")
            .setMessage("The current scene is finished and the camera is closed.")
            .setPositiveButton("Exit") { _, _ -> finish() }
            .setNegativeButton("Stay", null)
            .show()
    }

    override fun onDestroy() {
        if (nodalActive) nodalCompare.reset()
        runCatching { unregisterReceiver(usbDetached) }
        controller.shutdown()
        super.onDestroy()
    }

    override fun onPause() {
        if (nodalActive) nodalCompare.close()
        super.onPause()
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        @Suppress("DEPRECATION")
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQ_FOLDER) return
        folderCallback?.invoke(if (resultCode == RESULT_OK) data?.data else null)
        folderCallback = null
    }

    /** System file explorer for a folder (phone, SD card, USB drive); only its path is used ([Storage.pathFromTree]). */
    private fun pickFolder(onPicked: (Uri?) -> Unit) {
        folderCallback = onPicked
        val i = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
        @Suppress("DEPRECATION")
        startActivityForResult(i, REQ_FOLDER)
    }

    // ---- intents (USB attach, scripted actions) ----------------------------------------------------

    private fun handleIntent(intent: Intent?, startup: Boolean) {
        intent?.getStringExtra("settings")?.let { json ->
            val s = runCatching { AppSettings.fromJson(json) }.getOrNull()
            if (s != null) {
                AppSettings.save(this, s)
                openCamera(s)
            } else {
                message("bad settings json", Level.ERROR)
            }
            return
        }
        intent?.getStringExtra("notes")?.let { json ->
            runCatching {
                @Suppress("UNCHECKED_CAST")
                controller.annotations = (kr.penvmap.capture.core.Json.parse(json) as Map<String, Any?>).mapValues { it.value.toString() }
            }.onFailure { message("bad notes json", Level.ERROR) }
        }
        when (intent?.getStringExtra("action")) {
            "cc" -> controller.captureColorChecker()
            "cc_aim" -> enterAimMode(SceneAimKind.COLOR_CHECKER)
            "cc_close" -> exitAimMode()
            "calibration" -> controller.beginCalibration(CalibrationKind.fromId(intent.getStringExtra("type")))
            "calibration_capture" -> controller.calibrationKind?.let { kind ->
                controller.captureCalibration(CalibrationShot(kind, intent.getStringExtra("note") ?: "",
                    intent.getDoubleExtra("nd_od", Double.NaN).takeIf { it.isFinite() }))
            }
            "calibration_finish" -> controller.finishCalibration()
            "nodal_compare" -> showNodalCompare()
            "rotate" -> controller.rotate()
            "sun" -> enterSunMode()
            "sun_jog" -> controller.stageJog(intent.getDoubleExtra("yaw", 0.0), intent.getDoubleExtra("pitch", 0.0))
            "sun_capture" -> controller.captureSun(aimPanel.ndOd)
            "sun_close" -> exitSunMode()
            "floor" -> enterAimMode(SceneAimKind.FLOOR)
            "floor_capture" -> controller.captureFloor(intent.getStringExtra("note") ?: aimPanel.note)
            "floor_close" -> exitAimMode()
            "resume" -> controller.rotate(CaptureController.RotateMode.RESUME)
            "retake" -> controller.rotate(CaptureController.RotateMode.RETAKE)
            "rotate_new" -> controller.rotate(CaptureController.RotateMode.NEW_SCENE)
            "check" -> controller.quickCheck()
            "auto" -> controller.autoMax()
            "finish" -> controller.finishSession()
            "cancel" -> controller.cancelRotation()
            "max_down" -> controller.maxTimes(0.5)
            "max_up" -> controller.maxTimes(2.0)
            "min_down" -> controller.minTimes(0.5)
            "min_up" -> controller.minTimes(2.0)
            "restart" -> openCamera(AppSettings.load(this))
            "probe" -> controller.probe()
            "sessions", "gallery" -> openSessions()
            "process_scene" -> if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0) {
                runCatching {
                    val root = java.io.File(requireNotNull(intent.getStringExtra("scene"))).canonicalFile
                    require(root.toPath().startsWith(controller.baseDir.canonicalFile.toPath()))
                    check(controller.busyWith == null && controller.sessionPath != root.absolutePath) { "Capture is active" }
                    ProcessingJobs.start(this, root, intent.getStringExtra("mode") ?: "s0")
                }.onFailure { message(it.message ?: "Processing could not start", Level.ERROR) }
            }
            "settings_ui" -> showSettings()
            "fake_drop" -> controller.simulateDrop(intent.getLongExtra("ms", 8000L))
            else -> when {
                intent?.action == UsbManager.ACTION_USB_DEVICE_ATTACHED -> {
                    // A running capture waits for the camera and reopens it itself.
                    if (!controller.handlesReattach()) openCamera(AppSettings.load(this))
                }
                startup -> openCamera(AppSettings.load(this))
            }
        }
    }

    /** Opens the camera, asking for USB permission first when the FLIR needs it. */
    fun openCamera(s: AppSettings) {
        if (s.cameraMode == AppSettings.CAMERA_FLIR) {
            val device = FlirProbe.findDevice(this)
            if (device == null) {
                controller.adopt(s) // the new settings apply now; the camera opens when it is plugged in
                message("FLIR not connected", Level.WARN)
                return
            }
            if (!FlirProbe.hasPermission(this, device)) {
                controller.adopt(s)
                message("Asking for USB permission ...")
                FlirProbe.requestPermission(this, device) { granted ->
                    runOnUiThread { if (granted) controller.start(s) else message("USB permission denied", Level.ERROR) }
                }
                return
            }
        }
        controller.start(s)
    }

    // ---- CaptureController.Ui ----------------------------------------------------------------------

    override fun preview(bitmap: Bitmap, info: CaptureController.PreviewInfo) {
        haveFrame = true
        emptyText.visibility = View.GONE
        previewImage.setImageBitmap(bitmap)
        if (crosshair.visibility == View.VISIBLE) crosshair.invalidate()
        val s = state
        val idx = s?.previewIndex?.let { "${it + 1}/${s.exposures.size}" } ?: ""
        lastOverlay = "LIVE   ${fmtExposure(info.exposureUs)} ($idx)    over %.2f%%    under %.1f%%".format(info.over * 100, info.under * 100)
        if (!showPanorama) overlayText.text = lastOverlay
        if (fullImage != null && !fullPanorama) {
            fullImage?.setImageBitmap(bitmap)
            fullText?.text = lastOverlay
        }
        if (nodalActive) nodalCompare.preview(bitmap, info)
        updateExpand()
    }

    override fun state(s: CaptureController.State) {
        state = s
        val name = android.text.SpannableStringBuilder(s.experiment).append("  ✎")
        name.setSpan(android.text.style.ForegroundColorSpan(Palette.TEXT3), s.experiment.length, name.length, 0)
        name.setSpan(android.text.style.RelativeSizeSpan(0.7f), s.experiment.length, name.length, 0)
        experimentText.text = name
        folderButton.text = folderLabel()
        // Camera drops are not shown; they are counted per session in capture_meta.json "connection".
        val camConnecting = s.connecting && s.cameraLabel.startsWith("Connecting")
        val stageConnecting = s.connecting && s.stageLabel.startsWith("Connecting")
        kit.setChip(camChip, s.cameraLabel, if (camConnecting) Palette.AMBER else if (s.cameraOnline) Palette.GREEN else Palette.RED)
        kit.setChip(stageChip, s.stageLabel, if (stageConnecting) Palette.AMBER else if (s.stageOk) Palette.GREEN else Palette.AMBER)
        // Storage = the base path's drive: green below 70 % used, amber to 90 %, red above (or not connected).
        kit.setChip(
            storageChip,
            when {
                !s.storageOnline -> "Storage offline"
                s.storageUsed < 0 -> "Storage ?"
                else -> "Storage ${Math.round(s.storageUsed * 100)}%"
            },
            when {
                !s.storageOnline || s.storageUsed >= 0.9 -> Palette.RED
                s.storageUsed >= 0.7 -> Palette.AMBER
                s.storageUsed >= 0 -> Palette.GREEN
                else -> Palette.TEXT3
            },
        )
        if (!s.cameraOnline) {
            haveFrame = false
            previewImage.setImageDrawable(null)
            if (!showPanorama) overlayText.text = ""
            updateExpand()
        }
        emptyText.visibility = if (haveFrame) View.GONE else View.VISIBLE
        emptyText.text = if (s.cameraOnline) "Starting the preview ..." else "No camera"

        bracketSummary.text = "%d × %.1f EV  ·  %.1f EV range".format(s.exposures.size, s.stepEv, s.rangeEv)
        renderPills(s)
        checkText.visibility = if (s.check != null) View.VISIBLE else View.GONE
        checkText.text = s.check ?: ""
        checkText.setTextColor(if (s.checkOk == false) Palette.AMBER else Palette.GREEN)

        val open = s.sessionRel != null
        if (!open && havePanorama && !rotating) {
            // the scene is finished: its panorama goes, the live view comes back
            havePanorama = false
            tabs.visibility = View.GONE
            panoImage.setImageDrawable(null)
            setPanorama(false)
        }
        sceneIdText.text = s.sessionRel?.substringAfter("${s.experiment}/")?.replace("/", " ") ?: ""
        kit.enable(finishButton, open && controller.busyWith == null)
        sessionText.text = "${if (open) s.viewsDone else 0}/${s.totalViews} views · CC ${if (open) s.colorCheckers else 0}" +
            (if (open && s.sunShots > 0) " · Sun ${s.sunShots}" else "") +
            (if (open && s.floorShots > 0) " · Floor ${s.floorShots}" else "")
        sessionText.setTextColor(if (open) Palette.TEXT else Palette.TEXT3)
        rotateButton.text = if (s.resumeFrom != null) twoLine("Resume", "from view ${s.resumeFrom + 1} of ${s.totalViews}") else
            twoLine("Rotate", "~${fmtDuration(s.rotationEstimateS)}")
        exposureInfo?.text = bracketText(s)
        renderCalibration()
    }

    override fun busy(what: String?) {
        for (v in lockedWhileBusy) kit.enable(v, what == null)
        kit.enable(finishButton, what == null && state?.sessionRel != null)
        renderCalibration(what)
        // A chip reconnect is a centred popup that blocks the screen until it is done (about 3 s at most).
        if (what == CaptureController.CONNECTING_CAMERA || what == CaptureController.CONNECTING_STAGE) {
            showConnecting(what == CaptureController.CONNECTING_CAMERA)
            showIdle()
            return
        }
        connectDialog?.dismiss()
        connectDialog = null
        if (what == null) {
            rotating = false
            showIdle()
            return
        }
        if (what == CaptureController.STAGE_MOVE) {
            // Aiming mode keeps the controls in place while the stage moves.
            busySub.text = "Moving the stage …"
            busyLine.fraction = null
            return
        }
        busyCancel.visibility = View.VISIBLE
        when {
            what == "rotation" -> {
                rotating = true
                busySub.text = "Rotation  ·  moving to the first view"
                busyLine.fraction = 0.0
            }
            // During a rotation the controller's notes (waiting for the camera, writing) go on the same line.
            rotating -> busySub.text = "Rotation  ·  $what"
            else -> {
                busySub.text = what.replaceFirstChar { it.uppercase() } + " …"
                busyLine.fraction = null
            }
        }
    }

    private var connectDialog: android.app.Dialog? = null

    /** Centred card while a chip reconnect runs: spinner, what is being connected, Cancel. Blocks the screen behind. */
    private fun showConnecting(camera: Boolean) {
        if (connectDialog != null) return
        val s = controller.settings
        val card = kit.col().apply {
            gravity = Gravity.CENTER_HORIZONTAL
            background = kit.rounded(Palette.SURFACE, 26f, Palette.STROKE)
            setPadding(kit.dp(24), kit.dp(28), kit.dp(24), kit.dp(18))
        }
        // spinner inside a soft blue disc
        val ring = FrameLayout(this).apply { background = kit.rounded(0x1F4C8DFF, 999f) }
        ring.addView(Spinner(this), FrameLayout.LayoutParams(kit.dp(40), kit.dp(40), Gravity.CENTER))
        card.addView(ring, kit.lp(kit.dp(72), kit.dp(72)))
        card.addView(kit.text(if (camera) "Connecting the camera" else "Connecting the stage", 18f, bold = true).apply {
            gravity = Gravity.CENTER
        }, kit.lp(kit.match(), kit.wrap(), t = 18))
        val detail = when {
            !camera -> s.mechaUrl.removePrefix("http://")
            s.cameraMode == AppSettings.CAMERA_FAKE -> "Simulated camera"
            else -> "FLIR · USB-C"
        }
        card.addView(kit.text(detail, 13.5f, Palette.TEXT2).apply { gravity = Gravity.CENTER }, kit.lp(kit.match(), kit.wrap(), t = 6))
        card.addView(kit.button("Cancel", Kit.Style.SECONDARY, 14.5f) { controller.cancelRotation() }.apply {
            minHeight = kit.dp(44)
        }, kit.lp(kit.match(), kit.wrap(), t = 22))
        connectDialog = android.app.Dialog(this).apply {
            requestWindowFeature(android.view.Window.FEATURE_NO_TITLE)
            setContentView(card)
            setCancelable(false)
            setCanceledOnTouchOutside(false)
            window?.apply {
                setBackgroundDrawable(android.graphics.drawable.ColorDrawable(0x00000000))
                setLayout(kit.dp(290), android.view.ViewGroup.LayoutParams.WRAP_CONTENT)
                setDimAmount(0.6f)
            }
            show()
        }
    }

    /** Nothing running: the activity line is empty, the bar empty, Cancel hidden but keeping its place (no layout jump). */
    private fun showIdle() {
        busySub.text = ""
        busyLine.fraction = 0.0
        busyCancel.visibility = View.INVISIBLE
    }

    override fun progress(p: CaptureController.Progress?) {
        if (p == null) return
        rotating = true
        busyCancel.visibility = View.VISIBLE
        busySub.text = "yaw %.0f°  pitch %.0f°  ·  ~%s left".format(p.yawDeg, p.pitchDeg, fmtDuration(p.etaS))
        busyLine.fraction = p.done.toDouble() / p.total
    }

    override fun stitch(bitmap: Bitmap?, text: String) {
        if (bitmap != null) {
            panoImage.setImageBitmap(bitmap)
            if (!havePanorama) {
                havePanorama = true
                tabs.visibility = View.VISIBLE
                setPanorama(true)
            }
        }
        if (showPanorama) overlayText.text = "Panorama  ·  $text"
        if (fullImage != null && fullPanorama) {
            if (bitmap != null) fullImage?.setImageBitmap(bitmap)
            fullText?.text = "Panorama  ·  $text"
        }
    }

    override fun message(text: String, level: Level) {
        Log.i("MainActivity", text)
        val sb = android.text.SpannableStringBuilder("● ").append(text)
        sb.setSpan(android.text.style.ForegroundColorSpan(Palette.level(level)), 0, 1, 0)
        messageText.text = sb
        messageText.setTextColor(if (level == Level.ERROR) Palette.RED else Palette.TEXT2)
        if (level == Level.ERROR) Toast.makeText(this, text, Toast.LENGTH_LONG).show()
    }

    // ---- layout -------------------------------------------------------------------------------------

    private fun buildUi(): View {
        val root = kit.col().apply { setBackgroundColor(Palette.BG) }
        kit.padForSystemBars(root)
        val content = kit.col().apply { setPadding(kit.dp(14), kit.dp(6), kit.dp(14), kit.dp(10)) }
        root.addView(content, kit.lp(kit.match(), kit.match()))

        // header: title + sessions / settings
        val header = kit.col()
        val top = kit.row()
        calibrationButton = compact(kit.button("Calibration", Kit.Style.SECONDARY, 13f) {
            if (controller.calibrationMode) leaveCalibration() else enterCalibration()
        })
        top.addView(calibrationButton, kit.lp(0, kit.wrap(), 1f, r = 8))
        lockedWhileBusy += calibrationButton
        val sessions = compact(kit.button("Gallery", Kit.Style.SECONDARY, 13f) { openSessions() })
        val settings = compact(kit.button("Settings", Kit.Style.SECONDARY, 13f) { showSettings() })
        top.addView(sessions, kit.lp(kit.wrap(), kit.wrap(), r = 8))
        top.addView(settings, kit.lp(kit.wrap(), kit.wrap()))
        lockedWhileBusy += settings
        header.addView(top)
        // project (experiment) name: tap to rename; folder button: where sessions go
        val nameRow = kit.row()
        experimentText = kit.text("", 21f, bold = true).apply {
            id = R.id.experiment_name
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.START
            isClickable = true
            setPadding(kit.dp(2), kit.dp(6), kit.dp(8), kit.dp(6))
            background = kit.ripple(kit.rounded(0x00000000, 10f), 10f)
            setAutoSizeTextTypeUniformWithConfiguration(14, 21, 1, android.util.TypedValue.COMPLEX_UNIT_SP)
            setOnClickListener { if (controller.calibrationMode) chooseCalibration() else editExperiment() }
        }
        nameRow.addView(experimentText, kit.lp(0, kit.wrap(), 1f))
        folderButton = compact(kit.button(folderLabel(), Kit.Style.SECONDARY, 13f) { showSaveTo() }).apply {
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.MIDDLE
            maxWidth = kit.dp(170)
        }
        nameRow.addView(folderButton, kit.lp(kit.wrap(), kit.wrap()))
        lockedWhileBusy += listOf(experimentText, folderButton)
        header.addView(nameRow, kit.lp(t = 4))
        content.addView(header, kit.lp(b = 10))

        // status chips
        val chips = kit.row()
        // tap the camera / stage chip to connect again
        camChip = kit.chip("camera", Palette.TEXT3).apply {
            setOnClickListener {
                if (!lockedNow()) controller.reconnectCamera { openCamera(controller.settings) }
            }
        }
        stageChip = kit.chip("stage", Palette.TEXT3).apply {
            setOnClickListener {
                if (!lockedNow()) controller.reconnectStage()
            }
        }
        storageChip = kit.chip("storage", Palette.TEXT3).apply { setOnClickListener { showSaveTo() } }
        for (c in listOf(camChip, stageChip, storageChip)) chips.addView(c, kit.lp(kit.wrap(), kit.wrap(), r = 6))
        val chipScroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            addView(chips)
        }
        content.addView(chipScroll, kit.lp(b = 10))

        // preview / panorama card
        val card = kit.frame().apply {
            id = R.id.preview_card
            background = kit.rounded(0xFF000000.toInt(), 18f, Palette.STROKE)
        }
        kit.clipRounded(card, 18f)
        previewImage = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER }
        // the panorama across the full card width (the live stream is on its own tab)
        panoImage = FitWidthImageView(this).apply { visibility = View.GONE }
        expandButton = ImageView(this).apply {
            setImageResource(R.drawable.ic_fullscreen)
            scaleType = ImageView.ScaleType.CENTER
            background = kit.ripple(kit.rounded(0x99000000.toInt(), 999f, 0x55FFFFFF), 999f)
            contentDescription = "Full screen"
            visibility = View.GONE
            setOnClickListener { showFullScreen() }
        }
        overlayText = kit.text("", 12.5f, 0xFFDDE3EC.toInt(), bold = true).apply {
            setPadding(kit.dp(12), kit.dp(20), kit.dp(56), kit.dp(10))
            background = GradientDrawable(GradientDrawable.Orientation.BOTTOM_TOP, intArrayOf(0xCC000000.toInt(), 0x00000000))
        }
        tabs = kit.row().apply {
            visibility = View.GONE
            background = kit.rounded(0xAA0A0D12.toInt(), 999f)
            setPadding(kit.dp(3), kit.dp(3), kit.dp(3), kit.dp(3))
        }
        liveTab = tab("Live") { setPanorama(false) }
        panoTab = tab("Panorama") { setPanorama(true) }
        tabs.addView(liveTab)
        tabs.addView(panoTab)
        // live view RGB (superpixel S0 colour) / Raw (raw values as grey: 255 = saturated is pure white)
        modeTabs = kit.row().apply {
            visibility = View.GONE
            background = kit.rounded(0xAA0A0D12.toInt(), 999f)
            setPadding(kit.dp(3), kit.dp(3), kit.dp(3), kit.dp(3))
        }
        rgbTab = tab("RGB") { setPreviewMode(AppSettings.PREVIEW_RGB, save = true) }
        rawTab = tab("Raw") { setPreviewMode(AppSettings.PREVIEW_RAW, save = true) }
        modeTabs.addView(rgbTab)
        modeTabs.addView(rawTab)
        emptyText = kit.text("", 14f, Palette.TEXT3).apply {
            gravity = Gravity.CENTER
            setPadding(kit.dp(30), 0, kit.dp(30), 0)
        }
        card.addView(emptyText, FrameLayout.LayoutParams(kit.match(), kit.match()))
        card.addView(previewImage, FrameLayout.LayoutParams(kit.match(), kit.match()))
        card.addView(panoImage, FrameLayout.LayoutParams(kit.match(), kit.match()))
        card.addView(overlayText, FrameLayout.LayoutParams(kit.match(), kit.wrap(), Gravity.BOTTOM))
        card.addView(tabs, FrameLayout.LayoutParams(kit.wrap(), kit.wrap(), Gravity.TOP or Gravity.START).apply { setMargins(kit.dp(10), kit.dp(10), 0, 0) })
        card.addView(modeTabs, FrameLayout.LayoutParams(kit.wrap(), kit.wrap(), Gravity.TOP or Gravity.END).apply { setMargins(0, kit.dp(10), kit.dp(10), 0) })
        card.addView(expandButton, FrameLayout.LayoutParams(kit.dp(38), kit.dp(38), Gravity.BOTTOM or Gravity.END).apply { setMargins(0, 0, kit.dp(10), kit.dp(8)) })
        crosshair = Crosshair(this, previewImage).apply { visibility = View.GONE }
        card.addView(crosshair, FrameLayout.LayoutParams(kit.match(), kit.match()))
        content.addView(card, kit.lp(kit.match(), 0, 1f, b = 10))

        // bracket
        val bracketCard = kit.card(12).apply { id = R.id.bracket_card }
        val bh = kit.row()
        bh.addView(kit.label("Bracket"), kit.lp(0, kit.wrap(), 1f))
        bracketSummary = kit.text("", 12.5f, Palette.TEXT2)
        bh.addView(bracketSummary)
        bracketCard.addView(bh)
        pillsRow = kit.row()
        bracketCard.addView(pillsRow, kit.lp(t = 10))
        // HDR bracket tools, under the bracket they change
        val hdrTools = kit.row()
        for ((i, pair) in listOf<Pair<String, () -> Unit>>(
            "Check" to { controller.quickCheck() },
            "Auto max" to { controller.autoMax() },
            "Exposure" to { showExposure() },
        ).withIndex()) {
            val b = kit.button(pair.first, Kit.Style.SECONDARY, 13f, pair.second).apply {
                maxLines = 1
                minHeight = 0
                minimumHeight = 0
                setPadding(kit.dp(4), kit.dp(7), kit.dp(4), kit.dp(7))
            }
            hdrTools.addView(b, kit.lp(0, kit.wrap(), 1f, r = if (i < 2) 8 else 0))
            lockedWhileBusy += b
        }
        bracketCard.addView(hdrTools, kit.lp(t = 8))
        checkText = kit.text("", 12.5f, Palette.GREEN).apply { visibility = View.GONE }
        bracketCard.addView(checkText, kit.lp(t = 8))
        content.addView(bracketCard, kit.lp(b = 8))

        // scene: which folder, what is in it, what is running now (one card, fixed height)
        val sceneCard = kit.card(12)
        val head = kit.row()
        head.addView(kit.label("Scene"))
        sceneIdText = kit.text("", 12.5f, Palette.TEXT2).apply {
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        head.addView(sceneIdText, kit.lp(0, kit.wrap(), 1f, l = 10))
        sceneCard.addView(head)
        val body = kit.row()
        val texts = kit.col()
        sessionText = kit.text("", 15f, Palette.TEXT, bold = true).apply {
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        busySub = kit.text("", 12.5f, Palette.BLUE).apply {
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        texts.addView(sessionText)
        texts.addView(busySub, kit.lp(t = 4))
        body.addView(texts, kit.lp(0, kit.wrap(), 1f))
        busyCancel = kit.button("Cancel", Kit.Style.DANGER, 14f) { controller.cancelRotation() }.apply { minHeight = kit.dp(42) }
        body.addView(busyCancel, kit.lp(kit.wrap(), kit.wrap(), l = 8))
        sceneCard.addView(body, kit.lp(t = 2))
        busyLine = ProgressLine(this)
        sceneCard.addView(busyLine, kit.lp(kit.match(), kit.dp(6), t = 10))
        sceneCardView = sceneCard
        aimPanel = SceneAimPanel(this, { yaw, pitch -> controller.stageJog(yaw, pitch) },
            { controller.stageHome() }, { exitAimMode() }, { captureAimedShot() }, { controller.cancelRotation() },
            { showNodalCompare() })
        aimPanel.view.visibility = View.GONE
        sceneSlotView = SceneSlot(this, sceneCard, aimPanel.view)
        content.addView(sceneSlotView, kit.lp(b = 8))
        showIdle()

        // always one line tall (user), so a message never moves the buttons; longer ones end in "..."
        messageText = kit.text("", 13f, Palette.TEXT2).apply {
            setSingleLine(true)
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        content.addView(messageText, kit.lp(b = 8, l = 2))

        // primary actions
        val actions = kit.row()
        ccButton = kit.button("Color checker", Kit.Style.TEAL, 14.5f) { enterAimMode(SceneAimKind.COLOR_CHECKER) }.apply {
            minHeight = kit.dp(60)
            setPadding(kit.dp(4), kit.dp(8), kit.dp(4), kit.dp(8))
            // rainbow, red to violet from left to right
            val rainbow = GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                intArrayOf(0xFFE53935.toInt(), 0xFFF57C00.toInt(), 0xFFFBC02D.toInt(), 0xFF43A047.toInt(), 0xFF1E88E5.toInt(), 0xFF3949AB.toInt(), 0xFF8E24AA.toInt()),
            ).apply { cornerRadius = kit.dp(14).toFloat() }
            background = kit.ripple(rainbow, 14f)
            setTextColor(0xFFFFFFFF.toInt())
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setShadowLayer(6f, 0f, 1.5f, 0xCC000000.toInt())
        }
        rotateButton = kit.button("Rotate", Kit.Style.PRIMARY, 16f) { withStorage { onRotate() } }.apply { minHeight = kit.dp(60) }
        sunButton = kit.button("Sun", Kit.Style.SECONDARY, 15f) { enterSunMode() }.apply {
            minHeight = kit.dp(60)
            setPadding(kit.dp(4), kit.dp(8), kit.dp(4), kit.dp(8))
            val sunFill = GradientDrawable(GradientDrawable.Orientation.TOP_BOTTOM, intArrayOf(0xFFFFC107.toInt(), 0xFFFF6F00.toInt()))
                .apply { cornerRadius = kit.dp(14).toFloat() }
            background = kit.ripple(sunFill, 14f)
            setTextColor(0xFFFFFFFF.toInt())
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setShadowLayer(6f, 0f, 1.5f, 0xCC000000.toInt())
        }
        floorButton = kit.button("Floor", Kit.Style.SECONDARY, 14f) { enterAimMode(SceneAimKind.FLOOR) }.apply {
            contentDescription = "Floor supplement"
            setPadding(kit.dp(2), kit.dp(8), kit.dp(2), kit.dp(8))
        }
        // start (Color checker / Rotate) -- end (Finish) of a scene, side by side
        finishButton = kit.button("Finish", Kit.Style.SECONDARY, 15f) { confirmFinish() }.apply {
            minHeight = kit.dp(60)
            setPadding(kit.dp(4), kit.dp(8), kit.dp(4), kit.dp(8))
        }
        // Keep the existing preview/bracket positions and the same 64 dp action-row height.
        actions.addView(ccButton, kit.lp(0, kit.dp(64), 1.15f, r = 6))
        actions.addView(sunButton, kit.lp(0, kit.dp(64), 0.65f, r = 6))
        actions.addView(rotateButton, kit.lp(0, kit.dp(64), 1.2f, r = 6))
        actions.addView(floorButton, kit.lp(0, kit.dp(64), 0.8f, r = 6))
        actions.addView(finishButton, kit.lp(0, kit.dp(64), 0.85f))
        lockedWhileBusy += listOf(ccButton, sunButton, rotateButton, floorButton)
        content.addView(actions, kit.lp(b = 6))
        actionsRow = actions
        aimPanel.actions.visibility = View.GONE
        content.addView(aimPanel.actions, kit.lp(b = 6))
        return root
    }

    /** The top mode button enters directly; the experiment-name row selects the reference type. */
    private fun enterCalibration() {
        if (lockedNow()) return
        if (controller.sessionPath != null) {
            message("Finish the open scene before calibration", Level.WARN)
            return
        }
        if (aimKind != null) {
            exitAimMode { enterCalibration() }
            return
        }
        controller.beginCalibration()
        message("Tap the Calibration title to choose a type")
    }

    private fun chooseCalibration() {
        if (lockedNow() || !controller.calibrationMode) return
        if (controller.sessionPath != null) {
            message("Finish the current calibration before choosing another type", Level.WARN)
            return
        }
        val kinds = CalibrationKind.entries
        dialog().setTitle("Calibration type")
            .setSingleChoiceItems(kinds.map { it.label }.toTypedArray(), kinds.indexOf(controller.calibrationKind)) { d, i ->
                d.dismiss()
                val kind = kinds[i]
                if (kind != controller.calibrationKind) {
                    afterReferenceNd { controller.beginCalibration(kind) }
                }
            }.setNegativeButton("Cancel", null).show()
    }

    private fun afterReferenceNd(next: () -> Unit) {
        if (controller.calibrationKind != CalibrationKind.ND || aimPanel.ndOd == null) {
            next()
            return
        }
        dialog().setTitle("Remove the ND filter")
            .setMessage("Remove the ND filter from the lens before leaving ND calibration.")
            .setPositiveButton("Removed") { _, _ -> next() }
            .setNegativeButton("Stay", null).show()
    }

    private fun leaveCalibration() {
        if (lockedNow()) return
        if (controller.sessionPath != null) {
            message("Finish the current calibration before returning to Scene", Level.WARN)
            return
        }
        afterReferenceNd { controller.leaveCalibration() }
    }

    private fun finishCalibration() {
        if (lockedNow() || !controller.calibrationMode || controller.calibrationKind == null) return
        if ((state?.calibrationShots ?: 0) == 0) {
            controller.finishCalibration()
        } else {
            showNotes(calibration = true)
        }
    }

    private fun renderCalibration(work: String? = controller.busyWith) {
        val s = state ?: return
        val reference = s.calibrationMode
        kit.enable(floorButton, work == null && !reference && s.sessionRel != null && s.viewsDone > 0 && s.resumeFrom == null)
        val nodal = reference && s.calibrationKind == CalibrationKind.NODAL
        if (!nodal && nodalActive) nodalCompare.reset()
        nodalActive = nodal
        if (nodal) nodalCompare.updateState(s, work)
        calibrationButton.text = if (reference) "Scene" else "Calibration"
        if (reference) {
            aimKind = null
            setPanorama(false)
            experimentText.text = (s.calibrationKind?.label ?: "Calibration") + "  \u25be"
            experimentText.contentDescription = "Calibration type"
            aimPanel.renderCalibration(s, work)
            kit.enable(folderButton, work == null && s.sessionRel == null)
        } else {
            experimentText.contentDescription = "Experiment name"
            aimKind?.let { aimPanel.render(it, s, work) }
        }
        val aiming = reference || aimKind != null
        sceneSlotView.visibility = View.VISIBLE
        actionsRow.visibility = if (aiming) View.GONE else View.VISIBLE
        // The invisible scene card keeps the preview and bracket geometry identical in every mode.
        sceneCardView.visibility = if (aiming) View.INVISIBLE else View.VISIBLE
        aimPanel.view.visibility = if (aiming) View.VISIBLE else View.GONE
        aimPanel.actions.visibility = if (aiming) View.VISIBLE else View.GONE
        val showCrosshair = sunMode || (reference && s.calibrationKind == CalibrationKind.ND)
        if (showCrosshair) {
            val (fx, fy) = controller.principalPointInPreview()
            crosshair.fx = fx
            crosshair.fy = fy
        }
        crosshair.visibility = if (showCrosshair) View.VISIBLE else View.GONE
        kit.enable(experimentText, work == null)
    }

    /** Entering an aiming mode changes only the scene card and bottom actions, without moving the stage. */
    private fun enterAimMode(kind: SceneAimKind) {
        if (lockedNow()) return
        if (controller.calibrationMode) {
            message("Finish calibration before taking a scene shot", Level.WARN)
            return
        }
        if (kind == SceneAimKind.FLOOR && (state?.sessionRel == null || (state?.viewsDone ?: 0) == 0 || state?.resumeFrom != null)) {
            message("Finish the rotation first; take floor shots before Finish", Level.WARN)
            return
        }
        if (sunMode && kind != SceneAimKind.SUN) {
            exitAimMode { enterAimMode(kind) }
            return
        }
        aimKind = kind
        setPanorama(false)
        if (kind == SceneAimKind.SUN) {
            val (fx, fy) = controller.principalPointInPreview()
            crosshair.fx = fx
            crosshair.fy = fy
            message("ND on the lens; centre the sun, then Capture sun")
        } else if (kind == SceneAimKind.FLOOR) {
            message("No ND; move rig aside and aim at its old footprint. Note = details")
        } else {
            message("Aim at the color checker, then Capture checker")
        }
        renderCalibration()
    }

    private fun enterSunMode() = enterAimMode(SceneAimKind.SUN)

    private fun captureAimedShot() {
        if (lockedNow()) return
        withStorage {
            if (controller.calibrationMode) {
                controller.calibrationKind?.let { kind ->
                    controller.captureCalibration(CalibrationShot(kind, aimPanel.note, aimPanel.ndOd))
                }
            } else when (aimKind) {
                SceneAimKind.COLOR_CHECKER -> controller.captureColorChecker()
                SceneAimKind.SUN -> controller.captureSun(aimPanel.ndOd)
                SceneAimKind.FLOOR -> controller.captureFloor(aimPanel.note)
                null -> Unit
            }
        }
    }

    /** Close returns to the same open scene; only Finish closes the scene itself. */
    private fun exitAimMode(after: (() -> Unit)? = null) {
        if (lockedNow()) return
        if (controller.calibrationMode) {
            finishCalibration()
            return
        }
        val leave = {
            val wasFloor = aimKind == SceneAimKind.FLOOR
            aimKind = null
            renderCalibration()
            if (wasFloor) message("Return rig to its original position before other scene shots")
            after?.invoke()
        }
        if (!sunMode || aimPanel.ndOd == null) {
            leave()
            return
        }
        dialog().setTitle("Remove the ND filter")
            .setMessage("Take the ND filter off the lens before color checker, rotation or floor shots.")
            .setPositiveButton("Removed") { _, _ -> leave() }
            .setNegativeButton("Stay", null)
            .show()
    }

    private fun exitSunMode() = exitAimMode()

    /** Big label over a small caption, for the two main buttons. */
    private fun twoLine(title: String, caption: String, captionColor: Int = 0xCCFFFFFF.toInt()): CharSequence {
        val sb = android.text.SpannableStringBuilder(title).append("\n").append(caption)
        sb.setSpan(android.text.style.RelativeSizeSpan(0.72f), title.length + 1, sb.length, 0)
        sb.setSpan(android.text.style.ForegroundColorSpan(captionColor), title.length + 1, sb.length, 0)
        return sb
    }

    private fun compact(v: TextView) = v.apply {
        minHeight = kit.dp(38)
        setPadding(kit.dp(12), kit.dp(6), kit.dp(12), kit.dp(6))
    }

    private fun tab(label: String, onClick: () -> Unit) = kit.text(label, 12.5f, Palette.TEXT2, bold = true).apply {
        setPadding(kit.dp(12), kit.dp(6), kit.dp(12), kit.dp(6))
        isClickable = true
        setOnClickListener { onClick() }
    }

    private fun setPanorama(on: Boolean) {
        showPanorama = on && havePanorama
        panoImage.visibility = if (showPanorama) View.VISIBLE else View.GONE
        previewImage.visibility = if (showPanorama) View.GONE else View.VISIBLE
        for ((t, sel) in listOf(liveTab to !showPanorama, panoTab to showPanorama)) {
            t.background = if (sel) kit.rounded(Palette.BLUE, 999f) else null
            t.setTextColor(if (sel) 0xFFFFFFFF.toInt() else Palette.TEXT2)
        }
        overlayText.text = if (showPanorama) "Panorama" else lastOverlay
        updateExpand()
    }

    private fun updateExpand() {
        expandButton.visibility = if (showPanorama || haveFrame) View.VISIBLE else View.GONE
        modeTabs.visibility = if (haveFrame && !showPanorama) View.VISIBLE else View.GONE
    }

    /** Live view RGB / Raw (also in Settings > Preview); applies from the next frame. */
    private fun setPreviewMode(mode: String, save: Boolean) {
        if (save) controller.setPreviewMode(mode)
        for ((t, sel) in listOf(rgbTab to (mode == AppSettings.PREVIEW_RGB), rawTab to (mode == AppSettings.PREVIEW_RAW))) {
            t.background = if (sel) kit.rounded(Palette.BLUE, 999f) else null
            t.setTextColor(if (sel) 0xFFFFFFFF.toInt() else Palette.TEXT2)
        }
    }

    /**
     * Full screen of what the card shows -- the live stream or the panorama -- updated as frames come in.
     * A panorama is turned sideways to fill the portrait screen (turn the phone); Rotate switches.
     */
    private fun showFullScreen() {
        if (nodalActive && !showPanorama) { showNodalCompare(); return }
        val pano = showPanorama
        val bmp = ((if (pano) panoImage else previewImage).drawable as? android.graphics.drawable.BitmapDrawable)?.bitmap ?: return
        val d = android.app.Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen)
        val root = FrameLayout(this).apply { setBackgroundColor(0xFF000000.toInt()) }
        kit.padForSystemBars(root)
        val stage = FrameLayout(this)
        val img = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            setImageBitmap(bmp)
        }
        val text = kit.text(overlayText.text, 12.5f, 0xFFDDE3EC.toInt(), bold = true).apply {
            setPadding(kit.dp(12), kit.dp(20), kit.dp(12), kit.dp(10))
            background = GradientDrawable(GradientDrawable.Orientation.BOTTOM_TOP, intArrayOf(0xCC000000.toInt(), 0x00000000))
        }
        stage.addView(img, FrameLayout.LayoutParams(kit.match(), kit.match()))
        stage.addView(text, FrameLayout.LayoutParams(kit.match(), kit.wrap(), Gravity.BOTTOM))
        root.addView(stage, FrameLayout.LayoutParams(kit.match(), kit.match(), Gravity.CENTER))
        var turned = pano
        fun place() {
            val w = root.width - root.paddingLeft - root.paddingRight
            val h = root.height - root.paddingTop - root.paddingBottom
            if (w <= 0 || h <= 0) return
            val lp = stage.layoutParams as FrameLayout.LayoutParams
            lp.width = if (turned) h else w
            lp.height = if (turned) w else h
            stage.layoutParams = lp
            stage.rotation = if (turned) 90f else 0f
        }
        root.addOnLayoutChangeListener { _, l, t, r, b, oldL, oldT, oldR, oldB ->
            if (r - l != oldR - oldL || b - t != oldB - oldT) root.post { place() }
        }
        val bar = kit.row().apply { setPadding(kit.dp(6), kit.dp(6), kit.dp(6), kit.dp(6)) }
        bar.addView(kit.button("Rotate", Kit.Style.SECONDARY, 13.5f) {
            turned = !turned
            place()
        }, kit.lp(kit.wrap(), kit.wrap(), r = 8))
        bar.addView(kit.button("✕", Kit.Style.SECONDARY, 15f) { d.dismiss() })
        root.addView(bar, FrameLayout.LayoutParams(kit.wrap(), kit.wrap(), Gravity.TOP or Gravity.END))
        d.setContentView(root)
        d.window?.setLayout(android.view.ViewGroup.LayoutParams.MATCH_PARENT, android.view.ViewGroup.LayoutParams.MATCH_PARENT)
        fullImage = img
        fullText = text
        fullPanorama = pano
        d.setOnDismissListener {
            fullImage = null
            fullText = null
        }
        d.show()
    }

    private fun showNodalCompare() {
        if (!nodalActive || lockedNow()) return
        nodalCompare.show()
    }

    private fun renderPills(s: CaptureController.State) {
        if (pillsRow.childCount != s.exposures.size) {
            pillsRow.removeAllViews()
            for (i in s.exposures.indices) {
                val p = kit.text("", 11.5f, Palette.TEXT2, bold = true).apply {
                    gravity = Gravity.CENTER
                    setPadding(0, kit.dp(8), 0, kit.dp(8))
                    isClickable = true
                    setOnClickListener { controller.previewAt(i) }
                }
                pillsRow.addView(p, kit.lp(0, kit.wrap(), 1f, r = if (i < s.exposures.size - 1) 4 else 0))
            }
        }
        for (i in s.exposures.indices) {
            val p = pillsRow.getChildAt(i) as TextView
            p.text = fmtExposure(s.exposures[i])
            val sel = i == s.previewIndex
            p.background = kit.rounded(if (sel) Palette.BLUE else Palette.SURFACE2, 10f)
            p.setTextColor(if (sel) 0xFFFFFFFF.toInt() else Palette.TEXT2)
        }
    }

    // ---- actions ------------------------------------------------------------------------------------

    private fun dialog() = AlertDialog.Builder(this, android.R.style.Theme_Material_Dialog_Alert)

    private fun onRotate() {
        val s = state ?: return
        if (s.resumeFrom != null) {
            dialog()
                .setTitle("Resume the rotation?")
                .setMessage("It stopped before view ${s.resumeFrom + 1} of ${s.totalViews}.\n\nResume continues from there. Retake deletes the views taken so far and starts again from view 1 (in the same scene).")
                .setPositiveButton("Resume") { _, _ -> controller.rotate(CaptureController.RotateMode.RESUME) }
                .setNeutralButton("Retake") { _, _ -> controller.rotate(CaptureController.RotateMode.RETAKE) }
                .setNegativeButton("Cancel", null)
                .show()
            return
        }
        if (s.sessionRel != null && s.viewsDone > 0) {
            // A rotation is already in this open scene (Finish was not pressed): never overwrite it silently.
            dialog()
                .setTitle("This scene already has a rotation")
                .setMessage(
                    "${s.viewsDone} views in ${s.sessionRel.substringAfter("${s.experiment}/").replace('/', ' ')}.\n\n" +
                        "Retake: delete them and rotate again in the same scene (color checker shots stay).\n" +
                        "New scene: finish this scene and rotate in a new one.",
                )
                .setPositiveButton("New scene") { _, _ -> controller.rotate(CaptureController.RotateMode.NEW_SCENE) }
                .setNeutralButton("Retake") { _, _ -> controller.rotate(CaptureController.RotateMode.RETAKE) }
                .setNegativeButton("Cancel", null)
                .show()
            return
        }
        dialog()
            .setTitle("Start the rotation?")
            .setMessage(
                "${s.totalViews} views, ~${fmtDuration(s.rotationEstimateS)}. The scene stays open afterwards " +
                    "(Rotate again to retake it); Finish ends it.\n\n" +
                    "Bracket: ${s.exposures.joinToString("  ") { fmtExposure(it) }}",
            )
            .setPositiveButton("Rotate") { _, _ -> controller.rotate() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /** Finish asks for the scene's notes (all optional) and ends the scene with them. */
    private fun confirmFinish() = showNotes()

    private fun bracketText(s: CaptureController.State) =
        "${s.exposures.joinToString("  ") { fmtExposure(it) }}\n%d frames · %.2f EV apart · %.1f EV range · preview %s".format(
            s.exposures.size, s.stepEv, s.rangeEv, fmtExposure(s.exposures[s.previewIndex]),
        )

    /** Bracket adjustments ([ / ] / u / d, shortest exposure, frames) in one sheet. */
    private fun showExposure() {
        val s = state ?: return
        val box = kit.col().apply { setPadding(kit.dp(18), kit.dp(10), kit.dp(18), kit.dp(6)) }
        val info = kit.text(bracketText(s), 13f, Palette.TEXT2, mono = true)
        exposureInfo = info
        box.addView(info, kit.lp(b = 14))
        fun line(title: String, a: Pair<String, () -> Unit>, b: Pair<String, () -> Unit>) {
            val r = kit.row()
            r.addView(kit.text(title, 14.5f), kit.lp(0, kit.wrap(), 1f))
            r.addView(kit.button(a.first, Kit.Style.SECONDARY, 14f, a.second).apply { minHeight = kit.dp(44) }, kit.lp(kit.dp(84), kit.wrap(), r = 8))
            r.addView(kit.button(b.first, Kit.Style.SECONDARY, 14f, b.second).apply { minHeight = kit.dp(44) }, kit.lp(kit.dp(84), kit.wrap()))
            box.addView(r, kit.lp(b = 8))
        }
        line("Longest  [ ]", "÷2" to { controller.maxTimes(0.5) }, "×2" to { controller.maxTimes(2.0) })
        line("Shortest", "÷2" to { controller.minTimes(0.5) }, "×2" to { controller.minTimes(2.0) })
        line("Frames", "−1" to { controller.framesPlus(-1) }, "+1" to { controller.framesPlus(1) })
        line("Preview  d / u", "darker" to { controller.previewDown() }, "brighter" to { controller.previewUp() })
        box.addView(kit.text("The dataset preset is 21 µs – 344 ms × 8 (config/capture/flir.yaml); keep it fixed for every scene.", 12f, Palette.TEXT3), kit.lp(t = 4))
        dialog().setTitle("Exposure").setView(box).setPositiveButton("Done", null)
            .setOnDismissListener { exposureInfo = null }.show()
    }

    private fun lockedNow(): Boolean {
        if (controller.busyWith == null) return false
        message("Wait until ${controller.busyWith} finishes", Level.WARN)
        return true
    }

    /** Project (experiment) name, edited in place: the folder the next sessions go to. */
    private fun editExperiment() {
        if (lockedNow()) return
        if (controller.calibrationMode) return
        val current = controller.settings.experimentName
        val box = kit.col().apply { setPadding(kit.dp(18), kit.dp(8), kit.dp(18), kit.dp(4)) }
        box.addView(kit.text("Scenes go to <folder>/<project>/YYYYMMDD/HHMMSS. Use / for sub folders, e.g. FLIR/outdoor/POSTECH.", 12.5f, Palette.TEXT2))
        val field = kit.field(current, hint = "FLIR/outdoor/POSTECH")
        box.addView(field, kit.lp(t = 10))
        val recent = Storage.roots(this, controller.settings).flatMap { (_, b) -> SessionIndex.list(b) }
            .sortedByDescending { "${it.date}/${it.time}" }.map { it.experiment }.filter { it.isNotEmpty() && it != current }.distinct().take(6)
        if (recent.isNotEmpty()) {
            box.addView(kit.label("Recent"), kit.lp(t = 16, b = 4))
            for (r in recent) {
                box.addView(kit.text(r, 15f).apply {
                    setPadding(kit.dp(4), kit.dp(10), kit.dp(4), kit.dp(10))
                    isClickable = true
                    background = kit.ripple(kit.rounded(0x00000000, 8f), 8f)
                    setOnClickListener {
                        field.setText(r)
                        field.setSelection(r.length)
                    }
                })
            }
        }
        if (state?.sessionRel != null) {
            box.addView(kit.text("The open scene stays in $current; the new name starts with the next scene.", 12.5f, Palette.AMBER), kit.lp(t = 12))
        }
        val d = dialog().setTitle("Project name").setView(android.widget.ScrollView(this).apply { addView(box) })
            .setPositiveButton("Save") { _, _ ->
                val n = field.text.toString().trim().trim('/').replace(Regex("/+"), "/")
                when {
                    n.isEmpty() -> message("The name cannot be empty", Level.WARN)
                    n.split('/').any { it == ".." || it == "." } || n.any { it in "\\:*?\"<>|" } ->
                        message("Use letters, numbers, - _ and / only", Level.WARN)
                    n == current -> Unit
                    else -> {
                        controller.updateSettings(controller.settings.copy(experimentName = n))
                        message("Project: $n", Level.OK)
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .create()
        d.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_VISIBLE)
        d.show()
        field.requestFocus()
        field.setSelection(field.text.length)
    }

    /**
     * The folder button: the base path -- any folder from the file explorer (phone or USB drive), or the quick
     * choices Phone (Documents/penvmap) and SSD (<drive>/penvmap/data/dataset_FLIR). Nothing changes until Apply.
     */
    private fun showSaveTo() {
        if (controller.calibrationMode && controller.sessionPath != null) {
            message("Finish calibration before changing its storage folder", Level.WARN)
            return
        }
        if (lockedNow()) return
        pickBasePath("Base path", controller.baseDir, "Apply", forCapture = true) { dir ->
            val phoneDir = Storage.baseDir(this, Storage.DOCUMENTS)
            val ns = if (dir.absolutePath == phoneDir.absolutePath) controller.settings.copy(storageMode = Storage.DOCUMENTS)
            else controller.settings.copy(storageMode = Storage.CUSTOM, customBasePath = dir.absolutePath)
            controller.updateSettings(ns)
            val where = Storage.readable(this, dir.absolutePath)
            if (Storage.available(this, dir)) message("Saving to $where", Level.OK)
            else message("Base path not connected -- plug the drive in", Level.WARN)
        }
    }

    /**
     * The base-path picker (main screen: where scenes are saved; Gallery: which folder to show): the folder now,
     * Choose folder… (file explorer, phone or USB drive) and the quick Phone | SSD choices. [onApply] gets the
     * folder; nothing changes before that.
     */
    /** Header folder button: the base path's last folder name (the full path is in the Base path dialog). */
    private fun folderLabel(): String {
        val name = if (::controller.isInitialized) controller.baseDir.name.ifEmpty { "/" } else "…"
        return "📁 $name ▾"
    }

    fun pickBasePath(title: String, start: java.io.File, applyLabel: String, forCapture: Boolean, onApply: (java.io.File) -> Unit) {
        var chosen = start
        val box = kit.col().apply { setPadding(kit.dp(18), kit.dp(8), kit.dp(18), kit.dp(4)) }
        val phoneDir = Storage.baseDir(this, Storage.DOCUMENTS)
        val pathText = kit.text("", 14f, Palette.TEXT, mono = true)
        val statusText = kit.text("", 12.5f, Palette.TEXT2)
        val current = kit.card(12, Palette.SURFACE2)
        current.addView(pathText)
        current.addView(statusText, kit.lp(t = 4))
        box.addView(current, kit.lp(t = 4))
        val browse = kit.button("📂  Choose folder…", Kit.Style.SECONDARY, 14f) {}.apply { minHeight = kit.dp(44) }
        box.addView(browse, kit.lp(t = 8))
        val quick = kit.col()
        box.addView(quick, kit.lp(t = 8))
        val layout = kit.text("", 12f, Palette.TEXT3)
        box.addView(layout, kit.lp(t = 8))

        fun paint() {
            val online = Storage.available(this, chosen)
            pathText.text = (if (Storage.isPhone(chosen)) "📁 " else "💾 ") + Storage.readable(this, chosen.absolutePath)
            statusText.text = if (!online) (Storage.offlineReason(this, chosen) ?: "not connected") else
                Storage.usage(chosen)?.let { (free, used) -> "${SessionIndex.humanBytes(free)} free  ·  ${Math.round(used * 100)}% used" } ?: ""
            statusText.setTextColor(if (online) Palette.TEXT2 else Palette.RED)
            layout.text = "Scenes: ${Storage.readable(this, chosen.absolutePath)}/<project>/YYYYMMDD/HHMMSS"
            val sel = when {
                chosen.absolutePath == phoneDir.absolutePath -> 0
                Storage.isSsdDefault(chosen) -> 1
                else -> -1
            }
            quick.removeAllViews()
            quick.addView(kit.segmented(listOf("Phone", "SSD"), sel) { i ->
                if (i == 0) {
                    chosen = phoneDir
                    paint()
                } else {
                    chooseSsd(onCancel = { paint() }, needWrite = forCapture) { dir ->
                        chosen = dir
                        paint()
                    }
                }
            })
        }
        browse.setOnClickListener {
            chooseBaseFolder(needWrite = forCapture) { dir ->
                chosen = dir
                paint()
            }
        }
        paint()

        val d = dialog().setTitle(title).setView(android.widget.ScrollView(this).apply { addView(box) })
            .setPositiveButton(applyLabel) { _, _ -> onApply(chosen) }
            .setNegativeButton("Cancel", null)
        if (forCapture) d.setNeutralButton("Details") { _, _ -> showStorage() }
        d.show()
    }

    /**
     * Runs [next] with "All files access" (Android writes a USB drive as plain files only with it): asks for it
     * first and continues when the user is back from the settings screen. [onCancel] when it stays off.
     */
    private fun withAllFilesAccess(onCancel: () -> Unit = {}, next: () -> Unit) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            message("A USB drive as the base path needs Android 11 or later", Level.WARN)
            onCancel()
            return
        }
        if (Storage.hasAllFilesAccess()) {
            next()
            return
        }
        dialog().setTitle("Allow all files access")
            .setMessage("Android lets the app write to a USB drive only with \"All files access\". Turn it on for Penvmap Capture on the next screen, then come back.")
            .setPositiveButton("Open settings") { _, _ ->
                afterAllFilesAccess = next to onCancel
                startActivity(Intent(android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:$packageName")))
            }
            .setNegativeButton("Cancel") { _, _ -> onCancel() }
            .setOnCancelListener { onCancel() }
            .show()
    }

    /** Any folder as the base path (phone or USB drive): the system file explorer, then a test write there. */
    private fun chooseBaseFolder(needWrite: Boolean = true, onChosen: (java.io.File) -> Unit) = withAllFilesAccess {
        pickFolder { uri ->
            if (uri == null) return@pickFolder
            val dir = Storage.pathFromTree(this, uri)
            if (dir == null) {
                message("Pick a folder on the phone, an SD card or a USB drive", Level.ERROR)
                return@pickFolder
            }
            if (!needWrite || writable(dir)) onChosen(dir)
        }
    }

    /** The SSD quick choice: <connected drive>/penvmap/data/dataset_FLIR (several drives: the user picks one). */
    private fun chooseSsd(onCancel: () -> Unit, needWrite: Boolean = true, onChosen: (java.io.File) -> Unit) = withAllFilesAccess(onCancel) {
        val drives = Storage.usbDrives(this)
        fun take(root: java.io.File) {
            val dir = java.io.File(root, Storage.SSD_DIR)
            if (!needWrite || writable(dir)) onChosen(dir) else onCancel()
        }
        when (drives.size) {
            0 -> {
                // shown over the Base path dialog too
                Toast.makeText(this, "No USB drive connected -- plug the SSD in", Toast.LENGTH_LONG).show()
                message("No USB drive connected -- plug the SSD in", Level.WARN)
                onCancel()
            }
            1 -> take(drives[0].second)
            else -> dialog().setTitle("Which drive?")
                .setItems(drives.map { "${it.first}  (${it.second.absolutePath})" }.toTypedArray()) { _, which -> take(drives[which].second) }
                .setOnCancelListener { onCancel() }
                .show()
        }
    }

    /** Test write to a new base path; says why not when it fails. */
    private fun writable(dir: java.io.File): Boolean {
        val err = Storage.probeWrite(dir) ?: return true
        dialog().setTitle("Can't write there")
            .setMessage("${Storage.readable(this, dir.absolutePath)}\n\n$err")
            .setPositiveButton("OK", null)
            .show()
        return false
    }

    /** Captures write to the base path: when its drive is not connected, ask before using the phone instead. */
    private fun withStorage(action: () -> Unit) {
        val s = state
        if (s == null || s.storageOnline) {
            action()
            return
        }
        dialog().setTitle("${s.storageName} not connected")
            .setMessage("${Storage.readable(this, s.storagePath)} is not connected. Plug the drive in, or save to the phone (Documents/penvmap) instead.")
            .setPositiveButton("Save to phone") { _, _ ->
                controller.updateSettings(controller.settings.copy(storageMode = Storage.DOCUMENTS))
                action()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /** Where the files are: this scene, the base paths, and what the Gallery reads. */
    private fun showStorage() {
        val s = state ?: return
        val box = kit.col().apply { setPadding(kit.dp(18), kit.dp(10), kit.dp(18), kit.dp(6)) }
        fun block(title: String, body: String, mono: Boolean = false, color: Int = Palette.TEXT2) {
            box.addView(kit.label(title), kit.lp(t = 4))
            box.addView(kit.text(body, if (mono) 12f else 13.5f, color, mono = mono), kit.lp(t = 5, b = 12))
        }
        val now = Storage.readable(this, s.storagePath) + "/" + (s.sessionRel ?: "${s.experiment}/YYYYMMDD/HHMMSS  (next shot)") +
            if (s.storageOnline) "" else "\n(not connected)"
        block("This scene", now, mono = true, color = if (s.storageOnline) Palette.TEXT else Palette.RED)
        val cfg = controller.settings
        val roots = Storage.roots(this, cfg).mapNotNull { (mode, base) ->
            val list = SessionIndex.list(base)
            val head = (if (mode == Storage.CUSTOM) "SSD / USB drive" else "Phone") + if (base.absolutePath == s.storagePath) "  (base path)" else ""
            "$head\n${Storage.readable(this, base.absolutePath)}\n${list.size} scenes · ${SessionIndex.humanBytes(list.sumOf { it.bytes })}"
        }.toMutableList()
        cfg.customBasePath?.takeIf { !Storage.available(this, java.io.File(it)) }?.let {
            roots.add(0, "SSD / USB drive\n${Storage.readable(this, it)}\n${Storage.offlineReason(this, java.io.File(it))}")
        }
        block("Folders", roots.joinToString("\n\n"), mono = true)
        block("Drives", Storage.mountedDrives(this).ifEmpty { "no USB drive / SD card seen by Android" } +
            "\nAll files access: " + (if (Storage.hasAllFilesAccess()) "on" else "OFF"), mono = true)
        block("Gallery", "Lists the captures in the folders above, including older sessions. " +
            "The Android Gallery app does not show these files.")
        block("On the PC", "Phone: python app/tools/pull_sessions.py --pull --verify\n  or USB file transfer: Phone > Documents > penvmap\n" +
            "SSD: plug it into the PC; its penvmap folder is the base_dir", mono = true)
        dialog().setTitle("Where scenes are saved").setView(android.widget.ScrollView(this).apply { addView(box) })
            .setPositiveButton("Gallery") { _, _ -> openSessions() }
            .setNegativeButton("Close", null)
            .show()
    }

    fun applySettings(ns: AppSettings) {
        AppSettings.save(this, ns)
        setPreviewMode(ns.previewMode, save = false)
        openCamera(ns)
    }

    private fun openSessions() = SessionsScreen(this, controller).show()

    private fun showSettings() {
        if (controller.busyWith != null) {
            message("Settings are locked while ${controller.busyWith} runs", Level.WARN)
            return
        }
        openSettingsScreen()
    }

    private fun openSettingsScreen() = SettingsScreen(this, controller.settings) { ns ->
        applySettings(ns)
    }.show()

    /** Weather of the last finished scene: the next one is usually at the same place and time. */
    private var lastWeather = ""

    /** Optional session annotations; calibration returns to type selection after Finish. */
    private fun showNotes(calibration: Boolean = false) {
        val kind = if (calibration) controller.calibrationKind ?: return else null
        val a = controller.annotations
        val box = kit.col().apply { setPadding(kit.dp(18), kit.dp(8), kit.dp(18), kit.dp(4)) }
        val description = if (calibration) "Saved with this calibration capture group; all optional." else
            "Saved with the scene (capture_meta.json); all optional."
        box.addView(kit.text(description, 12.5f, Palette.TEXT3), kit.lp(b = 12))
        if (kind == CalibrationKind.ND && aimPanel.ndOd != null) {
            box.addView(kit.text("Remove the ND filter from the lens when finished.", 12.5f, Palette.AMBER), kit.lp(b = 12))
        }
        box.addView(kit.label("Weather"))
        val weathers = listOf("", "clear", "partly cloudy", "cloudy", "overcast", "rain", "fog", "snow", "night", "indoor")
        var weather = a["weather"] ?: if (calibration) "" else lastWeather
        val wrapRow = android.widget.GridLayout(this).apply { columnCount = 3 }
        val weatherChips = ArrayList<TextView>()
        fun paintWeather() {
            for (c in weatherChips) {
                val sel = c.tag == weather
                c.background = kit.rounded(if (sel) Palette.BLUE else Palette.SURFACE2, 999f)
                c.setTextColor(if (sel) 0xFFFFFFFF.toInt() else Palette.TEXT2)
            }
        }
        for (w in weathers.drop(1)) {
            val c = kit.text(w, 13f, Palette.TEXT2, bold = true).apply {
                tag = w
                gravity = Gravity.CENTER
                setPadding(kit.dp(10), kit.dp(7), kit.dp(10), kit.dp(7))
                setOnClickListener {
                    weather = if (weather == w) "" else w
                    paintWeather()
                }
            }
            weatherChips += c
            wrapRow.addView(c, android.widget.GridLayout.LayoutParams().apply {
                setMargins(kit.dp(3), kit.dp(3), kit.dp(3), kit.dp(3))
                width = 0
                columnSpec = android.widget.GridLayout.spec(android.widget.GridLayout.UNDEFINED, 1f)
            })
        }
        paintWeather()
        box.addView(wrapRow, kit.lp(t = 6, b = 12))
        box.addView(kit.label("Content"))
        val content = kit.field(a["content"] ?: "", hint = if (calibration)
            "target, lighting, aperture ..." else "building, sky, vegetation, water ...")
        box.addView(content, kit.lp(t = 6, b = 12))
        box.addView(kit.label("Notes"))
        val notes = kit.field(a["notes"] ?: "", hint = "anything worth remembering").apply {
            isSingleLine = false
            minLines = 2
        }
        box.addView(notes, kit.lp(t = 6))
        val scene = state?.sessionRel?.substringAfter("${state?.experiment}/")?.replace('/', ' ') ?: ""
        val title = if (calibration) "Finish ${kind!!.label}" else "Finish scene $scene"
        dialog().setTitle(title).setView(android.widget.ScrollView(this).apply { addView(box) })
            .setPositiveButton("Finish") { _, _ ->
                val annotations = linkedMapOf(
                    "weather" to weather,
                    "content" to content.text.toString().trim(),
                    "notes" to notes.text.toString().trim(),
                ).filterValues { it.isNotEmpty() }
                if (calibration) {
                    controller.finishCalibration(annotations)
                } else {
                    lastWeather = weather
                    controller.finishSession(annotations)
                    message("Scene $scene finished", Level.OK)
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    companion object {
        private const val REQ_FOLDER = 41

        fun fmtExposure(us: Long): String = when {
            us < 1000 -> "${us}µs"
            us < 10_000 -> "%.1fms".format(us / 1000.0)
            us < 1_000_000 -> "%.0fms".format(us / 1000.0)
            else -> "%.1fs".format(us / 1e6)
        }

        fun fmtDuration(s: Double): String = when {
            s < 60 -> "%.0f s".format(s)
            else -> "%d min %02d s".format((s / 60).toInt(), (s % 60).toInt())
        }
    }
}
