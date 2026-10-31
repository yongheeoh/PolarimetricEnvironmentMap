package kr.penvmap.capture

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.cgjnkim.mobileokad.flir.FlirProbe
import com.cgjnkim.mobileokad.flir.PenvmapCamera
import kr.penvmap.capture.core.BracketState
import kr.penvmap.capture.core.CalibrationKind
import kr.penvmap.capture.core.CalibrationShot
import kr.penvmap.capture.core.Exposure
import kr.penvmap.capture.core.FakeStage
import kr.penvmap.capture.core.FakeWorld
import kr.penvmap.capture.core.Intrinsics
import kr.penvmap.capture.core.LiveStitcher
import kr.penvmap.capture.core.Mosaic
import kr.penvmap.capture.core.Pose
import kr.penvmap.capture.core.RotationRunner
import kr.penvmap.capture.core.SavedView
import kr.penvmap.capture.core.SessionDirs
import kr.penvmap.capture.core.SessionIndex
import kr.penvmap.capture.core.SessionWriter
import kr.penvmap.capture.core.Stage
import java.io.File
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.ln

/**
 * The capture session, phone edition of capture/capture.py: preview with exposure statistics, color
 * checker shots (Space), the scene rotation with a live S0 envmap (Enter), the quick bracket check (c),
 * auto longest exposure (a), [ / ] and u / d, and capture_meta.json at the end of the session.
 *
 * Long operations run one at a time on [work]; the camera's own thread delivers preview frames. A camera
 * that drops off the USB bus during a capture is waited for, reopened and the same view taken again; a
 * rotation that cannot continue keeps its session open so it can be resumed from the next view.
 */
class CaptureController(private val context: Context, settings: AppSettings, private val ui: Ui) {

    enum class Level { INFO, OK, WARN, ERROR }

    /** A view could not be written (storage, not the camera): never retried by reconnecting the camera. */
    class SaveException(message: String, cause: Throwable) : IOException(message, cause)

    interface Ui {
        fun preview(bitmap: Bitmap, info: PreviewInfo)
        fun state(s: State)
        fun busy(what: String?)
        fun stitch(bitmap: Bitmap?, text: String)
        fun progress(p: Progress?)
        fun message(text: String, level: Level = Level.INFO)
    }

    class PreviewInfo(val exposureUs: Long, val over: Double, val under: Double,
                      val formatKey: String, val createdAtMs: Long)

    class State(
        val cameraLabel: String,
        val cameraOnline: Boolean,
        val stageLabel: String,
        val stageOk: Boolean,
        /** A chip's reconnect is running (its label says so). */
        val connecting: Boolean,
        val storageLabel: String,
        /** The base path (absolute) and its mode; [storageOnline] false = the picked drive is not mounted. */
        val storagePath: String,
        val storageMode: String,
        val storageOnline: Boolean,
        /** "Phone" or the drive's name. */
        val storageName: String,
        val freeBytes: Long,
        val sessionsLeft: Int,
        /** Used fraction of the base path's drive, -1 when unknown / not connected. */
        val storageUsed: Double,
        val experiment: String,
        val sessionRel: String?,
        val colorCheckers: Int,
        val sunShots: Int,
        val floorShots: Int,
        /** Where the stage was last sent (yaw, pitch), or null before the first move. */
        val stagePose: Pair<Double, Double>?,
        val viewsDone: Int,
        val totalViews: Int,
        val exposures: List<Long>,
        val previewIndex: Int,
        val stepEv: Double,
        val rangeEv: Double,
        val rotationEstimateS: Double,
        /** Times the camera dropped off USB / stopped answering since the session started (meta connection.drops). */
        val disconnects: Int,
        val resumeFrom: Int?,
        val check: String?,
        val checkOk: Boolean?,
        val calibrationMode: Boolean,
        val calibrationKind: CalibrationKind?,
        val calibrationShots: Int,
    )

    class Progress(val done: Int, val total: Int, val yawDeg: Double, val pitchDeg: Double, val etaS: Double, val note: String?)

    var settings: AppSettings = settings
        private set

    private val main = Handler(Looper.getMainLooper())
    private val location = LocationKeeper(context)
    private val work = Executors.newSingleThreadExecutor { r -> Thread(r, "capture-work") }
    private val previewBusy = AtomicBoolean(false)
    private val cancelRotation = AtomicBoolean(false)

    @Volatile private var source: CameraSource? = null
    private var stage: Stage? = null
    private var world: FakeWorld? = null
    @Volatile private var stageOk = false
    @Volatile private var cameraOnline = false
    @Volatile private var disconnects = 0
    private var lastDropMs = 0L
    private val intrinsics: Intrinsics by lazy {
        Intrinsics.fromJson(context.assets.open(AppSettings.INTRINSIC_ASSET).bufferedReader().use { it.readText() })
    }

    lateinit var bracket: BracketState
        private set
    @Volatile private var lastStats: Exposure.Stats? = null
    @Volatile private var lastCheck: Exposure.Check? = null
    @Volatile var busyWith: String? = null
        private set

    // Session state (one session = color checker shots + at most one rotation, like a PC run).
    private var session: SessionDirs? = null
    private var sessionStartNs = 0L
    private var sessionStartedAt = ""
    private var numColorChecker = 0
    @Volatile var calibrationMode = false
        private set
    @Volatile var calibrationKind: CalibrationKind? = null
        private set
    private val calibrationShots = ArrayList<Map<String, Any?>>()
    /** Sun shots of the open scene (index, stage pose, ND, time) for capture_meta.json. */
    private val sunShots = ArrayList<Map<String, Any?>>()
    private val floorShots = ArrayList<Map<String, Any?>>()
    /** Where the stage was last sent; null until the first move (the stage's position is not read back). */
    @Volatile var stagePose: Pair<Double, Double>? = null
        private set
    private val rotationViews = ArrayList<Map<String, Any?>>()
    private var rotationTiming: Map<String, Any?>? = null
    private var cameraSettings: Map<String, Any?>? = null
    /** Next view of an interrupted rotation, or null. The stitcher keeps the views taken so far. */
    @Volatile private var resumeFrom: Int? = null
    private var sessionStitcher: LiveStitcher? = null

    val baseDir: File get() = Storage.baseDir(context, settings)
    /** The base path the open session was started in (the base path changes only between sessions). */
    private var sessionBase: File? = null
    val sessionPath: String? get() = session?.root?.absolutePath
    /** The open session's drive (else the base path's) is mounted; the phone always is. */
    val storageOnline: Boolean get() = Storage.available(context, sessionBase ?: baseDir)

    /**
     * Calibration masks `files/masks/mask_{pitch}.png` (same files as main.py MASK_DIR: sensor orientation,
     * white = valid), like the PC's mask_dir: statistics use mask_0, the live envmap excludes masked pixels.
     * Missing folder / files = whole frame.
     */
    val maskDir: File get() = File(context.getExternalFilesDir(null), "masks")
    @Volatile private var masks: Map<Int, BooleanArray> = emptyMap()
    @Volatile private var smallMasks: Map<Int, BooleanArray> = emptyMap()
    private val checkMask: BooleanArray? get() = masks[0] ?: masks.values.firstOrNull()

    private fun loadMasks(width: Int, height: Int) {
        val out = LinkedHashMap<Int, BooleanArray>()
        for (pitch in settings.pitchDegs.map { it.toInt() }) {
            val f = File(maskDir, "mask_$pitch.png")
            if (!f.isFile) continue
            val bmp = android.graphics.BitmapFactory.decodeFile(f.absolutePath) ?: continue
            val scaled = if (bmp.width != width || bmp.height != height) Bitmap.createScaledBitmap(bmp, width, height, false) else bmp
            val px = IntArray(width * height)
            scaled.getPixels(px, 0, width, 0, 0, width, height)
            out[pitch] = BooleanArray(px.size) { ((px[it] shr 8) and 0xFF) >= 128 }
        }
        masks = out
        smallMasks = out.mapValues { Mosaic.downscaleMask(it.value, width, height) }
        if (out.isNotEmpty()) main.post { ui.message("masks: pitches ${out.keys.sorted()} from ${maskDir.absolutePath}") }
    }

    // ---- lifecycle --------------------------------------------------------------------------------

    /** Opens camera and stage per [newSettings] and starts the preview. */
    fun start(newSettings: AppSettings = settings) = run("opening") {
        stopSourcesBlocking()
        // A session never spans two base paths: close it under the old one first.
        if (Storage.baseDir(context, newSettings) != baseDir) closeSessionForNewBase()
        settings = newSettings
        world = if (settings.cameraMode == AppSettings.CAMERA_FAKE || settings.stageMode == AppSettings.STAGE_FAKE) {
            FakeWorld(intrinsics).also { it.mountRollDeg = settings.mountRollDeg }
        } else {
            null
        }
        val src = openCameraBlocking()
        openStageBlocking()
        pushState()
        main.post { ui.message("${if (src is FakeSource) "Simulated camera" else "FLIR"} ready" + if (stageOk) "" else " -- stage not reachable", if (stageOk) Level.OK else Level.WARN) }
    }

    /** Opens the camera per the settings: masks, the bracket for its exposure range, the preview. */
    private fun openCameraBlocking(): CameraSource {
        val src = try {
            openSource()
        } catch (e: Exception) {
            cameraOnline = false
            pushState()
            throw e
        }
        source = src
        cameraOnline = true
        runCatching { loadMasks(2448, 2048) }.onFailure { Log.w(TAG, "masks not loaded", it) }
        if (!::bracket.isInitialized || bracket.rangeUs != src.exposureRangeUs ||
            bracket.minUs != settings.bracketMinUs || bracket.maxUs != settings.bracketMaxUs || bracket.numBrackets != settings.numBrackets
        ) {
            bracket = BracketState(settings.bracketMinUs, settings.bracketMaxUs, settings.numBrackets, src.exposureRangeUs)
        }
        cameraSettings = runCatching { src.settings() }.getOrNull()
        src.setPreview(true, bracket.previewUs, ::onPreviewFrame)
        return src
    }

    /**
     * Camera chip: closes the camera and keeps trying to open it for up to [RECONNECT_MS] (a re-plugged
     * FLIR takes a moment to show up). A FLIR without USB permission goes to [needPermission] (the UI asks).
     */
    fun reconnectCamera(needPermission: () -> Unit) = run(CONNECTING_CAMERA) {
        stopSourcesBlocking()
        cameraOnline = false
        pushState()
        val deadline = System.currentTimeMillis() + RECONNECT_MS
        var last: Throwable? = null
        while (!cancelRotation.get()) {
            val fake = settings.cameraMode == AppSettings.CAMERA_FAKE
            val device = if (fake) null else FlirProbe.findDevice(context)
            if (device != null && !FlirProbe.hasPermission(context, device)) {
                main.post { needPermission() }
                return@run
            }
            if (fake || device != null) {
                try {
                    val src = openCameraBlocking()
                    pushState()
                    main.post { ui.message("${if (src is FakeSource) "Simulated camera" else "FLIR"} connected", Level.OK) }
                    return@run
                } catch (e: Exception) {
                    last = e
                    Log.w(TAG, "camera reconnect attempt failed", e)
                }
            }
            if (System.currentTimeMillis() >= deadline) break
            Thread.sleep(300)
        }
        main.post {
            ui.message(last?.let { "Camera did not open: ${it.message}" } ?: "FLIR not found -- check the USB-C cable", Level.WARN)
        }
    }

    /** The rotation stage per the settings (HTTP over its Wi-Fi, or simulated), pinged once. */
    private fun openStageBlocking() {
        stage = if (settings.stageMode == AppSettings.STAGE_FAKE) {
            world?.let { FakeWorldStage(it) } ?: FakeStage()
        } else {
            StageFactory.http(context, settings) { Log.i(TAG, it) }
        }
        stageOk = stage!!.ping()
    }

    /** Stage chip: connect to the stage again, pinging for up to [RECONNECT_MS] (the camera keeps running). */
    fun reconnectStage() = run(CONNECTING_STAGE) {
        stageOk = false
        pushState()
        openStageBlocking()
        val deadline = System.currentTimeMillis() + RECONNECT_MS
        while (!stageOk && !cancelRotation.get() && System.currentTimeMillis() < deadline) {
            Thread.sleep(400)
            stageOk = stage?.ping() == true
        }
        pushState()
        main.post {
            ui.message(if (stageOk) "Stage connected" else "Stage not reachable (${settings.mechaUrl}) -- check the phone's Wi-Fi", if (stageOk) Level.OK else Level.WARN)
        }
    }

    private fun openSource(): CameraSource {
        val src: CameraSource = if (settings.cameraMode == AppSettings.CAMERA_FAKE) {
            FakeSource(world ?: FakeWorld(intrinsics).also { world = it; it.mountRollDeg = settings.mountRollDeg }, settings.previewFps)
        } else {
            FlirSource(PenvmapCamera.Options(settings.blackLevelClamping, settings.defectCorrectStatic, previewFps = settings.previewFps)) { msg ->
                main.post { ui.message(msg, Level.WARN) }
            }
        }
        try {
            src.open(context)
        } catch (e: Exception) {
            runCatching { src.close() }
            throw e
        }
        return src
    }

    /**
     * Experiment name and base path changes: no camera / stage reopen. Queued behind any
     * running capture; a session never spans two storage folders, so an open one is finished first. A new
     * experiment name applies from the next session (the open one stays where it started).
     */
    fun updateSettings(ns: AppSettings) = run("saving settings") {
        if (Storage.baseDir(context, ns) != baseDir) closeSessionForNewBase()
        settings = ns.copy(bracketMinUs = bracket.minUs, bracketMaxUs = bracket.maxUs, numBrackets = bracket.numBrackets)
        AppSettings.save(context, settings)
        pushState()
    }

    /**
     * Live view RGB / Raw: display only, so it applies from the next frame without the queue (also during a
     * capture) and without reopening the camera.
     */
    fun setPreviewMode(mode: String) {
        settings = settings.copy(previewMode = mode)
        AppSettings.save(context, settings)
    }

    /**
     * Takes [ns] as the current settings without opening the camera -- the FLIR is not plugged in (or USB
     * permission is pending). Without this the screens kept showing, and Settings would save back, the
     * values from before.
     */
    fun adopt(ns: AppSettings) = run("saving settings") {
        if (Storage.baseDir(context, ns) != baseDir) closeSessionForNewBase()
        // The camera that was running (e.g. the simulated one) is not the one asked for: close it, or the
        // screen keeps showing its frames under the new camera's name.
        stopSourcesBlocking()
        cameraOnline = false
        settings = ns
        world = null
        bracket = BracketState(ns.bracketMinUs, ns.bracketMaxUs, ns.numBrackets, 21L..30_000_000L)
        openStageBlocking()
        pushState()
    }

    fun stop() {
        work.execute { stopSourcesBlocking() }
    }

    private fun stopSourcesBlocking() {
        val s = source
        source = null
        runCatching { s?.setPreview(false, 0, null) }
        runCatching { s?.close() }
    }

    fun shutdown() {
        cancelRotation.set(true)
        main.removeCallbacks(storageTick)
        runCatching { context.unregisterReceiver(mediaReceiver) }
        work.execute {
            finishSessionBlocking()
            stopSourcesBlocking()
        }
        work.shutdown()
    }

    // ---- USB plug events --------------------------------------------------------------------------

    /** The FLIR left the bus (cable, hub power). Idle: close the stale handle; capturing: the job recovers. */
    fun onUsbDetached() {
        if (settings.cameraMode != AppSettings.CAMERA_FLIR) return
        countDrop()
        cameraOnline = false
        pushState()
        main.post { ui.message("FLIR disconnected -- waiting for it to come back", Level.WARN) }
        if (busyWith == null) work.execute { if (busyWith == null) stopSourcesBlocking() }
    }

    /** True when a running job will pick the camera up itself (reopening now would race it). */
    fun handlesReattach(): Boolean = busyWith != null

    private fun countDrop() {
        val now = System.currentTimeMillis()
        if (now - lastDropMs > 3000) disconnects++
        lastDropMs = now
    }

    /** Simulated unplug of the FAKE camera for [ms] (adb action `fake_drop`), to test reconnection. */
    fun simulateDrop(ms: Long) {
        FakeSource.offlineUntilMs = System.currentTimeMillis() + ms
        main.post { ui.message("simulated unplug for ${ms / 1000} s", Level.WARN) }
    }

    // ---- preview ----------------------------------------------------------------------------------

    private fun onPreviewFrame(f: PreviewFrame) {
        // a frame still in flight from a camera that was just closed
        if (source == null) return
        if (!previewBusy.compareAndSet(false, true)) return
        try {
            val createdAtMs = android.os.SystemClock.elapsedRealtime()
            val previewSettings = settings
            val samples = Exposure.u8(f.pixels)
            val mask = checkMask?.takeIf { it.size == f.pixels.size }
            val stats = Exposure.overUnder(samples, f.pixels.size, 255.0, mask, stride = 3)
            lastStats = stats
            val turn = if (previewSettings.previewShowSensor) Pose.Upright.NONE else Pose.upright(previewSettings.mountRollDeg)
            val (w, h, rot) = if (previewSettings.previewMode == AppSettings.PREVIEW_RAW) {
                // raw values as they are (grey, 255 = saturated), half resolution by 2x2 max
                val (rw, rh, grey) = Mosaic.rawGray(f.pixels, f.width, f.height)
                Mosaic.rotate(grey, rw, rh, turn)
            } else {
                val s0 = Mosaic.s0(samples, f.width, f.height, 255.0)
                Mosaic.rotate(Mosaic.toU8(s0.bgr, subsample = 7), s0.width, s0.height, turn)
            }
            val bmp = bgrToBitmap(rot, w, h)
            val info = PreviewInfo(f.exposureUs, stats.over, stats.under,
                "${previewSettings.cameraMode}/${previewSettings.previewMode}/$turn", createdAtMs)
            if (!cameraOnline) {
                cameraOnline = true
                pushState()
            }
            if (source != null) main.post { ui.preview(bmp, info) }
        } catch (e: Exception) {
            Log.w(TAG, "preview processing failed", e)
        } finally {
            previewBusy.set(false)
        }
    }

    /**
     * Expected rotation time, from the S21 + FLIR + stage run of 2026-10-13 (21 us - 344 ms x 8, 18 views,
     * 145.6 s): per view the exposures + 2.25 s of bracket overhead (format switch, sequencer, stream) +
     * 0.23 s live envmap + 3.1 s yaw move with 0.2 s settle; +10 s per pitch row change; 14 s homing.
     * Files are written while the stage moves, so saving is not on the clock.
     */
    fun rotationEstimateS(views: Int = settings.yawDegs.size * settings.pitchDegs.size): Double {
        val b = bracket
        val perView = b.exposures.sum() / 1e6 + CAPTURE_OVERHEAD_S + ENVMAP_S + STAGE_YAW_S + (settings.stageSettleYawS - 0.2)
        val pitchChanges = (settings.pitchDegs.size - 1).coerceAtLeast(0)
        return views * perView + pitchChanges * (STAGE_PITCH_EXTRA_S + settings.stageSettlePitchS - 0.2) + STAGE_HOME_S
    }

    /** Measured dynamic range of the bracket with pemap's FLIR z_min (0.01) .. z_max (0.95). */
    private fun rangeEv(): Double {
        val b = bracket
        return ln(0.95 / 0.01) / ln(2.0) + ln(b.exposures.last().toDouble() / b.exposures.first()) / ln(2.0)
    }

    private fun sessionBytes(): Long =
        settings.numBrackets * 2448L * 2048L * 2 * (settings.yawDegs.size * settings.pitchDegs.size + 2)

    /** Rebuilds [State] and hands it to the UI (any thread). */
    fun pushState() {
        if (!::bracket.isInitialized) {
            bracket = BracketState(settings.bracketMinUs, settings.bracketMaxUs, settings.numBrackets, 21L..30_000_000L)
        }
        val online = storageOnline
        lastStorageOnline = online
        val free = if (!online) -1L else runCatching { baseDir.apply { mkdirs() }.usableSpace }.getOrDefault(-1L)
        val used = if (!online) -1.0 else Storage.usage(baseDir)?.second ?: -1.0
        val camLabel = when {
            busyWith == CONNECTING_CAMERA -> "Connecting camera …"
            settings.cameraMode == AppSettings.CAMERA_FAKE -> "Simulated camera"
            source != null && cameraOnline -> "FLIR"
            else -> "FLIR offline"
        }
        val stageLabel = when {
            busyWith == CONNECTING_STAGE -> "Connecting stage …"
            settings.stageMode == AppSettings.STAGE_FAKE -> "Simulated stage"
            stageOk -> "Stage"
            else -> "Stage offline"
        }
        val check = lastCheck?.let { c ->
            if (c.minStats == null || c.maxStats == null) null else
                "%s · t_min over %.3f%% · t_max under %.2f%%".format(if (c.ok) "Bracket OK" else "Bracket NG", c.minStats.over * 100, c.maxStats.under * 100) +
                    if (!c.ok && c.reason.isNotEmpty()) " · ${c.reason}" else ""
        }
        val dirs = session
        val s = State(
            cameraLabel = camLabel,
            cameraOnline = (settings.cameraMode == AppSettings.CAMERA_FAKE && busyWith != CONNECTING_CAMERA) || (source != null && cameraOnline),
            stageLabel = stageLabel,
            stageOk = (settings.stageMode == AppSettings.STAGE_FAKE && busyWith != CONNECTING_STAGE) || stageOk,
            connecting = busyWith == CONNECTING_CAMERA || busyWith == CONNECTING_STAGE,
            storageLabel = Storage.label(context, settings.storageMode, baseDir),
            storagePath = baseDir.absolutePath,
            storageMode = settings.storageMode,
            storageOnline = online,
            storageName = Storage.volumeName(context, baseDir),
            freeBytes = free,
            sessionsLeft = if (free > 0) (free / sessionBytes()).toInt() else 0,
            storageUsed = used,
            experiment = settings.experimentName,
            sessionRel = dirs?.root?.relativeTo(sessionBase ?: baseDir)?.path?.replace(File.separatorChar, '/'),
            colorCheckers = if (dirs != null) numColorChecker else 0,
            sunShots = if (dirs != null) sunShots.size else 0,
            floorShots = if (dirs != null) floorShots.size else 0,
            stagePose = stagePose,
            // During a rotation resumeFrom counts the views taken so far; rotationViews fills in at its end.
            viewsDone = if (dirs != null) maxOf(rotationViews.size, resumeFrom ?: 0) else 0,
            totalViews = settings.yawDegs.size * settings.pitchDegs.size,
            exposures = bracket.exposures,
            previewIndex = bracket.previewIndex,
            stepEv = bracket.stepEv,
            rangeEv = rangeEv(),
            rotationEstimateS = rotationEstimateS(),
            disconnects = disconnects,
            resumeFrom = resumeFrom.takeIf { busyWith != "rotation" }, // only an interrupted rotation offers Resume
            check = check,
            checkOk = lastCheck?.ok,
            calibrationMode = calibrationMode,
            calibrationKind = calibrationKind,
            calibrationShots = if (dirs != null) calibrationShots.size else 0,
        )
        main.post { ui.state(s) }
    }

    /** Per-session notes (weather, content, free text), written to capture_meta.json "annotations". */
    var annotations: Map<String, String> = emptyMap()
        set(value) {
            field = value
            if (session != null) work.execute { runCatching { writeMeta() } }
        }

    // ---- bracket adjustments (u / d / [ / ] / min / frames) ----------------------------------------

    fun previewUp() = adjust { bracket.previewUp(); null }
    fun previewDown() = adjust { bracket.previewDown(); null }
    fun previewAt(index: Int) = adjust {
        while (bracket.previewIndex < index && bracket.previewUp()) Unit
        while (bracket.previewIndex > index && bracket.previewDown()) Unit
        null
    }
    fun maxTimes(f: Double) = adjust { bracket.setMax((bracket.maxUs * f).toLong()) }
    fun minTimes(f: Double) = adjust { bracket.set(minUs = maxOf(1L, (bracket.minUs * f).toLong())) }
    fun framesPlus(d: Int) = adjust { bracket.set(numBrackets = (bracket.numBrackets + d).coerceIn(1, 32)) }

    private fun adjust(change: () -> String?) {
        if (!::bracket.isInitialized) return
        if (resumeFrom != null && busyWith == null) {
            // A resumed rotation must keep the bracket of its first views.
            ui.message("Finish or resume the rotation first", Level.WARN)
            return
        }
        val reason = change()
        if (reason != null) ui.message(reason, Level.WARN) else lastCheck = null
        source?.setPreviewExposure(bracket.previewUs)
        persistBracket()
        pushState()
    }

    private fun persistBracket() {
        settings = settings.copy(bracketMinUs = bracket.minUs, bracketMaxUs = bracket.maxUs, numBrackets = bracket.numBrackets)
        AppSettings.save(context, settings)
    }

    // ---- captures ---------------------------------------------------------------------------------

    /** Space: one color checker HDR stack at the current camera direction, without moving the stage. */
    fun captureColorChecker() = run("color checker") {
        check(!calibrationMode) { "Finish calibration before taking a scene color checker" }
        requireStorage()
        val dirs = ensureSession()
        val idx = SessionWriter.nextIndex(dirs.colorChecker)
        val pose = stagePose
        // Keep numeric pose fields for the PC reader; flag an unknown pose instead of claiming home.
        withCamera("color checker") { src ->
            captureView(src, dirs.colorChecker, idx, pose?.first ?: 0.0, pose?.second ?: 0.0,
                extra = mapOf("stage_pose_known" to (pose != null)))
        }
        numColorChecker++
        writeMeta()
        main.post { ui.message("Color checker $idx saved", Level.OK) }
        pushState()
    }

    /**
     * Enter: the scene rotation with the live envmap, then the session ends (PC behaviour). With
     * [resume] an interrupted rotation continues from its next view in the same session.
     */
    fun rotate(mode: RotateMode = RotateMode.START) = run("rotation") {
        check(!calibrationMode) { "Finish calibration before rotating a scene" }
        val resume = mode == RotateMode.RESUME
        requireStorage()
        val stg = stage ?: error("no stage")
        if (!stageOk) stageOk = stg.ping()
        if (!stageOk) {
            main.post { ui.message("Stage not reachable -- rotation skipped", Level.WARN) }
            return@run
        }
        when (mode) {
            RotateMode.START -> if (sceneHasRotation()) {
                main.post { ui.message("Scene already rotated -- Retake or New scene", Level.WARN) }
                return@run
            }
            RotateMode.NEW_SCENE -> finishSessionBlocking()
            RotateMode.RETAKE -> session?.let { clearRotation(it) }
            RotateMode.RESUME -> Unit
        }
        val startAt = if (resume) resumeFrom ?: 0 else 0
        val dirs = ensureSession()
        val stitcher = if (resume && sessionStitcher != null) sessionStitcher!! else
            LiveStitcher(intrinsics.scaled(0.25, 0.25), settings.stitchWidth, settings.stitchHeight, feather = true).also {
                sessionStitcher = it
                rotationViews.clear()
            }
        val runner = RotationRunner(stg, settings.yawDegs, settings.pitchDegs, settings.captureDelayS)
        stagePose = 0.0 to 0.0 // the runner starts and ends at home
        val total = runner.totalViews
        var last: SavedView? = null
        var lastPitch = 0.0
        main.post { ui.stitch(null, if (startAt > 0) "Resuming at view ${startAt + 1}/$total" else "Rotation: $total views") }
        val saves = AsyncSaves()
        val tStart = System.nanoTime()
        var doneThisRun = 0
        var failure: Throwable? = null
        var timing: Map<String, Any?> = emptyMap()
        val newViews = ArrayList<Map<String, Any?>>()
        try {
            val (v, t) = runner.run(object : RotationRunner.Callbacks {
                override fun capture(idx: Int, yawDeg: Double, pitchDeg: Double) {
                    // A view that could not be written (the drive went away) stops the rotation; Resume retakes it.
                    saves.firstFailed?.let { throw IOException("saving view ${it + 1} failed: ${saves.firstError}") }
                    // Bracket now; the files are written while the stage moves on (at most one view queued).
                    last = withCamera("view ${idx + 1}") { src -> captureView(src, dirs.scene, idx, yawDeg, pitchDeg, saves) }
                    lastPitch = pitchDeg
                    resumeFrom = idx + 1
                }

                override fun stitch(idx: Int) {
                    val s = last ?: return
                    stitcher.add(s.s0.bgr, s.s0.width, s.s0.height, s.rvec, smallMasks[lastPitch.toInt()]?.takeIf { it.size == s.s0.width * s.s0.height })
                    val bmp = bgrToBitmap(Mosaic.toU8(stitcher.result(), subsample = 11), stitcher.width, stitcher.height)
                    main.post { ui.stitch(bmp, "View ${idx + 1}/$total") }
                }

                override fun progress(done: Int, total: Int, yawDeg: Double, pitchDeg: Double) {
                    doneThisRun++
                    val perView = (System.nanoTime() - tStart) / 1e9 / doneThisRun
                    main.post { ui.progress(Progress(done, total, yawDeg, pitchDeg, perView * (total - done), null)) }
                    pushState()
                }

                override fun cancelled(): Boolean = cancelRotation.get()
            }, skipBefore = startAt)
            newViews += v
            timing = t
        } catch (e: Throwable) {
            failure = e
        }
        main.post { ui.busy("writing the last views ...") }
        val saveTimes = saves.await()
        saves.firstFailed?.let { bad ->
            resumeFrom = minOf(resumeFrom ?: bad, bad)
            if (failure == null) failure = IOException("saving view ${bad + 1} failed: ${saves.firstError}")
        }
        // Views that finished before a failure are in resumeFrom; their timings come from the save thread.
        val captured = if (failure == null) newViews else
            (startAt until (resumeFrom ?: startAt)).map { i -> linkedMapOf<String, Any?>("idx" to i) }
        rotationViews += captured.map { view -> view + ("save_s" to saveTimes[(view["idx"] as Number).toInt()]) }
        rotationTiming = (rotationTiming ?: emptyMap()) + timing + linkedMapOf<String, Any?>(
            "total_save_s" to RotationRunner.round1(saveTimes.values.sum()),
            "save_errors" to saves.errors.takeIf { it.isNotEmpty() },
            "resumed_at_view" to startAt.takeIf { it > 0 },
        ).filterValues { it != null }
        if (saves.errors.isNotEmpty()) main.post { ui.message("SAVE ERRORS: ${saves.errors.joinToString("; ")}", Level.ERROR) }
        main.post { ui.progress(null) }
        val f = failure
        when {
            f != null -> {
                // Keep the session: Resume continues from the next view.
                runCatching { writeMeta() }.onFailure { Log.w(TAG, "meta not written", it) }
                val next = resumeFrom ?: startAt
                main.post {
                    ui.stitch(bgrToBitmap(Mosaic.tonemap(stitcher.result()), stitcher.width, stitcher.height), "Stopped at view ${next + 1}/$total")
                    ui.message("Stopped at view ${next + 1}/$total -- Resume continues: ${f.message}", Level.ERROR)
                }
                pushState()
                throw f
            }
            cancelRotation.get() -> {
                // The scene stays open with the views taken so far: Resume, Retake or Finish.
                writeMeta()
                val next = resumeFrom ?: startAt
                main.post {
                    ui.stitch(bgrToBitmap(Mosaic.tonemap(stitcher.result()), stitcher.width, stitcher.height), "Cancelled at view ${next + 1}/$total")
                    ui.message("Cancelled at view ${next + 1}/$total -- views so far saved", Level.WARN)
                }
                pushState()
            }
            else -> {
                // The scene stays open: Rotate again retakes it, Finish ends it.
                resumeFrom = null
                sessionStitcher = null
                writeMeta()
                val text = "Rotation took ${timing["rotation_duration_s"]} s, uncovered %.1f%%".format(stitcher.uncoveredFraction() * 100)
                main.post {
                    ui.stitch(bgrToBitmap(Mosaic.tonemap(stitcher.result()), stitcher.width, stitcher.height), text)
                    ui.message("Rotation saved -- Finish ends the scene", Level.OK)
                }
                pushState()
            }
        }
    }

    /**
     * How Rotate treats the open scene. A scene stays open after its rotation (so it can be taken again) until
     * Finish; a second rotation in it must say whether it replaces the first one or starts a new scene.
     */
    enum class RotateMode {
        /** The scene's first rotation; refused when the open scene already has one. */
        START,
        /** Continue an interrupted rotation from its next view. */
        RESUME,
        /** Take the open scene's rotation again: its views are deleted first (color checker shots stay). */
        RETAKE,
        /** Finish the open scene and rotate in a new one. */
        NEW_SCENE,
    }

    /** The open scene has rotation views (complete or interrupted). */
    private fun sceneHasRotation(): Boolean = session != null && (rotationViews.isNotEmpty() || resumeFrom != null)

    /** Deletes the open scene's rotation (stack, ldr, stitch); the color checker shots and notes stay. */
    private fun clearRotation(dirs: SessionDirs) {
        for (d in listOf(dirs.scene.stack, dirs.scene.ldrRaw)) d.listFiles { f -> f.isFile }?.forEach { it.delete() }
        dirs.scene.stitchS0.delete()
        rotationViews.clear()
        rotationTiming = null
        resumeFrom = null
        sessionStitcher = null
    }

    /** Cancel: stops a rotation after the current view, or stops waiting for the camera. */
    fun cancelRotation() = cancelRotation.set(true)

    /** Floor references share the open scene but neither move the stage nor enter its live stitch. */
    fun captureFloor(note: String = "") = run("floor") {
        check(!calibrationMode) { "Finish calibration before taking a scene floor shot" }
        val dirs = checkNotNull(session) { "Capture the scene rotation before its floor shots" }
        check(rotationViews.isNotEmpty() && resumeFrom == null) { "Finish the rotation before moving the rig for floor shots" }
        requireStorage()
        check(dirs.root.isDirectory) { "The scene folder is missing; floor capture stopped" }
        dirs.create()
        dirs.floor.create()
        val idx = SessionWriter.nextIndex(dirs.floor)
        val pose = stagePose
        val metadata = kr.penvmap.capture.core.FloorShot(note).metadata(
            dirs.root.relativeTo(sessionBase ?: baseDir).invariantSeparatorsPath, pose, RotationRunner.isoNow())
        withCamera("floor") { src ->
            captureView(src, dirs.floor, idx, pose?.first ?: 0.0, pose?.second ?: 0.0,
                extra = mapOf("capture_kind" to "floor", "floor_shot" to metadata,
                    "rvec_reference" to "stage_local_after_repositioning", "stage_pose_known" to (pose != null)))
        }
        floorShots += metadata + ("idx" to idx)
        writeMeta()
        pushState()
        main.post { ui.message("Floor ${idx + 1} saved in this scene", Level.OK) }
    }

    // ---- sun ----------------------------------------------------------------------------------------

    /** Absolute stage move; remembers where the stage was sent. [jog]: aiming step, no settle wait. */
    private fun moveStage(yawDeg: Double, pitchDeg: Double, jog: Boolean = false) {
        val stg = stage ?: throw IOException("no stage")
        val (y0, p0) = stagePose ?: (Double.NaN to Double.NaN)
        val y = yawDeg.takeIf { it != y0 }
        val p = pitchDeg.takeIf { it != p0 }
        if (jog) stg.jogAbs(y, p) else stg.moveAbs(y, p, large = pitchDeg != p0 || abs(yawDeg - y0) > 10)
        stagePose = yawDeg to pitchDeg
    }

    /**
     * Sun mode: moves the stage by (dYaw, dPitch) degrees from where it was sent last. The first move goes
     * home (0, 0) first, so every pose is relative to the stage origin.
     */
    fun stageJog(dYawDeg: Double, dPitchDeg: Double, done: (() -> Unit)? = null) = run(STAGE_MOVE) {
        try {
            if (stage == null || !(stageOk || settings.stageMode == AppSettings.STAGE_FAKE)) throw IOException("stage not connected")
            val (y, p) = stagePose ?: (0.0 to 0.0).also { moveStage(0.0, 0.0) }
            moveStage(y + dYawDeg, (p + dPitchDeg).coerceIn(-PITCH_LIMIT_DEG, PITCH_LIMIT_DEG), jog = true)
            pushState()
        } finally {
            done?.let { main.post(it) }
        }
    }

    fun stageHome() = run(STAGE_MOVE) {
        moveStage(0.0, 0.0)
        pushState()
    }

    /**
     * Sun shot: the bracket at the stage pose the user aimed at the sun (sun near the image centre), saved
     * under nd/ with the pose, the stage's status reply and the ND filter. [ndOd] = optical density on the
     * lens (null = none).
     */
    fun captureSun(ndOd: Double?) = run("sun") {
        check(!calibrationMode) { "Use Capture in calibration, or finish it before a scene sun shot" }
        requireStorage()
        val (yaw, pitch) = stagePose ?: (0.0 to 0.0)
        val dirs = ensureSession()
        val idx = SessionWriter.nextIndex(dirs.sun)
        val at = RotationRunner.isoNow()
        val stageStatus = runCatching { stage?.status() }.getOrNull()
        val extra = linkedMapOf<String, Any?>(
            "sun_shot" to linkedMapOf(
                "stage_yaw_deg" to yaw,
                "stage_pitch_deg" to pitch,
                "stage_pose_known" to (stagePose != null),
                "nd_od" to ndOd,
                "captured_at" to at,
                "stage_status" to stageStatus,
                "note" to "rvec is the stage prior for (yaw, pitch); refine with the scene's rotation and the sun centre",
            ),
        )
        withCamera("sun") { src -> captureView(src, dirs.sun, idx, yaw, pitch, extra = extra) }
        sunShots += linkedMapOf("idx" to idx, "yaw_deg" to yaw, "pitch_deg" to pitch, "nd_od" to ndOd, "captured_at" to at)
        writeMeta()
        main.post { ui.message("Sun $idx saved (yaw %.1f°, pitch %.1f°%s)".format(yaw, pitch, ndOd?.let { ", ND $it" } ?: ""), Level.OK) }
        pushState()
    }

    /**
     * Where the optical axis (principal point) is in the preview image as shown (fractions 0..1), for the
     * sun-mode crosshair.
     */
    fun principalPointInPreview(): Pair<Float, Float> {
        val fx = ((intrinsics.k[2] + 0.5) / PenvmapCamera.SENSOR_WIDTH).toFloat()
        val fy = ((intrinsics.k[5] + 0.5) / PenvmapCamera.SENSOR_HEIGHT).toFloat()
        val turn = if (settings.previewShowSensor) Pose.Upright.NONE else Pose.upright(settings.mountRollDeg)
        return when (turn) {
            Pose.Upright.CCW -> fy to 1f - fx
            Pose.Upright.CW -> 1f - fy to fx
            Pose.Upright.R180 -> 1f - fx to 1f - fy
            Pose.Upright.NONE -> fx to fy
        }
    }

    /**
     * FLIR diagnostics (filr_android's FlirProbe): USB link speed, model, pixel formats, a grab test, and
     * the camera's own GenICam XML saved under files/Documents. Writes a report to files/diagnostics/.
     * The camera is closed for this and reopened afterwards.
     */
    fun probe() = run("FLIR diagnostics") {
        val device = FlirProbe.findDevice(context) ?: error("FLIR not connected")
        stopSourcesBlocking()
        try {
            val report = FlirProbe.run(context, device)
            val dir = File(context.getExternalFilesDir(null), "diagnostics").apply { mkdirs() }
            val stamp = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US).format(java.util.Date())
            val f = File(dir, "probe_$stamp.txt")
            f.writeText(report.summary + "\n\nGenICam XML: " + (report.xmlFileName ?: "(not saved)") + "\n")
            Log.i(TAG, report.summary)
            main.post { ui.message("Diagnostics saved: ${f.name}", Level.OK) }
        } finally {
            reopenBlocking()
        }
    }

    /** c: quick check of the bracket ends (both frames in one sequencer run). */
    fun quickCheck() = run("check") {
        val b = bracket
        val tMin = b.exposures.first()
        val tMax = b.exposures.last()
        val frames = withCamera("check") { src -> src.captureBracket(listOf(tMin, tMax)) }
        val stats = frames.frames.map { Exposure.overUnder(Exposure.u16le(it), frames.width * frames.height, 65535.0, checkMask?.takeIf { m -> m.size == frames.width * frames.height }, stride = 3) }
        lastCheck = Exposure.judge(tMin, tMax, stats[0], stats[1])
        source?.setPreviewExposure(b.previewUs)
        main.post { ui.message(if (lastCheck!!.ok) "Bracket OK" else "Bracket NG: ${lastCheck!!.reason}", if (lastCheck!!.ok) Level.OK else Level.WARN) }
        pushState()
    }

    /** a: shortest longest-exposure whose dark fraction is acceptable (shortest stays fixed). */
    fun autoMax() = run("auto") {
        val src = source ?: error("no camera")
        val b = bracket
        val candidates = Exposure.autoCandidates(b.minUs, b.maxUs, 2, src.exposureRangeUs)
        val (chosen, tried) = Exposure.autoSelect(candidates) { e ->
            runCatching {
                val f = src.captureBracket(listOf(e))
                Exposure.overUnder(Exposure.u16le(f.frames[0]), f.width * f.height, 65535.0, checkMask?.takeIf { m -> m.size == f.width * f.height }, stride = 3).under
            }.getOrNull()
        }
        val msg = if (chosen == null) {
            "AUTO: no candidate keeps the dark fraction <= 20% (tried ${tried.joinToString { it.first.toString() }})"
        } else {
            b.setMax(chosen) ?: run { persistBracket(); "AUTO: longest exposure $chosen us" }
        }
        src.setPreviewExposure(b.previewUs)
        main.post { ui.message(msg) }
        pushState()
    }

    // ---- camera recovery --------------------------------------------------------------------------

    /**
     * Runs [block] with the open camera. When it fails (the FLIR dropped off the bus, a stream stalled) the
     * camera is closed, waited for and reopened, and [block] runs again -- the same bracket, so a rotation
     * keeps its view. Cancellation and interrupts are not retried.
     */
    private fun <T> withCamera(what: String, block: (CameraSource) -> T): T {
        var attempt = 0
        while (true) {
            val src = source ?: reopenBlocking(what)
            try {
                return block(src)
            } catch (e: Exception) {
                // Only camera trouble is retried: a file that cannot be written is not fixed by reconnecting.
                if (e is InterruptedException || e is SaveException || cancelRotation.get()) throw e
                attempt++
                Log.w(TAG, "$what failed (attempt $attempt of ${RECOVERY_ATTEMPTS + 1})", e)
                if (attempt > RECOVERY_ATTEMPTS) throw e
                countDrop()
                cameraOnline = false
                pushState()
                main.post { ui.message("Camera not responding -- reconnecting ($attempt/$RECOVERY_ATTEMPTS)", Level.WARN) }
                stopSourcesBlocking()
                reopenBlocking(what)
            }
        }
    }

    /** Waits (up to [RECOVERY_WAIT_MS]) for the camera to be back on the bus and opens it again. */
    private fun reopenBlocking(what: String = "camera"): CameraSource {
        val deadline = System.currentTimeMillis() + RECOVERY_WAIT_MS
        var lastError: Throwable? = null
        while (true) {
            if (cancelRotation.get()) throw InterruptedException("cancelled while waiting for the camera")
            val present = settings.cameraMode == AppSettings.CAMERA_FAKE ||
                FlirProbe.findDevice(context)?.let { FlirProbe.hasPermission(context, it) } == true
            if (present) {
                try {
                    val src = openSource()
                    source = src
                    cameraOnline = true
                    src.setPreview(true, bracket.previewUs, ::onPreviewFrame)
                    pushState()
                    main.post { ui.message("Camera reconnected", Level.OK) }
                    return src
                } catch (e: Exception) {
                    lastError = e
                    Log.w(TAG, "reopen failed", e)
                }
            }
            val left = (deadline - System.currentTimeMillis()) / 1000
            if (left <= 0) throw IOException("camera did not come back within ${RECOVERY_WAIT_MS / 1000} s" + (lastError?.message?.let { ": $it" } ?: ""))
            main.post { ui.busy("$what -- waiting for the camera (${left} s)") }
            Thread.sleep(1000)
        }
    }

    // ---- session ----------------------------------------------------------------------------------

    /** A separate reference session; never repurpose an already open scene. */
    fun beginCalibration(kind: CalibrationKind? = null) = run("opening calibration") {
        check(session == null || session?.calibrationKind != null) { "Finish the open scene before calibration" }
        check(session == null || session?.calibrationKind == kind) { "Finish the current calibration before choosing another type" }
        if (session == null) {
            annotations = emptyMap()
            calibrationShots.clear()
        }
        calibrationMode = true
        calibrationKind = kind
        pushState()
    }

    /** Common raw capture at the current camera direction, without moving the stage. */
    fun captureCalibration(shot: CalibrationShot) = run("calibration capture") {
        check(calibrationMode && calibrationKind == shot.kind) { "Select the calibration type before Capture" }
        requireStorage()
        val dirs = ensureSession()
        val part = checkNotNull(dirs.calibration)
        val idx = SessionWriter.nextIndex(part)
        val pose = stagePose
        val metadata = shot.metadata(pose != null) + ("captured_at" to RotationRunner.isoNow())
        withCamera("calibration capture") { src ->
            captureView(src, part, idx, pose?.first ?: 0.0, pose?.second ?: 0.0,
                extra = mapOf("capture_kind" to "calibration", "calibration" to metadata))
        }
        calibrationShots += metadata + ("idx" to idx)
        writeMeta()
        pushState()
        main.post { ui.message("${shot.kind.label}: capture ${idx + 1} saved", Level.OK) }
    }

    /** Finish this type's capture group, remaining in Calibration with no type selected. */
    fun finishCalibration(notes: Map<String, String> = annotations) = run("finishing calibration") {
        check(calibrationMode) { "No calibration is open" }
        val kind = calibrationKind
        annotations = notes
        finishSessionBlocking()
        calibrationKind = null
        calibrationShots.clear()
        annotations = emptyMap()
        pushState()
        main.post { ui.message("${kind?.label ?: "Calibration"} finished; choose a calibration type", Level.OK) }
    }

    fun leaveCalibration() = run("leaving calibration") {
        check(session == null) { "Finish the current calibration before returning to Scene" }
        calibrationMode = false
        calibrationKind = null
        calibrationShots.clear()
        annotations = emptyMap()
        pushState()
    }

    private fun ensureSession(): SessionDirs {
        session?.let { open ->
            if (open.root.isDirectory) return open.create() // re-creates any sub folder that went missing
            // The open scene's folder was deleted outside the app (file manager, adb): start a new scene.
            Log.w(TAG, "open scene ${open.root} is gone -- starting a new one")
            main.post { ui.message("Scene folder deleted -- new scene started", Level.WARN) }
            session = null
            sessionBase = null
            resumeFrom = null
            sessionStitcher = null
        }
        val dirs = calibrationKind?.let { SessionDirs.prepareCalibration(baseDir, it) }
            ?: SessionDirs.prepare(baseDir, settings.experimentName)
        session = dirs
        sessionBase = baseDir
        main.post { location.refresh() }
        sessionStartNs = System.nanoTime()
        sessionStartedAt = RotationRunner.isoNow()
        numColorChecker = 0
        calibrationShots.clear()
        sunShots.clear()
        floorShots.clear()
        rotationViews.clear()
        rotationTiming = null
        resumeFrom = null
        sessionStitcher = null
        disconnects = 0 // drops are counted per session (capture_meta.json "connection")
        // Make a new session discoverable even if its first capture is interrupted.
        writeMeta()
        pushState()
        return dirs
    }

    /** Captures need the base path's drive (or the open session's) to be mounted. */
    private fun requireStorage() {
        if (!storageOnline) {
            val dir = sessionBase ?: baseDir
            throw IOException("${Storage.readable(context, dir.absolutePath)}: ${Storage.offlineReason(context, dir) ?: "not connected"}")
        }
    }

    /**
     * The base path changes: the open session ends where it started. When its drive is gone the session is
     * left as it is (its capture_meta.json from the last shot, "finished": false) and a new one starts.
     */
    private fun closeSessionForNewBase() {
        if (session == null) return
        val where = session?.root?.absolutePath
        runCatching { finishSessionBlocking() }.onFailure {
            Log.w(TAG, "could not finish $where", it)
            session = null
            sessionBase = null
            resumeFrom = null
            sessionStitcher = null
            main.post { ui.message("The open scene on the drive was left unfinished ($where)", Level.WARN) }
        }
    }

    // ---- drive watch ------------------------------------------------------------------------------

    @Volatile private var lastStorageOnline = true

    /** Mount / unmount of a drive (or the 3 s check): refreshes the screen and says when the base path comes or goes. */
    private fun storageChanged() {
        val online = storageOnline
        if (online == lastStorageOnline) return
        val where = Storage.readable(context, (sessionBase ?: baseDir).absolutePath)
        main.post { ui.message(if (online) "$where connected" else "$where is not connected", if (online) Level.OK else Level.WARN) }
        pushState()
    }

    private val mediaReceiver = object : android.content.BroadcastReceiver() {
        override fun onReceive(c: Context, intent: android.content.Intent) = storageChanged()
    }

    private val storageTick = object : Runnable {
        override fun run() {
            if (settings.storageMode == Storage.CUSTOM || sessionBase != null) storageChanged()
            main.postDelayed(this, 3000)
        }
    }

    init {
        val filter = android.content.IntentFilter().apply {
            for (a in listOf(
                android.content.Intent.ACTION_MEDIA_MOUNTED, android.content.Intent.ACTION_MEDIA_UNMOUNTED,
                android.content.Intent.ACTION_MEDIA_EJECT, android.content.Intent.ACTION_MEDIA_REMOVED,
                android.content.Intent.ACTION_MEDIA_BAD_REMOVAL,
            )) addAction(a)
            addDataScheme("file")
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) context.registerReceiver(mediaReceiver, filter, Context.RECEIVER_EXPORTED)
        else context.registerReceiver(mediaReceiver, filter)
        main.postDelayed(storageTick, 3000)
        // Scenes from before 2026-10-13 (the old app folder) join the phone's base path, once.
        work.execute {
            val n = runCatching { Storage.migrateOldAppFolder(context) }.onFailure { Log.w(TAG, "old app folder not moved", it) }.getOrDefault(0)
            if (n > 0) main.post { ui.message("Moved $n scene(s) to Phone/Documents/penvmap", Level.OK) }
        }
    }

    /** Ends the session: writes capture_meta.json; the next capture starts a new session. */
    fun finishSession() = run("finishing") {
        finishSessionBlocking()
        calibrationMode = false
        calibrationKind = null
        calibrationShots.clear()
        pushState()
    }

    /** Finish with the scene's notes (weather, content, free text): written into its capture_meta.json, then cleared. */
    fun finishSession(notes: Map<String, String>) = run("finishing") {
        annotations = notes
        finishSessionBlocking()
        annotations = emptyMap()
    }

    private fun finishSessionBlocking() {
        session ?: return
        writeMeta(final = true)
        session = null
        sessionBase = null
        resumeFrom = null
        sessionStitcher = null
        pushState()
    }

    private fun writeMeta(final: Boolean = false) {
        val dirs = session ?: return
        val date = dirs.root.parentFile!!.name + "/" + dirs.root.name
        val b = bracket
        val meta = linkedMapOf<String, Any?>(
            "camera" to (if (settings.cameraMode == AppSettings.CAMERA_FAKE) "fake" else "flir"),
            "pixel_format" to "BayerRG16",
            "raw_max" to null,
            "gain" to 0,
            "mount_roll_deg" to settings.mountRollDeg,
            "config" to settings.toConfigMap((sessionBase ?: baseDir).absolutePath, date) +
                ("experiment_name" to (dirs.calibrationKind?.experiment ?: settings.experimentName)) +
                ("mask_dir" to (if (masks.isNotEmpty()) maskDir.absolutePath else null)),
            "camera_settings" to cameraSettings,
            "num_color_checker" to numColorChecker,
            "num_sun" to sunShots.size,
            "sun_shots" to sunShots.toList().takeIf { it.isNotEmpty() },
            "num_floor" to floorShots.size,
            "floor_shots" to floorShots.toList().takeIf { it.isNotEmpty() },
            "bracket_min_us" to b.minUs,
            "bracket_max_us" to b.maxUs,
            "num_brackets" to b.numBrackets,
            "bracket_step_ev" to Math.round(b.stepEv * 1e4) / 1e4,
            "exposure_list_us" to b.exposures,
            "rotation_timing" to rotationTiming,
            "views" to rotationViews.toList(),
            "annotations" to annotations.takeIf { it.isNotEmpty() },
            "location" to runCatching { location.snapshot() }.getOrNull(),
            "session_timing" to linkedMapOf(
                "session_started_at" to sessionStartedAt,
                "session_finished_at" to RotationRunner.isoNow(),
                "session_duration_s" to RotationRunner.round1((System.nanoTime() - sessionStartNs) / 1e9),
                "finished" to final,
            ),
            "connection" to linkedMapOf("drops" to disconnects).takeIf { disconnects > 0 },
            "app" to linkedMapOf(
                "name" to "penvmap-capture-android",
                "version" to runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName }.getOrNull(),
                "phone" to "${Build.MANUFACTURER} ${Build.MODEL}",
                "android" to Build.VERSION.RELEASE,
                "camera_device" to source?.deviceInfo(),
                "storage" to settings.storageMode,
            ),
        )
        dirs.calibrationKind?.let { kind ->
            meta["capture_kind"] = "calibration"
            meta["num_calibration"] = calibrationShots.size
            meta["calibration"] = linkedMapOf("type" to kind.id, "scope" to "global", "shots" to calibrationShots.toList())
        }
        SessionWriter.saveMeta(dirs, meta)
    }

    /**
     * Bracket + save (capture/session.py `capture_view`). With [saves] the files are written on the
     * save thread and only the S0 / prior rotation for the live envmap are computed here.
     */
    private fun captureView(
        src: CameraSource, part: SessionDirs.Part, idx: Int, yawDeg: Double, pitchDeg: Double, saves: AsyncSaves? = null,
        extra: Map<String, Any?> = emptyMap(),
    ): SavedView {
        val b = bracket
        val exposures = b.exposures
        val frames = src.captureBracket(exposures)
        src.setPreviewExposure(b.previewUs)
        if (frames.info["chunk_verified"] == false) noteUnverifiedChunks(src)
        val camera = if (settings.cameraMode == AppSettings.CAMERA_FAKE) "fake" else "flir"
        val roll = settings.mountRollDeg
        val rawPng = settings.saveRawPng
        val write = {
            try {
                SessionWriter.saveView(
                    part, idx, frames.frames, frames.width, frames.height, frames.exposureUs, exposures,
                    yawDeg, pitchDeg, roll, frames.info, camera = camera, saveRawPng = rawPng, extra = extra,
                )
            } catch (e: Exception) {
                throw SaveException("could not save to ${part.stack.parentFile?.parentFile?.absolutePath}: ${e.message}", e)
            }
        }
        if (saves == null) return write()
        val mid = frames.frames[frames.frames.size / 2]
        val s0 = Mosaic.s0(Exposure.u16le(mid), frames.width, frames.height, 65535.0)
        saves.submit(idx) { write() }
        return SavedView(idx, s0, Pose.priorRvec(yawDeg, pitchDeg, roll))
    }

    private var chunkWarningShown = false

    /** Frames were kept without chunk verification: tell the user once and keep the bytes for diagnosis. */
    private fun noteUnverifiedChunks(src: CameraSource) {
        if (chunkWarningShown) return
        chunkWarningShown = true
        val dir = File(context.getExternalFilesDir(null), "diagnostics").apply { mkdirs() }
        src.diagnosticBytes()?.let { File(dir, "payload_tail_${System.currentTimeMillis()}.bin").writeBytes(it) }
        main.post { ui.message("WARNING: no chunk data -- programmed exposures used", Level.WARN) }
    }

    /**
     * Writes views on a background thread while the rotation goes on. At most one bracket waits to be
     * written (two in memory: the one being written and the one being captured), so a slow card makes
     * the rotation wait rather than run out of memory.
     */
    private class AsyncSaves {
        private val exec = Executors.newSingleThreadExecutor { r -> Thread(r, "capture-save") }
        private val slot = java.util.concurrent.Semaphore(1)
        private val futures = ArrayList<java.util.concurrent.Future<*>>()
        val times = java.util.concurrent.ConcurrentHashMap<Int, Double>()
        val errors: MutableList<String> = java.util.Collections.synchronizedList(ArrayList())
        /** The first view index whose files could not be written (and why), or null. */
        @Volatile var firstFailed: Int? = null
        @Volatile var firstError: String? = null

        fun submit(idx: Int, job: () -> Unit) {
            slot.acquire()
            futures += exec.submit {
                val t0 = System.nanoTime()
                try {
                    job()
                } catch (e: Throwable) {
                    Log.e(TAG, "saving view $idx failed", e)
                    errors += "view $idx: ${e.message}"
                    synchronized(this) {
                        if (firstFailed == null || idx < firstFailed!!) {
                            firstFailed = idx
                            firstError = e.message ?: e.toString()
                        }
                    }
                } finally {
                    times[idx] = RotationRunner.round3((System.nanoTime() - t0) / 1e9)
                    slot.release()
                }
            }
        }

        /** Waits for every queued write; returns the write time per view index. */
        fun await(): Map<Int, Double> {
            for (f in futures) runCatching { f.get() }
            exec.shutdown()
            return times.toMap()
        }
    }

    // ---- plumbing ---------------------------------------------------------------------------------

    private fun run(what: String, block: () -> Unit) {
        work.execute {
            if (ProcessingJobs.active(context)) {
                main.post { ui.message("Envmap processing is running. Finish or cancel it in Gallery before capturing.", Level.WARN) }
                return@execute
            }
            busyWith = what
            cancelRotation.set(false) // Cancel applies to the job that is running when it is pressed
            main.post { ui.busy(what) }
            try {
                block()
            } catch (e: Throwable) {
                Log.e(TAG, "$what failed", e)
                main.post { ui.message("$what failed: ${e.message}", Level.ERROR) }
            } finally {
                busyWith = null
                main.post { ui.busy(null) }
                pushState()
            }
        }
    }

    companion object {
        private const val TAG = "CaptureController"
        // Rotation timing measured on the phone, 2026-10-13 (see rotationEstimateS).
        private const val CAPTURE_OVERHEAD_S = 2.25
        private const val ENVMAP_S = 0.23
        private const val STAGE_YAW_S = 3.1
        private const val STAGE_PITCH_EXTRA_S = 10.0
        private const val STAGE_HOME_S = 14.0
        /** Reconnect attempts per capture before it fails (each waits up to [RECOVERY_WAIT_MS]). */
        private const val RECOVERY_ATTEMPTS = 3
        private const val RECOVERY_WAIT_MS = 60_000L
        /** How long a tap on the camera / stage chip keeps trying. */
        private const val RECONNECT_MS = 3_000L
        /** Sun mode: the stage's pitch is kept inside this (deg). */
        private const val PITCH_LIMIT_DEG = 90.0
        /** [Ui.busy] name of a sun-mode stage move (shown quietly, no Cancel). */
        const val STAGE_MOVE = "moving the stage"
        /** [Ui.busy] names of the chip reconnects (the UI shows them as a blocking popup). */
        const val CONNECTING_CAMERA = "connecting the camera"
        const val CONNECTING_STAGE = "connecting the stage"

        fun bgrToBitmap(bgr: ByteArray, w: Int, h: Int): Bitmap {
            val px = IntArray(w * h) { i ->
                val o = i * 3
                (0xFF shl 24) or ((bgr[o + 2].toInt() and 0xFF) shl 16) or ((bgr[o + 1].toInt() and 0xFF) shl 8) or (bgr[o].toInt() and 0xFF)
            }
            return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
        }
    }
}
