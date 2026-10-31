package com.cgjnkim.mobileokad.flir

import android.content.Context
import android.hardware.usb.UsbManager
import android.util.Log
import java.util.concurrent.FutureTask
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The FLIR BFS-U3-51S5PC driven the way the PC penvmap capture drives it.
 *
 * Same camera state as `hardware/sensor/flir.py` (the PC capture, verified): AdcBitDepth Bit12 before
 * the pixel format, BayerRG16 for the brackets, ExposureAuto / GainAuto / BalanceWhiteAuto Off, gain
 * 0, BalanceRatio 1, ExposureMode Timed, GammaEnable off, BlackLevel 0, and BlackLevelClamping /
 * DefectCorrectStatic left alone unless asked. Brackets use the hardware exposure sequencer with
 * chunk-verified frames ([SequencerProgram], [BracketCheck]) and are saved by the app byte for byte
 * as the camera sent them (little-endian uint16, 12-bit MSB aligned -- what Spinnaker gives the PC).
 *
 * The preview streams BayerRG8 (half the bytes) and is only for framing and exposure; captures
 * switch to BayerRG16 and back.
 *
 * Threading: the USB control channel takes one transaction at a time, so one worker thread owns the
 * camera. Commands are queued to it; when idle it grabs preview frames. Every public method may be
 * called from any thread; the blocking ones must not be called on the main thread.
 */
class PenvmapCamera : AutoCloseable {

    data class Options(
        /** BlackLevelClampingEnable: null leaves the camera's value (PC default). */
        val blackLevelClamping: Boolean? = null,
        /** DefectCorrectStaticEnable: null leaves the camera's value (PC default). */
        val defectCorrectStatic: Boolean? = null,
        /** Preview frame-rate cap. Frames the host cannot collect queue in the camera and lag. */
        val previewFps: Double = 4.0,
        val bracketRetries: Int = 2,
        /** Frame timeout = exposure + this (PC: GetNextImage(exposure + 3000 ms)). */
        val frameTimeoutMarginMs: Int = 3000,
    )

    /** A preview frame: the raw BayerRG8 mosaic, sensor orientation. */
    class Preview(
        val pixels: ByteArray,
        val width: Int,
        val height: Int,
        val exposureUs: Long,
        val frameId: Long,
    ) {
        /** Full-scale value of [pixels] (8-bit preview). */
        val fullScale: Int get() = 255
    }

    /**
     * One verified bracket. [frames] are the BayerRG16 images exactly as transferred: width*height
     * little-endian uint16 each, sensor orientation.
     */
    class Bracket(
        val frames: List<ByteArray>,
        val width: Int,
        val height: Int,
        /** Exposure each frame was taken with, from its chunk (µs). */
        val exposureUs: List<Double>,
        /** Same keys as the PC's BracketResult.info, plus a few transport facts. */
        val info: Map<String, Any?>,
    )

    fun interface PreviewListener {
        fun onPreview(frame: Preview)
    }

    @Volatile var lastError: String = ""
        private set

    /** Model, serial, firmware / device version, USB link speed. */
    @Volatile var deviceInfo: Map<String, String> = emptyMap()
        private set

    /** What the open-time preset did, per setting ("ok" or why not). */
    @Volatile var presetOutcomes: Map<String, String> = emptyMap()
        private set

    @Volatile var exposureRangeUs: LongRange = DEFAULT_EXPOSURE_RANGE
        private set

    @Volatile var sequencerCapacity: Int = 0
        private set

    /** How the last bracket's chunk data was laid out ("little-endian tags", "big-endian tags", "unparsed"). */
    @Volatile var chunkLayout: String = "unknown"
        private set

    /** Last 256 bytes of a payload whose chunks could not be parsed, for diagnosing the layout. */
    @Volatile var lastPayloadTail: ByteArray? = null
        private set

    val isOpen: Boolean get() = open.get()

    private var control: U3vControl? = null
    private var camera: FlirCamera? = null
    private var nodes: NodeAccess? = null
    private var sequencer: SequencerProgram? = null
    private var options = Options()

    private val open = AtomicBoolean(false)
    private val running = AtomicBoolean(false)
    private val tasks = LinkedBlockingQueue<Runnable>()
    private var worker: Thread? = null

    // Worker-thread state.
    @Volatile private var previewWanted = false
    @Volatile private var previewExposureUs = DEFAULT_PREVIEW_EXPOSURE_US
    @Volatile private var listener: PreviewListener? = null
    private var previewStreaming = false
    private var previewFailures = 0
    private var fullWidth = 0
    private var fullHeight = 0

    // ---- lifecycle ------------------------------------------------------------------------------

    /**
     * Opens the first FLIR camera on the bus and applies the PC preset. Blocking.
     * USB permission must already be granted ([FlirProbe.requestPermission]). Throws with the reason.
     */
    fun open(context: Context, options: Options = Options()) {
        if (open.get()) return
        this.options = options
        val device = FlirProbe.findDevice(context) ?: throw U3vException("no FLIR camera on the bus")
        val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        if (!manager.hasPermission(device)) throw U3vException("USB permission not granted")
        val connection = manager.openDevice(device) ?: throw U3vException("openDevice failed")
        val ctl = U3vControl.open(device, connection)
        try {
            ctl.configure()
            val gc = GenicamLoader.load(ctl) ?: throw U3vException("could not read the GenICam XML")
            val cam = FlirCamera(ctl, gc)
            val na = NodeAccess(ctl, gc)
            val seq = SequencerProgram(na)
            cam.resetStream()
            control = ctl
            camera = cam
            nodes = na
            sequencer = seq
            deviceInfo = linkedMapOf(
                "model" to ctl.readString(Abrm.MODEL_NAME),
                "serial" to ctl.readString(Abrm.SERIAL_NUMBER),
                "device_version" to ctl.readString(Abrm.DEVICE_VERSION),
                "manufacturer" to ctl.readString(Abrm.MANUFACTURER_NAME),
                "usb_link" to runCatching { FlirProbe.describeSpeed(ctl.readU32(ctl.sbrmAddress + Sbrm.CURRENT_SPEED)) }.getOrDefault("?"),
            )
            applyPreset(cam, na, seq)
            exposureRangeUs = runCatching { na.readInt("ExposureTime_Min")..na.readInt("ExposureTime_Max") }
                .getOrDefault(DEFAULT_EXPOSURE_RANGE)
            sequencerCapacity = seq.capacity()
            fullWidth = cam.widthMax
            fullHeight = cam.heightMax
            // Binning or decimation left on by another program would shrink the frame and mix analysers.
            // They are not addressable here (selector-computed); a power cycle restores the defaults.
            if (deviceInfo["model"].orEmpty().contains("51S5") && (fullWidth != SENSOR_WIDTH || fullHeight != SENSOR_HEIGHT)) {
                throw U3vException(
                    "camera reports ${fullWidth}x$fullHeight, not ${SENSOR_WIDTH}x$SENSOR_HEIGHT: binning or decimation is on. " +
                        "Unplug the camera for a few seconds (resets it) and reconnect.",
                )
            }
            cam.setRoi(fullWidth, fullHeight)
        } catch (e: Exception) {
            lastError = e.message ?: e.toString()
            runCatching { ctl.close() }
            control = null
            camera = null
            throw e
        }
        open.set(true)
        running.set(true)
        worker = Thread(::workerLoop, "penvmap-flir").apply { start() }
        Log.i(TAG, "opened ${deviceInfo["model"]} ${deviceInfo["serial"]} ${fullWidth}x$fullHeight, sequencer $sequencerCapacity sets")
    }

    override fun close() {
        if (!open.getAndSet(false)) return
        running.set(false)
        worker?.interrupt()
        runCatching { worker?.join(3000) }
        worker = null
        runCatching { sequencer?.restore(null) }
        runCatching { camera?.close() }
        camera = null
        control = null
        nodes = null
        sequencer = null
    }

    // ---- preview --------------------------------------------------------------------------------

    /** Starts or stops the preview stream. Frames go to [listener] on the camera thread. */
    fun setPreview(active: Boolean, exposureUs: Long, listener: PreviewListener?) {
        this.listener = listener
        previewExposureUs = exposureUs
        previewWanted = active
        if (active) tasks.offer(Runnable { applyPreviewExposure() })
    }

    fun setPreviewExposure(exposureUs: Long) {
        previewExposureUs = exposureUs
        tasks.offer(Runnable { applyPreviewExposure() })
    }

    // ---- bracket --------------------------------------------------------------------------------

    /**
     * Captures one verified bracket of [exposuresUs] (µs) with the hardware sequencer. Blocking.
     * Brackets longer than the sequencer capacity are taken in consecutive segments; a failed
     * verification retries the whole bracket (PC behaviour). The preview resumes afterwards.
     */
    fun captureBracket(exposuresUs: List<Long>): Bracket = call { bracketOnWorker(exposuresUs) }

    /** The camera's current raw-relevant settings, PC `read_settings` keys plus transport facts. */
    fun readSettings(): Map<String, Any?> = call { settingsOnWorker() }

    // ---- worker ---------------------------------------------------------------------------------

    private fun <T> call(block: () -> T): T {
        check(open.get()) { "camera is not open" }
        val task = FutureTask(java.util.concurrent.Callable { block() })
        tasks.put(task)
        try {
            return task.get()
        } catch (e: java.util.concurrent.ExecutionException) {
            throw e.cause ?: e
        }
    }

    private fun workerLoop() {
        while (running.get()) {
            val task = try {
                tasks.poll(if (previewWanted) 0 else IDLE_POLL_MS, TimeUnit.MILLISECONDS)
            } catch (e: InterruptedException) {
                break
            }
            if (task != null) {
                task.run()
                continue
            }
            if (previewWanted) grabPreview() else if (previewStreaming) stopPreviewStream()
        }
        runCatching { stopPreviewStream() }
    }

    private fun startPreviewStream() {
        val cam = camera ?: return
        val na = nodes ?: return
        cam.stopAcquisition()
        cam.drainStream()
        sequencer?.restore(previewExposureUs)
        cam.setFrameRateLimit(null)
        na.setEnum("PixelFormat", PREVIEW_FORMAT)
        na.optional { na.setEnum("AcquisitionMode", "Continuous") }
        na.optional { na.writeInt("ExposureTime_Val", previewExposureUs, verify = false) }
        cam.setFrameRateLimit(options.previewFps)?.let { Log.i(TAG, "preview frame-rate cap not applied: $it") }
        cam.startAcquisition()
        previewStreaming = true
    }

    private fun stopPreviewStream() {
        val cam = camera ?: return
        if (!previewStreaming) return
        previewStreaming = false
        cam.stopAcquisition()
        cam.drainStream()
    }

    private fun applyPreviewExposure() {
        val na = nodes ?: return
        na.optional { na.writeInt("ExposureTime_Val", previewExposureUs, verify = false) }
    }

    private fun grabPreview() {
        val cam = camera ?: return
        try {
            if (!previewStreaming) startPreviewStream()
            val timeout = (previewExposureUs / 1000).toInt() + PREVIEW_TIMEOUT_MARGIN_MS
            val frame = cam.grabFrame(timeout)
            previewFailures = 0
            val bytes = frame.width * frame.height
            // With chunk mode on the payload carries chunks too; take the image chunk when it is found.
            val image = ChunkData.parse(frame)?.chunks?.values?.firstOrNull { it.second == bytes }
            val pixels = when {
                frame.data.size == bytes -> frame.data
                image != null -> frame.data.copyOfRange(image.first, image.first + bytes)
                else -> frame.data.copyOf(bytes)
            }
            listener?.onPreview(Preview(pixels, frame.width, frame.height, previewExposureUs, frame.frameId))
        } catch (e: Exception) {
            lastError = e.message ?: e.toString()
            Log.w(TAG, "preview frame failed (${++previewFailures})", e)
            runCatching { stopPreviewStream() }
            previewStreaming = false
            if (previewFailures == 3) errorListener?.invoke("preview keeps failing: $lastError (cable / power?)")
            // Never let an interrupt (close()) escape the worker as an uncaught exception.
            if (previewFailures > 3) runCatching { Thread.sleep(500) }
        }
    }

    /** Told once when the preview keeps failing (camera unplugged, cable, power). Called on the camera thread. */
    @Volatile var errorListener: ((String) -> Unit)? = null

    private fun bracketOnWorker(exposuresUs: List<Long>): Bracket {
        val cam = camera ?: throw U3vException("camera is not open")
        val na = nodes ?: throw U3vException("camera is not open")
        val seq = sequencer ?: throw U3vException("camera is not open")
        val capacity = sequencerCapacity
        if (capacity < 2) throw U3vException("camera has no usable exposure sequencer")
        val segments = BracketCheck.segments(exposuresUs, capacity)
        stopPreviewStream()
        var lastFailure: Exception? = null
        try {
            for (attempt in 0..options.bracketRetries) {
                try {
                    val t0 = System.nanoTime()
                    chunkLayout = "unknown"
                    val frames = ArrayList<ByteArray>()
                    val actual = ArrayList<Double>()
                    val frameIds = ArrayList<Long>()
                    val sets = ArrayList<Int>()
                    for (segment in segments) {
                        val seg = segmentOnWorker(cam, na, seq, segment)
                        frames += seg.frames
                        actual += seg.exposureUs
                        frameIds += seg.frameIds
                        sets += seg.sets
                    }
                    val info = linkedMapOf<String, Any?>(
                        "mode" to "sequencer",
                        "segments" to segments.size,
                        "retries" to attempt,
                        "frame_ids" to frameIds,
                        "sequencer_sets" to sets,
                        "requested_exposure_us" to exposuresUs.map { it.toDouble() },
                        "bracket_time_s" to Math.round((System.nanoTime() - t0) / 1e6) / 1000.0,
                        "transport" to "android-u3v",
                        // false: chunks could not be parsed; frame ids are USB block ids and the
                        // exposures are the programmed ones (not read back from the frames).
                        "chunk_verified" to (chunkLayout != "unparsed"),
                        "chunk_layout" to chunkLayout,
                    )
                    return Bracket(frames, fullWidth, fullHeight, actual, info)
                } catch (e: Exception) {
                    lastFailure = e
                    Log.w(TAG, "sequencer bracket failed (attempt ${attempt + 1}/${options.bracketRetries + 1})", e)
                    runCatching { cam.stopAcquisition(); cam.drainStream() }
                    runCatching { seq.restore(null) }
                }
            }
            throw U3vException("sequencer bracket failed after ${options.bracketRetries + 1} attempts: ${lastFailure?.message}")
        } finally {
            runCatching { seq.restore(previewExposureUs) }
            // The preview stream is restarted by the worker loop on its next idle turn.
        }
    }

    private class Segment(val frames: List<ByteArray>, val exposureUs: List<Double>, val frameIds: List<Long>, val sets: List<Int>)

    private fun segmentOnWorker(cam: FlirCamera, na: NodeAccess, seq: SequencerProgram, exposures: List<Long>): Segment {
        cam.stopAcquisition()
        cam.drainStream()
        cam.setFrameRateLimit(null)
        // A previous segment leaves the sequencer On, and payload settings are locked while it is.
        na.optional { na.setEnumOrCurrent("SequencerMode", "Off") }
        if (na.enumEntry("PixelFormat") != CAPTURE_FORMAT) na.setEnum("PixelFormat", CAPTURE_FORMAT)
        val programmed = seq.program(exposures)
        val imageBytes = fullWidth * fullHeight * 2
        val frames = ArrayList<ByteArray>()
        val actual = ArrayList<Double>()
        val ids = ArrayList<Long>()
        val sets = ArrayList<Int>()
        cam.startAcquisition()
        try {
            for ((i, us) in programmed.withIndex()) {
                val frame = cam.grabFrame((us / 1000).toInt() + options.frameTimeoutMarginMs)
                if (frame.width != fullWidth || frame.height != fullHeight) {
                    throw U3vException("frame $i is ${frame.width}x${frame.height}, expected ${fullWidth}x$fullHeight")
                }
                val chunks = ChunkData.parse(frame)
                if (chunks == null) {
                    // Chunk layout not understood: keep the frame, but say so. FrameID continuity is
                    // still checked from the USB3 Vision leader; exposure and set are the programmed ones.
                    if (chunkLayout != "unparsed") {
                        chunkLayout = "unparsed"
                        lastPayloadTail = frame.data.copyOfRange(maxOf(0, frame.data.size - 256), frame.data.size)
                        Log.w(TAG, "frame $i: chunk data not parsable (payload ${frame.data.size} B, image $imageBytes B)")
                    }
                    if (frame.data.size < imageBytes) throw U3vException("frame $i payload ${frame.data.size} B < image $imageBytes B")
                    ids.lastOrNull()?.let { prev ->
                        if (frame.frameId != prev + 1) throw U3vException("frame loss: block id $prev -> ${frame.frameId}")
                    }
                    frames += if (frame.data.size == imageBytes) frame.data else frame.data.copyOf(imageBytes)
                    actual += us.toDouble()
                    ids += frame.frameId
                    sets += i
                    continue
                }
                chunkLayout = if (chunks.bigEndianTags) "big-endian tags" else "little-endian tags"
                val fid = chunks.frameId ?: throw U3vException("frame $i has no FrameID chunk")
                val exp = chunks.exposureUs?.toDouble() ?: throw U3vException("frame $i has no ExposureTime chunk")
                val set = chunks.sequencerSet ?: throw U3vException("frame $i has no SequencerSetActive chunk")
                BracketCheck.check(i, fid, ids.lastOrNull(), exp, us.toDouble(), set)?.let { throw U3vException(it) }
                frames += imageOf(frame.data, chunks, imageBytes)
                actual += exp
                ids += fid
                sets += set
            }
        } finally {
            cam.stopAcquisition()
            cam.drainStream()
        }
        return Segment(frames, actual, ids, sets)
    }

    /** The image chunk: the one whose length is exactly one BayerRG16 frame. */
    private fun imageOf(payload: ByteArray, chunks: ChunkData, imageBytes: Int): ByteArray {
        val entry = chunks.chunks.entries.firstOrNull { it.value.second == imageBytes }
            ?: throw U3vException("no image chunk of $imageBytes bytes (chunks ${chunks.chunks.values.map { it.second }})")
        val (offset, length) = entry.value
        return if (offset == 0 && payload.size == length) payload else payload.copyOfRange(offset, offset + length)
    }

    private fun applyPreset(cam: FlirCamera, na: NodeAccess, seq: SequencerProgram) {
        val outcomes = LinkedHashMap<String, String>()
        fun step(what: String, required: Boolean = false, action: () -> Unit) {
            try {
                action()
                outcomes[what] = "ok"
            } catch (e: Exception) {
                outcomes[what] = e.message ?: e.toString()
                if (required) throw U3vException("$what: ${e.message}")
                Log.i(TAG, "preset $what skipped: ${e.message}")
            }
        }
        step("SequencerOff") { seq.restore(null) }
        step("AcquisitionFrameRateEnable=off") { na.setBool("AcquisitionFrameRateEnable", false) }
        step("ExposureAuto=Off", required = true) { na.setEnum("ExposureAuto", "Off") }
        step("GainAuto=Off") { na.setEnum("GainAuto", "Off") }
        step("Gain=0") { cam.gainDb = 0.0 }
        step("BalanceWhiteAuto=Off") { na.setEnum("BalanceWhiteAuto", "Off") }
        step("BalanceRatio=1") { cam.setBalanceRatiosToOne()?.let { throw U3vException(it) } }
        step("ExposureMode=Timed") { na.setEnum("ExposureMode", "Timed") }
        step("GammaEnable=off") { na.setBool("GammaEnable", false) }
        step("IspEnable=off") { na.setBool("IspEnable", false) }
        step("BlackLevel=0") { cam.setBlackLevelToZero()?.let { throw U3vException(it) } }
        options.blackLevelClamping?.let { v -> step("BlackLevelClampingEnable=$v") { na.setBool("BlackLevelClampingEnable", v) } }
        options.defectCorrectStatic?.let { v -> step("DefectCorrectStaticEnable=$v") { na.setBool("DefectCorrectStaticEnable", v) } }
        // Geometry as on the PC (BayerRG16 2448 x 2048, nothing flipped). Another app may have left the image
        // reversed, which mirrors the mosaic and relabels every analyser angle. (Binning / decimation sit
        // behind selector-computed addresses and are checked through WidthMax / HeightMax in open().)
        for (node in listOf("ReverseX", "ReverseY")) {
            step("$node=off") { if (na.readBool(node) != false) na.setBool(node, false) }
        }
        step("AdcBitDepth=Bit12", required = true) { na.setEnum("AdcBitDepth", "Bit12") }
        step("PixelFormat=$CAPTURE_FORMAT", required = true) { na.setEnum("PixelFormat", CAPTURE_FORMAT) }
        step("ExposureTime=$DEFAULT_PREVIEW_EXPOSURE_US") { na.writeInt("ExposureTime_Val", DEFAULT_PREVIEW_EXPOSURE_US, verify = false) }
        step("Chunks") { seq.enableChunks() }
        for ((k, v) in seq.skipped) outcomes["skipped:$k"] = v
        presetOutcomes = outcomes
    }

    private fun settingsOnWorker(): Map<String, Any?> {
        val cam = camera ?: return emptyMap()
        val na = nodes ?: return emptyMap()
        val s = LinkedHashMap<String, Any?>()
        // PixelFormat as captured: the preview streams 8-bit, every saved bracket is CAPTURE_FORMAT.
        s["PixelFormat"] = CAPTURE_FORMAT
        for (name in listOf("AdcBitDepth", "ExposureAuto", "GainAuto", "BalanceWhiteAuto")) s[name] = na.enumEntry(name)
        val ratios = runCatching { cam.balanceRatios() }.getOrDefault(emptyMap())
        s["BalanceRatioRed"] = ratios["Red"]?.let { Math.round(it * 1e4) / 1e4 }
        s["BalanceRatioBlue"] = ratios["Blue"]?.let { Math.round(it * 1e4) / 1e4 }
        s["BlackLevel"] = na.readIntOrNull("BlackLevelAll_Val")?.toDouble()
        s["Gain"] = runCatching { Math.round(cam.gainDb * 1e3) / 1e3 }.getOrNull()
        for (name in listOf("GammaEnable", "BlackLevelClampingEnable", "DefectCorrectStaticEnable")) s[name] = na.readBool(name)
        s["IspEnable"] = na.readBool("IspEnable")
        s["ExposureMode"] = na.enumEntry("ExposureMode")
        s["DeviceLinkThroughputLimit"] = na.readIntOrNull("DeviceLinkThroughputLimit")
        s["SequencerCapacity"] = sequencerCapacity
        s["PreviewPixelFormat"] = PREVIEW_FORMAT
        s["Transport"] = "android-u3v"
        return s
    }

    companion object {
        private const val TAG = "PenvmapCamera"
        const val CAPTURE_FORMAT = "BayerRG16"
        const val PREVIEW_FORMAT = "BayerRG8"
        /** BFS-U3-51S5PC sensor, what the PC saves. */
        const val SENSOR_WIDTH = 2448
        const val SENSOR_HEIGHT = 2048
        const val DEFAULT_PREVIEW_EXPOSURE_US = 26_280L
        val DEFAULT_EXPOSURE_RANGE = 21L..30_000_000L
        private const val IDLE_POLL_MS = 200L
        private const val PREVIEW_TIMEOUT_MARGIN_MS = 2000
    }
}
