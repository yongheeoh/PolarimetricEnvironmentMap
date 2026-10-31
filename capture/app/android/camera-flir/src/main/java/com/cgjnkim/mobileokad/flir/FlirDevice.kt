package com.cgjnkim.mobileokad.flir

import android.content.Context
import android.graphics.Bitmap
import android.hardware.usb.UsbManager
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * A FLIR polarization camera as a capture device.
 *
 * The surface deliberately mirrors OakDevice's so that both cameras can later sit behind
 * one contract without either having to be reshaped. A worker thread keeps the newest
 * decoded frame parked; the UI polls it at display rate, so a slow consumer drops frames
 * instead of accumulating latency.
 *
 * USB permission is not requested from here -- call [FlirProbe.requestPermission] first.
 */
class FlirDevice : AutoCloseable {

    /**
     * How much of the sensor's resolution a capture keeps.
     *
     * The sensor is read out in full either way: the camera's own binning is left alone
     * because what it does to a polarization mosaic is undocumented, so this trades
     * resolution for signal after the frame arrives rather than bandwidth before it. See
     * [Mosaic.binnedSum].
     */
    enum class CaptureResolution(val label: String, val factor: Int) {
        /** Every super-pixel the sensor has: 612 x 512 Stokes pixels. */
        FULL("Full", 1),

        /**
         * Each 2x2 block of super-pixels summed: 306 x 256, four times the counts.
         *
         * Twice the signal to noise, so the DoLP noise floor halves -- measured on a real
         * capture, the darkest tenth of the frame fell from 0.073 to 0.046. Worth it when
         * the scene is dim and the polarization matters more than the detail.
         */
        HALF("Half (2x2 sum)", 2);

        companion object {
            val LABELS: List<String> = entries.map { it.label }
        }
    }

    /** One rendered Stokes view, ready to be shown or written as a PNG. */
    class RenderedView(val kind: String, val label: String, val width: Int, val height: Int, val argb: IntArray)

    /**
     * Everything one shutter press produces.
     *
     * The raw mosaic is kept alongside the derived Stokes on purpose: the analyser angle
     * assignment in [StokesDecoder] is not yet confirmed against hardware, and without
     * the mosaic a capture could not be re-derived if it turns out to need correcting.
     *
     * The Stokes arrays themselves are not carried out of here. They were, as a .npy, and
     * that file cost 10.8 MB of a 23.3 MB capture -- more than the mosaic it came from,
     * because it stored 12-bit numbers as float32. Decoding the mosaic reproduces it bit
     * for bit, so the viewer does that instead. [StokesImage.toNpy] remains for anything
     * that wants to export one.
     */
    class Capture(
        val views: List<RenderedView>,
        /** One unsigned sample per sensor pixel, still in the mosaic, 0..[rawMaxValue]. */
        val rawSamples: ShortArray,
        val rawWidth: Int,
        val rawHeight: Int,
        val rawMaxValue: Int,
        /** Human readable mosaic description, stored inside the raw file itself. */
        val rawDescription: String,
        val metadataJson: String,
        /**
         * When the sensor was integrating, on [System.nanoTime].
         *
         * Carried out of the module because only the app can say what else was happening
         * then -- where the phone was pointing, how hard it was being shaken -- and it
         * cannot ask about an interval it has not been told. Start is the trigger, which
         * for a free-running sensor is the instant integration begins; end is that plus
         * the exposure the camera reported for the frame.
         */
        val exposureStartHostNs: Long,
        val exposureEndHostNs: Long,
    )

    /**
     * Serialises the stream between the grab loop and a capture.
     *
     * A capture reconfigures the ROI, which means stopping and restarting acquisition;
     * the grab loop must not be halfway through a frame while that happens.
     */
    private val streamLock = ReentrantLock()

    /**
     * Relates the camera's clock to this host's, taken once at open.
     *
     * Without it a frame's device timestamp is a number on an unrelated clock. With it
     * the moment the sensor actually exposed becomes comparable to anything else on the
     * phone, which is the only way the interval between two cameras can be measured
     * rather than assumed.
     */
    @Volatile
    private var clock: FlirCamera.ClockOffset? = null

    /** Null when the camera's gain registers did not verify. */
    @Volatile
    var gainRangeDb: ClosedFloatingPointRange<Double>? = null
        private set

    private var fullWidth = 0
    private var fullHeight = 0

    /**
     * Frames per second the camera is allowed to produce for the preview.
     *
     * Derived from the streamed size rather than fixed, because it is not a taste setting: it is the
     * rate the bus can carry, and a camera allowed to exceed it is the direct cause of
     * preview latency. Acquisition is free-running, so frames the host cannot collect
     * queue up inside the camera and come out in order -- what reaches the screen is then
     * not the newest frame but the oldest one still waiting. At a 10 fps cap against a bus
     * that could only deliver 3.2, that backlog was the half second of lag.
     *
     * Reading *less* would make it worse, not better. The queue only stops growing when
     * the camera stops filling it, so the lever is production.
     */
    @Volatile
    private var previewFpsCap = PREVIEW_FPS

    /**
     * Per-frame bus throughput, for the window between two retunes.
     *
     * Only the grab thread touches it, which is also the only thread that writes the cap.
     */
    private val busRates = mutableListOf<Double>()

    /** The decode's share of the render thread's time, for the report line. */
    private val decodeMsTotal = java.util.concurrent.atomic.AtomicLong()

    /** Leader-to-screen, summed over the report window. */
    private val latencyMsTotal = java.util.concurrent.atomic.AtomicLong()

    private var camera: FlirCamera? = null
    private var genicam: Genicam? = null
    private var worker: Thread? = null

    private val open = AtomicBoolean(false)
    private val running = AtomicBoolean(false)
    private val previewSource = AtomicInteger(0)

    /**
     * Whether the AoLP views fade toward grey where there is little polarization.
     *
     * Applies to the preview and to what a capture writes, so a stored AoLP image is
     * what was on screen when the shutter was pressed. The Stokes arrays and the raw
     * mosaic are untouched either way -- this only decides how the angle is drawn.
     */
    @Volatile
    var modulateAolpByDolp: Boolean = true

    /**
     * Whether DoLP has the sensor's own noise subtracted before it is shown.
     *
     * A magnitude built from noisy components can only be too large, so an unpolarized
     * dark frame still reads a DoLP of 0.07 to 0.09 on this camera. [StokesNoise] fits
     * the noise on each frame from the disagreement between its two green cells and the
     * Wardle-Kronberg estimator removes it. Applies to the preview and to what a capture
     * writes as an image; the Stokes arrays and the raw mosaic are stored uncorrected
     * either way, and the fit is recorded in the metadata so the correction can be
     * repeated or undone later.
     */
    @Volatile
    var debiasDolp: Boolean = true

    /**
     * How many frames one shutter press takes, keeping the sharpest.
     *
     * One disables it entirely. Above one, the whole burst is a single acquisition, so
     * the cost of the extra frames is their transfer alone -- about 175 ms each at full
     * resolution on this bus.
     */
    @Volatile
    var burstFrames: Int = 1

    /**
     * Turns the burst on or off, returning whether it is now on.
     *
     * The frame count is this module's business -- it follows from the transfer rate and
     * from how much a fourth frame is still worth -- so the caller flips a switch rather
     * than naming a number.
     */
    fun toggleBurst(): Boolean {
        burstFrames = if (burstFrames <= 1) BURST_FRAMES else 1
        return burstFrames > 1
    }

    /**
     * Whether a burst is switched on but this exposure would suppress it anyway.
     *
     * [burstCountFor] already falls back to a single frame past a certain shutter, and
     * without this the UI would go on claiming a burst that is not going to be taken --
     * the same silent no-op the AUTO control used to be. Takes the exposure rather than
     * reading it, because the caller has it on screen already and reading it here would
     * put a USB round trip behind a label that refreshes at display rate.
     */
    fun burstSuppressedAt(exposureUs: Long): Boolean =
        burstFrames > 1 && exposureUs > BURST_MAX_EXPOSURE_US

    /**
     * Which frame of the last burst was kept, counted from zero.
     *
     * Zero both when the first frame won and when there was no burst, which is right:
     * both mean the trigger instant describes the frame that was kept.
     */
    @Volatile
    private var burstKeptIndex = 0

    /** Every frame's sharpness from the last burst, for the metadata. Null when off. */
    @Volatile
    private var burstScores: FloatArray? = null

    /**
     * Whether the preview sums the super-pixels the step would otherwise discard.
     *
     * Four times the signal at the same output size and the same bandwidth, without
     * touching the exposure or the gain -- the frame has already crossed the bus and
     * three quarters of it is currently thrown away. Off by default because the extra
     * signal is not scaled back down, so a bright scene clips: it is a switch for dark
     * scenes, and a demonstration of what the capture path could also do.
     *
     * Preview only. Nothing stored changes.
     *
     * Nothing turns it on any more: the control was taken off the screen, because a
     * preview that clips where the sensor did not is the opposite of what the pill beside
     * it now says. [MosaicPreview] keeps the summing path and its tests, so restoring the
     * switch is a line of layout, but the capability is dormant by decision rather than
     * by accident.
     */
    @Volatile
    var previewSumming: Boolean = false

    /**
     * Where AUTO aims, in stops away from a mid-grey.
     *
     * Only the host meter reads it -- a manual exposure is already the number the user
     * chose, and moving it by a compensation would be moving the same thing twice.
     *
     * Snapped to the third-stop lattice by the dial that sets it, the same lattice the
     * shutter clicks through, so a stop of compensation and a stop of shutter are the
     * same size of step.
     */
    @Volatile
    var exposureCompensationEv: Double = 0.0

    /**
     * Whether the preview stripes the pixels whose raw samples have clipped.
     *
     * Off by default: it is a check, not a way to look at a scene. See
     * [MosaicPreview.stripeClipped] for what it tests and why the test is on the raw
     * samples rather than on what ends up on screen.
     */
    @Volatile
    var previewZebra: Boolean = false

    /**
     * Resolution a capture is stored at. The preview is unaffected either way.
     *
     * Full by default: binning cannot be undone, and a capture that kept the detail can
     * always be binned later from its own archive file, while one that did not cannot be
     * un-binned.
     */
    @Volatile
    var captureResolution: CaptureResolution = CaptureResolution.FULL

    /**
     * Whether anything is looking at this camera's preview.
     *
     * Two cameras share one USB 2.0 bus, so a camera nobody is watching is not merely
     * wasting CPU on frames that are discarded -- it is taking bandwidth from the one
     * that is being watched. Acquisition is stopped rather than just skipped, because a
     * camera left streaming into a host that never reads fills its own buffer and the
     * backlog would show up as latency the moment the preview came back.
     */
    private val previewActive = AtomicBoolean(true)

    private val frameLock = Object()

    /**
     * Double buffered so the grab thread can render the next frame while the UI is still
     * reading the last one, without either holding a lock across the work.
     */
    private var frontBuffer: IntArray? = null
    private var backBuffer: IntArray? = null
    private var previewWidth = 0
    private var previewHeight = 0
    private var frameDirty = false

    /**
     * Size of the picture handed to the UI, which is the whole sensor's worth.
     *
     * The preview streams the whole sensor, so this is the rendered size and the two
     * agree. It is kept separate from the streamed size all the same: a frame smaller
     * than this is centred in it and the rest filled with [PREVIEW_MARGIN_COLOR], which
     * is what keeps a short frame -- or any future region smaller than the sensor --
     * visible as a margin rather than silently cropped, so the preview cannot quietly
     * stop agreeing with what the shutter records.
     */
    private var displayWidth = 0
    private var displayHeight = 0

    /** The region actually streamed for the preview, in sensor pixels. */
    private var previewRoiWidth = 0
    private var previewRoiHeight = 0

    @Volatile
    private var errorText: String = ""

    @Volatile
    private var name: String = ""

    /**
     * Cached because the UI asks for it on every slider movement, and reading it means
     * two USB round trips. The bounds do not change while the session is open.
     */
    @Volatile
    private var exposureRange: LongRange = DEFAULT_EXPOSURE_RANGE

    val isOpen: Boolean get() = open.get()
    val error: String get() = errorText
    val deviceName: String get() = name

    fun previewSourceNames(): List<String> = StokesView.LABELS

    /** Starts or stops acquisition to match whether the preview is being shown. */
    fun setPreviewActive(active: Boolean) {
        if (!previewActive.compareAndSet(!active, active)) return
        val cam = camera ?: return
        worker?.let { if (!it.isAlive) return }
        try {
            streamLock.withLock {
                if (active) {
                    cam.startAcquisition()
                } else {
                    cam.stopAcquisition()
                    cam.drainStream()
                }
            }
            Log.i(TAG, if (active) "preview resumed" else "preview idle, acquisition stopped")
        } catch (e: Exception) {
            Log.w(TAG, "could not change preview state", e)
        }
    }

    fun setPreviewSource(index: Int) {
        // The next grabbed frame renders in the new view; at preview rate that is under
        // a tenth of a second, and it keeps all the rendering off the UI thread.
        previewSource.set(index.coerceIn(0, StokesView.entries.size - 1))
    }

    /**
     * Opens the camera and starts acquisition. Blocking; not for the main thread.
     *
     * Returns false and sets [error] rather than throwing, matching OakDevice.
     */
    fun open(context: Context): Boolean {
        if (open.get()) return true
        errorText = ""

        try {
            val device = FlirProbe.findDevice(context)
                ?: throw U3vException("no FLIR camera on the bus")
            val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager
            if (!manager.hasPermission(device)) throw U3vException("USB permission not granted")
            val connection = manager.openDevice(device) ?: throw U3vException("openDevice failed")

            val control = U3vControl.open(device, connection)
            control.configure()

            val gc = GenicamLoader.load(control) ?: throw U3vException("could not read the GenICam XML")
            val cam = FlirCamera(control, gc)

            // Clear whatever a previous session left behind before touching anything.
            cam.resetStream()

            name = "${control.readString(Abrm.MODEL_NAME)} ${control.readString(Abrm.SERIAL_NUMBER)}"

            // Everything below except the pixel format is a convenience. A camera that
            // refuses one of them still takes pictures, so they are logged and shrugged
            // off rather than failing the open.
            // Kept as well as logged. Whether the camera actually took a setting decides
            // what the pixels mean, and a log line is gone by the time somebody asks the
            // question of an archive months later. Every capture now carries the answer.
            val outcomes = LinkedHashMap<String, String>()
            fun optional(what: String, reason: String?) {
                outcomes[what] = reason ?: "ok"
                if (reason != null) Log.i(TAG, "skipped $what: $reason")
            }

            // TriggerMode sits behind a selector-indexed address this driver cannot
            // compute; Off is the camera's default anyway.
            optional("TriggerMode=Off", cam.applyEnum("TriggerMode", "Off"))
            optional("AcquisitionMode=Continuous", cam.applyEnum("AcquisitionMode", "Continuous"))

            // A pinned frame rate locks the exposure controls, so this has to come off
            // before ExposureAuto or ExposureTime will be accepted at all.
            optional(
                "AcquisitionFrameRateEnable=off",
                cam.applyBoolean("AcquisitionFrameRateEnable", false),
            )

            // Everything the camera could do to a pixel after the sensor has to be off.
            // Stokes parameters are differences between neighbouring analyser readings,
            // so any non-linearity or any filter that mixes neighbours corrupts them:
            // gamma bends the scale, white balance scales the bands apart, sharpening
            // and defect correction interpolate across the mosaic, and black level
            // clamping moves the zero that DoLP is measured against.
            //
            // Two of these matter more than the rest, and neither was here before:
            //
            //  - IspEnable. With the ISP on, the camera demosaics and interpolates before
            //    the payload leaves it, which mixes the 4x4 super-pixel's analyser cells
            //    into each other. There is no recovering a Stokes vector from that: the
            //    four angles a pixel is supposed to hold have already been averaged with
            //    its neighbours' angles.
            //  - FfcEnable. Flat-field correction multiplies a per-pixel gain onto the
            //    sensor. S1 and S2 are differences between neighbouring analyser
            //    readings, so neighbours carrying different multipliers put a pattern
            //    into the polarization that the scene never had.
            //
            // The rest are the same argument at smaller scale: a colour matrix mixes the
            // bands, saturation and LUT bend the scale, sharpening interpolates across
            // the mosaic, and black level clamping moves the zero DoLP is measured from.
            optional("IspEnable=off", cam.applyBoolean("IspEnable", false))
            optional("FfcEnable=off", cam.applyBoolean("FfcEnable", false))
            optional("GammaEnable=off", cam.applyBoolean("GammaEnable", false))
            optional("SharpeningEnable=off", cam.applyBoolean("SharpeningEnable", false))
            optional("SharpeningAuto=off", cam.applyBoolean("SharpeningAuto", false))
            optional("ColorTransformationEnable=off", cam.applyBoolean("ColorTransformationEnable", false))
            optional("SaturationEnable=off", cam.applyBoolean("SaturationEnable", false))
            optional("LUTEnable=off", cam.applyBoolean("LUTEnable", false))
            optional("DefectCorrectStaticEnable=off", cam.applyBoolean("DefectCorrectStaticEnable", false))
            optional("BlackLevelClampingEnable=off", cam.applyBoolean("BlackLevelClampingEnable", false))
            // Clamping off is not the same as a black level of zero -- the first stops
            // the camera adjusting the pedestal, the second removes it. A pedestal lands
            // in every sample and so in S0, and DoLP is a magnitude over S0: an offset
            // nothing knows about makes every degree of polarization read low, by a
            // factor that moves with the brightness of the scene.
            optional("BlackLevel=0", cam.setBlackLevelToZero())
            // Switching the algorithm off is not the same as neutralising it. On this body
            // white balance is a pair of Green/Red and Green/Blue ratios, and "Off" only
            // stops them being updated -- whatever ratio is loaded stays loaded, which is
            // exactly the factory default somebody would call a hidden white balance. So
            // the ratios are set to unity as well, per selector.
            //
            // Both writes may well come back unaddressable: BalanceRatio sits behind
            // BalanceRatioSelector, and this driver cannot compute selector-indexed
            // addresses. That is worth attempting anyway, because the attempt is recorded
            // and "unaddressable" is itself the finding.
            // Switching the algorithm off is not the same as neutralising it: "Off" only
            // stops the ratios being updated, and whatever is loaded stays loaded. The
            // ratios themselves are set below, through the registers, because GenICam
            // cannot reach them on this body.
            optional("BalanceWhiteAuto=Off", cam.applyEnum("BalanceWhiteAuto", "Off"))
            optional("GainAuto=Off", cam.applyEnum("GainAuto", "Off"))
            // Selector-indexed on this body, so it is reported as unaddressable and
            // skipped rather than silently missed. Off is the default.
            optional("TestPattern=Off", cam.applyEnum("TestPattern", "Off"))

            // 12 bits, because S1 and S2 are differences of similar numbers: at 8 bits
            // the quantisation step is a large fraction of the polarization signal.
            optional("AdcBitDepth=Bit12", cam.applyEnum("AdcBitDepth", "Bit12"))

            // What the camera says these nodes even are, once per open. The colour ratios
            // were the thing nobody could answer from the archive, and half of answering
            // it is knowing whether the ratio is a float register, an integer behind a
            // converter, or indexed by a selector this driver cannot address.
            // What the camera is actually holding for white balance, read straight from
            // the registers its own description points at. One means the band is
            // untouched; anything else is a colour scaled before the pixels were measured.
            // Measured on this body at 1.0986 for red and 1.8311 for blue, and the archive
            // confirms they reached the pixels: a band multiplied by k can only land on
            // one code in k, and the blue histogram of a well-exposed capture occupies
            // 0.546 of its range against green's 1.000 -- which is 1/1.8311.
            //
            // So they are set to one here rather than only recorded. GenICam cannot reach
            // them, but the registers can, and leaving them alone would go on scaling
            // every red and blue band the camera produces.
            optional("BalanceRatio=1", cam.setBalanceRatiosToOne())
            balanceRatios = runCatching { cam.balanceRatios() }.getOrDefault(emptyMap())
            for ((band, ratio) in balanceRatios) {
                Log.i(TAG, "BalanceRatio[$band] = %.4f".format(ratio))
            }

            presetOutcomes = outcomes.toMap()

            try {
                cam.pixelFormat = PREVIEW_PIXEL_FORMAT
            } catch (e: U3vException) {
                // Name the lock rather than just relaying the status code: the usual
                // cause is something else still holding the camera.
                val lock = cam.lockState("PixelFormat_Lck")
                throw U3vException(
                    "${e.message} (PixelFormat_Lck=${lock ?: "?"}). The camera may still be " +
                        "streaming for another client; unplug it or close SpinView and retry."
                )
            }
            if (cam.pixelFormat != PREVIEW_PIXEL_FORMAT) {
                throw U3vException("camera refused the preview pixel format")
            }

            // A definite, short exposure rather than auto. Auto is not offered in every
            // mode on this camera, and an exposure inherited from a previous session can
            // be seconds long -- which starves the payload transfers and looks like a
            // stream failure rather than a slow frame. Short is also what handheld
            // shooting wants, so this is the right default either way.
            val exposure = runCatching {
                cam.exposureTimeUs = DEFAULT_EXPOSURE_US
                cam.exposureTimeUs
            }.getOrElse {
                Log.w(TAG, "could not set exposure", it)
                runCatching { cam.exposureTimeUs }.getOrDefault(-1L)
            }
            exposureRange = runCatching { cam.exposureRangeUs }.getOrDefault(DEFAULT_EXPOSURE_RANGE)
            // Reading the range also verifies the derived gain address before anything
            // is written to it.
            gainRangeDb = cam.gainRangeDb
            Log.i(TAG, "gain range ${gainRangeDb?.let { "%.1f..%.1f dB".format(it.start, it.endInclusive) } ?: "unavailable"}")

            clock = cam.latchClock()
            clock?.let { Log.i(TAG, "device clock latched, uncertainty ${it.uncertaintyNs / 1000} us") }
                ?: Log.i(TAG, "camera does not report a device timestamp")
            Log.i(TAG, "exposure now $exposure us, range $exposureRange")

            // The whole sensor, for the preview as well as the capture.
            //
            // The preview used to stream a centred three-quarter crop, which bought 16/9
            // of the frame rate because the preview is transfer-bound and that crop is
            // 56 percent of the bytes. It was paid for in field of view: the camera has
            // no scaler, so the outer eighth of each side was shown as a grey margin and
            // the shot was composed through a window narrower than the shutter takes.
            //
            // Framing is worth more than cadence here. The exchange is explicit rather
            // than free -- [PREVIEW_FPS] came down with the crop, and the cap the bus
            // arithmetic produces for a full frame comes down further -- and what the
            // preview shows is now the picture the capture will contain, at the whole
            // sensor's resolution. It also puts the host meter back on the full frame,
            // which is the region This Polar Camera meters.
            fullWidth = cam.widthMax
            fullHeight = cam.heightMax
            val pitch = MosaicPreview.pitch(PREVIEW_STEP)
            cam.setRoi(fullWidth, fullHeight)
            previewRoiWidth = cam.width
            previewRoiHeight = cam.height
            displayWidth = fullWidth / pitch
            displayHeight = fullHeight / pitch
            Log.i(TAG, "preview reads ${previewRoiWidth}x$previewRoiHeight of ${fullWidth}x$fullHeight")

            // Only now that the payload format and ROI are settled: the limiter makes
            // both read-only while it is on. From the streamed region, not the sensor:
            // the cap exists to match what crosses the bus.
            previewFpsCap = previewCapFor(previewRoiWidth, previewRoiHeight)
            optional(
                "frame rate cap %.1f fps".format(previewFpsCap),
                cam.setFrameRateLimit(previewFpsCap),
            )

            // Only if something is actually going to read the frames. The app opens both
            // cameras for the shutter but previews one at a time.
            if (previewActive.get()) cam.startAcquisition()

            camera = cam
            genicam = gc
            open.set(true)
            running.set(true)
            worker = Thread(::workerLoop, "flir-grab").apply { start() }
            renderThread = Thread(::renderLoop, "flir-render").apply { start() }
            Log.i(TAG, "opened $name at ${cam.width}x${cam.height}")
            return true
        } catch (e: Exception) {
            errorText = e.message ?: e.toString()
            Log.e(TAG, "open failed", e)
            closeCamera()
            return false
        }
    }

    // ---- exposure -----------------------------------------------------------

    val exposureRangeUs: LongRange get() = exposureRange

    /**
     * The noise the views are corrected with, kept in step with the analog gain.
     *
     * Not measured per frame. Shot noise per count is a property of the sensor, and
     * [StokesNoise.measure] cannot separate it from whatever the picture is doing over
     * the 2.8 px it measures across -- on a bright street it reads sixteen times the
     * truth and the correction then erases a third of the frame's real polarization.
     * Cached rather than read per frame because the gain lives in a USB register and the
     * preview loop must not touch one.
     */
    @Volatile
    private var correction: StokesNoise = StokesNoise.calibrated()

    /**
     * Register access is serialised on [streamLock], not because a capture is in
     * progress but because there is only one control channel and it cannot be shared.
     *
     * [U3vControl] says so itself -- one transaction at a time -- and it means it: the
     * GenCP request id is a plain counter, so two threads part way through their own
     * transactions read each other's acknowledgements and both get the wrong answer.
     * Three callers reach these setters now (the dial from the UI's worker, the host
     * meter from the grab thread, and the capture path), so the lock the capture already
     * held has to cover the others too.
     */
    var gainDb: Double
        get() = streamLock.withLock { camera?.runCatching { gainDb }?.getOrNull() } ?: 0.0
        set(value) = streamLock.withLock {
            camera?.runCatching { gainDb = value }?.onFailure { Log.w(TAG, "gain write failed", it) }
            correction = StokesNoise.calibrated(gainDb, fullScale = PREVIEW_FULL_SCALE)
        }

    var exposureTimeUs: Long
        get() = streamLock.withLock { camera?.runCatching { exposureTimeUs }?.getOrNull() } ?: 0L
        set(value) = streamLock.withLock {
            camera?.runCatching { exposureTimeUs = value }
                ?.onFailure { Log.w(TAG, "exposure write failed", it) }
            Unit
        }

    /**
     * Returns null when applied, or a sentence saying why it was not.
     *
     * Worth showing rather than logging: this camera does not offer continuous auto
     * exposure in the polarized formats, so an AUTO control that silently does nothing
     * would look broken. The reason names the camera's own objection.
     */
    fun setExposureAuto(on: Boolean): String? =
        streamLock.withLock {
            camera?.runCatching { setExposureAuto(on) }
                ?.onFailure { Log.w(TAG, "exposure auto failed", it) }
                ?.getOrElse { it.message ?: "exposure auto failed" }
                ?: "device closed"
        }

    // ---- host auto exposure -------------------------------------------------

    /**
     * Metering done here rather than in the camera, because this camera will not.
     *
     * [setExposureAuto] is the camera's own algorithm and it is unavailable in the
     * polarized pixel formats -- the AUTO control was a button that reported a refusal.
     * Turning this on runs [AutoExposure] against the preview instead. See that file for
     * why it meters the highlights.
     *
     * The preview has to be running: the meter reads the frames the preview produces,
     * and there is nothing else to read.
     *
     * Volatile because the three threads involved are all different: the UI flips it, the
     * render thread reads it to decide whether to meter, and the grab thread reads it to
     * decide whether to write. Without it a meter switched on from the UI can go
     * unnoticed by both of the threads that were supposed to run it.
     */
    @Volatile
    var autoExposure: Boolean = false
        set(on) {
            if (field == on) return
            field = on
            aeReading = null
            aeAppliedAtFrameId = -1L
            if (!on) {
                aeSettings = null
                return
            }
            // The camera's own auto has to be off or it fights the writes, and the
            // starting point has to be read once so the loop has something to step from.
            // Both are round trips, which is why they happen here and not per frame.
            streamLock.withLock {
                val cam = camera ?: return@withLock
                cam.setExposureAuto(false)?.let { Log.i(TAG, "ExposureAuto=Off skipped: $it") }
                aeSettings = AutoExposure.Settings(
                    exposureUs = runCatching { cam.exposureTimeUs }.getOrDefault(DEFAULT_EXPOSURE_US),
                    gainDb = runCatching { cam.gainDb }.getOrDefault(0.0),
                )
            }
            Log.i(TAG, "host auto exposure on, from $aeSettings")
        }

    /**
     * What the meter last set, or null when it is not running.
     *
     * Published so the UI can show the shutter the meter chose without a register read
     * of its own: it asks at display rate, and that is far too often for a round trip.
     */
    @Volatile
    var aeSettings: AutoExposure.Settings? = null
        private set

    /** The newest metered frame, from the render thread, waiting for the grab thread. */
    @Volatile
    private var aeReading: Pair<Long, AutoExposure.Reading>? = null

    /** Volatile because the property setter clears it from whichever thread flips AUTO. */
    @Volatile
    private var aeAppliedAtFrameId = -1L

    /**
     * Moves the frame-rate cap to what the bus has just been measured doing.
     *
     * On the grab thread, which is the one that owns the control channel, and only when
     * the change is worth a transaction -- see [PreviewCadence] for why the cap has to
     * track the link rather than being set once by hand. The OAK starting or stopping
     * beside this camera is the change this exists for: it takes a share of the bus, and
     * the preview used to keep whatever cap it was given either way -- queueing frames
     * inside the camera when the link got slower, and leaving frames unasked-for when it
     * got faster again.
     */
    private fun retune(cam: FlirCamera, bytesPerSecond: Double) {
        val payload = previewPayloadBytes(previewRoiWidth, previewRoiHeight)
        val next = PreviewCadence.capFor(bytesPerSecond, payload, PREVIEW_FPS, MIN_PREVIEW_FPS)
        if (!PreviewCadence.worthWriting(previewFpsCap, next)) return
        val failure = runCatching { cam.setFrameRateLimit(next) }.getOrElse { it.message }
        if (failure != null) {
            Log.i(TAG, "frame rate cap %.1f fps not applied: %s".format(next, failure))
            return
        }
        Log.i(
            TAG,
            "frame rate cap %.1f -> %.1f fps, bus %.0f MB/s".format(
                previewFpsCap, next, bytesPerSecond / 1e6,
            ),
        )
        previewFpsCap = next
    }

    /**
     * Applies one metering step, on the thread that owns the bus.
     *
     * The measuring happens on the render thread and the writing happens here, which is
     * not an arbitrary split. Writes go over the control channel that [U3vControl]
     * allows one transaction on at a time, and this thread is the one already
     * synchronised with the capture path; metering, by contrast, is pure arithmetic over
     * a frame that has already been copied.
     *
     * Nothing is written until the camera has had [AE_SETTLE_FRAMES] frames to apply the
     * last change. The frames in flight when a write lands were exposed under the old
     * setting, so metering them is measuring the past and correcting for it twice --
     * which is the classic way to make an auto exposure oscillate.
     */
    private fun applyAutoExposure(cam: FlirCamera) {
        val (frameId, reading) = aeReading ?: return
        if (!meterIsCurrent(frameId, aeAppliedAtFrameId)) return
        val current = aeSettings ?: return

        val limits = AutoExposure.Limits(
            // Not the sensor's whole range. Past this the frame is smeared by the hand
            // holding it whatever the meter thinks of the light, and [CaptureWarning] is
            // what covers the scene that is genuinely too dark for the shutter to reach.
            exposureUs = exposureRange.first..minOf(exposureRange.last, AE_MAX_EXPOSURE_US),
            gainDb = gainRangeDb,
        )
        val next = AutoExposure.step(current, reading, limits, exposureCompensationEv) ?: return

        val written = streamLock.withLock {
            runCatching {
                if (next.exposureUs != current.exposureUs) cam.writeExposureUs(next.exposureUs)
                if (gainRangeDb != null && next.gainDb != current.gainDb) {
                    cam.gainDb = next.gainDb
                    correction = StokesNoise.calibrated(next.gainDb, fullScale = PREVIEW_FULL_SCALE)
                }
            }.onFailure { failure ->
                Log.w(TAG, "auto exposure write failed", failure)
                // Part of it may have landed, so what the camera holds is no longer
                // what this loop believes. Read it back rather than stepping from a
                // guess: one round trip on a path that should not be taken often.
                aeSettings = runCatching {
                    AutoExposure.Settings(cam.exposureTimeUs, cam.gainDb)
                }.getOrNull()
            }.isSuccess
        }
        // Either way the camera has been given a frame's worth of time to settle, so the
        // counter moves. Retrying a failing write against every frame would spend the
        // control channel on it.
        aeAppliedAtFrameId = frameId
        if (!written) return

        aeSettings = next
        Log.i(
            TAG,
            "auto exposure: peak %.3f, %.4f%% clipped -> %d us, %.1f dB".format(
                reading.peak, reading.clipped * 100f, next.exposureUs, next.gainDb,
            ),
        )
    }

    // ---- preview ------------------------------------------------------------

    /**
     * How fast the picture is moving, measured between preview frames.
     *
     * The camera's own motion rather than the phone's, which is what the shake warning
     * has always wanted and never had: see [FrameDrift]. Null until two frames have
     * arrived, or when the scene is too plain for the match to mean anything.
     */
    @Volatile
    var previewDriftStokesPerSecond: Float? = null
        private set

    /** How well the last two preview frames matched, 0..1. Null when they did not. */
    @Volatile
    var previewDriftConfidence: Float? = null
        private set

    /** Luma of the previous preview frame, for [previewDrift]. */
    private var driftPrevious: FloatArray? = null
    private var driftPreviousAtMs = 0L
    private var driftWidth = 0
    private var driftHeight = 0

    /**
     * How sharp the last previewed frame was, as [MosaicPreview.blurRatio], or null.
     *
     * Measured on the frame rather than predicted from the gyroscope, because on 172
     * reviewed captures the frame is four times better at it: see that function.
     */
    @Volatile
    var previewBlurRatio: Float? = null
        private set

    /**
     * Mean S0 of the last previewed frame, in sensor counts, or null before one arrives.
     *
     * How bright the scene is in the units the noise model works in, so a caller can ask
     * what the polarization is worth at this light level rather than guessing from the
     * picture. Measured on the grab thread with the frame already in hand; nothing that
     * is stored depends on it.
     */
    @Volatile
    var previewS0Counts: Float? = null
        private set

    /** [width, height] of the rendered view, a quarter of the sensor in each axis. */
    fun previewSize(): Pair<Int, Int>? = synchronized(frameLock) {
        if (previewWidth == 0) null else displayWidth to displayHeight
    }

    /**
     * Copies the newest rendered view into [bitmap], which must be ARGB_8888 and sized
     * to [previewSize]. Returns false when nothing new has arrived since the last call.
     *
     * The rendering itself happened on the grab thread, so this is only a copy.
     */
    fun renderPreview(bitmap: Bitmap): Boolean = synchronized(frameLock) {
        val pixels = frontBuffer
        if (pixels == null || !frameDirty) {
            // Every refusal used to be silent, which is a poor way to run the one
            // function whose failure looks to the user exactly like a broken camera.
            note(if (pixels == null) "no frame rendered yet" else "no new frame since the last draw")
            return false
        }
        if (bitmap.width != displayWidth || bitmap.height != displayHeight) {
            note("bitmap is ${bitmap.width}x${bitmap.height}, frame is ${displayWidth}x$displayHeight")
            return false
        }
        // A frame wider than the picture it goes in would write outside it. This cannot
        // happen from the crop arithmetic, but the capture path streams the full sensor
        // for one frame and a mistake there would land here.
        if (previewWidth > displayWidth || previewHeight > displayHeight) {
            note("frame ${previewWidth}x$previewHeight exceeds the ${displayWidth}x$displayHeight picture")
            return false
        }
        // The size the buffer actually is, not the size it is believed to be. A preview
        // frame worth dropping is not worth crashing the UI thread over, and these two
        // have disagreed before.
        if (pixels.size < previewWidth * previewHeight) {
            note("buffer holds ${pixels.size}, needs ${previewWidth * previewHeight}")
            return false
        }
        lastRefusal = null
        frameDirty = false
        // Centred, with the rest of the sensor's field left as margin. Erased every
        // frame rather than once: the streamed size changes when a capture briefly takes
        // the whole sensor, and a stale border would then be a ring of the previous
        // frame's picture rather than a margin.
        val padX = (displayWidth - previewWidth) / 2
        val padY = (displayHeight - previewHeight) / 2
        if (padX > 0 || padY > 0) bitmap.eraseColor(PREVIEW_MARGIN_COLOR)
        bitmap.setPixels(pixels, 0, previewWidth, padX, padY, previewWidth, previewHeight)
        true
    }

    /**
     * Says why the preview is not drawing, once per reason.
     *
     * The UI asks for a frame at display rate, so logging every refusal would bury the
     * log; saying nothing at all is how a blank preview stayed unexplained.
     */
    private var lastRefusal: String? = null

    private fun note(reason: String) {
        if (reason != lastRefusal) {
            lastRefusal = reason
            Log.i(TAG, "preview not drawn: $reason")
        }
    }

    // ---- capture ------------------------------------------------------------    // ---- capture ------------------------------------------------------------

    /**
     * Renders every view of the newest frame and packages it with the raw mosaic.
     *
     * Blocking, and it reuses the frame the preview is already showing rather than
     * triggering a new one, so what is saved is what the user framed.
     */
    /**
     * Takes a picture at the sensor's full resolution.
     *
     * Deliberately not the frame the preview is showing: that one is a centre crop. The
     * ROI is widened, one frame is taken, and the preview crop is put back, so the
     * capture covers the whole field of view even though the preview did not.
     *
     * @param onTriggered run the instant the exposure begins, before any of the transfer.
     *   Another camera triggered from here is exposing within milliseconds of this one,
     *   which is the only way two cameras on one bus can see the same moment: everything
     *   that takes real time -- the transfer, the ROI restore, the decode -- happens
     *   after both shutters have already closed.
     */
    /**
     * What a frame was exposed at, read while the camera still held those values.
     *
     * A capture reconfigures the camera and puts it back, and the putting back used to
     * happen before anything asked what the frame had been taken at -- so a capture whose
     * shutter had been shortened for shake reported the *preview's* exposure instead, and
     * reported it as fact. The host meter could do the same thing from the grab thread in
     * the gap between releasing the stream lock and the read.
     *
     * Two registers carried out of the lock is the whole fix.
     */
    private data class TakenAt(
        val exposureUs: Long,
        val gainDb: Double,
        /**
         * Every node in [METADATA_NODES] as the camera held it for this frame.
         *
         * Read inside the lock for the same reason the two registers above are. The ROI
         * is widened to the full sensor for a capture and put back afterwards, so a node
         * snapshot taken after the restore reports the preview's Width and Height -- 1840
         * by 1536 against a frame that is 2448 by 2048. A record of the camera's state
         * that describes a different state than the frame is worse than no record.
         */
        val nodes: Map<String, String> = emptyMap(),
    )

    /**
     * When the kept frame was integrating, on System.nanoTime.
     *
     * The trigger instant is exact and the leader's arrival is not: the leader is sent
     * when an exposure ends but is only *read* when the host gets round to it, and after
     * a single-frame trigger the host deliberately goes off and triggers the other camera
     * first. That is why a lone frame's window is measured forward from the trigger.
     *
     * A burst breaks that, because the frame kept is whichever of four was sharpest and
     * the later three were exposed hundreds of milliseconds after the trigger. Measuring
     * them from the trigger described frame one every time -- and since this window is
     * what [PhoneMotion] integrates the gyroscope over, every burst capture's shake
     * figure was about a frame that had been thrown away. For those the leader *is* the
     * better clock: the grab loop is already blocked waiting when they arrive, so it
     * reads them immediately.
     */
    private fun exposureWindow(
        frame: FlirCamera.Frame,
        triggeredAtHostNs: Long,
        takenAt: TakenAt,
    ): LongRange = exposureWindowFor(
        triggeredAtHostNs = triggeredAtHostNs,
        leaderHostNs = frame.leaderHostNs,
        exposureUs = takenAt.exposureUs,
        keptIndex = burstKeptIndex,
    )

    fun capture(onTriggered: (triggerHostNs: Long) -> Unit = {}): Result<Capture> {
        val cam = camera ?: return Result.failure(IllegalStateException("device closed"))

        return runCatching {
            val t0 = System.nanoTime()
            fun lap(what: String, from: Long): Long {
                val now = System.nanoTime()
                Log.i(TAG, "capture: %-22s %5d ms".format(what, (now - from) / 1_000_000))
                return now
            }

            // Before anything is reconfigured, because both of these are about the frame
            // that is about to be taken and the preview is still running to inform them.
            steadyWaitedMs = waitForSteady()
            var triggeredAtHostNs = 0L
            // What the frame was actually taken at, read while it still is that -- see
            // [takenAt]. Filled inside the lock, before the finally restores anything.
            var takenAt = TakenAt(-1L, Double.NaN)
            val frame = streamLock.withLock {
                var t = t0
                cam.stopAcquisition()
                cam.drainStream()
                t = lap("stop + drain", t)
                // Everything that reconfigures the camera is inside the try, not only
                // the grab. Each of these writes registers and each of them can throw,
                // and a capture that fell over between widening the ROI and restoring it
                // used to leave the stream exactly as the capture wanted it: full sensor,
                // no frame-rate cap. The grab loop would then resync into that, which is
                // the unthrottled stream the cap exists to prevent.
                try {
                    // A capture wants every frame the sensor can give rather than the
                    // preview's throttled cadence.
                    cam.setFrameRateLimit(null)
                    // Twelve bits, which is what the archive is for. The preview streams
                    // eight to fit the bus; this is the line that keeps that a preview
                    // optimisation and not a change to what gets recorded. It goes after
                    // the frame-rate limiter is off, because an enabled limiter locks
                    // PixelFormat on this model.
                    cam.pixelFormat = CAPTURE_PIXEL_FORMAT
                    if (cam.pixelFormat != CAPTURE_PIXEL_FORMAT) {
                        throw U3vException("camera refused 12-bit for the capture")
                    }
                    // The whole sensor. The preview streams the whole sensor too now, so
                    // this is ordinarily a write that changes nothing -- kept because it
                    // is what makes the capture independent of whatever region the
                    // preview happens to be using, and the preview has been given a
                    // smaller one before.
                    cam.setRoi(fullWidth, fullHeight)
                    // How many frames this press wants, decided before the camera is
                    // told, because the count has to be the same number in both places:
                    // pinned to four and read three, the camera stops and the fourth grab
                    // waits out the frame timeout.
                    // Shortened only when the movement is past what a burst can rescue,
                    // and put back in the finally below whatever happens.
                    shortenedFrom = shortenExposureFor(cam)
                    lastShortenedFrom = shortenedFrom
                    val count = burstCountFor(cam)
                    // Exactly that many, when the camera will agree to it. A refusal is
                    // ordinary -- the node is clamped on some bodies and absent on others
                    // -- and leaves the sensor free-running, which is what this always did.
                    val finite = cam.setFrameCount(count)
                        .also { if (it != null) Log.i(TAG, "free-running capture of $count: $it") } == null
                    // TriggerMode is Off, so the sensor begins integrating when
                    // acquisition starts, finite or free-running.
                    cam.startAcquisition()
                    val triggerHostNs = System.nanoTime()
                    triggeredAtHostNs = triggerHostNs
                    t = lap("start acquisition", t)
                    runCatching { onTriggered(triggerHostNs) }
                        .onFailure { Log.e(TAG, "trigger callback failed", it) }
                    t = lap("trigger callback", t)
                    grabBurst(cam, count, finite, t).also {
                        // Before the finally, and before the meter gets another frame:
                        // these two registers are about to be written back to what the
                        // preview wants, and after that the camera can no longer answer
                        // what this frame was exposed at.
                        takenAt = TakenAt(
                            exposureUs = runCatching { cam.exposureTimeUs }.getOrDefault(-1L),
                            gainDb = runCatching { cam.gainDb }.getOrDefault(Double.NaN),
                            nodes = runCatching { cam.readNamedNodes(METADATA_NODES) }
                                .getOrDefault(emptyMap()),
                        )
                        lap("grab full frame", t)
                    }
                } finally {
                    restoreExposure(cam)
                    restorePreviewStream(cam)
                    lap("restore preview", t)
                }
            }

            val afterFrame = System.nanoTime()
            val resolution = captureResolution
            // Expanded once and then used for everything: the archive TIFF, the binning,
            // the noise fit and the Stokes decode all read the same samples, where the
            // packed bytes would have had to be unpicked separately for each.
            val full = Mosaic(
                StokesDecoder.unpack(frame.data, frame.width, frame.height, frame.pixelFormat),
                frame.width,
                frame.height,
                StokesDecoder.maxValueOf(frame.pixelFormat),
            )
            val mosaic = runCatching { full.binnedSum(resolution.factor) }
                .onFailure { Log.w(TAG, "could not bin at ${resolution.factor}x, keeping full", it) }
                .getOrDefault(full)

            // What the views are corrected with: the sensor's own constant, scaled for
            // this capture's gain and binning. It goes into the metadata so the gallery
            // makes the identical correction, or declines to.
            val noise = StokesNoise.calibrated(gainDb, resolution.factor)
            // And what this particular frame's own contents would have suggested, kept
            // beside it as a diagnostic. The two disagreeing by a little says nothing;
            // disagreeing by ten says the scene is textured, which is not news about the
            // camera. A run of captures where the fit sits far below the constant would
            // be, and that is what this is here to make visible.
            val fitted = runCatching { StokesNoise.measure(mosaic) }
                .onFailure { Log.w(TAG, "could not measure the noise", it) }.getOrNull()
            val image = StokesDecoder.decode(mosaic).withNoise(noise)
            val views = StokesView.entries.map { view ->
                RenderedView(
                    kind = view.fileKind,
                    label = view.label,
                    width = image.width,
                    height = image.height,
                    // Debiasing deliberately off for the file. It is a viewing
                    // correction resting on a constant that is still provisional, so
                    // baking it into a PNG would freeze today's guess into the archive;
                    // left out, every stored capture picks up a better constant the day
                    // one is measured, and the gallery -- which computes its views from
                    // the Stokes arrays -- shows the corrected picture meanwhile.
                    argb = StokesRender.render(
                        image, view, null,
                        StokesDisplay(
                            modulateAolpByDolp = modulateAolpByDolp,
                            debiasDolp = false,
                            // As the viewer shows it: a wall that is straight in the
                            // world is straight on screen, so framing a shot and reading
                            // it back later do not disagree about the lens.
                            rectify = true,
                            // And balanced the way the viewer balances it. Grey-world
                            // decides from the frame's own average, so a stored render
                            // and the viewer's render of the same capture came out two
                            // colours the moment the viewer stopped asking the frame.
                            // The camera's own ratios are at unity here -- the app sets
                            // them at open -- so there is nothing to divide out.
                            whiteBalance = WhiteBalance.FIXED,
                        ),
                    ),
                )
            }
            Capture(
                views = views,
                rawSamples = mosaic.samples,
                rawWidth = mosaic.width,
                rawHeight = mosaic.height,
                rawMaxValue = mosaic.maxValue.toInt(),
                rawDescription = rawDescription(mosaic, resolution),
                metadataJson = metadata(
                    image, frame, mosaic, resolution, fitted, triggeredAtHostNs, takenAt,
                ),
                exposureStartHostNs = exposureWindow(frame, triggeredAtHostNs, takenAt).first,
                exposureEndHostNs = exposureWindow(frame, triggeredAtHostNs, takenAt).last,
            ).also {
                lap("decode + render", afterFrame)
                Log.i(TAG, "capture: %-22s %5d ms".format("TOTAL", (System.nanoTime() - t0) / 1_000_000))
            }
        }
    }

    /**
     * Written into the raw file's ImageDescription.
     *
     * Anything reading the archive copy later needs the mosaic spelled out rather than
     * inferred from a file name, so it travels inside the file.
     */
    private fun rawDescription(mosaic: Mosaic, resolution: CaptureResolution): String =
        "FLIR Blackfly S BayerRGPolarized12p colour-polarization mosaic. " +
            "4x4 super-pixel: the analyser repeats every 2 px as (row,col) " +
            "(0,0)=90 (0,1)=45 (1,0)=135 (1,1)=0 degrees; the colour repeats every " +
            "4 px as RG/GB, one colour per 2x2 analyser cell. Samples are unsigned, " +
            "full scale ${mosaic.maxValue.toInt()}, stored in 16-bit words." +
            if (resolution.factor > 1) {
                " Binned ${resolution.factor}x${resolution.factor}: each sample is the sum " +
                    "of ${resolution.factor * resolution.factor} samples at the same colour " +
                    "and analyser angle from neighbouring super-pixels, so the layout above " +
                    "still holds and the sensor read out ${mosaic.width * resolution.factor}x" +
                    "${mosaic.height * resolution.factor}."
            } else {
                ""
            }

    private fun metadata(
        image: StokesImage,
        frame: FlirCamera.Frame,
        mosaic: Mosaic,
        resolution: CaptureResolution,
        fitted: StokesNoise?,
        triggeredAtHostNs: Long,
        takenAt: TakenAt,
    ): String {
        val w = frame.width
        val h = frame.height
        val exposure = takenAt.exposureUs
        val gain = takenAt.gainDb

        val window = exposureWindow(frame, triggeredAtHostNs, takenAt)
        val midExposureHostNs = (window.first + window.last) / 2
        return """
            {
              "device": "flir",
              "model": "$name",
              "pixelFormat": "BayerRGPolarized12p",
              "maxValue": ${mosaic.maxValue.toInt()},
              "sensor": {"width": $w, "height": $h},
              "binning": {"factor": ${resolution.factor}, "mode": "sum",
                          "width": ${mosaic.width}, "height": ${mosaic.height},
                          "note": "samples are sums over ${resolution.factor * resolution.factor} super-pixels; maxValue is scaled to match"},
              "stokes": {"width": ${image.width}, "height": ${image.height},
                         "bands": ["R","G","B"], "components": ["S0","S1","S2"],
                         "dtype": "float32", "layout": "(band, component, y, x)"},
              "exposureTimeUs": $exposure,
              "gainDb": ${if (gain.isNaN()) "null" else "%.3f".format(gain)},
              "frame": {"id": ${frame.frameId},
                        "midExposureHostNs": $midExposureHostNs,
                        "triggeredAtHostNs": $triggeredAtHostNs,
                        "leaderArrivedAtHostNs": ${frame.leaderHostNs},
                        "deviceTimestamp": ${frame.deviceTimestamp},
                        "clock": "midExposureHostNs is System.nanoTime; deviceTimestamp is the camera's own unit and is not comparable"},
              "drift": ${driftJson()},
              "shake": ${shakeJson(takenAt)},
              "rawPreset": ${presetJson()},
              "nodes": ${nodesJson(takenAt.nodes)},
              "metering": ${meteringJson()},
              "clipping": ${clippingJson(mosaic)},
              "burst": ${burstJson()},
              "noise": ${image.noise?.toJson() ?: "null"},
              "noiseFit": ${fitted?.toJson() ?: "null"},
              "noiseNote": "noise is the calibrated sensor constant and is what the views are corrected with; noiseFit is what this frame's own two green cells suggested, which includes whatever the scene does over 2.8 px and is only ever an upper bound",
              "analyserLayout": "TL=90 TR=45 BL=135 BR=0",
              "bayerCellOrder": "RG/GB, one colour per 2x2 analyser cell"
            }
        """.trimIndent()
    }

    /**
     * Waits, briefly, for a shaking hand to settle before the shutter fires.
     *
     * Hand tremor is a few hertz, so a frame taken a moment later is often taken at the
     * bottom of the swing rather than the middle of it. This watches the preview's own
     * registration -- the camera's movement, not the phone's -- and returns as soon as it
     * falls under what the viewfinder is already marking as shake, or when the wait runs
     * out, whichever is first.
     *
     * A steady hand never pays for this: the first reading already passes and it returns
     * immediately. There is deliberately no wait when nothing is being measured, because
     * a missing reading is not a still camera and must never be treated as one.
     *
     * @return milliseconds actually waited, for the metadata
     */
    private fun waitForSteady(): Long {
        if (!steadyShutter) return 0L
        val exposure = runCatching { camera?.exposureTimeUs ?: 0L }.getOrDefault(0L)
        val started = System.nanoTime()
        var waited = 0L
        while (waited < CaptureSmear.STEADY_WAIT_MS) {
            val smear = CaptureSmear.pixels(previewDriftStokesPerSecond, exposure) ?: return waited
            if (smear <= CaptureSmear.STEADY_PIXELS) break
            Thread.sleep(STEADY_POLL_MS)
            waited = (System.nanoTime() - started) / 1_000_000
        }
        if (waited > 0) Log.i(TAG, "waited $waited ms for the camera to settle")
        return waited
    }

    /**
     * Whether the shutter waits for a shaking hand, and whether it may shorten the
     * exposure when waiting will not be enough.
     *
     * Off by default. Both change what gets recorded rather than how it is shown, which
     * on this app is a decision for the person shooting and not a default.
     */
    var steadyShutter: Boolean = false
    var shortenUnderShake: Boolean = false

    /** The exposure a capture moved away from, so it can be put back. Null when untouched. */
    private var shortenedFrom: Long? = null

    /** The same, kept past the restore so the metadata can report it. */
    private var lastShortenedFrom: Long? = null

    /** How long the last shutter press waited for the camera to settle. */
    private var steadyWaitedMs: Long = 0L

    /**
     * Shortens this capture's exposure when the camera is moving too fast for anything
     * else to help, and reports what it moved away from.
     *
     * The trade is deliberate and it is not symmetric: noise is corrected for afterwards
     * from a calibrated model, and blur is not correctable at all. It is bounded at two
     * stops by [CaptureSmear.SHORTEN_MAX_FACTOR] so that a dim scene cannot be made
     * unmeasurable in the name of sharpness.
     */
    private fun shortenExposureFor(cam: FlirCamera): Long? {
        if (!shortenUnderShake) return null
        val now = runCatching { cam.exposureTimeUs }.getOrNull() ?: return null
        val wanted = CaptureSmear.shortenedExposureUs(previewDriftStokesPerSecond, now)
        if (wanted >= now) return null
        return runCatching {
            cam.exposureTimeUs = wanted
            Log.i(TAG, "exposure shortened ${now / 1000} -> ${wanted / 1000} ms for this frame")
            now
        }.getOrNull()
    }

    /** Puts back an exposure [shortenExposureFor] moved, whatever the capture did. */
    private fun restoreExposure(cam: FlirCamera) {
        val was = shortenedFrom ?: return
        shortenedFrom = null
        runCatching { cam.exposureTimeUs = was }
            .onFailure { Log.w(TAG, "could not restore the exposure to ${was / 1000} ms", it) }
    }

    /**
     * How many frames this shutter press will ask the sensor for.
     *
     * Split out of the burst itself because the camera has to be told the same number
     * before acquisition starts, and a count decided in two places is a count that will
     * eventually disagree with itself.
     *
     * Not worth a burst past a certain shutter. A burst helps because shake varies
     * between frames a fifth of a second apart; at a long exposure every frame is
     * smeared by the same slow drift and there is no good one to find, while the cost of
     * the extra frames grows with the exposure rather than with the transfer. Below this
     * the frame interval is set by the bus and the exposure is nearly free.
     */
    private fun burstCountFor(cam: FlirCamera): Int {
        val wanted = burstFrames
        if (wanted <= 1) return 1
        val exposure = runCatching { cam.exposureTimeUs }.getOrDefault(0L)
        if (exposure > BURST_MAX_EXPOSURE_US) {
            Log.i(TAG, "burst skipped: ${exposure / 1000} ms exposure is past ${BURST_MAX_EXPOSURE_US / 1000} ms")
            return 1
        }
        // And not worth one when the camera is sweeping: every frame carries the same
        // smear, so there is no quiet one among them to find. See CaptureSmear.
        val smear = CaptureSmear.pixels(previewDriftStokesPerSecond, exposure)
        if (smear != null && smear > CaptureSmear.BURST_FUTILE_PIXELS) {
            Log.i(TAG, "burst skipped: %.1f px of predicted smear is past %.0f".format(
                smear, CaptureSmear.BURST_FUTILE_PIXELS))
            return 1
        }
        return wanted
    }

    /**
     * Puts the stream back the way the preview needs it, whatever the capture did.
     *
     * Every step is attempted and its failure recorded, rather than the first failure
     * abandoning the rest. The steps are not alternatives -- they are five different
     * settings that all have to be right -- and the one that used to be skipped most
     * cheaply is the last: an exception from [FlirCamera.setRoi] left the frame-rate cap
     * off and the stream stopped, and the grab loop's own resync would then bring the
     * camera back free-running at the full sensor. That is precisely the unthrottled
     * stream the cap exists to prevent, arrived at by way of the code meant to restore
     * it.
     */
    private fun restorePreviewStream(cam: FlirCamera) {
        val failures = mutableListOf<String>()
        fun attempt(what: String, action: () -> Unit) {
            runCatching(action).onFailure { failures += "$what: ${it.message ?: it::class.simpleName}" }
        }

        attempt("stop") { cam.stopAcquisition() }
        attempt("drain") { cam.drainStream() }
        // Before the ROI and the cap: both are read-only while streaming, and the mode
        // is what decides whether the camera will keep producing frames at all.
        attempt("continuous") { cam.setContinuousAcquisition()?.let { error(it) } }
        // Back to eight bits, and before the cap for the same reason the capture set it
        // after clearing the cap: an enabled frame-rate limiter locks PixelFormat.
        attempt("preview pixel format") {
            cam.pixelFormat = PREVIEW_PIXEL_FORMAT
            if (cam.pixelFormat != PREVIEW_PIXEL_FORMAT) error("camera kept the capture format")
        }
        attempt("preview ROI") { cam.setRoi(previewRoiWidth, previewRoiHeight) }
        attempt("frame rate cap") { cam.setFrameRateLimit(previewFpsCap) }
        attempt("start") { if (previewActive.get()) cam.startAcquisition() }

        if (failures.isNotEmpty()) {
            Log.e(TAG, "preview stream restored with ${failures.size} failure(s): ${failures.joinToString("; ")}")
        }
    }

    /**
     * Grabs one frame, or several and keeps the sharpest.
     *
     * ### Why several
     *
     * Handheld, the median frame smears 3.2 Stokes pixels and only one in ten comes in
     * under a single pixel. Nothing about the exposure fixes that, but the shake is not
     * constant: taking four and keeping the best of them moves the median smear to about
     * 2.2 pixels, because three of the four are allowed to be bad.
     *
     * All of them come from one acquisition. The expensive part of a capture is stopping
     * the preview, draining, and switching the ROI -- around a second -- and that is paid
     * once whether one frame is taken or four. The extra frames cost only their transfer.
     *
     * ### Why the sharpest rather than the steadiest
     *
     * The gyroscope separates the frames a person keeps from the ones they delete with an
     * AUC of about 0.66: real, and far from decisive. Measuring the picture instead
     * avoids the whole chain that number depends on -- the unknown geometry between phone
     * and camera, the sensor clock alignment, the assumed focal length. And the objection
     * to an image sharpness metric, that it cannot tell a blurred frame from a plain
     * scene, does not apply here: these four frames are the same scene a fifth of a
     * second apart, so the scene cancels and only the smear differs.
     *
     * Every frame's score is recorded, not just the winner's, so the choice can be
     * checked afterwards and the two rankings compared against each other.
     */
    private fun grabBurst(cam: FlirCamera, count: Int, finite: Boolean, from: Long): FlirCamera.Frame {
        burstScores = null
        burstKeptIndex = 0
        if (count <= 1) return cam.grabFrame()

        var best: FlirCamera.Frame? = null
        var bestScore = Float.NEGATIVE_INFINITY
        val scores = FloatArray(count) { Float.NaN }
        for (i in 0 until count) {
            // A burst that fails part way still has frames in hand, and one of those is
            // worth far more than an exception: the shutter has already been pressed and
            // the moment is gone.
            val frame = runCatching { cam.grabFrame() }.getOrElse {
                Log.w(TAG, "burst frame ${i + 1} of $count failed", it)
                break
            }
            val score = MosaicPreview.sharpness(
                frame.data, frame.width, frame.height, frame.pixelFormat,
            )
            scores[i] = score
            if (score > bestScore) {
                bestScore = score
                best = frame
            }
        }
        burstScores = scores
        burstKeptIndex = scores.indexOfFirst { it == bestScore }.coerceAtLeast(0)
        // Nothing at all arrived. Asking again only makes sense while the sensor is
        // still making frames: pinned to a count, they have all been made and the next
        // grab would wait out the frame timeout to say so.
        val chosen = best
            ?: if (finite) throw U3vException("burst of $count produced no frame") else cam.grabFrame()
        Log.i(
            TAG,
            "burst of %d in %d ms, kept #%d: %s".format(
                scores.count { !it.isNaN() },
                (System.nanoTime() - from) / 1_000_000,
                scores.indexOfFirst { it == bestScore } + 1,
                scores.joinToString(" ") { if (it.isNaN()) "-" else "%.4f".format(it) },
            ),
        )
        return chosen
    }

    /**
     * How fast the picture was moving when the shutter was pressed, from the preview.
     *
     * Recorded so it can be scored against keep-or-delete verdicts the same way the
     * gyroscope was, on the same captures. That comparison is the point: the gyroscope
     * reached an AUC of about 0.8 and is strapped to the phone, and this is the first
     * measurement of the camera's own movement the archive has ever carried.
     */
    private fun driftJson(): String {
        val speed = previewDriftStokesPerSecond ?: return """{"available": false, """ +
            """"reason": "no match between preview frames; the scene may be too plain"}"""
        val conf = previewDriftConfidence ?: 0f
        return """{"available": true, "stokesPixelsPerSecond": %.2f, "confidence": %.3f, """
            .format(speed, conf) +
            """"source": "registration of consecutive preview frames", """ +
            """"note": "the camera's own displacement, in its own pixels: no focal length, """ +
            """no sensor clock and no assumption about how the phone is attached"}"""
    }

    /**
     * What this capture did about the shake it saw, so the policy stays measurable.
     *
     * The prediction and the action are both here on purpose. Every threshold in
     * [CaptureSmear] came from joining these fields against keep-or-delete verdicts, and
     * the only way the next revision can be as well founded is if the frames taken under
     * this one say what it did.
     */
    private fun shakeJson(takenAt: TakenAt): String {
        // The exposure the frame was taken at, not the one the camera has by now. This
        // read the register after the restore had already put the preview's exposure
        // back, so a capture whose shutter was shortened reported the smear it would have
        // had if it had not been -- the opposite of what the field is for.
        val exposure = takenAt.exposureUs.coerceAtLeast(0L)
        val smear = CaptureSmear.pixels(previewDriftStokesPerSecond, exposure)
        return """{"predictedSmearPixels": ${smear?.let { "%.2f".format(it) } ?: "null"}, """ +
            """"steadyShutter": $steadyShutter, "waitedMs": $steadyWaitedMs, """ +
            """"shortenUnderShake": $shortenUnderShake, """ +
            """"exposureBeforeShorteningUs": ${lastShortenedFrom ?: "null"}, """ +
            """"note": "smear is the preview's own registration times the exposure, in Stokes pixels; """ +
            """the exposure above is what this frame would have used had it not been shortened"}"""
    }

    /** What the camera did with each raw-preset setting when it was opened. */
    private var presetOutcomes: Map<String, String> = emptyMap()

    /** The white-balance ratios the camera was holding, by colour. */
    private var balanceRatios: Map<String, Double> = emptyMap()

    /**
     * Whether every setting meant to keep the camera's own pipeline off the pixels took.
     *
     * This is the field that answers "was white balance actually off for this capture",
     * and it could not be answered before: the settings were applied, logged, and the log
     * was gone long before anybody thought to ask of a months-old archive. The distinction
     * matters and is not cosmetic -- per-band DoLP and AoLP survive a band being scaled,
     * because every analyser angle in one colour's cell is scaled alike and the ratio
     * cancels, but the noise model does not. [StokesNoise] reads variance as a constant
     * times the signal in raw counts, so a band multiplied by k reports a shot gain k
     * times off, and everything resting on it -- the DoLP debias, the noise floor, the
     * valid-pixel fraction -- is wrong for that band and right for green.
     */
    private fun presetJson(): String {
        if (presetOutcomes.isEmpty()) return "null"
        val entries = presetOutcomes.entries.joinToString(", ") { (what, how) ->
            """"${what.replace("\"", "'")}": "${how.replace("\"", "'")}""""
        }
        val refused = presetOutcomes.count { it.value != "ok" }
        val ratios = balanceRatios.entries.joinToString(", ") { (band, r) ->
            """"$band": %.5f""".format(r)
        }
        return """{"applied": ${presetOutcomes.size - refused}, "notApplied": $refused, """ +
            """"balanceRatios": {$ratios}, """ +
            """"balanceRatioNote": "against green; 1 means the band was not scaled. """ +
            """A band scaled by k leaves per-band DoLP and AoLP unchanged but makes the """ +
            """noise model report a shot gain k times off for that band", """ +
            """"settings": {$entries}, """ +
            """"note": "everything meant to keep the camera's own processing off the pixels; """ +
            """'ok' means the write took and read back, anything else is what the camera said"}"""
    }

    /**
     * What the burst saw, or null when one frame was taken.
     *
     * Every frame's score is here, not only the winner's. Two things become answerable
     * that are not otherwise: whether picking by sharpness picked the frame a person
     * would have, and how much a burst actually bought over taking the first frame --
     * which is the ratio between the chosen score and `scores[0]`, and it costs nothing
     * to record now and cannot be recovered later.
     */
    /**
     * Every node in [METADATA_NODES] as the camera holds it now.
     *
     * The preset block says what this app *asked* for and what the camera said about each
     * request. This says what the camera is actually holding, which is a different claim
     * and the more useful one: a node that was never asked about, or that some other
     * client changed, or that the camera quietly clamped, shows up here and nowhere else.
     *
     * Read once per capture, inside the stream lock and before the ROI and the exposure
     * are put back -- see [TakenAt]. It is a few dozen register reads over the control
     * channel, which is the channel the stream shares: affordable once at the shutter,
     * which is why it is not on the preview path.
     */
    private fun nodesJson(values: Map<String, String>): String {
        if (values.isEmpty()) return "null"
        return values.entries.joinToString(", ", "{", "}") { (name, value) -> """"$name": $value""" }
    }

    /**
     * What the meter saw, and over what.
     *
     * The region matters as much as the numbers. This meters the preview ROI, which is a
     * three-quarter crop of the sensor sampled one pixel in four, while the capture reads
     * the whole frame -- so the meter has never seen the outer eighth of what it is
     * exposing for. That is a defensible trade for preview bandwidth and an indefensible
     * thing to leave unrecorded, because it is exactly the kind of difference that makes
     * two rigs disagree for a reason nobody can find afterwards.
     */
    private fun meteringJson(): String {
        val reading = aeReading?.second
            ?: return """{"policy": "${AutoExposure.POLICY.name.lowercase()}", """ +
                """"compensationEv": %.3f, "reading": null, """.format(exposureCompensationEv) +
                """"reason": "the host meter was not running"}"""
        return """{"policy": "${AutoExposure.POLICY.name.lowercase()}", """ +
            """"compensationEv": %.3f, "mean": %.4f, "peak": %.4f, "clippedFraction": %.5f, """
                .format(exposureCompensationEv, reading.mean, reading.peak, reading.clipped) +
            """"region": "the whole sensor, one pixel in """ +
            """${MosaicPreview.HIGHLIGHT_GRID * MosaicPreview.HIGHLIGHT_GRID}", """ +
            """"note": "the preview streams the full frame, so the meter saw what the capture stored"}"""
    }

    /**
     * How much of the frame that was actually stored is at full scale.
     *
     * Measured on the capture rather than on the preview, so it describes the file rather
     * than the thing that was metered. There is no retry: This Polar Camera has a
     * capture-and-retry helper for this (`_acquire_highlight_safe_raw`) and its own live
     * bracket path does not call it, so a clipped frame is recorded as clipped on both.
     * Recording it is what makes that choice reviewable.
     */
    private fun clippingJson(mosaic: Mosaic): String {
        val saturation = (MosaicPreview.SATURATION * mosaic.maxValue).toInt()
        var clipped = 0L
        var peak = 0
        for (sample in mosaic.samples) {
            val value = sample.toInt() and 0xFFFF
            if (value > peak) peak = value
            if (value >= saturation) clipped++
        }
        val total = mosaic.samples.size.toLong().coerceAtLeast(1L)
        return """{"clippedFraction": %.6f, "peak": %d, "fullScale": %d, """
            .format(clipped.toDouble() / total, peak, mosaic.maxValue.toInt()) +
            """"note": "over the stored frame, every sample; no retry is attempted"}"""
    }

    private fun burstJson(): String {
        val scores = burstScores ?: return "null"
        val valid = scores.filter { !it.isNaN() }
        if (valid.isEmpty()) return "null"
        val best = valid.max()
        return """{"frames": ${valid.size}, "kept": ${scores.indexOfFirst { it == best } + 1}, """ +
            """"metric": "mean squared neighbour difference over the mean squared, green cells", """ +
            """"sharpness": [${scores.joinToString(", ") { if (it.isNaN()) "null" else "%.6f".format(it) }}]}"""
    }

    // ---- rendering ----------------------------------------------------------

    /**
     * The newest frame waiting to be drawn, and nothing older.
     *
     * One slot on purpose. If the renderer is still busy when the next frame lands, the
     * waiting one is dropped rather than queued: a preview exists to show what the camera
     * is pointed at now, and a backlog of frames to draw is the same latency the frame
     * rate cap was introduced to prevent, just moved into the phone.
     */
    private val renderLock = Object()
    private var pendingFrame: FlirCamera.Frame? = null
    private val skippedRenders = AtomicLong()
    private val renderMsTotal = AtomicLong()
    private var renderThread: Thread? = null

    private fun offerToRenderer(frame: FlirCamera.Frame) {
        synchronized(renderLock) {
            if (pendingFrame != null) skippedRenders.incrementAndGet()
            pendingFrame = frame
            (renderLock as Object).notifyAll()
        }
    }

    /**
     * Turns raw mosaics into pixels, off the thread that reads the bus.
     *
     * Splitting these was worth 40 ms a frame: the decode and the transfer used to run
     * one after the other on the grab thread, so the camera had nobody reading from it
     * for a fifth of every cycle.
     */
    private fun renderLoop() {
        while (running.get()) {
            val frame = synchronized(renderLock) {
                while (running.get() && pendingFrame == null) {
                    runCatching { (renderLock as Object).wait(IDLE_POLL_MS) }
                }
                pendingFrame.also { pendingFrame = null }
            } ?: continue

            val started = System.currentTimeMillis()
            val view = StokesView.entries[previewSource.get()]
            // Only the views built from a magnitude are biased by noise. The correction
            // is a cached constant rather than a fit over the frame, which also gives the
            // preview back the two milliseconds a fit cost.
            val noise = if (debiasDolp && (view == StokesView.DOLP || view == StokesView.AOLP)) {
                correction
            } else {
                null
            }
            val rendered = runCatching {
                MosaicPreview.render(
                    frame.data, frame.width, frame.height, frame.pixelFormat, view, backBuffer,
                    modulateAolpByDolp, noise, PREVIEW_STEP, previewSumming, previewZebra,
                )
            }.getOrElse {
                Log.w(TAG, "preview render failed", it)
                continue
            }
            val afterRender = System.currentTimeMillis()

            // On screen now, before anything else is measured from this frame.
            //
            // This used to come last, after the mean, the blur ratio, the drift
            // registration and the meter -- about 110 ms of arithmetic at full
            // resolution, none of which the picture depends on. The frame was finished
            // and sat there while the shake machinery read the same bytes for its own
            // purposes. Nothing here feeds the render, so nothing here has any business
            // being in front of it.
            synchronized(frameLock) {
                backBuffer = frontBuffer
                frontBuffer = rendered
                // From the same arithmetic the render used, not a second copy of it.
                val pitch = MosaicPreview.pitch(PREVIEW_STEP)
                previewWidth = frame.width / pitch
                previewHeight = frame.height / pitch
                frameDirty = true
            }
            // From the leader, which is the first byte of this frame the host saw. The
            // number it reports is the payload transfer plus the decode, which is the
            // whole of what the app adds between the sensor and the screen.
            latencyMsTotal.addAndGet((System.nanoTime() - frame.leaderHostNs) / 1_000_000)

            // Reported in the twelve-bit basis whatever the preview is streaming: every
            // threshold that reads this -- the darkness warning, the noise model's own
            // units -- was fitted against counts out of 4095, and an eight-bit frame
            // would otherwise report a well-lit scene as sixteen times too dark.
            previewS0Counts = MosaicPreview.meanS0(
                frame.data, frame.width, frame.height, frame.pixelFormat,
            )?.let { it * StokesNoise.TWELVE_BIT_FULL_SCALE / StokesDecoder.maxValueOf(frame.pixelFormat) }
            previewBlurRatio = MosaicPreview.blurRatio(
                frame.data, frame.width, frame.height, frame.pixelFormat,
            )
            measureDrift(frame)
            // Only when something is going to act on it. The highlight walk reads a
            // quarter of the frame, which is worth paying for a control that is running
            // and not worth paying for one that is switched off.
            if (autoExposure) {
                aeReading = MosaicPreview
                    .meter(frame.data, frame.width, frame.height, frame.pixelFormat)
                    ?.let { frame.frameId to it }
            }
            // Split, because the two halves answer different questions and have very
            // different answers: the decode is the resolution's cost, and everything
            // after it -- the mean, the blur ratio, the drift registration -- is what the
            // shake and exposure machinery costs per frame. Measured at full resolution:
            // 40 ms of decode against 110 ms of the rest, of which the drift registration
            // is 90. That 110 no longer delays the picture, but it still occupies the
            // render thread, so it is what to attack if this ever has to keep up with a
            // faster link.
            decodeMsTotal.addAndGet(afterRender - started)
            renderMsTotal.addAndGet(System.currentTimeMillis() - started)
        }
    }

    /**
     * Registers this preview frame against the last one.
     *
     * Uses the same coarse green-cell samples the level probe reads, so the displacement
     * comes out in grid pixels and converts to Stokes pixels by the step alone -- no
     * focal length, no sensor clock, no assumption about how the phone is attached.
     *
     * At [DRIFT_STEP] rather than the preview's own step, which are no longer the same
     * number. It followed the preview until the preview went to full resolution, and that
     * quadrupled the grid this searches: 17 by 17 offsets over four times the samples is
     * sixteen times the arithmetic, and the registration went from about 20 ms a frame to
     * 90. Nothing wanted that. What the shake machinery needs from this is a displacement
     * of a few pixels measured reliably, and a coarser grid measures it just as reliably
     * and reports it in its own units, which is what [FrameDrift.Result.stokesPerSecond]
     * takes the step for.
     */
    private fun measureDrift(frame: FlirCamera.Frame) {
        val now = System.currentTimeMillis()
        val pitch = MosaicPreview.pitch(DRIFT_STEP)
        val w = frame.width / pitch
        val h = frame.height / pitch
        val luma = runCatching {
            MosaicPreview.lumaGrid(frame.data, frame.width, frame.height, frame.pixelFormat, DRIFT_STEP)
        }.getOrNull()
        if (luma == null || w < 8 || h < 8) {
            driftPrevious = null
            previewDriftStokesPerSecond = null
            previewDriftConfidence = null
            return
        }
        val previous = driftPrevious
        // A changed preview size means the ROI moved under us; there is nothing to
        // compare against and pretending otherwise would report a spurious jump.
        if (previous != null && driftWidth == w && driftHeight == h && driftPreviousAtMs > 0) {
            val drift = runCatching {
                FrameDrift.between(previous, luma, w, h, (now - driftPreviousAtMs).toFloat())
            }.getOrNull()?.takeIf { it.trusted }
            previewDriftStokesPerSecond = drift?.stokesPerSecond(DRIFT_STEP)
            previewDriftConfidence = drift?.confidence
        } else {
            previewDriftStokesPerSecond = null
            previewDriftConfidence = null
        }
        driftPrevious = luma
        driftPreviousAtMs = now
        driftWidth = w
        driftHeight = h
    }

    // ---- lifecycle ----------------------------------------------------------

    private fun workerLoop() {
        val cam = camera ?: return
        var frames = 0
        var window = System.currentTimeMillis()
        var decodeTotal = 0L
        // The camera numbers every frame it produces, so the gaps in what arrives are the
        // frames it made and the host never collected. Without this the fps line says how
        // fast the preview runs but not whether it is running behind, and those are the
        // two different problems: a slow preview is a slow preview, and a preview that
        // drops frames is a preview showing something older than the scene.
        var lastFrameId = -1L
        var dropped = 0L
        // Consecutive, and reset by any frame that arrives: a camera that recovers is
        // not on its way out, however often it has hiccupped over a long session.
        var stalls = 0

        while (running.get()) {
            try {
                if (!previewActive.get()) {
                    Thread.sleep(IDLE_POLL_MS)
                    continue
                }
                var triggeredAtHostNs = 0L
            val frame = streamLock.withLock { cam.grabFrame() }
                if (frame.payloadNs > 0 && frame.payloadBytes > 0) {
                    busRates += frame.payloadBytes * 1e9 / frame.payloadNs
                }

                // Handed to the render thread rather than decoded here. The decode is
                // 40 ms and the transfer is 210, and while this thread decoded, the bus
                // sat idle -- the camera cannot start sending the next frame until
                // someone is reading. Passing it on costs a reference and gets that 40 ms
                // back on every frame.
                offerToRenderer(frame)

                // The write, on the thread that owns the control channel. The reading it
                // acts on was measured by the renderer, one frame behind at worst, which
                // the settle count already allows for.
                if (autoExposure) applyAutoExposure(cam)

                // The transfer is 29 bulk reads and the decode walks 5 MP, so report
                // both: which of the two limits the preview is not obvious up front.
                if (lastFrameId >= 0 && frame.frameId > lastFrameId) {
                    dropped += frame.frameId - lastFrameId - 1
                }
                lastFrameId = frame.frameId

                if (++frames >= FPS_REPORT_FRAMES) {
                    val elapsed = System.currentTimeMillis() - window
                    val measured = PreviewCadence.throughput(busRates)
                    busRates.clear()
                    if (measured != null) retune(cam, measured)
                    Log.i(
                        TAG,
                        // Against the cap, because the gap between them is the latency:
                        // every frame the camera makes that the host does not collect
                        // waits in the camera and reaches the screen late.
                        ("%.1f fps over %d frames against a %.1f fps cap (%s), " +
                            "%d ms/frame rendering of which %d decode, " +
                            "%d ms leader to screen, %d dropped, %d not drawn").format(
                            frames * 1000.0 / elapsed, frames, previewFpsCap,
                            measured?.let { "bus %.0f MB/s".format(it / 1e6) } ?: "unmeasured",
                            renderMsTotal.getAndSet(0) / frames,
                            decodeMsTotal.getAndSet(0) / frames,
                            latencyMsTotal.getAndSet(0) / frames, dropped,
                            skippedRenders.getAndSet(0),
                        ),
                    )
                    frames = 0
                    decodeTotal = 0
                    dropped = 0
                    window = System.currentTimeMillis()
                }
                stalls = 0
            } catch (e: Exception) {
                if (!running.get()) continue
                // A bulk read that times out part way through a frame is not the same
                // event as the camera being unplugged, and until now both ended the
                // session for good: one stalled transfer and the preview was dead until
                // the user reconnected. At 7.5 MB a frame on a shared USB 2.0 bus a stall
                // is a normal occurrence, so the stream is resynced and the loop carries
                // on. The resync speaks over the control channel, which is exactly what a
                // vanished camera cannot answer -- so an unplug still lands below.
                if (++stalls <= MAX_STALLS && resyncStream(cam)) {
                    Log.w(TAG, "stream stalled ($stalls of $MAX_STALLS), resynced: ${e.message}")
                    continue
                }
                errorText = e.message ?: e.toString()
                Log.e(TAG, "grab failed", e)
                running.set(false)
                // Unplugging mid-stream lands here. Drop isOpen with it, or the UI
                // would keep polling a session whose grab loop has already exited
                // and would never try to reconnect.
                open.set(false)
            }
        }
    }

    /**
     * Puts the stream back in a known state after a transfer failed part way through.
     *
     * Stopping and draining discards whatever half a frame is still queued on the camera;
     * without it the next read would begin mid-payload and every frame after would be
     * offset. Returns false when the camera does not answer at all, which is the case the
     * caller must not retry.
     */
    private fun resyncStream(cam: FlirCamera): Boolean = runCatching {
        streamLock.withLock {
            cam.stopAcquisition()
            cam.drainStream()
            if (previewActive.get()) cam.startAcquisition()
        }
    }.onFailure { Log.w(TAG, "stream resync failed; treating the camera as gone", it) }.isSuccess

    /**
     * The rate this ROI can actually be delivered at, in frames per second.
     *
     * A Mono12p frame is a byte and a half per pixel, and the bus moves about
     * [PREVIEW_BYTES_PER_SECOND] of them, so the arithmetic is one division. Deliberately
     * a little pessimistic: a cap slightly under what the host can drain leaves it waiting
     * a few milliseconds for each frame, which costs nothing, while a cap slightly over
     * grows a queue that costs a frame of latency for every frame it holds.
     */
    internal fun previewCapFor(width: Int, height: Int): Double {
        val payloadBytes = previewPayloadBytes(width, height)
        if (payloadBytes <= 0) return PREVIEW_FPS
        return (PREVIEW_BYTES_PER_SECOND / payloadBytes).coerceIn(MIN_PREVIEW_FPS, PREVIEW_FPS)
    }

    /**
     * Bytes one preview frame carries, which is the only thing the cadence turns on.
     *
     * From [PREVIEW_PIXEL_FORMAT] rather than assumed: packed 12-bit is a byte and a half
     * a sample and eight-bit is one, and the difference between them is a third of the
     * frame rate. It was written as a constant times three halves, which was true of
     * every frame this app had streamed up to the day the preview went to eight bits.
     */
    internal fun previewPayloadBytes(width: Int, height: Int): Long {
        if (width <= 0 || height <= 0) return 0L
        val samples = width.toLong() * height
        return if (StokesDecoder.isPacked12(PREVIEW_PIXEL_FORMAT)) samples * 3 / 2 else samples
    }

    fun disconnect() = closeCamera()

    override fun close() = closeCamera()

    private fun closeCamera() {
        running.set(false)
        worker?.let { thread ->
            // The grab loop blocks on a bulk read, so give it the frame timeout to notice.
            runCatching { thread.join(JOIN_TIMEOUT_MS) }
        }
        worker = null
        // It waits on the slot rather than on the bus, so it needs waking to notice that
        // running went false; joining it before the buffers are cleared below keeps it
        // from swapping a frame into a preview that is being torn down.
        renderThread?.let { thread ->
            synchronized(renderLock) {
                pendingFrame = null
                (renderLock as Object).notifyAll()
            }
            runCatching { thread.join(JOIN_TIMEOUT_MS) }
        }
        renderThread = null
        runCatching { camera?.close() }.onFailure { Log.w(TAG, "close failed", it) }
        camera = null
        genicam = null
        synchronized(frameLock) {
            frontBuffer = null
            backBuffer = null
            previewWidth = 0
            previewHeight = 0
            frameDirty = false
        }
        open.set(false)
    }


    internal companion object {
        const val TAG = "FlirDevice"
        const val JOIN_TIMEOUT_MS = 6000L
        const val FPS_REPORT_FRAMES = 20

        /** How often an idle grab loop checks whether the preview came back. */
        const val IDLE_POLL_MS = 50L

        /**
         * The margin around the streamed region.
         *
         * Grey rather than black: black is what a dead preview looks like, and this is a
         * live part of the frame that simply is not being sent.
         */
        const val PREVIEW_MARGIN_COLOR = 0xFF202428.toInt()

        /**
         * What the preview streams: the same mosaic, eight bits a sample.
         *
         * The link is the preview's whole constraint. This camera enumerates at USB 2.0
         * High Speed on this phone -- 512 byte bulk packets, measured at 45 MB/s -- and a
         * full 12-bit frame is 7.5 MB of that, which is six frames a second before
         * anything else is considered. Eight bits is 5.0 MB, and the preview picture is
         * drawn into eight-bit pixels either way.
         *
         * What it costs is precision in the preview's own measurements, not in anything
         * stored: the DoLP debias and the darkness warning both work in counts, and both
         * are told which scale the frame is on rather than assuming twelve bits. The
         * capture switches to [CAPTURE_PIXEL_FORMAT] and back around every shutter press.
         */
        val PREVIEW_PIXEL_FORMAT = PixelFormats.BAYER_RG_POLARIZED_8

        /** What a capture is taken at, and what the archive holds. */
        val CAPTURE_PIXEL_FORMAT = PixelFormats.BAYER_RG_POLARIZED_12P

        /** Full scale of one analyser reading in the preview's format. */
        val PREVIEW_FULL_SCALE = StokesDecoder.maxValueOf(PREVIEW_PIXEL_FORMAT)

        /**
         * Super-pixels per sample for the drift registration: one in four.
         *
         * Deliberately not [PREVIEW_STEP]. The two were one number while the preview
         * sampled one super-pixel in four, and separating them is what keeps the display
         * resolution from setting the cost of a measurement that does not want it: the
         * registration searches 17 by 17 offsets, so its cost is the grid's area times
         * 289, and it was the single most expensive thing the render thread did.
         *
         * Two is what every shake threshold in [CaptureSmear] was fitted against, so this
         * is also the value those numbers mean.
         */
        const val DRIFT_STEP = 2

        /**
         * Longest exposure a burst is taken at: an eighth of a second.
         *
         * Above this the frames stop being independent draws of the shake -- a long
         * exposure integrates the same slow drift that a short one samples, so there is
         * less and less to choose between them -- while each extra frame starts costing
         * its own exposure on top of its transfer.
         */
        const val BURST_MAX_EXPOSURE_US = 125_000L

        /** Frames a burst takes when it is on. */
        const val BURST_FRAMES = 4

        /** How often the steady-shutter wait re-reads the drift, in ms. */
        private const val STEADY_POLL_MS = 25L

        /**
         * Super-pixels walked per preview sample: every one of them.
         *
         * It was 2, sampling one super-pixel in four, from when the decode ran on the
         * thread that was also reading the bus. Measured on this rig now that it has its
         * own thread: one in four renders in 42 ms a frame and all of them in 95 ms,
         * against the 182 ms a full frame takes to arrive. The finer decode is therefore
         * free -- it costs render time the frame rate was spending on waiting anyway --
         * and it is the difference between a 306 x 256 picture and a 612 x 512 one on a
         * screen 1156 pixels wide.
         *
         * If the bus ever gets faster this is the first thing to re-measure: past about
         * 10 fps the render becomes the limit rather than the transfer.
         */
        const val PREVIEW_STEP = 1

        /** Consecutive stalls tolerated before the camera is called lost. */
        private const val MAX_STALLS = 5

        /**
         * Frames per second the camera is allowed to produce while previewing.
         *
         * Just under what this host decodes, so nothing accumulates in the camera's
         * buffer and the preview stays close to real time. Raising it past the decode
         * rate does not make the preview faster, only later.
         */
        /**
         * Ceiling on the preview cadence, whatever the region would allow.
         *
         * Seven and a half, down from ten, and lowered deliberately rather than
         * discovered: the preview now streams the whole sensor, and on this bus a full
         * Mono12p frame cannot arrive ten times a second at any cap. What a ceiling above
         * the delivery rate buys is not frames but latency, since the camera queues
         * everything the host does not collect.
         */
        const val PREVIEW_FPS = 7.5

        /** Floor, so a mistake in the arithmetic cannot stop the preview altogether. */
        const val MIN_PREVIEW_FPS = 1.0

        /**
         * What the bus carries for this camera, in bytes per second.
         *
         * Measured on this rig: a full 2448 x 2048 Mono12p frame is 7.5 MB and arrives at
         * 5.7 fps, which is 43 MB/s, on a USB 2.0 bus shared with the OAK. Set a little
         * under that so the cap it produces is a little under what the host can drain.
         *
         * It has moved twice, and both times because the host got faster rather than the
         * bus: 22 MB/s with synchronous 16 KB reads, 36 with queued 256 KB ones, 43 once
         * the decode moved off the thread doing the reading. Whenever one of those
         * changes this number has to change with it -- a cap above what the host drains
         * is exactly how the preview grew half a second of latency, because the camera
         * queues every frame nobody collects.
         *
         * It is a property of the bus and not of this camera, so it will be optimistic
         * whenever the OAK is streaming too. The consequence of being optimistic is
         * exactly the latency this exists to prevent, which is why it is set low.
         */
        const val PREVIEW_BYTES_PER_SECOND = 41_000_000.0
        val DEFAULT_EXPOSURE_RANGE = 20L..30_000_000L

        /** 50 ms: enough light indoors through the polarizer to frame a shot. */
        const val DEFAULT_EXPOSURE_US = 50_000L

        /**
         * Frames the camera is given to apply a change before it is metered again.
         *
         * Two, because a frame is already being exposed when the write lands and the one
         * after it may be too. Metering those would read the old exposure, conclude the
         * correction did not work, and apply it a second time -- the overshoot that makes
         * an auto exposure pump.
         */
        const val AE_SETTLE_FRAMES = 2

        /**
         * Whether a metered frame is new enough to write from.
         *
         * The subtlety is the second half. The frame id is the camera's own block
         * counter, and it restarts from zero every time acquisition does -- which is once
         * per capture. Compared as a clock that only ever goes forwards, every frame
         * after a capture looks older than the last write, and AUTO stops dead for as
         * many frames as it had already seen: after a few minutes of preview, minutes of
         * a frozen meter, while the pill still reads AUTO. A counter that went backwards
         * is a new stream, and the frame on it is the newest there is.
         */
        internal fun meterIsCurrent(frameId: Long, appliedAtFrameId: Long): Boolean =
            frameId < appliedAtFrameId || frameId > appliedAtFrameId + AE_SETTLE_FRAMES

        /**
         * The longest shutter AUTO will choose: 48 milliseconds.
         *
         * It is a handheld limit, and this time that is meant literally. The lens on this
         * rig is 6 mm, which on a 2/3-inch sensor is about a 24 mm field of view in
         * 35 mm terms, and the oldest rule in photography puts the slowest safe handheld
         * shutter at one over the equivalent focal length -- 1/24 of a second, rounded to
         * the nearest detent at 1/21. Nothing about the rule is precise, but it is the
         * right order of magnitude and it is what a person holding this thing expects.
         *
         * It used to be an eighth of a second, a number taken from [BURST_MAX_EXPOSURE_US]
         * -- which answers a different question entirely: the exposure past which taking
         * four frames stops helping. The archive agrees the old value was too long.
         * Frames were rejected at a flat 6 to 7 percent from 8 ms through 67 ms, and at
         * 18 percent at 125 ms; the ceiling sat at the one exposure where blur starts
         * costing real captures.
         *
         * Below the ceiling AUTO spends shutter before gain, which is the right order --
         * see [AutoExposure.allocate]. Above it, gain is all that is left, and in a dark
         * enough scene that is what it will do. Tying the ceiling to the measured shake
         * instead, since the gyroscope already reports in exactly these units, is the
         * better answer whenever someone has a tripod.
         */
        const val AE_MAX_EXPOSURE_US = 48_000L

        /**
         * The nodes every capture records the camera's own value for.
         *
         * Chosen to answer the questions an archive gets asked years later: what was the
         * exposure really, was anything auto, was the pipeline off, what was read out.
         * Named after This Polar Camera's `FLIR_METADATA_NODES` so the two sidecars can
         * be compared field by field. Any that this driver cannot reach simply do not
         * appear -- see [FlirCamera.readNamedNodes].
         */
        val METADATA_NODES = listOf(
            // What the frame was taken at
            "ExposureTime", "ExposureAuto", "ExposureMode",
            "Gain", "GainAuto",
            "BlackLevelAll", "BlackLevelClampingEnable",
            // What was read out
            "PixelFormat", "AdcBitDepth",
            "Width", "Height", "OffsetX", "OffsetY",
            "BinningHorizontal", "BinningVertical",
            "DecimationHorizontal", "DecimationVertical",
            "ReverseX", "ReverseY",
            // How it was acquired
            "AcquisitionMode", "AcquisitionFrameRate", "AcquisitionFrameRateEnable",
            "AcquisitionFrameCount", "TriggerMode", "TriggerSource",
            // Everything that could touch a pixel after the sensor
            "IspEnable", "FfcEnable", "GammaEnable", "Gamma",
            "SharpeningEnable", "SharpeningAuto",
            "ColorTransformationEnable", "SaturationEnable", "LUTEnable",
            "DefectCorrectStaticEnable", "BalanceWhiteAuto", "TestPattern",
            // The body itself
            "DeviceTemperature",
        )

        /** The arithmetic behind [exposureWindow], with nothing to construct to test it. */
        internal fun exposureWindowFor(
            triggeredAtHostNs: Long,
            leaderHostNs: Long,
            exposureUs: Long,
            keptIndex: Int,
        ): LongRange {
            val exposureNs = exposureUs.coerceAtLeast(0L) * 1000
            val fromTrigger = triggeredAtHostNs..(triggeredAtHostNs + exposureNs)
            if (keptIndex <= 0) return fromTrigger
            val start = leaderHostNs - exposureNs
            // A leader that appears to predate the trigger is a clock that cannot be
            // believed; fall back rather than record a window that never happened.
            if (start < triggeredAtHostNs) return fromTrigger
            return start..leaderHostNs
        }
    }
}
