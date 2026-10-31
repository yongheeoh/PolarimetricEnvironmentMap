package kr.penvmap.capture

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.util.Log
import com.cgjnkim.mobileokad.flir.PenvmapCamera
import kr.penvmap.capture.core.FakeWorld
import kr.penvmap.capture.core.HttpStage
import kr.penvmap.capture.core.Stage
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean

/** A preview frame: 8-bit raw mosaic, sensor orientation. */
class PreviewFrame(val pixels: ByteArray, val width: Int, val height: Int, val exposureUs: Long)

/** One verified bracket: BayerRG16 frames (little-endian uint16), measured exposures, info for the json. */
class BracketFrames(val frames: List<ByteArray>, val width: Int, val height: Int, val exposureUs: List<Double>, val info: Map<String, Any?>)

/** What the controller needs from a camera: the real FLIR or the synthetic one. */
interface CameraSource {
    val label: String
    val exposureRangeUs: LongRange
    fun open(context: Context)
    fun setPreview(active: Boolean, exposureUs: Long, listener: ((PreviewFrame) -> Unit)?)
    fun setPreviewExposure(exposureUs: Long)
    fun captureBracket(exposuresUs: List<Long>): BracketFrames
    fun settings(): Map<String, Any?>
    fun deviceInfo(): Map<String, String>
    fun close()

    /** Raw bytes worth keeping when a frame could not be interpreted (FLIR chunk tail), else null. */
    fun diagnosticBytes(): ByteArray? = null
}

/** The FLIR over USB3 Vision ([PenvmapCamera]). [onError] hears about a preview that keeps failing. */
class FlirSource(private val options: PenvmapCamera.Options, onError: (String) -> Unit = {}) : CameraSource {
    private val cam = PenvmapCamera().apply { errorListener = onError }
    override val label = "FLIR"
    override val exposureRangeUs: LongRange get() = cam.exposureRangeUs

    override fun open(context: Context) = cam.open(context, options)

    override fun setPreview(active: Boolean, exposureUs: Long, listener: ((PreviewFrame) -> Unit)?) {
        cam.setPreview(active, exposureUs, listener?.let { l ->
            PenvmapCamera.PreviewListener { p -> l(PreviewFrame(p.pixels, p.width, p.height, p.exposureUs)) }
        })
    }

    override fun setPreviewExposure(exposureUs: Long) = cam.setPreviewExposure(exposureUs)

    override fun captureBracket(exposuresUs: List<Long>): BracketFrames {
        val b = cam.captureBracket(exposuresUs)
        return BracketFrames(b.frames, b.width, b.height, b.exposureUs, b.info)
    }

    override fun settings(): Map<String, Any?> = cam.readSettings() + mapOf("preset" to cam.presetOutcomes)
    override fun deviceInfo(): Map<String, String> = cam.deviceInfo
    override fun close() = cam.close()
    override fun diagnosticBytes(): ByteArray? = cam.lastPayloadTail
}

/**
 * Synthetic camera over [FakeWorld]: renders what the FLIR would see at the fake stage's pose, as a
 * BayerRG8 preview and BayerRG16 brackets with the same sizes and file layout as the real one.
 */
class FakeSource(private val world: FakeWorld, previewFps: Double = 4.0) : CameraSource {
    override val label = "FAKE"
    override val exposureRangeUs: LongRange = 21L..30_000_000L
    private val running = AtomicBoolean(false)
    @Volatile private var exposureUs = 26_280L
    @Volatile private var listener: ((PreviewFrame) -> Unit)? = null
    private var thread: Thread? = null
    private val lock = Object()
    private val frameMs = (1000.0 / previewFps).toLong().coerceAtLeast(30L)

    override fun open(context: Context) {
        if (offline()) throw java.io.IOException("FAKE camera unplugged (simulated)")
        running.set(true)
        thread = Thread({
            while (running.get()) {
                val l = listener
                if (l == null || offline()) {
                    Thread.sleep(200)
                    continue
                }
                val px = synchronized(lock) { world.mosaic(exposureUs.toDouble(), bits16 = false) }
                l(PreviewFrame(px, world.width, world.height, exposureUs))
                Thread.sleep(frameMs)
            }
        }, "fake-camera").apply { start() }
    }

    override fun setPreview(active: Boolean, exposureUs: Long, listener: ((PreviewFrame) -> Unit)?) {
        this.exposureUs = exposureUs
        this.listener = if (active) listener else null
    }

    override fun setPreviewExposure(exposureUs: Long) {
        this.exposureUs = exposureUs
    }

    override fun captureBracket(exposuresUs: List<Long>): BracketFrames = synchronized(lock) {
        if (offline()) throw java.io.IOException("FAKE camera unplugged (simulated)")
        val t0 = System.nanoTime()
        val radiance = world.superpixels()
        val frames = exposuresUs.map { world.mosaic(it.toDouble(), bits16 = true, radiance = radiance) }
        // Exposure time passes on a real camera; keep the order of magnitude without stalling tests.
        Thread.sleep(minOf(exposuresUs.sum() / 1000, 1500L))
        val info = linkedMapOf<String, Any?>(
            "mode" to "fake", "segments" to 1, "retries" to 0,
            "frame_ids" to exposuresUs.indices.toList(), "sequencer_sets" to exposuresUs.indices.toList(),
            "requested_exposure_us" to exposuresUs.map { it.toDouble() },
            "bracket_time_s" to Math.round((System.nanoTime() - t0) / 1e6) / 1000.0, "transport" to "fake",
        )
        BracketFrames(frames, world.width, world.height, exposuresUs.map { it.toDouble() }, info)
    }

    override fun settings(): Map<String, Any?> = linkedMapOf("PixelFormat" to "BayerRG16", "Transport" to "fake")
    override fun deviceInfo(): Map<String, String> = mapOf("model" to "FakeWorld", "serial" to "0")
    override fun close() {
        running.set(false)
        listener = null
    }

    companion object {
        /** Until when the fake camera behaves as unplugged (adb `--es action fake_drop`), to test reconnection. */
        @Volatile var offlineUntilMs = 0L

        fun offline() = System.currentTimeMillis() < offlineUntilMs
    }
}

/** Fake stage that turns the fake world, so a fake rotation stitches into a panorama. */
class FakeWorldStage(private val world: FakeWorld, private val moveS: Double = 0.2) : Stage {
    override fun ping() = true
    override fun moveAbs(yawDeg: Double?, pitchDeg: Double?, large: Boolean) {
        yawDeg?.let { world.yawDeg = it }
        pitchDeg?.let { world.pitchDeg = it }
        Thread.sleep((moveS * 1000).toLong())
    }
}

object StageFactory {
    private const val TAG = "StageFactory"

    /**
     * HTTP stage bound to a Wi-Fi network when there is one: a Wi-Fi without internet (the stage's
     * router) is otherwise passed over for mobile data and 192.168.8.1 is unreachable.
     */
    fun http(context: Context, s: AppSettings, log: (String) -> Unit): Stage {
        val wifi = wifiNetwork(context)
        Log.i(TAG, "stage ${s.mechaUrl} via ${if (wifi != null) "Wi-Fi $wifi" else "default network"}")
        val open: (URL) -> java.net.URLConnection = { url -> wifiNetwork(context)?.openConnection(url) ?: url.openConnection() }
        return HttpStage(s.mechaUrl, s.stageSettlePitchS, s.stageSettleYawS, open, log)
    }

    fun wifiNetwork(context: Context): Network? {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        @Suppress("DEPRECATION")
        return cm.allNetworks.firstOrNull { n -> cm.getNetworkCapabilities(n)?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true }
    }
}
