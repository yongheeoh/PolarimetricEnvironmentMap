package kr.penvmap.capture

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.*
import android.util.AtomicFile
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kr.penvmap.capture.core.Json
import kr.penvmap.capture.core.asMap
import java.io.File
import java.io.RandomAccessFile

/** One offline job. A separate process keeps native scientific memory out of capture. */
object ProcessingJobs {
    fun directory(c: Context) = File(c.filesDir, "processing").apply { mkdirs() }
    private fun stateFile(c: Context) = AtomicFile(File(directory(c), "status.json"))
    fun status(c: Context): Map<String, Any?> = runCatching {
        Json.parse(stateFile(c).openRead().bufferedReader().use { it.readText() }).asMap()
    }.getOrDefault(emptyMap())
    fun save(c: Context, value: Map<String, Any?>) {
        val file = stateFile(c); val stream = file.startWrite()
        try { stream.write(Json.write(value).toByteArray()); file.finishWrite(stream) }
        catch (e: Throwable) { file.failWrite(stream); throw e }
    }
    fun active(c: Context, s: Map<String, Any?> = status(c)): Boolean {
        val pid = (s["pid"] as? Number)?.toInt() ?: 0
        val alive = pid > 0 && (c.getSystemService(ActivityManager::class.java).runningAppProcesses.orEmpty()
            .any { it.pid == pid && it.processName == c.packageName + ":processing" })
        return alive || (s["state"] == "queued" && pid == 0 &&
            System.currentTimeMillis() - ((s["started"] as? Number)?.toLong() ?: 0L) < 30_000)
    }
    fun owns(c: Context, root: File): Boolean {
        val s = status(c); return active(c, s) && s["scene"] == root.absolutePath
    }
    fun cancel(c: Context) { File(directory(c), "cancel").writeText("cancel") }
    fun cancellationRequested(c: Context) = File(directory(c), "cancel").exists()
    fun start(c: Context, root: File, mode: String) {
        require(mode in listOf("s0", "full"))
        check(!active(c)) { "Another scene is processing" }
        File(directory(c), "cancel").delete()
        save(c, mapOf("state" to "queued", "scene" to root.absolutePath, "mode" to mode,
            "progress" to 0, "stage" to "Starting on this phone", "pid" to 0, "started" to System.currentTimeMillis()))
        try {
            c.startForegroundService(Intent(c, ProcessingService::class.java)
                .putExtra("scene", root.absolutePath).putExtra("mode", mode))
        } catch (e: Exception) { save(c, status(c) + mapOf("state" to "error", "stage" to e.toString())); throw e }
    }
}

class ProcessingService : Service() {
    private var worker: Thread? = null
    @Volatile private var cancelled = false
    private var wake: PowerManager.WakeLock? = null
    private var state: Map<String, Any?> = emptyMap()
    private var lastNotice = 0L
    private var lastProgressWrite = 0L
    override fun onBind(intent: Intent?) = null

    override fun onCreate() {
        super.onCreate()
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel("processing", "Envmap processing", NotificationManager.IMPORTANCE_LOW))
    }

    private fun notification(text: String, progress: Int): Notification {
        val open = PendingIntent.getActivity(this, 8, Intent(this, MainActivity::class.java)
            .putExtra("action", "gallery"), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, 9, Intent(this, ProcessingService::class.java)
            .setAction("cancel"), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, "processing").setSmallIcon(android.R.drawable.ic_menu_gallery)
            .setContentTitle("Penvmap · ${if (state["mode"] == "full") "Full Stokes" else "S0"}")
            .setContentText(text).setContentIntent(open).setOngoing(true)
            .setProgress(100, progress, progress == 0)
            .addAction(Notification.Action.Builder(null, "Cancel", stop).build()).build()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "cancel") { cancelled = true; ProcessingJobs.cancel(this); return START_NOT_STICKY }
        if (worker != null) return START_NOT_STICKY
        val scene = intent?.getStringExtra("scene") ?: run { stopSelf(); return START_NOT_STICKY }
        val mode = intent.getStringExtra("mode") ?: "s0"
        state = ProcessingJobs.status(this) + mapOf("state" to "running", "pid" to android.os.Process.myPid())
        ProcessingJobs.save(this, state)
        if (Build.VERSION.SDK_INT >= 29) {
            val type = if (Build.VERSION.SDK_INT >= 35) ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING else ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            startForeground(730, notification("Loading offline processing", 0), type)
        } else startForeground(730, notification("Loading offline processing", 0))
        wake = (getSystemService(POWER_SERVICE) as PowerManager).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "penvmap:processing").also { it.acquire(6 * 60 * 60 * 1000L) }
        worker = Thread({
            try {
                RandomAccessFile(File(ProcessingJobs.directory(this), "job.lock"), "rw").use { file ->
                    val lock = file.channel.tryLock() ?: error("Another scene is processing")
                    try {
                        val root = File(scene).canonicalFile
                        val meta = Json.parse(File(root, "meta/capture_meta.json").readText()).asMap()
                        check((meta["session_timing"] as? Map<*, *>)?.get("finished") == true) { "Finish this capture before processing" }
                        check(meta["capture_kind"] != "calibration") { "Select a scene, not a calibration capture" }
                        check((meta["camera"] as? String)?.lowercase() in listOf(null, "flir", "fake")) {
                            "Offline processing supports FLIR captures; use the PC for this camera"
                        }
                        val profiles = File(filesDir, "processing_profiles").apply { mkdirs() }
                        for (name in assets.list("processing_profiles").orEmpty()) {
                            assets.open("processing_profiles/$name").use { input -> File(profiles, name).outputStream().use { input.copyTo(it) } }
                        }
                        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
                        val result = Python.getInstance().getModule("mobile.pipeline").callAttr("run", scene,
                            profiles.absolutePath, mode, if (mode == "s0") 1024 else 2048, Callback()).toString()
                        val report = Json.parse(File(result, "run.json").readText()).asMap()
                        val notes = (report["warnings"] as? List<*>)?.size ?: 0
                        val missing = ((report["coverage"] as? Map<*, *>)?.get("remaining_missing_pixels") as? Number)?.toInt() ?: 0
                        val summary = "Complete" + (if (missing > 0) " · $missing uncovered pixels" else " · full coverage") +
                            (if (notes > 0) "\nReview masks/registration in envmap/analysis ($notes notes in run.json)." else "")
                        update("complete", summary, 100, result)
                    } finally { lock.release() }
                }
            } catch (e: Throwable) {
                android.util.Log.e("PenvmapProcessing", "Processing failed", e)
                update(if (isCancelled()) "cancelled" else "error", if (isCancelled()) "Cancelled; previous results preserved" else (e.message ?: e.toString()), (state["progress"] as? Number)?.toInt() ?: 0)
            } finally {
                wake?.let { if (it.isHeld) it.release() }
                stopForeground(STOP_FOREGROUND_REMOVE); stopSelf()
                // This process contains only the worker service; release NumPy/native allocations.
                Handler(Looper.getMainLooper()).postDelayed({ android.os.Process.killProcess(android.os.Process.myPid()) }, 1500)
            }
        }, "envmap-offline").also { it.start() }
        return START_NOT_STICKY
    }

    private fun isCancelled() = cancelled || File(ProcessingJobs.directory(this), "cancel").exists()
    inner class Callback {
        fun isCancelled(): Boolean = this@ProcessingService.isCancelled()
        fun onProgress(progress: Int, stage: String) {
            val now = SystemClock.elapsedRealtime()
            if (progress == (state["progress"] as? Number)?.toInt() && now - lastProgressWrite < 500) return
            lastProgressWrite = now
            update("running", stage, progress)
        }
    }
    @Synchronized private fun update(status: String, stage: String, progress: Int, result: String? = null) {
        state = state + mapOf("state" to status, "stage" to stage, "progress" to progress, "updated" to System.currentTimeMillis())
        if (result != null) state = state + ("result" to result)
        ProcessingJobs.save(this, state)
        if (System.currentTimeMillis() - lastNotice > 1000 && status == "running") {
            getSystemService(NotificationManager::class.java).notify(730, notification(stage, progress)); lastNotice = System.currentTimeMillis()
        }
    }
    override fun onTimeout(startId: Int, fgsType: Int) { cancelled = true; ProcessingJobs.cancel(this); update("error", "Android processing time limit reached; retry from Gallery", 0); stopSelf() }
    override fun onDestroy() { cancelled = true; wake?.let { if (it.isHeld) it.release() }; super.onDestroy() }
}
