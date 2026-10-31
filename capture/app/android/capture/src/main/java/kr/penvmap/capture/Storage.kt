package kr.penvmap.capture

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.storage.StorageManager
import android.os.storage.StorageVolume
import android.provider.DocumentsContract
import kr.penvmap.capture.core.SessionIndex
import java.io.File

/**
 * Where sessions are written: the base path (= the PC's base_dir).
 *
 * The base path is the phone's Documents/penvmap, or a folder the user picked on a USB SSD / SD card
 * ([CUSTOM], 2026-10-13). Android lets an app write such a drive as plain files only with "All files
 * access" (MANAGE_EXTERNAL_STORAGE, Android 11+), which includes the root of USB OTG drives and SD cards;
 * the session writer, Gallery and Resume then work on it unchanged. A drive that is not mounted is never
 * replaced by the phone silently: captures ask first (MainActivity), a rotation stops and can be resumed.
 */
object Storage {
    /** Android/data/kr.penvmap.capture/files/penvmap -- where scenes went before 2026-10-13; moved to Documents/penvmap at start. */
    const val APP = "app"
    /** Documents/penvmap -- kept when the app is removed, visible in My Files and over USB file transfer. */
    const val DOCUMENTS = "documents"
    /** A folder the user picked (USB SSD, SD card, ...): [AppSettings.customBasePath]. */
    const val CUSTOM = "custom"

    /**
     * The SSD quick choice, under the drive's root: the T9 is E: on the PC, so sessions land in
     * <data-root>/<project>/<date>/<time> when it is plugged back in.
     */
    const val SSD_DIR = "penvmap/data/dataset_FLIR"

    fun baseDir(context: Context, mode: String): File = when (mode) {
        @Suppress("DEPRECATION")
        DOCUMENTS -> File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), "penvmap")
        else -> File(context.getExternalFilesDir(null), "penvmap")
    }

    /** The base path the settings select: the picked folder, or Documents/penvmap. */
    fun baseDir(context: Context, s: AppSettings): File =
        if (s.storageMode == CUSTOM && s.customBasePath != null) appPath(File(s.customBasePath)) else baseDir(context, DOCUMENTS)

    /**
     * The path an app uses for a drive: some phones (Samsung, 2026-10-14) report a USB drive's directory as the
     * raw mount /mnt/media_rw/<id>, which apps cannot open; the same drive is /storage/<id> for apps.
     */
    fun appPath(path: String): String = path.replace(Regex("^/mnt/media_rw/"), "/storage/")

    fun appPath(dir: File): File = File(appPath(dir.path.replace(File.separatorChar, '/')))

    @Suppress("DEPRECATION")
    private fun phoneRoot(): String = Environment.getExternalStorageDirectory().absolutePath

    /** On the phone's own shared storage (always there), not on a removable drive. */
    fun isPhone(dir: File): Boolean = dir.absolutePath.let { it.startsWith(phoneRoot()) || it.startsWith("/sdcard") }

    /** "/storage/1234-ABCD/penvmap" -> /storage/1234-ABCD (null on the phone). */
    private fun volumeRoot(dir: File): File? = Regex("^/storage/([^/]+)").find(dir.absolutePath)
        ?.groupValues?.get(1)?.takeIf { it != "emulated" && it != "self" }?.let { File("/storage/$it") }

    private fun volume(context: Context, dir: File): StorageVolume? =
        runCatching { context.getSystemService(StorageManager::class.java).getStorageVolume(dir) }.getOrNull()

    /** The phone, or a drive that is mounted (writable) right now. */
    fun available(context: Context, dir: File): Boolean = offlineReason(context, dir) == null

    /**
     * Why [dir] cannot be written now, or null when it can: the drive is not there, is mounted read-only
     * (exFAT pulled out without ejecting is mounted that way), or is in another state. A folder that is
     * really there and writable counts as available whatever the volume reports.
     */
    fun offlineReason(context: Context, dir: File): String? {
        if (isPhone(dir)) return null
        var d: File? = dir
        while (d != null && !d.exists()) d = d.parentFile
        if (d != null && d.absolutePath != "/storage" && d.absolutePath != "/" && d.isDirectory && d.canWrite()) return null
        val vol = volume(context, dir)
        val state = vol?.state ?: volumeRoot(dir)?.let { Environment.getExternalStorageState(it) }
        return when (state) {
            Environment.MEDIA_MOUNTED -> if (d == null || !d.canWrite()) "drive found but not writable (All files access?)" else null
            Environment.MEDIA_MOUNTED_READ_ONLY -> "drive is mounted read-only -- eject it properly on the PC, or check it there"
            null, Environment.MEDIA_UNKNOWN, Environment.MEDIA_REMOVED, Environment.MEDIA_UNMOUNTED ->
                "drive ${volumeRoot(dir)?.name ?: ""} not found" + mountedDrives(context).let { if (it.isEmpty()) " (no USB drive mounted)" else " -- mounted now: $it" }
            else -> "drive state: $state"
        }
    }

    /** "T9 (/storage/1234-ABCD, mounted)" for every drive besides the phone, for messages. */
    fun mountedDrives(context: Context): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return ""
        return context.getSystemService(StorageManager::class.java).storageVolumes.filter { !it.isPrimary }
            .joinToString(", ") { "${it.getDescription(context)} (${it.directory?.let { d -> appPath(d).path } ?: "?"}, ${it.state})" }
    }

    /** "Phone", or the drive's name as Android shows it ("Samsung USB", "SD card", ...). */
    fun volumeName(context: Context, dir: File): String =
        if (isPhone(dir)) "Phone" else volume(context, dir)?.getDescription(context) ?: "USB drive"

    /** "/storage/emulated/0/Documents/penvmap" -> "Phone/Documents/penvmap" (what My Files / USB file transfer show). */
    fun readable(path: String): String = path.replace(Regex("^/storage/emulated/0"), "Phone").replace(Regex("^/sdcard"), "Phone")

    /** As [readable], with a drive's name: "/storage/1234-ABCD/penvmap" -> "Samsung USB/penvmap". */
    fun readable(context: Context, path: String): String {
        val dir = File(path)
        val root = volumeRoot(dir) ?: return readable(path)
        return volumeName(context, dir) + path.removePrefix(root.absolutePath)
    }

    fun label(context: Context, mode: String, dir: File) = when (mode) {
        CUSTOM -> readable(context, dir.absolutePath)
        DOCUMENTS -> "Phone · Documents/penvmap"
        else -> "Phone"
    }

    /** [dir] is the SSD quick choice on some drive. */
    fun isSsdDefault(dir: File): Boolean = !isPhone(dir) && dir.absolutePath.trimEnd('/').endsWith("/$SSD_DIR")

    /** Mounted USB drives / SD cards (name, root), Android 11+. */
    fun usbDrives(context: Context): List<Pair<String, File>> {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return emptyList()
        return context.getSystemService(StorageManager::class.java).storageVolumes
            .filter { !it.isPrimary && it.isRemovable && it.state == Environment.MEDIA_MOUNTED }
            .mapNotNull { v -> v.directory?.let { v.getDescription(context) to appPath(it) } }
    }

    /** Free bytes and used fraction of the drive [dir] is on (its nearest existing folder), or null. */
    fun usage(dir: File): Pair<Long, Double>? {
        var d: File? = dir
        while (d != null && !d.exists()) d = d.parentFile
        val total = d?.totalSpace ?: return null
        if (total <= 0L) return null
        val free = d.usableSpace
        return free to (1.0 - free.toDouble() / total)
    }

    /** Android 9 / 10 need the old storage permission to write to Documents/. */
    fun needsLegacyPermission(mode: String) = mode == DOCUMENTS && Build.VERSION.SDK_INT < Build.VERSION_CODES.R

    /** "All files access": needed to write a picked folder on a USB drive / SD card as plain files. */
    fun hasAllFilesAccess(): Boolean = Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && Environment.isExternalStorageManager()

    /**
     * The folder a picked tree URI stands for ("1234-ABCD:penvmap" -> /storage/1234-ABCD/penvmap). A drive's
     * root becomes <root>/penvmap, like the phone's Documents/penvmap. Null for other document providers.
     */
    fun pathFromTree(context: Context, tree: Uri): File? {
        if (tree.authority != "com.android.externalstorage.documents") return null
        val docId = runCatching { DocumentsContract.getTreeDocumentId(tree) }.getOrNull() ?: return null
        return pathFromDocId(docId, File(phoneRoot())) { uuid ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                context.getSystemService(StorageManager::class.java).storageVolumes
                    .firstOrNull { it.uuid.equals(uuid, ignoreCase = true) }?.directory
            } else {
                null
            }
        }
    }

    /** [pathFromTree] without Android: [volumeDir] maps a volume id to its mount point (else /storage/<id>). */
    fun pathFromDocId(docId: String, primary: File, volumeDir: (String) -> File?): File {
        val volId = docId.substringBefore(':')
        val rel = docId.substringAfter(':', "").trim('/')
        val root = if (volId == "primary") primary else volumeDir(volId)?.let { appPath(it) } ?: File("/storage/$volId")
        return if (rel.isEmpty()) File(root, "penvmap") else File(root, rel)
    }

    /** Creates [dir] and writes / reads back a small file; null when that works, else why not. */
    fun probeWrite(dir: File): String? = try {
        if (!dir.isDirectory && !dir.mkdirs()) {
            "cannot create the folder"
        } else {
            val f = File(dir, ".penvmap_write_test")
            f.writeBytes(ByteArray(4096))
            val ok = f.length() == 4096L
            f.delete()
            if (ok) null else "the test file came back with the wrong size"
        }
    } catch (e: Exception) {
        e.message ?: e.toString()
    }

    /** The base paths in use: the picked folder (when it is set and connected) and Documents/penvmap. */
    fun roots(context: Context, s: AppSettings): List<Pair<String, File>> {
        val out = ArrayList<Pair<String, File>>()
        s.customBasePath?.let { File(it) }?.takeIf { available(context, it) }?.let { out += CUSTOM to it }
        out += DOCUMENTS to baseDir(context, DOCUMENTS)
        return out.distinctBy { it.second.absolutePath }
    }

    /**
     * Moves the scenes of the old app folder (before 2026-10-13) into Documents/penvmap, same
     * <project>/<date>/<time> paths; a copy is checked file by file (sizes) before the old one is deleted.
     * Scenes already in Documents are left alone. Returns how many scenes moved.
     */
    fun migrateOldAppFolder(context: Context): Int {
        val old = baseDir(context, APP)
        val dest = baseDir(context, DOCUMENTS)
        var moved = 0
        for (info in SessionIndex.list(old)) {
            val to = File(dest, info.relPath)
            if (to.exists()) continue
            to.parentFile?.mkdirs()
            if (!info.root.renameTo(to)) {
                info.root.copyRecursively(to, overwrite = false)
                val same = info.root.walkTopDown().filter { it.isFile }.all { f -> File(to, f.relativeTo(info.root).path).length() == f.length() }
                if (!same) {
                    to.deleteRecursively()
                    continue
                }
                info.root.deleteRecursively()
            }
            moved++
        }
        // empty project / date folders left behind
        old.walkBottomUp().filter { it.isDirectory && it != old && it.list()?.isEmpty() == true }.forEach { it.delete() }
        return moved
    }
}
