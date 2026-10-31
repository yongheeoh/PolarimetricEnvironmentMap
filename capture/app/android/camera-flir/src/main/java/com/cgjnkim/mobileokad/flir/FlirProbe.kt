package com.cgjnkim.mobileokad.flir

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipInputStream

/**
 * Reads a FLIR camera's bootstrap registers and GenICam XML over the USB3 Vision
 * control channel.
 *
 * This exists to answer three questions before any Aravis work starts:
 *
 *  1. does the camera answer at all on Android (permission, claim, bulk transfers)?
 *  2. what USB speed did the link actually negotiate? The camera reports it in
 *     SBRM.CURRENT_SPEED, which settles the question a USB hub otherwise hides.
 *  3. which pixel formats does this particular model offer? The polarized colour
 *     sensor's format list is not the same as the mono one's, and the XML has it.
 */
object FlirProbe {

    private const val TAG = "FlirProbe"

    /** Teledyne FLIR / Point Grey. Product ids differ per model, so match on vendor. */
    const val VENDOR_ID = 0x1E10

    private const val ACTION_PERMISSION = "com.cgjnkim.mobileokad.flir.USB_PERMISSION"
    private val DUMP_PATH = "${Environment.DIRECTORY_DOCUMENTS}/MobileOkad"

    /** A GenICam XML far past this is a misread length, not a real document. */
    private const val XML_SIZE_LIMIT = 32 * 1024 * 1024

    data class Report(val summary: String, val xmlFileName: String?)

    fun findDevice(context: Context): UsbDevice? {
        val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        return manager.deviceList.values.firstOrNull { it.vendorId == VENDOR_ID }
    }

    fun hasPermission(context: Context, device: UsbDevice): Boolean {
        val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        return manager.hasPermission(device)
    }

    /**
     * Raises the system USB permission dialog and calls [onResult] once the user answers.
     *
     * Unlike the OAK, a USB3 Vision camera does not re-enumerate after boot, so this
     * happens exactly once per device rather than on every firmware transition.
     */
    fun requestPermission(context: Context, device: UsbDevice, onResult: (Boolean) -> Unit) {
        val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager

        val receiver = object : BroadcastReceiver() {
            override fun onReceive(ctx: Context, intent: Intent) {
                if (intent.action != ACTION_PERMISSION) return
                ctx.unregisterReceiver(this)
                onResult(intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false))
            }
        }
        val filter = IntentFilter(ACTION_PERMISSION)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            context.registerReceiver(receiver, filter)
        }

        // The system writes EXTRA_DEVICE and EXTRA_PERMISSION_GRANTED into this intent, so
        // it has to stay mutable; setting the package keeps it explicit enough for
        // Android 14's implicit-intent restriction.
        val pending = PendingIntent.getBroadcast(
            context,
            0,
            Intent(ACTION_PERMISSION).setPackage(context.packageName),
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_MUTABLE else 0,
        )
        manager.requestPermission(device, pending)
    }

    /** Blocking. Must not run on the main thread: every register read is a bulk transfer. */
    fun run(context: Context, device: UsbDevice): Report {
        val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        val connection = manager.openDevice(device)
            ?: throw U3vException("openDevice failed -- permission not granted?")

        U3vControl.open(device, connection).use { control ->
            control.configure()

            val lines = StringBuilder()
            fun line(label: String, value: Any?) {
                val text = "%-24s %s".format(label, value)
                lines.append(text).append('\n')
                Log.i(TAG, text)
            }

            line("USB", "%04x:%04x %s".format(device.vendorId, device.productId, device.deviceName))
            line("GENCP_VERSION", version(control.readU32(Abrm.GENCP_VERSION)))
            line("MANUFACTURER_NAME", control.readString(Abrm.MANUFACTURER_NAME))
            line("MODEL_NAME", control.readString(Abrm.MODEL_NAME))
            line("FAMILY_NAME", control.readString(Abrm.FAMILY_NAME))
            line("DEVICE_VERSION", control.readString(Abrm.DEVICE_VERSION))
            line("SERIAL_NUMBER", control.readString(Abrm.SERIAL_NUMBER))
            line("MAX_RESPONSE_TIME", "${control.maxDeviceResponseTimeMs} ms")
            line("SBRM_ADDRESS", hex(control.sbrmAddress))
            line("U3V_VERSION", version(control.readU32(control.sbrmAddress + Sbrm.U3V_VERSION)))
            line("MAX_CMD_TRANSFER", "${control.maxCmdTransfer} B")
            line("MAX_ACK_TRANSFER", "${control.maxAckTransfer} B")
            line("STREAM_CHANNELS", control.readU32(control.sbrmAddress + Sbrm.NUM_STREAM_CHANNELS))
            line("SIRM_ADDRESS", hex(control.readU64(control.sbrmAddress + Sbrm.SIRM_ADDRESS)))

            val speed = control.readU32(control.sbrmAddress + Sbrm.CURRENT_SPEED)
            line("CURRENT_SPEED", "${describeSpeed(speed)}  (raw ${hex(speed)})")

            val xml = readGenicamXml(control)
            val fileName = xml?.let { writeDump(context, device, it) }
            if (xml == null) {
                line("GENICAM_XML", "(read failed)")
                return Report(lines.toString().trimEnd(), null)
            }
            line("GENICAM_XML", "${xml.length} chars -> $fileName")

            val formats = pixelFormats(xml)
            line("PIXEL_FORMATS", "${formats.size} total")
            line("  polarized", formats.keys.filter { it.contains("Polarized") }.joinToString(", "))

            // Live reads, so these are the camera's current state rather than the
            // XML's defaults. Exposure bounds in particular only exist as registers.
            val gc = Genicam(xml)
            val current = gc.read(control, "PixelFormat_Val")
            line("  current", formats.entries.firstOrNull { it.value == current }?.key ?: "raw $current")

            val width = gc.read(control, "Width_Val")
            val height = gc.read(control, "Height_Val")
            val widthMax = gc.read(control, "WidthMax_Val")
            val heightMax = gc.read(control, "HeightMax_Val")
            line("ROI", "${width}x${height}  of ${widthMax}x${heightMax}")

            val expMin = gc.read(control, "ExposureTime_Min")
            val expMax = gc.read(control, "ExposureTime_Max")
            line("EXPOSURE_TIME", "$expMin .. $expMax us")

            line(
                "THROUGHPUT_LIMIT",
                "${gc.read(control, "DeviceLinkThroughputLimit_Val")} Bps" +
                    " (min ${gc.read(control, "DeviceLinkThroughputLimit_Min")}," +
                    " max ${gc.read(control, "DeviceLinkThroughputLimit_Max")})",
            )

            binningReport(gc, control, ::line)

            grabTest(control, gc, ::line)

            return Report(lines.toString().trimEnd(), fileName)
        }
    }

    /**
     * What the camera offers for reducing a frame on its own side, and on what terms.
     *
     * This is the question that decides whether a cheap full-field preview is possible
     * at all. The sensor's super-pixel is 4x4: a 2x2 polarizer unit (90/45 over 135/0)
     * inside each of the four Bayer cells, so colour repeats every 4 px and analyser
     * angle every 2. Whether binning survives that depends entirely on which pixels the
     * camera combines, and nothing but the camera can say:
     *
     *  - combining neighbours (a plain 2x2 sum) merges all four analysers, which is
     *    exactly S0 and is fine for an intensity preview, but leaves no polarization
     *    and no colour;
     *  - combining same-colour pixels the way a Bayer sensor normally does means a
     *    spacing of 2, which on this sensor is a different colour AND a different
     *    angle -- useless;
     *  - combining at a spacing of 4 would keep both, since 4 is a multiple of the
     *    polarizer period as well as the colour period. That is the only variant worth
     *    having, and the least likely to be what is implemented.
     *
     * So this reports what exists rather than assuming, and [grabTest] measures what it
     * actually does to the mosaic.
     */
    private fun binningReport(genicam: Genicam, control: U3vControl, line: (String, Any?) -> Unit) {
        for (node in listOf(
            "BinningHorizontal", "BinningVertical",
            "DecimationHorizontal", "DecimationVertical",
        )) {
            val current = genicam.read(control, "${node}_Val")
            if (current == null) {
                line(node.uppercase(), "no such register")
                continue
            }
            val min = genicam.read(control, "${node}_Min")
            val max = genicam.read(control, "${node}_Max")
            line(node.uppercase(), "$current (min $min, max $max)")
        }

        for (node in listOf(
            "BinningSelector", "BinningHorizontalMode", "BinningVerticalMode",
            "DecimationSelector", "DecimationHorizontalMode",
        )) {
            val e = genicam.enumeration(node)
            if (e == null) {
                line("  $node", "not an addressable enumeration")
                continue
            }
            val now = genicam.read(control, e.value)
            val selected = e.entries.entries.firstOrNull { it.value == now }?.key ?: "raw $now"
            line("  $node", "$selected  of [${e.entries.keys.joinToString(", ")}]")
        }
    }

    /**
     * Switches to the colour-polarized format and pulls one frame.
     *
     * The per-position means at the end are the real check: the four pixels of a
     * polarizer unit see the same scene through different analysers, so under polarized
     * light they must differ and under unpolarized light they must not. A mosaic read
     * with the wrong phase would scramble that relationship.
     */
    private fun grabTest(control: U3vControl, genicam: Genicam, line: (String, Any?) -> Unit) {
        val camera = FlirCamera(control, genicam)
        try {
            camera.pixelFormat = PixelFormats.BAYER_RG_POLARIZED_8
            line("SET_PIXEL_FORMAT", "BayerRGPolarized8 -> read back 0x%08x".format(camera.pixelFormat))
            line("ROI", "${camera.width}x${camera.height}")

            camera.startAcquisition()
            val started = System.currentTimeMillis()
            val frame = camera.grabFrame()
            val elapsed = System.currentTimeMillis() - started

            line("FRAME", "#${frame.frameId} ${frame.width}x${frame.height} fmt 0x%08x".format(frame.pixelFormat))
            line("  bytes", "${frame.data.size} in $elapsed ms (${frame.data.size / maxOf(1, elapsed)} kB/s)")

            mosaicMeans(frame, line)

            binnedGrab(camera, line)
        } finally {
            camera.stopAcquisition()
        }
    }

    /**
     * Turns binning on and measures the same frame again.
     *
     * This is the whole answer. Compare the two readings above and below:
     *
     *  - the 4x4 means keep their spread  -> colour survived, so pixels a multiple of
     *    4 apart were combined and binning is usable for the preview;
     *  - the 4x4 means flatten but the 2x2 means keep theirs -> the polarizer unit
     *    survived and colour did not;
     *  - both flatten -> neighbours were summed. That is exactly S0, so it is still a
     *    perfectly good intensity preview, but there is no polarization left in it and
     *    the AoLP and DoLP views cannot be driven from it.
     */
    private fun binnedGrab(camera: FlirCamera, line: (String, Any?) -> Unit) {
        val max = camera.binningMax
        if (max == null || max < 2) {
            line("BINNED_GRAB", "skipped: binning max is $max")
            return
        }
        camera.stopAcquisition()
        val why = camera.setBinning(2)
        if (why != null) {
            line("BINNED_GRAB", "binning 2 not applied: $why")
            runCatching { camera.setBinning(1) }
            return
        }
        try {
            line("BINNED_ROI", "${camera.width}x${camera.height}")
            camera.startAcquisition()
            val frame = camera.grabFrame()
            line("BINNED_FRAME", "${frame.width}x${frame.height}, ${frame.data.size} B")
            mosaicMeans(frame, line)
        } catch (e: Exception) {
            line("BINNED_GRAB", "failed: ${e.message}")
        } finally {
            runCatching {
                camera.stopAcquisition()
                camera.setBinning(1)
            }.onFailure { line("BINNED_GRAB", "could not restore binning 1: ${it.message}") }
        }
    }

    /**
     * The frame's structure, at both periods the sensor has one.
     *
     * The 2x2 breakdown is the analyser check: the four pixels of a polarizer unit see
     * the same scene through different analysers, so under polarized light they must
     * differ and under unpolarized light they must not. Point the camera at any LCD
     * screen for a decisive reading -- a monitor is very close to fully polarized, so
     * the four means should be far apart.
     *
     * The 4x4 breakdown is the colour check, and needs no special target: any real
     * scene puts different light in R, G and B. Both survive a reduction only if it
     * combined pixels a multiple of 4 apart.
     */
    private fun mosaicMeans(frame: FlirCamera.Frame, line: (String, Any?) -> Unit) {
        val unit = LongArray(4)
        val unitN = LongArray(4)
        val superPixel = LongArray(16)
        val superN = LongArray(16)
        var lo = 255
        var hi = 0
        for (y in 0 until frame.height) {
            val row = y * frame.width
            for (x in 0 until frame.width) {
                val v = frame.data[row + x].toInt() and 0xFF
                if (v < lo) lo = v
                if (v > hi) hi = v
                val pos = (y and 1) * 2 + (x and 1)
                unit[pos] += v
                unitN[pos]++
                val sp = (y and 3) * 4 + (x and 3)
                superPixel[sp] += v
                superN[sp]++
            }
        }
        line("  range", "$lo .. $hi")
        line(
            "  analyser means (2x2)",
            (0 until 4).joinToString(", ") { "p$it=%.1f".format(unit[it].toDouble() / unitN[it]) },
        )
        line(
            "  super-pixel means (4x4)",
            (0 until 4).joinToString("  |  ") { r ->
                (0 until 4).joinToString(" ") {
                    "%.0f".format(superPixel[r * 4 + it].toDouble() / superN[r * 4 + it])
                }
            },
        )
    }

    /**
     * Follows ABRM.MANIFEST_TABLE_ADDRESS to the first manifest entry and pulls the
     * document it points at. Entries are 64 bytes: subminor(2) minor(1) major(1)
     * schema(4) address(8) size(8) then five reserved 64 bit words.
     */
    private fun readGenicamXml(control: U3vControl): String? {
        try {
            val tableAddress = control.readU64(Abrm.MANIFEST_TABLE_ADDRESS)
            val entryCount = control.readU64(tableAddress)
            Log.i(TAG, "manifest at ${hex(tableAddress)}, $entryCount entries")
            if (entryCount <= 0) return null

            val entry = ByteBuffer.wrap(control.readMemory(tableAddress + 8, 64))
                .order(ByteOrder.LITTLE_ENDIAN)
            val schema = entry.getInt(4)
            val address = entry.getLong(8)
            val size = entry.getLong(16)

            // Schema type lives in bits 10-14 of the schema word: 0 raw, 1 zipped.
            val zipped = ((schema ushr 10) and 0x1F) == 1
            Log.i(TAG, "genicam at ${hex(address)}, $size bytes, zipped=$zipped")
            if (size <= 0 || size > XML_SIZE_LIMIT) {
                Log.w(TAG, "implausible genicam size $size")
                return null
            }

            val raw = control.readMemory(address, size.toInt())
            return if (zipped) unzipFirstEntry(raw) else String(raw, Charsets.UTF_8)
        } catch (e: Exception) {
            Log.e(TAG, "genicam read failed", e)
            return null
        }
    }

    private fun unzipFirstEntry(raw: ByteArray): String? {
        ZipInputStream(ByteArrayInputStream(raw)).use { zip ->
            var entry = zip.nextEntry
            while (entry != null) {
                if (!entry.isDirectory) {
                    Log.i(TAG, "zip entry ${entry.name}")
                    return zip.readBytes().toString(Charsets.UTF_8)
                }
                entry = zip.nextEntry
            }
        }
        return null
    }

    /**
     * Name to PFNC value for every entry of the PixelFormat enumeration.
     *
     * This camera declares its EnumEntry elements inline rather than as pEnumEntry
     * references, so the whole list is readable without a GenApi parse.
     */
    private fun pixelFormats(xml: String): Map<String, Long> {
        val block = Regex(
            """<Enumeration\s+Name="PixelFormat".*?</Enumeration>""",
            RegexOption.DOT_MATCHES_ALL,
        ).find(xml)?.value ?: return emptyMap()

        return Regex(
            """<EnumEntry\s+Name="([^"]+)".*?<Value>(0x[0-9A-Fa-f]+|\d+)</Value>""",
            RegexOption.DOT_MATCHES_ALL,
        ).findAll(block).associate { m ->
            val raw = m.groupValues[2]
            m.groupValues[1] to if (raw.startsWith("0x")) raw.substring(2).toLong(16) else raw.toLong()
        }
    }

    /**
     * USB3 Vision reports the negotiated speed as a bitfield. The raw value is logged
     * alongside the decode so a mapping change cannot silently mislead.
     */
    internal fun describeSpeed(value: Long): String {
        val names = buildList {
            if (value and 0x1L != 0L) add("LowSpeed")
            if (value and 0x2L != 0L) add("FullSpeed")
            if (value and 0x4L != 0L) add("HighSpeed (USB 2.0)")
            if (value and 0x8L != 0L) add("SuperSpeed (USB 3.x)")
            if (value and 0x10L != 0L) add("SuperSpeedPlus")
        }
        return if (names.isEmpty()) "unrecognised" else names.joinToString(" + ")
    }

    private fun version(value: Long): String = "${(value shr 16) and 0xFFFF}.${value and 0xFFFF}"

    private fun hex(value: Long): String = "0x%016x".format(value)

    /**
     * Writes the dump under Documents/, not Pictures/: MediaStore rejects non-media
     * files there, the same constraint CaptureStore already works around.
     */
    private fun writeDump(context: Context, device: UsbDevice, xml: String): String? {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val name = "${stamp}_flir_${device.serialNumberOrUnknown()}.xml"

        // RELATIVE_PATH and IS_PENDING are API 29. On older phones (the LG V30 is Android 9) the
        // dump goes to the app's own documents folder, which needs no permission and is still
        // reachable over adb / MTP under Android/data/<package>/files/Documents.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return try {
                val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: return null
                java.io.File(dir, name).writeBytes(xml.toByteArray())
                name
            } catch (e: Exception) {
                Log.e(TAG, "dump write failed", e)
                null
            }
        }

        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "text/xml")
            put(MediaStore.MediaColumns.RELATIVE_PATH, DUMP_PATH)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values
        ) ?: return null

        return try {
            resolver.openOutputStream(uri)?.use { it.write(xml.toByteArray()) }
                ?: throw IllegalStateException("could not open $name")
            values.clear()
            values.put(MediaStore.MediaColumns.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            name
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            Log.e(TAG, "dump write failed", e)
            null
        }
    }

    /** Reading the serial needs permission we may not have when naming the file. */
    private fun UsbDevice.serialNumberOrUnknown(): String =
        try {
            serialNumber ?: "unknown"
        } catch (e: SecurityException) {
            "unknown"
        }
}
