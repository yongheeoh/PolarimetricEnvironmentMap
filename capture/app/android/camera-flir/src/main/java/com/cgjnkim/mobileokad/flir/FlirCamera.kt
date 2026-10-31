package com.cgjnkim.mobileokad.flir

import android.util.Log
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Drives a Blackfly S over USB3 Vision: register writes for configuration, and the
 * stream endpoint for images.
 *
 * There is deliberately no Aravis and no native code behind this. Every node needed to
 * run the camera resolves to a literal register address in its own GenICam XML, so a
 * GenApi engine would buy nothing that Genicam's name lookup does not already give.
 */
internal class FlirCamera(
    private val control: U3vControl,
    private val genicam: Genicam,
) : Closeable {

    /** One acquired frame, still in the sensor's mosaic layout. */
    class Frame(
        val data: ByteArray,
        val width: Int,
        val height: Int,
        val pixelFormat: Long,
        /**
         * The device's own timestamp from the leader, in the camera's units.
         *
         * Useful for ordering this camera's own frames. Not comparable to anything else:
         * the unit is whatever the camera uses, and on this model it is demonstrably not
         * the same basis as the ABRM timestamp register, so converting it through
         * TIMESTAMP_INCREMENT produced times hours away from reality.
         */
        val deviceTimestamp: Long,
        /**
         * System.nanoTime() when the leader arrived, which is the usable time reference.
         *
         * The leader is sent at the start of the block transfer, so this trails the end
         * of the exposure by the leader's own latency -- microseconds on this link, and
         * measured rather than assumed to be zero. It is on the same clock as the OAK's
         * frame timestamps, so the two cameras can finally be compared.
         */
        val leaderHostNs: Long,
        val frameId: Long,
        /**
         * How long this frame's payload took to cross the bus, in nanoseconds.
         *
         * Measured around the payload transfers alone, which is what makes it a
         * measurement of the *bus* and not of the frame rate. While the payload is moving
         * the camera is sending as fast as the link allows; the waiting a frame-rate cap
         * causes happens before the leader, not inside the payload. [PreviewCadence] turns
         * this into the rate the preview is allowed to run at.
         */
        val payloadNs: Long = 0,
        /** Payload bytes this frame carried, the numerator for [payloadNs]. */
        val payloadBytes: Long = 0,
        /**
         * The valid payload size from the USB3 Vision trailer, or -1 when the trailer did not say.
         *
         * With chunk data on (payload type 0x4001) the camera's REQ_PAYLOAD_SIZE is the buffer it needs
         * at most, rounded up: on the BFS-U3-51S5PC a BayerRG16 frame plus its chunks is 10 027 096 B
         * of a 10 028 032 B payload (measured through Spinnaker, 2026-10-08). The chunk tags are at
         * the end of the *valid* bytes, so a parser walking back from the end of [data] needs this.
         */
        val validPayloadBytes: Long = -1,
    )

    // Resolved on first use: it needs the SBRM address that control.configure() reads.
    private val sirm: Long by lazy { control.readU64(control.sbrmAddress + Sbrm.SIRM_ADDRESS) }

    private var leaderSize = 0
    private var trailerSize = 0
    private var payloadSize = 0
    private var payloadCount = 0
    private var transfer1Size = 0
    private var expectedPayload = 0L
    private var acquiring = false

    // ---- configuration ------------------------------------------------------

    fun readU32(name: String): Long = control.readU32(genicam.require(name).address)

    fun writeU32(name: String, value: Long) {
        val reg = genicam.require(name)
        if (!reg.writable) throw U3vException("register $name is ${reg.access}")
        control.writeU32(reg.address, value)
    }

    /** Fires a GenICam Command node, whose register is write-only and takes 1. */
    private fun command(valueRegister: String) =
        control.writeU32(genicam.require(valueRegister).address, 1)

    /**
     * Selects [entry] of enumeration [node].
     *
     * Returns null on success, or a sentence saying why it was not applied. Callers
     * decide whether that matters: most of these are conveniences, and refusing to open
     * the camera because one optional control is locked would be the wrong trade.
     *
     * The checks are the camera's own: pIsLocked and the entry's pIsAvailable. Writing
     * a locked node is what produced INVALID_PARAMETER on ExposureAuto while the frame
     * rate was pinned.
     */
    fun applyEnum(node: String, entry: String): String? {
        val e = genicam.enumeration(node) ?: return "$node has no directly addressable value register"
        e.locked?.let { if (genicam.read(control, it) == 1L) return "$node is locked by the camera" }
        val value = e.entries[entry] ?: return "$node has no entry '$entry'"
        e.available[entry]?.let { if (genicam.read(control, it) == 0L) return "$node.$entry is unavailable" }
        if (!e.value.writable) return "${e.value.name} is ${e.value.access}"
        // A refusal is a reason, not a failure: the camera has the final say about what
        // is writable in the current mode, and its flags do not always predict it.
        val before = runCatching { genicam.read(control, e.value) }.getOrNull()
        runCatching { genicam.write(control, e.value, value) }
            .onFailure { return refused(node, it.message, before, value) }
        return verify(node, e.value, value, before)
    }

    /**
     * Writes unity to a ratio node, in whatever units its register holds.
     *
     * White balance on this body is a ratio against green, so one is "leave the band
     * alone". GenICam exposes it as a Float over an integer register whose scaling this
     * driver does not model, so the only value that can be written without knowing that
     * scaling is the one the camera's own minimum-to-maximum range brackets: the register
     * is read, and unity is written as the midpoint only when the range makes that
     * unambiguous. Otherwise the node is reported unaddressable and left alone, which is a
     * better outcome than writing a number whose meaning is guessed.
     *
     * @return null when it took, a reason when it did not
     */
    fun applyRatioOne(node: String): String? {
        val value = runCatching { genicam.readFloat(control, node) }.getOrNull()
            ?: return "$node is not a plain float register"
        if (kotlin.math.abs(value - 1.0) <= RATIO_SLACK) return null
        return "$node reads %.4f, not 1".format(value)
    }

    /**
     * The white-balance ratios the camera is holding, by colour.
     *
     * ### Why this is read the hard way
     *
     * `BalanceRatio` is a Float over a Converter over a selector-indexed IntReg, and this
     * driver does not evaluate GenApi formulas -- so the node resolves to nothing and the
     * question "is white balance actually neutral" had no answer. The camera's own
     * description gives the three pieces needed to answer it directly:
     *
     * ```
     * BalanceRatioAddr  = 0x00108C00 + 0x20 * selector index
     * BalanceRatio_Val  = [addr + 0x04]   4 bytes
     * BalanceRatio_Posn = [addr + 0x14]   4 bytes, the fractional bit count
     * ratio = Val / 2^Posn
     * ```
     *
     * ### Why it matters
     *
     * A ratio is against green, so one means the band is untouched. Anything else scales a
     * whole colour, which per-band DoLP and AoLP survive -- every analyser angle in a
     * colour's cell is scaled alike and the ratio cancels -- but the noise model does not:
     * [StokesNoise] reads variance as a constant times the signal in raw counts, so a band
     * multiplied by k reports a shot gain k times off, and the debias, the noise floor and
     * the valid-pixel fraction are then wrong for that band and right for green.
     */
    fun balanceRatios(): Map<String, Double> {
        val out = LinkedHashMap<String, Double>()
        for ((index, name) in BALANCE_BANDS.withIndex()) {
            val base = BALANCE_BASE + BALANCE_STRIDE * index
            val raw = runCatching { readLe(base + BALANCE_VAL, 4) }.getOrNull() ?: continue
            val posn = runCatching { readLe(base + BALANCE_POSN, 4) }.getOrNull() ?: continue
            // A nonsense exponent means the layout is not what it was read as; better to
            // report nothing for that band than a number nobody should trust.
            if (posn !in 0..31) continue
            out[name] = raw.toDouble() / (1L shl posn.toInt()).toDouble()
        }
        return out
    }

    /**
     * Sets both colour ratios to one, so the camera stops scaling red and blue.
     *
     * Written straight to the registers because the GenICam path cannot reach them: the
     * node is a Float over a Converter over a selector-indexed IntReg, and this driver
     * evaluates no formulas. The camera's own description gives the layout, and the
     * converter's `FormulaTo` is `FROM * 2^FRACBITS`, so one is `1 shl Posn`.
     *
     * Verified by reading back rather than trusted: a write that returns success is not a
     * setting that changed, and this one decides whether the archive's red and blue bands
     * are raw counts or counts times a number.
     *
     * @return null when both bands read one afterwards, else what went wrong
     */
    /**
     * Sets the black level to zero, so a sample of zero light reads as zero.
     *
     * A pedestal is added before the ADC, so it lands in every sample and therefore in
     * S0. DoLP is a magnitude over S0, so an offset that nothing knows about does not
     * merely darken the picture -- it makes every degree of polarization read low, by a
     * factor that changes with the brightness of the scene. This Polar Camera writes
     * `BlackLevel = 0.0` for the same reason (`preset.py`), and until now this driver
     * only ever turned *clamping* off, which is a different setting entirely.
     *
     * Written to `BlackLevelAll_Val` rather than through the `BlackLevel` node, which is
     * a Float over a Converter over a selector-indexed register this driver cannot
     * address. `BlackLevelAll` is the one the selector's own description says is the only
     * entry a user may set, and it is a literal register -- so this addresses "All"
     * outright instead of setting a selector and hoping.
     *
     * Zero needs no conversion. The converter is `FROM * 2^DECPT / 100`, and zero percent
     * is zero counts whatever the fractional-bit count turns out to be -- which is the
     * one value that can be written without first reading `BlackLevel_Posn`.
     *
     * @return null when it reads back zero, else what went wrong
     */
    fun setBlackLevelToZero(): String? {
        val before = runCatching { readU32("BlackLevelAll_Val") }.getOrNull()
        runCatching { writeU32("BlackLevelAll_Val", 0L) }
            .onFailure { return "write refused: ${it.message}" }
        val after = runCatching { readU32("BlackLevelAll_Val") }.getOrNull()
            ?: return null
        if (after != 0L) return "reads $after after writing 0"
        if (before != null && before != 0L) Log.i(TAG, "BlackLevel was $before, now 0")
        return null
    }

    fun setBalanceRatiosToOne(): String? {
        val failures = mutableListOf<String>()
        for ((index, name) in BALANCE_BANDS.withIndex()) {
            val base = BALANCE_BASE + BALANCE_STRIDE * index
            val posn = runCatching { readLe(base + BALANCE_POSN, 4) }.getOrNull()
            if (posn == null || posn !in 0..31) {
                failures += "$name: fractional bits unreadable"
                continue
            }
            val unity = 1L shl posn.toInt()
            runCatching { control.writeU32(base + BALANCE_VAL, unity) }
                .onFailure {
                    failures += "$name: ${it.message}"
                    return@onFailure
                }
            val after = runCatching { readLe(base + BALANCE_VAL, 4) }.getOrNull()
            if (after != unity) failures += "$name: reads $after after writing $unity"
        }
        return failures.takeIf { it.isNotEmpty() }?.joinToString("; ")
    }

    private fun readLe(address: Long, length: Int): Long {
        val raw = control.readMemory(address, length)
        var value = 0L
        for (i in raw.indices.reversed()) value = (value shl 8) or (raw[i].toLong() and 0xFF)
        return value
    }

    /** How this camera's own description defines [node], for diagnosing an odd one. */
    /**
     * What the camera currently holds for each of [names], as JSON-ready text.
     *
     * A record of the state the frame was taken under, so a capture can be read back
     * years later without the reader having to trust that the preset did what this
     * version of the code believed it did. This Polar Camera writes the same thing
     * (`read_flir_node_values` over `FLIR_METADATA_NODES`) and, like it, a node that
     * cannot be read is left out rather than written as null: absent means "not asked or
     * not answerable", which is different from a value.
     *
     * Enumerations come back as their entry name -- "Off" rather than 0 -- because that
     * is what the camera's own documentation calls them and a bare integer would need
     * this table to interpret.
     */
    fun readNamedNodes(names: List<String>): Map<String, String> {
        val out = LinkedHashMap<String, String>()
        for (requested in names) {
            // Most of the interesting nodes are a facade over a register named for them
            // with "_Val" on the end, and the facade is often a Converter or a
            // selector-indexed address this driver cannot evaluate. Asking for the
            // register directly is what makes ExposureTime and Gain appear at all.
            val name = if (genicam.reg(requested) != null ||
                genicam.enumeration(requested) != null ||
                genicam.boolean(requested) != null
            ) {
                requested
            } else {
                "${requested}_Val".takeIf { genicam.reg(it) != null } ?: requested
            }
            val enumeration = genicam.enumeration(name)
            if (enumeration != null) {
                val value = runCatching { genicam.read(control, enumeration.value) }.getOrNull()
                if (value != null) {
                    val entry = enumeration.entries.entries.firstOrNull { it.value == value }?.key
                    out[requested] = entry?.let { "\"$it\"" } ?: value.toString()
                    continue
                }
            }
            val boolean = genicam.boolean(name)
            if (boolean != null) {
                val value = runCatching { genicam.read(control, boolean.value) }.getOrNull()
                if (value != null) {
                    out[requested] = (value == boolean.onValue).toString()
                    continue
                }
            }
            val float = runCatching { genicam.readFloat(control, name) }.getOrNull()
            if (float != null && float.isFinite()) {
                out[requested] = "%.4f".format(float)
                continue
            }
            val integer = runCatching { genicam.read(control, name) }.getOrNull()
            if (integer != null) out[requested] = integer.toString()
        }
        return out
    }

    fun describeNode(node: String): String? = genicam.describe(node)

    /** The node [node]'s value is delegated to, for following a facade to its register. */
    fun pValueOf(node: String): String? = genicam.pValueOf(node)

    /** Same contract as [applyEnum], for a Boolean node. */
    fun applyBoolean(node: String, on: Boolean): String? {
        val b = genicam.boolean(node) ?: return "$node has no directly addressable value register"
        b.locked?.let { if (genicam.read(control, it) == 1L) return "$node is locked by the camera" }
        if (!b.value.writable) return "${b.value.name} is ${b.value.access}"
        val want = if (on) b.onValue else b.offValue
        val before = runCatching { genicam.read(control, b.value) }.getOrNull()
        runCatching { genicam.write(control, b.value, want) }
            .onFailure { return refused(node, it.message, before, want) }
        return verify(node, b.value, want, before)
    }

    /**
     * A refusal, with what the node actually reads.
     *
     * The value is the whole point. This camera denies writes to its ISP nodes, and
     * "IspEnable refused, and it reads 1" means every capture so far went through the
     * demosaic; "refused, and it reads 0" means the camera is refusing because the
     * setting is already forced off and there is nothing to fix. The bare refusal that
     * used to be logged could not tell those apart, and they are opposite conclusions.
     */
    private fun refused(node: String, message: String?, before: Long?, want: Long): String {
        val state = when {
            before == null -> "current value unreadable"
            before == want -> "already $before, so nothing needed changing"
            else -> "and it reads $before, not the wanted $want"
        }
        return "$node write refused: $message ($state)"
    }

    /**
     * Reads a register back after writing it, and says so when it did not take.
     *
     * A write that returns success is not the same as a setting that changed: this camera
     * accepts writes to nodes it is currently ignoring, and the only way to tell is to
     * look. Every such setting here exists to keep something from touching the pixels
     * before they are measured, so a silent no-op is the failure that matters -- it would
     * leave the captures processed while the log said they were raw.
     *
     * A value that was already right is logged as changed only when it actually changed,
     * because "IspEnable was 1, now 0" is the line that says how much of the archive was
     * taken with the sensor's own pipeline in the way.
     */
    private fun verify(node: String, reg: Genicam.Reg, want: Long, before: Long?): String? {
        val after = runCatching { genicam.read(control, reg) }.getOrNull()
            // Not every register reads back on this camera, and a write that succeeded is
            // still better evidence than nothing. Unverifiable is not refused.
            ?: return null
        if (after != want) return "$node still reads $after after writing $want"
        if (before != null && before != want) Log.i(TAG, "$node was $before, now $after")
        return null
    }

    var pixelFormat: Long
        get() = genicam.enumeration("PixelFormat")?.let { genicam.read(control, it.value) } ?: 0L
        set(value) {
            val e = genicam.enumeration("PixelFormat") ?: throw U3vException("PixelFormat is not addressable")
            genicam.write(control, e.value, value)
        }

    var exposureTimeUs: Long
        get() = readU32("ExposureTime_Val")
        set(value) {
            // Manual exposure only takes effect with the auto algorithm switched off.
            applyEnum("ExposureAuto", "Off")?.let { Log.w(TAG, "ExposureAuto=Off skipped: $it") }
            writeU32("ExposureTime_Val", value.coerceIn(exposureRangeUs))
        }

    val exposureRangeUs: LongRange
        get() = readU32("ExposureTime_Min")..readU32("ExposureTime_Max")

    /**
     * Writes the exposure and nothing else, for a caller that already knows the range.
     *
     * The property setter above costs four round trips: it re-asserts ExposureAuto=Off
     * and re-reads both ends of the range to clamp against them. That is right for a
     * value a person picked, which arrives once and may arrive out of a stale UI. It is
     * wrong for the host meter, which runs against every frame on the same control
     * channel the stream shares, has already been told the range once, and has already
     * switched the auto algorithm off when it started.
     *
     * The caller owns the clamping. Out of range, the camera refuses the write.
     */
    fun writeExposureUs(us: Long) = writeU32("ExposureTime_Val", us)

    /**
     * Caps how fast the camera produces frames, or removes the cap when [hz] is null.
     *
     * Without a cap the sensor outruns this host: the camera holds a 240 MB image
     * buffer, so surplus frames queue there and the preview drifts further behind real
     * time the longer it runs. Matching production to consumption keeps the delay at one
     * frame. Spinnaker gets the same effect host-side with StreamBufferHandlingMode
     * NewestOnly, which is a transport-layer setting this driver has no equivalent of --
     * here the acquisition engine is us.
     *
     * The limiter also makes payload settings read-only on this model, so it has to be
     * off while PixelFormat or the ROI is being changed.
     */
    fun setFrameRateLimit(hz: Double?): String? {
        if (hz == null) return applyBoolean("AcquisitionFrameRateEnable", false)
        applyBoolean("AcquisitionFrameRateEnable", true)?.let { return it }

        // AcquisitionFrameRate is a converter over FrameTime: rate = timebase / time.
        val timebase = genicam.read(control, "FrameRateNumerator_Val")
            ?: return "FrameRateNumerator_Val is not readable"
        val min = genicam.read(control, "FrameTime_Min") ?: 1L
        val max = genicam.read(control, "FrameTime_Max") ?: Long.MAX_VALUE
        val frameTime = (timebase / hz).toLong().coerceIn(min, max)

        return runCatching { writeU32("FrameTime_Val", frameTime); null }
            .getOrElse { "frame rate write refused: ${it.message}" }
    }

    // ---- clock ---------------------------------------------------------------

    /**
     * The camera's clock measured against this host's, so frames from different
     * cameras can be put on one timeline.
     *
     * A frame's leader carries the device's own timestamp, which says when the sensor
     * exposed it -- far more useful than when the host finished reading it. That number
     * is on the camera's clock, so it is only comparable to anything else once the two
     * clocks are related.
     *
     * The device clock is latched between two host readings, so the offset is known to
     * within the round trip rather than exactly; [uncertaintyNs] carries that.
     */
    class ClockOffset(
        val hostNs: Long,
        /** The device clock at [hostNs], in the device's own ticks. */
        val deviceTicks: Long,
        /** Nanoseconds per tick, from ABRM.TIMESTAMP_INCREMENT. */
        val incrementNs: Long,
        val uncertaintyNs: Long,
    ) {
        /**
         * Puts a frame's device timestamp on the host clock.
         *
         * Both timestamps are counted in the device's own ticks -- the leader uses the
         * same facility as the bootstrap register -- so the difference is taken first
         * and converted once. Converting each side separately is how this produced a
         * timestamp from before the epoch.
         */
        fun toHostNs(frameTicks: Long): Long = hostNs + (frameTicks - deviceTicks) * incrementNs
    }

    /** Null when the camera does not implement the optional timestamp registers. */
    fun latchClock(): ClockOffset? = runCatching {
        val increment = control.readU32(Abrm.TIMESTAMP_INCREMENT)
        if (increment <= 0) return null

        val before = System.nanoTime()
        control.writeU32(Abrm.TIMESTAMP_LATCH, 1)
        val ticks = control.readU64(Abrm.TIMESTAMP)
        val after = System.nanoTime()

        Log.i(TAG, "device clock: $ticks ticks, $increment ns per tick")
        ClockOffset(
            hostNs = (before + after) / 2,
            deviceTicks = ticks,
            incrementNs = increment,
            uncertaintyNs = (after - before) / 2,
        )
    }.getOrNull()

    // ---- gain ---------------------------------------------------------------

    /**
     * Analog gain in dB, or null when the registers did not verify.
     *
     * Gain sits behind a selector-indexed address like TriggerMode does, so
     * [Genicam.reg] refuses it. Unlike TriggerMode the selector is provably fixed:
     * GainSelector offers only "All" (0), and GainSelectorValueToIndex's formula is
     * `( SWITCH_TARGET = 0 ) ? 0 : 0`, whose branches are both zero. GainAddr is
     * therefore `0x00104800 + 0x20 * 0`, a constant.
     *
     * That derivation is still an assumption about this firmware, so [gainRangeDb]
     * checks the values it reads back before anything is written. A different layout
     * would give a nonsensical range and gain is then reported unavailable rather than
     * written to an address nobody has verified.
     */
    private val gainValue = GAIN_BASE + 0x04
    private val gainMin = GAIN_BASE + 0x08
    private val gainMax = GAIN_BASE + 0x0C

    /** Null when the registers do not look like a gain range. */
    val gainRangeDb: ClosedFloatingPointRange<Double>?
        get() = runCatching {
            val lo = toDb(control.readU32(gainMin).toInt())
            val hi = toDb(control.readU32(gainMax).toInt())
            val now = toDb(control.readU32(gainValue).toInt())
            // A real Blackfly S range is a few tens of dB starting at or below zero,
            // and the current value has to sit inside it.
            if (hi <= lo || hi - lo > GAIN_SPAN_LIMIT_DB || now < lo - 1 || now > hi + 1) {
                Log.w(TAG, "gain registers look wrong: $lo..$hi dB, now $now")
                null
            } else {
                lo..hi
            }
        }.getOrNull()

    var gainDb: Double
        get() = toDb(control.readU32(gainValue).toInt())
        set(value) {
            val range = gainRangeDb ?: throw U3vException("gain is not addressable on this camera")
            control.writeU32(gainValue, fromDb(value.coerceIn(range)).toLong())
        }

    /** The register counts 2^14 steps per 6.02 dB, per the camera's own converter. */
    private fun toDb(raw: Int): Double = raw * GAIN_DB_PER_STEP / GAIN_STEPS
    private fun fromDb(db: Double): Int = (db * GAIN_STEPS / GAIN_DB_PER_STEP).toInt()

    /** Returns null on success, or why the change was not applied. */
    fun setExposureAuto(on: Boolean): String? =
        applyEnum("ExposureAuto", if (on) "Continuous" else "Off")

    val width: Int get() = readU32("Width_Val").toInt()
    val height: Int get() = readU32("Height_Val").toInt()
    val widthMax: Int get() = readU32("WidthMax_Val").toInt()
    val heightMax: Int get() = readU32("HeightMax_Val").toInt()

    /**
     * The camera's own frame reduction, when it has one.
     *
     * Read defensively: a model that does not offer binning has no such register, and
     * on this sensor whether it is usable at all is a question about which pixels get
     * combined, not about whether the node exists. See FlirProbe.binningReport.
     */
    val binning: Int? get() = runCatching { readU32("BinningHorizontal_Val").toInt() }.getOrNull()

    val binningMax: Int? get() = runCatching { readU32("BinningHorizontal_Max").toInt() }.getOrNull()

    /**
     * Sets horizontal and vertical binning together. Acquisition must be stopped first.
     *
     * Returns null on success, or why it did not apply. Failure is expected and not an
     * error: several of these registers are read-only while the frame-rate limiter is
     * on, and some models lock them outright.
     */
    fun setBinning(n: Int): String? = try {
        writeU32("BinningHorizontal_Val", n.toLong())
        writeU32("BinningVertical_Val", n.toLong())
        val got = binning
        if (got != n) "asked for $n, camera reports $got" else null
    } catch (e: Exception) {
        e.message ?: e::class.simpleName
    }

    /**
     * Sets a centred region of interest. Acquisition must be stopped first.
     *
     * Sizes and offsets are snapped to a multiple of both the camera's own increment and
     * [MOSAIC], so the 4x4 colour-polarizer super-pixel keeps its phase: a ROI off by one
     * pixel silently relabels every analyser angle, which would corrupt the Stokes maths
     * without looking wrong.
     *
     * This crops rather than scales -- the sensor has no way to shrink a frame while
     * preserving the mosaic, since binning averages across analyser angles and the only
     * decimation mode on offer discards the pixels that carry them.
     */
    fun setRoi(w: Int, h: Int) {
        val stepX = step(genicam.read(control, "Width_Inc"))
        val stepY = step(genicam.read(control, "Height_Inc"))
        val rw = (w / stepX) * stepX
        val rh = (h / stepY) * stepY

        // Shrink before moving the offsets outward: the camera rejects a width that
        // would not fit at the current offset, and vice versa.
        writeU32("OffsetX_Val", 0)
        writeU32("OffsetY_Val", 0)
        writeU32("Width_Val", rw.toLong())
        writeU32("Height_Val", rh.toLong())
        writeU32("OffsetX_Val", (((widthMax - rw) / 2) / stepX * stepX).toLong())
        writeU32("OffsetY_Val", (((heightMax - rh) / 2) / stepY * stepY).toLong())
    }

    /** Least common multiple of the camera's increment and the mosaic period. */
    private fun step(increment: Long?): Int {
        val inc = (increment ?: 1L).toInt().coerceAtLeast(1)
        var a = inc
        var b = MOSAIC
        while (b != 0) {
            val t = b
            b = a % b
            a = t
        }
        return inc / a * MOSAIC
    }

    private var pipe: StreamPipe? = null

    /**
     * Empties the stream endpoint.
     *
     * Stopping acquisition can leave part of a frame queued, and the next read would
     * then start mid-payload and fail the leader magic check on every frame after it.
     *
     * Drained through the pipe whenever there is one. A synchronous read cannot do this
     * once the transfers are 256 KB: bulkTransfer clamps its length to 16 KB, so asking
     * for a leftover transfer that size gets a fraction of it and overruns the endpoint,
     * and every read then costs its full timeout while achieving nothing. That mistake
     * cost half a second on every shutter press before it was spotted.
     */
    fun drainStream() {
        val chunk = maxOf(payloadSize, transfer1Size, leaderSize, trailerSize, TRANSFER_SIZE)
        val scratch = ByteArray(chunk)
        val open = pipe?.takeIf { !it.failed }
        var guard = 0
        while (guard++ < DRAIN_READS_MAX) {
            val got = if (open != null) {
                if (!open.submit(chunk)) return
                open.collect(scratch, 0, DRAIN_TIMEOUT_MS.toLong(), expectEmpty = true)
            } else {
                runCatching { control.readStream(scratch, 0, minOf(chunk, TRANSFER_SIZE), DRAIN_TIMEOUT_MS) }
                    .getOrDefault(-1)
            }
            if (got <= 0) break
        }
        if (guard > DRAIN_READS_MAX) Log.w(TAG, "stream still had data after $DRAIN_READS_MAX drain reads")
        // The last read of a drain always comes back empty -- that is how the drain knows
        // it is done -- and an empty read through the pipe is a timeout, which retires it.
        // A retired pipe would send the next acquisition down the synchronous path, where
        // a 256 KB transfer cannot be read at all. So it is replaced here rather than
        // being allowed to limp.
        if (pipe?.failed == true) {
            pipe?.close()
            pipe = control.openStreamPipe(PIPE_DEPTH, LARGE_TRANSFER_SIZE)
        }
    }

    // ---- acquisition --------------------------------------------------------

    /**
     * Makes the camera produce exactly [frames] and then stop, or says why it will not.
     *
     * Acquisition must be stopped first: AcquisitionMode is read-only while streaming.
     *
     * ### Why bother, when a free-running camera also gives you a frame
     *
     * Because it gives you rather more than one. A capture stops the preview, drains,
     * widens the ROI and starts acquisition -- and from that instant a free-running
     * sensor makes frames as fast as the bus allows, for the whole time the host spends
     * reading the ones it wanted and then stopping the stream again. Those surplus
     * frames are exposed, transferred as far as the camera's own buffer, and thrown away
     * by [drainStream]. Asking for a finite count removes them: the sensor makes what
     * was asked for and stops on its own, so there is nothing to drain and nothing
     * competing for the bus with the frames that are being kept.
     *
     * It also makes the burst honest. A burst of four is four frames from one
     * acquisition either way, but pinned to four they are consecutive by construction
     * rather than by the host keeping up.
     *
     * ### Why the readback has to be exact
     *
     * `AcquisitionFrameCount` is clamped by the camera, not refused, so a body whose
     * range stops at three answers a request for four by quietly making three. The
     * caller would then wait for a fourth frame that is never coming, and the whole
     * capture would end in a frame timeout. An inexact readback is reported as a refusal
     * so the caller can fall back to free-running, where waiting is safe.
     */
    fun setFrameCount(frames: Int): String? {
        require(frames >= 1) { "frame count must be at least 1" }
        // SingleFrame deliberately does not touch AcquisitionFrameCount: it is commonly
        // unavailable in that mode, and the count is implied.
        if (frames == 1) return applyEnum("AcquisitionMode", "SingleFrame")

        applyEnum("AcquisitionMode", "MultiFrame")?.let { return it }
        val reg = genicam.reg("AcquisitionFrameCount_Val")
            ?: return "AcquisitionFrameCount has no directly addressable value register"
        if (!reg.writable) return "AcquisitionFrameCount is ${reg.access}"
        runCatching { genicam.write(control, reg, frames.toLong()) }
            .onFailure { return "AcquisitionFrameCount write refused: ${it.message}" }
        // Unreadable is not refused: the write succeeded, and that is better evidence
        // than nothing. Read back and disagreeing is refused, because that is the clamp.
        val got = genicam.read(control, reg) ?: return null
        if (got != frames.toLong()) return "AcquisitionFrameCount clamped to $got from $frames"
        return null
    }

    /** Back to free-running. Acquisition must be stopped first. */
    fun setContinuousAcquisition(): String? = applyEnum("AcquisitionMode", "Continuous")

    /**
     * Negotiates the stream layout with the camera and starts acquisition.
     *
     * The sizing arithmetic follows Aravis' arvuvstream.c: the device states the payload
     * it will send and the alignment it needs, and the host answers with how it wants
     * that split into bulk transfers.
     */
    fun startAcquisition() {
        if (acquiring) return

        val info = control.readU32(sirm + Sirm.INFO)
        expectedPayload = control.readU64(sirm + Sirm.REQ_PAYLOAD_SIZE)
        val reqLeader = control.readU32(sirm + Sirm.REQ_LEADER_SIZE).toInt()
        val reqTrailer = control.readU32(sirm + Sirm.REQ_TRAILER_SIZE).toInt()

        val alignment = 1 shl ((info and Uvsp.SIRM_INFO_ALIGNMENT_MASK) ushr Uvsp.SIRM_INFO_ALIGNMENT_SHIFT).toInt()
        if (expectedPayload < 1) throw U3vException("camera reported payload size $expectedPayload")
        if (alignment < 1) throw U3vException("camera reported alignment $alignment")

        // The transfer size is negotiated with the camera, so the host has to decide
        // first how it intends to read. bulkTransfer silently clamps to the kernel's
        // 16 KB usbfs buffer; a queued UsbRequest does not, so when the pipe opens the
        // frame can be asked for in far fewer, far larger pieces -- 29 transfers instead
        // of 459. There is no synchronous fallback at that size (a read shorter than the
        // transfer the camera sends overruns the endpoint), so the pipe is opened first
        // and the size follows from whether it exists.
        // A fresh one per acquisition: the transfer sizes are renegotiated below and the
        // old pipe's buffers were sized for the last ROI.
        pipe?.close()
        pipe = control.openStreamPipe(PIPE_DEPTH, LARGE_TRANSFER_SIZE)
        val wanted = if (pipe != null) LARGE_TRANSFER_SIZE else TRANSFER_SIZE
        val maxTransfer = (wanted / alignment) * alignment
        if (maxTransfer < 1) throw U3vException("alignment $alignment exceeds the transfer size")

        leaderSize = if (reqLeader < 1) maxTransfer else align(reqLeader, alignment)
        trailerSize = if (reqTrailer < 1) maxTransfer else align(reqTrailer, alignment)
        payloadSize = minOf(expectedPayload, maxTransfer.toLong()).toInt()
        payloadCount = (expectedPayload / payloadSize).toInt()
        transfer1Size = align((expectedPayload % payloadSize).toInt(), alignment)

        control.writeU32(sirm + Sirm.MAX_LEADER_SIZE, leaderSize.toLong())
        control.writeU32(sirm + Sirm.MAX_TRAILER_SIZE, trailerSize.toLong())
        control.writeU32(sirm + Sirm.PAYLOAD_SIZE, payloadSize.toLong())
        control.writeU32(sirm + Sirm.PAYLOAD_COUNT, payloadCount.toLong())
        control.writeU32(sirm + Sirm.TRANSFER1_SIZE, transfer1Size.toLong())
        control.writeU32(sirm + Sirm.TRANSFER2_SIZE, 0)

        Log.i(
            TAG,
            "stream: payload=$expectedPayload leader=$leaderSize trailer=$trailerSize " +
                "chunk=$payloadSize x$payloadCount + $transfer1Size (align $alignment), " +
                if (pipe != null) "queued $PIPE_DEPTH deep" else "synchronous",
        )

        control.writeU32(sirm + Sirm.CONTROL, Uvsp.SIRM_CONTROL_STREAM_ENABLE.toLong())
        command("AcquisitionStart_Val")
        acquiring = true
    }

    /**
     * Puts the camera back into a configurable state, whatever it was left in.
     *
     * A session that died without stopping acquisition leaves the camera streaming, and
     * that locks PixelFormat -- ACCESS_DENIED on the first configuration write of the
     * next run. Doing this unconditionally at open time makes a previous crash
     * recoverable without unplugging anything.
     *
     * Note that an enabled frame-rate limiter locks PixelFormat on this model too, so
     * open() clears that before touching the payload format rather than relying on this.
     */
    fun resetStream() {
        runCatching { command("AcquisitionStop_Val") }
            .onFailure { Log.i(TAG, "no acquisition to stop: ${it.message}") }
        runCatching { control.writeU32(sirm + Sirm.CONTROL, 0) }
            .onFailure { Log.i(TAG, "stream was not enabled: ${it.message}") }
        acquiring = false
    }

    /** Reads a lock flag by node name, for explaining a refusal. Null when unreadable. */
    fun lockState(name: String): Long? = genicam.read(control, name)

    fun stopAcquisition() {
        if (!acquiring) return
        acquiring = false
        // Anything still queued is abandoned, but the pipe itself stays: drainStream is
        // called right after this and it is the only thing that can read a transfer of
        // the negotiated size. It is closed when the next acquisition renegotiates, or
        // when the camera does.
        pipe?.cancelAll()
        runCatching { command("AcquisitionStop_Val") }
            .onFailure { Log.w(TAG, "AcquisitionStop failed", it) }
        runCatching { control.writeU32(sirm + Sirm.CONTROL, 0) }
            .onFailure { Log.w(TAG, "stream disable failed", it) }
    }

    /**
     * Reads one frame: a leader transfer, the payload transfers, then a trailer.
     *
     * Sizes come from the SIRM negotiation and must be requested exactly, because each
     * is its own bulk transfer on the wire and a short read would desynchronise the
     * stream for every frame after it.
     */
    fun grabFrame(timeoutMs: Int = FRAME_TIMEOUT_MS): Frame {
        if (!acquiring) throw U3vException("acquisition is not running")

        val leader = ByteArray(leaderSize)
        val n = control.readStream(leader, 0, leaderSize, timeoutMs)
        val leaderHostNs = System.nanoTime()
        if (n < Uvsp.HEADER_SIZE) {
            // A camera left in triggered mode sits here forever waiting for a trigger
            // this driver cannot clear, so name that possibility rather than just
            // reporting a timeout.
            throw U3vException(
                "no frame (leader read returned $n). If this persists, the camera may be " +
                    "in triggered mode; set TriggerMode to Off in SpinView once."
            )
        }

        val lb = ByteBuffer.wrap(leader).order(ByteOrder.LITTLE_ENDIAN)
        if (lb.getInt(0) != Uvsp.LEADER_MAGIC) {
            throw U3vException("bad leader magic 0x%08x".format(lb.getInt(0)))
        }
        val frameId = lb.getLong(8)
        val timestamp = lb.getLong(Uvsp.LEADER_TIMESTAMP)
        val format = lb.getInt(Uvsp.LEADER_PIXEL_FORMAT).toLong() and 0xFFFFFFFFL
        val w = lb.getInt(Uvsp.LEADER_WIDTH)
        val h = lb.getInt(Uvsp.LEADER_HEIGHT)
        val xPadding = lb.getShort(Uvsp.LEADER_X_PADDING).toInt() and 0xFFFF

        // TRANSFER1_SIZE is rounded up to the device's alignment, so the transfers can
        // add up to slightly more than the payload. The buffer has to cover that or the
        // last read would ask for less than the camera sends and overflow the endpoint.
        val payload = ByteArray(payloadCount * payloadSize + transfer1Size)
        val beforePayload = System.nanoTime()
        earlyTrailer = null
        val offset = readPayload(payload, timeoutMs)
        val payloadNs = System.nanoTime() - beforePayload

        // The trailer normally follows the payload transfers; with chunk data the payload can end
        // early and the trailer then arrives in place of a payload transfer (see readPayload).
        val trailer: ByteArray
        val trailerRead: Int
        val early = earlyTrailer
        if (early != null) {
            trailer = early
            trailerRead = early.size
        } else {
            trailer = ByteArray(trailerSize)
            trailerRead = control.readStream(trailer, 0, trailerSize, timeoutMs)
        }
        // Trailer: magic(4) reserved(2) size(2) block id(8) status(2) reserved(2) valid payload size(8) ...
        val validPayload = if (trailerRead >= Uvsp.TRAILER_PAYLOAD_SIZE + 8) {
            val tb = ByteBuffer.wrap(trailer).order(ByteOrder.LITTLE_ENDIAN)
            if (tb.getInt(0) == Uvsp.TRAILER_MAGIC) tb.getLong(Uvsp.TRAILER_PAYLOAD_SIZE) else -1L
        } else {
            -1L
        }

        // REQ_PAYLOAD_SIZE is the most the camera may send. With chunk data on, a BayerRG16 frame is
        // 936 B shorter than that (10 027 096 of 10 028 032 B on the BFS-U3-51S5PC), so a short payload
        // is complete when the trailer says that is all there was.
        if (offset < expectedPayload && validPayload != offset.toLong()) {
            throw U3vException("payload short: $offset of $expectedPayload (trailer says $validPayload)")
        }
        if (xPadding != 0) Log.w(TAG, "frame carries $xPadding bytes of row padding")

        // Hand back exactly the image, trimming the alignment slack read above.
        val image = if (payload.size > expectedPayload) payload.copyOf(expectedPayload.toInt()) else payload
        return Frame(
            image, w, h, format, timestamp, leaderHostNs, frameId,
            payloadNs = payloadNs, payloadBytes = offset.toLong(),
            validPayloadBytes = if (validPayload in 1..image.size.toLong()) validPayload else -1L,
        )
    }

    /**
     * Reads a whole frame's payload, several transfers in flight at once.
     *
     * The sizes are fixed by the SIRM negotiation and known before the frame starts, so
     * the whole sequence can be queued ahead rather than asked for one at a time. That is
     * the difference between using half of a USB 2.0 bus and using it: see [StreamPipe].
     *
     * Falls back to the synchronous reader for the rest of the frame if the pipe cannot
     * carry it. That path is slower but it is the one that has always worked, and a
     * camera that streams slowly is worth much more than one that does not stream.
     */
    private fun readPayload(payload: ByteArray, timeoutMs: Int): Int {
        val pipe = this.pipe
        if (pipe == null || pipe.failed) return readPayloadSynchronously(payload, timeoutMs)

        // Every transfer of this frame, in the order the camera will send them.
        val sizes = IntArray(payloadCount + if (transfer1Size > 0) 1 else 0) { i ->
            if (i < payloadCount) payloadSize else transfer1Size
        }
        var queued = 0
        var offset = 0
        var collected = 0
        while (queued < sizes.size && pipe.room > 0) {
            if (!pipe.submit(sizes[queued])) break
            queued++
        }
        if (queued == 0) return readPayloadSynchronously(payload, timeoutMs)

        while (collected < sizes.size) {
            val got = pipe.collect(payload, offset, timeoutMs.toLong())
            if (got < 0) {
                // Mid-frame: this frame is lost either way, and the stream is now at an
                // unknown offset, so the leftovers go and the caller resynchronises.
                pipe.cancelAll()
                throw U3vException(
                    "queued stream read failed at transfer $collected of ${sizes.size}"
                )
            }
            if (isTrailer(payload, offset, got)) {
                // The payload ended early (chunk data, see grabFrame) and this transfer is the
                // trailer. Anything still queued would belong to the next frame.
                earlyTrailer = payload.copyOfRange(offset, offset + got)
                payload.fill(0, offset, offset + got)
                pipe.cancelAll()
                return offset
            }
            offset += got
            collected++
            if (queued < sizes.size) {
                if (pipe.submit(sizes[queued])) queued++ else break
            }
        }
        // Anything the pipe could not take is read the old way, so a short queue is a
        // slowdown rather than a broken frame.
        while (collected < sizes.size) {
            val got = readInto(payload, offset, sizes[collected], timeoutMs)
            if (isTrailer(payload, offset, got)) {
                earlyTrailer = payload.copyOfRange(offset, offset + got)
                payload.fill(0, offset, offset + got)
                return offset
            }
            offset += got
            collected++
        }
        return offset
    }

    private fun readPayloadSynchronously(payload: ByteArray, timeoutMs: Int): Int {
        var offset = 0
        val sizes = IntArray(payloadCount + if (transfer1Size > 0) 1 else 0) { i ->
            if (i < payloadCount) payloadSize else transfer1Size
        }
        for (size in sizes) {
            val got = readInto(payload, offset, size, timeoutMs)
            if (isTrailer(payload, offset, got)) {
                earlyTrailer = payload.copyOfRange(offset, offset + got)
                payload.fill(0, offset, offset + got)
                return offset
            }
            offset += got
        }
        return offset
    }

    /** A trailer that arrived where a payload transfer was expected: short, and starting with "U3VT". */
    private var earlyTrailer: ByteArray? = null

    private fun isTrailer(buf: ByteArray, offset: Int, got: Int): Boolean {
        if (got < Uvsp.HEADER_SIZE || got > maxOf(trailerSize, Uvsp.HEADER_SIZE + 64)) return false
        return ByteBuffer.wrap(buf, offset, 4).order(ByteOrder.LITTLE_ENDIAN).int == Uvsp.TRAILER_MAGIC
    }

    /**
     * Reads one payload transfer straight into the frame buffer.
     *
     * A read shorter than the negotiated chunk is normal on the last transfer of a
     * frame, so it advances the offset by what arrived rather than failing.
     */
    private fun readInto(dst: ByteArray, offset: Int, size: Int, timeoutMs: Int): Int {
        val want = minOf(size, dst.size - offset)
        if (want <= 0) return 0
        val got = control.readStream(dst, offset, want, timeoutMs)
        if (got < 0) {
            throw U3vException(
                "stream read of $want B at offset $offset returned $got after ${timeoutMs} ms"
            )
        }
        return got
    }

    override fun close() {
        stopAcquisition()
        pipe?.close()
        pipe = null
        control.close()
    }

    private fun align(value: Int, alignment: Int) = (value + alignment - 1) / alignment * alignment

    companion object {
        private const val TAG = "FlirCamera"

        /**
         * How far a white-balance ratio may sit from one and still count as neutral.
         *
         * The camera quantises these, so an exactly-neutral body can report 0.9999. A
         * thousandth is far below anything that would matter to a noise model and far
         * above the quantisation.
         */
        private const val RATIO_SLACK = 0.001

        /**
         * Where the selector-indexed white-balance ratios live, from this camera's own
         * description: `0x00108C00 + 0x20 * index`, with the value at +4 and the
         * fractional bit count at +0x14.
         */
        private const val BALANCE_BASE = 0x00108C00L
        private const val BALANCE_STRIDE = 0x20L
        private const val BALANCE_VAL = 0x04L
        private const val BALANCE_POSN = 0x14L

        /** Selector order, which the description maps to indices 0 and 1. */
        private val BALANCE_BANDS = listOf("Red", "Blue")

        /** Colour-polarizer super-pixel: 2x2 Bayer, each cell a 2x2 polarizer unit. */
        const val MOSAIC = 4

        /**
         * Bulk transfer size, and it cannot be raised.
         *
         * UsbDeviceConnection.bulkTransfer silently clamps its length argument to
         * MAX_USBFS_BUFFER_SIZE. Negotiating a larger SIRM payload size therefore tells
         * the camera to send transfers the host can only partially read, and the
         * remainder overflows the endpoint: the read returns -1 at an offset that moves
         * around between attempts, which reads like a flaky stream rather than a size
         * limit. A full frame is 306 transfers at this size.
         */
        private const val TRANSFER_SIZE = 16384

        /**
         * Reads kept in flight on the stream endpoint.
         *
         * Four is enough to cover the gap between one transfer landing and the next being
         * submitted, which is the whole of what is being recovered; past that the bus is
         * already the limit and every extra slot is another 16 KB of pinned memory for
         * nothing.
         */
        private const val PIPE_DEPTH = 4

        /**
         * Transfer size used when reads are queued rather than synchronous.
         *
         * 256 KB, which turns a 7.5 MB frame into 29 transfers instead of 459. Only
         * reachable through UsbRequest: bulkTransfer clamps to 16 KB, and asking for less
         * than the camera sends in one transfer overruns the endpoint, so this size and
         * the synchronous reader are mutually exclusive.
         *
         * Measured on this rig, reading the same frame: 16 KB gave 2.8 frames a second,
         * 256 KB gave 4.8, and 1 MB also gave 4.8. The bus was never the limit at 16 KB
         * -- the per-transfer overhead of 459 round trips was -- and the knee is here, so
         * the larger size would only pin four times the memory for nothing.
         */
        private const val LARGE_TRANSFER_SIZE = 262144

        private const val FRAME_TIMEOUT_MS = 5000

        /** GainAddr, from the constant IntSwissKnife described on [gainRangeDb]. */
        private const val GAIN_BASE = 0x00104800L
        private const val GAIN_DB_PER_STEP = 6.020599913
        private const val GAIN_STEPS = 16384.0

        /** No Blackfly S offers anything like this much; a wider span means a bad read. */
        private const val GAIN_SPAN_LIMIT_DB = 120.0
        private const val DRAIN_TIMEOUT_MS = 50
        private const val DRAIN_READS_MAX = 512
    }
}

/** Streaming Interface Register Map offsets, from Aravis src/arvuvcpprivate.h. */
internal object Sirm {
    const val INFO = 0x0000L
    const val CONTROL = 0x0004L
    const val REQ_PAYLOAD_SIZE = 0x0008L
    const val REQ_LEADER_SIZE = 0x0010L
    const val REQ_TRAILER_SIZE = 0x0014L
    const val MAX_LEADER_SIZE = 0x0018L
    const val PAYLOAD_SIZE = 0x001CL
    const val PAYLOAD_COUNT = 0x0020L
    const val TRANSFER1_SIZE = 0x0024L
    const val TRANSFER2_SIZE = 0x0028L
    const val MAX_TRAILER_SIZE = 0x002CL
}
