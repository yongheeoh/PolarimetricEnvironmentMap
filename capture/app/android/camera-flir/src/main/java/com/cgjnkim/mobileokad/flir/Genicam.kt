package com.cgjnkim.mobileokad.flir

/**
 * Just enough of the GenICam XML to address registers by name.
 *
 * This is not a GenApi implementation. It resolves only nodes whose address is a
 * literal in the XML, which on this camera covers everything needed to drive it:
 * pixel format, ROI, exposure and the acquisition commands. Anything behind a
 * pAddress, SwissKnife or Converter is reported as absent rather than guessed at,
 * so a wrong address can never be silently invented.
 *
 * Reading the addresses out of the camera's own XML rather than hardcoding them
 * keeps this working across models and firmware revisions.
 */
/** Enough of a node's body to tell what kind it is, and not a kilobyte of XML. */
private const val DESCRIBE_CHARS = 220

/** How many facade nodes a float may hide behind before this gives up. */
private const val FLOAT_HOPS = 3

internal class Genicam(val xml: String) {

    data class Reg(
        val name: String,
        val address: Long,
        val length: Int,
        val access: String,
        /** Bit range for a MaskedIntReg, null for a whole-register IntReg. */
        val lsb: Int? = null,
        val msb: Int? = null,
    ) {
        val writable: Boolean get() = access == "RW" || access == "WO"
        val masked: Boolean get() = lsb != null && msb != null
    }

    /** An enumeration resolved to the register it actually writes. */
    data class EnumNode(
        val value: Reg,
        val locked: Reg?,
        val entries: Map<String, Long>,
        /** Entry name to the flag register that says whether it can be selected. */
        val available: Map<String, Reg>,
    )

    /** A boolean node resolved to its register and its on/off encodings. */
    data class BoolNode(val value: Reg, val locked: Reg?, val onValue: Long, val offValue: Long)

    private val regs = HashMap<String, Reg?>()
    private val enums = HashMap<String, Map<String, Long>>()

    /** Resolves an IntReg or MaskedIntReg by node name. Null when it is not a literal register. */
    fun reg(name: String): Reg? = regs.getOrPut(name) {
        val block = Regex(
            """<(?:Masked)?IntReg\s+Name="${Regex.escape(name)}".*?</(?:Masked)?IntReg>""",
            RegexOption.DOT_MATCHES_ALL,
        ).find(xml)?.value ?: return@getOrPut null

        // A pAddress means the real location is a computed base plus this offset, and
        // those bases are IntSwissKnife formulas over selector state -- GenApi's job,
        // not this class's. Returning the bare offset would address the bootstrap
        // registers instead, which is how TriggerMode_Val once resolved to 0x4.
        if (Regex("""<pAddress>""").containsMatchIn(block)) return@getOrPut null

        val address = number(Regex("""<Address>([^<]+)</Address>""").find(block)?.groupValues?.get(1))
            ?: return@getOrPut null
        val length = Regex("""<Length>(\d+)</Length>""").find(block)?.groupValues?.get(1)?.toIntOrNull()
            ?: return@getOrPut null
        val access = Regex("""<AccessMode>(\w+)</AccessMode>""").find(block)?.groupValues?.get(1) ?: "RO"

        // A MaskedIntReg addresses either a single <Bit> or an <LSB>..<MSB> range inside
        // the register; writing the whole word would clobber its neighbours.
        val bit = Regex("""<Bit>(\d+)</Bit>""").find(block)?.groupValues?.get(1)?.toIntOrNull()
        val lsb = bit ?: Regex("""<LSB>(\d+)</LSB>""").find(block)?.groupValues?.get(1)?.toIntOrNull()
        val msb = bit ?: Regex("""<MSB>(\d+)</MSB>""").find(block)?.groupValues?.get(1)?.toIntOrNull()

        Reg(name, address, length, access, lsb, msb)
    }

    fun require(name: String): Reg = reg(name) ?: throw U3vException("register $name is not a literal IntReg")

    /**
     * Name to value for an Enumeration's entries.
     *
     * Handles the inline `<EnumEntry Name=".."><Value>..</Value></EnumEntry>` form this
     * camera uses; the `pEnumEntry` reference form other vendors emit yields an empty map.
     */
    fun enumEntries(node: String): Map<String, Long> = enums.getOrPut(node) {
        val block = Regex(
            """<Enumeration\s+Name="${Regex.escape(node)}".*?</Enumeration>""",
            RegexOption.DOT_MATCHES_ALL,
        ).find(xml)?.value ?: return@getOrPut emptyMap()

        Regex("""<EnumEntry\s+Name="([^"]+)".*?<Value>([^<]+)</Value>""", RegexOption.DOT_MATCHES_ALL)
            .findAll(block)
            .mapNotNull { m -> number(m.groupValues[2])?.let { m.groupValues[1] to it } }
            .toMap()
    }

    /** Value of one entry of an enumeration, by entry name. */
    fun enumValue(node: String, entry: String): Long? = enumEntries(node)[entry]

    /** Reads a register whose address is a literal in the XML. Null when it is not. */
    /**
     * A float node's value, for the ones the camera stores as IEEE754 rather than as a
     * scaled integer.
     *
     * White balance is the reason this exists. It is a ratio against green, so the one
     * question worth asking of it is whether it is one, and that cannot be answered from
     * a raw integer whose scaling is not modelled -- which is where this investigation
     * stopped last time. A `FloatReg` carries the number itself.
     *
     * @return the value, or null when the node is not a plain float register
     */
    fun readFloat(control: RegisterPort, name: String): Double? {
        // A GenICam <Float> is a facade: the number lives in whatever <pValue> names,
        // which may itself be a facade. Two hops covers Float -> Converter -> FloatReg,
        // which is how this camera expresses its colour ratios, without risking a cycle.
        var target = name
        repeat(FLOAT_HOPS) {
            val direct = readFloatReg(control, target)
            if (direct != null) return direct
            target = pValueOf(target) ?: return@repeat
        }
        return readFloatReg(control, target)
    }

    /** The node a facade points at, or null when it points at nothing this can follow. */
    fun pValueOf(name: String): String? {
        val block = Regex(
            """<(\w+)\s+Name="${Regex.escape(name)}"[^>]*>(.*?)</\1>""",
            RegexOption.DOT_MATCHES_ALL,
        ).find(xml)?.groupValues?.get(2) ?: return null
        return Regex("""<pValue>([^<]+)</pValue>""").find(block)?.groupValues?.get(1)?.trim()
    }

    /** The element type of node [name] ("Integer", "Command", "IntReg", ...), or null when absent. */
    fun kind(name: String): String? =
        Regex("""<(\w+)\s+Name="${Regex.escape(name)}"[^>]*>""").find(xml)?.groupValues?.get(1)

    /**
     * The text of the first `<tag>` directly inside node [name], e.g. `CommandValue` or `pMax`.
     *
     * Enough for the Integer and Command facades, which this class otherwise has no model of:
     * both are a pValue pointing at a literal register plus a handful of scalar properties.
     */
    fun property(name: String, tag: String): String? {
        val block = Regex(
            """<(\w+)\s+Name="${Regex.escape(name)}"[^>]*>(.*?)</\1>""",
            RegexOption.DOT_MATCHES_ALL,
        ).find(xml)?.groupValues?.get(2) ?: return null
        return Regex("""<${Regex.escape(tag)}>([^<]*)</${Regex.escape(tag)}>""").find(block)?.groupValues?.get(1)?.trim()
    }

    private fun readFloatReg(control: RegisterPort, name: String): Double? {
        val block = Regex(
            """<FloatReg\s+Name="${Regex.escape(name)}".*?</FloatReg>""",
            RegexOption.DOT_MATCHES_ALL,
        ).find(xml)?.value ?: return null
        if (Regex("""<pAddress>""").containsMatchIn(block)) return null
        val address = number(
            Regex("""<Address>([^<]+)</Address>""").find(block)?.groupValues?.get(1)
        ) ?: return null
        val length = Regex("""<Length>(\d+)</Length>""").find(block)?.groupValues?.get(1)
            ?.toIntOrNull() ?: return null
        // Big-endian unless the node says otherwise, which is the GenICam default and
        // what this camera uses for its float registers.
        val little = Regex("""<Endianess>LittleEndian</Endianess>""").containsMatchIn(block)
        return try {
            val raw = control.readMemory(address, length)
            var bits = 0L
            if (little) {
                for (i in raw.indices.reversed()) bits = (bits shl 8) or (raw[i].toLong() and 0xFF)
            } else {
                for (b in raw) bits = (bits shl 8) or (b.toLong() and 0xFF)
            }
            if (length == 8) Double.fromBits(bits) else Float.fromBits(bits.toInt()).toDouble()
        } catch (e: Exception) {
            null
        }
    }

    /**
     * How a node is defined in this camera's own description, for when the question is
     * what kind of node it even is.
     *
     * Trimmed to the opening tag and the first few elements: enough to say whether it is a
     * FloatReg, an IntReg behind a converter, or selector-indexed, without putting a
     * kilobyte of XML in a log line.
     */
    fun describe(name: String): String? {
        val block = Regex(
            """<(\w+)\s+Name="${Regex.escape(name)}"[^>]*>(.*?)</\1>""",
            RegexOption.DOT_MATCHES_ALL,
        ).find(xml) ?: return null
        val kind = block.groupValues[1]
        val body = block.groupValues[2]
            .replace(Regex("""\s+"""), " ")
            .take(DESCRIBE_CHARS)
        return "$kind: $body"
    }

    fun read(control: RegisterPort, name: String): Long? = reg(name)?.let { read(control, it) }

    fun read(control: RegisterPort, r: Reg): Long? = try {
        val raw = control.readMemory(r.address, r.length)
        var value = 0L
        for (i in raw.indices.reversed()) value = (value shl 8) or (raw[i].toLong() and 0xFF)
        if (r.masked) (value ushr r.lsb!!) and mask(r) else value
    } catch (e: Exception) {
        null
    }

    /** A masked register is written read-modify-write so the other bits survive. */
    fun write(control: RegisterPort, r: Reg, value: Long) {
        if (!r.masked) {
            control.writeU32(r.address, value)
            return
        }
        val current = read(control, r.copy(lsb = null, msb = null))
            ?: throw U3vException("could not read ${r.name} before a masked write")
        val shift = r.lsb!!
        val m = mask(r) shl shift
        control.writeU32(r.address, (current and m.inv()) or ((value shl shift) and m))
    }

    private fun mask(r: Reg): Long {
        val width = r.msb!! - r.lsb!! + 1
        return if (width >= 64) -1L else (1L shl width) - 1
    }

    /**
     * Resolves an Enumeration to the register it writes, rather than assuming the
     * `<node>_Val` naming. TriggerMode and ExposureAuto happen to follow that
     * convention, but nothing in GenICam requires it.
     */
    fun enumeration(node: String): EnumNode? {
        val block = Regex(
            """<Enumeration\s+Name="${Regex.escape(node)}".*?</Enumeration>""",
            RegexOption.DOT_MATCHES_ALL,
        ).find(xml)?.value ?: return null

        val valueName = Regex("""<pValue>([^<]+)</pValue>""").find(block)?.groupValues?.get(1) ?: return null
        val value = reg(valueName) ?: return null
        val locked = Regex("""<pIsLocked>([^<]+)</pIsLocked>""").find(block)?.groupValues?.get(1)?.let { reg(it) }

        val entries = LinkedHashMap<String, Long>()
        val available = LinkedHashMap<String, Reg>()
        for (m in Regex("""<EnumEntry\s+Name="([^"]+)".*?</EnumEntry>""", RegexOption.DOT_MATCHES_ALL).findAll(block)) {
            val entryName = m.groupValues[1]
            val body = m.value
            number(Regex("""<Value>([^<]+)</Value>""").find(body)?.groupValues?.get(1))
                ?.let { entries[entryName] = it }
            Regex("""<pIsAvailable>([^<]+)</pIsAvailable>""").find(body)?.groupValues?.get(1)
                ?.let { flag -> reg(flag)?.let { available[entryName] = it } }
        }
        if (entries.isEmpty()) return null
        return EnumNode(value, locked, entries, available)
    }

    /** Resolves a Boolean node to its register and its OnValue/OffValue encodings. */
    fun boolean(node: String): BoolNode? {
        val block = Regex(
            """<Boolean\s+Name="${Regex.escape(node)}".*?</Boolean>""",
            RegexOption.DOT_MATCHES_ALL,
        ).find(xml)?.value ?: return null

        val value = Regex("""<pValue>([^<]+)</pValue>""").find(block)?.groupValues?.get(1)
            ?.let { reg(it) } ?: return null
        val locked = Regex("""<pIsLocked>([^<]+)</pIsLocked>""").find(block)?.groupValues?.get(1)?.let { reg(it) }
        val on = number(Regex("""<OnValue>([^<]+)</OnValue>""").find(block)?.groupValues?.get(1)) ?: 1L
        val off = number(Regex("""<OffValue>([^<]+)</OffValue>""").find(block)?.groupValues?.get(1)) ?: 0L
        return BoolNode(value, locked, on, off)
    }

    private fun number(text: String?): Long? {
        val t = text?.trim() ?: return null
        return if (t.startsWith("0x") || t.startsWith("0X")) t.substring(2).toLongOrNull(16) else t.toLongOrNull()
    }
}

/**
 * Pulls the GenICam XML a camera publishes through its manifest table.
 *
 * The manifest's first entry points at the document; entries are 64 bytes of
 * subminor(2) minor(1) major(1) schema(4) address(8) size(8) then reserved words, and
 * bits 10-14 of the schema word say whether the document is zipped.
 */
internal object GenicamLoader {

    private const val TAG = "FlirGenicam"

    /** A document far past this is a misread length rather than a real one. */
    private const val SIZE_LIMIT = 32 * 1024 * 1024

    fun load(control: U3vControl): Genicam? {
        try {
            val tableAddress = control.readU64(Abrm.MANIFEST_TABLE_ADDRESS)
            val entryCount = control.readU64(tableAddress)
            if (entryCount <= 0) return null

            val entry = java.nio.ByteBuffer.wrap(control.readMemory(tableAddress + 8, 64))
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            val schema = entry.getInt(4)
            val address = entry.getLong(8)
            val size = entry.getLong(16)
            val zipped = ((schema ushr 10) and 0x1F) == 1

            android.util.Log.i(TAG, "genicam at 0x${address.toString(16)}, $size bytes, zipped=$zipped")
            if (size <= 0 || size > SIZE_LIMIT) return null

            val raw = control.readMemory(address, size.toInt())
            val text = if (zipped) unzipFirstEntry(raw) else String(raw, Charsets.UTF_8)
            return text?.let { Genicam(it) }
        } catch (e: Exception) {
            android.util.Log.e(TAG, "genicam read failed", e)
            return null
        }
    }

    private fun unzipFirstEntry(raw: ByteArray): String? {
        java.util.zip.ZipInputStream(java.io.ByteArrayInputStream(raw)).use { zip ->
            var entry = zip.nextEntry
            while (entry != null) {
                if (!entry.isDirectory) return zip.readBytes().toString(Charsets.UTF_8)
                entry = zip.nextEntry
            }
        }
        return null
    }
}
