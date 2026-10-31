package com.cgjnkim.mobileokad.flir

/**
 * GenICam features by name, over literal registers only, with the PC capture's semantics.
 *
 * The PC side (`hardware/sensor/flir.py`, `_node_set`) writes a node and reads it back, raising
 * when a required node cannot be set and skipping an optional one. This is the same contract on
 * top of [Genicam]'s register resolution, so the sequencer program can be ported line by line:
 * every node it needs on the BFS-U3-51S5PC is a facade over an IntReg at a fixed address (checked
 * against the camera's node map, see app/NOTES.md). A node whose address is computed from a
 * selector (`pAddress`, e.g. TriggerMode) is reported as unaddressable rather than guessed.
 */
internal class NodeAccess(private val port: RegisterPort, private val genicam: Genicam) {

    class NodeException(message: String) : Exception(message)

    /** The literal register behind [name]: the name itself, or the register its pValue names. */
    fun register(name: String): Genicam.Reg? =
        genicam.reg(name) ?: genicam.pValueOf(name)?.let { genicam.reg(it) }

    fun has(name: String): Boolean = register(name) != null || genicam.enumeration(name) != null

    fun readInt(name: String): Long {
        val reg = register(name) ?: throw NodeException("$name has no directly addressable register")
        return genicam.read(port, reg) ?: throw NodeException("$name could not be read")
    }

    fun readIntOrNull(name: String): Long? = runCatching { readInt(name) }.getOrNull()

    /**
     * Writes an Integer feature (or a bare register) and returns what it reads back.
     *
     * [verify] off is for registers the camera quantises (ExposureTime: 125 µs reads 126), where
     * the read-back *is* the answer rather than a check; the caller keeps that value.
     */
    fun writeInt(name: String, value: Long, verify: Boolean = true): Long {
        val reg = register(name) ?: throw NodeException("$name has no directly addressable register")
        if (!reg.writable) throw NodeException("$name is ${reg.access}")
        checkUnlocked(name)
        try {
            genicam.write(port, reg, value)
        } catch (e: Exception) {
            throw NodeException("$name := $value refused: ${e.message}")
        }
        if (reg.access == "WO") return value
        val after = genicam.read(port, reg) ?: return value
        if (verify && after != value) throw NodeException("$name reads $after after writing $value")
        return after
    }

    /** Selects [entry] of enumeration [node]; throws when it cannot. */
    fun setEnum(node: String, entry: String) {
        val e = genicam.enumeration(node) ?: throw NodeException("$node has no directly addressable value register")
        val value = e.entries[entry] ?: throw NodeException("$node has no entry '$entry'")
        e.available[entry]?.let {
            if (genicam.read(port, it) == 0L) throw NodeException("$node.$entry is not available now")
        }
        if (!e.value.writable) throw NodeException("${e.value.name} is ${e.value.access}")
        checkUnlocked(node)
        try {
            genicam.write(port, e.value, value)
        } catch (ex: Exception) {
            throw NodeException("$node := $entry refused: ${ex.message}")
        }
        val after = genicam.read(port, e.value)
        if (after != null && after != value) throw NodeException("$node reads $after after selecting $entry ($value)")
    }

    /** The current entry of [node], or null when it cannot be read or matches no entry. */
    fun enumEntry(node: String): String? {
        val e = genicam.enumeration(node) ?: return null
        val value = genicam.read(port, e.value) ?: return null
        return e.entries.entries.firstOrNull { it.value == value }?.key
    }

    /**
     * PC `_node_set_or_current`: select [entry], but a node that is already there and refuses
     * the write (read-only in the current state) is not an error.
     */
    fun setEnumOrCurrent(node: String, entry: String) {
        try {
            setEnum(node, entry)
        } catch (e: NodeException) {
            if (enumEntry(node) != entry) throw e
        }
    }

    fun setBool(node: String, on: Boolean) {
        val b = genicam.boolean(node) ?: throw NodeException("$node has no directly addressable value register")
        if (!b.value.writable) throw NodeException("${b.value.name} is ${b.value.access}")
        checkUnlocked(node)
        val want = if (on) b.onValue else b.offValue
        try {
            genicam.write(port, b.value, want)
        } catch (e: Exception) {
            throw NodeException("$node := $on refused: ${e.message}")
        }
        val after = genicam.read(port, b.value)
        if (after != null && after != want) throw NodeException("$node reads $after after writing $want")
    }

    fun readBool(node: String): Boolean? {
        val b = genicam.boolean(node) ?: return null
        return genicam.read(port, b.value)?.let { it == b.onValue }
    }

    /** Fires a Command node: writes its CommandValue (1 when the node does not say) to its register. */
    fun execute(node: String) {
        val reg = register(node) ?: throw NodeException("$node has no directly addressable register")
        val value = genicam.property(node, "CommandValue")?.toLongOrNull() ?: 1L
        checkUnlocked(node)
        try {
            genicam.write(port, reg, value)
        } catch (e: Exception) {
            throw NodeException("$node refused: ${e.message}")
        }
    }

    /**
     * Runs [action] for an optional node: null when it was applied, otherwise why not.
     * Mirrors `_node_set(..., optional=True)` -- the reason is kept so the caller can record it.
     */
    inline fun optional(action: () -> Unit): String? = try {
        action()
        null
    } catch (e: NodeException) {
        e.message ?: "refused"
    }

    /**
     * Refuses to write a node the camera says is locked, when the lock is a literal register.
     *
     * Many locks here are IntSwissKnife formulas (`TLP || LCK`) that this driver does not evaluate;
     * those are not checked, and the camera's own refusal then decides.
     */
    private fun checkUnlocked(node: String) {
        val lockName = genicam.property(node, "pIsLocked") ?: return
        val lock = genicam.reg(lockName) ?: return
        if (genicam.read(port, lock) == 1L) throw NodeException("$node is locked by the camera")
    }
}
