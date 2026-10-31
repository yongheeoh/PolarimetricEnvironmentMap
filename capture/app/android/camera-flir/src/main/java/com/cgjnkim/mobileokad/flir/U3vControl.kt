package com.cgjnkim.mobileokad.flir

import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbInterface
import android.util.Log
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * The USB3 Vision control channel, spoken directly over UsbDeviceConnection.
 *
 * A USB3 Vision camera exposes its whole register space over two bulk endpoints on
 * the control interface, so reading the bootstrap registers and the GenICam XML needs
 * no vendor SDK and no native code -- only GenCP framing, which Uvcp provides. That is
 * enough to prove the camera answers on Android before committing to an Aravis port.
 *
 * Not thread safe: one transaction at a time, off the main thread.
 */
internal class U3vControl private constructor(
    private val connection: UsbDeviceConnection,
    private val controlInterface: UsbInterface,
    private val endpointOut: UsbEndpoint,
    private val endpointIn: UsbEndpoint,
    private val dataInterface: UsbInterface?,
    private val dataEndpoint: UsbEndpoint?,
) : Closeable, RegisterPort {

    private var requestId = 1

    /**
     * Conservative until configure() reads the real limit. It only has to be large
     * enough for the ABRM reads that discover the limit itself.
     */
    private var ackLimit = 1024
    private var timeoutMs = 1000

    var maxCmdTransfer = 0
        private set
    var maxAckTransfer = 0
        private set
    var sbrmAddress = 0L
        private set
    var maxDeviceResponseTimeMs = 0
        private set

    /**
     * Reads the registers needed to talk efficiently, and raises the transfer limits.
     * Safe to call once the interface is claimed; every later read depends on it.
     */
    fun configure() {
        // Before the first word is sent. A process killed while transfers were queued --
        // a force-stop, a crash, the debugger detaching -- can leave an endpoint halted,
        // and a halted endpoint fails every transfer with -1 until somebody clears it.
        // The symptom is that the camera looks broken from the very first register read
        // and stays that way across reconnects until it is physically unplugged, which
        // is a poor thing to require of anyone holding a rig together with one hand.
        clearHalt(endpointOut)
        clearHalt(endpointIn)
        dataEndpoint?.let { clearHalt(it) }

        maxDeviceResponseTimeMs = readU32(Abrm.MAX_DEVICE_RESPONSE_TIME).toInt()
        timeoutMs = max(TIMEOUT_FLOOR_MS, maxDeviceResponseTimeMs)

        sbrmAddress = readU64(Abrm.SBRM_ADDRESS)
        maxCmdTransfer = readU32(sbrmAddress + Sbrm.MAX_CMD_TRANSFER).toInt()
        maxAckTransfer = readU32(sbrmAddress + Sbrm.MAX_ACK_TRANSFER).toInt()

        // Android's bulkTransfer is only reliable up to a modest buffer, well below the
        // 64 KiB GenCP allows, so the ceiling here is the transport's and not the camera's.
        ackLimit = min(min(maxAckTransfer, Uvcp.PACKET_SIZE_MAX), ANDROID_BULK_LIMIT)
    }

    /** Splits into as many READMEM transactions as the negotiated ack size requires. */
    override fun readMemory(address: Long, size: Int): ByteArray {
        val out = ByteArray(size)
        val chunk = ackLimit - Uvcp.HEADER_SIZE
        require(chunk > 0) { "ack limit $ackLimit is smaller than a GenCP header" }

        var done = 0
        while (done < size) {
            val n = min(chunk, size - done)
            val payload = transact(address + done, n)
            System.arraycopy(payload, 0, out, done, min(n, payload.size))
            if (payload.size < n) {
                throw U3vException("short read at 0x${(address + done).toString(16)}: ${payload.size} of $n")
            }
            done += n
        }
        return out
    }

    fun readU32(address: Long): Long =
        ByteBuffer.wrap(readMemory(address, 4)).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL

    fun readU64(address: Long): Long =
        ByteBuffer.wrap(readMemory(address, 8)).order(ByteOrder.LITTLE_ENDIAN).long

    /** ABRM name fields are fixed width and NUL padded. */
    fun readString(address: Long, size: Int = Abrm.STRING_SIZE): String {
        val raw = readMemory(address, size)
        val end = raw.indexOf(0.toByte()).let { if (it < 0) raw.size else it }
        return String(raw, 0, end, Charsets.US_ASCII).trim()
    }

    fun writeMemory(address: Long, data: ByteArray) {
        // WRITEMEM has to fit one command transfer, and MAX_CMD_TRANSFER is small on
        // this link; every register written here is 4 or 8 bytes, so no chunking.
        val limit = (if (maxCmdTransfer > 0) maxCmdTransfer else Uvcp.PACKET_SIZE_MAX) - Uvcp.HEADER_SIZE - 8
        require(data.size <= limit) { "write of ${data.size} B exceeds the ${limit} B command payload" }
        transactWrite(address, data)
    }

    override fun writeU32(address: Long, value: Long) = writeMemory(
        address,
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value.toInt()).array(),
    )

    fun writeU64(address: Long, value: Long) = writeMemory(
        address,
        ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(value).array(),
    )

    /**
     * Clears a stalled endpoint: the standard CLEAR_FEATURE(ENDPOINT_HALT).
     *
     * Harmless when the endpoint is not halted, which is why it is unconditional rather
     * than a recovery path -- there is no way to ask whether it is, and trying costs one
     * control transfer at open.
     */
    private fun clearHalt(endpoint: UsbEndpoint) {
        val sent = runCatching {
            connection.controlTransfer(
                UsbConstants.USB_DIR_OUT or UsbConstants.USB_TYPE_STANDARD or RECIPIENT_ENDPOINT,
                REQUEST_CLEAR_FEATURE,
                FEATURE_ENDPOINT_HALT,
                endpoint.address,
                null,
                0,
                CLEAR_HALT_TIMEOUT_MS,
            )
        }.getOrDefault(-1)
        if (sent < 0) Log.i(TAG, "could not clear halt on endpoint 0x%02x".format(endpoint.address))
    }

    /**
     * A pipe of queued reads on the stream endpoint, or null when there is no stream.
     *
     * Owned by the caller and closed by it: this class only knows the connection and the
     * endpoint, which are the two things making one needs.
     */
    fun openStreamPipe(depth: Int, bufferSize: Int): StreamPipe? {
        val ep = dataEndpoint ?: return null
        return runCatching { StreamPipe(connection, ep, depth, bufferSize) }
            .onFailure { Log.w(TAG, "no queued stream reads, falling back to synchronous", it) }
            .getOrNull()
    }

    /**
     * Reads one bulk transfer from the stream endpoint.
     *
     * Transfer boundaries matter here: the camera sends the leader, each payload chunk
     * and the trailer as separate transfers of exactly the sizes negotiated through the
     * SIRM, so the caller has to ask for those sizes and not merge them.
     */
    fun readStream(buffer: ByteArray, offset: Int, length: Int, timeoutMs: Int): Int {
        val ep = dataEndpoint ?: throw U3vException("no stream endpoint claimed")
        return connection.bulkTransfer(ep, buffer, offset, length, timeoutMs)
    }

    /**
     * One READMEM round trip.
     *
     * Retries because a stale acknowledgement from a previous, timed out transaction can
     * still be sitting in the endpoint; those are recognised by a mismatched id and
     * dropped rather than treated as an error.
     */
    private fun transact(address: Long, size: Int): ByteArray =
        roundTrip(address, Uvcp.ACK_READ_MEMORY) { id -> Uvcp.readMemoryCmd(address, size, id) }

    /** Same round trip, but the ack carries only a byte count we discard. */
    private fun transactWrite(address: Long, data: ByteArray) {
        roundTrip(address, Uvcp.ACK_WRITE_MEMORY) { id -> Uvcp.writeMemoryCmd(address, data, id) }
    }

    private inline fun roundTrip(address: Long, expectedAck: Int, build: (Int) -> ByteArray): ByteArray {
        var lastError: String? = null

        for (attempt in 1..N_TRIES_MAX) {
            val id = nextRequestId()
            val cmd = build(id)

            val sent = connection.bulkTransfer(endpointOut, cmd, cmd.size, timeoutMs)
            if (sent != cmd.size) {
                lastError = "command write returned $sent, expected ${cmd.size}"
                continue
            }

            val buffer = ByteArray(ackLimit)
            var deadline = System.currentTimeMillis() + timeoutMs

            while (System.currentTimeMillis() <= deadline) {
                val remaining = max(1, (deadline - System.currentTimeMillis()).toInt())
                val received = connection.bulkTransfer(endpointIn, buffer, buffer.size, remaining)
                if (received < 0) {
                    lastError = "ack read failed"
                    break
                }

                val ack = Uvcp.parseAck(buffer, received)
                if (ack == null) {
                    lastError = "malformed ack ($received bytes)"
                    continue
                }
                if (ack.id != id) {
                    // Left over from an earlier transaction; drop it and keep waiting.
                    Log.d(TAG, "dropping stale ack id=${ack.id}, waiting for $id")
                    continue
                }
                if (ack.command == Uvcp.ACK_PENDING) {
                    // The camera asked for more time and will answer on this same id.
                    deadline = System.currentTimeMillis() + Uvcp.pendingAckTimeoutMs(ack.payload)
                    continue
                }
                if (ack.status != Uvcp.STATUS_SUCCESS) {
                    throw U3vException(
                        "0x${address.toString(16)} rejected: ${Uvcp.statusName(ack.status)}"
                    )
                }
                if (ack.command != expectedAck) {
                    lastError = "unexpected ack command 0x${ack.command.toString(16)}"
                    break
                }
                return ack.payload
            }

            Log.w(TAG, "attempt $attempt/$N_TRIES_MAX for 0x${address.toString(16)} failed: $lastError")
        }

        throw U3vException("0x${address.toString(16)} failed after $N_TRIES_MAX tries: $lastError")
    }

    private fun nextRequestId(): Int {
        // 0 is reserved, so wrap to 1 rather than to 0.
        requestId = if (requestId >= 0xFFFF) 1 else requestId + 1
        return requestId
    }

    override fun close() {
        dataInterface?.let { connection.releaseInterface(it) }
        connection.releaseInterface(controlInterface)
        connection.close()
    }

    companion object {
        private const val TAG = "FlirU3v"
        private const val N_TRIES_MAX = 5
        private const val TIMEOUT_FLOOR_MS = 1000
        /** USB standard CLEAR_FEATURE on an endpoint recipient, feature ENDPOINT_HALT. */
        private const val RECIPIENT_ENDPOINT = 0x02
        private const val REQUEST_CLEAR_FEATURE = 0x01
        private const val FEATURE_ENDPOINT_HALT = 0x00
        private const val CLEAR_HALT_TIMEOUT_MS = 200

        private const val ANDROID_BULK_LIMIT = 16384

        /**
         * Claims the control interface of [device] on an already permitted connection.
         *
         * The interface is located by descriptor (class 0xEF / subclass 0x05 / protocol 0)
         * the same way Aravis does, rather than by index, so a camera that orders its
         * interfaces differently still works.
         */
        fun open(device: UsbDevice, connection: UsbDeviceConnection): U3vControl {
            val iface = (0 until device.interfaceCount)
                .map { device.getInterface(it) }
                .firstOrNull {
                    it.interfaceClass == Uvcp.INTERFACE_CLASS &&
                        it.interfaceSubclass == Uvcp.INTERFACE_SUBCLASS &&
                        it.interfaceProtocol == Uvcp.PROTOCOL_CONTROL
                }
                ?: throw U3vException("no USB3 Vision control interface (0xEF/0x05/0x00) on this device")

            var out: UsbEndpoint? = null
            var input: UsbEndpoint? = null
            for (i in 0 until iface.endpointCount) {
                val ep = iface.getEndpoint(i)
                if (ep.type != UsbConstants.USB_ENDPOINT_XFER_BULK) continue
                if (ep.direction == UsbConstants.USB_DIR_IN) input = ep else out = ep
            }
            if (out == null || input == null) {
                throw U3vException("control interface ${iface.id} has no bulk IN/OUT pair")
            }

            if (!connection.claimInterface(iface, true)) {
                connection.close()
                throw U3vException("claimInterface(${iface.id}) refused")
            }

            // The data interface carries the image stream. It is optional so that a
            // control-only probe still works on a camera that exposes no stream channel.
            val dataIface = (0 until device.interfaceCount)
                .map { device.getInterface(it) }
                .firstOrNull {
                    it.interfaceClass == Uvcp.INTERFACE_CLASS &&
                        it.interfaceSubclass == Uvcp.INTERFACE_SUBCLASS &&
                        it.interfaceProtocol == Uvcp.PROTOCOL_DATA
                }
            var dataEp: UsbEndpoint? = null
            if (dataIface != null) {
                for (i in 0 until dataIface.endpointCount) {
                    val ep = dataIface.getEndpoint(i)
                    if (ep.type == UsbConstants.USB_ENDPOINT_XFER_BULK &&
                        ep.direction == UsbConstants.USB_DIR_IN
                    ) {
                        dataEp = ep
                    }
                }
                if (dataEp == null || !connection.claimInterface(dataIface, true)) {
                    Log.w(TAG, "data interface ${dataIface.id} could not be claimed; control only")
                    dataEp = null
                }
            }

            // The stream endpoint's packet size names the link the camera actually
            // negotiated, which nothing else on the phone will tell us: a bulk IN
            // endpoint is 512 bytes on USB 2.0 High Speed and 1024 on USB 3
            // SuperSpeed. It decides the preview frame rate -- 5 MP of packed 12-bit is
            // 7.5 MB a frame, so the difference is a few frames a second against a few
            // dozen -- and every bandwidth decision above depends on knowing which.
            val link = when (dataEp?.maxPacketSize) {
                null -> "no stream endpoint"
                1024 -> "USB 3 SuperSpeed (1024 B packets)"
                512 -> "USB 2.0 High Speed (512 B packets)"
                else -> "unrecognised link, ${dataEp.maxPacketSize} B packets"
            }
            Log.i(
                TAG,
                "control interface ${iface.id} OUT 0x%02x / IN 0x%02x, stream IN 0x%02x, %s"
                    .format(out.address, input.address, dataEp?.address ?: 0, link)
            )
            return U3vControl(connection, iface, out, input, if (dataEp != null) dataIface else null, dataEp)
        }
    }
}

internal class U3vException(message: String) : Exception(message)
