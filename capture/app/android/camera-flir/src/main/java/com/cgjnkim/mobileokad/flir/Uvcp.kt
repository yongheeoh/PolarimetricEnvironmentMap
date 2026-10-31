package com.cgjnkim.mobileokad.flir

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * USB3 Vision control protocol (GenCP) packet layout.
 *
 * Every constant and struct offset here is copied from Aravis' src/arvuvcpprivate.h
 * rather than derived, because Aravis is the implementation this module is expected
 * to hand over to once it is cross-compiled. Keeping the names identical makes the
 * two diffable.
 *
 * All fields are little endian on the wire.
 */
internal object Uvcp {

    /** ARV_UVCP_MAGIC. Reads as "U3VC" on the wire. */
    const val MAGIC = 0x43563355

    /** magic(4) | flags-or-status(2) | command(2) | size(2) | id(2) */
    const val HEADER_SIZE = 12

    /** ARV_UVCP_FLAGS_REQUEST_ACK */
    const val FLAGS_REQUEST_ACK = 1 shl 14

    const val CMD_READ_MEMORY = 0x0800
    const val ACK_READ_MEMORY = 0x0801
    const val CMD_WRITE_MEMORY = 0x0802
    const val ACK_WRITE_MEMORY = 0x0803
    const val ACK_PENDING = 0x0805

    const val STATUS_SUCCESS = 0x0000

    /** GenCP status codes, so a rejection says what the camera objected to. */
    private val STATUS_NAMES = mapOf(
        0x0000 to "SUCCESS",
        0x8001 to "NOT_IMPLEMENTED",
        0x8002 to "INVALID_PARAMETER",
        0x8003 to "INVALID_ADDRESS",
        0x8004 to "WRITE_PROTECT",
        0x8005 to "BAD_ALIGNMENT",
        0x8006 to "ACCESS_DENIED",
        0x8007 to "BUSY",
        0x800B to "MSG_TIMEOUT",
        0x800E to "INVALID_HEADER",
        0x800F to "WRONG_CONFIG",
    )

    fun statusName(status: Int): String =
        STATUS_NAMES[status] ?: "0x%04x".format(status)

    /**
     * Aravis' starting point, clamped later by SBRM.MAX_{CMD,ACK}_TRANSFER. It has to
     * be a guess: MAX_ACK_TRANSFER itself can only be read once the channel works.
     */
    const val PACKET_SIZE_MAX = 65536 + HEADER_SIZE

    const val DEFAULT_RESPONSE_TIME_MS = 5

    /** USB3 Vision descriptors, from arvuvinterfaceprivate.h. */
    const val INTERFACE_CLASS = 0xef
    const val INTERFACE_SUBCLASS = 0x05
    const val PROTOCOL_CONTROL = 0x00
    const val PROTOCOL_EVENT = 0x01
    const val PROTOCOL_DATA = 0x02

    /** READMEM command: 12 byte header followed by address(8) | reserved(2) | size(2). */
    fun readMemoryCmd(address: Long, size: Int, id: Int): ByteArray {
        val buf = ByteBuffer.allocate(HEADER_SIZE + 12).order(ByteOrder.LITTLE_ENDIAN)
        buf.putInt(MAGIC)
        buf.putShort(FLAGS_REQUEST_ACK.toShort())
        buf.putShort(CMD_READ_MEMORY.toShort())
        buf.putShort(12)               // size of the payload that follows the header
        buf.putShort(id.toShort())
        buf.putLong(address)
        buf.putShort(0)                // reserved, always 0
        buf.putShort(size.toShort())
        return buf.array()
    }

    /**
     * WRITEMEM command: header, then address(8), then the bytes to write. The header's
     * size field counts the address as well as the data.
     */
    fun writeMemoryCmd(address: Long, data: ByteArray, id: Int): ByteArray {
        val buf = ByteBuffer.allocate(HEADER_SIZE + 8 + data.size).order(ByteOrder.LITTLE_ENDIAN)
        buf.putInt(MAGIC)
        buf.putShort(FLAGS_REQUEST_ACK.toShort())
        buf.putShort(CMD_WRITE_MEMORY.toShort())
        buf.putShort((8 + data.size).toShort())
        buf.putShort(id.toShort())
        buf.putLong(address)
        buf.put(data)
        return buf.array()
    }

    /** Parsed acknowledgement header. [payload] is whatever followed it. */
    data class Ack(val status: Int, val command: Int, val size: Int, val id: Int, val payload: ByteArray)

    /**
     * Returns null when [length] bytes do not start with a well formed header, which on
     * this transport means a stale or truncated read rather than a protocol error.
     */
    fun parseAck(raw: ByteArray, length: Int): Ack? {
        if (length < HEADER_SIZE) return null
        val buf = ByteBuffer.wrap(raw, 0, length).order(ByteOrder.LITTLE_ENDIAN)
        if (buf.int != MAGIC) return null
        val status = buf.short.toInt() and 0xFFFF
        val command = buf.short.toInt() and 0xFFFF
        val size = buf.short.toInt() and 0xFFFF
        val id = buf.short.toInt() and 0xFFFF
        val available = minOf(size, length - HEADER_SIZE)
        if (available < 0) return null
        val payload = ByteArray(available)
        buf.get(payload)
        return Ack(status, command, size, id, payload)
    }

    /** PENDING_ACK payload is unknown(2) | timeout(2), in ms. */
    fun pendingAckTimeoutMs(payload: ByteArray): Int {
        if (payload.size < 4) return DEFAULT_RESPONSE_TIME_MS
        return ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN).getShort(2).toInt() and 0xFFFF
    }
}

/** Technology Agnostic Bootstrap Register Map, always based at address 0. */
internal object Abrm {
    const val GENCP_VERSION = 0x0000L
    const val MANUFACTURER_NAME = 0x0004L
    const val MODEL_NAME = 0x0044L
    const val FAMILY_NAME = 0x0084L
    const val DEVICE_VERSION = 0x00c4L
    const val MANUFACTURER_INFO = 0x0104L
    const val SERIAL_NUMBER = 0x0144L
    const val USER_DEFINED_NAME = 0x0184L
    const val DEVICE_CAPABILITY = 0x01c4L
    const val MAX_DEVICE_RESPONSE_TIME = 0x01ccL   // ms
    const val MANIFEST_TABLE_ADDRESS = 0x01d0L
    const val SBRM_ADDRESS = 0x01d8L
    const val TIMESTAMP = 0x01f0L            // 8 bytes, latched value
    const val TIMESTAMP_LATCH = 0x01f8L      // write 1 to sample the device clock
    const val TIMESTAMP_INCREMENT = 0x01fcL  // ns per tick

    /** Every name-like ABRM register is a fixed 64 byte NUL padded ASCII field. */
    const val STRING_SIZE = 64
}

/** Technology Specific Bootstrap Register Map, based at ABRM.SBRM_ADDRESS. */
internal object Sbrm {
    const val U3V_VERSION = 0x0000L
    const val MAX_CMD_TRANSFER = 0x0014L
    const val MAX_ACK_TRANSFER = 0x0018L
    const val NUM_STREAM_CHANNELS = 0x001cL
    const val SIRM_ADDRESS = 0x0020L
    const val CURRENT_SPEED = 0x0040L
}

/**
 * USB3 Vision stream protocol, from Aravis src/arvuvspprivate.h.
 *
 * A frame arrives on the data endpoint as three kinds of bulk transfer: one leader,
 * a fixed number of payload transfers, then one trailer.
 */
internal object Uvsp {
    const val LEADER_MAGIC = 0x4C563355   // "U3VL"
    const val TRAILER_MAGIC = 0x54563355  // "U3VT"

    /** magic(4) | unknown0(2) | size(2) | frame_id(8) */
    const val HEADER_SIZE = 16

    const val PAYLOAD_TYPE_IMAGE = 0x0001

    /**
     * Leader infos, following the header:
     * unknown0(2) payload_type(2) timestamp(8) pixel_format(4)
     * width(4) height(4) x_offset(4) y_offset(4) x_padding(2) unknown1(2)
     */
    const val LEADER_PAYLOAD_TYPE = HEADER_SIZE + 2
    const val LEADER_TIMESTAMP = HEADER_SIZE + 4
    const val LEADER_PIXEL_FORMAT = HEADER_SIZE + 12
    const val LEADER_WIDTH = HEADER_SIZE + 16
    const val LEADER_HEIGHT = HEADER_SIZE + 20
    const val LEADER_X_OFFSET = HEADER_SIZE + 24
    const val LEADER_Y_OFFSET = HEADER_SIZE + 28
    const val LEADER_X_PADDING = HEADER_SIZE + 32

    /** Trailer infos: unknown0(4) payload_size(8) */
    const val TRAILER_PAYLOAD_SIZE = HEADER_SIZE + 4

    const val SIRM_CONTROL_STREAM_ENABLE = 0x00000001
    const val SIRM_INFO_ALIGNMENT_MASK = 0xFF000000L
    const val SIRM_INFO_ALIGNMENT_SHIFT = 24
}

/** PFNC codes this module cares about, as declared by the camera's own PixelFormat enum. */
internal object PixelFormats {
    const val BAYER_RG_POLARIZED_8 = 0x01080064L
    const val POLARIZED_8 = 0x01080060L
    const val POLARIZED_12P = 0x010c0062L
    const val BAYER_RG_POLARIZED_12P = 0x010c0066L
    const val BAYER_RG_8 = 0x01080009L
    const val MONO_8 = 0x01080001L
}
