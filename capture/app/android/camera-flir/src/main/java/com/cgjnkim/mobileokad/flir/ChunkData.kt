package com.cgjnkim.mobileokad.flir

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * GenICam chunk data at the end of a payload.
 *
 * With ChunkModeActive the camera appends chunks after (and around) the image: each chunk is its
 * data followed by an 8-byte tag, chunk id then data length, so the layout is walked from the end
 * of the payload towards the start. The image itself is a chunk on this camera, so a well-formed
 * payload walks back to exactly offset 0.
 *
 * The tag byte order is not documented for USB3 Vision the way it is for GigE Vision (big endian),
 * so both orders are tried and the one that walks back to offset 0 wins. Ids are the camera's own
 * (ChunkID of each chunk port in its node map): FrameID 0x05040003, ExposureTime 0x05040008,
 * SequencerSetActive 0x0504000C.
 */
class ChunkData private constructor(
    /** Chunk id to (offset, length) inside the payload. */
    val chunks: Map<Long, Pair<Int, Int>>,
    val bigEndianTags: Boolean,
    private val payload: ByteArray,
) {
    fun bytes(id: Long): ByteArray? = chunks[id]?.let { (o, n) -> payload.copyOfRange(o, o + n) }

    /** Chunk values are little endian on this camera (Endianess=LittleEndian on every chunk register). */
    fun u64(id: Long): Long? = chunks[id]?.takeIf { it.second >= 8 }?.let { (o, _) ->
        ByteBuffer.wrap(payload, o, 8).order(ByteOrder.LITTLE_ENDIAN).long
    }

    fun u32(id: Long): Long? = chunks[id]?.takeIf { it.second >= 4 }?.let { (o, _) ->
        ByteBuffer.wrap(payload, o, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL
    }

    fun s32(id: Long): Int? = chunks[id]?.takeIf { it.second >= 4 }?.let { (o, _) ->
        ByteBuffer.wrap(payload, o, 4).order(ByteOrder.LITTLE_ENDIAN).int
    }

    val frameId: Long? get() = u64(FRAME_ID)
    val exposureUs: Long? get() = u32(EXPOSURE_TIME)
    val sequencerSet: Int? get() = s32(SEQUENCER_SET_ACTIVE)

    companion object {
        const val FRAME_ID = 0x05040003L
        const val EXPOSURE_TIME = 0x05040008L
        const val SEQUENCER_SET_ACTIVE = 0x0504000CL

        private const val TAG_SIZE = 8

        /**
         * Parses the chunks of [payload] (first [length] bytes). Returns null when neither tag order
         * walks the payload back to offset 0 -- i.e. there are no chunks or they are not laid out as
         * expected -- so the caller can fall back to treating the payload as a bare image.
         */
        fun parse(payload: ByteArray, length: Int = payload.size): ChunkData? {
            walk(payload, length, ByteOrder.LITTLE_ENDIAN)?.let { return ChunkData(it, false, payload) }
            walk(payload, length, ByteOrder.BIG_ENDIAN)?.let { return ChunkData(it, true, payload) }
            // The payload is the camera's maximum size, rounded up; without the trailer's valid size the
            // tags sit before zero padding. Strip whole zero words from the end and try again.
            var end = length
            while (end >= 8 && isZero(payload, end - 8, 8)) end -= 8
            if (end == length || end < 8) return null
            walk(payload, end, ByteOrder.LITTLE_ENDIAN)?.let { return ChunkData(it, false, payload) }
            walk(payload, end, ByteOrder.BIG_ENDIAN)?.let { return ChunkData(it, true, payload) }
            return null
        }

        /** Parses using the USB3 Vision trailer's valid payload size when the frame has one. */
        internal fun parse(frame: FlirCamera.Frame): ChunkData? {
            val valid = frame.validPayloadBytes
            return if (valid in 8..frame.data.size.toLong()) parse(frame.data, valid.toInt()) else parse(frame.data)
        }

        private fun isZero(b: ByteArray, from: Int, n: Int): Boolean {
            for (i in from until from + n) if (b[i] != 0.toByte()) return false
            return true
        }

        private fun walk(payload: ByteArray, length: Int, order: ByteOrder): Map<Long, Pair<Int, Int>>? {
            if (length < TAG_SIZE || length > payload.size) return null
            val out = LinkedHashMap<Long, Pair<Int, Int>>()
            val buf = ByteBuffer.wrap(payload).order(order)
            var end = length
            var guard = 0
            while (end > 0) {
                if (end < TAG_SIZE || guard++ > 64) return null
                val id = buf.getInt(end - TAG_SIZE).toLong() and 0xFFFFFFFFL
                val size = buf.getInt(end - 4).toLong() and 0xFFFFFFFFL
                val start = end - TAG_SIZE - size
                if (start < 0) return null
                out[id] = start.toInt() to size.toInt()
                end = start.toInt()
            }
            return out
        }
    }
}
