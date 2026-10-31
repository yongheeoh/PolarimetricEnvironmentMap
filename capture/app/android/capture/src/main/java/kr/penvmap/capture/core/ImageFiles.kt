package kr.penvmap.capture.core

import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.io.RandomAccessFile
import java.util.zip.CRC32
import java.util.zip.Deflater
import java.util.zip.DeflaterOutputStream

/**
 * `.npy` writer for the HDR stacks.
 *
 * Each group's `hdr/{i}.npy` holds a (T, H, W) uint16 stack. The frames arrive from the camera as BayerRG16
 * payloads -- little-endian uint16 -- which is exactly numpy's `<u2`, so they are written byte for byte
 * after the header, with no conversion and no second copy of the stack in memory.
 */
object Npy {

    data class RawFrame(val bytes: ByteArray, val width: Int, val height: Int)

    /** Read just one exposure for the gallery, without loading or rewriting the entire HDR stack. */
    fun readMiddleU16(file: File): RawFrame = RandomAccessFile(file, "r").use { input ->
        val magic = ByteArray(6).also { input.readFully(it) }
        require(magic.contentEquals(byteArrayOf(0x93.toByte(), 78, 85, 77, 80, 89))) { "Not an NPY file" }
        val major = input.readUnsignedByte()
        val minor = input.readUnsignedByte()
        require(major in 1..3 && minor == 0) { "Unsupported NPY version" }
        val headerSize = if (major == 1) java.lang.Short.reverseBytes(input.readShort()).toInt() and 0xffff
            else Integer.reverseBytes(input.readInt())
        require(headerSize in 1..65536) { "Invalid NPY header length" }
        val header = ByteArray(headerSize).also { input.readFully(it) }.toString(Charsets.UTF_8)
        require(Regex("['\"]descr['\"]\\s*:\\s*['\"]<u2['\"]").containsMatchIn(header) &&
            Regex("['\"]fortran_order['\"]\\s*:\\s*False").containsMatchIn(header)) { "Expected C-order uint16 RAW" }
        val shape = Regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)").find(header)?.groupValues?.get(1)
            ?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() }?.map { it.toInt() }
            ?: error("Missing NPY shape")
        require(shape.size == 3 && shape.all { it in 1..16384 }) { "Invalid RAW stack shape" }
        val (count, height, width) = shape
        val frameSize = height.toLong() * width * 2
        require(frameSize <= 128L * 1024 * 1024) { "RAW frame too large" }
        val dataStart = input.filePointer
        require(input.length() >= dataStart + frameSize * count) { "Truncated RAW stack" }
        input.seek(dataStart + frameSize * (count / 2))
        RawFrame(ByteArray(frameSize.toInt()).also { input.readFully(it) }, width, height)
    }

    fun header(descr: String, shape: IntArray): ByteArray {
        val dims = if (shape.size == 1) "(${shape[0]},)" else shape.joinToString(", ", "(", ")")
        val dict = "{'descr': '$descr', 'fortran_order': False, 'shape': $dims, }"
        // magic(6) + version(2) + header length(2) + dict, padded with spaces and ending in '\n' so the
        // data starts on a 64-byte boundary (numpy >= 1.x writes 64; readers accept any multiple of 16).
        val unpadded = 10 + dict.length + 1
        val pad = (64 - unpadded % 64) % 64
        val text = dict + " ".repeat(pad) + "\n"
        val out = ByteArrayOutputStream()
        out.write(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte(), 1, 0))
        out.write(text.length and 0xFF)
        out.write((text.length shr 8) and 0xFF)
        out.write(text.toByteArray(Charsets.US_ASCII))
        return out.toByteArray()
    }

    /** Writes frames of little-endian uint16 (each [height]*[width] samples) as a (T, H, W) `<u2` array. */
    fun writeU16Stack(file: File, frames: List<ByteArray>, height: Int, width: Int) {
        val frameBytes = height * width * 2
        for ((i, f) in frames.withIndex()) require(f.size >= frameBytes) { "frame $i has ${f.size} bytes, need $frameBytes" }
        BufferedOutputStream(FileOutputStream(file), 1 shl 20).use { out ->
            out.write(header("<u2", intArrayOf(frames.size, height, width)))
            for (f in frames) out.write(f, 0, frameBytes)
        }
    }

    /** Writes uint8 samples as a (T, H, W) `|u1` array (the 8-bit cameras' stacks). */
    fun writeU8Stack(file: File, frames: List<ByteArray>, height: Int, width: Int) {
        BufferedOutputStream(FileOutputStream(file), 1 shl 20).use { out ->
            out.write(header("|u1", intArrayOf(frames.size, height, width)))
            for (f in frames) out.write(f, 0, height * width)
        }
    }
}

/**
 * PNG writer for the two image kinds the capture saves: the mid-exposure raw frame as 16-bit
 * grayscale (`ldr/{i}.png`, lossless like the PC's cv2.imwrite of a uint16 array) and legacy 8-bit BGR
 * images. New captures save only RAW; previews are rendered in memory.
 */
object Png {

    /** 16-bit grayscale from little-endian uint16 samples (PNG stores big endian). */
    fun writeGray16(file: File, samplesLe: ByteArray, width: Int, height: Int, level: Int = 1) {
        val row = ByteArray(1 + width * 2)
        write(file, width, height, bitDepth = 16, colorType = 0, level = level) { y, out ->
            row[0] = 0
            var src = y * width * 2
            var dst = 1
            for (x in 0 until width) {
                row[dst] = samplesLe[src + 1]
                row[dst + 1] = samplesLe[src]
                src += 2
                dst += 2
            }
            out.write(row)
        }
    }

    /** 8-bit colour from interleaved BGR bytes (OpenCV order, as the PC writes them). */
    fun writeBgr8(file: File, bgr: ByteArray, width: Int, height: Int, level: Int = 6) {
        val row = ByteArray(1 + width * 3)
        write(file, width, height, bitDepth = 8, colorType = 2, level = level) { y, out ->
            row[0] = 0
            var src = y * width * 3
            var dst = 1
            for (x in 0 until width) {
                row[dst] = bgr[src + 2]
                row[dst + 1] = bgr[src + 1]
                row[dst + 2] = bgr[src]
                src += 3
                dst += 3
            }
            out.write(row)
        }
    }

    private fun write(file: File, width: Int, height: Int, bitDepth: Int, colorType: Int, level: Int, rows: (Int, OutputStream) -> Unit) {
        val idat = ByteArrayOutputStream()
        val deflater = Deflater(level)
        DeflaterOutputStream(idat, deflater, 1 shl 16).use { z -> for (y in 0 until height) rows(y, z) }
        deflater.end()
        DataOutputStream(BufferedOutputStream(FileOutputStream(file), 1 shl 16)).use { out ->
            out.write(byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(), 0x0D, 0x0A, 0x1A, 0x0A))
            val ihdr = ByteArrayOutputStream()
            DataOutputStream(ihdr).apply {
                writeInt(width)
                writeInt(height)
                writeByte(bitDepth)
                writeByte(colorType)
                writeByte(0)
                writeByte(0)
                writeByte(0)
            }
            chunk(out, "IHDR", ihdr.toByteArray())
            chunk(out, "IDAT", idat.toByteArray())
            chunk(out, "IEND", ByteArray(0))
        }
    }

    private fun chunk(out: DataOutputStream, type: String, data: ByteArray) {
        val t = type.toByteArray(Charsets.US_ASCII)
        out.writeInt(data.size)
        out.write(t)
        out.write(data)
        val crc = CRC32()
        crc.update(t)
        crc.update(data)
        out.writeInt(crc.value.toInt())
    }
}
