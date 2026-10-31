package com.cgjnkim.mobileokad.flir

import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbRequest
import android.util.Log
import java.io.Closeable
import java.nio.ByteBuffer

/**
 * Several bulk reads in flight on the stream endpoint at once.
 *
 * ### Why
 *
 * A 5 MP frame of packed 12-bit is 7.5 MB, and `bulkTransfer` silently clamps its length
 * to the kernel's 16 KB usbfs buffer, so a frame is 459 transfers. Issued synchronously
 * they run one at a time: the host asks, waits for the whole transfer to land, then asks
 * again, and the bus sits idle across every one of those 459 round trips. Measured on
 * this phone that comes to 21.8 MB/s -- about half of what USB 2.0 High Speed delivers in
 * practice, and it caps the preview at 2.9 frames a second.
 *
 * Queueing the reads instead lets the next transfer start while the last is still being
 * handed back, which is the whole of the fix. The camera is on a phone that negotiates
 * USB 2.0 and will not be moved off it, so the bus is the budget and this is how to spend
 * all of it rather than half.
 *
 * ### Order
 *
 * Bulk transfers on one endpoint complete in the order they were submitted, which is what
 * makes this usable at all: the leader, each payload chunk and the trailer are separate
 * transfers of exactly negotiated sizes, and a reader that took them out of order would
 * desynchronise the stream permanently. [collect] therefore expects its own oldest
 * request back, and treats anything else as a failure rather than trying to sort it out.
 *
 * Nothing else on this connection may use asynchronous requests while a frame is in
 * flight, for the same reason: `requestWait` is per connection, not per endpoint, and a
 * stray completion would be mistaken for a stream transfer.
 */
internal class StreamPipe(
    private val connection: UsbDeviceConnection,
    endpoint: UsbEndpoint,
    depth: Int,
    bufferSize: Int,
) : Closeable {

    private class Slot(val request: UsbRequest, val buffer: ByteBuffer)

    private val free = ArrayDeque<Slot>()
    private val busy = ArrayDeque<Slot>()
    private var broken = false

    init {
        repeat(depth) {
            val request = UsbRequest()
            if (!request.initialize(connection, endpoint)) {
                request.close()
                closeAll()
                throw U3vException("could not initialise a stream request")
            }
            free.addLast(Slot(request, ByteBuffer.allocateDirect(bufferSize)))
        }
    }

    /** How many more reads can be queued before one has to be collected. */
    val room: Int get() = free.size

    val outstanding: Int get() = busy.size

    /**
     * Queues one read of exactly [size] bytes.
     *
     * @return false when there is no free slot or the kernel refused the submission --
     *   the caller falls back rather than blocking, since a stream that cannot be queued
     *   can still be read synchronously.
     */
    fun submit(size: Int): Boolean {
        if (broken) return false
        val slot = free.removeFirstOrNull() ?: return false
        slot.buffer.clear()
        slot.buffer.limit(size)
        if (!slot.request.queue(slot.buffer)) {
            free.addFirst(slot)
            return false
        }
        busy.addLast(slot)
        return true
    }

    /**
     * Waits for the oldest queued read and copies it into [dst] at [offset].
     *
     * @return bytes transferred, or -1 when nothing was outstanding, the wait timed out,
     *   or a request other than the expected one came back
     */
    fun collect(dst: ByteArray, offset: Int, timeoutMs: Long, expectEmpty: Boolean = false): Int {
        val slot = busy.removeFirstOrNull() ?: return -1
        val done = runCatching { connection.requestWait(timeoutMs) }.getOrNull()
        if (done !== slot.request) {
            // Either the wait timed out or the endpoint handed back something else. The
            // pipe cannot say where the stream is any more, so it stops for good and the
            // caller goes back to synchronous reads.
            //
            // The request has to be cancelled before the slot is let go. A read that
            // timed out is still queued in the kernel, and a slot returned to the pool
            // with a live request behind it means the *next* acquisition's first frame
            // gets eaten by a read belonging to the last one -- which presents as a
            // capture that simply never returns, with nothing in the log.
            broken = true
            runCatching { slot.request.cancel() }
            runCatching { connection.requestWait(CANCEL_WAIT_MS) }
            free.addLast(slot)
            // A drain ends on an empty read by design, so that one is not a fault. Any
            // other is: it means a frame went missing mid-stream.
            if (!expectEmpty) {
                Log.w(TAG, "queued read did not come back; falling back to synchronous reads")
            }
            return -1
        }
        val n = slot.buffer.position()
        val copied = minOf(n, dst.size - offset)
        if (copied > 0) {
            slot.buffer.rewind()
            slot.buffer.get(dst, offset, copied)
        }
        free.addLast(slot)
        return n
    }

    /**
     * Abandons everything in flight.
     *
     * Called when a frame fails part way, so the next one does not collect this frame's
     * leftovers and read them as its own.
     */
    fun cancelAll() {
        while (busy.isNotEmpty()) {
            val slot = busy.removeFirst()
            runCatching { slot.request.cancel() }
            // The cancelled request still completes, so it has to be waited for or it
            // would surface inside the next frame.
            runCatching { connection.requestWait(CANCEL_WAIT_MS) }
            free.addLast(slot)
        }
    }

    /** True once the pipe has seen something it cannot reason about. */
    val failed: Boolean get() = broken

    override fun close() {
        runCatching { cancelAll() }
        closeAll()
    }

    private fun closeAll() {
        for (slot in free) runCatching { slot.request.close() }
        for (slot in busy) runCatching { slot.request.close() }
        free.clear()
        busy.clear()
    }

    companion object {
        private const val TAG = "StreamPipe"
        private const val CANCEL_WAIT_MS = 200L
    }
}
