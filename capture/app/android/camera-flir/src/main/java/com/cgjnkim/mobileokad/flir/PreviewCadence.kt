package com.cgjnkim.mobileokad.flir

/**
 * How fast the camera is allowed to produce preview frames, from what the bus is doing.
 *
 * ### Why this is measured and not set
 *
 * Acquisition is free-running. A camera permitted to produce faster than the host can
 * collect does not make the preview faster -- it queues the surplus inside itself and
 * hands it back in order, so what reaches the screen is the oldest frame still waiting.
 * The cap therefore has to sit just under the delivery rate, and the delivery rate is a
 * property of the link, of what else is on it, and of how fast the host happens to be
 * reading today.
 *
 * It used to be a constant, measured once by hand: 41 MB/s, "and whenever the host gets
 * faster this number has to change with it". It had already moved twice for that reason,
 * and it is optimistic whenever the OAK is streaming beside this camera and pessimistic
 * whenever it is not. A number that has to be re-measured by a person is a number that is
 * wrong between measurements.
 *
 * ### What is actually measured
 *
 * The time a frame's payload spends crossing the bus, over its bytes. Not the interval
 * between frames, which is the thing the cap itself sets: measuring that and feeding it
 * back would mean a preview that can only ever slow down, since every cap becomes its own
 * evidence. Inside a payload the camera is sending flat out, so the rate there is the
 * link's, whatever the cap outside it says.
 */
object PreviewCadence {

    /**
     * What the cap is allowed to claim of the measured rate.
     *
     * Under one on purpose. A cap slightly under what the host drains leaves the reader
     * waiting a few milliseconds per frame, which costs nothing; a cap slightly over grows
     * a queue that costs a frame of latency for every frame it holds. The asymmetry is
     * the whole reason for a margin.
     */
    const val SAFETY = 0.92

    /** Frames measured before the cap is moved off its starting value. */
    const val MIN_SAMPLES = 8

    /**
     * How far the new cap must be from the old one to be worth writing.
     *
     * Every change is a control transaction on the channel the capture path shares, and
     * the rate wanders by a few percent from frame to frame. Ten percent is wide enough
     * that noise does not write and narrow enough that the OAK starting or stopping does.
     */
    const val HYSTERESIS = 0.10

    /**
     * The rate to allow, given a measured throughput and what a frame costs.
     *
     * @param bytesPerSecond the measured payload throughput
     * @param payloadBytes the bytes one frame carries at the current size
     */
    fun capFor(
        bytesPerSecond: Double,
        payloadBytes: Long,
        ceiling: Double,
        floor: Double,
    ): Double {
        if (!bytesPerSecond.isFinite() || bytesPerSecond <= 0.0 || payloadBytes <= 0L) return ceiling
        return (bytesPerSecond * SAFETY / payloadBytes).coerceIn(floor, ceiling)
    }

    /** Whether [next] is far enough from [current] to be worth a write. */
    fun worthWriting(current: Double, next: Double): Boolean {
        if (!next.isFinite() || next <= 0.0) return false
        if (current <= 0.0) return true
        return kotlin.math.abs(next - current) / current > HYSTERESIS
    }

    /**
     * A throughput from a window of frames, at the median.
     *
     * The median rather than the mean: a single frame that arrived while the phone was
     * doing something else reads as a very slow one, and a mean would let that one frame
     * halve the preview's rate for the next window. The middle of a window is what the
     * link is doing.
     */
    fun throughput(samples: List<Double>): Double? {
        val usable = samples.filter { it.isFinite() && it > 0.0 }.sorted()
        if (usable.size < MIN_SAMPLES) return null
        return usable[usable.size / 2]
    }
}
