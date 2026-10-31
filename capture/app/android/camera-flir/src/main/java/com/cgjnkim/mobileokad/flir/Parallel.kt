package com.cgjnkim.mobileokad.flir

import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.max
import kotlin.math.min

/**
 * Splits per-pixel work across the cores.
 *
 * Everything it is used for here writes a disjoint slice of its own output and reads
 * only inputs that were finished before the split, so a slice needs no coordination
 * beyond being waited for. That is what makes the pattern safe to apply so widely:
 * the moment a loop's output feeds its own next iteration, it does not belong here.
 *
 * A phone's cores are not equal -- four fast and four slow is typical -- so the
 * measured gain is nearer three times than eight. That is still the difference between
 * a viewer that opens and one that pauses.
 */
object Parallel {

    private val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

    /**
     * Daemon threads, so a pool that lives as long as the process can never be the
     * reason the process stays alive. One fewer than the cores: the caller runs a
     * slice itself rather than handing out all of them and idling.
     */
    private val pool by lazy {
        Executors.newFixedThreadPool(max(1, cores - 1)) { r ->
            Thread(r, "flir-parallel").apply { isDaemon = true }
        }
    }

    /**
     * Runs [body] over [count] independent units, in slices.
     *
     * Below [minPerSlice] units per slice the handover costs more than the work it
     * moves, so small frames run where they are -- rows and pixels are very different
     * units, so the caller says which it is splitting. [body] must be safe to call from
     * several threads at once over disjoint ranges.
     */
    fun over(count: Int, minPerSlice: Int = MIN_PIXELS_PER_SLICE, body: (from: Int, to: Int) -> Unit) {
        val slices = if (serial.get() == true) 1 else min(cores, max(1, count / minPerSlice))
        if (slices <= 1) {
            body(0, count)
            return
        }
        val step = (count + slices - 1) / slices
        val pending = ArrayList<Future<*>>(slices - 1)
        for (i in 1 until slices) {
            val from = i * step
            if (from >= count) break
            pending += pool.submit { body(from, min(count, from + step)) }
        }
        body(0, min(count, step))
        for (f in pending) {
            try {
                f.get()
            } catch (e: ExecutionException) {
                // The slice's own failure, not a wrapper nobody can read.
                throw e.cause ?: e
            }
        }
    }

    /**
     * The same split for a sum: [slice] totals its own range and the totals are added
     * at the end, because a shared accumulator would need a lock per pixel.
     *
     * The result is not bit-identical to the serial sum -- floating-point addition is
     * not associative -- but these are means over hundreds of thousands of samples used
     * to pick a white balance, where the difference is far below what any of it decides.
     */
    fun sums(count: Int, width: Int, slice: (from: Int, to: Int) -> DoubleArray): DoubleArray {
        val slices = if (serial.get() == true) 1 else min(cores, max(1, count / MIN_PIXELS_PER_SLICE))
        if (slices <= 1) return slice(0, count)

        val step = (count + slices - 1) / slices
        val pending = ArrayList<Future<DoubleArray>>(slices - 1)
        for (i in 1 until slices) {
            val from = i * step
            if (from >= count) break
            pending += pool.submit<DoubleArray> { slice(from, min(count, from + step)) }
        }
        val total = slice(0, min(count, step))
        for (f in pending) {
            val part = try {
                f.get()
            } catch (e: ExecutionException) {
                throw e.cause ?: e
            }
            for (k in 0 until width) total[k] += part[k]
        }
        return total
    }

    private val serial = ThreadLocal<Boolean>()

    /**
     * Runs [body] with every split on this thread disabled.
     *
     * For work nobody is waiting on. Speculative renders -- the neighbouring captures a
     * swipe might land on -- were spreading themselves over every core alongside the one
     * render someone was actually looking at, and the foreground render, sharing what was
     * left, took two to four times as long as it does alone. Background work that keeps
     * to one core finishes later and costs nothing that matters.
     */
    fun <T> serially(body: () -> T): T {
        val previous = serial.get()
        serial.set(true)
        try {
            return body()
        } finally {
            serial.set(previous)
        }
    }

    /** Fewer pixels than this in a slice and the handover costs more than it saves. */
    private const val MIN_PIXELS_PER_SLICE = 8192

    /** The same, for a caller splitting rows: a row is thousands of pixels of work. */
    const val MIN_ROWS_PER_SLICE = 64
}
