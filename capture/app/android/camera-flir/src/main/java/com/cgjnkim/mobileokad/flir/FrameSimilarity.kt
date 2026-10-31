package com.cgjnkim.mobileokad.flir

import kotlin.math.hypot

/**
 * How alike two captures are, for finding the ones that are nearly the same picture.
 *
 * ### The problem this is for
 *
 * A walk through a scene at one capture every few seconds produces runs where the camera
 * barely moved, and those runs are the worst thing that can happen to a dataset split: put
 * two frames of the same shelf in train and validation and the validation score measures
 * memorisation. Time alone cannot find them -- three seconds apart while walking is a
 * different scene, and three seconds apart while standing still is the same one.
 *
 * ### Why this is the drift estimator
 *
 * [FrameDrift] already answers exactly this question for consecutive preview frames: it
 * searches for the offset that best lines one frame up with the last and reports the
 * normalised correlation it achieved. Two frames of the same scene, barely moved, line up
 * well at a small offset. Two frames of different scenes line up badly at every offset.
 *
 * Its search range is bounded, so a large but genuine translation of one scene reads as
 * *unlike* rather than as a big shift. That bias is the safe direction: a run is kept
 * whole rather than thinned on a weak match, and the cost of missing a duplicate is much
 * smaller than the cost of throwing away a frame somebody wanted.
 */
object FrameSimilarity {

    /**
     * @param score normalised correlation at the best offset, 0..1
     * @param shiftPixels how far the picture moved, in the grid's own pixels
     */
    class Match(val score: Float, val shiftPixels: Float) {

        /**
         * Whether these two are the same picture for a dataset's purposes.
         *
         * Both halves are needed. A high score alone is reached by two frames of the same
         * flat wall from opposite ends of a room, which are genuinely different views; a
         * small shift alone is reached by two unlike frames whose best -- and bad --
         * match happened to be near the middle.
         */
        val nearDuplicate: Boolean
            get() = score >= SAME_SCENE && shiftPixels <= STILL_PIXELS
    }

    /**
     * Matches two single-channel grids of the same size.
     *
     * @return null when the grids are too small or the wrong size to compare
     */
    fun between(previous: FloatArray, current: FloatArray, width: Int, height: Int): Match? {
        // The interval only scales the speed FrameDrift reports, which is not used here,
        // so any non-zero value does; one second keeps the numbers readable in a log.
        val drift = FrameDrift.between(previous, current, width, height, 1000f) ?: return null
        return Match(drift.confidence, hypot(drift.dx, drift.dy))
    }

    /**
     * Correlation above which two frames are looking at the same thing.
     *
     * [FrameDrift.MIN_CONFIDENCE] is 0.35 and marks "this match is worth believing at
     * all"; being the same scene is a much stronger claim than having found a real offset,
     * so this sits well above it.
     */
    const val SAME_SCENE = 0.70f

    /**
     * Movement below which the camera did not really go anywhere, in grid pixels.
     *
     * A twentieth of the short side of a Stokes grid. Above this the view has changed
     * enough that the second frame sees something the first did not, which is the thing
     * that makes a second frame worth keeping.
     */
    const val STILL_PIXELS = 12f
}
