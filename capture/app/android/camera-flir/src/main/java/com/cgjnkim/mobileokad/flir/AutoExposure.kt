package com.cgjnkim.mobileokad.flir

import kotlin.math.log2
import kotlin.math.pow
import kotlin.math.roundToLong

/**
 * The exposure the scene wants, decided by the host rather than by the camera.
 *
 * ### Why the host has to do this at all
 *
 * This camera does not offer `ExposureAuto.Continuous` in the polarized pixel formats --
 * it answers that the entry is unavailable, which is why [FlirDevice.setExposureAuto]
 * returns a refusal to show rather than a success. The AUTO control was therefore a
 * button that reported an error and changed nothing. Metering here is what makes it a
 * control.
 *
 * ### Which policy runs, and why it is not the one this file was written for
 *
 * [POLICY] selects it, and it is [Policy.MEAN] -- the same metering This Polar Camera
 * runs by default, with the same target, deadband, damping and clamp. The two rigs are
 * meant to produce comparable captures, and a scene metered to a mid-grey on one and to
 * a protected highlight on the other is not comparable in the way that matters.
 *
 * The argument below for highlight metering is still true. It was not overturned; it was
 * outranked by parity. [Policy.HIGHLIGHT] is kept whole, law and reasoning both, and
 * changing one constant brings it back -- which is worth knowing, because on this
 * archive MEAN will clip more: 55 percent of a 177-frame sample already had something at
 * full scale, and the frames it is about to clip are the specular reflections these
 * scenes are usually *of*.
 *
 * ### Why metering highlights was the original answer
 *
 * An ordinary camera meters the mean because a slightly blown highlight costs a little
 * detail in one corner. Here it costs the measurement. S1 and S2 are differences between
 * neighbouring analyser readings, so a cell that clipped did not merely lose its own
 * value -- it made the difference it takes part in wrong, and DoLP and AoLP over that
 * whole super-pixel become numbers with nothing behind them. Worse, clipping is silent:
 * a saturated polarized highlight reads as *low* polarization, because all four analyser
 * angles pin to full scale and their differences collapse to zero. Averaging would let
 * that happen in every scene with a specular reflection off water, glass or paint, which
 * is most of the scenes this rig exists to photograph.
 *
 * So the ceiling is the target, and the mean is left to fall where it falls. An
 * underexposed frame is recoverable -- the samples are still proportional to the light,
 * only noisier, and [StokesNoise] already says what that costs. A clipped one is not
 * recoverable by anything.
 *
 * ### Why it is asymmetric
 *
 * That same argument makes the two directions different problems, so they get different
 * gains. Darkening is a correction to damage in progress and moves at up to
 * [DARKEN_MAX_EV] a step. Brightening is an optimisation and moves at a quarter of the
 * error, capped at [BRIGHTEN_MAX_EV], because the alternative is hunting: a specular
 * glint that comes and goes as the rig moves would otherwise drive the exposure up and
 * down through a full stop every second, and every one of those frames is taken at an
 * exposure chosen for a scene that has already changed.
 *
 * ### Why a single hot pixel cannot drive it
 *
 * The peak alone is a bad statistic -- one stuck pixel at full scale would hold the whole
 * frame a stop under for the life of the camera. Darkening therefore requires that a
 * real fraction of the frame ([CLIP_TOLERANCE]) is at or past saturation, not merely
 * that something is. Brightening uses the peak directly, because there the peak is the
 * thing being protected and being cautious about it is the safe error.
 */
object AutoExposure {

    /** What one frame said about the light, from [MosaicPreview.meter]. */
    data class Reading(
        /** Brightest single sample as a fraction of full scale. */
        val peak: Float,
        /** Fraction of sampled pixels at or past [MosaicPreview.SATURATION]. */
        val clipped: Float,
        /**
         * Mean of every sampled pixel as a fraction of full scale.
         *
         * Linear sensor codes, not a perceptual average: This Polar Camera's
         * `normalized_frame_mean` is `cv2.mean(data) / maximum` over the raw frame, and
         * metering the same number is the point of measuring it this way.
         */
        val mean: Float,
    )

    /** Which law decides the step. */
    enum class Policy { MEAN, HIGHLIGHT }

    /**
     * The one that runs.
     *
     * [Policy.MEAN] for parity with the reference implementation. See the class comment
     * for what that costs and what the alternative protects.
     */
    val POLICY = Policy.MEAN

    /** The axes the meter is allowed to move, and how far. */
    data class Limits(
        /**
         * Already narrowed to what AUTO may use, which is not the whole hardware range:
         * the ceiling is a handheld shake limit, not the sensor's 30 s.
         */
        val exposureUs: LongRange,
        /** Null when the camera's gain register could not be verified; then gain is fixed. */
        val gainDb: ClosedFloatingPointRange<Double>?,
    )

    data class Settings(val exposureUs: Long, val gainDb: Double)

    /**
     * One correction, or null when the frame is already in band and nothing should move.
     *
     * Null is the common answer and is meant to be: a meter that writes a register on
     * every frame would spend the control channel it shares with the stream, and would
     * also make every capture's exposure a slightly different number for no reason.
     */
    fun step(
        current: Settings,
        reading: Reading,
        limits: Limits,
        compensationEv: Double = 0.0,
    ): Settings? {
        if (limits.exposureUs.isEmpty()) return null
        val stepEv = when (POLICY) {
            Policy.MEAN -> meanStepEv(reading, compensationEv)
            Policy.HIGHLIGHT -> highlightStepEv(reading)
        } ?: return null
        if (kotlin.math.abs(stepEv) < MIN_STEP_EV) return null
        val next = allocate(current, limits, stepEv)
        return if (next == current) null else next
    }

    /**
     * How far off a mid-grey the frame is, in stops, damped and clamped.
     *
     * This Polar Camera's `_mean_auto_exposure_step`, constant for constant. Symmetric --
     * no hysteresis, no reversal confirmation, no asymmetry between brightening and
     * darkening -- because on this policy neither direction is more urgent than the
     * other. That is exactly what the highlight law disagreed with, and the disagreement
     * is the reason both are in this file.
     *
     * [compensationEv] moves the target rather than the result, which is what makes it a
     * compensation and not a brightness slider: the loop still closes, it just closes
     * somewhere else. It is added to the error exactly as `calculate_auto_exposure` adds
     * it, so plus one stop asks for a frame one stop brighter than the meter would
     * otherwise have chosen -- the way a white wall is photographed without the meter
     * turning it grey.
     */
    internal fun meanStepEv(reading: Reading, compensationEv: Double = 0.0): Double? {
        val mean = reading.mean
        // A frame this could not be measured from says nothing, and guessing from
        // nothing is how an auto exposure runs away in the dark.
        if (!mean.isFinite() || mean < 0f) return null
        // Floored rather than refused: a frame genuinely at zero is a lens cap or a dark
        // room, and the right answer there is to open up at the full rate, not to freeze.
        if (!compensationEv.isFinite()) return null
        val measured = mean.toDouble().coerceAtLeast(MEAN_FLOOR)
        val errorEv = log2(MEAN_TARGET / measured) + compensationEv
        if (kotlin.math.abs(errorEv) <= MEAN_DEADBAND_EV) return 0.0
        return (MEAN_DAMPING * errorEv).coerceIn(-MEAN_MAX_STEP_EV, MEAN_MAX_STEP_EV)
    }

    /** The highlight law, kept whole. See the class comment. */
    internal fun highlightStepEv(reading: Reading): Double? {
        val peak = reading.peak
        if (!peak.isFinite() || peak <= 0f) return null

        val errorEv = when {
            // Really clipping. How far over is measured from the area, not the peak --
            // see [darkenEv].
            reading.clipped > CLIP_TOLERANCE -> -darkenEv(reading.clipped)
            // Below the target with nothing clipped: there is light being left on the
            // table, so take it, slowly.
            peak < TARGET_PEAK -> BRIGHTEN_DAMPING * log2(TARGET_PEAK / peak.toDouble())
            // Between the target and saturation with no real clipping. This is the band
            // the meter is trying to sit in, and moving inside it is pure hunting.
            else -> 0.0
        }
        return errorEv.coerceIn(-DARKEN_MAX_EV, BRIGHTEN_MAX_EV)
    }

    /**
     * How far to pull a clipping frame down, from how much of it is clipped.
     *
     * The peak cannot answer this. It is the one measurement that stops working exactly
     * when it is needed: a frame a fifth of a stop over and a frame four stops over both
     * read a peak of 1.0, because full scale is where the sensor stops counting. Sizing
     * the correction from `log2(target / peak)` therefore asks a saturated frame how
     * saturated it is and believes the answer, which is 0.15 stops however bad it is --
     * so a badly blown scene would crawl down a sixth of a stop at a time while every
     * frame in the meantime is unusable.
     *
     * The area does still carry the information. Real scenes have shoulders rather than
     * cliffs in their histograms, so the fraction of the frame at full scale grows
     * monotonically with how far over the exposure is. That is not enough to invert --
     * the rate depends on the scene -- but it is enough to rank, which is all a
     * proportional controller needs. This ramps logarithmically from [DARKEN_MIN_EV]
     * just past the tolerance to [DARKEN_MAX_EV] once [CLIP_SEVERE] of the frame has
     * gone, and lets the loop close the rest.
     *
     * Overshooting into underexposure is the acceptable error. The next frame measures
     * it and brightens back, and every frame along that path is one whose polarization
     * can still be recovered.
     */
    private fun darkenEv(clipped: Float): Double {
        val severity = log2(clipped.toDouble() / CLIP_TOLERANCE) /
            log2(CLIP_SEVERE.toDouble() / CLIP_TOLERANCE)
        return DARKEN_MIN_EV + (DARKEN_MAX_EV - DARKEN_MIN_EV) * severity.coerceIn(0.0, 1.0)
    }

    /**
     * Spends an EV correction on exposure first and on gain only with what is left.
     *
     * Gain is not a second shutter. It multiplies the read noise along with the signal,
     * and S1 and S2 are differences between analyser readings of similar size, so that
     * noise lands undiluted on the polarization -- the same argument [ExposureStops]
     * makes for putting the two dials side by side. Exposure is therefore taken to its
     * ceiling before a single dB is added, and when darkening the gain comes off first.
     *
     * The arithmetic is one line once both axes are expressed as stops of light: gain in
     * dB above its minimum is [GAIN_DB_PER_EV] per stop, so an exposure time and a gain
     * together are a single equivalent exposure, and allocating is clamping that number
     * into the exposure's range and handing the remainder to gain.
     */
    private fun allocate(current: Settings, limits: Limits, stepEv: Double): Settings {
        val gainMinDb = limits.gainDb?.start ?: current.gainDb
        val gainMaxDb = limits.gainDb?.endInclusive ?: current.gainDb
        val gainOffsetEv = (current.gainDb - gainMinDb) / GAIN_DB_PER_EV

        val minEv = log2(limits.exposureUs.first.toDouble())
        val maxEv = log2(limits.exposureUs.last.toDouble())
        val desiredEv = log2(current.exposureUs.coerceAtLeast(1L).toDouble()) + gainOffsetEv + stepEv

        val exposureEv = desiredEv.coerceIn(minEv, maxEv)
        val exposureUs = 2.0.pow(exposureEv).roundToLong().coerceIn(limits.exposureUs)

        // Whatever the shutter could not hold, in stops, and never negative: below the
        // shortest exposure the camera offers there is nothing gain can do to help.
        val gainEv = (desiredEv - exposureEv).coerceAtLeast(0.0)
        val gainDb = (gainMinDb + gainEv * GAIN_DB_PER_EV).coerceIn(gainMinDb, gainMaxDb)
        return Settings(exposureUs, gainDb)
    }

    /**
     * Where the mean is aimed, as a fraction of full scale.
     *
     * A half. This Polar Camera's `AUTO_EXPOSURE_TARGET`, and note that it is half of
     * *linear* full scale rather than a perceptual mid-grey, which is a brighter place to
     * sit than the 18 percent a photographic meter would choose.
     */
    const val MEAN_TARGET = 0.5

    /** A tenth of a stop. Inside this the frame is close enough and nothing is written. */
    const val MEAN_DEADBAND_EV = 0.1

    /** Half the error per frame, so the loop converges without overshooting into a hunt. */
    const val MEAN_DAMPING = 0.5

    /** Half a stop, in either direction. */
    const val MEAN_MAX_STEP_EV = 0.5

    /**
     * The darkest mean that is treated as a measurement rather than as zero.
     *
     * One part in sixty-five thousand, below any real frame's mean. It exists so a black
     * frame produces a large finite error instead of an infinite one.
     */
    private const val MEAN_FLOOR = 1.0 / 65536.0

    /** Where the peak is aimed under [Policy.HIGHLIGHT]: just under clipping. */
    const val TARGET_PEAK = 0.9f

    /**
     * How much of the frame has to be saturated before the exposure is pulled down.
     *
     * Two samples in ten thousand. Below that it is a stuck pixel or a light source in
     * the frame -- neither is a reason to underexpose everything else, and a small
     * specular that clips is often what a polarization photograph is *of*.
     */
    const val CLIP_TOLERANCE = 2e-4f

    /**
     * Clipping over this much of the frame is as bad as this meter distinguishes.
     *
     * A twentieth. Past here the scene is not a highlight in a photograph, it is an
     * overexposure, and the correction is already at its cap.
     */
    private const val CLIP_SEVERE = 0.05f

    /** The smallest useful correction for a frame that has only just started clipping. */
    private const val DARKEN_MIN_EV = 0.25

    /** A full stop, because clipping is destroying data now. */
    private const val DARKEN_MAX_EV = 1.0

    /** A quarter stop, because underexposure is only costing noise. */
    private const val BRIGHTEN_MAX_EV = 0.25

    private const val BRIGHTEN_DAMPING = 0.25

    /** Smaller than this is not worth a register write or a changed capture. */
    private const val MIN_STEP_EV = 0.05

    /** 20*log10(2): the dB that doubles the signal. */
    const val GAIN_DB_PER_EV = 6.020599913279624
}
