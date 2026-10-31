package com.cgjnkim.mobileokad.flir

/**
 * The exposure-sequencer bracket, configured exactly as the PC capture does it.
 *
 * Line-by-line port of `hardware/sensor/flir.py` (`_sequencer_segment` / `_restore_streaming_state`),
 * which is verified on this camera: one sequencer set per exposure, chained 0 -> 1 -> ... -> n-1 -> 0
 * and advanced by FrameStart, then a finite MultiFrame acquisition of exactly n frames. No frame is
 * discarded and every frame carries chunk data (FrameID, ExposureTime, SequencerSetActive) that
 * [BracketCapture] checks.
 *
 * Differences from the PC, all forced by the transport:
 *  - TriggerMode sits behind a selector-computed address and cannot be written here. It is Off by
 *    default and the PC never turns it on; if a frame never arrives, that is the first suspect.
 *  - StreamBufferHandlingMode is a Spinnaker host setting; this driver reads frames in order anyway.
 *  - ExposureTime is written to its integer register (µs, identity converter on this camera) and
 *    the read-back is kept, because the camera quantises (125 µs -> 126 µs).
 */
internal class SequencerProgram(private val nodes: NodeAccess) {

    /** Optional settings that did not apply, kept for the capture metadata. */
    val skipped = LinkedHashMap<String, String>()

    /** Number of sequencer sets, i.e. distinct exposures per hardware cycle (0 when unsupported). */
    fun capacity(): Int {
        val max = nodes.readIntOrNull("SequencerSetSelector_Max") ?: return 0
        return (max + 1).toInt()
    }

    /** Chunk mode with the fields the bracket is verified from. Done once per open, like the PC. */
    fun enableChunks() {
        nodes.setBool("ChunkModeActive", true)
        for (selector in CHUNKS) {
            val reason = nodes.optional {
                nodes.setEnum("ChunkSelector", selector)
                nodes.setBool("ChunkEnable", true)
            }
            if (reason != null) skipped["Chunk$selector"] = reason
        }
    }

    /**
     * Programs one set per exposure and arms a finite acquisition of exactly that many frames.
     * Acquisition must be stopped. Returns the exposures the camera actually holds, per set (µs).
     */
    fun program(exposuresUs: List<Long>): List<Long> {
        val n = exposuresUs.size
        require(n >= 1) { "empty bracket" }

        // Sequencer off first: while it is On (left by the previous segment) TriggerMode is read-only.
        nodes.setEnumOrCurrent("SequencerMode", "Off")
        if (nodes.enumEntry("TriggerMode") == null) skipped.putIfAbsent("TriggerMode", "selector-addressed; assumed Off")
        note("AcquisitionFrameRateEnable") { nodes.setBool("AcquisitionFrameRateEnable", false) }

        nodes.setEnum("SequencerConfigurationMode", "On")
        nodes.setEnum("SequencerFeatureSelector", "ExposureTime")
        note("SequencerFeatureEnable") { nodes.setBool("SequencerFeatureEnable", true) }
        val programmed = ArrayList<Long>(n)
        for ((i, us) in exposuresUs.withIndex()) {
            nodes.writeInt("SequencerSetSelector", i.toLong())
            nodes.setEnum("SequencerTriggerSource", "FrameStart")
            note("SequencerPathSelector") { nodes.writeInt("SequencerPathSelector", 0) }
            nodes.writeInt("SequencerSetNext", ((i + 1) % n).toLong())
            programmed += nodes.writeInt("ExposureTime_Val", us, verify = false)
            nodes.execute("SequencerSetSave")
        }
        note("SequencerSetStart") { nodes.writeInt("SequencerSetStart", 0) }
        nodes.setEnum("SequencerConfigurationMode", "Off")
        nodes.setEnum("SequencerMode", "On")

        if (n == 1) {
            nodes.setEnum("AcquisitionMode", "SingleFrame")
        } else {
            nodes.setEnum("AcquisitionMode", "MultiFrame")
            nodes.writeInt("AcquisitionFrameCount", n.toLong())
        }
        return programmed
    }

    /** Back to continuous single-exposure streaming for the preview. Best effort, like the PC. */
    fun restore(exposureUs: Long?) {
        nodes.optional { nodes.setEnum("SequencerMode", "Off") }
        nodes.optional { nodes.setEnum("SequencerConfigurationMode", "Off") }
        nodes.optional { nodes.setEnum("AcquisitionMode", "Continuous") }
        if (exposureUs != null) nodes.optional { nodes.writeInt("ExposureTime_Val", exposureUs, verify = false) }
    }

    private inline fun note(what: String, action: () -> Unit) {
        nodes.optional(action)?.let { skipped[what] = it }
    }

    companion object {
        /** Same list as the PC; the first three are what every frame is checked against. */
        val CHUNKS = listOf("ExposureTime", "FrameID", "Gain", "SequencerSetActive", "Timestamp")
    }
}
