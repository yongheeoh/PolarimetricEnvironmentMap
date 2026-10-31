package kr.penvmap.capture.core

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow

/**
 * Bracket policy, ported from capture/utils.py (`make_bracket`, `bracket_step_ev`) and the
 * `Bracket` dataclass in capture/capture.py: [numBrackets] exposures from [minUs] to [maxUs], evenly
 * spaced in EV; the shortest stays fixed and the preview / [ ] keys move the longest.
 */
object BracketPolicy {
    /** One frame covers ~4.3 EV (5-95 % of full scale); wider steps leave gaps. */
    const val MAX_BRACKET_STEP_EV = 4.0

    fun stepEv(minUs: Double, maxUs: Double, n: Int): Double =
        if (n <= 1) 0.0 else ln(maxUs / minUs) / ln(2.0) / (n - 1)

    /**
     * The exposures in µs. Rounds half to even like Python's round(). Throws when the bracket is
     * malformed or leaves [rangeUs] (rejected, not clipped, so the bracket never silently loses range).
     */
    fun make(minUs: Long, maxUs: Long, n: Int, rangeUs: LongRange? = null): List<Long> {
        require(n >= 1) { "num_brackets must be >= 1" }
        require(minUs in 1..maxUs) { "need 0 < bracket_min_us <= bracket_max_us, got $minUs, $maxUs" }
        val list = if (n == 1) {
            listOf(maxUs)
        } else {
            val ratio = maxUs.toDouble() / minUs.toDouble()
            (0 until n).map { k -> max(1.0, Math.rint(minUs * ratio.pow(k.toDouble() / (n - 1)))).toLong() }
        }
        if (rangeUs != null) {
            val bad = list.filter { it !in rangeUs }
            require(bad.isEmpty()) { "bracket $list us leaves the camera range [${rangeUs.first}, ${rangeUs.last}] us: $bad" }
        }
        return list
    }
}

/** Current bracket and the exposure shown in the preview (capture/capture.py `Bracket`). */
class BracketState(minUs: Long, maxUs: Long, numBrackets: Int, val rangeUs: LongRange?) {
    var minUs: Long = minUs
        private set
    var maxUs: Long = maxUs
        private set
    var numBrackets: Int = numBrackets
        private set
    var exposures: List<Long> = BracketPolicy.make(minUs, maxUs, numBrackets, rangeUs)
        private set
    var previewIndex: Int = exposures.size / 2
        private set

    val stepEv: Double get() = BracketPolicy.stepEv(minUs.toDouble(), maxUs.toDouble(), numBrackets)
    val previewUs: Long get() = exposures[previewIndex]
    val midUs: Long get() = exposures[exposures.size / 2]
    val exposureSeconds: List<Double> get() = exposures.map { it / 1e6 }

    /** Returns false (unchanged) and the reason when the new bracket is not valid. */
    fun set(minUs: Long = this.minUs, maxUs: Long = this.maxUs, numBrackets: Int = this.numBrackets): String? {
        if (numBrackets > 1 && maxUs <= minUs) return "longest exposure must stay above the shortest ($minUs us)"
        val list = try {
            BracketPolicy.make(minUs, maxUs, numBrackets, rangeUs)
        } catch (e: IllegalArgumentException) {
            return e.message
        }
        this.minUs = minUs
        this.maxUs = maxUs
        this.numBrackets = numBrackets
        exposures = list
        previewIndex = minOf(previewIndex, list.size - 1)
        return null
    }

    fun setMax(maxUs: Long): String? = set(maxUs = maxUs)

    fun previewUp(): Boolean = (previewIndex < exposures.size - 1).also { if (it) previewIndex++ }
    fun previewDown(): Boolean = (previewIndex > 0).also { if (it) previewIndex-- }
}
