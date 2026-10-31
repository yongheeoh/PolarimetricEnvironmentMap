package kr.penvmap.capture.core

/**
 * Exposure statistics and checks, ported from capture/utils.py. All thresholds are fractions of the
 * sensor's full scale, so 8-bit previews and 16-bit brackets share the logic.
 */
object Exposure {
    const val SAT_FRAC = 250.0 / 255.0
    const val DARK_FRAC = 5.0 / 255.0
    /** Allowed saturated fraction at the shortest exposure. */
    const val MAX_OVER_RATIO_AT_TMIN = 0.005
    /** Allowed dark fraction at the longest exposure. */
    const val MAX_UNDER_RATIO_AT_TMAX = 0.20

    data class Stats(val over: Double, val under: Double, val usable: Double)

    /** Raw samples read as unsigned values, from either 8-bit or little-endian 16-bit buffers. */
    fun interface Samples {
        operator fun get(i: Int): Int
    }

    fun u8(data: ByteArray) = Samples { data[it].toInt() and 0xFF }
    fun u16le(data: ByteArray) = Samples { (data[2 * it].toInt() and 0xFF) or ((data[2 * it + 1].toInt() and 0xFF) shl 8) }

    /**
     * (over, under, usable) fractions of [count] samples (`compute_over_under`). [mask], when given,
     * selects the samples that count (true = valid), like the PC's calibration-mask statistics.
     * [stride] > 1 subsamples for speed on the phone preview.
     */
    fun overUnder(samples: Samples, count: Int, fullScale: Double, mask: BooleanArray? = null, stride: Int = 1): Stats {
        val sat = SAT_FRAC * fullScale
        val dark = DARK_FRAC * fullScale
        var n = 0L
        var over = 0L
        var under = 0L
        var usable = 0L
        var i = 0
        while (i < count) {
            if (mask == null || mask[i]) {
                val v = samples[i].toDouble()
                n++
                if (v >= sat) over++
                if (v <= dark) under++
                if (v > dark && v < sat) usable++
            }
            i += stride
        }
        if (n == 0L) return Stats(Double.NaN, Double.NaN, Double.NaN)
        return Stats(over.toDouble() / n, under.toDouble() / n, usable.toDouble() / n)
    }

    data class Check(val ok: Boolean, val tMin: Long, val tMax: Long, val minStats: Stats?, val maxStats: Stats?, val reason: String)

    /** `_judge_bracket`: the shortest exposure must not saturate, the longest must not be too dark. */
    fun judge(tMin: Long, tMax: Long, minStats: Stats?, maxStats: Stats?): Check {
        if (minStats == null || maxStats == null) return Check(false, tMin, tMax, minStats, maxStats, "capture failed")
        val okMin = minStats.over <= MAX_OVER_RATIO_AT_TMIN
        val okMax = maxStats.under <= MAX_UNDER_RATIO_AT_TMAX
        val reason = buildString {
            if (!okMin) append("t_min over too high (%.3f%% > %.3f%%). ".format(minStats.over * 100, MAX_OVER_RATIO_AT_TMIN * 100))
            if (!okMax) append("t_max under too high (%.3f%% > %.3f%%). ".format(maxStats.under * 100, MAX_UNDER_RATIO_AT_TMAX * 100))
        }.trim()
        return Check(okMin && okMax, tMin, tMax, minStats, maxStats, reason)
    }

    /**
     * Candidates for `auto_select_max_exposure`: currentMax * 2^k for k = -scan..scan, unique, sorted,
     * kept inside [max(min, range.first), range.last]. They are tried short to long and the first one
     * whose dark fraction is acceptable wins -- the bracket is no longer than needed.
     */
    fun autoCandidates(minUs: Long, currentMaxUs: Long, scanOctaves: Int = 2, rangeUs: LongRange? = null): List<Long> {
        val lo = maxOf(minUs, rangeUs?.first ?: 0L)
        val hi = rangeUs?.last ?: Long.MAX_VALUE
        return (-scanOctaves..scanOctaves)
            .map { k -> (currentMaxUs * Math.pow(2.0, k.toDouble())).toLong() }
            .toSortedSet()
            .filter { it in lo..hi }
    }

    /** Runs the auto selection with [measureUnder] giving the dark fraction at one exposure (null = failed). */
    fun autoSelect(candidates: List<Long>, measureUnder: (Long) -> Double?): Pair<Long?, List<Pair<Long, Double?>>> {
        val tried = ArrayList<Pair<Long, Double?>>()
        for (c in candidates) {
            val under = measureUnder(c)
            tried += c to under
            if (under != null && under <= MAX_UNDER_RATIO_AT_TMAX) return c to tried
        }
        return null to tried
    }
}
