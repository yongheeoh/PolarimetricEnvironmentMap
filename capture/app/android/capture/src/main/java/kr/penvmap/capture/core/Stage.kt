package kr.penvmap.capture.core

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLConnection

/** The rotation stage: absolute yaw / pitch moves (deg). */
interface Stage {
    /** One quick request; true when the stage answers. */
    fun ping(): Boolean

    /** Moves and waits until the move is done plus the settle time ([large] = pitch / big moves). */
    fun moveAbs(yawDeg: Double?, pitchDeg: Double?, large: Boolean)

    /** A small aiming move (sun mode): like [moveAbs] but without the settle wait (nothing is captured right after). */
    fun jogAbs(yawDeg: Double?, pitchDeg: Double?) = moveAbs(yawDeg, pitchDeg, large = false)

    /** The stage's own status reply (raw), for the record; null when there is none. */
    fun status(): String? = null
}

/**
 * The PC stage protocol (hardware/stage/mecha.py): `GET {url}/i?ad1=<yaw>&ad2=<pitch>`; the reply
 * comes when the move is done, then the caller's settle time. [open] lets the Android layer bind the
 * request to the stage's Wi-Fi network (a Wi-Fi without internet is otherwise bypassed for mobile data).
 */
class HttpStage(
    private val baseUrl: String,
    private val settleLargeS: Double = 0.2,
    private val settleSmallS: Double = 0.2,
    private val open: (URL) -> URLConnection = { it.openConnection() },
    private val log: (String) -> Unit = {},
) : Stage {

    override fun ping(): Boolean = try {
        request("", connectMs = 2000, readMs = 2000)
        true
    } catch (e: IOException) {
        log("stage not reachable: ${e.javaClass.simpleName}: ${e.message}")
        false
    }

    override fun status(): String? = try {
        request("", connectMs = 2000, readMs = 2000).trim()
    } catch (e: IOException) {
        null
    }

    override fun jogAbs(yawDeg: Double?, pitchDeg: Double?) {
        send(yawDeg, pitchDeg)
    }

    override fun moveAbs(yawDeg: Double?, pitchDeg: Double?, large: Boolean) {
        if (!send(yawDeg, pitchDeg)) return
        Thread.sleep(((if (large) settleLargeS else settleSmallS) * 1000).toLong())
    }

    /** One move request; the reply comes when the move is done. False when there was nothing to move. */
    private fun send(yawDeg: Double?, pitchDeg: Double?): Boolean {
        val params = buildList {
            if (yawDeg != null) add("ad1=${fmt(yawDeg)}")
            if (pitchDeg != null) add("ad2=${fmt(pitchDeg)}")
        }
        if (params.isEmpty()) return false
        // Same tolerance as the PC: a timeout or refused connection is logged and the capture goes on.
        try {
            request(params.joinToString("&"), connectMs = 3000, readMs = 30_000)
        } catch (e: IOException) {
            log("stage move ${params.joinToString("&")} failed: ${e.message}")
        }
        return true
    }

    private fun request(query: String, connectMs: Int, readMs: Int): String {
        val url = URL(baseUrl.trimEnd('/') + "/i" + if (query.isNotEmpty()) "?$query" else "")
        val conn = open(url) as HttpURLConnection
        conn.connectTimeout = connectMs
        conn.readTimeout = readMs
        conn.useCaches = false
        try {
            val code = conn.responseCode
            val body = (if (code < 400) conn.inputStream else conn.errorStream)?.bufferedReader()?.use { it.readText() } ?: ""
            log("/i?$query -> $code ${body.trim()}")
            return body
        } finally {
            conn.disconnect()
        }
    }

    /** Python's str() of the PC's int / float degrees: 15 -> "15", 15.5 -> "15.5". */
    private fun fmt(d: Double): String = if (d == Math.rint(d)) d.toLong().toString() else d.toString()
}

/** Stand-in stage for testing without hardware: records moves and waits like the real one. */
class FakeStage(private val moveS: Double = 0.3, private val settleS: Double = 0.2) : Stage {
    val moves = ArrayList<Triple<Double?, Double?, Boolean>>()
    override fun ping() = true
    override fun moveAbs(yawDeg: Double?, pitchDeg: Double?, large: Boolean) {
        moves += Triple(yawDeg, pitchDeg, large)
        Thread.sleep(((moveS + settleS) * 1000).toLong())
    }
}
