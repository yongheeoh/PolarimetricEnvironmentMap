package kr.penvmap.capture.core

/**
 * Minimal JSON for the capture files.
 *
 * org.json is an Android-only stub on the JVM (unit tests would get "not mocked"), and the files only
 * hold maps, lists, strings, numbers, booleans and null. The writer mirrors Python's
 * `json.dumps(indent=2)` layout so a phone capture reads like a PC one; the parser covers what the
 * app reads back (calibration JSON, its own settings).
 */
object Json {

    fun write(value: Any?, indent: Int = 2): String = StringBuilder().also { append(it, value, indent, 0) }.toString()

    private fun append(sb: StringBuilder, value: Any?, indent: Int, level: Int) {
        when (value) {
            null -> sb.append("null")
            is String -> quote(sb, value)
            is Boolean -> sb.append(value)
            is Double -> sb.append(number(value))
            is Float -> sb.append(number(value.toDouble()))
            is Number -> sb.append(value.toLong())
            is Map<*, *> -> {
                if (value.isEmpty()) {
                    sb.append("{}")
                    return
                }
                sb.append("{")
                var first = true
                for ((k, v) in value) {
                    if (!first) sb.append(",")
                    first = false
                    newline(sb, indent, level + 1)
                    quote(sb, k.toString())
                    sb.append(": ")
                    append(sb, v, indent, level + 1)
                }
                newline(sb, indent, level)
                sb.append("}")
            }
            is Iterable<*> -> list(sb, value.toList(), indent, level)
            is Array<*> -> list(sb, value.toList(), indent, level)
            is DoubleArray -> list(sb, value.toList(), indent, level)
            is LongArray -> list(sb, value.toList(), indent, level)
            is IntArray -> list(sb, value.toList(), indent, level)
            else -> quote(sb, value.toString())
        }
    }

    private fun list(sb: StringBuilder, items: List<*>, indent: Int, level: Int) {
        if (items.isEmpty()) {
            sb.append("[]")
            return
        }
        sb.append("[")
        for ((i, item) in items.withIndex()) {
            if (i > 0) sb.append(",")
            newline(sb, indent, level + 1)
            append(sb, item, indent, level + 1)
        }
        newline(sb, indent, level)
        sb.append("]")
    }

    private fun newline(sb: StringBuilder, indent: Int, level: Int) {
        if (indent <= 0) return
        sb.append('\n')
        repeat(indent * level) { sb.append(' ') }
    }

    /** Finite doubles as JSON numbers; NaN / infinity have no JSON form and become null. */
    private fun number(d: Double): String {
        if (!d.isFinite()) return "null"
        if (d == Math.rint(d) && kotlin.math.abs(d) < 1e15) return "${d.toLong()}.0"
        return d.toString()
    }

    private fun quote(sb: StringBuilder, s: String) {
        sb.append('"')
        for (c in s) {
            when (c) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                else -> if (c < ' ') sb.append("\\u%04x".format(c.code)) else sb.append(c)
            }
        }
        sb.append('"')
    }

    // ---- parser -------------------------------------------------------------------------------------

    /** Parses JSON into Map<String, Any?> / List<Any?> / String / Double / Long / Boolean / null. */
    fun parse(text: String): Any? {
        val p = Parser(text)
        val v = p.value()
        p.skipWs()
        require(p.i == text.length) { "trailing characters at ${p.i}" }
        return v
    }

    private class Parser(val s: String) {
        var i = 0

        fun skipWs() {
            while (i < s.length && s[i].isWhitespace()) i++
        }

        fun value(): Any? {
            skipWs()
            require(i < s.length) { "unexpected end" }
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> lit("true", true)
                'f' -> lit("false", false)
                'n' -> lit("null", null)
                else -> if (c == '-' || c.isDigit()) num() else throw IllegalArgumentException("unexpected '$c' at $i")
            }
        }

        private fun lit(word: String, v: Any?): Any? {
            require(s.startsWith(word, i)) { "bad literal at $i" }
            i += word.length
            return v
        }

        private fun obj(): Map<String, Any?> {
            val out = LinkedHashMap<String, Any?>()
            i++
            skipWs()
            if (s[i] == '}') {
                i++
                return out
            }
            while (true) {
                skipWs()
                val k = str()
                skipWs()
                require(s[i] == ':') { "expected ':' at $i" }
                i++
                out[k] = value()
                skipWs()
                when (s[i++]) {
                    ',' -> continue
                    '}' -> return out
                    else -> throw IllegalArgumentException("expected ',' or '}' at ${i - 1}")
                }
            }
        }

        private fun arr(): List<Any?> {
            val out = ArrayList<Any?>()
            i++
            skipWs()
            if (s[i] == ']') {
                i++
                return out
            }
            while (true) {
                out += value()
                skipWs()
                when (s[i++]) {
                    ',' -> continue
                    ']' -> return out
                    else -> throw IllegalArgumentException("expected ',' or ']' at ${i - 1}")
                }
            }
        }

        private fun str(): String {
            require(s[i] == '"') { "expected string at $i" }
            i++
            val sb = StringBuilder()
            while (true) {
                val c = s[i++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        when (val e = s[i++]) {
                            '"', '\\', '/' -> sb.append(e)
                            'n' -> sb.append('\n')
                            't' -> sb.append('\t')
                            'r' -> sb.append('\r')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000c')
                            'u' -> {
                                sb.append(s.substring(i, i + 4).toInt(16).toChar())
                                i += 4
                            }
                            else -> throw IllegalArgumentException("bad escape at $i")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun num(): Any {
            val start = i
            if (s[i] == '-') i++
            while (i < s.length && (s[i].isDigit() || s[i] in ".eE+-")) i++
            val t = s.substring(start, i)
            return if (t.any { it in ".eE" }) t.toDouble() else t.toLongOrNull() ?: t.toDouble()
        }
    }
}

/** Typed access helpers for parsed JSON. */
@Suppress("UNCHECKED_CAST")
fun Any?.asMap(): Map<String, Any?> = this as Map<String, Any?>

@Suppress("UNCHECKED_CAST")
fun Any?.asList(): List<Any?> = this as List<Any?>

fun Any?.asDouble(): Double = (this as Number).toDouble()
