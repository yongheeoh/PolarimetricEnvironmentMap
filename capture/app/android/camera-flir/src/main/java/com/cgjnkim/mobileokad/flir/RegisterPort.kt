package com.cgjnkim.mobileokad.flir

/**
 * The camera's register space, as the rest of this module sees it.
 *
 * [U3vControl] is the real one (GenCP over USB). Pulling the two calls the GenICam layer
 * needs out into an interface lets the configuration logic -- the sequencer program in
 * particular -- be checked on the JVM against a register map built from this camera's own
 * node definitions, without a phone or a camera.
 */
internal interface RegisterPort {
    fun readMemory(address: Long, size: Int): ByteArray
    fun writeU32(address: Long, value: Long)
}
