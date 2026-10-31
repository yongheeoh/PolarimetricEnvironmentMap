package kr.penvmap.capture

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Looper
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Where the session was taken, for the dataset's place / time metadata. Optional: without the location
 * permission nothing is recorded and capture is unaffected.
 */
class LocationKeeper(private val context: Context) {

    @Volatile private var latest: Location? = null

    private val manager: LocationManager? get() = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager

    fun permitted(): Boolean =
        context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            context.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

    /** Asks the GPS / network providers for one fresh fix (answers arrive on the main looper). */
    @SuppressLint("MissingPermission")
    fun refresh() {
        if (!permitted()) return
        val m = manager ?: return
        for (provider in listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)) {
            if (!runCatching { m.isProviderEnabled(provider) }.getOrDefault(false)) continue
            runCatching {
                @Suppress("DEPRECATION")
                m.requestSingleUpdate(provider, object : LocationListener {
                    override fun onLocationChanged(location: Location) = offer(location)
                    @Deprecated("Deprecated in Java")
                    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
                    override fun onProviderEnabled(provider: String) {}
                    override fun onProviderDisabled(provider: String) {}
                }, Looper.getMainLooper())
            }
        }
    }

    private fun offer(l: Location) {
        val cur = latest
        if (cur == null || l.time > cur.time || l.accuracy < cur.accuracy) latest = l
    }

    /** Best known fix as a JSON-ready map, or null. */
    @SuppressLint("MissingPermission")
    fun snapshot(): Map<String, Any?>? {
        if (!permitted()) return null
        val m = manager
        val known = listOfNotNull(
            latest,
            runCatching { m?.getLastKnownLocation(LocationManager.GPS_PROVIDER) }.getOrNull(),
            runCatching { m?.getLastKnownLocation(LocationManager.NETWORK_PROVIDER) }.getOrNull(),
        ).maxByOrNull { it.time } ?: return null
        return linkedMapOf(
            "latitude" to known.latitude,
            "longitude" to known.longitude,
            "altitude_m" to (if (known.hasAltitude()) known.altitude else null),
            "accuracy_m" to (if (known.hasAccuracy()) known.accuracy.toDouble() else null),
            "provider" to known.provider,
            "fix_time" to SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date(known.time)),
        )
    }
}
