package com.grewwc.aserve;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;

import org.json.JSONException;
import org.json.JSONObject;

/** One bounded, foreground-only request. All methods and callbacks run on the UI thread. */
final class ForegroundLocation implements LocationListener {
    interface Result {
        void deliver(boolean ok, String error, JSONObject data);
    }

    private final Activity activity;
    private final Result result;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Runnable timeout = () -> finish(this.best, true, "location timeout");
    private LocationManager manager;
    private Location best;
    private boolean fine;
    private boolean finished;

    ForegroundLocation(Activity activity, Result result) {
        this.activity = activity;
        this.result = result;
    }

    void start() {
        fine = granted(Manifest.permission.ACCESS_FINE_LOCATION);
        if (!fine && !granted(Manifest.permission.ACCESS_COARSE_LOCATION)) {
            finish(null, false, "location permission denied");
            return;
        }
        manager = (LocationManager) activity.getSystemService(Activity.LOCATION_SERVICE);
        if (manager == null) {
            finish(null, false, "location unavailable");
            return;
        }
        for (String provider : new String[]{LocationManager.GPS_PROVIDER,
                LocationManager.NETWORK_PROVIDER, LocationManager.PASSIVE_PROVIDER}) {
            if (!fine && LocationManager.GPS_PROVIDER.equals(provider)) continue;
            try {
                consider(manager.getLastKnownLocation(provider));
            } catch (SecurityException | IllegalArgumentException ignored) {
                // A device may lack a provider or restrict it to precise permission.
            }
        }
        if (best != null && LocationQuality.acceptCached(fine, best.getProvider(), age(best), accuracy(best))) {
            finish(best, false, null);
            return;
        }
        boolean listening = false;
        boolean denied = false;
        for (String provider : new String[]{LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER}) {
            if (!fine && LocationManager.GPS_PROVIDER.equals(provider)) continue;
            try {
                if (!manager.isProviderEnabled(provider)) continue;
                // Register GPS first when precise permission is available, while network
                // updates provide a bounded best-available fallback indoors.
                manager.requestLocationUpdates(provider, 1000L, 0f, this, Looper.getMainLooper());
                listening = true;
            } catch (SecurityException error) {
                denied = true;
            } catch (IllegalArgumentException ignored) {
                // Provider absent on this device.
            }
        }
        if (!listening) {
            finish(null, false, denied ? "location permission denied" : "location services disabled");
            return;
        }
        handler.postDelayed(timeout, LocationQuality.TIMEOUT_MS);
    }

    void cancel() {
        finish(null, false, "location cancelled: app not in foreground");
    }

    @Override
    public void onLocationChanged(Location location) {
        if (finished) return;
        consider(location);
        if (best != null && LocationQuality.targetMet(fine, best.getProvider(), age(best), accuracy(best))) {
            finish(best, false, null);
        }
    }

    private void consider(Location candidate) {
        if (candidate == null || !LocationQuality.validCoordinates(candidate.getLatitude(), candidate.getLongitude())) return;
        long age = age(candidate);
        if (!LocationQuality.usable(age)) return;
        if (best == null || LocationQuality.better(fine, candidate.getProvider(), age, accuracy(candidate),
                best.getProvider(), age(best), accuracy(best))) {
            best = new Location(candidate);
        }
    }

    private void finish(Location location, boolean timedOut, String error) {
        if (finished) return;
        finished = true;
        handler.removeCallbacks(timeout);
        if (manager != null) {
            try {
                manager.removeUpdates(this);
            } catch (SecurityException | IllegalArgumentException ignored) {
                // Revoked permission also ends this request; never leave its timeout active.
            }
        }
        if (location == null || !LocationQuality.usable(age(location))) {
            result.deliver(false, error == null ? "location unavailable" : error, null);
            return;
        }
        if ((fine && !granted(Manifest.permission.ACCESS_FINE_LOCATION))
                || (!fine && !granted(Manifest.permission.ACCESS_COARSE_LOCATION))) {
            result.deliver(false, "location permission changed", null);
            return;
        }
        long age = age(location);
        float accuracy = accuracy(location);
        JSONObject data = new JSONObject();
        try {
            data.put("latitude", location.getLatitude());
            data.put("longitude", location.getLongitude());
            data.put("accuracy", Float.isFinite(accuracy) && accuracy > 0 ? accuracy : JSONObject.NULL);
            data.put("age_ms", age);
            data.put("provider", location.getProvider() == null ? "unknown" : location.getProvider());
            // Precision describes the grant; quality/accuracy describe the reported fix.
            data.put("precision", fine ? "precise" : "approximate");
            data.put("quality", LocationQuality.targetMet(fine, location.getProvider(), age, accuracy)
                    ? "target_met" : "best_available");
            data.put("stale", age > LocationQuality.FRESH_MS);
            data.put("timed_out", timedOut);
        } catch (JSONException invalid) {
            result.deliver(false, "location unavailable", null);
            return;
        }
        result.deliver(true, null, data);
    }

    private boolean granted(String permission) {
        return activity.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED;
    }

    private static long age(Location location) {
        return LocationQuality.ageMs(SystemClock.elapsedRealtimeNanos(), location.getElapsedRealtimeNanos());
    }

    private static float accuracy(Location location) {
        return location.hasAccuracy() ? location.getAccuracy() : Float.NaN;
    }

    @Override
    public void onStatusChanged(String provider, int status, Bundle extras) {}

    @Override
    public void onProviderEnabled(String provider) {}

    @Override
    public void onProviderDisabled(String provider) {}
}
