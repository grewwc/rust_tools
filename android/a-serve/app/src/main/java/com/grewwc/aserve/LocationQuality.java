package com.grewwc.aserve;

/** Explicit freshness/accuracy targets, not a guarantee of real-world position accuracy. */
final class LocationQuality {
    static final long FRESH_MS = 30_000L;
    static final long MAX_FALLBACK_MS = 120_000L;
    static final long TIMEOUT_MS = 15_000L;
    static final float PRECISE_METERS = 50f;
    static final float APPROXIMATE_METERS = 5000f;

    private LocationQuality() {}

    static long ageMs(long nowNanos, long fixNanos) {
        if (fixNanos <= 0 || fixNanos > nowNanos) return Long.MAX_VALUE;
        return (nowNanos - fixNanos) / 1_000_000L;
    }

    static boolean validCoordinates(double latitude, double longitude) {
        return Double.isFinite(latitude) && Double.isFinite(longitude)
                && latitude >= -90 && latitude <= 90 && longitude >= -180 && longitude <= 180;
    }

    static boolean usable(long ageMs) {
        return ageMs >= 0 && ageMs <= MAX_FALLBACK_MS;
    }

    static boolean targetMet(boolean fine, String provider, long ageMs, float accuracy) {
        if (ageMs < 0 || ageMs > FRESH_MS || !Float.isFinite(accuracy) || accuracy <= 0) return false;
        return fine ? "gps".equals(provider) && accuracy <= PRECISE_METERS
                : accuracy <= APPROXIMATE_METERS;
    }

    static boolean acceptCached(boolean fine, String provider, long ageMs, float accuracy) {
        // An approximate cached fix must not suppress a new foreground measurement.
        return fine && targetMet(true, provider, ageMs, accuracy);
    }

    static boolean better(boolean fine, String provider, long ageMs, float accuracy,
            String oldProvider, long oldAgeMs, float oldAccuracy) {
        if (!usable(ageMs)) return false;
        if (!usable(oldAgeMs)) return true;
        boolean fresh = ageMs <= FRESH_MS, oldFresh = oldAgeMs <= FRESH_MS;
        if (fresh != oldFresh) return fresh;
        boolean target = targetMet(fine, provider, ageMs, accuracy);
        boolean oldTarget = targetMet(fine, oldProvider, oldAgeMs, oldAccuracy);
        if (target != oldTarget) return target;
        float score = accuracyScore(accuracy), oldScore = accuracyScore(oldAccuracy);
        return score < oldScore || (score == oldScore && ageMs < oldAgeMs);
    }

    private static float accuracyScore(float accuracy) {
        return Float.isFinite(accuracy) && accuracy > 0 ? accuracy : Float.POSITIVE_INFINITY;
    }
}
