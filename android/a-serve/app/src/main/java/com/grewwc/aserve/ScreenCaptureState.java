package com.grewwc.aserve;

/**
 * Timing for one pending screen capture.
 *
 * <p>The user picks the moment: the request is first armed, then the user
 * triggers the shot from the notification. A short settle delay follows the
 * trigger so the notification shade is gone before the frame is taken.
 */
final class ScreenCaptureState {
    /** Sentinel meaning "no frame may be taken yet". */
    static final long NEVER = Long.MAX_VALUE;
    /** Delay after the user triggers, so the notification shade has closed. */
    static final long SETTLE_MS = 1300L;
    /** How long an armed request waits for the user trigger. */
    static final long ARM_TIMEOUT_MS = 115_000L;

    // The main thread arms and triggers; the frame worker reads. Volatile keeps
    // the worker from missing a trigger it should act on.
    private volatile String token;
    private volatile long captureAfter = NEVER;
    private volatile long armedAt;

    boolean armed() { return token != null; }

    String token() { return token; }

    long captureAfter() { return captureAfter; }

    /**
     * Arms a request. No frame is taken until the user triggers it.
     *
     * @throws IllegalStateException when another request is already pending
     */
    void arm(String requestToken, long now) {
        if (token != null && !token.equals(requestToken))
            throw new IllegalStateException("screen capture busy");
        token = requestToken;
        captureAfter = NEVER;
        armedAt = now;
    }

    /** Schedules the frame for shortly after the user triggered. */
    boolean trigger(long now) {
        if (token == null) return false;
        captureAfter = now + SETTLE_MS;
        return true;
    }

    /** True once the frame may be taken. */
    boolean due(long now) { return token != null && now >= captureAfter; }

    /** True when the request waited too long for the user trigger. */
    boolean expired(long now) { return token != null && now - armedAt >= ARM_TIMEOUT_MS; }

    void clear() {
        token = null;
        captureAfter = NEVER;
    }
}
