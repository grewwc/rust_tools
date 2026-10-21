package com.grewwc.aserve;

/**
 * Timing for one pending screen capture.
 *
 * <p>The user picks the moment: the request is first armed, then the user
 * triggers the shot. The floating bubble is the normal trigger, and it
 * delivers the frame that is on screen when it is tapped. The notification
 * action stays as a fallback, where the shade covers the content, so that
 * path delivers a frame from just before the tap instead.
 */
final class ScreenCaptureState {
    /** Sentinel meaning "no frame may be taken yet". */
    static final long NEVER = Long.MAX_VALUE;
    /** Delay before the fallback capture, so the notification shade has closed. */
    static final long SETTLE_MS = 1300L;
    /** Delay before the bubble capture, so the hidden bubble leaves the mirror. */
    static final long BUBBLE_SETTLE_MS = 120L;
    /** How long an armed request waits for the user trigger. */
    static final long ARM_TIMEOUT_MS = 115_000L;

    /** Where the delivered frame came from, reported to the caller. */
    static final String SOURCE_BUBBLE = "bubble";
    static final String SOURCE_PRE_TAP = "pre_tap";
    static final String SOURCE_LIVE = "live";

    // The main thread arms and triggers; the frame worker reads. Volatile keeps
    // the worker from missing a trigger it should act on.
    private volatile String token;
    private volatile long captureAfter = NEVER;
    private volatile boolean triggered;
    private volatile long armedAt;
    private volatile String source = SOURCE_LIVE;

    boolean armed() { return token != null; }

    String token() { return token; }

    long captureAfter() { return captureAfter; }

    /** True once the user picked the moment. */
    boolean triggered() { return triggered; }

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
        triggered = false;
        armedAt = now;
    }

    /** Records that the user picked the moment. */
    boolean trigger(long now) {
        if (token == null || triggered) return false;
        triggered = true;
        return true;
    }

    /** Schedules the fallback capture for shortly after the user triggered. */
    void scheduleLive(long now) {
        scheduleLive(now, SETTLE_MS, SOURCE_LIVE);
    }

    /** Schedules a capture for a given delay and records where it comes from. */
    void scheduleLive(long now, long delayMs, String source) {
        captureAfter = now + delayMs;
        this.source = source;
    }

    /** Where the frame that is about to be taken comes from. */
    String source() {
        return source;
    }

    /** True once the frame may be taken. */
    boolean due(long now) {
        return token != null && triggered && captureAfter != NEVER && now >= captureAfter;
    }

    /** True when the request waited too long for the user trigger. */
    boolean expired(long now) { return token != null && now - armedAt >= ARM_TIMEOUT_MS; }

    void clear() {
        token = null;
        captureAfter = NEVER;
        triggered = false;
    }
}
