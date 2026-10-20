package com.grewwc.aserve;

/** Consent is tied to one origin and one bounded, revocable process session. */
final class ScreenSession {
    static final long MAX_DURATION_MS = 30 * 60_000L;
    private final String origin;
    private final long started;
    private boolean stopped;

    ScreenSession(String origin, long started) {
        this.origin = origin;
        this.started = started;
    }

    boolean permits(String candidate, long now) {
        long age = now - started;
        return !stopped && origin != null && origin.equals(candidate)
                && age >= 0 && age < MAX_DURATION_MS;
    }

    void stop() { stopped = true; }
}
