package com.grewwc.aserve;

/**
 * Tracks when the camera preview may start and guarantees that a camera
 * request is answered exactly once.
 *
 * <p>Starting the preview needs two independent things: a window that shows
 * the preview surface, and the camera permission. They arrive in an arbitrary
 * order — the permission dialog can be answered before or after the first
 * resume — so the preview starts as soon as both are present. Tracking that
 * explicitly is what keeps a late permission grant from leaving the preview
 * black: the activity stays open but nothing ever starts the camera.</p>
 *
 * <p>Every exit path (cancel, back press, camera error, watchdog, explicit
 * stop) funnels through {@link #deliver()}, so the assistant can never be
 * left waiting for a frame that is no longer coming.</p>
 */
final class CameraFlow {

    private boolean built;
    private boolean permitted;
    private boolean started;
    private boolean delivered;

    /** The window shows the preview surface and its controls. */
    void setBuilt() {
        built = true;
    }

    /** The camera permission is granted. */
    void setPermitted() {
        permitted = true;
    }

    /**
     * True while the preview may run: both prerequisites are in place and the
     * request has not been answered yet.
     */
    boolean shouldStart() {
        return built && permitted && !delivered;
    }

    /** The preview was started. */
    void markStarted() {
        started = true;
    }

    boolean isStarted() {
        return started;
    }

    /**
     * Records that a result was handed back to the caller.
     *
     * @return true for the first call and false for every later one, so a
     *     cancellation and a watchdog cannot both answer the same request.
     */
    boolean deliver() {
        if (delivered) return false;
        delivered = true;
        return true;
    }

    boolean isDelivered() {
        return delivered;
    }
}
