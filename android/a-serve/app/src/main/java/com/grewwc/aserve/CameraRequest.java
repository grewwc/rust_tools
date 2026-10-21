package com.grewwc.aserve;

/**
 * Tracks whether a camera preview in this process still has a request waiting
 * for it.
 *
 * <p>The system restores a task after the process was killed. A preview that
 * comes back in a fresh process has no request to answer, so it closes itself
 * instead of leaving the user on a camera screen the assistant is not waiting
 * for. The state lives in the process, which is what makes the difference
 * between a restored preview and a live one.</p>
 */
final class CameraRequest {
    private static volatile boolean live;

    private CameraRequest() {
    }

    /** Records that a request is waiting for a preview frame. */
    static void markLive(boolean value) {
        live = value;
    }

    /** True while a request in this process waits for a frame. */
    static boolean isLive() {
        return live;
    }
}
