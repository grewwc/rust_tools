package com.grewwc.aserve;

/**
 * Decides whether a touch on the floating capture bubble was a tap or a drag.
 *
 * <p>A tap captures the screen; a drag only repositions the bubble. The rules
 * are kept free of Android types so they can be unit tested on the JVM.</p>
 */
final class BubbleTouch {
    /** A touch held longer than this is not a tap. */
    static final long MAX_TAP_MS = 600L;

    private float downX;
    private float downY;
    private long downAt;
    private boolean dragging;

    /** Starts tracking a new touch. */
    void down(float x, float y, long at) {
        downX = x;
        downY = y;
        downAt = at;
        dragging = false;
    }

    /**
     * Reports a movement. Returns true once the touch moved far enough to be
     * a drag, which is when the bubble should follow the finger.
     */
    boolean moved(float x, float y, int slop) {
        if (dragging) return true;
        if (Math.abs(x - downX) > slop || Math.abs(y - downY) > slop) dragging = true;
        return dragging;
    }

    /**
     * Ends the touch. Returns true when it was a tap, meaning the bubble
     * should capture the screen.
     */
    boolean up(float x, float y, long at, int slop) {
        boolean tap = !dragging
                && at - downAt <= MAX_TAP_MS
                && Math.abs(x - downX) <= slop
                && Math.abs(y - downY) <= slop;
        dragging = false;
        return tap;
    }

    /** Abandons the current touch without firing a tap. */
    void cancel() {
        dragging = false;
    }
}
