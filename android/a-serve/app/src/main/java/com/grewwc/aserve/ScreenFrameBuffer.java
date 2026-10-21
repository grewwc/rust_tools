package com.grewwc.aserve;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * A short history of recent screen frames.
 *
 * <p>The user picks the capture moment by tapping a notification action, which
 * slides the notification shade over the content they meant to show. A frame
 * taken right after that tap shows the shade, not the content. Keeping a short
 * history lets the service deliver the frame from just before the tap instead,
 * so the assistant sees the picture the user was looking at.
 *
 * <p>Frames are stored as raw RGBA bytes. Decoding every mirrored frame into a
 * bitmap would cost more CPU than the mirroring itself; only the delivered
 * frame is decoded.
 */
final class ScreenFrameBuffer {
    /** How far back a delivered frame may be, in milliseconds. */
    static final long HISTORY_MS = 900L;
    /** A frame younger than this is not "before the tap" yet. */
    static final long MIN_AGE_MS = 120L;
    /** Minimum spacing between two stored frames, in milliseconds. */
    static final long SAMPLE_INTERVAL_MS = 180L;
    /** How many frames to keep. */
    static final int CAPACITY = 5;

    private final byte[][] pixels = new byte[CAPACITY][];
    private final long[] elapsedAt = new long[CAPACITY];
    private final long[] wallAt = new long[CAPACITY];
    private int next;
    private int count;
    private long lastSample;
    private int width;
    private int height;
    private int rowStride;
    private int pixelStride;

    /** One stored frame, copied out so later samples cannot overwrite it. */
    static final class Frame {
        final byte[] pixels;
        final int width;
        final int height;
        final int rowStride;
        final int pixelStride;
        final long elapsedAt;
        final long wallAt;

        Frame(byte[] pixels, int width, int height, int rowStride, int pixelStride, long elapsedAt, long wallAt) {
            this.pixels = pixels;
            this.width = width;
            this.height = height;
            this.rowStride = rowStride;
            this.pixelStride = pixelStride;
            this.elapsedAt = elapsedAt;
            this.wallAt = wallAt;
        }

        /** How long ago the frame was mirrored, in milliseconds. */
        long age(long now) { return now - elapsedAt; }
    }

    /** True when enough time passed to sample another frame. */
    synchronized boolean readyToSample(long now) {
        return now - lastSample >= SAMPLE_INTERVAL_MS;
    }

    /**
     * Stores the newest frame.
     *
     * @return true when the frame was stored
     */
    synchronized boolean offer(ByteBuffer plane, int width, int height, int rowStride, int pixelStride,
            long elapsed, long wall) {
        if (width <= 0 || height <= 0 || pixelStride <= 0 || rowStride <= 0) return false;
        int rowBytes = width * pixelStride;
        int needed = (height - 1) * rowStride + rowBytes;
        if (needed <= 0) return false;
        ByteBuffer source = plane.duplicate();
        if (source.limit() < needed) return false;
        source.position(0);
        byte[] target = pixels[next];
        if (target == null || target.length < needed) target = new byte[needed];
        source.get(target, 0, needed);
        pixels[next] = target;
        elapsedAt[next] = elapsed;
        wallAt[next] = wall;
        this.width = width;
        this.height = height;
        this.rowStride = rowStride;
        this.pixelStride = pixelStride;
        next = (next + 1) % CAPACITY;
        if (count < CAPACITY) count++;
        lastSample = elapsed;
        return true;
    }

    /**
     * The stored frame closest to the given moment.
     *
     * @return null when nothing was stored yet
     */
    synchronized Frame pick(long targetElapsed) {
        Frame best = null;
        long bestDelta = Long.MAX_VALUE;
        for (int i = 0; i < count; i++) {
            int index = (next - 1 - i + CAPACITY * 2) % CAPACITY;
            long delta = Math.abs(elapsedAt[index] - targetElapsed);
            if (delta < bestDelta) {
                bestDelta = delta;
                best = new Frame(Arrays.copyOf(pixels[index], pixels[index].length), width, height,
                        rowStride, pixelStride, elapsedAt[index], wallAt[index]);
            }
        }
        return best;
    }

    /** Drops every stored frame. */
    synchronized void clear() {
        for (int i = 0; i < CAPACITY; i++) pixels[i] = null;
        next = 0;
        count = 0;
        lastSample = 0L;
    }
}
