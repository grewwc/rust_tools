package com.grewwc.aserve;

import java.nio.ByteBuffer;
import org.junit.Test;
import static org.junit.Assert.*;

/** The frame history that makes the tapped moment deliverable. */
public class ScreenFrameBufferTest {
    /** A one-pixel RGBA frame, tagged with a value so copies can be told apart. */
    private static ByteBuffer frame(int tag, int width, int height) {
        int rowStride = width * 4;
        ByteBuffer buffer = ByteBuffer.allocate(rowStride * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                buffer.put(y * rowStride + x * 4, (byte) tag);
                buffer.put(y * rowStride + x * 4 + 1, (byte) tag);
                buffer.put(y * rowStride + x * 4 + 2, (byte) tag);
                buffer.put(y * rowStride + x * 4 + 3, (byte) 255);
            }
        }
        buffer.position(0);
        return buffer;
    }

    @Test public void nothingIsPickedBeforeAFrameIsStored() {
        ScreenFrameBuffer buffer = new ScreenFrameBuffer();
        assertNull(buffer.pick(1000));
    }

    @Test public void theStoredFrameIsPickedBack() {
        ScreenFrameBuffer buffer = new ScreenFrameBuffer();
        assertTrue(buffer.offer(frame(7, 2, 2), 2, 2, 8, 4, 1000, 555000L));
        ScreenFrameBuffer.Frame picked = buffer.pick(1000);
        assertNotNull(picked);
        assertEquals(1000, picked.elapsedAt);
        assertEquals(555000L, picked.wallAt);
        assertEquals(2, picked.width);
        assertEquals(2, picked.height);
        assertEquals(7, picked.pixels[0] & 255);
    }

    @Test public void theClosestFrameToTheTargetWins() {
        ScreenFrameBuffer buffer = new ScreenFrameBuffer();
        buffer.offer(frame(1, 1, 1), 1, 1, 4, 4, 1000, 1L);
        buffer.offer(frame(2, 1, 1), 1, 1, 4, 4, 1200, 2L);
        buffer.offer(frame(3, 1, 1), 1, 1, 4, 4, 1400, 3L);
        assertEquals(2, buffer.pick(1250).pixels[0] & 255);
        assertEquals(1, buffer.pick(1050).pixels[0] & 255);
        assertEquals(3, buffer.pick(1350).pixels[0] & 255);
    }

    @Test public void aPickedFrameSurvivesLaterSamples() {
        ScreenFrameBuffer buffer = new ScreenFrameBuffer();
        buffer.offer(frame(1, 1, 1), 1, 1, 4, 4, 1000, 1L);
        ScreenFrameBuffer.Frame picked = buffer.pick(1000);
        for (int i = 1; i <= ScreenFrameBuffer.CAPACITY + 2; i++) {
            buffer.offer(frame(10 + i, 1, 1), 1, 1, 4, 4, 1000 + i * 200L, 1L + i);
        }
        assertEquals(1, picked.pixels[0] & 255);
        // The same slot was recycled, but a fresh pick must not see the old bytes.
        assertEquals(10 + ScreenFrameBuffer.CAPACITY + 2, buffer.pick(99999).pixels[0] & 255);
    }

    @Test public void samplingIsThrottled() {
        ScreenFrameBuffer buffer = new ScreenFrameBuffer();
        assertTrue(buffer.readyToSample(10_000));
        buffer.offer(frame(1, 1, 1), 1, 1, 4, 4, 10_000, 1L);
        assertFalse(buffer.readyToSample(10_000 + ScreenFrameBuffer.SAMPLE_INTERVAL_MS - 1));
        assertTrue(buffer.readyToSample(10_000 + ScreenFrameBuffer.SAMPLE_INTERVAL_MS));
    }

    @Test public void aTruncatedFrameIsRejected() {
        ScreenFrameBuffer buffer = new ScreenFrameBuffer();
        ByteBuffer tiny = ByteBuffer.allocate(4);
        assertFalse(buffer.offer(tiny, 4, 4, 16, 4, 1000, 1L));
        assertNull(buffer.pick(1000));
    }

    @Test public void clearingDropsEveryFrame() {
        ScreenFrameBuffer buffer = new ScreenFrameBuffer();
        buffer.offer(frame(1, 1, 1), 1, 1, 4, 4, 1000, 1L);
        buffer.clear();
        assertNull(buffer.pick(1000));
    }
}
