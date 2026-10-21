package com.grewwc.aserve;

import org.junit.Test;
import static org.junit.Assert.*;

/** Gesture rules for the floating capture button. */
public class BubbleTouchTest {
    private static final int SLOP = 24;

    @Test public void aShortStillTouchIsATap() {
        BubbleTouch touch = new BubbleTouch();
        touch.down(100f, 200f, 1000L);
        assertTrue(touch.up(104f, 203f, 1100L, SLOP));
    }

    @Test public void aMoveBeyondTheSlopIsADrag() {
        BubbleTouch touch = new BubbleTouch();
        touch.down(100f, 200f, 1000L);
        assertFalse(touch.moved(110f, 205f, SLOP));
        assertTrue(touch.moved(140f, 200f, SLOP));
    }

    @Test public void aDragDoesNotFireATap() {
        BubbleTouch touch = new BubbleTouch();
        touch.down(100f, 200f, 1000L);
        touch.moved(300f, 400f, SLOP);
        assertFalse(touch.up(300f, 400f, 1100L, SLOP));
    }

    @Test public void aTouchHeldTooLongIsNotATap() {
        BubbleTouch touch = new BubbleTouch();
        touch.down(100f, 200f, 1000L);
        assertFalse(touch.up(100f, 200f, 1000L + BubbleTouch.MAX_TAP_MS + 1, SLOP));
    }

    @Test public void aTapWithoutAnyMovementEventStillCounts() {
        // Some devices deliver only down and up, so the end position decides.
        BubbleTouch touch = new BubbleTouch();
        touch.down(100f, 200f, 1000L);
        assertTrue(touch.up(112f, 214f, 1200L, SLOP));
    }

    @Test public void aSecondTouchStartsFresh() {
        BubbleTouch touch = new BubbleTouch();
        touch.down(100f, 200f, 1000L);
        touch.moved(300f, 400f, SLOP);
        touch.down(100f, 200f, 2000L);
        assertTrue(touch.up(100f, 200f, 2100L, SLOP));
    }
}
