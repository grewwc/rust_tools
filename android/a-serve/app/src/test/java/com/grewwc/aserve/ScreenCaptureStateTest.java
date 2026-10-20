package com.grewwc.aserve;

import org.junit.Test;
import static org.junit.Assert.*;

public class ScreenCaptureStateTest {
    @Test public void armedRequestNeverCapturesOnItsOwn() {
        ScreenCaptureState state = new ScreenCaptureState();
        state.arm("t1", 1000);
        assertTrue(state.armed());
        assertEquals("t1", state.token());
        assertFalse(state.due(1000));
        assertFalse(state.due(1000 + ScreenCaptureState.SETTLE_MS));
        assertFalse(state.due(1000 + ScreenCaptureState.ARM_TIMEOUT_MS - 1));
    }

    @Test public void triggerSchedulesTheFrameAfterTheSettleDelay() {
        ScreenCaptureState state = new ScreenCaptureState();
        state.arm("t1", 1000);
        assertTrue(state.trigger(5000));
        assertFalse(state.due(5000 + ScreenCaptureState.SETTLE_MS - 1));
        assertTrue(state.due(5000 + ScreenCaptureState.SETTLE_MS));
    }

    @Test public void triggerWithoutArmedRequestFails() {
        ScreenCaptureState state = new ScreenCaptureState();
        assertFalse(state.trigger(1000));
    }

    @Test public void armRejectsAnotherPendingRequest() {
        ScreenCaptureState state = new ScreenCaptureState();
        state.arm("t1", 1000);
        try {
            state.arm("t2", 1100);
            fail("expected a busy failure");
        } catch (IllegalStateException expected) {
            assertEquals("screen capture busy", expected.getMessage());
        }
        state.arm("t1", 1200);
        assertEquals("t1", state.token());
    }

    @Test public void expiryOnlyAppliesToArmedRequests() {
        ScreenCaptureState state = new ScreenCaptureState();
        assertFalse(state.expired(Long.MAX_VALUE / 2));
        state.arm("t1", 1000);
        assertFalse(state.expired(1000 + ScreenCaptureState.ARM_TIMEOUT_MS - 1));
        assertTrue(state.expired(1000 + ScreenCaptureState.ARM_TIMEOUT_MS));
        state.clear();
        assertFalse(state.expired(1000 + ScreenCaptureState.ARM_TIMEOUT_MS));
        assertFalse(state.armed());
    }

    @Test public void clearingDropsThePendingFrame() {
        ScreenCaptureState state = new ScreenCaptureState();
        state.arm("t1", 1000);
        state.trigger(2000);
        state.clear();
        assertFalse(state.due(9999));
        assertEquals(ScreenCaptureState.NEVER, state.captureAfter());
    }
}
