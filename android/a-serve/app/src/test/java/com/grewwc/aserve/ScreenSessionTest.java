package com.grewwc.aserve;

import org.junit.Test;
import static org.junit.Assert.*;

public class ScreenSessionTest {
    @Test public void reusesConsentOnlyWhileSameSessionIsActive() {
        ScreenSession session = new ScreenSession("https://example.test:8080", 100);
        assertTrue(session.permits("https://example.test:8080", 100));
        assertTrue(session.permits("https://example.test:8080", 500));
        assertFalse(session.permits("https://example.test:8081", 500));
        assertFalse(session.permits("https://example.test.evil:8080", 500));
        assertFalse(session.permits(null, 500));
        session.stop();
        assertFalse(session.permits("https://example.test:8080", 501));
    }

    @Test public void expiryAndClockResetFailClosed() {
        ScreenSession session = new ScreenSession("https://example.test", 100);
        assertTrue(session.permits("https://example.test", 100 + ScreenSession.MAX_DURATION_MS - 1));
        assertFalse(session.permits("https://example.test", 100 + ScreenSession.MAX_DURATION_MS));
        assertFalse(session.permits("https://example.test", 99));
    }
}
