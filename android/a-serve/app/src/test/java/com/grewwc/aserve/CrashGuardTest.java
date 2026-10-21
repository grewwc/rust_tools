package com.grewwc.aserve;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * The crash report is the only evidence available when the app cannot start,
 * so its shape has to stay stable: it must name the version, carry the
 * throwable, and stay short enough to be pasted into the composer.
 */
public final class CrashGuardTest {
    @Test
    public void reportNamesTheVersionAndCarriesTheThrowable() {
        String text = CrashGuard.describe("9.9.9", new IllegalStateException("webview missing"));
        assertTrue(text.startsWith("a serve APK 9.9.9"));
        assertTrue(text.contains("IllegalStateException: webview missing"));
        assertTrue(text.contains("CrashGuardTest"));
    }

    @Test
    public void oversizedReportIsTruncatedToStayPasteable() {
        String text = CrashGuard.describe("1.0.0",
                new IllegalStateException("x".repeat(400_000)));
        assertTrue(text.length() <= CrashGuard.MAX_CHARS + 32);
        assertTrue(text.contains("...(truncated)"));
    }

    @Test
    public void missingThrowableStillProducesAReadableReport() {
        String text = CrashGuard.describe("1.0.0", null);
        assertTrue(text.contains("no throwable recorded"));
    }
}
