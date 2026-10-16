package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class ServerUrlTest {
    @Test
    public void normalizesAppEntryToOrigin() {
        assertEquals("https://10.37.110.250:8080",
                ServerUrl.normalize(" https://10.37.110.250:8080/app/ "));
        assertEquals("https://example.test", ServerUrl.normalize("https://EXAMPLE.test/"));
    }

    @Test
    public void freshAppUrlBypassesAStoredShellDocument() {
        assertEquals(
                "https://example.test:8080/app?shell_reload=42",
                ServerUrl.freshAppUrl("https://example.test:8080", 42));
    }

    @Test
    public void rejectsUnsafeOrAmbiguousUrls() {
        assertNull(ServerUrl.normalize("http://10.37.110.250:8080"));
        assertNull(ServerUrl.normalize("https://user@example.test"));
        assertNull(ServerUrl.normalize("https://example.test/other"));
        assertNull(ServerUrl.normalize("javascript:alert(1)"));
    }

    @Test
    public void comparesOriginsInsteadOfPrefixes() {
        assertTrue(ServerUrl.isSameOrigin(
                "https://example.test:8080", "https://example.test:8080/sessions"));
        assertFalse(ServerUrl.isSameOrigin(
                "https://example.test:8080", "https://example.test.evil:8080/app"));
        assertFalse(ServerUrl.isSameOrigin(
                "https://example.test:8080", "https://example.test/app"));
    }
}