package com.grewwc.aserve;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class NativeBridgePackageTest {
    @Test
    public void acceptsOnlyExactDottedAndroidPackageNames() {
        assertTrue(NativeBridge.isAndroidPackage("com.android.deskclock"));
        assertTrue(NativeBridge.isAndroidPackage("com.example.app_2"));

        assertFalse(NativeBridge.isAndroidPackage(null));
        assertFalse(NativeBridge.isAndroidPackage(""));
        assertFalse(NativeBridge.isAndroidPackage("deskclock"));
        assertFalse(NativeBridge.isAndroidPackage("com.android..deskclock"));
        assertFalse(NativeBridge.isAndroidPackage("2com.example"));
        assertFalse(NativeBridge.isAndroidPackage("com.example.desk-clock"));
        assertFalse(NativeBridge.isAndroidPackage("com.example/Activity"));
        assertFalse(NativeBridge.isAndroidPackage("a." + "x".repeat(254)));
    }
}
