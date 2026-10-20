package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class LocationQualityTest {
    @Test
    public void ageUsesMonotonicFixTimeAndRejectsMissingOrFutureTimestamps() {
        assertEquals(30_000L, LocationQuality.ageMs(100_000_000_000L, 70_000_000_000L));
        assertEquals(Long.MAX_VALUE, LocationQuality.ageMs(100, 101));
        assertEquals(Long.MAX_VALUE, LocationQuality.ageMs(100, 0));
        assertEquals(Long.MAX_VALUE, LocationQuality.ageMs(100, -1));
    }

    @Test
    public void onlyFreshAccurateGpsCacheCanSatisfyPreciseRequestImmediately() {
        assertTrue(LocationQuality.acceptCached(true, "gps", 30_000, 50));
        assertFalse(LocationQuality.acceptCached(true, "gps", 30_001, 5));
        assertFalse(LocationQuality.acceptCached(true, "gps", 0, 51));
        assertFalse(LocationQuality.acceptCached(true, "network", 0, 5));
        assertFalse(LocationQuality.acceptCached(false, "network", 0, 500));
        assertFalse(LocationQuality.acceptCached(false, "gps", 0, 5));
    }

    @Test
    public void approximateLiveFixIsAllowedButDoesNotMeetPreciseTarget() {
        assertTrue(LocationQuality.targetMet(false, "network", 100, 4000));
        assertFalse(LocationQuality.targetMet(true, "network", 100, 4000));
        assertFalse(LocationQuality.targetMet(false, "network", 100, 5001));
        assertFalse(LocationQuality.targetMet(true, "gps", 100, Float.NaN));
        assertFalse(LocationQuality.targetMet(true, "gps", 100, Float.POSITIVE_INFINITY));
        assertFalse(LocationQuality.targetMet(true, "gps", 100, 0));
        assertFalse(LocationQuality.targetMet(true, "gps", -1, 10));
    }

    @Test
    public void aNewCoarseFixDoesNotReplaceFreshAccurateGps() {
        assertFalse(LocationQuality.better(true, "network", 0, 1500, "gps", 20_000, 10));
        assertTrue(LocationQuality.better(true, "gps", 20_000, 10, "network", 0, 1500));
    }

    @Test
    public void freshnessWinsOverStaleNominalAccuracy() {
        assertTrue(LocationQuality.better(true, "network", 0, 1500, "gps", 31_000, 10));
        assertFalse(LocationQuality.better(true, "gps", 31_000, 10, "network", 0, 1500));
    }

    @Test
    public void fallbackHasAnExplicitAgeBound() {
        assertTrue(LocationQuality.usable(120_000));
        assertFalse(LocationQuality.usable(120_001));
        assertFalse(LocationQuality.usable(Long.MAX_VALUE));
        assertFalse(LocationQuality.usable(-1));
        assertFalse(LocationQuality.better(true, "gps", 120_001, 1, "network", 1, 1000));
        assertTrue(LocationQuality.better(true, "network", 1, 1000, "gps", Long.MAX_VALUE, 1));
    }

    @Test
    public void unknownAccuracyCannotBeatMeasuredAccuracyAndTiesPreferNewer() {
        assertFalse(LocationQuality.better(true, "network", 1, Float.NaN, "network", 2, 500));
        assertTrue(LocationQuality.better(false, "network", 1, 500, "network", 2, Float.NaN));
        assertTrue(LocationQuality.better(false, "network", 1, 500, "network", 2, 500));
        assertFalse(LocationQuality.better(false, "network", 2, 500, "network", 1, 500));
    }

    @Test
    public void rejectsImpossibleCoordinatesWithoutRejectingBoundaryValues() {
        assertTrue(LocationQuality.validCoordinates(-90, -180));
        assertTrue(LocationQuality.validCoordinates(90, 180));
        assertTrue(LocationQuality.validCoordinates(0, 0));
        assertFalse(LocationQuality.validCoordinates(90.01, 0));
        assertFalse(LocationQuality.validCoordinates(0, 180.01));
        assertFalse(LocationQuality.validCoordinates(Double.NaN, 0));
        assertFalse(LocationQuality.validCoordinates(0, Double.POSITIVE_INFINITY));
    }
}
