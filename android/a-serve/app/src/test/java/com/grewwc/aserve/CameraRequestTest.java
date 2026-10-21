package com.grewwc.aserve;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

public class CameraRequestTest {
    @Before public void reset() {
        CameraRequest.markLive(false);
    }

    @Test public void aFreshProcessHasNoLiveRequest() {
        assertFalse(CameraRequest.isLive());
    }

    @Test public void aWaitingRequestIsLive() {
        CameraRequest.markLive(true);
        assertTrue(CameraRequest.isLive());
    }

    @Test public void answeringTheRequestEndsIt() {
        CameraRequest.markLive(true);
        CameraRequest.markLive(false);
        assertFalse(CameraRequest.isLive());
    }
}
