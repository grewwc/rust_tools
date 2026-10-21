package com.grewwc.aserve;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * The camera preview needs a window and a permission, and they arrive in an
 * arbitrary order. These tests pin that ordering down, because a preview that
 * never starts leaves the user on a black screen with no way back.
 */
public class CameraFlowTest {

    @Test public void nothingStartsBeforeThePermissionIsGranted() {
        CameraFlow flow = new CameraFlow();
        flow.setBuilt();
        assertFalse("the window alone is not enough", flow.shouldStart());
    }

    @Test public void nothingStartsBeforeTheWindowIsBuilt() {
        CameraFlow flow = new CameraFlow();
        flow.setPermitted();
        assertFalse("the permission alone is not enough", flow.shouldStart());
    }

    @Test public void aPermissionGrantedAfterTheWindowStartsThePreview() {
        CameraFlow flow = new CameraFlow();
        flow.setBuilt();
        assertFalse(flow.shouldStart());
        flow.setPermitted();
        assertTrue("a late permission grant must still start the preview", flow.shouldStart());
    }

    @Test public void aPermissionGrantedBeforeTheWindowStartsThePreview() {
        CameraFlow flow = new CameraFlow();
        flow.setPermitted();
        assertFalse(flow.shouldStart());
        flow.setBuilt();
        assertTrue("a late window must still start the preview", flow.shouldStart());
    }

    @Test public void thePreviewKeepsRunningAcrossAResume() {
        CameraFlow flow = new CameraFlow();
        flow.setBuilt();
        flow.setPermitted();
        flow.markStarted();
        assertTrue("a pause must not stop the preview from restarting", flow.shouldStart());
    }

    @Test public void aRequestIsAnsweredOnlyOnce() {
        CameraFlow flow = new CameraFlow();
        assertTrue(flow.deliver());
        assertFalse("a second answer would overwrite the first", flow.deliver());
        assertTrue(flow.isDelivered());
    }

    @Test public void anAnsweredRequestDoesNotStartThePreview() {
        CameraFlow flow = new CameraFlow();
        flow.setBuilt();
        flow.setPermitted();
        flow.deliver();
        assertFalse(flow.shouldStart());
    }
}
