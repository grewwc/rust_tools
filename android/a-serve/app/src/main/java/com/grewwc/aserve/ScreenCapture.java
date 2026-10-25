package com.grewwc.aserve;

import android.app.Activity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Handler;
import android.os.Looper;
import java.lang.ref.WeakReference;
import java.util.UUID;
import org.json.JSONObject;

/** A capture request is bound to its original page, within a visible sharing session. */
public final class ScreenCapture {
    public static final int REQUEST = 103;
    private static final Handler MAIN = new Handler(Looper.getMainLooper());
    private static Pending pending;
    private final WeakReference<MainActivity> activity;
    private String consentToken;

    static final class Pending {
        final String token = UUID.randomUUID().toString();
        final WeakReference<MainActivity> activity;
        final String id;
        final String origin;
        Runnable timeout;
        Pending(MainActivity owner, String id) {
            this.activity = new WeakReference<>(owner);
            this.id = id;
            this.origin = owner.nativeOrigin();
        }
    }

    public ScreenCapture(MainActivity activity) { this.activity = new WeakReference<>(activity); }

    public void capture(String requestId) {
        MainActivity owner = activity.get();
        if (owner == null || owner.isDestroyed() || owner.isFinishing() || !owner.isTrustedPage()) return;
        if (pending != null || consentToken != null) {
            owner.deliverNativeResult(requestId, false, "another screen capture is pending", null);
            return;
        }
        Pending request = new Pending(owner, requestId);
        pending = request;
        request.timeout = () -> {
            consentToken = null;
            finish(request.token, false, "screen capture timed out", null);
            // The service ends its one-shot sharing session when the request is
            // dropped; the next capture asks for system consent again.
            ScreenCaptureService.cancelPending();
        };
        MAIN.postDelayed(request.timeout, 120_000);
        try {
            if (ScreenCaptureService.capture(request.origin, request.token)) return;
            consentToken = request.token;
            MediaProjectionManager manager = owner.getSystemService(MediaProjectionManager.class);
            owner.startActivityForResult(manager.createScreenCaptureIntent(), REQUEST);
        } catch (RuntimeException error) {
            consentToken = null;
            finish(request.token, false, "screen capture consent unavailable", null);
        }
    }

    public void onResult(int resultCode, Intent data) {
        String token = consentToken;
        consentToken = null;
        if (token == null || !isActive(token)) return;
        if (resultCode != Activity.RESULT_OK || data == null) {
            finish(token, false, "screen capture cancelled or denied", null);
            return;
        }
        MainActivity owner = activity.get();
        if (owner == null || !sameOrigin(pending, owner)) {
            finish(token, false, "screen capture origin changed", null);
            return;
        }
        try {
            owner.startForegroundService(new Intent(owner, ScreenCaptureService.class)
                    .putExtra("token", token).putExtra("origin", pending.origin).putExtra("consent", data));
        } catch (RuntimeException error) {
            finish(token, false, "screen capture service unavailable", null);
        }
    }

    public void close() {
        MainActivity owner = activity.get();
        if (pending != null && pending.activity.get() == owner)
            finish(pending.token, false, "screen sharing stopped", null);
        consentToken = null;
        ScreenCaptureService.stopSession();
    }

    static boolean isActive(String token) {
        return pending != null && pending.token.equals(token);
    }

    private static boolean sameOrigin(Pending request, MainActivity owner) {
        return !owner.isDestroyed() && !owner.isFinishing() && owner.isTrustedPage()
                && request.origin != null && request.origin.equals(owner.nativeOrigin());
    }

    static void finish(String token, boolean ok, String error, JSONObject data) {
        if (!isActive(token)) return;
        Pending request = pending;
        pending = null;
        MAIN.removeCallbacks(request.timeout);
        MainActivity owner = request.activity.get();
        if (owner != null && sameOrigin(request, owner)) owner.deliverNativeResult(request.id, ok, error, data);
    }
}
