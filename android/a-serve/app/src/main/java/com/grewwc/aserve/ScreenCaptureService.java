package com.grewwc.aserve;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.graphics.drawable.Icon;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Base64;
import android.util.DisplayMetrics;
import android.view.WindowManager;
import android.widget.Toast;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import org.json.JSONObject;

/**
 * Explicitly authorized, visible sharing session. The user picks the capture
 * moment: a request is armed, and the frame is taken only after the user taps
 * the notification action. Only requested frames leave the phone.
 */
public final class ScreenCaptureService extends Service {
    private static final String CHANNEL = "screen_capture";
    private static final int NOTIFICATION = 103;
    private static final String IDLE_TITLE = "a · 屏幕共享已开启";
    private static final String IDLE_TEXT = "仅在你请求时截屏并发送给当前助手；点击停止撤销授权";
    private static final DateTimeFormatter CLOCK = DateTimeFormatter.ofPattern("HH:mm:ss");
    /** How far before the user's tap the delivered frame is taken. */
    private static final long PRE_TAP_LOOKBACK_MS = 600L;
    /** How long the delivery may take before the request fails. */
    private static final long CAPTURE_TIMEOUT_MS = 20_000L;
    private static ScreenCaptureService active;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final HandlerThread frames = new HandlerThread("screen-frame");
    private Handler worker;
    private MediaProjection projection;
    private ImageReader reader;
    private VirtualDisplay display;
    private String origin;
    private ScreenSession consentSession;
    private final ScreenCaptureState state = new ScreenCaptureState();
    private final ScreenFrameBuffer frameBuffer = new ScreenFrameBuffer();
    /** Floating capture button, shown while a request waits for the user. */
    private ScreenBubble bubble;

    /** How often the overlay is checked while a capture request is armed. */
    private static final long BUBBLE_WATCH_MS = 2000;

    /**
     * Some systems take the overlay window down when the app moves into the
     * background. The request is still armed then, so the button is put back
     * instead of leaving the user without a way to trigger the capture.
     */
    private final Runnable bubbleWatch = new Runnable() {
        @Override public void run() {
            if (stopped || state.token() == null) return;
            if (bubble != null && !bubble.visible()) {
                if (!bubble.show()) return; // the overlay permission was revoked
                bubble.setArmed(true);
            }
            main.postDelayed(this, BUBBLE_WATCH_MS);
        }
    };
    private volatile boolean stopped;
    private volatile boolean capturing;
    private PendingIntent openIntent;
    private PendingIntent captureIntent;
    private PendingIntent stopIntent;
    private final MediaProjection.Callback callback = new MediaProjection.Callback() {
        @Override public void onStop() { endSession("screen sharing stopped by system"); }
    };
    private final Runnable timeout = () -> failPending("screen capture request expired");
    private final Runnable captureTimeout = () -> failPending("screen capture timed out");

    // Called on the main thread, like the controller and lifecycle callbacks.
    static boolean capture(String origin, String token) {
        if (active == null || active.stopped) return false;
        if (active.consentSession == null || !active.consentSession.permits(origin, SystemClock.elapsedRealtime())) {
            active.endSession("server changed or sharing session expired"); return false;
        }
        try {
            active.arm(token);
        } catch (RuntimeException error) {
            ScreenCapture.finish(token, false, error.getMessage() == null ? "screen capture unavailable" : error.getMessage(), null);
        }
        return true;
    }

    /** Arms a request; no frame is taken until the user triggers it. */
    private void arm(String requestToken) {
        if (stopped) throw new IllegalStateException("screen sharing stopped");
        state.arm(requestToken, SystemClock.elapsedRealtime());
        main.removeCallbacks(timeout);
        main.removeCallbacks(captureTimeout);
        main.postDelayed(timeout, ScreenCaptureState.ARM_TIMEOUT_MS);
        if (showBubble()) {
            updateNotification("a · 等待你确认截屏", "切到目标页面，点悬浮球截屏", null);
            Toast.makeText(this, "已就绪：切到目标页面，点悬浮球截屏", Toast.LENGTH_LONG).show();
            main.removeCallbacks(bubbleWatch);
            main.postDelayed(bubbleWatch, BUBBLE_WATCH_MS);
        } else {
            updateNotification("a · 等待你确认截屏", "切到目标页面，展开通知栏点『立即截屏』", null);
            Toast.makeText(this, "已就绪：切到目标页面，展开通知栏点『立即截屏』", Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Puts the floating capture button on screen. The overlay permission is
     * optional: without it the notification action stays the only trigger.
     */
    private boolean showBubble() {
        if (bubble == null) {
            if (!ScreenBubble.permitted(this)) return false;
            bubble = new ScreenBubble(this);
            bubble.setOnTap(this::captureFromBubble);
        }
        if (!bubble.show()) return false;
        bubble.setArmed(true);
        return true;
    }

    /** Hides the floating button; the request keeps waiting for the user. */
    private void hideBubble() {
        if (bubble == null) return;
        bubble.setArmed(false);
        bubble.hide();
    }

    /**
     * Called when the user taps the floating button: the screen is captured
     * exactly as it looks at that moment.
     */
    private void captureFromBubble() {
        if (stopped) return;
        long now = SystemClock.elapsedRealtime();
        if (!state.trigger(now)) {
            Toast.makeText(this, "当前没有待截取的请求", Toast.LENGTH_SHORT).show();
            return;
        }
        main.removeCallbacks(captureTimeout);
        main.postDelayed(captureTimeout, CAPTURE_TIMEOUT_MS);
        // The bubble floats above the content, so it is hidden while the frame
        // is taken: the delivered picture is the page the user was looking at.
        bubble.setDrawn(false);
        state.scheduleLive(now, ScreenCaptureState.BUBBLE_SETTLE_MS, ScreenCaptureState.SOURCE_BUBBLE);
        updateNotification("a · 正在截屏", "正在发送你点按时的画面", null);
        main.postDelayed(() -> { if (bubble != null) bubble.setDrawn(true); },
                ScreenCaptureState.BUBBLE_SETTLE_MS + 200L);
    }

    /** Called from the notification action: the user chose the moment. */
    private void captureNow() {
        if (stopped) return;
        long now = SystemClock.elapsedRealtime();
        if (!state.trigger(now)) {
            // The floating bubble already fired this request.
            if (state.triggered()) return;
            Toast.makeText(this, "当前没有待截取的请求", Toast.LENGTH_SHORT).show();
            return;
        }
        main.removeCallbacks(captureTimeout);
        main.postDelayed(captureTimeout, CAPTURE_TIMEOUT_MS);
        // The notification shade covers the content when the tap lands, so the
        // frame that arrives at that moment is not the one the user meant.
        // Deliver the frame they were looking at a moment before the tap.
        ScreenFrameBuffer.Frame frame = frameBuffer.pick(now - PRE_TAP_LOOKBACK_MS);
        long age = frame == null ? Long.MAX_VALUE : frame.age(now);
        if (frame != null && age >= ScreenFrameBuffer.MIN_AGE_MS && age <= ScreenFrameBuffer.HISTORY_MS) {
            updateNotification("a · 正在截屏", "正在发送你点按前的画面", null);
            final String requestToken = state.token();
            final Handler target = worker;
            if (target == null) { failPending("screen capture unavailable"); return; }
            target.post(() -> deliver(requestToken, frame, ScreenCaptureState.SOURCE_PRE_TAP));
            return;
        }
        // No usable history: take whatever is on screen once the shade closed.
        state.scheduleLive(now);
        updateNotification("a · 正在截屏", "即将截取当前画面", null);
        Toast.makeText(this, "正在截取屏幕…", Toast.LENGTH_SHORT).show();
    }

    static void triggerCapture() {
        if (active == null || active.stopped) return;
        active.captureNow();
    }

    static void stopSession() { if (active != null) active.endSession("screen sharing stopped"); }

    /** Drops the pending request but keeps the consent session for later requests. */
    static void cancelPending() {
        if (active == null || active.stopped) return;
        active.failPending("screen capture timed out");
    }

    private void failPending(String error) {
        String requestToken = state.token();
        state.clear();
        hideBubble();
        main.removeCallbacks(bubbleWatch);
        main.removeCallbacks(timeout);
        main.removeCallbacks(captureTimeout);
        if (requestToken != null) ScreenCapture.finish(requestToken, false, error, null);
        updateNotification(IDLE_TITLE, IDLE_TEXT, null);
    }

    @Override public IBinder onBind(Intent intent) { return null; }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && "stop".equals(intent.getAction())) {
            endSession("screen sharing stopped");
            return START_NOT_STICKY;
        }
        if (intent != null && "capture".equals(intent.getAction())) {
            triggerCapture();
            return START_NOT_STICKY;
        }
        if (active != null) return START_NOT_STICKY;
        String requestToken = intent == null ? null : intent.getStringExtra("token");
        origin = intent == null ? null : intent.getStringExtra("origin");
        try {
            NotificationManager notifications = getSystemService(NotificationManager.class);
            notifications.createNotificationChannel(new NotificationChannel(CHANNEL,
                    "屏幕共享", NotificationManager.IMPORTANCE_LOW));
            stopIntent = PendingIntent.getService(this, 103,
                    new Intent(this, ScreenCaptureService.class).setAction("stop"), PendingIntent.FLAG_IMMUTABLE);
            captureIntent = PendingIntent.getService(this, 105,
                    new Intent(this, ScreenCaptureService.class).setAction("capture"), PendingIntent.FLAG_IMMUTABLE);
            openIntent = PendingIntent.getActivity(this, 104,
                    new Intent(this, MainActivity.class), PendingIntent.FLAG_IMMUTABLE);
            Notification notification = buildNotification(IDLE_TITLE, IDLE_TEXT, null);
            if (Build.VERSION.SDK_INT >= 29)
                startForeground(NOTIFICATION, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
            else startForeground(NOTIFICATION, notification);
            if (!notifications.areNotificationsEnabled()
                    || notifications.getNotificationChannel(CHANNEL).getImportance() == NotificationManager.IMPORTANCE_NONE)
                throw new IllegalStateException("enable notifications for visible screen sharing");
            if (!ScreenCapture.isActive(requestToken) || origin == null)
                throw new IllegalStateException("screen capture request expired");
            Intent consent = intent.getParcelableExtra("consent");
            if (consent == null) throw new IllegalStateException("screen capture consent missing");
            projection = getSystemService(MediaProjectionManager.class)
                    .getMediaProjection(android.app.Activity.RESULT_OK, consent);
            if (projection == null) throw new IllegalStateException("screen capture permission unavailable");
            projection.registerCallback(callback, main);
            frames.start();
            worker = new Handler(frames.getLooper());
            DisplayMetrics metrics = new DisplayMetrics();
            getSystemService(WindowManager.class).getDefaultDisplay().getRealMetrics(metrics);
            double scale = Math.min(1.0, 2048.0 / Math.max(metrics.widthPixels, metrics.heightPixels));
            int width = Math.max(1, (int) (metrics.widthPixels * scale));
            int height = Math.max(1, (int) (metrics.heightPixels * scale));
            reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2);
            // One virtual display for the entire consent session, including on Android 14+.
            display = projection.createVirtualDisplay("a screen sharing", width, height,
                    metrics.densityDpi, DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, reader.getSurface(), null, main);
            active = this;
            consentSession = new ScreenSession(origin, SystemClock.elapsedRealtime());
            reader.setOnImageAvailableListener(this::consumeFrame, worker);
            main.postDelayed(() -> endSession("screen sharing session expired"), ScreenSession.MAX_DURATION_MS);
            arm(requestToken);
        } catch (RuntimeException error) {
            endSession(error.getMessage() == null ? "screen capture unavailable" : error.getMessage());
        }
        return START_NOT_STICKY;
    }

    private Notification buildNotification(String title, String text, Bitmap thumbnail) {
        Notification.Builder builder = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(android.R.drawable.ic_menu_camera).setContentTitle(title)
                .setContentText(text)
                .setContentIntent(openIntent)
                .addAction(android.R.drawable.ic_menu_camera, "立即截屏", captureIntent)
                .addAction(android.R.drawable.ic_media_pause, "停止共享", stopIntent)
                .setOngoing(true).setCategory(Notification.CATEGORY_SERVICE);
        if (thumbnail != null) builder.setLargeIcon(Icon.createWithBitmap(thumbnail));
        return builder.build();
    }

    private void updateNotification(String title, String text, Bitmap thumbnail) {
        try {
            getSystemService(NotificationManager.class).notify(NOTIFICATION, buildNotification(title, text, thumbnail));
        } catch (RuntimeException ignored) {
            // The notification only reports progress; failing to refresh it must not break capture.
        }
    }

    private void consumeFrame(ImageReader source) {
        if (stopped || capturing) { drain(source); return; }
        long now = SystemClock.elapsedRealtime();
        if (state.due(now)) { captureLive(source); return; }
        bufferFrame(source, now);
    }

    /** Drops the newest frame without keeping it. */
    private static void drain(ImageReader source) {
        try (Image image = source.acquireLatestImage()) {
            // Closing the image releases it back to the reader.
        }
    }

    /**
     * Keeps a short history of mirrored frames. The notification shade covers
     * the content when the user taps the capture action, so the frame that
     * arrives at that moment is not the one they meant to share.
     */
    private void bufferFrame(ImageReader source, long now) {
        if (!frameBuffer.readyToSample(now)) { drain(source); return; }
        try (Image image = source.acquireLatestImage()) {
            if (image == null) return;
            Image.Plane plane = image.getPlanes()[0];
            frameBuffer.offer(plane.getBuffer(), image.getWidth(), image.getHeight(),
                    plane.getRowStride(), plane.getPixelStride(), now, System.currentTimeMillis());
        }
    }

    /** Fallback capture: takes the frame that is on screen once the shade closed. */
    private void captureLive(ImageReader source) {
        final String requestToken = state.token();
        capturing = true;
        try (Image image = source.acquireLatestImage()) {
            if (image == null) throw new IllegalStateException("screen image unavailable");
            Image.Plane plane = image.getPlanes()[0];
            byte[] raw = copyPlane(plane.getBuffer(), image.getWidth(), image.getHeight(),
                    plane.getRowStride(), plane.getPixelStride());
            deliver(requestToken, raw, image.getWidth(), image.getHeight(),
                    plane.getRowStride(), plane.getPixelStride(), System.currentTimeMillis(), state.source());
        } catch (Exception | OutOfMemoryError error) {
            main.post(() -> finishFrame(requestToken, false, "screen image unavailable", null, null, 0L));
        }
    }

    /** Small preview of the delivered frame, so the user can verify which moment was sent. */
    private static Bitmap thumbnail(Bitmap source) {
        int width = source.getWidth(), height = source.getHeight();
        int scaled = Math.min(320, width);
        int scaledHeight = Math.max(1, height * scaled / Math.max(1, width));
        Bitmap thumb = Bitmap.createScaledBitmap(source, scaled, scaledHeight, true);
        return thumb == source ? source.copy(Bitmap.Config.ARGB_8888, false) : thumb;
    }

    /** Decodes, encodes and hands one frame to the page. */
    private void deliver(String requestToken, ScreenFrameBuffer.Frame frame, String source) {
        deliver(requestToken, frame.pixels, frame.width, frame.height, frame.rowStride, frame.pixelStride,
                frame.wallAt, source);
    }

    private void deliver(String requestToken, byte[] raw, int width, int height, int rowStride, int pixelStride,
            long wallAt, String source) {
        Bitmap bitmap = null;
        try {
            bitmap = decodeRgba(raw, width, height, rowStride, pixelStride);
            byte[] bytes = encodeJpeg(bitmap);
            JSONObject result = new JSONObject().put("image_base64", Base64.encodeToString(bytes, Base64.NO_WRAP))
                    .put("mime_type", "image/jpeg").put("filename", "screen.jpg")
                    .put("width", bitmap.getWidth()).put("height", bitmap.getHeight())
                    .put("captured_at", wallAt)
                    .put("capture_source", source);
            Bitmap thumb = thumbnail(bitmap);
            main.post(() -> finishFrame(requestToken, true, null, result, thumb, wallAt));
        } catch (Exception | OutOfMemoryError error) {
            main.post(() -> finishFrame(requestToken, false, "screen image unavailable", null, null, 0L));
        } finally { if (bitmap != null) bitmap.recycle(); }
    }

    /** Copies one RGBA frame out of the reader's plane buffer. */
    private static byte[] copyPlane(ByteBuffer buffer, int width, int height, int rowStride, int pixelStride) {
        int base = buffer.position();
        int needed = (height - 1) * rowStride + width * pixelStride;
        ByteBuffer source = buffer.duplicate();
        if (source.limit() - base < needed) throw new IllegalStateException("screen frame truncated");
        source.position(base);
        byte[] raw = new byte[needed];
        source.get(raw, 0, needed);
        return raw;
    }

    /** Converts a raw RGBA frame into an ARGB bitmap. */
    private static Bitmap decodeRgba(byte[] raw, int width, int height, int rowStride, int pixelStride) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] row = new int[width];
        for (int y = 0; y < height; y++) {
            int base = y * rowStride;
            for (int x = 0; x < width; x++) {
                int at = base + x * pixelStride;
                row[x] = 0xff000000 | ((raw[at] & 255) << 16)
                        | ((raw[at + 1] & 255) << 8) | (raw[at + 2] & 255);
            }
            bitmap.setPixels(row, 0, width, 0, y, width, 1);
        }
        return bitmap;
    }

    /** Encodes the frame as JPEG, shrinking it until it fits the transfer limit. */
    private static byte[] encodeJpeg(Bitmap source) {
        Bitmap bitmap = source;
        byte[] bytes;
        while (true) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 85, output)) {
                if (bitmap != source) bitmap.recycle();
                throw new IllegalStateException("JPEG encoding failed");
            }
            bytes = output.toByteArray();
            if (bytes.length <= 256 * 1024) break;
            Bitmap smaller = Bitmap.createScaledBitmap(bitmap,
                    Math.max(1, bitmap.getWidth() * 3 / 4), Math.max(1, bitmap.getHeight() * 3 / 4), true);
            if (bitmap != source) bitmap.recycle();
            bitmap = smaller;
        }
        if (bitmap != source) bitmap.recycle();
        return bytes;
    }

    private void finishFrame(String requestToken, boolean ok, String error, JSONObject data, Bitmap thumbnail,
            long capturedAt) {
        capturing = false;
        if (stopped || requestToken == null || !requestToken.equals(state.token())) return;
        state.clear();
        hideBubble();
        main.removeCallbacks(bubbleWatch);
        main.removeCallbacks(timeout);
        main.removeCallbacks(captureTimeout);
        if (ok) updateNotification("a · 已截取 " + stamp(capturedAt),
                "助手已收到这一帧；要其它画面请再说一次", thumbnail);
        else updateNotification(IDLE_TITLE, IDLE_TEXT, null);
        ScreenCapture.finish(requestToken, ok, error, data);
    }

    /** Wall-clock time of the delivered frame, falling back to now. */
    private static String stamp(long capturedAt) {
        if (capturedAt <= 0) return LocalTime.now().format(CLOCK);
        return java.time.Instant.ofEpochMilli(capturedAt)
                .atZone(java.time.ZoneId.systemDefault()).toLocalTime().format(CLOCK);
    }

    private void endSession(String error) {
        if (stopped) return;
        stopped = true;
        hideBubble();
        frameBuffer.clear();
        if (consentSession != null) consentSession.stop();
        if (active == this) active = null;
        main.removeCallbacksAndMessages(null);
        if (display != null) { display.release(); display = null; }
        if (projection != null) {
            MediaProjection old = projection; projection = null;
            old.unregisterCallback(callback); old.stop();
        }
        ImageReader oldReader = reader; reader = null;
        if (worker != null) {
            worker.post(() -> { if (oldReader != null) oldReader.close(); frames.quitSafely(); });
        } else if (oldReader != null) oldReader.close();
        String requestToken = state.token();
        state.clear();
        ScreenCapture.finish(requestToken, false, error, null);
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelf();
    }
    @Override public void onDestroy() { endSession("screen sharing service stopped"); super.onDestroy(); }
}
