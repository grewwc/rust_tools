package com.grewwc.aserve;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.util.Base64;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.TextureView;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collections;

/**
 * Live camera preview for the assistant.
 *
 * <p>The preview never writes to the gallery. The frame is read out of the
 * preview surface, compressed in memory and handed back to the page, so no
 * picture ends up in the camera roll and the user keeps full control over
 * what the assistant gets to see.</p>
 *
 * <p>The activity never leaves the user stranded: the window always shows a
 * status line and a cancel button, the camera has to open within
 * {@link #OPEN_TIMEOUT_MS}, a frame has to be taken within
 * {@link #FRAME_TIMEOUT_MS}, and every exit path answers the pending
 * request exactly once.</p>
 */
public final class CameraActivity extends Activity {
    static final int REQUEST = 4711;
    static final String EXTRA_FRAME = "camera_frame";
    static final String EXTRA_FRAME_PATH = "camera_frame_path";
    static final String EXTRA_ERROR = "camera_error";

    private static final int PERMISSION_REQUEST = 4712;
    private static final int MAX_EDGE = 1600;
    /** The camera has to open within this window, otherwise the user stares at a black screen. */
    private static final long OPEN_TIMEOUT_MS = 8000L;
    /** No frame may take longer than this, otherwise the preview closes itself. */
    private static final long FRAME_TIMEOUT_MS = 40000L;

    /** The preview that is currently on screen, so a stop request can close it. */
    private static CameraActivity running;

    private TextureView preview;
    private Button shutter;
    private TextView status;
    private CameraManager cameras;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private HandlerThread worker;
    private Handler workerHandler;
    private String cameraId;
    private Size previewSize = new Size(1280, 720);

    /** Decides when the preview may start and keeps the answer to a single one. */
    private final CameraFlow flow = new CameraFlow();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final Runnable openWatchdog = () -> {
        if (camera == null) fail("camera did not start");
    };
    private final Runnable frameWatchdog = () -> fail("camera timed out");

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        if (!CameraRequest.isLive()) {
            // Restored without a pending request: close before showing a
            // camera the assistant is not waiting for.
            finish();
            return;
        }
        running = this;
        // The window needs content before the permission dialog covers it, and
        // the preview has to start whether the permission is granted now or a
        // moment later.
        build();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST);
            return;
        }
        flow.setPermitted();
        startPreview();
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode != PERMISSION_REQUEST) return;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
            flow.setPermitted();
            startPreview();
        } else {
            fail("camera permission denied");
        }
    }

    @Override protected void onResume() {
        super.onResume();
        startPreview();
    }

    @Override protected void onPause() {
        super.onPause();
        closeCamera();
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        ui.removeCallbacks(openWatchdog);
        ui.removeCallbacks(frameWatchdog);
        closeCamera();
        if (running == this) running = null;
    }

    @Override public void onBackPressed() {
        // Answer the request on every way out, otherwise the assistant waits
        // for a frame that is never coming.
        cancel("camera cancelled");
    }

    /** Builds the window content: preview surface, status line and buttons. */
    private void build() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFF000000);
        preview = new TextureView(this);
        root.addView(preview, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        status = new TextView(this);
        status.setText("正在打开相机…");
        status.setTextColor(0xFFCCCCCC);
        status.setPadding(24, 20, 24, 20);
        root.addView(status);
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER);
        Button cancelButton = new Button(this);
        cancelButton.setText("取消");
        cancelButton.setOnClickListener(view -> cancel("camera cancelled"));
        shutter = new Button(this);
        shutter.setText("拍给助手");
        shutter.setOnClickListener(view -> grab());
        bar.addView(cancelButton);
        bar.addView(shutter);
        root.addView(bar);
        setContentView(root);
        flow.setBuilt();
    }

    /** Starts the preview as soon as the window and the permission are both there. */
    private void startPreview() {
        if (isFinishing() || flow.isDelivered() || !flow.shouldStart()) return;
        ensureWorker();
        flow.markStarted();
        openCamera();
    }

    private void ensureWorker() {
        if (worker != null) return;
        worker = new HandlerThread("a-serve-camera");
        worker.start();
        workerHandler = new Handler(worker.getLooper());
    }

    private void openCamera() {
        if (camera != null || preview == null || isFinishing()) return;
        try {
            cameras = (CameraManager) getSystemService(CAMERA_SERVICE);
            String fallback = null;
            for (String id : cameras.getCameraIdList()) {
                CameraCharacteristics info = cameras.getCameraCharacteristics(id);
                Integer facing = info.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {
                    cameraId = id;
                    break;
                }
                if (fallback == null) fallback = id;
            }
            if (cameraId == null) cameraId = fallback;
            if (cameraId == null) {
                fail("no camera on this device");
                return;
            }
            Size[] sizes = cameras.getCameraCharacteristics(cameraId)
                    .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    .getOutputSizes(SurfaceTexture.class);
            if (sizes != null && sizes.length > 0) previewSize = sizes[0];
            preview.setSurfaceTextureListener(listener);
            ui.removeCallbacks(openWatchdog);
            ui.postDelayed(openWatchdog, OPEN_TIMEOUT_MS);
            if (preview.isAvailable()) openDevice();
        } catch (Throwable error) {
            // A camera that cannot be opened must not take the app down with
            // it: vendor errors and memory failures end up here too.
            fail("camera unavailable: " + error.getMessage());
        }
    }

    private void openDevice() {
        if (camera != null || cameraId == null || isFinishing()) return;
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            fail("camera permission denied");
            return;
        }
        ensureWorker();
        try {
            cameras.openCamera(cameraId, stateCallback, workerHandler);
        } catch (Throwable error) {
            fail("camera unavailable: " + error.getMessage());
        }
    }

    private void createSession() {
        try {
            SurfaceTexture texture = preview.getSurfaceTexture();
            if (texture == null) {
                fail("camera preview missing");
                return;
            }
            texture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
            Surface surface = new Surface(texture);
            CaptureRequest.Builder builder = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            builder.addTarget(surface);
            camera.createCaptureSession(Collections.singletonList(surface),
                    new CameraCaptureSession.StateCallback() {
                        @Override public void onConfigured(CameraCaptureSession configured) {
                            session = configured;
                            try {
                                configured.setRepeatingRequest(builder.build(), null, workerHandler);
                                ui.removeCallbacks(frameWatchdog);
                                ui.postDelayed(frameWatchdog, FRAME_TIMEOUT_MS);
                                setStatus("相机已就绪");
                            } catch (Exception error) {
                                fail("camera preview failed");
                            }
                        }
                        @Override public void onConfigureFailed(CameraCaptureSession failed) {
                            fail("camera preview failed");
                        }
                    }, workerHandler);
        } catch (Exception error) {
            fail("camera preview failed");
        }
    }

    /** Reads the frame that is on screen and hands it back to the page. */
    private void grab() {
        if (preview == null) {
            fail("camera preview missing");
            return;
        }
        Bitmap frame = preview.getBitmap();
        if (frame == null) {
            Toast.makeText(this, "画面还没准备好", Toast.LENGTH_SHORT).show();
            setStatus("画面还没准备好，请稍候再试");
            return;
        }
        try {
            int edge = Math.max(frame.getWidth(), frame.getHeight());
            if (edge > MAX_EDGE) {
                float scale = (float) MAX_EDGE / edge;
                Bitmap smaller = Bitmap.createScaledBitmap(frame,
                        Math.max(1, Math.round(frame.getWidth() * scale)),
                        Math.max(1, Math.round(frame.getHeight() * scale)), true);
                if (smaller != frame) frame.recycle();
                frame = smaller;
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            if (!frame.compress(Bitmap.CompressFormat.JPEG, 85, out)) {
                fail("cannot encode the camera frame");
                return;
            }
            frame.recycle();
            Intent result = new Intent();
            result.putExtra(EXTRA_FRAME_PATH, storeFrame(out.toByteArray()));
            if (flow.deliver()) setResult(RESULT_OK, result);
            ui.removeCallbacks(frameWatchdog);
            finish();
        } catch (RuntimeException | IOException error) {
            fail("cannot encode the camera frame");
        }
    }

    /**
     * Writes the frame into the private cache and hands over its path. The
     * bytes travel in a file instead of the result intent: a base64 frame is
     * big enough to exceed the binder transaction limit, which would crash
     * the app while the result is delivered.
     */
    private String storeFrame(byte[] bytes) throws IOException {
        File file = new File(getCacheDir(), "camera-frame-" + System.nanoTime() + ".jpg");
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(bytes);
        }
        return file.getAbsolutePath();
    }

    private void closeCamera() {
        if (session != null) { session.close(); session = null; }
        if (camera != null) { camera.close(); camera = null; }
        if (worker != null) { worker.quitSafely(); worker = null; workerHandler = null; }
    }

    /** Leaves the preview without a frame and tells the assistant why. */
    private void cancel(String reason) {
        runOnUiThread(() -> {
            ui.removeCallbacks(openWatchdog);
            ui.removeCallbacks(frameWatchdog);
            if (flow.deliver()) {
                setResult(RESULT_CANCELED, new Intent().putExtra(EXTRA_ERROR, reason));
            }
            if (!isFinishing()) finish();
        });
    }

    /** Reports that the camera cannot be used and closes the preview. */
    private void fail(String error) {
        runOnUiThread(() -> {
            if (isFinishing()) return;
            ui.removeCallbacks(openWatchdog);
            ui.removeCallbacks(frameWatchdog);
            if (flow.deliver()) {
                setResult(RESULT_CANCELED, new Intent().putExtra(EXTRA_ERROR, error));
            }
            setStatus("相机无法使用：" + error);
            Toast.makeText(this, "相机无法使用：" + error, Toast.LENGTH_LONG).show();
            finish();
        });
    }

    private void setStatus(String text) {
        ui.post(() -> {
            if (status != null) status.setText(text);
        });
    }

    /** Closes the preview that is on screen, if any. */
    static void closeRunning() {
        CameraActivity open = running;
        if (open == null) return;
        open.runOnUiThread(() -> open.cancel("camera stopped"));
    }

    private final TextureView.SurfaceTextureListener listener = new TextureView.SurfaceTextureListener() {
        @Override public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            openDevice();
        }
        @Override public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) { }
        @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) { return true; }
        @Override public void onSurfaceTextureUpdated(SurfaceTexture surface) { }
    };

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override public void onOpened(CameraDevice device) {
            camera = device;
            ui.removeCallbacks(openWatchdog);
            createSession();
        }
        @Override public void onDisconnected(CameraDevice device) {
            fail("camera disconnected");
        }
        @Override public void onError(CameraDevice device, int error) {
            fail("camera error " + error);
        }
    };
}
