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
import android.util.Base64;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.TextureView;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.util.Collections;

/**
 * Live camera preview for the assistant.
 *
 * <p>The preview never writes to the gallery. The frame is read out of the
 * preview surface, compressed in memory and handed back to the page, so no
 * picture ends up in the camera roll and the user keeps full control over
 * what the assistant gets to see.</p>
 */
final class CameraActivity extends Activity {
    static final int REQUEST = 4711;
    static final String EXTRA_FRAME = "camera_frame";
    static final String EXTRA_ERROR = "camera_error";

    private static final int PERMISSION_REQUEST = 4712;
    private static final int MAX_EDGE = 1600;

    /** The preview that is currently on screen, so a stop request can close it. */
    private static CameraActivity running;

    private TextureView preview;
    private Button shutter;
    private CameraManager cameras;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private HandlerThread worker;
    private Handler workerHandler;
    private String cameraId;
    private Size previewSize = new Size(1280, 720);

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        running = this;
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST);
            return;
        }
        build();
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode != PERMISSION_REQUEST) return;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) build();
        else fail("camera permission denied");
    }

    /** Closes the preview that is on screen, if there is one. */
    static void closeRunning() {
        CameraActivity open = running;
        if (open == null) return;
        open.runOnUiThread(open::finish);
    }

    private void build() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFF000000);
        preview = new TextureView(this);
        root.addView(preview, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER);
        Button cancel = new Button(this);
        cancel.setText("取消");
        cancel.setOnClickListener(view -> {
            setResult(RESULT_CANCELED, new Intent().putExtra(EXTRA_ERROR, "camera cancelled"));
            finish();
        });
        shutter = new Button(this);
        shutter.setText("拍给助手");
        shutter.setOnClickListener(view -> grab());
        bar.addView(cancel);
        bar.addView(shutter);
        root.addView(bar);
        setContentView(root);
    }

    @Override protected void onResume() {
        super.onResume();
        if (preview != null) openCamera();
    }

    @Override protected void onPause() {
        super.onPause();
        closeCamera();
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        closeCamera();
        if (running == this) running = null;
    }

    private void openCamera() {
        if (camera != null || preview == null) return;
        try {
            cameras = (CameraManager) getSystemService(CAMERA_SERVICE);
            String fallback = null;
            for (String id : cameras.getCameraIdList()) {
                CameraCharacteristics info = cameras.getCameraCharacteristics(id);
                Integer facing = info.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) { cameraId = id; break; }
                if (fallback == null) fallback = id;
            }
            if (cameraId == null) cameraId = fallback;
            if (cameraId == null) { fail("no camera on this device"); return; }
            Size[] sizes = cameras.getCameraCharacteristics(cameraId)
                    .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    .getOutputSizes(SurfaceTexture.class);
            if (sizes != null && sizes.length > 0) previewSize = sizes[0];
            preview.setSurfaceTextureListener(listener);
            if (preview.isAvailable()) openDevice();
        } catch (Exception error) {
            fail("camera unavailable: " + error.getMessage());
        }
    }

    private final TextureView.SurfaceTextureListener listener = new TextureView.SurfaceTextureListener() {
        @Override public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
            openDevice();
        }
        @Override public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) { }
        @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) { return true; }
        @Override public void onSurfaceTextureUpdated(SurfaceTexture texture) { }
    };

    private void openDevice() {
        if (camera != null || cameraId == null) return;
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            fail("camera permission denied");
            return;
        }
        try {
            cameras.openCamera(cameraId, stateCallback, workerHandler);
        } catch (Exception error) {
            fail("camera unavailable: " + error.getMessage());
        }
    }

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override public void onOpened(CameraDevice device) {
            camera = device;
            createSession();
        }
        @Override public void onDisconnected(CameraDevice device) { closeCamera(); }
        @Override public void onError(CameraDevice device, int error) { fail("camera error " + error); }
    };

    private void createSession() {
        if (camera == null) return;
        try {
            SurfaceTexture texture = preview.getSurfaceTexture();
            if (texture == null) return;
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
                            } catch (Exception error) {
                                fail("camera preview failed");
                            }
                        }
                        @Override public void onConfigureFailed(CameraCaptureSession failed) { fail("camera preview failed"); }
                    }, workerHandler);
        } catch (Exception error) {
            fail("camera preview failed");
        }
    }

    /** Reads the frame that is on screen and hands it back to the page. */
    private void grab() {
        Bitmap frame = preview.getBitmap();
        if (frame == null) {
            Toast.makeText(this, "画面还没准备好", Toast.LENGTH_SHORT).show();
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
            result.putExtra(EXTRA_FRAME, Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP));
            setResult(RESULT_OK, result);
        } catch (RuntimeException error) {
            fail("cannot encode the camera frame");
            return;
        }
        finish();
    }

    private void closeCamera() {
        if (session != null) { session.close(); session = null; }
        if (camera != null) { camera.close(); camera = null; }
        if (worker != null) { worker.quitSafely(); worker = null; workerHandler = null; }
    }

    private void fail(String error) {
        runOnUiThread(() -> {
            if (isFinishing()) return;
            setResult(RESULT_CANCELED, new Intent().putExtra(EXTRA_ERROR, error));
            finish();
        });
    }

    @Override protected void onStart() {
        super.onStart();
        if (worker == null) {
            worker = new HandlerThread("a-serve-camera");
            worker.start();
            workerHandler = new Handler(worker.getLooper());
        }
    }
}
