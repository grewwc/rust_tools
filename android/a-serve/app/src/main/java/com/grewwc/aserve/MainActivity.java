package com.grewwc.aserve;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.database.Cursor;
import android.net.Uri;
import android.net.http.SslError;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.OpenableColumns;
import android.util.Base64;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.CookieManager;
import android.webkit.SslErrorHandler;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public final class MainActivity extends Activity {
    private static final String CONFIGURE_ACTION = "com.grewwc.aserve.CONFIGURE";
    private static final String PREFS = "a_serve";
    private static final String SERVER_ORIGIN = "server_origin";
    private static final int FILE_CHOOSER_REQUEST = 10;
    private static final int NATIVE_PERMISSION_REQUEST = 11;
    private static final int MAX_SHARED_IMAGE_BYTES = 8 * 1024 * 1024;

    private WebView webView;
    private ProgressBar progressBar;
    private SharedPreferences preferences;
    private String serverOrigin;
    private ValueCallback<Uri[]> fileChooserCallback;
    private boolean connectionDialogVisible;
    private NativeBridge nativeBridge;
    private ScreenCapture screenCapture;
    /** The native request waiting for a frame from the camera preview. */
    private String pendingCameraRequest;
    private boolean nativeResumed;
    private Runnable grantedNativeAction;
    private String pendingPermissionOrigin;
    private PhoneDocuments phoneDocuments;
    private String pendingPermissionRequestId;
    private String pendingPermissionAction;
    private String pendingPermissionPayload;
    private String pendingShareJson;
    private boolean pageReady;
    private long pausedAt;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Installed first: anything that fails below must still be recoverable
        // on the next launch instead of closing the app without a trace.
        CrashGuard.install(this);
        getWindow().setStatusBarColor(Color.rgb(17, 20, 27));
        getWindow().setNavigationBarColor(Color.rgb(8, 10, 15));

        preferences = getSharedPreferences(PREFS, MODE_PRIVATE);
        serverOrigin = ServerUrl.normalize(
                preferences.getString(SERVER_ORIGIN, BuildConfig.DEFAULT_SERVER_URL));
        String previousCrash = CrashGuard.read(this);
        try {
            createWebView();
        } catch (Throwable error) {
            // A WebView that the system refuses to provide (for example a
            // disabled or broken "Android System WebView" package) fails here,
            // before any UI exists. Show the reason in native views instead of
            // letting the process disappear.
            CrashGuard.record(this, error);
            if (webView != null) {
                try {
                    webView.destroy();
                } catch (Throwable ignored) {
                    // The view never became usable; dropping the reference is enough.
                }
                webView = null;
            }
            showStartupFailure(CrashGuard.describe(BuildConfig.VERSION_NAME, error));
            return;
        }

        boolean configure = CONFIGURE_ACTION.equals(getIntent().getAction());
        if (serverOrigin != null) loadApp();
        handleShareIntent(getIntent());
        if (configure || serverOrigin == null) showServerDialog(null, serverOrigin == null);
        if (previousCrash != null) showCrashReport(previousCrash);
    }

    private void createWebView() {
        FrameLayout root = new FrameLayout(this);
        webView = new WebView(this);
        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setProgressTintList(android.content.res.ColorStateList.valueOf(
                Color.rgb(59, 130, 246)));

        root.addView(webView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        FrameLayout.LayoutParams progressParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(3), Gravity.TOP);
        root.addView(progressBar, progressParams);
        setContentView(root);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        // The server owns the UI. Always fetch its current shell instead of
        // restoring an obsolete document after the server has been upgraded.
        settings.setCacheMode(WebSettings.LOAD_NO_CACHE);
        settings.setAllowFileAccess(false);
        // The system picker grants access only to files explicitly selected by the user.
        settings.setAllowContentAccess(true);
        settings.setAllowFileAccessFromFileURLs(false);
        settings.setAllowUniversalAccessFromFileURLs(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setSafeBrowsingEnabled(true);
        settings.setSupportMultipleWindows(false);
        settings.setMediaPlaybackRequiresUserGesture(true);
        settings.setGeolocationEnabled(false);
        settings.setSaveFormData(false);

        WebView.setWebContentsDebuggingEnabled(false);
        CookieManager.getInstance().setAcceptCookie(true);
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, false);

        webView.setWebViewClient(new AppWebViewClient());
        webView.setWebChromeClient(new AppWebChromeClient());
        webView.setDownloadListener((url, userAgent, contentDisposition, mimeType, length) ->
                openExternalUrl(url));
        // The bridge exposes only allowlisted native actions; every call checks
        // the loaded origin again before it can reach the phone.
        nativeBridge = new NativeBridge(this);
        webView.addJavascriptInterface(nativeBridge, NativeBridge.NAME);
    }

    private void loadApp() {
        if (webView == null) return;
        if (screenCapture != null) screenCapture.close();
        pageReady = false;
        webView.loadUrl(ServerUrl.freshAppUrl(serverOrigin, System.currentTimeMillis()));
    }

    /**
     * Turn an incoming {@code ACTION_SEND} into a page call. Shared content only
     * pre-fills the composer; sharing never sends a message or starts a turn.
     */
    private void handleShareIntent(Intent intent) {
        if (intent == null || !Intent.ACTION_SEND.equals(intent.getAction())) return;
        String type = intent.getType() == null ? "" : intent.getType();
        String text = intent.getStringExtra(Intent.EXTRA_TEXT);
        JSONObject payload = new JSONObject();
        try {
            if (type.startsWith("image/")) {
                Uri stream = intent.getParcelableExtra(Intent.EXTRA_STREAM);
                String encoded = readSharedImage(stream);
                if (encoded == null) {
                    Toast.makeText(this, "无法读取分享的图片", Toast.LENGTH_SHORT).show();
                    return;
                }
                payload.put("kind", "image");
                payload.put("filename", sharedFileName(stream));
                payload.put("data_base64", encoded);
                if (text != null && !text.trim().isEmpty()) payload.put("text", text);
            } else {
                if (text == null || text.trim().isEmpty()) return;
                payload.put("kind", "text");
                payload.put("text", text);
            }
        } catch (JSONException error) {
            return;
        }
        pendingShareJson = payload.toString();
        flushShare();
    }

    private void flushShare() {
        if (pendingShareJson == null || !pageReady || webView == null) return;
        String json = pendingShareJson;
        pendingShareJson = null;
        WebView view = webView;
        view.post(() -> {
            if (view != webView) return;
            view.evaluateJavascript(
                    "window.AServeShare&&window.AServeShare(" + JSONObject.quote(json) + ");", null);
        });
    }

    private String readSharedImage(Uri uri) {
        if (uri == null) return null;
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] chunk = new byte[8192];
        try (InputStream input = getContentResolver().openInputStream(uri)) {
            if (input == null) return null;
            int read;
            while ((read = input.read(chunk)) != -1) {
                if (buffer.size() + read > MAX_SHARED_IMAGE_BYTES) return null;
                buffer.write(chunk, 0, read);
            }
        } catch (IOException | SecurityException error) {
            return null;
        }
        if (buffer.size() == 0) return null;
        return Base64.encodeToString(buffer.toByteArray(), Base64.NO_WRAP);
    }

    private String sharedFileName(Uri uri) {
        String name = null;
        Cursor cursor = null;
        try {
            cursor = getContentResolver().query(
                    uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null);
            if (cursor != null && cursor.moveToFirst()) name = cursor.getString(0);
        } catch (SecurityException ignored) {
            // Fall back to the default name below.
        } finally {
            if (cursor != null) cursor.close();
        }
        if (name == null || name.isEmpty()) return "shared-image";
        return name.replaceAll("[^\\w.\\-]+", "_");
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    /** Whether the document currently loaded still belongs to the configured server. */
    boolean isTrustedPage() {
        String current = webView == null ? null : webView.getUrl();
        return current != null && serverOrigin != null && ServerUrl.isSameOrigin(serverOrigin, current);
    }

    String nativeOrigin() { return serverOrigin; }

    void performScreen(String requestId, boolean stop) {
        if (screenCapture == null) screenCapture = new ScreenCapture(this);
        if (stop) {
            screenCapture.close();
            deliverNativeResult(requestId, true, null, new JSONObject());
        } else screenCapture.capture(requestId);
    }

    /**
     * Opens the live camera preview. The frame is read out of the preview and
     * handed back in memory, so nothing is written to the gallery.
     */
    void performCamera(String requestId, boolean stop) {
        if (stop) {
            CameraActivity.closeRunning();
            deliverNativeResult(requestId, true, null, new JSONObject());
            return;
        }
        // One preview owns the pending frame. Overwriting the field would stack a
        // second preview and hand its cancellation to this request instead.
        if (pendingCameraRequest != null) {
            deliverNativeResult(requestId, false,
                    "camera preview is already open; finish that one first", null);
            return;
        }
        pendingCameraRequest = requestId;
        try {
            CameraRequest.markLive(true);
            startActivityForResult(new Intent(this, CameraActivity.class), CameraActivity.REQUEST);
        } catch (RuntimeException error) {
            pendingCameraRequest = null;
            CameraRequest.markLive(false);
            deliverNativeResult(requestId, false, "camera preview unavailable", null);
        }
    }

    private void deliverCameraResult(int resultCode, Intent data) {
        String requestId = pendingCameraRequest;
        pendingCameraRequest = null;
        CameraRequest.markLive(false);
        if (requestId == null) return;
        if (resultCode == RESULT_OK && data != null) {
            String path = data.getStringExtra(CameraActivity.EXTRA_FRAME_PATH);
            if (path == null || path.isEmpty()) {
                deliverNativeResult(requestId, false, "camera frame missing", null);
                return;
            }
            String frame;
            try {
                frame = readBase64(new File(path));
            } catch (IOException | RuntimeException error) {
                deliverNativeResult(requestId, false, "camera frame unreadable", null);
                return;
            } finally {
                new File(path).delete();
            }
            JSONObject payload = new JSONObject();
            try {
                payload.put("image_base64", frame);
                payload.put("mime_type", "image/jpeg");
                payload.put("capture_source", "camera");
            } catch (JSONException ignored) {
                // The keys are constants; serialization cannot fail here.
            }
            deliverNativeResult(requestId, true, null, payload);
            return;
        }
        String error = data == null ? null : data.getStringExtra(CameraActivity.EXTRA_ERROR);
        deliverNativeResult(requestId, false, error == null ? "camera cancelled" : error, null);
    }

    /** Reads a file into a base64 string, the shape the page expects. */
    private static String readBase64(File file) throws IOException {
        byte[] bytes = new byte[(int) file.length()];
        try (FileInputStream in = new FileInputStream(file)) {
            int read = 0;
            while (read < bytes.length) {
                int step = in.read(bytes, read, bytes.length - read);
                if (step < 0) break;
                read += step;
            }
        }
        return Base64.encodeToString(bytes, Base64.NO_WRAP);
    }

    /**
     * Opens the system page where the floating capture button is allowed.
     * The grant is stored by the system, so the user only does this once.
     */
    void openOverlaySettings(String requestId) {
        ScreenBubble.openSettings(this);
        deliverNativeResult(requestId, true, null, new JSONObject());
    }

    void performDocumentAction(String requestId, String action, JSONObject payload) {
        if (phoneDocuments == null) phoneDocuments = new PhoneDocuments(this);
        phoneDocuments.perform(requestId, action, payload);
    }

    /**
     * Ask for the runtime permissions one native action needs, at the moment the
     * action runs. Android requires a visible, foreground request; there is no
     * background grant for location, calendar or contacts.
     */
    void requestNativePermissions(
            String requestId, String action, String payloadJson, String[] permissions) {
        // Bridge calls arrive on a WebView thread; runtime permission requests
        // must be issued from the activity's UI thread.
        runOnUiThread(() -> {
            if (isFinishing() || isDestroyed()) {
                deliverNativeResult(requestId, false, "app not active", null);
                return;
            }
            if (pendingPermissionRequestId != null) {
                deliverNativeResult(requestId, false, "another permission request is pending", null);
                return;
            }
            pendingPermissionRequestId = requestId;
            pendingPermissionOrigin = serverOrigin;
            pendingPermissionAction = action;
            pendingPermissionPayload = payloadJson;
            requestPermissions(permissions, NATIVE_PERMISSION_REQUEST);
        });
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != NATIVE_PERMISSION_REQUEST) return;
        String requestId = pendingPermissionRequestId;
        String action = pendingPermissionAction;
        String payloadJson = pendingPermissionPayload;
        String origin = pendingPermissionOrigin;
        pendingPermissionOrigin = null;
        pendingPermissionRequestId = null;
        pendingPermissionAction = null;
        pendingPermissionPayload = null;
        if (requestId == null || action == null) return;
        if (origin == null || !origin.equals(serverOrigin) || !isTrustedPage()) return;
        boolean granted = grantResults.length > 0;
        if ("location".equals(action)) {
            granted = checkSelfPermission(android.Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
                    || checkSelfPermission(android.Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED;
        } else {
            for (int result : grantResults) granted &= result == PackageManager.PERMISSION_GRANTED;
        }
        if (!granted) {
            deliverNativeResult(requestId, false, "permission denied", null);
            return;
        }
        grantedNativeAction = () -> {
            if (nativeBridge != null && origin.equals(serverOrigin) && isTrustedPage())
                nativeBridge.resumeAfterPermission(requestId, action, payloadJson);
        };
        if (nativeResumed) { Runnable continuation = grantedNativeAction; grantedNativeAction = null; continuation.run(); }
    }

    /** Report one native action result back to the server-owned page. */
    void deliverNativeResult(String requestId, boolean ok, String error, JSONObject data) {
        if (webView == null) return;
        JSONObject result = new JSONObject();
        try {
            result.put("id", requestId);
            result.put("ok", ok);
            if (error != null) result.put("error", error);
            if (data != null) result.put("data", data);
        } catch (JSONException ignored) {
            return;
        }
        String js = "window.AServeNativeResult&&window.AServeNativeResult("
                + JSONObject.quote(result.toString()) + ");";
        String resultOrigin = serverOrigin;
        WebView view = webView;
        if (view == null) return;
        view.post(() -> {
            if (!isDestroyed() && view == webView && resultOrigin != null
                    && resultOrigin.equals(serverOrigin) && isTrustedPage())
                view.evaluateJavascript(js, null);
        });
    }

    private boolean handleNavigation(Uri uri) {
        String scheme = uri.getScheme();
        if ("about".equalsIgnoreCase(scheme) || "blob".equalsIgnoreCase(scheme)) return false;
        if (!"https".equalsIgnoreCase(scheme)) return true;
        if (ServerUrl.isSameOrigin(serverOrigin, uri.toString())) return false;
        openExternalUrl(uri.toString());
        return true;
    }

    private void openExternalUrl(String value) {
        Uri uri = Uri.parse(value);
        if (!"https".equalsIgnoreCase(uri.getScheme())) {
            Toast.makeText(this, "已阻止不安全的链接", Toast.LENGTH_SHORT).show();
            return;
        }
        try {
            startActivity(new Intent(Intent.ACTION_VIEW, uri));
        } catch (ActivityNotFoundException error) {
            Toast.makeText(this, "没有可打开此链接的应用", Toast.LENGTH_SHORT).show();
        }
    }

    private void showConnectionProblem(int messageId) {
        showConnectionProblem(getString(messageId));
    }

    private void showConnectionProblem(String message) {
        runOnUiThread(() -> {
            if (isFinishing() || connectionDialogVisible) return;
            connectionDialogVisible = true;
            AlertDialog dialog = showServerDialog(message, false);
            dialog.setOnDismissListener(ignored -> connectionDialogVisible = false);
        });
    }

    private AlertDialog showServerDialog(String message, boolean required) {
        int padding = dp(22);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(padding, dp(8), padding, 0);

        if (message != null) {
            TextView details = new TextView(this);
            details.setText(message);
            details.setTextColor(Color.rgb(205, 212, 224));
            details.setPadding(0, 0, 0, dp(14));
            content.addView(details);
        }

        EditText input = new EditText(this);
        input.setSingleLine(true);
        input.setHint(R.string.server_dialog_hint);
        input.setText(serverOrigin != null ? serverOrigin : BuildConfig.DEFAULT_SERVER_URL);
        input.setSelectAllOnFocus(true);
        content.addView(input, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        AlertDialog.Builder builder = new AlertDialog.Builder(this)
                .setTitle(getString(R.string.server_dialog_title, BuildConfig.VERSION_NAME))
                .setView(content)
                .setPositiveButton(R.string.server_dialog_save, null);
        if (!required) builder.setNegativeButton(R.string.server_dialog_cancel, null);

        AlertDialog dialog = builder.create();
        dialog.setCancelable(!required);
        dialog.setCanceledOnTouchOutside(!required);
        dialog.setOnShowListener(ignored -> dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                .setOnClickListener(view -> {
                    String normalized = ServerUrl.normalize(input.getText().toString());
                    if (normalized == null) {
                        input.setError(getString(R.string.server_url_invalid));
                        return;
                    }
                    serverOrigin = normalized;
                    preferences.edit().putString(SERVER_ORIGIN, normalized).apply();
                    webView.stopLoading();
                    webView.clearHistory();
                    loadApp();
                    dialog.dismiss();
                }));
        dialog.show();
        return dialog;
    }

    /**
     * Native-only screen for a failed startup. The page never loaded, so this
     * must not depend on the WebView in any way.
     */
    private void showStartupFailure(String report) {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(8, 10, 15));
        int pad = dp(20);
        root.setPadding(pad, pad, pad, pad);

        TextView heading = new TextView(this);
        heading.setText(R.string.webview_unavailable);
        heading.setTextColor(Color.WHITE);
        heading.setTextSize(17);
        heading.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(heading);

        TextView body = new TextView(this);
        body.setText(report);
        body.setTextColor(Color.rgb(205, 212, 224));
        body.setTextSize(11);
        body.setTypeface(Typeface.MONOSPACE);
        body.setTextIsSelectable(true);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(body, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        LinearLayout actions = new LinearLayout(this);
        TextView copy = new TextView(this);
        copy.setText(R.string.crash_copy);
        copy.setTextColor(Color.rgb(59, 130, 246));
        copy.setPadding(0, dp(12), dp(24), dp(12));
        copy.setOnClickListener(view -> copyToClipboard(report));
        TextView retry = new TextView(this);
        retry.setText(R.string.crash_retry);
        retry.setTextColor(Color.rgb(59, 130, 246));
        retry.setPadding(0, dp(12), 0, dp(12));
        retry.setOnClickListener(view -> recreate());
        actions.addView(copy);
        actions.addView(retry);
        root.addView(actions);

        setContentView(root);
    }

    /** Report the previous run's crash now that the page can receive text. */
    private void showCrashReport(String report) {
        if (isFinishing()) return;
        try {
            ScrollView scroll = new ScrollView(this);
            TextView body = new TextView(this);
            body.setText(report);
            body.setTextSize(11);
            body.setTypeface(Typeface.MONOSPACE);
            body.setTextIsSelectable(true);
            scroll.addView(body, new FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT));
            new AlertDialog.Builder(this)
                    .setTitle(R.string.crash_title)
                    .setView(scroll)
                    .setPositiveButton(R.string.crash_send,
                            (dialog, which) -> sendReportToAgent(report))
                    .setNeutralButton(R.string.crash_copy,
                            (dialog, which) -> copyToClipboard(report))
                    .setNegativeButton(R.string.crash_dismiss,
                            (dialog, which) -> CrashGuard.clear(this))
                    .create()
                    .show();
        } catch (Throwable error) {
            // A diagnostic dialog must never become the next crash.
        }
    }

    /**
     * Route the report through the composer: the page already holds the
     * credentials, so the text reaches the agent without a second channel.
     */
    private void sendReportToAgent(String report) {
        try {
            JSONObject payload = new JSONObject();
            payload.put("kind", "text");
            payload.put("text", getString(R.string.crash_prompt) + "\n\n" + report);
            pendingShareJson = payload.toString();
            flushShare();
            CrashGuard.clear(this);
        } catch (Throwable ignored) {
            // The copy button remains available.
        }
    }

    private void copyToClipboard(String text) {
        try {
            ClipboardManager clipboard =
                    (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            if (clipboard == null) return;
            clipboard.setPrimaryClip(ClipData.newPlainText("a serve crash", text));
            Toast.makeText(this, R.string.crash_copied, Toast.LENGTH_SHORT).show();
        } catch (Throwable ignored) {
            // Nothing else can be offered if the clipboard is unavailable.
        }
    }

    private void openFileChooser() {
        if (fileChooserCallback == null) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT)
                .addCategory(Intent.CATEGORY_OPENABLE)
                .setType("image/*")
                .putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        try {
            startActivityForResult(intent, FILE_CHOOSER_REQUEST);
        } catch (ActivityNotFoundException error) {
            fileChooserCallback.onReceiveValue(null);
            fileChooserCallback = null;
            Toast.makeText(this, "没有可选择图片的应用", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == ScreenCapture.REQUEST) {
            if (screenCapture != null) screenCapture.onResult(resultCode, data);
            return;
        }
        if (requestCode == CameraActivity.REQUEST) {
            deliverCameraResult(resultCode, data);
            return;
        }
        if (requestCode == PhoneDocuments.REQUEST) {
            if (phoneDocuments != null) phoneDocuments.onResult(resultCode, data);
            return;
        }
        if (requestCode != FILE_CHOOSER_REQUEST || fileChooserCallback == null) return;

        ArrayList<Uri> selected = new ArrayList<>();
        if (resultCode == RESULT_OK && data != null) {
            ClipData clips = data.getClipData();
            if (clips != null) {
                for (int i = 0; i < clips.getItemCount(); i++) {
                    selected.add(clips.getItemAt(i).getUri());
                }
            } else if (data.getData() != null) {
                selected.add(data.getData());
            }
        }
        fileChooserCallback.onReceiveValue(selected.isEmpty()
                ? null : selected.toArray(new Uri[0]));
        fileChooserCallback = null;
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        if (CONFIGURE_ACTION.equals(intent.getAction())) {
            showServerDialog(null, false);
        }
        handleShareIntent(intent);
        // A singleTask activity receives another launcher intent when its icon is
        // tapped again. Keep the live document so WebView focus and IME composition
        // survive; cold starts and explicit server changes still call loadApp().
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Startup may have failed before a WebView existed; the native error
        // screen stays usable only if the lifecycle callbacks stop here.
        if (webView == null) return;
        webView.onResume();
        nativeResumed = true;
        if (nativeBridge != null) nativeBridge.onHostResume();
        if (grantedNativeAction != null) { Runnable continuation = grantedNativeAction; grantedNativeAction = null; continuation.run(); }
        // The WebView keeps its document across app switches. A document loaded
        // before the current UI generation cannot run the newer reconnect logic,
        // so replace it once the app returns from a real background stay.
        if (pausedAt > 0 && SystemClock.elapsedRealtime() - pausedAt > 2000) {
            webView.evaluateJavascript(
                "String(window.__aserveUi||'')",
                value -> {
                    String marker = value == null ? "" : value.replace("\"", "").trim();
                    if (marker.isEmpty() || "null".equals(marker)) loadApp();
                });
        }
        pausedAt = 0;
    }

    @Override
    protected void onPause() {
        nativeResumed = false;
        if (nativeBridge != null) nativeBridge.onHostPause();
        pausedAt = SystemClock.elapsedRealtime();
        if (webView != null) webView.onPause();
        super.onPause();
    }

    @Override
    public void onBackPressed() {
        if (webView == null) super.onBackPressed();
        else if (webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (screenCapture != null) screenCapture.close();
        if (nativeBridge != null) nativeBridge.close();
        if (fileChooserCallback != null) fileChooserCallback.onReceiveValue(null);
        fileChooserCallback = null;
        if (webView != null) {
            try {
                webView.removeJavascriptInterface(NativeBridge.NAME);
                webView.stopLoading();
                webView.setWebChromeClient(null);
                webView.setWebViewClient(null);
                webView.destroy();
            } catch (Throwable ignored) {
                // A half-built WebView must not crash the shutdown path too.
            }
            webView = null;
        }
        super.onDestroy();
    }

    private final class AppWebViewClient extends WebViewClient {
        @Override
        public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
            return handleNavigation(request.getUrl());
        }

        @Override
        public boolean shouldOverrideUrlLoading(WebView view, String url) {
            return handleNavigation(Uri.parse(url));
        }

        @Override
        public void onPageFinished(WebView view, String url) {
            progressBar.setVisibility(View.GONE);
            pageReady = true;
            flushShare();
        }

        @Override
        public void onReceivedSslError(WebView view, SslErrorHandler handler, SslError error) {
            handler.cancel();
            int reason;
            switch (error.getPrimaryError()) {
                case SslError.SSL_NOTYETVALID:
                    reason = R.string.tls_not_yet_valid;
                    break;
                case SslError.SSL_EXPIRED:
                    reason = R.string.tls_expired;
                    break;
                case SslError.SSL_IDMISMATCH:
                    reason = R.string.tls_id_mismatch;
                    break;
                case SslError.SSL_UNTRUSTED:
                    reason = R.string.tls_untrusted;
                    break;
                case SslError.SSL_DATE_INVALID:
                    reason = R.string.tls_date_invalid;
                    break;
                default:
                    reason = R.string.tls_invalid;
                    break;
            }
            showConnectionProblem(getString(
                    R.string.server_tls_failed, BuildConfig.VERSION_NAME, getString(reason)));
        }

        @Override
        public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
            if (request.isForMainFrame()) showConnectionProblem(R.string.server_connection_failed);
        }

        @Override
        public void onReceivedHttpError(
                WebView view, WebResourceRequest request, WebResourceResponse response) {
            if (request.isForMainFrame() && response.getStatusCode() >= 400) {
                showConnectionProblem(R.string.server_connection_failed);
            }
        }
    }

    private final class AppWebChromeClient extends WebChromeClient {
        @Override
        public void onProgressChanged(WebView view, int progress) {
            progressBar.setVisibility(progress < 100 ? View.VISIBLE : View.GONE);
            progressBar.setProgress(progress);
        }

        @Override
        public boolean onShowFileChooser(
                WebView view,
                ValueCallback<Uri[]> callback,
                FileChooserParams params) {
            if (fileChooserCallback != null) fileChooserCallback.onReceiveValue(null);
            fileChooserCallback = callback;
            openFileChooser();
            return true;
        }
    }
}
