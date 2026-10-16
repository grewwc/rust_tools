package com.grewwc.aserve;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.net.Uri;
import android.net.http.SslError;
import android.os.Bundle;
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
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;

public final class MainActivity extends Activity {
    private static final String CONFIGURE_ACTION = "com.grewwc.aserve.CONFIGURE";
    private static final String PREFS = "a_serve";
    private static final String SERVER_ORIGIN = "server_origin";
    private static final int FILE_CHOOSER_REQUEST = 10;

    private WebView webView;
    private ProgressBar progressBar;
    private SharedPreferences preferences;
    private String serverOrigin;
    private ValueCallback<Uri[]> fileChooserCallback;
    private boolean connectionDialogVisible;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(Color.rgb(17, 20, 27));
        getWindow().setNavigationBarColor(Color.rgb(8, 10, 15));

        preferences = getSharedPreferences(PREFS, MODE_PRIVATE);
        serverOrigin = ServerUrl.normalize(
                preferences.getString(SERVER_ORIGIN, BuildConfig.DEFAULT_SERVER_URL));
        createWebView();

        boolean configure = CONFIGURE_ACTION.equals(getIntent().getAction());
        if (serverOrigin != null) loadApp();
        if (configure || serverOrigin == null) showServerDialog(null, serverOrigin == null);
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
    }

    private void loadApp() {
        webView.loadUrl(ServerUrl.freshAppUrl(serverOrigin, System.currentTimeMillis()));
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
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
        // A singleTask activity receives another launcher intent when its icon is
        // tapped again. Keep the live document so WebView focus and IME composition
        // survive; cold starts and explicit server changes still call loadApp().
    }

    @Override
    protected void onResume() {
        super.onResume();
        webView.onResume();
    }

    @Override
    protected void onPause() {
        webView.onPause();
        super.onPause();
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (fileChooserCallback != null) fileChooserCallback.onReceiveValue(null);
        fileChooserCallback = null;
        webView.stopLoading();
        webView.setWebChromeClient(null);
        webView.setWebViewClient(null);
        webView.destroy();
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
