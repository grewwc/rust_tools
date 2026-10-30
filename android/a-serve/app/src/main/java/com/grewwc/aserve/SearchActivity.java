package com.grewwc.aserve;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.CookieManager;
import android.webkit.RenderProcessGoneDetail;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.Locale;

/**
 * A visible, app-owned browser used only to run one search query and read its
 * result list. The page is untrusted: this activity never installs the
 * {@code AServeNative} bridge, blocks non-http(s) navigation, and hands result
 * links to the system browser instead of turning into a general browser.
 *
 * <p>The user owns verification. When the engine shows a consent page or a
 * CAPTCHA, the WebView stays visible and interactive; extraction resumes after
 * the page produces results. This is deliberately not an AccessibilityService,
 * so the app can never inspect or click another app's browser.</p>
 */
public final class SearchActivity extends Activity {
    static final int REQUEST = 0x5E42;
    static final String EXTRA_QUERY = "search_query";
    static final String EXTRA_ENGINE = "search_engine";
    static final String EXTRA_MAX = "search_max";
    static final String EXTRA_RESULT = "search_result";
    static final String EXTRA_ERROR = "search_error";

    private static final long POLL_MS = 750L;
    private static final long SETTLE_MS = 1_200L;
    // Stay below the page's 180s native timeout so the model gets a concrete
    // error instead of a request that disappears with the page timer.
    private static final long EXTRACT_TIMEOUT_MS = 165_000L;
    private static final int MAX_QUERY_CHARS = 500;
    private static final int MAX_RESULTS = 10;

    // Literal ARGB values keep class initialization free of Android calls, so
    // the URL and host rules stay testable on a plain JVM.
    private static final int BG = 0xFF080A0F;
    private static final int BAR = 0xFF11141B;
    private static final int TEXT = 0xFFE6EAF2;
    private static final int MUTED = 0xFF9AA4B2;
    private static final int ACCENT = 0xFF3B82F6;

    private static final String MOBILE_USER_AGENT =
            "Mozilla/5.0 (Linux; Android " + Build.VERSION.RELEASE + "; " + Build.MODEL + ") "
                    + "AppleWebKit/537.36 (KHTML, like Gecko) "
                    + "Chrome/120.0.0.0 Mobile Safari/537.36";

    private final Handler handler = new Handler(Looper.getMainLooper());
    private WebView webView;
    private TextView status;
    private String query = "";
    private String engine = "bing";
    private int max = 5;
    private boolean extracting;
    private boolean finished;
    private long deadlineAt;
    private Runnable pollRunnable;
    private String extractScript;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        Intent intent = getIntent();
        query = intent == null ? "" : intent.getStringExtra(EXTRA_QUERY);
        query = query == null ? "" : query.trim();
        engine = normalizeEngine(intent == null ? null : intent.getStringExtra(EXTRA_ENGINE));
        max = intent == null ? 5 : intent.getIntExtra(EXTRA_MAX, 5);
        if (engine == null || query.isEmpty()
                || query.codePointCount(0, query.length()) > MAX_QUERY_CHARS) {
            cancel(getString(R.string.search_invalid_request));
            return;
        }
        if (max < 1 || max > MAX_RESULTS) max = 5;
        // One absolute deadline: navigating through consent pages must not
        // restart the budget, or the page's own timeout would fire first.
        deadlineAt = SystemClock.elapsedRealtime() + EXTRACT_TIMEOUT_MS;
        build();
        loadSearch();
    }

    private void build() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        setContentView(root);

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(14), dp(6), dp(6), dp(6));
        bar.setBackgroundColor(BAR);
        root.addView(bar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView title = new TextView(this);
        title.setText(getString(R.string.search_title) + " · " + query);
        title.setTextColor(TEXT);
        title.setTextSize(16);
        title.setSingleLine(true);
        title.setEllipsize(TextUtils.TruncateAt.END);
        bar.addView(title, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        Button extract = new Button(this);
        extract.setText(R.string.search_extract);
        extract.setAllCaps(false);
        extract.setTextColor(ACCENT);
        extract.setBackgroundColor(Color.TRANSPARENT);
        extract.setOnClickListener(v -> requestExtract());
        bar.addView(extract);

        Button close = new Button(this);
        close.setText(R.string.search_close);
        close.setAllCaps(false);
        close.setTextColor(MUTED);
        close.setBackgroundColor(Color.TRANSPARENT);
        close.setOnClickListener(v -> cancel(getString(R.string.search_cancelled)));
        bar.addView(close);

        status = new TextView(this);
        status.setText(R.string.search_loading);
        status.setTextColor(MUTED);
        status.setTextSize(13);
        status.setPadding(dp(14), dp(4), dp(14), dp(6));
        root.addView(status);

        ProgressBar progress = new ProgressBar(this, null,
                android.R.attr.progressBarStyleHorizontal);
        progress.setIndeterminate(true);
        root.addView(progress, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(2)));

        webView = new WebView(this);
        webView.setBackgroundColor(Color.WHITE);
        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);
        configure(webView);
        root.addView(webView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        TextView hint = new TextView(this);
        hint.setText(R.string.search_hint);
        hint.setTextColor(MUTED);
        hint.setTextSize(12);
        hint.setPadding(dp(14), dp(6), dp(14), dp(8));
        root.addView(hint);

        setTitle(getString(R.string.search_title));
    }

    private void configure(WebView view) {
        WebSettings settings = view.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setSupportMultipleWindows(false);
        settings.setJavaScriptCanOpenWindowsAutomatically(false);
        settings.setMediaPlaybackRequiresUserGesture(true);
        settings.setLoadWithOverviewMode(true);
        settings.setUseWideViewPort(true);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);
        settings.setUserAgentString(MOBILE_USER_AGENT);

        CookieManager cookies = CookieManager.getInstance();
        cookies.setAcceptCookie(true);
        cookies.setAcceptThirdPartyCookies(view, false);

        view.setWebChromeClient(new WebChromeClient());
        view.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView target, WebResourceRequest request) {
                // Subframe navigations (ads, tracking pixels) are not user
                // clicks; letting them reach handleUri would open a browser.
                if (request != null && !request.isForMainFrame()) return false;
                return handleUri(request == null ? null : request.getUrl());
            }

            @Override
            @SuppressWarnings("deprecation")
            public boolean shouldOverrideUrlLoading(WebView target, String url) {
                return handleUri(url == null ? null : Uri.parse(url));
            }

            @Override
            public void onPageStarted(WebView target, String url, android.graphics.Bitmap favicon) {
                if (finished) return;
                status.setText(R.string.search_loading);
                stopPolling();
            }

            @Override
            public void onPageFinished(WebView target, String url) {
                if (finished) return;
                status.setText(R.string.search_waiting);
                startPolling(SETTLE_MS);
            }

            @Override
            public void onReceivedError(WebView target, WebResourceRequest request,
                                        WebResourceError error) {
                if (finished || request == null || !request.isForMainFrame()) return;
                status.setText(getString(R.string.search_load_failed,
                        error == null || error.getDescription() == null
                                ? "" : error.getDescription()));
            }

            @Override
            public boolean onRenderProcessGone(WebView target, RenderProcessGoneDetail detail) {
                cancel(getString(R.string.search_crashed));
                return true;
            }
        });
    }

    private void loadSearch() {
        String url = searchUrl(engine, query);
        if (url == null) {
            cancel(getString(R.string.search_invalid_request));
            return;
        }
        webView.loadUrl(url);
    }

    private boolean handleUri(Uri uri) {
        if (uri == null) return true;
        String scheme = uri.getScheme();
        if (!"http".equalsIgnoreCase(scheme) && !"https".equalsIgnoreCase(scheme)) return true;
        if (isSearchHost(engine, uri.getHost())) return false;
        if (!"https".equalsIgnoreCase(scheme)) {
            Toast.makeText(this, R.string.search_external_blocked, Toast.LENGTH_SHORT).show();
            return true;
        }
        try {
            startActivity(new Intent(Intent.ACTION_VIEW, uri));
        } catch (ActivityNotFoundException error) {
            Toast.makeText(this, R.string.search_external_blocked, Toast.LENGTH_SHORT).show();
        }
        return true;
    }

    private void requestExtract() {
        if (finished) return;
        if (webView == null) {
            cancel(getString(R.string.search_crashed));
            return;
        }
        extract(true);
    }

    private void startPolling(long delayMs) {
        stopPolling();
        pollRunnable = new Runnable() {
            @Override
            public void run() {
                if (finished) return;
                if (SystemClock.elapsedRealtime() > deadlineAt) {
                    cancel(getString(R.string.search_timeout));
                    return;
                }
                extract(false);
                handler.postDelayed(this, POLL_MS);
            }
        };
        handler.postDelayed(pollRunnable, delayMs);
    }

    private void stopPolling() {
        if (pollRunnable != null) {
            handler.removeCallbacks(pollRunnable);
            pollRunnable = null;
        }
    }

    private void extract(boolean manual) {
        if (finished || extracting || webView == null) return;
        String script = extractScript();
        if (script.isEmpty()) {
            cancel(getString(R.string.search_failed));
            return;
        }
        extracting = true;
        String js = script
                .replace("__ENGINE__", JSONObject.quote(engine))
                .replace("__MAX__", Integer.toString(max));
        webView.evaluateJavascript(js, value -> {
            extracting = false;
            if (finished) return;
            try {
                JSONObject payload = new JSONObject(decodeJavascriptString(value));
                String state = payload.optString("state", "empty");
                JSONArray results = payload.optJSONArray("results");
                if ("results".equals(state) && results != null && results.length() > 0) {
                    finishWithResults(payload);
                } else if ("verify".equals(state)) {
                    status.setText(R.string.search_verify);
                    if (manual) Toast.makeText(this, R.string.search_verify, Toast.LENGTH_LONG).show();
                } else {
                    JSONObject debug = payload.optJSONObject("debug");
                    int blocks = debug == null ? 0 : debug.optInt("blocks", 0);
                    int anchors = debug == null ? 0 : debug.optInt("anchors", 0);
                    if (manual) {
                        status.setText(getString(R.string.search_no_results_debug, blocks, anchors));
                        Toast.makeText(this, R.string.search_no_results, Toast.LENGTH_SHORT).show();
                    } else {
                        status.setText(R.string.search_waiting);
                    }
                }
            } catch (JSONException | RuntimeException error) {
                status.setText(R.string.search_waiting);
                if (manual) Toast.makeText(this, R.string.search_no_results, Toast.LENGTH_SHORT).show();
            }
        });
    }

    private static String decodeJavascriptString(String value) throws JSONException {
        if (value == null) return "";
        if (value.length() >= 2 && value.charAt(0) == '"') {
            return new JSONArray("[" + value + "]").getString(0);
        }
        return value;
    }

    private void finishWithResults(JSONObject payload) {
        if (finished) return;
        JSONObject result = new JSONObject();
        try {
            result.put("query", query);
            result.put("engine", engine);
            result.put("source", "mobile-browser");
            result.put("untrusted", true);
            result.put("results", payload.optJSONArray("results"));
        } catch (JSONException error) {
            cancel(getString(R.string.search_failed));
            return;
        }
        finished = true;
        stopPolling();
        setResult(RESULT_OK, new Intent().putExtra(EXTRA_RESULT, result.toString()));
        finish();
    }

    private void cancel(String message) {
        if (finished) return;
        finished = true;
        stopPolling();
        setResult(RESULT_CANCELED, new Intent().putExtra(EXTRA_ERROR, message));
        finish();
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
            return;
        }
        cancel(getString(R.string.search_cancelled));
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (webView != null && !finished) {
            webView.onResume();
            extract(false);
        }
    }

    @Override
    protected void onPause() {
        if (webView != null) webView.onPause();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        finished = true;
        stopPolling();
        if (webView != null) {
            webView.stopLoading();
            webView.loadUrl("about:blank");
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }

    static String normalizeEngine(String value) {
        String engine = value == null ? "" : value.trim().toLowerCase(Locale.US);
        if (engine.isEmpty()) return "bing";
        switch (engine) {
            case "google":
            case "bing":
            case "baidu":
            case "duckduckgo":
                return engine;
            default:
                return null;
        }
    }

    static String searchUrl(String engine, String query) {
        String normalized = normalizeEngine(engine);
        if (normalized == null || query == null || query.trim().isEmpty()) return null;
        String encoded;
        try {
            encoded = URLEncoder.encode(query.trim(), "UTF-8");
        } catch (UnsupportedEncodingException impossible) {
            throw new AssertionError(impossible);
        }
        switch (normalized) {
            case "google":
                return "https://www.google.com/search?q=" + encoded;
            case "bing":
                return "https://www.bing.com/search?q=" + encoded;
            case "baidu":
                return "https://www.baidu.com/s?wd=" + encoded;
            case "duckduckgo":
                return "https://duckduckgo.com/?q=" + encoded;
            default:
                return null;
        }
    }

    static boolean isSearchHost(String engine, String host) {
        String normalized = normalizeEngine(engine);
        if (normalized == null || host == null) return false;
        String value = host.toLowerCase(Locale.US);
        switch (normalized) {
            case "google":
                // Google serves consent and accounts from google.com, and
                // country domains such as google.co.jp. Match those exact
                // shapes so a lookalike like www.google.evil.example cannot
                // stay in this WebView.
                return value.equals("google.com")
                        || value.endsWith(".google.com")
                        || value.matches("^(www\\.)?google\\.[a-z]{2,3}(\\.[a-z]{2})?$");
            case "bing":
                return value.equals("bing.com") || value.endsWith(".bing.com");
            case "baidu":
                return value.equals("baidu.com") || value.endsWith(".baidu.com");
            case "duckduckgo":
                return value.equals("duckduckgo.com") || value.endsWith(".duckduckgo.com")
                        || value.equals("duck.com") || value.endsWith(".duck.com");
            default:
                return false;
        }
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    // Loaded from res/raw so the extraction rules stay plain JavaScript that
    // can be syntax-checked outside the APK build, instead of a Java string.
    private String extractScript() {
        if (extractScript != null) return extractScript;
        try (InputStream in = getResources().openRawResource(R.raw.search_extract);
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) > 0) out.write(buffer, 0, read);
            extractScript = out.toString("UTF-8");
        } catch (IOException error) {
            extractScript = "";
        }
        return extractScript;
    }
}
