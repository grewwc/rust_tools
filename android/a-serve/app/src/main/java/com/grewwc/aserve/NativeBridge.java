package com.grewwc.aserve;

import android.app.Activity;
import android.app.AlertDialog;
import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.ContentResolver;
import android.content.ContentUris;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.database.Cursor;
import android.net.Uri;
import android.provider.AlarmClock;
import android.provider.CalendarContract;
import android.provider.ContactsContract;
import android.provider.MediaStore;
import android.telephony.PhoneNumberUtils;
import android.webkit.JavascriptInterface;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.TimeZone;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Narrow native surface for the server-owned page.
 *
 * <p>Only allowlisted actions exist. The page never gets a generic
 * {@code startActivity}. Manual actions use native confirmation; chat-initiated
 * actions may skip it, except direct calls, which always require confirmation.
 * Calls are rejected unless the loaded page
 * still belongs to the configured server origin.</p>
 */
final class NativeBridge {
    static final String NAME = "AServeNative";

    private static final String[] ACTIONS = {"now", "calendar_read", "calendar_add", "sms",
            "alarm", "alarm_list", "timer", "note", "note_create", "note_list", "note_update",
            "contacts_read", "contact_add", "call_history", "call", "phone_search",
            "app_list", "app_open", "photo_index", "photo_search", "photo_open",
            "file_import", "file_read", "file_save", "location",
            "screen_capture", "screen_stop", "screen_overlay_settings", "camera_capture",
            "camera_stop", "rebind_ime"};
    private static final int MAX_NOTE_CHARS = 3000;
    private static final int MAX_TITLE_CHARS = 200;
    private static final int MAX_PLACE_CHARS = 200;
    private static final int MAX_DESCRIPTION_CHARS = 1000;
    private static final int MAX_SMS_CHARS = 1000;
    private static final int MAX_CONTACT_CHARS = 120;
    private static final int MAX_NUMBER_CHARS = 40;
    private static final int MAX_APP_PACKAGE_CHARS = 255;
    private static final long MAX_SAFE_MEDIA_ID = 9_007_199_254_740_991L;
    private static final String LOCAL_DATE_TIME = "yyyy-MM-dd'T'HH:mm";
    private final MainActivity activity;
    private boolean foreground;
    private ForegroundLocation locationRequest;
    private final java.util.concurrent.ExecutorService notebookWorker = java.util.concurrent.Executors.newSingleThreadExecutor();
    private final java.util.concurrent.ExecutorService phoneWorker = java.util.concurrent.Executors.newSingleThreadExecutor();
    private final PhotoModelManager photoModels;
    private LocalNotebook notebook;
    private PhotoSemanticIndex photoIndex;
    private volatile AtomicBoolean photoCancellation;
    private volatile CountDownLatch photoConfirmation;
    private volatile AlertDialog photoDialog;

    NativeBridge(MainActivity activity) {
        this.activity = activity;
        photoModels = new PhotoModelManager(activity.getApplicationContext());
    }

    void onHostResume() {
        foreground = true;
    }

    void onHostPause() {
        foreground = false;
        if (locationRequest != null) locationRequest.cancel();
    }

    void close() {
        onHostPause();
        AtomicBoolean cancellation = photoCancellation;
        if (cancellation != null) cancellation.set(true);
        CountDownLatch confirmation = photoConfirmation;
        if (confirmation != null) confirmation.countDown();
        AlertDialog dialog = photoDialog;
        if (dialog != null) activity.runOnUiThread(dialog::dismiss);
        notebookWorker.execute(() -> { if (notebook != null) notebook.close(); });
        notebookWorker.shutdown();
        phoneWorker.execute(() -> {
            if (photoIndex != null) {
                try { photoIndex.close(); } catch (java.io.IOException ignored) { }
            }
        });
        phoneWorker.shutdown();
    }

    /** Feature detection for the page; returns a JSON string. */
    @JavascriptInterface
    public String capabilities() {
        JSONObject caps = new JSONObject();
        try {
            caps.put("version", BuildConfig.VERSION_NAME);
            caps.put("actions", new JSONArray(java.util.Arrays.asList(ACTIONS)));
            caps.put("features", new JSONArray(java.util.Arrays.asList("alarm_weekdays", "precise_location",
                    "screen_manual_trigger", "screen_bubble", "camera_preview")));
        } catch (JSONException ignored) {
            // Both keys are constants; serialization cannot fail here.
        }
        return caps.toString();
    }

    /**
     * Entry point for every native action.
     *
     * @param requestId caller-generated id echoed in {@code AServeNativeResult}
     * @param action    one of the action names returned by {@link #capabilities()}
     * @param payloadJson JSON object with action-specific fields
     */
    @JavascriptInterface
    public void perform(String requestId, String action, String payloadJson) {
        if (requestId == null || requestId.isEmpty() || requestId.length() > 80) return;
        // WebView bridge calls arrive off the UI thread; every check below
        // touches the WebView or opens a dialog, so move the whole dispatch.
        activity.runOnUiThread(() -> performOnUi(requestId, action, payloadJson));
    }

    private void performOnUi(String requestId, String action, String payloadJson) {
        if (!activity.isTrustedPage()) {
            activity.deliverNativeResult(requestId, false, "untrusted page", null);
            return;
        }
        if (!java.util.Arrays.asList(ACTIONS).contains(action)) {
            activity.deliverNativeResult(requestId, false, "unknown action", null);
            return;
        }
        if ("location".equals(action) && !foreground) {
            activity.deliverNativeResult(requestId, false, "app not in foreground", null);
            return;
        }
        JSONObject payload;
        try {
            String raw = payloadJson == null || payloadJson.trim().isEmpty() ? "{}" : payloadJson;
            payload = new JSONObject(raw);
        } catch (JSONException error) {
            activity.deliverNativeResult(requestId, false, "invalid payload", null);
            return;
        }
        // Every runtime permission is requested here, before the action runs,
        // so each handler can assume it is already granted. A denial answers the
        // waiting tool with a concrete reason instead of failing mid-action.
        String[] missing = missingPermissions(action, payload);
        if (missing.length > 0) {
            activity.requestNativePermissions(requestId, action, payloadJson, missing);
            return;
        }
        resumeAfterPermission(requestId, action, payloadJson);
    }

    /** Rerun an action once the activity has granted its runtime permissions. */
    void resumeAfterPermission(String requestId, String action, String payloadJson) {
        if (!activity.isTrustedPage()) {
            activity.deliverNativeResult(requestId, false, "untrusted page", null);
            return;
        }
        JSONObject payload;
        try {
            String raw = payloadJson == null || payloadJson.trim().isEmpty() ? "{}" : payloadJson;
            payload = new JSONObject(raw);
        } catch (JSONException error) {
            activity.deliverNativeResult(requestId, false, "invalid payload", null);
            return;
        }
        // `auto` is set only by the serve page when the agent's phone_action
        // tool requested the action. The share sheet (notes) and the messaging
        // composer (sms) remain user-visible gates; alarms and calendar events
        // are created without a dialog because the user already asked for them
        // in the chat.
        boolean auto = payload.optBoolean("auto", false);
        if ("alarm".equals(action)) {
            requestAlarm(requestId, payload, auto);
        } else if ("alarm_list".equals(action)) {
            showAlarmList(requestId);
        } else if ("timer".equals(action)) {
            requestTimer(requestId, payload, auto);
        } else if ("note".equals(action)) {
            requestNote(requestId, payload, auto);
        } else if ("location".equals(action)) {
            requestLocation(requestId, auto);
        } else if ("now".equals(action)) {
            deliverNow(requestId);
        } else if ("calendar_read".equals(action)) {
            readCalendar(requestId, payload);
        } else if ("calendar_add".equals(action)) {
            addCalendarEvent(requestId, payload, auto);
        } else if ("sms".equals(action)) {
            composeSms(requestId, payload, auto);
        } else if ("contacts_read".equals(action) || "call_history".equals(action)) {
            readPhoneRecords(requestId, action, payload);
        } else if ("contact_add".equals(action)) {
            requestContactAdd(requestId, payload);
        } else if ("call".equals(action)) {
            requestCall(requestId, payload);
        } else if ("app_list".equals(action)) {
            listLaunchableApps(requestId, payload);
        } else if ("app_open".equals(action)) {
            openLaunchableApp(requestId, payload);
        } else if ("photo_open".equals(action)) {
            openPhoto(requestId, payload);
        } else if ("photo_index".equals(action) || "photo_search".equals(action)) {
            performPhotoAction(requestId, action, payload);
        } else if ("note_create".equals(action) || "note_list".equals(action) || "note_update".equals(action)) {
            String origin = activity.nativeOrigin();
            payload.remove("action");
            payload.remove("auto");
            notebookWorker.execute(() -> {
                try {
                    if (notebook == null) notebook = new LocalNotebook(activity.getApplicationContext());
                    JSONObject result = notebook.execute(origin, requestId, action, payload);
                    activity.runOnUiThread(() -> {
                        if (origin.equals(activity.nativeOrigin()) && activity.isTrustedPage())
                            activity.deliverNativeResult(requestId, true, null, result);
                    });
                } catch (Exception error) {
                    activity.runOnUiThread(() -> {
                        if (origin.equals(activity.nativeOrigin()) && activity.isTrustedPage())
                            activity.deliverNativeResult(requestId, false, error.getMessage(), null);
                    });
                }
            });
        } else if ("file_import".equals(action) || "file_read".equals(action) || "file_save".equals(action)) {
            activity.performDocumentAction(requestId, action, payload);
        } else if ("phone_search".equals(action)) {
            activity.performSearch(requestId, payload);
        } else if ("screen_capture".equals(action) || "screen_stop".equals(action)) {
            activity.performScreen(requestId, "screen_stop".equals(action));
        } else if ("camera_capture".equals(action) || "camera_stop".equals(action)) {
            activity.performCamera(requestId, "camera_stop".equals(action));
        } else if ("screen_overlay_settings".equals(action)) {
            activity.openOverlaySettings(requestId);
        } else if ("rebind_ime".equals(action)) {
            // The page asks for this when the keyboard appears after the app was
            // away: the IME can still be attached to a connection that no longer
            // reaches the page, and a keyboard in that state types into nothing.
            activity.rebindIme();
            activity.deliverNativeResult(requestId, true, null, null);
        } else {
            activity.deliverNativeResult(requestId, false, "unknown action", null);
        }
    }

    /**
     * Runtime permissions an action needs but the app does not hold yet.
     *
     * <p>Files no longer expose a per-action permission list elsewhere, so this
     * is the single place that maps an action to Android permissions.</p>
     */
    private String[] missingPermissions(String action, JSONObject payload) {
        List<String> needed = new ArrayList<>();
        if ("location".equals(action)) {
            // Request an upgrade once for installations with an old coarse-only
            // grant, then respect the user's choice without repeated prompts.
            android.content.SharedPreferences choices = activity.getSharedPreferences("location_precision", 0);
            if (activity.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
                    || (activity.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
                        && choices.getBoolean("fine_requested", false))) {
                return new String[0];
            }
            choices.edit().putBoolean("fine_requested", true).apply();
            return new String[]{Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION};
        } else if ("screen_capture".equals(action) && android.os.Build.VERSION.SDK_INT >= 33) {
            needed.add(Manifest.permission.POST_NOTIFICATIONS);
        } else if ("calendar_read".equals(action)) {
            needed.add(Manifest.permission.READ_CALENDAR);
        } else if ("calendar_add".equals(action)) {
            needed.add(Manifest.permission.READ_CALENDAR);
            needed.add(Manifest.permission.WRITE_CALENDAR);
        } else if ("sms".equals(action) && payload.optString("contact", "").trim().length() > 0) {
            needed.add(Manifest.permission.READ_CONTACTS);
        } else if ("contacts_read".equals(action)) {
            needed.add(Manifest.permission.READ_CONTACTS);
        } else if ("contact_add".equals(action)) {
            needed.add(Manifest.permission.WRITE_CONTACTS);
        } else if ("call_history".equals(action)) {
            needed.add(Manifest.permission.READ_CALL_LOG);
        } else if ("call".equals(action)) {
            needed.add(Manifest.permission.CALL_PHONE);
            if (payload.optString("contact", "").trim().length() > 0) {
                needed.add(Manifest.permission.READ_CONTACTS);
            }
        } else if ("photo_index".equals(action) || "photo_open".equals(action)) {
            needed.add(android.os.Build.VERSION.SDK_INT >= 33
                    ? Manifest.permission.READ_MEDIA_IMAGES
                    : Manifest.permission.READ_EXTERNAL_STORAGE);
        }
        List<String> missing = new ArrayList<>();
        for (String permission : needed) {
            if (activity.checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED) {
                missing.add(permission);
            }
        }
        return missing.toArray(new String[0]);
    }

    private void performPhotoAction(String requestId, String action, JSONObject payload) {
        String origin = activity.nativeOrigin();
        payload.remove("action");
        payload.remove("auto");
        phoneWorker.execute(() -> {
            try {
                PhotoModelManager.Prepared prepared = photoModels.prepare(origin);
                PhotoModelManager.Active active;
                if (prepared.needsDownload) {
                    AtomicBoolean cancelled = new AtomicBoolean(false);
                    photoCancellation = cancelled;
                    try {
                        if (!awaitPhotoDownloadConfirmation(origin, prepared, cancelled)) {
                            throw new PhotoModelManager.CancelledException();
                        }
                        AlertDialog progress = showPhotoDownloadProgress(origin, cancelled);
                        try {
                            active = photoModels.install(origin, prepared, cancelled,
                                    (stage, completed, total) -> updatePhotoProgress(
                                            progress, cancelled, stage, completed, total),
                                    PhotoSemanticIndex::validateModel);
                        } finally {
                            dismissPhotoDialog(progress);
                        }
                    } finally {
                        if (photoCancellation == cancelled) photoCancellation = null;
                    }
                } else {
                    if (prepared.active == null) {
                        throw new IOException("no verified photo model is installed");
                    }
                    active = prepared.active.withPrepared(prepared);
                }
                ensurePhotoIndex(active);
                JSONObject result = "photo_index".equals(action)
                        ? photoIndex.refresh(payload) : photoIndex.search(payload);
                putPhotoModelState(result, active, prepared);
                deliverPhoneWorkerResult(origin, requestId, true, null, result);
            } catch (Exception error) {
                String message = error.getMessage() == null
                        ? error.getClass().getSimpleName() : error.getMessage();
                deliverPhoneWorkerResult(origin, requestId, false, message, null);
            }
        });
    }

    private boolean awaitPhotoDownloadConfirmation(String origin,
                                                    PhotoModelManager.Prepared prepared,
                                                    AtomicBoolean cancelled)
            throws InterruptedException {
        CountDownLatch answered = new CountDownLatch(1);
        AtomicBoolean approved = new AtomicBoolean(false);
        AtomicReference<AlertDialog> shown = new AtomicReference<>();
        photoConfirmation = answered;
        activity.runOnUiThread(() -> {
            if (cancelled.get() || activity.isFinishing() || activity.isDestroyed()
                    || !origin.equals(activity.nativeOrigin()) || !activity.isTrustedPage()) {
                cancelled.set(true);
                answered.countDown();
                return;
            }
            PhotoModelContract.Manifest manifest = prepared.fetched.manifest;
            String operation = prepared.active == null ? "安装" : "更新";
            AlertDialog dialog = new AlertDialog.Builder(activity)
                    .setTitle(operation + "照片搜索模型")
                    .setMessage("版本 " + manifest.version + "，下载 "
                            + formatModelBytes(manifest.totalFileSize)
                            + "。模型仅用于设备上的照片索引和搜索。")
                    .setNegativeButton("取消", (ignored, which) -> {
                        cancelled.set(true);
                        answered.countDown();
                    })
                    .setPositiveButton("下载", (ignored, which) -> {
                        if (origin.equals(activity.nativeOrigin()) && activity.isTrustedPage()) {
                            approved.set(true);
                        } else {
                            cancelled.set(true);
                        }
                        answered.countDown();
                    })
                    .setOnCancelListener(ignored -> {
                        cancelled.set(true);
                        answered.countDown();
                    })
                    .create();
            shown.set(dialog);
            photoDialog = dialog;
            dialog.show();
        });
        boolean responded = answered.await(5, TimeUnit.MINUTES);
        if (!responded) cancelled.set(true);
        AlertDialog dialog = shown.get();
        if (!responded) dismissPhotoDialog(dialog);
        if (photoConfirmation == answered) photoConfirmation = null;
        if (photoDialog == dialog) photoDialog = null;
        return responded && approved.get() && !cancelled.get();
    }

    private AlertDialog showPhotoDownloadProgress(String origin, AtomicBoolean cancelled)
            throws InterruptedException, IOException {
        CountDownLatch shown = new CountDownLatch(1);
        AtomicReference<AlertDialog> result = new AtomicReference<>();
        photoConfirmation = shown;
        activity.runOnUiThread(() -> {
            if (cancelled.get() || activity.isFinishing() || activity.isDestroyed()
                    || !origin.equals(activity.nativeOrigin()) || !activity.isTrustedPage()) {
                cancelled.set(true);
                shown.countDown();
                return;
            }
            AlertDialog dialog = new AlertDialog.Builder(activity)
                    .setTitle("照片搜索模型")
                    .setMessage("准备下载…")
                    .setNegativeButton("取消", (ignored, which) -> cancelled.set(true))
                    .setOnCancelListener(ignored -> cancelled.set(true))
                    .create();
            result.set(dialog);
            photoDialog = dialog;
            dialog.show();
            shown.countDown();
        });
        if (!shown.await(10, TimeUnit.SECONDS) || result.get() == null) {
            cancelled.set(true);
            throw new IOException("cannot show photo model download progress");
        }
        if (photoConfirmation == shown) photoConfirmation = null;
        return result.get();
    }

    private void updatePhotoProgress(AlertDialog dialog, AtomicBoolean cancelled,
                                     String stage, long completed, long total) {
        String message = stage;
        if (total > 0) {
            long percent = Math.min(100, Math.max(0, completed * 100 / total));
            message += "\n" + percent + "% (" + formatModelBytes(completed)
                    + " / " + formatModelBytes(total) + ")";
        }
        String display = message;
        activity.runOnUiThread(() -> {
            if (!cancelled.get() && photoDialog == dialog && dialog.isShowing()) {
                dialog.setMessage(display);
            }
        });
    }

    private void dismissPhotoDialog(AlertDialog dialog) {
        if (dialog == null) return;
        activity.runOnUiThread(dialog::dismiss);
        if (photoDialog == dialog) photoDialog = null;
    }

    private void ensurePhotoIndex(PhotoModelManager.Active active) throws IOException {
        if (photoIndex != null && active.version.equals(photoIndex.modelVersion())) return;
        if (photoIndex != null) {
            photoIndex.close();
            photoIndex = null;
        }
        photoIndex = new PhotoSemanticIndex(activity, active.directory, active.version);
    }

    private static void putPhotoModelState(JSONObject result, PhotoModelManager.Active active,
                                           PhotoModelManager.Prepared prepared) throws JSONException {
        result.put("model_version", active.version);
        result.put("model_state", active.state);
        result.put("model_downloaded", active.downloaded);
        result.put("model_updated", "updated".equals(active.state));
        result.put("model_fallback", "installed_fallback".equals(active.state));
        result.put("model_manifest_checked", prepared.manifestChecked);
        if (active.manifestError != null) result.put("model_manifest_error", active.manifestError);
    }

    private static String formatModelBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format(Locale.ROOT, "%.1f KiB", bytes / 1024.0);
        return String.format(Locale.ROOT, "%.1f MiB", bytes / (1024.0 * 1024.0));
    }

    private void showAlarmList(String requestId) {
        activity.runOnUiThread(() -> {
            try {
                activity.startActivity(new Intent(AlarmClock.ACTION_SHOW_ALARMS));
                JSONObject data = new JSONObject();
                data.put("action", "alarm_list");
                data.put("dispatched", true);
                activity.deliverNativeResult(requestId, true, null, data);
            } catch (ActivityNotFoundException error) {
                activity.deliverNativeResult(requestId, false, "no alarm app", null);
            } catch (JSONException error) {
                activity.deliverNativeResult(requestId, false, "cannot encode alarm result", null);
            }
        });
    }

    private void listLaunchableApps(String requestId, JSONObject payload) {
        String origin = activity.nativeOrigin();
        phoneWorker.execute(() -> {
            try {
                String query = strictString(payload, "query", "", 200).trim().toLowerCase(Locale.ROOT);
                int max = strictInt(payload, "max", 10, 1, 20);
                int offset = strictInt(payload, "offset", 0, 0, 100000);
                List<JSONObject> matching = new ArrayList<>();
                for (JSONObject entry : launchableApps()) {
                    String label = entry.getString("label");
                    String packageName = entry.getString("package");
                    if (query.isEmpty() || label.toLowerCase(Locale.ROOT).contains(query)
                            || packageName.toLowerCase(Locale.ROOT).contains(query)) {
                        matching.add(entry);
                    }
                }
                JSONArray apps = new JSONArray();
                int end = Math.min(matching.size(), offset + max);
                for (int index = Math.min(offset, matching.size()); index < end; index++) {
                    apps.put(matching.get(index));
                }
                JSONObject data = new JSONObject();
                data.put("apps", apps);
                data.put("offset", offset);
                data.put("next_offset", end);
                data.put("has_more", end < matching.size());
                data.put("total", matching.size());
                deliverPhoneWorkerResult(origin, requestId, true, null, data);
            } catch (Exception error) {
                deliverPhoneWorkerResult(origin, requestId, false, error.getMessage(), null);
            }
        });
    }

    private void openLaunchableApp(String requestId, JSONObject payload) {
        try {
            String packageName = strictString(payload, "package", null, MAX_APP_PACKAGE_CHARS);
            if (!isAndroidPackage(packageName)) {
                activity.deliverNativeResult(requestId, false, "invalid Android package name", null);
                return;
            }
            JSONObject selected = null;
            for (JSONObject entry : launchableApps()) {
                if (packageName.equals(entry.getString("package"))) {
                    selected = entry;
                    break;
                }
            }
            if (selected == null) {
                activity.deliverNativeResult(requestId, false, "app is not installed or has no launcher activity", null);
                return;
            }
            Intent launch = activity.getPackageManager().getLaunchIntentForPackage(packageName);
            if (launch == null) {
                activity.deliverNativeResult(requestId, false, "app has no launcher activity", null);
                return;
            }
            JSONObject result = selected;
            activity.runOnUiThread(() -> {
                try {
                    activity.startActivity(launch);
                    result.remove("activity");
                    result.put("dispatched", true);
                    activity.deliverNativeResult(requestId, true, null, result);
                } catch (ActivityNotFoundException | SecurityException | JSONException error) {
                    activity.deliverNativeResult(requestId, false, "cannot open app", null);
                }
            });
        } catch (Exception error) {
            activity.deliverNativeResult(requestId, false, error.getMessage(), null);
        }
    }

    private void openPhoto(String requestId, JSONObject payload) {
        Object rawMediaId = payload.opt("media_id");
        if (!(rawMediaId instanceof Number)) {
            activity.deliverNativeResult(requestId, false, "media_id must be a positive integer", null);
            return;
        }
        double numericMediaId = ((Number) rawMediaId).doubleValue();
        long mediaId = ((Number) rawMediaId).longValue();
        if (!Double.isFinite(numericMediaId) || numericMediaId != mediaId
                || mediaId <= 0 || mediaId > MAX_SAFE_MEDIA_ID) {
            activity.deliverNativeResult(requestId, false, "media_id must be a positive integer", null);
            return;
        }

        Uri uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, mediaId);
        try {
            String mimeType = activity.getContentResolver().getType(uri);
            if (mimeType == null || !mimeType.startsWith("image/")) {
                activity.deliverNativeResult(requestId, false,
                        "photo is unavailable or no longer exists", null);
                return;
            }
            Intent view = new Intent(Intent.ACTION_VIEW)
                    .setDataAndType(uri, mimeType)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            if (view.resolveActivity(activity.getPackageManager()) == null) {
                activity.deliverNativeResult(requestId, false,
                        "no app can open this photo", null);
                return;
            }
            activity.startActivity(view);
            JSONObject result = new JSONObject();
            result.put("media_id", mediaId);
            result.put("content_uri", uri.toString());
            result.put("dispatched", true);
            activity.deliverNativeResult(requestId, true, null, result);
        } catch (ActivityNotFoundException | SecurityException | JSONException error) {
            activity.deliverNativeResult(requestId, false, "cannot open photo", null);
        }
    }

    private List<JSONObject> launchableApps() throws JSONException {
        Intent launcher = new Intent(Intent.ACTION_MAIN);
        launcher.addCategory(Intent.CATEGORY_LAUNCHER);
        PackageManager manager = activity.getPackageManager();
        List<ResolveInfo> resolved = manager.queryIntentActivities(launcher, PackageManager.MATCH_ALL);
        List<JSONObject> entries = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();
        for (ResolveInfo info : resolved) {
            if (info.activityInfo == null || info.activityInfo.packageName == null
                    || !isAndroidPackage(info.activityInfo.packageName)
                    || !seen.add(info.activityInfo.packageName)) continue;
            String packageName = info.activityInfo.packageName;
            CharSequence loaded = info.loadLabel(manager);
            String label = loaded == null ? packageName : loaded.toString().trim();
            if (label.isEmpty()) label = packageName;
            JSONObject entry = new JSONObject();
            entry.put("label", label);
            entry.put("package", packageName);
            entries.add(entry);
        }
        entries.sort((left, right) -> {
            int byLabel = left.optString("label").compareToIgnoreCase(right.optString("label"));
            return byLabel != 0 ? byLabel
                    : left.optString("package").compareTo(right.optString("package"));
        });
        return entries;
    }

    private static String strictString(JSONObject payload, String key, String fallback, int maxChars)
            throws JSONException {
        if (!payload.has(key) || payload.isNull(key)) return fallback;
        Object raw = payload.get(key);
        if (!(raw instanceof String) || ((String) raw).length() > maxChars)
            throw new JSONException("invalid '" + key + "'");
        return (String) raw;
    }

    private static int strictInt(JSONObject payload, String key, int fallback, int min, int max)
            throws JSONException {
        if (!payload.has(key) || payload.isNull(key)) return fallback;
        Object raw = payload.get(key);
        if (!(raw instanceof Number)) throw new JSONException("invalid '" + key + "'");
        double value = ((Number) raw).doubleValue();
        if (!Double.isFinite(value) || value != Math.rint(value) || value < min || value > max)
            throw new JSONException("invalid '" + key + "'");
        return (int) value;
    }

    static boolean isAndroidPackage(String value) {
        if (value == null || value.isEmpty() || value.length() > MAX_APP_PACKAGE_CHARS) return false;
        String[] segments = value.split("\\.", -1);
        if (segments.length < 2) return false;
        for (String segment : segments) {
            if (segment.isEmpty() || !isAsciiLetter(segment.charAt(0))) return false;
            for (int index = 1; index < segment.length(); index++) {
                char ch = segment.charAt(index);
                if (!isAsciiLetter(ch) && (ch < '0' || ch > '9') && ch != '_') return false;
            }
        }
        return true;
    }

    private static boolean isAsciiLetter(char ch) {
        return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
    }

    private void requestAlarm(String requestId, JSONObject payload, boolean auto) {
        final ClockRequest clock;
        try {
            Object weekdays = payload.opt("weekdays");
            if (weekdays instanceof JSONArray) {
                List<Object> values = new ArrayList<>();
                JSONArray array = (JSONArray) weekdays;
                for (int i = 0; i < array.length(); i++) values.add(array.opt(i));
                weekdays = values;
            }
            clock = ClockRequest.alarm(payload.opt("hour"), payload.opt("minute"),
                    payload.opt("label"), weekdays);
        } catch (IllegalArgumentException invalid) {
            activity.deliverNativeResult(requestId, false, invalid.getMessage(), null);
            return;
        }
        String detail = String.format(Locale.getDefault(), "设置闹钟 %02d:%02d", clock.hour, clock.minute);
        if (!clock.label.isEmpty()) detail += "\n标签：" + clock.label;
        if (!clock.weekdays.isEmpty()) detail += "\n重复日期（1=周一，7=周日）：" + clock.weekdays;
        String message = detail + "\n\n确认后向系统时钟发送请求；是否创建成功请在时钟中确认。";
        if (auto) {
            dispatchClock(requestId, clock, false);
            return;
        }
        confirm(requestId, "设置闹钟", message, () -> dispatchClock(requestId, clock, false));
    }

    private void requestTimer(String requestId, JSONObject payload, boolean auto) {
        final ClockRequest clock;
        try {
            clock = ClockRequest.timer(payload.opt("seconds"), payload.opt("label"));
        } catch (IllegalArgumentException invalid) {
            activity.deliverNativeResult(requestId, false, invalid.getMessage(), null);
            return;
        }
        if (auto) {
            dispatchClock(requestId, clock, true);
            return;
        }
        confirm(requestId, "设置计时器", "时长：" + clock.seconds + " 秒\n标签：" + clock.label
                + "\n\n确认后向系统时钟发送请求；是否启动请在时钟中确认。",
                () -> dispatchClock(requestId, clock, true));
    }

    private void dispatchClock(String requestId, ClockRequest clock, boolean timer) {
        if (!activity.isTrustedPage()) {
            activity.deliverNativeResult(requestId, false, "untrusted page", null);
            return;
        }
        Intent intent = new Intent(timer ? AlarmClock.ACTION_SET_TIMER : AlarmClock.ACTION_SET_ALARM)
                .putExtra(AlarmClock.EXTRA_SKIP_UI, true);
        if (timer) {
            intent.putExtra(AlarmClock.EXTRA_LENGTH, clock.seconds);
        } else {
            intent.putExtra(AlarmClock.EXTRA_HOUR, clock.hour).putExtra(AlarmClock.EXTRA_MINUTES, clock.minute);
            if (!clock.weekdays.isEmpty()) intent.putIntegerArrayListExtra(AlarmClock.EXTRA_DAYS, clock.calendarDays());
        }
        if (!clock.label.isEmpty()) intent.putExtra(AlarmClock.EXTRA_MESSAGE, clock.label);
        boolean skipUi = true;
        try {
            try {
                activity.startActivity(intent);
            } catch (ActivityNotFoundException first) {
                // Some clock apps reject the skip-UI flag; retry with the picker.
                skipUi = false;
                intent.putExtra(AlarmClock.EXTRA_SKIP_UI, false);
                activity.startActivity(intent);
            }
            JSONObject data = new JSONObject();
            data.put("action", timer ? "timer" : "alarm");
            data.put("status", "dispatched");
            data.put("skip_ui", skipUi);
            data.put("label", clock.label);
            if (timer) data.put("seconds", clock.seconds);
            else {
                data.put("hour", clock.hour);
                data.put("minute", clock.minute);
                data.put("weekdays", new JSONArray(clock.weekdays));
            }
            Toast.makeText(activity, timer ? "计时器请求已发送，请在时钟中确认" : "闹钟请求已发送，请在时钟中确认",
                    Toast.LENGTH_SHORT).show();
            activity.deliverNativeResult(requestId, true, null, data);
        } catch (ActivityNotFoundException error) {
            activity.deliverNativeResult(requestId, false, timer ? "no timer app" : "no alarm app", null);
        } catch (SecurityException error) {
            activity.deliverNativeResult(requestId, false, "clock permission denied", null);
        } catch (JSONException error) {
            activity.deliverNativeResult(requestId, false, "clock response unavailable; request may have been dispatched", null);
        }
    }

    private void requestNote(String requestId, JSONObject payload, boolean auto) {
        String title = clamp(payload.optString("title", ""), MAX_TITLE_CHARS).trim();
        String text = payload.optString("text", "").trim();
        if (text.isEmpty()) {
            activity.deliverNativeResult(requestId, false, "empty note", null);
            return;
        }
        if (text.length() > MAX_NOTE_CHARS) {
            activity.deliverNativeResult(requestId, false, "note too long", null);
            return;
        }
        String preview = text.length() > 280 ? text.substring(0, 280) + "…" : text;
        String message = "标题：" + (title.isEmpty() ? "（无）" : title) + "\n\n" + preview +
                "\n\n确认后打开系统分享面板；请选择笔记应用并保存。";
        if (auto) {
            shareNote(requestId, title, text);
            return;
        }
        String finalTitle = title;
        String finalText = text;
        confirm(requestId, "保存为笔记", message,
                () -> shareNote(requestId, finalTitle, finalText));
    }

    private void shareNote(String requestId, String title, String text) {
        Intent send = new Intent(Intent.ACTION_SEND).setType("text/plain");
        if (!title.isEmpty()) send.putExtra(Intent.EXTRA_SUBJECT, title);
        send.putExtra(Intent.EXTRA_TEXT, text);
        try {
            activity.startActivity(Intent.createChooser(send, "保存到笔记"));
            Toast.makeText(activity, "已打开分享面板，请选择笔记应用", Toast.LENGTH_SHORT).show();
            activity.deliverNativeResult(requestId, true, null, null);
        } catch (ActivityNotFoundException error) {
            activity.deliverNativeResult(requestId, false, "no note app", null);
        }
    }

    private void requestLocation(String requestId, boolean auto) {
        if (auto) {
            fetchLocation(requestId);
            return;
        }
        confirm(requestId, "使用当前位置",
                "尝试获取一次当前位置，用于天气、附近搜索等；可选择仅提供大致位置。\n\n只在前台定位，结果会附带精度和时效信息。",
                () -> fetchLocation(requestId));
    }

    /** Called by the activity after the runtime permission result, or directly. */
    void fetchLocation(String requestId) {
        activity.runOnUiThread(() -> {
            if (!foreground || activity.isFinishing() || activity.isDestroyed()) {
                activity.deliverNativeResult(requestId, false, "app not in foreground", null);
                return;
            }
            if (!activity.isTrustedPage()) {
                activity.deliverNativeResult(requestId, false, "untrusted page", null);
                return;
            }
            if (locationRequest != null) {
                activity.deliverNativeResult(requestId, false, "another location request is pending", null);
                return;
            }
            locationRequest = new ForegroundLocation(activity, (ok, error, data) -> {
                locationRequest = null;
                if (!activity.isTrustedPage()) activity.deliverNativeResult(requestId, false, "untrusted page", null);
                else activity.deliverNativeResult(requestId, ok, error, data);
            });
            locationRequest.start();
        });
    }

    /** Local wall clock of the phone; dates in this timezone drive every other action. */
    private void deliverNow(String requestId) {
        TimeZone zone = TimeZone.getDefault();
        long now = System.currentTimeMillis();
        SimpleDateFormat readable = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault());
        readable.setTimeZone(zone);
        SimpleDateFormat machine = new SimpleDateFormat(LOCAL_DATE_TIME, Locale.getDefault());
        machine.setTimeZone(zone);
        SimpleDateFormat weekday = new SimpleDateFormat("EEEE", Locale.getDefault());
        weekday.setTimeZone(zone);
        JSONObject data = new JSONObject();
        try {
            Date instant = new Date(now);
            data.put("local", readable.format(instant));
            data.put("local_iso", machine.format(instant));
            data.put("timezone", zone.getID());
            data.put("weekday", weekday.format(instant));
            data.put("epoch_ms", now);
        } catch (JSONException ignored) {
            activity.deliverNativeResult(requestId, false, "clock unavailable", null);
            return;
        }
        activity.deliverNativeResult(requestId, true, null, data);
    }

    /**
     * Upcoming events through {@code Instances}, so recurring entries appear as
     * their real occurrences rather than as a single repeating row.
     */
    private void readCalendar(String requestId, JSONObject payload) {
        int days = Math.max(1, Math.min(30, payload.optInt("days", 7)));
        int max = Math.max(1, Math.min(20, payload.optInt("max", 10)));
        TimeZone zone = TimeZone.getDefault();
        SimpleDateFormat local = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault());
        local.setTimeZone(zone);
        long start = System.currentTimeMillis();
        long end = start + days * 24L * 60L * 60L * 1000L;

        Uri.Builder builder = CalendarContract.Instances.CONTENT_URI.buildUpon();
        ContentUris.appendId(builder, start);
        ContentUris.appendId(builder, end);
        String[] projection = new String[]{
                CalendarContract.Instances.TITLE,
                CalendarContract.Instances.BEGIN,
                CalendarContract.Instances.END,
                CalendarContract.Instances.ALL_DAY,
                CalendarContract.Instances.EVENT_LOCATION,
                CalendarContract.Instances.CALENDAR_DISPLAY_NAME,
        };

        JSONArray events = new JSONArray();
        Cursor cursor = null;
        try {
            cursor = activity.getContentResolver().query(
                    builder.build(), projection, null, null,
                    CalendarContract.Instances.BEGIN + " ASC");
            if (cursor == null) {
                activity.deliverNativeResult(requestId, false, "calendar unavailable", null);
                return;
            }
            while (cursor.moveToNext() && events.length() < max) {
                JSONObject event = new JSONObject();
                event.put("title", blankIfNull(cursor.getString(0)));
                event.put("begin_local", local.format(new Date(cursor.getLong(1))));
                event.put("end_local", local.format(new Date(cursor.getLong(2))));
                event.put("all_day", cursor.getInt(3) == 1);
                event.put("location", blankIfNull(cursor.getString(4)));
                event.put("calendar", blankIfNull(cursor.getString(5)));
                events.put(event);
            }
        } catch (SecurityException error) {
            activity.deliverNativeResult(requestId, false, "calendar permission denied", null);
            return;
        } catch (JSONException error) {
            activity.deliverNativeResult(requestId, false, "calendar read failed", null);
            return;
        } finally {
            if (cursor != null) cursor.close();
        }

        JSONObject data = new JSONObject();
        try {
            data.put("events", events);
            data.put("timezone", zone.getID());
        } catch (JSONException ignored) {
            activity.deliverNativeResult(requestId, false, "calendar read failed", null);
            return;
        }
        activity.deliverNativeResult(requestId, true, null, data);
    }

    private void addCalendarEvent(String requestId, JSONObject payload, boolean auto) {
        String title = clamp(payload.optString("title", ""), MAX_TITLE_CHARS).trim();
        String beginLocal = payload.optString("begin_local", "").trim();
        int duration = payload.optInt("duration_min", 60);
        String place = clamp(payload.optString("location", ""), MAX_PLACE_CHARS).trim();
        String description = clamp(payload.optString("description", ""), MAX_DESCRIPTION_CHARS).trim();
        if (title.isEmpty()) {
            activity.deliverNativeResult(requestId, false, "empty title", null);
            return;
        }
        if (duration < 5 || duration > 600) {
            activity.deliverNativeResult(requestId, false, "invalid duration", null);
            return;
        }
        SimpleDateFormat parser = new SimpleDateFormat(LOCAL_DATE_TIME, Locale.getDefault());
        parser.setLenient(false);
        Date begin;
        try {
            begin = parser.parse(beginLocal);
        } catch (ParseException error) {
            begin = null;
        }
        if (begin == null) {
            activity.deliverNativeResult(requestId, false, "invalid start time", null);
            return;
        }
        long beginMillis = begin.getTime();
        long endMillis = beginMillis + duration * 60_000L;
        if (!auto) {
            SimpleDateFormat shown = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault());
            String message = "新建日程：" + title + "\n开始：" + shown.format(begin) +
                    "\n时长：" + duration + " 分钟";
            if (!place.isEmpty()) message += "\n地点：" + place;
            String finalTitle = title;
            String finalPlace = place;
            String finalDescription = description;
            confirm(requestId, "新建日程", message, () -> createCalendarEvent(
                    requestId, finalTitle, beginMillis, endMillis, finalPlace, finalDescription));
            return;
        }
        createCalendarEvent(requestId, title, beginMillis, endMillis, place, description);
    }

    private void createCalendarEvent(
            String requestId, String title, long beginMillis, long endMillis,
            String place, String description) {
        ContentResolver resolver = activity.getContentResolver();
        long calendarId = -1;
        String calendarName = "";
        Cursor cursor = null;
        try {
            cursor = resolver.query(
                    CalendarContract.Calendars.CONTENT_URI,
                    new String[]{
                            CalendarContract.Calendars._ID,
                            CalendarContract.Calendars.CALENDAR_DISPLAY_NAME,
                    },
                    CalendarContract.Calendars.CALENDAR_ACCESS_LEVEL + ">=?",
                    new String[]{String.valueOf(CalendarContract.Calendars.CAL_ACCESS_CONTRIBUTOR)},
                    CalendarContract.Calendars.IS_PRIMARY + " DESC");
            if (cursor != null && cursor.moveToFirst()) {
                calendarId = cursor.getLong(0);
                calendarName = blankIfNull(cursor.getString(1));
            }
        } catch (SecurityException error) {
            activity.deliverNativeResult(requestId, false, "calendar permission denied", null);
            return;
        } finally {
            if (cursor != null) cursor.close();
        }
        if (calendarId < 0) {
            activity.deliverNativeResult(requestId, false, "no writable calendar", null);
            return;
        }

        ContentValues values = new ContentValues();
        values.put(CalendarContract.Events.CALENDAR_ID, calendarId);
        values.put(CalendarContract.Events.TITLE, title);
        values.put(CalendarContract.Events.DTSTART, beginMillis);
        values.put(CalendarContract.Events.DTEND, endMillis);
        values.put(CalendarContract.Events.EVENT_TIMEZONE, TimeZone.getDefault().getID());
        if (!place.isEmpty()) values.put(CalendarContract.Events.EVENT_LOCATION, place);
        if (!description.isEmpty()) values.put(CalendarContract.Events.DESCRIPTION, description);

        Uri created;
        try {
            created = resolver.insert(CalendarContract.Events.CONTENT_URI, values);
        } catch (SecurityException error) {
            activity.deliverNativeResult(requestId, false, "calendar permission denied", null);
            return;
        } catch (IllegalArgumentException error) {
            activity.deliverNativeResult(requestId, false, "calendar rejected the event", null);
            return;
        }
        if (created == null) {
            activity.deliverNativeResult(requestId, false, "calendar rejected the event", null);
            return;
        }

        SimpleDateFormat shown = new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault());
        JSONObject data = new JSONObject();
        try {
            data.put("begin_local", shown.format(new Date(beginMillis)));
            data.put("end_local", shown.format(new Date(endMillis)));
            data.put("calendar", calendarName);
            data.put("event_id", ContentUris.parseId(created));
        } catch (JSONException | NumberFormatException ignored) {
            activity.deliverNativeResult(requestId, false, "calendar event created but unreadable", null);
            return;
        }
        Toast.makeText(activity, "已加入日历：" + title, Toast.LENGTH_SHORT).show();
        activity.deliverNativeResult(requestId, true, null, data);
    }

    private void deliverPhoneWorkerResult(String origin, String requestId, boolean ok,
                                          String error, JSONObject data) {
        activity.runOnUiThread(() -> {
            if (origin.equals(activity.nativeOrigin()) && activity.isTrustedPage())
                activity.deliverNativeResult(requestId, ok, error, data);
        });
    }

    private void readPhoneRecords(String requestId, String action, JSONObject payload) {
        String origin = activity.nativeOrigin();
        phoneWorker.execute(() -> {
            try {
                JSONObject data = "contacts_read".equals(action)
                        ? PhoneRecords.readContacts(activity.getContentResolver(), payload)
                        : PhoneRecords.readCallHistory(activity.getContentResolver(), payload);
                activity.runOnUiThread(() -> {
                    if (origin.equals(activity.nativeOrigin()) && activity.isTrustedPage())
                        activity.deliverNativeResult(requestId, true, null, data);
                });
            } catch (SecurityException error) {
                activity.runOnUiThread(() -> activity.deliverNativeResult(
                        requestId, false, "phone data permission denied", null));
            } catch (Exception error) {
                activity.runOnUiThread(() -> activity.deliverNativeResult(
                        requestId, false, "phone data unavailable", null));
            }
        });
    }

    private void requestContactAdd(String requestId, JSONObject payload) {
        String name = payload.optString("name", "").trim();
        String number = PhoneDataRules.normalizeDialable(payload.optString("number", "").trim());
        if (name.isEmpty() || name.length() > MAX_CONTACT_CHARS || number == null) {
            activity.deliverNativeResult(requestId, false, "invalid contact name or number", null);
            return;
        }
        confirm(requestId, "添加联系人", "姓名：" + name + "\n号码：" + number
                + "\n\n确认后会写入手机通讯录。", () -> insertContact(requestId, name, number));
    }

    private void insertContact(String requestId, String name, String number) {
        String origin = activity.nativeOrigin();
        phoneWorker.execute(() -> {
            try {
                JSONObject data = PhoneRecords.addContact(
                        activity.getContentResolver(), name, number);
                activity.runOnUiThread(() -> {
                    if (!origin.equals(activity.nativeOrigin()) || !activity.isTrustedPage()) return;
                    Toast.makeText(activity, "已添加联系人：" + name, Toast.LENGTH_SHORT).show();
                    activity.deliverNativeResult(requestId, true, null, data);
                });
            } catch (SecurityException error) {
                activity.runOnUiThread(() -> {
                    if (origin.equals(activity.nativeOrigin()) && activity.isTrustedPage())
                        activity.deliverNativeResult(
                                requestId, false, "contacts permission denied", null);
                });
            } catch (Exception error) {
                activity.runOnUiThread(() -> {
                    if (origin.equals(activity.nativeOrigin()) && activity.isTrustedPage())
                        activity.deliverNativeResult(
                                requestId, false, "unable to add contact", null);
                });
            }
        });
    }

    private void requestCall(String requestId, JSONObject payload) {
        String contact = clamp(payload.optString("contact", ""), MAX_CONTACT_CHARS).trim();
        String number = clamp(payload.optString("number", ""), MAX_NUMBER_CHARS).trim();
        String name = "";
        if (number.isEmpty()) {
            if (contact.isEmpty()) {
                activity.deliverNativeResult(requestId, false, "no contact or number", null);
                return;
            }
            List<String[]> matches = findContactNumbers(contact);
            if (matches.isEmpty()) {
                activity.deliverNativeResult(requestId, false, "no contact matched", null);
                return;
            }
            Set<String> distinct = new LinkedHashSet<>();
            for (String[] match : matches) distinct.add(match[1]);
            if (distinct.size() > 1) {
                StringBuilder choices = new StringBuilder();
                for (String[] match : matches) {
                    if (choices.length() > 0) choices.append("; ");
                    choices.append(match[0]).append(" ").append(match[1]);
                }
                activity.deliverNativeResult(requestId, false,
                        "contact is ambiguous, pass 'number' explicitly: " + choices, null);
                return;
            }
            name = matches.get(0)[0];
            number = matches.get(0)[1];
        }
        String dialable = PhoneDataRules.normalizeDialable(number);
        if (dialable == null) {
            activity.deliverNativeResult(requestId, false, "invalid number", null);
            return;
        }
        if (PhoneDataRules.isBlockedEmergency(dialable)
                || PhoneNumberUtils.isEmergencyNumber(dialable)) {
            activity.deliverNativeResult(requestId, false,
                    "emergency numbers cannot be called by the agent", null);
            return;
        }
        String finalName = name;
        String finalNumber = dialable;
        String target = name.isEmpty() ? dialable : name + "（" + dialable + "）";
        confirm(requestId, "拨打电话", "即将直接拨打 " + target
                        + "。通话可能产生费用。\n\n只有点击确认后才会呼出。",
                () -> placeCall(requestId, finalName, finalNumber));
    }

    private void placeCall(String requestId, String name, String number) {
        try {
            Intent intent = new Intent(Intent.ACTION_CALL, Uri.fromParts("tel", number, null));
            activity.startActivity(intent);
        } catch (SecurityException error) {
            activity.deliverNativeResult(requestId, false, "call permission denied", null);
            return;
        } catch (ActivityNotFoundException error) {
            activity.deliverNativeResult(requestId, false, "no phone app available", null);
            return;
        }
        JSONObject data = new JSONObject();
        try {
            data.put("dispatched", true);
            data.put("contact", name);
            data.put("number", number);
        } catch (JSONException ignored) {
            data = null;
        }
        activity.deliverNativeResult(requestId, true, null, data);
    }

    private void composeSms(String requestId, JSONObject payload, boolean auto) {
        String contact = clamp(payload.optString("contact", ""), MAX_CONTACT_CHARS).trim();
        String number = clamp(payload.optString("number", ""), MAX_NUMBER_CHARS).trim();
        String text = payload.optString("text", "").trim();
        if (text.isEmpty()) {
            activity.deliverNativeResult(requestId, false, "empty message", null);
            return;
        }
        if (text.length() > MAX_SMS_CHARS) {
            activity.deliverNativeResult(requestId, false, "message too long", null);
            return;
        }
        String name = "";
        if (number.isEmpty()) {
            if (contact.isEmpty()) {
                activity.deliverNativeResult(requestId, false, "no contact or number", null);
                return;
            }
            List<String[]> matches = findContactNumbers(contact);
            if (matches.isEmpty()) {
                activity.deliverNativeResult(requestId, false, "no contact matched", null);
                return;
            }
            Set<String> distinct = new LinkedHashSet<>();
            for (String[] match : matches) distinct.add(match[1]);
            if (distinct.size() > 1) {
                StringBuilder names = new StringBuilder();
                for (String[] match : matches) {
                    if (names.length() > 0) names.append("; ");
                    names.append(match[0]).append(" ").append(match[1]);
                }
                activity.deliverNativeResult(requestId, false,
                        "contact is ambiguous, pass 'number' explicitly: " + names, null);
                return;
            }
            name = matches.get(0)[0];
            number = matches.get(0)[1];
        }
        String dialable = number.replaceAll("[^0-9+]", "");
        if (dialable.isEmpty() || dialable.indexOf('+') > 0) {
            activity.deliverNativeResult(requestId, false, "invalid number", null);
            return;
        }
        if (!auto) {
            String message = "新建短信给 " +
                    (name.isEmpty() ? dialable : name + "（" + dialable + "）") + "：\n\n" +
                    (text.length() > 280 ? text.substring(0, 280) + "…" : text) +
                    "\n\n确认后打开短信应用；还需要你自己点发送。";
            String finalName = name;
            String finalNumber = dialable;
            confirm(requestId, "发送短信", message,
                    () -> openSmsComposer(requestId, finalName, finalNumber, text));
            return;
        }
        openSmsComposer(requestId, name, dialable, text);
    }

    /** Distinct {@code {display name, number}} pairs for a name query, capped. */
    private List<String[]> findContactNumbers(String query) {
        List<String[]> found = new ArrayList<>();
        String[] projection = new String[]{
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Phone.NUMBER,
        };
        Cursor cursor = null;
        try {
            cursor = activity.getContentResolver().query(
                    ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                    projection,
                    ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " LIKE ?",
                    new String[]{"%" + query + "%"},
                    ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " ASC");
            if (cursor == null) return found;
            Set<String> seen = new LinkedHashSet<>();
            while (cursor.moveToNext() && found.size() < 6) {
                String name = blankIfNull(cursor.getString(0));
                String number = blankIfNull(cursor.getString(1));
                if (number.trim().isEmpty()) continue;
                if (!seen.add(name + "\u0000" + number)) continue;
                found.add(new String[]{name, number});
            }
        } catch (SecurityException error) {
            // The permission gate answers the denial; an empty list here is only
            // reachable if the provider vanished between grant and query.
        } finally {
            if (cursor != null) cursor.close();
        }
        return found;
    }

    private void openSmsComposer(String requestId, String name, String number, String text) {
        Intent intent = new Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:" + number));
        intent.putExtra("sms_body", text);
        intent.putExtra("exit_on_sent", true);
        try {
            activity.startActivity(intent);
        } catch (ActivityNotFoundException error) {
            activity.deliverNativeResult(requestId, false, "no messaging app", null);
            return;
        }
        Toast.makeText(activity, "已打开短信，请确认后发送", Toast.LENGTH_SHORT).show();
        JSONObject data = new JSONObject();
        try {
            data.put("contact", name);
            data.put("number", number);
        } catch (JSONException ignored) {
            data = null;
        }
        activity.deliverNativeResult(requestId, true, null, data);
    }

    private static String blankIfNull(String value) {
        return value == null ? "" : value;
    }

    private void confirm(String requestId, String title, String message, Runnable onConfirm) {
        activity.runOnUiThread(() -> {
            if (activity.isFinishing() || activity.isDestroyed()) {
                activity.deliverNativeResult(requestId, false, "app not active", null);
                return;
            }
            new AlertDialog.Builder(activity)
                    .setTitle(title)
                    .setMessage(message)
                    .setNegativeButton("取消",
                            (dialog, which) -> activity.deliverNativeResult(requestId, false, "cancelled", null))
                    .setPositiveButton("确认", (dialog, which) -> onConfirm.run())
                    .setOnCancelListener(
                            dialog -> activity.deliverNativeResult(requestId, false, "cancelled", null))
                    .show();
        });
    }

    private static String clamp(String value, int max) {
        if (value == null) return "";
        return value.length() <= max ? value : value.substring(0, max);
    }
}