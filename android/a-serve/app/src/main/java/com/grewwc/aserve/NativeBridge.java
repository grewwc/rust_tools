package com.grewwc.aserve;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.AlarmClock;
import android.webkit.JavascriptInterface;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.Locale;

/**
 * Narrow native surface for the server-owned page.
 *
 * <p>Only three allowlisted actions exist. The page never gets a generic
 * {@code startActivity}, and every action that changes phone state is shown in
 * a native confirmation dialog first. Calls are rejected unless the loaded page
 * still belongs to the configured server origin.</p>
 */
final class NativeBridge {
    static final String NAME = "AServeNative";

    private static final int MAX_LABEL_CHARS = 120;
    private static final int MAX_NOTE_CHARS = 3000;
    private static final int MAX_TITLE_CHARS = 200;
    private static final long FRESH_LOCATION_MS = 5 * 60 * 1000L;
    private static final long LOCATION_TIMEOUT_MS = 12000L;

    private final MainActivity activity;

    NativeBridge(MainActivity activity) {
        this.activity = activity;
    }

    /** Feature detection for the page; returns a JSON string. */
    @JavascriptInterface
    public String capabilities() {
        JSONObject caps = new JSONObject();
        try {
            caps.put("version", BuildConfig.VERSION_NAME);
            caps.put("actions", new JSONArray().put("alarm").put("note").put("location"));
        } catch (JSONException ignored) {
            // Both keys are constants; serialization cannot fail here.
        }
        return caps.toString();
    }

    /**
     * Entry point for every native action.
     *
     * @param requestId caller-generated id echoed in {@code AServeNativeResult}
     * @param action    one of {@code alarm}, {@code note}, {@code location}
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
        JSONObject payload;
        try {
            String raw = payloadJson == null || payloadJson.trim().isEmpty() ? "{}" : payloadJson;
            payload = new JSONObject(raw);
        } catch (JSONException error) {
            activity.deliverNativeResult(requestId, false, "invalid payload", null);
            return;
        }
        // `auto` is set only by the serve page when the agent's phone_action
        // tool requested the action. The Android permission prompt (location)
        // and the share sheet (notes) remain the user-visible gates; alarms are
        // set silently because the user already asked for one in the chat.
        boolean auto = payload.optBoolean("auto", false);
        if ("alarm".equals(action)) {
            requestAlarm(requestId, payload, auto);
        } else if ("note".equals(action)) {
            requestNote(requestId, payload, auto);
        } else if ("location".equals(action)) {
            requestLocation(requestId, auto);
        } else {
            activity.deliverNativeResult(requestId, false, "unknown action", null);
        }
    }

    private void requestAlarm(String requestId, JSONObject payload, boolean auto) {
        int hour = payload.optInt("hour", -1);
        int minute = payload.optInt("minute", -1);
        String label = clamp(payload.optString("label", ""), MAX_LABEL_CHARS).trim();
        if (hour < 0 || hour > 23 || minute < 0 || minute > 59) {
            activity.deliverNativeResult(requestId, false, "invalid time", null);
            return;
        }
        String detail = String.format(Locale.getDefault(), "设置闹钟 %02d:%02d", hour, minute);
        if (!label.isEmpty()) detail += "\n标签：" + label;
        String title = "设置闹钟";
        String message = detail + "\n\n确认后打开系统时钟并创建该闹钟。";
        if (auto) {
            setAlarm(requestId, hour, minute, label);
            return;
        }
        confirm(requestId, title, message,
                () -> setAlarm(requestId, hour, minute, label));
    }

    private void setAlarm(String requestId, int hour, int minute, String label) {
        Intent intent = new Intent(AlarmClock.ACTION_SET_ALARM)
                .putExtra(AlarmClock.EXTRA_HOUR, hour)
                .putExtra(AlarmClock.EXTRA_MINUTES, minute)
                .putExtra(AlarmClock.EXTRA_SKIP_UI, true);
        if (!label.isEmpty()) intent.putExtra(AlarmClock.EXTRA_MESSAGE, label);
        try {
            activity.startActivity(intent);
            Toast.makeText(activity,
                    String.format(Locale.getDefault(), "闹钟已设置 %02d:%02d", hour, minute),
                    Toast.LENGTH_SHORT).show();
            activity.deliverNativeResult(requestId, true, null, null);
        } catch (ActivityNotFoundException first) {
            // Some clock apps reject the skip-UI flag; retry with the picker.
            intent.putExtra(AlarmClock.EXTRA_SKIP_UI, false);
            try {
                activity.startActivity(intent);
                Toast.makeText(activity,
                        String.format(Locale.getDefault(), "闹钟已设置 %02d:%02d", hour, minute),
                        Toast.LENGTH_SHORT).show();
                activity.deliverNativeResult(requestId, true, null, null);
            } catch (ActivityNotFoundException second) {
                activity.deliverNativeResult(requestId, false, "no alarm app", null);
            }
        } catch (SecurityException error) {
            activity.deliverNativeResult(requestId, false, "alarm permission denied", null);
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
        if (!activity.hasLocationPermission()) {
            activity.requestLocationPermission(requestId);
            return;
        }
        if (auto) {
            fetchLocation(requestId);
            return;
        }
        confirm(requestId, "使用大致位置",
                "获取一次大致位置并填入输入框，用于天气、附近搜索等。\n\n不会在后台持续定位，也不会自动发送。",
                () -> fetchLocation(requestId));
    }

    /** Called by the activity after the runtime permission result, or directly. */
    void fetchLocation(String requestId) {
        activity.runOnUiThread(() -> {
            LocationManager manager = (LocationManager) activity.getSystemService(Activity.LOCATION_SERVICE);
            if (manager == null) {
                activity.deliverNativeResult(requestId, false, "location unavailable", null);
                return;
            }
            Location best = null;
            for (String provider : new String[]{
                    LocationManager.NETWORK_PROVIDER,
                    LocationManager.GPS_PROVIDER,
                    LocationManager.PASSIVE_PROVIDER}) {
                try {
                    Location candidate = manager.getLastKnownLocation(provider);
                    if (candidate != null && (best == null || candidate.getTime() > best.getTime())) {
                        best = candidate;
                    }
                } catch (SecurityException | IllegalArgumentException ignored) {
                    // Provider missing or not permitted; try the next one.
                }
            }
            if (best != null && System.currentTimeMillis() - best.getTime() <= FRESH_LOCATION_MS) {
                deliverLocation(requestId, best);
                return;
            }

            String provider = null;
            for (String candidate : new String[]{
                    LocationManager.NETWORK_PROVIDER,
                    LocationManager.GPS_PROVIDER}) {
                try {
                    if (manager.isProviderEnabled(candidate)) {
                        provider = candidate;
                        break;
                    }
                } catch (IllegalArgumentException ignored) {
                    // Unknown provider on this device.
                }
            }
            if (provider == null) {
                activity.deliverNativeResult(requestId, false, "location services disabled", null);
                return;
            }

            final LocationManager locationManager = manager;
            final boolean[] finished = {false};
            final LocationListener[] listener = new LocationListener[1];
            Handler handler = new Handler(Looper.getMainLooper());
            Runnable timeout = () -> {
                if (finished[0]) return;
                finished[0] = true;
                try {
                    locationManager.removeUpdates(listener[0]);
                } catch (Exception ignored) {
                    // Updates already stopped.
                }
                activity.deliverNativeResult(requestId, false, "location timeout", null);
            };
            listener[0] = new LocationListener() {
                @Override
                public void onLocationChanged(Location location) {
                    if (finished[0]) return;
                    finished[0] = true;
                    handler.removeCallbacks(timeout);
                    try {
                        locationManager.removeUpdates(this);
                    } catch (Exception ignored) {
                        // Updates already stopped.
                    }
                    deliverLocation(requestId, location);
                }

                @Override
                public void onStatusChanged(String provider, int status, Bundle extras) {}

                @Override
                public void onProviderEnabled(String provider) {}

                @Override
                public void onProviderDisabled(String provider) {}
            };
            try {
                locationManager.requestLocationUpdates(provider, 0L, 0f, listener[0], Looper.getMainLooper());
            } catch (SecurityException error) {
                finished[0] = true;
                activity.deliverNativeResult(requestId, false, "location permission denied", null);
                return;
            }
            handler.postDelayed(timeout, LOCATION_TIMEOUT_MS);
        });
    }

    private void deliverLocation(String requestId, Location location) {
        JSONObject data = new JSONObject();
        try {
            data.put("latitude", location.getLatitude());
            data.put("longitude", location.getLongitude());
            if (location.hasAccuracy()) data.put("accuracy", Math.round(location.getAccuracy()));
            data.put("age_ms", Math.max(0L, System.currentTimeMillis() - location.getTime()));
        } catch (JSONException ignored) {
            activity.deliverNativeResult(requestId, false, "location unavailable", null);
            return;
        }
        activity.deliverNativeResult(requestId, true, null, data);
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