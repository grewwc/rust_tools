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
import android.database.Cursor;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.AlarmClock;
import android.provider.CalendarContract;
import android.provider.ContactsContract;
import android.webkit.JavascriptInterface;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

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

/**
 * Narrow native surface for the server-owned page.
 *
 * <p>Only allowlisted actions exist. The page never gets a generic
 * {@code startActivity}, and every action that changes phone state is shown in
 * a native confirmation dialog first. Calls are rejected unless the loaded page
 * still belongs to the configured server origin.</p>
 */
final class NativeBridge {
    static final String NAME = "AServeNative";

    private static final int MAX_LABEL_CHARS = 120;
    private static final int MAX_NOTE_CHARS = 3000;
    private static final int MAX_TITLE_CHARS = 200;
    private static final int MAX_PLACE_CHARS = 200;
    private static final int MAX_DESCRIPTION_CHARS = 1000;
    private static final int MAX_SMS_CHARS = 1000;
    private static final int MAX_CONTACT_CHARS = 120;
    private static final int MAX_NUMBER_CHARS = 40;
    private static final String LOCAL_DATE_TIME = "yyyy-MM-dd'T'HH:mm";
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
            caps.put("actions", new JSONArray()
                    .put("now").put("calendar_read").put("calendar_add").put("sms")
                    .put("alarm").put("note").put("location"));
        } catch (JSONException ignored) {
            // Both keys are constants; serialization cannot fail here.
        }
        return caps.toString();
    }

    /**
     * Entry point for every native action.
     *
     * @param requestId caller-generated id echoed in {@code AServeNativeResult}
     * @param action    one of {@code now}, {@code calendar_read}, {@code calendar_add},
     *                  {@code sms}, {@code alarm}, {@code note}, {@code location}
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
            needed.add(Manifest.permission.ACCESS_COARSE_LOCATION);
        } else if ("calendar_read".equals(action)) {
            needed.add(Manifest.permission.READ_CALENDAR);
        } else if ("calendar_add".equals(action)) {
            needed.add(Manifest.permission.READ_CALENDAR);
            needed.add(Manifest.permission.WRITE_CALENDAR);
        } else if ("sms".equals(action) && payload.optString("contact", "").trim().length() > 0) {
            needed.add(Manifest.permission.READ_CONTACTS);
        }
        List<String> missing = new ArrayList<>();
        for (String permission : needed) {
            if (activity.checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED) {
                missing.add(permission);
            }
        }
        return missing.toArray(new String[0]);
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