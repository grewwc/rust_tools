package com.grewwc.aserve;

import android.content.ContentProviderOperation;
import android.content.ContentProviderResult;
import android.content.ContentResolver;
import android.database.Cursor;
import android.provider.CallLog;
import android.provider.ContactsContract;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.TimeZone;

/** Bounded contact/call-history access plus one atomic contact insertion. */
final class PhoneRecords {
    private static final int MAX_PAGE = 20;
    private static final int MAX_OFFSET = 100_000;
    private static final int MAX_QUERY_CHARS = 200;

    private PhoneRecords() {}

    static JSONObject addContact(ContentResolver resolver, String name, String number)
            throws Exception {
        ArrayList<ContentProviderOperation> operations = new ArrayList<>();
        operations.add(ContentProviderOperation.newInsert(ContactsContract.RawContacts.CONTENT_URI)
                .withValue(ContactsContract.RawContacts.ACCOUNT_TYPE, null)
                .withValue(ContactsContract.RawContacts.ACCOUNT_NAME, null)
                .build());
        operations.add(ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI)
                .withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
                .withValue(ContactsContract.Data.MIMETYPE,
                        ContactsContract.CommonDataKinds.StructuredName.CONTENT_ITEM_TYPE)
                .withValue(ContactsContract.CommonDataKinds.StructuredName.DISPLAY_NAME, name)
                .build());
        operations.add(ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI)
                .withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
                .withValue(ContactsContract.Data.MIMETYPE,
                        ContactsContract.CommonDataKinds.Phone.CONTENT_ITEM_TYPE)
                .withValue(ContactsContract.CommonDataKinds.Phone.NUMBER, number)
                .withValue(ContactsContract.CommonDataKinds.Phone.TYPE,
                        ContactsContract.CommonDataKinds.Phone.TYPE_MOBILE)
                .build());
        ContentProviderResult[] results = resolver.applyBatch(
                ContactsContract.AUTHORITY, operations);
        if (results.length != operations.size() || results[0].uri == null) {
            throw new IllegalStateException("contacts provider did not create the contact");
        }
        return new JSONObject().put("name", name).put("number", number);
    }

    static JSONObject readContacts(ContentResolver resolver, JSONObject payload)
            throws JSONException {
        int max = bounded(payload.optInt("max", 10), 1, MAX_PAGE, 10);
        int offset = bounded(payload.optInt("offset", 0), 0, MAX_OFFSET, 0);
        String query = payload.optString("query", "").trim();
        if (query.length() > MAX_QUERY_CHARS) query = query.substring(0, MAX_QUERY_CHARS);
        String queryLower = query.toLowerCase(Locale.ROOT);
        String[] projection = {
                ContactsContract.CommonDataKinds.Phone.CONTACT_ID,
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Phone.NUMBER,
                ContactsContract.CommonDataKinds.Phone.TYPE,
                ContactsContract.CommonDataKinds.Phone.LABEL
        };
        JSONArray entries = new JSONArray();
        boolean hasMore = false;
        int matched = 0;
        Set<String> seen = new HashSet<>();
        try (Cursor cursor = resolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                projection,
                null,
                null,
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " COLLATE LOCALIZED ASC")) {
            while (cursor != null && cursor.moveToNext()) {
                long id = cursor.getLong(0);
                String name = blank(cursor.getString(1));
                String number = blank(cursor.getString(2));
                if (number.isEmpty()) continue;
                if (!queryLower.isEmpty()
                        && !name.toLowerCase(Locale.ROOT).contains(queryLower)
                        && !number.toLowerCase(Locale.ROOT).contains(queryLower)) {
                    continue;
                }
                String dedupe = id + "\u0000" + number;
                if (!seen.add(dedupe)) continue;
                if (matched++ < offset) continue;
                if (entries.length() >= max) {
                    hasMore = true;
                    break;
                }
                JSONObject entry = new JSONObject();
                entry.put("contact_id", id);
                entry.put("name", name);
                entry.put("number", number);
                entry.put("type", PhoneDataRules.phoneTypeName(cursor.getInt(3), cursor.getString(4)));
                entries.put(entry);
            }
        }
        JSONObject data = page(entries, offset, hasMore);
        data.put("query", query);
        data.put("contacts", entries);
        return data;
    }

    static JSONObject readCallHistory(ContentResolver resolver, JSONObject payload)
            throws JSONException {
        int max = bounded(payload.optInt("max", 10), 1, MAX_PAGE, 10);
        int offset = bounded(payload.optInt("offset", 0), 0, MAX_OFFSET, 0);
        String requestedType = payload.optString("call_type", "all");
        int typeCode = PhoneDataRules.callTypeCode(requestedType);
        if (typeCode < 0) {
            requestedType = "all";
            typeCode = 0;
        }
        String[] projection = {
                CallLog.Calls.NUMBER,
                CallLog.Calls.CACHED_NAME,
                CallLog.Calls.TYPE,
                CallLog.Calls.DATE,
                CallLog.Calls.DURATION
        };
        String selection = typeCode == 0 ? null : CallLog.Calls.TYPE + " = ?";
        String[] selectionArgs = typeCode == 0 ? null : new String[] {Integer.toString(typeCode)};
        JSONArray entries = new JSONArray();
        boolean hasMore = false;
        int matched = 0;
        SimpleDateFormat localTime = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.ROOT);
        localTime.setTimeZone(TimeZone.getDefault());
        try (Cursor cursor = resolver.query(
                CallLog.Calls.CONTENT_URI,
                projection,
                selection,
                selectionArgs,
                CallLog.Calls.DATE + " DESC")) {
            while (cursor != null && cursor.moveToNext()) {
                if (matched++ < offset) continue;
                if (entries.length() >= max) {
                    hasMore = true;
                    break;
                }
                long timestampMs = cursor.getLong(3);
                JSONObject entry = new JSONObject();
                entry.put("number", blank(cursor.getString(0)));
                entry.put("name", blank(cursor.getString(1)));
                entry.put("type", PhoneDataRules.callTypeName(cursor.getInt(2)));
                entry.put("timestamp_ms", timestampMs);
                entry.put("local_time", localTime.format(new Date(timestampMs)));
                entry.put("duration_seconds", Math.max(0L, cursor.getLong(4)));
                entries.put(entry);
            }
        }
        JSONObject data = page(entries, offset, hasMore);
        data.put("call_type", requestedType);
        data.put("timezone", TimeZone.getDefault().getID());
        data.put("calls", entries);
        return data;
    }

    private static JSONObject page(JSONArray entries, int offset, boolean hasMore)
            throws JSONException {
        JSONObject data = new JSONObject();
        data.put("offset", offset);
        data.put("next_offset", offset + entries.length());
        data.put("has_more", hasMore);
        return data;
    }

    private static int bounded(int value, int min, int max, int fallback) {
        return value < min || value > max ? fallback : value;
    }

    private static String blank(String value) {
        return value == null ? "" : value;
    }
}
