package com.grewwc.aserve;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import org.json.JSONObject;

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.UUID;

/** User-selected UTF-8 documents only; no broad storage permission or URI supplied by JS. */
final class PhoneDocuments {
    static final int REQUEST = 102;
    private final MainActivity activity;
    private final LinkedHashMap<String, Document> documents = new LinkedHashMap<>();
    private String pendingId;
    private String pendingAction;
    private String pendingOrigin;
    private String pendingText;

    private static final class Document {
        final String origin, name, text;
        Document(String origin, String name, String text) {
            this.origin = origin;
            this.name = name;
            this.text = text;
        }
    }

    PhoneDocuments(MainActivity activity) { this.activity = activity; }

    void perform(String id, String action, JSONObject payload) {
        try {
            if (!activity.isTrustedPage()) throw new IllegalStateException("untrusted page");
            if ("file_read".equals(action)) {
                String fileId = payload.getString("file_id");
                Document document = documents.get(fileId);
                if (document == null || !document.origin.equals(activity.nativeOrigin()))
                    throw new IllegalArgumentException("file expired; import it again");
                int offset = integer(payload, "offset", 0, 0, TextDocument.MAX_BYTES);
                int limit = integer(payload, "limit", 3000, 1, 3000);
                activity.deliverNativeResult(id, true, null, page(fileId, document, offset, limit));
                return;
            }
            if (pendingId != null) throw new IllegalStateException("another document picker is open");
            Intent intent;
            String text = null;
            if ("file_import".equals(action)) {
                intent = new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("text/*");
                intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"text/*", "application/json"});
            } else if ("file_save".equals(action)) {
                String filename = payload.getString("filename");
                text = payload.getString("text");
                if (!TextDocument.validFilename(filename) || text.trim().isEmpty()
                        || text.codePointCount(0, text.length()) > 3000)
                    throw new IllegalArgumentException("invalid filename or text (maximum 3000 characters)");
                intent = new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("text/plain");
                intent.putExtra(Intent.EXTRA_TITLE, filename);
            } else throw new IllegalArgumentException("unknown document action");
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            pendingId = id;
            pendingAction = action;
            pendingOrigin = activity.nativeOrigin();
            pendingText = text;
            activity.startActivityForResult(intent, REQUEST);
        } catch (Exception error) {
            if (id.equals(pendingId)) clearPending();
            activity.deliverNativeResult(id, false, error.getMessage(), null);
        }
    }

    private static int integer(JSONObject payload, String key, int fallback, int min, int max) throws Exception {
        if (!payload.has(key)) return fallback;
        Object value = payload.get(key);
        if (!(value instanceof Number)) throw new IllegalArgumentException("invalid " + key);
        double n = ((Number) value).doubleValue();
        if (!Double.isFinite(n) || n != Math.rint(n) || n < min || n > max)
            throw new IllegalArgumentException("invalid " + key);
        return (int) n;
    }

    private JSONObject page(String id, Document document, int offset, int limit) throws Exception {
        String text = TextDocument.page(document.text, offset, limit);
        int total = document.text.codePointCount(0, document.text.length());
        int next = offset + text.codePointCount(0, text.length());
        return new JSONObject().put("file_id", id).put("filename", document.name)
                .put("mime_type", "text/plain").put("total_chars", total).put("text", text)
                .put("offset", offset).put("next_offset", next).put("has_more", next < total);
    }

    void onResult(int resultCode, Intent data) {
        if (pendingId == null) return;
        final String id = pendingId, action = pendingAction, origin = pendingOrigin, text = pendingText;
        clearPending();
        if (!activity.isTrustedPage() || !origin.equals(activity.nativeOrigin())) return;
        if (resultCode != Activity.RESULT_OK || data == null || data.getData() == null) {
            activity.deliverNativeResult(id, false, "cancelled", null);
            return;
        }
        Uri uri = data.getData();
        if (!"content".equals(uri.getScheme())) {
            activity.deliverNativeResult(id, false, "document provider must return a content URI", null);
            return;
        }
        // Providers may be remote; do not block WebView/IME while reading or writing.
        new Thread(() -> {
            try {
                String name = "document.txt";
                try (Cursor cursor = activity.getContentResolver().query(uri,
                        new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
                    if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0))
                        name = cursor.getString(0);
                }
                if (name.length() > 120) name = name.substring(0, 120);
                JSONObject result;
                Document document = null;
                String fileId = null;
                if ("file_import".equals(action)) {
                    String imported;
                    try (InputStream input = activity.getContentResolver().openInputStream(uri)) {
                        imported = TextDocument.read(input);
                    }
                    document = new Document(origin, name, imported);
                    fileId = UUID.randomUUID().toString();
                    result = page(fileId, document, 0, TextDocument.PAGE_CHARS);
                } else {
                    byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
                    try (OutputStream output = activity.getContentResolver().openOutputStream(uri, "w")) {
                        if (output == null) throw new IllegalStateException("document could not be opened");
                        output.write(bytes);
                        output.flush();
                    }
                    result = new JSONObject().put("saved", true).put("filename", name)
                            .put("bytes", bytes.length).put("encoding", "UTF-8");
                }
                final Document stored = document;
                final String storedId = fileId;
                activity.runOnUiThread(() -> {
                    if (!activity.isTrustedPage() || !origin.equals(activity.nativeOrigin())) return;
                    if (stored != null) {
                        while (documents.size() >= 4) documents.remove(documents.keySet().iterator().next());
                        documents.put(storedId, stored);
                    }
                    activity.deliverNativeResult(id, true, null, result);
                });
            } catch (Exception error) {
                activity.runOnUiThread(() -> {
                    if (activity.isTrustedPage() && origin.equals(activity.nativeOrigin()))
                        activity.deliverNativeResult(id, false, error.getMessage(), null);
                });
            }
        }, "a-serve-document").start();
    }

    private void clearPending() {
        pendingId = pendingAction = pendingOrigin = pendingText = null;
    }
}
