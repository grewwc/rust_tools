package com.grewwc.aserve;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteException;
import android.database.sqlite.SQLiteOpenHelper;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * App-private, origin-scoped notes and todos. No permissions or UI are involved.
 *
 * <p>The caller must authenticate the currently trusted server page and supply its configured
 * HTTPS origin, never a payload-provided origin. Run execute off the UI thread. A mutation's
 * requestId must survive page reloads/retries; generating a new id means a new mutation.
 * Successful mutation results are retained indefinitely and replayed even after later edits.
 * Unknown fields (including bridge-only fields such as auto) must be removed by the caller.
 * Exceptions indicate failure; they must not be converted into a success response.</p>
 *
 * <p>Create/update return an entry object with entry_id, kind, title, text and completed.
 * List returns entries with preview instead of text, plus total, offset and has_more.
 * Search is a literal, case-sensitive substring of title or text, newest-created first.
 * Content limits count Unicode codepoints; requestId is limited to 80 UTF-16 code units
 * to match the bridge. No submitted content is silently truncated.</p>
 */
public final class LocalNotebook implements AutoCloseable {
    private static final String DATABASE_NAME = "local_notebook.db";
    private final Database helper;

    public LocalNotebook(Context context) {
        Context app = context.getApplicationContext();
        helper = new Database(app == null ? context : app);
    }

    public synchronized JSONObject execute(
            String origin, String requestId, String action, JSONObject payload) throws Exception {
        if (payload == null) throw new IllegalArgumentException("payload must be an object");
        Map<String, Object> fields = new HashMap<>();
        Iterator<String> keys = payload.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            Object value = payload.get(key);
            fields.put(key, value == JSONObject.NULL ? null : value);
        }
        LocalNotebookModel.Request request =
                new LocalNotebookModel.Request(origin, requestId, action, fields);
        SQLiteDatabase db = helper.getWritableDatabase();
        // Serialize the read/dedup/write/ledger sequence, including across helper instances.
        // Lists use the same transaction so total and page describe one database snapshot.
        db.beginTransaction();
        try {
            JSONObject result;
            if (!request.isMutation()) {
                result = list(db, request);
            } else {
                result = replay(db, request);
                if (result == null) {
                    LocalNotebookModel.Entry entry;
                    if ("note_create".equals(request.action)) {
                        entry = LocalNotebookModel.Entry.create(request, UUID.randomUUID().toString());
                        ContentValues values = values(entry);
                        values.put("origin", request.origin);
                        db.insertOrThrow("entries", null, values);
                    } else {
                        entry = get(db, request.origin, request.entryId).update(request);
                        int count = db.update("entries", values(entry), "origin = ? AND entry_id = ?",
                                new String[]{request.origin, request.entryId});
                        if (count != 1) throw new SQLiteException("notebook update affected wrong row count");
                    }
                    result = json(entry, false);
                    ContentValues executed = new ContentValues();
                    executed.put("origin", request.origin);
                    executed.put("request_id", request.requestId);
                    executed.put("arguments", request.arguments);
                    executed.put("result_json", result.toString());
                    db.insertOrThrow("executed_requests", null, executed);
                }
            }
            db.setTransactionSuccessful();
            return result;
        } finally {
            // The entry and replay record both commit before any result reaches the caller.
            // Disk errors during commit propagate, allowing a safe same-id retry.
            db.endTransaction();
        }
    }

    private static JSONObject replay(SQLiteDatabase db, LocalNotebookModel.Request request)
            throws Exception {
        try (Cursor cursor = db.query("executed_requests", new String[]{"arguments", "result_json"},
                "origin = ? AND request_id = ?", new String[]{request.origin, request.requestId},
                null, null, null)) {
            if (!cursor.moveToFirst()) return null;
            request.requireSameArguments(cursor.getString(0));
            JSONObject result = new JSONObject(cursor.getString(1));
            // Fail closed if stored JSON is malformed or violates the bounded result contract.
            if (result.length() != 5 || !(result.get("completed") instanceof Boolean)) {
                throw new SQLiteException("invalid stored notebook result");
            }
            new LocalNotebookModel.Entry(strictString(result, "entry_id"), strictString(result, "kind"),
                    strictString(result, "title"), strictString(result, "text"),
                    (Boolean) result.get("completed"));
            return result;
        }
    }

    private static String strictString(JSONObject object, String key) throws Exception {
        Object value = object.get(key);
        if (!(value instanceof String)) throw new SQLiteException("invalid stored notebook result");
        return (String) value;
    }

    private static LocalNotebookModel.Entry get(SQLiteDatabase db, String origin, String entryId) {
        try (Cursor cursor = db.query("entries", new String[]{"entry_id", "kind", "title", "text", "completed"},
                "origin = ? AND entry_id = ?", new String[]{origin, entryId}, null, null, null)) {
            if (!cursor.moveToFirst()) throw new IllegalArgumentException("notebook entry not found");
            return entry(cursor);
        }
    }

    private static JSONObject list(SQLiteDatabase db, LocalNotebookModel.Request request) throws Exception {
        String selection = "origin = ?";
        List<String> args = new ArrayList<>();
        args.add(request.origin);
        if (request.kind != null) {
            selection += " AND kind = ?";
            args.add(request.kind);
        }
        if (!request.query.isEmpty()) {
            selection += " AND (instr(title, ?) > 0 OR instr(text, ?) > 0)";
            args.add(request.query);
            args.add(request.query);
        }
        String[] arguments = args.toArray(new String[0]);
        long total;
        try (Cursor cursor = db.rawQuery("SELECT COUNT(*) FROM entries WHERE " + selection, arguments)) {
            if (!cursor.moveToFirst()) throw new SQLiteException("notebook count unavailable");
            total = cursor.getLong(0);
        }
        JSONArray entries = new JSONArray();
        try (Cursor cursor = db.query("entries", new String[]{"entry_id", "kind", "title", "text", "completed"},
                selection, arguments, null, null, "sequence DESC", request.offset + "," + request.max)) {
            while (cursor.moveToNext()) entries.put(json(entry(cursor), true));
        }
        JSONObject result = new JSONObject();
        result.put("entries", entries);
        result.put("total", total);
        result.put("offset", request.offset);
        // Subtraction avoids overflow for very large requested offsets.
        result.put("has_more", request.offset < total && entries.length() < total - request.offset);
        return result;
    }

    private static LocalNotebookModel.Entry entry(Cursor cursor) {
        for (int i = 0; i < 4; i++) {
            if (cursor.getType(i) != Cursor.FIELD_TYPE_STRING) {
                throw new SQLiteException("invalid stored notebook entry");
            }
        }
        long completed = cursor.getLong(4);
        if (cursor.getType(4) != Cursor.FIELD_TYPE_INTEGER || (completed != 0 && completed != 1)) {
            throw new SQLiteException("invalid stored notebook completion");
        }
        return new LocalNotebookModel.Entry(cursor.getString(0), cursor.getString(1),
                cursor.getString(2), cursor.getString(3), completed == 1);
    }

    private static JSONObject json(LocalNotebookModel.Entry entry, boolean preview) throws Exception {
        JSONObject result = new JSONObject();
        result.put("entry_id", entry.entryId);
        result.put("kind", entry.kind);
        result.put("title", entry.title);
        result.put("text", entry.text);
        if (preview) result.put("preview", entry.preview());
        result.put("completed", entry.completed);
        return result;
    }

    private static ContentValues values(LocalNotebookModel.Entry entry) {
        ContentValues values = new ContentValues();
        values.put("entry_id", entry.entryId);
        values.put("kind", entry.kind);
        values.put("title", entry.title);
        values.put("text", entry.text);
        values.put("completed", entry.completed ? 1 : 0);
        return values;
    }

    @Override
    public synchronized void close() {
        helper.close();
    }

    private static final class Database extends SQLiteOpenHelper {
        Database(Context context) {
            super(context, DATABASE_NAME, null, 1, db -> {
                // Android's default handler deletes corrupt databases. Preserve this one so
                // neither saved notes nor durable deduplication records disappear silently.
                throw new SQLiteException("local notebook is corrupt; database was preserved");
            });
        }

        @Override
        public void onConfigure(SQLiteDatabase db) {
            db.execSQL("PRAGMA synchronous=FULL");
        }

        @Override
        public void onCreate(SQLiteDatabase db) {
            db.execSQL("CREATE TABLE entries ("
                    + "sequence INTEGER PRIMARY KEY AUTOINCREMENT,"
                    + "origin TEXT NOT NULL, entry_id TEXT NOT NULL,"
                    + "kind TEXT NOT NULL CHECK(kind IN ('note','todo')),"
                    + "title TEXT NOT NULL, text TEXT NOT NULL,"
                    + "completed INTEGER NOT NULL CHECK(completed IN (0,1)),"
                    + "CHECK(kind = 'todo' OR completed = 0), UNIQUE(origin, entry_id))");
            db.execSQL("CREATE INDEX entries_origin_sequence ON entries(origin, sequence DESC)");
            db.execSQL("CREATE TABLE executed_requests ("
                    + "origin TEXT NOT NULL, request_id TEXT NOT NULL,"
                    + "arguments TEXT NOT NULL, result_json TEXT NOT NULL,"
                    + "PRIMARY KEY(origin, request_id))");
        }

        @Override
        public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
            throw new SQLiteException("unsupported notebook schema upgrade");
        }

        @Override
        public void onDowngrade(SQLiteDatabase db, int oldVersion, int newVersion) {
            throw new SQLiteException("unsupported notebook schema downgrade");
        }
    }
}
