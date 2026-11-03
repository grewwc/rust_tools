package com.grewwc.aserve;

import android.content.ContentResolver;
import android.content.ContentUris;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/** App-private incremental photo index using local OCR and CLIP embeddings. */
final class PhotoSemanticIndex implements Closeable {
    static final int EMBEDDING_SIZE = 512;
    private static final int TEXT_HIDDEN_SIZE = 768;
    private static final int IMAGE_SIZE = 224;
    private static final int MAX_OCR_CHARS = 32_768;
    private static final int MAX_QUERY_TOKENS = 64;
    private static final float DEFAULT_MIN_SCORE = 0.18f;
    private static final Comparator<ScoredMatch> BEST_MATCH_FIRST =
            Comparator.comparingDouble((ScoredMatch item) -> item.score).reversed()
                    .thenComparing(Comparator.comparingLong(
                            (ScoredMatch item) -> item.dateTaken).reversed())
                    .thenComparingLong(item -> item.mediaId);
    private static final String[] MEDIA_PROJECTION = {
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.MIME_TYPE,
            MediaStore.Images.Media.DATE_TAKEN,
            MediaStore.Images.Media.DATE_MODIFIED,
            MediaStore.Images.Media.SIZE,
            MediaStore.Images.Media.WIDTH,
            MediaStore.Images.Media.HEIGHT
    };

    private final Context context;
    private final IndexDatabase database;
    private final TextRecognizer recognizer;
    private final File modelDirectory;
    private final String modelVersion;
    private ClipRuntime clip;

    PhotoSemanticIndex(Context context, File modelDirectory, String modelVersion) throws IOException {
        this.context = context.getApplicationContext();
        if (modelDirectory == null || !modelDirectory.isDirectory()) {
            throw new IOException("photo model directory is unavailable");
        }
        this.modelDirectory = modelDirectory;
        this.modelVersion = modelVersion;
        this.database = new IndexDatabase(this.context, modelVersion);
        this.recognizer = TextRecognition.getClient(
                new ChineseTextRecognizerOptions.Builder().build());
    }

    synchronized JSONObject refresh(JSONObject payload) throws Exception {
        int limit = strictInt(payload, "max", 50, 1, 500);
        boolean retryFailed = payload.optBoolean("retry_failed", false);
        boolean reindexExisting = payload.optBoolean("reindex_existing", false);
        SQLiteDatabase db = database.getWritableDatabase();
        Map<Long, ExistingRow> existing = loadExisting(db);
        Set<Long> missing = new HashSet<>(existing.keySet());
        int indexed = 0;
        int unchanged = 0;
        int failed = 0;
        int pending = 0;
        int removed;

        ContentResolver resolver = context.getContentResolver();
        try (Cursor cursor = resolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                MEDIA_PROJECTION,
                null,
                null,
                MediaStore.Images.Media.DATE_MODIFIED + " ASC, "
                        + MediaStore.Images.Media._ID + " ASC")) {
            if (cursor == null) throw new IOException("Android did not return the photo library");
            while (cursor.moveToNext()) {
                MediaPhoto photo = MediaPhoto.fromCursor(cursor);
                missing.remove(photo.id);
                ExistingRow row = existing.get(photo.id);
                boolean changed = row == null
                        || reindexExisting
                        || row.modifiedSeconds != photo.modifiedSeconds
                        || row.sizeBytes != photo.sizeBytes
                        || (retryFailed && row.failed);
                if (!changed) {
                    unchanged++;
                    continue;
                }
                if (indexed + failed >= limit) {
                    pending++;
                    continue;
                }
                try {
                    indexOne(db, resolver, photo);
                    indexed++;
                } catch (Exception error) {
                    storeFailure(db, photo, boundedError(error));
                    failed++;
                }
            }
        }

        db.beginTransaction();
        try {
            for (Long id : missing) {
                db.delete("photos", "media_id = ?", new String[]{Long.toString(id)});
            }
            db.setTransactionSuccessful();
        } finally {
            db.endTransaction();
        }
        removed = missing.size();

        JSONObject data = status(db);
        data.put("indexed_now", indexed);
        data.put("unchanged", unchanged);
        data.put("failed_now", failed);
        data.put("removed", removed);
        data.put("pending", pending);
        data.put("complete", pending == 0);
        data.put("processing", "on_device");
        data.put("photo_bytes_uploaded", false);
        return data;
    }

    synchronized JSONObject search(JSONObject payload) throws Exception {
        String query = requiredString(payload, "query", 200);
        int max = strictInt(payload, "max", 10, 1, 20);
        int offset = strictInt(payload, "offset", 0, 0, 100_000);
        double requestedMin = payload.has("min_score")
                ? payload.getDouble("min_score") : DEFAULT_MIN_SCORE;
        if (!Double.isFinite(requestedMin) || requestedMin < -1.0 || requestedMin > 2.0) {
            throw new JSONException("min_score must be between -1 and 2");
        }
        float minScore = (float) requestedMin;
        float[] queryEmbedding = clip().embedText(query);
        String normalizedQuery = normalizeSearchText(query);
        int retainedLimit = offset + max;
        PriorityQueue<ScoredMatch> retained = new PriorityQueue<>(
                retainedLimit, BEST_MATCH_FIRST.reversed());
        int matched = 0;
        SQLiteDatabase db = database.getReadableDatabase();
        String[] columns = {
                "media_id", "content_uri", "display_name", "mime_type", "date_taken",
                "date_modified", "width", "height", "ocr_text", "embedding"
        };
        try (Cursor cursor = db.query("photos", columns, "embedding IS NOT NULL", null,
                null, null, null)) {
            while (cursor.moveToNext()) {
                byte[] bytes = cursor.getBlob(9);
                if (bytes == null || bytes.length != EMBEDDING_SIZE * 4) continue;
                float semantic = dot(queryEmbedding, decodeEmbedding(bytes));
                String name = valueOrEmpty(cursor.getString(2));
                String ocr = valueOrEmpty(cursor.getString(8));
                float lexical = lexicalScore(normalizedQuery, name, ocr);
                float score = semantic + lexical;
                if (score < minScore && lexical == 0.0f) continue;
                matched++;
                ScoredMatch candidate = new ScoredMatch(
                        cursor.getLong(0), cursor.getLong(4), semantic, lexical, score);
                if (retained.size() < retainedLimit) {
                    retained.add(candidate);
                } else if (BEST_MATCH_FIRST.compare(candidate, retained.peek()) < 0) {
                    retained.poll();
                    retained.add(candidate);
                }
            }
        }
        List<ScoredMatch> ranked = new ArrayList<>(retained);
        ranked.sort(BEST_MATCH_FIRST);

        JSONArray results = new JSONArray();
        int end = Math.min(ranked.size(), offset + max);
        for (int index = Math.min(offset, ranked.size()); index < end; index++) {
            ScoredPhoto photo = loadScoredPhoto(db, ranked.get(index));
            if (photo != null) results.put(photo.toJson(index + 1));
        }
        JSONObject data = status(db);
        data.put("query", query);
        data.put("results", results);
        data.put("offset", offset);
        data.put("returned", results.length());
        data.put("matched", matched);
        data.put("next_offset", end);
        data.put("has_more", end < matched);
        data.put("min_score", minScore);
        data.put("processing", "on_device");
        data.put("photo_bytes_uploaded", false);
        return data;
    }

    private ScoredPhoto loadScoredPhoto(SQLiteDatabase db, ScoredMatch match) {
        String[] columns = {
                "content_uri", "display_name", "mime_type", "date_taken", "date_modified",
                "width", "height", "ocr_text"
        };
        try (Cursor cursor = db.query("photos", columns, "media_id = ?",
                new String[]{Long.toString(match.mediaId)}, null, null, null, "1")) {
            if (!cursor.moveToFirst()) return null;
            return new ScoredPhoto(match.mediaId, cursor.getString(0),
                    valueOrEmpty(cursor.getString(1)), cursor.getString(2), cursor.getLong(3),
                    cursor.getLong(4), cursor.getInt(5), cursor.getInt(6),
                    valueOrEmpty(cursor.getString(7)), match.semantic, match.lexical, match.score);
        }
    }

    private void indexOne(SQLiteDatabase db, ContentResolver resolver, MediaPhoto photo)
            throws Exception {
        Uri uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, photo.id);
        Bitmap source = decodeForIndex(resolver, uri);
        try {
            String ocrText = recognize(source);
            float[] embedding = clip().embedImage(source);
            ContentValues values = photo.values(uri.toString());
            values.put("ocr_text", truncate(ocrText, MAX_OCR_CHARS));
            values.put("embedding", encodeEmbedding(embedding));
            values.putNull("error");
            values.put("indexed_at", System.currentTimeMillis());
            db.insertWithOnConflict("photos", null, values, SQLiteDatabase.CONFLICT_REPLACE);
        } finally {
            source.recycle();
        }
    }

    private void storeFailure(SQLiteDatabase db, MediaPhoto photo, String error) {
        Uri uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, photo.id);
        ContentValues values = photo.values(uri.toString());
        values.putNull("ocr_text");
        values.putNull("embedding");
        values.put("error", error);
        values.put("indexed_at", System.currentTimeMillis());
        db.insertWithOnConflict("photos", null, values, SQLiteDatabase.CONFLICT_REPLACE);
    }

    private JSONObject status(SQLiteDatabase db) throws JSONException {
        JSONObject data = new JSONObject();
        data.put("model_version", modelVersion);
        try (Cursor cursor = db.rawQuery(
                "SELECT COUNT(*), SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END), "
                        + "SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) FROM photos", null)) {
            cursor.moveToFirst();
            data.put("known", cursor.getLong(0));
            data.put("indexed", cursor.isNull(1) ? 0 : cursor.getLong(1));
            data.put("failed", cursor.isNull(2) ? 0 : cursor.getLong(2));
        }
        return data;
    }

    private Map<Long, ExistingRow> loadExisting(SQLiteDatabase db) {
        Map<Long, ExistingRow> rows = new HashMap<>();
        try (Cursor cursor = db.query("photos",
                new String[]{"media_id", "date_modified", "size_bytes", "error"},
                null, null, null, null, null)) {
            while (cursor.moveToNext()) {
                rows.put(cursor.getLong(0), new ExistingRow(
                        cursor.getLong(1), cursor.getLong(2), !cursor.isNull(3)));
            }
        }
        return rows;
    }

    private Bitmap decodeForIndex(ContentResolver resolver, Uri uri) throws IOException {
        if (Build.VERSION.SDK_INT >= 28) {
            ImageDecoder.Source source = ImageDecoder.createSource(resolver, uri);
            return ImageDecoder.decodeBitmap(source, (decoder, info, ignored) -> {
                int width = info.getSize().getWidth();
                int height = info.getSize().getHeight();
                int longest = Math.max(width, height);
                if (longest > 2048) {
                    float scale = 2048.0f / longest;
                    decoder.setTargetSize(Math.max(1, Math.round(width * scale)),
                            Math.max(1, Math.round(height * scale)));
                }
                decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            });
        }
        try (InputStream input = resolver.openInputStream(uri)) {
            if (input == null) throw new IOException("Cannot open photo");
            Bitmap bitmap = BitmapFactory.decodeStream(input);
            if (bitmap == null) throw new IOException("Cannot decode photo");
            return bitmap;
        }
    }

    private String recognize(Bitmap bitmap) throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<String> text = new AtomicReference<>("");
        AtomicReference<Exception> failure = new AtomicReference<>();
        recognizer.process(InputImage.fromBitmap(bitmap, 0))
                .addOnSuccessListener(result -> {
                    text.set(result.getText());
                    latch.countDown();
                })
                .addOnFailureListener(error -> {
                    failure.set(error);
                    latch.countDown();
                });
        if (!latch.await(45, TimeUnit.SECONDS)) throw new IOException("OCR timed out");
        if (failure.get() != null) throw failure.get();
        return text.get();
    }

    private ClipRuntime clip() throws Exception {
        if (clip == null) clip = new ClipRuntime(modelDirectory);
        return clip;
    }

    String modelVersion() { return modelVersion; }

    static void validateModel(File modelDirectory) throws Exception {
        try (ClipRuntime ignored = new ClipRuntime(modelDirectory)) {
            // Opening both ONNX sessions and the tokenizer validates the runtime contract.
        }
    }

    @Override
    public synchronized void close() throws IOException {
        recognizer.close();
        database.close();
        if (clip != null) clip.close();
    }

    private static float lexicalScore(String query, String name, String ocr) {
        if (query.isEmpty()) return 0.0f;
        String normalizedName = normalizeSearchText(name);
        String normalizedOcr = normalizeSearchText(ocr);
        if (normalizedName.contains(query)) return 0.55f;
        if (normalizedOcr.contains(query)) return 0.45f;
        String[] terms = query.split(" ");
        int matched = 0;
        for (String term : terms) {
            if (!term.isEmpty() && (normalizedName.contains(term) || normalizedOcr.contains(term))) {
                matched++;
            }
        }
        return matched == 0 ? 0.0f : 0.25f * matched / terms.length;
    }

    private static String normalizeSearchText(String value) {
        return Normalizer.normalize(valueOrEmpty(value), Normalizer.Form.NFKC)
                .toLowerCase(Locale.ROOT).replaceAll("\\s+", " ").trim();
    }

    private static float dot(float[] left, float[] right) {
        float total = 0.0f;
        for (int index = 0; index < left.length; index++) total += left[index] * right[index];
        return total;
    }

    private static byte[] encodeEmbedding(float[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) buffer.putFloat(value);
        return buffer.array();
    }

    private static float[] decodeEmbedding(byte[] bytes) {
        FloatBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        float[] values = new float[buffer.remaining()];
        buffer.get(values);
        return values;
    }

    private static String requiredString(JSONObject payload, String key, int max) throws JSONException {
        String value = payload.optString(key, "").trim();
        if (value.isEmpty()) throw new JSONException(key + " is required");
        if (value.codePointCount(0, value.length()) > max) {
            throw new JSONException(key + " is too long");
        }
        return value;
    }

    private static int strictInt(JSONObject payload, String key, int fallback, int min, int max)
            throws JSONException {
        if (!payload.has(key)) return fallback;
        Object value = payload.get(key);
        if (!(value instanceof Number)) throw new JSONException(key + " must be an integer");
        double number = ((Number) value).doubleValue();
        if (!Double.isFinite(number) || number != Math.rint(number) || number < min || number > max) {
            throw new JSONException(key + " must be an integer between " + min + " and " + max);
        }
        return (int) number;
    }

    private static String truncate(String value, int maxChars) {
        if (value == null || value.length() <= maxChars) return value == null ? "" : value;
        return value.substring(0, maxChars);
    }

    private static String valueOrEmpty(String value) {
        return value == null ? "" : value;
    }

    private static String boundedError(Exception error) {
        String message = error.getMessage();
        if (message == null || message.trim().isEmpty()) message = error.getClass().getSimpleName();
        return truncate(message, 500);
    }

    private static final class ClipRuntime implements Closeable {
        private static final float[] IMAGE_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
        private static final float[] IMAGE_STD = {0.26862954f, 0.26130258f, 0.27577711f};
        private final OrtEnvironment environment;
        private final OrtSession textSession;
        private final OrtSession visionSession;
        private final WordPieceTokenizer tokenizer;
        private final float[] projection;

        ClipRuntime(File modelDir) throws Exception {
            File textModel = requiredModelFile(modelDir, "clip_text.onnx");
            File visionModel = requiredModelFile(modelDir, "clip_vision.onnx");
            File dense = requiredModelFile(modelDir, "clip_dense.safetensors");
            File vocab = requiredModelFile(modelDir, "clip_vocab.txt");
            environment = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setIntraOpNumThreads(2);
            options.setInterOpNumThreads(1);
            textSession = environment.createSession(textModel.getAbsolutePath(), options);
            visionSession = environment.createSession(visionModel.getAbsolutePath(), options);
            tokenizer = new WordPieceTokenizer(vocab);
            projection = readProjection(dense);
            if (!textSession.getInputNames().containsAll(Arrays.asList("input_ids", "attention_mask"))
                    || !textSession.getOutputNames().contains("last_hidden_state")) {
                throw new IOException("Unexpected CLIP text model contract");
            }
            if (!visionSession.getInputNames().contains("pixel_values")
                    || !visionSession.getOutputNames().contains("image_embeds")) {
                throw new IOException("Unexpected CLIP vision model contract");
            }
        }

        float[] embedText(String text) throws Exception {
            WordPieceTokenizer.Encoding encoding = tokenizer.encode(text, MAX_QUERY_TOKENS);
            long[][] inputIds = {encoding.ids};
            long[][] attentionMask = {encoding.mask};
            try (OnnxTensor ids = OnnxTensor.createTensor(environment, inputIds);
                 OnnxTensor mask = OnnxTensor.createTensor(environment, attentionMask)) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", ids);
                inputs.put("attention_mask", mask);
                try (OrtSession.Result result = textSession.run(inputs)) {
                    Object value = result.get("last_hidden_state")
                            .orElseThrow(() -> new IOException("CLIP text output is missing")).getValue();
                    if (!(value instanceof float[][][])) {
                        throw new IOException("Unexpected CLIP text output type");
                    }
                    float[][] tokens = ((float[][][]) value)[0];
                    float[] pooled = new float[TEXT_HIDDEN_SIZE];
                    int count = 0;
                    for (int token = 0; token < tokens.length && token < encoding.mask.length; token++) {
                        if (encoding.mask[token] == 0) continue;
                        count++;
                        for (int hidden = 0; hidden < TEXT_HIDDEN_SIZE; hidden++) {
                            pooled[hidden] += tokens[token][hidden];
                        }
                    }
                    if (count == 0) throw new IOException("CLIP tokenizer produced no tokens");
                    for (int hidden = 0; hidden < pooled.length; hidden++) pooled[hidden] /= count;
                    float[] output = new float[EMBEDDING_SIZE];
                    for (int row = 0; row < EMBEDDING_SIZE; row++) {
                        int base = row * TEXT_HIDDEN_SIZE;
                        float sum = 0.0f;
                        for (int column = 0; column < TEXT_HIDDEN_SIZE; column++) {
                            sum += projection[base + column] * pooled[column];
                        }
                        output[row] = sum;
                    }
                    normalize(output);
                    return output;
                }
            }
        }

        float[] embedImage(Bitmap source) throws Exception {
            Bitmap bitmap = centerCrop(source, IMAGE_SIZE);
            float[] pixels = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
            int[] row = new int[IMAGE_SIZE];
            for (int y = 0; y < IMAGE_SIZE; y++) {
                bitmap.getPixels(row, 0, IMAGE_SIZE, 0, y, IMAGE_SIZE, 1);
                for (int x = 0; x < IMAGE_SIZE; x++) {
                    int color = row[x];
                    int position = y * IMAGE_SIZE + x;
                    pixels[position] = (((color >> 16) & 0xff) / 255.0f - IMAGE_MEAN[0]) / IMAGE_STD[0];
                    pixels[IMAGE_SIZE * IMAGE_SIZE + position] = (((color >> 8) & 0xff) / 255.0f - IMAGE_MEAN[1]) / IMAGE_STD[1];
                    pixels[2 * IMAGE_SIZE * IMAGE_SIZE + position] = ((color & 0xff) / 255.0f - IMAGE_MEAN[2]) / IMAGE_STD[2];
                }
            }
            bitmap.recycle();
            try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pixels),
                    new long[]{1, 3, IMAGE_SIZE, IMAGE_SIZE})) {
                try (OrtSession.Result result = visionSession.run(Collections.singletonMap("pixel_values", tensor))) {
                    Object value = result.get("image_embeds")
                            .orElseThrow(() -> new IOException("CLIP image output is missing")).getValue();
                    if (!(value instanceof float[][]) || ((float[][]) value)[0].length != EMBEDDING_SIZE) {
                        throw new IOException("Unexpected CLIP image output type");
                    }
                    float[] output = ((float[][]) value)[0].clone();
                    normalize(output);
                    return output;
                }
            }
        }

        @Override
        public void close() throws IOException {
            try {
                textSession.close();
                visionSession.close();
            } catch (OrtException error) {
                throw new IOException(error);
            }
        }

        private static Bitmap centerCrop(Bitmap source, int size) {
            float scale = Math.max((float) size / source.getWidth(), (float) size / source.getHeight());
            int width = Math.max(size, Math.round(source.getWidth() * scale));
            int height = Math.max(size, Math.round(source.getHeight() * scale));
            Bitmap scaled = Bitmap.createScaledBitmap(source, width, height, true);
            Bitmap crop = Bitmap.createBitmap(scaled, (width - size) / 2, (height - size) / 2, size, size);
            if (crop != scaled) scaled.recycle();
            return crop;
        }

        private static void normalize(float[] vector) throws IOException {
            double squared = 0.0;
            for (float value : vector) squared += value * value;
            double length = Math.sqrt(squared);
            if (!Double.isFinite(length) || length < 1e-12) {
                throw new IOException("CLIP returned an invalid embedding");
            }
            for (int index = 0; index < vector.length; index++) vector[index] /= (float) length;
        }

        private static File requiredModelFile(File directory, String name) throws IOException {
            File file = new File(directory, name);
            if (!file.isFile() || file.length() == 0) {
                throw new IOException("Installed photo model is missing " + name);
            }
            return file;
        }

        private static float[] readProjection(File file) throws IOException, JSONException {
            byte[] all;
            try (InputStream input = new java.io.FileInputStream(file)) {
                ByteArrayOutputStream output = new ByteArrayOutputStream();
                byte[] buffer = new byte[1024 * 1024];
                for (int read = input.read(buffer); read >= 0; read = input.read(buffer)) {
                    if (read > 0) output.write(buffer, 0, read);
                }
                all = output.toByteArray();
            }
            if (all.length < 8) throw new IOException("Invalid CLIP projection file");
            ByteBuffer bytes = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
            long headerLengthLong = bytes.getLong();
            if (headerLengthLong <= 0 || headerLengthLong > all.length - 8 || headerLengthLong > Integer.MAX_VALUE) {
                throw new IOException("Invalid CLIP projection header");
            }
            int headerLength = (int) headerLengthLong;
            byte[] headerBytes = new byte[headerLength];
            bytes.get(headerBytes);
            JSONObject header = new JSONObject(new String(headerBytes, StandardCharsets.UTF_8));
            JSONObject weight = header.getJSONObject("linear.weight");
            JSONArray shape = weight.getJSONArray("shape");
            JSONArray offsets = weight.getJSONArray("data_offsets");
            if (!"F32".equals(weight.getString("dtype"))
                    || shape.getInt(0) != EMBEDDING_SIZE || shape.getInt(1) != TEXT_HIDDEN_SIZE
                    || offsets.getInt(0) != 0 || offsets.getInt(1) != EMBEDDING_SIZE * TEXT_HIDDEN_SIZE * 4) {
                throw new IOException("Unexpected CLIP projection contract");
            }
            bytes.position(8 + headerLength);
            FloatBuffer floats = bytes.slice().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            float[] projection = new float[EMBEDDING_SIZE * TEXT_HIDDEN_SIZE];
            if (floats.remaining() < projection.length) throw new IOException("Truncated CLIP projection");
            floats.get(projection);
            return projection;
        }
    }

    private static final class WordPieceTokenizer {
        private final Map<String, Integer> vocabulary = new HashMap<>();
        private final int unknown;
        private final int cls;
        private final int sep;
        private final int pad;

        WordPieceTokenizer(File vocabFile) throws IOException {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new java.io.FileInputStream(vocabFile), StandardCharsets.UTF_8))) {
                String line;
                int id = 0;
                while ((line = reader.readLine()) != null) vocabulary.put(line, id++);
            }
            unknown = tokenId("[UNK]");
            cls = tokenId("[CLS]");
            sep = tokenId("[SEP]");
            pad = tokenId("[PAD]");
        }

        Encoding encode(String text, int length) {
            List<Integer> pieces = new ArrayList<>();
            pieces.add(cls);
            for (String token : basicTokens(text)) {
                if (pieces.size() >= length - 1) break;
                List<Integer> wordPieces = splitWordPiece(token);
                for (Integer piece : wordPieces) {
                    if (pieces.size() >= length - 1) break;
                    pieces.add(piece);
                }
            }
            pieces.add(sep);
            long[] ids = new long[length];
            long[] mask = new long[length];
            Arrays.fill(ids, pad);
            for (int index = 0; index < pieces.size(); index++) {
                ids[index] = pieces.get(index);
                mask[index] = 1;
            }
            return new Encoding(ids, mask);
        }

        private List<Integer> splitWordPiece(String token) {
            if (token.codePointCount(0, token.length()) > 100) return Collections.singletonList(unknown);
            List<Integer> output = new ArrayList<>();
            int start = 0;
            while (start < token.length()) {
                int end = token.length();
                Integer found = null;
                int foundEnd = start;
                while (end > start) {
                    String part = token.substring(start, end);
                    if (start > 0) part = "##" + part;
                    found = vocabulary.get(part);
                    if (found != null) {
                        foundEnd = end;
                        break;
                    }
                    end = token.offsetByCodePoints(end, -1);
                }
                if (found == null) return Collections.singletonList(unknown);
                output.add(found);
                start = foundEnd;
            }
            return output;
        }

        private List<String> basicTokens(String text) {
            List<String> tokens = new ArrayList<>();
            StringBuilder current = new StringBuilder();
            String cleaned = Normalizer.normalize(text, Normalizer.Form.NFC);
            for (int offset = 0; offset < cleaned.length();) {
                int codePoint = cleaned.codePointAt(offset);
                offset += Character.charCount(codePoint);
                if (isControl(codePoint)) continue;
                if (Character.isWhitespace(codePoint)) {
                    flush(current, tokens);
                } else if (isChinese(codePoint) || isPunctuation(codePoint)) {
                    flush(current, tokens);
                    tokens.add(new String(Character.toChars(codePoint)));
                } else {
                    current.appendCodePoint(codePoint);
                }
            }
            flush(current, tokens);
            return tokens;
        }

        private int tokenId(String token) throws IOException {
            Integer value = vocabulary.get(token);
            if (value == null) throw new IOException("CLIP vocabulary is missing " + token);
            return value;
        }

        private static void flush(StringBuilder current, List<String> tokens) {
            if (current.length() == 0) return;
            tokens.add(current.toString());
            current.setLength(0);
        }

        private static boolean isControl(int codePoint) {
            int type = Character.getType(codePoint);
            return type == Character.CONTROL || type == Character.FORMAT;
        }

        private static boolean isPunctuation(int codePoint) {
            int type = Character.getType(codePoint);
            return (codePoint >= 33 && codePoint <= 47) || (codePoint >= 58 && codePoint <= 64)
                    || (codePoint >= 91 && codePoint <= 96) || (codePoint >= 123 && codePoint <= 126)
                    || type == Character.CONNECTOR_PUNCTUATION || type == Character.DASH_PUNCTUATION
                    || type == Character.START_PUNCTUATION || type == Character.END_PUNCTUATION
                    || type == Character.INITIAL_QUOTE_PUNCTUATION
                    || type == Character.FINAL_QUOTE_PUNCTUATION
                    || type == Character.OTHER_PUNCTUATION;
        }

        private static boolean isChinese(int codePoint) {
            return (codePoint >= 0x4E00 && codePoint <= 0x9FFF)
                    || (codePoint >= 0x3400 && codePoint <= 0x4DBF)
                    || (codePoint >= 0x20000 && codePoint <= 0x2CEAF)
                    || (codePoint >= 0xF900 && codePoint <= 0xFAFF);
        }

        private static final class Encoding {
            final long[] ids;
            final long[] mask;

            Encoding(long[] ids, long[] mask) {
                this.ids = ids;
                this.mask = mask;
            }
        }
    }

    interface ModelVersionStore {
        void beginTransaction();
        String storedVersion();
        void clearPhotos();
        void writeVersion(String version);
        void markSuccessful();
        void endTransaction();
    }

    static void bindModelVersion(ModelVersionStore store, String requestedVersion) {
        if (requestedVersion == null || requestedVersion.isEmpty()) {
            throw new IllegalArgumentException("photo model version is required");
        }
        store.beginTransaction();
        try {
            String storedVersion = store.storedVersion();
            if (!requestedVersion.equals(storedVersion)) {
                store.clearPhotos();
                store.writeVersion(requestedVersion);
            }
            store.markSuccessful();
        } finally {
            store.endTransaction();
        }
    }

    private static final class IndexDatabase extends SQLiteOpenHelper {
        private final String modelVersion;

        IndexDatabase(Context context, String modelVersion) {
            super(context, "photo_semantic_index.db", null, 1);
            if (modelVersion == null || modelVersion.isEmpty()) {
                throw new IllegalArgumentException("photo model version is required");
            }
            this.modelVersion = modelVersion;
        }

        @Override
        public void onCreate(SQLiteDatabase db) {
            db.execSQL("CREATE TABLE photos ("
                    + "media_id INTEGER PRIMARY KEY, content_uri TEXT NOT NULL, display_name TEXT, "
                    + "mime_type TEXT, date_taken INTEGER NOT NULL, date_modified INTEGER NOT NULL, "
                    + "size_bytes INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, "
                    + "ocr_text TEXT, embedding BLOB, error TEXT, indexed_at INTEGER NOT NULL)");
            db.execSQL("CREATE INDEX photos_date_taken ON photos(date_taken DESC)");
            db.execSQL("CREATE TABLE photo_index_metadata ("
                    + "key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)");
        }

        @Override
        public void onOpen(SQLiteDatabase db) {
            super.onOpen(db);
            db.execSQL("CREATE TABLE IF NOT EXISTS photo_index_metadata ("
                    + "key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)");
            bindModelVersion(new ModelVersionStore() {
                @Override
                public void beginTransaction() {
                    db.beginTransaction();
                }

                @Override
                public String storedVersion() {
                    try (Cursor cursor = db.query("photo_index_metadata", new String[]{"value"},
                            "key = ?", new String[]{"model_version"}, null, null, null, "1")) {
                        return cursor.moveToFirst() ? cursor.getString(0) : null;
                    }
                }

                @Override
                public void clearPhotos() {
                    db.delete("photos", null, null);
                }

                @Override
                public void writeVersion(String version) {
                    db.execSQL("INSERT OR REPLACE INTO photo_index_metadata(key, value) "
                            + "VALUES ('model_version', ?)", new Object[]{version});
                }

                @Override
                public void markSuccessful() {
                    db.setTransactionSuccessful();
                }

                @Override
                public void endTransaction() {
                    db.endTransaction();
                }
            }, modelVersion);
        }

        @Override
        public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
            throw new IllegalStateException("Unsupported photo index schema upgrade");
        }
    }

    private static final class MediaPhoto {
        final long id;
        final String displayName;
        final String mimeType;
        final long dateTaken;
        final long modifiedSeconds;
        final long sizeBytes;
        final int width;
        final int height;

        MediaPhoto(long id, String displayName, String mimeType, long dateTaken,
                   long modifiedSeconds, long sizeBytes, int width, int height) {
            this.id = id;
            this.displayName = displayName;
            this.mimeType = mimeType;
            this.dateTaken = dateTaken;
            this.modifiedSeconds = modifiedSeconds;
            this.sizeBytes = sizeBytes;
            this.width = width;
            this.height = height;
        }

        static MediaPhoto fromCursor(Cursor cursor) {
            return new MediaPhoto(cursor.getLong(0), valueOrEmpty(cursor.getString(1)),
                    valueOrEmpty(cursor.getString(2)), cursor.getLong(3), cursor.getLong(4),
                    cursor.getLong(5), cursor.getInt(6), cursor.getInt(7));
        }

        ContentValues values(String uri) {
            ContentValues values = new ContentValues();
            values.put("media_id", id);
            values.put("content_uri", uri);
            values.put("display_name", displayName);
            values.put("mime_type", mimeType);
            values.put("date_taken", dateTaken);
            values.put("date_modified", modifiedSeconds);
            values.put("size_bytes", sizeBytes);
            values.put("width", width);
            values.put("height", height);
            return values;
        }
    }

    private static final class ExistingRow {
        final long modifiedSeconds;
        final long sizeBytes;
        final boolean failed;

        ExistingRow(long modifiedSeconds, long sizeBytes, boolean failed) {
            this.modifiedSeconds = modifiedSeconds;
            this.sizeBytes = sizeBytes;
            this.failed = failed;
        }
    }

    private static final class ScoredMatch {
        final long mediaId;
        final long dateTaken;
        final float semantic;
        final float lexical;
        final float score;

        ScoredMatch(long mediaId, long dateTaken, float semantic, float lexical, float score) {
            this.mediaId = mediaId;
            this.dateTaken = dateTaken;
            this.semantic = semantic;
            this.lexical = lexical;
            this.score = score;
        }
    }

    private static final class ScoredPhoto {
        final long mediaId;
        final String contentUri;
        final String displayName;
        final String mimeType;
        final long dateTaken;
        final long dateModified;
        final int width;
        final int height;
        final String ocr;
        final float semantic;
        final float lexical;
        final float score;

        ScoredPhoto(long mediaId, String contentUri, String displayName, String mimeType,
                    long dateTaken, long dateModified, int width, int height, String ocr,
                    float semantic, float lexical, float score) {
            this.mediaId = mediaId;
            this.contentUri = contentUri;
            this.displayName = displayName;
            this.mimeType = mimeType;
            this.dateTaken = dateTaken;
            this.dateModified = dateModified;
            this.width = width;
            this.height = height;
            this.ocr = ocr;
            this.semantic = semantic;
            this.lexical = lexical;
            this.score = score;
        }

        JSONObject toJson(int resultIndex) throws JSONException {
            JSONObject item = new JSONObject();
            item.put("result_index", resultIndex);
            item.put("media_id", mediaId);
            item.put("content_uri", contentUri);
            item.put("filename", displayName);
            item.put("mime_type", mimeType);
            item.put("date_taken_ms", dateTaken);
            item.put("date_modified_s", dateModified);
            item.put("width", width);
            item.put("height", height);
            item.put("semantic_score", Math.round(semantic * 10_000.0f) / 10_000.0f);
            item.put("lexical_score", Math.round(lexical * 10_000.0f) / 10_000.0f);
            item.put("score", Math.round(score * 10_000.0f) / 10_000.0f);
            if (!ocr.isEmpty()) item.put("ocr_excerpt", truncate(ocr.replaceAll("\\s+", " "), 240));
            return item;
        }
    }
}
