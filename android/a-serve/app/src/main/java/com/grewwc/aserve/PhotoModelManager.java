package com.grewwc.aserve;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/** Downloads and atomically installs verified photo models into app-private storage. */
final class PhotoModelManager {
    private static final int CONNECT_TIMEOUT_MS = 15_000;
    private static final int READ_TIMEOUT_MS = 30_000;
    private static final int MAX_POINTER_BYTES = 8 * 1024;
    private static final int MAX_REDIRECTS = 5;
    private static final int MAX_URL_CHARS = 8 * 1024;

    private final File root;
    private Active cachedActive;

    PhotoModelManager(Context context) {
        root = new File(context.getApplicationContext().getFilesDir(), "photo-models-v2");
    }

    Prepared prepare(String origin) throws IOException {
        requireTrustedOrigin(origin);
        Active active;
        try {
            active = loadActiveVerified();
        } catch (Exception error) {
            active = null;
        }
        PhotoModelContract.Manifest current = PhotoModelContract.current();
        PhotoModelContract.Resolution resolution = PhotoModelContract.resolve(
                active == null ? null : active.version,
                active == null ? null : active.manifest.archiveSha256,
                current);
        if (resolution == PhotoModelContract.Resolution.CURRENT) {
            return Prepared.ready(active, "current", true, null);
        }
        if (resolution == PhotoModelContract.Resolution.VERSION_CONFLICT) {
            throw new IOException("photo model version was changed without a new version identifier");
        }
        return Prepared.download(active,
                new Fetched(current, PhotoModelContract.currentManifestJson()));
    }

    Active install(String origin, Prepared prepared, AtomicBoolean cancelled,
                   Progress progress, Verifier verifier) throws Exception {
        if (!prepared.needsDownload || prepared.fetched == null) {
            throw new IllegalArgumentException("photo model download was not prepared");
        }
        requireTrustedOrigin(origin);
        try {
            return installCurrent(prepared, cancelled, progress, verifier);
        } catch (CancelledException error) {
            throw error;
        } catch (Exception error) {
            if (prepared.active != null) {
                return prepared.active.asFallback(boundedMessage(error));
            }
            throw error;
        }
    }

    private Active installCurrent(Prepared prepared, AtomicBoolean cancelled,
                                  Progress progress, Verifier verifier) throws Exception {
        ensureRoot();
        PhotoModelContract.Manifest manifest = prepared.fetched.manifest;
        File downloads = new File(root, "downloads");
        if (!downloads.isDirectory() && !downloads.mkdirs()) {
            throw new IOException("cannot create photo model download directory");
        }
        File stagingRoot = new File(root, "staging");
        if (!stagingRoot.isDirectory() && !stagingRoot.mkdirs()) {
            throw new IOException("cannot create photo model staging root");
        }
        File staging = new File(stagingRoot, UUID.randomUUID().toString());
        if (!staging.mkdir()) throw new IOException("cannot create photo model staging directory");

        boolean moved = false;
        try {
            long completed = 0;
            progress.update("Downloading photo models", 0, manifest.totalFileSize);
            for (PhotoModelContract.FileSpec file : manifest.files) {
                checkCancelled(cancelled);
                File downloaded = downloadFile(downloads, file, completed,
                        manifest.totalFileSize, cancelled, progress);
                PhotoModelArchive.copyVerified(downloaded, new File(staging, file.name), file,
                        () -> checkCancelled(cancelled));
                completed += file.size;
            }
            progress.update("Verifying photo models", manifest.totalFileSize,
                    manifest.totalFileSize);
            writeSynced(new File(staging, "manifest.json"), prepared.fetched.rawJson);
            PhotoModelArchive.verifyDirectory(staging, manifest);
            checkCancelled(cancelled);
            verifier.verify(staging);
            checkCancelled(cancelled);

            File models = new File(root, "models");
            if (!models.isDirectory() && !models.mkdirs()) {
                throw new IOException("cannot create photo model directory");
            }
            String directoryName = manifest.version + "-" + UUID.randomUUID();
            File installed = new File(models, directoryName);
            atomicMove(staging, installed, false);
            moved = true;
            activateOrCleanup(installed, () -> activate(directoryName, manifest));
            Active active = new Active(manifest.version, installed, manifest,
                    prepared.active == null ? "downloaded" : "updated", true, null);
            cachedActive = active;
            for (PhotoModelContract.FileSpec file : manifest.files) {
                downloadPartial(downloads, file).delete();
            }
            return active;
        } finally {
            if (!moved) deleteTree(staging);
        }
    }

    private File downloadFile(File downloads, PhotoModelContract.FileSpec file,
                              long completedBefore, long totalBytes, AtomicBoolean cancelled,
                              Progress progress) throws IOException {
        if (file.url == null || file.size > file.maxBytes) {
            throw new IOException("invalid embedded metadata for " + file.name);
        }
        File partial = downloadPartial(downloads, file);
        long offset = preparePartialForResume(partial, file.size);
        if (offset == file.size) {
            try {
                PhotoModelArchive.verifyFile(partial, file);
                progress.update("Downloading photo models", completedBefore + file.size, totalBytes);
                return partial;
            } catch (IOException error) {
                resetPartial(partial);
                offset = 0;
            }
        }

        checkCancelled(cancelled);
        HttpURLConnection connection = openDownload(file.url, offset, cancelled);
        try {
            int code = connection.getResponseCode();
            int mode;
            try {
                mode = PhotoModelContract.resumeResponseMode(
                        offset, code, connection.getHeaderField("Content-Range"), file.size);
            } catch (IllegalArgumentException error) {
                throw new IOException(error.getMessage(), error);
            }
            if (mode == 0 && offset > 0) {
                resetPartial(partial);
                offset = 0;
                progress.update("Downloading photo models", completedBefore, totalBytes);
            }
            long remaining = file.size - offset;
            long contentLength = connection.getContentLengthLong();
            if (contentLength >= 0 && contentLength != remaining) {
                throw new IOException("photo model response size mismatch for " + file.name);
            }
            try (InputStream input = connection.getInputStream();
                 RandomAccessFile output = new RandomAccessFile(partial, "rw")) {
                output.seek(offset);
                byte[] buffer = new byte[64 * 1024];
                long downloaded = offset;
                int read;
                while ((read = input.read(buffer)) != -1) {
                    checkCancelled(cancelled);
                    downloaded += read;
                    if (downloaded > file.size || downloaded > file.maxBytes) {
                        throw new IOException("photo model exceeds declared size for " + file.name);
                    }
                    output.write(buffer, 0, read);
                    progress.update("Downloading photo models", completedBefore + downloaded,
                            totalBytes);
                }
                output.getFD().sync();
                if (downloaded != file.size) {
                    throw new IOException("photo model download is incomplete for " + file.name);
                }
            }
        } finally {
            connection.disconnect();
        }
        verifyDownloadedFile(partial, file);
        return partial;
    }

    private static File downloadPartial(File downloads, PhotoModelContract.FileSpec file) {
        return new File(downloads, file.sha256 + "-" + file.name + ".partial");
    }

    private static HttpURLConnection openDownload(String rawUrl, long offset,
                                                   AtomicBoolean cancelled) throws IOException {
        URL url = new URL(rawUrl);
        for (int redirects = 0; ; redirects++) {
            checkCancelled(cancelled);
            validateDownloadUrl(url);
            HttpURLConnection connection = open(url);
            connection.setRequestProperty("Accept-Encoding", "identity");
            if (offset > 0) connection.setRequestProperty("Range", "bytes=" + offset + "-");
            int code = connection.getResponseCode();
            if (!isRedirect(code)) return connection;
            if (redirects >= MAX_REDIRECTS) {
                connection.disconnect();
                throw new IOException("too many photo model redirects");
            }
            String location = connection.getHeaderField("Location");
            connection.disconnect();
            if (location == null || location.trim().isEmpty()) {
                throw new IOException("photo model redirect has no location");
            }
            url = new URL(url, location);
        }
    }

    private static boolean isRedirect(int code) {
        return code == 301 || code == 302 || code == 303 || code == 307 || code == 308;
    }

    static void validateDownloadUrl(URL url) throws IOException {
        String value = url.toExternalForm();
        String host = url.getHost().toLowerCase(Locale.ROOT);
        boolean allowedHost = host.equals("huggingface.co") || host.endsWith(".huggingface.co")
                || host.equals("hf.co") || host.endsWith(".hf.co");
        if (value.length() > MAX_URL_CHARS || !"https".equalsIgnoreCase(url.getProtocol())
                || (url.getPort() != -1 && url.getPort() != 443)
                || url.getUserInfo() != null || url.getRef() != null || url.getPath().isEmpty()
                || host.endsWith(".") || !host.matches("[a-z0-9.-]+") || !allowedHost) {
            throw new IOException("untrusted photo model download URL");
        }
    }

    private Active loadActiveVerified() throws IOException, JSONException {
        if (cachedActive != null) return cachedActive;
        File pointer = new File(root, "active.json");
        if (!pointer.isFile()) return null;
        JSONObject json = new JSONObject(new String(readFileLimited(pointer, MAX_POINTER_BYTES),
                StandardCharsets.UTF_8));
        if (json.length() != 3 || !json.has("directory") || !json.has("version")
                || !json.has("archive_sha256")) {
            throw new IOException("invalid active photo model pointer");
        }
        String directoryName = json.getString("directory");
        if (!directoryName.matches("[A-Za-z0-9._-]{1,128}")) {
            throw new IOException("invalid active photo model directory");
        }
        File directory = new File(new File(root, "models"), directoryName);
        File manifestFile = new File(directory, "manifest.json");
        String raw = new String(readFileLimited(manifestFile, PhotoModelContract.MAX_MANIFEST_BYTES),
                StandardCharsets.UTF_8);
        PhotoModelContract.Manifest stored = parseManifest(raw);
        if (!json.getString("version").equals(stored.version)
                || !json.getString("archive_sha256").equals(stored.archiveSha256)) {
            throw new IOException("active photo model pointer does not match its manifest");
        }
        PhotoModelContract.Manifest verified = PhotoModelContract.isCurrent(stored)
                ? PhotoModelContract.current() : stored;
        PhotoModelArchive.verifyDirectory(directory, verified);
        cachedActive = new Active(verified.version, directory, verified,
                "installed", false, null);
        return cachedActive;
    }

    private void activate(String directoryName, PhotoModelContract.Manifest manifest)
            throws IOException, JSONException {
        JSONObject pointer = new JSONObject();
        pointer.put("directory", directoryName);
        pointer.put("version", manifest.version);
        pointer.put("archive_sha256", manifest.archiveSha256);
        File pending = new File(root, "active.pending-" + UUID.randomUUID());
        writeSynced(pending, pointer.toString());
        try {
            atomicMove(pending, new File(root, "active.json"), true);
        } finally {
            pending.delete();
        }
    }

    private void ensureRoot() throws IOException {
        if (!root.isDirectory() && !root.mkdirs()) throw new IOException("cannot create photo model storage");
    }

    private static HttpURLConnection open(URL url) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setInstanceFollowRedirects(false);
        connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
        connection.setReadTimeout(READ_TIMEOUT_MS);
        connection.setUseCaches(false);
        return connection;
    }

    private static void requireTrustedOrigin(String origin) throws IOException {
        if (origin == null || !origin.equals(ServerUrl.normalize(origin))) {
            throw new IOException("photo models require the configured HTTPS server origin");
        }
    }

    private static PhotoModelContract.Manifest parseManifest(String raw)
            throws JSONException {
        return PhotoModelContract.validate(asMap(new JSONObject(raw)));
    }

    private static Map<String, Object> asMap(JSONObject object) throws JSONException {
        Map<String, Object> result = new LinkedHashMap<>();
        Iterator<String> keys = object.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            Object value = object.get(key);
            if (value instanceof JSONObject) value = asMap((JSONObject) value);
            else if (value instanceof JSONArray) value = asList((JSONArray) value);
            else if (value == JSONObject.NULL) value = null;
            result.put(key, value);
        }
        return result;
    }

    private static List<Object> asList(JSONArray array) throws JSONException {
        List<Object> result = new ArrayList<>();
        for (int index = 0; index < array.length(); index++) {
            Object value = array.get(index);
            if (value instanceof JSONObject) value = asMap((JSONObject) value);
            else if (value instanceof JSONArray) value = asList((JSONArray) value);
            else if (value == JSONObject.NULL) value = null;
            result.add(value);
        }
        return result;
    }

    private static byte[] readLimited(InputStream input, long max) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        long total = 0;
        int read;
        while ((read = input.read(buffer)) != -1) {
            total += read;
            if (total > max) throw new IOException("response exceeds size limit");
            output.write(buffer, 0, read);
        }
        return output.toByteArray();
    }

    private static byte[] readFileLimited(File file, long max) throws IOException {
        if (!file.isFile() || file.length() > max) throw new IOException("model metadata is missing or too large");
        try (InputStream input = new FileInputStream(file)) {
            return readLimited(input, max);
        }
    }

    private static void writeSynced(File file, String value) throws IOException {
        try (FileOutputStream output = new FileOutputStream(file)) {
            output.write(value.getBytes(StandardCharsets.UTF_8));
            output.getFD().sync();
        }
    }

    private static void atomicMove(File source, File destination, boolean replace) throws IOException {
        try {
            if (replace) {
                Files.move(source.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.move(source.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE);
            }
        } catch (AtomicMoveNotSupportedException error) {
            throw new IOException("app-private storage does not support atomic model activation", error);
        }
    }

    private static void truncate(File file) throws IOException {
        try (RandomAccessFile output = new RandomAccessFile(file, "rw")) {
            output.setLength(0);
            output.getFD().sync();
        }
    }

    private static void resetPartial(File file) throws IOException {
        if (file.isFile() && !file.delete()) truncate(file);
    }

    static long preparePartialForResume(File partial, long expectedSize) throws IOException {
        long offset = PhotoModelContract.resumeOffset(
                partial.isFile() ? partial.length() : 0, expectedSize, partial.isFile());
        if (offset == 0 && partial.isFile() && partial.length() > 0) truncate(partial);
        return offset;
    }

    static void verifyDownloadedFile(File partial, PhotoModelContract.FileSpec expected)
            throws IOException {
        try {
            PhotoModelArchive.verifyFile(partial, expected);
        } catch (IOException error) {
            resetPartial(partial);
            throw error;
        }
    }

    static void activateOrCleanup(File installed, Activation activation) throws Exception {
        try {
            activation.run();
        } catch (Exception error) {
            if (!deleteTree(installed)) {
                error.addSuppressed(new IOException(
                        "cannot remove inactive photo model after activation failure"));
            }
            throw error;
        }
    }

    private static void checkCancelled(AtomicBoolean cancelled) throws CancelledException {
        if (cancelled.get()) throw new CancelledException();
    }

    private static boolean deleteTree(File file) {
        if (file == null || !file.exists()) return true;
        boolean deleted = true;
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) deleted &= deleteTree(child);
        }
        return file.delete() && deleted;
    }

    private static String boundedMessage(Exception error) {
        String message = error.getMessage();
        if (message == null || message.isEmpty()) message = error.getClass().getSimpleName();
        return message.length() <= 300 ? message : message.substring(0, 300);
    }

    interface Progress {
        void update(String stage, long completed, long total);
    }

    interface Verifier {
        void verify(File directory) throws Exception;
    }

    interface Activation {
        void run() throws Exception;
    }

    static final class Prepared {
        final Active active;
        final Fetched fetched;
        final boolean needsDownload;
        final String state;
        final boolean manifestChecked;
        final String manifestError;

        private Prepared(Active active, Fetched fetched, boolean needsDownload, String state,
                         boolean manifestChecked, String manifestError) {
            this.active = active;
            this.fetched = fetched;
            this.needsDownload = needsDownload;
            this.state = state;
            this.manifestChecked = manifestChecked;
            this.manifestError = manifestError;
        }

        static Prepared ready(Active active, String state, boolean checked, String error) {
            return new Prepared(active, null, false, state, checked, error);
        }

        static Prepared download(Active active, Fetched fetched) {
            return new Prepared(active, fetched, true, "download_required", true, null);
        }
    }

    static final class Active {
        final String version;
        final File directory;
        final PhotoModelContract.Manifest manifest;
        final String state;
        final boolean downloaded;
        final String manifestError;

        Active(String version, File directory, PhotoModelContract.Manifest manifest,
               String state, boolean downloaded, String manifestError) {
            this.version = version;
            this.directory = directory;
            this.manifest = manifest;
            this.state = state;
            this.downloaded = downloaded;
            this.manifestError = manifestError;
        }

        Active withPrepared(Prepared prepared) {
            return new Active(version, directory, manifest, prepared.state, false,
                    prepared.manifestError);
        }

        Active asFallback(String error) {
            return new Active(version, directory, manifest, "installed_fallback", false, error);
        }
    }

    static final class Fetched {
        final PhotoModelContract.Manifest manifest;
        final String rawJson;

        Fetched(PhotoModelContract.Manifest manifest, String rawJson) {
            this.manifest = manifest;
            this.rawJson = rawJson;
        }
    }

    static final class CancelledException extends IOException {
        CancelledException() { super("photo model download cancelled"); }
    }
}
