package com.grewwc.aserve;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Strict, platform-independent contract for downloadable photo models. */
final class PhotoModelContract {
    static final String MANIFEST_PATH = "/a-serve-photo-models.json";
    static final String ARCHIVE_PATH = "/a-serve-photo-models.zip";
    static final long MAX_MANIFEST_BYTES = 64 * 1024;
    static final long MAX_ARCHIVE_BYTES = 2_500_000_000L;
    static final long MAX_TOTAL_FILE_BYTES = 2_500_000_000L;
    static final List<String> FILE_NAMES = Collections.unmodifiableList(Arrays.asList(
            "clip_text.onnx", "clip_vision.onnx", "clip_dense.safetensors", "clip_vocab.txt"));

    private static final Set<String> ROOT_FIELDS = fields("version", "archive", "files");
    private static final Set<String> ARCHIVE_FIELDS = fields("path", "size", "sha256");
    private static final Set<String> FILE_FIELDS = fields("name", "size", "sha256");
    private static final Map<String, Long> FILE_LIMITS;

    static {
        Map<String, Long> limits = new LinkedHashMap<>();
        limits.put("clip_text.onnx", 1_500_000_000L);
        limits.put("clip_vision.onnx", 1_500_000_000L);
        limits.put("clip_dense.safetensors", 8_000_000L);
        limits.put("clip_vocab.txt", 32_000_000L);
        FILE_LIMITS = Collections.unmodifiableMap(limits);
    }

    private PhotoModelContract() {}

    static Manifest validate(Map<String, Object> root) {
        requireFields(root, ROOT_FIELDS, "manifest");
        String version = requiredString(root, "version", 64);
        if (!version.matches("[A-Za-z0-9][A-Za-z0-9._-]{0,63}")) {
            throw new IllegalArgumentException("invalid model version");
        }
        Map<String, Object> archiveObject = requiredMap(root, "archive");
        requireFields(archiveObject, ARCHIVE_FIELDS, "archive");
        String archivePath = requiredString(archiveObject, "path", 64);
        if (!ARCHIVE_PATH.equals(archivePath)) {
            throw new IllegalArgumentException("archive path must be " + ARCHIVE_PATH);
        }
        long archiveSize = requiredInteger(archiveObject, "size", 1, MAX_ARCHIVE_BYTES);
        String archiveSha256 = requiredSha256(archiveObject, "sha256");

        Object rawFiles = root.get("files");
        if (!(rawFiles instanceof List)) throw new IllegalArgumentException("files must be an array");
        List<?> inputFiles = (List<?>) rawFiles;
        if (inputFiles.size() != FILE_NAMES.size()) {
            throw new IllegalArgumentException("manifest must contain exactly four model files");
        }
        Map<String, FileSpec> byName = new LinkedHashMap<>();
        long total = 0;
        for (Object rawFile : inputFiles) {
            if (!(rawFile instanceof Map)) throw new IllegalArgumentException("file entry must be an object");
            @SuppressWarnings("unchecked")
            Map<String, Object> fileObject = (Map<String, Object>) rawFile;
            requireFields(fileObject, FILE_FIELDS, "file entry");
            String name = requiredString(fileObject, "name", 64);
            Long limit = FILE_LIMITS.get(name);
            if (limit == null) throw new IllegalArgumentException("unknown model file " + name);
            if (byName.containsKey(name)) throw new IllegalArgumentException("duplicate model file " + name);
            long size = requiredInteger(fileObject, "size", 1, limit);
            String sha256 = requiredSha256(fileObject, "sha256");
            total = Math.addExact(total, size);
            if (total > MAX_TOTAL_FILE_BYTES) throw new IllegalArgumentException("model files are too large");
            byName.put(name, new FileSpec(name, size, sha256));
        }
        if (!byName.keySet().equals(new LinkedHashSet<>(FILE_NAMES))) {
            throw new IllegalArgumentException("manifest model file set is incomplete");
        }
        List<FileSpec> files = new ArrayList<>();
        for (String name : FILE_NAMES) files.add(byName.get(name));
        return new Manifest(version, archiveSize, archiveSha256, files, total);
    }

    static int resumeResponseMode(long offset, int responseCode, String contentRange,
                                  long expectedSize) {
        if (offset == 0) {
            if (responseCode != 200) throw new IllegalArgumentException("archive server did not return 200");
            return 0;
        }
        if (responseCode == 200) return 0;
        if (responseCode != 206) throw new IllegalArgumentException("archive resume did not return 206");
        String expected = "bytes " + offset + "-";
        String suffix = "/" + expectedSize;
        if (contentRange == null || !contentRange.startsWith(expected) || !contentRange.endsWith(suffix)) {
            throw new IllegalArgumentException("invalid Content-Range for model archive");
        }
        int dash = contentRange.indexOf('-', 6);
        int slash = contentRange.lastIndexOf('/');
        if (dash < 0 || slash <= dash + 1) throw new IllegalArgumentException("invalid Content-Range");
        try {
            if (Long.parseLong(contentRange.substring(dash + 1, slash)) != expectedSize - 1) {
                throw new IllegalArgumentException("incomplete archive range");
            }
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException("invalid Content-Range", error);
        }
        return 1;
    }

    static long resumeOffset(long partialLength, long expectedSize, boolean identityMatches) {
        if (!identityMatches || partialLength <= 0 || partialLength > expectedSize) return 0;
        return partialLength;
    }

    static Resolution resolve(String installedVersion, String installedArchiveSha256,
                              Manifest remote, boolean manifestFetchFailed) {
        if (manifestFetchFailed) {
            return installedVersion == null ? Resolution.UNAVAILABLE : Resolution.INSTALLED_FALLBACK;
        }
        if (remote == null) throw new IllegalArgumentException("remote manifest is required");
        if (installedVersion == null) return Resolution.DOWNLOAD;
        if (installedVersion.equals(remote.version)) {
            return remote.archiveSha256.equals(installedArchiveSha256)
                    ? Resolution.CURRENT : Resolution.VERSION_CONFLICT;
        }
        return Resolution.DOWNLOAD;
    }

    private static void requireFields(Map<String, Object> value, Set<String> expected, String label) {
        if (value == null || !value.keySet().equals(expected)) {
            throw new IllegalArgumentException(label + " fields must be exactly " + expected);
        }
    }

    private static String requiredString(Map<String, Object> value, String key, int maxLength) {
        Object raw = value.get(key);
        if (!(raw instanceof String)) throw new IllegalArgumentException(key + " must be a string");
        String text = (String) raw;
        if (text.isEmpty() || text.length() > maxLength || !text.equals(text.trim())) {
            throw new IllegalArgumentException("invalid " + key);
        }
        return text;
    }

    private static String requiredSha256(Map<String, Object> value, String key) {
        String hash = requiredString(value, key, 64).toLowerCase(Locale.ROOT);
        if (!hash.matches("[0-9a-f]{64}")) throw new IllegalArgumentException("invalid " + key);
        return hash;
    }

    private static long requiredInteger(Map<String, Object> value, String key, long min, long max) {
        Object raw = value.get(key);
        if (!(raw instanceof Number)) throw new IllegalArgumentException(key + " must be an integer");
        Number number = (Number) raw;
        double asDouble = number.doubleValue();
        long asLong = number.longValue();
        if (!Double.isFinite(asDouble) || asDouble != asLong || asLong < min || asLong > max) {
            throw new IllegalArgumentException("invalid " + key);
        }
        return asLong;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> requiredMap(Map<String, Object> value, String key) {
        Object raw = value.get(key);
        if (!(raw instanceof Map)) throw new IllegalArgumentException(key + " must be an object");
        return (Map<String, Object>) raw;
    }

    private static Set<String> fields(String... names) {
        return Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(names)));
    }

    enum Resolution { CURRENT, INSTALLED_FALLBACK, DOWNLOAD, VERSION_CONFLICT, UNAVAILABLE }

    static final class Manifest {
        final String version;
        final long archiveSize;
        final String archiveSha256;
        final List<FileSpec> files;
        final long totalFileSize;

        Manifest(String version, long archiveSize, String archiveSha256,
                 List<FileSpec> files, long totalFileSize) {
            this.version = version;
            this.archiveSize = archiveSize;
            this.archiveSha256 = archiveSha256;
            this.files = Collections.unmodifiableList(new ArrayList<>(files));
            this.totalFileSize = totalFileSize;
        }

        FileSpec file(String name) {
            for (FileSpec file : files) if (file.name.equals(name)) return file;
            throw new IllegalArgumentException("unknown model file " + name);
        }
    }

    static final class FileSpec {
        final String name;
        final long size;
        final String sha256;

        FileSpec(String name, long size, String sha256) {
            this.name = name;
            this.size = size;
            this.sha256 = sha256;
        }
    }
}
