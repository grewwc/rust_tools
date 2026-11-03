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

/** Immutable metadata and validation rules for the Android photo models. */
final class PhotoModelContract {
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
    private static final Manifest CURRENT;

    static {
        Map<String, Long> limits = new LinkedHashMap<>();
        limits.put("clip_text.onnx", 160L * 1024 * 1024);
        limits.put("clip_vision.onnx", 110L * 1024 * 1024);
        limits.put("clip_dense.safetensors", 5L * 1024 * 1024);
        limits.put("clip_vocab.txt", 5L * 1024 * 1024);
        FILE_LIMITS = Collections.unmodifiableMap(limits);

        String textRevision = "58edf8cada9e398793dca955574a48cbb7f18be2";
        String visionRevision = "6b7c523703ae26e39be4b03774966f8571c7f523";
        List<FileSpec> files = Arrays.asList(
                new FileSpec("clip_text.onnx",
                        "https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/"
                                + textRevision + "/onnx/model_qint8_arm64.onnx",
                        135_336_307L,
                        "3bed77e83926519660b5c0833cb3321cfde952331edec4f7d501c7d384b8a8a7",
                        limits.get("clip_text.onnx")),
                new FileSpec("clip_vision.onnx",
                        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
                                + visionRevision + "/onnx/vision_model_quantized.onnx",
                        89_117_001L,
                        "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
                        limits.get("clip_vision.onnx")),
                new FileSpec("clip_dense.safetensors",
                        "https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/"
                                + textRevision + "/2_Dense/model.safetensors",
                        1_572_984L,
                        "d12568dc7300970a4d3dbb49068ad16cd89b99840b74b026f8e48071e9414f74",
                        limits.get("clip_dense.safetensors")),
                new FileSpec("clip_vocab.txt",
                        "https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1/resolve/"
                                + textRevision + "/vocab.txt",
                        995_526L,
                        "fe0fda7c425b48c516fc8f160d594c8022a0808447475c1a7c6d6479763f310c",
                        limits.get("clip_vocab.txt")));
        long total = 0;
        for (FileSpec file : files) total = Math.addExact(total, file.size);
        CURRENT = new Manifest("1", 161_554_433L,
                "d11cbf1b0b32289e7798bf501229556c4ebf2d48768c6b1fa939c8abb64d59de",
                files, total);
    }

    private PhotoModelContract() {}

    static Manifest current() {
        return CURRENT;
    }

    /** Keeps the previous on-disk schema so already installed verified sets remain readable. */
    static String currentManifestJson() {
        StringBuilder output = new StringBuilder(1024);
        output.append("{\"version\":\"").append(CURRENT.version)
                .append("\",\"archive\":{\"path\":\"").append(ARCHIVE_PATH)
                .append("\",\"size\":").append(CURRENT.archiveSize)
                .append(",\"sha256\":\"").append(CURRENT.archiveSha256)
                .append("\"},\"files\":[");
        for (int index = 0; index < CURRENT.files.size(); index++) {
            FileSpec file = CURRENT.files.get(index);
            if (index > 0) output.append(',');
            output.append("{\"name\":\"").append(file.name)
                    .append("\",\"size\":").append(file.size)
                    .append(",\"sha256\":\"").append(file.sha256).append("\"}");
        }
        return output.append("]}").toString();
    }

    static boolean isCurrent(Manifest manifest) {
        if (manifest == null || !CURRENT.version.equals(manifest.version)
                || CURRENT.archiveSize != manifest.archiveSize
                || !CURRENT.archiveSha256.equals(manifest.archiveSha256)
                || CURRENT.totalFileSize != manifest.totalFileSize
                || CURRENT.files.size() != manifest.files.size()) {
            return false;
        }
        for (int index = 0; index < CURRENT.files.size(); index++) {
            FileSpec expected = CURRENT.files.get(index);
            FileSpec actual = manifest.files.get(index);
            if (!expected.name.equals(actual.name) || expected.size != actual.size
                    || !expected.sha256.equals(actual.sha256)) return false;
        }
        return true;
    }

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
            if (responseCode != 200) throw new IllegalArgumentException("model server did not return 200");
            return 0;
        }
        if (responseCode == 200) return 0;
        if (responseCode != 206) throw new IllegalArgumentException("model resume did not return 206");
        String expected = "bytes " + offset + "-";
        String suffix = "/" + expectedSize;
        if (contentRange == null || !contentRange.startsWith(expected) || !contentRange.endsWith(suffix)) {
            throw new IllegalArgumentException("invalid Content-Range for model file");
        }
        int dash = contentRange.indexOf('-', 6);
        int slash = contentRange.lastIndexOf('/');
        if (dash < 0 || slash <= dash + 1) throw new IllegalArgumentException("invalid Content-Range");
        try {
            if (Long.parseLong(contentRange.substring(dash + 1, slash)) != expectedSize - 1) {
                throw new IllegalArgumentException("incomplete model file range");
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
                              Manifest current) {
        if (current == null) throw new IllegalArgumentException("current manifest is required");
        if (installedVersion == null) return Resolution.DOWNLOAD;
        if (installedVersion.equals(current.version)) {
            return current.archiveSha256.equals(installedArchiveSha256)
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

    enum Resolution { CURRENT, DOWNLOAD, VERSION_CONFLICT }

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
        final String url;
        final long size;
        final String sha256;
        final long maxBytes;

        FileSpec(String name, long size, String sha256) {
            this(name, null, size, sha256, FILE_LIMITS.getOrDefault(name, size));
        }

        FileSpec(String name, String url, long size, String sha256, long maxBytes) {
            if (size <= 0 || size > maxBytes) throw new IllegalArgumentException("invalid model file size");
            this.name = name;
            this.url = url;
            this.size = size;
            this.sha256 = sha256;
            this.maxBytes = maxBytes;
        }
    }
}
