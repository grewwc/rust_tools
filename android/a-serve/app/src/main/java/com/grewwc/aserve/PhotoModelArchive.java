package com.grewwc.aserve;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** Bounded archive verification and extraction with no Android dependencies. */
final class PhotoModelArchive {
    private PhotoModelArchive() {}

    static void verifyArchive(File archive, PhotoModelContract.Manifest manifest) throws IOException {
        if (!archive.isFile() || archive.length() != manifest.archiveSize) {
            throw new IOException("photo model archive size mismatch");
        }
        if (!sha256(archive).equals(manifest.archiveSha256)) {
            throw new IOException("photo model archive checksum mismatch");
        }
    }

    static void extract(File archive, File destination, PhotoModelContract.Manifest manifest,
                        Cancellation cancellation) throws IOException {
        if (!destination.isDirectory() && !destination.mkdirs()) {
            throw new IOException("cannot create photo model staging directory");
        }
        Set<String> seen = new HashSet<>();
        long total = 0;
        try (ZipInputStream zip = new ZipInputStream(new BufferedInputStream(new FileInputStream(archive)))) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                cancellation.check();
                String name = entry.getName();
                if (entry.isDirectory() || name == null || name.indexOf('/') >= 0
                        || name.indexOf('\\') >= 0 || !PhotoModelContract.FILE_NAMES.contains(name)) {
                    throw new IOException("unexpected photo model archive entry");
                }
                if (!seen.add(name)) throw new IOException("duplicate photo model archive entry " + name);
                PhotoModelContract.FileSpec expected = manifest.file(name);
                if (entry.getSize() > expected.size) throw new IOException("oversized photo model entry " + name);
                File output = new File(destination, name);
                MessageDigest digest = digest();
                long written = 0;
                try (FileOutputStream file = new FileOutputStream(output);
                     BufferedOutputStream stream = new BufferedOutputStream(file)) {
                    byte[] buffer = new byte[64 * 1024];
                    int read;
                    while ((read = zip.read(buffer)) != -1) {
                        cancellation.check();
                        written += read;
                        total += read;
                        if (written > expected.size || total > manifest.totalFileSize) {
                            throw new IOException("photo model archive exceeds declared size");
                        }
                        stream.write(buffer, 0, read);
                        digest.update(buffer, 0, read);
                    }
                    stream.flush();
                    file.getFD().sync();
                }
                if (written != expected.size || !hex(digest.digest()).equals(expected.sha256)) {
                    throw new IOException("photo model entry verification failed for " + name);
                }
                zip.closeEntry();
            }
        }
        if (seen.size() != PhotoModelContract.FILE_NAMES.size()
                || !seen.containsAll(PhotoModelContract.FILE_NAMES)
                || total != manifest.totalFileSize) {
            throw new IOException("photo model archive file set is incomplete");
        }
    }

    interface Cancellation {
        void check() throws IOException;
    }

    static void verifyDirectory(File directory, PhotoModelContract.Manifest manifest) throws IOException {
        String[] entries = directory.list();
        if (entries == null) throw new IOException("photo model directory is unreadable");
        Set<String> allowed = new HashSet<>(PhotoModelContract.FILE_NAMES);
        allowed.add("manifest.json");
        for (String name : entries) {
            if (!allowed.contains(name)) throw new IOException("unexpected installed model file " + name);
        }
        for (PhotoModelContract.FileSpec expected : manifest.files) {
            File file = new File(directory, expected.name);
            if (!file.isFile() || file.length() != expected.size
                    || !sha256(file).equals(expected.sha256)) {
                throw new IOException("installed photo model verification failed for " + expected.name);
            }
        }
    }

    static String sha256(File file) throws IOException {
        MessageDigest digest = digest();
        try (InputStream input = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) digest.update(buffer, 0, read);
        }
        return hex(digest.digest());
    }

    private static MessageDigest digest() throws IOException {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new IOException("SHA-256 is unavailable", error);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder output = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) output.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        return output.toString();
    }
}
