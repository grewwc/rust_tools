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

/** File and installed-directory verification with no Android dependencies. */
final class PhotoModelArchive {
    private PhotoModelArchive() {}

    static void verifyFile(File file, PhotoModelContract.FileSpec expected) throws IOException {
        if (!file.isFile() || file.length() != expected.size) {
            throw new IOException("photo model size mismatch for " + expected.name);
        }
        if (!sha256(file).equals(expected.sha256)) {
            throw new IOException("photo model checksum mismatch for " + expected.name);
        }
    }

    static void copyVerified(File source, File destination, PhotoModelContract.FileSpec expected,
                             Cancellation cancellation) throws IOException {
        if (!source.isFile() || source.length() != expected.size) {
            throw new IOException("downloaded photo model size mismatch for " + expected.name);
        }
        MessageDigest digest = digest();
        long written = 0;
        try (InputStream input = new BufferedInputStream(new FileInputStream(source));
             FileOutputStream file = new FileOutputStream(destination);
             BufferedOutputStream output = new BufferedOutputStream(file)) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                cancellation.check();
                written += read;
                if (written > expected.size || written > expected.maxBytes) {
                    throw new IOException("photo model exceeds declared size for " + expected.name);
                }
                output.write(buffer, 0, read);
                digest.update(buffer, 0, read);
            }
            output.flush();
            file.getFD().sync();
        }
        if (written != expected.size || !hex(digest.digest()).equals(expected.sha256)) {
            throw new IOException("downloaded photo model verification failed for " + expected.name);
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
            verifyFile(new File(directory, expected.name), expected);
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
