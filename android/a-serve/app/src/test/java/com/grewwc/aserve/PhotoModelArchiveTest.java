package com.grewwc.aserve;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public final class PhotoModelArchiveTest {
    @Test
    public void corruptCompletedDownloadIsDiscarded() throws Exception {
        File root = Files.createTempDirectory("photo-model-corrupt").toFile();
        try {
            File partial = new File(root, "model.partial");
            Files.write(partial.toPath(), new byte[]{1, 2, 3, 4});
            PhotoModelContract.Manifest manifest = new PhotoModelContract.Manifest(
                    "1", partial.length(), repeat('0', 64), Collections.emptyList(), 0);

            try {
                PhotoModelManager.verifyDownloadedArchive(partial, manifest);
                fail("corrupt archive must be rejected");
            } catch (IOException expected) {
                assertTrue(!partial.exists() || partial.length() == 0);
            }
        } finally {
            deleteTree(root);
        }
    }

    @Test
    public void failedActivationRemovesInactiveModelDirectory() throws Exception {
        File installed = Files.createTempDirectory("photo-model-inactive").toFile();
        Files.write(new File(installed, "clip_vocab.txt").toPath(), new byte[]{1});

        try {
            PhotoModelManager.activateOrCleanup(installed, () -> {
                throw new IOException("activation failed");
            });
            fail("activation failure must propagate");
        } catch (IOException expected) {
            assertFalse(installed.exists());
        } finally {
            deleteTree(installed);
        }
    }

    @Test
    public void extractionChecksCancellationBetweenChunks() throws Exception {
        File root = Files.createTempDirectory("photo-model-cancel").toFile();
        try {
            byte[] content = new byte[2 * 1024 * 1024];
            File archive = new File(root, "model.zip");
            try (ZipOutputStream zip = new ZipOutputStream(new FileOutputStream(archive))) {
                zip.putNextEntry(new ZipEntry("clip_vocab.txt"));
                zip.write(content);
                zip.closeEntry();
            }
            PhotoModelContract.FileSpec file = new PhotoModelContract.FileSpec(
                    "clip_vocab.txt", content.length, repeat('0', 64));
            PhotoModelContract.Manifest manifest = new PhotoModelContract.Manifest(
                    "1", archive.length(), repeat('0', 64), Collections.singletonList(file),
                    content.length);
            File destination = new File(root, "staging");
            AtomicInteger checks = new AtomicInteger();

            try {
                PhotoModelArchive.extract(archive, destination, manifest, () -> {
                    if (checks.incrementAndGet() >= 3) throw new IOException("cancelled");
                });
                fail("cancelled extraction must stop");
            } catch (IOException expected) {
                assertTrue(checks.get() >= 3);
                File extracted = new File(destination, "clip_vocab.txt");
                assertTrue(extracted.length() < content.length);
            }
        } finally {
            deleteTree(root);
        }
    }

    private static String repeat(char value, int count) {
        StringBuilder output = new StringBuilder(count);
        for (int index = 0; index < count; index++) output.append(value);
        return output.toString();
    }

    private static void deleteTree(File file) {
        if (file == null || !file.exists()) return;
        File[] children = file.listFiles();
        if (children != null) for (File child : children) deleteTree(child);
        file.delete();
    }
}
