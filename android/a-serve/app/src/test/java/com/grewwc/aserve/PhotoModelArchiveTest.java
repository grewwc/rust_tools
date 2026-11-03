package com.grewwc.aserve;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.concurrent.atomic.AtomicInteger;

public final class PhotoModelArchiveTest {
    @Test
    public void corruptCompletedDownloadIsDiscarded() throws Exception {
        File partial = Files.createTempFile("photo-model", ".partial").toFile();
        try {
            Files.write(partial.toPath(), new byte[] {1, 2, 3, 4});
            PhotoModelContract.FileSpec expected = new PhotoModelContract.FileSpec(
                    "clip_text.onnx", 4,
                    "0000000000000000000000000000000000000000000000000000000000000000");
            try {
                PhotoModelManager.verifyDownloadedFile(partial, expected);
                fail("corrupt download should fail verification");
            } catch (IOException expectedError) {
                assertTrue(!partial.exists() || partial.length() == 0);
            }
        } finally {
            partial.delete();
        }
    }

    @Test
    public void failedActivationRemovesInactiveModelDirectory() throws Exception {
        File installed = Files.createTempDirectory("photo-model-inactive").toFile();
        try {
            try {
                PhotoModelManager.activateOrCleanup(installed, () -> {
                    throw new IOException("activation failed");
                });
                fail("activation should fail");
            } catch (IOException expected) {
                assertFalse(installed.exists());
            }
        } finally {
            deleteTree(installed);
        }
    }

    @Test
    public void verifiedCopyChecksCancellationBetweenChunks() throws Exception {
        File root = Files.createTempDirectory("photo-model-cancel").toFile();
        try {
            File source = new File(root, "download.partial");
            Files.write(source.toPath(), new byte[3 * 64 * 1024]);
            File destination = new File(root, "staging/clip_vision.onnx");
            assertTrue(destination.getParentFile().mkdir());
            PhotoModelContract.FileSpec expected = new PhotoModelContract.FileSpec(
                    "clip_vision.onnx", source.length(),
                    "0000000000000000000000000000000000000000000000000000000000000000");
            AtomicInteger checks = new AtomicInteger();
            try {
                PhotoModelArchive.copyVerified(source, destination, expected, () -> {
                    if (checks.incrementAndGet() >= 3) throw new IOException("cancelled");
                });
                fail("copy should stop after cancellation");
            } catch (IOException expectedError) {
                assertTrue(expectedError.getMessage().contains("cancelled"));
                assertTrue(destination.length() < source.length());
            }
        } finally {
            deleteTree(root);
        }
    }

    private static void deleteTree(File file) {
        if (file == null || !file.exists()) return;
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) deleteTree(child);
        }
        file.delete();
    }
}
