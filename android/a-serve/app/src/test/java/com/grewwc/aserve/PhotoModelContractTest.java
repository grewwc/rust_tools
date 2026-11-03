package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

import java.io.File;
import java.net.URL;
import java.nio.file.Files;

public final class PhotoModelContractTest {
    @Test
    public void oversizedPartialFileIsResetBeforeRetry() throws Exception {
        File partial = Files.createTempFile("photo-model", ".partial").toFile();
        try {
            Files.write(partial.toPath(), new byte[9]);
            assertEquals(0, PhotoModelManager.preparePartialForResume(partial, 8));
            assertEquals(0, partial.length());
        } finally {
            partial.delete();
        }
    }

    @Test
    public void currentMetadataPinsEveryRuntimeDownload() throws Exception {
        PhotoModelContract.Manifest manifest = PhotoModelContract.current();
        assertEquals("1", manifest.version);
        assertEquals(PhotoModelContract.FILE_NAMES.size(), manifest.files.size());

        long total = 0;
        for (PhotoModelContract.FileSpec file : manifest.files) {
            assertNotNull(file.url);
            assertTrue(file.size > 0);
            assertTrue(file.size <= file.maxBytes);
            assertTrue(file.sha256.matches("[0-9a-f]{64}"));
            PhotoModelManager.validateDownloadUrl(new URL(file.url));
            total += file.size;
        }
        assertEquals(manifest.totalFileSize, total);
    }

    @Test
    public void sameVersionWithDifferentIdentityRequiresANewVersion() {
        PhotoModelContract.Manifest manifest = PhotoModelContract.current();
        assertEquals(PhotoModelContract.Resolution.VERSION_CONFLICT,
                PhotoModelContract.resolve(manifest.version,
                        "0000000000000000000000000000000000000000000000000000000000000000",
                        manifest));
    }

    @Test
    public void resumedResponseMustCoverTheExpectedRemainder() {
        assertEquals(1, PhotoModelContract.resumeResponseMode(
                10, 206, "bytes 10-99/100", 100));
        try {
            PhotoModelContract.resumeResponseMode(10, 206, "bytes 10-98/100", 100);
            fail("an incomplete range must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("incomplete"));
        }
    }

    @Test
    public void runtimeDownloadRejectsLookalikeAndInsecureHosts() throws Exception {
        assertRejected("https://huggingface.co.example.test/model.onnx");
        assertRejected("http://huggingface.co/model.onnx");
        assertRejected("https://user@huggingface.co/model.onnx");
    }

    private static void assertRejected(String value) throws Exception {
        try {
            PhotoModelManager.validateDownloadUrl(new URL(value));
            fail("untrusted URL must be rejected: " + value);
        } catch (java.io.IOException expected) {
            assertTrue(expected.getMessage().contains("untrusted"));
        }
    }
}
