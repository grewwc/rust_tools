package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

public final class PhotoSemanticIndexTest {
    @Test
    public void unchangedModelVersionPreservesEmbeddings() {
        FakeVersionStore store = new FakeVersionStore("v1", 7);

        PhotoSemanticIndex.bindModelVersion(store, "v1");

        assertEquals("v1", store.version);
        assertEquals(7, store.photoCount);
        assertEquals(Arrays.asList("begin", "read:v1", "success", "end:commit"),
                store.operations);
    }

    @Test
    public void changedModelVersionInvalidatesBeforeBinding() {
        FakeVersionStore store = new FakeVersionStore("v1", 7);

        PhotoSemanticIndex.bindModelVersion(store, "v2");

        assertEquals("v2", store.version);
        assertEquals(0, store.photoCount);
        assertEquals(Arrays.asList("begin", "read:v1", "clear", "write:v2",
                "success", "end:commit"), store.operations);
    }

    @Test
    public void missingBindingInvalidatesUnversionedRows() {
        FakeVersionStore store = new FakeVersionStore(null, 4);

        PhotoSemanticIndex.bindModelVersion(store, "v1");

        assertEquals("v1", store.version);
        assertEquals(0, store.photoCount);
        assertTrue(store.operations.indexOf("clear") < store.operations.indexOf("write:v1"));
    }

    @Test
    public void failedBindingRollsBackInvalidation() {
        FakeVersionStore store = new FakeVersionStore("v1", 7);
        store.failAfterWrite = true;

        try {
            PhotoSemanticIndex.bindModelVersion(store, "v2");
            fail("binding failure should escape");
        } catch (IllegalStateException expected) {
            assertEquals("write failed", expected.getMessage());
        }

        assertEquals("v1", store.version);
        assertEquals(7, store.photoCount);
        assertEquals("end:rollback", store.operations.get(store.operations.size() - 1));
    }

    private static final class FakeVersionStore implements PhotoSemanticIndex.ModelVersionStore {
        final List<String> operations = new ArrayList<>();
        String version;
        int photoCount;
        boolean failAfterWrite;
        private String startingVersion;
        private int startingPhotoCount;
        private boolean successful;

        FakeVersionStore(String version, int photoCount) {
            this.version = version;
            this.photoCount = photoCount;
        }

        @Override
        public void beginTransaction() {
            operations.add("begin");
            startingVersion = version;
            startingPhotoCount = photoCount;
            successful = false;
        }

        @Override
        public String storedVersion() {
            operations.add("read:" + version);
            return version;
        }

        @Override
        public void clearPhotos() {
            operations.add("clear");
            photoCount = 0;
        }

        @Override
        public void writeVersion(String newVersion) {
            operations.add("write:" + newVersion);
            version = newVersion;
            if (failAfterWrite) throw new IllegalStateException("write failed");
        }

        @Override
        public void markSuccessful() {
            operations.add("success");
            successful = true;
        }

        @Override
        public void endTransaction() {
            operations.add(successful ? "end:commit" : "end:rollback");
            if (!successful) {
                version = startingVersion;
                photoCount = startingPhotoCount;
            }
        }
    }
}