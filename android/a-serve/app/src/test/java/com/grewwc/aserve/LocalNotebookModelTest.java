package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.Map;

public final class LocalNotebookModelTest {
    private static final String ORIGIN = "https://example.test:8080";
    private static final String ID = "1b3d890b-a2f4-431c-8557-f8f24e214ca5";

    private static Map<String, Object> fields(Object... values) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < values.length; i += 2) result.put((String) values[i], values[i + 1]);
        return result;
    }

    private static LocalNotebookModel.Request request(String action, Object... values) {
        return new LocalNotebookModel.Request(ORIGIN, "stable-id", action, fields(values));
    }

    private static void rejects(String action, Object... values) {
        assertThrows(IllegalArgumentException.class, () -> request(action, values));
    }

    @Test
    public void createPreservesContentAndUsesSafeDefaults() {
        LocalNotebookModel.Request request = request("note_create", "text", "  body\n");
        assertEquals("note", request.kind);
        assertEquals("", request.title);
        assertEquals("  body\n", request.text);
        assertTrue(request.isMutation());
        LocalNotebookModel.Entry entry = LocalNotebookModel.Entry.create(request, ID);
        assertEquals(ID, entry.entryId);
        assertEquals(request.text, entry.text);
        assertFalse(entry.completed);
    }

    @Test
    public void createAcceptsExactLimitsAndRejectsOversizedValues() {
        LocalNotebookModel.Request request = request("note_create", "kind", "todo",
                "title", "t".repeat(200), "text", "x".repeat(3000));
        assertEquals(3000, request.text.length());
        assertEquals(200, request.title.length());
        assertEquals("todo", request.kind);
        rejects("note_create", "title", "t".repeat(201), "text", "body");
        rejects("note_create", "text", "x".repeat(3001));
    }

    @Test
    public void unicodeContentLimitsMatchCodepointBasedUpstreamValidation() {
        String emoji = "\ud83d\ude00";
        LocalNotebookModel.Request request = request("note_create", "title", emoji.repeat(200),
                "text", emoji.repeat(3000));
        assertEquals(6000, request.text.length());
        assertEquals(3000, request.text.codePointCount(0, request.text.length()));
        assertEquals(400, request.title.length());
        assertEquals(400, request("note_list", "query", emoji.repeat(200)).query.length());
        rejects("note_create", "text", emoji.repeat(3001));
        rejects("note_create", "title", emoji.repeat(201), "text", "body");
        rejects("note_list", "query", emoji.repeat(201));
        String preview = LocalNotebookModel.Entry.create(request, ID).preview();
        assertEquals(emoji.repeat(120), preview);
        assertEquals(120, preview.codePointCount(0, preview.length()));
    }

    @Test
    public void rejectsMissingBlankMistypedAndUnknownCreateFields() {
        rejects("note_create");
        for (Object invalid : new Object[]{null, "", " \t\n\u00a0\u2003", 123, true}) {
            rejects("note_create", "text", invalid);
        }
        rejects("note_create", "text", "body", "title", null);
        rejects("note_create", "text", "body", "title", 2);
        rejects("note_create", "text", "body", "kind", "TODO");
        rejects("note_create", "text", "body", "kind", null);
        rejects("note_create", "text", "body", "completed", false);
        rejects("note_create", "text", "body", "auto", true);
        rejects("note_delete", "entry_id", ID);
        rejects(null);
    }

    @Test
    public void rejectsStringsThatCannotRoundTripSafely() {
        rejects("note_create", "text", "a\u0000b");
        rejects("note_create", "text", "a\ud800");
        rejects("note_create", "text", "a\udc00");
        rejects("note_create", "text", "\ud800x");
        assertEquals("\ud83d\ude00", request("note_create", "text", "\ud83d\ude00").text);
    }

    @Test
    public void listDefaultsAndBoundariesAreExplicit() {
        LocalNotebookModel.Request defaults = request("note_list");
        assertNull(defaults.kind);
        assertEquals("", defaults.query);
        assertEquals(10, defaults.max);
        assertEquals(0, defaults.offset);
        assertFalse(defaults.isMutation());
        LocalNotebookModel.Request bounded = request("note_list", "query", "q".repeat(200),
                "kind", "todo", "max", 20, "offset", Long.MAX_VALUE);
        assertEquals(20, bounded.max);
        assertEquals(Long.MAX_VALUE, bounded.offset);
        assertEquals("todo", bounded.kind);
        assertEquals(1, request("note_list", "max", 1L).max);
        assertEquals("100%_", request("note_list", "query", "100%_").query);
    }

    @Test
    public void listRejectsCoercionOverflowAndUnknownFields() {
        for (Object invalid : new Object[]{0, -1, 21, Long.MAX_VALUE, 1.5, 1.0, "10", true, null}) {
            rejects("note_list", "max", invalid);
        }
        for (Object invalid : new Object[]{-1, -1L, 0.1, Double.POSITIVE_INFINITY, "0", null}) {
            rejects("note_list", "offset", invalid);
        }
        rejects("note_list", "query", "q".repeat(201));
        rejects("note_list", "query", null);
        rejects("note_list", "kind", "all");
        rejects("note_list", "completed", true);
    }

    @Test
    public void updateRequiresCanonicalUuidAndAtLeastOneValidChange() {
        rejects("note_update", "entry_id", ID);
        rejects("note_update", "text", "new");
        rejects("note_update", "entry_id", "1-1-1-1-1", "text", "new");
        rejects("note_update", "entry_id", "not-a-uuid", "title", "");
        rejects("note_update", "entry_id", ID, "text", " ");
        rejects("note_update", "entry_id", ID, "text", "x".repeat(3001));
        rejects("note_update", "entry_id", ID, "title", "x".repeat(201));
        rejects("note_update", "entry_id", ID, "kind", "todo");
        rejects("note_update", "entry_id", ID, "title", null);
        for (Object invalid : new Object[]{"false", 0, 1, null}) {
            rejects("note_update", "entry_id", ID, "completed", invalid);
        }
        assertEquals(ID, request("note_update", "entry_id", ID.toUpperCase(java.util.Locale.ROOT),
                "title", "").entryId);
    }

    @Test
    public void updatesPreserveUnchangedFieldsAndCanReopenTodo() {
        LocalNotebookModel.Entry entry = new LocalNotebookModel.Entry(ID, "todo", "title", "body", false);
        LocalNotebookModel.Entry done = entry.update(request("note_update", "entry_id", ID, "completed", true));
        assertTrue(done.completed);
        assertFalse(entry.completed);
        assertEquals("title", done.title);
        assertEquals("body", done.text);
        LocalNotebookModel.Entry reopened = done.update(request("note_update", "entry_id", ID,
                "completed", false, "title", "", "text", "new body"));
        assertFalse(reopened.completed);
        assertEquals("", reopened.title);
        assertEquals("new body", reopened.text);
        assertEquals("todo", reopened.kind);
        assertEquals(ID, reopened.entryId);
    }

    @Test
    public void completionIsRejectedForNotesEvenWhenFalse() {
        LocalNotebookModel.Entry entry = new LocalNotebookModel.Entry(ID, "note", "title", "body", false);
        for (boolean completed : new boolean[]{false, true}) {
            assertThrows(IllegalArgumentException.class, () -> entry.update(request("note_update",
                    "entry_id", ID, "completed", completed)));
        }
        assertThrows(IllegalArgumentException.class,
                () -> new LocalNotebookModel.Entry(ID, "note", "", "body", true));
        assertEquals("new", entry.update(request("note_update", "entry_id", ID, "text", "new")).text);
    }

    @Test
    public void previewsAreBoundedWithoutSplittingSurrogatePairs() {
        assertEquals("short", new LocalNotebookModel.Entry(ID, "note", "", "short", false).preview());
        assertEquals(120, new LocalNotebookModel.Entry(ID, "todo", "", "x".repeat(3000), false).preview().length());
        String split = "x".repeat(119) + "\ud83d\ude00" + "after";
        assertEquals("x".repeat(119) + "\ud83d\ude00",
                new LocalNotebookModel.Entry(ID, "note", "", split, false).preview());
        String exact = "x".repeat(118) + "\ud83d\ude00";
        assertEquals(exact, new LocalNotebookModel.Entry(ID, "note", "", exact, false).preview());
    }

    @Test
    public void replayIdentityIgnoresObjectKeyOrderButNotChangedArguments() {
        LocalNotebookModel.Request first = request("note_create", "title", "hello", "text", "body");
        LocalNotebookModel.Request reordered = request("note_create", "text", "body", "title", "hello");
        reordered.requireSameArguments(first.arguments);
        LocalNotebookModel.Request changed = request("note_create", "text", "different", "title", "hello");
        assertThrows(IllegalArgumentException.class, () -> changed.requireSameArguments(first.arguments));
        assertNotEquals(request("note_create", "text", "body").arguments,
                request("note_create", "text", "body", "kind", "note").arguments);
        assertNotEquals(request("note_create", "title", "a:b", "text", "c").arguments,
                request("note_create", "title", "a", "text", "b:c").arguments);
        assertNotEquals(request("note_update", "entry_id", ID, "completed", false).arguments,
                request("note_update", "entry_id", ID, "completed", true).arguments);
        assertNotEquals(request("note_create", "text", "body").arguments,
                request("note_update", "entry_id", ID, "text", "body").arguments);
    }

    @Test
    public void originNormalizationDoesNotMixDifferentServers() {
        assertEquals("https://example.test", LocalNotebookModel.normalizeOrigin("HTTPS://EXAMPLE.test:443/"));
        assertEquals("https://[::1]:8080", LocalNotebookModel.normalizeOrigin("https://[::1]:8080"));
        assertNotEquals(LocalNotebookModel.normalizeOrigin(ORIGIN),
                LocalNotebookModel.normalizeOrigin("https://example.test"));
        for (String invalid : new String[]{null, "", "http://example.test", "https://user@example.test",
                "https://example.test/path", "https://example.test?x", "https://example.test#x",
                "https://example.test:0", "https://example.test:65536", " https://example.test "}) {
            assertThrows(IllegalArgumentException.class, () -> LocalNotebookModel.normalizeOrigin(invalid));
        }
    }

    @Test
    public void requestIdAndPayloadAreRequiredAndBounded() {
        for (String invalid : new String[]{null, "", "   ", "x".repeat(81), "id\u0000"}) {
            assertThrows(IllegalArgumentException.class,
                    () -> new LocalNotebookModel.Request(ORIGIN, invalid, "note_list", fields()));
        }
        assertEquals("x".repeat(80),
                new LocalNotebookModel.Request(ORIGIN, "x".repeat(80), "note_list", fields()).requestId);
        assertThrows(IllegalArgumentException.class,
                () -> new LocalNotebookModel.Request(ORIGIN, "id", "note_list", null));
    }
}
