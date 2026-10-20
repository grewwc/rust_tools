package com.grewwc.aserve;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;

/** Android-independent validation and immutable notebook values. */
final class LocalNotebookModel {
    private LocalNotebookModel() {}

    static final class Request {
        final String origin;
        final String requestId;
        final String action;
        final String arguments;
        final String kind;
        final String title;
        final String text;
        final String entryId;
        final Boolean completed;
        final String query;
        final int max;
        final long offset;

        Request(String origin, String requestId, String action, Map<String, Object> payload) {
            this.origin = normalizeOrigin(origin);
            this.requestId = string(requestId, "requestId", 80, true);
            if (requestId.length() > 80) throw invalid("requestId is too long");
            if (payload == null) throw invalid("payload must be an object");
            this.action = action;
            switch (action == null ? "" : action) {
                case "note_create":
                    allowed(payload, "kind", "title", "text");
                    kind = payload.containsKey("kind") ? kind(payload.get("kind")) : "note";
                    title = payload.containsKey("title")
                            ? string(payload.get("title"), "title", 200, false) : "";
                    text = string(payload.get("text"), "text", 3000, true);
                    entryId = null;
                    completed = null;
                    query = "";
                    max = 10;
                    offset = 0;
                    break;
                case "note_list":
                    allowed(payload, "query", "kind", "max", "offset");
                    kind = payload.containsKey("kind") ? kind(payload.get("kind")) : null;
                    query = payload.containsKey("query")
                            ? string(payload.get("query"), "query", 200, false) : "";
                    max = payload.containsKey("max")
                            ? (int) integer(payload.get("max"), "max", 1, 20) : 10;
                    offset = payload.containsKey("offset")
                            ? integer(payload.get("offset"), "offset", 0, Long.MAX_VALUE) : 0;
                    title = null;
                    text = null;
                    entryId = null;
                    completed = null;
                    break;
                case "note_update":
                    allowed(payload, "entry_id", "title", "text", "completed");
                    entryId = uuid(payload.get("entry_id"));
                    title = payload.containsKey("title")
                            ? string(payload.get("title"), "title", 200, false) : null;
                    text = payload.containsKey("text")
                            ? string(payload.get("text"), "text", 3000, true) : null;
                    if (payload.containsKey("completed")) {
                        Object value = payload.get("completed");
                        if (!(value instanceof Boolean)) throw invalid("completed must be boolean");
                        completed = (Boolean) value;
                    } else {
                        completed = null;
                    }
                    if (title == null && text == null && completed == null) {
                        throw invalid("note_update needs title, text or completed");
                    }
                    kind = null;
                    query = "";
                    max = 10;
                    offset = 0;
                    break;
                default:
                    throw invalid("unknown notebook action");
            }
            // Include field presence and types, not JSONObject insertion order. Changing even
            // an explicit default is a different request; values are never trimmed or truncated.
            StringBuilder canonical = new StringBuilder();
            part(canonical, action);
            for (Map.Entry<String, Object> field : new TreeMap<>(payload).entrySet()) {
                part(canonical, field.getKey());
                Object value = field.getValue();
                canonical.append(value instanceof String ? 's' : value instanceof Boolean ? 'b' : 'i');
                part(canonical, value.toString());
            }
            arguments = canonical.toString();
        }

        boolean isMutation() {
            return !"note_list".equals(action);
        }

        void requireSameArguments(String storedArguments) {
            if (!arguments.equals(storedArguments)) {
                throw invalid("requestId already used with different arguments");
            }
        }
    }

    static final class Entry {
        final String entryId;
        final String kind;
        final String title;
        final String text;
        final boolean completed;

        Entry(String entryId, String kind, String title, String text, boolean completed) {
            this.entryId = uuid(entryId);
            this.kind = kind(kind);
            this.title = string(title, "title", 200, false);
            this.text = string(text, "text", 3000, true);
            if (completed && !"todo".equals(kind)) throw invalid("only todos can be completed");
            this.completed = completed;
        }

        static Entry create(Request request, String entryId) {
            if (!"note_create".equals(request.action)) throw invalid("expected note_create");
            return new Entry(entryId, request.kind, request.title, request.text, false);
        }

        Entry update(Request request) {
            if (!"note_update".equals(request.action) || !entryId.equals(request.entryId)) {
                throw invalid("update does not match entry");
            }
            if (request.completed != null && !"todo".equals(kind)) {
                throw invalid("completed is only allowed for todos");
            }
            return new Entry(entryId, kind, request.title == null ? title : request.title,
                    request.text == null ? text : request.text,
                    request.completed == null ? completed : request.completed);
        }

        String preview() {
            int end = text.offsetByCodePoints(0, Math.min(text.codePointCount(0, text.length()), 120));
            return text.substring(0, end);
        }
    }

    static String normalizeOrigin(String value) {
        String input = string(value, "origin", 2048, true);
        try {
            URI uri = new URI(input);
            if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost() == null
                    || uri.getRawUserInfo() != null || uri.getRawQuery() != null
                    || uri.getRawFragment() != null
                    || (uri.getRawPath() != null && !uri.getRawPath().isEmpty()
                            && !"/".equals(uri.getRawPath()))
                    || uri.getPort() == 0 || uri.getPort() > 65535) {
                throw invalid("origin must be an HTTPS server origin");
            }
            String host = uri.getHost().toLowerCase(Locale.ROOT);
            int port = uri.getPort();
            return "https://" + host + (port == -1 || port == 443 ? "" : ":" + port);
        } catch (URISyntaxException error) {
            throw invalid("origin must be an HTTPS server origin");
        }
    }

    private static void allowed(Map<String, Object> payload, String... keys) {
        Set<String> allowed = new HashSet<>(Arrays.asList(keys));
        for (String key : payload.keySet()) {
            if (!allowed.contains(key)) throw invalid("unexpected notebook payload field");
        }
    }

    private static String kind(Object value) {
        if (!"note".equals(value) && !"todo".equals(value)) throw invalid("kind must be note or todo");
        return (String) value;
    }

    private static String uuid(Object value) {
        String input = string(value, "entry_id", 36, true);
        try {
            String canonical = UUID.fromString(input).toString();
            if (!canonical.equalsIgnoreCase(input)) throw invalid("entry_id must be a UUID");
            return canonical;
        } catch (IllegalArgumentException error) {
            throw invalid("entry_id must be a UUID");
        }
    }

    private static String string(Object value, String key, int max, boolean nonempty) {
        if (!(value instanceof String)) throw invalid(key + " must be a string");
        String text = (String) value;
        if (text.codePointCount(0, text.length()) > max) throw invalid(key + " is too long");
        boolean content = false;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == 0 || Character.isLowSurrogate(c)) throw invalid(key + " has invalid characters");
            if (Character.isHighSurrogate(c)) {
                if (i + 1 == text.length() || !Character.isLowSurrogate(text.charAt(++i))) {
                    throw invalid(key + " has invalid characters");
                }
            }
            if (!Character.isWhitespace(c) && !Character.isSpaceChar(c)) content = true;
        }
        if (nonempty && !content) throw invalid(key + " must not be empty");
        return text;
    }

    private static long integer(Object value, String key, long min, long max) {
        if (!(value instanceof Byte || value instanceof Short
                || value instanceof Integer || value instanceof Long)) {
            throw invalid(key + " must be an integer");
        }
        long number = ((Number) value).longValue();
        if (number < min || number > max) throw invalid(key + " is out of range");
        return number;
    }

    private static void part(StringBuilder target, String value) {
        target.append(value.length()).append(':').append(value);
    }

    private static IllegalArgumentException invalid(String message) {
        return new IllegalArgumentException(message);
    }
}
