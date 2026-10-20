package com.grewwc.aserve;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;

/** Bounded UTF-8 documents; offsets count Unicode code points, not UTF-16 units. */
final class TextDocument {
    static final int MAX_BYTES = 1024 * 1024;
    static final int PAGE_CHARS = 3000;

    static String read(InputStream stream) throws IOException {
        if (stream == null) throw new IOException("document could not be opened");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int count;
        while ((count = stream.read(buffer)) != -1) {
            if (out.size() + count > MAX_BYTES) throw new IOException("text document exceeds 1 MiB");
            out.write(buffer, 0, count);
        }
        String text = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(out.toByteArray())).toString();
        if (text.startsWith("\uFEFF")) text = text.substring(1);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c < 32 && c != '\n' && c != '\r' && c != '\t')
                throw new IOException("only UTF-8 text documents are supported (not PDF/Office/binary)");
        }
        return text;
    }

    static String page(String text, int offset, int limit) {
        int length = text.codePointCount(0, text.length());
        if (offset < 0 || offset > length || limit < 1 || limit > PAGE_CHARS)
            throw new IllegalArgumentException("invalid text page range");
        int from = text.offsetByCodePoints(0, offset);
        int to = text.offsetByCodePoints(from, Math.min(limit, length - offset));
        return text.substring(from, to);
    }

    static boolean validFilename(String name) {
        if (name == null || name.trim().isEmpty() || name.length() > 120
                || name.equals(".") || name.equals("..")) return false;
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (c == '/' || c == '\\' || Character.isISOControl(c)) return false;
        }
        return true;
    }
}
