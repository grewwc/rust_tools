package com.grewwc.aserve;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Locale;

final class ServerUrl {
    private ServerUrl() {}

    static String normalize(String value) {
        if (value == null) return null;
        String input = value.trim();
        if (input.isEmpty()) return null;

        try {
            URI uri = new URI(input);
            if (!"https".equalsIgnoreCase(uri.getScheme())
                    || uri.getHost() == null
                    || uri.getRawUserInfo() != null
                    || uri.getRawQuery() != null
                    || uri.getRawFragment() != null) {
                return null;
            }
            int port = uri.getPort();
            if (port == 0 || port > 65535) return null;
            String path = uri.getPath();
            if (!(path == null || path.isEmpty() || "/".equals(path)
                    || "/app".equals(path) || "/app/".equals(path))) {
                return null;
            }

            String host = uri.getHost().toLowerCase(Locale.ROOT);
            if (host.indexOf(':') >= 0) host = "[" + host + "]";
            return "https://" + host + (port < 0 ? "" : ":" + port);
        } catch (URISyntaxException ignored) {
            return null;
        }
    }

    static String appUrl(String origin) {
        return origin + "/app";
    }

    static String freshAppUrl(String origin, long launchNonce) {
        return appUrl(origin) + "?shell_reload=" + Long.toUnsignedString(launchNonce);
    }

    static boolean isSameOrigin(String origin, String candidate) {
        String candidateOrigin = normalizeOriginOnly(candidate);
        return origin != null && origin.equals(candidateOrigin);
    }

    private static String normalizeOriginOnly(String value) {
        try {
            URI uri = new URI(value);
            if (!"https".equalsIgnoreCase(uri.getScheme())
                    || uri.getHost() == null
                    || uri.getRawUserInfo() != null) {
                return null;
            }
            int port = uri.getPort();
            if (port == 0 || port > 65535) return null;
            String host = uri.getHost().toLowerCase(Locale.ROOT);
            if (host.indexOf(':') >= 0) host = "[" + host + "]";
            return "https://" + host + (port < 0 ? "" : ":" + port);
        } catch (URISyntaxException ignored) {
            return null;
        }
    }
}