package com.grewwc.aserve;

import android.content.Context;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;

/**
 * Keeps the last fatal throwable where the next launch can show it.
 *
 * A crash during startup leaves nothing on screen and the phone is not
 * attached to a debugger, so the only durable evidence is a file written
 * before the process dies. {@link #install} therefore has to run before the
 * WebView is created: that is the earliest point where a failure in the
 * startup path can still be recorded.
 */
public final class CrashGuard {
    private static final String TAG = "AServeCrash";
    private static final String FILE_NAME = "a-serve-crash.txt";
    /** Bounds the report so it can still be pasted into the composer. */
    static final int MAX_CHARS = 60000;

    private CrashGuard() {}

    /**
     * Turn a throwable into the text shown on screen and sent to the agent.
     * Kept free of Android APIs so the formatting can be unit tested.
     */
    static String describe(String version, Throwable error) {
        StringWriter writer = new StringWriter();
        writer.write("a serve APK " + version + "\n");
        if (error == null) {
            writer.write("(no throwable recorded)");
        } else {
            error.printStackTrace(new PrintWriter(writer));
        }
        String text = writer.toString();
        if (text.length() <= MAX_CHARS) return text;
        return text.substring(0, MAX_CHARS) + "\n...(truncated)";
    }

    /** Record every uncaught throwable, then let the system end the process. */
    static void install(Context context) {
        Context app = context.getApplicationContext() == null
                ? context : context.getApplicationContext();
        Thread.UncaughtExceptionHandler previous = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler((thread, error) -> {
            record(app, error);
            if (previous != null) previous.uncaughtException(thread, error);
        });
    }

    /** Persist one report. Best effort: a diagnostic must never crash by itself. */
    static void record(Context context, Throwable error) {
        try {
            String text = describe(BuildConfig.VERSION_NAME, error);
            Log.e(TAG, text);
            File file = new File(context.getFilesDir(), FILE_NAME);
            try (FileOutputStream out = new FileOutputStream(file)) {
                out.write(text.getBytes(StandardCharsets.UTF_8));
            }
        } catch (Throwable ignored) {
            // Nothing useful can be reported if even the report cannot be written.
        }
    }

    /** Return the report left by the previous run, or null when it started clean. */
    static String read(Context context) {
        try {
            File file = new File(context.getFilesDir(), FILE_NAME);
            if (!file.isFile() || file.length() == 0) return null;
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            try (FileInputStream in = new FileInputStream(file)) {
                int read;
                while ((read = in.read(chunk)) != -1) buffer.write(chunk, 0, read);
            }
            String text = new String(buffer.toByteArray(), StandardCharsets.UTF_8).trim();
            return text.isEmpty() ? null : text;
        } catch (Throwable ignored) {
            return null;
        }
    }

    static void clear(Context context) {
        try {
            File file = new File(context.getFilesDir(), FILE_NAME);
            if (file.isFile()) file.delete();
        } catch (Throwable ignored) {
            // A stale report is harmless; the user can dismiss it.
        }
    }
}
