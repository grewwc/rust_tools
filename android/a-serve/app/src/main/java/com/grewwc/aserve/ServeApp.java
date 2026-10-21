package com.grewwc.aserve;

import android.app.Application;

/**
 * Installs the crash recorder before any activity exists.
 *
 * A failure inside {@code MainActivity.onCreate} still reaches the default
 * uncaught handler, but a failure while the process itself is being created
 * would not be recorded from an activity. Registering here covers both, and
 * the body stays trivial so it cannot become the crash it is meant to catch.
 */
public final class ServeApp extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        CrashGuard.install(this);
    }
}
