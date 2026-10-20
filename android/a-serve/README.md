# a serve Android shell

A small Android WebView shell for the existing `a serve` mobile UI. It loads
the server's `/app` page directly, so UI and server updates do not require a
new APK.

## Build

The project needs JDK 11 and Android SDK 33:

```bash
./gradlew :app:assembleDebug \
  -PserveUrl=https://10.37.110.250:8080
```

Point Gradle at the SDK with a `local.properties` next to this README (it is
gitignored, because the path is machine-specific):

```properties
sdk.dir=/home/<user>/android-sdk
```

`ANDROID_HOME` works too, but only for that shell; without either one Gradle
fails with `SDK location not found`.

The installable APK is written to `app/build/outputs/apk/debug/app-debug.apk`.

## Device setup

The APK bundles the public `a serve` CA used by its default server, so no
system-wide CA installation is required for that server. It also trusts the
Android system and user certificate stores for a manually configured server.
The app never bypasses TLS errors. Rebuild the APK after intentionally rotating
the default server's CA.

The default server is configured at build time. If it later changes, long
press the app icon and choose **服务器地址**. A failed connection also opens
the same settings dialog.

The activity uses `singleTask`, so tapping the launcher icon reuses the
existing app window instead of creating another browser tab.