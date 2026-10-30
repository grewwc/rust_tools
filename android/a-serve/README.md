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

Publishing it through `a serve` requires this order: bump `versionName` and
`ANDROID_APK_VERSION` (in `src/bin/ai/serve/info.rs`) together, build the APK,
copy it to `dist/a-serve.apk`, then rebuild and restart `a`. The daemon embeds
the APK at compile time, so building `a` before the copy keeps serving the
previous package under the old name. Confirm with `curl -sk <bind>/a-serve.apk
| sha256sum` against `sha256sum dist/a-serve.apk`.

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

## Chat-driven phone tools (1.10.0)

Ask in the normal composer; no extra action buttons or skill selection are
needed. Android-only `phone_action` supports timers, ISO-weekday repeating
alarms, and the existing calendar/SMS/share actions. Clock requests report
dispatch, not verified creation: the selected clock app controls the result.

Notes and todos are stored in the app's private SQLite database. Create,
search, and update entries by their returned IDs; completing a todo does not
schedule a reminder. Notes are not synchronized to another notes app. Database
writes and replay records commit together. Uninstalling the app removes them.

File import uses Android's document picker (no broad storage permission).
This version reads UTF-8 text, Markdown, CSV, JSON, and XML up to 1 MiB, in
3000-character pages. PDF/Office/image extraction is not implemented. Imported
file handles expire when the app process exits; reimport to read again. File
save uses the system create-document picker and accepts up to 3000 characters.

`phone_search` (1.16.0) opens a visible in-app search page for one query and
returns up to ten extracted result entries marked untrusted. Google, Bing,
Baidu, and DuckDuckGo are allowlisted; the URL template is fixed on the device.
The page loads in a dedicated WebView without the app's JavaScript bridge, and
result links open in the system browser, so the app never drives or reads
another app's browser. When the engine shows a consent or CAPTCHA page, the
window stays interactive: the user solves it there and extraction resumes
automatically. It uses the phone's network but not the user's Chrome cookies.

Location requests fine and coarse permission together and prefers a fresh GPS
fix. An existing coarse-only installation gets one upgrade prompt. If only
approximate access is granted, the result explicitly says so and includes
measured accuracy, provider, and age; it never promises GPS-level accuracy.
Later changes can be made in Android's app location-permission settings.

`screen_capture` starts a visible, revocable MediaProjection sharing session.
Android consent is required once per live session, not once forever. After
consent, a request is only armed: a floating capture button appears on the
phone screen, so you switch to the target app and tap the button, and the
delivered frame is the screen exactly as it looked at that tap. The button
floats above the content and is hidden while the frame is taken, so the
picture is the page you were looking at rather than the one behind a
notification shade. The button can be dragged; its position is remembered.

Drawing it needs the "display over other apps" permission, which the user
grants once in the system settings. The chat page has a **悬浮截屏球** entry
in its connection panel that opens that page. Without the permission the
notification action **立即截屏** stays the trigger: the shade covers the
content, so the app keeps a short in-memory history of mirrored frames and
delivers the one visible just before the tap, and falls back to taking the
frame once the shade closed.

An armed request expires after about two minutes without a tap. Sharing is
one-shot: the session ends automatically when the request finishes, including
after a delivered frame or an expired or failed request, so no sharing
indicator remains to be stopped manually. The conversation shows the delivered
frame, so you can confirm which moment was sent. Idle frames are discarded, not
uploaded. While a request is waiting, the foreground notification offers
**立即截屏** and **停止共享**. Use `screen_stop`, the notification, or Android's
sharing control to revoke it early. The session also ends on page replacement,
app destruction, system revocation, or after 30 minutes; no consent token
survives restart. Protected screens are not bypassed. Screen images are resized
to at most 2048 pixels per side and encoded as JPEG (at most 256 KiB); the tool
saves one server-side temporary image for model inspection. Screen contents can
therefore reach the configured model provider, just like manually uploaded
images.

Unit tests do not emulate vendor clock apps, Android permission dialogs,
document providers, SQLite device storage, or MediaProjection. Test these on a
device before treating native dispatch as end-to-end success.