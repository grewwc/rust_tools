# Serve Guide

## Scope

Applies to `src/bin/ai/serve/**`. Process supervision (`--serve-start`,
`--serve-stop`, `--serve-restart`, `--serve-status`) lives in `ctl.rs`; the
HTTP/SSE surface lives in `mod.rs`.

## Key invariants

1. **The daemon runs a frozen copy.** `start_daemon` re-execs
   `<state-dir>/run/a-<content-hash>` produced by `freeze_daemon_exe` (`ctl.rs`),
   never the install path. Installing a new `bin/a` does not change a running
   daemon — only a restart re-freezes.
2. **Deferred restart.** Never stop the daemon while it carries the current
   turn's SSE reply: `restart_daemon` and `schedule_restart_after_turn` (`ctl.rs`)
   defer in-turn requests until the owning turn exits, bound the wait with
   `DEFERRED_RESTART_WAIT`, and skip abandoned or superseded requests. Turn-free
   requests use the immediate path. A successful in-turn command confirms
   scheduling only; verify the restart in a later turn.
3. **Model turns are child processes.** The `/dirs add|ls|del` family (with
   `/add-dir <DIR>` as an add alias) is intercepted in the daemon;
   session grants live in `SessionServeConfig` and cwd/grants are read under the
   session lock before forwarding roots to a child. `run_one_shot_turn` (`mod.rs`)
   spawns model turns, so under the release build (`panic = "abort"` in
   `Cargo.toml`) a panicking tool aborts the turn child and every tool result in
   it, while the daemon survives; in unwind builds `run_tool_guarded`
   (`tools/registry/common.rs`) converts the panic into an ordinary tool error
   instead. Tool implementations must turn panics into ordinary tool errors.
4. **Health follows the TLS setting.** `serve_url` (`ctl.rs`) prefixes `https://`
   when a certificate is configured and `http://` otherwise, and `--serve-status`
   prints that URL as `bind=`. Probe `<bind>/healthz` with the same scheme;
   plain HTTP against a TLS port fails.
5. **The APK download is a compile-time embed.** `serve_android_apk` in
   `info.rs` serves `include_bytes!` of `android/a-serve/dist/a-serve.apk`, and
   the download name comes from `ANDROID_APK_VERSION`, which a test keeps equal
   to the gradle `versionName`. Copying a new APK into `dist/` does not change
   a built `bin/a`: bump both version strings, copy, then rebuild and restart
   the daemon. Verify by hashing the download and `dist/a-serve.apk`; equal
   hashes prove the served bytes, not just the filename.
6. **A restart preserves the daemon cwd.** Sessions without a `/cd` entry use
   the daemon's own working directory (`session_cwd.rs`), so inheriting the
   `--serve-restart` caller's cwd would move every such session. `start_daemon`
   launches the frozen copy with `daemon_start_cwd(previous)`: `serve.pid`
   records `cwd` and a restart replays it. On Linux, `running_daemon_cwd`
   (`ctl.rs`) recovers an unrecorded cwd before the stop only if the process
   is still executing a frozen copy from this state's run directory. If that
   identity cannot be verified, or the cwd is missing or relative, restart
   uses the caller's cwd.

## Restart verification

After a deferred restart, inspect the serve log and `a --serve-status` for the
new pid/uptime in a later turn, then confirm the intended frozen executable.
On Linux hash `/proc/<pid>/exe`; on macOS compare the frozen-copy name under
`<state-dir>/run/a-<content-hash>`. `report_status`, `file_hash`, and `freeze_copy`
(`ctl.rs`) own status reporting and frozen-copy identity; the installed `bin/a`
alone does not prove which daemon is running.
