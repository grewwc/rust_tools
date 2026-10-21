# Serve Guide

## Scope

Applies to `src/bin/ai/serve/**`. Process supervision (`--serve-start`,
`--serve-stop`, `--serve-restart`, `--serve-status`) lives in `ctl.rs`; the
HTTP/SSE surface lives in `mod.rs`.

## Key invariants

1. **The daemon runs a frozen copy.** `start_daemon` re-execs
   `<state-dir>/run/a-<content-hash>`, never the install path. Installing a
   new `bin/a` does not change a running daemon — only a restart re-freezes.
   Prove which binary runs by hashing `/proc/<pid>/exe`, not by reading
   `bin/a`.
2. **A restart requested from inside a turn is deferred.** Tool subprocesses
   inherit `A_SERVE_TURN_OWNER_PID` (the turn's own process), so
   `--serve-restart` spawns a detached helper that waits for that process to
   exit and only then stops and starts the daemon
   (`schedule_restart_after_turn`). The restart lands *after* the turn, not
   when the command returns.
3. **Deferral is capped and can be abandoned.** The helper waits at most
   `DEFERRED_RESTART_WAIT` (30 min). A long-lived owner — a `--terminal-host`
   session, for instance — never exits, so the helper reports "serve restart
   abandoned" and nothing restarts. After a deferred restart, read the serve
   log and re-check `--serve-status` instead of assuming it happened.
4. **Never stop the daemon that is carrying the current turn.** The daemon
   streams that turn's response over SSE, so an immediate stop/start truncates
   the reply. That is why the immediate path is taken only when
   `A_SERVE_TURN_OWNER_PID` is absent: a turn-free shell, or the deferred
   helper once the turn is gone.
5. **Each turn is a child process.** `run_one_shot_turn` spawns the turn, so a
   panicking tool aborts the turn and every tool result in it, while the daemon
   survives. Tool implementations must turn panics into ordinary tool errors.
6. **Health follows the TLS setting.** `serve_url` prefixes `https://` when a
   certificate is configured and `http://` otherwise, and `--serve-status`
   prints that URL as `bind=`. Probe `<bind>/healthz` with the same scheme;
   plain HTTP against a TLS port fails.
