//! Process supervision for `a --serve`: `--serve-start`, `--serve-stop`,
//! `--serve-restart`, and `--serve-status`.
//!
//! A serve daemon is machine-global (unlike `-bg` tasks, which are
//! session-scoped and tracked by `<sessionid>.pid` in the working directory),
//! so its state lives in one fixed place: `serve.pid` plus `serve.log` under
//! the user config dir (`A_SERVE_STATE_DIR` overrides it for tests and
//! relocation).
//!
//! Daemonization follows the `-bg` constraints: no `fork` on macOS (it copies
//! half-initialized objc state into the child). The parent re-execs a fresh
//! `a --serve-detached --serve --serve-bind <addr>` process through
//! `fork_guard::spawn`, and the child detaches its own session on startup via
//! `background::detach_daemon_session`, mirroring `--daemon-child`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::ai::cli::ParsedCli;
use crate::commonw::configw;

/// Env override for the state directory (tests / relocation).
const STATE_DIR_ENV: &str = "A_SERVE_STATE_DIR";
const PID_FILE: &str = "serve.pid";
const LOG_FILE: &str = "serve.log";
/// How long `--serve-start` waits for `/healthz` before reporting success anyway.
#[cfg(unix)]
const STARTUP_WAIT: Duration = Duration::from_secs(6);
/// SIGTERM grace period in `--serve-stop` before escalating to SIGKILL.
#[cfg(unix)]
const STOP_GRACE: Duration = Duration::from_secs(5);
/// Wait after SIGKILL before giving up in `--serve-stop`.
#[cfg(unix)]
const KILL_WAIT: Duration = Duration::from_secs(2);

/// On-disk serve state: one value per line (`pid`, `bind`, unix start time).
struct ServeState {
    pid: u32,
    bind: String,
    started_unix: u64,
}

/// Entry point for the four `--serve-*` management flags. Synchronous: no
/// runtime is needed (blocking HTTP at most).
pub(in crate::ai) fn run_serve_ctl(cli: &ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    let verbs = [cli.serve_start, cli.serve_stop, cli.serve_restart, cli.serve_status];
    if verbs.iter().filter(|flag| **flag).count() > 1 {
        return Err(
            "only one of --serve-start/--serve-stop/--serve-restart/--serve-status may be given"
                .into(),
        );
    }
    if cli.serve || cli.serve_chat || cli.serve_sessions {
        return Err(
            "cannot combine --serve/--serve-chat/--serve-sessions with --serve-start/--serve-stop/--serve-restart/--serve-status"
                .into(),
        );
    }
    if cli.serve_start {
        return start_daemon(cli, None);
    }
    if cli.serve_stop {
        return stop_daemon();
    }
    if cli.serve_restart {
        return restart_daemon(cli);
    }
    if cli.serve_status {
        return report_status();
    }
    Err("run_serve_ctl called without a serve management flag".into())
}

/// Fixed state directory: the user config dir, unless overridden for tests.
fn state_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(dir) = std::env::var_os(STATE_DIR_ENV) {
        if dir.is_empty() {
            return Err("A_SERVE_STATE_DIR is set but empty".into());
        }
        return Ok(PathBuf::from(dir));
    }
    Ok(crate::commonw::utils::get_config_dir()
        .ok_or("cannot resolve user config dir")?
        .join("rust_tools"))
}

fn parse_state(text: &str) -> Result<ServeState, String> {
    let mut lines = text.lines();
    let pid: u32 = lines
        .next()
        .ok_or("missing pid line")?
        .trim()
        .parse()
        .map_err(|_| "bad pid line".to_string())?;
    if pid == 0 {
        return Err("bad pid line".to_string());
    }
    let bind = lines.next().ok_or("missing bind line")?.trim().to_string();
    if bind.is_empty() {
        return Err("empty bind line".to_string());
    }
    let started_unix: u64 = lines
        .next()
        .ok_or("missing started line")?
        .trim()
        .parse()
        .map_err(|_| "bad started line".to_string())?;
    Ok(ServeState { pid, bind, started_unix })
}

fn format_state(state: &ServeState) -> String {
    format!("{}\n{}\n{}\n", state.pid, state.bind, state.started_unix)
}

/// Read the state file. `Ok(None)` means no daemon was ever recorded here;
/// a corrupt file is an error (it names a pid we must not guess at).
fn read_state_file(path: &Path) -> Result<Option<ServeState>, Box<dyn std::error::Error>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    parse_state(&text)
        .map(Some)
        .map_err(|err| {
            format!(
                "serve state file {} is corrupt ({err}); delete it by hand",
                path.display()
            )
            .into()
        })
}

fn write_state_file(path: &Path, state: &ServeState) -> std::io::Result<()> {
    std::fs::write(path, format_state(state))?;
    Ok(())
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(unix)]
fn send_signal(pid: u32, sig: libc::c_int) -> Result<(), Box<dyn std::error::Error>> {
    if unsafe { libc::kill(pid as libc::pid_t, sig) != 0 } {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(unix)]
fn wait_for_exit(pid: u32, budget: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < budget {
        if !process_alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    !process_alive(pid)
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn format_uptime(secs: u64) -> String {
    const DAY: u64 = 24 * 3600;
    const HOUR: u64 = 3600;
    const MIN: u64 = 60;
    if secs >= DAY {
        format!("{}d{}h", secs / DAY, secs % DAY / HOUR)
    } else if secs >= HOUR {
        format!("{}h{}m", secs / HOUR, secs % HOUR / MIN)
    } else if secs >= MIN {
        format!("{}m{}s", secs / MIN, secs % MIN)
    } else {
        format!("{secs}s")
    }
}

fn healthz_url(bind: &str) -> String {
    format!("http://{bind}/healthz")
}

/// GET the public liveness endpoint. Returns the reported server version.
fn query_healthz(bind: &str, timeout: Duration) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder().timeout(timeout).build()?;
    let resp = client.get(healthz_url(bind)).send()?;
    if !resp.status().is_success() {
        return Err(format!("/healthz returned {}", resp.status()).into());
    }
    let body: serde_json::Value = resp.json()?;
    Ok(body.get("version").and_then(|v| v.as_str()).map(str::to_string))
}

/// Poll `/healthz` until it answers or the budget runs out.
fn wait_for_healthz(bind: &str, budget: Duration) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let start = Instant::now();
    loop {
        match query_healthz(bind, Duration::from_secs(2)) {
            Ok(version) => return Ok(version),
            Err(err) if start.elapsed() < budget => {
                let _ = err;
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(err) => return Err(err),
        }
    }
}

/// Effective bind for a managed daemon: an explicit `--serve-bind` wins, then
/// the bind recorded by the previous daemon (restart replay), then the same
/// config/default resolution a foreground `a --serve` uses.
fn resolve_ctl_bind(cli: &ParsedCli, forced_bind: Option<&str>) -> String {
    if !cli.serve_bind.trim().is_empty() {
        return cli.serve_bind.trim().to_string();
    }
    if let Some(bind) = forced_bind {
        if !bind.trim().is_empty() {
            return bind.trim().to_string();
        }
    }
    let cfg_bind = configw::get_all_config()
        .get_opt(crate::ai::config_schema::AiConfig::SERVE_BIND)
        .unwrap_or_default();
    super::resolve_serve_bind(&cli.serve_bind, &cfg_bind)
}

#[cfg(not(unix))]
fn start_daemon(
    _cli: &ParsedCli,
    _forced_bind: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("--serve-start is unix-only (posix daemonize)".into())
}

/// Start the serve daemon: re-exec this binary detached, then record its pid.
#[cfg(unix)]
fn start_daemon(
    cli: &ParsedCli,
    forced_bind: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = state_dir()?;
    std::fs::create_dir_all(&dir)?;
    let pid_path = dir.join(PID_FILE);
    if let Some(state) = read_state_file(&pid_path)? {
        if process_alive(state.pid) {
            println!(
                "[serve] already running (pid={} bind=http://{}); use --serve-restart to restart it",
                state.pid, state.bind
            );
            return Ok(());
        }
        // A killed daemon leaves its state file behind; reclaim it.
        let _ = std::fs::remove_file(&pid_path);
    }
    let bind = resolve_ctl_bind(cli, forced_bind);
    let exe = std::env::current_exe()?;
    let log_path = dir.join(LOG_FILE);
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let dev_null = std::fs::OpenOptions::new().read(true).open("/dev/null")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--serve-detached")
        .arg("--serve")
        .arg("--serve-bind")
        .arg(&bind)
        .stdin(dev_null)
        .stdout(log.try_clone()?)
        .stderr(log);
    let child = crate::fork_guard::spawn(&mut cmd)?;
    write_state_file(
        &pid_path,
        &ServeState {
            pid: child.id(),
            bind: bind.clone(),
            started_unix: now_unix(),
        },
    )?;
    match wait_for_healthz(&bind, STARTUP_WAIT) {
        Ok(version) => {
            let version = version.map(|v| format!(" version={v}")).unwrap_or_default();
            println!(
                "[serve] started pid={} bind=http://{bind}{version} log={}",
                child.id(),
                log_path.display()
            );
        }
        Err(_) => println!(
            "[serve] started pid={} bind=http://{bind} (not answering /healthz yet; tail {})",
            child.id(),
            log_path.display()
        ),
    }
    Ok(())
}

#[cfg(not(unix))]
fn stop_daemon() -> Result<(), Box<dyn std::error::Error>> {
    Err("--serve-stop is unix-only (posix signals)".into())
}

/// Stop the serve daemon: SIGTERM with a grace period, then SIGKILL.
#[cfg(unix)]
fn stop_daemon() -> Result<(), Box<dyn std::error::Error>> {
    let pid_path = state_dir()?.join(PID_FILE);
    let Some(state) = read_state_file(&pid_path)? else {
        return Err(format!("serve is not running (no state file at {})", pid_path.display()).into());
    };
    if !process_alive(state.pid) {
        let _ = std::fs::remove_file(&pid_path);
        return Err(format!("serve pid {} is already gone (stale state cleaned up)", state.pid).into());
    }
    println!("[serve] stopping pid={} ...", state.pid);
    send_signal(state.pid, libc::SIGTERM)?;
    if wait_for_exit(state.pid, STOP_GRACE) {
        let _ = std::fs::remove_file(&pid_path);
        println!("[serve] stopped pid={}", state.pid);
        return Ok(());
    }
    eprintln!(
        "[serve] pid={} did not exit within {:?}; sending SIGKILL",
        state.pid, STOP_GRACE
    );
    send_signal(state.pid, libc::SIGKILL)?;
    if wait_for_exit(state.pid, KILL_WAIT) {
        let _ = std::fs::remove_file(&pid_path);
        println!("[serve] killed pid={}", state.pid);
        return Ok(());
    }
    Err(format!(
        "serve pid {} refused to exit; kill it by hand, then delete {}",
        state.pid,
        pid_path.display()
    )
    .into())
}

/// Restart the serve daemon, replaying the previous bind unless `--serve-bind`
/// overrides it. A failed stop still falls through to start (a dead daemon's
/// only residue is its state file, which start reclaims).
fn restart_daemon(cli: &ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    let previous_bind = state_dir()
        .ok()
        .and_then(|dir| read_state_file(&dir.join(PID_FILE)).ok().flatten())
        .map(|state| state.bind);
    match stop_daemon() {
        Ok(()) => {}
        Err(err) => eprintln!("[serve] stop phase: {err} (continuing to start)"),
    }
    start_daemon(cli, previous_bind.as_deref())
}

/// Liveness wording differs by platform: non-unix builds cannot signal-check
/// the recorded pid, so they only claim what the state file records.
#[cfg(unix)]
const PID_LIVENESS_WORD: &str = "is alive";
#[cfg(not(unix))]
const PID_LIVENESS_WORD: &str = "is recorded";

/// Report daemon state: recorded pid, `/healthz` answer, uptime, log path.
/// Exits non-zero (via `Err`) unless the daemon answers, so scripts can test it.
fn report_status() -> Result<(), Box<dyn std::error::Error>> {
    let dir = state_dir()?;
    let pid_path = dir.join(PID_FILE);
    let Some(state) = read_state_file(&pid_path)? else {
        return Err(format!("serve is not running (no state file at {})", pid_path.display()).into());
    };
    #[cfg(unix)]
    if !process_alive(state.pid) {
        let _ = std::fs::remove_file(&pid_path);
        return Err(format!(
            "serve pid {} is gone; stale state at {} was cleaned up",
            state.pid,
            pid_path.display()
        )
        .into());
    }
    match query_healthz(&state.bind, Duration::from_secs(3)) {
        Ok(version) => {
            let version = version.map(|v| format!(" version={v}")).unwrap_or_default();
            println!(
                "[serve] running pid={} bind=http://{} uptime={}{} log={}",
                state.pid,
                state.bind,
                format_uptime(now_unix().saturating_sub(state.started_unix)),
                version,
                dir.join(LOG_FILE).display()
            );
            Ok(())
        }
        Err(err) => Err(format!(
            "serve pid {} {} but not answering http://{}/healthz ({err}); still starting, or listening elsewhere?",
            state.pid, PID_LIVENESS_WORD, state.bind
        )
        .into()),
    }
}

/// Best-effort state-file write for a foreground `a --serve`, so
/// `--serve-status` and `--serve-stop` see it too. Never fails the server:
/// a missing state file only costs manageability, never availability.
pub(in crate::ai) fn note_foreground_serve(bind: &str) {
    let dir = match state_dir() {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!(
                "[serve] warning: cannot resolve state dir ({err}); --serve-status will not see this process"
            );
            return;
        }
    };
    let me = std::process::id();
    let pid_path = dir.join(PID_FILE);
    // Single-slot state file: never evict a live daemon record. A foreground
    // server on another bind would otherwise orphan the daemon (and on clean
    // exit delete the file, leaving a ghost daemon `--serve-stop` cannot see).
    #[cfg(unix)]
    if let Ok(Some(existing)) = read_state_file(&pid_path) {
        if existing.pid != me && process_alive(existing.pid) {
            eprintln!(
                "[serve] warning: serve daemon already running (pid={} bind=http://{}); this foreground server will not claim --serve-status/--serve-stop",
                existing.pid, existing.bind
            );
            return;
        }
    }
    let state = ServeState {
        pid: me,
        bind: bind.to_string(),
        started_unix: now_unix(),
    };
    if let Err(err) =
        std::fs::create_dir_all(&dir).and_then(|_| write_state_file(&pid_path, &state))
    {
        eprintln!(
            "[serve] warning: cannot write serve state file ({err}); --serve-status will not see this process"
        );
    }
}

/// Drop the state file after a clean foreground shutdown, but only if it
/// still points at this process (a newer daemon may have claimed it).
pub(in crate::ai) fn clear_foreground_serve() {
    let Ok(dir) = state_dir() else { return };
    let pid_path = dir.join(PID_FILE);
    if let Ok(Some(state)) = read_state_file(&pid_path) {
        if state.pid == std::process::id() {
            let _ = std::fs::remove_file(&pid_path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctl_args(extra: &[&str]) -> ParsedCli {
        let mut raw = vec!["a".to_string()];
        raw.extend(extra.iter().map(|s| s.to_string()));
        crate::ai::cli::parse_cli_args(raw.into_iter())
    }

    #[test]
    fn ctl_verbs_are_mutually_exclusive() {
        let cli = ctl_args(&["--serve-start", "--serve-stop"]);
        let err = run_serve_ctl(&cli).expect_err("two verbs must be rejected");
        assert!(err.to_string().contains("only one"), "unexpected: {err}");
    }

    #[test]
    fn ctl_rejects_combination_with_serve() {
        let cli = ctl_args(&["--serve", "--serve-status"]);
        let err = run_serve_ctl(&cli).expect_err("serve + verb must be rejected");
        assert!(err.to_string().contains("cannot combine"), "unexpected: {err}");
    }

    #[test]
    fn state_file_round_trip() {
        let state = ServeState {
            pid: 4242,
            bind: "127.0.0.1:8080".to_string(),
            started_unix: 1700000000,
        };
        let back = parse_state(&format_state(&state)).expect("round trip");
        assert_eq!(back.pid, 4242);
        assert_eq!(back.bind, "127.0.0.1:8080");
        assert_eq!(back.started_unix, 1700000000);
    }

    #[test]
    fn state_file_rejects_garbage() {
        for bad in [
            "",
            "\n\n\n",
            "abc\n127.0.0.1:8080\n1\n",
            "0\n127.0.0.1:8080\n1\n",
            "42\n\n1\n",
            "42\n127.0.0.1:8080\n",
        ] {
            assert!(parse_state(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn explicit_bind_wins_without_touching_config() {
        let cli = ctl_args(&["--serve-start", "--serve-bind", "127.0.0.1:9999"]);
        assert_eq!(resolve_ctl_bind(&cli, None), "127.0.0.1:9999");
        assert_eq!(resolve_ctl_bind(&cli, Some("10.0.0.1:1")), "127.0.0.1:9999");
    }

    #[test]
    fn restart_replay_is_used_when_no_explicit_bind() {
        let cli = ctl_args(&["--serve-restart"]);
        assert!(cli.serve_bind.trim().is_empty());
        assert_eq!(resolve_ctl_bind(&cli, Some("127.0.0.1:4321")), "127.0.0.1:4321");
    }

    #[test]
    fn uptime_formats_compactly() {
        assert_eq!(format_uptime(8), "8s");
        assert_eq!(format_uptime(90), "1m30s");
        assert_eq!(format_uptime(3725), "1h2m");
        assert_eq!(format_uptime(90061), "1d1h");
    }
}
