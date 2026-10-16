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
//!
//! The re-exec runs a frozen copy of the binary (`run/a-<hash>` under the
//! state dir) rather than the path the user launched: a serve daemon starts
//! one fresh process per turn from its own file, so a rebuild that replaces
//! that path would leave every later turn failing with `ENOENT`.
//!
//! The copy is what `current_exe()` reports for the daemon and its turns, so
//! `current_exe()`-relative lookups (`tool_descriptions` next to the binary,
//! the lowest-priority metadata source) resolve under `run/` instead of the
//! install dir, and the state dir must be executable (no `noexec` mount).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::fd::AsRawFd as _;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use super::tls::{ServeTlsConfig, default_serve_tls_config, resolve_serve_tls};
use crate::ai::cli::ParsedCli;
use crate::commonw::configw;

/// Env override for the state directory (tests / relocation).
const STATE_DIR_ENV: &str = "A_SERVE_STATE_DIR";
const PID_FILE: &str = "serve.pid";
const LOG_FILE: &str = "serve.log";
/// Persistent inode used to serialize daemon management lifecycle transactions.
#[cfg(unix)]
const CTL_LOCK_FILE: &str = "serve.ctl.lock";
/// Latest deferred restart request. Older helpers must not touch the daemon.
#[cfg(unix)]
const RESTART_REQUEST_FILE: &str = "serve.restart.request";
/// Internal handoff used by a detached restart helper. The helper waits for the
/// serve turn that requested a restart to exit before touching the daemon.
#[cfg(unix)]
const RESTART_AFTER_PID_ENV: &str = "A_SERVE_RESTART_AFTER_PID";
/// Identifies which deferred restart helper is still allowed to run.
#[cfg(unix)]
const RESTART_REQUEST_ENV: &str = "A_SERVE_RESTART_REQUEST";
/// Frozen daemon copies (`a-<content hash>`), see [`freeze_daemon_exe`].
#[cfg(unix)]
const RUN_DIR: &str = "run";
/// Frozen copies kept on disk: the running one plus its predecessor, so a
/// rollback target survives one upgrade.
#[cfg(unix)]
const KEEP_RUN_COPIES: usize = 2;
/// Age after which a `.a-*.tmp` copy left behind by an interrupted start is
/// reclaimed; a younger one may belong to a start that is still copying.
#[cfg(unix)]
const STALE_COPY_TEMP_AGE: Duration = Duration::from_secs(600);
/// How long `--serve-start` waits for `/healthz` before reporting success anyway.
#[cfg(unix)]
const STARTUP_WAIT: Duration = Duration::from_secs(6);
/// SIGTERM grace period in `--serve-stop` before escalating to SIGKILL.
#[cfg(unix)]
const STOP_GRACE: Duration = Duration::from_secs(5);
/// Wait after SIGKILL before giving up in `--serve-stop`.
#[cfg(unix)]
const KILL_WAIT: Duration = Duration::from_secs(2);
/// A turn normally exits immediately after its final response. Bound the wait
/// so a wedged turn cannot leave an immortal restart helper behind.
#[cfg(unix)]
const DEFERRED_RESTART_WAIT: Duration = Duration::from_secs(30 * 60);
/// Give the daemon a short window to flush the completed SSE response after the
/// turn process exits before the deferred helper sends SIGTERM.
#[cfg(unix)]
const RESPONSE_DRAIN_GRACE: Duration = Duration::from_secs(1);

/// On-disk serve state: the legacy three lines followed by optional JSON TLS paths.
struct ServeState {
    pid: u32,
    bind: String,
    started_unix: u64,
    tls: Option<ServeTlsConfig>,
}

/// Entry point for the four `--serve-*` management flags. Synchronous: no
/// runtime is needed (blocking HTTP at most).
pub(in crate::ai) fn run_serve_ctl(cli: &ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    let verbs = [
        cli.serve_start,
        cli.serve_stop,
        cli.serve_restart,
        cli.serve_status,
    ];
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
        return run_direct_serve_change(|| start_daemon(cli, None));
    }
    if cli.serve_stop {
        return run_direct_serve_change(stop_daemon);
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

/// Run one daemon lifecycle transaction while holding a stable cross-process
/// lock. The lock file is intentionally never removed: replacing its inode
/// would let existing waiters and new callers enter different critical sections.
#[cfg(unix)]
fn with_serve_ctl_lock_in<T>(
    dir: &Path,
    operation: impl FnOnce() -> Result<T, Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(dir.join(CTL_LOCK_FILE))?;
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // Closing the descriptor releases flock even when `operation` returns an
    // error or unwinds, so the lock file itself must remain in place.
    operation()
}

#[cfg(unix)]
fn with_serve_ctl_lock<T>(
    operation: impl FnOnce(&Path) -> Result<T, Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>> {
    let dir = state_dir()?;
    with_serve_ctl_lock_in(&dir, || operation(&dir))
}

#[cfg(unix)]
fn valid_restart_request(token: &str) -> bool {
    !token.is_empty()
        && token.len() <= 128
        && token
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'-')
}

#[cfg(unix)]
fn read_restart_request(dir: &Path) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let path = dir.join(RESTART_REQUEST_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let token = text.trim();
    if !valid_restart_request(token) {
        return Err(format!("corrupt deferred restart request at {}", path.display()).into());
    }
    Ok(Some(token.to_string()))
}

#[cfg(unix)]
fn write_restart_request(dir: &Path, token: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !valid_restart_request(token) {
        return Err("invalid deferred restart request token".into());
    }
    let path = dir.join(RESTART_REQUEST_FILE);
    let temp = dir.join(format!(".{RESTART_REQUEST_FILE}.{token}.tmp"));
    std::fs::write(&temp, format!("{token}\n"))?;
    if let Err(err) = std::fs::rename(&temp, &path) {
        let _ = std::fs::remove_file(&temp);
        return Err(err.into());
    }
    Ok(())
}

#[cfg(unix)]
fn clear_restart_request(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    match std::fs::remove_file(dir.join(RESTART_REQUEST_FILE)) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

/// Claim the latest request. A stale helper leaves the newer request intact.
#[cfg(unix)]
fn claim_restart_request(dir: &Path, token: &str) -> Result<bool, Box<dyn std::error::Error>> {
    if read_restart_request(dir)?.as_deref() != Some(token) {
        return Ok(false);
    }
    clear_restart_request(dir)?;
    Ok(true)
}

#[cfg(unix)]
fn new_restart_request() -> String {
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos}-{sequence}", std::process::id())
}

/// A direct management command is newer than every queued helper.
#[cfg(unix)]
fn run_direct_serve_change<T>(
    operation: impl FnOnce() -> Result<T, Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>> {
    with_serve_ctl_lock(|dir| {
        clear_restart_request(dir)?;
        operation()
    })
}

#[cfg(not(unix))]
fn run_direct_serve_change<T>(
    operation: impl FnOnce() -> Result<T, Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>> {
    operation()
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
    let tls_cert: Option<String> = lines
        .next()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|_| "bad TLS certificate line".to_string())?
        .flatten();
    let tls_key: Option<String> = lines
        .next()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|_| "bad TLS key line".to_string())?
        .flatten();
    let tls = match (tls_cert, tls_key) {
        (Some(cert), Some(key)) => Some(ServeTlsConfig { cert, key }),
        (None, None) => None,
        _ => return Err("incomplete TLS state".to_string()),
    };
    Ok(ServeState {
        pid,
        bind,
        started_unix,
        tls,
    })
}

fn format_state(state: &ServeState) -> String {
    let cert = state.tls.as_ref().map(|tls| tls.cert.as_str());
    let key = state.tls.as_ref().map(|tls| tls.key.as_str());
    format!(
        "{}\n{}\n{}\n{}\n{}\n",
        state.pid,
        state.bind,
        state.started_unix,
        serde_json::to_string(&cert).expect("TLS certificate path serializes"),
        serde_json::to_string(&key).expect("TLS key path serializes")
    )
}

/// Read the state file. `Ok(None)` means no daemon was ever recorded here;
/// a corrupt file is an error (it names a pid we must not guess at).
fn read_state_file(path: &Path) -> Result<Option<ServeState>, Box<dyn std::error::Error>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    parse_state(&text).map(Some).map_err(|err| {
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

fn serve_url(bind: &str, tls: Option<&ServeTlsConfig>) -> String {
    let scheme = if tls.is_some() { "https" } else { "http" };
    format!("{scheme}://{bind}")
}

/// GET the public liveness endpoint. Returns the reported server version.
fn query_healthz(
    bind: &str,
    tls: Option<&ServeTlsConfig>,
    timeout: Duration,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        // This probe authenticates the daemon by pid/state ownership. Its TLS
        // certificate may be signed by a phone-only private CA.
        .danger_accept_invalid_certs(tls.is_some())
        .build()?;
    let resp = client
        .get(format!("{}/healthz", serve_url(bind, tls)))
        .send()?;
    if !resp.status().is_success() {
        return Err(format!("/healthz returned {}", resp.status()).into());
    }
    let body: serde_json::Value = resp.json()?;
    Ok(body
        .get("version")
        .and_then(|v| v.as_str())
        .map(str::to_string))
}

/// Poll `/healthz` until it answers or the budget runs out.
fn wait_for_healthz(
    bind: &str,
    tls: Option<&ServeTlsConfig>,
    budget: Duration,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let start = Instant::now();
    loop {
        match query_healthz(bind, tls, Duration::from_secs(2)) {
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
    _previous: Option<&ServeState>,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("--serve-start is unix-only (posix daemonize)".into())
}

/// Start the serve daemon: re-exec a frozen copy of this binary detached, then
/// record its pid.
#[cfg(unix)]
fn start_daemon(
    cli: &ParsedCli,
    previous: Option<&ServeState>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = state_dir()?;
    std::fs::create_dir_all(&dir)?;
    let pid_path = dir.join(PID_FILE);
    if let Some(state) = read_state_file(&pid_path)? {
        if process_alive(state.pid) {
            let base = serve_url(&state.bind, state.tls.as_ref());
            println!(
                "[serve] already running (pid={} bind={base}); use --serve-restart to restart it",
                state.pid
            );
            return Ok(());
        }
        // A killed daemon leaves its state file behind; reclaim it.
        let _ = std::fs::remove_file(&pid_path);
    }
    let bind = resolve_ctl_bind(cli, previous.map(|state| state.bind.as_str()));
    let managed_tls = default_serve_tls_config()?;
    let tls = resolve_serve_tls(
        &cli.serve_tls_cert,
        &cli.serve_tls_key,
        previous
            .and_then(|state| state.tls.as_ref())
            .or(managed_tls.as_ref()),
    )?;
    // Freeze first: the daemon re-execs its own file for every turn, so it
    // must not run from a path that a rebuild can replace.
    let exe = freeze_daemon_exe(&dir)?;
    let log_path = dir.join(LOG_FILE);
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let dev_null = std::fs::OpenOptions::new().read(true).open("/dev/null")?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--serve-detached")
        .arg("--serve")
        .arg("--serve-bind")
        .arg(&bind)
        .stdin(dev_null)
        .stdout(log.try_clone()?)
        .stderr(log);
    if let Some(tls) = tls.as_ref() {
        cmd.arg("--serve-tls-cert")
            .arg(&tls.cert)
            .arg("--serve-tls-key")
            .arg(&tls.key);
    }
    let child = crate::fork_guard::spawn(&mut cmd)?;
    write_state_file(
        &pid_path,
        &ServeState {
            pid: child.id(),
            bind: bind.clone(),
            started_unix: now_unix(),
            tls: tls.clone(),
        },
    )?;
    let base = serve_url(&bind, tls.as_ref());
    match wait_for_healthz(&bind, tls.as_ref(), STARTUP_WAIT) {
        Ok(version) => {
            let version = version.map(|v| format!(" version={v}")).unwrap_or_default();
            println!(
                "[serve] started pid={} bind={base}{version} log={} exe={}",
                child.id(),
                log_path.display(),
                exe.display()
            );
        }
        Err(_) => println!(
            "[serve] started pid={} bind={base} (not answering /healthz yet; tail {})",
            child.id(),
            log_path.display()
        ),
    }
    Ok(())
}

/// Copy the running binary into `<dir>/run/` and return the frozen copy.
///
/// The daemon is started from this copy, so replacing the install path (a
/// rebuild, `cp --remove-destination`) cannot unlink the file every turn
/// re-execs. Copies are keyed by content: an unchanged binary reuses its copy,
/// and pruning keeps [`KEEP_RUN_COPIES`] generations without ever unlinking a
/// file some live process is executing. (The state file also blocks a second
/// start while a daemon runs, but it can be missing or hand-deleted, so
/// pruning does not rely on it.) The state dir must allow exec: the daemon
/// runs from this copy, not from the install path.
#[cfg(unix)]
fn freeze_daemon_exe(dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    freeze_copy(&crate::ai::exe_path::runtime_exe()?, dir)
}

/// [`freeze_daemon_exe`] against an explicit source, so tests can freeze a
/// scratch file instead of the test binary.
#[cfg(unix)]
fn freeze_copy(src: &Path, dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let run_dir = dir.join(RUN_DIR);
    std::fs::create_dir_all(&run_dir)?;
    let hash = file_hash(src)?;
    let dst = run_dir.join(format!("a-{hash:016x}"));
    if !dst.is_file() {
        // Copy under a temporary name and rename into place: a partial copy
        // (disk full, interrupted start) must never be trusted as the frozen
        // binary, and the rename publishes it only once it is complete.
        let tmp = run_dir.join(format!(".a-{hash:016x}-{}.tmp", std::process::id()));
        if let Err(err) = publish_copy(src, &tmp, &dst) {
            let _ = std::fs::remove_file(&tmp);
            return Err(err);
        }
    }
    prune_run_copies(&run_dir, &dst);
    Ok(dst)
}

/// Write `src` to `tmp`, then publish it as `dst`.
#[cfg(unix)]
fn publish_copy(src: &Path, tmp: &Path, dst: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::copy(src, tmp)?;
    // `fs::copy` carries the source mode; force the exec bits so a copy made
    // under a restrictive umask still runs.
    std::fs::set_permissions(tmp, std::fs::Permissions::from_mode(0o755))?;
    std::fs::rename(tmp, dst)?;
    Ok(())
}

/// Content hash naming a frozen copy, so rebuilt binaries never overwrite the
/// copy a running daemon re-execs.
#[cfg(unix)]
fn file_hash(path: &Path) -> std::io::Result<u64> {
    use std::hash::Hasher as _;
    use std::io::Read as _;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.write(&buf[..read]);
    }
    Ok(hasher.finish())
}

/// Drop older frozen copies, keeping `keep` and the newest other generation
/// that no live process is executing. Best effort: a leftover copy only costs
/// disk space, while dropping an executing one would cost a daemon its binary.
#[cfg(unix)]
fn prune_run_copies(run_dir: &Path, keep: &Path) {
    let Ok(entries) = std::fs::read_dir(run_dir) else {
        return;
    };
    let executing = executing_paths();
    let mut others: Vec<(SystemTime, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let Some(modified) = entry.metadata().ok().and_then(|meta| meta.modified().ok()) else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        if name.starts_with(".a-") && name.ends_with(".tmp") {
            // Interrupted copy; reclaimed once it is clearly abandoned.
            let stale = SystemTime::now()
                .duration_since(modified)
                .is_ok_and(|age| age > STALE_COPY_TEMP_AGE);
            if stale {
                let _ = std::fs::remove_file(path);
            }
            continue;
        }
        if !name.starts_with("a-") || path == keep || is_executing(&path, &executing) {
            continue;
        }
        others.push((modified, path));
    }
    others.sort_by(|left, right| right.0.cmp(&left.0));
    for (_, path) in others.into_iter().skip(KEEP_RUN_COPIES.saturating_sub(1)) {
        let _ = std::fs::remove_file(path);
    }
}

/// Whether some live process is executing `path`.
#[cfg(unix)]
fn is_executing(path: &Path, executing: &[PathBuf]) -> bool {
    std::fs::canonicalize(path).is_ok_and(|canonical| executing.contains(&canonical))
}

/// Executable paths of live processes (Linux `/proc/<pid>/exe`). Empty where
/// there is no cheap equivalent, which only widens what may be pruned.
#[cfg(target_os = "linux")]
fn executing_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return paths;
    };
    for entry in entries.flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .bytes()
            .all(|byte| byte.is_ascii_digit())
        {
            continue;
        }
        if let Ok(target) = std::fs::read_link(entry.path().join("exe")) {
            paths.push(target);
        }
    }
    paths
}

#[cfg(all(unix, not(target_os = "linux")))]
fn executing_paths() -> Vec<PathBuf> {
    Vec::new()
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
        return Err(format!(
            "serve is not running (no state file at {})",
            pid_path.display()
        )
        .into());
    };
    if !process_alive(state.pid) {
        let _ = std::fs::remove_file(&pid_path);
        return Err(format!(
            "serve pid {} is already gone (stale state cleaned up)",
            state.pid
        )
        .into());
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

/// Restart the serve daemon. A tool launched by a serve turn inherits that
/// turn's owner PID, so its restart is handed to a detached helper rather than
/// terminating the SSE connection that must carry this command's result.
fn restart_daemon(cli: &ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        if let Some(pid) = env_pid(RESTART_AFTER_PID_ENV) {
            let request = std::env::var(RESTART_REQUEST_ENV)
                .map_err(|_| "deferred restart helper is missing its request token")?;
            println!("[serve] waiting for turn pid={pid} before restart ...");
            if !wait_for_exit(pid, DEFERRED_RESTART_WAIT) {
                return Err(format!(
                    "serve restart abandoned: turn pid {pid} did not exit within {:?}",
                    DEFERRED_RESTART_WAIT
                )
                .into());
            }
            std::thread::sleep(RESPONSE_DRAIN_GRACE);
            // SAFETY: serve management runs before the async runtime or worker
            // threads start, so no concurrent environment access exists here.
            unsafe {
                std::env::remove_var(RESTART_AFTER_PID_ENV);
                std::env::remove_var(RESTART_REQUEST_ENV);
            }
            return with_serve_ctl_lock(|dir| {
                if !claim_restart_request(dir, &request)? {
                    println!("[serve] skipped stale deferred restart request");
                    return Ok(());
                }
                restart_daemon_now(cli)
            });
        }
        if let Some(pid) = env_pid(crate::ai::background::SERVE_TURN_OWNER_PID_ENV) {
            return with_serve_ctl_lock(|dir| schedule_restart_after_turn(cli, pid, dir));
        }
    }
    run_direct_serve_change(|| restart_daemon_now(cli))
}

/// Restart immediately, replaying the previous bind unless `--serve-bind`
/// overrides it. A failed stop still falls through to start (a dead daemon's
/// only residue is its state file, which start reclaims).
fn restart_daemon_now(cli: &ParsedCli) -> Result<(), Box<dyn std::error::Error>> {
    let previous = state_dir()
        .ok()
        .and_then(|dir| read_state_file(&dir.join(PID_FILE)).ok().flatten());
    match stop_daemon() {
        Ok(()) => {}
        Err(err) => eprintln!("[serve] stop phase: {err} (continuing to start)"),
    }
    start_daemon(cli, previous.as_ref())
}

#[cfg(unix)]
fn env_pid(name: &str) -> Option<u32> {
    parse_pid(std::env::var_os(name).as_deref())
}

#[cfg(unix)]
fn parse_pid(value: Option<&std::ffi::OsStr>) -> Option<u32> {
    value
        .and_then(std::ffi::OsStr::to_str)
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|pid| *pid != 0)
}

/// A restart requested by a tool inside a serve turn cannot synchronously stop
/// its own daemon: that would cut off the SSE response carrying the result.
/// Re-exec a detached helper that waits for the owning turn process instead.
#[cfg(unix)]
fn schedule_restart_after_turn(
    cli: &ParsedCli,
    turn_pid: u32,
    dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let request = new_restart_request();
    let log_path = dir.join(LOG_FILE);
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let mut cmd = std::process::Command::new(crate::ai::exe_path::runtime_exe()?);
    cmd.arg("--serve-detached").arg("--serve-restart");
    if !cli.serve_bind.trim().is_empty() {
        cmd.arg("--serve-bind").arg(&cli.serve_bind);
    }
    if !cli.serve_tls_cert.trim().is_empty() {
        cmd.arg("--serve-tls-cert").arg(&cli.serve_tls_cert);
    }
    if !cli.serve_tls_key.trim().is_empty() {
        cmd.arg("--serve-tls-key").arg(&cli.serve_tls_key);
    }
    cmd.env(RESTART_AFTER_PID_ENV, turn_pid.to_string())
        .env(RESTART_REQUEST_ENV, &request)
        .env_remove(crate::ai::background::SERVE_TURN_OWNER_PID_ENV)
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    let helper = crate::fork_guard::spawn(&mut cmd)?;
    // Publish only after spawn succeeds. The helper cannot claim the request
    // before this process releases the lifecycle lock.
    write_restart_request(dir, &request)?;
    println!(
        "[serve] restart scheduled after turn pid={turn_pid} completes (helper pid={}, log={})",
        helper.id(),
        log_path.display()
    );
    Ok(())
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
        return Err(format!(
            "serve is not running (no state file at {})",
            pid_path.display()
        )
        .into());
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
    let base = serve_url(&state.bind, state.tls.as_ref());
    match query_healthz(&state.bind, state.tls.as_ref(), Duration::from_secs(3)) {
        Ok(version) => {
            let version = version.map(|v| format!(" version={v}")).unwrap_or_default();
            println!(
                "[serve] running pid={} bind={} uptime={}{} log={}",
                state.pid,
                base,
                format_uptime(now_unix().saturating_sub(state.started_unix)),
                version,
                dir.join(LOG_FILE).display()
            );
            Ok(())
        }
        Err(err) => Err(format!(
            "serve pid {} {} but not answering {base}/healthz ({err}); still starting, or listening elsewhere?",
            state.pid, PID_LIVENESS_WORD
        )
        .into()),
    }
}

/// Best-effort state-file write for a foreground `a --serve`, so
/// `--serve-status` and `--serve-stop` see it too. Never fails the server:
/// a missing state file only costs manageability, never availability.
pub(super) fn note_foreground_serve(bind: &str, tls: Option<&ServeTlsConfig>) {
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
                "[serve] warning: serve daemon already running (pid={} bind={}); this foreground server will not claim --serve-status/--serve-stop",
                existing.pid,
                serve_url(&existing.bind, existing.tls.as_ref())
            );
            return;
        }
    }
    let state = ServeState {
        pid: me,
        bind: bind.to_string(),
        started_unix: now_unix(),
        tls: tls.cloned(),
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

    #[cfg(unix)]
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("a-serve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[cfg(unix)]
    #[test]
    fn deferred_restart_requests_are_latest_wins() {
        let dir = scratch_dir("restart-request");
        write_restart_request(&dir, "1-10-0").expect("first request");
        write_restart_request(&dir, "1-11-0").expect("newer request");

        assert!(!claim_restart_request(&dir, "1-10-0").expect("stale claim"));
        assert_eq!(
            read_restart_request(&dir).expect("read request").as_deref(),
            Some("1-11-0")
        );
        assert!(claim_restart_request(&dir, "1-11-0").expect("latest claim"));
        assert_eq!(read_restart_request(&dir).expect("request cleared"), None);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Child-side probe used by `serve_ctl_lock_serializes_processes`.
    #[cfg(unix)]
    #[test]
    fn serve_ctl_lock_subprocess() {
        use std::io::Write as _;

        let Some(dir) = std::env::var_os("A_SERVE_CTL_LOCK_TEST_DIR").map(PathBuf::from) else {
            return;
        };
        let role = std::env::var("A_SERVE_CTL_LOCK_TEST_ROLE").expect("lock test role");
        with_serve_ctl_lock_in(&dir, || {
            let mut events = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("events"))?;
            writeln!(events, "{role}-enter")?;
            events.flush()?;
            if role == "holder" {
                std::fs::write(dir.join("holder-entered"), b"ready")?;
                std::thread::sleep(Duration::from_millis(350));
            }
            writeln!(events, "{role}-exit")?;
            Ok(())
        })
        .expect("lock probe");
    }

    #[cfg(unix)]
    #[test]
    fn serve_ctl_lock_serializes_processes() {
        let dir = scratch_dir("ctl-lock");
        let exe = std::env::current_exe().expect("test executable");
        let spawn_probe = |role: &str| {
            let mut cmd = std::process::Command::new(&exe);
            cmd.arg("serve_ctl_lock_subprocess")
                .arg("--test-threads=1")
                .env("A_SERVE_CTL_LOCK_TEST_DIR", &dir)
                .env("A_SERVE_CTL_LOCK_TEST_ROLE", role)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            cmd.spawn().expect("spawn lock probe")
        };

        let mut holder = spawn_probe("holder");
        let deadline = Instant::now() + Duration::from_secs(3);
        while !dir.join("holder-entered").is_file() && Instant::now() < deadline {
            if let Some(status) = holder.try_wait().expect("poll holder") {
                panic!("holder exited before acquiring lock: {status}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            dir.join("holder-entered").is_file(),
            "holder never acquired lock"
        );

        let contender = spawn_probe("contender");
        let contender_output = contender.wait_with_output().expect("wait contender");
        let holder_output = holder.wait_with_output().expect("wait holder");
        assert!(
            holder_output.status.success(),
            "holder failed: {}",
            String::from_utf8_lossy(&holder_output.stderr)
        );
        assert!(
            contender_output.status.success(),
            "contender failed: {}",
            String::from_utf8_lossy(&contender_output.stderr)
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("events")).expect("events"),
            "holder-enter\nholder-exit\ncontender-enter\ncontender-exit\n"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Pin a file's mtime so pruning order does not depend on clock resolution.
    #[cfg(unix)]
    fn set_mtime(path: &Path, unix_secs: u64) {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open for mtime");
        file.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(unix_secs))
            .expect("set mtime");
    }

    /// A frozen copy must be exec-ready, and freezing the same bytes again must
    /// reuse it instead of piling up duplicates.
    #[cfg(unix)]
    #[test]
    fn frozen_copy_is_executable_and_reused() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = scratch_dir("freeze");
        let src = dir.join("src-bin");
        std::fs::write(&src, b"#!/bin/true\n").expect("write source");
        let frozen = freeze_copy(&src, &dir).expect("freeze");
        assert!(frozen.is_file(), "copy missing at {}", frozen.display());
        let mode = std::fs::metadata(&frozen)
            .expect("copy metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o755, "frozen copy must be executable");
        assert_eq!(freeze_copy(&src, &dir).expect("freeze again"), frozen);
    }

    /// Rollback needs the previous generation, but copies must not accumulate.
    #[cfg(unix)]
    #[test]
    fn freezing_prunes_all_but_one_predecessor() {
        let dir = scratch_dir("prune");
        let run_dir = dir.join(RUN_DIR);
        std::fs::create_dir_all(&run_dir).expect("run dir");
        let older = run_dir.join("a-0000000000000001");
        let previous = run_dir.join("a-0000000000000002");
        std::fs::write(&older, b"old").expect("seed older");
        std::fs::write(&previous, b"previous").expect("seed previous");
        set_mtime(&older, 1_000);
        set_mtime(&previous, 2_000);
        let src = dir.join("src-bin");
        std::fs::write(&src, b"#!/bin/true\n").expect("write source");

        let frozen = freeze_copy(&src, &dir).expect("freeze");
        assert!(frozen.is_file());
        assert!(
            previous.is_file(),
            "previous copy stays available for rollback"
        );
        assert!(!older.is_file(), "older copies are pruned");
    }

    /// The prune guard must recognize the file a live process executes.
    #[cfg(target_os = "linux")]
    #[test]
    fn executing_paths_include_the_test_binary() {
        let exe = std::env::current_exe().expect("test binary path");
        let executing = executing_paths();
        assert!(
            is_executing(&exe, &executing),
            "{} runs this test",
            exe.display()
        );
    }

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
        assert!(
            err.to_string().contains("cannot combine"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn state_file_round_trip() {
        let state = ServeState {
            pid: 4242,
            bind: "127.0.0.1:8080".to_string(),
            started_unix: 1700000000,
            tls: Some(ServeTlsConfig {
                cert: "/tmp/local cert.pem".into(),
                key: "/tmp/local key.pem".into(),
            }),
        };
        let back = parse_state(&format_state(&state)).expect("round trip");
        assert_eq!(back.pid, 4242);
        assert_eq!(back.bind, "127.0.0.1:8080");
        assert_eq!(back.started_unix, 1700000000);
        assert_eq!(back.tls, state.tls);
        assert!(
            parse_state("42\n127.0.0.1:8080\n1\n")
                .unwrap()
                .tls
                .is_none()
        );
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
        assert_eq!(
            resolve_ctl_bind(&cli, Some("127.0.0.1:4321")),
            "127.0.0.1:4321"
        );
    }

    #[cfg(unix)]
    #[test]
    fn deferred_restart_pid_requires_a_positive_integer() {
        assert_eq!(parse_pid(Some(std::ffi::OsStr::new("42"))), Some(42));
        for invalid in [
            None,
            Some(std::ffi::OsStr::new("")),
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("nope")),
        ] {
            assert_eq!(parse_pid(invalid), None);
        }
    }

    #[test]
    fn uptime_formats_compactly() {
        assert_eq!(format_uptime(8), "8s");
        assert_eq!(format_uptime(90), "1m30s");
        assert_eq!(format_uptime(3725), "1h2m");
        assert_eq!(format_uptime(90061), "1d1h");
    }
}
