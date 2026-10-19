//! Long-lived command sessions for `execute_command`.
//!
//! A one-shot tool call cannot keep a build, REPL, or interactive prompt alive
//! between model rounds. This module owns a small, bounded registry of child
//! processes: `start` yields partial output plus a session id, later calls poll
//! the same child or write to its stdin. Sessions are reaped on idle/limit so a
//! forgotten process cannot live for the whole daemon lifetime.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Once};
use std::time::{Duration, Instant};
#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    /// Ask the kernel to terminate this process when its parent exits. This is
    /// the macOS counterpart of Linux `PR_SET_PDEATHSIG` and covers parent
    /// SIGKILL, where no in-process cleanup can run.
    fn proc_setpcontrol(control: libc::c_int) -> libc::c_int;
}

#[cfg(target_os = "macos")]
const PROC_SETPC_TERMINATE: libc::c_int = 3;

/// Maximum live sessions per process.
const MAX_SESSIONS: usize = 8;
/// A session is reaped after this long without a tool call touching it.
const SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(600);
/// How often the detached reaper wakes up to kill idle sessions. The timeout is
/// enforced even when the model starts a session and never touches
/// `execute_command` again.
const REAPER_INTERVAL: Duration = Duration::from_secs(30);
/// Maximum bytes retained per stream; older bytes are dropped from the buffer.
const MAX_BUFFER_BYTES: usize = 128 * 1024;
/// Maximum bytes returned to the model in one poll.
const MAX_RETURN_BYTES: usize = 16_000;
/// Polling interval while a session has produced nothing yet.
const POLL_INTERVAL: Duration = Duration::from_millis(10);
/// Upper bound on reaping a killed session. SIGKILL normally reaps instantly;
/// a child stuck in uninterruptible I/O must not block shutdown indefinitely.
const TERMINATE_WAIT: Duration = Duration::from_millis(500);

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct StreamBuffer {
    bytes: Vec<u8>,
    dropped: u64,
    total: u64,
    exited: Option<ExitStatus>,
}

impl StreamBuffer {
    fn append(&mut self, chunk: &[u8]) {
        self.total = self.total.saturating_add(chunk.len() as u64);
        self.bytes.extend_from_slice(chunk);
        if self.bytes.len() > MAX_BUFFER_BYTES {
            let excess = self.bytes.len() - MAX_BUFFER_BYTES;
            self.bytes.drain(0..excess);
            self.dropped = self.dropped.saturating_add(excess as u64);
        }
    }

    /// Return bytes after `cursor`, bounded by [`MAX_RETURN_BYTES`]. The
    /// returned cursor is the absolute stream position consumed.
    fn since(&self, cursor: u64) -> (String, u64) {
        let first_available = self.dropped;
        let start = cursor.max(first_available);
        let offset = start.saturating_sub(first_available) as usize;
        let end = self
            .bytes
            .len()
            .min(offset.saturating_add(MAX_RETURN_BYTES));
        let body = String::from_utf8_lossy(&self.bytes[offset.min(self.bytes.len())..end]).to_string();
        let mut out = String::new();
        if cursor < first_available {
            out.push_str(&format!(
                "[{} bytes of older output permanently dropped and cannot be re-read]\n",
                first_available - cursor
            ));
        }
        out.push_str(&body);
        (out, start.saturating_add((end.saturating_sub(offset)) as u64))
    }
}

struct Session {
    child: Child,
    pgid: u32,
    owner_session_id: String,
    pty: bool,
    stdin: Option<ChildStdin>,
    stdout: Arc<Mutex<StreamBuffer>>,
    stderr: Arc<Mutex<StreamBuffer>>,
    readers: Arc<AtomicUsize>,
    cursor_out: u64,
    cursor_err: u64,
    last_used: Instant,
    terminate_attempted: bool,
}

impl Session {
    fn snapshot(&mut self, wait: Duration) -> String {
        let deadline = Instant::now() + wait;
        loop {
            if is_cancel_requested() {
                // Ctrl+C must not wait out the rest of the yield: keep the
                // session alive for a later poll, but return immediately with
                // whatever was produced so far.
                let mut body = self.render(None);
                body.push_str("\n[yield cancelled; session retained, poll again to resume]\n");
                return body;
            }
            let status = self.child.try_wait().ok().flatten();
            if let Some(status) = status {
                // The child closing its pipes is not the same as the reader
                // threads having consumed them. Drain before rendering so the
                // final bytes are not dropped with the session entry.
                self.drain_readers();
                self.stdout.lock().unwrap().exited = Some(status);
                self.stderr.lock().unwrap().exited = Some(status);
                return self.render(Some(status));
            }
            if Instant::now() >= deadline {
                return self.render(None);
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn render(&mut self, status: Option<ExitStatus>) -> String {
        let (stdout, next_out) = self.stdout.lock().unwrap().since(self.cursor_out);
        let (stderr, next_err) = self.stderr.lock().unwrap().since(self.cursor_err);
        self.cursor_out = next_out;
        self.cursor_err = next_err;
        self.last_used = Instant::now();

        let mut out = String::new();
        match status {
            Some(status) if !status.success() => {
                // Match the one-shot runner's canonical failure marker. Every
                // consumer (completion gate, progress classifier, checkpoints)
                // already treats a leading `Exit code:` as a failed command;
                // a session header must not invent a second dialect.
                out.push_str(&format!("Exit code: {}\n", status.code().unwrap_or(-1)));
                out.push_str(&format!(
                    "[command session exited: {}]\n",
                    status
                        .code()
                        .map(|code| format!("exit code {code}"))
                        .unwrap_or_else(|| "terminated by signal".to_string())
                ));
            }
            Some(status) => {
                out.push_str(&format!(
                    "[command session exited: {}]\n",
                    status
                        .code()
                        .map(|code| format!("exit code {code}"))
                        .unwrap_or_else(|| "terminated by signal".to_string())
                ));
            }
            None => out.push_str("[command session running]\n"),
        }
        if !stdout.is_empty() {
            out.push_str("stdout:\n");
            out.push_str(stdout.trim_end());
            out.push('\n');
        }
        if !stderr.is_empty() {
            out.push_str("stderr:\n");
            out.push_str(stderr.trim_end());
            out.push('\n');
        }
        if out.ends_with("[command session running]\n") {
            out.push_str("(no new output in this yield)");
        }
        out
    }

    fn has_pending_output(&self) -> bool {
        self.cursor_out < self.stdout.lock().unwrap().total
            || self.cursor_err < self.stderr.lock().unwrap().total
    }

    fn terminate(&mut self) {
        if self.terminate_attempted {
            return;
        }
        self.terminate_attempted = true;
        // A session can outlive its child in the registry while buffered output
        // is still unconsumed. Once the child is reaped, its pid (and therefore
        // the process-group id under `setsid`) may be recycled; never signal a
        // group that no longer belongs to this session.
        if self.child.try_wait().ok().flatten().is_some() {
            self.drain_readers();
            return;
        }
        #[cfg(unix)]
        unsafe {
            // `setsid` makes the child a session leader, so its pid is also
            // the process-group id. Killing the group reaps shell
            // grandchildren that a direct child kill would leave behind.
            libc::kill(-(self.pgid as libc::pid_t), libc::SIGKILL);
        }
        let _ = self.child.kill();
        self.wait_bounded(TERMINATE_WAIT);
    }

    /// Reap a killed child without ever blocking shutdown indefinitely. A child
    /// in uninterruptible I/O can survive SIGKILL for an unbounded time; after
    /// the deadline the process-group kill and parent-death signal remain the
    /// backstops, and the zombie is left to process exit rather than stalling
    /// every later session in the drain loop.
    fn wait_bounded(&mut self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) | Err(_) => return,
                Ok(None) if Instant::now() >= deadline => return,
                Ok(None) => std::thread::sleep(POLL_INTERVAL),
            }
        }
    }

    fn drain_readers(&self) {
        let deadline = Instant::now() + Duration::from_millis(250);
        while self.readers.load(Ordering::Acquire) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        super::process_registry::unregister(&self.owner_session_id, self.pgid);
        // Do not signal a recycled process group after a normal exit; only
        // clean up when the direct child is still live.
        if !self.terminate_attempted && self.child.try_wait().ok().flatten().is_none() {
            self.terminate();
        }
    }
}

/// Global id -> session map. The global lock is only held for lookup /
/// insert / remove; all waiting on child output happens under the
/// per-session lock so one session's 30s yield never blocks other sessions,
/// the reaper, or shutdown.
static SESSIONS: LazyLock<Mutex<HashMap<String, Arc<Mutex<Session>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static REAPER: Once = Once::new();

/// Lock the global registry, recovering from a poisoned mutex. A tool call
/// that panics mid-registry-op must not wedge every later command session;
/// the worst case of continuing with recovered state is a stale entry, which
/// the idle reaper and fail-closed lookups already tolerate.
fn lock_registry() -> std::sync::MutexGuard<'static, HashMap<String, Arc<Mutex<Session>>>> {
    SESSIONS.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Lock a session or one of its stream buffers, recovering from poisoning.
/// A panic while formatting a tool result must not turn that session into a
/// permanent panic source for every later poll/close.
fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Mirror the one-shot runner's cancellation contract: an explicit tool
/// cancel or a turn-level `cancel_stream` (Ctrl+C) stops waiting.
fn is_cancel_requested() -> bool {
    crate::ai::tools::registry::common::is_tool_cancel_requested()
        || crate::ai::driver::runtime_ctx::try_current()
            .is_some_and(|context| context.app_proto.cancel_stream.load(Ordering::Acquire))
}

/// Scope a command session to the agent session that created it. A handle
/// leaked across agent sessions fails closed with the same message as an
/// unknown id so existence is not oracle-able.
fn check_owner(session: &Session, session_id: &str) -> Result<(), String> {
    let current = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    if session.owner_session_id == current {
        Ok(())
    } else {
        Err(format!("Unknown command session: {session_id}"))
    }
}

/// Start the process-wide session reaper once. A detached thread is required:
/// `reap_idle` on the tool-call path only runs when the model calls
/// `execute_command` again, which is exactly what a forgotten session does not do.
fn ensure_reaper() {
    REAPER.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("command-session-reaper".to_string())
            .spawn(|| {
                loop {
                    std::thread::sleep(REAPER_INTERVAL);
                    reap_idle();
                }
            });
    });
}

/// Start a session and return its id plus the first output snapshot.
pub(crate) fn start(
    command: &str,
    cwd: Option<&str>,
    pty: bool,
    input: Option<&str>,
    yield_time_ms: u64,
) -> Result<String, String> {
    ensure_reaper();
    reap_idle();
    {
        let sessions = lock_registry();
        if sessions.len() >= MAX_SESSIONS {
            return Err(format!(
                "Command session limit reached ({MAX_SESSIONS}); close or wait for an existing session"
            ));
        }
    }

    let mut cmd = if pty {
        // macOS `script` allocates a PTY while keeping the child's stdin/stdout
        // available to the session reader/writer threads. Use the same `sh -c`
        // shape as the one-shot runner; `bash -lc` additionally sources login
        // profiles and silently changes PATH/aliases between the two paths.
        let mut cmd = Command::new("script");
        cmd.arg("-q").arg("/dev/null").arg("sh").arg("-c").arg(command);
        cmd
    } else {
        crate::cmd::run::build_session_command(command, cwd)
            .map_err(|err| format!("Command session failed to build command: {err}"))?
    };
    // The non-PTY builder already applied `cwd`; `script` is the wrapper that
    // needs it on the outer process.
    if pty && let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Same environment contract as the one-shot runner: default non-interactive
    // (pager/editor/prompt suppression), PTY keeps terminal capabilities.
    crate::cmd::run::apply_session_env_policy(&mut cmd, pty);
    #[cfg(unix)]
    unsafe {
        // Give each session its own process group so reaping is transitive.
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            #[cfg(target_os = "linux")]
            unsafe {
                let _ = libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
            }
            #[cfg(target_os = "macos")]
            let _ = proc_setpcontrol(PROC_SETPC_TERMINATE);
            Ok(())
        });
    }
    if pty {
        cmd.env("TERM", "xterm-256color");
    }

    // All spawns go through the macOS fork guard: forking while another
    // thread initializes Objective-C classes crashes the child.
    let mut child = crate::fork_guard::spawn(&mut cmd).map_err(|err| {
        format!("Command session failed to start: {err}")
    })?;
    // `setsid()` in `pre_exec` makes the child a session leader, so its pid is
    // also its process-group id; `spawn` fails rather than silently continuing
    // when `setsid` fails.
    let pgid = child.id();
    let stdin = child.stdin.take();
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Command session missing stdout pipe".to_string());
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Command session missing stderr pipe".to_string());
        }
    };
    let stdout_buf = Arc::new(Mutex::new(StreamBuffer::default()));
    let stderr_buf = Arc::new(Mutex::new(StreamBuffer::default()));
    let readers = Arc::new(AtomicUsize::new(2));
    spawn_reader(stdout, Arc::clone(&stdout_buf), Arc::clone(&readers));
    spawn_reader(stderr, Arc::clone(&stderr_buf), Arc::clone(&readers));

    let id = format!("cmd-{}-{}", child.id(), NEXT_SESSION.fetch_add(1, Ordering::Relaxed));
    let mut session = Session {
        child,
        pgid,
        owner_session_id: crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
        pty,
        stdin,
        stdout: stdout_buf,
        stderr: stderr_buf,
        readers,
        cursor_out: 0,
        cursor_err: 0,
        last_used: Instant::now(),
        terminate_attempted: false,
    };
    super::process_registry::register(&session.owner_session_id, pgid);
    if let Some(input) = input {
        let stdin = session
            .stdin
            .as_mut()
            .ok_or_else(|| "Command session stdin is unavailable".to_string())?;
        stdin
            .write_all(input.as_bytes())
            .and_then(|_| stdin.flush())
            .map_err(|err| format!("Command session stdin write failed: {err}"))?;
    }
    let pty_for_hint = session.pty;
    let body = session.snapshot(Duration::from_millis(yield_time_ms));
    let exited = session.stdout.lock().unwrap().exited.is_some();
    let drained = !session.has_pending_output();
    let rendered = render_with_id(&id, &body, exited, drained, pty_for_hint);
    if exited && drained {
        let _ = session.child.wait();
    } else {
        lock_registry().insert(id, Arc::new(Mutex::new(session)));
    }
    Ok(rendered)
}

/// Look up one session without holding the global lock across the wait.
/// Sessions are scoped to their creating agent session: a handle presented
/// from another agent session fails closed as unknown.
fn lookup_owned(session_id: &str) -> Result<Arc<Mutex<Session>>, String> {
    let handle = lock_registry()
        .get(session_id)
        .cloned()
        .ok_or_else(|| format!("Unknown command session: {session_id}"))?;
    check_owner(&lock_or_recover(&handle), session_id)?;
    Ok(handle)
}

/// Drop a drained session from the registry. Only removes the entry still
/// pointing at this exact session so a recycled id is never disturbed.
fn remove_if_drained(session_id: &str, handle: &Arc<Mutex<Session>>) {
    // NOTE: each buffer lock lives in its own statement. A tuple like
    // `(stdout.lock()..., has_pending_output())` keeps the first guard alive
    // until the end of the statement and self-deadlocks on the non-reentrant
    // buffer mutex.
    let exited = {
        let session = lock_or_recover(handle);
        session.stdout.lock().unwrap().exited.is_some()
    };
    let drained = !lock_or_recover(handle).has_pending_output();
    if exited && drained {
        let mut sessions = lock_registry();
        if sessions
            .get(session_id)
            .is_some_and(|current| Arc::ptr_eq(current, handle))
        {
            if let Some(entry) = sessions.remove(session_id) {
                let _ = lock_or_recover(&entry).child.wait();
            }
        }
    }
}

/// Poll an existing session. Returns output produced since the previous call.
/// Only the global lookup holds the registry lock; the yield waits under the
/// per-session lock so other sessions stay responsive.
pub(crate) fn poll(session_id: &str, yield_time_ms: u64) -> Result<String, String> {
    reap_idle();
    let handle = lookup_owned(session_id)?;
    let (body, exited, drained, pty) = {
        let mut session = lock_or_recover(&handle);
        let body = session.snapshot(Duration::from_millis(yield_time_ms));
        let exited = session.stdout.lock().unwrap().exited.is_some();
        let drained = !session.has_pending_output();
        (body, exited, drained, session.pty)
    };
    remove_if_drained(session_id, &handle);
    Ok(render_with_id(session_id, &body, exited, drained, pty))
}

/// Write bytes to an existing session, then poll it.
pub(crate) fn write_stdin(
    session_id: &str,
    bytes: &str,
    yield_time_ms: u64,
) -> Result<String, String> {
    if bytes.len() > 16 * 1024 {
        return Err("Command session write_stdin exceeds the 16 KiB limit".to_string());
    }
    reap_idle();
    let handle = lookup_owned(session_id)?;
    let (body, exited, drained, pty) = {
        let mut session = lock_or_recover(&handle);
        let stdin = session
            .stdin
            .as_mut()
            .ok_or_else(|| format!("Command session {session_id} no longer accepts input"))?;
        stdin
            .write_all(bytes.as_bytes())
            .and_then(|_| stdin.flush())
            .map_err(|err| format!("Command session stdin write failed: {err}"))?;
        let body = session.snapshot(Duration::from_millis(yield_time_ms));
        let exited = session.stdout.lock().unwrap().exited.is_some();
        let drained = !session.has_pending_output();
        (body, exited, drained, session.pty)
    };
    remove_if_drained(session_id, &handle);
    Ok(render_with_id(session_id, &body, exited, drained, pty))
}

/// Explicitly terminate and remove a session. Gives a stuck non-PTY process a
/// way out before the 10-minute idle reaper.
pub(crate) fn close(session_id: &str) -> Result<String, String> {
    let handle = lookup_owned(session_id)?;
    {
        let mut sessions = lock_registry();
        if !sessions
            .get(session_id)
            .is_some_and(|current| Arc::ptr_eq(current, &handle))
        {
            return Err(format!("Unknown command session: {session_id}"));
        }
        sessions.remove(session_id);
    }
    lock_or_recover(&handle).terminate();
    Ok(format!("[session {session_id} closed]\n"))
}

/// Kill every live session. Used at process shutdown and by tests.
pub(crate) fn shutdown_all() {
    let removed: Vec<Arc<Mutex<Session>>> = {
        let mut sessions = lock_registry();
        sessions.drain().map(|(_, session)| session).collect()
    };
    for handle in removed {
        lock_or_recover(&handle).terminate();
    }
}

/// Best-effort variant for forced-exit paths that must not block on a lock held
/// by an in-flight tool call. Sessions that lose the `try_lock` race are left
/// to the process group teardown; this is strictly better than not trying.
pub(crate) fn shutdown_all_best_effort() {
    let removed: Vec<Arc<Mutex<Session>>> = match SESSIONS.try_lock() {
        Ok(mut sessions) => sessions.drain().map(|(_, session)| session).collect(),
        Err(std::sync::TryLockError::Poisoned(poison)) => {
            poison.into_inner().drain().map(|(_, session)| session).collect()
        }
        Err(std::sync::TryLockError::WouldBlock) => return,
    };
    for handle in removed {
        match handle.try_lock() {
            Ok(mut session) => session.terminate(),
            Err(std::sync::TryLockError::Poisoned(poison)) => {
                poison.into_inner().terminate();
            }
            Err(std::sync::TryLockError::WouldBlock) => {}
        }
    }
}

fn spawn_reader<R>(mut reader: R, buffer: Arc<Mutex<StreamBuffer>>, readers: Arc<AtomicUsize>)
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match reader.read(&mut chunk) {
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Ok(0) | Err(_) => break,
                Ok(n) => buffer.lock().unwrap().append(&chunk[..n]),
            }
        }
        readers.fetch_sub(1, Ordering::Release);
    });
}

fn reap_idle() {
    // Snapshot handles under the global lock, then evaluate idleness per
    // session without holding it. A session under active use is skipped via
    // `try_lock` and reconsidered on the next reaper pass.
    let live: Vec<(String, Arc<Mutex<Session>>)> = {
        lock_registry()
            .iter()
            .map(|(id, handle)| (id.clone(), Arc::clone(handle)))
            .collect()
    };
    for (id, handle) in live {
        let idle = match handle.try_lock() {
            Ok(session) => Instant::now().duration_since(session.last_used) > SESSION_IDLE_TIMEOUT,
            Err(_) => continue,
        };
        if !idle {
            continue;
        }
        let removed = {
            let mut sessions = lock_registry();
            if sessions
                .get(&id)
                .is_some_and(|current| Arc::ptr_eq(current, &handle))
            {
                sessions.remove(&id)
            } else {
                None
            }
        };
        if let Some(entry) = removed {
            entry
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .terminate();
        }
    }
}

fn render_with_id(id: &str, body: &str, exited: bool, drained: bool, pty: bool) -> String {
    // A failed command's `Exit code:` must be the first line, matching the
    // one-shot runner and every status classifier that consumes it. Session
    // metadata follows instead of pushing the failure marker below a header.
    let mut out = match body.split_once('\n') {
        Some((first, rest)) if first.starts_with("Exit code:") => {
            format!("{first}\n[session {id}]\n{rest}")
        }
        _ => format!("[session {id}]\n{body}"),
    };
    if exited && drained {
        out.push_str("\n[session closed; start a new command to continue]\n");
    } else if exited {
        out.push_str("\n[session exited; more buffered output pending; poll again]\n");
    } else {
        out.push_str(&format!(
            "\n[continue with execute_command {{\"session_id\": \"{id}\", \"pty\": {pty}, \"write_stdin\": \"...\"}} to send input, omit write_stdin to poll, or pass \"close_session\": true to terminate it]\n"
        ));
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Serializes every test that touches the process-wide session registry,
    /// including the service-level round-trip test in `service::command`.
    /// Without this, a concurrent `shutdown_all` wipes another test's live
    /// session and turns registry interference into flakes.
    pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn test_guard() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    fn unique_command(marker: &str) -> String {
        format!("echo {marker}-$$")
    }

    /// Extract the session id from `render_with_id` output (`[session <id>]`).
    fn session_id_of(rendered: &str) -> String {
        rendered
            .lines()
            .find_map(|line| line.strip_prefix("[session "))
            .and_then(|line| line.strip_suffix(']'))
            .expect("session id")
            .to_string()
    }

    /// Poll until `want` appears or the deadline passes. Shell spawn latency
    /// (`bash -lc` sources login profiles) varies wildly across machines, so
    /// timing-sensitive assertions must poll with generous deadlines instead
    /// of assuming one short yield is enough.
    fn poll_until(session_id: &str, want: &str, deadline_ms: u64) -> String {
        let deadline = Instant::now() + Duration::from_millis(deadline_ms);
        let mut combined = String::new();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let out = poll(session_id, remaining.as_millis().min(1_000) as u64).unwrap();
            combined.push_str(&out);
            if combined.contains(want) || Instant::now() >= deadline {
                return combined;
            }
        }
    }

    #[test]
    fn session_yields_partial_output_and_exits_after_stdin() {
        let _guard = test_guard();
        // `read line` produces no output until stdin arrives, so this yield
        // always waits it out; keep it short, the session persists anyway.
        let out = start("read line; echo GOT:$line", None, false, None, 500).unwrap();
        let id = session_id_of(&out);
        assert!(out.contains("command session running"), "out: {out}");
        let out = write_stdin(&id, "hello\n", 5_000).unwrap();
        let mut combined = out;
        if !combined.contains("GOT:hello") {
            combined.push_str(&poll_until(&id, "GOT:hello", 10_000));
        }
        assert!(combined.contains("GOT:hello"), "out: {combined}");
        if !combined.contains("session closed") {
            combined.push_str(&poll_until(&id, "session closed", 10_000));
        }
        assert!(combined.contains("session closed"), "out: {combined}");
    }

    #[test]
    fn session_poll_is_bounded_and_unknown_ids_fail_closed() {
        let _guard = test_guard();
        let out = start(&unique_command("ready"), None, false, None, 10_000).unwrap();
        assert!(out.contains("ready-"), "out: {out}");
        assert!(poll("cmd-does-not-exist", 10).is_err());
    }

    #[test]
    fn exited_session_drains_reader_threads_before_closing() {
        let _guard = test_guard();
        let mut out = start(
            "for i in $(seq 1 40); do printf 'line-%04d-%01000d\\n' \"$i\" 0; done",
            None,
            false,
            None,
            5_000,
        )
        .unwrap();
        let id = out
            .lines()
            .next()
            .and_then(|line| line.strip_prefix("[session "))
            .and_then(|line| line.strip_suffix(']'))
            .expect("session id")
            .to_string();
        for _ in 0..100 {
            if out.contains("session closed") {
                break;
            }
            out.push_str(&poll(&id, 1_000).unwrap());
        }
        assert!(out.contains("line-0040"), "tail output was lost: {out}");
        assert!(out.contains("session closed"), "out: {out}");
    }

    #[test]
    fn session_nonzero_exit_uses_canonical_exit_code_marker() {
        let _guard = test_guard();
        let out = start("false", None, false, None, 2_000).unwrap();
        assert!(
            out.starts_with("Exit code: 1"),
            "session failure must use the shared marker: {out}"
        );
    }

    #[test]
    fn poll_collects_for_the_yield_window_instead_of_returning_on_first_byte() {
        let _guard = test_guard();
        let out = start(
            "printf 'first\\n'; sleep 1; printf 'second\\n'",
            None,
            false,
            None,
            0,
        )
        .unwrap();
        let id = session_id_of(&out);
        let started = Instant::now();
        let body = poll(&id, 300).unwrap();
        let elapsed = started.elapsed();
        shutdown_all();
        assert!(
            elapsed >= Duration::from_millis(250),
            "poll returned before the yield window: {elapsed:?}\n{body}"
        );
        assert!(body.contains("first"), "body: {body}");
        assert!(
            !body.contains("second"),
            "poll must collect until the window, not until process exit: {body}"
        );
    }

    #[test]
    fn idle_session_is_reaped_without_another_tool_call() {
        let _guard = test_guard();
        let out = start("sleep 30", None, false, None, 50).unwrap();
        let id = out
            .lines()
            .next()
            .and_then(|line| line.strip_prefix("[session "))
            .and_then(|line| line.strip_suffix(']'))
            .expect("session id")
            .to_string();
        {
            let cached = lock_registry().get(&id).cloned();
            let handle = cached.expect("session");
            handle.lock().unwrap().last_used =
                Instant::now() - SESSION_IDLE_TIMEOUT - Duration::from_secs(1);
        }
        reap_idle();
        assert!(poll(&id, 10).is_err(), "idle session must be removed");
    }

    #[test]
    fn shutdown_all_kills_live_sessions() {
        let _guard = test_guard();
        let out = start("sleep 30", None, false, None, 50).unwrap();
        let id = session_id_of(&out);
        shutdown_all();
        assert!(poll(&id, 10).is_err(), "shutdown must clear sessions");
    }

    #[cfg(unix)]
    #[test]
    fn terminate_does_not_signal_group_after_child_exit() {
        let _guard = test_guard();
        // A sentinel process group stands in for a recycled pgid. If a session
        // whose child already exited signals its saved group on close, the
        // sentinel dies and the test fails.
        let mut sentinel_cmd = Command::new("sleep");
        sentinel_cmd
            .arg("30")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        unsafe {
            sentinel_cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut sentinel = crate::fork_guard::spawn(&mut sentinel_cmd).expect("sentinel");
        let sentinel_pgid = sentinel.id();

        // Keep more output than one poll can return, so the exited child stays
        // in the registry with pending output.
        let out = start(
            "for i in $(seq 1 2000); do printf 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'; done",
            None,
            false,
            None,
            1,
        )
        .unwrap();
        let id = session_id_of(&out);
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut exited = false;
        while Instant::now() < deadline {
            let Some(handle) = lock_registry().get(&id).cloned() else {
                break;
            };
            let mut session = handle.lock().unwrap();
            if session.child.try_wait().unwrap().is_some() {
                exited = true;
                break;
            }
        }
        assert!(exited, "session child did not exit in time");

        let handle = lock_registry().get(&id).cloned().expect("retained session");
        {
            let mut session = handle.lock().unwrap();
            let original_pgid = session.pgid;
            session.pgid = sentinel_pgid;
            session.terminate();
            session.pgid = original_pgid;
        }
        std::thread::sleep(Duration::from_millis(100));
        let sentinel_alive = sentinel.try_wait().unwrap().is_none();
        unsafe {
            libc::kill(-(sentinel_pgid as libc::pid_t), libc::SIGKILL);
        }
        let _ = sentinel.wait();
        assert!(
            sentinel_alive,
            "terminate must not signal the saved group after the child exited"
        );
    }

    #[test]
    fn session_registers_and_unregisters_its_process_group() {
        let _guard = test_guard();
        let out = start("sleep 30", None, false, None, 50).unwrap();
        let id = session_id_of(&out);
        let pgid = {
            let cached = lock_registry().get(&id).cloned();
            let handle = cached.expect("session");
            let mut session = handle.lock().unwrap();
            session.owner_session_id = "owner-reg-test".to_string();
            super::super::process_registry::register(&session.owner_session_id, session.pgid);
            session.pgid
        };
        assert!(super::super::process_registry::is_registered_pgid(
            "owner-reg-test",
            pgid
        ));
        shutdown_all();
        assert!(!super::super::process_registry::is_registered_pgid(
            "owner-reg-test",
            pgid
        ));
    }

    #[test]
    fn terminate_is_idempotent() {
        let _guard = test_guard();
        let out = start("sleep 30", None, false, None, 50).unwrap();
        let id = session_id_of(&out);
        let cached = lock_registry().get(&id).cloned();
        let handle = cached.expect("session");
        {
            let mut session = handle.lock().unwrap();
            session.terminate();
            let started = Instant::now();
            session.terminate();
            assert!(started.elapsed() < Duration::from_millis(100));
            assert!(session.terminate_attempted);
        }
        {
            lock_registry().remove(&id);
        }
    }

    #[test]
    fn since_clamps_dropped_cursor_without_repeating_the_drop_hint() {
        let mut buffer = StreamBuffer::default();
        buffer.append(&vec![b'x'; MAX_BUFFER_BYTES + 72_000]);
        let total = buffer.total;
        let first_available = buffer.dropped;
        assert!(first_available > 0);

        let (first, next) = buffer.since(0);
        assert!(
            first.contains(&format!(
                "{first_available} bytes of older output permanently dropped"
            ))
        );
        assert_eq!(next, first_available + MAX_RETURN_BYTES as u64);

        let (second, mut cursor) = buffer.since(next);
        assert!(!second.contains("permanently dropped"));
        assert_eq!(cursor, first_available + (MAX_RETURN_BYTES * 2) as u64);

        while cursor < total {
            let (chunk, next) = buffer.since(cursor);
            assert!(!chunk.contains("permanently dropped"));
            assert!(next > cursor, "cursor must advance: {cursor} -> {next}");
            cursor = next;
        }

        let (empty, next) = buffer.since(cursor);
        assert!(empty.is_empty());
        assert_eq!(next, total);
    }

    #[test]
    fn start_with_initial_input_reaches_the_child() {
        let _guard = test_guard();
        let out = start(
            "read line; echo GOT:$line",
            None,
            false,
            Some("initial-hello\n"),
            5_000,
        )
        .expect("session with initial input");
        let id = session_id_of(&out);
        let mut combined = out;
        if !combined.contains("GOT:initial-hello") {
            combined.push_str(&poll_until(&id, "GOT:initial-hello", 10_000));
        }
        assert!(combined.contains("GOT:initial-hello"), "out: {combined}");
    }

    #[test]
    fn wait_bounded_returns_for_a_live_child() {
        let _guard = test_guard();
        let out = start("sleep 30", None, false, None, 50).unwrap();
        let id = session_id_of(&out);
        {
            let cached = lock_registry().get(&id).cloned();
            let handle = cached.expect("session");
            let mut session = handle.lock().unwrap();
            let began = Instant::now();
            session.wait_bounded(Duration::from_millis(30));
            assert!(
                began.elapsed() < Duration::from_millis(500),
                "bounded wait must not block on a live child"
            );
        }
        shutdown_all();
    }

    #[test]
    fn foreign_owner_handle_fails_closed() {
        let _guard = test_guard();
        // NB: fast-exiting commands are never registered (exited + drained on
        // arrival), so use a live session for handle lookups.
        let out = start("sleep 30", None, false, None, 50).unwrap();
        let id = session_id_of(&out);
        {
            let cached = lock_registry().get(&id).cloned();
            let handle = cached.expect("session");
            handle.lock().unwrap().owner_session_id = "some-other-agent-session".to_string();
        }
        // Current (test) agent session is "" — the foreign-owned handle must
        // fail closed exactly like an unknown id.
        assert!(poll(&id, 10).is_err());
        assert!(write_stdin(&id, "x\n", 10).is_err());
        assert!(close(&id).is_err());
        shutdown_all();
    }

    #[test]
    fn close_terminates_and_removes_session() {
        let _guard = test_guard();
        let out = start("sleep 30", None, false, None, 50).unwrap();
        let id = session_id_of(&out);
        let done = close(&id).expect("close should succeed");
        assert!(done.contains("closed"), "out: {done}");
        assert!(poll(&id, 10).is_err(), "closed session must be gone");
        assert!(close(&id).is_err(), "double close must fail closed");
    }
}
