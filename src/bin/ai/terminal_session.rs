//! Persistent, cwd-independent PTY sessions. Only this module owns the host/client
//! transport; the driver remains the sole owner of conversation state.
//!
//! Replay is an in-memory, 256 KiB sanitized terminal transcript, not a VT screen
//! emulator. It omits terminal queries, OSC/DCS strings and unsafe controls. After
//! truncation, old screen contents may be incomplete; live bytes are never filtered.
//! Completed sessions remain attachable for ten minutes. No output/history files
//! are written. Lock files deliberately persist so flock always names one inode.

#[cfg(unix)]
mod client;
#[cfg(unix)]
mod host;
#[cfg(unix)]
mod registry;
#[cfg(unix)]
mod replay;
#[cfg(unix)]
mod wire;
#[cfg(all(test, unix))]
mod tests;

use super::cli::ParsedCli;
use std::io;
#[cfg(unix)]
use std::{path::PathBuf, sync::OnceLock};

#[cfg(unix)]
struct Worker {
    socket: PathBuf,
    token: String,
    initial_terminal: String,
}
#[cfg(unix)]
static WORKER: OnceLock<Worker> = OnceLock::new();

type EntryResult = Result<Option<i32>, Box<dyn std::error::Error>>;

/// Call synchronously, before CLI parsing, threads, or runtime construction.
/// Internal markers are argv-only: nested commands do not inherit worker identity.
pub(super) fn prepare_entry(args: &mut Vec<String>) -> EntryResult {
    #[cfg(unix)]
    match args.get(1).map(String::as_str) {
        Some("--terminal-host") => {
            if args.len() < 5 {
                return Err("invalid internal terminal host arguments".into());
            }
            let root = registry::root()?;
            let name = registry::bootstrap_name(&args[2])?;
            let token = args[3].clone();
            let worker_args = args[4..].to_vec();
            // This is a fresh executable, never a post-fork callback.
            if unsafe { libc::setsid() } == -1 {
                return Err(io::Error::last_os_error().into());
            }
            return Ok(Some(host::run(root, name, token, worker_args)?));
        }
        Some("--terminal-worker") => {
            if args.len() < 5 {
                return Err("invalid internal terminal worker arguments".into());
            }
            let socket = registry::root()?.join(registry::bootstrap_name(&args[2])?);
            let token = args[3].clone();
            let initial_terminal = args[4].clone();
            if unsafe { libc::setsid() } == -1
                || unsafe { libc::ioctl(libc::STDIN_FILENO, libc::TIOCSCTTY as _, 0) } == -1
            {
                return Err(io::Error::last_os_error().into());
            }
            WORKER.set(Worker { socket, token, initial_terminal })
                .map_err(|_| "terminal worker already initialized")?;
            args.drain(1..5);
        }
        _ => {}
    }
    Ok(None)
}

pub(super) fn is_worker() -> bool {
    #[cfg(unix)]
    { WORKER.get().is_some() }
    #[cfg(not(unix))]
    { false }
}

pub(super) fn maybe_run_client(cli: &ParsedCli, original_args: &[String]) -> EntryResult {
    #[cfg(unix)]
    { client::maybe_run(cli, original_args) }
    #[cfg(not(unix))]
    { let _ = (cli, original_args); Ok(None) }
}

/// Claims the new alias before releasing the previous session alias. Failures
/// leave the old claim intact. This never accesses canonical history storage.
pub(super) fn register_session(session_id: &str) -> io::Result<()> {
    #[cfg(unix)]
    if let Some(worker) = WORKER.get() {
        worker.request(wire::Request::Register {
            token: worker.token.clone(), session: session_id.into(),
        })?;
    }
    let _ = session_id;
    Ok(())
}

/// Disconnects the client only; no worker TTY changes or process signals.
pub(super) fn detach() -> io::Result<()> {
    #[cfg(unix)]
    if let Some(worker) = WORKER.get() {
        worker.request(wire::Request::Detach { token: worker.token.clone() })?;
        return Ok(());
    }
    Err(io::Error::new(io::ErrorKind::NotConnected, "not a persistent terminal worker"))
}

/// Worker-only override. The host updates identity on cross-terminal attachment.
pub(super) fn current_terminal_key() -> Option<String> {
    #[cfg(unix)]
    if let Some(worker) = WORKER.get() {
        return Some(worker.request(wire::Request::Terminal { token: worker.token.clone() })
            .ok().and_then(|reply| reply.terminal)
            .unwrap_or_else(|| worker.initial_terminal.clone()));
    }
    None
}

#[cfg(unix)]
impl Worker {
    fn request(&self, request: wire::Request) -> io::Result<wire::Reply> {
        wire::rpc(&self.socket, request)
    }
}
