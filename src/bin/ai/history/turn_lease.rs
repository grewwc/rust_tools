//! Cross-process turn ownership for one session.
//!
//! A session's SQLite file is writable by every process that opens it, and the
//! serve daemon's per-session mutex only serializes requests inside one daemon.
//! A turn child that outlives its daemon would therefore race a new turn on the
//! same history. This lease is the cross-process authority: the holder keeps an
//! exclusive `flock` for the whole turn, and every other turn fails closed
//! instead of interleaving history rows.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

/// One exclusive cross-process lease backed by a lock file beside the session
/// database. Dropping it (or exiting the process, including a panic) releases
/// the lock. The lock file is never unlinked: removing it while another process
/// is waiting on the same path would let a third opener create a new inode and
/// bypass the waiters.
#[derive(Debug)]
struct FileLease {
    _file: File,
}

impl FileLease {
    fn acquire(
        history_file: &Path,
        suffix: &str,
        state: &str,
        action: &str,
        session_id: &str,
    ) -> io::Result<Self> {
        let Some(parent) = history_file.parent() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "session history path has no parent directory",
            ));
        };
        fs::create_dir_all(parent)?;
        let path = history_file.with_extension(suffix);
        let mut file = open_lock_file(&path)?;
        match try_lock_exclusive(&file) {
            Ok(()) => {
                let _ = write_owner(&mut file);
                Ok(Self { _file: file })
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                let owner = read_owner(&path).unwrap_or_else(|| "unknown".to_string());
                Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!(
                        "session '{session_id}' is already {state} (pid {owner}); refusing to start a concurrent {action}"
                    ),
                ))
            }
            Err(err) => Err(err),
        }
    }
}

/// The exclusive turn lease for one session.
#[derive(Debug)]
pub(in crate::ai) struct TurnLease {
    _lease: FileLease,
}

impl TurnLease {
    /// Acquire `history_file`'s turn lease without waiting.
    ///
    /// The lock file is `<history_file stem>.turn.lock` beside the session
    /// database, so it follows the exact file the turn writes even when the
    /// session id was sanitized on its way to disk. A lease held by another
    /// live process returns `io::ErrorKind::WouldBlock` with the owner pid when
    /// the owner marker can be read.
    pub(in crate::ai) fn acquire(
        history_file: &Path,
        session_id: &str,
    ) -> io::Result<Self> {
        FileLease::acquire(
            history_file,
            "turn.lock",
            "owned by another live turn",
            "turn",
            session_id,
        )
        .map(|lease| Self { _lease: lease })
    }

    /// Acquire the lease for a history file when that file is a shared session
    /// SQLite database. Non-SQLite history files (for example a subagent's
    /// private blob) cannot be opened by a second turn and need no lease.
    pub(in crate::ai) fn acquire_for_history(
        history_file: &Path,
        session_id: &str,
    ) -> io::Result<Option<Self>> {
        if !crate::ai::history::is_sqlite_path(history_file) {
            return Ok(None);
        }
        Self::acquire(history_file, session_id).map(Some)
    }
}

/// The exclusive open lease for one session.
///
/// Unlike [`TurnLease`], this is held for as long as a client has the session
/// open, so a second terminal or browser tab cannot attach at all. A turn child
/// does not take this lease: its parent client already holds it.
#[derive(Debug)]
pub(in crate::ai) struct SessionOpenLease {
    _lease: FileLease,
}

impl SessionOpenLease {
    /// Acquire `history_file`'s open lease without waiting.
    ///
    /// The lock file is `<history_file stem>.open.lock` beside the session
    /// database. A lease held by another live process returns
    /// `io::ErrorKind::WouldBlock` with the owner pid when the owner marker can
    /// be read.
    pub(in crate::ai) fn acquire(
        history_file: &Path,
        session_id: &str,
    ) -> io::Result<Self> {
        FileLease::acquire(
            history_file,
            "open.lock",
            "open in another terminal",
            "session",
            session_id,
        )
        .map(|lease| Self { _lease: lease })
    }

    /// Acquire the open lease only for a shared session SQLite database.
    /// Subagent-private history files are single-writer by construction.
    pub(in crate::ai) fn acquire_for_history(
        history_file: &Path,
        session_id: &str,
    ) -> io::Result<Option<Self>> {
        if !crate::ai::history::is_sqlite_path(history_file) {
            return Ok(None);
        }
        Self::acquire(history_file, session_id).map(Some)
    }
}

#[cfg(unix)]
fn open_lock_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(not(unix))]
fn open_lock_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path)
}

#[cfg(unix)]
fn try_lock_exclusive(file: &File) -> io::Result<()> {
    // SAFETY: `flock` only manipulates the lock attached to this open file
    // description; the descriptor is valid for the lifetime of `file`.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
fn try_lock_exclusive(_file: &File) -> io::Result<()> {
    // Without an OS file lock there is no cross-process exclusion. Refuse the
    // lease instead of silently pretending this turn owns the session.
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "cross-process turn lease requires an OS file lock (unix flock)",
    ))
}

fn write_owner(file: &mut File) -> io::Result<()> {
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    write!(file, "{}", std::process::id())
}

fn read_owner(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|owner| owner.trim().to_string())
        .filter(|owner| !owner.is_empty())
}

/// Read-only probe for a turn lease held by another live process.
///
/// The serve daemon uses this to notice a turn started by a terminal or a
/// background `/bg` session, which never appears in its in-process turn table.
/// The probe never creates the lock file and never writes the owner marker: a
/// free lock is acquired and released immediately by closing this descriptor,
/// and a held lock only reports its owner. Non-Unix targets always report no
/// owner because the lease itself is unsupported there.
pub(in crate::ai) fn turn_lease_owner(history_file: &Path) -> Option<String> {
    if !crate::ai::history::is_sqlite_path(history_file) {
        return None;
    }
    let path = history_file.with_extension("turn.lock");
    let file = open_existing_lock_file(&path).ok()?;
    match try_lock_exclusive(&file) {
        Ok(()) => None,
        Err(err) if err.kind() == io::ErrorKind::WouldBlock => read_owner(&path),
        Err(_) => None,
    }
}

#[cfg(unix)]
fn open_existing_lock_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(not(unix))]
fn open_existing_lock_file(path: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).open(path)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "a-turn-lease-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create temp root");
        root
    }

    #[test]
    fn second_acquire_fails_until_first_is_dropped() {
        let root = temp_root();
        let history = root.join("session-a.sqlite");
        let first = TurnLease::acquire(&history, "session-a").expect("first lease");
        let err = TurnLease::acquire(&history, "session-a").expect_err("second lease must fail");
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
        assert!(err.to_string().contains("already owned"));
        drop(first);
        let second = TurnLease::acquire(&history, "session-a").expect("lease after drop");
        drop(second);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn different_sessions_do_not_block_each_other() {
        let root = temp_root();
        let first = TurnLease::acquire(&root.join("session-a.sqlite"), "session-a")
            .expect("first lease");
        let second = TurnLease::acquire(&root.join("session-b.sqlite"), "session-b")
            .expect("second session lease");
        drop(first);
        drop(second);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn second_open_lease_fails_until_first_is_dropped() {
        let root = temp_root();
        let history = root.join("session-a.sqlite");
        let first = SessionOpenLease::acquire(&history, "session-a").expect("first open lease");
        let err =
            SessionOpenLease::acquire(&history, "session-a").expect_err("second open must fail");
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
        assert!(err.to_string().contains("open in another terminal"));
        drop(first);
        let second = SessionOpenLease::acquire(&history, "session-a").expect("open after drop");
        drop(second);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn turn_and_open_leases_are_independent() {
        let root = temp_root();
        let history = root.join("session-a.sqlite");
        let open = SessionOpenLease::acquire(&history, "session-a").expect("open lease");
        let turn = TurnLease::acquire(&history, "session-a").expect("turn lease beside open");
        drop(turn);
        drop(open);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn probe_reports_a_held_turn_lease_without_touching_it() {
        let root = temp_root();
        let history = root.join("session-a.sqlite");
        fs::write(&history, b"").expect("history file");
        let lock = history.with_extension("turn.lock");
        assert_eq!(turn_lease_owner(&history), None);
        assert!(
            !lock.exists(),
            "an idle probe must not create the lock file"
        );
        let lease = TurnLease::acquire(&history, "session-a").expect("lease");
        assert_eq!(
            turn_lease_owner(&history),
            Some(std::process::id().to_string())
        );
        drop(lease);
        assert_eq!(turn_lease_owner(&history), None);
        fs::remove_dir_all(&root).ok();
    }
}
