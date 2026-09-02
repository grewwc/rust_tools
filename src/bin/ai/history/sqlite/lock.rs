use std::{
    fs,
    fs::OpenOptions,
    io,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::fd::AsRawFd;

use rustc_hash::FxHashMap;

static SESSION_STATE_LOCKS: LazyLock<Mutex<FxHashMap<PathBuf, Arc<Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

pub(super) fn session_state_lock_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("history.sqlite");
    path.with_file_name(format!(".{file_name}.state.lock"))
}

pub(in crate::ai::history) fn delete_session_state_lock(path: &Path) -> io::Result<()> {
    match fs::remove_file(session_state_lock_path(path)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Reclaims a path's per-path lock entry from [`SESSION_STATE_LOCKS`].
///
/// Subagent history files are unique per pid / task_id; if we only deleted the
/// on-disk `.state.lock` file without reclaiming the in-process map entry, the map
/// would grow without bound until process exit after a long-lived main session
/// spawns many subagents. The entry is cleaned up when the subagent lifecycle ends
/// (`delete_subagent_history`); it is removed only when `Arc::strong_count == 1`
/// (no other thread still holds a clone of the lock), so we never yank a lock that
/// `with_session_state_lock` is currently using and break mutual exclusion for
/// that path.
pub(in crate::ai::history) fn remove_session_state_lock_entry(path: &Path) {
    let mut locks = SESSION_STATE_LOCKS.lock().unwrap_or_else(|poison| {
        warn_session_lock_poison(path, "history lock registry");
        poison.into_inner()
    });
    if let Some(existing) = locks.get(path)
        && Arc::strong_count(existing) == 1
    {
        locks.remove(path);
    }
}

/// Serializes rollbacks that replace the entire live SQLite file against the
/// regular canonical writer. The in-process mutex avoids `flock` semantic
/// differences between threads of the same process; the file lock handles
/// cross-process mutual exclusion.
pub(in crate::ai::history) fn with_session_state_lock<T>(
    path: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock = {
        let mut locks = SESSION_STATE_LOCKS.lock().unwrap_or_else(|poison| {
            warn_session_lock_poison(path, "history lock registry");
            poison.into_inner()
        });
        locks
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    };
    let _guard = lock.lock().unwrap_or_else(|poison| {
        warn_session_lock_poison(path, "per-session history state lock");
        poison.into_inner()
    });

    let lock_path = session_state_lock_path(path);
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;
    #[cfg(unix)]
    unsafe {
        if libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let result = operation();
    #[cfg(unix)]
    unsafe {
        let _ = libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN);
    }
    result
}

/// A poisoned lock means a previous holder panicked while holding the in-process
/// history lock. The on-disk cross-process `flock` and the SQLite transaction are
/// still real safety boundaries, so we keep recovering here (preserving the
/// original semantics), but no longer silently: a one-time warning is printed to
/// help investigate the upstream defect that caused the panic.
fn warn_session_lock_poison(path: &Path, which: &str) {
    eprintln!(
        "[Warning] {} was poisoned (a previous holder panicked). \
         Recovering, but the earlier panic should be investigated. path={}",
        which,
        path.display()
    );
}

fn session_state_lock_timeout(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::WouldBlock,
        format!("timed out acquiring history state lock: {}", path.display()),
    )
}

fn wait_for_session_state_lock(path: &Path, deadline: Instant) -> io::Result<()> {
    if Instant::now() >= deadline {
        return Err(session_state_lock_timeout(path));
    }
    std::thread::sleep(
        deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(10)),
    );
    Ok(())
}

/// Same as [`with_session_state_lock`], but bounds both the in-process mutex and
/// the cross-process flock by the caller-supplied deadline, for short-timeout
/// retry paths.
pub(super) fn with_session_state_lock_until<T>(
    path: &Path,
    deadline: Instant,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock = loop {
        match SESSION_STATE_LOCKS.try_lock() {
            Ok(mut locks) => {
                break locks
                    .entry(path.to_path_buf())
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone();
            }
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                warn_session_lock_poison(path, "history lock registry");
                let mut locks = poisoned.into_inner();
                break locks
                    .entry(path.to_path_buf())
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone();
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                wait_for_session_state_lock(path, deadline)?;
            }
        }
    };
    let _guard = loop {
        match lock.try_lock() {
            Ok(guard) => break guard,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                warn_session_lock_poison(path, "per-session history state lock");
                break poisoned.into_inner();
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                wait_for_session_state_lock(path, deadline)?;
            }
        }
    };

    let lock_path = session_state_lock_path(path);
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;
    #[cfg(unix)]
    loop {
        // SAFETY: `lock_file` stays open for the whole critical section; the kernel
        // releases the flock automatically when it is dropped.
        let status = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if status == 0 {
            break;
        }
        let error = io::Error::last_os_error();
        if !matches!(
            error.raw_os_error(),
            Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN
        ) {
            return Err(error);
        }
        wait_for_session_state_lock(path, deadline)?;
    }

    operation()
}
