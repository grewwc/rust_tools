use std::{
    cell::RefCell,
    fs,
    io,
    path::{Path, PathBuf},
    time::Duration,
};

use rusqlite::{Connection, Error as RusqliteError, ErrorCode, OpenFlags};

// Each worker thread caches its most recently opened history connection so that
// every read/write avoids `Connection::open` + PRAGMA setup, and so that
// `prepare_cached`'s statement cache is genuinely reused across calls.
//
// Safety:
// - `thread_local` guarantees the connection is only accessed from its owning
//   thread; `Connection` is not `Sync` and is therefore never used concurrently.
// - Read paths do not hold the per-session `Mutex` (concurrent writers still write
//   through their own connections), but every read opens a fresh
//   `conn.transaction()` and commits within the same call: WAL's per-transaction
//   snapshot always sees the latest committed state, so a reused connection never
//   reads stale data.
// - The `(dev, ino, len)` fingerprint has a single job: verify before reuse that
//   the underlying file was not renamed/replaced wholesale. File replacement
//   operations (rollback/compact/reset) change the inode, in which case we drop
//   the old connection and reopen, matching the semantics of opening a fresh one.
thread_local! {
    static CACHED_HISTORY_CONN: RefCell<Option<CachedHistoryConn>> = const { RefCell::new(None) };
}

struct CachedHistoryConn {
    path: PathBuf,
    fingerprint: FileFingerprint,
    conn: Connection,
}

#[derive(PartialEq, Eq)]
struct FileFingerprint {
    dev: u64,
    ino: u64,
    len: u64,
}

#[cfg(unix)]
fn file_fingerprint(metadata: &fs::Metadata) -> FileFingerprint {
    use std::os::unix::fs::MetadataExt;
    FileFingerprint {
        dev: metadata.dev(),
        ino: metadata.ino(),
        len: metadata.len(),
    }
}

#[cfg(not(unix))]
fn file_fingerprint(metadata: &fs::Metadata) -> FileFingerprint {
    FileFingerprint {
        dev: 0,
        ino: 0,
        len: metadata.len(),
    }
}
pub(super) fn open_history_db(path: &Path) -> Result<Connection, io::Error> {
    open_history_db_with_busy_timeout(path, Duration::from_secs(5))
}

pub(super) fn open_history_db_with_busy_timeout(
    path: &Path,
    busy_timeout: Duration,
) -> Result<Connection, io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // This function is synchronous and is reused via `?` by async callers (the
    // compaction read/write paths). It must therefore never retry with a blocking
    // sleep — that would stall a tokio worker. Transient I/O failures (such as
    // SQLITE_IOERR_FSTAT when concurrently opening the WAL for the first time) are
    // uniformly reported as `WouldBlock` and left to the async call site's
    // non-blocking retry loop (`tokio::time::sleep`); purely synchronous call sites
    // rely on SQLite's own busy_timeout. Non-transient errors propagate unchanged.
    try_open_history_db(path, busy_timeout)
}

fn try_open_history_db(path: &Path, busy_timeout: Duration) -> Result<Connection, io::Error> {
    let conn = fresh_connection(path, busy_timeout)?;
    Ok(conn)
}

fn fresh_connection(path: &Path, busy_timeout: Duration) -> Result<Connection, io::Error> {
    let conn = Connection::open(path).map_err(|error| sqlite_error(path, "open", error))?;
    conn.busy_timeout(busy_timeout)
        .map_err(|error| sqlite_error(path, "set busy_timeout", error))?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(|error| sqlite_error(path, "set journal_mode=WAL", error))?;
    Ok(conn)
}

/// Runs a read-only operation on a cached history connection. Connections are
/// cached per thread and automatically reconnected when the underlying file's
/// inode/size changes (rollback/compact/reset and other replacement operations),
/// giving the same semantics as opening a fresh connection each time while saving
/// the per-turn open + PRAGMA and statement recompilation costs.
///
/// Note: borrowed data must not escape the closure (`Connection` is not `Sync`
/// and is owned by the thread_local). Intended only for hot-path read-only
/// queries.
pub(super) fn with_cached_read_conn<T>(
    path: &Path,
    busy_timeout: Duration,
    op: impl FnOnce(&mut Connection) -> io::Result<T>,
) -> Result<T, io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let metadata = fs::metadata(path)
        .map_err(|error| io::Error::other(format!("metadata {}: {error}", path.display())))?;
    let fingerprint = file_fingerprint(&metadata);
    let path_owned = path.to_path_buf();

    CACHED_HISTORY_CONN.with(|slot| {
        let mut slot = slot.borrow_mut();
        let need_reconnect = match slot.as_ref() {
            Some(cached) => cached.path != path_owned || cached.fingerprint != fingerprint,
            None => true,
        };
        if need_reconnect {
            let conn = fresh_connection(&path_owned, busy_timeout)?;
            *slot = Some(CachedHistoryConn {
                path: path_owned,
                fingerprint,
                conn,
            });
        }
        let cached = slot.as_mut().expect("cached connection just initialized");
        op(&mut cached.conn)
    })
}

pub(super) fn sqlite_error_kind(error: &RusqliteError) -> io::ErrorKind {
    match error {
        RusqliteError::SqliteFailure(inner, _)
            if matches!(
                inner.code,
                ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked | ErrorCode::SystemIoFailure
            ) =>
        {
            io::ErrorKind::WouldBlock
        }
        _ => io::ErrorKind::Other,
    }
}

pub(super) fn sqlite_error(path: &Path, operation: &str, error: RusqliteError) -> io::Error {
    let kind = sqlite_error_kind(&error);
    let detail = match &error {
        RusqliteError::SqliteFailure(inner, message) => {
            let message = message
                .as_deref()
                .map(|message| format!("; message={message}"))
                .unwrap_or_default();
            format!(
                "SQLite {operation} failed for {}: {error}; code={:?}; extended_code={}{}",
                path.display(),
                inner.code,
                inner.extended_code,
                message
            )
        }
        _ => format!("SQLite {operation} failed for {}: {error}", path.display()),
    };
    io::Error::new(kind, detail)
}

/// Read-only metadata for the session list; enumerating must not create
/// directories, initialize the database, or switch the journal mode.
pub(super) fn open_history_db_read_only(path: &Path) -> Result<Connection, io::Error> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| io::Error::other(error.to_string()))?;
    conn.busy_timeout(Duration::from_secs(5))
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(conn)
}
