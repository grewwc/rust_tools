use std::{
    cell::RefCell,
    collections::VecDeque,
    fs, io,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    rc::Rc,
    time::Duration,
};

use rusqlite::{Connection, Error as RusqliteError, ErrorCode, OpenFlags};

// SQLite keeps a WAL file at its high-water mark and reuses the space after a
// checkpoint, so a single large write would leave the extra bytes allocated for
// the life of the session. Limit the retained file to roughly one default
// auto-checkpoint cycle (1000 pages of 4 KiB): SQLite truncates a larger WAL on
// the first commit after a checkpoint resets it, without discarding that
// commit's frames.
pub(super) const HISTORY_WAL_SIZE_LIMIT_BYTES: i64 = 4 * 1024 * 1024;

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

/// Upper bound on connections kept by the thread-local write cache. A worker
/// normally touches one session at a time; the extra slots cover short
/// interleavings without letting a long-lived process accumulate descriptors.
const MAX_CACHED_WRITE_CONNS: usize = 4;

thread_local! {
    static CACHED_WRITE_CONNS: RefCell<VecDeque<CachedWriteConn>> =
        const { RefCell::new(VecDeque::new()) };
}

struct CachedWriteConn {
    path: PathBuf,
    identity: FileIdentity,
    conn: Connection,
}

/// Identity that changes only when the path is replaced by a different file.
///
/// The read cache also fingerprints `len`, because a read connection must
/// reconnect when another process checkpoints. A write connection's own
/// commits can change the length, so it only tracks device/inode; otherwise
/// every checkpoint would discard a perfectly usable connection.
#[cfg(unix)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct FileIdentity {
    dev: u64,
    ino: u64,
}

#[cfg(unix)]
fn file_identity(metadata: &fs::Metadata) -> Option<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    Some(FileIdentity {
        dev: metadata.dev(),
        ino: metadata.ino(),
    })
}

#[cfg(not(unix))]
#[derive(Clone, Copy, PartialEq, Eq)]
struct FileIdentity;

#[cfg(not(unix))]
fn file_identity(_metadata: &fs::Metadata) -> Option<FileIdentity> {
    // Without a stable inode, a cached write connection could keep writing to
    // a file that was replaced at the same path. Preserve fresh-connection
    // semantics on those platforms.
    None
}

/// A connection checked out from the thread-local write cache.
///
/// It dereferences to [`Connection`], so call sites keep using the normal
/// rusqlite API. Dropping it returns the connection to the cache when the file
/// is still the same one and no transaction is open; otherwise it is closed.
pub(super) struct HistoryConnection {
    path: PathBuf,
    identity: Option<FileIdentity>,
    conn: Option<Connection>,
    // The cache lives in thread-local storage, so the guard must stay on the
    // thread that created it.
    _not_send: PhantomData<Rc<()>>,
}

impl HistoryConnection {
    /// Close the database instead of returning it to the cache. Replacement
    /// paths (`rename`/Online Backup) require the file to have no open users.
    pub(super) fn close(mut self) {
        self.conn = None;
    }
}

impl Deref for HistoryConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        self.conn
            .as_ref()
            .expect("history connection is present until drop")
    }
}

impl DerefMut for HistoryConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.conn
            .as_mut()
            .expect("history connection is present until drop")
    }
}

impl Drop for HistoryConnection {
    fn drop(&mut self) {
        let Some(conn) = self.conn.take() else {
            return;
        };
        let Some(identity) = self.identity else {
            return;
        };
        // A transaction left open by an early return must not leak into the
        // next user of this connection.
        if !conn.is_autocommit() && conn.execute_batch("ROLLBACK").is_err() {
            return;
        }
        let Ok(metadata) = fs::metadata(&self.path) else {
            return;
        };
        let Some(current_identity) = file_identity(&metadata) else {
            return;
        };
        if current_identity != identity {
            return;
        }
        let path = self.path.clone();
        let _ = CACHED_WRITE_CONNS.try_with(|slot| {
            let Ok(mut slot) = slot.try_borrow_mut() else {
                return;
            };
            // Keep at most one pooled connection per path so a thread cannot
            // pin several descriptors for the same session.
            slot.retain(|entry| entry.path != path);
            if slot.len() >= MAX_CACHED_WRITE_CONNS {
                slot.pop_front();
            }
            slot.push_back(CachedWriteConn {
                path,
                identity,
                conn,
            });
        });
    }
}

fn take_cached_write_conn(path: &Path, identity: FileIdentity) -> Option<Connection> {
    CACHED_WRITE_CONNS
        .try_with(|slot| {
            let Ok(mut slot) = slot.try_borrow_mut() else {
                return None;
            };
            let mut found = None;
            while let Some(index) = slot
                .iter()
                .position(|entry| entry.path == path && entry.identity == identity)
            {
                let entry = slot.remove(index).expect("index came from position");
                if entry.conn.is_autocommit() {
                    found = Some(entry.conn);
                    break;
                }
            }
            // Drop stale or unusable entries for this path.
            slot.retain(|entry| entry.path != path);
            found
        })
        .ok()
        .flatten()
}

/// Close the thread's cached write connection for `path`, if one is cached.
///
/// Callers use this before publishing, replacing, or discarding a database file.
/// A cached handle keeps the `-wal`/`-shm` sidecars on disk and keeps the inode
/// alive after the path is removed; closing it later also unlinks the `-wal`
/// path by name, which by then can belong to a different database at the same
/// path. Closing while the caller still controls the path keeps that harmless.
pub(in crate::ai) fn close_cached_write_conn(path: &Path) {
    let _ = CACHED_WRITE_CONNS.try_with(|slot| {
        let Ok(mut slot) = slot.try_borrow_mut() else {
            return;
        };
        slot.retain(|entry| entry.path != path);
    });
}

pub(super) fn open_history_db(path: &Path) -> Result<HistoryConnection, io::Error> {
    open_history_db_with_busy_timeout(path, Duration::from_secs(5))
}

pub(super) fn open_history_db_with_busy_timeout(
    path: &Path,
    busy_timeout: Duration,
) -> Result<HistoryConnection, io::Error> {
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

fn try_open_history_db(
    path: &Path,
    busy_timeout: Duration,
) -> Result<HistoryConnection, io::Error> {
    let path_owned = path.to_path_buf();
    let identity = fs::metadata(path)
        .ok()
        .and_then(|metadata| file_identity(&metadata));
    let cached = match identity {
        Some(identity) => take_cached_write_conn(path, identity).map(|conn| (conn, identity)),
        None => {
            close_cached_write_conn(path);
            None
        }
    };
    let (conn, identity) = match cached {
        Some((conn, identity)) => (conn, Some(identity)),
        None => {
            let conn = fresh_connection(path, busy_timeout)?;
            let identity = fs::metadata(path)
                .ok()
                .and_then(|metadata| file_identity(&metadata));
            (conn, identity)
        }
    };
    conn.busy_timeout(busy_timeout)
        .map_err(|error| sqlite_error(path, "set busy_timeout", error))?;
    Ok(HistoryConnection {
        path: path_owned,
        identity,
        conn: Some(conn),
        _not_send: PhantomData,
    })
}

fn fresh_connection(path: &Path, busy_timeout: Duration) -> Result<Connection, io::Error> {
    let conn = Connection::open(path).map_err(|error| sqlite_error(path, "open", error))?;
    conn.busy_timeout(busy_timeout)
        .map_err(|error| sqlite_error(path, "set busy_timeout", error))?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(|error| sqlite_error(path, "set journal_mode=WAL", error))?;
    conn.pragma_update(None, "journal_size_limit", HISTORY_WAL_SIZE_LIMIT_BYTES)
        .map_err(|error| sqlite_error(path, "set journal_size_limit", error))?;
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
