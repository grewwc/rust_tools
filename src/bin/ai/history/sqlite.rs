use std::{
    cell::RefCell,
    fs,
    fs::OpenOptions,
    io,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::fd::AsRawFd;

use rusqlite::{
    Connection, Error as RusqliteError, ErrorCode, OpenFlags, OptionalExtension,
    TransactionBehavior, params,
};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::types::ToolCall;

use super::{
    blob,
    compress::{self, COMPRESSED_TOOL_EVIDENCE_MARKER, is_summary_note_text, value_to_string},
    types::{Message, ROLE_INTERNAL_NOTE, SkillActivationEvent, ToolExecutionOutcome},
};

const STALE_PATCH_TARGETS_META_KEY: &str = "stale_patch_targets_v1";
const LLM_PRUNE_MARKS_META_KEY: &str = "llm_prune_marks_v1";
const LAST_ACTIVITY_META_KEY: &str = "last_activity_unix_ms";
const SESSION_MARKED_META_KEY: &str = "session_marked";

static SESSION_STATE_LOCKS: LazyLock<Mutex<FxHashMap<PathBuf, Arc<Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

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

/// In-process cache of `history_revision`, invalidated by file metadata
/// fingerprints (main DB len/mtime, WAL sidecar len/mtime).
///
/// `read_history_revision` runs `Connection::open` + an SQL query on every call
/// (per-turn context cache validation + session list refresh call it several
/// times), yet the revision only changes when messages are written. We therefore
/// cache the result keyed by file metadata fingerprints: any write from this
/// process or an external one changes the len/mtime of the main DB or the WAL
/// sidecar, so a fingerprint mismatch triggers a re-query, with semantics
/// identical to having no cache.
///
/// **WAL is critical**: the history DB runs in WAL mode, so commits only touch the
/// `-wal` sidecar and the main DB len/mtime can stay unchanged before a checkpoint.
/// If the fingerprint covered only the main DB, the cache would keep returning the
/// old revision while a live connection exists, breaking `history_revision`'s role
/// as a cross-connection signal. The fingerprint therefore covers both the main DB
/// and the `-wal` sidecar; a change in either invalidates it.
static HISTORY_REVISION_CACHE: LazyLock<
    Mutex<FxHashMap<PathBuf, (((u64, Option<SystemTime>), (u64, Option<SystemTime>)), i64)>>,
> = LazyLock::new(|| Mutex::new(FxHashMap::default()));

fn history_file_fingerprint(path: &Path) -> ((u64, Option<SystemTime>), (u64, Option<SystemTime>)) {
    let main = std::fs::metadata(path)
        .map(|m| (m.len(), m.modified().ok()))
        .unwrap_or((0, None));
    // In WAL mode commits land in the `-wal` sidecar first and the main DB mtime
    // does not change before a checkpoint; including the sidecar metadata in the
    // fingerprint ensures uncheckpointed writes still invalidate the cache.
    let wal_path = PathBuf::from(format!("{}-wal", path.display()));
    let wal = std::fs::metadata(&wal_path)
        .map(|m| (m.len(), m.modified().ok()))
        .unwrap_or((0, None));
    (main, wal)
}

fn session_state_lock_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("history.sqlite");
    path.with_file_name(format!(".{file_name}.state.lock"))
}

pub(super) fn delete_session_state_lock(path: &Path) -> io::Result<()> {
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
pub(super) fn remove_session_state_lock_entry(path: &Path) {
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
pub(super) fn with_session_state_lock<T>(
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
fn with_session_state_lock_until<T>(
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

/// The rebuildable context consumed by the model's actual request. `messages` is
/// always the one canonical record of the session; the messages here are only a
/// compression snapshot, and adding the raw messages after `source_message_id`
/// rebuilds the current context. `canonical_generation` rejects stale snapshots
/// produced by a concurrent rewind/clear.
pub(in crate::ai) struct ContextHistory {
    pub(in crate::ai) messages: Vec<Message>,
    pub(in crate::ai) source_message_id: i64,
    pub(in crate::ai) canonical_generation: i64,
    pub(in crate::ai) snapshot_is_current: bool,
}

pub(in crate::ai) struct RecentTurnWindow {
    pub(in crate::ai) messages: Vec<Message>,
    pub(in crate::ai) start_message_id: Option<i64>,
    pub(in crate::ai) has_older_messages: bool,
}

/// Lightweight metadata needed for the `/ss` list display.
pub(in crate::ai) struct SessionListMetadata {
    pub(in crate::ai) first_user_prompt: Option<String>,
    pub(in crate::ai) session_title: Option<String>,
    pub(in crate::ai) last_activity_unix_ms: Option<i64>,
    pub(in crate::ai) history_revision: i64,
    /// Whether the user marked this session as important via `/mark`.
    pub(in crate::ai) marked: bool,
}

fn open_history_db(path: &Path) -> Result<Connection, io::Error> {
    open_history_db_with_busy_timeout(path, Duration::from_secs(5))
}

fn open_history_db_with_busy_timeout(
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
fn with_cached_read_conn<T>(
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

fn sqlite_error_kind(error: &RusqliteError) -> io::ErrorKind {
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

fn sqlite_error(path: &Path, operation: &str, error: RusqliteError) -> io::Error {
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
fn open_history_db_read_only(path: &Path) -> Result<Connection, io::Error> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| io::Error::other(error.to_string()))?;
    conn.busy_timeout(Duration::from_secs(5))
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(conn)
}

fn init_history_schema(conn: &Connection) -> Result<(), io::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT,
            tool_call_id TEXT,
            reasoning_content TEXT,
            source_model TEXT,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
        CREATE TABLE IF NOT EXISTS context_messages (
            position INTEGER PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT,
            tool_call_id TEXT,
            reasoning_content TEXT
        );
        CREATE TABLE IF NOT EXISTS context_snapshot (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            source_message_id INTEGER NOT NULL,
            source_generation INTEGER NOT NULL,
            projection_fingerprint TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE IF NOT EXISTS image_digests (
            message_key TEXT PRIMARY KEY,
            digest TEXT NOT NULL,
            image_paths TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE IF NOT EXISTS tool_execution_outcomes (
            tool_call_id TEXT PRIMARY KEY,
            execution_signature TEXT NOT NULL,
            succeeded INTEGER NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE IF NOT EXISTS interrupted_stream_diagnostics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assistant_text TEXT NOT NULL,
            reasoning_text TEXT NOT NULL,
            source_model TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE IF NOT EXISTS skill_activation_events (
            id INTEGER PRIMARY KEY,
            requested_skill TEXT NOT NULL,
            injected_skill TEXT,
            source TEXT NOT NULL,
            outcome TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );",
    )
    .map_err(|error| io::Error::new(sqlite_error_kind(&error), error.to_string()))?;
    add_column_if_missing(conn, "messages", "tool_calls", "TEXT")?;
    add_column_if_missing(conn, "messages", "tool_call_id", "TEXT")?;
    add_column_if_missing(conn, "messages", "reasoning_content", "TEXT")?;
    add_column_if_missing(conn, "messages", "source_model", "TEXT")?;
    // An old snapshot cannot prove it matches the current projection policy; an
    // empty fingerprint lets the read path safely ignore it.
    add_column_if_missing(
        conn,
        "context_snapshot",
        "projection_fingerprint",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    Ok(())
}

fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<(), io::Error> {
    let sql = format!("ALTER TABLE {table} ADD COLUMN {column} {definition}");
    match conn.execute(&sql, []) {
        Ok(_) => Ok(()),
        Err(err) if err.to_string().contains("duplicate column name") => Ok(()),
        Err(error) => Err(io::Error::new(sqlite_error_kind(&error), error.to_string())),
    }
}

/// Reads the history DB's write version (stored in the meta table under
/// key='history_revision'). Every message write/delete/replace increments it via
/// `bump_history_revision` in the same transaction, so it is a **cross-connection**
/// monotonically increasing global signal that reliably reflects "did the DB
/// content change".
///
/// `PRAGMA data_version` cannot replace it: it is a **connection-local** comparison
/// value — each freshly opened `Connection` treats it only as "has the DB been
/// changed by another connection since I opened", and a new connection's initial
/// reading does not vary with external writes (a new connection consistently
/// returns 2 in practice), so it cannot serve as a cross-connection cache
/// invalidation signal. When the key is missing (an old DB that has never had a
/// revision written), 0 is returned, consistent with "never modified".
pub(in crate::ai) fn read_history_revision(path: &Path) -> Option<i64> {
    // In-process cache: if the fingerprint (main DB len/mtime, WAL sidecar
    // len/mtime) is unchanged, reuse the previous result instead of opening a
    // connection + SQL query on every call (context cache validation per turn /
    // session list refresh call this frequently).
    let fingerprint = history_file_fingerprint(path);
    if let Ok(cache) = HISTORY_REVISION_CACHE.lock() {
        if let Some((cached_fp, rev)) = cache.get(path) {
            if *cached_fp == fingerprint {
                return Some(*rev);
            }
        }
    }
    let conn = Connection::open(path).ok()?;
    // The meta table may not exist yet (brand-new DB) or may never have had a
    // revision written: both cases are treated as 0 ("never modified") so the
    // return value stays stable and comparable. Only an unopenable connection
    // yields None.
    let value: Option<i64> = conn
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key='history_revision' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    let revision = value.unwrap_or(0);
    if let Ok(mut cache) = HISTORY_REVISION_CACHE.lock() {
        cache.insert(path.to_path_buf(), (fingerprint, revision));
    }
    Some(revision)
}

/// Removes the revision cache entry for the given history path.
/// Called when a history file is deleted or renamed, so the cache does not grow
/// without bound with every unique session/subagent path.
pub(in crate::ai) fn remove_history_revision_cache_entry(path: &Path) {
    if let Ok(mut cache) = HISTORY_REVISION_CACHE.lock() {
        cache.remove(path);
    }
}

#[cfg(test)]
pub(in crate::ai) fn history_revision_cache_contains(path: &Path) -> bool {
    HISTORY_REVISION_CACHE
        .lock()
        .is_ok_and(|cache| cache.contains_key(path))
}

/// Records the most recent message write time; consecutive writes within the same
/// millisecond stay monotonically increasing.
fn touch_session_activity(conn: &Connection) -> io::Result<()> {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64;
    let previous = read_i64_meta_from_conn(conn, LAST_ACTIVITY_META_KEY).unwrap_or(0);
    let next = now_ms.max(previous.saturating_add(1));
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value, created_at)
         VALUES (?1, ?2, unixepoch())",
        params![LAST_ACTIVITY_META_KEY, next.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

/// Increments the history version inside a write transaction and refreshes the
/// logical activity time in sync.
fn bump_history_revision(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('history_revision', '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
        [],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    touch_session_activity(conn)
}

fn history_generation(conn: &Connection) -> io::Result<i64> {
    conn.query_row(
        "SELECT COALESCE(
            (SELECT CAST(value AS INTEGER) FROM meta WHERE key='history_generation'),
            0
         )",
        [],
        |row| row.get(0),
    )
    .map_err(|error| io::Error::other(error.to_string()))
}

/// Any operation that rewrites the canonical messages must invalidate derived
/// snapshots.
fn invalidate_context_snapshot(conn: &Connection) -> io::Result<()> {
    conn.execute("DELETE FROM context_messages", [])
        .map_err(|error| io::Error::other(error.to_string()))?;
    conn.execute("DELETE FROM context_snapshot", [])
        .map_err(|error| io::Error::other(error.to_string()))?;
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('history_generation', '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
        [],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

/// Atomically reserves a session-global turn sequence number.
///
/// The number is stored in SQLite metadata rather than process memory, so restarts
/// and multiple processes recovering the same session never produce duplicates.
/// For an existing session the first allocation continues from the persisted
/// user-turn count, matching the earlier `turn_index` semantics.
pub(in crate::ai) fn reserve_turn_index_sqlite(path: &Path) -> io::Result<usize> {
    with_session_state_lock(path, || reserve_turn_index_sqlite_unlocked(path))
}

fn reserve_turn_index_sqlite_unlocked(path: &Path) -> io::Result<usize> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|e| io::Error::other(e.to_string()))?;

    let current = tx
        .query_row("SELECT value FROM meta WHERE key = 'turn_seq'", [], |row| {
            row.get::<_, String>(0)
        })
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or_else(|| {
            tx.query_row(
                "SELECT COUNT(*) FROM messages WHERE role = 'user'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0)
        })
        .max(0);
    let next = current
        .checked_add(1)
        .ok_or_else(|| io::Error::other("turn sequence overflow"))?;
    tx.execute(
        "INSERT INTO meta(key, value) VALUES('turn_seq', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![next.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    touch_session_activity(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))?;
    usize::try_from(current).map_err(io::Error::other)
}

/// Reads the live DB's three monotonic counters before a rollback:
/// `history_generation`, `history_revision`, `turn_seq`. A rollback uses
/// `backup_sqlite` to overwrite the live path with the whole checkpoint DB, which
/// would restore all three counters to their old checkpoint-time values and break
/// the "monotonic across rollbacks" invariant. This function reads the live values
/// before the overwrite so that
/// `rebase_metadata_after_rollback` raises them back above the live values after the overwrite.
/// When the database is missing or the meta row is absent, each counter reads 0, matching the “never modified” baseline.
pub(in crate::ai) struct LiveRollbackMetadata {
    pub(in crate::ai) generation: i64,
    pub(in crate::ai) revision: i64,
    pub(in crate::ai) turn_seq: i64,
}

impl LiveRollbackMetadata {
    fn zero() -> Self {
        Self {
            generation: 0,
            revision: 0,
            turn_seq: 0,
        }
    }
}

pub(in crate::ai) fn read_live_rollback_metadata(path: &Path) -> io::Result<LiveRollbackMetadata> {
    // `Connection::open` would create the file; here we only need to read an existing live DB, falling back to the 0 baseline when it is missing.
    if !path.exists() {
        return Ok(LiveRollbackMetadata::zero());
    }
    let conn = Connection::open(path).map_err(|e| io::Error::other(e.to_string()))?;
    conn.busy_timeout(Duration::from_secs(5))
        .map_err(|e| io::Error::other(e.to_string()))?;
    let meta_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
            [],
            |_| Ok(()),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .is_some();
    if !meta_exists {
        return Ok(LiveRollbackMetadata::zero());
    }
    let read = |key: &str| -> io::Result<i64> {
        let value = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1 LIMIT 1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| io::Error::other(e.to_string()))?;
        value.map_or(Ok(0), |value| {
            value.parse::<i64>().map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid SQLite meta value for {key}: {error}"),
                )
            })
        })
    };
    Ok(LiveRollbackMetadata {
        generation: read("history_generation")?,
        revision: read("history_revision")?,
        turn_seq: read("turn_seq")?,
    })
}

/// Called after `backup_sqlite` overwrites the live DB with a checkpoint: clear the derived snapshots
/// (canonical has rolled back to the checkpoint, so the old snapshots no longer match), and raise the three monotonic counters
/// above their pre-rollback live values, guaranteeing:
/// - `history_generation` is strictly greater than the live value -> any
///     stale snapshot write still holding an old generation is rejected by fencing (the
///     generation comparison in `write_context_snapshot_sqlite` returns false).
/// - `turn_seq` is not lower than the live value -> turn numbers allocated after the rollback never reuse already-used numbers.
/// - `history_revision` is strictly greater than the live value -> cross-connection file caches observe the change and reload.
/// Everything commits in a single Immediate transaction, avoiding an inconsistent window of
/// “rolled back but counters not raised” after the overwrite.
pub(in crate::ai) fn rebase_metadata_after_rollback(
    path: &Path,
    live_generation: i64,
    live_revision: i64,
    live_turn_seq: i64,
) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|e| io::Error::other(e.to_string()))?;
    // canonical has rolled back, so the old derived snapshots (possibly tagged with the generation at checkpoint time)
    // no longer match; clear them to force the next read to recompute from canonical.
    tx.execute("DELETE FROM context_messages", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM context_snapshot", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    // Each counter takes MAX(checkpoint value, live value) then +1 (generation/revision must strictly
    // increase; turn_seq can simply take the live value, since it is the “next number to allocate”).
    let bump_gen = live_generation.saturating_add(1).max(0);
    let bump_rev = live_revision.saturating_add(1).max(0);
    let bump_turn = live_turn_seq.max(0);
    tx.execute(
        "INSERT INTO meta (key, value) VALUES ('history_generation', ?1)
         ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
        params![bump_gen.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute(
        "INSERT INTO meta (key, value) VALUES ('turn_seq', ?1)
         ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
        params![bump_turn.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute(
        "INSERT INTO meta (key, value) VALUES ('history_revision', ?1)
         ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
        params![bump_rev.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    touch_session_activity(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

/// Prepare a rebased temporary DB inside the session state lock, then publish it to the live path
/// via Online Backup's atomic rename. A crash before the final publish leaves the live DB unchanged; after the publish, metadata
/// is fully raised, so there is no intermediate “DB rolled back but counters still stale” state.
pub(in crate::ai) fn restore_sqlite_after_rollback(
    checkpoint: &Path,
    live_path: &Path,
) -> io::Result<()> {
    with_session_state_lock(live_path, || {
        let live = read_live_rollback_metadata(live_path)?;
        let parent = live_path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let working = parent.join(format!(".rollback-rebased-{}.sqlite", uuid::Uuid::new_v4()));

        let result = (|| {
            backup_sqlite(checkpoint, &working)?;
            rebase_metadata_after_rollback(
                &working,
                live.generation,
                live.revision,
                live.turn_seq,
            )?;
            // The second backup materializes the working WAL into the final temporary main DB, then
            // publishes it with a single rename, so the rebase transaction is not dropped by moving only the main file.
            backup_sqlite(&working, live_path)
        })();
        let _ = fs::remove_file(&working);
        let _ = remove_sqlite_sidecars(&working);
        result
    })
}

/// Create a consistent snapshot with the SQLite Online Backup API and write it to the target via atomic replacement.
/// Copying the WAL main file directly would miss pages not yet checkpointed; the backup API reads both the main DB and the WAL
/// from the same SQLite snapshot. After the main DB replacement succeeds, the old sidecar files are removed so a stale WAL/SHM
/// cannot mix with the new main DB.
pub(in crate::ai) fn backup_sqlite(source: &Path, target: &Path) -> io::Result<()> {
    if !source.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("SQLite source does not exist: {}", source.display()),
        ));
    }
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("history.sqlite");
    let temporary = parent.join(format!(".{file_name}.backup-{}.tmp", uuid::Uuid::new_v4()));

    let result = (|| {
        let source_conn = Connection::open(source).map_err(|e| io::Error::other(e.to_string()))?;
        source_conn
            .backup(rusqlite::MAIN_DB, &temporary, None)
            .map_err(|e| io::Error::other(e.to_string()))?;
        drop(source_conn);

        fs::rename(&temporary, target)?;
        remove_sqlite_sidecars(target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
        let _ = remove_sqlite_sidecars(&temporary);
    }
    result
}

/// Copy the parent session's history DB to a standalone per-process file for a sub-agent.
///
/// The `inherit.history` semantics are “the sub-agent may read the parent session's context, but writes must not flow back into the parent DB”.
/// Reusing the parent DB's canonical file directly would let the sub-agent's internal prompt/tool traces pollute the parent session
/// and interleave writes to the same session across concurrent sub-agents. Here we use SQLite Online Backup to make one
/// consistent snapshot copy; the sub-agent only reads and writes its own fork file afterward, keeping the parent DB isolated.
///
/// When the parent session has no history file yet (first session), publish a brand-new empty DB instead of reusing a leftover child DB.
pub(in crate::ai) fn fork_history_for_subagent(parent: &Path, child: &Path) -> io::Result<()> {
    with_session_state_lock(child, || match fs::metadata(parent) {
        Ok(_metadata) => {
            // Ensure the child file's parent directory exists, or the temporary file for `backup_sqlite` would fail to be created.
            if let Some(dir) = child.parent() {
                fs::create_dir_all(dir)?;
            }
            backup_sqlite(parent, child)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // The parent session has no history file, so the sub-agent starts from a blank history.
            reset_history_for_subagent_unlocked(child)
        }
        Err(error) => Err(error),
    })
}

/// When dispatching a sub-agent for the first time without history inheritance, publish a brand-new empty DB instead of reusing a leftover DB for the same pid.
pub(in crate::ai) fn reset_history_for_subagent(child: &Path) -> io::Result<()> {
    with_session_state_lock(child, || reset_history_for_subagent_unlocked(child))
}

fn reset_history_for_subagent_unlocked(child: &Path) -> io::Result<()> {
    let temporary = child.with_extension(format!(
        "sqlite.empty-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    if let Some(parent) = temporary.parent() {
        fs::create_dir_all(parent)?;
    }
    let result = (|| {
        let conn = open_history_db(&temporary)?;
        init_history_schema(&conn)?;
        drop(conn);
        backup_sqlite(&temporary, child)
    })();
    let _ = fs::remove_file(&temporary);
    let _ = remove_sqlite_sidecars(&temporary);
    result
}

fn remove_sqlite_sidecars(path: &Path) -> io::Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = PathBuf::from(format!("{}{}", path.display(), suffix));
        match fs::remove_file(sidecar) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Cheaply count the number of messages with role='user' in the current history DB.
/// This lets boundary compact “count first, then decide whether to do a full read” on the hot path,
/// avoiding deserializing tens of thousands of messages (including large tool outputs) at the end of every turn.
pub(in crate::ai) fn count_user_turns_sqlite(path: &Path) -> io::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let conn = open_history_db(path)?;
    // The schema may not exist yet (brand-new session); return 0 directly when the messages table is missing.
    let table_exists: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(0);
    }
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(1) FROM messages WHERE role = 'user'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(count.max(0) as usize)
}

/// Cheaply measure the payload size of persisted messages, so history-to-disk compaction still triggers when the user-turn count is low
/// but tool output has grown large. The sqlite file size cannot be used: WAL/free pages are not
/// reclaimed right after messages are replaced, which would make every turn misjudge the budget as exceeded.
pub(in crate::ai) fn total_message_chars_sqlite(path: &Path) -> io::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let conn = open_history_db(path)?;
    let table_exists: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(0);
    }
    let total: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(length(content) + COALESCE(length(tool_calls), 0) + COALESCE(length(reasoning_content), 0)), 0) FROM messages",
            [],
            |row| row.get(0),
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(total.max(0) as usize)
}

/// Cheaply measure the size of old tool evidence already folded into internal_notes. It has an inline cap independent of the global history
/// budget, so individual evidence items cannot keep accumulating under few user turns before the total budget is hit.
pub(in crate::ai) fn compressed_tool_evidence_chars_sqlite(path: &Path) -> io::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let conn = open_history_db(path)?;
    let table_exists: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(0);
    }
    let total: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(length(content)), 0)
             FROM messages
             WHERE role = ?1 AND instr(content, ?2) > 0",
            params![ROLE_INTERNAL_NOTE, COMPRESSED_TOOL_EVIDENCE_MARKER],
            |row| row.get(0),
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(total.max(0) as usize)
}

/// Persist a structured success/failure and execution signature for every real tool call. The tool result body still lives only in
/// `messages`, so the request projection can fold resolved errors while the human history keeps the original diagnostics.
pub(in crate::ai) fn append_tool_execution_outcomes_sqlite(
    path: &Path,
    outcomes: &[ToolExecutionOutcome],
) -> io::Result<()> {
    if outcomes.is_empty() || !blob::is_sqlite_path(path) {
        return Ok(());
    }
    with_session_state_lock(path, || {
        let mut conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction()
            .map_err(|error| io::Error::other(error.to_string()))?;
        {
            let mut statement = tx
                .prepare(
                    "INSERT INTO tool_execution_outcomes
                        (tool_call_id, execution_signature, succeeded)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(tool_call_id) DO NOTHING",
                )
                .map_err(|error| io::Error::other(error.to_string()))?;
            for outcome in outcomes {
                statement
                    .execute(params![
                        outcome.tool_call_id,
                        outcome.execution_signature,
                        outcome.succeeded
                    ])
                    .map_err(|error| io::Error::other(error.to_string()))?;
            }
        }
        bump_history_revision(&tx)?;
        tx.commit()
            .map_err(|error| io::Error::other(error.to_string()))
    })
}

/// Stores received partial model output after an interrupted stream in an audit-only
/// side table. It is deliberately separate from `messages` and `context_messages`,
/// so the output can never be included in a later model request.
pub(in crate::ai) fn append_interrupted_stream_diagnostic_sqlite(
    path: &Path,
    source_model: &str,
    assistant_text: &str,
    reasoning_text: &str,
) -> io::Result<()> {
    if (assistant_text.is_empty() && reasoning_text.is_empty()) || !blob::is_sqlite_path(path) {
        return Ok(());
    }
    with_session_state_lock(path, || {
        let mut conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction()
            .map_err(|error| io::Error::other(error.to_string()))?;
        tx.execute(
            "INSERT INTO interrupted_stream_diagnostics
                (assistant_text, reasoning_text, source_model)
             VALUES (?1, ?2, ?3)",
            params![assistant_text, reasoning_text, source_model],
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        bump_history_revision(&tx)?;
        tx.commit()
            .map_err(|error| io::Error::other(error.to_string()))
    })
}

/// Read the structured tool results needed by the request projection. Older sessions without the side table safely degrade to an empty set,
/// never guessing success/failure from natural language in the history body.
pub(in crate::ai) fn read_tool_execution_outcomes_sqlite(
    path: &Path,
) -> io::Result<Vec<ToolExecutionOutcome>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(Vec::new());
    }
    let conn = open_history_db(path)?;
    let table_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master
             WHERE type='table' AND name='tool_execution_outcomes'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(Vec::new());
    }
    let mut statement = conn
        .prepare(
            "SELECT tool_call_id, execution_signature, succeeded
             FROM tool_execution_outcomes ORDER BY created_at ASC, rowid ASC",
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
    let rows = statement
        .query_map([], |row| {
            Ok(ToolExecutionOutcome {
                tool_call_id: row.get(0)?,
                execution_signature: row.get(1)?,
                succeeded: row.get::<_, i64>(2)? != 0,
            })
        })
        .map_err(|error| io::Error::other(error.to_string()))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| io::Error::other(error.to_string()))
}

/// Persist the actual injection result of explicit skill selection. The raw record is a diagnostic side channel and never pollutes canonical
/// messages; at runtime, bounded historical facts can be derived from successful records.
pub(in crate::ai) fn append_skill_activation_event_sqlite(
    path: &Path,
    event: &SkillActivationEvent,
) -> io::Result<()> {
    if !blob::is_sqlite_path(path) {
        return Ok(());
    }
    with_session_state_lock(path, || {
        let mut conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction()
            .map_err(|error| io::Error::other(error.to_string()))?;
        tx.execute(
            "INSERT INTO skill_activation_events
                (requested_skill, injected_skill, source, outcome)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                event.requested_skill,
                event.injected_skill,
                event.source,
                event.outcome,
            ],
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        bump_history_revision(&tx)?;
        tx.commit()
            .map_err(|error| io::Error::other(error.to_string()))
    })
}

/// Read the audit records of explicit skill injection within a session. Older sessions without the side table safely degrade to empty.
pub(in crate::ai) fn read_skill_activation_events_sqlite(
    path: &Path,
) -> io::Result<Vec<SkillActivationEvent>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(Vec::new());
    }
    let conn = open_history_db(path)?;
    let table_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master
             WHERE type='table' AND name='skill_activation_events'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(Vec::new());
    }
    let mut statement = conn
        .prepare(
            "SELECT requested_skill, injected_skill, source, outcome
             FROM skill_activation_events ORDER BY created_at ASC, rowid ASC",
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
    let rows = statement
        .query_map([], |row| {
            Ok(SkillActivationEvent {
                requested_skill: row.get(0)?,
                injected_skill: row.get(1)?,
                source: row.get(2)?,
                outcome: row.get(3)?,
            })
        })
        .map_err(|error| io::Error::other(error.to_string()))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| io::Error::other(error.to_string()))
}

/// Read the association IDs used by persisted tool messages. The live context may have pruned the older messages,
/// but generating a new occurrence ID must still avoid these IDs from the full history.
pub(in crate::ai) fn read_tool_message_ids_sqlite(path: &Path) -> io::Result<Vec<String>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(Vec::new());
    }
    let conn = open_history_db(path)?;
    let mut statement = conn
        .prepare(
            "SELECT DISTINCT tool_call_id FROM messages
             WHERE role = 'tool' AND tool_call_id IS NOT NULL",
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
    let rows = statement
        .query_map([], |row| row.get(0))
        .map_err(|error| io::Error::other(error.to_string()))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| io::Error::other(error.to_string()))
}

/// Read the current session's stale-patch ledger. `None` means the database is old and has never written this state,
/// so the caller should replay once from the still-visible structured messages and write it back; `Some(empty)` means it is known to be empty,
/// so history that may contain old failure records must not be scanned again.
pub(in crate::ai) fn read_stale_patch_targets_sqlite(
    path: &Path,
) -> io::Result<Option<FxHashSet<PathBuf>>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(None);
    }
    let conn = open_history_db(path)?;
    let table_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(None);
    }
    let raw: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key=?1 LIMIT 1",
            params![STALE_PATCH_TARGETS_META_KEY],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    raw.map(|raw| {
        serde_json::from_str::<Vec<PathBuf>>(&raw)
            .map(|paths| paths.into_iter().collect())
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid stale patch target metadata: {error}"),
                )
            })
    })
    .transpose()
}

/// Atomically replace the current session's stale-patch ledger. An empty set is explicitly written as `[]` to distinguish
/// “known empty” from “old database not yet initialized”; this runtime metadata does not change model history, so
/// `history_revision` is not incremented.
pub(in crate::ai) fn write_stale_patch_targets_sqlite(
    path: &Path,
    targets: &FxHashSet<PathBuf>,
) -> io::Result<()> {
    if !blob::is_sqlite_path(path) {
        return Ok(());
    }
    let mut paths = targets.iter().cloned().collect::<Vec<_>>();
    paths.sort();
    let encoded =
        serde_json::to_string(&paths).map_err(|error| io::Error::other(error.to_string()))?;
    with_session_state_lock(path, || {
        let conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        conn.execute(
            "INSERT INTO meta (key, value, created_at) VALUES (?1, ?2, unixepoch())
             ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at",
            params![STALE_PATCH_TARGETS_META_KEY, encoded],
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        Ok(())
    })
}

/// Read the current session's model-guided prune counts. Missing or non-SQLite history safely degrades to empty.
pub(in crate::ai) fn read_llm_prune_marks_sqlite(path: &Path) -> io::Result<FxHashMap<String, u8>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(FxHashMap::default());
    }
    let conn = open_history_db(path)?;
    let table_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(FxHashMap::default());
    }
    let raw: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key=?1 LIMIT 1",
            params![LLM_PRUNE_MARKS_META_KEY],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    let Some(raw) = raw else {
        return Ok(FxHashMap::default());
    };
    let entries = serde_json::from_str::<Vec<(String, u8)>>(&raw).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid LLM prune mark metadata: {error}"),
        )
    })?;
    Ok(entries
        .into_iter()
        .filter(|(id, count)| !id.trim().is_empty() && *count > 0)
        .take(1_024)
        .collect())
}

/// Atomically replace the current session's model-guided prune counts. This side state does not change canonical
/// messages, so `history_revision` is not incremented; an empty table deletes the meta row directly to avoid leaving empty state.
pub(in crate::ai) fn write_llm_prune_marks_sqlite(
    path: &Path,
    marks: &FxHashMap<String, u8>,
) -> io::Result<()> {
    if !blob::is_sqlite_path(path) {
        return Ok(());
    }
    let mut entries = marks
        .iter()
        .filter(|(id, count)| !id.trim().is_empty() && **count > 0)
        .map(|(id, count)| (id.clone(), *count))
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let encoded =
        serde_json::to_string(&entries).map_err(|error| io::Error::other(error.to_string()))?;
    with_session_state_lock(path, || {
        let conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        if entries.is_empty() {
            conn.execute(
                "DELETE FROM meta WHERE key=?1",
                params![LLM_PRUNE_MARKS_META_KEY],
            )
            .map_err(|error| io::Error::other(error.to_string()))?;
        } else {
            conn.execute(
                "INSERT INTO meta (key, value, created_at) VALUES (?1, ?2, unixepoch())
                 ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at",
                params![LLM_PRUNE_MARKS_META_KEY, encoded],
            )
            .map_err(|error| io::Error::other(error.to_string()))?;
        }
        Ok(())
    })
}

fn clear_llm_prune_marks_meta(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "DELETE FROM meta WHERE key=?1",
        params![LLM_PRUNE_MARKS_META_KEY],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

/// An outcome belongs only to the tool message with the same `tool_call_id`. After history replacement, compaction, or branch truncation,
/// side records that have lost their message owner are cleared immediately, so a deleted occurrence's state cannot pollute the retained history.
fn prune_orphan_tool_execution_outcomes(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "DELETE FROM tool_execution_outcomes
         WHERE tool_call_id NOT IN (
             SELECT DISTINCT tool_call_id FROM messages
             WHERE role = 'tool' AND tool_call_id IS NOT NULL
         )",
        [],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

/// Older history may have reused `tool_call_id` before the occurrence IDs were fixed. Once a later replacement or
/// truncation keeps only one of them, counting the current messages alone cannot tell which occurrence the outcome belonged to,
/// so these ambiguous side states must be permanently discarded before the message set changes.
fn drop_ambiguous_tool_execution_outcomes(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "DELETE FROM tool_execution_outcomes
         WHERE tool_call_id IN (
             SELECT tool_call_id FROM messages
             WHERE role = 'tool' AND tool_call_id IS NOT NULL
             GROUP BY tool_call_id HAVING COUNT(1) > 1
         )",
        [],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

pub(in crate::ai) fn append_history_sqlite(path: &Path, entries: Vec<Message>) -> io::Result<()> {
    append_history_sqlite_for_model(path, entries, None)
}

/// Only append raw messages to canonical history. The model origin is kept as side metadata and never rewrites
/// `Message` itself; provider-specific projections are only produced later when building a rebuildable context view.
pub(in crate::ai) fn append_history_sqlite_for_model(
    path: &Path,
    entries: Vec<Message>,
    source_model: Option<&str>,
) -> io::Result<()> {
    with_session_state_lock(path, || {
        append_history_sqlite_for_model_unlocked(path, entries, source_model)
    })
}

fn append_history_sqlite_for_model_unlocked(
    path: &Path,
    entries: Vec<Message>,
    source_model: Option<&str>,
) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    if entries.is_empty() {
        return Ok(());
    }
    let first_user_in_blob = entries
        .iter()
        .find(|message| message.role == "user")
        .map(|message| value_to_string(&message.content));
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    {
        let existing_first: Option<String> = tx
            .query_row(
                "SELECT value FROM meta WHERE key='first_user_prompt' LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .unwrap_or(None);
        if existing_first.is_none() {
            let first_existing_user: Option<String> = tx
                .query_row(
                    "SELECT content FROM messages WHERE role='user' ORDER BY id ASC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .optional()
                .unwrap_or(None);
            let first_user_prompt = first_existing_user.or(first_user_in_blob.clone());
            if let Some(v) = first_user_prompt.as_deref() {
                let _ = tx.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('first_user_prompt', ?1)",
                    params![v],
                );
            }
        }
        insert_messages(&tx, entries, source_model)?;
    }
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

pub(in crate::ai) fn replace_all_messages_sqlite(
    path: &Path,
    messages: &[Message],
) -> io::Result<()> {
    with_session_state_lock(path, || {
        replace_all_messages_sqlite_unlocked(path, messages)
    })
}

/// Wake-note dedup (approach 1): for the same process and the same set of task_ids, only the latest
/// TASK_WAIT_TIMEOUT “still waiting” wake note is kept. The caller calls this before appending an introspection note:
/// it deletes all old waiting notes within the last `WAKE_NOTE_DEDUP_SCAN` messages whose identity matches the note about to be appended
/// (the caller then appends the latest one at the tail); returns `Ok(false)` for non-“still waiting” wake notes or when nothing matches.
pub(in crate::ai) fn coalesce_repeated_wait_wake_notes_sqlite(
    path: &Path,
    note: &Message,
) -> io::Result<bool> {
    with_session_state_lock(path, || {
        coalesce_repeated_wait_wake_notes_sqlite_unlocked(path, note)
    })
}

fn coalesce_repeated_wait_wake_notes_sqlite_unlocked(
    path: &Path,
    note: &Message,
) -> io::Result<bool> {
    // fast path: no IO at all for wake notes that are not “still waiting”
    if note.role != super::types::ROLE_INTERNAL_NOTE {
        return Ok(false);
    }
    let Some(text) = note.content.as_str() else {
        return Ok(false);
    };
    let Some(identity) = super::types::parse_still_waiting_wake_identity(text) else {
        return Ok(false);
    };

    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let mut stmt = conn
        .prepare(
            // Consistent with the blob backend: the window is the last WAKE_NOTE_DEDUP_SCAN messages of the history (any role),
            // then identity matching runs over the internal_note rows inside it — LIMIT applies before the role filter.
            "SELECT id, content
             FROM (SELECT id, content, role FROM messages ORDER BY id DESC LIMIT ?1)
             WHERE role = ?2
             ORDER BY id ASC",
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    let rows = stmt
        .query_map(
            params![
                super::types::WAKE_NOTE_DEDUP_SCAN as i64,
                super::types::ROLE_INTERNAL_NOTE
            ],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    let mut to_delete = Vec::<i64>::new();
    for row in rows {
        let (id, content_json) = row.map_err(|e| io::Error::other(e.to_string()))?;
        let content = decode_message_content(&content_json);
        let Some(content_text) = content.as_str() else {
            continue;
        };
        if super::types::parse_still_waiting_wake_identity(content_text).as_ref()
            == Some(&identity)
        {
            to_delete.push(id);
        }
    }
    if to_delete.is_empty() {
        return Ok(false);
    }

    drop(stmt);
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    for id in &to_delete {
        tx.execute("DELETE FROM messages WHERE id=?1", params![id])
            .map_err(|e| io::Error::other(e.to_string()))?;
    }
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))?;
    Ok(true)
}

fn replace_all_messages_sqlite_unlocked(path: &Path, messages: &[Message]) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    drop_ambiguous_tool_execution_outcomes(&tx)?;
    tx.execute("DELETE FROM messages", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    invalidate_context_snapshot(&tx)?;
    tx.execute("DELETE FROM meta WHERE key='first_user_prompt'", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    insert_messages(&tx, messages.to_vec(), None)?;
    prune_orphan_tool_execution_outcomes(&tx)?;
    refresh_first_user_prompt_meta(&tx, messages)?;
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

fn insert_messages(
    conn: &Connection,
    messages: Vec<Message>,
    source_model: Option<&str>,
) -> io::Result<()> {
    let mut stmt = conn
        .prepare(
            "INSERT INTO messages
                (role, content, tool_calls, tool_call_id, reasoning_content, source_model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    for message in messages {
        let content =
            serde_json::to_string(&message.content).map_err(|e| io::Error::other(e.to_string()))?;
        let tool_calls = message
            .tool_calls
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| io::Error::other(e.to_string()))?;
        stmt.execute(params![
            message.role,
            content,
            tool_calls,
            message.tool_call_id,
            message.reasoning_content,
            source_model,
        ])
        .map_err(|e| io::Error::other(e.to_string()))?;
    }
    Ok(())
}

fn refresh_first_user_prompt_meta(conn: &Connection, messages: &[Message]) -> io::Result<()> {
    let Some(first_user_prompt) = messages
        .iter()
        .find(|message| message.role == "user")
        .map(|message| value_to_string(&message.content))
    else {
        return Ok(());
    };
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('first_user_prompt', ?1)",
        params![first_user_prompt],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

fn read_messages_with_sql(
    conn: &Connection,
    sql: &str,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok((role, content, tool_calls, tool_call_id, reasoning_content))
    })?;
    let mut messages = Vec::new();
    for row in rows {
        let (role, content, tool_calls, tool_call_id, reasoning_content) = row?;
        messages.push(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        });
    }
    Ok(messages)
}

fn read_projected_canonical_messages_after_id(
    conn: &Connection,
    after_id: i64,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content, source_model
         FROM messages
         WHERE id > ?1
         ORDER BY id ASC",
    )?;
    let rows = stmt.query_map(params![after_id], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        let source_model: Option<String> = row.get(5)?;
        Ok((
            Message {
                role,
                content: decode_message_content(&content),
                tool_calls: decode_tool_calls(tool_calls.as_deref()),
                tool_call_id,
                reasoning_content,
            },
            source_model,
        ))
    })?;

    let mut messages = Vec::new();
    for row in rows {
        let (message, source_model) = row?;
        messages.push(match source_model.as_deref() {
            Some(model) => {
                compress::sanitize_message_for_persisted_history_for_model(model, &message)
            }
            None => compress::sanitize_message_for_persisted_history(&message),
        });
    }
    Ok(messages)
}

/// Return the model context layer: the latest replaceable compaction snapshot plus the projection of raw messages after the snapshot watermark.
/// The read completes within a single SQLite snapshot transaction, so `source_message_id` exactly describes the canonical watermark
/// that the returned value has consumed; concurrent appends naturally become the tail of the next read instead of being swallowed by the snapshot.
///
/// Cross-turn image summary: write the image summary of a user message containing images into the history metadata table.
/// `message_key` is a stable fingerprint of the message content (see `request::image_message_fingerprint`),
/// so the summary travels with the message: the next turn loads the history, retrieves it with the same fingerprint, and replaces the old images to avoid resending them.
pub(in crate::ai) fn upsert_image_digest_sqlite(
    path: &Path,
    message_key: &str,
    digest: &str,
    image_paths: &[String],
) -> io::Result<()> {
    if !blob::is_sqlite_path(path) {
        return Ok(());
    }
    let encoded_paths =
        serde_json::to_string(image_paths).map_err(|error| io::Error::other(error.to_string()))?;
    with_session_state_lock(path, || {
        let conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        conn.execute(
            "INSERT INTO image_digests (message_key, digest, image_paths, created_at)
             VALUES (?1, ?2, ?3, unixepoch())
             ON CONFLICT(message_key) DO UPDATE SET
                 digest = excluded.digest,
                 image_paths = excluded.image_paths,
                 created_at = excluded.created_at",
            params![message_key, digest, encoded_paths],
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        Ok(())
    })
}

/// Cross-turn image summary: read the persisted image summary by the message content fingerprint.
/// Returns (summary text, original image path); None when the history DB lacks the table or the key (preserving the original-image semantics).
pub(in crate::ai) fn read_image_digest_sqlite(
    path: &Path,
    message_key: &str,
) -> io::Result<Option<(String, Vec<String>)>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(None);
    }
    let conn = open_history_db_read_only(path)?;
    let has_table = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'image_digests'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .unwrap_or(false);
    if !has_table {
        return Ok(None);
    }
    let row: Option<(String, String)> = conn
        .query_row(
            "SELECT digest, image_paths FROM image_digests WHERE message_key = ?1 LIMIT 1",
            params![message_key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    row.map(|(digest, paths_json)| {
        serde_json::from_str::<Vec<String>>(&paths_json)
            .map(|paths| (digest, paths))
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid image digest paths metadata: {error}"),
                )
            })
    })
    .transpose()
}

pub(in crate::ai) fn read_context_history_sqlite(
    path: &Path,
    projection_fingerprint: &str,
) -> io::Result<ContextHistory> {
    with_cached_read_conn(path, Duration::from_secs(2), |conn| {
        read_context_history_on_conn(conn, projection_fingerprint)
    })
}

fn read_context_history_on_conn(
    conn: &mut Connection,
    projection_fingerprint: &str,
) -> io::Result<ContextHistory> {
    init_history_schema(conn)?;
    let tx = conn
        .transaction()
        .map_err(|error| io::Error::other(error.to_string()))?;
    let canonical_generation = history_generation(&tx)?;
    let latest_message_id = tx
        .query_row("SELECT COALESCE(MAX(id), 0) FROM messages", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(|error| io::Error::other(error.to_string()))?;
    let snapshot = tx
        .query_row(
            "SELECT source_message_id, source_generation, projection_fingerprint
             FROM context_snapshot WHERE singleton = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .filter(|(_, generation, fingerprint)| {
            *generation == canonical_generation && fingerprint == projection_fingerprint
        });

    let (mut messages, after_id, has_snapshot) = if let Some((source_message_id, _, _)) = snapshot {
        let messages = read_messages_with_sql(
            &tx,
            "SELECT role, content, tool_calls, tool_call_id, reasoning_content
                 FROM context_messages ORDER BY position ASC",
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        (messages, source_message_id, true)
    } else {
        (Vec::new(), 0, false)
    };
    messages.extend(
        read_projected_canonical_messages_after_id(&tx, after_id)
            .map_err(|error| io::Error::other(error.to_string()))?,
    );
    tx.commit()
        .map_err(|error| io::Error::other(error.to_string()))?;

    Ok(ContextHistory {
        messages,
        source_message_id: latest_message_id,
        canonical_generation,
        snapshot_is_current: has_snapshot && after_id == latest_message_id,
    })
}

/// Atomically replace the rebuildable context snapshot. If canonical is rewritten (rewind/clear, etc.) after the snapshot was read,
/// the generation changes and the stale result is rejected; an ordinary concurrent append does not change the generation, and its
/// message id is greater than the passed watermark, so a later read merges it back as the tail.
pub(in crate::ai) fn write_context_snapshot_sqlite(
    path: &Path,
    messages: &[Message],
    source_message_id: i64,
    canonical_generation: i64,
    projection_fingerprint: &str,
) -> io::Result<bool> {
    write_context_snapshot_sqlite_with_busy_timeout(
        path,
        messages,
        source_message_id,
        canonical_generation,
        projection_fingerprint,
        Duration::from_secs(5),
    )
}

pub(in crate::ai) fn write_context_snapshot_sqlite_with_busy_timeout(
    path: &Path,
    messages: &[Message],
    source_message_id: i64,
    canonical_generation: i64,
    projection_fingerprint: &str,
    busy_timeout: Duration,
) -> io::Result<bool> {
    let deadline = Instant::now() + busy_timeout;
    with_session_state_lock_until(path, deadline, || {
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .max(Duration::from_millis(1));
        let mut conn = open_history_db_with_busy_timeout(path, remaining)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| sqlite_error(path, "begin context snapshot transaction", error))?;
        if history_generation(&tx)? != canonical_generation {
            return Ok(false);
        }

        tx.execute("DELETE FROM context_messages", [])
            .map_err(|error| sqlite_error(path, "clear context_messages", error))?;
        insert_context_messages(&tx, path, messages)?;
        tx.execute(
            "INSERT INTO context_snapshot
                (singleton, source_message_id, source_generation, projection_fingerprint)
             VALUES (1, ?1, ?2, ?3)
             ON CONFLICT(singleton) DO UPDATE SET
                source_message_id = excluded.source_message_id,
                source_generation = excluded.source_generation,
                projection_fingerprint = excluded.projection_fingerprint",
            params![
                source_message_id,
                canonical_generation,
                projection_fingerprint
            ],
        )
        .map_err(|error| sqlite_error(path, "upsert context_snapshot", error))?;
        bump_history_revision(&tx)?;
        tx.commit()
            .map_err(|error| sqlite_error(path, "commit context snapshot transaction", error))?;
        Ok(true)
    })
}

fn insert_context_messages(conn: &Connection, path: &Path, messages: &[Message]) -> io::Result<()> {
    let mut stmt = conn
        .prepare(
            "INSERT INTO context_messages
                (position, role, content, tool_calls, tool_call_id, reasoning_content)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .map_err(|error| sqlite_error(path, "prepare context_messages insert", error))?;
    for (position, message) in messages.iter().enumerate() {
        let content = serde_json::to_string(&message.content)
            .map_err(|error| io::Error::other(error.to_string()))?;
        let tool_calls = message
            .tool_calls
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| io::Error::other(error.to_string()))?;
        stmt.execute(params![
            position as i64,
            message.role,
            content,
            tool_calls,
            message.tool_call_id,
            message.reasoning_content,
        ])
        .map_err(|error| {
            sqlite_error(
                path,
                &format!("insert context_messages row {position}"),
                error,
            )
        })?;
    }
    Ok(())
}

pub(in crate::ai) fn build_message_arr_sqlite(
    history_count: usize,
    history_file: &Path,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    let messages = read_messages_with_sql(
        &conn,
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         ORDER BY id ASC",
    )?;
    if history_count >= messages.len() {
        return Ok(messages);
    }
    Ok(messages[messages.len() - history_count..].to_vec())
}

pub(in crate::ai) fn read_recent_messages_sqlite(
    history_file: &Path,
    limit: usize,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        })
    })?;

    let mut messages = Vec::new();
    for row in rows {
        messages.push(row?);
    }
    Ok(messages)
}

pub(in crate::ai) fn read_recent_turn_window_sqlite(
    history_file: &Path,
    keep_last_user_turns: usize,
) -> Result<RecentTurnWindow, Box<dyn std::error::Error>> {
    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Ok(RecentTurnWindow {
                messages: Vec::new(),
                start_message_id: None,
                has_older_messages: false,
            });
        }
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    if keep_last_user_turns == 0 {
        let messages = read_messages_with_sql(
            &conn,
            "SELECT role, content, tool_calls, tool_call_id, reasoning_content
             FROM messages
             ORDER BY id ASC",
        )?;
        return Ok(RecentTurnWindow {
            messages,
            start_message_id: None,
            has_older_messages: false,
        });
    }

    let threshold_user_id: Option<i64> = conn
        .query_row(
            "SELECT id FROM messages
             WHERE role='user'
             ORDER BY id DESC
             LIMIT 1 OFFSET ?1",
            params![keep_last_user_turns.saturating_sub(1) as i64],
            |row| row.get(0),
        )
        .optional()?;

    let Some(start_message_id) = threshold_user_id else {
        let messages = read_messages_with_sql(
            &conn,
            "SELECT role, content, tool_calls, tool_call_id, reasoning_content
             FROM messages
             ORDER BY id ASC",
        )?;
        return Ok(RecentTurnWindow {
            messages,
            start_message_id: None,
            has_older_messages: false,
        });
    };

    let has_older_messages = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM messages WHERE id < ?1 LIMIT 1)",
            params![start_message_id],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        != 0;

    let messages = read_messages_since_id(&conn, start_message_id)?;
    Ok(RecentTurnWindow {
        messages,
        start_message_id: Some(start_message_id),
        has_older_messages,
    })
}

pub(in crate::ai) fn read_latest_history_summary_before_id_sqlite(
    history_file: &Path,
    before_message_id: i64,
) -> Result<Option<Message>, Box<dyn std::error::Error>> {
    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         WHERE id < ?1 AND role = ?2
         ORDER BY id DESC
         LIMIT 8",
    )?;
    let rows = stmt.query_map(params![before_message_id, ROLE_INTERNAL_NOTE], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        })
    })?;

    for row in rows {
        let message = row?;
        let Some(text) = message.content.as_str() else {
            continue;
        };
        // Summary-prefix recognition uniformly goes through compress::is_summary_note_text (the single source of truth).
        // Previously three prefixes were hardcoded here, missing `长期记忆摘要（压缩保留）`, so the fast path
        // could not find the summary continuation point produced by the overflow path and fell back to a full slow re-compaction every turn.
        if is_summary_note_text(text) {
            return Ok(Some(message));
        }
    }
    Ok(None)
}

/// Read the most recent context checkpoint markers before the sliding window. They are the only
/// index for the body assets and must not silently vanish from the request context just because the SQLite fast path only loads recent turns.
/// The request normalization layer still restricts the final projection to the most recent 8 entries.
pub(in crate::ai) fn read_context_checkpoint_markers_before_id_sqlite(
    history_file: &Path,
    before_message_id: i64,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         WHERE id < ?1
           AND role = ?2
           AND instr(content, '[context_checkpoint') > 0
         ORDER BY id DESC
         LIMIT 8",
    )?;
    let rows = stmt.query_map(params![before_message_id, ROLE_INTERNAL_NOTE], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        })
    })?;

    let mut markers = Vec::new();
    for row in rows {
        let message = row?;
        if message
            .content
            .as_str()
            .is_some_and(|text| text.trim_start().starts_with("[context_checkpoint "))
        {
            markers.push(message);
        }
    }
    markers.reverse();
    Ok(markers)
}

/// Relocate the asset paths of the context checkpoint markers in the history to a new session.
/// On fork, the source assets directory is passed in for an exact prefix replacement; on archive import the source path is unknown, so only
/// the controlled relative tail of `context-checkpoints/<file>` is accepted, avoiding rewrites of arbitrary text or absolute paths.
pub(in crate::ai) fn remap_context_checkpoint_paths_sqlite(
    history_file: &Path,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> io::Result<usize> {
    with_session_state_lock(history_file, || {
        remap_context_checkpoint_paths_sqlite_unlocked(history_file, source_assets, target_assets)
    })
}

fn remap_context_checkpoint_paths_sqlite_unlocked(
    history_file: &Path,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> io::Result<usize> {
    let mut conn = open_history_db(history_file)?;
    init_history_schema(&conn)?;
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    let rows = {
        let mut stmt = tx
            .prepare(
                "SELECT id, content
                 FROM messages
                 WHERE role = ?1
                   AND instr(content, '[context_checkpoint path=') > 0",
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        let rows = stmt
            .query_map([ROLE_INTERNAL_NOTE], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| io::Error::other(e.to_string()))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| io::Error::other(e.to_string()))?
    };

    let mut remapped = 0usize;
    for (id, encoded_content) in rows {
        let content = decode_message_content(&encoded_content);
        let Some(text) = content.as_str() else {
            continue;
        };
        let Some(remapped_text) =
            remap_context_checkpoint_marker(text, source_assets, target_assets)
        else {
            continue;
        };
        let encoded = serde_json::to_string(&Value::String(remapped_text))
            .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute(
            "UPDATE messages SET content = ?1 WHERE id = ?2",
            params![encoded, id],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        remapped += 1;
    }
    if remapped > 0 {
        invalidate_context_snapshot(&tx)?;
        bump_history_revision(&tx)?;
    }
    tx.commit().map_err(|e| io::Error::other(e.to_string()))?;
    Ok(remapped)
}

fn remap_context_checkpoint_marker(
    text: &str,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> Option<String> {
    const PREFIX: &str = "[context_checkpoint path=";
    let leading_len = text.len().checked_sub(text.trim_start().len())?;
    let (leading, trimmed) = text.split_at(leading_len);
    let rest = trimmed.strip_prefix(PREFIX)?;
    let closing = rest.find(']')?;
    let recorded = Path::new(&rest[..closing]);
    let relative = source_assets
        .and_then(|source| recorded.strip_prefix(source).ok())
        .and_then(checked_context_checkpoint_relative)
        .or_else(|| checked_context_checkpoint_relative(recorded))?;
    let remapped = target_assets.join(relative);
    Some(format!(
        "{leading}{PREFIX}{}{}",
        remapped.display(),
        &rest[closing..]
    ))
}

fn checked_context_checkpoint_relative(path: &Path) -> Option<PathBuf> {
    let mut found_checkpoint_dir = false;
    let mut relative = PathBuf::new();
    let mut has_file = false;
    for component in path.components() {
        if !found_checkpoint_dir {
            if let std::path::Component::Normal(part) = component
                && part == "context-checkpoints"
            {
                relative.push(part);
                found_checkpoint_dir = true;
            }
            continue;
        };
        match component {
            std::path::Component::Normal(part) => {
                relative.push(part);
                has_file = true;
            }
            _ => return None,
        }
    }
    (found_checkpoint_dir && has_file).then_some(relative)
}

pub(in crate::ai) fn clear_session_history_sqlite(path: &Path) -> io::Result<()> {
    with_session_state_lock(path, || clear_session_history_sqlite_unlocked(path))
}

fn clear_session_history_sqlite_unlocked(path: &Path) -> io::Result<()> {
    let mut conn = match open_history_db(path) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    init_history_schema(&conn)?;
    // Transaction wrapper: DELETE messages / DELETE meta / bump revision must commit atomically,
    // otherwise a crash in the middle leaves the inconsistent state “messages cleared but revision unchanged”,
    // which would make the context cache misjudge that nothing changed and keep serving the old history.
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM messages", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    invalidate_context_snapshot(&tx)?;
    tx.execute("DELETE FROM tool_execution_outcomes", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM interrupted_stream_diagnostics", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM skill_activation_events", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    // Keep the history_revision row: it is the cache-invalidation counter and must stay **monotonically increasing** across clears.
    // history_generation is the fencing token for concurrent snapshot writes and must also increase monotonically after a clear;
    // turn_seq is likewise session-scoped identity; clearing the context must not let old numbers be reused.
    // If they were deleted along with the rest, the bump would restart at 1; after the version regresses it could collide with
    // the revision of early cache entries, and already-invalidated old history would be wrongly hit.
    tx.execute(
        "DELETE FROM meta
         WHERE key NOT IN ('history_revision', 'history_generation', 'turn_seq')",
        [],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

/// Keep only the first `keep` rows of the messages table (ascending by id). Used for session branching:
/// copy the full sqlite then roll back to the given message count. `keep == 0` is equivalent to clear.
pub(in crate::ai) fn truncate_messages_sqlite(path: &Path, keep: usize) -> io::Result<()> {
    with_session_state_lock(path, || truncate_messages_sqlite_unlocked(path, keep))
}

fn truncate_messages_sqlite_unlocked(path: &Path, keep: usize) -> io::Result<()> {
    let mut conn = match open_history_db(path) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    init_history_schema(&conn)?;
    // Transaction wrapper: DELETE + bump revision commit atomically, so a crash in the middle cannot leave
    // the inconsistent “messages deleted but revision unchanged” state (the context cache serving a wrongly empty result).
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    drop_ambiguous_tool_execution_outcomes(&tx)?;
    if keep == 0 {
        tx.execute("DELETE FROM messages", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        invalidate_context_snapshot(&tx)?;
        tx.execute("DELETE FROM tool_execution_outcomes", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute("DELETE FROM interrupted_stream_diagnostics", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute("DELETE FROM skill_activation_events", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        clear_llm_prune_marks_meta(&tx)?;
        bump_history_revision(&tx)?;
        return tx.commit().map_err(|e| io::Error::other(e.to_string()));
    }
    // Take the largest id among the first `keep` rows and delete every row after it.
    let cutoff: Option<i64> = tx
        .query_row(
            "SELECT id FROM messages ORDER BY id ASC LIMIT 1 OFFSET ?1",
            params![(keep as i64) - 1],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?;
    if let Some(cutoff_id) = cutoff {
        tx.execute("DELETE FROM messages WHERE id > ?1", params![cutoff_id])
            .map_err(|e| io::Error::other(e.to_string()))?;
        invalidate_context_snapshot(&tx)?;
        clear_llm_prune_marks_meta(&tx)?;
    }
    prune_orphan_tool_execution_outcomes(&tx)?;
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

/// Keep the messages table down to the first `keep_turns` complete user turns.
///
/// A user turn starts at a `role='user'` message and ends before the next user message; truncating at the next user message
/// keeps an assistant tool call and its following tool result on the same side.
pub(in crate::ai) fn truncate_messages_to_user_turns_sqlite(
    path: &Path,
    keep_turns: usize,
) -> io::Result<()> {
    if keep_turns == 0 {
        return truncate_messages_sqlite(path, 0);
    }

    with_session_state_lock(path, || {
        truncate_messages_to_user_turns_sqlite_unlocked(path, keep_turns)
    })
}

fn truncate_messages_to_user_turns_sqlite_unlocked(
    path: &Path,
    keep_turns: usize,
) -> io::Result<()> {
    let mut conn = match open_history_db(path) {
        Ok(connection) => connection,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    init_history_schema(&conn)?;
    let tx = conn
        .transaction()
        .map_err(|error| io::Error::other(error.to_string()))?;
    drop_ambiguous_tool_execution_outcomes(&tx)?;
    let next_turn_start: Option<i64> = tx
        .query_row(
            "SELECT id FROM messages WHERE role = 'user' ORDER BY id ASC LIMIT 1 OFFSET ?1",
            params![keep_turns as i64],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    if let Some(next_turn_start) = next_turn_start {
        tx.execute(
            "DELETE FROM messages WHERE id >= ?1",
            params![next_turn_start],
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        invalidate_context_snapshot(&tx)?;
        clear_llm_prune_marks_meta(&tx)?;
    }
    prune_orphan_tool_execution_outcomes(&tx)?;
    bump_history_revision(&tx)?;
    tx.commit()
        .map_err(|error| io::Error::other(error.to_string()))
}

pub(in crate::ai) fn read_first_user_prompt_sqlite(path: &Path) -> io::Result<Option<String>> {
    let conn = open_history_db(path)?;
    read_first_user_prompt_from_conn(&conn)
}

fn read_first_user_prompt_from_conn(conn: &Connection) -> io::Result<Option<String>> {
    let meta: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='first_user_prompt' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    if meta
        .as_deref()
        .is_some_and(|prompt| !super::sessions::is_preserved_content_message(prompt))
    {
        return Ok(meta);
    }

    // The cached first message may be an image/text archival protocol message. Keep scanning forward for the first real user request,
    // so an existing session is not wrongly shown as `new session` after the internal protocol messages are filtered out.
    let mut stmt = conn
        .prepare("SELECT content FROM messages WHERE role='user' ORDER BY id ASC")
        .map_err(|e| io::Error::other(e.to_string()))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| io::Error::other(e.to_string()))?;
    let mut prompts = Vec::with_capacity(3);
    for raw in rows {
        let raw = raw.map_err(|e| io::Error::other(e.to_string()))?;
        let prompt = value_to_string(&decode_message_content(&raw));
        if !super::sessions::is_preserved_content_message(&prompt) {
            prompts.push(prompt);
            if prompts.len() == 3 {
                break;
            }
        }
    }
    Ok((!prompts.is_empty()).then(|| prompts.join("\n---\n")))
}

/// Read the session title (stored in the meta table under key='session_title').
pub(in crate::ai) fn read_session_title_sqlite(path: &Path) -> io::Result<Option<String>> {
    let conn = open_history_db(path)?;
    Ok(read_session_title_from_conn(&conn))
}

fn read_session_title_from_conn(conn: &Connection) -> Option<String> {
    let title: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='session_title' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    title.filter(|title| !title.trim().is_empty())
}

fn read_i64_meta_from_conn(conn: &Connection, key: &str) -> Option<i64> {
    conn.query_row(
        "SELECT value FROM meta WHERE key=?1 LIMIT 1",
        params![key],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .ok()
    .flatten()
    .and_then(|value| value.parse::<i64>().ok())
}

/// When an older session has not written an explicit activity time, use the creation time of the last canonical message
/// as the activity time. `messages.created_at` is in Unix seconds; the list interface uniformly returns milliseconds.
fn read_latest_message_activity_unix_ms(conn: &Connection) -> Option<i64> {
    conn.query_row("SELECT MAX(created_at) FROM messages", [], |row| {
        row.get::<_, Option<i64>>(0)
    })
    .ok()
    .flatten()
    .and_then(|seconds| seconds.checked_mul(1_000))
}

/// Read the title, first user request, and activity time for the `/ss` list in a single read-only connection.
///
/// The two metadata items keep the list layer's original fault-tolerance semantics: a failure in one query does not affect the other, nor does it let
/// a corrupted or old-format session block the whole list.
pub(in crate::ai) fn read_session_list_metadata_sqlite(
    path: &Path,
) -> io::Result<SessionListMetadata> {
    let conn = open_history_db_read_only(path)?;
    Ok(SessionListMetadata {
        first_user_prompt: read_first_user_prompt_from_conn(&conn).unwrap_or(None),
        session_title: read_session_title_from_conn(&conn),
        last_activity_unix_ms: read_i64_meta_from_conn(&conn, LAST_ACTIVITY_META_KEY)
            .or_else(|| read_latest_message_activity_unix_ms(&conn)),
        history_revision: read_i64_meta_from_conn(&conn, "history_revision").unwrap_or(0),
        marked: read_session_marked_from_conn(&conn),
    })
}

/// Read the session "important" mark stored under meta key `session_marked`.
/// Missing or unparsable values are treated as unmarked (no session is important
/// by default).
fn read_session_marked_from_conn(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT value FROM meta WHERE key='session_marked' LIMIT 1",
        [],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .unwrap_or(None)
    .is_some_and(|value| value == "1")
}

/// Read the session "important" mark (`/mark`).
pub(in crate::ai) fn read_session_marked_sqlite(path: &Path) -> io::Result<bool> {
    let conn = open_history_db_read_only(path)?;
    Ok(read_session_marked_from_conn(&conn))
}

/// Persist the session "important" mark (`/mark` / `/unmark`). Uses a key/value
/// in the meta table so the flag survives clear-history and is copied by fork /
/// import (both copy the whole SQLite file).
pub(in crate::ai) fn write_session_marked_sqlite(path: &Path, marked: bool) -> io::Result<()> {
    with_session_state_lock(path, || {
        let mut conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction()
            .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute(
            "INSERT OR REPLACE INTO meta (key, value, created_at) VALUES (?1, ?2, unixepoch())",
            rusqlite::params![SESSION_MARKED_META_KEY, if marked { "1" } else { "0" }],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        touch_session_activity(&tx)?;
        tx.commit().map_err(|e| io::Error::other(e.to_string()))
    })
}

/// Read the source of the session title (`model` / `fallback`); when missing, the caller treats it as legacy data.
pub(in crate::ai) fn read_session_title_origin_sqlite(path: &Path) -> io::Result<Option<String>> {
    let conn = open_history_db(path)?;
    let origin: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='session_title_origin' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    Ok(origin.filter(|value| !value.trim().is_empty()))
}

/// Atomically write the session title and its source, so a fallback is never mistaken for a model title and permanently skips upgrading.
pub(in crate::ai) fn write_session_title_sqlite(
    path: &Path,
    title: &str,
    origin: &str,
) -> io::Result<()> {
    with_session_state_lock(path, || {
        let mut conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction()
            .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute(
            "INSERT OR REPLACE INTO meta (key, value, created_at) VALUES ('session_title', ?1, unixepoch())",
            rusqlite::params![title],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute(
            "INSERT OR REPLACE INTO meta (key, value, created_at) VALUES ('session_title_origin', ?1, unixepoch())",
            rusqlite::params![origin],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        touch_session_activity(&tx)?;
        tx.commit().map_err(|e| io::Error::other(e.to_string()))
    })
}

fn decode_message_content(content: &str) -> Value {
    serde_json::from_str(content).unwrap_or_else(|_| Value::String(content.to_string()))
}

fn decode_tool_calls(tool_calls: Option<&str>) -> Option<Vec<ToolCall>> {
    tool_calls.and_then(|raw| serde_json::from_str(raw).ok())
}

pub(in crate::ai) fn read_all_messages_sqlite(path: &Path) -> io::Result<Vec<Message>> {
    let conn = open_history_db(path)?;

    read_messages_with_sql(
        &conn,
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         ORDER BY id ASC",
    )
    .map_err(|e| io::Error::other(e.to_string()))
}

fn read_messages_since_id(
    conn: &Connection,
    start_message_id: i64,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         WHERE id >= ?1
         ORDER BY id ASC",
    )?;
    let rows = stmt.query_map(params![start_message_id], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok((role, content, tool_calls, tool_call_id, reasoning_content))
    })?;
    let mut messages = Vec::new();
    for row in rows {
        let (role, content, tool_calls, tool_call_id, reasoning_content) = row?;
        messages.push(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        });
    }
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn sqlite_error_classifies_transient_failures_as_retryable() {
        // BUSY/LOCKED and SQLITE_IOERR system I/O failures (e.g. FSTAT/SHMOPEN when
        // concurrently opening the WAL for the first time) are transient and must map to WouldBlock so the caller can retry with a short backoff.
        for result_code in [
            rusqlite::ffi::SQLITE_BUSY,
            rusqlite::ffi::SQLITE_LOCKED,
            rusqlite::ffi::SQLITE_IOERR,
            rusqlite::ffi::SQLITE_IOERR_FSTAT,
        ] {
            let error =
                rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(result_code), None);
            assert_eq!(
                sqlite_error_kind(&error),
                io::ErrorKind::WouldBlock,
                "schema and transaction paths must share retry classification"
            );
            let io_error = sqlite_error(Path::new("history.db"), "test operation", error);
            assert_eq!(io_error.kind(), io::ErrorKind::WouldBlock);
        }

        // Non-transient failures (e.g. a read-only filesystem) must not be retried.
        let error = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_READONLY),
            None,
        );
        let io_error = sqlite_error(Path::new("history.db"), "test operation", error);
        assert_eq!(io_error.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn context_snapshot_busy_timeout_includes_session_state_lock_wait() {
        let dir = std::env::temp_dir().join(format!(
            "snapshot_state_lock_timeout_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let holder_path = path.clone();
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            with_session_state_lock(&holder_path, || {
                locked_tx.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(250));
                Ok(())
            })
            .unwrap();
        });
        locked_rx.recv().unwrap();

        let started = Instant::now();
        let error = write_context_snapshot_sqlite_with_busy_timeout(
            &path,
            &[],
            0,
            0,
            "fingerprint",
            Duration::from_millis(30),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
        assert!(started.elapsed() < Duration::from_millis(200));

        holder.join().unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn tool_msg(id: &str, text: &str) -> Message {
        let mut message = msg("tool", text);
        message.tool_call_id = Some(id.to_string());
        message
    }

    fn outcome(id: &str) -> ToolExecutionOutcome {
        ToolExecutionOutcome {
            tool_call_id: id.to_string(),
            execution_signature: format!("signature-{id}"),
            succeeded: true,
        }
    }

    #[test]
    fn turn_sequence_is_atomic_and_survives_history_clear() {
        let dir = std::env::temp_dir().join(format!(
            "turn_sequence_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "first"), msg("user", "second")]).unwrap();

        // When an older session is upgraded for the first time, numbering continues from the persisted user-turn count.
        assert_eq!(reserve_turn_index_sqlite(&path).unwrap(), 2);
        assert_eq!(reserve_turn_index_sqlite(&path).unwrap(), 3);

        clear_session_history_sqlite(&path).unwrap();
        assert_eq!(reserve_turn_index_sqlite(&path).unwrap(), 4);

        // Each thread opens its own connection, covering the same transaction-contention paths across runtimes and processes.
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || reserve_turn_index_sqlite(&path).unwrap())
            })
            .collect();
        let mut indexes: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        indexes.sort_unstable();
        assert_eq!(indexes, (5..13).collect::<Vec<_>>());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rollback_metadata_read_error_does_not_replace_live_history() {
        let dir = std::env::temp_dir().join(format!(
            "rollback_metadata_error_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let live = dir.join("live.sqlite");
        let checkpoint = dir.join("checkpoint.sqlite");
        append_history_sqlite(&live, vec![msg("user", "live message")]).unwrap();
        append_history_sqlite(&checkpoint, vec![msg("user", "checkpoint message")]).unwrap();
        let conn = Connection::open(&live).unwrap();
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('turn_seq', 'invalid')
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [],
        )
        .unwrap();
        drop(conn);

        let error = restore_sqlite_after_rollback(&checkpoint, &live).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let messages = read_all_messages_sqlite(&live).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(value_to_string(&messages[0].content), "live message");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P1 regression: `read_history_revision` must observe the write increment **across connections**.
    /// Every write path opens a new connection to read the version, mimicking build_context_history's cache-invalidation check.
    /// The old implementation used the connection-local `PRAGMA data_version`, which always returns a fixed value on new connections and cannot invalidate the cache.
    #[test]
    fn history_revision_increments_across_fresh_connections() {
        let dir = std::env::temp_dir().join(format!(
            "hist_rev_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        // A brand-new DB has never written a revision: treat it as 0 (“never modified”).
        assert_eq!(read_history_revision(&path), Some(0));

        append_history_sqlite(&path, vec![msg("user", "hi")]).unwrap();
        let r1 = read_history_revision(&path).unwrap();
        assert!(r1 > 0, "append should bump revision, got {r1}");

        append_history_sqlite(&path, vec![msg("assistant", "hello")]).unwrap();
        let r2 = read_history_revision(&path).unwrap();
        assert!(r2 > r1, "second append should bump again: {r1} -> {r2}");

        replace_all_messages_sqlite(&path, &[msg("user", "reset")]).unwrap();
        let r3 = read_history_revision(&path).unwrap();
        assert!(r3 > r2, "replace should bump: {r2} -> {r3}");

        truncate_messages_sqlite(&path, 0).unwrap();
        let r4 = read_history_revision(&path).unwrap();
        assert!(r4 > r3, "truncate should bump: {r3} -> {r4}");

        // clear deletes the meta first and then bumps, a structure unlike the other write paths, so it needs its own coverage.
        append_history_sqlite(&path, vec![msg("user", "again")]).unwrap();
        let r5 = read_history_revision(&path).unwrap();
        clear_session_history_sqlite(&path).unwrap();
        let r6 = read_history_revision(&path).unwrap();
        assert!(
            r6 > r5,
            "clear should bump even after wiping meta: {r5} -> {r6}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_revision_cache_invalidates_on_wal_write_with_live_connection() {
        // Verify: with a live connection, WAL writes do not checkpoint the main DB, but the `-wal` sidecar changes
        // must invalidate the revision cache, or the cache would return a stale value.
        let dir = std::env::temp_dir().join(format!(
            "hist_rev_wal_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        append_history_sqlite(&path, vec![msg("user", "first")]).unwrap();
        let r1 = read_history_revision(&path).unwrap();
        assert!(r1 > 0);

        // Keep the connection alive to prevent the WAL checkpoint from writing back to the main DB
        let guard = rusqlite::Connection::open(&path).unwrap();

        // Short-lived connection writes: the WAL grows but the main DB mtime may stay unchanged
        append_history_sqlite(&path, vec![msg("user", "second")]).unwrap();

        // The cache must invalidate (WAL sidecar metadata changed) and return the new revision
        let r2 = read_history_revision(&path).unwrap();
        assert!(
            r2 > r1,
            "revision must increment even with live connection: {r1} -> {r2}"
        );

        drop(guard);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn llm_prune_marks_round_trip_and_clear_on_history_truncate() {
        let dir = std::env::temp_dir().join(format!(
            "llm_prune_marks_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.sqlite");
        append_history_sqlite(&path, vec![msg("user", "seed")]).unwrap();

        let marks = [("call_b".to_string(), 2_u8), ("call_a".to_string(), 1_u8)]
            .into_iter()
            .collect::<FxHashMap<_, _>>();
        write_llm_prune_marks_sqlite(&path, &marks).unwrap();
        assert_eq!(read_llm_prune_marks_sqlite(&path).unwrap(), marks);

        truncate_messages_sqlite(&path, 0).unwrap();
        assert!(read_llm_prune_marks_sqlite(&path).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_patch_targets_survive_history_replacement_and_clear_with_session() {
        let dir = std::env::temp_dir().join(format!(
            "stale_patch_meta_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.sqlite");
        append_history_sqlite(&path, vec![msg("user", "before compression")]).unwrap();

        let targets =
            FxHashSet::from_iter([PathBuf::from("/tmp/a.rs"), PathBuf::from("/tmp/b.rs")]);
        write_stale_patch_targets_sqlite(&path, &targets).unwrap();
        assert_eq!(
            read_stale_patch_targets_sqlite(&path).unwrap(),
            Some(targets.clone())
        );

        // replace_all_messages is the persisted-history compaction/rewrite path; the ledger must not be lost with the message shape.
        replace_all_messages_sqlite(&path, &[msg("user", "after compression")]).unwrap();
        assert_eq!(
            read_stale_patch_targets_sqlite(&path).unwrap(),
            Some(targets)
        );

        // An explicitly empty set must also be distinguished from “old DB with no meta yet”, so recovery never wrongly takes the legacy replay path.
        write_stale_patch_targets_sqlite(&path, &FxHashSet::default()).unwrap();
        assert_eq!(
            read_stale_patch_targets_sqlite(&path).unwrap(),
            Some(FxHashSet::default())
        );

        clear_session_history_sqlite(&path).unwrap();
        assert_eq!(read_stale_patch_targets_sqlite(&path).unwrap(), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn first_user_prompt_skips_preserved_content_notices() {
        let dir = std::env::temp_dir().join(format!(
            "first_prompt_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        append_history_sqlite(
            &path,
            vec![
                msg("user", "较早的用户图片内容已归档，原文未丢失。"),
                msg("user", r#"[[PRESERVED_CONTENT_STUB_V1]]{"kind":"image"}"#),
                msg("user", "较早的用户图片内容已归档，归档文件: /tmp/2"),
                msg("user", "较早的用户图片内容已归档，归档文件: /tmp/3"),
                msg("user", "较早的用户图片内容已归档，归档文件: /tmp/4"),
                msg("user", "这是实际用户请求"),
                msg("user", "这是后续用户请求"),
            ],
        )
        .unwrap();

        assert_eq!(
            read_first_user_prompt_sqlite(&path).unwrap().as_deref(),
            Some("这是实际用户请求\n---\n这是后续用户请求")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_list_metadata_reads_fields_and_legacy_activity_without_creating_database() {
        let dir = std::env::temp_dir().join(format!(
            "session_list_metadata_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let missing_path = dir.join("missing.db");
        assert!(read_session_list_metadata_sqlite(&missing_path).is_err());
        assert!(!missing_path.exists());

        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "first prompt")]).unwrap();
        write_session_title_sqlite(&path, "generated title", "model").unwrap();

        // Simulate an older session that has not written last_activity_unix_ms. The list must use the message time,
        // not the SQLite -shm file time, which a read-only connection creates/refreshes.
        let conn = open_history_db(&path).unwrap();
        conn.execute("UPDATE messages SET created_at = ?1", [1_700_000_000_i64])
            .unwrap();
        conn.execute("DELETE FROM meta WHERE key = 'last_activity_unix_ms'", [])
            .unwrap();
        drop(conn);

        let metadata = read_session_list_metadata_sqlite(&path).unwrap();
        assert_eq!(metadata.first_user_prompt.as_deref(), Some("first prompt"));
        assert_eq!(metadata.session_title.as_deref(), Some("generated title"));
        assert_eq!(metadata.last_activity_unix_ms, Some(1_700_000_000_000));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn structured_tool_outcomes_follow_message_lifecycle() {
        let dir = std::env::temp_dir().join(format!(
            "tool_outcome_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let original_messages = vec![tool_msg("call-1", "first"), tool_msg("call-2", "second")];
        append_history_sqlite(&path, original_messages.clone()).unwrap();
        let expected = vec![outcome("call-1"), outcome("call-2")];
        append_tool_execution_outcomes_sqlite(&path, &expected).unwrap();

        assert_eq!(
            read_tool_execution_outcomes_sqlite(&path).unwrap(),
            expected
        );
        assert_eq!(read_all_messages_sqlite(&path).unwrap(), original_messages);

        replace_all_messages_sqlite(&path, &[tool_msg("call-2", "second")]).unwrap();
        assert_eq!(
            read_tool_execution_outcomes_sqlite(&path).unwrap(),
            vec![outcome("call-2")]
        );

        append_history_sqlite(&path, vec![tool_msg("call-3", "third")]).unwrap();
        append_tool_execution_outcomes_sqlite(&path, &[outcome("call-3")]).unwrap();
        truncate_messages_sqlite(&path, 1).unwrap();
        assert_eq!(
            read_tool_execution_outcomes_sqlite(&path).unwrap(),
            vec![outcome("call-2")]
        );

        clear_session_history_sqlite(&path).unwrap();
        assert!(
            read_tool_execution_outcomes_sqlite(&path)
                .unwrap()
                .is_empty()
        );

        // Older history may have reused the same ID; before the message set changes, its ambiguous outcome must be permanently discarded,
        // otherwise deleting the newer occurrence would wrongly bind its state to the retained older message.
        append_history_sqlite(
            &path,
            vec![
                tool_msg("legacy-reused", "older"),
                tool_msg("legacy-reused", "newer"),
            ],
        )
        .unwrap();
        append_tool_execution_outcomes_sqlite(&path, &[outcome("legacy-reused")]).unwrap();
        replace_all_messages_sqlite(&path, &[tool_msg("legacy-reused", "older")]).unwrap();
        assert!(
            read_tool_execution_outcomes_sqlite(&path)
                .unwrap()
                .is_empty()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skill_activation_events_preserve_injection_audit_without_messages() {
        let dir = std::env::temp_dir().join(format!(
            "skill_activation_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let event = SkillActivationEvent {
            requested_skill: "bytedcli".to_string(),
            injected_skill: Some("bytedcli".to_string()),
            source: "/skills-inline".to_string(),
            outcome: "injected".to_string(),
        };

        append_skill_activation_event_sqlite(&path, &event).unwrap();

        assert_eq!(
            read_skill_activation_events_sqlite(&path).unwrap(),
            vec![event]
        );
        assert!(read_all_messages_sqlite(&path).unwrap().is_empty());

        clear_session_history_sqlite(&path).unwrap();
        assert!(
            read_skill_activation_events_sqlite(&path)
                .unwrap()
                .is_empty()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn interrupted_stream_diagnostics_are_isolated_from_model_history() {
        let dir = std::env::temp_dir().join(format!(
            "interrupted_stream_diagnostic_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        append_interrupted_stream_diagnostic_sqlite(
            &path,
            "test-model",
            "partial body",
            "partial reasoning",
        )
        .unwrap();

        assert!(read_all_messages_sqlite(&path).unwrap().is_empty());
        let conn = Connection::open(&path).unwrap();
        let row: (String, String, String) = conn
            .query_row(
                "SELECT assistant_text, reasoning_text, source_model
                 FROM interrupted_stream_diagnostics",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            row,
            (
                "partial body".to_string(),
                "partial reasoning".to_string(),
                "test-model".to_string(),
            )
        );

        clear_session_history_sqlite(&path).unwrap();
        let remaining: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM interrupted_stream_diagnostics",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn structured_tool_outcomes_ignore_non_sqlite_history_files() {
        let dir = std::env::temp_dir().join(format!(
            "tool_outcome_text_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.txt");
        std::fs::write(&path, "plain text history\n").unwrap();

        append_tool_execution_outcomes_sqlite(&path, &[outcome("call-1")]).unwrap();

        assert!(
            read_tool_execution_outcomes_sqlite(&path)
                .unwrap()
                .is_empty()
        );
        assert!(read_tool_message_ids_sqlite(&path).unwrap().is_empty());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "plain text history\n"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncating_by_user_turn_prunes_removed_tool_outcomes() {
        let dir = std::env::temp_dir().join(format!(
            "tool_outcome_turn_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(
            &path,
            vec![
                msg("user", "first turn"),
                tool_msg("call-1", "first result"),
                msg("user", "second turn"),
                tool_msg("call-2", "second result"),
            ],
        )
        .unwrap();
        append_tool_execution_outcomes_sqlite(&path, &[outcome("call-1"), outcome("call-2")])
            .unwrap();

        truncate_messages_to_user_turns_sqlite(&path, 1).unwrap();

        assert_eq!(
            read_tool_execution_outcomes_sqlite(&path).unwrap(),
            vec![outcome("call-1")]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn canonical_messages_are_unchanged_by_context_snapshots() {
        const PROJECTION: &str = "test-policy-a";
        let dir = std::env::temp_dir().join(format!(
            "context_snapshot_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let assistant = Message {
            role: "assistant".to_string(),
            content: Value::String("准备读取文件".to_string()),
            tool_calls: Some(vec![ToolCall {
                id: "call-1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: Some("provider 原始 continuation state".to_string()),
        };
        let original = vec![msg("user", "读取文件"), assistant.clone()];
        append_history_sqlite_for_model(&path, original.clone(), Some("glm-5.2-opencode")).unwrap();

        let source = read_context_history_sqlite(&path, PROJECTION).unwrap();
        assert_eq!(read_all_messages_sqlite(&path).unwrap(), original);
        assert_ne!(
            source.messages[1].reasoning_content,
            assistant.reasoning_content
        );

        let snapshot = vec![msg(ROLE_INTERNAL_NOTE, "[History summary]\n已读取文件")];
        write_context_snapshot_sqlite(
            &path,
            &snapshot,
            source.source_message_id,
            source.canonical_generation,
            PROJECTION,
        )
        .unwrap();
        assert_eq!(read_all_messages_sqlite(&path).unwrap(), original);

        let tail = msg("user", "继续分析");
        append_history_sqlite_for_model(&path, vec![tail.clone()], Some("gpt-5.5")).unwrap();
        let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
        assert_eq!(projected.messages, vec![snapshot[0].clone(), tail.clone()]);
        assert!(!projected.snapshot_is_current);

        let mut expected_canonical = original;
        expected_canonical.push(tail);
        assert_eq!(read_all_messages_sqlite(&path).unwrap(), expected_canonical);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_context_snapshot_cannot_resurrect_truncated_history() {
        const PROJECTION: &str = "test-policy-a";
        let dir = std::env::temp_dir().join(format!(
            "stale_context_snapshot_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "保留"), msg("assistant", "删除")]).unwrap();
        let stale_source = read_context_history_sqlite(&path, PROJECTION).unwrap();

        truncate_messages_sqlite(&path, 1).unwrap();
        let written = write_context_snapshot_sqlite(
            &path,
            &[msg(ROLE_INTERNAL_NOTE, "包含已删除内容的旧快照")],
            stale_source.source_message_id,
            stale_source.canonical_generation,
            PROJECTION,
        )
        .unwrap();
        assert!(!written);
        assert_eq!(
            read_context_history_sqlite(&path, PROJECTION)
                .unwrap()
                .messages,
            vec![msg("user", "保留")]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn context_snapshot_is_ignored_when_projection_policy_changes() {
        let dir = std::env::temp_dir().join(format!(
            "context_snapshot_policy_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let canonical = vec![msg("user", "必须从 canonical 重建")];
        append_history_sqlite(&path, canonical.clone()).unwrap();

        let source = read_context_history_sqlite(&path, "policy-a").unwrap();
        write_context_snapshot_sqlite(
            &path,
            &[msg(ROLE_INTERNAL_NOTE, "旧策略摘要")],
            source.source_message_id,
            source.canonical_generation,
            "policy-a",
        )
        .unwrap();
        assert!(
            read_context_history_sqlite(&path, "policy-a")
                .unwrap()
                .snapshot_is_current
        );

        let rebuilt = read_context_history_sqlite(&path, "policy-b").unwrap();
        assert_eq!(rebuilt.messages, canonical);
        assert!(!rebuilt.snapshot_is_current);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_context_snapshot_cannot_resurrect_cleared_history() {
        const PROJECTION: &str = "test-policy-a";
        let dir = std::env::temp_dir().join(format!(
            "stale_context_snapshot_clear_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "已经清除")]).unwrap();
        let stale_source = read_context_history_sqlite(&path, PROJECTION).unwrap();

        clear_session_history_sqlite(&path).unwrap();
        let current = msg("user", "清除后的新会话");
        append_history_sqlite(&path, vec![current.clone()]).unwrap();
        let written = write_context_snapshot_sqlite(
            &path,
            &[msg(ROLE_INTERNAL_NOTE, "包含已清除内容的旧快照")],
            stale_source.source_message_id,
            stale_source.canonical_generation,
            PROJECTION,
        )
        .unwrap();

        assert!(!written);
        assert_eq!(
            read_context_history_sqlite(&path, PROJECTION)
                .unwrap()
                .messages,
            vec![current]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rollback_state_lock_serializes_every_public_side_state_writer() {
        let dir = std::env::temp_dir().join(format!(
            "history_side_writer_lock_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "seed")]).unwrap();

        type Writer = Box<dyn FnOnce(&Path) -> io::Result<()> + Send>;
        let writers: Vec<(&str, Writer)> = vec![
            (
                "tool outcomes",
                Box::new(|path| append_tool_execution_outcomes_sqlite(path, &[outcome("call-1")])),
            ),
            (
                "stale patch targets",
                Box::new(|path| {
                    let mut targets = FxHashSet::default();
                    targets.insert(PathBuf::from("src/main.rs"));
                    write_stale_patch_targets_sqlite(path, &targets)
                }),
            ),
            (
                "context snapshot",
                Box::new(|path| {
                    let source = read_context_history_sqlite(path, "lock-test")?;
                    write_context_snapshot_sqlite(
                        path,
                        &[msg(ROLE_INTERNAL_NOTE, "snapshot")],
                        source.source_message_id,
                        source.canonical_generation,
                        "lock-test",
                    )
                    .map(|_| ())
                }),
            ),
            (
                "session title",
                Box::new(|path| write_session_title_sqlite(path, "title", "test")),
            ),
        ];

        for (name, writer) in writers {
            let (lock_held_tx, lock_held_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let lock_path = path.clone();
            let lock_thread = std::thread::spawn(move || {
                with_session_state_lock(&lock_path, || {
                    lock_held_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
            });
            lock_held_rx.recv().unwrap();

            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let writer_path = path.clone();
            let writer_thread = std::thread::spawn(move || {
                let result = writer(&writer_path);
                done_tx.send(()).unwrap();
                result
            });
            assert!(
                done_rx
                    .recv_timeout(std::time::Duration::from_millis(100))
                    .is_err(),
                "{name} bypassed the rollback state lock"
            );

            release_tx.send(()).unwrap();
            lock_thread.join().unwrap().unwrap();
            writer_thread.join().unwrap().unwrap();
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn wake_note_text(pid: u64, ids: &[&str], checkpoint: &str) -> String {
        format!(
            "[Process {pid} Woke Up] Original goal: test goal\nNew mailbox messages:\n[TASK_WAIT_TIMEOUT]\nWall-clock task_wait budget elapsed after 30s. Re-call `task_wait` with the same task_ids to collect any ready results and receive the budget-elapsed status. task_ids=[{}]\nProgress: {checkpoint}\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
            ids.join(", ")
        )
    }

    #[test]
    fn wait_wake_notes_coalesce_keeps_latest_in_sqlite_history() {
        let dir = std::env::temp_dir().join(format!(
            "wait_wake_coalesce_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        append_history_sqlite(
            &path,
            vec![
                msg("user", "goal"),
                msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-1")),
                msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-2")),
                msg(ROLE_INTERNAL_NOTE, &wake_note_text(7, &["task_x"], "checkpoint-3")),
            ],
        )
        .unwrap();

        let latest =
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-4"));
        assert!(coalesce_repeated_wait_wake_notes_sqlite(&path, &latest).unwrap());
        // The caller then appends the latest one at the tail.
        append_history_sqlite(&path, vec![latest]).unwrap();

        let notes: Vec<_> = read_all_messages_sqlite(&path)
            .unwrap()
            .into_iter()
            .filter(|m| m.role == ROLE_INTERNAL_NOTE)
            .collect();
        assert_eq!(notes.len(), 2);
        assert!(notes[0].content.as_str().unwrap().contains("checkpoint-3"));
        assert!(notes[1].content.as_str().unwrap().contains("checkpoint-4"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wait_wake_coalesce_sqlite_window_is_last_total_messages() {
        let dir = std::env::temp_dir().join(format!(
            "sqlite_wake_dedup_window_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        // Window semantics consistent with the blob backend: scan the last WAKE_NOTE_DEDUP_SCAN messages of the history (any role),
        // not “the most recent WAKE_NOTE_DEDUP_SCAN internal_notes”.
        // The old waiting note is at position 1, followed by WAKE_NOTE_DEDUP_SCAN+1 user messages, so it is outside the window
        // and must not be deleted (an internal_note-window scan would have hit and removed it).
        let mut history = vec![msg(
            ROLE_INTERNAL_NOTE,
            &wake_note_text(6, &["task_a"], "checkpoint-old"),
        )];
        history.extend(
            (0..crate::ai::history::types::WAKE_NOTE_DEDUP_SCAN as usize + 1)
                .map(|i| msg("user", &format!("filler {i}"))),
        );
        append_history_sqlite(&path, history).unwrap();

        let latest =
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "checkpoint-new"));
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &latest).unwrap());

        let notes: Vec<_> = read_all_messages_sqlite(&path)
            .unwrap()
            .into_iter()
            .filter(|m| m.role == ROLE_INTERNAL_NOTE)
            .collect();
        // The old waiting note outside the window is retained and not wrongly deleted.
        assert_eq!(notes.len(), 1);
        assert!(notes[0].content.as_str().unwrap().contains("checkpoint-old"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wait_wake_coalesce_sqlite_is_noop_when_nothing_matches() {
        let dir = std::env::temp_dir().join(format!(
            "wait_wake_coalesce_noop_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(
            &path,
            vec![
                msg("user", "goal"),
                msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "checkpoint-1")),
            ],
        )
        .unwrap();

        // Same pid but a different task set: the identity differs, so no dedup.
        let other = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_z"], "checkpoint-x"));
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &other).unwrap());

        // Non-internal_note message: the fast path does no IO.
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &msg("user", "hello")).unwrap());

        // A real-result wake (parse returns None): no dedup.
        let result_wake = msg(
            ROLE_INTERNAL_NOTE,
            "[Process 6 Woke Up] Original goal: g\nNew mailbox messages:\n[EVENT_WAKE]\nready\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
        );
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &result_wake).unwrap());

        // Missing database: best-effort returns false without error.
        // Note: only a valid wait note gets past the fast path to actually reach the open_history_db branch.
        let missing = dir.join("missing.db");
        let wait_note = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "c"));
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&missing, &wait_note).unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
