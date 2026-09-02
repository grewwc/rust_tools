use std::{
    io,
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OptionalExtension, params};
use rustc_hash::FxHashMap;

use super::LAST_ACTIVITY_META_KEY;

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
pub(super) fn touch_session_activity(conn: &Connection) -> io::Result<()> {
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
pub(super) fn bump_history_revision(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('history_revision', '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
        [],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    touch_session_activity(conn)
}

pub(super) fn history_generation(conn: &Connection) -> io::Result<i64> {
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
pub(super) fn invalidate_context_snapshot(conn: &Connection) -> io::Result<()> {
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
pub(super) fn read_i64_meta_from_conn(conn: &Connection, key: &str) -> Option<i64> {
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
