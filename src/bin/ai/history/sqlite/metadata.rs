use std::{io, path::Path};

use rusqlite::{Connection, OptionalExtension};

use super::super::compress::value_to_string;
use super::super::sessions::is_preserved_content_message;
use super::connection::{open_history_db, open_history_db_read_only};
use super::lock::with_session_state_lock;
use super::migrations::init_history_schema;
use super::revision::{read_i64_meta_from_conn, touch_session_activity};
use super::store::{SessionListMetadata, decode_message_content};
use super::{LAST_ACTIVITY_META_KEY, SESSION_MARKED_META_KEY};

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
        .is_some_and(|prompt| !is_preserved_content_message(prompt))
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
        if !is_preserved_content_message(&prompt) {
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
