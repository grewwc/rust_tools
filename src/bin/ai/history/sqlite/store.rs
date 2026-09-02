use std::{io, path::Path};

use serde_json::Value;

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

use crate::ai::types::ToolCall;

use super::super::compress::{COMPRESSED_TOOL_EVIDENCE_MARKER, value_to_string};
use super::super::types::{
    Message, ROLE_INTERNAL_NOTE, WAKE_NOTE_DEDUP_SCAN, parse_still_waiting_wake_identity,
};
use super::{
    connection::open_history_db,
    lock::with_session_state_lock,
    migrations::init_history_schema,
    outcomes::{drop_ambiguous_tool_execution_outcomes, prune_orphan_tool_execution_outcomes},
    revision::{bump_history_revision, invalidate_context_snapshot, touch_session_activity},
};

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
    if note.role != ROLE_INTERNAL_NOTE {
        return Ok(false);
    }
    let Some(text) = note.content.as_str() else {
        return Ok(false);
    };
    let Some(identity) = parse_still_waiting_wake_identity(text) else {
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
                WAKE_NOTE_DEDUP_SCAN as i64,
                ROLE_INTERNAL_NOTE
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
        if parse_still_waiting_wake_identity(content_text).as_ref()
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

pub(super) fn read_messages_with_sql(
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
pub(super) fn decode_message_content(content: &str) -> Value {
    serde_json::from_str(content).unwrap_or_else(|_| Value::String(content.to_string()))
}

pub(super) fn decode_tool_calls(tool_calls: Option<&str>) -> Option<Vec<ToolCall>> {
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

pub(super) fn read_messages_since_id(
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
