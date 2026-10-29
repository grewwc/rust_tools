use std::{
    io::{self, Read, Write},
    path::Path,
    time::{Duration, Instant},
};

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

use super::super::compress;
use super::super::compress::is_summary_note_text;
use super::super::tool_guidance::ToolGuidanceFilter;
use super::super::types::{Message, ROLE_INTERNAL_NOTE};
use super::connection::{
    open_history_db, open_history_db_with_busy_timeout, sqlite_error, with_cached_read_conn,
};
use super::lock::with_session_state_lock_until;
use super::migrations::init_history_schema;
use super::revision::{bump_history_revision, history_generation};
use super::store::{
    ContextHistory, RecentTurnWindow, decode_message_content, decode_tool_calls,
    read_messages_since_id, read_messages_with_sql,
};

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

/// Append the canonical messages that survive a truncation but lie past the
/// retained snapshot watermark, so the snapshot covers the whole surviving
/// prefix and `snapshot_is_current` holds on the next turn.
///
/// Truncation already deleted everything past the cutoff, so every row after
/// the watermark belongs to the surviving history. The appended rows use the
/// same persisted projection as a freshly written snapshot; guidance filtering
/// still runs whenever the projection is served.
pub(super) fn extend_context_snapshot_through_latest(conn: &Connection) -> io::Result<()> {
    let watermark: Option<i64> = conn
        .query_row(
            "SELECT source_message_id FROM context_snapshot WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    let Some(watermark) = watermark else {
        return Ok(());
    };
    let latest: i64 = conn
        .query_row("SELECT COALESCE(MAX(id), 0) FROM messages", [], |row| {
            row.get(0)
        })
        .map_err(|error| io::Error::other(error.to_string()))?;
    if latest <= watermark {
        return Ok(());
    }
    let tail = read_projected_canonical_messages_after_id(conn, watermark)
        .map_err(|error| io::Error::other(error.to_string()))?;
    let next_position: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(position) + 1, 0) FROM context_messages",
            [],
            |row| row.get(0),
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
    let mut stmt = conn
        .prepare(
            "INSERT INTO context_messages
                (position, role, content, tool_calls, tool_call_id, reasoning_content)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
    for (offset, message) in tail.iter().enumerate() {
        let content = serde_json::to_string(&message.content)
            .map_err(|error| io::Error::other(error.to_string()))?;
        let tool_calls = message
            .tool_calls
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| io::Error::other(error.to_string()))?;
        stmt.execute(params![
            next_position + offset as i64,
            message.role,
            content,
            tool_calls,
            message.tool_call_id,
            message.reasoning_content,
        ])
        .map_err(|error| io::Error::other(error.to_string()))?;
    }
    drop(stmt);
    conn.execute(
        "UPDATE context_snapshot SET source_message_id = ?1 WHERE singleton = 1",
        params![latest],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
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

    // Apply the turn boundary only after the canonical tail is merged. A new
    // real user in that tail expires guidance from the persisted snapshot.
    let original_len = messages.len();
    let mut guidance = ToolGuidanceFilter::new(&messages);
    messages.retain(|message| guidance.keep(message));
    let guidance_unchanged = messages.len() == original_len;
    Ok(ContextHistory {
        messages,
        source_message_id: latest_message_id,
        canonical_generation,
        snapshot_is_current: has_snapshot && after_id == latest_message_id && guidance_unchanged,
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

        archive_context_snapshot(&tx, path)?;
        prune_context_snapshot_history(&tx, path)?;
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

/// Number of replaced snapshots kept as rewind fallbacks.
///
/// Each entry holds one projection: the compressed prefix plus whatever raw
/// tail a retained rewind appended, stored as one gzip payload so a fallback
/// keeps a single compressed copy instead of every projection row. The ring is
/// therefore bounded in entries rather than bytes.
const CONTEXT_SNAPSHOT_HISTORY_KEEP: usize = 8;

/// Move the snapshot being replaced into the bounded rewind ring.
///
/// A rewind that cuts before the newest snapshot's watermark cannot reuse it,
/// but an older projection whose watermark still survives can serve as the
/// prefix. Keeping several turn boundaries lets consecutive rewinds stay fast
/// instead of recompressing the whole canonical history.
fn archive_context_snapshot(conn: &Connection, path: &Path) -> io::Result<()> {
    let current: Option<(i64, i64, String)> = conn
        .query_row(
            "SELECT source_message_id, source_generation, projection_fingerprint
             FROM context_snapshot WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()
        .map_err(|error| sqlite_error(path, "read context_snapshot", error))?;
    let Some((source_message_id, source_generation, fingerprint)) = current else {
        return Ok(());
    };
    if source_generation != history_generation(conn)? {
        // A canonical rewrite already invalidated the projection; it cannot
        // prove coverage of any surviving prefix, and neither can older rows.
        conn.execute("DELETE FROM context_snapshot_history", [])
            .map_err(|error| sqlite_error(path, "clear context_snapshot_history", error))?;
        return Ok(());
    }
    // History watermarks increase monotonically; an equivalent or newer range
    // is not an older fallback.
    // A row without a payload cannot be promoted (for example one written by a
    // build that predates payloads); drop it first so it neither blocks a new
    // archive nor takes a ring slot.
    conn.execute(
        "DELETE FROM context_snapshot_history WHERE payload IS NULL",
        [],
    )
    .map_err(|error| sqlite_error(path, "drop unpromotable ring entries", error))?;
    let newest = conn
        .query_row(
            "SELECT source_message_id FROM context_snapshot_history
             ORDER BY source_message_id DESC LIMIT 1",
            [],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(|error| sqlite_error(path, "read context_snapshot_history", error))?;
    if newest.is_some_and(|watermark| watermark >= source_message_id) {
        return Ok(());
    }
    let payload = encode_context_history_payload(&read_archived_context_messages(conn, path)?)?;
    conn.execute(
        "INSERT INTO context_snapshot_history
            (source_message_id, projection_fingerprint, payload)
         VALUES (?1, ?2, ?3)",
        params![source_message_id, fingerprint, payload],
    )
    .map_err(|error| sqlite_error(path, "save context_snapshot_history", error))?;
    Ok(())
}

/// Drop the oldest archived projections beyond the rewind ring.
fn prune_context_snapshot_history(conn: &Connection, path: &Path) -> io::Result<()> {
    conn.execute(
        "DELETE FROM context_snapshot_history
         WHERE snapshot_id IN (
            SELECT snapshot_id FROM context_snapshot_history
            ORDER BY source_message_id DESC, snapshot_id DESC
            LIMIT -1 OFFSET ?1
         )",
        [CONTEXT_SNAPSHOT_HISTORY_KEEP as i64],
    )
    .map_err(|error| sqlite_error(path, "prune context_snapshot_history", error))?;
    Ok(())
}

/// One row of a context projection, kept in exactly the strings
/// `context_messages` stores so promotion never re-encodes message JSON.
#[derive(serde::Deserialize, serde::Serialize)]
struct ArchivedContextMessage {
    position: i64,
    role: String,
    content: String,
    tool_calls: Option<String>,
    tool_call_id: Option<String>,
    reasoning_content: Option<String>,
}

/// Read the current projection in the form archived into a ring entry.
fn read_archived_context_messages(
    conn: &Connection,
    path: &Path,
) -> io::Result<Vec<ArchivedContextMessage>> {
    let mut stmt = conn
        .prepare(
            "SELECT position, role, content, tool_calls, tool_call_id, reasoning_content
             FROM context_messages ORDER BY position ASC",
        )
        .map_err(|error| sqlite_error(path, "read context_messages for archive", error))?;
    let rows = stmt
        .query_map([], |row| {
            Ok(ArchivedContextMessage {
                position: row.get(0)?,
                role: row.get(1)?,
                content: row.get(2)?,
                tool_calls: row.get(3)?,
                tool_call_id: row.get(4)?,
                reasoning_content: row.get(5)?,
            })
        })
        .map_err(|error| sqlite_error(path, "read context_messages for archive", error))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| sqlite_error(path, "read context_messages for archive", error))
}

/// Serialize one ring entry into a single gzip payload.
///
/// A projection repeats message JSON structure and consecutive boundaries often
/// share text, so one compressed payload is several times smaller than storing
/// the same rows again. The ring is only read as a whole during a rewind, which
/// makes a single payload a good fit.
fn encode_context_history_payload(messages: &[ArchivedContextMessage]) -> io::Result<Vec<u8>> {
    let json = serde_json::to_vec(messages).map_err(io::Error::other)?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&json)?;
    encoder.finish()
}

fn decode_context_history_payload(payload: &[u8]) -> io::Result<Vec<ArchivedContextMessage>> {
    let mut json = Vec::new();
    GzDecoder::new(payload).read_to_end(&mut json)?;
    serde_json::from_slice(&json).map_err(io::Error::other)
}

/// Restore the newest archived projection whose whole source range survives a
/// truncation that keeps rows up to `max_kept_id`.
///
/// Archived projections that cover removed rows can never serve a later rewind
/// and are dropped. The promoted row itself stays in the ring: it still covers
/// only surviving rows, so a deeper rewind can reuse it. Returns whether a
/// projection was restored.
pub(super) fn promote_context_snapshot_from_history(
    conn: &Connection,
    max_kept_id: i64,
    generation: i64,
) -> io::Result<bool> {
    let entry = conn
        .query_row(
            "SELECT snapshot_id, payload FROM context_snapshot_history
             WHERE source_message_id <= ?1 AND payload IS NOT NULL
             ORDER BY source_message_id DESC LIMIT 1",
            [max_kept_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    let Some((snapshot_id, payload)) = entry else {
        return Ok(false);
    };
    let messages = decode_context_history_payload(&payload)?;
    conn.execute("DELETE FROM context_messages", [])
        .map_err(|error| io::Error::other(error.to_string()))?;
    {
        let mut stmt = conn
            .prepare(
                "INSERT INTO context_messages
                    (position, role, content, tool_calls, tool_call_id, reasoning_content)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )
            .map_err(|error| io::Error::other(error.to_string()))?;
        for message in &messages {
            stmt.execute(params![
                message.position,
                message.role,
                message.content,
                message.tool_calls,
                message.tool_call_id,
                message.reasoning_content,
            ])
            .map_err(|error| io::Error::other(error.to_string()))?;
        }
    }
    conn.execute(
        "INSERT INTO context_snapshot
            (singleton, source_message_id, source_generation, projection_fingerprint)
         SELECT 1, source_message_id, ?2, projection_fingerprint
         FROM context_snapshot_history WHERE snapshot_id = ?1
         ON CONFLICT(singleton) DO UPDATE SET
            source_message_id = excluded.source_message_id,
            source_generation = excluded.source_generation,
            projection_fingerprint = excluded.projection_fingerprint",
        params![snapshot_id, generation],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    conn.execute(
        "DELETE FROM context_snapshot_history WHERE source_message_id > ?1",
        [max_kept_id],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(true)
}

fn insert_context_messages(conn: &Connection, path: &Path, messages: &[Message]) -> io::Result<()> {
    let mut stmt = conn
        .prepare(
            "INSERT INTO context_messages
                (position, role, content, tool_calls, tool_call_id, reasoning_content)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .map_err(|error| sqlite_error(path, "prepare context_messages insert", error))?;
    let mut guidance = ToolGuidanceFilter::new(messages);
    for (position, message) in messages
        .iter()
        .filter(|message| guidance.keep(message))
        .enumerate()
    {
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
