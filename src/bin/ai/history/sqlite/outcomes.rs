use std::{
    io,
    path::{Path, PathBuf},
};

use rusqlite::{Connection, OptionalExtension, params};
use rustc_hash::{FxHashMap, FxHashSet};

use super::super::blob;
use super::super::types::{SkillActivationEvent, ToolExecutionOutcome};
use super::connection::{open_history_db, open_history_db_read_only};
use super::lock::with_session_state_lock;
use super::migrations::init_history_schema;
use super::revision::bump_history_revision;
use super::{LLM_PRUNE_MARKS_META_KEY, STALE_PATCH_TARGETS_META_KEY};

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

pub(super) fn clear_llm_prune_marks_meta(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "DELETE FROM meta WHERE key=?1",
        params![LLM_PRUNE_MARKS_META_KEY],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

/// An outcome belongs only to the tool message with the same `tool_call_id`. After history replacement, compaction, or branch truncation,
/// side records that have lost their message owner are cleared immediately, so a deleted occurrence's state cannot pollute the retained history.
pub(super) fn prune_orphan_tool_execution_outcomes(conn: &Connection) -> io::Result<()> {
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
pub(super) fn drop_ambiguous_tool_execution_outcomes(conn: &Connection) -> io::Result<()> {
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
