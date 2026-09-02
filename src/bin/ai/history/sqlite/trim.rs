use std::{
    io,
    path::{Path, PathBuf},
};

use rusqlite::{OptionalExtension, params};
use serde_json::Value;

use super::super::types::ROLE_INTERNAL_NOTE;
use super::connection::open_history_db;
use super::lock::with_session_state_lock;
use super::migrations::init_history_schema;
use super::outcomes::{
    clear_llm_prune_marks_meta, drop_ambiguous_tool_execution_outcomes,
    prune_orphan_tool_execution_outcomes,
};
use super::revision::{bump_history_revision, invalidate_context_snapshot};
use super::store::decode_message_content;

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
