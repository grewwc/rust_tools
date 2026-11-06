use std::{error::Error, io, path::Path};

use super::super::types::Message;
use super::{
    connection::open_history_db,
    migrations::init_history_schema,
    store::{decode_message_content, decode_tool_calls},
};

/// Resolve a user-row index in Rust, then count its canonical prefix in the
/// same read snapshot. Only user content and origin metadata are decoded.
pub(in crate::ai) fn with_rewind_user_messages_sqlite<T>(
    path: &Path,
    resolve: impl FnOnce(&[Message]) -> Result<(usize, T), Box<dyn Error>>,
) -> Result<(usize, usize, T), Box<dyn Error>> {
    let mut conn = match open_history_db(path) {
        Ok(conn) => conn,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let (_, value) = resolve(&[])?;
            return Ok((0, 0, value));
        }
        Err(error) => return Err(error.into()),
    };
    init_history_schema(&conn)?;
    let tx = conn.transaction()?;
    let (ids, messages) = {
        let mut stmt = tx.prepare(
            "SELECT id, content, reasoning_content FROM messages
             WHERE role = 'user' ORDER BY id ASC",
        )?;
        let mut rows = stmt.query([])?;
        let mut ids = Vec::new();
        let mut messages = Vec::new();
        while let Some(row) = rows.next()? {
            ids.push(row.get::<_, i64>(0)?);
            let content: String = row.get(1)?;
            messages.push(Message {
                role: "user".to_string(),
                content: decode_message_content(&content),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: row.get(2)?,
            });
        }
        (ids, messages)
    };
    let (index, value) = resolve(&messages)?;
    let keep: i64 = tx.query_row(
        "SELECT COUNT(*) FROM messages WHERE id < ?1",
        [ids[index]],
        |row| row.get(0),
    )?;
    let total: i64 = tx.query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
    tx.commit()?;
    Ok((keep as usize, total as usize, value))
}

/// Project exactly the fields consumed by stale_patch_targets_from_messages.
/// Calls on every role and every tool result remain in canonical order.
pub(in crate::ai) fn read_rewind_ledger_messages_sqlite(
    path: &Path,
    keep: usize,
) -> Result<Vec<Message>, Box<dyn Error>> {
    if keep == 0 {
        return Ok(Vec::new());
    }
    let conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    // Limit the canonical prefix before filtering, including every survivor
    // when history has shrunk below the plan's keep count.
    let mut stmt = conn.prepare(
        "SELECT role, ledger_content, tool_calls, tool_call_id
         FROM (
             SELECT id, role, CASE WHEN role = 'tool' THEN content ELSE 'null' END AS ledger_content,
                    tool_calls, tool_call_id
             FROM messages ORDER BY id ASC LIMIT ?1
         )
         WHERE tool_calls IS NOT NULL OR role = 'tool'
         ORDER BY id ASC",
    )?;
    let mut rows = stmt.query([keep as i64])?;
    let mut messages = Vec::new();
    while let Some(row) = rows.next()? {
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        messages.push(Message {
            role: row.get(0)?,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id: row.get(3)?,
            reasoning_content: None,
        });
    }
    Ok(messages)
}
