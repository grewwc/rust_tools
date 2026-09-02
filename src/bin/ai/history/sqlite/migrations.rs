use std::io;

use rusqlite::Connection;

use super::connection::sqlite_error_kind;

pub(super) fn init_history_schema(conn: &Connection) -> Result<(), io::Error> {
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
