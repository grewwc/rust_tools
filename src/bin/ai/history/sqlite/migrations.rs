use std::io;

use rusqlite::Connection;

use super::connection::sqlite_error_kind;

/// `PRAGMA user_version` value written after every table and column this module
/// creates exists. Bump it whenever a change here adds or alters schema; an
/// existing database then runs the migration once more.
pub(super) const HISTORY_SCHEMA_VERSION: i32 = 2;

pub(super) fn init_history_schema(conn: &Connection) -> Result<(), io::Error> {
    // Several operations run per turn and each one opens (or reuses) a
    // connection. Replaying ten CREATE IF NOT EXISTS statements, five failing
    // ALTER TABLE probes, and a sqlite_master query on every call costs more
    // than the actual work; the version marker is the fast path.
    //
    // The marker is only trusted when it matches exactly: a lower value means a
    // database written by an older build, and a higher one means a database this
    // build does not understand. Both replay the additive DDL so a partially
    // restored file still gets its tables back, at the price of not trusting a
    // marker that a manual edit could have left behind.
    let version = history_schema_version(conn)?;
    if version == HISTORY_SCHEMA_VERSION {
        return Ok(());
    }
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
        CREATE TABLE IF NOT EXISTS context_snapshot_history (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_message_id INTEGER NOT NULL,
            projection_fingerprint TEXT NOT NULL,
            payload BLOB,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE INDEX IF NOT EXISTS idx_context_snapshot_history_watermark
            ON context_snapshot_history(source_message_id);
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
    add_column_if_missing(conn, "context_snapshot_history", "payload", "BLOB")?;
    migrate_context_snapshot_ring_payloads(conn, version)?;
    drop_legacy_previous_snapshot_tables(conn)?;
    // Never lower a marker written by a newer build; its migrations have run and
    // this build cannot tell what they changed.
    if version < HISTORY_SCHEMA_VERSION {
        conn.pragma_update(None, "user_version", HISTORY_SCHEMA_VERSION)
            .map_err(|error| io::Error::new(sqlite_error_kind(&error), error.to_string()))?;
    }
    Ok(())
}

fn history_schema_version(conn: &Connection) -> Result<i32, io::Error> {
    conn.query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|error| io::Error::new(sqlite_error_kind(&error), error.to_string()))
}

/// Ring entries written before v2 kept their messages in
/// `context_messages_history`, one row per message per snapshot. v2 stores one
/// compressed payload on the entry itself. A pre-v2 entry has no payload and
/// cannot be promoted, and the ring is only a rewind cache rebuilt on demand, so
/// drop those entries together with the retired table.
fn migrate_context_snapshot_ring_payloads(
    conn: &Connection,
    version: i32,
) -> Result<(), io::Error> {
    if version >= 2 {
        return Ok(());
    }
    conn.execute_batch(
        "DELETE FROM context_snapshot_history;
         DROP TABLE IF EXISTS context_messages_history;",
    )
    .map_err(|error| io::Error::other(error.to_string()))
}

/// Older builds kept one `_previous` snapshot pair as a rewind fallback; the
/// bounded snapshot ring replaces it. Drop the pair only while it still exists:
/// an unconditional `DROP TABLE IF EXISTS` on the hot read path would attempt a
/// schema write on every open for databases created before the ring existed.
fn drop_legacy_previous_snapshot_tables(conn: &Connection) -> Result<(), io::Error> {
    let legacy_tables: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table'
               AND name IN ('context_snapshot_previous', 'context_messages_previous')",
            [],
            |row| row.get(0),
        )
        .map_err(|error| io::Error::new(sqlite_error_kind(&error), error.to_string()))?;
    if legacy_tables == 0 {
        return Ok(());
    }
    conn.execute_batch(
        "DROP TABLE IF EXISTS context_messages_previous;
         DROP TABLE IF EXISTS context_snapshot_previous;",
    )
    .map_err(|error| io::Error::other(error.to_string()))
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
