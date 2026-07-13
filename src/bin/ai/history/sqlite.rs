use std::{fs, io, path::Path, time::Duration};

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

use crate::ai::types::ToolCall;

use super::{
    compress::{compact_persisted_history, value_to_string},
    types::{MAX_HISTORY_TURNS, Message, ROLE_INTERNAL_NOTE},
};

pub(in crate::ai) struct RecentTurnWindow {
    pub(in crate::ai) messages: Vec<Message>,
    pub(in crate::ai) start_message_id: Option<i64>,
    pub(in crate::ai) has_older_messages: bool,
}

fn open_history_db(path: &Path) -> Result<Connection, io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path).map_err(|e| io::Error::other(e.to_string()))?;
    conn.busy_timeout(Duration::from_secs(5))
        .map_err(|e| io::Error::other(e.to_string()))?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(|e| io::Error::other(e.to_string()))?;
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
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );",
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    add_column_if_missing(conn, "messages", "tool_calls", "TEXT")?;
    add_column_if_missing(conn, "messages", "tool_call_id", "TEXT")?;
    add_column_if_missing(conn, "messages", "reasoning_content", "TEXT")?;
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
        Err(err) => Err(io::Error::other(err.to_string())),
    }
}

/// 读取 history DB 的写入版本号（存于 meta 表 key='history_revision'）。
/// 每次消息写入/删除/替换都会在同一事务内 `bump_history_revision` 递增该值，
/// 因此它是一个**跨连接**单调递增的全局信号，可靠地反映"库内容是否变化"。
///
/// 不能用 `PRAGMA data_version` 代替：它是**连接局部**的比较值——每个新开的
/// `Connection` 只把它当作"自本连接打开以来是否被其他连接改过"的基准，新连接
/// 读到的初值不随外部写入而变（实测新连接恒返回 2），因此无法作为跨连接缓存
/// 失效依据。缺失（老库尚未写入过 revision）时返回 0，与"从未修改"一致。
pub(in crate::ai) fn read_history_revision(path: &Path) -> Option<i64> {
    let conn = Connection::open(path).ok()?;
    // meta 表可能尚未创建（全新库）或尚未写入过 revision：两种情况都视为 0
    // （"从未修改"），保证返回值稳定可比。只有连接本身打不开才返回 None。
    let value: Option<i64> = conn
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key='history_revision' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    Some(value.unwrap_or(0))
}

/// 在写事务内递增 `meta.history_revision`。所有会改变 messages 内容的写路径
/// （append / replace / clear / truncate）都必须调用它，使 `read_history_revision`
/// 能被跨连接观察到变化。
fn bump_history_revision(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('history_revision', '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
        [],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

/// 对 SQLite 数据库执行 WAL checkpoint（TRUNCATE），把 -wal 日志合并进主库。
/// 用于导出归档前确保主库文件包含全部数据，避免跨机器迁移时丢失未 checkpoint 的 WAL。
pub(in crate::ai) fn checkpoint_wal(path: &Path) -> io::Result<()> {
    let conn = Connection::open(path).map_err(|e| io::Error::other(e.to_string()))?;
    conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

/// 廉价查询当前 history DB 中 role='user' 的消息数。
/// 用于 boundary compact 在 hot path 上"先 count 再决定是否全量读"，
/// 避免每个 turn 收尾都把几万条消息（含大块 tool 输出）反序列化一遍。
pub(in crate::ai) fn count_user_turns_sqlite(path: &Path) -> io::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let conn = open_history_db(path)?;
    // schema 可能尚未创建（全新 session），messages 表不存在时直接返 0。
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

pub(in crate::ai) fn append_history_sqlite(path: &Path, entries: Vec<Message>) -> io::Result<()> {
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
        insert_messages(&tx, entries)?;
    }
    let user_turns: i64 = tx
        .query_row(
            "SELECT COUNT(1) FROM messages WHERE role = 'user'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    if user_turns.max(0) as usize <= MAX_HISTORY_TURNS {
        bump_history_revision(&tx)?;
        return tx.commit().map_err(|e| io::Error::other(e.to_string()));
    }
    let messages = read_messages_with_sql(
        &tx,
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         ORDER BY id ASC",
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    let compacted = compact_persisted_history(messages.clone());
    if compacted != messages {
        tx.execute("DELETE FROM messages", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        insert_messages(&tx, compacted)?;
    }
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

pub(in crate::ai) fn append_history_sqlite_uncompacted(
    path: &Path,
    entries: Vec<Message>,
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
        insert_messages(&tx, entries)?;
    }
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

pub(in crate::ai) fn replace_all_messages_sqlite(
    path: &Path,
    messages: &[Message],
) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM messages", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM meta WHERE key='first_user_prompt'", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    insert_messages(&tx, messages.to_vec())?;
    refresh_first_user_prompt_meta(&tx, messages)?;
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

fn insert_messages(conn: &Connection, messages: Vec<Message>) -> io::Result<()> {
    let mut stmt = conn
        .prepare(
            "INSERT INTO messages (role, content, tool_calls, tool_call_id, reasoning_content)
             VALUES (?1, ?2, ?3, ?4, ?5)",
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
            message.reasoning_content
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
        if text.starts_with("历史摘要（自动压缩")
            || text.starts_with("对话摘要（自动压缩")
            || text.starts_with("[mid-turn-summary]")
        {
            return Ok(Some(message));
        }
    }
    Ok(None)
}

pub(in crate::ai) fn clear_session_history_sqlite(path: &Path) -> io::Result<()> {
    let mut conn = match open_history_db(path) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    init_history_schema(&conn)?;
    // 事务包裹：DELETE messages / DELETE meta / bump revision 必须原子提交，
    // 否则中途崩溃会留下"messages 已清空但 revision 未变"的不一致状态，
    // 导致 context 缓存误判为未变化而继续供应旧历史。
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM messages", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    // 保留 history_revision 行：它是缓存失效计数器，须跨 clear **单调递增**。
    // 若连同它一起删掉，bump 会从 1 重新开始，版本号回退后可能与早期缓存
    // 条目的 revision 撞车，反而让已失效的旧历史被误命中。
    tx.execute("DELETE FROM meta WHERE key != 'history_revision'", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

/// 把 messages 表保留到前 `keep` 条（按 id 升序）。用于 session branch：
/// 复制完整 sqlite 后再回滚到指定消息数。`keep == 0` 等价于 clear。
pub(in crate::ai) fn truncate_messages_sqlite(path: &Path, keep: usize) -> io::Result<()> {
    let mut conn = match open_history_db(path) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    init_history_schema(&conn)?;
    // 事务包裹：DELETE + bump revision 原子提交，避免中途崩溃留下
    // "已删消息但 revision 未变"的不一致状态（context 缓存供应错误的空结果）。
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    if keep == 0 {
        tx.execute("DELETE FROM messages", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        bump_history_revision(&tx)?;
        return tx.commit().map_err(|e| io::Error::other(e.to_string()));
    }
    // 取前 `keep` 条的最大 id，删掉其后的所有行。
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
    }
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

pub(in crate::ai) fn read_first_user_prompt_sqlite(path: &Path) -> io::Result<Option<String>> {
    let conn = open_history_db(path)?;
    let meta: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='first_user_prompt' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    if meta.is_some() {
        return Ok(meta);
    }
    // 读取前 3 条用户消息，用于生成更完整的摘要
    let fallback: Vec<String> = conn
        .prepare("SELECT content FROM messages WHERE role='user' ORDER BY id ASC LIMIT 3")
        .map_err(|e| io::Error::other(e.to_string()))?
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| io::Error::other(e.to_string()))?
        .filter_map(|r| r.ok())
        .collect();
    if fallback.is_empty() {
        return Ok(None);
    }
    // 合并前几条消息的内容
    let combined: Vec<String> = fallback
        .iter()
        .map(|content| value_to_string(&decode_message_content(content)))
        .collect();
    Ok(Some(combined.join("\n---\n")))
}

/// 读取 LLM 生成的 session 标题（存储在 meta 表中，key='session_title'）。
pub(in crate::ai) fn read_session_title_sqlite(path: &Path) -> io::Result<Option<String>> {
    let conn = open_history_db(path)?;
    let title: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='session_title' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    Ok(title.filter(|s| !s.trim().is_empty()))
}

/// 写入 LLM 生成的 session 标题到 meta 表。
pub(in crate::ai) fn write_session_title_sqlite(path: &Path, title: &str) -> io::Result<()> {
    let conn = open_history_db(path)?;
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value, created_at) VALUES ('session_title', ?1, unixepoch())",
        rusqlite::params![title],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
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

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    /// P1 回归：`read_history_revision` 必须**跨连接**观察到写入递增。
    /// 每次写路径都开新连接读版本号，模拟 build_context_history 的缓存失效判定。
    /// 旧实现用连接局部的 `PRAGMA data_version`，新连接恒返回固定值，无法失效缓存。
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

        // 全新库尚未写入过 revision：视为 0（"从未修改"）。
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

        // clear 会 DELETE meta 后再 bump，结构与其它写路径不同，需单独覆盖。
        append_history_sqlite(&path, vec![msg("user", "again")]).unwrap();
        let r5 = read_history_revision(&path).unwrap();
        clear_session_history_sqlite(&path).unwrap();
        let r6 = read_history_revision(&path).unwrap();
        assert!(r6 > r5, "clear should bump even after wiping meta: {r5} -> {r6}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
