use std::{
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

use crate::ai::types::ToolCall;

use super::{
    compress::{
        COMPRESSED_TOOL_EVIDENCE_MARKER, compact_persisted_history, is_summary_note_text,
        value_to_string,
    },
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

/// 用 SQLite Online Backup API 创建一致快照，并以原子替换的方式写入目标。
/// 直接复制 WAL 主文件会遗漏尚未 checkpoint 的页；backup API 会从 source 的同一
/// SQLite 快照读取主库和 WAL。主库替换成功后会移除旧侧车文件，避免旧 WAL/SHM
/// 与新主库混用。
pub(in crate::ai) fn backup_sqlite(source: &Path, target: &Path) -> io::Result<()> {
    if !source.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("SQLite source does not exist: {}", source.display()),
        ));
    }
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("history.sqlite");
    let temporary = parent.join(format!(".{file_name}.backup-{}.tmp", uuid::Uuid::new_v4()));

    let result = (|| {
        let source_conn = Connection::open(source).map_err(|e| io::Error::other(e.to_string()))?;
        source_conn
            .backup(rusqlite::DatabaseName::Main, &temporary, None)
            .map_err(|e| io::Error::other(e.to_string()))?;
        drop(source_conn);

        fs::rename(&temporary, target)?;
        remove_sqlite_sidecars(target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
        let _ = remove_sqlite_sidecars(&temporary);
    }
    result
}

fn remove_sqlite_sidecars(path: &Path) -> io::Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = PathBuf::from(format!("{}{}", path.display(), suffix));
        match fs::remove_file(sidecar) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
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

/// 廉价统计持久化消息的有效载荷大小，用于在 user turn 数尚少、但工具输出已经
/// 很大时仍能触发历史落盘压缩。不能以 sqlite 文件大小判断：WAL/空闲页不会在
/// 替换消息后立刻回收，会导致每轮都误判为超限。
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

/// 廉价统计已经折叠成 internal_note 的旧工具证据体积。它有独立于全局 history
/// 预算的内联上限，避免少量 user turn 下逐条证据在达到总预算前持续累积。
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
    // 读路径**不落盘**：`compact_persisted_history` 仅用于内存返回值，保持
    // `build_context_history` 现有的截断/归一行为不变。历史上这里会在 compacted
    // 与原始不同时 `replace_all_messages_sqlite` 覆盖库并 bump revision——那让
    // “读函数”产生写副作用：① 每次读都可能失效 `ContextHistoryCacheKey`（含
    // history_revision）造成缓存抖动；② 在 app-aware 压缩之前就用启发式摘要
    // 不可逆覆盖原文，使 >200 轮的会话被“双压”降保真。真正的落盘压缩由
    // `compact_session_history_with_app_inner` 专责，读路径恢复只读。
    let compacted = compact_persisted_history(messages);
    if history_count >= compacted.len() {
        return Ok(compacted);
    }
    Ok(compacted[compacted.len() - history_count..].to_vec())
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
        // 摘要前缀识别统一走 compress::is_summary_note_text（唯一真源）。
        // 此前这里硬编码 3 种前缀、漏掉 `长期记忆摘要（压缩保留）`，导致 fastpath
        // 找不到 overflow 路径产生的摘要接续点、每轮回退到全量慢路径重新压缩。
        if is_summary_note_text(text) {
            return Ok(Some(message));
        }
    }
    Ok(None)
}

/// 读取滑动窗口之前最近的 context checkpoint markers。它们是正文 asset 的唯一
/// 索引，不能因为 SQLite fast path 只加载 recent turns 而从请求上下文静默消失。
/// 请求正规化层仍会将最终投影限制为最近 8 条。
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

/// 把 history 中 context checkpoint marker 的 assets 路径重定位到新 session。
/// fork 时传入源 assets 目录做精确前缀替换；归档导入时源路径未知，会仅接受
/// `context-checkpoints/<file>` 的受控相对尾部，避免把普通文本或任意绝对路径改写。
pub(in crate::ai) fn remap_context_checkpoint_paths_sqlite(
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
    if meta
        .as_deref()
        .is_some_and(|prompt| !super::sessions::is_preserved_content_message(prompt))
    {
        return Ok(meta);
    }

    // 缓存的首条消息可能是图片/文本归档协议。继续向后查找第一条真实用户请求，
    // 避免过滤内部协议后把已有会话错误显示成 `new session`。
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
        if !super::sessions::is_preserved_content_message(&prompt) {
            prompts.push(prompt);
            if prompts.len() == 3 {
                break;
            }
        }
    }
    Ok((!prompts.is_empty()).then(|| prompts.join("\n---\n")))
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
    init_history_schema(&conn)?;
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
        assert!(
            r6 > r5,
            "clear should bump even after wiping meta: {r5} -> {r6}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn first_user_prompt_skips_preserved_content_notices() {
        let dir = std::env::temp_dir().join(format!(
            "first_prompt_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        append_history_sqlite(
            &path,
            vec![
                msg("user", "较早的用户图片内容已归档，原文未丢失。"),
                msg("user", r#"[[PRESERVED_CONTENT_STUB_V1]]{"kind":"image"}"#),
                msg("user", "较早的用户图片内容已归档，归档文件: /tmp/2"),
                msg("user", "较早的用户图片内容已归档，归档文件: /tmp/3"),
                msg("user", "较早的用户图片内容已归档，归档文件: /tmp/4"),
                msg("user", "这是实际用户请求"),
                msg("user", "这是后续用户请求"),
            ],
        )
        .unwrap();

        assert_eq!(
            read_first_user_prompt_sqlite(&path).unwrap().as_deref(),
            Some("这是实际用户请求\n---\n这是后续用户请求")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
