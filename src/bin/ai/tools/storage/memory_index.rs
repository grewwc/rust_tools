//! SQLite + FTS5 索引层 (方案 B)。
//!
//! JSONL 仍是事实来源；本模块给 `agent_memory.jsonl` 旁挂一份 SQLite 索引：
//!   - `entries`        ：AgentMemoryEntry 的结构化镜像 + `hits / last_accessed` LFU 计数
//!   - `entries_fts`    ：基于 FTS5 的全文倒排（虚拟表，content=entries），毫秒级关键词召回
//!   - `meta`           ：`schema_version` / `source_mtime` / `source_len` 一致性元信息
//!
//! 启动时调 [`MemoryIndex::open_or_init`] —— 文件 mtime/len 与上次记录一致就立即可用，
//! 否则把整个 JSONL 重新摄入（rebuild）。重建发生在调用方线程内，因为只有这条路径
//! 能保证后续 search/recent 不返回脏数据。daemon 异步重建留给 `record_drift_async`
//! 在写入路径里使用：JSONL 已经更新但本进程懒得阻塞时排队后台修复。
//!
//! 设计要点
//!   - 永远不在持有 `with_memory_file_lock` 之外的状态下做"读 JSONL → 写 DB"，
//!     避免与 enforce/rotate/append 并发。
//!   - JSONL 写失败时绝不写 DB；DB 写失败时记录 trace event，不阻断 JSONL 主路径
//!     —— 保持 "JSONL = source of truth, DB = best-effort cache" 的约束。
//!   - 没有 `PRAGMA user_version` / migrations 框架，沿用仓库其它处的
//!     `CREATE TABLE IF NOT EXISTS` 模式 + `meta` 表里自管 `schema_version`。

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use rusqlite::{Connection, OptionalExtension, params};

use super::memory_store::AgentMemoryEntry;

/// schema 版本，每次破坏性 schema 变更 +1，触发全量重建
const SCHEMA_VERSION: i64 = 1;

#[derive(Debug)]
pub(crate) struct MemoryIndex {
    /// 索引文件路径，例如 `~/.config/rust_tools/agent_memory.db`
    db_path: PathBuf,
    /// JSONL 事实来源路径，例如 `~/.config/rust_tools/agent_memory.jsonl`
    source_path: PathBuf,
    /// SQLite 连接；放在 Mutex 里以支持多线程并发读写
    /// （rusqlite::Connection 不是 Send + Sync, 但 Mutex<Connection> 是）。
    conn: Mutex<Connection>,
}

impl MemoryIndex {
    /// 打开 / 初始化索引；schema 不一致或 source 文件 mtime/len 与记录漂移时
    /// **同步**重建。调用方应在持有 `with_memory_file_lock` 之内调用。
    pub(crate) fn open_or_init(db_path: PathBuf, source_path: PathBuf) -> Result<Self, String> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create db parent dir failed: {e}"))?;
        }
        let conn = Connection::open(&db_path)
            .map_err(|e| format!("open sqlite at {}: {e}", db_path.display()))?;
        // WAL 让"主进程写 + 后台 daemon 读"并发更友好。
        let _ = conn.pragma_update(None, "journal_mode", "WAL");
        let _ = conn.pragma_update(None, "synchronous", "NORMAL");

        Self::init_schema(&conn)?;

        let me = Self {
            db_path,
            source_path,
            conn: Mutex::new(conn),
        };
        // 一致性检查 + 必要时重建
        if me.is_drifted()? {
            me.rebuild_from_source()?;
        }
        Ok(me)
    }

    fn init_schema(conn: &Connection) -> Result<(), String> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS entries (
                id            TEXT PRIMARY KEY,
                timestamp     TEXT NOT NULL,
                category      TEXT NOT NULL,
                note          TEXT NOT NULL,
                tags_json     TEXT NOT NULL DEFAULT '[]',
                source        TEXT,
                priority      INTEGER NOT NULL DEFAULT 100,
                owner_pid     INTEGER,
                owner_pgid    INTEGER,
                hits          INTEGER NOT NULL DEFAULT 0,
                last_access   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_entries_cat       ON entries(category);
            CREATE INDEX IF NOT EXISTS idx_entries_prio_ts   ON entries(priority DESC, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_entries_ts        ON entries(timestamp DESC);

            -- contentless FTS：避免和 entries 表内容双倍存储；我们手动维护写入。
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                id UNINDEXED,
                category,
                note,
                tags,
                tokenize = 'unicode61'
            );

            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            );
            "#,
        )
        .map_err(|e| format!("init schema: {e}"))?;

        // schema_version 不匹配时清空所有表，触发后续 rebuild
        let stored: Option<String> = conn
            .query_row("SELECT v FROM meta WHERE k = 'schema_version'", [], |r| {
                r.get(0)
            })
            .optional()
            .map_err(|e| format!("read schema_version: {e}"))?;
        let stored_v = stored
            .as_deref()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(-1);
        if stored_v != SCHEMA_VERSION {
            conn.execute_batch("DELETE FROM entries; DELETE FROM entries_fts; DELETE FROM meta;")
                .map_err(|e| format!("clear stale schema: {e}"))?;
            conn.execute(
                "INSERT INTO meta(k,v) VALUES ('schema_version', ?1)",
                params![SCHEMA_VERSION.to_string()],
            )
            .map_err(|e| format!("write schema_version: {e}"))?;
        }
        Ok(())
    }

    /// 比较 source JSONL 的 mtime/len 与 meta 中记录的值；任一不一致即视为漂移。
    fn is_drifted(&self) -> Result<bool, String> {
        let (mtime, len) = read_source_signature(&self.source_path);
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("conn lock poisoned: {e}"))?;
        let prev_mtime: Option<String> = conn
            .query_row("SELECT v FROM meta WHERE k='source_mtime'", [], |r| {
                r.get(0)
            })
            .optional()
            .map_err(|e| format!("read source_mtime: {e}"))?;
        let prev_len: Option<String> = conn
            .query_row("SELECT v FROM meta WHERE k='source_len'", [], |r| r.get(0))
            .optional()
            .map_err(|e| format!("read source_len: {e}"))?;
        Ok(prev_mtime.as_deref() != Some(mtime.as_str())
            || prev_len.as_deref() != Some(len.to_string().as_str()))
    }

    fn write_source_signature(&self) -> Result<(), String> {
        let (mtime, len) = read_source_signature(&self.source_path);
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("conn lock poisoned: {e}"))?;
        conn.execute(
            "INSERT INTO meta(k,v) VALUES('source_mtime', ?1)
             ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            params![mtime],
        )
        .map_err(|e| format!("write source_mtime: {e}"))?;
        conn.execute(
            "INSERT INTO meta(k,v) VALUES('source_len', ?1)
             ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            params![len.to_string()],
        )
        .map_err(|e| format!("write source_len: {e}"))?;
        Ok(())
    }

    /// 读 JSONL 全量摄入，事务内完成。任何中途失败都回滚。
    pub(crate) fn rebuild_from_source(&self) -> Result<usize, String> {
        let content = match std::fs::read_to_string(&self.source_path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(format!("read source jsonl: {e}")),
        };
        let entries: Vec<AgentMemoryEntry> = content
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() {
                    return None;
                }
                serde_json::from_str::<AgentMemoryEntry>(line).ok()
            })
            .collect();

        let mut conn = self
            .conn
            .lock()
            .map_err(|e| format!("conn lock poisoned: {e}"))?;
        let tx = conn.transaction().map_err(|e| format!("begin tx: {e}"))?;
        tx.execute("DELETE FROM entries", [])
            .map_err(|e| format!("clear entries: {e}"))?;
        tx.execute("DELETE FROM entries_fts", [])
            .map_err(|e| format!("clear entries_fts: {e}"))?;

        for entry in &entries {
            // 没有 id 的条目无法稳定建索引——跳过；search 路径会回退扫 JSONL。
            let Some(id) = entry.id.as_deref() else {
                continue;
            };
            let tags_json = serde_json::to_string(&entry.tags).unwrap_or_else(|_| "[]".into());
            let tags_text = entry.tags.join(" ");
            let prio = entry.priority.unwrap_or(100) as i64;
            tx.execute(
                "INSERT OR REPLACE INTO entries
                 (id,timestamp,category,note,tags_json,source,priority,owner_pid,owner_pgid,hits,last_access)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,
                         COALESCE((SELECT hits FROM entries WHERE id=?1),0),
                         COALESCE((SELECT last_access FROM entries WHERE id=?1),0))",
                params![
                    id,
                    entry.timestamp,
                    entry.category,
                    entry.note,
                    tags_json,
                    entry.source,
                    prio,
                    entry.owner_pid.map(|v| v as i64),
                    entry.owner_pgid.map(|v| v as i64),
                ],
            )
            .map_err(|e| format!("insert entries: {e}"))?;
            tx.execute(
                "INSERT INTO entries_fts(id,category,note,tags) VALUES(?1,?2,?3,?4)",
                params![id, entry.category, entry.note, tags_text],
            )
            .map_err(|e| format!("insert fts: {e}"))?;
        }

        tx.commit().map_err(|e| format!("commit rebuild: {e}"))?;
        drop(conn);

        self.write_source_signature()?;
        Ok(entries.len())
    }

    /// 单条增量写入 —— 应当在 `MemoryStore::append` 成功 fsync JSONL 后调用。
    pub(crate) fn upsert_entry(&self, entry: &AgentMemoryEntry) -> Result<(), String> {
        let Some(id) = entry.id.as_deref() else {
            return Ok(()); // 无 id 直接放弃索引，与 rebuild 一致
        };
        let tags_json = serde_json::to_string(&entry.tags).unwrap_or_else(|_| "[]".into());
        let tags_text = entry.tags.join(" ");
        let prio = entry.priority.unwrap_or(100) as i64;
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("conn lock poisoned: {e}"))?;
        conn.execute(
            "INSERT INTO entries
             (id,timestamp,category,note,tags_json,source,priority,owner_pid,owner_pgid)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
             ON CONFLICT(id) DO UPDATE SET
                timestamp=excluded.timestamp,
                category=excluded.category,
                note=excluded.note,
                tags_json=excluded.tags_json,
                source=excluded.source,
                priority=excluded.priority,
                owner_pid=excluded.owner_pid,
                owner_pgid=excluded.owner_pgid",
            params![
                id,
                entry.timestamp,
                entry.category,
                entry.note,
                tags_json,
                entry.source,
                prio,
                entry.owner_pid.map(|v| v as i64),
                entry.owner_pgid.map(|v| v as i64),
            ],
        )
        .map_err(|e| format!("upsert entries: {e}"))?;
        // FTS5 做先 delete-then-insert：contentless 配置下 ON CONFLICT 行为不友好。
        conn.execute("DELETE FROM entries_fts WHERE id=?1", params![id])
            .map_err(|e| format!("delete fts: {e}"))?;
        conn.execute(
            "INSERT INTO entries_fts(id,category,note,tags) VALUES(?1,?2,?3,?4)",
            params![id, entry.category, entry.note, tags_text],
        )
        .map_err(|e| format!("insert fts: {e}"))?;
        drop(conn);
        // signature 在 append 路径里由 MemoryStore 统一刷新，这里不刷
        Ok(())
    }

    pub(crate) fn delete_id(&self, id: &str) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("conn lock poisoned: {e}"))?;
        conn.execute("DELETE FROM entries WHERE id=?1", params![id])
            .map_err(|e| format!("delete entries: {e}"))?;
        conn.execute("DELETE FROM entries_fts WHERE id=?1", params![id])
            .map_err(|e| format!("delete fts: {e}"))?;
        Ok(())
    }

    pub(crate) fn refresh_signature(&self) -> Result<(), String> {
        self.write_source_signature()
    }

    /// FTS5 全文检索；返回按 `bm25(entries_fts) + priority + recency + LFU` 综合排序后的 id 列表。
    /// caller 拿 id 之后用 `MemoryStore::all()` 或后续传 id 的精确读取拿到完整条目。
    pub(crate) fn search_ids(&self, query: &str, limit: usize) -> Result<Vec<String>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("conn lock poisoned: {e}"))?;
        let escaped = sanitize_fts_query(query);
        if escaped.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = conn
            .prepare(
                "SELECT e.id
                 FROM entries_fts f
                 JOIN entries e ON e.id = f.id
                 WHERE entries_fts MATCH ?1
                 ORDER BY
                   (-bm25(entries_fts)) * 1.0
                   + (e.priority / 50.0)
                   + (e.hits * 0.05)
                 DESC
                 LIMIT ?2",
            )
            .map_err(|e| format!("prepare search: {e}"))?;
        let ids = stmt
            .query_map(params![escaped, limit as i64], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| format!("exec search: {e}"))?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        Ok(ids)
    }

    /// 把命中条目的 hits 计数 +=1, last_access 设为当前秒数（unix）。
    /// 这是 LFU 的核心；JSONL 端无法做原地 update，只在 SQLite 维护。
    pub(crate) fn record_hits(&self, ids: &[String]) -> Result<(), String> {
        if ids.is_empty() {
            return Ok(());
        }
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let mut conn = self
            .conn
            .lock()
            .map_err(|e| format!("conn lock poisoned: {e}"))?;
        let tx = conn.transaction().map_err(|e| format!("begin tx: {e}"))?;
        for id in ids {
            tx.execute(
                "UPDATE entries SET hits = hits + 1, last_access = ?2 WHERE id = ?1",
                params![id, now],
            )
            .map_err(|e| format!("record hit: {e}"))?;
        }
        tx.commit().map_err(|e| format!("commit hits: {e}"))?;
        Ok(())
    }

    /// 取一组 entry 的 LFU 计数（搜索辅助；目前未广泛使用，预留给 GC 评分）。
    #[allow(dead_code)]
    pub(crate) fn hits_for(&self, id: &str) -> Result<i64, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("conn lock poisoned: {e}"))?;
        conn.query_row("SELECT hits FROM entries WHERE id=?1", params![id], |r| {
            r.get::<_, i64>(0)
        })
        .optional()
        .map_err(|e| format!("query hits: {e}"))
        .map(|v| v.unwrap_or(0))
    }

    pub(crate) fn db_path(&self) -> &Path {
        &self.db_path
    }
}

/// FTS5 MATCH 输入需要做基本转义；这里只允许字母数字 / 下划线 / CJK 字符 +
/// 空白，其它都视为分隔符。我们最终把 token 用空格连起来作为隐式 OR 查询。
fn sanitize_fts_query(q: &str) -> String {
    let mut out = String::new();
    let mut in_token = false;
    for ch in q.chars() {
        let keep = ch.is_alphanumeric() || ch == '_';
        if keep {
            out.push(ch);
            in_token = true;
        } else if in_token {
            out.push(' ');
            in_token = false;
        }
    }
    let trimmed = out.trim().to_string();
    if trimmed.is_empty() {
        return String::new();
    }
    // 给每个 token 加 `*` 让短前缀也能命中（FTS5 prefix search）。
    trimmed
        .split_whitespace()
        .map(|t| format!("{}*", t))
        .collect::<Vec<_>>()
        .join(" ")
}

fn read_source_signature(path: &Path) -> (String, u64) {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return (String::from("absent"), 0),
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos().to_string())
        .unwrap_or_else(|| String::from("0"));
    (mtime, meta.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_paths(tag: &str) -> (PathBuf, PathBuf) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir();
        (
            dir.join(format!("rt_memidx_{tag}_{nanos}.jsonl")),
            dir.join(format!("rt_memidx_{tag}_{nanos}.db")),
        )
    }

    fn entry(id: &str, cat: &str, note: &str, prio: u8) -> AgentMemoryEntry {
        AgentMemoryEntry {
            id: Some(id.to_string()),
            timestamp: "2025-03-01T00:00:00Z".to_string(),
            category: cat.to_string(),
            note: note.to_string(),
            tags: vec![],
            source: None,
            priority: Some(prio),
            owner_pid: None,
            owner_pgid: None,
        }
    }

    fn write_jsonl(path: &Path, entries: &[AgentMemoryEntry]) {
        let mut buf = String::new();
        for e in entries {
            buf.push_str(&serde_json::to_string(e).unwrap());
            buf.push('\n');
        }
        std::fs::write(path, buf).unwrap();
    }

    #[test]
    fn rebuild_then_search_returns_matching_ids() {
        let (jsonl, db) = unique_paths("rebuild");
        write_jsonl(
            &jsonl,
            &[
                entry(
                    "e1",
                    "user_preference",
                    "prefer concise commit messages",
                    210,
                ),
                entry("e2", "project_memory", "the build script is build.sh", 180),
                entry("e3", "tool_stat", "ripgrep used 5 times", 50),
            ],
        );
        let idx = MemoryIndex::open_or_init(db.clone(), jsonl.clone()).unwrap();
        let hits = idx.search_ids("commit", 10).unwrap();
        assert!(hits.contains(&"e1".to_string()));
        let hits2 = idx.search_ids("build", 10).unwrap();
        assert!(hits2.contains(&"e2".to_string()));
        let _ = std::fs::remove_file(&jsonl);
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn record_hits_increments_lfu_counter() {
        let (jsonl, db) = unique_paths("hits");
        write_jsonl(
            &jsonl,
            &[entry("h1", "user_preference", "remember to lint", 210)],
        );
        let idx = MemoryIndex::open_or_init(db.clone(), jsonl.clone()).unwrap();
        idx.record_hits(&["h1".to_string()]).unwrap();
        idx.record_hits(&["h1".to_string()]).unwrap();
        assert_eq!(idx.hits_for("h1").unwrap(), 2);
        let _ = std::fs::remove_file(&jsonl);
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn drift_triggers_rebuild() {
        let (jsonl, db) = unique_paths("drift");
        write_jsonl(&jsonl, &[entry("d1", "general", "first version", 100)]);
        let idx = MemoryIndex::open_or_init(db.clone(), jsonl.clone()).unwrap();
        assert!(!idx.search_ids("first", 5).unwrap().is_empty());
        // 直接修改 JSONL 模拟外部进程写入
        std::thread::sleep(std::time::Duration::from_millis(10));
        write_jsonl(
            &jsonl,
            &[
                entry("d1", "general", "first version", 100),
                entry("d2", "general", "totally new content", 100),
            ],
        );
        let idx2 = MemoryIndex::open_or_init(db.clone(), jsonl.clone()).unwrap();
        assert!(!idx2.search_ids("totally", 5).unwrap().is_empty());
        let _ = std::fs::remove_file(&jsonl);
        let _ = std::fs::remove_file(&db);
    }
}
