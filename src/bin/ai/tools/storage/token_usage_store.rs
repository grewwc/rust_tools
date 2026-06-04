//! LLM token 用量统计存储（独立 SQLite 表）。
//!
//! 审计的"采集"由 OS 层负责：内核 `LlmOps::llm_account` 在每次 LLM 调用结束时
//! 把用量追加进有界账本（见 `aios_kernel::primitives::LlmUsageRing`）。本模块是
//! 审计的"落库"侧：从内核账本 drain 出 [`LlmUsageRecord`] 并写入一张单独的表
//! `token_usage`，记录：
//!   - `created_at`     ：落库时间（Unix epoch 秒，即调用结束时刻）
//!   - `model`          ：模型名
//!   - `input_tokens`   ：输入 token（prompt_tokens）
//!   - `output_tokens`  ：输出 token（completion_tokens）
//!   - `total_tokens`   ：总 token（prompt + completion）
//!
//! 数据库默认放在 `~/.config/rust_tools/token_usage.db`，与 `agent_memory.db`
//! 同目录。连接放在全局 `LazyLock<Mutex<Connection>>` 单例里，避免与 `app.os`
//! 的 kernel 锁竞争。写入是 best-effort：失败仅打印 warning，不阻断主流程。
//!
//! 沿用仓库约定：没有 migrations 框架，统一 `CREATE TABLE IF NOT EXISTS`。
//! 支持按保留天数清理过旧数据（`cleanup_old`），写入路径会按一定频率自动触发。

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, params};

use aios_kernel::primitives::LlmUsageRecord;

use crate::ai::config_schema::AiConfig;
use crate::commonw::configw;

/// 默认保留天数：超过该天数的记录会在自动清理时删除。
const DEFAULT_RETAIN_DAYS: u64 = 90;
/// 每写入多少条触发一次自动清理（避免每次写都扫全表）。
const CLEANUP_EVERY_N_INSERTS: u64 = 100;

/// 自插入计数器，用于按频率触发自动清理。
static INSERT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// 已 drain 落库的内核账本游标（kernel `LlmUsageRecord::seq`）。
/// 调用方据此向内核 `llm_usage_drain_since(cursor)` 拿增量记录，落库成功后推进。
static DRAIN_CURSOR: AtomicU64 = AtomicU64::new(0);

/// 全局连接单例。`None` 表示初始化失败（路径不可写等），后续写入直接跳过。
static STORE: LazyLock<Option<Mutex<Connection>>> = LazyLock::new(|| match open_store() {
    Ok(conn) => Some(Mutex::new(conn)),
    Err(e) => {
        eprintln!("[TokenUsage] init failed, usage stats disabled: {e}");
        None
    }
});

/// 解析数据库文件路径：优先配置项 `ai.token_usage.db`，否则用默认路径。
fn db_path() -> PathBuf {
    let cfg = configw::get_all_config();
    let raw = cfg
        .get_opt(AiConfig::TOKEN_USAGE_DB)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "~/.config/rust_tools/token_usage.db".to_string());
    PathBuf::from(crate::commonw::utils::expanduser(raw.trim()).as_ref())
}

/// 打开并初始化数据库连接。
fn open_store() -> Result<Connection, String> {
    let path = db_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create token_usage db parent dir failed: {e}"))?;
    }
    let conn = Connection::open(&path)
        .map_err(|e| format!("open token_usage db at {}: {e}", path.display()))?;
    let _ = conn.pragma_update(None, "journal_mode", "WAL");
    let _ = conn.pragma_update(None, "synchronous", "NORMAL");
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS token_usage (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at    INTEGER NOT NULL,
            model         TEXT NOT NULL,
            input_tokens  INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            total_tokens  INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_token_usage_created_at ON token_usage(created_at);
        "#,
    )
    .map_err(|e| format!("init token_usage schema: {e}"))?;
    Ok(conn)
}

/// 是否启用 token 统计（默认开启，设为 false 关闭）。
fn enabled() -> bool {
    let cfg = configw::get_all_config();
    !cfg.get_opt(AiConfig::TOKEN_USAGE_ENABLE)
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 当前 drain 游标：调用方据此向内核拿增量账本记录。
pub(crate) fn drain_cursor() -> u64 {
    DRAIN_CURSOR.load(Ordering::Relaxed)
}

/// 把从内核 drain 出的账本记录批量落库。best-effort：失败仅 warning，不返回错误。
///
/// `new_head` 是内核账本当前 head seq（`llm_usage_head_seq()`）；落库成功后游标
/// 推进到该值，下次只 drain 新增记录。`records` 应为 `drain_since(drain_cursor())`
/// 的结果（升序、seq 严格大于旧游标）。
pub(crate) fn persist_drained(records: &[LlmUsageRecord], new_head: u64) {
    if !enabled() {
        // 关闭统计时也推进游标，避免重新开启后回放历史账本。
        DRAIN_CURSOR.store(new_head, Ordering::Relaxed);
        return;
    }
    let Some(store) = STORE.as_ref() else {
        return;
    };
    if records.is_empty() {
        DRAIN_CURSOR.store(new_head, Ordering::Relaxed);
        return;
    }
    let mut conn = match store.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let ts = now_secs() as i64;
    let tx = match conn.transaction() {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("[TokenUsage] begin tx failed: {e}");
            return;
        }
    };
    let mut inserted = 0u64;
    {
        let mut stmt = match tx.prepare_cached(
            "INSERT INTO token_usage (created_at, model, input_tokens, output_tokens, total_tokens) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[TokenUsage] prepare failed: {e}");
                return;
            }
        };
        for r in records {
            if let Err(e) = stmt.execute(params![
                ts,
                r.model,
                r.prompt_tokens as i64,
                r.completion_tokens as i64,
                r.total_tokens as i64,
            ]) {
                eprintln!("[TokenUsage] insert failed: {e}");
            } else {
                inserted += 1;
            }
        }
    }
    if let Err(e) = tx.commit() {
        eprintln!("[TokenUsage] commit failed: {e}");
        return;
    }
    // 落库成功，推进游标。
    DRAIN_CURSOR.store(new_head, Ordering::Relaxed);

    // 按频率触发自动清理，避免每次写入都扫全表。
    let n = INSERT_COUNTER.fetch_add(inserted, Ordering::Relaxed) + inserted;
    if inserted > 0 && n % CLEANUP_EVERY_N_INSERTS < inserted {
        let retain_days = configw::get_all_config()
            .get_opt(AiConfig::TOKEN_USAGE_RETAIN_DAYS)
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|d| *d > 0)
            .unwrap_or(DEFAULT_RETAIN_DAYS);
        cleanup_old_locked(&mut conn, retain_days);
    }
}

/// 删除早于 `retain_days` 天的记录（持有连接锁时调用）。
fn cleanup_old_locked(conn: &mut Connection, retain_days: u64) {
    let cutoff = now_secs().saturating_sub(retain_days * 86400);
    if let Err(e) = conn.execute(
        "DELETE FROM token_usage WHERE created_at < ?1",
        params![cutoff as i64],
    ) {
        eprintln!("[TokenUsage] cleanup failed: {e}");
    }
}

/// 一段时间窗口内的 token 用量合计。
#[derive(Debug, Clone, Default)]
pub(crate) struct UsageTotals {
    pub calls: u64,
    pub input: u64,
    pub output: u64,
    pub total: u64,
}

/// 按模型聚合的一行用量。
#[derive(Debug, Clone)]
pub(crate) struct UsageByModel {
    pub model: String,
    pub calls: u64,
    pub input: u64,
    pub output: u64,
    pub total: u64,
}

/// 查询某时间窗口内的总用量。`window_secs=None` 表示全部历史；
/// 否则统计最近 `window_secs` 秒。`None` 返回值表示存储不可用。
pub(crate) fn query_totals(window_secs: Option<u64>) -> Option<UsageTotals> {
    let store = STORE.as_ref()?;
    let conn = match store.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let cutoff = window_secs.map(|w| now_secs().saturating_sub(w) as i64);
    let sql = "SELECT COUNT(*), \
               COALESCE(SUM(input_tokens),0), \
               COALESCE(SUM(output_tokens),0), \
               COALESCE(SUM(total_tokens),0) \
               FROM token_usage WHERE (?1 IS NULL OR created_at >= ?1)";
    conn.query_row(sql, params![cutoff], |row| {
        Ok(UsageTotals {
            calls: row.get::<_, i64>(0)? as u64,
            input: row.get::<_, i64>(1)? as u64,
            output: row.get::<_, i64>(2)? as u64,
            total: row.get::<_, i64>(3)? as u64,
        })
    })
    .ok()
}

/// 查询某时间窗口内按模型聚合的用量，按总 token 降序。
pub(crate) fn query_by_model(window_secs: Option<u64>) -> Option<Vec<UsageByModel>> {
    let store = STORE.as_ref()?;
    let conn = match store.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let cutoff = window_secs.map(|w| now_secs().saturating_sub(w) as i64);
    let sql = "SELECT model, COUNT(*), \
               COALESCE(SUM(input_tokens),0), \
               COALESCE(SUM(output_tokens),0), \
               COALESCE(SUM(total_tokens),0) \
               FROM token_usage WHERE (?1 IS NULL OR created_at >= ?1) \
               GROUP BY model ORDER BY 5 DESC";
    let mut stmt = conn.prepare(sql).ok()?;
    let rows = stmt
        .query_map(params![cutoff], |row| {
            Ok(UsageByModel {
                model: row.get::<_, String>(0)?,
                calls: row.get::<_, i64>(1)? as u64,
                input: row.get::<_, i64>(2)? as u64,
                output: row.get::<_, i64>(3)? as u64,
                total: row.get::<_, i64>(4)? as u64,
            })
        })
        .ok()?;
    Some(rows.filter_map(|r| r.ok()).collect())
}

/// 数据库文件路径（供 `/usage` 展示）。
pub(crate) fn store_path() -> PathBuf {
    db_path()
}

/// 是否启用（供 `/usage` 展示）。
pub(crate) fn is_enabled() -> bool {
    enabled()
}
