//! LLM token usage stats storage (separate SQLite table), with cost and per-day trends.
//!
//! The "collection" side of auditing is handled by the OS layer: the kernel's `LlmOps::llm_account`
//! appends usage to a bounded ledger at the end of every LLM call (see
//! `aios_kernel::primitives::LlmUsageRing`). This module is the "persistence" side of auditing:
//! it drains [`LlmUsageRecord`] entries from the kernel ledger into a separate `token_usage`
//! table, recording:
//!   - `created_at`     : persistence time (Unix epoch seconds, i.e. when the call ended)
//!   - `model`          : model name
//!   - `input_tokens`   : input tokens (prompt_tokens)
//!   - `output_tokens`  : output tokens (completion_tokens)
//!   - `total_tokens`   : total tokens (prompt + completion)
//!
//! The database defaults to `~/.config/rust_tools/token_usage.db`, in the same directory as
//! `agent_memory.db`. The connection lives in a global `LazyLock<Mutex<Connection>>` singleton to avoid
//! contending with the `app.os` kernel lock. Writes are best-effort: failures only log a warning and never block the main flow.
//!
//! Follows the repo convention: no migrations framework; tables are created with `CREATE TABLE IF NOT EXISTS`.
//! Supports purging data older than the retention window (`cleanup_old`), auto-triggered periodically from the write path. New
//! columns are migrated incrementally via `ALTER TABLE ADD COLUMN`, ignoring "duplicate column" errors.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, params};

use aios_kernel::primitives::LlmUsageRecord;

use crate::ai::config_schema::AiConfig;
use crate::commonw::configw;

/// Default retention days: records older than this are deleted during automatic cleanup.
const DEFAULT_RETAIN_DAYS: u64 = 90;
/// Number of writes that trigger one automatic cleanup pass (avoids a full-table scan on every write).
const CLEANUP_EVERY_N_INSERTS: u64 = 100;
/// Conversion factor from cost_micros to cents (1 cent = 10,000 μ$).
const MICROS_PER_CENT: u64 = 10_000;

/// Self-insertion counter used to trigger automatic cleanup at a fixed frequency.
static INSERT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Kernel ledger cursor up to which records have been drained and persisted (kernel `LlmUsageRecord::seq`).
/// Callers use it to fetch incremental records via the kernel's `llm_usage_drain_since(cursor)`, advancing it after a successful persist.
static DRAIN_CURSOR: AtomicU64 = AtomicU64::new(0);

/// Global connection singleton. `None` means initialization failed (e.g. path not writable); subsequent writes are skipped.
static STORE: LazyLock<Option<Mutex<Connection>>> = LazyLock::new(|| match open_store() {
    Ok(conn) => Some(Mutex::new(conn)),
    Err(e) => {
        eprintln!("[TokenUsage] init failed, usage stats disabled: {e}");
        None
    }
});

/// Resolve the database file path: prefer the `ai.token_usage.db` config key, else the default path.
fn db_path() -> PathBuf {
    let cfg = configw::get_all_config();
    let raw = cfg
        .get_opt(AiConfig::TOKEN_USAGE_DB)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "~/.config/rust_tools/token_usage.db".to_string());
    PathBuf::from(crate::commonw::utils::expanduser(raw.trim()).as_ref())
}

/// Open and initialize the database connection.
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
        // Create the full schema for new databases; skip when an existing DB already has the table.
        r#"
        CREATE TABLE IF NOT EXISTS token_usage (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at    INTEGER NOT NULL,
            model         TEXT NOT NULL,
            input_tokens  INTEGER NOT NULL,
            output_tokens  INTEGER NOT NULL,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens   INTEGER NOT NULL,
            cost_micros    INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_token_usage_created_at ON token_usage(created_at);
        "#,
    )
    .map_err(|e| format!("init token_usage schema: {e}"))?;

    // Incremental migration: add the new columns to existing databases (ignoring "duplicate column" errors).
    let migrations = [
        "ALTER TABLE token_usage ADD COLUMN cost_micros INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE token_usage ADD COLUMN reasoning_tokens INTEGER NOT NULL DEFAULT 0",
    ];
    for sql in &migrations {
        if let Err(e) = conn.execute_batch(sql) {
            let msg = e.to_string();
            // SQLite 3.35+ returns 1 "duplicate column name"; older versions return different wording.
            if !msg.to_lowercase().contains("duplicate column") {
                eprintln!("[TokenUsage] migration warning: {e}");
            }
        }
    }

    Ok(conn)
}

/// Whether token accounting is enabled (on by default; set to false to turn it off).
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

/// Current drain cursor: callers use it to fetch incremental ledger records from the kernel.
pub(crate) fn drain_cursor() -> u64 {
    DRAIN_CURSOR.load(Ordering::Relaxed)
}

/// Batch-persist ledger records drained from the kernel. Best-effort: failures only log a warning and no error is returned.
///
/// `new_head` is the current head seq of the kernel ledger (`llm_usage_head_seq()`); after a successful
/// persist the cursor advances to that value so the next drain only fetches new records. `records` should be
/// the result of `drain_since(drain_cursor())` (ascending, with seq strictly greater than the old cursor).
pub(crate) fn persist_drained(records: &[LlmUsageRecord], new_head: u64) {
    if !enabled() {
        // Advance the cursor even when accounting is disabled, so re-enabling it does not replay the historical ledger.
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
            "INSERT INTO token_usage (created_at, model, input_tokens, output_tokens, \
             reasoning_tokens, total_tokens, cost_micros) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
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
                r.reasoning_tokens as i64,
                r.total_tokens as i64,
                r.cost_micros as i64,
            ]) {
                eprintln!("[TokenUsage] insert failed: {e}");
                // Roll back the whole batch and keep the drain cursor, so failed records never cause a permanent accounting gap.
                return;
            } else {
                inserted += 1;
            }
        }
    }
    if let Err(e) = tx.commit() {
        eprintln!("[TokenUsage] commit failed: {e}");
        return;
    }
    // Persist succeeded; advance the cursor.
    DRAIN_CURSOR.store(new_head, Ordering::Relaxed);

    // Trigger automatic cleanup at a fixed frequency to avoid a full-table scan on every write.
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

/// Delete records older than `retain_days` days (call while holding the connection lock).
fn cleanup_old_locked(conn: &mut Connection, retain_days: u64) {
    let cutoff = now_secs().saturating_sub(retain_days * 86400);
    if let Err(e) = conn.execute(
        "DELETE FROM token_usage WHERE created_at < ?1",
        params![cutoff as i64],
    ) {
        eprintln!("[TokenUsage] cleanup failed: {e}");
    }
}

/// Aggregated token usage over a time window.
#[derive(Debug, Clone, Default)]
pub(crate) struct UsageTotals {
    pub calls: u64,
    pub input: u64,
    pub output: u64,
    /// The subset of `output` spent on reasoning/thinking.
    pub reasoning: u64,
    pub total: u64,
    pub cost_micros: u64,
}

/// One usage row aggregated per model.
#[derive(Debug, Clone)]
pub(crate) struct UsageByModel {
    pub model: String,
    pub calls: u64,
    pub input: u64,
    pub output: u64,
    /// The subset of `output` spent on reasoning/thinking.
    pub reasoning: u64,
    pub total: u64,
    pub cost_micros: u64,
}

/// One usage row aggregated per day.
#[derive(Debug, Clone)]
pub(crate) struct DailyUsage {
    pub day: String,
    pub calls: u64,
    pub input: u64,
    pub output: u64,
    /// The subset of `output` spent on reasoning/thinking.
    pub reasoning: u64,
    pub total: u64,
    pub cost_micros: u64,
}

/// Query total usage over a time window. `window_secs=None` means all history;
/// otherwise only the last `window_secs` seconds are counted. A `None` return means storage is unavailable.
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
               COALESCE(SUM(reasoning_tokens),0), \
               COALESCE(SUM(total_tokens),0), \
               COALESCE(SUM(cost_micros),0) \
               FROM token_usage WHERE (?1 IS NULL OR created_at >= ?1)";
    conn.query_row(sql, params![cutoff], |row| {
        Ok(UsageTotals {
            calls: row.get::<_, i64>(0)? as u64,
            input: row.get::<_, i64>(1)? as u64,
            output: row.get::<_, i64>(2)? as u64,
            reasoning: row.get::<_, i64>(3)? as u64,
            total: row.get::<_, i64>(4)? as u64,
            cost_micros: row.get::<_, i64>(5)? as u64,
        })
    })
    .ok()
}

/// Query per-model aggregated usage over a time window, ordered by total tokens descending.
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
               COALESCE(SUM(reasoning_tokens),0), \
               COALESCE(SUM(total_tokens),0), \
               COALESCE(SUM(cost_micros),0) \
               FROM token_usage WHERE (?1 IS NULL OR created_at >= ?1) \
               GROUP BY model ORDER BY 6 DESC";
    let mut stmt = conn.prepare(sql).ok()?;
    let rows = stmt
        .query_map(params![cutoff], |row| {
            Ok(UsageByModel {
                model: row.get::<_, String>(0)?,
                calls: row.get::<_, i64>(1)? as u64,
                input: row.get::<_, i64>(2)? as u64,
                output: row.get::<_, i64>(3)? as u64,
                reasoning: row.get::<_, i64>(4)? as u64,
                total: row.get::<_, i64>(5)? as u64,
                cost_micros: row.get::<_, i64>(6)? as u64,
            })
        })
        .ok()?;
    Some(rows.filter_map(|r| r.ok()).collect())
}

/// Query per-day aggregated usage for the last N calendar days (one row per day), newest first.
pub(crate) fn query_daily_breakdown(days: u64) -> Option<Vec<DailyUsage>> {
    query_daily_impl(days, None)
}

/// Query the most recent N days that have data (one row per day), newest first.
/// Difference from [`query_daily_breakdown`]: that function truncates to a calendar-day window; this one has no time window,
/// and just takes the first `limit` days that have data, fitting the "default overview shows the most recent days with data" case.
pub(crate) fn query_recent_days(limit: usize) -> Option<Vec<DailyUsage>> {
    query_daily_impl(0, Some(limit as i64))
}

/// Internal implementation of per-day usage aggregation.
///
/// - `days > 0`: only count the last `days` calendar days (`cutoff = now - days*86400`).
/// - `limit > 0`: keep only the first `limit` days that have data (`ORDER BY day DESC LIMIT limit`).
fn query_daily_impl(days: u64, limit: Option<i64>) -> Option<Vec<DailyUsage>> {
    let store = STORE.as_ref()?;
    let conn = match store.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    // days=0 means no time window (cutoff=0); otherwise only the last `days` calendar days are counted.
    let cutoff = if days == 0 {
        0i64
    } else {
        now_secs().saturating_sub(days * 86_400) as i64
    };
    let limit_clause = match limit {
        Some(n) if n > 0 => format!("LIMIT {n}"),
        _ => String::new(),
    };
    // Use 'localtime' to convert UTC epoch seconds to the local date before grouping; otherwise calls made in the
    // small hours of UTC+8 get attributed to the previous day, making "today's usage" show up as yesterday.
    let sql = format!(
        "\
        SELECT DATE(created_at, 'unixepoch', 'localtime') AS day, \
               COUNT(*), \
               COALESCE(SUM(input_tokens),0), \
               COALESCE(SUM(output_tokens),0), \
               COALESCE(SUM(reasoning_tokens),0), \
               COALESCE(SUM(total_tokens),0), \
               COALESCE(SUM(cost_micros),0) \
        FROM token_usage \
        WHERE created_at >= ?1 \
        GROUP BY day \
        ORDER BY day DESC {limit_clause}"
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[TokenUsage] prepare daily query failed: {e}");
            return None;
        }
    };
    let rows = match stmt.query_map(params![cutoff], |row| {
        Ok(DailyUsage {
            day: row.get::<_, String>(0)?,
            calls: row.get::<_, i64>(1)? as u64,
            input: row.get::<_, i64>(2)? as u64,
            output: row.get::<_, i64>(3)? as u64,
            reasoning: row.get::<_, i64>(4)? as u64,
            total: row.get::<_, i64>(5)? as u64,
            cost_micros: row.get::<_, i64>(6)? as u64,
        })
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[TokenUsage] query daily rows failed: {e}");
            return None;
        }
    };
    Some(rows.filter_map(|r| r.ok()).collect())
}

/// Database file path (displayed by `/usage`).
pub(crate) fn store_path() -> PathBuf {
    db_path()
}

/// Whether enabled (displayed by `/usage`).
pub(crate) fn is_enabled() -> bool {
    enabled()
}

/// Format cost_micros (micro-dollars) as a human-readable string. 1 USD = 1,000,000 μ$.
pub(crate) fn format_cost(micros: u64) -> String {
    if micros >= 100_000_000 {
        // ≥ $100: show whole dollars
        format!("${}", micros / 1_000_000)
    } else if micros >= 1_000_000 {
        // $1 – $99.99: show dollars with two decimal places
        let dol = micros / 1_000_000;
        let cent = (micros % 1_000_000) / 10_000;
        format!("${}.{:02}", dol, cent)
    } else if micros >= 10_000 {
        // 1¢ – 99.99¢: show cents
        let c = micros / 10_000;
        let f = (micros % 10_000) / 100;
        if f == 0 {
            format!("{}¢", c)
        } else {
            format!("{}.{:02}¢", c, f)
        }
    } else if micros > 0 {
        // < 1¢
        format!("{:.2}¢", micros as f64 / 10_000.0)
    } else {
        "0".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_cost() {
        assert_eq!(format_cost(0), "0");
        assert_eq!(format_cost(5000), "0.50¢"); // 0.5¢
        assert_eq!(format_cost(10_000), "1¢");
        assert_eq!(format_cost(15_000), "1.50¢");
        assert_eq!(format_cost(100_000), "10¢");
        assert_eq!(format_cost(1_000_000), "$1.00");
        assert_eq!(format_cost(1_500_000), "$1.50");
        assert_eq!(format_cost(100_000_000), "$100");
    }
}
