//! 会话级文件变更审计日志（mutation log）。
//!
//! 每次 `write_file` / `apply_patch` 写盘或删盘时，以 best-effort 方式追加一条
//! JSONL 记录到当前 session 的 assets 目录下 `mutation_log.jsonl`。供 `/audit`
//! 子代理读取，了解 main agent 本会话通过工具改了哪些文件，从而只 review 属于
//! 自己的改动，而非工作区里其他并发需求留下的未提交改动。
//!
//! 设计要点：
//! - 绝不影响真实写盘：记录失败只静默丢弃，绝不向上传播错误。
//! - 跳过会话运行时产物（临时文件 / overflow / checkpoint / 子代理 scratch 目录）：
//!   这些不属于「主 agent 的项目改动」，不应污染审计视图。
//! - 每条记录含 before/after 内容（超过上限截断）：给审计子代理算 diff 用；需要完整
//!   内容时可 read_file 读原文件。
//! - 并发安全：同进程内主 agent 与并行子代理共享同一日志，append 走进程级锁串行化。
//! - 仅在活动 driver context（真实 turn）内生效；测试 / 一次性调用静默跳过。

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use super::file_store::current_session_assets_dir;

/// 全局单调递增序号，保证同一 session 内记录顺序稳定。
static SEQ: AtomicU64 = AtomicU64::new(0);

/// 串行化 append：后台子代理与主 agent 是同进程内的并行 tokio 任务，共享同一
/// session 的 mutation_log.jsonl。无锁的 `OpenOptions.append` + 分段 `writeln!`
/// 会让大记录的多次 write() 交错，损坏整行并殃及相邻记录。用进程级锁把「整行拼装 +
/// 单次写入」串行化，配合 O_APPEND 保证每条记录原子落盘。
static APPEND_LOCK: Mutex<()> = Mutex::new(());

/// 单条记录 before/after 内容上限：每次写盘都存全量前后内容会让日志随会话无界增长
/// （编辑 1MB 文件 100 次 ≈ 200MB）。超限内容截断并标注，`/audit` 需要完整前后内容时
/// 可用 read_file 读原文件。审计只需知道「改了哪些文件、大致改了什么」，无需逐字节留存。
const MAX_CONTENT_BYTES: usize = 16 * 1024;

/// 一条文件变更记录。
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct MutationEntry {
    /// 单调递增序号（全局）。
    pub seq: u64,
    /// ISO8601 UTC 时间戳。
    pub ts: String,
    /// 文件绝对路径。
    pub path: String,
    /// 操作类型：`"write"` 或 `"delete"`。
    pub op: String,
    /// 改动前内容（新文件为 None；删除时为被删内容）。
    pub before: Option<String>,
    /// 改动后内容（删除为 None）。
    pub after: Option<String>,
}

/// 当前 session 的 mutation log 文件路径。
pub(crate) fn log_path() -> Option<PathBuf> {
    current_session_assets_dir().map(|d| d.join("mutation_log.jsonl"))
}

/// 追加一条变更记录。best-effort：任何失败均静默丢弃，绝不影响真实写盘。
///
/// 落在 session 运行时目录（assets / 子代理 scratch / checkpoint 等）下的路径直接
/// 跳过——它们不是主 agent 的项目改动，不应污染审计视图。过大的 before/after 内容
/// 会被截断，避免日志随会话无界增长。
pub(crate) fn record(path: &std::path::Path, op: &str, before: Option<&str>, after: Option<&str>) {
    let Some(assets_dir) = current_session_assets_dir() else {
        // 无活动 driver context（测试 / 一次性调用）：静默跳过。
        return;
    };
    if should_skip(path, &assets_dir) {
        return;
    }

    let entry = MutationEntry {
        seq: SEQ.fetch_add(1, Ordering::Relaxed),
        ts: Utc::now().to_rfc3339(),
        path: path.to_string_lossy().into_owned(),
        op: op.to_string(),
        before: before.map(cap_content),
        after: after.map(cap_content),
    };
    append_entry(&assets_dir.join("mutation_log.jsonl"), &entry);
}

/// 把内容裁到 `MAX_CONTENT_BYTES` 以内（按字符边界安全截断），超出时附标注。
fn cap_content(content: &str) -> String {
    if content.len() <= MAX_CONTENT_BYTES {
        return content.to_string();
    }
    let mut end = MAX_CONTENT_BYTES;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n…[truncated {} more bytes; read the file for full content]",
        &content[..end],
        content.len() - end
    )
}

/// 是否应跳过该路径的记录。session 运行时目录（sessions root 之下的 assets、
/// 子代理 scratch `subagent-cwd-*`、checkpoints、子代理 memory 等）都不是项目改动。
/// `assets_dir` 形如 `<sessions_root>/<id>.assets`，其父目录即 sessions root；跳过
/// 整个 root 一次性覆盖 assets 与所有兄弟运行时产物，避免并行子代理的 scratch 写入
/// 被误记为主 agent 的项目改动。
fn should_skip(path: &std::path::Path, assets_dir: &std::path::Path) -> bool {
    let sessions_root = assets_dir.parent().unwrap_or(assets_dir);
    path.starts_with(sessions_root)
}

/// 追加一条 JSONL 记录到指定日志文件。best-effort：任何失败均静默丢弃。
///
/// 进程级锁串行化「整行拼装 + 单次写入」：同进程内的并行子代理共享同一日志，
/// 分段写会交错损坏记录。锁 + O_APPEND 保证每条记录整体原子落盘。
fn append_entry(log_path: &std::path::Path, entry: &MutationEntry) {
    let Ok(mut line) = serde_json::to_string(entry) else {
        return;
    };
    line.push('\n');
    // 确保父目录存在（首次写入时可能尚未创建）。
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _guard = APPEND_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_path) {
        // 单次 write_all：整行（含换行）一起写，配合 O_APPEND 原子追加。
        let _ = file.write_all(line.as_bytes());
    }
}

/// 读取当前 session 的全部变更记录（按写入顺序）。无日志或读取失败时返回空。
pub(crate) fn read_all() -> Vec<MutationEntry> {
    let Some(log_path) = log_path() else {
        return Vec::new();
    };
    read_entries(&log_path)
}

/// 从 JSONL 日志文件读取全部记录。读取失败或文件不存在时返回空；非法行被跳过。
fn read_entries(log_path: &std::path::Path) -> Vec<MutationEntry> {
    let Ok(content) = std::fs::read_to_string(log_path) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|line| serde_json::from_str::<MutationEntry>(line).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_is_safe_noop_without_driver_context() {
        // 无活动 driver context 时 record / read_all 必须静默空操作，绝不 panic。
        record(
            std::path::Path::new("/tmp/nonexistent_audit_test.rs"),
            "write",
            None,
            Some("x"),
        );
        assert!(read_all().is_empty());
        assert!(log_path().is_none());
    }

    #[test]
    fn should_skip_paths_under_assets_dir() {
        let assets = std::path::Path::new("/home/u/.history_file.sessions/abc.assets");
        assert!(should_skip(&assets.join("tmp/scratch.rs"), assets));
        assert!(should_skip(&assets.join("mutation_log.jsonl"), assets));
        assert!(!should_skip(
            std::path::Path::new("/proj/src/main.rs"),
            assets
        ));
    }

    #[test]
    fn should_skip_subagent_scratch_sibling_dirs() {
        // 子代理 scratch 位于 <sessions_root>/subagent-cwd-<id>/，是 <id>.assets 的兄弟
        // 目录。它们必须被跳过，否则并行子代理的写入会被误记为主 agent 的项目改动。
        let assets = std::path::Path::new("/home/u/.history_file.sessions/abc.assets");
        let root = std::path::Path::new("/home/u/.history_file.sessions");
        assert!(should_skip(&root.join("subagent-cwd-t1/foo.rs"), assets));
        assert!(should_skip(
            &root.join("checkpoints/abc/gen-1/x.sqlite"),
            assets
        ));
        assert!(should_skip(&root.join("def.assets/tmp/y.rs"), assets));
        // sessions root 之外的真实项目改动仍要记录。
        assert!(!should_skip(
            std::path::Path::new("/proj/src/main.rs"),
            assets
        ));
    }

    #[test]
    fn cap_content_truncates_oversized_payload_on_char_boundary() {
        let small = "hello";
        assert_eq!(cap_content(small), small);

        // 多字节字符横跨上限边界时不得 panic，且必须标注截断。
        let big = "€".repeat(MAX_CONTENT_BYTES); // 每个 € 3 字节，总长远超上限
        let capped = cap_content(&big);
        assert!(capped.len() < big.len());
        assert!(capped.contains("truncated"));
    }

    #[test]
    fn append_and_read_entries_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            ".agent_mutation_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let log_path = dir.join("mutation_log.jsonl");
        append_entry(
            &log_path,
            &MutationEntry {
                seq: 1,
                ts: "t1".into(),
                path: "/proj/a.rs".into(),
                op: "write".into(),
                before: Some("old".into()),
                after: Some("new".into()),
            },
        );
        append_entry(
            &log_path,
            &MutationEntry {
                seq: 2,
                ts: "t2".into(),
                path: "/proj/b.rs".into(),
                op: "delete".into(),
                before: Some("gone".into()),
                after: None,
            },
        );
        let entries = read_entries(&log_path);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].path, "/proj/a.rs");
        assert_eq!(entries[0].before.as_deref(), Some("old"));
        assert_eq!(entries[0].after.as_deref(), Some("new"));
        assert_eq!(entries[1].op, "delete");
        assert_eq!(entries[1].after, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_entries_returns_empty_for_missing_file() {
        let entries = read_entries(std::path::Path::new(
            "/nonexistent/.agent/mutation_log.jsonl",
        ));
        assert!(entries.is_empty());
    }

    #[test]
    fn read_entries_skips_malformed_lines() {
        let dir = std::env::temp_dir().join(format!(
            ".agent_mutation_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let log_path = dir.join("mutation_log.jsonl");
        append_entry(
            &log_path,
            &MutationEntry {
                seq: 1,
                ts: "t1".into(),
                path: "/proj/a.rs".into(),
                op: "write".into(),
                before: None,
                after: Some("x".into()),
            },
        );
        // 追加一行非法 JSON，read_entries 应跳过它而保留合法条目。
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&log_path) {
            let _ = writeln!(f, "not valid json");
        }
        let entries = read_entries(&log_path);
        assert_eq!(entries.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
