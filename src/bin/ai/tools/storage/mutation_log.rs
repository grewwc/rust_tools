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
use std::path::{Path, PathBuf};
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

/// Recognizable prefix of the truncation marker. `is_capped` and the rendering
/// side use it to tell whether a before/after snapshot is complete.
const TRUNCATED_MARKER: &str = "[truncated ";

/// Per-entry cap for the authoritative diff. The diff normally contains only the
/// changed lines, far smaller than the file itself; in the extreme case (full
/// rewrite) it is capped here. Capping happens at a line boundary, so the
/// rendered output is still a truthful partial diff — unlike truncated
/// snapshots, it never misjudges the truncation edge as a deletion.
const MAX_DIFF_BYTES: usize = 64 * 1024;

/// Cap for one before/after side of `entry_diff`: content beyond this is never
/// scanned when building the diff. Without it a full rewrite of a multi-GB file
/// materialized a complete `+ …` diff for the entire new content in memory
/// before the 64 KiB output cap applied (P2 regression). 1 MiB is far above
/// real-world per-write sizes and the largest diff fixture in the test suite
/// (~48 KiB), so the common path is unaffected.
const MAX_DIFF_INPUT_BYTES: usize = 1024 * 1024;

/// Initial file presence, independent of whether its contents were captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BeforeState {
    Absent,
    Present,
    Unknown,
}

/// One recorded file mutation.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct MutationEntry {
    /// Globally increasing sequence number.
    pub seq: u64,
    /// ISO8601 UTC timestamp.
    pub ts: String,
    /// Absolute file path.
    pub path: String,
    /// Operation: `"write"` or `"delete"`.
    pub op: String,
    /// Captured preimage. None can mean absence or unavailable contents; consult
    /// `effective_before_state` before treating it as an empty file.
    pub before: Option<String>,
    /// Missing in legacy logs, whose None preimages retain their old meaning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_state: Option<BeforeState>,
    /// Captured postimage; None for deletion.
    pub after: Option<String>,
    /// Line-level diff of this write (`- ` old lines / `+ ` new lines), computed
    /// from the before/after content within `MAX_DIFF_INPUT_BYTES`. Once a
    /// snapshot is truncated a reliable diff cannot be rebuilt (the truncation
    /// edge reads as a deletion), so this diff is the authoritative source for
    /// audit and display; old logs that lack the field deserialize to None and
    /// rendering falls back to snapshot diffs.
    pub diff: Option<String>,
}

impl MutationEntry {
    pub(crate) fn effective_before_state(&self) -> BeforeState {
        self.before_state.unwrap_or(if self.before.is_some() {
            BeforeState::Present
        } else {
            BeforeState::Absent
        })
    }
}

/// 当前 session 的 mutation log 文件路径。
pub(crate) fn log_path() -> Option<PathBuf> {
    current_session_assets_dir().map(|d| d.join("mutation_log.jsonl"))
}

/// Legacy entry point for callers whose None preimage means known absence.
pub(crate) fn record(path: &Path, op: &str, before: Option<&str>, after: Option<&str>) {
    let before_state = if before.is_some() {
        BeforeState::Present
    } else {
        BeforeState::Absent
    };
    record_with_before_state(path, op, before, after, before_state);
}

/// Best-effort recording with file presence separate from snapshot availability.
/// Runtime artifacts are skipped and recording failures never affect the write.
pub(crate) fn record_with_before_state(
    path: &Path,
    op: &str,
    before: Option<&str>,
    after: Option<&str>,
    before_state: BeforeState,
) {
    let Some(assets_dir) = current_session_assets_dir() else {
        // Tests and one-shot callers without a driver context do not record.
        return;
    };
    if should_skip(path, &assets_dir) {
        return;
    }

    let entry = make_entry(path, op, before, after, before_state);
    append_entry(&assets_dir.join("mutation_log.jsonl"), &entry);
}

fn make_entry(
    path: &Path,
    op: &str,
    before: Option<&str>,
    after: Option<&str>,
    before_state: BeforeState,
) -> MutationEntry {
    MutationEntry {
        seq: SEQ.fetch_add(1, Ordering::Relaxed),
        ts: Utc::now().to_rfc3339(),
        path: path.to_string_lossy().into_owned(),
        op: op.to_string(),
        before: before.map(cap_content),
        before_state: Some(before_state),
        after: after.map(cap_content),
        diff: if before.is_none() && before_state != BeforeState::Absent {
            Some("…[before snapshot unavailable; contents not compared]\n".to_string())
        } else {
            entry_diff(before, after)
        },
    }
}

/// 把内容裁到 `MAX_CONTENT_BYTES` 以内（按字符边界安全截断），超出时附标注。
pub(crate) fn cap_content(content: &str) -> String {
    if content.len() <= MAX_CONTENT_BYTES {
        return content.to_string();
    }
    let mut end = MAX_CONTENT_BYTES;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n…{TRUNCATED_MARKER}{} more bytes; read the file for full content]",
        &content[..end],
        content.len() - end
    )
}

/// Whether `cap_content` truncated the content. Rendering uses this to decide
/// whether a snapshot can be diffed directly.
pub(crate) fn is_capped(s: &str) -> bool {
    s.contains(TRUNCATED_MARKER)
}

/// Truncates `s` to at most `MAX_DIFF_INPUT_BYTES` bytes at a char boundary.
/// Returns a prefix slice (no allocation) plus whether truncation happened.
fn cap_diff_input(s: &str) -> (&str, bool) {
    if s.len() <= MAX_DIFF_INPUT_BYTES {
        return (s, false);
    }
    let mut end = MAX_DIFF_INPUT_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    (&s[..end], true)
}

/// Computes the line-level diff of this write from the before/after content.
/// For files within `MAX_DIFF_INPUT_BYTES` the diff is computed on the full
/// content and is always reliable; rendering prefers it over snapshot diffs to
/// avoid false deletions at the truncation edge. Content beyond the cap is never
/// scanned, so a large rewrite cannot materialize an O(file) diff; the result is
/// then a truthful leading partial and a truncation line is appended so a
/// partial diff is never mistaken for a complete one — or for "unchanged".
/// Returns None only when there is truly nothing to show (new empty file, both
/// absent, or unchanged content within the scanned window).
pub(crate) fn entry_diff(before: Option<&str>, after: Option<&str>) -> Option<String> {
    let (b, b_capped) = match before {
        Some(s) => {
            let (t, c) = cap_diff_input(s);
            (Some(t), c)
        }
        None => (None, false),
    };
    let (a, a_capped) = match after {
        Some(s) => {
            let (t, c) = cap_diff_input(s);
            (Some(t), c)
        }
        None => (None, false),
    };
    let capped = b_capped || a_capped;

    let mut out = match (b, a) {
        (None, None) => return None,
        (None, Some(a)) => {
            let lines: Vec<&str> = a.lines().collect();
            if lines.is_empty() {
                return None;
            }
            let mut out = String::new();
            for l in &lines {
                out.push_str(&format!("+ {l}\n"));
            }
            out
        }
        (Some(b), None) => {
            let lines: Vec<&str> = b.lines().collect();
            if lines.is_empty() {
                return None;
            }
            let mut out = String::new();
            for l in &lines {
                out.push_str(&format!("- {l}\n"));
            }
            out
        }
        (Some(b), Some(a)) => {
            let bv: Vec<&str> = b.lines().collect();
            let av: Vec<&str> = a.lines().collect();
            let mut prefix = 0;
            while prefix < bv.len() && prefix < av.len() && bv[prefix] == av[prefix] {
                prefix += 1;
            }
            let mut suffix = 0;
            while suffix < bv.len() - prefix
                && suffix < av.len() - prefix
                && bv[bv.len() - 1 - suffix] == av[av.len() - 1 - suffix]
            {
                suffix += 1;
            }
            let mut out = String::new();
            for l in &bv[prefix..bv.len() - suffix] {
                out.push_str(&format!("- {l}\n"));
            }
            for l in &av[prefix..av.len() - suffix] {
                out.push_str(&format!("+ {l}\n"));
            }
            out
        }
    };

    if out.is_empty() {
        if capped {
            // Matching prefixes prove neither equality nor a difference in the
            // unscanned tail, including when the full inputs happen to be equal.
            out.push_str(&format!(
                "…[no differing lines within the first {MAX_DIFF_INPUT_BYTES} bytes; \
                 remaining content not compared; leading portion only]\n"
            ));
        } else {
            return None;
        }
    } else if capped {
        out.push_str(&format!(
            "…[diff input truncated at {MAX_DIFF_INPUT_BYTES} bytes; leading portion only — \
             read the file for the full change]\n"
        ));
    }
    Some(cap_diff(out))
}

/// Caps the diff text at `MAX_DIFF_BYTES`, cutting at a line boundary when
/// possible (never leaves a partial line). The capped diff is the leading
/// truthful portion, so rendering stays a correct partial view — unlike the
/// false deletions caused by diffing truncated snapshots.
fn cap_diff(mut out: String) -> String {
    if out.len() <= MAX_DIFF_BYTES {
        return out;
    }
    let mut end = MAX_DIFF_BYTES;
    while end > 0 && !out.is_char_boundary(end) {
        end -= 1;
    }
    // 尽量在最近换行处截断，保证末尾是完整行。
    while end > 0 && out.as_bytes()[end - 1] != b'\n' {
        end -= 1;
    }
    out.truncate(end);
    out.push_str(&format!(
        "…[diff truncated at {MAX_DIFF_BYTES} bytes; see the file for the rest]\n"
    ));
    out
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
        // The truncation marker must be recognized by is_capped (rendering uses
        // it to decide whether a snapshot can be diffed directly).
        assert!(is_capped(&capped));
        assert!(!is_capped(small));
    }

    #[test]
    fn entry_diff_modify_emits_only_changed_lines() {
        let before = "a\nb\nc\nd\ne\n";
        let after = "a\nb\nX\nd\ne\n";
        assert_eq!(
            entry_diff(Some(before), Some(after)).as_deref(),
            Some("- c\n+ X\n")
        );
    }

    #[test]
    fn entry_diff_created_and_deleted_emit_single_sided_lines() {
        assert_eq!(
            entry_diff(None, Some("x\ny\n")).as_deref(),
            Some("+ x\n+ y\n")
        );
        assert_eq!(
            entry_diff(Some("x\ny\n"), None).as_deref(),
            Some("- x\n- y\n")
        );
    }

    #[test]
    fn entry_diff_none_for_unchanged_or_empty() {
        assert!(entry_diff(Some("same"), Some("same")).is_none());
        assert!(entry_diff(None, None).is_none());
        assert!(entry_diff(None, Some("")).is_none());
        assert!(entry_diff(Some(""), None).is_none());
    }

    #[test]
    fn entry_diff_caps_huge_change_at_line_boundary() {
        // Full-file rewrite: when the diff exceeds MAX_DIFF_BYTES it must be cut
        // at a line boundary with a marker, and must never panic.
        let before = format!("{}\n", "b".repeat(MAX_DIFF_BYTES + 100));
        let after = format!("{}\n", "a".repeat(MAX_DIFF_BYTES + 100));
        let d = entry_diff(Some(&before), Some(&after)).unwrap();
        assert!(d.contains("truncated"));
        assert!(d.ends_with('\n'));
    }

    #[test]
    fn entry_diff_caps_huge_change_with_multibyte_chars() {
        // Multi-byte characters straddling the 64KiB cap must not panic and the
        // cut must stay on a line boundary.
        let before = format!("{}\n", "€€€".repeat(MAX_DIFF_BYTES));
        let after = format!("{}\n", "¥¥¥".repeat(MAX_DIFF_BYTES));
        let d = entry_diff(Some(&before), Some(&after)).unwrap();
        assert!(d.contains("truncated"));
        assert!(d.ends_with('\n'));
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
                before_state: None,
                after: Some("new".into()),
                diff: Some("- old\n+ new\n".into()),
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
                before_state: None,
                after: None,
                diff: Some("- gone\n".into()),
            },
        );
        let entries = read_entries(&log_path);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].path, "/proj/a.rs");
        assert_eq!(entries[0].before.as_deref(), Some("old"));
        assert_eq!(entries[0].after.as_deref(), Some("new"));
        assert_eq!(entries[0].diff.as_deref(), Some("- old\n+ new\n"));
        assert_eq!(entries[1].op, "delete");
        assert_eq!(entries[1].after, None);
        assert_eq!(entries[1].diff.as_deref(), Some("- gone\n"));
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
                before_state: None,
                after: Some("x".into()),
                diff: None,
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

    #[test]
    fn read_entries_defaults_missing_diff_field_to_none() {
        // Old-format logs have no diff field: deserialization must yield None
        // (backward compatible) rather than failing to parse.
        let dir = std::env::temp_dir().join(format!(
            ".agent_mutation_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Bare OpenOptions does not create parent dirs (append_entry does), so
        // create it explicitly here.
        let _ = std::fs::create_dir_all(&dir);
        let log_path = dir.join("mutation_log.jsonl");
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = writeln!(
                f,
                r#"{{"seq":1,"ts":"t","path":"/proj/a.rs","op":"write","before":"old","after":"new"}}"#
            );
        }
        let entries = read_entries(&log_path);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].before.as_deref(), Some("old"));
        assert_eq!(entries[0].diff, None);
        assert_eq!(entries[0].before_state, None);
        assert_eq!(entries[0].effective_before_state(), BeforeState::Present);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_missing_before_state_retains_create_delete_semantics() {
        for (before, after, op, expected) in [
            (None, Some("new"), "write", BeforeState::Absent),
            (Some("old"), None, "delete", BeforeState::Present),
        ] {
            let entry: MutationEntry = serde_json::from_value(serde_json::json!({
                "seq": 1, "ts": "t", "path": "/proj/a.rs", "op": op,
                "before": before, "after": after,
            })).unwrap();
            assert_eq!(entry.before_state, None);
            assert_eq!(entry.effective_before_state(), expected);
            assert_eq!(entry.diff, None);
        }
    }

    #[test]
    fn unavailable_preimage_records_state_without_fabricating_additions() {
        for state in [BeforeState::Present, BeforeState::Unknown] {
            for after in [Some("replacement\n"), Some(""), None] {
                let entry = make_entry(Path::new("/proj/a.rs"), "write", None, after, state);
                let json = serde_json::to_string(&entry).unwrap();
                let decoded: MutationEntry = serde_json::from_str(&json).unwrap();
                assert_eq!(decoded.before_state, Some(state));
                assert_eq!(decoded.effective_before_state(), state);
                assert!(decoded.before.is_none());
                let diff = decoded.diff.unwrap();
                assert!(diff.contains("before snapshot unavailable"));
                assert!(diff.contains("not compared"));
                assert!(!diff.lines().any(|line| line.starts_with("+ ") || line.starts_with("- ")));
            }
        }
        let created = make_entry(Path::new("/proj/new.rs"), "write", None, Some("new\n"), BeforeState::Absent);
        assert_eq!(created.diff.as_deref(), Some("+ new\n"));
    }

    #[test]
    fn identical_large_inputs_do_not_claim_an_unobserved_tail_difference() {
        let input = "same line\n".repeat(MAX_DIFF_INPUT_BYTES / 5);
        assert!(input.len() > MAX_DIFF_INPUT_BYTES);
        let diff = entry_diff(Some(&input), Some(&input)).unwrap();
        assert!(diff.contains("remaining content not compared"));
        assert!(!diff.contains("differ beyond"));
        assert!(!diff.lines().any(|line| line.starts_with("+ ") || line.starts_with("- ")));
        assert!(diff.len() < 512);
    }

    #[test]
    fn entry_diff_bounds_huge_content_instead_of_building_full_diff() {
        // Regression (P2): a multi-MB write used to materialize a full `+ line`
        // diff for the whole content in memory before the 64 KiB cap applied.
        let mut big = String::new();
        for i in 0..200_000 {
            big.push_str(&format!("line {i}\n"));
        }
        assert!(big.len() > MAX_DIFF_INPUT_BYTES);
        let d = entry_diff(None, Some(&big)).expect("diff must exist");
        // Leading additions only; the diff must stay far below the file size.
        assert!(d.starts_with("+ line 0\n"), "diff: {d}");
        assert!(
            d.len() < MAX_DIFF_INPUT_BYTES,
            "diff must be bounded, got {} bytes",
            d.len()
        );
        // The reader must see a truncation notice (input cap or output cap
        // marker), never silently believe the diff is complete.
        assert!(d.contains("truncated"), "diff: {d}");
    }

    #[test]
    fn entry_diff_capped_window_never_reports_unchanged() {
        // Regression: before/after share the first 1 MiB but differ beyond it.
        // Returning None would read as "unchanged"; the truncation must surface.
        let mut a = String::new();
        let mut b = String::new();
        for i in 0..300_000 {
            let line = format!("same line {i}\n");
            a.push_str(&line);
            b.push_str(&line);
        }
        a.push_str("DIFFERENT TAIL\n");
        let d = entry_diff(Some(&b), Some(&a)).expect("must not be None for capped input");
        assert!(
            d.contains("leading portion only"),
            "must surface truncation, got: {d}"
        );
    }

    #[test]
    fn entry_diff_single_line_huge_rewrite_is_bounded() {
        // Pathological case from the P2 report: one multi-MB line (minified
        // JSON / single-line log). The whole line must not be copied.
        let line = "x".repeat(3 * 1024 * 1024);
        let d = entry_diff(None, Some(&line)).unwrap();
        assert!(d.len() < 200 * 1024, "bounded: {} bytes", d.len());
        // cap_diff walks back to the nearest newline; a single-line diff reduces
        // to the truncation notice, which is truthful and still bounded.
        assert!(d.contains("truncated"), "diff: {d}");
    }

    #[test]
    fn entry_diff_cap_input_handles_multibyte_char_at_window_boundary() {
        // A 3-byte UTF-8 char straddling the 1 MiB cap must not panic and must
        // produce a valid bounded diff (cap_diff_input walks back to a char
        // boundary before slicing).
        let mut s = String::new();
        s.push_str(&"a".repeat(MAX_DIFF_INPUT_BYTES - 1));
        s.push('中'); // 3 bytes, crosses the cap boundary
        s.push_str(&"b".repeat(100));
        assert!(s.len() > MAX_DIFF_INPUT_BYTES);
        let d = entry_diff(None, Some(&s)).unwrap();
        assert!(d.len() < MAX_DIFF_INPUT_BYTES, "bounded: {} bytes", d.len());
        assert!(d.contains("truncated"), "diff: {d}");
    }

    #[test]
    fn entry_diff_small_content_behavior_is_unchanged() {
        assert_eq!(entry_diff(Some("a\nb\n"), Some("a\nb\n")), None);
        assert_eq!(entry_diff(None, Some("")), None);
        assert_eq!(entry_diff(Some("a\nb\nc\n"), Some("a\nb\nx\n")), Some("- c\n+ x\n".into()));
    }
}
