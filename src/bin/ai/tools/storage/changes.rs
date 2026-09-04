//! 会话级文件变更聚合与外部 Diff 打开。
//!
//! 聚合来源：
//! - 优先 `mutation_log`（本会话内通过 write_file / apply_patch 的精确 before/after）。
//! - 为空时回退到 `git diff HEAD` / `git status`（覆盖 execute_command 间接触及的文件）。
//!
//! 产物：
//! - 统一的变更摘要文本与统计。
//! - 联合 patch（mutation 派生或 git 原生），可落盘到 `session_assets/changes.patch`。
//! - 外部编辑器打开能力：VS Code (`code --diff` / `code <patch>`）、Cursor、JetBrains
//!   IDEA、git difftool、系统 `open`，优先读 `ai.diff.editor` 配置，未配置时自动探测。

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::ai::config_schema::AiConfig;
use crate::ai::tools::storage::mutation_log::{MutationEntry, is_capped, read_all};

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

/// 支持的外部打开方式。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorKind {
    Auto,
    Vscode,
    Cursor,
    Idea,
    Git,
    SystemOpen,
}

impl EditorKind {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Some(Self::Auto),
            "vscode" | "code" | "vs_code" => Some(Self::Vscode),
            "cursor" => Some(Self::Cursor),
            "idea" | "intellij" | "jetbrains" => Some(Self::Idea),
            "git" | "difftool" | "git-difftool" => Some(Self::Git),
            "open" | "system" | "systemopen" => Some(Self::SystemOpen),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Vscode => "vscode",
            Self::Cursor => "cursor",
            Self::Idea => "idea",
            Self::Git => "git difftool",
            Self::SystemOpen => "open",
        }
    }
}

/// 从配置 `ai.diff.editor` 解析首选编辑器，未配置则 Auto。
pub fn configured_editor() -> EditorKind {
    let raw = crate::commonw::configw::get_all_config().get(AiConfig::DIFF_EDITOR, "");
    if raw.trim().is_empty() {
        return EditorKind::Auto;
    }
    EditorKind::from_str(&raw).unwrap_or(EditorKind::Auto)
}

/// 探测本机可用编辑器（best-effort，不阻塞过久）。
fn probe_executable(name: &str) -> bool {
    // `which` 在 macOS/Linux 可用；Windows 回退到 where。
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn resolve_auto_editor(cwd: &Path) -> EditorKind {
    if probe_executable("code") {
        return EditorKind::Vscode;
    }
    if probe_executable("cursor") {
        return EditorKind::Cursor;
    }
    if is_inside_git_work_tree(cwd) && probe_executable("git") {
        return EditorKind::Git;
    }
    // macOS `open` 几乎总是存在；Linux `xdg-open` 同理，用 open 统称。
    EditorKind::SystemOpen
}

pub fn resolve_editor(requested: Option<EditorKind>, cwd: &Path) -> EditorKind {
    match requested.unwrap_or_else(configured_editor) {
        EditorKind::Auto => resolve_auto_editor(cwd),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Git helpers
// ---------------------------------------------------------------------------

fn git_output(cwd: &Path, args: &[&str]) -> Option<String> {
    crate::fork_guard::output(Command::new("git").args(args).current_dir(cwd))
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
}

pub fn is_inside_git_work_tree(cwd: &Path) -> bool {
    git_output(cwd, &["rev-parse", "--is-inside-work-tree"])
        .map(|s| s.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn git_is_available() -> bool {
    git_output(
        &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        &["--version"],
    )
    .is_some()
}

// ---------------------------------------------------------------------------
// Mutation grouping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FileChange {
    pub path: String,
    pub rel: String,
    pub net: String,
    pub write_count: usize,
    pub delete_count: usize,
    pub before_first: Option<String>,
    pub after_last: Option<String>,
    /// Authoritative per-write diffs (`- `/`+ ` lines) recorded at write time, in
    /// write order. Large-file snapshots are truncated and unreliable, so
    /// rendering and patch generation prefer these diffs.
    pub diffs: Vec<String>,
}

fn cwd_for_display() -> Option<PathBuf> {
    crate::ai::driver::runtime_ctx::effective_cwd().ok()
}

fn to_rel(path: &str, cwd: Option<&Path>) -> String {
    if let Some(c) = cwd {
        if let Ok(rel) = Path::new(path).strip_prefix(c) {
            return rel.to_string_lossy().into_owned();
        }
    }
    path.to_string()
}

fn grouped_changes(entries: &[MutationEntry]) -> Vec<FileChange> {
    let cwd = cwd_for_display();
    let mut grouped: Vec<FileChange> = Vec::new();
    for e in entries {
        if let Some(g) = grouped.iter_mut().find(|g| g.path == e.path) {
            g.after_last.clone_from(&e.after);
            // op 最终状态以最后一条为准，net 随后重算
            if e.op == "write" {
                g.write_count += 1;
            } else {
                g.delete_count += 1;
            }
            if let Some(d) = &e.diff {
                g.diffs.push(d.clone());
            }
        } else {
            grouped.push(FileChange {
                path: e.path.clone(),
                rel: to_rel(&e.path, cwd.as_deref()),
                net: String::new(), // 占位
                write_count: if e.op == "write" { 1 } else { 0 },
                delete_count: if e.op == "delete" { 1 } else { 0 },
                before_first: e.before.clone(),
                after_last: e.after.clone(),
                diffs: e.diff.iter().cloned().collect(),
            });
        }
    }
    for g in &mut grouped {
        let last_deleted = g.after_last.is_none() && g.delete_count > 0;
        // 若最后一条是 delete，则 after_last 为 None，视为 deleted；否则按 before 判 created/modified
        g.net = if last_deleted {
            "deleted".to_string()
        } else if g.before_first.is_none() && g.after_last.is_some() {
            "created".to_string()
        } else {
            "modified".to_string()
        };
    }
    grouped
}

/// Whether the before_first / after_last snapshots are complete (present and
/// not truncated). Diffing truncated snapshots misreads the truncation edge as
/// a deletion (false deletion); in that case the authoritative diff recorded at
/// write time is used instead.
pub(crate) fn snapshots_full(before_first: &Option<String>, after_last: &Option<String>) -> bool {
    match (before_first, after_last) {
        (Some(b), Some(a)) => !is_capped(b) && !is_capped(a),
        (Some(b), None) => !is_capped(b),
        (None, Some(a)) => !is_capped(a),
        (None, None) => true,
    }
}

/// Renderable diff block for one file. When snapshots are complete it keeps the
/// existing net-diff logic (small-file behavior unchanged); when truncated it
/// uses the authoritative diff recorded at write time so the truncation edge is
/// never rendered as a deletion.
fn file_snippet(g: &FileChange, max_lines: usize) -> Option<String> {
    if snapshots_full(&g.before_first, &g.after_last) {
        return diff_snippet(
            g.before_first.as_deref(),
            g.after_last.as_deref(),
            max_lines,
        );
    }
    if !g.diffs.is_empty() {
        return diff_block_from_lines(&g.diffs.join(""), max_lines);
    }
    // Old logs (no diff field) with truncated snapshots: fall back to a snapshot
    // diff (possibly distorted, but better than no information).
    diff_snippet(
        g.before_first.as_deref(),
        g.after_last.as_deref(),
        max_lines,
    )
}

/// Renders stored `- `/`+ ` lines as a ```diff block, capped at max_lines with a
/// marker when longer.
pub(crate) fn diff_block_from_lines(diff: &str, max_lines: usize) -> Option<String> {
    let lines: Vec<&str> = diff.lines().collect();
    if lines.is_empty() {
        return None;
    }
    let total = lines.len();
    let shown: Vec<&str> = if total <= max_lines {
        lines.iter().map(|s| *s).collect()
    } else {
        lines.iter().take(max_lines).map(|s| *s).collect()
    };
    let mut out = format!("```diff\n{}", shown.join("\n"));
    if total > max_lines {
        out.push_str(&format!(
            "\n（差异共 {total} 行，已展示前 {max_lines} 行；完整内容见 mutation log）"
        ));
    }
    out.push_str("\n```");
    Some(out)
}

pub fn session_grouped_changes() -> Vec<FileChange> {
    let entries = read_all();
    if entries.is_empty() {
        return Vec::new();
    }
    grouped_changes(&entries)
}

// ---------------------------------------------------------------------------
// Patch generation
// ---------------------------------------------------------------------------

fn diff_snippet(before: Option<&str>, after: Option<&str>, max_lines: usize) -> Option<String> {
    let (b, a): (Vec<&str>, Vec<&str>) = match (before, after) {
        (None, None) => return None,
        (None, Some(a)) => (Vec::new(), a.lines().collect()),
        (Some(b), None) => (b.lines().collect(), Vec::new()),
        (Some(b), Some(a)) => (b.lines().collect(), a.lines().collect()),
    };
    if b == a {
        return None;
    }
    let mut prefix = 0;
    while prefix < b.len() && prefix < a.len() && b[prefix] == a[prefix] {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < b.len() - prefix
        && suffix < a.len() - prefix
        && b[b.len() - 1 - suffix] == a[a.len() - 1 - suffix]
    {
        suffix += 1;
    }
    let mut lines: Vec<String> = Vec::new();
    for l in &b[prefix..b.len() - suffix] {
        lines.push(format!("- {l}"));
    }
    for l in &a[prefix..a.len() - suffix] {
        lines.push(format!("+ {l}"));
    }
    if lines.is_empty() {
        return None;
    }
    let total = lines.len();
    if total <= max_lines {
        Some(format!("```diff\n{}\n```", lines.join("\n")))
    } else {
        let shown: Vec<&str> = lines.iter().take(max_lines).map(|s| s.as_str()).collect();
        Some(format!(
            "```diff\n{}\n```\n（差异共 {total} 行，已展示前 {max_lines} 行；完整内容见 mutation log）",
            shown.join("\n")
        ))
    }
}

/// 由 mutation 日志派生的联合 patch（简单 unified 格式，非严格 git 格式但可被编辑器打开）。
pub fn mutation_patch(entries: &[MutationEntry]) -> Option<String> {
    let grouped = grouped_changes(entries);
    if grouped.is_empty() {
        return None;
    }
    let mut out = String::new();
    for g in &grouped {
        out.push_str(&format!("diff -- {} {}\n", g.rel, g.rel));
        match g.net.as_str() {
            "created" => {
                out.push_str(&format!("--- /dev/null\n+++ b/{}\n", g.rel));
                // A `--- /dev/null` patch body must be pure additions (the new
                // content). Only a single-write create yields a full dump in
                // diffs; any later write produces an incremental diff (pure `+`
                // for appends, `- `/`+ ` for edits) whose lines reference a base
                // version that does not exist under /dev/null. So only use
                // diffs.last() for the single-write case and otherwise dump the
                // (capped) final snapshot, which is at least a valid all-`+` body.
                if !snapshots_full(&g.before_first, &g.after_last) {
                    let mut used_diff = false;
                    if g.diffs.len() == 1 {
                        if let Some(d) = g.diffs.last() {
                            out.push_str(d);
                            if !d.ends_with('\n') {
                                out.push('\n');
                            }
                            used_diff = true;
                        }
                    }
                    if !used_diff {
                        if let Some(after) = g.after_last.as_deref() {
                            for line in after.lines() {
                                out.push_str(&format!("+{line}\n"));
                            }
                        }
                    }
                } else if let Some(after) = g.after_last.as_deref() {
                    for line in after.lines() {
                        out.push_str(&format!("+{line}\n"));
                    }
                }
            }
            "deleted" => {
                out.push_str(&format!("--- a/{}\n+++ /dev/null\n", g.rel));
                // A `+++ /dev/null` patch body must be pure deletions. The delete
                // entry's own diff is the full `-` dump; prefer it, and only fall
                // back to the (capped) before snapshot when no authoritative diff
                // exists (old logs written without the diff field), so the patch is
                // never an empty body under the deletion header.
                if !snapshots_full(&g.before_first, &g.after_last) {
                    let mut used_diff = false;
                    if let Some(d) = g.diffs.last() {
                        out.push_str(d);
                        if !d.ends_with('\n') {
                            out.push('\n');
                        }
                        used_diff = true;
                    }
                    if !used_diff && let Some(before) = g.before_first.as_deref() {
                        for line in before.lines() {
                            out.push_str(&format!("-{line}\n"));
                        }
                    }
                } else if let Some(before) = g.before_first.as_deref() {
                    for line in before.lines() {
                        out.push_str(&format!("-{line}\n"));
                    }
                }
            }
            _ => {
                out.push_str(&format!("--- a/{}\n+++ b/{}\n", g.rel, g.rel));
                if !snapshots_full(&g.before_first, &g.after_last) && !g.diffs.is_empty() {
                    // With truncated snapshots, use the authoritative diff
                    // directly (`- `/`+ ` lines are the patch body).
                    let body = g.diffs.join("");
                    out.push_str(&body);
                    if !body.ends_with('\n') {
                        out.push('\n');
                    }
                } else if let Some(snippet) =
                    diff_snippet(g.before_first.as_deref(), g.after_last.as_deref(), 200)
                {
                    // 去掉 ```diff 围栏，仅保留差异行
                    for line in snippet.lines() {
                        if line.starts_with("```") || line.starts_with("（差异") {
                            continue;
                        }
                        out.push_str(line);
                        out.push('\n');
                    }
                } else if g.before_first != g.after_last {
                    // 回退：全量 before/after
                    if let Some(b) = g.before_first.as_deref() {
                        for l in b.lines() {
                            out.push_str(&format!("-{l}\n"));
                        }
                    }
                    if let Some(a) = g.after_last.as_deref() {
                        for l in a.lines() {
                            out.push_str(&format!("+{l}\n"));
                        }
                    }
                }
            }
        }
        out.push('\n');
    }
    if out.trim().is_empty() {
        None
    } else {
        Some(out)
    }
}

/// 联合 patch：优先 mutation 派生，否则回退到 `git diff HEAD`。
pub fn combined_patch() -> Option<String> {
    let entries = read_all();
    if !entries.is_empty() {
        if let Some(p) = mutation_patch(&entries) {
            return Some(p);
        }
    }
    // git 回退
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd().ok()?;
    if !is_inside_git_work_tree(&cwd) {
        return None;
    }
    let diff = git_output(&cwd, &["diff", "HEAD", "--no-color"])?;
    if diff.trim().is_empty() {
        // 无 HEAD 或无差异时尝试无 HEAD 的工作区 diff
        let diff2 = git_output(&cwd, &["diff", "--no-color"])?;
        if diff2.trim().is_empty() {
            return None;
        }
        return Some(diff2);
    }
    Some(diff)
}

/// 将联合 patch 落盘到 `session_assets/changes.patch`（无活动会话时回退到系统
/// 临时目录 `changes.<pid>.patch`），返回绝对路径。
pub fn write_combined_patch() -> Result<PathBuf, String> {
    let patch = combined_patch()
        .ok_or_else(|| "当前会话无可导出的变更（无 mutation log 且 git 无差异）".to_string())?;
    // 优先写入 <session_assets>/changes.patch；无活动会话（one-shot / 测试）时
    // 回退到 runtime_ctx::temp_dir()（`<tmp>/.agent_tmp/default/`），保证
    // `/changes --open` 在非交互式调用下也能生成 patch 并打开。
    let session_dir = crate::ai::tools::storage::file_store::current_session_assets_dir();
    let dir = match &session_dir {
        Some(assets) => assets.clone(),
        None => crate::ai::driver::runtime_ctx::temp_dir()
            .map_err(|e| format!("无法确定 patch 输出目录（无活动会话且临时目录不可用）：{e}"))?,
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("创建输出目录失败: {e}"))?;
    let name = if session_dir.is_some() {
        "changes.patch".to_string()
    } else {
        // 无会话回退：加 pid 避免并发 one-shot 调用互相覆盖
        format!("changes.{}.patch", std::process::id())
    };
    let path = dir.join(name);
    std::fs::write(&path, patch.as_bytes()).map_err(|e| format!("写入 patch 失败: {e}"))?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// Summary / stat
// ---------------------------------------------------------------------------

pub fn format_session_summary() -> String {
    format_session_summary_with_git(true)
}

pub fn format_session_summary_with_git(include_git_extra: bool) -> String {
    let entries = read_all();
    if entries.is_empty() {
        if !include_git_extra {
            return "当前会话无文件变更（include_git=false，仅统计 mutation log）。".to_string();
        }
        return fallback_git_summary();
    }
    let grouped = grouped_changes(&entries);
    let cwd = cwd_for_display();
    let log_path = crate::ai::tools::storage::mutation_log::log_path();
    let mut out =
        String::from("以下是本会话通过 write_file / apply_patch 改动的文件（按首次改动顺序）。\n");
    const CAP: usize = 14_000;
    let mut truncated = false;
    for (i, g) in grouped.iter().enumerate() {
        if out.len() >= CAP {
            truncated = true;
            break;
        }
        let mut counts = String::new();
        if g.write_count > 0 {
            counts.push_str(&format!("{} write", g.write_count));
        }
        if g.delete_count > 0 {
            if !counts.is_empty() {
                counts.push_str(", ");
            }
            counts.push_str(&format!("{} delete", g.delete_count));
        }
        out.push_str(&format!("{}. {}  [{}]  ({counts})\n", i + 1, g.rel, g.net));
        if let Some(snippet) = file_snippet(g, 30) {
            out.push_str(&snippet);
            out.push_str("\n\n");
        }
    }
    if truncated {
        out.push_str("…（更多改动省略）\n\n");
    }
    if let Some(lp) = &log_path {
        out.push_str(&format!(
            "完整 before/after 见 mutation log：{}\n可用 read_file 读取该日志获取每个改动的原始与最终内容。\n",
            lp.display()
        ));
    }
    // 若 git 亦有额外差异且调用方允许混入 git 状态，追加提示；include_git=false 时必须保持纯会话视图
    if include_git_extra {
        if let Some(cwd) = cwd {
            if is_inside_git_work_tree(&cwd) {
                let status = git_output(&cwd, &["status", "--porcelain=v1"]).unwrap_or_default();
                if !status.trim().is_empty() {
                    // 统计 git 中未被 mutation 覆盖的文件数（简单提示）
                    let git_files: Vec<&str> =
                        status.lines().filter(|l| !l.trim().is_empty()).collect();
                    let git_extra = git_files.len().saturating_sub(grouped.len());
                    if git_extra > 0 {
                        out.push_str(&format!(
                            "\n提示：git 工作区另有约 {git_extra} 个未被本会话 mutation log 覆盖的改动（可能含并发任务的改动），可执行 git status 查看。\n"
                        ));
                    }
                }
            }
        }
    }
    out
}

fn fallback_git_summary() -> String {
    let cwd = match crate::ai::driver::runtime_ctx::effective_cwd() {
        Ok(p) => p,
        Err(_) => return "当前会话无工具级变更记录，且无法确定工作目录。".to_string(),
    };
    if !is_inside_git_work_tree(&cwd) {
        return "当前会话无工具级变更记录，且不在 git 仓库内（无可回退的 git 差异）".to_string();
    }
    let status = git_output(&cwd, &["status", "--porcelain=v1"]).unwrap_or_default();
    if status.trim().is_empty() {
        return "当前会话无工具级变更记录，且 git 工作区干净（无未提交改动）".to_string();
    }
    const MAX_DIFF_BYTES: usize = 32_768;
    let diff = git_output(&cwd, &["diff", "HEAD", "--no-color"]).unwrap_or_default();
    let mut sections = vec![format!("## git status --porcelain\n{status}")];
    if diff.len() <= MAX_DIFF_BYTES {
        if !diff.trim().is_empty() {
            sections.push(format!("## git diff HEAD\n{diff}"));
        }
    } else {
        let stat = git_output(&cwd, &["diff", "HEAD", "--stat", "--no-color"]).unwrap_or_default();
        sections.push(format!(
            "## git diff HEAD --stat\n（完整 diff 超过 {MAX_DIFF_BYTES} 字节，仅展示统计）\n{stat}"
        ));
    }
    let mut out = String::from(
        "（本会话无工具级 mutation log，以下为工作区未提交改动，可能含并发需求的改动）\n\n",
    );
    out.push_str(&sections.join("\n\n"));
    out
}

pub fn git_stat() -> Option<String> {
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd().ok()?;
    if !is_inside_git_work_tree(&cwd) {
        return None;
    }
    git_output(&cwd, &["diff", "HEAD", "--stat", "--no-color"])
        .or_else(|| git_output(&cwd, &["diff", "--stat", "--no-color"]))
        .filter(|s| !s.trim().is_empty())
}

pub fn combined_summary() -> String {
    format_session_summary()
}

// ---------------------------------------------------------------------------
// External open
// ---------------------------------------------------------------------------

/// 打开变更：生成 patch 并用指定编辑器打开，返回面向用户的状态文案。
pub fn open_changes(requested: Option<EditorKind>) -> Result<String, String> {
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let editor = resolve_editor(requested, &cwd);
    let patch_path = write_combined_patch()?;

    // 按编辑器分发
    let result = match editor {
        EditorKind::Vscode => open_with_vscode(&patch_path, &cwd),
        EditorKind::Cursor => open_with_cursor(&patch_path),
        EditorKind::Idea => open_with_idea(&patch_path),
        EditorKind::Git => open_with_git_difftool(&cwd),
        EditorKind::SystemOpen => open_with_system(&patch_path),
        EditorKind::Auto => unreachable!("Auto 已在 resolve_editor 中消解"),
    };

    match result {
        Ok(detail) => Ok(format!(
            "已生成 patch：{}\n已用 {} 打开：{detail}",
            patch_path.display(),
            editor.label()
        )),
        Err(e) => Err(format!(
            "patch 已生成于 {}，但用 {} 打开失败：{e}\n可手动执行：code {}  或  open {}",
            patch_path.display(),
            editor.label(),
            patch_path.display(),
            patch_path.display()
        )),
    }
}

fn open_with_vscode(patch_path: &Path, cwd: &Path) -> Result<String, String> {
    // 优先尝试 per-file --diff（当恰好单文件时体验更好），否则直接打开 patch 文件。
    let grouped = session_grouped_changes();
    if grouped.len() == 1 {
        let g = &grouped[0];
        // 为 before/after 创建临时文件供 code --diff 使用
        // 与 service::changes::try_open_single_file_diff 保持一致：优先会话临时目录，避免直写系统 /tmp 越界
        let tmp =
            crate::ai::driver::runtime_ctx::temp_dir().unwrap_or_else(|_| std::env::temp_dir());
        let before_path = tmp.join(format!("a_changes_before_{}", sanitize_filename(&g.rel)));
        let after_path = tmp.join(format!("a_changes_after_{}", sanitize_filename(&g.rel)));
        // best-effort 写入，不影响主流程：失败则回退到直接打开 patch
        let before_ok = g
            .before_first
            .as_deref()
            .map(|c| std::fs::write(&before_path, c).is_ok())
            .unwrap_or(true);
        let after_ok = g
            .after_last
            .as_deref()
            .map(|c| std::fs::write(&after_path, c).is_ok())
            .unwrap_or(true);
        // Truncated snapshots are not the full file, so code --diff would show a
        // wrong comparison; fall back to opening the patch instead (the patch is
        // built from the authoritative diff recorded at write time).
        if before_ok && after_ok && snapshots_full(&g.before_first, &g.after_last) {
            // 尝试 --diff 双文件对比（后台 detached）
            let mut cmd = Command::new("code");
            cmd.arg("--diff")
                .arg(&before_path)
                .arg(&after_path)
                .current_dir(cwd)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            // 若在无窗口环境 code 不存在，会在 spawn 阶段失败
            match cmd.spawn() {
                Ok(_) => {
                    return Ok(format!(
                        "code --diff {} {}",
                        before_path.display(),
                        after_path.display()
                    ));
                }
                Err(e) => return Err(format!("code --diff 失败: {e}")),
            }
        }
    }
    // 回退：直接用 code 打开 patch 文件
    let mut cmd = Command::new("code");
    cmd.arg(patch_path).current_dir(cwd);
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd.spawn()
        .map(|_| format!("code {}", patch_path.display()))
        .map_err(|e| format!("code 打开失败: {e}"))
}

fn open_with_cursor(patch_path: &Path) -> Result<String, String> {
    Command::new("cursor")
        .arg(patch_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| format!("cursor {}", patch_path.display()))
        .map_err(|e| format!("cursor 打开失败: {e}"))
}

fn open_with_idea(patch_path: &Path) -> Result<String, String> {
    // JetBrains Toolbox 常用 `idea` / `webstorm` 等；先试 idea，其次 open
    for bin in ["idea", "webstorm", "pycharm", "clion"] {
        if probe_executable(bin) {
            return Command::new(bin)
                .arg("diff")
                .arg(patch_path)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .map(|_| format!("{bin} diff {}", patch_path.display()))
                .map_err(|e| format!("{bin} 打开失败: {e}"));
        }
    }
    Err("未找到可用的 JetBrains IDE 可执行文件（idea/webstorm/pycharm/clion）".to_string())
}

fn open_with_git_difftool(cwd: &Path) -> Result<String, String> {
    if !is_inside_git_work_tree(cwd) {
        return Err("不在 git 仓库内，无法使用 git difftool".to_string());
    }
    // 使用 --dir-diff 可一次性展示所有文件的外部工具对比；用 --no-prompt 避免交互阻塞
    // 后台 detached 启动，不等待完成
    Command::new("git")
        .args(["difftool", "--dir-diff", "--no-prompt"])
        .current_dir(cwd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| "git difftool --dir-diff --no-prompt".to_string())
        .map_err(|e| format!("git difftool 启动失败: {e}"))
}

fn open_with_system(patch_path: &Path) -> Result<String, String> {
    let bin = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(windows) {
        "start"
    } else {
        "xdg-open"
    };
    Command::new(bin)
        .arg(patch_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| format!("{bin} {}", patch_path.display()))
        .map_err(|e| format!("{bin} 打开失败: {e}"))
}

fn sanitize_filename(s: &str) -> String {
    s.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_from_str_maps_known_values() {
        assert_eq!(EditorKind::from_str("vscode"), Some(EditorKind::Vscode));
        assert_eq!(EditorKind::from_str("code"), Some(EditorKind::Vscode));
        assert_eq!(EditorKind::from_str("cursor"), Some(EditorKind::Cursor));
        assert_eq!(EditorKind::from_str("idea"), Some(EditorKind::Idea));
        assert_eq!(EditorKind::from_str("git"), Some(EditorKind::Git));
        assert_eq!(EditorKind::from_str("open"), Some(EditorKind::SystemOpen));
        assert_eq!(EditorKind::from_str("auto"), Some(EditorKind::Auto));
        assert_eq!(EditorKind::from_str("unknown"), None);
    }

    #[test]
    fn sanitize_replaces_path_separators() {
        assert_eq!(sanitize_filename("a/b\\c"), "a_b_c");
        assert_eq!(sanitize_filename("foo:bar"), "foo_bar");
    }

    #[test]
    fn grouped_changes_computes_net_correctly() {
        let entries = vec![
            MutationEntry {
                seq: 0,
                ts: "t".into(),
                path: "/tmp/proj/a.rs".into(),
                op: "write".into(),
                before: None,
                after: Some("hello".into()),
                diff: Some("+ hello\n".into()),
            },
            MutationEntry {
                seq: 1,
                ts: "t".into(),
                path: "/tmp/proj/b.rs".into(),
                op: "write".into(),
                before: Some("old".into()),
                after: Some("new".into()),
                diff: Some("- old\n+ new\n".into()),
            },
            MutationEntry {
                seq: 2,
                ts: "t".into(),
                path: "/tmp/proj/b.rs".into(),
                op: "write".into(),
                before: Some("new".into()),
                after: Some("new2".into()),
                diff: Some("- new\n+ new2\n".into()),
            },
        ];
        let grouped = grouped_changes(&entries);
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].net, "created");
        assert_eq!(grouped[1].net, "modified");
        assert_eq!(grouped[1].write_count, 2);
        // Authoritative diffs are collected into FileChange in write order.
        assert_eq!(grouped[0].diffs, vec!["+ hello\n".to_string()]);
        assert_eq!(
            grouped[1].diffs,
            vec!["- old\n+ new\n".to_string(), "- new\n+ new2\n".to_string()]
        );
    }

    #[test]
    fn mutation_patch_contains_expected_headers() {
        let entries = vec![MutationEntry {
            seq: 0,
            ts: "t".into(),
            path: "/tmp/proj/new.rs".into(),
            op: "write".into(),
            before: None,
            after: Some("line1\nline2".into()),
            diff: Some("+ line1\n+ line2\n".into()),
        }];
        let patch = mutation_patch(&entries).unwrap();
        assert!(patch.contains("--- /dev/null"));
        assert!(patch.contains("+++ b/"));
        assert!(patch.contains("+line1"));
    }

    #[test]
    fn write_combined_patch_no_session_falls_back_to_temp_dir() {
        // 回归：无活动会话（one-shot / 测试）时，/changes --open 不得再报
        // "无活动会话，无法确定 patch 输出目录"，应回退到系统临时目录。
        // 测试环境无 DRIVER_CTX，本前置条件成立。
        assert!(
            crate::ai::tools::storage::file_store::current_session_assets_dir().is_none(),
            "测试环境应无活动会话"
        );
        match write_combined_patch() {
            Ok(path) => {
                let s = path.to_string_lossy();
                assert!(s.contains(".agent_tmp"), "应回退到系统临时目录: {s}");
                assert!(
                    s.ends_with(&format!("changes.{}.patch", std::process::id())),
                    "路径应含 pid: {s}"
                );
                assert!(path.exists(), "patch 文件应已写入: {s}");
                let _ = std::fs::remove_file(&path);
            }
            Err(e) => {
                assert!(
                    e.contains("无可导出变更"),
                    "无活动会话时不得报缺少会话错误，实际: {e}"
                );
            }
        }
    }

    #[test]
    fn truncated_snapshots_use_stored_diff_without_false_deletions() {
        // Regression: when before/after snapshots are truncated (> MAX_CONTENT_BYTES)
        // the display must use the authoritative diff recorded at write time and
        // must not render the truncation edge as a deletion (an untouched 295-line
        // tail was once shown as `-`).
        use crate::ai::tools::storage::mutation_log::{cap_content, entry_diff};

        let mut before = String::new();
        let mut after = String::new();
        for i in 0..1500 {
            before.push_str(&format!("keep line {i}\n"));
            after.push_str(&format!("keep line {i}\n"));
        }
        before.push_str("TO_BE_REMOVED\n");
        after.push_str("TO_BE_ADDED\n");
        for i in 0..1500 {
            before.push_str(&format!("tail line {i}\n"));
            after.push_str(&format!("tail line {i}\n"));
        }
        // Second write to the same file: verify multi-write diffs are all rendered
        // with no false deletions.
        let after2 = after.replace("TO_BE_ADDED", "TO_BE_ADDED_V2");
        // Snapshots exceed the 16KiB cap → truncated; the authoritative diff still
        // pinpoints the changed lines.
        assert!(
            before.len() > 16 * 1024,
            "test fixture must exceed the snapshot cap"
        );
        let entries = vec![
            MutationEntry {
                seq: 0,
                ts: "t".into(),
                path: "/proj/big.rs".into(),
                op: "write".into(),
                before: Some(cap_content(&before)),
                after: Some(cap_content(&after)),
                diff: entry_diff(Some(&before), Some(&after)),
            },
            MutationEntry {
                seq: 1,
                ts: "t".into(),
                path: "/proj/big.rs".into(),
                op: "write".into(),
                before: Some(cap_content(&after)),
                after: Some(cap_content(&after2)),
                diff: entry_diff(Some(&after), Some(&after2)),
            },
        ];
        let grouped = grouped_changes(&entries);
        assert_eq!(grouped.len(), 1);
        let g = &grouped[0];
        assert_eq!(
            g.diffs.len(),
            2,
            "authoritative diffs of both writes must be collected"
        );
        assert!(!snapshots_full(&g.before_first, &g.after_last));
        let snippet = file_snippet(g, 30).unwrap();
        assert!(snippet.contains("- TO_BE_REMOVED"), "snippet: {snippet}");
        assert!(snippet.contains("+ TO_BE_ADDED"), "snippet: {snippet}");
        assert!(snippet.contains("+ TO_BE_ADDED_V2"), "snippet: {snippet}");
        // The untouched tail must never be rendered as deleted (false-deletion regression).
        assert!(!snippet.contains("- tail line"), "snippet: {snippet}");
        // mutation_patch uses the authoritative diff as well.
        let patch = mutation_patch(&entries).unwrap();
        assert!(patch.contains("- TO_BE_REMOVED"), "patch: {patch}");
        assert!(patch.contains("+ TO_BE_ADDED"), "patch: {patch}");
        assert!(patch.contains("+ TO_BE_ADDED_V2"), "patch: {patch}");
        assert!(!patch.contains("- tail line"), "patch: {patch}");
    }

    #[test]
    fn created_file_patch_stays_pure_addition_after_second_write() {
        // Regression: a `--- /dev/null` patch for a created >16KiB file that was
        // then edited again must not take the last write's incremental diff - it
        // carries `- ` lines referencing a base version that does not exist and
        // would drop the creation content. The body must be pure additions of the
        // final snapshot.
        use crate::ai::tools::storage::mutation_log::{cap_content, entry_diff};

        let mut v1 = String::new();
        for i in 0..3000 {
            v1.push_str(&format!("content line {i}\n"));
        }
        let v2 = format!("{v1}extra line at end\n");
        assert!(
            v1.len() > 16 * 1024,
            "test fixture must exceed the snapshot cap"
        );
        let entries = vec![
            MutationEntry {
                seq: 0,
                ts: "t".into(),
                path: "/proj/created.rs".into(),
                op: "write".into(),
                before: None,
                after: Some(cap_content(&v1)),
                diff: entry_diff(None, Some(&v1)),
            },
            MutationEntry {
                seq: 1,
                ts: "t".into(),
                path: "/proj/created.rs".into(),
                op: "write".into(),
                before: Some(cap_content(&v1)),
                after: Some(cap_content(&v2)),
                diff: entry_diff(Some(&v1), Some(&v2)),
            },
        ];
        let grouped = grouped_changes(&entries);
        assert_eq!(grouped.len(), 1);
        let g = &grouped[0];
        assert_eq!(g.net, "created");
        assert_eq!(g.diffs.len(), 2);
        assert!(!snapshots_full(&g.before_first, &g.after_last));
        let patch = mutation_patch(&entries).unwrap();
        // Body must be all additions: no `- ` lines after the headers.
        let mut in_body = false;
        let mut had_deletion = false;
        for l in patch.lines() {
            if l.starts_with("--- ") || l.starts_with("+++ ") {
                in_body = true;
                continue;
            }
            if in_body && l.starts_with("- ") {
                had_deletion = true;
                break;
            }
        }
        assert!(
            !had_deletion,
            "created patch must not contain deletion lines:\n{patch}"
        );
        // Creation content must still be present (from the final snapshot dump).
        assert!(patch.contains("+content line 0"), "patch: {patch}");
        // The second write's change sits past the 16KiB snapshot cap, so it is
        // honestly dropped: the patch must carry the truncation marker instead of
        // a fake incremental body that claims a base version.
        assert!(patch.contains("[truncated"), "patch: {patch}");
        assert!(!patch.contains("extra line at end"), "patch: {patch}");
    }

    #[test]
    fn deleted_file_patch_falls_back_to_snapshot_without_diff_field() {
        // Regression: old mutation logs have no `diff` field. A `+++ /dev/null`
        // patch for a deleted >16KiB file must not have an empty body: without a
        // stored diff it falls back to the (capped) before snapshot, rendered as
        // pure deletions.
        use crate::ai::tools::storage::mutation_log::cap_content;

        let mut before = String::new();
        for i in 0..3000 {
            before.push_str(&format!("content line {i}\n"));
        }
        assert!(
            before.len() > 16 * 1024,
            "test fixture must exceed the snapshot cap"
        );
        let entries = vec![MutationEntry {
            seq: 0,
            ts: "t".into(),
            path: "/proj/deleted.rs".into(),
            op: "delete".into(),
            before: Some(cap_content(&before)),
            after: None,
            // Old log format: no authoritative diff recorded.
            diff: None,
        }];
        let grouped = grouped_changes(&entries);
        assert_eq!(grouped.len(), 1);
        let g = &grouped[0];
        assert_eq!(g.net, "deleted");
        assert!(g.diffs.is_empty());
        assert!(!snapshots_full(&g.before_first, &g.after_last));
        let patch = mutation_patch(&entries).unwrap();
        // The body must not be empty: the capped before snapshot is dumped as
        // deletion lines.
        assert!(patch.contains("-content line 0"), "patch: {patch}");
        // No addition lines may appear under the deletion header.
        let mut in_body = false;
        let mut had_addition = false;
        for l in patch.lines() {
            if l.starts_with("--- ") || l.starts_with("+++ ") {
                in_body = true;
                continue;
            }
            if in_body && l.starts_with('+') && !l.starts_with("+++") {
                had_addition = true;
                break;
            }
        }
        assert!(
            !had_addition,
            "deleted patch must not contain addition lines:\n{patch}"
        );
    }
}
