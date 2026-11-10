//! Session change summaries and external diff viewers.
//!
//! The opt-in deep-audit inventory unions session mutations with Git status and staged/unstaged
//! diff paths. Complete paths and collection failures stay separate from capped
//! snippets; session records do not establish request/task ownership.
//! Patch exports and external viewers retain their existing snapshot rules.

use rust_tools::commonw::FastMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::ai::config_schema::AiConfig;
use crate::ai::tools::storage::mutation_log::{BeforeState, MutationEntry, is_capped, read_all};

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

/// Supported external diff viewers.
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

/// Read the configured diff viewer, defaulting to automatic detection.
pub fn configured_editor() -> EditorKind {
    let raw = crate::commonw::configw::get_all_config().get(AiConfig::DIFF_EDITOR, "");
    if raw.trim().is_empty() {
        return EditorKind::Auto;
    }
    EditorKind::from_str(&raw).unwrap_or(EditorKind::Auto)
}

/// Probe available viewers on a best-effort basis.
fn probe_executable(name: &str) -> bool {
    // Use `which` on macOS/Linux and `where` on Windows.
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
    // Treat macOS `open` and Linux `xdg-open` as the system viewer.
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

#[derive(Debug, serde::Serialize)]
pub(crate) struct InventoryFile {
    pub(crate) path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) path_bytes_hex: Option<String>,
    pub(crate) observations: Vec<String>,
    pub(crate) ownership: &'static str,
}

pub(crate) struct ChangeInventory {
    pub(crate) files: Vec<InventoryFile>,
    pub(crate) session_changes: Vec<FileChange>,
    pub(crate) is_git_repo: bool,
    pub(crate) collection_errors: Vec<String>,
    pub(crate) coverage_gaps: Vec<String>,
    log_path: Option<PathBuf>,
    git_stats: Vec<(&'static str, String)>,
    path_indices: FastMap<PathBuf, usize>,
}

/// Collect every observed path before rendering any capped content. Git numstat
/// is the NUL-delimited path companion to stat, whose display may abbreviate names.
pub(crate) fn collect_current_changes(include_git: bool) -> ChangeInventory {
    let mut inventory = ChangeInventory {
        files: Vec::new(),
        session_changes: Vec::new(),
        is_git_repo: false,
        collection_errors: Vec::new(),
        coverage_gaps: Vec::new(),
        log_path: crate::ai::tools::storage::mutation_log::log_path(),
        git_stats: Vec::new(),
        path_indices: FastMap::default(),
    };
    let entries = inventory.read_mutations();
    inventory.session_changes = grouped_changes(&entries);
    for entry in &entries {
        inventory.add_path(
            PathBuf::from(&entry.path),
            format!("session mutation: {}", entry.op),
        );
    }
    if include_git {
        match crate::ai::driver::runtime_ctx::effective_cwd() {
            Ok(cwd) => inventory.collect_git(&cwd),
            Err(error) => inventory.collection_errors.push(format!(
                "Git collection: working directory unavailable: {error}"
            )),
        }
    } else {
        inventory.is_git_repo = crate::ai::driver::runtime_ctx::effective_cwd()
            .ok()
            .is_some_and(|cwd| is_inside_git_work_tree(&cwd));
    }
    if !inventory.files.is_empty() {
        inventory.coverage_gaps.push(
            "Request/task ownership is unknown for every path: mutation records identify session tool operations but carry no request/task owner, and Git changes may include concurrent work. Keep these paths in scope until ownership is resolved.".to_string(),
        );
    }
    inventory
}

impl ChangeInventory {
    fn read_mutations(&mut self) -> Vec<MutationEntry> {
        let Some(path) = &self.log_path else {
            self.coverage_gaps.push("No active session mutation log is available; session tool changes cannot be enumerated.".to_string());
            return Vec::new();
        };
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
            Err(error) => {
                self.collection_errors
                    .push(format!("Session mutation log {}: {error}", path.display()));
                return Vec::new();
            }
        };
        parse_inventory_mutations(&content, &mut self.collection_errors)
    }

    fn add_path(&mut self, path: PathBuf, observation: String) {
        if let Some(&index) = self.path_indices.get(&path) {
            if !self.files[index].observations.contains(&observation) {
                self.files[index].observations.push(observation);
            }
            return;
        }
        let display = path
            .to_str()
            .map(str::to_string)
            .unwrap_or_else(|| format!("{:?}", path.as_os_str()));
        let path_bytes_hex = path.to_str().is_none().then(|| {
            path.as_os_str()
                .as_encoded_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<Vec<_>>()
                .join("")
        });
        self.path_indices.insert(path, self.files.len());
        self.files.push(InventoryFile {
            path: display,
            path_bytes_hex,
            observations: vec![observation],
            ownership: "unknown",
        });
    }

    fn collect_git(&mut self, cwd: &Path) {
        let root = match checked_git_output(cwd, &["rev-parse", "--show-toplevel"]) {
            Ok(mut root) => {
                if root.last() == Some(&b'\n') {
                    root.pop();
                }
                match git_path_from_bytes(&root) {
                    Ok(root) => root,
                    Err(error) => {
                        self.collection_errors.push(error);
                        return;
                    }
                }
            }
            Err(error) => {
                self.collection_errors.push(error);
                return;
            }
        };
        self.is_git_repo = true;
        match checked_git_output(
            &root,
            &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        ) {
            Ok(status) => match parse_status_paths(&status) {
                Ok(paths) => {
                    for (path, status) in paths {
                        self.add_path(root.join(path), status);
                    }
                }
                Err(error) => self.collection_errors.push(error),
            },
            Err(error) => self.collection_errors.push(error),
        }
        for (label, staged) in [
            ("git diff --stat (unstaged)", false),
            ("git diff --cached --stat (staged)", true),
        ] {
            let mut args = vec![
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--no-relative",
            ];
            if staged {
                args.push("--cached");
            }
            args.push("--stat");
            match checked_git_output(&root, &args) {
                Ok(stat) => self
                    .git_stats
                    .push((label, String::from_utf8_lossy(&stat).into_owned())),
                Err(error) => self.collection_errors.push(error),
            }
            args.pop();
            args.extend(["--numstat", "-z"]);
            match checked_git_output(&root, &args) {
                Ok(stat) => match parse_numstat_paths(&stat) {
                    Ok(paths) => {
                        for path in paths {
                            self.add_path(root.join(path), label.to_string());
                        }
                    }
                    Err(error) => self.collection_errors.push(error),
                },
                Err(error) => self.collection_errors.push(error),
            }
            if self.session_changes.is_empty() {
                args.pop();
                args.pop();
                match checked_git_output(&root, &args) {
                    Ok(diff) => self.git_stats.push((
                        if staged {
                            "git diff --cached (staged snippet)"
                        } else {
                            "git diff (unstaged snippet)"
                        },
                        String::from_utf8_lossy(&diff).into_owned(),
                    )),
                    Err(error) => self.collection_errors.push(error),
                }
            }
        }
    }

    /// Ownership uncertainty is separate from whether all sources were collected.
    pub(crate) fn collection_complete(&self) -> bool {
        self.log_path.is_some() && self.collection_errors.is_empty()
    }

    /// This section has no snippet cap: later files must remain visible.
    pub(crate) fn format_inventory(&self) -> String {
        let mut out = format!(
            "## Complete observed path inventory ({} paths)\n\n",
            self.files.len()
        );
        for file in &self.files {
            let path = serde_json::to_string(&file.path).expect("path string is serializable");
            out.push_str(&format!(
                "- {path} [{}; ownership: {}]\n",
                file.observations.join("; "),
                file.ownership
            ));
        }
        if self.files.is_empty() {
            out.push_str("No changed paths were observed from the available sources.\n");
        }
        for (label, items) in [
            ("Collection failures", &self.collection_errors),
            ("Coverage gaps", &self.coverage_gaps),
        ] {
            if !items.is_empty() {
                out.push_str(&format!("\n## {label}\n\n"));
                for item in items {
                    out.push_str(&format!("- {item}\n"));
                }
            }
        }
        out
    }

    pub(crate) fn format_snippets(&self) -> String {
        let mut out = format_mutation_snippets(&self.session_changes, self.log_path.as_deref());
        const CAP: usize = 32_768;
        for (label, stat) in &self.git_stats {
            if stat.trim().is_empty() {
                continue;
            }
            let remaining = CAP.saturating_sub(out.len());
            let mut end = stat.len().min(remaining);
            while !stat.is_char_boundary(end) {
                end -= 1;
            }
            out.push_str(&format!("\n## {label}\n{}", &stat[..end]));
            if end < stat.len() {
                out.push_str(
                    "\n[Git snippet truncated; the complete path inventory above is unaffected]\n",
                );
            }
        }
        out
    }
}

fn parse_inventory_mutations(content: &str, errors: &mut Vec<String>) -> Vec<MutationEntry> {
    content
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match serde_json::from_str(line) {
            Ok(entry) => Some(entry),
            Err(error) => {
                errors.push(format!(
                    "Session mutation log record {} could not be collected: {error}",
                    index + 1
                ));
                None
            }
        })
        .collect()
}

fn checked_git_output(cwd: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = crate::fork_guard::output(Command::new("git").args(args).current_dir(cwd))
        .map_err(|error| format!("git {} collection failed: {error}", args.join(" ")))?;
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(1024)
            .collect::<String>();
        return Err(format!(
            "git {} collection failed ({}): {}",
            args.join(" "),
            output.status,
            error.trim()
        ));
    }
    Ok(output.stdout)
}

fn git_path_from_bytes(bytes: &[u8]) -> Result<PathBuf, String> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        Ok(PathBuf::from(std::ffi::OsString::from_vec(bytes.to_vec())))
    }
    #[cfg(not(unix))]
    {
        String::from_utf8(bytes.to_vec())
            .map(PathBuf::from)
            .map_err(|_| "Git path collection failed: filename is not UTF-8".to_string())
    }
}

fn nul_records(bytes: &[u8]) -> Result<Vec<&[u8]>, String> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let Some(body) = bytes.strip_suffix(&[0]) else {
        return Err("Git path collection failed: incomplete NUL-delimited output".to_string());
    };
    Ok(body.split(|byte| *byte == 0).collect())
}

fn parse_status_paths(bytes: &[u8]) -> Result<Vec<(PathBuf, String)>, String> {
    let records = nul_records(bytes)?;
    let mut paths = Vec::new();
    let mut index = 0;
    while index < records.len() {
        let record = records[index];
        if record.len() < 4 || record[2] != b' ' {
            return Err("git status path collection failed: malformed record".to_string());
        }
        let status = String::from_utf8_lossy(&record[..2]);
        paths.push((
            git_path_from_bytes(&record[3..])?,
            format!("git status: {status}"),
        ));
        index += 1;
        if record[..2].iter().any(|byte| matches!(*byte, b'R' | b'C')) {
            let old = records
                .get(index)
                .filter(|path| !path.is_empty())
                .ok_or_else(|| {
                    "git status path collection failed: missing rename/copy source".to_string()
                })?;
            paths.push((
                git_path_from_bytes(old)?,
                format!("git status: {status} (rename/copy source)"),
            ));
            index += 1;
        }
    }
    Ok(paths)
}

fn parse_numstat_paths(bytes: &[u8]) -> Result<Vec<PathBuf>, String> {
    let records = nul_records(bytes)?;
    let mut paths = Vec::new();
    let mut index = 0;
    while index < records.len() {
        let mut fields = records[index].splitn(3, |byte| *byte == b'\t');
        let path = fields.nth(2).ok_or_else(|| {
            "git diff path collection failed: malformed numstat record".to_string()
        })?;
        index += 1;
        if path.is_empty() {
            for _ in 0..2 {
                let path = records
                    .get(index)
                    .filter(|path| !path.is_empty())
                    .ok_or_else(|| {
                        "git diff path collection failed: missing rename path".to_string()
                    })?;
                paths.push(git_path_from_bytes(path)?);
                index += 1;
            }
        } else {
            paths.push(git_path_from_bytes(path)?);
        }
    }
    Ok(paths)
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
    /// Presence before the first mutation; later snapshots must not replace it.
    pub before_state: BeforeState,
    pub after_last: Option<String>,
    /// Authoritative per-write diffs (`- `/`+ ` lines) recorded at write time, in
    /// write order. Large-file snapshots are truncated and unreliable, so
    /// rendering and patch generation prefer these diffs.
    pub diffs: Vec<String>,
}

impl FileChange {
    pub(crate) fn before_unavailable(&self) -> bool {
        self.before_state == BeforeState::Unknown
            || (self.before_state == BeforeState::Present && self.before_first.is_none())
    }

    /// Only known absence can stand in for an empty preimage in a comparison.
    pub(crate) fn snapshots_full(&self) -> bool {
        !self.before_unavailable()
            && !self.before_first.as_deref().is_some_and(is_capped)
            && !self.after_last.as_deref().is_some_and(is_capped)
    }

    pub(crate) fn unavailable_notice(&self) -> &'static str {
        if self.before_state == BeforeState::Unknown {
            "…[net diff unavailable: initial file presence unknown; before snapshot unavailable; contents not compared]"
        } else {
            "…[net diff unavailable: file was present but before snapshot unavailable; contents not compared]"
        }
    }
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

pub(crate) fn grouped_changes(entries: &[MutationEntry]) -> Vec<FileChange> {
    let cwd = cwd_for_display();
    let mut grouped: Vec<FileChange> = Vec::new();
    for e in entries {
        if let Some(g) = grouped.iter_mut().find(|g| g.path == e.path) {
            g.after_last.clone_from(&e.after);
            // Keep the initial state and use only the last postimage for net status.
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
                net: String::new(), // Computed after grouping.
                write_count: if e.op == "write" { 1 } else { 0 },
                delete_count: if e.op == "delete" { 1 } else { 0 },
                before_first: e.before.clone(),
                before_state: e.effective_before_state(),
                after_last: e.after.clone(),
                diffs: e.diff.iter().cloned().collect(),
            });
        }
    }
    for g in &mut grouped {
        let last_deleted = g.after_last.is_none() && g.delete_count > 0;
        // Missing contents do not establish that the file was created.
        g.net = if last_deleted {
            "deleted".to_string()
        } else if g.before_state == BeforeState::Absent && g.after_last.is_some() {
            "created".to_string()
        } else if g.before_state == BeforeState::Unknown {
            "written".to_string()
        } else {
            "modified".to_string()
        };
    }
    grouped
}

/// Renderable diff block for one file. When snapshots are complete it keeps the
/// existing net-diff logic (small-file behavior unchanged); when truncated it
/// uses the authoritative diff recorded at write time so the truncation edge is
/// never rendered as a deletion.
pub(crate) fn file_snippet(g: &FileChange, max_lines: usize) -> Option<String> {
    if g.before_unavailable() {
        return Some(g.unavailable_notice().to_string());
    }
    if g.snapshots_full() {
        return diff_snippet(
            g.before_first.as_deref(),
            g.after_last.as_deref(),
            max_lines,
        );
    }
    if !g.diffs.is_empty() {
        return diff_block_from_lines(&g.diffs.join(""), max_lines);
    }
    // Legacy capped snapshots cannot prove either a net diff or equality.
    Some("…[snapshots truncated; contents not compared; no recorded diff available]".to_string())
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

pub(crate) fn diff_snippet(
    before: Option<&str>,
    after: Option<&str>,
    max_lines: usize,
) -> Option<String> {
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

/// Derive a combined patch from mutation records in a viewer-compatible unified format.
pub fn mutation_patch(entries: &[MutationEntry]) -> Option<String> {
    let grouped = grouped_changes(entries);
    if grouped.is_empty() {
        return None;
    }
    let mut out = String::new();
    for g in &grouped {
        out.push_str(&format!("diff -- {} {}\n", g.rel, g.rel));
        if g.before_unavailable() {
            // Do not invent an empty preimage or apply later incremental diffs
            // as though they described the unavailable initial snapshot.
            out.push_str(g.unavailable_notice());
            out.push_str("\n\n");
            continue;
        }
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
                if !g.snapshots_full() {
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
                if !g.snapshots_full() {
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
                if !g.snapshots_full() {
                    // With truncated snapshots, use the authoritative diff
                    // directly (`- `/`+ ` lines are the patch body).
                    let body = g.diffs.join("");
                    out.push_str(&body);
                    if !body.ends_with('\n') {
                        out.push('\n');
                    }
                    if g.diffs.is_empty() {
                        out.push_str("…[snapshots truncated; contents not compared; no recorded diff available]\n");
                    }
                } else if let Some(snippet) =
                    diff_snippet(g.before_first.as_deref(), g.after_last.as_deref(), 200)
                {
                    // Strip the Markdown fence and retain only diff lines.
                    for line in snippet.lines() {
                        if line.starts_with("```") || line.starts_with("（差异") {
                            continue;
                        }
                        out.push_str(line);
                        out.push('\n');
                    }
                } else if g.before_first != g.after_last {
                    // Fall back to the complete before/after snapshots.
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

/// Prefer a mutation-derived patch, then fall back to `git diff HEAD`.
pub fn combined_patch() -> Option<String> {
    let entries = read_all();
    if !entries.is_empty() {
        if let Some(p) = mutation_patch(&entries) {
            return Some(p);
        }
    }
    // Fall back to Git.
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd().ok()?;
    if !is_inside_git_work_tree(&cwd) {
        return None;
    }
    let diff = git_output(&cwd, &["diff", "HEAD", "--no-color"])?;
    if diff.trim().is_empty() {
        // Try the worktree diff without HEAD when the initial diff is empty.
        let diff2 = git_output(&cwd, &["diff", "--no-color"])?;
        if diff2.trim().is_empty() {
            return None;
        }
        return Some(diff2);
    }
    Some(diff)
}

/// Write a combined patch to session assets or a process-specific temporary
/// path when no session is active, returning the absolute path.
pub fn write_combined_patch() -> Result<PathBuf, String> {
    let patch = combined_patch()
        .ok_or_else(|| "当前会话无可导出的变更（无 mutation log 且 git 无差异）".to_string())?;
    // Prefer session assets; without an active session, use the runtime temp
    // directory so noninteractive `/changes --open` can still export a patch.
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
        // Include the process ID to avoid concurrent one-shot exports overwriting each other.
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
            "Recorded snapshots and per-write diffs: {}\nSnapshots may be truncated or unavailable; read the log for recorded evidence.\n",
            lp.display()
        ));
    }
    // Add a Git-status hint only when the caller allows workspace context.
    if include_git_extra {
        if let Some(cwd) = cwd {
            if is_inside_git_work_tree(&cwd) {
                let status = git_output(&cwd, &["status", "--porcelain=v1"]).unwrap_or_default();
                if !status.trim().is_empty() {
                    // Estimate the number of Git paths outside the mutation log.
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

pub(crate) fn format_mutation_snippets(grouped: &[FileChange], log_path: Option<&Path>) -> String {
    let mut out = String::from("## Session mutation snippets (first-change order)\n\n");
    const CAP: usize = 14_000;
    let mut shown = 0;
    for (i, g) in grouped.iter().enumerate() {
        if out.len() >= CAP {
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
        out.push_str(&format!(
            "{}. {:?}  [{}]  ({counts})\n",
            i + 1,
            g.rel,
            g.net
        ));
        if let Some(snippet) = file_snippet(g, 30) {
            let remaining = CAP.saturating_sub(out.len());
            let mut end = snippet.len().min(remaining);
            while !snippet.is_char_boundary(end) {
                end -= 1;
            }
            out.push_str(&snippet[..end]);
            if end < snippet.len() {
                if snippet.starts_with("```diff\n") && !snippet[..end].contains("\n```") {
                    out.push_str("\n```");
                }
                out.push_str("\n[mutation snippet truncated]\n");
            }
            out.push_str("\n\n");
        }
        shown += 1;
    }
    if shown < grouped.len() {
        out.push_str(&format!("[Mutation snippets shown for {shown} of {} files; the complete path inventory is separate]\n\n", grouped.len()));
    }
    if let Some(lp) = log_path {
        out.push_str(&format!(
            "Recorded snapshots and per-write diffs: {}\nSnapshots may be truncated or unavailable; read the log for recorded evidence.\n",
            lp.display()
        ));
    }
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

/// Export changes and open the requested viewer, returning a status message.
pub fn open_changes(requested: Option<EditorKind>) -> Result<String, String> {
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let editor = resolve_editor(requested, &cwd);
    let patch_path = write_combined_patch()?;

    // Dispatch to the resolved viewer.
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
    // Prefer per-file comparison for a single file, then open the patch.
    let grouped = session_grouped_changes();
    if grouped.len() == 1 {
        let g = &grouped[0];
        // Prepare comparison files in the session temp directory, matching
        // service::changes::try_open_single_file_diff.
        let tmp =
            crate::ai::driver::runtime_ctx::temp_dir().unwrap_or_else(|_| std::env::temp_dir());
        let before_path = tmp.join(format!("a_changes_before_{}", sanitize_filename(&g.rel)));
        let after_path = tmp.join(format!("a_changes_after_{}", sanitize_filename(&g.rel)));
        // If writing comparison files fails, fall back to opening the patch.
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
        if before_ok && after_ok && g.snapshots_full() {
            // Launch the two-file comparison in the background.
            let mut cmd = Command::new("code");
            cmd.arg("--diff")
                .arg(&before_path)
                .arg(&after_path)
                .current_dir(cwd)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            // Missing executables fail when spawning the viewer.
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
    // Fall back to opening the patch in VS Code.
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
    // Probe the usual JetBrains Toolbox executable names.
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
    // Compare all files with `--dir-diff`, disable prompts, and launch
    // in the background without waiting for the viewer.
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

    fn empty_inventory() -> ChangeInventory {
        ChangeInventory {
            files: Vec::new(),
            session_changes: Vec::new(),
            is_git_repo: false,
            collection_errors: Vec::new(),
            coverage_gaps: Vec::new(),
            log_path: None,
            git_stats: Vec::new(),
            path_indices: FastMap::default(),
        }
    }

    #[test]
    fn collect_current_changes_unions_nonempty_mutations_with_all_git_paths() {
        let root =
            std::env::temp_dir().join(format!("audit-inventory-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&root).unwrap();
        let root = std::fs::canonicalize(root).unwrap();
        checked_git_output(&root, &["init", "-q"]).unwrap();
        for name in ["tracked.rs", "deleted.rs", "old\tname.rs"] {
            std::fs::write(root.join(name), "first\nsecond\nthird\n").unwrap();
        }
        checked_git_output(&root, &["add", "."]).unwrap();
        checked_git_output(
            &root,
            &[
                "-c",
                "user.name=Audit Tests",
                "-c",
                "user.email=audit-tests@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "baseline",
            ],
        )
        .unwrap();
        checked_git_output(&root, &["mv", "old\tname.rs", "renamed\nfile.rs"]).unwrap();
        std::fs::write(root.join("staged.rs"), "staged\n").unwrap();
        checked_git_output(&root, &["add", "staged.rs"]).unwrap();
        std::fs::write(root.join("tracked.rs"), "updated\n").unwrap();
        std::fs::remove_file(root.join("deleted.rs")).unwrap();
        std::fs::write(root.join("untracked\t\nfile.txt"), "new\n").unwrap();
        let entries = [MutationEntry {
            seq: 1,
            ts: "t".into(),
            path: root.join("mutation-only.rs").display().to_string(),
            op: "delete".into(),
            before: Some("removed\n".into()),
            before_state: Some(BeforeState::Present),
            after: None,
            diff: None,
        }];
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join(".git/audit-test/history.jsonl");
        app.session_id = "inventory-test".into();
        let context = crate::ai::driver::runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        let inventory =
            crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || {
                crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(context, || {
                    let log_path = crate::ai::tools::storage::mutation_log::log_path().unwrap();
                    assert!(log_path.starts_with(root.join(".git")));
                    std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
                    std::fs::write(
                        &log_path,
                        serde_json::to_string(&entries[0]).unwrap() + "\n",
                    )
                    .unwrap();
                    let session_only = collect_current_changes(false);
                    assert_eq!(session_only.files.len(), 1);
                    assert_eq!(session_only.files[0].path, entries[0].path);
                    collect_current_changes(true)
                })
            });
        assert_eq!(inventory.session_changes.len(), 1);
        assert!(inventory.collection_complete());
        assert!(
            inventory.collection_errors.is_empty(),
            "{:?}",
            inventory.collection_errors
        );
        for name in [
            "mutation-only.rs",
            "tracked.rs",
            "deleted.rs",
            "old\tname.rs",
            "renamed\nfile.rs",
            "staged.rs",
            "untracked\t\nfile.txt",
        ] {
            let path = root.join(name).display().to_string();
            assert!(
                inventory.files.iter().any(|file| file.path == path),
                "missing {name:?}"
            );
        }
        assert_eq!(inventory.files.len(), 7);
        assert!(
            inventory
                .files
                .iter()
                .all(|file| file.ownership == "unknown")
        );
        let staged = inventory
            .files
            .iter()
            .find(|file| file.path.ends_with("/staged.rs"))
            .unwrap();
        assert!(
            staged
                .observations
                .iter()
                .any(|source| source.contains("--cached --stat"))
        );
        let tracked = inventory
            .files
            .iter()
            .find(|file| file.path.ends_with("/tracked.rs"))
            .unwrap();
        assert!(
            tracked
                .observations
                .iter()
                .any(|source| source.contains("--stat (unstaged)"))
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn collect_current_changes_inventory_survives_caps_and_exposes_collection_failures() {
        let mut inventory = empty_inventory();
        assert!(!inventory.collection_complete());
        inventory.log_path = Some(PathBuf::from("/proj/mutation_log.jsonl"));
        assert!(inventory.collection_complete());
        let entries = [
            MutationEntry {
                seq: 1,
                ts: "t".into(),
                path: "/proj/first.rs".into(),
                op: "write".into(),
                before: None,
                before_state: Some(BeforeState::Absent),
                after: Some("x".repeat(20_000)),
                diff: None,
            },
            MutationEntry {
                seq: 2,
                ts: "t".into(),
                path: "/proj/later.rs".into(),
                op: "delete".into(),
                before: Some("old\n".into()),
                before_state: Some(BeforeState::Present),
                after: None,
                diff: None,
            },
        ];
        inventory.session_changes = grouped_changes(&entries);
        for entry in &entries {
            inventory.add_path(
                PathBuf::from(&entry.path),
                format!("session mutation: {}", entry.op),
            );
        }
        assert!(inventory.format_inventory().contains("/proj/later.rs"));
        assert!(!inventory.format_snippets().contains("/proj/later.rs"));
        assert!(inventory.format_snippets().contains("1 of 2 files"));
        let valid = serde_json::to_string(&entries[0]).unwrap();
        let parsed = parse_inventory_mutations(
            &format!("{valid}\ninvalid JSON\n"),
            &mut inventory.collection_errors,
        );
        assert_eq!(parsed.len(), 1);
        assert!(inventory.collection_errors[0].contains("record 2"));
        assert!(!inventory.collection_complete());
        assert!(inventory.format_inventory().contains("Collection failures"));
        assert!(parse_status_paths(b"R  new\0").is_err());
        assert!(parse_numstat_paths(b"1\t1\t\0old\0").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn inventory_keeps_non_utf8_git_filename_bytes() {
        use std::os::unix::ffi::OsStrExt;
        let paths = parse_status_paths(b"?? name-\xff\0").unwrap();
        assert_eq!(paths[0].0.as_os_str().as_bytes(), b"name-\xff");
        let mut inventory = empty_inventory();
        inventory.add_path(paths[0].0.clone(), paths[0].1.clone());
        assert_eq!(
            inventory.files[0].path_bytes_hex.as_deref(),
            Some("6e616d652dff")
        );
    }

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
    fn unavailable_initial_snapshot_survives_later_mutations_and_renderers() {
        for state in [BeforeState::Present, BeforeState::Unknown] {
            let mut entries = vec![MutationEntry {
                seq: 0,
                ts: "t".into(),
                path: "/proj/existing.rs".into(),
                op: "write".into(),
                before: None,
                before_state: Some(state),
                after: Some("first\n".into()),
                diff: None,
            }];
            let later = [
                (
                    "write",
                    Some("first\n"),
                    Some("second\n"),
                    BeforeState::Present,
                ),
                ("delete", Some("second\n"), None, BeforeState::Present),
                ("write", None, Some("recreated\n"), BeforeState::Absent),
            ];
            for index in 0..=later.len() {
                if index > 0 {
                    let (op, before, after, before_state) = later[index - 1];
                    entries.push(MutationEntry {
                        seq: index as u64,
                        ts: "t".into(),
                        path: entries[0].path.clone(),
                        op: op.into(),
                        before: before.map(str::to_string),
                        before_state: Some(before_state),
                        after: after.map(str::to_string),
                        diff: crate::ai::tools::storage::mutation_log::entry_diff(before, after),
                    });
                }
                let grouped = grouped_changes(&entries);
                let g = &grouped[0];
                assert_eq!(g.before_state, state);
                assert!(g.before_first.is_none());
                assert!(!g.snapshots_full());
                let expected = if index == 2 {
                    "deleted"
                } else if state == BeforeState::Present {
                    "modified"
                } else {
                    "written"
                };
                assert_eq!(g.net, expected);
                for rendered in [
                    file_snippet(g, 30).unwrap(),
                    mutation_patch(&entries).unwrap(),
                ] {
                    assert!(
                        rendered.contains("before snapshot unavailable"),
                        "{rendered}"
                    );
                    assert!(rendered.contains("not compared"), "{rendered}");
                    assert!(!rendered.contains("/dev/null"), "{rendered}");
                    assert!(
                        !rendered
                            .lines()
                            .any(|line| line.starts_with('+') || line.starts_with('-')),
                        "{rendered}"
                    );
                }
            }
        }
    }

    #[test]
    fn known_absence_and_empty_preimage_remain_distinct() {
        for (state, before, expected) in [
            (BeforeState::Absent, None, "created"),
            (BeforeState::Present, Some(""), "modified"),
        ] {
            let entry = MutationEntry {
                seq: 0,
                ts: "t".into(),
                path: "/proj/file.rs".into(),
                op: "write".into(),
                before: before.map(str::to_string),
                before_state: Some(state),
                after: Some("new\n".into()),
                diff: None,
            };
            let grouped = grouped_changes(&[entry]);
            assert_eq!(grouped[0].net, expected);
            assert!(grouped[0].snapshots_full());
            assert!(file_snippet(&grouped[0], 30).unwrap().contains("+ new"));
        }
    }

    #[test]
    fn capped_identical_snapshots_keep_comparison_uncertainty() {
        use crate::ai::tools::storage::mutation_log::{cap_content, entry_diff};
        let content = "same line\n".repeat(200_000);
        for diff in [None, entry_diff(Some(&content), Some(&content))] {
            let entry = MutationEntry {
                seq: 0,
                ts: "t".into(),
                path: "/proj/large.rs".into(),
                op: "write".into(),
                before: Some(cap_content(&content)),
                before_state: Some(BeforeState::Present),
                after: Some(cap_content(&content)),
                diff,
            };
            let grouped = grouped_changes(std::slice::from_ref(&entry));
            for rendered in [
                file_snippet(&grouped[0], 30).unwrap(),
                mutation_patch(&[entry]).unwrap(),
            ] {
                assert!(rendered.contains("not compared"), "{rendered}");
                assert!(!rendered.contains("differ beyond"), "{rendered}");
            }
        }
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
                before_state: None,
                diff: Some("+ hello\n".into()),
            },
            MutationEntry {
                seq: 1,
                ts: "t".into(),
                path: "/tmp/proj/b.rs".into(),
                op: "write".into(),
                before: Some("old".into()),
                after: Some("new".into()),
                before_state: None,
                diff: Some("- old\n+ new\n".into()),
            },
            MutationEntry {
                seq: 2,
                ts: "t".into(),
                path: "/tmp/proj/b.rs".into(),
                op: "write".into(),
                before: Some("new".into()),
                after: Some("new2".into()),
                before_state: None,
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
            before_state: None,
            diff: Some("+ line1\n+ line2\n".into()),
        }];
        let patch = mutation_patch(&entries).unwrap();
        assert!(patch.contains("--- /dev/null"));
        assert!(patch.contains("+++ b/"));
        assert!(patch.contains("+line1"));
    }

    #[test]
    fn write_combined_patch_no_session_falls_back_to_temp_dir() {
        // Without an active driver context, patch export must fall back to the
        // runtime temp directory rather than fail for lack of session assets.
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
                    e.contains("无可导出的变更"),
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
                before_state: None,
                diff: entry_diff(Some(&before), Some(&after)),
            },
            MutationEntry {
                seq: 1,
                ts: "t".into(),
                path: "/proj/big.rs".into(),
                op: "write".into(),
                before: Some(cap_content(&after)),
                after: Some(cap_content(&after2)),
                before_state: None,
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
        assert!(!g.snapshots_full());
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
                before_state: None,
                diff: entry_diff(None, Some(&v1)),
            },
            MutationEntry {
                seq: 1,
                ts: "t".into(),
                path: "/proj/created.rs".into(),
                op: "write".into(),
                before: Some(cap_content(&v1)),
                after: Some(cap_content(&v2)),
                before_state: None,
                diff: entry_diff(Some(&v1), Some(&v2)),
            },
        ];
        let grouped = grouped_changes(&entries);
        assert_eq!(grouped.len(), 1);
        let g = &grouped[0];
        assert_eq!(g.net, "created");
        assert_eq!(g.diffs.len(), 2);
        assert!(!g.snapshots_full());
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
            before_state: None,
            // Old log format: no authoritative diff recorded.
            diff: None,
        }];
        let grouped = grouped_changes(&entries);
        assert_eq!(grouped.len(), 1);
        let g = &grouped[0];
        assert_eq!(g.net, "deleted");
        assert!(g.diffs.is_empty());
        assert!(!g.snapshots_full());
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
