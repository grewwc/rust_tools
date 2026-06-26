use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::storage::file_store::FileStore;

fn params_apply_patch() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Preferred absolute path to the file to patch (some sensitive paths are blocked). The runtime also accepts `path` as a compatibility alias."
            },
            "patch": {
                "type": "string",
                "description": "Patch text. Accepted formats: raw unified-diff hunks starting with @@, or a single-file `*** Begin Patch` envelope with `*** Update File:` / `*** Add File:`."
            }
        },
        "required": ["file_path", "patch"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "apply_patch",
        description: "Apply a localized patch to one file. Supports raw unified-diff hunks and the common single-file `*** Begin Patch` envelope. Prefer this for updating an existing document or source file with the smallest localized change instead of rewriting the entire file. Creates missing parent directories; fails if context/removals do not match.",
        parameters: params_apply_patch,
        execute: execute_apply_patch,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["executor", "builtin", "core"],
    }
});

#[derive(Debug, Clone)]
struct UnifiedHunk {
    old_start: usize,
    lines: Vec<UnifiedLine>,
}

#[derive(Debug, Clone)]
enum UnifiedLine {
    Context(String),
    Remove(String),
    Add(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatchEnvelopeOp {
    Update,
    Add,
}

#[derive(Debug, Clone)]
struct PatchEnvelope {
    op: PatchEnvelopeOp,
    target_path: String,
    body_lines: Vec<String>,
}

fn parse_unified_hunks(patch: &str) -> Result<Vec<UnifiedHunk>, String> {
    let mut hunks = Vec::new();
    let mut iter = patch.lines().peekable();
    while let Some(line) = iter.next() {
        let Some(rest) = line.strip_prefix("@@") else {
            continue;
        };
        let rest = rest.trim();
        let Some(rest) = rest.strip_prefix('-') else {
            return Err("invalid hunk header".to_string());
        };
        let mut parts = rest.split_whitespace();
        let old_part = parts.next().ok_or("invalid hunk header")?;
        let _new_part = parts.next().ok_or("invalid hunk header")?;

        let old_start = old_part
            .split(',')
            .next()
            .ok_or("invalid hunk header")?
            .parse::<isize>()
            .map_err(|_| "invalid hunk header")?;
        let old_start = if old_start <= 0 {
            0
        } else {
            old_start as usize
        };

        let mut lines = Vec::new();
        while let Some(next) = iter.peek().copied() {
            if next.starts_with("@@") {
                break;
            }
            let l = iter.next().unwrap_or_default();
            if l.starts_with("\\ No newline at end of file") {
                continue;
            }
            let mut chars = l.chars();
            let prefix = chars
                .next()
                .ok_or_else(|| "invalid hunk line: empty".to_string())?;
            let body = chars.as_str();
            match prefix {
                ' ' => lines.push(UnifiedLine::Context(body.to_string())),
                '-' => lines.push(UnifiedLine::Remove(body.to_string())),
                '+' => lines.push(UnifiedLine::Add(body.to_string())),
                _ => return Err(format!("invalid hunk line: {}", l)),
            }
        }
        hunks.push(UnifiedHunk { old_start, lines });
    }
    if hunks.is_empty() {
        return Err("no hunks found".to_string());
    }
    Ok(hunks)
}

fn optional_file_path_arg(args: &Value) -> Option<&str> {
    args.get("file_path")
        .or_else(|| args.get("path"))
        .and_then(Value::as_str)
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            std::path::Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

fn effective_base_dir() -> PathBuf {
    crate::ai::driver::runtime_ctx::effective_cwd()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn resolve_patch_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return normalize_lexical(path);
    }
    normalize_lexical(&effective_base_dir().join(path))
}

fn ensure_patch_target_matches(target_path: &Path, envelope_path: &str) -> Result<(), String> {
    let resolved_target = resolve_patch_path(target_path);
    let resolved_envelope = resolve_patch_path(Path::new(envelope_path));
    if resolved_target == resolved_envelope {
        return Ok(());
    }
    Err(format!(
        "patch target mismatch: tool arg points to {}, but patch envelope points to {}. Rebuild the patch for the same file before retrying.",
        target_path.display(),
        envelope_path
    ))
}

fn parse_patch_envelope(patch: &str) -> Result<Option<PatchEnvelope>, String> {
    let mut lines = patch.lines();
    let Some(first) = lines.find(|line| !line.trim().is_empty()) else {
        return Ok(None);
    };
    if first.trim() != "*** Begin Patch" {
        return Ok(None);
    }

    let header = lines
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| "invalid patch envelope: missing file header".to_string())?;
    let (op, target_path) = if let Some(path) = header.strip_prefix("*** Update File: ") {
        (PatchEnvelopeOp::Update, path.trim())
    } else if let Some(path) = header.strip_prefix("*** Add File: ") {
        (PatchEnvelopeOp::Add, path.trim())
    } else if header.starts_with("*** Delete File: ") {
        return Err("apply_patch does not support Delete File envelopes".to_string());
    } else {
        return Err(
            "invalid patch envelope: expected `*** Update File:` or `*** Add File:`"
                .to_string(),
        );
    };

    let mut body_lines = Vec::new();
    let mut ended = false;
    for line in lines {
        if line == "*** End Patch" {
            ended = true;
            break;
        }
        if line == "*** End of File" {
            continue;
        }
        if line.starts_with("*** Update File: ")
            || line.starts_with("*** Add File: ")
            || line.starts_with("*** Delete File: ")
        {
            return Err("multi-file patch not supported: apply_patch edits one file per call"
                .to_string());
        }
        body_lines.push(line.to_string());
    }
    if !ended {
        return Err("invalid patch envelope: missing `*** End Patch`".to_string());
    }
    Ok(Some(PatchEnvelope {
        op,
        target_path: target_path.to_string(),
        body_lines,
    }))
}

fn normalize_patch_text(path: &Path, patch: &str) -> Result<(String, String), String> {
    let Some(envelope) = parse_patch_envelope(patch)? else {
        return Ok((path.display().to_string(), patch.to_string()));
    };

    ensure_patch_target_matches(path, &envelope.target_path)?;
    let normalized_patch = match envelope.op {
        PatchEnvelopeOp::Update => envelope.body_lines.join("\n"),
        PatchEnvelopeOp::Add => {
            if path.exists() {
                return Err(
                    "Add File patch targets an existing file. Use Update File or write_file instead."
                        .to_string(),
                );
            }
            for line in &envelope.body_lines {
                if !line.starts_with('+') {
                    return Err(format!(
                        "invalid Add File line: {}. Every content line in an Add File envelope must start with `+`.",
                        line
                    ));
                }
            }
            let mut normalized =
                format!("@@ -0,0 +1,{} @@", envelope.body_lines.len());
            if !envelope.body_lines.is_empty() {
                normalized.push('\n');
                normalized.push_str(&envelope.body_lines.join("\n"));
            }
            normalized
        }
    };
    Ok((envelope.target_path, normalized_patch))
}

fn lines_match(actual: &str, expected: &str) -> bool {
    actual == expected || actual.trim_end() == expected.trim_end()
}

fn find_hunk_offset(orig_lines: &[String], hunk: &UnifiedHunk) -> Option<usize> {
    let context_and_remove: Vec<&str> = hunk
        .lines
        .iter()
        .filter_map(|line| match line {
            UnifiedLine::Context(s) | UnifiedLine::Remove(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    if context_and_remove.is_empty() {
        return None;
    }
    let nominal = hunk.old_start.saturating_sub(1);
    let search_radius = 50usize;
    let start = nominal.saturating_sub(search_radius);
    let end = (nominal + search_radius).min(orig_lines.len());
    for candidate in start..end {
        if candidate + context_and_remove.len() > orig_lines.len() {
            continue;
        }
        let all_match = context_and_remove
            .iter()
            .enumerate()
            .all(|(i, expected)| lines_match(&orig_lines[candidate + i], expected));
        if all_match {
            return Some(candidate);
        }
    }
    None
}

/// 提取 hunk 的 context+remove 行（即"期望在原文件中匹配到"的行）。
fn hunk_expected_lines(hunk: &UnifiedHunk) -> Vec<&str> {
    hunk.lines
        .iter()
        .filter_map(|line| match line {
            UnifiedLine::Context(s) | UnifiedLine::Remove(s) => Some(s.as_str()),
            _ => None,
        })
        .collect()
}

/// 在全文件范围内统计 hunk 的 context+remove 块能匹配到的位置（0-based 行号）。
/// 用于检测"多处匹配"歧义，避免静默改错地方。
fn all_hunk_match_positions(orig_lines: &[String], hunk: &UnifiedHunk) -> Vec<usize> {
    let expected = hunk_expected_lines(hunk);
    if expected.is_empty() {
        return Vec::new();
    }
    let mut positions = Vec::new();
    let mut candidate = 0usize;
    while candidate + expected.len() <= orig_lines.len() {
        let all_match = expected
            .iter()
            .enumerate()
            .all(|(i, exp)| lines_match(&orig_lines[candidate + i], exp));
        if all_match {
            positions.push(candidate);
        }
        candidate += 1;
    }
    positions
}

/// 构造带上下文的 "context mismatch" 错误：列出 patch 期望匹配的行，以及原文件
/// 在标称位置附近的实际行，帮助模型快速自我修正，而不是只看到一句 "context mismatch"。
fn describe_context_mismatch(orig_lines: &[String], hunk: &UnifiedHunk) -> String {
    let expected = hunk_expected_lines(hunk);
    let nominal = hunk.old_start.saturating_sub(1);

    let mut msg = String::from("context mismatch: patch hunk could not be located.\n");
    msg.push_str(&format!(
        "Hunk header declared @@ -{} (1-based line {}).\n",
        hunk.old_start, hunk.old_start
    ));

    msg.push_str("Patch expected these lines (context/removed):\n");
    for (i, line) in expected.iter().take(10).enumerate() {
        msg.push_str(&format!("  expected[{}]: {}\n", i, line));
    }
    if expected.len() > 10 {
        msg.push_str(&format!(
            "  ... ({} more expected lines)\n",
            expected.len() - 10
        ));
    }

    // 标称位置附近的实际文件内容（前后各 3 行窗口）。
    let win_start = nominal.saturating_sub(3);
    let win_end = (nominal + expected.len().max(1) + 3).min(orig_lines.len());
    if win_start < win_end {
        msg.push_str(&format!(
            "Actual file content around line {} (1-based):\n",
            win_start + 1
        ));
        for (offset, line) in orig_lines[win_start..win_end].iter().enumerate() {
            msg.push_str(&format!("  {:>6}: {}\n", win_start + offset + 1, line));
        }
    } else {
        msg.push_str(&format!(
            "File has {} line(s); declared position is out of range.\n",
            orig_lines.len()
        ));
    }

    msg.push_str(
        "Hint: re-read the file with read_file/read_file_lines to get exact current content, then rebuild the patch from the raw file text only. Do not copy the leading line numbers, any truncation notice, or the Symbol outline block into the patch.",
    );
    msg
}

fn try_apply_hunk_at(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    start: usize,
) -> Option<(Vec<String>, usize)> {
    let mut out = Vec::new();
    let mut idx = start;
    for line in &hunk.lines {
        match line {
            UnifiedLine::Context(s) => {
                let cur = orig_lines.get(idx)?;
                if !lines_match(cur, s) {
                    return None;
                }
                out.push(cur.clone());
                idx += 1;
            }
            UnifiedLine::Remove(s) => {
                let cur = orig_lines.get(idx)?;
                if !lines_match(cur, s) {
                    return None;
                }
                idx += 1;
            }
            UnifiedLine::Add(s) => {
                out.push(s.clone());
            }
        }
    }
    Some((out, idx))
}

fn apply_unified_patch(original: &str, patch: &str) -> Result<String, String> {
    let had_trailing_newline = original.ends_with('\n');
    let hunks = parse_unified_hunks(patch)?;
    let orig_lines: Vec<String> = original.lines().map(|s| s.to_string()).collect();

    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize;

    for hunk in &hunks {
        let nominal = hunk.old_start.saturating_sub(1);
        let nominal_ok = nominal <= orig_lines.len()
            && nominal >= cursor
            && try_apply_hunk_at(&orig_lines, hunk, nominal).is_some();

        let apply_at = if nominal_ok {
            nominal
        } else {
            // 标称位置匹配不上时，先检查全文件范围内有多少处能匹配：
            // 多处匹配说明 hunk 的 context 不足以唯一定位，强行用第一处会改错地方。
            let positions = all_hunk_match_positions(&orig_lines, hunk);
            let forward: Vec<usize> = positions.iter().copied().filter(|&p| p >= cursor).collect();
            if forward.len() > 1 {
                let shown: Vec<String> = forward
                    .iter()
                    .take(5)
                    .map(|p| (p + 1).to_string())
                    .collect();
                return Err(format!(
                    "ambiguous patch: hunk context matches {} locations (1-based lines: {}{}). \
                     Add more surrounding context lines to the hunk so it uniquely identifies the target.",
                    forward.len(),
                    shown.join(", "),
                    if forward.len() > 5 { ", ..." } else { "" }
                ));
            }
            match find_hunk_offset(&orig_lines, hunk) {
                Some(offset) => {
                    if offset < cursor {
                        return Err("hunks out of order".to_string());
                    }
                    offset
                }
                None => return Err(describe_context_mismatch(&orig_lines, hunk)),
            }
        };

        out.extend_from_slice(&orig_lines[cursor..apply_at]);
        let (hunk_out, new_idx) = try_apply_hunk_at(&orig_lines, hunk, apply_at)
            .ok_or_else(|| describe_context_mismatch(&orig_lines, hunk))?;
        out.extend(hunk_out);
        cursor = new_idx;
    }

    out.extend_from_slice(&orig_lines[cursor..]);
    let mut s = out.join("\n");
    if had_trailing_newline {
        s.push('\n');
    }
    Ok(s)
}

pub(crate) fn execute_apply_patch(args: &Value) -> Result<String, String> {
    let patch = args["patch"].as_str().ok_or("missing patch")?;
    let initial_file_path = optional_file_path_arg(args);
    let envelope = parse_patch_envelope(patch)?;
    let file_path = initial_file_path
        .map(str::to_string)
        .or_else(|| envelope.as_ref().map(|parsed| parsed.target_path.clone()))
        .ok_or("missing file_path")?;

    let store = FileStore::new(PathBuf::from(&file_path));
    store.validate_write_access().map_err(|err| err.to_string())?;
    let path = store.path().to_path_buf();
    let (_, normalized_patch) = if let Some(envelope) = envelope {
        ensure_patch_target_matches(&path, &envelope.target_path)?;
        normalize_patch_text(&path, patch)?
    } else {
        (file_path.clone(), patch.to_string())
    };
    let original = if path.exists() {
        store.read_to_string().map_err(|err| err.to_string())?
    } else {
        String::new()
    };
    let next = apply_unified_patch(&original, &normalized_patch)?;
    store.write_all(&next).map_err(|err| err.to_string())?;
    Ok(format!("Successfully patched {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{apply_unified_patch, execute_apply_patch, parse_unified_hunks};
    use crate::ai::test_support::ENV_LOCK;
    use std::{fs, path::PathBuf};

    fn make_temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_patch_tools_test_{}_{}", name, uuid::Uuid::new_v4()));
        path
    }

    #[test]
    fn parse_unified_hunks_rejects_empty_hunk_line_instead_of_panicking() {
        let patch = "@@ -1,1 +1,1 @@\n\n-foo\n+bar\n";
        assert!(parse_unified_hunks(patch).is_err());
    }

    #[test]
    fn apply_unified_patch_applies_simple_hunk() {
        let original = "line1\nline2\nline3\n";
        let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n";
        let result = apply_unified_patch(original, patch).unwrap();
        assert_eq!(result, "line1\nchanged\nline3\n");
    }

    #[test]
    fn apply_unified_patch_context_mismatch_includes_actual_content() {
        let original = "alpha\nbeta\ngamma\n";
        // 期望删除一行不存在的内容，应触发带上下文的 context mismatch。
        let patch = "@@ -2,1 +2,1 @@\n-not_present\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        // 错误里应回显期望行与实际文件内容，便于模型自我修正。
        assert!(err.contains("not_present"), "err was: {err}");
        assert!(err.contains("beta"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_detects_ambiguous_match() {
        // 同样的行在文件里出现多次，且标称位置匹配不上，应报歧义错误。
        let original = "dup\nmid\ndup\ntail\n";
        let patch = "@@ -9,1 +9,1 @@\n-dup\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
    }

    #[test]
    fn execute_apply_patch_accepts_path_alias_and_begin_patch_envelope() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("update").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "alpha\nbeta\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "path": path.to_string_lossy(),
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\n@@ -1,2 +1,2 @@\n alpha\n-beta\n+changed\n*** End Patch\n",
                    path.display()
                )
            });
            execute_apply_patch(&args).expect("apply_patch should accept path alias and envelope");
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "alpha\nchanged\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_supports_add_file_envelope_without_file_path_arg() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("add_parent");
        let path = base.join("new.txt");
        fs::create_dir_all(&base).unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "patch": "*** Begin Patch\n*** Add File: new.txt\n+hello\n+world\n*** End Patch\n"
            });
            execute_apply_patch(&args).expect("apply_patch should infer target from Add File envelope");
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "hello\nworld");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_rejects_mismatched_envelope_target() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("mismatch_parent");
        let path = base.join("a.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "alpha\n").unwrap();

        let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "patch": "*** Begin Patch\n*** Update File: b.txt\n@@ -1,1 +1,1 @@\n-alpha\n+beta\n*** End Patch\n"
            });
            execute_apply_patch(&args).expect_err("mismatched target must be rejected")
        });

        assert!(err.contains("patch target mismatch"), "err was: {err}");
        let _ = fs::remove_dir_all(base);
    }
}
