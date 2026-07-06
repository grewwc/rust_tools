use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::common::ToolStreamWriter;
use crate::ai::tools::common::ToolStreamingRegistration;
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

inventory::submit!(ToolStreamingRegistration {
    name: "apply_patch",
    execute_streaming: execute_apply_patch_streaming,
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
            // 空行（含 CRLF 下只剩 \r 的行）：模型常把空 context 行写成完全没有
            // 前导空格的空行。按空 context 行处理，与 `git apply` 的宽容一致。
            if l == "" || l == "\r" {
                lines.push(UnifiedLine::Context(String::new()));
                continue;
            }
            let mut chars = l.chars();
            let prefix = chars
                .next()
                .ok_or_else(|| "invalid hunk line: empty".to_string())?;
            // 容忍 CRLF：剥离行尾 \r，避免 Add 行把 \r 写入文件内容。
            let body = chars.as_str().strip_suffix('\r').unwrap_or(chars.as_str());
            match prefix {
                ' ' => lines.push(UnifiedLine::Context(body.to_string())),
                '-' => lines.push(UnifiedLine::Remove(body.to_string())),
                '+' => lines.push(UnifiedLine::Add(body.to_string())),
                _ => return Err(format!("invalid hunk line: {}", l)),
            }
        }
        // 剥离尾部空 context 行：hunk body 循环只在遇到下一个 `@@` 才结束，所以
        // hunk 之间或 patch 末尾的空行（分隔/尾随）会被吞进当前 hunk，变成末尾的
        // 空 context 行，凭空要求原文件对应位置也有空行，导致本能匹配的 patch
        // 报 context mismatch。真实的中间空行后面必然还有本 hunk 的内容行，不会被
        // 误删；只有纯尾随的空 context 行才在此剥除。
        while matches!(lines.last(), Some(UnifiedLine::Context(s)) if s.is_empty()) {
            lines.pop();
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
            "invalid patch envelope: expected `*** Update File:` or `*** Add File:`".to_string(),
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
            return Err(
                "multi-file patch not supported: apply_patch edits one file per call".to_string(),
            );
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
        PatchEnvelopeOp::Update => {
            // *** Begin Patch 的 Update 格式允许省略 @@ hunk header（Cursor/Aider 风格），
            // 模型常只写 +/−/space 前缀行而不带 @@。如果 body 中没有任何 @@ header，
            // 合成一个，让 parse_unified_hunks 能识别。old_start=1 让匹配从文件开头
            // 尝试，失败后自动回退到全文件模糊搜索。
            let has_hunk_header = envelope.body_lines.iter().any(|l| l.starts_with("@@"));
            if has_hunk_header {
                envelope.body_lines.join("\n")
            } else {
                let mut normalized = String::from("@@ -1,0 +1,0 @@");
                if !envelope.body_lines.is_empty() {
                    normalized.push('\n');
                    normalized.push_str(&envelope.body_lines.join("\n"));
                }
                normalized
            }
        }
        PatchEnvelopeOp::Add => {
            if path.exists() {
                return Err(
                    "Add File patch targets an existing file. Use Update File or write_file instead."
                        .to_string(),
                );
            }
            // 空行代表新增文件中的空行，补上 + 前缀以便 parse_unified_hunks 识别为 Add 行。
            let normalized_body: Vec<String> = envelope
                .body_lines
                .iter()
                .map(|line| {
                    if line.is_empty() {
                        "+".to_string()
                    } else {
                        line.clone()
                    }
                })
                .collect();
            for line in &normalized_body {
                if !line.starts_with('+') {
                    return Err(format!(
                        "invalid Add File line: {}. Every content line in an Add File envelope must start with `+`.",
                        line
                    ));
                }
            }
            let mut normalized = format!("@@ -0,0 +1,{} @@", normalized_body.len());
            if !normalized_body.is_empty() {
                normalized.push('\n');
                normalized.push_str(&normalized_body.join("\n"));
            }
            normalized
        }
    };
    Ok((envelope.target_path, normalized_patch))
}

fn lines_match(actual: &str, expected: &str) -> bool {
    actual == expected || actual.trim_end() == expected.trim_end()
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
            // forward 已经过滤了 p >= cursor，所以这里不会有 "hunks out of order"。
            // 之前回退到 find_hunk_offset（±50 窗口）会在唯一匹配超出窗口时误报
            // context mismatch；直接使用 forward 的唯一结果即可。
            if let Some(&offset) = forward.first() {
                offset
            } else if !positions.is_empty() {
                // 所有匹配都在 cursor 之前——hunk 顺序错误
                return Err("hunks out of order".to_string());
            } else {
                return Err(describe_context_mismatch(&orig_lines, hunk));
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

fn emit_stream_line(on_chunk: &mut ToolStreamWriter<'_>, line: &str) {
    let mut rendered = line.to_string();
    rendered.push('\n');
    on_chunk(rendered.as_bytes());
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
    store
        .validate_write_access()
        .map_err(|err| err.to_string())?;
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

pub(crate) fn execute_apply_patch_streaming(
    args: &Value,
    on_chunk: &mut ToolStreamWriter<'_>,
) -> Result<String, String> {
    let patch = args["patch"].as_str().ok_or("missing patch")?;
    emit_stream_line(on_chunk, "parsing patch envelope");

    let initial_file_path = optional_file_path_arg(args);
    let envelope = parse_patch_envelope(patch)?;
    let file_path = initial_file_path
        .map(str::to_string)
        .or_else(|| envelope.as_ref().map(|parsed| parsed.target_path.clone()))
        .ok_or("missing file_path")?;

    let store = FileStore::new(PathBuf::from(&file_path));
    emit_stream_line(on_chunk, &format!("target: {}", store.path().display()));
    emit_stream_line(on_chunk, "validating write access");
    store
        .validate_write_access()
        .map_err(|err| err.to_string())?;

    let path = store.path().to_path_buf();
    let (_, normalized_patch) = if let Some(envelope) = envelope {
        ensure_patch_target_matches(&path, &envelope.target_path)?;
        normalize_patch_text(&path, patch)?
    } else {
        (file_path.clone(), patch.to_string())
    };

    let hunk_count = normalized_patch
        .lines()
        .filter(|line| line.starts_with("@@"))
        .count()
        .max(1);
    emit_stream_line(on_chunk, &format!("applying {hunk_count} hunk(s)"));

    let original = if path.exists() {
        emit_stream_line(on_chunk, "reading current file");
        store.read_to_string().map_err(|err| err.to_string())?
    } else {
        emit_stream_line(on_chunk, "creating new file from patch");
        String::new()
    };
    let next = apply_unified_patch(&original, &normalized_patch)?;

    emit_stream_line(on_chunk, &format!("writing {} byte(s)", next.len()));
    store.write_all(&next).map_err(|err| err.to_string())?;

    let result = format!("Successfully patched {}", path.display());
    emit_stream_line(on_chunk, &result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{apply_unified_patch, execute_apply_patch, parse_unified_hunks};
    use crate::ai::test_support::ENV_LOCK;
    use std::{fs, path::PathBuf};

    fn make_temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ai_patch_tools_test_{}_{}",
            name,
            uuid::Uuid::new_v4()
        ));
        path
    }

    #[test]
    fn parse_unified_hunks_treats_empty_hunk_line_as_context() {
        // 模型常把空 context 行写成完全没有前导空格的空行，应当作空 context 行处理，
        // 而不是报错。这与 `git apply` 对空 context 行的宽容一致。
        let patch = "@@ -1,3 +1,3 @@\n foo\n\n bar\n";
        let hunks =
            parse_unified_hunks(patch).expect("empty hunk line should be treated as context");
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].lines.len(), 3);
    }

    #[test]
    fn apply_unified_patch_tolerates_empty_context_line() {
        // 模型常把空 context 行写成空字符串（无前导空格），apply_patch 应正常匹配。
        let original = "foo\n\nbar\n";
        let patch = "@@ -1,3 +1,3 @@\n foo\n\n-bar\n+baz\n";
        let result = apply_unified_patch(original, patch)
            .expect("empty context line should be tolerated");
        assert_eq!(result, "foo\n\nbaz\n");
    }

    #[test]
    fn apply_unified_patch_strips_trailing_cr_from_crlf_patch() {
        // CRLF patch：Add 行尾的 \r 不应写入文件内容。
        let original = "foo\nbar\n";
        let patch = "@@ -2,1 +2,1 @@\r\n-bar\r\n+baz\r\n";
        let result = apply_unified_patch(original, patch)
            .expect("CRLF patch should be tolerated");
        assert_eq!(result, "foo\nbaz\n");
    }

    #[test]
    fn apply_unified_patch_tolerates_empty_context_line_in_crlf_patch() {
        // CRLF patch 中的空 context 行（只剩 \r 的行）也应被当作空 context 行。
        let original = "foo\r\n\r\nbar\r\n";
        let patch = "@@ -1,3 +1,3 @@\r\n foo\r\n\r\r\n-bar\r\n+baz\r\n";
        let result = apply_unified_patch(original, patch)
            .expect("empty CRLF context line should be tolerated");
        // 原文件是 CRLF，但 patch 的 Add 行已剥离 \r，输出统一为 LF。
        assert_eq!(result, "foo\n\nbaz\n");
    }

    #[test]
    fn parse_unified_hunks_strips_trailing_blank_context_between_hunks() {
        // 两个 hunk 之间用空行分隔（可读性写法）。之前空行会被吞进 hunk1 变成末尾
        // 的空 context 行，凭空要求原文件对应位置有空行 → context mismatch。
        // 修复后应剥离该尾随空行，hunk1 只保留 remove+add 两行。
        let patch = "@@ -1,1 +1,1 @@\n-a\n+b\n\n@@ -5,1 +5,1 @@\n-c\n+d\n";
        let hunks = parse_unified_hunks(patch).expect("blank separator should be tolerated");
        assert_eq!(hunks.len(), 2);
        assert_eq!(hunks[0].lines.len(), 2, "hunk1 should not swallow the blank separator");
    }

    #[test]
    fn apply_unified_patch_multi_hunk_separated_by_blank_line() {
        // 复现真实高频场景：多 hunk 之间空行分隔。修复前 hunk1 末尾多一条空 context
        // 行，导致整个 patch 报 context mismatch。
        let original = "a\nkeep1\nkeep2\nkeep3\nc\n";
        let patch = "@@ -1,1 +1,1 @@\n-a\n+b\n\n@@ -5,1 +5,1 @@\n-c\n+d\n";
        let result = apply_unified_patch(original, patch)
            .expect("multi-hunk patch separated by a blank line should apply");
        assert_eq!(result, "b\nkeep1\nkeep2\nkeep3\nd\n");
    }

    #[test]
    fn apply_unified_patch_tolerates_trailing_blank_line_in_patch() {
        // patch 末尾有多余空行（模型常见输出）。修复前末尾空行被并入最后一个 hunk
        // 变成空 context 行 → 匹配失败。
        let original = "line1\nline2\nline3\n";
        let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n\n";
        let result = apply_unified_patch(original, patch)
            .expect("trailing blank line in patch should be tolerated");
        assert_eq!(result, "line1\nchanged\nline3\n");
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
    fn apply_unified_patch_finds_unique_match_beyond_search_radius() {
        // 文件有 150 行，唯一匹配在第 130 行（0-based 129）。
        // hunk 头标称第 1 行，nominal=0，find_hunk_offset 的 ±50 窗口搜索 [0,50)，
        // 找不到 129 处的匹配。但 all_hunk_match_positions 能在全文件找到唯一匹配。
        // 此前代码忽略 forward.len()==1 的结果，回退到 find_hunk_offset 导致误报
        // "context mismatch"。
        let mut lines: Vec<String> = (0..130).map(|i| format!("filler{i}")).collect();
        lines.push("unique_target".to_string());
        lines.push("after_target".to_string());
        lines.extend((0..18).map(|i| format!("tail{i}")));
        let original = lines.join("\n") + "\n";

        let patch = "@@ -1,2 +1,2 @@\n-unique_target\n+changed\n+after_target\n";
        // 故意用错误的标称行号(-1) 模拟 stale line numbers
        let result = apply_unified_patch(&original, patch).unwrap_or_else(|err| {
            panic!("apply_patch should find unique match beyond ±50 radius, but got: {err}")
        });
        assert!(result.contains("changed"), "result should contain changed line: {result}");
        assert!(result.contains("after_target"), "result should preserve after_target: {result}");
        assert!(
            !result.contains("unique_target"),
            "result should not contain old line: {result}"
        );
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
    fn execute_apply_patch_update_envelope_without_hunk_header() {
        // *** Begin Patch 的 Update 格式省略 @@ header（Cursor/Aider 风格），
        // 只写 +/−/space 前缀行。模型经常这样写，不应报 "no hunks found"。
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("update_nohdr").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "path": path.to_string_lossy(),
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\n alpha\n-beta\n+changed\n*** End Patch\n",
                    path.display()
                )
            });
            execute_apply_patch(&args)
                .expect("apply_patch should accept Update envelope without @@ header");
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "alpha\nchanged\ngamma\n");
        let _ = fs::remove_dir_all(base);
    }
    #[test]
    fn apply_unified_patch_multi_hunk_with_stale_line_numbers() {
        // 两个 hunk，标称行号都是 1（过时），但各自的目标在文件中唯一且按顺序排列。
        // 验证 cursor 推进 + forward 过滤在多 hunk 场景下正常工作，不会把第二个
        // hunk 误匹配到第一个 hunk 的目标位置。
        let mut lines: Vec<String> = (0..60).map(|i| format!("filler{i}")).collect();
        lines.push("target_a".to_string());
        lines.push("after_a".to_string());
        lines.extend((0..60).map(|i| format!("mid{i}")));
        lines.push("target_b".to_string());
        lines.push("after_b".to_string());
        let original = lines.join("\n") + "\n";

        let patch = "\
@@ -1,2 +1,2 @@
-target_a
+changed_a
+after_a
@@ -1,2 +1,2 @@
-target_b
+changed_b
+after_b
";
        let result = apply_unified_patch(&original, patch).unwrap_or_else(|err| {
            panic!("multi-hunk patch should succeed with stale line numbers, but got: {err}")
        });
        assert!(result.contains("changed_a"), "missing changed_a: {result}");
        assert!(result.contains("changed_b"), "missing changed_b: {result}");
        assert!(result.contains("after_a"), "missing after_a: {result}");
        assert!(result.contains("after_b"), "missing after_b: {result}");
        assert!(!result.contains("target_a"), "should not contain target_a: {result}");
        assert!(!result.contains("target_b"), "should not contain target_b: {result}");
        // 中间填充行应保持不变
        assert!(result.contains("filler0"), "filler0 should remain: {result}");
        assert!(result.contains("mid0"), "mid0 should remain: {result}");
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
            execute_apply_patch(&args)
                .expect("apply_patch should infer target from Add File envelope");
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "hello\nworld");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_add_file_tolerates_empty_lines() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("add_empty");
        let path = base.join("new.txt");
        fs::create_dir_all(&base).unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "patch": "*** Begin Patch\n*** Add File: new.txt\n+hello\n\n+world\n*** End Patch\n"
            });
            execute_apply_patch(&args)
                .expect("apply_patch should tolerate empty lines in Add File envelope");
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "hello\n\nworld");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_streaming_dispatch_emits_progress() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("streaming").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "alpha\nbeta\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base, || {
            let args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "patch": "@@ -2,1 +2,1 @@\n-beta\n+changed\n"
            });
            let mut streamed = Vec::new();
            let mut capture = |chunk: &[u8]| streamed.extend_from_slice(chunk);
            let result = crate::ai::tools::common::execute_tool_call_with_args_streaming(
                "call_apply_patch_streaming",
                "apply_patch",
                &args,
                &mut capture,
            )
            .expect("streaming apply_patch should succeed");

            let streamed = String::from_utf8(streamed).expect("streamed output must be utf-8");
            assert!(
                streamed.contains("parsing patch envelope"),
                "streamed: {streamed}"
            );
            assert!(streamed.contains("target:"), "streamed: {streamed}");
            assert!(
                streamed.contains("applying 1 hunk(s)"),
                "streamed: {streamed}"
            );
            assert!(streamed.contains("writing "), "streamed: {streamed}");
            assert!(
                streamed.contains(&format!("Successfully patched {}", path.display())),
                "streamed: {streamed}"
            );
            assert_eq!(
                result.content,
                format!("Successfully patched {}", path.display())
            );
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "alpha\nchanged\n");
        let _ = fs::remove_file(&path);
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
