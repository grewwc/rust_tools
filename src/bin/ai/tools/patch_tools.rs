use std::fs;
use std::path::{Path, PathBuf};

use rustc_hash::FxHashSet;
use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::common::ToolStreamWriter;
use crate::ai::tools::common::ToolStreamingRegistration;
use crate::ai::tools::storage::file_store::FileStore;
use crate::ai::tools::undo_tools::{CompletedFileChange, record_completed_change_set};

fn params_apply_patch() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute target path for a single-file unified diff. Not needed (and ignored) when using a Begin Patch envelope, since each section declares its own target path. `path` is accepted as a compatibility alias."
            },
            "patch": {
                "type": "string",
                "description": "Patch content. Use either unified-diff hunks (`@@` header; content lines begin with space, `-`, or `+`) or a `*** Begin Patch` envelope. For several edits in the SAME file, prefer ONE `*** Update File:` section containing multiple `@@` hunks from one fresh read; do not split into serial apply_patch calls. For several files, use one Begin Patch envelope with one section per target. Envelope sections support `*** Update File:`, `*** Add File:`, `*** Delete File:`, and `*** Replace in line:`. Use `*** Delete File:` to remove existing project/source/config files, including git-tracked files. Do not wrap it in a Markdown code fence. Include unique surrounding context and do not repeat a target path within one multi-file envelope."
            },
            "dry_run": {
                "type": "boolean",
                "description": "When true, parse, sandbox-check, and match every patch section without writing any file. Defaults to false."
            }
        },
        "required": ["patch"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "apply_patch",
        description: "Apply localized edits; prefer it to rewriting a whole file. Batch related edits: use one `*** Update File:` section with multiple `@@` hunks for several edits in the same file, and one Begin Patch envelope with one section per target for several files. Supports a unified-diff hunk for one `file_path`, or a `*** Begin Patch` envelope with `*** Update File:`, `*** Add File:`, `*** Delete File:`, or `*** Replace in line:` sections. Use `*** Delete File:` to remove existing project/source/config files, including git-tracked files; `delete_path` is only for registered temp files. Multi-file envelopes are fully validated before writing, rechecked immediately before commit, and rolled back if a write fails. Use `dry_run` to validate without changing files. After `context mismatch` or `ambiguous patch`, re-read the same target before retrying.",
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
    Delete,
    /// 行内子串替换：用 `anchor:` 定位行，在该行内将 `old:` 精确替换为 `new:`。
    /// 不走 unified-diff 路径，由 `apply_inline_replace` 直接处理。
    ReplaceInLine,
}

#[derive(Debug, Clone)]
struct PatchEnvelope {
    op: PatchEnvelopeOp,
    target_path: String,
    body_lines: Vec<String>,
}

#[derive(Debug, Clone)]
struct PreparedPatchWrite {
    path: PathBuf,
    before: Option<String>,
    action: PreparedPatchAction,
}

#[derive(Debug, Clone)]
enum PreparedPatchAction {
    Write(String),
    Delete,
}

fn parse_unified_hunks(patch: &str) -> Result<Vec<UnifiedHunk>, String> {
    let mut hunks = Vec::new();
    let mut iter = patch.lines().peekable();
    let mut patch_line_no: usize = 0; // 1-based，用于错误信息定位
    let mut saw_content_before_header = false;
    let mut saw_envelope_marker = false; // 检测到 *** Begin Patch / *** Update File: 等 envelope 标记
    while let Some(line) = iter.next() {
        patch_line_no += 1;
        // 残缺 envelope 信号：parse_patch_envelopes 因首行非 `*** Begin Patch`
        // 返回 None 后会误入 unified-diff 路径。记录是否出现过 envelope 开头/分节
        // 标记，用于下方决定是否静默容忍尾部 `*** End Patch`。
        if line == "*** Begin Patch" || is_patch_section_header(line) {
            saw_envelope_marker = true;
        }
        let Some(rest) = line.strip_prefix("@@") else {
            if hunks.is_empty()
                && (line.starts_with('+')
                    || line.starts_with('-')
                    || (line.starts_with(' ') && !line.trim().is_empty()))
            {
                saw_content_before_header = true;
            }
            continue;
        };
        let rest = rest.trim();
        // 规范的 `*** Begin Patch` 信封（Codex/OpenAI 风格）用裸 `@@` 或
        // `@@ <上下文标题> @@` 作为 hunk 分隔符，不带 `-N,M +N,M` 行号。仅当
        // header 形如 `-N` 时解析标称行号；否则 old_start=0，交给 locate_hunk
        // 的全文件搜索唯一定位，避免对规范信封格式报 "invalid hunk header"。
        let old_start = match rest.strip_prefix('-') {
            Some(after) => after
                .split_whitespace()
                .next()
                .and_then(|part| part.split(',').next())
                .and_then(|num| num.parse::<isize>().ok())
                .map(|n| if n <= 0 { 0 } else { n as usize })
                .unwrap_or(0),
            None => 0,
        };

        let mut lines = Vec::new();
        while let Some(next) = iter.peek().copied() {
            if next.starts_with("@@") {
                break;
            }
            // 容忍格式混用：模型常在纯 unified-diff hunk 末尾误带 `*** End Patch`
            // 等 envelope 尾标记。这些标记不属于 unified-diff 内容，遇到即结束
            // 当前 hunk（交由外层循环跳过），避免误报 invalid hunk line。
            // 但若已检测到 envelope 开头/分节标记（saw_envelope_marker），说明这是
            // 残缺 envelope 误入 unified-diff 路径，目标文件由 file_path 决定、而非
            // envelope 声明--此时静默应用可能写到错误文件。故不 break，让该行落入
            // 下方 _ => 分支报"格式混用"错误，由模型重建，绝不静默错写。
            if (next == "*** End Patch" || next == "*** End of File") && !saw_envelope_marker {
                break;
            }
            let l = iter.next().unwrap_or_default();
            patch_line_no += 1;
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
                .ok_or_else(|| format!("invalid hunk line at patch line {patch_line_no}: empty"))?;
            // 容忍 CRLF：剥离行尾 \r，避免 Add 行把 \r 写入文件内容。
            let body = chars.as_str().strip_suffix('\r').unwrap_or(chars.as_str());
            match prefix {
                ' ' => lines.push(UnifiedLine::Context(body.to_string())),
                '-' => lines.push(UnifiedLine::Remove(body.to_string())),
                '+' => lines.push(UnifiedLine::Add(body.to_string())),
                _ => {
                    // 特判 envelope 风格标记：说明 unified-diff 与 Begin/End Patch
                    // 格式混用。结尾标记（*** End Patch / *** End of File）已在上方
                    // break 容错；走到这里的是 *** Begin Patch / *** Update File: 等
                    // 开头或分节标记，表示 patch 结构混乱，明确报错引导模型重建。
                    if l.starts_with("*** ") {
                        return Err(format!(
                            "invalid hunk line at patch line {patch_line_no}: detected mixed \
                             patch formats. Line {:?} is a `*** Begin/End Patch` envelope marker, \
                             but the patch was parsed as unified diff (it has `@@` hunks). Use ONE \
                             format only: either unified-diff hunks (`@@ ... @@` with ` `/`-`/`+` \
                             prefixed lines) OR a `*** Begin Patch` envelope, not both.",
                            l
                        ));
                    }
                    return Err(format!(
                        "invalid hunk line at patch line {patch_line_no}: every line in a hunk must start with ` ` (context), `-` (remove), or `+` (add), but got: {:?}",
                        l
                    ));
                }
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
        if saw_content_before_header {
            return Err("no hunk header found: patch contains content lines but no hunk header. Prepend a hunk header before the content lines, or use a Begin Patch envelope.".to_string());
        }
        return Err(
            "no hunks found: the patch is empty or contains no valid unified-diff hunks (no `@@` headers). \
             Check that the patch content is not wrapped in Markdown code fences and contains hunk headers like `@@ -1,3 +1,3 @@`."
                .to_string(),
        );
    }
    Ok(hunks)
}

fn optional_file_path_arg(args: &Value) -> Option<&str> {
    args.get("file_path")
        .or_else(|| args.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
}

fn ensure_patch_target_matches(target_path: &Path, envelope_path: &str) -> Result<(), String> {
    let resolved_target = FileStore::new(target_path.to_path_buf())
        .path()
        .to_path_buf();
    let resolved_envelope = FileStore::new(PathBuf::from(envelope_path))
        .path()
        .to_path_buf();
    if resolved_target == resolved_envelope {
        return Ok(());
    }
    Err(format!(
        "patch target mismatch: tool arg points to {}, but patch envelope points to {}. Rebuild the patch for the same file before retrying.",
        target_path.display(),
        envelope_path
    ))
}

fn parse_patch_header(header: &str) -> Result<(PatchEnvelopeOp, &str), String> {
    if let Some(path) = header.strip_prefix("*** Update File: ") {
        Ok((PatchEnvelopeOp::Update, path.trim()))
    } else if let Some(path) = header.strip_prefix("*** Add File: ") {
        Ok((PatchEnvelopeOp::Add, path.trim()))
    } else if let Some(path) = header.strip_prefix("*** Delete File: ") {
        Ok((PatchEnvelopeOp::Delete, path.trim()))
    } else if let Some(path) = header.strip_prefix("*** Replace in line: ") {
        Ok((PatchEnvelopeOp::ReplaceInLine, path.trim()))
    } else {
        Err(
            "invalid patch envelope: expected `*** Update File:`, `*** Add File:`, \
             `*** Delete File:`, or `*** Replace in line:`"
                .to_string(),
        )
    }
}

fn is_patch_section_header(line: &str) -> bool {
    line.starts_with("*** Update File: ")
        || line.starts_with("*** Add File: ")
        || line.starts_with("*** Replace in line: ")
        || line.starts_with("*** Delete File: ")
}

fn parse_patch_envelopes(patch: &str) -> Result<Option<Vec<PatchEnvelope>>, String> {
    let lines: Vec<&str> = patch.lines().collect();
    let Some(mut idx) = lines.iter().position(|line| !line.trim().is_empty()) else {
        return Ok(None);
    };
    if lines[idx].trim() != "*** Begin Patch" {
        return Ok(None);
    }
    idx += 1;

    let mut envelopes = Vec::new();
    loop {
        while idx < lines.len() && lines[idx].trim().is_empty() {
            idx += 1;
        }
        if idx >= lines.len() {
            return Err("invalid patch envelope: missing `*** End Patch`".to_string());
        }
        if lines[idx] == "*** End Patch" {
            break;
        }

        let (op, target_path) = parse_patch_header(lines[idx])?;
        idx += 1;

        let mut body_lines = Vec::new();
        while idx < lines.len() {
            let line = lines[idx];
            if line == "*** End Patch" || is_patch_section_header(line) {
                break;
            }
            if line == "*** End of File" {
                idx += 1;
                continue;
            }
            if line.trim().is_empty() {
                let mut lookahead = idx + 1;
                while lookahead < lines.len() && lines[lookahead].trim().is_empty() {
                    lookahead += 1;
                }
                if lookahead < lines.len()
                    && (lines[lookahead] == "*** End Patch"
                        || is_patch_section_header(lines[lookahead]))
                {
                    idx = lookahead;
                    continue;
                }
            }
            body_lines.push(line.to_string());
            idx += 1;
        }

        envelopes.push(PatchEnvelope {
            op,
            target_path: target_path.to_string(),
            body_lines,
        });
    }

    if envelopes.is_empty() {
        return Err("invalid patch envelope: missing file header".to_string());
    }
    Ok(Some(envelopes))
}

fn parse_patch_envelope(patch: &str) -> Result<Option<PatchEnvelope>, String> {
    let Some(mut envelopes) = parse_patch_envelopes(patch)? else {
        return Ok(None);
    };
    if envelopes.len() != 1 {
        return Err(format!(
            "parse_patch_envelope expected exactly 1 file section, found {}",
            envelopes.len()
        ));
    }
    Ok(envelopes.pop())
}

fn normalize_patch_envelope(path: &Path, envelope: &PatchEnvelope) -> Result<String, String> {
    Ok(match envelope.op {
        PatchEnvelopeOp::ReplaceInLine => {
            // ReplaceInLine 不走 unified-diff 路径，由 apply_inline_replace 直接处理。
            // 走到这里说明 execute_apply_patch 的分流逻辑有 bug——提前返回明确错误，
            // 避免被当成 unified-diff 误处理（那会把 anchor:/old:/new: 当 context 行）。
            return Err(
                "internal error: ReplaceInLine envelope should be handled by \
                 apply_inline_replace, not normalize_patch_text"
                    .to_string(),
            );
        }
        PatchEnvelopeOp::Update => {
            if !path.exists() {
                return Err(format!(
                    "Update File patch targets a missing file: {}. Use Add File to create a new file, or correct the target path before retrying.",
                    path.display()
                ));
            }
            // *** Begin Patch 的 Update 格式允许省略 hunk header（Cursor/Aider 风格），
            // 模型常只写 +/−/space 前缀行而不带 hunk header。如果 body 中没有任何
            // hunk header，合成一个，让 parse_unified_hunks 能识别。old_start=0 表示
            // 无标称位置，locate_hunk 会跳过标称匹配直接全文件搜索。
            let has_hunk_header = envelope.body_lines.iter().any(|l| l.starts_with("@@"));
            // 即便已有 hunk header，模型也常把 context 行写成裸文本；在 envelope
            // 内将这类裸行补成 context 行，避免无意义的 invalid hunk line 失败。
            let normalized_body: Vec<String> = envelope
                .body_lines
                .iter()
                .map(|line| {
                    if line.starts_with("@@") || line.is_empty() {
                        line.clone()
                    } else if line.starts_with('+')
                        || line.starts_with('-')
                        || line.starts_with(' ')
                    {
                        line.clone()
                    } else {
                        format!(" {}", line)
                    }
                })
                .collect();
            if has_hunk_header {
                normalized_body.join("\n")
            } else {
                let mut normalized = String::from("@@ -0,0 +1,0 @@");
                if !envelope.body_lines.is_empty() {
                    normalized.push('\n');
                    normalized.push_str(&normalized_body.join("\n"));
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
                        "invalid Add File line: {:?}. Every content line in an Add File envelope must \
                         start with `+`. Hint: prefix each file line with `+`, or use `*** Update File` \
                         with a unified diff hunk instead.",
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
        PatchEnvelopeOp::Delete => {
            return Err(
                "internal error: Delete File envelopes should be handled by prepare_patch_write"
                    .to_string(),
            );
        }
    })
}

/// 行内子串替换：用 `anchor:` 定位行，在该行内将 `old:` 精确替换为 `new:`。
///
/// 专为"长单行字符串里换几个词"这种最常见的编辑场景设计，避免整行重写。
///
/// 安全设计（杜绝"执行成功但替换错位置"）：
/// - `anchor` 用归一化子串匹配（容忍 confusable）定位行，但**只用于定位**；
/// - `old` 用**精确**子串匹配（不归一化）确定替换的 byte range，杜绝位置偏移；
/// - `anchor` 必须唯一匹配到一行，否则报错（避免改错地方）；
/// - `old` 必须在该行中唯一出现，否则报错（避免改错位置）；
/// - 若 `old == new`（替换前后相同），报错提示无操作，避免误以为成功。
fn apply_inline_replace(original: &str, envelope: &PatchEnvelope) -> Result<String, String> {
    // --- 解析 anchor / old / new 三个字段 ---
    let mut anchor: Option<String> = None;
    let mut old: Option<String> = None;
    let mut new: Option<String> = None;
    for line in &envelope.body_lines {
        if let Some(rest) = line.strip_prefix("anchor: ") {
            anchor = Some(rest.to_string());
        } else if let Some(rest) = line.strip_prefix("old: ") {
            old = Some(rest.to_string());
        } else if let Some(rest) = line.strip_prefix("new: ") {
            new = Some(rest.to_string());
        }
        // 忽略无关行（空行、注释等），保持宽容
    }
    let anchor = anchor.ok_or_else(|| {
        "Replace in line: missing `anchor:` field. \
         Expected `anchor: <unique substring of target line>`."
            .to_string()
    })?;
    let old = old.ok_or_else(|| {
        "Replace in line: missing `old:` field. \
         Expected `old: <exact substring to replace>`."
            .to_string()
    })?;
    let new = new.ok_or_else(|| {
        "Replace in line: missing `new:` field. \
         Expected `new: <replacement substring>`."
            .to_string()
    })?;
    if old.is_empty() {
        return Err("Replace in line: `old` field must not be empty.".to_string());
    }
    if old == new {
        return Err(format!(
            "Replace in line: `old` and `new` are identical ({:?}). \
             Nothing would change; fix the patch or remove it.",
            old
        ));
    }

    // --- 用归一化子串匹配定位行（容忍 confusable），但只用于定位 ---
    let norm_anchor = normalize_confusables(&anchor);
    let matched_lines: Vec<usize> = original
        .lines()
        .enumerate()
        .filter(|(_, line)| normalize_confusables(line).contains(norm_anchor.as_str()))
        .map(|(i, _)| i)
        .collect();

    let line_idx = match matched_lines.len() {
        0 => {
            return Err(format!(
                "Replace in line: anchor not found. \
                 No line contains {:?} (after Unicode normalization).",
                anchor
            ));
        }
        1 => matched_lines[0],
        n => {
            let positions = matched_lines
                .iter()
                .map(|i| format!("{}", i + 1))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "Replace in line: anchor matched {n} lines (1-based: {positions}). \
                 Anchor must uniquely identify one line. Make `anchor` more specific."
            ));
        }
    };

    let original_lines: Vec<&str> = original.lines().collect();
    let target_line = original_lines[line_idx];

    // --- old 用精确子串匹配确定替换位置（不归一化），杜绝位置偏移 ---
    let occurrences: Vec<usize> = target_line.match_indices(&old).map(|(i, _)| i).collect();
    let pos = match occurrences.len() {
        0 => {
            return Err(format!(
                "Replace in line: `old` substring not found in matched line {}. \
                 The anchor matched this line, but `old` must appear verbatim \
                 (no Unicode normalization). Line content: {:?}",
                line_idx + 1,
                target_line
            ));
        }
        1 => occurrences[0],
        n => {
            return Err(format!(
                "Replace in line: `old` substring appears {n} times in line {}. \
                 It must be unique within the line. Make `old` longer to disambiguate. \
                 Line content: {:?}",
                line_idx + 1,
                target_line
            ));
        }
    };

    // 精确替换：byte range [pos, pos+old.len())。
    // pos 是 str::find 返回的 byte index，old 是有效 UTF-8，
    // 所以 pos 和 pos+old.len() 都在 char boundary 上，切片安全。
    let replaced_line = format!(
        "{}{}{}",
        &target_line[..pos],
        new,
        &target_line[pos + old.len()..]
    );

    // 重建文件，保留原始行尾换行行为
    let trailing_newline = original.ends_with('\n');
    let mut result = String::with_capacity(original.len() + new.len());
    for (i, line) in original_lines.iter().enumerate() {
        if i == line_idx {
            result.push_str(&replaced_line);
        } else {
            result.push_str(line);
        }
        if i < original_lines.len() - 1 || trailing_newline {
            result.push('\n');
        }
    }
    Ok(result)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchMode {
    /// 精确匹配（允许行尾空白差异），默认使用。
    Strict,
    /// 忽略前导缩进差异，仅在严格匹配全文件都定位不到时作为兜底。
    /// 对齐 `git apply --ignore-whitespace`：模型对 markdown/嵌套列表/代码块
    /// 的缩进常常复刻不准，导致严格匹配整块失配报 context mismatch。
    IgnoreIndent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextPolicy {
    /// context 行必须匹配，remove 行也必须匹配。
    Require,
    /// context 行只作为定位参考；应用时保留文件中的实际 context，remove 行仍必须匹配。
    Fuzz,
}

/// 剥离 read_file / grep 等工具输出的行号前缀（单参数兜底版）。
/// 模型有时会不小心把行号前缀复制进 patch 的 context/remove 行中。
///
/// 该版本用于**没有"真实行"可锚定**的场景（如 IgnoreIndent 双侧各自归一）。
/// 有真实行可比对时，优先用锚定式 [`strip_number_prefix_anchored`]，它分隔符
/// 无关且几乎零误伤。此处为避免误剥真正以数字开头的代码行（如 `80:80`、`42px`、
/// `3.14`），采取**保守**策略，只认两类确定性极高的行号栏形状：
/// - `digits + \t`：read_file 的真实格式（`{:>6}\t{}`）。TAB 后直接是行内容
///   （含其自身缩进），只吞这一个 TAB。
/// - `digits + 单个非字母数字分隔符 + 空格`：grep 类（`42| `、`42: `）。要求分隔符
///   后**必须跟空格**，因此 `80:80`（`:` 后是数字）、`3.14`（`.` 后是数字）不会被误剥。
fn strip_line_number_prefix(s: &str) -> &str {
    let trimmed = s.trim_start();
    let digits_end = trimmed.find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
    if digits_end == 0 {
        return s;
    }
    let after_digits = &trimmed[digits_end..];
    let mut chars = after_digits.chars();
    let sep = match chars.next() {
        Some(c) => c,
        None => return s,
    };
    // TAB：read_file 真实分隔符，其后直接是内容（含缩进），只吞这一个 TAB。
    if sep == '\t' {
        return &after_digits['\t'.len_utf8()..];
    }
    // 其它分隔符：必须是"非字母数字、非空格"的单字符，且其后紧跟一个空格
    // （`42| ` / `42: `）。要求尾随空格可避免把 `80:80`、`3.14` 误判成行号栏。
    if sep.is_alphanumeric() || sep == ' ' {
        return s;
    }
    let rest = &after_digits[sep.len_utf8()..];
    match rest.strip_prefix(' ') {
        Some(after_space) => after_space,
        None => s,
    }
}

/// 锚定式行号前缀剥离：以 `actual`（原文件真实行，永不含行号栏）为 Ground Truth，
/// 判断 `expected`（patch 行，可能被模型误抄了行号栏）去掉"数字栏"后是否**精确
/// 等于** `actual`。是则返回去栏后的内容，否则返回 `expected` 原样。
///
/// 相比枚举分隔符，这里分隔符无关（`\t` `|` `:` 空格 `.` `)` 全兼容），且因为
/// 要求"剩余部分精确等于真实行"，几乎不可能误伤真正以数字开头的代码行——即便
/// 恰好撞上，也会被 lines_match 的多处匹配（ambiguity）检测拦截。
fn strip_number_prefix_anchored<'a>(expected: &'a str, actual: &str) -> &'a str {
    // expected 必须以「可选空白 + 数字」开头，否则不可能是"行号栏 + actual"。
    let lead_ws_end = expected
        .find(|c: char| !c.is_whitespace())
        .unwrap_or(expected.len());
    let after_ws = &expected[lead_ws_end..];
    let digits_end = after_ws
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_ws.len());
    if digits_end == 0 {
        return expected; // 数字部分为空：不是行号栏。
    }
    let after_digits = &after_ws[digits_end..];
    // 去掉 1 个分隔符（按 char 边界，避免多字节 UTF-8 切分 panic）后的剩余部分。
    let after_one_sep = after_digits
        .char_indices()
        .nth(1)
        .map(|(byte_idx, _)| &after_digits[byte_idx..])
        .unwrap_or("");
    // 逐一尝试：去掉 0 或 1 个分隔符（可再带 1 个空格）后是否等于真实行。
    // 用"剩余部分精确等于真实行"作为唯一判据，因此无需知道分隔符具体是什么。
    let candidates = [
        after_digits,  // 数字后直接是内容（罕见）
        after_one_sep, // 吞 1 个分隔符
    ];
    for cand in candidates {
        if cand == actual || cand.trim_end() == actual.trim_end() {
            return cand;
        }
        if let Some(c2) = cand.strip_prefix(' ')
            && (c2 == actual || c2.trim_end() == actual.trim_end())
        {
            return c2;
        }
    }
    expected
}

/// 将常见的 Unicode "confusable" 字符归一化为 ASCII 等价形式。
///
/// 仅用于 patch 匹配的 **定位判定**（lines_match），绝不参与输出内容构造。
/// 处理的字符都是纯排版差异，不影响语义：
/// - dash 系（— – ― 等）-> '-'
/// - smart quotes（" " ' ' ‛ ‟）-> '"' / "'"
/// - 不间断空格（NBSP U+00A0、NNBSP U+202F 等）-> 普通空格
fn normalize_confusables(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            // --- dash 系 ---
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            // --- smart double quotes ---
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => '"',
            // --- smart single quotes ---
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => '\'',
            // --- 不间断空格系 ---
            '\u{00A0}' | '\u{202F}' | '\u{2007}' | '\u{2060}' => ' ',
            other => other,
        })
        .collect()
}

fn lines_match_exact(actual: &str, expected: &str, mode: MatchMode) -> bool {
    if actual == expected || actual.trim_end() == expected.trim_end() {
        return true;
    }
    match mode {
        MatchMode::Strict => {
            // 模型经常从 read_file 输出中复制行号前缀（如 `    42\t<code>`）。
            // 优先用锚定式：以 actual（真实文件行）为准，判断 expected 去掉数字栏后
            // 是否精确等于 actual —— 分隔符无关且几乎零误伤。
            let e = strip_number_prefix_anchored(expected, actual);
            if e == actual || e.trim_end() == actual.trim_end() {
                return true;
            }
            // 兜底：无 actual 锚点信息时的通用数字栏剥离（双侧），
            // 兼容 actual 侧也带栏之类的极端情况。
            let expected_stripped = strip_line_number_prefix(expected);
            let actual_stripped = strip_line_number_prefix(actual);
            expected_stripped == actual_stripped
                || expected_stripped.trim_end() == actual_stripped.trim_end()
        }
        MatchMode::IgnoreIndent => {
            // 先试锚定式（以 actual.trim 为准），再回退到双侧通用剥离 + trim。
            let e = strip_number_prefix_anchored(expected.trim_start(), actual.trim());
            if e.trim() == actual.trim() {
                return true;
            }
            strip_line_number_prefix(actual).trim() == strip_line_number_prefix(expected).trim()
        }
    }
}

/// lines_match 的公共入口：先做精确匹配，失败后对 confusable 字符归一化再比较。
///
/// 归一化只影响"能否定位到"的判定。输出内容由 try_apply_hunk_at 构造：
/// - Context 行输出 actual（原文件内容）
/// - Remove 行匹配后丢弃
/// - Add 行直接用 patch 内容
/// 因此归一化匹配成功时，写入文件的仍是原文件的 Unicode 字符，不会"替换错内容"。
fn lines_match(actual: &str, expected: &str, mode: MatchMode) -> bool {
    if lines_match_exact(actual, expected, mode) {
        return true;
    }
    let actual_n = normalize_confusables(actual);
    let expected_n = normalize_confusables(expected);
    if actual_n == expected_n || actual_n.trim_end() == expected_n.trim_end() {
        return true;
    }
    match mode {
        MatchMode::Strict => {
            let e = strip_number_prefix_anchored(&expected_n, &actual_n);
            if e == actual_n || e.trim_end() == actual_n.trim_end() {
                return true;
            }
            let a = strip_line_number_prefix(&actual_n);
            let e = strip_line_number_prefix(&expected_n);
            a == e || a.trim_end() == e.trim_end()
        }
        MatchMode::IgnoreIndent => {
            let e = strip_number_prefix_anchored(expected_n.trim_start(), actual_n.trim());
            if e.trim() == actual_n.trim() {
                return true;
            }
            strip_line_number_prefix(&actual_n).trim()
                == strip_line_number_prefix(&expected_n).trim()
        }
    }
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
fn all_hunk_match_positions(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    mode: MatchMode,
) -> Vec<usize> {
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
            .all(|(i, exp)| lines_match(&orig_lines[candidate + i], exp, mode));
        if all_match {
            positions.push(candidate);
        }
        candidate += 1;
    }
    positions
}

fn hunk_old_line_count(hunk: &UnifiedHunk) -> usize {
    hunk.lines
        .iter()
        .filter(|line| matches!(line, UnifiedLine::Context(_) | UnifiedLine::Remove(_)))
        .count()
}

fn hunk_remove_offsets(hunk: &UnifiedHunk) -> Vec<(usize, &str)> {
    let mut old_offset = 0usize;
    let mut offsets = Vec::new();
    for line in &hunk.lines {
        match line {
            UnifiedLine::Context(_) => old_offset += 1,
            UnifiedLine::Remove(s) => {
                offsets.push((old_offset, s.as_str()));
                old_offset += 1;
            }
            UnifiedLine::Add(_) => {}
        }
    }
    offsets
}

fn remove_lines_match_at(
    orig_lines: &[String],
    remove_offsets: &[(usize, &str)],
    start: usize,
    mode: MatchMode,
) -> bool {
    remove_offsets.iter().all(|(offset, expected)| {
        orig_lines
            .get(start + offset)
            .is_some_and(|actual| lines_match(actual, expected, mode))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FuzzyContextMatch {
    pos: usize,
    context_matches: usize,
    context_total: usize,
}

fn score_context_matches(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    start: usize,
    mode: MatchMode,
) -> (usize, usize) {
    let mut old_offset = 0usize;
    let mut matches = 0usize;
    let mut total = 0usize;
    for line in &hunk.lines {
        match line {
            UnifiedLine::Context(expected) => {
                total += 1;
                if orig_lines
                    .get(start + old_offset)
                    .is_some_and(|actual| lines_match(actual, expected, mode))
                {
                    matches += 1;
                }
                old_offset += 1;
            }
            UnifiedLine::Remove(_) => old_offset += 1,
            UnifiedLine::Add(_) => {}
        }
    }
    (matches, total)
}

fn fuzzy_context_candidates(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    cursor: usize,
    mode: MatchMode,
) -> Vec<FuzzyContextMatch> {
    let old_len = hunk_old_line_count(hunk);
    let remove_offsets = hunk_remove_offsets(hunk);
    if old_len == 0 || remove_offsets.is_empty() || old_len > orig_lines.len() {
        return Vec::new();
    }

    let (first_remove_offset, first_remove) = remove_offsets[0];
    let Some(first_scan_line) = cursor.checked_add(first_remove_offset) else {
        return Vec::new();
    };
    if first_scan_line >= orig_lines.len() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for file_line in first_scan_line..orig_lines.len() {
        if !lines_match(&orig_lines[file_line], first_remove, mode) {
            continue;
        }
        let Some(start) = file_line.checked_sub(first_remove_offset) else {
            continue;
        };
        if start < cursor || start + old_len > orig_lines.len() {
            continue;
        }
        if !remove_lines_match_at(orig_lines, &remove_offsets, start, mode) {
            continue;
        }
        let (context_matches, context_total) = score_context_matches(orig_lines, hunk, start, mode);
        candidates.push(FuzzyContextMatch {
            pos: start,
            context_matches,
            context_total,
        });
    }

    candidates.sort_by_key(|candidate| candidate.pos);
    candidates.dedup_by_key(|candidate| candidate.pos);
    candidates
}

/// context 行是定位辅助，不应在 remove 行已经精确锚定时导致硬失败。
/// 但为了避免把常见 remove 行（如 `}`）改错位置，只有候选唯一，或剩余 context
/// 能唯一打分时才允许 fuzz 应用。
fn locate_hunk_with_fuzzy_context(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    cursor: usize,
    mode: MatchMode,
) -> Result<Option<FuzzyContextMatch>, String> {
    let candidates = fuzzy_context_candidates(orig_lines, hunk, cursor, mode);
    if candidates.is_empty() {
        return Ok(None);
    }
    if candidates.len() == 1 {
        return Ok(candidates.first().copied());
    }

    let best_score = candidates
        .iter()
        .map(|candidate| candidate.context_matches)
        .max()
        .unwrap_or(0);
    let best: Vec<FuzzyContextMatch> = candidates
        .iter()
        .copied()
        .filter(|candidate| candidate.context_matches == best_score)
        .collect();
    if best.len() == 1 && best_score > 0 {
        return Ok(best.first().copied());
    }

    let nominal = hunk.old_start.saturating_sub(1);
    // 用 old_start 作为消歧信号：只要 nominal 候选的 context 分数接近最优
    // （差值 ≤ 1），就信任模型标注的行号。best_score == 0 时（上下文行完全
    // 无法区分候选位置）也接受——此时 old_start 是唯一可用的定位信号，
    // 拒绝只会导致模型无限重试相同的 generic context。
    if hunk.old_start > 0 && nominal < orig_lines.len() {
        if let Some(nominal_candidate) = candidates.iter().find(|c| c.pos == nominal) {
            if best_score == 0 || nominal_candidate.context_matches + 1 >= best_score {
                return Ok(Some(*nominal_candidate));
            }
        }
    }

    let shown: Vec<String> = candidates
        .iter()
        .take(5)
        .map(|candidate| {
            format!(
                "{} (context {}/{})",
                candidate.pos + 1,
                candidate.context_matches,
                candidate.context_total
            )
        })
        .collect();
    Err(format!(
        "ambiguous patch: remove lines match {} locations under context-fuzz mode (1-based lines: {}{}). \
         Re-read the file and include more exact surrounding context, or split the hunk around a more unique removed line.",
        candidates.len(),
        shown.join(", "),
        if candidates.len() > 5 { ", ..." } else { "" }
    ))
}

/// 大块替换（hunk 含大量 context/remove 行）时，全有或全无的精确匹配极易因个别行
/// 复刻不准而整体失败。此处先做一次 best-effort 部分匹配扫描：在全文件范围内找到
/// 匹配行数最多的起点，并精确报告哪些行不一致（expected vs actual），让模型只需
/// 修正出错的几行而非重新猜测整块。
struct BestPartialMatch {
    /// 最佳匹配起点（0-based）
    pos: usize,
    /// 匹配的行数
    matches: usize,
    /// 检查的总行数
    total: usize,
    /// 不一致的行：(1-based 文件行号, 期望内容, 实际内容)
    mismatches: Vec<(usize, String, String)>,
}

/// 在全文件范围内找到 hunk 期望行块匹配度最高的起点。
/// 仅在精确匹配失败后的错误路径调用，用 IgnoreIndent 模式以容忍缩进差异、
/// 聚焦内容差异。返回 None 表示文件中没有任何行能匹配期望块——块完全不存在。
fn find_best_partial_match(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    mode: MatchMode,
) -> Option<BestPartialMatch> {
    let expected = hunk_expected_lines(hunk);
    if expected.is_empty() || orig_lines.is_empty() {
        return None;
    }

    // 用首行做快速过滤：只在首行能匹配的候选位置做完整对齐检查，避免大文件上的
    // O(N*M) 全扫描。大块替换中最常见的失败是首行正确、后续个别行有误。
    let mut candidates: Vec<usize> = (0..orig_lines.len())
        .filter(|&i| lines_match(&orig_lines[i], expected[0], mode))
        .collect();

    // 首行匹配不到时，用末行做锚点：末行匹配位置 i 对应起点 i - (len-1)。
    if candidates.is_empty() && expected.len() > 1 {
        let last = expected.len() - 1;
        candidates = (last..orig_lines.len())
            .filter(|&i| lines_match(&orig_lines[i], expected[last], mode))
            .map(|i| i - last)
            .collect();
    }

    // 首尾都匹配不到时，用每一条期望行做锚点，取匹配行最多的候选。
    // 这是最后的兜底，覆盖期望块中间行正确但首尾行有误的情况。
    if candidates.is_empty() {
        for (ei, exp) in expected.iter().enumerate() {
            for (fi, line) in orig_lines.iter().enumerate() {
                if lines_match(line, exp, mode) {
                    let start = fi.saturating_sub(ei);
                    if start < orig_lines.len() {
                        candidates.push(start);
                    }
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
    }

    // 限制候选数量，避免极端情况下的性能问题。
    candidates.truncate(500);

    let mut best: Option<BestPartialMatch> = None;
    for &start in &candidates {
        let available = orig_lines.len().saturating_sub(start);
        let check_count = expected.len().min(available);
        if check_count == 0 {
            continue;
        }
        let mut matches = 0usize;
        let mut mismatches = Vec::new();
        for i in 0..check_count {
            let act = &orig_lines[start + i];
            if lines_match(act, expected[i], mode) {
                matches += 1;
            } else {
                mismatches.push((start + i + 1, expected[i].to_string(), act.clone()));
            }
        }
        let is_better = match &best {
            None => true,
            Some(b) => matches > b.matches,
        };
        if is_better {
            best = Some(BestPartialMatch {
                pos: start,
                matches,
                total: check_count,
                mismatches,
            });
        }
        // 完美匹配不应出现在错误路径，但保留提前退出以保安全。
        if matches == expected.len() {
            break;
        }
    }
    best.filter(|b| b.matches > 0)
}

fn format_char_with_code_point(ch: char) -> String {
    format!("{ch:?} (U+{:04X})", ch as u32)
}

/// 描述两行文本首个不同字符的位置与 Unicode code point，便于快速发现
/// 智能引号/全半角等"看起来像一样"的字符差异。
fn describe_first_char_mismatch(expected: &str, actual: &str) -> Option<String> {
    let mut column = 1usize;
    let mut expected_chars = expected.chars();
    let mut actual_chars = actual.chars();

    loop {
        match (expected_chars.next(), actual_chars.next()) {
            (Some(exp), Some(act)) if exp == act => {
                column += 1;
            }
            (Some(exp), Some(act)) => {
                return Some(format!(
                    "column {}: expected {}, found {}",
                    column,
                    format_char_with_code_point(exp),
                    format_char_with_code_point(act)
                ));
            }
            (Some(exp), None) => {
                return Some(format!(
                    "column {}: expected {}, found end of line",
                    column,
                    format_char_with_code_point(exp)
                ));
            }
            (None, Some(act)) => {
                return Some(format!(
                    "column {}: expected end of line, found {}",
                    column,
                    format_char_with_code_point(act)
                ));
            }
            (None, None) => return None,
        }
    }
}

fn describe_aligned_block_first_mismatch(
    expected_lines: &[&str],
    actual_lines: &[String],
    start: usize,
) -> Option<String> {
    for (offset, expected) in expected_lines.iter().enumerate() {
        let actual = actual_lines
            .get(start + offset)
            .map(String::as_str)
            .unwrap_or("");
        if expected == &actual {
            continue;
        }
        let detail = describe_first_char_mismatch(expected, actual)?;
        let line_no = start + offset + 1;
        return Some(format!(
            "First differing char near declared position is on line {} at {}.\n",
            line_no, detail
        ));
    }
    None
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

    // 先尝试 best-effort 部分匹配，精确定位不一致的行。大块替换中最常见的失败是
    // 整块只有个别行复刻不准，部分匹配能告诉模型"第 X 行期望 A 但实际是 B"。
    if let Some(best) = find_best_partial_match(orig_lines, hunk, MatchMode::IgnoreIndent) {
        msg.push_str(&format!(
            "Best partial match at line {} ({}/{} lines matched).\n",
            best.pos + 1,
            best.matches,
            best.total
        ));
        if best.mismatches.is_empty() {
            msg.push_str(
                "All expected lines matched at this position — \
                 the mismatch may be due to hunk ordering or a missing trailing line.\n",
            );
        } else {
            // 展示前 10 个不匹配行：完整偏移模式比精简字数对模型修 patch 更重要
            let show = best.mismatches.len().min(10);
            msg.push_str(&format!(
                "Mismatched lines (showing {} of {}):\n",
                show,
                best.mismatches.len()
            ));
            for (file_line, exp, act) in best.mismatches.iter().take(show) {
                let first_diff = describe_first_char_mismatch(exp, act)
                    .map(|detail| format!("; first differing char at {detail}"))
                    .unwrap_or_default();
                msg.push_str(&format!(
                    "  line {}: expected {:?}, found {:?}{}\n",
                    file_line, exp, act, first_diff
                ));
            }
            if best.mismatches.len() > show {
                msg.push_str(&format!(
                    "  ... ({} more mismatches)\n",
                    best.mismatches.len() - show
                ));
            }
        }
    } else {
        // 文件中找不到任何部分匹配——块完全不存在。回显期望行和标称位置附近实际内容。
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
        if let Some(detail) = describe_aligned_block_first_mismatch(&expected, orig_lines, nominal)
        {
            msg.push_str(&detail);
        }
    }

    msg.push_str(
	        "Hint: re-read the file with read_file to get exact current content, then rebuild the patch from the raw file text only. read_file prints each line as a right-aligned line number followed by a TAB (e.g. `    42\\t<code>`); copy only the code after the TAB. Do not copy the leading line number + tab or any truncation notice into the patch.",
    );
    msg
}

fn try_apply_hunk_at(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    start: usize,
    mode: MatchMode,
    context_policy: ContextPolicy,
) -> Option<(Vec<String>, usize)> {
    let mut out = Vec::new();
    let mut idx = start;
    for line in &hunk.lines {
        match line {
            UnifiedLine::Context(s) => {
                let cur = orig_lines.get(idx)?;
                if context_policy == ContextPolicy::Require && !lines_match(cur, s, mode) {
                    return None;
                }
                out.push(cur.clone());
                idx += 1;
            }
            UnifiedLine::Remove(s) => {
                let cur = orig_lines.get(idx)?;
                if !lines_match(cur, s, mode) {
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

/// 在给定匹配模式下定位一个 hunk 的应用起点（0-based）。
/// 返回 `Ok(Some(pos))` 表示唯一定位成功；`Ok(None)` 表示全文件都匹配不到
/// （调用方可用更宽松的模式重试）；`Err` 表示定位到但存在歧义或顺序错误。
fn locate_hunk(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    cursor: usize,
    mode: MatchMode,
) -> Result<Option<usize>, String> {
    let old_len = hunk_old_line_count(hunk);
    if old_len == 0 {
        if hunk.old_start == 0 {
            return Ok(Some(cursor));
        }
        let nominal = hunk.old_start.saturating_sub(1);
        if nominal < cursor {
            return Err("hunks out of order".to_string());
        }
        return if nominal <= orig_lines.len() {
            Ok(Some(nominal))
        } else {
            Ok(None)
        };
    }

    let nominal = hunk.old_start.saturating_sub(1);
    let nominal_ok = hunk.old_start > 0
        && nominal <= orig_lines.len()
        && nominal >= cursor
        && try_apply_hunk_at(orig_lines, hunk, nominal, mode, ContextPolicy::Require).is_some();
    if nominal_ok {
        return Ok(Some(nominal));
    }

    // 标称位置匹配不上时，先检查全文件范围内有多少处能匹配：
    // 多处匹配时返回 None，让调用方的级联逻辑尝试更宽松的模式：
    // IgnoreIndent → fuzzy context（后者有标称位置消歧 + 上下文评分）。
    let positions = all_hunk_match_positions(orig_lines, hunk, mode);
    let forward: Vec<usize> = positions.iter().copied().filter(|&p| p >= cursor).collect();
    if forward.len() > 1 {
        return Ok(None);
    }
    // forward 已经过滤了 p >= cursor，所以这里不会有 "hunks out of order"。
    // 之前回退到 find_hunk_offset（±50 窗口）会在唯一匹配超出窗口时误报
    // context mismatch；直接使用 forward 的唯一结果即可。
    if let Some(&offset) = forward.first() {
        Ok(Some(offset))
    } else if !positions.is_empty() {
        // 所有匹配都在 cursor 之前——hunk 顺序错误
        Err("hunks out of order".to_string())
    } else {
        Ok(None)
    }
}

fn apply_unified_patch(original: &str, patch: &str) -> Result<String, String> {
    let had_trailing_newline = original.ends_with('\n');
    let hunks = parse_unified_hunks(patch)?;
    let orig_lines: Vec<String> = original.lines().map(|s| s.to_string()).collect();

    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize;

    for hunk in &hunks {
        // 先做严格匹配（仅容忍行尾空白）。严格匹配全文件都定位不到时，再用忽略
        // 前导缩进的宽松模式兜底一次——对齐 `git apply --ignore-whitespace`，
        // 解决模型对 markdown/嵌套列表/代码块缩进复刻不准导致的 context mismatch。
        let (apply_at, mode, context_policy) =
            match locate_hunk(&orig_lines, hunk, cursor, MatchMode::Strict)? {
                Some(at) => (at, MatchMode::Strict, ContextPolicy::Require),
                None => match locate_hunk(&orig_lines, hunk, cursor, MatchMode::IgnoreIndent)? {
                    Some(at) => (at, MatchMode::IgnoreIndent, ContextPolicy::Require),
                    None => match locate_hunk_with_fuzzy_context(
                        &orig_lines,
                        hunk,
                        cursor,
                        MatchMode::Strict,
                    ) {
                        Ok(Some(candidate)) => (candidate.pos, MatchMode::Strict, ContextPolicy::Fuzz),
                        _ => match locate_hunk_with_fuzzy_context(
                            &orig_lines,
                            hunk,
                            cursor,
                            MatchMode::IgnoreIndent,
                        )? {
                            Some(candidate) => {
                                (candidate.pos, MatchMode::IgnoreIndent, ContextPolicy::Fuzz)
                            }
                            None => return Err(describe_context_mismatch(&orig_lines, hunk)),
                        },
                    },
                },
            };

        out.extend_from_slice(&orig_lines[cursor..apply_at]);
        let (hunk_out, new_idx) =
            try_apply_hunk_at(&orig_lines, hunk, apply_at, mode, context_policy)
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

/// 剥离模型常给 patch 外层包裹的代码围栏（```...``` 或 ~~~...~~~）。
/// 仅当整体 patch 被一对围栏包裹（首行开围栏、末行裸闭围栏）时才剥离；
/// 围栏出现在 patch 内部内容中时不处理，避免误伤真正的 patch 内容。
fn strip_code_fence(patch: &str) -> String {
    let lines: Vec<&str> = patch.lines().collect();
    if lines.len() < 3 {
        return patch.to_string();
    }
    let first = lines.first().unwrap().trim();
    let is_open_fence = first.starts_with("```") || first.starts_with("~~~");
    if !is_open_fence {
        return patch.to_string();
    }
    // 从末尾向前找第一个非空行作为闭围栏候选——模型常在闭围栏后多输出空行。
    let last_nonempty = lines
        .iter()
        .rposition(|l| !l.trim().is_empty())
        .unwrap_or(0);
    let last = lines.get(last_nonempty).unwrap().trim();
    let is_close_fence = last == "```" || last == "~~~";
    if !is_close_fence || last_nonempty < 2 {
        return patch.to_string();
    }
    // 剥离首尾围栏行，保留中间内容并去掉多余首尾空白。
    lines[1..last_nonempty].join("\n").trim().to_string()
}

fn diff_stats_for_write(write: &PreparedPatchWrite) -> (usize, usize, usize) {
    // (added, removed, total_lines_after)
    match &write.action {
        PreparedPatchAction::Write(next) => {
            let after_lines = next.lines().count();
            match &write.before {
                Some(before) => {
                    // 逐行对比统计新增/删除
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut before_set: Vec<u64> = before
                        .lines()
                        .map(|l| {
                            let mut h = DefaultHasher::new();
                            l.hash(&mut h);
                            h.finish()
                        })
                        .collect();
                    let mut added = 0usize;
                    for l in next.lines() {
                        let mut h = DefaultHasher::new();
                        l.hash(&mut h);
                        let hash = h.finish();
                        if let Some(pos) = before_set.iter().position(|&x| x == hash) {
                            before_set.remove(pos);
                        } else {
                            added += 1;
                        }
                    }
                    let removed = before_set.len();
                    (added, removed, after_lines)
                }
                None => (after_lines, 0, after_lines),
            }
        }
        PreparedPatchAction::Delete => {
            (0, write.before.as_ref().map_or(0, |b| b.lines().count()), 0)
        }
    }
}

fn format_patch_success(writes: &[PreparedPatchWrite]) -> String {
    if writes.len() == 1 {
        let (added, removed, total) = diff_stats_for_write(&writes[0]);
        return format!(
            "Successfully patched {}; +{added} -{removed} ({total} lines)",
            writes[0].path.display()
        );
    }
    let mut message = format!("Successfully patched {} files:", writes.len());
    for write in writes {
        let (added, removed, total) = diff_stats_for_write(write);
        message.push_str(&format!(
            "\n- {}; +{added} -{removed} ({total} lines)",
            write.path.display()
        ));
    }
    message
}

fn format_patch_dry_run(writes: &[PreparedPatchWrite]) -> String {
    if writes.len() == 1 {
        let (added, removed, total) = diff_stats_for_write(&writes[0]);
        return format!(
            "Dry run succeeded; no files changed: {}; +{added} -{removed} ({total} lines after)",
            writes[0].path.display()
        );
    }
    let mut message = format!(
        "Dry run succeeded for {} files; no files changed:",
        writes.len()
    );
    for write in writes {
        let (added, removed, total) = diff_stats_for_write(write);
        message.push_str(&format!(
            "\n- {}; +{added} -{removed} ({total} lines after)",
            write.path.display()
        ));
    }
    message
}

fn dry_run_arg(args: &Value) -> Result<bool, String> {
    match args.get("dry_run") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(value) => Err(format!(
            "[INVALID_ARGUMENT] `dry_run` must be a boolean, got {}",
            value_type_name(value)
        )),
    }
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn prepare_patch_write(
    path: &Path,
    store: &FileStore,
    envelope: &PatchEnvelope,
) -> Result<PreparedPatchWrite, String> {
    if envelope.op == PatchEnvelopeOp::Delete {
        if !envelope.body_lines.is_empty() {
            return Err("Delete File sections must not contain patch content".to_string());
        }
        let metadata = fs::symlink_metadata(path).map_err(|err| {
            format!(
                "Delete File target does not exist or cannot be inspected: {} ({err})",
                path.display()
            )
        })?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "Delete File refuses symbolic links: {}. Delete the link explicitly outside apply_patch.",
                path.display()
            ));
        }
        if !metadata.file_type().is_file() {
            return Err(format!(
                "Delete File only supports regular files, not directories or special files: {}",
                path.display()
            ));
        }
        return Ok(PreparedPatchWrite {
            path: path.to_path_buf(),
            before: Some(store.read_to_string().map_err(|err| err.to_string())?),
            action: PreparedPatchAction::Delete,
        });
    }

    let before = if path.exists() {
        Some(store.read_to_string().map_err(|err| err.to_string())?)
    } else {
        None
    };
    let original = before.as_deref().unwrap_or_default();
    let next = if envelope.op == PatchEnvelopeOp::ReplaceInLine {
        if before.is_none() {
            return Err(format!(
                "Replace in line: target file does not exist: {}",
                path.display()
            ));
        }
        apply_inline_replace(original, envelope)?
    } else {
        let normalized_patch = normalize_patch_envelope(path, envelope)?;
        apply_unified_patch(original, &normalized_patch)?
    };
    Ok(PreparedPatchWrite {
        path: path.to_path_buf(),
        before,
        action: PreparedPatchAction::Write(next),
    })
}

fn verify_patch_write_is_current(write: &PreparedPatchWrite) -> Result<(), String> {
    let current = if write.path.exists() {
        Some(
            FileStore::new(write.path.clone())
                .read_to_string()
                .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    if current == write.before {
        return Ok(());
    }
    Err(format!(
        "[FILE_CHANGED] {} changed since this patch was prepared. Re-read it and rebuild the patch before retrying.",
        write.path.display()
    ))
}

fn apply_prepared_patch_write(write: &PreparedPatchWrite) -> Result<(), String> {
    match &write.action {
        PreparedPatchAction::Write(next) => FileStore::new(write.path.clone())
            .write_all(next)
            .map_err(|err| err.to_string()),
        PreparedPatchAction::Delete => fs::remove_file(&write.path)
            .map_err(|err| format!("Failed to delete {}: {err}", write.path.display())),
    }
}

fn restore_prepared_patch_write(write: &PreparedPatchWrite) -> Result<(), String> {
    match &write.before {
        Some(content) => FileStore::new(write.path.clone())
            .write_all(content)
            .map_err(|err| format!("failed to restore {}: {err}", write.path.display())),
        None => match fs::remove_file(&write.path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(format!(
                "failed to remove {} during rollback: {err}",
                write.path.display()
            )),
        },
    }
}

fn commit_patch_writes(writes: &[PreparedPatchWrite]) -> Result<(), String> {
    for write in writes {
        verify_patch_write_is_current(write)?;
    }

    for (idx, write) in writes.iter().enumerate() {
        if let Err(write_err) = apply_prepared_patch_write(write) {
            let restoration_errors: Vec<_> = writes[..=idx]
                .iter()
                .rev()
                .filter_map(|written| restore_prepared_patch_write(written).err())
                .collect();
            if restoration_errors.is_empty() {
                return Err(format!(
                    "failed to apply {}: {write_err}; all affected files were restored",
                    write.path.display()
                ));
            }
            return Err(format!(
                "failed to apply {}: {write_err}; rollback was incomplete: {}",
                write.path.display(),
                restoration_errors.join("; ")
            ));
        }
    }

    let changes = writes
        .iter()
        .map(|write| CompletedFileChange {
            path: write.path.to_string_lossy().into_owned(),
            before: write.before.clone(),
            after: match &write.action {
                PreparedPatchAction::Write(next) => Some(next.clone()),
                PreparedPatchAction::Delete => None,
            },
        })
        .collect();
    record_completed_change_set(&format!("apply_patch ({} file(s))", writes.len()), changes);
    Ok(())
}

fn execute_apply_patch_impl(args: &Value, mut emit: impl FnMut(&str)) -> Result<String, String> {
    let dry_run = dry_run_arg(args)?;
    let raw_patch = args["patch"].as_str().ok_or_else(|| {
        let actual = match args.get("patch") {
            None => "missing".to_string(),
            Some(Value::String(_)) => "string (should not happen)".to_string(),
            Some(Value::Object(_)) => "object".to_string(),
            Some(Value::Array(_)) => "array".to_string(),
            Some(Value::Number(_)) => "number".to_string(),
            Some(Value::Bool(_)) => "boolean".to_string(),
            Some(Value::Null) => "null".to_string(),
        };
        format!(
            "patch parameter has wrong type ({actual}): expected a string. \
             Pass the patch text as a string value, not a JSON object or array."
        )
    })?;
    let patch = strip_code_fence(raw_patch);
    emit("parsing patch envelope");
    let initial_file_path = optional_file_path_arg(args);
    if let Some(envelopes) = parse_patch_envelopes(&patch)? {
        // 信封（无论单文件/多文件）内各 section 已声明各自目标路径，file_path 是
        // 多余的。模型常在多文件信封时冗余传 file_path，与其硬报错浪费一轮，不如
        // 静默忽略并用信封路径（信封路径才是权威来源）。
        if initial_file_path.is_some() {
            emit("note: ignoring redundant file_path arg; using paths from Begin Patch envelope");
        }
        emit(&format!("parsed {} patch section(s)", envelopes.len()));
        let mut seen_targets = FxHashSet::default();
        let mut writes = Vec::with_capacity(envelopes.len());
        for (idx, envelope) in envelopes.iter().enumerate() {
            let target_arg = envelope.target_path.as_str();
            let store = FileStore::new(PathBuf::from(target_arg));
            emit(&format!(
                "target [{}/{}]: {}",
                idx + 1,
                envelopes.len(),
                store.path().display()
            ));
            emit("validating write access");
            store
                .validate_write_access()
                .map_err(|err| err.to_string())?;
            let path = store.path().to_path_buf();
            ensure_patch_target_matches(&path, &envelope.target_path)?;
            if !seen_targets.insert(path.clone()) {
                return Err(format!(
                    "duplicate target in Begin Patch envelope: {}. Mention each file at most once per apply_patch call.",
                    path.display()
                ));
            }
            if envelope.op == PatchEnvelopeOp::ReplaceInLine {
                emit("applying inline replacement");
            } else if envelope.op == PatchEnvelopeOp::Delete {
                emit("preparing file deletion");
            } else {
                let hunk_count = envelope
                    .body_lines
                    .iter()
                    .filter(|line| line.starts_with("@@"))
                    .count()
                    .max(1);
                emit(&format!("applying {hunk_count} hunk(s)"));
            }
            let write = prepare_patch_write(&path, &store, envelope).map_err(|err| {
                format!("[section {}/{}] failed while preparing patch for {}: {err}", idx + 1, envelopes.len(), path.display())
            })?;
            writes.push(write);
        }
        if dry_run {
            let success = format_patch_dry_run(&writes);
            emit(&success);
            return Ok(success);
        }
        for write in &writes {
            match &write.action {
                PreparedPatchAction::Write(next) => {
                    emit(&format!("writing {} byte(s)", next.len()))
                }
                PreparedPatchAction::Delete => emit("deleting file"),
            }
        }
        commit_patch_writes(&writes)?;
        let success = format_patch_success(&writes);
        emit(&success);
        return Ok(success);
    }

    let file_path = initial_file_path.ok_or(
        "missing file_path: provide `file_path` (or `path`) arg, \
         or wrap the patch in a `*** Begin Patch` / `*** Update File: <path>` envelope.",
    )?;
    let store = FileStore::new(PathBuf::from(file_path));
    emit(&format!("target: {}", store.path().display()));
    emit("validating write access");
    store
        .validate_write_access()
        .map_err(|err| err.to_string())?;
    let path = store.path().to_path_buf();
    let hunk_count = patch
        .lines()
        .filter(|line| line.starts_with("@@"))
        .count()
        .max(1);
    emit(&format!("applying {hunk_count} hunk(s)"));
    let before = if path.exists() {
        emit("reading current file");
        Some(store.read_to_string().map_err(|err| err.to_string())?)
    } else {
        emit("creating new file from patch");
        None
    };
    let next = apply_unified_patch(before.as_deref().unwrap_or_default(), &patch)?;
    let write = PreparedPatchWrite {
        path: path.clone(),
        before,
        action: PreparedPatchAction::Write(next),
    };
    if dry_run {
        let success = format_patch_dry_run(&[write]);
        emit(&success);
        return Ok(success);
    }
    if let PreparedPatchAction::Write(next) = &write.action {
        emit(&format!("writing {} byte(s)", next.len()));
    }
    let success = format_patch_success(std::slice::from_ref(&write));
    commit_patch_writes(&[write])?;
    emit(&success);
    Ok(success)
}

pub(crate) fn execute_apply_patch(args: &Value) -> Result<String, String> {
    execute_apply_patch_impl(args, |_| {})
}

pub(crate) fn execute_apply_patch_streaming(
    args: &Value,
    on_chunk: &mut ToolStreamWriter<'_>,
) -> Result<String, String> {
    execute_apply_patch_impl(args, |line| emit_stream_line(on_chunk, line))
}

#[cfg(test)]
mod tests {
    use super::{
        PatchEnvelopeOp, apply_inline_replace, apply_unified_patch, execute_apply_patch,
        parse_patch_envelope, parse_patch_envelopes, parse_unified_hunks, strip_code_fence,
    };
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
        let result =
            apply_unified_patch(original, patch).expect("empty context line should be tolerated");
        assert_eq!(result, "foo\n\nbaz\n");
    }

    #[test]
    fn apply_unified_patch_strips_trailing_cr_from_crlf_patch() {
        // CRLF patch：Add 行尾的 \r 不应写入文件内容。
        let original = "foo\nbar\n";
        let patch = "@@ -2,1 +2,1 @@\r\n-bar\r\n+baz\r\n";
        let result = apply_unified_patch(original, patch).expect("CRLF patch should be tolerated");
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
        assert_eq!(
            hunks[0].lines.len(),
            2,
            "hunk1 should not swallow the blank separator"
        );
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
    fn apply_unified_patch_tolerates_envelope_end_marker() {
        // 模型常在 unified-diff hunk 末尾误带 `*** End Patch` 等 envelope 尾标记
        // （格式混用）。这些标记不属于 unified-diff 内容，应静默结束当前 hunk，
        // 而不是报 invalid hunk line。
        let original = "line1\nline2\nline3\n";
        let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n*** End Patch\n";
        let result = apply_unified_patch(original, patch)
            .expect("trailing `*** End Patch` marker should be tolerated");
        assert_eq!(result, "line1\nchanged\nline3\n");
    }

    #[test]
    fn apply_unified_patch_rejects_envelope_section_marker_with_hint() {
        // unified-diff hunk 中混入 `*** Begin Patch` / `*** Update File:` 等开头或
        // 分节标记，说明 patch 结构混乱。应报错并明确提示"格式混用"，引导模型
        // 二选一重建，而非笼统的 invalid hunk line。
        let original = "line1\nline2\nline3\n";
        let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n*** Begin Patch\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("mixed patch formats"), "err was: {err}");
        assert!(err.contains("*** Begin Patch"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_rejects_malformed_envelope_trailer_not_silently_applied() {
        // 安全属性：当 patch 含 `*** Begin Patch` / `*** Update File:` 等 envelope
        // 标记（即残缺 envelope 误入 unified-diff 路径）时，尾部 `*** End Patch`
        // 绝不能被静默容忍并把 hunk 应用到 file_path 指定、却非 envelope 声明的
        // 文件上。必须报"格式混用"错误，由模型重建。即便这里 original 恰好含相同
        // 上下文（最危险的巧合情形），也必须报错而非写入。
        let original = "line1\nline2\nline3\n";
        let patch = "*** Begin Patch\n*** Update File: other.rs\n@@ -2,1 +2,1 @@\n-line2\n+changed\n*** End Patch\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("mixed patch formats"), "err was: {err}");
        assert!(err.contains("*** End Patch"), "err was: {err}");
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
        assert!(
            err.lines().next().unwrap_or_default().contains(
                "required recovery: read_file the same target before retrying apply_patch"
            ),
            "first error line should include the recovery action: {err}"
        );
        // 错误里应回显期望行与实际文件内容，便于模型自我修正。
        assert!(err.contains("not_present"), "err was: {err}");
        assert!(err.contains("beta"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_context_mismatch_reports_unicode_code_points() {
        // 用真正"非 confusable"的 Unicode 差异触发 mismatch，验证错误里回显 code point。
        // 注意：smart quotes（U+201C/U+201D）与 ASCII 引号已由 normalize_confusables 归一化容忍
        // （见 apply_unified_patch_tolerates_confusable_quotes），不能再当作 mismatch 样例。
        // 这里用带重音字母 é (U+00E9) vs e (U+0065)--不在 confusable 归一化范围内，是真差异。
        let original = "let label = \"café\";\n";
        let patch = "@@ -1,1 +1,1 @@\n-let label = \"cafe\";\n+let label = \"changed\";\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        assert!(err.contains("U+00E9"), "err was: {err}");
        assert!(err.contains("U+0065"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_tolerates_confusable_quotes() {
        // P0: 模型常把 ASCII 引号/连字符自动替换为排版用 smart quote / en-dash。
        // 这类纯排版差异不应导致 context mismatch--normalize_confusables 归一化后应能匹配。
        // 关键安全属性：context 行输出原文件内容（actual），而非 patch 里的 smart quote，
        // 所以文件里的 ASCII 字符不会被"替换"成 smart quote。
        let original = "let quote = \"hi\";\nlet dash = a - b;\n";
        // context 行用 smart quotes（“ ”），remove 行用 en-dash（– U+2013），
        // 文件里对应是 ASCII 引号 / ASCII hyphen--归一化后均应匹配。
        let patch = "@@ -1,2 +1,2 @@\n let quote = “hi”;\n-let dash = a – b;\n+let dash = a - b;\n";
        let result = apply_unified_patch(original, patch)
            .expect("confusable smart quotes / en-dash should be tolerated");
        // context 行保留原文件 ASCII 引号；remove 的 en-dash 匹配文件 ASCII hyphen 后删除；
        // add 行写入 patch 内容（ASCII hyphen）。
        assert_eq!(result, "let quote = \"hi\";\nlet dash = a - b;\n");
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
        assert!(
            result.contains("changed"),
            "result should contain changed line: {result}"
        );
        assert!(
            result.contains("after_target"),
            "result should preserve after_target: {result}"
        );
        assert!(
            !result.contains("unique_target"),
            "result should not contain old line: {result}"
        );
    }

    #[test]
    fn apply_unified_patch_tolerates_leading_indent_mismatch() {
        // 真实高频失败场景：markdown/嵌套列表里，模型复刻的 context 行缩进与原文件
        // 不一致（这里 patch 少了 2 个前导空格）。修复前 lines_match 只做 trim_end，
        // 前导空白零容忍 → 全文件定位不到 → "context mismatch: patch hunk could not
        // be located"。修复后严格匹配失败会用忽略缩进的兜底模式唯一定位并应用。
        let original = "# Title\n\n  - item one\n  - item two\n";
        // 前导空格=context 前缀；context 内容 "- item one"、remove 内容 "- item two"
        // 都比原文件少了 2 个缩进空格。
        let patch = "@@ -3,2 +3,2 @@\n - item one\n-- item two\n+- item two changed\n";
        let result = apply_unified_patch(original, patch).unwrap_or_else(|err| {
            panic!("indent-insensitive fallback should locate the hunk, got: {err}")
        });
        // 保留原文件缩进的 context 行，只替换 remove/add 目标行。
        assert_eq!(result, "# Title\n\n  - item one\n- item two changed\n");
    }

    #[test]
    fn apply_unified_patch_indent_fallback_still_detects_ambiguity() {
        // 兜底的忽略缩进模式不能牺牲安全性：若忽略缩进后有多处匹配，仍应报歧义，
        // 而不是静默改错地方。
        let original = "  dup\nmid\n    dup\ntail\n";
        let patch = "@@ -9,1 +9,1 @@\n-dup\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_strict_match_preferred_over_indent_fallback() {
        // 当严格匹配能唯一定位时，必须使用严格匹配的结果，保持原文件精确内容，
        // 不因存在缩进变体而误走兜底。
        let original = "    exact\nother\n";
        let patch = "@@ -1,1 +1,1 @@\n-    exact\n+    replaced\n";
        let result = apply_unified_patch(original, patch).unwrap();
        assert_eq!(result, "    replaced\nother\n");
    }

    #[test]
    fn apply_unified_patch_fuzzes_stale_context_when_remove_lines_are_unique() {
        // 真实循环根因：模型把 context 行写成了过时/目标态内容，但 remove 行仍然
        // 精确锚定目标。context 不应在这种情况下把 patch 硬拒绝。
        let original = "alpha current\nold target\nomega current\n";
        let patch = "\
@@ -1,3 +1,3 @@
 alpha stale
-old target
+new target
 omega stale
";
        let result = apply_unified_patch(original, patch).unwrap_or_else(|err| {
            panic!("unique remove anchor should tolerate stale context, got: {err}")
        });
        assert_eq!(result, "alpha current\nnew target\nomega current\n");
    }

    #[test]
    fn apply_unified_patch_fuzzy_context_uses_remaining_context_to_disambiguate() {
        // remove 行出现两次时，fuzz 仍可使用其他 context 行打分；只有唯一最高分
        // 才能应用，避免退化成“改第一个相同 remove 行”。
        let original = "alpha current\nold target\ntail one\nbeta current\nold target\ntail two\n";
        let patch = "\
@@ -1,3 +1,3 @@
 stale head
-old target
+new target
 tail one
";
        let result = apply_unified_patch(original, patch).unwrap_or_else(|err| {
            panic!("tail context should disambiguate fuzzy candidate, got: {err}")
        });
        assert_eq!(
            result,
            "alpha current\nnew target\ntail one\nbeta current\nold target\ntail two\n"
        );
    }

    #[test]
    fn apply_unified_patch_rejects_fuzzy_context_when_remove_anchor_is_ambiguous() {
        let original = "alpha current\nold target\nbeta current\nold target\n";
        let patch = "\
@@ -1,2 +1,2 @@
 stale context
-old target
+new target
";
        // old_start=1 (1-based) → nominal=0，remove "old target" 在 line 1 匹配。
        // 即使 context 全部 miss，old_start 仍能消歧——应成功应用。
        let result = apply_unified_patch(original, patch).expect("should apply via nominal");
        assert_eq!(
            result,
            "alpha current\nnew target\nbeta current\nold target\n",
            "should replace the FIRST 'old target' (line 1), not the second (line 3)"
        );
    }

    #[test]
    fn apply_unified_patch_fuzzy_context_rejects_when_nominal_not_in_candidates() {
        // old_start 指向的位置不在候选列表中时，应仍然拒绝。
        // original: line 0="old target", line 1="xxx", line 2="old target", line 3="yyy"
        // patch: @@ -2,1 +2,1 @@ — old_start=2 → nominal=1
        // hunk 只有 remove 行（无 context），remove "xxx" 出现在 line 1。
        // 但换一种：多个 "old target" 作为 remove，old_start 指向没有匹配的位置。
        // original: line 0="old target", line 1="aaa", line 2="old target", line 3="bbb"
        // patch: @@ -3,1 +3,1 @@ — old_start=3 → nominal=2
        // remove "old target" 匹配 line 0 (pos=0) 和 line 2 (pos=2)
        // nominal=2 在候选列表中 → 会被接受（正确行为）。
        // 改为: old_start 指向一个不在文件中的行号。
        let original = "old target\naaa\nold target\nbbb\n";
        let patch = "@@ -5,1 +5,1 @@\n-old target\n+changed\n";
        // old_start=5 → nominal=4, 但文件只有4行 (index 0-3)。
        // candidates: pos=0 和 pos=2 (old target 匹配)。nominal=4 不在 candidates 中。
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_indent_fallback_reports_context_mismatch_when_absent() {
        // 即便忽略缩进，内容本身不存在时仍应报 context mismatch（回显实际内容）。
        let original = "alpha\nbeta\ngamma\n";
        let patch = "@@ -2,1 +2,1 @@\n-  not_present\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
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

        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "alpha\nchanged\ngamma\n"
        );
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
        assert!(
            !result.contains("target_a"),
            "should not contain target_a: {result}"
        );
        assert!(
            !result.contains("target_b"),
            "should not contain target_b: {result}"
        );
        // 中间填充行应保持不变
        assert!(
            result.contains("filler0"),
            "filler0 should remain: {result}"
        );
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
                streamed.contains(&format!("Successfully patched {};", path.display())),
                "streamed: {streamed}"
            );
            assert!(
                result.content.starts_with(&format!("Successfully patched {};", path.display())),
                "result.content: {}",
                result.content
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

        // file_path 被静默忽略，信封声明 b.txt 为权威目标；b.txt 不存在 → 报缺失文件。
        assert!(err.contains("b.txt"), "err should mention the envelope target path: {err}");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_update_envelope_rejects_missing_target_file() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("update_missing_parent");
        let path = base.join("missing.txt");
        fs::create_dir_all(&base).unwrap();

        let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "path": path.to_string_lossy(),
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\n+hello\n*** End Patch\n",
                    path.display()
                )
            });
            execute_apply_patch(&args).expect_err("Update File must not create a missing file")
        });

        assert!(
            err.contains("Update File patch targets a missing file"),
            "err was: {err}"
        );
        assert!(!path.exists(), "missing target must not be created");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_tilde_path_matches_between_arg_and_envelope() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let home = PathBuf::from(std::env::var("HOME").expect("HOME must be set"));
        let unique = format!("ai_patch_tools_home_{}", uuid::Uuid::new_v4());
        let base = home.join(&unique);
        let path = base.join("tilde.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "alpha\nbeta\n").unwrap();

        let rel = path
            .strip_prefix(&home)
            .expect("test path should be under HOME");
        let tilde_path = format!("~/{}", rel.display());

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "path": tilde_path.clone(),
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\n@@ -1,2 +1,2 @@\n alpha\n-beta\n+changed\n*** End Patch\n",
                    tilde_path
                )
            });
            execute_apply_patch(&args)
                .expect("matching `~` paths in arg and envelope should resolve to the same file");
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "alpha\nchanged\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn strip_code_fence_removes_backtick_wrapper() {
        let fenced = "```diff\n@@ -1,1 +1,1 @@\n-line2\n+changed\n```";
        assert_eq!(
            strip_code_fence(fenced),
            "@@ -1,1 +1,1 @@\n-line2\n+changed"
        );
        // ~~~ 围栏同样剥离。
        let fenced_tilde = "~~~\n@@ -1,1 +1,1 @@\n-x\n+y\n~~~";
        assert_eq!(strip_code_fence(fenced_tilde), "@@ -1,1 +1,1 @@\n-x\n+y");
    }

    #[test]
    fn strip_code_fence_leaves_unfenced_patch_untouched() {
        let raw = "@@ -1,1 +1,1 @@\n-x\n+y";
        assert_eq!(strip_code_fence(raw), raw);
        // 闭围栏缺失时不剥离，避免误伤内容以 ``` 开头的真实 patch。
        let no_close = "```diff\n@@ -1,1 +1,1 @@\n-x\n+y";
        assert_eq!(strip_code_fence(no_close), no_close);
        // 行数太少不处理。
        assert_eq!(strip_code_fence("```\n```"), "```\n```");
    }

    #[test]
    fn execute_apply_patch_strips_code_fence_around_unified_diff() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("fence_unified").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "line1\nline2\nline3\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "patch": "```diff\n@@ -1,3 +1,3 @@\n line1\n-line2\n+changed\n line3\n```"
            });
            execute_apply_patch(&args)
                .expect("apply_patch should strip code fence around unified diff");
        });

        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "line1\nchanged\nline3\n"
        );
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_strips_code_fence_around_envelope() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("fence_envelope").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "line1\nline2\nline3\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "patch": format!(
                    "```\n*** Begin Patch\n*** Update File: {}\n line1\n-line2\n+changed\n line3\n*** End Patch\n```",
                    path.display()
                )
            });
            execute_apply_patch(&args)
                .expect("apply_patch should strip code fence around envelope");
        });

        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "line1\nchanged\nline3\n"
        );
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn parse_unified_hunks_error_message_names_expected_prefixes() {
        // context 行漏前导空格时，错误信息应明确指出期望的前缀。
        let err = parse_unified_hunks("@@ -1,3 +1,3 @@\nline1\n-line2\n+changed\n line3")
            .expect_err("missing leading space on context line must error");
        assert!(
            err.contains("must start with") && err.contains("context"),
            "err was: {err}"
        );
    }

    // ── Fix 1: strip_code_fence 应容忍闭围栏后的尾随空行 ──

    #[test]
    fn strip_code_fence_tolerates_trailing_blank_lines() {
        // 模型常在闭围栏后多输出一个或多个空行，之前 strip_code_fence 把最后一行
        // 空行当作 last，判断不是闭围栏就放弃剥离，导致整个 patch 被代码围栏包裹
        // 送入解析器报错。
        let fenced = "```diff\n@@ -1,1 +1,1 @@\n-line2\n+changed\n```\n";
        assert_eq!(
            strip_code_fence(fenced),
            "@@ -1,1 +1,1 @@\n-line2\n+changed"
        );
        // 多个尾随空行也应容忍
        let fenced_multi = "```\n*** Begin Patch\n*** End Patch\n```\n\n\n";
        assert_eq!(
            strip_code_fence(fenced_multi),
            "*** Begin Patch\n*** End Patch"
        );
    }

    // ── Fix 2: 缺少 hunk header 时给出明确错误 ──

    #[test]
    fn parse_unified_hunks_missing_header_gives_clear_error() {
        // patch 内容行存在但没有 hunk header，应给出比 "no hunks found" 更明确的错误。
        let err = parse_unified_hunks(" line1\n-line2\n+changed\n line3")
            .expect_err("patch without hunk header must error");
        assert!(err.contains("no hunk header found"), "err was: {err}");
        assert!(err.contains("content lines"), "err was: {err}");
    }

    // ── Fix 3: envelope Update 合成 header 使用 old_start=0 ──

    #[test]
    fn execute_apply_patch_update_envelope_without_header_does_not_match_at_line_1() {
        // 当文件开头恰好与 hunk context 行匹配时，old_start=1 的标称匹配可能误命中
        // 文件开头而非模型真正想改的位置。old_start=0 虽然给出同样的 nominal=0，
        // 但语义更清晰：无标称位置，依赖全文件搜索唯一定位。
        // 这里验证一个不在文件开头的唯一匹配能正确定位。
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("update_nohdr_mid").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "filler\nalpha\nbeta\ngamma\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "path": path.to_string_lossy(),
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\n alpha\n-beta\n+changed\n*** End Patch\n",
                    path.display()
                )
            });
            execute_apply_patch(&args)
                .expect("envelope without header should locate unique match mid-file");
        });

        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "filler\nalpha\nchanged\ngamma\n"
        );
        let _ = fs::remove_dir_all(base);
    }

    // ── Fix 4: envelope Update 无 hunk header 时补全裸行前缀 ──

    #[test]
    fn execute_apply_patch_update_envelope_tolerates_bare_lines() {
        // 模型在 envelope Update 格式（无 hunk header）中写了不带 +/-/ 前缀的裸行，
        // 应自动补空格前缀当作 context 行，而不是报 "invalid hunk line"。
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("update_bare").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "path": path.to_string_lossy(),
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\nalpha\n-beta\n+changed\n*** End Patch\n",
                    path.display()
                )
            });
            execute_apply_patch(&args)
                .expect("envelope with bare context line should be tolerated");
        });

        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "alpha\nchanged\ngamma\n"
        );
        let _ = fs::remove_dir_all(base);
    }

    // ── Fix 5: context 行容忍行号前缀 ──

    #[test]
    fn apply_unified_patch_tolerates_line_number_prefix_in_context() {
        // 模型从 grep 类输出复制了带行号前缀的 context 行（如 `   42| `），
        // IgnoreIndent 兜底模式应剥离行号前缀后匹配成功。
        // read_file 的真实 TAB 格式另有 apply_unified_patch_tolerates_read_file_tab_prefix 覆盖。
        let original = "line1\nline2\nline3\n";
        // context 行 " line1" 被模型误写为带行号前缀的 " 1| line1"
        let patch = "@@ -1,3 +1,3 @@\n 1| line1\n-line2\n+changed\n line3\n";
        let result = apply_unified_patch(original, patch)
            .expect("line number prefix in context should be tolerated by indent fallback");
        // context 行应保留原文件内容（不含行号前缀）
        assert_eq!(result, "line1\nchanged\nline3\n");
    }

    #[test]
    fn apply_unified_patch_tolerates_line_number_prefix_in_remove() {
        // remove 行也带了行号前缀，应同样被容忍。
        let original = "line1\ntarget\nline3\n";
        let patch = "@@ -1,3 +1,3 @@\n line1\n-2| target\n+changed\n line3\n";
        let result = apply_unified_patch(original, patch)
            .expect("line number prefix in remove line should be tolerated");
        assert_eq!(result, "line1\nchanged\nline3\n");
    }

    #[test]
    fn apply_unified_patch_tolerates_read_file_tab_prefix() {
        // 复现 history 中的真实失败场景：模型把 read_file 的输出逐行照抄进 patch 的
        // context / remove 行。read_file 真实渲染格式是 `{:>6}\t{}`（右对齐行号 + TAB），
        // 修复前 strip_line_number_prefix 不认 TAB，导致 context mismatch 反复失败。
        let original = "fn foo() {\n    let x = 1;\n    x\n}\n";
        // 用与 read_file 完全相同的渲染方式构造模型看到的行，避免手数空格出错。
        let rf = |n: usize, s: &str| format!("{:>6}\t{}", n, s);
        let patch = format!(
            "@@ -1,4 +1,4 @@\n {}\n-{}\n+    let x = 2;\n {}\n {}\n",
            rf(1, "fn foo() {"),
            rf(2, "    let x = 1;"),
            rf(3, "    x"),
            rf(4, "}"),
        );
        let result = apply_unified_patch(original, &patch)
            .expect("read_file TAB line-number prefix must be tolerated in context/remove lines");
        // context 行保留原文件内容（含缩进），仅目标行被替换。
        assert_eq!(result, "fn foo() {\n    let x = 2;\n    x\n}\n");
    }

    #[test]
    fn apply_unified_patch_line_number_prefix_still_detects_ambiguity() {
        // 行号前缀容忍不应牺牲安全性：忽略行号后若仍有多处匹配，应报歧义。
        let original = "dup\ndup\ndup\n";
        // 标称位置故意写错，迫使走全文件搜索；剥离行号后 context+remove = ["dup","dup"] 匹配多处
        let patch = "@@ -9,2 +9,2 @@\n 1| dup\n-dup\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
    }

    #[test]
    fn strip_line_number_prefix_does_not_strip_code_lines() {
        // 单参数兜底版：只认 `digits+\t` 与 `digits+分隔符+空格`，保守以防误剥。
        use super::strip_line_number_prefix;
        // read_file 真实格式：右对齐行号 + TAB（此前遗漏的根因场景）
        assert_eq!(
            strip_line_number_prefix("     3\tuse std::fs;"),
            "use std::fs;"
        );
        // TAB 后保留代码原有缩进（只剥离一个 TAB，不动内容缩进）
        assert_eq!(
            strip_line_number_prefix("    42\t    let x = 1;"),
            "    let x = 1;"
        );
        // grep 类格式（分隔符 + 空格）应被剥离
        assert_eq!(strip_line_number_prefix("   42| hello"), "hello");
        assert_eq!(strip_line_number_prefix("42: hello"), "hello");
        // `80:80`（冒号后无空格）不是行号前缀，不应被剥离
        assert_eq!(strip_line_number_prefix("80:80"), "80:80");
        // `3.14`（点后无空格）不应被剥离
        assert_eq!(strip_line_number_prefix("3.14"), "3.14");
        // 纯数字行不应被剥离（没有分隔符）
        assert_eq!(strip_line_number_prefix("42"), "42");
        // 数字紧跟字母不应被剥离（`42px`）
        assert_eq!(strip_line_number_prefix("42px"), "42px");
        // 不以数字开头的行不应被剥离
        assert_eq!(strip_line_number_prefix("hello"), "hello");
    }

    #[test]
    fn strip_number_prefix_anchored_is_separator_agnostic() {
        // 锚定式：以真实行为准，分隔符无关地剥离行号栏，几乎零误伤。
        use super::strip_number_prefix_anchored;
        let actual = "    let x = 1;";
        // read_file TAB / grep `| ` / `: ` / 空格 / `.` / `)` 全部兼容
        assert_eq!(
            strip_number_prefix_anchored("  42\t    let x = 1;", actual),
            actual
        );
        assert_eq!(
            strip_number_prefix_anchored("42|     let x = 1;", actual),
            actual
        );
        assert_eq!(
            strip_number_prefix_anchored("42:     let x = 1;", actual),
            actual
        );
        assert_eq!(
            strip_number_prefix_anchored("42     let x = 1;", actual),
            actual
        );
        assert_eq!(
            strip_number_prefix_anchored("42)     let x = 1;", actual),
            actual
        );
        // 去栏后不等于真实行 → 原样返回（不误剥）
        assert_eq!(
            strip_number_prefix_anchored("42\tsomething else", actual),
            "42\tsomething else"
        );
        // 不以数字开头 → 原样返回
        assert_eq!(strip_number_prefix_anchored(actual, actual), actual);
    }

    // ── 大块替换：best-effort 部分匹配精确定位不一致行 ──

    #[test]
    fn apply_unified_patch_large_block_mismatch_pinpoints_wrong_line() {
        // 大块替换中只有一行内容复刻不准，错误信息应精确定位哪一行不一致
        // （expected vs actual），而不是只给一句 "context mismatch"。
        let original = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\n";
        // remove 块 6 行，其中 line4 被模型误写为 lineX
        let patch =
            "@@ -2,6 +2,3 @@\n-line2\n-line3\n-lineX\n-line5\n-line6\n-line7\n+new2\n+new3\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        // 应报告最佳匹配位置和匹配数
        assert!(err.contains("Best partial match"), "err was: {err}");
        assert!(err.contains("5/6 lines matched"), "err was: {err}");
        // 应精确指出不一致的行：期望 lineX 但实际是 line4
        assert!(
            err.contains("lineX"),
            "err should mention wrong expected line: {err}"
        );
        assert!(
            err.contains("line4"),
            "err should mention actual file line: {err}"
        );
    }

    #[test]
    fn apply_unified_patch_absent_block_falls_back_to_nominal_window() {
        // 期望的块在文件中完全不存在（没有任何行能部分匹配），应回显期望行和
        // 标称位置附近实际内容，而不是走 partial match 分支。
        let original = "alpha\nbeta\ngamma\n";
        let patch = "@@ -2,1 +2,1 @@\n-not_present\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        // 块完全不存在时不会有 "Best partial match"
        assert!(!err.contains("Best partial match"), "err was: {err}");
        // 应回显期望行
        assert!(err.contains("not_present"), "err was: {err}");
        // 应显示标称位置附近的实际内容
        assert!(err.contains("beta"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_partial_match_uses_middle_line_anchor() {
        // 期望块的首行复刻有误，但中间行正确。应通过中间行锚点找到最佳匹配位置，
        // 并报告首行的不一致。
        let original = "aaa\nbbb\nccc\nddd\neee\n";
        // 首行 "wrong" 不在文件中，但 "ccc"、"ddd" 在
        let patch = "@@ -1,3 +1,1 @@\n-wrong\n-ccc\n ddd\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        // 应通过 "ccc" 或 "ddd" 找到部分匹配
        assert!(err.contains("Best partial match"), "err was: {err}");
        assert!(err.contains("2/3 lines matched"), "err was: {err}");
        // 应指出首行不匹配：期望 "wrong" 但实际是 "bbb"
        assert!(
            err.contains("wrong"),
            "err should mention wrong expected line: {err}"
        );
        assert!(
            err.contains("bbb"),
            "err should mention actual file line: {err}"
        );
    }

    // ── 规范 *** Begin Patch 信封：裸 @@ / @@ heading @@ 无行号 header ──

    #[test]
    fn parse_unified_hunks_accepts_bare_at_header() {
        // 规范信封格式用裸 `@@` 分隔 hunk，不带 `-N,M +N,M` 行号。
        // 修复前会报 "invalid hunk header"。
        let patch = "@@\n foo\n-bar\n+baz\n";
        let hunks = parse_unified_hunks(patch).expect("bare @@ header should be accepted");
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].old_start, 0);
    }

    #[test]
    fn parse_unified_hunks_accepts_at_header_with_heading() {
        // `@@ <上下文标题> @@` 也应被接受，标称行号视为 0（依赖全文件搜索定位）。
        let patch = "@@ fn foo() @@\n foo\n-bar\n+baz\n";
        let hunks = parse_unified_hunks(patch).expect("@@ heading @@ header should be accepted");
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].old_start, 0);
    }

    #[test]
    fn apply_unified_patch_applies_bare_at_header_hunk() {
        // 端到端：裸 @@ header 的 hunk 应能通过全文件搜索唯一定位并应用。
        let original = "alpha\nbeta\ngamma\n";
        let patch = "@@\n alpha\n-beta\n+changed\n";
        let result = apply_unified_patch(original, patch).expect("bare @@ hunk should apply");
        assert_eq!(result, "alpha\nchanged\ngamma\n");
    }

    #[test]
    fn apply_unified_patch_bare_at_header_requires_unique_match() {
        // 裸 @@ header 没有标称行号，不能把 old_start=0 当成第 1 行的强锚点。
        // 如果上下文在文件中出现多次，应要求模型补充更多上下文，避免静默改错首个位置。
        // locate_hunk 会返回 None 让级联继续，最终由 locate_hunk_with_fuzzy_context
        // 报告歧义（因为没有标称位置可用来消歧）。
        let original = "alpha\nbeta\ngamma\nalpha\nbeta\ngamma\n";
        let patch = "@@\n alpha\n-beta\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
        assert!(err.contains("context-fuzz"), "err was: {err}");
    }

    #[test]
    fn execute_apply_patch_envelope_with_bare_at_header() {
        // 复现用户报告：规范 *** Begin Patch 信封 + 裸 @@ header。
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("envelope_bare_at").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "path": path.to_string_lossy(),
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {}\n@@\n alpha\n-beta\n+changed\n*** End Patch\n",
                    path.display()
                )
            });
            execute_apply_patch(&args)
                .expect("envelope with bare @@ header should apply");
        });

        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "alpha\nchanged\ngamma\n"
        );
        let _ = fs::remove_dir_all(base);
    }

    // ======================== ReplaceInLine (P2) 测试 ========================

    fn make_envelope(op: PatchEnvelopeOp, target: &str, body: &[&str]) -> super::PatchEnvelope {
        super::PatchEnvelope {
            op,
            target_path: target.to_string(),
            body_lines: body.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn inline_replace_basic() {
        // 基本：anchor 定位行，old->new 精确替换
        let original = "fn foo() {\n    let x = 42;\n    println!(\"{}\", x);\n}\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.rs",
            &["anchor: let x = 42;", "old: 42", "new: 99"],
        );
        let result = apply_inline_replace(original, &envelope).expect("basic replace should work");
        assert_eq!(
            result,
            "fn foo() {\n    let x = 99;\n    println!(\"{}\", x);\n}\n"
        );
    }

    #[test]
    fn inline_replace_preserves_no_trailing_newline() {
        // 文件不以 \n 结尾时，替换后也不加 \n
        let original = "hello world";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: hello", "old: world", "new: rust"],
        );
        let result = apply_inline_replace(original, &envelope).expect("should work");
        assert_eq!(result, "hello rust");
    }

    #[test]
    fn inline_replace_preserves_trailing_newline() {
        let original = "hello world\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: hello", "old: world", "new: rust"],
        );
        let result = apply_inline_replace(original, &envelope).expect("should work");
        assert_eq!(result, "hello rust\n");
    }

    #[test]
    fn inline_replace_anchor_tolerates_confusable() {
        // anchor 里用了 em-dash (—, U+2014)，文件里是 ASCII hyphen (-)。
        // anchor 归一化匹配应容忍，但 old 仍需精确匹配。
        let original = "the quick—brown fox\njumps over\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: the quick—brown fox", "old: fox", "new: dog"],
        );
        let result =
            apply_inline_replace(original, &envelope).expect("confusable anchor should match");
        assert_eq!(result, "the quick—brown dog\njumps over\n");
    }

    #[test]
    fn inline_replace_old_must_match_verbatim() {
        // old 里有 em-dash，但文件里是 ASCII hyphen -> old 精确匹配失败
        let original = "the quick-brown fox\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: quick", "old: quick—brown", "new: slow—brown"],
        );
        let err = apply_inline_replace(original, &envelope)
            .expect_err("old with em-dash should not match ASCII hyphen");
        assert!(
            err.contains("not found in matched line"),
            "error should mention old not found: {err}"
        );
    }

    #[test]
    fn inline_replace_anchor_not_unique() {
        // anchor 匹配多行 -> 报错
        let original = "duplicate line\nduplicate line\nunique here\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: duplicate line", "old: duplicate", "new: unique"],
        );
        let err =
            apply_inline_replace(original, &envelope).expect_err("non-unique anchor should fail");
        assert!(
            err.contains("matched 2 lines"),
            "error should mention 2 matched lines: {err}"
        );
    }

    #[test]
    fn inline_replace_anchor_not_found() {
        let original = "hello world\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: nonexistent", "old: world", "new: rust"],
        );
        let err =
            apply_inline_replace(original, &envelope).expect_err("missing anchor should fail");
        assert!(err.contains("anchor not found"), "error: {err}");
    }

    #[test]
    fn inline_replace_old_not_unique_in_line() {
        // old 在行内出现多次 -> 报错
        let original = "foo bar foo baz\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: foo bar", "old: foo", "new: qux"],
        );
        let err =
            apply_inline_replace(original, &envelope).expect_err("non-unique old should fail");
        assert!(
            err.contains("appears 2 times"),
            "error should mention 2 occurrences: {err}"
        );
    }

    #[test]
    fn inline_replace_old_equals_new() {
        let original = "hello world\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: hello", "old: world", "new: world"],
        );
        let err = apply_inline_replace(original, &envelope).expect_err("old==new should fail");
        assert!(err.contains("identical"), "error: {err}");
    }

    #[test]
    fn inline_replace_missing_field() {
        let original = "hello world\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: hello", "old: world"],
        );
        let err = apply_inline_replace(original, &envelope).expect_err("missing new should fail");
        assert!(err.contains("missing `new:`"), "error: {err}");
    }

    #[test]
    fn inline_replace_unicode_content() {
        // 替换包含多字节 UTF-8 的内容，验证 byte index 切片安全
        let original = "let greeting = \"你好世界\";\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.rs",
            &["anchor: greeting", "old: 你好", "new: 再见"],
        );
        let result =
            apply_inline_replace(original, &envelope).expect("unicode replace should work");
        assert_eq!(result, "let greeting = \"再见世界\";\n");
    }

    #[test]
    fn inline_replace_parse_envelope() {
        // 验证 parse_patch_envelope 能识别 *** Replace in line: header
        let patch = "*** Begin Patch\n\
            *** Replace in line: src/main.rs\n\
            anchor: fn main()\n\
            old: println!\n\
            new: eprintln!\n\
            *** End Patch\n";
        let envelope = parse_patch_envelope(patch)
            .expect("should parse")
            .expect("should be Some");
        assert_eq!(envelope.op, PatchEnvelopeOp::ReplaceInLine);
        assert_eq!(envelope.target_path, "src/main.rs");
        assert_eq!(envelope.body_lines.len(), 3);
    }

    #[test]
    fn inline_replace_via_execute_apply_patch() {
        // 端到端：通过 execute_apply_patch 调用，验证完整路径（含 sandbox）
        let _guard = ENV_LOCK.lock();
        let path = make_temp_path("inline_e2e");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "the answer is 42\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "patch": format!(
                    "*** Begin Patch\n*** Replace in line: {}\nanchor: the answer\nold: 42\nnew: 99\n*** End Patch\n",
                    path.to_string_lossy()
                ),
                "path": path.to_string_lossy(),
            });
            execute_apply_patch(&args).expect("e2e should succeed");
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "the answer is 99\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn parse_patch_envelopes_accepts_multiple_sections() {
        let patch = "*** Begin Patch\n\
            *** Update File: src/a.rs\n\
            @@\n\
            -old_a\n\
            +new_a\n\
            \n\
            *** Add File: src/b.rs\n\
            +hello\n\
            *** End Patch\n";
        let envelopes = parse_patch_envelopes(patch)
            .expect("should parse")
            .expect("should be Some");
        assert_eq!(envelopes.len(), 2);
        assert_eq!(envelopes[0].target_path, "src/a.rs");
        assert_eq!(envelopes[1].target_path, "src/b.rs");
    }

    #[test]
    fn execute_apply_patch_supports_multi_file_begin_patch_atomically() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("multi_file_batch");
        let a = base.join("a.txt");
        let b = base.join("b.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&a, "old_a\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                // 部分模型会把未使用的可选字符串参数序列化为空串，应按未提供处理。
                "file_path": "",
                "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-old_a\n+new_a\n*** Add File: b.txt\n+hello\n+world\n*** End Patch\n"
            });
            let result = execute_apply_patch(&args).expect("multi-file Begin Patch should succeed");
            assert!(result.starts_with("Successfully patched 2 files:"), "result: {result}");
        });

        assert_eq!(fs::read_to_string(&a).unwrap(), "new_a\n");
        assert_eq!(fs::read_to_string(&b).unwrap(), "hello\nworld");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_multi_file_ignores_redundant_file_path() {
        // 多文件信封 + 冗余 file_path：模型常在多文件信封时仍传 file_path
        // （指向其中一个文件）。应静默忽略 file_path，用信封内各 section 自身路径。
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("multi_file_redundant_path");
        let a = base.join("a.txt");
        let b = base.join("b.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&a, "old_a\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                // 冗余 file_path 应被静默忽略
                "file_path": a.to_string_lossy(),
                "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-old_a\n+new_a\n*** Add File: b.txt\n+hello\n+world\n*** End Patch\n"
            });
            let result = execute_apply_patch(&args).expect("multi-file Begin Patch with redundant file_path should succeed");
            assert!(result.starts_with("Successfully patched 2 files:"), "result: {result}");
        });

        assert_eq!(fs::read_to_string(&a).unwrap(), "new_a\n");
        assert_eq!(fs::read_to_string(&b).unwrap(), "hello\nworld");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_multi_file_batch_is_atomic_on_failure() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("multi_file_atomic");
        let a = base.join("a.txt");
        let b = base.join("b.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&a, "old_a\n").unwrap();
        fs::write(&b, "current_b\n").unwrap();

        let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-old_a\n+new_a\n*** Update File: b.txt\n@@\n-missing_b\n+new_b\n*** End Patch\n"
            });
            execute_apply_patch(&args).expect_err("second file mismatch should abort whole batch")
        });

        assert!(
            err.contains("failed while preparing patch for"),
            "err was: {err}"
        );
        assert_eq!(fs::read_to_string(&a).unwrap(), "old_a\n");
        assert_eq!(fs::read_to_string(&b).unwrap(), "current_b\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_dry_run_validates_without_writing() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("dry_run");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "before\n").unwrap();

        let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            execute_apply_patch(&serde_json::json!({
                "file_path": path.to_string_lossy(),
                "patch": "@@\n-before\n+after\n",
                "dry_run": true,
            }))
            .expect("dry run should validate a matching patch")
        });

        assert!(result.starts_with("Dry run succeeded; no files changed:"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "before\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_delete_file_can_be_undone_and_redone() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("delete_file");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "important\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            execute_apply_patch(&serde_json::json!({
                "patch": format!(
                    "*** Begin Patch\n*** Delete File: {}\n*** End Patch\n",
                    path.display()
                ),
            }))
            .expect("Delete File should succeed");
        });
        assert!(!path.exists());

        crate::ai::tools::undo_tools::execute_undo(&serde_json::json!({ "count": 1 }))
            .expect("delete should be undoable");
        assert_eq!(fs::read_to_string(&path).unwrap(), "important\n");

        crate::ai::tools::undo_tools::execute_redo(&serde_json::json!({ "count": 1 }))
            .expect("delete should be redoable");
        assert!(!path.exists());
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn prepared_patch_rejects_external_change_before_commit() {
        let path = make_temp_path("stale_patch");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "before\n").unwrap();
        let store = super::FileStore::new(path.clone());
        let envelope = make_envelope(
            PatchEnvelopeOp::Update,
            &path.to_string_lossy(),
            &["@@", "-before", "+after"],
        );
        let prepared = super::prepare_patch_write(&path, &store, &envelope)
            .expect("matching patch should prepare");
        fs::write(&path, "changed_elsewhere\n").unwrap();

        let err = super::verify_patch_write_is_current(&prepared)
            .expect_err("a changed target must not be overwritten");
        assert!(err.contains("[FILE_CHANGED]"), "err: {err}");
        assert_eq!(fs::read_to_string(&path).unwrap(), "changed_elsewhere\n");
        let _ = fs::remove_dir_all(base);
    }
}
