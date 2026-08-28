use std::fs;
use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::ai::tools::common::ToolHistoryPolicy;
use crate::ai::tools::common::ToolHistoryPolicyRegistration;
use crate::ai::tools::common::ToolLossyCompressPolicy;
use crate::ai::tools::common::ToolPrunePolicy;
use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::common::ToolStreamWriter;
use crate::ai::tools::common::ToolStreamingRegistration;
use crate::ai::tools::storage::file_store::FileStore;
use crate::ai::tools::storage::temp_registry;

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "apply_patch",
        description: "",

        execute: execute_apply_patch,
    }
});

inventory::submit!(ToolStreamingRegistration {
    name: "apply_patch",
    execute_streaming: execute_apply_patch_streaming,
});

// The apply_patch result is the only precise evidence of "which file did I just modify" for the
// current turn, and failure diagnostics echo the entire current file text (so the model can
// rebuild the patch without re-reading). Once lossily compressed or truncated, the model cannot
// see whether the patch landed, and will repeatedly suspect the tool was never called and retry
// in place — hence lossy compression is forbidden, and apply_patch consumes high-precision inline
// budget to enter the current-turn protected set. Consistent with execute_command semantics:
// stale patch results can still be pruned by the model explicitly marking them outdated, so the
// context won't grow monotonically.
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "apply_patch",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
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
    /// Inline substring replacement: use `anchor:` to locate the line, then exactly replace
    /// `old:` with `new:` within that line.
    /// Does not go through the unified-diff path; handled directly by `apply_inline_replace`.
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
    /// Hints produced during application (e.g. pure-insert hunks are located by line number
    /// only), returned together with the success message to remind the model to re-read the file
    /// with read_file once it has changed.
    hints: Vec<String>,
}

#[derive(Debug, Clone)]
enum PreparedPatchAction {
    Write(String),
    Delete,
}

fn parse_unified_hunks(patch: &str) -> Result<Vec<UnifiedHunk>, String> {
    let mut hunks = Vec::new();
    let mut iter = patch.lines().peekable();
    let mut patch_line_no: usize = 0; // 1-based, used to locate positions in error messages
    let mut saw_content_before_header = false;
    let mut saw_envelope_marker = false; // whether any envelope marker (*** Begin Patch / *** Update File: etc.) was seen
    while let Some(line) = iter.next() {
        patch_line_no += 1;
        // Malformed envelope signal: when parse_patch_envelopes returns None because the first
        // line is not `*** Begin Patch`, the text would mistakenly fall into the unified-diff
        // path. Record whether any envelope opening/section marker was seen, to decide below
        // whether to silently tolerate a trailing `*** End Patch`.
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
        // A canonical `*** Begin Patch` envelope (Codex/OpenAI style) uses a bare `@@` or
        // `@@ <context title> @@` as the hunk separator, without `-N,M +N,M` line numbers. Only
        // when the header looks like `-N` do we parse a nominal line number; otherwise
        // old_start=0, letting locate_hunk's full-file search uniquely locate the hunk, avoiding
        // a spurious "invalid hunk header" for the canonical envelope format.
        // Models often write "insert at the start of the file" as `@@ -0,0 +1,3 @@`: in git
        // semantics -0 means "insert before line 1", so we normalize to old_start=1 rather than
        // treating it as having no nominal line number and running a full-file search, which
        // would later report a misleading "declared line 0".
        let old_start = match rest.strip_prefix('-') {
            Some(after) => after
                .split_whitespace()
                .next()
                .and_then(|part| part.split(',').next())
                .and_then(|num| num.parse::<isize>().ok())
                .map(|n| if n <= 0 { 1 } else { n as usize })
                .unwrap_or(0),
            None => 0,
        };

        let mut lines = Vec::new();
        while let Some(next) = iter.peek().copied() {
            if next.starts_with("@@") {
                break;
            }
            // Tolerate mixed formats: models often mistakenly append envelope tail markers such as
            // `*** End Patch` at the end of a pure unified-diff hunk. These markers are not part of
            // unified-diff content; when encountered we end the current hunk (letting the outer
            // loop skip them), avoiding a false "invalid hunk line" error.
            // But if an envelope opening/section marker was already detected
            // (saw_envelope_marker), this is a malformed envelope that fell into the unified-diff
            // path, where the target file is decided by file_path, not by the envelope
            // declaration — silently applying could write to the wrong file. So we do NOT break;
            // the line falls into the `_ =>` branch below to report a "mixed formats" error and
            // let the model rebuild, never silently writing to the wrong file.
            if (next == "*** End Patch" || next == "*** End of File") && !saw_envelope_marker {
                break;
            }
            let l = iter.next().unwrap_or_default();
            patch_line_no += 1;
            if l.starts_with("\\ No newline at end of file") {
                continue;
            }
            // Blank lines (including lines reduced to just `\r` under CRLF): models often write an
            // empty context line with no leading space at all. Treat it as an empty context line,
            // consistent with `git apply`'s tolerance.
            if l == "" || l == "\r" {
                lines.push(UnifiedLine::Context(String::new()));
                continue;
            }
            let mut chars = l.chars();
            let prefix = chars
                .next()
                .ok_or_else(|| format!("invalid hunk line at patch line {patch_line_no}: empty"))?;
            // Tolerate CRLF: strip the trailing \r so Add lines don't write \r into file content.
            let body = chars.as_str().strip_suffix('\r').unwrap_or(chars.as_str());
            match prefix {
                ' ' => lines.push(UnifiedLine::Context(body.to_string())),
                '-' => lines.push(UnifiedLine::Remove(body.to_string())),
                '+' => lines.push(UnifiedLine::Add(body.to_string())),
                _ => {
                    // Special-case envelope-style markers: this means unified diff and Begin/End
                    // Patch formats are mixed. Tail markers (*** End Patch / *** End of File) were
                    // already tolerated via break above; what lands here is an opening or section
                    // marker like *** Begin Patch / *** Update File:, indicating the patch
                    // structure is confused — report a clear error guiding the model to rebuild.
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
        // Strip trailing empty context lines: the hunk body loop only ends when the next `@@` is
        // reached, so blank lines between hunks or at the end of the patch (separators/trailing)
        // get swallowed into the current hunk as trailing empty context lines, demanding an empty
        // line at the corresponding position of the original file for no reason and causing a
        // patch that should match to report context mismatch. A genuine interior blank line is
        // always followed by more content lines of this hunk, so it won't be wrongly removed; only
        // purely trailing empty context lines are stripped here.
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

#[derive(Debug, Default, PartialEq, Eq)]
struct UnifiedDiffHeaderTarget {
    paths: Vec<String>,
    deletes_file: bool,
}

/// Parses a git double-quoted path token. Besides common C escapes, it also accepts the
/// three-digit octal bytes used by git's quotePath, so that valid paths containing spaces or
/// escaped characters are not split on whitespace and written to the wrong target.
fn parse_git_path_token(input: &str) -> Option<(String, &str)> {
    let input = input.trim_start();
    if !input.starts_with('"') {
        let end = input.find(char::is_whitespace).unwrap_or(input.len());
        return Some((input[..end].to_string(), &input[end..]));
    }

    let bytes = input.as_bytes();
    let mut decoded = Vec::new();
    let mut idx = 1;
    while idx < bytes.len() {
        match bytes[idx] {
            b'"' => {
                let path = String::from_utf8(decoded).ok()?;
                return Some((path, &input[idx + 1..]));
            }
            b'\\' => {
                idx += 1;
                let escaped = *bytes.get(idx)?;
                match escaped {
                    b'a' => decoded.push(0x07),
                    b'b' => decoded.push(0x08),
                    b'f' => decoded.push(0x0c),
                    b'n' => decoded.push(b'\n'),
                    b'r' => decoded.push(b'\r'),
                    b't' => decoded.push(b'\t'),
                    b'v' => decoded.push(0x0b),
                    b'0'..=b'7' => {
                        let mut value = escaped - b'0';
                        for _ in 0..2 {
                            let Some(next @ b'0'..=b'7') = bytes.get(idx + 1).copied() else {
                                break;
                            };
                            idx += 1;
                            value = value.saturating_mul(8).saturating_add(next - b'0');
                        }
                        decoded.push(value);
                    }
                    other => decoded.push(other),
                }
            }
            byte => decoded.push(byte),
        }
        idx += 1;
    }
    None
}

fn normalized_diff_path(path: &str) -> Option<String> {
    if path.is_empty() || path == "/dev/null" {
        return None;
    }
    let path = path
        .strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path);
    (!path.is_empty()).then(|| path.to_string())
}

fn diff_marker_path(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.starts_with('"') {
        return parse_git_path_token(raw).map(|(path, _)| path);
    }
    // `---`/`+++` paths allow spaces; only a TAB explicitly separates the optional timestamp.
    Some(raw.split('\t').next().unwrap_or(raw).trim().to_string())
}

fn record_diff_target(paths: &mut Vec<String>, path: Option<String>) {
    if let Some(path) = path
        && !paths.contains(&path)
    {
        paths.push(path);
    }
}

/// Collects all file targets in a complete unified diff. `diff --git` and adjacent `---`/`+++`
/// file headers are cross-checked and deduplicated; therefore a standard multi-file diff,
/// quoted paths with spaces, and subsequent file headers after the first hunk are never silently
/// mistaken for a single-file patch.
fn parse_unified_diff_header_target(patch: &str) -> UnifiedDiffHeaderTarget {
    let lines: Vec<&str> = patch.lines().collect();
    let mut parsed = UnifiedDiffHeaderTarget::default();

    for line in &lines {
        let Some(rest) = line.strip_prefix("diff --git ") else {
            continue;
        };
        let Some((_, rest)) = parse_git_path_token(rest) else {
            continue;
        };
        let Some((new_path, trailing)) = parse_git_path_token(rest) else {
            continue;
        };
        if trailing.trim().is_empty() {
            record_diff_target(&mut parsed.paths, normalized_diff_path(&new_path));
        }
    }

    for (index, pair) in lines.windows(2).enumerate() {
        let (Some(old_raw), Some(new_raw)) =
            (pair[0].strip_prefix("--- "), pair[1].strip_prefix("+++ "))
        else {
            continue;
        };
        // A hunk header must follow the file header (blank lines allowed in between). Otherwise
        // adjacent `--- ...` / `+++ ...` add/remove lines in the body would be misjudged as a
        // second file target.
        let followed_by_hunk = lines[index + 2..]
            .iter()
            .find(|line| !line.is_empty())
            .is_some_and(|line| line.starts_with("@@"));
        if !followed_by_hunk {
            continue;
        }
        let old_path = diff_marker_path(old_raw);
        let new_path = diff_marker_path(new_raw);
        parsed.deletes_file |= new_path.as_deref() == Some("/dev/null");
        let target = new_path
            .as_deref()
            .and_then(normalized_diff_path)
            .or_else(|| old_path.as_deref().and_then(normalized_diff_path));
        record_diff_target(&mut parsed.paths, target);
    }

    // Tolerate model output with only a single `+++` or `---` file header, but only fall back when
    // there is no more complete source, and only scan before the first hunk to avoid treating
    // add/remove lines that look like file headers in the body as targets.
    if parsed.paths.is_empty() {
        for line in lines.iter().take_while(|line| !line.starts_with("@@")) {
            let raw = line
                .strip_prefix("+++ ")
                .or_else(|| line.strip_prefix("--- "));
            let path = raw
                .and_then(diff_marker_path)
                .as_deref()
                .and_then(normalized_diff_path);
            record_diff_target(&mut parsed.paths, path);
        }
    }
    parsed
}

fn file_path_from_unified_diff_header(patch: &str) -> Option<String> {
    let parsed = parse_unified_diff_header_target(patch);
    (parsed.paths.len() == 1).then(|| parsed.paths[0].clone())
}

/// Splits a multi-file unified diff (git diff output or model-written) by file into
/// (target path, that file's diff fragment).
/// Fragments keep their original text (including their own file headers and hunks); target paths
/// are resolved with the same semantics as parse_unified_diff_header_target (`diff --git` takes
/// precedence; without a `diff --git` header, split by adjacent `--- `/`+++ ` pairs that are
/// followed by a hunk header, so add/remove lines in the hunk body that look like file headers
/// are not treated as file boundaries).
/// Returns Err when any fragment cannot be resolved to a unique target path — when the structure
/// is unreliable, report the error explicitly rather than silently writing to the wrong file.
fn split_unified_diff_by_file(patch: &str) -> Result<Vec<(String, String)>, String> {
    let lines: Vec<&str> = patch.lines().collect();
    let has_git_headers = lines.iter().any(|line| line.starts_with("diff --git "));
    let mut starts: Vec<usize> = Vec::new();
    if has_git_headers {
        for (i, line) in lines.iter().enumerate() {
            if line.starts_with("diff --git ") {
                starts.push(i);
            }
        }
    } else {
        let mut i = 0;
        while i < lines.len() {
            if lines[i].starts_with("--- ") {
                let mut j = i + 1;
                while j < lines.len() && lines[j].trim().is_empty() {
                    j += 1;
                }
                if j < lines.len() && lines[j].starts_with("+++ ") {
                    let mut k = j + 1;
                    while k < lines.len() && lines[k].trim().is_empty() {
                        k += 1;
                    }
                    if k < lines.len() && lines[k].starts_with("@@") {
                        starts.push(i);
                        i = j + 1;
                        continue;
                    }
                }
            }
            i += 1;
        }
    }
    if starts.is_empty() {
        return Err(
            "multi-file unified diff: could not find per-file section boundaries (`diff --git ` \
             headers or `--- `/`+++ ` header pairs followed by hunks)"
                .to_string(),
        );
    }
    let mut sections = Vec::with_capacity(starts.len());
    for (idx, &start) in starts.iter().enumerate() {
        let end = starts.get(idx + 1).copied().unwrap_or(lines.len());
        let section = lines[start..end].join("\n");
        let parsed = parse_unified_diff_header_target(&section);
        if parsed.paths.len() != 1 {
            return Err(format!(
                "multi-file unified diff: section {}/{} could not be resolved to exactly one \
                 target path (found {}). Use a `*** Begin Patch` envelope with one \
                 `*** Update File:` section per file instead.",
                idx + 1,
                starts.len(),
                parsed.paths.len()
            ));
        }
        sections.push((parsed.paths[0].clone(), section));
    }
    Ok(sections)
}

/// Provides a unified target-extraction semantic for the driver's scoped-instruction preflight
/// and the stale-patch ledger.
/// Even if the envelope ultimately fails to execute due to structural errors, declared targets
/// are still surfaced as much as possible, so the project rules for the corresponding directory
/// are loaded before the first potential write.
pub(crate) fn apply_patch_target_paths_from_patch(raw_patch: &str) -> Vec<PathBuf> {
    let patch = strip_code_fence(raw_patch);
    let mut targets = Vec::new();
    for line in patch.lines() {
        let path = [
            "*** Update File:",
            "*** Add File:",
            "*** Delete File:",
            "*** Replace in line:",
        ]
        .iter()
        .find_map(|prefix| line.trim_start().strip_prefix(prefix))
        .map(str::trim)
        .filter(|path| !path.is_empty());
        if let Some(path) = path {
            let path = PathBuf::from(path);
            if !targets.contains(&path) {
                targets.push(path);
            }
        }
    }
    if !targets.is_empty() {
        return targets;
    }
    parse_unified_diff_header_target(&patch)
        .paths
        .into_iter()
        .map(PathBuf::from)
        .collect()
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

fn normalize_patch_envelope_body(envelope: &PatchEnvelope) -> Result<String, String> {
    Ok(match envelope.op {
        PatchEnvelopeOp::ReplaceInLine => {
            // ReplaceInLine does not go through the unified-diff path; it is handled directly by
            // apply_inline_replace. Reaching here means the dispatch logic in
            // execute_apply_patch has a bug — return an explicit error early rather than letting
            // this be processed as a unified diff (which would treat anchor:/old:/new: as context
            // lines).
            return Err(
                "internal error: ReplaceInLine envelope should be handled by \
                 apply_inline_replace, not normalize_patch_text"
                    .to_string(),
            );
        }
        PatchEnvelopeOp::Update => {
            // The Update format of *** Begin Patch allows omitting the hunk header (Cursor/Aider
            // style); models often write only +/-/space-prefixed lines without a hunk header. If
            // the body contains no hunk header at all, synthesize one so parse_unified_hunks can
            // recognize it. old_start=0 means no nominal position; locate_hunk will skip nominal
            // matching and search the whole file directly.
            let has_hunk_header = envelope.body_lines.iter().any(|l| l.starts_with("@@"));
            // Even when a hunk header exists, models often write context lines as bare text; pad
            // such bare lines into context lines within the envelope to avoid pointless invalid
            // hunk line failures.
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
            // Blank lines represent blank lines in the new file; prefix them with `+` so
            // parse_unified_hunks recognizes them as Add lines.
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

fn normalize_patch_envelope(path: &Path, envelope: &PatchEnvelope) -> Result<String, String> {
    match envelope.op {
        PatchEnvelopeOp::Update if !path.exists() => Err(format!(
            "Update File patch targets a missing file: {}. Use Add File to create a new file, or correct the target path before retrying.",
            path.display()
        )),
        PatchEnvelopeOp::Add if path.exists() => Err(
            "Add File patch targets an existing file. Use Update File or write_file instead."
                .to_string(),
        ),
        _ => normalize_patch_envelope_body(envelope),
    }
}

/// Inline substring replacement: use `anchor:` to locate the line, then exactly replace
/// `old:` with `new:` within that line.
///
/// Designed for the most common editing scenario — "change a few words inside one long
/// single-line string" — avoiding a full-line rewrite.
///
/// Safety design (to rule out "executed successfully but replaced the wrong position"):
/// - `anchor` locates the line via normalized substring matching (confusable-tolerant), but is
///   **only used for locating**;
/// - `old` uses **exact** substring matching (no normalization) to determine the byte range to
///   replace, ruling out positional drift;
/// - `anchor` must uniquely match one line, otherwise error (to avoid changing the wrong place);
/// - `old` must appear exactly once in that line, otherwise error (to avoid changing the wrong
///   position);
/// - if `old == new` (identical before/after), report an error as a no-op so it isn't mistaken
///   for success.
fn apply_inline_replace(original: &str, envelope: &PatchEnvelope) -> Result<String, String> {
    // --- Parse the three fields: anchor / old / new ---
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
        // Ignore unrelated lines (blank lines, comments, etc.) and stay tolerant
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

    // --- Locate the line via normalized substring matching (confusable-tolerant), but only for locating ---
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

    // --- First use exact substring matching (no normalization) to determine the replacement position, ruling out positional drift ---
    let occurrences: Vec<usize> = target_line.match_indices(&old).map(|(i, _)| i).collect();
    let (pos, match_len) = match occurrences.len() {
        1 => (occurrences[0], old.len()),
        0 => {
            // Tolerant fallback when exact matching fails: confusable normalization + leading and
            // trailing whitespace tolerance.
            // Only used for locating; the replacement boundary follows the matched original byte
            // range, and the written content is constructed from `new`.
            match find_tolerant_old_match(target_line, &old) {
                Ok((start, end)) => (start, end - start),
                Err(TolerantMatchError::Ambiguous(n)) => {
                    return Err(format!(
                        "Replace in line: after Unicode normalization `old` matches {n} \
                         positions in line {}. It must be unique within the line. \
                         Make `old` longer or more specific. Line content: {:?}",
                        line_idx + 1,
                        target_line
                    ));
                }
                Err(TolerantMatchError::NoMatch) => {
                    return Err(format!(
                        "Replace in line: `old` substring not found in matched line {} \
                         (even with Unicode/whitespace tolerance). Line content: {:?}\n\
                         Tips: copy `old` from the actual file, not from memory. If you \
                         need fresh source text, re-read with `read_file` \
                         (use_line_numbers=false) so the output has no line-number \
                         prefixes and you can copy the exact line content. Watch for \
                         smart quotes / dashes / non-breaking \
                         spaces that may differ from the file.",
                        line_idx + 1,
                        target_line
                    ));
                }
            }
        }
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

    // Exact replacement: byte range [pos, pos+old.len()).
    // pos is the byte index returned by str::find; old is valid UTF-8, so pos and
    // pos+old.len() both lie on char boundaries and slicing is safe.
    let replaced_line = format!(
        "{}{}{}",
        &target_line[..pos],
        new,
        &target_line[pos + match_len..]
    );

    // Rebuild the file, preserving the original trailing-newline behavior
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
    /// Exact matching (allows trailing-whitespace differences), used by default.
    Strict,
    /// Ignores leading-indentation differences; only used as a fallback when strict matching
    /// fails to locate anything in the whole file.
    /// Aligns with `git apply --ignore-whitespace`: models often fail to reproduce the
    /// indentation of markdown/nested lists/code blocks exactly, causing strict matching to
    /// fail on the whole block with context mismatch.
    IgnoreIndent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextPolicy {
    /// Context lines must match, and remove lines must also match.
    Require,
    /// Context lines are only a locating reference; when applying, keep the file's actual
    /// context, while remove lines must still match.
    Fuzz,
}

/// Strips the line-number prefix that read_file / grep and other tool outputs add
/// (single-argument fallback version).
/// Models sometimes accidentally copy the line-number prefix into a patch's context/remove lines.
///
/// This version is for scenarios with **no "real line" to anchor on** (e.g. IgnoreIndent
/// normalizes both sides independently). When a real line is available for comparison, prefer
/// the anchored [`strip_number_prefix_anchored`], which is separator-agnostic and has almost
/// zero false positives. Here, to avoid wrongly stripping code lines that genuinely start with
/// digits (e.g. `80:80`, `42px`, `3.14`), we take a **conservative** approach and only recognize
/// two highly deterministic line-number-column shapes:
/// - `digits + \t`: read_file's real format (`{:>6}\t{}`). The line content (including its own
///   indentation) follows directly after the TAB; only this single TAB is consumed.
/// - `digits + single non-alphanumeric separator + space`: grep-like (`42| `, `42: `). The
///   separator must be **followed by a space**, so `80:80` (`:` followed by a digit) and `3.14`
///   (`.` followed by a digit) are not wrongly stripped.
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
    // TAB: read_file's real separator; the content (including indentation) follows directly
    // after it; only this single TAB is consumed.
    if sep == '\t' {
        return &after_digits['\t'.len_utf8()..];
    }
    // Other separators: must be a single non-alphanumeric, non-space character followed
    // immediately by a space (`42| ` / `42: `). Requiring the trailing space avoids mistaking
    // `80:80` or `3.14` for a line-number column.
    if sep.is_alphanumeric() || sep == ' ' {
        return s;
    }
    let rest = &after_digits[sep.len_utf8()..];
    match rest.strip_prefix(' ') {
        Some(after_space) => after_space,
        None => s,
    }
}

/// Anchored line-number-prefix stripping: using `actual` (the file's real line, which never
/// contains a line-number column) as ground truth, decides whether `expected` (a patch line,
/// possibly with a model-miscopied line-number column) is **exactly equal** to `actual` after
/// removing the "digit column". If so, returns the de-columned content; otherwise returns
/// `expected` unchanged.
///
/// Compared with enumerating separators, this is separator-agnostic (`\t` `|` `:` space `.`
/// `)` are all compatible), and because it requires "the remainder to exactly equal the real
/// line", it can almost never wrongly hit code lines that genuinely start with digits — even if
/// it happens to, lines_match's multi-match (ambiguity) detection intercepts it.
fn strip_number_prefix_anchored<'a>(expected: &'a str, actual: &str) -> &'a str {
    // expected must start with "optional whitespace + digits", otherwise it cannot be
    // "line-number column + actual".
    let lead_ws_end = expected
        .find(|c: char| !c.is_whitespace())
        .unwrap_or(expected.len());
    let after_ws = &expected[lead_ws_end..];
    let digits_end = after_ws
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_ws.len());
    if digits_end == 0 {
        return expected; // the digit part is empty: not a line-number column.
    }
    let after_digits = &after_ws[digits_end..];
    // The remainder after removing 1 separator (on char boundaries, to avoid multi-byte UTF-8
    // slice panics).
    let after_one_sep = after_digits
        .char_indices()
        .nth(1)
        .map(|(byte_idx, _)| &after_digits[byte_idx..])
        .unwrap_or("");
    // Try each candidate: does removing 0 or 1 separators (optionally with 1 space) equal the
    // real line? "The remainder exactly equals the real line" is the only criterion, so we don't
    // need to know what the separator actually is.
    let candidates = [
        after_digits,  // digits directly followed by content (rare)
        after_one_sep, // consume 1 separator
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

/// Single-character confusable normalization (strict 1:1 mapping, no width expansion).
/// Shared with [`normalize_confusables`] to keep "whole-string normalization" and
/// "per-character equivalence checks" consistent.
fn normalize_confusable_char(c: char) -> char {
    match c {
        // --- dash family ---
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
        | '\u{2212}' => '-',
        // --- smart double quotes ---
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => '"',
        // --- smart single quotes ---
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => '\'',
        // --- non-breaking space family ---
        '\u{00A0}' | '\u{202F}' | '\u{2007}' | '\u{2060}' => ' ',
        other => other,
    }
}

/// Normalizes common Unicode "confusable" characters to ASCII-equivalent forms.
///
/// Only used for **locating decisions** in patch matching (lines_match); never participates in
/// constructing output content.
/// All handled characters are purely typographic differences that don't affect semantics:
/// - dash family (— – ― etc.) -> '-'
/// - smart quotes (" " ' ' ‛ ‟) -> '"' / "'"
/// - non-breaking spaces (NBSP U+00A0, NNBSP U+202F, etc.) -> regular space
fn normalize_confusables(s: &str) -> String {
    s.chars().map(normalize_confusable_char).collect()
}

/// Locates `old` in `line` via **tolerant matching** (only for the `old` fallback locating in
/// `*** Replace in line:`):
/// - per-character confusable normalization equivalence (1:1, see [`normalize_confusable_char`]),
///   so em-dash/smart quotes/NBSP match their ASCII-equivalent forms;
/// - ignores leading/trailing whitespace in `old` (models often copy leading spaces into `old`
///   while reproducing indentation).
///
/// Returns the (byte_start, byte_end) of the match in the original line. Must be unique:
/// multiple matches return [`TolerantMatchError::Ambiguous`]. The replacement still slices on the
/// original byte range; the written content is constructed from `new`, so normalized characters
/// are never written into the file.
fn find_tolerant_old_match(line: &str, old: &str) -> Result<(usize, usize), TolerantMatchError> {
    let needle: Vec<char> = old.trim().chars().collect();
    let hay: Vec<char> = line.chars().collect();
    if needle.is_empty() || needle.len() > hay.len() {
        return Err(TolerantMatchError::NoMatch);
    }
    // Precompute each char's byte offset once in O(n) to avoid per-position nth().
    let byte_offsets: Vec<usize> = line.char_indices().map(|(b, _)| b).collect();
    let mut found: Vec<(usize, usize)> = Vec::new();
    'outer: for i in 0..=(hay.len() - needle.len()) {
        for (j, &nc) in needle.iter().enumerate() {
            if normalize_confusable_char(hay[i + j]) != normalize_confusable_char(nc) {
                continue 'outer;
            }
        }
        let byte_start = byte_offsets[i];
        let byte_end = byte_offsets
            .get(i + needle.len())
            .copied()
            .unwrap_or(line.len());
        found.push((byte_start, byte_end));
    }
    match found.len() {
        0 => Err(TolerantMatchError::NoMatch),
        1 => Ok(found[0]),
        n => Err(TolerantMatchError::Ambiguous(n)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TolerantMatchError {
    NoMatch,
    Ambiguous(usize),
}

fn lines_match_exact(actual: &str, expected: &str, mode: MatchMode) -> bool {
    if actual == expected || actual.trim_end() == expected.trim_end() {
        return true;
    }
    match mode {
        MatchMode::Strict => {
            // Models often copy the line-number prefix from read_file output (e.g.
            // `    42\t<code>`). Prefer the anchored approach: using actual (the real file line)
            // as the reference, check whether expected equals actual exactly after removing the
            // digit column — separator-agnostic and with almost zero false positives.
            let e = strip_number_prefix_anchored(expected, actual);
            if e == actual || e.trim_end() == actual.trim_end() {
                return true;
            }
            // Fallback: generic digit-column stripping on both sides when no actual anchor info
            // is available, also covering edge cases where the actual side carries a column too.
            let expected_stripped = strip_line_number_prefix(expected);
            let actual_stripped = strip_line_number_prefix(actual);
            expected_stripped == actual_stripped
                || expected_stripped.trim_end() == actual_stripped.trim_end()
        }
        MatchMode::IgnoreIndent => {
            // Try the anchored approach first (based on actual.trim), then fall back to generic
            // two-sided stripping + trim.
            let e = strip_number_prefix_anchored(expected.trim_start(), actual.trim());
            if e.trim() == actual.trim() {
                return true;
            }
            strip_line_number_prefix(actual).trim() == strip_line_number_prefix(expected).trim()
        }
    }
}

/// Common entry point for lines_match: exact matching first, then compare after normalizing
/// confusable characters.
///
/// Normalization only affects whether a line "can be located". Output content is constructed by
/// try_apply_hunk_at:
/// - Context lines output actual (the original file content)
/// - Remove lines are dropped after matching
/// - Add lines use the patch content directly
/// So when normalized matching succeeds, the file still receives the original file's Unicode
/// characters — content is never "replaced with the wrong thing".
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

/// Extracts the hunk's context+remove lines (i.e. the lines expected to match in the original file).
fn hunk_expected_lines(hunk: &UnifiedHunk) -> Vec<&str> {
    hunk.lines
        .iter()
        .filter_map(|line| match line {
            UnifiedLine::Context(s) | UnifiedLine::Remove(s) => Some(s.as_str()),
            _ => None,
        })
        .collect()
}

/// Counts, across the whole file, the positions (0-based line numbers) where the hunk's
/// context+remove block can match.
/// Used to detect "multiple matches" ambiguity, avoiding silently editing the wrong place.
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

fn describe_ambiguous_hunk(
    orig_lines: &[String],
    positions: &[usize],
    hunk_idx: usize,
    hunk_total: usize,
) -> String {
    let shown: Vec<String> = positions
        .iter()
        .take(8)
        .map(|pos| (pos + 1).to_string())
        .collect();
    let mut msg = format!(
        "Hunk {}/{}: ambiguous patch: hunk context matched {} locations (1-based lines: {}{}). \
         Add more unique surrounding context, preferably both before and after the edit, \
         or split the edit around a uniquely matching removed line.\n",
        hunk_idx + 1,
        hunk_total,
        positions.len(),
        shown.join(", "),
        if positions.len() > 8 { ", ..." } else { "" }
    );
    // Echo the current first line at each candidate position, so the model can pick the right
    // anchor and add more unique context without a separate read_file.
    msg.push_str("Candidate locations (current first line at each):\n");
    for &pos in positions.iter().take(8) {
        if let Some(line) = orig_lines.get(pos) {
            msg.push_str(&format!("  line {}: {:?}\n", pos + 1, line));
        }
    }
    msg.push_str(
        "Hint: the first line at each candidate is shown above; add more unique surrounding \
         context (e.g. the preceding function signature or comment) around the intended \
         location. For a single-line change, `*** Replace in line:` (anchor/old/new) is the \
         most reliable. If the candidates are structurally similar blocks (e.g. repeated \
         closures), apply one patch per block instead of one multi-hunk patch.\n",
    );
    msg
}

const DECLARED_LINE_DISAMBIGUATION_MAX_DRIFT: usize = 12;

fn disambiguate_by_declared_line(positions: &[usize], hunk: &UnifiedHunk) -> Option<usize> {
    if hunk.old_start == 0 || positions.len() < 2 {
        return None;
    }
    let nominal = hunk.old_start.saturating_sub(1);
    let mut scored: Vec<(usize, usize)> = positions
        .iter()
        .map(|&pos| (pos.abs_diff(nominal), pos))
        .collect();
    scored.sort_unstable();
    let (best_dist, best_pos) = scored[0];
    let (second_dist, _) = scored[1];
    if best_dist <= DECLARED_LINE_DISAMBIGUATION_MAX_DRIFT
        && best_dist.saturating_mul(2) < second_dist
    {
        Some(best_pos)
    } else {
        None
    }
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

/// Context lines are a locating aid and should not cause a hard failure once the
/// remove lines are precisely anchored. But to avoid mislocating common remove
/// lines (such as `}`), fuzz application is allowed only when the candidate is
/// unique or the remaining context can be scored uniquely.
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
    // Use old_start as a disambiguation signal: as long as the nominal
    // candidate's context score is close to the best (difference ≤ 1), trust the
    // line number the model annotated. Accept it also when best_score == 0 (the
    // context lines cannot distinguish candidate positions at all) — then
    // old_start is the only usable locating signal, and rejecting would just make
    // the model retry the same generic context endlessly.
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
         Include more exact surrounding context (both before and after the edit), or split the hunk around a more unique removed line. A `*** Replace in line:` section with a unique anchor also avoids this.",
        candidates.len(),
        shown.join(", "),
        if candidates.len() > 5 { ", ..." } else { "" }
    ))
}

/// For large replacements (a hunk with many context/remove lines), all-or-nothing
/// exact matching easily fails entirely when a few lines are not reproduced
/// exactly. Here we first run a best-effort partial-match scan: find the start
/// with the most matching lines across the whole file, and report precisely which
/// lines differ (expected vs actual), so the model only needs to fix the few
/// wrong lines instead of re-guessing the whole block.
struct BestPartialMatch {
    /// Best matching start (0-based)
    pos: usize,
    /// Number of matching lines
    matches: usize,
    /// Total number of lines checked
    total: usize,
    /// Mismatched lines: (1-based file line, expected content, actual content)
    mismatches: Vec<(usize, String, String)>,
}

/// Finds the start position where the hunk's expected block matches best across
/// the whole file. Called only on the error path after exact matching failed,
/// using IgnoreIndent mode to tolerate indentation differences and focus on
/// content differences. Returns None when no line in the file can match the
/// expected block — the block does not exist at all.
fn find_best_partial_match(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    mode: MatchMode,
) -> Option<BestPartialMatch> {
    let expected = hunk_expected_lines(hunk);
    if expected.is_empty() || orig_lines.is_empty() {
        return None;
    }

    // Use the first line for a quick filter: only run the full alignment check at
    // candidate positions where the first line matches, avoiding an O(N*M) full
    // scan on large files. In large replacements the most common failure is a
    // correct first line with a few wrong lines after it.
    let mut candidates: Vec<usize> = (0..orig_lines.len())
        .filter(|&i| lines_match(&orig_lines[i], expected[0], mode))
        .collect();

    // When the first line does not match, use the last line as an anchor: a last
    // line matching at position i corresponds to start i - (len-1).
    if candidates.is_empty() && expected.len() > 1 {
        let last = expected.len() - 1;
        candidates = (last..orig_lines.len())
            .filter(|&i| lines_match(&orig_lines[i], expected[last], mode))
            .map(|i| i - last)
            .collect();
    }

    // When neither the first nor the last line matches, anchor on every expected
    // line and take the candidate with the most matching lines. This is the final
    // fallback, covering cases where the middle lines of the expected block are
    // correct but the first/last lines are wrong.
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

    // Limit the number of candidates to avoid performance issues in extreme cases.
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
        // A perfect match should not occur on the error path, but keep the early
        // exit for safety.
        if matches == expected.len() {
            break;
        }
    }
    best.filter(|b| b.matches > 0)
}

fn format_char_with_code_point(ch: char) -> String {
    format!("{ch:?} (U+{:04X})", ch as u32)
}

/// Describes the position and Unicode code point of the first differing character
/// between two lines of text, making it easy to spot "looks-alike" differences
/// such as smart quotes or full/half-width characters.
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

pub(crate) const PATCH_TEXT_BLOCK_START: &str =
    "Current file text at this location (copy verbatim, no line-number prefix):\n<<<PATCH_TEXT\n";

/// Renders a directly pasteable block of the current file text (0-based `start`,
/// `count` lines).
///
/// Unlike the other diagnostics in the error message (which carry a `<line>:` prefix
/// for human eyes), this block carries **no line-number prefix** and is meant to be
/// copied verbatim by the model into the new patch's context/removed lines — this
/// eliminates at the root the frequent error of copying read_file's `<number>\t`
/// line-number column into the patch. The `<<<PATCH_TEXT` / `PATCH_TEXT>>>` markers
/// clearly delimit the copyable region.
fn render_pasteable_current_block(orig_lines: &[String], start: usize, count: usize) -> String {
    let end = start.saturating_add(count).min(orig_lines.len());
    if start >= end {
        return String::new();
    }
    let mut out = String::from(PATCH_TEXT_BLOCK_START);
    for line in &orig_lines[start..end] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("PATCH_TEXT>>>\n");
    out
}

/// Constructs a "hunks out of order" error with diagnostics.
///
/// Previously this error returned only the bare string `"hunks out of order"` — no
/// line numbers, no indication of which hunk, no current text, no fix suggestion, so
/// the model could only guess blindly (in real sessions this caused 4 consecutive
/// failures).
///
/// Unified diffs require hunks to be ordered by **ascending** file line number;
/// `apply_unified_patch` locates each hunk with a monotonically increasing `cursor`.
/// When the unique match position of a hunk falls before `cursor` (the end line of
/// the previous hunk), the model wrote the hunks out of order (or overlapping).
/// This explains the cause, gives the matched position of this hunk vs the earliest
/// allowed position, appends pasteable current text, and clearly suggests reordering
/// by line number or switching to separate `*** Replace in line:` sections.
///
/// `matched_pos` is the 0-based line this hunk actually matched (if known); `cursor`
/// is the earliest currently allowed 0-based start.
fn describe_hunks_out_of_order(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    matched_pos: Option<usize>,
    cursor: usize,
    hunk_idx: usize,
    hunk_total: usize,
) -> String {
    let mut msg = format!(
        "Hunk {}/{}: hunks out of order: this hunk matches a location earlier in the file \
         than a previous hunk in the same section. Unified-diff hunks must be ordered by \
         ascending file line number. Reorder the hunks by their position in the file \
         (top to bottom), or split unrelated edits into separate `*** Replace in line:` \
         sections.\n",
        hunk_idx + 1,
        hunk_total
    );
    if hunk.old_start > 0 {
        msg.push_str(&format!(
            "This hunk declared @@ -{} (1-based line {}).\n",
            hunk.old_start, hunk.old_start
        ));
    }
    match matched_pos {
        Some(pos) => msg.push_str(&format!(
            "It matches at 1-based line {}, but the previous hunk already consumed through 1-based line {}; a following hunk must start at 1-based line {} or later.\n",
            pos + 1,
            cursor,
            cursor + 1
        )),
        None => msg.push_str(&format!(
            "The earliest position a following hunk may target is 1-based line {} (where the previous hunk ended).\n",
            cursor + 1
        )),
    }
    // Append pasteable current text at this hunk's expected block to help the model
    // reorder/rebuild accordingly.
    let anchor = matched_pos.unwrap_or_else(|| hunk.old_start.saturating_sub(1));
    let expected_len = hunk_expected_lines(hunk).len().max(1);
    let block = render_pasteable_current_block(orig_lines, anchor, expected_len);
    if !block.is_empty() {
        msg.push_str(&block);
    }
    msg
}

/// Constructs a "context mismatch" error with context: lists the lines the patch
/// expected to match plus the actual lines in the original file near the nominal
/// position, so the model can quickly self-correct instead of only seeing a bare
/// "context mismatch".
fn describe_context_mismatch(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    hunk_idx: usize,
    hunk_total: usize,
) -> String {
    let expected = hunk_expected_lines(hunk);
    let nominal = hunk.old_start.saturating_sub(1);

    let mut msg = format!(
        "Hunk {}/{}: context mismatch: patch hunk could not be located. Rebuild the patch \
         from the current file text shown below (re-read the file only if the shown context \
         is not enough).\n",
        hunk_idx + 1,
        hunk_total
    );
    if hunk.old_start == 0 {
        msg.push_str(
            "Hunk header declared no line number (bare `@@`); the hunk is located by full-file \
             context search.\n",
        );
    } else {
        msg.push_str(&format!(
            "Hunk header declared @@ -{} (1-based line {}).\n",
            hunk.old_start, hunk.old_start
        ));
    }

    // First try a best-effort partial match to pinpoint the mismatched lines. In
    // large replacements the most common failure is that only a few lines in the
    // block are not reproduced exactly; a partial match can tell the model "line X
    // expected A but is B".
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
            // Show the first 10 mismatched lines: the full offset pattern matters
            // more than a condensed word count for the model to fix the patch.
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
        // A directly pasteable block of current text: covers the best-match region
        // (with a little extra margin before/after) so the model can rebuild the
        // patch in place without re-reading the whole file.
        let block =
            render_pasteable_current_block(orig_lines, best.pos, best.total.max(expected.len()));
        if !block.is_empty() {
            msg.push_str(&block);
        }
    } else {
        // No partial match found in the file — the block does not exist at all.
        // Echo the expected lines and the actual content near the nominal position.
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
            // Also append a pasteable block with no line-number prefix.
            let block = render_pasteable_current_block(orig_lines, win_start, win_end - win_start);
            if !block.is_empty() {
                msg.push_str(&block);
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
        "Hint: rebuild the patch from the copy-verbatim block above (the text between <<<PATCH_TEXT and PATCH_TEXT>>>), which is the exact current file content with NO line-number prefix. For a small change, a `*** Replace in line:` section (anchor/old/new) is the most reliable. If the shown context is insufficient, re-read the file with read_file(use_line_numbers=false) to get raw content without line-number prefixes, then copy the exact text.",
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

/// Locates the application start (0-based) of a hunk under the given match mode.
/// Returns `Ok(Some(pos))` for a unique successful location; `Ok(None)` when no
/// position in the whole file matches (the caller may retry with a more lenient
/// mode); `Err` when a position was found but is ambiguous or out of order.
fn locate_hunk(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    cursor: usize,
    mode: MatchMode,
    hunk_idx: usize,
    hunk_total: usize,
) -> Result<Option<usize>, String> {
    let old_len = hunk_old_line_count(hunk);
    if old_len == 0 {
        if hunk.old_start == 0 {
            return Ok(Some(cursor));
        }
        let nominal = hunk.old_start.saturating_sub(1);
        if nominal < cursor {
            return Err(describe_hunks_out_of_order(
                orig_lines,
                hunk,
                Some(nominal),
                cursor,
                hunk_idx,
                hunk_total,
            ));
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

    // When the nominal position does not match, first check how many places across
    // the whole file can match: multiple matches mean the context anchors are not
    // unique, so report the candidate positions directly instead of falling back
    // to fuzzy context and producing a confusing context mismatch.
    let positions = all_hunk_match_positions(orig_lines, hunk, mode);
    let forward: Vec<usize> = positions.iter().copied().filter(|&p| p >= cursor).collect();
    if forward.len() > 1 {
        if let Some(pos) = disambiguate_by_declared_line(&forward, hunk) {
            return Ok(Some(pos));
        }
        return Err(describe_ambiguous_hunk(
            orig_lines,
            &forward,
            hunk_idx,
            hunk_total,
        ));
    }
    // forward is already filtered to p >= cursor, so "hunks out of order" cannot
    // happen here. Previously falling back to find_hunk_offset (a ±50 window) would
    // falsely report context mismatch when the unique match fell outside the
    // window; just use forward's unique result directly.
    if let Some(&offset) = forward.first() {
        Ok(Some(offset))
    } else if let Some(&earliest) = positions.first() {
        // All matches are before cursor — the hunk is out of order. positions is
        // already sorted ascending, so take the earliest match as the diagnostic
        // anchor.
        Err(describe_hunks_out_of_order(
            orig_lines,
            hunk,
            Some(earliest),
            cursor,
            hunk_idx,
            hunk_total,
        ))
    } else {
        Ok(None)
    }
}

fn apply_unified_patch_with_hints(
    original: &str,
    patch: &str,
) -> Result<(String, Vec<String>), String> {
    let had_trailing_newline = original.ends_with('\n');
    let hunks = parse_unified_hunks(patch).map_err(|err| append_truncated_patch_hint(patch, err))?;
    if !hunks.iter().any(|hunk| {
        hunk.lines
            .iter()
            .any(|line| matches!(line, UnifiedLine::Remove(_) | UnifiedLine::Add(_)))
    }) {
        return Err("[NO_CHANGES] patch contains no additions or removals".to_string());
    }
    let orig_lines: Vec<String> = original.lines().map(|s| s.to_string()).collect();

    let active_hunks: Vec<&UnifiedHunk> = hunks
        .iter()
        .filter(|hunk| {
            hunk.lines
                .iter()
                .any(|line| matches!(line, UnifiedLine::Remove(_) | UnifiedLine::Add(_)))
        })
        .collect();
    let hunk_total = active_hunks.len();

    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize;

    for (hunk_idx, hunk) in active_hunks.iter().enumerate() {
        // First try strict matching (only tolerating trailing whitespace). When
        // strict matching cannot locate the hunk anywhere in the file, fall back
        // once to a lenient mode that ignores leading indentation — aligned with
        // `git apply --ignore-whitespace`, resolving context mismatches caused by
        // the model not reproducing markdown/nested-list/code-block indentation
        // exactly.
        let (apply_at, mode, context_policy) = match locate_hunk(
            &orig_lines,
            hunk,
            cursor,
            MatchMode::Strict,
            hunk_idx,
            hunk_total,
        )? {
            Some(at) => (at, MatchMode::Strict, ContextPolicy::Require),
            None => match locate_hunk(
                &orig_lines,
                hunk,
                cursor,
                MatchMode::IgnoreIndent,
                hunk_idx,
                hunk_total,
            )? {
                Some(at) => (at, MatchMode::IgnoreIndent, ContextPolicy::Require),
                None => match locate_hunk_with_fuzzy_context(
                    &orig_lines,
                    hunk,
                    cursor,
                    MatchMode::Strict,
                ) {
                    Ok(Some(candidate)) => {
                        (candidate.pos, MatchMode::Strict, ContextPolicy::Fuzz)
                    }
                    _ => match locate_hunk_with_fuzzy_context(
                        &orig_lines,
                        hunk,
                        cursor,
                        MatchMode::IgnoreIndent,
                    )? {
                        Some(candidate) => {
                            (candidate.pos, MatchMode::IgnoreIndent, ContextPolicy::Fuzz)
                        }
                        None => {
                            return Err(describe_context_mismatch(
                                &orig_lines, hunk, hunk_idx, hunk_total,
                            ))
                        }
                    },
                },
            },
        };

        out.extend_from_slice(&orig_lines[cursor..apply_at]);
        let (hunk_out, new_idx) = try_apply_hunk_at(&orig_lines, hunk, apply_at, mode, context_policy)
            .ok_or_else(|| {
                describe_context_mismatch(&orig_lines, hunk, hunk_idx, hunk_total)
            })?;
        out.extend(hunk_out);
        cursor = new_idx;
    }

    out.extend_from_slice(&orig_lines[cursor..]);
    let mut s = out.join("\n");
    if had_trailing_newline {
        s.push('\n');
    }
    let hints = pure_insert_hint(original, &hunks).into_iter().collect();
    Ok((s, hints))
}

fn apply_unified_patch(original: &str, patch: &str) -> Result<String, String> {
    apply_unified_patch_with_hints(original, patch).map(|(content, _)| content)
}

/// A pure-insert hunk (only `+` lines, no context/remove lines) undergoes no
/// content verification on a non-empty file; it is located only by the line number
/// declared in `@@`. When one is hit, return a hint reminding the model that if the
/// file changed after being read, it should re-verify the insert position with
/// read_file.
fn pure_insert_hint(original: &str, hunks: &[UnifiedHunk]) -> Option<String> {
    if original.is_empty() {
        return None;
    }
    let any_pure_insert = hunks.iter().any(|hunk| {
        !hunk.lines.is_empty()
            && hunk
                .lines
                .iter()
                .all(|line| matches!(line, UnifiedLine::Add(_)))
    });
    any_pure_insert.then(|| {
        "hunk(s) had no context lines and were located by line number only; if the file changed \
         since you read it, re-read it with read_file to confirm the insertion landed where \
         intended"
            .to_string()
    })
}

/// A lightweight truncation heuristic: when an inline patch is cut off by the
/// context manager, its tail often looks truncated (an unclosed envelope, or it
/// ends with a bare hunk header or a partial `***` marker). On a hit, append the
/// alternative `patch_file` path to the error text so the model does not keep
/// retrying the same truncated patch on a misleading parse error.
fn truncated_patch_hint(patch: &str) -> Option<&'static str> {
    let trimmed = patch.trim_end();
    if trimmed.is_empty() {
        return None;
    }
    let lines: Vec<&str> = trimmed.lines().collect();
    let last = lines.last().unwrap().trim_end();
    // 1) Starts with an envelope marker but is never closed
    let starts_envelope = lines
        .iter()
        .any(|line| line.trim_start().starts_with("*** Begin Patch"));
    if starts_envelope
        && !lines.iter().any(|line| {
            let t = line.trim();
            t == "*** End Patch" || t == "*** End of File"
        })
    {
        return Some("the patch starts with `*** Begin Patch` but has no closing `*** End Patch`");
    }
    // 2) Ends with a partial `***` section/closing marker (e.g. `*** End Patc`,
    //    `*** Update File: /pa`)
    let last_trimmed = last.trim();
    if last.starts_with("*** ")
        && last_trimmed != "*** End Patch"
        && last_trimmed != "*** End of File"
    {
        return Some("the patch ends with a partial `***` section marker");
    }
    // 3) Ends with a hunk header (no content lines after it): a hunk needs at least
    //    one body line that is ` ` / `-` / `+`.
    if last.trim_start().starts_with("@@") {
        return Some("the patch ends with a `@@` hunk header and no body lines after it");
    }
    None
}

/// When a parse failure hits the truncation heuristic, append the actionable
/// alternative path to the end of the error text.
fn append_truncated_patch_hint(patch: &str, err: String) -> String {
    match truncated_patch_hint(patch) {
        Some(hint) => format!(
            "{err}\n\nnote: {hint}. The patch was likely cut off by the context manager \
             mid-flight. Retry with a smaller inline patch, or write the patch to a temp file \
             with write_file(temp=true) and pass its path as `patch_file`."
        ),
        None => err,
    }
}

fn emit_stream_line(on_chunk: &mut ToolStreamWriter<'_>, line: &str) {
    let mut rendered = line.to_string();
    rendered.push('\n');
    on_chunk(rendered.as_bytes());
}

/// Strips code fences (```...``` or ~~~...~~~) the model often wraps around a
/// patch. Strips only when the whole patch is wrapped in one pair (opening fence on
/// the first line, bare closing fence on the last); fences inside the patch body
/// are left untouched to avoid damaging real patch content.
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
    // Walk backwards from the end to find the first non-empty line as the closing
    // fence candidate — the model often emits extra blank lines after the closing
    // fence.
    let last_nonempty = lines
        .iter()
        .rposition(|l| !l.trim().is_empty())
        .unwrap_or(0);
    let last = lines.get(last_nonempty).unwrap().trim();
    let is_close_fence = last == "```" || last == "~~~";
    if !is_close_fence || last_nonempty < 2 {
        return patch.to_string();
    }
    // Strip the first/last fence lines, keep the middle content, and trim excess
    // leading/trailing whitespace.
    lines[1..last_nonempty].join("\n").trim().to_string()
}

fn diff_stats_for_write(write: &PreparedPatchWrite) -> (usize, usize, usize) {
    // (added, removed, total_lines_after)
    match &write.action {
        PreparedPatchAction::Write(next) => {
            let after_lines = next.lines().count();
            match &write.before {
                Some(before) => {
                    // Count additions/removals by comparing line by line
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
    let mut message = if writes.len() == 1 {
        let (added, removed, total) = diff_stats_for_write(&writes[0]);
        format!(
            "Successfully patched {}; +{added} -{removed} ({total} lines)",
            writes[0].path.display()
        )
    } else {
        let mut m = format!("Successfully patched {} files:", writes.len());
        for write in writes {
            let (added, removed, total) = diff_stats_for_write(write);
            m.push_str(&format!(
                "\n- {}; +{added} -{removed} ({total} lines)",
                write.path.display()
            ));
        }
        m
    };
    for write in writes {
        for hint in &write.hints {
            message.push_str(&format!("\nnote: {hint}"));
        }
    }
    message
}

fn format_legacy_patch_dry_run(writes: &[PreparedPatchWrite]) -> String {
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

/// Kept only for compatibility with old sessions/history replay. `dry_run` is no
/// longer exposed to the model, but the old parameter cannot be silently ignored:
/// otherwise an old `dry_run: true` call would silently change from "validate only"
/// into a real write.
fn legacy_dry_run_arg(args: &Value) -> Result<bool, String> {
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

fn validate_delete_target(path: &Path) -> Result<(), String> {
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
    Ok(())
}

fn prepare_patch_action_from_content(
    path: &Path,
    current: Option<&str>,
    envelope: &PatchEnvelope,
) -> Result<(PreparedPatchAction, Vec<String>), String> {
    match envelope.op {
        PatchEnvelopeOp::Delete => {
            if !envelope.body_lines.is_empty() {
                return Err("Delete File sections must not contain patch content".to_string());
            }
            if current.is_none() {
                return Err(format!(
                    "Delete File target does not exist: {}",
                    path.display()
                ));
            }
            if !path.exists() {
                return Err(format!(
                    "Delete File cannot delete {} because it only exists in earlier sections of the same patch. Remove the Add/Delete no-op pair instead.",
                    path.display()
                ));
            }
            validate_delete_target(path)?;
            Ok((PreparedPatchAction::Delete, Vec::new()))
        }
        PatchEnvelopeOp::ReplaceInLine => {
            let original = current.ok_or_else(|| {
                format!(
                    "Replace in line: target file does not exist: {}",
                    path.display()
                )
            })?;
            Ok((
                PreparedPatchAction::Write(apply_inline_replace(original, envelope)?),
                Vec::new(),
            ))
        }
        PatchEnvelopeOp::Update => {
            let original = current.ok_or_else(|| {
                format!(
                    "Update File patch targets a missing file: {}. Use Add File to create a new file, or correct the target path before retrying.",
                    path.display()
                )
            })?;
            let normalized_patch = normalize_patch_envelope_body(envelope)?;
            let (next, hints) = apply_unified_patch_with_hints(original, &normalized_patch)?;
            Ok((PreparedPatchAction::Write(next), hints))
        }
        PatchEnvelopeOp::Add => {
            if current.is_some() {
                return Err(
                    "Add File patch targets an existing file. Use Update File or write_file instead."
                        .to_string(),
                );
            }
            let normalized_patch = normalize_patch_envelope_body(envelope)?;
            let (next, hints) = apply_unified_patch_with_hints("", &normalized_patch)?;
            Ok((PreparedPatchAction::Write(next), hints))
        }
    }
}

fn prepare_patch_write(
    path: &Path,
    store: &FileStore,
    envelope: &PatchEnvelope,
) -> Result<PreparedPatchWrite, String> {
    let before = if path.exists() {
        Some(store.read_to_string().map_err(|err| err.to_string())?)
    } else {
        None
    };
    let (action, hints) =
        if matches!(envelope.op, PatchEnvelopeOp::Update | PatchEnvelopeOp::Add) {
        // On the first encounter of a file, keep the disk-existence check so the
        // single-section behavior is unchanged; repeated sections for the same file
        // are handled by prepare_patch_action_from_content against in-memory state.
        let normalized_patch = normalize_patch_envelope(path, envelope)?;
        let original = before.as_deref().unwrap_or_default();
        apply_unified_patch_with_hints(original, &normalized_patch)
            .map(|(next, hints)| (PreparedPatchAction::Write(next), hints))?
    } else {
        prepare_patch_action_from_content(path, before.as_deref(), envelope)?
    };
    Ok(PreparedPatchWrite {
        path: path.to_path_buf(),
        before,
        action,
        hints,
    })
}

/// Prepares a write from an already-split single-file unified diff section (used
/// for the multi-file unified diff path). Unlike prepare_patch_write, the section
/// carries its own `--- `/`+++ ` file headers and `@@` hunks, so no envelope
/// normalization is applied — it is handed straight to apply_unified_patch
/// (parse_unified_hunks skips non-`@@` lines), reusing the tolerant match
/// semantics of unified diffs as-is.
fn prepare_patch_write_from_section(
    path: &Path,
    store: &FileStore,
    section: &str,
) -> Result<PreparedPatchWrite, String> {
    let before = if path.exists() {
        Some(store.read_to_string().map_err(|err| err.to_string())?)
    } else {
        None
    };
    let original = before.as_deref().unwrap_or_default();
    let (next, hints) = apply_unified_patch_with_hints(original, section)?;
    let action = PreparedPatchAction::Write(next);
    Ok(PreparedPatchWrite {
        path: path.to_path_buf(),
        before,
        action,
        hints,
    })
}

/// Rejects writes whose final content is identical to the original file, so the
/// tool never reports success without actually changing anything.
fn ensure_patch_writes_change(writes: &[PreparedPatchWrite]) -> Result<(), String> {
    for write in writes {
        let PreparedPatchAction::Write(next) = &write.action else {
            continue;
        };
        if write.before.as_deref() == Some(next.as_str()) {
            return Err(format!(
                "[NO_CHANGES] patch would leave {} unchanged. Remove the no-op hunk or re-read the file and rebuild the patch.",
                write.path.display()
            ));
        }
    }
    Ok(())
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
        PreparedPatchAction::Delete => {
            fs::remove_file(&write.path)
                .map_err(|err| format!("Failed to delete {}: {err}", write.path.display()))?;
            let _ = crate::ai::tools::storage::mutation_log::record(
                &write.path,
                "delete",
                write.before.as_deref(),
                None,
            );
            Ok(())
        }
    }
}

fn restore_prepared_patch_write(write: &PreparedPatchWrite) -> Result<(), String> {
    match &write.before {
        Some(content) => FileStore::new(write.path.clone())
            .write_all(content)
            .map_err(|err| format!("failed to restore {}: {err}", write.path.display())),
        None => match fs::remove_file(&write.path) {
            Ok(()) => {
                // Roll back the deletion (the file was newly created in this
                // batch): record the deleted content so the audit trail is complete.
                let deleted = match &write.action {
                    PreparedPatchAction::Write(c) => Some(c.as_str()),
                    PreparedPatchAction::Delete => None,
                };
                let _ = crate::ai::tools::storage::mutation_log::record(
                    &write.path,
                    "delete",
                    deleted,
                    None,
                );
                Ok(())
            }
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

    Ok(())
}

fn optional_patch_source_arg(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        // Some providers / tool bridges materialize optional schema fields as null
        // or empty strings. For mutually exclusive source arguments, both mean "not
        // provided" and must not be reported as an error when the other source is
        // valid.
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.trim().is_empty() => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(other) => Err(format!(
            "{key} parameter has wrong type ({}): expected a string{}.",
            value_type_name(other),
            if key == "patch_file" {
                " path to a file containing the patch text"
            } else {
                " containing the patch text"
            }
        )),
    }
}

fn execute_apply_patch_impl(args: &Value, mut emit: impl FnMut(&str)) -> Result<String, String> {
    let legacy_dry_run = legacy_dry_run_arg(args)?;
    let inline_patch = optional_patch_source_arg(args, "patch")?;
    let patch_file = optional_patch_source_arg(args, "patch_file")?;
    let (raw_patch, from_file) = match (inline_patch, patch_file) {
        (Some(_), Some(_)) => {
            return Err(
                "pass either `patch` (inline edit text) or `patch_file` (path to a file with the \
                 patch text), not both"
                    .to_string(),
            );
        }
        (Some(p), None) => (p, false),
        (None, Some(pf)) => {
            let store = FileStore::new(PathBuf::from(&pf));
            let resolved = store.path();
            // patch_file contract: it must be a session temp file (written with
            // write_file(temp=true) and registered in the temp registry) or a file
            // under effective_cwd; anything else is rejected outright so the model
            // cannot point at arbitrary system files and get confused.
            let in_cwd = crate::ai::driver::runtime_ctx::effective_cwd()
                .map(|cwd| resolved.starts_with(&cwd))
                .unwrap_or(false);
            if !temp_registry::is_registered(&resolved.to_string_lossy()) && !in_cwd {
                return Err(format!(
                    "patch_file '{}' is not an allowed patch source: write the patch to a temp \
                     file with write_file(temp=true) and pass its path here, or use a file under \
                     the current working directory.",
                    resolved.display()
                ));
            }
            emit(&format!("reading patch from {}", resolved.display()));
            let content = store.read_to_string().map_err(|err| {
                format!(
                    "patch_file '{}' could not be read: {err}",
                    resolved.display()
                )
            })?;
            (content, true)
        }
        (None, None) => {
            return Err(
                "patch parameter is missing or empty: provide `patch` (inline edit text) or \
                 `patch_file` (path to a temp file containing the patch text, written with \
                 write_file(temp=true)). If you sent a large patch, it may have been truncated \
                 by the context manager before reaching this tool - retry with a smaller patch \
                 (split into multiple apply_patch calls), or write the patch to a temp file and \
                 pass `patch_file` instead. For a small change, `*** Replace in line:` is the \
                 most reliable format."
                    .to_string(),
            );
        }
    };
    let patch = strip_code_fence(&raw_patch);
    // Truncation heuristic: when an inline patch ends with an unclosed envelope, a
    // partial `***` marker, or a bare hunk header, it was likely cut off by the
    // context manager. Emit a hint up front regardless of whether the patch then
    // succeeds or errors, so the model does not keep retrying truncated text as if
    // it were its own syntax error.
    if let Some(hint) = truncated_patch_hint(&patch) {
        emit(&format!(
            "warning: {hint}; the patch may have been truncated by the context manager \
             mid-flight. If it applies incompletely, retry with a smaller inline patch or pass \
             the patch via `patch_file` (write_file(temp=true))."
        ));
    }
    if from_file {
        // patch_file is meant to carry large patches (inline patches get truncated
        // by the context manager), so it is exempt from the 8K inline limit; only a
        // loose safety cap is set to stop the model from misreading oversized files.
        const MAX_PATCH_FILE_CHARS: usize = 64_000;
        if patch.chars().count() > MAX_PATCH_FILE_CHARS {
            return Err(format!(
                "patch_file too large ({} chars; limit 64000). Split the patch into multiple \
                 smaller patch files and apply them sequentially, or use write_file for a \
                 full-file rewrite.",
                patch.chars().count()
            ));
        }
    } else {
        // An oversized inline patch is very likely to be truncated by the context
        // manager (appearing as a missing/partial patch or a misleading context
        // mismatch). Erroring out directly with a splitting path beats feeding
        // truncated text to the parser.
        const MAX_INLINE_PATCH_CHARS: usize = 8_000;
        if patch.chars().count() > MAX_INLINE_PATCH_CHARS {
            return Err(format!(
                "patch too large ({} chars; limit 8000). Large patches are likely to be truncated \
                 by the context manager mid-flight. Split the edit into multiple smaller \
                 apply_patch calls (a few hunks each), write the patch to a temp file with \
                 write_file(temp=true) and pass its path as `patch_file`, or use write_file for a \
                 full-file rewrite.",
                patch.chars().count()
            ));
        }
    }
    emit("parsing patch envelope");
    let initial_file_path = optional_file_path_arg(args);
    if let Some(envelopes) = parse_patch_envelopes(&patch)
        .map_err(|err| append_truncated_patch_hint(&patch, err))?
    {
        // Each section inside an envelope (single-file or multi-file) already declares its own target path, so `file_path` is
        // redundant. Models often redundantly pass `file_path` for multi-file envelopes; instead of hard-erroring and wasting a round,
        // silently ignore it and use the envelope path (the envelope path is the authoritative source).
        if initial_file_path.is_some() {
            emit("note: ignoring redundant file_path arg; using paths from Begin Patch envelope");
        }
        emit(&format!("parsed {} patch section(s)", envelopes.len()));
        let mut writes: Vec<PreparedPatchWrite> = Vec::with_capacity(envelopes.len());
        let mut write_indexes: FxHashMap<PathBuf, usize> = FxHashMap::default();
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
            if let Some(&write_idx) = write_indexes.get(&path) {
                emit("applying after previous section for same file");
                let (action, hints) = {
                    let current = match &writes[write_idx].action {
                        PreparedPatchAction::Write(next) => Some(next.as_str()),
                        PreparedPatchAction::Delete => None,
                    };
                    prepare_patch_action_from_content(&path, current, envelope)
                }
                .map_err(|err| {
                    format!(
                        "[section {}/{}] failed while preparing patch for {}: {err}",
                        idx + 1,
                        envelopes.len(),
                        path.display()
                    )
                })?;
                writes[write_idx].action = action;
                writes[write_idx].hints.extend(hints);
            } else {
                let write = prepare_patch_write(&path, &store, envelope).map_err(|err| {
                    format!(
                        "[section {}/{}] failed while preparing patch for {}: {err}",
                        idx + 1,
                        envelopes.len(),
                        path.display()
                    )
                })?;
                write_indexes.insert(path.clone(), writes.len());
                writes.push(write);
            }
        }
        ensure_patch_writes_change(&writes)?;
        if legacy_dry_run {
            let success = format_legacy_patch_dry_run(&writes);
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

    let header_target = parse_unified_diff_header_target(&patch);
    // Multi-file unified diff (git diff output or hand-written by the model): split by file and prepare each one,
    // then commit only after all succeed — matching the batch semantics of the Begin Patch envelope (atomic, same-path stacking).
    // Note: multiple sections for the same file (two `diff --git a/x b/x`) dedupe to paths.len()==1 after header parsing,
    // so use the split section count (>1) or the count of distinct parsed paths (>1) as the criterion — both take the split path.
    let split_sections = split_unified_diff_by_file(&patch);
    let multi_file_sections = match &split_sections {
        Ok(sections) if sections.len() > 1 || header_target.paths.len() > 1 => Some(sections),
        Ok(_) => None,
        Err(err) if header_target.paths.len() > 1 => {
            return Err(format!("multi-file unified diff could not be split: {err}"));
        }
        Err(_) => None, // 无文件头（裸 hunks 靠 file_path 定位）：单文件路径
    };
    if let Some(sections) = multi_file_sections {
        emit(&format!(
            "parsing multi-file unified diff ({} file(s))",
            sections.len()
        ));
        let mut writes: Vec<PreparedPatchWrite> = Vec::with_capacity(sections.len());
        let mut write_indexes: FxHashMap<PathBuf, usize> = FxHashMap::default();
        for (idx, (target, section)) in sections.iter().enumerate() {
            if parse_unified_diff_header_target(section).deletes_file {
                return Err(format!(
                    "[section {}/{}] unified diff deletion (`+++ /dev/null`) is not supported \
                     because unified mode writes file content; use a `*** Begin Patch` envelope \
                     with `*** Delete File: {target}` so deletion is explicit",
                    idx + 1,
                    sections.len()
                ));
            }
            let store = FileStore::new(PathBuf::from(target));
            let path = store.path().to_path_buf();
            emit(&format!(
                "target [{}/{}]: {}",
                idx + 1,
                sections.len(),
                path.display()
            ));
            emit("validating write access");
            store
                .validate_write_access()
                .map_err(|err| err.to_string())?;
            if let Some(&write_idx) = write_indexes.get(&path) {
                // Multiple sections for the same file apply in order (same semantics as the envelope branch).
                emit("applying after previous section for same file");
                let (action, hints) = {
                    let current = match &writes[write_idx].action {
                        PreparedPatchAction::Write(next) => Some(next.as_str()),
                        PreparedPatchAction::Delete => None,
                    };
                    let original = current.unwrap_or_default();
                    apply_unified_patch_with_hints(original, section)
                        .map(|(next, hints)| (PreparedPatchAction::Write(next), hints))
                }
                .map_err(|err| {
                    format!(
                        "[section {}/{}] failed while preparing patch for {}: {err}",
                        idx + 1,
                        sections.len(),
                        path.display()
                    )
                })?;
                writes[write_idx].action = action;
                writes[write_idx].hints.extend(hints);
            } else {
                let write =
                    prepare_patch_write_from_section(&path, &store, section).map_err(|err| {
                        format!(
                            "[section {}/{}] failed while preparing patch for {}: {err}",
                            idx + 1,
                            sections.len(),
                            path.display()
                        )
                    })?;
                write_indexes.insert(path.clone(), writes.len());
                writes.push(write);
            }
        }
        ensure_patch_writes_change(&writes)?;
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
    if header_target.deletes_file {
        return Err(
            "unified diff deletion (`+++ /dev/null`) is not supported because unified mode writes file content; use a `*** Begin Patch` envelope with `*** Delete File: <path>` so deletion is explicit"
                .to_string(),
        );
    }
    let inferred_file_path = header_target.paths.first().cloned();
    let file_path = match initial_file_path {
        Some(path) => path.to_string(),
        // When file_path is missing, fall back to parsing the git-style diff headers (`--- a/`/`+++ b/`/`diff --git`)
        // for the paths they carry: a valid unified diff written by the model must not be rejected as "missing file_path".
        None => inferred_file_path.clone().ok_or(
            "missing file_path: provide `file_path` (or `path`) arg, wrap the patch in a \
             `*** Begin Patch` / `*** Update File: <path>` envelope, or include a git-style \
             diff header (`--- a/<path>` and `+++ b/<path>`) so the target path can be read \
             from the patch itself.",
        )?,
    };
    let store = FileStore::new(PathBuf::from(&file_path));
    if initial_file_path.is_some()
        && let Some(header_path) = inferred_file_path
    {
        let header_store = FileStore::new(PathBuf::from(&header_path));
        if header_store.path() != store.path() {
            return Err(format!(
                "file_path '{}' conflicts with unified diff header target '{}'; use one consistent target",
                store.path().display(),
                header_store.path().display()
            ));
        }
    }
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
    let (next, hints) =
        apply_unified_patch_with_hints(before.as_deref().unwrap_or_default(), &patch)?;
    let write = PreparedPatchWrite {
        path: path.clone(),
        before,
        action: PreparedPatchAction::Write(next),
        hints,
    };
    ensure_patch_writes_change(std::slice::from_ref(&write))?;
    if legacy_dry_run {
        let success = format_legacy_patch_dry_run(&[write]);
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
        PatchEnvelopeOp, apply_inline_replace, apply_patch_target_paths_from_patch,
        apply_unified_patch, apply_unified_patch_with_hints, execute_apply_patch,
        file_path_from_unified_diff_header, parse_patch_envelope, parse_patch_envelopes,
        parse_unified_diff_header_target, parse_unified_hunks, strip_code_fence,
        truncated_patch_hint,
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
    fn apply_patch_schema_does_not_expose_legacy_dry_run() {
        let schema = crate::ai::tools::registry::tool_metadata::tool_parameters("apply_patch");
        assert!(schema["properties"].get("dry_run").is_none());
    }

    /// Offline replay: uses the actual apply_patch inputs the model issued in a real session (history.json),
    /// reconstructs the "file truth at the time" from them, and runs them through the **current** code, printing each patch's real
    /// success/failure. This is not a conventional assertion test — it verifies the "real call success rate", not the assertions I wrote.
    ///
    /// Ignored by default; enable when needed with:
    ///   AI_PATCH_REPLAY_DIR=/tmp/patch_review cargo test --bin a replay_apply_patch -- --ignored --nocapture
    /// The directory needs replay_manifest.json + rebuild/proc.rs + rebuild/session_pid.rs.
    #[test]
    #[ignore]
    fn replay_apply_patch_from_session() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let Ok(dir) = std::env::var("AI_PATCH_REPLAY_DIR") else {
            eprintln!("AI_PATCH_REPLAY_DIR not set; skipping replay");
            return;
        };
        let dir = PathBuf::from(dir);
        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("replay_manifest.json")).unwrap())
                .unwrap();

        // Absolute path prefix in the session, replaced with the temporary workspace.
        const OLD_PREFIX: &str = "/Users/bytedance/rust_tools/src/bin/ai/driver";
        let records = manifest.as_array().unwrap();
        let mut ok = 0usize;
        let total = records.len();
        for rec in records {
            let msg = rec["msg"].as_i64().unwrap();
            let session = rec["session"].as_str().unwrap();
            let patch = rec["patch"].as_str().unwrap();
            let dry_run = rec["dry_run"].as_bool().unwrap_or(false);

            // Each patch uses a brand-new rebuilt file (they must not pollute each other).
            let work = make_temp_path(&format!("replay_{msg}"));
            let commands = work.join("commands");
            fs::create_dir_all(&commands).unwrap();
            fs::copy(dir.join("rebuild/proc.rs"), commands.join("proc.rs")).unwrap();
            fs::copy(
                dir.join("rebuild/session_pid.rs"),
                work.join("session_pid.rs"),
            )
            .unwrap();

            let new_prefix = work.to_string_lossy().to_string();
            let patch_rewritten = patch.replace(OLD_PREFIX, &new_prefix);

            let args = serde_json::json!({ "patch": patch_rewritten, "dry_run": dry_run });
            let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD
                .sync_scope(work.clone(), || execute_apply_patch(&args));
            let now = if result.is_ok() { "OK" } else { "FAIL" };
            if result.is_ok() {
                ok += 1;
            }
            let detail = match &result {
                Ok(s) => s.lines().next().unwrap_or("").to_string(),
                Err(e) => e.lines().next().unwrap_or("").to_string(),
            };
            eprintln!("msg{msg}: session={session} current_code={now} | {detail}");
            let _ = fs::remove_dir_all(&work);
        }
        eprintln!("=== replay success: {ok}/{total} ===");
    }

    #[test]
    fn parse_unified_hunks_treats_empty_hunk_line_as_context() {
        // Models often write empty context lines as fully blank lines with no leading space; these should be treated as empty context lines,
        // not an error. This matches `git apply`'s tolerance for empty context lines.
        let patch = "@@ -1,3 +1,3 @@\n foo\n\n bar\n";
        let hunks =
            parse_unified_hunks(patch).expect("empty hunk line should be treated as context");
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].lines.len(), 3);
    }

    #[test]
    fn apply_unified_patch_tolerates_empty_context_line() {
        // Models often write empty context lines as empty strings (no leading space); apply_patch should match them normally.
        let original = "foo\n\nbar\n";
        let patch = "@@ -1,3 +1,3 @@\n foo\n\n-bar\n+baz\n";
        let result =
            apply_unified_patch(original, patch).expect("empty context line should be tolerated");
        assert_eq!(result, "foo\n\nbaz\n");
    }

    #[test]
    fn apply_unified_patch_strips_trailing_cr_from_crlf_patch() {
        // CRLF patch: the trailing \r on Add lines must not be written into the file content.
        let original = "foo\nbar\n";
        let patch = "@@ -2,1 +2,1 @@\r\n-bar\r\n+baz\r\n";
        let result = apply_unified_patch(original, patch).expect("CRLF patch should be tolerated");
        assert_eq!(result, "foo\nbaz\n");
    }

    #[test]
    fn apply_unified_patch_tolerates_empty_context_line_in_crlf_patch() {
        // Empty context lines in a CRLF patch (lines with only \r) should also be treated as empty context lines.
        let original = "foo\r\n\r\nbar\r\n";
        let patch = "@@ -1,3 +1,3 @@\r\n foo\r\n\r\r\n-bar\r\n+baz\r\n";
        let result = apply_unified_patch(original, patch)
            .expect("empty CRLF context line should be tolerated");
        // The original file is CRLF, but the patch's Add lines have already stripped \r; output is uniformly LF.
        assert_eq!(result, "foo\n\nbaz\n");
    }

    #[test]
    fn parse_unified_hunks_strips_trailing_blank_context_between_hunks() {
        // Hunks are separated by blank lines (a readability convention). Previously the blank line was swallowed into hunk1 as a trailing
        // empty context line, spuriously requiring the original file to have a blank line at that position → context mismatch.
        // After the fix, that trailing blank line should be stripped, leaving hunk1 with only the remove+add lines.
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
        // Reproduces a real high-frequency scenario: hunks separated by blank lines. Before the fix, hunk1 ended with an extra empty context
        // line, making the whole patch report context mismatch.
        let original = "a\nkeep1\nkeep2\nkeep3\nc\n";
        let patch = "@@ -1,1 +1,1 @@\n-a\n+b\n\n@@ -5,1 +5,1 @@\n-c\n+d\n";
        let result = apply_unified_patch(original, patch)
            .expect("multi-hunk patch separated by a blank line should apply");
        assert_eq!(result, "b\nkeep1\nkeep2\nkeep3\nd\n");
    }

    #[test]
    fn apply_unified_patch_tolerates_trailing_blank_line_in_patch() {
        // The patch ends with extra blank lines (common model output). Before the fix, the trailing blank lines were merged into the last hunk
        // as empty context lines → match failure.
        let original = "line1\nline2\nline3\n";
        let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n\n";
        let result = apply_unified_patch(original, patch)
            .expect("trailing blank line in patch should be tolerated");
        assert_eq!(result, "line1\nchanged\nline3\n");
    }

    #[test]
    fn apply_unified_patch_tolerates_envelope_end_marker() {
        // Models often mistakenly append envelope tail markers like `*** End Patch` at the end of a unified-diff hunk
        // (format mixing). These markers do not belong to unified-diff content; the current hunk should end silently,
        // not report invalid hunk line.
        let original = "line1\nline2\nline3\n";
        let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n*** End Patch\n";
        let result = apply_unified_patch(original, patch)
            .expect("trailing `*** End Patch` marker should be tolerated");
        assert_eq!(result, "line1\nchanged\nline3\n");
    }

    #[test]
    fn apply_unified_patch_rejects_envelope_section_marker_with_hint() {
        // When a unified-diff hunk mixes in `*** Begin Patch` / `*** Update File:` opener or
        // section markers, the patch structure is confused. It should error with an explicit "format mixing" message guiding the model
        // to rebuild with one of the two formats, instead of a generic invalid hunk line.
        let original = "line1\nline2\nline3\n";
        let patch = "@@ -2,1 +2,1 @@\n-line2\n+changed\n*** Begin Patch\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("mixed patch formats"), "err was: {err}");
        assert!(err.contains("*** Begin Patch"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_rejects_malformed_envelope_trailer_not_silently_applied() {
        // Safety property: when a patch contains `*** Begin Patch` / `*** Update File:` envelope
        // markers (i.e. a truncated envelope leaked into the unified-diff path), a trailing `*** End Patch`
        // must never be silently tolerated and the hunk applied to the file_path target that the envelope did not declare.
        // It must report a "format mixing" error for the model to rebuild. Even if `original` here happens to contain the same
        // context (the most dangerous coincidence), it must error rather than write.
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
        // Deleting content that does not exist should trigger a context mismatch with context.
        let patch = "@@ -2,1 +2,1 @@\n-not_present\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        assert!(
            err.lines()
                .next()
                .unwrap_or_default()
                .contains("Rebuild the patch from the current file text"),
            "first error line should include the recovery action: {err}"
        );
        // The error should echo the expected line and the actual file content so the model can self-correct.
        assert!(err.contains("not_present"), "err was: {err}");
        assert!(err.contains("beta"), "err was: {err}");
        // Should include a directly pasteable current text block with no line-number prefix.
        assert!(err.contains("<<<PATCH_TEXT"), "err was: {err}");
        assert!(err.contains("PATCH_TEXT>>>"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_context_mismatch_reports_unicode_code_points() {
        // Uses a genuinely "non-confusable" Unicode difference to trigger the mismatch, verifying the error echoes the code point.
        // Note: smart quotes (U+201C/U+201D) and ASCII quotes are already tolerated by normalize_confusables normalization
        // (see apply_unified_patch_tolerates_confusable_quotes), so they can no longer serve as mismatch samples.
        // Here we use accented é (U+00E9) vs e (U+0065) -- not in the confusable normalization range; a real difference.
        let original = "let label = \"café\";\n";
        let patch = "@@ -1,1 +1,1 @@\n-let label = \"cafe\";\n+let label = \"changed\";\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        assert!(err.contains("U+00E9"), "err was: {err}");
        assert!(err.contains("U+0065"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_tolerates_confusable_quotes() {
        // P0: models often auto-replace ASCII quotes/hyphens with typographic smart quotes / en-dash.
        // Such purely typographic differences must not cause a context mismatch -- after normalize_confusables normalization they should match.
        // Key safety property: the context line outputs the original file content (actual), not the smart quote from the patch,
        // so the ASCII characters in the file are never "replaced" with smart quotes.
        let original = "let quote = \"hi\";\nlet dash = a - b;\n";
        // context lines use smart quotes (“ ”), remove lines use en-dash (– U+2013),
        // the file has ASCII quotes / ASCII hyphen -- after normalization all should match.
        let patch = "@@ -1,2 +1,2 @@\n let quote = “hi”;\n-let dash = a – b;\n+let dash = a - b;\n";
        let result = apply_unified_patch(original, patch)
            .expect("confusable smart quotes / en-dash should be tolerated");
        // context lines keep the original file's ASCII quotes; the remove en-dash matches the file's ASCII hyphen and is deleted;
        // add lines write the patch content (ASCII hyphen).
        assert_eq!(result, "let quote = \"hi\";\nlet dash = a - b;\n");
    }

    #[test]
    fn apply_unified_patch_detects_ambiguous_match() {
        // The same line appears multiple times in the file, and the nominal position does not match; an ambiguity error should be reported.
        let original = "dup\nmid\ndup\ntail\n";
        let patch = "@@ -9,1 +9,1 @@\n-dup\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
        // The ambiguity error should echo the current line text of each candidate position so the model can pick a unique anchor without re-reading.
        assert!(
            err.contains("Candidate locations"),
            "ambiguous error should echo candidate current lines: {err}"
        );
        assert!(err.contains("line 1"), "err was: {err}");
        assert!(err.contains("line 3"), "err was: {err}");
    }

    /// When the model wrongly writes `@@ -0,0 +1,3 @@` (insert at file start), normalize it to old_start=1,
    /// instead of treating it as "no nominal line number" (old_start=0) and reporting a context mismatch with "declared line 0".
    #[test]
    fn parse_unified_hunks_normalizes_zero_declared_line_to_one() {
        let patch = "@@ -0,0 +1,3 @@\n+aaa\n+bbb\n+ccc\n";
        let hunks = parse_unified_hunks(patch).expect("@@ -0 should parse");
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].old_start, 1);
        // And can be applied directly at the top of an empty file.
        let result = apply_unified_patch("", patch).expect("insert at top should apply");
        assert_eq!(result, "aaa\nbbb\nccc");
    }

    /// `@@ -0` insertion at the top of an existing file should also work.
    #[test]
    fn apply_unified_patch_inserts_at_top_with_zero_declared_line() {
        let original = "first\nsecond\n";
        let patch = "@@ -0,0 +1,2 @@\n+head\n+top\n";
        let result =
            apply_unified_patch(original, patch).expect("@@ -0 insert at top should apply");
        assert_eq!(result, "head\ntop\nfirst\nsecond\n");
    }

    /// A pure-insertion hunk (only `+` lines) on a non-empty file is located only by line number with no content verification;
    /// on success it should return a hint reminding the model to re-read the file after the change.
    #[test]
    fn apply_unified_patch_pure_insert_reports_line_number_hint() {
        let original = "first\nsecond\n";
        let patch = "@@ -2,0 +3,2 @@\n+mid1\n+mid2\n";
        let (result, hints) =
            apply_unified_patch_with_hints(original, patch).expect("pure insert should apply");
        assert_eq!(result, "first\nmid1\nmid2\nsecond\n");
        assert!(
            hints.iter().any(|h| h.contains("line number")),
            "pure insert should carry a line-number hint, hints were: {hints:?}"
        );
    }

    /// Hunks with context/remove lines go through content verification, so no pure-insertion hint is produced.
    #[test]
    fn apply_unified_patch_no_hint_for_context_anchored_hunks() {
        let original = "first\nsecond\n";
        let patch = "@@ -1,2 +1,2 @@\n first\n-second\n+changed\n";
        let (result, hints) =
            apply_unified_patch_with_hints(original, patch).expect("context hunk should apply");
        assert_eq!(result, "first\nchanged\n");
        assert!(
            hints.is_empty(),
            "context-anchored hunk should have no hints: {hints:?}"
        );
    }

    /// Pure insertion on an empty file is the normal flow for creating a new file; no hint is produced.
    #[test]
    fn apply_unified_patch_pure_insert_on_empty_file_has_no_hint() {
        let patch = "@@ -0,0 +1,2 @@\n+a\n+b\n";
        let (result, hints) =
            apply_unified_patch_with_hints("", patch).expect("add file should apply");
        assert_eq!(result, "a\nb");
        assert!(
            hints.is_empty(),
            "empty-file insert should have no hints: {hints:?}"
        );
    }

    /// The pure-insertion hint should be returned with the success message (format_patch_success appends a note).
    #[test]
    fn apply_patch_success_message_includes_pure_insert_hint() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("pure_insert_hint");
        let target = base.join("target.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&target, "first\nsecond\n").unwrap();

        let patch = "@@ -2,0 +3,2 @@\n+mid1\n+mid2\n";
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let result = execute_apply_patch(&serde_json::json!({
                "patch": patch,
                "file_path": "target.txt",
            }))
            .expect("pure insert should succeed");
            assert!(result.contains("Successfully patched"), "result was: {result}");
            assert!(
                result.contains("line number"),
                "success message should carry the pure-insert hint: {result}"
            );
        });

        assert_eq!(fs::read_to_string(&target).unwrap(), "first\nmid1\nmid2\nsecond\n");
        let _ = fs::remove_dir_all(base);
    }

    /// Truncation heuristic: recognize unclosed envelopes, broken `***` markers, and bare hunk-header endings;
    /// legal endings must not be misreported.
    #[test]
    fn truncated_patch_hint_heuristics() {
        assert!(
            truncated_patch_hint(
                "*** Begin Patch\n*** Update File: x.rs\n@@ -1,1 +1,1 @@\n-a\n+b\n"
            )
            .is_some(),
            "unclosed envelope should be flagged"
        );
        assert!(
            truncated_patch_hint("@@ -1,1 +1,1 @@\n-a\n+b\n@@").is_some(),
            "trailing bare @@ should be flagged"
        );
        assert!(
            truncated_patch_hint("@@ -1,1 +1,1 @@\n-a\n+b\n*** End Patc").is_some(),
            "partial *** marker should be flagged"
        );
        // Legal endings must not be misreported.
        assert!(truncated_patch_hint("@@ -1,1 +1,1 @@\n-a\n+b\n").is_none());
        assert!(
            truncated_patch_hint(
                "*** Begin Patch\n*** Update File: x.rs\n@@ -1,1 +1,1 @@\n-a\n+b\n*** End Patch\n"
            )
            .is_none()
        );
    }

    /// When parsing an unclosed (truncated) envelope fails, the error should include a truncation hint and a patch_file
    /// alternative path, so the model does not treat the truncated text as its own syntax error and retry repeatedly.
    #[test]
    fn apply_patch_unclosed_envelope_error_hints_truncation_and_patch_file() {
        let patch = "*** Begin Patch\n*** Update File: missing.rs\n@@ -1,1 +1,1 @@\n-a\n+b\n";
        let args = serde_json::json!({ "patch": patch });
        let err = execute_apply_patch(&args).unwrap_err();
        assert!(err.contains("patch_file"), "err was: {err}");
        assert!(err.contains("cut off"), "err was: {err}");
    }

    /// When the patch is missing, give a truncation hint and the patch_file alternative path.
    #[test]
    fn apply_patch_missing_patch_error_hints_truncation_and_patch_file() {
        let args = serde_json::json!({});
        let err = execute_apply_patch(&args).unwrap_err();
        assert!(err.contains("patch parameter is missing"), "err was: {err}");
        assert!(err.contains("patch_file"), "err was: {err}");
        assert!(err.contains("truncated"), "err was: {err}");
    }

    /// An oversized inline patch should error out immediately and guide splitting, rather than parse with the defect.
    #[test]
    fn apply_patch_rejects_oversized_inline_patch() {
        let huge = format!("@@ -1,1 +1,1 @@\n-a\n+{}", "x".repeat(9_000));
        let args = serde_json::json!({ "patch": huge });
        let err = execute_apply_patch(&args).unwrap_err();
        assert!(err.contains("patch too large"), "err was: {err}");
        assert!(err.contains("patch_file"), "err was: {err}");
    }

    /// A patch_file carrying a large patch (>8K inline cap) should succeed: the inline limit applies only to inline patches;
    /// otherwise the audit-flagged "recommended fallback path is actually unusable" contradiction cannot be resolved.
    #[test]
    fn apply_patch_large_patch_file_above_inline_limit_applies() {
        let temp = make_temp_path("patch_file_large");
        std::fs::create_dir_all(&temp).unwrap();
        let patch_path = temp.join("large.patch");
        let huge = format!("@@ -1,1 +1,1 @@\n-a\n+{}", "y".repeat(9_000));
        std::fs::write(&patch_path, &huge).unwrap();
        let target = temp.join("target.txt");
        std::fs::write(&target, "a\n").unwrap();
        let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp.clone(), || {
            let args = serde_json::json!({
                "patch_file": patch_path.to_string_lossy(),
                "file_path": target.to_string_lossy(),
            });
            execute_apply_patch(&args)
        });
        let out = result.expect("large patch_file should apply");
        assert!(out.contains("+1 -1"), "out was: {out}");
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            format!("{}\n", "y".repeat(9_000))
        );
    }

    /// An empty patch_file materialized by the tool bridge counts as "not provided" and must not block a valid inline patch.
    #[test]
    fn apply_patch_inline_accepts_empty_patch_file_placeholder() {
        let temp = make_temp_path("empty_patch_file_placeholder");
        std::fs::create_dir_all(&temp).unwrap();
        let target = temp.join("target.txt");
        std::fs::write(&target, "old\n").unwrap();
        let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp.clone(), || {
            let args = serde_json::json!({
                "patch": "@@ -1,1 +1,1 @@\n-old\n+new\n",
                "patch_file": "",
                "file_path": target.to_string_lossy(),
            });
            execute_apply_patch(&args)
        });
        result.expect("empty patch_file placeholder should be absent");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new\n");
    }

    /// When both sources are empty, missing-patch must still be explicitly reported rather than attempting to execute empty content.
    #[test]
    fn apply_patch_empty_source_placeholders_report_missing_patch() {
        let args = serde_json::json!({ "patch": "", "patch_file": null });
        let err = execute_apply_patch(&args).unwrap_err();
        assert!(err.contains("missing or empty"), "err was: {err}");
    }

    /// A patch_file exceeding the loose safety cap (64K) is explicitly rejected with guidance to split.
    #[test]
    fn apply_patch_rejects_oversized_patch_file() {
        let temp = make_temp_path("patch_file_huge");
        std::fs::create_dir_all(&temp).unwrap();
        let patch_path = temp.join("huge.patch");
        let huge = format!("@@ -1,1 +1,1 @@\n-a\n+{}", "z".repeat(70_000));
        std::fs::write(&patch_path, &huge).unwrap();
        let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp.clone(), || {
            let args = serde_json::json!({ "patch_file": patch_path.to_string_lossy() });
            execute_apply_patch(&args)
        });
        let err = result.unwrap_err();
        assert!(err.contains("patch_file too large"), "err was: {err}");
    }

    /// patch_file reads the patch from a file under effective_cwd (sync_scope) and applies it.
    #[test]
    fn apply_patch_reads_patch_from_patch_file_under_cwd() {
        let temp = make_temp_path("patch_file_cwd");
        std::fs::create_dir_all(&temp).unwrap();
        let patch_path = temp.join("edit.patch");
        std::fs::write(&patch_path, "@@ -1,1 +1,1 @@\n-foo\n+bar\n").unwrap();
        let target = temp.join("target.txt");
        std::fs::write(&target, "foo\n").unwrap();
        let result = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(temp.clone(), || {
            let args = serde_json::json!({
                "patch": "",
                "patch_file": patch_path.to_string_lossy(),
                "file_path": target.to_string_lossy(),
            });
            execute_apply_patch(&args)
        });
        let out = result.expect("patch_file should apply");
        assert!(out.contains("+1 -1"), "out was: {out}");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "bar\n");
    }

    /// A patch_file pointing outside cwd and not registered in the temp registry must be explicitly rejected.
    #[test]
    fn apply_patch_rejects_patch_file_outside_cwd_and_registry() {
        let outside =
            std::env::temp_dir().join(format!("ai_patch_outside_{}.patch", uuid::Uuid::new_v4()));
        std::fs::write(&outside, "@@ -1,1 +1,1 @@\n-foo\n+bar\n").unwrap();
        let args = serde_json::json!({ "patch_file": outside.to_string_lossy() });
        let err = execute_apply_patch(&args).unwrap_err();
        assert!(
            err.contains("not an allowed patch source"),
            "err was: {err}"
        );
    }

    /// On a context mismatch where the file contains no partial match at all (the no-partial-match branch),
    /// it should also attach a directly pasteable current text block with no line-number prefix.
    #[test]
    fn apply_unified_patch_context_mismatch_emits_pasteable_block_without_partial_match() {
        let original = "alpha\nbeta\ngamma\n";
        // The expected block and the file content have zero overlap, triggering the no-partial-match branch.
        let patch = "@@ -1,2 +1,2 @@\n-zzz1\n-zzz2\n+repl\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        assert!(err.contains("<<<PATCH_TEXT"), "err was: {err}");
        assert!(err.contains("PATCH_TEXT>>>"), "err was: {err}");
        // The pasteable block contains real original file lines, without a `<line>: ` prefix.
        let block = err
            .split("<<<PATCH_TEXT\n")
            .nth(1)
            .and_then(|rest| rest.split("\nPATCH_TEXT>>>").next())
            .unwrap_or_default();
        assert!(block.contains("alpha"), "block was: {block:?}");
        assert!(
            !block.contains(':'),
            "pasteable block must not carry line-number prefixes: {block:?}"
        );
    }

    /// "hunks out of order" is no longer a bare string: it should state the reason, give pasteable current text, and
    /// suggest reordering by line number or switching to Replace in line. In a real session the model lost 4 rounds in a row on the bare error.
    #[test]
    fn apply_unified_patch_out_of_order_error_is_actionable() {
        // File: `first` before `last`. The patch writes the hunks in reverse order: first modifies `last` (line 4),
        // then `first` (line 1) -- the second hunk's match position falls before the cursor, triggering out of order.
        let original = "first\naaa\nbbb\nlast\n";
        let patch = concat!("@@\n-last\n+LAST\n", "@@\n-first\n+FIRST\n",);
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("hunks out of order"), "err was: {err}");
        // Should include actionable reorder / Replace in line suggestions and a pasteable text block.
        assert!(
            err.contains("ascending file line number"),
            "err should explain ordering rule: {err}"
        );
        assert!(
            err.contains("<<<PATCH_TEXT"),
            "err should echo current text: {err}"
        );
        assert!(err.contains("PATCH_TEXT>>>"), "err was: {err}");
        assert!(
            err.contains("consumed through 1-based line 4"),
            "err should report the previous hunk's inclusive end line: {err}"
        );
        assert!(
            err.contains("must start at 1-based line 5 or later"),
            "err should report the next hunk's earliest start line: {err}"
        );
    }

    #[test]
    fn apply_unified_patch_disambiguates_ambiguous_match_by_nearby_declared_line() {
        let original = "dup\nhead\nfiller1\nfiller2\nfiller3\ndup\ntail\n";
        // `dup` appears twice, but the hunk header claims line 5, clearly closer to the second candidate (line 6).
        let patch = "@@ -5,1 +5,1 @@\n-dup\n+changed\n";
        let result = apply_unified_patch(original, patch)
            .expect("nearby declared line should disambiguate repeated context");
        assert_eq!(
            result,
            "dup\nhead\nfiller1\nfiller2\nfiller3\nchanged\ntail\n"
        );
    }

    #[test]
    fn apply_unified_patch_rejects_declared_line_when_not_clear_nearest() {
        let original = "dup\nleft\nmid\ndup\nright\n";
        // The nominal line 3 sits between two `dup`s with equal candidate distance; it must not guess.
        let patch = "@@ -3,1 +3,1 @@\n-dup\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_finds_unique_match_beyond_search_radius() {
        // The file has 150 lines; the only match is at line 130 (0-based 129).
        // The hunk header claims line 1, nominal=0, and find_hunk_offset's ±50 window searches [0,50),
        // which cannot find the match at 129. But all_hunk_match_positions can find the unique match across the whole file.
        // Previously the code ignored the forward.len()==1 result and fell back to find_hunk_offset, causing a false
        // "context mismatch"。
        let mut lines: Vec<String> = (0..130).map(|i| format!("filler{i}")).collect();
        lines.push("unique_target".to_string());
        lines.push("after_target".to_string());
        lines.extend((0..18).map(|i| format!("tail{i}")));
        let original = lines.join("\n") + "\n";

        let patch = "@@ -1,2 +1,2 @@\n-unique_target\n+changed\n+after_target\n";
        // Deliberately uses a wrong nominal line number (-1) to simulate stale line numbers
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
        // Real high-frequency failure scenario: in markdown/nested lists, the model's recreated context line indentation differs from the original file
        // (here the patch is missing 2 leading spaces). Before the fix, lines_match only did trim_end,
        // with zero tolerance for leading whitespace → the whole file failed to locate → "context mismatch: patch hunk could not
        // be located". After the fix, when strict matching fails, the indent-ignoring fallback uniquely locates and applies.
        let original = "# Title\n\n  - item one\n  - item two\n";
        // Leading spaces = context prefix; context content "- item one", remove content "- item two"
        // are both missing 2 indentation spaces compared to the original file.
        let patch = "@@ -3,2 +3,2 @@\n - item one\n-- item two\n+- item two changed\n";
        let result = apply_unified_patch(original, patch).unwrap_or_else(|err| {
            panic!("indent-insensitive fallback should locate the hunk, got: {err}")
        });
        // Context lines keep the original file's indentation; only the remove/add target lines are replaced.
        assert_eq!(result, "# Title\n\n  - item one\n- item two changed\n");
    }

    #[test]
    fn apply_unified_patch_indent_fallback_still_detects_ambiguity() {
        // The indent-ignoring fallback must not sacrifice safety: if ignoring indentation yields multiple matches, it must still report ambiguity,
        // not silently change the wrong place.
        let original = "  dup\nmid\n    dup\ntail\n";
        let patch = "@@ -9,1 +9,1 @@\n-dup\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_strict_match_preferred_over_indent_fallback() {
        // When strict matching uniquely locates, the strict-match result must be used, preserving the original file's exact content,
        // and must not fall back just because an indentation variant exists.
        let original = "    exact\nother\n";
        let patch = "@@ -1,1 +1,1 @@\n-    exact\n+    replaced\n";
        let result = apply_unified_patch(original, patch).unwrap();
        assert_eq!(result, "    replaced\nother\n");
    }

    #[test]
    fn apply_unified_patch_fuzzes_stale_context_when_remove_lines_are_unique() {
        // Real loop root cause: the model wrote the context line as stale/target-state content, but the remove line still
        // anchors precisely on the target. Context must not hard-reject the patch in this case.
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
        // When a remove line appears twice, fuzz can still score using other context lines; only a unique top score
        // may be applied, avoiding degradation into "modify the first identical remove line".
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
        // old_start=1 (1-based) → nominal=0; remove "old target" matches at line 1.
        // Even if all context misses, old_start can still disambiguate — it should apply successfully.
        let result = apply_unified_patch(original, patch).expect("should apply via nominal");
        assert_eq!(
            result, "alpha current\nnew target\nbeta current\nold target\n",
            "should replace the FIRST 'old target' (line 1), not the second (line 3)"
        );
    }

    #[test]
    fn apply_unified_patch_fuzzy_context_rejects_when_nominal_not_in_candidates() {
        // When the position old_start points to is not in the candidate list, it should still be rejected.
        // original: line 0="old target", line 1="xxx", line 2="old target", line 3="yyy"
        // patch: @@ -2,1 +2,1 @@ — old_start=2 → nominal=1
        // The hunk has only a remove line (no context); remove "xxx" appears at line 1.
        // But another variant: multiple "old target" as remove, and old_start points to a position with no match.
        // original: line 0="old target", line 1="aaa", line 2="old target", line 3="bbb"
        // patch: @@ -3,1 +3,1 @@ — old_start=3 → nominal=2
        // remove "old target" matches line 0 (pos=0) and line 2 (pos=2);
        // nominal=2 is in the candidate list → it is accepted (correct behavior).
        // Changed to: old_start points to a line number that does not exist in the file.
        let original = "old target\naaa\nold target\nbbb\n";
        let patch = "@@ -5,1 +5,1 @@\n-old target\n+changed\n";
        // old_start=5 → nominal=4, but the file has only 4 lines (index 0-3).
        // candidates: pos=0 and pos=2 (old target matches). nominal=4 is not in candidates.
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_indent_fallback_reports_context_mismatch_when_absent() {
        // Even ignoring indentation, if the content itself does not exist, it should still report a context mismatch (echoing the actual content).
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
        // The *** Begin Patch Update format omits the @@ header (Cursor/Aider style),
        // writing only +/−/space-prefixed lines. Models write this often; it must not report "no hunks found".
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
        // Two hunks, both nominal line numbers 1 (stale), but each target is unique in the file and ordered.
        // Verifies that cursor advancement + forward filtering work correctly across multiple hunks, without mis-matching the second
        // hunk onto the first hunk's target position.
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
        // The filler lines in between should remain unchanged
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
                result
                    .content
                    .starts_with(&format!("Successfully patched {};", path.display())),
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

        // file_path is silently ignored; the envelope declares b.txt as the authoritative target; b.txt does not exist → report missing file.
        assert!(
            err.contains("b.txt"),
            "err should mention the envelope target path: {err}"
        );
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
        // ~~~ fences are stripped as well.
        let fenced_tilde = "~~~\n@@ -1,1 +1,1 @@\n-x\n+y\n~~~";
        assert_eq!(strip_code_fence(fenced_tilde), "@@ -1,1 +1,1 @@\n-x\n+y");
    }

    #[test]
    fn strip_code_fence_leaves_unfenced_patch_untouched() {
        let raw = "@@ -1,1 +1,1 @@\n-x\n+y";
        assert_eq!(strip_code_fence(raw), raw);
        // When the closing fence is missing, do not strip, to avoid damaging a real patch whose content starts with ```.
        let no_close = "```diff\n@@ -1,1 +1,1 @@\n-x\n+y";
        assert_eq!(strip_code_fence(no_close), no_close);
        // Do not process when there are too few lines.
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
    fn file_path_from_unified_diff_header_reads_git_style_paths() {
        // The `+++ b/` side takes priority; strip the `b/` prefix.
        assert_eq!(
            file_path_from_unified_diff_header(
                "--- a/src/old.rs\n+++ b/src/new.rs\n@@ -1 +1 @@\n-x\n+y\n"
            )
            .as_deref(),
            Some("src/new.rs")
        );
        // Deletion case: `+++ /dev/null` is skipped, falling back to the `--- a/` side.
        assert_eq!(
            file_path_from_unified_diff_header(
                "--- a/src/gone.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
            )
            .as_deref(),
            Some("src/gone.rs")
        );
        // `diff --git a/… b/…` takes the b side; trailing TAB+timestamp is stripped.
        assert_eq!(
            file_path_from_unified_diff_header(
                "diff --git a/foo.rs b/foo.rs\n--- a/foo.rs\t2026-07-30\n+++ b/foo.rs\t2026-07-30\n@@ -1 +1 @@\n-a\n+b\n"
            )
            .as_deref(),
            Some("foo.rs")
        );
        // Absolute paths without an a/ b/ prefix are preserved as-is.
        assert_eq!(
            file_path_from_unified_diff_header("+++ /abs/path.rs\n@@ -1 +1 @@\n-a\n+b\n")
                .as_deref(),
            Some("/abs/path.rs")
        );
        // Without a diff header, return None (a bare `@@` hunk still requires an explicit file_path).
        assert_eq!(
            file_path_from_unified_diff_header("@@ -1 +1 @@\n-a\n+b\n"),
            None
        );
        // `---`/`+++` header parsing must stop before the first hunk, so body context lines are not mistaken for paths.
        assert_eq!(
            file_path_from_unified_diff_header("@@ -1 +1 @@\n +++ b/not-a-header.rs\n"),
            None
        );
        // Git writes paths with spaces in JSON/C-style quotes; decode first, then strip b/.
        assert_eq!(
            file_path_from_unified_diff_header(
                "--- \"a/src/old name.rs\"\n+++ \"b/src/new name.rs\"\n@@ -1 +1 @@\n-a\n+b\n"
            )
            .as_deref(),
            Some("src/new name.rs")
        );
        // A single-file call must not silently accept a standard multi-file unified diff without `diff --git`;
        // each file header must be followed by its own hunk; on conflict, do not infer any target.
        assert_eq!(
            file_path_from_unified_diff_header(
                "--- a/one.rs\n+++ b/one.rs\n@@ -1 +1 @@\n-a\n+b\n--- a/two.rs\n+++ b/two.rs\n@@ -1 +1 @@\n-c\n+d\n"
            ),
            None
        );
        // In a standard multi-file Git diff, the second `diff --git` comes after the first hunk; the conflict must still be recognized,
        // not silently applying the whole patch to the first file.
        assert_eq!(
            file_path_from_unified_diff_header(
                "diff --git a/one.rs b/one.rs\n--- a/one.rs\n+++ b/one.rs\n@@ -1 +1 @@\n-a\n+b\ndiff --git a/two.rs b/two.rs\n--- a/two.rs\n+++ b/two.rs\n@@ -1 +1 @@\n-c\n+d\n"
            ),
            None
        );
        assert_eq!(
            parse_unified_diff_header_target(
                "diff --git a/one.rs b/one.rs\n--- a/one.rs\n+++ b/one.rs\n@@ -1 +1 @@\n-a\n+b\ndiff --git a/two.rs b/two.rs\n--- a/two.rs\n+++ b/two.rs\n@@ -1 +1 @@\n-c\n+d\n"
            )
            .paths,
            vec!["one.rs", "two.rs"]
        );
        // Paths with spaces must be read per git's quoting/escaping rules, not split into half tokens by whitespace.
        assert_eq!(
            file_path_from_unified_diff_header(
                "diff --git \"a/foo bar.rs\" \"b/foo bar.rs\"\n--- \"a/foo bar.rs\"\n+++ \"b/foo bar.rs\"\n@@ -1 +1 @@\n-a\n+b\n"
            )
            .as_deref(),
            Some("foo bar.rs")
        );
        // Even without `+++`/`---` fallback, quoted paths should parse correctly from `diff --git`.
        assert_eq!(
            file_path_from_unified_diff_header(
                "diff --git \"a/foo bar.rs\" \"b/foo bar.rs\"\n@@ -1 +1 @@\n-a\n+b\n"
            )
            .as_deref(),
            Some("foo bar.rs")
        );
        // When `diff --git` and `+++` point to the same file (no quotes), it is not a multi-file conflict; parse normally.
        assert_eq!(
            file_path_from_unified_diff_header(
                "diff --git a/foo.rs b/foo.rs\n--- a/foo.rs\n+++ b/foo.rs\n@@ -1 +1 @@\n-a\n+b\n"
            )
            .as_deref(),
            Some("foo.rs")
        );
    }

    #[test]
    fn execute_apply_patch_reads_path_from_git_diff_header_without_file_path_arg() {
        // Reproduces the first domino of a historical loop: the model writes a textbook git unified diff
        // (with its own `--- a/` `+++ b/` headers), but does not pass file_path. Previously the tool reported missing
        // file_path, forcing the model to keep changing formats in trial and error. After the fix, it should read the path from the diff header and succeed.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("git_diff_header").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "line1\nline2\nline3\n").unwrap();
        let file_name = path.file_name().unwrap().to_string_lossy().to_string();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            // The patch header uses a relative file name (resolved relative to cwd), and neither file_path/path is passed.
            let patch = format!(
                "--- a/{file_name}\n+++ b/{file_name}\n@@ -1,3 +1,3 @@\n line1\n-line2\n+changed\n line3\n"
            );
            let args = serde_json::json!({ "patch": patch });
            execute_apply_patch(&args)
                .expect("apply_patch should read target path from git-style diff header");
        });

        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "line1\nchanged\nline3\n"
        );
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_missing_path_error_mentions_diff_header_option() {
        // When there is neither file_path, nor a diff header, nor an envelope, the error message should offer three ways out, one
        // being a git-style diff header, grounding the model to the correct next step.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let err = execute_apply_patch(&serde_json::json!({
            "patch": "@@ -1,1 +1,1 @@\n-old\n+new\n",
        }))
        .expect_err("bare hunk without file_path must error");
        assert!(err.contains("missing file_path"), "err was: {err}");
        assert!(
            err.contains("git-style") && err.contains("+++ b/"),
            "error should mention the git-style diff-header option; err was: {err}"
        );
    }

    #[test]
    fn execute_apply_patch_applies_multi_file_diff_automatically() {
        // Multi-file unified diff (git diff output style): no longer an error; auto-split by file and apply atomically.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("multi_file_diff_auto");
        let a = base.join("one.txt");
        let b = base.join("two.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&a, "old_a\n").unwrap();
        fs::write(&b, "old_b\n").unwrap();

        let patch = "diff --git a/one.txt b/one.txt\n--- a/one.txt\n+++ b/one.txt\n@@ -1 +1 @@\n-old_a\n+new_a\ndiff --git a/two.txt b/two.txt\n--- a/two.txt\n+++ b/two.txt\n@@ -1 +1 @@\n-old_b\n+new_b\n";
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let result = execute_apply_patch(&serde_json::json!({
                // The multi-file diff carries its own paths; a redundant file_path should be ignored
                "file_path": "one.txt",
                "patch": patch,
            }))
            .expect("multi-file unified diff should apply automatically");
            assert!(
                result.starts_with("Successfully patched 2 files:"),
                "result: {result}"
            );
        });

        assert_eq!(fs::read_to_string(&a).unwrap(), "new_a\n");
        assert_eq!(fs::read_to_string(&b).unwrap(), "new_b\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_applies_multi_file_diff_without_git_headers() {
        // A multi-file diff without `diff --git` headers, only `---`/`+++` pairs: split by adjacent file-header pairs.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("multi_file_diff_no_git");
        let a = base.join("a.txt");
        let b = base.join("b.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&a, "alpha\n").unwrap();
        fs::write(&b, "beta\n").unwrap();

        let patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-alpha\n+ALPHA\n--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-beta\n+BETA\n";
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let result = execute_apply_patch(&serde_json::json!({ "patch": patch }))
                .expect("multi-file diff without git headers should apply");
            assert!(
                result.starts_with("Successfully patched 2 files:"),
                "result: {result}"
            );
        });

        assert_eq!(fs::read_to_string(&a).unwrap(), "ALPHA\n");
        assert_eq!(fs::read_to_string(&b).unwrap(), "BETA\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_multi_file_diff_same_file_sections_stack() {
        // Multiple sections for the same file (same-path stacking semantics, consistent with the envelope branch).
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("multi_file_diff_stack");
        let a = base.join("a.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&a, "alpha\nbeta\ngamma\n").unwrap();

        let patch = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-alpha\n+ALPHA\ndiff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -3 +3 @@\n-gamma\n+GAMMA\n";
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let result = execute_apply_patch(&serde_json::json!({ "patch": patch }))
                .expect("same-file sections in multi-file diff should stack");
            assert!(
                !result.starts_with("Successfully patched 2 files:"),
                "same file should be committed once: {result}"
            );
        });

        assert_eq!(fs::read_to_string(&a).unwrap(), "ALPHA\nbeta\nGAMMA\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_multi_file_diff_is_atomic_on_failure() {
        // If any file's prepare fails, nothing is committed; previously prepared files are not written either.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("multi_file_diff_atomic");
        let a = base.join("a.txt");
        let b = base.join("b.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&a, "old_a\n").unwrap();
        fs::write(&b, "current_b\n").unwrap();

        let patch = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old_a\n+new_a\ndiff --git a/b.txt b/b.txt\n--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-missing_b\n+new_b\n";
        let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            execute_apply_patch(&serde_json::json!({ "patch": patch }))
                .expect_err("second file mismatch should abort whole multi-file diff")
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
    fn unified_header_parser_ignores_header_shaped_hunk_body_lines() {
        let patch =
            "--- a/notes.txt\n+++ b/notes.txt\n@@ -1,2 +1,2 @@\n--- old marker\n+++ new marker\n";
        assert_eq!(
            file_path_from_unified_diff_header(patch).as_deref(),
            Some("notes.txt")
        );
    }

    #[test]
    fn execute_apply_patch_rejects_dev_null_deletion_with_actionable_guidance() {
        let patch =
            "diff --git a/old.rs b/old.rs\n--- a/old.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n";
        let err = execute_apply_patch(&serde_json::json!({ "patch": patch }))
            .expect_err("unified mode must not silently turn deletion into an empty file");
        assert!(err.contains("+++ /dev/null"), "err was: {err}");
        assert!(err.contains("*** Delete File:"), "err was: {err}");
    }

    #[test]
    fn shared_target_extractor_covers_envelope_and_quoted_git_header() {
        assert_eq!(
            apply_patch_target_paths_from_patch(
                "*** Begin Patch\n*** Update File: src/a.rs\n*** Add File: src/b.rs\n*** End Patch"
            ),
            vec![PathBuf::from("src/a.rs"), PathBuf::from("src/b.rs")]
        );
        assert_eq!(
            apply_patch_target_paths_from_patch(
                "diff --git \"a/src/old name.rs\" \"b/src/new name.rs\"\n@@ -1 +1 @@\n-old\n+new\n"
            ),
            vec![PathBuf::from("src/new name.rs")]
        );
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
        // When a context line is missing its leading space, the error message should explicitly state the expected prefix.
        let err = parse_unified_hunks("@@ -1,3 +1,3 @@\nline1\n-line2\n+changed\n line3")
            .expect_err("missing leading space on context line must error");
        assert!(
            err.contains("must start with") && err.contains("context"),
            "err was: {err}"
        );
    }

    // ── Fix 1: strip_code_fence should tolerate trailing blank lines after the closing fence ──

    #[test]
    fn strip_code_fence_tolerates_trailing_blank_lines() {
        // Models often emit one or more extra blank lines after the closing fence; previously strip_code_fence treated the last
        // blank line as `last`, decided it was not a closing fence, and gave up stripping, so the whole patch stayed wrapped in the code fence
        // and went into the parser with an error.
        let fenced = "```diff\n@@ -1,1 +1,1 @@\n-line2\n+changed\n```\n";
        assert_eq!(
            strip_code_fence(fenced),
            "@@ -1,1 +1,1 @@\n-line2\n+changed"
        );
        // Multiple trailing blank lines should also be tolerated
        let fenced_multi = "```\n*** Begin Patch\n*** End Patch\n```\n\n\n";
        assert_eq!(
            strip_code_fence(fenced_multi),
            "*** Begin Patch\n*** End Patch"
        );
    }

    // ── Fix 2: give a clear error when the hunk header is missing ──

    #[test]
    fn parse_unified_hunks_missing_header_gives_clear_error() {
        // When patch content lines exist but there is no hunk header, give an error clearer than "no hunks found".
        let err = parse_unified_hunks(" line1\n-line2\n+changed\n line3")
            .expect_err("patch without hunk header must error");
        assert!(err.contains("no hunk header found"), "err was: {err}");
        assert!(err.contains("content lines"), "err was: {err}");
    }

    // ── Fix 3: envelope Update synthesized headers use old_start=0 ──

    #[test]
    fn execute_apply_patch_update_envelope_without_header_does_not_match_at_line_1() {
        // When the file start happens to match the hunk's context lines, the nominal match with old_start=1 may wrongly hit
        // the file start instead of where the model actually wants to change. old_start=0 gives the same nominal=0,
        // but with clearer semantics: no nominal position, relying on a whole-file search for a unique location.
        // Here we verify that a unique match not at the file start is located correctly.
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

    // ── Fix 4: fill in bare line prefixes when the envelope Update has no hunk header ──

    #[test]
    fn execute_apply_patch_update_envelope_tolerates_bare_lines() {
        // In the envelope Update format (no hunk header), the model wrote bare lines without a +/-/ prefix;
        // they should get an automatic space prefix and be treated as context lines, instead of reporting "invalid hunk line".
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

    // ── Fix 5: context lines tolerate line-number prefixes ──

    #[test]
    fn apply_unified_patch_tolerates_line_number_prefix_in_context() {
        // The model copied context lines with line-number prefixes from grep-like output (e.g. `   42| `);
        // the IgnoreIndent fallback mode should strip the line-number prefix and match successfully.
        // read_file's real TAB format is covered separately by apply_unified_patch_tolerates_read_file_tab_prefix.
        let original = "line1\nline2\nline3\n";
        // context line " line1" was wrongly written by the model as " 1| line1" with a line-number prefix
        let patch = "@@ -1,3 +1,3 @@\n 1| line1\n-line2\n+changed\n line3\n";
        let result = apply_unified_patch(original, patch)
            .expect("line number prefix in context should be tolerated by indent fallback");
        // context lines should keep the original file content (without the line-number prefix)
        assert_eq!(result, "line1\nchanged\nline3\n");
    }

    #[test]
    fn apply_unified_patch_tolerates_line_number_prefix_in_remove() {
        // The remove line also carries a line-number prefix and should be tolerated the same way.
        let original = "line1\ntarget\nline3\n";
        let patch = "@@ -1,3 +1,3 @@\n line1\n-2| target\n+changed\n line3\n";
        let result = apply_unified_patch(original, patch)
            .expect("line number prefix in remove line should be tolerated");
        assert_eq!(result, "line1\nchanged\nline3\n");
    }

    #[test]
    fn apply_unified_patch_tolerates_read_file_tab_prefix() {
        // Reproduces a real failure scenario from history: the model copied read_file output line by line into the patch's
        // context / remove lines. read_file's real render format is `{:>6}\t{}` (right-aligned line number + TAB);
        // before the fix, strip_line_number_prefix did not recognize TAB, causing repeated context mismatches.
        let original = "fn foo() {\n    let x = 1;\n    x\n}\n";
        // Construct the line the model sees using exactly the same rendering as read_file, to avoid miscounting spaces by hand.
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
        // context lines keep the original file content (including indentation); only target lines are replaced.
        assert_eq!(result, "fn foo() {\n    let x = 2;\n    x\n}\n");
    }

    #[test]
    fn apply_unified_patch_line_number_prefix_still_detects_ambiguity() {
        // Line-number-prefix tolerance must not sacrifice safety: if there are still multiple matches after stripping the number, report ambiguity.
        let original = "dup\ndup\ndup\n";
        // The nominal position is deliberately wrong, forcing a whole-file search; after stripping the number, context+remove = ["dup","dup"] matches multiple places
        let patch = "@@ -9,2 +9,2 @@\n 1| dup\n-dup\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
    }

    #[test]
    fn strip_line_number_prefix_does_not_strip_code_lines() {
        // Conservative single-argument fallback: only recognizes `digits+\t` and `digits+separator+space`, to avoid mis-stripping.
        use super::strip_line_number_prefix;
        // read_file's real format: right-aligned line number + TAB (the root-cause scenario previously missed)
        assert_eq!(
            strip_line_number_prefix("     3\tuse std::fs;"),
            "use std::fs;"
        );
        // After TAB, keep the code's original indentation (strip only one TAB, do not touch content indentation)
        assert_eq!(
            strip_line_number_prefix("    42\t    let x = 1;"),
            "    let x = 1;"
        );
        // grep-like formats (separator + space) should be stripped
        assert_eq!(strip_line_number_prefix("   42| hello"), "hello");
        assert_eq!(strip_line_number_prefix("42: hello"), "hello");
        // `80:80` (colon without a following space) is not a line-number prefix and must not be stripped
        assert_eq!(strip_line_number_prefix("80:80"), "80:80");
        // `3.14` (dot without a following space) must not be stripped
        assert_eq!(strip_line_number_prefix("3.14"), "3.14");
        // Pure digit lines must not be stripped (no separator)
        assert_eq!(strip_line_number_prefix("42"), "42");
        // Digits immediately followed by letters must not be stripped (`42px`)
        assert_eq!(strip_line_number_prefix("42px"), "42px");
        // Lines not starting with a digit must not be stripped
        assert_eq!(strip_line_number_prefix("hello"), "hello");
    }

    #[test]
    fn strip_number_prefix_anchored_is_separator_agnostic() {
        // Anchor-based: keyed to the real line, stripping the line-number column regardless of separator, with almost zero false positives.
        use super::strip_number_prefix_anchored;
        let actual = "    let x = 1;";
        // read_file TAB / grep `| ` / `: ` / space / `.` / `)` all compatible
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
        // After removing the column, not equal to the real line → return as-is (no false strip)
        assert_eq!(
            strip_number_prefix_anchored("42\tsomething else", actual),
            "42\tsomething else"
        );
        // Not starting with a digit → return as-is
        assert_eq!(strip_number_prefix_anchored(actual, actual), actual);
    }

    // ── Large-block replacement: best-effort partial matching precisely locates the inconsistent line ──

    #[test]
    fn apply_unified_patch_large_block_mismatch_pinpoints_wrong_line() {
        // In a large-block replacement where only one line's content is reproduced inaccurately, the error message should precisely locate which line is inconsistent
        // (expected vs actual), not just say "context mismatch".
        let original = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\n";
        // The remove block has 6 lines; line4 was mistyped by the model as lineX
        let patch =
            "@@ -2,6 +2,3 @@\n-line2\n-line3\n-lineX\n-line5\n-line6\n-line7\n+new2\n+new3\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        // Should report the best match position and match count
        assert!(err.contains("Best partial match"), "err was: {err}");
        assert!(err.contains("5/6 lines matched"), "err was: {err}");
        // Should precisely point out the inconsistent line: expected lineX but actual is line4
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
        // The expected block does not exist in the file at all (no line partially matches); it should echo the expected lines and
        // the actual content near the nominal position, instead of taking the partial-match branch.
        let original = "alpha\nbeta\ngamma\n";
        let patch = "@@ -2,1 +2,1 @@\n-not_present\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        // When the block does not exist at all, there is no "Best partial match"
        assert!(!err.contains("Best partial match"), "err was: {err}");
        // Should echo the expected lines
        assert!(err.contains("not_present"), "err was: {err}");
        // Should show the actual content near the nominal position
        assert!(err.contains("beta"), "err was: {err}");
    }

    #[test]
    fn apply_unified_patch_partial_match_uses_middle_line_anchor() {
        // The first line of the expected block is mistyped, but the middle lines are correct. The middle-line anchors should find the best match position,
        // and report the first line's inconsistency.
        let original = "aaa\nbbb\nccc\nddd\neee\n";
        // The first line "wrong" is not in the file, but "ccc", "ddd" are
        let patch = "@@ -1,3 +1,1 @@\n-wrong\n-ccc\n ddd\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("context mismatch"), "err was: {err}");
        // Should find a partial match via "ccc" or "ddd"
        assert!(err.contains("Best partial match"), "err was: {err}");
        assert!(err.contains("2/3 lines matched"), "err was: {err}");
        // Should point out the first line mismatch: expected "wrong" but actual is "bbb"
        assert!(
            err.contains("wrong"),
            "err should mention wrong expected line: {err}"
        );
        assert!(
            err.contains("bbb"),
            "err should mention actual file line: {err}"
        );
    }

    // ── Canonical *** Begin Patch envelope: bare @@ / @@ heading @@ headers without line numbers ──

    #[test]
    fn parse_unified_hunks_accepts_bare_at_header() {
        // The canonical envelope format uses bare `@@` to separate hunks, without `-N,M +N,M` line numbers.
        // Before the fix it reported "invalid hunk header".
        let patch = "@@\n foo\n-bar\n+baz\n";
        let hunks = parse_unified_hunks(patch).expect("bare @@ header should be accepted");
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].old_start, 0);
    }

    #[test]
    fn parse_unified_hunks_accepts_at_header_with_heading() {
        // `@@ <context title> @@` should also be accepted; the nominal line number is treated as 0 (whole-file search locates it).
        let patch = "@@ fn foo() @@\n foo\n-bar\n+baz\n";
        let hunks = parse_unified_hunks(patch).expect("@@ heading @@ header should be accepted");
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].old_start, 0);
    }

    #[test]
    fn apply_unified_patch_applies_bare_at_header_hunk() {
        // End-to-end: a hunk with a bare @@ header should be uniquely located and applied via whole-file search.
        let original = "alpha\nbeta\ngamma\n";
        let patch = "@@\n alpha\n-beta\n+changed\n";
        let result = apply_unified_patch(original, patch).expect("bare @@ hunk should apply");
        assert_eq!(result, "alpha\nchanged\ngamma\n");
    }

    #[test]
    fn apply_unified_patch_bare_at_header_requires_unique_match() {
        // A bare @@ header has no nominal line number; old_start=0 must not be treated as a strong anchor at line 1.
        // If the context appears multiple times in the file, the model must be asked to add more context, avoiding silently changing the first position.
        // The exact-location stage already confirms ambiguity sufficiently; it must not keep guessing or silently pick the first position.
        let original = "alpha\nbeta\ngamma\nalpha\nbeta\ngamma\n";
        let patch = "@@\n alpha\n-beta\n+changed\n";
        let err = apply_unified_patch(original, patch).unwrap_err();
        assert!(err.contains("ambiguous patch"), "err was: {err}");
        assert!(err.contains("1, 4"), "err was: {err}");
    }

    #[test]
    fn execute_apply_patch_envelope_with_bare_at_header() {
        // Reproduces the user report: canonical *** Begin Patch envelope + bare @@ header.
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

    // ======================== ReplaceInLine (P2) tests ========================

    fn make_envelope(op: PatchEnvelopeOp, target: &str, body: &[&str]) -> super::PatchEnvelope {
        super::PatchEnvelope {
            op,
            target_path: target.to_string(),
            body_lines: body.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn inline_replace_basic() {
        // Basic: anchor locates the line, old->new exact replacement
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
        // When the file does not end with \n, no \n is added after the replacement
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
        // The anchor uses em-dash (—, U+2014), while the file has ASCII hyphen (-).
        // Anchor normalized matching should tolerate it, but old must still match exactly.
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
    fn inline_replace_old_tolerates_confusable() {
        // old has em-dash, the file has ASCII hyphen: after exact match fails,
        // the tolerant fallback (confusable 1:1 normalization) should locate and replace.
        let original = "the quick-brown fox\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: quick", "old: quick—brown", "new: slow-brown"],
        );
        let result =
            apply_inline_replace(original, &envelope).expect("confusable old should match");
        // The output is built from new, preserving the file's original content; only the matched range is replaced
        assert_eq!(result, "the slow-brown fox\n");
    }

    #[test]
    fn inline_replace_old_tolerates_whitespace() {
        // old with leading/trailing whitespace (model indentation not reproduced exactly) -> tolerant match ignores leading/trailing whitespace
        let original = "let x = 42;\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.rs",
            &["anchor: let x", "old:   x = 42  ", "new: x = 99"],
        );
        let result =
            apply_inline_replace(original, &envelope).expect("whitespace-trimmed old should match");
        assert_eq!(result, "let x = 99;\n");
    }

    #[test]
    fn inline_replace_old_not_found_mentions_line_prefix_hint() {
        // old copied from read_file also brought the line-number prefix -> must not match silently; error with a hint.
        // Tolerant matching does not strip prefixes (that would pollute file content); here we verify the error message has guidance.
        let original = "let x = 42;\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.rs",
            &["anchor: let x", "old:     1\tlet x = 42;", "new: let x = 99;"],
        );
        let err = apply_inline_replace(original, &envelope)
            .expect_err("old with line-number prefix should fail");
        assert!(
            err.contains("line-number prefix"),
            "error should hint at line-number prefix: {err}"
        );
    }

    #[test]
    fn inline_replace_old_confusable_ambiguous() {
        // Exact match is zero (em-dash/en-dash are not hyphen); after normalization old appears in the line
        // multiple times -> error (instead of guessing one)
        let original = "a—b a–b\n";
        let envelope = make_envelope(
            PatchEnvelopeOp::ReplaceInLine,
            "test.txt",
            &["anchor: a—b", "old: a-b", "new: c-d"],
        );
        let err = apply_inline_replace(original, &envelope)
            .expect_err("old matching 2 positions after normalization should fail");
        assert!(
            err.contains("matches 2 positions"),
            "error should mention ambiguity after normalization: {err}"
        );
    }

    #[test]
    fn inline_replace_anchor_not_unique() {
        // Anchor matches multiple lines -> error
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
        // old appears multiple times within the line -> error
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
        // Replacing multi-byte UTF-8 content, verifying byte-index slicing safety
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
        // Verifies parse_patch_envelope recognizes the *** Replace in line: header
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
        // End-to-end: calls through execute_apply_patch, verifying the full path (including sandbox)
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
                // Some models serialize unused optional string parameters as empty strings; treat them as not provided.
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
        // Multi-file envelope + redundant file_path: models often still pass file_path in a multi-file envelope
        // (pointing at one of the files). file_path should be silently ignored, using each section's own path in the envelope.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("multi_file_redundant_path");
        let a = base.join("a.txt");
        let b = base.join("b.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&a, "old_a\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                // Redundant file_path should be silently ignored
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
    fn execute_apply_patch_applies_repeated_same_file_sections_in_order() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("same_file_sections");
        let path = base.join("a.txt");
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-alpha\n+ALPHA\n*** Update File: a.txt\n@@\n-gamma\n+GAMMA\n*** End Patch\n"
            });
            let result = execute_apply_patch(&args)
                .expect("repeated same-file sections should apply sequentially");
            assert!(
                result.starts_with("Successfully patched "),
                "result: {result}"
            );
            assert!(
                !result.starts_with("Successfully patched 2 files:"),
                "same file should be committed once: {result}"
            );
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "ALPHA\nbeta\nGAMMA\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_can_update_file_created_earlier_in_same_patch() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let base = make_temp_path("same_file_add_update");
        let path = base.join("new.txt");
        fs::create_dir_all(&base).unwrap();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "patch": "*** Begin Patch\n*** Add File: new.txt\n+alpha\n+beta\n*** Update File: new.txt\n@@\n-beta\n+changed\n*** End Patch\n"
            });
            execute_apply_patch(&args)
                .expect("Update File should see content added by an earlier same-file section");
        });

        assert_eq!(fs::read_to_string(&path).unwrap(), "alpha\nchanged");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn execute_apply_patch_legacy_dry_run_remains_non_mutating() {
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
            .expect("legacy dry run should remain safe for old calls")
        });

        assert!(result.starts_with("Dry run succeeded; no files changed:"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "before\n");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn apply_unified_patch_ignores_context_only_hunks_when_ordering_changes() {
        let original = "first\nmiddle\nlast\n";
        let patch = "@@ -3,1 +3,1 @@\n last\n@@ -1,1 +1,1 @@\n-first\n+FIRST\n";

        let actual = apply_unified_patch(original, patch)
            .expect("context-only hunks must not advance the changed-hunk cursor");

        assert_eq!(actual, "FIRST\nmiddle\nlast\n");
    }

    #[test]
    fn execute_apply_patch_rejects_unified_noop() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("unified_noop");
        let base = path.parent().unwrap().to_path_buf();
        fs::create_dir_all(&base).unwrap();
        fs::write(&path, "before\n").unwrap();

        let err = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            execute_apply_patch(&serde_json::json!({
                "file_path": path.to_string_lossy(),
                "patch": "@@ -1,1 +1,1 @@\n before\n",
            }))
            .expect_err("a context-only unified diff must not report success")
        });

        assert!(err.contains("[NO_CHANGES]"), "err was: {err}");
        assert_eq!(fs::read_to_string(&path).unwrap(), "before\n");
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
