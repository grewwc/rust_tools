use super::*;
use std::path::{Path, PathBuf};
use serde_json::Value;
use crate::ai::tools::storage::file_store::FileStore;
pub(crate) fn parse_unified_hunks(patch: &str) -> Result<Vec<UnifiedHunk>, String> {
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

pub(crate) fn optional_file_path_arg(args: &Value) -> Option<&str> {
    args.get("file_path")
        .or_else(|| args.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct UnifiedDiffHeaderTarget {
    pub(super) paths: Vec<String>,
    pub(super) deletes_file: bool,
}

/// Parses a git double-quoted path token. Besides common C escapes, it also accepts the
/// three-digit octal bytes used by git's quotePath, so that valid paths containing spaces or
/// escaped characters are not split on whitespace and written to the wrong target.
pub(crate) fn parse_git_path_token(input: &str) -> Option<(String, &str)> {
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

pub(crate) fn normalized_diff_path(path: &str) -> Option<String> {
    if path.is_empty() || path == "/dev/null" {
        return None;
    }
    let path = path
        .strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path);
    (!path.is_empty()).then(|| path.to_string())
}

pub(crate) fn diff_marker_path(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.starts_with('"') {
        return parse_git_path_token(raw).map(|(path, _)| path);
    }
    // `---`/`+++` paths allow spaces; only a TAB explicitly separates the optional timestamp.
    Some(raw.split('\t').next().unwrap_or(raw).trim().to_string())
}

pub(crate) fn record_diff_target(paths: &mut Vec<String>, path: Option<String>) {
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
pub(crate) fn parse_unified_diff_header_target(patch: &str) -> UnifiedDiffHeaderTarget {
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

pub(crate) fn file_path_from_unified_diff_header(patch: &str) -> Option<String> {
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
pub(crate) fn split_unified_diff_by_file(patch: &str) -> Result<Vec<(String, String)>, String> {
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

pub(crate) fn ensure_patch_target_matches(target_path: &Path, envelope_path: &str) -> Result<(), String> {
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

pub(crate) fn parse_patch_header(header: &str) -> Result<(PatchEnvelopeOp, &str), String> {
    if let Some(path) = header.strip_prefix("*** Update File: ") {
        Ok((PatchEnvelopeOp::Update, path.trim()))
    } else if let Some(path) = header.strip_prefix("*** Add File: ") {
        Ok((PatchEnvelopeOp::Add, path.trim()))
    } else if let Some(path) = header.strip_prefix("*** Delete File: ") {
        Ok((PatchEnvelopeOp::Delete, path.trim()))
    } else if let Some(path) = header.strip_prefix("*** Replace in line: ") {
        Ok((PatchEnvelopeOp::ReplaceInLine, path.trim()))
    } else {
        let kind = [
            "*** Update File",
            "*** Add File",
            "*** Delete File",
            "*** Replace in line",
        ]
        .iter()
        .copied()
        .find(|marker| header.starts_with(marker));
        if let Some(kind) = kind {
            return Err(format!(
                "invalid patch envelope: malformed section header `{kind}`; write it as \
                 `{kind}: <path>` (for example `{kind}: src/main.rs`)"
            ));
        }
        Err(
            "invalid patch envelope: expected `*** Update File:`, `*** Add File:`, \
             `*** Delete File:`, or `*** Replace in line:`"
                .to_string(),
        )
    }
}

pub(crate) fn is_patch_section_header(line: &str) -> bool {
    line.starts_with("*** Update File: ")
        || line.starts_with("*** Add File: ")
        || line.starts_with("*** Replace in line: ")
        || line.starts_with("*** Delete File: ")
}

/// Patch control markers that must start a top-level section. `is_patch_section_header` only
/// recognizes headers written as `*** <Kind>: <path>`, so a malformed variant (a missing path
/// or a missing colon, for example a bare `*** Replace in line:`) stays inside the body of the
/// enclosing section. Treated as body text such a line is folded into a context line, and the
/// envelope is then reported as producing no changes, hiding the real format error.
pub(crate) fn patch_section_marker_kind(line: &str) -> Option<&'static str> {
    const MARKERS: [&str; 7] = [
        "*** Begin Patch",
        "*** End Patch",
        "*** End of File",
        "*** Update File",
        "*** Add File",
        "*** Delete File",
        "*** Replace in line",
    ];
    MARKERS
        .iter()
        .copied()
        .find(|marker| line.starts_with(marker))
}

pub(crate) fn parse_patch_envelopes(patch: &str) -> Result<Option<Vec<PatchEnvelope>>, String> {
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

pub(crate) fn parse_patch_envelope(patch: &str) -> Result<Option<PatchEnvelope>, String> {
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

pub(crate) fn normalize_patch_envelope_body(envelope: &PatchEnvelope) -> Result<String, String> {
    if let Some(marker) = envelope
        .body_lines
        .iter()
        .find(|line| patch_section_marker_kind(line).is_some())
    {
        return Err(format!(
            "invalid patch: detected mixed patch formats. {marker:?} is a patch section marker, \
             but it appears inside the body of another section. A section header cannot be \
             nested: close the current section and start a new top-level one, for example \
             `*** Replace in line: <path>` followed by its `anchor:`, `old:`, and `new:` lines. \
             If this line is file content instead of a marker, prefix it with a space (context \
             line) or `+` (added line)."
        ));
    }
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

pub(crate) fn normalize_patch_envelope(path: &Path, envelope: &PatchEnvelope) -> Result<String, String> {
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
