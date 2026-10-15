use std::{fs, io, path::Path};

use crate::commonw::utils::expanduser;
use crate::strw::split::split_by_str_keep_quotes;

use super::types::FileParseResult;

const ATTACHMENT_INLINE_MAX_CHARS: usize = 12_000;
const ATTACHMENT_INLINE_MAX_LINES: usize = 240;

pub(super) fn parse_files(content: &str) -> FileParseResult {
    let files = split_by_str_keep_quotes(content, ",", "\"", false);
    let mut parsed = FileParseResult::default();
    for file in files {
        classify_file_reference(&mut parsed, file.trim());
    }
    parsed
}

pub(super) fn classify_file_reference(parsed: &mut FileParseResult, raw: &str) {
    let file = expanduser(raw.trim()).to_string();
    if file.is_empty() {
        return;
    }
    if parsed.text_files.iter().any(|candidate| candidate == &file)
        || parsed
            .image_files
            .iter()
            .any(|candidate| candidate == &file)
        || parsed
            .binary_files
            .iter()
            .any(|candidate| candidate == &file)
    {
        return;
    }
    if Path::new(&file).exists() && is_image_path(&file) {
        parsed.image_files.push(file);
    } else if fs::read_to_string(&file).is_ok() {
        parsed.text_files.push(file);
    } else if Path::new(&file).exists() {
        parsed.binary_files.push(file);
    }
}

pub(super) fn is_image_path(path: &str) -> bool {
    let Some(ext) = Path::new(path).extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "png" | "gif" | "webp" | "bmp" | "tif" | "tiff" | "ico" | "qoi" | "avif"
    )
}

pub(super) fn image_mime_type(path: &str) -> &'static str {
    let Some(ext) = Path::new(path).extension().and_then(|ext| ext.to_str()) else {
        return "image/jpeg";
    };
    match ext.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "ico" => "image/x-icon",
        "qoi" => "image/qoi",
        "avif" => "image/avif",
        _ => "image/jpeg",
    }
}

pub(super) fn text_file_contents(files: &[String]) -> io::Result<String> {
    let mut blocks = Vec::with_capacity(files.len());
    for file in files {
        blocks.push(render_text_attachment_block(file)?);
    }
    Ok(blocks.join("\n\n"))
}

fn render_text_attachment_block(file: &str) -> io::Result<String> {
    let content = fs::read_to_string(file)?;
    let total_lines = content.lines().count();
    let total_chars = content.chars().count();
    let mut out = format!("[Attached text file: {}]\n", file);

    if total_lines <= ATTACHMENT_INLINE_MAX_LINES && total_chars <= ATTACHMENT_INLINE_MAX_CHARS {
        out.push_str(&content);
        if !content.ends_with('\n') && !content.is_empty() {
            out.push('\n');
        }
        out.push_str("[/Attached text file]");
        return Ok(out);
    }

    let preview = crate::ai::tools::service::file::render_line_excerpt(
        &content,
        0,
        ATTACHMENT_INLINE_MAX_LINES.min(total_lines),
        Some(ATTACHMENT_INLINE_MAX_CHARS),
        true,
    );
    if !preview.text.is_empty() {
        out.push_str(&preview.text);
        if !preview.text.ends_with('\n') {
            out.push('\n');
        }
    }

    let next_offset = if preview.truncated_mid_line {
        preview.shown_lines.max(1)
    } else {
        preview.shown_lines.saturating_add(1).max(1)
    };
    out.push_str(&format!(
        "\n[Attachment preview only: showing lines 1-{} of {} ({} chars total). If more detail is needed, call read_file(file_path=\"{}\", offset={}, limit=200).]\n",
        preview.shown_lines.max(1),
        total_lines,
        total_chars,
        file,
        next_offset,
    ));
    out.push_str("[/Attached text file]");
    Ok(out)
}

/// Diff line tags rendered into recall anchors. Dropping the marker column makes a
/// preview untrustworthy: a context line's marker is a single space, so trimming it
/// renders pre-existing code exactly like part of the change.
const DIFF_ADDED_TAG: &str = "[+]";
const DIFF_REMOVED_TAG: &str = "[-]";
const DIFF_CONTEXT_TAG: &str = "[ctx]";

/// One rendered recall anchor, split so callers can dedupe on `prefix` and truncate
/// only the body.
pub(super) struct PreviewAnchor {
    /// `L<line>` for lines carrying a `read_file` line-number prefix, else `#<idx>`.
    /// `#<idx>` is a 0-based index into the archived output, not a line address:
    /// printing it as `L<idx>` invites citations that resolve against nothing.
    pub prefix: String,
    /// Line body with the `read_file` prefix and the diff marker removed.
    pub body: String,
}

/// Diff line tags for every line of `content`: `Some(tag)` only for lines inside a
/// hunk body (`[+]`/`[-]`/`[ctx]`), `None` for hunk headers and for anything outside a
/// hunk. Membership follows the line counts declared in the `@@` header — the diff's own
/// geometry — instead of a guess from the text, so a removed row whose content starts
/// with `-- `, a `\ No newline` note, or a trailing listing after the last hunk cannot
/// pass for part of the change.
pub(super) fn diff_line_tags(content: &str) -> Vec<Option<&'static str>> {
    let mut tags = Vec::new();
    // Remaining (old-side, new-side) body lines of the open hunk.
    let mut open: Option<(usize, usize)> = None;
    for line in content.lines() {
        // read_file output carries a `{:>6}\t` prefix; `header` is the trimmed text used
        // for both marker classification and header recognition.
        let trimmed = line.trim();
        let header = match split_line_number_prefix(trimmed) {
            Some((_, rest)) => rest.trim_start(),
            None => trimmed,
        };

        if let Some((old_left, new_left)) = open {
            // Classify on the trimmed text: an indented hunk body still starts with its
            // marker there, and the hunk's own counts decide where it ends.
            let (tag, used_old, used_new) = match header.as_bytes().first() {
                Some(b'+') => (Some(DIFF_ADDED_TAG), 0, 1),
                Some(b'-') => (Some(DIFF_REMOVED_TAG), 1, 0),
                // `\ No newline at end of file` annotates the previous line instead of
                // being one, so it consumes no budget.
                Some(b'\\') => (None, 0, 0),
                _ => (Some(DIFF_CONTEXT_TAG), 1, 1),
            };
            let left = (
                old_left.saturating_sub(used_old),
                new_left.saturating_sub(used_new),
            );
            open = (left != (0, 0)).then_some(left);
            tags.push(tag);
            continue;
        }

        if header.starts_with("@@") {
            // Unified (`@@`) and combined (`@@@`) hunk headers both open a hunk.
            open = Some(hunk_line_budget(header));
        }
        tags.push(None);
    }
    tags
}

/// Renders one preview line as an anchor. `tag` comes from `diff_line_tags`; for an
/// added/removed line the marker itself is dropped, so the `[+]`/`[-]` tag is the
/// only marker a reader sees. Whitespace-only lines still render, with an empty
/// body: probe sampling promises the first and last omitted lines.
pub(super) fn preview_anchor(idx: usize, line: &str, tag: Option<&'static str>) -> PreviewAnchor {
    let trimmed = line.trim();
    // read_file output carries a `{:>6}\t` line-number prefix; stripping it keeps the
    // real line number in the label so a long overflowed file stays locatable.
    let (label, marked) = match split_line_number_prefix(trimmed) {
        Some((no, rest)) => (format!("L{no}"), rest.trim()),
        None => (format!("#{idx}"), trimmed),
    };
    let body = match tag {
        // The marker sits at the start of the trimmed text, so an indented hunk body is
        // stripped correctly too; a tag whose marker is missing keeps the text unchanged.
        Some(DIFF_ADDED_TAG) => marked.strip_prefix('+').unwrap_or(marked),
        Some(DIFF_REMOVED_TAG) => marked.strip_prefix('-').unwrap_or(marked),
        _ => marked,
    };
    let body = body.trim();
    let prefix = match tag {
        Some(tag) if !body.is_empty() => format!("{label}: {tag} "),
        Some(tag) => format!("{label}: {tag}"),
        None => format!("{label}: "),
    };
    PreviewAnchor {
        prefix,
        body: body.to_string(),
    }
}

/// Body-line budget of a hunk header: (old-side lines, new-side lines) summed over the
/// header's ranges. A header without parsable ranges counts as unbounded, so such a hunk
/// ends at the next header instead of swallowing the rest of the payload.
fn hunk_line_budget(header: &str) -> (usize, usize) {
    let mut old = 0usize;
    let mut new = 0usize;
    let mut parsed = false;
    for token in header.split_whitespace() {
        let (side, rest) = if let Some(rest) = token.strip_prefix('+') {
            (&mut new, rest)
        } else if let Some(rest) = token.strip_prefix('-') {
            (&mut old, rest)
        } else {
            continue;
        };
        let count = match rest.split_once(',') {
            Some((_, count)) => count.parse().unwrap_or(1),
            None => 1,
        };
        *side = side.saturating_add(count);
        parsed = true;
    }
    if parsed {
        (old, new)
    } else {
        (usize::MAX, usize::MAX)
    }
}

/// Structural key-line predicate, applied to the marker-stripped body. Matching the
/// raw line never sees an added line (`+    /// x` does not start with `///`), so the
/// index filled up with unchanged context and hid what the change actually did.
fn is_key_line(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.starts_with("fn ")
        || lower.starts_with("pub fn ")
        || lower.starts_with("pub(crate) fn ")
        || lower.starts_with("pub(super) fn ")
        || lower.starts_with("async fn ")
        || lower.starts_with("pub async fn ")
        || lower.starts_with("impl ")
        || lower.starts_with("struct ")
        || lower.starts_with("pub struct ")
        || lower.starts_with("trait ")
        || lower.starts_with("enum ")
        || lower.starts_with("pub enum ")
        || lower.starts_with("mod ")
        || lower.starts_with("#[")
        || lower.starts_with("//!")
        || lower.starts_with("///")
        || lower.starts_with("class ")
        || lower.starts_with("def ")
        || lower.starts_with("func ")
        || lower.starts_with("interface ")
        || lower.starts_with("type ")
        || lower.starts_with("pub type ")
        || lower.starts_with("const ")
        || lower.starts_with("pub const ")
        || lower.starts_with("use ")
        || lower.contains("error")
        || lower.contains("failed")
        || lower.contains("panic")
        || lower.contains("exception")
        || lower.contains("timeout")
        || lower.contains("warning")
        || lower.contains("todo")
        || lower.contains("fixme")
        || lower.contains(": error")
        || lower.contains(": warning")
}

/// Extract structured key lines from text content to provide recall anchors for the
/// overflow stub.
///
/// Recognized line types (kept consistent with `line_trim_middle`'s mid-segment sampling):
/// - Rust/code structure: `fn`/`pub fn`/`impl`/`struct`/`trait`/`enum`/`#[`/`mod`
/// - Doc comments: `//!`/`///`
/// - Errors/warnings: `error`/`failed`/`panic`/`exception`/`timeout`/`warning`
/// - Markers: `TODO`/`FIXME`
///
/// Diff content keeps its marker as an explicit tag, so an unchanged context line is
/// never read as part of the change. Each line is truncated to 200 chars to keep the
/// stub compact. At most `max` lines are kept.
pub(super) fn extract_key_lines(content: &str, max: usize) -> Vec<String> {
    let tags = diff_line_tags(content);
    let mut result = Vec::with_capacity(max);
    for (idx, line) in content.lines().enumerate() {
        if result.len() >= max {
            break;
        }
        let anchor = preview_anchor(idx, line, tags.get(idx).copied().flatten());
        if !is_key_line(&anchor.body) {
            continue;
        }
        let body = if anchor.body.chars().count() > 200 {
            let kept: String = anchor.body.chars().take(200).collect();
            format!("{kept} …")
        } else {
            anchor.body
        };
        result.push(format!("{}{body}", anchor.prefix));
    }
    result
}

/// Parse read_file output's `{:>6}\t{content}` line-number prefix, returning
/// (real line number, body after the prefix). Returns None for non-matching formats
/// (e.g. plain command output), whose lines are then labeled by 0-based index.
pub(super) fn split_line_number_prefix(trimmed: &str) -> Option<(usize, &str)> {
    let bytes = trimmed.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i >= bytes.len() || bytes[i] != b'\t' {
        return None;
    }
    let no: usize = trimmed[..i].parse().ok()?;
    Some((no, &trimmed[i + 1..]))
}

#[cfg(test)]
#[path = "files_tests.rs"]
mod tests;
