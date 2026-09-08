use std::path::PathBuf;

use serde_json::Value;

use crate::ai::tools::common::ToolStreamWriter;
use crate::ai::tools::storage::file_store::{FileStore, is_read_file_overflow_artifact};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RenderedLineExcerpt {
    pub(crate) text: String,
    pub(crate) shown_lines: usize,
    pub(crate) truncated_mid_line: bool,
    pub(crate) next_char_offset: Option<usize>,
}

pub(crate) fn render_line_excerpt(
    content: &str,
    start: usize,
    end: usize,
    max_chars: Option<usize>,
    with_line_numbers: bool,
) -> RenderedLineExcerpt {
    render_line_excerpt_from_char(content, start, end, max_chars, with_line_numbers, 0, start)
}

fn render_line_excerpt_from_char(
    content: &str,
    start: usize,
    end: usize,
    max_chars: Option<usize>,
    with_line_numbers: bool,
    first_line_char_offset: usize,
    number_base: usize,
) -> RenderedLineExcerpt {
    let lines: Vec<&str> = content.lines().collect();
    let mut text = String::new();
    let mut shown_lines = 0usize;
    let mut truncated_mid_line = false;
    let mut next_char_offset = None;
    // Running char count of `text`; replaces re-scanning the whole rendered
    // buffer with chars().count() on every line (was O(rendered) per line).
    let mut text_chars = 0usize;

    for (idx, line) in lines[start..end].iter().enumerate() {
        let line_char_offset = if idx == 0 { first_line_char_offset } else { 0 };
        let line: String = line.chars().skip(line_char_offset).collect();
        let rendered = if with_line_numbers {
            format!("{:>6}\t{}", number_base + idx + 1, line)
        } else {
            line
        };
        let rendered_chars = rendered.chars().count();
        if let Some(limit) = max_chars {
            if !text.is_empty() {
                if text_chars.saturating_add(1) >= limit {
                    break;
                }
                text.push('\n');
                text_chars += 1;
            }

            let remaining = limit.saturating_sub(text_chars);
            if rendered_chars > remaining {
                if remaining == 0 {
                    break;
                }
                text.push_str(&truncate_chars_to_limit(&rendered, remaining));
                shown_lines += 1;
                truncated_mid_line = true;
                let prefix_chars = if with_line_numbers {
                    format!("{:>6}\t", number_base + idx + 1).chars().count()
                } else {
                    0
                };
                next_char_offset =
                    Some(line_char_offset + remaining.saturating_sub(prefix_chars + 1));
                break;
            }
        } else if !text.is_empty() {
            text.push('\n');
            text_chars += 1;
        }

        text.push_str(&rendered);
        text_chars += rendered_chars;
        shown_lines += 1;
    }

    RenderedLineExcerpt {
        text,
        shown_lines,
        truncated_mid_line,
        next_char_offset,
    }
}

/// Detects whether content carries the outermost line-number prefix of read_file output, and if so strips only that layer.
///
/// read_file output format is `{:>6}\t{content}` (6-char right-aligned line number + tab + content).
/// When such output is written to an archive file and read back via read_file, the prefixes nest,
/// e.g. `     1\t     1\t原始内容`. Each re-read strips only the outermost layer produced by the current tool;
/// it must not keep guessing about inner layers: the original file content itself may legitimately start with the same format.
fn strip_rendered_line_number_layer(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return content.to_string();
    }
    // Quick check: the first line must start with spaces+digits+tab, otherwise it cannot be read_file format.
    // Uses split_once to avoid fully parsing non-matching content.
    if read_file_number_prefix_rest(lines[0]).is_none() {
        return content.to_string();
    }
    // Strip at most once per line, removing exactly the display layer rendered by read_file when the snapshot was saved.
    // Do not loop-strip, to avoid dropping legitimate `     7\tvalue` lines from the original file as display layers.
    let stripped: Vec<String> = lines
        .iter()
        .map(|line| {
            read_file_number_prefix_rest(line)
                .unwrap_or(line)
                .to_string()
        })
        .collect();
    stripped.join("\n")
}

fn read_file_number_prefix_rest(line: &str) -> Option<&str> {
    let (num_part, rest) = line.split_once('\t')?;
    // `render_line_excerpt` renders line numbers with `{:>6}\t`; the `1\tfoo`/`12\tfoo` shapes
    // common in plain TSV/logs must not be misdetected and stripped.
    if num_part.chars().count() < 6 {
        return None;
    }
    if !num_part.chars().all(|ch| ch == ' ' || ch.is_ascii_digit()) {
        return None;
    }
    let trimmed = num_part.trim();
    if trimmed.is_empty() || !trimmed.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some(rest)
}

/// Strips one display line-number layer, only for the session archive `overflow-history.md`.
///
/// `read_file` overflow snapshots store the full rendered result of the earlier tool call. On re-read, keep the
/// original line numbers from that result instead of stripping and renumbering by asset-relative lines; otherwise it is
/// no longer an exact snapshot, and genuine `use_line_numbers=false` content like `123\t...` gets misread as a display layer.
fn should_strip_rendered_line_number_layer(path: &std::path::Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("overflow-history.md")
        && path
            .parent()
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".assets"))
}

/// Historical `read_file` snapshots already embed the display form chosen at call time. To avoid adding another
/// asset-relative line-number layer, keep their original rendering when re-reading; plain files still honor the caller's switch.
fn should_render_read_file_line_numbers(path: &std::path::Path, requested: bool) -> bool {
    requested && !is_read_file_overflow_artifact(path)
}

fn truncate_chars_to_limit(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    if max_chars <= 1 {
        return text.chars().take(max_chars).collect();
    }
    let mut out = text
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

fn image_read_redirect_message(file_path: &str) -> String {
    format!(
        "Image file detected at {}. This read request has been auto-upgraded to image-input semantics (same intent as attaching it with `-f`). Continue by analyzing the image directly instead of reading it as UTF-8 text.",
        file_path
    )
}

fn resolve_file_path_arg(args: &Value) -> Result<&str, String> {
    args.get("file_path")
        .or_else(|| args.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing file_path".to_string())
}

/// Normalizes file_path when temp=true: rejects absolute paths and out-of-bounds parent references, keeping only the file name.
///
/// This avoids `PathBuf::join` replacing the whole base when given an absolute path, writing the file into the
/// project source tree while still registering it in the temp registry. The model only passes a relative file name (e.g. `script.py`).
fn temp_file_name(file_path: &str) -> Result<std::path::PathBuf, String> {
    let p = std::path::Path::new(file_path);
    if p.is_absolute() {
        return Err(format!(
            "temp=true requires a relative filename, got absolute path: {file_path}"
        ));
    }
    // Keep only the file name, discarding any directory parts, so the target always lands inside the per-session temp dir.
    let name = p
        .file_name()
        .ok_or_else(|| format!("temp=true requires a file name, got: {file_path}"))?;
    Ok(std::path::PathBuf::from(name))
}

fn emit_stream_line(on_chunk: &mut ToolStreamWriter<'_>, line: &str) {
    let mut rendered = line.to_string();
    rendered.push('\n');
    on_chunk(rendered.as_bytes());
}

pub(crate) fn execute_read_file(args: &Value) -> Result<String, String> {
    let file_path = resolve_file_path_arg(args)?;
    let store = FileStore::new(PathBuf::from(file_path));
    store.validate_read_access().map_err(|e| e.to_string())?;
    store.ensure_exists().map_err(|e| e.to_string())?;
    if crate::ai::files::is_image_path(file_path) {
        return Ok(image_read_redirect_message(file_path));
    }

    let offset = args["offset"].as_u64().unwrap_or(1) as usize;
    let char_offset = args["char_offset"].as_u64().unwrap_or(0) as usize;
    let limit = args["limit"].as_u64().unwrap_or(1000) as usize;
    // Numbered lines by default (grounding axis); with use_line_numbers=false, return raw content,
    // so the result can feed directly into apply_patch as exact source text or into other tools.
    let use_line_numbers = should_render_read_file_line_numbers(
        store.path(),
        args["use_line_numbers"].as_bool().unwrap_or(true),
    );

    // Files above FULL_READ_MAX_BYTES are streamed through the bounded windowed reader (chunked
    // reads under the shared kernel lock) instead of being read whole: a huge file read under the
    // global lock would stall every other kernel tenant for the whole file. If stat fails (e.g. a
    // sensitive-path denial), fall through to the historical whole-file path so its error/trace
    // behavior stays unchanged.
    if let Ok(stat) = store.stat() {
        if stat.size > FULL_READ_MAX_BYTES {
            return execute_read_file_windowed(&store, offset, char_offset, limit, use_line_numbers);
        }
    }

    let raw_content = store.read_to_string().map_err(|e| e.to_string())?;
    let content = if should_strip_rendered_line_number_layer(store.path()) {
        strip_rendered_line_number_layer(&raw_content)
    } else {
        raw_content
    };
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    // An out-of-bounds offset or an empty file must not silently return an empty string: the model misreads "empty result"
    // as "file is empty" and draws wrong conclusions (a past session re-read an archive file as empty and retried repeatedly).
    // Here the two abnormal cases are distinguished explicitly; the normal paging path keeps its original behavior.
    if total == 0 {
        return Ok(
            "... [note: file is empty (0 lines); read_file returned no lines. \
Verify the path or use execute_command to inspect the file.]"
                .to_string(),
        );
    }
    if offset > total {
        return Ok(format!(
            "... [note: offset {offset} is beyond the end of file (total: {total} lines); \
no lines shown. Continue with offset=1 to read from the start, or offset={total} \
to read the last line.]"
        ));
    }
    let start = offset.saturating_sub(1);
    let end = (start + limit).min(total);
    let first_line_chars = lines[start].chars().count();
    if char_offset > first_line_chars {
        return Err(format!(
            "char_offset {char_offset} exceeds line {offset} length ({first_line_chars} chars)"
        ));
    }

    let excerpt = render_line_excerpt_from_char(
        &content,
        start,
        end,
        Some(MAX_READ_FILE_RESULT_CHARS),
        use_line_numbers,
        char_offset,
        start,
    );
    // Compute the continuation anchor from the actually rendered line count: the char cap may truncate before the requested `end`,
    // and reusing `end` would make the continuation offset skip unshown lines (silent data loss).
    let shown_end = start + excerpt.shown_lines;
    let size_capped = shown_end < end || excerpt.truncated_mid_line;
    let rendered = append_truncation_notice(
        excerpt.text,
        start,
        shown_end,
        Some(total),
        size_capped,
        excerpt.truncated_mid_line,
        excerpt.next_char_offset,
    );
    Ok(rendered)
}

/// Hard character cap for a single read_file result.
///
/// Line paging (offset/limit) only bounds "line count", not "character volume": minified JS/JSON and
/// pathological single-line files with hundreds of thousands of chars can produce MB-scale results from a 1-line read; raw content
/// entering messages would blow up the context instantly. This cap clamps a single read result to the inline-budget scale (64K);
/// the excess is fetched via the unified offset continuation contract instead of being silently dropped.
const MAX_READ_FILE_RESULT_CHARS: usize = 64_000;

/// Per-line keep cap for the windowed reader: a window never retains more than this many chars of
/// any single line. A multi-GB single-line file (minified JS, compressed JSON, single-line logs)
/// must not be loaded into memory just because the requested window contains that line; characters
/// past the cap are counted and dropped, and `char_offset` pages the rest of the line on demand.
const WINDOW_LINE_CAP_CHARS: usize = MAX_READ_FILE_RESULT_CHARS;

/// Total content cap for a windowed read. Once this many chars are retained, further window lines
/// are counted (for exact totals) but not kept; the renderer's own output cap is
/// `MAX_READ_FILE_RESULT_CHARS`, so content beyond this could never be displayed.
const WINDOW_CAP_CHARS: usize = MAX_READ_FILE_RESULT_CHARS;

/// Files whose byte size is at or below this limit keep the historical whole-file read (exact line
/// totals and truncation wording). Larger files use the bounded windowed reader in
/// `execute_read_file_windowed`, so a huge read no longer holds the shared kernel lock for the
/// whole file nor loads it entirely into memory.
const FULL_READ_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// Chunk size for the windowed reader: bounds how long each read holds the global kernel lock
/// (~1 MiB of page-cache IO is sub-millisecond on SSD), so one big read can no longer stall every
/// other tenant of the shared kernel.
const WINDOW_READ_CHUNK_BYTES: usize = 1024 * 1024;

/// When this read did not reach end-of-file, append an explicit note telling the model that more lines remain
/// and how to continue reading. Prevents the model from misreading a "truncated result" as the "complete file".
///
/// `shown_end` must be the **actually rendered line number** (`start + shown_lines`), not derived from the requested
/// `limit` — otherwise, when the char cap truncates early, the continuation `offset` points to the wrong place and middle
/// lines get silently skipped. `size_capped` means this truncation was triggered by the char cap (not exhausted lines);
/// `truncated_mid_line` means the last line was cut mid-line due to size (the rest is discarded).
fn append_truncation_notice(
    mut rendered: String,
    start: usize,
    shown_end: usize,
    total: Option<usize>,
    size_capped: bool,
    truncated_mid_line: bool,
    next_char_offset: Option<usize>,
) -> String {
    let remaining = total.map(|t| t.saturating_sub(shown_end));
    // When the exact total is known and everything was shown, there is nothing to continue with.
    // (The windowed reader reports `total: None` only when it knows more lines exist past the
    // scanned window, in which case a notice is always required.)
    if let Some(rem) = remaining {
        if rem == 0 && !truncated_mid_line {
            return rendered;
        }
    }
    if !rendered.is_empty() {
        rendered.push('\n');
    }
    let continue_offset = shown_end + 1;
    if size_capped {
        if truncated_mid_line {
            if let Some(next_char_offset) = next_char_offset {
                rendered.push_str(&format!(
                    "... [truncated: capped at {MAX_READ_FILE_RESULT_CHARS} chars; line {shown_end} truncated mid-line. Continue same line: offset={shown_end}, char_offset={next_char_offset}, limit=1.]"
                ));
            }
        } else if let Some(total) = total {
            let rem = total.saturating_sub(shown_end);
            rendered.push_str(&format!(
                "... [truncated: capped at {MAX_READ_FILE_RESULT_CHARS} chars; lines {}-{} of {}; {} more not shown. Continue: offset={}]",
                start + 1,
                shown_end,
                total,
                rem,
                continue_offset
            ));
        } else {
            rendered.push_str(&format!(
                "... [truncated: capped at {MAX_READ_FILE_RESULT_CHARS} chars; lines {}-{} shown; file continues, line count unknown. Continue: offset={}]",
                start + 1,
                shown_end,
                continue_offset
            ));
        }
    } else if let Some(total) = total {
        let rem = total.saturating_sub(shown_end);
        rendered.push_str(&format!(
            "... [truncated: lines {}-{} of {}; {} more not shown. Continue: offset={}]",
            start + 1,
            shown_end,
            total,
            rem,
            continue_offset
        ));
    } else {
        rendered.push_str(&format!(
            "... [truncated: lines {}-{} shown; file continues, line count unknown. Continue: offset={}]",
            start + 1,
            shown_end,
            continue_offset
        ));
    }
    rendered
}

/// Result of a bounded windowed read (`read_window_lines`).
struct WindowedRead {
    /// Raw window lines (no display prefixes), joined by '\n'.
    content: String,
    /// Exact line count of the whole file when the scan reached end-of-file; `None` when the file
    /// continues past the window (exact count unknown).
    total_lines: Option<usize>,
    /// Number of logical lines retained in the window.
    line_count: usize,
    /// The last line of `content` was truncated at the keep cap (`WINDOW_LINE_CAP_CHARS`): its full
    /// text is longer than what was retained, so the renderer must report a mid-line truncation
    /// anchor (`char_offset` continuation) even when its own char cap did not trigger.
    truncated_line: bool,
}

/// Ensures the string representation preserves a terminal blank logical row.
///
/// `WindowedRead` joins line text with `\n`, so a final blank row has no characters of its own
/// and `str::lines()` would omit it. Adding one delimiter lets the renderer recover the stored
/// logical line count. This must also run after archive-prefix stripping because that path joins
/// its stripped rows in the same way.
fn preserve_window_line_count(content: &mut String, line_count: usize) {
    let represented_lines = content.lines().count();
    if represented_lines < line_count {
        debug_assert_eq!(represented_lines + 1, line_count);
        content.push('\n');
    }
}

/// Bounded, windowed read for files larger than `FULL_READ_MAX_BYTES`.
///
/// Reads the requested line window `[offset, offset+limit)` in bounded chunks (each chunk at most
/// `WINDOW_READ_CHUNK_BYTES`), never loading the whole file into memory and never holding the
/// shared kernel lock for more than one chunk at a time. Rendering and notices mirror the
/// whole-file path; the only difference is that, when the scan stops before end-of-file, exact
/// line totals in the truncation notice are replaced by an explicit "file continues" phrase (the
/// continuation `offset` is unaffected, so paging keeps working exactly as before).
fn execute_read_file_windowed(
    store: &FileStore,
    offset: usize,
    char_offset: usize,
    limit: usize,
    use_line_numbers: bool,
) -> Result<String, String> {
    let start_abs = offset.saturating_sub(1);
    let window = read_window_lines(store, offset, limit, char_offset)?;

    // Mirror the whole-file path for the abnormal cases (empty file / out-of-bounds offset). The
    // windowed reader only reports `total_lines = Some` when it scanned to end-of-file, so the
    // exact totals in these messages are always known.
    if window.line_count == 0 {
        match window.total_lines {
            Some(0) => {
                return Ok(
                    "... [note: file is empty (0 lines); read_file returned no lines. \
Verify the path or use execute_command to inspect the file.]"
                        .to_string(),
                );
            }
            Some(total) => {
                return Ok(format!(
                    "... [note: offset {offset} is beyond the end of file (total: {total} lines); \
no lines shown. Continue with offset=1 to read from the start, or offset={total} \
to read the last line.]"
                ));
            }
            // limit == 0 (no lines requested) with more content: fall through to an empty render so
            // the continuation notice still tells the model how to proceed.
            None => {}
        }
    }

    let window_lines = window.line_count;
    let mut window_content = window.content;
    preserve_window_line_count(&mut window_content, window_lines);
    // The continuation anchor for a truncated window line is a character offset into the raw file
    // line, so it must come from the unstripped content (stripping only affects archive re-reads
    // and is line-local, so line boundaries are unchanged).
    let raw_last_line_chars = window_content
        .lines()
        .last()
        .map(|line| line.chars().count())
        .unwrap_or(0);
    // Strip the display line-number layer of session archive files (line-local, never changes line
    // boundaries), evaluating the same whole-file gate the whole-file path uses.
    if should_strip_rendered_line_number_layer(store.path()) && !window_content.is_empty() {
        window_content = strip_window_line_number_layer(store, &window_content)?;
    }
    preserve_window_line_count(&mut window_content, window_lines);
    // `char_offset` bounds are validated inside `read_window_lines`, which knows the window's
    // first line length exactly (a skip that outruns the line is an error there).

    let excerpt = render_line_excerpt_from_char(
        &window_content,
        0,
        window_lines,
        Some(MAX_READ_FILE_RESULT_CHARS),
        use_line_numbers,
        // The windowed reader already skipped `char_offset` chars of the window's first line, so
        // the renderer must not skip again; its `next_char_offset` is re-based to the full line
        // below when the truncated line is the window's first line.
        0,
        start_abs,
    );
    let mut truncated_mid_line = excerpt.truncated_mid_line;
    let mut next_char_offset = excerpt.next_char_offset;
    if truncated_mid_line {
        // The renderer truncated inside a line. Its anchor is relative to the current render
        // start; when that line is the window's first line, re-base it to the full line because
        // the windowed reader already consumed `char_offset` chars.
        if excerpt.shown_lines == 1 {
            next_char_offset = next_char_offset.map(|n| n + char_offset);
        }
    } else if window.truncated_line && excerpt.shown_lines == window_lines {
        // The window's last line was cut at the keep cap but the renderer output it in full (its
        // content fit the render cap): the model still needs the continuation anchor to page the
        // rest of that line via char_offset.
        truncated_mid_line = true;
        let is_first = window_lines <= 1;
        next_char_offset = Some(if is_first {
            char_offset + raw_last_line_chars
        } else {
            raw_last_line_chars
        });
    }
    // Compute the continuation anchor from the actually rendered line count (see the whole-file
    // path for why `shown_end` must come from the renderer, not from the requested window size).
    let shown_end = start_abs + excerpt.shown_lines;
    let size_capped = shown_end < start_abs + window_lines || truncated_mid_line;
    let rendered = append_truncation_notice(
        excerpt.text,
        start_abs,
        shown_end,
        window.total_lines,
        size_capped,
        truncated_mid_line,
        next_char_offset,
    );
    Ok(rendered)
}

/// Where the tentative trailing '\r' of the current unterminated line (a potential chunk-split
/// '\r\n' terminator) was accumulated. Recording the exact spot at accumulation time makes
/// consuming it later unambiguous even when the line exceeds the keep cap — there the tail of
/// `buf` is content, not the line's last char, and `extra` may count chars that were never
/// buffered, so inferring the spot from `buf`/`extra` state would be wrong.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum PendingCr {
    /// No tentative '\r' to consume.
    #[default]
    None,
    /// The '\r' fell inside the `char_offset`-skipped prefix or a line outside the retained
    /// window: never buffered, nothing to remove.
    Dropped,
    /// The '\r' is the last char of `buf`.
    InBuf,
    /// The '\r' was only counted in `extra`.
    InExtra,
}

/// Accumulates a bounded prefix of a single (possibly very long) line for the windowed reader.
///
/// The windowed reader never keeps more than `WINDOW_LINE_CAP_CHARS` characters of any one line:
/// a multi-GB single-line file (minified JS, compressed JSON, single-line logs) must not be loaded
/// into memory just because the requested window contains that line. Characters past the cap are
/// counted (`extra`) but dropped, and the `char_offset` continuation anchor lets the model page
/// the rest of the line on demand.
#[derive(Default)]
struct LineAccum {
    /// Kept prefix of the current line (at most `WINDOW_LINE_CAP_CHARS` chars).
    buf: String,
    /// Number of chars in `buf`.
    kept: usize,
    /// Chars of this line dropped past the keep cap (the line is longer than the kept prefix).
    extra: usize,
    /// Chars of the window's first line still to skip (`char_offset` continuation into the line).
    skip: usize,
    /// The current line has content in the file (used for exact line counts at EOF).
    started: bool,
    /// This line is part of the kept window (it began accumulating before the window filled).
    in_window_line: bool,
    /// Where the current (unterminated) line's trailing '\r' was accumulated, if any: a '\r\n'
    /// terminator split across a chunk boundary must remove exactly this '\r' once its '\n'
    /// arrives (at EOF the '\r' stays content, matching `str::lines()`).
    pending_cr: PendingCr,
}

/// Consumes the '\r' remembered via `LineAccum::pending_cr` once the '\n' of a chunk-split
/// '\r\n' arrives. The removal is exact because the accumulation site recorded where the '\r'
/// went; `buf`/`extra` have not changed since (no char was accumulated in between).
fn consume_pending_cr(acc: &mut LineAccum, window_chars: &mut usize) {
    match acc.pending_cr {
        PendingCr::InBuf => {
            // The '\r' is the last char pushed and nothing followed it.
            acc.buf.pop();
            acc.kept -= 1;
            *window_chars -= 1;
        }
        PendingCr::InExtra => acc.extra -= 1,
        PendingCr::Dropped | PendingCr::None => {}
    }
}

/// Feeds one fragment of the current line into `acc`, applying the window's first-line
/// `char_offset` skip and the per-line keep cap. `pending_first` is consumed when the window's
/// first line begins accumulating. Dropped characters of a retained line still count toward
/// `extra` so truncation is detected across chunk boundaries. Lines outside the retained window
/// only have their existence remembered for exact EOF line counts.
fn line_accumulate_fragment(
    acc: &mut LineAccum,
    frag: &str,
    in_window: bool,
    window_chars: &mut usize,
    pending_first: &mut bool,
    char_offset: usize,
    window_cap: usize,
) {
    if !in_window {
        // Before the window, remember only whether this line exists for exact EOF counts.
        if !frag.is_empty() {
            acc.started = true;
        }
        return;
    }
    if *pending_first {
        acc.skip = char_offset;
        *pending_first = false;
    }
    acc.in_window_line = true;
    if frag.is_empty() {
        return; // empty line ("\n" or "\r\n"): belongs to the window but has no content
    }
    acc.started = true;
    for ch in frag.chars() {
        if acc.skip > 0 {
            acc.skip -= 1;
            continue;
        }
        if acc.kept < WINDOW_LINE_CAP_CHARS && *window_chars < window_cap {
            acc.buf.push(ch);
            acc.kept += 1;
            *window_chars += 1;
        } else {
            acc.extra += 1;
        }
    }
}

/// Streams a file in bounded chunks and returns the requested line window `[offset, offset+limit)`
/// as a plain string.
///
/// Line semantics mirror `str::lines()` on the whole file: `\n` terminates every line (including
/// empty ones), a trailing `\r` before `\n` is consumed (CRLF), and an unterminated final fragment
/// at EOF counts as one last line. Each `FileStore::read_range` call holds the shared kernel lock
/// for at most `WINDOW_READ_CHUNK_BYTES` of IO, so even a multi-GB file can be paged without
/// stalling other kernel tenants. Memory stays bounded regardless of line length: at most
/// `WINDOW_CAP_CHARS` of window content plus one line prefix (`WINDOW_LINE_CAP_CHARS`) are ever
/// retained, and the window's first line may be continued via `char_offset` (chars already
/// consumed by the caller are skipped here, so the returned content starts at that offset).
/// Scanning stops as soon as the window is complete (plus one probe chunk to learn whether more
/// lines exist); an out-of-bounds offset scans to EOF so the exact-total diagnostic stays true.
fn read_window_lines(
    store: &FileStore,
    offset: usize,
    limit: usize,
    char_offset: usize,
) -> Result<WindowedRead, String> {
    let start_abs = offset.saturating_sub(1);
    if limit == 0 {
        // No lines requested; probe once so the notice can distinguish end-of-file from "more".
        let probe = store
            .read_range(0, WINDOW_READ_CHUNK_BYTES)
            .map_err(|e| e.to_string())?;
        let total_lines = if probe.content.is_empty() { Some(0) } else { None };
        return Ok(WindowedRead {
            content: String::new(),
            total_lines,
            line_count: 0,
            truncated_line: false,
        });
    }

    let mut need_skip = start_abs; // lines still to skip before the window
    let mut window = String::new();
    let mut window_lines = 0usize; // lines actually kept in `window`
    let mut window_chars = 0usize; // chars retained in `window` (bounded by WINDOW_CAP_CHARS)
    let mut total_seen = 0usize; // terminated lines seen (kept or not)
    let mut in_window = start_abs == 0; // offset == 1: the window starts at the first line
    let mut pending_first = in_window; // the next kept line is the window's first (carries char_offset)
    let mut acc = LineAccum::default();
    let mut truncated_line = false; // `window`'s last line was truncated at the keep cap
    let mut window_filled = false; // keep counting after the cap, without retaining later lines
    let mut pos: u64 = 0;
    let mut eof = false;

    while !eof {
        let range = store
            .read_range(pos, WINDOW_READ_CHUNK_BYTES)
            .map_err(|e| e.to_string())?;
        pos = range.next_offset;
        eof = range.hit_eof;
        if range.content.is_empty() {
            // Only possible at end-of-file: a bounded read returns empty content exactly then.
            break;
        }
        // A line may span a chunk boundary; feed each chunk through the bounded line state instead
        // of concatenating unterminated fragments (which would grow with the line length).
        let chunk = range.content;
        let bytes = chunk.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // Reserve the separator before retaining this line's content. Empty lines consume
            // a separator too, so even an unbounded requested line count cannot grow the window.
            let window_cap = WINDOW_CAP_CHARS - usize::from(window_lines > 0);
            let Some(rel) = bytes[i..].iter().position(|&b| b == b'\n') else {
                // The rest of the chunk belongs to the current (unterminated) line.
                let rest = &chunk[i..];
                // A trailing '\r' may still be the first half of a '\r\n' whose '\n' arrives in
                // the next chunk: hold it out of the fragment and record exactly where it went
                // (see `PendingCr`) so `consume_pending_cr` can remove precisely that '\r'.
                let prev = rest.strip_suffix('\r');
                line_accumulate_fragment(
                    &mut acc,
                    prev.unwrap_or(rest),
                    in_window && !window_filled,
                    &mut window_chars,
                    &mut pending_first,
                    char_offset,
                    window_cap,
                );
                match prev {
                    None => acc.pending_cr = PendingCr::None,
                    Some(_) => {
                        // Mirror `line_accumulate_fragment`'s accounting for the held-out '\r'
                        // itself (it is the next char after `prev`).
                        if !in_window || window_filled {
                            // Outside the retained window, remember '\r' only for line counting.
                            acc.started = true;
                            acc.pending_cr = PendingCr::Dropped;
                        } else {
                            if pending_first {
                                acc.skip = char_offset;
                                pending_first = false;
                            }
                            acc.in_window_line = true;
                            acc.started = true;
                            if acc.skip > 0 {
                                acc.skip -= 1;
                                acc.pending_cr = PendingCr::Dropped;
                            } else if acc.kept < WINDOW_LINE_CAP_CHARS
                                && window_chars < window_cap
                            {
                                acc.buf.push('\r');
                                acc.kept += 1;
                                window_chars += 1;
                                acc.pending_cr = PendingCr::InBuf;
                            } else {
                                acc.extra += 1;
                                acc.pending_cr = PendingCr::InExtra;
                            }
                        }
                    }
                }
                break;
            };
            let nl = i + rel;
            let mut frag = &chunk[i..nl];
            // CRLF: a trailing '\r' is the line terminator, not content (matches str::lines()).
            if frag.ends_with('\r') {
                frag = &frag[..frag.len() - 1];
            } else if frag.is_empty() && acc.pending_cr != PendingCr::None {
                // The '\r' of a '\r\n' split across a chunk boundary was accumulated with the
                // previous chunk's remainder; consume it now that the '\n' is here.
                consume_pending_cr(&mut acc, &mut window_chars);
            }
            line_accumulate_fragment(
                &mut acc,
                frag,
                in_window && !window_filled,
                &mut window_chars,
                &mut pending_first,
                char_offset,
                window_cap,
            );
            total_seen += 1;
            if acc.in_window_line {
                if acc.skip > 0 {
                    // char_offset ran past the end of the window's first line.
                    return Err(format!(
                        "char_offset {char_offset} exceeds line {offset} length"
                    ));
                }
                if window_lines > 0 {
                    window.push('\n');
                    window_chars += 1;
                }
                window.push_str(&acc.buf);
                window_lines += 1;
                if acc.extra > 0 {
                    truncated_line = true;
                }
                window_filled = window_chars >= WINDOW_CAP_CHARS || truncated_line;
                if window_lines >= limit {
                    // Window complete: stop scanning, then probe whether the file continues so the
                    // truncation notice can distinguish "rest of file" from "end of file".
                    // `i` is the line's start, not its '\n' (which sits at `nl`): only a terminator
                    // with bytes after it means the chunk continues, otherwise `eof`/probe below
                    // decide whether the file really ends here.
                    let more_in_chunk = nl + 1 < bytes.len();
                    let total_lines = if more_in_chunk {
                        None
                    } else if eof {
                        Some(total_seen)
                    } else {
                        match store.read_range(pos, WINDOW_READ_CHUNK_BYTES) {
                            Ok(range) if range.content.is_empty() => Some(total_seen),
                            _ => None,
                        }
                    };
                    return Ok(WindowedRead {
                        content: window,
                        total_lines,
                        line_count: window_lines,
                        truncated_line,
                    });
                }
            }
            acc = LineAccum::default();
            if !in_window && need_skip > 0 {
                need_skip -= 1;
                if need_skip == 0 {
                    in_window = true; // the next terminated line starts the window
                    pending_first = true;
                }
            }
            i = nl + 1;
        }
    }

    // End of file reached. If we are still skipping, the requested offset is beyond the file's end.
    if need_skip > 0 {
        let total = total_seen + usize::from(acc.started);
        return Ok(WindowedRead {
            content: String::new(),
            total_lines: Some(total),
            line_count: 0,
            truncated_line: false,
        });
    }
    // The file ends inside the window: append the final unterminated line.
    if acc.in_window_line {
        if acc.skip > 0 {
            // char_offset ran past the end of the window's first line.
            return Err(format!(
                "char_offset {char_offset} exceeds line {offset} length"
            ));
        }
        if window_lines > 0 {
            window.push('\n');
            window_chars += 1;
        }
        window.push_str(&acc.buf);
        window_lines += 1;
        if acc.extra > 0 {
            truncated_line = true;
        }
    }
    debug_assert!(window_chars <= WINDOW_CAP_CHARS);
    let total = total_seen + usize::from(acc.started || acc.in_window_line);
    Ok(WindowedRead {
        content: window,
        total_lines: Some(total),
        line_count: window_lines,
        truncated_line,
    })
}

/// Applies `strip_rendered_line_number_layer` semantics to a *window* of a large archive file.
///
/// The whole-file strip first checks the file's first line for the display prefix and then strips
/// that layer per line. For the windowed reader only the requested window is in memory, so the
/// gate is evaluated by reading the file's first line (one small bounded read) and the strip is
/// applied per window line. Stripping is line-local and never changes line boundaries, so the
/// result matches what the whole-file path would produce for the same window.
fn strip_window_line_number_layer(store: &FileStore, window_content: &str) -> Result<String, String> {
    // Evaluate the whole-file gate: only strip when the archive's first line carries the prefix.
    let gate = store
        .read_range(0, WINDOW_READ_CHUNK_BYTES)
        .map_err(|e| e.to_string())?;
    let first_line = gate.content.lines().next().unwrap_or("");
    if read_file_number_prefix_rest(first_line).is_none() {
        return Ok(window_content.to_string());
    }
    let stripped: Vec<String> = window_content
        .lines()
        .map(|line| read_file_number_prefix_rest(line).unwrap_or(line).to_string())
        .collect();
    Ok(stripped.join("\n"))
}

pub(crate) fn execute_write_file(args: &Value) -> Result<String, String> {
    let file_path = resolve_file_path_arg(args)?;
    let content = args["content"].as_str().ok_or("Missing content")?;
    let is_temp = args["temp"].as_bool().unwrap_or(false);

    let resolved_path = if is_temp {
        let name = temp_file_name(file_path)?;
        let temp_dir = crate::ai::driver::runtime_ctx::temp_dir()
            .map_err(|e| format!("Failed to create temp dir: {}", e))?;
        temp_dir.join(name)
    } else {
        PathBuf::from(file_path)
    };

    let store = FileStore::new(resolved_path);
    // Temp files live in runtime-controlled temp dirs (session assets or system temp), outside the
    // user's project space, so skip the sandbox write check (consistent with tool-overflow behavior).
    if !is_temp {
        store.validate_write_access().map_err(|e| e.to_string())?;
    }
    store.write_all(content).map_err(|e| e.to_string())?;

    // After a temp file write succeeds, register it in the persistent registry for audit tracking.
    if is_temp {
        let abs_path = store.path().display().to_string();
        super::super::storage::temp_registry::register(&abs_path)?;
    }

    Ok(format!("Successfully wrote to {}", store.path().display()))
}

pub(crate) fn execute_write_file_streaming(
    args: &Value,
    on_chunk: &mut ToolStreamWriter<'_>,
) -> Result<String, String> {
    let file_path = resolve_file_path_arg(args)?;
    let content = args["content"].as_str().ok_or("Missing content")?;
    let is_temp = args["temp"].as_bool().unwrap_or(false);
    let resolved_path = if is_temp {
        let name = temp_file_name(file_path)?;
        let temp_dir = crate::ai::driver::runtime_ctx::temp_dir()
            .map_err(|e| format!("Failed to create temp dir: {}", e))?;
        temp_dir.join(name)
    } else {
        PathBuf::from(file_path)
    };
    let store = FileStore::new(resolved_path);
    let target = store.path().display().to_string();

    emit_stream_line(on_chunk, &format!("target: {target}"));
    // Temp files live in runtime-controlled temp dirs, outside the user's project space; skip the sandbox write check.
    if !is_temp {
        emit_stream_line(on_chunk, "validating write access");
        store.validate_write_access().map_err(|e| e.to_string())?;
    }

    emit_stream_line(on_chunk, &format!("writing {} byte(s)", content.len()));
    store.write_all(content).map_err(|e| e.to_string())?;

    // After a temp file write succeeds, register it in the persistent registry for audit tracking.
    if is_temp {
        let abs_path = store.path().display().to_string();
        super::super::storage::temp_registry::register(&abs_path)?;
    }

    let result = format!("Successfully wrote to {}", store.path().display());
    emit_stream_line(on_chunk, &result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::tools::storage::temp_registry;
    use std::fs;

    fn make_temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_tools_test_{}_{}", name, uuid::Uuid::new_v4()));
        path
    }

    #[test]
    fn test_write_and_read_file_roundtrip() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("roundtrip");
        let content = "Hello, integration test!\nLine 2\nLine 3";
        let base = path.parent().unwrap().to_path_buf();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base, || {
            let write_args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "content": content
            });
            let write_result = execute_write_file(&write_args);
            assert!(write_result.is_ok(), "write failed: {:?}", write_result);

            let read_args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "offset": 1,
                "limit": 100
            });
            let read_result = execute_read_file(&read_args);
            assert!(read_result.is_ok(), "read failed: {:?}", read_result);

            let output = read_result.unwrap();
            assert!(output.contains("Hello, integration test!"));
            assert!(output.contains("Line 2"));
            assert!(output.contains("Line 3"));
        });

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_can_continue_inside_one_very_long_line() {
        let path = make_temp_path("single_long_line");
        let content = format!(
            "{}END_MARKER",
            "x".repeat(MAX_READ_FILE_RESULT_CHARS + 2_000)
        );
        fs::write(&path, content).unwrap();

        let first = execute_read_file(&serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 1
        }))
        .unwrap();
        let marker = "char_offset=";
        let start = first.find(marker).expect("char continuation present") + marker.len();
        let next_char_offset: usize = first[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .unwrap();

        let second = execute_read_file(&serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "char_offset": next_char_offset,
            "limit": 1
        }))
        .unwrap();
        assert!(second.contains("END_MARKER"), "{second}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_offset_beyond_eof_returns_diagnostic() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("offset_beyond_eof");
        fs::write(&path, "line1\nline2\nline3\n").unwrap();
        let base = path.parent().unwrap().to_path_buf();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base, || {
            let read_args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "offset": 99,
                "limit": 10
            });
            let read_result = execute_read_file(&read_args);
            assert!(read_result.is_ok(), "read failed: {:?}", read_result);
            let output = read_result.unwrap();
            // Before the fix, an out-of-bounds offset silently returned ""; the model misread that as "file is empty";
            // it must now return an explicit diagnostic that includes the total line count.
            assert!(output.contains("beyond the end of file"), "{output}");
            assert!(output.contains("total: 3"), "{output}");
        });

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_empty_file_returns_diagnostic() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("empty_file");
        fs::write(&path, "").unwrap();
        let base = path.parent().unwrap().to_path_buf();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base, || {
            let read_args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "offset": 1,
                "limit": 100
            });
            let read_result = execute_read_file(&read_args);
            assert!(read_result.is_ok(), "read failed: {:?}", read_result);
            let output = read_result.unwrap();
            assert!(output.contains("empty"), "{output}");
            assert!(!output.is_empty(), "must not silently return empty");
        });

        let _ = fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn test_read_file_follows_file_symlink() {
        let target = make_temp_path("symlink_target").with_extension("txt");
        let alias = make_temp_path("symlink_alias").with_extension("txt");
        fs::write(&target, "real file content\n").unwrap();
        std::os::unix::fs::symlink(&target, &alias).unwrap();

        let args = serde_json::json!({
            "file_path": alias.to_string_lossy(),
            "offset": 1,
            "limit": 10
        });
        let output = execute_read_file(&args).expect("read_file should follow file symlinks");

        assert!(output.contains("real file content"), "output: {output}");
        let _ = fs::remove_file(&alias);
        let _ = fs::remove_file(&target);
    }

    #[test]
    fn test_write_file_streaming_dispatch_emits_progress() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("streaming");
        let content = "Hello, streaming write!";
        let base = path.parent().unwrap().to_path_buf();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base, || {
            let args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "content": content
            });
            let mut streamed = Vec::new();
            let mut capture = |chunk: &[u8]| streamed.extend_from_slice(chunk);
            let result = crate::ai::tools::common::execute_tool_call_with_args_streaming(
                "call_write_file_streaming",
                "write_file",
                &args,
                &mut capture,
            )
            .expect("streaming write_file should succeed");

            let streamed = String::from_utf8(streamed).expect("streamed output must be utf-8");
            assert!(streamed.contains("target:"), "streamed: {streamed}");
            assert!(
                streamed.contains("validating write access"),
                "streamed: {streamed}"
            );
            assert!(streamed.contains("writing "), "streamed: {streamed}");
            assert!(
                streamed.contains(&format!("Successfully wrote to {}", path.display())),
                "streamed: {streamed}"
            );
            assert_eq!(
                result.content,
                format!("Successfully wrote to {}", path.display())
            );
        });

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_truncation_notice_when_limit_hit() {
        let path = make_temp_path("truncate");
        let content = (1..=50)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, &content).unwrap();

        let read_args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 10
        });
        let output = execute_read_file(&read_args).unwrap();
        assert!(output.contains("line10"), "output: {output}");
        assert!(!output.contains("line11"), "output: {output}");
        // On truncation, must note that more lines remain and how to continue reading.
        assert!(output.contains("truncated"), "output: {output}");
        assert!(output.contains("40 more not shown"), "output: {output}");
        assert!(output.contains("offset=11"), "output: {output}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_no_notice_when_fully_read() {
        let path = make_temp_path("full");
        fs::write(&path, "a\nb\nc").unwrap();

        let read_args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 100
        });
        let output = execute_read_file(&read_args).unwrap();
        assert!(!output.contains("truncated"), "output: {output}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_size_cap_uses_actual_shown_lines_for_continue_offset() {
        // Pathological file: very wide lines, far fewer lines than the requested limit, but char volume exceeds the hard cap.
        // Key regression point: the continuation offset in the truncation note must be based on the "actually rendered line count",
        // not the requested limit — otherwise middle lines get silently skipped.
        let path = make_temp_path("bigchars");
        let wide_line = "x".repeat(2_000);
        let content = (0..100)
            .map(|_| wide_line.clone())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, &content).unwrap();

        let read_args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 1000
        });
        let output = execute_read_file(&read_args).unwrap();

        // Must note "truncated due to size" and call out the character cap explicitly.
        assert!(output.contains("[truncated: capped at"), "output: {output}");
        assert!(output.contains("truncated"), "output: {output}");
        // Output must not exceed the hard cap by much (rendered line prefixes + note; a reasonable margin is allowed).
        assert!(
            output.chars().count() <= MAX_READ_FILE_RESULT_CHARS + 2_000,
            "output len {} exceeds cap",
            output.chars().count()
        );

        // Parse the continuation offset from the note and verify it points to "the line after the last actually displayed line",
        // and that continuing fetches the immediately following content (no skipped lines). Wide-line files that hit the char cap
        // are truncated "mid-line"; continuation resumes from the same line's breakpoint via offset + char_offset, never skipping ahead.
        let marker = "Continue same line: offset=";
        let idx = output.find(marker).expect("mid-line continue present");
        let rest = &output[idx + marker.len()..];
        // Note format: {offset}, char_offset={char_offset}, limit=1.
        let offset_digits = rest
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>();
        let continue_offset: usize = offset_digits.parse().expect("offset is a number");
        let char_marker = "char_offset=";
        let char_idx = rest.find(char_marker).expect("char_offset present") + char_marker.len();
        let next_char_offset: usize = rest[char_idx..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .expect("char_offset is a number");
        assert!(
            continue_offset > 1,
            "offset should advance: {continue_offset}"
        );
        assert!(
            next_char_offset > 0,
            "char_offset should advance: {next_char_offset}"
        );

        // Re-reading the same line from the breakpoint must render line number == continue_offset, proving no silent line skipping.
        let next_args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": continue_offset,
            "char_offset": next_char_offset,
            "limit": 1
        });
        let next = execute_read_file(&next_args).unwrap();
        let first_line_no: usize = next
            .lines()
            .next()
            .and_then(|l| l.split('\t').next())
            .and_then(|n| n.trim().parse().ok())
            .expect("first rendered line number");
        assert_eq!(
            first_line_no, continue_offset,
            "continue must resume the same line, not skip ahead"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_reads_last_line_without_trailing_newline() {
        // Without a trailing newline, the old implementation counted by '\n' and missed the last line.
        let path = make_temp_path("lastline");
        fs::write(&path, "first\nsecond\nthird").unwrap();

        let read_args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 100
        });
        let output = execute_read_file(&read_args).unwrap();
        assert!(output.contains("third"), "output: {output}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_raw_mode_returns_content_without_line_numbers() {
        // The original content itself contains lines like `  7\tvalue`: raw mode must return them verbatim,
        // with no line-number prefix and unaffected by strip logic (strip only applies to archive paths).
        let path = make_temp_path("raw");
        let content = "alpha\nbeta\n  7\tvalue\nlast";
        fs::write(&path, content).unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 100,
            "use_line_numbers": false
        });
        let output = execute_read_file(&args).unwrap();
        assert_eq!(output, content, "raw output: {output}");
        assert!(!output.contains("truncated"), "output: {output}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_raw_mode_preserves_paging_and_notice() {
        let path = make_temp_path("raw_page");
        fs::write(&path, "a\nb\nc\nd\ne").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 2,
            "limit": 2,
            "use_line_numbers": false
        });
        let output = execute_read_file(&args).unwrap();
        assert!(output.starts_with("b\nc"), "output: {output}");
        assert!(output.contains("truncated"), "output: {output}");
        assert!(output.contains("offset=4"), "output: {output}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_defaults_to_line_numbers() {
        // Default (use_line_numbers not passed) must match historical behavior: line-number prefix.
        let path = make_temp_path("default_ln");
        fs::write(&path, "alpha\nbeta").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 100
        });
        let output = execute_read_file(&args).unwrap();
        assert!(output.starts_with("     1\talpha"), "output: {output}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_respects_offset_limit() {
        let path = make_temp_path("lines");
        let lines: Vec<String> = (1..=20).map(|i| format!("line {}", i)).collect();
        let content = lines.join("\n");

        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &content).unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 5,
            "limit": 6
        });
        let result = execute_read_file(&args);
        assert!(result.is_ok(), "read failed: {:?}", result);

        let output = result.unwrap();
        assert!(output.contains("line 5"));
        assert!(output.contains("line 6"));
        assert!(output.contains("line 7"));
        assert!(output.contains("line 8"));
        assert!(output.contains("line 9"));
        assert!(output.contains("line 10"));
        assert!(!output.contains("line 11"));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_strip_rendered_line_number_layer_preserves_plain_tabular_content() {
        let content = "123\talpha\n124\tbeta";
        assert_eq!(strip_rendered_line_number_layer(content), content);
    }

    #[test]
    fn test_strip_rendered_line_number_layer_preserves_source_prefix() {
        // The outer `1` is the display line number added when read_file output was saved; the inner `7` is source content.
        let snapshot = "     1\t     7\tvalue";
        assert_eq!(strip_rendered_line_number_layer(snapshot), "     7\tvalue");
    }

    #[test]
    fn test_read_file_preserves_session_archive_rendering_rules() {
        let session_assets = make_temp_path("overflow_history_assets").with_extension("assets");
        fs::create_dir_all(&session_assets).unwrap();
        let overflow_history = session_assets.join("overflow-history.md");
        let tool_overflow_dir = session_assets.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_overflow_dir).unwrap();
        let read_file_snapshot = tool_overflow_dir.join("20260722T101112Z-read_file-abc123.txt");

        fs::write(&overflow_history, "     1\talpha\n     2\tbeta\n").unwrap();
        let read_args = serde_json::json!({
            "file_path": overflow_history.to_string_lossy(),
            "offset": 1,
            "limit": 100
        });
        let output = execute_read_file(&read_args).unwrap();
        assert!(output.contains("     1\talpha"), "output: {output}");
        assert!(output.contains("     2\tbeta"), "output: {output}");
        assert!(
            !output.contains("     1\t     1\talpha"),
            "output should not contain nested line numbers: {output}"
        );

        // read_file snapshots require authorization from the current driver session; context-free service unit tests
        // verify the re-rendering rules, and the allow/deny authorization boundary is covered by storage::file_store regression tests.
        let snapshot = "   120\talpha\n   121\tbeta\n";
        fs::write(&read_file_snapshot, snapshot).unwrap();
        assert!(!should_strip_rendered_line_number_layer(
            &read_file_snapshot
        ));
        assert!(!should_render_read_file_line_numbers(
            &read_file_snapshot,
            true
        ));
        let rendered = render_line_excerpt(
            &fs::read_to_string(&read_file_snapshot).unwrap(),
            0,
            2,
            None,
            should_render_read_file_line_numbers(&read_file_snapshot, true),
        );
        assert_eq!(rendered.text, "   120\talpha\n   121\tbeta");
        assert!(
            !rendered.text.contains("     1\t   120\talpha"),
            "snapshot must not gain a second, asset-relative line-number layer: {}",
            rendered.text
        );

        let _ = fs::remove_dir_all(&session_assets);
    }

    #[test]
    fn test_read_file_image_returns_redirect_message() {
        let path = make_temp_path("image");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"fake image bytes").unwrap();
        let path = path.with_extension("png");
        fs::write(&path, b"fake image bytes").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
        });
        let result = execute_read_file(&args).unwrap();
        assert!(result.contains("auto-upgraded to image-input semantics"));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_write_file_creates_parent_dirs() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let mut path = make_temp_path("nested");
        path.push("a");
        path.push("b");
        path.push("c");
        path.push("deep.txt");

        let content = "deeply nested content";
        let base = path
            .ancestors()
            .find(|candidate| {
                candidate.file_name().map_or(false, |name| {
                    name.to_string_lossy().starts_with("ai_tools_test_nested")
                })
            })
            .map(PathBuf::from)
            .unwrap_or_else(|| path.parent().unwrap().to_path_buf());

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let args = serde_json::json!({
                "file_path": path.to_string_lossy(),
                "content": content
            });
            let result = execute_write_file(&args);
            assert!(result.is_ok(), "write failed: {:?}", result);
        });

        assert!(path.exists(), "file should exist");
        let read_back = fs::read_to_string(&path).unwrap();
        assert_eq!(read_back, content);

        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn test_temp_file_name_rejects_absolute_path() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let abs = make_temp_path("abs_reject");

        let args = serde_json::json!({
            "file_path": abs.to_string_lossy(),
            "content": "x",
            "temp": true
        });
        let result = execute_write_file(&args);
        assert!(
            result.is_err(),
            "temp=true must reject absolute path, got: {:?}",
            result
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("relative filename"),
            "error should explain relative filename requirement: {err}"
        );
        assert!(!abs.exists(), "file must not be created at absolute path");
    }

    #[test]
    fn test_temp_file_name_strips_directory_components() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        // The model may pass "subdir/script.py"; only the file name should be kept, landing at the temp dir root.
        let args = serde_json::json!({
            "file_path": "subdir/script.py",
            "content": "print('hi')\n",
            "temp": true
        });
        let result = execute_write_file(&args);
        assert!(result.is_ok(), "write failed: {:?}", result);

        let temp_dir = crate::ai::driver::runtime_ctx::temp_dir().unwrap();
        let written = temp_dir.join("script.py");
        assert!(written.exists(), "file should exist at {written:?}");
        let read_back = fs::read_to_string(&written).unwrap();
        assert_eq!(read_back, "print('hi')\n");
        let _ = temp_registry::unregister(&written.display().to_string());
        let _ = fs::remove_file(&written);
    }

    #[test]
    fn test_write_file_temp_relative_filename_writes_to_temp_dir() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let args = serde_json::json!({
            "file_path": "fixture.json",
            "content": "{\"k\":1}",
            "temp": true
        });
        let result = execute_write_file(&args);
        assert!(result.is_ok(), "write failed: {:?}", result);

        let temp_dir = crate::ai::driver::runtime_ctx::temp_dir().unwrap();
        let written = temp_dir.join("fixture.json");
        assert!(written.exists(), "file should exist at {written:?}");
        let _ = temp_registry::unregister(&written.display().to_string());
        let _ = fs::remove_file(&written);
    }

    #[test]
    fn test_file_tools_accept_path_alias() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let path = make_temp_path("path_alias").with_extension("txt");
        let base = path.parent().unwrap().to_path_buf();

        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
            let write_args = serde_json::json!({
                "path": path.to_string_lossy(),
                "content": "line1\nline2\nline3"
            });
            execute_write_file(&write_args).expect("write_file should accept path alias");

            let read_args = serde_json::json!({
                "path": path.to_string_lossy(),
                "offset": 1,
                "limit": 10
            });
            let output = execute_read_file(&read_args).expect("read_file should accept path alias");
            assert!(output.contains("line1"), "output: {output}");
            assert!(output.contains("line3"), "output: {output}");

            let lines_args = serde_json::json!({
                "path": path.to_string_lossy(),
                "offset": 2,
                "limit": 1
            });
            let lines = execute_read_file(&lines_args).expect("read_file should accept path alias");
            assert!(lines.contains("line2"), "output: {lines}");
            assert!(!lines.contains("line3"), "output: {lines}");
        });

        let _ = fs::remove_file(&path);
    }

    /// Builds `count` lines of 17 bytes each ("0123456789abcdef\n"). 246723 lines ≈ 4.19 MiB,
    /// i.e. just below FULL_READ_MAX_BYTES (whole-file path); 246724 lines pushes just above it
    /// (windowed path).
    fn big_grid_content(count: usize) -> String {
        let mut s = String::with_capacity(count * 17);
        for _ in 0..count {
            s.push_str("0123456789abcdef\n");
        }
        s
    }

    #[test]
    fn test_read_file_whole_path_still_reports_exact_totals_below_limit() {
        let path = make_temp_path("grid_below");
        fs::write(&path, big_grid_content(246_723)).unwrap();
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 10,
            "use_line_numbers": true,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(
            output.contains("lines 1-10 of 246723; 246713 more not shown"),
            "output: {output}"
        );
        assert!(output.contains("offset=11"), "output: {output}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_windowed_big_grid_pages_with_bounded_notice() {
        let path = make_temp_path("grid_above");
        fs::write(&path, big_grid_content(246_724)).unwrap();
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 10,
            "use_line_numbers": true,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(output.starts_with("     1\t0123456789abcdef"), "output: {output}");
        assert!(output.contains("  10\t0123456789abcdef"), "output: {output}");
        // The window is fully rendered but the file continues: the notice must not claim an exact
        // total, but the continuation offset must be unaffected.
        assert!(
            output.contains("file continues, line count unknown"),
            "output: {output}"
        );
        assert!(output.contains("offset=11"), "output: {output}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_windowed_preserves_terminal_blank_rows_for_paging() {
        let path = make_temp_path("terminal_blank");
        let mut content = String::from("\nfirst\n\nvisible\n");
        content.push_str(&big_grid_content(246_724));
        fs::write(&path, &content).unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 1,
            "use_line_numbers": true,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(
            output.starts_with("     1\t\n"),
            "blank first row must render as line 1: {output:?}"
        );
        assert!(output.contains("lines 1-1 shown"), "output: {output}");
        assert!(output.contains("offset=2"), "output: {output}");

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 2,
            "limit": 2,
            "use_line_numbers": true,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(
            output.starts_with("     2\tfirst\n     3\t\n"),
            "terminal blank row must render as line 3: {output:?}"
        );
        assert!(output.contains("lines 2-3 shown"), "output: {output}");
        assert!(output.contains("offset=4"), "output: {output}");

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 4,
            "limit": 1,
            "use_line_numbers": true,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(
            output.starts_with("     4\tvisible"),
            "continuation offset must not skip the row after a blank line: {output}"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_windowed_deep_offset_renders_correct_lines() {
        let path = make_temp_path("grid_deep");
        fs::write(&path, big_grid_content(246_724)).unwrap();
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 100_000,
            "limit": 3,
            "use_line_numbers": true,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(output.starts_with("100000\t0123456789abcdef"), "output: {output}");
        assert!(output.contains("100001\t0123456789abcdef"), "output: {output}");
        assert!(output.contains("100002\t0123456789abcdef"), "output: {output}");
        assert!(output.contains("offset=100003"), "output: {output}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_windowed_oob_reports_exact_total() {
        let path = make_temp_path("grid_oob");
        fs::write(&path, big_grid_content(246_724)).unwrap();
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 300_000,
            "limit": 10,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(
            output.contains("offset 300000 is beyond the end of file (total: 246724 lines)"),
            "output: {output}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_windowed_multibyte_crlf_across_chunk_boundaries() {
        let path = make_temp_path("big_utf8");
        // "日本語\r\n" is 11 bytes per line; 400_000 lines ≈ 4.4 MiB > FULL_READ_MAX_BYTES. 1 MiB
        // chunks cut through both the multibyte chars and the CRLF pairs, exercising the tail
        // char-boundary trim and the CRLF normalization across chunk boundaries.
        let mut content = String::with_capacity(400_000 * 11);
        for _ in 0..400_000 {
            content.push_str("日本語\r\n");
        }
        fs::write(&path, &content).unwrap();
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 4,
            "use_line_numbers": true,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(output.starts_with("     1\t日本語"), "output: {output}");
        assert!(output.contains("     2\t日本語"), "output: {output}");
        assert!(output.contains("     4\t日本語"), "output: {output}");
        assert!(!output.contains('\r'), "CR must be consumed, got: {output}");
        assert!(
            output.contains("file continues, line count unknown"),
            "output: {output}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_windowed_strips_archive_layer() {
        // Strip only applies to the session archive `overflow-history.md` under a `*.assets` dir.
        let dir = std::env::temp_dir().join(format!(
            "ai_tools_test_assets_{}.assets",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overflow-history.md");
        // Numbered lines mirroring a rendered read_file snapshot; ~17 bytes per line × 450_000
        // lines ≈ 7 MiB, so the windowed path (and its strip gate) is exercised.
        let mut content = String::new();
        for i in 1..=450_000 {
            content.push_str(&format!("{:>6}\tvalue {i}\n", i));
        }
        fs::write(&path, &content).unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 2,
            "use_line_numbers": true,
        });
        let output = execute_read_file(&args).unwrap();
        // The embedded snapshot layer is stripped once, then the window is re-rendered with
        // absolute line numbers; for a well-formed snapshot this restores the original rendering
        // exactly, and must never nest a second `     1\t     1\t` layer.
        assert!(
            output.starts_with("     1\tvalue 1\n     2\tvalue 2"),
            "output: {output}"
        );
        assert!(!output.contains("     1\t     1\t"), "nested layer: {output}");

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 100_000,
            "limit": 2,
            "use_line_numbers": true,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(
            output.starts_with("100000\tvalue 100000\n100001\tvalue 100001"),
            "output: {output}"
        );
        assert!(
            output.contains("file continues, line count unknown"),
            "output: {output}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_read_file_windowed_single_huge_line_is_bounded_and_pages() {
        let path = make_temp_path("huge_single_line");
        // A 5 MiB single line (no newline) forces the windowed path; the reader must retain only
        // a bounded prefix of the line (never the whole file) and page the rest via char_offset.
        let line = "x".repeat(5 * 1024 * 1024);
        fs::write(&path, &line).unwrap();
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 1,
            "use_line_numbers": false,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(output.starts_with("xxxxx"), "output: {output}");
        assert!(
            output.chars().count() < MAX_READ_FILE_RESULT_CHARS + 2_000,
            "bounded output, got {} chars",
            output.chars().count()
        );
        assert!(
            output.contains("line 1 truncated mid-line"),
            "output: {output}"
        );
        assert!(
            output.contains(&format!("char_offset={MAX_READ_FILE_RESULT_CHARS}")),
            "output: {output}"
        );

        // Continue the same line from the reported char_offset.
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 1,
            "char_offset": MAX_READ_FILE_RESULT_CHARS as u64,
            "use_line_numbers": false,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(output.starts_with("xxxxx"), "output: {output}");
        assert!(
            output.chars().count() < MAX_READ_FILE_RESULT_CHARS + 2_000,
            "bounded continuation, got {} chars",
            output.chars().count()
        );
        assert!(
            output.contains(&format!(
                "char_offset={}",
                2 * MAX_READ_FILE_RESULT_CHARS
            )),
            "output: {output}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_windowed_single_huge_line_numbered_anchor() {
        let path = make_temp_path("huge_single_line_num");
        let line = "y".repeat(5 * 1024 * 1024);
        fs::write(&path, &line).unwrap();
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 1,
        });
        let output = execute_read_file(&args).unwrap();
        // With line numbers the 7-char prefix pushes the rendered line past the render cap, so the
        // renderer truncates and its anchor (re-based to the full line) is 64_000 - 7 - 1.
        assert!(
            output.contains(&format!(
                "char_offset={}",
                MAX_READ_FILE_RESULT_CHARS - 7 - 1
            )),
            "output: {output}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_windowed_caps_window_content_after_cap() {
        let path = make_temp_path("cap_window");
        // line1 30 KiB + line2 5 MiB, total > 4 MiB → windowed. The window's keep cap fills inside
        // line 2, so the returned window is bounded and line 2 is truncated mid-line.
        let mut content = String::new();
        content.push_str(&"a".repeat(30 * 1024));
        content.push('\n');
        content.push_str(&"b".repeat(5 * 1024 * 1024));
        content.push('\n');
        fs::write(&path, &content).unwrap();
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 2,
            "use_line_numbers": false,
        });
        let output = execute_read_file(&args).unwrap();
        assert!(
            output.starts_with(&"a".repeat(30 * 1024)),
            "output: {output}"
        );
        assert!(
            output.chars().count() < MAX_READ_FILE_RESULT_CHARS + 2_000,
            "bounded output, got {} chars",
            output.chars().count()
        );
        assert!(
            output.contains("line 2 truncated mid-line"),
            "output: {output}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_windowed_char_offset_past_line_end_errors() {
        let path = make_temp_path("char_offset_oob");
        fs::write(&path, big_grid_content(246_724)).unwrap();
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 1,
            "char_offset": 100, // line 1 has only 16 chars
        });
        let output = execute_read_file(&args);
        let err = output.unwrap_err();
        assert!(
            err.contains("char_offset 100 exceeds line 1 length"),
            "err: {err}"
        );
        let _ = fs::remove_file(&path);
    }

    fn assert_newline_heavy_read_is_bounded(prefix: &str) {
        let path = make_temp_path("newline_heavy_huge_limit");
        let mut content = String::from(prefix);
        content.push_str(&"\n".repeat(FULL_READ_MAX_BYTES as usize + 1));
        fs::write(&path, &content).unwrap();
        let store = FileStore::new(path.clone());
        let window = read_window_lines(&store, 1, usize::MAX, 0).unwrap();
        let output = execute_read_file(&serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": usize::MAX,
            "use_line_numbers": true,
        }))
        .unwrap();
        let _ = fs::remove_file(&path);

        assert!(
            output.chars().count() <= MAX_READ_FILE_RESULT_CHARS + 2_000,
            "rendered output must remain bounded"
        );
        let shown_lines = output.lines().filter(|line| line.contains('\t')).count();
        assert!(shown_lines > 0, "blank rows must still be rendered");
        assert!(
            output.contains(&format!("Continue: offset={}", shown_lines + 1)),
            "continuation must follow the last rendered blank row"
        );
        // The render cap alone is insufficient: separators retained before rendering must also
        // consume the window budget, even when almost every logical line has zero text chars.
        assert!(
            window.content.chars().count() <= WINDOW_CAP_CHARS,
            "newline-heavy window retained {} chars across {} lines; cap is {WINDOW_CAP_CHARS}",
            window.content.chars().count(),
            window.line_count
        );
    }

    #[test]
    fn test_read_file_windowed_all_newlines_huge_limit_is_bounded() {
        assert_newline_heavy_read_is_bounded("");
    }

    #[test]
    fn test_read_file_windowed_nearly_all_newlines_huge_limit_is_bounded() {
        assert_newline_heavy_read_is_bounded("x\n");
    }

    #[test]
    fn test_read_window_lines_reserves_separator_at_exact_cap() {
        let path = make_temp_path("separator_at_cap");
        let first = "a".repeat(WINDOW_CAP_CHARS - 1);
        fs::write(&path, format!("{first}\n尾\r\nnext")).unwrap();
        let store = FileStore::new(path.clone());
        let window = read_window_lines(&store, 1, 3, 0).unwrap();
        assert_eq!(window.content.chars().count(), WINDOW_CAP_CHARS);
        assert_eq!(window.content, format!("{first}\n"));
        assert_eq!(window.line_count, 2);
        assert_eq!(window.total_lines, Some(3));
        assert!(window.truncated_line);
        let resumed = read_window_lines(&store, 2, 2, 0).unwrap();
        assert_eq!(resumed.content, "尾\nnext");
        assert!(!resumed.truncated_line);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_window_lines_detects_extra_chars_after_exact_chunk_cap() {
        let path = make_temp_path("extra_after_chunk_cap");
        let padding = "z".repeat(WINDOW_READ_CHUNK_BYTES - WINDOW_CAP_CHARS - 1);
        let kept = "a".repeat(WINDOW_CAP_CHARS);
        fs::write(&path, format!("{padding}\n{kept}尾\nlast")).unwrap();
        let store = FileStore::new(path.clone());
        let window = read_window_lines(&store, 2, 2, 0).unwrap();
        assert_eq!(window.content, kept);
        assert_eq!(window.line_count, 1);
        assert_eq!(window.total_lines, Some(3));
        assert!(window.truncated_line);
        let resumed = read_window_lines(&store, 2, 2, WINDOW_CAP_CHARS).unwrap();
        assert_eq!(resumed.content, "尾\nlast");
        assert!(!resumed.truncated_line);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_window_lines_single_huge_line_keeps_bounded_prefix() {
        let path = make_temp_path("huge_single_unit");
        let line = "z".repeat(8 * 1024 * 1024);
        fs::write(&path, &line).unwrap();
        let store = FileStore::new(path.clone());
        let window = read_window_lines(&store, 1, 1, 0).unwrap();
        assert_eq!(window.total_lines, Some(1));
        assert_eq!(window.line_count, 1);
        assert!(window.truncated_line, "single huge line must be truncated");
        assert!(
            window.content.chars().count() <= MAX_READ_FILE_RESULT_CHARS,
            "bounded content, got {} chars",
            window.content.chars().count()
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_window_lines_counts_lines_past_window_cap() {
        // Once the window's content cap fills inside the huge first line, later lines are not kept
        // but must still be counted so the exact EOF total stays true.
        let path = make_temp_path("cap_lines");
        let mut content = String::new();
        content.push_str(&"a".repeat(5 * 1024 * 1024));
        content.push('\n');
        content.push_str("short2\nshort3"); // no trailing newline
        fs::write(&path, &content).unwrap();
        let store = FileStore::new(path.clone());
        let window = read_window_lines(&store, 1, 3, 0).unwrap();
        assert_eq!(window.total_lines, Some(3), "exact total must survive the cap");
        assert!(window.truncated_line);
        assert_eq!(window.line_count, 1, "only the huge first line is kept");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_window_lines_crlf_split_across_chunk_boundary() {
        // Engineer a '\r\n' whose '\r' is the very last byte of a read chunk and whose '\n' is the
        // first byte of the next chunk: the '\r' must not leak into the line content.
        let path = make_temp_path("crlf_split");
        let mut content = String::new();
        content.push_str("a\n");
        content.push_str(&"x".repeat(WINDOW_READ_CHUNK_BYTES - 5));
        content.push('\n');
        content.push_str("x\r"); // '\r' lands at byte WINDOW_READ_CHUNK_BYTES - 1
        content.push('\n'); // '\n' lands at byte WINDOW_READ_CHUNK_BYTES (next chunk's first byte)
        content.push_str("tail\n");
        fs::write(&path, &content).unwrap();
        let store = FileStore::new(path.clone());
        let window = read_window_lines(&store, 3, 2, 0).unwrap();
        assert_eq!(window.content, "x\ntail", "CRLF must not leave a '\\r' behind");
        assert_eq!(window.line_count, 2);
        assert_eq!(window.total_lines, Some(4));
        assert!(!window.content.contains('\r'));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_window_lines_crlf_split_beyond_keep_cap_keeps_content_cr() {
        // The line exceeds the keep cap and a *content* '\r' is the last char that fits in the
        // buffer, while the CRLF terminator's '\r' is split across the chunk boundary. Popping by
        // `buf.ends_with('\r')` would remove the content '\r' instead of the terminator's, so the
        // removal must be told exactly where that '\r' was accumulated.
        let path = make_temp_path("crlf_cap_split");
        // Pad line 1 so line 2's terminator '\r' lands on the last byte of the first chunk.
        let mut content = String::new();
        content.push_str(&"z".repeat(WINDOW_READ_CHUNK_BYTES - WINDOW_LINE_CAP_CHARS - 3));
        content.push('\n');
        content.push_str(&"a".repeat(WINDOW_LINE_CAP_CHARS - 1));
        content.push('\r'); // content '\r' fills the keep cap exactly (char no. WINDOW_LINE_CAP_CHARS)
        content.push('d'); // first char past the cap (extra = 1)
        content.push('\r'); // CRLF terminator's '\r': byte WINDOW_READ_CHUNK_BYTES - 1
        content.push('\n'); // byte WINDOW_READ_CHUNK_BYTES: first byte of the next chunk
        content.push_str(&"y".repeat(4 * 1024 * 1024)); // keep the file past FULL_READ_MAX_BYTES
        fs::write(&path, &content).unwrap();
        let store = FileStore::new(path.clone());
        let window = read_window_lines(&store, 2, 1, 0).unwrap();
        let mut want = "a".repeat(WINDOW_LINE_CAP_CHARS - 1);
        want.push('\r');
        assert_eq!(
            window.content, want,
            "content '\\r' at the cap must survive the CRLF split"
        );
        assert!(window.truncated_line, "line 2 is longer than the keep cap");
        assert_eq!(window.total_lines, None, "line 3 keeps the file going");
        let _ = fs::remove_file(&path);
    }
}
