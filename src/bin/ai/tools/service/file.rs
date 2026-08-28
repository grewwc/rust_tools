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
    render_line_excerpt_from_char(content, start, end, max_chars, with_line_numbers, 0)
}

fn render_line_excerpt_from_char(
    content: &str,
    start: usize,
    end: usize,
    max_chars: Option<usize>,
    with_line_numbers: bool,
    first_line_char_offset: usize,
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
            format!("{:>6}\t{}", start + idx + 1, line)
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
                    format!("{:>6}\t", start + idx + 1).chars().count()
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

    // Numbered lines by default (grounding axis); with use_line_numbers=false, return raw content,
    // so the result can feed directly into apply_patch as exact source text or into other tools.
    let use_line_numbers = should_render_read_file_line_numbers(
        store.path(),
        args["use_line_numbers"].as_bool().unwrap_or(true),
    );
    let excerpt = render_line_excerpt_from_char(
        &content,
        start,
        end,
        Some(MAX_READ_FILE_RESULT_CHARS),
        use_line_numbers,
        char_offset,
    );
    // Compute the continuation anchor from the actually rendered line count: the char cap may truncate before the requested `end`,
    // and reusing `end` would make the continuation offset skip unshown lines (silent data loss).
    let shown_end = start + excerpt.shown_lines;
    let size_capped = shown_end < end || excerpt.truncated_mid_line;
    let rendered = append_truncation_notice(
        excerpt.text,
        start,
        shown_end,
        total,
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
    total: usize,
    size_capped: bool,
    truncated_mid_line: bool,
    next_char_offset: Option<usize>,
) -> String {
    let remaining = total.saturating_sub(shown_end);
    if remaining == 0 && !truncated_mid_line {
        return rendered;
    }
    if !rendered.is_empty() {
        rendered.push('\n');
    }
    let continue_offset = shown_end + 1;
    if size_capped {
        if truncated_mid_line {
            if let Some(next_char_offset) = next_char_offset {
                rendered.push_str(&format!(
                    "... [truncated: output capped at {MAX_READ_FILE_RESULT_CHARS} chars; line {shown_end} was truncated mid-line. Continue that same line with offset={shown_end}, char_offset={next_char_offset}, limit=1.]"
                ));
            }
        } else {
            rendered.push_str(&format!(
                "... [truncated: output capped at {MAX_READ_FILE_RESULT_CHARS} chars; showing lines {}-{} of {}; {} more line(s) not shown. Continue with offset={} to read the rest.]",
                start + 1,
                shown_end,
                total,
                remaining,
                continue_offset
            ));
        }
    } else {
        rendered.push_str(&format!(
            "... [truncated: showing lines {}-{} of {}; {} more line(s) not shown. Continue with offset={} to read the rest.]",
            start + 1,
            shown_end,
            total,
            remaining,
            continue_offset
        ));
    }
    rendered
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
        assert!(output.contains("40 more line"), "output: {output}");
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
        assert!(output.contains("output capped at"), "output: {output}");
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
        let marker = "Continue that same line with offset=";
        let idx = output.find(marker).expect("mid-line continue present");
        let rest = &output[idx + marker.len()..];
        // Note format: {offset}, char_offset={char_offset}, limit=1.
        let offset_digits = rest.chars().take_while(|c| c.is_ascii_digit()).collect::<String>();
        let continue_offset: usize = offset_digits.parse().expect("offset is a number");
        let char_marker = "char_offset=";
        let char_idx =
            rest.find(char_marker).expect("char_offset present") + char_marker.len();
        let next_char_offset: usize = rest[char_idx..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .expect("char_offset is a number");
        assert!(continue_offset > 1, "offset should advance: {continue_offset}");
        assert!(next_char_offset > 0, "char_offset should advance: {next_char_offset}");

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
}
