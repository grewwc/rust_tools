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

    for (idx, line) in lines[start..end].iter().enumerate() {
        let line_char_offset = if idx == 0 { first_line_char_offset } else { 0 };
        let line: String = line.chars().skip(line_char_offset).collect();
        let rendered = if with_line_numbers {
            format!("{:>6}\t{}", start + idx + 1, line)
        } else {
            line
        };
        if let Some(limit) = max_chars {
            if !text.is_empty() {
                if text.chars().count().saturating_add(1) >= limit {
                    break;
                }
                text.push('\n');
            }

            let remaining = limit.saturating_sub(text.chars().count());
            if rendered.chars().count() > remaining {
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
        }

        text.push_str(&rendered);
        shown_lines += 1;
    }

    RenderedLineExcerpt {
        text,
        shown_lines,
        truncated_mid_line,
        next_char_offset,
    }
}

/// 检测内容是否带有 read_file 输出格式的最外层行号前缀，若是则只剥除这一层。
///
/// read_file 输出格式为 `{:>6}\t{content}`（6位右对齐行号 + tab + 内容）。
/// 当这种输出被写入归档文件后再次被 read_file 读取时，会出现多层嵌套
/// （如 `     1\t     1\t原始内容`）。每次回读只需去掉当前工具产生的最外层，
/// 不能继续猜测内层：原始文件内容本身也可能合法地以同一格式开头。
fn strip_rendered_line_number_layer(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return content.to_string();
    }
    // 快速判断：第一行必须以空格+数字+tab开头，否则不可能是 read_file 格式。
    // 使用 split_once 避免对非匹配内容做全量解析。
    if read_file_number_prefix_rest(lines[0]).is_none() {
        return content.to_string();
    }
    // 每行至多剥除一次，恰好移除保存快照时由 read_file 渲染的展示层。
    // 不循环剥离，避免将原始文件中合法的 `     7\tvalue` 当作展示层丢掉。
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
    // `render_line_excerpt` 使用 `{:>6}\t` 渲染行号；普通 TSV/日志里常见的
    // `1\tfoo`/`12\tfoo` 不应被误判并剥离。
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

/// 仅对会话归档的 `overflow-history.md` 剥离一层展示行号。
///
/// `read_file` 外溢快照保存的是先前工具调用的完整渲染结果。重读时应直接保留
/// 该结果里的原始行号，而不是剥离后按 asset 的相对行号重新编号；否则既不再是
/// 精确快照，也会把 `use_line_numbers=false` 的真实 `123\t...` 内容误当展示层。
fn should_strip_rendered_line_number_layer(path: &std::path::Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("overflow-history.md")
        && path
            .parent()
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".assets"))
}

/// 历史 `read_file` 快照已经包含调用当时选择的展示形式。为避免外层再添一层
/// asset 相对行号，回读快照时保持其原始渲染；普通文件仍遵从调用方的开关。
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

/// temp=true 时规范化 file_path：拒绝绝对路径与越界的父目录引用，只保留文件名。
///
/// 这样避免 `PathBuf::join` 遇到绝对路径时整体替换 base，把文件误写到项目源码
/// 目录却仍被注册进 temp registry。模型只需传相对文件名（如 `script.py`）。
fn temp_file_name(file_path: &str) -> Result<std::path::PathBuf, String> {
    let p = std::path::Path::new(file_path);
    if p.is_absolute() {
        return Err(format!(
            "temp=true requires a relative filename, got absolute path: {file_path}"
        ));
    }
    // 只取文件名，丢弃任何目录部分，确保落点始终在 per-session temp dir 内。
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
    // offset 越界或文件为空时不能静默返回空串：模型会把「空结果」误判为
    // 「文件为空」，进而得出错误结论（历史会话曾把归档文件读成空后反复重试）。
    // 这里显式区分两种异常情况，正常分页路径保持原有行为。
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

    // 默认带行号（grounding 轴）；use_line_numbers=false 时返回原始内容，
    // 便于把结果直接作为 apply_patch 的精确源文本或其他工具的输入。
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
    // 用实际渲染行数计算续读锚点：字符上限可能在请求的 `end` 之前就截断，
    // 若沿用 `end` 会让续读 offset 跳过未显示的行（静默丢数据）。
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

/// 单次 read_file 结果的字符硬上限。
///
/// 行分页（offset/limit）只约束"行数"，无法约束"字符量"：minified JS/JSON、
/// 单行几十万字符的病理文件即使只读 1 行也能产出 MB 级结果，raw 进入 messages
/// 会瞬间撑爆上下文。此上限把单条读取结果钳到与 inline 预算同量级（64K），
/// 超出部分通过统一的 offset 续读契约让模型分页取回，而不是静默丢弃。
const MAX_READ_FILE_RESULT_CHARS: usize = 64_000;

/// 当本次读取没有覆盖到文件末尾时，追加一条明确提示，告知模型文件仍有
/// 剩余行未显示以及如何继续读取。避免模型把"截断结果"误判为"完整文件"。
///
/// `shown_end` 必须是**实际渲染到的行号**（`start + shown_lines`），不能用请求的
/// `limit` 推算——否则字符上限提前截断时，续读 `offset` 会指向错误位置，导致中间
/// 若干行被静默跳过。`size_capped` 表示本次截断是由字符上限触发（而非行数用尽），
/// `truncated_mid_line` 表示最后一行因体积在行中被截断（其余部分已丢弃）。
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
    // temp 文件落在 runtime 控制的临时目录（session assets 或系统 temp），不属于
    // 用户项目空间，跳过沙箱写权限检查（与 tool-overflow 行为一致）。
    if !is_temp {
        store.validate_write_access().map_err(|e| e.to_string())?;
    }
    store.write_all(content).map_err(|e| e.to_string())?;

    // temp 文件写入成功后注册到持久化注册表，供审计跟踪。
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
    // temp 文件落在 runtime 控制的临时目录，不属于用户项目空间，跳过沙箱写权限检查。
    if !is_temp {
        emit_stream_line(on_chunk, "validating write access");
        store.validate_write_access().map_err(|e| e.to_string())?;
    }

    emit_stream_line(on_chunk, &format!("writing {} byte(s)", content.len()));
    store.write_all(content).map_err(|e| e.to_string())?;

    // temp 文件写入成功后注册到持久化注册表，供审计跟踪。
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
            // 修复前 offset 越界会静默返回 ""，模型误判为「文件为空」；
            // 现在必须返回带总行数的明确诊断。
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
        // 截断时必须提示还有剩余行以及如何继续读取。
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
        // 病理文件：每行很宽，行数远少于请求的 limit，但字符量超过硬上限。
        // 关键回归点：截断提示的续读 offset 必须基于"实际渲染的行数"，
        // 而不是请求的 limit——否则中间若干行会被静默跳过。
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

        // 必须提示"因体积截断"，且明确标注字符上限。
        assert!(output.contains("output capped at"), "output: {output}");
        assert!(output.contains("truncated"), "output: {output}");
        // 输出不得超过硬上限太多（渲染行前缀 + 提示，留合理余量）。
        assert!(
            output.chars().count() <= MAX_READ_FILE_RESULT_CHARS + 2_000,
            "output len {} exceeds cap",
            output.chars().count()
        );

        // 从提示里解析续读 offset，验证它指向"实际显示的最后一行的下一行"，
        // 且续读能拿到紧接着的内容（不跳行）。
        let marker = "Continue with offset=";
        let idx = output.find(marker).expect("continue offset present");
        let rest = &output[idx + marker.len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        let continue_offset: usize = digits.parse().expect("offset is a number");
        assert!(
            continue_offset > 1,
            "offset should advance: {continue_offset}"
        );

        // 用续读 offset 再读一次，第一行行号必须正好等于 continue_offset，
        // 证明没有静默跳过任何行。
        let next_args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": continue_offset,
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
            "continue offset must not skip lines"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_reads_last_line_without_trailing_newline() {
        // 文件末尾无换行符时，旧实现按 '\n' 计数会漏掉最后一行。
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
        // 原始内容本身包含形如 `  7\tvalue` 的行：raw 模式必须原样返回，
        // 不带行号前缀、也不受 strip 逻辑影响（strip 只作用于归档路径）。
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
        // 缺省（未传 use_line_numbers）时行为必须与历史一致：带行号前缀。
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
        // 外层 `1` 是保存 read_file 输出时产生的展示行号；内层 `7` 是源文件内容。
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

        // read_file 快照需要当前 driver 会话授权；无上下文的 service 单测验证其
        // 重渲染规则，正反授权边界由 storage::file_store 的回归测试覆盖。
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
        // 模型可能传 "subdir/script.py"；应只保留文件名，落在 temp dir 根下。
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
