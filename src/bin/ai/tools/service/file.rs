use std::path::PathBuf;

use serde_json::Value;

use crate::ai::tools::common::ToolStreamWriter;
use crate::ai::tools::storage::file_store::FileStore;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RenderedLineExcerpt {
    pub(crate) text: String,
    pub(crate) shown_lines: usize,
    pub(crate) truncated_mid_line: bool,
}

pub(crate) fn render_line_excerpt(
    content: &str,
    start: usize,
    end: usize,
    max_chars: Option<usize>,
) -> RenderedLineExcerpt {
    let lines: Vec<&str> = content.lines().collect();
    let mut text = String::new();
    let mut shown_lines = 0usize;
    let mut truncated_mid_line = false;

    for (idx, line) in lines[start..end].iter().enumerate() {
        let rendered = format!("{:>6}\t{}", start + idx + 1, line);
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
    }
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
    let limit = args["limit"].as_u64().unwrap_or(1000) as usize;
    let content = store.read_to_string().map_err(|e| e.to_string())?;
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = offset.saturating_sub(1).min(total);
    let end = (start + limit).min(total);

    let excerpt = render_line_excerpt(&content, start, end, Some(MAX_READ_FILE_RESULT_CHARS));
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
    );
    Ok(append_symbol_outline_if_useful(
        rendered, file_path, &content, start,
    ))
}

/// 为受支持的语言在读取结果末尾附加一段紧凑的符号大纲，让模型每次读文件都能
/// 获得结构化代码视图，而不必逐行 grep。不支持的语言或无符号时原样返回。
fn append_symbol_outline_if_useful(
    mut rendered: String,
    file_path: &str,
    content: &str,
    start: usize,
) -> String {
    // 仅在首块读取时附大纲，避免分页读取同一文件时把同一份 outline 反复塞回上下文。
    if start > 0 {
        return rendered;
    }
    const MAX_OUTLINE_SYMBOLS: usize = 60;
    if let Some(outline) = crate::ai::tools::ast_symbols::document_symbol_outline(
        file_path,
        content,
        MAX_OUTLINE_SYMBOLS,
    ) {
        if !rendered.is_empty() {
            rendered.push_str("\n\n");
        }
        rendered.push_str(&outline);
    }
    rendered
}

/// 单次 read_file / read_file_lines 结果的字符硬上限。
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
        rendered.push_str(&format!(
            "... [truncated: output capped at {MAX_READ_FILE_RESULT_CHARS} chars; showing lines {}-{} of {}; {} more line(s) not shown. Continue with offset={} to read the rest.]",
            start + 1,
            shown_end,
            total,
            remaining,
            continue_offset
        ));
        if truncated_mid_line {
            rendered.push_str(&format!(
                "\n... [note: line {shown_end} was truncated mid-line due to size; the remainder of that line is omitted.]"
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

    super::super::undo_tools::snapshot_file_before_write(&resolved_path.to_string_lossy());

    let store = FileStore::new(resolved_path);
    // temp 文件落在 runtime 控制的临时目录（session assets 或系统 temp），不属于
    // 用户项目空间，跳过沙箱写权限检查（与 tool-overflow 行为一致）。
    if !is_temp {
        store.validate_write_access().map_err(|e| e.to_string())?;
    }
    store.write_all(content).map_err(|e| e.to_string())?;

    // temp 文件写入成功后注册到持久化注册表，使其可被 delete_path 清理。
    if is_temp {
        let abs_path = store.path().display().to_string();
        super::super::storage::temp_registry::register(&abs_path)?;
    }

    super::super::undo_tools::commit_change_set(&format!("write_file: {}", store.path().display()));

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
    emit_stream_line(on_chunk, "snapshotting previous file state");
    super::super::undo_tools::snapshot_file_before_write(&target);

    // temp 文件落在 runtime 控制的临时目录，不属于用户项目空间，跳过沙箱写权限检查。
    if !is_temp {
        emit_stream_line(on_chunk, "validating write access");
        store.validate_write_access().map_err(|e| e.to_string())?;
    }

    emit_stream_line(on_chunk, &format!("writing {} byte(s)", content.len()));
    store.write_all(content).map_err(|e| e.to_string())?;

    // temp 文件写入成功后注册到持久化注册表，使其可被 delete_path 清理。
    if is_temp {
        let abs_path = store.path().display().to_string();
        super::super::storage::temp_registry::register(&abs_path)?;
    }

    super::super::undo_tools::commit_change_set(&format!("write_file: {}", file_path));

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
    fn test_read_file_lines_reads_last_line_without_trailing_newline() {
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
    fn test_read_file_lines_respects_offset_limit() {
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
    fn test_read_file_appends_symbol_outline_for_supported_language() {
        let path = make_temp_path("outline").with_extension("rs");
        let content = "fn alpha() {}\n\nstruct Beta {\n    x: i32,\n}\n\nfn gamma() {}\n";
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, content).unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 100
        });
        let output = execute_read_file(&args).unwrap();
        assert!(output.contains("Symbol outline"), "output: {output}");
        assert!(output.contains("alpha"), "output: {output}");
        assert!(output.contains("Beta"), "output: {output}");
        assert!(output.contains("gamma"), "output: {output}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_no_outline_for_unsupported_language() {
        let path = make_temp_path("plain").with_extension("txt");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "just some plain text\nno symbols here\n").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 100
        });
        let output = execute_read_file(&args).unwrap();
        assert!(!output.contains("Symbol outline"), "output: {output}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_read_file_lines_skips_outline_for_later_chunks() {
        let path = make_temp_path("outline_late").with_extension("rs");
        let content = "fn alpha() {}\n\nstruct Beta {\n    x: i32,\n}\n\nfn gamma() {}\n";
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, content).unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 3,
            "limit": 2
        });
        let output = execute_read_file(&args).unwrap();
        assert!(!output.contains("Symbol outline"), "output: {output}");
        assert!(output.contains("Beta"), "output: {output}");

        let _ = fs::remove_file(&path);
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
            let lines = execute_read_file(&lines_args)
                .expect("read_file should accept path alias");
            assert!(lines.contains("line2"), "output: {lines}");
            assert!(!lines.contains("line3"), "output: {lines}");
        });

        let _ = fs::remove_file(&path);
    }
}
