use std::path::PathBuf;

use serde_json::Value;

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

pub(crate) fn execute_read_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
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

    let rendered = render_line_excerpt(&content, start, end, None).text;
    let rendered = append_truncation_notice(rendered, start, end, total);
    Ok(append_symbol_outline_if_useful(
        rendered,
        file_path,
        &content,
        start,
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

/// 当本次读取没有覆盖到文件末尾时，追加一条明确提示，告知模型文件仍有
/// 剩余行未显示以及如何继续读取。避免模型把"截断结果"误判为"完整文件"。
fn append_truncation_notice(
    mut rendered: String,
    start: usize,
    end: usize,
    total: usize,
) -> String {
    let remaining = total.saturating_sub(end);
    if remaining > 0 {
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str(&format!(
            "... [truncated: showing lines {}-{} of {}; {} more line(s) not shown. Continue with offset={} to read the rest.]",
            start + 1,
            end,
            total,
            remaining,
            end + 1
        ));
    }
    rendered
}

pub(crate) fn execute_read_file_lines(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let store = FileStore::new(PathBuf::from(file_path));
    store.validate_read_access().map_err(|e| e.to_string())?;
    store.ensure_exists().map_err(|e| e.to_string())?;
    if crate::ai::files::is_image_path(file_path) {
        return Ok(image_read_redirect_message(file_path));
    }

    let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
    let limit = args["limit"].as_u64().unwrap_or(200).clamp(1, 400) as usize;
    let content = store.read_to_string().map_err(|e| e.to_string())?;
    // 用 lines() 统计总行数，避免按 '\n' 计数在"末尾无换行符"时漏掉最后一行。
    let total = content.lines().count();
    let start = offset.saturating_sub(1);
    if start >= total {
        return Ok(String::new());
    }
    let end = (start + limit).min(total);

    let rendered = render_line_excerpt(&content, start, end, None).text;
    let rendered = append_truncation_notice(rendered, start, end, total);
    Ok(append_symbol_outline_if_useful(
        rendered,
        file_path,
        &content,
        start,
    ))
}

pub(crate) fn execute_write_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let content = args["content"].as_str().ok_or("Missing content")?;

    super::super::undo_tools::snapshot_file_before_write(file_path);

    let store = FileStore::new(PathBuf::from(file_path));
    store.validate_write_access().map_err(|e| e.to_string())?;
    store.write_all(content).map_err(|e| e.to_string())?;

    super::super::undo_tools::commit_change_set(&format!("write_file: {}", file_path));

    Ok(format!("Successfully wrote to {}", store.path().display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::test_support::ENV_LOCK;
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
    fn test_read_file_lines_reads_last_line_without_trailing_newline() {
        // 文件末尾无换行符时，旧实现按 '\n' 计数会漏掉最后一行。
        let path = make_temp_path("lastline");
        fs::write(&path, "first\nsecond\nthird").unwrap();

        let read_args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "offset": 1,
            "limit": 100
        });
        let output = execute_read_file_lines(&read_args).unwrap();
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
        let result = execute_read_file_lines(&args);
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
        let output = execute_read_file_lines(&args).unwrap();
        assert!(!output.contains("Symbol outline"), "output: {output}");
        assert!(output.contains("Beta"), "output: {output}");

        let _ = fs::remove_file(&path);
    }
}
