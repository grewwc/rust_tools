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

/// 从文本内容中提取结构化关键行，为 overflow stub 提供召回锚点。
///
/// 识别的行类型（与 `line_trim_middle` 的中段采样保持一致）：
/// - Rust/代码结构：`fn`/`pub fn`/`impl`/`struct`/`trait`/`enum`/`#[`/`mod`
/// - 文档注释：`//!`/`///`
/// - 错误/警告：`error`/`failed`/`panic`/`exception`/`timeout`/`warning`
/// - 标记：`TODO`/`FIXME`
///
/// 每行截断到 200 字符以控制 stub 体积。最多保留 `max` 行。
pub(super) fn extract_key_lines(content: &str, max: usize) -> Vec<String> {
    let mut result = Vec::with_capacity(max);
    for (idx, line) in content.lines().enumerate() {
        if result.len() >= max {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // read_file 输出带 `{:>6}\t` 行号前缀：前缀会挡住 fn/struct 等前缀匹配，
        // 剥掉前缀、用真实行号做 L 标签，让长文件外溢后的结构索引真正可用。
        let (label, matched) = match split_line_number_prefix(trimmed) {
            Some((no, rest)) => (no, rest.trim()),
            None => (idx, trimmed),
        };
        let lower = matched.to_ascii_lowercase();
        let is_key = lower.starts_with("fn ")
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
            || lower.contains(": warning");
        if is_key {
            let truncated = if matched.chars().count() > 200 {
                let kept: String = matched.chars().take(200).collect();
                format!("L{label}: {kept} …")
            } else {
                format!("L{label}: {matched}")
            };
            result.push(truncated);
        }
    }
    result
}

/// 解析 read_file 输出的 `{:>6}\t{content}` 行号前缀，返回 (真实行号, 前缀后的正文)。
/// 非该格式（如普通命令输出）返回 None，保持原有 `L{idx}` 语义。
fn split_line_number_prefix(trimmed: &str) -> Option<(usize, &str)> {
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
