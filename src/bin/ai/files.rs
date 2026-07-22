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

#[cfg(test)]
#[path = "files_tests.rs"]
mod tests;
