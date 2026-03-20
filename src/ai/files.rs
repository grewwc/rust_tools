use std::{fs, io, path::Path};

use crate::common::utils::expanduser;
use crate::strw::split::split_by_str_keep_quotes;

use super::types::FileParseResult;

pub(super) fn parse_files(content: &str) -> FileParseResult {
    let files = split_by_str_keep_quotes(content, ",", "\"", false);
    let mut parsed = FileParseResult::default();
    for file in files {
        let file = expanduser(file.trim()).to_string();
        if file.is_empty() {
            continue;
        }
        if fs::read_to_string(&file).is_ok() {
            parsed.text_files.push(file);
        } else if is_image_path(&file) {
            parsed.image_files.push(file);
        } else {
            parsed.binary_files.push(file);
        }
    }
    parsed
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
    let mut content = String::new();
    for file in files {
        content.push_str(&fs::read_to_string(file)?);
        content.push('\n');
    }
    Ok(content)
}
