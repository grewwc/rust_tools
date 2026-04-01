use std::path::PathBuf;

use serde_json::Value;

use crate::ai::tools::storage::file_store::FileStore;

fn render_lines(content: &str, start: usize, end: usize, max_lines: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let result: Vec<String> = lines[start..end]
        .iter()
        .take(max_lines)
        .enumerate()
        .map(|(idx, line)| format!("{:>6}\t{}", start + idx + 1, line))
        .collect();
    result.join("\n")
}

pub(crate) fn execute_read_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let store = FileStore::new(PathBuf::from(file_path));
    store.validate_access()?;
    store.ensure_exists()?;

    let offset = args["offset"].as_u64().unwrap_or(1) as usize;
    let limit = args["limit"].as_u64().unwrap_or(1000) as usize;
    let content = store.read_to_string()?;
    let lines: Vec<&str> = content.lines().collect();
    let start = offset.saturating_sub(1).min(lines.len());
    let end = (start + limit).min(lines.len());

    Ok(render_lines(&content, start, end, 10))
}

pub(crate) fn execute_read_file_lines(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let store = FileStore::new(PathBuf::from(file_path));
    store.validate_access()?;
    store.ensure_exists()?;

    let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
    let limit = args["limit"].as_u64().unwrap_or(200).clamp(1, 400) as usize;
    let content = store.read_to_string()?;
    let lines: Vec<&str> = content.lines().collect();
    let start = offset.saturating_sub(1);
    if start >= lines.len() {
        return Ok(String::new());
    }
    let end = (start + limit).min(lines.len());

    Ok(render_lines(&content, start, end, usize::MAX))
}

pub(crate) fn execute_write_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let content = args["content"].as_str().ok_or("Missing content")?;

    let store = FileStore::new(PathBuf::from(file_path));
    store.validate_access()?;
    store.write_all(content)?;

    Ok(format!("Successfully wrote to {}", store.path().display()))
}
