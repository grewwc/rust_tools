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
    store.validate_access().map_err(|e| e.to_string())?;
    store.ensure_exists().map_err(|e| e.to_string())?;

    let offset = args["offset"].as_u64().unwrap_or(1) as usize;
    let limit = args["limit"].as_u64().unwrap_or(1000) as usize;
    let content = store.read_to_string().map_err(|e| e.to_string())?;
    let lines: Vec<&str> = content.lines().collect();
    let start = offset.saturating_sub(1).min(lines.len());
    let end = (start + limit).min(lines.len());

    Ok(render_lines(&content, start, end, 10))
}

pub(crate) fn execute_read_file_lines(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let store = FileStore::new(PathBuf::from(file_path));
    store.validate_access().map_err(|e| e.to_string())?;
    store.ensure_exists().map_err(|e| e.to_string())?;

    let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
    let limit = args["limit"].as_u64().unwrap_or(200).clamp(1, 400) as usize;
    let content = store.read_to_string().map_err(|e| e.to_string())?;
    // let lines: Vec<&str> = content.lines().collect();
    let num_lines = content.bytes().filter(|&b| b == b'\n').count();
    let start = offset.saturating_sub(1);
    if start >= num_lines {
        return Ok(String::new());
    }
    let end = (start + limit).min(num_lines);

    Ok(render_lines(&content, start, end, usize::MAX))
}

pub(crate) fn execute_write_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let content = args["content"].as_str().ok_or("Missing content")?;

    super::super::undo_tools::snapshot_file_before_write(file_path);

    let store = FileStore::new(PathBuf::from(file_path));
    store.validate_access().map_err(|e| e.to_string())?;
    store.write_all(content).map_err(|e| e.to_string())?;

    super::super::undo_tools::commit_change_set(&format!("write_file: {}", file_path));

    Ok(format!("Successfully wrote to {}", store.path().display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_tools_test_{}_{}", name, uuid::Uuid::new_v4()));
        path
    }

    #[test]
    fn test_write_and_read_file_roundtrip() {
        let path = make_temp_path("roundtrip");
        let content = "Hello, integration test!\nLine 2\nLine 3";

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
    fn test_write_file_creates_parent_dirs() {
        let mut path = make_temp_path("nested");
        path.push("a");
        path.push("b");
        path.push("c");
        path.push("deep.txt");

        let content = "deeply nested content";
        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "content": content
        });
        let result = execute_write_file(&args);
        assert!(result.is_ok(), "write failed: {:?}", result);

        assert!(path.exists(), "file should exist");
        let read_back = fs::read_to_string(&path).unwrap();
        assert_eq!(read_back, content);

        let base = path
            .ancestors()
            .find(|p| {
                p.file_name().map_or(false, |n| {
                    n.to_string_lossy().starts_with("ai_tools_test_nested")
                })
            })
            .unwrap_or_else(|| path.parent().unwrap());
        let _ = fs::remove_dir_all(base);
    }
}
