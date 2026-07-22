use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

fn params_list_directory() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (non-recursive). Absolute path recommended."
            }
        },
        "required": ["path"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "list_directory",
        description: "List direct children of a directory (non-recursive). Each line is a child name; directories are suffixed with '/'.",
        parameters: params_list_directory,
        execute: execute_list_directory,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin"],
    }
});

pub(crate) fn execute_list_directory(args: &Value) -> Result<String, String> {
    let path = args["path"].as_str().ok_or("Missing path")?;
    let dir_path = PathBuf::from(path);

    if !dir_path.exists() {
        return Err(format!("Directory not found: {}", path));
    }

    if !dir_path.is_dir() {
        return Err(format!("Not a directory: {}", path));
    }

    let entries: Vec<_> = fs::read_dir(&dir_path)
        .map_err(|e| format!("Failed to read directory: {}", e))?
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir { format!("{}/", name) } else { name }
        })
        .collect();

    Ok(entries.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_search_test_{}_{}", name, uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    #[test]
    fn test_list_directory_returns_entries() {
        let dir = make_temp_dir("listdir");
        fs::write(dir.join("file1.txt"), "").unwrap();
        fs::write(dir.join("file2.txt"), "").unwrap();
        fs::create_dir(dir.join("subdir")).unwrap();

        let args = serde_json::json!({
            "path": dir.to_string_lossy()
        });
        let result = execute_list_directory(&args);
        assert!(result.is_ok(), "list_dir failed: {:?}", result);
        let output = result.unwrap();
        assert!(
            output.contains("file1.txt"),
            "should list file1.txt, got: {}",
            output
        );
        assert!(
            output.contains("file2.txt"),
            "should list file2.txt, got: {}",
            output
        );
        assert!(
            output.contains("subdir/"),
            "should list subdir with trailing /, got: {}",
            output
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
