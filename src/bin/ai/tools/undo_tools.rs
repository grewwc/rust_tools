use serde_json::Value;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};

#[derive(Debug, Clone)]
struct FileSnapshot {
    path: String,
    content: Option<String>,
    existed_before: bool,
}

#[derive(Debug, Clone)]
struct ChangeSet {
    files: Vec<FileSnapshot>,
    description: String,
    timestamp: i64,
}

static UNDO_STACK: Mutex<VecDeque<ChangeSet>> = Mutex::new(VecDeque::new());
static REDO_STACK: Mutex<VecDeque<ChangeSet>> = Mutex::new(VecDeque::new());
const MAX_UNDO_HISTORY: usize = 50;

pub(crate) fn snapshot_file_before_write(path: &str) {
    let file_path = Path::new(path);
    let existed_before = file_path.exists();
    let content = if existed_before {
        fs::read_to_string(file_path).ok()
    } else {
        None
    };

    let snapshot = FileSnapshot {
        path: path.to_string(),
        content,
        existed_before,
    };

    let mut stack = UNDO_STACK.lock().unwrap();
    if stack.is_empty() {
        stack.push_back(ChangeSet {
            files: vec![snapshot],
            description: String::new(),
            timestamp: chrono::Local::now().timestamp(),
        });
    } else {
        stack.back_mut().unwrap().files.push(snapshot);
    }
}

pub(crate) fn commit_change_set(description: &str) {
    let mut undo_stack = UNDO_STACK.lock().unwrap();
    let mut redo_stack = REDO_STACK.lock().unwrap();

    if let Some(last) = undo_stack.back_mut() {
        if last.description.is_empty() {
            last.description = description.to_string();
        }
    }

    redo_stack.clear();

    if undo_stack.len() > MAX_UNDO_HISTORY {
        let to_remove = undo_stack.len() - MAX_UNDO_HISTORY;
        for _ in 0..to_remove {
            undo_stack.pop_front();
        }
    }
}

fn params_undo() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of changes to undo (default: 1)."
            }
        }
    })
}

fn params_redo() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of changes to redo (default: 1)."
            }
        }
    })
}

fn params_undo_status() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "undo",
        description: "Undo the most recent file changes. Restores files to their state before the last modification. Use 'count' to undo multiple changes. Each undo can be reversed with redo.",
        parameters: params_undo,
        execute: execute_undo,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "redo",
        description: "Redo previously undone changes. Restores changes that were undone. Use 'count' to redo multiple changes.",
        parameters: params_redo,
        execute: execute_redo,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "undo_status",
        description:
            "Show the current undo/redo stack status, listing available undo and redo operations.",
        parameters: params_undo_status,
        execute: execute_undo_status,
        groups: &["builtin"],
    }
});

pub(crate) fn execute_undo(args: &Value) -> Result<String, String> {
    let count = args["count"].as_u64().unwrap_or(1) as usize;
    if count == 0 {
        return Err("count must be at least 1".to_string());
    }

    let mut undo_stack = UNDO_STACK.lock().unwrap();
    let mut redo_stack = REDO_STACK.lock().unwrap();

    if undo_stack.is_empty() {
        return Ok("Nothing to undo.".to_string());
    }

    let actual_count = count.min(undo_stack.len());
    let mut undone = Vec::new();

    for _ in 0..actual_count {
        if let Some(change_set) = undo_stack.pop_back() {
            let mut restored_files = Vec::new();

            for snapshot in &change_set.files {
                let path = Path::new(&snapshot.path);

                if snapshot.existed_before {
                    if let Some(content) = &snapshot.content {
                        if let Some(parent) = path.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        fs::write(path, content)
                            .map_err(|e| format!("Failed to restore {}: {}", snapshot.path, e))?;
                        restored_files.push(format!("  Restored: {}", snapshot.path));
                    }
                } else {
                    let _ = fs::remove_file(path);
                    restored_files.push(format!("  Deleted (was created): {}", snapshot.path));
                }
            }

            redo_stack.push_back(change_set.clone());
            undone.push(format!(
                "Undid: {}\n{}",
                if change_set.description.is_empty() {
                    "file changes"
                } else {
                    &change_set.description
                },
                restored_files.join("\n")
            ));
        }
    }

    if redo_stack.len() > MAX_UNDO_HISTORY {
        let to_remove = redo_stack.len() - MAX_UNDO_HISTORY;
        for _ in 0..to_remove {
            redo_stack.pop_front();
        }
    }

    Ok(format!(
        "Undid {} change(s):\n\n{}",
        actual_count,
        undone.join("\n\n")
    ))
}

pub(crate) fn execute_redo(args: &Value) -> Result<String, String> {
    let count = args["count"].as_u64().unwrap_or(1) as usize;
    if count == 0 {
        return Err("count must be at least 1".to_string());
    }

    let mut undo_stack = UNDO_STACK.lock().unwrap();
    let mut redo_stack = REDO_STACK.lock().unwrap();

    if redo_stack.is_empty() {
        return Ok("Nothing to redo.".to_string());
    }

    let actual_count = count.min(redo_stack.len());
    let mut redone = Vec::new();

    for _ in 0..actual_count {
        if let Some(change_set) = redo_stack.pop_back() {
            let mut reapplied_files = Vec::new();

            for snapshot in &change_set.files {
                let path = Path::new(&snapshot.path);

                if !snapshot.existed_before {
                    if let Some(content) = &snapshot.content {
                        if let Some(parent) = path.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        fs::write(path, content)
                            .map_err(|e| format!("Failed to reapply {}: {}", snapshot.path, e))?;
                        reapplied_files.push(format!("  Recreated: {}", snapshot.path));
                    }
                } else if snapshot.content.is_some() {
                    if let Some(parent) = path.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    fs::write(path, snapshot.content.as_ref().unwrap())
                        .map_err(|e| format!("Failed to reapply {}: {}", snapshot.path, e))?;
                    reapplied_files.push(format!("  Reapplied changes to: {}", snapshot.path));
                }
            }

            undo_stack.push_back(change_set.clone());
            redone.push(format!(
                "Redid: {}\n{}",
                if change_set.description.is_empty() {
                    "file changes"
                } else {
                    &change_set.description
                },
                reapplied_files.join("\n")
            ));
        }
    }

    Ok(format!(
        "Redid {} change(s):\n\n{}",
        actual_count,
        redone.join("\n\n")
    ))
}

pub(crate) fn execute_undo_status(_args: &Value) -> Result<String, String> {
    let undo_stack = UNDO_STACK.lock().unwrap();
    let redo_stack = REDO_STACK.lock().unwrap();

    let mut output = String::new();

    output.push_str(&format!("Undo stack: {} entries\n", undo_stack.len()));
    if !undo_stack.is_empty() {
        for (i, change_set) in undo_stack.iter().rev().enumerate().take(10) {
            let desc = if change_set.description.is_empty() {
                "file changes"
            } else {
                &change_set.description
            };
            let file_count = change_set.files.len();
            output.push_str(&format!("  {}. {} ({} file(s))\n", i + 1, desc, file_count));
        }
        if undo_stack.len() > 10 {
            output.push_str(&format!("  ... and {} more\n", undo_stack.len() - 10));
        }
    }

    output.push_str(&format!("\nRedo stack: {} entries\n", redo_stack.len()));
    if !redo_stack.is_empty() {
        for (i, change_set) in redo_stack.iter().rev().enumerate().take(10) {
            let desc = if change_set.description.is_empty() {
                "file changes"
            } else {
                &change_set.description
            };
            let file_count = change_set.files.len();
            output.push_str(&format!("  {}. {} ({} file(s))\n", i + 1, desc, file_count));
        }
        if redo_stack.len() > 10 {
            output.push_str(&format!("  ... and {} more\n", redo_stack.len() - 10));
        }
    }

    if undo_stack.is_empty() && redo_stack.is_empty() {
        output.push_str("\nNo undo/redo history available.");
    }

    Ok(output)
}
