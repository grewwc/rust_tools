//! Read-recall service: the `list_read_files` tool.
//!
//! The session read registry (storage::read_registry) records every model-issued
//! `read_file` call. This tool turns it into a recall list, so after several
//! turns or context compression the model can answer "which files did I already
//! read, and at what path" — e.g. when the user says a file was updated and it
//! must re-read the same path — instead of re-locating files with
//! find/execute_command.
//!
//! Each entry reports whether the file changed on disk since the last read
//! (mtime > last-read time). Both sides use millisecond precision, so a
//! modification right after the read is detected.

use std::path::PathBuf;

use chrono::{Local, TimeZone};
use serde_json::{Value, json};

use crate::ai::tools::registry::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::storage::read_registry;

fn execute_list_read_files(args: &Value) -> Result<String, String> {
    let v = handle_list_read_files(args);
    serde_json::to_string_pretty(&v).map_err(|e| format!("failed to serialize: {e}"))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "list_read_files",
        description: "",
        execute: execute_list_read_files,
    }
});

/// Cap on listed entries; the most recently read files win.
const MAX_LISTED: usize = 100;

pub(crate) fn handle_list_read_files(args: &Value) -> Value {
    let path_filter = args
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .unwrap_or_else(|_| PathBuf::from("."));
    let all = read_registry::list();
    let total = all.len();

    let mut files: Vec<Value> = Vec::new();
    let mut truncated = false;
    for e in all {
        if !path_filter.is_empty()
            && !e.abs.contains(path_filter)
            && !e.original.contains(path_filter)
        {
            continue;
        }
        if files.len() >= MAX_LISTED {
            truncated = true;
            break;
        }
        let abs = PathBuf::from(&e.abs);
        let rel = abs
            .strip_prefix(&cwd)
            .map(|r| r.display().to_string())
            .unwrap_or_else(|_| e.abs.clone());
        let status = match std::fs::metadata(&abs) {
            Ok(m) => {
                let changed = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| (d.as_millis() as u64) > e.last_read)
                    .unwrap_or(false);
                if changed {
                    "changed"
                } else {
                    "unchanged"
                }
            }
            Err(_) => "missing",
        };
        let last_read = Local
            .timestamp_opt((e.last_read / 1000) as i64, 0)
            .single()
            .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| e.last_read.to_string());
        files.push(json!({
            "path": rel,
            "abs": e.abs,
            "last_read": last_read,
            "status": status,
        }));
    }

    let note = if files.is_empty() && total == 0 {
        "No read_file calls recorded in this session yet.".to_string()
    } else if files.is_empty() {
        format!(
            "No files match the path filter (total read_file calls this session: {total})."
        )
    } else if truncated {
        format!(
            "Showing the {MAX_LISTED} most recently read files (total read_file calls this session: {total}); pass a `path` filter to narrow."
        )
    } else {
        String::new()
    };
    let mut out = json!({ "count": files.len(), "files": files });
    if !note.is_empty() {
        out["note"] = json!(note);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::test_support::ENV_LOCK;

    use crate::ai::tools::service::file::execute_read_file;

    #[test]
    fn list_read_files_reports_paths_filter_and_status() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let session_id = format!("reads_svc_test_{}", uuid::Uuid::new_v4());
        let base = std::env::temp_dir().join(format!("reads_svc_base_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        let file = base.join("a.txt");
        std::fs::write(&file, "hello").unwrap();

        crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope((session_id, 0usize), || {
            crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(base.clone(), || {
                // Reading through the real tool executor records the entry.
                assert!(execute_read_file(&json!({ "file_path": "a.txt" })).is_ok());

                let listed = handle_list_read_files(&json!({}));
                assert_eq!(listed["count"], 1);
                let f0 = &listed["files"][0];
                assert_eq!(f0["path"], "a.txt");
                assert_eq!(f0["abs"], file.to_string_lossy().as_ref());
                assert_eq!(f0["status"], "unchanged");

                // Substring filter narrows the list.
                let filtered = handle_list_read_files(&json!({ "path": "zzz" }));
                assert_eq!(filtered["count"], 0);
                assert!(filtered["note"].as_str().unwrap().contains("No files match"));

                // mtime newer than last read => "changed".
                std::thread::sleep(std::time::Duration::from_millis(1100));
                std::fs::write(&file, "hello changed").unwrap();
                let listed2 = handle_list_read_files(&json!({}));
                assert_eq!(listed2["files"][0]["status"], "changed");

                // File gone => "missing".
                std::fs::remove_file(&file).unwrap();
                let listed3 = handle_list_read_files(&json!({}));
                assert_eq!(listed3["files"][0]["status"], "missing");
            });
        });

        let _ = std::fs::remove_dir_all(&base);
    }
}
