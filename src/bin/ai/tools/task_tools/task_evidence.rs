use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags, OptionalExtension};
use rust_tools::commonw::FastMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::ai::history::{SessionStore, with_sessions_lifecycle_lock};
use crate::ai::types::App;

use super::super::{TaskReport, TaskTerminalStatus};

pub(in crate::ai::tools::task_tools) type StatusCursor =
    FastMap<String, (String, String, u64, u64)>;
static STATUS_CURSOR_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Default)]
pub(in crate::ai::tools::task_tools) struct EvidenceReadOptions {
    fields: Option<Vec<String>>,
    max_chars: Option<usize>,
}

impl EvidenceReadOptions {
    pub(super) fn for_optimizations(args: &Value, enabled: bool) -> Result<Self, String> {
        if !enabled {
            return Ok(Self::default());
        }
        Self::parse(args)
    }

    pub(super) fn parse(args: &Value) -> Result<Self, String> {
        let fields = args
            .get("fields")
            .map(|value| {
                let values = value
                    .as_array()
                    .filter(|values| !values.is_empty())
                    .ok_or("fields must be a non-empty array of field names")?;
                values
                    .iter()
                    .map(|field| {
                        field
                            .as_str()
                            .filter(|field| !field.is_empty())
                            .map(str::to_string)
                            .ok_or_else(|| "fields must contain non-empty strings".to_string())
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .transpose()?;
        let max_chars = args
            .get("max_chars")
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| "max_chars must be a non-negative integer".to_string())
            })
            .transpose()?;
        Ok(Self { fields, max_chars })
    }

    pub(super) fn render_progress(&self, task_id: &str, full: String) -> Result<String, String> {
        if self.fields.is_none() && self.max_chars.is_none() {
            return Ok(full);
        }
        let document = if self.fields.is_some() {
            serde_json::from_str(&full).map_err(|error| {
                format!("Cannot select progress evidence fields: {error}; omit fields to read the original evidence")
            })?
        } else {
            Value::Null
        };
        self.render(task_id, "progress", full, document)
    }

    pub(super) fn render_delivered(
        &self,
        task_id: &str,
        full: String,
        mut document: Value,
        history_file: &Path,
        session_id: &str,
    ) -> Result<String, String> {
        if self.fields.is_none() && self.max_chars.is_none() {
            return Ok(full);
        }
        let payload = document["payload"].as_str().unwrap_or_default().to_string();
        let fields = document
            .as_object_mut()
            .expect("delivered evidence envelope");
        if let Ok(Value::Object(payload_fields)) = serde_json::from_str(&payload) {
            extend_missing_fields(fields, payload_fields);
        }
        extend_delivered_report_fields(history_file, session_id, task_id, fields)?;
        self.render(task_id, "delivered", full, document)
    }

    fn render(
        &self,
        task_id: &str,
        source: &str,
        full: String,
        document: Value,
    ) -> Result<String, String> {
        let recovery = format!(
            "Read the full evidence with task_evidence_read(task_id={task_id:?}, source={source:?}) and omit fields and max_chars. Reading does not collect or integrate the task."
        );
        let mut response = json!({ "task_id": task_id, "source": source });
        for key in ["status", "error"] {
            if let Some(value) = document.get(key) {
                response[key] = value.clone();
            }
        }
        let projected = if let Some(fields) = &self.fields {
            let mut selected = Map::new();
            let mut missing = Vec::new();
            for field in fields {
                match document.get(field) {
                    Some(value) => {
                        selected.insert(field.clone(), value.clone());
                    }
                    None => missing.push(field.as_str()),
                }
            }
            if !missing.is_empty() {
                let mut available = document
                    .as_object()
                    .map(|document| document.keys().map(String::as_str).collect::<Vec<_>>())
                    .unwrap_or_default();
                available.sort_unstable();
                return Err(format!(
                    "Requested evidence fields unavailable: {}. Available fields: {}. {recovery}",
                    missing.join(", "),
                    available.join(", ")
                ));
            }
            response["selection"] = json!({
                "fields": fields,
                "shown_fields": selected.len(),
                "total_fields": document.as_object().map_or(0, Map::len),
                "recovery_hint": recovery,
            });
            Value::Object(selected)
        } else if source == "delivered" {
            document["payload"].clone()
        } else {
            Value::String(full)
        };
        if let Some(max_chars) = self.max_chars {
            let text = match projected {
                Value::String(text) => text,
                value => serde_json::to_string_pretty(&value).map_err(|error| error.to_string())?,
            };
            let total_chars = text.chars().count();
            let prefix: String = text.chars().take(max_chars).collect();
            response["payload"] = Value::String(prefix);
            response["shown_chars"] = json!(total_chars.min(max_chars));
            response["total_chars"] = json!(total_chars);
            response["truncated"] = json!(total_chars > max_chars);
            response["recovery_hint"] = json!(recovery);
        } else {
            response["payload"] = projected;
        }
        serde_json::to_string_pretty(&response)
            .map_err(|error| format!("Failed to render selected evidence: {error}"))
    }
}

fn extend_missing_fields(document: &mut Map<String, Value>, extra: Map<String, Value>) {
    for (key, value) in extra {
        document.entry(key).or_insert(value);
    }
}

/// Read structured output without the ledger reader's optional diagnostic writes.
fn extend_delivered_report_fields(
    history_file: &Path,
    session_id: &str,
    task_id: &str,
    document: &mut Map<String, Value>,
) -> Result<(), String> {
    SessionStore::validate_session_id(session_id).map_err(|error| error.to_string())?;
    let path = SessionStore::new(history_file)
        .session_assets_dir(session_id)
        .join("context-checkpoints/task-evidence.sqlite");
    let connection = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| format!("Failed to read delivered evidence fields: {error}"))?;
    let has_report: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM pragma_table_info('task_evidence') WHERE name = 'report_json')",
        [], |row| row.get(0),
    ).map_err(|error| format!("Failed to inspect delivered evidence fields: {error}"))?;
    if !has_report {
        return Ok(());
    }
    let report_json: Option<Option<String>> = connection
        .query_row(
            "SELECT report_json FROM task_evidence WHERE task_id = ?1 AND delivered_at_unix_ms > 0",
            [task_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| format!("Failed to read delivered evidence fields: {error}"))?;
    let Some(Some(report_json)) = report_json else {
        return Ok(());
    };
    let Ok(mut report) = serde_json::from_str::<TaskReport>(&report_json) else {
        return Ok(());
    };
    if !report.supported() || report.tasks.len() != 1 {
        return Ok(());
    }
    let task = report.tasks.pop().expect("one report task checked above");
    if task.task_id.as_deref() != Some(task_id)
        || document.get("status").and_then(Value::as_str) != Some(task.status.as_str())
    {
        return Ok(());
    }
    if let Some(error) = &task.error {
        document.insert("error".to_string(), Value::String(error.clone()));
    }
    if !task.output_truncated
        && let Some(output) = &task.output
        && let Ok(Value::Object(output_fields)) = serde_json::from_str(output)
    {
        extend_missing_fields(document, output_fields);
    }
    if let Value::Object(metadata) =
        serde_json::to_value(task).map_err(|error| error.to_string())?
    {
        extend_missing_fields(document, metadata);
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct StoredStatusCursor {
    version: u32,
    session_id: String,
    owner_pid: u64,
    cursor: StatusCursor,
}

fn status_cursor_path(
    history_file: &Path,
    session_id: &str,
    owner_pid: u64,
) -> io::Result<PathBuf> {
    SessionStore::validate_session_id(session_id)?;
    Ok(SessionStore::new(history_file)
        .session_assets_dir(session_id)
        .join("context-checkpoints")
        .join(format!("task-status-cursor-{owner_pid}.json")))
}

pub(in crate::ai::tools::task_tools) fn read_status_cursor(
    history_file: &Path,
    session_id: &str,
    owner_pid: u64,
) -> io::Result<StatusCursor> {
    let path = status_cursor_path(history_file, session_id, owner_pid)?;
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(StatusCursor::default()),
        Err(error) => return Err(error),
    };
    let stored: StoredStatusCursor = serde_json::from_slice(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if stored.version != 1 || stored.session_id != session_id || stored.owner_pid != owner_pid {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "status cursor identity/version mismatch",
        ));
    }
    Ok(stored.cursor)
}

pub(in crate::ai::tools::task_tools) fn merge_status_cursor(
    cursor: &mut StatusCursor,
    updates: &StatusCursor,
) {
    for (task_id, update) in updates {
        let should_replace = cursor.get(task_id).is_none_or(|existing| {
            let old_terminal = TaskTerminalStatus::parse(&existing.0).is_some();
            let new_terminal = TaskTerminalStatus::parse(&update.0).is_some();
            if old_terminal != new_terminal {
                return new_terminal;
            }
            (update.3, update.2) >= (existing.3, existing.2)
        });
        if should_replace {
            cursor.insert(task_id.clone(), update.clone());
        }
    }
}

pub(in crate::ai::tools::task_tools) fn write_status_cursor(
    history_file: &Path,
    session_id: &str,
    owner_pid: u64,
    updates: &StatusCursor,
) -> io::Result<()> {
    let path = status_cursor_path(history_file, session_id, owner_pid)?;
    let store = SessionStore::new(history_file);
    let _guard = STATUS_CURSOR_WRITE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    with_sessions_lifecycle_lock(store.sessions_root(), || {
        // A turn that outlives session deletion must not recreate that session's state.
        if !store.session_exists(session_id)? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "status cursor session no longer exists",
            ));
        }
        let mut cursor = read_status_cursor(history_file, session_id, owner_pid)?;
        merge_status_cursor(&mut cursor, updates);
        let bytes = serde_json::to_vec(&StoredStatusCursor {
            version: 1,
            session_id: session_id.to_string(),
            owner_pid,
            cursor,
        })
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::create_dir_all(path.parent().expect("status cursor parent"))?;
        let temporary = path.with_extension(format!("json.{}.tmp", uuid::Uuid::new_v4().simple()));
        if let Err(error) = fs::write(&temporary, bytes).and_then(|_| fs::rename(&temporary, &path))
        {
            let _ = fs::remove_file(temporary);
            return Err(error);
        }
        Ok(())
    })
}

pub(in crate::ai) fn initial_agent_team_task_status_cursor(app: &App) -> StatusCursor {
    if !app
        .config
        .agent_team_optimizations()
        .persist_status_cursor()
    {
        return StatusCursor::default();
    }
    let owner_pid = app
        .os
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .current_process_id();
    owner_pid
        .and_then(|owner_pid| {
            read_status_cursor(&app.config.history_file, &app.session_id, owner_pid).ok()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_evidence_master_off_ignores_projection_and_preserves_bytes() {
        let original = " opaque progress bytes α界🙂\n";
        for args in [
            json!({"fields": ["missing"], "max_chars": 0}),
            json!({"fields": [], "max_chars": -1}),
            json!({"fields": null, "max_chars": "invalid"}),
        ] {
            let options = EvidenceReadOptions::for_optimizations(&args, false).unwrap();
            assert_eq!(
                options
                    .render_progress("task-id", original.to_string())
                    .unwrap(),
                original
            );
            assert_eq!(
                options
                    .render_delivered(
                        "task-id",
                        original.to_string(),
                        Value::Null,
                        Path::new("/unused/history.sqlite"),
                        "unused-session",
                    )
                    .unwrap(),
                original,
            );
        }
        let options =
            EvidenceReadOptions::for_optimizations(&json!({"max_chars": 2}), true).unwrap();
        let bounded: Value = serde_json::from_str(
            &options
                .render_progress("task-id", "α界🙂".to_string())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(bounded["payload"], "α界");
        assert_eq!(bounded["truncated"], true);
        assert!(EvidenceReadOptions::for_optimizations(&json!({"fields": []}), true).is_err());
    }

    #[test]
    fn task_evidence_read_options_preserve_original_and_validate_projection() {
        let original = " {\"phase\":\"read\",\"checkpoint_excerpt\":\"界 α\"} \n";
        let defaults = EvidenceReadOptions::parse(&json!({})).unwrap();
        assert_eq!(
            defaults
                .render_progress("task-id", original.to_string())
                .unwrap(),
            original
        );
        let delivered = " {\"payload\":\"opaque original bytes\"} \n";
        assert_eq!(
            defaults
                .render_delivered(
                    "opaque / task id ",
                    delivered.to_string(),
                    Value::Null,
                    Path::new("/unused/history.sqlite"),
                    "unused-session",
                )
                .unwrap(),
            delivered
        );

        let selection =
            EvidenceReadOptions::parse(&json!({"fields": ["checkpoint_excerpt"]})).unwrap();
        let selected: Value = serde_json::from_str(
            &selection
                .render_progress("task-id", original.to_string())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(selected["payload"], json!({"checkpoint_excerpt": "界 α"}));
        assert_eq!(selected["selection"]["shown_fields"], 1);
        assert_eq!(selected["selection"]["total_fields"], 2);
        assert!(
            selected["selection"]["recovery_hint"]
                .as_str()
                .unwrap()
                .contains("omit fields and max_chars")
        );
        let missing = EvidenceReadOptions::parse(&json!({"fields": ["conclusion"]})).unwrap();
        let error = missing
            .render_progress("task-id", original.to_string())
            .unwrap_err();
        assert!(error.contains("conclusion"));
        assert!(error.contains("checkpoint_excerpt"));

        for (limit, prefix, truncated) in [(0, "", true), (2, "α界", true), (99, "α界🙂z", false)]
        {
            let bounded = EvidenceReadOptions::parse(&json!({"max_chars": limit})).unwrap();
            let value: Value = serde_json::from_str(
                &bounded
                    .render_progress("task-id", "α界🙂z".to_string())
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(value["payload"], prefix);
            assert_eq!(value["shown_chars"], prefix.chars().count());
            assert_eq!(value["total_chars"], 4);
            assert_eq!(value["truncated"], truncated);
        }
        let combined =
            EvidenceReadOptions::parse(&json!({"fields": ["checkpoint_excerpt"], "max_chars": 7}))
                .unwrap();
        let value: Value = serde_json::from_str(
            &combined
                .render_progress("task-id", original.to_string())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(value["payload"].as_str().unwrap().chars().count(), 7);
        assert_eq!(value["selection"]["fields"], json!(["checkpoint_excerpt"]));
        assert_eq!(value["truncated"], true);

        for args in [
            json!({"fields": []}),
            json!({"fields": null}),
            json!({"fields": [""]}),
            json!({"fields": [7]}),
            json!({"max_chars": -1}),
            json!({"max_chars": 1.5}),
            json!({"max_chars": "2"}),
            json!({"max_chars": null}),
        ] {
            assert!(EvidenceReadOptions::parse(&args).is_err(), "{args}");
        }
    }

    #[test]
    fn task_status_cursor_store_merges_revisions_and_scopes() {
        let root =
            std::env::temp_dir().join(format!("status-cursor-{}", uuid::Uuid::new_v4().simple()));
        let history_file = root.join("history.sqlite");
        let session_id = "cursor-session";
        let store = SessionStore::new(&history_file);
        fs::create_dir_all(store.sessions_root()).unwrap();
        fs::write(store.session_history_file(session_id), b"").unwrap();
        assert!(
            read_status_cursor(&history_file, session_id, 11)
                .unwrap()
                .is_empty()
        );
        assert!(
            !status_cursor_path(&history_file, session_id, 11)
                .unwrap()
                .exists()
        );
        let first = StatusCursor::from_iter([(
            "task-a".to_string(),
            ("running".to_string(), "read".to_string(), 3, 7),
        )]);
        let second = StatusCursor::from_iter([(
            "task-b".to_string(),
            ("running".to_string(), "verify".to_string(), 1, 2),
        )]);
        std::thread::scope(|scope| {
            let a = scope.spawn(|| write_status_cursor(&history_file, session_id, 11, &first));
            let b = scope.spawn(|| write_status_cursor(&history_file, session_id, 11, &second));
            a.join().unwrap().unwrap();
            b.join().unwrap().unwrap();
        });
        let mut merged = first.clone();
        merged.extend(second);
        assert_eq!(
            read_status_cursor(&history_file, session_id, 11).unwrap(),
            merged
        );
        let stale = StatusCursor::from_iter([(
            "task-a".to_string(),
            ("running".to_string(), "old".to_string(), 1, 2),
        )]);
        write_status_cursor(&history_file, session_id, 11, &stale).unwrap();
        assert_eq!(
            read_status_cursor(&history_file, session_id, 11).unwrap(),
            merged
        );
        let terminal = StatusCursor::from_iter([(
            "task-a".to_string(),
            ("completed".to_string(), "read".to_string(), 3, 7),
        )]);
        write_status_cursor(&history_file, session_id, 11, &terminal).unwrap();
        let late_progress = StatusCursor::from_iter([(
            "task-a".to_string(),
            ("running".to_string(), "late".to_string(), 4, 8),
        )]);
        write_status_cursor(&history_file, session_id, 11, &late_progress).unwrap();
        assert_eq!(
            read_status_cursor(&history_file, session_id, 11).unwrap()["task-a"].0,
            "completed"
        );
        assert!(
            read_status_cursor(&history_file, session_id, 12)
                .unwrap()
                .is_empty()
        );
        assert!(
            read_status_cursor(&history_file, "other-session", 11)
                .unwrap()
                .is_empty()
        );
        assert!(read_status_cursor(&history_file, "../escape", 11).is_err());
        let path = status_cursor_path(&history_file, session_id, 11).unwrap();
        let original = fs::read(&path).unwrap();
        fs::remove_file(store.session_history_file(session_id)).unwrap();
        assert_eq!(
            write_status_cursor(&history_file, session_id, 11, &first)
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(fs::read(path).unwrap(), original);
        fs::remove_dir_all(root).unwrap();
    }
}
