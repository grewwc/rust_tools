use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags, OptionalExtension};
use rust_tools::commonw::FastMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::ai::history::{SessionStore, with_sessions_lifecycle_lock};
use crate::ai::types::App;

use super::super::{TaskReport, TaskReportTask, TaskTerminalStatus};

pub(in crate::ai::tools::task_tools) type StatusCursor =
    FastMap<String, (String, String, u64, u64)>;
static STATUS_CURSOR_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
const DELIVERED_SUMMARY_MAX_BYTES: usize = 2 * 1024;
const EVIDENCE_PAGE_CHARS: usize = 2048;

#[derive(Default, PartialEq, Eq)]
enum EvidenceView {
    #[default]
    Raw,
    Summary,
}

#[derive(Default)]
pub(in crate::ai::tools::task_tools) struct EvidenceReadOptions {
    fields: Option<Vec<String>>,
    max_chars: Option<usize>,
    cursor: Option<usize>,
    view: EvidenceView,
}

impl EvidenceReadOptions {
    pub(super) fn for_optimizations(args: &Value, enabled: bool) -> Result<Self, String> {
        let mut options = Self::parse(args)?;
        if enabled && args.get("view").is_none() && !options.has_projection() {
            options.view = EvidenceView::Summary;
        }
        Ok(options)
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
        let cursor = args
            .get("cursor")
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        "cursor must be a non-negative Unicode character offset".to_string()
                    })
            })
            .transpose()?;
        let view = match args.get("view") {
            None => EvidenceView::Raw,
            Some(Value::String(view)) if view == "raw" => EvidenceView::Raw,
            Some(Value::String(view)) if view == "summary" => EvidenceView::Summary,
            Some(_) => return Err("view must be 'summary' or 'raw'".to_string()),
        };
        if view == EvidenceView::Summary
            && (fields.is_some() || cursor.is_some() || max_chars.is_some())
        {
            return Err("fields, max_chars, and cursor require view=raw; omit view for automatic projection".to_string());
        }
        Ok(Self {
            fields,
            max_chars,
            cursor,
            view,
        })
    }

    fn has_projection(&self) -> bool {
        self.fields.is_some() || self.max_chars.is_some() || self.cursor.is_some()
    }

    pub(super) fn needs_full_delivered(&self) -> bool {
        self.view == EvidenceView::Raw && !self.has_projection()
    }

    pub(super) fn render_progress(&self, task_id: &str, full: String) -> Result<String, String> {
        if !self.has_projection() {
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
        report: Option<&TaskReportTask>,
    ) -> Result<String, String> {
        if self.needs_full_delivered() && !document.is_object() {
            return Ok(full);
        }
        let mut raw_document = document.clone();
        let payload = document["payload"].as_str().unwrap_or_default().to_string();
        let fields = document
            .as_object_mut()
            .expect("delivered evidence envelope");
        if let Ok(Value::Object(payload_fields)) = serde_json::from_str(&payload) {
            extend_missing_fields(fields, payload_fields);
        }
        let report = report.filter(|report| {
            fields.get("status").and_then(Value::as_str) == Some(report.status.as_str())
                && report.task_id.as_deref() == Some(task_id)
        });
        if let Some(report) = report {
            extend_delivered_report_fields(fields, report)?;
        }
        if fields.get("status").and_then(Value::as_str)
            == Some(TaskTerminalStatus::Cancelled.as_str())
        {
            let reason = fields
                .get("cancel_reason")
                .or_else(|| fields.get("error"))
                .cloned();
            fields.remove("error");
            raw_document
                .as_object_mut()
                .expect("delivered evidence envelope")
                .remove("error");
            if let Some(reason) = reason {
                fields.insert("cancel_reason".to_string(), reason.clone());
                raw_document["cancel_reason"] = reason;
            }
            if let Some(partial_output) = fields.get("partial_output") {
                raw_document["partial_output"] = partial_output.clone();
            }
        }
        // Report metadata owns advisory claims; body-authored assessments remain in payload/output.
        let metadata = report
            .and_then(|report| report.verification.clone())
            .or_else(|| {
                fields
                    .get("verification")
                    .filter(|value| !value.is_null())
                    .map(|value| {
                        super::super::verification::TaskVerificationMetadata::from_value(
                            value.clone(),
                        )
                    })
            });
        let assessment = metadata
            .as_ref()
            .map_or_else(
                || Ok(json!({"available": false, "acceptance": "needs_parent_review"})),
                |metadata| serde_json::to_value(metadata.assessment()),
            )
            .map_err(|error| error.to_string())?;
        if let Some(metadata) = &metadata {
            fields.insert(
                "verification".to_string(),
                serde_json::to_value(metadata).map_err(|error| error.to_string())?,
            );
        } else {
            fields.remove("verification");
        }
        fields.insert("assessment".to_string(), assessment.clone());
        if self.needs_full_delivered() {
            // Keep the original payload and standalone-report deduplication in complete raw reads.
            if raw_document.get("verification").is_some()
                || report
                    .and_then(|report| report.verification.as_ref())
                    .is_none()
            {
                if let Some(verification) = fields.get("verification") {
                    raw_document["verification"] = verification.clone();
                }
            }
            raw_document["assessment"] = assessment;
            return serde_json::to_string_pretty(&raw_document).map_err(|error| error.to_string());
        }
        if self.view == EvidenceView::Summary {
            return self.render_summary(task_id, &document);
        }
        self.render(task_id, "delivered", full, document)
    }

    fn render_summary(&self, task_id: &str, document: &Value) -> Result<String, String> {
        let payload = document["payload"].as_str().unwrap_or_default();
        let total_chars = payload.chars().count();
        // All views use the same normalized advisory, with only the summary representation bounded.
        let verification = document.get("verification");
        let assessment = document.get("assessment");
        let gaps = assessment.and_then(|value| value["gaps"].as_array());
        let mut response = json!({
            "task_id": json_text_prefix(task_id, 256),
            "source": "delivered", "view": "summary",
            "status": json_text_prefix(document["status"].as_str().unwrap_or_default(), 64),
            "payload": "", "shown_chars": 0, "total_chars": total_chars,
            "shown_bytes": 0, "total_bytes": payload.len(),
            "truncated": total_chars > 0,
            "next_cursor": if total_chars > 0 { Some(0) } else { None },
            "verification": {
                "available": verification.is_some(), "summary_only": true,
                "max_bytes": super::super::verification::TASK_VERIFICATION_MAX_BYTES,
                "max_checks": super::super::verification::TASK_VERIFICATION_MAX_CHECKS,
            },
            "assessment": {"available": verification.is_some(), "summary_only": true},
            "recovery_hint": format!("Reuse the exact requested task_id: {} for full evidence; add cursor/max_chars to page or fields=[verification,assessment] for metadata. Read-only; does not integrate.", super::super::full_delivered_evidence_hint(super::super::EvidenceRecoveryFormat::Compact))
        });
        if response["task_id"].as_str() != Some(task_id) {
            response["task_id_truncated"] = json!(true);
        }
        if document["status"].as_str() != response["status"].as_str() {
            response["status_truncated"] = json!(true);
        }
        if let Some(error) = document.get("error").and_then(Value::as_str) {
            response["error"] = json!(json_text_prefix(error, 192));
            response["error_truncated"] = json!(response["error"].as_str() != Some(error));
        }
        if let Some(reason) = document.get("cancel_reason").and_then(Value::as_str) {
            response["cancel_reason"] = json!(json_text_prefix(reason, 192));
            response["cancel_reason_truncated"] =
                json!(response["cancel_reason"].as_str() != Some(reason));
            response["cancel_reason_total_bytes"] = json!(reason.len());
            response["cancel_reason_shown_bytes"] =
                json!(response["cancel_reason"].as_str().unwrap_or_default().len());
        }
        if let Some(verification) = verification {
            response["verification"]["total_bytes"] = json!(
                serde_json::to_vec(verification)
                    .map_err(|error| error.to_string())?
                    .len()
            );
            response["verification"]["total_checks"] =
                json!(verification["checks"].as_array().map_or(0, Vec::len));
        }
        if let Some(assessment) = assessment {
            for key in ["provenance", "schema_status", "acceptance"] {
                if let Some(value) = assessment.get(key).and_then(Value::as_str) {
                    response["assessment"][key] = json!(json_text_prefix(value, 64));
                }
            }
            response["assessment"]["total_gaps"] = json!(gaps.map_or(0, Vec::len));
            if let Some(gap) = gaps.and_then(|gaps| gaps.first()).and_then(Value::as_str) {
                response["assessment"]["first_gap"] = json!(json_text_prefix(gap, 192));
            }
        }
        // Reserve space for the decimal counters that grow after adding the preview.
        let metadata_bytes = serde_json::to_vec(&response)
            .map_err(|error| error.to_string())?
            .len();
        let budget = DELIVERED_SUMMARY_MAX_BYTES.saturating_sub(metadata_bytes + 60);
        let preview = json_text_prefix(payload, budget);
        let shown_chars = preview.chars().count();
        response["shown_bytes"] = json!(preview.len());
        response["payload"] = json!(preview);
        response["shown_chars"] = json!(shown_chars);
        response["truncated"] = json!(shown_chars < total_chars);
        response["next_cursor"] = json!(if shown_chars < total_chars {
            Some(shown_chars)
        } else {
            None
        });
        let rendered = serde_json::to_string(&response).map_err(|error| error.to_string())?;
        if rendered.len() > DELIVERED_SUMMARY_MAX_BYTES {
            return Err("Delivered summary metadata exceeds its byte budget; read view=raw with fields or max_chars".to_string());
        }
        Ok(rendered)
    }

    fn render(
        &self,
        task_id: &str,
        source: &str,
        full: String,
        document: Value,
    ) -> Result<String, String> {
        let recovery = format!(
            "Read the full evidence with {}. To continue, repeat the same fields with cursor=next_cursor and max_chars>0. Reading does not collect or integrate the task.",
            super::super::task_evidence_recovery_hint(task_id, source)
        );
        let mut response = json!({ "task_id": task_id, "source": source });
        for key in ["status", "error", "verification", "assessment"] {
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
        if self.max_chars.is_some() || self.cursor.is_some() {
            let max_chars = self.max_chars.unwrap_or(EVIDENCE_PAGE_CHARS);
            let text = match projected {
                Value::String(text) => text,
                value => serde_json::to_string_pretty(&value).map_err(|error| error.to_string())?,
            };
            let total_chars = text.chars().count();
            let cursor = self.cursor.unwrap_or(0);
            if cursor > total_chars {
                return Err(format!(
                    "cursor {cursor} exceeds total_chars {total_chars}; start with cursor=0"
                ));
            }
            let prefix: String = text.chars().skip(cursor).take(max_chars).collect();
            let shown_chars = prefix.chars().count();
            let next_cursor = cursor + shown_chars;
            response["payload"] = Value::String(prefix);
            response["cursor"] = json!(cursor);
            response["shown_chars"] = json!(shown_chars);
            response["total_chars"] = json!(total_chars);
            response["truncated"] = json!(next_cursor < total_chars);
            response["next_cursor"] = json!(if next_cursor < total_chars {
                Some(next_cursor)
            } else {
                None
            });
            response["recovery_hint"] = json!(recovery);
        } else {
            response["payload"] = projected;
        }
        serde_json::to_string_pretty(&response)
            .map_err(|error| format!("Failed to render selected evidence: {error}"))
    }
}

/// Limit the encoded JSON string contents, including escaping, without splitting UTF-8.
fn json_text_prefix(text: &str, max_bytes: usize) -> String {
    let mut bytes = 0;
    text.chars()
        .take_while(|character| {
            let encoded = match character {
                '"' | '\\' | '\n' | '\r' | '\t' | '\u{08}' | '\u{0c}' => 2,
                character if character.is_control() && *character <= '\u{1f}' => 6,
                character => character.len_utf8(),
            };
            bytes += encoded;
            bytes <= max_bytes
        })
        .collect()
}

fn extend_missing_fields(document: &mut Map<String, Value>, extra: Map<String, Value>) {
    for (key, value) in extra {
        document.entry(key).or_insert(value);
    }
}

/// Read optional metadata without diagnostics, migrations, or any other ledger writes.
pub(super) fn read_delivered_task_report(
    history_file: &Path,
    session_id: &str,
    task_id: &str,
) -> Result<Option<TaskReportTask>, String> {
    SessionStore::validate_session_id(session_id).map_err(|error| error.to_string())?;
    let path = SessionStore::new(history_file)
        .session_assets_dir(session_id)
        .join("context-checkpoints/task-evidence.sqlite");
    if !path.is_file() {
        return Ok(None);
    }
    let connection = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| format!("Failed to read delivered evidence fields: {error}"))?;
    let has_report: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM pragma_table_info('task_evidence') WHERE name = 'report_json')",
        [], |row| row.get(0),
    ).map_err(|error| format!("Failed to inspect delivered evidence fields: {error}"))?;
    if !has_report {
        return Ok(None);
    }
    let row: Option<(String, Option<String>)> = connection
        .query_row(
            "SELECT status, report_json FROM task_evidence WHERE task_id = ?1 AND delivered_at_unix_ms > 0",
            [task_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(|error| format!("Failed to read delivered evidence fields: {error}"))?;
    let Some((status, Some(report_json))) = row else {
        return Ok(None);
    };
    let Ok(mut report) = serde_json::from_str::<TaskReport>(&report_json) else {
        return Ok(None);
    };
    if !report.supported() || report.tasks.len() != 1 {
        return Ok(None);
    }
    let task = report.tasks.pop().expect("one report task checked above");
    if task.task_id.as_deref() != Some(task_id) || status != task.status {
        return Ok(None);
    }
    Ok(Some(task))
}

fn extend_delivered_report_fields(
    document: &mut Map<String, Value>,
    task: &TaskReportTask,
) -> Result<(), String> {
    let cancelled = task.status == TaskTerminalStatus::Cancelled.as_str();
    if !cancelled && let Some(error) = &task.error {
        document.insert("error".to_string(), Value::String(error.clone()));
    }
    if !task.output_truncated
        && let Some(output) = &task.output
        && let Ok(Value::Object(output_fields)) = serde_json::from_str(output)
    {
        extend_missing_fields(document, output_fields);
    }
    if let Value::Object(mut metadata) =
        serde_json::to_value(task).map_err(|error| error.to_string())?
    {
        if cancelled {
            metadata.remove("error");
        }
        extend_missing_fields(document, metadata);
    }
    if cancelled {
        document.remove("error");
        if let Some(reason) = task.cancel_reason.as_ref().or(task.error.as_ref()) {
            document.insert("cancel_reason".to_string(), Value::String(reason.clone()));
        }
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
    fn task_evidence_master_off_honors_projection_and_preserves_unselected_bytes() {
        let original = " opaque progress bytes α界🙂\n";
        for enabled in [false, true] {
            let defaults = EvidenceReadOptions::for_optimizations(&json!({}), enabled).unwrap();
            assert_eq!(
                defaults
                    .render_progress("task-id", original.to_string())
                    .unwrap(),
                original
            );
            let options =
                EvidenceReadOptions::for_optimizations(&json!({"max_chars": 2}), enabled).unwrap();
            let bounded: Value = serde_json::from_str(
                &options
                    .render_progress("task-id", "α界🙂".to_string())
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(bounded["payload"], "α界");
            assert_eq!(bounded["truncated"], true);
            assert_eq!(bounded["next_cursor"], 2);
            for invalid in [
                json!({"fields": []}),
                json!({"max_chars": -1}),
                json!({"fields": null, "max_chars": "invalid"}),
            ] {
                assert!(EvidenceReadOptions::for_optimizations(&invalid, enabled).is_err());
            }
        }
        let raw = EvidenceReadOptions::for_optimizations(&json!({}), false).unwrap();
        assert_eq!(
            raw.render_delivered("task-id", original.to_string(), Value::Null, None)
                .unwrap(),
            original
        );
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
                    None,
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
                .contains("omitting fields/max_chars/cursor")
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
    fn task_evidence_read_summary_bounds_escaped_bytes_and_recovers_metadata() {
        let verification = super::super::super::verification::TaskVerificationMetadata::from_value(
            super::super::super::verification::fix_verification_fixture(),
        );
        let payload = "\"\\\n\u{00}α界🙂".repeat(4096);
        let document = json!({"task_id": "task-id", "status": "completed", "payload": payload,
            "verification": verification, "assessment": verification.assessment()});
        let summary = EvidenceReadOptions::for_optimizations(&json!({}), true)
            .unwrap()
            .render_delivered("task-id", String::new(), document.clone(), None)
            .unwrap();
        assert!(
            summary.len() <= DELIVERED_SUMMARY_MAX_BYTES,
            "{}",
            summary.len()
        );
        let value: Value = serde_json::from_str(&summary).unwrap();
        assert_eq!(value["view"], "summary");
        assert_eq!(value["verification"]["available"], true);
        assert_eq!(value["verification"]["total_checks"], 2);
        assert_eq!(value["assessment"]["schema_status"], "valid");
        assert_eq!(value["assessment"]["acceptance"], "needs_parent_review");
        assert_eq!(value["total_bytes"], payload.len());
        assert_eq!(value["total_chars"], payload.chars().count());
        assert_eq!(value["next_cursor"], value["shown_chars"]);
        assert_eq!(
            value["shown_bytes"],
            value["payload"].as_str().unwrap().len()
        );
        assert!(payload.starts_with(value["payload"].as_str().unwrap()));
        assert!(summary.len() * 10 < serde_json::to_vec(&document).unwrap().len());

        for (invalid, message) in [
            (
                json!({"schema": "ai.task-verification", "version": 1, "checks": vec![json!({}); 17]}),
                "16-check",
            ),
            (
                json!({"schema": "ai.task-verification", "version": 1, "note": "界".repeat(8192)}),
                "8 KiB",
            ),
        ] {
            let metadata =
                super::super::super::verification::TaskVerificationMetadata::from_value(invalid);
            let document = json!({"status": "completed", "payload": payload,
                "verification": metadata, "assessment": metadata.assessment()});
            let summary = EvidenceReadOptions::for_optimizations(&json!({}), true)
                .unwrap()
                .render_delivered("task-id", String::new(), document.clone(), None)
                .unwrap();
            assert!(summary.len() <= DELIVERED_SUMMARY_MAX_BYTES);
            let summary: Value = serde_json::from_str(&summary).unwrap();
            assert_eq!(summary["status"], "completed");
            assert_eq!(summary["assessment"]["schema_status"], "invalid");
            assert!(
                summary["assessment"]["first_gap"]
                    .as_str()
                    .unwrap()
                    .contains(message)
            );
            let full = serde_json::to_string_pretty(&document).unwrap();
            let raw = EvidenceReadOptions::for_optimizations(&json!({"view": "raw"}), true)
                .unwrap()
                .render_delivered("task-id", full.clone(), document, None)
                .unwrap();
            assert_eq!(raw.as_bytes(), full.as_bytes());
        }
    }

    #[test]
    fn task_evidence_read_pagination_preserves_assessment_and_unicode_offsets() {
        let verification = super::super::super::verification::TaskVerificationMetadata::from_value(
            json!({"schema": "ai.task-verification"}),
        );
        let document = json!({"status": "completed", "payload": "α界🙂z",
            "verification": verification,
            "assessment": verification.assessment()});
        for enabled in [false, true] {
            let mut recovered = String::new();
            let mut cursor = 0;
            loop {
                let options = EvidenceReadOptions::for_optimizations(
                    &json!({"max_chars": 2, "cursor": cursor}),
                    enabled,
                )
                .unwrap();
                let page: Value = serde_json::from_str(
                    &options
                        .render_delivered("task-id", String::new(), document.clone(), None)
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(page["verification"], document["verification"]);
                assert_eq!(page["assessment"], document["assessment"]);
                assert_eq!(page["cursor"], cursor);
                recovered.push_str(page["payload"].as_str().unwrap());
                match page["next_cursor"].as_u64() {
                    Some(next) => {
                        assert!(next > cursor);
                        cursor = next;
                    }
                    None => {
                        assert_eq!(page["truncated"], false);
                        break;
                    }
                }
            }
            assert_eq!(recovered, "α界🙂z");
            let options =
                EvidenceReadOptions::for_optimizations(&json!({"max_chars": 0}), enabled).unwrap();
            let page: Value = serde_json::from_str(
                &options
                    .render_delivered("task-id", String::new(), document.clone(), None)
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(page["payload"], "");
            assert_eq!(page["next_cursor"], 0);
            assert_eq!(page["verification"], document["verification"]);
            assert_eq!(page["assessment"], document["assessment"]);
            for (cursor, error) in [(4, false), (5, true)] {
                let options =
                    EvidenceReadOptions::for_optimizations(&json!({"cursor": cursor}), enabled)
                        .unwrap();
                let result =
                    options.render_delivered("task-id", String::new(), document.clone(), None);
                if error {
                    assert!(result.unwrap_err().contains("total_chars 4"));
                } else {
                    let page: Value = serde_json::from_str(&result.unwrap()).unwrap();
                    assert_eq!(page["payload"], "");
                    assert_eq!(page["next_cursor"], Value::Null);
                    assert_eq!(page["truncated"], false);
                }
            }
        }
        for args in [
            json!({"cursor": -1}),
            json!({"cursor": "2"}),
            json!({"cursor": null}),
            json!({"view": "other"}),
            json!({"view": "summary", "fields": ["verification"]}),
            json!({"view": "summary", "cursor": 0}),
        ] {
            assert!(EvidenceReadOptions::parse(&args).is_err(), "{args}");
        }
        let options = EvidenceReadOptions::parse(
            &json!({"fields": ["assessment"], "cursor": 0, "max_chars": 1}),
        )
        .unwrap();
        let page: Value = serde_json::from_str(
            &options
                .render_delivered("task-id", String::new(), document.clone(), None)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(page["payload"], "{");
        assert_eq!(page["next_cursor"], 1);
        assert_eq!(page["selection"]["fields"], json!(["assessment"]));
        assert_eq!(page["assessment"], document["assessment"]);
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
