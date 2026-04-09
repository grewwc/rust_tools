use std::{fs, path::PathBuf};

use serde_json::Value;

use crate::ai::{history::SessionStore, types::App};

use super::preview::{tail_chars, truncate_chars};
use super::super::{TOOL_OVERFLOW_PREVIEW_CHARS, types::LargeToolSummary};

pub(super) fn summarize_large_tool_output(content: &str) -> LargeToolSummary {
    let trimmed = content.trim();
    if (trimmed.starts_with('{') || trimmed.starts_with('['))
        && let Ok(json) = serde_json::from_str::<Value>(trimmed)
    {
        let pretty = serde_json::to_string_pretty(&json).unwrap_or_else(|_| content.to_string());
        let mut top_level_keys = Vec::new();
        let mut field_samples = Vec::new();
        let summary = match &json {
            Value::Object(map) => {
                top_level_keys = map.keys().take(12).cloned().collect::<Vec<_>>();
                if map.len() > 12 {
                    top_level_keys.push("...".to_string());
                }
                field_samples = map
                    .iter()
                    .take(6)
                    .map(|(key, value)| format!("{}: {}", key, json_value_sample(value, 90)))
                    .collect();
                format!("JSON object with {} top-level keys", map.len())
            }
            Value::Array(arr) => {
                if let Some(Value::Object(map)) = arr.first() {
                    top_level_keys = map.keys().take(12).cloned().collect::<Vec<_>>();
                    if map.len() > 12 {
                        top_level_keys.push("...".to_string());
                    }
                    field_samples = map
                        .iter()
                        .take(6)
                        .map(|(key, value)| format!("{}: {}", key, json_value_sample(value, 90)))
                        .collect();
                } else if let Some(first) = arr.first() {
                    field_samples.push(format!("item[0]: {}", json_value_sample(first, 90)));
                }
                format!("JSON array with {} items", arr.len())
            }
            other => format!("JSON {} value", json_type_name(other)),
        };
        return LargeToolSummary {
            body: pretty,
            summary,
            top_level_keys,
            field_samples,
        };
    }

    let important = content
        .lines()
        .map(str::trim)
        .find(|line| {
            let lower = line.to_ascii_lowercase();
            !line.is_empty()
                && (lower.contains("error")
                    || lower.contains("failed")
                    || lower.contains("panic")
                    || lower.contains("exception")
                    || lower.contains("timeout"))
        })
        .map(|s| s.to_string());
    let fallback = content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .to_string();
    let summary = important.unwrap_or(fallback);
    LargeToolSummary {
        body: content.to_string(),
        summary: truncate_chars(&summary, 240),
        top_level_keys: Vec::new(),
        field_samples: Vec::new(),
    }
}

pub(super) fn write_tool_overflow_file(
    app: &App,
    tool_name: &str,
    body: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let store = SessionStore::new(app.session_history_file.as_path());
    store.ensure_root_dir()?;
    let dir = store.session_assets_dir(&app.session_id).join("tool-overflow");
    fs::create_dir_all(&dir)?;
    let sanitized_name = tool_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();
    let filename = format!(
        "{}-{}-{}.txt",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        sanitized_name,
        uuid::Uuid::new_v4().simple()
    );
    let path = dir.join(filename);
    fs::write(&path, body)?;
    Ok(path.canonicalize().unwrap_or(path))
}

pub(super) fn build_model_overflow_stub(
    path: Option<&PathBuf>,
    summary: &LargeToolSummary,
) -> String {
    let overflow_notice = if let Some(path) = path {
        format!(
            "Output too large; full result saved to session file.\n- file_path: {}\n- summary: {}\n",
            path.display(),
            summary.summary
        )
    } else {
        format!(
            "Output too large; full result omitted from context.\n- summary: {}\n",
            summary.summary
        )
    };

    let mut content_for_model = overflow_notice;
    if !summary.top_level_keys.is_empty() {
        content_for_model.push_str("- top_level_keys:\n");
        for key in &summary.top_level_keys {
            content_for_model.push_str(&format!("  - {key}\n"));
        }
    }
    if !summary.field_samples.is_empty() {
        content_for_model.push_str("- field_samples:\n");
        for sample in &summary.field_samples {
            content_for_model.push_str(&format!("  - {sample}\n"));
        }
    }
    content_for_model.push_str(&format!(
        "- tail_preview: {}\n",
        tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS)
    ));
    content_for_model
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn json_value_sample(value: &Value, max_chars: usize) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => format!("{:?}", truncate_chars(v, max_chars)),
        Value::Array(arr) => {
            if let Some(first) = arr.first() {
                format!(
                    "array(len={}, first={})",
                    arr.len(),
                    json_value_sample(first, max_chars / 2)
                )
            } else {
                "array(len=0)".to_string()
            }
        }
        Value::Object(map) => {
            let mut keys = map.keys().take(5).cloned().collect::<Vec<_>>();
            if map.len() > 5 {
                keys.push("...".to_string());
            }
            format!("object(keys={})", keys.join(", "))
        }
    }
}
