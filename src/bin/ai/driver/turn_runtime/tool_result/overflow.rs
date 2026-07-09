use std::{fs, path::PathBuf};

use serde_json::Value;

use crate::ai::{history::SessionStore, types::App};

use super::super::{TOOL_OVERFLOW_HEAD_CHARS, TOOL_OVERFLOW_PREVIEW_CHARS, types::LargeToolSummary};
use super::preview::{tail_chars, truncate_chars};

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
            key_lines: Vec::new(),
        };
    }

    // 对文本内容提取结构化关键行（函数/类型定义、错误行等），
    // 让模型拿到"文件大纲"级别的召回锚点，而非只有一行 summary。
    let key_lines = extract_key_lines(content, 20);

    // summary 仍取第一个错误行（如果有），否则取第一个非空行。
    let summary = content
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
        .or_else(|| content.lines().map(str::trim).find(|line| !line.is_empty()))
        .unwrap_or("")
        .to_string();
    LargeToolSummary {
        body: content.to_string(),
        summary: truncate_chars(&summary, 240),
        top_level_keys: Vec::new(),
        field_samples: Vec::new(),
        key_lines,
    }
}

pub(super) fn write_tool_overflow_file(
    app: &App,
    tool_name: &str,
    body: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let store = SessionStore::new(app.config.history_file.as_path());
    store.ensure_root_dir()?;
    let dir = store
        .session_assets_dir(&app.session_id)
        .join("tool-overflow");
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
            "Output too large; full result saved to a file. The COMPLETE output is NOT in context.\n\
             To read it, call read_file on this path (read in chunks, e.g. offset=1, limit=200), \
             then narrow to precise line ranges once located.\n\
             The summary / tail_preview below are PARTIAL — do not rely on them if you need full detail.\n\
             - file_path: {}\n- summary: {}\n",
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
    // 文本内容：追加结构化关键行和 head 预览，让模型有足够的召回锚点
    // 来判断是否需要重新 read_file，而不必盲目重读。
    if !summary.key_lines.is_empty() {
        content_for_model.push_str(&format!("- key_lines ({}):\n", summary.key_lines.len()));
        for line in &summary.key_lines {
            content_for_model.push_str(&format!("  {line}\n"));
        }
    }
    content_for_model.push_str(&format!(
        "- head_preview: {}\n",
        truncate_chars(&summary.body, TOOL_OVERFLOW_HEAD_CHARS)
    ));
    content_for_model.push_str(&format!(
        "- tail_preview: {}\n",
        tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS)
    ));
    content_for_model
}

/// 从文本内容中提取结构化关键行，为 overflow stub 提供召回锚点。
///
/// 识别的行类型（与 `line_trim_middle` 的中段采样保持一致）：
/// - Rust/代码结构：`fn`/`pub fn`/`impl`/`struct`/`trait`/`enum`/`#[`/`mod`
/// - 文档注释：`//!`/`///`
/// - 错误/警告：`error`/`failed`/`panic`/`exception`/`timeout`/`warning`
/// - 标记：`TODO`/`FIXME`
///
/// 每行截断到 200 字符以控制 stub 体积。最多保留 `max` 行。
fn extract_key_lines(content: &str, max: usize) -> Vec<String> {
    let mut result = Vec::with_capacity(max);
    for (idx, line) in content.lines().enumerate() {
        if result.len() >= max {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        let is_key = lower.starts_with("fn ")
            || lower.starts_with("pub fn ")
            || lower.starts_with("pub(crate) fn ")
            || lower.starts_with("pub(super) fn ")
            || lower.starts_with("async fn ")
            || lower.starts_with("pub async fn ")
            || lower.starts_with("impl ")
            || lower.starts_with("struct ")
            || lower.starts_with("pub struct ")
            || lower.starts_with("trait ")
            || lower.starts_with("enum ")
            || lower.starts_with("pub enum ")
            || lower.starts_with("mod ")
            || lower.starts_with("#[")
            || lower.starts_with("//!")
            || lower.starts_with("///")
            || lower.starts_with("class ")
            || lower.starts_with("def ")
            || lower.starts_with("func ")
            || lower.starts_with("interface ")
            || lower.starts_with("type ")
            || lower.starts_with("pub type ")
            || lower.starts_with("const ")
            || lower.starts_with("pub const ")
            || lower.starts_with("use ")
            || lower.contains("error")
            || lower.contains("failed")
            || lower.contains("panic")
            || lower.contains("exception")
            || lower.contains("timeout")
            || lower.contains("warning")
            || lower.contains("todo")
            || lower.contains("fixme")
            || lower.contains(": error")
            || lower.contains(": warning");
        if is_key {
            let truncated = if trimmed.chars().count() > 200 {
                let kept: String = trimmed.chars().take(200).collect();
                format!("L{idx}: {kept} …")
            } else {
                format!("L{idx}: {trimmed}")
            };
            result.push(truncated);
        }
    }
    result
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
