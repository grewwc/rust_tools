//! Always-on request-side records for `<session assets>/stream_outcomes.jsonl`.
//!
//! The stream side (`stream::wire_log`) only records finalized responses: a
//! request rejected by the gateway (e.g. HTTP 400) never produces a stream,
//! so without this module neither the request controls nor the gateway error
//! body are persisted anywhere. Each record is one small JSONL line,
//! discriminated by `kind` (`request` / `request_error`).
//!
//! Design constraints (mirroring the stream-side writer):
//! - Best-effort: recording never influences the request; failures are dropped.
//! - Bounded: error bodies are truncated, tool-name lists are capped, and the
//!   file rotates past 8 MiB. Message *contents* are never copied (canonical
//!   history already persists them); only counts and sizes are recorded.
//! - No secrets: API keys travel in headers and are never part of the logged
//!   fields; the endpoint URL carries no credentials.
//! - Layering: `request` must not depend on `stream`, so the tiny
//!   append/rotate boilerplate is duplicated here instead of shared. Both
//!   writers append whole lines with `O_APPEND`, so concurrent writers cannot
//!   interleave bytes inside a line.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

use chrono::Utc;
use serde_json::{Value, json};

use crate::ai::driver::runtime_ctx::{current_subagent_depth, current_turn_id_or_zero};
use crate::ai::history::Message;
use crate::ai::tools::storage::file_store::current_session_assets_dir;

use super::types::RequestBody;

/// Outcome file shared with the stream-side writer.
const OUTCOME_LOG_FILE: &str = "stream_outcomes.jsonl";

/// Rotation threshold, mirroring the stream-side writer.
const MAX_LOG_BYTES: u64 = 8 * 1024 * 1024;

/// Error-body cap: gateway error payloads are short; beyond this they are cut.
const MAX_ERROR_BODY_CHARS: usize = 4096;

/// Tool-name cap for the request record.
const MAX_TOOL_NAMES: usize = 32;

static APPEND_LOCK: Mutex<()> = Mutex::new(());

/// Append one request-controls line. Call after the provider adapter hook has
/// run (`build_http_body_for_request`), so the captured fields are what the
/// wire serialization actually sees.
pub(in crate::ai) fn append_request_record(
    session_id: &str,
    model: &str,
    endpoint: &str,
    thinking_disabled_override: bool,
    effort_override: Option<Option<&str>>,
    request: &RequestBody<'_>,
) {
    append_line(&request_record_json(
        session_id,
        model,
        endpoint,
        thinking_disabled_override,
        effort_override,
        request,
    ));
}

/// Append one gateway-rejection line (non-2xx with a response body).
pub(in crate::ai) fn append_request_error(
    session_id: &str,
    model: &str,
    endpoint: &str,
    status: u16,
    body: &str,
) {
    append_line(&json!({
        "kind": "request_error",
        "ts": Utc::now().to_rfc3339(),
        "session": session_id,
        "turn": current_turn_id_or_zero(),
        "depth": current_subagent_depth(),
        "model": model,
        "endpoint": endpoint,
        "status": status,
        "body": truncate_chars(body, MAX_ERROR_BODY_CHARS),
    }));
}

fn request_record_json(
    session_id: &str,
    model: &str,
    endpoint: &str,
    thinking_disabled_override: bool,
    effort_override: Option<Option<&str>>,
    request: &RequestBody<'_>,
) -> Value {
    let (tool_count, tool_names) = tool_summary(request.tools.as_ref());
    let (message_count, message_chars) = message_stats(request.messages);
    let effort_override_label = match effort_override {
        None => Value::Null,
        Some(None) => Value::String("off".to_string()),
        Some(Some(tier)) => Value::String(tier.to_string()),
    };
    json!({
        "kind": "request",
        "ts": Utc::now().to_rfc3339(),
        "session": session_id,
        "turn": current_turn_id_or_zero(),
        "depth": current_subagent_depth(),
        "model": model,
        "endpoint": endpoint,
        "controls": {
            "reasoning_effort": request.reasoning_effort,
            "thinking": request.thinking,
            "reasoning": request.reasoning,
            "tools": { "count": tool_count, "names": tool_names },
            "tool_choice": request.tool_choice.is_some(),
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        },
        "overrides": {
            "thinking_disabled": thinking_disabled_override,
            "effort": effort_override_label,
        },
        "messages": { "count": message_count, "chars": message_chars },
    })
}

/// `(total tool count, function names)`; names are capped, the count is exact.
fn tool_summary(tools: Option<&Value>) -> (usize, Vec<String>) {
    let Some(Value::Array(items)) = tools else {
        return (0, Vec::new());
    };
    let mut names = Vec::new();
    for item in items.iter().take(MAX_TOOL_NAMES) {
        let name = item
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str);
        if let Some(name) = name {
            names.push(name.to_string());
        }
    }
    (items.len(), names)
}

/// `(message count, content chars)`; contents are measured, never copied.
fn message_stats(messages: &[Message]) -> (usize, usize) {
    let mut chars = 0;
    for message in messages {
        chars += match &message.content {
            Value::String(text) => text.chars().count(),
            other => serde_json::to_string(other)
                .map(|rendered| rendered.chars().count())
                .unwrap_or(0),
        };
    }
    (messages.len(), chars)
}

/// Char-boundary-safe truncation for untrusted gateway payloads.
fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let end = text
        .char_indices()
        .map(|(index, _)| index)
        .nth(max)
        .unwrap_or(text.len());
    format!("{}…[truncated]", &text[..end])
}

fn append_line(record: &Value) {
    let Some(dir) = current_session_assets_dir() else {
        return;
    };
    let Ok(mut line) = serde_json::to_string(record) else {
        return;
    };
    line.push('\n');
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(OUTCOME_LOG_FILE);
    rotate_if_large(&path);
    let _guard = APPEND_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = file.write_all(line.as_bytes());
    }
}

fn rotate_if_large(path: &Path) {
    let too_large = std::fs::metadata(path)
        .map(|meta| meta.len() > MAX_LOG_BYTES)
        .unwrap_or(false);
    if too_large {
        let name = path
            .file_name()
            .map(|name| format!("{}.1", name.to_string_lossy()))
            .unwrap_or_default();
        let _ = std::fs::rename(path, path.with_file_name(name));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    fn message(role: &str, content: Value) -> Message {
        Message {
            role: role.to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn tool_summary_extracts_function_names_and_caps() {
        assert_eq!(tool_summary(None), (0, Vec::new()));
        let tools = json!([
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "other"},
            {"type": "function", "function": {"name": "execute_command"}},
        ]);
        let (count, names) = tool_summary(Some(&tools));
        assert_eq!(count, 3);
        assert_eq!(names, vec!["read_file", "execute_command"]);
    }

    #[test]
    fn message_stats_measures_without_copying_content() {
        let secret = "super-secret-prompt-content".to_string();
        let secret_chars = secret.chars().count();
        let messages = vec![
            message("user", Value::String(secret)),
            message("assistant", json!([{"type": "text", "text": "hi"}])),
        ];
        let (count, chars) = message_stats(&messages);
        assert_eq!(count, 2);
        // String chars + the serialized JSON length of the parts array.
        assert_eq!(
            chars,
            secret_chars + json!([{"type": "text", "text": "hi"}]).to_string().chars().count()
        );
    }

    #[test]
    fn truncate_chars_respects_char_boundaries() {
        assert_eq!(truncate_chars("abc", 8), "abc");
        // Multibyte cut must not split a character.
        assert_eq!(truncate_chars("a好b", 2), "a好…[truncated]");
    }

    #[test]
    fn request_record_pins_schema_and_wire_values() {
        let messages = vec![message("user", Value::String("hello".to_string()))];
        let mut thinking = Map::new();
        thinking.insert("enable_thinking".to_string(), Value::Bool(true));
        let tools = json!([{"type": "function", "function": {"name": "read_file"}}]);
        let request = RequestBody {
            model: "beam-free".to_string(),
            messages: &messages,
            stream: true,
            thinking,
            enable_search: None,
            tools: Some(tools),
            tool_choice: None,
            reasoning_effort: Some("max"),
            reasoning: None,
            stream_options: None,
            max_tokens: None,
            reasoning_items: None,
            tool_errors: None,
            reasoning_encrypted_replay: false,
            estimated_prompt_tokens: 0,
            estimated_prompt_tokens_upper: 0,
            message_counts: Default::default(),
            prompt_cache_key: None,
            explicit_cache_breakpoints: false,
        };
        let record = request_record_json("sess", "beam-free", "https://example.test/v1", false, None, &request);
        assert_eq!(record["kind"], "request");
        assert_eq!(record["controls"]["reasoning_effort"], "max");
        assert_eq!(record["controls"]["thinking"]["enable_thinking"], true);
        assert_eq!(record["controls"]["tools"]["count"], 1);
        assert_eq!(record["controls"]["tools"]["names"][0], "read_file");
        assert_eq!(record["overrides"]["thinking_disabled"], false);
        assert!(record["overrides"]["effort"].is_null());
        assert_eq!(record["messages"], json!({"count": 1, "chars": 5}));
    }

    #[test]
    fn appenders_are_noop_without_a_session_dir() {
        // Unit tests run without a driver context: nothing is written, nothing panics.
        append_request_error("sess", "model", "https://example.test/v1", 400, r#"{"error":{}}"#);
        assert!(current_session_assets_dir().is_none());
    }
}
