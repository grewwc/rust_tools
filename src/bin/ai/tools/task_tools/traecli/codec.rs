//! Pure TraeCLI result, checkpoint, and item-text decoding.
//! The parent runner owns live state and event publication.

use serde_json::Value;

use super::TraeCliOutcome;

/// Decode stdout without process status; the runner applies the exit status to
/// the same reduction before publishing evidence or returning success.
pub(super) fn parse_stream_json_output(output: &str) -> Option<TraeCliOutcome> {
    reduce_stream_json_output(output).outcome
}

pub(super) struct StreamReduction {
    pub(super) outcome: Option<TraeCliOutcome>,
    pub(super) error: Option<String>,
}

/// Keep the last non-empty body independently of protocol failure. A completed
/// turn clears earlier recoverable errors; a later error or failed turn remains
/// a failure even when an answer has already been streamed.
pub(super) fn reduce_stream_json_output(output: &str) -> StreamReduction {
    let mut outcome = TraeCliOutcome::default();
    let mut turns: u64 = 0;
    let mut input_tokens: u64 = 0;
    let mut output_tokens: u64 = 0;
    let mut saw_usage = false;
    let mut message: Option<String> = None;
    let mut error_text: Option<String> = None;

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.starts_with('{') {
            continue;
        }
        let value: Value = match serde_json::from_str(trimmed) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let event_type = value.get("type").and_then(Value::as_str).unwrap_or("");
        match event_type {
            "thread.started" => {
                if outcome.session_id.is_none() {
                    outcome.session_id = value
                        .get("thread_id")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
            }
            "item.completed" => {
                let item = value.get("item");
                let item_type = item.and_then(|i| i.get("type")).and_then(Value::as_str);
                if item_type == Some("agent_message") {
                    if let Some(text) = item.and_then(|i| i.get("text")).and_then(Value::as_str) {
                        if !text.trim().is_empty() {
                            let text = traecli_message_text(text, |_, _| {});
                            if !text.trim().is_empty() {
                                message = Some(text);
                            }
                        }
                    }
                }
            }
            "turn.completed" => {
                error_text = None;
                turns += 1;
                if let Some(usage) = value.get("usage") {
                    if let Some(n) = usage.get("input_tokens").and_then(Value::as_u64) {
                        input_tokens += n;
                        saw_usage = true;
                    }
                    if let Some(n) = usage.get("output_tokens").and_then(Value::as_u64) {
                        output_tokens += n;
                        saw_usage = true;
                    }
                }
            }
            "error" | "turn.failed" => {
                error_text = Some(protocol_error_text(&value).unwrap_or(trimmed).to_string());
            }
            _ => {}
        }
    }

    if turns == 0 && message.is_none() && error_text.is_none() {
        return StreamReduction {
            outcome: None,
            error: None,
        };
    }
    if turns > 0 {
        outcome.num_turns = Some(turns);
    }
    if saw_usage {
        outcome.input_tokens = Some(input_tokens);
        outcome.output_tokens = Some(output_tokens);
    }
    outcome.is_error = message.is_none() || error_text.is_some();
    outcome.result = message.unwrap_or_else(|| error_text.clone().unwrap_or_default());
    StreamReduction {
        outcome: Some(outcome),
        error: error_text,
    }
}

pub(super) fn protocol_error_text(value: &Value) -> Option<&str> {
    [
        value.get("message").and_then(Value::as_str),
        value.get("error").and_then(Value::as_str),
        value
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str),
    ]
    .into_iter()
    .flatten()
    .find(|text| !text.trim().is_empty())
}

/// An execution signal is sticky for one attempt. Unknown item types may be
/// new tools, so only known bookkeeping and reasoning items are replayable.
pub(super) fn event_may_have_effect(value: &Value) -> bool {
    if !matches!(
        value.get("type").and_then(Value::as_str),
        Some("item.started" | "item.updated" | "item.completed" | "item.failed")
    ) {
        return false;
    }
    match value
        .get("item")
        .and_then(|item| item.get("type"))
        .and_then(Value::as_str)
    {
        Some("reasoning" | "todo_list" | "error") => false,
        Some("agent_message") => {
            value.get("type").and_then(Value::as_str) == Some("item.completed")
                && value
                    .get("item")
                    .and_then(|item| item.get("text"))
                    .and_then(Value::as_str)
                    .is_some_and(|text| !text.trim().is_empty())
        }
        _ => true,
    }
}

/// Only complete explicit checkpoint blocks become parent progress; ordinary messages,
/// reasoning, and tool output stay out of the checkpoint carrier.
pub(super) fn traecli_message_text(text: &str, mut checkpoint: impl FnMut(&str, &str)) -> String {
    const OPEN: &str = "<context_checkpoint>";
    const CLOSE: &str = "</context_checkpoint>";
    let mut remaining = String::new();
    let mut cursor = text;
    while let Some(start) = cursor.find(OPEN) {
        remaining.push_str(&cursor[..start]);
        let after_open = &cursor[start + OPEN.len()..];
        let Some(end) = after_open.find(CLOSE) else {
            remaining.push_str(&cursor[start..]);
            return remaining;
        };
        let raw = after_open[..end].trim();
        let (summary, body) = raw.split_once('\n').unwrap_or((raw, ""));
        let summary = summary
            .trim()
            .strip_prefix("summary:")
            .unwrap_or(summary)
            .trim();
        let body = body.trim();
        if !summary.is_empty() && !body.is_empty() {
            checkpoint(summary, body);
        } else {
            remaining.push_str(&cursor[start..start + OPEN.len() + end + CLOSE.len()]);
        }
        cursor = &after_open[end + CLOSE.len()..];
    }
    remaining.push_str(cursor);
    remaining
}

pub(super) fn traecli_item_id(value: &Value) -> Option<&str> {
    value
        .get("item")
        .and_then(|item| item.get("id"))
        .or_else(|| value.get("item_id"))
        .and_then(Value::as_str)
}

pub(super) fn traecli_item_phase(value: &Value) -> &'static str {
    match value
        .get("item")
        .and_then(|item| item.get("type"))
        .and_then(Value::as_str)
    {
        Some("reasoning") => "thinking",
        Some("agent_message") => "writing response",
        Some("command_execution") => "running command",
        Some("tool_call" | "tool_use") => "running tool",
        _ => "working",
    }
}

/// Summarize a streamed item for the status line without echoing full tool
/// output. Commands keep the command; tool calls keep the tool name and a
/// compact argument preview.
pub(super) fn traecli_item_activity(item: &Value) -> String {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
    match item_type {
        "command_execution" => item
            .get("command")
            .and_then(Value::as_str)
            .map(|command| format!("command: {}", truncate_head(command, 80)))
            .unwrap_or_else(|| "command started".to_string()),
        "tool_call" | "tool_use" => {
            let name = item
                .get("name")
                .or_else(|| item.get("tool_name"))
                .and_then(Value::as_str)
                .unwrap_or("tool");
            match traecli_item_input(item) {
                Some(input) if !input.is_empty() => {
                    format!("tool {name}: {}", truncate_head(&input, 70))
                }
                _ => format!("tool {name}"),
            }
        }
        "reasoning" => "reasoning".to_string(),
        "agent_message" => "drafting response".to_string(),
        _ if item_type.is_empty() => "item started".to_string(),
        _ => format!("item: {item_type}"),
    }
}

pub(super) fn traecli_item_completion(item: &Value) -> String {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
    match item_type {
        "command_execution" => {
            let command = item
                .get("command")
                .and_then(Value::as_str)
                .map(|command| truncate_head(command, 70))
                .unwrap_or_else(|| "command".to_string());
            match item.get("exit_code").and_then(Value::as_i64) {
                Some(code) => format!("completed: {command} · exit {code}"),
                None => format!("completed: {command}"),
            }
        }
        "tool_call" | "tool_use" => {
            let name = item
                .get("name")
                .or_else(|| item.get("tool_name"))
                .and_then(Value::as_str)
                .unwrap_or("tool");
            format!("completed tool {name}")
        }
        "agent_message" => "response ready".to_string(),
        "reasoning" => "reasoning complete".to_string(),
        _ => "processing output".to_string(),
    }
}

fn traecli_item_input(item: &Value) -> Option<String> {
    let value = item
        .get("input")
        .or_else(|| item.get("arguments"))
        .or_else(|| item.get("parameters"))?;
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Null => None,
        other => serde_json::to_string(other).ok(),
    }
}

/// First non-empty, non-JSON line of stdout, if any.
pub(super) fn stdout_hint(output: &str) -> Option<String> {
    output
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('{'))
        .map(|line| truncate_head(line, 500))
}

/// Char-boundary-safe head truncation with an ellipsis marker.
pub(super) fn truncate_head(text: &str, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        let mut out: String = text.chars().take(max_chars).collect();
        out.push_str("...");
        out
    } else {
        text.to_string()
    }
}
