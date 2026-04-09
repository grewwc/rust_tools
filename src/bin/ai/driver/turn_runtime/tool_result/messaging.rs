use colored::Colorize;
use serde_json::Value;

use crate::ai::{
    driver::{print::print_tool_output_block, tools::ExecuteToolCallsResult},
    history::Message,
    types::App,
};

use super::execution::prepare_tool_result;
use super::super::types::PreparedToolResult;

pub(super) fn append_message_pair(
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    message: Message,
) {
    messages.push(message.clone());
    turn_messages.push(message);
}

pub(super) fn record_hidden_self_note(app: &App, turn_messages: &mut Vec<Message>, hidden_meta: &str) {
    let hidden_meta = hidden_meta.trim();
    if hidden_meta.is_empty() {
        return;
    }

    let record = Message {
        role: "system".to_string(),
        content: Value::String(format!("self_note:\n{hidden_meta}")),
        tool_calls: None,
        tool_call_id: None,
    };
    turn_messages.push(record);

    let entry = crate::ai::tools::storage::memory_store::AgentMemoryEntry {
        id: None,
        timestamp: chrono::Local::now().to_rfc3339(),
        category: "self_note".to_string(),
        note: hidden_meta.to_string(),
        tags: vec!["agent".to_string(), "policy".to_string()],
        source: Some(format!("session:{}", app.session_id)),
        priority: Some(255),
    };
    let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

pub(super) fn append_cached_tool_results_note(
    exec_result: &ExecuteToolCallsResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) {
    if !exec_result.cached_hits.iter().any(|hit| *hit) {
        return;
    }

    let cached_names = exec_result
        .executed_tool_calls
        .iter()
        .zip(exec_result.cached_hits.iter())
        .filter_map(|(tool_call, cached)| cached.then_some(tool_call.function.name.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let cache_note = Message {
        role: "system".to_string(),
        content: Value::String(format!(
            "Context note: reused cached tool results from the current session for identical calls within the recent TTL. Treat these results as already verified context unless the user asks to refresh. Tools: {cached_names}"
        )),
        tool_calls: None,
        tool_call_id: None,
    };
    append_message_pair(messages, turn_messages, cache_note);
}

pub(super) fn print_tool_result_preview(tool_name: &str, prepared: &PreparedToolResult) {
    println!(
        "\n{} {}",
        "[Tool]".bright_green().bold(),
        tool_name.bright_cyan().bold()
    );
    print_tool_output_block(&prepared.content_for_terminal);
}

pub(super) fn append_tool_result_messages(
    app: &App,
    stream_assistant_text: &str,
    exec_result: &ExecuteToolCallsResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) {
    let assistant_msg = Message {
        role: "assistant".to_string(),
        content: Value::String(stream_assistant_text.to_string()),
        tool_calls: Some(exec_result.executed_tool_calls.clone()),
        tool_call_id: None,
    };
    append_message_pair(messages, turn_messages, assistant_msg);

    for (tool_call, result) in exec_result
        .executed_tool_calls
        .iter()
        .zip(exec_result.tool_results.iter())
    {
        let prepared = prepare_tool_result(app, &tool_call.function.name, &result.content);
        print_tool_result_preview(&tool_call.function.name, &prepared);
        let tool_message = Message {
            role: "tool".to_string(),
            content: Value::String(prepared.content_for_model),
            tool_calls: None,
            tool_call_id: Some(result.tool_call_id.clone()),
        };
        append_message_pair(messages, turn_messages, tool_message);
    }
}

pub(super) fn record_final_stream_response(
    app: &App,
    stream_result: crate::ai::types::StreamResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    final_assistant_text: &mut String,
    final_assistant_recorded: &mut bool,
) {
    let assistant_msg = Message {
        role: "assistant".to_string(),
        content: Value::String(stream_result.assistant_text.clone()),
        tool_calls: None,
        tool_call_id: None,
    };
    append_message_pair(messages, turn_messages, assistant_msg);
    *final_assistant_text = stream_result.assistant_text;
    *final_assistant_recorded = true;
    record_hidden_self_note(app, turn_messages, &stream_result.hidden_meta);
}
