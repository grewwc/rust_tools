use super::*;

const TOOL_TERMINAL_PREVIEW_MAX_CHARS: usize = 4_000;
const TOOL_TERMINAL_PREVIEW_MAX_LINES: usize = 80;
const TOOL_TERMINAL_PREVIEW_HEAD_LINES: usize = 40;
const TOOL_TERMINAL_PREVIEW_TAIL_LINES: usize = 20;
const READ_FILE_TERMINAL_PREVIEW_MAX_CHARS: usize = 2_200;
const READ_FILE_TERMINAL_PREVIEW_MAX_LINES: usize = 40;
const READ_FILE_TERMINAL_PREVIEW_HEAD_LINES: usize = 24;
const READ_FILE_TERMINAL_PREVIEW_TAIL_LINES: usize = 8;
const WEB_SEARCH_TERMINAL_PREVIEW_MAX_CHARS: usize = 1_800;
const WEB_SEARCH_TERMINAL_PREVIEW_MAX_LINES: usize = 18;

struct ToolTerminalPreviewPolicy {
    max_chars: usize,
    max_lines: usize,
    head_lines: usize,
    tail_lines: usize,
    summary_first: bool,
}

fn terminal_preview_policy(tool_name: &str) -> ToolTerminalPreviewPolicy {
    match tool_name {
        "read_file_lines" | "read_file" => ToolTerminalPreviewPolicy {
            max_chars: READ_FILE_TERMINAL_PREVIEW_MAX_CHARS,
            max_lines: READ_FILE_TERMINAL_PREVIEW_MAX_LINES,
            head_lines: READ_FILE_TERMINAL_PREVIEW_HEAD_LINES,
            tail_lines: READ_FILE_TERMINAL_PREVIEW_TAIL_LINES,
            summary_first: false,
        },
        "web_search" => ToolTerminalPreviewPolicy {
            max_chars: WEB_SEARCH_TERMINAL_PREVIEW_MAX_CHARS,
            max_lines: WEB_SEARCH_TERMINAL_PREVIEW_MAX_LINES,
            head_lines: 12,
            tail_lines: 0,
            summary_first: true,
        },
        _ => ToolTerminalPreviewPolicy {
            max_chars: TOOL_TERMINAL_PREVIEW_MAX_CHARS,
            max_lines: TOOL_TERMINAL_PREVIEW_MAX_LINES,
            head_lines: TOOL_TERMINAL_PREVIEW_HEAD_LINES,
            tail_lines: TOOL_TERMINAL_PREVIEW_TAIL_LINES,
            summary_first: false,
        },
    }
}

fn truncate_chars(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let mut out = String::with_capacity(max_chars + 32);
    for (i, ch) in content.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push('…');
    out
}

fn build_terminal_preview(tool_name: &str, content: &str) -> String {
    let policy = terminal_preview_policy(tool_name);
    let line_count = content.lines().count();
    let char_count = content.chars().count();
    if line_count <= policy.max_lines && char_count <= policy.max_chars {
        return content.to_string();
    }

    if policy.summary_first {
        let mut preview = String::new();
        let mut kept = 0usize;
        for line in content.lines().map(str::trim).filter(|line| !line.is_empty()) {
            preview.push_str(line);
            preview.push('\n');
            kept += 1;
            if kept >= policy.head_lines || preview.chars().count() >= policy.max_chars {
                break;
            }
        }
        if preview.is_empty() {
            preview = truncate_chars(content, policy.max_chars);
        }
        return format!(
            "{}\n... (summary-first terminal preview; {} lines, {} chars total)",
            preview.trim_end(),
            line_count,
            char_count
        );
    }

    if line_count <= 1 {
        let head_budget = policy.max_chars / 2;
        let tail_budget = policy.max_chars.saturating_sub(head_budget);
        let head = truncate_chars(content, head_budget);
        let tail = tail_chars(content, tail_budget);
        return format!(
            "{head}\n... (truncated for terminal preview; {} chars total)\n{tail}",
            char_count
        );
    }

    let lines = content.lines().collect::<Vec<_>>();
    let omitted = line_count.saturating_sub(policy.head_lines + policy.tail_lines);
    let mut preview = String::new();
    for line in lines.iter().take(policy.head_lines) {
        preview.push_str(line);
        preview.push('\n');
    }
    preview.push_str(&format!(
        "... (truncated for terminal preview; {} lines, {} chars total",
        line_count, char_count
    ));
    if omitted > 0 {
        preview.push_str(&format!(", {} lines omitted", omitted));
    }
    preview.push_str(")\n");
    if policy.tail_lines > 0 && line_count > policy.head_lines {
        let tail_start = line_count.saturating_sub(policy.tail_lines);
        for line in lines.iter().skip(tail_start) {
            preview.push_str(line);
            preview.push('\n');
        }
    }

    let contains_truncation_notice = preview.contains("truncated for terminal preview");
    
    if preview.chars().count() > policy.max_chars {
        let truncated = truncate_chars(&preview, policy.max_chars);
        
        let suffix = if contains_truncation_notice {
            "... (truncated for terminal preview)"
        } else {
            "... (terminal preview capped)"
        };
        return format!(
            "{}\n{}",
            truncated, suffix
        );
    }

    preview
}

fn summarize_large_tool_output(content: &str) -> LargeToolSummary {
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

fn tail_chars(content: &str, max_chars: usize) -> String {
    let total = content.chars().count();
    if total <= max_chars {
        return content.to_string();
    }
    let skip = total.saturating_sub(max_chars);
    let mut out = String::with_capacity(max_chars + 32);
    out.push('…');
    for (i, ch) in content.chars().enumerate() {
        if i < skip {
            continue;
        }
        out.push(ch);
    }
    out
}

fn write_tool_overflow_file(
    app: &App,
    tool_name: &str,
    body: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let store = SessionStore::new(app.session_history_file.as_path());
    store.ensure_root_dir()?;
    let mut dir = store.session_history_file(&app.session_id);
    if let Some(parent) = dir.parent() {
        fs::create_dir_all(parent)?;
    }
    dir.set_extension("");
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

pub(super) fn prepare_tool_result(app: &App, tool_name: &str, content: &str) -> PreparedToolResult {
    if content.chars().count() <= MAX_TOOL_RESULT_INLINE_CHARS {
        return PreparedToolResult {
            content_for_model: content.to_string(),
            content_for_terminal: build_terminal_preview(tool_name, content),
        };
    }

    let summary = summarize_large_tool_output(content);
    let path = write_tool_overflow_file(app, tool_name, &summary.body).ok();
    let overflow_notice = if let Some(path) = path.as_ref() {
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

    let content_for_terminal = if let Some(path) = path {
        format!(
            "{}\n[Saved full output to {}]\n",
            build_terminal_preview(tool_name, &tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS)),
            path.display()
        )
    } else {
        build_terminal_preview(tool_name, &tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS))
    };

    PreparedToolResult {
        content_for_model,
        content_for_terminal,
    }
}

fn append_message_pair(messages: &mut Vec<Message>, turn_messages: &mut Vec<Message>, message: Message) {
    messages.push(message.clone());
    turn_messages.push(message);
}

fn record_hidden_self_note(app: &App, turn_messages: &mut Vec<Message>, hidden_meta: &str) {
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

fn append_cached_tool_results_note(
    exec_result: &tools::ExecuteToolCallsResult,
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

fn print_tool_result_preview(tool_name: &str, _result_content: &str, prepared: &PreparedToolResult) {
    println!(
        "\n{} {}",
        "[Tool]".bright_green().bold(),
        tool_name.bright_cyan().bold()
    );
    print_tool_output_block(&prepared.content_for_terminal);
}

fn append_tool_result_messages(
    app: &App,
    stream_assistant_text: &str,
    exec_result: &tools::ExecuteToolCallsResult,
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
        print_tool_result_preview(&tool_call.function.name, &result.content, &prepared);
        let tool_message = Message {
            role: "tool".to_string(),
            content: Value::String(prepared.content_for_model),
            tool_calls: None,
            tool_call_id: Some(result.tool_call_id.clone()),
        };
        append_message_pair(messages, turn_messages, tool_message);
    }
}

fn record_final_stream_response(
    app: &App,
    stream_result: StreamResult,
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

#[crate::ai::agent_hang_span(
    "pre-fix",
    "C",
    "turn_runtime::run_turn:execute_tool_calls",
    "[DEBUG] executing tool calls",
    "[DEBUG] executed tool calls",
    {
        "iteration": _iteration,
        "tool_calls": tool_calls
            .iter()
            .map(|tool| tool.function.name.clone())
            .collect::<Vec<_>>(),
    },
    {
        "iteration": _iteration,
        "tool_result_count": __agent_hang_result
            .as_ref()
            .map(|v| v.tool_results.len())
            .unwrap_or(0),
        "cached_hits": __agent_hang_result
            .as_ref()
            .map(|v| v.cached_hits.clone())
            .unwrap_or_default(),
        "ok": __agent_hang_result.is_ok(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
fn execute_tool_calls_for_round(
    session_id: &str,
    mcp_client: &mut McpClient,
    tool_calls: &[crate::ai::types::ToolCall],
    _iteration: usize,
) -> Result<tools::ExecuteToolCallsResult, Box<dyn std::error::Error>> {
    tools::execute_tool_calls(session_id, mcp_client, tool_calls)
}

fn handle_tool_call_round(
    app: &App,
    mcp_client: &mut McpClient,
    stream_result: &StreamResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    iteration: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let exec_result = execute_tool_calls_for_round(
        &app.session_id,
        mcp_client,
        &stream_result.tool_calls,
        iteration,
    )?;

    append_cached_tool_results_note(&exec_result, messages, turn_messages);
    append_tool_result_messages(
        app,
        &stream_result.assistant_text,
        &exec_result,
        messages,
        turn_messages,
    );

    super::persist_pending_turn_messages(
        app,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
    );

    Ok(())
}

pub(super) fn handle_iteration_execution(
    app: &App,
    mcp_client: &mut McpClient,
    execution: IterationExecution,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    final_assistant_text: &mut String,
    final_assistant_recorded: &mut bool,
    force_final_response: &mut bool,
    iteration: usize,
    max_iterations: usize,
) -> Result<TurnLoopStep, Box<dyn std::error::Error>> {
    match execution {
        IterationExecution::Exit(outcome) => Ok(TurnLoopStep::Return(outcome)),
        IterationExecution::RequestFailed(text) => {
            *final_assistant_text = text;
            Ok(TurnLoopStep::Break)
        }
        IterationExecution::FinalResponse(stream_result) => {
            crate::ai::agent_hang_debug!(
                "pre-fix",
                "A",
                "turn_runtime::run_turn:final-response",
                "[DEBUG] final assistant response without tool calls",
                {
                    "iteration": iteration,
                    "assistant_chars": stream_result.assistant_text.chars().count(),
                },
            );
            record_final_stream_response(
                app,
                stream_result,
                messages,
                turn_messages,
                final_assistant_text,
                final_assistant_recorded,
            );
            Ok(TurnLoopStep::Break)
        }
        IterationExecution::ToolCall(stream_result) => {
            handle_tool_call_round(
                app,
                mcp_client,
                &stream_result,
                messages,
                turn_messages,
                one_shot_mode,
                persisted_turn_messages,
                iteration,
            )?;

            input::clear_stdin_buffer();

            if iteration >= max_iterations {
                if *force_final_response {
                    *final_assistant_text = format!(
                        "Agent reached the tool iteration limit ({max_iterations}) without producing a final answer."
                    );
                    return Ok(TurnLoopStep::Break);
                }
                *force_final_response = true;
            }

            Ok(TurnLoopStep::Continue)
        }
    }
}
