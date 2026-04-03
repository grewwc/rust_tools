use std::{fs, path::PathBuf};

use colored::Colorize;
use serde_json::Value;

use crate::ai::{
    history::{Message, SessionStore, append_history_messages, build_context_history},
    mcp::McpClient,
    request, stream,
    types::{App, StreamOutcome, StreamResult},
};

use super::{
    drain_response, input,
    print::{print_assistant_banner, print_tool_output_block},
    reflection, tools,
};

const MAX_TOOL_RESULT_INLINE_CHARS: usize = 32_000;
const TOOL_OVERFLOW_PREVIEW_CHARS: usize = 800;

struct PreparedToolResult {
    content_for_model: String,
    content_for_terminal: String,
}

struct LargeToolSummary {
    body: String,
    summary: String,
    top_level_keys: Vec<String>,
    field_samples: Vec<String>,
}

// #region debug-point agent-hang:reporter
fn report_agent_hang_debug(
    run_id: &'static str,
    hypothesis_id: &'static str,
    location: &'static str,
    msg: &'static str,
    data: Value,
) {
    std::thread::spawn(move || {
        let mut debug_server_url = "http://127.0.0.1:7777/event".to_string();
        let mut debug_session_id = "agent-hang".to_string();
        if let Ok(env_text) = fs::read_to_string(".dbg/agent-hang.env") {
            for line in env_text.lines() {
                if let Some(value) = line.strip_prefix("DEBUG_SERVER_URL=") {
                    if !value.trim().is_empty() {
                        debug_server_url = value.trim().to_string();
                    }
                } else if let Some(value) = line.strip_prefix("DEBUG_SESSION_ID=") {
                    if !value.trim().is_empty() {
                        debug_session_id = value.trim().to_string();
                    }
                }
            }
        }
        let payload = serde_json::json!({
            "sessionId": debug_session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
            "ts": chrono::Utc::now().timestamp_millis(),
        });
        if let Ok(client) = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_millis(300))
            .build()
        {
            let _ = client.post(debug_server_url).json(&payload).send();
        }
    });
}
// #endregion

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
    content
        .chars()
        .skip(total.saturating_sub(max_chars))
        .collect::<String>()
}

fn write_tool_overflow_file(
    app: &App,
    tool_name: &str,
    body: &str,
    extension: &str,
) -> Result<PathBuf, String> {
    let store = SessionStore::new(app.config.history_file.as_path());
    let dir = store
        .session_assets_dir(&app.session_id)
        .join("tool_overflow");
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create tool overflow dir: {}", e))?;

    let mut safe_tool = String::new();
    for ch in tool_name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            safe_tool.push(ch);
        } else {
            safe_tool.push('_');
        }
    }
    let filename = format!(
        "{}_{}.{}",
        safe_tool,
        uuid::Uuid::new_v4(),
        extension.trim_start_matches('.')
    );
    let path = dir.join(filename);
    fs::write(&path, body).map_err(|e| format!("failed to write tool overflow file: {}", e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
    }
    Ok(path)
}

fn prepare_tool_result(app: &App, tool_name: &str, content: &str) -> PreparedToolResult {
    if content.chars().count() <= MAX_TOOL_RESULT_INLINE_CHARS {
        return PreparedToolResult {
            content_for_model: content.to_string(),
            content_for_terminal: content.to_string(),
        };
    }

    let large = summarize_large_tool_output(content);
    let body = large.body;
    let extension = if body.trim_start().starts_with('{') || body.trim_start().starts_with('[') {
        "json"
    } else {
        "txt"
    };

    match write_tool_overflow_file(app, tool_name, &body, extension) {
        Ok(path) => {
            let lines = body.lines().count();
            let chars = body.chars().count();
            let head = truncate_chars(&body, TOOL_OVERFLOW_PREVIEW_CHARS);
            let tail = truncate_chars(
                &tail_chars(&body, TOOL_OVERFLOW_PREVIEW_CHARS),
                TOOL_OVERFLOW_PREVIEW_CHARS,
            );
            let top_level_keys = if large.top_level_keys.is_empty() {
                String::new()
            } else {
                format!("\n- top_level_keys: {}", large.top_level_keys.join(", "))
            };
            let field_samples = if large.field_samples.is_empty() {
                String::new()
            } else {
                format!(
                    "\n- field_samples:\n  - {}",
                    large.field_samples.join("\n  - ")
                )
            };
            let stub = format!(
                "Output too large; full result saved to a session file.\n- file_path: {}\n- chars: {}\n- lines: {}\n- summary: {}{}{}\n- preview_head:\n{}\n- preview_tail:\n{}\n- read_next: use read_file_lines with this file_path and a limit between 200 and 400.",
                path.display(),
                chars,
                lines,
                if large.summary.trim().is_empty() {
                    "(empty)".to_string()
                } else {
                    large.summary
                },
                top_level_keys,
                field_samples,
                head.trim_end(),
                tail.trim_end(),
            );
            PreparedToolResult {
                content_for_model: stub.clone(),
                content_for_terminal: stub,
            }
        }
        Err(err) => {
            let fallback = format!(
                "{}\n... (output truncated; overflow spill failed: {})",
                truncate_chars(content, MAX_TOOL_RESULT_INLINE_CHARS),
                err
            );
            PreparedToolResult {
                content_for_model: fallback.clone(),
                content_for_terminal: fallback,
            }
        }
    }
}

pub(super) enum TurnOutcome {
    Continue,
    Quit,
}

pub(super) async fn run_turn(
    app: &mut App,
    mcp_client: &mut McpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    question: String,
    next_model: String,
    one_shot_mode: bool,
    should_quit: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    // #region debug-point A:turn-start
    report_agent_hang_debug(
        "pre-fix",
        "A",
        "turn_runtime::run_turn:start",
        "[DEBUG] run_turn started",
        serde_json::json!({
            "history_count": history_count,
            "question_len": question.chars().count(),
            "model": next_model,
            "one_shot_mode": one_shot_mode,
        }),
    );
    // #endregion

    // #region debug-point E:prepare-skill-begin
    report_agent_hang_debug(
        "pre-fix",
        "E",
        "turn_runtime::run_turn:prepare_skill_for_turn:begin",
        "[DEBUG] preparing skill for turn",
        serde_json::json!({}),
    );
    // #endregion
    let mut skill_turn =
        super::skill_runtime::prepare_skill_for_turn(app, mcp_client, skill_manifests, &question)
            .await;
    // #region debug-point E:prepare-skill-end
    report_agent_hang_debug(
        "pre-fix",
        "E",
        "turn_runtime::run_turn:prepare_skill_for_turn:end",
        "[DEBUG] prepared skill for turn",
        serde_json::json!({
            "matched_skill": skill_turn.matched_skill_name(),
        }),
    );
    // #endregion
    if let Some(name) = skill_turn.matched_skill_name() {
        println!("[skill: {}]", name.cyan());
    }

    // #region debug-point E:build-history-begin
    report_agent_hang_debug(
        "pre-fix",
        "E",
        "turn_runtime::run_turn:build_context_history:begin",
        "[DEBUG] building context history",
        serde_json::json!({}),
    );
    // #endregion
    let history = build_context_history(
        history_count,
        &app.session_history_file,
        app.config.history_max_chars,
        app.config.history_keep_last,
        app.config.history_summary_max_chars,
    )?;
    // #region debug-point E:build-history-end
    report_agent_hang_debug(
        "pre-fix",
        "E",
        "turn_runtime::run_turn:build_context_history:end",
        "[DEBUG] built context history",
        serde_json::json!({
            "history_messages": history.len(),
        }),
    );
    // #endregion
    let mut messages = Vec::with_capacity(history.len() + 2);

    {
        let integrated = crate::commonw::configw::get_all_config()
            .get_opt("ai.critic_revise.integrated")
            .unwrap_or_else(|| "true".to_string())
            .trim()
            .ne("false");
        let reflect_integrated = crate::commonw::configw::get_all_config()
            .get_opt("ai.reflection.integrated")
            .unwrap_or_else(|| "true".to_string())
            .trim()
            .ne("false");
        if integrated || reflect_integrated {
            let mut sys = String::new();
            if integrated {
                sys.push_str("Before replying, internally perform a brief CRITIC→REVISE pass to ensure correctness, missing steps, and clear structure. Do not output the critic. Output only the final improved answer.\n");
            }
            if reflect_integrated {
                sys.push_str("At the very end of your message, include a compact self experience note enclosed within <meta:self_note> and </meta:self_note>. The note should be 2-6 short bullets grouped under 'Do:' and 'Avoid:'. Do not mention these tags in the visible content.\n");
            }
            if !sys.is_empty() {
                skill_turn.append_system_prompt(&sys);
            }
        }
    }
    if let Some(guidelines) = super::reflection::build_persistent_guidelines(&question, 1200) {
        if !guidelines.trim().is_empty() {
            skill_turn.append_system_prompt(&format!("\n{guidelines}"));
        }
    }

    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(skill_turn.system_prompt().to_string()),
        tool_calls: None,
        tool_call_id: None,
    });

    messages.extend(history);
    let user_message = Message {
        role: "user".to_string(),
        content: request::build_content(&next_model, &question, &app.attached_image_files)?,
        tool_calls: None,
        tool_call_id: None,
    };
    messages.push(user_message.clone());
    let mut turn_messages = Vec::with_capacity(8);
    turn_messages.push(user_message);
    let mut persisted_turn_messages = 0usize;

    let max_iterations = app
        .agent_context
        .as_ref()
        .map(|c| c.max_iterations)
        .unwrap_or(0)
        .max(1);

    let mut iteration = 0usize;
    let mut force_final_response = false;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    loop {
        iteration += 1;
        // #region debug-point A:iteration-begin
        report_agent_hang_debug(
            "pre-fix",
            "A",
            "turn_runtime::run_turn:iteration:begin",
            "[DEBUG] turn iteration started",
            serde_json::json!({
                "iteration": iteration,
                "force_final_response": force_final_response,
                "message_count": messages.len(),
            }),
        );
        // #endregion
        if iteration > 1 {
            let prev_skill = skill_turn.matched_skill_name().map(|s| s.to_string());
            let new_skill_turn = super::skill_runtime::prepare_skill_for_turn(
                app,
                mcp_client,
                skill_manifests,
                &question,
            )
            .await;
            let next_skill = new_skill_turn.matched_skill_name().map(|s| s.to_string());

            if prev_skill != next_skill {
                match next_skill.as_deref() {
                    Some(name) => println!("[skill switched: {}]", name.cyan()),
                    None => println!("[skill switched: <none>]"),
                }
            }
            skill_turn = new_skill_turn;
            messages[0].content = Value::String(skill_turn.system_prompt().to_string());
        }

        let mut current_history = String::new();
        app.streaming
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if force_final_response {
            messages.push(Message {
                role: "system".to_string(),
                content: Value::String(
                    "Tool limit reached. Do not call any more tools. Provide the best possible final answer using the information already collected.".to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        let saved_tools = if force_final_response {
            app.agent_context
                .as_mut()
                .map(|ctx| std::mem::take(&mut ctx.tools))
        } else {
            None
        };

        // #region debug-point B:request-begin
        report_agent_hang_debug(
            "pre-fix",
            "B",
            "turn_runtime::run_turn:do_request_messages:begin",
            "[DEBUG] sending model request",
            serde_json::json!({
                "iteration": iteration,
                "message_count": messages.len(),
                "model": next_model,
            }),
        );
        // #endregion
        let request_result = request::do_request_messages(app, &next_model, &messages, true).await;
        // #region debug-point B:request-end
        report_agent_hang_debug(
            "pre-fix",
            "B",
            "turn_runtime::run_turn:do_request_messages:end",
            "[DEBUG] model request finished",
            serde_json::json!({
                "iteration": iteration,
                "ok": request_result.is_ok(),
            }),
        );
        // #endregion

        if let Some(saved_tools) = saved_tools
            && let Some(ctx) = app.agent_context.as_mut()
        {
            ctx.tools = saved_tools;
        }

        let mut response = match request_result {
            Ok(response) => response,
            Err(err) => {
                app.streaming
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                persist_pending_turn_messages(
                    app,
                    one_shot_mode,
                    &turn_messages,
                    &mut persisted_turn_messages,
                );
                let err_text = err.to_string();
                if request::is_transient_error(&err) {
                    eprintln!("[Warning] {}", err_text);
                } else {
                    eprintln!("[Error] {}", err_text);
                }
                if err_text.contains("function.arguments")
                    && err_text.contains("must be in JSON format")
                {
                    eprintln!(
                        "[Info] 检测到模型返回了非法 tool arguments，本轮已跳过，继续下一轮对话。"
                    );
                } else {
                    eprintln!("[Info] 本轮请求失败，已保持会话存活，可直接继续提问。");
                }
                final_assistant_text = "[本轮请求失败，请重试或换个问法]".to_string();
                break;
            }
        };
        if app
            .cancel_stream
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            app.streaming
                .store(false, std::sync::atomic::Ordering::Relaxed);
            persist_pending_turn_messages(
                app,
                one_shot_mode,
                &turn_messages,
                &mut persisted_turn_messages,
            );
            println!("\nInterrupted.");
            return Ok(if should_quit {
                TurnOutcome::Quit
            } else {
                TurnOutcome::Continue
            });
        }
        request::print_info(&next_model);
        print_assistant_banner();
        // #region debug-point B:stream-begin
        report_agent_hang_debug(
            "pre-fix",
            "B",
            "turn_runtime::run_turn:stream_response:begin",
            "[DEBUG] streaming response started",
            serde_json::json!({
                "iteration": iteration,
            }),
        );
        // #endregion
        let stream_result =
            match stream::stream_response(app, &mut response, &mut current_history).await {
                Ok(result) => result,
                Err(err) => {
                    app.streaming
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("\n[Error] 流式响应处理失败：{}", err);
                    eprintln!("[Info] 尝试继续对话...");
                    let _ = drain_response(&mut response).await;
                    StreamResult {
                        outcome: StreamOutcome::Completed,
                        tool_calls: Vec::new(),
                        assistant_text: "[响应解析失败，请重试]".to_string(),
                        hidden_meta: String::new(),
                    }
                }
            };
        // #region debug-point B:stream-end
        report_agent_hang_debug(
            "pre-fix",
            "B",
            "turn_runtime::run_turn:stream_response:end",
            "[DEBUG] streaming response finished",
            serde_json::json!({
                "iteration": iteration,
                "outcome": format!("{:?}", stream_result.outcome),
                "assistant_chars": stream_result.assistant_text.chars().count(),
                "tool_calls": stream_result.tool_calls.len(),
                "history_chars": current_history.chars().count(),
            }),
        );
        // #endregion

        input::clear_stdin_buffer();

        if stream_result.outcome == StreamOutcome::Cancelled {
            persist_pending_turn_messages(
                app,
                one_shot_mode,
                &turn_messages,
                &mut persisted_turn_messages,
            );
            println!("\nInterrupted.");
            return Ok(if should_quit {
                TurnOutcome::Quit
            } else {
                TurnOutcome::Continue
            });
        }
        if app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            persist_pending_turn_messages(
                app,
                one_shot_mode,
                &turn_messages,
                &mut persisted_turn_messages,
            );
            println!();
            return Ok(TurnOutcome::Quit);
        }
        drain_response(&mut response).await?;
        app.streaming
            .store(false, std::sync::atomic::Ordering::Relaxed);

        if stream_result.outcome != StreamOutcome::ToolCall {
            // #region debug-point A:final-response
            report_agent_hang_debug(
                "pre-fix",
                "A",
                "turn_runtime::run_turn:final-response",
                "[DEBUG] final assistant response without tool calls",
                serde_json::json!({
                    "iteration": iteration,
                    "assistant_chars": stream_result.assistant_text.chars().count(),
                }),
            );
            // #endregion
            let assistant_msg = Message {
                role: "assistant".to_string(),
                content: Value::String(stream_result.assistant_text.clone()),
                tool_calls: None,
                tool_call_id: None,
            };
            messages.push(assistant_msg.clone());
            turn_messages.push(assistant_msg);
            final_assistant_text = stream_result.assistant_text;
            final_assistant_recorded = true;
            if !stream_result.hidden_meta.trim().is_empty() {
                let record = Message {
                    role: "system".to_string(),
                    content: Value::String(format!(
                        "self_note:\n{}",
                        stream_result.hidden_meta.trim()
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                };
                turn_messages.push(record);
                let entry = crate::ai::tools::storage::memory_store::AgentMemoryEntry {
                    id: None,
                    timestamp: chrono::Local::now().to_rfc3339(),
                    category: "self_note".to_string(),
                    note: stream_result.hidden_meta.trim().to_string(),
                    tags: vec!["agent".to_string(), "policy".to_string()],
                    source: Some(format!("session:{}", app.session_id)),
                    priority: Some(255), // Permanent: agent policies are never deleted
                };
                let store =
                    crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
                let _ = store.append(&entry);
                store.maintain_after_append();
            }
            break;
        }

        // #region debug-point C:tool-exec-begin
        report_agent_hang_debug(
            "pre-fix",
            "C",
            "turn_runtime::run_turn:execute_tool_calls:begin",
            "[DEBUG] executing tool calls",
            serde_json::json!({
                "iteration": iteration,
                "tool_calls": stream_result
                    .tool_calls
                    .iter()
                    .map(|tool| tool.function.name.clone())
                    .collect::<Vec<_>>(),
            }),
        );
        // #endregion
        let exec_result =
            tools::execute_tool_calls(&app.session_id, mcp_client, &stream_result.tool_calls)?;
        // #region debug-point C:tool-exec-end
        report_agent_hang_debug(
            "pre-fix",
            "C",
            "turn_runtime::run_turn:execute_tool_calls:end",
            "[DEBUG] executed tool calls",
            serde_json::json!({
                "iteration": iteration,
                "tool_result_count": exec_result.tool_results.len(),
                "cached_hits": exec_result.cached_hits,
            }),
        );
        // #endregion

        if exec_result.cached_hits.iter().any(|hit| *hit) {
            let cached_names = exec_result
                .executed_tool_calls
                .iter()
                .zip(exec_result.cached_hits.iter())
                .filter_map(|(tool_call, cached)| {
                    cached.then_some(tool_call.function.name.as_str())
                })
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
            messages.push(cache_note.clone());
            turn_messages.push(cache_note);
        }

        let assistant_msg = Message {
            role: "assistant".to_string(),
            content: Value::String(stream_result.assistant_text.clone()),
            tool_calls: Some(exec_result.executed_tool_calls.clone()),
            tool_call_id: None,
        };
        messages.push(assistant_msg.clone());
        turn_messages.push(assistant_msg);

        for (tool_call, result) in exec_result
            .executed_tool_calls
            .iter()
            .zip(exec_result.tool_results.iter())
        {
            let prepared = prepare_tool_result(app, &tool_call.function.name, &result.content);
            println!(
                "\n{} {}",
                "[Tool]".bright_green().bold(),
                tool_call.function.name.bright_cyan().bold()
            );
            if tool_call.function.name == "web_search"
                && prepared.content_for_model == result.content
            {
                let mut preview = String::new();
                let mut truncated = false;
                for (lines, line) in result.content.lines().enumerate() {
                    if lines >= 12 || preview.len() >= 1200 {
                        truncated = true;
                        break;
                    }
                    preview.push_str(line);
                    preview.push('\n');
                }
                print_tool_output_block(&preview);
                if truncated {
                    print_tool_output_block("... (truncated)");
                }
            } else {
                print_tool_output_block(&prepared.content_for_terminal);
            }
            let tool_message = Message {
                role: "tool".to_string(),
                content: Value::String(prepared.content_for_model),
                tool_calls: None,
                tool_call_id: Some(result.tool_call_id.clone()),
            };
            messages.push(tool_message.clone());
            turn_messages.push(tool_message);
        }

        persist_pending_turn_messages(
            app,
            one_shot_mode,
            &turn_messages,
            &mut persisted_turn_messages,
        );

        input::clear_stdin_buffer();

        if iteration >= max_iterations {
            if force_final_response {
                final_assistant_text = format!(
                    "Agent reached the tool iteration limit ({max_iterations}) without producing a final answer."
                );
                break;
            }
            force_final_response = true;
        }
    }

    if !final_assistant_text.trim().is_empty() {
        if !final_assistant_recorded {
            println!("\n{}", final_assistant_text.yellow());
            turn_messages.push(Message {
                role: "assistant".to_string(),
                content: Value::String(final_assistant_text.clone()),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        {
            let integrated_reflect = crate::commonw::configw::get_all_config()
                .get_opt("ai.reflection.integrated")
                .unwrap_or_else(|| "true".to_string())
                .trim()
                .ne("false");
            if !integrated_reflect {
                reflection::maybe_append_self_reflection(
                    app,
                    &next_model,
                    &question,
                    &final_assistant_text,
                    &mut turn_messages,
                )
                .await;
            }
        }
        persist_pending_turn_messages(
            app,
            one_shot_mode,
            &turn_messages,
            &mut persisted_turn_messages,
        );
        println!();
        // Background Critic→Revise (fire-and-forget)
        {
            let integrated = crate::commonw::configw::get_all_config()
                .get_opt("ai.critic_revise.integrated")
                .unwrap_or_else(|| "true".to_string())
                .trim()
                .ne("false");
            if integrated {
                // skip background critic→revise when integrated into main turn
            } else {
                let path = app.session_history_file.clone();
                let model_bg = crate::commonw::configw::get_all_config()
                    .get_opt("ai.critic_revise.model")
                    .unwrap_or_else(|| "qwen3.5-flash".to_string());
                let q_bg = question.clone();
                let a_bg = final_assistant_text.clone();
                tokio::spawn(async move {
                    super::reflection::run_critic_revise_background(path, model_bg, q_bg, a_bg)
                        .await;
                });
            }
        }
    } else {
        println!("{}", "(no response)".dimmed());
    }

    Ok(if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    })
}

fn persist_pending_turn_messages(
    app: &App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) {
    if one_shot_mode || *persisted_turn_messages >= turn_messages.len() {
        return;
    }

    if let Err(err) = append_history_messages(
        &app.session_history_file,
        &turn_messages[*persisted_turn_messages..],
    ) {
        eprintln!("[Warning] Failed to save history: {}", err);
        return;
    }

    *persisted_turn_messages = turn_messages.len();
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, atomic::AtomicBool};

    use serde_json::Value;

    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        history::{SessionStore, build_message_arr},
        types::AppConfig,
    };

    fn test_app(history_file: PathBuf) -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                history_file: history_file.clone(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 12_000,
                history_keep_last: 256,
                history_summary_max_chars: 4_000,
                intent_model: None,
            },
            session_id: "test".to_string(),
            session_history_file: history_file,
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            pending_files: None,
            pending_clipboard: false,
            pending_short_output: false,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            writer: None,
            prompt_editor: None,
            agent_context: None,
        }
    }

    fn extract_stub_path(stub: &str) -> Option<PathBuf> {
        stub.lines()
            .find_map(|line| line.strip_prefix("- file_path: "))
            .map(PathBuf::from)
    }

    #[test]
    fn persist_pending_turn_messages_only_appends_new_entries() {
        let path =
            std::env::temp_dir().join(format!("ai-turn-history-{}.sqlite", uuid::Uuid::new_v4()));
        let app = test_app(path.clone());

        let mut turn_messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }];
        let mut persisted = 0usize;

        persist_pending_turn_messages(&app, false, &turn_messages, &mut persisted);
        assert_eq!(persisted, 1);

        turn_messages.push(Message {
            role: "tool".to_string(),
            content: Value::String("tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
        });
        turn_messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String("done".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        persist_pending_turn_messages(&app, false, &turn_messages, &mut persisted);
        assert_eq!(persisted, 3);

        let loaded = build_message_arr(16, &path).unwrap();
        assert_eq!(loaded, turn_messages);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn prepare_tool_result_spills_large_output_to_session_file() {
        let history_file =
            std::env::temp_dir().join(format!("ai-tool-overflow-{}.sqlite", uuid::Uuid::new_v4()));
        let app = test_app(history_file.clone());
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        std::fs::write(store.session_history_file(&app.session_id), b"test").unwrap();

        let content = "x".repeat(MAX_TOOL_RESULT_INLINE_CHARS + 256);
        let prepared = prepare_tool_result(&app, "mcp_big_payload", &content);

        assert!(
            prepared
                .content_for_model
                .contains("Output too large; full result saved")
        );
        let path = extract_stub_path(&prepared.content_for_model).unwrap();
        assert!(path.is_absolute());
        assert!(Path::new(&path).exists());
        let saved = std::fs::read_to_string(&path).unwrap();
        assert_eq!(saved, content);

        let _ = store.delete_session(&app.session_id);
        assert!(!path.exists());
    }

    #[test]
    fn prepare_tool_result_json_stub_includes_keys_and_samples() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-overflow-json-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file.clone());
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        std::fs::write(store.session_history_file(&app.session_id), b"test").unwrap();

        let payload = serde_json::json!({
            "id": 123,
            "name": "example payload",
            "items": [
                { "kind": "doc", "token": "abc", "size": 42 }
            ],
            "meta": {
                "source": "mcp",
                "ok": true
            }
        });
        let content = format!("{}{}", payload, " ".repeat(MAX_TOOL_RESULT_INLINE_CHARS));
        let prepared = prepare_tool_result(&app, "mcp_json_payload", &content);

        assert!(prepared.content_for_model.contains("- top_level_keys:"));
        assert!(prepared.content_for_model.contains("id"));
        assert!(prepared.content_for_model.contains("name"));
        assert!(prepared.content_for_model.contains("- field_samples:"));
        assert!(prepared.content_for_model.contains("items:"));
        assert!(prepared.content_for_model.contains("meta:"));

        let _ = store.delete_session(&app.session_id);
    }
}
