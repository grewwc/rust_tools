use colored::Colorize;
use serde_json::Value;

use crate::ai::{
    history::{Message, append_history_messages, build_context_history}, mcp::McpClient, request, stream, types::{App, StreamOutcome, StreamResult}
};

use super::{
    drain_response, input,
    print::{print_assistant_banner, print_tool_output_block},
    skill_runtime::SkillTurnGuard,
    reflection,
    tools,
};

pub(super) enum TurnOutcome {
    Continue,
    Quit,
}

pub(super) async fn run_turn(
    app: &mut App,
    mcp_client: &mut McpClient,
    skill_turn: SkillTurnGuard,
    history_count: usize,
    question: String,
    next_model: String,
    one_shot_mode: bool,
    should_quit: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    if let Some(name) = skill_turn.matched_skill_name() {
        println!("[skill: {}]", name.cyan());
    }

    let history = build_context_history(
        history_count,
        &app.session_history_file,
        app.config.history_max_chars,
        app.config.history_keep_last,
        app.config.history_summary_max_chars,
    )?;
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(skill_turn.system_prompt().to_string()),
        tool_calls: None,
        tool_call_id: None,
    });
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
                messages.push(Message {
                    role: "system".to_string(),
                    content: Value::String(sys),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }
    }
    
    
    if let Some(guidelines) = super::reflection::build_persistent_guidelines(&question, 1200) {
        if !guidelines.trim().is_empty() {
            messages.push(Message {
                role: "system".to_string(),
                content: Value::String(guidelines),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }
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
        let mut current_history = String::new();
        app.streaming.store(true, std::sync::atomic::Ordering::Relaxed);
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

        let request_result = request::do_request_messages(app, &next_model, &messages, true).await;

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
                if request::is_transient_error(&err) {
                    eprintln!("[Warning] {}", err);
                    break;
                }
                return Err(err.into());
            }
        };
        if app
            .cancel_stream
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            app.streaming
                .store(false, std::sync::atomic::Ordering::Relaxed);
            println!("\nInterrupted.");
            return Ok(if should_quit {
                TurnOutcome::Quit
            } else {
                TurnOutcome::Continue
            });
        }
        request::print_info(&next_model);
        print_assistant_banner();
        let stream_result = match stream::stream_response(app, &mut response, &mut current_history).await
        {
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

        input::clear_stdin_buffer();

        if stream_result.outcome == StreamOutcome::Cancelled {
            println!("\nInterrupted.");
            return Ok(if should_quit {
                TurnOutcome::Quit
            } else {
                TurnOutcome::Continue
            });
        }
        if app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            println!();
            return Ok(TurnOutcome::Quit);
        }
        drain_response(&mut response).await?;
        app.streaming
            .store(false, std::sync::atomic::Ordering::Relaxed);

        if stream_result.outcome != StreamOutcome::ToolCall {
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
                    content: Value::String(format!("self_note:\n{}", stream_result.hidden_meta.trim())),
                    tool_calls: None,
                    tool_call_id: None,
                };
                turn_messages.push(record);
                let entry = crate::ai::tools::storage::memory_store::AgentMemoryEntry {
                    timestamp: chrono::Local::now().to_rfc3339(),
                    category: "self_note".to_string(),
                    note: stream_result.hidden_meta.trim().to_string(),
                    tags: vec!["agent".to_string(), "policy".to_string()],
                    source: Some(format!("session:{}", app.session_id)),
                    priority: Some(255), // Permanent: agent policies are never deleted
                };
                let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
                let _ = store.append(&entry);
                store.maintain_after_append();
            }
            break;
        }

        let exec_result = tools::execute_tool_calls(mcp_client, &stream_result.tool_calls)?;

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
            println!(
                "\n{} {}",
                "[Tool]".bright_green().bold(),
                tool_call.function.name.bright_cyan().bold()
            );
            if tool_call.function.name == "web_search" {
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
                print_tool_output_block(&result.content);
            }
            let tool_message = Message {
                role: "tool".to_string(),
                content: Value::String(result.content.clone()),
                tool_calls: None,
                tool_call_id: Some(result.tool_call_id.clone()),
            };
            messages.push(tool_message.clone());
            turn_messages.push(tool_message);
        }

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
        if !one_shot_mode
            && let Err(e) = append_history_messages(&app.session_history_file, &turn_messages)
        {
            eprintln!("[Warning] Failed to save history: {}", e);
        }
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
                super::reflection::run_critic_revise_background(path, model_bg, q_bg, a_bg).await;
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

