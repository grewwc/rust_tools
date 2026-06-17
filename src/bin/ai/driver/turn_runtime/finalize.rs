use crate::ai::{
    driver::{print::format_empty_state, reflection},
    history::{
        Message, compact_session_history_at_boundary_with_app, compact_session_history_with_app,
        value_to_string,
    },
    types::App,
};
use colored::Colorize;
use serde_json::Value;

use super::{TurnOutcome, persistence::persist_pending_turn_messages};

const SUBAGENT_TOOL_EVIDENCE_MAX_CALLS: usize = 8;
const SUBAGENT_TOOL_EVIDENCE_MAX_CHARS_PER_RESULT: usize = 700;
const SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS: usize = 4_000;

fn ensure_final_assistant_recorded(
    final_assistant_text: &str,
    final_assistant_recorded: bool,
    turn_messages: &mut Vec<Message>,
) {
    if final_assistant_recorded {
        return;
    }

    println!("\n{}", final_assistant_text.yellow());
    turn_messages.push(Message {
        role: "assistant".to_string(),
        content: Value::String(final_assistant_text.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

fn normalized_tool_args(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| raw.trim().to_string())
}

fn collect_subagent_tool_evidence(turn_messages: &[Message]) -> Vec<String> {
    use rustc_hash::FxHashMap;

    let mut id_to_call: FxHashMap<String, (String, String)> = FxHashMap::default();
    for message in turn_messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            id_to_call.insert(
                tool_call.id.clone(),
                (
                    tool_call.function.name.clone(),
                    normalized_tool_args(&tool_call.function.arguments),
                ),
            );
        }
    }

    let mut evidence = Vec::new();
    for message in turn_messages {
        if message.role != "tool" {
            continue;
        }
        let Some(tool_call_id) = message.tool_call_id.as_ref() else {
            continue;
        };
        let Some((tool_name, args)) = id_to_call.get(tool_call_id) else {
            continue;
        };
        let content = value_to_string(&message.content);
        let content = content.trim();
        if content.is_empty() {
            continue;
        }
        evidence.push(format!(
            "- {}({}) => {}",
            tool_name,
            args,
            truncate_chars(content, SUBAGENT_TOOL_EVIDENCE_MAX_CHARS_PER_RESULT)
        ));
        if evidence.len() >= SUBAGENT_TOOL_EVIDENCE_MAX_CALLS {
            break;
        }
    }
    evidence
}

fn format_subagent_result_for_parent(
    final_assistant_text: &str,
    turn_messages: &[Message],
) -> String {
    let final_assistant_text = final_assistant_text.trim();
    let evidence = collect_subagent_tool_evidence(turn_messages);
    if evidence.is_empty() {
        return final_assistant_text.to_string();
    }

    let mut evidence_block = String::from("[Subagent tool evidence]\n");
    evidence_block.push_str(
        "The subagent used these tool results while producing the answer; treat them as already observed context.\n",
    );
    evidence_block.push_str(&evidence.join("\n"));

    let mut output = truncate_chars(&evidence_block, SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS);
    if final_assistant_text.is_empty() {
        return output;
    }
    output.push_str("\n\n");
    output.push_str("[Subagent final answer]\n");
    output.push_str(final_assistant_text);
    output
}

async fn maybe_append_post_turn_reflection(
    app: &mut App,
    next_model: &str,
    question: &str,
    final_assistant_text: &str,
    turn_messages: &mut Vec<Message>,
) {
    let integrated_reflect = crate::commonw::configw::get_all_config()
        .get_opt("ai.reflection.integrated")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .ne("false");
    if !integrated_reflect {
        reflection::maybe_append_self_reflection(
            app,
            next_model,
            question,
            final_assistant_text,
            turn_messages,
        )
        .await;
    }
}

async fn write_post_turn_project_knowledge(
    app: &mut App,
    next_model: &str,
    question: &str,
    final_assistant_text: &str,
    turn_messages: &mut Vec<Message>,
) {
    reflection::maybe_write_back_project_knowledge(
        app,
        next_model,
        question,
        final_assistant_text,
        turn_messages,
    )
    .await;
}

fn maybe_spawn_critic_revise_background(app: &App, question: &str, final_assistant_text: &str) {
    let integrated = crate::commonw::configw::get_all_config()
        .get_opt("ai.critic_revise.integrated")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .ne("false");
    if integrated {
        return;
    }

    let path = app.session_history_file.clone();
    let model_bg = crate::commonw::configw::get_all_config()
        .get_opt("ai.critic_revise.model")
        .unwrap_or_else(|| "qwen3.5-flash".to_string());
    let q_bg = question.to_string();
    let a_bg = final_assistant_text.to_string();

    // 登记 daemon：critic/revise 是典型的后台反思类。
    use aios_kernel::primitives::DaemonKind;
    let kernel = app.os.clone();
    let (handle, cancel_token) = {
        let mut os = match kernel.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let parent_pid = os.current_process_id();
        os.daemon_register(
            "critic_revise_background".to_string(),
            DaemonKind::Reflection,
            parent_pid,
        )
    };
    // 必须在释放 kernel 锁之后再调用：alloc_interrupt_futex 内部会再次锁同一把
    // Arc<Mutex<Kernel>>（GLOBAL_OS 与 app.os 共享），在持锁时调用会自死锁。
    let interrupt_futex =
        crate::ai::driver::signal::alloc_interrupt_futex("critic_revise_interrupt");

    tokio::spawn(async move {
        tokio::select! {
            _ = crate::ai::driver::signal::wait_for_interrupt_sources(
                Some(cancel_token.clone()),
                interrupt_futex,
            ) => {}
            _ = super::super::reflection::run_critic_revise_background(path, model_bg, q_bg, a_bg) => {}
        }
        let mut os = match kernel.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        os.daemon_exit(handle, None);
        // 必须先释放 kernel 锁再 destroy：destroy_interrupt_futex 内部会再次锁同一把
        // Arc<Mutex<Kernel>>（GLOBAL_OS 与 app.os 共享），在持锁时调用会自死锁。
        drop(os);
        if let Some(addr) = interrupt_futex {
            crate::ai::driver::signal::destroy_interrupt_futex(addr);
        }
    });
}

pub(super) async fn finalize_turn(
    app: &mut App,
    next_model: &str,
    question: &str,
    final_assistant_text: &str,
    final_assistant_recorded: bool,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    should_quit: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    if !final_assistant_text.trim().is_empty() {
        // Publish to the optional sub-agent result slot before doing the
        // (potentially long) post-turn work below, so a parent that is
        // waiting on us can race ahead even if reflection / writeback take
        // a while.
        let subagent_output_for_parent =
            format_subagent_result_for_parent(final_assistant_text, turn_messages);
        crate::ai::driver::runtime_ctx::publish_subagent_result(&subagent_output_for_parent).await;
        ensure_final_assistant_recorded(
            final_assistant_text,
            final_assistant_recorded,
            turn_messages,
        );
        maybe_append_post_turn_reflection(
            app,
            next_model,
            question,
            final_assistant_text,
            turn_messages,
        )
        .await;
        write_post_turn_project_knowledge(
            app,
            next_model,
            question,
            final_assistant_text,
            turn_messages,
        )
        .await;
        persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);
        // 任务边界判定：当前 turn 没有再调工具，意味着 agent 已经把答案交付，
        // 这是一个自然的"任务完成"切点；用更激进的阈值（160 turns）触发摘要，
        // 避免对话一直堆到硬上限（200 turns）才被动压缩。
        let had_tool_calls = turn_messages
            .iter()
            .any(|m| m.role == "tool" || m.tool_calls.as_ref().map_or(false, |c| !c.is_empty()));
        let compact_result = if had_tool_calls {
            compact_session_history_with_app(app).await
        } else {
            compact_session_history_at_boundary_with_app(app).await
        };
        if let Err(err) = compact_result {
            eprintln!("[Warning] Failed to compact persisted history: {}", err);
        }
        println!();
        maybe_spawn_critic_revise_background(app, question, final_assistant_text);

        let mut first_observer_emitted = false;
        let mut poisoned: Vec<String> = Vec::new();
        for obs in app.observers.iter_mut() {
            if obs.is_poisoned() {
                continue;
            }
            let ctx = crate::ai::driver::observer::FinalizeContext {
                question: question.to_string(),
                final_text: final_assistant_text.to_string(),
                had_tool_calls,
            };
            let obs_name = obs.name().to_string();
            let output = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                obs.on_finalize(&ctx)
            })) {
                Ok(o) => o,
                Err(_) => {
                    eprintln!(
                        "[Warning] observer '{}' panicked in on_finalize; disabling for rest of conversation.",
                        obs_name
                    );
                    obs.mark_poisoned();
                    poisoned.push(obs_name);
                    continue;
                }
            };
            if output.display_lines.is_empty() {
                continue;
            }
            if first_observer_emitted {
                println!("---");
            }
            first_observer_emitted = true;
            for line in &output.display_lines {
                println!("{}", line);
            }
        }
        let _ = poisoned;
    } else {
        println!("{}", format_empty_state("no response"));
    }

    Ok(if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};

    fn tool_call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[test]
    fn subagent_parent_result_includes_tool_evidence() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "call-1",
                    "read_file_lines",
                    serde_json::json!({"file_path":"src/lib.rs","offset":10,"limit":20}),
                )]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("    10\tfn load_config() {".to_string()),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                reasoning_content: None,
            },
        ];

        let output = format_subagent_result_for_parent("done", &turn_messages);
        assert!(output.starts_with("[Subagent tool evidence]"));
        assert!(output.contains("[Subagent tool evidence]"));
        assert!(output.contains("read_file_lines("));
        assert!(output.contains("\"file_path\":\"src/lib.rs\""));
        assert!(output.contains("fn load_config()"));
        assert!(output.contains("[Subagent final answer]\ndone"));
    }

    #[test]
    fn subagent_parent_result_without_tools_is_plain_final_text() {
        let output = format_subagent_result_for_parent("done", &[]);
        assert_eq!(output, "done");
    }

    #[test]
    fn subagent_parent_result_keeps_evidence_after_long_final_text() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "call-1",
                    "read_file",
                    serde_json::json!({"file_path":"src/lib.rs"}),
                )]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("pub mod ai;".to_string()),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                reasoning_content: None,
            },
        ];

        let long_final = "x".repeat(SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS + 100);
        let output = format_subagent_result_for_parent(long_final.as_str(), &turn_messages);
        assert!(output.starts_with("[Subagent tool evidence]"));
        assert!(output.contains("[Subagent tool evidence]"));
        assert!(output.contains("read_file("));
        assert!(output.contains("pub mod ai;"));
        assert!(output.contains("[Subagent final answer]\n"));
        assert!(output.ends_with(&long_final));
    }
}
