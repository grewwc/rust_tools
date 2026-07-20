use crate::ai::{
    driver::{print::format_empty_state, reflection},
    history::{
        Message, compact_session_history_at_boundary_with_app, compact_session_history_with_app,
        value_to_string,
    },
    types::App,
};
use colored::Colorize;
use rust_tools::commonw::FastSet;
use serde_json::Value;
use std::sync::{LazyLock, Mutex};

use super::{TurnOutcome, persistence::persist_pending_turn_messages};

const SUBAGENT_TOOL_EVIDENCE_MAX_CALLS: usize = 8;
const SUBAGENT_TOOL_EVIDENCE_MAX_CHARS_PER_RESULT: usize = 700;
const SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS: usize = 4_000;
static SESSION_TITLE_IN_FLIGHT: LazyLock<Mutex<FastSet<String>>> =
    LazyLock::new(|| Mutex::new(FastSet::default()));

/// 标题任务必须从基础 history 文件推导 sessions 根目录；不能传入当前 session
/// 数据库的父目录，否则会错误地拼接出嵌套的 `.sessions` 路径。
fn session_title_store(history_file: &std::path::Path) -> crate::ai::history::SessionStore {
    crate::ai::history::SessionStore::new(history_file)
}

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

fn subagent_result_payload_for_parent(
    final_assistant_text: &str,
    turn_messages: &[Message],
) -> Option<String> {
    let output = format_subagent_result_for_parent(final_assistant_text, turn_messages);
    (!output.trim().is_empty()).then_some(output)
}

async fn maybe_append_post_turn_reflection(
    app: &mut App,
    next_model: &str,
    question: &str,
    final_assistant_text: &str,
    turn_messages: &mut Vec<Message>,
    had_tool_error: bool,
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
            had_tool_error,
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

pub(in crate::ai::driver::turn_runtime) fn should_generate_session_title_in_background(
    one_shot_mode: bool,
    should_quit: bool,
) -> bool {
    !one_shot_mode && !should_quit
}

fn mark_session_title_generation_started(session_id: &str) -> bool {
    let mut in_flight = SESSION_TITLE_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.insert(session_id.to_string())
}

fn mark_session_title_generation_finished(session_id: &str) {
    let mut in_flight = SESSION_TITLE_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.remove(session_id);
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
    had_tool_error: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    if let Some(subagent_output_for_parent) =
        subagent_result_payload_for_parent(final_assistant_text, turn_messages)
    {
        // 尽早发布给父 agent：即便本轮没有最终 assistant 正文，只要留下了可复用的
        // subagent 证据（如 read_file / code_search 结果），父 agent 也必须感知。
        // 这样同步 `task` 与异步 `task_wait` 都能拿到同一份父侧 payload。
        crate::ai::driver::runtime_ctx::publish_subagent_result(&subagent_output_for_parent).await;
    }

    if !final_assistant_text.trim().is_empty() {
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
            had_tool_error,
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
        // goal 模式下，run_loop 通过此标志判定目标是否完成：
        // 一轮结束时没有调用任何工具 = agent 已交付最终结果。
        app.last_turn_had_tool_calls = had_tool_calls;
        {
            // 用块限制 compact_result 的生命周期，避免它跨越 await 导致 Send 问题
            let compact_result = if had_tool_calls {
                compact_session_history_with_app(app).await
            } else {
                compact_session_history_at_boundary_with_app(app).await
            };
            if let Err(err) = compact_result {
                eprintln!("[Warning] Failed to compact persisted history: {}", err);
            }
        }
        // 尝试为当前对话生成 LLM 概括性标题（如果尚未生成且已有足够上下文）。
        // 交互式前台 turn 不应在这里等待一个后台质量任务。
        maybe_generate_session_title(
            app,
            should_generate_session_title_in_background(one_shot_mode, should_quit),
        )
        .await;
        // println!();
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
        app.last_turn_had_tool_calls = false;
    }

    Ok(if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    })
}

/// 在 turn 结束后尝试用 LLM 生成 session 概括性标题。
/// 条件：至少有 1 个 user turn，且没有已生成标题或现有标题质量过低。
pub(super) async fn maybe_generate_session_title(app: &App, run_in_background: bool) {
    if !mark_session_title_generation_started(&app.session_id) {
        return;
    }

    if run_in_background {
        let task_app = app.clone();
        let session_id = task_app.session_id.clone();
        tokio::spawn(async move {
            generate_session_title_if_missing(&task_app).await;
            mark_session_title_generation_finished(&session_id);
        });
        return;
    }

    generate_session_title_if_missing(app).await;
    mark_session_title_generation_finished(&app.session_id);
}

async fn generate_session_title_if_missing(app: &App) {
    use crate::ai::history::{is_low_quality_session_title, normalize_generated_session_title};
    // eprintln!("[session-title] checking: session_id={} history_file={}", app.session_id, app.session_history_file.display());

    // SessionStore 接收基础 history 文件，并据此推导 `<stem>.sessions/`。
    // 传入当前 session sqlite 的父目录会额外拼出一层 `.sessions`，导致标题任务
    // 读取不到当前会话的消息。
    let store = session_title_store(&app.config.history_file);

    if let Some(existing_title) = store.read_session_title(&app.session_id).ok().flatten() {
        if !is_low_quality_session_title(&existing_title) {
            return;
        }
    }

    let all_messages = match store.read_all_messages(&app.session_id) {
        Ok(m) if !m.is_empty() => {
            // eprintln!("[session-title] read {} messages", m.len());
            m
        }
        Ok(_) => {
            // eprintln!("[session-title] no messages found for session {}", app.session_id);
            return;
        }
        Err(_) => {
            // eprintln!("[session-title] failed to read messages: {e}");
            return;
        }
    };

    // 至少需要 1 个 user turn 即可生成标题（首条消息通常已包含核心意图）
    let user_turns = all_messages.iter().filter(|m| m.role == "user").count();
    // eprintln!("[session-title] user_turns={user_turns}");
    if user_turns < 1 {
        return;
    }

    // 调用 LLM 生成标题
    let title = crate::ai::request::generate_session_title_via_model(app, &all_messages).await;
    // eprintln!("[session-title] LLM returned: {:?}", title.as_deref().unwrap_or("None"));

    if let Some(t) = title {
        let t = normalize_generated_session_title(&t);
        if !t.is_empty() {
            if let Some(existing_title) = store.read_session_title(&app.session_id).ok().flatten()
                && !is_low_quality_session_title(&existing_title)
            {
                return;
            }
            if let Err(_) = store.write_session_title(&app.session_id, &t) {
                // eprintln!("[session-title] failed to save title: {}", err);
            } else {
                // eprintln!("[session-title] generated: {}", t);
            }
        }
    }
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
    fn subagent_result_payload_for_parent_uses_tool_evidence_without_final_text() {
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

        let output = subagent_result_payload_for_parent("", &turn_messages)
            .expect("tool evidence should still publish to parent");
        assert!(output.starts_with("[Subagent tool evidence]"));
        assert!(output.contains("read_file("));
        assert!(output.contains("pub mod ai;"));
        assert!(!output.contains("[Subagent final answer]"));
    }

    #[test]
    fn subagent_result_payload_for_parent_is_none_when_completely_empty() {
        assert!(subagent_result_payload_for_parent("", &[]).is_none());
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

    #[test]
    fn session_title_background_policy_only_backgrounds_live_interactive_turns() {
        assert!(should_generate_session_title_in_background(false, false));
        assert!(!should_generate_session_title_in_background(true, false));
        assert!(!should_generate_session_title_in_background(false, true));
        assert!(!should_generate_session_title_in_background(true, true));
    }

    #[test]
    fn session_title_store_reads_active_session_from_base_history_file() {
        let store = session_title_store(std::path::Path::new("/tmp/a.history.sqlite"));

        assert_eq!(
            store.session_history_file("current-session"),
            std::path::Path::new("/tmp/a.history.sessions/current-session.sqlite")
        );
    }
}
