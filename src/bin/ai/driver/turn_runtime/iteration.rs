use colored::Colorize;
use serde_json::Value;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;

use crate::ai::{
    driver::{
        drain_response, input, print::print_assistant_banner_with_app_and_skill, skill_runtime,
    },
    history::Message,
    mcp::McpClient,
    request::{self, do_request_messages},
    stream,
    types::{App, StreamOutcome, StreamResult},
};

use super::{
    TurnOutcome, context_budget, persistence::persist_pending_turn_messages,
    types::IterationExecution,
};

struct StreamingFlagGuard {
    flag: Arc<AtomicBool>,
}

impl StreamingFlagGuard {
    fn new(flag: &Arc<AtomicBool>) -> Self {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        Self {
            flag: Arc::clone(flag),
        }
    }
}

impl Drop for StreamingFlagGuard {
    fn drop(&mut self) {
        self.flag.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(super) fn refresh_skill_turn_for_iteration(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    question: &str,
    iteration: usize,
    skill_turn: &mut super::super::skill_runtime::SkillTurnGuard,
    messages: &mut [Message],
) {
    if iteration <= 1 {
        return;
    }

    let prev_skill = skill_turn.matched_skill_name().map(|s| s.to_string());
    let intent = skill_turn.intent().clone();
    let inherited_restore = skill_turn.take_restore_agent_context();

    // 模型通过 activate_skill 工具显式请求激活某个 skill 时优先采纳：直接按名字
    // 强制激活，跳过自动路由打分。名字校验在工具侧已做（必须真实存在），这里
    // 用 skill_manifests 再兜一次，未命中则回退到自动路由。
    let requested = crate::ai::tools::skill_tools::take_pending_skill_activation();
    let mut new_skill_turn = requested
        .as_deref()
        .and_then(|name| {
            skill_runtime::force_activate_named_skill(
                app,
                mcp_client,
                skill_manifests,
                question,
                name,
                &intent,
            )
        })
        .unwrap_or_else(|| {
            skill_runtime::rebuild_skill_turn_with_existing_selection(
                app,
                mcp_client,
                skill_manifests,
                question,
                prev_skill.as_deref(),
                &intent,
            )
        });
    if inherited_restore.is_some() {
        new_skill_turn.set_restore_agent_context(inherited_restore);
    }
    let next_skill = new_skill_turn.matched_skill_name().map(|s| s.to_string());

    if prev_skill != next_skill {
        match next_skill.as_deref() {
            Some(name) => println!("[skill switched: {}]", name.cyan()),
            None => println!("[skill switched: <none>]"),
        }
    }

    *skill_turn = new_skill_turn;
    if let Some(system_message) = messages.first_mut() {
        // 仅当新旧 system prompt 文本不同才覆写。
        // 同一段字符串的覆写不仅没用，还会让上游 prompt cache（例如 anthropic
        // 的 cache_control 命中、或者 driver 内部的字符串 hash 复用）连续失效，
        // 在长 turn 多 iteration 场景里是无声的 token 浪费。
        let next_prompt = skill_turn.system_prompt();
        let same = matches!(&system_message.content, Value::String(s) if s == next_prompt);
        if !same {
            system_message.content = Value::String(next_prompt.to_string());
        }
    }
}

fn continue_or_quit(should_quit: bool) -> TurnOutcome {
    if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    }
}

fn interrupted_iteration_execution(
    app: &mut App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
    should_quit: bool,
) -> IterationExecution {
    IterationExecution::Exit(finish_interrupted_turn(
        app,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
        should_quit,
    ))
}

fn shutdown_iteration_execution(
    app: &App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> IterationExecution {
    IterationExecution::Exit(finish_shutdown_turn(
        app,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
    ))
}

fn finish_interrupted_turn(
    app: &mut App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
    should_quit: bool,
) -> TurnOutcome {
    app.streaming
        .store(false, std::sync::atomic::Ordering::Relaxed);
    // 仅消费“本轮由 cancel_stream 触发”的中断，避免误清其它来源
    // （例如 shutdown/request-level interrupt）的全局中断位。
    let _ = crate::ai::types::take_stream_cancelled(app);
    app.ignore_next_prompt_interrupt = true;
    persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);
    println!("\nInterrupted.");
    continue_or_quit(should_quit)
}

fn finish_shutdown_turn(
    app: &App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> TurnOutcome {
    persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);
    println!();
    TurnOutcome::Quit
}

fn handle_request_error(
    app: &App,
    err: request::RequestError,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> String {
    app.streaming
        .store(false, std::sync::atomic::Ordering::Relaxed);
    persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);
    let err_text = err.to_string();
    if request::is_transient_error(&err) {
        eprintln!("[Warning] {}", err_text);
    } else {
        eprintln!("[Error] {}", err_text);
    }
    if err_text.contains("function.arguments") && err_text.contains("must be in JSON format") {
        eprintln!("[Info] 检测到模型返回了非法 tool arguments，本轮已跳过，继续下一轮对话。");
    } else {
        eprintln!("[Info] 本轮请求失败，已保持会话存活，可直接继续提问。");
    }
    "[本轮请求失败，请重试或换个问法]".to_string()
}

fn request_interrupt_pending(shutdown: &AtomicBool, cancel_stream: &AtomicBool) -> bool {
    shutdown.load(std::sync::atomic::Ordering::Relaxed)
        || cancel_stream.load(std::sync::atomic::Ordering::Relaxed)
}

fn request_interrupt_futex_ready() -> bool {
    crate::ai::driver::signal::request_interrupt_ready()
}

async fn wait_for_request_interrupt(shutdown: Arc<AtomicBool>, cancel_stream: Arc<AtomicBool>) {
    let notify = crate::ai::driver::signal::request_interrupt_notify();
    loop {
        if request_interrupt_pending(shutdown.as_ref(), cancel_stream.as_ref())
            || request_interrupt_futex_ready()
        {
            return;
        }
        // 注册等待 future 后再次检查，避免 signal_request_interrupt 与注册之间的 race。
        let notified = notify.notified();
        if request_interrupt_pending(shutdown.as_ref(), cancel_stream.as_ref())
            || request_interrupt_futex_ready()
        {
            return;
        }
        // 50ms 兜底兼容外部 futex 唤醒（不经 Notify 通道）。
        tokio::select! {
            _ = notified => {}
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
    }
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "B",
    "turn_runtime::run_turn:do_request_messages",
    "[DEBUG] sending model request",
    "[DEBUG] model request finished",
    {
        "iteration": _iteration,
        "message_count": messages.len(),
        "model": next_model,
    },
    {
        "iteration": _iteration,
        "ok": __agent_hang_result.is_ok(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
async fn request_model_response(
    app: &mut App,
    next_model: &str,
    messages: &mut Vec<Message>,
    force_final_response: bool,
    _iteration: usize,
) -> Result<(reqwest::Response, String), request::RequestError> {
    if force_final_response {
        messages.push(Message {
            role: "system".to_string(),
            content: Value::String(
                "Tool limit reached. Do not call any more tools. Provide the best possible final answer using the information already collected.".to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    let saved_tools = if force_final_response {
        if let Some(ctx) = app.agent_context.as_mut() {
            Some(std::mem::replace(&mut ctx.tools, Vec::new()))
        } else {
            None
        }
    } else {
        None
    };

    let budget_report = context_budget::apply_pre_request_context_budget(app, messages);
    if budget_report.rolled_back {
        crate::ai::driver::print::print_tool_note_line(
            "context-budget",
            "compression rolled back because protected system/current-user context changed",
        );
    } else if budget_report.changed {
        crate::ai::driver::print::print_tool_note_line(
            "context-budget",
            &format!(
                "compressed {} -> {} chars (target={}, lossless_removed={} messages/{} chars, critical={}, offload_only={}, lossy_candidates={} segments/{} chars)",
                budget_report.before_chars,
                budget_report.after_chars,
                budget_report.target_chars,
                budget_report.lossless_removed_messages,
                budget_report.lossless_saved_chars,
                budget_report.critical_segments,
                budget_report.offload_only_segments,
                budget_report.lossy_candidate_segments,
                budget_report.lossy_candidate_chars
            ),
        );
    }

    let mut actual_model = next_model.to_string();
    let mut request_result = do_request_messages(app, next_model, messages, true).await;
    if let Err(err) = &request_result
        && let Some(fallback_spec) = crate::ai::driver::runtime_ctx::auto_model_fallback_spec()
        && request::should_try_model_fallback(err)
    {
        if request::should_temporarily_disable_auto_selected_model(err) {
            crate::ai::models::mark_model_temporarily_unavailable(next_model, &err.to_string());
        }
        if let Some(fallback_model) =
            crate::ai::models::fallback_subagent_model_after_failure(next_model, fallback_spec)
        {
            eprintln!(
                "[model] auto-selected model '{}' failed; retrying subagent with '{}'",
                next_model, fallback_model
            );
            actual_model = fallback_model.clone();
            request_result = do_request_messages(app, &fallback_model, messages, true).await;
            if let Err(fallback_err) = &request_result
                && request::should_temporarily_disable_auto_selected_model(fallback_err)
            {
                crate::ai::models::mark_model_temporarily_unavailable(
                    &fallback_model,
                    &fallback_err.to_string(),
                );
            }
        }
    }

    if let Some(saved_tools) = saved_tools
        && let Some(ctx) = app.agent_context.as_mut()
    {
        ctx.tools = saved_tools;
    }

    request_result.map(|response| (response, actual_model))
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "B",
    "turn_runtime::run_turn:stream_response",
    "[DEBUG] streaming response started",
    "[DEBUG] streaming response finished",
    {
        "iteration": _iteration,
    },
    {
        "iteration": _iteration,
        "outcome": format!("{:?}", __agent_hang_result.outcome),
        "assistant_chars": __agent_hang_result.assistant_text.chars().count(),
        "tool_calls": __agent_hang_result.tool_calls.len(),
        "history_chars": current_history.chars().count(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
async fn stream_model_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
    terminal_dedupe_candidate: Option<&str>,
    active_skill_name: Option<&str>,
    _iteration: usize,
) -> StreamResult {
    print_assistant_banner_with_app_and_skill(Some(app), active_skill_name);
    let stream_ok =
        match stream::stream_response(app, response, current_history, terminal_dedupe_candidate)
            .await
        {
            Ok(result) => Some(result),
            Err(err) => {
                app.streaming
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                eprintln!("\n[Error] 流式响应处理失败：{}", err);
                eprintln!("[Info] 尝试继续对话...");
                None
            }
        };
    let stream_result = match stream_ok {
        Some(result) => result,
        None => StreamResult {
            outcome: StreamOutcome::Completed,
            tool_calls: Vec::new(),
            assistant_text: "[响应解析失败，请重试]".to_string(),
            hidden_meta: String::new(),
            reasoning_text: String::new(),
            skip_response_drain: false,
        },
    };
    stream_result
}

async fn finalize_stream_interaction(
    app: &mut App,
    response: &mut reqwest::Response,
    stream_result: StreamResult,
    turn_messages: &[Message],
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    should_quit: bool,
) -> Result<IterationExecution, Box<dyn std::error::Error>> {
    input::clear_stdin_buffer();

    if stream_result.outcome == StreamOutcome::Cancelled {
        return Ok(interrupted_iteration_execution(
            app,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
            should_quit,
        ));
    }
    if app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(shutdown_iteration_execution(
            app,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
        ));
    }

    if !stream_result.skip_response_drain {
        // Parse-error fallback may still leave bytes buffered. Keep this bounded
        // so unusual provider behavior cannot hang the turn.
        match tokio::time::timeout(Duration::from_millis(200), drain_response(response)).await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(err),
            Err(_) => {
                eprintln!("[Warning] 响应流收尾 drain 超时，已跳过剩余字节读取以避免会话卡住。");
            }
        }
    }
    app.streaming
        .store(false, std::sync::atomic::Ordering::Relaxed);

    Ok(match stream_result.outcome {
        StreamOutcome::ToolCall => IterationExecution::ToolCall(stream_result),
        _ => IterationExecution::FinalResponse(stream_result),
    })
}

pub(super) async fn execute_turn_iteration(
    app: &mut App,
    next_model: &str,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    should_quit: bool,
    force_final_response: bool,
    terminal_dedupe_candidate: Option<&str>,
    active_skill_name: Option<&str>,
    iteration: usize,
) -> Result<IterationExecution, Box<dyn std::error::Error>> {
    let mut current_history = String::new();
    request::clear_stale_request_interrupt_before_request(app);
    let _streaming_guard = StreamingFlagGuard::new(&app.streaming);
    crate::ai::driver::runtime_ctx::publish_subagent_phase("calling model");

    let shutdown = app.shutdown.clone();
    let cancel_stream = app.cancel_stream.clone();
    let request_result = tokio::select! {
        response = request_model_response(
            app,
            next_model,
            messages,
            force_final_response,
            iteration,
        ) => response,
        _ = wait_for_request_interrupt(shutdown.clone(), cancel_stream.clone()) => {
            return Ok(interrupted_iteration_execution(
                app,
                one_shot_mode,
                turn_messages,
                persisted_turn_messages,
                should_quit,
            ));
        }
    };

    let (mut response, actual_model) = match request_result {
        Ok(response) => response,
        Err(err) => {
            let err_text = err.to_string();
            if crate::ai::driver::runtime_ctx::has_subagent_result_slot() {
                return Err(err_text.into());
            }
            return Ok(IterationExecution::RequestFailed(handle_request_error(
                app,
                err,
                one_shot_mode,
                turn_messages,
                persisted_turn_messages,
            )));
        }
    };

    if app
        .cancel_stream
        .swap(false, std::sync::atomic::Ordering::Relaxed)
    {
        return Ok(interrupted_iteration_execution(
            app,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
            should_quit,
        ));
    }

    request::print_info(app, &actual_model);
    let stream_result = stream_model_response(
        app,
        &mut response,
        &mut current_history,
        terminal_dedupe_candidate,
        active_skill_name,
        iteration,
    )
    .await;
    finalize_stream_interaction(
        app,
        &mut response,
        stream_result,
        turn_messages,
        one_shot_mode,
        persisted_turn_messages,
        should_quit,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::{StreamingFlagGuard, request_interrupt_pending};
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, atomic::Ordering};

    #[test]
    fn streaming_flag_guard_resets_on_drop() {
        let streaming = Arc::new(AtomicBool::new(false));
        {
            let _guard = StreamingFlagGuard::new(&streaming);
            assert!(streaming.load(Ordering::Relaxed));
        }
        assert!(!streaming.load(Ordering::Relaxed));
    }

    #[test]
    fn request_interrupt_pending_tracks_shutdown_or_stream_cancel() {
        let shutdown = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(false);
        assert!(!request_interrupt_pending(&shutdown, &cancel_stream));

        cancel_stream.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(request_interrupt_pending(&shutdown, &cancel_stream));

        cancel_stream.store(false, std::sync::atomic::Ordering::Relaxed);
        shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(request_interrupt_pending(&shutdown, &cancel_stream));
    }
}
