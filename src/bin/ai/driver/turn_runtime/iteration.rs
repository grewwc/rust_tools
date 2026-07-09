use colored::Colorize;
use rust_tools::commonw::FastSet;
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
    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS, MID_TURN_LLM_SUMMARY_MAX_CHARS,
    PRE_REQUEST_LLM_SUMMARY_MIN_GROWTH, TurnOutcome, context_budget,
    persistence::persist_pending_turn_messages,
    pre_request_llm_summary_threshold,
    types::{IterationExecution, ToolCallExecution},
};

/// 记录上次成功 pre-request LLM 摘要后的 messages 总字符数。
/// 用于增长量守卫：成功摘要后上下文需增长 [`PRE_REQUEST_LLM_SUMMARY_MIN_GROWTH`]
/// 以上才会再次触发。失败/no-op 不能写入该游标，否则会把后续真正需要的
/// LLM compact 静默挡掉。
/// 进程级 static——同一 agent 进程内全局生效，与 orchestrator 的 supervisor
/// 冷却机制互补（orchestrator 管理工具调用间的压缩，此处管理请求前的兜底）。
static LAST_PRE_REQUEST_LLM_SUMMARY_CHARS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn should_try_pre_request_llm_summary(after_chars: usize, llm_threshold: usize) -> bool {
    if after_chars <= llm_threshold {
        return false;
    }
    let last_summary_chars =
        LAST_PRE_REQUEST_LLM_SUMMARY_CHARS.load(std::sync::atomic::Ordering::Relaxed);
    let growth = after_chars.saturating_sub(last_summary_chars);
    last_summary_chars == 0 || growth >= PRE_REQUEST_LLM_SUMMARY_MIN_GROWTH
}

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

fn request_visible_tool_names(app: &App, force_final_response: bool) -> FastSet<String> {
    if force_final_response {
        return FastSet::default();
    }
    app.agent_context
        .as_ref()
        .map(|ctx| {
            ctx.tools
                .iter()
                .map(|tool| tool.function.name.clone())
                .collect()
        })
        .unwrap_or_default()
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
            )
        })
        .unwrap_or_else(|| {
            skill_runtime::rebuild_skill_turn_with_existing_selection(
                app,
                mcp_client,
                skill_manifests,
                question,
                prev_skill.as_deref(),
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

    let budget_report = context_budget::apply_pre_request_context_budget(app, next_model, messages);
    if let Some(reason) = budget_report.rollback_reason {
        crate::ai::driver::print::print_tool_note_line("context-budget", reason.note());
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

    // === Pre-request LLM 摘要兜底 ===
    // 无损+弱损压缩后仍远超阈值时，调用 LLM 把早期对话压成摘要。
    // 这是发送请求前的最后一道防线，避免超大上下文导致模型 4xx 或质量退化
    // （用户报告的 "295K 压到 294K 就停了" 问题）。
    // 阈值取 history_max_chars * 2（默认 240K），比 orchestrator 的 hard
    // threshold（*3.5 = 420K）更积极——后者只在工具调用间隙触发，此处覆盖
    // 每次请求前的最后检查。
    // 增长量守卫：自上次成功 LLM 摘要后需增长 ≥ MIN_GROWTH 才再次触发。
    // 失败/no-op 不写游标，避免把后续真正需要的 LLM compact 静默挡掉。
    let llm_threshold = pre_request_llm_summary_threshold(next_model, app.config.history_max_chars);
    if should_try_pre_request_llm_summary(budget_report.after_chars, llm_threshold) {
        crate::ai::driver::print::print_tool_note_line(
            "compress",
            &format!(
                "pre-request LLM summary: {} > {} chars, requesting summary…",
                budget_report.after_chars, llm_threshold
            ),
        );
        let drained: Vec<Message> = std::mem::take(messages);
        let (after_msgs, llm_before, llm_after, did_summarize) =
            crate::ai::history::mid_turn_llm_summarize(
                app,
                drained,
                MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
                MID_TURN_LLM_SUMMARY_MAX_CHARS,
                app.config.history_max_chars,
            )
            .await;
        *messages = after_msgs;
        if did_summarize {
            LAST_PRE_REQUEST_LLM_SUMMARY_CHARS
                .store(llm_after, std::sync::atomic::Ordering::Relaxed);
            crate::ai::driver::print::print_tool_note_line(
                "compress",
                &format!("pre-request (llm): {} → {} chars", llm_before, llm_after),
            );
        } else {
            crate::ai::driver::print::print_tool_note_line(
                "compress",
                "pre-request LLM summary skipped \
                 (no early dialog to summarize or call failed); \
                 agent may hit context limit",
            );
        }
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
        "ok": __agent_hang_result.is_ok(),
        "outcome": format!("{:?}", __agent_hang_result.as_ref().ok().map(|r| r.outcome)),
        "assistant_chars": __agent_hang_result.as_ref().map(|r| r.assistant_text.chars().count()).unwrap_or(0),
        "tool_calls": __agent_hang_result.as_ref().map(|r| r.tool_calls.len()).unwrap_or(0),
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
) -> Result<StreamResult, String> {
    print_assistant_banner_with_app_and_skill(Some(app), active_skill_name);
    match stream::stream_response(app, response, current_history, terminal_dedupe_candidate).await {
        Ok(result) => Ok(result),
        Err(err) => {
            app.streaming
                .store(false, std::sync::atomic::Ordering::Relaxed);
            Err(err.to_string())
        }
    }
}

async fn finalize_stream_interaction(
    app: &mut App,
    response: &mut reqwest::Response,
    stream_result: StreamResult,
    turn_messages: &[Message],
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    should_quit: bool,
    force_final_response: bool,
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
        StreamOutcome::ToolCall => IterationExecution::ToolCall(ToolCallExecution {
            stream_result,
            allowed_tool_names: request_visible_tool_names(app, force_final_response),
        }),
        StreamOutcome::EmptyResponse => IterationExecution::EmptyResponse,
        StreamOutcome::Truncated => IterationExecution::Truncated(stream_result),
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

    // 流式响应中途可能因后端瞬态错误（如 "Cancelled by backend"）中断，
    // 对这类可重试错误重试整条请求+流，避免直接放弃整轮对话。
    const MAX_STREAM_RETRIES: usize = 16;
    let mut stream_attempt = 0usize;
    loop {
        request::print_info(app, &actual_model);
        match stream_model_response(
            app,
            &mut response,
            &mut current_history,
            terminal_dedupe_candidate,
            active_skill_name,
            iteration,
        )
        .await
        {
            Ok(stream_result) => {
                return finalize_stream_interaction(
                    app,
                    &mut response,
                    stream_result,
                    turn_messages,
                    one_shot_mode,
                    persisted_turn_messages,
                    should_quit,
                    force_final_response,
                )
                .await;
            }
            Err(err_msg) => {
                if stream_attempt < MAX_STREAM_RETRIES
                    && request::is_retryable_stream_error(&err_msg)
                {
                    stream_attempt += 1;
                    eprintln!(
                        "\n[Info] 流式响应中断（{}），第 {}/{} 次重试...",
                        err_msg, stream_attempt, MAX_STREAM_RETRIES
                    );
                    current_history.clear();
                    app.streaming
                        .store(true, std::sync::atomic::Ordering::Relaxed);

                    if request::should_abort_retry_wait(app) {
                        return Ok(interrupted_iteration_execution(
                            app,
                            one_shot_mode,
                            turn_messages,
                            persisted_turn_messages,
                            should_quit,
                        ));
                    }
                    if request::sleep_with_cancel(
                        app,
                        request::retry_delay(stream_attempt),
                    )
                    .await
                    {
                        return Ok(interrupted_iteration_execution(
                            app,
                            one_shot_mode,
                            turn_messages,
                            persisted_turn_messages,
                            should_quit,
                        ));
                    }

                    request::clear_stale_request_interrupt_before_request(app);
                    match do_request_messages(app, &actual_model, messages, true).await {
                        Ok(new_response) => {
                            response = new_response;
                        }
                        Err(retry_err) => {
                            let err_text = retry_err.to_string();
                            if crate::ai::driver::runtime_ctx::has_subagent_result_slot() {
                                return Err(err_text.into());
                            }
                            return Ok(IterationExecution::RequestFailed(
                                handle_request_error(
                                    app,
                                    retry_err,
                                    one_shot_mode,
                                    turn_messages,
                                    persisted_turn_messages,
                                ),
                            ));
                        }
                    }
                    continue;
                }

                // 不可重试或已用完重试次数——回退到旧行为，继续对话
                eprintln!("\n[Error] 流式响应处理失败：{}", err_msg);
                eprintln!("[Info] 尝试继续对话...");
                let stream_result = StreamResult {
                    outcome: StreamOutcome::Completed,
                    tool_calls: Vec::new(),
                    assistant_text: "[响应解析失败，请重试]".to_string(),
                    hidden_meta: String::new(),
                    reasoning_text: String::new(),
                    skip_response_drain: false,
                    truncated_by_length: false,
                    stream_error: false,
                    finish_reason_value: None,
                    usage_prompt_tokens: 0,
                    usage_completion_tokens: 0,
                    usage_reasoning_tokens: 0,
                };
                return finalize_stream_interaction(
                    app,
                    &mut response,
                    stream_result,
                    turn_messages,
                    one_shot_mode,
                    persisted_turn_messages,
                    should_quit,
                    force_final_response,
                )
                .await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LAST_PRE_REQUEST_LLM_SUMMARY_CHARS, StreamingFlagGuard, request_interrupt_pending,
        should_try_pre_request_llm_summary,
    };
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

    #[test]
    fn failed_pre_request_llm_summary_does_not_poison_retry_cursor() {
        LAST_PRE_REQUEST_LLM_SUMMARY_CHARS.store(0, Ordering::Relaxed);
        let threshold = 240_000;
        let after_chars = 240_457;

        assert!(should_try_pre_request_llm_summary(after_chars, threshold));

        // 失败/no-op 尝试不应写入 LAST_PRE_REQUEST_LLM_SUMMARY_CHARS；否则同样
        // 超阈值的下一次请求会被 growth < 20K 静默挡掉。
        assert_eq!(
            LAST_PRE_REQUEST_LLM_SUMMARY_CHARS.load(Ordering::Relaxed),
            0
        );
        assert!(should_try_pre_request_llm_summary(after_chars, threshold));

        LAST_PRE_REQUEST_LLM_SUMMARY_CHARS.store(230_000, Ordering::Relaxed);
        assert!(!should_try_pre_request_llm_summary(after_chars, threshold));
        assert!(should_try_pre_request_llm_summary(251_000, threshold));

        LAST_PRE_REQUEST_LLM_SUMMARY_CHARS.store(0, Ordering::Relaxed);
    }
}
