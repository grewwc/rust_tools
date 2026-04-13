use colored::Colorize;
use serde_json::Value;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crate::ai::{
    driver::{
        drain_response, input,
        print::print_assistant_banner_with_app,
        skill_runtime,
    },
    history::Message,
    mcp::McpClient,
    request::{self, do_request_messages},
    stream,
    types::{App, StreamOutcome, StreamResult},
};

use super::{
    TurnOutcome,
    persistence::persist_pending_turn_messages,
    types::IterationExecution,
};

pub(super) async fn refresh_skill_turn_for_iteration(
    app: &mut App,
    mcp_client: &mut McpClient,
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
    let mut new_skill_turn = skill_runtime::rebuild_skill_turn_with_existing_selection(
        app,
        mcp_client,
        skill_manifests,
        question,
        prev_skill.as_deref(),
        &intent,
    );
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
        system_message.content = Value::String(skill_turn.system_prompt().to_string());
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
    app.ignore_next_prompt_interrupt = true;
    persist_pending_turn_messages(
        app,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
    );
    println!("\nInterrupted.");
    continue_or_quit(should_quit)
}

fn finish_shutdown_turn(
    app: &App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> TurnOutcome {
    persist_pending_turn_messages(
        app,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
    );
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
    persist_pending_turn_messages(
        app,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
    );
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

async fn wait_for_request_interrupt(shutdown: &AtomicBool, cancel_stream: &AtomicBool) {
    loop {
        if request_interrupt_pending(shutdown, cancel_stream) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
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
) -> Result<reqwest::Response, request::RequestError> {
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
        if let Some(ctx) = app.agent_context.as_mut() {
            Some(std::mem::replace(&mut ctx.tools, Vec::new()))
        } else {
            None
        }
    } else {
        None
    };

    let request_result = do_request_messages(app, next_model, messages, true).await;

    if let Some(saved_tools) = saved_tools
        && let Some(ctx) = app.agent_context.as_mut()
    {
        ctx.tools = saved_tools;
    }

    request_result
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
    _iteration: usize,
) -> StreamResult {
    print_assistant_banner_with_app(Some(app));
    let stream_result = match stream::stream_response(
        app,
        response,
        current_history,
        terminal_dedupe_candidate,
    )
    .await
    {
        Ok(result) => result,
        Err(err) => {
            app.streaming
                .store(false, std::sync::atomic::Ordering::Relaxed);
            eprintln!("\n[Error] 流式响应处理失败：{}", err);
            eprintln!("[Info] 尝试继续对话...");
            let _ = drain_response(response).await;
            StreamResult {
                outcome: StreamOutcome::Completed,
                tool_calls: Vec::new(),
                assistant_text: "[响应解析失败，请重试]".to_string(),
                hidden_meta: String::new(),
            }
        }
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

    drain_response(response).await?;
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
    iteration: usize,
) -> Result<IterationExecution, Box<dyn std::error::Error>> {
    let mut current_history = String::new();
    app.streaming
        .store(true, std::sync::atomic::Ordering::Relaxed);

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
        _ = wait_for_request_interrupt(shutdown.as_ref(), cancel_stream.as_ref()) => {
            return Ok(interrupted_iteration_execution(
                app,
                one_shot_mode,
                turn_messages,
                persisted_turn_messages,
                should_quit,
            ));
        }
    };

    let mut response = match request_result {
        Ok(response) => response,
        Err(err) => {
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

    request::print_info(next_model);
    let stream_result =
        stream_model_response(
            app,
            &mut response,
            &mut current_history,
            terminal_dedupe_candidate,
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
    use super::request_interrupt_pending;
    use std::sync::atomic::AtomicBool;

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
