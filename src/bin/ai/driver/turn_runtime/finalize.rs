use colored::Colorize;
use serde_json::Value;
use crate::ai::{
    driver::{print::format_empty_state, reflection},
    history::{Message, compact_session_history_with_app},
    types::App,
};

use super::{TurnOutcome, persistence::persist_pending_turn_messages};

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
    });
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

fn maybe_spawn_critic_revise_background(
    app: &App,
    question: &str,
    final_assistant_text: &str,
) {
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
    let (handle, cancel_token, interrupt_futex) = {
        let mut os = match kernel.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let parent_pid = os.current_process_id();
        let (handle, cancel_token) = os.daemon_register(
            "critic_revise_background".to_string(),
            DaemonKind::Reflection,
            parent_pid,
        );
        let interrupt_futex =
            crate::ai::driver::signal::alloc_interrupt_futex("critic_revise_interrupt");
        (handle, cancel_token, interrupt_futex)
    };

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
        persist_pending_turn_messages(
            app,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
        );
        if let Err(err) = compact_session_history_with_app(app).await {
            eprintln!("[Warning] Failed to compact persisted history: {}", err);
        }
        println!();
        maybe_spawn_critic_revise_background(app, question, final_assistant_text);

        let had_tool_calls = turn_messages.iter().any(|m| m.role == "tool" || m.tool_calls.as_ref().map_or(false, |c| !c.is_empty()));
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
                    eprintln!("[Warning] observer '{}' panicked in on_finalize; disabling for rest of conversation.", obs_name);
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
