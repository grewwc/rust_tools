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
    tokio::spawn(async move {
        super::super::reflection::run_critic_revise_background(path, model_bg, q_bg, a_bg).await;
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
        for obs in app.observers.iter_mut() {
            let output = obs.on_finalize(&crate::ai::driver::observer::FinalizeContext {
                question: question.to_string(),
                final_text: final_assistant_text.to_string(),
                had_tool_calls,
            });
            for line in &output.display_lines {
                println!("{}", line);
            }
            for entry in &output.memory_entries {
                let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
                let _ = store.append(&entry.to_agent_memory_entry());
            }
        }
    } else {
        println!("{}", format_empty_state("no response"));
    }

    Ok(if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    })
}
