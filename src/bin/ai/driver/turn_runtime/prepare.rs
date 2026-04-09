use colored::Colorize;
use serde_json::Value;

use crate::ai::mcp::McpClient;
use crate::ai::{
    driver::{reflection, skill_runtime},
    history::{Message, build_context_history},
    request,
    types::App,
};

use super::types::TurnPreparation;

#[crate::ai::agent_hang_span(
    "post-fix",
    "K",
    "turn_runtime::run_turn:prepare_turn",
    "[DEBUG] preparing turn",
    "[DEBUG] prepared turn",
    {
        "history_count": history_count,
        "question_len": question.chars().count(),
        "model": next_model,
    },
    {
        "message_count": __agent_hang_result.as_ref().map(|v| v.messages.len()).unwrap_or(0),
        "turn_message_count": __agent_hang_result
            .as_ref()
            .map(|v| v.turn_messages.len())
            .unwrap_or(0),
        "max_iterations": __agent_hang_result
            .as_ref()
            .map(|v| v.max_iterations)
            .unwrap_or(0),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(super) async fn prepare_turn(
    app: &mut App,
    mcp_client: &mut McpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    question: &str,
    next_model: &str,
) -> Result<TurnPreparation, Box<dyn std::error::Error>> {
    let history = build_context_history(
        history_count,
        &app.session_history_file,
        app.config.history_max_chars,
        app.config.history_keep_last,
        app.config.history_summary_max_chars,
    )?;
    let mut skill_turn =
        skill_runtime::prepare_skill_for_turn(app, mcp_client, skill_manifests, question)
            .await;

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

    let recall_intent = skill_turn.intent();
    let skip_recall_for_skill_context = skill_turn.skip_recall_by_skill();
    let skip_recall_for_light_turn = skill_turn.matched_skill_name().is_none()
        && question.chars().count() <= 64
        && matches!(
            recall_intent.core,
            crate::ai::driver::intent_recognition::CoreIntent::Casual
                | crate::ai::driver::intent_recognition::CoreIntent::QueryConcept
        );
    let skip_recall = skip_recall_for_skill_context || skip_recall_for_light_turn;
    if !skip_recall {
        let recall_bundle = reflection::build_recall_bundle(question, 1200, 2000);
        if let Some(guidelines) = recall_bundle.guidelines {
            if !guidelines.trim().is_empty() {
                skill_turn.append_system_prompt(&format!("\n{guidelines}"));
            }
        }
        if let Some(recalled) = recall_bundle.recalled
            && !recalled.content.trim().is_empty()
        {
            let project_part = recalled
                .project_hint
                .as_deref()
                .map(|project| format!(" project={project}"))
                .unwrap_or_default();
            let category_part = if recalled.categories.is_empty() {
                String::new()
            } else {
                format!(" categories={}", recalled.categories.join(","))
            };
            let confidence_part = if recalled.high_confidence_project_memory {
                " high_confidence=true"
            } else {
                " high_confidence=false"
            };
            println!(
                "{} count={}{}{}{}",
                "[Memory] recalled".bright_blue().bold(),
                recalled.entry_count,
                project_part,
                category_part,
                confidence_part
            );
            skill_turn.append_system_prompt(&format!("\n{}", recalled.content));
            if recalled.high_confidence_project_memory {
                skill_turn.append_system_prompt(
                    "\nMemory-first project answer policy:\n- High-confidence project memory has been recalled for this question.\n- Prefer answering directly from the recalled knowledge when it already covers the user's ask.\n- Do NOT read files, grep the repo, or call search tools unless a specific detail is missing, ambiguous, or the user explicitly asks you to verify against the current code.\n- If only part of the answer is covered, answer the covered part first and use tools only to fill the missing pieces.",
                );
            } else {
                skill_turn.append_system_prompt(
                    "\nKnowledge usage policy:\n- Recalled knowledge is available and relevant for this turn. Build your answer primarily from it.\n- Only call file-read/repo-search tools if the recalled knowledge is missing key details the user specifically asked about.\n- Do NOT re-scan the entire project when the recalled knowledge already covers the user's question.",
                );
            }
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
        content: request::build_content(next_model, question, &app.attached_image_files)?,
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

    Ok(TurnPreparation {
        skill_turn,
        messages,
        turn_messages,
        persisted_turn_messages: 0,
        max_iterations,
    })
}
