use crate::ai::{mcp::McpClient, types::App};

use super::{
    finalize::finalize_turn,
    iteration::{execute_turn_iteration, refresh_skill_turn_for_iteration},
    prepare::prepare_turn,
    tool_result::handle_iteration_execution,
    types::{TurnLoopStep, TurnOutcome, TurnPreparation},
};

#[crate::ai::agent_hang_span(
    "pre-fix",
    "A",
    "turn_runtime::run_turn",
    "[DEBUG] run_turn started",
    "[DEBUG] run_turn finished",
    {
        "history_count": history_count,
        "question_len": question.chars().count(),
        "model": next_model.as_str(),
        "one_shot_mode": one_shot_mode,
        "should_quit": should_quit,
    },
    {
        "ok": __agent_hang_result.is_ok(),
        "outcome": __agent_hang_result
            .as_ref()
            .map(|v| format!("{:?}", v))
            .unwrap_or_else(|err| err.to_string()),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(in crate::ai::driver) async fn run_turn(
    app: &mut App,
    mcp_client: &mut McpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    question: String,
    next_model: String,
    one_shot_mode: bool,
    should_quit: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    let TurnPreparation {
        mut skill_turn,
        mut messages,
        mut turn_messages,
        mut persisted_turn_messages,
        max_iterations,
    } = prepare_turn(
        app,
        mcp_client,
        skill_manifests,
        history_count,
        &question,
        &next_model,
    )
    .await?;

    let mut iteration = 0usize;
    let mut force_final_response = false;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    loop {
        iteration += 1;
        refresh_skill_turn_for_iteration(
            app,
            mcp_client,
            skill_manifests,
            &question,
            iteration,
            &mut skill_turn,
            &mut messages,
        )
        .await;
        let execution = execute_turn_iteration(
            app,
            &next_model,
            &mut messages,
            &turn_messages,
            one_shot_mode,
            &mut persisted_turn_messages,
            should_quit,
            force_final_response,
            iteration,
        )
        .await?;
        match handle_iteration_execution(
            app,
            &question,
            mcp_client,
            execution,
            &mut messages,
            &mut turn_messages,
            one_shot_mode,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            iteration,
            max_iterations,
        )? {
            TurnLoopStep::Continue => {
                let mut new_tools = crate::ai::tools::enable_tools::drain_pending_enable();
                let pending_mcp = crate::ai::tools::enable_tools::drain_pending_mcp_names();
                if !pending_mcp.is_empty() {
                    let mcp_all = mcp_client.get_all_tools();
                    for tool in mcp_all {
                        if pending_mcp.iter().any(|n| n == &tool.function.name) {
                            new_tools.push(tool);
                        }
                    }
                }
                if !new_tools.is_empty() {
                    if let Some(ctx) = app.agent_context.as_mut() {
                        for tool in new_tools {
                            if !ctx.tools.iter().any(|t| t.function.name == tool.function.name) {
                                ctx.tools.push(tool);
                            }
                        }
                    }
                }
            }
            TurnLoopStep::Break => break,
            TurnLoopStep::Return(outcome) => return Ok(outcome),
        }
    }
    finalize_turn(
        app,
        &next_model,
        &question,
        &final_assistant_text,
        final_assistant_recorded,
        &mut turn_messages,
        one_shot_mode,
        &mut persisted_turn_messages,
        should_quit,
    )
    .await
}
