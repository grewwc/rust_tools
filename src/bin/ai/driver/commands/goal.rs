use colored::Colorize;

use crate::ai::types::App;

/// Builds the initial goal-mode prompt: wraps the user's goal into an explicit
/// user message with instructions to keep executing.
pub(crate) fn build_goal_prompt(goal: &str) -> String {
    format!(
        "你正在 GOAL MODE 下工作。这是一个长期自主任务，目标是：\n\
         ---\n\
         {goal}\n\
         ---\n\
         \n\
         请全力以赴地完成这个目标。你可以调用任何可用的工具来推进工作。\n\
         在每一轮结束时，如果你认为目标已经完全达成，请不要再调用任何工具，\n\
         直接用一段文字总结你完成的工作即可。如果目标尚未达成，请继续执行下一步。"
    )
}

/// Builds the goal-mode continuation prompt, injected automatically after the
/// previous turn to drive the agent forward.
pub(crate) fn build_goal_continuation_prompt(goal: &str) -> String {
    format!(
        "[GOAL MODE - 继续] 你的目标是：{goal}\n\
         \n\
         请回顾你目前的进展，继续推进目标的实现。\n\
         - 如果目标已经完全达成，不要再调用工具，直接用文字总结你的工作成果。\n\
         - 如果还有未完成的部分，立即继续执行下一步行动。"
    )
}

/// End-of-turn decision for goal mode when no continuation was triggered this turn.
///
/// At the start of each `run_loop` iteration, if a goal is set and the previous
/// turn called tools, a continuation prompt is injected to keep going (this
/// function is not reached). Otherwise this function decides: no tool calls in
/// the previous turn means either the agent delivered its final result (goal
/// achieved) or the turn was interrupted by Ctrl+C. Both set `had_tool_calls`
/// to false, but their meaning is opposite: an interruption is not an
/// achievement, so goal mode must be preserved.
pub(crate) fn should_exit_goal_on_idle(
    goal_active: bool,
    one_shot_mode: bool,
    interrupted: bool,
) -> bool {
    goal_active && !one_shot_mode && !interrupted
}

/// Handles the `/goal` interactive command.
///
/// Usage:
/// - `/goal`            -- enter the goal waiting state; the next user input
///                         becomes the goal
/// - `/goal <content>`  -- enter goal mode directly with `<content>` as the goal
/// - `/goal exit`       -- exit goal mode (`/goal off`, `/goal stop` are synonyms)
/// - `/goal status`     -- show the current goal mode state
pub fn try_handle_goal_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if !trimmed.starts_with("/goal") {
        return Ok(false);
    }

    let rest = trimmed["/goal".len()..].trim();

    // /goal status -- show the state
    if rest.eq_ignore_ascii_case("status") {
        match &app.goal_mode {
            None => println!("Goal mode: {}", "off".dimmed()),
            Some(g) if g.is_empty() => {
                println!("Goal mode: {} (waiting for goal input)", "pending".yellow())
            }
            Some(g) => println!("Goal mode: {}\n  Goal: {}", "active".green().bold(), g),
        }
        return Ok(true);
    }

    // /goal exit / off / stop -- exit goal mode
    if rest.eq_ignore_ascii_case("exit")
        || rest.eq_ignore_ascii_case("off")
        || rest.eq_ignore_ascii_case("stop")
        || rest.eq_ignore_ascii_case("quit")
    {
        if app.goal_mode.is_some() {
            app.goal_mode = None;
            println!("{} Goal mode deactivated.", "[goal]".cyan().bold());
        } else {
            println!("{} Goal mode is not active.", "[goal]".dimmed());
        }
        return Ok(true);
    }

    // /goal <content> -- set the goal directly and enter goal mode
    if !rest.is_empty() {
        app.goal_mode = Some(rest.to_string());
        let prompt = build_goal_prompt(rest);
        app.forced_question = Some(prompt);
        println!(
            "{} Goal mode activated. Goal: {}",
            "[goal]".cyan().bold(),
            rest
        );
        return Ok(true);
    }

    // /goal -- enter the waiting state; the next input becomes the goal
    if app.goal_mode.is_some() {
        // Already in goal mode; a bare /goal is a no-op (avoids overwriting the goal)
        println!(
            "{} Goal mode is already active. Use '/goal exit' to stop or '/goal status' to check.",
            "[goal]".yellow()
        );
        return Ok(true);
    }
    app.goal_mode = Some(String::new());
    println!(
        "{} Goal mode: waiting for goal input.\n\
         Type your goal and press Enter. (or '/goal exit' to cancel)",
        "[goal]".cyan().bold()
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_app() -> App {
        use std::sync::{Arc, atomic::AtomicBool};
        crate::ai::types::App {
            cli: crate::ai::cli::ParsedCli::default(),
            config: crate::ai::types::AppConfig {
                api_key: String::new(),
                base_history_file: std::path::PathBuf::new(),
                history_file: std::path::PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 8000,
                history_keep_last: 10,
                history_summary_max_chars: 4000,
                intent_model: None,
            },
            session_id: "test".to_string(),
            session_history_file: std::path::PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: "test".to_string(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skills: Vec::new(),
            forced_skill_source: None,
            pending_skill_continuation: None,
            forced_question: None,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            prompt_editor: None,
            agent_context: None,
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: Vec::new(),
            last_known_prompt_tokens: None,
            last_known_cached_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
            last_turn_interrupted: false,
            prune_marks: Default::default(),
            turn_reasoning_items: Default::default(),
            stale_patch_targets: Default::default(),
            tool_middlewares: Vec::new(),
            llm_middlewares: Vec::new(),
        hooks: Default::default(),
        }
    }

    #[test]
    fn goal_command_not_triggered_for_non_goal_input() {
        let mut app = test_app();
        assert!(!try_handle_goal_command(&mut app, "hello world").unwrap());
        assert!(!try_handle_goal_command(&mut app, "/help").unwrap());
        assert!(app.goal_mode.is_none());
    }

    #[test]
    fn goal_bare_enters_waiting_state() {
        let mut app = test_app();
        assert!(try_handle_goal_command(&mut app, "/goal").unwrap());
        assert_eq!(app.goal_mode, Some(String::new()));
        assert!(app.forced_question.is_none());
    }

    #[test]
    fn goal_with_content_sets_goal_and_forced_question() {
        let mut app = test_app();
        assert!(try_handle_goal_command(&mut app, "/goal refactor the auth module").unwrap());
        assert_eq!(app.goal_mode.as_deref(), Some("refactor the auth module"));
        assert!(app.forced_question.is_some());
        assert!(
            app.forced_question
                .as_ref()
                .unwrap()
                .contains("refactor the auth module")
        );
    }

    #[test]
    fn goal_exit_clears_goal_mode() {
        let mut app = test_app();
        app.goal_mode = Some("some goal".to_string());
        assert!(try_handle_goal_command(&mut app, "/goal exit").unwrap());
        assert!(app.goal_mode.is_none());

        // Also test off/stop/quit
        app.goal_mode = Some("some goal".to_string());
        assert!(try_handle_goal_command(&mut app, "/goal off").unwrap());
        assert!(app.goal_mode.is_none());

        app.goal_mode = Some("some goal".to_string());
        assert!(try_handle_goal_command(&mut app, "/goal stop").unwrap());
        assert!(app.goal_mode.is_none());

        app.goal_mode = Some("some goal".to_string());
        assert!(try_handle_goal_command(&mut app, "/goal quit").unwrap());
        assert!(app.goal_mode.is_none());
    }

    #[test]
    fn goal_status_shows_state() {
        let mut app = test_app();
        // Off
        assert!(try_handle_goal_command(&mut app, "/goal status").unwrap());
        assert!(app.goal_mode.is_none());

        // Waiting
        app.goal_mode = Some(String::new());
        assert!(try_handle_goal_command(&mut app, "/goal status").unwrap());

        // Active
        app.goal_mode = Some("do something".to_string());
        assert!(try_handle_goal_command(&mut app, "/goal status").unwrap());
    }

    #[test]
    fn goal_bare_while_active_does_not_overwrite() {
        let mut app = test_app();
        app.goal_mode = Some("existing goal".to_string());
        assert!(try_handle_goal_command(&mut app, "/goal").unwrap());
        assert_eq!(app.goal_mode.as_deref(), Some("existing goal"));
    }

    #[test]
    fn goal_continuation_prompt_contains_goal() {
        let prompt = build_goal_continuation_prompt("test goal");
        assert!(prompt.contains("test goal"));
        assert!(prompt.contains("GOAL MODE"));
    }

    #[test]
    fn idle_goal_exits_only_on_natural_completion() {
        // Natural completion (not interrupted) -> exit goal mode and notify.
        assert!(should_exit_goal_on_idle(true, false, false));
        // Interrupted by Ctrl+C -> keep goal mode; do not report a false completion.
        assert!(!should_exit_goal_on_idle(true, false, true));
        // Goal not active -> no-op.
        assert!(!should_exit_goal_on_idle(false, false, false));
        // One-shot mode -> do not manage the goal lifecycle.
        assert!(!should_exit_goal_on_idle(true, true, false));
    }
}
