use colored::Colorize;

use crate::ai::types::App;

/// 构造 goal 模式的初始 prompt——把用户的目标包装成一条明确的、带有持续执行指令的 user message。
pub(crate) fn build_goal_prompt(goal: &str) -> String {
    format!(
        "你正在 GOAL MODE 下工作。你的目标是：\n\
         ---\n\
         {goal}\n\
         ---\n\
         \n\
         请全力以赴地完成这个目标。你可以调用任何可用的工具来推进工作。\n\
         在每一轮结束时，如果你认为目标已经完全达成，请不要再调用任何工具，\n\
         直接用一段文字总结你完成的工作即可。如果目标尚未达成，请继续执行下一步。"
    )
}

/// 构造 goal 模式的后续 continuation prompt——在上一轮结束后自动注入，驱动 agent 继续推进。
pub(crate) fn build_goal_continuation_prompt(goal: &str) -> String {
    format!(
        "[GOAL MODE - 继续]\n\
         你的目标是：{goal}\n\
         \n\
         请回顾你目前的进展，继续推进目标的实现。\n\
         - 如果目标已经完全达成，不要再调用工具，直接用文字总结你的工作成果。\n\
         - 如果还有未完成的部分，立即继续执行下一步行动。"
    )
}

/// 处理 `/goal` 交互式命令。
///
/// 用法：
/// - `/goal`            — 进入 goal 等待状态，下一条用户输入将被作为目标
/// - `/goal <内容>`      — 直接以 `<内容>` 为目标进入 goal 模式
/// - `/goal exit`       — 退出 goal 模式（`/goal off`、`/goal stop` 同义）
/// - `/goal status`     — 查看当前 goal 模式状态
pub fn try_handle_goal_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if !trimmed.starts_with("/goal") {
        return Ok(false);
    }

    let rest = trimmed["/goal".len()..].trim();

    // /goal status — 查看状态
    if rest.eq_ignore_ascii_case("status") {
        match &app.goal_mode {
            None => println!("Goal mode: {}", "off".dimmed()),
            Some(g) if g.is_empty() => println!(
                "Goal mode: {} (waiting for goal input)",
                "pending".yellow()
            ),
            Some(g) => println!(
                "Goal mode: {}\n  Goal: {}",
                "active".green().bold(),
                g
            ),
        }
        return Ok(true);
    }

    // /goal exit / off / stop — 退出 goal 模式
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

    // /goal <内容> — 直接设定目标并进入 goal 模式
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

    // /goal — 进入等待状态，下一条输入作为目标
    if app.goal_mode.is_some() {
        // 已在 goal 模式中，再次输入 /goal 无操作（避免覆盖已有目标）
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
                agent_route_model_path: std::path::PathBuf::new(),
                skill_match_model_path: std::path::PathBuf::new(),
            },
            session_id: "test".to_string(),
            session_history_file: std::path::PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: "test".to_string(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skill: None,
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
            goal_mode: None,
            last_turn_had_tool_calls: false,
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
        assert!(app.forced_question.as_ref().unwrap().contains("refactor the auth module"));
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
}
