use crate::ai::{tools::task_tools::SubagentBackend, types::App};

fn ensure_agent_team_sticky(app: &mut App) {
    // Keep the first stored spelling and drop later case variants so the sticky
    // selection stays a set even if the persisted vector already holds duplicates.
    let mut seen = false;
    app.session_default_skills.retain(|name| {
        if name.eq_ignore_ascii_case("agent-team") {
            if seen {
                return false;
            }
            seen = true;
        }
        true
    });
    if !seen {
        app.session_default_skills.push("agent-team".to_string());
    }
}

fn print_use_state(app: &App) {
    let backend = app.session_default_subagent_backend;
    println!("Session default subagent backend: {}", backend.as_str());
    println!(
        "TraeCLI worker model: {}",
        crate::ai::tools::task_tools::traecli_model_for(
            app.session_default_subagent_model.as_deref()
        )
    );
    println!(
        "Sticky agent-team: {}",
        if app
            .session_default_skills
            .iter()
            .any(|name| name.eq_ignore_ascii_case("agent-team"))
        {
            "active"
        } else {
            "off"
        }
    );
}

pub(crate) fn try_handle_use_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let (command_line, prompt) = trimmed
        .split_once('\n')
        .map_or((trimmed, None), |(head, tail)| {
            (head.trim(), Some(tail.trim_start()))
        });
    let normalized = if let Some(rest) = command_line.strip_prefix('/') {
        rest
    } else if let Some(rest) = command_line.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };

    let mut parts = normalized.split_whitespace();
    if parts.next() != Some("use") {
        return Ok(false);
    }
    let Some(arg) = parts.next() else {
        print_use_state(app);
        if prompt.is_some_and(|prompt| !prompt.is_empty()) {
            app.forced_question = prompt.map(str::to_string);
        }
        return Ok(true);
    };
    let model = parts.next();
    if parts.next().is_some() {
        println!("Too many arguments. Usage: /use [traecli [model]|kernel]");
        return Ok(true);
    }

    let valid = match arg {
        "traecli" => {
            app.session_default_subagent_backend = SubagentBackend::TraeCli;
            if let Some(model) = model {
                app.session_default_subagent_model = Some(model.to_string());
            }
            ensure_agent_team_sticky(app);
            super::session::persist_session_runtime_state(app);
            print_use_state(app);
            true
        }
        "kernel" if model.is_none() => {
            app.session_default_subagent_backend = SubagentBackend::Kernel;
            ensure_agent_team_sticky(app);
            super::session::persist_session_runtime_state(app);
            print_use_state(app);
            true
        }
        "kernel" => {
            println!("Kernel does not take a worker model. Usage: /use kernel");
            false
        }
        _ => {
            println!("Unknown backend '{arg}'. Usage: /use [traecli [model]|kernel]");
            false
        }
    };
    if valid && prompt.is_some_and(|prompt| !prompt.is_empty()) {
        app.forced_question = prompt.map(str::to_string);
    }
    Ok(true)
}

fn parse_reset_command(input: &str) -> Option<Result<(), &'static str>> {
    let trimmed = input.trim();
    let normalized = trimmed
        .strip_prefix('/')
        .or_else(|| trimmed.strip_prefix(':'))?;
    let mut parts = normalized.split_whitespace();
    if parts.next() != Some("reset") {
        return None;
    }
    Some(if parts.next().is_some() {
        Err("Usage: /reset (no arguments or trailing prompt)")
    } else {
        Ok(())
    })
}

pub(crate) fn try_handle_reset_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(parsed) = parse_reset_command(input) else {
        return Ok(false);
    };
    if let Err(usage) = parsed {
        println!("{usage}");
        return Ok(true);
    }

    app.session_default_subagent_backend = SubagentBackend::Kernel;
    app.session_default_subagent_model = None;
    app.session_default_skills
        .retain(|name| !name.eq_ignore_ascii_case("agent-team"));
    if let Some(continuation) = app.pending_skill_continuation.as_mut() {
        continuation
            .skill_names
            .retain(|name| !name.eq_ignore_ascii_case("agent-team"));
        if continuation.skill_names.is_empty() {
            app.pending_skill_continuation = None;
        }
    }
    super::session::persist_session_runtime_state(app);
    print_use_state(app);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        history::{Message, SessionStore, append_history_messages, build_message_arr},
        types::AppConfig,
    };
    use std::sync::{Arc, atomic::AtomicBool};
    use std::{fs, path::PathBuf};
    use uuid::Uuid;

    fn test_app() -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                base_history_file: PathBuf::new(),
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 12000,
                history_keep_last: 8,
                history_summary_max_chars: 4000,
                intent_model: None,
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::new(),
            current_model: String::new(),
            current_agent: "build".to_string(),
            session_default_subagent_backend: SubagentBackend::Kernel,
            session_default_subagent_model: None,
            session_default_skills: Vec::new(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skills: Vec::new(),
            forced_skill_source: None,
            scoped_preflight_required: Vec::new(),
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
            tool_middlewares: Vec::new(),
            llm_middlewares: Vec::new(),
            hooks: Default::default(),
            last_known_prompt_tokens: None,
            last_known_cached_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
            last_turn_interrupted: false,
            prune_marks: Default::default(),
            turn_reasoning_items: Default::default(),
            turn_tool_errors: Default::default(),
            stale_patch_targets: Default::default(),
        }
    }

    #[test]
    fn use_command_switches_backend_and_keeps_session_defaults() {
        let mut app = test_app();
        app.current_model = "parent-model".to_string();
        assert!(try_handle_use_command(&mut app, " /use traecli GPT-5.5 ").unwrap());
        assert_eq!(
            app.session_default_subagent_backend,
            SubagentBackend::TraeCli
        );
        assert_eq!(
            app.session_default_subagent_model.as_deref(),
            Some("GPT-5.5")
        );
        assert_eq!(app.session_default_skills, ["agent-team"]);
        assert_eq!(app.current_model, "parent-model");
        assert!(try_handle_use_command(&mut app, ":use traecli").unwrap());
        assert_eq!(
            app.session_default_subagent_model.as_deref(),
            Some("GPT-5.5")
        );
        assert_eq!(app.session_default_skills, ["agent-team"]);
        app.pending_skill_continuation = Some(crate::ai::types::PendingSkillContinuation {
            skill_names: vec!["agent-team".to_string()],
        });
        assert!(try_handle_use_command(&mut app, ":use kernel").unwrap());
        assert_eq!(
            app.session_default_subagent_backend,
            SubagentBackend::Kernel
        );
        assert_eq!(app.session_default_skills, ["agent-team"]);
        assert_eq!(
            app.pending_skill_continuation.as_ref().unwrap().skill_names,
            ["agent-team"]
        );
        assert_eq!(
            app.session_default_subagent_model.as_deref(),
            Some("GPT-5.5")
        );
        assert_eq!(app.current_model, "parent-model");
    }

    #[test]
    fn use_kernel_adds_agent_team_to_empty_sticky_skills() {
        let mut app = test_app();
        assert!(try_handle_use_command(&mut app, "/use kernel").unwrap());
        assert_eq!(
            app.session_default_subagent_backend,
            SubagentBackend::Kernel
        );
        assert_eq!(app.session_default_skills, ["agent-team"]);
        assert!(app.session_default_subagent_model.is_none());
        assert!(app.pending_skill_continuation.is_none());
    }

    #[test]
    fn use_command_preserves_other_sticky_skills_and_deduplicates_agent_team() {
        let mut app = test_app();
        app.session_default_skills = vec![
            "first-skill".to_string(),
            "AGENT-TEAM".to_string(),
            "agent-team".to_string(),
            "last-skill".to_string(),
        ];
        app.pending_skill_continuation = Some(crate::ai::types::PendingSkillContinuation {
            skill_names: vec!["AGENT-TEAM".to_string(), "continuation-skill".to_string()],
        });
        for command in ["/use kernel", ":use traecli", ":use kernel"] {
            assert!(try_handle_use_command(&mut app, command).unwrap());
            assert_eq!(
                app.session_default_skills,
                ["first-skill", "AGENT-TEAM", "last-skill"]
            );
            assert_eq!(
                app.pending_skill_continuation.as_ref().unwrap().skill_names,
                ["AGENT-TEAM", "continuation-skill"]
            );
        }
    }

    #[test]
    fn reset_command_clears_coupled_defaults_and_preserves_other_state() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root =
            std::env::temp_dir().join(format!("rust-tools-reset-command-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let history_file = root.join("history.sqlite");
        let mut app = test_app();
        app.config.base_history_file = history_file.clone();
        app.config.history_file = history_file.clone();
        app.session_id = "reset-command".to_string();
        app.session_history_file =
            SessionStore::new(&history_file).session_history_file(&app.session_id);
        app.session_default_subagent_backend = SubagentBackend::TraeCli;
        app.session_default_subagent_model = Some("saved-worker-model".to_string());
        app.session_default_skills = vec![
            "first-skill".to_string(),
            "AGENT-TEAM".to_string(),
            "last-skill".to_string(),
            "agent-team".to_string(),
        ];
        app.pending_skill_continuation = Some(crate::ai::types::PendingSkillContinuation {
            skill_names: vec![
                "AGENT-TEAM".to_string(),
                "continuation-skill".to_string(),
                "agent-team".to_string(),
            ],
        });
        app.current_model = "parent-model".to_string();
        app.current_agent = "parent-agent".to_string();
        app.forced_skills = vec!["agent-team".to_string(), "forced-skill".to_string()];
        app.forced_skill_source = Some(crate::ai::types::ForcedSkillSource::InlineReference);
        app.forced_question = Some("pending prompt".to_string());
        app.cli.reasoning_effort_override = Some(None);
        let history = [Message {
            role: "user".to_string(),
            content: serde_json::Value::String("Keep this conversation".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        append_history_messages(&app.session_history_file, &history).unwrap();
        super::super::session::persist_session_runtime_state(&app);

        assert!(super::super::try_handle_local_command(&mut app, "/reset").unwrap());

        assert_eq!(
            app.session_default_subagent_backend,
            SubagentBackend::Kernel
        );
        assert!(app.session_default_subagent_model.is_none());
        assert_eq!(app.session_default_skills, ["first-skill", "last-skill"]);
        assert_eq!(
            app.pending_skill_continuation.as_ref().unwrap().skill_names,
            ["continuation-skill"]
        );
        assert_eq!(app.current_model, "parent-model");
        assert_eq!(app.current_agent, "parent-agent");
        assert_eq!(app.forced_skills, ["agent-team", "forced-skill"]);
        assert_eq!(
            app.forced_skill_source,
            Some(crate::ai::types::ForcedSkillSource::InlineReference)
        );
        assert_eq!(app.forced_question.as_deref(), Some("pending prompt"));
        assert_eq!(app.cli.reasoning_effort_override, Some(None));
        assert_eq!(
            serde_json::to_value(build_message_arr(usize::MAX, &app.session_history_file).unwrap())
                .unwrap(),
            serde_json::to_value(&history).unwrap()
        );
        let stored = SessionStore::new(&history_file)
            .read_session_serve_config(&app.session_id)
            .unwrap();
        assert!(stored.subagent_backend.is_none());
        assert!(stored.subagent_model.is_none());
        assert_eq!(stored.default_skills, ["first-skill", "last-skill"]);
        assert_eq!(stored.model.as_deref(), Some("parent-model"));
        assert_eq!(stored.agent.as_deref(), Some("parent-agent"));
        assert_eq!(stored.reasoning_effort.as_deref(), Some("off"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn reset_command_clears_agent_team_only_continuation_and_accepts_both_prefixes() {
        for command in ["/reset", " \r\n :reset \t\r\n "] {
            let mut app = test_app();
            app.session_default_subagent_backend = SubagentBackend::TraeCli;
            app.session_default_subagent_model = Some("saved-worker-model".to_string());
            app.session_default_skills = vec!["AGENT-TEAM".to_string()];
            app.pending_skill_continuation = Some(crate::ai::types::PendingSkillContinuation {
                skill_names: vec!["Agent-Team".to_string()],
            });
            assert!(try_handle_reset_command(&mut app, command).unwrap());
            assert_eq!(
                app.session_default_subagent_backend,
                SubagentBackend::Kernel
            );
            assert!(app.session_default_subagent_model.is_none());
            assert!(app.session_default_skills.is_empty());
            assert!(app.pending_skill_continuation.is_none());
        }
    }

    #[test]
    fn reset_command_rejects_arguments_and_trailing_prompt_without_changing_state() {
        let mut app = test_app();
        app.session_default_subagent_backend = SubagentBackend::TraeCli;
        app.session_default_subagent_model = Some("saved-worker-model".to_string());
        app.session_default_skills = vec!["AGENT-TEAM".to_string(), "other-skill".to_string()];
        app.pending_skill_continuation = Some(crate::ai::types::PendingSkillContinuation {
            skill_names: vec!["agent-team".to_string(), "continuation-skill".to_string()],
        });
        app.current_model = "parent-model".to_string();
        app.current_agent = "parent-agent".to_string();
        app.forced_skills = vec!["forced-skill".to_string()];
        app.forced_skill_source = Some(crate::ai::types::ForcedSkillSource::InlineReference);
        app.forced_question = Some("pending prompt".to_string());
        for input in [
            "/reset extra",
            ":reset extra",
            "/reset\ntrailing prompt",
            ":reset\r\ntrailing prompt",
        ] {
            assert_eq!(
                parse_reset_command(input),
                Some(Err("Usage: /reset (no arguments or trailing prompt)"))
            );
            assert!(try_handle_reset_command(&mut app, input).unwrap());
            assert_eq!(
                app.session_default_subagent_backend,
                SubagentBackend::TraeCli
            );
            assert_eq!(
                app.session_default_subagent_model.as_deref(),
                Some("saved-worker-model")
            );
            assert_eq!(app.session_default_skills, ["AGENT-TEAM", "other-skill"]);
            assert_eq!(
                app.pending_skill_continuation.as_ref().unwrap().skill_names,
                ["agent-team", "continuation-skill"]
            );
            assert_eq!(app.current_model, "parent-model");
            assert_eq!(app.current_agent, "parent-agent");
            assert_eq!(app.forced_skills, ["forced-skill"]);
            assert_eq!(
                app.forced_skill_source,
                Some(crate::ai::types::ForcedSkillSource::InlineReference)
            );
            assert_eq!(app.forced_question.as_deref(), Some("pending prompt"));
        }
    }

    #[test]
    fn reset_command_ignores_other_input() {
        let mut app = test_app();
        for input in [
            "",
            "reset",
            "/resetting",
            ":resetting",
            "/use kernel",
            "/model parent-model",
            "/sessions clear-history",
        ] {
            assert!(!try_handle_reset_command(&mut app, input).unwrap());
        }
        assert!(app.session_default_skills.is_empty());
    }

    #[test]
    fn use_command_ignores_other_input() {
        let mut app = test_app();
        assert!(!try_handle_use_command(&mut app, "/model foo").unwrap());
        assert!(!try_handle_use_command(&mut app, "use traecli").unwrap());
        assert!(!try_handle_use_command(&mut app, "/backend traecli").unwrap());
        assert!(!try_handle_use_command(&mut app, ":backend traecli").unwrap());
        assert!(!try_handle_use_command(&mut app, "/reset").unwrap());
        assert!(!try_handle_use_command(&mut app, ":reset").unwrap());
    }

    #[test]
    fn use_command_rejects_invalid_arguments_without_changing_state() {
        let mut app = test_app();
        for input in [
            "/use other\n任务",
            "/use kernel GPT-5.5\n任务",
            "/use traecli M extra\n任务",
        ] {
            assert!(try_handle_use_command(&mut app, input).unwrap());
            assert_eq!(
                app.session_default_subagent_backend,
                SubagentBackend::Kernel
            );
            assert!(app.session_default_subagent_model.is_none());
            assert!(app.session_default_skills.is_empty());
            assert!(app.forced_question.is_none());
        }
    }

    #[test]
    fn use_command_without_model_uses_defaults_and_bare_use_is_read_only() {
        let mut app = test_app();
        assert!(try_handle_use_command(&mut app, "/use").unwrap());
        assert!(app.session_default_subagent_model.is_none());
        assert!(app.session_default_skills.is_empty());
        assert!(try_handle_use_command(&mut app, "/use traecli").unwrap());
        assert!(app.session_default_subagent_model.is_none());
        assert_eq!(app.session_default_skills, ["agent-team"]);
        assert!(try_handle_use_command(&mut app, ":use").unwrap());
        assert_eq!(
            app.session_default_subagent_backend,
            SubagentBackend::TraeCli
        );
        assert_eq!(app.session_default_skills, ["agent-team"]);
    }

    #[test]
    fn use_command_starts_trailing_prompt_with_updated_defaults() {
        let mut app = test_app();
        let root = std::env::temp_dir().join(format!("rust-tools-use-command-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let history_file = root.join("history.sqlite");
        app.config.base_history_file = history_file.clone();
        app.config.history_file = history_file.clone();
        app.session_id = "use-command".to_string();
        assert!(
            try_handle_use_command(&mut app, "/use traecli GPT-6.1-Sol\n帮我做一个xxx任务")
                .unwrap()
        );
        assert_eq!(
            app.session_default_subagent_backend,
            SubagentBackend::TraeCli
        );
        assert_eq!(
            app.session_default_subagent_model.as_deref(),
            Some("GPT-6.1-Sol")
        );
        assert_eq!(app.session_default_skills, ["agent-team"]);
        assert_eq!(app.forced_question.as_deref(), Some("帮我做一个xxx任务"));
        let stored = SessionStore::new(&history_file)
            .read_session_serve_config(&app.session_id)
            .unwrap();
        assert_eq!(stored.subagent_backend.as_deref(), Some("traecli"));
        assert_eq!(stored.subagent_model.as_deref(), Some("GPT-6.1-Sol"));
        assert_eq!(stored.default_skills, ["agent-team"]);

        app.forced_question = None;
        assert!(try_handle_use_command(&mut app, "/use traecli\n任务").unwrap());
        assert_eq!(
            app.session_default_subagent_model.as_deref(),
            Some("GPT-6.1-Sol")
        );
        assert_eq!(app.forced_question.as_deref(), Some("任务"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn use_command_bare_and_kernel_forms_start_trailing_prompt() {
        let mut app = test_app();
        app.session_default_subagent_backend = SubagentBackend::TraeCli;
        app.session_default_subagent_model = Some("saved-model".to_string());
        app.session_default_skills = vec!["agent-team".to_string()];

        assert!(try_handle_use_command(&mut app, "/use\n任务").unwrap());
        assert_eq!(
            app.session_default_subagent_backend,
            SubagentBackend::TraeCli
        );
        assert_eq!(
            app.session_default_subagent_model.as_deref(),
            Some("saved-model")
        );
        assert_eq!(app.session_default_skills, ["agent-team"]);
        assert_eq!(app.forced_question.as_deref(), Some("任务"));

        app.forced_question = None;
        app.pending_skill_continuation = Some(crate::ai::types::PendingSkillContinuation {
            skill_names: vec!["agent-team".to_string()],
        });
        assert!(try_handle_use_command(&mut app, "/use kernel\n任务").unwrap());
        assert_eq!(
            app.session_default_subagent_backend,
            SubagentBackend::Kernel
        );
        assert_eq!(
            app.session_default_subagent_model.as_deref(),
            Some("saved-model")
        );
        assert_eq!(app.session_default_skills, ["agent-team"]);
        assert_eq!(
            app.pending_skill_continuation.as_ref().unwrap().skill_names,
            ["agent-team"]
        );
        assert_eq!(app.forced_question.as_deref(), Some("任务"));
    }

    #[test]
    fn use_command_preserves_multiline_prompt_and_accepts_crlf() {
        let mut app = test_app();
        assert!(
            try_handle_use_command(
                &mut app,
                "\r\n  :use traecli\r\n\r\n  第一行\r\n第二行  \r\n"
            )
            .unwrap()
        );
        assert_eq!(app.forced_question.as_deref(), Some("第一行\r\n第二行"));
    }

    #[test]
    fn use_command_runtime_snapshots_preserve_settings_and_children_skip_team() {
        let mut app = test_app();
        app.session_default_subagent_backend = SubagentBackend::TraeCli;
        app.session_default_subagent_model = Some("GPT-6.1-Sol".to_string());
        app.session_default_skills = vec!["agent-team".to_string(), "other-skill".to_string()];
        for snapshot in [
            app.snapshot_for_driver_context(),
            app.snapshot_for_detached_helper(),
        ] {
            assert_eq!(
                snapshot.session_default_subagent_backend,
                SubagentBackend::TraeCli
            );
            assert_eq!(
                snapshot.session_default_subagent_model,
                app.session_default_subagent_model
            );
            assert_eq!(snapshot.session_default_skills, app.session_default_skills);
        }
        let child = app.fork_for_subagent();
        assert_eq!(
            child.session_default_subagent_backend,
            SubagentBackend::TraeCli
        );
        assert_eq!(
            child.session_default_subagent_model,
            app.session_default_subagent_model
        );
        assert_eq!(child.session_default_skills, ["other-skill"]);
        assert_eq!(app.session_default_skills, ["agent-team", "other-skill"]);
    }
}
