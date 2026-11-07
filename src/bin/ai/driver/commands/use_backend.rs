use crate::ai::{
    history::SessionServeConfig,
    tools::task_tools::{BackendId, ModelContext, backends::backend_registry},
    types::App,
};

fn ensure_agent_team_sticky(config: &mut SessionServeConfig) {
    // Keep the first stored spelling and drop later case variants so the sticky
    // selection stays a set even if the persisted vector already holds duplicates.
    let mut seen = false;
    config.default_skills.retain(|name| {
        if name.eq_ignore_ascii_case("agent-team") {
            if seen {
                return false;
            }
            seen = true;
        }
        true
    });
    if !seen {
        config.default_skills.push("agent-team".to_string());
    }
}

pub(crate) fn parse_use_arguments(args: &str) -> Result<(Option<&str>, Option<&str>), String> {
    let mut parts = args.split_whitespace();
    let backend = parts.next();
    let model = parts.next();
    if parts.next().is_some() {
        return Err(format!(
            "Too many arguments. Usage: {}",
            backend_registry().use_usage()
        ));
    }
    Ok((backend, model))
}

/// Apply the same backend validation and sticky-skill rule for local and HTTP commands.
pub(crate) fn apply_subagent_use(
    config: &mut SessionServeConfig,
    name: &str,
    model: Option<&str>,
) -> Result<(), String> {
    let Some(backend) = backend_registry().use_id(name) else {
        return Err(format!(
            "Unknown backend '{name}'. Usage: {}",
            backend_registry().use_usage()
        ));
    };
    let implementation = backend.resolve();
    let descriptor = implementation.descriptor();
    if model.is_some() && !descriptor.supports_worker_model {
        return Err(format!(
            "{} does not take a worker model. Usage: /use {}",
            descriptor.text.display_name, descriptor.use_names[0]
        ));
    }
    // Keep explicit choices distinct from the available session default.
    config.subagent_backend = Some(backend.as_str().to_string());
    if let Some(model) = model {
        config.subagent_model = Some(model.to_string());
    }
    ensure_agent_team_sticky(config);
    Ok(())
}

pub(crate) fn reset_subagent_config(config: &mut SessionServeConfig) {
    config.subagent_backend = None;
    config.subagent_model = None;
    config
        .default_skills
        .retain(|name| !name.eq_ignore_ascii_case("agent-team"));
}

pub(crate) fn format_use_state(config: &SessionServeConfig, kernel_model: &str) -> String {
    let backend = BackendId::from_session_config(config.subagent_backend.as_deref());
    let mut output = format!("Session default subagent backend: {}\n", backend.as_str());
    if let Some(worker_backend) = backend_registry().worker_backend_for(backend) {
        let implementation = worker_backend.resolve();
        if let Some(label) = implementation.descriptor().text.worker_model_label {
            let identity = implementation.attribution(&ModelContext {
                kernel_model,
                backend_model: config.subagent_model.as_deref(),
                session_worker_model: None,
                kernel_reasoning_effort: None,
                backend_reasoning_effort: None,
                selection_explanation: "",
            });
            output.push_str(&format!("{label}: {}\n", identity.model));
        }
    }
    output.push_str(&format!(
        "Sticky agent-team: {}\n",
        if config
            .default_skills
            .iter()
            .any(|name| name.eq_ignore_ascii_case("agent-team"))
        {
            "active"
        } else {
            "off"
        }
    ));
    output
}

fn app_subagent_config(app: &App) -> SessionServeConfig {
    SessionServeConfig {
        subagent_backend: app.session_default_subagent_backend.stored_value(),
        subagent_model: app.session_default_subagent_model.clone(),
        default_skills: app.session_default_skills.clone(),
        ..SessionServeConfig::default()
    }
}

fn apply_app_subagent_config(app: &mut App, config: SessionServeConfig) {
    app.session_default_subagent_backend =
        BackendId::from_session_config(config.subagent_backend.as_deref());
    app.session_default_subagent_model = config.subagent_model;
    app.session_default_skills = config.default_skills;
}

fn print_use_state(app: &App) {
    print!(
        "{}",
        format_use_state(&app_subagent_config(app), &app.current_model)
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
    let (arg, model) = match parse_use_arguments(&normalized.trim_start()["use".len()..]) {
        Ok(args) => args,
        Err(usage) => {
            println!("{usage}");
            return Ok(true);
        }
    };
    let Some(arg) = arg else {
        print_use_state(app);
        if prompt.is_some_and(|prompt| !prompt.is_empty()) {
            app.forced_question = prompt.map(str::to_string);
        }
        return Ok(true);
    };
    let mut config = app_subagent_config(app);
    if let Err(usage) = apply_subagent_use(&mut config, arg, model) {
        println!("{usage}");
        return Ok(true);
    }
    apply_app_subagent_config(app, config);
    super::session::persist_session_runtime_state(app);
    print_use_state(app);
    if prompt.is_some_and(|prompt| !prompt.is_empty()) {
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
    Some(validate_reset_arguments(
        &normalized.trim_start()["reset".len()..],
    ))
}

pub(crate) fn validate_reset_arguments(args: &str) -> Result<(), &'static str> {
    if !args.trim().is_empty() {
        Err("Usage: /reset (no arguments or trailing prompt)")
    } else {
        Ok(())
    }
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

    let mut config = app_subagent_config(app);
    reset_subagent_config(&mut config);
    apply_app_subagent_config(app, config);
    if let Some(continuation) = app.pending_skill_continuation.as_mut() {
        continuation
            .skill_names
            .retain(|name| !name.eq_ignore_ascii_case("agent-team"));
        if continuation.skill_names.is_empty() {
            app.pending_skill_continuation = None;
        }
    }
    crate::ai::tools::enable_tools::clear_task_family_explicitly_enabled_tools(&app.session_id);
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
            session_default_subagent_backend: BackendId::KERNEL,
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
        assert_eq!(app.session_default_subagent_backend, BackendId::TRAECLI);
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
        assert_eq!(app.session_default_subagent_backend, BackendId::KERNEL);
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
        assert_eq!(app.session_default_subagent_backend, BackendId::KERNEL);
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
        app.session_default_subagent_backend = BackendId::TRAECLI;
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
            BackendId::from_session_config(None)
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
        assert_eq!(
            stored.subagent_backend.as_deref(),
            Some(app.session_default_subagent_backend.as_str())
        );
        assert!(stored.subagent_model.is_none());
        assert_eq!(stored.default_skills, ["first-skill", "last-skill"]);
        assert_eq!(stored.model.as_deref(), Some("parent-model"));
        assert_eq!(stored.agent.as_deref(), Some("parent-agent"));
        assert_eq!(stored.reasoning_effort.as_deref(), Some("off"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn reset_command_drops_task_family_explicit_enables_only() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root =
            std::env::temp_dir().join(format!("rust-tools-reset-task-tools-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let history_file = root.join("history.sqlite");
        let mut app = test_app();
        app.config.base_history_file = history_file.clone();
        app.config.history_file = history_file.clone();
        app.session_id = "reset-task-tools".to_string();
        app.session_history_file =
            SessionStore::new(&history_file).session_history_file(&app.session_id);
        app.session_default_subagent_backend = BackendId::TRAECLI;
        app.session_default_skills = vec!["agent-team".to_string()];
        // The reset clears every Builtin-tagged member of the `task` and
        // `agent_team` groups, so derive the `agent_team` members from the same
        // registry metadata instead of hardcoding a list that could drift.
        let agent_team_tools: Vec<String> = crate::ai::tools::tool_definitions_for_groups(&[
            crate::ai::tools::ToolGroup::AgentTeam,
        ])
        .into_iter()
        .map(|definition| definition.function.name)
        .filter(|name| {
            crate::ai::tools::registry::tool_metadata::tool_groups(name)
                .contains(&crate::ai::tools::ToolGroup::Builtin)
        })
        .collect();
        assert!(
            agent_team_tools.iter().any(|name| name == "manage_team")
                && agent_team_tools
                    .iter()
                    .any(|name| name == "run_agent_graph"),
            "agent_team group must expose the multi-agent orchestration tools, got {agent_team_tools:?}"
        );
        let mut seeded_tools = vec![
            "task_spawn".to_string(),
            "task_wait".to_string(),
            "knowledge_search".to_string(),
        ];
        seeded_tools.extend(agent_team_tools.iter().cloned());
        crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
            (app.session_id.clone(), 0),
            || {
                crate::ai::tools::enable_tools::set_explicit_enabled_tool_names(seeded_tools);
            },
        );

        assert!(try_handle_reset_command(&mut app, "/reset").unwrap());

        let remaining = crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
            (app.session_id.clone(), 0),
            crate::ai::tools::enable_tools::explicit_enabled_tool_names,
        );
        assert!(!remaining.iter().any(|name| name == "task_spawn"));
        assert!(!remaining.iter().any(|name| name == "task_wait"));
        for name in &agent_team_tools {
            assert!(
                !remaining
                    .iter()
                    .any(|remaining_name| remaining_name == name),
                "reset must drop agent_team tool {name} from explicit enables"
            );
        }
        assert!(remaining.iter().any(|name| name == "knowledge_search"));
        // Drop the fake owner's global explicit-enable entry so no state
        // outlives this test.
        crate::ai::tools::enable_tools::clear_explicitly_enabled_tools(&app.session_id);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn reset_command_clears_agent_team_only_continuation_and_accepts_both_prefixes() {
        for command in ["/reset", " \r\n :reset \t\r\n "] {
            let mut app = test_app();
            app.session_default_subagent_backend = BackendId::TRAECLI;
            app.session_default_subagent_model = Some("saved-worker-model".to_string());
            app.session_default_skills = vec!["AGENT-TEAM".to_string()];
            app.pending_skill_continuation = Some(crate::ai::types::PendingSkillContinuation {
                skill_names: vec!["Agent-Team".to_string()],
            });
            assert!(try_handle_reset_command(&mut app, command).unwrap());
            assert_eq!(
                app.session_default_subagent_backend,
                BackendId::from_session_config(None)
            );
            assert!(app.session_default_subagent_model.is_none());
            assert!(app.session_default_skills.is_empty());
            assert!(app.pending_skill_continuation.is_none());
        }
    }

    #[test]
    fn reset_command_rejects_arguments_and_trailing_prompt_without_changing_state() {
        let mut app = test_app();
        app.session_default_subagent_backend = BackendId::TRAECLI;
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
            assert_eq!(app.session_default_subagent_backend, BackendId::TRAECLI);
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
            assert_eq!(app.session_default_subagent_backend, BackendId::KERNEL);
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
        assert_eq!(app.session_default_subagent_backend, BackendId::TRAECLI);
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
        assert_eq!(app.session_default_subagent_backend, BackendId::TRAECLI);
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
        app.session_default_subagent_backend = BackendId::TRAECLI;
        app.session_default_subagent_model = Some("saved-model".to_string());
        app.session_default_skills = vec!["agent-team".to_string()];

        assert!(try_handle_use_command(&mut app, "/use\n任务").unwrap());
        assert_eq!(app.session_default_subagent_backend, BackendId::TRAECLI);
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
        assert_eq!(app.session_default_subagent_backend, BackendId::KERNEL);
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
        app.session_default_subagent_backend = BackendId::TRAECLI;
        app.session_default_subagent_model = Some("GPT-6.1-Sol".to_string());
        app.session_default_skills = vec!["agent-team".to_string(), "other-skill".to_string()];
        for snapshot in [
            app.snapshot_for_driver_context(),
            app.snapshot_for_detached_helper(),
        ] {
            assert_eq!(
                snapshot.session_default_subagent_backend,
                BackendId::TRAECLI
            );
            assert_eq!(
                snapshot.session_default_subagent_model,
                app.session_default_subagent_model
            );
            assert_eq!(snapshot.session_default_skills, app.session_default_skills);
        }
        let child = app.fork_for_subagent();
        assert_eq!(child.session_default_subagent_backend, BackendId::TRAECLI);
        assert_eq!(
            child.session_default_subagent_model,
            app.session_default_subagent_model
        );
        assert_eq!(child.session_default_skills, ["other-skill"]);
        assert_eq!(app.session_default_skills, ["agent-team", "other-skill"]);
    }
}
