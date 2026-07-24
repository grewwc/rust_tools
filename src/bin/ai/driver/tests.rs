use super::note_search::read_recent_history;
use super::{
    DispatchOutcomeTag, ProcessDispatchMeta, SCHED_COOLDOWN_EPOCHS_DEFAULT,
    SCHEDULER_DISPATCH_META, background_execute_limit, background_pop_limit,
    build_background_process_question, decode_background_process_task_goal,
    has_pending_foreground_process, maybe_auto_route_agent, one_shot_cli_mode,
    reset_scheduler_test_state, resolve_background_subagent_override,
    resolve_startup_session_choice, resolve_startup_session_choice_with_selector,
    should_preload_mcp, should_publish_subagent_task_result, should_suspend_session_on_sigint,
    should_resume_suspended_terminal_session, update_dispatch_meta,
};
use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
use crate::ai::cli::ParsedCli;
use crate::ai::history::{Message, SuspendedSessionStore, append_history_messages};
use crate::ai::skills::SkillManifest;
use crate::ai::tools::task_tools::{InheritOptions, OsTaskGoal, encode_os_task_goal};
use crate::ai::types::{AgentContext, App, AppConfig};
use aios_kernel::kernel::{EventId, ProcessState, WaitPolicy, WaitReason};
use aios_kernel::primitives::ResourceLimit;
use std::sync::{Arc, atomic::AtomicBool};
use std::{fs, path::PathBuf};

#[test]
fn background_dispatch_limits_scale_with_backlog() {
    assert_eq!(background_pop_limit(0), 4);
    assert_eq!(background_pop_limit(7), 6);
    assert_eq!(background_pop_limit(20), 8);

    assert_eq!(background_execute_limit(0), 4);
    assert_eq!(background_execute_limit(9), 5);
    assert_eq!(background_execute_limit(30), 6);
}

#[test]
fn scheduler_meta_opens_circuit_after_consecutive_failures() {
    reset_scheduler_test_state();
    let mut meta = ProcessDispatchMeta::default();
    meta = update_dispatch_meta(meta, DispatchOutcomeTag::Failed, 10);
    assert_eq!(meta.failure_streak, 1);
    assert_eq!(meta.cooldown_until_epoch, 0);

    meta = update_dispatch_meta(meta, DispatchOutcomeTag::Failed, 11);
    assert_eq!(meta.failure_streak, 2);
    assert_eq!(meta.cooldown_until_epoch, 0);

    meta = update_dispatch_meta(meta, DispatchOutcomeTag::Failed, 12);
    assert_eq!(meta.failure_streak, 3);
    assert_eq!(
        meta.cooldown_until_epoch,
        12 + SCHED_COOLDOWN_EPOCHS_DEFAULT
    );

    meta = update_dispatch_meta(meta, DispatchOutcomeTag::Advanced, 13);
    assert_eq!(meta.failure_streak, 0);
}

#[test]
fn subagent_task_result_stays_open_while_process_is_parked() {
    let waiting = ProcessState::Waiting {
        reason: WaitReason::Events {
            event_ids: vec![EventId::new(1)],
            policy: WaitPolicy::Any,
            timeout_tick: None,
        },
    };
    assert!(!should_publish_subagent_task_result(
        true,
        "",
        Some(&waiting)
    ));
    assert!(should_publish_subagent_task_result(
        true,
        "final answer",
        Some(&waiting)
    ));
    assert!(should_publish_subagent_task_result(
        false,
        "",
        Some(&waiting)
    ));
    assert!(should_publish_subagent_task_result(true, "", None));
}

#[test]
fn encoded_background_task_goal_rejects_corrupt_payload() {
    let encoded = encode_os_task_goal(&OsTaskGoal {
        task_id: "task_123".to_string(),
        result_channel_id: 7,
        completion_futex_addr: 9,
        description: "inspect".to_string(),
        prompt: "look around".to_string(),
        agent_name: "build".to_string(),
        model: "qwen3.7-max-alibaba".to_string(),
        is_model_auto_selected: false,
        auto_model_fallback: None,
        selection_explanation: "explicit override".to_string(),
        spawn_depth: 0,
    })
    .unwrap();
    assert!(
        decode_background_process_task_goal(&encoded)
            .unwrap()
            .is_some()
    );

    let corrupted = encoded.replacen('{', "", 1);
    let err = decode_background_process_task_goal(&corrupted).unwrap_err();
    assert!(err.contains("failed to decode"));
    assert!(err.contains("parent agent/model"));
}

#[test]
fn background_subagent_override_requires_known_enabled_agent() {
    let mut plan = primary_agent("plan", "Read-only planning agent");
    plan.mode = AgentMode::Subagent;
    plan.disabled = true;
    let build = primary_agent("build", "Default agent");

    let err = resolve_background_subagent_override(&[build.clone(), plan.clone()], Some("plan"))
        .unwrap_err();
    assert!(err.contains("disabled"));

    let err = resolve_background_subagent_override(&[build], Some("missing")).unwrap_err();
    assert!(err.contains("could not be found"));
}

#[test]
fn background_task_wakeup_prompt_prefers_mailbox_and_decoded_goal() {
    let encoded = encode_os_task_goal(&OsTaskGoal {
        task_id: "task_123".to_string(),
        result_channel_id: 7,
        completion_futex_addr: 9,
        description: "inspect".to_string(),
        prompt: "inspect codebase state".to_string(),
        agent_name: "build".to_string(),
        model: "qwen3.7-max-alibaba".to_string(),
        is_model_auto_selected: false,
        auto_model_fallback: None,
        selection_explanation: "explicit override".to_string(),
        spawn_depth: 0,
    })
    .unwrap();
    let mailbox = vec![
        "[task_wait PARKED] ...".to_string(),
        "[mailbox] task task_123 completed".to_string(),
    ];

    let question =
        build_background_process_question(42, &encoded, Some("inspect codebase state"), &mailbox);

    assert!(question.contains("[Process 42 Woke Up]"));
    assert!(question.contains("Original goal: inspect codebase state"));
    assert!(question.contains("[mailbox] task task_123 completed"));
    assert!(!question.contains("AIOS_SUBAGENT_TASK:"));
}

#[test]
fn background_task_without_mailbox_reuses_decoded_goal_prompt() {
    let question = build_background_process_question(
        7,
        "AIOS_SUBAGENT_TASK:{\"ignored\":true}",
        Some("search the repository"),
        &[],
    );

    assert_eq!(question, "search the repository");
}

#[test]
fn finalize_turn_quota_charges_turn_usage_once() {
    let app = test_app("build");
    let pid = {
        let mut os = app.os.lock().unwrap();
        os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None)
    };

    {
        let mut os = app.os.lock().unwrap();
        let mut lim = ResourceLimit::unlimited();
        lim.max_turns = 2;
        os.rlimit_set(pid, lim).unwrap();
        assert_eq!(os.rusage_get(pid).unwrap().turns, 0);

        let (terminate1, msg1) = super::finalize_turn_quota(os.as_mut(), pid);
        assert!(terminate1);
        assert_eq!(msg1, "Completed");
        assert_eq!(os.rusage_get(pid).unwrap().turns, 1);

        let (terminate2, msg2) = super::finalize_turn_quota(os.as_mut(), pid);
        assert!(terminate2);
        assert_eq!(msg2, "Completed");
        assert_eq!(os.rusage_get(pid).unwrap().turns, 2);

        let (terminate3, msg3) = super::finalize_turn_quota(os.as_mut(), pid);
        assert!(terminate3);
        assert!(msg3.contains("Resource limit exceeded"));
        assert_eq!(os.rusage_get(pid).unwrap().turns, 3);
    }
}

#[test]
fn terminate_and_cleanup_removes_scheduler_meta_entry() {
    reset_scheduler_test_state();
    let app = test_app("build");
    let pid = {
        let mut os = app.os.lock().unwrap();
        os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None)
    };
    {
        let mut map = SCHEDULER_DISPATCH_META.lock().unwrap();
        map.insert(pid, ProcessDispatchMeta::default());
        assert!(map.contains_key(&pid));
    }
    {
        let mut os = app.os.lock().unwrap();
        super::terminate_and_cleanup(os.as_mut(), pid, "Completed".to_string(), true);
    }
    let map = SCHEDULER_DISPATCH_META.lock().unwrap();
    assert!(!map.contains_key(&pid));
}

fn primary_agent(name: &str, description: &str) -> AgentManifest {
    AgentManifest {
        name: name.to_string(),
        description: description.to_string(),
        mode: AgentMode::Primary,
        model: None,
        temperature: None,
        max_steps: None,
        prompt: String::new(),
        system_prompt: None,
        tools: Vec::new(),
        tool_groups: Vec::new(),
        mcp_servers: Vec::new(),
        disable_mcp_tools: false,
        model_tier: Some(AgentModelTier::Heavy),
        disabled: false,
        hidden: false,
        color: None,
        source_path: None,
    }
}

fn test_app(current_agent: &str) -> App {
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
            agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/bin/ai/config/agent_route/agent_route_model.json"),
            skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/bin/ai/config/skill_match/skill_match_model.json"),
        },
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        active_persona: crate::ai::persona::default_persona(),
        client: reqwest::Client::new(),
        current_model: "test-model".to_string(),
        current_agent: current_agent.to_string(),
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
        agent_context: Some(AgentContext {
            tools: Vec::new(),
            mcp_servers: Default::default(),
            max_iterations: super::DEFAULT_MAX_ITERATIONS,
        }),
        last_skill_bias: None,
        os: super::new_local_kernel(),
        agent_reload_counter: None,
        observers: vec![Box::new(
            crate::ai::driver::thinking::ThinkingOrchestrator::new(),
        )],
        last_known_prompt_tokens: None,
        last_known_cached_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
        turn_reasoning_items: Default::default(),
        stale_patch_targets: Default::default(),
    }
}

fn test_startup_config(base_history_file: &std::path::Path) -> AppConfig {
    AppConfig {
        api_key: String::new(),
        base_history_file: base_history_file.to_path_buf(),
        history_file: base_history_file.to_path_buf(),
        endpoint: String::new(),
        vl_default_model: String::new(),
        history_max_chars: 12000,
        history_keep_last: 8,
        history_summary_max_chars: 4000,
        intent_model: None,
        agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/ai/config/agent_route/agent_route_model.json"),
        skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/bin/ai/config/skill_match/skill_match_model.json"),
    }
}

#[test]
fn one_shot_mode_still_preloads_mcp_before_turn() {
    let probe = super::McpConfigProbe {
        config_path: "/tmp/mcp.json".to_string(),
        exists: true,
        server_count: 1,
    };

    assert!(should_preload_mcp(true, &probe));
    assert!(should_preload_mcp(false, &probe));
}

#[test]
fn interactive_flag_disables_one_shot_cli_mode() {
    let mut cli = ParsedCli::default();
    cli.args = vec!["解释一下上次的笔记".to_string()];
    assert!(one_shot_cli_mode(&cli));

    cli.interactive = true;
    assert!(!one_shot_cli_mode(&cli));
}

#[test]
fn sigint_does_not_suspend_new_empty_session() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "rt_sigint_empty_session_{}",
        uuid::Uuid::new_v4().simple()
    ));
    unsafe {
        std::env::set_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR", root.join("suspended"));
        std::env::set_var("TERM_SESSION_ID", "term-empty");
    }

    let mut app = test_app("build");
    app.config.history_file = root.join("history.sqlite");
    app.session_id = "empty-session".to_string();

    assert!(!should_suspend_session_on_sigint(&app));
    super::suspend_session_on_sigint(&app);
    assert!(
        SuspendedSessionStore::new()
            .peek_entries_for_terminal_key("terminal:term-empty")
            .unwrap()
            .is_empty()
    );

    // 显式恢复的会话即使暂时没有用户消息，也应保留原有挂起语义。
    app.cli.session = Some(app.session_id.clone());
    assert!(should_suspend_session_on_sigint(&app));

    unsafe {
        std::env::remove_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR");
        std::env::remove_var("TERM_SESSION_ID");
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn resume_predicate_requires_clean_interactive_start() {
    let cli = ParsedCli::default();
    assert!(should_resume_suspended_terminal_session(&cli));

    let mut cli = ParsedCli::default();
    cli.args = vec!["继续".to_string()];
    assert!(!should_resume_suspended_terminal_session(&cli));

    let mut cli = ParsedCli::default();
    cli.session = Some(String::new());
    assert!(!should_resume_suspended_terminal_session(&cli));

    let mut cli = ParsedCli::default();
    cli.new_session = true;
    assert!(!should_resume_suspended_terminal_session(&cli));
}

#[test]
fn startup_choice_auto_resumes_terminal_bound_session() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "rt_startup_resume_{}",
        uuid::Uuid::new_v4().simple()
    ));
    let suspended_root = root.join("suspended");
    let persona_path = root.join("personas.json");
    let base_history = root.join("history.sqlite");
    let suspended_history = root.join("history.persona-reviewer.sqlite");

    unsafe {
        std::env::set_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR", &suspended_root);
        std::env::set_var("TERM_SESSION_ID", "term-456");
    }

    let persona_store = crate::ai::persona::PersonaStore::for_tests_with_path(persona_path);
    let reviewer = persona_store
        .create_persona("Reviewer", None, "You are a reviewer.")
        .unwrap();
    SuspendedSessionStore::new()
        .save_for_terminal_key(
            "terminal:term-456",
            "sess-123",
            &suspended_history,
            &reviewer.id,
            "test-model",
        )
        .unwrap();

    let choice = resolve_startup_session_choice(
        &ParsedCli::default(),
        &test_startup_config(&base_history),
        &persona_store,
        crate::ai::persona::default_persona(),
    )
    .unwrap();

    assert_eq!(choice.session_id, "sess-123");
    assert_eq!(choice.history_file, suspended_history);
    assert_eq!(choice.active_persona.id, reviewer.id);
    assert_eq!(choice.model.as_deref(), Some("test-model"));
    assert!(choice.startup_notice.is_some());
    assert!(
        SuspendedSessionStore::new()
            .take_for_terminal_key("terminal:term-456")
            .unwrap()
            .is_none()
    );

    unsafe {
        std::env::remove_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR");
        std::env::remove_var("TERM_SESSION_ID");
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn startup_choice_skips_auto_resume_when_prompt_args_exist() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "rt_startup_resume_skip_{}",
        uuid::Uuid::new_v4().simple()
    ));
    let suspended_root = root.join("suspended");
    let base_history = root.join("history.sqlite");

    unsafe {
        std::env::set_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR", &suspended_root);
        std::env::set_var("TERM_SESSION_ID", "term-789");
    }

    let persona_store =
        crate::ai::persona::PersonaStore::for_tests_with_path(root.join("personas.json"));
    let suspended_history = root.join("history.persona-default.sqlite");
    SuspendedSessionStore::new()
        .save_for_terminal_key(
            "terminal:term-789",
            "sess-keep",
            &suspended_history,
            "default",
            "test-model",
        )
        .unwrap();

    let mut cli = ParsedCli::default();
    cli.args = vec!["继续这个问题".to_string()];

    let choice = resolve_startup_session_choice(
        &cli,
        &test_startup_config(&base_history),
        &persona_store,
        crate::ai::persona::default_persona(),
    )
    .unwrap();

    assert_ne!(choice.session_id, "sess-keep");
    assert_eq!(
        choice.history_file,
        crate::ai::persona::history_file_for_persona(&base_history, "default")
    );
    assert!(
        SuspendedSessionStore::new()
            .take_for_terminal_key("terminal:term-789")
            .unwrap()
            .is_some()
    );

    unsafe {
        std::env::remove_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR");
        std::env::remove_var("TERM_SESSION_ID");
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn startup_choice_skips_auto_resume_when_new_session_requested() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "rt_startup_new_session_{}",
        uuid::Uuid::new_v4().simple()
    ));
    let suspended_root = root.join("suspended");
    let base_history = root.join("history.sqlite");

    unsafe {
        std::env::set_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR", &suspended_root);
        std::env::set_var("TERM_SESSION_ID", "term-new");
    }

    let persona_store =
        crate::ai::persona::PersonaStore::for_tests_with_path(root.join("personas.json"));
    let suspended_history = root.join("history.persona-default.sqlite");
    SuspendedSessionStore::new()
        .save_for_terminal_key(
            "terminal:term-new",
            "sess-keep",
            &suspended_history,
            "default",
            "test-model",
        )
        .unwrap();

    let mut cli = ParsedCli::default();
    cli.new_session = true;

    let choice = resolve_startup_session_choice(
        &cli,
        &test_startup_config(&base_history),
        &persona_store,
        crate::ai::persona::default_persona(),
    )
    .unwrap();

    assert_ne!(choice.session_id, "sess-keep");
    assert_eq!(
        choice.history_file,
        crate::ai::persona::history_file_for_persona(&base_history, "default")
    );
    assert_eq!(
        SuspendedSessionStore::new()
            .peek_entries_for_terminal_key("terminal:term-new")
            .unwrap()
            .len(),
        1
    );

    unsafe {
        std::env::remove_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR");
        std::env::remove_var("TERM_SESSION_ID");
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn startup_choice_can_select_specific_suspended_session_from_multiple() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "rt_startup_resume_select_{}",
        uuid::Uuid::new_v4().simple()
    ));
    let suspended_root = root.join("suspended");
    let base_history = root.join("history.sqlite");
    let history_a = root.join("history.persona-default.sqlite");
    let history_b = root.join("history.persona-reviewer.sqlite");

    unsafe {
        std::env::set_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR", &suspended_root);
        std::env::set_var("TERM_SESSION_ID", "term-select");
    }

    let persona_store =
        crate::ai::persona::PersonaStore::for_tests_with_path(root.join("personas.json"));
    SuspendedSessionStore::new()
        .save_for_terminal_key(
            "terminal:term-select",
            "sess-1",
            &history_a,
            "default",
            "model-a",
        )
        .unwrap();
    SuspendedSessionStore::new()
        .save_for_terminal_key(
            "terminal:term-select",
            "sess-2",
            &history_b,
            "default",
            "model-b",
        )
        .unwrap();

    let choice = resolve_startup_session_choice_with_selector(
        &ParsedCli::default(),
        &test_startup_config(&base_history),
        &persona_store,
        crate::ai::persona::default_persona(),
        |previews| {
            assert_eq!(previews.len(), 2);
            assert_eq!(previews[0].entry.session_id, "sess-2");
            assert_eq!(previews[1].entry.session_id, "sess-1");
            Ok(Some(1))
        },
    )
    .unwrap();

    assert_eq!(choice.session_id, "sess-1");
    assert_eq!(choice.history_file, history_a);
    let remaining = SuspendedSessionStore::new()
        .peek_entries_for_terminal_key("terminal:term-select")
        .unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].session_id, "sess-2");

    unsafe {
        std::env::remove_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR");
        std::env::remove_var("TERM_SESSION_ID");
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn startup_choice_can_start_new_without_consuming_suspended_stack() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "rt_startup_resume_skip_stack_{}",
        uuid::Uuid::new_v4().simple()
    ));
    let suspended_root = root.join("suspended");
    let base_history = root.join("history.sqlite");
    let history_a = root.join("history.persona-default.sqlite");
    let history_b = root.join("history.persona-reviewer.sqlite");

    unsafe {
        std::env::set_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR", &suspended_root);
        std::env::set_var("TERM_SESSION_ID", "term-stack");
    }

    let persona_store =
        crate::ai::persona::PersonaStore::for_tests_with_path(root.join("personas.json"));
    SuspendedSessionStore::new()
        .save_for_terminal_key(
            "terminal:term-stack",
            "sess-1",
            &history_a,
            "default",
            "model-a",
        )
        .unwrap();
    SuspendedSessionStore::new()
        .save_for_terminal_key(
            "terminal:term-stack",
            "sess-2",
            &history_b,
            "default",
            "model-b",
        )
        .unwrap();

    let choice = resolve_startup_session_choice_with_selector(
        &ParsedCli::default(),
        &test_startup_config(&base_history),
        &persona_store,
        crate::ai::persona::default_persona(),
        |_previews| Ok(None),
    )
    .unwrap();

    assert_ne!(choice.session_id, "sess-1");
    assert_ne!(choice.session_id, "sess-2");
    assert_eq!(
        choice.history_file,
        crate::ai::persona::history_file_for_persona(&base_history, "default")
    );
    assert!(
        choice
            .startup_notice
            .as_deref()
            .unwrap_or_default()
            .contains("已跳过")
    );
    assert_eq!(
        SuspendedSessionStore::new()
            .peek_entries_for_terminal_key("terminal:term-stack")
            .unwrap()
            .len(),
        2
    );

    unsafe {
        std::env::remove_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR");
        std::env::remove_var("TERM_SESSION_ID");
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn startup_choice_rejects_resume_and_session_together() {
    let base_history = PathBuf::from("/tmp/history.sqlite");
    let persona_store =
        crate::ai::persona::PersonaStore::for_tests_with_path(std::env::temp_dir().join(format!(
            "rt_personas_conflict_{}.json",
            uuid::Uuid::new_v4().simple()
        )));
    let mut cli = ParsedCli::default();
    cli.resume = true;
    cli.session = Some(String::new());

    let err = resolve_startup_session_choice(
        &cli,
        &test_startup_config(&base_history),
        &persona_store,
        crate::ai::persona::default_persona(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("--resume"));
}

#[test]
fn startup_choice_rejects_resume_and_clear_together() {
    let base_history = PathBuf::from("/tmp/history.sqlite");
    let persona_store =
        crate::ai::persona::PersonaStore::for_tests_with_path(std::env::temp_dir().join(format!(
            "rt_personas_clear_conflict_{}.json",
            uuid::Uuid::new_v4().simple()
        )));
    let mut cli = ParsedCli::default();
    cli.resume = true;
    cli.clear = true;

    let err = resolve_startup_session_choice(
        &cli,
        &test_startup_config(&base_history),
        &persona_store,
        crate::ai::persona::default_persona(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("--resume"));
    assert!(err.to_string().contains("--clear"));
}

#[test]
fn startup_choice_rejects_resume_and_new_session_together() {
    let base_history = PathBuf::from("/tmp/history.sqlite");
    let persona_store =
        crate::ai::persona::PersonaStore::for_tests_with_path(std::env::temp_dir().join(format!(
            "rt_personas_new_conflict_{}.json",
            uuid::Uuid::new_v4().simple()
        )));
    let mut cli = ParsedCli::default();
    cli.resume = true;
    cli.new_session = true;

    let err = resolve_startup_session_choice(
        &cli,
        &test_startup_config(&base_history),
        &persona_store,
        crate::ai::persona::default_persona(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("--resume"));
    assert!(err.to_string().contains("--new-session"));
}

#[test]
fn auto_route_switches_to_build_for_code_question_when_enabled() {
    // auto-routing 默认禁用（见 auto_agent_routing_enabled）。这里通过临时
    // config 打开 heuristic 策略，验证一个明显匹配 build 的代码问题会
    // 正确选择 build agent。
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());

    let cfg_path = std::env::temp_dir().join(format!(
        "rt_auto_route_{}.configw",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(
        &cfg_path,
        "ai.agents.auto_route.enable = true\nai.agents.auto_route.strategy = heuristic\n",
    )
    .unwrap();
    let old_cfg = std::env::var_os("CONFIGW_PATH");
    unsafe { std::env::set_var("CONFIGW_PATH", &cfg_path) };
    crate::commonw::configw::refresh();

    let build = primary_agent(
        "build",
        "Build agent for code development, compilation, debugging, bug fixing, and feature implementation",
    );
    let plan = primary_agent(
        "plan",
        "Planning agent for analysis, code review, strategy, and architecture",
    );
    // 起始 agent 必须不同于目标，否则 choose_semantic_route 会因
    // best.agent.name == current_agent 而直接返回 None。
    let mut app = test_app("plan");

    maybe_auto_route_agent(
        &mut app,
        &[build.clone(), plan.clone()],
        "fix the bug and debug the compilation error",
    );

    // 恢复 config，避免污染其它测试。
    match old_cfg {
        Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
        None => unsafe { std::env::remove_var("CONFIGW_PATH") },
    }
    crate::commonw::configw::refresh();
    let _ = std::fs::remove_file(&cfg_path);

    assert_eq!(app.current_agent, "build");
    assert_eq!(
        app.current_agent_manifest
            .as_ref()
            .map(|agent| agent.name.as_str()),
        Some("build")
    );
}

#[test]
fn read_recent_history_sqlite_preserves_previous_ordering() {
    let path = std::env::temp_dir().join(format!(
        "rt_recent_history_{}_{}.sqlite",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));

    let messages = (1..=12)
        .map(|idx| Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(format!("m{idx}")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        })
        .collect::<Vec<_>>();
    append_history_messages(path.as_path(), &messages).unwrap();

    let mut app = test_app("build");
    app.session_history_file = path.clone();

    let recent = read_recent_history(&app)
        .into_iter()
        .filter_map(|msg| msg.content.as_str().map(|s| s.to_string()))
        .collect::<Vec<_>>();

    assert_eq!(
        recent,
        vec![
            "m12", "m11", "m10", "m9", "m8", "m7", "m6", "m5", "m4", "m3"
        ]
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", path.display()));
}

#[test]
fn pending_foreground_process_blocks_new_prompt() {
    let app = test_app("build");

    {
        let mut os = app.os.lock().unwrap();
        let root = os.begin_foreground("foreground".to_string(), "goal".to_string(), 10, 8, None);
        os.wait_on_events(vec![EventId::new(1)], WaitPolicy::All, None)
            .unwrap();
        assert!(matches!(
            os.get_process(root).map(|proc| &proc.state),
            Some(ProcessState::Waiting { .. })
        ));
    }

    assert!(has_pending_foreground_process(&app));
}

#[test]
fn terminated_foreground_process_does_not_block_new_prompt() {
    let app = test_app("build");

    {
        let mut os = app.os.lock().unwrap();
        let root = os.begin_foreground("foreground".to_string(), "goal".to_string(), 10, 8, None);
        os.terminate_current("done".to_string());
        assert!(matches!(
            os.get_process(root).map(|proc| &proc.state),
            Some(ProcessState::Terminated)
        ));
    }

    assert!(!has_pending_foreground_process(&app));
}

fn sample_skill(name: &str) -> SkillManifest {
    serde_json::from_value(serde_json::json!({
        "name": name,
        "description": "sample"
    }))
    .unwrap()
}

#[test]
fn background_task_inherit_history_uses_parent_history_file() {
    let original = PathBuf::from("/tmp/session.sqlite");
    let process = PathBuf::from("/tmp/session.proc-42.sqlite");
    let skills = Arc::new(vec![sample_skill("s1")]);

    let (effective_history, effective_skills) = super::resolve_background_subagent_context(
        process,
        original.as_path(),
        &skills,
        Some("task_1"),
        InheritOptions {
            history: true,
            memory: false,
            cwd: true,
            skills: true,
        },
    );

    assert_eq!(effective_history, original);
    assert_eq!(effective_skills.len(), 1);
}

#[test]
fn background_task_disable_skills_uses_empty_skill_set() {
    let original = PathBuf::from("/tmp/session.sqlite");
    let process = PathBuf::from("/tmp/session.proc-43.sqlite");
    let skills = Arc::new(vec![sample_skill("s1")]);

    let (effective_history, effective_skills) = super::resolve_background_subagent_context(
        process.clone(),
        original.as_path(),
        &skills,
        Some("task_2"),
        InheritOptions {
            history: false,
            memory: false,
            cwd: true,
            skills: false,
        },
    );

    assert_eq!(effective_history, process);
    assert!(effective_skills.is_empty());
}

#[test]
fn non_task_background_process_keeps_process_history_and_skills() {
    let original = PathBuf::from("/tmp/session.sqlite");
    let process = PathBuf::from("/tmp/session.proc-99.sqlite");
    let skills = Arc::new(vec![sample_skill("s1")]);

    let (effective_history, effective_skills) = super::resolve_background_subagent_context(
        process.clone(),
        original.as_path(),
        &skills,
        None,
        InheritOptions::default(),
    );

    assert_eq!(effective_history, process);
    assert_eq!(effective_skills.len(), 1);
}
