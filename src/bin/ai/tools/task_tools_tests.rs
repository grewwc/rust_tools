use super::{
    AsyncTaskEntry, InheritOptions, OsTaskGoal, OutstandingTaskSnapshot, PreparedSubagentTask,
    SUBAGENT_BLOCKED_PARENT_GUIDANCE, SUBAGENT_PARENT_SUMMARY_REMINDER,
    SUBAGENT_WALL_CLOCK_TIMEOUT, SelectedSubagent, StoredTaskResult, SubagentBackend,
    TASK_REGISTRY, TASK_RETRY_REGISTRY, TASK_WAIT_NOOP_COUNTS, WaitManySource,
    append_current_process_cancel_source, build_outstanding_task_anchor,
    build_selection_explanation, bump_task_wait_noop_count, capped_subagent_manifest,
    discard_tasks_for_session, encode_os_task_goal, epoll_wait_many, epoll_wait_many_channels,
    execute_task_cancel, execute_task_integrate, execute_task_retry, execute_task_spawn,
    execute_task_status, execute_task_wait, expire_task_wait_states_for_test,
    format_task_result_with_id, insert_task_entry_for_test, is_encoded_task_goal,
    is_retryable_task_status, parse_subagent_handoff, parse_task_wait_options,
    prepare_subagent_task, reap_timed_out_subagents, record_subagent_progress_update,
    register_retry_spec, remove_task_entry, render_outstanding_task_anchor,
    reset_task_wait_noop_counts_for_test, resolve_side_note_history_file, select_subagent,
    set_task_reasoning_effort, task_reasoning_effort, task_wait_key,
    task_wait_state_count_for_test, validate_subagent_response, wait_sources_for_channel_and_futex,
    wake_expired_task_waits, with_task_entry_by_pid, wrap_subagent_prompt,
};
use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};

#[test]
fn inherit_options_accepts_plus_and_slash_separators() {
    // Regression: descriptions used to render the default as prose ("cwd+skills"), and a
    // model copying that verbatim hit a hard parse error under comma-only splitting.
    let canonical = InheritOptions::from_value(&serde_json::json!("cwd,skills")).unwrap();
    assert_eq!(canonical, InheritOptions::default());
    assert_eq!(
        InheritOptions::from_value(&serde_json::json!("cwd+skills")).unwrap(),
        canonical
    );
    assert_eq!(
        InheritOptions::from_value(&serde_json::json!("cwd/skills")).unwrap(),
        canonical
    );
    assert_eq!(
        InheritOptions::from_value(&serde_json::json!("history + memory/cwd")).unwrap(),
        InheritOptions::from_value(&serde_json::json!("history,memory,cwd")).unwrap()
    );
    // Unknown tokens are still rejected after splitting on the lenient separators.
    assert!(InheritOptions::from_value(&serde_json::json!("cwd+telepathy")).is_err());
}
use crate::ai::cli::ParsedCli;
use crate::ai::driver::runtime_ctx::{
    DRIVER_CTX, DriverContext, SubagentProgressEvent, SubagentProgressEventKind,
    SubagentProgressSnapshot,
};
use crate::ai::mcp::McpClient;
use crate::ai::tools::registry::common::tool_history_policy;
use crate::ai::types::{App, AppConfig};
use aios_kernel::{
    kernel::{EventId, KernelInternal, Syscall, WaitPolicy},
    local::LocalOS,
    primitives::{ChannelId, FutexAddr, FutexOps, IpcOps},
};
use std::sync::{Arc, atomic::AtomicBool};
use std::time::{Duration, Instant};
/// RAII guard for the process-global kernel handle (`os_tools::GLOBAL_OS`) used by the task
/// tools. A task_tools test installs its own kernel so `execute_task_*` calls route through it;
/// on the happy path the tests used to reset it manually, but a panicking test skipped that reset
/// and left a stale kernel installed process-wide, hijacking `FileStore` reads of unrelated tests
/// (their `try_vfs_*` helpers prefer a set kernel over bare `std::fs`). Restoring on drop covers
/// panics too, because test panics unwind and drop locals.
struct OsToolsGlobalGuard(Option<aios_kernel::kernel::SharedKernel>);

fn init_os_tools_guard(os: aios_kernel::kernel::SharedKernel) -> OsToolsGlobalGuard {
    let mut g = crate::ai::tools::os_tools::GLOBAL_OS
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let previous = std::mem::replace(&mut *g, Some(os));
    OsToolsGlobalGuard(previous)
}

impl Drop for OsToolsGlobalGuard {
    fn drop(&mut self) {
        if let Ok(mut g) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
            *g = self.0.take();
        }
    }
}

fn manifest(name: &str, description: &str, mode: AgentMode) -> AgentManifest {
    AgentManifest {
        name: name.to_string(),
        description: description.to_string(),
        mode,
        model: None,
        temperature: None,
        max_steps: None,
        prompt: String::new(),
        system_prompt: None,
        tools: Vec::new(),
        tool_groups: Vec::new(),
        mcp_servers: Vec::new(),
        disable_mcp_tools: false,
        audit_checkpoints: false,
        model_tier: Some(AgentModelTier::Standard),
        disabled: false,
        hidden: false,
        auto_select: true,
        pure: false,
        color: None,
        source_path: None,
    }
}

fn prepared_retry_task() -> PreparedSubagentTask {
    PreparedSubagentTask {
        description: "retry test".to_string(),
        prompt: "retry the same task".to_string(),
        response_schema: None,
        agent_name: "build".to_string(),
        model: "qwen3.7-max".to_string(),
        reasoning_effort: None,
        is_model_auto_selected: false,
        auto_model_fallback: None,
        selection_explanation: "test fixture".to_string(),
        inherit: InheritOptions::default(),
        backend: SubagentBackend::Kernel,
        show_backend_launch_tag: false,
        backend_model: None,
        backend_continue_from: None,
        image_files: Vec::new(),
    }
}

fn progress_snapshot(sequence: u64, elapsed_secs: u64, phase: &str) -> SubagentProgressSnapshot {
    SubagentProgressSnapshot {
        phase: phase.to_string(),
        checkpoint_summary: None,
        checkpoint_excerpt: None,
        checkpoint_revision: 0,
        elapsed: Duration::from_secs(elapsed_secs),
        stale_for: Duration::ZERO,
        checkpoint_due: false,
        sequence,
        timeline: vec![SubagentProgressEvent {
            sequence,
            kind: SubagentProgressEventKind::Phase,
            elapsed: Duration::from_secs(elapsed_secs),
            summary: phase.to_string(),
        }],
    }
}

#[test]
fn concurrent_progress_persistence_keeps_newest_valid_snapshot() {
    let root = std::env::temp_dir().join(format!(
        "task-progress-race-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let path = root.join("task.json");
    let newer_path = path.clone();
    let older_path = path.clone();
    let (newer_done_tx, newer_done_rx) = std::sync::mpsc::channel();

    let newer_writer = std::thread::spawn(move || {
        super::persist_subagent_progress_snapshot_at(
            &newer_path,
            "task",
            &progress_snapshot(2, 20, "newer"),
        )
        .unwrap();
        newer_done_tx.send(()).unwrap();
    });
    let older_writer = std::thread::spawn(move || {
        newer_done_rx.recv().unwrap();
        super::persist_subagent_progress_snapshot_at(
            &older_path,
            "task",
            &progress_snapshot(1, 10, "older"),
        )
        .unwrap();
    });
    newer_writer.join().unwrap();
    older_writer.join().unwrap();

    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(persisted["sequence"], 2);
    assert_eq!(persisted["phase"], "newer");
    let entries = std::fs::read_dir(&root).unwrap().count();
    assert_eq!(entries, 1, "临时快照文件不应残留");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn task_wait_defaults_are_short_and_resume_on_first_result() {
    let (timeout_secs, wait_policy) =
        parse_task_wait_options(&serde_json::json!({}), false).unwrap();
    let (ordinary_timeout_secs, _) =
        parse_task_wait_options(&serde_json::json!({ "timeout_secs": 900 }), false).unwrap();
    let (agent_team_timeout_secs, _) =
        parse_task_wait_options(&serde_json::json!({ "timeout_secs": 900 }), true).unwrap();

    assert_eq!(timeout_secs, 30);
    assert_eq!(wait_policy, WaitPolicy::Any);
    assert_eq!(ordinary_timeout_secs, 60);
    assert_eq!(agent_team_timeout_secs, 300);
}

#[test]
fn subagent_prompt_preserves_the_evidence_bar() {
    let prompt = wrap_subagent_prompt("review change", "inspect the diff", None);

    assert!(prompt.contains("Reuse observed evidence"));
    assert!(prompt.contains("prefer one targeted broad call"));
    assert!(prompt.contains("Ground factual claims in observed evidence"));
    assert!(prompt.contains("check likely counter-evidence"));
    assert!(prompt.contains("return a concise partial result"));
    assert!(prompt.contains("unresolved hypotheses, missing evidence"));
    assert!(prompt.contains("<subagent_handoff>"));
    assert!(prompt.contains("authorization_required|permission_required|user_input_required"));
    // Without an explicit schema the default contract still asks for an
    // auditable shape, guidance-only (no JSON enforcement).
    assert!(prompt.contains("Default response contract"));
    assert!(prompt.contains("**Evidence**"));
    assert!(prompt.contains("**Unresolved**"));
    assert!(!prompt.contains("Return exactly one JSON value"));
    assert!(!prompt.contains("Do not wait for perfect certainty"));
}

#[test]
fn subagent_response_schema_is_prompted_and_validated() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "summary": { "type": "string" }
        },
        "required": ["summary"],
        "additionalProperties": false
    });
    let prompt = wrap_subagent_prompt("summarize", "inspect the evidence", Some(&schema));

    assert!(prompt.contains("Return exactly one JSON value"));
    assert!(prompt.contains("<response_schema>"));
    assert!(prompt.contains("This exceptional control response overrides"));
    assert!(!prompt.contains("Default response contract"));
    assert!(validate_subagent_response(Some(&schema), r#"{"summary":"done"}"#).is_ok());
    assert!(
        validate_subagent_response(Some(&schema), r#"{"other":"done"}"#)
            .unwrap_err()
            .contains("did not match response_schema")
    );
    assert!(
        validate_subagent_response(Some(&schema), "not json")
            .unwrap_err()
            .contains("not valid JSON")
    );
}

#[test]
fn subagent_handoff_is_strict_and_carries_parent_action() {
    let handoff = parse_subagent_handoff(
        r#"<subagent_handoff>{"kind":"authorization_required","reason":"token expired","suggested_action":"authenticate in the foreground"}</subagent_handoff>"#,
    )
    .unwrap()
    .expect("handoff should parse");

    assert_eq!(handoff.kind.as_str(), "authorization_required");
    assert_eq!(handoff.reason, "token expired");
    assert_eq!(
        handoff.suggested_action.as_deref(),
        Some("authenticate in the foreground")
    );
    assert!(is_retryable_task_status("blocked"));
    assert!(
        parse_subagent_handoff(
            r#"<subagent_handoff>{"kind":"permission_required","reason":""}</subagent_handoff>"#
        )
        .unwrap_err()
        .contains("must not be empty")
    );
    assert!(
        parse_subagent_handoff(
            r#"<subagent_handoff>{"kind":"permission_required","reason":"denied"}"#
        )
        .unwrap_err()
        .contains("closing tag")
    );
    assert!(
        parse_subagent_handoff("ordinary final response")
            .unwrap()
            .is_none()
    );
}

#[test]
fn task_spawn_rejects_unbounded_fan_out() {
    let tasks = vec![serde_json::json!({}); super::MAX_SUBAGENT_SPAWN_BATCH_SIZE + 1];
    let error = execute_task_spawn(&serde_json::json!({ "tasks": tasks })).unwrap_err();

    assert_eq!(
        error,
        format!(
            "task_spawn accepts at most {} tasks per call",
            super::MAX_SUBAGENT_SPAWN_BATCH_SIZE
        )
    );
}

#[test]
fn task_spawn_requires_a_non_empty_tasks_array_and_preflights_every_item() {
    let legacy = execute_task_spawn(&serde_json::json!({
        "description": "legacy single form",
        "prompt": "must not run",
    }))
    .unwrap_err();
    assert_eq!(legacy, "task_spawn requires a 'tasks' array");

    let empty = execute_task_spawn(&serde_json::json!({ "tasks": [] })).unwrap_err();
    assert_eq!(empty, "task_spawn requires at least one task");

    let incomplete = execute_task_spawn(&serde_json::json!({
        "tasks": [{ "description": "first" }],
    }))
    .unwrap_err();
    assert!(
        incomplete.contains("task_spawn tasks[0]: Missing 'prompt' parameter"),
        "err: {incomplete}"
    );
}

pub(super) fn test_app_with_model(current_model: String) -> App {
    App {
        cli: ParsedCli::default(),
        scoped_preflight_required: Vec::new(),
        hooks: Default::default(),
        config: AppConfig {
            api_key: String::new(),
            base_history_file: std::path::PathBuf::new(),
            history_file: std::path::PathBuf::new(),
            endpoint: String::new(),
            vl_default_model: String::new(),
            history_max_chars: 12000,
            history_keep_last: 8,
            history_summary_max_chars: 4000,
            intent_model: None,
        },
        session_id: String::new(),
        session_history_file: std::path::PathBuf::new(),
        active_persona: crate::ai::persona::default_persona(),
        client: reqwest::Client::new(),
        current_model,
        current_agent: "build".to_string(),
        session_default_subagent_backend: Default::default(),
        session_default_subagent_model: None,
        session_default_skills: Vec::new(),
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
        turn_tool_errors: Default::default(),
        stale_patch_targets: Default::default(),
        tool_middlewares: Vec::new(),
        llm_middlewares: Vec::new(),
    }
}

#[test]
fn auto_select_prefers_navigator_for_codebase_investigation() {
    let mut build = manifest("build", "Main build agent", AgentMode::Primary);
    build.model_tier = Some(AgentModelTier::Heavy);
    let mut navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    navigator.model_tier = Some(AgentModelTier::Light);
    let review = manifest("review", "Read-only review agent", AgentMode::Subagent);

    let all_agents = vec![build, navigator, review];

    let selected = select_subagent(
        &all_agents,
        None,
        "Locate routing logic",
        "Find where automatic agent routing happens and summarize the files involved.",
    )
    .unwrap();

    assert_eq!(selected.agent.name, "navigator");
    assert!(selected.auto_selected);
}

#[test]
fn response_schema_is_rejected_for_evidence_gated_review_agents() {
    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let audit = manifest("audit", "Deep code review agent", AgentMode::Subagent);
    let ctx = DriverContext::new(
        test_app_with_model(current_model),
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![audit]),
    );
    let args = serde_json::json!({
        "description": "Review the current changes",
        "prompt": "Review the current changes.",
        "agent": "audit",
        "response_schema": {"type": "object"},
    });

    let error = match DRIVER_CTX.sync_scope(ctx, || prepare_subagent_task(&args)) {
        Ok(_) => panic!("the report envelope and response_schema cannot be combined"),
        Err(error) => error,
    };

    assert!(error.contains("<audit_report>"), "{error}");
    assert!(error.contains("response_schema"), "{error}");
}

#[test]
fn prepare_subagent_task_auto_selects_model_and_fallback() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let cfg_path = std::env::temp_dir().join(format!(
        "rt_subagent_model_{}.configw",
        uuid::Uuid::new_v4().simple()
    ));
    // 默认已改为继承主 agent 模型；显式关闭继承，强制走自动选择路径。
    std::fs::write(&cfg_path, "ai.subagent.model_inherit = false\n").unwrap();
    let old_cfg = std::env::var_os("CONFIGW_PATH");
    unsafe { std::env::set_var("CONFIGW_PATH", &cfg_path) };
    crate::commonw::configw::refresh();

    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let ctx = DriverContext::new(
        test_app_with_model(current_model),
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![navigator]),
    );
    let args = serde_json::json!({
        "description": "Locate task tool",
        "prompt": "Find where task spawning is implemented.",
        "agent": "navigator"
    });

    let prepared = DRIVER_CTX.sync_scope(ctx, || prepare_subagent_task(&args));

    match old_cfg {
        Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
        None => unsafe { std::env::remove_var("CONFIGW_PATH") },
    }
    crate::commonw::configw::refresh();
    let _ = std::fs::remove_file(&cfg_path);

    let prepared = prepared.unwrap();

    assert!(prepared.is_model_auto_selected);
    assert!(prepared.auto_model_fallback.is_some());
    assert!(
        prepared
            .selection_explanation
            .contains("model_reason=auto-selected for agent_tier=")
    );
    assert!(prepared.prompt.contains("Runtime constraints:"));
    assert!(
        prepared
            .prompt
            .contains("avoid equivalent read/search/list/command variants")
    );
    assert!(
        prepared
            .prompt
            .contains("<parent_task_prompt>\nFind where task spawning is implemented.")
    );
    assert!(prepared.prompt.contains("</parent_task_prompt>"));
}

#[test]
fn prepare_subagent_task_inherits_parent_model_by_default() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let cfg_path = std::env::temp_dir().join(format!(
        "rt_subagent_inherit_{}.configw",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(&cfg_path, "").unwrap();
    let old_cfg = std::env::var_os("CONFIGW_PATH");
    unsafe { std::env::set_var("CONFIGW_PATH", &cfg_path) };
    crate::commonw::configw::refresh();

    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let mut app = test_app_with_model(current_model.clone());
    app.current_agent = "sharp".to_string();
    app.cli.reasoning_effort_override = Some(None);
    let ctx = DriverContext::new(
        app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![navigator]),
    );
    let args = serde_json::json!({
        "description": "Locate task tool",
        "prompt": "Find where task spawning is implemented.",
        "agent": "navigator"
    });

    let prepared = DRIVER_CTX.sync_scope(ctx, || prepare_subagent_task(&args));

    match old_cfg {
        Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
        None => unsafe { std::env::remove_var("CONFIGW_PATH") },
    }
    crate::commonw::configw::refresh();
    let _ = std::fs::remove_file(&cfg_path);

    let prepared = prepared.unwrap();
    assert_eq!(prepared.model, current_model);
    let expected_effort = crate::ai::models::default_reasoning_effort(&current_model)
        .map(|effort| effort.as_str())
        .unwrap_or("server default");
    assert_eq!(prepared.reasoning_effort.as_deref(), Some(expected_effort));
    assert!(!prepared.is_model_auto_selected);
    assert!(prepared.auto_model_fallback.is_none());
    assert!(
        prepared
            .selection_explanation
            .contains("model_reason=inherited parent agent current model")
    );
}

#[test]
fn prepare_subagent_task_uses_session_default_backend() {
    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let mut app = test_app_with_model(current_model);
    app.session_default_subagent_backend = SubagentBackend::TraeCli;
    app.session_default_subagent_model = Some("GPT-6.1-Sol".to_string());
    let ctx = DriverContext::new(
        app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![navigator]),
    );
    let args = serde_json::json!({
        "description": "Locate task tool",
        "prompt": "Find where task spawning is implemented.",
        "agent": "navigator"
    });

    let prepared = DRIVER_CTX
        .sync_scope(ctx, || prepare_subagent_task(&args))
        .unwrap();

    assert_eq!(prepared.backend, SubagentBackend::TraeCli);
    assert_eq!(prepared.backend_model.as_deref(), Some("GPT-6.1-Sol"));
    assert!(prepared.reasoning_effort.is_none());
    assert!(prepared.show_backend_launch_tag);
}

#[test]
fn traecli_model_attribution_agrees_across_spawn_status_and_delivery() {
    const CHILD_ENV: &str = "RUST_TOOLS_MODEL_ATTRIBUTION_TEST_CHILD";
    if std::env::var_os(CHILD_ENV).is_none() {
        // Default-model resolution requires process-global environment changes.
        // Isolate them from concurrent fake-backend subprocess tests.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ai::tools::task_tools::tests::traecli_model_attribution_agrees_across_spawn_status_and_delivery",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated attribution test failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
            "the isolated attribution test must execute its assertions"
        );
        return;
    }
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    struct RestoreEnv([(&'static str, Option<std::ffi::OsString>); 2]);
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            for (name, value) in &self.0 {
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }
    let _restore_env = RestoreEnv([
        ("TRAECLI_BIN", std::env::var_os("TRAECLI_BIN")),
        ("TRAECLI_MODEL", std::env::var_os("TRAECLI_MODEL")),
    ]);
    let backend_model = "backend-test-model";
    unsafe {
        // Admission only checks this file exists; no backend process runs in this test.
        std::env::set_var("TRAECLI_BIN", std::env::current_exe().unwrap());
        std::env::set_var("TRAECLI_MODEL", backend_model);
    }
    let parent_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let root = std::env::temp_dir().join(format!(
        "task-model-attribution-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let mut app = test_app_with_model(parent_model.clone());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    app.config.history_file = root.join("history.sqlite");
    app.session_default_subagent_backend = SubagentBackend::TraeCli;
    let owner_pid = app.os.lock().unwrap().begin_foreground(
        "parent".to_string(),
        "goal".to_string(),
        10,
        8,
        None,
    );
    let context = DriverContext::new(
        app.clone(),
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![manifest(
            "navigator",
            "Read-only codebase navigation agent",
            AgentMode::Subagent,
        )]),
    );
    crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        (app.session_id.clone(), 0usize),
        || {
            DRIVER_CTX.sync_scope(context, || {
                let mut args = serde_json::json!({
                    "description": "Inspect attribution",
                    "prompt": "Find task model attribution.",
                    "agent": "navigator",
                    "model": parent_model
                });
                let prepared = prepare_subagent_task(&args).unwrap();
                assert_eq!(prepared.model, parent_model);
                assert_ne!(prepared.model, backend_model);
                assert_eq!(prepared.backend_model.as_deref(), Some(backend_model));
                assert!(prepared.reasoning_effort.is_none());
                unsafe { std::env::set_var("TRAECLI_MODEL", "later-backend-model") };
                let frozen = super::spawn_subagent_kernel_task(&prepared).unwrap();

                args["backend_model"] = serde_json::json!(backend_model);
                let spawned: serde_json::Value = serde_json::from_str(
                    &execute_task_spawn(&serde_json::json!({"tasks": [args.clone()]})).unwrap(),
                )
                .unwrap();
                assert_eq!(spawned["tasks"][0]["model"], backend_model);
                let task_id = spawned["tasks"][0]["task_id"].as_str().unwrap();
                let audit = crate::ai::history::read_task_spawn_audit(
                    &app.config.history_file,
                    &app.session_id,
                )
                .unwrap();
                assert_eq!(audit.len(), 1);
                assert_eq!(audit[0].model, backend_model);

                let task_ids = [frozen.task_id.as_str(), task_id];
                for task_id in task_ids {
                    let entry = super::with_task_entry(task_id, Clone::clone).unwrap();
                    assert_eq!(entry.model, backend_model);
                    set_task_reasoning_effort(task_id, Some("xhigh".to_string()));
                    let os = app.os.lock().unwrap();
                    let goal = super::decode_os_task_goal(&os.get_process(entry.pid).unwrap().goal)
                        .unwrap();
                    assert_eq!(goal.model, parent_model);
                    assert_eq!(goal.backend_model.as_deref(), Some(backend_model));
                }
                let snapshots = super::subagent_status_snapshots(&app.session_id);
                let statuses =
                    super::subagent_terminal_statuses(app.os.lock().unwrap().as_mut(), snapshots);
                assert_eq!(statuses.len(), 2);
                assert!(statuses.iter().all(|status| {
                    status.model.as_deref() == Some(backend_model)
                        && status.reasoning_effort.as_deref() == Some("xhigh")
                }));
                let pending = execute_task_status(&serde_json::json!({})).unwrap();
                for task_id in task_ids {
                    let row = pending
                        .lines()
                        .find(|line| line.starts_with(task_id))
                        .unwrap();
                    assert_eq!(row.split_whitespace().nth(3), Some(backend_model));
                    let entry = super::with_task_entry(task_id, Clone::clone).unwrap();
                    app.os
                        .lock()
                        .unwrap()
                        .channel_send(
                            Some(owner_pid),
                            ChannelId(entry.result_channel_id),
                            serde_json::json!({"status": "completed", "output": "confirmed result"})
                                .to_string(),
                        )
                        .unwrap();
                }
                let delivered = execute_task_status(&serde_json::json!({})).unwrap();
                let header = format!(
                    "[Task: Inspect attribution via navigator @ {backend_model} (effort: xhigh)]"
                );
                assert_eq!(delivered.matches(&header).count(), 2);
                let reason = format!("model_reason=backend model '{backend_model}'");
                assert_eq!(delivered.matches(&reason).count(), 2);
                assert!(!delivered.contains("model_reason=explicit model override"));
                let evidence = crate::ai::history::read_task_spawn_audit(
                    &app.config.history_file,
                    &app.session_id,
                )
                .unwrap()
                .into_iter()
                .filter(|record| {
                    record.delivered_at_unix_ms > 0 && record.integrated_at_unix_ms.is_none()
                })
                .collect::<Vec<_>>();
                assert_eq!(evidence.len(), 2);
                assert!(evidence.iter().all(|record| {
                    record.model == backend_model
                        && record.payload.contains(&header)
                        && record.payload.contains(&reason)
                }));

                unsafe { std::env::set_var("TRAECLI_BIN", root.join("absent-traecli")) };
                let mut kernel_args = args.clone();
                kernel_args["backend"] = serde_json::json!("kernel");
                let failed_batch: serde_json::Value = serde_json::from_str(
                    &execute_task_spawn(&serde_json::json!({"tasks": [kernel_args, args]}))
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(failed_batch["failed"], 1);
                assert_eq!(failed_batch["tasks"][0]["model"], parent_model);
                assert_eq!(failed_batch["tasks"][1]["model"], backend_model);
                discard_tasks_for_session(&app.session_id);
            })
        },
    );
    let sessions_root = crate::ai::history::SessionStore::new(&app.config.history_file)
        .sessions_root()
        .to_path_buf();
    let _ = std::fs::remove_dir_all(sessions_root);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn use_command_task_preparation_prefers_explicit_worker_model_over_session() {
    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let mut app = test_app_with_model(current_model.clone());
    app.session_default_subagent_backend = SubagentBackend::TraeCli;
    app.session_default_subagent_model = Some("session-worker-model".to_string());
    let ctx = DriverContext::new(
        app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![navigator]),
    );
    DRIVER_CTX.sync_scope(ctx, || {
        for (backend, per_call, expected_backend, expected_model) in [
            (None, None, SubagentBackend::TraeCli, Some("session-worker-model")),
            (Some("traecli"), None, SubagentBackend::TraeCli, Some("session-worker-model")),
            (None, Some("per-call-model"), SubagentBackend::TraeCli, Some("per-call-model")),
            (Some("kernel"), None, SubagentBackend::Kernel, None),
        ] {
            let args = serde_json::json!({
                "description": "Locate task tool", "prompt": "Find task spawning.", "agent": "navigator",
                "backend": backend, "backend_model": per_call
            });
            let prepared = prepare_subagent_task(&args).unwrap();
            assert_eq!(prepared.backend, expected_backend);
            assert_eq!(prepared.backend_model.as_deref(), expected_model);
            assert_eq!(crate::ai::driver::runtime_ctx::try_current().unwrap().app_proto.current_model, current_model);
        }
    });
}

#[test]
fn use_command_task_preparation_applies_worker_model_with_kernel_session() {
    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let mut app = test_app_with_model(current_model);
    app.session_default_subagent_model = Some("session-worker-model".to_string());
    let ctx = DriverContext::new(
        app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![navigator]),
    );
    DRIVER_CTX.sync_scope(ctx, || {
        for backend_args in [serde_json::json!({"backend": "traecli"}), serde_json::json!({"backend_continue_from": "session-id"})] {
            let mut args = serde_json::json!({"description": "Locate task tool", "prompt": "Find task spawning.", "agent": "navigator"});
            args.as_object_mut().unwrap().extend(backend_args.as_object().unwrap().clone());
            let prepared = prepare_subagent_task(&args).unwrap();
            assert_eq!(prepared.backend, SubagentBackend::TraeCli);
            assert_eq!(prepared.backend_model.as_deref(), Some("session-worker-model"));
        }
    });
}

#[test]
fn prepare_subagent_task_carries_parent_images_to_traecli_prompt() {
    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let mut app = test_app_with_model(current_model);
    app.session_default_subagent_backend = SubagentBackend::TraeCli;
    app.attached_image_files = vec![
        "/tmp/original-a.png".to_string(),
        "/tmp/original-b.png".to_string(),
    ];
    let ctx = DriverContext::new(
        app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![navigator]),
    );
    let args = serde_json::json!({
        "description": "Inspect the screenshot",
        "prompt": "Read the attached image and report what it shows.",
        "agent": "navigator"
    });

    let prepared = DRIVER_CTX
        .sync_scope(ctx, || prepare_subagent_task(&args))
        .unwrap();

    assert_eq!(
        prepared.image_files,
        vec![
            "/tmp/original-a.png".to_string(),
            "/tmp/original-b.png".to_string()
        ]
    );
    assert!(prepared.prompt.contains("## Original images"));
    assert!(prepared.prompt.contains("/tmp/original-a.png"));
    assert!(prepared.prompt.contains("/tmp/original-b.png"));
}

#[test]
fn prepare_subagent_task_explicit_backend_overrides_session_default() {
    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let mut app = test_app_with_model(current_model);
    app.session_default_subagent_backend = SubagentBackend::TraeCli;
    let ctx = DriverContext::new(
        app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![navigator]),
    );
    let args = serde_json::json!({
        "description": "Locate task tool",
        "prompt": "Find where task spawning is implemented.",
        "agent": "navigator",
        "backend": "kernel"
    });

    let prepared = DRIVER_CTX
        .sync_scope(ctx, || prepare_subagent_task(&args))
        .unwrap();

    assert_eq!(prepared.backend, SubagentBackend::Kernel);
    assert!(prepared.show_backend_launch_tag);
}

#[test]
fn prepare_subagent_task_continue_from_omitted_backend_infers_traecli() {
    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let ctx = DriverContext::new(
        test_app_with_model(current_model),
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![navigator]),
    );
    let args = serde_json::json!({
        "description": "Continue task",
        "prompt": "Continue the earlier investigation.",
        "agent": "navigator",
        "backend_continue_from": "task_previous"
    });

    let prepared = DRIVER_CTX
        .sync_scope(ctx, || prepare_subagent_task(&args))
        .unwrap();

    assert_eq!(prepared.backend, SubagentBackend::TraeCli);
    assert!(prepared.show_backend_launch_tag);
}

#[test]
fn prepare_subagent_task_continue_from_rejects_explicit_kernel() {
    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("model registry must contain at least one model");
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let ctx = DriverContext::new(
        test_app_with_model(current_model),
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(vec![navigator]),
    );
    let args = serde_json::json!({
        "description": "Continue task",
        "prompt": "Continue the earlier investigation.",
        "agent": "navigator",
        "backend": "kernel",
        "backend_continue_from": "task_previous"
    });

    let error = match DRIVER_CTX.sync_scope(ctx, || prepare_subagent_task(&args)) {
        Ok(_) => panic!("explicit kernel backend with backend_continue_from should fail"),
        Err(error) => error,
    };

    assert!(error.contains("backend_continue_from"), "{error}");
    assert!(error.contains("traecli"), "{error}");
}

#[test]
fn task_integrate_closes_durable_evidence() {
    let root =
        std::env::temp_dir().join(format!("task-integrate-{}", uuid::Uuid::new_v4().simple()));
    let history_file = root.join("history.sqlite");
    let session_id = format!("task-integrate-{}", uuid::Uuid::new_v4().simple());
    crate::ai::history::record_delivered_task_evidence(
        &history_file,
        &session_id,
        crate::ai::history::DeliveredTaskEvidence {
            task_id: "task-1",
            description: "review parser",
            agent_name: "build",
            model: "test-model",
            status: "completed",
            payload: "[Subagent final answer]\nconfirmed",
        },
    )
    .unwrap();
    let mut app = test_app_with_model(String::new());
    app.config.history_file = history_file.clone();
    app.session_id = session_id.clone();
    let context = DriverContext::new(
        app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    let result = DRIVER_CTX.sync_scope(context, || {
        execute_task_integrate(&serde_json::json!({
            "tasks": [{
                "task_id": "task-1",
                "disposition": "accepted",
                "summary": "used confirmed conclusion"
            }]
        }))
    });
    assert!(result.unwrap().contains("task-1"));
    assert!(
        crate::ai::history::render_unintegrated_task_evidence(&history_file, &session_id)
            .unwrap()
            .is_none()
    );

    let sessions_root = crate::ai::history::SessionStore::new(&history_file)
        .sessions_root()
        .to_path_buf();
    let _ = std::fs::remove_dir_all(sessions_root);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn task_integrate_batch_rolls_back_when_any_task_is_undelivered() {
    let root = std::env::temp_dir().join(format!(
        "task-integrate-batch-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let history_file = root.join("history.sqlite");
    let session_id = format!("task-integrate-batch-{}", uuid::Uuid::new_v4().simple());
    crate::ai::history::record_delivered_task_evidence(
        &history_file,
        &session_id,
        crate::ai::history::DeliveredTaskEvidence {
            task_id: "task-delivered",
            description: "review parser",
            agent_name: "build",
            model: "test-model",
            status: "completed",
            payload: "[Subagent final answer]\nconfirmed",
        },
    )
    .unwrap();
    let mut app = test_app_with_model(String::new());
    app.config.history_file = history_file.clone();
    app.session_id = session_id.clone();
    let context = DriverContext::new(
        app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    let result = DRIVER_CTX.sync_scope(context, || {
        execute_task_integrate(&serde_json::json!({
            "tasks": [
                {
                    "task_id": "task-delivered",
                    "disposition": "accepted",
                    "summary": "first item is valid"
                },
                {
                    "task_id": "task-undelivered",
                    "disposition": "accepted",
                    "summary": "second item has no delivered evidence"
                }
            ]
        }))
    });
    let error = result.unwrap_err();
    assert!(error.contains("task-undelivered"), "{error}");
    assert!(error.contains("No integrations were applied"), "{error}");
    assert!(
        crate::ai::history::read_integrated_task_ids(&history_file, &session_id)
            .unwrap()
            .is_empty(),
        "the valid first item must be rolled back with the invalid second item"
    );

    let sessions_root = crate::ai::history::SessionStore::new(&history_file)
        .sessions_root()
        .to_path_buf();
    let _ = std::fs::remove_dir_all(sessions_root);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn capped_subagent_manifest_preserves_explicit_iteration_budget_within_hard_cap() {
    let mut agent = manifest("build", "Main build agent", AgentMode::Subagent);
    agent.max_steps = Some(128);

    let capped = capped_subagent_manifest(&agent);

    assert_eq!(capped.max_steps, Some(128));
    assert_eq!(agent.max_steps, Some(128));
}

#[test]
fn capped_subagent_manifest_preserves_stricter_iteration_budget() {
    let mut agent = manifest("small", "Small subagent", AgentMode::Subagent);
    agent.max_steps = Some(12);

    let capped = capped_subagent_manifest(&agent);

    assert_eq!(capped.max_steps, Some(12));
}

#[test]
fn capped_subagent_manifest_limits_large_iteration_budget_to_hard_cap() {
    let mut agent = manifest("build", "Main build agent", AgentMode::Subagent);
    agent.max_steps = Some(9999);

    let capped = capped_subagent_manifest(&agent);

    assert_eq!(
        capped.max_steps,
        Some(super::SUBAGENT_MAX_ITERATIONS_HARD_CAP)
    );
    assert_eq!(agent.max_steps, Some(9999));
}

#[test]
fn explicit_primary_agent_is_rejected_for_task_tool() {
    let build = manifest("build", "Main build agent", AgentMode::Primary);
    let navigator = manifest(
        "navigator",
        "Read-only codebase navigation agent",
        AgentMode::Subagent,
    );
    let all_agents = vec![build, navigator];

    let err =
        select_subagent(&all_agents, Some("build"), "Inspect code", "Look up files").unwrap_err();

    assert!(err.contains("not a subagent"));
}

#[test]
fn auto_selection_matches_task_to_subagent_description() {
    let mut explore = manifest(
        "navigator",
        "Read-only codebase exploration agent",
        AgentMode::Subagent,
    );
    explore.model_tier = Some(AgentModelTier::Light);

    let review = manifest("critic", "Code review agent", AgentMode::Subagent);
    let all_agents = vec![explore, review];

    let selected = select_subagent(
        &all_agents,
        None,
        "Find handler",
        "Search the codebase and locate where the request handler is defined.",
    )
    .unwrap();

    assert_eq!(selected.agent.name, "navigator");
}

#[test]
fn auto_selection_excludes_opted_out_agents_despite_a_higher_score() {
    // The reviewer's text mirrors the task wording almost verbatim, so it would
    // win the lexical comparison; auto_select: false must remove it first. This
    // guards the failure where a contract-bound audit agent was auto-picked for
    // code-writing work.
    let mut build = manifest(
        "build",
        "Default agent for all tasks with progressive tool loading",
        AgentMode::All,
    );
    build.prompt = "End-to-end delivery across code, systems and documents.".to_string();

    let mut review = manifest(
        "audit-fast",
        "Quick bounded code review for light/simple changes",
        AgentMode::All,
    );
    review.prompt =
        "Review code changes conservatively and report real defects with evidence.".to_string();
    review.auto_select = false;

    let all_agents = vec![build, review];
    let description = "Review code changes conservatively";
    let prompt = "Review code changes conservatively: report real defects with evidence.";

    let selected = select_subagent(&all_agents, None, description, prompt).unwrap();
    assert_eq!(selected.agent.name, "build");
    assert!(selected.auto_selected);

    let explicit = select_subagent(&all_agents, Some("audit-fast"), description, prompt).unwrap();
    assert_eq!(explicit.agent.name, "audit-fast");
    assert!(!explicit.auto_selected);
}

#[test]
fn auto_selection_errors_when_every_subagent_opts_out() {
    let mut first = manifest("audit", "Evidence-driven reviewer", AgentMode::All);
    first.auto_select = false;
    let mut second = manifest("audit-fast", "Quick bounded review", AgentMode::All);
    second.auto_select = false;

    let err =
        select_subagent(&vec![first, second], None, "Add tests", "Write unit tests.").unwrap_err();

    assert!(err.contains("auto_select"), "{err}");
}

#[test]
fn selection_explanation_mentions_quality_tier_for_auto_model_choice() {
    // 之前这里硬编码 "qwen3-max"；该模型已经从模型注册表移除。
    // 改为从真实条目中找一个 Alibaba+flagship 的模型，确保解释里出现
    // "flagship" 和 "alibaba" 这两个 tier/adapter 关键字。
    use crate::ai::provider::{ApiProvider, ModelQualityTier};
    let model = crate::ai::model_names::all()
        .iter()
        .find(|m| m.adapter == ApiProvider::Alibaba && m.quality_tier == ModelQualityTier::Flagship)
        .map(|m| m.name.clone());
    let Some(model) = model else {
        eprintln!(
            "[test] skipping selection_explanation_mentions_quality_tier_for_auto_model_choice: \
                 no Alibaba+Flagship model present in model registry"
        );
        return;
    };

    let agent = manifest("build", "Main build agent", AgentMode::Subagent);
    let selected = SelectedSubagent {
        agent: &agent,
        auto_selected: true,
        score: 48,
    };

    let explanation = build_selection_explanation(&selected, &model, None, false);

    assert!(explanation.contains("quality_tier"));
    assert!(explanation.contains("flagship"));
    assert!(explanation.contains("alibaba"));
}

#[test]
fn selection_explanation_mentions_explicit_overrides() {
    let agent = manifest(
        "explore",
        "Read-only codebase exploration agent",
        AgentMode::Subagent,
    );
    let selected = SelectedSubagent {
        agent: &agent,
        auto_selected: false,
        score: 0,
    };

    let explanation = build_selection_explanation(&selected, "gpt-4o", Some("gpt-4o"), false);

    assert!(explanation.contains("explicit agent override"));
    assert!(explanation.contains("explicit model override"));
}

#[test]
fn blank_model_override_is_treated_as_auto_selection() {
    let agent = manifest(
        "explore",
        "Read-only codebase exploration agent",
        AgentMode::Subagent,
    );
    let selected = SelectedSubagent {
        agent: &agent,
        auto_selected: true,
        score: 0,
    };

    let explanation = build_selection_explanation(&selected, "deepseek-v4-flash", Some(" "), false);

    assert!(explanation.contains("auto-selected"));
    assert!(!explanation.contains("explicit model override"));
}

#[test]
fn epoll_wait_many_channels_returns_ready_without_suspending() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
    let channel = os.channel_create(Some(root), 1, "task-ready".to_string());
    os.channel_send(Some(root), channel, "payload".to_string())
        .unwrap();

    let wait = epoll_wait_many_channels(
        &mut os,
        "task_wait:test-ready",
        &[channel.raw()],
        WaitPolicy::All,
        None,
    )
    .unwrap();

    assert_eq!(
        wait.ready_sources,
        vec![WaitManySource::Channel(channel.raw())]
    );
    assert!(wait.pending_sources.is_empty());
    assert!(!wait.suspended);
    assert!(wait.event_ids.is_empty());
}

#[test]
fn epoll_wait_many_channels_preserves_all_wait_suspension() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
    let channel_a = os.channel_create(Some(root), 1, "task-a".to_string());
    let channel_b = os.channel_create(Some(root), 1, "task-b".to_string());

    let wait = epoll_wait_many_channels(
        &mut os,
        "task_wait:test-suspend",
        &[channel_a.raw(), channel_b.raw()],
        WaitPolicy::All,
        None,
    )
    .unwrap();

    assert!(wait.ready_sources.is_empty());
    assert_eq!(
        wait.pending_sources,
        vec![
            WaitManySource::Channel(channel_a.raw()),
            WaitManySource::Channel(channel_b.raw())
        ]
    );
    assert_eq!(wait.event_ids.len(), 2);
    assert!(wait.suspended);
    assert!(os.current_process_id().is_none());
}

#[test]
fn epoll_wait_many_supports_mixed_ready_sources() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
    let channel = os.channel_create(Some(root), 1, "mixed-channel".to_string());
    let futex = os.futex_create(0, "mixed-futex".to_string());
    let event = EventId::new(77);
    os.channel_send(Some(root), channel, "payload".to_string())
        .unwrap();
    let _ = os.futex_store(futex, 1);
    os.notify_events_completed(&[event]);

    let wait = epoll_wait_many(
        &mut os,
        "mixed-ready",
        &[
            WaitManySource::Channel(channel.raw()),
            WaitManySource::Futex {
                addr: futex,
                expected: 0,
            },
            WaitManySource::Event(event),
        ],
        WaitPolicy::Any,
        None,
    )
    .unwrap();

    assert_eq!(
        wait.ready_sources,
        vec![
            WaitManySource::Channel(channel.raw()),
            WaitManySource::Futex {
                addr: futex,
                expected: 0
            },
            WaitManySource::Event(event),
        ]
    );
    assert!(wait.pending_sources.is_empty());
    assert!(!wait.suspended);
}

#[test]
fn epoll_wait_many_supports_mixed_all_wait_suspension() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
    let channel = os.channel_create(Some(root), 1, "mixed-channel".to_string());
    let futex = os.futex_create(0, "mixed-futex".to_string());
    let event = EventId::new(88);
    os.notify_events_completed(&[event]);

    let wait = epoll_wait_many(
        &mut os,
        "mixed-all",
        &[
            WaitManySource::Channel(channel.raw()),
            WaitManySource::Futex {
                addr: futex,
                expected: 0,
            },
            WaitManySource::Event(event),
        ],
        WaitPolicy::All,
        None,
    )
    .unwrap();

    assert_eq!(wait.ready_sources, vec![WaitManySource::Event(event)]);
    assert_eq!(
        wait.pending_sources,
        vec![
            WaitManySource::Channel(channel.raw()),
            WaitManySource::Futex {
                addr: futex,
                expected: 0
            },
        ]
    );
    assert_eq!(wait.event_ids.len(), 2);
    assert!(wait.suspended);
    assert!(os.current_process_id().is_none());
}

#[test]
fn task_wait_low_level_wait_wakes_on_task_result_even_with_cancel_source() {
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
    let channel = os.channel_create(Some(root), 1, "task-result".to_string());
    let futex = os.futex_create(0, "task-complete".to_string());
    let mut sources =
        wait_sources_for_channel_and_futex(&mut os, channel.raw(), Some(futex)).unwrap();
    append_current_process_cancel_source(&mut os, &mut sources).unwrap();

    let wait = epoll_wait_many(
        &mut os,
        "task_wait:with-cancel-source",
        &sources,
        WaitPolicy::Any,
        None,
    )
    .unwrap();

    assert!(wait.suspended);
    assert!(os.current_process_id().is_none());

    os.channel_send(Some(root), channel, "payload".to_string())
        .unwrap();

    let root_proc = os.get_process(root).unwrap();
    assert_eq!(root_proc.state, aios_kernel::kernel::ProcessState::Ready);
}

#[test]
fn task_wait_formats_empty_subagent_result_explicitly() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let entry = AsyncTaskEntry {
        evidence_history_file: None,
        session_id: String::new(),
        owner_pid: 0,
        pid: 42,
        result_channel_id: 1,
        completion_futex_addr: FutexAddr(1),
        description: "verify behavior".to_string(),
        agent_name: "explore".to_string(),
        backend: None,
        model: "qwen3.7-max".to_string(),
        is_model_auto_selected: true,
        auto_model_fallback: None,
        selection_explanation: "model_reason=auto-selected".to_string(),
        inherit: InheritOptions::default(),
        abort_handle: None,
        last_progress_notification_at: None,
        last_progress_persisted_at: None,
        cancel_stream: Arc::new(AtomicBool::new(false)),
        started_at: Instant::now(),
    };
    let result = StoredTaskResult {
        status: "failed".to_string(),
        output: String::new(),
        error: Some("request timed out waiting for response headers".to_string()),
        progress: Some("calling model".to_string()),
        backend_session: None,
    };

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let without_effort = format_task_result_with_id(&task_id, &entry, result.clone());
    assert!(without_effort.starts_with(&format!(
        "[task_id={task_id}]\n[Task: verify behavior via explore @ qwen3.7-max]"
    )));
    assert!(!without_effort.contains("(effort:"));

    set_task_reasoning_effort(&task_id, Some("high".to_string()));
    let output = format_task_result_with_id(&task_id, &entry, result);
    assert!(output.starts_with(&format!(
        "[task_id={task_id}]\n[Task: verify behavior via explore @ qwen3.7-max (effort: high)]"
    )));
    let _ = remove_task_entry(&task_id);
    assert!(task_reasoning_effort(&task_id).is_none());

    assert!(output.contains("FAILED"));
    assert!(output.contains("Failure reason: request timed out waiting for response headers"));
    assert!(output.contains("Last known progress: calling model"));
    assert!(output.contains("(subagent did not produce any final assistant text)"));
    assert!(output.contains("task_retry"));
    assert!(output.contains(&task_id));
    assert!(output.contains(SUBAGENT_PARENT_SUMMARY_REMINDER));

    let blocked = format_task_result_with_id(
        &task_id,
        &entry,
        StoredTaskResult {
            status: "blocked".to_string(),
            output: "authenticate in the foreground".to_string(),
            error: Some("authorization_required: token expired".to_string()),
            progress: None,
            backend_session: None,
        },
    );
    assert!(blocked.contains("BLOCKED"));
    assert!(blocked.contains(SUBAGENT_BLOCKED_PARENT_GUIDANCE));
    assert!(blocked.contains("task_retry"));
}

#[test]
fn task_retry_status_matches_the_runtime_timeout_value() {
    assert!(is_retryable_task_status("blocked"));
    assert!(is_retryable_task_status("failed"));
    assert!(is_retryable_task_status("timeout"));
    assert!(is_retryable_task_status("cancelled"));
    assert!(!is_retryable_task_status("timed_out"));
    assert!(!is_retryable_task_status("completed"));
}

#[test]
fn subagent_depth_rejects_task_orchestration_execution() {
    crate::ai::driver::runtime_ctx::SUBAGENT_DEPTH.sync_scope(1, || {
        for (tool_name, result) in [
            (
                "task_spawn",
                execute_task_spawn(&serde_json::json!({
                    "tasks": [{
                        "description": "nested",
                        "prompt": "should not run"
                    }]
                })),
            ),
            (
                "task_wait",
                execute_task_wait(&serde_json::json!({ "task_ids": ["task_parent"] })),
            ),
            (
                "task_retry",
                execute_task_retry(&serde_json::json!({ "task_id": "task_parent" })),
            ),
            ("task_status", execute_task_status(&serde_json::json!({}))),
            (
                "task_integrate",
                execute_task_integrate(&serde_json::json!({
                    "tasks": [{
                        "task_id": "task_parent",
                        "disposition": "accepted",
                        "summary": "nested"
                    }]
                })),
            ),
            (
                "task_cancel",
                execute_task_cancel(&serde_json::json!({ "task_ids": ["task_parent"] })),
            ),
        ] {
            let err = result.expect_err("subagent task orchestration should be rejected");
            assert!(
                err.contains(tool_name),
                "error should name {tool_name}: {err}"
            );
            assert!(
                err.contains("top-level agents") && err.contains("leaf task"),
                "error should explain subagent leaf-task boundary: {err}"
            );
        }
    });
}

#[test]
fn task_wait_rejects_foreign_session_task_ids() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = "session-a".to_string();
    let _os_guard = init_os_tools_guard(app.os.clone());
    let owner_pid = {
        let mut os = app.os.lock().unwrap();
        os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None)
    };
    let own_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let foreign_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let own_task_id_for_call = own_task_id.clone();
    let foreign_task_id_for_call = foreign_task_id.clone();
    insert_task_entry_for_test(
        own_task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: "session-a".to_string(),
            owner_pid,
            pid: 1,
            result_channel_id: 11,
            completion_futex_addr: FutexAddr(21),
            description: "own task".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );
    insert_task_entry_for_test(
        foreign_task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: "session-b".to_string(),
            owner_pid,
            pid: 2,
            result_channel_id: 12,
            completion_futex_addr: FutexAddr(22),
            description: "foreign task".to_string(),
            agent_name: "build".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    let result = crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        ("session-a".to_string(), 0usize),
        || {
            execute_task_wait(
                &serde_json::json!({ "task_ids": [own_task_id_for_call, foreign_task_id_for_call] }),
            )
        },
    );

    let err = result.expect_err("foreign task_id should be rejected");
    assert!(err.contains("owned by another session"));
    assert!(remove_task_entry(&own_task_id).is_some());
    assert!(remove_task_entry(&foreign_task_id).is_some());
}

#[test]
fn task_wait_distinguishes_delivered_and_unknown_registry_misses() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "task-wait-tombstone-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    app.config.history_file = root.join("history.sqlite");
    let _os_guard = init_os_tools_guard(app.os.clone());
    {
        let mut os = app.os.lock().unwrap();
        os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
    }
    let delivered_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let unknown_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    crate::ai::history::record_delivered_task_evidence(
        app.config.history_file.as_path(),
        &app.session_id,
        crate::ai::history::DeliveredTaskEvidence {
            task_id: &delivered_task_id,
            description: "already delivered",
            agent_name: "explore",
            model: "qwen3.7-max",
            status: "completed",
            payload: "delivered result",
        },
    )
    .unwrap();
    let context = DriverContext::new(
        app.clone(),
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );

    let (delivered, unknown) = crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        (app.session_id.clone(), 0usize),
        || {
            DRIVER_CTX.sync_scope(context, || {
                (
                    execute_task_wait(
                        &serde_json::json!({ "task_ids": [delivered_task_id.clone()] }),
                    ),
                    execute_task_wait(
                        &serde_json::json!({ "task_ids": [unknown_task_id.clone()] }),
                    ),
                )
            })
        },
    );

    assert!(delivered.unwrap().contains("results were delivered"));
    let unknown = unknown.expect_err("unknown task id must not be reported as delivered");
    assert!(unknown.contains("Unknown task_id"));
    assert!(unknown.contains(&unknown_task_id));
    assert!(!unknown.contains("results were delivered"));

    let sessions_root = crate::ai::history::SessionStore::new(&app.config.history_file)
        .sessions_root()
        .to_path_buf();
    let _ = std::fs::remove_dir_all(sessions_root);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn task_wait_escalates_repeated_noop_wait_on_delivered_task_to_error() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "task-wait-noop-escalation-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    app.config.history_file = root.join("history.sqlite");
    let _os_guard = init_os_tools_guard(app.os.clone());
    {
        let mut os = app.os.lock().unwrap();
        os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
    }
    let delivered_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    crate::ai::history::record_delivered_task_evidence(
        app.config.history_file.as_path(),
        &app.session_id,
        crate::ai::history::DeliveredTaskEvidence {
            task_id: &delivered_task_id,
            description: "already delivered",
            agent_name: "explore",
            model: "qwen3.7-max",
            status: "completed",
            payload: "delivered result",
        },
    )
    .unwrap();
    // DriverContext is not Clone, so rebuild it per call (all fields are cheap Arc clones).
    let make_context = || {
        DriverContext::new(
            app.clone(),
            Arc::new(std::sync::Mutex::new(McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        )
    };
    let run = |task_id: &str| {
        crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
            (app.session_id.clone(), 0usize),
            || {
                DRIVER_CTX.sync_scope(make_context(), || {
                    execute_task_wait(&serde_json::json!({ "task_ids": [task_id] }))
                })
            },
        )
    };

    // First two no-op polls on the delivered id stay soft (below the escalation threshold)...
    let first = run(&delivered_task_id).unwrap();
    assert!(first.contains("results were delivered"));
    let second = run(&delivered_task_id).unwrap();
    assert!(second.contains("results were delivered"));
    // ...the third consecutive no-op poll for the same key escalates to an error...
    let third = run(&delivered_task_id).expect_err("no-op wait must escalate to an error");
    assert!(third.contains("Repeated task_wait"));
    assert!(third.contains("task_integrate"));
    // ...and stays an error on further repeats instead of going soft again.
    let fourth = run(&delivered_task_id).expect_err("further repeats must stay errors");
    assert!(fourth.contains("Repeated task_wait"));

    // Per-key isolation: a different delivered id has its own counter and stays soft.
    let other_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    crate::ai::history::record_delivered_task_evidence(
        app.config.history_file.as_path(),
        &app.session_id,
        crate::ai::history::DeliveredTaskEvidence {
            task_id: &other_task_id,
            description: "other delivered",
            agent_name: "explore",
            model: "qwen3.7-max",
            status: "completed",
            payload: "other result",
        },
    )
    .unwrap();
    let other = run(&other_task_id).unwrap();
    assert!(other.contains("results were delivered"));

    // Reset the global counter so later tests start clean.
    reset_task_wait_noop_counts_for_test();

    let sessions_root = crate::ai::history::SessionStore::new(&app.config.history_file)
        .sessions_root()
        .to_path_buf();
    let _ = std::fs::remove_dir_all(sessions_root);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn task_status_and_outstanding_anchor_filter_same_session_sibling_owner_tasks() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let own_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let sibling_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (
        owner_pid,
        sibling_owner_pid,
        own_pid,
        sibling_pid,
        own_channel,
        own_futex,
        sibling_channel,
        sibling_futex,
    ) = {
        let mut os = app.os.lock().unwrap();
        let owner_pid = os.begin_foreground("owner".to_string(), "goal".to_string(), 10, 8, None);
        let own_pid = os
            .spawn(
                Some(owner_pid),
                "own-child".to_string(),
                "own goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let own_channel = os.channel_create(Some(own_pid), 1, "own-result".to_string());
        let own_futex = os.futex_create(0, "own-complete".to_string());
        let sibling_owner_pid =
            os.begin_foreground("sibling-owner".to_string(), "goal".to_string(), 10, 8, None);
        let sibling_pid = os
            .spawn(
                Some(sibling_owner_pid),
                "sibling-child".to_string(),
                "sibling goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let sibling_channel = os.channel_create(Some(sibling_pid), 1, "sibling-result".to_string());
        let sibling_futex = os.futex_create(0, "sibling-complete".to_string());
        os.set_current_pid(Some(owner_pid));
        (
            owner_pid,
            sibling_owner_pid,
            own_pid,
            sibling_pid,
            own_channel.raw(),
            own_futex,
            sibling_channel.raw(),
            sibling_futex,
        )
    };

    for (task_id, owner_pid, pid, channel, futex, description) in [
        (
            own_task_id.clone(),
            owner_pid,
            own_pid,
            own_channel,
            own_futex,
            "own visible task",
        ),
        (
            sibling_task_id.clone(),
            sibling_owner_pid,
            sibling_pid,
            sibling_channel,
            sibling_futex,
            "sibling hidden task",
        ),
    ] {
        insert_task_entry_for_test(
            task_id,
            AsyncTaskEntry {
                evidence_history_file: None,
                session_id: app.session_id.clone(),
                owner_pid,
                pid,
                result_channel_id: channel,
                completion_futex_addr: futex,
                description: description.to_string(),
                agent_name: "explore".to_string(),
                backend: None,
                model: "qwen3.7-max".to_string(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: "explicit override".to_string(),
                inherit: InheritOptions::default(),
                abort_handle: None,
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
                started_at: Instant::now(),
            },
        );
    }

    let (status, anchor) = crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        (app.session_id.clone(), 0usize),
        || {
            (
                execute_task_status(&serde_json::json!({})).unwrap(),
                build_outstanding_task_anchor(&app.session_id)
                    .unwrap()
                    .unwrap(),
            )
        },
    );

    assert!(status.contains("own visible task"));
    assert!(!status.contains("sibling hidden task"));
    let row = status
        .lines()
        .find(|line| line.starts_with(&own_task_id))
        .expect("owned task must have a status row");
    assert_eq!(row.split_whitespace().nth(4), Some("-"));
    assert!(anchor.contains(&own_task_id));
    assert!(!anchor.contains(&sibling_task_id));

    assert!(remove_task_entry(&own_task_id).is_some());
    assert!(remove_task_entry(&sibling_task_id).is_some());
}

#[test]
fn task_cancel_refuses_same_session_foreign_owner_task() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (owner_pid, sibling_owner_pid, child_pid, channel, futex) = {
        let mut os = app.os.lock().unwrap();
        let owner_pid = os.begin_foreground("owner".to_string(), "goal".to_string(), 10, 8, None);
        let sibling_owner_pid =
            os.begin_foreground("sibling-owner".to_string(), "goal".to_string(), 10, 8, None);
        let child_pid = os
            .spawn(
                Some(sibling_owner_pid),
                "sibling-child".to_string(),
                "sibling goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create(Some(child_pid), 1, "sibling-result".to_string());
        let futex = os.futex_create(0, "sibling-complete".to_string());
        os.set_current_pid(Some(owner_pid));
        (
            owner_pid,
            sibling_owner_pid,
            child_pid,
            channel.raw(),
            futex,
        )
    };
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid: sibling_owner_pid,
            pid: child_pid,
            result_channel_id: channel,
            completion_futex_addr: futex,
            description: "sibling task".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    let output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_cancel(&serde_json::json!({ "task_ids": [task_id.clone()] }))
        })
        .expect("foreign-owner cancel should return a status message");

    assert!(output.contains("not found or not owned by this process/session"));
    assert!(remove_task_entry(&task_id).is_some());
    let mut os = app.os.lock().unwrap();
    os.set_current_pid(Some(owner_pid));
    assert!(super::is_task_pending(os.as_mut(), child_pid).unwrap());
    os.set_current_pid(Some(sibling_owner_pid));
    os.kill_process(child_pid, "test cleanup".to_string())
        .unwrap();
    drop(os);
}

#[test]
fn orphaned_task_recovery_retains_session_scope_and_retry_collection_gate() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "orphaned-task-control-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let mut app = test_app_with_model(String::new());
    app.session_id = format!("owner-session-{}", uuid::Uuid::new_v4().simple());
    app.config.history_file = root.join("history.sqlite");
    let other_session = format!("other-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());
    let context = DriverContext::from_app_snapshot(
        &app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    let mut prepared = prepared_retry_task();
    prepared.model.clear();
    let owner = app
        .os
        .lock()
        .unwrap()
        .begin_foreground("old".into(), "goal".into(), 10, 8, None);
    let spawn = |session_id: &str| {
        DRIVER_CTX.sync_scope(context.clone(), || {
            crate::ai::driver::runtime_ctx::TURN_IDENTITY
                .sync_scope((session_id.to_string(), 0), || {
                    super::spawn_subagent_kernel_task(&prepared).unwrap()
                })
        })
    };
    let task = spawn(&app.session_id);
    let foreign = spawn(&other_session);
    {
        let mut os = app.os.lock().unwrap();
        crate::ai::driver::terminate_and_cleanup(
            os.as_mut(),
            owner,
            "Failed: fixture".into(),
            true,
        );
    }
    let replacement =
        app.os
            .lock()
            .unwrap()
            .begin_foreground("replacement".into(), "goal".into(), 10, 8, None);
    let run = |owner_pid, action: &dyn Fn() -> Result<String, String>| {
        app.os.lock().unwrap().set_current_pid(Some(owner_pid));
        DRIVER_CTX.sync_scope(context.clone(), || {
            crate::ai::driver::runtime_ctx::TURN_IDENTITY
                .sync_scope((app.session_id.clone(), 1), || action())
        })
    };
    let foreign_wait = run(replacement, &|| {
        execute_task_wait(&serde_json::json!({"task_ids": [&foreign.task_id]}))
    })
    .unwrap_err();
    assert!(foreign_wait.contains("another session"), "{foreign_wait}");
    super::with_task_entry(&foreign.task_id, |entry| assert_eq!(entry.owner_pid, owner)).unwrap();
    assert_eq!(
        app.os
            .lock()
            .unwrap()
            .get_process(foreign.pid)
            .unwrap()
            .parent_pid,
        None
    );
    let before_collection = run(replacement, &|| {
        execute_task_retry(&serde_json::json!({"task_id": &task.task_id}))
    })
    .unwrap_err();
    assert!(
        before_collection.contains("no collected terminal failure"),
        "{before_collection}"
    );
    let cancelled = run(replacement, &|| {
        execute_task_cancel(&serde_json::json!({"task_ids": [&task.task_id]}))
    })
    .unwrap();
    assert!(cancelled.contains("Cancelled 1 task"), "{cancelled}");
    let still_uncollected = run(replacement, &|| {
        execute_task_retry(&serde_json::json!({"task_id": &task.task_id}))
    })
    .unwrap_err();
    assert!(
        still_uncollected.contains("no collected terminal failure"),
        "{still_uncollected}"
    );
    let result = run(replacement, &|| {
        execute_task_wait(&serde_json::json!({"task_ids": [&task.task_id]}))
    })
    .unwrap();
    assert!(result.contains("CANCELLED"), "{result}");
    assert!(super::with_task_entry(&task.task_id, |_| ()).is_none());
    {
        let mut os = app.os.lock().unwrap();
        crate::ai::driver::terminate_and_cleanup(
            os.as_mut(),
            replacement,
            "Failed: fixture again".into(),
            true,
        );
    }
    let next_owner =
        app.os
            .lock()
            .unwrap()
            .begin_foreground("next".into(), "goal".into(), 10, 8, None);
    let retried = run(next_owner, &|| {
        execute_task_retry(&serde_json::json!({"task_id": &task.task_id}))
    })
    .unwrap();
    assert!(
        retried.contains(&format!("retry_of={}", task.task_id)),
        "{retried}"
    );
    assert_eq!(
        TASK_RETRY_REGISTRY
            .lock()
            .unwrap()
            .get_ref(&task.task_id)
            .unwrap()
            .owner_pid,
        next_owner
    );
    DRIVER_CTX.sync_scope(context, || {
        discard_tasks_for_session(&app.session_id);
        discard_tasks_for_session(&other_session);
    });
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn task_wait_wall_clock_budget_waker_returns_budget_elapsed_without_tick_timeout() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (owner_pid, child_pid, channel, futex) = {
        let mut os = app.os.lock().unwrap();
        let owner_pid = os.begin_foreground("owner".to_string(), "goal".to_string(), 10, 8, None);
        let child_pid = os
            .spawn(
                Some(owner_pid),
                "pending-child".to_string(),
                "pending goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create(Some(child_pid), 1, "pending-result".to_string());
        let futex = os.futex_create(0, "pending-complete".to_string());
        (owner_pid, child_pid, channel.raw(), futex)
    };
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid,
            pid: child_pid,
            result_channel_id: channel,
            completion_futex_addr: futex,
            description: "pending task".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );
    let parked = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_wait(&serde_json::json!({
                "task_ids": [task_id.clone()],
                "timeout_secs": 1,
            }))
        })
        .expect("first task_wait should park");
    assert!(parked.contains("[task_wait PARKED]"));
    let scheduler_wake_after = super::next_task_wait_wakeup_delay()
        .expect("parked task_wait must expose its wall-clock deadline to the scheduler");
    assert!(scheduler_wake_after <= Duration::from_secs(1));

    expire_task_wait_states_for_test();
    wake_expired_task_waits();
    {
        let mut os = app.os.lock().unwrap();
        let proc = os
            .get_process(owner_pid)
            .expect("owner process should still exist");
        assert!(matches!(
            proc.state,
            aios_kernel::kernel::ProcessState::Ready
        ));
        proc.mailbox
            .iter()
            .find(|message| message.contains("[TASK_WAIT_TIMEOUT]"))
            .expect("timeout wake-up message should be queued");
        let resumed = os
            .pop_foreground_ready()
            .expect("expired wait should wake owner");
        assert_eq!(resumed.pid, owner_pid);
    }

    let elapsed = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_wait(&serde_json::json!({
                "task_ids": [task_id.clone()],
                "timeout_secs": 1,
            }))
        })
        .expect("expired task_wait should report budget elapsed");
    assert!(elapsed.contains("[task_wait BUDGET ELAPSED]"));
    assert!(!elapsed.contains("[task_wait PARKED]"));

    assert!(remove_task_entry(&task_id).is_some());
    let mut os = app.os.lock().unwrap();
    os.set_current_pid(Some(owner_pid));
    os.kill_process(child_pid, "test cleanup".to_string())
        .unwrap();
}

#[test]
fn task_status_collects_completed_results_and_cleans_up_resources() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let root = std::env::temp_dir().join(format!(
        "task-status-ledger-{}",
        uuid::Uuid::new_v4().simple()
    ));
    app.config.history_file = root.join("history.sqlite");
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (pid, result_channel_id, completion_futex_addr) = {
        let mut os = app.os.lock().unwrap();
        let pid = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let channel = os.channel_create(Some(pid), 1, "task-result".to_string());
        let completion_futex = os.futex_create(0, "task-complete".to_string());
        os.channel_send(
            Some(pid),
            channel,
            serde_json::json!({
                "status": "completed",
                "output": "subagent final answer",
                "error": null,
            })
            .to_string(),
        )
        .unwrap();
        (pid, channel.raw(), completion_futex)
    };

    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid: pid,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "inspect parser".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );
    set_task_reasoning_effort(&task_id, Some("high".to_string()));

    let context = DriverContext::new(
        app.clone(),
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    let output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            DRIVER_CTX.sync_scope(context, || execute_task_status(&serde_json::json!({})))
        })
        .expect("task_status should succeed");

    assert!(output.lines().next().unwrap().contains("Effort"));
    let row = output
        .lines()
        .find(|line| line.starts_with(&task_id))
        .expect("completed task must have a status row");
    assert_eq!(row.split_whitespace().nth(4), Some("high"));
    assert!(output.contains("[Task: inspect parser via explore @ qwen3.7-max (effort: high)]"));
    assert!(output.contains("Completed task results below (already collected"));
    assert!(output.contains("subagent final answer"));
    assert!(output.contains("COMPLETED"));
    assert!(task_reasoning_effort(&task_id).is_none());
    assert!(remove_task_entry(&task_id).is_none());

    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());
    drop(os);
    let ledger = crate::ai::history::render_unintegrated_task_evidence(
        app.config.history_file.as_path(),
        &app.session_id,
    )
    .unwrap()
    .expect("collected task result must survive channel cleanup");
    assert!(ledger.contains(&task_id));
    assert!(ledger.contains("subagent final answer"));

    let context = DriverContext::new(
        app.clone(),
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        (app.session_id.clone(), 0usize),
        || {
            DRIVER_CTX.sync_scope(context, || {
                let args = serde_json::json!({"task_id": task_id, "source": "delivered"});
                let output = super::execute_task_evidence_read(&args).unwrap();
                let evidence: serde_json::Value = serde_json::from_str(&output).unwrap();
                assert_eq!(evidence["task_id"], task_id);
                assert_eq!(evidence["source"], "delivered");
                assert_eq!(evidence["status"], "completed");
                assert!(
                    evidence["payload"]
                        .as_str()
                        .unwrap()
                        .contains("subagent final answer")
                );
                let after = crate::ai::history::render_unintegrated_task_evidence(
                    &app.config.history_file,
                    &app.session_id,
                )
                .unwrap()
                .unwrap();
                assert_eq!(
                    after, ledger,
                    "reading must not integrate or rewrite evidence"
                );
                assert!(
                    super::execute_task_evidence_read(&serde_json::json!({
                        "task_id": task_id, "source": "invalid"
                    }))
                    .unwrap_err()
                    .contains("source must be")
                );
                assert!(
                    super::execute_task_evidence_read(&serde_json::json!({
                        "task_id": "never-delivered", "source": "delivered"
                    }))
                    .unwrap_err()
                    .contains("No delivered evidence")
                );
            })
        },
    );

    let sessions_root = crate::ai::history::SessionStore::new(&app.config.history_file)
        .sessions_root()
        .to_path_buf();
    let _ = std::fs::remove_dir_all(sessions_root);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn task_wait_any_returns_ready_result_without_waiting_for_pending_task() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let ready_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let pending_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (root_pid, pending_pid, ready_channel, ready_futex, pending_channel, pending_futex) = {
        let mut os = app.os.lock().unwrap();
        let root_pid = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let pending_pid = os
            .spawn(
                Some(root_pid),
                "pending".to_string(),
                "pending goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let ready_channel = os.channel_create(Some(root_pid), 1, "ready-result".to_string());
        let ready_futex = os.futex_create(0, "ready-complete".to_string());
        let pending_channel = os.channel_create(Some(pending_pid), 1, "pending-result".to_string());
        let pending_futex = os.futex_create(0, "pending-complete".to_string());
        os.channel_send(
            Some(root_pid),
            ready_channel,
            serde_json::json!({
                "status": "completed",
                "output": "first result",
                "error": null,
            })
            .to_string(),
        )
        .unwrap();
        (
            root_pid,
            pending_pid,
            ready_channel.raw(),
            ready_futex,
            pending_channel.raw(),
            pending_futex,
        )
    };

    for (task_id, pid, channel, futex, description) in [
        (
            ready_task_id.clone(),
            root_pid,
            ready_channel,
            ready_futex,
            "ready task",
        ),
        (
            pending_task_id.clone(),
            pending_pid,
            pending_channel,
            pending_futex,
            "pending task",
        ),
    ] {
        insert_task_entry_for_test(
            task_id,
            AsyncTaskEntry {
                evidence_history_file: None,
                session_id: app.session_id.clone(),
                owner_pid: root_pid,
                pid,
                result_channel_id: channel,
                completion_futex_addr: futex,
                description: description.to_string(),
                agent_name: "explore".to_string(),
                backend: None,
                model: "qwen3.7-max".to_string(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: "explicit override".to_string(),
                inherit: InheritOptions::default(),
                abort_handle: None,
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
                started_at: Instant::now(),
            },
        );
    }

    let output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_wait(&serde_json::json!({
                "task_ids": [ready_task_id, pending_task_id],
                "wait_policy": "any",
                "timeout_secs": 30,
            }))
        })
        .expect("task_wait(any) should return the ready result");

    assert!(output.contains("first result"));
    assert!(!output.contains("task_wait PARKED"));
    assert!(!output.contains("task_wait BUDGET ELAPSED"));
    assert!(remove_task_entry(&ready_task_id).is_none());
    assert!(remove_task_entry(&pending_task_id).is_some());

    let mut os = app.os.lock().unwrap();
    os.set_current_pid(Some(root_pid));
    os.kill_process(pending_pid, "test cleanup".to_string())
        .unwrap();
    let _ = os.channel_close(None, ChannelId(pending_channel));
    let _ = os.channel_destroy(None, ChannelId(pending_channel));
    let _ = os.futex_destroy(pending_futex);
    drop(os);
}

#[test]
fn task_status_cleans_up_terminated_task_without_result() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (root_pid, pid, result_channel_id, completion_futex_addr) = {
        let mut os = app.os.lock().unwrap();
        let root_pid = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let pid = os
            .spawn(
                Some(root_pid),
                "terminated".to_string(),
                "terminated goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create(Some(pid), 1, "task-result".to_string());
        let completion_futex = os.futex_create(0, "task-complete".to_string());
        os.set_current_pid(Some(root_pid));
        os.kill_process(pid, "subagent crashed".to_string())
            .unwrap();
        (root_pid, pid, channel.raw(), completion_futex)
    };

    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid: root_pid,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "terminated task".to_string(),
            agent_name: "build".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    let output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_status(&serde_json::json!({}))
        })
        .expect("task_status should collect terminated task");

    assert!(output.contains("terminated without publishing any output"));
    assert!(output.contains("task_retry"));
    assert!(output.contains(&task_id));
    assert!(remove_task_entry(&task_id).is_none());
    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());
    assert!(os.get_process(pid).is_none());
    drop(os);
}

#[test]
fn task_status_reuses_persisted_terminal_evidence_after_channel_loss() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let root = std::env::temp_dir().join(format!(
        "task-status-evidence-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let history_file = root.join("history.sqlite");
    let session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    crate::ai::history::record_delivered_task_evidence(
        &history_file,
        &session_id,
        crate::ai::history::DeliveredTaskEvidence {
            task_id: &task_id,
            description: "terminated task",
            agent_name: "build",
            model: "qwen3.7-max",
            status: "completed",
            payload: "[Subagent final answer]\ndurable output",
        },
    )
    .unwrap();

    for terminated in [true, false] {
        let mut app = test_app_with_model("qwen3.7-max".to_string());
        app.config.history_file = history_file.clone();
        app.session_id = session_id.clone();
        let os_handle = app.os.clone();
        let _os_guard = init_os_tools_guard(os_handle.clone());

        let (root_pid, pid, result_channel_id, completion_futex_addr) = {
            let mut os = os_handle.lock().unwrap();
            let root_pid = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
            let pid = os
                .spawn(
                    Some(root_pid),
                    "terminated".to_string(),
                    "terminated goal".to_string(),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();
            let channel = os.channel_create(Some(pid), 1, "task-result".to_string());
            let completion_futex = os.futex_create(0, "task-complete".to_string());
            os.set_current_pid(Some(root_pid));
            if terminated {
                os.kill_process(pid, "subagent crashed".to_string())
                    .unwrap();
            }
            (root_pid, pid, channel.raw(), completion_futex)
        };

        insert_task_entry_for_test(
            task_id.clone(),
            AsyncTaskEntry {
                evidence_history_file: None,
                session_id: app.session_id.clone(),
                owner_pid: root_pid,
                pid,
                result_channel_id,
                completion_futex_addr,
                description: "terminated task".to_string(),
                agent_name: "build".to_string(),
                backend: None,
                model: "qwen3.7-max".to_string(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: "explicit override".to_string(),
                inherit: InheritOptions::default(),
                abort_handle: None,
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
                started_at: Instant::now(),
            },
        );

        let context = DriverContext::new(
            app,
            Arc::new(std::sync::Mutex::new(McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        let output = DRIVER_CTX
            .sync_scope(context, || {
                crate::ai::driver::runtime_ctx::TURN_IDENTITY
                    .sync_scope((session_id.clone(), 0usize), || {
                        execute_task_status(&serde_json::json!({}))
                    })
            })
            .expect("task_status should reuse persisted evidence");

        assert!(output.contains("durable output"), "{output}");
        assert!(
            !output.contains("terminated without publishing any output"),
            "{output}"
        );
        assert!(remove_task_entry(&task_id).is_none());

        let os = os_handle.lock().unwrap();
        assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
        assert!(os.futex_event_id(completion_futex_addr).is_none());
        drop(os);
    }
    let sessions_root = crate::ai::history::SessionStore::new(&history_file)
        .sessions_root()
        .to_path_buf();
    let _ = std::fs::remove_dir_all(sessions_root);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn task_retry_rejects_a_different_parent_process() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let owner_pid = {
        let mut os = app.os.lock().unwrap();
        let owner_pid = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        os.set_current_pid(Some(owner_pid));
        owner_pid
    };
    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    register_retry_spec(
        &task_id,
        app.session_id.clone(),
        owner_pid.saturating_add(1),
        prepared_retry_task(),
        task_id.clone(),
    );

    let error = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_retry(&serde_json::json!({ "task_id": task_id }))
        })
        .expect_err("a sibling process must not retry another parent's task");
    assert!(error.contains("different parent process"));

    discard_tasks_for_session(&app.session_id);
}

#[test]
fn discard_session_removes_retry_specs_after_active_tasks_are_gone() {
    let session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    register_retry_spec(
        &task_id,
        session_id.clone(),
        1,
        prepared_retry_task(),
        task_id.clone(),
    );
    assert!(
        TASK_RETRY_REGISTRY
            .lock()
            .unwrap()
            .get_ref(&task_id)
            .is_some()
    );

    discard_tasks_for_session(&session_id);

    assert!(
        TASK_RETRY_REGISTRY
            .lock()
            .unwrap()
            .get_ref(&task_id)
            .is_none()
    );
}

#[test]
fn discard_session_clears_task_wait_noop_counts() {
    let session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let key = task_wait_key(&session_id, 1, &WaitPolicy::All, &["task-1".to_string()]);
    assert_eq!(bump_task_wait_noop_count(&key), 1);

    discard_tasks_for_session(&session_id);

    assert!(
        TASK_WAIT_NOOP_COUNTS.lock().unwrap().get(&key).is_none(),
        "session deletion must drop the dead no-op wait counter"
    );
}

#[tokio::test]
async fn task_cancel_aborts_worker_and_leaves_result_collectable_via_task_wait() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (root, pid, result_channel_id, completion_futex_addr) = {
        let mut os = app.os.lock().unwrap();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let pid = os
            .spawn(
                Some(root),
                "child".to_string(),
                "goal".to_string(),
                20,
                8,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create_tagged_with_holders(
            Some(root),
            1,
            "task-result".to_string(),
            aios_kernel::primitives::ChannelOwnerTag::TaskResult,
            vec![
                "task_result.producer".to_string(),
                "task_result.consumer".to_string(),
            ],
        );
        let completion_futex = os.futex_create(0, "task-complete".to_string());
        (root, pid, channel.raw(), completion_futex)
    };

    let worker = tokio::spawn(std::future::pending::<()>());
    let abort_handle = worker.abort_handle();
    let cancel_stream = Arc::new(AtomicBool::new(false));
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid: root,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "cancel branch".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: Some(abort_handle),
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: cancel_stream.clone(),
            started_at: Instant::now(),
        },
    );

    let cancel_output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_cancel(&serde_json::json!({ "task_ids": [task_id.clone()] }))
        })
        .expect("task_cancel should succeed");
    assert!(
        worker
            .await
            .expect_err("task_cancel must stop the Tokio worker")
            .is_cancelled()
    );
    assert!(cancel_stream.load(std::sync::atomic::Ordering::Acquire));
    assert!(cancel_output.contains("Required next step: collect these terminal results"));
    assert!(
        crate::ai::tools::task_tools::with_task_entry(&task_id, |_| ()).is_some(),
        "cancelled task must stay collectable until task_wait/task_status consumes it"
    );

    let wait_output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_wait(&serde_json::json!({ "task_ids": [task_id.clone()] }))
        })
        .expect("task_wait should collect cancelled result");

    assert!(wait_output.contains("CANCELLED"));
    assert!(wait_output.contains("Failure reason: cancelled by parent agent"));
    assert!(remove_task_entry(&task_id).is_none());

    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());
    assert!(os.get_process(pid).is_none());
}

#[tokio::test]
async fn discarding_session_tasks_aborts_workers_and_releases_results() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (root, pid, result_channel_id, completion_futex_addr) = {
        let mut os = app.os.lock().unwrap();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let pid = os
            .spawn(
                Some(root),
                "child".to_string(),
                "goal".to_string(),
                20,
                8,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create_tagged_with_holders(
            Some(root),
            1,
            "task-result".to_string(),
            aios_kernel::primitives::ChannelOwnerTag::TaskResult,
            vec![
                "task_result.producer".to_string(),
                "task_result.consumer".to_string(),
            ],
        );
        let completion_futex = os.futex_create(0, "task-complete".to_string());
        (root, pid, channel.raw(), completion_futex)
    };

    let worker = tokio::spawn(std::future::pending::<()>());
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid: root,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "deleted session branch".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: Some(worker.abort_handle()),
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    discard_tasks_for_session(&app.session_id);

    assert!(
        worker
            .await
            .expect_err("session deletion must stop its Tokio worker")
            .is_cancelled()
    );
    assert!(remove_task_entry(&task_id).is_none());
    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());
    assert!(os.get_process(pid).is_none());
    drop(os);
}

#[tokio::test]
async fn wall_clock_reaper_aborts_worker_and_leaves_timeout_result_collectable() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (root, pid, result_channel_id, completion_futex_addr) = {
        let mut os = app.os.lock().unwrap();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let pid = os
            .spawn(
                Some(root),
                "child".to_string(),
                "goal".to_string(),
                20,
                8,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create_tagged_with_holders(
            Some(root),
            1,
            "task-result".to_string(),
            aios_kernel::primitives::ChannelOwnerTag::TaskResult,
            vec![
                "task_result.producer".to_string(),
                "task_result.consumer".to_string(),
            ],
        );
        let completion_futex = os.futex_create(0, "task-complete".to_string());
        (root, pid, channel.raw(), completion_futex)
    };

    let worker = tokio::spawn(std::future::pending::<()>());
    let abort_handle = worker.abort_handle();
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid: root,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "timeout branch".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: Some(abort_handle),
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now() - SUBAGENT_WALL_CLOCK_TIMEOUT - Duration::from_secs(1),
        },
    );

    reap_timed_out_subagents();
    assert!(
        worker
            .await
            .expect_err("wall-clock reaper must stop the Tokio worker")
            .is_cancelled()
    );

    let wait_output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_wait(&serde_json::json!({ "task_ids": [task_id.clone()] }))
        })
        .expect("task_wait should collect timeout result");
    assert!(wait_output.contains("TIMEOUT"));
    assert!(wait_output.contains("exceeded wall-clock lifetime"));
    assert!(remove_task_entry(&task_id).is_none());

    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());
    assert!(os.get_process(pid).is_none());
}

#[tokio::test]
async fn task_wait_wall_clock_timeout_aborts_worker_before_publishing_result() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (root, pid, result_channel_id, completion_futex_addr) = {
        let mut os = app.os.lock().unwrap();
        let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 8, None);
        let pid = os
            .spawn(
                Some(root),
                "child".to_string(),
                "goal".to_string(),
                20,
                8,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create_tagged_with_holders(
            Some(root),
            1,
            "task-result".to_string(),
            aios_kernel::primitives::ChannelOwnerTag::TaskResult,
            vec![
                "task_result.producer".to_string(),
                "task_result.consumer".to_string(),
            ],
        );
        let completion_futex = os.futex_create(0, "task-complete".to_string());
        (root, pid, channel.raw(), completion_futex)
    };

    let worker = tokio::spawn(std::future::pending::<()>());
    let cancel_stream = Arc::new(AtomicBool::new(false));
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid: root,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "timeout in task_wait".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: Some(worker.abort_handle()),
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: cancel_stream.clone(),
            started_at: Instant::now() - SUBAGENT_WALL_CLOCK_TIMEOUT - Duration::from_secs(1),
        },
    );

    let wait_output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_wait(&serde_json::json!({ "task_ids": [task_id.clone()] }))
        })
        .expect("task_wait should collect its timeout result");

    assert!(
        worker
            .await
            .expect_err("task_wait wall-clock timeout must stop the Tokio worker")
            .is_cancelled()
    );
    assert!(cancel_stream.load(std::sync::atomic::Ordering::Acquire));
    assert!(wait_output.contains("TIMEOUT"));
    assert!(remove_task_entry(&task_id).is_none());

    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());
    drop(os);
}

#[test]
fn subagent_result_tools_are_never_lossy_and_gate_evidence_pruning_on_integration() {
    // Evidence-bearing results become unloadable only once the durable ledger
    // records their task as integrated; without that session fact they are live
    // state. The integration call itself is a plain acknowledgement and stays
    // outside the pruning protocol entirely.
    let gated = [
        "task_wait",
        "task_status",
        "task_retry",
        "task_spawn",
        "task_evidence_read",
    ];
    for name in gated {
        let policy = tool_history_policy(name);
        assert!(
            !policy.allows_lossy_compress(),
            "{name} result must not be lossy-compressed"
        );
        assert!(
            !policy.allows_prune(),
            "{name} must not be prunable without consulting the evidence ledger"
        );
        assert_eq!(
            policy.prune_policy(),
            crate::ai::tools::registry::common::ToolPrunePolicy::AfterIntegration,
            "{name} must gate pruning on integration"
        );
    }
    assert_eq!(
        tool_history_policy("task_integrate").prune_policy(),
        crate::ai::tools::registry::common::ToolPrunePolicy::Never
    );
}

#[test]
fn encoded_task_goal_prefix_is_detectable_without_decoding() {
    let goal = encode_os_task_goal(&OsTaskGoal {
        task_id: "task_test".to_string(),
        result_channel_id: 7,
        completion_futex_addr: 9,
        description: "inspect".to_string(),
        prompt: "look around".to_string(),
        agent_name: "explore".to_string(),
        model: "qwen3.7-max-alibaba".to_string(),
        is_model_auto_selected: false,
        auto_model_fallback: None,
        selection_explanation: "explicit agent/model override".to_string(),
        spawn_depth: 0,
        parent_agent_team_mode: false,
        response_schema: None,
        backend: None,
        backend_model: None,
        backend_continue_from: None,
        active_skill_names: Vec::new(),
        image_files: Vec::new(),
    })
    .unwrap();

    assert!(is_encoded_task_goal(&goal));
    assert!(!is_encoded_task_goal("plain foreground goal"));
}

#[test]
fn kernel_parent_policy_is_frozen_at_spawn() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let mut app = crate::ai::middleware::test_util::test_app();
    app.session_id = format!("kernel-parent-policy-{}", uuid::Uuid::new_v4().simple());
    app.os
        .lock()
        .unwrap()
        .begin_foreground("parent".to_string(), "goal".to_string(), 10, 8, None);
    let context = DriverContext::new(
        app.clone(),
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    let prepared = prepared_retry_task();
    crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope((app.session_id.clone(), 0), || {
        DRIVER_CTX.sync_scope(context, || {
            for team_parent in [true, false] {
                crate::ai::driver::runtime_ctx::set_active_skill_names(if team_parent {
                    vec!["AgEnT-TeAm".to_string()]
                } else {
                    Vec::new()
                });
                let spawned = super::spawn_subagent_kernel_task(&prepared).unwrap();
                crate::ai::driver::runtime_ctx::set_active_skill_names(if team_parent {
                    Vec::new()
                } else {
                    vec!["agent-team".to_string()]
                });
                let goal = {
                    let os = app.os.lock().unwrap();
                    super::decode_os_task_goal(&os.get_process(spawned.pid).unwrap().goal).unwrap()
                };
                assert_eq!(goal.parent_agent_team_mode, team_parent);
                assert!(!SubagentBackend::from_stored(goal.backend.as_deref()).is_traecli());
                let mut legacy = serde_json::to_value(&goal).unwrap();
                legacy
                    .as_object_mut()
                    .unwrap()
                    .remove("parent_agent_team_mode");
                let legacy: OsTaskGoal = serde_json::from_value(legacy).unwrap();
                assert!(!legacy.parent_agent_team_mode);
            }
            discard_tasks_for_session(&app.session_id);
        });
    });
}

#[test]
fn task_entry_can_be_looked_up_by_pid() {
    let task_id = format!("task_test_{}", uuid::Uuid::new_v4().simple());
    {
        let mut registry = TASK_REGISTRY.lock().unwrap();
        registry.insert(
            task_id.clone(),
            AsyncTaskEntry {
                evidence_history_file: None,
                session_id: String::new(),
                owner_pid: 0,
                pid: 4242,
                result_channel_id: 11,
                completion_futex_addr: FutexAddr(13),
                description: "inspect".to_string(),
                agent_name: "explore".to_string(),
                backend: None,
                model: "qwen3.7-max-alibaba".to_string(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: "explicit override".to_string(),
                inherit: InheritOptions::default(),
                abort_handle: None,
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
                started_at: Instant::now(),
            },
        );
    }

    let looked_up = with_task_entry_by_pid(4242, |entry| {
        (
            entry.agent_name.clone(),
            entry.model.clone(),
            entry.result_channel_id,
        )
    });

    assert_eq!(
        looked_up,
        Some(("explore".to_string(), "qwen3.7-max-alibaba".to_string(), 11))
    );

    let _ = remove_task_entry(&task_id);
}

#[test]
fn outstanding_task_anchor_lists_ids_and_required_follow_up() {
    let note = render_outstanding_task_anchor(&[
        OutstandingTaskSnapshot {
            task_id: "task_alpha".to_string(),
            status: "running".to_string(),
            agent_name: "explore".to_string(),
            model: "qwen3.7-max".to_string(),
            reasoning_effort: Some("high".to_string()),
            description: "inspect parser".to_string(),
            progress: Some("running command · plan: inspect parser".to_string()),
        },
        OutstandingTaskSnapshot {
            task_id: "task_beta".to_string(),
            status: "completed".to_string(),
            agent_name: "build".to_string(),
            model: "gpt-5.5".to_string(),
            reasoning_effort: None,
            description: "verify fix".to_string(),
            progress: None,
        },
    ]);

    assert!(note.contains("[pending-subagent-tasks]"));
    assert!(note.contains("Outstanding task_ids: [task_alpha, task_beta]"));
    assert!(note.contains("task_id=task_alpha status=running"));
    assert!(note.contains("model=qwen3.7-max effort=high desc=inspect parser"));
    assert!(note.contains("progress: running command · plan: inspect parser"));
    assert!(note.contains("task_id=task_beta status=completed"));
    assert!(note.contains("model=gpt-5.5 desc=verify fix"));
    assert!(note.contains("use `task_wait` with the same task_ids"));
}

#[test]
fn subagent_progress_update_never_wakes_parked_parent() {
    // 回归 agent-team fan-out 卡死：进度事件曾用 wake_process 把已 park 的父代理拨回
    // Ready，逼它每 15s 跑一整轮模型只为再 park 一次（前台忙等自旋）。修复后进度
    // 事件只刷新状态行，绝不改变父代理调度态。
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (owner_pid, child_pid, channel, futex) = {
        let mut os = app.os.lock().unwrap();
        let owner_pid = os.begin_foreground("owner".to_string(), "goal".to_string(), 10, 8, None);
        let child_pid = os
            .spawn(
                Some(owner_pid),
                "pending-child".to_string(),
                "pending goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create(Some(child_pid), 1, "pending-result".to_string());
        let futex = os.futex_create(0, "pending-complete".to_string());
        // 让 owner 真正进入 Waiting：等待一个永不完成的独立事件（模拟 task_wait park）。
        os.set_current_pid(Some(owner_pid));
        let park_event = os.futex_event_id(futex).expect("futex has an event id");
        os.wait_on_events(vec![park_event], WaitPolicy::All, None)
            .expect("owner should park on the pending event");
        (owner_pid, child_pid, channel.raw(), futex)
    };

    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid,
            pid: child_pid,
            result_channel_id: channel,
            completion_futex_addr: futex,
            description: "pending task".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    // 首次进度事件（未节流，一定 notify）：修复后只应刷新状态行，不得动 owner。
    record_subagent_progress_update(
        &task_id,
        &progress_snapshot(1, 5, "calling model"),
        SubagentProgressEventKind::Phase,
    );

    {
        let os = app.os.lock().unwrap();
        let proc = os
            .get_process(owner_pid)
            .expect("owner process should still exist");
        assert!(
            matches!(
                proc.state,
                aios_kernel::kernel::ProcessState::Waiting { .. }
            ),
            "progress update must NOT flip a parked parent to Ready (was {:?})",
            proc.state
        );
        assert!(
            !proc
                .mailbox
                .iter()
                .any(|message| message.contains("[TASK_PROGRESS]")),
            "progress update must not inject a wake-up mailbox message into a parked parent"
        );
    }

    assert!(remove_task_entry(&task_id).is_some());
    let mut os = app.os.lock().unwrap();
    os.set_current_pid(Some(owner_pid));
    os.kill_process(child_pid, "test cleanup".to_string())
        .unwrap();
    drop(os);
}

#[test]
fn wake_expired_task_waits_reaps_orphaned_state_when_owner_gone() {
    // 回归「expired 未回收」缺口：owner 已终止但 wait state 仍在，此前 wake_process
    // 落空、状态以 expired=true 永久滞留（内存泄漏 + 再无清理路径）。修复后 waker
    // 检测到 owner 已不存在/终止即删除该孤儿状态，自愈。
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (supervisor_pid, owner_pid, child_pid, channel, futex) = {
        let mut os = app.os.lock().unwrap();
        // supervisor 作为 foreground root，owner 挂在它下面——这样 supervisor 可以
        // 合法终止 owner（kill_process 只允许终止自己的后代），模拟父代理异常退出。
        let supervisor_pid =
            os.begin_foreground("supervisor".to_string(), "goal".to_string(), 10, 8, None);
        let owner_pid = os
            .spawn(
                Some(supervisor_pid),
                "owner".to_string(),
                "owner goal".to_string(),
                20,
                8,
                None,
                None,
            )
            .unwrap();
        let child_pid = os
            .spawn(
                Some(owner_pid),
                "pending-child".to_string(),
                "pending goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create(Some(child_pid), 1, "pending-result".to_string());
        let futex = os.futex_create(0, "pending-complete".to_string());
        // execute_task_wait 按 current pid 记账 owner；切到 owner 建立 wait state。
        os.set_current_pid(Some(owner_pid));
        (supervisor_pid, owner_pid, child_pid, channel.raw(), futex)
    };

    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid,
            pid: child_pid,
            result_channel_id: channel,
            completion_futex_addr: futex,
            description: "pending task".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    // 建立一条 parked wait state，然后强制过期。
    let parked = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_wait(&serde_json::json!({
                "task_ids": [task_id.clone()],
                "timeout_secs": 1,
            }))
        })
        .expect("first task_wait should park");
    assert!(parked.contains("[task_wait PARKED]"));
    assert!(
        task_wait_state_count_for_test() >= 1,
        "parked task_wait must record a wait state"
    );

    // owner 终止：模拟父代理在 park 后异常退出，留下孤儿 wait state。
    // kill_process 只允许终止自己的后代，故从 supervisor（owner 的父）发起终止。
    {
        let mut os = app.os.lock().unwrap();
        os.set_current_pid(Some(supervisor_pid));
        os.kill_process(owner_pid, "owner gone".to_string())
            .unwrap();
    }

    expire_task_wait_states_for_test();
    wake_expired_task_waits();

    assert_eq!(
        task_wait_state_count_for_test(),
        0,
        "orphaned wait state (terminated owner) must be reaped, not left lingering"
    );

    let _ = remove_task_entry(&task_id);
}

#[test]
fn side_note_target_validation_accepts_stopped_subagent() {
    // Regression: SIGSTOP pauses a subagent (Stopped) but does not finish it; a
    // side-note sent during the pause must queue normally and become visible
    // after SIGCONT instead of being rejected as "already finished".
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (owner_pid, child_pid, channel, futex) = {
        let mut os = app.os.lock().unwrap();
        let owner_pid = os.begin_foreground("owner".to_string(), "goal".to_string(), 10, 8, None);
        let child_pid = os
            .spawn(
                Some(owner_pid),
                "stopped-child".to_string(),
                "stopped goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create(Some(child_pid), 1, "stopped-result".to_string());
        let futex = os.futex_create(0, "stopped-complete".to_string());
        os.set_current_pid(Some(owner_pid));
        (owner_pid, child_pid, channel.raw(), futex)
    };
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid,
            pid: child_pid,
            result_channel_id: channel,
            completion_futex_addr: futex,
            description: "stopped task".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    let validate = || {
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .sync_scope((app.session_id.clone(), 0usize), || {
                super::validate_side_note_target(&task_id)
            })
    };

    // SIGSTOP: still pending, side-notes must queue.
    {
        let mut os = app.os.lock().unwrap();
        os.set_current_pid(Some(owner_pid));
        os.signal_process(child_pid, aios_kernel::kernel::Signal::SigStop)
            .unwrap();
        assert!(super::is_task_pending(os.as_mut(), child_pid).unwrap());
        os.set_current_pid(Some(owner_pid));
    }
    assert!(validate().is_ok());

    // SIGCONT: resumed, still accepted.
    {
        let mut os = app.os.lock().unwrap();
        os.set_current_pid(Some(owner_pid));
        os.signal_process(child_pid, aios_kernel::kernel::Signal::SigCont)
            .unwrap();
        assert!(super::is_task_pending(os.as_mut(), child_pid).unwrap());
        os.set_current_pid(Some(owner_pid));
    }
    assert!(validate().is_ok());

    // Terminated: genuinely finished, side-notes must be refused.
    {
        let mut os = app.os.lock().unwrap();
        os.set_current_pid(Some(owner_pid));
        os.kill_process(child_pid, "test cleanup".to_string())
            .unwrap();
        assert!(!super::is_task_pending(os.as_mut(), child_pid).unwrap());
        os.set_current_pid(Some(owner_pid));
    }
    let err = validate().expect_err("terminated task must reject side-notes");
    assert!(err.contains("already finished"), "unexpected error: {err}");

    assert!(remove_task_entry(&task_id).is_some());
}

/// A targeted side-note must land in the queue the subagent drains, which is
/// the child's own history file, not the lead's.
#[test]
fn side_note_target_resolves_child_history_file() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (owner_pid, child_pid, channel, futex) = {
        let mut os = app.os.lock().unwrap();
        let owner_pid = os.begin_foreground("owner".to_string(), "goal".to_string(), 10, 8, None);
        let child_pid = os
            .spawn(
                Some(owner_pid),
                "child".to_string(),
                "child goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create(Some(child_pid), 1, "child-result".to_string());
        let futex = os.futex_create(0, "child-complete".to_string());
        os.set_current_pid(Some(owner_pid));
        (owner_pid, child_pid, channel.raw(), futex)
    };
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid,
            pid: child_pid,
            result_channel_id: channel,
            completion_futex_addr: futex,
            description: "child task".to_string(),
            agent_name: "explore".to_string(),
            backend: None,
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    let parent_history = std::path::Path::new("/sessions/lead.sqlite");
    let (targeted, foreground) = crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        (app.session_id.clone(), 0usize),
        || {
            (
                resolve_side_note_history_file(&task_id, parent_history),
                resolve_side_note_history_file("foreground", parent_history),
            )
        },
    );
    assert_eq!(
        targeted.unwrap(),
        Some(std::path::PathBuf::from(format!(
            "/sessions/lead.proc-{child_pid}.sqlite"
        )))
    );
    assert_eq!(foreground.unwrap(), None);

    assert!(remove_task_entry(&task_id).is_some());
}

/// TraeCLI accepts deferred notes in the parent's queue and exposes their
/// pending count while retaining the normal ownership and terminal checks.
#[test]
fn side_note_target_queues_external_backend_for_continuation() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (owner_pid, child_pid, channel, futex) = {
        let mut os = app.os.lock().unwrap();
        let owner_pid = os.begin_foreground("owner".to_string(), "goal".to_string(), 10, 8, None);
        let child_pid = os
            .spawn(
                Some(owner_pid),
                "external-child".to_string(),
                "external goal".to_string(),
                20,
                4,
                None,
                None,
            )
            .unwrap();
        let channel = os.channel_create(Some(child_pid), 1, "external-result".to_string());
        let futex = os.futex_create(0, "external-complete".to_string());
        os.set_current_pid(Some(owner_pid));
        (owner_pid, child_pid, channel.raw(), futex)
    };
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            evidence_history_file: None,
            session_id: app.session_id.clone(),
            owner_pid,
            pid: child_pid,
            result_channel_id: channel,
            completion_futex_addr: futex,
            description: "external task".to_string(),
            agent_name: "build".to_string(),
            backend: Some(SubagentBackend::TraeCli),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    let root = std::env::temp_dir().join(format!("traecli-side-note-{task_id}"));
    app.session_history_file = root.join("lead.sqlite");
    let parent_history = app.session_history_file.clone();
    let context = DriverContext::from_app_snapshot(
        &app,
        Arc::new(std::sync::Mutex::new(McpClient::new())),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        (app.session_id.clone(), 0usize),
        || {
            DRIVER_CTX.sync_scope(context, || {
                assert_eq!(
                    resolve_side_note_history_file(&task_id, &parent_history).unwrap(),
                    Some(parent_history.clone())
                );
                let send = |note: &str| {
                    crate::ai::tools::service::side_note::execute_send_side_note(
                        &serde_json::json!({"task_id": task_id, "note": note}),
                    )
                };
                assert!(send(" ").unwrap_err().contains("empty"));
                assert!(send(&"x".repeat(8001)).unwrap_err().contains("8000"));
                let output = send("Keep the parser behavior.").unwrap();
                assert!(output.contains("backend_continue_from"), "{output}");
                assert!(output.contains("does not interrupt"), "{output}");
                assert!(
                    output.contains("retained after completion or cancellation"),
                    "{output}"
                );
                let queue =
                    crate::ai::driver::side_note::side_note_file(&parent_history, Some(&task_id));
                assert!(queue.exists());
                let status = execute_task_status(&serde_json::json!({})).unwrap();
                assert!(status.contains("pending side-notes: 1"), "{status}");
                crate::ai::driver::runtime_ctx::set_active_skill_names(vec!["agent-team".into()]);
                let quiet_status = execute_task_status(&serde_json::json!({})).unwrap();
                assert!(
                    quiet_status.contains("pending side-notes: 1"),
                    "{quiet_status}"
                );
                {
                    let mut os = app.os.lock().unwrap();
                    os.set_current_pid(Some(owner_pid));
                    os.kill_process(child_pid, "test cancellation".into())
                        .unwrap();
                    os.set_current_pid(Some(owner_pid));
                }
                let error = send("New guidance").unwrap_err();
                assert!(error.contains("already finished"), "{error}");
                assert!(error.contains("backend_continue_from"), "{error}");
                assert!(queue.exists(), "cancellation must retain queued notes");
                assert!(remove_task_entry(&task_id).is_some());
                assert!(queue.exists(), "collection must retain queued notes");
                let queued = std::fs::read_to_string(&queue).unwrap();
                assert!(
                    super::prepare_traecli_continuation(
                        Some(&task_id),
                        &parent_history,
                        "Continue the task."
                    )
                    .is_err()
                );
                assert_eq!(std::fs::read_to_string(&queue).unwrap(), queued);
                super::record_traecli_session(&task_id, Some(uuid::Uuid::new_v4().to_string()));
                let (resume_session, prompt, side_note_files) =
                    super::prepare_traecli_continuation(
                        Some(&task_id),
                        &parent_history,
                        "Continue the task.",
                    )
                    .unwrap();
                assert!(prompt.contains("Keep the parser behavior."), "{prompt}");
                assert_eq!(std::fs::read_to_string(&queue).unwrap(), queued);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;

                    let spec = super::ExternalSubagentSpec {
                        prompt: prompt.clone(),
                        cwd: root.clone(),
                        workspace_allowed: true,
                        model: Some("test-model".into()),
                        skills: None,
                        resume_session,
                        side_note_files,
                        reasoning_effort: None,
                        image_files: Vec::new(),
                        phase_slot: None,
                    };
                    let error = super::traecli::run_traecli_blocking_with(
                        &spec,
                        Arc::new(AtomicBool::new(false)),
                        root.join("absent-traecli").to_str().unwrap(),
                        Duration::from_secs(2),
                        Vec::new(),
                    )
                    .unwrap_err();
                    assert!(error.contains("failed to spawn traecli"), "{error}");
                    assert_eq!(std::fs::read_to_string(&queue).unwrap(), queued);

                    let script = root.join("traecli.sh");
                    std::fs::write(
                        &script,
                        format!(
                            r#"#!/bin/sh
cat > "$0.prompt"
test ! -e "lead.assets/side_notes/{task_id}.jsonl" || exit 1
/bin/echo '{{"type":"item.completed","item":{{"type":"agent_message","text":"ok"}}}}'
"#
                        ),
                    )
                    .unwrap();
                    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                        .unwrap();
                    let outcome = super::traecli::run_traecli_blocking_with(
                        &spec,
                        Arc::new(AtomicBool::new(false)),
                        script.to_str().unwrap(),
                        Duration::from_secs(2),
                        Vec::new(),
                    )
                    .unwrap();
                    assert_eq!(outcome.result, "ok");
                    assert!(!queue.exists(), "successful spawn must clear queued notes");
                    assert_eq!(
                        std::fs::read_to_string(root.join("traecli.sh.prompt")).unwrap(),
                        prompt
                    );
                }
                super::traecli::forget_traecli_session(&task_id);
            });
        },
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// After the wait budget expires, `wait_policy=any` must not hide tasks that
/// are still running: the caller was woken for the timeout, so the reply has to
/// carry both the ready results and the pending ids.
#[test]
fn task_wait_any_after_budget_reports_ready_and_pending() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    let _os_guard = init_os_tools_guard(app.os.clone());

    let task_done = format!("task_{}", uuid::Uuid::new_v4().simple());
    let task_pending = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (owner_pid, done_pid, done_channel, pending_pid) = {
        let mut os = app.os.lock().unwrap();
        let owner_pid = os.begin_foreground("owner".to_string(), "goal".to_string(), 10, 8, None);
        let mut spawn_child = |name: &str| {
            let pid = os
                .spawn(
                    Some(owner_pid),
                    name.to_string(),
                    format!("{name} goal"),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();
            let channel = os.channel_create(Some(pid), 1, format!("{name}-result"));
            let futex = os.futex_create(0, format!("{name}-complete"));
            (pid, channel.raw(), futex)
        };
        let (done_pid, done_channel, done_futex) = spawn_child("done-child");
        let (pending_pid, pending_channel, pending_futex) = spawn_child("pending-child");
        os.set_current_pid(Some(owner_pid));
        for (task_id, pid, channel, futex, name) in [
            (
                task_done.clone(),
                done_pid,
                done_channel,
                done_futex,
                "done",
            ),
            (
                task_pending.clone(),
                pending_pid,
                pending_channel,
                pending_futex,
                "pending",
            ),
        ] {
            insert_task_entry_for_test(
                task_id,
                AsyncTaskEntry {
                    evidence_history_file: None,
                    session_id: app.session_id.clone(),
                    owner_pid,
                    pid,
                    result_channel_id: channel,
                    completion_futex_addr: futex,
                    description: format!("{name} task"),
                    agent_name: "explore".to_string(),
                    backend: None,
                    model: "qwen3.7-max".to_string(),
                    is_model_auto_selected: false,
                    auto_model_fallback: None,
                    selection_explanation: "explicit override".to_string(),
                    inherit: InheritOptions::default(),
                    abort_handle: None,
                    last_progress_notification_at: None,
                    last_progress_persisted_at: None,
                    cancel_stream: Arc::new(AtomicBool::new(false)),
                    started_at: Instant::now(),
                },
            );
        }
        (owner_pid, done_pid, done_channel, pending_pid)
    };

    let task_ids = [task_done.clone(), task_pending.clone()];
    let parked = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_wait(&serde_json::json!({
                "task_ids": task_ids.clone(),
                "timeout_secs": 1,
                "wait_policy": "any",
            }))
        })
        .expect("first wait should park");
    assert!(parked.contains("[task_wait PARKED]"));

    expire_task_wait_states_for_test();
    {
        let mut os = app.os.lock().unwrap();
        // The parked wait leaves the owner suspended; restore the active pid the
        // tool layer reads before the follow-up call.
        os.set_current_pid(Some(owner_pid));
        os.channel_send(
            Some(done_pid),
            aios_kernel::primitives::ChannelId(done_channel),
            serde_json::json!({
                "status": "completed",
                "output": "done child answer",
                "error": null,
            })
            .to_string(),
        )
        .unwrap();
    }

    let elapsed = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_wait(&serde_json::json!({
                "task_ids": task_ids.clone(),
                "timeout_secs": 1,
                "wait_policy": "any",
            }))
        })
        .expect("expired wait should report ready plus pending");
    assert!(
        elapsed.contains("done child answer"),
        "ready result must not be dropped: {elapsed}"
    );
    assert!(
        elapsed.contains("[task_wait BUDGET ELAPSED]"),
        "pending task must stay visible: {elapsed}"
    );
    assert!(
        elapsed.contains(&task_pending),
        "pending id must be reported: {elapsed}"
    );

    assert!(remove_task_entry(&task_done).is_none());
    assert!(remove_task_entry(&task_pending).is_some());
    let mut os = app.os.lock().unwrap();
    os.set_current_pid(Some(owner_pid));
    let _ = os.kill_process(done_pid, "test cleanup".to_string());
    let _ = os.kill_process(pending_pid, "test cleanup".to_string());
}

struct ParentCheckpointTask {
    kernel: aios_kernel::SharedKernel,
    session_id: String,
    task_id: String,
    owner_pid: u64,
    child_pid: u64,
    channel: u64,
    futex: FutexAddr,
    slot: crate::ai::driver::runtime_ctx::SubagentPhaseSlot,
    _os_guard: OsToolsGlobalGuard,
}

impl ParentCheckpointTask {
    fn new() -> Self {
        let app = test_app_with_model(String::new());
        let kernel = app.os.clone();
        let os_guard = init_os_tools_guard(kernel.clone());
        let session_id = format!("checkpoint-session-{}", uuid::Uuid::new_v4().simple());
        let task_id = format!("checkpoint-task-{}", uuid::Uuid::new_v4().simple());
        let (owner_pid, child_pid, channel, futex) = {
            let mut os = kernel.lock().unwrap();
            let owner = os.begin_foreground("owner".into(), "goal".into(), 10, 8, None);
            let child = os
                .spawn(
                    Some(owner),
                    "child".into(),
                    "goal".into(),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();
            let channel = os.channel_create(Some(child), 1, "checkpoint-result".into());
            let futex = os.futex_create(0, "checkpoint-complete".into());
            os.set_current_pid(Some(owner));
            (owner, child, channel.raw(), futex)
        };
        insert_task_entry_for_test(
            task_id.clone(),
            AsyncTaskEntry {
                evidence_history_file: None,
                session_id: session_id.clone(),
                owner_pid,
                pid: child_pid,
                result_channel_id: channel,
                completion_futex_addr: futex,
                description: "checkpoint task".into(),
                agent_name: "build".into(),
                backend: None,
                model: String::new(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: String::new(),
                inherit: InheritOptions::default(),
                abort_handle: None,
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
                started_at: Instant::now(),
            },
        );
        let slot = crate::ai::driver::runtime_ctx::new_subagent_progress_slot();
        super::TASK_PROGRESS_REGISTRY
            .lock()
            .unwrap()
            .insert(task_id.clone(), slot.clone());
        Self {
            kernel,
            session_id,
            task_id,
            owner_pid,
            child_pid,
            channel,
            futex,
            slot,
            _os_guard: os_guard,
        }
    }

    fn publish(&self, body: &str) {
        use crate::ai::driver::runtime_ctx::{
            SUBAGENT_PHASE, SUBAGENT_TASK_ID, publish_subagent_checkpoint_summary,
        };
        SUBAGENT_TASK_ID.sync_scope(self.task_id.clone(), || {
            SUBAGENT_PHASE.sync_scope(self.slot.clone(), || {
                publish_subagent_checkpoint_summary("saved checkpoint", body);
            });
        });
    }

    fn wait(&self) -> String {
        self.kernel
            .lock()
            .unwrap()
            .set_current_pid(Some(self.owner_pid));
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .sync_scope((self.session_id.clone(), 0), || {
                execute_task_wait(&serde_json::json!({
                    "task_ids": [self.task_id.clone()],
                    "timeout_secs": 60,
                }))
            })
            .unwrap()
    }

    fn status(&self) -> String {
        self.kernel
            .lock()
            .unwrap()
            .set_current_pid(Some(self.owner_pid));
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .sync_scope((self.session_id.clone(), 0), || {
                execute_task_status(&serde_json::json!({}))
            })
            .unwrap()
    }

    fn finish(&self, output: &str) {
        self.kernel
            .lock()
            .unwrap()
            .channel_send(
                Some(self.child_pid),
                ChannelId(self.channel),
                serde_json::json!({"status": "completed", "output": output}).to_string(),
            )
            .unwrap();
    }
}

impl Drop for ParentCheckpointTask {
    fn drop(&mut self) {
        remove_task_entry(&self.task_id);
        super::TASK_PROGRESS_REGISTRY
            .lock()
            .unwrap()
            .remove(&self.task_id);
        super::clear_task_wait_state(&task_wait_key(
            &self.session_id,
            self.owner_pid,
            &WaitPolicy::Any,
            &[self.task_id.clone()],
        ));
        let mut os = self.kernel.lock().unwrap();
        os.set_current_pid(Some(self.owner_pid));
        let _ = os.channel_destroy(None, ChannelId(self.channel));
        let _ = os.futex_destroy(self.futex);
        let _ = os.kill_process(self.child_pid, "test cleanup".into());
    }
}

#[test]
fn task_wait_parked_delivers_checkpoint_excerpt() {
    let _env_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let task = ParentCheckpointTask::new();
    task.publish("first saved line\nsecond\u{1b}[31m\u{7}\r line");

    let parked = task.wait();
    assert!(parked.contains("[task_wait PARKED]"), "{parked}");
    assert!(parked.contains(
        "Assistant-authored progress (partial checkpoint excerpt; unverified):\n    first saved line\n    second [31m   line"
    ), "{parked}");
    assert_eq!(parked.matches("first saved line").count(), 1);
    assert!(!parked.contains('\u{1b}'));
    assert!(!parked.contains('\u{7}'));
    assert!(!parked.contains('\r'));
    assert!(matches!(
        task.kernel
            .lock()
            .unwrap()
            .get_process(task.owner_pid)
            .unwrap()
            .state,
        aios_kernel::kernel::ProcessState::Waiting { .. }
    ));
}

#[test]
fn task_wait_repeated_parks_deduplicate_checkpoint_revision() {
    let _env_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let task = ParentCheckpointTask::new();
    task.publish("same checkpoint body");
    assert_eq!(task.wait().matches("same checkpoint body").count(), 1);

    let repeated = task.wait();
    assert!(repeated.contains("[task_wait PARKED]"), "{repeated}");
    assert!(!repeated.contains("same checkpoint body"), "{repeated}");
    assert!(!task.status().contains("same checkpoint body"));

    // Identical text saved again is a new revision, so body-based deduplication is incorrect.
    task.publish("same checkpoint body");
    assert_eq!(task.wait().matches("same checkpoint body").count(), 1);
    assert!(!task.wait().contains("same checkpoint body"));
}

#[test]
fn task_wait_budget_elapsed_delivers_checkpoint_once() {
    let _env_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let task = ParentCheckpointTask::new();
    assert!(task.wait().contains("[task_wait PARKED]"));
    task.publish("saved during wait budget");
    expire_task_wait_states_for_test();

    let elapsed = task.wait();
    assert!(elapsed.contains("[task_wait BUDGET ELAPSED]"), "{elapsed}");
    assert_eq!(elapsed.matches("saved during wait budget").count(), 1);
    assert!(!task.status().contains("saved during wait budget"));
    let repeated = task.wait();
    assert!(repeated.contains("[task_wait PARKED]"), "{repeated}");
    assert!(!repeated.contains("saved during wait budget"));
    let os = task.kernel.lock().unwrap();
    assert!(os.channel_meta(ChannelId(task.channel)).is_some());
    assert!(os.get_process(task.child_pid).is_some_and(|process| {
        !matches!(process.state, aios_kernel::kernel::ProcessState::Terminated)
    }));
}

#[test]
fn timeout_wake_note_delivers_checkpoint_once_without_publication_wake() {
    let _env_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let task = ParentCheckpointTask::new();
    assert!(task.wait().contains("[task_wait PARKED]"));
    task.publish("saved before timeout wake");
    {
        let os = task.kernel.lock().unwrap();
        let owner = os.get_process(task.owner_pid).unwrap();
        assert!(matches!(
            owner.state,
            aios_kernel::kernel::ProcessState::Waiting { .. }
        ));
        assert!(
            owner.mailbox.is_empty(),
            "publication must only refresh progress"
        );
    }
    expire_task_wait_states_for_test();
    wake_expired_task_waits();
    wake_expired_task_waits();
    {
        let os = task.kernel.lock().unwrap();
        let owner = os.get_process(task.owner_pid).unwrap();
        assert!(matches!(
            owner.state,
            aios_kernel::kernel::ProcessState::Ready
        ));
        let notes = owner
            .mailbox
            .iter()
            .filter(|note| note.contains("[TASK_WAIT_TIMEOUT]"))
            .collect::<Vec<_>>();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].matches("saved before timeout wake").count(), 1);
        assert!(notes[0].contains("partial checkpoint excerpt; unverified"));
    }
    let elapsed = task.wait();
    assert!(elapsed.contains("[task_wait BUDGET ELAPSED]"), "{elapsed}");
    assert!(!elapsed.contains("saved before timeout wake"));
    assert!(!task.status().contains("saved before timeout wake"));
}

#[test]
fn task_status_delivers_checkpoint_body_once_per_revision() {
    let _env_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let task = ParentCheckpointTask::new();
    task.publish("status checkpoint body");
    let first = task.status();
    assert!(
        first.contains(&format!("progress[{}]", task.task_id)),
        "{first}"
    );
    assert!(first.contains("partial checkpoint excerpt; unverified"));
    assert_eq!(first.matches("status checkpoint body").count(), 1);
    let repeated = task.status();
    assert!(!repeated.contains("status checkpoint body"), "{repeated}");
    assert!(
        repeated.contains("saved checkpoint"),
        "ordinary progress remains pollable"
    );
    assert!(!task.wait().contains("status checkpoint body"));

    task.publish("status checkpoint body");
    assert_eq!(task.status().matches("status checkpoint body").count(), 1);
    assert!(!task.status().contains("status checkpoint body"));
}

#[test]
fn task_status_suppresses_excerpt_when_delivering_terminal_result() {
    let _env_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let task = ParentCheckpointTask::new();
    task.publish("final shared body");
    task.finish("final shared body");
    let status = task.status();
    assert!(status.contains("Completed task results below"), "{status}");
    assert_eq!(status.matches("final shared body").count(), 1);
    assert!(!status.contains("partial checkpoint excerpt"), "{status}");
    assert!(remove_task_entry(&task.task_id).is_none());
}

#[test]
fn task_wait_suppresses_excerpt_when_delivering_terminal_result() {
    let _env_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let task = ParentCheckpointTask::new();
    task.publish("final wait shared body");
    task.finish("final wait shared body");
    let result = task.wait();
    assert_eq!(result.matches("final wait shared body").count(), 1);
    assert!(!result.contains("partial checkpoint excerpt"), "{result}");
    assert!(!result.contains("[task_wait PARKED]"), "{result}");
    assert!(remove_task_entry(&task.task_id).is_none());
}
