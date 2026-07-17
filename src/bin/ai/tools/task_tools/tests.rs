use super::{
    AsyncTaskEntry, InheritOptions, OsTaskGoal, OutstandingTaskSnapshot,
    SUBAGENT_PARENT_SUMMARY_REMINDER, SUBAGENT_WALL_CLOCK_TIMEOUT, SelectedSubagent,
    StoredTaskResult, TASK_REGISTRY,
    WaitManySource, append_current_process_cancel_source, build_selection_explanation,
    encode_os_task_goal, epoll_wait_many, epoll_wait_many_channels, execute_task_cancel,
    execute_task_status, execute_task_wait, format_task_result, insert_task_entry_for_test,
    is_encoded_task_goal, prepare_subagent_task, reap_timed_out_subagents, remove_task_entry,
    render_outstanding_task_anchor, select_subagent, wait_sources_for_channel_and_futex,
    with_task_entry_by_pid,
};
use super::{ToolRegistration, ToolSpec};
use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
use crate::ai::cli::ParsedCli;
use crate::ai::driver::runtime_ctx::{DRIVER_CTX, DriverContext};
use crate::ai::mcp::McpClient;
use crate::ai::tools::registry::common::tool_history_policy;
use crate::ai::types::{App, AppConfig};
use aios_kernel::{
    kernel::{EventId, KernelInternal, Syscall, WaitPolicy},
    local::LocalOS,
    primitives::{ChannelId, FutexAddr, FutexOps, IpcOps},
};
use serde_json::Value;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::{Duration, Instant};

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
        model_tier: Some(AgentModelTier::Standard),
        disabled: false,
        hidden: false,
        color: None,
        source_path: None,
    }
}

fn test_app_with_model(current_model: String) -> App {
    App {
        cli: ParsedCli::default(),
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
            agent_route_model_path: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/bin/ai/config/agent_route/agent_route_model.json"),
            skill_match_model_path: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/bin/ai/config/skill_match/skill_match_model.json"),
        },
        session_id: String::new(),
        session_history_file: std::path::PathBuf::new(),
        active_persona: crate::ai::persona::default_persona(),
        client: reqwest::Client::new(),
        current_model,
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
        last_known_cached_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
        turn_reasoning_items: Default::default(),
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
    let mut review = manifest("review", "Read-only review agent", AgentMode::Subagent);

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
fn prepare_subagent_task_auto_selects_model_and_fallback() {
    let current_model = crate::ai::model_names::all()
        .first()
        .map(|model| crate::ai::model_names::model_handle(model))
        .expect("models.json must contain at least one model");
    let mut navigator = manifest(
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

    let prepared = DRIVER_CTX
        .sync_scope(ctx, || prepare_subagent_task(&args))
        .unwrap();

    assert!(prepared.is_model_auto_selected);
    assert!(prepared.auto_model_fallback.is_some());
    assert!(
        prepared
            .selection_explanation
            .contains("model_reason=auto-selected for agent_tier=")
    );
}

#[test]
fn explicit_primary_agent_is_rejected_for_task_tool() {
    let mut build = manifest("build", "Main build agent", AgentMode::Primary);
    let mut navigator = manifest(
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
fn tfidf_auto_selection_matches_task_to_subagent_description() {
    let mut explore = manifest(
        "navigator",
        "Read-only codebase exploration agent",
        AgentMode::Subagent,
    );
    explore.model_tier = Some(AgentModelTier::Light);

    let mut review = manifest("critic", "Code review agent", AgentMode::Subagent);
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
fn selection_explanation_mentions_quality_tier_for_auto_model_choice() {
    // 之前这里硬编码 "qwen3-max"；该模型已经从 models.json 移除。
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
                 no Alibaba+Flagship model present in models.json"
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
    let entry = AsyncTaskEntry {
        session_id: String::new(),
        result_observed: false,
        pid: 42,
        result_channel_id: 1,
        completion_futex_addr: FutexAddr(1),
        description: "verify behavior".to_string(),
        agent_name: "explore".to_string(),
        model: "qwen3.7-max".to_string(),
        is_model_auto_selected: true,
        auto_model_fallback: None,
        selection_explanation: "model_reason=auto-selected".to_string(),
        inherit: InheritOptions::default(),
        abort_handle: None,
        started_at: Instant::now(),
    };
    let result = StoredTaskResult {
        status: "failed".to_string(),
        output: String::new(),
        error: Some("request timed out waiting for response headers".to_string()),
    };

    let output = format_task_result(&entry, result);

    assert!(output.contains("FAILED"));
    assert!(output.contains("Error: request timed out waiting for response headers"));
    assert!(output.contains("(subagent did not produce any final assistant text)"));
    assert!(output.contains(SUBAGENT_PARENT_SUMMARY_REMINDER));
}

#[test]
fn task_wait_rejects_foreign_session_task_ids() {
    let own_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let foreign_task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let own_task_id_for_call = own_task_id.clone();
    let foreign_task_id_for_call = foreign_task_id.clone();
    insert_task_entry_for_test(
        own_task_id.clone(),
        AsyncTaskEntry {
            session_id: "session-a".to_string(),
            result_observed: false,
            pid: 1,
            result_channel_id: 11,
            completion_futex_addr: FutexAddr(21),
            description: "own task".to_string(),
            agent_name: "explore".to_string(),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            started_at: Instant::now(),
        },
    );
    insert_task_entry_for_test(
        foreign_task_id.clone(),
        AsyncTaskEntry {
            session_id: "session-b".to_string(),
            result_observed: false,
            pid: 2,
            result_channel_id: 12,
            completion_futex_addr: FutexAddr(22),
            description: "foreign task".to_string(),
            agent_name: "build".to_string(),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
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
fn task_status_collects_completed_results_and_cleans_up_resources() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

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
            session_id: app.session_id.clone(),
            result_observed: false,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "inspect parser".to_string(),
            agent_name: "explore".to_string(),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            started_at: Instant::now(),
        },
    );

    let output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_status(&serde_json::json!({}))
        })
        .expect("task_status should succeed");

    assert!(output.contains("Completed task results below (already collected"));
    assert!(output.contains("subagent final answer"));
    assert!(output.contains("COMPLETED"));
    assert!(remove_task_entry(&task_id).is_none());

    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());

    if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn task_wait_any_returns_ready_result_without_waiting_for_pending_task() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

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
        let pending_channel =
            os.channel_create(Some(pending_pid), 1, "pending-result".to_string());
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
                session_id: app.session_id.clone(),
                result_observed: false,
                pid,
                result_channel_id: channel,
                completion_futex_addr: futex,
                description: description.to_string(),
                agent_name: "explore".to_string(),
                model: "qwen3.7-max".to_string(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: "explicit override".to_string(),
                inherit: InheritOptions::default(),
                abort_handle: None,
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

    if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn task_status_cleans_up_terminated_task_without_result() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (pid, result_channel_id, completion_futex_addr) = {
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
        (pid, channel.raw(), completion_futex)
    };

    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            session_id: app.session_id.clone(),
            result_observed: false,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "terminated task".to_string(),
            agent_name: "build".to_string(),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: None,
            started_at: Instant::now(),
        },
    );

    let output = crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .sync_scope((app.session_id.clone(), 0usize), || {
            execute_task_status(&serde_json::json!({}))
        })
        .expect("task_status should collect terminated task");

    assert!(output.contains("terminated without publishing any output"));
    assert!(remove_task_entry(&task_id).is_none());
    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());
    drop(os);

    if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn task_cancel_aborts_worker_and_leaves_result_collectable_via_task_wait() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (pid, result_channel_id, completion_futex_addr) = {
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
        (pid, channel.raw(), completion_futex)
    };

    let worker = tokio::spawn(std::future::pending::<()>());
    let abort_handle = worker.abort_handle();
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            session_id: app.session_id.clone(),
            result_observed: false,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "cancel branch".to_string(),
            agent_name: "explore".to_string(),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: Some(abort_handle),
            started_at: Instant::now(),
        },
    );

    let cancel_output = crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        (app.session_id.clone(), 0usize),
        || execute_task_cancel(&serde_json::json!({ "task_ids": [task_id.clone()] })),
    )
    .expect("task_cancel should succeed");
    assert!(
        worker
            .await
            .expect_err("task_cancel must stop the Tokio worker")
            .is_cancelled()
    );
    assert!(cancel_output.contains("Required next step: collect these terminal results"));
    assert!(
        crate::ai::tools::task_tools::with_task_entry(&task_id, |_| ()).is_some(),
        "cancelled task must stay collectable until task_wait/task_status consumes it"
    );

    let wait_output = crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        (app.session_id.clone(), 0usize),
        || execute_task_wait(&serde_json::json!({ "task_ids": [task_id.clone()] })),
    )
    .expect("task_wait should collect cancelled result");

    assert!(wait_output.contains("CANCELLED"));
    assert!(wait_output.contains("Error: cancelled by parent agent"));
    assert!(remove_task_entry(&task_id).is_none());

    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());

    if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn wall_clock_reaper_aborts_worker_and_leaves_timeout_result_collectable() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_model("qwen3.7-max".to_string());
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (pid, result_channel_id, completion_futex_addr) = {
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
        (pid, channel.raw(), completion_futex)
    };

    let worker = tokio::spawn(std::future::pending::<()>());
    let abort_handle = worker.abort_handle();
    insert_task_entry_for_test(
        task_id.clone(),
        AsyncTaskEntry {
            session_id: app.session_id.clone(),
            result_observed: false,
            pid,
            result_channel_id,
            completion_futex_addr,
            description: "timeout branch".to_string(),
            agent_name: "explore".to_string(),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: InheritOptions::default(),
            abort_handle: Some(abort_handle),
            started_at: Instant::now()
                - SUBAGENT_WALL_CLOCK_TIMEOUT
                - Duration::from_secs(1),
        },
    );

    reap_timed_out_subagents();
    assert!(
        worker
            .await
            .expect_err("wall-clock reaper must stop the Tokio worker")
            .is_cancelled()
    );

    let wait_output = crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
        (app.session_id.clone(), 0usize),
        || execute_task_wait(&serde_json::json!({ "task_ids": [task_id.clone()] })),
    )
    .expect("task_wait should collect timeout result");
    assert!(wait_output.contains("TIMEOUT"));
    assert!(wait_output.contains("exceeded wall-clock lifetime"));
    assert!(remove_task_entry(&task_id).is_none());

    let os = app.os.lock().unwrap();
    assert!(os.channel_meta(ChannelId(result_channel_id)).is_none());
    assert!(os.futex_event_id(completion_futex_addr).is_none());

    if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn subagent_result_tools_are_never_lossy_or_pruned() {
    for name in ["task", "task_wait", "task_status"] {
        let policy = tool_history_policy(name);
        assert!(
            !policy.allows_lossy_compress(),
            "{name} result must not be lossy-compressed"
        );
        assert!(
            !policy.allows_prune(),
            "{name} result must not be LLM-pruned"
        );
    }
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
    })
    .unwrap();

    assert!(is_encoded_task_goal(&goal));
    assert!(!is_encoded_task_goal("plain foreground goal"));
}

#[test]
fn task_entry_can_be_looked_up_by_pid() {
    let task_id = format!("task_test_{}", uuid::Uuid::new_v4().simple());
    {
        let mut registry = TASK_REGISTRY.lock().unwrap();
        registry.insert(
            task_id.clone(),
            AsyncTaskEntry {
                session_id: String::new(),
                result_observed: false,
                pid: 4242,
                result_channel_id: 11,
                completion_futex_addr: FutexAddr(13),
                description: "inspect".to_string(),
                agent_name: "explore".to_string(),
                model: "qwen3.7-max-alibaba".to_string(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: "explicit override".to_string(),
                inherit: InheritOptions::default(),
                abort_handle: None,
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
            description: "inspect parser".to_string(),
        },
        OutstandingTaskSnapshot {
            task_id: "task_beta".to_string(),
            status: "completed".to_string(),
            agent_name: "build".to_string(),
            model: "gpt-5.5".to_string(),
            description: "verify fix".to_string(),
        },
    ]);

    assert!(note.contains("[pending-subagent-tasks]"));
    assert!(note.contains("Outstanding task_ids: [task_alpha, task_beta]"));
    assert!(note.contains("task_id=task_alpha status=running"));
    assert!(note.contains("task_id=task_beta status=completed"));
    assert!(note.contains("use `task_wait` with the same task_ids"));
}

fn params_question() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user."
            },
            "header": {
                "type": "string",
                "description": "Very short label (max 30 chars) for context."
            },
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Display text for this option (1-5 words)."
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief explanation of what this option means."
                        }
                    },
                    "required": ["label", "description"]
                },
                "description": "Available choices for the user."
            },
            "multiple": {
                "type": "boolean",
                "description": "Allow selecting multiple choices (default: false)."
            }
        },
        "required": ["question", "header", "options"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "question",
        description: "Ask the user questions during execution. Use this to gather preferences, clarify ambiguous instructions, get decisions on implementation choices, or offer choices about direction. Returns the user's selected answer(s).",
        parameters: params_question,
        execute: execute_question,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

pub(crate) fn execute_question(args: &Value) -> Result<String, String> {
    let question = args["question"]
        .as_str()
        .ok_or("Missing 'question' parameter")?;

    let header = args["header"]
        .as_str()
        .ok_or("Missing 'header' parameter")?;

    let options = args["options"]
        .as_array()
        .ok_or("Missing 'options' parameter (must be an array)")?;

    if options.is_empty() {
        return Err("options array cannot be empty".to_string());
    }

    let multiple = args["multiple"].as_bool().unwrap_or(false);

    println!("\n--- Question: {} ---", header);
    println!("{}", question);
    println!();

    for (i, opt) in options.iter().enumerate() {
        let label = opt["label"].as_str().unwrap_or("?");
        let desc = opt["description"].as_str().unwrap_or("");
        println!("  {}. {} - {}", i + 1, label, desc);
    }
    println!();

    if multiple {
        println!("Enter option numbers separated by commas (or type your own answer):");
    } else {
        println!("Enter option number (or type your own answer):");
    }

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|e| format!("Failed to read input: {}", e))?;

    let input = input.trim();

    if input.is_empty() {
        return Err("No answer provided".to_string());
    }

    if multiple {
        let selections: Vec<&str> = input.split(',').map(|s| s.trim()).collect();
        let mut selected_labels = Vec::new();

        for sel in &selections {
            if let Ok(idx) = sel.parse::<usize>() {
                if idx > 0 && idx <= options.len() {
                    if let Some(label) = options[idx - 1]["label"].as_str() {
                        selected_labels.push(label.to_string());
                    }
                } else {
                    return Ok(format!("[User answer] {}", input));
                }
            } else {
                return Ok(format!("[User answer] {}", input));
            }
        }

        Ok(format!("[User selected] {}", selected_labels.join(", ")))
    } else {
        if let Ok(idx) = input.parse::<usize>() {
            if idx > 0 && idx <= options.len() {
                if let Some(label) = options[idx - 1]["label"].as_str() {
                    return Ok(format!("[User selected] {}", label));
                }
            }
        }

        Ok(format!("[User answer] {}", input))
    }
}
