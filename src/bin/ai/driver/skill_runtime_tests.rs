use super::{
    ContextKind, PromptContext, SystemPromptBuilder, ToolGroup, available_tool_names,
    build_hidden_execution_primitive_catalog, build_hidden_mcp_tool_catalog,
    build_hidden_task_tool_catalog, build_project_instruction_prompt,
    build_scoped_project_instruction_prompt, build_system_prompt, builtin_tools_for_skill,
    declares_hidden_group, ensure_required_baseline_tools, escape_xml_attr,
    filter_mcp_tools_by_allowed_servers, has_tool, manifest_tool_definitions,
    merge_with_runtime_enabled_tools, push_project_context, resolve_max_iterations,
    select_mcp_tools, session_context_prompt, tool_uses_mcp_server,
};
use crate::ai::agents::{AgentManifest, AgentMode};
use crate::ai::driver::runtime_ctx::{SUBAGENT_CWD, SUBAGENT_DEPTH};
use crate::ai::history::SkillActivationEvent;
use crate::ai::mcp::McpClient;
use crate::ai::skills::SkillManifest;
use crate::ai::tools::enable_tools::set_explicit_enabled_tool_names;
use crate::ai::types::{FunctionDefinition, PendingSkillContinuation, ToolDefinition};
use rust_tools::cw::SkipSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

static EXPLICIT_TOOL_TEST_GUARD: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[test]
fn mcp_server_filter_matches_longest_server_name_prefix() {
    let allowed = vec!["foo".to_string(), "foo_bar".to_string()];
    assert!(tool_uses_mcp_server("mcp_foo_bar_search", &allowed));
    assert!(tool_uses_mcp_server("mcp_foo_lookup", &allowed));
    assert!(!tool_uses_mcp_server("mcp_bar_search", &allowed));
}

#[test]
fn active_agent_max_steps_override_default_iterations() {
    let agent = AgentManifest {
        name: "build".to_string(),
        description: String::new(),
        mode: AgentMode::Primary,
        model: None,
        temperature: None,
        max_steps: Some(17),
        prompt: String::new(),
        system_prompt: None,
        tools: Vec::new(),
        tool_groups: vec!["builtin".to_string(), "executor".to_string()],
        mcp_servers: Vec::new(),
        disable_mcp_tools: false,
        model_tier: None,
        disabled: false,
        hidden: false,
        color: None,
        source_path: None,
    };

    assert_eq!(resolve_max_iterations(Some(&agent), false), 17);
    assert_eq!(resolve_max_iterations(Some(&agent), true), 17);
    assert_eq!(
        resolve_max_iterations(None, true),
        super::super::EXECUTOR_MAX_ITERATIONS
    );
    assert_eq!(
        resolve_max_iterations(None, false),
        super::super::DEFAULT_MAX_ITERATIONS
    );
}

#[test]
fn default_core_tools_exclude_lazy_skill_discovery_tools() {
    let tools = builtin_tools_for_skill(&[], None);
    let names = tools
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();
    assert!(names.iter().any(|name| name == "enable_tools"));
    assert!(names.iter().any(|name| name == "read_file"));
    // Skill discovery/activation, task orchestration, and knowledge memory are low-frequency capabilities: by default they do not
    // expand into the resident core set; `enable_tools` enables them on demand (the builtin group stays, discoverable dynamically).
    assert!(!names.iter().any(|name| name == "activate_skill"));
    assert!(!names.iter().any(|name| name == "list_skills"));
    assert!(!names.iter().any(|name| name == "load_skill"));
    assert!(!names.iter().any(|name| name == "save_skill"));
    assert!(!names.iter().any(|name| name == "task_spawn"));
    assert!(!names.iter().any(|name| name == "task_integrate"));
    assert!(!names.iter().any(|name| name == "knowledge_save"));
    assert!(!names.iter().any(|name| name == "knowledge_search"));
}

#[test]
fn manifest_tools_keep_skill_discovery_lazy() {
    let groups = vec!["core".to_string(), "executor".to_string()];
    let tools = manifest_tool_definitions(&groups, &[])
        .expect("non-empty manifest groups should resolve tool definitions");
    let names = tools
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();

    assert!(names.iter().any(|name| name == "enable_tools"));
    assert!(names.iter().any(|name| name == "read_file"));
    assert!(!names.iter().any(|name| name == "activate_skill"));
    assert!(!names.iter().any(|name| name == "list_skills"));
    assert!(!names.iter().any(|name| name == "load_skill"));
}

#[test]
fn skill_declaring_task_group_eagerly_loads_only_the_task_family() {
    // Regression guard: task* were moved out of core (lazy builtin-only) so the
    // default turn stays slim, but agent-team / audit_own_changes hard-require the
    // task family in their flows (delegation / spawning the audit subagent). Their
    // manifests declare the narrow `task` group so expansion loads exactly that
    // family. Heavy non-flow orchestration tools (`manage_team`, `run_agent_graph`)
    // live in the narrow `agent_team` group instead of `builtin`: `[core, task]`
    // stays slim (the broad `builtin` tag means "nearly the whole registry",
    // ~15KB of unrelated schemas), while the agent-team skill opts into them by
    // declaring `agent_team` in its manifest.
    let names_for = |groups: Vec<String>| {
        manifest_tool_definitions(&groups, &[])
            .expect("non-empty manifest groups should resolve tool definitions")
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>()
    };

    let skill_groups = names_for(vec!["core".to_string(), "task".to_string()]);
    assert!(
        skill_groups.iter().any(|name| name == "task_spawn"),
        "skill with [core, task] must eagerly expose task_spawn"
    );
    assert!(skill_groups.iter().any(|name| name == "task"));
    assert!(skill_groups.iter().any(|name| name == "task_integrate"));
    assert!(
        !skill_groups.iter().any(|name| name == "run_agent_graph"),
        "[core, task] must not drag in heavyweight non-flow orchestration tools"
    );
    assert!(!skill_groups.iter().any(|name| name == "manage_team"));
    assert!(!skill_groups.iter().any(|name| name == "save_skill"));
    assert!(!skill_groups.iter().any(|name| name == "knowledge_save"));

    let plain_groups = names_for(vec!["core".to_string(), "executor".to_string()]);
    assert!(
        !plain_groups.iter().any(|name| name == "task_spawn"),
        "default [core, executor] manifest must keep task* lazy"
    );

    let agent_team_groups = names_for(vec![
        "core".to_string(),
        "task".to_string(),
        "agent_team".to_string(),
    ]);
    assert!(
        agent_team_groups.iter().any(|name| name == "manage_team"),
        "[core, task, agent_team] (agent-team skill) must expose manage_team"
    );
    assert!(
        agent_team_groups
            .iter()
            .any(|name| name == "run_agent_graph")
    );
    assert!(
        agent_team_groups
            .iter()
            .any(|name| name == "send_side_note"),
        "send_side_note rides in via the task group for orchestration skills"
    );
}

#[test]
fn active_skill_prompt_uses_explicit_user_input_handoff_only_for_skills() {
    let active_skill = skill("external-skill", "external workflow");
    let available = {
        let mut available = SkipSet::new(16);
        available.insert("request_user_input".to_string());
        Box::new(available)
    };

    let active_prompt = build_system_prompt(
        None,
        &[&active_skill],
        &available,
        &PromptContext::default(),
    )
    .render_system_prompt();
    assert!(active_prompt.contains("<interactive_skill_handoff>"));
    assert!(active_prompt.contains("`request_user_input`"));

    let ordinary_prompt = build_system_prompt(None, &[], &available, &PromptContext::default())
        .render_system_prompt();
    assert!(!ordinary_prompt.contains("<interactive_skill_handoff>"));
}

#[test]
fn explicit_user_input_continuation_restores_only_the_next_turn() {
    let external = skill("external-skill", "external workflow");
    let replacement = skill("replacement", "replacement workflow");
    let mcp_client = McpClient::new();
    let mut app = super::super::tests::test_app("build");

    app.pending_skill_continuation = Some(PendingSkillContinuation {
        skill_names: vec![external.name.clone()],
    });
    let guard = super::prepare_skill_for_turn(
        &mut app,
        &mcp_client,
        std::slice::from_ref(&external),
        "the requested answer",
    )
    .unwrap();
    assert_eq!(guard.primary_skill_name(), Some(external.name.as_str()));
    assert!(app.pending_skill_continuation.is_none());
    drop(guard);

    let next_guard = super::prepare_skill_for_turn(
        &mut app,
        &mcp_client,
        std::slice::from_ref(&external),
        "an unrelated new request",
    )
    .unwrap();
    assert!(next_guard.matched_skill_names().is_empty());
    drop(next_guard);

    app.pending_skill_continuation = Some(PendingSkillContinuation {
        skill_names: vec![external.name.clone()],
    });
    app.forced_skills = vec![replacement.name.clone()];
    let forced_guard = super::prepare_skill_for_turn(
        &mut app,
        &mcp_client,
        &[external, replacement],
        "use the explicitly selected skill",
    )
    .unwrap();
    assert_eq!(forced_guard.primary_skill_name(), Some("replacement"));
    assert!(app.pending_skill_continuation.is_none());
}

#[tokio::test]
async fn subagent_builtin_tools_hide_task_orchestration_family() {
    SUBAGENT_DEPTH
        .scope(1, async {
            let tools = builtin_tools_for_skill(&[], None);
            let names = tools
                .into_iter()
                .map(|tool| tool.function.name)
                .collect::<Box<SkipSet<_>>>();

            for hidden in [
                "task",
                "task_spawn",
                "task_wait",
                "task_status",
                "task_integrate",
                "task_cancel",
            ] {
                assert!(!names.contains_str(hidden), "{hidden} should be hidden");
            }
            assert!(names.contains_str("read_file"));
        })
        .await;
}

fn executor_group_agent(name: &str) -> AgentManifest {
    let mut agent = agent(name, Vec::new());
    agent.mode = AgentMode::All;
    agent.tool_groups = vec!["core".to_string(), "executor".to_string()];
    agent.disable_mcp_tools = true;
    agent
}

#[test]
fn executor_group_defers_process_primitives_but_keeps_core_editing() {
    // build/executor use tool_groups: [core, executor]. Execution primitives (process/IPC/shm/env)
    // are lazy by default and stay out of the resident tool set; apply_patch/write_file (core∩executor) are kept,
    // so editing capability is preserved with zero loss.
    let build_agent = executor_group_agent("build");
    let tools = builtin_tools_for_skill(&[], Some(&build_agent));
    let names = tools
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();

    // Resident: core editing/retrieval capabilities
    assert!(names.iter().any(|n| n == "apply_patch"));
    assert!(names.iter().any(|n| n == "write_file"));
    assert!(names.iter().any(|n| n == "read_file"));
    // Resident: baseline self-service capabilities (enable_tools is the progressive-discovery entry point)
    assert!(names.iter().any(|n| n == "enable_tools"));

    // Lazy: heavy execution primitives and task orchestration (task_spawn etc.) are not resident
    assert!(!names.iter().any(|n| n == "task_spawn"));
    for deferred in [
        "spawn_process",
        "spawn_daemon",
        "send_ipc_message",
        "read_mailbox",
        "shm_create",
        "shm_read",
        "shm_write",
        "shm_delete",
        "signal_process",
        "kill_process",
        "wait_process",
        "reap_process",
        "set_env",
        "set_working_dir",
        "set_process_group",
        "signal_process_group",
        "sleep_process",
        "ps_processes",
        "ps_ipc",
    ] {
        assert!(
            !names.iter().any(|n| n == deferred),
            "executor primitive `{deferred}` should be lazy-loaded, not eagerly mounted"
        );
    }
}

#[test]
fn explicit_tools_list_is_not_filtered_by_lazy_load() {
    // Tools explicitly named via `tools:` become resident: even a named execution primitive is not culled.
    let mut agent = agent("custom", Vec::new());
    agent.tool_groups = Vec::new();
    agent.tools = vec!["read_file".to_string(), "spawn_process".to_string()];

    let tools = builtin_tools_for_skill(&[], Some(&agent));
    let names = tools
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();

    assert!(names.iter().any(|n| n == "read_file"));
    assert!(
        names.iter().any(|n| n == "spawn_process"),
        "explicitly named executor primitive must stay eager"
    );
}

#[test]
fn hidden_execution_primitive_catalog_advertises_deferred_tools() {
    // After lazy loading the model must still stay aware: when these
    // primitives are not loaded this turn, the catalog must list them and
    // give the enable_tools path.
    let mut available = SkipSet::new(16);
    available.insert("read_file".to_string());
    available.insert("enable_tools".to_string());

    let catalog = build_hidden_execution_primitive_catalog(&Box::new(available))
        .expect("deferred primitives should produce a catalog");
    assert!(catalog.contains("enable the needed tools via `enable_tools`"));
    // The catalog shows the first MAX_DISPLAY(8) entries in sorted order and
    // folds the rest into "and N more". `kill_process` sorts first and must
    // be inside the display window; assert on it rather than `spawn_process`
    // (the latter sorts last and falls outside the truncation).
    assert!(catalog.contains("kill_process"));
    assert!(catalog.contains("more."));
}

#[test]
fn hidden_execution_primitive_catalog_suppressed_when_all_loaded() {
    // When every executor primitive is already loaded (e.g. an explicit
    // full whitelist), the catalog is suppressed: nothing to advertise.
    let mut available = SkipSet::new(16);
    for (name, _desc) in crate::ai::tools::deferred_eager_load_tool_summaries() {
        available.insert(name);
    }
    assert!(build_hidden_execution_primitive_catalog(&Box::new(available)).is_none());
}

#[test]
fn hidden_task_tool_catalog_advertises_lazy_subagent_family() {
    // A default core turn carries no task group: the catalog must list the
    // unloaded subagent tool names and give the enable_tools path so the
    // model knows subagent tools can be loaded on demand.
    let mut available = SkipSet::new(16);
    available.insert("read_file".to_string());
    available.insert("enable_tools".to_string());

    let catalog = build_hidden_task_tool_catalog(&Box::new(available))
        .expect("unloaded task family should produce a catalog");
    assert!(catalog.contains("enable the needed tools via `enable_tools`"));
    assert!(catalog.contains("task_spawn"));
    assert!(catalog.contains("task_spawn_batch"));
    // The display window takes the first MAX_DISPLAY(8) entries in family
    // order and folds the rest into "and N more".
    assert!(catalog.contains("more."));
}

#[test]
fn hidden_task_tool_catalog_suppressed_when_family_loaded() {
    // Once the task family is loaded (explicit enable or a skill/agent
    // declaration), the catalog must not repeat the nudge.
    let mut available = SkipSet::new(16);
    for name in [
        "task",
        "task_spawn",
        "task_spawn_batch",
        "task_wait",
        "task_status",
        "task_retry",
        "task_cancel",
        "task_evidence_read",
        "task_audit",
        "task_integrate",
        "manage_team",
        "run_agent_graph",
    ] {
        available.insert(name.to_string());
    }
    assert!(build_hidden_task_tool_catalog(&Box::new(available)).is_none());
}

#[tokio::test]
async fn hidden_task_tool_catalog_suppressed_inside_subagent() {
    // Subagents can never enable the task family (it is hidden when
    // SUBAGENT_DEPTH > 0), so the nudge would be misleading there; the
    // catalog must stay silent at depth > 0.
    SUBAGENT_DEPTH.scope(1, async {
        let mut available = SkipSet::new(16);
        available.insert("read_file".to_string());
        available.insert("enable_tools".to_string());
        assert!(build_hidden_task_tool_catalog(&Box::new(available)).is_none());
    });
}

#[test]
fn declares_hidden_group_only_true_for_executor_agents() {
    let build_agent = executor_group_agent("build");
    assert!(declares_hidden_group(&[], Some(&build_agent)));

    // plan/explore use an explicit tools list without any executor group.
    let mut plan_agent = agent("plan", Vec::new());
    plan_agent.mode = AgentMode::All;
    plan_agent.tool_groups = Vec::new();
    plan_agent.tools = vec!["read_file".to_string()];
    assert!(!declares_hidden_group(&[], Some(&plan_agent)));

    assert!(!declares_hidden_group(&[], None));
}

#[test]
fn lazy_load_measurably_shrinks_build_agent_tool_payload() {
    // Quantify, via the production serialization path, the actual per-request tools-token savings of
    // lazy-loading culling of execution primitives: tools in the request body is compact JSON of Vec<ToolDefinition>, and
    // request/builder.rs's estimate_tools_tokens counts serde_json::to_string characters / 2 (conservative conversion,
    // CHARS_PER_TOKEN_CONSERVATIVE) into each turn's prompt. Compare:
    //   baseline = the executor group fully expanded via tool_groups (pre-lazy-loading behavior)
    //   optimized = current builtin_tools_for_skill (after lazy loading, deferred primitives culled)
    const CHARS_PER_TOKEN_CONSERVATIVE: usize = 2;
    let build_agent = executor_group_agent("build");

    // baseline: expand [core, executor] fully with no deferred filtering
    // (the pre-lazy-load behavior).
    let groups = [ToolGroup::Core, ToolGroup::Executor];
    let baseline_tools =
        ensure_required_baseline_tools(crate::ai::tools::tool_definitions_for_groups(&groups));
    // optimized: current production path (manifest_tool_definitions culls deferred primitives).
    let optimized_tools = builtin_tools_for_skill(&[], Some(&build_agent));

    let ser = |tools: &[ToolDefinition]| -> usize {
        serde_json::to_string(tools)
            .map(|s| s.chars().count())
            .unwrap_or(0)
    };
    let baseline_chars = ser(&baseline_tools);
    let optimized_chars = ser(&optimized_tools);
    let saved_chars = baseline_chars.saturating_sub(optimized_chars);
    let saved_tokens = saved_chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE);
    let pct = (saved_chars as f64 / baseline_chars as f64) * 100.0;

    eprintln!(
        "[lazy-load measurement] build agent tools: baseline={} tools / {} chars (~{} tok), \
         optimized={} tools / {} chars (~{} tok), saved={} chars (~{} tok, {:.1}%)",
        baseline_tools.len(),
        baseline_chars,
        baseline_chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE),
        optimized_tools.len(),
        optimized_chars,
        optimized_chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE),
        saved_chars,
        saved_tokens,
        pct,
    );

    // The optimization must genuinely shrink the payload: culled primitive count == deferred catalog size, and the token
    // savings must be significant (conservative lower bound 800 tok/turn, measured ~1.1–1.2k). If anyone ever adds these
    // primitives back to core or removes the filter, this assertion goes red immediately.
    let deferred_count = crate::ai::tools::deferred_eager_load_tool_summaries().len();
    assert_eq!(
        baseline_tools.len() - optimized_tools.len(),
        deferred_count,
        "optimized set must drop exactly the deferred executor primitives"
    );
    assert!(
        saved_tokens >= 800,
        "expected >=800 tokens saved per turn, got {saved_tokens}"
    );
}

#[test]
fn system_prompt_only_mentions_tools_available_this_turn() {
    let mut available = SkipSet::new(16);
    available.insert("read_file".to_string());
    available.insert("apply_patch".to_string());
    available.insert("enable_tools".to_string());

    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("<tool_usage>"));
    assert!(prompt.contains("<tool_discovery>"));
    assert!(prompt.contains("<trust_boundary>"));
    assert!(prompt.contains("Additional capabilities are available via `enable_tools`"));
    assert!(!prompt.contains("Capability catalog (not yet loaded"));
    assert!(!prompt.contains("Configured MCP tools are available"));
    assert!(!prompt.contains("Process / IPC / shared-memory primitives are available"));
    assert!(!prompt.contains("Feishu/Lark"));
    assert!(!prompt.contains("Web search:"));
    assert!(!prompt.contains("<knowledge_retrieval>"));
    assert!(!prompt.contains("cargo_test"));
    assert!(!prompt.contains("execute_command"));
    assert!(!prompt.contains("apply_patch"));
    assert!(!prompt.contains("<compressed_context_recovery>"));
}

#[test]
fn system_prompt_forbids_breaking_other_modules_to_satisfy_a_requirement() {
    // Unconditionally rendered system constraint: implementing a requirement must not come at the cost of breaking other modules; conflicts must be reported, not silently broken.
    // Goal mode is the most likely to "sacrifice existing functionality to reach the goal", so both the default and goal paths must verify the constraint is present.
    let cases = [
        PromptContext::default(),
        PromptContext {
            goal_mode: Some("finish the goal".to_string()),
            is_background: false,
        },
    ];
    for ctx in cases {
        let prompt = build_system_prompt(None, &[], &Box::new(SkipSet::new(16)), &ctx)
            .render_system_prompt();
        assert!(prompt.contains("<system_constraints>"));
        assert!(
            prompt.contains("Never break another module's functionality to satisfy a requirement")
        );
        assert!(prompt.contains("apply the `intellectual_honesty` conflict rule"));
    }
}

#[test]
fn system_prompt_bridges_compressed_context_recovery_via_search_overflow() {
    // search_overflow is a core resident tool but must be explicitly bridged to the compression pipeline:
    // after context compression, absence claims must first search the session archive instead of directly asserting "not found".
    let mut available = SkipSet::new(16);
    available.insert("search_overflow".to_string());

    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();

    assert!(prompt.contains("<compressed_context_recovery>"));
    assert!(prompt.contains("search the session archive with `search_overflow`"));
    assert!(prompt.contains("absence claims must cover the archived scope"));
    assert!(prompt.contains("verbatim excerpts"));
    assert!(prompt.contains("scope=all"));
}

#[test]
fn system_prompt_explains_tool_result_evidence_markers() {
    // The `[reference: ...]` markers are runtime-injected on historical tool
    // results; the model must read them as snapshots, not live state. The
    // guidance is unconditional (every tool result can carry a marker).
    let mut available = SkipSet::new(16);
    available.insert("read_file".to_string());
    available.insert("execute_command".to_string());

    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("<tool_result_evidence>"));
    assert!(prompt.contains("[reference: session-history]"));
    assert!(prompt.contains("[reference: stale-file]"));
    assert!(prompt.contains("[reference: git-history]"));
}

#[test]
fn system_prompt_includes_runtime_environment_and_effective_cwd() {
    let available = SkipSet::new(16);
    let effective_cwd = std::env::temp_dir().join("rust_tools_prompt_cwd");
    let prompt = SUBAGENT_CWD.sync_scope(effective_cwd.clone(), || {
        build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
            .render_system_prompt()
    });

    assert!(prompt.contains("<execution_environment>"));
    assert!(prompt.contains("Operating system:"));
    assert!(prompt.contains(std::env::consts::OS));
    assert!(prompt.contains(std::env::consts::ARCH));
    assert!(prompt.contains("Shell:"));
    assert!(prompt.contains(&format!(
        "Effective working directory: `{}`",
        effective_cwd.display()
    )));
    assert!(prompt.contains("Relative tool paths resolve against this directory"));
    assert!(prompt.contains("not necessarily the project root"));
    assert!(prompt.contains("Write commands for this OS/shell"));
}

#[test]
fn session_context_prompt_mentions_id_layout_and_read_only_rule() {
    let mut builder = SystemPromptBuilder::new();
    builder.push_labeled(
        ContextKind::Behavior,
        "session_context",
        session_context_prompt(
            "f6bb0f1c-ce48-4283-96ab-27ab297ed6b7",
            Path::new(
                "/Users/u/.history_file.sessions/f6bb0f1c-ce48-4283-96ab-27ab297ed6b7.sqlite",
            ),
            Path::new("/Users/u/.history_file.sqlite"),
        ),
    );
    let rendered = builder.render_system_prompt();
    assert!(
        rendered.contains("<session_context>")
            && !rendered.contains("<session_context>\n<session_context>")
            && rendered.contains("f6bb0f1c-ce48-4283-96ab-27ab297ed6b7")
            && rendered.contains(".history_file.sessions")
            && rendered.contains("<id>.sqlite")
            && rendered.contains("sqlite3")
            && rendered.contains("Read-only rule"),
        "session_context must carry the id, storage layout, read paths and the read-only rule; got:\n{rendered}"
    );
}

#[test]
fn system_prompt_routes_project_file_deletes_to_apply_patch() {
    let mut available = SkipSet::new(16);
    available.insert("write_file".to_string());
    available.insert("apply_patch".to_string());

    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();

    assert!(prompt.contains("<temporary_files>"));
    assert!(prompt.contains("git-tracked file"));
    assert!(prompt.contains("`apply_patch`"));
    assert!(prompt.contains("ONE `apply_patch` call with multiple `@@` hunks"));
    assert!(prompt.contains("unique anchor"));
    assert!(prompt.contains("structurally similar blocks"));
    assert!(prompt.contains("Do not split related edits into serial read/patch cycles"));
    assert!(prompt.contains("`*** Delete File: <path>`"));
}

#[test]
fn system_prompt_enforces_concise_response_style_with_correctness_safeguard() {
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    // The style section must exist and require "answer first, no rambling"
    assert!(prompt.contains("<response_style>"));
    assert!(prompt.contains("Lead with the answer or action"));
    // Must keep the "conciseness must not trade away correctness" safety pad to prevent over-compression from causing wrong judgments
    assert!(prompt.contains("Be concise without sacrificing correctness"));
    assert!(prompt.contains("Skip preambles, restatements, meta-commentary"));
    assert!(prompt.contains("status only at real milestones or plan changes"));
}

#[test]
fn system_prompt_renders_safety_redlines_and_no_hallucination() {
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    // Dangerous-operation red lines: unconditionally rendered, containing all three elements — forbidden / bypass / confirmation
    assert!(prompt.contains("<safety_redlines>"));
    assert!(prompt.contains("Never perform dangerous operations"));
    assert!(prompt.contains("Never bypass or work around safety mechanisms"));
    assert!(prompt.contains("state the exact command and its consequences and wait for approval"));
    // Anti-hallucination policy: unconditionally rendered. Detailed fact
    // tracing lives in correctness_guardrails; this test protects the
    // complementary evidence/inference boundary, metadata limit, and
    // conclusion-calibration rules.
    assert!(prompt.contains("<no_hallucination>"));
    assert!(prompt.contains("Distinguish evidence from inference"));
    assert!(prompt.contains("label every inference and state its evidentiary basis"));
    assert!(prompt.contains("beyond a field's documented semantics"));
    assert!(prompt.contains(
        "does not establish provenance, lineage, capability, intent, or comparative rank"
    ));
    assert!(prompt.contains("Calibrate conclusions to the evidence"));
    assert!(prompt.contains("do not introduce unstated premises, causal links, or facts"));
}

#[test]
fn system_prompt_uses_criterion_based_parallel_delegation() {
    let mut available = SkipSet::new(16);
    available.insert("plan".to_string());
    available.insert("task_spawn".to_string());
    available.insert("task_wait".to_string());
    available.insert("task_status".to_string());

    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();

    // Delegation is the default choice for substantive steps (serial or parallel), but with clear boundaries; the parent keeps
    // trivial steps, tightly coupled edits, and final review, and never runs dependent steps concurrently.
    assert!(prompt.contains("fan out MULTIPLE focused, independent subtasks concurrently"));
    assert!(prompt.contains("mark `delegate: true` on every substantive step"));
    assert!(
        prompt.contains("delegated steps without it run one at a time via the synchronous `task`")
    );
    assert!(prompt.contains("distinct, bounded goal"));
    assert!(prompt.contains("latency or context-isolation benefit outweighs handoff overhead"));
    // Pre-division shared discovery must complete serially; never run
    // dependent steps concurrently, never delegate just to create
    // parallelism, and keep work in the parent when the net benefit is
    // unclear.
    assert!(prompt.contains("Keep pre-division shared discovery sequential"));
    assert!(prompt.contains("never run dependent steps concurrently"));
    assert!(prompt.contains("do not delegate merely to create parallelism"));
    assert!(prompt.contains("net benefit is marginal or uncertain"));
    // The parent keeps final decisions, tightly coupled edits, and end-to-end
    // synthesis.
    assert!(prompt.contains("Keep final decisions, tightly coupled overlapping edits"));
    assert!(prompt.contains("broad read-only discovery"));
    assert!(prompt.contains("explicit result contract"));
    // Serial steps may also be delegated for context isolation (parallel
    // branches are not required).
    assert!(prompt.contains("experiments out of the parent context"));
    assert!(prompt.contains("parallel branches are not required"));
    assert!(prompt.contains("continue every independent parent-side step while they run"));
    assert!(prompt.contains("only when the parent is blocked on subagent results"));
    assert!(prompt.contains("Use `task_status` for a non-blocking peek while continuing"));
    assert!(!prompt.contains("certainty is not required"));
}

#[test]
fn system_prompt_routes_single_delegation_to_sync_task_when_available() {
    // The Async Subagent Orchestration section is the single authority for
    // task routing. When the synchronous `task` tool is also available, a
    // SINGLE delegated subtask must be steered to `task` instead of a lone
    // task_spawn+task_wait; the Planning section must not repeat that rule
    // (dedup keeps token cost down without losing guidance).
    let mut available = SkipSet::new(16);
    available.insert("task".to_string());
    available.insert("task_spawn".to_string());
    available.insert("task_wait".to_string());
    available.insert("task_status".to_string());
    available.insert("plan".to_string());

    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();

    // Canonical home: Async Subagent Orchestration section (task branch).
    assert!(prompt.contains(
        "For a single delegated subtask whose result you need back, prefer the synchronous `task`"
    ));
    // The Planning section no longer duplicates the routing rule...
    assert!(!prompt.contains("for a SINGLE subtask use the synchronous `task`"));
    assert!(!prompt.contains("is just a slower `task`"));
    // ...but still keeps its own unique plan/delegate guidance.
    assert!(prompt.contains(
        "Simple tasks: act directly. Complex ones: call `plan` first — before the first tool call"
    ));
    assert!(prompt.contains("mark `delegate: true` on every substantive step"));
}

#[test]
fn system_prompt_task_only_tools_skip_empty_planning_section() {
    // Edge case: only task_* tools available (no plan / spawn_process /
    // process / ipc). The Planning section's outer gate includes
    // task_spawn/task_wait, so the section *would* open - but with no
    // plan/sub-process/ipc tools every inner block is skipped, `lines`
    // stays empty, and push_tool_guidance_section early-returns. Routing
    // guidance must come solely from the Async section, with no empty
    // Planning header left over.
    let mut available = SkipSet::new(16);
    available.insert("task".to_string());
    available.insert("task_spawn".to_string());
    available.insert("task_wait".to_string());
    available.insert("task_status".to_string());
    available.insert("task_integrate".to_string());
    available.insert("task_cancel".to_string());
    available.insert("task_spawn_batch".to_string());

    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();

    // No empty Planning section header.
    assert!(!prompt.contains("<planning_subprocess_execution>"));
    // Routing guidance still present, from the Async section alone.
    assert!(prompt.contains(
        "For a single delegated subtask whose result you need back, prefer the synchronous `task`"
    ));
    assert!(prompt.contains("<async_subagent_orchestration>"));
}

#[test]
fn system_prompt_forbids_guessing_without_sufficient_evidence() {
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("<correctness_guardrails>"));
    assert!(prompt.contains("Ground factual claims in observed evidence"));
    assert!(prompt.contains("state what is verified, what is unknown"));
    assert!(prompt.contains("Calibrate verification effort to a claim's consequence"));
    assert!(prompt.contains("prefer direct evidence when reasonably accessible"));
    assert!(prompt.contains("separate evidence-backed premises from judgment"));
    assert!(prompt.contains("navigation aids rather than independent proof"));
    assert!(prompt.contains("reopen underlying evidence only when it could materially change"));
    assert!(prompt.contains("limit absence claims to the scope actually searched"));
    assert!(prompt.contains("locate relevant callers and dependents"));
    assert!(prompt.contains("compilation and tests prove only covered behavior"));
    assert!(prompt.contains("consequences supported by traced evidence"));
    assert!(prompt.contains("keep unresolved hypotheses separate"));
    assert!(prompt.contains("distinguish introduced behavior from pre-existing behavior"));
    assert!(prompt.contains("reset, checkout, restore, stash drop"));
    assert!(prompt.contains("temporary branch/worktree or stash push then pop"));
    // Anti-hallucination bullet: every concrete specific must trace to
    // session-observed evidence; when evidence is insufficient, one
    // targeted lookup first, otherwise report verified / unknown / next
    // step. The negative provenance tail and the abstention-preference
    // bullet were trimmed as restatements of no_hallucination's
    // evidence-to-conclusion gate, which requires unresolved questions
    // to remain distinct from supported findings. That prompt owns the
    // duty and must keep rendering (asserted in
    // system_prompt_renders_safety_redlines_and_no_hallucination).
    // The efficiency guard lives in task_convergence's stopping rule and
    // must keep rendering.
    assert!(prompt.contains("must trace to evidence observed in this session"));
    assert!(prompt.contains("targeted lookup; otherwise state what is verified"));
    assert!(prompt.contains("Do not pursue perfect certainty or unrelated detail"));
}

#[test]
fn system_prompt_requires_self_contained_comments() {
    // Comments rule lives in correctness_guardrails (always-precedence
    // section), not in an agent/skill manifest — it must render for every
    // session regardless of active agent or skill.
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("<correctness_guardrails>"));
    assert!(prompt.contains("Write comments for a reader who only has the code"));
    assert!(prompt.contains("Never reference a discussion-only shorthand or codename"));
    assert!(prompt.contains("state what was decided and why"));
}

#[test]
fn system_prompt_resists_sycophancy() {
    // The intellectual-honesty block must render for every session: the
    // model must earn agreement by evidence, not grant it because the user
    // holds the view. Assert it is present on the default interactive path,
    // in goal mode, and on a skill turn (no mode may drop it).
    let available = SkipSet::new(16);
    let default_prompt = build_system_prompt(
        None,
        &[],
        &Box::new(available.clone()),
        &PromptContext::default(),
    )
    .render_system_prompt();
    assert!(default_prompt.contains("<intellectual_honesty>"));
    assert!(default_prompt.contains(
        "Agreement must be earned by the facts, not granted because the user holds the view"
    ));
    assert!(default_prompt.contains("say so directly and respectfully"));

    let goal_ctx = PromptContext {
        goal_mode: Some("ship the feature".to_string()),
        is_background: false,
    };
    let goal_prompt = build_system_prompt(None, &[], &Box::new(available.clone()), &goal_ctx)
        .render_system_prompt();
    assert!(goal_prompt.contains("<intellectual_honesty>"));

    let skill = skill("demo", "a demo skill");
    let skill_prompt = build_system_prompt(
        None,
        &[&skill],
        &Box::new(available),
        &PromptContext::default(),
    )
    .render_system_prompt();
    assert!(skill_prompt.contains("<intellectual_honesty>"));
}

#[test]
fn system_prompt_scope_discipline_bullets_have_no_leaked_indentation() {
    // Regression: the non-goal Scope Discipline block once dropped the
    // `\n\` line-continuation on two bullets, baking ~13 spaces of source
    // indentation into the rendered prompt. Assert every rendered line is
    // left-trimmed (no leading whitespace leaks from the source literal).
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("<scope_discipline>"));
    // The three bullets must each start at column 0 (bullet marker), not be
    // prefixed by leaked source indentation.
    assert!(
        prompt.contains(
            "\n- Investigate the user's explicit request plus only the direct dependencies"
        )
    );
    assert!(
        prompt.contains("\n- Do not implement refactors or optimizations unrelated to the task.")
    );
    assert!(prompt.contains("\n- For broad requests, define investigation boundaries"));
    // Guard against the exact defect: no bullet prefixed by leading spaces.
    assert!(!prompt.contains("\n             - Do not implement refactors"));
    assert!(!prompt.contains("\n             - For broad requests, define"));
}

#[test]
fn system_prompt_defines_an_end_to_end_behavior_contract() {
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();

    assert!(prompt.contains("current plan and interpretation as hypotheses"));
    assert!(prompt.contains("user correction, failed check, or new evidence"));
    assert!(prompt.contains("conclusions and actions that depended on it"));
    assert!(prompt.contains("Do not patch only the literal symptom"));
    assert!(prompt.contains("approval of adjacent behavior"));
    assert!(prompt.contains("observable outcomes and preserved invariants"));
    assert!(prompt.contains("what must change, what must stay unchanged"));
    assert!(prompt.contains("disappearance of the original symptom"));
}

#[test]
fn system_prompt_links_task_convergence_criteria_to_plan_when_available() {
    // When plan is available: task_convergence injects the "write acceptance
    // criteria into the plan" bridge line; the planning block injected under
    // the same condition carries the plan_update usage notes, and the two
    // together close the plan → execute → accept loop.
    let mut available = SkipSet::new(16);
    available.insert("plan".to_string());
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("For multi-step tasks, encode these criteria into the `plan`"));
    assert!(prompt.contains("Track step progress with `plan_update`"));
    assert!(prompt.contains("Treat the plan as a living roadmap"));
    assert!(prompt.contains("before the first tool call, so the plan is the roadmap"));

    // When plan is unavailable (e.g. culled by a skill whitelist): the bridging line is absent, but the task_convergence body remains.
    let empty = build_system_prompt(
        None,
        &[],
        &Box::new(SkipSet::new(16)),
        &PromptContext::default(),
    )
    .render_system_prompt();
    assert!(empty.contains("<task_convergence>"));
    assert!(!empty.contains("encode these criteria into the `plan`"));
}

#[test]
fn system_prompt_bounds_tool_exploration() {
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("Give every call a concrete decision goal"));
    assert!(prompt.contains("no further call can change the decision"));
    assert!(prompt.contains("Before exploration, state the question it can answer"));
    assert!(prompt.contains("Do not batch code reads or reread visible content"));
}

#[test]
fn system_prompt_uses_success_criteria_for_normal_and_goal_convergence() {
    let available = SkipSet::new(16);
    let normal = build_system_prompt(
        None,
        &[],
        &Box::new(available.clone()),
        &PromptContext::default(),
    )
    .render_system_prompt();
    assert!(normal.contains("<task_convergence>"));
    assert!(normal.contains("task-level success criteria"));
    assert!(normal.contains(
        "the next call can verify it, rule out a live hypothesis, or complete required work"
    ));
    assert!(normal.contains("Stop when all criteria are verified or a specific blocker remains"));
    assert!(normal.contains("evidence count alone is not a stopping rule"));
    assert!(!normal.contains("3+ pieces of converging evidence"));

    let goal = build_system_prompt(
        None,
        &[],
        &Box::new(available),
        &PromptContext {
            goal_mode: Some("analyze the design".to_string()),
            is_background: false,
        },
    )
    .render_system_prompt();
    assert!(goal.contains("Analysis-only goals are complete"));
    assert!(!goal.contains("your job is to act, not analyze"));
    assert!(!goal.contains("every detail of the goal is complete"));
}

#[test]
fn asking_user_guidance_only_on_default_interactive_path() {
    let available = Box::new(SkipSet::new(16));
    let build_agent = agent("build", vec![]);

    let interactive = build_system_prompt(
        Some(&build_agent),
        &[],
        &available,
        &PromptContext::default(),
    )
    .render_system_prompt();
    assert!(interactive.contains("<asking_the_user>"));
    assert!(interactive.contains("information only the user can provide"));

    let goal = build_system_prompt(
        Some(&build_agent),
        &[],
        &available,
        &PromptContext {
            goal_mode: Some("finish the goal".to_string()),
            is_background: false,
        },
    )
    .render_system_prompt();
    assert!(!goal.contains("<asking_the_user>"));

    let background = build_system_prompt(
        Some(&build_agent),
        &[],
        &available,
        &PromptContext {
            goal_mode: None,
            is_background: true,
        },
    )
    .render_system_prompt();
    assert!(!background.contains("<asking_the_user>"));

    let skill_turn = build_system_prompt(
        Some(&build_agent),
        &[&skill("humanizer", "Rewrite text naturally")],
        &available,
        &PromptContext::default(),
    )
    .render_system_prompt();
    assert!(!skill_turn.contains("<asking_the_user>"));
}

#[test]
fn system_prompt_stops_repeating_failed_approach_without_ending_task() {
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();

    assert!(prompt.contains("On failure, diagnose before retrying"));
    assert!(prompt.contains("switch to a materially different safe recovery"));
    assert!(!prompt.contains("after 3 failed attempts on the same issue, stop and report"));
}

#[test]
fn system_prompt_keeps_code_grounding_calls_serial() {
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("Navigate code serially"));
    assert!(prompt.contains("read one sufficiently broad needed region, then patch it"));
    assert!(prompt.contains("Do not batch code reads"));
    assert!(
        !prompt.contains("Work in batches: when several independent read-only lookups are needed")
    );
}

#[test]
fn generic_system_prompt_does_not_hardcode_repo_specific_tool_names() {
    let available = SkipSet::new(16);
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(!prompt.contains("cargo_test"));
    assert!(!prompt.contains("execute_command / cargo_test"));
    assert!(!prompt.contains("execute_command"));
    assert!(!prompt.contains("apply_patch"));
    assert!(prompt.contains("Use tools for requested work"));
    assert!(prompt.contains("if unavailable, say so instead of pretending"));
}

#[test]
fn system_prompt_mentions_mcp_discovery_when_enable_tools_available() {
    let mut available = SkipSet::new(16);
    available.insert("enable_tools".to_string());
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("discover and enable matching `mcp_*` tools first"));
}

#[test]
fn system_prompt_guides_tree_for_layout_when_available() {
    // When no tree, do not inject the navigation section; with a tree, prompt to grasp the structure with tree first, then read_file,
    // avoiding blind ls / recursive reads by the model.
    let without = build_system_prompt(
        None,
        &[],
        &Box::new(SkipSet::new(16)),
        &PromptContext::default(),
    )
    .render_system_prompt();
    assert!(!without.contains("<codebase_navigation>"));

    let mut available = SkipSet::new(16);
    available.insert("tree".to_string());
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("<codebase_navigation>"));
    assert!(prompt.contains("Use `tree` to grasp directory layout before reading files"));
}

#[test]
fn hidden_mcp_tool_catalog_lists_real_available_tools() {
    let catalog = build_hidden_mcp_tool_catalog(
        &[
            tool("mcp_feishu_docs_get_text_by_url"),
            tool("mcp_feishu_doc_create_from_markdown"),
            tool("mcp_pdf-extract_pdf_extract_text"),
        ],
        &[tool("mcp_feishu_doc_create_from_markdown")],
    )
    .unwrap();

    assert!(catalog.contains("Configured MCP tools are available but not loaded"));
    assert!(catalog.contains("discover and enable matching `mcp_*` tools"));
    assert!(catalog.contains("`enable_tools`"));
    assert!(catalog.contains("`mcp_feishu_docs_get_text_by_url`"));
    assert!(catalog.contains("`mcp_pdf-extract_pdf_extract_text`"));
    assert!(!catalog.contains("`mcp_feishu_doc_create_from_markdown`"));
}

#[test]
fn hidden_mcp_tool_catalog_omits_prompt_when_everything_is_loaded() {
    let catalog = build_hidden_mcp_tool_catalog(
        &[tool("mcp_feishu_docs_get_text_by_url")],
        &[tool("mcp_feishu_docs_get_text_by_url")],
    );
    assert!(catalog.is_none());
}

#[test]
fn narrow_skill_whitelist_still_lets_model_discover_explicitly_requested_mcp_tools() {
    // Scenario: the user explicitly asks to "write a Feishu doc with MCP
    // tools", but the active skill's narrow tools: whitelist replaces the
    // tool set with a single dedicated tool, and the default agent carries
    // disable_mcp_tools (no mcp_* pre-mounted). Before the fix: the narrow
    // whitelist squeezed out enable_tools too, the hidden MCP catalog gate
    // (has_tool("enable_tools")) closed with it, and all three MCP discovery
    // paths were severed, making the request physically impossible to serve.
    // After the fix: enable_tools is restored as an always-on baseline, the
    // catalog gate holds again, and the model can discover and enable
    // mcp_feishu_*.
    let mut narrow_skill = skill("feishu-upload", "Upload markdown into Feishu docs");
    narrow_skill.tools = vec!["write_file".to_string()];

    // 1) After the narrow whitelist replaces the tool set, the baseline
    // fallback still restores discovery/loading and basic read-only entries.
    let builtin_tools = builtin_tools_for_skill(&[&narrow_skill], None);
    let builtin_names = builtin_tools
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect::<Vec<_>>();
    assert!(
        builtin_names.contains(&"write_file".to_string()),
        "explicitly declared tools in the skill whitelist must be preserved"
    );
    assert!(
        builtin_names.contains(&"enable_tools".to_string()),
        "enable_tools must be restored as an always-on baseline, otherwise the model cannot discover/enable MCP tools"
    );
    assert!(
        builtin_names.contains(&"read_file".to_string()),
        "read_file must stay as a baseline read-only capability to read the user-named test.md"
    );

    // 2) The default agent's disable_mcp_tools => no mcp_* is pre-mounted
    // this turn.
    let all_mcp_tools = vec![
        tool("mcp_feishu_doc_create_from_markdown"),
        tool("mcp_feishu_docs_get_text_by_url"),
        tool("mcp_pdf-extract_pdf_extract_text"),
    ];
    let loaded_mcp_tools: Vec<ToolDefinition> = Vec::new();

    // 3) available_tools contains enable_tools => the catalog injection gate
    // holds (the production code's has_tool("enable_tools") check).
    let available_tools = available_tool_names(&builtin_tools, &loaded_mcp_tools);
    assert!(
        has_tool(&available_tools, "enable_tools"),
        "the catalog injection gate depends on enable_tools being present in available_tools"
    );

    // 4) The hidden MCP catalog exposes the user-requested mcp_feishu_* to
    // the model as the discovery entry point.
    let catalog = build_hidden_mcp_tool_catalog(&all_mcp_tools, &loaded_mcp_tools)
        .expect("a discovery hint is required when unloaded mcp_* tools exist");
    assert!(catalog.contains("discover and enable matching `mcp_*` tools"));
    assert!(catalog.contains("`enable_tools`"));
    assert!(catalog.contains("`mcp_feishu_doc_create_from_markdown`"));
    assert!(catalog.contains("`mcp_feishu_docs_get_text_by_url`"));
}

#[test]
fn system_prompt_guides_optional_skill_discovery_when_available() {
    let mut available = SkipSet::new(16);
    available.insert("enable_tools".to_string());
    available.insert("activate_skill".to_string());
    available.insert("list_skills".to_string());
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("activate_skill"));
    assert!(prompt.contains("list_skills"));
    assert!(prompt.contains("Skills are optional"));
    assert!(prompt.contains("proactively call `list_skills`"));
    assert!(prompt.contains("technical keywords alone"));
    assert!(prompt.contains("routine source-code, repository, file, or terminal investigation"));
    assert!(prompt.contains("unloads automatically"));
    assert!(prompt.contains("enable_tools"));
}

#[test]
fn system_prompt_prefers_enable_tools_when_no_skill_active() {
    let mut available = SkipSet::new(16);
    available.insert("enable_tools".to_string());
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("No skill is active for this turn"));
    assert!(prompt.contains("enable_tools(operation=list)"));
    assert!(prompt.contains("enabling only the specific tools you need"));
}

#[test]
fn skill_activation_history_reminder_preserves_past_selection_without_reactivation() {
    let events = vec![
        SkillActivationEvent {
            requested_skill: "missing".to_string(),
            injected_skill: None,
            source: "/skills-inline".to_string(),
            outcome: "not-found".to_string(),
        },
        SkillActivationEvent {
            requested_skill: "bytedcli".to_string(),
            injected_skill: Some("bytedcli".to_string()),
            source: "/skills-inline".to_string(),
            outcome: "injected".to_string(),
        },
        SkillActivationEvent {
            requested_skill: "unsafe".to_string(),
            injected_skill: Some("unsafe\nnew instruction".to_string()),
            source: "/skills-inline".to_string(),
            outcome: "injected".to_string(),
        },
    ];

    let history = super::build_skill_activation_history_reminder(&events).unwrap();
    let mut builder = SystemPromptBuilder::new();
    builder.push_labeled(
        ContextKind::Fact,
        "Session Skill Activation History",
        history,
    );
    let reminder = builder.render_context_reminder().unwrap();

    assert!(reminder.contains("\"bytedcli\" was successfully selected"));
    assert!(reminder.contains("\"unsafe\\nnew instruction\" was successfully selected"));
    assert!(!reminder.contains("\"unsafe\nnew instruction\" was successfully selected"));
    assert!(!reminder.contains("missing"));
    assert!(reminder.contains("historical records only"));
    assert!(reminder.contains("do not reactivate a skill"));
    assert!(reminder.contains("current active-skill state"));
}

#[test]
fn skill_activation_history_reminder_keeps_six_recent_unique_selections() {
    let mut events = vec![SkillActivationEvent {
        requested_skill: "duplicate".to_string(),
        injected_skill: Some("duplicate".to_string()),
        source: "old-source".to_string(),
        outcome: "injected".to_string(),
    }];
    events.extend((0..7).map(|index| SkillActivationEvent {
        requested_skill: format!("skill-{index}"),
        injected_skill: Some(format!("skill-{index}")),
        source: "test-source".to_string(),
        outcome: "injected".to_string(),
    }));
    events.push(SkillActivationEvent {
        requested_skill: "duplicate".to_string(),
        injected_skill: Some("duplicate".to_string()),
        source: "new-source".to_string(),
        outcome: "injected".to_string(),
    });

    let history = super::build_skill_activation_history_reminder(&events).unwrap();

    assert_eq!(history.matches("was successfully selected via").count(), 6);
    assert_eq!(history.matches("\"duplicate\"").count(), 1);
    assert!(history.contains("new-source"));
    assert!(!history.contains("old-source"));
    assert!(history.contains("\"skill-6\""));
    assert!(!history.contains("\"skill-1\""));
}

#[test]
fn system_prompt_never_mentions_discover_skills() {
    let mut available = SkipSet::new(16);
    available.insert("enable_tools".to_string());
    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("enable_tools(operation=list)"));
    assert!(!prompt.contains("discover_skills"));
}

#[test]
fn render_groups_same_kind_sections_into_single_tag_block() {
    // identity/behavior/policy should each appear as exactly one tag pair, ordered identity→behavior→policy,
    // This guarantees that "appended afterwards" identity sections like persona are not pushed to the end of the prompt
    // but cluster with the generic identity into one block.
    let mut builder = SystemPromptBuilder::new();
    builder.push(ContextKind::Identity, "Generic identity.");
    builder.push(ContextKind::Behavior, "Behavior one.");
    builder.push(ContextKind::Policy, "Policy one.");
    builder.push(ContextKind::Behavior, "Behavior two.");
    // Simulate persona being appended after build_system_prompt:
    builder.push(ContextKind::Identity, "Persona identity.");

    let prompt = builder.render_system_prompt();

    assert_eq!(prompt.matches("<identity>").count(), 1);
    assert_eq!(prompt.matches("<behavior>").count(), 1);
    assert_eq!(prompt.matches("<policy>").count(), 1);

    let identity_pos = prompt.find("<identity>").unwrap();
    let behavior_pos = prompt.find("<behavior>").unwrap();
    let policy_pos = prompt.find("<policy>").unwrap();
    assert!(identity_pos < behavior_pos && behavior_pos < policy_pos);

    // The two identity sections must land inside the same identity block, in insertion order.
    let generic_pos = prompt.find("Generic identity.").unwrap();
    let persona_pos = prompt.find("Persona identity.").unwrap();
    let identity_close = prompt.find("</identity>").unwrap();
    assert!(generic_pos < persona_pos);
    assert!(persona_pos < identity_close);

    // Sections of the same kind are separated by blank lines; insertion order is kept within the group.
    let behavior_one = prompt.find("Behavior one.").unwrap();
    let behavior_two = prompt.find("Behavior two.").unwrap();
    assert!(behavior_one < behavior_two);
}

#[test]
fn render_excludes_fact_sections_from_system_prompt() {
    let mut builder = SystemPromptBuilder::new();
    builder.push(ContextKind::Identity, "Identity.");
    builder.push_labeled(ContextKind::Fact, "Project Type", "Rust project.");

    let prompt = builder.render_system_prompt();
    assert!(prompt.contains("Identity."));
    assert!(!prompt.contains("Rust project."));
    assert!(!prompt.contains("Project Type"));
}

#[test]
fn render_uses_xml_tags_for_labeled_system_sections() {
    let mut builder = SystemPromptBuilder::new();
    builder.push_labeled(
        ContextKind::Behavior,
        "runtime_guard",
        "Verify before claiming completion.",
    );
    builder.push_labeled(ContextKind::Behavior, "  ", "Unlabeled fallback.");

    let prompt = builder.render_system_prompt();

    assert!(prompt.contains(
        "<behavior>\n<runtime_guard>\nVerify before claiming completion.\n</runtime_guard>\n\nUnlabeled fallback.\n</behavior>"
    ));
    assert!(!prompt.contains("## \n"));
}

#[test]
fn render_keeps_capabilities_in_system_prompt_out_of_fact_reminder() {
    let mut builder = SystemPromptBuilder::new();
    builder.push(
        ContextKind::Capability,
        "Configured MCP tools can be loaded with `enable_tools`.",
    );
    builder.push_labeled(ContextKind::Fact, "Project Type", "Rust project.");

    let prompt = builder.render_system_prompt();
    let reminder = builder.render_context_reminder().unwrap();

    assert!(prompt.contains("<capabilities>"));
    assert!(prompt.contains("Configured MCP tools can be loaded"));
    assert!(!prompt.contains("You should not respond to this context"));
    assert!(!reminder.contains("Configured MCP tools can be loaded"));
    assert!(reminder.contains("Project Type"));
    assert!(reminder.contains("## Project Type"));
}

#[test]
fn active_skill_resources_are_system_capabilities() {
    let available = SkipSet::new(16);
    let mut active_skill = skill("resource-skill", "Uses bundled references");
    active_skill.resource_path = Some("/private/resource-skill/resources".to_string());

    let builder = build_system_prompt(
        None,
        &[&active_skill],
        &Box::new(available),
        &PromptContext::default(),
    );
    let prompt = builder.render_system_prompt();
    let reminder = builder.render_context_reminder().unwrap_or_default();

    assert!(prompt.contains("<active_skill_resources>"));
    assert!(prompt.contains("/private/resource-skill/resources"));
    assert!(!reminder.contains("/private/resource-skill/resources"));
}

#[test]
fn active_skill_prompt_precedes_agent_prompt_and_declares_priority() {
    let available = SkipSet::new(16);
    let mut build_agent = agent("build", vec![]);
    build_agent.prompt = "You are the build agent.".to_string();
    let mut humanizer = skill("humanizer", "Rewrite text naturally");
    humanizer.prompt = "You are a writing editor.".to_string();

    let prompt = build_system_prompt(
        Some(&build_agent),
        &[&humanizer],
        &Box::new(available),
        &PromptContext::default(),
    )
    .render_system_prompt();

    let skill_pos = prompt.find("Active skill: humanizer").unwrap();
    let agent_pos = prompt.find("<agent_instructions>").unwrap();
    assert!(skill_pos < agent_pos);
    assert!(
        prompt.contains("<skill_instructions>\nYou are a writing editor.\n</skill_instructions>")
    );
    assert!(
        prompt.contains("<agent_instructions>\nYou are the build agent.\n</agent_instructions>")
    );
    assert!(prompt.contains("primary behavior contract"));
    assert!(prompt.contains("skill instructions override agent instructions"));
}

#[test]
fn skill_only_prompt_keeps_guardrails_non_overridable() {
    let available = SkipSet::new(16);
    let mut humanizer = skill("humanizer", "Rewrite text naturally");
    humanizer.prompt = "You are a writing editor.".to_string();

    let prompt = build_system_prompt(
        None,
        &[&humanizer],
        &Box::new(available),
        &PromptContext::default(),
    )
    .render_system_prompt();

    assert!(prompt.contains("skill instructions override generic assistant guidelines"));
    assert!(prompt.contains("except the correctness guardrails (including git-safety rules), safety redlines, and policy sections, which always take precedence"));
    assert!(!prompt.contains("<agent_instructions>"));
}

#[test]
fn system_prompt_uses_knowledge_save_for_user_memory_requests() {
    let mut available = SkipSet::new(16);
    available.insert("knowledge_save".to_string());
    available.insert("knowledge_search".to_string());
    available.insert("knowledge_list".to_string());

    let prompt = build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
        .render_system_prompt();
    assert!(prompt.contains("<knowledge_save>"));
    assert!(prompt.contains("call `knowledge_save`"));
    assert!(prompt.contains("`common_sense`, `coding_guideline`"));
    assert!(prompt.contains("Save each distinct durable fact at most once per turn"));
    assert!(prompt.contains("<knowledge_retrieval>"));
    assert!(prompt.contains("Only when the user explicitly asks"));
    assert!(prompt.contains("Reuse a successful knowledge search"));
    assert!(prompt.contains("Never fabricate memory"));
    assert!(prompt.contains("Use `knowledge_list` when asked what is remembered"));
    assert!(!prompt.contains("call `memory_save`"));
}

fn temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.push(format!(
        "rust_tools_skill_runtime_{name}_{}_{}",
        std::process::id(),
        nanos
    ));
    path
}

#[test]
fn project_instruction_prompt_includes_repo_docs_from_cwd_scope() {
    let root = temp_dir("project_prompt");
    let nested = root.join("apps/web/src");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(&nested).unwrap();
    fs::write(root.join("AGENTS.md"), "Use cargo fmt before commit.\n").unwrap();
    fs::write(root.join("apps/web/claude.md"), "Web app uses pnpm.\n").unwrap();

    let prompt = SUBAGENT_CWD
        .sync_scope(nested.clone(), build_project_instruction_prompt)
        .expect("project instruction prompt");

    assert!(prompt.contains(
        "- The current working directory provides project-specific instruction documents."
    ));
    assert!(prompt.contains("<instructions path="));
    assert!(prompt.contains("AGENTS.md"));
    assert!(prompt.contains("Use cargo fmt before commit."));
    assert!(prompt.contains("claude.md"));
    assert!(prompt.contains("Web app uses pnpm."));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn escape_xml_attr_escapes_all_five_special_chars() {
    assert_eq!(
        escape_xml_attr("a&b<c>d\"e'f"),
        "a&amp;b&lt;c&gt;d&quot;e&apos;f"
    );
    // Ordinary path preserved as-is
    assert_eq!(
        escape_xml_attr("/src/bin/ai/AGENTS.md"),
        "/src/bin/ai/AGENTS.md"
    );
    // Empty string is safe
    assert_eq!(escape_xml_attr(""), "");
}

#[test]
fn scoped_project_instruction_prompt_follows_observed_target_path() {
    let root = temp_dir("target_project_prompt");
    let target = root.join("src/bin/ai/driver/iteration.rs");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(root.join("AGENTS.md"), "Root safety rules.\n").unwrap();
    fs::write(root.join("src/bin/ai/AGENTS.md"), "AI runtime rules.\n").unwrap();
    fs::write(
        root.join("src/bin/ai/driver/AGENTS.md"),
        "Driver-specific rules.\n",
    )
    .unwrap();
    fs::write(&target, "// source\n").unwrap();

    let prompt = SUBAGENT_CWD
        .sync_scope(root.clone(), || {
            build_scoped_project_instruction_prompt(std::slice::from_ref(&target))
        })
        .expect("target-scoped instruction prompt");

    assert!(prompt.contains("- These documents apply to files already touched in this turn."));
    assert!(prompt.contains("<instructions path="));
    assert!(prompt.contains("AI runtime rules."));
    assert!(prompt.contains("Driver-specific rules."));
    assert!(!prompt.contains("Root safety rules."));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn scoped_project_instruction_push_confirms_required_target_is_loaded() {
    let root = temp_dir("required_target_project_prompt");
    let target = root.join("src/bin/ai/driver/iteration.rs");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(root.join("AGENTS.md"), "Root safety rules.\n").unwrap();
    fs::write(
        root.join("src/bin/ai/driver/AGENTS.md"),
        "Required driver rules.\n",
    )
    .unwrap();
    fs::write(&target, "// source\n").unwrap();

    SUBAGENT_CWD.sync_scope(root.clone(), || {
        let mut guard = super::SkillTurnGuard {
            restore_agent_context: None,
            builder: SystemPromptBuilder::new(),
            cached_system_prompt: None,
            cached_context_reminder: None,
            matched_skill_names: Vec::new(),
        };

        assert!(guard.push_scoped_project_instructions(std::slice::from_ref(&target), &[]));
        assert!(guard.system_prompt().contains("Required driver rules."));
    });

    let _ = fs::remove_dir_all(root);
}

#[test]
fn context_reminder_injects_active_skill_pointer_into_user_message_reminder() {
    let mut guard = super::SkillTurnGuard {
        restore_agent_context: None,
        builder: SystemPromptBuilder::new(),
        cached_system_prompt: None,
        cached_context_reminder: None,
        matched_skill_names: vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
    };
    let reminder = guard
        .context_reminder()
        .expect("reminder should be present with active skills");
    assert!(reminder.contains("<system-reminder>"));
    assert!(reminder.contains("Active skills at turn start (in activation order):"));
    assert!(reminder.contains("  1. alpha"));
    assert!(reminder.contains("  2. beta"));
    assert!(reminder.contains("  3. gamma"));
    assert!(reminder.contains("<skill_instructions>"));
    assert!(reminder.contains("primary behavior contract for this turn"));
}

#[test]
fn context_reminder_single_active_skill_uses_singular_pointer() {
    let mut guard = super::SkillTurnGuard {
        restore_agent_context: None,
        builder: SystemPromptBuilder::new(),
        cached_system_prompt: None,
        cached_context_reminder: None,
        matched_skill_names: vec!["solo".to_string()],
    };
    let reminder = guard
        .context_reminder()
        .expect("reminder should be present with an active skill");
    assert!(reminder.contains("Active skill at turn start: solo."));
    assert!(!reminder.contains("in activation order"));
}

#[test]
fn context_reminder_omits_skill_pointer_without_active_skills() {
    let mut guard = super::SkillTurnGuard {
        restore_agent_context: None,
        builder: SystemPromptBuilder::new(),
        cached_system_prompt: None,
        cached_context_reminder: None,
        matched_skill_names: Vec::new(),
    };
    assert!(guard.context_reminder().is_none());
}

#[test]
fn project_context_is_appended_separately_from_base_prompt() {
    let root = temp_dir("project_context");
    let nested = root.join("apps/web/src");
    fs::create_dir_all(&nested).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname='demo'\nversion='0.1.0'\n",
    )
    .unwrap();
    fs::write(root.join("AGENTS.md"), "Use cargo fmt before commit.\n").unwrap();

    let (base_prompt, enriched_prompt, reminder) = SUBAGENT_CWD.sync_scope(nested.clone(), || {
        let available = SkipSet::new(16);
        let mut builder =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default());
        let base_prompt = builder.render_system_prompt();
        push_project_context(&mut builder);
        let enriched_prompt = builder.render_system_prompt();
        let reminder = builder.render_context_reminder().unwrap_or_default();
        (base_prompt, enriched_prompt, reminder)
    });

    assert!(!base_prompt.contains("Use cargo fmt before commit."));
    assert!(enriched_prompt.contains("Use cargo fmt before commit."));
    assert!(reminder.contains("Project Type"));
    assert!(reminder.contains("Rust project"));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn project_instructions_remain_available() {
    let root = temp_dir("project_context_general_mode");
    let nested = root.join("apps/web/src");
    fs::create_dir_all(&nested).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname='demo'\nversion='0.1.0'\n",
    )
    .unwrap();
    fs::write(root.join("AGENTS.md"), "Always follow repo safety rules.\n").unwrap();

    let prompt = SUBAGENT_CWD.sync_scope(nested.clone(), || {
        let available = SkipSet::new(16);
        let mut builder =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default());
        push_project_context(&mut builder);
        builder.render_system_prompt()
    });

    assert!(prompt.contains("<project_local_instructions>"));
    assert!(prompt.contains("Always follow repo safety rules."));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn project_action_intent_keeps_project_context_available() {
    let root = temp_dir("project_context_work_signal");
    let nested = root.join("apps/web/src");
    fs::create_dir_all(&nested).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname='demo'\nversion='0.1.0'\n",
    )
    .unwrap();
    fs::write(root.join("AGENTS.md"), "Always follow repo safety rules.\n").unwrap();

    let prompt = SUBAGENT_CWD.sync_scope(nested.clone(), || {
        let available = SkipSet::new(16);
        let mut builder =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default());
        push_project_context(&mut builder);
        builder.render_system_prompt()
    });

    assert!(prompt.contains("<project_local_instructions>"));
    assert!(prompt.contains("Always follow repo safety rules."));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn prompt_introspection_query_still_keeps_project_instructions() {
    let root = temp_dir("project_context_prompt_query");
    let nested = root.join("apps/web/src");
    fs::create_dir_all(&nested).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname='demo'\nversion='0.1.0'\n",
    )
    .unwrap();
    fs::write(root.join("AGENTS.md"), "Always follow repo safety rules.\n").unwrap();

    let prompt = SUBAGENT_CWD.sync_scope(nested.clone(), || {
        let available = SkipSet::new(16);
        let mut builder =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default());
        push_project_context(&mut builder);
        builder.render_system_prompt()
    });

    assert!(prompt.contains("<project_local_instructions>"));
    assert!(prompt.contains("Always follow repo safety rules."));

    let _ = fs::remove_dir_all(root);
}

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: String::new(),
            parameters: serde_json::json!({}),
        },
    }
}

fn skill(name: &str, description: &str) -> SkillManifest {
    SkillManifest {
        name: name.to_string(),
        version: "1.0.0".to_string(),
        description: description.to_string(),
        author: None,
        tools: Vec::new(),
        tool_groups: Vec::new(),
        mcp_servers: Vec::new(),
        disable_builtin_tools: false,
        disable_mcp_tools: false,
        prompt: String::new(),
        system_prompt: None,
        priority: 0,
        excludes: Vec::new(),
        parent: None,
        source_path: Some(format!("builtin:{name}.skill")),
        resource_path: None,
    }
}

fn skill_with_prompt(name: &str, description: &str, prompt: &str) -> SkillManifest {
    let mut skill = skill(name, description);
    skill.prompt = prompt.to_string();
    skill
}

fn agent(name: &str, mcp_servers: Vec<&str>) -> AgentManifest {
    AgentManifest {
        name: name.to_string(),
        description: String::new(),
        mode: AgentMode::Primary,
        model: None,
        temperature: None,
        max_steps: None,
        prompt: String::new(),
        system_prompt: None,
        tools: Vec::new(),
        tool_groups: Vec::new(),
        mcp_servers: mcp_servers.into_iter().map(|s| s.to_string()).collect(),
        disable_mcp_tools: false,
        model_tier: None,
        disabled: false,
        hidden: false,
        color: None,
        source_path: None,
    }
}

#[test]
fn manifest_skill_group_does_not_expose_request_user_input_without_active_skill() {
    let mut active_agent = agent("ordinary", vec![]);
    active_agent.tool_groups.push("skill".to_string());

    let tools = super::builtin_tools_for_skill(&[], Some(&active_agent));

    assert!(
        !tools
            .iter()
            .any(|tool| tool.function.name == "request_user_input"),
        "manifest tool_groups must not expose the skill-only handoff control tool"
    );
}

#[test]
fn active_agent_mcp_servers_auto_load_matching_mcp_tools() {
    let all_tools = vec![
        tool("mcp_feishu_docs_search"),
        tool("mcp_feishu_docs_get_text_by_url"),
        tool("mcp_other_lookup"),
    ];
    let build_agent = agent("build", vec!["feishu"]);
    let allowed_servers = build_agent.mcp_servers.clone();

    let tools = filter_mcp_tools_by_allowed_servers(all_tools, &allowed_servers);
    let names = tools
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();

    assert!(names.contains(&"mcp_feishu_docs_search".to_string()));
    assert!(names.contains(&"mcp_feishu_docs_get_text_by_url".to_string()));
    assert!(!names.contains(&"mcp_other_lookup".to_string()));
}

#[test]
fn agent_without_mcp_servers_field_lazy_loads_mcp_tools() {
    // With no mcp_servers whitelist, lazy-load by default: pre-mount no MCP tool schemas;
    // the model loads them on demand via the hidden MCP catalog + enable_tools.
    let all_tools = vec![
        tool("mcp_feishu_docs_search"),
        tool("mcp_ocr_extract"),
        tool("mcp_other_lookup"),
    ];
    let build_agent = agent("build", vec![]);

    let tools = select_mcp_tools(all_tools, &[], Some(&build_agent));

    assert!(tools.is_empty());
}

#[test]
fn agent_disable_mcp_tools_hides_default_mcp_tools() {
    let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_ocr_extract")];
    let mut build_agent = agent("build", vec![]);
    build_agent.disable_mcp_tools = true;

    let tools = select_mcp_tools(all_tools, &[], Some(&build_agent));

    assert!(tools.is_empty());
}

#[test]
fn skill_mcp_servers_can_opt_in_when_agent_disables_mcp_tools() {
    let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_ocr_extract")];
    let mut build_agent = agent("build", vec![]);
    build_agent.disable_mcp_tools = true;
    let mut s = skill("feishu-docs", "");
    s.mcp_servers = vec!["feishu".to_string()];

    let tools = select_mcp_tools(all_tools, &[&s], Some(&build_agent));
    let names = tools
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();

    assert_eq!(names, vec!["mcp_feishu_docs_search".to_string()]);
}

#[test]
fn no_active_agent_or_skill_lazy_loads_mcp_tools() {
    // With neither a skill nor an agent whitelist, lazy-load by default too: return empty and leave loading to
    // the hidden MCP catalog + enable_tools on demand.
    let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_other_lookup")];

    let tools = select_mcp_tools(all_tools, &[], None);

    assert!(tools.is_empty());
}

#[test]
fn skill_disable_mcp_tools_overrides_agent_default_fallback() {
    let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_other_lookup")];
    let build_agent = agent("build", vec![]);
    let mut s = skill("focus", "");
    s.disable_mcp_tools = true;

    let tools = select_mcp_tools(all_tools, &[&s], Some(&build_agent));
    assert!(tools.is_empty());
}

#[test]
fn explicit_agent_whitelist_still_narrows_when_set() {
    let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_ocr_extract")];
    let agent = agent("build", vec!["feishu"]);

    let tools = select_mcp_tools(all_tools, &[], Some(&agent));
    let names = tools
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();

    assert_eq!(names, vec!["mcp_feishu_docs_search".to_string()]);
}

#[test]
fn runtime_enabled_tools_are_preserved_when_refreshing_context() {
    let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
    set_explicit_enabled_tool_names(vec![
        "enable_tools".to_string(),
        "knowledge_search".to_string(),
    ]);
    let merged = merge_with_runtime_enabled_tools(
        vec![tool("read_file"), tool("enable_tools")],
        vec![],
        &[
            tool("read_file"),
            tool("enable_tools"),
            tool("knowledge_search"),
        ],
    );
    let names = merged
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"read_file".to_string()));
    assert!(names.contains(&"enable_tools".to_string()));
    assert!(names.contains(&"knowledge_search".to_string()));
    set_explicit_enabled_tool_names(Vec::new());
}

#[test]
fn runtime_enabled_builtin_is_restored_when_current_context_missed_writeback() {
    let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
    set_explicit_enabled_tool_names(vec!["knowledge_consolidate".to_string()]);

    let merged = merge_with_runtime_enabled_tools(vec![tool("read_file")], vec![], &[]);
    let names = merged
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();

    assert!(names.contains(&"read_file".to_string()));
    assert!(names.contains(&"knowledge_consolidate".to_string()));
    set_explicit_enabled_tool_names(Vec::new());
}

#[test]
fn subagent_runtime_enabled_task_tools_stay_hidden() {
    let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
    SUBAGENT_DEPTH.sync_scope(1, || {
        // The explicit-enabled list is isolated per owner (including subagent context); it must be written
        // within sync_scope so reads use the same owner.
        set_explicit_enabled_tool_names(vec![
            "task_wait".to_string(),
            "task_cancel".to_string(),
            "knowledge_search".to_string(),
        ]);
        let merged = merge_with_runtime_enabled_tools(
            vec![tool("read_file"), tool("enable_tools")],
            vec![],
            &[
                tool("task_wait"),
                tool("task_cancel"),
                tool("knowledge_search"),
            ],
        );
        let names = merged
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"knowledge_search".to_string()));
        assert!(!names.contains(&"task_wait".to_string()));
        assert!(!names.contains(&"task_cancel".to_string()));
        set_explicit_enabled_tool_names(Vec::new());
    });
}

#[test]
fn non_explicit_skill_tools_do_not_leak_into_next_context() {
    let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
    set_explicit_enabled_tool_names(vec!["knowledge_search".to_string()]);
    let merged = merge_with_runtime_enabled_tools(
        vec![tool("read_file")],
        vec![],
        &[
            tool("read_file"),
            tool("apply_patch"),
            tool("knowledge_search"),
        ],
    );
    let names = merged
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"read_file".to_string()));
    assert!(names.contains(&"knowledge_search".to_string()));
    assert!(!names.contains(&"apply_patch".to_string()));
    set_explicit_enabled_tool_names(Vec::new());
}

#[test]
fn explicit_tool_lists_keep_baseline_entries_available() {
    let merged = ensure_required_baseline_tools(vec![tool("read_file")]);
    let names = merged
        .into_iter()
        .map(|tool| tool.function.name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"enable_tools".to_string()));
    // Basic read-only / retrieval capabilities should be re-added as the resident baseline, so a narrow-whitelist skill cannot
    // cull the most basic reading tools like read_file and leave the main Agent unable to read user-named files.
    assert!(names.contains(&"read_file".to_string()));
    // Task orchestration / knowledge memory series have been moved out of the baseline: under narrow-whitelist skills they are no longer
    // auto-re-added, but the main Agent can still progressively discover and enable them via the resident enable_tools,
    // so delegation of subagents is never truly lost.
    assert!(!names.contains(&"task".to_string()));
    assert!(!names.contains(&"task_spawn".to_string()));
    assert!(!names.contains(&"task_wait".to_string()));
    assert!(!names.contains(&"task_status".to_string()));
    assert!(!names.contains(&"knowledge_save".to_string()));
    // Other non-baseline builtin tools still must not be dragged into the whitelist without cause.
    assert!(!names.contains(&"plan".to_string()));
    assert!(!names.contains(&"write_file".to_string()));
    assert!(!names.contains(&"apply_patch".to_string()));
}

#[test]
fn multi_skill_merges_tool_groups_from_all_skills() {
    let mut skill_a = skill("alpha", "alpha skill");
    skill_a.tool_groups.push("skill".to_string());
    let mut skill_b = skill("beta", "beta skill");
    skill_b.tool_groups.push("builtin".to_string());
    let tools = super::builtin_tools_for_skill(&[&skill_a, &skill_b], None);
    let names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
    assert!(names.contains(&"enable_tools".to_string()));
    // The skill group provides list_skills / activate_skill; the builtin group provides enable_tools
    assert!(names.contains(&"list_skills".to_string()));
    assert!(names.contains(&"activate_skill".to_string()));
}

#[test]
fn multi_skill_system_prompt_lists_all_active_skills() {
    let skill_a = skill_with_prompt("alpha", "alpha skill", "Do alpha things.");
    let skill_b = skill_with_prompt("beta", "beta skill", "Do beta things.");
    let available_tools: Box<SkipSet<String>> = Box::new(SkipSet::new(16));
    let ctx = PromptContext {
        goal_mode: None,
        is_background: false,
    };
    let builder = super::build_system_prompt(None, &[&skill_a, &skill_b], &available_tools, &ctx);
    let prompt = builder.render_system_prompt();
    assert!(prompt.contains("alpha"));
    assert!(prompt.contains("beta"));
    assert!(prompt.contains("activation order"));
    assert!(prompt.contains("Do alpha things."));
    assert!(prompt.contains("Do beta things."));
}

#[test]
fn multi_skill_any_disable_builtin_tools_disables_all() {
    let skill_a = skill("alpha", "alpha skill");
    let mut skill_b = skill("beta", "beta skill");
    skill_b.disable_builtin_tools = true;
    let tools = super::builtin_tools_for_skill(&[&skill_a, &skill_b], None);
    assert!(tools.is_empty());
}

#[test]
fn multi_skill_any_disable_mcp_tools_disables_all() {
    // select_mcp_tools checks disable_mcp_tools: any skill disabling it returns empty
    let mcp_tool = tool("mcp_server_a_lookup");
    let mut skill_a = skill("alpha", "alpha skill");
    skill_a.mcp_servers.push("server_a".to_string());
    let mut skill_b = skill("beta", "beta skill");
    skill_b.disable_mcp_tools = true;
    let tools = super::select_mcp_tools(vec![mcp_tool], &[&skill_a, &skill_b], None);
    assert!(tools.is_empty());
}

#[test]
fn multi_skill_merges_mcp_servers_deduplicated() {
    let mut skill_a = skill("alpha", "alpha skill");
    skill_a.mcp_servers.push("server_a".to_string());
    skill_a.mcp_servers.push("server_b".to_string());
    let mut skill_b = skill("beta", "beta skill");
    skill_b.mcp_servers.push("server_b".to_string());
    skill_b.mcp_servers.push("server_c".to_string());
    let servers = super::resolved_mcp_servers(&[&skill_a, &skill_b], None);
    assert_eq!(servers, vec!["server_a", "server_b", "server_c"]);
}
