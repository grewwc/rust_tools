// =============================================================================
// AIOS Driver - Agent Operating System Main Entry
// =============================================================================
// This module is the main entry point for the AIOS system.
// It handles:
// - CLI argument parsing and config loading
// - Session management (history, state persistence)
// - Process OS initialization (kernel creation)
// - MCP client initialization
// - Agent loading and auto-routing
// - The main run_loop() that coordinates foreground and background processes
// 
// Key concepts:
//   - App: Main application state holding all runtime information
//   - run(): Async entry point, initializes everything and starts run_loop
//   - run_loop(): Main event loop that handles:
//     1. Scheduler ticks (advance_tick for background processes)
//     2. Background process execution (pop_all_ready)
//     3. Foreground input handling (input::next_question)
//     4. Running turns (turn_runtime::run_turn)
// =============================================================================

use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use uuid::Uuid;

use crate::ai::{
    agents::{self, AgentManifest},
    cli::{self},
    config,
    history::SessionStore,
    mcp::{McpClient, SharedMcpClient},
    models,
    prompt::PromptEditor,
    skills::{self, SkillManifest},
    types::{AgentContext, App},
};
use crate::commonw::configw;

pub mod agent_router;
pub mod commands;
pub mod decision_log;
pub mod input;
pub mod intent_model;
pub mod intent_recognition;
pub mod mcp_init;
pub mod model;
pub mod os;
pub mod observer;
pub mod params;
pub mod print;
pub mod reflection;
pub mod signal;
pub mod skill_match_model;
pub mod skill_matching;
pub mod skill_ranking;
pub mod skill_runtime;
pub mod text_similarity;
pub mod thinking;
pub mod tools;
pub mod turn_runtime;

pub use commands::try_handle_interactive_command;
pub use mcp_init::*;
pub use model::*;
pub use skill_matching::*;
pub use skill_ranking::*;
pub use text_similarity::*;

pub(crate) fn new_local_kernel() -> crate::ai::kernel::SharedKernel {
    crate::ai::kernel::new_shared_kernel(os::LocalOS::new())
}

/// Default max LLM iterations allowed per turn (prevents infinite loops)
const DEFAULT_MAX_ITERATIONS: usize = 1024;

/// Max iterations for subagent (executor) processes
const EXECUTOR_MAX_ITERATIONS: usize = 64;

#[crate::ai::agent_hang_span(
    "pre-fix",
    "S",
    "driver::run:load_all_skills",
    "[DEBUG] loading skills",
    "[DEBUG] loaded skills",
    { "no_skills": no_skills },
    {
        "count": __agent_hang_result.len(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
fn load_skill_manifests(no_skills: bool) -> Vec<SkillManifest> {
    if no_skills {
        Vec::new()
    } else {
        skills::load_all_skills()
    }
}

/// Activate a primary agent for the current session.
/// Updates app's current_agent, current_agent_manifest,
/// and switches the model if specified by the agent.
fn activate_primary_agent(app: &mut App, agent: &AgentManifest) {
    app.current_agent = agent.name.clone();
    app.current_agent_manifest = Some(agent.clone());
    if let Some(model) = &agent.model {
        app.current_model = model.clone();
    }
}

/// Check if auto-agent routing is enabled in config.
/// Auto-routing selects the best agent based on the question content.
fn auto_agent_routing_enabled() -> bool {
    !configw::get_all_config()
        .get_opt("ai.agents.auto_route.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false")
}

/// Get auto-routing strategy from config: "model" or "heuristic".
fn auto_route_strategy() -> String {
    configw::get_all_config()
        .get_opt("ai.agents.auto_route.strategy")
        .unwrap_or_else(|| "model".to_string())
        .trim()
        .to_lowercase()
}

/// Auto-route to a different agent based on question content.
/// Activated when:
///   1. No explicit agent specified via CLI (-a/--agent)
///   2. Auto-routing is enabled in config
///   3. A suitable agent is found based on the question
/// 
/// Uses either:
///   - model strategy: Use ML model to predict best agent
///   - heuristic strategy: Rule-based routing
fn maybe_auto_route_agent(app: &mut App, agent_manifests: &[AgentManifest], question: &str) {
    if app.cli.agent.is_some() || !auto_agent_routing_enabled() {
        return;
    }

    let history = read_recent_history(app);
    let decision = match auto_route_strategy().as_str() {
        "heuristic" => {
            let router = agent_router::HeuristicRouter;
            agent_router::AgentRouter::route(
                &router,
                agent_manifests,
                question,
                &history,
                &app.current_agent,
            )
        }
        _ => {
            let model_path = app.config.agent_route_model_path.clone();
            let router = agent_router::ModelRouter::new(model_path);
            agent_router::AgentRouter::route(
                &router,
                agent_manifests,
                question,
                &history,
                &app.current_agent,
            )
        }
    };
    let Some(decision) = decision else {
        return;
    };

    let Some(agent) = agents::find_agent_by_name(agent_manifests, &decision.agent_name) else {
        return;
    };
    if !agent.is_primary() || agent.disabled {
        return;
    }

    let old_agent = app.current_agent.clone();
    activate_primary_agent(app, agent);

    println!(
        "\n[Agent 自动切换: {} → {}] (原因: {})\n",
        old_agent, app.current_agent, decision.reason
    );
}

/// Read recent history entries from the session file.
/// Used by auto-routing to understand conversation context.
fn read_recent_history(app: &App) -> Vec<crate::ai::history::Message> {
    use crate::ai::history::build_message_arr;
    match build_message_arr(usize::MAX, &app.session_history_file) {
        Ok(entries) => entries.into_iter().rev().take(10).collect(),
        Err(_) => Vec::new(),
    }
}

/// Loads all agents fresh from disk, enabling hot-reload of newly added/modified agents.
/// Returns the updated manifests.
fn reload_agent_manifests(agent_manifests: &mut Vec<AgentManifest>) {
    let new_agents = agents::load_all_agents();
    if new_agents.len() != agent_manifests.len() {
        let added = new_agents.len() as i64 - agent_manifests.len() as i64;
        if added > 0 {
            println!("[Agent 发现] 新发现 {} 个 agent(s)，已自动加载", added);
        } else {
            println!("[Agent 发现] agent 列表已更新，共 {} 个", new_agents.len());
        }
        *agent_manifests = new_agents;
    }
}

/// Main entry point for AIOS.
/// Initializes all components and starts the run_loop.
/// 
/// Initialization steps:
///   1. Parse CLI arguments
///   2. Load config
///   3. Create session store and session ID
///   4. Setup signal handlers (Ctrl+C)
///   5. Create output writer (for -o flag)
///   6. Initialize HTTP client
///   7. Create local kernel (process OS)
///   8. Load skills and MCP clients
///   9. Load and activate agents
///   10. Enter run_loop
pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli::parse_cli_args(std::env::args());
    let config = config::load_config()?;
    let session_store = SessionStore::new(config.history_file.as_path());
    let session_arg = cli.session.clone().unwrap_or_default();
    let session_id = if session_arg.trim().is_empty() {
        Uuid::new_v4().to_string()
    } else {
        session_arg.trim().to_string()
    };

    if cli.help {
        cli::print_help();
        return Ok(());
    }

    if let Err(err) = session_store.ensure_root_dir() {
        eprintln!("[Warning] Failed to create sessions dir: {}", err);
    }

    let shutdown = Arc::new(AtomicBool::new(false));
    let streaming = Arc::new(AtomicBool::new(false));
    let cancel_stream = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&shutdown);
    let streaming_flag = Arc::clone(&streaming);
    let cancel_stream_flag = Arc::clone(&cancel_stream);
    ctrlc::set_handler(move || {
        signal::handle_sigint(
            signal_flag.as_ref(),
            streaming_flag.as_ref(),
            cancel_stream_flag.as_ref(),
        );
    })?;

    let writer =
        config::open_output_writer(cli.out.as_deref())?.map(|f| Arc::new(std::sync::Mutex::new(f)));
    let current_model = models::initial_model(&cli);
    let client = reqwest::Client::builder().build()?;
    let prompt_editor = if cli.args.is_empty() {
        Some(PromptEditor::new(
            &session_id,
            config.history_file.as_path(),
        ))
    } else {
        None
    };

    let os_arc = new_local_kernel();
    crate::ai::tools::os_tools::init_os_tools_globals(os_arc.clone());

    let mut app = App {
        pending_files: if cli.files.trim().is_empty() {
            None
        } else {
            Some(cli.files.clone())
        },
        pending_short_output: cli.short_output,
        current_model,
        current_agent: "build".to_string(),
        current_agent_manifest: None,
        session_id: session_id.clone(),
        session_history_file: session_store.session_history_file(&session_id),
        cli,
        config,
        client,
        attached_image_files: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        ignore_next_prompt_interrupt: false,
        writer,
        prompt_editor,
        agent_context: Some(AgentContext {
            tools: super::tools::tool_definitions_for_groups(&["core"]),
            mcp_servers: rust_tools::commonw::FastMap::default(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }),
        last_skill_bias: None,
        os: os_arc,
        agent_reload_counter: None,
            observers: vec![Box::new(crate::ai::driver::thinking::ThinkingOrchestrator::new())],
        };

    let mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
    let skill_manifests = load_skill_manifests(app.cli.no_skills);
    let mcp_report = init_mcp(
        &mut app,
        &mut mcp_client.lock().unwrap_or_else(|err| err.into_inner()),
    )
    .await;

    if app.cli.list_tools {
        print::print_builtin_tools(&app);
        return Ok(());
    }
    if app.cli.list_skills {
        print::print_skills(&skill_manifests);
        return Ok(());
    }
    if app.cli.list_mcp_tools {
        print::print_mcp_tools(&mcp_report, &mcp_client.lock().unwrap());
        return Ok(());
    }

    let mut agent_manifests = agents::load_all_agents();
    if app.cli.list_agents {
        commands::help::print_agents_list(&agent_manifests);
        return Ok(());
    }

    if let Some(default_agent) = agents::find_agent_by_name(&agent_manifests, &app.current_agent)
        && default_agent.is_primary()
        && !default_agent.disabled
    {
        activate_primary_agent(&mut app, default_agent);
    }

    if let Some(agent_name) = &app.cli.agent {
        if let Some(agent) = agents::find_agent_by_name(&agent_manifests, agent_name) {
            if agent.is_primary() && !agent.disabled {
                activate_primary_agent(&mut app, agent);
                println!("[agent] using: {}", agent.name);
            } else {
                eprintln!(
                    "[Warning] Agent '{}' is not available, using default",
                    agent_name
                );
            }
        } else {
            eprintln!("[Warning] Agent '{}' not found, using default", agent_name);
        }
    }

    if mcp_report.loaded {
        println!(
            "{}",
            print::format_section_header(
                "mcp",
                Some(&format!(
                    "{} servers, {} tools",
                    mcp_report.server_count, mcp_report.tool_count
                ))
            )
        );
    }
    run_loop(
        &mut app,
        &mcp_client,
        &skill_manifests,
        &mut agent_manifests,
    )
    .await
}

/// Generate history file path for a background process.
/// appends .proc-{pid} to the session history filename.
fn process_history_path(base: &Path, pid: u64) -> PathBuf {
    let file_name = base
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{name}.proc-{pid}"))
        .unwrap_or_else(|| format!("session.proc-{pid}"));
    base.with_file_name(file_name)
}

/// Main event loop for AIOS.
/// Coordinates execution of both foreground and background processes.
/// 
/// Loop structure per iteration:
///   1. Scheduler tick: advance_tick() to wake sleeping processes
///   2. Agent hot-reload: check for new agents every 5 ticks
///   3. Shutdown check: exit if shutdown flag is set
///   4. Background execution:
///      - pop_all_ready() to get ready processes
///      - spawn async tasks for each
///      - wait for all to complete
///   5. Foreground input:
///      - get next question from input::next_question()
///      - handle interactive commands
///      - run turn via turn_runtime::run_turn()
///   6. Termination check: exit if quit requested
/// 
/// one_shot_mode: When CLI args provided (non-interactive)
///   - runs once and exits
///   - deletes session after completion
async fn run_loop(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &[SkillManifest],
    agent_manifests: &mut Vec<AgentManifest>,
) -> Result<(), Box<dyn std::error::Error>> {
    let one_shot_mode = !app.cli.args.is_empty();
    let mut should_quit = one_shot_mode;

    let cleanup_one_shot = |app: &App| {
        if one_shot_mode {
            let store = SessionStore::new(app.config.history_file.as_path());
            let _ = store.delete_session(&app.session_id);
        }
    };
    let handle_post_command = |app: &App, should_quit: &mut bool| {
        if *should_quit {
            cleanup_one_shot(app);
            true
        } else {
            *should_quit = false;
            false
        }
    };

    loop {
        {
            let mut os = app.os.lock().unwrap_or_else(|err| err.into_inner());
            os.advance_tick();
        }

        if let Some(counter) = app.agent_reload_counter.as_mut() {
            *counter += 1;
            if *counter % 5 == 0 {
                reload_agent_manifests(agent_manifests);
            }
        } else {
            app.agent_reload_counter = Some(0);
        }

        if app.shutdown.load(Ordering::Relaxed) {
            cleanup_one_shot(app);
            return Ok(());
        }

        let mut history_count = usize::MAX;
        let mut question = String::new();

        let background_procs: Vec<crate::ai::kernel::Process> = {
            let mut os = app.os.lock().unwrap();
            os.pop_all_ready(4)
        };

        if !background_procs.is_empty() {
            use colored::Colorize;
            for proc in &background_procs {
                println!(
                    "\n{} Process {} ({})",
                    "[OS Dispatch]".bright_blue().bold(),
                    proc.pid,
                    proc.name
                );
            }

            let original_history_file = app.session_history_file.clone();

            let mut task_specs: Vec<(u64, String, PathBuf)> = Vec::new();
            for proc in &background_procs {
                let pid = proc.pid;
                let proc_question = if !proc.mailbox.is_empty() {
                    let messages: Vec<String> = proc.mailbox.iter().cloned().collect();
                    {
                        let mut os = app.os.lock().unwrap();
                        if let Some(actual) = os.get_process_mut(pid) {
                            actual.mailbox.clear();
                        }
                    }
                    format!(
                        "[Process {} Woke Up] Original goal: {}\nNew mailbox messages:\n{}\n\nWake-up handling rules:\n- If the mailbox indicates async tool wake-up, first decide whether you need `tool_status` for a full snapshot, `tool_wait` to collect newly finished results, or `tool_cancel` to stop low-value branches.\n- Do not blindly wait again if enough completed results already support the answer.\n- Prefer continuing reasoning immediately when the wake-up messages already identify the relevant finished tasks.\n\nResume execution based on the goal and these messages.",
                        pid,
                        proc.goal,
                        messages.join("\n---\n")
                    )
                } else {
                    format!(
                        "[Process {}] Goal: {}\nExecute this goal autonomously and provide the final result.",
                        pid, proc.goal
                    )
                };

                {
                    let mut os = app.os.lock().unwrap();
                    os.set_current_pid(Some(pid));
                    if let Some(p) = os.get_process_mut(pid) {
                        if p.history_file.is_none() {
                            p.history_file =
                                Some(process_history_path(&original_history_file, pid));
                        }
                        let _ = os.process_pending_signals();
                    }
                }

                let history_path = process_history_path(&original_history_file, pid);
                task_specs.push((pid, proc_question, history_path));
            }

            let mut join_set: tokio::task::JoinSet<(
                u64,
                Result<turn_runtime::TurnOutcome, String>,
            )> = tokio::task::JoinSet::new();

            for (pid, proc_question, history_path) in task_specs {
                let mut task_app = app.clone();
                task_app.session_history_file = history_path;
                task_app.cancel_stream.store(false, Ordering::Relaxed);
                let task_mcp = mcp_client.clone();
                let task_skills = skill_manifests.to_vec();
                let next_model = app.current_model.clone();

                join_set.spawn(crate::ai::kernel::TASK_PID.scope(Some(pid), async move {
                    crate::ai::tools::registry::common::clear_tool_cancel();
                    let result = turn_runtime::run_turn(
                        &mut task_app,
                        &task_mcp,
                        &task_skills,
                        usize::MAX,
                        proc_question,
                        next_model,
                        None,
                        false,
                        false,
                    )
                    .await
                    .map_err(|e| format!("{}", e));
                    (pid, result)
                }));
            }

            while let Some(join_result) = join_set.join_next().await {
                let (pid, turn_result) = match join_result {
                    Ok(pair) => pair,
                    Err(join_err) => {
                        eprintln!("[OS] Task panicked: {}", join_err);
                        continue;
                    }
                };

                {
                    let mut os = app.os.lock().unwrap();
                    os.set_current_pid(Some(pid));
                    match turn_result {
                        Ok(_outcome) => {
                            os.increment_turns_used_for(pid);
                            let mut should_terminate = true;
                            let mut termination_result = "Completed".to_string();
                            if let Some(p) = os.get_process_mut(pid) {
                                if p.quota_turns > 0 {
                                    p.quota_turns -= 1;
                                }
                                if p.quota_turns == 0 {
                                    termination_result =
                                        "Terminated: Max LLM quota reached.".to_string();
                                }
                                if matches!(
                                    p.state,
                                    crate::ai::kernel::ProcessState::Waiting { .. }
                                        | crate::ai::kernel::ProcessState::Sleeping { .. }
                                        | crate::ai::kernel::ProcessState::Stopped
                                ) {
                                    should_terminate = false;
                                }
                            }
                            if should_terminate {
                                os.cleanup_process_resources(pid);
                                os.set_current_pid(Some(pid));
                                os.terminate_current(termination_result);
                                os.drop_terminated(pid);
                            } else if os.is_round_robin() {
                                os.set_current_pid(Some(pid));
                                os.requeue_current();
                            }
                        }
                        Err(err) => {
                            os.cleanup_process_resources(pid);
                            os.set_current_pid(Some(pid));
                            os.terminate_current(format!("Failed: {}", err));
                            os.drop_terminated(pid);
                        }
                    }

                    let restarted = os.check_daemon_restart();
                    if !restarted.is_empty() {
                        use colored::Colorize;
                        for rid in &restarted {
                            println!(
                                "{} Daemon process {} restarted.",
                                "[OS]".bright_blue().bold(),
                                rid
                            );
                        }
                    }
                }
            }

            continue;
        }

        {
            let Some(ctx) = input::next_question(app)? else {
                cleanup_one_shot(app);
                return Ok(());
            };
            if ctx.question.trim().is_empty() {
                should_quit = false;
                continue;
            }
            question = ctx.question;
            history_count = ctx.history_count;
        }

        if try_handle_interactive_command(app, mcp_client, &question, agent_manifests)? {
            if handle_post_command(app, &mut should_quit) {
                return Ok(());
            }
            continue;
        }
        maybe_auto_route_agent(app, &*agent_manifests, &question);
        let precomputed_ocr = if !app.attached_image_files.is_empty()
            && !crate::ai::models::is_vl_model(&app.current_model)
        {
            crate::ai::driver::model::ocr_images_for_attached_input(
                mcp_client,
                &app.attached_image_files,
            )
            .ok()
            .flatten()
        } else {
            None
        };
        let ocr_succeeded_for_images = precomputed_ocr
            .as_ref()
            .map(|ocr| ocr.images.iter().all(|image| image.error.is_none()))
            .unwrap_or(false);
        let next_model = resolve_model_for_input(app, ocr_succeeded_for_images, &mut question);
        app.current_model = next_model.clone();

        {
            let mut os = app.os.lock().unwrap();
            os.begin_foreground(
                "foreground".to_string(),
                question.clone(),
                10,
                usize::MAX,
                None,
            );
        }

        let original_history_file = app.session_history_file.clone();
        let original_writer = app.writer.clone();

        app.cancel_stream.store(false, Ordering::Relaxed);
        crate::ai::tools::registry::common::clear_tool_cancel();

        {
            let mut os = app.os.lock().unwrap();
            if os.process_pending_signals() {
                app.session_history_file = original_history_file;
                app.writer = original_writer;
                continue;
            }
        }

        let fg_pid = {
            let os = app.os.lock().unwrap();
            os.current_process_id()
        };

        let turn_outcome = crate::ai::kernel::TASK_PID
            .scope(
                fg_pid,
                turn_runtime::run_turn(
                    app,
                    mcp_client,
                    skill_manifests,
                    history_count,
                    question,
                    next_model,
                    precomputed_ocr,
                    one_shot_mode,
                    should_quit,
                ),
            )
            .await;

        match turn_outcome {
            Ok(outcome) => {
                let mut os = app.os.lock().unwrap();
                let current_pid = os.current_process_id();
                let mut should_terminate = true;
                let mut termination_result = "Completed".to_string();

                if let Some(pid) = current_pid {
                    os.increment_turns_used_for(pid);
                    if let Some(proc) = os.get_process_mut(pid) {
                        if proc.quota_turns > 0 {
                            proc.quota_turns -= 1;
                        }
                        if proc.quota_turns == 0 {
                            termination_result = "Terminated: Max LLM quota reached.".to_string();
                        }

                        if matches!(
                            proc.state,
                            crate::ai::kernel::ProcessState::Waiting { .. }
                                | crate::ai::kernel::ProcessState::Sleeping { .. }
                                | crate::ai::kernel::ProcessState::Stopped
                        ) {
                            should_terminate = false;
                        }
                    }
                }

                if should_terminate {
                    if let Some(pid) = current_pid {
                        os.cleanup_process_resources(pid);
                        os.terminate_current(termination_result);
                        os.drop_terminated(pid);
                    }
                }

                let restarted = os.check_daemon_restart();
                if !restarted.is_empty() {
                    use colored::Colorize;
                    for pid in &restarted {
                        println!(
                            "{} Daemon process {} restarted.",
                            "[OS]".bright_blue().bold(),
                            pid
                        );
                    }
                }

                if os.is_round_robin() && os.has_ready() {
                    os.requeue_current();
                }
                outcome
            }
            Err(err) => {
                let mut os = app.os.lock().unwrap();
                let current_pid = os.current_process_id();
                if let Some(pid) = current_pid {
                    os.cleanup_process_resources(pid);
                }
                os.terminate_current(format!("Failed: {}", err));
                if let Some(pid) = current_pid {
                    os.drop_terminated(pid);
                }
                app.session_history_file = original_history_file;
                app.writer = original_writer;
                eprintln!("[Error] 当前轮请求失败：{}", err);
                if one_shot_mode || should_quit {
                    cleanup_one_shot(app);
                    return Err(err);
                }
                eprintln!("[Info] 会话保持运行，请继续输入下一条消息。\n");
                should_quit = false;
                continue;
            }
        };
        app.session_history_file = original_history_file;
        app.writer = original_writer;
        if matches!(turn_outcome, Ok(turn_runtime::TurnOutcome::Quit)) || should_quit {
            for obs in app.observers.iter_mut() {
                if obs.is_poisoned() {
                    continue;
                }
                let obs_name = obs.name().to_string();
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    obs.on_conversation_end();
                })).is_err() {
                    eprintln!("[Warning] observer '{}' panicked in on_conversation_end; disabling.", obs_name);
                    obs.mark_poisoned();
                }
            }
            cleanup_one_shot(app);
            return Ok(());
        }
        if let Some(writer) = app.writer.as_ref() {
            let mut guard = writer.lock().unwrap();
            guard.write_all(b"\n---\n")?;
            guard.flush()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::maybe_auto_route_agent;
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
    use crate::ai::cli::ParsedCli;
    use crate::ai::types::{AgentContext, App, AppConfig};
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    fn primary_agent(name: &str, description: &str, routing_tags: &[&str]) -> AgentManifest {
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
            routing_tags: routing_tags.iter().map(|tag| (*tag).to_string()).collect(),
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
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 12000,
                history_keep_last: 8,
                history_summary_max_chars: 4000,
                intent_model: None,
                intent_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/intent/intent_model.json"),
                agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            client: reqwest::Client::new(),
            current_model: "test-model".to_string(),
            current_agent: current_agent.to_string(),
            current_agent_manifest: None,
            pending_files: None,
            pending_short_output: false,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            writer: None,
            prompt_editor: None,
            agent_context: Some(AgentContext {
                tools: Vec::new(),
                mcp_servers: Default::default(),
                max_iterations: super::DEFAULT_MAX_ITERATIONS,
            }),
            last_skill_bias: None,
            os: super::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(crate::ai::driver::thinking::ThinkingOrchestrator::new())],
        }
    }

    #[test]
    fn auto_route_falls_back_to_build_instead_of_current_agent() {
        let build = primary_agent(
            "build",
            "Default agent for development work",
            &["fix", "debug"],
        );
        let prompt_skill = primary_agent(
            "prompt-skill",
            "Specialized agent for optimizing and generating prompts and skills",
            &["prompt", "skill", "optimize"],
        );
        let mut app = test_app("prompt-skill");

        maybe_auto_route_agent(
            &mut app,
            &[build.clone(), prompt_skill.clone()],
            "这个问题为什么会这样？",
        );

        assert_eq!(app.current_agent, "build");
        assert_eq!(
            app.current_agent_manifest
                .as_ref()
                .map(|agent| agent.name.as_str()),
            Some("build")
        );
    }
}
