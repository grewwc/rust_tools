// =============================================================================
// AIOS Driver - Agent Operating System Main Entry
// =============================================================================
// This module is the main entry point for the AIOS system.
// It handles:
// - CLI argument parsing and config loading
// - Session management (history, state persistence)
// - Process OS initialization (kernel creation)
// - MCP client initialization
// - Agent loading and activation
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
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use aios_kernel::primitives::{RlimitDim, RlimitVerdict};
use tokio::sync::Notify;

use crate::ai::{
    agents::{self, AgentManifest},
    cli::{self},
    config,
    config_schema::AiConfig,
    history::{SessionStore, SuspendedSessionStore},
    mcp::{McpClient, SharedMcpClient},
    models,
    prompt::PromptEditor,
    skills::SkillManifest,
    types::{AgentContext, App},
};
use crate::commonw::configw;

mod agent_routing;
mod background_dispatch;
pub mod commands;
pub mod decision_log;
pub mod hook_registry;
pub mod hooks;
pub mod input;
pub mod mcp_init;
mod mcp_lifecycle;
pub mod model;
pub mod note_search;
pub mod observer;
pub mod print;
mod process_context;
pub mod reflection;
pub mod runtime_ctx;
mod scheduler;
mod session;
pub mod session_pid;
pub mod side_note;
pub mod signal;
pub mod skill_runtime;
mod skill_watcher;
pub mod thinking;
pub mod tools;
pub mod turn_runtime;

use agent_routing::*;
use background_dispatch::{SubagentStatusLine, dispatch_background_batch};
pub use commands::try_handle_interactive_command;
pub use commands::try_handle_local_command;
pub use mcp_init::*;
use mcp_lifecycle::*;
pub use model::*;
use process_context::*;
use scheduler::*;
use session::*;

tokio::task_local! {
    pub(super) static TASK_PID: Option<u64>;
}

fn current_task_pid() -> Option<u64> {
    TASK_PID.try_with(|v| *v).unwrap_or(None)
}

/// Number of background subagent tokio tasks currently dispatched and not yet finished.
///
/// Background subagents run on worker threads via `tokio::spawn` and stream to the
/// terminal with `println!` (raw `\n`). The interactive input box (multiline TUI) turns
/// on raw mode, which disables the TTY's ONLCR, so a raw `\n` no longer gets a `\r`
/// appended and each subagent output line shifts right (staircase misalignment).
///
/// Use this counter before opening the input box to tell whether background subagents
/// are still running: as long as it is > 0, do not enter the raw-mode input box; let the
/// scheduler loop keep ticking so subagents output normally in cooked mode, avoiding
/// garbled concurrent terminal writes (without losing any subagent output).
static BG_SUBAGENT_INFLIGHT: AtomicUsize = AtomicUsize::new(0);
static SCHEDULER_NOTIFY: Notify = Notify::const_new();
const SCHEDULER_TICK_DURATION: Duration = Duration::from_millis(10);

pub(crate) fn notify_scheduler() {
    SCHEDULER_NOTIFY.notify_one();
}

pub(crate) fn notify_scheduler_after(delay: Duration) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    runtime.spawn(async move {
        tokio::time::sleep(delay).await;
        notify_scheduler();
    });
}

#[derive(Default)]
struct SchedulerClock {
    partial_tick: Duration,
}

impl SchedulerClock {
    fn duration_until_ticks(&self, ticks: u64) -> Duration {
        let ticks = u32::try_from(ticks).unwrap_or(u32::MAX);
        SCHEDULER_TICK_DURATION
            .saturating_mul(ticks)
            .saturating_sub(self.partial_tick)
    }

    fn consume_elapsed(&mut self, elapsed: Duration) -> u64 {
        let total = self.partial_tick.saturating_add(elapsed);
        let tick_nanos = SCHEDULER_TICK_DURATION.as_nanos();
        let ticks = (total.as_nanos() / tick_nanos).min(u64::MAX as u128) as u64;
        self.partial_tick = Duration::from_nanos((total.as_nanos() % tick_nanos) as u64);
        ticks
    }

    async fn wait(&mut self, app: &App) {
        let kernel_wake_after = {
            let os = app.os.lock().unwrap_or_else(|err| err.into_inner());
            os.next_wakeup_tick().map(|wake_tick| {
                self.duration_until_ticks(wake_tick.saturating_sub(os.current_tick()).max(1))
            })
        };
        // task_wait uses a real wall-clock budget, so it cannot rely only on a one-shot
        // delayed notify. Fold its deadline into the scheduler wait so the foreground is
        // rescanned and woken on time even if a notification is lost.
        let task_wait_wake_after = crate::ai::tools::task_tools::next_task_wait_wakeup_delay();
        let wake_after = match (kernel_wake_after, task_wait_wake_after) {
            (Some(kernel), Some(task_wait)) => Some(kernel.min(task_wait)),
            (Some(delay), None) | (None, Some(delay)) => Some(delay),
            (None, None) => None,
        };
        let started_at = tokio::time::Instant::now();
        if let Some(delay) = wake_after {
            tokio::select! {
                biased;
                _ = tokio::time::sleep(delay) => {}
                _ = SCHEDULER_NOTIFY.notified() => {}
            }
        } else {
            SCHEDULER_NOTIFY.notified().await;
        }

        let elapsed_ticks = self.consume_elapsed(started_at.elapsed());
        if elapsed_ticks > 0 {
            let mut os = app.os.lock().unwrap_or_else(|err| err.into_inner());
            os.advance_ticks(elapsed_ticks);
        }
    }
}

fn bg_subagents_inflight() -> bool {
    BG_SUBAGENT_INFLIGHT.load(Ordering::Acquire) > 0
}

/// RAII guard: `inc` before dispatching a background subagent, automatically `dec` when
/// the subagent task ends (including on panic), so the count never leaks.
pub(super) struct BgSubagentGuard;

impl BgSubagentGuard {
    fn new() -> Self {
        BG_SUBAGENT_INFLIGHT.fetch_add(1, Ordering::AcqRel);
        notify_scheduler();
        BgSubagentGuard
    }
}

impl Drop for BgSubagentGuard {
    fn drop(&mut self) {
        BG_SUBAGENT_INFLIGHT.fetch_sub(1, Ordering::AcqRel);
        notify_scheduler();
    }
}

pub(crate) fn new_local_kernel() -> aios_kernel::kernel::SharedKernel {
    aios_kernel::kernel::new_shared_kernel(aios_kernel::local::LocalOS::new())
}

fn should_auto_drop_terminated(os: &dyn aios_kernel::kernel::Syscall, pid: u64) -> bool {
    os.get_process(pid)
        .map(|proc| proc.parent_pid.is_none())
        .unwrap_or(false)
}

/// Unified teardown flow for terminating a process: cleanup + drop.
///
/// When `set_current` is `true`, the `pid` is first marked as the current pid (for the
/// case where scheduling switched away earlier and we now want to terminate it); when
/// `false`, the caller is assumed to already be in the `pid` context.
pub(super) fn terminate_and_cleanup(
    os: &mut (dyn aios_kernel::kernel::Kernel + Send),
    pid: u64,
    result: String,
    set_current: bool,
) {
    os.cleanup_process_resources(pid);
    if let Ok(mut map) = SCHEDULER_DISPATCH_META.lock() {
        map.remove(&pid);
    }
    if set_current {
        os.set_current_pid(Some(pid));
    }
    os.terminate_current(result);
    if should_auto_drop_terminated(os, pid) {
        os.drop_terminated(pid);
    }
}

pub(super) fn format_rlimit_termination_result(verdict: RlimitVerdict) -> String {
    match verdict {
        RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        } => {
            let dim = match dimension {
                RlimitDim::Turns => "turns",
                RlimitDim::ToolCalls => "tool_calls",
                RlimitDim::TokensIn => "tokens_in",
                RlimitDim::TokensOut => "tokens_out",
                RlimitDim::CostMicros => "cost_micros",
                RlimitDim::WallclockTicks => "wallclock_ticks",
                RlimitDim::ToolCallBytes => "tool_call_bytes",
                RlimitDim::FsBytes => "fs_bytes",
            };
            format!("Terminated: Resource limit exceeded ({dim}: used={used}, limit={limit}).")
        }
        _ => "Completed".to_string(),
    }
}

/// Default max LLM iterations allowed per turn (prevents infinite loops).
/// 4096 was too high: with no mid-turn governance between "stop only on byte-exact
/// repetition" and "run to the cap", a single turn could pile up hundreds of thousands
/// of characters of context. The mid-turn circuit breaker (the orchestrator's iteration
/// soft limit) already handles timely convergence, so this hard cap just needs to clamp
/// to a more reasonable magnitude.
const DEFAULT_MAX_ITERATIONS: usize = 64 * 64;

/// Max iterations for subagent (executor) processes
const EXECUTOR_MAX_ITERATIONS: usize = 64 * 64;

fn one_shot_cli_mode(cli: &cli::ParsedCli) -> bool {
    // `a -ns` with no real content falls into interactive mode automatically (same as
    // `-ns -i`), so it is not one-shot; otherwise it would trigger one-shot cleanup
    // semantics (delete session on exit, no resume after failure, etc.).
    if note_search::note_search_interactive_mode(cli) {
        return false;
    }
    !cli.args.is_empty() && !cli.interactive
}

/// Ctrl+C only sets a flag in the signal handler; the driver event loop then safely
/// persists the current session.
fn should_suspend_session_on_sigint(app: &App) -> bool {
    if app.cli.session.is_some() {
        return true;
    }

    let store = SessionStore::new(app.config.history_file.as_path());
    !store.is_empty_session(&app.session_id).unwrap_or(false)
}

fn suspend_session_on_sigint(app: &App) {
    // A session that was just created but has no user messages yet will be deleted
    // during exit cleanup, so it must not leave a suspended entry pointing at it.
    if !should_suspend_session_on_sigint(app) {
        return;
    }

    if let Err(err) = SuspendedSessionStore::new().suspend_current_terminal(
        &app.session_id,
        app.config.history_file.as_path(),
        &app.active_persona.id,
        &app.current_model,
    ) {
        eprintln!("[suspend] Ctrl+C 退出时保存当前模型失败：{err}");
    }
}

fn decision_log_persist_enabled() -> bool {
    configw::get_all_config()
        .get_opt(AiConfig::DECISION_LOG_PERSIST_ENABLE)
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("true")
}

/// Main entry point for AIOS.
/// Initializes all components and starts the run_loop.
///
/// Initialization steps:
///   1. Parse CLI arguments
///   2. Load config
///   3. Create session store and session ID
///   4. Setup signal handlers (Ctrl+C)
///   5. Initialize HTTP client
///   6. Create local kernel (process OS)
///   7. Load skills and MCP clients
///   8. Load and activate agents
///   9. Enter run_loop
pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli::parse_cli_args(std::env::args());
    run_with_cli(cli).await
}

/// Run AIOS with already-parsed CLI arguments.
/// Reused by entry points such as background mode that need to modify the cli first
/// (injecting a session id / persistence directives).
pub(in crate::ai) async fn run_with_cli(
    cli: cli::ParsedCli,
) -> Result<(), Box<dyn std::error::Error>> {
    aios_kernel::kernel::register_current_pid_provider(current_task_pid);

    // The cli has already been parsed by the caller (run() or the background entry),
    // so use it directly here.

    // Purely local commands (help, list tools/skills/agents) do not call the LLM and
    // must be handled before ensure_models_available / load_config: otherwise, when the
    // model registry (models/) is empty or the config is broken, even `a --help` fails
    // to run, creating a dead loop where you must configure the environment to see help.
    if cli.help {
        cli::print_help();
        return Ok(());
    }

    // --version: print the version and exit (purely local, no LLM). Must be
    // handled here, before startup proceeds: otherwise the flag is treated as a
    // prompt and a full model session starts (slow, and pointless for a version
    // query). Mirrors the `re` tool's `--version` handling.
    if cli.version {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    // --generate-completions: generate the shell completion script (purely local, no LLM)
    if cli.generate_completions {
        let shell = cli.args.first().cloned().unwrap_or_else(|| {
            std::env::var("SHELL")
                .unwrap_or_default()
                .rsplit('/')
                .next()
                .unwrap_or("bash")
                .to_string()
        });
        cli::generate_completion_script(&shell);
        return Ok(());
    }

    if cli.list_tools {
        let tool_summaries =
            super::tools::tool_summaries_for_groups(&[super::tools::ToolGroup::Core]);
        print::print_builtin_tool_summaries(&tool_summaries);
        return Ok(());
    }

    if cli.list_skills {
        let skill_manifests = load_skill_manifests(cli.no_skills);
        print::print_skills(&skill_manifests);
        return Ok(());
    }

    if cli.list_agents {
        let agent_manifests = agents::load_all_agents();
        commands::help::print_agents_list(&agent_manifests);
        return Ok(());
    }

    if let Err(err) = models::ensure_models_available() {
        return Err(err.into());
    }
    let mut config = config::load_config()?;
    let persona_store = crate::ai::persona::PersonaStore::new();
    let active_persona = match persona_store.active_persona() {
        Ok(persona) => persona,
        Err(err) => {
            eprintln!("[persona] failed to load personas: {}", err);
            crate::ai::persona::default_persona()
        }
    };
    let startup_choice =
        resolve_startup_session_choice(&cli, &config, &persona_store, active_persona)?;
    let active_persona = startup_choice.active_persona;
    config.history_file = startup_choice.history_file.clone();
    let session_store = SessionStore::new(config.history_file.as_path());
    let session_id = startup_choice.session_id.clone();
    let startup_notice = startup_choice.startup_notice.clone();

    // Handle --clear --session <id>: clear the history and checkpoint of the given
    // session before startup.
    if cli.clear {
        let target = cli.session.as_deref().map(str::trim).unwrap_or("");
        if target.is_empty() {
            eprintln!("[clear] --clear 需要配合 --session <id> 使用");
        } else {
            match session_store.clear_session_history(target) {
                Ok(()) => println!("[clear] session {} 的历史和 checkpoint 已清空", target),
                Err(err) => eprintln!("[clear] 清空 session {} 失败: {}", target, err),
            }
        }
        return Ok(());
    }

    if let Err(err) = session_store.ensure_root_dir() {
        eprintln!("[Warning] Failed to create sessions dir: {}", err);
    }
    // Register the current process's PID in the sessions directory so the `/proc`
    // command can discover active sessions. This must happen before any session
    // restore/read to prevent prune from deleting an open session during the startup
    // window. The guard removes the PID file automatically when the function exits
    // (normal return or panic); even if SIGKILLed, `/proc` cleans up leftovers via PID
    // liveness probing.
    let _session_pid_guard =
        session_pid::SessionPidGuard::register(session_store.sessions_root(), &session_id);

    // A crash can occur between checkpoint rollback publishing the live SQLite and the
    // assets; finish transaction recovery first so later turns never read
    // cross-version state.
    session_store.recover_checkpoint_state(&session_id)?;

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

    // Prefer the model saved by the suspended session (if any), otherwise the default
    // model from CLI/config.
    let current_model = if let Some(ref model) = startup_choice.model
        && !model.is_empty()
    {
        model.clone()
    } else {
        models::initial_model(&cli)
    };
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;
    // One-shot modes (e.g. `-n/-nd/-ne`) carry positional args, but later flows may
    // still fall back to interactive multiline input/editing; for example `-ne` opens a
    // pre-filled editor after matching an entry. So do not bind the prompt editor to
    // "no positional args", otherwise "no input space after matching an entry, judged
    // as cancelled" can occur.
    let prompt_editor = Some(PromptEditor::new(
        &session_id,
        config.history_file.as_path(),
    ));

    let os_arc = new_local_kernel();
    crate::ai::tools::os_tools::init_os_tools_globals(os_arc.clone());

    // Build the tool-permission middleware from config. Empty rules => no
    // middleware installed (zero-config path stays all-allow). This install site
    // is reached by every `run_with_cli` entry, including a re-exec'd background
    // daemon (`-bg`), so the background process DOES enforce the policy. Only
    // in-process subagents skip it: `App::clone` resets the middleware vectors,
    // so agent-spawned helpers inherit no gate. Note a background daemon's stdin
    // is `/dev/null`, so any `ask` rule fails closed to deny there.
    let tool_middlewares = {
        let cfg = configw::get_all_config();
        let rules = cfg
            .get_opt(crate::ai::config_schema::AiConfig::TOOLS_PERMISSIONS)
            .unwrap_or_default();
        let default = cfg
            .get_opt(crate::ai::config_schema::AiConfig::TOOLS_PERMISSIONS_DEFAULT)
            .unwrap_or_default();
        match crate::ai::tools::permissions::ToolPermissions::from_config(&rules, &default) {
            Some((perms, warnings)) => {
                for warning in warnings {
                    eprintln!("[tool-permissions] {warning}");
                }
                let mw: std::sync::Arc<dyn crate::ai::middleware::ToolMiddleware> =
                    std::sync::Arc::new(crate::ai::tools::permissions::PermissionMiddleware::new(
                        perms,
                    ));
                vec![mw]
            }
            None => Vec::new(),
        }
    };

    let mut app = App {
        pending_files: if cli.files.trim().is_empty() {
            None
        } else {
            Some(cli.files.clone())
        },
        forced_skills: Vec::new(),
        forced_skill_source: None,
        pending_skill_continuation: None,
        forced_question: None,
        current_model,
        current_agent: "build".to_string(),
        current_agent_manifest: None,
        session_id: session_id.clone(),
        session_history_file: session_store.session_history_file(&session_id),
        active_persona,
        cli,
        config,
        client,
        attached_image_files: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        ignore_next_prompt_interrupt: false,
        prompt_editor,
        agent_context: Some(AgentContext {
            tools: Vec::new(),
            mcp_servers: rust_tools::cw::SkipMap::default(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }),
        last_skill_bias: None,
        os: os_arc,
        agent_reload_counter: None,
        // The ThinkingOrchestrator's JSON control protocol is not yet decoupled from
        // the main answer, and it does not consume the state JSON it asks the model to
        // produce. Production requests do not register it, so that optional observers
        // never hijack user answers into STRICT JSON; re-attach it explicitly once it
        // becomes an independent control call.
        observers: Vec::new(),
        last_known_prompt_tokens: None,
        last_known_cached_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
        turn_reasoning_items: Default::default(),
        stale_patch_targets: Default::default(),
        tool_middlewares,
        llm_middlewares: Vec::new(),
        hooks: Default::default(),
    };
    commands::session::restore_session_local_runtime_state(&mut app)?;
    if let Some(notice) = startup_notice {
        println!("{notice}");
    }
    // Handle --note-delete / -nd: the model matches a knowledge-base entry from the
    // input text and deletes it after confirmation.
    if let Some(query) = app.cli.note_delete.clone() {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                note_search::handle_note_delete(&mut app, &query),
            )
            .await;
    }

    // Handle --note-edit / -ne: the model matches a knowledge-base entry from the input
    // text, rewrites it in the editor, and saves.
    if let Some(query) = app.cli.note_edit.clone() {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                note_search::handle_note_edit(&mut app, &query),
            )
            .await;
    }

    // Handle --note / -n: quickly save a memo to the knowledge base and exit.
    // Even with no text (just a clipboard image to save), entering the save flow when
    // -n is passed.
    if app.cli.note_flag {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                note_search::handle_note_save(&mut app),
            )
            .await;
    }

    // Handle --note-search / -ns: with a query, run a single-turn notebook search and
    // exit directly; with no real content (e.g. `a -ns`) or with `-i`, enter
    // interactive mode where run_loop keeps answering notebook searches each turn.
    if app.cli.note_search && !note_search::note_search_interactive_mode(&app.cli) {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                note_search::handle_memo_search(&app),
            )
            .await;
    }
    if app.cli.consolidate_knowledge {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                note_search::handle_consolidate_knowledge(&app),
            )
            .await;
    }

    if decision_log_persist_enabled() {
        let decision_log_path = app
            .session_history_file
            .with_extension("decision-log.jsonl");
        crate::ai::driver::decision_log::set_decision_log_persist_path(decision_log_path);
    } else {
        crate::ai::driver::decision_log::clear_decision_log_persist_path();
    }

    let mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));

    let mcp_probe = probe_mcp_config(&app);
    if app.cli.list_mcp_tools {
        let mcp_report = init_mcp(
            &mut app,
            &mut mcp_client.lock().unwrap_or_else(|err| err.into_inner()),
        )
        .await;
        print::print_mcp_tools(
            &mcp_report,
            &mcp_client.lock().unwrap_or_else(|err| err.into_inner()),
        );
        return Ok(());
    }

    if let Some(ctx) = app.agent_context.as_mut() {
        ctx.tools = super::tools::tool_definitions_for_groups(&[super::tools::ToolGroup::Core]);
    }

    // Hold manifests behind an Arc: every foreground turn / background subagent
    // dispatch hands DriverContext a snapshot. Previously Arc::new(x.to_vec()) /
    // Arc::new(x.clone()) deep-copied every agent+skill prompt body. With an Arc these
    // snapshots become cheap pointer clones; reload just replaces the Arc wholesale.
    let mut skill_manifests: Arc<Vec<SkillManifest>> = Arc::new(Vec::new());
    let mut agent_manifests: Arc<Vec<AgentManifest>> = Arc::new(Vec::new());

    if let Err(err) = persona_store.remember_session(&app.active_persona.id, &app.session_id) {
        eprintln!("[persona] failed to persist session binding: {}", err);
    }

    // Old sessions may lack a generative title; backfill it in the background right on
    // restore so it is not deferred until a new turn completes, without letting the
    // title model request block the input UI from starting.
    turn_runtime::maybe_generate_session_title(&app, true).await;

    run_loop(
        &mut app,
        &mcp_client,
        mcp_probe,
        &mut skill_manifests,
        &mut agent_manifests,
    )
    .await
}

/// Resume a foreground ready process: build the wake-up prompt, run one turn of
/// run_turn, then follow the quota / termination / failure teardown flow per result.
async fn run_foreground_resume(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &Arc<Vec<SkillManifest>>,
    agent_manifests: &Arc<Vec<AgentManifest>>,
    proc: aios_kernel::kernel::Process,
) {
    let pid = proc.pid;
    let proc_question = if !proc.mailbox.is_empty() {
        let messages: Vec<String> = proc.mailbox.iter().cloned().collect();
        {
            let mut os = app.os.lock().unwrap();
            if let Some(actual) = os.get_process_mut(pid) {
                actual.mailbox.clear();
            }
        }
        format_wakeup_prompt(pid, &proc.goal, &messages)
    } else {
        format!(
            "[Process {} Resumed] Goal: {}\nContinue execution.",
            pid, proc.goal
        )
    };

    {
        let mut os = app.os.lock().unwrap();
        os.set_current_pid(Some(pid));
        let _ = os.process_pending_signals();
    }

    let next_model = app.current_model.clone();
    crate::ai::types::clear_stream_cancel(app);
    crate::ai::tools::registry::common::clear_tool_cancel();

    let driver_ctx = runtime_ctx::DriverContext::from_app_snapshot(
        app,
        mcp_client.clone(),
        skill_manifests.clone(),
        agent_manifests.clone(),
    );
    let persona_memory_path = app.current_persona_memory_file();

    let turn_outcome = runtime_ctx::DRIVER_CTX
        .scope(
            driver_ctx,
            runtime_ctx::PERSONA_MEMORY_PATH.scope(
                persona_memory_path,
                TASK_PID.scope(
                    Some(pid),
                    runtime_ctx::IS_RESUME_TURN.scope(
                        true,
                        turn_runtime::run_turn(
                            app,
                            mcp_client,
                            skill_manifests,
                            usize::MAX,
                            proc_question,
                            String::new(),
                            next_model,
                            None,
                            false,
                            false,
                        ),
                    ),
                ),
            ),
        )
        .await;

    match turn_outcome {
        Ok(_outcome) => {
            let mut os = app.os.lock().unwrap();
            os.set_current_pid(Some(pid));
            let outcome = classify_process_outcome(&**os, pid);
            record_scheduler_outcome(os.as_mut(), pid, outcome);
            let (should_terminate, termination_result) = finalize_turn_quota(os.as_mut(), pid);
            if should_terminate {
                terminate_and_cleanup(os.as_mut(), pid, termination_result, true);
            }
        }
        Err(err) => {
            let mut os = app.os.lock().unwrap();
            record_scheduler_outcome(os.as_mut(), pid, DispatchOutcomeTag::Failed);
            terminate_and_cleanup(os.as_mut(), pid, format!("Failed: {}", err), true);
        }
    }
}

/// Main event loop for AIOS.
/// Coordinates execution of both foreground and background processes.
///
/// Loop structure per iteration:
///   1. Scheduler tick: advance_tick() to wake sleeping processes
///   2. Agent hot-reload: check for new agents every 5 ticks
///   3. Shutdown check: exit if shutdown flag is set
///   4. Background execution:
///      - spawn async tasks for each
///      - wait for all to complete
///   5. Foreground input:
///      - get next question from input::next_question()
///      - handle interactive commands
///      - run turn via turn_runtime::run_turn()
///   6. Termination check: exit if quit requested
///
/// one_shot_mode: When CLI args provided and `--interactive` is not set
///   - runs once and exits
///   - deletes session after completion
async fn run_loop(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    mcp_probe: McpConfigProbe,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let one_shot_mode = one_shot_cli_mode(&app.cli);
    let mut should_quit = one_shot_mode;
    let mut mcp_initialized = false;
    let mut manifests_loaded = false;
    let mut skill_watcher = None;
    let mut skill_watcher_started = false;
    // External skill directories (especially Trae's recursive ones) can be large. In
    // interactive mode scan in the background only after the input box's first frame
    // is drawn; the full snapshot is still taken over before the user submits their
    // first input.
    let mut initial_skill_manifests = if !one_shot_mode && !app.cli.no_skills {
        match skill_watcher::spawn_initial_skill_manifest_load() {
            Ok(loader) => Some(loader),
            Err(err) => {
                eprintln!("[Warning] 技能预加载线程未启动，将在首条输入后同步加载：{err}");
                None
            }
        }
    } else {
        None
    };
    let mut mcp_preload_task = if should_preload_mcp(one_shot_mode, &mcp_probe) {
        Some(spawn_mcp_preload_task(mcp_probe.config_path.clone()))
    } else {
        None
    };
    let mut subagent_status_line = SubagentStatusLine::new();

    let cleanup_one_shot = |app: &App| {
        // Session ending: clean up background process groups left by this session (e.g.
        // resident services spawned via `python app.py &`). Every exit path goes
        // through this closure, so this is the unified safety net.
        let _ = crate::ai::tools::storage::process_registry::kill_session(&app.session_id);
        // One-shot mode (and not restoring a given session): always delete the session.
        // Interactive mode: if no existing session was restored and the current session
        // has no user messages (user hit Ctrl+C directly without ever typing anything),
        // also delete the empty session.
        if one_shot_mode && app.cli.session.is_none() {
            let store = SessionStore::new(app.config.history_file.as_path());
            let _ = store.delete_session(&app.session_id);
            return;
        }
        if app.cli.session.is_none() {
            let store = SessionStore::new(app.config.history_file.as_path());
            if store.is_empty_session(&app.session_id).unwrap_or(false) {
                let _ = store.delete_session(&app.session_id);
            }
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

    let mut scheduler_clock = SchedulerClock::default();
    loop {
        let epoch = next_scheduler_epoch();

        // Proactively reap subagent processes stuck past their wall-clock lifetime.
        // The same-named check inside task_wait only fires when the main agent calls it
        // explicitly; scanning after scheduler events here ensures stuck background
        // subagents are still terminated promptly even when the main agent moves on and
        // never calls task_wait, avoiding permanently occupied scheduler resources. The
        // function takes locks in two steps (registry first, then kernel), so it does
        // not form a cycle with task_wait's lock order (registry -> kernel); and the
        // app.os lock is already released here, so there is no re-entrant deadlock.
        crate::ai::tools::task_tools::reap_timed_out_subagents();
        // `task_wait.timeout_secs` is a real wall-clock budget that does not depend on
        // the scheduler tick rate. On expiry, actively wake the waiting foreground
        // process so the next task_wait returns BUDGET ELAPSED instead of waiting
        // forever because of tick drift.
        crate::ai::tools::task_tools::wake_expired_task_waits();

        if let Some(counter) = app.agent_reload_counter.as_mut() {
            *counter += 1;
            if manifests_loaded && *counter % 5 == 0 {
                if let Some(message) = reload_agent_manifests(agent_manifests) {
                    subagent_status_line.finish();
                    println!("{message}");
                }
            }
        } else {
            app.agent_reload_counter = Some(0);
        }

        if app.shutdown.load(Ordering::Relaxed) {
            if signal::take_sigint_shutdown_request() {
                suspend_session_on_sigint(app);
            }
            cleanup_one_shot(app);
            return Ok(());
        }

        if should_preload_mcp(one_shot_mode, &mcp_probe)
            && !mcp_initialized
            && mcp_preload_task.is_none()
            && !signal::request_interrupt_ready()
        {
            mcp_preload_task = Some(spawn_mcp_preload_task(mcp_probe.config_path.clone()));
        }

        let history_count;
        let mut question;
        let attachments_text;

        dispatch_background_batch(
            app,
            mcp_client,
            skill_manifests,
            agent_manifests,
            &mut manifests_loaded,
            epoch,
        );
        subagent_status_line.refresh(app);

        let fg_proc = {
            let mut os = app.os.lock().unwrap();
            os.pop_foreground_ready()
        };
        if let Some(proc) = fg_proc {
            subagent_status_line.finish();
            run_foreground_resume(app, mcp_client, skill_manifests, agent_manifests, proc).await;
            continue;
        }

        if has_pending_foreground_process(app) {
            scheduler_clock.wait(app).await;
            continue;
        }

        // While background subagents are still in flight, do not open the interactive
        // input box (it enters raw mode, so subagent streamed `\n` output lacks `\r` and
        // shifts right line by line). Wait for the background status notification; only
        // accept new input once subagents have finished writing in cooked mode and the
        // count is back to zero. one-shot mode has no interactive input box, so it is
        // unaffected.
        if !one_shot_mode && bg_subagents_inflight() {
            scheduler_clock.wait(app).await;
            continue;
        }

        subagent_status_line.finish();

        {
            // ── Goal mode auto-continuation ──
            // When a goal is set and the last turn called tools, skip user input and
            // inject a continuation prompt so the agent keeps pushing the goal forward.
            let goal_continuation = app
                .goal_mode
                .as_ref()
                .filter(|g| !g.is_empty() && app.last_turn_had_tool_calls && !one_shot_mode)
                .map(|g| commands::goal::build_goal_continuation_prompt(g));

            if let Some(cont) = goal_continuation {
                question = cont;
                attachments_text = String::new();
                history_count = 0;
            } else {
                // Goal active but no tool calls in the last turn:
                // - if interrupted by Ctrl+C (last_turn_interrupted), keep goal mode
                //   and silently fall back to waiting for user input, without falsely
                //   reporting "Goal achieved";
                // - otherwise treat the goal as reached, print the notice, and exit
                //   goal mode.
                let goal_active = app.goal_mode.as_ref().map_or(false, |g| !g.is_empty());
                if commands::goal::should_exit_goal_on_idle(
                    goal_active,
                    one_shot_mode,
                    app.last_turn_interrupted,
                ) {
                    use colored::Colorize;
                    println!(
                        "{} Goal achieved. Exiting goal mode.",
                        "[goal]".green().bold()
                    );
                    app.goal_mode = None;
                }

                if let Some(notifier) = initial_skill_manifests
                    .as_mut()
                    .and_then(|loader| loader.take_prompt_ready_notifier())
                {
                    // `-i <prompt>` consumes the CLI argument directly without entering
                    // the input box, so the preload thread must be released immediately,
                    // otherwise it waits forever for a notification when taking over
                    // the manifests later.
                    if !app.cli.args.is_empty() {
                        let _ = notifier.send(());
                    } else if let Some(editor) = app.prompt_editor.as_mut() {
                        editor.set_first_render_notifier(notifier);
                    } else {
                        let _ = notifier.send(());
                    }
                }

                let Some(ctx) = input::next_question(app)? else {
                    cleanup_one_shot(app);
                    return Ok(());
                };
                if ctx.question.trim().is_empty() {
                    should_quit = false;
                    continue;
                }
                question = ctx.question;
                attachments_text = ctx.attachments_text;
                history_count = ctx.history_count;
            }
        }

        // ── Local-command fast path ──
        // Dispatch commands that do not depend on skill/agent manifests first
        // (/usage, /help, /model, ...). When matched and no forced_question is
        // injected (no need to continue the LLM flow), skip the expensive manifest
        // scan — bringing read-only commands like `a /usage` from ~1s down to ~0.1s.
        // Commands like /goal that inject a forced_question and continue the
        // conversation still need the manifests loaded.
        if try_handle_local_command(app, mcp_client, &question)? {
            if let Some(rest) = app.forced_question.take() {
                question = rest;
            } else {
                if handle_post_command(app, &mut should_quit) {
                    return Ok(());
                }
                continue;
            }
        }

        // one-shot mode takes the CLI input directly; interactive mode adopts the
        // manifests scanned right after the first screen before the user's first
        // input. If the preload thread exits abnormally, fall back to the original
        // synchronous path so this turn's skill and agent semantics stay complete.
        if !manifests_loaded {
            if let Some(loaded_skill_manifests) = initial_skill_manifests
                .take()
                .and_then(|loader| loader.recv().ok())
            {
                install_runtime_manifests(
                    app,
                    skill_manifests,
                    agent_manifests,
                    &mut manifests_loaded,
                    loaded_skill_manifests,
                );
            } else {
                ensure_runtime_manifests_loaded(
                    app,
                    skill_manifests,
                    agent_manifests,
                    &mut manifests_loaded,
                );
            }
        }

        // The watcher thread starts only after the first snapshot is ready: completion
        // always reads from the in-memory snapshot, and only file changes trigger a
        // background rescan that replaces it.
        if !skill_watcher_started && !one_shot_mode {
            skill_watcher_started = true;
            match skill_watcher::start_skill_manifest_watcher(app.cli.no_skills) {
                Ok(watcher) => skill_watcher = watcher,
                Err(err) => eprintln!("[Warning] 技能热加载监听未启动：{err}"),
            }
        }

        // The watcher thread has already refreshed the completion cache; after input is
        // submitted, swap the driver's runtime snapshot so the following /skills
        // command and the next turn's routing also use the same fresh manifests.
        if let Some(watcher) = skill_watcher.as_mut()
            && let Some(updated) = watcher.take_latest()
        {
            *skill_manifests = updated;
        }

        if try_handle_interactive_command(
            app,
            mcp_client,
            &question,
            agent_manifests,
            skill_manifests,
        )? {
            // For /skills <name> <rest>, the parsed rest replaces the question and the
            // conversation continues.
            if let Some(rest) = app.forced_question.take() {
                question = rest;
            } else {
                if handle_post_command(app, &mut should_quit) {
                    return Ok(());
                }
                continue;
            }
        }

        // The /memo command needs to call the model asynchronously to organize content,
        // so it is handled separately here.
        if commands::memo::try_handle_memo_command(app, &question).await? {
            should_quit = false;
            continue;
        }

        if !one_shot_mode {
            // The title depends only on the input the user already submitted, so it
            // should not wait for the first model response. Dispatch the background
            // generation task here immediately, before MCP init, image preprocessing,
            // and the main request; once written to the store, the prompt update
            // channel notifies the frontend, and later request headers read the latest
            // title directly.
            turn_runtime::maybe_generate_session_title_for_input(app, &question).await;
        }

        // ── Goal mode wait state ──
        // After the user types `/goal`, the next non-slash message becomes the goal
        // content. It is wrapped into a goal prompt sent to the LLM while goal_mode is
        // updated.
        if app.goal_mode.as_ref().map_or(false, |g| g.is_empty()) {
            let goal_text = question.clone();
            app.goal_mode = Some(goal_text.clone());
            question = commands::goal::build_goal_prompt(&goal_text);
        }

        if note_search::note_search_interactive_mode(&app.cli) {
            match note_search::handle_note_search_interactive_turn(app, &question, history_count)
                .await
            {
                Ok(()) => {}
                Err(err) => {
                    eprintln!("[Error] 当前轮 notebook 检索失败：{}", err);
                    eprintln!("[Info] 会话保持运行，请继续输入下一条消息。\n");
                }
            }
            should_quit = false;
            continue;
        }

        if !one_shot_mode {
            try_finalize_mcp_preload(
                app,
                mcp_client,
                &mcp_probe,
                &mut mcp_initialized,
                &mut mcp_preload_task,
            )
            .await;
        }

        ensure_mcp_initialized_for_turn(
            app,
            mcp_client,
            &mcp_probe,
            &mut mcp_initialized,
            &mut mcp_preload_task,
        )
        .await;

        let precomputed_ocr = if !app.attached_image_files.is_empty()
            && !crate::ai::models::is_vl_model(&app.current_model)
        {
            // When a non-VL model receives attached images, the main agent no longer
            // calls the OCR tool directly; instead a subagent pinned to a VL model
            // parses the images, then control returns to the main agent.
            let image_parse_ctx = runtime_ctx::DriverContext::from_app_snapshot(
                app,
                mcp_client.clone(),
                skill_manifests.clone(),
                agent_manifests.clone(),
            );
            runtime_ctx::DRIVER_CTX
                .scope(
                    image_parse_ctx,
                    crate::ai::driver::model::parse_attached_images_via_subagent(
                        mcp_client,
                        &app.attached_image_files,
                    ),
                )
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        let has_usable_ocr_for_images = precomputed_ocr
            .as_ref()
            .map(|ocr| ocr.has_usable_text())
            .unwrap_or(false);
        let next_model = resolve_model_for_input(app, has_usable_ocr_for_images, &mut question);
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

        crate::ai::types::clear_stream_cancel(app);
        crate::ai::tools::registry::common::clear_tool_cancel();

        {
            let mut os = app.os.lock().unwrap();
            if os.process_pending_signals() {
                app.session_history_file = original_history_file;
                continue;
            }
        }

        let fg_pid = {
            let os = app.os.lock().unwrap();
            os.current_process_id()
        };

        let driver_ctx = runtime_ctx::DriverContext::from_app_snapshot(
            app,
            mcp_client.clone(),
            skill_manifests.clone(),
            agent_manifests.clone(),
        );

        hooks::run_lifecycle_hook(hooks::HookEvent::TurnStart, None, None);
        let persona_memory_path = app.current_persona_memory_file();

        let turn_outcome = runtime_ctx::DRIVER_CTX
            .scope(
                driver_ctx,
                runtime_ctx::PERSONA_MEMORY_PATH.scope(
                    persona_memory_path,
                    TASK_PID.scope(
                        fg_pid,
                        turn_runtime::run_turn(
                            app,
                            mcp_client,
                            &*skill_manifests,
                            history_count,
                            question,
                            attachments_text,
                            next_model,
                            precomputed_ocr,
                            one_shot_mode,
                            should_quit,
                        ),
                    ),
                ),
            )
            .await;

        hooks::run_lifecycle_hook(hooks::HookEvent::TurnEnd, None, None);

        match turn_outcome {
            Ok(outcome) => {
                let mut os = app.os.lock().unwrap();
                let current_pid = os.current_process_id();
                let (should_terminate, termination_result) = if let Some(pid) = current_pid {
                    let outcome_tag = classify_process_outcome(&**os, pid);
                    record_scheduler_outcome(os.as_mut(), pid, outcome_tag);
                    finalize_turn_quota(os.as_mut(), pid)
                } else {
                    (true, "Completed".to_string())
                };

                if should_terminate {
                    if let Some(pid) = current_pid {
                        terminate_and_cleanup(os.as_mut(), pid, termination_result, false);
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
                    record_scheduler_outcome(os.as_mut(), pid, DispatchOutcomeTag::Failed);
                    terminate_and_cleanup(os.as_mut(), pid, format!("Failed: {}", err), false);
                } else {
                    os.terminate_current(format!("Failed: {}", err));
                }
                app.session_history_file = original_history_file;
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
        // Cooperative yields such as task_wait / tool_wait make this turn's run_turn
        // return `Continue`, with the foreground process parked in Waiting until a
        // background subagent writes back results and wakes it. In one-shot mode
        // `should_quit` is always true; exiting here would kill the process the moment
        // the subagent has not yet been scheduled (the subagent stays in Ready
        // forever). So: as long as this turn yielded (Continue) and an unterminated
        // foreground process is still waiting, keep looping so the scheduler dispatches
        // the subagent, collects results, and wakes the foreground to resume, exiting
        // only after the foreground truly produces its final answer.
        let parked_awaiting_subagents =
            matches!(turn_outcome, Ok(turn_runtime::TurnOutcome::Continue))
                && has_pending_foreground_process(app);
        if (matches!(turn_outcome, Ok(turn_runtime::TurnOutcome::Quit)) || should_quit)
            && !parked_awaiting_subagents
        {
            if !one_shot_mode {
                for obs in app.observers.iter_mut() {
                    if obs.is_poisoned() {
                        continue;
                    }
                    let obs_name = obs.name().to_string();
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        obs.on_conversation_end();
                    }))
                    .is_err()
                    {
                        eprintln!(
                            "[Warning] observer '{}' panicked in on_conversation_end; disabling.",
                            obs_name
                        );
                        obs.mark_poisoned();
                    }
                }
            }
            hooks::run_lifecycle_hook(hooks::HookEvent::SessionEnd, None, None);
            cleanup_one_shot(app);
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests;
