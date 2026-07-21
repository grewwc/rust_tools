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
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use aios_kernel::primitives::{RlimitDim, RlimitVerdict};

use crate::ai::{
    agents::{self, AgentManifest},
    cli::{self},
    config,
    config_schema::AiConfig,
    history::SessionStore,
    mcp::{McpClient, SharedMcpClient},
    models,
    prompt::PromptEditor,
    skills::SkillManifest,
    types::{AgentContext, App},
};
use crate::commonw::configw;

pub mod agent_router;
mod agent_routing;
mod background_dispatch;
pub mod commands;
pub mod decision_log;
pub mod embedding;
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
pub mod signal;
pub mod skill_match_model;
pub mod skill_ranking;
pub mod skill_runtime;
pub mod text_similarity;
pub mod thinking;
pub mod tools;
pub mod turn_runtime;

use agent_routing::*;
use background_dispatch::dispatch_background_batch;
pub use commands::try_handle_interactive_command;
pub use mcp_init::*;
use mcp_lifecycle::*;
pub use model::*;
use process_context::*;
use scheduler::*;
use session::*;
pub use text_similarity::*;

tokio::task_local! {
    pub(super) static TASK_PID: Option<u64>;
}

fn current_task_pid() -> Option<u64> {
    TASK_PID.try_with(|v| *v).unwrap_or(None)
}

/// 当前已派发、尚未结束的后台子 agent tokio 任务数量。
///
/// 后台子 agent 通过 `tokio::spawn` 跑在 worker 线程上，会用 `println!`（裸 `\n`）
/// 流式写终端。而交互式输入框（multiline TUI）会开启 raw mode，关闭 TTY 的 ONLCR，
/// 此时裸 `\n` 不再补 `\r`，子 agent 的输出就会逐行右移（阶梯式错位）。
///
/// 用这个计数器在"打开输入框前"判断是否仍有后台子 agent 在跑：只要 > 0 就不进入
/// raw mode 输入框，让调度循环继续 tick、子 agent 在 cooked 模式下正常输出，避免
/// 并发写终端造成的显示混乱（同时不丢失任何子 agent 输出）。
static BG_SUBAGENT_INFLIGHT: AtomicUsize = AtomicUsize::new(0);

fn bg_subagents_inflight() -> bool {
    BG_SUBAGENT_INFLIGHT.load(Ordering::Acquire) > 0
}

/// RAII 守卫：派发后台子 agent 前 `inc`，子 agent 任务结束（含 panic）时自动 `dec`，
/// 保证计数不泄漏。
pub(super) struct BgSubagentGuard;

impl BgSubagentGuard {
    fn new() -> Self {
        BG_SUBAGENT_INFLIGHT.fetch_add(1, Ordering::AcqRel);
        BgSubagentGuard
    }
}

impl Drop for BgSubagentGuard {
    fn drop(&mut self) {
        BG_SUBAGENT_INFLIGHT.fetch_sub(1, Ordering::AcqRel);
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

/// 进程终止 + 清理 + 自动 drop 的统一收尾流程。
///
/// `set_current` 为 `true` 时，会先把 `pid` 标记为当前 pid（适用于先前调度切换走、
/// 现在要终止它的场景）；为 `false` 时假设调用方已经在 `pid` 的上下文里。
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
/// 4096 过高：在「字节完全重复才停」与「跑满上限」之间缺乏中段治理，单轮可
/// 堆出数十万字符上下文。中段断路器（orchestrator 的 iteration soft limit）
/// 已负责及时收敛，这里作为硬上限收敛到更合理的量级即可。
const DEFAULT_MAX_ITERATIONS: usize = 64 * 16;

/// Max iterations for subagent (executor) processes
const EXECUTOR_MAX_ITERATIONS: usize = 64 * 16;

fn one_shot_cli_mode(cli: &cli::ParsedCli) -> bool {
    !cli.args.is_empty() && !cli.interactive
}

fn decision_log_persist_enabled() -> bool {
    configw::get_all_config()
        .get_opt(AiConfig::DECISION_LOG_PERSIST_ENABLE)
        .unwrap_or_else(|| "false".to_string())
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

/// 用已解析好的 CLI 参数运行 AIOS。
/// 供 background 模式等需要预先修改 cli（注入 session id / 持久化指令）的入口复用。
pub(in crate::ai) async fn run_with_cli(
    cli: cli::ParsedCli,
) -> Result<(), Box<dyn std::error::Error>> {
    aios_kernel::kernel::register_current_pid_provider(current_task_pid);

    // cli 已由调用方解析完毕（run() 或 background 入口），此处直接使用。

    // 纯本地命令（帮助、列工具/技能/agent）不调用 LLM，必须在 ensure_models_available /
    // load_config 之前处理：否则 models.json 为空或配置损坏时，连 `a --help` 都跑不起来，
    // 形成“想看帮助先得把环境配好”的死循环。
    if cli.help {
        cli::print_help();
        return Ok(());
    }

    // --generate-completions: 生成 shell 补全脚本（纯本地，不调 LLM）
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
        let tool_summaries = super::tools::tool_summaries_for_groups(&["core"]);
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

    // 处理 --clear --session <id>：启动前清空指定 session 的 history 与 checkpoint。
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
    // 崩溃可能发生在 checkpoint rollback 发布 live SQLite 与 assets 之间；先完成
    // 事务恢复，避免后续 turn 读取到跨版本的状态。
    session_store.recover_checkpoint_state(&session_id)?;

    // 注册当前进程的 PID 到 sessions 目录，供 `/proc` 命令发现活跃 session。
    // guard 在函数退出（正常返回 / panic）时自动删除 PID 文件；
    // 即使被 SIGKILL 杀死，`/proc` 也会通过 PID 存活探测清理残留。
    let _session_pid_guard =
        session_pid::SessionPidGuard::register(session_store.sessions_root(), &session_id);

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

    // 优先使用挂起 session 保存的模型（如果有），否则使用 CLI/配置的默认模型
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
    // one-shot 模式（如 `-n/-nd/-ne`）虽然携带位置参数，但后续流程仍可能回退到
    // 交互式多行输入/编辑；例如 `-ne` 在命中条目后需要打开预填编辑器。
    // 因此不要把 prompt editor 绑定到“无位置参数”这一条件，否则会出现
    // “命中条目后没有输入空间，直接被判定为取消”的问题。
    let prompt_editor = Some(PromptEditor::new(
        &session_id,
        config.history_file.as_path(),
    ));

    let os_arc = new_local_kernel();
    crate::ai::tools::os_tools::init_os_tools_globals(os_arc.clone());

    let mut app = App {
        pending_files: if cli.files.trim().is_empty() {
            None
        } else {
            Some(cli.files.clone())
        },
        forced_skill: None,
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
    };
    if let Some(notice) = startup_notice {
        println!("{notice}");
    }
    // 处理 --note-delete / -nd：输入一段话，模型自动匹配知识库条目，确认后删除。
    if let Some(query) = app.cli.note_delete.clone() {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                note_search::handle_note_delete(&mut app, &query),
            )
            .await;
    }

    // 处理 --note-edit / -ne：输入一段话，模型匹配知识库条目，在编辑器中改写后保存。
    if let Some(query) = app.cli.note_edit.clone() {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                note_search::handle_note_edit(&mut app, &query),
            )
            .await;
    }

    // 处理 --note / -n：快速保存 memo 到知识库并退出。
    // 即使没有文本（只想保存剪贴板图片），只要传了 -n 也要进入保存流程。
    if app.cli.note_flag {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                note_search::handle_note_save(&mut app),
            )
            .await;
    }

    // 处理 --note-search / -ns：默认单轮 notebook 检索后直接退出；若带 `-i`
    // 则进入交互模式，由 run_loop 在每轮输入时继续执行 notebook 检索问答。
    if app.cli.note_search && !app.cli.interactive {
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
        ctx.tools = super::tools::tool_definitions_for_groups(&["core"]);
    }

    // 用 Arc 持有 manifests：每个 foreground turn / 后台子 agent 派发都要给
    // DriverContext 一份快照，过去用 Arc::new(x.to_vec()) / Arc::new(x.clone())
    // 会把全部 agent+skill 的 prompt 正文深拷贝一遍。改成 Arc 后这些快照退化
    // 成廉价的指针 clone；reload 时整体替换 Arc 即可。
    let mut skill_manifests: Arc<Vec<SkillManifest>> = Arc::new(Vec::new());
    let mut agent_manifests: Arc<Vec<AgentManifest>> = Arc::new(Vec::new());

    if let Err(err) = persona_store.remember_session(&app.active_persona.id, &app.session_id) {
        eprintln!("[persona] failed to persist session binding: {}", err);
    }

    // 旧 session 可能还没有生成式标题；恢复时立即在后台补齐，避免必须再完成一个
    // 新 turn 才触发，同时不让标题模型请求阻塞输入界面启动。
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

/// 处理一个 foreground ready 进程的恢复执行：构造 wake-up prompt、跑一轮 run_turn、
/// 然后根据结果走 quota / 终止 / 失败收尾流程。
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

    let driver_ctx = runtime_ctx::DriverContext::new(
        app.clone(),
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
    let mut mcp_loading_announced = false;
    let mut manifests_loaded = false;
    let mut mcp_preload_task = if should_preload_mcp(one_shot_mode, &mcp_probe) {
        Some(spawn_mcp_preload_task(mcp_probe.config_path.clone()))
    } else {
        None
    };

    let cleanup_one_shot = |app: &App| {
        // 会话结束：清理本会话遗留的后台进程组（如 `python app.py &` 派生的
        // 常驻服务）。在所有退出路径都会经过本闭包，故此处统一兜底。
        let _ = crate::ai::tools::storage::process_registry::kill_session(&app.session_id);
        // one-shot 模式（且非恢复指定 session）：总是删除 session。
        // 交互模式：如果未恢复已有 session 且当前 session 无任何用户消息
        // （用户直接 Ctrl+C 退出，从未输入有效内容），也删除空 session。
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

    loop {
        let epoch = next_scheduler_epoch();
        {
            let mut os = app.os.lock().unwrap_or_else(|err| err.into_inner());
            os.advance_tick();
        }

        // 主动回收超过 wall-clock 总寿命的卡死 subagent 进程。task_wait 内的同名检查
        // 只在主 agent 主动调用时触发；此处每 epoch 扫描，确保主 agent 去做别的事、
        // 长期不调 task_wait 时，卡死的后台 subagent 也能被及时终止，避免永久占用
        // 调度器资源。函数内分两步取锁（先 registry 后 kernel），不与 task_wait 的
        // 锁顺序（registry -> kernel）形成环；且此处已释放 app.os 锁，无重入死锁。
        crate::ai::tools::task_tools::reap_timed_out_subagents();

        if let Some(counter) = app.agent_reload_counter.as_mut() {
            *counter += 1;
            if manifests_loaded && *counter % 5 == 0 {
                reload_agent_manifests(agent_manifests);
            }
        } else {
            app.agent_reload_counter = Some(0);
        }

        if app.shutdown.load(Ordering::Relaxed) {
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

        let fg_proc = {
            let mut os = app.os.lock().unwrap();
            os.pop_foreground_ready()
        };
        if let Some(proc) = fg_proc {
            run_foreground_resume(app, mcp_client, skill_manifests, agent_manifests, proc).await;
            continue;
        }

        if has_pending_foreground_process(app) {
            tokio::time::sleep(Duration::from_millis(10)).await;
            continue;
        }

        // 仍有后台子 agent 在途时，不打开交互式输入框（它会进入 raw mode，导致子 agent
        // 的流式输出 `\n` 缺 `\r` 而逐行右移）。继续 tick 调度循环，等子 agent 在 cooked
        // 模式下把输出写完、计数归零后再接收新输入。one-shot 模式没有交互输入框，不受影响。
        if !one_shot_mode && bg_subagents_inflight() {
            tokio::time::sleep(Duration::from_millis(20)).await;
            continue;
        }

        {
            // ── Goal 模式自动续推 ──
            // 当 goal 已设定且上一轮调用了工具时，跳过用户输入，直接注入
            // continuation prompt 让 agent 继续推进目标。
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
                // goal 激活但上一轮无工具调用：
                // - 若是被 Ctrl+C 打断（last_turn_interrupted），保留 goal 模式，
                //   静默回落到等待用户输入，不误报「Goal achieved」；
                // - 否则视为目标已达成，打印提示并退出 goal 模式。
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

        if !one_shot_mode {
            announce_mcp_loading_if_needed(&mcp_probe, mcp_initialized, &mut mcp_loading_announced);
        }

        ensure_runtime_manifests_loaded(
            app,
            skill_manifests,
            agent_manifests,
            &mut manifests_loaded,
        );

        if try_handle_interactive_command(
            app,
            mcp_client,
            &question,
            agent_manifests,
            skill_manifests,
        )? {
            // /skills <name> <rest> 时，解析出的 rest 替换 question 继续问答
            if let Some(rest) = app.forced_question.take() {
                question = rest;
            } else {
                if handle_post_command(app, &mut should_quit) {
                    return Ok(());
                }
                continue;
            }
        }

        // ── Goal 模式等待状态 ──
        // 用户输入 `/goal` 后，下一条非 slash 消息作为目标内容。
        // 将目标包装成 goal prompt 发送给 LLM，同时更新 goal_mode。
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
        maybe_auto_route_agent(app, &*agent_manifests, &question);

        if !one_shot_mode {
            announce_mcp_loading_if_needed(&mcp_probe, mcp_initialized, &mut mcp_loading_announced);

            try_finalize_mcp_preload(
                app,
                mcp_client,
                &mcp_probe,
                &mut mcp_initialized,
                &mut mcp_loading_announced,
                &mut mcp_preload_task,
            )
            .await;
        }

        ensure_mcp_initialized_for_turn(
            app,
            mcp_client,
            &mcp_probe,
            &mut mcp_initialized,
            &mut mcp_loading_announced,
            &mut mcp_preload_task,
            !one_shot_mode,
        )
        .await;

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

        let driver_ctx = runtime_ctx::DriverContext::new(
            app.clone(),
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
        // task_wait / tool_wait 等协作式让出会让本轮 run_turn 以 `Continue` 返回，
        // 而前台进程此时停在 Waiting（park），等后台子 agent 写回结果再被唤醒。
        // one-shot 模式下 `should_quit` 恒为 true，若此处直接退出，就会在子 agent
        // 还没被调度的瞬间结束进程（子 agent 永远停在 Ready）。因此：只要本轮是
        // 让出（Continue）且仍有未终止的前台进程在等待，就继续 loop，让调度器派发
        // 子 agent、收集结果并唤醒前台续跑，直到前台真正产出最终回答后再退出。
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
