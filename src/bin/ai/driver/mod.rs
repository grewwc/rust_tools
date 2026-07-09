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
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use aios_kernel::primitives::{ResourceUsageDelta, RlimitDim, RlimitVerdict};
use rust_tools::cw::SkipMap;
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::ai::{
    agents::{self, AgentManifest},
    cli::{self},
    config,
    config_schema::AiConfig,
    history::{
        SessionStore, SuspendedSessionEntry, SuspendedSessionStore,
        format_suspended_timestamp_label,
    },
    mcp::{McpClient, SharedMcpClient},
    models,
    prompt::PromptEditor,
    skills::{self, SkillManifest},
    tools::task_tools::{decode_os_task_goal, is_encoded_task_goal, with_task_entry_by_pid},
    types::{AgentContext, App},
};
use crate::commonw::configw;

pub mod agent_router;
pub mod commands;
pub mod decision_log;
pub mod embedding;
pub mod hooks;
pub mod input;
pub mod mcp_init;
pub mod model;
pub mod note_search;
pub mod observer;
pub mod print;
pub mod reflection;
pub mod runtime_ctx;
pub mod session_pid;
pub mod signal;
pub mod skill_match_model;
pub mod skill_ranking;
pub mod skill_runtime;
pub mod text_similarity;
pub mod thinking;
pub mod tools;
pub mod turn_runtime;

pub use commands::try_handle_interactive_command;
pub use mcp_init::*;
pub use model::*;
pub use skill_ranking::*;
pub use text_similarity::*;

tokio::task_local! {
    static TASK_PID: Option<u64>;
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
struct BgSubagentGuard;

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

#[derive(Debug, Clone)]
struct StartupSessionChoice {
    active_persona: crate::ai::persona::PersonaProfile,
    history_file: PathBuf,
    session_id: String,
    model: Option<String>,
    startup_notice: Option<String>,
}

#[derive(Debug, Clone)]
struct SuspendedSessionPreview {
    entry: SuspendedSessionEntry,
    persona_label: String,
    summary: Option<String>,
    modified_label: Option<String>,
    suspended_label: String,
}

fn build_suspended_session_previews(
    entries: Vec<SuspendedSessionEntry>,
    persona_store: &crate::ai::persona::PersonaStore,
) -> Vec<SuspendedSessionPreview> {
    let personas = persona_store.list_personas().unwrap_or_default();
    let mut sessions_by_history: FxHashMap<PathBuf, Vec<_>> = FxHashMap::default();

    entries
        .into_iter()
        .map(|entry| {
            let persona_label = personas
                .iter()
                .find(|persona| persona.id == entry.persona_id)
                .map(|persona| persona.name.clone())
                .unwrap_or_else(|| entry.persona_id.clone());
            let session_info = sessions_by_history
                .entry(entry.history_file.clone())
                .or_insert_with(|| {
                    SessionStore::new(entry.history_file.as_path())
                        .list_sessions()
                        .unwrap_or_default()
                })
                .iter()
                .find(|session| session.id == entry.session_id);
            SuspendedSessionPreview {
                persona_label,
                summary: session_info.and_then(|session| session.summary.clone()),
                modified_label: session_info.and_then(|session| {
                    session
                        .modified_local
                        .as_ref()
                        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                }),
                suspended_label: format_suspended_timestamp_label(&entry.suspended_at),
                entry,
            }
        })
        .collect()
}

fn prompt_select_suspended_session(
    previews: &[SuspendedSessionPreview],
) -> std::io::Result<Option<usize>> {
    if previews.is_empty() {
        return Ok(None);
    }
    if previews.len() == 1 || !std::io::stdin().is_terminal() {
        return Ok(Some(0));
    }

    println!(
        "[resume] 当前 terminal 有 {} 个挂起 session：",
        previews.len()
    );
    for (index, preview) in previews.iter().enumerate() {
        println!(
            "  {}. {}  persona={}  modified={}  suspended={}",
            index + 1,
            preview.entry.session_id,
            preview.persona_label,
            preview.modified_label.as_deref().unwrap_or("-"),
            preview.suspended_label
        );
        if let Some(summary) = preview
            .summary
            .as_deref()
            .filter(|summary| !summary.is_empty())
        {
            println!("     {summary}");
        }
    }

    loop {
        let input = crate::commonw::prompt::read_line(&format!(
            "选择要恢复的 session [1-{}，回车=1，n=新 session]: ",
            previews.len()
        ));
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(Some(0));
        }
        let lower = trimmed.to_ascii_lowercase();
        if lower == "n" || lower == "new" {
            return Ok(None);
        }
        if let Ok(index) = trimmed.parse::<usize>()
            && (1..=previews.len()).contains(&index)
        {
            return Ok(Some(index - 1));
        }
        eprintln!(
            "[resume] 无效选择：请输入 1-{}，或输入 n 新建 session。",
            previews.len()
        );
    }
}

fn build_resume_startup_notice(
    session_id: &str,
    remaining_suspended: usize,
    persona_fallback: bool,
) -> String {
    let mut notice = format!("[resume] 已恢复挂起 session: {session_id}");
    if persona_fallback {
        notice.push_str("（原 persona 不存在，已按当前 persona 打开）");
    }
    if remaining_suspended > 0 {
        notice.push_str(&format!(
            "；当前 terminal 还有 {} 个挂起 session，运行 `a --resume` 可继续选择。",
            remaining_suspended
        ));
    }
    notice.push_str("；运行 `a --new-session` 可强制新建 session。");
    notice
}

fn should_resume_suspended_terminal_session(cli: &cli::ParsedCli) -> bool {
    if cli.new_session {
        return false;
    }
    if cli.resume {
        return true;
    }
    if cli.session.is_some() || cli.clear || !cli.args.is_empty() {
        return false;
    }
    if cli.help
        || cli.list_tools
        || cli.list_mcp_tools
        || cli.list_skills
        || cli.list_agents
        || cli.note_search
        || cli.note_flag
        || cli.note_delete.is_some()
        || cli.note_edit.is_some()
        || cli.consolidate_knowledge
        || cli.generate_completions
    {
        return false;
    }
    true
}

fn resolve_startup_session_choice_with_selector<F>(
    cli: &cli::ParsedCli,
    config: &crate::ai::types::AppConfig,
    persona_store: &crate::ai::persona::PersonaStore,
    active_persona: crate::ai::persona::PersonaProfile,
    selector: F,
) -> Result<StartupSessionChoice, Box<dyn std::error::Error>>
where
    F: FnMut(&[SuspendedSessionPreview]) -> std::io::Result<Option<usize>>,
{
    resolve_startup_session_choice_with_selector_inner(
        cli,
        config,
        persona_store,
        active_persona,
        selector,
    )
}

fn resolve_startup_session_choice_with_selector_inner<F>(
    cli: &cli::ParsedCli,
    config: &crate::ai::types::AppConfig,
    persona_store: &crate::ai::persona::PersonaStore,
    active_persona: crate::ai::persona::PersonaProfile,
    mut selector: F,
) -> Result<StartupSessionChoice, Box<dyn std::error::Error>>
where
    F: FnMut(&[SuspendedSessionPreview]) -> std::io::Result<Option<usize>>,
{
    if cli.resume && cli.session.is_some() {
        return Err("`--resume` 不能和 `--session` 同时使用".into());
    }
    if cli.resume && cli.clear {
        return Err("`--resume` 不能和 `--clear` 同时使用".into());
    }
    if cli.resume && cli.new_session {
        return Err("`--resume` 不能和 `--new-session` 同时使用".into());
    }
    if cli.new_session && cli.session.is_some() {
        return Err("`--new-session` 不能和 `--session` 同时使用".into());
    }
    if cli.new_session && cli.clear {
        return Err("`--new-session` 不能和 `--clear` 同时使用".into());
    }

    let mut choice = StartupSessionChoice {
        history_file: crate::ai::persona::history_file_for_persona(
            config.base_history_file.as_path(),
            &active_persona.id,
        ),
        active_persona,
        session_id: cli
            .session
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(|id| id.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string()),
        model: None,
        startup_notice: None,
    };

    if !should_resume_suspended_terminal_session(cli) {
        return Ok(choice);
    }

    let suspended_store = SuspendedSessionStore::new();
    match suspended_store.list_current_terminal() {
        Ok(entries) if entries.is_empty() => {
            if cli.resume {
                choice.startup_notice = Some(
                    "[resume] 当前 terminal 没有可恢复的挂起 session，已创建新 session。"
                        .to_string(),
                );
            }
        }
        Ok(entries) => {
            let previews = build_suspended_session_previews(entries, persona_store);
            let selected_index = if previews.len() == 1 && !cli.resume {
                Some(0)
            } else {
                selector(&previews)?
            };
            let Some(selected_index) = selected_index else {
                choice.startup_notice = Some(format!(
                    "[resume] 已跳过当前 terminal 的 {} 个挂起 session，已创建新 session。运行 `a --resume` 可再次选择恢复。",
                    previews.len()
                ));
                return Ok(choice);
            };
            let selected = previews
                .get(selected_index)
                .ok_or_else(|| format!("invalid suspended session selection: {selected_index}"))?;
            let Some(entry) = suspended_store.take_selected_current_terminal(&selected.entry)?
            else {
                if cli.resume {
                    return Err("选中的挂起 session 已不存在，请重试".into());
                }
                choice.startup_notice =
                    Some("[resume] 选中的挂起 session 已不存在，已创建新 session。".to_string());
                return Ok(choice);
            };
            choice.history_file = entry.history_file.clone();
            choice.session_id = entry.session_id.clone();
            // 恢复挂起时保存的模型，而非使用默认模型
            choice.model = entry.model.clone();

            let remaining = previews.len().saturating_sub(1);
            let mut persona_fallback = false;
            match persona_store.list_personas() {
                Ok(personas) => {
                    if let Some(persona) = personas.into_iter().find(|p| p.id == entry.persona_id) {
                        choice.active_persona = persona;
                    } else {
                        persona_fallback = true;
                    }
                }
                Err(err) => {
                    eprintln!("[resume] failed to load personas: {}", err);
                }
            }
            choice.startup_notice = Some(build_resume_startup_notice(
                &choice.session_id,
                remaining,
                persona_fallback,
            ));
        }
        Err(err) => {
            if cli.resume {
                return Err(err.into());
            }
            if err.kind() != std::io::ErrorKind::Unsupported {
                eprintln!("[resume] 自动恢复已跳过：{}", err);
            }
        }
    }

    Ok(choice)
}

fn resolve_startup_session_choice(
    cli: &cli::ParsedCli,
    config: &crate::ai::types::AppConfig,
    persona_store: &crate::ai::persona::PersonaStore,
    active_persona: crate::ai::persona::PersonaProfile,
) -> Result<StartupSessionChoice, Box<dyn std::error::Error>> {
    resolve_startup_session_choice_with_selector_inner(
        cli,
        config,
        persona_store,
        active_persona,
        prompt_select_suspended_session,
    )
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
fn terminate_and_cleanup(
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

fn format_rlimit_termination_result(verdict: RlimitVerdict) -> String {
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
const DEFAULT_MAX_ITERATIONS: usize = 2048;

/// Max iterations for subagent (executor) processes
const EXECUTOR_MAX_ITERATIONS: usize = 2048;

const BG_DISPATCH_BASE_BATCH_DEFAULT: usize = 4;
const BG_DISPATCH_MAX_BATCH_DEFAULT: usize = 8;
const BG_DISPATCH_EXECUTE_MAX_DEFAULT: usize = 6;
const SCHED_FAIL_STREAK_OPEN_THRESHOLD_DEFAULT: u32 = 3;
const SCHED_COOLDOWN_EPOCHS_DEFAULT: u64 = 6;
const SCHED_EVAL_PERIOD_EPOCHS_DEFAULT: u64 = 24;
const SCHED_EVAL_MIN_SAMPLES_DEFAULT: usize = 8;
const SCHED_COST_PENALTY_DIVISOR_DEFAULT: u64 = 50_000;
const SCHED_TOKEN_PENALTY_DIVISOR_DEFAULT: u64 = 4_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchOutcomeTag {
    Advanced,
    Blocked,
    Failed,
}

#[derive(Debug, Clone, Copy)]
struct ProcessDispatchMeta {
    failure_streak: u32,
    success_streak: u32,
    cooldown_until_epoch: u64,
    last_dispatch_epoch: u64,
    last_outcome: DispatchOutcomeTag,
}

impl Default for ProcessDispatchMeta {
    fn default() -> Self {
        Self {
            failure_streak: 0,
            success_streak: 0,
            cooldown_until_epoch: 0,
            last_dispatch_epoch: 0,
            last_outcome: DispatchOutcomeTag::Advanced,
        }
    }
}

static SCHEDULER_DISPATCH_META: LazyLock<Mutex<SkipMap<u64, ProcessDispatchMeta>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));
static SCHEDULER_EPOCH: AtomicU64 = AtomicU64::new(0);
static SCHEDULER_LAST_EVAL_EPOCH: AtomicU64 = AtomicU64::new(0);

fn scheduler_cfg_usize(key: &str, default: usize) -> usize {
    if cfg!(test) {
        return default;
    }
    configw::get_all_config()
        .get_opt(key)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn scheduler_cfg_u32(key: &str, default: u32) -> u32 {
    if cfg!(test) {
        return default;
    }
    configw::get_all_config()
        .get_opt(key)
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

fn scheduler_cfg_u64(key: &str, default: u64) -> u64 {
    if cfg!(test) {
        return default;
    }
    configw::get_all_config()
        .get_opt(key)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn sched_base_batch() -> usize {
    scheduler_cfg_usize(
        AiConfig::SCHEDULER_BASE_BATCH,
        BG_DISPATCH_BASE_BATCH_DEFAULT,
    )
    .max(1)
}

fn sched_max_batch() -> usize {
    scheduler_cfg_usize(AiConfig::SCHEDULER_MAX_BATCH, BG_DISPATCH_MAX_BATCH_DEFAULT)
        .max(sched_base_batch())
}

fn sched_execute_max() -> usize {
    scheduler_cfg_usize(
        AiConfig::SCHEDULER_EXECUTE_MAX,
        BG_DISPATCH_EXECUTE_MAX_DEFAULT,
    )
    .max(sched_base_batch())
}

fn sched_fail_threshold() -> u32 {
    scheduler_cfg_u32(
        AiConfig::SCHEDULER_FAIL_STREAK_THRESHOLD,
        SCHED_FAIL_STREAK_OPEN_THRESHOLD_DEFAULT,
    )
    .max(1)
}

fn sched_cooldown_epochs() -> u64 {
    scheduler_cfg_u64(
        AiConfig::SCHEDULER_COOLDOWN_EPOCHS,
        SCHED_COOLDOWN_EPOCHS_DEFAULT,
    )
    .max(1)
}

fn sched_eval_period_epochs() -> u64 {
    scheduler_cfg_u64(
        AiConfig::SCHEDULER_EVAL_PERIOD_EPOCHS,
        SCHED_EVAL_PERIOD_EPOCHS_DEFAULT,
    )
    .max(1)
}

fn sched_eval_min_samples() -> usize {
    scheduler_cfg_usize(
        AiConfig::SCHEDULER_EVAL_MIN_SAMPLES,
        SCHED_EVAL_MIN_SAMPLES_DEFAULT,
    )
    .max(1)
}

fn sched_cost_penalty_divisor() -> u64 {
    scheduler_cfg_u64(
        AiConfig::SCHEDULER_COST_PENALTY_DIVISOR_MICROS,
        SCHED_COST_PENALTY_DIVISOR_DEFAULT,
    )
    .max(1)
}

fn sched_token_penalty_divisor() -> u64 {
    scheduler_cfg_u64(
        AiConfig::SCHEDULER_TOKEN_PENALTY_DIVISOR,
        SCHED_TOKEN_PENALTY_DIVISOR_DEFAULT,
    )
    .max(1)
}

fn next_scheduler_epoch() -> u64 {
    SCHEDULER_EPOCH
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1)
}

fn current_scheduler_epoch() -> u64 {
    SCHEDULER_EPOCH.load(Ordering::Relaxed)
}

fn background_pop_limit(ready_count: usize) -> usize {
    let base = sched_base_batch();
    let max_batch = sched_max_batch();
    if ready_count <= base {
        return base;
    }
    if ready_count <= 8 {
        return (base + 2).min(max_batch);
    }
    max_batch
}

fn background_execute_limit(ready_count: usize) -> usize {
    let base = sched_base_batch();
    let execute_max = sched_execute_max();
    if ready_count <= base {
        return base;
    }
    if ready_count <= 10 {
        return (base + 1).min(execute_max);
    }
    execute_max
}

fn scheduler_score(
    proc: &aios_kernel::kernel::Process,
    meta: ProcessDispatchMeta,
    epoch: u64,
) -> i64 {
    // 更高优先级（更小 priority 值）+ 更久未被调度 + 更低失败 streak。
    let priority_score = (255u16.saturating_sub(proc.priority as u16) as i64) * 4;
    let quota_score = (proc.quota_turns.min(32) as i64) * 2;
    let age = epoch.saturating_sub(meta.last_dispatch_epoch).min(64) as i64;
    let age_score = age * 3;
    let failure_penalty = (meta.failure_streak as i64) * 40;
    let cost_penalty = (proc.usage.cost_micros / sched_cost_penalty_divisor()) as i64;
    let token_penalty =
        ((proc.usage.tokens_in + proc.usage.tokens_out) / sched_token_penalty_divisor()) as i64;
    let outcome_bias = match meta.last_outcome {
        DispatchOutcomeTag::Advanced => 12,
        DispatchOutcomeTag::Blocked => 6,
        DispatchOutcomeTag::Failed => -20,
    };
    priority_score + quota_score + age_score + outcome_bias
        - failure_penalty
        - cost_penalty
        - token_penalty
}

fn update_dispatch_meta(
    mut meta: ProcessDispatchMeta,
    outcome: DispatchOutcomeTag,
    epoch: u64,
) -> ProcessDispatchMeta {
    meta.last_outcome = outcome;
    match outcome {
        DispatchOutcomeTag::Advanced => {
            meta.failure_streak = 0;
            meta.success_streak = meta.success_streak.saturating_add(1).min(100);
        }
        DispatchOutcomeTag::Blocked => {
            meta.failure_streak = meta.failure_streak.saturating_sub(1);
            meta.success_streak = 0;
        }
        DispatchOutcomeTag::Failed => {
            meta.success_streak = 0;
            meta.failure_streak = meta.failure_streak.saturating_add(1).min(100);
            if meta.failure_streak >= sched_fail_threshold() {
                meta.cooldown_until_epoch = epoch.saturating_add(sched_cooldown_epochs());
            }
        }
    }
    meta
}

fn maybe_promote_half_open(meta: ProcessDispatchMeta, epoch: u64) -> ProcessDispatchMeta {
    if meta.cooldown_until_epoch > 0 && epoch >= meta.cooldown_until_epoch {
        ProcessDispatchMeta {
            cooldown_until_epoch: 0,
            failure_streak: meta.failure_streak.saturating_sub(1),
            ..meta
        }
    } else {
        meta
    }
}

fn classify_process_outcome(os: &dyn aios_kernel::kernel::Kernel, pid: u64) -> DispatchOutcomeTag {
    let Some(proc) = os.get_process(pid) else {
        return DispatchOutcomeTag::Advanced;
    };
    if matches!(
        proc.state,
        aios_kernel::kernel::ProcessState::Waiting { .. }
            | aios_kernel::kernel::ProcessState::Sleeping { .. }
            | aios_kernel::kernel::ProcessState::Stopped
    ) {
        DispatchOutcomeTag::Blocked
    } else {
        DispatchOutcomeTag::Advanced
    }
}

fn should_publish_subagent_task_result(
    result_ok: bool,
    captured_output: &str,
    proc_state: Option<&aios_kernel::kernel::ProcessState>,
) -> bool {
    if !result_ok || !captured_output.trim().is_empty() {
        return true;
    }
    !matches!(
        proc_state,
        Some(
            aios_kernel::kernel::ProcessState::Waiting { .. }
                | aios_kernel::kernel::ProcessState::Sleeping { .. }
                | aios_kernel::kernel::ProcessState::Stopped
        )
    )
}

fn decode_background_process_task_goal(
    goal: &str,
) -> Result<Option<crate::ai::tools::task_tools::OsTaskGoal>, String> {
    if !is_encoded_task_goal(goal) {
        return Ok(None);
    }
    decode_os_task_goal(goal).map(Some).ok_or_else(|| {
        "failed to decode encoded subagent task metadata; refusing to fall back to the parent agent/model"
            .to_string()
    })
}

fn resolve_background_subagent_override<'a>(
    agent_manifests: &'a [AgentManifest],
    agent_override: Option<&str>,
) -> Result<Option<&'a AgentManifest>, String> {
    let Some(agent_name) = agent_override.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let Some(agent) = agents::find_agent_by_name(agent_manifests, agent_name) else {
        return Err(format!(
            "Selected subagent '{}' could not be found in runtime manifests.",
            agent_name
        ));
    };
    if agent.disabled {
        return Err(format!("Selected subagent '{}' is disabled.", agent_name));
    }
    Ok(Some(agent))
}

fn publish_background_task_failure(
    os: &mut (dyn aios_kernel::kernel::Kernel + Send),
    pid: u64,
    result_channel_id: Option<u64>,
    completion_futex_addr: Option<aios_kernel::primitives::FutexAddr>,
    error: &str,
) {
    os.set_current_pid(Some(pid));
    if let Some(result_channel_id) = result_channel_id {
        let payload = serde_json::json!({
            "status": "failed",
            "output": "",
            "error": error,
        })
        .to_string();
        let _ = os.channel_send(
            Some(pid),
            aios_kernel::primitives::ChannelId(result_channel_id),
            payload,
        );
        let _ = os.channel_close(Some(pid), aios_kernel::primitives::ChannelId(result_channel_id));
        let _ = os.channel_release_named(
            aios_kernel::primitives::ChannelId(result_channel_id),
            "task_result.producer",
        );
    }
    if let Some(addr) = completion_futex_addr {
        let _ = os.futex_store(addr, 1);
    }
    record_scheduler_outcome(os, pid, DispatchOutcomeTag::Failed);
    terminate_and_cleanup(os, pid, format!("Failed: {}", error), true);
}

fn apply_priority_handoff(proc: &mut aios_kernel::kernel::Process, outcome: DispatchOutcomeTag) {
    match outcome {
        DispatchOutcomeTag::Advanced => {
            proc.priority = proc.priority.saturating_sub(1);
        }
        DispatchOutcomeTag::Blocked => {}
        DispatchOutcomeTag::Failed => {
            proc.priority = proc.priority.saturating_add(8);
        }
    }
}

fn record_scheduler_outcome(
    os: &mut dyn aios_kernel::kernel::Kernel,
    pid: u64,
    outcome: DispatchOutcomeTag,
) {
    let epoch = current_scheduler_epoch();
    let mut meta_map = SCHEDULER_DISPATCH_META
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let prev = *meta_map
        .get_ref(&pid)
        .unwrap_or(&ProcessDispatchMeta::default());
    let next = update_dispatch_meta(prev, outcome, epoch);
    meta_map.insert(pid, next);

    if let Some(proc) = os.get_process_mut(pid) {
        apply_priority_handoff(proc, outcome);
        if outcome == DispatchOutcomeTag::Failed && next.failure_streak >= sched_fail_threshold() {
            proc.mailbox.push_back(format!(
                "[scheduler-circuit-open] Consecutive failures reached {}. Cooling down for {} dispatch epochs.",
                next.failure_streak,
                sched_cooldown_epochs()
            ));
        }
    }
}

fn mark_dispatched_pids(pids: &[u64], epoch: u64) {
    let mut meta_map = SCHEDULER_DISPATCH_META
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    for pid in pids {
        let mut meta = meta_map.get(pid).unwrap_or(ProcessDispatchMeta::default());
        meta.last_dispatch_epoch = epoch;
        meta_map.insert(*pid, meta);
    }
}

fn log_scheduler_decision(
    session_id: &str,
    epoch: u64,
    chosen: &str,
    alternatives: Vec<String>,
    reasoning: String,
    success: bool,
) {
    let store = crate::ai::driver::decision_log::get_decision_log_store();
    crate::ai::driver::decision_log::log_scheduler_dispatch(
        store,
        session_id,
        epoch as usize,
        &format!("scheduler_epoch={epoch}"),
        alternatives,
        chosen,
        &reasoning,
        success,
    );
}

fn maybe_emit_scheduler_eval(epoch: u64, session_id: &str) {
    let last = SCHEDULER_LAST_EVAL_EPOCH.load(Ordering::Relaxed);
    if epoch.saturating_sub(last) < sched_eval_period_epochs() {
        return;
    }
    let map = SCHEDULER_DISPATCH_META
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    if map.len() < sched_eval_min_samples() {
        return;
    }
    let mut failing = 0usize;
    let mut cooling = 0usize;
    let mut blocked = 0usize;
    for meta in map.values() {
        if meta.failure_streak > 0 {
            failing += 1;
        }
        if meta.cooldown_until_epoch > epoch {
            cooling += 1;
        }
        if matches!(meta.last_outcome, DispatchOutcomeTag::Blocked) {
            blocked += 1;
        }
    }
    let total = map.len();
    drop(map);
    let failing_rate = failing as f64 / total as f64;
    let cooling_rate = cooling as f64 / total as f64;
    let blocked_rate = blocked as f64 / total as f64;
    let healthy = failing_rate < 0.35 && cooling_rate < 0.20;

    crate::ai::driver::print::print_tool_note_line(
        "scheduler-eval",
        &format!(
            "epoch={epoch} total={total} failing={failing} cooling={cooling} blocked={blocked}"
        ),
    );
    log_scheduler_decision(
        session_id,
        epoch,
        if healthy {
            "scheduler_healthy"
        } else {
            "scheduler_attention"
        },
        vec![
            format!("failing_rate={:.3}", failing_rate),
            format!("cooling_rate={:.3}", cooling_rate),
            format!("blocked_rate={:.3}", blocked_rate),
        ],
        "periodic scheduler health evaluation".to_string(),
        healthy,
    );
    SCHEDULER_LAST_EVAL_EPOCH.store(epoch, Ordering::Relaxed);
}

#[cfg(test)]
fn reset_scheduler_test_state() {
    SCHEDULER_EPOCH.store(0, Ordering::Relaxed);
    if let Ok(mut map) = SCHEDULER_DISPATCH_META.lock() {
        map.clear();
    }
}

fn select_background_batch(
    os: &mut dyn aios_kernel::kernel::Kernel,
    epoch: u64,
    session_id: &str,
) -> Vec<aios_kernel::kernel::Process> {
    let ready_count = os.ready_count();
    let pop_limit = background_pop_limit(ready_count);
    let execute_limit = background_execute_limit(ready_count);
    let popped = os.pop_all_ready(pop_limit);
    if popped.is_empty() {
        return Vec::new();
    }

    let meta_snapshot = {
        let map = SCHEDULER_DISPATCH_META
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        map.clone()
    };

    let mut eligible = Vec::new();
    let mut deferred = Vec::new();
    let mut alt_notes = Vec::new();
    for proc in popped {
        let meta = maybe_promote_half_open(
            meta_snapshot
                .get(&proc.pid)
                .unwrap_or(ProcessDispatchMeta::default()),
            epoch,
        );
        let score = scheduler_score(&proc, meta, epoch);
        alt_notes.push(format!(
            "pid={} score={} cooldown_until={} fail_streak={}",
            proc.pid, score, meta.cooldown_until_epoch, meta.failure_streak
        ));
        if meta.cooldown_until_epoch > epoch {
            deferred.push(proc);
        } else {
            eligible.push(proc);
        }
    }

    eligible.sort_by(|a, b| {
        let a_meta = maybe_promote_half_open(
            meta_snapshot
                .get(&a.pid)
                .unwrap_or(ProcessDispatchMeta::default()),
            epoch,
        );
        let b_meta = maybe_promote_half_open(
            meta_snapshot
                .get(&b.pid)
                .unwrap_or(ProcessDispatchMeta::default()),
            epoch,
        );
        scheduler_score(b, b_meta, epoch).cmp(&scheduler_score(a, a_meta, epoch))
    });

    let keep = execute_limit.min(eligible.len());
    let mut selected = eligible;
    let mut spill = selected.split_off(keep);
    deferred.append(&mut spill);

    let deferred_count = deferred.len();
    // pop_all_ready 会把进程置 Running；对未执行者需要显式回到 ready。
    for proc in deferred {
        os.set_current_pid(Some(proc.pid));
        let _ = os.requeue_current();
    }

    if let Some(first) = selected.first() {
        os.set_current_pid(Some(first.pid));
    }
    let selected_pids: Vec<u64> = selected.iter().map(|p| p.pid).collect();
    mark_dispatched_pids(&selected_pids, epoch);
    if !selected_pids.is_empty() {
        log_scheduler_decision(
            session_id,
            epoch,
            &format!("selected={:?}", selected_pids),
            alt_notes,
            format!("eligible={} deferred={}", selected.len(), deferred_count),
            true,
        );
    }
    selected
}

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

fn has_pending_foreground_process(app: &App) -> bool {
    let os = app.os.lock().unwrap();
    os.list_processes().into_iter().any(|proc| {
        proc.is_foreground
            && !matches!(
                proc.state,
                aios_kernel::kernel::ProcessState::Terminated
                    | aios_kernel::kernel::ProcessState::Ready
            )
    })
}

/// Check if auto-agent routing is enabled in config.
/// Auto-routing selects the best agent based on the question content.
///
/// 默认禁用：TF-IDF + logistic regression 的浅层文本匹配误报率高（如问题中
/// 出现 "skill" 就切到 prompt-skill agent），且 agent 切换会改变 system
/// prompt / 工具集 / model tier，影响整轮对话。用户可通过 `-a <agent>`
/// 手动指定，或配置 `ai.agents.auto_route.enable = true` 重新启用。
fn auto_agent_routing_enabled() -> bool {
    !configw::get_all_config()
        .get_opt("ai.agents.auto_route.enable")
        .unwrap_or_else(|| "false".to_string())
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

    let history = note_search::read_recent_history(app);
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

/// Loads all agents fresh from disk, enabling hot-reload of newly added/modified agents.
/// Returns the updated manifests.
fn reload_agent_manifests(agent_manifests: &mut Arc<Vec<AgentManifest>>) {
    let new_agents = agents::load_all_agents();
    let old_fingerprint = agent_manifests_fingerprint(agent_manifests.as_slice());
    let new_fingerprint = agent_manifests_fingerprint(new_agents.as_slice());
    if old_fingerprint == new_fingerprint {
        return;
    }
    let added = new_agents.len() as i64 - agent_manifests.len() as i64;
    if added > 0 {
        println!("[Agent 发现] 新发现 {} 个 agent(s)，已自动加载", added);
    } else if added < 0 {
        println!(
            "[Agent 发现] 移除 {} 个 agent(s)，共 {} 个",
            -added,
            new_agents.len()
        );
    } else {
        println!(
            "[Agent 发现] 检测到 agent 内容变更，已重新加载，共 {} 个",
            new_agents.len()
        );
    }
    *agent_manifests = Arc::new(new_agents);
}

/// 基于 manifest 关键字段计算稳定指纹，用于检测增删改三类变更。
fn agent_manifests_fingerprint(agents: &[AgentManifest]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut entries: Vec<&AgentManifest> = agents.iter().collect();
    entries.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.source_path.cmp(&b.source_path))
    });
    let mut hasher = Sha256::new();
    for m in entries {
        hasher.update(m.name.as_bytes());
        hasher.update(b"\0");
        hasher.update(m.source_path.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(m.description.as_bytes());
        hasher.update(b"\0");
        hasher.update(m.prompt.as_bytes());
        hasher.update(b"\0");
        hasher.update(m.system_prompt.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(m.model.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", m.mode).as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", m.temperature).as_bytes());
        hasher.update(b"\0");
        hasher.update(format!("{:?}", m.max_steps).as_bytes());
        hasher.update(b"\0");
        for t in &m.tools {
            hasher.update(t.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        for g in &m.tool_groups {
            hasher.update(g.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        for s in &m.mcp_servers {
            hasher.update(s.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        hasher.update([m.disable_mcp_tools as u8]);
        hasher.update(b"\0");
        for tag in &m.routing_tags {
            hasher.update(tag.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\0");
        hasher.update([m.disabled as u8, m.hidden as u8]);
        hasher.update(b"\0");
        hasher.update(m.color.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"|");
    }
    hasher.finalize().into()
}

fn ensure_runtime_manifests_loaded(
    app: &mut App,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
    manifests_loaded: &mut bool,
) {
    if *manifests_loaded {
        return;
    }

    *skill_manifests = Arc::new(load_skill_manifests(app.cli.no_skills));
    *agent_manifests = Arc::new(agents::load_all_agents());

    if !skill_manifests.is_empty() {
        crate::ai::knowledge::indexing::embedder::warm_up();
    }

    if let Some(default_agent) = agents::find_agent_by_name(agent_manifests, &app.current_agent)
        && default_agent.is_primary()
        && !default_agent.disabled
    {
        activate_primary_agent(app, default_agent);
    }

    if let Some(agent_name) = &app.cli.agent {
        if let Some(agent) = agents::find_agent_by_name(agent_manifests, agent_name) {
            if agent.is_primary() && !agent.disabled {
                activate_primary_agent(app, agent);
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

    *manifests_loaded = true;
}

fn apply_prepared_mcp_with_shared_client(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    prepared: PreparedMcpInit,
) -> McpInitReport {
    let mut guard = mcp_client.lock().unwrap_or_else(|err| err.into_inner());
    apply_prepared_mcp_init(app, &mut guard, prepared)
}

fn announce_mcp_loading_if_needed(
    mcp_probe: &McpConfigProbe,
    mcp_initialized: bool,
    mcp_loading_announced: &mut bool,
) {
    if *mcp_loading_announced || mcp_initialized || !mcp_probe.exists {
        return;
    }
    print!(
        "{}",
        print::format_section_header(
            "mcp",
            Some(&format!(
                "{} configured servers · loading...",
                mcp_probe.server_count
            ))
        )
    );
    std::io::stdout().flush().ok();
    *mcp_loading_announced = true;
}

fn emit_mcp_loaded_header(report: &McpInitReport, mcp_loading_announced: &mut bool) {
    if !report.loaded {
        return;
    }

    let header = print::format_section_header(
        "mcp",
        Some(&format!(
            "{} servers, {} tools",
            report.server_count, report.tool_count
        )),
    );
    if *mcp_loading_announced {
        print!("\r\x1b[2K{}\n", header);
    } else {
        println!("{header}");
    }
}

async fn finalize_mcp_preload_task(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    mcp_probe: &McpConfigProbe,
    task: tokio::task::JoinHandle<Option<PreparedMcpInit>>,
) -> Option<McpInitReport> {
    match task.await {
        Ok(Some(prepared)) => Some(apply_prepared_mcp_with_shared_client(
            app, mcp_client, prepared,
        )),
        Ok(None) => None,
        Err(err) => {
            if app.shutdown.load(Ordering::Relaxed) || signal::request_interrupt_ready() {
                return None;
            }
            eprintln!("[mcp] background preload task failed: {}", err);
            let fallback =
                prepare_mcp_initialization_from_path(mcp_probe.config_path.clone()).await;
            Some(apply_prepared_mcp_with_shared_client(
                app, mcp_client, fallback,
            ))
        }
    }
}

async fn try_finalize_mcp_preload(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    mcp_probe: &McpConfigProbe,
    mcp_initialized: &mut bool,
    mcp_loading_announced: &mut bool,
    mcp_preload_task: &mut Option<tokio::task::JoinHandle<Option<PreparedMcpInit>>>,
) {
    if *mcp_initialized || !mcp_probe.exists {
        return;
    }

    let Some(task) = mcp_preload_task.as_ref() else {
        return;
    };
    if !task.is_finished() {
        return;
    }

    let task = mcp_preload_task.take().unwrap();
    let Some(report) = finalize_mcp_preload_task(app, mcp_client, mcp_probe, task).await else {
        return;
    };

    *mcp_initialized = true;
    emit_mcp_loaded_header(&report, mcp_loading_announced);
}

async fn ensure_mcp_initialized_for_turn(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    mcp_probe: &McpConfigProbe,
    mcp_initialized: &mut bool,
    mcp_loading_announced: &mut bool,
    mcp_preload_task: &mut Option<tokio::task::JoinHandle<Option<PreparedMcpInit>>>,
    show_status: bool,
) {
    if *mcp_initialized || !mcp_probe.exists {
        return;
    }

    if show_status {
        announce_mcp_loading_if_needed(mcp_probe, *mcp_initialized, mcp_loading_announced);
    }

    let report = if let Some(task) = mcp_preload_task.take() {
        finalize_mcp_preload_task(app, mcp_client, mcp_probe, task).await
    } else {
        let prepared = prepare_mcp_initialization_from_path(mcp_probe.config_path.clone()).await;
        Some(apply_prepared_mcp_with_shared_client(
            app, mcp_client, prepared,
        ))
    };

    let Some(report) = report else {
        return;
    };

    *mcp_initialized = true;
    if show_status {
        emit_mcp_loaded_header(&report, mcp_loading_announced);
    }
}

fn spawn_mcp_preload_task(config_path: String) -> tokio::task::JoinHandle<Option<PreparedMcpInit>> {
    tokio::spawn(async move {
        let interrupt_futex = signal::alloc_interrupt_futex("mcp_preload_interrupt");
        let prepared =
            prepare_mcp_initialization_from_path_interruptible(config_path, interrupt_futex).await;
        if let Some(addr) = interrupt_futex {
            signal::destroy_interrupt_futex(addr);
        }
        prepared
    })
}

fn should_preload_mcp(_one_shot_mode: bool, mcp_probe: &McpConfigProbe) -> bool {
    mcp_probe.exists
}

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

    // 处理 --clear --session <id>：启动前清空指定 session 的 history
    if cli.clear {
        let target = cli.session.as_deref().map(str::trim).unwrap_or("");
        if target.is_empty() {
            eprintln!("[clear] --clear 需要配合 --session <id> 使用");
        } else {
            match session_store.clear_session_history(target) {
                Ok(()) => println!("[clear] session {} 的历史已清空", target),
                Err(err) => eprintln!("[clear] 清空 session {} 失败: {}", target, err),
            }
        }
        return Ok(());
    }

    if let Err(err) = session_store.ensure_root_dir() {
        eprintln!("[Warning] Failed to create sessions dir: {}", err);
    }

    // 注册当前进程的 PID 到 sessions 目录，供 `/proc` 命令发现活跃 session。
    // guard 在函数退出（正常返回 / panic）时自动删除 PID 文件；
    // 即使被 SIGKILL 杀死，`/proc` 也会通过 PID 存活探测清理残留。
    let _session_pid_guard = session_pid::SessionPidGuard::register(
        session_store.sessions_root(),
        &session_id,
    );

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
        goal_mode: None,
        last_turn_had_tool_calls: false,
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
            .scope(app.current_persona_memory_file(), note_search::handle_memo_search(&app))
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

    run_loop(
        &mut app,
        &mcp_client,
        mcp_probe,
        &mut skill_manifests,
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

fn resolve_background_subagent_context(
    history_path: PathBuf,
    original_history_file: &Path,
    skill_manifests: &Arc<Vec<SkillManifest>>,
    task_id: Option<&str>,
    inherit: crate::ai::tools::task_tools::InheritOptions,
) -> (PathBuf, Arc<Vec<SkillManifest>>) {
    let is_task_subagent = task_id.is_some();
    let effective_history = if is_task_subagent && inherit.history {
        original_history_file.to_path_buf()
    } else {
        history_path
    };
    let effective_skills = if is_task_subagent && !inherit.skills {
        Arc::new(Vec::new())
    } else {
        skill_manifests.clone()
    };
    (effective_history, effective_skills)
}

fn build_background_process_question(
    pid: u64,
    raw_goal: &str,
    decoded_task_goal_prompt: Option<&str>,
    mailbox_messages: &[String],
) -> String {
    if !mailbox_messages.is_empty() {
        let original_goal = decoded_task_goal_prompt.unwrap_or(raw_goal);
        return format_wakeup_prompt(pid, original_goal, mailbox_messages);
    }
    if let Some(prompt) = decoded_task_goal_prompt {
        return prompt.to_string();
    }
    format!(
        "[Process {}] Goal: {}\nExecute this goal autonomously and provide the final result.",
        pid, raw_goal
    )
}

/// 构造进程被唤醒（mailbox 非空）时的 wake-up prompt。
/// foreground / background 路径共享同一段 prompt，避免双份硬编码漂移。
fn format_wakeup_prompt(pid: u64, goal: &str, messages: &[String]) -> String {
    format!(
        "[Process {} Woke Up] Original goal: {}\nNew mailbox messages:\n{}\n\nWake-up handling rules:\n- The async machinery has TWO families with similar but distinct semantics:\n    * subagent tasks  — `task_spawn` / `task_wait` / `task_status` (no cancel; long-lived task_id; task_wait's `timeout_secs` is a per-call wait budget, NOT a stall signal — re-call task_wait or pass wait_policy=\"any\" to keep waiting)\n    * generic tools   — `tool_spawn` / `tool_wait` / `tool_status` / `tool_cancel`\n  Pick the family that matches what you actually spawned; do NOT call task_cancel (it does not exist) or tool_wait on a task_id.\n- If the mailbox indicates event wake-up after `task_wait PARKED`, immediately re-call `task_wait` with the same task_ids and wait_policy to collect results from task result channels. Use `task_status` only if you need a non-blocking snapshot.\n- If the mailbox indicates generic async tool wake-up, use `tool_status` / `tool_wait` / `tool_cancel` as appropriate.\n- Do not abandon parked subagent tasks as stuck merely because the previous `task_wait` returned PARKED.\n- Prefer continuing reasoning immediately when the wake-up messages already identify the relevant finished tasks.\n\nResume execution based on the goal and these messages.",
        pid,
        goal,
        messages.join("\n---\n")
    )
}

/// 一轮 turn 执行成功后对目标进程的 quota 收尾逻辑。
/// 行为与改造前完全一致：扣减 `quota_turns` 一次，如果归零则准备 "Max LLM quota reached"
/// 终止理由；如果进程已经处于 Waiting/Sleeping/Stopped 则保持其挂起状态、不触发终止。
///
/// 返回 `(should_terminate, termination_result)`，由调用方决定是否真的调用
/// `terminate_and_cleanup`，从而保留 foreground / background 各自后续的特殊处理
/// （如 round-robin requeue 等）。
fn finalize_turn_quota(os: &mut dyn aios_kernel::kernel::Kernel, pid: u64) -> (bool, String) {
    let verdict = os.rusage_charge(
        pid,
        ResourceUsageDelta {
            turns: 1,
            ..Default::default()
        },
    );
    let mut should_terminate = true;
    let mut termination_result = format_rlimit_termination_result(verdict.clone());
    if let Some(p) = os.get_process_mut(pid)
        && matches!(
            p.state,
            aios_kernel::kernel::ProcessState::Waiting { .. }
                | aios_kernel::kernel::ProcessState::Sleeping { .. }
                | aios_kernel::kernel::ProcessState::Stopped
        )
    {
        should_terminate = false;
        if matches!(verdict, RlimitVerdict::Exceeded { .. }) {
            p.mailbox.push_back(format!(
                "[rlimit-warning] {}",
                format_rlimit_termination_result(verdict)
            ));
        }
        termination_result = "Completed".to_string();
    }
    (should_terminate, termination_result)
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

        let background_procs: Vec<aios_kernel::kernel::Process> = {
            let mut os = app.os.lock().unwrap();
            select_background_batch(os.as_mut(), epoch, app.session_id.as_str())
        };
        maybe_emit_scheduler_eval(epoch, app.session_id.as_str());

        if !background_procs.is_empty() {
            ensure_runtime_manifests_loaded(
                app,
                skill_manifests,
                agent_manifests,
                &mut manifests_loaded,
            );

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

            let mut task_specs: Vec<(
                u64,
                String,
                PathBuf,
                Option<String>,
                Option<String>,
                Option<u64>,
                Option<aios_kernel::primitives::FutexAddr>,
                Option<String>,
                Option<crate::ai::models::AutoModelFallbackSpec>,
                bool,
            )> = Vec::new();
            for proc in &background_procs {
                let pid = proc.pid;
                let task_goal = match decode_background_process_task_goal(&proc.goal) {
                    Ok(goal) => goal,
                    Err(err) => {
                        let (result_channel_id, completion_futex_addr) =
                            with_task_entry_by_pid(pid, |entry| {
                                (Some(entry.result_channel_id), Some(entry.completion_futex_addr))
                            })
                            .unwrap_or((None, None));
                        let mut os = app.os.lock().unwrap();
                        publish_background_task_failure(
                            os.as_mut(),
                            pid,
                            result_channel_id,
                            completion_futex_addr,
                            &format!("Corrupted subagent task goal for pid {}: {}", pid, err),
                        );
                        continue;
                    }
                };
                let mailbox_messages: Vec<String> = proc.mailbox.iter().cloned().collect();
                // mailbox 非空时 build_background_process_question 走 format_wakeup_prompt，
                // 生成的是系统调度通知（非用户输入），持久化时应标记为 internal_note。
                let is_resume_wakeup = !mailbox_messages.is_empty();
                if !mailbox_messages.is_empty() {
                    let mut os = app.os.lock().unwrap();
                    if let Some(actual) = os.get_process_mut(pid) {
                        actual.mailbox.clear();
                    }
                }
                let proc_question = build_background_process_question(
                    pid,
                    &proc.goal,
                    task_goal.as_ref().map(|goal| goal.prompt.as_str()),
                    &mailbox_messages,
                );

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
                task_specs.push((
                    pid,
                    proc_question,
                    history_path,
                    task_goal.as_ref().map(|goal| goal.agent_name.clone()),
                    task_goal.as_ref().map(|goal| goal.model.clone()),
                    task_goal.as_ref().map(|goal| goal.result_channel_id),
                    task_goal
                        .as_ref()
                        .map(|goal| aios_kernel::primitives::FutexAddr(goal.completion_futex_addr)),
                    task_goal.as_ref().map(|goal| goal.task_id.clone()),
                    task_goal.as_ref().and_then(|goal| goal.auto_model_fallback),
                    is_resume_wakeup,
                ));
            }

            for (
                pid,
                proc_question,
                history_path,
                agent_override,
                model_override,
                result_channel_id,
                completion_futex_addr,
                task_id,
                auto_model_fallback,
                is_resume_wakeup,
            ) in task_specs
            {
                let mut task_app = app.clone();
                crate::ai::types::clear_stream_cancel(&task_app);
                let task_mcp = mcp_client.clone();
                let task_os = app.os.clone();
                let task_agent = match resolve_background_subagent_override(
                    agent_manifests.as_slice(),
                    agent_override.as_deref(),
                ) {
                    Ok(agent) => agent,
                    Err(err) => {
                        let mut os = app.os.lock().unwrap();
                        publish_background_task_failure(
                            os.as_mut(),
                            pid,
                            result_channel_id,
                            completion_futex_addr,
                            &err,
                        );
                        continue;
                    }
                };
                if let Some(agent) = task_agent {
                    activate_primary_agent(&mut task_app, agent);
                }
                let next_model = model_override.unwrap_or_else(|| app.current_model.clone());

                let inherit = task_id
                    .as_deref()
                    .and_then(|tid| {
                        crate::ai::tools::task_tools::with_task_entry(tid, |e| e.inherit)
                    })
                    .unwrap_or_default();
                let (effective_history, task_skills) = resolve_background_subagent_context(
                    history_path,
                    original_history_file.as_path(),
                    skill_manifests,
                    task_id.as_deref(),
                    inherit,
                );
                task_app.session_history_file = effective_history;
                let task_driver_ctx = runtime_ctx::DriverContext::new(
                    task_app.clone(),
                    task_mcp.clone(),
                    task_skills.clone(),
                    agent_manifests.clone(),
                );
                let scope_task_id = task_id.clone().unwrap_or_else(|| format!("pid-{pid}"));
                let parent_history_for_scopes = original_history_file.clone();

                // Slot used by the sub-agent's `finalize_turn` to publish
                // its final assistant text. Cloned into the result-channel
                // payload below so `task_wait` can surface what the
                // sub-agent actually produced (instead of just "completed
                // with empty output").
                let result_slot_for_payload: runtime_ctx::SubagentResultSlot =
                    std::sync::Arc::new(tokio::sync::Mutex::new(None));
                let result_slot_for_scope = result_slot_for_payload.clone();

                let inner_fut = TASK_PID.scope(Some(pid), async move {
                    crate::ai::tools::registry::common::clear_tool_cancel();
                    let run = runtime_ctx::IS_RESUME_TURN.scope(
                        is_resume_wakeup,
                        turn_runtime::run_turn(
                            &mut task_app,
                            &task_mcp,
                            &task_skills,
                            usize::MAX,
                            proc_question,
                            String::new(),
                            next_model,
                            None,
                            false,
                            false,
                        ),
                    );
                    let result = if let Some(spec) = auto_model_fallback {
                        runtime_ctx::AUTO_MODEL_FALLBACK.scope(spec, run).await
                    } else {
                        run.await
                    }
                    .map_err(|e| format!("{}", e));
                    let captured_output = if result_channel_id.is_some() {
                        result_slot_for_payload
                            .lock()
                            .await
                            .clone()
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };
                    let mut os = task_os.lock().unwrap();
                    os.set_current_pid(Some(pid));
                    let publish_task_result = should_publish_subagent_task_result(
                        result.is_ok(),
                        &captured_output,
                        os.get_process(pid).map(|proc| &proc.state),
                    );
                    if publish_task_result && let Some(result_channel_id) = result_channel_id {
                        let payload = serde_json::json!({
                            "status": if result.is_ok() { "completed" } else { "failed" },
                            "output": captured_output,
                            "error": result.as_ref().err().cloned(),
                        })
                        .to_string();
                        let _ = os.channel_send(
                            Some(pid),
                            aios_kernel::primitives::ChannelId(result_channel_id),
                            payload,
                        );
                        let _ = os.channel_close(
                            Some(pid),
                            aios_kernel::primitives::ChannelId(result_channel_id),
                        );
                        let _ = os.channel_release_named(
                            aios_kernel::primitives::ChannelId(result_channel_id),
                            "task_result.producer",
                        );
                    }
                    if publish_task_result && let Some(addr) = completion_futex_addr {
                        let _ = os.futex_store(addr, 1);
                    }
                    match result {
                        Ok(_outcome) => {
                            let outcome = classify_process_outcome(&**os, pid);
                            record_scheduler_outcome(os.as_mut(), pid, outcome);
                            os.increment_turns_used_for(pid);
                            let (should_terminate, termination_result) =
                                finalize_turn_quota(os.as_mut(), pid);
                            if should_terminate {
                                terminate_and_cleanup(os.as_mut(), pid, termination_result, true);
                            } else if os.is_round_robin() {
                                os.set_current_pid(Some(pid));
                                os.requeue_current();
                            }
                        }
                        Err(err) => {
                            record_scheduler_outcome(os.as_mut(), pid, DispatchOutcomeTag::Failed);
                            terminate_and_cleanup(
                                os.as_mut(),
                                pid,
                                format!("Failed: {}", err),
                                true,
                            );
                        }
                    }
                });

                type BoxedTaskFuture =
                    std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
                let mut wrapped: BoxedTaskFuture = Box::pin(inner_fut);
                let persona_memory_path = app.current_persona_memory_file();
                wrapped = Box::pin(
                    runtime_ctx::PERSONA_MEMORY_PATH.scope(persona_memory_path.clone(), wrapped),
                );
                wrapped = Box::pin(
                    runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot_for_scope, wrapped),
                );
                if !inherit.memory {
                    let mem_path = runtime_ctx::make_subagent_memory_path(
                        &parent_history_for_scopes,
                        &scope_task_id,
                    );
                    // sub-agent 默认私有 memory：finalize 后把白名单条目
                    // (is_permanent_memory) 合并回主 memory 文件，让 long-term
                    // assets 能跨 task 共享，但普通 task_event 留在私有文件，
                    // 不污染主记忆。
                    let main_path = persona_memory_path;
                    let private_for_merge = mem_path.clone();
                    wrapped = Box::pin(runtime_ctx::SUBAGENT_MEMORY_PATH.scope(mem_path, wrapped));
                    // 这里包一层 outer future：sub-agent run 完成后 merge。
                    // merge_subagent_whitelist 内部用 for_tests_with_path
                    // 直接绑定 main_path，绕过 SUBAGENT_MEMORY_PATH override，
                    // 避免白名单条目又被写回私有文件（=死循环）。
                    let inner = wrapped;
                    wrapped = Box::pin(async move {
                        inner.await;
                        let _ = crate::ai::tools::service::memory::merge_subagent_whitelist(
                            &private_for_merge,
                            &main_path,
                        );
                    });
                }
                if !inherit.cwd {
                    let scratch_base = parent_history_for_scopes
                        .parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| PathBuf::from("."));
                    if let Some(scratch) =
                        runtime_ctx::make_subagent_cwd(&scratch_base, &scope_task_id)
                    {
                        wrapped = Box::pin(runtime_ctx::SUBAGENT_CWD.scope(scratch, wrapped));
                    }
                }

                // 计入在途后台子 agent：guard 随 spawned future 一同 move 进任务，
                // 任务结束（正常 / 错误 / panic）时 Drop 自动 dec，避免输入框被永久门控。
                let inflight_guard = BgSubagentGuard::new();
                let guarded_fut = async move {
                    let _guard = inflight_guard;
                    wrapped.await
                };
                tokio::spawn(runtime_ctx::DRIVER_CTX.scope(task_driver_ctx, guarded_fut));
            }
        }

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
                .map(|g| {
                    commands::goal::build_goal_continuation_prompt(g)
                });

            if let Some(cont) = goal_continuation {
                question = cont;
                attachments_text = String::new();
                history_count = 0;
            } else {
                // goal 激活但上一轮无工具调用 → 目标已达成，退出 goal 模式
                if app.goal_mode.as_ref().map_or(false, |g| !g.is_empty())
                    && !one_shot_mode
                {
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
            match note_search::handle_note_search_interactive_turn(app, &question, history_count).await {
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
