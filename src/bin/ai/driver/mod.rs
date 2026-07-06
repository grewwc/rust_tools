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
use rustc_hash::{FxHashMap, FxHashSet};
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
pub mod observer;
pub mod print;
pub mod reflection;
pub mod runtime_ctx;
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

/// Default max LLM iterations allowed per turn (prevents infinite loops)
const DEFAULT_MAX_ITERATIONS: usize = 4096;

/// Max iterations for subagent (executor) processes
const EXECUTOR_MAX_ITERATIONS: usize = 512;

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
    use crate::ai::history::{build_message_arr, read_recent_messages_sqlite};

    let is_sqlite_history = matches!(
        app.session_history_file
            .extension()
            .and_then(|ext| ext.to_str()),
        Some("sqlite") | Some("db")
    );

    if is_sqlite_history {
        return read_recent_messages_sqlite(app.session_history_file.as_path(), 10)
            .unwrap_or_default();
    }

    build_message_arr(10, &app.session_history_file)
        .map(|entries| entries.into_iter().rev().collect())
        .unwrap_or_default()
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

fn note_search_interactive_mode(cli: &cli::ParsedCli) -> bool {
    cli.note_search && cli.interactive
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
    };
    if let Some(notice) = startup_notice {
        println!("{notice}");
    }
    // 处理 --note-delete / -nd：输入一段话，模型自动匹配知识库条目，确认后删除。
    if let Some(query) = app.cli.note_delete.clone() {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                handle_note_delete(&mut app, &query),
            )
            .await;
    }

    // 处理 --note-edit / -ne：输入一段话，模型匹配知识库条目，在编辑器中改写后保存。
    if let Some(query) = app.cli.note_edit.clone() {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                handle_note_edit(&mut app, &query),
            )
            .await;
    }

    // 处理 --note / -n：快速保存 memo 到知识库并退出。
    // 即使没有文本（只想保存剪贴板图片），只要传了 -n 也要进入保存流程。
    if app.cli.note_flag {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                handle_note_save(&mut app),
            )
            .await;
    }

    // 处理 --note-search / -ns：默认单轮 notebook 检索后直接退出；若带 `-i`
    // 则进入交互模式，由 run_loop 在每轮输入时继续执行 notebook 检索问答。
    if app.cli.note_search && !app.cli.interactive {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(app.current_persona_memory_file(), handle_memo_search(&app))
            .await;
    }
    if app.cli.consolidate_knowledge {
        return runtime_ctx::PERSONA_MEMORY_PATH
            .scope(
                app.current_persona_memory_file(),
                handle_consolidate_knowledge(&app),
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

/// 处理 --note / -n 参数：快速保存 memo 到知识库并退出。
/// 如果剪贴板有图片，使用视觉模型理解内容；
/// 否则使用 `-n` 后面提供的文本；若也没有文本，则进入多行输入框让用户输入。
async fn handle_note_save(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
    use arboard::Clipboard;
    use image::buffer::ConvertBuffer;
    use image::{ImageBuffer, Rgb, Rgba};
    use std::fs;

    let store = MemoryStore::from_env_or_config();
    // -n 是字符串 flag，只会捕获其后的第一个 token（如 `a -n aeolus 线上日志路径：...`
    // 只会把 "aeolus" 当作 note 值），其余 token 落到位置参数里。这里把位置参数拼接回来，
    // 避免内容被截断、导致后续检索不到完整笔记。
    let provided_text = {
        let mut parts: Vec<String> = Vec::new();
        if let Some(text) = app.cli.note.clone() {
            let text = text.trim();
            if !text.is_empty() {
                parts.push(text.to_string());
            }
        }
        let extra = app.cli.args.join(" ");
        let extra = extra.trim();
        if !extra.is_empty() {
            parts.push(extra.to_string());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    };

    // 图片持久化目录：与 memory 文件同目录下的 note_images/。
    // 之前的实现把截图写进 /tmp 然后立即删除、并存 image_path: None，
    // 导致图片彻底丢失、memo 永远无法再引用原图。改成持久化保存。
    let images_dir = store
        .path()
        .parent()
        .map(|parent| parent.join("note_images"))
        .unwrap_or_else(|| PathBuf::from("note_images"));

    // 尝试从剪贴板获取图片并持久化保存
    let clipboard_image_path: Option<String> = match Clipboard::new() {
        Ok(mut clipboard) => {
            if let Ok(image) = clipboard.get_image() {
                let data = image.bytes;
                if !data.is_empty() {
                    let image_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
                        image.width as u32,
                        image.height as u32,
                        data.to_vec(),
                    );
                    if let Some(buf) = image_buf {
                        let rgb_buf: ImageBuffer<Rgb<u8>, Vec<u8>> = buf.convert();
                        if let Err(err) = fs::create_dir_all(&images_dir) {
                            eprintln!("[note] Failed to create image dir: {}", err);
                            None
                        } else {
                            let file_name = format!(
                                "note_{}_{}.png",
                                chrono::Local::now().format("%Y%m%d_%H%M%S"),
                                std::process::id()
                            );
                            let save_path = images_dir.join(file_name);
                            if rgb_buf.save(&save_path).is_ok() {
                                Some(save_path.to_string_lossy().into_owned())
                            } else {
                                None
                            }
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        Err(_) => None,
    };

    let note_content = if let Some(image_path) = &clipboard_image_path {
        // 有图片，调用视觉模型理解内容
        println!("[note] Detected image in clipboard, analyzing...");

        let model = crate::ai::models::default_vl_model();

        // 构建包含图片的消息
        let content = crate::ai::request::build_content(
            &model,
            "请详细描述这张图片的内容，包括关键信息、文字、数据等。用中文回答。",
            &[image_path.clone()],
        )?;

        let messages = vec![serde_json::json!({
            "role": "user",
            "content": content,
        })];

        // 调用模型
        match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
            Ok(response) => {
                if let Some(content) = response.pointer("/choices/0/message/content") {
                    content.as_str().unwrap_or("无法解析图片内容").to_string()
                } else {
                    "无法获取模型响应".to_string()
                }
            }
            Err(err) => {
                eprintln!("[note] Failed to analyze image: {}", err);
                let _ = fs::remove_file(image_path);
                return Err(err);
            }
        }
    } else {
        // 没有图片：取得原始文本（来自 -n 后面的文本，或多行输入框），
        // 统一先交给模型理解、整理后再保存，避免直接堆原文。
        let raw = if let Some(text) = provided_text.filter(|t| !t.trim().is_empty()) {
            text
        } else {
            // 既没有图片也没有文本：进入多行输入框，让用户手动输入要保存的内容。
            println!("[note] 剪贴板没有图片，请输入要保存的内容（多行；提交后保存，留空取消）：");
            let input = match app.prompt_editor.as_mut() {
                Some(editor) => editor.read_multi_line().ok().flatten(),
                None => None,
            };
            match input {
                Some(s) if !s.trim().is_empty() => s,
                _ => {
                    eprintln!("[note] 未输入任何内容，已取消");
                    return Err("no content to save".into());
                }
            }
        };

        // 调用模型理解并整理用户输入，使其更适合作为知识库 memo。
        println!("[note] 正在整理内容...");
        let model = crate::ai::models::initial_model(&app.cli);
        let messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "你是一个笔记整理助手。请把用户输入的内容理解、整理、改写为一条清晰、结构化、便于日后检索的笔记。\
                            保留所有关键信息和事实，去除口语化冗余，必要时用简洁的要点组织。直接输出整理后的笔记正文，不要添加任何解释或前后缀。用中文回答。",
            }),
            serde_json::json!({
                "role": "user",
                "content": raw,
            }),
        ];
        match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
            Ok(response) => response
                .pointer("/choices/0/message/content")
                .and_then(|c| c.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or(raw),
            Err(err) => {
                // 整理失败时退回保存原始输入，避免丢失用户内容。
                eprintln!("[note] 整理失败，保存原始输入: {}", err);
                raw
            }
        }
    };

    // 保存到知识库（图片已持久化，路径写入 image_path 以便后续引用）
    let now = chrono::Local::now().to_rfc3339();
    let entry = AgentMemoryEntry {
        id: Some(format!("mem_{}", uuid::Uuid::new_v4().simple())),
        timestamp: now,
        category: "memo".to_string(),
        note: note_content.clone(),
        tags: vec![],
        source: Some("cli_note".to_string()),
        priority: Some(150),
        owner_pid: None,
        owner_pgid: None,
        image_path: clipboard_image_path.clone(),
    };

    match store.append(&entry) {
        Ok(()) => {
            if let Some(image_path) = &clipboard_image_path {
                println!(
                    "[note] Image content saved to knowledge base [memo] (image: {}):",
                    image_path
                );
            } else {
                println!("[note] Saved to knowledge base [memo]:");
            }
            println!("  {}", note_content.chars().take(200).collect::<String>());
            if note_content.chars().count() > 200 {
                println!("  ...");
            }
        }
        Err(err) => {
            eprintln!("[note] Failed to save: {}", err);
            return Err(err.into());
        }
    }
    Ok(())
}

/// 一个轻量的终端 "Searching..." 动画提示。
///
/// 在 stderr 上用回车 `\r` 原地刷新一帧帧 spinner，`stop()` / drop 时清除当前行，
/// 不会污染随后的正式输出（正式结果走 stdout）。仅在 stderr 为 TTY 时启用，
/// 管道 / 重定向场景自动静默，避免写入垃圾字符。
struct SearchSpinner {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl SearchSpinner {
    fn start(label: &str) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        // 非 TTY（被管道/重定向）时不画动画，返回一个空 spinner。
        if !std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            return Self { stop, handle: None };
        }
        let label = label.to_string();
        let stop_cloned = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            use std::io::Write as _;
            const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut i = 0usize;
            while !stop_cloned.load(Ordering::Relaxed) {
                let mut err = std::io::stderr();
                let _ = write!(err, "\r{} {}...", FRAMES[i % FRAMES.len()], label);
                let _ = err.flush();
                i += 1;
                std::thread::sleep(Duration::from_millis(80));
            }
            // 清除当前行（足够覆盖 "<frame> <label>..."）。
            let mut err = std::io::stderr();
            let _ = write!(err, "\r{}\r", " ".repeat(label.len() + 8));
            let _ = err.flush();
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn stop(self) {
        // 显式消费，触发 Drop。
        drop(self);
    }
}

impl Drop for SearchSpinner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

const NOTE_SEARCH_QUERY_HISTORY_MAX_MESSAGES: usize = 4;
const NOTE_SEARCH_QUERY_HISTORY_MAX_CHARS: usize = 200;

fn truncate_note_search_excerpt(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let mut out = trimmed.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

fn build_note_search_retrieval_query(
    question: &str,
    recent_history: &[crate::ai::history::Message],
) -> String {
    let question = question.trim();
    if question.is_empty() {
        return String::new();
    }

    let snippets = recent_history
        .iter()
        .filter(|message| matches!(message.role.as_str(), "user" | "assistant"))
        .filter_map(|message| {
            let content = crate::ai::history::value_to_string(&message.content);
            let content =
                truncate_note_search_excerpt(&content, NOTE_SEARCH_QUERY_HISTORY_MAX_CHARS);
            if content.is_empty() {
                return None;
            }
            let role = if message.role == "user" {
                "用户"
            } else {
                "助手"
            };
            Some(format!("{role}: {content}"))
        })
        .take(NOTE_SEARCH_QUERY_HISTORY_MAX_MESSAGES)
        .collect::<Vec<_>>();
    let mut snippets = snippets;
    snippets.reverse();

    if snippets.is_empty() {
        return question.to_string();
    }

    format!(
        "当前问题：{question}\n最近对话上下文：\n{}",
        snippets.join("\n")
    )
}

fn build_note_search_chat_history(
    app: &App,
    history_count: usize,
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let overflow_dir = {
        let store = SessionStore::new(app.config.history_file.as_path());
        Some(store.session_assets_dir(&app.session_id))
    };
    let history = crate::ai::history::build_context_history(
        history_count,
        &app.session_history_file,
        app.config.history_max_chars,
        app.config.history_keep_last,
        app.config.history_summary_max_chars,
        overflow_dir,
    )?;

    Ok(history
        .into_iter()
        .filter(|message| matches!(message.role.as_str(), "user" | "assistant"))
        .filter_map(|message| {
            let content = crate::ai::history::value_to_string(&message.content);
            let content = content.trim().to_string();
            if content.is_empty() {
                return None;
            }
            Some(serde_json::json!({
                "role": message.role,
                "content": content,
            }))
        })
        .collect())
}

fn select_note_search_candidates<'a>(
    candidates: &'a [crate::ai::tools::service::memory::ScoredMemo],
) -> Vec<&'a crate::ai::tools::service::memory::ScoredMemo> {
    if candidates
        .first()
        .is_some_and(|candidate| candidate.semantic)
    {
        let top = candidates[0].score.max(1e-6);
        let threshold = top * 0.6;
        candidates
            .iter()
            .enumerate()
            .filter(|(index, candidate)| *index == 0 || candidate.score >= threshold)
            .take(8)
            .map(|(_, candidate)| candidate)
            .collect()
    } else {
        candidates.iter().collect()
    }
}

async fn answer_memo_search(
    app: &App,
    question: &str,
    history_count: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let question = question.trim().to_string();
    if question.is_empty() {
        eprintln!("[note-search] 用法: a -ns <查询内容>");
        return Err("note-search requires a query".into());
    }

    // 安装远程 embedding provider（若已配置）。必须在任何 embedder::is_ready()
    // 调用之前执行——GLOBAL_PROVIDER 是 OnceLock，首次读取即定型。
    // 未配置 / 配置不全时此调用无副作用，检索退回 BM25/lexical。
    crate::ai::knowledge::indexing::embedder::warm_up();

    // 检索 + 模型总结都可能耗时，给一个 "Searching..." 动画提示（输出前自动清除）。
    let _spinner = SearchSpinner::start("Searching memo");
    let retrieval_query = if note_search_interactive_mode(&app.cli) {
        build_note_search_retrieval_query(&question, &read_recent_history(app))
    } else {
        question.clone()
    };

    // 检索相关 memo 条目作为上下文。
    let candidates = match crate::ai::tools::service::memory::search_memo_candidates_scored(
        &retrieval_query,
        20,
    ) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("[note-search] 检索失败: {}", err);
            return Err(err.into());
        }
    };
    if candidates.is_empty() {
        return Ok(format!("没有在知识库中找到与「{}」相关的内容。", question));
    }

    // 按语义分数收紧喂给 LLM 的条数，进一步防止大知识库撑爆上下文。
    // 仅在本次确实用了语义打分（embedding 可用）时收紧——此时分数可比较、
    // 排序可信；否则保持全部 20 条交给 LLM（与历史行为一致，不丢候选）。
    // 收紧策略：保留 top1 锚点；其余条目要求语义分数 >= top1 的 60% 才纳入，
    // 至多 8 条。这样只砍掉明显不相关的长尾，不影响真正相关的笔记。
    let selected = select_note_search_candidates(&candidates);

    // 把检索到的条目作为上下文，让模型基于这些内容回答用户的问题。
    let mut context = String::new();
    for (idx, candidate) in selected.iter().enumerate() {
        context.push_str(&format!("[{}] {}\n", idx + 1, candidate.entry.note));
    }

    let mut messages = vec![serde_json::json!({
        "role": "system",
        "content": "你处于 notebook 检索问答模式。下面会给出当前问题，以及本轮从用户 notebook（memo）里检索到的若干条笔记。\
                    每一轮都必须优先依据本轮检索结果回答。最近对话仅用于理解省略、代词和追问；如果最近对话与本轮检索结果冲突，以本轮检索结果为准。\
                    如果检索结果里没有足够信息回答，就直接说明。用中文回答，使用 Markdown 格式。",
    })];
    if note_search_interactive_mode(&app.cli) {
        messages.extend(build_note_search_chat_history(app, history_count)?);
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": format!("当前问题：{}\n\n本轮 notebook 检索结果：\n{}", question, context),
    }));

    match crate::ai::request::do_request_json(app, &app.current_model, &messages, false, true).await
    {
        Ok(response) => {
            let answer = response
                .pointer("/choices/0/message/content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if answer.is_empty() {
                // 模型无输出时退回展示已选中的原始条目（复用上面的检索结果，不重复检索）。
                Ok(selected
                    .iter()
                    .enumerate()
                    .map(|(i, candidate)| format!("{}. {}", i + 1, candidate.entry.note))
                    .collect::<Vec<_>>()
                    .join("\n\n"))
            } else {
                Ok(answer)
            }
        }
        Err(err) => {
            eprintln!("[note-search] 总结失败: {}", err);
            Err(err.into())
        }
    }
}

fn persist_note_search_turn(app: &App, question: &str, answer: &str) {
    let question = question.trim();
    let answer = answer.trim();
    if question.is_empty() || answer.is_empty() {
        return;
    }

    let messages = vec![
        crate::ai::history::Message {
            role: "user".to_string(),
            content: serde_json::Value::String(question.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(answer.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];
    if let Err(err) = crate::ai::history::append_history_messages_uncompacted(
        &app.session_history_file,
        &messages,
    ) {
        eprintln!("[Warning] Failed to save notebook search history: {}", err);
    }
}

async fn handle_note_search_interactive_turn(
    app: &App,
    question: &str,
    history_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    crate::ai::types::clear_stream_cancel(app);
    crate::ai::tools::registry::common::clear_tool_cancel();
    let _guard = signal::ForegroundTurnGuard::enter();
    let answer = answer_memo_search(app, question, history_count).await?;
    crate::ai::stream::render_markdown_block(&answer).ok();
    persist_note_search_turn(app, question, &answer);
    Ok(())
}

/// 处理 --note-search / -ns：从知识库中检索 memo 类条目，再用模型根据检索到的
/// 内容总结、回答用户的问题（而不是直接堆砌原始条目）。
async fn handle_memo_search(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    let query = app.cli.args.join(" ");
    let answer = answer_memo_search(app, &query, 0).await?;
    crate::ai::stream::render_markdown_block(&answer).ok();
    Ok(())
}

// 处理 consolidate 计划里的 merge 项：只为有效 merge（ids 非空且 merged_content 非空）
// 生成新条目，并把对应源 IDs 自动并入删除集合，避免“旧条目 + 合并条目”并存。
fn build_consolidation_merge_entries(
    merge_plan: &[&serde_json::Value],
) -> (
    FxHashSet<String>,
    usize,
    Vec<crate::ai::tools::storage::memory_store::AgentMemoryEntry>,
) {
    let mut merge_delete_ids = FxHashSet::default();
    let mut merged_count = 0usize;
    let mut new_entries = Vec::new();

    for item in merge_plan {
        let ids: Vec<&str> = item["ids"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let content = item["merged_content"].as_str().unwrap_or("").trim();
        if ids.is_empty() || content.is_empty() {
            continue;
        }

        merged_count += ids.len();
        merge_delete_ids.extend(ids.iter().map(|id| (*id).to_string()));
        new_entries.push(crate::ai::tools::storage::memory_store::AgentMemoryEntry {
            id: Some(crate::ai::tools::service::memory::next_memory_id()),
            timestamp: chrono::Local::now().to_rfc3339(),
            category: "user_memory".into(),
            note: content.to_string(),
            tags: vec!["consolidated".into()],
            source: None,
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        });
    }

    (merge_delete_ids, merged_count, new_entries)
}

/// 处理 --consolidate-knowledge：读取全部知识条目 → 模型分析 → 执行整理。
///
/// **优化策略**（避免 60s 超时）：
/// 1. 只分析优先级 < 200 的条目（≥200 受保护）
/// 2. 按时间倒序取**最近 15 条**（之前 30 条还是太多）
/// 3. 每条内容截断到**40 字**（之前 80 字）
/// 4. 用 JSON 数组格式（比文本格式更省 token）
/// 5. 英文 system prompt（模型响应更快）
async fn handle_consolidate_knowledge(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
    use serde_json::Value;

    let store = MemoryStore::from_env_or_config();
    let all_entries = store.all().map_err(|e| format!("读取失败：{}", e))?;

    if all_entries.is_empty() {
        println!("📭 知识库为空，无需整理。");
        return Ok(());
    }

    // 过滤：优先级 < 200 的才分析；按时间倒序；取最近 15 条
    let mut candidates: Vec<&AgentMemoryEntry> = all_entries
        .iter()
        .filter(|e| e.priority.unwrap_or(100) < 200)
        .collect();
    candidates.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    candidates.truncate(15);

    if candidates.is_empty() {
        println!("📭 没有可整理的条目（全部优先级 ≥ 200，已保护）。");
        return Ok(());
    }

    // 构建紧凑的 JSON 数组（比文本格式更省 token）
    let mut entries_json = Vec::new();
    for entry in &candidates {
        let id = entry.id.as_deref().unwrap_or("unknown");
        let prio = entry.priority.unwrap_or(100);
        let ts_short: String = entry.timestamp.chars().take(10).collect();
        let preview: String = if entry.note.chars().count() > 40 {
            entry.note.chars().take(40).collect::<String>() + "…"
        } else {
            entry.note.clone()
        };
        entries_json.push(serde_json::json!({
            "id": id,
            "cat": entry.category,
            "pri": prio,
            "tags": entry.tags,
            "date": ts_short,
            "src": entry.source.as_deref().unwrap_or(""),
            "text": preview,
        }));
    }

    let sys = "You are a knowledge curator. Analyze entries and suggest deletions/merges.\n\
        Return ONLY valid JSON:\n\
        {\"reasoning\":\"1-sentence summary\",\"delete_ids\":[\"id1\",\"id2\"],\"merge_plan\":[{\"ids\":[\"id1\",\"id2\"],\"merged_content\":\"...\"}]}\n\
        Rules: delete duplicates/obsolete; merge related; keep useful. Priority>=200 already filtered out.";

    let prompt = format!(
        "Analyze these {} entries:\n{}",
        candidates.len(),
        serde_json::to_string(&entries_json).unwrap()
    );
    let messages = vec![
        serde_json::json!({"role": "system", "content": sys}),
        serde_json::json!({"role": "user", "content": prompt}),
    ];

    // 知识整理用主模型（用户的默认对话模型）。走流式链路：响应头立即返回、
    // 数据按 chunk 增量到达，避免非流式"等整段 body 生成完"被 60s 超时撑爆。
    let model = crate::ai::models::initial_model(&app.cli);
    let spinner = SearchSpinner::start("整理知识库");
    let raw = match crate::ai::request::do_request_text_streaming(app, &model, &messages).await {
        Ok(text) => {
            spinner.stop();
            text
        }
        Err(err) => {
            spinner.stop();
            eprintln!("[consolidate] Request failed: {}", err);
            return Err(err);
        }
    };

    let raw = raw.trim();
    let cleaned = raw
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    if cleaned.is_empty() || raw.is_empty() {
        println!("⚠  Empty response. No changes.");
        return Ok(());
    }

    let plan: Value = match serde_json::from_str(cleaned) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[consolidate] JSON parse error: {}", e);
            eprintln!(
                "[consolidate] Raw: {}",
                raw.chars().take(200).collect::<String>()
            );
            return Ok(());
        }
    };

    if let Some(reasoning) = plan["reasoning"].as_str() {
        println!("\n🔍 {}\n", reasoning);
    }

    let delete_ids: Vec<&str> = plan["delete_ids"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let merge_plan: Vec<&Value> = plan["merge_plan"]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    let (merge_delete_ids, merged_count, new_entries) =
        build_consolidation_merge_entries(&merge_plan);

    let mut delete_id_set: FxHashSet<String> =
        delete_ids.iter().map(|id| (*id).to_string()).collect();
    delete_id_set.extend(merge_delete_ids);

    if delete_id_set.is_empty() && new_entries.is_empty() {
        println!("✅ Already well-organized. Nothing to change.");
        return Ok(());
    }

    let delete_refs: Vec<&str> = delete_id_set.iter().map(String::as_str).collect();
    match store.apply_batch_update(&delete_refs, &new_entries) {
        Ok(report) => {
            if !delete_refs.is_empty() {
                println!("🗑  Deleted {} entries", report.deleted);
            }
            if !new_entries.is_empty() {
                println!(
                    "💾 Merged {} entries into {} new",
                    merged_count, report.appended
                );
            }
        }
        Err(e) => {
            eprintln!("  Consolidation error: {}", e);
        }
    }

    println!("\n✨ Done.");
    Ok(())
}
/// 处理 --note-delete / -nd <一段话>：用模型在知识库中匹配最相关的 memo 条目，
/// 找到对应 id，删除前请用户确认。
async fn handle_note_delete(app: &mut App, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // 拼接查询：flag 的值 + 其余位置参数；都为空时进入多行输入框。
    let mut query = query.trim().to_string();
    if !app.cli.args.is_empty() {
        let extra = app.cli.args.join(" ");
        if !query.is_empty() {
            query.push(' ');
        }
        query.push_str(extra.trim());
    }
    let query = query.trim().to_string();
    let query = if query.is_empty() {
        println!("[note-delete] 请描述你想删除的内容（多行；提交后开始匹配，留空取消）：");
        let input = match app.prompt_editor.as_mut() {
            Some(editor) => editor.read_multi_line().ok().flatten(),
            None => None,
        };
        match input {
            Some(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => {
                eprintln!("[note-delete] 未输入任何内容，已取消");
                return Ok(());
            }
        }
    } else {
        query
    };

    // 检索候选条目。
    let candidates = match crate::ai::tools::service::memory::search_memo_candidates(&query, 10) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("[note-delete] 检索失败: {}", err);
            return Err(err.into());
        }
    };
    if candidates.is_empty() {
        println!(
            "[note-delete] 没有找到与「{}」相关的可删除 memo 条目。",
            query
        );
        return Ok(());
    }

    // 让模型从候选中挑选最匹配的一条（返回其序号，或 NONE）。
    let mut listing = String::new();
    for (idx, e) in candidates.iter().enumerate() {
        let note_preview: String = e.note.chars().take(300).collect();
        listing.push_str(&format!("{}. {}\n", idx + 1, note_preview));
    }

    let model = crate::ai::models::initial_model(&app.cli);
    let messages = vec![
        serde_json::json!({
            "role": "system",
            "content": "你是一个知识库删除助手。用户会给出一段描述，以及若干条带编号的候选笔记。\
                        请判断哪些条目符合用户想删除的内容——可能是一条，也可能是多条。\
                        只输出这些条目的编号，用英文逗号分隔（如 1 或 1,3,4）。\
                        如果没有任何一条明显匹配，只输出 NONE。不要输出任何解释或多余字符。",
        }),
        serde_json::json!({
            "role": "user",
            "content": format!("用户描述：{}\n\n候选条目：\n{}", query, listing),
        }),
    ];

    let chosen =
        match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
            Ok(response) => response
                .pointer("/choices/0/message/content")
                .and_then(|c| c.as_str())
                .map(|s| s.trim().to_string())
                .unwrap_or_default(),
            Err(err) => {
                eprintln!("[note-delete] 模型匹配失败: {}", err);
                String::new()
            }
        };

    // 解析模型返回的若干编号（支持逗号 / 空格 / 顿号等分隔），去重并保持升序。
    let mut chosen_indices: Vec<usize> = Vec::new();
    {
        let mut num = String::new();
        let flush = |num: &mut String, out: &mut Vec<usize>| {
            if let Ok(n) = num.parse::<usize>() {
                if n >= 1 && n <= candidates.len() {
                    let idx = n - 1;
                    if !out.contains(&idx) {
                        out.push(idx);
                    }
                }
            }
            num.clear();
        };
        for c in chosen.chars() {
            if c.is_ascii_digit() {
                num.push(c);
            } else {
                flush(&mut num, &mut chosen_indices);
            }
        }
        flush(&mut num, &mut chosen_indices);
    }
    chosen_indices.sort_unstable();

    if chosen_indices.is_empty() {
        println!(
            "[note-delete] 模型未能从候选中确定要删除的条目，已取消。可换个更具体的描述重试。"
        );
        return Ok(());
    }

    let targets: Vec<&crate::ai::tools::storage::memory_store::AgentMemoryEntry> =
        chosen_indices.iter().map(|&i| &candidates[i]).collect();

    // 删除前确认 + 精选。列出条目后，用户可以：
    //   - 直接回车 / y / all / a：删除全部列出条目
    //   - 输入编号（如 1,3）：只删除指定编号
    //   - n / 回车以外的取消词：取消
    println!("\n[note-delete] 匹配到以下 {} 条条目：", targets.len());
    for (n, target) in targets.iter().enumerate() {
        println!("  [{}]", n + 1);
        if let Some(id) = target.id.as_deref().filter(|s| !s.is_empty()) {
            println!("    id: {}", id);
        }
        println!("    时间: {}", target.timestamp);
        println!(
            "    内容: {}",
            target.note.chars().take(500).collect::<String>()
        );
    }
    print!("\n请输入要删除的编号（如 1,3；输入 all 删除全部，直接回车=全部，n=取消）: ");
    std::io::stdout().flush().ok();

    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).ok();
    let answer = answer.trim().to_lowercase();

    // 解析用户选择，得到最终要删除的 targets 子集。
    let selected: Vec<&crate::ai::tools::storage::memory_store::AgentMemoryEntry> = if answer
        .is_empty()
        || answer == "y"
        || answer == "yes"
        || answer == "all"
        || answer == "a"
    {
        targets.clone()
    } else if answer == "n" || answer == "no" || answer == "q" || answer == "cancel" {
        println!("[note-delete] 已取消，未删除任何内容。");
        return Ok(());
    } else {
        // 解析编号列表（针对上面列出的 1..=targets.len()）。
        let mut picks: Vec<usize> = Vec::new();
        let mut num = String::new();
        let flush = |num: &mut String, out: &mut Vec<usize>| {
            if let Ok(n) = num.parse::<usize>() {
                if n >= 1 && n <= targets.len() {
                    let idx = n - 1;
                    if !out.contains(&idx) {
                        out.push(idx);
                    }
                }
            }
            num.clear();
        };
        for c in answer.chars() {
            if c.is_ascii_digit() {
                num.push(c);
            } else {
                flush(&mut num, &mut picks);
            }
        }
        flush(&mut num, &mut picks);
        picks.sort_unstable();
        if picks.is_empty() {
            println!("[note-delete] 未识别到有效编号，已取消，未删除任何内容。");
            return Ok(());
        }
        picks.into_iter().map(|i| targets[i]).collect()
    };

    let mut deleted = 0usize;
    let mut failed = 0usize;
    for target in &selected {
        match crate::ai::tools::service::memory::delete_memo_entry(target) {
            Ok(_) => deleted += 1,
            Err(err) => {
                failed += 1;
                eprintln!(
                    "[note-delete] 删除失败 (时间 {}): {}",
                    target.timestamp, err
                );
            }
        }
    }
    println!(
        "[note-delete] 完成：已删除 {} 条，失败 {} 条。",
        deleted, failed
    );
    if failed > 0 && deleted == 0 {
        return Err("all deletions failed".into());
    }
    Ok(())
}

/// 处理 --note-edit / -ne <一段话>：用模型在知识库中匹配相关 memo 条目，
/// 匹配到多条时让用户选定一条，在编辑器中预填原文改写后保存（保留 id、更新时间戳）。
async fn handle_note_edit(app: &mut App, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // 状态行着色：与黑底白字的 note 正文区分开。
    const NE: &str = "\x1b[1;36m[note-edit]\x1b[0m"; // 青色加粗标签
    const FIELD: &str = "\x1b[2m"; // 字段名（id/时间/内容）暗灰
    const HINT: &str = "\x1b[1;32m"; // 操作提示绿色加粗
    const IDX: &str = "\x1b[1;33m"; // 候选编号黄色加粗
    const RST: &str = "\x1b[0m";

    // 拼接查询：flag 的值 + 其余位置参数；都为空时进入多行输入框。
    let mut query = query.trim().to_string();
    if !app.cli.args.is_empty() {
        let extra = app.cli.args.join(" ");
        if !query.is_empty() {
            query.push(' ');
        }
        query.push_str(extra.trim());
    }
    let query = query.trim().to_string();
    let query = if query.is_empty() {
        println!("{NE} 请描述你想修改的内容（多行；提交后开始匹配，留空取消）：");
        let input = match app.prompt_editor.as_mut() {
            Some(editor) => editor.read_multi_line().ok().flatten(),
            None => None,
        };
        match input {
            Some(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => {
                eprintln!("{NE} 未输入任何内容，已取消");
                return Ok(());
            }
        }
    } else {
        query
    };

    // 检索 + 模型匹配都可能耗时，给一个状态条动画（输出前自动清除），与 -ns 一致。
    let spinner = SearchSpinner::start("匹配知识库条目");

    // 检索候选条目。
    let candidates = match crate::ai::tools::service::memory::search_memo_candidates(&query, 10) {
        Ok(c) => c,
        Err(err) => {
            spinner.stop();
            eprintln!("{NE} 检索失败: {}", err);
            return Err(err.into());
        }
    };
    if candidates.is_empty() {
        spinner.stop();
        println!("{NE} 没有找到与「{}」相关的可修改 memo 条目。", query);
        return Ok(());
    }

    // 让模型从候选中挑选匹配的条目（可能多条），返回编号。
    let mut listing = String::new();
    for (idx, e) in candidates.iter().enumerate() {
        let note_preview: String = e.note.chars().take(300).collect();
        listing.push_str(&format!("{}. {}\n", idx + 1, note_preview));
    }

    let model = crate::ai::models::initial_model(&app.cli);
    let messages = vec![
        serde_json::json!({
            "role": "system",
            "content": "你是一个知识库编辑助手。用户会给出一段描述，以及若干条带编号的候选笔记。\
                        请判断哪些条目符合用户想修改的内容——可能是一条，也可能是多条。\
                        只输出这些条目的编号，用英文逗号分隔（如 1 或 1,3,4）。\
                        如果没有任何一条明显匹配，只输出 NONE。不要输出任何解释或多余字符。",
        }),
        serde_json::json!({
            "role": "user",
            "content": format!("用户描述：{}\n\n候选条目：\n{}", query, listing),
        }),
    ];

    let mut matched_err: Option<String> = None;
    let chosen =
        match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
            Ok(response) => response
                .pointer("/choices/0/message/content")
                .and_then(|c| c.as_str())
                .map(|s| s.trim().to_string())
                .unwrap_or_default(),
            Err(err) => {
                matched_err = Some(format!("{}", err));
                String::new()
            }
        };
    spinner.stop();
    if let Some(err) = matched_err {
        eprintln!("{NE} 模型匹配失败: {}", err);
    }

    // 解析模型返回的编号集合。
    let parse_indices = |s: &str, max: usize| -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        let mut num = String::new();
        let flush = |num: &mut String, out: &mut Vec<usize>| {
            if let Ok(n) = num.parse::<usize>() {
                if n >= 1 && n <= max {
                    let idx = n - 1;
                    if !out.contains(&idx) {
                        out.push(idx);
                    }
                }
            }
            num.clear();
        };
        for c in s.chars() {
            if c.is_ascii_digit() {
                num.push(c);
            } else {
                flush(&mut num, &mut out);
            }
        }
        flush(&mut num, &mut out);
        out.sort_unstable();
        out
    };

    let mut matched = parse_indices(&chosen, candidates.len());
    if matched.is_empty() {
        println!("{NE} 模型未能从候选中确定要修改的条目，已取消。可换个更具体的描述重试。");
        return Ok(());
    }

    // 匹配到多条：列出后让用户选定恰好一条来编辑（编辑是针对单条内容的）。
    let target_idx = if matched.len() == 1 {
        matched[0]
    } else {
        println!("\n{NE} 匹配到以下 {IDX}{}{RST} 条条目：", matched.len());
        for (n, &ci) in matched.iter().enumerate() {
            let e = &candidates[ci];
            println!("  {IDX}[{}]{RST}", n + 1);
            if let Some(id) = e.id.as_deref().filter(|s| !s.is_empty()) {
                println!("    {FIELD}id:{RST} {}", id);
            }
            println!("    {FIELD}时间:{RST} {}", e.timestamp);
            println!(
                "    {FIELD}内容:{RST} {}",
                e.note.chars().take(500).collect::<String>()
            );
        }
        print!("\n{HINT}请输入要修改的编号（只能选一条；n=取消）:{RST} ");
        std::io::stdout().flush().ok();
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer).ok();
        let answer = answer.trim().to_lowercase();
        if answer == "n" || answer == "no" || answer == "q" || answer == "cancel" {
            println!("{NE} 已取消，未修改任何内容。");
            return Ok(());
        }
        let picks = parse_indices(&answer, matched.len());
        match picks.first() {
            Some(&p) => matched.remove(p),
            None => {
                println!("{NE} 未识别到有效编号，已取消，未修改任何内容。");
                return Ok(());
            }
        }
    };

    let target = candidates[target_idx].clone();

    // 在编辑器中预填原文，让用户改写。
    println!("\n{NE} 将打开编辑器修改以下条目（原文已预填；留空或不改动即取消）：");
    if let Some(id) = target.id.as_deref().filter(|s| !s.is_empty()) {
        println!("    {FIELD}id:{RST} {}", id);
    }
    println!("    {FIELD}时间:{RST} {}", target.timestamp);

    let new_note = match app.prompt_editor.as_mut() {
        Some(editor) => {
            editor.set_prefill(target.note.clone());
            editor.read_multi_line().ok().flatten()
        }
        None => None,
    };
    let new_note = match new_note {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => {
            println!("{NE} 未输入新内容，已取消。");
            return Ok(());
        }
    };
    if new_note == target.note.trim() {
        println!("{NE} 内容未变化，已取消。");
        return Ok(());
    }

    // 在保存前用 LLM 整理用户改写后的内容：只做格式/表达上的润色，
    // 严格禁止改变语义。整理失败则回退到用户编辑的原文，不阻塞保存。
    let final_note = {
        let spinner = SearchSpinner::start("整理修改内容");
        let mut tidy_err: Option<String> = None;
        let tidy_messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "你是一个知识库整理助手。用户会给出一段刚刚在编辑器里改写完的笔记内容。\
                            请帮用户整理这段内容，使其更清晰、更易读。\
                            \n严格约束：\n\
                            1. 绝对不要改变内容的语义、事实或意图，只能调整格式、排版、标点和表达方式；\n\
                            2. 不要增删任何实质性信息；\n\
                            3. 保留原文的语言（中文保持中文，英文保持英文）；\n\
                            4. 只输出整理后的正文，不要输出任何解释、前后缀或 markdown 代码块标记。",
            }),
            serde_json::json!({
                "role": "user",
                "content": new_note.clone(),
            }),
        ];
        let result =
            match crate::ai::request::do_request_json(app, &model, &tidy_messages, false, false).await {
                Ok(response) => response
                    .pointer("/choices/0/message/content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
                Err(err) => {
                    tidy_err = Some(format!("{}", err));
                    None
                }
            };
        spinner.stop();
        if let Some(err) = tidy_err {
            eprintln!("{NE} 模型整理失败（将保存原文）: {}", err);
        }
        match result {
            Some(tidied) if tidied != new_note => {
                println!("{NE} 已整理修改内容（语义未变）：");
                println!("  {FIELD}整理后:{RST} {}", tidied.chars().take(500).collect::<String>());
                tidied
            }
            _ => new_note,
        }
    };

    match crate::ai::tools::service::memory::update_memo_entry(&target, &final_note) {
        Ok(_) => {
            println!("{NE} 已更新该条目。");
            Ok(())
        }
        Err(err) => {
            eprintln!("{NE} 更新失败: {}", err);
            Err(err.into())
        }
    }
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
        if one_shot_mode && app.cli.session.is_none() {
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
                    let run = turn_runtime::run_turn(
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
        if note_search_interactive_mode(&app.cli) {
            match handle_note_search_interactive_turn(app, &question, history_count).await {
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
mod tests {
    use super::{
        DispatchOutcomeTag, ProcessDispatchMeta, SCHED_COOLDOWN_EPOCHS_DEFAULT,
        SCHEDULER_DISPATCH_META, background_execute_limit, background_pop_limit,
        build_background_process_question, build_consolidation_merge_entries,
        build_note_search_retrieval_query, decode_background_process_task_goal,
        has_pending_foreground_process, maybe_auto_route_agent, one_shot_cli_mode,
        read_recent_history, resolve_background_subagent_override,
        resolve_startup_session_choice, resolve_startup_session_choice_with_selector,
        reset_scheduler_test_state, should_preload_mcp, should_publish_subagent_task_result,
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
            agent_name: "explore".to_string(),
            model: "qwen3.7-max-alibaba".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
        })
        .unwrap();
        assert!(decode_background_process_task_goal(&encoded).unwrap().is_some());

        let corrupted = encoded.replacen('{', "", 1);
        let err = decode_background_process_task_goal(&corrupted).unwrap_err();
        assert!(err.contains("failed to decode"));
        assert!(err.contains("parent agent/model"));
    }

    #[test]
    fn background_subagent_override_requires_known_enabled_agent() {
        let mut explore = primary_agent(
            "explore",
            "Read-only codebase exploration agent",
            &["find", "search"],
        );
        explore.mode = AgentMode::Subagent;
        explore.disabled = true;
        let build = primary_agent("build", "Default agent", &["implement"]);

        let err =
            resolve_background_subagent_override(&[build.clone(), explore.clone()], Some("explore"))
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
            agent_name: "explore".to_string(),
            model: "qwen3.7-max-alibaba".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
        })
        .unwrap();
        let mailbox = vec![
            "[task_wait PARKED] ...".to_string(),
            "[mailbox] task task_123 completed".to_string(),
        ];

        let question = build_background_process_question(
            42,
            &encoded,
            Some("inspect codebase state"),
            &mailbox,
        );

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
            disable_mcp_tools: false,
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
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
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
        let persona_store = crate::ai::persona::PersonaStore::for_tests_with_path(
            std::env::temp_dir().join(format!(
                "rt_personas_conflict_{}.json",
                uuid::Uuid::new_v4().simple()
            )),
        );
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
        let persona_store = crate::ai::persona::PersonaStore::for_tests_with_path(
            std::env::temp_dir().join(format!(
                "rt_personas_clear_conflict_{}.json",
                uuid::Uuid::new_v4().simple()
            )),
        );
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
        let persona_store = crate::ai::persona::PersonaStore::for_tests_with_path(
            std::env::temp_dir().join(format!(
                "rt_personas_new_conflict_{}.json",
                uuid::Uuid::new_v4().simple()
            )),
        );
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
    fn note_search_followup_query_includes_recent_history() {
        let history = vec![
            Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String(
                    "第一条讲的是 trait object 和 dyn 的区别。".to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: serde_json::Value::String("帮我找 trait object 的笔记".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let query = build_note_search_retrieval_query("再展开第一条", &history);
        assert!(query.contains("当前问题：再展开第一条"));
        assert!(query.contains("用户: 帮我找 trait object 的笔记"));
        assert!(query.contains("助手: 第一条讲的是 trait object 和 dyn 的区别。"));

        let user_pos = query
            .find("用户: 帮我找 trait object 的笔记")
            .expect("user context should be present");
        let assistant_pos = query
            .find("助手: 第一条讲的是 trait object 和 dyn 的区别。")
            .expect("assistant context should be present");
        assert!(user_pos < assistant_pos);
    }

    #[test]
    fn consolidation_merge_plan_auto_deletes_valid_merge_ids() {
        let merge_plan = [
            serde_json::json!({
                "ids": ["id_a", "id_b"],
                "merged_content": "合并后的内容"
            }),
            serde_json::json!({
                "ids": ["ignored"],
                "merged_content": ""
            }),
        ];
        let merge_plan_refs: Vec<&serde_json::Value> = merge_plan.iter().collect();

        let (delete_ids, merged_count, new_entries) =
            build_consolidation_merge_entries(&merge_plan_refs);

        assert_eq!(merged_count, 2);
        assert_eq!(delete_ids.len(), 2);
        assert!(delete_ids.contains("id_a"));
        assert!(delete_ids.contains("id_b"));
        assert!(!delete_ids.contains("ignored"));
        assert_eq!(new_entries.len(), 1);
        assert!(
            new_entries[0]
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with("mem_"))
        );
        assert_eq!(new_entries[0].note, "合并后的内容");
        assert_eq!(new_entries[0].tags, vec!["consolidated".to_string()]);
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
            let root =
                os.begin_foreground("foreground".to_string(), "goal".to_string(), 10, 8, None);
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
            let root =
                os.begin_foreground("foreground".to_string(), "goal".to_string(), 10, 8, None);
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
        let process = PathBuf::from("/tmp/session.sqlite.proc-42");
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
        let process = PathBuf::from("/tmp/session.sqlite.proc-43");
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
        let process = PathBuf::from("/tmp/session.sqlite.proc-99");
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
}
