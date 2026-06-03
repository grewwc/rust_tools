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
        Arc, LazyLock, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use aios_kernel::primitives::{ResourceUsageDelta, RlimitDim, RlimitVerdict};
use rust_tools::cw::SkipMap;
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
    tools::task_tools::decode_os_task_goal,
    types::{AgentContext, App},
};
use crate::commonw::configw;

pub mod agent_router;
pub mod commands;
pub mod decision_log;
pub mod embedding;
pub mod hooks;
pub mod input;
pub mod intent_model;
pub mod intent_recognition;
pub mod mcp_init;
pub mod model;
pub mod observer;
pub mod params;
pub mod print;
pub mod reflection;
pub mod runtime_ctx;
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

tokio::task_local! {
    static TASK_PID: Option<u64>;
}

fn current_task_pid() -> Option<u64> {
    TASK_PID.try_with(|v| *v).unwrap_or(None)
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
const DEFAULT_MAX_ITERATIONS: usize = 1024;

/// Max iterations for subagent (executor) processes
const EXECUTOR_MAX_ITERATIONS: usize = 64;

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
    scheduler_cfg_usize("ai.scheduler.base_batch", BG_DISPATCH_BASE_BATCH_DEFAULT).max(1)
}

fn sched_max_batch() -> usize {
    scheduler_cfg_usize("ai.scheduler.max_batch", BG_DISPATCH_MAX_BATCH_DEFAULT)
        .max(sched_base_batch())
}

fn sched_execute_max() -> usize {
    scheduler_cfg_usize("ai.scheduler.execute_max", BG_DISPATCH_EXECUTE_MAX_DEFAULT)
        .max(sched_base_batch())
}

fn sched_fail_threshold() -> u32 {
    scheduler_cfg_u32(
        "ai.scheduler.fail_streak_threshold",
        SCHED_FAIL_STREAK_OPEN_THRESHOLD_DEFAULT,
    )
    .max(1)
}

fn sched_cooldown_epochs() -> u64 {
    scheduler_cfg_u64(
        "ai.scheduler.cooldown_epochs",
        SCHED_COOLDOWN_EPOCHS_DEFAULT,
    )
    .max(1)
}

fn sched_eval_period_epochs() -> u64 {
    scheduler_cfg_u64(
        "ai.scheduler.eval_period_epochs",
        SCHED_EVAL_PERIOD_EPOCHS_DEFAULT,
    )
    .max(1)
}

fn sched_eval_min_samples() -> usize {
    scheduler_cfg_usize(
        "ai.scheduler.eval_min_samples",
        SCHED_EVAL_MIN_SAMPLES_DEFAULT,
    )
    .max(1)
}

fn sched_cost_penalty_divisor() -> u64 {
    scheduler_cfg_u64(
        "ai.scheduler.cost_penalty_divisor_micros",
        SCHED_COST_PENALTY_DIVISOR_DEFAULT,
    )
    .max(1)
}

fn sched_token_penalty_divisor() -> u64 {
    scheduler_cfg_u64(
        "ai.scheduler.token_penalty_divisor",
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
    let report = match task.await {
        Ok(Some(prepared)) => apply_prepared_mcp_with_shared_client(app, mcp_client, prepared),
        Ok(None) => return,
        Err(err) => {
            if app.shutdown.load(Ordering::Relaxed) || signal::request_interrupt_ready() {
                return;
            }
            eprintln!("[mcp] background preload task failed: {}", err);
            let fallback =
                prepare_mcp_initialization_from_path(mcp_probe.config_path.clone()).await;
            apply_prepared_mcp_with_shared_client(app, mcp_client, fallback)
        }
    };

    *mcp_initialized = true;
    if report.loaded {
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

fn should_preload_mcp(one_shot_mode: bool, mcp_probe: &McpConfigProbe) -> bool {
    mcp_probe.exists && !one_shot_mode
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
    aios_kernel::kernel::register_current_pid_provider(current_task_pid);

    let mut cli = cli::parse_cli_args(std::env::args());
    if let Err(err) = models::ensure_models_available() {
        return Err(err.into());
    }
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

    let writer =
        config::open_output_writer(cli.out.as_deref())?.map(|f| Arc::new(std::sync::Mutex::new(f)));
    let current_model = models::initial_model(&cli);
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;
    let prompt_editor = if cli.args.is_empty() {
        Some(PromptEditor::new(
            &session_id,
            config.history_file.as_path(),
        ))
    } else {
        None
    };

    // 处理 --clipboard：把当前剪贴板内容拼接到首轮提问头部
    if cli.clipboard {
        let clip = crate::clipboardw::string_content::get_clipboard_content();
        if !clip.trim().is_empty() {
            if cli.args.is_empty() {
                cli.args.push(clip);
            } else {
                let original = std::mem::take(&mut cli.args);
                let combined = format!("{}\n\n{}", clip, original.join(" "));
                cli.args.push(combined);
            }
        } else {
            eprintln!("[clipboard] 剪贴板为空，已忽略 --clipboard");
        }
    }

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

    // 处理 --note-delete / -nd：输入一段话，模型自动匹配知识库条目，确认后删除。
    if let Some(query) = app.cli.note_delete.clone() {
        return handle_note_delete(&mut app, &query).await;
    }

    // 处理 --note / -n：快速保存 memo 到知识库并退出。
    // 即使没有文本（只想保存剪贴板图片），只要传了 -n 也要进入保存流程。
    if app.cli.note_flag {
        return handle_note_save(&mut app).await;
    }

    // 处理 --memo-search / -ms：只从知识库中检索 memo，不调用 LLM / 任何工具，
    // 直接打印结果并退出。
    if app.cli.memo_search {
        return handle_memo_search(&app).await;
    }

    let decision_log_path = app
        .session_history_file
        .with_extension("decision-log.jsonl");
    crate::ai::driver::decision_log::set_decision_log_persist_path(decision_log_path);

    let mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));

    let mcp_probe = probe_mcp_config(&app);
    if app.cli.list_mcp_tools {
        let mcp_report = init_mcp(
            &mut app,
            &mut mcp_client.lock().unwrap_or_else(|err| err.into_inner()),
        )
        .await;
        print::print_mcp_tools(&mcp_report, &mcp_client.lock().unwrap());
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
    use image::{ImageBuffer, Rgb, Rgba};
    use image::buffer::ConvertBuffer;
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
        match crate::ai::request::do_request_json(app, &model, &messages, false).await {
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
        match crate::ai::request::do_request_json(app, &model, &messages, false).await {
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
                println!("[note] Image content saved to knowledge base [memo] (image: {}):", image_path);
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

/// 处理 --memo-search / -ms：从知识库中检索 memo 类条目，再用模型根据检索到的
/// 内容总结、回答用户的问题（而不是直接堆砌原始条目）。
async fn handle_memo_search(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    let query = app.cli.args.join(" ");
    let query = query.trim().to_string();
    if query.is_empty() {
        eprintln!("[memo-search] 用法: a -ms <查询内容>");
        return Err("memo-search requires a query".into());
    }

    // 检索相关 memo 条目作为上下文。
    let candidates = match crate::ai::tools::service::memory::search_memo_candidates(&query, 20) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("[memo-search] 检索失败: {}", err);
            return Err(err.into());
        }
    };
    if candidates.is_empty() {
        crate::ai::stream::render_markdown_block(&format!(
            "没有在知识库中找到与「{}」相关的内容。",
            query
        ))
        .ok();
        return Ok(());
    }

    // 把检索到的条目作为上下文，让模型基于这些内容回答用户的问题。
    let mut context = String::new();
    for (idx, e) in candidates.iter().enumerate() {
        context.push_str(&format!("[{}] {}\n", idx + 1, e.note));
    }

    let model = crate::ai::models::initial_model(&app.cli);
    let messages = vec![
        serde_json::json!({
            "role": "system",
            "content": "你是一个知识库问答助手。下面会给出用户的问题，以及从用户私人知识库检索到的若干条相关笔记。\
                        请仅基于这些笔记的内容，直接、简洁地回答用户的问题，必要时用要点组织。\
                        如果笔记中没有足够信息回答，就如实说明。用中文回答，使用 Markdown 格式。",
        }),
        serde_json::json!({
            "role": "user",
            "content": format!("问题：{}\n\n知识库检索结果：\n{}", query, context),
        }),
    ];

    match crate::ai::request::do_request_json(app, &model, &messages, false).await {
        Ok(response) => {
            let answer = response
                .pointer("/choices/0/message/content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if answer.is_empty() {
                // 模型无输出时退回展示原始条目。
                let raw = crate::ai::tools::service::memory::search_memo_candidates(&query, 20)
                    .map(|cands| {
                        cands
                            .iter()
                            .enumerate()
                            .map(|(i, e)| format!("{}. {}", i + 1, e.note))
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    })
                    .unwrap_or_default();
                crate::ai::stream::render_markdown_block(&raw).ok();
            } else {
                crate::ai::stream::render_markdown_block(&answer).ok();
            }
            Ok(())
        }
        Err(err) => {
            eprintln!("[memo-search] 总结失败: {}", err);
            Err(err.into())
        }
    }
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
        println!("[note-delete] 没有找到与「{}」相关的可删除 memo 条目。", query);
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

    let chosen = match crate::ai::request::do_request_json(app, &model, &messages, false).await {
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
        let mut flush = |num: &mut String, out: &mut Vec<usize>| {
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
        println!("[note-delete] 模型未能从候选中确定要删除的条目，已取消。可换个更具体的描述重试。");
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
    print!(
        "\n请输入要删除的编号（如 1,3；输入 all 删除全部，直接回车=全部，n=取消）: "
    );
    std::io::stdout().flush().ok();

    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).ok();
    let answer = answer.trim().to_lowercase();

    // 解析用户选择，得到最终要删除的 targets 子集。
    let selected: Vec<&crate::ai::tools::storage::memory_store::AgentMemoryEntry> =
        if answer.is_empty() || answer == "y" || answer == "yes" || answer == "all" || answer == "a"
        {
            targets.clone()
        } else if answer == "n" || answer == "no" || answer == "q" || answer == "cancel" {
            println!("[note-delete] 已取消，未删除任何内容。");
            return Ok(());
        } else {
            // 解析编号列表（针对上面列出的 1..=targets.len()）。
            let mut picks: Vec<usize> = Vec::new();
            let mut num = String::new();
            let mut flush = |num: &mut String, out: &mut Vec<usize>| {
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
                eprintln!("[note-delete] 删除失败 (时间 {}): {}", target.timestamp, err);
            }
        }
    }
    println!("[note-delete] 完成：已删除 {} 条，失败 {} 条。", deleted, failed);
    if failed > 0 && deleted == 0 {
        return Err("all deletions failed".into());
    }
    Ok(())
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

/// 构造进程被唤醒（mailbox 非空）时的 wake-up prompt。
/// foreground / background 路径共享同一段 prompt，避免双份硬编码漂移。
fn format_wakeup_prompt(pid: u64, goal: &str, messages: &[String]) -> String {
    format!(
        "[Process {} Woke Up] Original goal: {}\nNew mailbox messages:\n{}\n\nWake-up handling rules:\n- The async machinery has TWO families with similar but distinct semantics:\n    * subagent tasks  — `task_spawn` / `task_wait` / `task_status` (no cancel; long-lived task_id; task_wait's `timeout_secs` is a per-call wait budget, NOT a stall signal — re-call task_wait or pass wait_policy=\"any\" to keep waiting)\n    * generic tools   — `tool_spawn` / `tool_wait` / `tool_status` / `tool_cancel`\n  Pick the family that matches what you actually spawned; do NOT call task_cancel (it does not exist) or tool_wait on a task_id.\n- If the mailbox indicates async wake-up, first decide whether you need `task_status` / `tool_status` for a snapshot, `task_wait` / `tool_wait` to collect newly finished results, or `tool_cancel` to stop low-value branches.\n- Do not blindly wait again if enough completed results already support the answer.\n- Prefer continuing reasoning immediately when the wake-up messages already identify the relevant finished tasks.\n\nResume execution based on the goal and these messages.",
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

    let turn_outcome = runtime_ctx::DRIVER_CTX
        .scope(
            driver_ctx,
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
/// one_shot_mode: When CLI args provided (non-interactive)
///   - runs once and exits
///   - deletes session after completion
async fn run_loop(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    mcp_probe: McpConfigProbe,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let one_shot_mode = !app.cli.args.is_empty();
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
            )> = Vec::new();
            for proc in &background_procs {
                let pid = proc.pid;
                let task_goal = decode_os_task_goal(&proc.goal);
                let proc_question = if let Some(goal) = &task_goal {
                    goal.prompt.clone()
                } else if !proc.mailbox.is_empty() {
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
            ) in task_specs
            {
                let mut task_app = app.clone();
                crate::ai::types::clear_stream_cancel(&task_app);
                let task_mcp = mcp_client.clone();
                let task_os = app.os.clone();
                if let Some(agent_name) = &agent_override
                    && let Some(agent) = agents::find_agent_by_name(agent_manifests, agent_name)
                    && !agent.disabled
                {
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
                    std::sync::Arc::new(std::sync::Mutex::new(None));
                let result_slot_for_scope = result_slot_for_payload.clone();

                let inner_fut = TASK_PID.scope(Some(pid), async move {
                    crate::ai::tools::registry::common::clear_tool_cancel();
                    let result = turn_runtime::run_turn(
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
                    )
                    .await
                    .map_err(|e| format!("{}", e));
                    let mut os = task_os.lock().unwrap();
                    os.set_current_pid(Some(pid));
                    if let Some(result_channel_id) = result_channel_id {
                        let captured_output = result_slot_for_payload
                            .lock()
                            .ok()
                            .and_then(|guard| guard.clone())
                            .unwrap_or_default();
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
                    if let Some(addr) = completion_futex_addr {
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
                    //
                    // 主 memory 路径在父任务作用域内解析，这里 task_local 还没装上
                    // SUBAGENT_MEMORY_PATH，所以 from_env_or_config 拿到的是主路径。
                    let main_path =
                        crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config()
                            .path()
                            .to_path_buf();
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

                tokio::spawn(runtime_ctx::DRIVER_CTX.scope(task_driver_ctx, wrapped));
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

        if try_handle_interactive_command(app, mcp_client, &question, agent_manifests)? {
            if handle_post_command(app, &mut should_quit) {
                return Ok(());
            }
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
        let original_writer = app.writer.clone();

        crate::ai::types::clear_stream_cancel(app);
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

        let driver_ctx = runtime_ctx::DriverContext::new(
            app.clone(),
            mcp_client.clone(),
            skill_manifests.clone(),
            agent_manifests.clone(),
        );

        hooks::run_lifecycle_hook(hooks::HookEvent::TurnStart, None, None);

        let turn_outcome = runtime_ctx::DRIVER_CTX
            .scope(
                driver_ctx,
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
        if let Some(writer) = app.writer.as_ref() {
            let mut guard = writer.lock().unwrap();
            guard.write_all(b"\n---\n")?;
            guard.flush()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DispatchOutcomeTag, ProcessDispatchMeta, SCHED_COOLDOWN_EPOCHS_DEFAULT,
        SCHEDULER_DISPATCH_META, background_execute_limit, background_pop_limit,
        has_pending_foreground_process, maybe_auto_route_agent, read_recent_history,
        reset_scheduler_test_state, should_preload_mcp, update_dispatch_meta,
    };
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
    use crate::ai::cli::ParsedCli;
    use crate::ai::history::{Message, append_history_messages};
    use crate::ai::skills::SkillManifest;
    use crate::ai::types::{AgentContext, App, AppConfig};
    use aios_kernel::kernel::{EventId, ProcessState, WaitPolicy};
    use aios_kernel::primitives::ResourceLimit;
    use crate::ai::tools::task_tools::InheritOptions;
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

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
    fn finalize_turn_quota_charges_turn_usage_once() {
        let mut app = test_app("build");
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
        let mut app = test_app("build");
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
            observers: vec![Box::new(
                crate::ai::driver::thinking::ThinkingOrchestrator::new(),
            )],
        }
    }

    #[test]
    fn one_shot_mode_does_not_start_background_mcp_preload() {
        let probe = super::McpConfigProbe {
            config_path: "/tmp/mcp.json".to_string(),
            exists: true,
            server_count: 1,
        };

        assert!(!should_preload_mcp(true, &probe));
        assert!(should_preload_mcp(false, &probe));
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
        let mut app = test_app("build");

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
        let mut app = test_app("build");

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
