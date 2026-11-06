//! Background process scheduler: epoch management, dispatch scoring,
//! cooldown / circuit-breaker, and batch selection.
//!
//! Extracted from `driver/mod.rs` to establish a clear boundary between
//! the scheduler and the main run loop (review Finding #1, Phase 2).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

use rust_tools::cw::SkipMap;

use crate::ai::agents::{self, AgentManifest};
use crate::ai::config_schema::AiConfig;
use crate::ai::tools::task_tools::{decode_os_task_goal, is_encoded_task_goal};
use crate::commonw::configw;

pub(super) const BG_DISPATCH_BASE_BATCH_DEFAULT: usize = 4;
pub(super) const BG_DISPATCH_MAX_BATCH_DEFAULT: usize = 8;
pub(super) const BG_DISPATCH_EXECUTE_MAX_DEFAULT: usize = 6;
pub(super) const SCHED_FAIL_STREAK_OPEN_THRESHOLD_DEFAULT: u32 = 3;
pub(super) const SCHED_COOLDOWN_EPOCHS_DEFAULT: u64 = 6;
pub(super) const SCHED_EVAL_PERIOD_EPOCHS_DEFAULT: u64 = 24;
pub(super) const SCHED_EVAL_MIN_SAMPLES_DEFAULT: usize = 8;
pub(super) const SCHED_COST_PENALTY_DIVISOR_DEFAULT: u64 = 50_000;
pub(super) const SCHED_TOKEN_PENALTY_DIVISOR_DEFAULT: u64 = 4_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DispatchOutcomeTag {
    Advanced,
    Blocked,
    Failed,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ProcessDispatchMeta {
    pub(super) failure_streak: u32,
    pub(super) success_streak: u32,
    pub(super) cooldown_until_epoch: u64,
    pub(super) last_dispatch_epoch: u64,
    pub(super) last_outcome: DispatchOutcomeTag,
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

pub(super) static SCHEDULER_DISPATCH_META: LazyLock<Mutex<SkipMap<u64, ProcessDispatchMeta>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));
pub(super) static SCHEDULER_EPOCH: AtomicU64 = AtomicU64::new(0);
pub(super) static SCHEDULER_LAST_EVAL_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Parse a config value without trimming or clamping; accessors own their bounds.
fn parse_scheduler_config<T: std::str::FromStr>(value: Option<&str>, default: T) -> T {
    value.and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub(super) fn scheduler_cfg_usize(key: &str, default: usize) -> usize {
    if cfg!(test) {
        return default;
    }
    parse_scheduler_config(configw::get_all_config().get_opt(key).as_deref(), default)
}

pub(super) fn scheduler_cfg_u32(key: &str, default: u32) -> u32 {
    if cfg!(test) {
        return default;
    }
    parse_scheduler_config(configw::get_all_config().get_opt(key).as_deref(), default)
}

pub(super) fn scheduler_cfg_u64(key: &str, default: u64) -> u64 {
    if cfg!(test) {
        return default;
    }
    parse_scheduler_config(configw::get_all_config().get_opt(key).as_deref(), default)
}

pub(super) fn sched_base_batch() -> usize {
    scheduler_cfg_usize(
        AiConfig::SCHEDULER_BASE_BATCH,
        BG_DISPATCH_BASE_BATCH_DEFAULT,
    )
    .max(1)
}

pub(super) fn sched_max_batch() -> usize {
    scheduler_cfg_usize(AiConfig::SCHEDULER_MAX_BATCH, BG_DISPATCH_MAX_BATCH_DEFAULT)
        .max(sched_base_batch())
}

pub(super) fn sched_execute_max() -> usize {
    scheduler_cfg_usize(
        AiConfig::SCHEDULER_EXECUTE_MAX,
        BG_DISPATCH_EXECUTE_MAX_DEFAULT,
    )
    .max(sched_base_batch())
}

pub(super) fn sched_fail_threshold() -> u32 {
    scheduler_cfg_u32(
        AiConfig::SCHEDULER_FAIL_STREAK_THRESHOLD,
        SCHED_FAIL_STREAK_OPEN_THRESHOLD_DEFAULT,
    )
    .max(1)
}

pub(super) fn sched_cooldown_epochs() -> u64 {
    scheduler_cfg_u64(
        AiConfig::SCHEDULER_COOLDOWN_EPOCHS,
        SCHED_COOLDOWN_EPOCHS_DEFAULT,
    )
    .max(1)
}

pub(super) fn sched_eval_period_epochs() -> u64 {
    scheduler_cfg_u64(
        AiConfig::SCHEDULER_EVAL_PERIOD_EPOCHS,
        SCHED_EVAL_PERIOD_EPOCHS_DEFAULT,
    )
    .max(1)
}

pub(super) fn sched_eval_min_samples() -> usize {
    scheduler_cfg_usize(
        AiConfig::SCHEDULER_EVAL_MIN_SAMPLES,
        SCHED_EVAL_MIN_SAMPLES_DEFAULT,
    )
    .max(1)
}

pub(super) fn sched_cost_penalty_divisor() -> u64 {
    scheduler_cfg_u64(
        AiConfig::SCHEDULER_COST_PENALTY_DIVISOR_MICROS,
        SCHED_COST_PENALTY_DIVISOR_DEFAULT,
    )
    .max(1)
}

pub(super) fn sched_token_penalty_divisor() -> u64 {
    scheduler_cfg_u64(
        AiConfig::SCHEDULER_TOKEN_PENALTY_DIVISOR,
        SCHED_TOKEN_PENALTY_DIVISOR_DEFAULT,
    )
    .max(1)
}

pub(super) fn next_scheduler_epoch() -> u64 {
    SCHEDULER_EPOCH
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1)
}

pub(super) fn current_scheduler_epoch() -> u64 {
    SCHEDULER_EPOCH.load(Ordering::Relaxed)
}

pub(super) fn background_pop_limit(ready_count: usize) -> usize {
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

pub(super) fn background_execute_limit(ready_count: usize) -> usize {
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

pub(super) fn scheduler_score(
    proc: &aios_kernel::kernel::Process,
    meta: ProcessDispatchMeta,
    epoch: u64,
) -> i64 {
    // Higher priority (smaller priority value) + longer since last dispatch +
    // lower failure streak.
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

pub(super) fn update_dispatch_meta(
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

pub(super) fn maybe_promote_half_open(
    meta: ProcessDispatchMeta,
    epoch: u64,
) -> ProcessDispatchMeta {
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

pub(super) fn classify_process_outcome(
    os: &dyn aios_kernel::kernel::Kernel,
    pid: u64,
) -> DispatchOutcomeTag {
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

pub(super) fn should_publish_subagent_task_result(
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

pub(super) fn decode_background_process_task_goal(
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

pub(super) fn resolve_background_subagent_override<'a>(
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

pub(super) fn publish_background_task_failure(
    shared_os: &aios_kernel::SharedKernel,
    pid: u64,
    result_channel_id: Option<u64>,
    completion_futex_addr: Option<aios_kernel::primitives::FutexAddr>,
    error: &str,
) {
    publish_background_task_failure_with_io(
        shared_os,
        pid,
        result_channel_id,
        completion_futex_addr,
        error,
        || {},
        || {},
    );
}

pub(super) fn publish_background_task_failure_with_io(
    shared_os: &aios_kernel::SharedKernel,
    pid: u64,
    result_channel_id: Option<u64>,
    completion_futex_addr: Option<aios_kernel::primitives::FutexAddr>,
    error: &str,
    before_publication: impl FnOnce(),
    before_cleanup: impl FnOnce(),
) {
    super::background_dispatch::publish_background_preparation_failure_with_io(
        shared_os,
        pid,
        result_channel_id,
        completion_futex_addr,
        error,
        before_publication,
        before_cleanup,
    );
}

pub(super) fn apply_priority_handoff(
    proc: &mut aios_kernel::kernel::Process,
    outcome: DispatchOutcomeTag,
) {
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

pub(super) fn record_scheduler_outcome(
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

pub(super) fn mark_dispatched_pids(pids: &[u64], epoch: u64) {
    let mut meta_map = SCHEDULER_DISPATCH_META
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    for pid in pids {
        let mut meta = meta_map.get(pid).unwrap_or(ProcessDispatchMeta::default());
        meta.last_dispatch_epoch = epoch;
        meta_map.insert(*pid, meta);
    }
}

/// A decision captured during selection and emitted after the kernel guard closes.
pub(super) struct SchedulerDecision {
    epoch: u64,
    chosen: String,
    alternatives: Vec<String>,
    reasoning: String,
    success: bool,
}

impl SchedulerDecision {
    pub(super) fn emit(self, session_id: &str) {
        log_scheduler_decision(
            session_id,
            self.epoch,
            &self.chosen,
            self.alternatives,
            self.reasoning,
            self.success,
        );
    }
}

pub(super) fn log_scheduler_decision(
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

pub(super) fn maybe_emit_scheduler_eval(epoch: u64, session_id: &str) {
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

    // crate::ai::driver::print::print_tool_note_line(
    //     "scheduler-eval",
    //     &format!(
    //         "epoch={epoch} total={total} failing={failing} cooling={cooling} blocked={blocked}"
    //     ),
    // );
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
pub(super) fn reset_scheduler_test_state() {
    SCHEDULER_EPOCH.store(0, Ordering::Relaxed);
    if let Ok(mut map) = SCHEDULER_DISPATCH_META.lock() {
        map.clear();
    }
}

pub(super) fn select_background_batch(
    os: &mut dyn aios_kernel::kernel::Kernel,
    epoch: u64,
) -> (Vec<aios_kernel::kernel::Process>, Option<SchedulerDecision>) {
    let ready_count = os.ready_count();
    let pop_limit = background_pop_limit(ready_count);
    let execute_limit = background_execute_limit(ready_count);
    let popped = os.pop_all_ready(pop_limit);
    if popped.is_empty() {
        return (Vec::new(), None);
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
    // pop_all_ready puts processes into Running; those not executed must be
    // explicitly requeued to ready.
    for proc in deferred {
        os.set_current_pid(Some(proc.pid));
        let _ = os.requeue_current();
    }

    if let Some(first) = selected.first() {
        os.set_current_pid(Some(first.pid));
    }
    let selected_pids: Vec<u64> = selected.iter().map(|p| p.pid).collect();
    mark_dispatched_pids(&selected_pids, epoch);
    let decision = if !selected_pids.is_empty() {
        Some(SchedulerDecision {
            epoch,
            chosen: format!("selected={:?}", selected_pids),
            alternatives: alt_notes,
            reasoning: format!("eligible={} deferred={}", selected.len(), deferred_count),
            success: true,
        })
    } else {
        None
    };
    (selected, decision)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_config_parse_preserves_unsigned_values_and_fallbacks() {
        for value in [
            None,
            Some(""),
            Some("invalid"),
            Some("-1"),
            Some("1.0"),
            Some("0x10"),
            Some("1_000"),
            Some(" 17"),
            Some("17 "),
            Some("\t17\n"),
            Some(" "),
            Some("184467440737095516160000000000"),
        ] {
            assert_eq!(parse_scheduler_config(value, 11usize), 11, "{value:?}");
            assert_eq!(parse_scheduler_config(value, 13u32), 13, "{value:?}");
            assert_eq!(parse_scheduler_config(value, 19u64), 19, "{value:?}");
        }
        for (value, expected) in [("17", 17u64), ("+17", 17), ("00017", 17), ("0", 0)] {
            assert_eq!(
                parse_scheduler_config(Some(value), 11usize),
                expected as usize,
                "{value:?}"
            );
            assert_eq!(
                parse_scheduler_config(Some(value), 13u32),
                expected as u32,
                "{value:?}"
            );
            assert_eq!(
                parse_scheduler_config(Some(value), 19u64),
                expected,
                "{value:?}"
            );
        }

        let usize_max = usize::MAX.to_string();
        let usize_overflow = (usize::MAX as u128 + 1).to_string();
        assert_eq!(
            parse_scheduler_config(Some(&usize_max), 11usize),
            usize::MAX
        );
        assert_eq!(parse_scheduler_config(Some(&usize_overflow), 11usize), 11);
        assert_eq!(parse_scheduler_config(Some("4294967295"), 13u32), u32::MAX);
        assert_eq!(parse_scheduler_config(Some("4294967296"), 13u32), 13);
        assert_eq!(
            parse_scheduler_config(Some("4294967296"), 19u64),
            4_294_967_296
        );
        assert_eq!(
            parse_scheduler_config(Some("18446744073709551615"), 19u64),
            u64::MAX
        );
        assert_eq!(
            parse_scheduler_config(Some("18446744073709551616"), 19u64),
            19
        );
    }

    #[test]
    fn scheduler_config_test_readers_preserve_bypass_and_accessor_defaults() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        for default in [0usize, 17, usize::MAX] {
            assert_eq!(
                scheduler_cfg_usize(AiConfig::SCHEDULER_BASE_BATCH, default),
                default
            );
        }
        for default in [0u32, 17, u32::MAX] {
            assert_eq!(
                scheduler_cfg_u32(AiConfig::SCHEDULER_FAIL_STREAK_THRESHOLD, default),
                default
            );
        }
        for default in [0u64, 17, u64::MAX] {
            assert_eq!(
                scheduler_cfg_u64(AiConfig::SCHEDULER_COOLDOWN_EPOCHS, default),
                default
            );
        }
        assert_eq!(sched_base_batch(), BG_DISPATCH_BASE_BATCH_DEFAULT);
        assert_eq!(sched_max_batch(), BG_DISPATCH_MAX_BATCH_DEFAULT);
        assert_eq!(sched_execute_max(), BG_DISPATCH_EXECUTE_MAX_DEFAULT);
        assert_eq!(
            sched_fail_threshold(),
            SCHED_FAIL_STREAK_OPEN_THRESHOLD_DEFAULT
        );
        assert_eq!(sched_cooldown_epochs(), SCHED_COOLDOWN_EPOCHS_DEFAULT);
        assert_eq!(sched_eval_period_epochs(), SCHED_EVAL_PERIOD_EPOCHS_DEFAULT);
        assert_eq!(sched_eval_min_samples(), SCHED_EVAL_MIN_SAMPLES_DEFAULT);
        assert_eq!(
            sched_cost_penalty_divisor(),
            SCHED_COST_PENALTY_DIVISOR_DEFAULT
        );
        assert_eq!(
            sched_token_penalty_divisor(),
            SCHED_TOKEN_PENALTY_DIVISOR_DEFAULT
        );
    }

    #[test]
    fn scheduler_selection_preserves_scores_ties_requeue_order_and_metadata() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        reset_scheduler_test_state();
        let shared_os = crate::ai::driver::new_local_kernel();
        let mut os = shared_os.lock().unwrap();
        let pids = (0..11)
            .map(|index| {
                os.spawn(
                    None,
                    format!("fixture-{index}"),
                    "goal".to_string(),
                    10,
                    10,
                    None,
                    None,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let usage = &mut os.get_process_mut(pids[6]).unwrap().usage;
        usage.cost_micros = 100_000;
        usage.tokens_in = 4_000;
        usage.tokens_out = 4_000;
        let before = {
            let mut map = SCHEDULER_DISPATCH_META.lock().unwrap();
            map.insert(
                pids[2],
                ProcessDispatchMeta {
                    failure_streak: 3,
                    cooldown_until_epoch: 30,
                    last_dispatch_epoch: 4,
                    last_outcome: DispatchOutcomeTag::Failed,
                    ..ProcessDispatchMeta::default()
                },
            );
            map.insert(
                pids[3],
                ProcessDispatchMeta {
                    failure_streak: 3,
                    cooldown_until_epoch: 20,
                    last_dispatch_epoch: 8,
                    last_outcome: DispatchOutcomeTag::Failed,
                    ..ProcessDispatchMeta::default()
                },
            );
            map.insert(
                pids[4],
                ProcessDispatchMeta {
                    last_dispatch_epoch: 19,
                    ..ProcessDispatchMeta::default()
                },
            );
            map.insert(
                pids[5],
                ProcessDispatchMeta {
                    failure_streak: 2,
                    last_outcome: DispatchOutcomeTag::Blocked,
                    ..ProcessDispatchMeta::default()
                },
            );
            map.clone()
        };

        let session_id = "scheduler-selection-regression";
        let (selected, decision) = select_background_batch(os.as_mut(), 20);
        drop(os);
        decision
            .expect("selected processes must produce a decision")
            .emit(session_id);
        let mut os = shared_os.lock().unwrap();
        let expected_selected = [pids[0], pids[1], pids[7], pids[6], pids[4], pids[5]];
        assert_eq!(
            selected.iter().map(|proc| proc.pid).collect::<Vec<_>>(),
            expected_selected
        );
        assert_eq!(os.current_process_id(), Some(pids[0]));
        for pid in &pids {
            let proc = os.get_process(*pid).unwrap();
            assert_eq!(
                proc.state,
                if expected_selected.contains(pid) {
                    aios_kernel::kernel::ProcessState::Running
                } else {
                    aios_kernel::kernel::ProcessState::Ready
                }
            );
            assert_eq!(proc.priority, 10);
            assert_eq!(proc.quota_turns, 10);
        }
        let logs = crate::ai::driver::decision_log::get_decision_log_store()
            .recent_by_session(session_id, 1);
        assert_eq!(logs.len(), 1);
        let decision = &logs[0];
        assert_eq!(decision.turn_id, 20);
        assert_eq!(
            decision.chosen_option,
            format!("selected={expected_selected:?}")
        );
        assert_eq!(decision.reasoning, "eligible=6 deferred=2");
        assert!(decision.outcome.as_ref().unwrap().success);
        assert_eq!(
            decision.alternatives_considered,
            [
                (0, 1072, 0, 0),
                (1, 1072, 0, 0),
                (2, 908, 30, 3),
                (3, 936, 0, 2),
                (4, 1015, 0, 0),
                (5, 986, 0, 2),
                (6, 1068, 0, 0),
                (7, 1072, 0, 0),
            ]
            .map(|(index, score, cooldown, failures)| format!(
                "pid={} score={} cooldown_until={} fail_streak={}",
                pids[index], score, cooldown, failures
            ))
        );
        {
            let map = SCHEDULER_DISPATCH_META.lock().unwrap();
            for pid in &pids {
                let previous = before.get(pid).unwrap_or_default();
                let current = map.get(pid).unwrap_or_default();
                assert_eq!(
                    current.last_dispatch_epoch,
                    if expected_selected.contains(pid) {
                        20
                    } else {
                        previous.last_dispatch_epoch
                    }
                );
                assert_eq!(current.failure_streak, previous.failure_streak);
                assert_eq!(current.success_streak, previous.success_streak);
                assert_eq!(current.cooldown_until_epoch, previous.cooldown_until_epoch);
                assert_eq!(current.last_outcome, previous.last_outcome);
                assert_eq!(
                    map.contains_key(pid),
                    expected_selected.contains(pid) || before.contains_key(pid)
                );
            }
        }
        assert_eq!(
            os.pop_all_ready(usize::MAX)
                .iter()
                .map(|proc| proc.pid)
                .collect::<Vec<_>>(),
            [pids[8], pids[9], pids[10], pids[2], pids[3]]
        );
        reset_scheduler_test_state();
    }

    #[test]
    fn scheduler_selection_skips_empty_and_cooling_batches() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let session_id = "scheduler-selection-no-decision";
        for count in [0, 3] {
            reset_scheduler_test_state();
            let shared_os = crate::ai::driver::new_local_kernel();
            let mut os = shared_os.lock().unwrap();
            let foreground =
                os.begin_foreground("fg".to_string(), "goal".to_string(), 10, 10, None);
            let pids = (0..count)
                .map(|index| {
                    os.spawn(
                        Some(foreground),
                        format!("cooling-{index}"),
                        "goal".to_string(),
                        10,
                        10,
                        None,
                        None,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            {
                let mut map = SCHEDULER_DISPATCH_META.lock().unwrap();
                for pid in &pids {
                    map.insert(
                        *pid,
                        ProcessDispatchMeta {
                            failure_streak: 3,
                            cooldown_until_epoch: 20,
                            last_dispatch_epoch: 2,
                            last_outcome: DispatchOutcomeTag::Failed,
                            ..ProcessDispatchMeta::default()
                        },
                    );
                }
            }
            let (selected, decision) = select_background_batch(os.as_mut(), 10);
            assert!(selected.is_empty());
            assert!(decision.is_none());
            assert_eq!(
                os.current_process_id(),
                if count == 0 { Some(foreground) } else { None }
            );
            assert_eq!(os.ready_count(), count);
            assert_eq!(
                os.pop_all_ready(usize::MAX)
                    .iter()
                    .map(|proc| proc.pid)
                    .collect::<Vec<_>>(),
                pids
            );
            let map = SCHEDULER_DISPATCH_META.lock().unwrap();
            assert_eq!(map.len(), count);
            for meta in map.values() {
                assert_eq!(meta.last_dispatch_epoch, 2);
                assert_eq!(meta.failure_streak, 3);
                assert_eq!(meta.cooldown_until_epoch, 20);
            }
        }
        assert!(
            crate::ai::driver::decision_log::get_decision_log_store()
                .recent_by_session(session_id, 1)
                .is_empty()
        );
        reset_scheduler_test_state();
    }
}
