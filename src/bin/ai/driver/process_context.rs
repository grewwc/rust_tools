//! Process context helpers: history path generation, background subagent
//! context resolution, wake-up prompt construction, and turn quota finalization.
//!
//! Extracted from `driver/mod.rs` (review Finding #1, Phase 2).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aios_kernel::primitives::{ResourceUsageDelta, RlimitVerdict};

use crate::ai::skills::SkillManifest;

/// Generate history file path for a background process.
/// appends .proc-{pid} to the session history filename.
pub(super) fn process_history_path(base: &Path, pid: u64) -> PathBuf {
    let file_name = base
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{name}.proc-{pid}"))
        .unwrap_or_else(|| format!("session.proc-{pid}"));
    base.with_file_name(file_name)
}

pub(super) fn resolve_background_subagent_context(
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

pub(super) fn build_background_process_question(
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
pub(super) fn format_wakeup_prompt(pid: u64, goal: &str, messages: &[String]) -> String {
    format!(
        "[Process {} Woke Up] Original goal: {}\nNew mailbox messages:\n{}\n\nWake-up handling rules:\n- The async machinery has TWO families with similar but distinct semantics:\n    * subagent tasks  — `task_spawn` / `task_wait` / `task_status` / `task_cancel` (long-lived task_id; `task_wait.timeout_secs` is a per-call wait budget, NOT a stall signal; `task_cancel` terminates the subagent but you still need `task_wait` or `task_status` to collect the terminal cancelled result)\n    * generic tools   — `tool_spawn` / `tool_wait` / `tool_status` / `tool_cancel`\n  Pick the family that matches what you actually spawned; do NOT call tool_wait/tool_cancel on a subagent task_id.\n- If the mailbox indicates event wake-up after `task_wait PARKED`, immediately re-call `task_wait` with the same task_ids and wait_policy to collect results from task result channels. Use `task_status` only if you need a non-blocking snapshot.\n- If you used `task_cancel`, still collect the cancelled terminal result with `task_wait` or `task_status`; cancellation is not the same thing as collection.\n- If the mailbox indicates generic async tool wake-up, use `tool_status` / `tool_wait` / `tool_cancel` as appropriate.\n- Do not abandon parked subagent tasks as stuck merely because the previous `task_wait` returned PARKED.\n- Prefer continuing reasoning immediately when the wake-up messages already identify the relevant finished tasks.\n\nResume execution based on the goal and these messages.",
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
pub(super) fn finalize_turn_quota(
    os: &mut dyn aios_kernel::kernel::Kernel,
    pid: u64,
) -> (bool, String) {
    let verdict = os.rusage_charge(
        pid,
        ResourceUsageDelta {
            turns: 1,
            ..Default::default()
        },
    );
    let mut should_terminate = true;
    let mut termination_result = super::format_rlimit_termination_result(verdict.clone());
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
                super::format_rlimit_termination_result(verdict)
            ));
        }
        termination_result = "Completed".to_string();
    }
    (should_terminate, termination_result)
}
