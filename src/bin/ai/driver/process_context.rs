//! Process context helpers: history path generation, background subagent
//! context resolution, wake-up prompt construction, and turn quota finalization.
//!
//! Extracted from `driver/mod.rs` (review Finding #1, Phase 2).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aios_kernel::primitives::{ResourceUsageDelta, RlimitVerdict};

use crate::ai::skills::SkillManifest;

const RESUME_SKILL_NAMES_ENV: &str = "__AIOS_INTERNAL_RESUME_SKILL_NAMES";

/// Store the skill set that belongs to a suspended process. A resumed turn is
/// still the same logical turn, so rebuilding it with the default core tools
/// would remove tools supplied by the active skill (for example `task_wait`).
pub(super) fn store_resume_skill_names(
    process: &mut aios_kernel::kernel::Process,
    skill_names: &[String],
) {
    if skill_names.is_empty() {
        process.env.remove(RESUME_SKILL_NAMES_ENV);
        return;
    }
    if let Ok(encoded) = serde_json::to_string(skill_names) {
        process
            .env
            .insert(RESUME_SKILL_NAMES_ENV.to_string(), encoded);
    }
}

/// Recover the skill set saved immediately before this process suspended.
pub(super) fn resume_skill_names(process: &aios_kernel::kernel::Process) -> Vec<String> {
    process
        .env
        .get(RESUME_SKILL_NAMES_ENV)
        .and_then(|encoded| serde_json::from_str(encoded).ok())
        .unwrap_or_default()
}

pub(in crate::ai::driver) fn history_path_with_suffix(base: &Path, suffix: &str) -> PathBuf {
    let file_name = match base.file_stem() {
        Some(stem) => {
            let mut name = stem.to_os_string();
            name.push(suffix);
            if let Some(extension) = base.extension() {
                name.push(".");
                name.push(extension);
            }
            name
        }
        None => {
            let mut name = std::ffi::OsString::from("session");
            name.push(suffix);
            name
        }
    };
    base.with_file_name(file_name)
}

/// Generate history file path for a background process.
/// Inserts `.proc-{pid}` before the final extension so SQLite history files
/// stay recognizable as SQLite by extension-based dispatch.
pub(crate) fn process_history_path(base: &Path, pid: u64) -> PathBuf {
    history_path_with_suffix(base, &format!(".proc-{pid}"))
}

pub(super) fn resolve_background_subagent_context(
    history_path: PathBuf,
    original_history_file: &Path,
    skill_manifests: &Arc<Vec<SkillManifest>>,
    task_id: Option<&str>,
    inherit: crate::ai::tools::task_tools::InheritOptions,
    initialize_history: bool,
) -> Result<(PathBuf, Arc<Vec<SkillManifest>>), String> {
    let is_task_subagent = task_id.is_some();
    // 是否首次派发由 Process.history_file 决定，不能由 mailbox 推断：SIGCONT 唤醒
    // 不会写 mailbox，但仍必须复用 child 库，避免覆盖此前的执行历史。
    crate::ai::history::prepare_subagent_history(
        original_history_file,
        &history_path,
        is_task_subagent && inherit.history,
        initialize_history,
    )
    .map_err(|err| format!("准备子代理历史失败：{err}"))?;
    let effective_history = history_path;
    let effective_skills = if is_task_subagent && !inherit.skills {
        Arc::new(Vec::new())
    } else {
        skill_manifests.clone()
    };
    Ok((effective_history, effective_skills))
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

/// Build the shared foreground/background wake-up prompt. Tool-specific
/// handling rules are added after the resumed turn's schema is resolved.
pub(super) fn format_wakeup_prompt(pid: u64, goal: &str, messages: &[String]) -> String {
    if messages.len() == 1
        && crate::ai::tools::task_tools::is_agent_team_wait_deadline_message(&messages[0])
    {
        messages[0].clone()
    } else {
        format!(
            "[Process {} Woke Up] Original goal: {}\nNew mailbox messages:\n{}\n\nWake-up handling rules:\n- Call only tools present in the current schema. Mailbox tool recommendations are conditional on those tools being available.\n- If a required tool is missing, use an available tool-loading facility to restore access; if none is available, report the missing capability.\n- A cooperative scheduler yield is normal waiting, not a stall. Preserve unfinished subagent tasks.\n- Prefer continuing reasoning immediately when the wake-up messages already identify the relevant finished tasks.\n\nResume execution based on the goal and these messages.",
            pid,
            goal,
            messages.join("\n---\n")
        )
    }
}

/// Render task wake-up instructions from the schema actually sent to the model.
pub(super) fn wakeup_task_guidance(has_tool: impl Fn(&str) -> bool) -> Option<String> {
    let mut lines = Vec::new();
    if has_tool("task_wait") {
        lines.push(
            "- If the mailbox indicates event wake-up after a parked task wait, immediately re-call `task_wait` with the same task_ids and wait_policy to collect results from task result channels. Its timeout_secs is a per-call wait budget, not a stall signal.",
        );
    }
    if has_tool("task_status") {
        lines.push("- Use `task_status` for a non-blocking snapshot and to collect ready results.");
    }
    if has_tool("task_integrate") {
        lines.push(
            "- After a task result is delivered, call `task_integrate` with the task_id, disposition, and parent conclusion; delivery alone does not close the evidence lifecycle.",
        );
    }
    if lines.is_empty() {
        return None;
    }
    lines.push("- Cancellation alone does not collect a terminal result.");
    Some(lines.join("\n"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use aios_kernel::kernel::{KernelInternal, Syscall};

    #[test]
    fn agent_team_quiet_deadline_wakeup_omits_goal_and_guidance() {
        let notice = "[task-wait] still running: task-a(idle=12s,phase=read)";
        assert_eq!(
            format_wakeup_prompt(1, "original goal", &[notice.to_string()]),
            notice
        );
        for messages in [
            vec!["[EVENT_WAKE] terminal task event".to_string()],
            vec![notice.to_string(), "terminal error".to_string()],
        ] {
            let prompt = format_wakeup_prompt(1, "original goal", &messages);
            assert!(prompt.contains("Original goal: original goal"));
            assert!(prompt.contains("Wake-up handling rules:"));
            assert!(prompt.contains(messages.last().unwrap()));
        }
    }

    #[test]
    fn process_history_path_preserves_sqlite_extension() {
        let got = process_history_path(Path::new("/tmp/session.sqlite"), 42);

        assert_eq!(got, PathBuf::from("/tmp/session.proc-42.sqlite"));
    }

    #[test]
    fn history_path_with_suffix_preserves_db_extension() {
        let got = history_path_with_suffix(Path::new("/tmp/session.db"), ".proc-7");

        assert_eq!(got, PathBuf::from("/tmp/session.proc-7.db"));
    }

    #[test]
    fn suspended_process_skill_names_round_trip() {
        let mut os = aios_kernel::local::LocalOS::new();
        let pid = os.begin_foreground("foreground".to_string(), "goal".to_string(), 2, 8, None);
        let expected = vec!["agent-team".to_string(), "audit".to_string()];

        store_resume_skill_names(os.get_process_mut(pid).unwrap(), &expected);
        assert_eq!(resume_skill_names(os.get_process(pid).unwrap()), expected);

        store_resume_skill_names(os.get_process_mut(pid).unwrap(), &[]);
        assert!(resume_skill_names(os.get_process(pid).unwrap()).is_empty());
    }

    #[test]
    fn wakeup_task_guidance_only_names_available_tools() {
        let tool_names = ["task_wait", "task_status", "task_integrate"];
        for mask in 0..8 {
            let has_tool = |name: &str| {
                tool_names
                    .iter()
                    .enumerate()
                    .any(|(index, tool)| *tool == name && mask & (1 << index) != 0)
            };
            let guidance = wakeup_task_guidance(has_tool);
            assert_eq!(guidance.is_none(), mask == 0);
            let guidance = guidance.unwrap_or_default();
            for tool in tool_names {
                assert_eq!(guidance.contains(&format!("`{tool}`")), has_tool(tool));
            }
            assert!(!guidance.contains("`task_spawn`"));
            assert!(!guidance.contains("`task_cancel`"));
        }
    }

    #[test]
    fn wakeup_prompt_preserves_mailbox_without_unconditional_tool_demands() {
        let mailbox = "[EVENT_WAKE] Recommended next action: re-call task_wait.";
        let prompt = format_wakeup_prompt(42, "collect the result", &[mailbox.to_string()]);

        assert!(prompt.contains("[Process 42 Woke Up] Original goal: collect the result"));
        assert!(prompt.contains(mailbox));
        assert_eq!(prompt.matches("task_wait").count(), 1);
        assert!(prompt.contains("Call only tools present in the current schema."));
        assert!(!prompt.contains("`task_integrate`"));
    }
}
