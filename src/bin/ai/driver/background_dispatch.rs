//! Background process dispatch: select ready processes, decode task goals,
//! build questions, clone app context, and spawn each as a tokio task.
//!
//! Extracted from `driver/mod.rs` `run_loop` to establish a clear boundary
//! between background dispatch and the foreground interaction loop
//! (review Finding #1, Phase 2).

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ai::{
    agents::AgentManifest,
    mcp::SharedMcpClient,
    skills::SkillManifest,
    tools::task_tools::{capped_subagent_manifest, with_task_entry_by_pid},
    types::App,
};

use super::agent_routing::{activate_primary_agent, ensure_runtime_manifests_loaded};
use super::process_context::{
    build_background_process_question, finalize_turn_quota, process_history_path,
    resolve_background_subagent_context,
};
use super::runtime_ctx;
use super::scheduler::{
    DispatchOutcomeTag, classify_process_outcome, decode_background_process_task_goal,
    maybe_emit_scheduler_eval, publish_background_task_failure, record_scheduler_outcome,
    resolve_background_subagent_override, select_background_batch,
    should_publish_subagent_task_result,
};
use super::turn_runtime;
use super::{BgSubagentGuard, TASK_PID, terminate_and_cleanup};

const MAX_SUBAGENT_STATUS_DETAILS: usize = 3;

/// 后台 subagent 的历史只在进程仍可继续调度时保留。正常终止、失败、panic 或
/// `task_cancel` 导致 future 被 abort 时都会通过 Drop 清理。私有 memory 文件与
/// 独占 cwd scratch 目录与 history 同生命周期，一并在此回收，避免长跑堆积。
struct BackgroundSubagentHistoryGuard {
    path: Option<PathBuf>,
    memory_path: Option<PathBuf>,
    cwd_dir: Option<PathBuf>,
}

impl BackgroundSubagentHistoryGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path: Some(path),
            memory_path: None,
            cwd_dir: None,
        }
    }

    /// 登记随任务派生、需与 history 同生命周期回收的私有 memory 文件与独占 cwd
    /// scratch 目录。路径是确定性拼接，可在构造点一次算好传入。
    fn with_scoped_artifacts(
        mut self,
        memory_path: Option<PathBuf>,
        cwd_dir: Option<PathBuf>,
    ) -> Self {
        self.memory_path = memory_path;
        self.cwd_dir = cwd_dir;
        self
    }

    fn preserve(&mut self) {
        self.path = None;
        self.memory_path = None;
        self.cwd_dir = None;
    }

    fn preserve_memory(&mut self) {
        self.memory_path = None;
    }
}

impl Drop for BackgroundSubagentHistoryGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = crate::ai::history::delete_subagent_history(&path);
        }
        if let Some(memory_path) = self.memory_path.take() {
            let _ = crate::ai::history::delete_subagent_memory(&memory_path);
        }
        if let Some(cwd_dir) = self.cwd_dir.take() {
            let _ = std::fs::remove_dir_all(&cwd_dir);
        }
    }
}

/// 前台唯一的 subagent 状态展示。后台任务保持静默，只在调度循环的安全点刷新
/// 一条紧凑状态行，避免并发正文或多行 ANSI 重绘打乱 terminal。
pub(super) struct SubagentStatusLine {
    last_line: Option<String>,
    is_tty: bool,
}

/// 状态栏字段必须保持单行且不能携带终端控制字符。任务描述来自模型参数，
/// 因此不能直接写入前台 TTY。
fn sanitize_status_field(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());
    let mut pending_space = false;
    for ch in value.chars() {
        if ch.is_control() || ch.is_whitespace() {
            pending_space = !sanitized.is_empty();
            continue;
        }
        if pending_space {
            sanitized.push(' ');
            pending_space = false;
        }
        sanitized.push(ch);
    }
    sanitized
}

impl SubagentStatusLine {
    pub(super) fn new() -> Self {
        Self {
            last_line: None,
            is_tty: std::io::stdout().is_terminal(),
        }
    }

    pub(super) fn refresh(&mut self, app: &App) {
        let statuses = {
            let mut os = app.os.lock().unwrap_or_else(|err| err.into_inner());
            crate::ai::tools::task_tools::subagent_terminal_statuses(
                os.as_mut(),
                app.session_id.as_str(),
            )
        };
        if statuses.is_empty() {
            return;
        }

        let line = crate::ai::stream::clamp_line_to_terminal_row_with_reserve(
            &render_subagent_status_line(&statuses),
            0,
        );
        if self.last_line.as_deref() == Some(line.as_str()) {
            return;
        }

        if self.is_tty {
            print!("\r\x1b[2K{line}");
            let _ = std::io::stdout().flush();
        } else {
            println!("{line}");
        }
        self.last_line = Some(line);
    }

    /// 前台即将恢复流式输出或输入框前，把动态行固定并结束，之后不再占用光标行。
    pub(super) fn finish(&mut self) {
        let Some(line) = self.last_line.take() else {
            return;
        };
        if self.is_tty {
            print!("\r\x1b[2K{line}\n");
            let _ = std::io::stdout().flush();
        }
    }
}

fn render_subagent_status_line(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> String {
    let total = statuses.len();
    let active = statuses
        .iter()
        .filter(|status| !is_terminal_subagent_state(&status.state))
        .count();
    let mut parts = vec![if active == total {
        format!("subagents · {total} active")
    } else {
        format!("subagents · {active}/{total} active")
    }];

    parts.extend(
        statuses
            .iter()
            .take(MAX_SUBAGENT_STATUS_DETAILS)
            .map(|status| {
                format!(
                    "{} {} {} ({})",
                    sanitize_status_field(&status.description),
                    format_subagent_elapsed(status.elapsed_secs),
                    sanitize_status_field(&status.state),
                    sanitize_status_field(&status.agent_name)
                )
            }),
    );

    if total > MAX_SUBAGENT_STATUS_DETAILS {
        parts.push(format!("+{} more", total - MAX_SUBAGENT_STATUS_DETAILS));
    }

    parts.join(" · ")
}

fn is_terminal_subagent_state(state: &str) -> bool {
    matches!(
        state,
        "completed" | "failed" | "cancelled" | "canceled" | "timeout" | "terminated"
    )
}

fn format_subagent_elapsed(elapsed_secs: u64) -> String {
    if elapsed_secs < 60 {
        return format!("{elapsed_secs}s");
    }
    if elapsed_secs < 60 * 60 {
        return format!("{}m{}s", elapsed_secs / 60, elapsed_secs % 60);
    }
    format!("{}h{}m", elapsed_secs / 3600, (elapsed_secs % 3600) / 60)
}

impl Drop for SubagentStatusLine {
    fn drop(&mut self) {
        // 所有提前返回和错误路径都必须先结束未换行的动态状态行，避免 shell prompt
        // 或后续错误文本接在状态栏末尾。
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BackgroundSubagentHistoryGuard, format_subagent_elapsed, render_subagent_status_line,
        sanitize_status_field,
    };
    use crate::ai::tools::task_tools::SubagentTerminalStatus;

    #[test]
    fn subagent_status_field_is_safe_single_line_text() {
        assert_eq!(
            sanitize_status_field("review\ncode\r\x1b[31m now\tplease"),
            "review code [31m now please"
        );
    }

    #[test]
    fn subagent_status_line_shows_counts_elapsed_and_truncates_details() {
        let statuses = vec![
            SubagentTerminalStatus {
                description: "review stream".to_string(),
                agent_name: "explorer".to_string(),
                state: "running".to_string(),
                elapsed_secs: 38,
            },
            SubagentTerminalStatus {
                description: "review driver".to_string(),
                agent_name: "explorer".to_string(),
                state: "waiting".to_string(),
                elapsed_secs: 61,
            },
            SubagentTerminalStatus {
                description: "compile check".to_string(),
                agent_name: "executor".to_string(),
                state: "completed".to_string(),
                elapsed_secs: 3661,
            },
            SubagentTerminalStatus {
                description: "extra task".to_string(),
                agent_name: "executor".to_string(),
                state: "running".to_string(),
                elapsed_secs: 5,
            },
        ];

        assert_eq!(
            render_subagent_status_line(&statuses),
            "subagents · 3/4 active · review stream 38s running (explorer) · review driver 1m1s waiting (explorer) · compile check 1h1m completed (executor) · +1 more"
        );
        assert_eq!(format_subagent_elapsed(59), "59s");
        assert_eq!(format_subagent_elapsed(60), "1m0s");
    }

    #[test]
    fn background_subagent_history_guard_cleans_or_preserves_history() {
        let root = std::env::temp_dir().join(format!(
            "background-subagent-history-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let cleaned = root.join("session.proc-1.sqlite");
        std::fs::write(&cleaned, b"db").unwrap();
        std::fs::write(format!("{}-wal", cleaned.display()), b"wal").unwrap();
        drop(BackgroundSubagentHistoryGuard::new(cleaned.clone()));
        assert!(!cleaned.exists());
        assert!(!std::path::Path::new(&format!("{}-wal", cleaned.display())).exists());

        let preserved = root.join("session.proc-2.sqlite");
        std::fs::write(&preserved, b"db").unwrap();
        let mut guard = BackgroundSubagentHistoryGuard::new(preserved.clone());
        guard.preserve();
        drop(guard);
        assert!(preserved.exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn background_subagent_guard_cleans_scoped_memory_and_cwd_on_drop() {
        let root = std::env::temp_dir().join(format!(
            "background-subagent-scoped-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let history = root.join("session.subagent-task_x.sqlite");
        std::fs::write(&history, b"db").unwrap();
        // 私有 memory：jsonl 本体 + 派生 .db 及其 WAL sidecar。
        let memory = root.join("agent_memory.subagent-task_x.jsonl");
        std::fs::write(&memory, b"[]").unwrap();
        let memory_db = root.join("agent_memory.subagent-task_x.db");
        std::fs::write(&memory_db, b"db").unwrap();
        std::fs::write(format!("{}-wal", memory_db.display()), b"wal").unwrap();
        // 独占 cwd scratch 目录（含内容，需递归删除）。
        let cwd = root.join("subagent-cwd-task_x");
        std::fs::create_dir_all(cwd.join("nested")).unwrap();
        std::fs::write(cwd.join("scratch.txt"), b"tmp").unwrap();

        drop(
            BackgroundSubagentHistoryGuard::new(history.clone())
                .with_scoped_artifacts(Some(memory.clone()), Some(cwd.clone())),
        );

        assert!(!history.exists(), "history db must be cleaned");
        assert!(!memory.exists(), "memory jsonl must be cleaned");
        assert!(!memory_db.exists(), "derived memory .db must be cleaned");
        assert!(
            !std::path::Path::new(&format!("{}-wal", memory_db.display())).exists(),
            "memory .db WAL sidecar must be cleaned"
        );
        assert!(!cwd.exists(), "cwd scratch dir must be recursively cleaned");

        // preserve() 必须让 memory/cwd 也一并保留（正常结束 / resume 场景）。
        let history2 = root.join("session.subagent-task_y.sqlite");
        let memory2 = root.join("agent_memory.subagent-task_y.jsonl");
        let cwd2 = root.join("subagent-cwd-task_y");
        std::fs::write(&history2, b"db").unwrap();
        std::fs::write(&memory2, b"[]").unwrap();
        std::fs::create_dir_all(&cwd2).unwrap();
        let mut guard = BackgroundSubagentHistoryGuard::new(history2.clone())
            .with_scoped_artifacts(Some(memory2.clone()), Some(cwd2.clone()));
        guard.preserve();
        drop(guard);
        assert!(history2.exists() && memory2.exists() && cwd2.exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn background_subagent_memory_is_merged_before_cleanup() {
        let root = std::env::temp_dir().join(format!(
            "background-subagent-merge-order-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let history = root.join("session.subagent-task_merge.sqlite");
        let private_memory = root.join("agent_memory.subagent-task_merge.jsonl");
        let main_memory = root.join("agent_memory.jsonl");
        std::fs::write(&history, b"db").unwrap();
        std::fs::write(
            &private_memory,
            serde_json::json!({
                "id": "mem-merge",
                "timestamp": "2026-07-31T00:00:00Z",
                "category": "project_memory",
                "note": "durable subagent conclusion",
                "tags": [],
                "source": "test",
                "priority": 180,
                "owner_pid": 42,
                "owner_pgid": 42,
                "image_path": null
            })
            .to_string()
                + "\n",
        )
        .unwrap();
        let guard = BackgroundSubagentHistoryGuard::new(history.clone())
            .with_scoped_artifacts(Some(private_memory.clone()), None);

        assert_eq!(
            crate::ai::tools::service::memory::merge_subagent_whitelist(
                &private_memory,
                &main_memory
            )
            .unwrap(),
            1
        );
        drop(guard);

        assert!(!private_memory.exists());
        assert!(
            std::fs::read_to_string(&main_memory)
                .unwrap()
                .contains("durable subagent conclusion")
        );
        let _ = std::fs::remove_dir_all(root);
    }
}

/// Dispatch a batch of background processes: select ready processes, decode
/// task goals, build questions, clone app context, and spawn each as a
/// tokio task with proper scope setup.
pub(super) fn dispatch_background_batch(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
    manifests_loaded: &mut bool,
    epoch: u64,
) {
    let background_procs: Vec<aios_kernel::kernel::Process> = {
        let mut os = app.os.lock().unwrap();
        select_background_batch(os.as_mut(), epoch, app.session_id.as_str())
    };
    maybe_emit_scheduler_eval(epoch, app.session_id.as_str());

    if background_procs.is_empty() {
        return;
    }

    ensure_runtime_manifests_loaded(app, skill_manifests, agent_manifests, manifests_loaded);

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
        usize,
        bool,
        bool,
    )> = Vec::new();
    for proc in &background_procs {
        let pid = proc.pid;
        let task_goal = match decode_background_process_task_goal(&proc.goal) {
            Ok(goal) => goal,
            Err(err) => {
                let (result_channel_id, completion_futex_addr) =
                    with_task_entry_by_pid(pid, |entry| {
                        (
                            Some(entry.result_channel_id),
                            Some(entry.completion_futex_addr),
                        )
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

        let initialize_history = {
            let mut os = app.os.lock().unwrap();
            os.set_current_pid(Some(pid));
            let mut initialize_history = false;
            if let Some(p) = os.get_process_mut(pid) {
                initialize_history = p.history_file.is_none();
                if initialize_history {
                    p.history_file = Some(process_history_path(&original_history_file, pid));
                }
                let _ = os.process_pending_signals();
            }
            initialize_history
        };

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
            task_goal.as_ref().map(|goal| goal.spawn_depth).unwrap_or(0),
            is_resume_wakeup,
            initialize_history,
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
        spawn_depth,
        is_resume_wakeup,
        initialize_history,
    ) in task_specs
    {
        let mut task_app = app.clone();
        // 后台任务必须拥有独立的 streaming/cancel_stream 标志：`App::clone` 默认与
        // 父 App 共享同一组 Arc，多个并发后台 run_turn 会互相覆写 streaming、互相
        // 清除 cancel（clear_stream_cancel 会重置共享 cancel_stream）。后台任务的
        // cancel_stream 必须与 registry 条目共享：同步命令会轮询它并在 timeout/cancel
        // 时杀掉实际 OS 进程组，单靠 Tokio abort 无法打断正在执行的同步 poll。
        // shutdown 仍与父 App 共享：会话级退出需传播到后台任务让其优雅收尾。
        task_app.streaming = Arc::new(AtomicBool::new(false));
        task_app.cancel_stream = task_id
            .as_deref()
            .and_then(|task_id| {
                crate::ai::tools::task_tools::with_task_entry(task_id, |entry| {
                    entry.cancel_stream.clone()
                })
            })
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
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
            let capped_agent = capped_subagent_manifest(agent);
            activate_primary_agent(&mut task_app, &capped_agent);
        }
        let next_model = model_override.unwrap_or_else(|| app.current_model.clone());

        let inherit = task_id
            .as_deref()
            .and_then(|tid| crate::ai::tools::task_tools::with_task_entry(tid, |e| e.inherit))
            .unwrap_or_default();
        let (effective_history, task_skills) = match resolve_background_subagent_context(
            history_path,
            original_history_file.as_path(),
            skill_manifests,
            task_id.as_deref(),
            inherit,
            initialize_history,
        ) {
            Ok(context) => context,
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
        task_app.session_history_file = effective_history;
        let task_driver_ctx = runtime_ctx::DriverContext::new(
            task_app.clone(),
            task_mcp.clone(),
            task_skills.clone(),
            agent_manifests.clone(),
        );
        let scope_task_id = task_id.clone().unwrap_or_else(|| format!("pid-{pid}"));
        let parent_history_for_scopes = original_history_file.clone();

        // 私有 memory 文件与独占 cwd scratch 目录都随任务派生，需与 history
        // 同生命周期回收（正常结束由 preserve 保留，异常/abort/panic 由 Drop 清理）。
        // 两者路径都必须复用建立时的同源逻辑，避免第二份定义漂移导致漏删/误删。
        let scoped_memory_path = (!inherit.memory).then(|| {
            runtime_ctx::make_subagent_memory_path(&parent_history_for_scopes, &scope_task_id)
        });
        let scratch_base_for_cwd = parent_history_for_scopes
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let scoped_cwd_dir = if inherit.cwd {
            None
        } else {
            runtime_ctx::make_subagent_cwd(&scratch_base_for_cwd, &scope_task_id)
        };
        let history_guard =
            BackgroundSubagentHistoryGuard::new(task_app.session_history_file.clone())
                .with_scoped_artifacts(scoped_memory_path.clone(), scoped_cwd_dir.clone());
        let preserve_scoped_artifacts = Arc::new(AtomicBool::new(false));
        let preserve_scoped_artifacts_for_inner = preserve_scoped_artifacts.clone();
        let memory_merge_failed = Arc::new(AtomicBool::new(false));
        let memory_merge_failed_for_inner = memory_merge_failed.clone();
        let private_memory_for_merge = scoped_memory_path.clone();
        let persona_memory_path = app.current_persona_memory_file();
        let main_memory_for_merge = persona_memory_path.clone();

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
            let memory_merge_error = private_memory_for_merge.and_then(|private_memory| {
                crate::ai::tools::service::memory::merge_subagent_whitelist(
                    &private_memory,
                    &main_memory_for_merge,
                )
                .err()
            });
            if memory_merge_error.is_some() {
                memory_merge_failed_for_inner.store(true, Ordering::Release);
            }
            let mut os = task_os.lock().unwrap();
            os.set_current_pid(Some(pid));
            let task_succeeded = result.is_ok() && memory_merge_error.is_none();
            let publish_task_result = should_publish_subagent_task_result(
                task_succeeded,
                &captured_output,
                os.get_process(pid).map(|proc| &proc.state),
            );
            if publish_task_result && let Some(result_channel_id) = result_channel_id {
                let payload = serde_json::json!({
                    "status": if task_succeeded { "completed" } else { "failed" },
                    "output": captured_output,
                    "error": result.as_ref().err().cloned().or(memory_merge_error),
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
            let preserve_history = match result {
                Ok(_outcome) => {
                    let outcome = classify_process_outcome(&**os, pid);
                    record_scheduler_outcome(os.as_mut(), pid, outcome);
                    os.increment_turns_used_for(pid);
                    let (should_terminate, termination_result) =
                        finalize_turn_quota(os.as_mut(), pid);
                    if should_terminate {
                        terminate_and_cleanup(os.as_mut(), pid, termination_result, true);
                        false
                    } else if os.is_round_robin() {
                        os.set_current_pid(Some(pid));
                        os.requeue_current();
                        true
                    } else {
                        true
                    }
                }
                Err(err) => {
                    record_scheduler_outcome(os.as_mut(), pid, DispatchOutcomeTag::Failed);
                    terminate_and_cleanup(os.as_mut(), pid, format!("Failed: {}", err), true);
                    false
                }
            };
            drop(os);
            preserve_scoped_artifacts_for_inner.store(preserve_history, Ordering::Release);
        });

        type BoxedTaskFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
        let mut wrapped: BoxedTaskFuture = Box::pin(inner_fut);
        wrapped =
            Box::pin(runtime_ctx::PERSONA_MEMORY_PATH.scope(persona_memory_path.clone(), wrapped));
        wrapped = Box::pin(runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot_for_scope, wrapped));
        if let Some(mem_path) = scoped_memory_path {
            // sub-agent 默认私有 memory：finalize 后把白名单条目
            // (is_permanent_memory) 合并回主 memory 文件，让 long-term
            // assets 能跨 task 共享，但普通 task_event 留在私有文件，
            // 不污染主记忆。
            wrapped = Box::pin(runtime_ctx::SUBAGENT_MEMORY_PATH.scope(mem_path, wrapped));
        }
        if let Some(scratch) = scoped_cwd_dir {
            wrapped = Box::pin(runtime_ctx::SUBAGENT_CWD.scope(scratch, wrapped));
        }
        // 设置子代理嵌套深度，供 `task_spawn` / `task` 在子代理内部
        // 检测递归扇出时使用。
        wrapped = Box::pin(runtime_ctx::SUBAGENT_DEPTH.scope(spawn_depth, wrapped));
        // 后台任务只把最终结果交给 task_wait/task_status 聚合；禁止各 subagent
        // 直接争用前台 terminal，避免并发流式输出和 ANSI 光标控制互相破坏。
        wrapped = Box::pin(runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.scope(true, wrapped));

        // 计入在途后台子 agent：guard 随 spawned future 一同 move 进任务，
        // 任务结束（正常 / 错误 / panic）时 Drop 自动 dec，避免输入框被永久门控。
        let inflight_guard = BgSubagentGuard::new();
        let guarded_fut = async move {
            let _inflight_guard = inflight_guard;
            let mut history_guard = history_guard;
            wrapped.await;
            if memory_merge_failed.load(Ordering::Acquire) {
                history_guard.preserve_memory();
            }
            if preserve_scoped_artifacts.load(Ordering::Acquire) {
                history_guard.preserve();
            }
        };
        let handle = tokio::spawn(runtime_ctx::DRIVER_CTX.scope(task_driver_ctx, guarded_fut));
        crate::ai::tools::task_tools::set_task_abort_handle(&scope_task_id, handle.abort_handle());
    }
}
