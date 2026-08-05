// =============================================================================
// AIOS Driver Runtime Context - Sub-agent dispatch context bridge
// =============================================================================
// `DRIVER_CTX` is a `tokio::task_local!` that exposes a snapshot of the
// pieces required to spawn a sub-agent's `run_turn` from inside a tool
// invocation.
//
// It is set up once per foreground/background turn in `driver::run_loop`
// and inherited by every nested `tokio::spawn` that participates in
// sub-agent dispatch (see `task_tools::execute_task`).
//
// Holding `Arc<DriverContext>` keeps the structure cheap to clone while
// still letting tools synthesise a fresh `task_app` for the spawned
// sub-agent without having to plumb additional parameters through every
// tool call.
//
// In addition to the parent-runtime snapshot, this module exposes several
// finer-grained task-locals that drive persona isolation plus the
// `inherit.memory` / `inherit.cwd` flags of the `task` / `task_spawn`
// tools:
//
//   - `PERSONA_MEMORY_PATH` overrides `MemoryStore::from_env_or_config`
//     for the whole foreground turn so each persona gets an isolated
//     long-term memory / memo store.
//
//   - `SUBAGENT_MEMORY_PATH` overrides `MemoryStore::from_env_or_config`
//     more strongly than `PERSONA_MEMORY_PATH`, so a sub-agent that opted
//     out of `inherit.memory` writes / reads its own jsonl file instead of
//     the persona-shared one.
//
//   - `SUBAGENT_CWD` overrides the project-wide `effective_cwd()` helper
//     so tools that consult it (e.g. ripgrep / find / fingerprint) honour
//     the sub-agent's scoped working directory instead of the parent's.
//
//   - `AUTO_MODEL_FALLBACK` marks sub-agent turns whose model was chosen
//     automatically. Request failures in that scope may retry with another
//     healthy auto-selected model; explicit model overrides do not.
// =============================================================================

use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::ai::{
    agents::AgentManifest, mcp::SharedMcpClient, models::AutoModelFallbackSpec,
    skills::SkillManifest, types::App,
};
use tokio::sync::Mutex;

/// 子代理发布给父代理的结果。`parent_payload` 可包含工具证据，供父代理复用；
/// `final_assistant_text` 保留模型原始最终正文，供 `response_schema` 校验。
/// 两者必须分开，避免证据包装破坏结构化 JSON 输出。
#[derive(Clone, Debug, Default)]
pub(crate) struct SubagentResult {
    pub(crate) parent_payload: String,
    pub(crate) final_assistant_text: String,
}

/// Slot used by a sub-agent's `finalize_turn` to publish its result back to
/// the caller. The parent installs a fresh slot before invoking `run_turn`,
/// then reads it once the child returns.
pub(crate) type SubagentResultSlot = Arc<Mutex<SubagentResult>>;

/// 子代理的实时进度。`phase` 随模型请求和工具调用变化，最近一次持久化的
/// 工作计划则跨 phase 保留，供父代理的单行状态渲染器组合展示。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SubagentProgress {
    phase: String,
    checkpoint_summary: Option<String>,
}

impl SubagentProgress {
    fn new(phase: &str) -> Self {
        Self {
            phase: compact_progress_text(phase, 80),
            checkpoint_summary: None,
        }
    }

    pub(crate) fn display(&self) -> String {
        match self.checkpoint_summary.as_deref() {
            Some(summary) if !summary.is_empty() => format!("{} · plan: {summary}", self.phase),
            _ => self.phase.clone(),
        }
    }
}

/// 与 `SubagentResultSlot` 不同，这里使用同步锁：子代理写入时不跨 `.await`，
/// 父代理的前台等待/刷新循环也只做一次短快照。
pub(crate) type SubagentPhaseSlot = Arc<std::sync::Mutex<SubagentProgress>>;

/// 一次性收口信号：父等待循环在预留的收口时间到达时置位，子代理在下一轮
/// 请求模型前消费它并切换到无工具的最终回答模式。
pub(crate) type SubagentWrapUpSignal = Arc<AtomicBool>;

/// Snapshot of the live runtime that a sub-agent dispatch needs.
///
/// All fields are independently cloneable so that downstream consumers can
/// take what they need without holding a long-lived borrow on the
/// foreground turn.
pub(crate) struct DriverContext {
    /// Prototype `App` cloned from the parent turn. Mutate the clone, never
    /// the prototype.
    pub(crate) app_proto: App,
    pub(crate) mcp_client: SharedMcpClient,
    pub(crate) skill_manifests: Arc<Vec<SkillManifest>>,
    pub(crate) agent_manifests: Arc<Vec<AgentManifest>>,
}

impl DriverContext {
    pub(crate) fn new(
        app_proto: App,
        mcp_client: SharedMcpClient,
        skill_manifests: Arc<Vec<SkillManifest>>,
        agent_manifests: Arc<Vec<AgentManifest>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            app_proto,
            mcp_client,
            skill_manifests,
            agent_manifests,
        })
    }
}

tokio::task_local! {
    pub(crate) static DRIVER_CTX: Arc<DriverContext>;
    /// 当前人格绑定的 memory 文件。前台 turn / one-shot note 流程会把它
    /// scope 进来，让不同 persona 的长期记忆完全隔离。
    pub(crate) static PERSONA_MEMORY_PATH: PathBuf;
    /// When set, every `MemoryStore::from_env_or_config()` inside this
    /// task scope reads/writes from this path instead of the shared
    /// `RUST_TOOLS_MEMORY_FILE` / `ai.memory.file` location. Used by
    /// `inherit.memory == false` to give the sub-agent a private memory
    /// jsonl.
    pub(crate) static SUBAGENT_MEMORY_PATH: PathBuf;
    /// When set, every `runtime_ctx::effective_cwd()` consumer inside this
    /// task scope sees this directory as the active working directory
    /// instead of `std::env::current_dir()`. Used by `inherit.cwd ==
    /// false` to scope the sub-agent to a per-task scratch workspace.
    pub(crate) static SUBAGENT_CWD: PathBuf;
    /// When set, the sub-agent's `finalize_turn` publishes its final
    /// assistant text into this slot so the spawning tool can return it
    /// to the parent agent. Absence means "no parent is interested".
    pub(crate) static SUBAGENT_RESULT_SLOT: SubagentResultSlot;
    /// When set, `runtime_ctx::publish_subagent_phase` writes the sub-agent's
    /// current execution phase here so the spawning `task` tool's heartbeat
    /// line can surface it. Absence means "no parent is showing a heartbeat".
    pub(crate) static SUBAGENT_PHASE: SubagentPhaseSlot;
    /// 父同步等待即将到达硬超时时置位；子代理只消费一次，用于请求其立即收口。
    pub(crate) static SUBAGENT_WRAP_UP_SIGNAL: SubagentWrapUpSignal;
    /// 后台 subagent 不拥有 terminal。其完整响应仍照常解析、持久化并通过 result
    /// slot 返回父 agent，但流式正文、thinking、工具状态等不得直接写 stdout/stderr，
    /// 否则多个并发任务会互相覆盖光标并打乱前台输出。
    pub(crate) static SUPPRESS_TERMINAL_OUTPUT: bool;
    /// 子代理嵌套深度。顶层 agent 未设置（等价于 0）；每 spawn 一层
    /// 子代理时递增。用于防止 `mode: all` 的 heavy agent 递归扇出
    /// 导致资源耗尽——`task_spawn` / `task` 在超过 `MAX_SUBAGENT_SPAWN_DEPTH`
    /// 时拒绝继续委派。
    pub(crate) static SUBAGENT_DEPTH: usize;
    /// 当前 turn 的 (session_id, turn_id) 元组。由 driver run_loop 在每
    /// 轮调度前 enter，被 DecisionLog / 反馈写入路径读取，把工具调用结
    /// 果对回到正确的 (session, turn)。未设置时下游获取到 ("", 0)。
    pub(crate) static TURN_IDENTITY: (String, usize);
    pub(crate) static AUTO_MODEL_FALLBACK: AutoModelFallbackSpec;
    /// 当设置时，标识当前 turn 是 foreground 进程被唤醒后的恢复执行
    /// （而非用户主动输入）。`prepare_turn` 据此将持久化的 question 消息
    /// 标记为 `internal_note` 而非 `user`，避免唤醒 prompt 被计入
    /// `/history user`、history 压缩的 user-turn 计数、以及被模型误读为
    /// 用户重复提问。
    pub(crate) static IS_RESUME_TURN: bool;
}

/// 读取当前 turn 的 session_id；未在 turn 内调用时返回空串。
pub(crate) fn current_session_id_or_empty() -> String {
    TURN_IDENTITY
        .try_with(|(s, _)| s.clone())
        .unwrap_or_default()
}

/// 读取当前 turn 的 turn_id；未在 turn 内调用时返回 0。
pub(crate) fn current_turn_id_or_zero() -> usize {
    TURN_IDENTITY.try_with(|(_, t)| *t).unwrap_or(0)
}

/// 返回当前 turn 是否是 foreground 进程唤醒后的恢复执行。
pub(crate) fn is_resume_turn() -> bool {
    IS_RESUME_TURN.try_with(|v| *v).unwrap_or(false)
}

/// 把父侧 payload 与原始最终正文一起发布到 active result slot。顶层前台
/// turn 没有安装 slot 时静默跳过。
pub(crate) async fn publish_subagent_result(
    parent_payload: &str,
    final_assistant_text: &str,
) {
    if parent_payload.trim().is_empty() {
        return;
    }
    let slot = match SUBAGENT_RESULT_SLOT.try_with(|slot| slot.clone()) {
        Ok(slot) => slot,
        Err(_) => return,
    };
    let mut guard = slot.lock().await;
    *guard = SubagentResult {
        parent_payload: parent_payload.to_string(),
        final_assistant_text: final_assistant_text.to_string(),
    };
}

pub(crate) fn has_subagent_result_slot() -> bool {
    SUBAGENT_RESULT_SLOT.try_with(|_| ()).is_ok()
}

/// 当前任务是否拥有 terminal 输出权。默认允许；后台 subagent 作用域显式关闭。
pub(crate) fn terminal_output_enabled() -> bool {
    !SUPPRESS_TERMINAL_OUTPUT.try_with(|value| *value).unwrap_or(false)
}

/// Publish the sub-agent's current execution phase into the active phase
/// slot if one was installed by the spawning tool. Silent no-op when no
/// slot is set (top-level foreground turn, unit tests, …).
pub(crate) fn publish_subagent_phase(phase: &str) {
    let Ok(slot) = SUBAGENT_PHASE.try_with(|slot| slot.clone()) else {
        return;
    };
    if let Ok(mut guard) = slot.lock() {
        let phase = compact_progress_text(phase, 80);
        if guard.phase != phase {
            guard.phase = phase;
        }
    }
}

/// 记录最近一次已成功落盘的工作计划。阶段变化只更新 `phase`，不会丢失这条
/// 可供父代理展示的任务级进展。
pub(crate) fn publish_subagent_checkpoint_summary(summary: &str) {
    let Ok(slot) = SUBAGENT_PHASE.try_with(|slot| slot.clone()) else {
        return;
    };
    if let Ok(mut guard) = slot.lock() {
        let summary = summary
            .strip_prefix("working_checkpoint:")
            .unwrap_or(summary)
            .trim();
        guard.checkpoint_summary = Some(compact_progress_text(summary, 120));
    }
}

pub(crate) fn new_subagent_progress_slot() -> SubagentPhaseSlot {
    Arc::new(std::sync::Mutex::new(SubagentProgress::new("starting")))
}

pub(crate) fn subagent_progress_snapshot(slot: &SubagentPhaseSlot) -> Option<String> {
    slot.lock().ok().map(|progress| progress.display())
}

fn compact_progress_text(value: &str, max_chars: usize) -> String {
    // 进度最终会进入父进程 TTY；先在共享槽写入边界移除 ESC、BEL、回车等
    // 终端控制字符，避免任一前台渲染路径误把模型提供的工具参数当作控制序列。
    let sanitized: String = value
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect();
    let normalized = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let prefix: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

/// 消费一次父代理发来的收口请求。未安装该 task-local 时说明当前 turn 没有
/// 预超时收口策略，保持普通执行路径。
pub(crate) fn take_subagent_wrap_up_request() -> bool {
    SUBAGENT_WRAP_UP_SIGNAL
        .try_with(|signal| signal.swap(false, Ordering::AcqRel))
        .unwrap_or(false)
}

/// 非消费式检查：父代理是否已发出预超时收口请求。用于在当前模型请求等待期间
/// 中断请求并立即进入强制收口迭代（见 `iteration.rs` 的请求中断分支）。
pub(crate) fn has_subagent_wrap_up_pending() -> bool {
    SUBAGENT_WRAP_UP_SIGNAL
        .try_with(|signal| signal.load(Ordering::Acquire))
        .unwrap_or(false)
}

/// Try to read the current `DRIVER_CTX`. Returns `None` when called from a
/// thread that has no active scope (e.g. unit tests or one-shot tool
/// invocations outside a turn).
pub(crate) fn try_current() -> Option<Arc<DriverContext>> {
    DRIVER_CTX.try_with(Arc::clone).ok()
}

pub(crate) fn auto_model_fallback_spec() -> Option<AutoModelFallbackSpec> {
    AUTO_MODEL_FALLBACK.try_with(|value| *value).ok()
}

/// Read the current sub-agent nesting depth. `0` means top-level agent
/// (no `SUBAGENT_DEPTH` task-local set). Each spawn level increments by 1.
pub(crate) fn current_subagent_depth() -> usize {
    SUBAGENT_DEPTH.try_with(|d| *d).unwrap_or(0)
}

/// Read the optional sub-agent memory path override. `None` means
/// "fall back to persona memory file / shared memory file".
pub(crate) fn override_memory_path() -> Option<PathBuf> {
    SUBAGENT_MEMORY_PATH
        .try_with(|p| p.clone())
        .ok()
        .or_else(|| PERSONA_MEMORY_PATH.try_with(|p| p.clone()).ok())
}

/// Resolve the effective working directory for tools that consult the
/// process cwd. Honours `SUBAGENT_CWD` first, then falls back to
/// `std::env::current_dir()`.
pub(crate) fn effective_cwd() -> std::io::Result<PathBuf> {
    if let Ok(p) = SUBAGENT_CWD.try_with(|p| p.clone()) {
        return Ok(p);
    }
    std::env::current_dir()
}

// =============================================================================
// Per-session temp directory + persistent temp-file registry
// =============================================================================
// agent 在执行任务时常需要写临时/中间文件（脚本、片段输出、转储等）。
// `temp_dir()` 提供一个统一的、按 session 隔离的临时目录，按需创建。
//
// 优先使用 session assets 目录（与 tool-overflow 同源），路径为
// `~/.history_file.sessions/<session>.assets/tmp/`——落在项目外、按 session
// 隔离，不污染工作区。当 DRIVER_CTX 不可用（测试 / 一次性调用）时，回退到
// `<std::env::temp_dir()/.agent_tmp/<session>/`（系统临时目录，不污染项目）。
//
// 通过 `write_file(temp=true)` 写入此目录的文件会被记录在持久化注册表
// （`storage::temp_registry`）中，供审计跟踪。临时文件在会话结束时由运行时统一清理。
// 注册表以 JSON 文件持久化，会话终止后重启仍可读取。
// =============================================================================

/// 返回当前 session 的临时目录路径，按需创建目录。供 `write_file(temp=true)`
/// 等需要写入临时文件的场景使用。
///
/// 优先返回 `<sessions_root>/<session>.assets/tmp/`（与 tool-overflow 同源，
/// 落在项目外），`DRIVER_CTX` 不可用时回退到 `<std::env::temp_dir()/.agent_tmp/<session>/`。
pub(crate) fn temp_dir() -> std::io::Result<PathBuf> {
    // 优先使用 session assets 目录（与 tool-overflow 同源），让临时文件
    // 落在项目外、按 session 隔离的 ~/.history_file.sessions/<id>.assets/tmp/。
    if let Some(ctx) = try_current() {
        let history_file = ctx.app_proto.config.history_file.clone();
        let session_id = ctx.app_proto.session_id.clone();
        let store = crate::ai::history::SessionStore::new(&history_file);
        store.ensure_root_dir()?;
        let dir = store.session_assets_dir(&session_id).join("tmp");
        std::fs::create_dir_all(&dir)?;
        return Ok(dir);
    }

    // fallback：无 DRIVER_CTX（测试 / 一次性调用）时使用系统临时目录，
    // 不落到 effective_cwd 下，避免污染项目工作区。
    let base = std::env::temp_dir();
    let session = current_session_id_or_empty();
    let session_part = if session.is_empty() {
        "default".to_string()
    } else {
        session
    };
    let dir = base.join(".agent_tmp").join(session_part);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Deterministic scratch-workspace path for a sub-agent that opted out of
/// `inherit.cwd`. Single source of truth for the path rule so that the
/// creation site ([`make_subagent_cwd`]) and the cleanup site (the history
/// guard) can never drift out of sync.
pub(crate) fn subagent_cwd_path(base: &Path, task_id: &str) -> PathBuf {
    base.join(format!("subagent-cwd-{task_id}"))
}

/// Build a default scratch workspace path for a sub-agent that opted out
/// of `inherit.cwd`. The directory is created on demand. Returns `None`
/// if the directory cannot be created (caller should fall back to
/// inheriting cwd in that case).
pub(crate) fn make_subagent_cwd(base: &Path, task_id: &str) -> Option<PathBuf> {
    let dir = subagent_cwd_path(base, task_id);
    std::fs::create_dir_all(&dir).ok().map(|_| dir)
}

/// Build the per-subagent memory file path next to the parent's history
/// file. Used by `inherit.memory == false`.
pub(crate) fn make_subagent_memory_path(base_history: &Path, task_id: &str) -> PathBuf {
    let parent = base_history.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("agent_memory.subagent-{task_id}.jsonl"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subagent_wrap_up_request_is_consumed_once() {
        let signal = Arc::new(AtomicBool::new(true));
        let (first, second) = SUBAGENT_WRAP_UP_SIGNAL.sync_scope(signal, || {
            (
                take_subagent_wrap_up_request(),
                take_subagent_wrap_up_request(),
            )
        });

        assert!(first);
        assert!(!second);
    }

    #[test]
    fn override_memory_path_is_none_outside_scope() {
        assert!(override_memory_path().is_none());
    }

    #[test]
    fn terminal_output_is_disabled_only_inside_suppressed_scope() {
        assert!(terminal_output_enabled());
        let enabled = SUPPRESS_TERMINAL_OUTPUT.sync_scope(true, terminal_output_enabled);
        assert!(!enabled);
        assert!(terminal_output_enabled());
    }

    #[test]
    fn subagent_progress_keeps_saved_plan_while_phase_changes() {
        let slot = new_subagent_progress_slot();
        let rendered = SUBAGENT_PHASE.sync_scope(slot.clone(), || {
            publish_subagent_checkpoint_summary("working_checkpoint: inspect the task path");
            publish_subagent_phase("calling model");
            subagent_progress_snapshot(&slot)
        });

        assert_eq!(rendered.as_deref(), Some("calling model · plan: inspect the task path"));
    }

    #[tokio::test]
    async fn published_subagent_result_keeps_parent_payload_and_raw_final_separate() {
        let slot: SubagentResultSlot = Arc::new(Mutex::new(SubagentResult::default()));
        SUBAGENT_RESULT_SLOT
            .scope(slot.clone(), async {
                publish_subagent_result(
                    "[Subagent tool evidence]\nread_file(...)\n\n[Subagent final answer]\n{\"ok\":true}",
                    "{\"ok\":true}",
                )
                .await;
            })
            .await;

        let result = slot.lock().await.clone();
        assert!(result.parent_payload.starts_with("[Subagent tool evidence]"));
        assert_eq!(result.final_assistant_text, "{\"ok\":true}");
    }

    #[test]
    fn override_memory_path_returns_value_inside_scope() {
        let want = PathBuf::from("/tmp/agent_memory.subagent-test.jsonl");
        let got = SUBAGENT_MEMORY_PATH.sync_scope(want.clone(), || override_memory_path());
        assert_eq!(got, Some(want));
    }

    #[test]
    fn override_memory_path_falls_back_to_persona_scope() {
        let want = PathBuf::from("/tmp/agent_memory.persona-test.jsonl");
        let got = PERSONA_MEMORY_PATH.sync_scope(want.clone(), || override_memory_path());
        assert_eq!(got, Some(want));
    }

    #[test]
    fn effective_cwd_falls_back_to_process_cwd() {
        let process_cwd = std::env::current_dir().unwrap();
        let got = effective_cwd().unwrap();
        assert_eq!(got, process_cwd);
    }

    #[test]
    fn effective_cwd_honours_subagent_override() {
        let want = std::env::temp_dir();
        let got = SUBAGENT_CWD.sync_scope(want.clone(), || effective_cwd().unwrap());
        assert_eq!(got, want);
    }

    #[test]
    fn make_subagent_memory_path_lands_next_to_parent_history() {
        let parent = PathBuf::from("/tmp/sessions/session-foo.jsonl");
        let got = make_subagent_memory_path(&parent, "abc123");
        assert_eq!(
            got,
            PathBuf::from("/tmp/sessions/agent_memory.subagent-abc123.jsonl")
        );
    }

    #[test]
    fn make_subagent_memory_path_handles_root_history() {
        let parent = PathBuf::from("session.jsonl");
        let got = make_subagent_memory_path(&parent, "abc");
        assert_eq!(got, PathBuf::from("agent_memory.subagent-abc.jsonl"));
    }

    #[test]
    fn make_subagent_cwd_creates_scoped_directory() {
        let base = std::env::temp_dir().join(format!(
            "rust_tools_runtime_ctx_test_{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let got = make_subagent_cwd(&base, "tid").unwrap();
        assert!(got.is_dir());
        assert!(got.starts_with(&base));
        assert!(got.ends_with("subagent-cwd-tid"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn is_resume_turn_defaults_false_outside_scope() {
        assert!(!is_resume_turn());
    }

    #[test]
    fn is_resume_turn_true_inside_scope() {
        let got = IS_RESUME_TURN.sync_scope(true, || is_resume_turn());
        assert!(got);
    }
}
