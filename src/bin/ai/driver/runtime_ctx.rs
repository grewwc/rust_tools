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
// In addition to the parent-runtime snapshot, this module exposes two
// finer-grained task-locals that drive the `inherit.memory` and
// `inherit.cwd` flags of the `task` / `task_spawn` tools:
//
//   - `SUBAGENT_MEMORY_PATH` overrides `MemoryStore::from_env_or_config`
//     so a sub-agent that opted out of `inherit.memory` writes / reads its
//     own jsonl file instead of the shared one.
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
use std::sync::Arc;

use crate::ai::{
    agents::AgentManifest, mcp::SharedMcpClient, models::AutoModelFallbackSpec,
    skills::SkillManifest, types::App,
};
use tokio::sync::Mutex;

/// Slot used by a sub-agent's `finalize_turn` to publish its final
/// assistant text back to the caller. The parent task installs a fresh
/// async `Mutex<Option<String>>` via `SUBAGENT_RESULT_SLOT.scope(...)` before
/// invoking `run_turn`, then reads the slot once `run_turn` returns. This
/// lets `task` / `task_spawn` actually surface the sub-agent's answer
/// instead of just an "OK / FAILED" status line.
pub(crate) type SubagentResultSlot = Arc<Mutex<Option<String>>>;

/// Slot used by a sub-agent to publish its **current phase** (e.g.
/// "preparing context" / "calling model") so the spawning `task` tool can
/// show it on the waiting heartbeat line. Unlike `SubagentResultSlot` this
/// is a plain `std::sync::Mutex` because it is written from the sub-agent
/// task and read from the parent's blocking wait loop with no `.await`
/// across the lock. The parent installs a fresh slot via
/// `SUBAGENT_PHASE.scope(...)` before invoking `run_turn`.
pub(crate) type SubagentPhaseSlot = Arc<std::sync::Mutex<String>>;

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
    /// 当前 turn 的 (session_id, turn_id) 元组。由 driver run_loop 在每
    /// 轮调度前 enter，被 DecisionLog / 反馈写入路径读取，把工具调用结
    /// 果对回到正确的 (session, turn)。未设置时下游获取到 ("", 0)。
    pub(crate) static TURN_IDENTITY: (String, usize);
    pub(crate) static AUTO_MODEL_FALLBACK: AutoModelFallbackSpec;
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

/// Publish the sub-agent's final assistant text into the active result
/// slot if one was installed by the spawning tool. Silent no-op when no
/// slot is set (e.g. top-level foreground turn).
pub(crate) async fn publish_subagent_result(text: &str) {
    if text.trim().is_empty() {
        return;
    }
    let slot = match SUBAGENT_RESULT_SLOT.try_with(|slot| slot.clone()) {
        Ok(slot) => slot,
        Err(_) => return,
    };
    let mut guard = slot.lock().await;
    *guard = Some(text.to_string());
}

/// Publish the sub-agent's current execution phase into the active phase
/// slot if one was installed by the spawning tool. Silent no-op when no
/// slot is set (top-level foreground turn, unit tests, …).
pub(crate) fn publish_subagent_phase(phase: &str) {
    let Ok(slot) = SUBAGENT_PHASE.try_with(|slot| slot.clone()) else {
        return;
    };
    if let Ok(mut guard) = slot.lock() {
        if *guard != phase {
            *guard = phase.to_string();
        }
    }
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

/// Read the optional sub-agent memory path override. `None` means
/// "fall back to shared memory file".
pub(crate) fn override_memory_path() -> Option<PathBuf> {
    SUBAGENT_MEMORY_PATH.try_with(|p| p.clone()).ok()
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

/// Build a default scratch workspace path for a sub-agent that opted out
/// of `inherit.cwd`. The directory is created on demand. Returns `None`
/// if the directory cannot be created (caller should fall back to
/// inheriting cwd in that case).
pub(crate) fn make_subagent_cwd(base: &Path, task_id: &str) -> Option<PathBuf> {
    let dir = base.join(format!("subagent-cwd-{task_id}"));
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
    fn override_memory_path_is_none_outside_scope() {
        assert!(override_memory_path().is_none());
    }

    #[test]
    fn override_memory_path_returns_value_inside_scope() {
        let want = PathBuf::from("/tmp/agent_memory.subagent-test.jsonl");
        let got = SUBAGENT_MEMORY_PATH.sync_scope(want.clone(), || override_memory_path());
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
}
