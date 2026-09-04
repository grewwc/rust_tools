//! plan 状态持久化：显式接收 `&App`（工具入口解析会话上下文），会话资产路径与
//! side_note 共用 `driver::side_note::assets_dir_for_history` 推导；原子写（临时文件 + rename）
//! + 进程内互斥锁，避免并发读写撕裂状态文件。

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::ai::driver::side_note::assets_dir_for_history;
use crate::ai::types::App;

use super::model::{PlanState, StepStatus, StepTransition};

pub(crate) const PLAN_STATE_FILE_NAME: &str = "plan-state.json";

/// 会话内活跃计划的持久化路径（位于 session assets 根下）。
pub(crate) fn plan_state_path(app: &App) -> PathBuf {
    // 会话资产根与 side_note / checkpoint 共用同一推导：`session_history_file`
    // 即 `<sessions_root>/<id>.sqlite`，取其父目录 + stem 即 `<sessions_root>/<id>.assets`。
    assets_dir_for_history(&app.session_history_file).join(PLAN_STATE_FILE_NAME)
}

pub(crate) fn load_plan_state(app: &App) -> Result<Option<PlanState>, String> {
    let path = plan_state_path(app);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read plan state {}: {e}", path.display())),
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|e| format!("plan state {} is corrupt: {e}", path.display()))
}

/// 原子落盘（tmp + rename），避免中途崩溃留下半截 JSON。
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// 会话内计划读-改-写互斥：`record_plan` / `update_plan_step` 的 load→mutate→save
/// 必须原子化，否则并发调用（如同轮多个工具、后台子代理）会互相覆盖对方的状态。
/// 仅进程内线程间互斥；跨进程安全由 tmp+rename 原子落盘兜底（最后写者胜）。
static PLAN_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn save_plan_state(app: &App, state: &PlanState) -> Result<(), String> {
    let path = plan_state_path(app);
    let parent = path
        .parent()
        .ok_or_else(|| "invalid plan-state path".to_string())?;
    std::fs::create_dir_all(parent).map_err(|e| format!("cannot create plan-state dir: {e}"))?;
    // tmp 名按 进程 id + 单调序号 唯一，同一会话内并发写互不覆盖彼此的 tmp 文件；
    // rename 本身原子，最终路径上只会出现完整 JSON。
    let tmp = parent.join(format!(
        ".{}.{}-{}.tmp",
        PLAN_STATE_FILE_NAME,
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes =
        serde_json::to_vec_pretty(state).map_err(|e| format!("serialize plan-state: {e}"))?;
    if let Err(e) = std::fs::write(&tmp, &bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("write plan state {}: {e}", path.display()));
    }
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("persist plan state {}: {e}", path.display())
    })?;
    Ok(())
}

pub(crate) fn record_plan(
    app: &App,
    summary: &str,
    raw_steps: &[Value],
) -> Result<PlanState, String> {
    let _guard = PLAN_LOCK
        .lock()
        .map_err(|_| "plan-state lock poisoned".to_string())?;
    let previous = load_plan_state(app)?;
    let state = PlanState::build(summary, raw_steps, previous.as_ref())?;
    save_plan_state(app, &state)?;
    Ok(state)
}

pub(crate) fn update_plan_step(
    app: &App,
    step: u64,
    status: StepStatus,
    note: Option<String>,
) -> Result<(PlanState, StepTransition), String> {
    let _guard = PLAN_LOCK
        .lock()
        .map_err(|_| "plan-state lock poisoned".to_string())?;
    let mut state = match load_plan_state(app)? {
        Some(state) => state,
        None => {
            return Err(
                "No active plan in this session. Call `plan` (or plan again) to create the step list first."
                    .to_string(),
            )
        }
    };
    let transition = state.apply_update(step, status, note)?;
    save_plan_state(app, &state)?;
    Ok((state, transition))
}
