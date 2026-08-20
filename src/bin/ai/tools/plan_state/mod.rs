//! plan 步骤状态机（model / store / render 三层）。
//!
//! - `model`：状态模型（`StepStatus` / `PlanState` 纯数据，无 I/O、无渲染）
//! - `store`：持久化（显式 `&App`，原子写，路径复用 `driver::assets_dir_for_history`）
//! - `render`：渲染 + 委派编排提示文案
//!
//! 本文件承担 `plan_update` 工具入口与模块重导出（`plan` 工具本体在 `plan_tools`）。

mod model;
mod render;
mod store;

pub(crate) use model::{PlanState, StepStatus};
pub(crate) use store::{record_plan, update_plan_step};

use serde_json::Value;

use crate::ai::tools::common::{
    ToolDisplayConfig, ToolDisplayRegistration, ToolHistoryPolicy, ToolHistoryPolicyRegistration,
    ToolLossyCompressPolicy, ToolPrunePolicy, ToolRegistration, ToolSpec,
};

fn execute_plan_update(args: &Value) -> Result<String, String> {
    let step = args
        .get("step")
        .and_then(|v| v.as_u64())
        .ok_or("Missing 'step' parameter: integer step number to update (see the active plan).")?;
    let status_name = args
        .get("status")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'status' parameter: one of pending|running|done|failed|skipped.")?;
    let status = StepStatus::parse(status_name).ok_or_else(|| {
        format!(
            "Invalid status '{status_name}': expected one of {}.",
            StepStatus::ALL.join(", ")
        )
    })?;
    let note = args
        .get("note")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let app = crate::ai::driver::runtime_ctx::try_current()
        .map(|ctx| ctx.app_proto.clone())
        .ok_or("plan_update requires an active driver session; no session context is available.")?;
    let (state, transition) = update_plan_step(&app, step, status, note)?;
    let mut result = String::new();
    if let Some(warning) = &transition.warning {
        result.push_str(warning);
        result.push('\n');
    }
    result.push_str(&state.render());
    Ok(result)
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "plan_update",
        description: "",
        execute: execute_plan_update,
        groups: &["builtin", "core"],
    }
});
inventory::submit!(ToolDisplayRegistration {
    name: "plan_update",
    config: ToolDisplayConfig {
        print_args: false,
        print_result: true,
    },
});
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "plan_update",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Allow,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});
