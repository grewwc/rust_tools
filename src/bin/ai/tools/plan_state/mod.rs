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

/// plan_update 的终端紧凑回显：模型仍收到完整渲染（保持计划锚点完整），终端只回显
/// 本次变更的步骤行、进度行与可能的终态覆盖警告，避免每次更新都重打整份计划。
fn compact_plan_update_echo(content: &str, args: &Value) -> String {
    let step = args.get("step").and_then(|v| v.as_u64());
    let status = args.get("status").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = String::new();

    // 终态覆盖警告：完整渲染若以 "Warning:" 开头，说明本次更新覆盖了既有终态，保留提示。
    if let Some(first) = content.lines().next()
        && first.starts_with("Warning:")
    {
        out.push_str(first);
        out.push('\n');
    }

    if let Some(n) = step {
        out.push_str(&format!("Step {n}"));
        if !status.is_empty() {
            out.push_str(&format!(" → {status}"));
        }
        // 变更后该步的渲染行（含 tool/action/note 后缀），确认具体内容。
        // parallelizable 步骤的行首有 "  || " 前缀：trim_start 只消空白，需再剥离 "||"。
        let prefix = format!("Step {n}.");
        if let Some(line) = content.lines().find(|l| {
            let t = l.trim_start();
            let t = t.strip_prefix("||").map(str::trim_start).unwrap_or(t);
            t.starts_with(&prefix)
        }) {
            out.push('\n');
            out.push_str(line.trim());
        }
    } else {
        out.push_str("Plan updated.");
    }

    // 进度行。
    if let Some(progress) = content.lines().find(|l| l.starts_with("Progress: ")) {
        out.push('\n');
        out.push_str(progress);
    }
    out
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
        display: Some(compact_plan_update_echo),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compact_echo_keeps_changed_step_and_progress() {
        let content = "\
Plan: verify compact echo
Step 1. [execute_command] Scaffold (done)
Step 2. [apply_patch] Implement (done, note: patched)
Step 3. [execute_command] Verify (pending)
Progress: 2/3 steps done, 1 pending.";
        let args = json!({"step": 2, "status": "done"});
        let out = compact_plan_update_echo(content, &args);
        assert!(out.contains("Step 2 → done"), "got: {out}");
        assert!(
            out.contains("Step 2. [apply_patch] Implement (done, note: patched)"),
            "got: {out}"
        );
        assert!(
            out.contains("Progress: 2/3 steps done, 1 pending."),
            "got: {out}"
        );
        // 未变更的步骤不回显。
        assert!(!out.contains("Step 1."), "got: {out}");
        assert!(!out.contains("Step 3."), "got: {out}");
    }

    #[test]
    fn compact_echo_preserves_warning() {
        let content = "Warning: step 1 was marked done, and is now running — this overrides a terminal status.\n\
Plan: demo
Step 1. [execute_command] A (running)
Step 2. [apply_patch] B (pending)
Progress: 0/2 steps done, 1 running, 1 pending.";
        let args = json!({"step": 1, "status": "running"});
        let out = compact_plan_update_echo(content, &args);
        assert!(out.starts_with("Warning:"), "got: {out}");
        assert!(out.contains("Step 1 → running"), "got: {out}");
    }

    /// 走真实 `render()` 锁定行格式与紧凑回显提取逻辑的耦合：render 改格式时本测试
    /// 会失败，提醒同步更新 `compact_plan_update_echo` 的匹配（不要只改手写 fixture）。
    #[test]
    fn compact_echo_stays_in_sync_with_render_format() {
        let raw = serde_json::json!([
            { "step": 1, "action": "Read", "tool": "read_file" },
            { "step": 2, "action": "Patch", "tool": "apply_patch" },
            { "step": 3, "action": "Check", "tool": "execute_command", "parallelizable": true }
        ]);
        let mut state = PlanState::build("Demo", raw.as_array().unwrap(), None).unwrap();
        state.apply_update(1, StepStatus::Done, None).unwrap();
        state.apply_update(3, StepStatus::Done, None).unwrap();

        let rendered = state.render();
        let args = json!({"step": 3, "status": "done"});
        let out = compact_plan_update_echo(&rendered, &args);

        assert!(out.contains("Step 3 → done"), "got: {out}");
        // parallelizable 前缀 "  || " 应被 trim 掉、不破坏步骤行提取。
        assert!(
            out.contains("Step 3. [execute_command] Check (done)"),
            "got: {out}"
        );
        assert!(out.contains("Progress: "), "got: {out}");
    }
}
