//! Plan step state machine split into model, store, and render layers.
//!
//! - `model`: pure `StepStatus` / `PlanState` data without I/O or rendering
//! - `store`: persistence with explicit `&App`, atomic writes, and shared asset paths
//! - `render`: user-facing output and delegation guidance
//!
//! This module also hosts the `plan_update` entry point and public re-exports.

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
    let ctx = crate::ai::driver::runtime_ctx::try_current()
        .ok_or("plan_update requires an active driver session; no session context is available.")?;
    let (state, transition) = update_plan_step(&ctx.app_proto, step, status, note)?;
    let mut result = String::new();
    if let Some(warning) = &transition.warning {
        result.push_str(warning);
        result.push('\n');
    }
    result.push_str(&state.render());
    Ok(result)
}

/// Compact terminal echo for plan_update. The model still receives the complete
/// plan, while the terminal shows only the changed step, progress, and warnings.
fn compact_plan_update_echo(content: &str, args: &Value) -> String {
    let step = args.get("step").and_then(|v| v.as_u64());
    let status = args.get("status").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = String::new();

    // Preserve a leading warning when the update overrides a terminal status.
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
        // Include the updated rendered step with its tool, action, and note suffix.
        // Parallel steps start with `||`, which must be stripped after whitespace.
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

    // Include the aggregate progress line.
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
    }
});
inventory::submit!(ToolDisplayRegistration {
    name: "plan_update",
    config: ToolDisplayConfig {
        print_args: false,
        print_result: true,
        emphasize_result: true,
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
        // Unchanged steps are omitted from the compact echo.
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

    /// Exercise the real renderer so format changes also require updating compact extraction.
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
        // The parallel `||` prefix must not prevent extraction of the step line.
        assert!(
            out.contains("Step 3. [execute_command] Check (done)"),
            "got: {out}"
        );
        assert!(out.contains("Progress: "), "got: {out}");
    }
}
