use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::common::{ToolDisplayConfig, ToolDisplayRegistration};
use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
};

fn execute_complete(args: &Value) -> Result<String, String> {
    let summary = args["summary"].as_str().unwrap_or("").trim();
    if summary.is_empty() {
        return Err(
            "Missing 'summary': describe what was accomplished (one or two sentences)."
                .to_string(),
        );
    }

    // Record the assertion as session state so the outstanding-work gate can check
    // it against the plan when the final response arrives. Recording failures do
    // not suppress the ack, but must remain visible as a warning: without the
    // persisted claim the gate treats the turn as an unasserted final.
    if let Some(ctx) = crate::ai::driver::runtime_ctx::try_current() {
        // Snapshot the plan's `updated_at_ms` so the gate can verify the
        // assertion later. A claim recorded without the snapshot is permanently
        // stale (the gate demands `claim.plan_updated_at_ms == plan.updated_at_ms`),
        // yet its ack would look identical to a healthy recording and every later
        // final would be reopened as unasserted. Surface the failure instead.
        let plan_updated_at_ms =
            match crate::ai::tools::plan_state::load_plan_state(&ctx.app_proto) {
                Ok(plan) => plan.map(|state| state.updated_at_ms),
                Err(err) => {
                    return Ok(format!(
                        "Completion assertion noted but could not snapshot the plan state ({err}); \
                         the final response will be treated as an unasserted final.\n\
                         Now write the final response text for this turn."
                    ));
                }
            };
        match crate::ai::tools::plan_state::record_completion_claim(
            &ctx.app_proto,
            summary,
            plan_updated_at_ms,
        ) {
            Ok(_) => Ok(format!(
                "Completion assertion recorded: {summary}\n\
                 Now write the final response text for this turn. The runtime verifies \
                 the assertion against the session plan: if plan steps are still \
                 unsettled, the turn reopens instead of ending."
            )),
            Err(e) => Ok(format!(
                "Completion assertion noted but could not persist ({e}); the final response \
                 will be treated as an unasserted final.\nNow write the final response text \
                 for this turn."
            )),
        }
    } else {
        Ok(
            "Completion assertion noted (no session context to persist it); \
             write the final response text for this turn."
                .to_string(),
        )
    }
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "complete",
        description: "",

        execute: execute_complete,
    }
});

// The ack is a machine receipt for the assertion, not user-facing content.
inventory::submit!(ToolDisplayRegistration {
    name: "complete",
    config: ToolDisplayConfig {
        print_args: false,
        print_result: false,
        emphasize_result: false,
        display: None,
    },
});

// The claim itself persists as session state; keep the history copy intact so the
// model cannot invalidate its own assertion record.
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "complete",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Allow,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_summary_is_rejected() {
        let err = execute_complete(&serde_json::json!({})).unwrap_err();
        assert!(err.contains("Missing 'summary'"), "{err}");
        let err = execute_complete(&serde_json::json!({"summary": "  "})).unwrap_err();
        assert!(err.contains("Missing 'summary'"), "{err}");
    }

    #[test]
    fn ack_without_session_context() {
        // Unit tests run outside a turn, so no runtime context exists: the tool
        // still acks instead of failing, and the gate simply sees no claim.
        let ok =
            execute_complete(&serde_json::json!({"summary": "fixed the bug"})).unwrap();
        assert!(ok.contains("final response text"), "{ok}");
    }

    #[test]
    fn unreadable_plan_state_acks_as_unasserted() {
        use crate::ai::driver::runtime_ctx::{DRIVER_CTX, DriverContext};
        use crate::ai::mcp::{McpClient, SharedMcpClient};
        use std::sync::{Arc, Mutex};
        use std::time::{SystemTime, UNIX_EPOCH};

        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let base = std::env::temp_dir().join(format!(
            "complete-unreadable-plan-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let _guard = TempDirGuard(base.clone());
        let mut app = crate::ai::middleware::test_util::test_app();
        app.session_history_file = base.join("session.sqlite");
        app.session_id = "complete-unreadable-plan-test".to_string();

        // A corrupt plan-state.json makes `load_plan_state` fail: the ack must say
        // the final will be unasserted instead of silently recording a claim whose
        // plan snapshot is permanently stale.
        let assets = crate::ai::driver::side_note::assets_dir_for_history(
            &app.session_history_file,
        );
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(
            assets.join("plan-state.json"),
            b"{ not valid json",
        )
        .unwrap();

        let ctx = DriverContext::new(
            app.clone(),
            Arc::new(Mutex::new(McpClient::new())) as SharedMcpClient,
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        DRIVER_CTX.sync_scope(ctx, || {
            let ok =
                execute_complete(&serde_json::json!({"summary": "fixed the bug"})).unwrap();
            assert!(
                ok.contains("could not snapshot the plan state"),
                "ack must surface the snapshot failure: {ok}"
            );
            assert!(
                ok.contains("unasserted final"),
                "ack must warn about the unasserted final: {ok}"
            );
        });
    }
}
