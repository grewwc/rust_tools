use super::{
    CachedFileFingerprint, TOOL_CACHE_TTL_MINUTES, ToolCachePayload, ToolFailureKind, ToolRoute,
    build_tool_cache_key, classify_tool_error, collect_tool_cache_file_fingerprints,
    execute_tool_calls, execute_with_safe_retry, is_cacheable_tool_name,
    is_parallel_safe_tool_call, is_tool_cache_entry_fresh, parallel_safe_batch_len,
    prepare_tool_call, should_retry_once, should_store_or_load_tool_cache,
    tool_cache_validation_matches,
};
use crate::ai::mcp::McpClient;
use crate::ai::tools::registry::common::{ToolParallelPolicy, tool_parallel_policy};
use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
use crate::ai::types::{FunctionCall, ToolCall};
use aios_kernel::primitives::ResourceLimit;
use chrono::{Duration, Utc};
use rust_tools::commonw::FastSet;
use serde_json::json;
use std::fs;
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

struct ReceiptGateFixture {
    root: std::path::PathBuf,
    skills: std::path::PathBuf,
    manifest: std::path::PathBuf,
    shared_mcp: crate::ai::mcp::SharedMcpClient,
}

impl ReceiptGateFixture {
    fn call(name: &str, id: &str, args: serde_json::Value) -> ToolCall {
        let mut call = tool_call(name);
        call.id = id.into();
        call.function.arguments = args.to_string();
        call
    }

    fn run(&self, calls: &[ToolCall]) -> super::ExecuteToolCallsResult {
        self.run_observed(calls, None)
    }

    fn run_observed(
        &self,
        calls: &[ToolCall],
        observer: Option<&mut dyn super::ToolExecutionObserver>,
    ) -> super::ExecuteToolCallsResult {
        execute_tool_calls(
            "receipt-gate",
            &McpClient::new(),
            &self.shared_mcp,
            calls,
            None,
            observer,
        )
        .unwrap()
    }

    fn record(&self) -> String {
        let mut args = json!({"op": "snapshot", "target_tool": "delete_skill", "target_args": {"name": "demo"}});
        let snapshot = self.run(&[Self::call("decision_receipt", "snapshot", args.clone())]);
        let snapshot: serde_json::Value =
            serde_json::from_str(&snapshot.tool_results[0].content).unwrap();
        assert_eq!(snapshot["source_paths"].as_array().unwrap().len(), 2);
        assert!(
            fs::read_to_string(&self.manifest)
                .unwrap()
                .starts_with("skill ")
        );
        args["op"] = json!("write");
        args["kind"] = json!("skill_deletion");
        args["source_fingerprint"] = snapshot["source_fingerprint"].clone();
        let receipt = self.run(&[Self::call("decision_receipt", "record", args)]);
        let receipt: serde_json::Value =
            serde_json::from_str(&receipt.tool_results[0].content).unwrap();
        receipt["id"].as_str().unwrap().to_string()
    }

    fn list(&self, id: Option<&str>) -> serde_json::Value {
        let args = id.map_or_else(
            || json!({"op": "list"}),
            |id| json!({"op": "list", "id": id}),
        );
        let output = self.run(&[Self::call("decision_receipt", "list", args)]);
        serde_json::from_str(&output.tool_results[0].content).unwrap()
    }

    fn configure_skills(&self, path: &std::path::Path) {
        fs::write(
            self.root.join("test.configw"),
            format!("ai.skills.dir = \"{}\"\n", path.display()),
        )
        .unwrap();
        crate::commonw::configw::refresh();
    }

    fn marker(&self, id: &str, extension: &str) -> std::path::PathBuf {
        self.root
            .join("receipt-gate.assets/decision-receipts")
            .join(format!("{id}.{extension}"))
    }

    fn delete(id: &str, receipt: &str) -> ToolCall {
        Self::call(
            "delete_skill",
            id,
            json!({"name": "demo", "decision_receipt_id": receipt}),
        )
    }
}

fn with_receipt_gate_fixture(test: impl FnOnce(&ReceiptGateFixture)) {
    use crate::ai::driver::runtime_ctx::{DRIVER_CTX, DriverContext};
    use std::sync::Arc;

    let _env = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (_kernel_guard, _kernel, _pid) = setup_tool_kernel();
    let root = std::env::temp_dir().join(format!("receipt-gate-{}", uuid::Uuid::new_v4()));
    let skills = root.join("skills");
    let manifest = skills.join("demo/SKILL.md");
    fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    fs::write(&manifest, "skill v1").unwrap();
    let config = root.join("test.configw");
    fs::write(
        &config,
        format!("ai.skills.dir = \"{}\"\n", skills.display()),
    )
    .unwrap();
    struct RestoreConfig(Option<std::ffi::OsString>);
    impl Drop for RestoreConfig {
        fn drop(&mut self) {
            unsafe {
                if let Some(path) = &self.0 {
                    std::env::set_var("CONFIGW_PATH", path);
                } else {
                    std::env::remove_var("CONFIGW_PATH");
                }
            }
            crate::commonw::configw::refresh();
        }
    }
    let _restore = RestoreConfig(std::env::var_os("CONFIGW_PATH"));
    unsafe { std::env::set_var("CONFIGW_PATH", config) };
    crate::commonw::configw::refresh();
    let mut app = super::super::tests::test_app("build");
    app.session_id = "receipt-gate".to_string();
    app.session_history_file = root.join("receipt-gate.sqlite");
    let shared_mcp = Arc::new(Mutex::new(McpClient::new()));
    let ctx = DriverContext::from_app_snapshot(
        &app,
        shared_mcp.clone(),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    let fixture = ReceiptGateFixture {
        root,
        skills,
        manifest,
        shared_mcp,
    };
    DRIVER_CTX.sync_scope(ctx, || test(&fixture));
    fs::remove_dir_all(fixture.root).unwrap();
}

#[test]
fn decision_receipt_gate_preserves_batch_slots_and_rechecks_at_dispatch() {
    with_receipt_gate_fixture(|fixture| {
        let old = fixture.record();
        fs::write(&fixture.manifest, "skill v2").unwrap();
        let rejected = fixture.run(&[
            ReceiptGateFixture::call("delete_skill", "missing", json!({"name": "demo"})),
            ReceiptGateFixture::call("decision_receipt", "unrelated", json!({"op": "list"})),
            ReceiptGateFixture::call(
                "delete_skill",
                "mismatch",
                json!({"name": "other", "decision_receipt_id": old}),
            ),
            ReceiptGateFixture::delete("stale", &old),
        ]);
        assert!(rejected.had_error);
        for (slot, (id, reason)) in [
            (0, ("missing", "missing decision_receipt_id")),
            (2, ("mismatch", "does not match")),
            (3, ("stale", "source changed")),
        ] {
            let result = &rejected.tool_results[slot];
            assert_eq!(result.tool_call_id, id);
            assert!(result.content.contains(reason));
            assert!(rejected.execution_outcomes[slot].is_none());
            assert!(rejected.pre_dispatch_error_ids.contains(id));
        }
        assert!(
            rejected.tool_results[0]
                .content
                .contains("narrowly re-read")
        );
        assert!(
            rejected.tool_results[3]
                .content
                .contains("narrowly re-read")
        );
        assert_eq!(rejected.tool_results[1].tool_call_id, "unrelated");
        assert!(rejected.execution_outcomes[1].as_ref().unwrap().succeeded);
        assert!(fixture.manifest.is_file());

        let new = fixture.record();
        let accepted = fixture.run(&[
            ReceiptGateFixture::delete("fresh", &new),
            ReceiptGateFixture::delete("used", &new),
            ReceiptGateFixture::call(
                "decision_receipt",
                "after",
                json!({"op": "list", "id": new}),
            ),
        ]);
        assert!(
            accepted.execution_outcomes[0].as_ref().unwrap().succeeded,
            "{}",
            accepted.tool_results[0].content
        );
        assert!(!fixture.skills.join("demo").exists());
        assert!(accepted.execution_outcomes[1].is_none());
        let missing = &accepted.tool_results[1].content;
        assert!(missing.contains("source changed"));
        assert!(missing.contains("Report the intended target as missing; do not retry"));
        assert!(!missing.contains("op=snapshot"));
        assert!(accepted.execution_outcomes[2].as_ref().unwrap().succeeded);
    });
}

#[test]
fn decision_receipt_rejects_replacement_after_gate_check() {
    struct ReplaceOnStart<'a> {
        fixture: &'a ReceiptGateFixture,
        started: bool,
    }
    impl super::ToolExecutionObserver for ReplaceOnStart<'_> {
        fn on_tool_started(&mut self, call: &ToolCall) {
            assert_eq!(call.function.name, "delete_skill");
            let manifest = &self.fixture.manifest;
            let bytes = fs::read(manifest).unwrap();
            fs::rename(manifest, self.fixture.root.join("saved-manifest")).unwrap();
            fs::write(manifest, bytes).unwrap();
            self.started = true;
        }
    }

    with_receipt_gate_fixture(|fixture| {
        let single = fixture.skills.join("demo.skill");
        fs::write(&single, "skill file").unwrap();
        let id = fixture.record();
        let mut observer = ReplaceOnStart {
            fixture,
            started: false,
        };
        let output = fixture.run_observed(
            &[ReceiptGateFixture::delete("replace", &id)],
            Some(&mut observer),
        );
        assert!(
            observer.started,
            "the initial gate must pass before replacement"
        );
        assert!(output.tool_results[0].content.contains("source changed"));
        assert!(output.execution_outcomes[0].is_none());
        assert!(output.pre_dispatch_error_ids.contains("replace"));
        assert_eq!(fs::read_to_string(&fixture.manifest).unwrap(), "skill v1");
        assert_eq!(fs::read_to_string(&single).unwrap(), "skill file");
    });
}

#[cfg(unix)]
#[test]
fn decision_receipt_marker_write_failure_stays_stale_and_cannot_dispatch() {
    use crate::ai::tools::storage::decision_receipts::Ledger;
    use std::os::unix::fs::symlink;

    with_receipt_gate_fixture(|fixture| {
        let id = fixture.record();
        let ledger = Ledger::current().unwrap();
        let receipt = ledger.read(&id).unwrap();
        let marker = fixture.marker(&id, "changed");
        // A dangling marker makes try_exists return false but create_new fail,
        // even when this test runs with elevated filesystem privileges.
        symlink(fixture.root.join("nonexistent-marker"), &marker).unwrap();
        // Change what the registered resolver sees without changing the
        // source fingerprint. Only the durable reservation can keep it stale.
        fixture.configure_skills(&fixture.root.join("missing-skills"));
        assert_eq!(fixture.list(Some(&id))["freshness"], "stale");
        assert!(fixture.marker(&id, "verifying").is_file());
        fs::remove_file(marker).unwrap();
        fixture.configure_skills(&fixture.skills);
        assert_eq!(
            receipt.decision.source.fingerprint().unwrap(),
            receipt.decision.source_fingerprint
        );
        assert_eq!(fixture.list(Some(&id))["freshness"], "stale");
        let rejected = fixture.run(&[ReceiptGateFixture::delete("blocked", &id)]);
        assert!(
            rejected.tool_results[0]
                .content
                .contains("receipt is stale")
        );
        assert!(rejected.execution_outcomes[0].is_none());
        assert!(fixture.manifest.is_file());
    });
}

#[cfg(unix)]
#[test]
fn decision_receipt_intermediate_symlink_retarget_restore_blocks_dispatch() {
    use crate::ai::tools::storage::decision_receipts::{Decision, Ledger, Sources};
    use std::os::unix::fs::symlink;

    with_receipt_gate_fixture(|fixture| {
        let hop = fixture.root.join("hop");
        let alias = fixture.root.join("alias");
        symlink(".", &hop).unwrap();
        symlink("hop", &alias).unwrap();
        let configured = alias.join("skills");
        fixture.configure_skills(&configured);
        let action_id = fixture.record();
        let ledger = Ledger::current().unwrap();
        let file_source = Sources::File(configured.join("demo/SKILL.md"));
        let file_receipt = ledger
            .write(Decision {
                kind: "file_selection".into(),
                target_tool: "example_create".into(),
                target_args: json!({}),
                cwd: crate::ai::driver::runtime_ctx::effective_cwd().unwrap(),
                source_fingerprint: file_source.fingerprint().unwrap(),
                source: file_source,
                numeric_params: json!({}),
                selected_ids: Vec::new(),
            })
            .unwrap();

        // `hop` is reached by expanding `alias`, not by scanning the lexical
        // components of the configured path. Restore the exact file and tree.
        fs::create_dir_all(fixture.root.join("alternate/skills")).unwrap();
        fs::rename(&hop, fixture.root.join("saved-hop")).unwrap();
        symlink("alternate", &hop).unwrap();
        assert_ne!(
            fs::canonicalize(&configured).unwrap(),
            fs::canonicalize(&fixture.skills).unwrap()
        );
        fs::remove_file(&hop).unwrap();
        symlink(".", &hop).unwrap();
        assert_eq!(
            fs::canonicalize(&configured).unwrap(),
            fs::canonicalize(&fixture.skills).unwrap()
        );

        assert_ne!(fixture.list(Some(&file_receipt.id))["freshness"], "fresh");
        let rejected = fixture.run(&[ReceiptGateFixture::delete("restored-link", &action_id)]);
        assert!(rejected.execution_outcomes[0].is_none());
        assert!(
            rejected.tool_results[0]
                .content
                .contains("receipt is stale")
        );
        assert!(fixture.manifest.is_file());
    });
}

#[test]
fn decision_receipt_missing_source_directory_invalidates_across_restore() {
    with_receipt_gate_fixture(|fixture| {
        let id = fixture.record();
        let hidden = fixture.root.join("hidden-skills");
        fs::rename(&fixture.skills, &hidden).unwrap();
        let rejected = fixture.run(&[ReceiptGateFixture::delete("missing-directory", &id)]);
        assert!(rejected.execution_outcomes[0].is_none());
        assert!(
            rejected.tool_results[0]
                .content
                .contains("Report the intended target as missing")
        );

        fs::rename(hidden, &fixture.skills).unwrap();
        assert!(fixture.marker(&id, "changed").is_file());
        let rejected = fixture.run(&[ReceiptGateFixture::delete("restored-directory", &id)]);
        assert!(rejected.execution_outcomes[0].is_none());
        assert!(
            rejected.tool_results[0]
                .content
                .contains("receipt is stale")
        );
        assert_eq!(fixture.list(Some(&id))["freshness"], "changed");
        assert!(fixture.manifest.is_file());
    });
}

#[test]
fn decision_receipt_executor_preserves_nested_replacement_after_authorization() {
    use crate::ai::tools::{decision_receipts, registry::common};

    with_receipt_gate_fixture(|fixture| {
        let package = fixture.manifest.parent().unwrap();
        let nested = package.join("nested/data.txt");
        fs::create_dir(nested.parent().unwrap()).unwrap();
        fs::write(&nested, "verified data").unwrap();
        let sibling_target = fixture.skills.join("demo.skill");
        fs::write(&sibling_target, "verified file skill").unwrap();
        let id = fixture.record();
        let mut args = json!({"name": "demo", "decision_receipt_id": id});
        let verified = decision_receipts::verify_before_dispatch(
            common::irreversible_tool("delete_skill").unwrap(),
            &mut args,
        )
        .unwrap();
        let authorized = verified.authorize_execution(&args).unwrap();
        let root_mtime = fs::metadata(package).unwrap().modified().unwrap();
        fs::rename(&nested, fixture.root.join("verified-data")).unwrap();
        fs::write(&nested, "new data that must survive").unwrap();
        assert_eq!(
            fs::metadata(package).unwrap().modified().unwrap(),
            root_mtime
        );

        let error = common::execute_authorized_tool_call("nested-replacement", authorized)
            .err()
            .unwrap();
        assert!(
            error.contains("Cannot delete changed skill target"),
            "{error}"
        );
        assert_eq!(
            fs::read_to_string(nested).unwrap(),
            "new data that must survive"
        );
        assert!(fixture.manifest.is_file());
        assert!(
            sibling_target.is_file(),
            "the batch must fail before any removal"
        );
    });
}

#[cfg(unix)]
#[test]
fn decision_receipt_dangling_skill_link_rejects_without_retry() {
    use std::os::unix::fs::symlink;

    with_receipt_gate_fixture(|fixture| {
        let id = fixture.record();
        let link = fixture.skills.join("demo.skill");
        symlink("nonexistent", &link).unwrap();
        let rejected = fixture.run(&[ReceiptGateFixture::delete("dangling", &id)]);
        assert!(rejected.execution_outcomes[0].is_none());
        let text = &rejected.tool_results[0].content;
        assert!(text.contains("do not retry"), "{text}");
        assert!(!text.contains("op=snapshot"));
        assert!(link.is_symlink());
        assert!(fixture.manifest.is_file());
    });
}

#[cfg(unix)]
#[test]
fn decision_receipt_dangling_companion_reports_present_skill_as_unverifiable() {
    use std::os::unix::fs::symlink;

    with_receipt_gate_fixture(|fixture| {
        let link = fixture.skills.join("demo.skill");
        symlink("nonexistent", &link).unwrap();
        let rejected = fixture.run(&[ReceiptGateFixture::call(
            "delete_skill",
            "unverified",
            json!({"name": "demo"}),
        )]);
        assert!(rejected.execution_outcomes[0].is_none());
        let text = &rejected.tool_results[0].content;
        assert!(
            text.contains("companion entry is unreadable or unverifiable"),
            "{text}"
        );
        assert!(
            text.contains("Report the source as unavailable; do not retry"),
            "{text}"
        );
        assert!(!text.contains("target as missing"), "{text}");
        assert!(!text.contains("op=snapshot"), "{text}");
        assert!(link.is_symlink());
        assert!(fixture.manifest.is_file());
    });
}

#[cfg(unix)]
#[test]
fn decision_receipt_list_rejects_previously_bound_link_that_becomes_dangling() {
    use std::os::unix::fs::symlink;

    with_receipt_gate_fixture(|fixture| {
        let referents = fixture.skills.join("referents");
        fs::create_dir(&referents).unwrap();
        let target = referents.join("source.skill");
        fs::write(&target, "linked source").unwrap();
        let link = fixture.skills.join("demo.skill");
        symlink("referents/source.skill", &link).unwrap();
        let detail_id = fixture.record();
        let page_id = fixture.record();
        assert_eq!(fixture.list(Some(&detail_id))["freshness"], "fresh");

        fs::rename(target, fixture.root.join("saved-source")).unwrap();
        assert!(link.is_symlink());
        let detail = fixture.list(Some(&detail_id));
        assert_ne!(
            detail["freshness"], "fresh",
            "the unchanged link no longer resolves"
        );
        // This second receipt has not already been invalidated by the detail lookup.
        let page = fixture.list(None);
        let row = page["receipts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == page_id)
            .unwrap();
        assert_eq!(row["freshness"], detail["freshness"]);
        let rejected = fixture.run(&[ReceiptGateFixture::delete("broken-referent", &detail_id)]);
        assert!(rejected.execution_outcomes[0].is_none());
        assert!(fixture.manifest.is_file());
    });
}

#[cfg(unix)]
#[test]
fn decision_receipt_registered_top_level_link_binds_its_referent_chain() {
    use std::os::unix::fs::symlink;

    with_receipt_gate_fixture(|fixture| {
        let referents = fixture.skills.join("referents");
        fs::create_dir(&referents).unwrap();
        fs::write(referents.join("original"), "original").unwrap();
        fs::write(referents.join("different"), "different").unwrap();
        let hop = referents.join("hop");
        symlink("original", &hop).unwrap();
        let link = fixture.skills.join("demo.skill");
        symlink("referents/hop", &link).unwrap();
        let id = fixture.record();
        let old_destination = fs::canonicalize(&link).unwrap();

        // Change only a nested hop, then restore the same final destination.
        fs::rename(&hop, referents.join("original-hop")).unwrap();
        symlink("different", &hop).unwrap();
        assert_ne!(fs::canonicalize(&link).unwrap(), old_destination);
        fs::remove_file(&hop).unwrap();
        symlink("original", &hop).unwrap();
        assert_eq!(fs::canonicalize(&link).unwrap(), old_destination);
        let rejected = fixture.run(&[ReceiptGateFixture::delete("restored-referent-chain", &id)]);
        assert!(
            rejected.execution_outcomes[0].is_none(),
            "{}",
            rejected.tool_results[0].content
        );
        assert!(link.is_symlink());
        assert!(fixture.manifest.is_file());

        let fresh = fixture.record();
        let accepted = fixture.run(&[ReceiptGateFixture::delete(
            "reverified-referent-chain",
            &fresh,
        )]);
        assert!(
            accepted.execution_outcomes[0]
                .as_ref()
                .is_some_and(|outcome| outcome.succeeded),
            "{}",
            accepted.tool_results[0].content
        );
        assert!(!link.is_symlink());
        assert!(!fixture.manifest.exists());
        assert_eq!(fs::read_to_string(old_destination).unwrap(), "original");
    });
}

#[cfg(unix)]
#[test]
fn decision_receipt_unobserved_directory_relocation_invalidates_dispatch() {
    with_receipt_gate_fixture(|fixture| {
        let id = fixture.record();
        let moved = fixture.root.join("moved-skills");
        fs::rename(&fixture.skills, &moved).unwrap();
        fs::rename(moved, &fixture.skills).unwrap();

        // No receipt lookup observed the missing directory.
        let rejected = fixture.run(&[ReceiptGateFixture::delete("restored-unobserved", &id)]);
        assert!(
            rejected.execution_outcomes[0].is_none(),
            "{}",
            rejected.tool_results[0].content
        );
        assert!(fixture.manifest.is_file());
    });
}

#[test]
fn subagent_tool_phase_includes_file_target_without_rendering() {
    let args = json!({"file_path": "/repo/src/main.rs"});
    assert_eq!(
        super::subagent_tool_phase("read_file", &args),
        "using read_file · /repo/src/main.rs"
    );
    assert_eq!(
        super::subagent_tool_phase("task_status", &json!({})),
        "using task_status"
    );
}

#[test]
fn execution_signature_canonicalizes_arguments_and_tracks_environment() {
    let first = ToolCall {
        id: "first".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: r#"{"b":2,"a":{"y":1,"x":0}}"#.to_string(),
        },
    };
    let second = ToolCall {
        id: "second".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: r#"{"a":{"x":0,"y":1},"b":2}"#.to_string(),
        },
    };

    let first =
        super::tool_execution_outcome("session", "/repo", &ToolRoute::Builtin, &first, false);
    let second =
        super::tool_execution_outcome("session", "/repo", &ToolRoute::Builtin, &second, true);
    assert_eq!(first.execution_signature, second.execution_signature);

    let different_cwd = super::tool_execution_outcome(
        "session",
        "/other",
        &ToolRoute::Builtin,
        &ToolCall {
            id: "third".to_string(),
            ..first_tool_call()
        },
        true,
    );
    assert_ne!(first.execution_signature, different_cwd.execution_signature);
}

fn first_tool_call() -> ToolCall {
    ToolCall {
        id: "template".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: r#"{"b":2,"a":{"y":1,"x":0}}"#.to_string(),
        },
    }
}

#[test]
fn malformed_tool_name_returns_a_bounded_error_without_echoing_source_text() {
    let call = ToolCall {
        id: "malformed".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: crate::ai::types::INVALID_TOOL_CALL_NAME.to_string(),
            arguments: "{}".to_string(),
        },
    };

    let error = prepare_tool_call(&McpClient::new(), &call, None).unwrap_err();

    assert_eq!(error.tool_call_id, "malformed");
    assert!(error.content.contains("malformed tool name"));
    assert!(
        !error
            .content
            .contains(crate::ai::types::INVALID_TOOL_CALL_NAME)
    );
}

static TOOL_KERNEL_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct ToolKernelTestGuard {
    _lock: MutexGuard<'static, ()>,
}

impl Drop for ToolKernelTestGuard {
    fn drop(&mut self) {
        if let Ok(mut g) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
            *g = None;
        }
    }
}

fn setup_tool_kernel() -> (ToolKernelTestGuard, aios_kernel::kernel::SharedKernel, u64) {
    let lock = TOOL_KERNEL_TEST_LOCK.lock().unwrap();
    if let Ok(mut g) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *g = None;
    }
    let guard = ToolKernelTestGuard { _lock: lock };
    let kernel = crate::ai::driver::new_local_kernel();
    let root = {
        let mut os = kernel.lock().unwrap();
        os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None)
    };
    crate::ai::tools::os_tools::init_os_tools_globals(kernel.clone());
    (guard, kernel, root)
}

#[test]
fn read_file_is_forced_to_sequential_path() {
    let mcp = McpClient::new();
    assert!(is_cacheable_tool_name("read_file"));
    assert!(!is_parallel_safe_tool_call(&mcp, &tool_call("read_file")));
    assert_eq!(parallel_safe_batch_len(&mcp, &[tool_call("read_file")]), 0);
}

#[test]
fn parallel_batch_stops_at_serial_grounding_tool() {
    let mcp = McpClient::new();
    let calls = vec![tool_call("read_file"), tool_call("knowledge_search")];
    assert_eq!(parallel_safe_batch_len(&mcp, &calls), 0);
}

#[test]
fn parallel_batch_stops_at_mutating_tool() {
    let mcp = McpClient::new();
    // write_file / execute_command have side effects and cannot run in parallel; the batch must stop there.
    assert!(!is_parallel_safe_tool_call(&mcp, &tool_call("write_file")));
    assert!(!is_parallel_safe_tool_call(
        &mcp,
        &tool_call("execute_command")
    ));
    let calls = vec![tool_call("knowledge_search"), tool_call("write_file")];
    assert_eq!(parallel_safe_batch_len(&mcp, &calls), 1);
}

#[test]
fn parallel_batch_excludes_tools_without_readonly_policy() {
    let mcp = McpClient::new();
    // tree has no explicit read-only policy, so it must run sequentially.
    assert!(!is_parallel_safe_tool_call(&mcp, &tool_call("tree")));
}

#[test]
fn name_substrings_cannot_make_mutating_tools_parallel() {
    let mcp = McpClient::new();
    // `forget` contains `get`, and `read_mailbox` drains the mailbox: both
    // must stay serial despite their read-looking names.
    for name in ["knowledge_forget", "read_mailbox", "read_file"] {
        assert_eq!(
            tool_parallel_policy(name),
            ToolParallelPolicy::Serial,
            "{name} must be explicitly serial"
        );
        assert!(!is_parallel_safe_tool_call(&mcp, &tool_call(name)));
    }
}

#[test]
fn read_only_tool_policy_is_registry_driven() {
    let mcp = McpClient::new();
    for name in ["knowledge_search", "session_read", "web_fetch"] {
        assert_eq!(
            tool_parallel_policy(name),
            ToolParallelPolicy::ReadOnly,
            "{name} must be registered read-only"
        );
        assert!(is_parallel_safe_tool_call(&mcp, &tool_call(name)));
    }
}

#[test]
fn parallel_batch_caps_at_max_concurrency() {
    let mcp = McpClient::new();
    let calls: Vec<ToolCall> = (0..super::PARALLEL_READONLY_MAX_CONCURRENCY + 4)
        .map(|_| tool_call("knowledge_search"))
        .collect();
    assert_eq!(
        parallel_safe_batch_len(&mcp, &calls),
        super::PARALLEL_READONLY_MAX_CONCURRENCY
    );
}

#[test]
fn parallel_batch_not_formed_for_single_readonly_call() {
    let mcp = McpClient::new();
    let calls = vec![tool_call("knowledge_search"), tool_call("write_file")];
    assert_eq!(parallel_safe_batch_len(&mcp, &calls), 1);
}

#[test]
fn execute_tool_calls_rejects_tools_hidden_from_current_turn_schema() {
    let (_guard, kernel, root) = setup_tool_kernel();
    let path = std::env::temp_dir().join(format!(
        "turn-schema-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&path, "hello").unwrap();

    let mut call = tool_call("read_file");
    call.function.arguments = format!(r#"{{"file_path":"{}"}}"#, path.to_string_lossy());
    let allowed_tool_names: FastSet<String> = FastSet::default();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(McpClient::new()));
    let result = execute_tool_calls(
        "sess-turn-schema",
        &McpClient::new(),
        &shared_mcp,
        &[call],
        Some(&allowed_tool_names),
        None,
    )
    .unwrap();

    assert_eq!(result.tool_results.len(), 1);
    assert!(
        result.tool_results[0]
            .content
            .contains("not available in this turn's tool schema")
    );
    assert_eq!(
        kernel.lock().unwrap().rusage_get(root).unwrap().tool_calls,
        0
    );

    let _ = fs::remove_file(path);
}

#[test]
fn unavailable_tool_error_points_to_enable_tools_when_available() {
    let allowed_tool_names: FastSet<String> = ["enable_tools".to_string()].into_iter().collect();
    let call = tool_call("task_wait");

    let error = prepare_tool_call(&McpClient::new(), &call, Some(&allowed_tool_names)).unwrap_err();

    assert!(
        error
            .content
            .contains("not available in this turn's tool schema")
    );
    assert!(error.content.contains("enable_tools"));
    assert!(error.content.contains("lazily loaded"));
}

#[test]
fn unavailable_tool_error_omits_enable_hint_when_enable_tools_is_hidden() {
    let allowed_tool_names: FastSet<String> = FastSet::default();
    let call = tool_call("task_wait");

    let error = prepare_tool_call(&McpClient::new(), &call, Some(&allowed_tool_names)).unwrap_err();

    assert!(!error.content.contains("lazily loaded"));
    assert!(!error.content.contains("enable_tools"));
}

#[test]
fn execute_tool_calls_preflights_kernel_tool_quota_before_running_tool() {
    let (_guard, kernel, root) = setup_tool_kernel();
    {
        let mut os = kernel.lock().unwrap();
        let mut lim = ResourceLimit::unlimited();
        lim.max_tool_calls = 0;
        os.rlimit_set(root, lim).unwrap();
    }

    let path = std::env::temp_dir().join(format!(
        "tool-quota-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&path, "hello").unwrap();

    let mut call = tool_call("read_file");
    call.function.arguments = format!(r#"{{"file_path":"{}"}}"#, path.to_string_lossy());
    let allowed_tool_names: FastSet<String> = ["read_file".to_string()].into_iter().collect();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(McpClient::new()));
    let result = execute_tool_calls(
        "sess-tool-quota",
        &McpClient::new(),
        &shared_mcp,
        &[call],
        Some(&allowed_tool_names),
        None,
    )
    .unwrap();

    assert_eq!(result.tool_results.len(), 1);
    assert!(
        result.tool_results[0]
            .content
            .contains("kernel tool-call quota")
    );
    assert_eq!(
        kernel.lock().unwrap().rusage_get(root).unwrap().tool_calls,
        0
    );

    let _ = fs::remove_file(path);
}

#[test]
fn cacheable_tool_name_prefers_read_only_tools() {
    assert!(is_cacheable_tool_name("read_file"));
    assert!(!is_cacheable_tool_name("create_file"));
    assert!(!is_cacheable_tool_name("execute_command"));
}

#[test]
fn classify_tool_error_distinguishes_argument_and_transient_cases() {
    assert_eq!(
        classify_tool_error("failed to parse arguments: expected value"),
        ToolFailureKind::Argument
    );
    assert_eq!(
        classify_tool_error("request timeout while fetching data"),
        ToolFailureKind::Transient
    );
    assert_eq!(
        classify_tool_error("Error: execute_command canceled by user"),
        ToolFailureKind::Canceled
    );
}

#[test]
fn should_retry_once_only_for_explicitly_retry_safe_builtin_tools() {
    let builtin = ToolRoute::Builtin;
    let mcp = ToolRoute::Mcp {
        server_name: "demo".to_string(),
        tool_name: "read_file".to_string(),
    };
    assert!(should_retry_once(
        &builtin,
        "read_file",
        "timeout while reading"
    ));
    assert!(should_retry_once(
        &builtin,
        "knowledge_search",
        "timeout while reading"
    ));
    assert!(should_retry_once(
        &builtin,
        "web_fetch",
        "timeout while fetching"
    ));
    // Name substrings must not grant retry safety to mutating or
    // non-idempotent tools.
    assert!(!should_retry_once(
        &builtin,
        "knowledge_forget",
        "timeout while deleting"
    ));
    assert!(!should_retry_once(
        &builtin,
        "read_mailbox",
        "timeout while reading"
    ));
    assert!(!should_retry_once(
        &builtin,
        "task_evidence_read",
        "timeout while reading"
    ));
    assert!(!should_retry_once(
        &builtin,
        "execute_command",
        "timeout while reading"
    ));
    assert!(!should_retry_once(
        &builtin,
        "create_file",
        "timeout while writing"
    ));
    assert!(!should_retry_once(
        &mcp,
        "read_file",
        "timeout while reading"
    ));
}

#[test]
fn execute_with_safe_retry_retries_once_for_safe_transient_error() {
    let mut calls = 0usize;
    let result = execute_with_safe_retry(&ToolRoute::Builtin, "read_file", || {
        calls += 1;
        if calls == 1 {
            Err("request timed out".to_string())
        } else {
            Ok(crate::ai::types::ToolResult {
                task_report: None,
                tool_call_id: "tc-1".to_string(),
                content: "ok".to_string(),
            })
        }
    });
    assert!(result.is_ok());
    assert_eq!(calls, 2);
}

#[test]
fn execute_with_safe_retry_does_not_retry_non_safe_tools() {
    let mut calls = 0usize;
    let result = execute_with_safe_retry(&ToolRoute::Builtin, "create_file", || {
        calls += 1;
        Err("request timed out".to_string())
    });
    assert!(result.is_err());
    assert_eq!(calls, 1);
}

#[test]
fn tool_cache_key_is_stable_for_same_args() {
    let key1 = build_tool_cache_key("read_file", &json!({"path":"a","start":1}));
    let key2 = build_tool_cache_key("read_file", &json!({"path":"a","start":1}));
    let key3 = build_tool_cache_key("read_file", &json!({"path":"a","start":2}));
    assert_eq!(key1, key2);
    assert_ne!(key1, key3);
}

#[test]
fn tool_cache_entry_obeys_ttl() {
    let fresh = AgentMemoryEntry {
        id: None,
        timestamp: Utc::now().to_rfc3339(),
        category: "tool_cache".to_string(),
        note: "{}".to_string(),
        tags: Vec::new(),
        source: None,
        distilled: None,
        priority: Some(80),
        owner_pid: None,
        owner_pgid: None,
        image_path: None,
        model: None,
    };
    let stale = AgentMemoryEntry {
        timestamp: (Utc::now() - Duration::minutes(TOOL_CACHE_TTL_MINUTES + 1)).to_rfc3339(),
        ..fresh.clone()
    };
    assert!(is_tool_cache_entry_fresh(&fresh));
    assert!(!is_tool_cache_entry_fresh(&stale));
}

#[test]
fn tool_cache_requires_file_fingerprints() {
    let path = temp_file_path("tool_cache_requires_fingerprint");
    fs::write(&path, "hello").unwrap();

    let read_args = json!({
        "file_path": path.to_string_lossy(),
        "offset": 1,
        "limit": 10
    });
    assert!(should_store_or_load_tool_cache("read_file", &read_args));

    assert!(!should_store_or_load_tool_cache(
        "read_file",
        &json!({"file_path":"/path/that/does/not/exist"})
    ));
    assert!(!should_store_or_load_tool_cache(
        "knowledge_search",
        &json!({"query":"durable preference"})
    ));
    assert!(!should_store_or_load_tool_cache(
        "tree",
        &json!({"path": path.parent().unwrap().to_string_lossy()})
    ));

    let _ = fs::remove_file(path);
}

fn temp_file_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.push(format!(
        "rust_tools_{name}_{}_{}",
        std::process::id(),
        nanos
    ));
    path
}

fn with_tool_cache_store(check: impl FnOnce(super::MemoryStore, serde_json::Value)) {
    struct TestDir(std::path::PathBuf);
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let dir = TestDir(temp_file_path("tool_cache_store"));
    fs::create_dir(&dir.0).unwrap();
    let source = dir.0.join("source.txt");
    fs::write(&source, "source content").unwrap();
    let memory_path = dir.0.join("memory.jsonl");
    crate::ai::driver::runtime_ctx::SUBAGENT_MEMORY_PATH.sync_scope(memory_path.clone(), || {
        check(
            super::MemoryStore::for_tests_with_path(memory_path),
            json!({"file_path": source, "offset": 1, "limit": 10}),
        );
    });
}

fn store_test_tool_cache(args: &serde_json::Value, content: &str) -> ToolCall {
    let call = tool_call_with_args("read_file", &args.to_string());
    super::store_tool_cache_result(
        "cache-test",
        &call,
        args,
        &crate::ai::types::ToolResult {
            task_report: None,
            tool_call_id: call.id.clone(),
            content: content.to_string(),
        },
        &collect_tool_cache_file_fingerprints("read_file", args),
    );
    call
}

#[test]
fn tool_cache_roundtrip_preserves_content_and_validation() {
    with_tool_cache_store(|store, args| {
        let content = "Quoted: \"value\"\\path\nUnicode: 中文🙂\n[truncated: continue offset=11]";
        let mut call = store_test_tool_cache(&args, content);
        assert_eq!(store.recent(10).unwrap().len(), 1);
        call.id = "new-call-id".to_string();
        let cached = super::load_cached_tool_result("cache-test", &call, &args).unwrap();
        assert_eq!(cached.content, content);
        assert_eq!(cached.tool_call_id, call.id);
        assert!(super::load_cached_tool_result("other-session", &call, &args).is_none());
        let changed_args = json!({"file_path": args["file_path"], "offset": 2, "limit": 10});
        assert!(super::load_cached_tool_result("cache-test", &call, &changed_args).is_none());
        fs::write(
            args["file_path"].as_str().unwrap(),
            "changed source content",
        )
        .unwrap();
        assert!(super::load_cached_tool_result("cache-test", &call, &args).is_none());
    });
}

#[test]
fn tool_cache_enforces_serialized_byte_boundary_without_truncation() {
    with_tool_cache_store(|store, args| {
        let payload = ToolCachePayload {
            tool_name: "read_file".to_string(),
            args: args.clone(),
            result: String::new(),
            file_fingerprints: collect_tool_cache_file_fingerprints("read_file", &args),
        };
        let budget = super::MAX_NOTE_BYTES - serde_json::to_string(&payload).unwrap().len();
        for content in [
            "a".repeat(budget + 1),
            "中".repeat(budget / 3 + 1),
            "\n".repeat(budget / 2 + 1),
            "\0".repeat(budget / 6 + 1),
            "a".repeat(12_001),
        ] {
            let call = store_test_tool_cache(&args, &content);
            assert!(store.recent(10).unwrap().is_empty());
            assert!(super::load_cached_tool_result("cache-test", &call, &args).is_none());
        }

        let content = "a".repeat(budget);
        let call = store_test_tool_cache(&args, &content);
        let entries = store.recent(10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].note.len(), super::MAX_NOTE_BYTES);
        let cached = super::load_cached_tool_result("cache-test", &call, &args).unwrap();
        assert_eq!(cached.content, content);
    });
}

#[test]
fn tool_cache_skips_corrupt_legacy_notes_without_hiding_valid_entries() {
    for oversized in [false, true] {
        with_tool_cache_store(|store, args| {
            let call = store_test_tool_cache(&args, "complete result");
            let mut broken = store.recent(10).unwrap().remove(0);
            broken.id = None;
            broken.note = if oversized {
                let mut payload: ToolCachePayload = serde_json::from_str(&broken.note).unwrap();
                payload.result = "x".repeat(super::MAX_NOTE_BYTES);
                serde_json::to_string(&payload).unwrap()
            } else {
                "{broken JSON".to_string()
            };
            store.append(&broken).unwrap();
            let entries = store.recent(10).unwrap();
            assert_eq!(entries.len(), 2);
            assert!(serde_json::from_str::<ToolCachePayload>(&entries[0].note).is_err());
            if oversized {
                assert!(
                    entries[0]
                        .note
                        .contains("[note truncated to fit memory store cap]")
                );
            }
            let cached = super::load_cached_tool_result("cache-test", &call, &args).unwrap();
            assert_eq!(cached.content, "complete result");
        });
    }
}

#[test]
fn tool_cache_does_not_store_empty_or_failed_results() {
    with_tool_cache_store(|store, args| {
        for content in ["", " \n\t", "Error: failed to read file"] {
            let call = store_test_tool_cache(&args, content);
            assert!(store.recent(10).unwrap().is_empty());
            assert!(super::load_cached_tool_result("cache-test", &call, &args).is_none());
        }
    });
}

#[test]
fn tool_cache_rejects_legacy_metadata_only_fingerprints_and_unavailable_sources() {
    with_tool_cache_store(|store, args| {
        let call = store_test_tool_cache(&args, "complete result");
        let mut entry = store.recent(10).unwrap().remove(0);
        let mut note: serde_json::Value = serde_json::from_str(&entry.note).unwrap();
        for fingerprint in note["file_fingerprints"].as_array_mut().unwrap() {
            fingerprint
                .as_object_mut()
                .unwrap()
                .remove("content_sha256");
            fingerprint
                .as_object_mut()
                .unwrap()
                .remove("validated_bytes");
        }
        let legacy: ToolCachePayload = serde_json::from_value(note.clone()).unwrap();
        assert!(!tool_cache_validation_matches(&legacy));
        fs::write(store.path(), "").unwrap();
        entry.id = None;
        entry.note = note.to_string();
        store.append(&entry).unwrap();
        assert!(super::load_cached_tool_result("cache-test", &call, &args).is_none());
        fs::remove_file(args["file_path"].as_str().unwrap()).unwrap();
        assert!(!tool_cache_validation_matches(&legacy));
        assert!(super::load_cached_tool_result("cache-test", &call, &args).is_none());
    });
}

#[test]
fn tool_cache_does_not_certify_a_result_after_source_changes_during_execution() {
    with_tool_cache_store(|store, args| {
        let call = tool_call_with_args("read_file", &args.to_string());
        let before = collect_tool_cache_file_fingerprints("read_file", &args);
        let content = crate::ai::tools::service::file::execute_read_file(&args).unwrap();
        let path = args["file_path"].as_str().unwrap();
        let mtime = fs::metadata(path).unwrap().modified().unwrap();
        fs::write(path, "changed source").unwrap();
        fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(mtime))
            .unwrap();
        assert_eq!(before[0].size, fs::metadata(path).unwrap().len());
        super::store_tool_cache_result(
            "cache-test",
            &call,
            &args,
            &crate::ai::types::ToolResult {
                task_report: None,
                tool_call_id: call.id.clone(),
                content,
            },
            &before,
        );
        assert!(store.recent(10).unwrap().is_empty());
        assert!(super::load_cached_tool_result("cache-test", &call, &args).is_none());
    });
}

#[test]
fn tool_cache_large_file_validation_is_bounded_and_checks_returned_content() {
    use std::io::{Seek, SeekFrom, Write};
    with_tool_cache_store(|_, mut args| {
        let path = args["file_path"].as_str().unwrap().to_string();
        fs::write(&path, "version-0\n".repeat(1_000_000)).unwrap();
        args["offset"] = json!(100);
        args["limit"] = json!(5);
        let fingerprints = collect_tool_cache_file_fingerprints("read_file", &args);
        assert_eq!(fingerprints.len(), 1);
        let validated = fingerprints[0].validated_bytes;
        assert!(
            validated < fingerprints[0].size,
            "do not scan the unrelated large-file tail"
        );
        assert!(validated >= 2 * crate::ai::tools::service::file::WINDOW_READ_CHUNK_BYTES as u64);
        let content = crate::ai::tools::service::file::execute_read_file(&args).unwrap();
        let call = store_test_tool_cache(&args, &content);
        assert_eq!(
            super::load_cached_tool_result("cache-test", &call, &args)
                .unwrap()
                .content,
            content
        );
        let mut file = fs::File::options().write(true).open(&path).unwrap();
        let mtime = fs::metadata(&path).unwrap().modified().unwrap();
        // A same-size, same-mtime change outside the dependency prefix leaves this window valid.
        file.seek(SeekFrom::Start(validated + 100)).unwrap();
        file.write_all(b"version-1").unwrap();
        file.set_times(fs::FileTimes::new().set_modified(mtime))
            .unwrap();
        assert_eq!(
            super::load_cached_tool_result("cache-test", &call, &args)
                .unwrap()
                .content,
            content
        );
        // The returned line must invalidate even with identical metadata.
        file.seek(SeekFrom::Start(99 * 10)).unwrap();
        file.write_all(b"version-1").unwrap();
        file.set_times(fs::FileTimes::new().set_modified(mtime))
            .unwrap();
        assert!(super::load_cached_tool_result("cache-test", &call, &args).is_none());
        let changed = crate::ai::tools::service::file::execute_read_file(&args).unwrap();
        assert!(changed.contains("version-1"));
    });
}

#[test]
fn file_backed_cache_validation_rejects_stale_entries() {
    let path = temp_file_path("tool_cache_validation");
    fs::write(&path, "hello").unwrap();

    let args = json!({
        "file_path": path.to_string_lossy(),
        "offset": 1,
        "limit": 10
    });
    let payload = ToolCachePayload {
        tool_name: "read_file".to_string(),
        args: args.clone(),
        result: "cached".to_string(),
        file_fingerprints: collect_tool_cache_file_fingerprints("read_file", &args),
    };
    assert!(tool_cache_validation_matches(&payload));

    fs::write(&path, "hello, updated").unwrap();
    assert!(!tool_cache_validation_matches(&payload));

    let _ = fs::remove_file(path);
}

#[test]
fn legacy_file_cache_entries_without_fingerprint_are_rejected() {
    let path = temp_file_path("tool_cache_legacy");
    fs::write(&path, "hello").unwrap();

    let args = json!({
        "file_path": path.to_string_lossy(),
        "offset": 1,
        "limit": 10
    });
    let payload = ToolCachePayload {
        tool_name: "read_file".to_string(),
        args,
        result: "cached".to_string(),
        file_fingerprints: Vec::<CachedFileFingerprint>::new(),
    };

    assert!(!tool_cache_validation_matches(&payload));

    let _ = fs::remove_file(path);
}

fn tool_call(name: &str) -> ToolCall {
    ToolCall {
        id: format!("call-{name}"),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: "{}".to_string(),
        },
    }
}

fn tool_call_with_args(name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: format!("call-{name}"),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    }
}

#[test]
fn overflow_stub_argument_guard_catches_minimal_stubs() {
    let mcp = McpClient::new();
    for arguments in [
        r#"{"_context_overflow_truncated":true,"original_chars":900,"archive_file_path":"/tmp/archive.md"}"#,
        r#"{"_context_overflow_truncated":true,"original_chars":900,"archive_file_path":null}"#,
        r#"{"_context_overflow_truncated":"true","original_chars":"900","archive_file_path":"/tmp/archive.md"}"#,
    ] {
        let pointer: serde_json::Value = serde_json::from_str(arguments).unwrap();
        for name in ["apply_patch", "write_file", "read_file"] {
            let call = tool_call_with_args(name, arguments);
            let error = super::prepare_tool_call(&mcp, &call, None)
                .expect_err("minimal archive metadata must be rejected before dispatch");
            assert_eq!(error.tool_call_id, call.id);
            assert!(error.content.contains("context-overflow pointer stub"));
            if let Some(path) = pointer["archive_file_path"].as_str() {
                assert!(error.content.contains(path));
            } else {
                assert!(error.content.contains("This stub has no archive path."));
            }
            assert!(
                error
                    .content
                    .contains("This call was not executed; no file changed.")
            );
            assert!(
                error
                    .content
                    .contains("re-read the target file's current state")
            );
            assert!(error.content.contains("not the whole overflow archive"));
        }
    }
}

#[test]
fn overflow_stub_argument_guard_catches_transcribed_stubs() {
    // Stringified archive markers must be rejected centrally before they reach
    // individual tool parsers.
    let mcp = McpClient::new();
    let string_stub = tool_call_with_args(
        "apply_patch",
        r#"{"_context_overflow_truncated": "true", "original_chars": "1622", "archive_file_path": "/tmp/archive.md", "preview": "{\"patch\": \"*** Begin P"}"#,
    );
    let err = super::prepare_tool_call(&mcp, &string_stub, None)
        .expect_err("stringified stub arguments must be rejected");
    assert!(err.content.contains("context-overflow pointer stub"));
    assert!(err.content.contains("/tmp/archive.md"));
    // A JSON bool marker with numeric original_chars is caught too.
    let runtime_stub = tool_call_with_args(
        "write_file",
        r#"{"_context_overflow_truncated": true, "original_chars": 900, "archive_file_path": "/tmp/a.md", "preview": "xyz"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &runtime_stub, None).is_err());
    // Marker-less pointer shape (all three archive-pointer keys) is also rejected.
    let keys_only = tool_call_with_args(
        "apply_patch",
        r#"{"original_chars": 10, "archive_file_path": "/tmp/b.md", "preview": "aa"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &keys_only, None).is_err());
    // Real arguments (including a legit `patch_file` recovery call) pass through.
    let real = tool_call_with_args(
        "apply_patch",
        r#"{"patch": "*** Begin Patch\n*** End Patch\n"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &real, None).is_ok());
    let recovery = tool_call_with_args("apply_patch", r#"{"patch_file": "/tmp/x.patch"}"#);
    assert!(super::prepare_tool_call(&mcp, &recovery, None).is_ok());
    // A false marker must not trip the stub guard; it is still rejected as an
    // undeclared top-level argument.
    let benign = tool_call_with_args(
        "apply_patch",
        r#"{"_context_overflow_truncated": false, "patch": "x"}"#,
    );
    let err = super::prepare_tool_call(&mcp, &benign, None)
        .expect_err("undeclared marker key must be rejected, but not as a stub");
    assert!(err.content.contains("unknown argument(s)"), "{err:?}");
    assert!(
        !err.content.contains("context-overflow pointer stub"),
        "{err:?}"
    );
}

#[test]
fn plan_update_not_found_gets_plan_specific_hint() {
    // Plan-not-found errors must reach the plan-specific branch before the
    // generic path branch so the hint repairs the plan step number.
    let err = "Step 6 not found in the active plan.";
    let hint = super::remediation_hint("plan_update", err, None).unwrap();
    assert!(
        hint.contains("Suggestion: reuse an existing step number"),
        "plan hint differs: {hint}"
    );
    assert!(hint.contains("plan-state.json"), "{hint}");
    assert!(!hint.contains("search/list tool"), "{hint}");

    // Generic file-not-found errors keep the original, file-oriented hint.
    let generic = super::remediation_hint("read_file", "no such file: /x/y", None).unwrap();
    assert!(
        generic.contains("verify the path or identifier"),
        "{generic}"
    );
    assert!(!generic.contains("plan-state.json"), "{generic}");
}

#[test]
fn user_cancellation_gets_stop_hint_not_generic_execute_hint() {
    // A denied/canceled action is a user decision, not a tool failure: the model
    // must get an explicit "don't retry" signal instead of the generic
    // execute_command hint ("break the command into smaller pieces") that would
    // otherwise nudge it into an analyze-and-retry loop after a git denial.
    let err = "Error: execute_command failed: git commit canceled by user";
    let hint = super::remediation_hint("execute_command", err, None).unwrap();
    assert!(hint.contains("Do not retry it"), "{hint}");
    assert!(
        !hint.contains("break the command into smaller pieces"),
        "{hint}"
    );

    // Non-canceled execute_command errors keep the generic hint, but only when
    // a fallback read tool is visible this turn (remediation_hint returns None
    // for an empty visibility set, so the unwrap here needs "read_file").
    let allowed: FastSet<String> = ["read_file".to_string()].into_iter().collect();
    let generic =
        super::remediation_hint("execute_command", "exit code 1", Some(&allowed)).unwrap();
    assert!(
        generic.contains("break the command into smaller pieces"),
        "{generic}"
    );
}

#[test]
fn schema_gate_rejects_wrong_typed_arguments_before_dispatch() {
    // A present-but-wrongly-typed value must fail at the central dispatch
    // point instead of reaching the service layer, where `as_u64().unwrap_or`
    // would silently coerce it into a default (e.g. `offset: "500"` reading
    // from line 1 while the model believes it read from line 500).
    let mcp = McpClient::new();
    let mistyped = tool_call_with_args(
        "read_file",
        r#"{"file_path": "src/main.rs", "offset": "500"}"#,
    );
    let err = super::prepare_tool_call(&mcp, &mistyped, None)
        .expect_err("wrong-typed offset must be rejected centrally");
    assert!(
        err.content.contains("invalid arguments for 'read_file'"),
        "{err:?}"
    );
    // Correctly typed arguments still pass through.
    let typed = tool_call_with_args(
        "read_file",
        r#"{"file_path": "src/main.rs", "offset": 500}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &typed, None).is_ok());
    // Missing required fields are rejected as well.
    let missing = tool_call_with_args("read_file", r#"{"offset": 500}"#);
    assert!(super::prepare_tool_call(&mcp, &missing, None).is_err());
}

#[test]
fn schema_gate_normalizes_compat_shapes_before_validating() {
    let mcp = McpClient::new();
    // Historical `path` alias for `file_path` (accepted by
    // `resolve_file_path_arg` / `optional_file_path_arg`) must pass the gate.
    let alias = tool_call_with_args("read_file", r#"{"path": "src/main.rs"}"#);
    assert!(super::prepare_tool_call(&mcp, &alias, None).is_ok());
    // Provider-materialized null for an optional field means "not provided".
    let null_optional = tool_call_with_args(
        "apply_patch",
        r#"{"patch": null, "patch_file": "/tmp/compat.patch"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &null_optional, None).is_ok());
    // Explicit null for a required field is still rejected.
    let null_required = tool_call_with_args("read_file", r#"{"file_path": null}"#);
    assert!(super::prepare_tool_call(&mcp, &null_required, None).is_err());
    // Unknown extra fields are rejected for builtin tools: the published
    // schema is the whole contract the model sees, and services silently
    // ignore undeclared keys.
    let extra = tool_call_with_args(
        "read_file",
        r#"{"file_path": "src/main.rs", "note": "junk"}"#,
    );
    let err = super::prepare_tool_call(&mcp, &extra, None)
        .expect_err("undeclared read_file key must be rejected centrally");
    assert!(err.content.contains("unknown argument(s)"), "{err:?}");
}

#[test]
fn schema_gate_resolves_all_declared_compat_aliases() {
    let mcp = McpClient::new();
    // `write_file` shares the historical `path` alias via
    // `resolve_file_path_arg`.
    let write_alias =
        tool_call_with_args("write_file", r#"{"path": "src/main.rs", "content": "hi"}"#);
    assert!(super::prepare_tool_call(&mcp, &write_alias, None).is_ok());
    // `apply_patch` accepts `path` via `optional_file_path_arg`.
    let patch_alias = tool_call_with_args(
        "apply_patch",
        r#"{"path": "/tmp/compat.patch", "patch_file": "/tmp/compat.patch"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &patch_alias, None).is_ok());
    // `send_side_note` aliases (`content`, `target_task_id`, `target`) are
    // honored by `execute_send_side_note` but undeclared in the schema.
    let side_note_alias = tool_call_with_args(
        "send_side_note",
        r#"{"content": "steer left", "target_task_id": "task-1"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &side_note_alias, None).is_ok());
    let side_note_target = tool_call_with_args(
        "send_side_note",
        r#"{"note": "steer left", "target": "task-1"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &side_note_target, None).is_ok());
    // `execute_command` accepts the shell-tool `cmd` spelling; the service
    // reads it as a fallback for `command`.
    let command_alias =
        tool_call_with_args("execute_command", r#"{"cmd": "printf ok", "pty": false}"#);
    let prepared =
        super::prepare_tool_call(&mcp, &command_alias, None).expect("cmd alias must pass the gate");
    assert_eq!(prepared.args["command"], "printf ok");
    assert!(prepared.args.get("cmd").is_none());
    // The canonical key wins when both spellings are present.
    let command_both = tool_call_with_args(
        "execute_command",
        r#"{"command": "printf canonical", "cmd": "printf alias", "pty": false}"#,
    );
    let prepared = super::prepare_tool_call(&mcp, &command_both, None)
        .expect("canonical command plus alias must pass the gate");
    assert_eq!(prepared.args["command"], "printf canonical");
    assert!(prepared.args.get("cmd").is_none());
    // The canonical key wins when both spellings are present; the alias must
    // not linger as an unknown field.
    let both = tool_call_with_args(
        "read_file",
        r#"{"file_path": "src/main.rs", "path": "other.rs"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &both, None).is_ok());
    // An alias for another tool's field stays unknown here: `content` is a
    // real `write_file` parameter, not a `read_file` alias for `note`.
    let foreign = tool_call_with_args(
        "read_file",
        r#"{"file_path": "src/main.rs", "content": "junk"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &foreign, None).is_err());
}

#[test]
fn schema_gate_rejects_silent_divergence_beyond_read_file() {
    let mcp = McpClient::new();
    // `write_file` ignores undeclared keys while overwriting: a model passing
    // `append: true` believes it appends but the service truncates.
    let append = tool_call_with_args(
        "write_file",
        r#"{"file_path": "src/main.rs", "content": "hi", "append": true}"#,
    );
    let err = super::prepare_tool_call(&mcp, &append, None)
        .expect_err("write_file append must be rejected centrally");
    assert!(err.content.contains("unknown argument(s)"), "{err:?}");
    // Negative integers pass `type: integer` but `as_u64().unwrap_or` silently
    // coerces them into defaults downstream.
    let negative_timeout = tool_call_with_args(
        "execute_command",
        r#"{"command": "ls", "pty": false, "timeout": -1}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &negative_timeout, None).is_err());
    let negative_depth = tool_call_with_args("tree", r#"{"max_depth": -1}"#);
    assert!(super::prepare_tool_call(&mcp, &negative_depth, None).is_err());
    // In-range values still pass.
    let sane_timeout = tool_call_with_args(
        "execute_command",
        r#"{"command": "ls", "pty": false, "timeout": 60}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &sane_timeout, None).is_ok());
    let sane_depth = tool_call_with_args("tree", r#"{"max_depth": 2}"#);
    assert!(super::prepare_tool_call(&mcp, &sane_depth, None).is_ok());
}

#[test]
fn execute_command_session_control_is_split_into_command_session() {
    let mcp = McpClient::new();
    // Session control belongs to `command_session`; `execute_command` must
    // reject its fields at the schema gate.
    let legacy = tool_call_with_args(
        "execute_command",
        r#"{"command": "ls", "pty": false, "session_id": ""}"#,
    );
    let err = super::prepare_tool_call(&mcp, &legacy, None)
        .expect_err("execute_command must reject session fields");
    assert!(err.content.contains("unknown argument(s)"), "{err:?}");
    assert!(err.content.contains("session_id"), "{err:?}");

    let missing_action = tool_call_with_args("command_session", r#"{"session_id": "cmd-1"}"#);
    assert!(
        super::prepare_tool_call(&mcp, &missing_action, None).is_err(),
        "command_session must require action"
    );

    let poll = tool_call_with_args(
        "command_session",
        r#"{"action": "poll", "session_id": "cmd-1"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &poll, None).is_ok());

    let bad_action = tool_call_with_args(
        "command_session",
        r#"{"action": "resume", "session_id": "cmd-1"}"#,
    );
    assert!(
        super::prepare_tool_call(&mcp, &bad_action, None).is_err(),
        "command_session must reject unknown actions at the schema gate"
    );
}

#[test]
fn schema_gate_rejects_negative_read_window_before_dispatch() {
    let mcp = McpClient::new();
    // Negative integers pass `type: integer`, but the service reads them via
    // `as_u64().unwrap_or(default)` and would silently coerce them into
    // defaults; the schema minimums close that hole at the gate.
    let negative_offset =
        tool_call_with_args("read_file", r#"{"file_path": "src/main.rs", "offset": -1}"#);
    assert!(super::prepare_tool_call(&mcp, &negative_offset, None).is_err());
    let zero_limit =
        tool_call_with_args("read_file", r#"{"file_path": "src/main.rs", "limit": 0}"#);
    assert!(super::prepare_tool_call(&mcp, &zero_limit, None).is_err());
    let sane_window = tool_call_with_args(
        "read_file",
        r#"{"file_path": "src/main.rs", "offset": 10, "limit": 20}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &sane_window, None).is_ok());
}

#[test]
fn schema_gate_rejects_out_of_range_integers_before_dispatch() {
    // Every optional integer below would otherwise pass `type: integer` and
    // then be silently coerced by the service layer (`as_u64().unwrap_or`,
    // `.clamp`, `.min`, or `as u8` truncation), so the model-requested value
    // and the executed value diverge. The schema bounds close that hole at
    // the gate with a retryable error.
    let mcp = McpClient::new();
    let cases = [
        ("task_wait", r#"{"task_ids": ["t"], "timeout_secs": -1}"#),
        ("task_wait", r#"{"task_ids": ["t"], "timeout_secs": 900}"#),
        ("search_overflow", r#"{"query": "x", "context_lines": -1}"#),
        ("search_overflow", r#"{"query": "x", "max_results": 0}"#),
        ("search_overflow", r#"{"query": "x", "max_results": 201}"#),
        ("knowledge_search", r#"{"query": "x", "limit": 0}"#),
        ("knowledge_list", r#"{"limit": 101}"#),
        (
            "knowledge_semantic_search",
            r#"{"query": "x", "limit": 21}"#,
        ),
        ("knowledge_save", r#"{"content": "x", "priority": 256}"#),
        ("knowledge_save", r#"{"content": "x", "priority": -1}"#),
        ("list_skills", r#"{"limit": 0}"#),
        ("sleep_process", r#"{"turns": 0}"#),
        (
            "spawn_process",
            r#"{"name": "p", "goal": "g", "priority": 300}"#,
        ),
        (
            "spawn_process",
            r#"{"name": "p", "goal": "g", "quota_turns": -1}"#,
        ),
        (
            "spawn_daemon",
            r#"{"name": "d", "goal": "g", "max_restarts": -1}"#,
        ),
        (
            "session_distill",
            r#"{"archive": "/tmp/a.zip", "limit": 101}"#,
        ),
    ];
    for (tool, args) in cases {
        let call = tool_call_with_args(tool, args);
        assert!(
            super::prepare_tool_call(&mcp, &call, None).is_err(),
            "{tool} must reject out-of-range integer: {args}"
        );
    }
    // Boundary values the services honor verbatim still pass the gate: 0
    // context lines, unlimited quota (0), no restarts (0), and the documented
    // timeout ceiling are all meaningful calls.
    let ok_cases = [
        ("task_wait", r#"{"task_ids": ["t"], "timeout_secs": 60}"#),
        ("search_overflow", r#"{"query": "x", "context_lines": 0}"#),
        (
            "spawn_process",
            r#"{"name": "p", "goal": "g", "quota_turns": 0}"#,
        ),
        (
            "spawn_daemon",
            r#"{"name": "d", "goal": "g", "max_restarts": 0}"#,
        ),
    ];
    for (tool, args) in ok_cases {
        let call = tool_call_with_args(tool, args);
        assert!(
            super::prepare_tool_call(&mcp, &call, None).is_ok(),
            "{tool} must accept meaningful boundary value: {args}"
        );
    }
}

/// Registered contract for every builtin integer argument: (tool, path,
/// minimum, maximum, why). `path` uses `.` for object fields and `[]` for
/// array items (for example, `days[]` or `steps[].step`). A missing minimum
/// requires fail-closed parsing or signed semantics; a missing maximum may
/// reflect an unclamped service. The test below walks grouped builtin schemas,
/// so a new integer parameter fails until its author registers it here.
const INT_BOUND_REGISTRY: &[(&str, &str, Option<i64>, Option<i64>, &str)] = &[
    (
        "execute_command",
        "timeout",
        Some(1),
        Some(300),
        "service clamps 1-300",
    ),
    (
        "execute_command",
        "yield_time_ms",
        Some(5000),
        Some(60000),
        "service clamps 5000-60000; each poll is a model round",
    ),
    (
        "command_session",
        "yield_time_ms",
        Some(5000),
        Some(60000),
        "service clamps 5000-60000; each poll is a model round",
    ),
    (
        "task_wait",
        "timeout_secs",
        Some(1),
        Some(60),
        "service clamps 1-60",
    ),
    (
        "search_overflow",
        "context_lines",
        Some(0),
        Some(5),
        "unwrap_or(2).min(5); 0 is valid",
    ),
    (
        "search_overflow",
        "max_results",
        Some(1),
        Some(200),
        "service clamps 1-200",
    ),
    (
        "knowledge_search",
        "limit",
        Some(1),
        None,
        "no upper clamp; 0 would return empty",
    ),
    (
        "knowledge_list",
        "limit",
        Some(1),
        Some(100),
        "unwrap_or(20).min(100)",
    ),
    (
        "knowledge_semantic_search",
        "limit",
        Some(1),
        Some(20),
        "service clamps 1-20",
    ),
    (
        "knowledge_save",
        "priority",
        Some(0),
        Some(255),
        "u8 range; service rejects >255",
    ),
    (
        "knowledge_consolidate",
        "save_entries[].priority",
        Some(0),
        Some(255),
        "same u8 parser",
    ),
    (
        "list_skills",
        "limit",
        Some(1),
        Some(100),
        "service clamps 1-100",
    ),
    (
        "list_skill_resources",
        "limit",
        Some(1),
        Some(200),
        "pre-existing bound, pinned",
    ),
    (
        "read_skill_resource",
        "limit",
        Some(1),
        Some(65536),
        "pre-existing bound, pinned",
    ),
    (
        "read_file",
        "offset",
        Some(1),
        None,
        "no upper clamp in service",
    ),
    (
        "read_file",
        "char_offset",
        Some(0),
        None,
        "0 is valid; no upper clamp",
    ),
    (
        "read_file",
        "limit",
        Some(1),
        None,
        "no upper clamp in service",
    ),
    (
        "run_agent_graph",
        "max_parallel",
        Some(1),
        Some(8),
        "pre-existing bound, pinned",
    ),
    (
        "run_agent_graph",
        "static_graph.nodes[].max_retries",
        Some(0),
        None,
        "pre-existing, pinned",
    ),
    (
        "run_agent_graph",
        "dynamic_graph.policy.min_selected",
        Some(2),
        None,
        "pre-existing, pinned",
    ),
    (
        "run_agent_graph",
        "dynamic_graph.policy.max_selected",
        Some(2),
        None,
        "pre-existing, pinned",
    ),
    (
        "show_changes",
        "limit_snippet",
        Some(0),
        Some(4000),
        "pre-existing bound, pinned",
    ),
    (
        "show_changes",
        "since_seq",
        Some(0),
        None,
        "inclusive persistent mutation seq; no upper bound",
    ),
    (
        "show_changes",
        "until_seq",
        Some(0),
        None,
        "inclusive persistent mutation seq; no upper bound",
    ),
    (
        "show_changes",
        "turn_id",
        Some(0),
        None,
        "exact turn provenance filter; no upper bound",
    ),
    (
        "show_changes",
        "after_seq",
        Some(0),
        None,
        "exclusive history cursor; no upper bound",
    ),
    (
        "show_changes",
        "history_limit",
        Some(1),
        Some(100),
        "service clamps 1-100",
    ),
    (
        "show_changes",
        "history_diff_lines",
        Some(0),
        Some(100),
        "service clamps 0-100; 0 omits diffs",
    ),
    (
        "sleep_process",
        "turns",
        Some(1),
        None,
        "documented minimum 1; no upper clamp",
    ),
    (
        "spawn_process",
        "priority",
        Some(0),
        Some(255),
        "u8 range; larger values truncate via `as u8`",
    ),
    (
        "spawn_process",
        "quota_turns",
        Some(0),
        None,
        "0 means unlimited per ResourceLimit::from_legacy",
    ),
    (
        "spawn_daemon",
        "priority",
        Some(0),
        Some(255),
        "u8 range, as above",
    ),
    (
        "spawn_daemon",
        "quota_turns",
        Some(0),
        None,
        "0 means unlimited, as above",
    ),
    (
        "spawn_daemon",
        "max_restarts",
        Some(0),
        None,
        "0 disables restarts; no upper clamp",
    ),
    (
        "session_distill",
        "limit",
        Some(1),
        Some(100),
        "service clamps 1-100",
    ),
    (
        "session_list",
        "limit",
        Some(0),
        Some(100),
        "unwrap_or(20).min(100); 0 returns no rows",
    ),
    (
        "session_read",
        "limit",
        Some(0),
        Some(200),
        "unwrap_or(default).min(200); 0 returns no messages",
    ),
    (
        "session_read",
        "from",
        Some(0),
        None,
        "0 is the oldest canonical index; no upper clamp",
    ),
    (
        "session_search",
        "max_results",
        Some(0),
        Some(100),
        "unwrap_or(20).min(100); 0 stops immediately",
    ),
    (
        "tree",
        "max_depth",
        Some(0),
        Some(6),
        "pre-existing bound, pinned",
    ),
    (
        "manage_team",
        "budget.max_parallel",
        Some(1),
        Some(8),
        "validate_budget enforces 1-8",
    ),
    (
        "manage_team",
        "budget.max_tasks",
        Some(1),
        Some(512),
        "validate_budget enforces 1-512",
    ),
    (
        "manage_team",
        "budget.max_total_attempts",
        Some(1),
        Some(4096),
        "validate_budget enforces <=4096",
    ),
    (
        "manage_team",
        "budget.max_messages",
        Some(1),
        Some(2048),
        "validate_budget enforces 1-2048",
    ),
    (
        "manage_team",
        "lease_secs",
        Some(1),
        Some(86400),
        "service clamps 1-86400",
    ),
    (
        "phone_action",
        "days",
        Some(1),
        Some(30),
        "calendar_read rejects values outside 1-30",
    ),
    (
        "phone_action",
        "max",
        Some(1),
        Some(20),
        "calendar_read and note_list enforce 1-20",
    ),
    (
        "phone_action",
        "duration_min",
        Some(5),
        Some(600),
        "calendar_add rejects durations outside 5-600",
    ),
    (
        "phone_action",
        "hour",
        Some(0),
        Some(23),
        "alarm requires an integer hour in 0-23",
    ),
    (
        "phone_action",
        "minute",
        Some(0),
        Some(59),
        "alarm requires an integer minute in 0-59",
    ),
    (
        "phone_action",
        "weekdays[]",
        Some(1),
        Some(7),
        "alarm validates each unique ISO weekday in 1-7",
    ),
    (
        "phone_action",
        "seconds",
        Some(1),
        Some(86400),
        "timer requires an integer duration in 1-86400",
    ),
    (
        "phone_action",
        "offset",
        Some(0),
        Some(1048576),
        "file_read enforces MAX_FILE_CHARS; note_list enforces the narrower action-specific maximum 100000",
    ),
    (
        "phone_action",
        "limit",
        Some(1),
        Some(3000),
        "file_read enforces 1-3000 Unicode codepoints",
    ),
    (
        "kill_process",
        "pid",
        None,
        None,
        "service ok_or fails closed; no silent default",
    ),
    (
        "reap_process",
        "pid",
        None,
        None,
        "service ok_or fails closed; no silent default",
    ),
    (
        "wait_process",
        "pid",
        None,
        None,
        "service ok_or fails closed; no silent default",
    ),
    (
        "signal_process",
        "pid",
        None,
        None,
        "service ok_or fails closed; no silent default",
    ),
    (
        "send_ipc_message",
        "pid",
        None,
        None,
        "service ok_or fails closed; no silent default",
    ),
    (
        "set_process_group",
        "pid",
        None,
        None,
        "service ok_or fails closed; no silent default",
    ),
    (
        "set_process_group",
        "pgid",
        None,
        None,
        "service ok_or fails closed; no silent default",
    ),
    (
        "signal_process_group",
        "pgid",
        None,
        None,
        "service ok_or fails closed; no silent default",
    ),
    (
        "plan",
        "steps[].step",
        None,
        None,
        "parse_step_specs ok_or fails closed; labels only",
    ),
    (
        "plan_update",
        "step",
        None,
        None,
        "service ok_or fails closed; unknown steps report not-found",
    ),
    (
        "save_skill",
        "priority",
        None,
        None,
        "signed as_i64; negative precedence is valid",
    ),
    (
        "save_skill",
        "subskills[].priority",
        None,
        None,
        "signed as_i64; negative precedence is valid",
    ),
];

fn collect_int_bounds(
    schema: &serde_json::Value,
    prefix: String,
    out: &mut Vec<(String, Option<i64>, Option<i64>)>,
) {
    // Visit the node itself: an array's items can be an integer schema,
    // not just an object containing integer properties.
    if schema.get("type").and_then(|v| v.as_str()) == Some("integer") {
        out.push((
            prefix.clone(),
            schema.get("minimum").and_then(|v| v.as_i64()),
            schema.get("maximum").and_then(|v| v.as_i64()),
        ));
    }
    if let Some(props) = schema.get("properties").and_then(|v| v.as_object()) {
        for (name, prop) in props {
            let path = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}.{name}")
            };
            collect_int_bounds(prop, path, out);
        }
    }
    if let Some(items) = schema.get("items") {
        collect_int_bounds(items, format!("{prefix}[]"), out);
    }
}

#[test]
fn collect_int_bounds_visits_root_integers() {
    let mut found = Vec::new();
    collect_int_bounds(
        &serde_json::json!({"type": "integer", "minimum": -2, "maximum": 3}),
        String::new(),
        &mut found,
    );
    assert_eq!(found, vec![(String::new(), Some(-2), Some(3))]);
}

#[test]
fn collect_int_bounds_preserves_nested_paths_and_optional_bounds() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 0},
            "ids": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 7}},
            "matrix": {"type": "array", "items": {
                "type": "array", "items": {"type": "integer", "maximum": 9}
            }},
            "options": {"type": "object", "properties": {
                "priority": {"type": "integer"},
                "label": {"type": "string"}
            }},
            "rows": {"type": "array", "items": {
                "type": "object", "properties": {
                    "rank": {"type": "integer", "minimum": -5, "maximum": 5},
                    "tags": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "ratio": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "default": {"type": "integer", "minimum": 99}
    });
    let mut found = Vec::new();
    collect_int_bounds(&schema, String::new(), &mut found);
    found.sort();
    assert_eq!(
        found,
        vec![
            ("count".to_string(), Some(0), None),
            ("ids[]".to_string(), Some(1), Some(7)),
            ("matrix[][]".to_string(), None, Some(9)),
            ("options.priority".to_string(), None, None),
            ("rows[].rank".to_string(), Some(-5), Some(5)),
        ]
    );
}

#[test]
fn builtin_integer_schemas_match_registered_bounds() {
    use crate::ai::tools::{ToolGroup, tool_definitions_for_groups};
    use std::collections::BTreeSet;

    let mut unregistered = Vec::new();
    let mut drifted = Vec::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    // `ToolGroup::ALL` walks every grouped builtin tool; the only ungrouped
    // tool (`request_user_input`) declares no integer arguments, so it cannot
    // hide an unregistered bound.
    for def in tool_definitions_for_groups(ToolGroup::ALL) {
        let name = def.function.name.as_str();
        let schema = &def.function.parameters;
        let mut found = Vec::new();
        collect_int_bounds(schema, String::new(), &mut found);
        for (path, min, max) in found {
            seen.insert((name.to_string(), path.clone()));
            match INT_BOUND_REGISTRY
                .iter()
                .find(|(tool, prop, _, _, _)| *tool == name && *prop == path)
            {
                None => unregistered.push(format!("{name}.{path}")),
                Some((_, _, expected_min, expected_max, _)) => {
                    if *expected_min != min || *expected_max != max {
                        drifted.push(format!(
                            "{name}.{path}: schema=({min:?},{max:?}) registered=({expected_min:?},{expected_max:?})"
                        ));
                    }
                }
            }
        }
    }
    let stale: Vec<String> = INT_BOUND_REGISTRY
        .iter()
        .filter(|(tool, prop, _, _, _)| !seen.contains(&(tool.to_string(), prop.to_string())))
        .map(|(tool, prop, _, _, _)| format!("{tool}.{prop}"))
        .collect();
    assert!(
        unregistered.is_empty() && drifted.is_empty() && stale.is_empty(),
        "integer bound registry mismatch:\nnew unregistered: {unregistered:?}\ndrifted: {drifted:?}\nstale entries: {stale:?}"
    );
}

#[test]
fn search_overflow_description_stays_in_sync_with_implementation() {
    // Compare the documented default with the parser constant and pin the
    // expected hit-reason legend so drift fails this test.
    let raw = include_str!("../../tool_descriptions/search_overflow.json");
    let schema: serde_json::Value =
        serde_json::from_str(raw).expect("search_overflow.json must stay valid JSON");
    let case_sensitive_desc = schema["parameters"]["properties"]["case_sensitive"]["description"]
        .as_str()
        .unwrap_or_default();

    // The published default is compared against the value the parser actually
    // applies, not merely pinned on its own: a check that only reads the
    // description stays green when the implementation flips, so the two sides
    // are derived and compared.
    let impl_default = crate::ai::tools::SEARCH_OVERFLOW_CASE_SENSITIVE_DEFAULT;
    assert!(
        !(case_sensitive_desc.contains("default: false")
            && case_sensitive_desc.contains("default: true")),
        "the case_sensitive description must not state two different defaults: {case_sensitive_desc}"
    );
    let documented_default = if case_sensitive_desc.contains("default: false") {
        false
    } else if case_sensitive_desc.contains("default: true") {
        true
    } else {
        panic!("the case_sensitive description must state its default: {case_sensitive_desc}");
    };
    assert_eq!(
        documented_default, impl_default,
        "the documented case_sensitive default drifted from the implementation default"
    );
    // The "retry with case_sensitive=false" advice became obsolete the moment
    // false became the default; a stale copy would again mislead the model.
    assert!(
        !raw.contains("retry with case_sensitive=false"),
        "obsolete case_sensitive=false retry advice still present in search_overflow.json"
    );

    // The hit-reason markers the renderer emits (>> phrase / >t term / >b CJK
    // fragment) must be documented so the model reads a >b line as a near-miss
    // instead of proof that the wording is present.
    assert!(
        raw.contains(">> whole-query phrase hit") && raw.contains(">b CJK-fragment hit"),
        "hit-reason legend missing from search_overflow.json"
    );

    // The legend must also state the matching mode: under the case-insensitive
    // default the archived casing of a hit may differ from the query, and a
    // legend that omits this reads as "the wording is present exactly as typed".
    assert!(
        raw.contains("matching ignores case by default"),
        "hit-reason legend must state the case-insensitive default"
    );

    // A regex query inherits the same default, so its description must say so
    // instead of letting the model assume a pattern is matched verbatim.
    let is_regex_desc = schema["parameters"]["properties"]["is_regex"]["description"]
        .as_str()
        .unwrap_or_default();
    assert!(
        is_regex_desc.contains("case-insensitively"),
        "is_regex description must state the case-insensitive default: {is_regex_desc}"
    );

    // The parameter set is a stable model-facing contract: silently adding or
    // removing a parameter must fail this test rather than change the surface
    // the model can call.
    let keys: std::collections::BTreeSet<String> = schema["parameters"]["properties"]
        .as_object()
        .expect("parameters.properties must be an object")
        .keys()
        .cloned()
        .collect();
    let expected: std::collections::BTreeSet<String> = [
        "query",
        "scope",
        "file_pattern",
        "is_regex",
        "case_sensitive",
        "context_lines",
        "max_results",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    assert_eq!(
        keys, expected,
        "search_overflow parameter set drifted from the documented contract"
    );
}

#[test]
fn mcp_arg_validators_are_reused_per_schema_text() {
    // MCP tool calls arrive one per model round; compiling the server schema on
    // every call repeats work whose result cannot differ while the schema text
    // is unchanged. Reuse is keyed by that text, not by tool name, because a
    // reconnect can swap the schema behind the same name.
    let schema = json!({
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    });
    let first = super::mcp_arg_validator("mcp__fs__read", &schema).expect("schema is valid");
    let second = super::mcp_arg_validator("mcp__fs__read", &schema).expect("schema is valid");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "an unchanged schema must reuse the compiled validator"
    );
    // Same schema text under a different tool name is still the same contract.
    let shared = super::mcp_arg_validator("mcp__other__read", &schema).expect("schema is valid");
    assert!(std::sync::Arc::ptr_eq(&first, &shared));

    // A changed schema for the same tool name must NOT reuse the old validator:
    // it is a different server-side contract.
    let changed = json!({
        "type": "object",
        "properties": {"path": {"type": "integer"}},
        "required": ["path"],
    });
    let third = super::mcp_arg_validator("mcp__fs__read", &changed).expect("schema is valid");
    assert!(!std::sync::Arc::ptr_eq(&first, &third));
    assert!(first.is_valid(&json!({"path": "x"})));
    assert!(third.is_valid(&json!({"path": 1})));
    assert!(!third.is_valid(&json!({"path": "x"})));
}
