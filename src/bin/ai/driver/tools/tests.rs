use super::{
    CachedFileFingerprint, TOOL_CACHE_TTL_MINUTES, ToolCachePayload, ToolFailureKind, ToolRoute,
    build_tool_cache_key, classify_tool_error, collect_tool_cache_file_fingerprints,
    execute_tool_calls, execute_with_safe_retry, is_cacheable_tool_name,
    is_parallel_safe_tool_call, is_tool_cache_entry_fresh, parallel_safe_batch_len,
    should_retry_once, should_store_or_load_tool_cache, tool_cache_validation_matches,
};
use crate::ai::mcp::McpClient;
use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
use crate::ai::types::{FunctionCall, ToolCall};
use aios_kernel::primitives::ResourceLimit;
use chrono::{Duration, Utc};
use rust_tools::commonw::FastSet;
use serde_json::json;
use std::fs;
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

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
fn parallel_batch_excludes_non_cacheable_tools() {
    let mcp = McpClient::new();
    // tree is not in the read-only reusable whitelist, so it must run sequentially.
    assert!(!is_parallel_safe_tool_call(&mcp, &tool_call("tree")));
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
    // Only 1 parallel-safe call exists, so the caller falls back to the sequential path (batch_len == 1 < 2).
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
fn should_retry_once_only_for_safe_builtin_read_only_tools() {
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
fn overflow_stub_argument_guard_catches_transcribed_stubs() {
    // Stringified stub as a model transcribed it after seeing the compressor's
    // pointer in context (observed for apply_patch and write_file in production
    // sessions): must be rejected centrally, not reach the tool's own parser.
    let mcp = McpClient::new();
    let string_stub = tool_call_with_args(
        "apply_patch",
        r#"{"_context_overflow_truncated": "true", "original_chars": "1622", "archive_file_path": "/tmp/archive.md", "preview": "{\"patch\": \"*** Begin P"}"#,
    );
    let err = super::prepare_tool_call(&mcp, &string_stub, None)
        .expect_err("stringified stub arguments must be rejected");
    assert!(err.content.contains("context-overflow pointer stub"));
    assert!(err.content.contains("/tmp/archive.md"));
    // Runtime-native shape (JSON bool marker + numeric original_chars) is caught too.
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
    let recovery = tool_call_with_args(
        "apply_patch",
        r#"{"patch_file": "/tmp/x.patch"}"#,
    );
    assert!(super::prepare_tool_call(&mcp, &recovery, None).is_ok());
    // A marker key that is not true-ish (e.g. a nested value named like the
    // marker with boolean false) must not trip the stub guard: it is rejected
    // as an undeclared argument instead, never as a pointer stub.
    let benign = tool_call_with_args(
        "apply_patch",
        r#"{"_context_overflow_truncated": false, "patch": "x"}"#,
    );
    let err = super::prepare_tool_call(&mcp, &benign, None)
        .expect_err("undeclared marker key must be rejected, but not as a stub");
    assert!(err.content.contains("unknown argument(s)"), "{err:?}");
    assert!(!err.content.contains("context-overflow pointer stub"), "{err:?}");
}

#[test]
fn plan_update_not_found_gets_plan_specific_hint() {
    // "Step 6 not found in the active plan." matches both the plan branch and the
    // generic "not found" branch; the plan branch must win (it is ordered first), or
    // the model receives a misleading file-path hint after context compression folded
    // the plan-creation turns (regression: hallucinated step 6/5 updates).
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
    assert!(generic.contains("verify the path or identifier"), "{generic}");
    assert!(!generic.contains("plan-state.json"), "{generic}");
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
    assert!(err.content.contains("invalid arguments for 'read_file'"), "{err:?}");
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
    let write_alias = tool_call_with_args(
        "write_file",
        r#"{"path": "src/main.rs", "content": "hi"}"#,
    );
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
    let negative_depth =
        tool_call_with_args("tree", r#"{"max_depth": -1}"#);
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
        ("knowledge_semantic_search", r#"{"query": "x", "limit": 21}"#),
        ("knowledge_save", r#"{"content": "x", "priority": 256}"#),
        ("knowledge_save", r#"{"content": "x", "priority": -1}"#),
        ("list_skills", r#"{"limit": 0}"#),
        ("sleep_process", r#"{"turns": 0}"#),
        ("spawn_process", r#"{"name": "p", "goal": "g", "priority": 300}"#),
        ("spawn_process", r#"{"name": "p", "goal": "g", "quota_turns": -1}"#),
        ("spawn_daemon", r#"{"name": "d", "goal": "g", "max_restarts": -1}"#),
        ("session_distill", r#"{"archive": "/tmp/a.zip", "limit": 101}"#),
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
        ("spawn_process", r#"{"name": "p", "goal": "g", "quota_turns": 0}"#),
        ("spawn_daemon", r#"{"name": "d", "goal": "g", "max_restarts": 0}"#),
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
/// minimum, maximum, why). `path` uses `.` for nested objects and `[].` for
/// array items. A `None` bound is intentional only with a reason:
/// fail-closed service parsing (`as_u64().ok_or`, loud on any bad value, so
/// there is no silent-default hole) or signed semantics (`as_i64`, where
/// negatives are valid). The test below walks the whole registry, so a new
/// integer parameter fails until its author registers it here.
const INT_BOUND_REGISTRY: &[(&str, &str, Option<i64>, Option<i64>, &str)] = &[
    ("execute_command", "timeout", Some(1), Some(300), "service clamps 1-300"),
    ("task_wait", "timeout_secs", Some(1), Some(60), "service clamps 1-60"),
    ("search_overflow", "context_lines", Some(0), Some(5), "unwrap_or(2).min(5); 0 is valid"),
    ("search_overflow", "max_results", Some(1), Some(200), "service clamps 1-200"),
    ("knowledge_search", "limit", Some(1), None, "no upper clamp; 0 would return empty"),
    ("knowledge_list", "limit", Some(1), Some(100), "unwrap_or(20).min(100)"),
    ("knowledge_semantic_search", "limit", Some(1), Some(20), "service clamps 1-20"),
    ("knowledge_save", "priority", Some(0), Some(255), "u8 range; service rejects >255"),
    ("knowledge_consolidate", "save_entries[].priority", Some(0), Some(255), "same u8 parser"),
    ("list_skills", "limit", Some(1), Some(100), "service clamps 1-100"),
    ("list_skill_resources", "limit", Some(1), Some(200), "pre-existing bound, pinned"),
    ("read_skill_resource", "limit", Some(1), Some(65536), "pre-existing bound, pinned"),
    ("read_file", "offset", Some(1), None, "no upper clamp in service"),
    ("read_file", "char_offset", Some(0), None, "0 is valid; no upper clamp"),
    ("read_file", "limit", Some(1), None, "no upper clamp in service"),
    ("run_agent_graph", "max_parallel", Some(1), Some(8), "pre-existing bound, pinned"),
    ("run_agent_graph", "static_graph.nodes[].max_retries", Some(0), None, "pre-existing, pinned"),
    ("run_agent_graph", "dynamic_graph.policy.min_selected", Some(2), None, "pre-existing, pinned"),
    ("run_agent_graph", "dynamic_graph.policy.max_selected", Some(2), None, "pre-existing, pinned"),
    ("show_changes", "limit_snippet", Some(0), Some(4000), "pre-existing bound, pinned"),
    ("sleep_process", "turns", Some(1), None, "documented minimum 1; no upper clamp"),
    ("spawn_process", "priority", Some(0), Some(255), "u8 range; larger values truncate via `as u8`"),
    ("spawn_process", "quota_turns", Some(0), None, "0 means unlimited per ResourceLimit::from_legacy"),
    ("spawn_daemon", "priority", Some(0), Some(255), "u8 range, as above"),
    ("spawn_daemon", "quota_turns", Some(0), None, "0 means unlimited, as above"),
    ("spawn_daemon", "max_restarts", Some(0), None, "0 disables restarts; no upper clamp"),
    ("session_distill", "limit", Some(1), Some(100), "service clamps 1-100"),
    ("tree", "max_depth", Some(0), Some(6), "pre-existing bound, pinned"),
    ("manage_team", "budget.max_parallel", Some(1), Some(8), "validate_budget enforces 1-8"),
    ("manage_team", "budget.max_tasks", Some(1), Some(512), "validate_budget enforces 1-512"),
    ("manage_team", "budget.max_total_attempts", Some(1), Some(4096), "validate_budget enforces <=4096"),
    ("manage_team", "budget.max_messages", Some(1), Some(2048), "validate_budget enforces 1-2048"),
    ("manage_team", "lease_secs", Some(1), Some(86400), "service clamps 1-86400"),
    ("kill_process", "pid", None, None, "service ok_or fails closed; no silent default"),
    ("reap_process", "pid", None, None, "service ok_or fails closed; no silent default"),
    ("wait_process", "pid", None, None, "service ok_or fails closed; no silent default"),
    ("signal_process", "pid", None, None, "service ok_or fails closed; no silent default"),
    ("send_ipc_message", "pid", None, None, "service ok_or fails closed; no silent default"),
    ("set_process_group", "pid", None, None, "service ok_or fails closed; no silent default"),
    ("set_process_group", "pgid", None, None, "service ok_or fails closed; no silent default"),
    ("signal_process_group", "pgid", None, None, "service ok_or fails closed; no silent default"),
    ("plan", "steps[].step", None, None, "parse_step_specs ok_or fails closed; labels only"),
    ("plan_update", "step", None, None, "service ok_or fails closed; unknown steps report not-found"),
    ("save_skill", "priority", None, None, "signed as_i64; negative precedence is valid"),
    ("save_skill", "subskills[].priority", None, None, "signed as_i64; negative precedence is valid"),
];

fn collect_int_bounds(
    schema: &serde_json::Value,
    prefix: String,
    out: &mut Vec<(String, Option<i64>, Option<i64>)>,
) {
    let Some(props) = schema.get("properties").and_then(|v| v.as_object()) else {
        return;
    };
    for (name, prop) in props {
        let path = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}.{name}")
        };
        match prop.get("type").and_then(|v| v.as_str()) {
            Some("integer") => out.push((
                path,
                prop.get("minimum").and_then(|v| v.as_i64()),
                prop.get("maximum").and_then(|v| v.as_i64()),
            )),
            Some("object") => collect_int_bounds(prop, path, out),
            Some("array") => {
                if let Some(items) = prop.get("items") {
                    collect_int_bounds(items, format!("{path}[]"), out);
                }
            }
            _ => {}
        }
    }
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
