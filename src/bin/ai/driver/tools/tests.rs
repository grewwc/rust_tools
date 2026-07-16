use super::{
    ASYNC_TOOL_REGISTRY, AsyncToolEntry, AsyncToolPipeKind, AsyncToolPipeMessage, AsyncToolState,
    CachedFileFingerprint, RunOneResult, TOOL_CACHE_TTL_MINUTES, ToolCachePayload, ToolFailureKind,
    ToolRoute, aggregate_async_tool_pipe_messages, async_tool_pipe_message_from_final,
    async_tool_pipe_message_from_started, async_tool_pipe_message_from_stream,
    build_tool_cache_key, classify_tool_error, collect_tool_cache_file_fingerprints,
    delete_async_tool_snapshot, execute_tool_calls, execute_with_safe_retry,
    is_cacheable_tool_name, is_parallel_safe_tool_call, is_tool_cache_entry_fresh,
    load_async_tool_snapshot, lookup_wait_sources, parallel_safe_batch_len,
    persist_async_tool_snapshot, remediation_hint, send_async_tool_pipe_message, should_retry_once,
    stream_preview_from_aggregate, tool_cache_validation_matches,
};
use crate::ai::mcp::McpClient;
use crate::ai::tools::registry::common::current_process_tool_cancel_futex;
use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
use crate::ai::tools::task_tools::WaitManySource;
use crate::ai::types::{FunctionCall, ToolCall};
use aios_kernel::{
    kernel::EventId,
    primitives::{ChannelId, ResourceLimit},
};
use chrono::{Duration, Utc};
use rust_tools::commonw::FastSet;
use serde_json::json;
use std::fs;
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

static ASYNC_TOOL_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct AsyncToolTestGuard {
    _lock: MutexGuard<'static, ()>,
}

impl Drop for AsyncToolTestGuard {
    fn drop(&mut self) {
        if let Ok(mut registry) = ASYNC_TOOL_REGISTRY.lock() {
            registry.clear();
        }
        if let Ok(mut g) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
            *g = None;
        }
    }
}

fn setup_async_tool_kernel() -> (AsyncToolTestGuard, aios_kernel::kernel::SharedKernel, u64) {
    let lock = ASYNC_TOOL_TEST_LOCK.lock().unwrap();
    if let Ok(mut registry) = ASYNC_TOOL_REGISTRY.lock() {
        registry.clear();
    }
    if let Ok(mut g) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *g = None;
    }
    let guard = AsyncToolTestGuard { _lock: lock };
    let kernel = crate::ai::driver::new_local_kernel();
    let root = {
        let mut os = kernel.lock().unwrap();
        os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None)
    };
    crate::ai::tools::os_tools::init_os_tools_globals(kernel.clone());
    (guard, kernel, root)
}

#[test]
fn parallel_batch_groups_consecutive_readonly_builtin_tools() {
    let mcp = McpClient::new();
    let calls = vec![
        tool_call("read_file"),
        tool_call("find_path"),
        tool_call("get_symbol_info"),
    ];
    assert!(is_parallel_safe_tool_call(&mcp, &calls[0]));
    assert_eq!(parallel_safe_batch_len(&mcp, &calls), 3);
}

#[test]
fn parallel_batch_stops_at_mutating_tool() {
    let mcp = McpClient::new();
    // write_file / execute_command 带副作用，不可并行，应在其处截断。
    assert!(!is_parallel_safe_tool_call(&mcp, &tool_call("write_file")));
    assert!(!is_parallel_safe_tool_call(
        &mcp,
        &tool_call("execute_command")
    ));
    let calls = vec![tool_call("read_file"), tool_call("write_file")];
    assert_eq!(parallel_safe_batch_len(&mcp, &calls), 1);
}

#[test]
fn parallel_batch_excludes_barriering_tools() {
    let mcp = McpClient::new();
    // find_path / list_directory / web_search 会触发 barrier，必须顺序执行。
    assert!(!is_parallel_safe_tool_call(&mcp, &tool_call("find_path")));
    assert!(!is_parallel_safe_tool_call(
        &mcp,
        &tool_call("list_directory")
    ));
    assert!(!is_parallel_safe_tool_call(&mcp, &tool_call("web_search")));
}

#[test]
fn parallel_batch_caps_at_max_concurrency() {
    let mcp = McpClient::new();
    let calls: Vec<ToolCall> = (0..super::PARALLEL_READONLY_MAX_CONCURRENCY + 4)
        .map(|_| tool_call("read_file"))
        .collect();
    assert_eq!(
        parallel_safe_batch_len(&mcp, &calls),
        super::PARALLEL_READONLY_MAX_CONCURRENCY
    );
}

#[test]
fn parallel_batch_not_formed_for_single_readonly_call() {
    let mcp = McpClient::new();
    let calls = vec![tool_call("read_file"), tool_call("write_file")];
    // 仅 1 个可并行调用，调用方应回退到顺序路径（batch_len == 1 < 2）。
    assert_eq!(parallel_safe_batch_len(&mcp, &calls), 1);
}

#[test]
fn execute_tool_calls_rejects_tools_hidden_from_current_turn_schema() {
    let (_guard, kernel, root) = setup_async_tool_kernel();
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
fn tool_spawn_cannot_bypass_current_turn_schema() {
    let (_guard, _kernel, _root) = setup_async_tool_kernel();
    let path = std::env::temp_dir().join(format!(
        "turn-schema-spawn-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&path, "hello").unwrap();

    let mut call = tool_call("tool_spawn");
    call.function.arguments = json!({
        "tool_name": "read_file",
        "arguments": {
            "file_path": path.to_string_lossy(),
        }
    })
    .to_string();
    let allowed_tool_names: FastSet<String> = ["tool_spawn".to_string()].into_iter().collect();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(McpClient::new()));
    let result = execute_tool_calls(
        "sess-turn-schema-spawn",
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
            .contains("tool 'read_file' is not available in this turn's tool schema")
    );

    let _ = fs::remove_file(path);
}

#[test]
fn execute_tool_calls_preflights_kernel_tool_quota_before_running_tool() {
    let (_guard, kernel, root) = setup_async_tool_kernel();
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

fn sample_completed_entry(result_channel_id: Option<u64>) -> AsyncToolEntry {
    AsyncToolEntry {
        result_channel_id,
        completion_futex_addr: None,
        session_id: "sess-1".to_string(),
        tool_name: "read_file".to_string(),
        started_at: std::time::Instant::now(),
        state: AsyncToolState::Completed((
            ToolRoute::Builtin,
            RunOneResult {
                tool_result: crate::ai::types::ToolResult {
                    tool_call_id: "call-1".to_string(),
                    content: "payload".to_string(),
                },
                ok: true,
                executed: true,
                cached: false,
            },
        )),
    }
}

#[test]
fn cacheable_tool_name_prefers_read_only_tools() {
    assert!(is_cacheable_tool_name("read_file"));
    assert!(is_cacheable_tool_name("find_path"));
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
fn remediation_hint_only_mentions_alternatives_available_in_current_turn() {
    // read_file 现在唯一的等价备选是 code_search（read_file_lines 已并入 read_file）。
    let available: FastSet<String> = ["code_search".to_string()].into_iter().collect();
    let hint = remediation_hint("read_file", "not found", Some(&available)).expect("hint");
    assert!(hint.contains("`code_search`"));
    assert!(!hint.contains("search_files"));
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

#[test]
fn async_tool_snapshot_roundtrip_uses_channel() {
    let (_guard, kernel, root) = setup_async_tool_kernel();
    let channel_id = {
        let mut os = kernel.lock().unwrap();
        os.channel_create(Some(root), 1, "async-tool-test".to_string())
            .raw()
    };
    let entry = sample_completed_entry(Some(channel_id));

    persist_async_tool_snapshot("tooltask_1", &entry);
    let snapshot =
        load_async_tool_snapshot(&entry).expect("snapshot should be readable via channel");
    assert_eq!(snapshot.task_id, "tooltask_1");
    assert_eq!(snapshot.status, "completed");
    assert_eq!(snapshot.content.as_deref(), Some("payload"));

    delete_async_tool_snapshot(&entry);
    assert!(load_async_tool_snapshot(&entry).is_none());
}

#[test]
fn lookup_wait_sources_include_channel_event_and_futex() {
    let (_guard, kernel, root) = setup_async_tool_kernel();
    let (channel_id, futex_addr) = {
        let mut os = kernel.lock().unwrap();
        let channel = os.channel_create(Some(root), 1, "async-tool-event".to_string());
        let futex = os.futex_create(0, "async-tool-futex".to_string());
        (channel.raw(), futex)
    };

    let mut registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
    registry.insert(
        "tooltask_lookup".to_string(),
        AsyncToolEntry {
            result_channel_id: Some(channel_id),
            completion_futex_addr: Some(futex_addr),
            session_id: "sess-lookup".to_string(),
            tool_name: "read_file".to_string(),
            started_at: std::time::Instant::now(),
            state: AsyncToolState::Running,
        },
    );
    drop(registry);

    let wait_sources = {
        let mut os = kernel.lock().unwrap();
        lookup_wait_sources(os.as_mut(), "sess-lookup", &["tooltask_lookup".to_string()]).unwrap()
    };
    let cancel_futex = {
        let mut os = kernel.lock().unwrap();
        current_process_tool_cancel_futex(os.as_mut())
            .unwrap()
            .unwrap()
    };
    assert_eq!(
        wait_sources,
        vec![
            WaitManySource::Channel(channel_id),
            WaitManySource::Event(EventId::new({
                let os = kernel.lock().unwrap();
                os.channel_event_id(ChannelId(channel_id)).unwrap().as_u64()
            })),
            WaitManySource::Futex {
                addr: futex_addr,
                expected: 0
            },
            WaitManySource::Futex {
                addr: cancel_futex,
                expected: 0
            },
        ]
    );
}

#[test]
fn async_tool_pipe_aggregate_keeps_stream_and_final() {
    let messages = vec![
        AsyncToolPipeMessage {
            task_id: "tooltask_pipe".to_string(),
            session_id: "sess-pipe".to_string(),
            tool_name: "execute_command".to_string(),
            seq: 0,
            kind: AsyncToolPipeKind::Started,
            content: None,
            status: Some("running".to_string()),
            ok: None,
            cached: None,
            executed: None,
            reason: None,
            elapsed_secs: 0.1,
        },
        AsyncToolPipeMessage {
            task_id: "tooltask_pipe".to_string(),
            session_id: "sess-pipe".to_string(),
            tool_name: "execute_command".to_string(),
            seq: 1,
            kind: AsyncToolPipeKind::StreamChunk,
            content: Some("hello ".to_string()),
            status: Some("running".to_string()),
            ok: None,
            cached: None,
            executed: None,
            reason: None,
            elapsed_secs: 0.2,
        },
        AsyncToolPipeMessage {
            task_id: "tooltask_pipe".to_string(),
            session_id: "sess-pipe".to_string(),
            tool_name: "execute_command".to_string(),
            seq: 2,
            kind: AsyncToolPipeKind::Final,
            content: Some("hello world".to_string()),
            status: Some("completed".to_string()),
            ok: Some(true),
            cached: Some(false),
            executed: Some(true),
            reason: None,
            elapsed_secs: 0.3,
        },
    ];

    let agg = aggregate_async_tool_pipe_messages(&messages);
    assert_eq!(agg.stream_chunk_count, 1);
    assert_eq!(agg.final_content.as_deref(), Some("hello world"));
    assert_eq!(agg.last_status.as_deref(), Some("completed"));
    assert_eq!(
        stream_preview_from_aggregate(&agg).as_deref(),
        Some("hello ")
    );
}

#[test]
fn load_async_tool_snapshot_reads_streaming_pipe_messages() {
    let (_guard, kernel, root) = setup_async_tool_kernel();
    let channel_id = {
        let mut os = kernel.lock().unwrap();
        os.channel_create(Some(root), 8, "async-tool-pipe".to_string())
            .raw()
    };
    let entry = AsyncToolEntry {
        result_channel_id: Some(channel_id),
        completion_futex_addr: None,
        session_id: "sess-stream".to_string(),
        tool_name: "execute_command".to_string(),
        started_at: std::time::Instant::now(),
        state: AsyncToolState::Running,
    };

    send_async_tool_pipe_message(
        &entry,
        &async_tool_pipe_message_from_started("tooltask_stream", &entry, 0),
    );
    send_async_tool_pipe_message(
        &entry,
        &async_tool_pipe_message_from_stream("tooltask_stream", &entry, 1, b"partial "),
    );
    let final_entry = AsyncToolEntry {
        state: AsyncToolState::Completed((
            ToolRoute::Builtin,
            RunOneResult {
                tool_result: crate::ai::types::ToolResult {
                    tool_call_id: "call-stream".to_string(),
                    content: "partial final".to_string(),
                },
                ok: true,
                executed: true,
                cached: false,
            },
        )),
        ..entry
    };
    send_async_tool_pipe_message(
        &final_entry,
        &async_tool_pipe_message_from_final("tooltask_stream", &final_entry, 2),
    );

    let snapshot =
        load_async_tool_snapshot(&final_entry).expect("snapshot should decode from pipe");
    assert_eq!(snapshot.status, "completed");
    assert_eq!(snapshot.content.as_deref(), Some("partial final"));
}

#[test]
fn delete_async_tool_snapshot_destroys_result_pipe() {
    let (_guard, kernel, root) = setup_async_tool_kernel();
    let channel_id = {
        let mut os = kernel.lock().unwrap();
        os.channel_create(Some(root), 4, "async-tool-destroy".to_string())
            .raw()
    };
    let entry = sample_completed_entry(Some(channel_id));
    persist_async_tool_snapshot("tooltask_destroy", &entry);

    delete_async_tool_snapshot(&entry);

    let os = kernel.lock().unwrap();
    assert!(os.channel_event_id(ChannelId(channel_id)).is_none());
}
