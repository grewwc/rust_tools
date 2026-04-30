use chrono::{DateTime, Duration, Local, Utc};
use colored::Colorize;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use serde_json::{Value, json};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::{LazyLock, Mutex, atomic::{AtomicU64, Ordering}};
use std::thread;
use std::time::{Duration as StdDuration, Instant};
use crate::ai::tools::os_tools::GLOBAL_OS;
use aios_kernel::{
    kernel::{EventId, WaitPolicy},
    primitives::{ChannelId, ChannelOwnerTag, FutexAddr},
};
use rust_tools::commonw::FastMap;

use crate::ai::{
    mcp::{McpClient, SharedMcpClient},
    tools::task_tools::{
        WaitManySource, append_current_process_cancel_source, epoll_wait_many,
        wait_sources_for_channel_and_futex,
    },
    tools as builtin_tools,
    types::{ToolCall, ToolResult},
};
use crate::commonw::prompt::prompt_yes_or_no_interruptible;

mod barrier;
mod oauth;

static TOOL_FAILURES: LazyLock<Mutex<FastMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

#[derive(Debug, Clone)]
enum ToolRoute {
    Builtin,
    Mcp {
        server_name: String,
        tool_name: String,
    },
}

#[derive(Debug, Clone)]
struct PreparedToolCall {
    route: ToolRoute,
    args: Value,
}

pub(super) struct ExecuteToolCallsResult {
    pub(super) executed_tool_calls: Vec<ToolCall>,
    pub(super) tool_results: Vec<ToolResult>,
    pub(super) cached_hits: Vec<bool>,
}

pub(super) struct RunOneResult {
    pub(super) tool_result: ToolResult,
    pub(super) ok: bool,
    pub(super) executed: bool,
    pub(super) cached: bool,
}

pub(super) trait ToolExecutionObserver {
    fn on_tool_started(&mut self, _tool_call: &ToolCall) {}

    fn on_tool_stream(&mut self, _tool_call: &ToolCall, _chunk: &[u8]) {}

    fn on_tool_finished(&mut self, _tool_call: &ToolCall, _run_result: &RunOneResult) {}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolCachePayload {
    tool_name: String,
    args: Value,
    result: String,
}

const TOOL_CACHE_RECENT_LIMIT: usize = 400;
const TOOL_CACHE_MAX_RESULT_CHARS: usize = 12_000;
const TOOL_CACHE_TTL_MINUTES: i64 = 30;
const ASYNC_TOOL_REGISTRY_LIMIT: usize = 200;

enum AsyncToolState {
    Running,
    Completed((ToolRoute, RunOneResult)),
    Canceled { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AsyncToolSnapshot {
    task_id: String,
    event_id: String,
    session_id: String,
    tool_name: String,
    status: String,
    ok: Option<bool>,
    cached: Option<bool>,
    executed: Option<bool>,
    reason: Option<String>,
    content: Option<String>,
    elapsed_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AsyncToolPipeKind {
    Started,
    StreamChunk,
    Final,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AsyncToolPipeMessage {
    task_id: String,
    session_id: String,
    tool_name: String,
    seq: u64,
    kind: AsyncToolPipeKind,
    content: Option<String>,
    status: Option<String>,
    ok: Option<bool>,
    cached: Option<bool>,
    executed: Option<bool>,
    reason: Option<String>,
    elapsed_secs: f64,
}

#[derive(Debug, Default)]
struct AsyncToolPipeAggregate {
    chunks: Vec<String>,
    last_status: Option<String>,
    ok: Option<bool>,
    cached: Option<bool>,
    executed: Option<bool>,
    reason: Option<String>,
    final_content: Option<String>,
    stream_chunk_count: usize,
}

struct AsyncToolEntry {
    result_channel_id: Option<u64>,
    completion_futex_addr: Option<FutexAddr>,
    session_id: String,
    tool_name: String,
    started_at: Instant,
    state: AsyncToolState,
}

static ASYNC_TOOL_NEXT_ID: AtomicU64 = AtomicU64::new(1);
static ASYNC_TOOL_REGISTRY: LazyLock<Mutex<FastMap<String, AsyncToolEntry>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

fn next_async_tool_id() -> String {
    format!(
        "tooltask_{}",
        ASYNC_TOOL_NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}

fn create_async_tool_channel(task_id: &str) -> Option<u64> {
    let guard = GLOBAL_OS.lock().ok()?;
    let os = guard.as_ref()?;
    let mut os = os.lock().ok()?;
    let owner_pid = os.current_process_id();
    Some(
        os.channel_create_tagged_with_holders(
            owner_pid,
            4096,
            format!("async_tool_result:{task_id}"),
            ChannelOwnerTag::AsyncToolResult,
            vec![
                "async_tool.producer".to_string(),
                "async_tool.consumer".to_string(),
            ],
        )
        .raw(),
    )
}

fn create_async_tool_completion_futex(task_id: &str) -> Option<FutexAddr> {
    let guard = GLOBAL_OS.lock().ok()?;
    let os = guard.as_ref()?;
    let mut os = os.lock().ok()?;
    Some(os.futex_create(0, format!("async_tool_completion:{task_id}")))
}

fn signal_async_tool_completion(addr: Option<FutexAddr>) {
    let Some(addr) = addr else {
        return;
    };
    if let Ok(guard) = GLOBAL_OS.lock()
        && let Some(os) = guard.as_ref()
        && let Ok(mut os) = os.lock()
    {
        let _ = os.futex_store(addr, 1);
    }
}

fn prune_completed_async_tools(registry: &mut FastMap<String, AsyncToolEntry>) {
    if registry.len() <= ASYNC_TOOL_REGISTRY_LIMIT {
        return;
    }
    let completed_keys: Vec<String> = registry
        .iter()
        .filter(|(_, entry)| matches!(entry.state, AsyncToolState::Completed(_) | AsyncToolState::Canceled { .. }))
        .map(|(key, _)| key.clone())
        .collect();
    for key in completed_keys {
        registry.remove(&key);
        if registry.len() <= ASYNC_TOOL_REGISTRY_LIMIT {
            break;
        }
    }
    if let Ok(guard) = GLOBAL_OS.lock()
        && let Some(os) = guard.as_ref()
        && let Ok(mut os) = os.lock()
    {
        let _ = os.channel_gc_closed_empty();
    }
}

fn async_tool_event_id(entry: &AsyncToolEntry) -> Option<EventId> {
    let channel_id = entry.result_channel_id?;
    let guard = GLOBAL_OS.lock().ok()?;
    let os = guard.as_ref()?;
    let os = os.lock().ok()?;
    os.channel_event_id(ChannelId(channel_id))
}

fn async_tool_snapshot_from_entry(task_id: &str, entry: &AsyncToolEntry) -> AsyncToolSnapshot {
    let event_id = async_tool_event_id(entry)
        .map(|id| id.to_string())
        .unwrap_or_else(|| "evt_unavailable".to_string());
    match &entry.state {
        AsyncToolState::Running => AsyncToolSnapshot {
            task_id: task_id.to_string(),
            event_id: event_id.clone(),
            session_id: entry.session_id.clone(),
            tool_name: entry.tool_name.clone(),
            status: "running".to_string(),
            ok: None,
            cached: None,
            executed: None,
            reason: None,
            content: None,
            elapsed_secs: entry.started_at.elapsed().as_secs_f64(),
        },
        AsyncToolState::Completed((_route, run_result)) => AsyncToolSnapshot {
            task_id: task_id.to_string(),
            event_id: event_id.clone(),
            session_id: entry.session_id.clone(),
            tool_name: entry.tool_name.clone(),
            status: if run_result.ok {
                "completed".to_string()
            } else {
                "failed".to_string()
            },
            ok: Some(run_result.ok),
            cached: Some(run_result.cached),
            executed: Some(run_result.executed),
            reason: None,
            content: Some(run_result.tool_result.content.clone()),
            elapsed_secs: entry.started_at.elapsed().as_secs_f64(),
        },
        AsyncToolState::Canceled { reason } => AsyncToolSnapshot {
            task_id: task_id.to_string(),
            event_id,
            session_id: entry.session_id.clone(),
            tool_name: entry.tool_name.clone(),
            status: "canceled".to_string(),
            ok: Some(false),
            cached: Some(false),
            executed: Some(false),
            reason: Some(reason.clone()),
            content: None,
            elapsed_secs: entry.started_at.elapsed().as_secs_f64(),
        },
    }
}

fn async_tool_pipe_message_from_started(
    task_id: &str,
    entry: &AsyncToolEntry,
    seq: u64,
) -> AsyncToolPipeMessage {
    AsyncToolPipeMessage {
        task_id: task_id.to_string(),
        session_id: entry.session_id.clone(),
        tool_name: entry.tool_name.clone(),
        seq,
        kind: AsyncToolPipeKind::Started,
        content: None,
        status: Some("running".to_string()),
        ok: None,
        cached: None,
        executed: None,
        reason: None,
        elapsed_secs: entry.started_at.elapsed().as_secs_f64(),
    }
}

fn async_tool_pipe_message_from_stream(
    task_id: &str,
    entry: &AsyncToolEntry,
    seq: u64,
    chunk: &[u8],
) -> AsyncToolPipeMessage {
    AsyncToolPipeMessage {
        task_id: task_id.to_string(),
        session_id: entry.session_id.clone(),
        tool_name: entry.tool_name.clone(),
        seq,
        kind: AsyncToolPipeKind::StreamChunk,
        content: Some(String::from_utf8_lossy(chunk).to_string()),
        status: Some("running".to_string()),
        ok: None,
        cached: None,
        executed: None,
        reason: None,
        elapsed_secs: entry.started_at.elapsed().as_secs_f64(),
    }
}

fn async_tool_pipe_message_from_final(
    task_id: &str,
    entry: &AsyncToolEntry,
    seq: u64,
) -> AsyncToolPipeMessage {
    let snapshot = async_tool_snapshot_from_entry(task_id, entry);
    AsyncToolPipeMessage {
        task_id: snapshot.task_id,
        session_id: snapshot.session_id,
        tool_name: snapshot.tool_name,
        seq,
        kind: AsyncToolPipeKind::Final,
        content: snapshot.content,
        status: Some(snapshot.status),
        ok: snapshot.ok,
        cached: snapshot.cached,
        executed: snapshot.executed,
        reason: snapshot.reason,
        elapsed_secs: snapshot.elapsed_secs,
    }
}

fn send_async_tool_pipe_message(entry: &AsyncToolEntry, msg: &AsyncToolPipeMessage) {
    let Some(channel_id) = entry.result_channel_id else {
        return;
    };
    if let Ok(payload) = serde_json::to_string(msg)
        && let Ok(guard) = GLOBAL_OS.lock()
        && let Some(os) = guard.as_ref()
        && let Ok(mut os) = os.lock()
    {
        let _ = os.channel_send(None, ChannelId(channel_id), payload);
    }
}

fn close_async_tool_pipe(entry: &AsyncToolEntry) {
    let Some(channel_id) = entry.result_channel_id else {
        return;
    };
    if let Ok(guard) = GLOBAL_OS.lock()
        && let Some(os) = guard.as_ref()
        && let Ok(mut os) = os.lock()
    {
        let _ = os.channel_close(None, ChannelId(channel_id));
    }
}

fn release_async_tool_pipe_ref(entry: &AsyncToolEntry, holder: &str) {
    let Some(channel_id) = entry.result_channel_id else {
        return;
    };
    if let Ok(guard) = GLOBAL_OS.lock()
        && let Some(os) = guard.as_ref()
        && let Ok(mut os) = os.lock()
    {
        let _ = os.channel_release_named(ChannelId(channel_id), holder);
    }
}

fn parse_async_tool_pipe_messages(payloads: &[String]) -> Vec<AsyncToolPipeMessage> {
    payloads
        .iter()
        .filter_map(|payload| serde_json::from_str::<AsyncToolPipeMessage>(payload).ok())
        .collect()
}

fn peek_async_tool_pipe_messages(entry: &AsyncToolEntry) -> Vec<AsyncToolPipeMessage> {
    let Some(channel_id) = entry.result_channel_id else {
        return Vec::new();
    };
    let Ok(guard) = GLOBAL_OS.lock() else {
        return Vec::new();
    };
    let Some(os) = guard.as_ref() else {
        return Vec::new();
    };
    let Ok(os) = os.lock() else {
        return Vec::new();
    };
    let Ok(payloads) = os.channel_peek_all(None, ChannelId(channel_id)) else {
        return Vec::new();
    };
    parse_async_tool_pipe_messages(&payloads)
}

fn drain_async_tool_pipe_messages(entry: &AsyncToolEntry) -> Vec<AsyncToolPipeMessage> {
    let Some(channel_id) = entry.result_channel_id else {
        return Vec::new();
    };
    let Ok(guard) = GLOBAL_OS.lock() else {
        return Vec::new();
    };
    let Some(os) = guard.as_ref() else {
        return Vec::new();
    };
    let Ok(mut os) = os.lock() else {
        return Vec::new();
    };
    let Ok(payloads) = os.channel_try_recv_all(None, ChannelId(channel_id)) else {
        return Vec::new();
    };
    parse_async_tool_pipe_messages(&payloads)
}

fn aggregate_async_tool_pipe_messages(messages: &[AsyncToolPipeMessage]) -> AsyncToolPipeAggregate {
    let mut agg = AsyncToolPipeAggregate::default();
    for msg in messages {
        match msg.kind {
            AsyncToolPipeKind::Started => {
                agg.last_status = msg.status.clone();
            }
            AsyncToolPipeKind::StreamChunk => {
                if let Some(content) = &msg.content {
                    agg.chunks.push(content.clone());
                    agg.stream_chunk_count += 1;
                }
                agg.last_status = msg.status.clone();
            }
            AsyncToolPipeKind::Final => {
                agg.last_status = msg.status.clone();
                agg.ok = msg.ok;
                agg.cached = msg.cached;
                agg.executed = msg.executed;
                agg.reason = msg.reason.clone();
                agg.final_content = msg.content.clone();
            }
        }
    }
    agg
}

fn truncate_stream_preview(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let head = max_chars / 2;
    let tail = max_chars.saturating_sub(head);
    let prefix = text.chars().take(head).collect::<String>();
    let suffix = text
        .chars()
        .skip(total.saturating_sub(tail))
        .collect::<String>();
    format!("{prefix}\n...[truncated]...\n{suffix}")
}

fn stream_preview_from_aggregate(agg: &AsyncToolPipeAggregate) -> Option<String> {
    if agg.chunks.is_empty() {
        return None;
    }
    Some(truncate_stream_preview(&agg.chunks.join(""), 2000))
}

fn async_tool_result_json(
    task_id: &str,
    tool_name: &str,
    status: &str,
    ok: Option<bool>,
    cached: Option<bool>,
    executed: Option<bool>,
    reason: Option<&str>,
    elapsed_secs: f64,
    agg: &AsyncToolPipeAggregate,
) -> Value {
    json!({
        "task_id": task_id,
        "tool_name": tool_name,
        "status": status,
        "ok": ok,
        "cached": cached,
        "executed": executed,
        "reason": reason,
        "elapsed_secs": elapsed_secs,
        "stream_chunk_count": agg.stream_chunk_count,
        "stream_preview": stream_preview_from_aggregate(agg),
        "has_final": agg.final_content.is_some(),
    })
}

struct AsyncToolPipeObserver {
    task_id: String,
    session_id: String,
    tool_name: String,
    result_channel_id: Option<u64>,
    completion_futex_addr: Option<FutexAddr>,
    started_at: Instant,
    next_seq: u64,
}

impl AsyncToolPipeObserver {
    fn running_entry(&self) -> AsyncToolEntry {
        AsyncToolEntry {
            result_channel_id: self.result_channel_id,
            completion_futex_addr: self.completion_futex_addr,
            session_id: self.session_id.clone(),
            tool_name: self.tool_name.clone(),
            started_at: self.started_at,
            state: AsyncToolState::Running,
        }
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        seq
    }

    fn emit_started(&mut self) {
        let entry = self.running_entry();
        let task_id = self.task_id.clone();
        let seq = self.next_seq();
        let msg = async_tool_pipe_message_from_started(&task_id, &entry, seq);
        send_async_tool_pipe_message(&entry, &msg);
    }

    fn emit_stream(&mut self, chunk: &[u8]) {
        let entry = self.running_entry();
        let task_id = self.task_id.clone();
        let seq = self.next_seq();
        let msg = async_tool_pipe_message_from_stream(&task_id, &entry, seq, chunk);
        send_async_tool_pipe_message(&entry, &msg);
    }

    fn emit_finished(&mut self, run_result: &RunOneResult) {
        let entry = AsyncToolEntry {
            result_channel_id: self.result_channel_id,
            completion_futex_addr: self.completion_futex_addr,
            session_id: self.session_id.clone(),
            tool_name: self.tool_name.clone(),
            started_at: self.started_at,
            state: if run_result.ok {
                AsyncToolState::Completed((ToolRoute::Builtin, RunOneResult {
                    tool_result: run_result.tool_result.clone(),
                    ok: run_result.ok,
                    executed: run_result.executed,
                    cached: run_result.cached,
                }))
            } else {
                AsyncToolState::Completed((ToolRoute::Builtin, RunOneResult {
                    tool_result: run_result.tool_result.clone(),
                    ok: run_result.ok,
                    executed: run_result.executed,
                    cached: run_result.cached,
                }))
            },
        };
        let task_id = self.task_id.clone();
        let seq = self.next_seq();
        let msg = async_tool_pipe_message_from_final(&task_id, &entry, seq);
        send_async_tool_pipe_message(&entry, &msg);
        close_async_tool_pipe(&entry);
        release_async_tool_pipe_ref(&entry, "async_tool.producer");
    }
}

impl ToolExecutionObserver for AsyncToolPipeObserver {
    fn on_tool_started(&mut self, _tool_call: &ToolCall) {
        self.emit_started();
    }

    fn on_tool_stream(&mut self, _tool_call: &ToolCall, chunk: &[u8]) {
        self.emit_stream(chunk);
    }

    fn on_tool_finished(&mut self, _tool_call: &ToolCall, run_result: &RunOneResult) {
        self.emit_finished(run_result);
    }
}

fn persist_async_tool_snapshot(task_id: &str, entry: &AsyncToolEntry) {
    if !matches!(
        entry.state,
        AsyncToolState::Completed(_) | AsyncToolState::Canceled { .. }
    ) {
        return;
    }
    let msg = async_tool_pipe_message_from_final(task_id, entry, u64::MAX);
    send_async_tool_pipe_message(entry, &msg);
    close_async_tool_pipe(entry);
    release_async_tool_pipe_ref(entry, "async_tool.producer");
}

fn load_async_tool_snapshot(entry: &AsyncToolEntry) -> Option<AsyncToolSnapshot> {
    let messages = peek_async_tool_pipe_messages(entry);
    let agg = aggregate_async_tool_pipe_messages(&messages);
    Some(AsyncToolSnapshot {
        task_id: messages.first()?.task_id.clone(),
        event_id: async_tool_event_id(entry)
            .map(|id| id.to_string())
            .unwrap_or_else(|| "evt_unavailable".to_string()),
        session_id: messages.first()?.session_id.clone(),
        tool_name: messages.first()?.tool_name.clone(),
        status: agg
            .last_status
            .unwrap_or_else(|| "running".to_string()),
        ok: agg.ok,
        cached: agg.cached,
        executed: agg.executed,
        reason: agg.reason,
        content: agg.final_content.or_else(|| {
            if agg.chunks.is_empty() {
                None
            } else {
                Some(agg.chunks.join(""))
            }
        }),
        elapsed_secs: messages.last()?.elapsed_secs,
    })
}

fn delete_async_tool_snapshot(entry: &AsyncToolEntry) {
    if let Ok(guard) = GLOBAL_OS.lock()
        && let Some(os) = guard.as_ref()
        && let Ok(mut os) = os.lock()
    {
        if let Some(channel_id) = entry.result_channel_id {
            let _ = os.channel_try_recv_all(None, ChannelId(channel_id));
            let _ = os.channel_close(None, ChannelId(channel_id));
            let _ = os.channel_release_named(ChannelId(channel_id), "async_tool.consumer");
            let _ = os.channel_destroy(None, ChannelId(channel_id));
        }
        if let Some(addr) = entry.completion_futex_addr {
            let _ = os.futex_destroy(addr);
        }
    }
}

fn lookup_wait_sources(
    os: &mut dyn aios_kernel::kernel::Kernel,
    session_id: &str,
    task_ids: &[String],
) -> Result<Vec<WaitManySource>, String> {
    let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
    let mut wait_sources = Vec::new();
    for task_id in task_ids {
        let Some(entry) = registry.get(task_id) else {
            return Err(format!("Unknown task_id: {}", task_id));
        };
        if entry.session_id != session_id {
            return Err(format!("Task {} does not belong to this session.", task_id));
        }
        let channel_id = entry.result_channel_id.ok_or_else(|| {
            format!(
                "Task {} has no available result channel; async wait requires AIOS channel support.",
                task_id
            )
        })?;
        wait_sources.extend(wait_sources_for_channel_and_futex(
            os,
            channel_id,
            entry.completion_futex_addr,
        )?);
    }
    append_current_process_cancel_source(os, &mut wait_sources)?;
    Ok(wait_sources)
}

fn parse_wait_policy(args: &Value) -> Result<WaitPolicy, String> {
    match args
        .get("wait_policy")
        .and_then(Value::as_str)
        .unwrap_or("all")
        .to_ascii_lowercase()
        .as_str()
    {
        "all" => Ok(WaitPolicy::All),
        "any" => Ok(WaitPolicy::Any),
        other => Err(format!(
            "Invalid wait_policy '{}'. Expected 'all' or 'any'.",
            other
        )),
    }
}

fn collect_async_task_snapshot(
    session_id: &str,
    task_ids: &[String],
) -> Result<(Vec<Value>, Vec<Value>), String> {
    let mut terminal_results = Vec::new();
    let mut pending = Vec::new();
    let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
    for task_id in task_ids {
        let Some(entry) = registry.get(task_id) else {
            return Err(format!("Unknown task_id: {}", task_id));
        };
        if entry.session_id != session_id {
            return Err(format!("Task {} does not belong to this session.", task_id));
        }
        let pipe_messages = peek_async_tool_pipe_messages(entry);
        let agg = aggregate_async_tool_pipe_messages(&pipe_messages);
        if let Some(snapshot) = load_async_tool_snapshot(entry) {
            if snapshot.session_id != session_id {
                return Err(format!("Task {} does not belong to this session.", task_id));
            }
            match snapshot.status.as_str() {
                "running" => pending.push(async_tool_result_json(
                    &snapshot.task_id,
                    &snapshot.tool_name,
                    &snapshot.status,
                    snapshot.ok,
                    snapshot.cached,
                    snapshot.executed,
                    snapshot.reason.as_deref(),
                    snapshot.elapsed_secs,
                    &agg,
                )),
                "completed" | "failed" => {
                    let mut value = async_tool_result_json(
                        &snapshot.task_id,
                        &snapshot.tool_name,
                        &snapshot.status,
                        snapshot.ok,
                        snapshot.cached,
                        snapshot.executed,
                        snapshot.reason.as_deref(),
                        snapshot.elapsed_secs,
                        &agg,
                    );
                    if let Some(obj) = value.as_object_mut() {
                        obj.insert("content".to_string(), json!(snapshot.content.unwrap_or_default()));
                    }
                    terminal_results.push(value);
                }
                "canceled" => terminal_results.push(async_tool_result_json(
                    &snapshot.task_id,
                    &snapshot.tool_name,
                    &snapshot.status,
                    Some(false),
                    Some(false),
                    Some(false),
                    snapshot.reason.as_deref(),
                    snapshot.elapsed_secs,
                    &agg,
                )),
                _ => {}
            }
            continue;
        }
        match &entry.state {
            AsyncToolState::Running => pending.push(async_tool_result_json(
                task_id,
                &entry.tool_name,
                "running",
                None,
                None,
                None,
                None,
                entry.started_at.elapsed().as_secs_f64(),
                &agg,
            )),
            AsyncToolState::Completed((_route, run_result)) => {
                let mut value = async_tool_result_json(
                    task_id,
                    &entry.tool_name,
                    if run_result.ok { "completed" } else { "failed" },
                    Some(run_result.ok),
                    Some(run_result.cached),
                    Some(run_result.executed),
                    None,
                    entry.started_at.elapsed().as_secs_f64(),
                    &agg,
                );
                if let Some(obj) = value.as_object_mut() {
                    obj.insert("content".to_string(), json!(run_result.tool_result.content));
                }
                terminal_results.push(value);
            }
            AsyncToolState::Canceled { reason } => terminal_results.push(async_tool_result_json(
                task_id,
                &entry.tool_name,
                "canceled",
                Some(false),
                Some(false),
                Some(false),
                Some(reason),
                entry.started_at.elapsed().as_secs_f64(),
                &agg,
            )),
        }
    }
    Ok((terminal_results, pending))
}

fn route_tool_call(mcp_client: &McpClient, tool_name: &str) -> ToolRoute {
    if let Some((server_name, tool_name)) = mcp_client.parse_tool_name_for_known_server(tool_name) {
        ToolRoute::Mcp {
            server_name,
            tool_name,
        }
    } else {
        ToolRoute::Builtin
    }
}

fn parse_tool_args(tool_call: &ToolCall) -> Result<Value, ToolResult> {
    let raw_args = tool_call.function.arguments.trim();
    if raw_args.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(raw_args).map_err(|err| ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: format!("Error: failed to parse arguments: {}", err),
    })
}

fn prepare_tool_call(
    mcp_client: &McpClient,
    tool_call: &ToolCall,
) -> Result<PreparedToolCall, ToolResult> {
    Ok(PreparedToolCall {
        route: route_tool_call(mcp_client, &tool_call.function.name),
        args: parse_tool_args(tool_call)?,
    })
}

fn requires_user_confirmation_for_tool(_tool_name: &str) -> bool {
    false
}

fn confirm_tool_execution(tool_call: &ToolCall, args: &Value) -> Result<(), RunOneResult> {
    if !requires_user_confirmation_for_tool(&tool_call.function.name) {
        return Ok(());
    }

    let confirm =
        prompt_yes_or_no_interruptible(&format!("Confirm tool execution:{} (y/n): ", args));
    if confirm == Some(true) {
        return Ok(());
    }

    println!("canceled by user.");
    Err(RunOneResult {
        tool_result: ToolResult {
            tool_call_id: tool_call.id.clone(),
            content: if confirm.is_none() {
                format!(
                    "Error: {} canceled by user (Ctrl+C)",
                    tool_call.function.name
                )
            } else {
                format!("Error: {} canceled by user", tool_call.function.name)
            },
        },
        ok: false,
        executed: false,
        cached: false,
    })
}

fn remediation_hint(tool_name: &str, err: &str) -> Option<String> {
    let err_lower = err.to_lowercase();

    if tool_name == "mcp_feishu_docs_get_text_by_url" && err_lower.contains("unsupported url") {
        return Some(
            "Suggestion: this tool only works for supported Feishu/Lark docs URLs. Do not retry with the same URL. Use mcp_feishu_docs_search to find the document first, or ask the user for a direct Feishu docs/wiki/sheet URL.".to_string(),
        );
    }

    if err_lower.contains("failed to parse arguments") || err_lower.contains("invalid type") {
        return Some(
            "Suggestion: fix the tool arguments to match the declared JSON schema before retrying."
                .to_string(),
        );
    }

    if err_lower.contains("no such file") || err_lower.contains("not found") {
        return Some(
            "Suggestion: verify the path or identifier first, or use a search/list tool to discover the correct target before retrying.".to_string(),
        );
    }

    if err_lower.contains("timeout") || err_lower.contains("timed out") {
        return Some(
            "Suggestion: retry once with a narrower query or a smaller scope. If it still fails, switch to another tool or ask the user.".to_string(),
        );
    }

    None
}

fn format_tool_error(tool_call: &ToolCall, err: &str) -> ToolResult {
    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: if let Some(hint) = remediation_hint(&tool_call.function.name, err) {
            format!(
                "Error: {} failed: {}\n{}",
                tool_call.function.name, err, hint
            )
        } else {
            format!("Error: {} failed: {}", tool_call.function.name, err)
        },
    }
}

fn execute_prepared_tool_call(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
) -> Result<ToolResult, String> {
    match &prepared.route {
        ToolRoute::Builtin => {
            if tool_call.function.name == "tool_spawn" {
                execute_tool_spawn(session_id, mcp_client, shared_mcp_client, &tool_call.id, &prepared.args)
            } else if tool_call.function.name == "tool_wait" {
                execute_tool_wait(session_id, &tool_call.id, &prepared.args)
            } else if tool_call.function.name == "tool_status" {
                execute_tool_status(session_id, &tool_call.id, &prepared.args)
            } else if tool_call.function.name == "tool_cancel" {
                execute_tool_cancel(session_id, &tool_call.id, &prepared.args)
            } else if tool_call.function.name == "execute_command" {
                builtin_tools::command_tools::execute_command_streaming(&prepared.args, |chunk| {
                    if let Some(observer) = observer.as_deref_mut() {
                        observer.on_tool_stream(tool_call, chunk);
                    }
                })
                .map(|content| ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content,
                })
            } else {
                builtin_tools::execute_tool_call_with_args(
                    &tool_call.id,
                    &tool_call.function.name,
                    &prepared.args,
                )
                .map_err(|e| e.to_string())
            }
        }
        ToolRoute::Mcp {
            server_name,
            tool_name,
        } => oauth::execute_mcp_tool_call(
            mcp_client,
            tool_call,
            server_name,
            tool_name,
            &prepared.args,
        ),
    }
}

fn execute_prepared_builtin_tool_call(
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
) -> Result<ToolResult, String> {
    builtin_tools::execute_tool_call_with_args(
        &tool_call.id,
        &tool_call.function.name,
        &prepared.args,
    )
    .map_err(|e| e.to_string())
}

fn validate_spawnable_tool(target_tool_name: &str, route: &ToolRoute) -> Result<(), String> {
    if let ToolRoute::Mcp { tool_name, .. } = route {
        if tool_name.starts_with("oauth_") {
            return Err("OAuth helper MCP tools cannot be spawned asynchronously.".to_string());
        }
        if matches!(target_tool_name, "tool_spawn" | "tool_wait" | "tool_status") {
            return Err(format!("Tool '{}' cannot be spawned recursively.", target_tool_name));
        }
        return Ok(());
    }

    let Some(spec) = crate::ai::tools::registry::common::get_tool_spec(target_tool_name) else {
        return Err(format!("Unknown tool: {}", target_tool_name));
    };

    if spec.async_policy != crate::ai::tools::registry::common::ToolAsyncPolicy::Spawnable {
        return Err(format!(
            "Tool '{}' is not marked as spawnable for async execution.",
            target_tool_name
        ));
    }

    if matches!(target_tool_name, "tool_spawn" | "tool_wait" | "tool_status") {
        return Err(format!("Tool '{}' cannot be spawned recursively.", target_tool_name));
    }

    Ok(())
}

fn execute_tool_spawn(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_call_id: &str,
    args: &Value,
) -> Result<ToolResult, String> {
    let target_tool_name = args
        .get("tool_name")
        .and_then(Value::as_str)
        .ok_or("Missing 'tool_name' parameter")?;
    let target_args = args
        .get("arguments")
        .cloned()
        .ok_or("Missing 'arguments' parameter")?;

    let async_task_id = next_async_tool_id();
    let synthetic_tool_call = ToolCall {
        id: format!("async-call-{}", async_task_id),
        tool_type: "function".to_string(),
        function: crate::ai::types::FunctionCall {
            name: target_tool_name.to_string(),
            arguments: serde_json::to_string(&target_args)
                .map_err(|e| format!("Failed to serialize arguments: {}", e))?,
        },
    };
    let prepared = PreparedToolCall {
        route: route_tool_call(mcp_client, target_tool_name),
        args: target_args,
    };

    validate_spawnable_tool(target_tool_name, &prepared.route)?;
    let result_channel_id = create_async_tool_channel(&async_task_id);
    let completion_futex_addr = create_async_tool_completion_futex(&async_task_id);
    let started_at = Instant::now();

    if let Some(tool_result) = load_cached_tool_result(session_id, &synthetic_tool_call, &prepared.args)
    {
        let run_result = RunOneResult {
            tool_result,
            ok: true,
            executed: false,
            cached: true,
        };
        let mut registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
        registry.insert(
            async_task_id.clone(),
            AsyncToolEntry {
                result_channel_id,
                completion_futex_addr,
                session_id: session_id.to_string(),
                tool_name: target_tool_name.to_string(),
                started_at,
                state: AsyncToolState::Completed((prepared.route.clone(), run_result)),
            },
        );
        signal_async_tool_completion(completion_futex_addr);
        if let Some(entry) = registry.get(&async_task_id) {
            let started = async_tool_pipe_message_from_started(&async_task_id, entry, 0);
            send_async_tool_pipe_message(entry, &started);
            persist_async_tool_snapshot(&async_task_id, entry);
        }
        prune_completed_async_tools(&mut registry);
        return Ok(ToolResult {
            tool_call_id: tool_call_id.to_string(),
            content: json!({
                "task_id": async_task_id,
                "tool_name": target_tool_name,
                "status": "completed",
                "cached": true,
            })
            .to_string(),
        });
    }

    let session_id = session_id.to_string();
    let session_id_for_registry = session_id.clone();
    let tool_name = target_tool_name.to_string();
    let prepared_for_thread = prepared.clone();
    let tool_call_for_thread = synthetic_tool_call.clone();
    let route_for_registry = prepared.route.clone();
    let shared_mcp_client_for_thread = shared_mcp_client.clone();
    let async_task_id_for_thread = async_task_id.clone();
    let tool_name_for_thread = tool_name.clone();
    let result_channel_id_for_thread = result_channel_id;
    let completion_futex_addr_for_thread = completion_futex_addr;
    let started_at_for_thread = started_at;

    thread::spawn(move || {
        let mut pipe_observer = AsyncToolPipeObserver {
            task_id: async_task_id_for_thread.clone(),
            session_id: session_id.clone(),
            tool_name: tool_name_for_thread.clone(),
            result_channel_id: result_channel_id_for_thread,
            completion_futex_addr: completion_futex_addr_for_thread,
            started_at: started_at_for_thread,
            next_seq: 0,
        };
        pipe_observer.on_tool_started(&tool_call_for_thread);
        let result = match &prepared_for_thread.route {
            ToolRoute::Builtin => {
                if tool_call_for_thread.function.name == "execute_command" {
                    builtin_tools::command_tools::execute_command_streaming(&prepared_for_thread.args, |chunk| {
                        pipe_observer.on_tool_stream(&tool_call_for_thread, chunk);
                    })
                    .map(|content| ToolResult {
                        tool_call_id: tool_call_for_thread.id.clone(),
                        content,
                    })
                } else {
                    execute_prepared_builtin_tool_call(&tool_call_for_thread, &prepared_for_thread)
                }
            }
            ToolRoute::Mcp { .. } => {
                let guard = shared_mcp_client_for_thread
                    .lock()
                    .map_err(|_| "Shared MCP client poisoned".to_string());
                match guard {
                    Ok(mc) => oauth::execute_mcp_tool_call(
                        &mc,
                        &tool_call_for_thread,
                        match &prepared_for_thread.route {
                            ToolRoute::Mcp { server_name, .. } => server_name,
                            ToolRoute::Builtin => unreachable!(),
                        },
                        match &prepared_for_thread.route {
                            ToolRoute::Mcp { tool_name, .. } => tool_name,
                            ToolRoute::Builtin => unreachable!(),
                        },
                        &prepared_for_thread.args,
                    )
                    .map_err(|e| e.to_string()),
                    Err(err) => Err(err),
                }
            }
        };
        let run_result = finalize_execution_result(
            &session_id,
            &tool_call_for_thread,
            &prepared_for_thread,
            result,
            true,
            false,
        );
        pipe_observer.on_tool_finished(&tool_call_for_thread, &run_result);
        signal_async_tool_completion(completion_futex_addr_for_thread);

        if let Ok(mut registry) = ASYNC_TOOL_REGISTRY.lock() {
            if let Some(entry) = registry.get_mut(&async_task_id_for_thread)
                && matches!(entry.state, AsyncToolState::Running)
            {
                entry.state = AsyncToolState::Completed((route_for_registry, run_result));
            }
        }
    });

    let mut registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
    registry.insert(
        async_task_id.clone(),
        AsyncToolEntry {
            result_channel_id,
            completion_futex_addr,
            session_id: session_id_for_registry,
            tool_name,
            started_at,
            state: AsyncToolState::Running,
        },
    );
    prune_completed_async_tools(&mut registry);

    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content: json!({
            "task_id": async_task_id,
            "tool_name": target_tool_name,
            "status": "running",
            "cached": false,
        })
        .to_string(),
    })
}

fn execute_tool_status(session_id: &str, tool_call_id: &str, args: &Value) -> Result<ToolResult, String> {
    let filter_task_ids = args
        .get("task_ids")
        .and_then(Value::as_array)
        .map(|items| {
            items.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        });

    let mut results = Vec::new();
    {
        let mut registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
        let task_ids: Vec<String> = if let Some(ids) = filter_task_ids {
            ids
        } else {
            registry
                .iter()
                .filter(|(_, entry)| entry.session_id == session_id)
                .map(|(task_id, _)| task_id.clone())
                .collect()
        };

        for task_id in task_ids {
            let Some(entry) = registry.get_mut(&task_id) else {
                continue;
            };
            let pipe_messages = peek_async_tool_pipe_messages(entry);
            let agg = aggregate_async_tool_pipe_messages(&pipe_messages);
            if let Some(snapshot) = load_async_tool_snapshot(entry) {
                if snapshot.session_id != session_id {
                    continue;
                }
                let mut value = async_tool_result_json(
                    &snapshot.task_id,
                    &snapshot.tool_name,
                    &snapshot.status,
                    snapshot.ok,
                    snapshot.cached,
                    snapshot.executed,
                    snapshot.reason.as_deref(),
                    snapshot.elapsed_secs,
                    &agg,
                );
                if let Some(obj) = value.as_object_mut()
                    && let Some(content) = snapshot.content
                {
                    obj.insert("content".to_string(), json!(content));
                }
                results.push(value);
                continue;
            }
            if entry.session_id != session_id {
                continue;
            }
            match &entry.state {
                AsyncToolState::Running => results.push(async_tool_result_json(
                    &task_id,
                    &entry.tool_name,
                    "running",
                    None,
                    None,
                    None,
                    None,
                    entry.started_at.elapsed().as_secs_f64(),
                    &agg,
                )),
                AsyncToolState::Completed((_route, run_result)) => results.push(async_tool_result_json(
                    &task_id,
                    &entry.tool_name,
                    if run_result.ok { "completed" } else { "failed" },
                    Some(run_result.ok),
                    Some(run_result.cached),
                    Some(run_result.executed),
                    None,
                    entry.started_at.elapsed().as_secs_f64(),
                    &agg,
                )),
                AsyncToolState::Canceled { reason } => results.push(async_tool_result_json(
                    &task_id,
                    &entry.tool_name,
                    "canceled",
                    Some(false),
                    Some(false),
                    Some(false),
                    Some(reason),
                    entry.started_at.elapsed().as_secs_f64(),
                    &agg,
                )),
            }
        }
    }

    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content: json!({ "results": results }).to_string(),
    })
}

fn execute_tool_cancel(session_id: &str, tool_call_id: &str, args: &Value) -> Result<ToolResult, String> {
    let task_ids = args
        .get("task_ids")
        .and_then(Value::as_array)
        .ok_or("Missing 'task_ids' array parameter")?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<Vec<_>>();
    if task_ids.is_empty() {
        return Err("task_ids array cannot be empty".to_string());
    }

    let reason = args
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("canceled by model")
        .to_string();

    let mut results = Vec::new();
    let mut registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
    for task_id in task_ids {
        let Some(entry) = registry.get_mut(&task_id) else {
            return Err(format!("Unknown task_id: {}", task_id));
        };
        if entry.session_id != session_id {
            return Err(format!("Task {} does not belong to this session.", task_id));
        }

        match &entry.state {
            AsyncToolState::Running => {
                entry.state = AsyncToolState::Canceled {
                    reason: reason.clone(),
                };
                persist_async_tool_snapshot(&task_id, entry);
                signal_async_tool_completion(entry.completion_futex_addr);
                results.push(json!({
                    "task_id": task_id,
                    "tool_name": entry.tool_name,
                    "status": "canceled",
                    "reason": reason,
                }));
            }
            AsyncToolState::Completed((_route, run_result)) => {
                results.push(json!({
                    "task_id": task_id,
                    "tool_name": entry.tool_name,
                    "status": if run_result.ok { "completed" } else { "failed" },
                    "reason": "task already finished",
                }));
            }
            AsyncToolState::Canceled { reason } => {
                results.push(json!({
                    "task_id": task_id,
                    "tool_name": entry.tool_name,
                    "status": "canceled",
                    "reason": reason,
                }));
            }
        }
    }
    prune_completed_async_tools(&mut registry);

    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content: json!({ "results": results }).to_string(),
    })
}

fn execute_tool_wait(session_id: &str, tool_call_id: &str, args: &Value) -> Result<ToolResult, String> {
    let task_ids = args
        .get("task_ids")
        .and_then(Value::as_array)
        .ok_or("Missing 'task_ids' array parameter")?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<Vec<_>>();
    if task_ids.is_empty() {
        return Err("task_ids array cannot be empty".to_string());
    }

    let wait_policy = parse_wait_policy(args)?;
    let timeout_ticks = args.get("timeout_ticks").and_then(Value::as_u64);
    let (initial_terminal, initial_pending) = collect_async_task_snapshot(session_id, &task_ids)?;
    let satisfied = match wait_policy {
        WaitPolicy::Any => !initial_terminal.is_empty(),
        WaitPolicy::All => initial_pending.is_empty(),
    };
    if satisfied {
        if initial_pending.is_empty() {
            let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
            for task_id in &task_ids {
                if let Some(entry) = registry.get(task_id) {
                    delete_async_tool_snapshot(entry);
                }
            }
        }
        let all_done = initial_pending.is_empty();
        let completed_for_results = initial_terminal.clone();
        let pending_for_results = initial_pending.clone();
        let suggested_next_actions = if all_done {
            vec!["continue_reasoning"]
        } else if !initial_terminal.is_empty() {
            vec!["continue_reasoning_with_partial_results", "use_tool_status", "use_tool_cancel"]
        } else {
            vec!["use_tool_status", "continue_reasoning", "use_tool_cancel"]
        };
        return Ok(ToolResult {
            tool_call_id: tool_call_id.to_string(),
            content: json!({
                "all_done": all_done,
                "completed": initial_terminal,
                "pending": initial_pending,
                "results": {
                    "completed": completed_for_results,
                    "pending": pending_for_results,
                },
                "wait_policy": match wait_policy { WaitPolicy::Any => "any", WaitPolicy::All => "all" },
                "suggested_next_actions": suggested_next_actions,
            })
            .to_string(),
        });
    }

    if let Ok(guard) = GLOBAL_OS.lock()
        && let Some(os) = guard.as_ref()
        && let Ok(mut os) = os.lock()
        && os.current_process_id().is_some()
    {
        let wait_sources = lookup_wait_sources(os.as_mut(), session_id, &task_ids)?;
        let wait = epoll_wait_many(
            os.as_mut(),
            &format!("tool_wait:{}:{}", session_id, task_ids.join(",")),
            &wait_sources,
            wait_policy.clone(),
            timeout_ticks,
        )?;
        if wait.suspended {
            return Ok(ToolResult {
                tool_call_id: tool_call_id.to_string(),
                content: json!({
                    "status": "suspended",
                    "wait_policy": match wait_policy { WaitPolicy::Any => "any", WaitPolicy::All => "all" },
                    "task_ids": task_ids,
                    "event_ids": wait.event_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                    "timeout_tick": wait.timeout_tick,
                    "message": "Current process suspended via epoll_wait_many. Yield control now; after wake-up, inspect mailbox and use tool_status or tool_wait again if needed."
                })
                .to_string(),
            });
        }
        // Condition already satisfied; fall through to collect results.
    } else {
        // No OS context — use the inline polling path below.
        let already_satisfied = false;
        let _ = already_satisfied;
    }

    // This point is reached in two cases:
    //   1. OS context available but wait condition was already satisfied (no suspension needed).
    //   2. No OS context — fall through to poll-based wait.
    // In case 1 all tasks are done so the loop below exits immediately.
    let max_wait_ms = args
        .get("max_wait_ms")
        .and_then(Value::as_u64)
        .or_else(|| args.get("timeout_secs").and_then(Value::as_u64).map(|secs| secs.saturating_mul(1000)))
        .unwrap_or(1500);
    let deadline = Instant::now() + StdDuration::from_millis(max_wait_ms);
    while Instant::now() < deadline {
        let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
        let mut has_terminal = false;
        let mut has_running = false;
        for task_id in &task_ids {
            let Some(entry) = registry.get(task_id) else {
                return Err(format!("Unknown task_id: {}", task_id));
            };
            if entry.session_id != session_id {
                return Err(format!("Task {} does not belong to this session.", task_id));
            }
            match entry.state {
                AsyncToolState::Running => has_running = true,
                AsyncToolState::Completed(_) | AsyncToolState::Canceled { .. } => has_terminal = true,
            }
        }
        drop(registry);
        if has_terminal || !has_running {
            break;
        }
        thread::sleep(StdDuration::from_millis(50));
    }

    let (terminal_results, pending) = collect_async_task_snapshot(session_id, &task_ids)?;
    if pending.is_empty() {
        let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
        for task_id in &task_ids {
            if let Some(entry) = registry.get(task_id) {
                delete_async_tool_snapshot(entry);
            }
        }
    }

    let all_done = pending.is_empty();
    let completed_for_results = terminal_results.clone();
    let pending_for_results = pending.clone();
    let suggested_next_actions = if all_done {
        vec!["continue_reasoning"]
    } else if !terminal_results.is_empty() {
        vec!["continue_reasoning_with_partial_results", "use_tool_status", "use_tool_cancel"]
    } else {
        vec!["use_tool_status", "continue_reasoning", "use_tool_cancel"]
    };

    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content: json!({
            "all_done": all_done,
            "wait_window_ms": max_wait_ms,
            "completed": terminal_results,
            "pending": pending,
            "results": {
                "completed": completed_for_results,
                "pending": pending_for_results,
            },
            "suggested_next_actions": suggested_next_actions,
        })
        .to_string(),
    })
}

fn record_tool_failure(tool_name: &str) {
    if let Ok(mut map) = TOOL_FAILURES.lock() {
        let counter = map.entry(tool_name.to_string()).or_insert(0);
        *counter = counter.saturating_add(1).min(100);
    }
}

fn finalize_execution_result(
    session_id: &str,
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
    result: Result<ToolResult, String>,
    executed: bool,
    cached: bool,
) -> RunOneResult {
    let run_result = match result {
        Ok(tool_result) => {
            if executed && !cached {
                store_tool_cache_result(session_id, tool_call, &prepared.args, &tool_result);
            }
            RunOneResult {
                tool_result,
                ok: true,
                executed,
                cached,
            }
        }
        Err(err) => RunOneResult {
            tool_result: format_tool_error(tool_call, &err),
            ok: false,
            executed,
            cached,
        },
    };
    if run_result.executed && !run_result.ok {
        record_tool_failure(&tool_call.function.name);
    }
    run_result
}

fn is_parallel_task_tool_call(tool_call: &ToolCall, prepared: &PreparedToolCall) -> bool {
    matches!(prepared.route, ToolRoute::Builtin) && tool_call.function.name == "task"
}

fn print_run_status(tool_call: &ToolCall, run_result: &RunOneResult) {
    let name = &tool_call.function.name;
    if run_result.cached {
        println!("\n[Cached] {}", name.bright_blue());
    } else if !run_result.executed {
        println!("\n[Skipped] {}", name.yellow());
    } else if run_result.ok {
        println!("\n[Completed] {}", name.green());
    } else {
        println!("\n[Failed] {}", name.red());
    }
}

fn parallel_task_batch_len(mcp_client: &McpClient, tool_calls: &[ToolCall], start: usize) -> usize {
    let mut len = 0usize;
    for tool_call in &tool_calls[start..] {
        let Ok(prepared) = prepare_tool_call(mcp_client, tool_call) else {
            break;
        };
        if !is_parallel_task_tool_call(tool_call, &prepared) {
            break;
        }
        len += 1;
    }
    len
}

fn execute_parallel_task_batch(
    session_id: &str,
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
) -> Vec<(ToolRoute, RunOneResult)> {
    let mut ordered_results: Vec<Option<(ToolRoute, RunOneResult)>> =
        std::iter::repeat_with(|| None).take(tool_calls.len()).collect();
    let mut pending = Vec::new();

    for (idx, tool_call) in tool_calls.iter().enumerate() {
        let prepared = match prepare_tool_call(mcp_client, tool_call) {
            Ok(prepared) => prepared,
            Err(tool_result) => {
                ordered_results[idx] = Some((
                    route_tool_call(mcp_client, &tool_call.function.name),
                    RunOneResult {
                        tool_result,
                        ok: false,
                        executed: true,
                        cached: false,
                    },
                ));
                continue;
            }
        };

        if let Err(result) = confirm_tool_execution(tool_call, &prepared.args) {
            ordered_results[idx] = Some((prepared.route, result));
            continue;
        }

        if let Some(tool_result) = load_cached_tool_result(session_id, tool_call, &prepared.args) {
            ordered_results[idx] = Some((
                prepared.route,
                RunOneResult {
                    tool_result,
                    ok: true,
                    executed: false,
                    cached: true,
                },
            ));
            continue;
        }

        pending.push((idx, tool_call.clone(), prepared));
    }

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(pending.len());
        for (idx, tool_call, prepared) in pending {
            handles.push(scope.spawn(move || {
                let result = execute_prepared_builtin_tool_call(&tool_call, &prepared);
                (idx, tool_call, prepared, result)
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok((idx, tool_call, prepared, result)) => {
                    let run_result = finalize_execution_result(
                        session_id,
                        &tool_call,
                        &prepared,
                        result,
                        true,
                        false,
                    );
                    ordered_results[idx] = Some((prepared.route, run_result));
                }
                Err(_) => {
                    // This path should be unreachable in normal operation, but keep the
                    // batch resilient if one worker panics.
                }
            }
        }
    });

    ordered_results
        .into_iter()
        .enumerate()
        .map(|(idx, item)| {
            item.unwrap_or_else(|| {
                let tool_call = &tool_calls[idx];
                (
                    route_tool_call(mcp_client, &tool_call.function.name),
                    RunOneResult {
                        tool_result: format_tool_error(
                            tool_call,
                            "parallel task worker panicked before producing a result",
                        ),
                        ok: false,
                        executed: true,
                        cached: false,
                    },
                )
            })
        })
        .collect()
}

fn run_one(
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    session_id: &str,
    tool_call: &ToolCall,
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
) -> (ToolRoute, RunOneResult) {
    let prepared = match prepare_tool_call(mcp_client, tool_call) {
        Ok(prepared) => prepared,
        Err(tool_result) => {
            return (
                route_tool_call(mcp_client, &tool_call.function.name),
                RunOneResult {
                    tool_result,
                    ok: false,
                    executed: true,
                    cached: false,
                },
            );
        }
    };

    if let Err(result) = confirm_tool_execution(tool_call, &prepared.args) {
        return (prepared.route, result);
    }

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os_arc) = guard.as_ref() {
            if let Ok(os) = os_arc.lock() {
                if let Some(current_pid) = os.current_process_id() {
                    if let Some(proc) = os.get_process(current_pid) {
                        if !proc.allowed_tools.is_empty() && !proc.allowed_tools.contains(&tool_call.function.name) {
                            let content = format!("Error: tool '{}' is not in the allowed whitelist for this process.", tool_call.function.name);
                            return (
                                prepared.route,
                                RunOneResult {
                                    tool_result: ToolResult {
                                        tool_call_id: tool_call.id.clone(),
                                        content,
                                    },
                                    ok: false,
                                    executed: false,
                                    cached: false,
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    if let Some(tool_result) = load_cached_tool_result(session_id, tool_call, &prepared.args) {
        return (
            prepared.route,
            RunOneResult {
                tool_result,
                ok: true,
                executed: false,
                cached: true,
            },
        );
    }

    println!("\n[Running] {}", tool_call.function.name.cyan());

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os_arc) = guard.as_ref() {
            if let Ok(mut os) = os_arc.lock() {
                if let Some(pid) = os.current_process_id() {
                    os.increment_tool_calls_used_for(pid);
                }
            }
        }
    }

    if let Some(observer) = observer.as_deref_mut() {
        observer.on_tool_started(tool_call);
    }

    let result = execute_prepared_tool_call(
        session_id,
        mcp_client,
        shared_mcp_client,
        tool_call,
        &prepared,
        observer,
    );
    let run_result = finalize_execution_result(session_id, tool_call, &prepared, result, true, false);

    (prepared.route, run_result)
}

pub(super) fn execute_tool_calls(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_calls: &[ToolCall],
    observer: Option<&mut dyn ToolExecutionObserver>,
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return tokio::task::block_in_place(|| {
            execute_tool_calls_inner(session_id, mcp_client, shared_mcp_client, tool_calls, observer)
        });
    }
    execute_tool_calls_inner(session_id, mcp_client, shared_mcp_client, tool_calls, observer)
}

fn execute_tool_calls_inner(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_calls: &[ToolCall],
    mut observer: Option<&mut dyn ToolExecutionObserver>,
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    let mut executed_tool_calls = Vec::with_capacity(tool_calls.len());
    let mut tool_results = Vec::with_capacity(tool_calls.len());
    let mut cached_hits = Vec::with_capacity(tool_calls.len());

    let mut idx = 0usize;
    while idx < tool_calls.len() {
        if crate::ai::tools::registry::common::is_tool_cancel_requested() {
            for deferred in &tool_calls[idx..] {
                println!("\n[Deferred] {}", deferred.function.name.yellow());
            }
            break;
        }

        let batch_len = parallel_task_batch_len(mcp_client, tool_calls, idx);
        if batch_len > 1 {
            let batch = &tool_calls[idx..idx + batch_len];
            let batch_results = execute_parallel_task_batch(session_id, mcp_client, batch);
            for (tool_call, (route, run_result)) in batch.iter().zip(batch_results.into_iter()) {
                executed_tool_calls.push(tool_call.clone());
                cached_hits.push(run_result.cached);
                let should_barrier = barrier::should_barrier_after(
                    &route,
                    tool_call,
                    run_result.ok,
                    &run_result.tool_result.content,
                );
                notify_tool_finished(&mut observer, tool_call, &run_result);
                print_run_status(tool_call, &run_result);
                tool_results.push(run_result.tool_result);
                if should_barrier {
                    for deferred in &tool_calls[idx + batch_len..] {
                        println!("\n[Deferred] {}", deferred.function.name.yellow());
                    }
                    return Ok(ExecuteToolCallsResult {
                        executed_tool_calls,
                        tool_results,
                        cached_hits,
                    });
                }
                if crate::ai::tools::registry::common::is_tool_cancel_requested() {
                    for deferred in &tool_calls[idx + batch_len..] {
                        println!("\n[Deferred] {}", deferred.function.name.yellow());
                    }
                    return Ok(ExecuteToolCallsResult {
                        executed_tool_calls,
                        tool_results,
                        cached_hits,
                    });
                }
            }
            idx += batch_len;
            continue;
        }

        let tool_call = &tool_calls[idx];
        let is_last = idx + 1 >= tool_calls.len();
        let (route, run_result) =
            run_one(mcp_client, shared_mcp_client, session_id, tool_call, &mut observer);
        let should_barrier = barrier::should_barrier_after(
            &route,
            tool_call,
            run_result.ok,
            &run_result.tool_result.content,
        );

        executed_tool_calls.push(tool_call.clone());
        cached_hits.push(run_result.cached);
        notify_tool_finished(&mut observer, tool_call, &run_result);
        print_run_status(tool_call, &run_result);
        tool_results.push(run_result.tool_result);

        if should_barrier && !is_last {
            for deferred in &tool_calls[idx + 1..] {
                println!("\n[Deferred] {}", deferred.function.name.yellow());
            }
            break;
        }

        if crate::ai::tools::registry::common::is_tool_cancel_requested() {
            for deferred in &tool_calls[idx + 1..] {
                println!("\n[Deferred] {}", deferred.function.name.yellow());
            }
            break;
        }
        idx += 1;
    }

    Ok(ExecuteToolCallsResult {
        executed_tool_calls,
        tool_results,
        cached_hits,
    })
}

fn notify_tool_finished(
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
    tool_call: &ToolCall,
    run_result: &RunOneResult,
) {
    if let Some(observer) = observer.as_deref_mut() {
        observer.on_tool_finished(tool_call, run_result);
    }
}

fn load_cached_tool_result(session_id: &str, tool_call: &ToolCall, args: &Value) -> Option<ToolResult> {
    if !is_cacheable_tool_name(&tool_call.function.name) {
        return None;
    }
    let source = format!("session:{session_id}");
    let cache_key = build_tool_cache_key(&tool_call.function.name, args);
    let store = MemoryStore::from_env_or_config();
    let entries = store.recent(TOOL_CACHE_RECENT_LIMIT).ok()?;
    for entry in entries {
        if entry.category != "tool_cache" {
            continue;
        }
        if !is_tool_cache_entry_fresh(&entry) {
            continue;
        }
        if entry.source.as_deref() != Some(source.as_str()) {
            continue;
        }
        if entry.tags.first().map(String::as_str) != Some(tool_call.function.name.as_str()) {
            continue;
        }
        if entry.tags.get(1).map(String::as_str) != Some(cache_key.as_str()) {
            continue;
        }
        let payload = serde_json::from_str::<ToolCachePayload>(&entry.note).ok()?;
        if payload.tool_name != tool_call.function.name || payload.args != *args {
            continue;
        }
        return Some(ToolResult {
            tool_call_id: tool_call.id.clone(),
            content: payload.result,
        });
    }
    None
}

fn is_tool_cache_entry_fresh(entry: &AgentMemoryEntry) -> bool {
    let Ok(timestamp) = DateTime::parse_from_rfc3339(&entry.timestamp) else {
        return false;
    };
    let timestamp = timestamp.with_timezone(&Utc);
    Utc::now().signed_duration_since(timestamp) <= Duration::minutes(TOOL_CACHE_TTL_MINUTES)
}

fn store_tool_cache_result(session_id: &str, tool_call: &ToolCall, args: &Value, tool_result: &ToolResult) {
    if !is_cacheable_tool_name(&tool_call.function.name) {
        return;
    }
    if tool_result.content.trim().is_empty() || tool_result.content.starts_with("Error:") {
        return;
    }
    let payload = ToolCachePayload {
        tool_name: tool_call.function.name.clone(),
        args: args.clone(),
        result: truncate_chars(&tool_result.content, TOOL_CACHE_MAX_RESULT_CHARS),
    };
    let Ok(note) = serde_json::to_string(&payload) else {
        return;
    };
    let cache_key = build_tool_cache_key(&tool_call.function.name, args);
    let entry = AgentMemoryEntry {
        id: None,
        timestamp: Local::now().to_rfc3339(),
        category: "tool_cache".to_string(),
        note,
        tags: vec![tool_call.function.name.clone(), cache_key],
        source: Some(format!("session:{session_id}")),
        priority: Some(80),
        owner_pid: None,
        owner_pgid: None,
    };
    let store = MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

fn is_cacheable_tool_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let mutating = [
        "create", "delete", "remove", "update", "write", "save", "append", "insert",
        "rename", "move", "install", "run", "execute", "oauth", "open_browser",
        "report_event", "memory", "kill_terminal", "edit", "apply_patch",
    ];
    if mutating.iter().any(|needle| lower.contains(needle)) {
        return false;
    }
    let reusable = ["search", "read", "get", "list", "view", "fetch", "export"];
    reusable.iter().any(|needle| lower.contains(needle))
}

fn build_tool_cache_key(name: &str, args: &Value) -> String {
    let args_json = serde_json::to_string(args).unwrap_or_else(|_| args.to_string());
    format!("{:x}", md5::compute(format!("{name}\n{args_json}")))
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 || s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::{
        aggregate_async_tool_pipe_messages, async_tool_pipe_message_from_final,
        async_tool_pipe_message_from_started, async_tool_pipe_message_from_stream,
        build_tool_cache_key, delete_async_tool_snapshot, is_cacheable_tool_name,
        is_tool_cache_entry_fresh, load_async_tool_snapshot, lookup_wait_sources,
        parallel_task_batch_len, persist_async_tool_snapshot, send_async_tool_pipe_message,
        stream_preview_from_aggregate, AsyncToolEntry, AsyncToolPipeKind, AsyncToolPipeMessage,
        AsyncToolState, RunOneResult, ToolRoute, ASYNC_TOOL_REGISTRY, TOOL_CACHE_TTL_MINUTES,
    };
    use aios_kernel::{kernel::EventId, primitives::ChannelId};
    use crate::ai::tools::task_tools::WaitManySource;
    use crate::ai::tools::registry::common::current_process_tool_cancel_futex;
    use crate::ai::mcp::McpClient;
    use crate::ai::types::{FunctionCall, ToolCall};
    use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
    use chrono::{Duration, Utc};
    use serde_json::json;
    use std::sync::{LazyLock, Mutex, MutexGuard};

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
        assert!(is_cacheable_tool_name("grep_search"));
        assert!(!is_cacheable_tool_name("create_file"));
        assert!(!is_cacheable_tool_name("execute_command"));
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
        };
        let stale = AgentMemoryEntry {
            timestamp: (Utc::now() - Duration::minutes(TOOL_CACHE_TTL_MINUTES + 1)).to_rfc3339(),
            ..fresh.clone()
        };
        assert!(is_tool_cache_entry_fresh(&fresh));
        assert!(!is_tool_cache_entry_fresh(&stale));
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
    fn parallel_task_batch_len_only_groups_contiguous_task_calls() {
        let client = McpClient::new();
        let tool_calls = vec![tool_call("task"), tool_call("task"), tool_call("read_file"), tool_call("task")];

        assert_eq!(parallel_task_batch_len(&client, &tool_calls, 0), 2);
        assert_eq!(parallel_task_batch_len(&client, &tool_calls, 2), 0);
        assert_eq!(parallel_task_batch_len(&client, &tool_calls, 3), 1);
    }

    #[test]
    fn async_tool_snapshot_roundtrip_uses_channel() {
        let (_guard, kernel, root) = setup_async_tool_kernel();
        let channel_id = {
            let mut os = kernel.lock().unwrap();
            os.channel_create(Some(root), 1, "async-tool-test".to_string()).raw()
        };
        let entry = sample_completed_entry(Some(channel_id));

        persist_async_tool_snapshot("tooltask_1", &entry);
        let snapshot = load_async_tool_snapshot(&entry).expect("snapshot should be readable via channel");
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
            current_process_tool_cancel_futex(os.as_mut()).unwrap().unwrap()
        };
        assert_eq!(
            wait_sources,
            vec![
                WaitManySource::Channel(channel_id),
                WaitManySource::Event(EventId::new(
                    {
                        let os = kernel.lock().unwrap();
                        os.channel_event_id(ChannelId(channel_id)).unwrap().as_u64()
                    }
                )),
                WaitManySource::Futex { addr: futex_addr, expected: 0 },
                WaitManySource::Futex { addr: cancel_futex, expected: 0 },
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
        assert_eq!(stream_preview_from_aggregate(&agg).as_deref(), Some("hello "));
    }

    #[test]
    fn load_async_tool_snapshot_reads_streaming_pipe_messages() {
        let (_guard, kernel, root) = setup_async_tool_kernel();
        let channel_id = {
            let mut os = kernel.lock().unwrap();
            os.channel_create(Some(root), 8, "async-tool-pipe".to_string()).raw()
        };
        let entry = AsyncToolEntry {
            result_channel_id: Some(channel_id),
            completion_futex_addr: None,
            session_id: "sess-stream".to_string(),
            tool_name: "execute_command".to_string(),
            started_at: std::time::Instant::now(),
            state: AsyncToolState::Running,
        };

        send_async_tool_pipe_message(&entry, &async_tool_pipe_message_from_started("tooltask_stream", &entry, 0));
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

        let snapshot = load_async_tool_snapshot(&final_entry).expect("snapshot should decode from pipe");
        assert_eq!(snapshot.status, "completed");
        assert_eq!(snapshot.content.as_deref(), Some("partial final"));
    }

    #[test]
    fn delete_async_tool_snapshot_destroys_result_pipe() {
        let (_guard, kernel, root) = setup_async_tool_kernel();
        let channel_id = {
            let mut os = kernel.lock().unwrap();
            os.channel_create(Some(root), 4, "async-tool-destroy".to_string()).raw()
        };
        let entry = sample_completed_entry(Some(channel_id));
        persist_async_tool_snapshot("tooltask_destroy", &entry);

        delete_async_tool_snapshot(&entry);

        let os = kernel.lock().unwrap();
        assert!(os.channel_event_id(ChannelId(channel_id)).is_none());
    }
}

pub(super) fn penalty_for_skill_tools(skill: &crate::ai::skills::SkillManifest) -> f64 {
    if skill.tools.is_empty() {
        return 0.0;
    }
    let tools = &skill.tools;
    let Ok(map) = TOOL_FAILURES.lock() else {
        return 0.0;
    };
    let mut score = 0.0f64;
    for t in tools {
        if let Some(c) = map.get(t) {
            score += (*c as f64).min(10.0);
        }
    }
    score
}
