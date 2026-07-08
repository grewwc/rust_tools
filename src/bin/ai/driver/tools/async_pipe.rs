//! 异步工具管道系统（async tool pipe）。
//!
//! 管理 task_spawn/task_status/task_wait 异步工具的生命周期：
//! - `AsyncToolEntry` / `AsyncToolState`：注册表项与状态机
//! - channel / futex 创建与信号：`create_async_tool_channel` 等
//! - pipe 消息流：`AsyncToolPipeMessage` / `send_async_tool_pipe_message` / `parse_async_tool_pipe_messages`
//! - `AsyncToolPipeObserver`：将工具执行事件桥接为 pipe 消息
//! - 快照持久化：`persist_async_tool_snapshot` / `load_async_tool_snapshot` / `delete_async_tool_snapshot`
//! - 等待源解析：`lookup_wait_sources` / `parse_wait_policy` / `collect_async_task_snapshot`

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use rust_tools::cw::SkipMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use aios_kernel::primitives::{ChannelId, ChannelOwnerTag, FutexAddr};
use aios_kernel::kernel::EventId;
use super::ToolExecutionObserver;
use crate::ai::tools::os_tools::GLOBAL_OS;
use aios_kernel::kernel::WaitPolicy;
use crate::ai::tools::task_tools::{
    WaitManySource, append_current_process_cancel_source,
    wait_sources_for_channel_and_futex,
};
use crate::ai::types::ToolCall;

use super::{RunOneResult, ToolRoute};

pub(super) const ASYNC_TOOL_REGISTRY_LIMIT: usize = 200;

pub(super) enum AsyncToolState {
    Running,
    Completed((ToolRoute, RunOneResult)),
    Canceled { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AsyncToolSnapshot {
    pub(super) task_id: String,
    pub(super) event_id: String,
    pub(super) session_id: String,
    pub(super) tool_name: String,
    pub(super) status: String,
    pub(super) ok: Option<bool>,
    pub(super) cached: Option<bool>,
    pub(super) executed: Option<bool>,
    pub(super) reason: Option<String>,
    pub(super) content: Option<String>,
    pub(super) elapsed_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum AsyncToolPipeKind {
    Started,
    StreamChunk,
    Final,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AsyncToolPipeMessage {
    pub(super) task_id: String,
    pub(super) session_id: String,
    pub(super) tool_name: String,
    pub(super) seq: u64,
    pub(super) kind: AsyncToolPipeKind,
    pub(super) content: Option<String>,
    pub(super) status: Option<String>,
    pub(super) ok: Option<bool>,
    pub(super) cached: Option<bool>,
    pub(super) executed: Option<bool>,
    pub(super) reason: Option<String>,
    pub(super) elapsed_secs: f64,
}

#[derive(Debug, Default)]
pub(super) struct AsyncToolPipeAggregate {
    pub(crate) chunks: Vec<String>,
    pub(crate) last_status: Option<String>,
    pub(crate) ok: Option<bool>,
    pub(crate) cached: Option<bool>,
    pub(crate) executed: Option<bool>,
    pub(crate) reason: Option<String>,
    pub(crate) final_content: Option<String>,
    pub(crate) stream_chunk_count: usize,
}

pub(super) struct AsyncToolEntry {
    pub(super) result_channel_id: Option<u64>,
    pub(super) completion_futex_addr: Option<FutexAddr>,
    pub(super) session_id: String,
    pub(super) tool_name: String,
    pub(super) started_at: Instant,
    pub(super) state: AsyncToolState,
}

pub(super) static ASYNC_TOOL_NEXT_ID: AtomicU64 = AtomicU64::new(1);
pub(super) static ASYNC_TOOL_REGISTRY: LazyLock<Mutex<SkipMap<String, AsyncToolEntry>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));

pub(super) fn next_async_tool_id() -> String {
    format!(
        "tooltask_{}",
        ASYNC_TOOL_NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}

pub(super) fn create_async_tool_channel(task_id: &str) -> Option<u64> {
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

pub(super) fn create_async_tool_completion_futex(task_id: &str) -> Option<FutexAddr> {
    let guard = GLOBAL_OS.lock().ok()?;
    let os = guard.as_ref()?;
    let mut os = os.lock().ok()?;
    Some(os.futex_create(0, format!("async_tool_completion:{task_id}")))
}

pub(super) fn signal_async_tool_completion(addr: Option<FutexAddr>) {
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

pub(super) fn prune_completed_async_tools(registry: &mut SkipMap<String, AsyncToolEntry>) {
    if registry.len() <= ASYNC_TOOL_REGISTRY_LIMIT {
        return;
    }
    let completed_keys: Vec<String> = registry
        .iter()
        .filter(|(_, entry)| {
            matches!(
                entry.state,
                AsyncToolState::Completed(_) | AsyncToolState::Canceled { .. }
            )
        })
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

pub(super) fn async_tool_event_id(entry: &AsyncToolEntry) -> Option<EventId> {
    let channel_id = entry.result_channel_id?;
    let guard = GLOBAL_OS.lock().ok()?;
    let os = guard.as_ref()?;
    let os = os.lock().ok()?;
    os.channel_event_id(ChannelId(channel_id))
}

pub(super) fn async_tool_snapshot_from_entry(task_id: &str, entry: &AsyncToolEntry) -> AsyncToolSnapshot {
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

pub(super) fn async_tool_pipe_message_from_started(
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

pub(super) fn async_tool_pipe_message_from_stream(
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

pub(super) fn async_tool_pipe_message_from_final(
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

pub(super) fn send_async_tool_pipe_message(entry: &AsyncToolEntry, msg: &AsyncToolPipeMessage) {
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

pub(super) fn close_async_tool_pipe(entry: &AsyncToolEntry) {
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

pub(super) fn release_async_tool_pipe_ref(entry: &AsyncToolEntry, holder: &str) {
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

pub(super) fn parse_async_tool_pipe_messages(payloads: &[String]) -> Vec<AsyncToolPipeMessage> {
    payloads
        .iter()
        .filter_map(|payload| serde_json::from_str::<AsyncToolPipeMessage>(payload).ok())
        .collect()
}

pub(super) fn peek_async_tool_pipe_messages(entry: &AsyncToolEntry) -> Vec<AsyncToolPipeMessage> {
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

pub(super) fn drain_async_tool_pipe_messages(entry: &AsyncToolEntry) -> Vec<AsyncToolPipeMessage> {
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

pub(super) fn aggregate_async_tool_pipe_messages(messages: &[AsyncToolPipeMessage]) -> AsyncToolPipeAggregate {
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

pub(super) fn truncate_stream_preview(text: &str, max_chars: usize) -> String {
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

pub(super) fn stream_preview_from_aggregate(agg: &AsyncToolPipeAggregate) -> Option<String> {
    if agg.chunks.is_empty() {
        return None;
    }
    Some(truncate_stream_preview(&agg.chunks.join(""), 2000))
}

pub(super) fn async_tool_result_json(
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

pub(super) struct AsyncToolPipeObserver {
    pub(super) task_id: String,
    pub(super) session_id: String,
    pub(super) tool_name: String,
    pub(super) result_channel_id: Option<u64>,
    pub(super) completion_futex_addr: Option<FutexAddr>,
    pub(super) started_at: Instant,
    pub(super) next_seq: u64,
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
                AsyncToolState::Completed((
                    ToolRoute::Builtin,
                    RunOneResult {
                        tool_result: run_result.tool_result.clone(),
                        ok: run_result.ok,
                        executed: run_result.executed,
                        cached: run_result.cached,
                    },
                ))
            } else {
                AsyncToolState::Completed((
                    ToolRoute::Builtin,
                    RunOneResult {
                        tool_result: run_result.tool_result.clone(),
                        ok: run_result.ok,
                        executed: run_result.executed,
                        cached: run_result.cached,
                    },
                ))
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

pub(super) fn persist_async_tool_snapshot(task_id: &str, entry: &AsyncToolEntry) {
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

pub(super) fn load_async_tool_snapshot(entry: &AsyncToolEntry) -> Option<AsyncToolSnapshot> {
    let messages = peek_async_tool_pipe_messages(entry);
    let agg = aggregate_async_tool_pipe_messages(&messages);
    Some(AsyncToolSnapshot {
        task_id: messages.first()?.task_id.clone(),
        event_id: async_tool_event_id(entry)
            .map(|id| id.to_string())
            .unwrap_or_else(|| "evt_unavailable".to_string()),
        session_id: messages.first()?.session_id.clone(),
        tool_name: messages.first()?.tool_name.clone(),
        status: agg.last_status.unwrap_or_else(|| "running".to_string()),
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

pub(super) fn delete_async_tool_snapshot(entry: &AsyncToolEntry) {
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

pub(super) fn lookup_wait_sources(
    os: &mut dyn aios_kernel::kernel::Kernel,
    session_id: &str,
    task_ids: &[String],
) -> Result<Vec<WaitManySource>, String> {
    let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
    let mut wait_sources = Vec::new();
    for task_id in task_ids {
        let Some(entry) = registry.get_ref(task_id) else {
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

pub(super) fn parse_wait_policy(args: &Value) -> Result<WaitPolicy, String> {
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

pub(super) fn collect_async_task_snapshot(
    session_id: &str,
    task_ids: &[String],
) -> Result<(Vec<Value>, Vec<Value>), String> {
    let mut terminal_results = Vec::new();
    let mut pending = Vec::new();
    let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
    for task_id in task_ids {
        let Some(entry) = registry.get_ref(task_id) else {
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
                        obj.insert(
                            "content".to_string(),
                            json!(snapshot.content.unwrap_or_default()),
                        );
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
