use rustc_hash::FxHashSet;
use serde_json::Value;

use crate::ai::{
    history::{Message, last_real_user_index},
    types::App,
};

use super::mid_turn_compress_soft_threshold;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentKind {
    SystemPrompt,
    CurrentUser,
    RecentUser,
    PrecisionToolResult,
    ToolResult,
    InternalNote,
    Assistant,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SegmentPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompressionMode {
    Never,
    OffloadOnly,
    SafeLossy,
}

#[derive(Debug, Clone)]
struct ContextSegment {
    index: usize,
    kind: SegmentKind,
    priority: SegmentPriority,
    compression: CompressionMode,
    chars: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ContextBudgetRollbackReason {
    NoAdditionalSavings,
    ProtectedContextChanged,
}

impl ContextBudgetRollbackReason {
    pub(super) fn note(self) -> &'static str {
        match self {
            ContextBudgetRollbackReason::NoAdditionalSavings => {
                "lossy compression rolled back because it did not improve beyond lossless prepass"
            }
            ContextBudgetRollbackReason::ProtectedContextChanged => {
                "compression rolled back because protected system/current-user context changed"
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct ContextBudgetReport {
    pub(super) before_chars: usize,
    pub(super) after_chars: usize,
    pub(super) target_chars: usize,
    pub(super) changed: bool,
    pub(super) rolled_back: bool,
    pub(super) rollback_reason: Option<ContextBudgetRollbackReason>,
    pub(super) critical_segments: usize,
    pub(super) offload_only_segments: usize,
    pub(super) lossy_candidate_segments: usize,
    pub(super) lossy_candidate_chars: usize,
    pub(super) lossless_removed_messages: usize,
    pub(super) lossless_saved_chars: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct ProtectedMessage {
    role: String,
    content: Value,
    tool_calls: Option<Vec<crate::ai::types::ToolCall>>,
    tool_call_id: Option<String>,
    reasoning_content: Option<String>,
}

impl From<&Message> for ProtectedMessage {
    fn from(message: &Message) -> Self {
        Self {
            role: message.role.clone(),
            content: message.content.clone(),
            tool_calls: message.tool_calls.clone(),
            tool_call_id: message.tool_call_id.clone(),
            reasoning_content: message.reasoning_content.clone(),
        }
    }
}

pub(super) fn apply_pre_request_context_budget(
    app: &App,
    model: &str,
    messages: &mut Vec<Message>,
) -> ContextBudgetReport {
    let target_chars = mid_turn_compress_soft_threshold(model, app.config.history_max_chars);
    let scan = quick_scan(messages);
    let mut report = ContextBudgetReport {
        before_chars: scan.total_chars,
        after_chars: scan.total_chars,
        target_chars,
        ..ContextBudgetReport::default()
    };

    if scan.total_chars <= target_chars && !scan.has_lossless_candidate {
        return report;
    }

    let mut after_lossless_chars = scan.total_chars;
    if scan.has_lossless_candidate {
        let lossless = apply_lossless_prepass(messages);
        report.lossless_removed_messages = lossless.removed_messages;
        report.lossless_saved_chars = lossless.saved_chars;
        if lossless.removed_messages > 0 {
            report.changed = true;
            after_lossless_chars = scan.total_chars.saturating_sub(lossless.saved_chars);
            report.after_chars = after_lossless_chars;
        }
    }

    if after_lossless_chars <= target_chars {
        if report.changed {
            fill_segment_summary(&mut report, messages);
        }
        return report;
    }

    fill_segment_summary(&mut report, messages);
    let protected = collect_protected_messages(messages);
    let overflow_dir = {
        use crate::ai::history::SessionStore;
        let store = SessionStore::new(app.config.history_file.as_path());
        store.session_assets_dir(&app.session_id)
    };
    let original = messages.clone();
    let drained = std::mem::take(messages);
    let (compressed, _, after_chars) =
        crate::ai::history::mid_turn_compress(drained, target_chars, Some(overflow_dir.as_path()));
    *messages = compressed;
    // mid_turn_compress 把压缩状态提示插在最后一个 user 之后（工具循环场景下这是
    // 当前轮活动区）。但请求边界要求 current user 必须是发送序列的最后一条，故这里
    // 把该提示前移到 current user 之前：既维持 user 末尾契约，又让模型仍能看到
    // 「本次用的是压缩投影、勿把可恢复证据误判为上下文已满」（见 CONTEXT_COMPACTION_STATE）。
    reposition_context_compaction_state_before_last_user(messages);
    report.after_chars = after_chars;
    report.changed = report.changed || after_chars < after_lossless_chars;

    let protected_preserved = protected_messages_preserved(messages, &protected);
    let rollback_reason = if !protected_preserved {
        Some(ContextBudgetRollbackReason::ProtectedContextChanged)
    } else if after_chars > after_lossless_chars {
        Some(ContextBudgetRollbackReason::NoAdditionalSavings)
    } else {
        None
    };

    if let Some(reason) = rollback_reason {
        *messages = original;
        report.after_chars = after_lossless_chars;
        report.changed = report.lossless_removed_messages > 0;
        report.rolled_back = after_chars < scan.total_chars;
        if report.rolled_back {
            report.rollback_reason = Some(reason);
        }
    }
    report
}

/// mid_turn_compress 会把 `CONTEXT_COMPACTION_STATE` 提示插在最后一个 user
/// 之后。请求边界要求 current user 必须是发送序列的最后一条，故这里把该提示前移
/// 到最后一个 user **之前**：既维持 user 末尾契约，又保住提示对模型的可见性。
/// note 内容原样搬移，不重构文本（单一来源仍在 compress 模块）。
fn reposition_context_compaction_state_before_last_user(messages: &mut Vec<Message>) {
    let Some(note_index) = messages
        .iter()
        .position(crate::ai::history::compress::is_context_compaction_state)
    else {
        return;
    };
    let Some(last_user_index) = last_real_user_index(messages) else {
        // 无 user 消息：请求边界契约不适用，保持原样。
        return;
    };
    // 已在最后一个 user 之前，无需移动。
    if note_index < last_user_index {
        return;
    }
    let note = messages.remove(note_index);
    // remove 发生在 last_user_index 之后，last_user_index 不变；插到它之前。
    messages.insert(last_user_index, note);
}

#[derive(Debug, Default)]
struct QuickScan {
    total_chars: usize,
    has_lossless_candidate: bool,
}

fn quick_scan(messages: &[Message]) -> QuickScan {
    let mut scan = QuickScan::default();
    let mut seen_internal_notes: FxHashSet<&Value> = FxHashSet::default();
    for message in messages {
        scan.total_chars = scan.total_chars.saturating_add(message_chars(message));
        if !scan.has_lossless_candidate {
            if is_empty_non_protocol_message(message) {
                scan.has_lossless_candidate = true;
            } else if message.role == crate::ai::history::ROLE_INTERNAL_NOTE
                && !seen_internal_notes.insert(&message.content)
            {
                scan.has_lossless_candidate = true;
            }
        }
    }
    scan
}

#[derive(Debug, Default)]
struct LosslessStats {
    removed_messages: usize,
    saved_chars: usize,
}

fn apply_lossless_prepass(messages: &mut Vec<Message>) -> LosslessStats {
    let mut seen_internal_notes: FxHashSet<String> = FxHashSet::default();
    let mut stats = LosslessStats::default();
    messages.retain(|message| {
        if is_empty_non_protocol_message(message) {
            stats.removed_messages += 1;
            stats.saved_chars = stats.saved_chars.saturating_add(message_chars(message));
            return false;
        }
        if message.role == crate::ai::history::ROLE_INTERNAL_NOTE {
            let key = stable_message_key(message);
            if !seen_internal_notes.insert(key) {
                stats.removed_messages += 1;
                stats.saved_chars = stats.saved_chars.saturating_add(message_chars(message));
                return false;
            }
        }
        true
    });
    stats
}

fn is_empty_non_protocol_message(message: &Message) -> bool {
    if message.role == "system" || message.role == "user" || message.role == "tool" {
        return false;
    }
    if message
        .tool_calls
        .as_ref()
        .map(|calls| !calls.is_empty())
        .unwrap_or(false)
        || message.tool_call_id.is_some()
        || message
            .reasoning_content
            .as_deref()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
    {
        return false;
    }
    content_text_is_empty(&message.content)
}

fn stable_message_key(message: &Message) -> String {
    format!(
        "{}\n{}\n{:?}\n{:?}\n{:?}",
        message.role,
        message.content,
        message.tool_calls,
        message.tool_call_id,
        message.reasoning_content
    )
}

fn summarize_segments(
    before_chars: usize,
    target_chars: usize,
    segments: &[ContextSegment],
) -> ContextBudgetReport {
    ContextBudgetReport {
        before_chars,
        after_chars: before_chars,
        target_chars,
        changed: false,
        rolled_back: false,
        rollback_reason: None,
        critical_segments: segments
            .iter()
            .filter(|segment| segment.priority == SegmentPriority::Critical)
            .count(),
        offload_only_segments: segments
            .iter()
            .filter(|segment| segment.compression == CompressionMode::OffloadOnly)
            .count(),
        lossy_candidate_segments: segments
            .iter()
            .filter(|segment| segment.compression == CompressionMode::SafeLossy)
            .count(),
        lossy_candidate_chars: segments
            .iter()
            .filter(|segment| segment.compression == CompressionMode::SafeLossy)
            .map(|segment| segment.chars)
            .sum(),
        lossless_removed_messages: 0,
        lossless_saved_chars: 0,
    }
}

fn fill_segment_summary(report: &mut ContextBudgetReport, messages: &[Message]) {
    let segments = classify_segments(messages);
    let summary = summarize_segments(report.before_chars, report.target_chars, &segments);
    report.critical_segments = summary.critical_segments;
    report.offload_only_segments = summary.offload_only_segments;
    report.lossy_candidate_segments = summary.lossy_candidate_segments;
    report.lossy_candidate_chars = summary.lossy_candidate_chars;
}

fn classify_segments(messages: &[Message]) -> Vec<ContextSegment> {
    // 合成 user 消息不构成轮次边界：真实用户消息必须保持 Critical/Never 保护，
    // 否则会被降级为 RecentUser 而可被 offload。
    let last_user_index = last_real_user_index(messages);
    let precision_tool_ids = precision_tool_call_ids(messages);
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let chars = message_chars(message);
            let (kind, priority, compression) =
                classify_message(message, index, last_user_index, &precision_tool_ids);
            ContextSegment {
                index,
                kind,
                priority,
                compression,
                chars,
            }
        })
        .collect()
}

fn classify_message(
    message: &Message,
    index: usize,
    last_user_index: Option<usize>,
    precision_tool_ids: &rustc_hash::FxHashSet<String>,
) -> (SegmentKind, SegmentPriority, CompressionMode) {
    if message.role == "system" {
        return (
            SegmentKind::SystemPrompt,
            SegmentPriority::Critical,
            CompressionMode::Never,
        );
    }
    if message.role == "user" && Some(index) == last_user_index {
        return (
            SegmentKind::CurrentUser,
            SegmentPriority::Critical,
            CompressionMode::Never,
        );
    }
    if message.role == "user" {
        return (
            SegmentKind::RecentUser,
            SegmentPriority::High,
            CompressionMode::OffloadOnly,
        );
    }
    if message.role == "tool" {
        let precision = message
            .tool_call_id
            .as_ref()
            .map(|id| precision_tool_ids.contains(id))
            .unwrap_or(false);
        if precision {
            return (
                SegmentKind::PrecisionToolResult,
                SegmentPriority::High,
                CompressionMode::OffloadOnly,
            );
        }
        return (
            SegmentKind::ToolResult,
            SegmentPriority::Medium,
            CompressionMode::SafeLossy,
        );
    }
    if message.role == crate::ai::history::ROLE_INTERNAL_NOTE {
        return (
            SegmentKind::InternalNote,
            SegmentPriority::Medium,
            CompressionMode::SafeLossy,
        );
    }
    if message.role == "assistant" {
        return (
            SegmentKind::Assistant,
            SegmentPriority::Medium,
            CompressionMode::SafeLossy,
        );
    }
    (
        SegmentKind::Other,
        SegmentPriority::Low,
        CompressionMode::SafeLossy,
    )
}

fn precision_tool_call_ids(messages: &[Message]) -> rustc_hash::FxHashSet<String> {
    let mut out = rustc_hash::FxHashSet::default();
    for message in messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            if is_precision_tool(&tool_call.function.name) {
                out.insert(tool_call.id.clone());
            }
        }
    }
    out
}

fn is_precision_tool(tool_name: &str) -> bool {
    matches!(tool_name, "read_file")
}

fn collect_protected_messages(messages: &[Message]) -> Vec<ProtectedMessage> {
    let last_user_index = last_real_user_index(messages);
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            (message.role == "system" || (message.role == "user" && Some(index) == last_user_index))
                .then(|| ProtectedMessage::from(message))
        })
        .collect()
}

fn protected_messages_preserved(messages: &[Message], protected: &[ProtectedMessage]) -> bool {
    if protected.is_empty() {
        return true;
    }
    let current = collect_protected_messages(messages);
    current == protected
}

fn message_chars(message: &Message) -> usize {
    // 统一走 history 层的权威计费口径（含 content + tool_calls + reasoning_content，
    // 图片按名义成本），避免此处只算 content 导致带大 tool_calls/reasoning 的消息
    // 在预算门控里被低估。
    crate::ai::history::message_billable_chars(message)
}

fn content_text_is_empty(content: &Value) -> bool {
    match content {
        Value::String(text) => text.trim().is_empty(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .all(|text| text.trim().is_empty()),
        other => other.to_string().trim().is_empty(),
    }
}

#[cfg(test)]
fn message_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    use serde_json::Value;

    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        history::Message,
        types::{App, AppConfig, FunctionCall, ToolCall},
    };

    fn test_app(history_file: PathBuf) -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                base_history_file: history_file.clone(),
                history_file: history_file.clone(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 1_000,
                history_keep_last: 256,
                history_summary_max_chars: 4_000,
                intent_model: None,
            },
            session_id: "test".to_string(),
            session_history_file: history_file,
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skill: None,
            forced_skill_source: None,
            pending_skill_continuation: None,
            forced_question: None,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            prompt_editor: None,
            agent_context: None,
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: Vec::new(),
            last_known_prompt_tokens: None,
            last_known_cached_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
            last_turn_interrupted: false,
            prune_marks: Default::default(),
            turn_reasoning_items: Default::default(),
            stale_patch_targets: Default::default(),
        }
    }

    fn msg(role: &str, content: impl Into<String>) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(content.into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn assistant_tool_call(id: &str, name: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn tool_result(id: &str, content: impl Into<String>) -> Message {
        Message {
            role: "tool".to_string(),
            content: Value::String(content.into()),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            reasoning_content: None,
        }
    }

    #[test]
    fn context_budget_preserves_system_and_current_user_exactly() {
        let history_file = std::env::temp_dir().join(format!(
            "context-budget-preserve-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);
        let system = msg("system", "system prompt must stay exact");
        let current_user = msg("user", "latest user input must stay exact");
        let mut messages = vec![
            system.clone(),
            msg("assistant", "old narration ".repeat(4_000)),
            current_user.clone(),
        ];

        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);

        assert!(report.before_chars > report.target_chars);
        assert_eq!(messages[0], system);
        assert_eq!(messages.last().unwrap(), &current_user);
    }

    #[test]
    fn context_budget_keeps_compaction_state_visible_before_last_user() {
        let history_file = std::env::temp_dir().join(format!(
            "context-budget-compaction-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);
        let current_user = msg("user", "latest user input must stay exact");
        let mut messages = vec![
            msg("system", "system prompt must stay exact"),
            msg("assistant", "old narration ".repeat(4_000)),
            current_user.clone(),
        ];

        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);

        // 实际压缩生效才会注入压缩状态提示。
        assert!(report.changed);
        // current user 仍是发送序列最后一条（请求边界契约）。
        assert_eq!(messages.last().unwrap(), &current_user);
        // 压缩状态提示对模型可见，且被前移到最后一个 user 之前。
        let note_index = messages
            .iter()
            .position(crate::ai::history::compress::is_context_compaction_state)
            .expect("compaction state note must remain visible to the model");
        let last_user_index = last_real_user_index(&messages).expect("current user present");
        assert!(
            note_index < last_user_index,
            "compaction note must sit before the last user message"
        );
    }

    #[test]
    fn context_budget_runs_lossless_prepass_without_budget_pressure() {
        let history_file = std::env::temp_dir().join(format!(
            "context-budget-lossless-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);
        let system = msg("system", "system prompt must stay exact");
        let current_user = msg("user", "latest user input must stay exact");
        let duplicate_note = msg(crate::ai::history::ROLE_INTERNAL_NOTE, "same reminder");
        let tool_call = assistant_tool_call("call-1", "read_file");
        let mut messages = vec![
            system.clone(),
            duplicate_note.clone(),
            msg("assistant", "   "),
            duplicate_note,
            tool_call.clone(),
            current_user.clone(),
        ];

        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);

        assert!(report.changed);
        assert_eq!(report.lossless_removed_messages, 2);
        assert!(report.lossless_saved_chars > 0);
        assert_eq!(messages[0], system);
        assert_eq!(messages.last().unwrap(), &current_user);
        assert!(messages.iter().any(|message| message == &tool_call));
        assert_eq!(
            messages
                .iter()
                .filter(|message| message.role == crate::ai::history::ROLE_INTERNAL_NOTE)
                .count(),
            1
        );
    }

    #[test]
    fn context_budget_classifies_precision_tools_as_offload_only() {
        let messages = vec![
            msg("system", "s"),
            assistant_tool_call("call-1", "read_file"),
            tool_result("call-1", "src/main.rs:1: fn main()"),
            msg("user", "current"),
        ];

        let segments = classify_segments(&messages);
        let tool_segment = segments
            .iter()
            .find(|segment| segment.index == 2)
            .expect("tool segment");

        assert_eq!(tool_segment.kind, SegmentKind::PrecisionToolResult);
        assert_eq!(tool_segment.compression, CompressionMode::OffloadOnly);
        assert_eq!(tool_segment.priority, SegmentPriority::High);
    }

    #[test]
    fn context_budget_offloads_large_precision_tool_without_lossy_summary() {
        let history_file = std::env::temp_dir().join(format!(
            "context-budget-precision-offload-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);
        let current_user = msg("user", "latest user input must stay exact");
        let exact_output = (0..600usize)
            .map(|idx| {
                format!(
                    "src/main.rs:{}: precise match {}\n",
                    idx + 1,
                    "x".repeat(80)
                )
            })
            .collect::<String>();
        // 大体量 read_file 结果必须位于「最近 6 条工具结果」保护窗之外才会被外溢，
        // 否则近端窗口会逐字保留（防止刚检索到的内容被卸载导致模型重复检索）。
        let mut messages = vec![
            msg("system", "system prompt must stay exact"),
            assistant_tool_call("call-1", "read_file"),
            tool_result("call-1", exact_output.clone()),
        ];
        for i in 0..6usize {
            let id = format!("recent-{i}");
            messages.push(assistant_tool_call(&id, "execute_command"));
            messages.push(tool_result(&id, format!("recent tool output {i}")));
        }
        messages.push(current_user.clone());

        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);

        assert!(report.changed);
        assert_eq!(messages.last().unwrap(), &current_user);
        let tool_content = messages
            .iter()
            .find(|message| {
                message.role == "tool" && message.tool_call_id.as_deref() == Some("call-1")
            })
            .and_then(|message| message.content.as_str())
            .expect("tool content");
        assert!(tool_content.contains("Output preserved for tool `read_file`"));
        assert!(tool_content.contains("- file_path:"));
        assert!(!tool_content.contains("tool_output_lines:"));
    }

    /// 回归覆盖两条路径：工具密集历史经常规压缩已达标时，不调用有损 LLM 摘要；
    /// 只有压缩后仍超阈值的对话密集历史才调用 LLM，并有效缩小上下文。
    #[tokio::test]
    async fn llm_summary_runs_only_when_post_compression_context_still_exceeds_threshold() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // 1. 本地 mock LLM 服务器：完整读取 Content-Length 后返回 OpenAI 格式响应
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let served = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let served_clone = served.clone();
        let server = std::thread::spawn(move || {
            let _ = listener.set_nonblocking(true);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while std::time::Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut sock, _)) => {
                        let _ = sock.set_read_timeout(Some(std::time::Duration::from_secs(5)));
                        // 读请求头
                        let mut buf = [0u8; 8192];
                        let mut header = Vec::new();
                        loop {
                            let n = sock.read(&mut buf).unwrap_or(0);
                            if n == 0 {
                                break;
                            }
                            header.extend_from_slice(&buf[..n]);
                            if header.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                        // 读完整 body（按 Content-Length）
                        let head = String::from_utf8_lossy(&header).to_string();
                        let len: usize = head
                            .lines()
                            .find_map(|l| {
                                let l = l.trim();
                                l.strip_prefix("Content-Length:")
                                    .or_else(|| l.strip_prefix("content-length:"))
                                    .and_then(|v| v.trim().parse().ok())
                            })
                            .unwrap_or(0);
                        // header 中可能已包含部分 body（\r\n\r\n 之后）
                        let body_start = header
                            .windows(4)
                            .position(|w| w == b"\r\n\r\n")
                            .map(|p| p + 4)
                            .unwrap_or(header.len());
                        let mut body = header[body_start..].to_vec();
                        let mut got = body.len();
                        while got < len {
                            let mut chunk = vec![0u8; len - got];
                            let n = sock.read(&mut chunk).unwrap_or(0);
                            if n == 0 {
                                break;
                            }
                            body.extend_from_slice(&chunk[..n]);
                            got += n;
                        }
                        let body_str = String::from_utf8_lossy(&body).to_string();
                        let body_preview: String = body_str.chars().take(200).collect();
                        assert!(
                            body_str.contains("摘要") || body_str.contains("summar"),
                            "mock 收到的请求体不像是摘要请求: {}",
                            body_preview
                        );
                        let resp_body = r#"{"choices":[{"message":{"content":"MOCK_SUMMARY: 早期工具调用与对话要点 1/2/3。"}}]}"#;
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp_body.len(),
                            resp_body
                        );
                        let _ = sock.write_all(resp.as_bytes());
                        served_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        break;
                    }
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
                }
            }
        });

        // 2. app：endpoint 指向 mock；history_max_chars 贴近生产默认
        let history_file = std::env::temp_dir().join(format!(
            "llm_summary_repro_{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let mut app = test_app(history_file);
        app.config.endpoint = format!("http://{addr}");
        app.config.history_max_chars = 90_000;
        app.session_id = format!("llm_summary_repro_{}", uuid::Uuid::new_v4());

        // 3. 工具密集超长会话：常规压缩可以安全降到阈值内，不应再做有损摘要。
        let mut messages = vec![msg("system", "你是测试助手，请遵循项目规则。")];
        for turn in 0..4 {
            messages.push(msg("user", format!("第 {turn} 轮：请帮我检查代码")));
            messages.push(assistant_tool_call(&format!("call_{turn}"), "read_file"));
            messages.push(tool_result(
                &format!("call_{turn}"),
                format!("line {turn}: {}", "x".repeat(60_000)),
            ));
            messages.push(msg(
                "assistant",
                format!("第 {turn} 轮完成：发现 {}。", "y".repeat(2_000)),
            ));
        }
        messages.push(msg("user", "最后：请总结以上所有结果"));

        let before = crate::ai::history::messages_total_chars_pub(&messages);
        assert!(
            before > 180_000,
            "测试会话应远超 pre-request LLM 阈值，实际 {before}"
        );

        // 4. 先验证常规压缩达标后门控保持关闭，保留精确上下文。
        let mut work = messages.clone();
        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut work);
        let llm_threshold =
            crate::ai::driver::turn_runtime::pre_request_llm_summary_threshold(
                &app.current_model,
                app.config.history_max_chars,
            );
        let gate_open = crate::ai::driver::turn_runtime::should_try_llm_summary(
            &app.session_id,
            report.after_chars,
            llm_threshold,
        );
        assert!(
            !gate_open,
            "常规压缩已达标后不应调用有损 LLM 摘要: after_chars={} threshold={}",
            report.after_chars,
            llm_threshold
        );

        // 5. 大量小段旧 user 消息不能被常规压缩静默删除；压缩后仍超阈值时，
        //    LLM 摘要作为兜底应真正执行。每段低于 user 原文外溢阈值，确保覆盖该路径。
        let mut summary_work = vec![msg("system", "你是测试助手，请遵循项目规则。")];
        for turn in 0..220 {
            summary_work.push(msg(
                "user",
                format!("第 {turn} 轮问题：{}", "u".repeat(900)),
            ));
            summary_work.push(msg("assistant", format!("第 {turn} 轮简短答复")));
        }
        summary_work.push(msg("user", "最后：请总结以上所有结果"));
        let dense_report =
            apply_pre_request_context_budget(&app, &app.current_model, &mut summary_work);
        assert!(
            dense_report.after_chars > llm_threshold,
            "测试历史经常规压缩后应仍超阈值: after_chars={} threshold={}",
            dense_report.after_chars,
            llm_threshold
        );
        assert!(
            crate::ai::driver::turn_runtime::should_try_llm_summary(
                &app.session_id,
                dense_report.after_chars,
                llm_threshold,
            ),
            "压缩后仍超阈值时 LLM 摘要门控应打开"
        );

        let (after_msgs, llm_before, llm_after, was_effective) =
            crate::ai::history::mid_turn_llm_summarize(
                &app,
                summary_work,
                2,
                4_000,
                app.config.history_max_chars,
            )
            .await;

        server.join().unwrap();
        assert!(
            served.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "mock LLM 服务器没有收到任何摘要请求"
        );
        assert!(was_effective, "LLM 摘要执行但被认为无效");
        assert!(
            llm_after < llm_before,
            "LLM 摘要后体积未下降: {llm_before} -> {llm_after}"
        );
        assert!(
            after_msgs.iter().any(|m| {
                m.role == "internal_note"
                    && m.content.to_string().contains("mid-turn-summary")
            }),
            "结果中缺少 [mid-turn-summary] 摘要 note"
        );
    }
}
