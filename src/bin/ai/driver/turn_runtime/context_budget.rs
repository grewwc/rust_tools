use rustc_hash::FxHashSet;
use serde_json::Value;

use crate::ai::{history::Message, types::App};

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
    let last_user_index = messages.iter().rposition(|message| message.role == "user");
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
    matches!(tool_name, "read_file" | "web_search" | "web_fetch")
}

fn collect_protected_messages(messages: &[Message]) -> Vec<ProtectedMessage> {
    let last_user_index = messages.iter().rposition(|message| message.role == "user");
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
                agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
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
    fn context_budget_runs_lossless_prepass_without_budget_pressure() {
        let history_file = std::env::temp_dir().join(format!(
            "context-budget-lossless-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);
        let system = msg("system", "system prompt must stay exact");
        let current_user = msg("user", "latest user input must stay exact");
        let duplicate_note = msg(crate::ai::history::ROLE_INTERNAL_NOTE, "same reminder");
        let tool_call = assistant_tool_call("call-1", "list_directory");
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
            messages.push(assistant_tool_call(&id, "list_directory"));
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
}
