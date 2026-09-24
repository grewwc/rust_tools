use rustc_hash::FxHashSet;
use serde_json::Value;

use crate::ai::{
    history::{Message, last_real_user_index},
    types::App,
};

#[cfg(test)]
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
                "lossy compression rolled back because it did not improve beyond deterministic prepasses"
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
    pub(super) tool_spill_saved_chars: usize,
    pub(super) memory_projection_removed_messages: usize,
    pub(super) memory_projection_selected_messages: usize,
    pub(super) memory_projection_saved_chars: usize,
    pub(super) memory_projection_index_chars: usize,
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

#[cfg(test)]
fn apply_pre_request_context_budget(
    app: &App,
    model: &str,
    messages: &mut Vec<Message>,
) -> ContextBudgetReport {
    let target_chars = mid_turn_compress_soft_threshold(model, app.config.history_max_chars);
    apply_context_budget_target(
        app,
        messages,
        target_chars,
        true,
        super::max_tool_result_inline_chars(model),
    )
}

/// Refresh fixed plan overhead before measuring the normalized request.
pub(super) fn refresh_active_plan(app: &App, messages: &mut Vec<Message>) {
    ActivePlanProjection::take(messages);
    ActivePlanProjection::new(active_plan_text(app)).restore(messages);
}

pub(super) fn apply_measured_context_budget(
    app: &App,
    budget: crate::ai::request::CurrentRequestBudget,
    messages: &mut Vec<Message>,
) -> ContextBudgetReport {
    let chars = crate::ai::history::messages_total_chars_pub(messages);
    // Below token pressure, only deterministic cleanup/recall runs. A character
    // threshold (including one changed by model selection) cannot enable loss.
    let target = if budget.exceeds_soft_target() {
        budget.compression_target_chars(chars)
    } else {
        chars.saturating_add(
            budget.limits.soft_target_tokens.saturating_sub(budget.prompt_tokens).saturating_mul(2),
        )
    };
    apply_context_budget_target(
        app,
        messages,
        target,
        budget.exceeds_soft_target(),
        budget.inline_cap_chars,
    )
}

fn apply_context_budget_target(
    app: &App,
    messages: &mut Vec<Message>,
    target_chars: usize,
    allow_lossy: bool,
    tool_inline_cap_chars: usize,
) -> ContextBudgetReport {
    // Persisted plan state is a fixed-cost request projection, not compressible history.
    // Remove the previous pair before reserving the current one so refreshes cannot
    // accumulate charges or let recall/compression spend the plan's headroom.
    ActivePlanProjection::take(messages);
    let active_context = active_plan_text(app);
    let active_plan = ActivePlanProjection::new(if last_real_user_index(messages).is_some() {
        active_context.clone()
    } else {
        String::new()
    });
    let reserved_chars = active_plan.chars();
    let mut report = apply_history_context_budget(
        app,
        messages,
        active_plan.remaining_target(target_chars),
        &active_context,
        allow_lossy,
        tool_inline_cap_chars,
    );
    active_plan.restore(messages);
    report.before_chars = report.before_chars.saturating_add(reserved_chars);
    report.after_chars = crate::ai::history::messages_total_chars_pub(messages);
    report.target_chars = target_chars;
    report
}

fn apply_history_context_budget(
    app: &App,
    messages: &mut Vec<Message>,
    target_chars: usize,
    active_context: &str,
    allow_lossy: bool,
    tool_inline_cap_chars: usize,
) -> ContextBudgetReport {
    let scan = quick_scan(messages);
    let mut report = ContextBudgetReport {
        before_chars: scan.total_chars,
        after_chars: scan.total_chars,
        target_chars,
        ..ContextBudgetReport::default()
    };

    // Size-gated tool-result offload is unconditional (independent of the
    // soft target / allow_lossy): single results beyond the model's inline
    // cap no longer occupy every request just because total context is under
    // pressure. The protected tail window (most recent tool groups) stays raw.
    let overflow_dir = {
        let store = crate::ai::history::SessionStore::new(app.config.history_file.as_path());
        store.session_assets_dir(&app.session_id)
    };
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd().ok();
    let tool_spilled = crate::ai::history::compress::cap_oversized_tool_results_for_context(
        messages,
        tool_inline_cap_chars,
        crate::ai::history::compress::KEEP_RECENT_TOOL_GROUPS,
        Some(overflow_dir.as_path()),
        cwd.as_deref(),
    );
    if tool_spilled > 0 {
        let after_spill_chars = crate::ai::history::messages_total_chars_pub(messages);
        report.changed = true;
        report.after_chars = after_spill_chars;
        report.tool_spill_saved_chars = scan.total_chars.saturating_sub(after_spill_chars);
    }

    if scan.total_chars <= target_chars
        && !scan.has_lossless_candidate
        && !super::context_memory::has_dense_recoverable_memory(messages)
        && !super::context_memory::has_archive_recall_sources(messages)
    {
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

    let memory_projection = super::context_memory::apply_query_aware_memory_projection_with_context(
        messages,
        target_chars,
        &overflow_dir,
        active_context,
    );
    report.memory_projection_removed_messages = memory_projection.removed_messages;
    report.memory_projection_selected_messages = memory_projection.selected_messages;
    report.memory_projection_saved_chars = memory_projection.saved_chars;
    report.memory_projection_index_chars = memory_projection.index_chars;
    let after_prepass_chars = if memory_projection.changed {
        report.changed = true;
        report.after_chars = memory_projection.after_chars;
        memory_projection.after_chars
    } else {
        after_lossless_chars
    };

    if !allow_lossy || after_prepass_chars <= target_chars {
        if report.changed {
            fill_segment_summary(&mut report, messages);
        }
        return report;
    }

    fill_segment_summary(&mut report, messages);
    let protected = collect_protected_messages(messages);
    let original = messages.clone();
    let drained = std::mem::take(messages);
    let (compressed, _, after_chars) = crate::ai::history::mid_turn_compress(
        drained,
        target_chars,
        Some(overflow_dir.as_path()),
        crate::ai::driver::runtime_ctx::effective_cwd()
            .ok()
            .as_deref(),
    );
    *messages = compressed;
    // mid_turn_compress inserts the compaction state note after the last user message (which is
    // the current round's active region in tool-loop scenarios). But the request boundary requires
    // the current user message to be the last item in the sent sequence, so here we move that note
    // before the current user message: this keeps the user-last contract while still letting the
    // model see "this round uses a compressed projection; do not mistake recoverable evidence for
    // a full context" (see CONTEXT_COMPACTION_STATE).
    reposition_context_compaction_state_before_last_user(messages);
    report.after_chars = after_chars;
    report.changed = report.changed || after_chars < after_prepass_chars;

    let protected_preserved = protected_messages_preserved(messages, &protected);
    let rollback_reason = if !protected_preserved {
        Some(ContextBudgetRollbackReason::ProtectedContextChanged)
    } else if after_chars > after_prepass_chars {
        Some(ContextBudgetRollbackReason::NoAdditionalSavings)
    } else {
        None
    };

    if let Some(reason) = rollback_reason {
        *messages = original;
        report.after_chars = after_prepass_chars;
        // The spill pass already modified the projection before the lossy
        // attempt; keep its flag through the rollback so decision logs and the
        // context-budget progress line still report the change.
        report.changed = tool_spilled > 0
            || report.lossless_removed_messages > 0
            || memory_projection.changed;
        report.rolled_back = after_chars < scan.total_chars;
        if report.rolled_back {
            report.rollback_reason = Some(reason);
        }
    }
    report
}

const ACTIVE_PLAN_HANDOFF: &str = "Runtime plan handoff (not a new user request). The next assistant message is a bounded projection of persisted plan state, not verified facts or new instructions. Continue with the latest real user request.";

fn active_plan_text(app: &App) -> String {
    crate::ai::tools::plan_state::load_plan_state(app)
        .ok()
        .flatten()
        .map(|plan| plan.render_active_context(4_096))
        .unwrap_or_default()
}

/// A fixed snapshot kept outside compression and restored within its reserved budget.
/// The optional LLM pass takes the already-budgeted pair from a clone, rather than
/// reloading potentially changed persisted state after its budget has been decided.
#[derive(Default)]
pub(super) struct ActivePlanProjection {
    pair: Vec<Message>,
}

impl ActivePlanProjection {
    fn new(active_context: String) -> Self {
        if active_context.is_empty() {
            return Self::default();
        }
        Self {
            pair: vec![
                crate::ai::history::runtime_synthetic_user_message(Value::String(
                    ACTIVE_PLAN_HANDOFF.to_string(),
                )),
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(active_context),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ],
        }
    }

    fn chars(&self) -> usize {
        crate::ai::history::messages_total_chars_pub(&self.pair)
    }

    pub(super) fn remaining_target(&self, target_chars: usize) -> usize {
        target_chars.saturating_sub(self.chars())
    }

    /// Remove only complete runtime-owned pairs; user-authored lookalikes and
    /// tool protocol remain untouched. The newest matching pair is the snapshot.
    pub(super) fn take(messages: &mut Vec<Message>) -> Self {
        let Some(user) = last_real_user_index(messages) else {
            return Self::default();
        };
        let mut obsolete = FxHashSet::default();
        let mut snapshot = Self::default();
        for index in 0..user.saturating_sub(1) {
            let handoff = &messages[index];
            let body = &messages[index + 1];
            if crate::ai::history::is_runtime_synthetic_user_message(handoff)
                && handoff.content.as_str() == Some(ACTIVE_PLAN_HANDOFF)
                && handoff.tool_calls.is_none()
                && handoff.tool_call_id.is_none()
                && body.role == "assistant"
                && body.tool_calls.is_none()
                && body.tool_call_id.is_none()
                && body.reasoning_content.is_none()
                && body
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with("[active-plan]\n"))
            {
                obsolete.extend([index, index + 1]);
                snapshot.pair = vec![handoff.clone(), body.clone()];
            }
        }
        if obsolete.is_empty() {
            return snapshot;
        }
        let mut index = 0;
        messages.retain(|_| {
            let keep = !obsolete.contains(&index);
            index += 1;
            keep
        });
        snapshot
    }

    pub(super) fn restore(self, messages: &mut Vec<Message>) {
        if let Some(user) = last_real_user_index(messages) {
            messages.splice(user..user, self.pair);
        }
    }
}

/// mid_turn_compress inserts the `CONTEXT_COMPACTION_STATE` note after the last user
/// message. The request boundary requires the current user message to be the last item in the
/// sent sequence, so here we move the note **before** the last user message: this keeps the
/// user-last contract while preserving the note's visibility to the model.
/// The note content is moved verbatim, not rewritten (the single source stays in the compress module).
fn reposition_context_compaction_state_before_last_user(messages: &mut Vec<Message>) {
    let Some(note_index) = messages
        .iter()
        .position(crate::ai::history::compress::is_context_compaction_state)
    else {
        return;
    };
    let Some(last_user_index) = last_real_user_index(messages) else {
        // No user message: the request-boundary contract does not apply, leave as is.
        return;
    };
    // Already before the last user message, no move needed.
    if note_index < last_user_index {
        return;
    }
    // Move the note from `note_index` (after the last user message) to `last_user_index`
    // with a single slice rotation (intermediate elements shift back by one). This is
    // equivalent to `remove(note_index)` + `insert(last_user_index, note)` but performs
    // one O(span) in-place shift instead of two O(n) shifts with reallocation.
    messages[last_user_index..=note_index].rotate_right(1);
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
        tool_spill_saved_chars: 0,
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
        memory_projection_removed_messages: 0,
        memory_projection_selected_messages: 0,
        memory_projection_saved_chars: 0,
        memory_projection_index_chars: 0,
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
    // Synthetic user messages do not constitute a turn boundary: real user messages must keep
    // Critical/Never protection, otherwise they would be downgraded to RecentUser and become offloadable.
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
    // Protected messages (all `system` messages plus the last real user message) keep
    // their relative positions across mid-turn compression (it truncates/folds in place
    // and never reorders), so compare by position with borrowed references. This avoids
    // a second `collect_protected_messages` deep clone of the system prompt.
    let last_user_index = last_real_user_index(messages);
    let mut expected = protected.iter();
    for (index, message) in messages.iter().enumerate() {
        let is_protected_position =
            message.role == "system" || (message.role == "user" && Some(index) == last_user_index);
        if !is_protected_position {
            continue;
        }
        let Some(protected_message) = expected.next() else {
            return false;
        };
        if protected_message.role != message.role
            || protected_message.content != message.content
            || protected_message.tool_calls != message.tool_calls
            || protected_message.tool_call_id != message.tool_call_id
            || protected_message.reasoning_content != message.reasoning_content
        {
            return false;
        }
    }
    expected.next().is_none()
}

fn message_chars(message: &Message) -> usize {
    // Always use the authoritative billing metric from the history layer (content + tool_calls +
    // reasoning_content, images at nominal cost) so that messages with large tool_calls/reasoning
    // are not underestimated by this budget gate.
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
            scoped_preflight_required: Vec::new(),
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
            forced_skills: Vec::new(),
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
            tool_middlewares: Vec::new(),
            llm_middlewares: Vec::new(),
            hooks: Default::default(),
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
    fn context_budget_measured_tokens_gate_lossy_work_not_character_size_or_model_name() {
        let history_file = std::env::temp_dir().join(format!("context-budget-measured-{}.sqlite", uuid::Uuid::new_v4()));
        let mut app = test_app(history_file);
        let canonical = vec![
            msg("system", "exact system"),
            msg("assistant", "old narration ".repeat(4_000)),
            msg("user", "exact current user"),
        ];
        let budget: crate::ai::request::CurrentRequestBudget = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 8_000, "source": "normalized_estimate",
            "limits": { "physical_context_tokens": 12_000, "output_reserve_tokens": 1_000,
                "safety_margin_tokens": 1_000, "input_allowance_tokens": 10_000, "soft_target_tokens": 8_000 }
        })).unwrap();
        let mut messages = canonical.clone();
        let report = apply_measured_context_budget(&app, budget, &mut messages);
        assert!(!report.changed);
        assert_eq!(messages, canonical);
        app.current_model = "another selection with the same measured headroom".to_string();
        apply_measured_context_budget(&app, budget, &mut messages);
        assert_eq!(messages, canonical);

        let pressured = crate::ai::request::CurrentRequestBudget { prompt_tokens: 40_000, ..budget };
        let report = apply_measured_context_budget(&app, pressured, &mut messages);
        // The token gate opened: the target shrank below the current size even
        // though neither character size nor model name changed. But the
        // mid-turn lossy pipeline (dedupe, structured tool trimming, group
        // folding, reasoning cleanup) only acts on tool/agent traffic — a
        // plain narration can only be shrunk by a cross-turn LLM summary, and
        // the test env injects no model. Correct contract: the gate opens, the
        // attempt finds nothing compressible, and the projection rolls back
        // byte-for-byte (no change flags, content intact).
        assert!(report.target_chars < report.before_chars);
        assert_eq!(report.after_chars, report.before_chars);
        assert!(!report.changed);
        assert_eq!(messages.first(), canonical.first());
        assert_eq!(messages.last(), canonical.last());
        assert_eq!(canonical[1].content.as_str().unwrap().len(), "old narration ".len() * 4_000);
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

    fn record_long_plan(app: &App) -> String {
        // Summary and action fields have separate excerpt limits; multiple steps
        // are needed to exercise the full 4096-character projection budget.
        let steps = (1..=12)
            .map(|step| serde_json::json!({"step": step, "action": "验证行为 ".repeat(100)}))
            .collect::<Vec<_>>();
        crate::ai::tools::plan_state::record_plan(
            app,
            &format!("alpha_recovery_signal {}", "长期计划 ".repeat(1_000)),
            &steps,
        )
        .unwrap();
        active_plan_text(app)
    }

    #[test]
    fn context_budget_plan_triggers_compression_before_crossing_soft_target() {
        let root = std::env::temp_dir().join(format!("plan-soft-{}", uuid::Uuid::new_v4()));
        let app = test_app(root.join("test.sqlite"));
        let plan_text = record_long_plan(&app);
        let plan_chars = ActivePlanProjection::new(plan_text.clone()).chars();
        assert!(plan_chars > 3_000);
        let target =
            mid_turn_compress_soft_threshold(&app.current_model, app.config.history_max_chars);
        let mut messages = vec![msg("system", "rules"), msg("user", "continue")];
        let fixed = crate::ai::history::messages_total_chars_pub(&messages);
        messages.insert(1, msg("assistant", "x".repeat(target - fixed - 1_000)));
        let canonical = messages.clone();
        let before = crate::ai::history::messages_total_chars_pub(&messages);
        assert!(before < target && before + plan_chars > target);

        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        assert!(report.changed);
        assert_eq!(report.before_chars, before + plan_chars);
        assert_eq!(report.target_chars, target);
        assert_eq!(
            report.after_chars,
            crate::ai::history::messages_total_chars_pub(&messages)
        );
        assert!(report.after_chars <= target);
        assert_eq!(messages.first(), canonical.first());
        assert_eq!(messages.last(), canonical.last());
        let plan = ActivePlanProjection::take(&mut messages);
        assert_eq!(plan.pair[1].content.as_str(), Some(plan_text.as_str()));
        plan.restore(&mut messages);
        let again = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        // Compression creates an archive that recall may discover on the next pass;
        // only the plan pair must be stable, not the entire memory projection.
        assert_eq!(again.before_chars, report.after_chars);
        assert_eq!(
            again.after_chars,
            crate::ai::history::messages_total_chars_pub(&messages)
        );
        assert!(again.after_chars <= target);
        let refreshed = ActivePlanProjection::take(&mut messages);
        assert_eq!(refreshed.chars(), plan_chars);
        assert_eq!(refreshed.pair[1].content.as_str(), Some(plan_text.as_str()));
        assert!(
            !messages
                .iter()
                .any(|m| message_text(&m.content).starts_with("[active-plan]\n"))
        );
        assert_eq!(canonical.len(), 3);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn context_budget_plan_counts_fast_path_and_irreducible_context() {
        let root = std::env::temp_dir().join(format!("plan-fixed-{}", uuid::Uuid::new_v4()));
        let app = test_app(root.join("test.sqlite"));
        let plan_chars = ActivePlanProjection::new(record_long_plan(&app)).chars();
        let target =
            mid_turn_compress_soft_threshold(&app.current_model, app.config.history_max_chars);
        for excess in [0, 1_000] {
            let user = msg("user", "continue");
            let system = msg(
                "system",
                "s".repeat(target - plan_chars - message_chars(&user) + excess),
            );
            let mut messages = vec![system.clone(), user.clone()];
            let report = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
            assert_eq!(report.before_chars, target + excess);
            assert_eq!(report.after_chars, target + excess);
            assert_eq!(
                report.after_chars,
                crate::ai::history::messages_total_chars_pub(&messages)
            );
            assert_eq!(messages.first(), Some(&system));
            assert_eq!(messages.last(), Some(&user));
            assert_eq!(
                ActivePlanProjection::take(&mut messages).chars(),
                plan_chars
            );
            assert_eq!(messages, vec![system, user]);
        }
        // Without a real user there is no insertion point and no phantom plan charge.
        let mut messages = vec![msg("system", "rules")];
        let before = messages.clone();
        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        assert_eq!(messages, before);
        assert_eq!(report.before_chars, report.after_chars);
        assert_eq!(
            report.after_chars,
            crate::ai::history::messages_total_chars_pub(&messages)
        );
        let _ = std::fs::remove_dir_all(root);
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

        // The compaction state note is only injected when actual compression took effect.
        assert!(report.changed);
        // The current user message is still the last item in the sent sequence (request-boundary contract).
        assert_eq!(messages.last().unwrap(), &current_user);
        // The compaction state note stays visible to the model and is moved before the last user message.
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

    #[tokio::test]
    async fn context_budget_plan_snapshot_survives_llm_backstop_without_reloading() {
        let root = std::env::temp_dir().join(format!("plan-summary-{}", uuid::Uuid::new_v4()));
        let app = test_app(root.join("test.sqlite"));
        let plan_text = record_long_plan(&app);
        // Only the current turn exists, so the summary path cannot call an LLM;
        // its mechanical hard-budget backstop still has oversized output to shrink.
        let mut messages = vec![
            msg("system", "rules"),
            msg("user", "continue"),
            assistant_tool_call("current", "execute_command"),
            tool_result("current", "output ".repeat(12_000)),
        ];
        apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        let original = messages.clone();
        let before = crate::ai::history::messages_total_chars_pub(&messages);
        let mut work = messages.clone();
        let plan = ActivePlanProjection::take(&mut work);
        let reserved = plan.chars();
        assert!(reserved > 3_000);
        assert_eq!(plan.remaining_target(1), 0);
        crate::ai::tools::plan_state::update_plan_step(
            &app,
            1,
            crate::ai::tools::plan_state::StepStatus::Failed,
            Some("New state".into()),
        )
        .unwrap();
        let target = 36_000;
        let (mut summarized, history_before, history_after, effective, inserted) =
            crate::ai::history::mid_turn_llm_summarize(
                &app,
                work,
                super::super::MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
                super::super::MID_TURN_LLM_SUMMARY_MAX_CHARS,
                plan.remaining_target(target),
                None,
            )
            .await;
        plan.restore(&mut summarized);
        let after = crate::ai::history::messages_total_chars_pub(&summarized);
        assert_eq!(history_before + reserved, before);
        assert_eq!(history_after + reserved, after);
        assert!(after < before && after <= target);
        assert!(effective);
        assert!(!inserted);
        assert_eq!(messages, original);
        let restored = ActivePlanProjection::take(&mut summarized);
        assert_eq!(restored.pair[1].content.as_str(), Some(plan_text.as_str()));
        restored.restore(&mut summarized);
        let mut report = super::super::CompressionReport::default();
        report.record_llm_summary_attempt("pre-request", before, after, effective, inserted);
        assert!(
            report
                .render()
                .unwrap()
                .contains(&format!("{before} → {after} chars"))
        );
        let _ = std::fs::remove_dir_all(root);
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
    fn context_budget_preserves_hierarchical_memory_index_through_later_compression() {
        let history_file = std::env::temp_dir().join(format!(
            "context-budget-memory-index-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);
        let current_user = msg("user", "continue checkpoint topic-3");
        let mut messages = vec![msg("system", "system prompt must stay exact")];
        for index in 0..16 {
            messages.push(msg(
                crate::ai::history::ROLE_INTERNAL_NOTE,
                format!(
                    "[context_checkpoint path=/tmp/test.assets/context-checkpoints/{index}.md] \
                     topic-{index} {}",
                    "x".repeat(1_200)
                ),
            ));
        }
        messages.push(msg("assistant", "old narration ".repeat(4_000)));
        messages.push(current_user.clone());

        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);

        assert!(report.memory_projection_removed_messages > 0);
        assert_eq!(messages.last(), Some(&current_user));
        assert!(messages.iter().any(|message| {
            crate::ai::history::value_to_string(&message.content)
                .starts_with(crate::ai::history::compress::QUERY_MEMORY_INDEX_PREFIX)
        }));
    }

    #[test]
    fn context_budget_recalls_sparse_archive_using_current_plan() {
        let root =
            std::env::temp_dir().join(format!("context-budget-recall-{}", uuid::Uuid::new_v4()));
        let mut app = test_app(root.join("history.json"));
        app.config.history_max_chars = 100_000;
        let store = crate::ai::history::SessionStore::new(&app.config.history_file);
        let assets = store.session_assets_dir(&app.session_id);
        app.session_history_file = store.sessions_root().join("test.sqlite");
        let source = assets.join("context-checkpoints/recall.md");
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, "alpha_recovery_signal: source evidence\n").unwrap();
        crate::ai::tools::plan_state::record_plan(
            &app,
            "Review alpha_recovery_signal",
            &[serde_json::json!({"step": 1, "action": "Review alpha_recovery_signal"})],
        )
        .unwrap();
        let mut messages = vec![
            msg("system", "rules"),
            msg(
                crate::ai::history::ROLE_INTERNAL_NOTE,
                format!(
                    "[context_checkpoint path={}] archived source",
                    source.display()
                ),
            ),
            msg("user", "continue"),
        ];
        let canonical = messages.clone();
        assert!(!super::super::context_memory::has_dense_recoverable_memory(
            &messages
        ));
        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        assert!(report.changed);
        assert_eq!(report.memory_projection_removed_messages, 0);
        assert_eq!(
            report.after_chars,
            crate::ai::history::messages_total_chars_pub(&messages)
        );
        let recall = messages
            .iter()
            .position(|m| {
                m.role == "assistant"
                    && message_text(&m.content).starts_with("[query-memory-recall-v1]")
            })
            .unwrap();
        assert!(
            message_text(&messages[recall].content)
                .contains("alpha_recovery_signal: source evidence")
        );
        assert!(crate::ai::history::is_runtime_synthetic_user_message(
            &messages[recall - 1]
        ));
        assert_eq!(messages.first(), canonical.first());
        assert_eq!(messages.last(), canonical.last());
        assert_eq!(canonical.len(), 3);
        let once = messages.clone();
        apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        assert_eq!(messages, once);

        // A plan can consume all remaining headroom; recall must not spend it again.
        let plan_chars = ActivePlanProjection::new(record_long_plan(&app)).chars();
        let target =
            mid_turn_compress_soft_threshold(&app.current_model, app.config.history_max_chars);
        let mut tight = canonical.clone();
        let non_system = crate::ai::history::messages_total_chars_pub(&tight[1..]);
        tight[0] = msg("system", "s".repeat(target - non_system - plan_chars));
        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut tight);
        assert_eq!(report.before_chars, target);
        assert_eq!(report.after_chars, target);
        assert_eq!(
            report.after_chars,
            crate::ai::history::messages_total_chars_pub(&tight)
        );
        assert!(
            !tight
                .iter()
                .any(|m| message_text(&m.content).starts_with("[query-memory-recall-v1]"))
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn context_budget_plan_projection_refreshes_status_without_promoting_content() {
        let root =
            std::env::temp_dir().join(format!("context-budget-plan-{}", uuid::Uuid::new_v4()));
        let app = test_app(root.join("test.sqlite"));
        let mut messages = vec![msg("system", "rules"), msg("user", "continue")];
        let canonical = messages.clone();
        apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        assert_eq!(messages, canonical);
        crate::ai::tools::plan_state::record_plan(
            &app,
            "Current work",
            &[serde_json::json!({"step": 1, "action": "Verify behavior"})],
        )
        .unwrap();
        apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        let once = messages.clone();
        apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        assert_eq!(messages, once);
        assert!(crate::ai::history::is_runtime_synthetic_user_message(
            &messages[1]
        ));
        assert_eq!(messages[2].role, "assistant");
        assert!(message_text(&messages[2].content).contains("1=pending"));
        assert!(message_text(&messages[2].content).contains("assistant-derived"));
        crate::ai::tools::plan_state::update_plan_step(
            &app,
            1,
            crate::ai::tools::plan_state::StepStatus::Failed,
            Some("Verification failed".into()),
        )
        .unwrap();
        apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        assert_eq!(messages.len(), 4);
        assert!(message_text(&messages[2].content).contains("1=failed"));
        assert!(message_text(&messages[2].content).contains("Verification failed"));
        assert_eq!(messages.first(), canonical.first());
        assert_eq!(messages.last(), canonical.last());
        std::fs::write(
            crate::ai::tools::plan_state::plan_state_path(&app),
            "invalid JSON",
        )
        .unwrap();
        apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
        assert_eq!(messages, canonical);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn context_budget_plan_projection_replaces_small_budget_pairs_without_orphan_handoffs() {
        let root = std::env::temp_dir().join(format!("plan-small-{}", uuid::Uuid::new_v4()));
        let app = test_app(root.join("test.sqlite"));
        crate::ai::tools::plan_state::record_plan(
            &app,
            "計画🙂",
            &[serde_json::json!({"step": 1, "action": "界🙂".repeat(200)})],
        )
        .unwrap();
        let old_plan = crate::ai::tools::plan_state::load_plan_state(&app)
            .unwrap()
            .unwrap();
        crate::ai::tools::plan_state::update_plan_step(
            &app,
            1,
            crate::ai::tools::plan_state::StepStatus::Failed,
            Some("Fresh failure".into()),
        )
        .unwrap();
        let fresh = active_plan_text(&app);
        // A real user's lookalike pair and tool protocol must survive replacement.
        let canonical = vec![
            msg("system", "rules"),
            msg("user", ACTIVE_PLAN_HANDOFF),
            msg("assistant", "[active-plan]\nuser-authored lookalike"),
            assistant_tool_call("kept", "read_file"),
            tool_result("kept", "exact source"),
            msg("user", "continue"),
        ];
        for budget in (0..=256).chain([512, 4_096]) {
            let body = old_plan.render_active_context(budget);
            let projection = ActivePlanProjection::new(body.clone());
            let reserved = projection.chars();
            let mut messages = canonical.clone();
            projection.restore(&mut messages);
            assert_eq!(
                messages.len(),
                canonical.len() + if body.is_empty() { 0 } else { 2 },
                "budget={budget}"
            );
            let taken = ActivePlanProjection::take(&mut messages);
            assert_eq!(messages, canonical, "budget={budget}");
            assert_eq!(taken.chars(), reserved, "budget={budget}");
            if body.is_empty() {
                assert!(taken.pair.is_empty());
                assert_eq!(reserved, 0);
            } else {
                assert_eq!(taken.pair.len(), 2);
                assert!(crate::ai::history::is_runtime_synthetic_user_message(
                    &taken.pair[0]
                ));
                assert_eq!(taken.pair[1].role, "assistant");
                assert_eq!(taken.pair[1].content.as_str(), Some(body.as_str()));
                assert!(body.starts_with("[active-plan]\n"));
                assert!(body.contains("assistant-derived, not independently verified."));
            }
            taken.restore(&mut messages);
            apply_pre_request_context_budget(&app, &app.current_model, &mut messages);
            let replacement = ActivePlanProjection::take(&mut messages);
            assert_eq!(replacement.pair.len(), 2, "budget={budget}");
            assert_eq!(replacement.pair[1].content.as_str(), Some(fresh.as_str()));
            assert!(fresh.contains("Fresh failure"));
            assert_eq!(messages, canonical, "budget={budget}");
        }
        let _ = std::fs::remove_dir_all(root);
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
        // A large read_file result is only offloaded once it falls outside the "last 6 tool results"
        // protection window; otherwise the near-end window keeps it verbatim (to prevent the model
        // from re-searching content that was just retrieved).
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

    /// Regression covering both paths: tool-dense history that already meets the budget after
    /// regular compression never triggers the lossy LLM summary; only conversation-dense history
    /// that still exceeds the threshold after compression calls the LLM and effectively shrinks the context.
    #[tokio::test]
    async fn llm_summary_runs_only_when_post_compression_context_still_exceeds_threshold() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // 1. Local mock LLM server: read the full Content-Length then return an OpenAI-format response
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
                        // Read the request headers
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
                        // Read the full body (per Content-Length)
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
                        // The header may already contain part of the body (after \r\n\r\n)
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

        // 2. app: endpoint points at the mock; history_max_chars is pinned here so
        // this test stays deterministic regardless of future production-default changes
        let history_file =
            std::env::temp_dir().join(format!("llm_summary_repro_{}.jsonl", uuid::Uuid::new_v4()));
        let mut app = test_app(history_file);
        app.config.endpoint = format!("http://{addr}");
        app.config.history_max_chars = 90_000;
        app.session_id = format!("llm_summary_repro_{}", uuid::Uuid::new_v4());

        // 3. Tool-dense, very long session: regular compression can safely stay under the threshold, so no lossy summary should run.
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

        // 4. First verify the gate stays closed once regular compression meets the threshold, preserving exact context.
        let mut work = messages.clone();
        let report = apply_pre_request_context_budget(&app, &app.current_model, &mut work);
        let llm_threshold = crate::ai::driver::turn_runtime::pre_request_llm_summary_threshold(
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
            report.after_chars, llm_threshold
        );

        // 5. Many small old user messages cannot be silently dropped by regular compression; when
        //    the context still exceeds the threshold after compression, the LLM summary fallback
        //    must actually run. Each segment stays below the user-original offload threshold to cover this path.
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

        let (after_msgs, llm_before, llm_after, was_effective, llm_summary_inserted) =
            crate::ai::history::mid_turn_llm_summarize(
                &app,
                summary_work,
                2,
                4_000,
                app.config.history_max_chars,
                crate::ai::driver::runtime_ctx::effective_cwd()
                    .ok()
                    .as_deref(),
            )
            .await;

        server.join().unwrap();
        assert!(
            served.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "mock LLM 服务器没有收到任何摘要请求"
        );
        assert!(was_effective, "LLM 摘要执行但被认为无效");
        assert!(
            llm_summary_inserted,
            "the LLM summary should report an inserted incremental summary"
        );
        assert!(
            llm_after < llm_before,
            "LLM 摘要后体积未下降: {llm_before} -> {llm_after}"
        );
        let summary = after_msgs
            .iter()
            .find(|message| crate::ai::history::compress::is_incremental_summary(message))
            .expect("the LLM summary must produce a source-bound increment");
        let text = crate::ai::history::value_to_string(&summary.content);
        assert!(text.contains("MOCK_SUMMARY"), "{text}");
        assert!(text.contains("summary-sources"), "{text}");
    }

    #[test]
    fn spill_keeps_changed_flag_under_lossy_pressure() {
        // The unconditional size-gated spill runs before any lossy work and must
        // keep `changed = true` even when the lossy path expands afterwards:
        // decision logs and the context-budget progress line rely on it to
        // reflect that the projection was actually modified. Regression for the
        // rollback branch overwriting the spill flag with the lossless/memory
        // flags.
        let history_file = std::env::temp_dir().join(format!(
            "cb-spill-changed-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);
        let mut messages = vec![
            msg("system", "system prompt"),
            // An oversized complete tool group outside the protected recent
            // window: the unconditional spill must archive it regardless of
            // budget, while the four recent groups stay raw.
            assistant_tool_call("g1", "read_file"),
            tool_result("g1", &"y".repeat(70_000)),
            assistant_tool_call("g2", "read_file"),
            tool_result("g2", "r2"),
            assistant_tool_call("g3", "read_file"),
            tool_result("g3", "r3"),
            assistant_tool_call("g4", "read_file"),
            tool_result("g4", "r4"),
            assistant_tool_call("g5", "read_file"),
            tool_result("g5", "r5"),
            msg("user", "current question"),
        ];
        let report = apply_context_budget_target(&app, &mut messages, 100, true, 32_000);
        assert!(
            report.tool_spill_saved_chars > 0,
            "oversized result must be spilled before lossy work"
        );
        assert!(
            report.changed,
            "spill must keep changed=true through the lossy path"
        );
        let text = message_text(&messages[2].content);
        assert!(
            text.contains("PRESERVED_TOOL_OVERFLOW_STUB"),
            "the spilled result must be replaced by a recallable stub: {text}"
        );
    }

    #[test]
    fn spill_cap_chars_follow_injected_model_not_app_current_model() {
        // The unconditional spill line must come from the model the caller
        // measured, not from `app.current_model` (stale during model-fallback
        // retries). A 40K result stays inline under a 1M-window model (64K
        // line) but spills under an unknown/empty model (32K line).
        let history_file = std::env::temp_dir().join(format!(
            "spill-cap-model-{}.json",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);
        let messages = vec![
            msg("system", "system prompt"),
            assistant_tool_call("g1", "read_file"),
            tool_result("g1", &"y".repeat(40_000)),
            assistant_tool_call("g2", "read_file"),
            tool_result("g2", "r2"),
            assistant_tool_call("g3", "read_file"),
            tool_result("g3", "r3"),
            assistant_tool_call("g4", "read_file"),
            tool_result("g4", "r4"),
            assistant_tool_call("g5", "read_file"),
            tool_result("g5", "r5"),
            msg("user", "current question"),
        ];
        // 1M-token window model (registered name, not the file name): inline
        // line is 64K, the 40K result stays raw.
        let wide = apply_pre_request_context_budget(
            &app,
            "deepseek-v4-pro",
            &mut messages.clone(),
        );
        assert_eq!(
            wide.tool_spill_saved_chars, 0,
            "40K result must stay inline under the 1M-window model line"
        );
        // Empty/unknown model (app.current_model): 32K line, same result spills.
        let narrow = apply_pre_request_context_budget(&app, "", &mut messages.clone());
        assert!(
            narrow.tool_spill_saved_chars > 0,
            "the same 40K result must spill under the 32K fallback line"
        );
    }
}
