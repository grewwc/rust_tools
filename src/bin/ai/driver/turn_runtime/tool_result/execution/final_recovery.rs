//! Final-response recovery gates: no-tool synthesis retries, reasoning-only retries,
//! task-evidence reopen markers, and injected-context echo recovery.
//!
//! Deliberately no wording classification lives here: whether a final answer is
//! complete is decided by machine-verifiable turn state (tool-result receipts,
//! mutation records, pause/denial markers), never by matching phrases in prose.
//! The only text comparison left is an exact-prefix check against the runtime's
//! own injected-context sentinels (single-sourced from `request::normalize`),
//! which is provenance verification, not semantic classification.

use super::*;

pub(in crate::ai::driver::turn_runtime) const INJECTED_CONTEXT_ECHO_RETRY_MARKER: &str =
    "[injected-context-echo-retry]";
pub(in crate::ai::driver::turn_runtime) const INJECTED_CONTEXT_ECHO_RETRY_NOTE: &str = "Your previous response reproduced a runtime-injected context note verbatim instead of answering. \
Runtime notes are context for you only; they are never the user-facing answer. \
Runtime-injected notes (model-authored-note / history-summary / context-handoff headers and self_note: channel lines) are context for you only; they are never the user-facing answer. \
Do not quote, restate, or continue any of them. \
Produce the actual answer to the user's request now, using tools first if verification is still required; if you cannot verify, state that limitation in your own words.";
pub(in crate::ai::driver::turn_runtime) const INJECTED_CONTEXT_ECHO_STOP: &str = "[Model echoed a runtime internal note instead of giving a real answer; please retry or switch models]";

/// Turn-local state for final-response recovery. Prompt markers remain useful context for
/// the model, but runtime control flow must not infer retry budgets from persisted text:
/// old markers can survive into later turns. General quality gates share one retry, while
/// unsettled plan steps have their own bounded retry budget so another gate cannot
/// suppress the plan reminder or make it spin to the iteration cap.
pub(in crate::ai::driver::turn_runtime) const OUTSTANDING_WORK_MAX_REOPENS: usize = 3;

#[derive(Debug, Default)]
pub(in crate::ai::driver::turn_runtime) struct FinalGateState {
    retry_consumed: bool,
    no_tool_retry_consumed: bool,
    outstanding_work_reopens: usize,
    // Actual execution observations and model-derived dispositions belong to this
    // turn, not the compressed request projection or session-global tool ledger.
    pub(super) critical_premises: crate::ai::tools::critical_premises::PremiseState,
    pub(super) critical_evidence: Vec<crate::ai::tools::critical_premises::EvidenceRecord>,
    // Canonical positions keep increasing even when the bounded cache evicts records.
    pub(super) critical_evidence_next_index: usize,
    pub(super) critical_review_consumed: bool,
    // Earlier audit stages have separate canonical message buffers. Keep their IDs
    // reserved even after their observation bodies leave the bounded evidence cache.
    pub(super) critical_prior_tool_call_ids: rust_tools::commonw::FastSet<String>,
}

impl FinalGateState {
    pub(in crate::ai::driver::turn_runtime) const CRITICAL_PREMISES_WARNING: &'static str =
        super::critical_premises::CRITICAL_PREMISES_WARNING;

    pub(in crate::ai::driver::turn_runtime) fn has_pending_critical_premises(&self) -> bool {
        !self.critical_premises.pending().is_empty()
    }

    pub(in crate::ai::driver::turn_runtime) fn begin_audit_followup(
        &mut self,
        messages: &[Message],
    ) {
        self.critical_prior_tool_call_ids
            .extend(messages.iter().flat_map(|message| {
                message
                    .tool_calls
                    .iter()
                    .flatten()
                    .map(|call| call.id.clone())
                    .chain(message.tool_call_id.iter().cloned())
            }));
        // A followup has a new synthesis stage and may introduce a new plan.
        // Its load-bearing premises, evidence ordering and shared quality/review
        // budgets still belong to the same logical turn.
        self.no_tool_retry_consumed = false;
        self.outstanding_work_reopens = 0;
    }

    #[cfg(test)]
    pub(in crate::ai::driver::turn_runtime) fn set_critical_premises_for_test(
        &mut self,
        state: crate::ai::tools::critical_premises::PremiseState,
    ) {
        self.critical_premises = state;
    }

    pub(super) fn from_current_turn_markers(messages: &[Message]) -> Self {
        let retry_consumed = [
            INJECTED_CONTEXT_ECHO_RETRY_MARKER,
            COMPLETION_EVIDENCE_REQUIRED_MARKER,
            RECOVERABLE_PAUSE_RETRY_MARKER,
            FINAL_CITATION_RETRY_MARKER,
            OUTSTANDING_WORK_REQUIRED_MARKER,
        ]
        .iter()
        .any(|marker| current_turn_has_internal_marker(messages, marker));
        Self {
            retry_consumed,
            no_tool_retry_consumed: current_turn_has_internal_marker(
                messages,
                NO_TOOL_SYNTHESIS_RETRY_MARKER,
            ),
            outstanding_work_reopens: usize::from(current_turn_has_internal_marker(
                messages,
                OUTSTANDING_WORK_REQUIRED_MARKER,
            )),
            ..Self::default()
        }
    }

    pub(super) fn can_reopen(
        &self,
        force_final_response: bool,
        iteration: usize,
        max_iterations: usize,
    ) -> bool {
        !self.retry_consumed && !force_final_response && iteration < max_iterations
    }

    pub(super) fn consume_retry(&mut self) {
        self.retry_consumed = true;
    }

    pub(super) fn can_reopen_outstanding_work(
        &self,
        force_final_response: bool,
        iteration: usize,
        max_iterations: usize,
    ) -> bool {
        self.outstanding_work_reopens < OUTSTANDING_WORK_MAX_REOPENS
            && !force_final_response
            && iteration < max_iterations
    }

    pub(super) fn consume_outstanding_work_retry(&mut self) -> usize {
        // Outstanding work may reopen again because it has an independent
        // structural budget, but the same reopen spends the shared quality-gate
        // retry so later gates cannot cascade another retry onto this turn.
        self.retry_consumed = true;
        self.outstanding_work_reopens = self.outstanding_work_reopens.saturating_add(1);
        self.outstanding_work_reopens
    }

    pub(super) fn no_tool_retry_consumed(&self) -> bool {
        self.no_tool_retry_consumed
    }

    pub(super) fn consume_no_tool_retry(&mut self) {
        self.no_tool_retry_consumed = true;
    }
}

/// Marker lookup is scoped to the current real-user turn. This is a compatibility path for
/// direct gate tests and rebuilt request projections; production retry budgets are owned by
/// [`FinalGateState`] and do not depend on marker text.
pub(in crate::ai::driver::turn_runtime) fn current_turn_has_internal_marker(
    messages: &[Message],
    marker: &str,
) -> bool {
    let turn_start = crate::ai::history::last_real_user_index(messages).unwrap_or(0);
    messages.iter().skip(turn_start).any(|message| {
        message.role == ROLE_INTERNAL_NOTE
            && message
                .content
                .as_str()
                .is_some_and(|text| text.starts_with(marker))
    })
}

use crate::ai::request::{
    DERIVED_CONTEXT_HANDOFF, DERIVED_CONTEXT_RETURN, HISTORY_SUMMARY_CONTEXT_HEADER,
    MODEL_SELF_NOTE_CONTEXT_HEADER,
};

/// Exact-prefix sentinels of context notes the runtime injects into the request
/// projection, single-sourced from `request::normalize` (plus the `self_note:`
/// internal-note channel convention shared with `request::normalize`,
/// `completion_gate`, and `finalize`).
/// A legitimate user-visible answer never starts with one of these — if the model
/// spits an injected note back verbatim as its answer, that is an echo.
/// Matching is byte-exact against our own emitted strings; there is no paraphrase
/// list to maintain and nothing to rot when prompt wording changes.
pub(in crate::ai::driver::turn_runtime) const INJECTED_CONTEXT_ECHO_SENTINELS: &[&str] = &[
    MODEL_SELF_NOTE_CONTEXT_HEADER,
    HISTORY_SUMMARY_CONTEXT_HEADER,
    DERIVED_CONTEXT_HANDOFF,
    DERIVED_CONTEXT_RETURN,
    "self_note:",
];
pub(in crate::ai::driver::turn_runtime) const NO_TOOL_SYNTHESIS_RETRY_MARKER: &str =
    "[no-tool-synthesis-retry]";
pub(in crate::ai::driver::turn_runtime) const NO_TOOL_SYNTHESIS_RETRY_NOTE: &str = "The previous no-tool synthesis response incorrectly returned a tool call. Do not call any tool. Produce the final answer now from the evidence already present in the conversation, and explicitly mark anything unverified as incomplete.";
pub(in crate::ai::driver::turn_runtime) const NO_TOOL_SYNTHESIS_WARNING: &str = "The model returned tool calls twice during the no-tool wrap-up stage; the runtime has stopped retrying. Judge the task state only from the evidence already obtained, and treat anything unverified as incomplete.";
pub(in crate::ai::driver::turn_runtime) const REASONING_ONLY_RETRY_MARKER: &str =
    "[reasoning-only-retry]";
pub(in crate::ai::driver::turn_runtime) const REASONING_ONLY_RETRY_NOTE: &str = "The previous response contained hidden reasoning but no visible assistant answer. Retry the step normally with the same capabilities, including tools and internal reasoning when needed, and ensure the response eventually includes visible assistant content.";
pub(in crate::ai::driver::turn_runtime) const REASONING_ONLY_SYNTHESIS_MARKER: &str =
    "[reasoning-only-synthesis]";
pub(in crate::ai::driver::turn_runtime) const REASONING_ONLY_SYNTHESIS_NOTE: &str = "Multiple consecutive responses contained hidden reasoning but no visible assistant answer. Produce the concrete user-facing final answer now. Do not call tools and do not return hidden reasoning alone.";
/// Maximum automatic retries when the response contains only hidden reasoning
/// (only after reaching this limit does the final no-reasoning synthesis kick in).
pub(in crate::ai::driver::turn_runtime) const REASONING_ONLY_MAX_RETRIES: usize = 3;
pub(in crate::ai::driver::turn_runtime) const REASONING_ONLY_SYNTHESIS_RETRY_MARKER: &str =
    "[reasoning-only-synthesis-retry]";
pub(in crate::ai::driver::turn_runtime) const REASONING_ONLY_SYNTHESIS_RETRY_NOTE: &str = "The response still contained hidden reasoning with no visible assistant answer, even after the synthesis instruction. Produce the concrete user-facing final answer now; do not call tools and do not return hidden reasoning alone.";
/// Maximum further automatic retries when the response still contains only hidden
/// reasoning even after the forced no-reasoning synthesis; beyond that the round stops
/// with a user-visible error, avoiding empty spins that repeat identical byte-for-byte
/// requests up to max_iterations.
pub(in crate::ai::driver::turn_runtime) const REASONING_ONLY_POST_SYNTHESIS_MAX_RETRIES: usize = 2;

/// Persistent turn-local marker for subagent-evidence reopens. It survives request
/// projection cleanup so unintegratable results cannot drive retries to the iteration
/// cap. Once the dedicated budget is exhausted, finalization keeps the evidence and
/// attaches a warning.
pub(in crate::ai::driver::turn_runtime) const TASK_EVIDENCE_REOPEN_MARKER: &str =
    "[task-evidence-reopen-count]";
/// Bound subagent-evidence reopens while allowing another chance to call `task_integrate`.
pub(in crate::ai::driver::turn_runtime) const TASK_EVIDENCE_REOPEN_MAX: usize = 3;

/// Count the completion-gate reopen markers already injected into the current `messages`.
/// The marker is an internal_note that reopens do not clear, so it accumulates across iterations.
pub(in crate::ai::driver::turn_runtime) fn task_evidence_reopen_count(
    messages: &[Message],
) -> usize {
    let turn_start = crate::ai::history::last_real_user_index(messages).unwrap_or(0);
    messages
        .iter()
        .skip(turn_start)
        .filter(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && message
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with(TASK_EVIDENCE_REOPEN_MARKER))
        })
        .count()
}

/// Append one reopen-count marker (not cleared by the reopen retain, used to accumulate
/// the count across iterations). Consistent with the other markers in this file: marker
/// prefix + one human-readable sentence, so the projection to the model never shows a
/// bare semantics-free label (internal_note is mapped to system/assistant, see request/normalize).
pub(in crate::ai::driver::turn_runtime) fn push_task_evidence_reopen_marker(
    messages: &mut Vec<Message>,
    attempt: usize,
) {
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(format!(
            "{TASK_EVIDENCE_REOPEN_MARKER}\nOutstanding subagent results were re-surfaced \
             (attempt {attempt}/{TASK_EVIDENCE_REOPEN_MAX}). Call `task_integrate` for each \
             listed task_id now; after the limit the turn will finalize with the evidence attached."
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

pub(in crate::ai::driver::turn_runtime) fn append_runtime_warning_once(
    text: &mut String,
    warning: &str,
) {
    if text.contains(warning) {
        return;
    }
    // Normalize only the append seam; preserve whitespace inside the model's body.
    text.truncate(text.trim_end().len());
    if !text.is_empty() {
        text.push_str("\n\n");
    }
    text.push_str(warning);
}

pub(in crate::ai::driver::turn_runtime) fn append_user_visible_final_notice(
    target: &mut Option<String>,
    notice: &str,
) {
    let text = target.get_or_insert_with(String::new);
    append_runtime_warning_once(text, notice);
}

/// Shared shape of a one-time no-tool final-response recovery: either accept the
/// final (`Allow`), bounce it back for a tool-free synthesis (`RetryWithoutTools`),
/// or stop the round with a user-visible note (`Warn`). Flow control only — the
/// trigger conditions are machine-verifiable turn state, never prose wording.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum FinalRecoveryAction {
    Allow,
    RetryWithoutTools,
    Warn,
}

/// Detect a final body beginning with an injected-context sentinel after stripping an
/// appended runtime warning. This is a prefix check: additional model text after the
/// sentinel does not exempt the response. Sentinels quoted elsewhere in the body do
/// not trigger this gate.
pub(in crate::ai::driver::turn_runtime) fn looks_like_injected_context_echo(
    final_text: &str,
) -> bool {
    // The runtime may append `\n\n[Runtime warning] ...` after the real answer; only
    // the model's body text counts.
    let visible = final_text
        .split_once("\n\n[Runtime warning]")
        .map_or(final_text, |(before, _)| before);
    let visible = visible.trim();
    if visible.is_empty() {
        return false;
    }
    INJECTED_CONTEXT_ECHO_SENTINELS
        .iter()
        .any(|sentinel| visible.starts_with(sentinel))
}

/// Echo gate: on a hit, give one no-tool synthesis retry (preserving pre-reopen
/// capabilities); if the second response still regurgitates, stop the round with a
/// user-visible error so injected notes are never persisted/rendered as the answer.
pub(in crate::ai::driver::turn_runtime) fn injected_context_echo_recovery_action(
    messages: &mut Vec<Message>,
    final_text: &str,
) -> FinalRecoveryAction {
    if !looks_like_injected_context_echo(final_text) {
        return FinalRecoveryAction::Allow;
    }
    let already_retried =
        current_turn_has_internal_marker(messages, INJECTED_CONTEXT_ECHO_RETRY_MARKER);
    if already_retried {
        return FinalRecoveryAction::Warn;
    }
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(format!(
            "{INJECTED_CONTEXT_ECHO_RETRY_MARKER}\n{INJECTED_CONTEXT_ECHO_RETRY_NOTE}"
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    FinalRecoveryAction::RetryWithoutTools
}
