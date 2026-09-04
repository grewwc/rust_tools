//! Final-response recovery gates: dangling-action finals, unsupported
//! runtime-limit claims, no-tool synthesis retries, reasoning-only retries,
//! task-evidence reopen markers, and injected-context echo recovery.

use super::*;

pub(in crate::ai::driver::turn_runtime) const INJECTED_CONTEXT_ECHO_RETRY_MARKER: &str =
    "[injected-context-echo-retry]";
pub(in crate::ai::driver::turn_runtime) const INJECTED_CONTEXT_ECHO_RETRY_NOTE: &str = "Your previous response reproduced a runtime-injected context note verbatim instead of answering. \
Runtime notes are context for you only; they are never the user-facing answer. \
Do not quote, restate, or continue any runtime note — including lines that begin with \
\"[Model-authored note from an earlier turn\", \"[Compressed history summary\", \"[Runtime context handoff\", or \"self_note:\". \
Produce the actual answer to the user's request now, using tools first if verification is still required; if you cannot verify, state that limitation in your own words.";
pub(in crate::ai::driver::turn_runtime) const INJECTED_CONTEXT_ECHO_STOP: &str = "[Model echoed a runtime internal note instead of giving a real answer; please retry or switch models]";

/// Turn-local state for final-response recovery. Prompt markers remain useful context for
/// the model, but runtime control flow must not infer retry budgets from persisted text:
/// old markers can survive into later turns and independent gates can otherwise each spend
/// their own retry, producing a cascade of near-duplicate conclusions.
#[derive(Debug, Default)]
pub(in crate::ai::driver::turn_runtime) struct FinalGateState {
    retry_consumed: bool,
    no_tool_retry_consumed: bool,
}

impl FinalGateState {
    pub(super) fn from_current_turn_markers(messages: &[Message]) -> Self {
        let retry_consumed = [
            INJECTED_CONTEXT_ECHO_RETRY_MARKER,
            UNSUPPORTED_RUNTIME_LIMIT_RETRY_MARKER,
            DANGLING_FINAL_RECOVERY_MARKER,
            COMPLETION_EVIDENCE_REQUIRED_MARKER,
            FINAL_CITATION_RETRY_MARKER,
        ]
        .iter()
        .any(|marker| current_turn_has_internal_marker(messages, marker));
        Self {
            retry_consumed,
            no_tool_retry_consumed: current_turn_has_internal_marker(
                messages,
                NO_TOOL_SYNTHESIS_RETRY_MARKER,
            ),
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

/// Prefixes of context notes that the runtime injects into the request projection.
/// These are all runtime-authored text; a legitimate user-visible answer never starts
/// with them — if the model spits them back verbatim as its answer, that is an echo.
/// The source strings are defined in `request/normalize.rs` (`MODEL_SELF_NOTE_CONTEXT_HEADER`,
/// `HISTORY_SUMMARY_CONTEXT_HEADER`, `DERIVED_CONTEXT_HANDOFF/RETURN`) and in this file's
/// `COMPLETION_EVIDENCE_REQUIRED_MARKER`; here we match on stable prefixes to avoid exposing
/// long constants across modules.
pub(in crate::ai::driver::turn_runtime) const INJECTED_CONTEXT_ECHO_PREFIXES: &[&str] = &[
    "[Model-authored note from an earlier turn",
    "[Compressed history summary for task continuity.",
    "[Runtime context handoff",
    "self_note:",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum FinalClaimKind {
    NoClaim,
    Completion,
    NoImpact,
}

pub(in crate::ai::driver::turn_runtime) const DANGLING_FINAL_RECOVERY_MARKER: &str =
    "[dangling-final-recovery]";
pub(in crate::ai::driver::turn_runtime) const DANGLING_FINAL_WARNING: &str = "[Runtime warning] The model still described a future inspection step after a one-time no-tool wrap-up retry, so this turn ended without a complete conclusion.";
pub(in crate::ai::driver::turn_runtime) const UNSUPPORTED_RUNTIME_LIMIT_RETRY_MARKER: &str =
    "[unsupported-runtime-limit-retry]";
pub(in crate::ai::driver::turn_runtime) const UNSUPPORTED_RUNTIME_LIMIT_WARNING: &str = "[Runtime warning] The model claimed that a read-only phase limit prevented changes, but no matching runtime/tool evidence was observed; the requested work may be incomplete.";
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

/// Marker and cap for the completion gate's dedicated quota on reopening the current
/// turn due to unintegrated subagent evidence.
///
/// Background: while task evidence remains unintegrated, the completion gate bounces the
/// round back (reopen) and asks the model to `task_integrate`. But that veto was originally
/// bounded only by `iteration < max_iterations` (4096), and each reopen cleared the old
/// prompt marker with no accumulated count. When the subagent hit an **unintegratable**
/// dead end such as TIMED_OUT, or the model kept refusing to call `task_integrate`, the turn
/// would reopen forever and spin to the hard cap (one amplifier of the muse-spark dead loop).
/// Here a persistent count marker records the reopen count within one turn; beyond the cap
/// we stop reopening and fall back to the same degraded path as `iteration >= max_iterations`
/// (attaching a warning and letting the ledger finalize).
pub(in crate::ai::driver::turn_runtime) const TASK_EVIDENCE_REOPEN_MARKER: &str =
    "[task-evidence-reopen-count]";
/// Maximum number of reopens within one turn for “unintegrated evidence / unclosed subagent”.
/// Set to 3: enough chances for the model to call `task_integrate` once it has the ledger,
/// yet well before the iteration hard cap, avoiding infinite spinning on dead ends.
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
    if !text.trim().is_empty() {
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

pub(in crate::ai::driver::turn_runtime) fn contains_only_runtime_warnings(text: &str) -> bool {
    let mut saw_warning = false;
    for paragraph in text
        .split("\n\n")
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if paragraph.starts_with("[Runtime warning]") {
            saw_warning = true;
        } else {
            return false;
        }
    }
    saw_warning
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum DanglingFinalRecoveryAction {
    Allow,
    RetryWithoutTools,
    Warn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum UnsupportedRuntimeLimitAction {
    Allow,
    ReopenWithTools,
    Warn,
}

pub(in crate::ai::driver::turn_runtime) fn text_range_is_quoted(
    text: &str,
    start: usize,
    end: usize,
) -> bool {
    for (open, close) in [
        ("\"", "\""),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
        ("《", "》"),
    ] {
        let before = &text[..start];
        let after = &text[end..];
        if open == close {
            if before.matches(open).count() % 2 == 1 && after.contains(close) {
                return true;
            }
        } else if before.rfind(open).is_some_and(|open_index| {
            before
                .rfind(close)
                .is_none_or(|close_index| open_index > close_index)
                && after.contains(close)
        }) {
            return true;
        }
    }
    false
}

pub(in crate::ai::driver::turn_runtime) fn plan_request_phrase_is_negated(
    text: &str,
    start: usize,
) -> bool {
    let clause = text[..start]
        .rsplit(|ch: char| matches!(ch, '.' | ';' | '!' | '?' | '。' | '；' | '！' | '？' | '\n'))
        .next()
        .unwrap_or_default();
    let english_negated = clause
        .split(|ch: char| !ch.is_ascii_alphabetic() && ch != '\'')
        .filter(|token| !token.is_empty())
        .rev()
        .take(8)
        .any(|token| {
            matches!(
                token,
                "not" | "never" | "without" | "don't" | "dont" | "avoid"
            ) || token.ends_with("n't")
        });
    if english_negated {
        return true;
    }

    let chinese_tail = clause
        .chars()
        .rev()
        .take(12)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    ["不要", "不用", "无需", "别", "不需要", "不必"]
        .iter()
        .any(|marker| chinese_tail.contains(marker))
}

pub(in crate::ai::driver::turn_runtime) fn contains_active_plan_request_phrase(
    question: &str,
    phrase: &str,
) -> bool {
    question.match_indices(phrase).any(|(start, _)| {
        let end = start + phrase.len();
        let bytes = question.as_bytes();
        let bounded_before = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let bounded_after = end == bytes.len() || !bytes[end].is_ascii_alphanumeric();
        bounded_before
            && bounded_after
            && !text_range_is_quoted(question, start, end)
            && !plan_request_phrase_is_negated(question, start)
    })
}

pub(in crate::ai::driver::turn_runtime) fn question_requests_plan(question: &str) -> bool {
    let question = question.to_ascii_lowercase();
    let exact = question.trim_matches(|ch: char| ch.is_whitespace() || ch.is_ascii_punctuation());
    if matches!(exact, "next steps" | "实施步骤") {
        return true;
    }

    [
        "give me a plan",
        "provide a plan",
        "create a plan",
        "make a plan",
        "draft a plan",
        "outline a plan",
        "give me next steps",
        "provide next steps",
        "outline next steps",
        "list the next steps",
        "what are the next steps",
        "next steps for",
        "what should i do next",
        "给我一个计划",
        "给出一个计划",
        "制定计划",
        "制定一个计划",
        "列出下一步",
        "给出下一步",
        "下一步怎么做",
        "给出实施步骤",
        "列出实施步骤",
    ]
    .iter()
    .any(|marker| contains_active_plan_request_phrase(&question, marker))
}

pub(in crate::ai::driver::turn_runtime) fn text_claims_read_only_phase_limit(text: &str) -> bool {
    if [
        "触发了只读阶段上限",
        "触发只读阶段上限",
        "达到了只读阶段上限",
        "达到只读阶段上限",
        "到达了只读阶段上限",
        "到达只读阶段上限",
    ]
    .iter()
    .any(|marker| text.contains(marker))
    {
        return true;
    }

    let lower = text.to_ascii_lowercase();
    [
        "hit the read-only phase limit",
        "reached the read-only phase limit",
        "triggered the read-only phase limit",
        "hit the read only phase limit",
        "reached the read only phase limit",
        "triggered the read only phase limit",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub(in crate::ai::driver::turn_runtime) fn text_admits_changes_not_applied(text: &str) -> bool {
    if [
        "尚未写入",
        "尚未修改",
        "还未写入",
        "还未修改",
        "未能写入",
        "未能修改",
        "无法写入",
        "无法修改",
        "没有写入",
        "没有修改",
    ]
    .iter()
    .any(|marker| text.contains(marker))
    {
        return true;
    }

    let lower = text.to_ascii_lowercase();
    [
        "no changes were made",
        "have not written",
        "haven't written",
        "could not write",
        "couldn't write",
        "unable to write",
        "unable to modify",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// Do not treat the model's self-reported execution limits as runtime fact: only allow
/// when the current turn's tool/runtime evidence actually reports the same limit. For
/// the known “read-only phase limit” hallucination, reopen only once and keep the tools.
pub(in crate::ai::driver::turn_runtime) fn unsupported_runtime_limit_action(
    question: &str,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    final_text: &str,
    turn_had_tool_error: bool,
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> UnsupportedRuntimeLimitAction {
    if question_requests_plan(question)
        || !text_claims_read_only_phase_limit(final_text)
        || !text_admits_changes_not_applied(final_text)
        || (turn_had_tool_error
            && turn_messages.iter().any(|message| {
                (message.role == "tool" || message.role == ROLE_INTERNAL_NOTE)
                    && message
                        .content
                        .as_str()
                        .is_some_and(text_claims_read_only_phase_limit)
            }))
    {
        return UnsupportedRuntimeLimitAction::Allow;
    }

    let already_retried =
        current_turn_has_internal_marker(messages, UNSUPPORTED_RUNTIME_LIMIT_RETRY_MARKER);
    if already_retried || force_final_response || iteration >= max_iterations {
        return UnsupportedRuntimeLimitAction::Warn;
    }

    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(format!(
            "{UNSUPPORTED_RUNTIME_LIMIT_RETRY_MARKER}\n\
             The previous final claimed that a read-only phase limit prevented the requested changes, but no tool or runtime evidence in this turn reported such a limit.\n\
             Continue the requested work with the available tools. If an operation is actually blocked, attempt it and report the exact observed error. Do not invent execution phases or limits."
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    UnsupportedRuntimeLimitAction::ReopenWithTools
}

/// Strip inline code spans (backtick-wrapped fragments) and return the plain prose, so
/// symbols such as `.` `:` inside code like `foo.rs`, `.ok()`, `a:b` do not pollute the
/// sentence count and the colon-termination check. Strip only when backticks are paired;
/// when the backtick count is odd (truncated/unpaired), return the text unchanged to
/// avoid deleting the tail of the prose.
pub(in crate::ai::driver::turn_runtime) fn strip_inline_code_spans(text: &str) -> String {
    if text.matches('`').count() % 2 != 0 {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut in_code = false;
    for ch in text.chars() {
        if ch == '`' {
            in_code = !in_code;
            continue;
        }
        if !in_code {
            out.push(ch);
        }
    }
    out
}

/// Count “prose sentence terminators” to decide whether a text is more like a
/// “multi-sentence, formed conclusion” or a one-line “I'll go do X now” aside.
/// The CJK full-stop/exclamation/question marks always count as terminators; ASCII
/// `.` `!` `?` count only when followed by
/// whitespace or the end of the text — otherwise dots in `driver/mod.rs`,
/// `.ok().flatten()`, `3.14` would be miscounted as sentences, dressing a short aside
/// up as a formed conclusion and slipping past the dangling-final gate (one root cause
/// of a model “stopping mid-sentence” while being silently treated as a final response).
pub(in crate::ai::driver::turn_runtime) fn prose_sentence_terminator_count(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut count = 0usize;
    for (index, ch) in chars.iter().enumerate() {
        match ch {
            '。' | '！' | '？' => count += 1,
            '.' | '!' | '?' => {
                let next_is_prose_boundary =
                    chars.get(index + 1).is_none_or(|next| next.is_whitespace());
                if next_is_prose_boundary {
                    count += 1;
                }
            }
            _ => {}
        }
    }
    count
}

/// Detect a dangling final response that verbally promises to keep reading/checking but
/// makes no tool call and delivers no conclusion.
///
/// Stay conservative: only check non-plan tasks with existing tool evidence and short
/// texts without structured conclusions. This is not a general semantic classifier; it
/// fixes the known failure mode where the model mistakes a next-step aside for a final
/// response at the end of a long tool chain.
pub(in crate::ai::driver::turn_runtime) fn looks_like_dangling_action_final(
    question: &str,
    turn_messages: &[Message],
    final_text: &str,
) -> bool {
    if question_requests_plan(question)
        || !turn_messages.iter().any(|message| {
            message.role == "tool"
                || message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
        })
    {
        return false;
    }

    // The runtime may have appended other warnings; classify only the model's raw
    // visible text.
    let candidate = final_text
        .find("[Runtime warning]")
        .map(|index| &final_text[..index])
        .unwrap_or(final_text)
        .trim();
    if candidate.is_empty() {
        return contains_only_runtime_warnings(final_text);
    }
    if candidate.chars().count() > 900 || candidate.contains("```") {
        return false;
    }

    // Classification looks at prose semantics only; strip inline code spans first so
    // symbols in `foo.rs`/`.ok()`/`a:b` do not pollute the sentence count and the
    // colon-termination check.
    let prose = strip_inline_code_spans(candidate);
    let prose = prose.trim();
    if prose.is_empty() {
        // The body is all code fragments with no prose left after stripping: not a
        // “stopped mid-sentence” aside — allow conservatively.
        return false;
    }

    let structured_lines = prose
        .lines()
        .map(str::trim_start)
        .filter(|line| {
            line.starts_with("- ")
                || line.starts_with("* ")
                || line.starts_with("# ")
                || line
                    .split_once('.')
                    .is_some_and(|(prefix, _)| prefix.chars().all(|ch| ch.is_ascii_digit()))
        })
        .count();
    let sentence_ends = prose_sentence_terminator_count(prose);
    if structured_lines >= 2 || sentence_ends > 4 {
        return false;
    }

    // Strong signal: a body ending in a colon is the typical “I'll do X:” teaser that
    // should be followed by a tool call or a list but is cut off here. This kind of
    // “stopped mid-sentence” dangling final is independent of exact wording, so it does
    // not rely on the future-action word list below — the list only covers a limited set
    // of fixed phrases, which is exactly why id=455-style “first look at... check...:”
    // text previously slipped through both the stream classifier and this gate.
    //
    // The criterion applies to the last character of the **raw candidate** (code spans
    // not stripped), not the stripped prose: a normal final like `See the fix: \`bar()\``
    // ends with a code span and really delivers content — its last character is a
    // backtick, not a colon, so it must not be misjudged; only when the colon itself is
    // the last visible character is it a genuinely truncated teaser.
    let ends_with_dangling_colon = candidate.ends_with(':') || candidate.ends_with('：');

    let lower = prose.to_ascii_lowercase();
    let has_future_inspection = ends_with_dangling_colon
        || [
            "let me read",
            "let me inspect",
            "let me check",
            "let me examine",
            "let me look at",
            "let me review",
            "let me trace",
            "let me verify",
            "let me investigate",
            "let me search",
            "let me open",
            "i'll read",
            "i'll inspect",
            "i'll check",
            "i'll examine",
            "i will read",
            "i will inspect",
            "i will check",
            "i will examine",
            "我再读",
            "我再看",
            "我再检查",
            "让我再读",
            "让我再看",
            "让我检查",
            "接下来我会读",
            "接下来我会看",
            "接下来我会检查",
            "接下来让我",
            "下一步我会读",
            "下一步我会检查",
            "现在我来读",
            "现在我来检查",
        ]
        .iter()
        .any(|marker| lower.contains(marker));
    if !has_future_inspection {
        return false;
    }

    ![
        "conclusion:",
        "findings:",
        "root cause",
        "the issue is",
        "the bug is",
        "verified finding",
        "no verified finding",
        "结论：",
        "结论:",
        "根因：",
        "根因:",
        "问题是：",
        "问题是:",
        "已验证",
        "未发现问题",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub(in crate::ai::driver::turn_runtime) fn dangling_final_recovery_action(
    question: &str,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    final_text: &str,
) -> DanglingFinalRecoveryAction {
    if !looks_like_dangling_action_final(question, turn_messages, final_text) {
        return DanglingFinalRecoveryAction::Allow;
    }

    let already_retried =
        current_turn_has_internal_marker(messages, DANGLING_FINAL_RECOVERY_MARKER);
    if already_retried {
        return DanglingFinalRecoveryAction::Warn;
    }

    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(format!(
            "{DANGLING_FINAL_RECOVERY_MARKER}\n\
             Your previous response did not deliver findings or a conclusion; it only promised more inspection or repeated runtime warnings.\n\
             This is a one-time synthesis recovery, not a new investigation round. Do not call tools.\n\
             Based only on evidence already present in the context, give the final answer now. If evidence is insufficient, state the exact unresolved gap and why it could not be verified; do not narrate future actions."
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    DanglingFinalRecoveryAction::RetryWithoutTools
}

pub(in crate::ai::driver::turn_runtime) fn final_text_claim_kind(text: &str) -> FinalClaimKind {
    if ["没有影响", "未影响", "不会影响", "不影响", "保持不变"]
        .iter()
        .any(|claim| text.contains(claim))
    {
        return FinalClaimKind::NoImpact;
    }
    if [
        "已完成",
        "已修复",
        "全部修复",
        "修复完成",
        "已更新",
        "已经更新",
        "已修改",
        "已经修改",
    ]
    .iter()
    .any(|claim| text.contains(claim))
    {
        return FinalClaimKind::Completion;
    }

    let text = text.to_ascii_lowercase();
    if [
        "no impact",
        "unaffected",
        "unchanged",
        "does not affect",
        "doesn't affect",
    ]
    .iter()
    .any(|claim| text.contains(claim))
    {
        return FinalClaimKind::NoImpact;
    }
    if [
        "completed",
        "fixed",
        "resolved",
        "implemented",
        "done",
        "updated",
        "changed",
    ]
    .iter()
    .any(|word| contains_non_negated_completion_word(&text, word))
        || [
            "changes are ready",
            "change is ready",
            "implementation is ready",
            "fix is ready",
            "patch is ready",
        ]
        .iter()
        .any(|claim| text.contains(claim))
    {
        return FinalClaimKind::Completion;
    }
    FinalClaimKind::NoClaim
}

/// Decide whether the final response merely regurgitates a context note the runtime
/// injected, verbatim, without giving a real answer. Hit signature: after stripping the
/// `[Runtime warning]` section the runtime appended post-hoc, the remaining visible body
/// starts with some injected-note prefix. Such responses are worthless to the user and
/// leak internal prompts to the terminal (especially common with weak models after a
/// completion-evidence / dangling gate reopen).
///
/// Stay conservative: only handle the case where the whole body is an injected note. If
/// the model quotes/discusses these prefixes in the body (prefix not at the start, or
/// followed by its own text) it is not an echo and is left to the other gates.
pub(in crate::ai::driver::turn_runtime) fn looks_like_injected_context_echo(
    final_text: &str,
) -> bool {
    // The runtime may append `\n\n[Runtime warning] ...` after the real answer; classify
    // only the model's body text.
    let visible = final_text
        .split_once("\n\n[Runtime warning]")
        .map_or(final_text, |(before, _)| before);
    let visible = visible.trim();
    if visible.is_empty() {
        return false;
    }
    INJECTED_CONTEXT_ECHO_PREFIXES
        .iter()
        .any(|prefix| visible.starts_with(prefix))
}

/// Echo gate: on a hit, give one no-tool synthesis retry (preserving pre-reopen
/// capabilities); if the second response still regurgitates, stop the round with a
/// user-visible error so injected notes are never persisted/rendered as the answer.
pub(in crate::ai::driver::turn_runtime) fn injected_context_echo_recovery_action(
    messages: &mut Vec<Message>,
    final_text: &str,
) -> DanglingFinalRecoveryAction {
    if !looks_like_injected_context_echo(final_text) {
        return DanglingFinalRecoveryAction::Allow;
    }
    let already_retried =
        current_turn_has_internal_marker(messages, INJECTED_CONTEXT_ECHO_RETRY_MARKER);
    if already_retried {
        return DanglingFinalRecoveryAction::Warn;
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
    DanglingFinalRecoveryAction::RetryWithoutTools
}
