use crate::ai::tools::task_tools::{EvidenceRecoveryFormat, full_delivered_evidence_hint};
use crate::ai::{
    driver::print::{format_empty_state, print_assistant_banner_with_app_and_skill},
    history::{
        Message, SessionTitle, SessionTitleOrigin, compact_session_history_at_boundary_with_app,
        compact_session_history_with_app, generate_session_summary, is_low_quality_session_title,
        is_runtime_synthetic_user_message, normalize_generated_session_title, value_to_string,
    },
    types::App,
};
use rust_tools::commonw::FastSet;
use serde_json::Value;
use std::{
    borrow::Cow,
    sync::{LazyLock, Mutex},
};

use super::{
    TurnOutcome, persistence::persist_pending_turn_messages_for_model, tool_result::FinalGateState,
};

const SUBAGENT_TOOL_EVIDENCE_MAX_CALLS: usize = 8;
const SUBAGENT_TOOL_EVIDENCE_MAX_CHARS_PER_RESULT: usize = 700;
const SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS: usize = 4_000;
const AGENT_TEAM_SUBAGENT_TOOL_EVIDENCE_MAX_CALLS: usize = 4;
const AGENT_TEAM_SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS: usize = 1_200;
const MODEL_CONTEXT_ECHO_PREFIX: &str = "[Model-authored note from an earlier turn;";
pub(crate) const SERVE_TITLE_SEED_ENV: &str = "A_SERVE_TITLE_SEED";
pub(crate) const SERVE_TITLE_SESSION_ENV: &str = "A_SERVE_TITLE_SESSION";

/// The submitted title policy remains distinct from the turn child's command input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SessionTitleSeed {
    Legacy(Option<String>),
    Disabled,
    Explicit(String),
}

impl SessionTitleSeed {
    pub(crate) fn from_optional(seed: Option<&str>, pending_input: Option<&str>) -> Self {
        match seed.map(str::trim) {
            None => Self::Legacy(pending_input.map(str::to_string)),
            Some("") => Self::Disabled,
            Some(question) => Self::Explicit(question.to_string()),
        }
    }

    pub(crate) fn child_env_value(&self) -> Option<&str> {
        match self {
            Self::Legacy(_) => None,
            Self::Disabled => Some(""),
            Self::Explicit(question) => Some(question),
        }
    }
}

fn session_title_seed_for_process(
    session_id: &str,
    pending_input: Option<&str>,
) -> SessionTitleSeed {
    if !crate::ai::background::is_serve_child()
        || std::env::var(SERVE_TITLE_SESSION_ENV).ok().as_deref() != Some(session_id)
    {
        return SessionTitleSeed::from_optional(None, pending_input);
    }
    match std::env::var(SERVE_TITLE_SEED_ENV) {
        Ok(seed) => SessionTitleSeed::from_optional(Some(&seed), pending_input),
        Err(std::env::VarError::NotPresent) => SessionTitleSeed::from_optional(None, pending_input),
        Err(std::env::VarError::NotUnicode(_)) => SessionTitleSeed::Disabled,
    }
}

static SESSION_TITLE_IN_FLIGHT: LazyLock<Mutex<FastSet<String>>> =
    LazyLock::new(|| Mutex::new(FastSet::default()));
/// Within this process, overlapping turns must not run persisted post-turn
/// compression for the same session concurrently.
static SESSION_COMPACTION_IN_FLIGHT: LazyLock<Mutex<FastSet<String>>> =
    LazyLock::new(|| Mutex::new(FastSet::default()));

tokio::task_local! {
    /// The spawning parent scopes its formatting policy around the child turn.
    /// Child skills do not determine how evidence is delivered to the parent.
    pub(in crate::ai::driver) static SUBAGENT_PARENT_AGENT_TEAM_MODE: bool;
}

/// The title task must derive the sessions root from the base history file; do not pass the
/// parent directory of the current session database, or a nested `.sessions` path is built by mistake.
fn session_title_store(history_file: &std::path::Path) -> crate::ai::history::SessionStore {
    crate::ai::history::SessionStore::new(history_file)
}

fn ensure_final_assistant_recorded(
    final_assistant_text: &str,
    final_assistant_recorded: bool,
    turn_messages: &mut Vec<Message>,
) {
    if final_assistant_recorded {
        return;
    }

    turn_messages.push(Message {
        role: "assistant".to_string(),
        content: Value::String(final_assistant_text.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Prepare all delivery channels before publication, without reopening model review.
/// A recorded body stays intact in history; only its new warning needs another row.
fn prepare_critical_premises_delivery<'a>(
    final_assistant_text: &'a str,
    final_assistant_recorded: bool,
    user_visible_final_suffix: Option<&'a str>,
    has_pending_critical_premises: bool,
    turn_messages: &mut Vec<Message>,
) -> (Cow<'a, str>, bool, Option<Cow<'a, str>>) {
    if !has_pending_critical_premises || final_assistant_text.trim().is_empty() {
        return (
            Cow::Borrowed(final_assistant_text),
            final_assistant_recorded,
            user_visible_final_suffix.map(Cow::Borrowed),
        );
    }

    let warning = format!(
        "[Runtime warning] {}",
        FinalGateState::CRITICAL_PREMISES_WARNING
    );
    if final_assistant_text.contains(FinalGateState::CRITICAL_PREMISES_WARNING) {
        // Recorded answers retain their explicit redraw/stream-deduplication policy.
        // Local fallbacks hide the body's warning tail, so give both visible consumers
        // a warning-only suffix rather than replaying an old narration candidate.
        let suffix = if final_assistant_recorded {
            user_visible_final_suffix.map(Cow::Borrowed)
        } else {
            let suffix = user_visible_final_suffix
                .map(str::trim)
                .filter(|text| text.starts_with("[Runtime warning]"));
            Some(match suffix {
                Some(suffix)
                    if suffix
                        .matches(FinalGateState::CRITICAL_PREMISES_WARNING)
                        .count()
                        == 1 =>
                {
                    Cow::Borrowed(suffix)
                }
                Some(suffix) => {
                    // Keep other blank-line-separated gate notices while replacing
                    // missing or duplicate critical notices with one canonical warning.
                    let other_notices = suffix
                        .split("\n\n")
                        .filter(|notice| {
                            !notice.contains(FinalGateState::CRITICAL_PREMISES_WARNING)
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    Cow::Owned(if other_notices.is_empty() {
                        warning
                    } else {
                        format!("{other_notices}\n\n{warning}")
                    })
                }
                None => Cow::Owned(warning),
            })
        };
        return (
            Cow::Borrowed(final_assistant_text),
            final_assistant_recorded,
            suffix,
        );
    }
    if final_assistant_recorded {
        ensure_final_assistant_recorded(&warning, false, turn_messages);
    }
    (
        Cow::Owned(format!("{final_assistant_text}\n\n{warning}")),
        final_assistant_recorded,
        Some(Cow::Owned(warning)),
    )
}

/// Hide local fallback bodies that echo notes intended for the model while
/// retaining the original answer in canonical history. Recorded responses render
/// only an explicitly user-visible suffix to avoid repeating streamed text.
fn terminal_final_text_to_render(
    final_assistant_text: &str,
    final_assistant_recorded: bool,
    user_visible_suffix: Option<&str>,
) -> Option<String> {
    // The final model response already written to canonical history was streamed live by the stream
    // runtime; finalize only paints local fallbacks that never went through streaming, so the body
    // is not duplicated in the terminal.
    if final_assistant_recorded {
        return user_visible_suffix
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
    }
    let terminal_text = final_assistant_text
        .split_once("\n\n[Runtime warning]")
        .map_or(final_assistant_text, |(visible, _)| visible);
    let trimmed = terminal_text.trim_start();
    if trimmed.starts_with(MODEL_CONTEXT_ECHO_PREFIX)
        || trimmed.starts_with("self_note:")
        || trimmed.starts_with("[Runtime warning]")
    {
        return None;
    }
    // This fallback bypasses the streaming projection, so a machine-checked audit
    // envelope must be hidden here as well; canonical history keeps the original.
    let mut visible_text = crate::ai::stream::hide_audit_report(terminal_text);
    if let Some(warning) = user_visible_suffix
        .map(str::trim)
        .filter(|text| text.starts_with("[Runtime warning]"))
    {
        if !visible_text.is_empty() {
            visible_text.push_str("\n\n");
        }
        visible_text.push_str(warning);
    }
    Some(visible_text)
}

fn serve_final_text_to_commit(
    final_assistant_text: &str,
    user_visible_suffix: Option<&str>,
) -> Option<String> {
    let base = terminal_final_text_to_render(final_assistant_text, false, None)?;
    let base = base.trim();
    let suffix = user_visible_suffix
        .map(str::trim)
        .filter(|text| !text.is_empty());
    let visible = match (base.is_empty(), suffix) {
        (true, Some(suffix)) => suffix.to_string(),
        (true, None) => return None,
        (false, Some(suffix)) if suffix.starts_with(base) => suffix.to_string(),
        (false, Some(suffix)) => format!("{base}\n\n{suffix}"),
        (false, None) => base.to_string(),
    };
    Some(visible)
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

fn normalized_tool_args(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| raw.trim().to_string())
}

fn collect_subagent_tool_evidence(turn_messages: &[Message]) -> Vec<String> {
    use rustc_hash::FxHashMap;

    let mut id_to_call: FxHashMap<String, (String, String)> = FxHashMap::default();
    for message in turn_messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            id_to_call.insert(
                tool_call.id.clone(),
                (
                    tool_call.function.name.clone(),
                    normalized_tool_args(&tool_call.function.arguments),
                ),
            );
        }
    }

    let mut evidence = Vec::new();
    for message in turn_messages {
        if message.role != "tool" {
            continue;
        }
        let Some(tool_call_id) = message.tool_call_id.as_ref() else {
            continue;
        };
        let Some((tool_name, args)) = id_to_call.get(tool_call_id) else {
            continue;
        };
        let content = value_to_string(&message.content);
        let content = content.trim();
        if content.is_empty() {
            continue;
        }
        evidence.push(format!(
            "- {}({}) => {}",
            tool_name,
            args,
            truncate_chars(content, SUBAGENT_TOOL_EVIDENCE_MAX_CHARS_PER_RESULT)
        ));
        if evidence.len() >= SUBAGENT_TOOL_EVIDENCE_MAX_CALLS {
            break;
        }
    }
    evidence
}

fn format_subagent_result_for_parent(
    final_assistant_text: &str,
    turn_messages: &[Message],
) -> String {
    let parent_agent_team_mode = SUBAGENT_PARENT_AGENT_TEAM_MODE
        .try_with(|mode| *mode)
        .unwrap_or(false);
    format_subagent_result_for_parent_with_policy(
        final_assistant_text,
        turn_messages,
        parent_agent_team_mode,
    )
}

fn format_subagent_result_for_parent_with_policy(
    final_assistant_text: &str,
    turn_messages: &[Message],
    parent_agent_team_mode: bool,
) -> String {
    if parent_agent_team_mode {
        format_agent_team_subagent_result_for_parent(final_assistant_text, turn_messages)
    } else {
        format_full_subagent_result_for_parent(final_assistant_text, turn_messages)
    }
}

fn format_full_subagent_result_for_parent(
    final_assistant_text: &str,
    turn_messages: &[Message],
) -> String {
    let recovery_hint = full_delivered_evidence_hint(EvidenceRecoveryFormat::ToolCall);
    let final_assistant_text = final_assistant_text.trim();
    let evidence = collect_subagent_tool_evidence(turn_messages);
    if evidence.is_empty() {
        return final_assistant_text.to_string();
    }

    let mut evidence_block = String::from("[Subagent tool evidence]\n");
    evidence_block.push_str(
        "The subagent used these tool results while producing the answer; treat them as already observed context.\n",
    );
    evidence_block.push_str(&evidence.join("\n"));

    let mut output = truncate_chars(&evidence_block, SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS);
    if evidence_block.chars().count() > SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS {
        // Mark block truncation explicitly. The recovery pointer addresses the
        // delivered payload; individual tool results above are already excerpts.
        output.push_str(&format!(
            "\n[evidence truncated: showing {} of {} characters. Recover the full delivered \
             payload with {recovery_hint}.]",
            output.chars().count(),
            evidence_block.chars().count()
        ));
    }
    if final_assistant_text.is_empty() {
        return output;
    }
    output.push_str("\n\n");
    output.push_str("[Subagent final answer]\n");
    output.push_str(final_assistant_text);
    output
}

fn agent_team_evidence_line(text: &str, max_chars: usize) -> String {
    let sanitized: String = text
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect();
    truncate_chars(
        &sanitized.split_whitespace().collect::<Vec<_>>().join(" "),
        max_chars,
    )
}

fn format_agent_team_subagent_result_for_parent(
    final_assistant_text: &str,
    turn_messages: &[Message],
) -> String {
    let recovery_hint = full_delivered_evidence_hint(EvidenceRecoveryFormat::ToolCall);
    use super::tool_result::completion_tool_result_succeeded;
    use rust_tools::commonw::FastMap;

    let final_assistant_text = final_assistant_text.trim();
    let calls: FastMap<_, _> = turn_messages
        .iter()
        .filter_map(|message| message.tool_calls.as_ref())
        .flatten()
        .map(|call| (call.id.as_str(), call))
        .collect();
    let mut evidence = Vec::new();
    for message in turn_messages.iter().rev() {
        if message.role != "tool" {
            continue;
        }
        let Some(call) = message.tool_call_id.as_deref().and_then(|id| calls.get(id)) else {
            continue;
        };
        let content = value_to_string(&message.content);
        if content.trim().is_empty() {
            continue;
        }
        let failed = !completion_tool_result_succeeded(&message.content);
        evidence.push((failed, *call, content));
    }
    if evidence.is_empty() {
        return final_assistant_text.to_string();
    }
    // Inspect the entire turn before selecting excerpts: a late failure must not be hidden by successes.
    evidence.sort_by_key(|(failed, _, _)| !*failed);
    let failures = evidence.iter().filter(|(failed, _, _)| *failed).count();
    let shown = evidence
        .len()
        .min(AGENT_TEAM_SUBAGENT_TOOL_EVIDENCE_MAX_CALLS);
    let mut output = format!(
        "[Subagent tool evidence]\nUnverified excerpts: {shown}/{} results; failures={failures}, \
         failure excerpts shown={}.\n",
        evidence.len(),
        failures.min(shown),
    );
    let pointer = format!("Full delivered payload: {recovery_hint}. Ellipses mark excerpts.\n");
    let line_limit = AGENT_TEAM_SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS
        .saturating_sub(output.chars().count() + pointer.chars().count())
        / shown;
    for (failed, call, content) in evidence.iter().take(shown) {
        let prefix = format!(
            "- {}({}){} => ",
            agent_team_evidence_line(&call.function.name, 40),
            agent_team_evidence_line(&normalized_tool_args(&call.function.arguments), 60),
            if *failed { " [failed]" } else { "" },
        );
        output.push_str(&prefix);
        output.push_str(&agent_team_evidence_line(
            content,
            line_limit.saturating_sub(prefix.chars().count() + 1),
        ));
        output.push('\n');
    }
    output.push_str(&pointer);
    debug_assert!(output.chars().count() <= AGENT_TEAM_SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS);
    if !final_assistant_text.is_empty() {
        output.push_str("\n[Subagent final answer]\n");
        output.push_str(final_assistant_text);
    }
    output
}

fn subagent_result_payload_for_parent(
    final_assistant_text: &str,
    turn_messages: &[Message],
) -> Option<String> {
    let output = format_subagent_result_for_parent(final_assistant_text, turn_messages);
    (!output.trim().is_empty()).then_some(output)
}

pub(in crate::ai::driver::turn_runtime) fn should_generate_session_title_in_background(
    one_shot_mode: bool,
    should_quit: bool,
) -> bool {
    !one_shot_mode && !should_quit
}

fn should_compact_session_history_in_background(
    one_shot_mode: bool,
    should_quit: bool,
    is_subagent: bool,
    is_serve_child: bool,
) -> bool {
    // Subagent history is managed by the caller's lifecycle guard; if compression escapes the current
    // future, the guard may delete the temp SQLite first and the background task then touches a missing file.
    // Serve children must commit snapshots within their process lifetime, including resumed turns
    // that are not one-shot. Detached work can be lost at exit, releasing the parent's turn lock
    // before snapshot persistence finishes and leaving the persisted tail growing.
    !is_subagent && !is_serve_child && !one_shot_mode && !should_quit
}

fn should_run_post_turn_title(is_serve_child: bool) -> bool {
    !is_serve_child
}

fn mark_session_title_generation_started(session_id: &str) -> bool {
    let mut in_flight = SESSION_TITLE_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.insert(session_id.to_string())
}

fn mark_session_title_generation_finished(session_id: &str) {
    let mut in_flight = SESSION_TITLE_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.remove(session_id);
}

/// Own the in-flight slot until completion, cancellation, or task unwind.
pub(in crate::ai::driver::turn_runtime) struct SessionCompactionGuard {
    session_id: String,
}

impl Drop for SessionCompactionGuard {
    fn drop(&mut self) {
        let mut in_flight = SESSION_COMPACTION_IN_FLIGHT
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        in_flight.remove(&self.session_id);
    }
}

pub(in crate::ai::driver::turn_runtime) fn try_start_session_compaction(
    session_id: &str,
) -> Option<SessionCompactionGuard> {
    let mut in_flight = SESSION_COMPACTION_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight
        .insert(session_id.to_string())
        .then(|| SessionCompactionGuard {
            session_id: session_id.to_string(),
        })
}

/// Interactive finalization dispatches persisted compression to the background: the compression
/// result is a replaceable context snapshot. The next preparation never waits for this task: it
/// can safely read the generation-bound snapshot plus the canonical tail after its watermark.
/// The foreground turn therefore does not have to wait for a CPU compression + SQLite write transaction
/// after the answer is delivered.
///
/// `at_boundary` records whether tools were called again this round (no tool calls = "answer
/// delivered", using the more aggressive threshold). After return, the spawned task owns clearing the
/// in-flight guard.
fn spawn_background_compaction(app: &App, at_boundary: bool) {
    let Some(compaction_guard) = try_start_session_compaction(&app.session_id) else {
        // A compression is already running (e.g. the previous round's background task has not finished);
        // skipping this round is fine because a later post-turn dispatch re-evaluates the latest
        // canonical history, while request projection independently merges the canonical tail.
        return;
    };
    let task_app = app.snapshot_for_detached_helper();
    // Background tasks do not own the foreground terminal: compression warnings/logs must not steal the foreground output cursor.
    tokio::spawn(
        crate::ai::driver::runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.scope(true, async move {
            let _compaction_guard = compaction_guard;
            let compact_result = if at_boundary {
                compact_session_history_at_boundary_with_app(
                    &task_app,
                    crate::ai::driver::runtime_ctx::effective_cwd()
                        .ok()
                        .as_deref(),
                )
                .await
            } else {
                compact_session_history_with_app(
                    &task_app,
                    crate::ai::driver::runtime_ctx::effective_cwd()
                        .ok()
                        .as_deref(),
                )
                .await
            };
            // Runs inside the terminal-output-suppressed scope above: a background
            // failure must not write to the TTY while the textarea owns it (the raw
            // newline would shift the parked cursor and overwrite the help line).
            // Compaction is a best-effort write-back cache; a future post-turn
            // dispatch retries it without delaying a model request.
            if let Err(err) = compact_result
                && crate::ai::driver::runtime_ctx::terminal_output_enabled()
            {
                eprintln!("[Warning] Failed to compact persisted history: {}", err);
            }
        }),
    );
}

/// Interactive turns outside serve children dispatch persisted compaction to the
/// background. One-shot, exiting, subagent, and serve-child turns await the
/// attempt before the owning history's lifetime ends.
async fn dispatch_finalize_compaction(app: &App, at_boundary: bool, run_in_background: bool) {
    if run_in_background {
        spawn_background_compaction(app, at_boundary);
        return;
    }
    let compact_result = if at_boundary {
        compact_session_history_at_boundary_with_app(
            app,
            crate::ai::driver::runtime_ctx::effective_cwd()
                .ok()
                .as_deref(),
        )
        .await
    } else {
        compact_session_history_with_app(
            app,
            crate::ai::driver::runtime_ctx::effective_cwd()
                .ok()
                .as_deref(),
        )
        .await
    };
    if let Err(err) = compact_result
        && crate::ai::driver::runtime_ctx::terminal_output_enabled()
    {
        eprintln!("[Warning] Failed to compact persisted history: {}", err);
    }
}

/// The title task may start before the current user message is persisted; patch this committed input
/// into the snapshot so the helper model does not have to wait for the main request or the first assistant response.
fn session_title_messages(
    mut persisted_messages: Vec<Message>,
    pending_user_input: Option<&str>,
) -> Vec<Message> {
    let Some(user_input) = pending_user_input
        .map(str::trim)
        .filter(|input| !input.is_empty())
    else {
        return persisted_messages;
    };

    persisted_messages.push(Message {
        role: "user".to_string(),
        content: Value::String(user_input.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    persisted_messages
}

/// Only legacy clients load the persisted transcript. Explicit intent is the whole source.
fn session_title_source(
    seed: &SessionTitleSeed,
    load_legacy: impl FnOnce(Option<&str>) -> Option<Vec<Message>>,
) -> Option<Vec<Message>> {
    match seed {
        SessionTitleSeed::Legacy(pending) => load_legacy(pending.as_deref()),
        SessionTitleSeed::Disabled => None,
        SessionTitleSeed::Explicit(question) => {
            Some(session_title_messages(Vec::new(), Some(question)))
        }
    }
}

fn has_session_title_source(messages: &[Message]) -> bool {
    messages.iter().any(|message| {
        message.role == "user"
            // Synthetic handoffs carry derived evidence rather than new user intent,
            // so they must not seed title generation.
            && !is_runtime_synthetic_user_message(message)
            // A user message that only captured the terminal's resume-selection
            // prompt echo (no real intent) must not seed title generation.
            && !crate::ai::history::strip_terminal_prompt_echo(&value_to_string(&message.content))
                .trim()
                .is_empty()
    })
}

/// The matching basis between the fallback and legacy provenance-less titles must stay identical to what was persisted.
fn fallback_session_title(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|message| message.role == "user" && !is_runtime_synthetic_user_message(message))
        .map(|message| value_to_string(&message.content))
        .map(|text| normalize_generated_session_title(&generate_session_summary(&text)))
        .find(|title| !title.is_empty())
        .unwrap_or_default()
}

fn should_generate_model_session_title(
    existing: Option<&SessionTitle>,
    fallback_title: &str,
) -> bool {
    let Some(existing) = existing else {
        return true;
    };
    // A user-set title is authoritative: the raw-fragment quality heuristics
    // below apply to auto-generated titles only, never to an explicit user
    // choice (a long or filler-prefixed user title must not be regenerated).
    if existing.origin == SessionTitleOrigin::User {
        return false;
    }
    if is_low_quality_session_title(&existing.text) {
        return true;
    }

    match existing.origin {
        SessionTitleOrigin::Fallback => true,
        // Legacy titles without provenance are upgraded only when they exactly match the old fallback,
        // avoiding accidental overwrite of good historical titles the model already generated.
        SessionTitleOrigin::Legacy => {
            !fallback_title.is_empty()
                && normalize_generated_session_title(&existing.text) == fallback_title
        }
        // A model title that passed the quality check does not need regeneration.
        SessionTitleOrigin::Model => false,
        // Unreachable: User was returned above. Kept for exhaustive matching.
        SessionTitleOrigin::User => false,
    }
}

fn should_write_fallback_session_title(
    existing: Option<&SessionTitle>,
    fallback_title: &str,
) -> bool {
    match existing {
        None => true,
        // A user-set title is authoritative; never replace it with a fallback.
        Some(existing) if existing.origin == SessionTitleOrigin::User => false,
        Some(existing) if is_low_quality_session_title(&existing.text) => true,
        // Stamp provenance when migrating a legacy fallback; model-based upgrades can then be retried reliably.
        Some(existing) => {
            existing.origin == SessionTitleOrigin::Legacy
                && normalize_generated_session_title(&existing.text) == fallback_title
        }
    }
}

pub(super) async fn finalize_turn(
    app: &mut App,
    _next_model: &str,
    response_source_model: &str,
    question: &str,
    final_assistant_text: &str,
    final_assistant_recorded: bool,
    request_error_detail: Option<&str>,
    user_visible_final_suffix: Option<&str>,
    has_pending_critical_premises: bool,
    active_skill_name: Option<&str>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    should_quit: bool,
    _had_tool_error: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    let (delivery_text, final_assistant_recorded, delivery_suffix) =
        prepare_critical_premises_delivery(
            final_assistant_text,
            final_assistant_recorded,
            user_visible_final_suffix,
            has_pending_critical_premises,
            turn_messages,
        );
    let final_assistant_text = delivery_text.as_ref();
    let user_visible_final_suffix = delivery_suffix.as_deref();
    if let Some(subagent_output_for_parent) =
        subagent_result_payload_for_parent(final_assistant_text, turn_messages)
    {
        // Publish to the parent agent early: even without final assistant prose this round, any reusable
        // subagent evidence (e.g. read_file results) must be visible to the parent.
        // This keeps the foreground runner and the asynchronous `task_wait` on the same parent-side payload.
        crate::ai::driver::runtime_ctx::publish_subagent_result(
            &subagent_output_for_parent,
            final_assistant_text,
        )
        .await;
    }

    if !final_assistant_text.trim().is_empty() {
        if let Some(detail) = request_error_detail {
            turn_messages.push(crate::ai::history::request_error_message(
                final_assistant_text,
                detail,
            ));
        } else {
            ensure_final_assistant_recorded(
                final_assistant_text,
                final_assistant_recorded,
                turn_messages,
            );
        }
        // Persist before any terminal rendering. Rendering is display-only and can
        // fail (e.g. EAGAIN on a non-blocking tty); such a failure must never drop
        // the canonical assistant message from history.
        persist_pending_turn_messages_for_model(
            app,
            response_source_model,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
        );
        // History owns the diagnostic before replay can replace the live row.
        if let Some(detail) = request_error_detail {
            crate::ai::background::publish_serve_json_frame(
                crate::ai::background::ServeLiveKind::RequestError,
                detail,
            );
        }
        // Serve children commit the accepted body through their live FIFO below;
        // their stdout must not render a second copy.
        if crate::ai::driver::runtime_ctx::terminal_output_enabled()
            && !crate::ai::background::serve_live_streaming()
            && let Some(visible_text) = terminal_final_text_to_render(
                final_assistant_text,
                final_assistant_recorded,
                user_visible_final_suffix,
            )
        {
            print_assistant_banner_with_app_and_skill(Some(app), active_skill_name);
            // The digest is extra image understanding for the model; it is stripped from the final echo as well.
            let visible_text = crate::ai::request::strip_digest_blocks(&visible_text);
            // Display-only post-processing (e.g. `scripts/postprocess_terminal.py`
            // via `ai.output.postprocess_command`). Best-effort: on any failure
            // the original text is shown unchanged; canonical history is untouched.
            let visible_text = super::output_postprocess::postprocess_terminal_text(visible_text);
            crate::ai::stream::render_markdown_block(&visible_text)?;
            // Mirror what the user just read: if a cursor-position query times
            // out, the input box falls back to the alternate screen with no
            // transcript to sit under, and shows this tail above itself.
            if let Some(editor) = app.prompt_editor.as_mut() {
                editor.set_alternate_screen_tail(&visible_text);
            }
        }
        if crate::ai::background::serve_live_streaming()
            && crate::ai::driver::runtime_ctx::terminal_output_enabled()
        {
            // Live deltas are provisional. Commit the authoritative body even
            // when it is empty so prose from a rejected attempt is cleared.
            let visible_text =
                serve_final_text_to_commit(final_assistant_text, user_visible_final_suffix)
                    .map(|text| crate::ai::request::strip_digest_blocks(&text))
                    .unwrap_or_default();
            crate::ai::background::publish_committed_serve_output(&visible_text);
            // The accepted body is final; end the user-visible turn here rather
            // than at child exit. Compression and title generation below are
            // silent bookkeeping and can take minutes, during which the client
            // would otherwise keep its next input blocked behind this turn.
            // A side note or talk item already queued keeps the stream open so
            // the run loop still delivers that follow-up in place.
            if one_shot_mode
                && !crate::ai::driver::side_note::has_pending_side_notes(
                    &app.session_history_file,
                    None,
                )
                && !crate::ai::driver::talk::has_pending_talk(&app.session_history_file)
            {
                crate::ai::background::mark_serve_output_complete();
            }
        }
        // Tool-free turns use the more aggressive boundary compaction policy.
        let had_tool_calls = turn_messages
            .iter()
            .any(|m| m.role == "tool" || m.tool_calls.as_ref().map_or(false, |c| !c.is_empty()));
        // The goal loop also consumes this flag describing the whole turn's tool traffic.
        app.last_turn_had_tool_calls = had_tool_calls;

        let mut first_observer_emitted = false;
        let mut poisoned: Vec<String> = Vec::new();
        for obs in app.observers.iter_mut() {
            if obs.is_poisoned() {
                continue;
            }
            let ctx = crate::ai::driver::observer::FinalizeContext {
                question: question.to_string(),
                final_text: final_assistant_text.to_string(),
                had_tool_calls,
            };
            let obs_name = obs.name().to_string();
            let output = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                obs.on_finalize(&ctx)
            })) {
                Ok(o) => o,
                Err(_) => {
                    if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                        eprintln!(
                            "[Warning] observer '{}' panicked in on_finalize; disabling for rest of conversation.",
                            obs_name
                        );
                    }
                    obs.mark_poisoned();
                    poisoned.push(obs_name);
                    continue;
                }
            };
            if output.display_lines.is_empty() {
                continue;
            }
            if first_observer_emitted && crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                println!("---");
            }
            first_observer_emitted = true;
            if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                for line in &output.display_lines {
                    println!("{}", line);
                }
            }
        }
        let _ = poisoned;
        // Serve children await the snapshot attempt while holding the parent's
        // session turn lock, including resumed turns. Tool-free turns select the
        // boundary policy.
        let serve_child = crate::ai::background::is_serve_child();
        dispatch_finalize_compaction(
            app,
            !had_tool_calls,
            should_compact_session_history_in_background(
                one_shot_mode,
                should_quit,
                crate::ai::driver::runtime_ctx::has_subagent_result_slot(),
                serve_child,
            ),
        )
        .await;
        // Run title bookkeeping after visible output so it cannot delay delivery.
        // Interactive turns detach the work; serve children leave model title
        // generation and recovery to the parent.
        if should_run_post_turn_title(serve_child) {
            maybe_generate_session_title(
                app,
                should_generate_session_title_in_background(one_shot_mode, should_quit),
            )
            .await;
        }
    } else {
        if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
            println!("{}", format_empty_state("no response"));
        }
        app.last_turn_had_tool_calls = false;
    }

    Ok(if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    })
}

/// After a turn ends, attempt to generate a summary session title with the LLM.
/// Conditions: at least 1 user turn, and no generated title yet or the existing title is low quality.
pub(super) async fn maybe_generate_session_title(app: &App, run_in_background: bool) {
    let title_seed = session_title_seed_for_process(&app.session_id, None);
    if title_seed == SessionTitleSeed::Disabled {
        return;
    }
    let store = session_title_store(&app.config.history_file);
    // A restored session that already has a model title need not decode the whole canonical history just to check for a title.
    if store.has_generated_title(&app.session_id) {
        return;
    }
    if matches!(title_seed, SessionTitleSeed::Legacy(_))
        && !store
            .read_all_messages(&app.session_id)
            .is_ok_and(|messages| has_session_title_source(&messages))
    {
        // A brand-new session has no history; do not hold the in-flight marker, or when the user submits
        // the first input right away, the actually needed title task may be blocked by this empty one.
        return;
    }
    if !mark_session_title_generation_started(&app.session_id) {
        return;
    }

    if run_in_background {
        let task_app = app.snapshot_for_detached_helper();
        let session_id = task_app.session_id.clone();
        tokio::spawn(async move {
            generate_session_title_if_missing(&task_app, &title_seed).await;
            mark_session_title_generation_finished(&session_id);
        });
        return;
    }

    generate_session_title_if_missing(app, &title_seed).await;
    mark_session_title_generation_finished(&app.session_id);
}

/// Dispatch title generation immediately after the user submits input, without waiting for this turn's history to persist or for the model response.
pub(super) async fn maybe_generate_session_title_for_input(app: &App, user_input: &str) {
    let title_seed = session_title_seed_for_process(&app.session_id, Some(user_input));
    if title_seed == SessionTitleSeed::Disabled
        || user_input.trim().is_empty()
        || !mark_session_title_generation_started(&app.session_id)
    {
        return;
    }

    let task_app = app.snapshot_for_detached_helper();
    let session_id = task_app.session_id.clone();
    tokio::spawn(async move {
        generate_session_title_if_missing(&task_app, &title_seed).await;
        mark_session_title_generation_finished(&session_id);
    });
}

async fn generate_session_title_if_missing(app: &App, title_seed: &SessionTitleSeed) {
    let store = session_title_store(&app.config.history_file);

    let Some(all_messages) = session_title_source(title_seed, |pending_user_input| {
        let persisted_messages = match store.read_all_messages(&app.session_id) {
            Ok(messages) => messages,
            // The first submitted input can arrive before the session exists.
            Err(_) if pending_user_input.is_some() => Vec::new(),
            Err(_) => return None,
        };
        Some(session_title_messages(
            persisted_messages,
            pending_user_input,
        ))
    }) else {
        return;
    };
    if !has_session_title_source(&all_messages) {
        return;
    }

    let fallback_title = fallback_session_title(&all_messages);
    let existing = store
        .read_session_title_with_origin(&app.session_id)
        .ok()
        .flatten();
    if !should_generate_model_session_title(existing.as_ref(), &fallback_title) {
        return;
    }

    // Serve children hold the session lock until exit, so an auxiliary model call
    // here could delay the next turn after visible completion. The daemon uses
    // `generate_session_title_outside_turn` for model titles; children may write
    // only a fallback under the submitted title policy.
    let model_title = if crate::ai::background::is_serve_child() {
        None
    } else {
        crate::ai::request::generate_session_title_via_model(app, &all_messages)
            .await
            .map(|title| normalize_generated_session_title(&title))
    };

    // Log rejected model titles separately from transport failures so a retained
    // fallback does not hide which stage failed.
    if let Some(title) = model_title.as_deref()
        && (title.is_empty() || is_low_quality_session_title(title))
    {
        crate::ai::driver::decision_log::log_session_title_failure(
            crate::ai::driver::decision_log::get_decision_log_store(),
            &app.session_id,
            crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
            "low_quality_filtered",
            title,
        );
    }
    let generated_title =
        model_title.filter(|title| !title.is_empty() && !is_low_quality_session_title(title));

    // While the network request ran, another process may have written a title; re-read and overwrite only when an upgrade is still possible.
    let current = store
        .read_session_title_with_origin(&app.session_id)
        .ok()
        .flatten();
    if let Some(title) = generated_title {
        if should_generate_model_session_title(current.as_ref(), &fallback_title) {
            if store
                .write_session_title_with_origin(&app.session_id, &title, SessionTitleOrigin::Model)
                .is_ok()
            {
                crate::ai::prompt::notify_session_title_updated(&app.session_id, &title);
            }
        }
        return;
    }

    // On failure, keep a qualified existing title and write a fallback only when
    // eligible; later title-generation dispatches can retry the model.
    if !fallback_title.is_empty()
        && should_write_fallback_session_title(current.as_ref(), &fallback_title)
    {
        if store
            .write_session_title_with_origin(
                &app.session_id,
                &fallback_title,
                SessionTitleOrigin::Fallback,
            )
            .is_ok()
        {
            crate::ai::prompt::notify_session_title_updated(&app.session_id, &fallback_title);
        }
    }
}

/// Generate a served session's model title outside the child holding the turn lock.
///
/// The daemon can call this alongside the turn or after the child exits for
/// recovery. The submitted title policy determines the source. Failed or
/// rejected requests leave the existing title unchanged; returns true only
/// when a model title is written.
pub(crate) async fn generate_session_title_outside_turn(
    history_file: &std::path::Path,
    session_id: &str,
    current_model: &str,
    title_seed: &SessionTitleSeed,
) -> bool {
    generate_session_title_outside_turn_with_request(
        history_file,
        session_id,
        title_seed,
        |messages| async move {
            let Ok(config) = crate::ai::config::load_config() else {
                return None;
            };
            request_outside_turn_title(session_id, current_model, &config, &messages).await
        },
    )
    .await
}

async fn generate_session_title_outside_turn_with_request<F, Fut>(
    history_file: &std::path::Path,
    session_id: &str,
    title_seed: &SessionTitleSeed,
    request: F,
) -> bool
where
    F: FnOnce(Vec<Message>) -> Fut,
    Fut: std::future::Future<Output = Option<String>>,
{
    if title_seed == &SessionTitleSeed::Disabled {
        return false;
    }
    let store = session_title_store(history_file);
    // Settle the cheap cases before decoding the canonical history: the daemon
    // runs this after every served turn, and a title that is already
    // authoritative produces no request, so the read would be pure overhead.
    // A low-quality model title still falls through to regeneration below.
    let existing = store
        .read_session_title_with_origin(session_id)
        .ok()
        .flatten();
    if existing.as_ref().is_some_and(|title| {
        title.origin == SessionTitleOrigin::User
            || (title.origin == SessionTitleOrigin::Model
                && !is_low_quality_session_title(&title.text))
    }) {
        return false;
    }
    let Some(messages) = session_title_source(title_seed, |pending_user_input| {
        let messages = match store.read_all_messages(session_id) {
            Ok(messages) => messages,
            // A legacy client's pending prompt can precede the first session write.
            Err(_)
                if pending_user_input
                    .map(str::trim)
                    .is_some_and(|s| !s.is_empty()) =>
            {
                Vec::new()
            }
            Err(_) => return None,
        };
        // Avoid appending a pending prompt when it already matches the stored tail.
        let pending = pending_user_input
            .map(str::trim)
            .filter(|input| !input.is_empty())
            .filter(|input| {
                !messages.last().is_some_and(|m| {
                    m.role == "user" && value_to_string(&m.content).trim() == *input
                })
            });
        Some(session_title_messages(messages, pending))
    }) else {
        return false;
    };
    if !has_session_title_source(&messages) {
        return false;
    }
    let fallback_title = fallback_session_title(&messages);
    if !should_generate_model_session_title(existing.as_ref(), &fallback_title) {
        return false;
    }
    // Deduplicate title requests within this process so adjacent turns do not
    // each pay a model round-trip for the same session.
    if !mark_session_title_generation_started(session_id) {
        return false;
    }
    let generated = request(messages).await;
    mark_session_title_generation_finished(session_id);

    let Some(title) = generated else {
        return false;
    };
    let title = normalize_generated_session_title(&title);
    if title.is_empty() || is_low_quality_session_title(&title) {
        crate::ai::driver::decision_log::log_session_title_failure(
            crate::ai::driver::decision_log::get_decision_log_store(),
            session_id,
            crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
            "low_quality_filtered",
            &title,
        );
        return false;
    }
    // Re-read: a local turn or an explicit `/title` may have settled the title
    // while this request was in flight.
    let current = store
        .read_session_title_with_origin(session_id)
        .ok()
        .flatten();
    if !should_generate_model_session_title(current.as_ref(), &fallback_title) {
        return false;
    }
    store
        .write_session_title_with_origin(session_id, &title, SessionTitleOrigin::Model)
        .is_ok()
}

/// Issue the title request for [`generate_session_title_outside_turn`]; the
/// caller clears the in-flight marker after this helper returns.
async fn request_outside_turn_title(
    session_id: &str,
    current_model: &str,
    config: &crate::ai::types::AppConfig,
    messages: &[Message],
) -> Option<String> {
    let Ok(client) = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
    else {
        return None;
    };
    let ctx = crate::ai::request::AuxRequestContext {
        config,
        client: &client,
        session_id,
    };
    crate::ai::request::generate_session_title_with_context(&ctx, current_model, messages).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};

    #[tokio::test]
    async fn request_error_finalization_persists_one_row_with_optional_warning() {
        let placeholder = "[Request failed this round; please retry or rephrase]";
        let detail = "request failed: 502 upstream reset";
        for critical_premises in [false, true] {
            let root = std::env::temp_dir().join(format!("request-error-{}", uuid::Uuid::new_v4()));
            let mut app = crate::ai::middleware::test_util::test_app();
            app.config.history_file = root.join("history.sqlite");
            app.session_id = "request-error".into();
            app.cli.session = Some(app.session_id.clone());
            app.session_history_file =
                crate::ai::history::SessionStore::new(&app.config.history_file)
                    .session_history_file(&app.session_id);
            let mut messages = Vec::new();
            let mut persisted = 0;
            crate::ai::driver::runtime_ctx::SUPPRESS_TERMINAL_OUTPUT
                .scope(
                    true,
                    finalize_turn(
                        &mut app,
                        "test-model",
                        "test-model",
                        "",
                        placeholder,
                        false,
                        Some(detail),
                        None,
                        critical_premises,
                        None,
                        &mut messages,
                        true,
                        &mut persisted,
                        true,
                        false,
                    ),
                )
                .await
                .unwrap();
            let restored =
                crate::ai::history::read_all_messages_sqlite(&app.session_history_file).unwrap();
            assert_eq!(
                restored.len(),
                1,
                "a final warning must not duplicate the retry row"
            );
            assert_eq!(persisted, 1);
            let body = restored[0].content.as_str().unwrap();
            assert!(body.starts_with(placeholder));
            assert_eq!(
                body.contains(FinalGateState::CRITICAL_PREMISES_WARNING),
                critical_premises
            );
            assert_eq!(
                crate::ai::history::request_error_detail(&restored[0]),
                Some(detail)
            );
            let projected = crate::ai::request::normalize_messages_for_request_for_test(&restored);
            assert_eq!(projected[0].content, restored[0].content);
            assert!(
                projected[0].reasoning_content.is_none(),
                "display diagnostics must not reach a provider"
            );
            std::fs::remove_dir_all(root).unwrap();
        }
        assert_eq!(
            terminal_final_text_to_render(placeholder, false, None).as_deref(),
            Some(placeholder)
        );
    }

    #[test]
    fn compact_session_history_skips_in_flight_projection_without_waiting() {
        let session_id = format!("compaction-wait-{}", uuid::Uuid::new_v4());
        let guard = try_start_session_compaction(&session_id).unwrap();
        assert!(try_start_session_compaction(&session_id).is_none());

        drop(guard);
        let next_guard = try_start_session_compaction(&session_id).unwrap();
        assert!(try_start_session_compaction(&session_id).is_none());
        drop(next_guard);
        assert!(try_start_session_compaction(&session_id).is_some());
    }

    #[tokio::test]
    async fn compact_session_history_releases_cancelled_in_flight_projection() {
        let session_id = format!("compaction-cancel-{}", uuid::Uuid::new_v4());
        let guard = try_start_session_compaction(&session_id).unwrap();
        let task = tokio::spawn(async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        });
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(try_start_session_compaction(&session_id).is_some());
    }

    fn tool_call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn agent_team_tool_messages(results: &[String]) -> Vec<Message> {
        let mut messages = vec![Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(
                (0..results.len())
                    .map(|index| {
                        tool_call(
                            &format!("call-{index}"),
                            "read_file",
                            serde_json::json!({"file_path": format!("file-{index}.rs")}),
                        )
                    })
                    .collect(),
            ),
            tool_call_id: None,
            reasoning_content: None,
        }];
        messages.extend(results.iter().enumerate().map(|(index, result)| Message {
            role: "tool".to_string(),
            content: Value::String(result.clone()),
            tool_calls: None,
            tool_call_id: Some(format!("call-{index}")),
            reasoning_content: None,
        }));
        messages
    }

    #[test]
    fn complete_evidence_recovery_full_formatter_truncation_keeps_contract() {
        let messages = agent_team_tool_messages(&vec!["observed detail\n".repeat(300); 10]);
        let output = format_full_subagent_result_for_parent("final answer", &messages);
        assert!(output.contains("[evidence truncated: showing"));
        crate::ai::tools::task_tools::assert_full_evidence_recovery_hints(&output);
        assert!(output.ends_with("[Subagent final answer]\nfinal answer"));
    }

    #[test]
    fn agent_team_parent_context_evidence_bounds_and_preserves_late_errors_and_final() {
        let mut results = (0..10)
            .map(|index| format!("result-{index}\n{}", "界".repeat(1_000)))
            .collect::<Vec<_>>();
        results[9] = format!("Error: late read failed\n{}", "界".repeat(1_000));
        let messages = agent_team_tool_messages(&results);
        let final_text = "full final answer 界\n".repeat(1_000);
        let output = format_agent_team_subagent_result_for_parent(&final_text, &messages);
        let (evidence, final_answer) = output.split_once("\n[Subagent final answer]\n").unwrap();
        assert!(evidence.chars().count() <= 1_200);
        assert_eq!(
            evidence
                .lines()
                .filter(|line| line.starts_with("- "))
                .count(),
            4
        );
        assert!(evidence.contains("4/10 results; failures=1, failure excerpts shown=1"));
        assert!(evidence.contains("[failed] => Error: late read failed"));
        assert!(evidence.contains("task_evidence_read"));
        crate::ai::tools::task_tools::assert_full_evidence_recovery_hints(evidence);
        assert!(evidence.contains("source=\"delivered\""));
        assert!(evidence.contains("view=\"raw\""));
        assert!(evidence.contains("exact task_id"));
        assert!(evidence.contains("omitting fields/max_chars/cursor"));
        assert_eq!(final_answer, final_text.trim());

        for result in &mut results {
            *result = "Error: failed read".to_string();
        }
        let output =
            format_agent_team_subagent_result_for_parent("", &agent_team_tool_messages(&results));
        assert!(output.chars().count() <= 1_200);
        assert!(output.contains("failures=10, failure excerpts shown=4"));
        assert_eq!(output.matches("[failed] => Error:").count(), 4);
        assert!(!output.contains("[Subagent final answer]"));
        assert_eq!(
            format_agent_team_subagent_result_for_parent("done", &[]),
            "done"
        );
        assert_eq!(format_agent_team_subagent_result_for_parent("", &[]), "");
    }

    #[test]
    fn agent_team_parent_context_evidence_mode_preserves_normal_formatter() {
        use crate::ai::driver::runtime_ctx;
        use std::sync::Arc;

        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let context = runtime_ctx::DriverContext::new(
            crate::ai::middleware::test_util::test_app(),
            Arc::new(Mutex::new(crate::ai::mcp::McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        let messages = agent_team_tool_messages(&vec!["observed evidence".to_string(); 10]);
        runtime_ctx::DRIVER_CTX.sync_scope(context, || {
            assert_eq!(
                format_subagent_result_for_parent("done", &messages),
                format_full_subagent_result_for_parent("done", &messages)
            );
            assert_eq!(collect_subagent_tool_evidence(&messages).len(), 8);
            runtime_ctx::set_active_skill_names(vec!["agent-team".to_string()]);
            assert_eq!(
                format_subagent_result_for_parent("done", &messages),
                format_full_subagent_result_for_parent("done", &messages)
            );
            runtime_ctx::set_active_skill_names(vec!["other".to_string()]);
            assert_eq!(
                format_subagent_result_for_parent("done", &messages),
                format_full_subagent_result_for_parent("done", &messages)
            );
        });
    }

    #[tokio::test]
    async fn agent_team_parent_context_policy_uses_parent_for_kernel_child() {
        use crate::ai::driver::runtime_ctx;
        use std::sync::Arc;

        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut app = crate::ai::middleware::test_util::test_app();
        app.session_default_skills = vec!["agent-team".to_string()];
        let parent = runtime_ctx::DriverContext::new(
            app,
            Arc::new(Mutex::new(crate::ai::mcp::McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        let parent_agent_team_mode = runtime_ctx::DRIVER_CTX.sync_scope(parent.clone(), || {
            runtime_ctx::set_active_skill_names(vec!["agent-team".to_string()]);
            crate::ai::tools::task_tools::is_agent_team_mode()
        });
        let child_app = parent.app_proto.fork_for_subagent();
        assert!(child_app.session_default_skills.is_empty());
        let child = runtime_ctx::DriverContext::new(
            child_app,
            parent.mcp_client.clone(),
            parent.skill_manifests.clone(),
            parent.agent_manifests.clone(),
        );
        let messages = agent_team_tool_messages(&vec!["observed evidence".to_string(); 10]);
        runtime_ctx::DRIVER_CTX
            .scope(child, async {
                assert!(!crate::ai::tools::task_tools::is_agent_team_mode());
                let output = SUBAGENT_PARENT_AGENT_TEAM_MODE
                    .scope(parent_agent_team_mode, async {
                        tokio::task::yield_now().await;
                        subagent_result_payload_for_parent("done", &messages).unwrap()
                    })
                    .await;
                assert_eq!(
                    output,
                    format_agent_team_subagent_result_for_parent("done", &messages)
                );
                assert!(output.contains("4/10 results"));
                assert!(output.contains("task_evidence_read"));
                assert_eq!(
                    subagent_result_payload_for_parent("done", &messages).unwrap(),
                    format_full_subagent_result_for_parent("done", &messages)
                );
                runtime_ctx::set_active_skill_names(vec!["agent-team".to_string()]);
                SUBAGENT_PARENT_AGENT_TEAM_MODE
                    .scope(false, async {
                        assert_eq!(
                            subagent_result_payload_for_parent("done", &messages).unwrap(),
                            format_full_subagent_result_for_parent("done", &messages)
                        );
                    })
                    .await;
            })
            .await;
    }

    async fn assert_kernel_parent_policy_result(team_parent: bool) {
        use crate::ai::driver::runtime_ctx;
        use std::sync::Arc;

        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut app = crate::ai::middleware::test_util::test_app();
        if team_parent {
            app.session_default_skills = vec!["agent-team".to_string()];
        }
        let parent = runtime_ctx::DriverContext::new(
            app,
            Arc::new(Mutex::new(crate::ai::mcp::McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        let parent_agent_team_mode = runtime_ctx::DRIVER_CTX.sync_scope(parent.clone(), || {
            runtime_ctx::set_active_skill_names(if team_parent {
                vec!["agent-team".to_string()]
            } else {
                Vec::new()
            });
            crate::ai::tools::task_tools::is_agent_team_mode()
        });
        assert_eq!(parent_agent_team_mode, team_parent);
        let child_app = parent.app_proto.fork_for_subagent();
        assert!(
            !child_app
                .session_default_skills
                .iter()
                .any(|skill| skill == "agent-team")
        );
        let child = runtime_ctx::DriverContext::new(
            child_app,
            parent.mcp_client.clone(),
            parent.skill_manifests.clone(),
            parent.agent_manifests.clone(),
        );
        let tool_error = "Error: failed to inspect kernel fixture";
        let mut results = vec!["observed kernel evidence".to_string(); 10];
        results[0] = tool_error.to_string();
        let messages = agent_team_tool_messages(&results);
        let original_messages = serde_json::to_value(&messages).unwrap();
        let final_text = format!("  {}  ", "Complete answer 界\n".repeat(1_000));
        let result_slot = Arc::new(tokio::sync::Mutex::new(
            runtime_ctx::SubagentResult::default(),
        ));
        let runtime_error = "unchanged kernel runtime error";
        let result = runtime_ctx::DRIVER_CTX
            .scope(
                child,
                runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot.clone(), async {
                    assert!(!crate::ai::tools::task_tools::is_agent_team_mode());
                    super::super::SUBAGENT_PARENT_AGENT_TEAM_MODE
                        .scope(parent_agent_team_mode, async {
                            tokio::task::yield_now().await;
                            let payload =
                                subagent_result_payload_for_parent(&final_text, &messages).unwrap();
                            runtime_ctx::publish_subagent_result(&payload, &final_text).await;
                            Err::<(), _>(runtime_error)
                        })
                        .await
                }),
            )
            .await;
        assert_eq!(result, Err(runtime_error));
        assert!(SUBAGENT_PARENT_AGENT_TEAM_MODE.try_with(|_| ()).is_err());
        let captured = result_slot.lock().await;
        assert_eq!(captured.final_assistant_text, final_text);
        let (evidence, answer) = captured
            .parent_payload
            .split_once("[Subagent final answer]\n")
            .unwrap();
        assert_eq!(answer, final_text.trim());
        assert!(evidence.contains(tool_error));
        let expected =
            format_subagent_result_for_parent_with_policy(&final_text, &messages, team_parent);
        assert_eq!(captured.parent_payload, expected);
        if team_parent {
            assert!(evidence.chars().count() <= AGENT_TEAM_SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS);
            assert!(evidence.contains("4/10 results; failures=1, failure excerpts shown=1"));
            assert!(evidence.contains("[failed] => Error:"));
        } else {
            assert_eq!(
                captured.parent_payload,
                format_full_subagent_result_for_parent(&final_text, &messages)
            );
            assert_eq!(
                evidence
                    .lines()
                    .filter(|line| line.starts_with("- "))
                    .count(),
                8
            );
        }
        assert_eq!(serde_json::to_value(&messages).unwrap(), original_messages);
    }

    #[tokio::test]
    async fn kernel_parent_policy_team_parent_compacts_child_evidence() {
        assert_kernel_parent_policy_result(true).await;
    }

    #[tokio::test]
    async fn kernel_parent_policy_non_team_parent_preserves_full_child_evidence() {
        assert_kernel_parent_policy_result(false).await;
    }

    #[tokio::test]
    async fn kernel_parent_policy_scope_is_isolated_and_restored() {
        use std::{future::Future, task::Poll};

        assert!(SUBAGENT_PARENT_AGENT_TEAM_MODE.try_with(|_| ()).is_err());
        SUBAGENT_PARENT_AGENT_TEAM_MODE
            .scope(false, async {
                let (team, normal) = tokio::join!(
                    SUBAGENT_PARENT_AGENT_TEAM_MODE.scope(true, async {
                        tokio::task::yield_now().await;
                        SUBAGENT_PARENT_AGENT_TEAM_MODE.with(|mode| *mode)
                    }),
                    SUBAGENT_PARENT_AGENT_TEAM_MODE.scope(false, async {
                        tokio::task::yield_now().await;
                        SUBAGENT_PARENT_AGENT_TEAM_MODE.with(|mode| *mode)
                    }),
                );
                assert_eq!((team, normal), (true, false));
                assert!(!SUBAGENT_PARENT_AGENT_TEAM_MODE.with(|mode| *mode));
                {
                    let mut cancelled =
                        Box::pin(SUBAGENT_PARENT_AGENT_TEAM_MODE.scope(true, async {
                            assert!(SUBAGENT_PARENT_AGENT_TEAM_MODE.with(|mode| *mode));
                            std::future::pending::<()>().await;
                        }));
                    std::future::poll_fn(|cx| {
                        assert!(cancelled.as_mut().poll(cx).is_pending());
                        Poll::Ready(())
                    })
                    .await;
                }
                assert!(!SUBAGENT_PARENT_AGENT_TEAM_MODE.with(|mode| *mode));
                {
                    let mut panicking =
                        Box::pin(SUBAGENT_PARENT_AGENT_TEAM_MODE.scope(true, async {
                            assert!(SUBAGENT_PARENT_AGENT_TEAM_MODE.with(|mode| *mode));
                            panic!("expected kernel parent policy scope panic");
                        }));
                    std::future::poll_fn(|cx| {
                        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            panicking.as_mut().poll(cx)
                        }));
                        assert!(panic.is_err());
                        Poll::Ready(())
                    })
                    .await;
                }
                assert!(!SUBAGENT_PARENT_AGENT_TEAM_MODE.with(|mode| *mode));
            })
            .await;
        assert!(SUBAGENT_PARENT_AGENT_TEAM_MODE.try_with(|_| ()).is_err());
    }

    #[test]
    fn subagent_parent_result_includes_tool_evidence() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "call-1",
                    "read_file",
                    serde_json::json!({"file_path":"src/lib.rs","offset":10,"limit":20}),
                )]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("    10\tfn load_config() {".to_string()),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                reasoning_content: None,
            },
        ];

        let output = format_subagent_result_for_parent("done", &turn_messages);
        assert!(output.starts_with("[Subagent tool evidence]"));
        assert!(output.contains("[Subagent tool evidence]"));
        assert!(output.contains("read_file("));
        assert!(output.contains("\"file_path\":\"src/lib.rs\""));
        assert!(output.contains("fn load_config()"));
        assert!(output.contains("[Subagent final answer]\ndone"));
    }

    #[test]
    fn subagent_parent_result_without_tools_is_plain_final_text() {
        let output = format_subagent_result_for_parent("done", &[]);
        assert_eq!(output, "done");
    }

    #[test]
    fn critical_premises_pending_fallback_reaches_all_delivery_channels() {
        let warning = format!(
            "[Runtime warning] {}",
            FinalGateState::CRITICAL_PREMISES_WARNING
        );
        for original in [
            "local fallback",
            "partial repetition output",
            "truncated partial output",
        ] {
            let mut turn_messages = agent_team_tool_messages(&["observed result".to_string()]);
            let initial_len = turn_messages.len();
            let (body, recorded, suffix) = prepare_critical_premises_delivery(
                original,
                false,
                Some("unrelated old narration"),
                true,
                &mut turn_messages,
            );
            let expected = format!("{original}\n\n{warning}");
            assert_eq!(body.as_ref(), expected.as_str());
            assert!(!recorded);
            assert_eq!(suffix.as_deref(), Some(warning.as_str()));
            assert_eq!(turn_messages.len(), initial_len);
            let payload =
                subagent_result_payload_for_parent(body.as_ref(), &turn_messages).unwrap();
            assert!(payload.contains(&expected));
            for parent_agent_team_mode in [false, true] {
                let payload = format_subagent_result_for_parent_with_policy(
                    body.as_ref(),
                    &turn_messages,
                    parent_agent_team_mode,
                );
                assert!(payload.contains(&expected));
                assert!(payload.contains("observed result"));
            }
            ensure_final_assistant_recorded(body.as_ref(), recorded, &mut turn_messages);
            assert_eq!(turn_messages.len(), initial_len + 1);
            assert_eq!(turn_messages[initial_len].role, "assistant");
            assert_eq!(
                value_to_string(&turn_messages[initial_len].content),
                expected
            );
            assert_eq!(
                terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref()),
                Some(expected.clone())
            );
            let committed = serve_final_text_to_commit(body.as_ref(), suffix.as_deref()).unwrap();
            assert_eq!(committed, expected);
            assert_eq!(committed.matches(&warning).count(), 1);
        }
    }

    #[test]
    fn critical_premises_recorded_body_gets_warning_row_only() {
        let original = "already streamed answer";
        let warning = format!(
            "[Runtime warning] {}",
            FinalGateState::CRITICAL_PREMISES_WARNING
        );
        let mut turn_messages = Vec::new();
        ensure_final_assistant_recorded(original, false, &mut turn_messages);
        let (body, recorded, suffix) = prepare_critical_premises_delivery(
            original,
            true,
            Some("unrelated old narration"),
            true,
            &mut turn_messages,
        );
        let expected = format!("{original}\n\n{warning}");
        assert!(recorded);
        assert_eq!(body.as_ref(), expected.as_str());
        assert_eq!(suffix.as_deref(), Some(warning.as_str()));
        ensure_final_assistant_recorded(body.as_ref(), recorded, &mut turn_messages);
        assert_eq!(turn_messages.len(), 2);
        assert_eq!(value_to_string(&turn_messages[0].content), original);
        assert_eq!(turn_messages[1].role, "assistant");
        assert_eq!(value_to_string(&turn_messages[1].content), warning);
        assert_eq!(
            terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref()),
            Some(warning)
        );
        assert_eq!(
            serve_final_text_to_commit(body.as_ref(), suffix.as_deref()),
            Some(expected.clone())
        );
        assert_eq!(
            subagent_result_payload_for_parent(body.as_ref(), &turn_messages),
            Some(expected)
        );
    }

    #[test]
    fn critical_premises_no_pending_is_exact_noop() {
        let original = "  unchanged body \n";
        for recorded in [false, true] {
            for suffix in [None, Some(" unrelated narration candidate ")] {
                let mut turn_messages = Vec::new();
                ensure_final_assistant_recorded("existing row", false, &mut turn_messages);
                let (body, prepared_recorded, prepared_suffix) = prepare_critical_premises_delivery(
                    original,
                    recorded,
                    suffix,
                    false,
                    &mut turn_messages,
                );
                assert!(matches!(&body, Cow::Borrowed(_)));
                assert_eq!(body.as_ref(), original);
                assert_eq!(prepared_recorded, recorded);
                assert_eq!(prepared_suffix.as_deref(), suffix);
                assert_eq!(turn_messages.len(), 1);
                assert_eq!(value_to_string(&turn_messages[0].content), "existing row");
            }
        }
    }

    #[test]
    fn critical_premises_empty_is_exact_noop() {
        for original in ["", " \n\t"] {
            for recorded in [false, true] {
                for suffix in [None, Some(" unrelated narration candidate ")] {
                    let mut turn_messages = Vec::new();
                    ensure_final_assistant_recorded("existing row", false, &mut turn_messages);
                    let (body, prepared_recorded, prepared_suffix) =
                        prepare_critical_premises_delivery(
                            original,
                            recorded,
                            suffix,
                            true,
                            &mut turn_messages,
                        );
                    assert!(matches!(&body, Cow::Borrowed(_)));
                    assert_eq!(body.as_ref(), original);
                    assert_eq!(prepared_recorded, recorded);
                    assert_eq!(prepared_suffix.as_deref(), suffix);
                    assert_eq!(turn_messages.len(), 1);
                    assert_eq!(value_to_string(&turn_messages[0].content), "existing row");
                }
            }
        }
    }

    #[test]
    fn critical_premises_already_warned_fallback_reaches_visible_consumers_once() {
        let warning = format!(
            "[Runtime warning] {}",
            FinalGateState::CRITICAL_PREMISES_WARNING
        );
        let warned = format!("existing body\n\n{warning}");
        let duplicate_body = format!("{warned}\n\n{warning}");
        let duplicate_suffix = format!("{warning}\n\n{warning}");
        for original in [warned.as_str(), duplicate_body.as_str()] {
            for candidate in [
                None,
                Some(" unrelated narration candidate "),
                Some(" \n\t"),
                Some(warning.as_str()),
                Some(duplicate_suffix.as_str()),
            ] {
                let mut turn_messages = Vec::new();
                ensure_final_assistant_recorded("existing row", false, &mut turn_messages);
                let (body, recorded, suffix) = prepare_critical_premises_delivery(
                    original,
                    false,
                    candidate,
                    true,
                    &mut turn_messages,
                );
                assert!(matches!(&body, Cow::Borrowed(_)));
                assert_eq!(body.as_ref(), original);
                assert!(!recorded);
                assert_eq!(suffix.as_deref(), Some(warning.as_str()));
                assert_eq!(turn_messages.len(), 1);
                assert_eq!(value_to_string(&turn_messages[0].content), "existing row");
                for visible in [
                    terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref())
                        .unwrap(),
                    serve_final_text_to_commit(body.as_ref(), suffix.as_deref()).unwrap(),
                ] {
                    assert_eq!(visible, warned);
                    assert_eq!(visible.matches(&warning).count(), 1);
                    assert!(!visible.contains("unrelated narration candidate"));
                }
                ensure_final_assistant_recorded(body.as_ref(), recorded, &mut turn_messages);
                assert_eq!(turn_messages.len(), 2);
                assert_eq!(value_to_string(&turn_messages[0].content), "existing row");
                assert_eq!(turn_messages[1].role, "assistant");
                assert_eq!(value_to_string(&turn_messages[1].content), original);
            }
        }
    }

    #[test]
    fn critical_premises_already_warned_fallback_preserves_other_gate_suffixes() {
        let warning = format!(
            "[Runtime warning] {}",
            FinalGateState::CRITICAL_PREMISES_WARNING
        );
        let completion = "[Runtime warning] Completion is unverified.";
        let citation = "[Runtime warning] Final citations are unverified.";
        let original = format!("existing body\n\n{warning}");
        let complete_suffix = format!("{completion}\n\n{warning}\n\n{citation}");
        let duplicate_suffix = format!("{complete_suffix}\n\n{warning}");
        let expected_deduped = format!("{completion}\n\n{citation}\n\n{warning}");
        let cases = [
            (completion, format!("{completion}\n\n{warning}")),
            (complete_suffix.as_str(), complete_suffix.clone()),
            (duplicate_suffix.as_str(), expected_deduped),
        ];
        for (candidate, expected_suffix) in cases {
            let mut turn_messages = Vec::new();
            ensure_final_assistant_recorded("existing row", false, &mut turn_messages);
            let (body, recorded, suffix) = prepare_critical_premises_delivery(
                &original,
                false,
                Some(candidate),
                true,
                &mut turn_messages,
            );
            assert!(matches!(&body, Cow::Borrowed(_)));
            assert_eq!(body.as_ref(), original);
            assert!(!recorded);
            assert_eq!(suffix.as_deref(), Some(expected_suffix.as_str()));
            assert_eq!(turn_messages.len(), 1);
            assert_eq!(value_to_string(&turn_messages[0].content), "existing row");
            for visible in [
                terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref()).unwrap(),
                serve_final_text_to_commit(body.as_ref(), suffix.as_deref()).unwrap(),
            ] {
                assert_eq!(visible, format!("existing body\n\n{expected_suffix}"));
                assert_eq!(visible.matches(&warning).count(), 1);
                assert_eq!(visible.matches(completion).count(), 1);
                if candidate.contains(citation) {
                    assert_eq!(visible.matches(citation).count(), 1);
                }
            }
        }
    }

    #[test]
    fn critical_premises_already_warned_recorded_preserves_delivery_policy() {
        let warning = format!(
            "[Runtime warning] {}",
            FinalGateState::CRITICAL_PREMISES_WARNING
        );
        let completion = "[Runtime warning] Completion is unverified.";
        let notices = format!("{warning}\n\n{completion}");
        let original = format!("existing body\n\n{notices}");
        for candidate in [
            None,
            Some(" unrelated narration candidate "),
            Some(notices.as_str()),
            Some(original.as_str()),
        ] {
            let mut turn_messages = Vec::new();
            ensure_final_assistant_recorded(&original, false, &mut turn_messages);
            let (body, recorded, suffix) = prepare_critical_premises_delivery(
                &original,
                true,
                candidate,
                true,
                &mut turn_messages,
            );
            assert!(matches!(&body, Cow::Borrowed(_)));
            assert_eq!(body.as_ref(), original);
            assert!(recorded);
            assert_eq!(suffix.as_deref(), candidate);
            ensure_final_assistant_recorded(body.as_ref(), recorded, &mut turn_messages);
            assert_eq!(turn_messages.len(), 1);
            assert_eq!(turn_messages[0].role, "assistant");
            assert_eq!(value_to_string(&turn_messages[0].content), original);
            let terminal =
                terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref());
            assert_eq!(terminal, candidate.map(|text| text.trim().to_string()));
            let served = serve_final_text_to_commit(body.as_ref(), suffix.as_deref()).unwrap();
            match candidate {
                None => {
                    // No suffix does not authorize redrawing an already recorded answer.
                    assert_eq!(terminal, None);
                    assert_eq!(served, "existing body");
                }
                Some(text) if text.contains(&warning) => {
                    assert_eq!(served, original);
                    assert_eq!(served.matches(&warning).count(), 1);
                    assert_eq!(served.matches(completion).count(), 1);
                    let terminal = terminal.unwrap();
                    assert_eq!(terminal.matches(&warning).count(), 1);
                    assert_eq!(terminal.matches(completion).count(), 1);
                    if text == notices {
                        assert_eq!(terminal, notices);
                        assert!(!terminal.contains("existing body"));
                    }
                }
                Some(_) => {
                    assert_eq!(served, "existing body\n\nunrelated narration candidate");
                }
            }
        }
    }

    #[test]
    fn critical_premises_already_warned_preserves_safe_hiding() {
        let warning = format!(
            "[Runtime warning] {}",
            FinalGateState::CRITICAL_PREMISES_WARNING
        );
        for original in [
            format!("self_note:completion_evidence_required\n\n{warning}"),
            format!("[Model-authored note from an earlier turn; internal context]\n\n{warning}"),
            warning.clone(),
        ] {
            let mut turn_messages = Vec::new();
            let (body, recorded, suffix) = prepare_critical_premises_delivery(
                &original,
                false,
                None,
                true,
                &mut turn_messages,
            );
            assert!(matches!(&body, Cow::Borrowed(_)));
            assert_eq!(body.as_ref(), original);
            assert!(!recorded);
            assert!(turn_messages.is_empty());
            assert_eq!(
                terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref()),
                None
            );
            assert_eq!(
                serve_final_text_to_commit(body.as_ref(), suffix.as_deref()),
                None
            );
        }

        let original = format!(
            "Visible answer.\n\n<audit_report>{{\"findings\":[]}}</audit_report>\n\n{warning}"
        );
        let mut turn_messages = Vec::new();
        let (body, recorded, suffix) =
            prepare_critical_premises_delivery(&original, false, None, true, &mut turn_messages);
        assert_eq!(body.as_ref(), original);
        assert!(!recorded);
        assert!(turn_messages.is_empty());
        ensure_final_assistant_recorded(body.as_ref(), recorded, &mut turn_messages);
        assert_eq!(turn_messages.len(), 1);
        assert_eq!(value_to_string(&turn_messages[0].content), original);
        for visible in [
            terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref()).unwrap(),
            serve_final_text_to_commit(body.as_ref(), suffix.as_deref()).unwrap(),
        ] {
            assert!(visible.starts_with("Visible answer."));
            assert!(visible.ends_with(&warning));
            assert_eq!(visible.matches(&warning).count(), 1);
            assert!(!visible.contains("<audit_report>"));
            assert!(!visible.contains("\"findings\""));
        }
    }

    #[test]
    fn critical_premises_fallback_never_renders_unrelated_candidate() {
        let candidate = "unrelated old narration";
        assert_eq!(
            terminal_final_text_to_render("local fallback", false, Some(candidate)),
            Some("local fallback".to_string())
        );
        let mut turn_messages = Vec::new();
        let (body, recorded, suffix) = prepare_critical_premises_delivery(
            "local fallback",
            false,
            Some(candidate),
            true,
            &mut turn_messages,
        );
        for visible in [
            terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref()).unwrap(),
            serve_final_text_to_commit(body.as_ref(), suffix.as_deref()).unwrap(),
        ] {
            assert_eq!(visible, body.as_ref());
            assert!(!visible.contains(candidate));
            assert!(visible.contains(FinalGateState::CRITICAL_PREMISES_WARNING));
        }
    }

    #[test]
    fn critical_premises_warning_preserves_internal_note_and_audit_hiding() {
        for original in [
            "self_note:completion_evidence_required",
            "[Model-authored note from an earlier turn; internal context]",
            "[Runtime warning] Internal warning only.",
        ] {
            let (body, recorded, suffix) =
                prepare_critical_premises_delivery(original, false, None, true, &mut Vec::new());
            assert_eq!(
                terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref()),
                None
            );
            assert_eq!(
                serve_final_text_to_commit(body.as_ref(), suffix.as_deref()),
                None
            );
        }

        let original = "Visible answer.\n\n<audit_report>{\"findings\":[]}</audit_report>";
        let mut turn_messages = Vec::new();
        let (body, recorded, suffix) =
            prepare_critical_premises_delivery(original, false, None, true, &mut turn_messages);
        ensure_final_assistant_recorded(body.as_ref(), recorded, &mut turn_messages);
        assert_eq!(value_to_string(&turn_messages[0].content), body.as_ref());
        assert!(body.contains(original));
        for visible in [
            terminal_final_text_to_render(body.as_ref(), recorded, suffix.as_deref()).unwrap(),
            serve_final_text_to_commit(body.as_ref(), suffix.as_deref()).unwrap(),
        ] {
            assert!(visible.starts_with("Visible answer."));
            assert!(visible.ends_with(FinalGateState::CRITICAL_PREMISES_WARNING));
            assert_eq!(
                visible
                    .matches(FinalGateState::CRITICAL_PREMISES_WARNING)
                    .count(),
                1
            );
            assert!(!visible.contains("<audit_report>"));
            assert!(!visible.contains("\"findings\""));
        }
    }

    #[test]
    fn terminal_final_text_only_renders_unstreamed_visible_fallbacks() {
        assert_eq!(
            terminal_final_text_to_render("实时正文", true, None),
            None,
            "streamed model text must not be redrawn during finalize"
        );
        assert_eq!(
            terminal_final_text_to_render(
                "[Model-authored note from an earlier turn; this is not authoritative evidence.]\nself_note:completion_evidence_required",
                false,
                None,
            ),
            None
        );
        assert_eq!(
            terminal_final_text_to_render("self_note:completion_evidence_required", false, None),
            None
        );
        assert_eq!(
            terminal_final_text_to_render(
                "[Runtime warning] Completion is unverified.",
                false,
                None,
            ),
            None
        );
        assert_eq!(
            terminal_final_text_to_render("本地 fallback。", false, None),
            Some("本地 fallback。".to_string())
        );
        assert_eq!(
            terminal_final_text_to_render("完成修复。", false, None),
            Some("完成修复。".to_string())
        );
        assert_eq!(
            terminal_final_text_to_render(
                "完成修复。\n\n[Runtime warning] Completion is unverified.",
                false,
                None,
            ),
            Some("完成修复。".to_string())
        );
        assert_eq!(
            terminal_final_text_to_render(
                "已流式输出。\n\n[Runtime warning] Completion is unverified.",
                true,
                Some("[Runtime warning] Completion is unverified."),
            ),
            Some("[Runtime warning] Completion is unverified.".to_string())
        );
        // Fallback rendering must hide the audit envelope as the streaming path does.
        let unstreamed_audit = terminal_final_text_to_render(
            "审计完成。\n\n<audit_report>{\"findings\":[]}</audit_report>",
            false,
            None,
        )
        .expect("unstreamed fallback is rendered");
        assert!(unstreamed_audit.starts_with("审计完成。"));
        assert!(!unstreamed_audit.contains("<audit_report>"));
        assert!(!unstreamed_audit.contains("\"findings\""));
    }

    #[test]
    fn transactional_serve_body_commits_only_user_visible_text() {
        let warning = "[Runtime warning] Completion is unverified.";
        assert_eq!(
            serve_final_text_to_commit("accepted answer", None),
            Some("accepted answer".to_string())
        );
        assert_eq!(
            serve_final_text_to_commit(
                "accepted answer\n\n[Runtime warning] Completion is unverified.",
                Some(warning),
            ),
            Some(format!("accepted answer\n\n{warning}"))
        );
        assert_eq!(
            serve_final_text_to_commit("self_note:completion_evidence_required", None),
            None
        );
    }

    #[test]
    fn subagent_result_payload_for_parent_uses_tool_evidence_without_final_text() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "call-1",
                    "read_file",
                    serde_json::json!({"file_path":"src/lib.rs"}),
                )]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("pub mod ai;".to_string()),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                reasoning_content: None,
            },
        ];

        let output = subagent_result_payload_for_parent("", &turn_messages)
            .expect("tool evidence should still publish to parent");
        assert!(output.starts_with("[Subagent tool evidence]"));
        assert!(output.contains("read_file("));
        assert!(output.contains("pub mod ai;"));
        assert!(!output.contains("[Subagent final answer]"));
    }

    #[test]
    fn subagent_result_payload_for_parent_is_none_when_completely_empty() {
        assert!(subagent_result_payload_for_parent("", &[]).is_none());
    }

    #[test]
    fn subagent_parent_result_keeps_evidence_after_long_final_text() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "call-1",
                    "read_file",
                    serde_json::json!({"file_path":"src/lib.rs"}),
                )]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("pub mod ai;".to_string()),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                reasoning_content: None,
            },
        ];

        let long_final = "x".repeat(SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS + 100);
        let output = format_subagent_result_for_parent(long_final.as_str(), &turn_messages);
        assert!(output.starts_with("[Subagent tool evidence]"));
        assert!(output.contains("[Subagent tool evidence]"));
        assert!(output.contains("read_file("));
        assert!(output.contains("pub mod ai;"));
        assert!(output.contains("[Subagent final answer]\n"));
        assert!(output.ends_with(&long_final));
    }

    fn session_title_test_store(
        input: &str,
    ) -> (crate::ai::history::SessionStore, std::path::PathBuf, String) {
        let root = std::env::temp_dir().join(format!("a-title-policy-{}", uuid::Uuid::new_v4()));
        let history_file = root.join("history.sqlite");
        let store = session_title_store(&history_file);
        store.ensure_root_dir().unwrap();
        let session_id = uuid::Uuid::new_v4().to_string();
        crate::ai::history::append_history_messages(
            &store.session_history_file(&session_id),
            &session_title_messages(Vec::new(), Some(input)),
        )
        .unwrap();
        (store, history_file, session_id)
    }

    #[tokio::test]
    async fn session_title_disabled_seed_skips_persisted_audit_and_retry() {
        let command = "/audit inspect the current diff";
        let (store, history_file, session_id) = session_title_test_store(command);
        let seed = SessionTitleSeed::from_optional(Some(""), Some(command));
        let requests = std::sync::atomic::AtomicUsize::new(0);
        for _ in 0..2 {
            let generated = generate_session_title_outside_turn_with_request(
                &history_file,
                &session_id,
                &seed,
                |_| {
                    requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    std::future::ready(Some("Audit findings".to_string()))
                },
            )
            .await;
            assert!(!generated);
            assert_eq!(requests.load(std::sync::atomic::Ordering::Relaxed), 0);
            assert!(
                store
                    .read_session_title_with_origin(&session_id)
                    .unwrap()
                    .is_none()
            );
        }
        assert_eq!(
            value_to_string(&store.read_all_messages(&session_id).unwrap()[0].content),
            command
        );
        let _ = std::fs::remove_dir_all(history_file.parent().unwrap());
    }

    #[tokio::test]
    async fn session_title_explicit_seed_overrides_persisted_command_and_keeps_title_guards() {
        let command = "/audit inspect the current diff";
        let question = "Restore login after timeout";
        let (store, history_file, session_id) = session_title_test_store(command);
        let seed = SessionTitleSeed::from_optional(Some(question), Some(command));
        let generated = generate_session_title_outside_turn_with_request(
            &history_file,
            &session_id,
            &seed,
            |messages| {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].role, "user");
                assert_eq!(value_to_string(&messages[0].content), question);
                assert!(
                    !messages
                        .iter()
                        .any(|message| value_to_string(&message.content).contains("/audit"))
                );
                std::future::ready(Some("Login timeout recovery".to_string()))
            },
        )
        .await;
        assert!(generated);
        let title = store
            .read_session_title_with_origin(&session_id)
            .unwrap()
            .unwrap();
        assert_eq!(title.text, "Login timeout recovery");
        assert_eq!(title.origin, SessionTitleOrigin::Model);
        assert_eq!(
            value_to_string(&store.read_all_messages(&session_id).unwrap()[0].content),
            command
        );
        for origin in [SessionTitleOrigin::User, SessionTitleOrigin::Model] {
            store
                .write_session_title_with_origin(&session_id, "Keep this title", origin)
                .unwrap();
            let requests = std::sync::atomic::AtomicUsize::new(0);
            assert!(
                !generate_session_title_outside_turn_with_request(
                    &history_file,
                    &session_id,
                    &seed,
                    |_| {
                        requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        std::future::ready(Some("Replacement title".to_string()))
                    },
                )
                .await
            );
            assert_eq!(requests.load(std::sync::atomic::Ordering::Relaxed), 0);
            let title = store
                .read_session_title_with_origin(&session_id)
                .unwrap()
                .unwrap();
            assert_eq!(title.text, "Keep this title");
            assert_eq!(title.origin, origin);
        }
        let _ = std::fs::remove_dir_all(history_file.parent().unwrap());
    }

    struct ServeTitleTestEnv {
        old_seed: Option<std::ffi::OsString>,
        old_session: Option<std::ffi::OsString>,
        old_child: bool,
    }

    impl Drop for ServeTitleTestEnv {
        fn drop(&mut self) {
            // Tests hold ENV_LOCK for the full lifetime of this guard.
            unsafe {
                match &self.old_seed {
                    Some(seed) => std::env::set_var(SERVE_TITLE_SEED_ENV, seed),
                    None => std::env::remove_var(SERVE_TITLE_SEED_ENV),
                }
                match &self.old_session {
                    Some(session) => std::env::set_var(SERVE_TITLE_SESSION_ENV, session),
                    None => std::env::remove_var(SERVE_TITLE_SESSION_ENV),
                }
            }
            crate::ai::background::set_serve_child_for_test(self.old_child);
        }
    }

    #[tokio::test]
    async fn session_title_serve_child_fallback_obeys_seed_policy() {
        let _env_lock = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _env_restore = ServeTitleTestEnv {
            old_seed: std::env::var_os(SERVE_TITLE_SEED_ENV),
            old_session: std::env::var_os(SERVE_TITLE_SESSION_ENV),
            old_child: crate::ai::background::is_serve_child(),
        };
        crate::ai::background::set_serve_child_for_test(true);
        let command = "/audit inspect the current diff";
        for value in [Some(""), Some("Login recovery"), None] {
            // ENV_LOCK serializes these process-policy changes with other tests.
            unsafe {
                match value {
                    Some(seed) => std::env::set_var(SERVE_TITLE_SEED_ENV, seed),
                    None => std::env::remove_var(SERVE_TITLE_SEED_ENV),
                }
            }
            let (store, history_file, session_id) = session_title_test_store(command);
            // Bind the handoff to the served App, including when workers share its process.
            unsafe {
                std::env::set_var(SERVE_TITLE_SESSION_ENV, &session_id);
            }
            let mut app = crate::ai::middleware::test_util::test_app();
            app.config.history_file = history_file.clone();
            app.config.base_history_file = history_file.clone();
            app.session_id = session_id.clone();
            app.session_history_file = store.session_history_file(&session_id);
            let seed = session_title_seed_for_process(&session_id, Some(command));
            assert_eq!(seed, SessionTitleSeed::from_optional(value, Some(command)));
            assert_eq!(
                session_title_seed_for_process("another-session", Some("Worker request")),
                SessionTitleSeed::Legacy(Some("Worker request".to_string()))
            );
            crate::ai::background::set_serve_child_for_test(false);
            assert_eq!(
                session_title_seed_for_process(&session_id, Some(command)),
                SessionTitleSeed::Legacy(Some(command.to_string()))
            );
            crate::ai::background::set_serve_child_for_test(true);
            maybe_generate_session_title(&app, false).await;
            let title = store.read_session_title_with_origin(&session_id).unwrap();
            if value == Some("") {
                maybe_generate_session_title_for_input(&app, command).await;
                assert!(title.is_none());
                assert!(
                    store
                        .read_session_title_with_origin(&session_id)
                        .unwrap()
                        .is_none()
                );
            } else {
                let title = title.expect("child fallback");
                assert_eq!(title.origin, SessionTitleOrigin::Fallback);
                assert_eq!(
                    title.text,
                    value
                        .map(str::to_string)
                        .unwrap_or_else(|| fallback_session_title(
                            &store.read_all_messages(&session_id).unwrap()
                        ))
                );
            }
            assert_eq!(
                value_to_string(&store.read_all_messages(&session_id).unwrap()[0].content),
                command
            );
            let _ = std::fs::remove_dir_all(history_file.parent().unwrap());
        }
    }

    #[test]
    fn session_title_fallback_and_legacy_titles_remain_eligible_for_model_upgrade() {
        let fallback = "修复 session title";
        let fallback_title = SessionTitle {
            text: fallback.to_string(),
            origin: SessionTitleOrigin::Fallback,
        };
        let legacy_fallback_title = SessionTitle {
            text: fallback.to_string(),
            origin: SessionTitleOrigin::Legacy,
        };
        let model_title = SessionTitle {
            text: "完成标题生成修复".to_string(),
            origin: SessionTitleOrigin::Model,
        };
        let user_title = SessionTitle {
            text: "用户手动指定的标题".to_string(),
            origin: SessionTitleOrigin::User,
        };
        // Explicit user titles stay authoritative even when they resemble long
        // auto-generated fragments.
        let long_user_title = SessionTitle {
            text: "这是一个用户手动设置的很长标题，远远超过四十个字符的上限".to_string(),
            origin: SessionTitleOrigin::User,
        };

        assert!(should_generate_model_session_title(
            Some(&fallback_title),
            fallback
        ));
        assert!(should_generate_model_session_title(
            Some(&legacy_fallback_title),
            fallback
        ));
        assert!(!should_generate_model_session_title(
            Some(&model_title),
            fallback
        ));
        assert!(!should_generate_model_session_title(
            Some(&user_title),
            fallback
        ));
        assert!(!should_generate_model_session_title(
            Some(&long_user_title),
            fallback
        ));
        assert!(!should_write_fallback_session_title(
            Some(&long_user_title),
            fallback
        ));
    }

    #[test]
    fn session_title_generation_uses_submitted_user_input_before_persistence() {
        let messages = session_title_messages(Vec::new(), Some("  修复标题显示延迟  "));

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(value_to_string(&messages[0].content), "修复标题显示延迟");
        assert_eq!(fallback_session_title(&messages), "修复标题显示延迟");
        assert!(has_session_title_source(&messages));
        assert!(!has_session_title_source(&[]));
    }

    #[test]
    fn session_title_source_and_fallback_skip_runtime_synthetic_user_messages() {
        use crate::ai::history::runtime_synthetic_user_message;

        let synthetic = runtime_synthetic_user_message(Value::String(
            "[Subagent evidence] 子代理交接内容...".to_string(),
        ));
        // Runtime-synthesized user messages only: must not count as a real title provenance
        assert!(!has_session_title_source(&[synthetic.clone()]));

        let real = Message {
            role: "user".to_string(),
            content: Value::String("修复 session title".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        assert!(has_session_title_source(&[synthetic, real.clone()]));
        assert_eq!(fallback_session_title(&[real]), "修复 session title");
    }

    #[test]
    fn session_title_background_policy_only_backgrounds_live_interactive_turns() {
        assert!(should_generate_session_title_in_background(false, false));
        assert!(!should_generate_session_title_in_background(true, false));
        assert!(!should_generate_session_title_in_background(false, true));
        assert!(!should_generate_session_title_in_background(true, true));
    }

    #[test]
    fn persisted_history_compaction_stays_within_subagent_lifetime() {
        assert!(should_compact_session_history_in_background(
            false, false, false, false
        ));
        assert!(!should_compact_session_history_in_background(
            false, false, true, false
        ));
        assert!(!should_compact_session_history_in_background(
            true, false, false, false
        ));
        assert!(!should_compact_session_history_in_background(
            false, true, false, false
        ));
    }

    #[test]
    fn serve_child_compaction_stays_within_child_lifetime() {
        assert!(!should_compact_session_history_in_background(
            false, false, false, true
        ));
        assert!(!should_compact_session_history_in_background(
            true, false, false, true
        ));
        assert!(should_compact_session_history_in_background(
            false, false, false, false
        ));
    }

    #[test]
    fn serve_child_skips_post_turn_title_generation() {
        assert!(!should_run_post_turn_title(true));
        assert!(should_run_post_turn_title(false));
    }

    #[test]
    fn session_title_store_reads_active_session_from_base_history_file() {
        let store = session_title_store(std::path::Path::new("/tmp/a.history.sqlite"));

        assert_eq!(
            store.session_history_file("current-session"),
            std::path::Path::new("/tmp/a.history.sessions/current-session.sqlite")
        );
    }
}
