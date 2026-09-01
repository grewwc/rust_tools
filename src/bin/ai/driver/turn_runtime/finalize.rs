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
use std::sync::{LazyLock, Mutex};

use super::{TurnOutcome, persistence::persist_pending_turn_messages_for_model};

const SUBAGENT_TOOL_EVIDENCE_MAX_CALLS: usize = 8;
const SUBAGENT_TOOL_EVIDENCE_MAX_CHARS_PER_RESULT: usize = 700;
const SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS: usize = 4_000;
const MODEL_CONTEXT_ECHO_PREFIX: &str = "[Model-authored note from an earlier turn;";
static SESSION_TITLE_IN_FLIGHT: LazyLock<Mutex<FastSet<String>>> =
    LazyLock::new(|| Mutex::new(FastSet::default()));
/// In-flight dedup mirroring the title task: prevents the same session from running persisted
/// compression multiple times concurrently (the background compression dispatched at foreground
/// finalize may overlap with the next round's defensive compression in prepare).
static SESSION_COMPACTION_IN_FLIGHT: LazyLock<Mutex<FastSet<String>>> =
    LazyLock::new(|| Mutex::new(FastSet::default()));

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

/// The model occasionally echoes internal_note entries from the request projection verbatim. Runtime
/// notes intended only for the model must stay in canonical history but must not be printed to the
/// terminal as user-visible answers. For already-streamed answers, only append hints that the runtime
/// explicitly marked as user-visible are painted.
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
    Some(terminal_text.to_string())
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
    if final_assistant_text.is_empty() {
        return output;
    }
    output.push_str("\n\n");
    output.push_str("[Subagent final answer]\n");
    output.push_str(final_assistant_text);
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
) -> bool {
    // Subagent history is managed by the caller's lifecycle guard; if compression escapes the current
    // future, the guard may delete the temp SQLite first and the background task then touches a missing file.
    !is_subagent && !one_shot_mode && !should_quit
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

/// Returns true when the compression in-flight slot was acquired; false means a compression task is
/// already running for this session and the caller should skip to avoid recompressing the same history.
pub(in crate::ai::driver::turn_runtime) fn mark_session_compaction_started(
    session_id: &str,
) -> bool {
    let mut in_flight = SESSION_COMPACTION_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.insert(session_id.to_string())
}

pub(in crate::ai::driver::turn_runtime) fn mark_session_compaction_finished(session_id: &str) {
    let mut in_flight = SESSION_COMPACTION_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.remove(session_id);
}

/// Interactive finalization dispatches persisted compression to the background: the compression
/// result is only a write-back cache for the context snapshot, and the next `build_context_history`
/// always recompresses from the canonical layer, so deferred persistence does not affect correctness.
/// The foreground turn therefore does not have to wait for a CPU compression + SQLite write transaction
/// after the answer is delivered.
///
/// `at_boundary` records whether tools were called again this round (no tool calls = "answer
/// delivered", using the more aggressive threshold). After return, the spawned task owns clearing the
/// in-flight slot.
fn spawn_background_compaction(app: &App, at_boundary: bool) {
    if !mark_session_compaction_started(&app.session_id) {
        // A compression is already running (e.g. the previous round's background task has not finished);
        // skipping this round is fine — once it completes, the next prepare re-evaluates against the
        // latest canonical history.
        return;
    }
    let task_app = app.clone();
    let session_id = task_app.session_id.clone();
    // Background tasks do not own the foreground terminal: compression warnings/logs must not steal the foreground output cursor.
    tokio::spawn(
        crate::ai::driver::runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.scope(true, async move {
            let compact_result = if at_boundary {
                compact_session_history_at_boundary_with_app(
                    &task_app,
                    crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
                )
                .await
            } else {
                compact_session_history_with_app(
                    &task_app,
                    crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
                )
                .await
            };
            if let Err(err) = compact_result {
                eprintln!("[Warning] Failed to compact persisted history: {}", err);
            }
            mark_session_compaction_finished(&session_id);
        }),
    );
}

/// Persisted-compression dispatch at finalize. Foreground interactive turns go through the background
/// (so the prompt returns without blocking); one-shot runs, soon-to-exit processes, and subagents take
/// the foreground `.await` path, ensuring the snapshot is persisted before the owning history's lifetime ends.
async fn dispatch_finalize_compaction(app: &App, at_boundary: bool, run_in_background: bool) {
    if run_in_background {
        spawn_background_compaction(app, at_boundary);
        return;
    }
    let compact_result = if at_boundary {
        compact_session_history_at_boundary_with_app(
            app,
            crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
        )
        .await
    } else {
        compact_session_history_with_app(
            app,
            crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
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

fn has_session_title_source(messages: &[Message]) -> bool {
    messages.iter().any(|message| {
        message.role == "user"
            // Runtime-synthesized user messages such as subagent evidence handoffs are not real turns
            // (AGENTS.md invariant 12) and must not seed the title; otherwise multi-subagent sessions
            // like agent-team would treat the handoff content as user intent and pollute title generation.
            && !is_runtime_synthetic_user_message(message)
            && !value_to_string(&message.content).trim().is_empty()
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
        SessionTitleOrigin::Model => false,
    }
}

fn should_write_fallback_session_title(
    existing: Option<&SessionTitle>,
    fallback_title: &str,
) -> bool {
    match existing {
        None => true,
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
    user_visible_final_suffix: Option<&str>,
    active_skill_name: Option<&str>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    should_quit: bool,
    _had_tool_error: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    if let Some(subagent_output_for_parent) =
        subagent_result_payload_for_parent(final_assistant_text, turn_messages)
    {
        // Publish to the parent agent early: even without final assistant prose this round, any reusable
        // subagent evidence (e.g. read_file results) must be visible to the parent.
        // This keeps the synchronous `task` and the asynchronous `task_wait` on the same parent-side payload.
        crate::ai::driver::runtime_ctx::publish_subagent_result(
            &subagent_output_for_parent,
            final_assistant_text,
        )
        .await;
    }

    if !final_assistant_text.trim().is_empty() {
        ensure_final_assistant_recorded(
            final_assistant_text,
            final_assistant_recorded,
            turn_messages,
        );
        if crate::ai::driver::runtime_ctx::terminal_output_enabled()
            && let Some(visible_text) =
                terminal_final_text_to_render(
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
            let visible_text =
                super::output_postprocess::postprocess_terminal_text(visible_text);
            crate::ai::stream::render_markdown_block(&visible_text)?;
        }
        persist_pending_turn_messages_for_model(
            app,
            response_source_model,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
        );
        // Task boundary detection: no tool calls in this turn means the agent has delivered its answer —
        // a natural "task complete" cut point. Trigger the summary with the more aggressive threshold
        // (160 turns) instead of waiting until the conversation piles up to the hard cap (200 turns).
        let had_tool_calls = turn_messages
            .iter()
            .any(|m| m.role == "tool" || m.tool_calls.as_ref().map_or(false, |c| !c.is_empty()));
        // In goal mode, run_loop uses this flag to decide whether the goal is complete:
        // no tool calls at the end of a round = the agent delivered its final result.
        app.last_turn_had_tool_calls = had_tool_calls;
        // Persisted compression: foreground interactive turns dispatch to the background to avoid waiting
        // on CPU compression + a SQLite write transaction after the answer is delivered (the snapshot is only
        // a write-back cache; the next round recomputes from canonical).
        // One-shot runs, soon-to-exit processes, and subagent turns take the current future so the owning
        // history does not end its lifetime first.
        // `at_boundary` = no tool calls this round (answer delivered), using the more aggressive threshold.
        dispatch_finalize_compaction(
            app,
            !had_tool_calls,
            should_compact_session_history_in_background(
                one_shot_mode,
                should_quit,
                crate::ai::driver::runtime_ctx::has_subagent_result_slot(),
            ),
        )
        .await;
        // Try to generate an LLM summary title for the current conversation (when none exists and there is enough context).
        // Interactive foreground turns must not wait here on a background quality task.
        maybe_generate_session_title(
            app,
            should_generate_session_title_in_background(one_shot_mode, should_quit),
        )
        .await;
        // println!();

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
    let store = session_title_store(&app.config.history_file);
    // A restored session that already has a model title need not decode the whole canonical history just to check for a title.
    if store.has_generated_title(&app.session_id) {
        return;
    }
    if !store
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
        let task_app = app.clone();
        let session_id = task_app.session_id.clone();
        tokio::spawn(async move {
            generate_session_title_if_missing(&task_app, None).await;
            mark_session_title_generation_finished(&session_id);
        });
        return;
    }

    generate_session_title_if_missing(app, None).await;
    mark_session_title_generation_finished(&app.session_id);
}

/// Dispatch title generation immediately after the user submits input, without waiting for this turn's history to persist or for the model response.
pub(super) async fn maybe_generate_session_title_for_input(app: &App, user_input: &str) {
    if user_input.trim().is_empty() || !mark_session_title_generation_started(&app.session_id) {
        return;
    }

    let task_app = app.clone();
    let session_id = task_app.session_id.clone();
    let user_input = user_input.to_string();
    tokio::spawn(async move {
        generate_session_title_if_missing(&task_app, Some(&user_input)).await;
        mark_session_title_generation_finished(&session_id);
    });
}

async fn generate_session_title_if_missing(app: &App, pending_user_input: Option<&str>) {
    // SessionStore takes the base history file and derives `<stem>.sessions/` from it.
    // Passing the parent directory of the current session sqlite would append an extra `.sessions`
    // layer, leaving the title task unable to read the current session's messages.
    let store = session_title_store(&app.config.history_file);

    let persisted_messages = match store.read_all_messages(&app.session_id) {
        Ok(messages) => messages,
        // When triggered by the first input, the session sqlite may not have been created by the main turn yet;
        // the pending input alone is enough to generate a title, so treat "no history file yet" as empty
        // history instead of missing the entire first round.
        Err(_) if pending_user_input.is_some() => Vec::new(),
        Err(_) => return,
    };
    let all_messages = session_title_messages(persisted_messages, pending_user_input);
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

    let model_title = crate::ai::request::generate_session_title_via_model(app, &all_messages)
        .await
        .map(|title| normalize_generated_session_title(&title));

    // The model returned a title but the quality filter rejected it — a silent "request made, no result"
    // path that must be recorded in the decision log; otherwise a transport failure and a low-quality
    // fallback are indistinguishable (both surface as a long-lived fallback title).
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

    // On failure, write only the fallback and never overwrite an existing qualified title; the model is retried on the next completing round.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};

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
            false, false, false
        ));
        assert!(!should_compact_session_history_in_background(
            false, false, true
        ));
        assert!(!should_compact_session_history_in_background(
            true, false, false
        ));
        assert!(!should_compact_session_history_in_background(
            false, true, false
        ));
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
