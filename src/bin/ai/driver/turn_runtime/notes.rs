// =============================================================================
// Model-Visible Notes
// =============================================================================
// Extracted from orchestrator.rs during a logic-preserving split.
// Injectors for model-visible loop / progress / checkpoint notes and force-final reason recording.
// =============================================================================

use super::*;

pub(super) fn inject_task_anchor_note(
    messages: &mut Vec<crate::ai::history::Message>,
    question: &str,
    iteration: usize,
    reason: &str,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let goal = truncate_chars(question.trim(), TASK_ANCHOR_MAX_QUESTION_CHARS);
    let note = format!(
        "[task-anchor] reason={reason}, iteration={iteration}.\nPrimary task goal: {goal}\n\
Keep goal continuity in mind:\n- First summarize the facts confirmed so far\n- State the single next action\n- If information is insufficient, describe the blocker and stop repeating tool calls"
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// After a tool-loop detection hit, inject an internal_note into messages so the agent can
/// self-reflect (instead of force_final directly, giving the agent a chance to break the loop).
pub(super) fn inject_loop_breaker_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[loop-detected] You have called the same tool with the same arguments for the last 4 rounds; the earlier tool results are still in context, so repeating the call would produce no new information.\n\
        Do not call that same argument set again. Decide the next step from the existing evidence:\n\
        (a) If you have enough information, perform a substantive action or answer the user directly;\n\
        (b) If information is insufficient, pick only one different and concrete action (e.g. read a previously uncovered line range, search a new symbol/target, or modify a file);\n\
        (c) If you truly cannot proceed, state the single missing key piece of information and why.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

pub(super) fn inject_hard_loop_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[loop-hard-stop] Despite the repeat-call notice, you called the same tool with the same arguments for 6 consecutive rounds; this is judged an ineffective loop.\n\
        From now on you are in no-tool wrap-up mode: do not issue any more tool calls;\n\
        give a phase summary and current conclusion based on existing information; if the task is not yet complete, clearly state the gap, remaining work, and suggested next steps.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

pub(super) const TOOL_STOP_REASON_PREFIX: &str = "[runtime-tool-stop]";

/// Record the first root cause for entering no-tool wrap-up mode only in the current request context.
pub(in crate::ai::driver::turn_runtime) fn record_force_final_reason(
    messages: &mut Vec<crate::ai::history::Message>,
    reason: &str,
    iteration: usize,
    target: Option<&str>,
) {
    use crate::ai::history::{Message, ROLE_INTERNAL_NOTE};
    use serde_json::Value;

    if messages.iter().any(|message| {
        message.role == ROLE_INTERNAL_NOTE
            && message
                .content
                .as_str()
                .is_some_and(|content| content.starts_with(TOOL_STOP_REASON_PREFIX))
    }) {
        return;
    }

    // Persist to the decision log (disk JSONL): an observable channel for no-tool-handoff root
    // causes. The decision log is a session-side record that never enters the model context, so
    // there is no replay problem of an internal note being promoted to system; control state in
    // canonical turn_messages still lives only in the current request projection.
    crate::ai::driver::decision_log::log_runtime_stop(
        crate::ai::driver::decision_log::get_decision_log_store(),
        &crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
        crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
        reason,
        target,
        iteration,
    );

    let target_suffix = target.map(|t| format!(", target={t}")).unwrap_or_default();
    let event = Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!(
            "{TOOL_STOP_REASON_PREFIX} reason={reason}, iteration={iteration}, action=no_tool_handoff{target_suffix}"
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    // Runtime stop reasons belong only to the current request projection; if written to canonical
    // turn_messages, the next round would promote the stale no-tool-handoff control state back to
    // system and replay it forever.
    messages.push(event);
}

/// Approximate low-yield repetition hit: the same tool keeps hitting the same target resource
/// (only paging/search-window parameters vary).
/// Reminds the agent to judge whether these calls actually advance the problem: converge if it is
/// just fragmented paging, continue if each round serves a distinct, well-defined sub-question.
/// Soft notice; does not force convergence.
pub(super) fn inject_coarse_loop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-yield-repetition] You have been calling the same tool on the same target for several rounds, with the main variation being only paging/search-window parameters.\n\
        This often means low-yield repetition, but it is not necessarily an error: if the calls serve distinct and well-defined sub-questions, you may continue;\n\
        otherwise prefer: (a) reading a larger line range at once (raise read_file's limit) or locating with a search tool in one shot;\n\
        (b) reusing content you already read instead of re-reading the same file/segment; (c) if you already have enough information, answer directly.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Target-level repetition notice for mixed tool rounds: the same target keeps being re-examined
/// across different tool batches.
/// Same level as the coarse notice (gentle, non-blocking), but the wording highlights the specific
/// anti-pattern of "checking the same thing with a different tool", steering the model to reuse
/// what it already read instead of re-checking with yet another tool.
pub(super) fn inject_target_repeat_loop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-yield-repetition] For several rounds you have kept re-gathering evidence on the same target (the same file / the same search target),\n\
        merely switching tools or padding each round with different side calls to dodge the repetition — but you gained no new information.\n\
        Stop and do one thing: reuse what you already read/searched about that target instead of checking the same thing with another tool.\n\
        Then choose one: (a) if you have enough information, immediately take the next substantive action or answer directly;\n\
        (b) if you really must continue, write down exactly which new piece of information about that target is still missing and why switching tools would obtain it.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

    /// Soft notice for repeated "re-read from the top" rescans of the same target: the same file
    /// keeps being read from the beginning (page width changes each round, or each round mixes in a
    /// different new archive path), accumulating many full re-reads.
pub(super) fn inject_target_rescan_note(
    messages: &mut Vec<crate::ai::history::Message>,
    target: &str,
    reads: u32,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
            "[target-rescan] File `{target}` has been re-read from the beginning {reads} times within the recent window.\n\
            If you have already covered the full content, converge now and answer based on the evidence you have; if you still need more of that file, delegate the remaining exploration to a subagent with the exact file and range to inspect.\n\
            Re-reading the same range injects byte-identical content: it is suppressed/deduped and does NOT count as new progress. What you already read is still in this turn's context (or archived - see its preserved stub's `file_path`); do not re-read it from the top. Continue from the exact offset you last reached, or use `search_overflow` to locate the archived content.\n\
            If you really must re-read it yourself, write down exactly which new piece of information you expect to gain and why."
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

    /// Hard stop for "re-read from the top" rescans of the same target: full re-reads of the same
    /// file exceed the hard threshold, judged a paging + mixed-round loop, forcing no-tool wrap-up.
pub(super) fn inject_target_rescan_hard_stop_note(
    messages: &mut Vec<crate::ai::history::Message>,
    target: &str,
    reads: u32,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
            "[low-yield-hard-stop] File `{target}` has been re-read from the beginning {reads} times within recent rounds; this is judged a pagination loop.\n\
            The content you already read is still available in this turn's context or archived (see preserved stubs' `file_path`); base your conclusion on it instead of re-reading.\n\
        From now on you are in no-tool wrap-up mode: do not issue any more tool calls;\n\
        give a phase summary and current conclusion based on existing information; if the task is not yet complete, clearly state the current gap, remaining work, and suggested next steps."
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Low-yield `execute_command` coarse repetition escalated to hard-stop: many consecutive rounds
/// on the same coarse target only vary window/sort details, which is basically judged ineffective
/// exploration.
pub(super) fn inject_coarse_hard_loop_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-yield-hard-stop] You have repeatedly called `execute_command` on the same target for several rounds, varying mainly window/sort details; this is judged ineffective exploration.\n\
        From now on you are in no-tool wrap-up mode: do not issue any more tool calls;\n\
        give a phase summary and current conclusion based on existing information; if the task is not yet complete, clearly state the current gap, remaining work, and suggested next steps.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget level 1 (soft reflection): several consecutive rounds with no measurable
/// information gain.
/// For convergence-style notices while tools may still continue (soft / breadth / ledger), the
/// ledger / summary the model is asked to write is **internal self-reflection** and must go into
/// the hidden `<meta:self_note>...</meta:self_note>` channel: the stream layer
/// [`push_text_with_hidden_meta`](crate::ai::stream) strips it from visible output, persists it
/// as an internal_note, and the model still reads it next round. Without constraining the landing
/// spot, the model would write this mid-reflection into the user-facing body, get it streamed
/// immediately as a "preliminary conclusion", and duplicate the real final answer (the direct
/// cause of that incident). Hard-stop / iteration-limit force-final notices do not apply this
/// constraint -- there the body text is the final answer.
pub(super) const SELF_NOTE_REFLECTION_CHANNEL_HINT: &str = "\n\
    Important (placement constraint): the ledger / summary asked for above is internal self-reflection; write it in full \
    between `<meta:self_note>` and `</meta:self_note>`; it is not shown to the user but stays in your subsequent context.\n\
    Keep the user-facing text of this round empty or limited to the next step you are continuing with; write a real final conclusion only when you are genuinely wrapping up.";

/// Reflective notice that does not block tools -- gives the model the right to explain "why
/// continue in the same direction" and to keep exploring.
pub(super) fn inject_low_progress_soft_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[low-progress-review] The runtime recently observed no new target, success-state change, or new tool-result content.\n\
        This is a heuristic check; it does not mean the work on the same target is necessarily ineffective, and do not drop necessary steps just because of this note.\n\
        Before calling a tool, confirm which missing piece of evidence the next call would add, and what result would end this branch.\n\
        If existing evidence is enough, run the narrowest verification and answer; if not, you may continue along the clearly stated gap.\
        {SELF_NOTE_REFLECTION_CHANNEL_HINT}"
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Read-only breadth check: new targets still count as information gain; this only reminds the
/// model to consolidate first when the target surface is too broad, without blocking tools, so
/// large investigations are not misjudged as low progress.
pub(super) fn inject_read_only_breadth_note(
    messages: &mut Vec<crate::ai::history::Message>,
    agent_team_active: bool,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let team_guidance = if agent_team_active {
        "\nBecause `agent-team` is active, do not continue a broad serial sweep: delegate any remaining branches now (serial ones one at a time via the synchronous `task`, passing prior results), or state the concrete dependency that makes delegation unsafe."
    } else {
        ""
    };
    let note = format!(
        "[read-only-breadth-check] You have already covered many different target resources in read-only analysis,\n\
        which may be a necessary broad sweep, or may have slid from filling key evidence into endlessly expanding branches.\n\
        Tools remain available; but before continuing, write down in at most 6 lines:\n\
        1) confirmed facts (at most 3); 2) current conclusion or most likely explanation;\n\
        3) the single still-missing key evidence; 4) the single next tool action.\n\
        If you can already answer, give the conclusion directly instead of expanding the search surface just to re-confirm.{team_guidance} {SELF_NOTE_REFLECTION_CHANNEL_HINT}"
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget level 2 (ledger): still no progress after the soft notice; requires writing a
/// lightweight decision ledger so the model explicitly states its basis for continuing. Still
/// does not hard-block tools.
pub(super) fn inject_progress_ledger_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[low-progress-ledger] Within the response window after the previous phase check, the runtime still observed no new target,\n\
        success-state change, or new tool-result content. To continue, first write a decision ledger in at most 6 lines:\n\
        1) confirmed facts (bullets, at most 3)\n\
        2) the single key question still to resolve\n\
        3) candidate branches A / B and which you pick now, and why\n\
        4) the single next action based on the chosen branch\n\
        If the gap is clear, you may execute that action; if you cannot articulate a gap, wrap up on existing evidence.\
        {SELF_NOTE_REFLECTION_CHANNEL_HINT}"
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget level 3 (hard stop): still no progress after soft notice + ledger, switch to
/// no-tool wrap-up.
pub(super) fn inject_low_progress_hard_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-progress-hard-stop] After soft notices, response windows, and the ledger, the runtime still observed no measurable progress.\n\
        To avoid burning more budget, you are now in no-tool wrap-up mode: do not issue any more tool calls;\n\
        give a phase conclusion based on the information gathered: what has been confirmed, what is still missing,\n\
        and the suggested next step to finish the task (for change tasks, state directly which files to modify and how).";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Tiered, phase-aware tool-round checkpoint; it schedules the next step but never marks a
/// just-completed tool as failed.
pub(super) fn inject_tool_round_checkpoint_note(
    messages: &mut Vec<crate::ai::history::Message>,
    iteration: usize,
    checkpoint: ToolRoundCheckpoint,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[tool-round-checkpoint] level={} phase={} round={iteration} threshold={}.\n\
        {}\n\
        {}\n\
        Checkpoint does not change delegation rules: do not hand off the current branch due to context or iteration pressure; delegate bounded sub-steps (serial or parallel) and review their results.",
        checkpoint.level.label(),
        checkpoint.phase.recent_progress(),
        checkpoint.threshold,
        checkpoint.level.guidance(),
        checkpoint.phase.guidance(),
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Self-reflection prompt after max_iterations is hit (replaces an outright force_final surrender).
pub(super) fn inject_iteration_limit_reflect_note(
    messages: &mut Vec<crate::ai::history::Message>,
    max_iterations: usize,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[iteration-limit] You have iterated {max_iterations} rounds without converging.\n\
        Answer the user directly with the information you have. If information is insufficient, clearly tell the user where you are stuck,\
        what material is missing, and a suggested next step — do not issue any more tool calls."
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// When the synchronous wait is close to its hard timeout, ask the subagent to stop expanding new
/// branches and prioritize delivering verifiable conclusions.
pub(super) fn inject_subagent_pre_timeout_wrap_up_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;

    let note = "[subagent-pre-timeout-wrap-up] The foreground wait time for the current synchronous sub-task is about to run out.\n\
        You are now in no-tool wrap-up mode: do not issue new tool calls or expand into new audit branches.\n\
        Immediately produce a final answer based on the evidence gathered: first list the verified conclusions;\n\
        separately mark risks that are not yet verified — never guess.";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}
