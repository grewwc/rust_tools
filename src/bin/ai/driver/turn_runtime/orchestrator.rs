// =============================================================================
// Turn Orchestrator - Main Turn Execution Coordinator
// =============================================================================
// This module contains run_turn(), the main entry point for executing a single turn.
//
// Flow:
//   1. prepare_turn(): Build initial messages
//   2. Loop (max_iterations):
//        - Call LLM with current messages
//        - Execute any tool calls
//        - Handle results and add back to messages
//   3. finalize_turn(): Build final response
//   4. Return TurnOutcome (Quit, Success, or Error)
// =============================================================================

use std::io::Write;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ai::{history, mcp::SharedMcpClient, types::App};

use super::{
    CompressionReport, MID_TURN_COMPRESS_COOLDOWN_ITERATIONS, MID_TURN_COMPRESS_DELTA_THRESHOLD,
    MID_TURN_COMPRESS_SOFT_FLOOR, MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
    MID_TURN_LLM_SUMMARY_MAX_CHARS,
    finalize::finalize_turn,
    iteration::{execute_turn_iteration, refresh_skill_turn_for_iteration},
    mid_turn_compress_hard_threshold, mid_turn_compress_soft_threshold,
    persistence::persist_pending_turn_messages,
    prepare::prepare_turn,
    record_llm_summary_attempt_chars, should_try_llm_summary,
    tool_result::{
        FinalGateState, audit_evidence_gate_action, completion_evidence_state,
        completion_tool_result_succeeded, handle_iteration_execution_for_model,
        is_evidence_gated_audit_agent, tool_call_is_successful_mutation_candidate,
    },
    types::{IterationExecution, TurnLoopStep, TurnOutcome, TurnPreparation},
};

/// Tool-call loop detection windows:
/// - soft: 4 consecutive rounds with an identical (tool_name, normalized_args) pair trigger a
///   reflection prompt
/// - hard: 6 consecutive identical rounds after the soft prompt force convergence and stop the
///   tool loop
const TOOL_LOOP_SOFT_WINDOW: usize = 4;
const TOOL_LOOP_HARD_WINDOW: usize = 6;
/// Approximate low-yield repetition window: N consecutive rounds that call the same tool on the
/// same target resource (ignoring paging args such as offset/limit) count as a hit. This catches
/// real bloat that byte-exact detection misses, such as repeatedly paging through the same file or
/// re-running lookups that only tweak paging parameters. A single gentle prompt is injected first;
/// if the model keeps hammering the same coarse target for a long time (especially directory-probing
/// `execute_command` calls), it escalates to a hard stop to avoid burning hundreds of rounds.
const TOOL_LOOP_COARSE_WINDOW: usize = 5;
const TOOL_LOOP_COARSE_HARD_WINDOW: usize = 8;
const TOOL_SIGNATURE_HISTORY_LIMIT: usize = TOOL_LOOP_COARSE_HARD_WINDOW + 2;
/// Re-scan thresholds for repeated "read from the top" on the same target. Pagination loops that
/// change offset/limit/page width every round (always starting from the top of the same file) make
/// the whole-round signature never match, evading the exact/coarse/target full-round checks; here we
/// accumulate "reads started from the top" per target — the 3rd triggers a soft prompt and the 4th a
/// hard stop. write_file/apply_patch on the target clears the count (re-reading from the top after
/// an edit is legitimate). The count is neither cleared by soft prompts nor reset by made_progress.
// P1-2: two consecutive "read from the top" calls on the same file fire the soft prompt early,
// steering the model away from paging-style re-reads as soon as possible.
const TARGET_RESCAN_SOFT_THRESHOLD: u32 = 2;
const TARGET_RESCAN_HARD_THRESHOLD: u32 = 4;
/// Read-from-top count window (in rounds): when a target has not been read from the top for more
/// than this many rounds, its count expires and restarts from 1. A real paging loop re-reads the
/// same file from the top every round or every other round, so the count never expires and
/// necessarily triggers within 4 rounds; a legitimate re-read spread across many rounds after
/// context compression expires the count and never accumulates to the hard threshold.
const TARGET_RESCAN_WINDOW_ROUNDS: usize = 8;
const TASK_ANCHOR_MAX_QUESTION_CHARS: usize = 220;

/// Argument keys treated as "volatile paging/window" and stripped when computing coarse signatures.
/// After stripping, different paginations of the same file or different result caps of the same
/// lookup collapse into the same coarse signature.
const VOLATILE_ARG_KEYS: &[&str] = &["offset", "limit", "page", "cursor", "max_results"];

/// First fixed threshold for tool-round checkpoints. The default turn hard budget is 4096; the
/// 24 / 48 / 96 checkpoint tiers only schedule convergence, never disable tools, and the
/// accumulated round count is not reset by mutation.
const TOOL_ROUND_CHECKPOINT: usize = 24;
const TOOL_ROUND_CHECKPOINT_MULTIPLIERS: [usize; 3] = [1, 2, 4];

/// Retry cap for consecutive "stream-read interrupted" truncations (stream_error). Past this we
/// give up on the turn to avoid retrying forever while the server keeps dropping the stream
/// (especially for background tasks whose max_iterations = usize::MAX).
const MAX_STREAM_ERROR_RETRIES: usize = 16;
/// Retry cap for consecutive "model output too long / truncated tool-call JSON" truncations.
/// stream_error uses its own separate cap and does not count toward this one.
const MAX_MODEL_TRUNCATION_RETRIES: usize = 3;
/// Ratio threshold of reasoning to completion tokens: at or above it, the truncation is classified
/// as "reasoning ate the budget", where lowering reasoning_effort directly shortens the chain of
/// thought and frees budget for visible text; below it, the truncation is mostly caused by overly
/// long visible text, so downgrading yields nothing (the model is not reasoning) and only a shrink
/// prompt should be injected.
const REASONING_BUDGET_DOMINANCE_RATIO: f64 = 0.5;

/// === Long-loop-aware mid-turn compression ===
/// The mid-turn compression soft threshold is derived from the model's token window (flagship
/// 256K → ~135K chars). For a long-loop turn with "moderate history size but many tool iterations",
/// the history peak can stay below that threshold the whole time, so compression never fires and
/// every round re-sends the full history-so-far plus every tool schema. Total bytes sent then grow
/// O(n²) with iteration count and blow through the TPM limit within minutes (real case: a provider
/// refactor session with 56 iterations in one turn, history peak ~120K < 135K threshold, sent
/// ~2.8M tokens in-turn, exceeding the 380K TPM limit ~7x).
///
/// Mitigation: once a single turn's tool iteration count reaches this threshold we treat the turn as
/// a long loop and lower the effective mid-turn soft threshold to [`MID_TURN_COMPRESS_SOFT_FLOOR`]
/// (36K), so content-level dedup (folding byte-identical re-reads) and pruning of stale results kick
/// in early to curb the O(n²) accumulation. Short turns (below the threshold) keep the
/// window-derived ratio threshold, leaving normal single-turn large tasks their full exploration
/// space.
const LONG_LOOP_COMPRESS_ITERATION_THRESHOLD: usize = 12;

/// === Progress Budget (information-gain progress budget) ===
/// This is a third layer on top of the exact/coarse loop detectors, governing divergent loops whose
/// parameters change every round but never advance the task — the first two layers judge by
/// "signature repetition" and structurally cannot catch bloat that searches new symbols / reads new
/// files every round while converging to nothing (real case: a "delete method" change request that
/// spent 60+ consecutive rounds only reading/retrieving with zero apply_patch).
///
/// Core idea: charge by "information gain" — a behavioral signal — not by "number of actions".
/// Touching a new target resource this round (successfully reading / retrieving a new target) or
/// calling a mutation-class tool counts as progress; failed calls (no target) and repeatedly
/// re-examining the same target do not. We no longer guess task intent from the user's question
/// text. Early exploration is nearly free; the further along, the more "continuing without
/// progress" must be justified explicitly. The thing being penalized is "unexplained,
/// progress-free repetition", not exploration itself.
///
/// Free-exploration rounds: before this many rounds, even continuous no-progress never prompts
/// (locating code before deleting it, or exploring an unfamiliar codebase first, is normal).
const PROGRESS_FREE_EXPLORE_ROUNDS: usize = 20;
/// Touching many distinct targets without converging suggests the task may be sliding from
/// "gathering key evidence" into "endlessly expanding branches". This threshold injects a single
/// non-blocking breadth check only; it never counts new targets as no-progress, so legitimate
/// exploration space in large investigations is preserved.
const READ_ONLY_BREADTH_CHECK_TARGETS: usize = 32;
/// Pure read-only divergence hard-stop threshold: this many **distinct** read-only targets have been
/// touched without converging while the turn never performed any mutation. `seen_targets` is a
/// monotonic set that no reset ever clears, so the "distinct read-only targets accumulated" count is
/// fully decoupled from content novelty — it cannot be bypassed by "new target every round / new
/// bytes resetting the ordinary no-progress counter" (exactly why the ordinary soft→hard ladder
/// fails on pure read-only divergence). It is the last brake forcing convergence only when there is
/// zero mutation (pure investigation); any read+modify task is unaffected. Set at 3x
/// `READ_ONLY_BREADTH_CHECK_TARGETS` to leave large read-only investigations room while still firing
/// well before the iteration hard cap.
const READ_ONLY_BREADTH_HARD_STOP_TARGETS: usize = 96;
/// Grace window: after a soft prompt, if the model offers a materially different justification (new
/// target / changed reasoning fingerprint), escalation is paused within this window, giving it room
/// to keep exploring.
const PROGRESS_GRACE_WINDOW: usize = 6;
/// After a low-progress episode is interrupted by real progress, at least this many rounds must pass
/// before a soft prompt may be injected again, so complex tasks do not keep receiving the same
/// convergence prompt in the normal "explore → small progress → explore again" rhythm.
const PROGRESS_EPISODE_COOLDOWN: usize = 16;
/// Extra consecutive no-progress rounds required to escalate from "soft prompt / ledger" to
/// "hard-stop convergence".
const PROGRESS_NO_PROGRESS_HARD_MARGIN: usize = 16;
/// Scoped-instruction preflight uses a separate budget and does not consume normal tool iterations;
/// this cap stops the model from indefinitely extending a single turn by switching directories
/// repeatedly.
const MAX_SCOPED_PREFLIGHT_GRACE_ROUNDS: usize = 8;

#[derive(Default)]
struct ScopedPreflightTargets {
    required: Vec<std::path::PathBuf>,
}

impl ScopedPreflightTargets {
    fn required(&self) -> &[std::path::PathBuf] {
        &self.required
    }

    fn record_pause(&mut self, targets: Vec<std::path::PathBuf>) {
        self.required = targets;
    }
}


// These submodules are stored as siblings of orchestrator.rs. Note the crate also has a module with
// the same name, driver::thinking::orchestrator; on a name collision rustc resolves this module's
// submodules by "directory-style" paths (looking under an orchestrator/ directory), so #[path] must
// pin them to the sibling files explicitly, otherwise E0583 is raised.

#[path = "checkpoint.rs"]
pub(crate) mod checkpoint;
#[path = "loop_detection.rs"]
mod loop_detection;
#[path = "notes.rs"]
mod notes;
#[path = "progress.rs"]
pub(crate) mod progress;

use checkpoint::*;
use loop_detection::*;
use notes::*;
use progress::*;
pub(super) use notes::record_force_final_reason;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[derive(Default)]
struct TurnSupervisor {
    iteration: usize,
    skip_tool_signature_rounds: usize,
    loop_breaker_injected: bool,
    hard_loop_stop_injected: bool,
    coarse_loop_note_injected: bool,
    next_tool_round_checkpoint_level: usize,
    iteration_limit_note_injected: bool,
    scoped_preflight_grace_rounds: usize,
    task_anchor_injected: bool,
    last_compress_iteration: usize,
    last_compress_after_chars: usize,
/// Mid-turn state waiting to be emitted together with the next pre-request LLM compression result.
    pending_compression_report: CompressionReport,
    tool_signature_history: Vec<Vec<String>>,
    tool_signature_history_coarse: Vec<Vec<String>>,
/// History of per-round "coarse target resource" sets (same read_file file / same coarse
/// execute_command, ignoring paging args). Catches loops where whole-round signatures never match
/// but the same target is repeatedly probed inside different tool batches — plain whole-round
/// signature comparison cannot handle such mixed batches (adding one decoy tool per round evades it).
    tool_target_history: Vec<Vec<String>>,
    target_repeat_note_injected: bool,
    progress: ProgressLedger,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolLoopSignal {
    None,
/// Approximate low-yield repetition: the same tool keeps hitting the same target resource (ignoring
/// paging args). Prompt gently once.
    Coarse,
/// The same target resource is probed repeatedly across mixed tool rounds: whole-round signatures
/// all differ (each round weaves in different decoy tools), evading the exact/coarse whole-round
/// comparison, but a particular read_file target appears in every round of the window. Prompt gently
/// once.
    TargetRepeat,
/// The same target is repeatedly re-read from the top (a loop that keeps varying the paging width
/// and mixing in new archive paths each round). Prompt gently once.
    TargetRescan(String, u32),
/// The same target has been re-read from the top past the hard threshold: switch to no-tool handoff
/// and force an answer.
    TargetRescanHard(String, u32),
/// `execute_command` spins idly on the same coarse target for too long; force convergence directly.
    CoarseHard,
    Soft,
    Hard,
/// Progress Budget level one: several consecutive rounds with no information gain (no new target and
/// no substantive action) inject a reflective soft prompt without blocking tools.
    LowProgressSoft,
/// Progress Budget level two: still no progress after the soft prompt, ask for a lightweight
/// decision ledger (confirmed facts / open questions / candidate and ruled-out branches), still not
/// a hard block.
    LowProgressLedger,
/// Progress Budget level three: still no consecutive progress after the soft prompt and ledger,
/// switch to no-tool handoff mode.
    LowProgressHard,
/// Many distinct targets have been covered; remind the model to first summarize current evidence
/// and the single critical gap.
    ReadOnlyBreadth,
/// Last brake for pure read-only divergence: zero mutation in this turn, yet far more distinct
/// read-only targets than the breadth threshold have been touched without converging. Decoupled
/// from content novelty; switch to no-tool handoff and force an answer.
    ReadOnlyBreadthHard,
}

/// Runtime state for the Progress Budget, held on `TurnSupervisor`. Charged by "information gain"
/// rather than number of actions; only "unexplained, progress-free repetition" is penalized.
/// Progress is a behavioral signal: touching a new target resource this round or calling a
/// mutation-class tool (`round_has_mutation`) counts as advancing; task intent is no longer guessed
/// from the user's question text.
#[derive(Default)]
struct ProgressLedger {
/// Accumulated target resources touched ("new target = information gain" judgment).
    seen_targets: FxHashSet<String>,
/// Accumulated successful read-only tool results seen. New content counts as new evidence even
/// when it comes from the same target.
    seen_evidence_fingerprints: FxHashSet<u64>,
/// Consecutive no-progress rounds. Cleared whenever any round is judged as progress.
    consecutive_no_progress: usize,
/// Previous round's reasoning fingerprint (a fingerprint change after a soft prompt is treated as a
/// new justification and granted grace).
    last_reasoning_fp: Option<u64>,
/// Iteration until which grace lasts: no escalation before this point, giving the model room to
/// keep exploring.
    grace_until_iteration: usize,
/// A reasoning change buys at most one grace per turn, preventing endless renewal by rewriting the
/// justification every round.
    grace_consumed: bool,
    soft_injected: bool,
    ledger_injected: bool,
    hard_injected: bool,
    read_only_breadth_injected: bool,
/// Whether any project mutation happened this turn. Once true, the pure read-only divergence hard
/// stop permanently yields (read+modify tasks are not bound by that brake). plan / task lifecycle
/// counts only as progress and does not set this. Set monotonically, never cleared by reset.
    observed_project_mutation_this_turn: bool,
/// One-shot gate for the pure read-only divergence hard stop (`ReadOnlyBreadthHard` fires once).
    read_only_breadth_hard_injected: bool,
/// Earliest iteration at which a new low-progress episode may inject soft again. Substantive
/// progress resets the current episode but keeps this cooldown, preventing complex tasks from being
/// interrupted by the same prompt repeatedly.
    next_episode_iteration: usize,
/// Per-target "read from the top" re-scan counts (re-scan detection). Tuple field one is the number
/// of reads-from-top within the window; field two is the iteration of the last read-from-top (if a
/// target has not been read from the top for more than TARGET_RESCAN_WINDOW_ROUNDS rounds the count
/// expires and restarts). Decoupled from signature history: neither soft clearing the history nor a
/// new target resetting the no-progress counter affects it; any write_file/apply_patch substantive
/// change this round clears the whole table (re-reading from the top after an edit is legitimate
/// and must not be misjudged as a paging loop).
    from_top_reads: FxHashMap<String, (u32, usize)>,
/// Targets that already received a TargetRescan soft prompt (once per target per round).
    rescan_note_injected: FxHashSet<String>,
}

impl ProgressLedger {
/// Reset the escalation ladder: clear the no-progress counter and the one-shot soft/ledger/hard/grace
/// state so billing restarts from zero. Shared by two scenarios:
/// 1. Truncation retry (mark_truncation_skip): after a truncation clears history, re-reading is
///    expected behavior; keep the same semantics as mark_truncation_skip in the exact/coarse
///    detectors so a new loop after truncation recovery cannot skip the soft prompt straight to
///    hard-stop.
/// 2. Substantive progress (the made_progress branch of assess_progress): when the model responds
///    to a soft prompt with an action that genuinely advances the task, treat it as "this round's
///    nudge worked" and grant a full fresh budget instead of continuing to accumulate — otherwise,
///    in long tasks, one early divergence makes every later convergence nudge slide toward
///    hard-stop faster.
    fn reset_escalation(&mut self) {
        self.consecutive_no_progress = 0;
        self.soft_injected = false;
        self.ledger_injected = false;
        self.hard_injected = false;
        self.grace_until_iteration = 0;
        self.grace_consumed = false;
    }

/// Truncation is an external constraint; it must not inherit the previous episode's prompt cooldown.
    fn reset_after_truncation(&mut self) {
        self.reset_escalation();
        self.next_episode_iteration = 0;
// Re-reading from the top after truncation recovery is legitimate context rebuilding: the re-scan
// counts and prompt marks must be cleared together with the history, otherwise the first read-from-top
// after recovery inherits stale counts and jumps straight to hard-stop.
// Note this must not be merged into reset_escalation: the substantive-progress path must not clear
// these counts (new targets in mixed rounds keep resetting the no-progress counter; the re-scan
// counts are exactly the independent defense against that escape).
        self.from_top_reads.clear();
        self.rescan_note_injected.clear();
    }
}

impl TurnSupervisor {
    fn next_iteration(&mut self) -> usize {
        self.iteration = self.iteration.saturating_add(1);
        self.iteration
    }

    fn grant_scoped_preflight_grace(&mut self) -> bool {
        if self.scoped_preflight_grace_rounds >= MAX_SCOPED_PREFLIGHT_GRACE_ROUNDS {
            return false;
        }
        self.scoped_preflight_grace_rounds += 1;
        true
    }

    fn effective_max_iterations(&self, max_iterations: usize) -> usize {
        max_iterations.saturating_add(self.scoped_preflight_grace_rounds)
    }

    fn should_try_mid_turn_compress(&self, total_chars: usize, soft_threshold: usize) -> bool {
        let cooldown_passed = self.iteration.saturating_sub(self.last_compress_iteration)
            >= MID_TURN_COMPRESS_COOLDOWN_ITERATIONS;
        let delta_significant = total_chars.saturating_sub(self.last_compress_after_chars)
            >= MID_TURN_COMPRESS_DELTA_THRESHOLD;
        total_chars > soft_threshold
            && cooldown_passed
            && (self.last_compress_after_chars == 0 || delta_significant)
    }

/// The mid-turn compression soft threshold actually in effect this round.
///
/// For long loops (tool iterations >= [`LONG_LOOP_COMPRESS_ITERATION_THRESHOLD`]) the threshold drops
/// to [`MID_TURN_COMPRESS_SOFT_FLOOR`] so content-level dedup and pruning of stale results engage
/// early, curbing the O(n²) re-send accumulation; short turns keep the window-derived baseline
/// threshold, leaving normal single-turn large tasks unaffected. This gating and the actual
/// [`mid_turn_compress`](crate::ai::history::mid_turn_compress) call must share this method's return
/// value — the latter has a no-op early exit for `before <= soft_threshold`, and if the two
/// thresholds disagree, the gate can open while compression does nothing.
    fn effective_mid_turn_soft_threshold(&self, base_soft: usize) -> usize {
        if self.iteration >= LONG_LOOP_COMPRESS_ITERATION_THRESHOLD {
            base_soft.min(MID_TURN_COMPRESS_SOFT_FLOOR)
        } else {
            base_soft
        }
    }

    fn mark_compress(&mut self, after_chars: usize) {
        self.last_compress_iteration = self.iteration;
        self.last_compress_after_chars = after_chars;
    }

/// After substantive task progress, drop the samples of the earlier ineffective loop and restore the
/// soft → hard escalation ladder. Otherwise, when the model has already responded to the soft prompt
/// by switching to effective actions, a later single repetition would skip soft and reuse the stale
/// flags straight into hard-stop.
    fn reset_tool_loop_escalation(&mut self) {
        self.tool_signature_history.clear();
        self.tool_signature_history_coarse.clear();
        self.tool_target_history.clear();
        self.hard_loop_stop_injected = false;
        self.loop_breaker_injected = false;
        self.coarse_loop_note_injected = false;
        self.target_repeat_note_injected = false;
    }

/// Reset tool-loop detection state on truncation retry: truncation is an external constraint
/// (output cap / model availability fluctuation), so re-invoking the same tools on retry is expected
/// behavior and must not count toward the loop-detection window. Truncation already has its own
/// independent `consecutive_truncations` cap; loop detection need not add another layer.
///
/// Clearing history plus skipping the current iteration's signature recording keeps truncation
/// retries from being misjudged as a loop. All one-shot flags are reset: once truncation clears the
/// history, the full soft/coarse/hard escalation ladder should restart from zero, otherwise a new
/// loop formed after truncation recovery would skip the soft prompt straight to hard-stop.
    fn mark_truncation_skip(&mut self) {
        self.reset_tool_loop_escalation();
        self.skip_tool_signature_rounds += 1;
        self.progress.reset_after_truncation();
    }

    fn record_tool_signatures(
        &mut self,
        messages: &[crate::ai::history::Message],
        free_explore_rounds: usize,
    ) -> ToolLoopSignal {
        self.record_tool_signatures_for_progress(messages, messages, free_explore_rounds)
    }

    fn record_tool_signatures_for_progress(
        &mut self,
        messages: &[crate::ai::history::Message],
        progress_messages: &[crate::ai::history::Message],
        free_explore_rounds: usize,
    ) -> ToolLoopSignal {
        let signature_messages = if progress_messages.is_empty() {
            messages
        } else {
            progress_messages
        };
// Truncation-retry skip: after history is cleared, do not record this round's signature so
// truncation retries are not misjudged as a tool loop. `skip_tool_signature_rounds` is incremented
// by `mark_truncation_skip()` and decremented once per skipped round.
        if self.skip_tool_signature_rounds > 0 {
            self.skip_tool_signature_rounds -= 1;
            return ToolLoopSignal::None;
        }
        let Some(sigs) = extract_round_tool_signatures(signature_messages) else {
            return ToolLoopSignal::None;
        };
        self.tool_signature_history.push(sigs);
        if self.tool_signature_history.len() > TOOL_SIGNATURE_HISTORY_LIMIT {
            let drop = self.tool_signature_history.len() - TOOL_SIGNATURE_HISTORY_LIMIT;
            self.tool_signature_history.drain(0..drop);
        }
        if let Some(coarse) = extract_round_tool_signatures_coarse(signature_messages) {
            self.tool_signature_history_coarse.push(coarse);
            if self.tool_signature_history_coarse.len() > TOOL_SIGNATURE_HISTORY_LIMIT {
                let drop = self.tool_signature_history_coarse.len() - TOOL_SIGNATURE_HISTORY_LIMIT;
                self.tool_signature_history_coarse.drain(0..drop);
            }
        }
// Target-level history: maintained in parallel with coarse signatures, used by the
// target-intersection detection for mixed tool rounds. Bound by TOOL_SIGNATURE_HISTORY_LIMIT like
// exact/coarse.
        self.tool_target_history
            .push(extract_round_probe_targets(signature_messages));
        if self.tool_target_history.len() > TOOL_SIGNATURE_HISTORY_LIMIT {
            let drop = self.tool_target_history.len() - TOOL_SIGNATURE_HISTORY_LIMIT;
            self.tool_target_history.drain(0..drop);
        }
        if !self.hard_loop_stop_injected
            && detect_tool_loop(&self.tool_signature_history, TOOL_LOOP_HARD_WINDOW)
        {
            self.hard_loop_stop_injected = true;
            return ToolLoopSignal::Hard;
        }
        if !self.loop_breaker_injected
            && detect_tool_loop(&self.tool_signature_history, TOOL_LOOP_SOFT_WINDOW)
        {
            self.loop_breaker_injected = true;
            // The soft prompt has explicitly asked to stop repeated calls. Clear the samples that
            // triggered it, giving the model a full hard window to respond instead of being forced
            // to converge after only two more rounds (in the old logic soft=4 and hard=6, leaving a
            // real response window of just two rounds).
            self.tool_signature_history.clear();
            // Design-hole fix: soft only cleared the exact samples; coarse/target history was left
            // intact, and their one-shot gates (coarse_loop_note_injected /
            // target_repeat_note_injected) were permanently blocked by the `!loop_breaker_injected`
            // check below — after soft, if the model keeps paging the same target with a different
            // parameter set, coarse/target could never fire again until the iteration cap converged
            // it. Here we re-arm both gates and clear the corresponding history so paging /
            // mixed-round loops that "change posture after soft" can still be caught by later
            // coarse/target detection (the prompt stays one-shot and soft-priority, adding no extra
            // pressure).
            self.coarse_loop_note_injected = false;
            self.target_repeat_note_injected = false;
            self.tool_signature_history_coarse.clear();
            self.tool_target_history.clear();
            return ToolLoopSignal::Soft;
        }
        if !self.hard_loop_stop_injected
            && detect_execute_command_coarse_loop(
                &self.tool_signature_history_coarse,
                TOOL_LOOP_COARSE_HARD_WINDOW,
            )
        {
            self.hard_loop_stop_injected = true;
            return ToolLoopSignal::CoarseHard;
        }
        // When neither the byte-exact soft nor hard triggers, look at the coarse level: bloat from
        // repeatedly paging the same target resource or fine-tuning lookup parameters is caught
        // here. Prompt once only, and yield to exact detection.
        if !self.coarse_loop_note_injected
            && detect_tool_loop(&self.tool_signature_history_coarse, TOOL_LOOP_COARSE_WINDOW)
        {
            self.coarse_loop_note_injected = true;
            return ToolLoopSignal::Coarse;
        }
        // Whole-round signatures (exact/coarse) require the whole-round sets to be equal, so they
        // miss mixed rounds that repeatedly probe the same target across different tool batches.
        // Target intersection fills the gap here: hitting the same target every round within the
        // window fires. Yields to all whole-round detections above and, like coarse, prompts only
        // once.
        if !self.target_repeat_note_injected
            && !self.coarse_loop_note_injected
            && detect_target_repeat_loop(&self.tool_target_history, TOOL_LOOP_COARSE_WINDOW)
        {
            self.target_repeat_note_injected = true;
            return ToolLoopSignal::TargetRepeat;
        }
        // Same-target "read from the top" re-scan detection: the last line of defense against
        // paging + width-changing + mixed-round loops. The three whole-round detections above all
        // require "whole-round signature sets equal / the same target co-occurring every round",
        // while this kind of loop changes offset/limit/page width every round and mixes in new
        // archive paths and execute_command calls, so the whole-round sets are never equal; after
        // soft fires, history is cleared and each round's new target resets the no-progress
        // counter. Here we accumulate read-from-top counts per target file (cat/head/
        // tail -c +1/read_file offset=1 …), independent of whole-round signature sets — no matter
        // how the page width changes or how many new targets are mixed in, the count is unaffected.
        // write_file/apply_patch on that file clears it (re-reading from the top after an edit is
        // legitimate). One soft prompt fires, then counting continues; the 4th read-from-top
        // hard-stops.
        // Substantive file changes this round (write_file/apply_patch, any target) → post-edit
        // verifying re-reads are legitimate, so clear the whole rescan table to keep multi-file
        // "read A, modify B" workflows from being misjudged as a paging loop. Note
        // round_has_mutation cannot be used: it treats non-read-only execute_command like sed/awk
        // as mutation, which would misjudge a sed read-only re-read as a "substantive change" and
        // neutralize rescan detection (command_reads_from_top, in contrast, treats `sed -n 1` as a
        // read-from-top).
        if !extract_round_mutated_targets(signature_messages).is_empty() {
            self.progress.from_top_reads.clear();
            self.progress.rescan_note_injected.clear();
        }
        let from_top_targets = extract_round_from_top_read_targets(signature_messages);
        let mut rescan_signal: Option<ToolLoopSignal> = None;
        for target in from_top_targets {
            let entry = self
                .progress
                .from_top_reads
                .entry(target.clone())
                .or_insert((0, self.iteration));
            // Window expiry: if more than WINDOW rounds have passed since the last read-from-top,
            // the old count is void and counting restarts.
            if self.iteration.saturating_sub(entry.1) > TARGET_RESCAN_WINDOW_ROUNDS {
                *entry = (0, self.iteration);
                // Decay = entering a new re-read episode: clear the soft-prompt mark for that
                // target in sync so every episode gets its own soft warning. Otherwise
                // rescan_note_injected keeps the previous episode's mark; when the second episode
                // accumulates to soft, insert returns false → skipping the soft prompt straight to
                // hard-stop, contradicting the soft→hard escalation invariant (the
                // mutation/truncation clear paths always clear the pair together).
                self.progress.rescan_note_injected.remove(&target);
            }
            entry.0 += 1;
            entry.1 = self.iteration;
            let reads = entry.0;
            if reads >= TARGET_RESCAN_HARD_THRESHOLD {
                rescan_signal = Some(ToolLoopSignal::TargetRescanHard(target, reads));
                break;
            }
            if reads >= TARGET_RESCAN_SOFT_THRESHOLD
                && self.progress.rescan_note_injected.insert(target.clone())
            {
                rescan_signal = Some(ToolLoopSignal::TargetRescan(target, reads));
            }
        }
        if let Some(signal) = rescan_signal {
            return signal;
        }
        // When neither exact nor coarse catches a "signature repetition" loop, hand off to the
        // Progress Budget: it catches divergent loops whose parameters change every round but never
        // advance the task.
        self.assess_progress(messages, progress_messages, free_explore_rounds)
    }

    /// Progress Budget judgment: charged by "information gain" rather than number of actions.
    /// Called only as a fallback when the exact/coarse signature detection misses. Progress is a
    /// pure behavioral signal; intent is no longer guessed from question text:
    ///
    /// - Touching a new target resource this round (`extract_round_targets` first occurrence) →
    ///   information gain, counts as progress;
    /// - A successful read-only tool returning content not seen before → new evidence, counts as
    ///   progress;
    /// - Or calling a mutation-class tool this round (`round_has_mutation`) → substantive action,
    ///   counts as progress.
    ///
    /// Within the free-exploration zone (iteration <= free_explore_rounds) nothing is charged at
    /// all; beyond it, escalation follows stable thresholds: soft prompt → fixed response window →
    /// ledger → hard stop. There is also an in-turn cooldown between soft episodes so complex tasks
    /// do not keep receiving the same prompt in their normal explore/advance rhythm.
    fn assess_progress(
        &mut self,
        messages: &[crate::ai::history::Message],
        progress_messages: &[crate::ai::history::Message],
        free_explore_rounds: usize,
    ) -> ToolLoopSignal {
        // Keep the three kinds of progress signals separate: ReadOnlyBreadth is triggered only by
        // new targets; call-pattern history is cleared only after a new target / mutation action.
        // When the same target returns new content, only the Progress Budget is reset, preserving
        // the exact/coarse detectors' one-shot ability to warn about repeated paging patterns.
        let round_had_mutation = round_has_mutation(progress_messages);
        let round_had_project_mutation = round_has_project_mutation(progress_messages);
        let mut added_new_target = false;
        for t in extract_round_targets(progress_messages) {
            if self.progress.seen_targets.insert(t) {
                added_new_target = true;
            }
        }
        let mut added_new_evidence = false;
        for fingerprint in extract_round_evidence_fingerprints(progress_messages) {
            if self.progress.seen_evidence_fingerprints.insert(fingerprint) {
                added_new_evidence = true;
            }
        }
        let made_progress = round_had_mutation || added_new_target || added_new_evidence;

        let reasoning_fp = extract_round_reasoning_fingerprint(progress_messages)
            .or_else(|| extract_round_reasoning_fingerprint(messages));
        if made_progress {
        // Any real progress ends the current low-progress episode; next_episode_iteration is
        // preserved so a brief advance does not immediately re-inject soft. Seen targets/evidence
        // also accumulate across rounds.
            let recovered_from_low_progress_episode = self.progress.soft_injected
                || self.progress.ledger_injected
                || self.progress.hard_injected
                || self.progress.grace_consumed
                || self.progress.grace_until_iteration > 0;
            if recovered_from_low_progress_episode {
                self.progress.next_episode_iteration = self
                    .progress
                    .next_episode_iteration
                    .max(self.iteration + PROGRESS_EPISODE_COOLDOWN);
                self.progress.reset_escalation();
            }
        // Only **structural** progress (new target / mutation action) with a previously injected
        // loop prompt clears the exact/coarse/target retry windows. New read-only result content
        // already counts toward made_progress above (the Progress Budget does not escalate on
        // efficient paging), but signature / target history is **never** cleared: otherwise
        // repeatedly paging the same file would never fill the coarse window because every page
        // differs, bypassing the loop brake — exactly what the "progress hashes must ignore
        // offset/limit to prevent budget escape" invariant guards against.
            if (round_had_mutation || added_new_target)
                && (self.hard_loop_stop_injected
                    || self.loop_breaker_injected
                    || self.coarse_loop_note_injected
                    || self.target_repeat_note_injected)
            {
                self.reset_tool_loop_escalation();
            }
            self.progress.consecutive_no_progress = 0;
            self.progress.last_reasoning_fp = reasoning_fp;
        // Only a real project change permanently disables the pure read-only divergence hard stop.
        // plan / task lifecycle is task progress, but it must not let later unbounded serial scans
        // bypass this last brake.
            if round_had_project_mutation {
                self.progress.observed_project_mutation_this_turn = true;
            }
        // Last brake for pure read-only divergence: the project was never modified this turn, yet
        // far more distinct read-only targets than the breadth threshold have been touched without
        // converging. `seen_targets` is monotonic and never cleared by any reset, so this criterion
        // is fully decoupled from "new target / new bytes resetting the ordinary no-progress
        // counter" — it is the only signal that can catch "pure-investigation divergence that keeps
        // switching targets". Yielding after the ordinary breadth prompt, it fires once and forces
        // convergence.
            if !self.progress.read_only_breadth_hard_injected
                && !self.progress.observed_project_mutation_this_turn
                && self.iteration > free_explore_rounds
                && self.progress.seen_targets.len() >= READ_ONLY_BREADTH_HARD_STOP_TARGETS
            {
                self.progress.read_only_breadth_hard_injected = true;
                return ToolLoopSignal::ReadOnlyBreadthHard;
            }
            if !self.progress.read_only_breadth_injected
                && !round_had_mutation
                && added_new_target
                && self.iteration > free_explore_rounds
                && self.progress.seen_targets.len() >= READ_ONLY_BREADTH_CHECK_TARGETS
            {
                self.progress.read_only_breadth_injected = true;
                return ToolLoopSignal::ReadOnlyBreadth;
            }
            return ToolLoopSignal::None;
        }

            // Free-exploration zone: exploration is completely free — no charging, no escalation
            // (locating code before deleting it, or feeling out an unfamiliar codebase, is normal).
            // Only the reasoning fingerprint baseline is updated.
        if self.iteration <= free_explore_rounds {
            self.progress.last_reasoning_fp = reasoning_fp;
            return ToolLoopSignal::None;
        }

        self.progress.consecutive_no_progress += 1;

            // Every soft prompt is followed by a fixed response window. In the old logic only
            // models exposing reasoning_content with a changed fingerprint got grace; models that
            // do not emit reasoning would receive the ledger in the very next round. After the base
            // window ends, a new justification can extend it once more, but rolling renewal is not
            // allowed.
        let reasoning_changed =
            reasoning_fp.is_some() && reasoning_fp != self.progress.last_reasoning_fp;
        self.progress.last_reasoning_fp = reasoning_fp;
        if self.iteration < self.progress.grace_until_iteration {
            return ToolLoopSignal::None;
        }
        if self.progress.soft_injected && reasoning_changed && !self.progress.grace_consumed {
            self.progress.grace_until_iteration = self.iteration + PROGRESS_GRACE_WINDOW;
            self.progress.grace_consumed = true;
            return ToolLoopSignal::None;
        }

        let soft_threshold = no_progress_soft_threshold(self.iteration, free_explore_rounds);
        if self.progress.consecutive_no_progress < soft_threshold {
            return ToolLoopSignal::None;
        }

            // The escalation ladder proceeds strictly soft prompt → ledger → hard stop, each level
            // one-shot. Hard stop additionally requires consecutive no-progress to reach
            // soft_threshold + margin, so it cannot jump past the soft layers straight to
            // convergence.
        if !self.progress.soft_injected {
            if self.iteration < self.progress.next_episode_iteration {
                return ToolLoopSignal::None;
            }
            self.progress.soft_injected = true;
            self.progress.grace_until_iteration = self.iteration + PROGRESS_GRACE_WINDOW;
            self.progress.next_episode_iteration = self.iteration + PROGRESS_EPISODE_COOLDOWN;
            return ToolLoopSignal::LowProgressSoft;
        }
        if !self.progress.ledger_injected {
            self.progress.ledger_injected = true;
            return ToolLoopSignal::LowProgressLedger;
        }
        let hard_threshold = soft_threshold + PROGRESS_NO_PROGRESS_HARD_MARGIN;
        if self.progress.consecutive_no_progress >= hard_threshold && !self.progress.hard_injected {
            self.progress.hard_injected = true;
            return ToolLoopSignal::LowProgressHard;
        }
        ToolLoopSignal::None
    }

    /// Tiered tool-round checkpoints: keep the cumulative round count and inject different
    /// scheduling prompts at 24 / 48 / 96 (scaled by the first tier for small budgets) depending on
    /// the current work phase.
    fn maybe_inject_tool_round_checkpoint(
        &mut self,
        messages: &mut Vec<crate::ai::history::Message>,
        max_iterations: usize,
        phase: ToolRoundCheckpointPhase,
    ) -> Option<ToolRoundCheckpoint> {
        let level_index = self.next_tool_round_checkpoint_level;
        let threshold = tool_round_checkpoint_threshold(max_iterations, level_index)?;
        if self.iteration < threshold {
            return None;
        }
        self.next_tool_round_checkpoint_level += 1;
        let checkpoint = ToolRoundCheckpoint {
            level: ToolRoundCheckpointLevel::from_index(level_index),
            phase,
            threshold,
        };
        inject_tool_round_checkpoint_note(messages, self.iteration, checkpoint);
        Some(checkpoint)
    }

    fn maybe_inject_iteration_limit_note(
        &mut self,
        messages: &mut Vec<crate::ai::history::Message>,
        max_iterations: usize,
        force_final_response: bool,
    ) {
        if force_final_response && !self.iteration_limit_note_injected {
            self.iteration_limit_note_injected = true;
            inject_iteration_limit_reflect_note(messages, max_iterations);
        }
    }

    fn maybe_inject_task_anchor(
        &mut self,
        messages: &mut Vec<crate::ai::history::Message>,
        question: &str,
        reason: &str,
    ) {
        if self.task_anchor_injected {
            return;
        }
        self.task_anchor_injected = true;
        inject_task_anchor_note(messages, question, self.iteration, reason);
    }
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "A",
    "turn_runtime::run_turn",
    "[DEBUG] run_turn started",
    "[DEBUG] run_turn finished",
    {
        "history_count": history_count,
        "question_len": question.chars().count(),
        "model": next_model.as_str(),
        "one_shot_mode": one_shot_mode,
        "should_quit": should_quit,
    },
    {
        "ok": __agent_hang_result.is_ok(),
        "outcome": __agent_hang_result
            .as_ref()
            .map(|v| format!("{:?}", v))
            .unwrap_or_else(|err| err.to_string()),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(in crate::ai::driver) async fn run_turn(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    question: String,
    attachments_text: String,
    next_model: String,
    precomputed_ocr: Option<crate::ai::driver::model::OcrExtraction>,
    one_shot_mode: bool,
    should_quit: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
        // Step 3: turn-start hook (preserving the original pairing semantics; must fire before the
        // /audit shortcut). Allocate the real turn identity first (atomic increment in the session
        // SQLite), then fire the start hook, so on_turn_start / on_turn_end read the real
        // turn_index rather than a fake 0. Allocation failure = this round never started: return
        // directly without firing any lifecycle hook (the start/end pairing is preserved).
    let turn_index = history::reserve_turn_index(&app.session_history_file)?;
    app.fire_turn_start_hooks(turn_index);

    let result = (async {
        // `/audit` is a synchronous subagent call directly requested by the user. It must be handled
        // after the parent DRIVER_CTX is established but before the child agent enters a recursive
        // turn, so that the task's isolation and evidence lifecycle can be reused.
        if crate::ai::driver::runtime_ctx::current_subagent_depth() == 0 {
            if let Some(command) = crate::ai::driver::commands::audit::parse_audit_command(&question) {
                return Ok(execute_audit_command(app, command, should_quit));
            }
        }
        // Inject (session_id, turn_id) into task-local storage so downstream tool calls and
        // feedback write paths see the correct identity. turn_id is allocated atomically by the
        // session SQLite and covers normal, resume, and internal turns; it never repeats across
        // restarts or processes.
        let session_id = app.session_id.clone();
        let turn_id = turn_index;
        // Only the foreground main turn raises the "turn active" flag: child agents (sync /
        // background) hold their own private signal flags and always run under a
        // SUBAGENT_RESULT_SLOT scope, so they are excluded here. This flag makes Ctrl+C during
        // streaming=false gaps — prepare / thinking / phase switches / mid-turn compression —
        // cancel the current round instead of quitting the session. The guard is dropped
        // automatically when this future drops.
        let _foreground_turn_guard = (!crate::ai::driver::runtime_ctx::has_subagent_result_slot())
            .then(crate::ai::driver::signal::ForegroundTurnGuard::enter);
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .scope((session_id, turn_id), async {
        // enable_tools' per-turn state must track the entire future rather than relying only on
        // run_turn_body's happy-path tail cleanup; abort / early returns also Drop it.
                let _enable_turn_guard = crate::ai::tools::enable_tools::EnableTurnStateGuard::enter();
        // Keep same-terminal side-note input listening (Ctrl+G) active for the whole turn
        // (prepare / streaming / thinking / tool execution / finalize). The RAII guard drops when
        // this future ends, restoring the canonical terminal so prompt_user input boxes between
        // turns are not affected by leftover cbreak. Sub-agents (depth>0) / background (terminal
        // output suppressed) / non-tty stdin are excluded by side_note_input_enabled(); no other
        // stdin consumer exists inside a turn (the input box only opens between turns, and
        // request_user_input only sets a flag without reading stdin), so cbreak owning stdin for
        // the whole duration has no conflicts.
                let _side_note_input =
                    if crate::ai::stream::side_note_input::side_note_input_enabled() {
                        Some(crate::ai::stream::side_note_input::SideNoteInputGuard::spawn(
                            app.session_history_file.clone(),
                        ))
                    } else {
                        None
                    };
                run_turn_body(
                    app,
                    mcp_client,
                    skill_manifests,
                    history_count,
                    turn_index,
                    question,
                    attachments_text,
                    next_model,
                    precomputed_ocr,
                    one_shot_mode,
                    should_quit,
                )
                .await
            })
            .await
    })
    .await;

        // Step 3: turn-end hook. Covers every return path (including the /audit shortcut), paired
        // with the start hook.
    app.fire_turn_end_hooks(turn_index);

    result
}

fn execute_audit_command(
    app: &App,
    command: crate::ai::driver::commands::audit::AuditCommand,
    should_quit: bool,
) -> TurnOutcome {
    match command {
        crate::ai::driver::commands::audit::AuditCommand::Usage => {
            println!("Usage: /audit [--fast] <instruction>");
            println!("  --fast  快速审计：当前会话模型 + high 思考 + 更少步数 + 更短超时，适合轻量复查");
        }
        crate::ai::driver::commands::audit::AuditCommand::Run { instruction, fast } => {
            // By default only cwd/skills are inherited, keeping unrelated parent conversation and
            // memory out of the audit task. But the subagent cannot see the parent conversation at
            // all, so it must be told explicitly what the main agent has changed: several
            // requirements are often modified in parallel, and only by seeing the current workspace
            // diff can the subagent judge which changes belong to this audit.
            let prompt = crate::ai::driver::commands::audit::build_audit_prompt(&instruction);
            let description = if fast {
                format!("/audit --fast {instruction}")
            } else {
                format!("/audit {instruction}")
            };
            // fast mode: the audit-fast agent (fewer steps) carries the lightweight audit contract;
            // the model is pinned to the current session model (bypassing the automatic tier bump
            // from prompt-difficulty classification) and thinking is pinned to high; the timeout
            // shrinks accordingly.
            let agent_name = if fast { "audit-fast" } else { "audit" };
            let (hard_timeout, wrap_up_lead_time) = if fast {
                (
                    crate::ai::driver::commands::audit::FAST_AUDIT_SUBAGENT_HARD_TIMEOUT,
                    Some(crate::ai::driver::commands::audit::FAST_AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME),
                )
            } else {
                (
                    crate::ai::driver::commands::audit::AUDIT_SUBAGENT_HARD_TIMEOUT,
                    Some(crate::ai::driver::commands::audit::AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME),
                )
            };
            let mut args = serde_json::json!({
                "description": description,
                "prompt": prompt,
                "agent": agent_name,
            });
            if fast {
                args["model"] = serde_json::Value::String(app.current_model.clone());
                args["reasoning_effort"] = serde_json::Value::String("high".to_string());
            }
            match crate::ai::driver::tools::execute_direct_subagent_task(
                "slash-audit",
                &args,
                hard_timeout,
                wrap_up_lead_time,
            ) {
                Ok(result) => println!(
                    "\n{}",
                    crate::ai::driver::commands::audit::terminal_audit_result(&result.content)
                ),
                Err(error) => println!("\n[audit] Unable to start audit subagent: {error}"),
            }
        }
    }

    if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_turn_body(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    turn_index: usize,
    question: String,
    attachments_text: String,
    next_model: String,
    precomputed_ocr: Option<crate::ai::driver::model::OcrExtraction>,
    one_shot_mode: bool,
    should_quit: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    // Clear the previous round's interrupt flag at the start of every round so it
    // reflects only whether THIS round was interrupted by Ctrl+C.
    app.last_turn_interrupted = false;
    // Requesting user input is a turn-level side channel from the tool layer to the
    // driver; clear stale state first so an abnormally-exited previous round does not
    // carry its pending continuation into the current turn.
    crate::ai::tools::skill_tools::clear_pending_user_input_request();
    // The reasoning-items side channel is turn-scoped in-memory state: clear it at the
    // start of each round so the previous round's encrypted reasoning does not leak
    // into this round's request (its call_id would no longer match either).
    app.turn_reasoning_items.clear();
    let TurnPreparation {
        mut skill_turn,
        mut messages,
        mut turn_messages,
        mut persisted_turn_messages,
        max_iterations,
    } = match prepare_turn(
        app,
        mcp_client,
        skill_manifests,
        history_count,
        turn_index,
        &question,
        &attachments_text,
        &next_model,
        precomputed_ocr,
    )
    .await
    {
        Ok(prep) => prep,
        Err(err) => return Err(err),
    };

    persist_pending_turn_messages(
        app,
        one_shot_mode,
        &turn_messages,
        &mut persisted_turn_messages,
    );

    // History-budget compression is owned by prepare_turn (it builds the projection with
    // summaries / overflow pointers according to history_max_chars); the mid-turn budget
    // path handles the fallback for very long rounds. Do not attach a second pipeline
    // pass here — whether really compressing or observe-only, it would add clone+infer
    // cost every turn, and a real second pass would drop unrecoverable context. The
    // pipeline module stays as optional test/observation infrastructure; callers mount it
    // explicitly per stage when needed.

    let mut supervisor = TurnSupervisor::default();
    let mut force_final_response = false;
    let mut pre_timeout_wrap_up_requested = false;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut final_response_model = None::<String>;
    let mut terminal_dedupe_candidate = None;
    let mut final_gate_state = FinalGateState::default();
    // Collect the names of explicit-enabled tools actually called this turn; used to age
    // out unused entries at the end of the turn.
    let mut tools_used_this_turn: rust_tools::cw::SkipSet<String> =
        rust_tools::cw::SkipSet::default();
    let mut consecutive_empty_responses: usize = 0;
    let mut consecutive_truncations: usize = 0;
    // Count stream-read-interruption truncations (stream_error) separately. They are
    // unrelated to overlong model output (network jitter / server-side stream drop), so
    // they neither participate in reasoning downgrades nor accumulate
    // consecutive_truncations; still, sustained stream drops need a bounded fallback, or
    // a background task with a usize::MAX iteration budget would retry forever.
    let mut consecutive_stream_errors: usize = 0;
    let mut turn_had_tool_error = false;
    // Save the reasoning-effort override in effect when this turn starts (either the
    // user's explicit `/model effort` choice, or None = model default). Truncation
    // retries temporarily lower it to Low to give the output-token budget to actual
    // content; restore it uniformly at turn end (including all break exits) so the
    // user's session-level setting is never polluted.
    let saved_effort_override = app.cli.reasoning_effort_override;
    // Likewise save the thinking fallback switch: truncation retries may set it to force
    // the thinking chain of always-thinking models off; restore it at turn end so later
    // turns are unaffected.
    let saved_thinking_disabled = app.cli.thinking_disabled_override;
    // Similarly save the adaptive max_tokens override: on a zero-output truncation we
    // automatically lower max_tokens and retry. The downgrade is temporary — restore the
    // original value as soon as there is normal output (normal completion or normal
    // truncation), because the original value is itself reasonable (the first request
    // succeeded). Also restored as a safety net at turn end.
    let saved_max_tokens_override = app.cli.max_tokens_override;
    // Whether we are currently in the zero-output downgraded state.
    let mut mt_downgraded = false;
    // VL image-digest (429 TPM mitigation) state: pending_digest_source holds the
    // untruncated raw text (including reasoning) of the previous round's tool-call
    // response, used as the source to piggyback the digest parse; once
    // image_digest_resolved is set, the image has either been replaced by a digest or,
    // when neither path could obtain one, kept as the user decided — in both cases we
    // stop retrying every round instead of repeatedly firing fallback requests.
    let mut pending_digest_source: Option<String> = None;
    let mut image_digest_resolved = false;
    // Cross-turn image digest: record this round's resolved digest (content fingerprint,
    // digest, original image paths); persist it into the history metadata table on
    // normal turn end, and on the next turn load history with the same fingerprint to
    // retrieve the digest and replace the old image, avoiding resending the previous
    // turn's image in the new round.
    let mut turn_digest: Option<(String, String, Vec<String>)> = None;
    // Mutation targets rejected by preflight must get prioritized scoped-instruction
    // budget at every subsequent prompt rebuild this turn; it cannot be consumed once,
    // because intermediate reads and mid-turn compression may make the same mutation
    // target disappear from observable history and get re-paused.
    let mut scoped_preflight_targets = ScopedPreflightTargets::default();
    let loop_result = 'turn: loop {
        let iteration = supervisor.next_iteration();
        let effective_max_iterations = supervisor.effective_max_iterations(max_iterations);
        // From round two onward, replace the images inlined in user messages of the
        // request projection with the textual digest produced in the previous round, to
        // avoid replaying base64 in the tool loop and tripping Doubao/Ark-side 429 TPM
        // rate limits. Prefer piggybacking the previous response to parse the digest;
        // fall back to a single one-off VL request with tools disabled; if both fail,
        // keep the original image. Only the request-projection messages change; the
        // canonical turn_messages are untouched. Placed before the mcp-lock block: the
        // fallback request contains .await and must never hold a std::Mutex across it.
        if !image_digest_resolved && iteration >= 2 {
            // Handle only THIS turn's user images: use rposition to find the last user
            // message containing an image. The head of `messages` may load images from
            // earlier turns in history (only old images without a persisted digest, e.g.
            // legacy sessions — ones with a digest were already replaced with text
            // during prepare). But the digests we collect/generate describe only this
            // turn's images (instruction injection lives in the current user message,
            // and the fallback uses app.attached_image_files = current turn), so using
            // them to replace old images would mismatch. What we swap out here is this
            // turn's original image — replayed every round in the request projection and
            // exactly the digest target.
            if let Some(idx) = messages.iter().rposition(|m| {
                m.role == "user" && crate::ai::request::content_has_image(&m.content)
            }) {
                let image_paths = app.attached_image_files.clone();
                let digest = match pending_digest_source
                    .as_deref()
                    .and_then(crate::ai::request::parse_digest)
                {
                    Some(d) => Some(d),
                    None => {
                        crate::ai::request::describe_image_for_digest(
                            app,
                            &next_model,
                            &image_paths,
                        )
                        .await
                    }
                };
                if let Some(digest) = digest {
                    crate::ai::request::swap_images_with_digest(
                        &mut messages[idx].content,
                        &digest,
                        &image_paths,
                    );
            // Cross-turn digest: record this round's digest into turn_digest; on normal turn
            // completion it is persisted into the history metadata (see the Ok(_) branch below),
            // and when the next turn loads history the fingerprint retrieves the digest to replace
            // the old image, avoiding cross-turn image re-sends.
            // The fingerprint must come from turn_messages (canonical), not the just-replaced
            // messages[idx]: the request projection carries injected image-processing protocol
            // instructions / reminders whose parts are not persisted, while the loading side
            // (replace_old_images_with_persisted_digests) fingerprints the raw persisted content —
            // only the canonical version matches. If turn_messages has no image-bearing user
            // message (e.g. a resume turn dropped the image), skip silently without persisting.
                    if let Some(fp) =
                        crate::ai::request::last_image_user_message_fingerprint(&turn_messages)
                    {
                        turn_digest = Some((fp, digest, image_paths));
                    }
                }
                image_digest_resolved = true;
            } else {
                image_digest_resolved = true;
            }
        }
        {
            let mc = mcp_client.lock().unwrap();
            let scoped_project_instructions_ready = refresh_skill_turn_for_iteration(
                app,
                &mc,
                skill_manifests,
                &question,
                iteration,
                &mut skill_turn,
                scoped_preflight_targets.required(),
                &mut messages,
            );
            if !scoped_project_instructions_ready {
                let targets = scoped_preflight_targets.required()
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                break 'turn Err(std::io::Error::other(format!(
                    "failed to load target-scoped project instructions before mutation: {targets}"
                ))
                .into());
            }
        }
        if crate::ai::driver::runtime_ctx::take_subagent_wrap_up_request() {
            pre_timeout_wrap_up_requested = true;
            record_force_final_reason(&mut messages, "subagent_pre_timeout_wrap_up", iteration, None);
            force_final_response = true;
            inject_subagent_pre_timeout_wrap_up_note(&mut messages);
        }
        // Side-note real-time guidance: at the top of every iteration, guidance queued in the file
        // is injected into context so the next LLM request sees it immediately. Applies to
        // foreground tasks and subagents alike; subagents are targeted via task-local
        // SUBAGENT_TASK_ID (falling back to the AIOS_SUBAGENT_TASK_ID / SUBAGENT_TASK_ID env vars).
        // Everything goes through side_note::poll_and_inject to avoid two injection paths diverging
        // and leaking original text back.
        {
            let before = messages.len();
            let cnt =
                crate::ai::driver::side_note::poll_and_inject(&app.session_history_file, &mut messages);
            if cnt > 0 {
                let injected = messages[before..].to_vec();
                turn_messages.extend(injected);
                crate::ai::driver::print::print_tool_note_line(
                    "side-note",
                    &format!("injected {cnt} note(s)"),
                );
            }
        }
        let active_skill_name = skill_turn.primary_skill_name().map(str::to_string);
        let compression_report = std::mem::take(&mut supervisor.pending_compression_report);
        let mut response_model = None;
        let execution = match execute_turn_iteration(
            app,
            &next_model,
            &mut response_model,
            &mut messages,
            &turn_messages,
            one_shot_mode,
            &mut persisted_turn_messages,
            should_quit,
            force_final_response,
            terminal_dedupe_candidate.as_deref(),
            active_skill_name.as_deref(),
            iteration,
            compression_report,
        )
        .await
        {
            Ok(e) => e,
            Err(err) => break 'turn Err(err),
        };
            // The pre-timeout wrap-up signal fires mid-model-request: abandon the current request
            // and enter a forced wrap-up iteration immediately instead of waiting for the current
            // (possibly very long) iteration to finish naturally. Consume the signal so the next
            // iteration's top does not inject it again.
        if matches!(&execution, IterationExecution::WrapUpFinal) {
            let _ = crate::ai::driver::runtime_ctx::take_subagent_wrap_up_request();
            pre_timeout_wrap_up_requested = true;
            record_force_final_reason(&mut messages, "subagent_pre_timeout_wrap_up", iteration, None);
            force_final_response = true;
            inject_subagent_pre_timeout_wrap_up_note(&mut messages);
            continue 'turn;
        }
        let was_final_response = matches!(&execution, IterationExecution::FinalResponse(_));
        let had_tool_call_execution = matches!(&execution, IterationExecution::ToolCall(_));
                // Piggyback collection: while the image digest is unresolved, cache this round's
                // untruncated tool-call response text (assistant_text + reasoning_text). At the top
                // of the next loop round we try to parse a digest from it; a hit saves one fallback
                // request. The stream_result raw text must be used — the assistant narration
                // written back to messages is truncated to 800 chars and may cut off the digest's
                // trailing sentinel.
        if !image_digest_resolved && let IterationExecution::ToolCall(tce) = &execution {
            let sr = &tce.stream_result;
            pending_digest_source = Some(format!("{}\n{}", sr.assistant_text, sr.reasoning_text));
        }
        // Digest-only guard: the injected digest instruction can make the first round
        // return ONLY the image digest — no answer, no tool calls. The terminal strips
        // the digest, so ending the turn here looks like the model was interrupted and
        // answered nothing. Instead, cache the digest as the pending source and continue
        // the loop: the top of iteration 2 swaps the image for the digest and asks the
        // model again to answer the user's actual question. The guard only fires while
        // the image digest is unresolved (before iteration 2, since the swap block above
        // always resolves it by then) and never on a forced final (interrupt / iteration
        // limit / health hard-stop), so it can convert at most one round per turn.
        if !image_digest_resolved
            && !force_final_response
            && let IterationExecution::FinalResponse(sr) = &execution
        {
            let raw = format!("{}\n{}", sr.assistant_text, sr.reasoning_text);
            if crate::ai::request::is_digest_only_response(&raw) {
                pending_digest_source = Some(raw);
                continue 'turn;
            }
        }
        {
            let mc = mcp_client.lock().unwrap().routing_snapshot();
                // Calibrate the max_tokens clamp of later requests with the actual prompt_tokens
                // returned by the server. Char-based estimates are conservative (overestimating);
                // the server's actual value is more accurate and reduces unnecessary clamping down.
            let usage_prompt = match &execution {
                IterationExecution::Truncated(sr) | IterationExecution::FinalResponse(sr) => {
                    Some((sr.usage_prompt_tokens, sr.usage_cached_prompt_tokens))
                }
                IterationExecution::ToolCall(tce) => Some((
                    tce.stream_result.usage_prompt_tokens,
                    tce.stream_result.usage_cached_prompt_tokens,
                )),
                _ => None,
            };
            if let Some((pt, cached)) = usage_prompt.filter(|(pt, _)| *pt > 0) {
                app.last_known_prompt_tokens = Some(pt);
                app.last_known_cached_prompt_tokens = Some(cached.min(pt));
            }
                // Empty-response retry count: give up after more than 5 consecutive empty
                // responses to avoid wasting iteration budget
            if matches!(execution, IterationExecution::EmptyResponse) {
                consecutive_empty_responses += 1;
                if consecutive_empty_responses > 5 {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ✗ {} consecutive empty responses; giving up retry",
                        consecutive_empty_responses
                    );
                    final_assistant_text = "[Model returned empty responses repeatedly; please retry or switch models]".to_string();
                    break 'turn Ok(None);
                }
            } else {
                consecutive_empty_responses = 0;
            }
                // Truncation retry count: give up when repeated truncations (output cap / truncated
                // tool JSON) still cannot converge, to avoid endless retries burning budget. The
                // threshold is 3: it gives the model two chances to shrink and rewrite.
            if let IterationExecution::Truncated(stream_result) = &execution {
                consecutive_truncations += 1;
                // Reset tool-loop detection: repeated calls during truncation retries are expected
                // behavior and must not be misjudged as a tool dead-loop that triggers a hard-stop
                // convergence.
                supervisor.mark_truncation_skip();

                if stream_result.stream_error {
                // Truncation caused by stream-read errors (network jitter / abnormal server stream
                // drop). The model did not output too much, so lowering reasoning_effort or
                // injecting a shrink prompt is pointless. It does not accumulate into
                // consecutive_truncations (it is not the model's fault), but a separate counter,
                // consecutive_stream_errors, provides a cap so persistent server stream drops do
                // not retry forever.
                    consecutive_truncations = 0;
                    consecutive_stream_errors += 1;
                    if consecutive_stream_errors > MAX_STREAM_ERROR_RETRIES {
                        let _ = writeln!(
                            std::io::stderr(),
                            "  ✗ {} consecutive response-stream read interruptions; giving up retry",
                            consecutive_stream_errors
                        );
                        final_assistant_text =
                            "[Response stream interrupted repeatedly; the server may be unstable. Please retry later or switch models]"
                                .to_string();
                        break 'turn Ok(None);
                    }
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ⚠ Response-stream read interrupted (consecutive #{consecutive_stream_errors}); auto-retrying, stopping after {MAX_STREAM_ERROR_RETRIES} consecutive failures…"
                    );
                } else {
                // Real truncation: the model hit the output cap or produced half-cut tool JSON.
                    consecutive_stream_errors = 0;

                // Zero-output truncation detection: completion=0 + finish_reason=length means the
                // server rejected the max_tokens value (typically a relay/compatibility layer
                // returns an empty response for an oversized max_tokens instead of an error).
                // Lowering reasoning_effort or disabling thinking is useless here — the problem is
                // not that the model outputs too much but that max_tokens itself is rejected by
                // the server. Strategy: halve max_tokens and retry until the server accepts it.
                    let is_zero_completion = stream_result.usage_completion_tokens == 0
                        && stream_result
                            .finish_reason_value
                            .as_deref()
                            .is_some_and(|r| r == "length");
                    if is_zero_completion {
                        let current_max = app
                            .cli
                            .max_tokens_override
                            .or_else(|| crate::ai::models::max_output_tokens(&app.current_model))
                            .unwrap_or(32768);
                        let halved = (current_max / 2).max(4096);
                        let _ = writeln!(
                            std::io::stderr(),
                            "  ⚠ Zero-output truncation (completion=0); auto-downgrading max_tokens {} → {} and retrying",
                            current_max,
                            halved
                        );
                        app.cli.max_tokens_override = Some(halved);
                        mt_downgraded = true;
                    } else if mt_downgraded {
                // Normal truncation (there was output but it was cut off): the server accepted the
                // current max_tokens, so restore the original value to give later iterations a
                // larger output budget.
                        app.cli.max_tokens_override = saved_max_tokens_override;
                        mt_downgraded = false;
                    }

            // Whether lowering reasoning_effort actually shortens the thought chain for this model.
            // Model-level wire declarations take precedence over provider defaults: for example
            // DashScope DeepSeek uses the enable_thinking switch, yet reasoning intensity is still
            // governed by the top-level reasoning_effort. Only boolean-switch dialects without a
            // declared effective effort wire need thinking disabled outright.
                    let effort_helps =
                        crate::ai::models::reasoning_effort_reduces_thinking(&next_model);

                    if effort_helps {
            // Split truncation by type: distinguish "reasoning ate the budget" from "visible text
            // too long". Judged by the server-reported reasoning_tokens share — when the reasoning
            // share is high, downgrading is effective (it directly shortens the thought chain and
            // hands budget to visible text); when the text is too long, downgrading cannot save the
            // budget (the model is not reasoning) and only wastes quality, so the injected shrink
            // prompt suffices. When reasoning_tokens is not reported (0) or completion is 0, treat
            // it conservatively as "unknown" and fall through the downgrade ladder as a safety net,
            // so the new logic does not regress providers that do not report usage details.
                        let reasoning_reported = stream_result.usage_reasoning_tokens > 0
                            && stream_result.usage_completion_tokens > 0;
                        let reasoning_dominant = reasoning_reported
                            && stream_result.usage_reasoning_tokens as f64
                                / stream_result.usage_completion_tokens as f64
                                >= REASONING_BUDGET_DOMINANCE_RATIO;
                        let text_too_long = reasoning_reported && !reasoning_dominant;

                        if !text_too_long {
            // Progressive reasoning-effort downgrade, handing output budget from reasoning over to
            // actual content. resolve_reasoning_effort reads this field live every iteration, so a
            // change takes effect on the very next request.
            //
            // 1 truncation → High (slightly reduce reasoning overhead)
            // 2 truncations → Medium (shorten the thought chain further)
            // 3+ truncations → Low (the floor that keeps minimal reasoning ability)
            //
            // Compared with the old version, which zeroed it after 2 truncations (None/disabled)
            // and castrated reasoning heavily, keeping reasoning ability is gentler: measured
            // savings are ~15-20% of budget per effort tier, and the model keeps its ability to
            // think. Disabling thinking is reserved for the 3rd-truncation fallback (see below).
            // This ladder deliberately never sends `reasoning_effort: "none"`: an explicit none
            // castrates reasoning entirely, while omitting the field makes the server fall back to
            // its own default tier (gpt-5.x defaults to medium), raising the budget — neither is
            // suitable as a truncation-convergence mechanism.
                            app.cli.reasoning_effort_override =
                                Some(match consecutive_truncations {
                                    1 => Some(crate::ai::provider::ReasoningEffort::High),
                                    2 => Some(crate::ai::provider::ReasoningEffort::Medium),
                                    _ => Some(crate::ai::provider::ReasoningEffort::Low),
                                });
                            let note = if reasoning_dominant {
                                "reasoning ate the output budget; downgrading effort to free budget for visible text"
                            } else {
                                "reasoning usage unreported; conservative effort downgrade"
                            };
                            crate::ai::driver::decision_log::log_truncation_downgrade(
                                crate::ai::driver::decision_log::get_decision_log_store(),
                                &crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
                                crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
                                &next_model,
                                consecutive_truncations,
                                stream_result.usage_reasoning_tokens,
                                stream_result.usage_completion_tokens,
                                true,
                                note,
                            );
                        } else {
            // Visible-text-too-long truncation: do not lower effort (the model is not reasoning, so
            // lowering it cannot save budget); rely only on the already-injected output_truncated
            // shrink prompt to make the model produce less. Record the decision for later audit.
                            crate::ai::driver::decision_log::log_truncation_downgrade(
                                crate::ai::driver::decision_log::get_decision_log_store(),
                                &crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
                                crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
                                &next_model,
                                consecutive_truncations,
                                stream_result.usage_reasoning_tokens,
                                stream_result.usage_completion_tokens,
                                false,
                                "visible text dominated the output budget; keeping effort, prompting shrink only",
                            );
                        }
            // If the effort ladder still truncates at tier 3, lowering effort alone is no longer
            // enough to converge; force thinking off on top of it as a fallback, handing the entire
            // output budget to visible content.
                        if consecutive_truncations >= MAX_MODEL_TRUNCATION_RETRIES {
                            app.cli.thinking_disabled_override = true;
                        }
                    } else {
            // Lowering effort does not work for this dialect: do not waste retry rounds on a useless
            // ladder; force thinking off at the first real truncation, handing the entire output
            // budget to visible content.
                        app.cli.thinking_disabled_override = true;
                    }
                }

                let partial_text = stream_result.assistant_text.trim();
                let has_visible_text = !partial_text.is_empty();

            // The model produced visible text but keeps hitting the length cap (typically a
            // reasoning model whose reasoning fills the budget). Further retries usually do not
            // help — the model will keep producing content of the same length. After one downgraded
            // retry, accept the partial text as the final answer. stream_error scenarios do not
            // count toward consecutive_truncations, so they never reach this branch.
                if has_visible_text
                    && consecutive_truncations >= MAX_MODEL_TRUNCATION_RETRIES
                    && !stream_result.stream_error
                {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ▲ {} consecutive truncated outputs; keeping the partial text produced so far",
                        consecutive_truncations
                    );
                    final_assistant_text = partial_text.to_string();
                    // A truncation finalizes outside the normal FinalResponse gate path. Audit
                    // reports must still be parsed or withheld here so a partial protocol payload
                    // cannot bypass the evidence gate and reach the user as a verified finding.
                    if is_evidence_gated_audit_agent(app.current_agent.as_str()) {
                        let effective_cwd = crate::ai::driver::runtime_ctx::effective_cwd().ok();
                        let _ = audit_evidence_gate_action(
                            app.current_agent.as_str(),
                            &mut messages,
                            &turn_messages,
                            &mut final_assistant_text,
                            effective_cwd.as_deref(),
                            true,
                            iteration,
                            effective_max_iterations,
                        );
                    }
                    break 'turn Ok(None);
                }

            // stream_error already reset consecutive_truncations to 0 above, so this branch is
            // unreachable for it.
                if consecutive_truncations >= MAX_MODEL_TRUNCATION_RETRIES
                    && !stream_result.stream_error
                {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ✗ {} consecutive truncated responses; giving up retry",
                        consecutive_truncations
                    );
            // Keep whatever partial text the model already produced (if any) — it is more valuable
            // than discarding it outright.
                    final_assistant_text = if has_visible_text {
                        partial_text.to_string()
                    } else {
                        "[Model output truncated repeatedly; please shrink per-operation scope (e.g., write files in chunks) or switch models]"
                            .to_string()
                    };
                    break 'turn Ok(None);
                }
            } else {
                consecutive_truncations = 0;
            // Not a truncation: restore the max_tokens downgraded due to zero output.
                if mt_downgraded {
                    app.cli.max_tokens_override = saved_max_tokens_override;
                    mt_downgraded = false;
                }
            }
            let step = match handle_iteration_execution_for_model(
                app,
                response_model.as_deref().unwrap_or(&next_model),
                &question,
                &mc,
                mcp_client,
                execution,
                &mut messages,
                &mut turn_messages,
                one_shot_mode,
                &mut persisted_turn_messages,
                &mut final_assistant_text,
                &mut final_assistant_recorded,
                &mut force_final_response,
                &mut terminal_dedupe_candidate,
                &mut final_gate_state,
                skill_turn.matched_skill_names().is_empty(),
                iteration,
                effective_max_iterations,
                consecutive_truncations,
                &mut turn_had_tool_error,
            ) {
                Ok(s) => s,
                Err(err) => break 'turn Err(err),
            };
            match step {
                TurnLoopStep::ScopedPreflightContinue(targets) => {
                    if supervisor.grant_scoped_preflight_grace() {
                        scoped_preflight_targets.record_pause(targets);
        // This round performed no mutation and must not count toward progress/loop statistics.
                        continue 'turn;
                    }
            // Once the separate preflight budget is exhausted, keep a safe rejection and converge,
            // so switching directories repeatedly cannot indefinitely expand the iteration budget.
                    record_force_final_reason(
                        &mut messages,
                        "scoped_preflight_budget_exhausted",
                        iteration,
                        None,
                    );
                    force_final_response = true;
                    continue 'turn;
                }
                TurnLoopStep::Continue => {
                    let mut new_tools = crate::ai::tools::enable_tools::drain_pending_enable();
                    let pending_mcp = crate::ai::tools::enable_tools::drain_pending_mcp_names();
                    if !pending_mcp.is_empty() {
                        let mcp_all = mc.get_all_tools();
                        for tool in mcp_all {
                            if pending_mcp.iter().any(|n| n == &tool.function.name) {
                                new_tools.push(tool);
                            }
                        }
                    }
                    if !new_tools.is_empty() {
                        if let Some(ctx) = app.agent_context.as_mut() {
                            for tool in new_tools {
                                if !ctx
                                    .tools
                                    .iter()
                                    .any(|t| t.function.name == tool.function.name)
                                {
                                    ctx.tools.push(tool);
                                }
                            }
                        }
                    }
            // Record the tool names this round's assistant actually invoked (deduplicated), kept
            // for aging out unused explicit tools at turn end.
                    if let Some(last_assistant) =
                        messages.iter().rev().find(|m| m.role == "assistant")
                        && let Some(tool_calls) = &last_assistant.tool_calls
                    {
                        for tc in tool_calls {
                            tools_used_this_turn.insert(tc.function.name.clone());
                        }
                    }
                }
                TurnLoopStep::Break => {
                    if was_final_response {
                        final_response_model.clone_from(&response_model);
                    }
                    break 'turn Ok(None);
                }
                TurnLoopStep::Return(outcome) => break 'turn Ok(Some(outcome)),
            }
        }
        // ↓↓↓ Post-processing for the Continue branch (the mc lock has been released; awaiting is
        // safe) ↓↓↓
        let progress_messages = if had_tool_call_execution {
            current_tool_round_messages(&messages)
        } else {
            Vec::new()
        };

        // === Mid-turn progressive compression ===
        // After each round's tool execution, check the total chars of messages; when it exceeds the
        // soft threshold, reuse the cross-turn compression pipeline so long tool-call chains do not
        // blow up the context. Throttling: ① a cooldown of N rounds ② skip when the delta is below
        // DELTA, avoiding repeated no-op compression. The threshold is derived dynamically from
        // history_max_chars (with a floor fallback) so it does not stay locked at 36K/80K after the
        // user adjusts history_max_chars.
        let history_max_chars = app.config.history_max_chars;
        let mid_turn_soft_base = mid_turn_compress_soft_threshold(&next_model, history_max_chars);
            // For long loops, lower the soft threshold to SOFT_FLOOR to curb the O(n²) re-send
            // accumulation (see [`LONG_LOOP_COMPRESS_ITERATION_THRESHOLD`]). The gate and the actual
            // mid_turn_compress call below share the same `mid_turn_soft`, avoiding "the gate opens
            // but compression no-ops".
        let mid_turn_soft = supervisor.effective_mid_turn_soft_threshold(mid_turn_soft_base);
        let mid_turn_hard = mid_turn_compress_hard_threshold(&next_model, history_max_chars);
        let total_chars = crate::ai::history::messages_total_chars_pub(&messages);
        if supervisor.should_try_mid_turn_compress(total_chars, mid_turn_soft) {
            // Resolve the session overflow directory consistently with cross-turn compression
            // (prepare.rs): mid-turn compression uses it to spill large outputs of "incompressible"
            // tools like read_file/grep to files with zero compression plus a preview stub, freeing
            // context without losing information (the model can read_file again).
            let overflow_dir = {
                use crate::ai::history::SessionStore;
                let store = SessionStore::new(app.config.history_file.as_path());
                store.session_assets_dir(&app.session_id)
            };
            let drained: Vec<crate::ai::history::Message> = std::mem::take(&mut messages);
            let (compressed, before, after) = crate::ai::history::mid_turn_compress(
                drained,
                mid_turn_soft,
                Some(overflow_dir.as_path()),
                crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
            );
            messages = compressed;
            supervisor.mark_compress(after);
            let mut compression_report = CompressionReport::default();
            if after < before {
                compression_report.record("mid-turn", before, after);
            }
            // Hard threshold: if still over budget after the lossless + weakly-lossy pipelines,
            // call the LLM summary fallback to fold early conversation into a single internal_note,
            // and merge the compression stages into one status line.
            if after > mid_turn_hard
                && should_try_llm_summary(&app.session_id, after, mid_turn_hard)
            {
                let drained: Vec<crate::ai::history::Message> = std::mem::take(&mut messages);
                let (after_msgs, llm_before, llm_after, was_effective, llm_summary_inserted) =
                    crate::ai::history::mid_turn_llm_summarize(
                        app,
                        drained,
                        MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
                        MID_TURN_LLM_SUMMARY_MAX_CHARS,
                        history_max_chars,
                    crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
                    )
                    .await;
                messages = after_msgs;
                record_llm_summary_attempt_chars(&app.session_id, llm_after);
                compression_report.record_llm_summary_attempt(
                    format!("mid-turn LLM (limit {mid_turn_hard})"),
                    llm_before,
                    llm_after,
                    was_effective,
                    llm_summary_inserted,
                );
                compression_report.emit();
            } else {
                supervisor.pending_compression_report = compression_report;
            }
        }

        // === Tool-loop detection ===
        // If the execute layer has already decided to move to a final response, keep only the
        // iteration-limit scheduling and do not stack loop/progress/checkpoint prompts on top.
        // Conversely, a health hard-stop must not pose as an iteration limit either.
        let force_final_before_health = force_final_response;
        let tool_loop_signal = if force_final_before_health {
            ToolLoopSignal::None
        } else {
            supervisor.record_tool_signatures_for_progress(
                &messages,
                if progress_messages.is_empty() {
                    &messages
                } else {
                    &progress_messages
                },
                PROGRESS_FREE_EXPLORE_ROUNDS,
            )
        };
        let health_signal_injected = tool_loop_signal != ToolLoopSignal::None;
        match tool_loop_signal {
            ToolLoopSignal::None => {}
            ToolLoopSignal::Coarse => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "possible low-yield repetition detected (same target, paging only): injecting converge hint",
                );
                inject_coarse_loop_note(&mut messages);
            }
            ToolLoopSignal::TargetRepeat => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "possible low-yield repetition detected (same target across mixed tool rounds): injecting converge hint",
                );
                inject_target_repeat_loop_note(&mut messages);
            }
            ToolLoopSignal::TargetRescan(target, reads) => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    &format!(
                        "same file re-read from the beginning {reads} times (pagination loop): injecting converge hint (target: {target})"
                    ),
                );
                inject_target_rescan_note(&mut messages, &target, reads);
            }
            ToolLoopSignal::TargetRescanHard(target, reads) => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    &format!(
                        "same file re-read from the beginning past the hard threshold (pagination loop): switching to no-tool handoff (target: {target}, reads: {reads})"
                    ),
                );
                inject_target_rescan_hard_stop_note(&mut messages, &target, reads);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "target-rescan-hard-stop",
                );
                record_force_final_reason(&mut messages, "target_rescan", iteration, Some(&target));
                force_final_response = true;
            }
            ToolLoopSignal::CoarseHard => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "low-yield execute_command repetition hard-stop: switching to no-tool handoff",
                );
                inject_coarse_hard_loop_stop_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "low-yield-hard-stop",
                );
                record_force_final_reason(&mut messages, "low_yield_repetition", iteration, None);
                force_final_response = true;
            }
            ToolLoopSignal::Soft => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "tool-loop detected: injecting self-reflect prompt",
                );
                inject_loop_breaker_note(&mut messages);
                // Inject a task anchor once only for high-risk anomalies, to reduce the chance of
                // goal drift.
                supervisor.maybe_inject_task_anchor(&mut messages, &question, "tool-loop");
            }
            ToolLoopSignal::Hard => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "tool-loop hard-stop: switching to no-tool handoff",
                );
                inject_hard_loop_stop_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "tool-loop-hard-stop",
                );
                record_force_final_reason(&mut messages, "tool_loop", iteration, None);
                force_final_response = true;
            }
            ToolLoopSignal::LowProgressSoft => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "progress-budget: ambiguous recent progress, injecting one review prompt",
                );
                inject_low_progress_soft_note(&mut messages);
            }
            ToolLoopSignal::LowProgressLedger => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "progress-budget: no new measurable evidence after response window, requesting decision ledger",
                );
                inject_progress_ledger_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "low-progress-ledger",
                );
            }
            ToolLoopSignal::ReadOnlyBreadth => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "read-only analysis breadth is high: requesting evidence summary before expanding further",
                );
                let agent_team_active = skill_turn
                    .matched_skill_names()
                    .iter()
                    .any(|name| name == "agent-team");
                inject_read_only_breadth_note(&mut messages, agent_team_active);
            }
            ToolLoopSignal::LowProgressHard => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "progress-budget hard-stop: switching to no-tool handoff",
                );
                inject_low_progress_hard_stop_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "low-progress-hard-stop",
                );
                record_force_final_reason(&mut messages, "progress_no_progress", iteration, None);
                force_final_response = true;
            }
            ToolLoopSignal::ReadOnlyBreadthHard => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "read-only breadth hard-stop: no mutation after wide investigation, switching to no-tool handoff",
                );
                inject_low_progress_hard_stop_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "read-only-breadth-hard-stop",
                );
                record_force_final_reason(&mut messages, "read_only_breadth", iteration, None);
                force_final_response = true;
            }
        }

        // === Tiered, phase-aware tool-round checkpoints ===
        // The phase is judged from the pre-compression current tool round while the accumulated
        // round count stays unchanged; a checkpoint only schedules the next step and never reports
        // the just-completed mutation as a failure. When a more specific health signal already
        // fired this round, the checkpoint is skipped so multiple convergence prompts do not stack.
        let effective_max_iterations = supervisor.effective_max_iterations(max_iterations);
        if !health_signal_injected && !force_final_before_health {
            let checkpoint_phase = tool_round_checkpoint_phase(&progress_messages, &turn_messages);
            if let Some(checkpoint) = supervisor.maybe_inject_tool_round_checkpoint(
                &mut messages,
                effective_max_iterations,
                checkpoint_phase,
            ) {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    &format!(
                        "tool-round checkpoint reached: round={} threshold={} level={} recent_progress={} action={}",
                        supervisor.iteration,
                        checkpoint.threshold,
                        checkpoint.level.label(),
                        checkpoint.phase.recent_progress(),
                        checkpoint.phase.action(),
                    ),
                );
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "tool-round-checkpoint",
                );
            }
        }

        // === Iteration-limit self-reflection ===
        // execute.rs sets force_final_response to true when iteration >= max_iterations. At that
        // point, on top of the existing "Tool limit reached" system prompt, one more specific
        // reflection prompt is injected (only once, to avoid spamming).
        supervisor.maybe_inject_iteration_limit_note(
            &mut messages,
            effective_max_iterations,
            force_final_before_health && !pre_timeout_wrap_up_requested,
        );
        if force_final_before_health && !pre_timeout_wrap_up_requested {
            supervisor.maybe_inject_task_anchor(&mut messages, &question, "iteration-limit");
        }
    };

            // Restore the reasoning-effort override from before this turn: truncation retries may
            // have lowered it to Low temporarily; restore it uniformly here (covering every break
            // 'turn exit) so the downgrade does not leak into later turns and pollute the user's
            // session-level setting.
    app.cli.reasoning_effort_override = saved_effort_override;
    app.cli.thinking_disabled_override = saved_thinking_disabled;
    app.cli.max_tokens_override = saved_max_tokens_override;

            // Age out explicit-enabled tools that were not used this turn. After N consecutive
            // turns of idle use they are demoted, preventing "enabled once, welded forever".
    crate::ai::tools::enable_tools::age_unused_explicit_tools(tools_used_this_turn.iter());

    let loop_result = loop_result.map_err(|e: Box<dyn std::error::Error>| e.to_string());

            // A one-shot continuation is set up only when an active skill explicitly requested user
            // input through a tool AND this round ended normally. This avoids guessing from
            // natural-language question marks / keywords, and external skills need no manifest
            // changes.
    let requested_user_input = crate::ai::tools::skill_tools::take_pending_user_input_request();
    if requested_user_input && loop_result.is_ok() && !app.last_turn_interrupted {
        if !skill_turn.matched_skill_names().is_empty() {
            app.pending_skill_continuation =
                Some(crate::ai::types::PendingSkillContinuation {
                    skill_names: skill_turn.matched_skill_names().to_vec(),
                });
        }
    }

    let final_skill_name = skill_turn.primary_skill_name().map(str::to_owned);
    skill_turn.restore_agent_context(app);

    match loop_result {
        Ok(Some(outcome)) => {
            app.last_turn_had_tool_calls = false;
            Ok(outcome)
        }
        Ok(_) => {
            // Cross-turn image-digest persistence (only on the normal turn-completion path; the
            // interrupt / quit / error branches skip it, the digest is dropped, and the original
            // image is re-sent once next round). Digests parsed inside the tool loop were already
            // recorded in turn_digest; on a single-round response (no tool loop) the digest block
            // does not run, so instead parse the digest from the final reply text.
            let mut digest_to_persist = turn_digest.take();
            if digest_to_persist.is_none() {
                if let Some(digest) = crate::ai::request::parse_digest(&final_assistant_text) {
                    if let Some(fp) =
                        crate::ai::request::last_image_user_message_fingerprint(&turn_messages)
                    {
                        digest_to_persist = Some((fp, digest, app.attached_image_files.clone()));
                    }
                }
            }
            if let Some((fp, digest, paths)) = digest_to_persist {
                let _ = crate::ai::history::upsert_image_digest_sqlite(
                    &app.session_history_file,
                    &fp,
                    &digest,
                    &paths,
                );
            }
            finalize_turn(
                app,
                &next_model,
                final_response_model.as_deref().unwrap_or(&next_model),
                &question,
                &final_assistant_text,
                final_assistant_recorded,
                terminal_dedupe_candidate.as_deref(),
                final_skill_name.as_deref(),
                &mut turn_messages,
                one_shot_mode,
                &mut persisted_turn_messages,
                should_quit,
                turn_had_tool_error,
            )
            .await
        }
        Err(err_str) => Err(err_str.into()),
    }
}
