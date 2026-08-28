// =============================================================================
// Orchestrator tests (extracted from orchestrator.rs, logic preserved)
// =============================================================================

use super::*;

#[test]
fn subagent_pre_timeout_wrap_up_note_requires_immediate_final_answer() {
    let mut messages = Vec::new();
    inject_subagent_pre_timeout_wrap_up_note(&mut messages);

    let note = messages
        .last()
        .and_then(|message| message.content.as_str())
        .expect("wrap-up note should be textual");
    assert!(note.contains("no-tool wrap-up mode"));
    assert!(note.contains("final answer"));
    assert!(!note.contains("`/audit`"));
}

#[test]
fn force_final_reason_is_request_only_and_deduplicated() {
    let mut messages = Vec::new();

    record_force_final_reason(&mut messages, "iteration_limit", 24, None);
    record_force_final_reason(&mut messages, "tool_loop_exact", 25, None);

    assert_eq!(messages.len(), 1);
    let note = messages[0]
        .content
        .as_str()
        .expect("stop reason should be textual");
    assert!(note.contains("reason=iteration_limit"));
    assert!(note.contains("iteration=24"));
}

#[test]
fn detect_tool_loop_triggers_after_window_of_identical_signatures() {
    let sig = vec!["read_file::{\"path\":\"a.rs\"}".to_string()];
    // Below the window size
    let history = vec![sig.clone(); TOOL_LOOP_SOFT_WINDOW - 1];
    assert!(!detect_tool_loop(&history, TOOL_LOOP_SOFT_WINDOW));
    // Fills the soft window but not yet the hard window
    let history = vec![sig.clone(); TOOL_LOOP_SOFT_WINDOW];
    assert!(detect_tool_loop(&history, TOOL_LOOP_SOFT_WINDOW));
    assert!(!detect_tool_loop(&history, TOOL_LOOP_HARD_WINDOW));
    // Fills the hard window and is fully identical
    let history = vec![sig.clone(); TOOL_LOOP_HARD_WINDOW];
    assert!(detect_tool_loop(&history, TOOL_LOOP_HARD_WINDOW));
    // Fills the window but with one differing round
    let mut history = vec![sig.clone(); TOOL_LOOP_HARD_WINDOW];
    history[1] = vec!["read_file::{\"path\":\"b.rs\"}".to_string()];
    assert!(!detect_tool_loop(&history, TOOL_LOOP_HARD_WINDOW));
}

#[test]
fn scoped_preflight_grace_has_separate_bounded_budget() {
    let mut supervisor = TurnSupervisor::default();
    assert_eq!(supervisor.effective_max_iterations(1), 1);
    for expected_rounds in 1..=MAX_SCOPED_PREFLIGHT_GRACE_ROUNDS {
        assert!(supervisor.grant_scoped_preflight_grace());
        assert_eq!(supervisor.effective_max_iterations(1), 1 + expected_rounds);
    }
    assert!(!supervisor.grant_scoped_preflight_grace());
    assert_eq!(
        supervisor.effective_max_iterations(1),
        1 + MAX_SCOPED_PREFLIGHT_GRACE_ROUNDS
    );
}

#[test]
fn scoped_preflight_targets_survive_intervening_prompt_rebuilds() {
    let target = std::path::PathBuf::from("src/bin/ai/driver/turn_runtime/orchestrator.rs");
    let mut targets = ScopedPreflightTargets::default();
    targets.record_pause(vec![target.clone()]);

    assert_eq!(targets.required(), [target.as_path()]);
    assert_eq!(targets.required(), [target.as_path()]);
}

#[test]
fn detect_tool_loop_triggers_for_short_periodic_cycles() {
    let a = vec!["tree::{\"path\":\"src\"}".to_string()];
    let b = vec!["read_file::{\"path\":\"src/bin/a.rs\"}".to_string()];
    let c = vec!["tree::{\"path\":\"src/bin\"}".to_string()];

    assert!(detect_tool_loop(
        &[a.clone(), b.clone(), a.clone(), b.clone()],
        TOOL_LOOP_SOFT_WINDOW
    ));
    assert!(detect_tool_loop(
        &[a.clone(), b.clone(), c.clone(), a, b, c],
        TOOL_LOOP_HARD_WINDOW
    ));
}

#[test]
fn detect_tool_loop_ignores_empty_signatures() {
    let history = vec![Vec::<String>::new(); TOOL_LOOP_HARD_WINDOW];
    assert!(!detect_tool_loop(&history, TOOL_LOOP_SOFT_WINDOW));
    assert!(!detect_tool_loop(&history, TOOL_LOOP_HARD_WINDOW));
}

#[test]
fn detect_execute_command_coarse_loop_requires_execute_command_only_signatures() {
    let execute_sig = vec!["execute_command::{\"command\":\"ls:/tmp\"}".to_string()];
    let read_sig = vec!["read_file::{\"path\":\"src/main.rs\"}".to_string()];
    let history = vec![execute_sig.clone(); TOOL_LOOP_COARSE_HARD_WINDOW];
    assert!(detect_execute_command_coarse_loop(
        &history,
        TOOL_LOOP_COARSE_HARD_WINDOW
    ));

    let mut mixed = vec![execute_sig; TOOL_LOOP_COARSE_HARD_WINDOW];
    mixed[0] = read_sig;
    assert!(!detect_execute_command_coarse_loop(
        &mixed,
        TOOL_LOOP_COARSE_HARD_WINDOW
    ));
}

#[test]
fn detect_tool_loop_matches_cycle_prefix_when_window_not_divisible_by_period() {
    // Regression: soft window 4 is not divisible by period 3; the old
    // implementation hit `window % period != 0` and `continue`d directly, so
    // A-B-C-A-B-C got an unannounced hard stop at round 6 (hard is checked
    // first and soft second, so soft could never fire first and the escalation
    // invariant was broken). The fix treats "several full cycles plus a
    // partial-cycle prefix" as a loop too, so the 3-cycle case gets Soft first
    // at round 4 (A-B-C-A).
    let a = vec!["tree::{\"path\":\"src\"}".to_string()];
    let b = vec!["read_file::{\"path\":\"src/bin/a.rs\"}".to_string()];
    let c = vec!["tree::{\"path\":\"src/bin\"}".to_string()];

    // Below the window: no false positive.
    assert!(!detect_tool_loop(
        &[a.clone(), b.clone(), c.clone()],
        TOOL_LOOP_SOFT_WINDOW
    ));
    // 3 cycles + 1 prefix exactly fills soft window 4: should trigger Soft.
    assert!(detect_tool_loop(
        &[a.clone(), b.clone(), c.clone(), a.clone()],
        TOOL_LOOP_SOFT_WINDOW
    ));
    // A full hard window (6 rounds of whole cycles) still triggers, and the
    // divisible path is unaffected.
    assert!(detect_tool_loop(
        &[
            a.clone(),
            b.clone(),
            c.clone(),
            a.clone(),
            b.clone(),
            c.clone()
        ],
        TOOL_LOOP_HARD_WINDOW
    ));
    // A non-matching prefix (round 4 unrelated to the cycle) must not misfire.
    assert!(!detect_tool_loop(
        &[a.clone(), b.clone(), c.clone(), b.clone()],
        TOOL_LOOP_SOFT_WINDOW
    ));
}

#[test]
fn execute_command_is_read_only_skips_nav_segments_and_requires_all_substantive_read_only() {
    // Leading cd/export segments have no side effects and should be skipped:
    // `cd X && git status` is read-only.
    assert!(execute_command_is_read_only(
        "cd /tmp && git status --short"
    ));
    assert!(execute_command_is_read_only(
        "cd /tmp && export FOO=1 && ls -la"
    ));
    // Any substantive segment that may change → not read-only (closes the old
    // implementation's blind spot of looking only at the first segment).
    assert!(!execute_command_is_read_only("ls /tmp && rm -rf build"));
    assert!(!execute_command_is_read_only(
        "cd /tmp && git checkout master"
    ));
    // Purely read-only commands still hold.
    assert!(execute_command_is_read_only("git log --oneline -5"));
    assert!(execute_command_is_read_only("ls -la /tmp"));
    // cargo verification subcommands do not modify source → read-only;
    // rerunning the same verification command must no longer count as a
    // Mutation.
    assert!(execute_command_is_read_only("cargo check --bin a"));
    assert!(execute_command_is_read_only(
        "cd /Users/bytedance/rust_tools && cargo test --bin a focused_test"
    ));
    assert!(execute_command_is_read_only("cargo build --release"));
    assert!(execute_command_is_read_only("cargo clippy"));
    assert!(execute_command_is_read_only("cargo fmt --check"));
    // Rewrites source / executes arbitrary programs → not read-only.
    assert!(!execute_command_is_read_only("cargo add serde"));
    assert!(!execute_command_is_read_only("cargo fmt"));
    assert!(!execute_command_is_read_only("cargo clippy --fix"));
    assert!(!execute_command_is_read_only("cargo run --bin a"));
}

#[test]
fn low_information_probe_only_matches_pure_echo_commands() {
    use super::checkpoint::command_is_low_information_probe;
    // A pure echo probe (with cd/export leads) carries no information → must
    // not refresh the no-progress budget.
    assert!(command_is_low_information_probe("echo \"integrate\""));
    assert!(command_is_low_information_probe("echo x"));
    assert!(command_is_low_information_probe("cd /tmp && echo done"));
    assert!(command_is_low_information_probe("echo a && echo b"));
    // Contains any real read-only segment → not a pure probe; still accounted
    // for normally (legitimate exploration is not penalized).
    assert!(!command_is_low_information_probe("cat version.txt"));
    assert!(!command_is_low_information_probe("echo hi && cargo check --bin a"));
    assert!(!command_is_low_information_probe("echo start && grep foo src.rs"));
    assert!(!command_is_low_information_probe("ls -la"));
    // Empty command / leading segments only → no substantive echo segment, not
    // a probe.
    assert!(!command_is_low_information_probe(""));
    assert!(!command_is_low_information_probe("cd /tmp"));
}

#[test]
fn distinct_echo_probes_do_not_each_count_as_new_evidence() {
    use super::progress::extract_round_evidence_fingerprints;
    let echo_round = |call_id: &str, cmd: &str, out: &str| {
        vec![
            crate::ai::history::Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String(String::new()),
                tool_calls: Some(vec![crate::ai::types::ToolCall {
                    id: call_id.to_string(),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall {
                        name: "execute_command".to_string(),
                        arguments: format!("{{\"command\":\"{cmd}\"}}"),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            crate::ai::history::Message {
                role: "tool".to_string(),
                content: serde_json::Value::String(out.to_string()),
                tool_calls: None,
                tool_call_id: Some(call_id.to_string()),
                reasoning_content: None,
            },
        ]
    };

    // Mutually distinct echo probes: before the fix each produced a fresh
    // fingerprint that refreshed the budget; after the fix none counts as
    // evidence.
    assert!(
        extract_round_evidence_fingerprints(&echo_round("c1", "echo \\\"integrate\\\"", "integrate"))
            .is_empty(),
        "echo probe must not count as new evidence"
    );
    assert!(
        extract_round_evidence_fingerprints(&echo_round("c2", "echo \\\"x\\\"", "x")).is_empty()
    );
    // Control: real read-only exploration (cat of file contents) still
    // produces an evidence fingerprint, unaffected.
    assert!(
        !extract_round_evidence_fingerprints(&echo_round("c3", "cat version.txt", "1.2.3"))
            .is_empty(),
        "genuine read-only inspection must still register as evidence"
    );
}

#[test]
fn cargo_verify_evidence_normalizes_volatile_output() {
    // The same verification result (differing only in duration/compile
    // progress) → identical after normalization → same fingerprint.
    let a = normalize_verify_output(
        "   Compiling rust_tools v0.1.0 (/Users/bytedance/rust_tools)\n    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.05s\n     Running unittests src/lib.rs (target/debug/deps/rust_tools-0123abc)\nrunning 2 tests\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s",
    );
    let b = normalize_verify_output(
        "    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.09s\nrunning 2 tests\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s",
    );
    assert_eq!(a, b);
    assert!(a.contains("2 passed"));
    // A different verification result (failure) → still different after
    // normalization.
    let c = normalize_verify_output(
        "running 2 tests\ntest result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s",
    );
    assert_ne!(a, c);
    // Classifier: cargo verification subcommand (possibly with cd leads) →
    // true; non-cargo / non-verification → false.
    assert!(command_is_cargo_verify("cargo test --bin a focused_test"));
    assert!(command_is_cargo_verify("cd /tmp && cargo check --bin a"));
    assert!(!command_is_cargo_verify("git status"));
    assert!(!command_is_cargo_verify("cargo run --bin a"));
}

#[test]
fn turn_supervisor_emits_soft_then_hard_loop_signal() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let assistant_with_same_read = |id: &str| crate::ai::history::Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{\"path\":\"src/main.rs\",\"offset\":140,\"limit\":80}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    };

    // Collect the signal for each round: no trigger for the first
    // SOFT_WINDOW-1 rounds; Soft fires on round SOFT_WINDOW. Soft clears the
    // old samples, so HARD_WINDOW more repeats after the notice are required
    // before Hard fires.
    let mut signals = Vec::new();
    for i in 0..(TOOL_LOOP_SOFT_WINDOW + TOOL_LOOP_HARD_WINDOW) {
        messages.push(assistant_with_same_read(&format!("tc-{i}")));
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }
    assert!(
        signals[..TOOL_LOOP_SOFT_WINDOW - 1]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None)),
        "should stay quiet before the soft window fills"
    );
    assert!(matches!(
        signals[TOOL_LOOP_SOFT_WINDOW - 1],
        ToolLoopSignal::Soft
    ));
    assert!(matches!(
        signals[TOOL_LOOP_SOFT_WINDOW + TOOL_LOOP_HARD_WINDOW - 1],
        ToolLoopSignal::Hard
    ));
}

#[test]
fn task_progress_after_loop_soft_restarts_tool_loop_ladder() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();

    // Repeat the same read-only call until the soft threshold.
    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        messages.push(pb_read_msg("src/main.rs", &format!("read-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_SOFT_WINDOW - 1 {
            assert!(matches!(signal, ToolLoopSignal::Soft));
        }
    }
    assert!(supervisor.loop_breaker_injected);

    // A real mutation after soft means the task is progressing; the old loop
    // state must be cleared.
    messages.push(pb_apply_patch_msg("patch-1"));
    assert!(matches!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS,),
        ToolLoopSignal::None
    ));
    assert!(!supervisor.loop_breaker_injected);
    assert!(supervisor.tool_signature_history.is_empty());

    // A new round of repetition must earn soft again first, not inherit the
    // old state and jump straight to hard-stop.
    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        pb_successful_read_round(&mut messages, "src/other.rs", 5, &format!("retry-{i}"), "body");
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_SOFT_WINDOW - 1 {
            assert!(matches!(signal, ToolLoopSignal::Soft));
        } else {
            assert!(matches!(signal, ToolLoopSignal::None));
        }
    }
}

#[test]
fn soft_rearms_coarse_and_target_gates_so_post_soft_page_cycling_is_still_caught() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();

    // Phase 1: repeat the same read-only call until soft fires.
    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        messages.push(pb_read_msg("src/main.rs", &format!("read-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_SOFT_WINDOW - 1 {
            assert!(matches!(signal, ToolLoopSignal::Soft));
        }
    }
    assert!(supervisor.loop_breaker_injected);
    // The soft handler cleared the coarse/target history and re-armed their
    // gates.
    assert!(supervisor.tool_signature_history_coarse.is_empty());
    assert!(supervisor.tool_target_history.is_empty());
    assert!(!supervisor.coarse_loop_note_injected);
    assert!(!supervisor.target_repeat_note_injected);

    // Phase 2: after soft, the model switches to a batch of `ls` variants over
    // the same log directory. exact signatures all differ, so exact soft/hard
    // never re-fires; but every coarse signature is `ls:/data01/logs`, so
    // Coarse should fire once COARSE_WINDOW fills (the old implementation kept
    // that gate permanently blocked by loop_breaker_injected and leaked to the
    // iteration cap).
    // Note: the first `ls` round sees `/data01/logs` as a brand-new target and
    // takes assess_progress's new-target + already-injected-loop branch,
    // triggering reset_tool_loop_escalation (clears the coarse history and
    // re-arms the gate). COARSE_WINDOW more rounds of the same coarse sample
    // must be accumulated afterwards.
    let ls_variants = [
        "ls -lt /data01/logs | head -20",
        "ls -la /data01/logs | head -30",
        "ls /data01/logs 2>/dev/null",
        "ls -l /data01/logs | tail -5",
        "ls -lt /data01/logs | head -10",
        "ls -la /data01/logs | head -50",
    ];
    for (i, cmd) in ls_variants.iter().enumerate() {
        messages.push(pb_execute_command_msg(cmd, &format!("ls-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        // Round 5 (ls-5, index 5) is the first round that fills COARSE_WINDOW
        // after the reset.
        if i == TOOL_LOOP_COARSE_WINDOW {
            assert!(
                matches!(signal, ToolLoopSignal::Coarse),
                "post-soft coarse page cycling should be caught at window size"
            );
        }
    }
    assert!(supervisor.coarse_loop_note_injected);
}

#[test]
fn mark_truncation_skip_resets_full_loop_detection_ladder() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let assistant_with_read = |id: &str| crate::ai::history::Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{\"path\":\"src/main.rs\",\"offset\":5}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    };

    // Round 1: accumulate until soft fires.
    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        messages.push(assistant_with_read(&format!("tc-{i}")));
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    }
    // Verify soft fired and the flag is set.
    assert!(supervisor.loop_breaker_injected);
    assert!(!supervisor.hard_loop_stop_injected);

    // Truncation-triggered marker: history cleared, skip +1, all flags reset.
    supervisor.mark_truncation_skip();
    assert!(supervisor.tool_signature_history.is_empty());
    assert!(supervisor.tool_signature_history_coarse.is_empty());
    assert_eq!(supervisor.skip_tool_signature_rounds, 1);
    // Key check: every one-shot flag is reset.
    assert!(!supervisor.hard_loop_stop_injected);
    assert!(!supervisor.loop_breaker_injected);
    assert!(!supervisor.coarse_loop_note_injected);

    // Truncated iteration: signature recording is skipped.
    messages.push(assistant_with_read("tc-skip"));
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::None));
    assert!(supervisor.tool_signature_history.is_empty());
    assert_eq!(supervisor.skip_tool_signature_rounds, 0);

    // Round 2: accumulate again after recovery; verify soft can fire a second
    // time.
    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        messages.push(assistant_with_read(&format!("tc2-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_SOFT_WINDOW - 1 {
            // The 4th occurrence should trigger soft.
            assert!(matches!(signal, ToolLoopSignal::Soft));
            assert!(supervisor.loop_breaker_injected);
        } else {
            assert!(matches!(signal, ToolLoopSignal::None));
        }
    }

    // After soft a full hard window must be re-accumulated, verifying the full
    // escalation ladder is restored.
    for i in 0..TOOL_LOOP_HARD_WINDOW {
        messages.push(assistant_with_read(&format!("tc3-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_HARD_WINDOW - 1 {
            // Only after 6 more repeats following soft should hard fire.
            assert!(matches!(signal, ToolLoopSignal::Hard));
            assert!(supervisor.hard_loop_stop_injected);
        }
    }
}

#[test]
fn turn_supervisor_compress_gate_respects_cooldown_and_delta() {
    const SOFT: usize = super::super::MID_TURN_COMPRESS_SOFT_FLOOR;
    let mut s = TurnSupervisor::default();
    s.iteration = 3;
    assert!(s.should_try_mid_turn_compress(SOFT + 10, SOFT));

    s.mark_compress(SOFT + 10);
    assert!(!s.should_try_mid_turn_compress(SOFT + 20, SOFT));

    s.iteration += MID_TURN_COMPRESS_COOLDOWN_ITERATIONS;
    assert!(!s.should_try_mid_turn_compress(
        s.last_compress_after_chars + MID_TURN_COMPRESS_DELTA_THRESHOLD - 1,
        SOFT,
    ));
    assert!(s.should_try_mid_turn_compress(
        s.last_compress_after_chars + MID_TURN_COMPRESS_DELTA_THRESHOLD,
        SOFT,
    ));
}

/// Long-loop awareness: short turns keep the baseline soft threshold; once the
/// iteration count reaches the threshold, the effective soft threshold drops
/// to SOFT_FLOOR so content-level dedup kicks in early and curbs O(n²)
/// resending of accumulated content.
/// This is the direct fix for incidents like aefa66f2 (medium history ~120K +
/// 56 iterating rounds) hitting TPM: the 135K baseline never fired, while the
/// lowered 36K threshold starts compression mid-way through long loops.
#[test]
fn long_loop_lowers_effective_mid_turn_soft_threshold() {
    const FLOOR: usize = super::super::MID_TURN_COMPRESS_SOFT_FLOOR;
    // The flagship large-window model's baseline soft threshold is far above
    // FLOOR (simulating 135K).
    let base = 135_000usize;
    assert!(base > FLOOR, "precondition: base threshold above floor");

    let mut s = TurnSupervisor::default();

    // Short turn (below the long-loop threshold): effective threshold ==
    // baseline; normal single-round large tasks are unaffected.
    s.iteration = LONG_LOOP_COMPRESS_ITERATION_THRESHOLD - 1;
    assert_eq!(s.effective_mid_turn_soft_threshold(base), base);
    // Here ~120K of history (< 135K baseline) does not trigger compression —
    // exactly the old behavior's blind spot.
    assert!(
        !s.should_try_mid_turn_compress(120_000, s.effective_mid_turn_soft_threshold(base))
    );

    // Long loop (threshold reached): the effective threshold drops to FLOOR
    // and the same ~120K history triggers compression immediately.
    s.iteration = LONG_LOOP_COMPRESS_ITERATION_THRESHOLD;
    assert_eq!(s.effective_mid_turn_soft_threshold(base), FLOOR);
    assert!(s.should_try_mid_turn_compress(120_000, s.effective_mid_turn_soft_threshold(base)));

    // If the baseline is already below FLOOR (a tiny history_max_chars), min()
    // guarantees the threshold is never raised.
    let tiny_base = FLOOR / 2;
    assert_eq!(s.effective_mid_turn_soft_threshold(tiny_base), tiny_base);
}

#[test]
fn task_anchor_note_truncates_goal_text() {
    let mut messages = Vec::new();
    let long_q = "x".repeat(TASK_ANCHOR_MAX_QUESTION_CHARS + 30);
    inject_task_anchor_note(&mut messages, long_q.as_str(), 5, "test");
    let text = messages[0].content.as_str().unwrap_or_default().to_string();
    assert!(text.contains("[task-anchor]"));
    assert!(text.contains("iteration=5"));
    assert!(text.contains("…"));
}

#[test]
fn strip_volatile_args_removes_paging_keys() {
    let mut v = serde_json::json!({
        "path": "src/main.rs",
        "offset": 100,
        "limit": 80,
        "page": 2,
        "cursor": "abc",
        "max_results": 50,
        "keep": "yes"
    });
    strip_volatile_args(&mut v);
    let obj = v.as_object().unwrap();
    assert!(obj.contains_key("path"));
    assert!(obj.contains_key("keep"));
    for key in VOLATILE_ARG_KEYS {
        assert!(
            !obj.contains_key(*key),
            "volatile key {key} should be stripped"
        );
    }
}

#[test]
fn coarse_execute_command_signature_collapses_log_listing_window_variants() {
    let a = coarse_execute_command_signature("ls -lt /data01/dataagent_be/logs/ | head -20");
    let b = coarse_execute_command_signature("ls -la /data01/dataagent_be/logs/ | head -30");
    let c = coarse_execute_command_signature("ls /data01/dataagent_be/logs/ 2>/dev/null");
    assert_eq!(a, "ls:/data01/dataagent_be/logs");
    assert_eq!(a, b);
    assert_eq!(b, c);
}

#[test]
fn coarse_execute_command_signature_collapses_middle_paging_offsets() {
    // Root cause of pagination-loop escapes: offset literals in
    // `tail -c +N` / `sed -n 'N,Mp'` used to be kept in the coarse signature,
    // so paging the same file through different windows produced a distinct
    // signature every round and neither coarse nor target-repeat could catch
    // it. Offsets/line ranges describe the "window", not the target resource,
    // and should be stripped (like read_file's offset/limit stripping) so
    // paging variants collapse into one signature.
    let a = coarse_execute_command_signature("tail -c +2401 src/all_desc.txt | head -c 2400");
    let b = coarse_execute_command_signature("tail -c +4801 src/all_desc.txt | head -c 2400");
    let c = coarse_execute_command_signature("tail -c +1 src/all_desc.txt | head -c 1400");
    assert_eq!(a, "tail:src/all_desc.txt");
    assert_eq!(a, b);
    assert_eq!(b, c);
    let d = coarse_execute_command_signature("sed -n '40,80p' src/all_desc.txt");
    let e = coarse_execute_command_signature("sed -n '1,40p' src/all_desc.txt");
    assert_eq!(d, "sed:src/all_desc.txt");
    assert_eq!(d, e);
    // Different target files remain distinguished to avoid false merges.
    assert_ne!(a, coarse_execute_command_signature("tail -c +2401 other.txt"));
}

#[test]
fn coarse_catches_middle_paging_loop_never_from_top() {
    // Pure "paging" loop: the same file is read repeatedly from mid-file
    // offsets (tail -c +N with a different offset each round, never +1, so the
    // from-top rescan detector cannot catch it). Whole-round raw commands all
    // differ, so exact matching fails; before the fix the coarse signature
    // kept the offset literal and the whole-round coarse sets never matched
    // either, letting the loop escape to the iteration cap. After the fix
    // offsets are stripped and everything folds to `tail:<file>`: Coarse soft
    // notice on round 5, CoarseHard hard stop on round 8.
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let file = "src/all_desc.txt";
    let mut signals = Vec::new();
    for i in 0..TOOL_LOOP_COARSE_HARD_WINDOW {
        let offset = 2401 + i * 2400; // Mid-file offset, never 1 (from-top).
        pb_execute_command_round(
            &mut messages,
            &format!("tail -c +{offset} {file} | head -c 2400"),
            &format!("p{i}"),
        );
        signals.push(supervisor.record_tool_signatures(
            &messages,
            PROGRESS_FREE_EXPLORE_ROUNDS,
        ));
    }
    assert!(
        matches!(signals[TOOL_LOOP_COARSE_WINDOW - 1], ToolLoopSignal::Coarse),
        "5th middle-paging round should hit Coarse, got {:?}",
        signals[TOOL_LOOP_COARSE_WINDOW - 1]
    );
    assert!(
        matches!(
            signals[TOOL_LOOP_COARSE_HARD_WINDOW - 1],
            ToolLoopSignal::CoarseHard
        ),
        "8th middle-paging round should hit CoarseHard, got {:?}",
        signals[TOOL_LOOP_COARSE_HARD_WINDOW - 1]
    );
}

#[test]
fn coarse_execute_command_signature_keeps_search_pattern_and_path() {
    let sig = coarse_execute_command_signature(
        "grep -rl \"24394294\" /data01/dataagent_be/logs/ 2>/dev/null | head -10",
    );
    assert_eq!(sig, "grep:/data01/dataagent_be/logs#24394294");
}

#[test]
fn coarse_execute_command_signature_collapses_git_forensics_variants() {
    let log_and_status = coarse_execute_command_signature(
        "git log --oneline --decorate -5 && git status --short",
    );
    let show_pair = coarse_execute_command_signature(
        "git show --stat --oneline 5dfc5676f && git show --stat --oneline 76530274f",
    );
    let diff_pair = coarse_execute_command_signature(
        "git diff --stat 5dfc5676f^ 76530274f && git diff --stat 5dfc5676f 76530274f",
    );
    assert_eq!(log_and_status, "git:inspect");
    assert_eq!(log_and_status, show_pair);
    assert_eq!(show_pair, diff_pair);
}

#[test]
fn coarse_execute_command_signature_keeps_git_global_option_before_subcommand() {
    let with_global = coarse_execute_command_signature("git -C /tmp/worktree status --short");
    let plain = coarse_execute_command_signature("git status --short");
    assert_eq!(with_global, "git:inspect");
    assert_eq!(with_global, plain);
}

#[test]
fn turn_supervisor_emits_coarse_signal_for_same_file_paging() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let assistant_paging_read = |id: &str, offset: usize| crate::ai::history::Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: format!(
                    "{{\"path\":\"src/main.rs\",\"offset\":{offset},\"limit\":80}}"
                ),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    };

    // Offsets increase every round: byte-exact signatures all differ → no
    // soft/hard; with offset/limit stripped the coarse signatures match →
    // Coarse fires once COARSE_WINDOW fills.
    let mut signals = Vec::new();
    for i in 0..TOOL_LOOP_COARSE_WINDOW {
        messages.push(assistant_paging_read(&format!("tc-{i}"), i * 80));
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }
    assert!(
        signals[..TOOL_LOOP_COARSE_WINDOW - 1]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None)),
        "exact paging must not trip soft/hard before coarse window fills"
    );
    assert!(matches!(signals.last().unwrap(), ToolLoopSignal::Coarse));
    assert!(supervisor.coarse_loop_note_injected);

    // Coarse notices only once: continuing the same paging returns no further
    // Coarse.
    messages.push(assistant_paging_read(
        "tc-extra",
        TOOL_LOOP_COARSE_WINDOW * 80,
    ));
    assert!(matches!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    ));
}

#[test]
fn turn_supervisor_emits_coarse_signal_for_execute_command_log_listing_variants() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let commands = [
        "ls -la /data01/dataagent_be/logs/ | head -30",
        "ls -lt /data01/dataagent_be/logs/ | head -20",
        "ls /data01/dataagent_be/logs/ | head -30",
        "ls -lt /data01/dataagent_be/logs/ 2>/dev/null | head -40",
        "ls -la /data01/dataagent_be/logs/ | head -50",
    ];
    for (i, command) in commands.iter().enumerate() {
        messages.push(crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: format!("tc-{i}"),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "execute_command".to_string(),
                    arguments: serde_json::json!({ "command": command }).to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i < TOOL_LOOP_COARSE_WINDOW - 1 {
            assert!(matches!(signal, ToolLoopSignal::None));
        } else {
            assert!(matches!(signal, ToolLoopSignal::Coarse));
        }
    }
    assert!(supervisor.coarse_loop_note_injected);
}

#[test]
fn turn_supervisor_emits_coarse_signal_for_execute_command_git_forensics_variants() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let commands = [
        "git log --oneline --decorate -5 && git status --short",
        "git show --stat --oneline 5dfc5676f && git show --stat --oneline 76530274f",
        "git diff --stat 5dfc5676f^ 76530274f && git diff --stat 5dfc5676f 76530274f",
        "git show --format=fuller --name-status 5dfc5676f && git show --format=fuller --name-status 76530274f",
        "git reflog -8 --date=iso --format='%h %gd %gs %cd' && git status --short --branch",
    ];
    for (i, command) in commands.iter().enumerate() {
        messages.push(crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: format!("git-tc-{i}"),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "execute_command".to_string(),
                    arguments: serde_json::json!({ "command": command }).to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i < TOOL_LOOP_COARSE_WINDOW - 1 {
            assert!(matches!(signal, ToolLoopSignal::None));
        } else {
            assert!(matches!(signal, ToolLoopSignal::Coarse));
        }
    }
    assert!(supervisor.coarse_loop_note_injected);
}

#[test]
fn turn_supervisor_escalates_execute_command_git_forensics_to_hard_stop() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let commands = [
        "git log --oneline --decorate -5 && git status --short",
        "git show --stat --oneline 5dfc5676f && git show --stat --oneline 76530274f",
        "git diff --stat 5dfc5676f^ 76530274f && git diff --stat 5dfc5676f 76530274f",
        "git show --format=fuller --name-status 5dfc5676f && git show --format=fuller --name-status 76530274f",
        "git reflog -8 --date=iso --format='%h %gd %gs %cd' && git status --short --branch",
        "git diff-tree --no-commit-id --name-status -r 5dfc5676f && git diff-tree --no-commit-id --name-status -r 76530274f",
        "git show --format=fuller --name-status 76530274f -- && git status --short",
        "git log --graph --decorate --oneline -10 && git reflog -5 --date=iso",
    ];
    let mut signals = Vec::new();
    for (i, command) in commands.iter().enumerate() {
        messages.push(crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: format!("git-hard-{i}"),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "execute_command".to_string(),
                    arguments: serde_json::json!({ "command": command }).to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }
    assert!(
        signals[..TOOL_LOOP_COARSE_WINDOW - 1]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None))
    );
    assert!(matches!(
        signals[TOOL_LOOP_COARSE_WINDOW - 1],
        ToolLoopSignal::Coarse
    ));
    assert!(
        signals[TOOL_LOOP_COARSE_WINDOW..TOOL_LOOP_COARSE_HARD_WINDOW - 1]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None))
    );
    assert!(matches!(
        signals[TOOL_LOOP_COARSE_HARD_WINDOW - 1],
        ToolLoopSignal::CoarseHard
    ));
    assert!(supervisor.hard_loop_stop_injected);
}

#[test]
fn turn_supervisor_escalates_execute_command_coarse_loop_to_hard_stop() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let commands = [
        "ls -la /data01/dataagent_be/logs/ | head -30",
        "ls -lt /data01/dataagent_be/logs/ | head -20",
        "ls /data01/dataagent_be/logs/ | head -30",
        "ls -lt /data01/dataagent_be/logs/ 2>/dev/null | head -40",
        "ls -la /data01/dataagent_be/logs/ | head -50",
        "ls -lt /data01/dataagent_be/logs/ | head -10",
        "ls -la /data01/dataagent_be/logs/ 2>/dev/null | head -25",
        "ls -lt /data01/dataagent_be/logs/ | head -60",
    ];
    let mut signals = Vec::new();
    for (i, command) in commands.iter().enumerate() {
        messages.push(crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: format!("tc-hard-{i}"),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "execute_command".to_string(),
                    arguments: serde_json::json!({ "command": command }).to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        });
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }
    assert!(
        signals[..TOOL_LOOP_COARSE_WINDOW - 1]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None))
    );
    assert!(matches!(
        signals[TOOL_LOOP_COARSE_WINDOW - 1],
        ToolLoopSignal::Coarse
    ));
    assert!(
        signals[TOOL_LOOP_COARSE_WINDOW..TOOL_LOOP_COARSE_HARD_WINDOW - 1]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None))
    );
    assert!(matches!(
        signals[TOOL_LOOP_COARSE_HARD_WINDOW - 1],
        ToolLoopSignal::CoarseHard
    ));
    assert!(supervisor.hard_loop_stop_injected);
}

#[test]
fn coarse_loop_note_allows_distinct_sub_questions() {
    let mut messages = Vec::new();
    inject_coarse_loop_note(&mut messages);
    let text = messages[0].content.as_str().unwrap_or_default().to_string();
    assert!(text.contains("[low-yield-repetition]"));
    assert!(text.contains("distinct and well-defined sub-questions"));
    assert!(text.contains("not necessarily an error"));
}

#[test]
fn tool_round_checkpoints_scale_and_stop_before_hard_limit() {
    assert_eq!(TOOL_ROUND_CHECKPOINT, 24);
    assert_eq!(initial_tool_round_checkpoint(0), 1);
    assert_eq!(initial_tool_round_checkpoint(1), 1);
    assert_eq!(initial_tool_round_checkpoint(10), 5);
    assert_eq!(initial_tool_round_checkpoint(4096), TOOL_ROUND_CHECKPOINT);
    assert_eq!(tool_round_checkpoint_threshold(4096, 0), Some(24));
    assert_eq!(tool_round_checkpoint_threshold(4096, 1), Some(48));
    assert_eq!(tool_round_checkpoint_threshold(4096, 2), Some(96));
    assert_eq!(tool_round_checkpoint_threshold(10, 0), Some(5));
    assert_eq!(tool_round_checkpoint_threshold(10, 1), None);
}

#[test]
fn tool_round_checkpoints_are_staged() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    supervisor.iteration = 23;
    assert!(
        supervisor
            .maybe_inject_tool_round_checkpoint(
                &mut messages,
                4096,
                ToolRoundCheckpointPhase::Explore,
            )
            .is_none()
    );
    for (iteration, expected_level) in [(24, "review"), (48, "restrict"), (96, "finalize")] {
        supervisor.iteration = iteration;
        let checkpoint = supervisor
            .maybe_inject_tool_round_checkpoint(
                &mut messages,
                4096,
                ToolRoundCheckpointPhase::Explore,
            )
            .expect("checkpoint should fire");
        assert_eq!(checkpoint.threshold, iteration);
        assert!(
            messages
                .last()
                .and_then(|message| message.content.as_str())
                .is_some_and(|text| text.contains(&format!("level={expected_level}")))
        );
    }
    supervisor.iteration = 97;
    assert!(
        supervisor
            .maybe_inject_tool_round_checkpoint(
                &mut messages,
                4096,
                ToolRoundCheckpointPhase::Explore,
            )
            .is_none()
    );
    assert_eq!(messages.len(), 3);
}

fn checkpoint_tool_round(
    call_id: &str,
    name: &str,
    arguments: serde_json::Value,
    result: &str,
) -> Vec<crate::ai::history::Message> {
    vec![
        crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: call_id.to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        crate::ai::history::Message {
            role: "tool".to_string(),
            content: serde_json::Value::String(result.to_string()),
            tool_calls: None,
            tool_call_id: Some(call_id.to_string()),
            reasoning_content: None,
        },
    ]
}

#[test]
fn tool_round_checkpoint_phase_tracks_mutation_verification_and_failure() {
    let read_round = checkpoint_tool_round(
        "read",
        "read_file",
        serde_json::json!({"file_path": "/tmp/demo"}),
        "contents",
    );
    assert_eq!(
        tool_round_checkpoint_phase(&read_round, &read_round),
        ToolRoundCheckpointPhase::Explore
    );

    let mutation_round = checkpoint_tool_round(
        "patch",
        "apply_patch",
        serde_json::json!({"patch": "demo", "dry_run": false}),
        "Done!",
    );
    assert_eq!(
        tool_round_checkpoint_phase(&mutation_round, &mutation_round),
        ToolRoundCheckpointPhase::ImplementedNeedsVerification
    );

    let verification_round = checkpoint_tool_round(
        "check",
        "execute_command",
        serde_json::json!({"command": "cargo check --bin a"}),
        "Finished dev profile",
    );
    assert_eq!(
        tool_round_checkpoint_phase(&verification_round, &verification_round),
        ToolRoundCheckpointPhase::Explore
    );
    let mut verified_turn = mutation_round.clone();
    verified_turn.extend(verification_round.clone());
    assert_eq!(
        tool_round_checkpoint_phase(&verification_round, &verified_turn),
        ToolRoundCheckpointPhase::VerifiedNeedsFinalization
    );

    let failed_round = checkpoint_tool_round(
        "test",
        "execute_command",
        serde_json::json!({"command": "cargo test --bin a focused_test"}),
        "Exit code: 101\nfailed",
    );
    let mut failed_turn = mutation_round;
    failed_turn.extend(failed_round.clone());
    assert_eq!(
        tool_round_checkpoint_phase(&failed_round, &failed_turn),
        ToolRoundCheckpointPhase::RecoveringFromError
    );

    let mut verification_before_mutation = checkpoint_tool_round(
        "check-first",
        "execute_command",
        serde_json::json!({"command": "cargo check --bin a"}),
        "Finished dev profile",
    );
    verification_before_mutation.extend(checkpoint_tool_round(
        "patch-last",
        "apply_patch",
        serde_json::json!({"patch": "demo", "dry_run": false}),
        "Done!",
    ));
    assert_eq!(
        tool_round_checkpoint_phase(
            &verification_before_mutation,
            &verification_before_mutation,
        ),
        ToolRoundCheckpointPhase::ImplementedNeedsVerification
    );

    let failed_check = checkpoint_tool_round(
        "failed-check",
        "execute_command",
        serde_json::json!({"command": "cargo test --bin a focused_test"}),
        "Exit code: 101\nfailed",
    );
    let repair = checkpoint_tool_round(
        "repair",
        "apply_patch",
        serde_json::json!({"patch": "demo", "dry_run": false}),
        "Done!",
    );
    let passing_check = checkpoint_tool_round(
        "passing-check",
        "execute_command",
        serde_json::json!({"command": "cargo test --bin a focused_test"}),
        "test result: ok",
    );
    let mut repaired_after_failure = vec![failed_check[0].clone()];
    repaired_after_failure[0]
        .tool_calls
        .as_mut()
        .expect("assistant batch")
        .extend(repair[0].tool_calls.clone().expect("repair call"));
    repaired_after_failure[0]
        .tool_calls
        .as_mut()
        .expect("assistant batch")
        .extend(passing_check[0].tool_calls.clone().expect("passing call"));
    repaired_after_failure.extend([
        failed_check[1].clone(),
        repair[1].clone(),
        passing_check[1].clone(),
    ]);
    assert_eq!(
        tool_round_checkpoint_phase(&repaired_after_failure, &repaired_after_failure),
        ToolRoundCheckpointPhase::VerifiedNeedsFinalization
    );
}

#[test]
fn agent_team_temp_write_does_not_disable_read_only_progress_guard() {
    let temp_write_round = checkpoint_tool_round(
        "temp-write",
        "write_file",
        serde_json::json!({
            "file_path": "probe.py",
            "content": "print('ok')\n",
            "temp": true,
        }),
        "File written successfully",
    );

    assert!(!round_has_mutation(&temp_write_round));
}

#[test]
fn agent_team_delegation_does_not_disable_read_only_progress_guard() {
    let mut messages = checkpoint_tool_round(
        "spawn-1",
        "task_spawn",
        serde_json::json!({
            "description": "inspect independent branch",
            "prompt": "Inspect one independent branch and report evidence.",
        }),
        r#"{"task_id":"child-1"}"#,
    );
    assert!(round_has_mutation(&messages));
    assert!(!round_has_project_mutation(&messages));

    let mut supervisor = TurnSupervisor::default();
    supervisor.next_iteration();
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::None));
    assert!(!supervisor.progress.observed_project_mutation_this_turn);

    let mut hard_stopped = false;
    for i in 1..=(READ_ONLY_BREADTH_HARD_STOP_TARGETS + 4) {
        supervisor.next_iteration();
        pb_successful_read_round(
            &mut messages,
            &format!("src/after_delegation_{i}.rs"),
            0,
            &format!("read-{i}"),
            &format!("independent evidence {i}"),
        );
        let signal =
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if matches!(signal, ToolLoopSignal::ReadOnlyBreadthHard) {
            hard_stopped = true;
            break;
        }
    }
    assert!(
        hard_stopped,
        "delegation is progress, but must not permanently disable the read-only breadth ceiling"
    );
}

#[test]
fn agent_team_read_only_breadth_note_redirects_to_delegation() {
    let mut team_messages = Vec::new();
    inject_read_only_breadth_note(&mut team_messages, true);
    let team_note = team_messages[0]
        .content
        .as_str()
        .expect("breadth note should be text");
    assert!(team_note.contains("`agent-team` is active"));
    assert!(team_note.contains("delegate any remaining branches now"));

    let mut ordinary_messages = Vec::new();
    inject_read_only_breadth_note(&mut ordinary_messages, false);
    let ordinary_note = ordinary_messages[0]
        .content
        .as_str()
        .expect("breadth note should be text");
    assert!(!ordinary_note.contains("`agent-team` is active"));
}

// ===== Progress Budget (behavioral-signal progress budget) tests =====
// These cases deliberately make every round's tool signature differ (different
// paths) to bypass exact/coarse "signature repetition" detection and verify
// the third layer, assess_progress's "information gain" rule: a successful
// read of a new target counts as progress; a failed read (no target) does not.

fn pb_read_msg(path: &str, id: &str) -> crate::ai::history::Message {
    crate::ai::history::Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: format!("{{\"path\":\"{path}\"}}"),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn pb_successful_read_round(
    messages: &mut Vec<crate::ai::history::Message>,
    path: &str,
    offset: usize,
    id: &str,
    result: &str,
) {
    messages.push(crate::ai::history::Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: format!("{{\"path\":\"{path}\",\"offset\":{offset}}}"),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    });
    messages.push(pb_tool_result(id, result));
}

fn pb_apply_patch_msg(id: &str) -> crate::ai::history::Message {
    crate::ai::history::Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "apply_patch".to_string(),
                arguments: "{\"file_path\":\"src/foo.rs\",\"patch\":\"@@\\n-old\\n+new\"}"
                    .to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn pb_write_file_msg_with_content(
    path: &str,
    id: &str,
    content: &str,
) -> crate::ai::history::Message {
    crate::ai::history::Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "write_file".to_string(),
                arguments: serde_json::json!({
                    "file_path": path,
                    "content": content,
                })
                .to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn pb_write_file_msg(path: &str, id: &str) -> crate::ai::history::Message {
    pb_write_file_msg_with_content(path, id, &format!("updated {id}\n"))
}

fn pb_execute_command_msg(command: &str, id: &str) -> crate::ai::history::Message {
    crate::ai::history::Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "execute_command".to_string(),
                arguments: serde_json::json!({ "command": command }).to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn pb_execute_command_round(
    messages: &mut Vec<crate::ai::history::Message>,
    command: &str,
    id: &str,
) {
    messages.push(pb_execute_command_msg(command, id));
    messages.push(pb_tool_result(id, &format!("output of {command}")));
}

fn pb_read_round(messages: &mut Vec<crate::ai::history::Message>, path: &str, id: &str) {
    messages.push(pb_read_msg(path, id));
    messages.push(pb_tool_result(id, &format!("body of {path}")));
}

fn pb_task_tool_msg(
    tool_name: &str,
    args: serde_json::Value,
    id: &str,
) -> crate::ai::history::Message {
    crate::ai::history::Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: tool_name.to_string(),
                arguments: args.to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn pb_tool_result(id: &str, text: &str) -> crate::ai::history::Message {
    crate::ai::history::Message {
        role: "tool".to_string(),
        content: serde_json::Value::String(text.to_string()),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    }
}

fn pb_task_round(
    messages: &mut Vec<crate::ai::history::Message>,
    tool_name: &str,
    args: serde_json::Value,
    id: &str,
    result: &str,
) {
    messages.push(pb_task_tool_msg(tool_name, args, id));
    messages.push(pb_tool_result(id, result));
}

/// A failed read-only call round: the assistant issues read_file followed by a
/// tool result indicating the read failed. Failed calls do not enter
/// `extract_round_targets` (no targets → no information gain → no progress),
/// and because each round uses a different path they bypass exact/coarse
/// signature loop detection, making this the only no-progress driver of the
/// progress-budget escalation ladder after "unify on behavioral signals"
/// (successful new-target reads always count as progress).
fn pb_failed_read_round(messages: &mut Vec<crate::ai::history::Message>, path: &str, id: &str) {
    pb_failed_read_round_reasoning(messages, path, id, None);
}

/// Variant of `pb_failed_read_round` with reasoning: the failed read round
/// carries a reasoning snippet, used to verify grace extension (a changed
/// reasoning fingerprint after the soft notice buys one grace round).
fn pb_failed_read_round_reasoning(
    messages: &mut Vec<crate::ai::history::Message>,
    path: &str,
    id: &str,
    reasoning: Option<&str>,
) {
    let mut assistant = pb_read_msg(path, id);
    assistant.reasoning_content = reasoning.map(str::to_string);
    messages.push(assistant);
    messages.push(crate::ai::history::Message {
        role: "tool".to_string(),
        content: serde_json::Value::String("Error: File not found".to_string()),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    });
}

fn pb_dedup_read_round(messages: &mut Vec<crate::ai::history::Message>, path: &str, id: &str) {
    messages.push(pb_read_msg(path, id));
    messages.push(crate::ai::history::Message {
        role: "tool".to_string(),
        content: serde_json::Value::String(
            "[deduped: byte-identical `read_file` result already present verbatim earlier in this conversation; content unchanged since then.]\n- original_tool_call_id: later\n- canonical_tool_call_id: earlier\n- preview: fn main() {}".to_string(),
        ),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    });
}

fn pb_blocked_outside_workspace_round(
    messages: &mut Vec<crate::ai::history::Message>,
    command: &str,
    id: &str,
) {
    messages.push(pb_execute_command_msg(command, id));
    messages.push(crate::ai::history::Message {
        role: "tool".to_string(),
        content: serde_json::Value::String(
            "Error: execute_command failed: Command blocked: command references path ~/.config/mcp.json (resolves to /Users/bytedance/.config/mcp.json) which is outside the current workspace\nSuggestion: inspect files inside the current project instead.".to_string(),
        ),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    });
}

#[test]
fn progress_budget_no_gain_reading_triggers_soft_after_free_rounds() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let mut signals = Vec::new();
    // iterations 1..=25: the free zone (<=20) stays fully silent; from 21 on,
    // no-progress accumulates, and at round 25 consecutive=5 reaches
    // soft_threshold(25)=5, firing the soft notice.
    // Failed reads manufacture "no information gain" rounds: successful
    // new-target reads always count as progress and cannot accumulate
    // no-progress.
    for i in 1..=25 {
        supervisor.next_iteration();
        pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("tc-{i}"));
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }
    assert!(
        signals[..24]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None)),
        "should stay silent through free-explore + sub-threshold rounds"
    );
    assert!(matches!(signals[24], ToolLoopSignal::LowProgressSoft));
    assert!(supervisor.progress.soft_injected);
}

#[test]
fn progress_budget_same_target_paging_is_progress_but_coarse_brake_still_fires() {
    let mut supervisor = TurnSupervisor::default();
    supervisor.iteration = 30;
    let mut messages = Vec::new();

    for i in 0..12 {
        supervisor.next_iteration();
        pb_successful_read_round(
            &mut messages,
            "src/large.rs",
            i * 100,
            &format!("page-{i}"),
            &format!("lines {}..{}: unique-{i}", i * 100, i * 100 + 99),
        );
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_COARSE_WINDOW - 1 {
            // Paging the same file fills the coarse window: exact signatures
            // differ because offsets differ (no soft/hard), but the coarse
            // signatures match once offsets are stripped, so the one-shot
            // Coarse brake should still fire. This is the key guarantee of the
            // "progress hashing must ignore offset/limit to prevent budget
            // escapes" invariant.
            assert!(
                matches!(signal, ToolLoopSignal::Coarse),
                "round {i}: same-file paging must still trip the coarse brake: {signal:?}"
            );
        } else {
            assert!(
                matches!(signal, ToolLoopSignal::None),
                "round {i}: {signal:?}"
            );
        }
        // New evidence always counts as progress: Progress Budget never
        // escalates to soft/ledger/hard because of efficient paging.
        assert_eq!(supervisor.progress.consecutive_no_progress, 0);
    }

    // Coarse notices once, and new evidence must not clear the signature
    // history (otherwise the window never fills and the brake fails).
    assert!(
        supervisor.coarse_loop_note_injected,
        "coarse paging brake must have fired exactly once"
    );
    assert_eq!(supervisor.progress.seen_targets.len(), 1);
    // Across the 12 paging rounds, the coarse-hit round (i=COARSE_WINDOW-1)
    // early-returns before entering assess_progress and its evidence is not
    // recorded; the other 11 rounds each record one distinct fingerprint.
    assert_eq!(supervisor.progress.seen_evidence_fingerprints.len(), 11);
}

#[test]
fn progress_budget_readonly_novel_targets_warn_once_after_breadth_threshold() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let rounds = READ_ONLY_BREADTH_CHECK_TARGETS.max(PROGRESS_FREE_EXPLORE_ROUNDS) + 2;
    let mut breadth_warnings = 0;
    for i in 1..=rounds {
        supervisor.next_iteration();
        messages.push(pb_read_msg(&format!("src/f{i}.rs"), &format!("tc-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        match signal {
            ToolLoopSignal::ReadOnlyBreadth => {
                breadth_warnings += 1;
                assert!(i > PROGRESS_FREE_EXPLORE_ROUNDS);
                assert!(i >= READ_ONLY_BREADTH_CHECK_TARGETS);
            }
            ToolLoopSignal::None => {}
            _ => panic!("fresh read-only targets must not trigger no-progress signal"),
        }
    }
    assert_eq!(breadth_warnings, 1);
    assert_eq!(supervisor.progress.consecutive_no_progress, 0);
}

/// Pure read-only divergence (zero mutations, constantly switching to new
/// target reads) must be forcibly wrapped up once the accumulated target count
/// reaches `READ_ONLY_BREADTH_HARD_STOP_TARGETS` — even though every round
/// reads a new target / new bytes, repeatedly resetting the regular
/// no-progress counter. This criterion is decoupled from content novelty and
/// is the final brake for the "diagnostic agent diverging read-only for two
/// hours without wrapping up" incident.
#[test]
fn progress_budget_pure_readonly_divergence_hard_stops_after_breadth_ceiling() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let mut saw_hard_stop_at = None;
    for i in 1..=(READ_ONLY_BREADTH_HARD_STOP_TARGETS + 4) {
        supervisor.next_iteration();
        // Every round reads a brand-new file and gets a successful result: both
        // a new target and new evidence, so the regular no-progress counter
        // stays 0 and only the decoupled breadth hard stop can catch it.
        pb_successful_read_round(
            &mut messages,
            &format!("src/probe_{i}.rs"),
            0,
            &format!("tc-{i}"),
            &format!("unique content for probe {i}"),
        );
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if matches!(signal, ToolLoopSignal::ReadOnlyBreadthHard) {
            saw_hard_stop_at = Some(i);
            break;
        }
    }
    assert_eq!(
        saw_hard_stop_at,
        Some(READ_ONLY_BREADTH_HARD_STOP_TARGETS),
        "pure read-only divergence must hard-stop exactly at the breadth ceiling"
    );
    assert_eq!(
        supervisor.progress.consecutive_no_progress, 0,
        "hard stop must be independent of the (reset-every-round) no-progress counter"
    );
}

/// Once any mutation happened in this turn, the breadth hard stop stands down
/// permanently: normal "read many files + edit" implementation tasks are not
/// subject to this brake, avoiding degraded agent effectiveness.
#[test]
fn progress_budget_breadth_hard_stop_never_fires_after_mutation() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    // Perform one successful project mutation in round 1 to set
    // observed_project_mutation_this_turn.
    supervisor.next_iteration();
    messages.push(pb_apply_patch_msg("patch-1"));
    messages.push(pb_tool_result(
        "patch-1",
        "Successfully patched src/lib.rs; +3 -1 (2 lines)",
    ));
    let _ = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(supervisor.progress.observed_project_mutation_this_turn);

    // Subsequent read-only exploration far exceeds the hard-stop threshold yet
    // must never trigger ReadOnlyBreadthHard.
    for i in 1..=(READ_ONLY_BREADTH_HARD_STOP_TARGETS + 8) {
        supervisor.next_iteration();
        pb_successful_read_round(
            &mut messages,
            &format!("src/after_mut_{i}.rs"),
            0,
            &format!("rc-{i}"),
            &format!("post-mutation probe {i}"),
        );
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(
            !matches!(signal, ToolLoopSignal::ReadOnlyBreadthHard),
            "breadth hard-stop must never fire once a mutation happened this turn (round {i})"
        );
    }
    assert!(!supervisor.progress.read_only_breadth_hard_injected);
}

#[test]
fn progress_budget_does_not_inject_readonly_breadth_after_mutation() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();

    // First accumulate to one step below the breadth threshold, simulating a
    // state where much evidence has been read but ReadOnlyBreadth has not yet
    // fired.
    for i in 1..READ_ONLY_BREADTH_CHECK_TARGETS {
        supervisor.next_iteration();
        messages.push(pb_read_msg(&format!("src/f{i}.rs"), &format!("tc-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(
            matches!(signal, ToolLoopSignal::None),
            "pre-threshold read-only exploration should stay silent: {signal:?}"
        );
    }

    supervisor.next_iteration();
    messages.push(pb_apply_patch_msg("patch-after-breadth"));
    messages.push(pb_tool_result(
        "patch-after-breadth",
        "Patch applied successfully.",
    ));
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);

    assert!(
        matches!(signal, ToolLoopSignal::None),
        "mutation progress must not inject read-only breadth convergence prompt: {signal:?}"
    );
    assert!(!supervisor.progress.read_only_breadth_injected);
    assert_eq!(supervisor.progress.consecutive_no_progress, 0);
}

#[test]
fn progress_budget_ignores_failed_readonly_targets() {
    let mut supervisor = TurnSupervisor::default();
    supervisor.iteration = 30;
    let mut messages = Vec::new();
    let mut last = ToolLoopSignal::None;

    for i in 0..5 {
        pb_failed_read_round(
            &mut messages,
            &format!("src/missing-{i}.rs"),
            &format!("failed-{i}"),
        );
        last = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    }

    assert!(matches!(last, ToolLoopSignal::LowProgressSoft));
    assert!(supervisor.progress.seen_targets.is_empty());
}

#[test]
fn progress_budget_ignores_dedup_only_read_targets() {
    let mut supervisor = TurnSupervisor::default();
    supervisor.iteration = 30;
    let mut messages = Vec::new();
    let mut last = ToolLoopSignal::None;

    for i in 0..5 {
        pb_dedup_read_round(
            &mut messages,
            &format!("src/repeated-{i}.rs"),
            &format!("dedup-{i}"),
        );
        last = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    }

    assert!(matches!(last, ToolLoopSignal::LowProgressSoft));
    assert!(
        supervisor.progress.seen_targets.is_empty(),
        "dedup-only stubs should not count as fresh evidence targets"
    );
}

#[test]
fn blocked_outside_workspace_command_normalizes_to_stable_target() {
    let mut messages = Vec::new();
    pb_blocked_outside_workspace_round(&mut messages, "cat ~/.config/mcp.json", "blocked-1");

    assert_eq!(
        extract_round_targets(&messages),
        vec![
            "execute_command:blocked-outside-workspace:/Users/bytedance/.config/mcp.json"
                .to_string()
        ]
    );
}

#[test]
fn browser_navigation_and_read_count_as_progress_targets() {
    // Browser tools are not in MUTATION_TOOL_NAMES and their args carry no
    // path/query; without extracting url/selector, navigating to a new page or
    // reading a new selector would be misjudged as no progress, and a normal
    // multi-step browsing turn would be wrongly stopped by LowProgressHard
    // under the progress-budget ladder.
    let mut messages = Vec::new();
    pb_task_round(
        &mut messages,
        "mcp_browser_navigate",
        serde_json::json!({ "url": "https://example.com/page" }),
        "nav-1",
        "ok",
    );
    assert_eq!(
        extract_round_targets(&messages),
        vec!["mcp_browser_navigate:url:https://example.com/page".to_string()]
    );

    let mut read_messages = Vec::new();
    pb_task_round(
        &mut read_messages,
        "mcp_browser_get_text",
        serde_json::json!({ "selector": "#main" }),
        "read-1",
        "hello",
    );
    assert_eq!(
        extract_round_targets(&read_messages),
        vec!["mcp_browser_get_text:selector:#main".to_string()]
    );
}

#[test]
fn target_repeat_catches_repeated_blocked_outside_workspace_commands() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let commands = [
        "cat ~/.config/mcp.json",
        "grep api ~/.config/mcp.json",
        "head -20 ~/.config/mcp.json",
        "tail -20 ~/.config/mcp.json",
        "wc -l ~/.config/mcp.json",
    ];
    let mut signals = Vec::new();
    for (i, command) in commands.iter().enumerate() {
        pb_blocked_outside_workspace_round(&mut messages, command, &format!("blocked-{i}"));
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }

    assert!(
        signals[..TOOL_LOOP_COARSE_WINDOW - 1]
            .iter()
            .all(|signal| matches!(signal, ToolLoopSignal::None)),
        "blocked command variants should not trigger earlier exact/coarse loops: {signals:?}"
    );
    assert!(matches!(
        signals[TOOL_LOOP_COARSE_WINDOW - 1],
        ToolLoopSignal::TargetRepeat
    ));
    assert!(supervisor.target_repeat_note_injected);
}

/// Repeated evidence gathering on the same target across mixed tool rounds:
/// every round reads the same file A, interleaved with a tree that differs
/// every round, so whole-round exact/coarse signatures never match and
/// detect_tool_loop is bypassed; the target-intersection check should catch
/// "A present every round" and emit one TargetRepeat.
#[test]
fn turn_supervisor_emits_target_repeat_for_mixed_tool_rounds_on_same_file() {
    fn mixed_round(i: usize) -> crate::ai::history::Message {
        crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(String::new()),
            tool_calls: Some(vec![
                // Constant every round: re-read the same file A.
                crate::ai::types::ToolCall {
                    id: format!("read-A-{i}"),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall {
                        name: "read_file".to_string(),
                        arguments: "{\"path\":\"src/bin/ai/mod.rs\",\"offset\":100}".to_string(),
                    },
                },
                // A different filler directory read each round: keeps
                // whole-round signatures unequal, escaping whole-round equality
                // checks.
                crate::ai::types::ToolCall {
                    id: format!("search-{i}"),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall {
                        name: "tree".to_string(),
                        arguments: format!("{{\"path\":\"src/probe_{i}\"}}"),
                    },
                },
            ]),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let mut signals = Vec::new();
    for i in 0..TOOL_LOOP_COARSE_WINDOW {
        messages.push(mixed_round(i));
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }

    // Whole-round signatures differ every round: exact / coarse whole-round
    // equality never hits.
    assert!(
        signals[..TOOL_LOOP_COARSE_WINDOW - 1]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None)),
        "whole-round signatures differ every round; nothing should fire early: {signals:?}"
    );
    // When the coarse window fills, the target intersection (file A) hits and
    // emits one TargetRepeat.
    assert!(
        matches!(
            signals[TOOL_LOOP_COARSE_WINDOW - 1],
            ToolLoopSignal::TargetRepeat
        ),
        "same-file across mixed rounds must trigger TargetRepeat: {signals:?}"
    );
    assert!(supervisor.target_repeat_note_injected);
}

/// Counter-case guard: every round reads a different file (no common target),
/// the target intersection is empty, and TargetRepeat must not be reported.
#[test]
fn target_repeat_ignores_distinct_targets_each_round() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let mut signals = Vec::new();
    for i in 0..TOOL_LOOP_COARSE_WINDOW {
        messages.push(pb_read_msg(&format!("src/f{i}.rs"), &format!("tc-{i}")));
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }
    assert!(
        signals
            .iter()
            .all(|s| !matches!(s, ToolLoopSignal::TargetRepeat)),
        "distinct targets each round must not trigger TargetRepeat: {signals:?}"
    );
    assert!(!supervisor.target_repeat_note_injected);
}

#[test]
fn target_repeat_does_not_fire_on_write_file_progress() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let target = "main-test/zi_ping.txt";

    for i in 0..TOOL_LOOP_COARSE_WINDOW {
        let id = format!("write-{i}");
        messages.push(pb_write_file_msg(target, &id));
        messages.push(pb_tool_result(&id, "Successfully wrote file."));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(
            matches!(signal, ToolLoopSignal::None),
            "successful write_file progress must not trigger TargetRepeat: {signal:?}"
        );
    }

    assert!(!supervisor.target_repeat_note_injected);
    assert!(supervisor.tool_target_history.iter().all(Vec::is_empty));
    assert_eq!(supervisor.progress.consecutive_no_progress, 0);
}

#[test]
fn target_rescan_catches_pagination_loop_with_mixed_rounds() {
    // Reproduces the real escape pattern of session f319d490: the same file is
    // read from the top repeatedly (page width changes every round: 2400B →
    // 1400B → read_file line paging), interleaved with new archive paths that
    // differ every round — whole-round signatures never match and all three
    // whole-round detectors (exact/coarse/target-repeat) miss; the rescan
    // counter accumulates per target, decoupled from whole-round signatures:
    // the 3rd from-top read injects the soft notice, the 4th hard-stops.
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let file = "src/all_desc.txt";

    // 1st from-top read (tail -c +1) plus subsequent paging.
    pb_execute_command_round(&mut messages, &format!("tail -c +1 {file} | head -c 2400"), "t0");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    );
    for i in 1..3 {
        let offset = i * 2400 + 1;
        pb_execute_command_round(
            &mut messages,
            &format!("tail -c +{offset} {file} | head -c 2400"),
            &format!("t0p{i}"),
        );
        assert_eq!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::None
        );
    }
    // Interleave new archive-path reads that differ every round (the key to
    // escaping the previous three detectors).
    pb_read_round(&mut messages, &format!("{file}.archive-1"), "a1");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    );
    // 2nd from-top read: page width switched to 1400 bytes → soft notice (soft
    // threshold=2).
    pb_execute_command_round(&mut messages, &format!("tail -c +1 {file} | head -c 1400"), "t1");
    assert!(
        matches!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::TargetRescan(..)
        ),
        "round t1 应软提示：第 2 次从头读"
    );
    pb_execute_command_round(&mut messages, &format!("tail -c +1401 {file} | head -c 1400"), "t1p1");
    // After the fix: tail/sed offset/line-range literals are stripped, the
    // window [tail,tail,archive,tail,tail] folds into the 3-cycle
    // [tail,tail,archive], and coarse hits early when the window fills
    // (catching this paging + interleaved-rounds loop before the from-top
    // rescan does).
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(
        matches!(signal, ToolLoopSignal::Coarse),
        "paging+archive mixed window should now hit Coarse, got {signal:?}"
    );
    pb_read_round(&mut messages, &format!("{file}.archive-2"), "a2");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    );
    // 3rd from-top read (read_file offset omitted) → no new signal (the soft
    // notice was already injected in this episode).
    pb_read_round(&mut messages, file, "r3");
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert_eq!(
        signal,
        ToolLoopSignal::None,
        "3rd from-top read: soft already injected this episode (fires once), got {signal:?}"
    );
    // 4th from-top read (offset=0) → hard stop.
    pb_successful_read_round(&mut messages, file, 0, "r4", "content page");
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(
        matches!(signal, ToolLoopSignal::TargetRescanHard(..)),
        "expected TargetRescanHard at 4th from-top read, got {signal:?}"
    );
}

#[test]
fn target_rescan_resets_on_write_file_edit() {
    // Re-reading from the top after editing is legitimate verification:
    // write_file modifying the target resets the rescan counter to zero, and a
    // read→edit→read cycle must not trigger any rescan signal.
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let target = "src/config.toml";
    for i in 0..3 {
        pb_read_round(&mut messages, target, &format!("r{i}"));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert_eq!(signal, ToolLoopSignal::None, "round r{i} must not signal");
        let id = format!("w{i}");
        messages.push(pb_write_file_msg(target, &id));
        messages.push(pb_tool_result(&id, "Successfully wrote file."));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert_eq!(signal, ToolLoopSignal::None, "round w{i} must not signal");
    }
    assert!(
        !supervisor.progress.from_top_reads.contains_key(target),
        "write_file must reset the from-top counter"
    );
}

#[test]
fn target_rescan_ignores_monotonic_pagination() {
    // Monotonically forward paging (offsets increase, never revisits) is not a
    // rescan: only 1 from-top read counted.
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let file = "src/big-file.rs";
    for i in 0..8 {
        let offset = i * 100;
        pb_successful_read_round(&mut messages, file, offset, &format!("p{i}"), &format!("page-{i}"));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(
            !matches!(
                signal,
                ToolLoopSignal::TargetRescan(..) | ToolLoopSignal::TargetRescanHard(..)
            ),
            "monotonic pagination must not trigger rescan at round {i}: {signal:?}"
        );
    }
}

#[test]
fn target_rescan_window_decays_stale_counts() {
    // Legitimate re-reads spanning many rounds after context compression must
    // not accumulate: when more than TARGET_RESCAN_WINDOW_ROUNDS=8 rounds have
    // passed since the last from-top read, the counter expires and restarts
    // from 1.
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let target = "src/spaced-file.rs";
    // 1st from-top read at iteration=1 → count 1.
    supervisor.iteration = 1;
    pb_execute_command_round(&mut messages, &format!("tail -c +1 {target} | head -c 2400"), "r1");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    );
    // Another from-top read after 10 rounds (gap=10 > 8) → the counter
    // expired, restarts from 1, nothing fires.
    supervisor.iteration = 11;
    pb_execute_command_round(&mut messages, &format!("cat {target}"), "r11");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    );
    // Consecutive re-reads inside the window: commands differ every round
    // (tail/cat/sed/nl/head) to avoid tripping exact/coarse loop detection.
    // The 2nd (cumulative 2) soft notice, the 3rd no new signal (soft already
    // injected), the 4th hard stop.
    supervisor.iteration = 12;
    pb_execute_command_round(&mut messages, &format!("sed -n 1,40p {target}"), "r12");
    assert!(
        matches!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::TargetRescan(..)
        ),
        "round r12: 第 2 次（累计 2）软提示"
    );
    supervisor.iteration = 13;
    pb_execute_command_round(&mut messages, &format!("nl -ba {target}"), "r13");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "round r13: 第 3 次（累计 3）软提示已在本 episode 注入，不再重复"
    );
    supervisor.iteration = 14;
    pb_execute_command_round(&mut messages, &format!("head -n 200 {target}"), "r14");
    assert!(
        matches!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::TargetRescanHard(..)
        ),
        "round r14 should hard-stop"
    );
}

#[test]
fn target_rescan_window_decay_grants_fresh_soft_warning_per_episode() {
    // Regression: decay opens a new re-read episode, and every loop segment
    // must get its own soft warning. The old implementation only reset the
    // counter on decay without clearing rescan_note_injected, so when the
    // second segment accumulated to soft, insert returned false → the soft
    // notice was skipped and it went straight to a hard stop (the soft→hard
    // escalation invariant was broken).
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let target = "src/episode-file.rs";
    // Episode 1: 1st no trigger, 2nd soft notice with rescan_note_injected
    // recorded, 3rd no new signal.
    for (iter, cmd) in [
        (1, format!("tail -c +1 {target} | head -c 2400")),
        (2, format!("cat {target}")),
        (3, format!("sed -n 1,40p {target}")),
    ] {
        supervisor.iteration = iter;
        pb_execute_command_round(&mut messages, &cmd, &format!("r{iter}"));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if iter == 2 {
            assert!(
                matches!(signal, ToolLoopSignal::TargetRescan(..)),
                "episode1 round r2 should soft-signal (soft threshold=2)"
            );
        } else if iter == 3 {
            assert_eq!(signal, ToolLoopSignal::None, "episode1 round r3: soft injected once");
        } else {
            assert_eq!(signal, ToolLoopSignal::None, "episode1 round r{iter}");
        }
    }
    // After 9 rounds (gap=9 > TARGET_RESCAN_WINDOW_ROUNDS=8) → the counter
    // decays and restarts from 1, and the soft-notice flag must be cleared
    // together with the episode (the point of this fix).
    supervisor.iteration = 12;
    pb_execute_command_round(&mut messages, &format!("nl -ba {target}"), "r12");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "decay round r12"
    );
    // Episode 2: the 2nd occurrence soft-notices again — this segment's own
    // warning; soft must not be skipped in favor of a hard stop just because
    // the previous segment already warned; the 3rd occurrence gives no new
    // signal.
    supervisor.iteration = 13;
    pb_execute_command_round(&mut messages, &format!("head -n 200 {target}"), "r13");
    assert!(
        matches!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::TargetRescan(..)
        ),
        "episode2 round r13 should get its own soft warning"
    );
    supervisor.iteration = 14;
    pb_execute_command_round(&mut messages, &format!("less {target}"), "r14");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "episode2 round r14: soft already injected this episode"
    );
}

#[test]
fn target_rescan_resets_on_other_file_mutation() {
    // Multi-file workflow: editing any target (not only the re-read one)
    // clears the whole rescan table, so a later verification from-top re-read
    // of another file restarts from 1 and must not be misjudged as a paging
    // loop.
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let target = "src/read-a.rs";
    // From-top reads with two distinct signatures (offset 0 / 1), each
    // accumulating once.
    for (offset, id) in [(0usize, "r1"), (1usize, "r2")] {
        supervisor.iteration = id.trim_start_matches('r').parse().unwrap();
        pb_successful_read_round(&mut messages, target, offset, id, "line:0");
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if id == "r1" {
            assert_eq!(signal, ToolLoopSignal::None, "round {id}");
        } else {
            assert!(
                matches!(signal, ToolLoopSignal::TargetRescan(..)),
                "round {id}: 第 2 次从头读软提示"
            );
        }
    }
    // Round 3: edit the other file B → the table clears A's counts.
    supervisor.iteration = 4;
    let id = "w4";
    messages.push(pb_write_file_msg("src/edit-b.rs", id));
    messages.push(pb_tool_result(id, "Successfully wrote file."));
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "edit-other-file round must not signal"
    );
    // After the edit, from-top re-reads of A restart from 1: cat / offset=0 /
    // offset=1 are three distinct signatures. The 2nd soft notice, the 3rd no
    // new signal (soft already injected).
    supervisor.iteration = 5;
    pb_execute_command_round(&mut messages, &format!("cat {target}"), "c5");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "round r5"
    );
    supervisor.iteration = 6;
    pb_successful_read_round(&mut messages, target, 0, "r6", "line:0");
    assert!(
        matches!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::TargetRescan(..)
        ),
        "round r6 should soft-signal (编辑后第 2 次从头读)"
    );
    supervisor.iteration = 7;
    pb_successful_read_round(&mut messages, target, 1, "r7", "line:0");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "round r7: soft already injected this episode"
    );
}

#[test]
fn target_rescan_helper_detects_from_top_reads() {
    use crate::ai::driver::turn_runtime::orchestrator::{
        command_reads_from_top, extract_round_from_top_read_targets, first_path_token,
        normalize_rescan_path,
    };
    for from_top in [
        "cat /tmp/a.txt",
        "head -c 2400 /tmp/a.txt",
        "less /tmp/a.txt",
        "more /tmp/a.txt",
        "view /tmp/a.txt",
        "nl -ba /tmp/a.txt",
        "tail -c +1 /tmp/a.txt | head -c 2400",
        "tail -n +1 /tmp/a.txt",
        "sed -n '1,40p' /tmp/a.txt",
        "sed -n 1,40p /tmp/a.txt",
    ] {
        assert!(command_reads_from_top(from_top), "expected from-top: {from_top}");
    }
    for not_from_top in [
        "tail -c +2401 /tmp/a.txt | head -c 2400",
        "tail -n +40 /tmp/a.txt",
        "sed -n '40,80p' /tmp/a.txt",
        "grep -n pattern /tmp/a.txt",
        "ls /tmp",
    ] {
        assert!(!command_reads_from_top(not_from_top), "expected NOT from-top: {not_from_top}");
    }
    assert_eq!(
        first_path_token("tail -c +1 /tmp/a.txt | head -c 2400").as_deref(),
        Some("/tmp/a.txt")
    );
    assert_eq!(first_path_token("head -c 2400 /tmp/a.txt").as_deref(), Some("/tmp/a.txt"));
    assert_eq!(normalize_rescan_path("./src/a.rs"), normalize_rescan_path("src/a.rs"));
    // read_file offset omitted/0/1 → from-top; offset=100 → not.
    let mut messages = Vec::new();
    messages.push(pb_read_msg("/tmp/a.txt", "no-offset"));
    assert_eq!(
        extract_round_from_top_read_targets(&messages),
        vec!["/tmp/a.txt".to_string()]
    );
    let mut messages = Vec::new();
    pb_successful_read_round(&mut messages, "/tmp/a.txt", 100, "off-100", "body");
    assert!(extract_round_from_top_read_targets(&messages).is_empty());
}

#[test]
fn repeated_identical_write_file_still_hits_exact_tool_loop() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let mut signals = Vec::new();

    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        let id = format!("write-same-{i}");
        messages.push(pb_write_file_msg_with_content(
            "main-test/zi_ping.txt",
            &id,
            "same content\n",
        ));
        messages.push(pb_tool_result(&id, "Successfully wrote file."));
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }

    assert!(
        signals[..TOOL_LOOP_SOFT_WINDOW - 1]
            .iter()
            .all(|signal| matches!(signal, ToolLoopSignal::None)),
        "identical write_file calls should stay quiet before soft window fills: {signals:?}"
    );
    assert!(
        matches!(signals[TOOL_LOOP_SOFT_WINDOW - 1], ToolLoopSignal::Soft),
        "identical write_file calls should still be caught by exact loop detection: {signals:?}"
    );
}

/// Regression: repeatedly writing the same "sandbox boundary rejected" path
/// (different content each round, same file_path) used to be miscounted as
/// mutation-progress and, never entering the target history, escaped every
/// loop guard. After the fix: blocked writes no longer count as progress, are
/// normalized into a stable target, and the target-repeat guard hits within a
/// few rounds.
#[test]
fn repeated_blocked_write_to_same_path_triggers_target_repeat() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let mut signals = Vec::new();
    let blocked = "Error: write_file failed: File error (/out/picks.json): Write blocked: \
         path '/out/picks.json' is outside the allowed write directory (effective_cwd).\n\
         Writable root: '/work'. Do NOT retry the same absolute path.";

    for i in 0..TOOL_LOOP_COARSE_WINDOW {
        let id = format!("blocked-write-{i}");
        // content differs every round: whole-round exact/coarse signatures
        // never match, escaping detect_tool_loop.
        messages.push(pb_write_file_msg_with_content(
            "/out/picks.json",
            &id,
            &format!("attempt {i} content\n"),
        ));
        messages.push(pb_tool_result(&id, blocked));
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }

    assert!(
        signals[..TOOL_LOOP_COARSE_WINDOW - 1]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None)),
        "varying-content blocked writes must not fire exact/coarse loops early: {signals:?}"
    );
    assert!(
        matches!(
            signals[TOOL_LOOP_COARSE_WINDOW - 1],
            ToolLoopSignal::TargetRepeat
        ),
        "repeated blocked write to same path must trigger TargetRepeat: {signals:?}"
    );
    assert!(supervisor.target_repeat_note_injected);
}

/// Regression: a blocked write must not zero the no-progress budget.
/// Previously `_ => true` counted failed writes as mutations too, resetting
/// consecutive_no_progress on every retry so the progress budget never
/// escalated.
#[test]
fn blocked_write_does_not_count_as_mutation_progress() {
    let blocked = "Error: write_file failed: File error (/out/x.json): Write blocked: \
         path '/out/x.json' is outside the allowed write directory (effective_cwd).";
    let mut messages = Vec::new();
    messages.push(pb_write_file_msg_with_content(
        "/out/x.json",
        "w1",
        "data\n",
    ));
    messages.push(pb_tool_result("w1", blocked));
    assert!(
        !round_has_mutation(&messages),
        "a blocked write must not be counted as mutation progress"
    );

    // Control: a successful write still counts as a mutation.
    let mut ok = Vec::new();
    ok.push(pb_write_file_msg_with_content(
        "in-root.json",
        "w2",
        "data\n",
    ));
    ok.push(pb_tool_result(
        "w2",
        "Successfully wrote to /work/in-root.json",
    ));
    assert!(
        round_has_mutation(&ok),
        "a successful write must still count as mutation progress"
    );
}

#[test]
fn write_blocked_outside_root_path_parses_and_classifies() {
    let text = "Error: write_file failed: File error (/out/p.json): Write blocked: \
         path '/out/p.json' is outside the allowed write directory (effective_cwd).\n\
         Writable root: '/work'.";
    assert_eq!(
        write_blocked_outside_root_path(text).as_deref(),
        Some("/out/p.json")
    );
    assert_eq!(
        classify_tool_result_progress(text),
        ToolResultProgressStatus::BlockedOutsideWorkspace("/out/p.json".to_string())
    );
    // An ordinary failure (not write-blocked) is still classified as Failure.
    assert_eq!(
        classify_tool_result_progress("Error: read_file failed: File not found"),
        ToolResultProgressStatus::Failure
    );
    // Text without the marker does not match.
    assert!(write_blocked_outside_root_path("Successfully wrote to /x").is_none());
}

#[test]
fn target_repeat_loop_note_mentions_reuse_over_reprobe() {
    let mut messages = Vec::new();
    inject_target_repeat_loop_note(&mut messages);
    let text = messages[0].content.as_str().unwrap_or_default().to_string();
    assert!(text.contains("[low-yield-repetition]"));
    assert!(text.contains("the same target"));
    assert!(text.contains("checking the same thing with another tool"));
}

#[test]
fn progress_budget_mutation_action_resets_no_progress() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    // Pin inside the metered zone (iteration=30 → soft_threshold=5).
    supervisor.iteration = 30;
    for i in 0..4 {
        pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(matches!(signal, ToolLoopSignal::None));
    }
    assert_eq!(supervisor.progress.consecutive_no_progress, 4);
    // One real mutation action: the no-progress counter resets to zero.
    messages.push(pb_apply_patch_msg("patch-1"));
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::None));
    assert_eq!(supervisor.progress.consecutive_no_progress, 0);
}

#[test]
fn progress_budget_uses_pre_compress_current_round_for_apply_patch_progress() {
    let mut supervisor = TurnSupervisor::default();
    let mut compressed_messages = Vec::new();
    // Pin inside the metered zone with 4 no-progress rounds pre-seeded; if the
    // next round were still judged on the compressed view, LowProgressSoft
    // would fire. The real current round is apply_patch and must reset the
    // counter based on the original tool round.
    supervisor.iteration = 30;
    supervisor.progress.consecutive_no_progress = 4;

    pb_failed_read_round(
        &mut compressed_messages,
        "src/missing.rs",
        "read-after-compress",
    );

    let mut current_round = Vec::new();
    current_round.push(pb_apply_patch_msg("patch-current"));
    current_round.push(pb_tool_result(
        "patch-current",
        "Patch applied successfully.",
    ));

    let signal = supervisor.record_tool_signatures_for_progress(
        &compressed_messages,
        &current_round,
        PROGRESS_FREE_EXPLORE_ROUNDS,
    );

    assert!(
        matches!(signal, ToolLoopSignal::None),
        "apply_patch in the raw current round must not be hidden by compressed messages"
    );
    assert_eq!(supervisor.progress.consecutive_no_progress, 0);
}

#[test]
fn task_wait_status_idle_results_do_not_count_as_mutation_progress() {
    let idle_results = [
        (
            "task_wait",
            serde_json::json!({ "task_ids": ["task-1"], "timeout_secs": 1 }),
            "[task_wait] All 1 referenced task(s) already completed and their results were delivered by an earlier task_wait call. No tasks remain to wait on; continue reasoning with the results you already collected.",
        ),
        (
            "task_wait",
            serde_json::json!({ "task_ids": ["task-1"], "wait_policy": "all" }),
            "[task_wait PARKED] Yielded CPU so 1 pending subagent task(s) can run. This is normal cooperative scheduling, NOT a timeout and NOT a stall.",
        ),
        (
            "task_wait",
            serde_json::json!({ "task_ids": ["task-1"], "timeout_secs": 1 }),
            "[task_wait BUDGET ELAPSED] 1 pending subagent task(s) still running in the background. wait_policy=all, timeout_secs=1.",
        ),
        (
            "task_status",
            serde_json::json!({}),
            "No async tasks currently tracked.",
        ),
    ];

    for (idx, (tool_name, args, result)) in idle_results.into_iter().enumerate() {
        let mut messages = Vec::new();
        pb_task_round(
            &mut messages,
            tool_name,
            args,
            &format!("task-idle-{idx}"),
            result,
        );
        assert!(
            !round_has_mutation(&messages),
            "{tool_name} idle result must not reset progress budget: {result}"
        );
    }
}

#[test]
fn task_wait_status_delivered_task_output_counts_as_mutation_progress() {
    let delivered =
        "[Task: inspect driver via explorer @ sonnet] SUCCESS after 0.1s\nConfirmed result.";
    let status_delivered = format!(
        "TaskID              PID      Agent          Model          State       Description\n\
         task-1              42       explorer       sonnet         completed   inspect\n\n\
         Completed task results below (already collected — no need to wait for these):\n{delivered}"
    );
    let cases = [
        (
            "task_wait",
            serde_json::json!({ "task_ids": ["task-1"] }),
            delivered,
        ),
        (
            "task_status",
            serde_json::json!({}),
            status_delivered.as_str(),
        ),
    ];

    for (idx, (tool_name, args, result)) in cases.into_iter().enumerate() {
        let mut messages = Vec::new();
        pb_task_round(
            &mut messages,
            tool_name,
            args,
            &format!("task-result-{idx}"),
            result,
        );
        assert!(
            round_has_mutation(&messages),
            "{tool_name} with collected subagent output must count as progress"
        );
    }
}

#[test]
fn task_wait_status_idle_polling_does_not_reset_progress_budget() {
    let mut supervisor = TurnSupervisor::default();
    supervisor.iteration = 30;
    let mut messages = Vec::new();

    for i in 0..4 {
        pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(matches!(signal, ToolLoopSignal::None));
    }
    assert_eq!(supervisor.progress.consecutive_no_progress, 4);

    pb_task_round(
        &mut messages,
        "task_wait",
        serde_json::json!({ "task_ids": ["task-1"], "timeout_secs": 1 }),
        "task-wait-idle",
        "[task_wait BUDGET ELAPSED] 1 pending subagent task(s) still running in the background. wait_policy=all, timeout_secs=1.",
    );
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::LowProgressSoft));
    assert_eq!(supervisor.progress.consecutive_no_progress, 5);
}

#[test]
fn task_wait_status_delivered_result_resets_progress_budget() {
    let mut supervisor = TurnSupervisor::default();
    supervisor.iteration = 30;
    let mut messages = Vec::new();

    for i in 0..4 {
        pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(matches!(signal, ToolLoopSignal::None));
    }
    assert_eq!(supervisor.progress.consecutive_no_progress, 4);

    pb_task_round(
        &mut messages,
        "task_status",
        serde_json::json!({}),
        "task-status-result",
        "TaskID              PID      Agent          Model          State       Description\n\
         task-1              42       explorer       sonnet         completed   inspect\n\n\
         Completed task results below (already collected — no need to wait for these):\n\
         [Task: inspect driver via explorer @ sonnet] SUCCESS after 0.1s\nConfirmed result.",
    );
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::None));
    assert_eq!(supervisor.progress.consecutive_no_progress, 0);
}

#[test]
fn task_wait_status_coarse_signatures_ignore_polling_noise() {
    let wait_a = pb_task_tool_msg(
        "task_wait",
        serde_json::json!({
            "task_ids": ["task-b", "task-a", "task-a"],
            "timeout_secs": 1,
            "wait_policy": "any"
        }),
        "wait-a",
    );
    let wait_b = pb_task_tool_msg(
        "task_wait",
        serde_json::json!({
            "task_ids": ["task-a", "task-b"],
            "timeout_secs": 600,
            "wait_policy": "all"
        }),
        "wait-b",
    );
    assert_eq!(
        extract_round_tool_signatures_coarse(&[wait_a]).unwrap(),
        extract_round_tool_signatures_coarse(&[wait_b]).unwrap()
    );

    let status_a = pb_task_tool_msg(
        "task_status",
        serde_json::json!({ "noise": "a" }),
        "status-a",
    );
    let status_b = pb_task_tool_msg(
        "task_status",
        serde_json::json!({ "noise": "b", "limit": 10 }),
        "status-b",
    );
    assert_eq!(
        extract_round_tool_signatures_coarse(&[status_a]).unwrap(),
        extract_round_tool_signatures_coarse(&[status_b]).unwrap()
    );
}

#[test]
fn progress_budget_real_progress_resets_ladder_but_preserves_episode_cooldown() {
    // After a soft notice is injected, if the model takes an action that truly
    // advances the task, the whole escalation ladder (soft_injected /
    // ledger_injected / hard_injected / grace) must reset so the next
    // no-progress round starts over from soft instead of skipping to
    // ledger/hard because of residual soft_injected. Otherwise, in long tasks
    // a single early divergence makes every subsequent refocusing reminder
    // slide to a hard stop faster.
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    // Pin inside the metered zone (iteration=30 -> over=10 -> soft_threshold=5).
    supervisor.iteration = 30;

    // Phase 1: 5 consecutive no-information-gain rounds (failed reads)
    // accumulate to soft_threshold=5, firing the soft notice.
    let mut last = ToolLoopSignal::None;
    for i in 0..5 {
        pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
        last = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    }
    assert!(matches!(last, ToolLoopSignal::LowProgressSoft));
    assert!(supervisor.progress.soft_injected);

    // Phase 2: one real mutation action (apply_patch) -> substantive progress,
    // resetting the whole escalation ladder.
    messages.push(pb_apply_patch_msg("patch-1"));
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::None));
    assert!(!supervisor.progress.soft_injected);
    assert!(!supervisor.progress.ledger_injected);
    assert!(!supervisor.progress.hard_injected);
    assert_eq!(supervisor.progress.consecutive_no_progress, 0);

    // Phase 3: with another 5 consecutive no-information-gain rounds, the
    // escalation ladder has been reset, but the episode cooldown suppresses
    // high-frequency repetition of the same soft notice so complex tasks are
    // not constantly interrupted by "make a bit of progress, get noticed
    // again".
    for i in 0..5 {
        pb_failed_read_round(&mut messages, &format!("src/g{i}.rs"), &format!("r2-{i}"));
        last = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    }
    assert!(
        matches!(last, ToolLoopSignal::None),
        "real progress must reset escalation without immediately repeating the same soft prompt"
    );
    assert!(!supervisor.progress.soft_injected);
    assert!(!supervisor.progress.ledger_injected);

    // After the cooldown expires, if there is still no progress, a new episode
    // restarts from soft without skipping levels.
    supervisor.iteration = supervisor.progress.next_episode_iteration;
    pb_failed_read_round(&mut messages, "src/g-final.rs", "r2-final");
    last = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(last, ToolLoopSignal::LowProgressSoft));
    assert!(supervisor.progress.soft_injected);
    assert!(!supervisor.progress.ledger_injected);
}

#[test]
fn progress_budget_escalates_soft_then_ledger_then_hard() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    // The late phase of long tasks still uses the stable soft_threshold=5;
    // after soft there is a fixed response window, then the ledger stage, and
    // only at soft_threshold + margin does the hard stop happen.
    supervisor.iteration = 50;
    let mut signals = Vec::new();
    for i in 0..(5 + PROGRESS_NO_PROGRESS_HARD_MARGIN) {
        supervisor.next_iteration();
        pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
        signals
            .push(supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS));
    }
    assert!(
        signals[..4]
            .iter()
            .all(|signal| matches!(signal, ToolLoopSignal::None))
    );
    assert!(matches!(signals[4], ToolLoopSignal::LowProgressSoft));
    assert!(
        signals[5..10]
            .iter()
            .all(|signal| matches!(signal, ToolLoopSignal::None))
    );
    assert!(matches!(signals[10], ToolLoopSignal::LowProgressLedger));
    assert!(
        signals[11..signals.len() - 1]
            .iter()
            .all(|signal| matches!(signal, ToolLoopSignal::None))
    );
    assert!(matches!(
        signals.last(),
        Some(ToolLoopSignal::LowProgressHard)
    ));
    assert!(supervisor.progress.hard_injected);
}

#[test]
fn progress_budget_grace_window_pauses_escalation_on_new_reasoning() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    // 5 consecutive rounds after iteration=30 trigger soft; soft itself first
    // grants every model a fixed response window.
    supervisor.iteration = 30;
    let mut last = ToolLoopSignal::None;
    for i in 0..5 {
        supervisor.next_iteration();
        pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
        last = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    }
    assert!(matches!(last, ToolLoopSignal::LowProgressSoft));
    assert!(supervisor.progress.soft_injected);
    let base_grace_until = supervisor.progress.grace_until_iteration;

    // Within the response window, even a model that exposes no reasoning does
    // not immediately receive the ledger on the round after soft.
    supervisor.next_iteration();
    pb_failed_read_round_reasoning(&mut messages, "src/g.rs", "r-grace", None);
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(
        matches!(signal, ToolLoopSignal::None),
        "soft must always grant a response window instead of injecting ledger next round"
    );
    assert!(!supervisor.progress.ledger_injected);
    assert_eq!(supervisor.progress.grace_until_iteration, base_grace_until);
    assert!(!supervisor.progress.grace_consumed);

    // Advance to the final round inside the base window with reasoning
    // unchanged.
    while supervisor.iteration + 1 < base_grace_until {
        supervisor.next_iteration();
        let id = format!("r-base-{}", supervisor.iteration);
        pb_failed_read_round_reasoning(
            &mut messages,
            &format!("src/base-{}.rs", supervisor.iteration),
            &id,
            Some("先看调用方再决定删除策略"),
        );
        assert!(matches!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::None
        ));
    }

    // Giving a substantively different rationale when the base window expires
    // buys one extra extension.
    supervisor.next_iteration();
    pb_failed_read_round_reasoning(
        &mut messages,
        "src/extended.rs",
        "r-extended",
        Some("换一个思路：检查配置加载顺序"),
    );
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::None));
    assert!(supervisor.progress.grace_consumed);
    assert!(supervisor.progress.grace_until_iteration > base_grace_until);

    // After grace expires, further reasoning changes must not keep rolling the
    // extension forward.
    let grace_until = supervisor.progress.grace_until_iteration;
    supervisor.iteration = grace_until;
    pb_failed_read_round_reasoning(
        &mut messages,
        "src/h.rs",
        "r-after-grace",
        Some("再换一个思路：检查配置加载顺序"),
    );
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::LowProgressLedger));
    assert_eq!(supervisor.progress.grace_until_iteration, grace_until);
}
