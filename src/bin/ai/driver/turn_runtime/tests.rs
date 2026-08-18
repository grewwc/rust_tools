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
    // 不足窗口
    let history = vec![sig.clone(); TOOL_LOOP_SOFT_WINDOW - 1];
    assert!(!detect_tool_loop(&history, TOOL_LOOP_SOFT_WINDOW));
    // 满 soft 窗口触发，但尚不满 hard 窗口
    let history = vec![sig.clone(); TOOL_LOOP_SOFT_WINDOW];
    assert!(detect_tool_loop(&history, TOOL_LOOP_SOFT_WINDOW));
    assert!(!detect_tool_loop(&history, TOOL_LOOP_HARD_WINDOW));
    // 满 hard 窗口且完全相同
    let history = vec![sig.clone(); TOOL_LOOP_HARD_WINDOW];
    assert!(detect_tool_loop(&history, TOOL_LOOP_HARD_WINDOW));
    // 满窗口但有一轮不同
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
    // 回归：soft 窗口 4 无法被周期 3 整除，旧实现 `window % period != 0` 直接
    // continue，导致 A-B-C-A-B-C 在第 6 轮被无预警 hard-stop（hard 先查、soft
    // 后查，soft 永不可能先触发，升级不变量被破坏）。修复后退化为「若干完整
    // 周期 + 一个周期前缀」也判为循环，使 3 周期在第 4 轮（A-B-C-A）先拿到 Soft。
    let a = vec!["tree::{\"path\":\"src\"}".to_string()];
    let b = vec!["read_file::{\"path\":\"src/bin/a.rs\"}".to_string()];
    let c = vec!["tree::{\"path\":\"src/bin\"}".to_string()];

    // 不足窗口不误报。
    assert!(!detect_tool_loop(
        &[a.clone(), b.clone(), c.clone()],
        TOOL_LOOP_SOFT_WINDOW
    ));
    // 3 周期 + 1 前缀正好填满 soft 窗口 4：应触发 Soft。
    assert!(detect_tool_loop(
        &[a.clone(), b.clone(), c.clone(), a.clone()],
        TOOL_LOOP_SOFT_WINDOW
    ));
    // 完整 hard 窗口（6 轮整周期）仍触发，且整除路径不受影响。
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
    // 前缀不匹配（第 4 轮与周期无关）不得误报。
    assert!(!detect_tool_loop(
        &[a.clone(), b.clone(), c.clone(), b.clone()],
        TOOL_LOOP_SOFT_WINDOW
    ));
}

#[test]
fn execute_command_is_read_only_skips_nav_segments_and_requires_all_substantive_read_only() {
    // 前导 cd/export 无副作用，应跳过：`cd X && git status` 是只读。
    assert!(execute_command_is_read_only(
        "cd /tmp && git status --short"
    ));
    assert!(execute_command_is_read_only(
        "cd /tmp && export FOO=1 && ls -la"
    ));
    // 任一实质段可能变更 → 非只读（堵住旧实现「只看首段」的盲区）。
    assert!(!execute_command_is_read_only("ls /tmp && rm -rf build"));
    assert!(!execute_command_is_read_only(
        "cd /tmp && git checkout master"
    ));
    // 纯只读命令仍成立。
    assert!(execute_command_is_read_only("git log --oneline -5"));
    assert!(execute_command_is_read_only("ls -la /tmp"));
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

    // 收集每一轮的信号：前 SOFT_WINDOW-1 轮不触发，第 SOFT_WINDOW 轮触发 Soft。
    // Soft 会清空旧样本，因此必须在提示后再次重复 HARD_WINDOW 轮才触发 Hard。
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

    // 同一只读调用重复到 soft 阈值。
    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        messages.push(pb_read_msg("src/main.rs", &format!("read-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_SOFT_WINDOW - 1 {
            assert!(matches!(signal, ToolLoopSignal::Soft));
        }
    }
    assert!(supervisor.loop_breaker_injected);

    // soft 后的实际变更表示任务在推进，必须清除旧循环状态。
    messages.push(pb_apply_patch_msg("patch-1"));
    assert!(matches!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS,),
        ToolLoopSignal::None
    ));
    assert!(!supervisor.loop_breaker_injected);
    assert!(supervisor.tool_signature_history.is_empty());

    // 新的一轮重复必须重新先得到 soft，而不是沿用旧状态直接 hard-stop。
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

    // 阶段一：同一只读调用重复到 soft 触发。
    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        messages.push(pb_read_msg("src/main.rs", &format!("read-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_SOFT_WINDOW - 1 {
            assert!(matches!(signal, ToolLoopSignal::Soft));
        }
    }
    assert!(supervisor.loop_breaker_injected);
    // soft 处理器清空了 coarse/target 历史并重武装了它们的门。
    assert!(supervisor.tool_signature_history_coarse.is_empty());
    assert!(supervisor.tool_target_history.is_empty());
    assert!(!supervisor.coarse_loop_note_injected);
    assert!(!supervisor.target_repeat_note_injected);

    // 阶段二：soft 后模型换一批 `ls` 变体继续翻同一日志目录。exact 签名各不相同，
    // 不会重触发精确 soft/hard；但 coarse 签名都是 `ls:/data01/logs`，应在填满
    // COARSE_WINDOW 后触发 Coarse（旧实现中该门被 loop_breaker_injected 永久挡死，
    // 会一直漏到迭代上限）。
    // 注意：第一个 `ls` 轮因 `/data01/logs` 是全新目标，会走 assess_progress 的
    // 新目标 + 已注入过 loop 分支，触发 reset_tool_loop_escalation（清空 coarse
    // 历史并重武装门）。因此需要在其后再积累 COARSE_WINDOW 轮同 coarse 样本。
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
        // 第 5 轮（ls-5，索引 5）是 reset 后填满 COARSE_WINDOW 的首轮。
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

    // 第一轮：积累到 soft 触发。
    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        messages.push(assistant_with_read(&format!("tc-{i}")));
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    }
    // 验证 soft 已触发，flag 已设置。
    assert!(supervisor.loop_breaker_injected);
    assert!(!supervisor.hard_loop_stop_injected);

    // 截断触发标记：历史清空，skip +1，所有 flag 重置。
    supervisor.mark_truncation_skip();
    assert!(supervisor.tool_signature_history.is_empty());
    assert!(supervisor.tool_signature_history_coarse.is_empty());
    assert_eq!(supervisor.skip_tool_signature_rounds, 1);
    // 关键验证：所有一次性标志都被重置。
    assert!(!supervisor.hard_loop_stop_injected);
    assert!(!supervisor.loop_breaker_injected);
    assert!(!supervisor.coarse_loop_note_injected);

    // 截断迭代：跳过签名记录。
    messages.push(assistant_with_read("tc-skip"));
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::None));
    assert!(supervisor.tool_signature_history.is_empty());
    assert_eq!(supervisor.skip_tool_signature_rounds, 0);

    // 第二轮：恢复后重新积累，验证 soft 能再次触发。
    for i in 0..TOOL_LOOP_SOFT_WINDOW {
        messages.push(assistant_with_read(&format!("tc2-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_SOFT_WINDOW - 1 {
            // 第 4 次应触发 soft。
            assert!(matches!(signal, ToolLoopSignal::Soft));
            assert!(supervisor.loop_breaker_injected);
        } else {
            assert!(matches!(signal, ToolLoopSignal::None));
        }
    }

    // soft 后需重新积累完整 hard window，验证完整升级阶梯恢复。
    for i in 0..TOOL_LOOP_HARD_WINDOW {
        messages.push(assistant_with_read(&format!("tc3-{i}")));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if i == TOOL_LOOP_HARD_WINDOW - 1 {
            // 收到 soft 后又重复 6 次，才应触发 hard。
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

/// 长循环感知：短 turn 保持基准软阈值不变；一旦迭代轮次达阈值，有效软阈值
/// 被下调到 SOFT_FLOOR，让内容级去重尽早介入，遏制 O(n²) 累积重发。
/// 这是 aefa66f2 那类「历史中等(~120K) + 56 轮迭代」撞 TPM 的直接修复：
/// 基准阈值 135K 永不触发，下调到 36K 后长循环中段即开始压缩。
#[test]
fn long_loop_lowers_effective_mid_turn_soft_threshold() {
    const FLOOR: usize = super::super::MID_TURN_COMPRESS_SOFT_FLOOR;
    // flagship 大窗口模型的基准软阈值远高于 FLOOR（模拟 135K）。
    let base = 135_000usize;
    assert!(base > FLOOR, "precondition: base threshold above floor");

    let mut s = TurnSupervisor::default();

    // 短 turn（未达长循环阈值）：有效阈值 == 基准，不误伤正常单轮大任务。
    s.iteration = LONG_LOOP_COMPRESS_ITERATION_THRESHOLD - 1;
    assert_eq!(s.effective_mid_turn_soft_threshold(base), base);
    // 此时 ~120K 历史（< 135K 基准）不触发压缩——正是旧行为的空窗。
    assert!(
        !s.should_try_mid_turn_compress(120_000, s.effective_mid_turn_soft_threshold(base))
    );

    // 长循环（达阈值）：有效阈值降到 FLOOR，同样 ~120K 历史立即触发压缩。
    s.iteration = LONG_LOOP_COMPRESS_ITERATION_THRESHOLD;
    assert_eq!(s.effective_mid_turn_soft_threshold(base), FLOOR);
    assert!(s.should_try_mid_turn_compress(120_000, s.effective_mid_turn_soft_threshold(base)));

    // 若基准本就低于 FLOOR（history_max_chars 很小的场景），min() 保证不抬高阈值。
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
    // 翻页循环逃逸根因：tail -c +N / sed -n 'N,Mp' 的偏移字面量此前被保留在
    // coarse 签名里，同一文件换窗口翻页时每轮签名互不相同，coarse / target-repeat
    // 都抓不到。偏移/行号区间属于「窗口」而非目标资源，应被剥掉（与 read_file 的
    // offset/limit 剥离同理），使翻页变体折叠为同一签名。
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
    // 不同目标文件仍保持区分，避免误合并。
    assert_ne!(a, coarse_execute_command_signature("tail -c +2401 other.txt"));
}

#[test]
fn coarse_catches_middle_paging_loop_never_from_top() {
    // 纯「翻页」循环：同一文件反复从中间偏移读（tail -c +N，每轮偏移不同、
    // 永不为 +1，from-top 重扫检测抓不到）。整轮原始命令各不相同，exact 落空；
    // 修复前 coarse 签名保留偏移字面量，整轮 coarse 集合也永不相等，循环一直
    // 逃逸到迭代上限。修复后偏移被剥掉，全部折叠为 `tail:<file>`：第 5 轮触发
    // Coarse 软提示，第 8 轮触发 CoarseHard 硬停。
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let file = "src/all_desc.txt";
    let mut signals = Vec::new();
    for i in 0..TOOL_LOOP_COARSE_HARD_WINDOW {
        let offset = 2401 + i * 2400; // 中间偏移，永不为 1（from-top）。
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

    // 每轮 offset 递增：字节精确签名各不相同 → soft/hard 不触发；
    // 剥离 offset/limit 后 coarse 签名一致 → 满 COARSE_WINDOW 后触发 Coarse。
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

    // coarse 只提示一次：继续同样翻页不再返回 Coarse。
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

// ===== Progress Budget（行为信号进展预算）测试 =====
// 这些用例都刻意让每轮工具签名各不相同（不同 path），从而绕过 exact/coarse
// 「签名重复」检测，专门验证第三层 assess_progress 的「信息增益」判定：成功的
// 新目标读取算进展，失败读取（无目标）不算进展。

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

/// 失败的只读调用轮：assistant 发起 read_file，紧跟一条 tool 结果表示读取失败。
/// 失败调用不进入 `extract_round_targets`（无目标 → 无信息增益 → 无进展），
/// 且因每轮 path 不同而绕过 exact/coarse 签名循环检测，是进展预算升级阶梯
/// 在「统一为行为信号」后唯一的无进展驱动方式（成功的新目标读取都算进展）。
fn pb_failed_read_round(messages: &mut Vec<crate::ai::history::Message>, path: &str, id: &str) {
    pb_failed_read_round_reasoning(messages, path, id, None);
}

/// `pb_failed_read_round` 的带 reasoning 变体：失败读取轮附带一段 reasoning，
/// 用于验证 grace 宽限（软提示后 reasoning 指纹变化 → 换取一次宽限）。
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
    // iteration 1..=25：免费区（<=20）全静默；21 起累加无进展，
    // 25 轮时 consecutive=5 达到 soft_threshold(25)=5，触发软提示。
    // 用失败读取制造「无信息增益」轮：成功的新目标读取都算进展，无法累计无进展。
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
            // 同文件翻页填满 coarse 窗口：exact 签名因 offset 不同而各异（soft/hard
            // 不触发），但剥离 offset 后的 coarse 签名一致，仍应触发一次性 Coarse 刹车。
            // 这是「进展哈希须忽略 offset/limit 以防预算逃逸」不变量的关键保证。
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
        // 新证据始终算进展：Progress Budget 绝不因高效翻页误升级 soft/ledger/hard。
        assert_eq!(supervisor.progress.consecutive_no_progress, 0);
    }

    // Coarse 只提示一次，且新证据不会清空签名历史（否则窗口永远填不满、刹车失效）。
    assert!(
        supervisor.coarse_loop_note_injected,
        "coarse paging brake must have fired exactly once"
    );
    assert_eq!(supervisor.progress.seen_targets.len(), 1);
    // 12 轮翻页里，coarse 命中的那一轮（i=COARSE_WINDOW-1）在进入 assess_progress
    // 前就早退，其证据不入账；其余 11 轮各记一条独立指纹。
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

/// 纯只读发散（零 mutation，持续换新目标读取）必须在累计目标数达到
/// `READ_ONLY_BREADTH_HARD_STOP_TARGETS` 时被强制收口——即便每轮都读到新目标
/// / 新字节而使常规无进展计数被反复复位。这条判据与内容新颖度解耦，是本次
/// 「诊断 agent 只读发散两小时不收口」事故的最终刹车。
#[test]
fn progress_budget_pure_readonly_divergence_hard_stops_after_breadth_ceiling() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let mut saw_hard_stop_at = None;
    for i in 1..=(READ_ONLY_BREADTH_HARD_STOP_TARGETS + 4) {
        supervisor.next_iteration();
        // 每轮都读一个全新文件并拿到成功结果：既是新目标又是新证据，
        // 常规 no-progress 计数始终为 0，只有解耦的广度硬停能兜住。
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

/// 一旦本 turn 发生过任何 mutation，广度硬停永久让路：正常「读大量文件 + 改」
/// 的实现型任务不受该刹车约束，避免劣化 agent 效果。
#[test]
fn progress_budget_breadth_hard_stop_never_fires_after_mutation() {
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    // 第 1 轮就做一次成功项目 mutation，置位 observed_project_mutation_this_turn。
    supervisor.next_iteration();
    messages.push(pb_apply_patch_msg("patch-1"));
    messages.push(pb_tool_result(
        "patch-1",
        "Successfully patched src/lib.rs; +3 -1 (2 lines)",
    ));
    let _ = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(supervisor.progress.observed_project_mutation_this_turn);

    // 随后大量只读探索远超硬停阈值，也绝不能触发 ReadOnlyBreadthHard。
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

    // 先积累到 breadth 阈值前一格，模拟已经读过很多证据但还没触发
    // ReadOnlyBreadth 的状态。
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
    // 浏览器工具不在 MUTATION_TOOL_NAMES、参数也不带 path/query；若不提取
    // url/selector，navigate 新页面与读取新 selector 会被误判为无进展，正常的
    // 多步浏览 turn 会在进展预算阶梯下被 LowProgressHard 误停。
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

/// 混合工具轮里同一目标反复取证：每轮都读同一个文件 A，但穿插一个每轮都不同的
/// tree，使整轮 exact/coarse 签名各不相等而逃过 detect_tool_loop；此时
/// 目标交集检测应抓到「A 每轮都在」并发出一次 TargetRepeat。
#[test]
fn turn_supervisor_emits_target_repeat_for_mixed_tool_rounds_on_same_file() {
    fn mixed_round(i: usize) -> crate::ai::history::Message {
        crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(String::new()),
            tool_calls: Some(vec![
                // 每轮恒定：反复读同一个文件 A。
                crate::ai::types::ToolCall {
                    id: format!("read-A-{i}"),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall {
                        name: "read_file".to_string(),
                        arguments: "{\"path\":\"src/bin/ai/mod.rs\",\"offset\":100}".to_string(),
                    },
                },
                // 每轮不同的陪衬目录读取：让整轮签名各不相等，逃过整轮判等。
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

    // 整轮签名每轮都不同：exact / coarse 整轮判等一律不命中。
    assert!(
        signals[..TOOL_LOOP_COARSE_WINDOW - 1]
            .iter()
            .all(|s| matches!(s, ToolLoopSignal::None)),
        "whole-round signatures differ every round; nothing should fire early: {signals:?}"
    );
    // 填满 coarse 窗口时，目标交集（文件 A）命中，发出一次 TargetRepeat。
    assert!(
        matches!(
            signals[TOOL_LOOP_COARSE_WINDOW - 1],
            ToolLoopSignal::TargetRepeat
        ),
        "same-file across mixed rounds must trigger TargetRepeat: {signals:?}"
    );
    assert!(supervisor.target_repeat_note_injected);
}

/// 反例守卫：每轮读的文件都不同（无公共目标），目标交集为空，
/// 不得误报 TargetRepeat。
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
    // 复现 f319d490 会话的真实逃逸模式：同一文件从文件头反复读取（每轮页宽
    // 变化：2400B → 1400B → read_file 行分页），中间混入每轮不同的新归档
    // 路径——整轮签名永不相等，exact/coarse/target-repeat 三道整轮检测全部
    // 落空；重扫计数按目标累计、与整轮签名解耦：第 3 次从头读注入软提示，
    // 第 4 次从头读硬停止。
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let file = "src/all_desc.txt";

    // 第 1 次从头读（tail -c +1）+ 后续分页。
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
    // 混入每轮不同的新归档路径读取（此前三道检测的逃逸关键）。
    pb_read_round(&mut messages, &format!("{file}.archive-1"), "a1");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    );
    // 第 2 次从头读：换成 1400 字节页宽。
    pb_execute_command_round(&mut messages, &format!("tail -c +1 {file} | head -c 1400"), "t1");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    );
    pb_execute_command_round(&mut messages, &format!("tail -c +1401 {file} | head -c 1400"), "t1p1");
    // 修复后：tail/sed 的偏移/行号字面量被剥掉，窗口 [tail,tail,archive,
    // tail,tail] 折叠成 3 周期 [tail,tail,archive]，coarse 在填满窗口时
    // 提前命中（比 from-top 重扫更早抓住这个翻页+混轮循环）。
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
    // 第 3 次从头读（read_file offset 缺省）→ 软提示。
    pb_read_round(&mut messages, file, "r3");
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(
        matches!(signal, ToolLoopSignal::TargetRescan(..)),
        "expected TargetRescan at 3rd from-top read, got {signal:?}"
    );
    // 第 4 次从头读（offset=0）→ 硬停止。
    pb_successful_read_round(&mut messages, file, 0, "r4", "content page");
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(
        matches!(signal, ToolLoopSignal::TargetRescanHard(..)),
        "expected TargetRescanHard at 4th from-top read, got {signal:?}"
    );
}

#[test]
fn target_rescan_resets_on_write_file_edit() {
    // 编辑后从头重读是合法验证：write_file 修改目标清零重扫计数，
    // 读→改→读 循环不应触发任何重扫信号。
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
    // 单调向前翻页（offset 递增、从不回头）不是重扫：只计 1 次从头读。
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
    // 上下文压缩后跨多轮的合法重读不应累积：距上次从头读取超过窗口轮数
    // （TARGET_RESCAN_WINDOW_ROUNDS=8）时，计数过期并从 1 重新累计。
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let target = "src/spaced-file.rs";
    // 第 1 次从头读，iteration=1 → 计数 1。
    supervisor.iteration = 1;
    pb_execute_command_round(&mut messages, &format!("tail -c +1 {target} | head -c 2400"), "r1");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    );
    // 跨 10 轮后再从头读（gap=10 > 8）→ 计数过期，重新从 1 计，不触发。
    supervisor.iteration = 11;
    pb_execute_command_round(&mut messages, &format!("cat {target}"), "r11");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None
    );
    // 窗口内连续重读：每轮命令互异（tail/cat/sed/nl/head），避免误触精确/粗粒度
    // 循环检测。第 2 次（累计 2）不触发，第 3 次软提示，第 4 次硬停止。
    supervisor.iteration = 12;
    pb_execute_command_round(&mut messages, &format!("sed -n 1,40p {target}"), "r12");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "round r12"
    );
    supervisor.iteration = 13;
    pb_execute_command_round(&mut messages, &format!("nl -ba {target}"), "r13");
    assert!(
        matches!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::TargetRescan(..)
        ),
        "round r13 should soft-signal"
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
    // 回归：衰减开启新的重读 episode，每一段循环都必须拿到自己的 soft 预警。
    // 旧实现衰减时只重置计数、不清 rescan_note_injected，第二段累计到 soft 时
    // insert 返回 false → 跳过软提示直接硬停（soft→hard 升级不变量被破坏）。
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let target = "src/episode-file.rs";
    // Episode 1：第 1/2 次不触发，第 3 次软提示并记录 rescan_note_injected。
    for (iter, cmd) in [
        (1, format!("tail -c +1 {target} | head -c 2400")),
        (2, format!("cat {target}")),
        (3, format!("sed -n 1,40p {target}")),
    ] {
        supervisor.iteration = iter;
        pb_execute_command_round(&mut messages, &cmd, &format!("r{iter}"));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        if iter == 3 {
            assert!(
                matches!(signal, ToolLoopSignal::TargetRescan(..)),
                "episode1 round r3 should soft-signal"
            );
        } else {
            assert_eq!(signal, ToolLoopSignal::None, "episode1 round r{iter}");
        }
    }
    // 跨 9 轮（gap=9 > TARGET_RESCAN_WINDOW_ROUNDS=8）→ 计数衰减并从 1 重新计，
    // 同时软提示标记必须随 episode 一并清除（本次修复点）。
    supervisor.iteration = 12;
    pb_execute_command_round(&mut messages, &format!("nl -ba {target}"), "r12");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "decay round r12"
    );
    // Episode 2：第 2 次不触发，第 3 次必须再次软提示——本段自己的预警，
    // 不能因为上一段已提示过就跳过 soft 直接硬停。
    supervisor.iteration = 13;
    pb_execute_command_round(&mut messages, &format!("head -n 200 {target}"), "r13");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "episode2 round r13"
    );
    supervisor.iteration = 14;
    pb_execute_command_round(&mut messages, &format!("less {target}"), "r14");
    assert!(
        matches!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::TargetRescan(..)
        ),
        "episode2 round r14 should get its own soft warning"
    );
}

#[test]
fn target_rescan_resets_on_other_file_mutation() {
    // 多文件工作流：编辑任意目标（不限于被重读的目标）都整表清空 rescan 计数，
    // 后续对另一文件的验证性从头重读从 1 重新累计，不应被误判为翻页循环。
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    let target = "src/read-a.rs";
    // 两种不同签名（offset 0 / 1）的从头读，各累计一次。
    for (offset, id) in [(0usize, "r1"), (1usize, "r2")] {
        supervisor.iteration = id.trim_start_matches('r').parse().unwrap();
        pb_successful_read_round(&mut messages, target, offset, id, "line:0");
        assert_eq!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::None,
            "round {id}"
        );
    }
    // 第 3 轮：编辑其他文件 B → 整表清空 A 的计数。
    supervisor.iteration = 4;
    let id = "w4";
    messages.push(pb_write_file_msg("src/edit-b.rs", id));
    messages.push(pb_tool_result(id, "Successfully wrote file."));
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "edit-other-file round must not signal"
    );
    // 编辑后从头重读 A 重新从 1 计：cat / offset=0 / offset=1 三种互异签名。
    // 第 2 次不触发，第 3 次才软提示。
    supervisor.iteration = 5;
    pb_execute_command_round(&mut messages, &format!("cat {target}"), "c5");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "round r5"
    );
    supervisor.iteration = 6;
    pb_successful_read_round(&mut messages, target, 0, "r6", "line:0");
    assert_eq!(
        supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
        ToolLoopSignal::None,
        "round r6"
    );
    supervisor.iteration = 7;
    pb_successful_read_round(&mut messages, target, 1, "r7", "line:0");
    assert!(
        matches!(
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS),
            ToolLoopSignal::TargetRescan(..)
        ),
        "round r7 should soft-signal"
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
    // read_file offset 缺省/0/1 → from-top；offset=100 → 不是。
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

/// 回归：反复写同一个「沙箱越界被拒」的路径（content 每轮不同、file_path 相同），
/// 曾因失败被误计为 mutation-progress、且不进入 target 历史而逃过所有 loop guard。
/// 修复后：blocked 写不再算进展，且归一成稳定目标，target-repeat guard 在少数轮内命中。
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
        // content 每轮不同：exact/coarse 整轮签名各不相等，逃过 detect_tool_loop。
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

/// 回归：blocked write 不得清零无进展预算。此前 `_ => true` 把失败写也计为
/// mutation，每次重试都重置 consecutive_no_progress，使 progress-budget 永不升级。
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

    // 对照：成功写入仍算 mutation。
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
    // 普通失败（非 write-blocked）仍归类为 Failure。
    assert_eq!(
        classify_tool_result_progress("Error: read_file failed: File not found"),
        ToolResultProgressStatus::Failure
    );
    // 无 marker 文本不误匹配。
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
    // 固定在计费区（iteration=30 → soft_threshold=5）。
    supervisor.iteration = 30;
    for i in 0..4 {
        pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
        let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(matches!(signal, ToolLoopSignal::None));
    }
    assert_eq!(supervisor.progress.consecutive_no_progress, 4);
    // 一次真正的变更动作：无进展计数清零。
    messages.push(pb_apply_patch_msg("patch-1"));
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::None));
    assert_eq!(supervisor.progress.consecutive_no_progress, 0);
}

#[test]
fn progress_budget_uses_pre_compress_current_round_for_apply_patch_progress() {
    let mut supervisor = TurnSupervisor::default();
    let mut compressed_messages = Vec::new();
    // 固定在计费区，并预置 4 轮无进展；下一轮若仍按压缩后视图判定，
    // 会触发 LowProgressSoft。真实当前轮是 apply_patch，必须按原始工具轮清零。
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
    // 软提示注入后，若模型给出真正推进任务的动作，应重置整个升级阶梯
    // （soft_injected / ledger_injected / hard_injected / grace），使下一轮无进展
    // 重新从 soft 开始，而非因 soft_injected 残留直接跳级到 ledger/hard。
    // 否则长任务中模型只要早期发散过一次，每次收敛提醒都会更快滑向硬停。
    let mut supervisor = TurnSupervisor::default();
    let mut messages = Vec::new();
    // 固定在计费区（iteration=30 -> over=10 -> soft_threshold=5）。
    supervisor.iteration = 30;

    // 阶段一：连续 5 轮无信息增益（失败读取）累计到 soft_threshold=5，触发软提示。
    let mut last = ToolLoopSignal::None;
    for i in 0..5 {
        pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
        last = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    }
    assert!(matches!(last, ToolLoopSignal::LowProgressSoft));
    assert!(supervisor.progress.soft_injected);

    // 阶段二：一次真正的变更动作（apply_patch）-> 实质进展，重置整个升级阶梯。
    messages.push(pb_apply_patch_msg("patch-1"));
    let signal = supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
    assert!(matches!(signal, ToolLoopSignal::None));
    assert!(!supervisor.progress.soft_injected);
    assert!(!supervisor.progress.ledger_injected);
    assert!(!supervisor.progress.hard_injected);
    assert_eq!(supervisor.progress.consecutive_no_progress, 0);

    // 阶段三：再次连续 5 轮无信息增益时，升级阶梯已经重置，但 episode cooldown
    // 会抑制同一 soft 的高频重复，避免复杂任务被「推进一点、再提示一次」持续打断。
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

    // cooldown 到期后，若仍无进展，新的 episode 重新从 soft 开始，不会跳级。
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
    // 长任务后段仍使用稳定 soft_threshold=5；soft 后先给固定响应窗口，再进入
    // ledger，最后到 soft_threshold + margin 才 hard stop。
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
    // iteration=30 后连续 5 轮触发 soft；soft 自身先给所有模型固定响应窗口。
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

    // 响应窗口内即使模型不暴露 reasoning，也不会在 soft 下一轮立刻收到 ledger。
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

    // 在基础窗口内推进到最后一轮，reasoning 保持不变。
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

    // 基础窗口到期时给出实质不同的理由，可额外延长一次。
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

    // grace 到期后即使 reasoning 再变化，也不能继续滚动续期。
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
