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

use crate::ai::{mcp::SharedMcpClient, types::App};

use super::{
    MID_TURN_COMPRESS_COOLDOWN_ITERATIONS, MID_TURN_COMPRESS_DELTA_THRESHOLD,
    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS, MID_TURN_LLM_SUMMARY_MAX_CHARS,
    finalize::finalize_turn,
    iteration::{execute_turn_iteration, refresh_skill_turn_for_iteration},
    mid_turn_compress_hard_threshold, mid_turn_compress_soft_threshold,
    prepare::prepare_turn,
    tool_result::handle_iteration_execution,
    types::{IterationExecution, TurnLoopStep, TurnOutcome, TurnPreparation},
};

/// 工具调用循环检测窗口：
/// - soft: 连续 4 轮调用 (tool_name, normalized_args) 完全一致，先注入反思提示
/// - hard: 连续 6 轮完全一致，直接强制收敛，不再继续工具循环
const TOOL_LOOP_SOFT_WINDOW: usize = 4;
const TOOL_LOOP_HARD_WINDOW: usize = 6;
/// 近似循环窗口：连续 N 轮对「同一目标资源」调用同一工具（忽略 offset/limit
/// 等翻页参数）即命中。用于抓字节精确检测漏掉的「同文件反复翻页 / 仅微调分页
/// 参数的重复检索」这类真实膨胀。仅注入一次软提示，不强制收敛。
const TOOL_LOOP_COARSE_WINDOW: usize = 5;
const TOOL_SIGNATURE_HISTORY_LIMIT: usize = TOOL_LOOP_HARD_WINDOW + 2;
const TASK_ANCHOR_MAX_QUESTION_CHARS: usize = 220;

/// 计算 coarse 签名时需剥离的「易变翻页/窗口」参数键。剥离后同一文件的不同
/// 分页、同一检索的不同结果上限会折叠成同一 coarse 签名。
const VOLATILE_ARG_KEYS: &[&str] = &["offset", "limit", "page", "cursor", "max_results"];

/// 中段断路器绝对上限：单轮工具迭代达到该值即注入一次收敛提示，远早于
/// max_iterations 硬上限，用于治理「合法但低效」的工具刷屏。
const TOOL_ITERATION_SOFT_LIMIT: usize = 384;

/// 连续「流读取中断型」截断（stream_error）的重试上限。超过即放弃本 turn，
/// 避免服务端持续断流时无限重试（尤其后台任务的 max_iterations = usize::MAX）。
const MAX_STREAM_ERROR_RETRIES: usize = 16;

/// 单轮工具迭代软阈值：取 max_iterations 的一半与绝对上限的较小值，保证一定
/// 早于硬上限触发；对超大 ceiling 也能在中段就提醒收敛。
fn tool_iteration_soft_limit(max_iterations: usize) -> usize {
    (max_iterations / 2).max(1).min(TOOL_ITERATION_SOFT_LIMIT)
}

/// 提取最近一轮 assistant 消息中的 (tool_name, args_json) 签名集合。
/// 任何一个签名与窗口内某轮完全一致即认为有循环倾向。
fn extract_round_tool_signatures(messages: &[crate::ai::history::Message]) -> Option<Vec<String>> {
    extract_round_tool_signatures_inner(messages, false)
}

/// 提取「粗粒度」签名：剥离 offset/limit/page 等易变翻页参数后再归一化。
/// 用于抓字节精确检测漏掉的同文件翻页 / 仅微调分页参数的重复检索。
fn extract_round_tool_signatures_coarse(
    messages: &[crate::ai::history::Message],
) -> Option<Vec<String>> {
    extract_round_tool_signatures_inner(messages, true)
}

fn extract_round_tool_signatures_inner(
    messages: &[crate::ai::history::Message],
    coarse: bool,
) -> Option<Vec<String>> {
    use serde_json::Value;
    let last_assistant = messages.iter().rev().find(|m| m.role == "assistant")?;
    let tool_calls = last_assistant.tool_calls.as_ref()?;
    if tool_calls.is_empty() {
        return None;
    }
    let mut sigs: Vec<String> = Vec::with_capacity(tool_calls.len());
    for tc in tool_calls.iter() {
        let name = tc.function.name.as_str();
        let args_raw = tc.function.arguments.as_str();
        // 归一化 args：解析为 Value 后再 to_string，去掉空白噪音。
        // coarse 模式下先剥离易变翻页参数，让同一目标资源的不同分页折叠为同一签名。
        let args_norm = serde_json::from_str::<Value>(args_raw)
            .map(|mut v| {
                if coarse {
                    strip_volatile_args(&mut v);
                }
                v.to_string()
            })
            .unwrap_or_else(|_| args_raw.to_string());
        sigs.push(format!("{name}::{args_norm}"));
    }
    sigs.sort();
    Some(sigs)
}

/// 从 args Value（若为 object）中移除翻页/窗口类易变键。
fn strip_volatile_args(value: &mut serde_json::Value) {
    if let Some(map) = value.as_object_mut() {
        for key in VOLATILE_ARG_KEYS {
            map.remove(*key);
        }
    }
}

fn detect_tool_loop(history: &[Vec<String>], window: usize) -> bool {
    if window == 0 || history.len() < window {
        return false;
    }
    let tail = &history[history.len() - window..];
    let first = &tail[0];
    if first.is_empty() {
        return false;
    }
    tail.iter().all(|sigs| sigs == first)
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out = String::new();
    for ch in s.chars().take(max_chars.saturating_sub(1)) {
        out.push(ch);
    }
    out.push('…');
    out
}

fn inject_task_anchor_note(
    messages: &mut Vec<crate::ai::history::Message>,
    question: &str,
    iteration: usize,
    reason: &str,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let goal = truncate_chars(question.trim(), TASK_ANCHOR_MAX_QUESTION_CHARS);
    let note = format!(
        "[task-anchor] reason={reason}, iteration={iteration}.\n主任务目标: {goal}\n\
请优先保持目标连续性：\n- 先总结目前已确认事实\n- 明确下一步唯一动作\n- 若信息不足，说明阻塞点并停止重复工具调用"
    );
    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

#[derive(Default)]
struct TurnSupervisor {
    iteration: usize,
    skip_tool_signature_rounds: usize,
    loop_breaker_injected: bool,
    hard_loop_stop_injected: bool,
    coarse_loop_note_injected: bool,
    iteration_soft_limit_note_injected: bool,
    iteration_limit_note_injected: bool,
    task_anchor_injected: bool,
    last_compress_iteration: usize,
    last_compress_after_chars: usize,
    tool_signature_history: Vec<Vec<String>>,
    tool_signature_history_coarse: Vec<Vec<String>>,
}

enum ToolLoopSignal {
    None,
    /// 近似循环：同一工具反复命中同一目标资源（忽略翻页参数）。软提示一次。
    Coarse,
    Soft,
    Hard,
}

impl TurnSupervisor {
    fn next_iteration(&mut self) -> usize {
        self.iteration = self.iteration.saturating_add(1);
        self.iteration
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

    fn mark_compress(&mut self, after_chars: usize) {
        self.last_compress_iteration = self.iteration;
        self.last_compress_after_chars = after_chars;
    }

    /// 截断重试时重置工具循环检测状态：截断是外部约束（输出上限 / 模型可用性波动），
    /// 重试时重复调用相同工具属于预期行为，不应计入循环检测窗口。
    /// 截断本身已有独立的 `consecutive_truncations` 上限兜底，不需要循环检测再叠加。
    ///
    /// 清空历史 + 跳过当前迭代的签名记录，使截断重试不被误判为循环。
    /// 重置所有一次性标志：截断清空历史后，soft/coarse/hard 的完整升级阶梯
    /// 应从零重新开始，否则截断恢复后形成的新循环会跳过 soft 提示直接到 hard-stop。
    fn mark_truncation_skip(&mut self) {
        self.tool_signature_history.clear();
        self.tool_signature_history_coarse.clear();
        self.skip_tool_signature_rounds += 1;
        self.hard_loop_stop_injected = false;
        self.loop_breaker_injected = false;
        self.coarse_loop_note_injected = false;
    }

    fn record_tool_signatures(
        &mut self,
        messages: &[crate::ai::history::Message],
    ) -> ToolLoopSignal {
        // 截断重试跳过：清空历史后不记录本轮签名，避免截断重试
        // 被误判为工具循环。`skip_tool_signature_rounds` 由
        // `mark_truncation_skip()` 递增，每跳过一次递减。
        if self.skip_tool_signature_rounds > 0 {
            self.skip_tool_signature_rounds -= 1;
            return ToolLoopSignal::None;
        }
        let Some(sigs) = extract_round_tool_signatures(messages) else {
            return ToolLoopSignal::None;
        };
        self.tool_signature_history.push(sigs);
        if self.tool_signature_history.len() > TOOL_SIGNATURE_HISTORY_LIMIT {
            let drop = self.tool_signature_history.len() - TOOL_SIGNATURE_HISTORY_LIMIT;
            self.tool_signature_history.drain(0..drop);
        }
        if let Some(coarse) = extract_round_tool_signatures_coarse(messages) {
            self.tool_signature_history_coarse.push(coarse);
            if self.tool_signature_history_coarse.len() > TOOL_SIGNATURE_HISTORY_LIMIT {
                let drop = self.tool_signature_history_coarse.len() - TOOL_SIGNATURE_HISTORY_LIMIT;
                self.tool_signature_history_coarse.drain(0..drop);
            }
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
            return ToolLoopSignal::Soft;
        }
        // 字节精确的 soft/hard 均未命中时，再看粗粒度：同一目标资源反复翻页/微调
        // 检索参数的膨胀会在这里被抓到。仅提示一次，且让位于精确检测。
        if !self.coarse_loop_note_injected
            && !self.loop_breaker_injected
            && detect_tool_loop(&self.tool_signature_history_coarse, TOOL_LOOP_COARSE_WINDOW)
        {
            self.coarse_loop_note_injected = true;
            return ToolLoopSignal::Coarse;
        }
        ToolLoopSignal::None
    }

    /// 中段断路器：单轮迭代到达软阈值时注入一次收敛提示（远早于 max_iterations
    /// 硬上限），治理「合法但低效」的工具刷屏。仅注入一次。
    fn maybe_inject_iteration_soft_limit_note(
        &mut self,
        messages: &mut Vec<crate::ai::history::Message>,
        max_iterations: usize,
    ) -> bool {
        if self.iteration_soft_limit_note_injected {
            return false;
        }
        let soft_limit = tool_iteration_soft_limit(max_iterations);
        if self.iteration < soft_limit {
            return false;
        }
        self.iteration_soft_limit_note_injected = true;
        inject_iteration_soft_limit_note(messages, self.iteration);
        true
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

/// 把 mid-turn 压缩状态以 status line 形式输出到终端。
fn print_compress_status(stage: &str, before: usize, after: usize) {
    crate::ai::driver::print::print_tool_note_line(
        "compress",
        &format!("{stage}: {} → {} chars", before, after),
    );
}

/// 工具循环检测命中后，向 messages 注入一条 internal_note 让 agent 自我反思
/// （而非直接 force_final，给 agent 一个跳出循环的机会）。
fn inject_loop_breaker_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[loop-detected] 你最近 4 轮都在用相同参数调用相同工具，明显在打转。\n\
        请立刻：(a) 停止该工具调用 (b) 总结已收集到的信息并直接回答用户 \
        (c) 如果信息确实不足，向用户说明卡住的原因。";
    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

fn inject_hard_loop_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[loop-hard-stop] 你已连续 3 轮用相同参数调用相同工具，判定为无效循环。\n\
        从现在起不要再发起任何工具调用：请基于已有信息直接回答用户；\n\
        如果信息仍不足，明确说明缺口与建议的下一步。";
    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 近似循环命中：同一工具反复命中同一目标资源（仅翻页/检索参数在变）。提醒
/// agent 一次性读大块或收敛检索，而非逐页刷屏。软提示，不强制收敛。
fn inject_coarse_loop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[loop-approx] 你最近多轮都在对同一目标反复调用同一工具，只是翻页/检索参数在微调，效率很低。\n\
        请改为：(a) 一次读取更大的行范围（提高 read_file 的 limit）或用检索工具一次定位，而不是逐页翻；\n\
        (b) 复用已读到的内容，不要重复读同一文件同一段；(c) 若已够回答就直接作答。";
    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 中段断路器：单轮迭代到达软阈值（远早于 max_iterations）时的一次性收敛提示。
fn inject_iteration_soft_limit_note(
    messages: &mut Vec<crate::ai::history::Message>,
    iteration: usize,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[iteration-soft-limit] 你已经在本轮内迭代 {iteration} 次工具调用，明显偏多。\n\
        请先停下来复盘：(a) 总结目前已确认的事实与还差什么；(b) 明确下一步唯一动作；\n\
        (c) 优先用批量/大范围读取与精准检索收敛，避免继续碎片化地翻页或反复搜索；\n\
        (d) 若已经足够回答，就直接给出结论。"
    );
    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// max_iterations 触发后的自反思 prompt（替代纯 force_final 举手投降）。
fn inject_iteration_limit_reflect_note(
    messages: &mut Vec<crate::ai::history::Message>,
    max_iterations: usize,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[iteration-limit] 你已经迭代 {max_iterations} 轮但仍未收敛。\n\
        请用现有信息直接回答用户。如果信息不足，请明确告诉用户卡在哪里、\
        缺什么资料、建议下一步怎么做——不要再发起任何工具调用。"
    );
    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn detect_tool_loop_ignores_empty_signatures() {
        let history = vec![Vec::<String>::new(); TOOL_LOOP_HARD_WINDOW];
        assert!(!detect_tool_loop(&history, TOOL_LOOP_SOFT_WINDOW));
        assert!(!detect_tool_loop(&history, TOOL_LOOP_HARD_WINDOW));
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

        // 收集每一轮的信号：前 SOFT_WINDOW-1 轮不触发，第 SOFT_WINDOW 轮触发 Soft，
        // 第 HARD_WINDOW 轮触发 Hard。
        let mut signals = Vec::new();
        for i in 0..TOOL_LOOP_HARD_WINDOW {
            messages.push(assistant_with_same_read(&format!("tc-{i}")));
            signals.push(supervisor.record_tool_signatures(&messages));
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
            signals[TOOL_LOOP_HARD_WINDOW - 1],
            ToolLoopSignal::Hard
        ));
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
                    arguments: "{\"path\":\"src/main.rs\"}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        };

        // 第一轮：积累到 soft 触发。
        for i in 0..TOOL_LOOP_SOFT_WINDOW {
            messages.push(assistant_with_read(&format!("tc-{i}")));
            supervisor.record_tool_signatures(&messages);
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
        let signal = supervisor.record_tool_signatures(&messages);
        assert!(matches!(signal, ToolLoopSignal::None));
        assert!(supervisor.tool_signature_history.is_empty());
        assert_eq!(supervisor.skip_tool_signature_rounds, 0);

        // 第二轮：恢复后重新积累，验证 soft 能再次触发。
        for i in 0..TOOL_LOOP_SOFT_WINDOW {
            messages.push(assistant_with_read(&format!("tc2-{i}")));
            let signal = supervisor.record_tool_signatures(&messages);
            if i == TOOL_LOOP_SOFT_WINDOW - 1 {
                // 第 4 次应触发 soft。
                assert!(matches!(signal, ToolLoopSignal::Soft));
                assert!(supervisor.loop_breaker_injected);
            } else {
                assert!(matches!(signal, ToolLoopSignal::None));
            }
        }

        // 继续积累到 hard 触发，验证完整升级阶梯恢复。
        for i in 0..(TOOL_LOOP_HARD_WINDOW - TOOL_LOOP_SOFT_WINDOW) {
            messages.push(assistant_with_read(&format!("tc3-{i}")));
            let signal = supervisor.record_tool_signatures(&messages);
            if i == (TOOL_LOOP_HARD_WINDOW - TOOL_LOOP_SOFT_WINDOW - 1) {
                // 第 6 次应触发 hard。
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
            signals.push(supervisor.record_tool_signatures(&messages));
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
            supervisor.record_tool_signatures(&messages),
            ToolLoopSignal::None
        ));
    }

    #[test]
    fn tool_iteration_soft_limit_clamps_to_absolute_ceiling() {
        assert_eq!(tool_iteration_soft_limit(0), 1);
        assert_eq!(tool_iteration_soft_limit(1), 1);
        assert_eq!(tool_iteration_soft_limit(10), 5);
        assert_eq!(tool_iteration_soft_limit(768), TOOL_ITERATION_SOFT_LIMIT);
        assert_eq!(
            tool_iteration_soft_limit(100_000),
            TOOL_ITERATION_SOFT_LIMIT
        );
    }

    #[test]
    fn iteration_soft_limit_note_injected_once() {
        let mut supervisor = TurnSupervisor::default();
        let mut messages = Vec::new();
        let max_iterations = 10; // soft_limit = 5
        // 未到软阈值不注入
        supervisor.iteration = 4;
        assert!(!supervisor.maybe_inject_iteration_soft_limit_note(&mut messages, max_iterations));
        assert!(messages.is_empty());
        // 到达软阈值注入一次
        supervisor.iteration = 5;
        assert!(supervisor.maybe_inject_iteration_soft_limit_note(&mut messages, max_iterations));
        assert_eq!(messages.len(), 1);
        assert!(
            messages[0]
                .content
                .as_str()
                .unwrap_or_default()
                .contains("[iteration-soft-limit]")
        );
        // 再次调用不重复注入
        supervisor.iteration = 8;
        assert!(!supervisor.maybe_inject_iteration_soft_limit_note(&mut messages, max_iterations));
        assert_eq!(messages.len(), 1);
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
    // 把 (session_id, turn_id) 注入 task_local，让下游工具调用与反馈
    // 写入路径能拿到正确身份。turn_id 复用 history_count（每个 turn 在
    // 进入时的 history 长度，全 session 内单调递增）。
    let session_id = app.session_id.clone();
    let turn_id = history_count;
    // 仅前台主 turn 抬起「turn 活动」标志：子 agent（sync / background）持有私有
    // 信号标志，且都通过 SUBAGENT_RESULT_SLOT 作用域执行，据此排除。该标志让
    // prepare / 思考 / 阶段切换 / mid-turn 压缩等 streaming=false 的空窗里的
    // Ctrl+C 也走「取消本轮」而非「退出会话」。guard 随本 future drop 自动落下。
    let _foreground_turn_guard = (!crate::ai::driver::runtime_ctx::has_subagent_result_slot())
        .then(crate::ai::driver::signal::ForegroundTurnGuard::enter);
    crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .scope(
            (session_id, turn_id),
            run_turn_body(
                app,
                mcp_client,
                skill_manifests,
                history_count,
                question,
                attachments_text,
                next_model,
                precomputed_ocr,
                one_shot_mode,
                should_quit,
            ),
        )
        .await
}

#[allow(clippy::too_many_arguments)]
async fn run_turn_body(
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

    let mut supervisor = TurnSupervisor::default();
    let mut force_final_response = false;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut terminal_dedupe_candidate = None;
    // 收集本 turn 实际调用过的 explicit-enabled tool 名字，turn 末用于老化未用项。
    let mut tools_used_this_turn: rust_tools::cw::SkipSet<String> =
        rust_tools::cw::SkipSet::default();
    let mut consecutive_empty_responses: usize = 0;
    let mut consecutive_truncations: usize = 0;
    // 独立统计"流读取中断型"截断（stream_error）。它与模型输出过长无关（网络抖动 /
    // 服务端断流），因此不参与 reasoning 降档，也不累加 consecutive_truncations；
    // 但持续断流仍需有上限兜底，否则 usize::MAX 迭代预算的后台任务会无限重试。
    let mut consecutive_stream_errors: usize = 0;
    let mut turn_had_tool_error = false;
    // 保存进入本 turn 时的 reasoning effort 覆盖值（可能是用户 `/model effort` 的
    // 显式选择，或 None=用模型默认）。截断重试时会临时把它降到 Low，把输出 token
    // 预算从 reasoning 让给实际内容；turn 结束后（含所有 break 出口）统一恢复，
    // 不污染用户的会话级设置。
    let saved_effort_override = app.cli.reasoning_effort_override;
    // 同理保存 thinking 兜底开关：截断重试可能置位它以强制关闭 always-thinking
    // 模型的思考链，turn 末统一恢复，不污染后续 turn。
    let saved_thinking_disabled = app.cli.thinking_disabled_override;
    let loop_result = 'turn: loop {
        let iteration = supervisor.next_iteration();
        {
            let mc = mcp_client.lock().unwrap();
            refresh_skill_turn_for_iteration(
                app,
                &mc,
                skill_manifests,
                &question,
                iteration,
                &mut skill_turn,
                &mut messages,
            );
        }
        let active_skill_name = skill_turn.matched_skill_name().map(str::to_string);
        let execution = match execute_turn_iteration(
            app,
            &next_model,
            &mut messages,
            &turn_messages,
            one_shot_mode,
            &mut persisted_turn_messages,
            should_quit,
            force_final_response,
            terminal_dedupe_candidate.as_deref(),
            active_skill_name.as_deref(),
            iteration,
        )
        .await
        {
            Ok(e) => e,
            Err(err) => break 'turn Err(err),
        };
        {
            let mc = mcp_client.lock().unwrap().routing_snapshot();
            // 空响应重试计数：连续 >2 次空响应则放弃，避免浪费迭代预算
            if matches!(execution, IterationExecution::EmptyResponse) {
                consecutive_empty_responses += 1;
                if consecutive_empty_responses > 5 {
                    let _ = writeln!(std::io::stderr(), "  ✗ 连续 {} 次空响应，停止重试", consecutive_empty_responses);
                    final_assistant_text = "[模型连续返回空响应，请重试或切换模型]".to_string();
                    break 'turn Ok(None);
                }
            } else {
                consecutive_empty_responses = 0;
            }
            // 截断重试计数：连续多次被截断（输出上限/工具 JSON 半截）仍无法收敛时
            // 放弃，避免无限重试烧预算。阈值取 3：给模型两次收缩重写的机会。
            if let IterationExecution::Truncated(stream_result) = &execution {
                consecutive_truncations += 1;
                // 重置工具循环检测：截断重试期间的重复调用是预期行为，
                // 不应被误判为工具死循环并触发 hard-stop 强制收敛。
                supervisor.mark_truncation_skip();

                if stream_result.stream_error {
                    // 流读取错误（网络抖动 / 服务端异常断流）导致的截断。
                    // 模型并没有输出太多，降 reasoning_effort 和注入收缩提示都无意义。
                    // 不累积 consecutive_truncations（这不是模型的错），但用独立计数
                    // consecutive_stream_errors 兜底，避免服务端持续断流时无限重试。
                    consecutive_truncations = 0;
                    consecutive_stream_errors += 1;
                    if consecutive_stream_errors > MAX_STREAM_ERROR_RETRIES {
                        let _ = writeln!(
                            std::io::stderr(),
                            "  ✗ 连续 {} 次响应流读取中断，停止重试",
                            consecutive_stream_errors
                        );
                        final_assistant_text =
                            "[响应流多次读取中断，疑似服务端不稳定，请稍后重试或切换模型]"
                                .to_string();
                        break 'turn Ok(None);
                    }
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ⚠ 响应流读取中断（疑似服务端不稳定），自动重试…"
                    );
                } else {
                    // 真截断：模型撞输出上限或工具 JSON 半截。
                    consecutive_stream_errors = 0;

                    // 该模型降 reasoning_effort 是否真能缩短思考链。
                    // enable_thinking 布尔开关方言（GLM 等）请求体里根本不带
                    // effort，降档是空操作——必须直接关 thinking 才能腾出输出预算。
                    // dialect 分派与请求层一致：用 request_model_name + endpoint。
                    let endpoint = crate::ai::models::endpoint_for_model(&next_model, "");
                    let provider = crate::ai::models::model_provider(&next_model);
                    let request_model = crate::ai::models::request_model_name(&next_model);
                    let effort_helps = crate::ai::provider::reasoning_effort_reduces_thinking_for(
                        provider, &request_model, &endpoint,
                    );

                    if effort_helps {
                        // 渐进式 reasoning effort 降档，把输出预算从 reasoning 让给实际内容。
                        // resolve_reasoning_effort 每次迭代实时读该字段，改了立即对下一次生效。
                        //
                        // 1 次截断 → Low（减半推理开销）
                        // 2 次截断 → Minimal（仅保留最低推理）
                        // 3 次以上 → 完全禁用 reasoning（把全部输出预算让给可见内容）
                        app.cli.reasoning_effort_override = Some(match consecutive_truncations {
                            1 => Some(crate::ai::provider::ReasoningEffort::Low),
                            2 => Some(crate::ai::provider::ReasoningEffort::Minimal),
                            _ => None, // Some(None) = 禁用 reasoning，不下发 effort 字段
                        });
                    } else {
                        // 降 effort 对该方言无效：不浪费重试轮次走无效阶梯，
                        // 首次真截断即强制关闭 thinking，把整个输出预算让给可见内容。
                        app.cli.thinking_disabled_override = true;
                    }
                }

                let partial_text = stream_result.assistant_text.trim();
                let has_visible_text = !partial_text.is_empty();

                // 模型已产出可见文本但仍连续撞长度上限（典型：推理模型 reasoning 占满
                // 预算）。继续重试通常无帮助——模型会反复产出同样长度的内容。
                // 给一次降档重试机会后即接受部分文本作为最终回答。
                // 但 stream_error 场景不计入 consecutive_truncations，不会触发此分支。
                if has_visible_text && consecutive_truncations >= 16 && !stream_result.stream_error {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ▲ 连续 {} 次输出被截断，保留已产出的部分文本",
                        consecutive_truncations
                    );
                    final_assistant_text = partial_text.to_string();
                    break 'turn Ok(None);
                }

                // stream_error 已在上面重置 consecutive_truncations=0，不会进入此分支。
                if consecutive_truncations >= 16 && !stream_result.stream_error {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ✗ 连续 {} 次响应被截断，停止重试",
                        consecutive_truncations
                    );
                    // 保留模型已产出的部分文本（若有），比直接丢弃更有价值。
                    final_assistant_text = if has_visible_text {
                        partial_text.to_string()
                    } else {
                        "[模型输出多次被截断，请缩小单次操作规模（如分块写文件）或切换模型]"
                            .to_string()
                    };
                    break 'turn Ok(None);
                }
            } else {
                consecutive_truncations = 0;
            }
            let step = match handle_iteration_execution(
                app, &question, &mc, mcp_client, execution,
                &mut messages, &mut turn_messages, one_shot_mode,
                &mut persisted_turn_messages, &mut final_assistant_text,
                &mut final_assistant_recorded, &mut force_final_response,
                &mut terminal_dedupe_candidate,
                skill_turn.matched_skill_name().is_none(), iteration, max_iterations,
                consecutive_truncations,
                &mut turn_had_tool_error,
            ) {
                Ok(s) => s,
                Err(err) => break 'turn Err(err),
            };
            match step {
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
                    // 记录本轮 assistant 实际调用过的 tool 名字（去重），
                    // 留给 turn 末用于老化未用 explicit tool。
                    if let Some(last_assistant) =
                        messages.iter().rev().find(|m| m.role == "assistant")
                        && let Some(tool_calls) = &last_assistant.tool_calls
                    {
                        for tc in tool_calls {
                            tools_used_this_turn.insert(tc.function.name.clone());
                        }
                    }
                }
                TurnLoopStep::Break => break 'turn Ok(None),
                TurnLoopStep::Return(outcome) => break 'turn Ok(Some(outcome)),
            }
        }
        // ↓↓↓ Continue 分支的后续处理（已离开 mc 锁，可以安全 await）↓↓↓

        // === Mid-turn 渐进式压缩 ===
        // 每轮 tool 执行完毕后检查 messages 总字符；超过软阈值时
        // 复用跨 turn 压缩管线，避免长链工具调用把上下文撑爆。
        // 节流：①冷却 N 轮 ②增量小于 DELTA 时跳过，避免 no-op 反复压缩。
        // 阈值按 history_max_chars 动态计算（floor 兜底），避免用户调整
        // history_max_chars 后 mid-turn 阈值依旧死锁在 36K/80K。
        let history_max_chars = app.config.history_max_chars;
        let mid_turn_soft = mid_turn_compress_soft_threshold(&next_model, history_max_chars);
        let mid_turn_hard = mid_turn_compress_hard_threshold(&next_model, history_max_chars);
        let total_chars = crate::ai::history::messages_total_chars_pub(&messages);
        if supervisor.should_try_mid_turn_compress(total_chars, mid_turn_soft) {
            // 与跨 turn 压缩（prepare.rs）一致地解析会话 overflow 目录：mid-turn
            // 压缩据此把 read_file/grep 等「不可压缩」工具的大输出零压缩外溢到
            // 文件 + 留预览 stub，既释放上下文又不丢信息（模型可重新 read_file）。
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
            );
            messages = compressed;
            supervisor.mark_compress(after);
            if after < before {
                print_compress_status("mid-turn", before, after);
            }
            // 硬阈值：无损 + 弱损管线之后仍超额，调用 LLM 摘要兜底，
            // 把早期对话压成单条 internal_note，并在终端打 status line。
            if after > mid_turn_hard {
                crate::ai::driver::print::print_tool_note_line(
                    "compress",
                    &format!(
                        "hard threshold exceeded ({after} > {mid_turn_hard}), \
                         requesting LLM summary…"
                    ),
                );
                let drained: Vec<crate::ai::history::Message> = std::mem::take(&mut messages);
                let (after_msgs, llm_before, llm_after, did_summarize) =
                    crate::ai::history::mid_turn_llm_summarize(
                        app,
                        drained,
                        MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
                        MID_TURN_LLM_SUMMARY_MAX_CHARS,
                        history_max_chars,
                    )
                    .await;
                messages = after_msgs;
                if did_summarize {
                    print_compress_status("mid-turn (llm)", llm_before, llm_after);
                } else {
                    crate::ai::driver::print::print_tool_note_line(
                        "compress",
                        "llm summary skipped (no early dialog or call failed); \
                         agent may hit context limit",
                    );
                }
            }
        }

        // === 工具循环检测 ===
        match supervisor.record_tool_signatures(&messages) {
            ToolLoopSignal::None => {}
            ToolLoopSignal::Coarse => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "approx tool-loop detected (same target, paging only): injecting converge prompt",
                );
                inject_coarse_loop_note(&mut messages);
            }
            ToolLoopSignal::Soft => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "tool-loop detected: injecting self-reflect prompt",
                );
                inject_loop_breaker_note(&mut messages);
                // 高风险异常才注入一次任务锚点，降低目标漂移概率。
                supervisor.maybe_inject_task_anchor(&mut messages, &question, "tool-loop");
            }
            ToolLoopSignal::Hard => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "tool-loop hard-stop: forcing final response",
                );
                inject_hard_loop_stop_note(&mut messages);
                supervisor.maybe_inject_task_anchor(
                    &mut messages,
                    &question,
                    "tool-loop-hard-stop",
                );
                force_final_response = true;
            }
        }

        // === 中段迭代断路器 ===
        // 远早于 max_iterations 硬上限：单轮迭代到达软阈值时注入一次收敛提示，
        // 治理「合法但低效」的工具刷屏（翻页、碎片检索），不强制收敛。
        if supervisor.maybe_inject_iteration_soft_limit_note(&mut messages, max_iterations) {
            crate::ai::driver::print::print_tool_note_line(
                "agent-health",
                "tool-iteration soft limit reached: injecting converge prompt",
            );
        }

        // === Iteration limit 自反思 ===
        // execute.rs 在 iteration >= max_iterations 时会把
        // force_final_response 置 true。此时除原有的 "Tool limit reached"
        // system prompt 外，再额外补一条更具体的反思 prompt
        // （只注入一次，避免重复刷屏）。
        supervisor.maybe_inject_iteration_limit_note(
            &mut messages,
            max_iterations,
            force_final_response,
        );
        if force_final_response {
            supervisor.maybe_inject_task_anchor(&mut messages, &question, "iteration-limit");
        }
    };

    // 恢复进入本 turn 前的 reasoning effort 覆盖值：截断重试可能把它临时降到了
    // Low，这里统一还原（覆盖所有 break 'turn 出口），避免把降档泄漏到后续 turn
    // 污染用户的会话级设置。
    app.cli.reasoning_effort_override = saved_effort_override;
    app.cli.thinking_disabled_override = saved_thinking_disabled;

    // 老化未在本 turn 使用的 explicit-enabled tool。
    // 连续 N 个 turn 闲置就 demote，避免"启用一次永久焊接"。
    crate::ai::tools::enable_tools::age_unused_explicit_tools(tools_used_this_turn.iter());

    skill_turn.restore_agent_context(app);

    let loop_result = loop_result.map_err(|e: Box<dyn std::error::Error>| e.to_string());

    match loop_result {
        Ok(Some(outcome)) => Ok(outcome),
        Ok(_) => {
            finalize_turn(
                app,
                &next_model,
                &question,
                &final_assistant_text,
                final_assistant_recorded,
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
