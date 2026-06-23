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

use crate::ai::{mcp::SharedMcpClient, types::App};

use super::{
    MID_TURN_COMPRESS_COOLDOWN_ITERATIONS, MID_TURN_COMPRESS_DELTA_THRESHOLD,
    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS, MID_TURN_LLM_SUMMARY_MAX_CHARS,
    finalize::finalize_turn,
    iteration::{execute_turn_iteration, refresh_skill_turn_for_iteration},
    mid_turn_compress_hard_threshold, mid_turn_compress_soft_threshold,
    prepare::prepare_turn,
    tool_result::handle_iteration_execution,
    types::{TurnLoopStep, TurnOutcome, TurnPreparation},
};

/// 工具调用循环检测窗口：
/// - soft: 连续 2 轮调用 (tool_name, normalized_args) 完全一致，先注入反思提示
/// - hard: 连续 3 轮完全一致，直接强制收敛，不再继续工具循环
const TOOL_LOOP_SOFT_WINDOW: usize = 2;
const TOOL_LOOP_HARD_WINDOW: usize = 3;
const TOOL_SIGNATURE_HISTORY_LIMIT: usize = TOOL_LOOP_HARD_WINDOW + 2;
const TASK_ANCHOR_MAX_QUESTION_CHARS: usize = 220;

/// 提取最近一轮 assistant 消息中的 (tool_name, args_json) 签名集合。
/// 任何一个签名与窗口内某轮完全一致即认为有循环倾向。
fn extract_round_tool_signatures(messages: &[crate::ai::history::Message]) -> Option<Vec<String>> {
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
        // 归一化 args：解析为 Value 后再 to_string，去掉空白噪音
        let args_norm = serde_json::from_str::<Value>(args_raw)
            .map(|v| v.to_string())
            .unwrap_or_else(|_| args_raw.to_string());
        sigs.push(format!("{name}::{args_norm}"));
    }
    sigs.sort();
    Some(sigs)
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
    loop_breaker_injected: bool,
    hard_loop_stop_injected: bool,
    iteration_limit_note_injected: bool,
    task_anchor_injected: bool,
    last_compress_iteration: usize,
    last_compress_after_chars: usize,
    tool_signature_history: Vec<Vec<String>>,
}

enum ToolLoopSignal {
    None,
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

    fn record_tool_signatures(
        &mut self,
        messages: &[crate::ai::history::Message],
    ) -> ToolLoopSignal {
        let Some(sigs) = extract_round_tool_signatures(messages) else {
            return ToolLoopSignal::None;
        };
        self.tool_signature_history.push(sigs);
        if self.tool_signature_history.len() > TOOL_SIGNATURE_HISTORY_LIMIT {
            let drop = self.tool_signature_history.len() - TOOL_SIGNATURE_HISTORY_LIMIT;
            self.tool_signature_history.drain(0..drop);
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
        ToolLoopSignal::None
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
        let history = vec![sig.clone(), sig.clone()];
        assert!(detect_tool_loop(&history, TOOL_LOOP_SOFT_WINDOW));
        assert!(!detect_tool_loop(&history, TOOL_LOOP_HARD_WINDOW));
        // 满窗口且完全相同
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

        messages.push(assistant_with_same_read("tc-1"));
        assert!(matches!(
            supervisor.record_tool_signatures(&messages),
            ToolLoopSignal::None
        ));
        messages.push(assistant_with_same_read("tc-2"));
        assert!(matches!(
            supervisor.record_tool_signatures(&messages),
            ToolLoopSignal::Soft
        ));
        messages.push(assistant_with_same_read("tc-3"));
        assert!(matches!(
            supervisor.record_tool_signatures(&messages),
            ToolLoopSignal::Hard
        ));
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
            iteration,
        )
        .await
        {
            Ok(e) => e,
            Err(err) => break 'turn Err(err),
        };
        {
            let mc = mcp_client.lock().unwrap().routing_snapshot();
            let step = match handle_iteration_execution(
                app,
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
                skill_turn.matched_skill_name().is_none(),
                iteration,
                max_iterations,
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
        let mid_turn_soft = mid_turn_compress_soft_threshold(history_max_chars);
        let mid_turn_hard = mid_turn_compress_hard_threshold(history_max_chars);
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
            )
            .await
        }
        Err(err_str) => Err(err_str.into()),
    }
}
