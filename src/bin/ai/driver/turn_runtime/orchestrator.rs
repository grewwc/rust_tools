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
/// 近似低收益重复窗口：连续 N 轮对「同一目标资源」调用同一工具（忽略
/// offset/limit 等翻页参数）即命中。用于抓字节精确检测漏掉的「同文件反复
/// 翻页 / 仅微调分页参数的重复检索」这类真实膨胀。先注入一次温和提示；
/// 若继续长时间刷同一 coarse 目标（尤其 `execute_command` 的目录探测类命令），
/// 则升级为 hard-stop，避免白跑上百轮。
const TOOL_LOOP_COARSE_WINDOW: usize = 5;
const TOOL_LOOP_COARSE_HARD_WINDOW: usize = 8;
const TOOL_SIGNATURE_HISTORY_LIMIT: usize = TOOL_LOOP_COARSE_HARD_WINDOW + 2;
const TASK_ANCHOR_MAX_QUESTION_CHARS: usize = 220;

/// 计算 coarse 签名时需剥离的「易变翻页/窗口」参数键。剥离后同一文件的不同
/// 分页、同一检索的不同结果上限会折叠成同一 coarse 签名。
const VOLATILE_ARG_KEYS: &[&str] = &["offset", "limit", "page", "cursor", "max_results"];

/// 中段断路器绝对上限：单轮工具迭代达到该值即注入一次收敛提示，远早于
/// max_iterations 硬上限，用于治理「合法但低效」的工具刷屏。
/// 默认 turn 预算是 2048；若把软提示 cap 放到几百轮，像日志排查这类探索型
/// 问题会先空转一两百轮才被提醒收敛，明显过晚。
const TOOL_ITERATION_SOFT_LIMIT: usize = 128;

/// 连续「流读取中断型」截断（stream_error）的重试上限。超过即放弃本 turn，
/// 避免服务端持续断流时无限重试（尤其后台任务的 max_iterations = usize::MAX）。
const MAX_STREAM_ERROR_RETRIES: usize = 16;

/// 单轮工具迭代软阈值：取 max_iterations 的一半与绝对上限的较小值，保证一定
/// 早于硬上限触发；对默认 2048 这类超大 ceiling，也要在明显失控前提醒收敛。
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
/// 对 `execute_command` 额外折叠 shell 中的低收益变体（如 `| head -20/-30`、
/// `2>/dev/null`、`ls -la/-lt` 的细微差异），让同目标资源的反复试探能命中。
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
                    normalize_coarse_tool_args(name, &mut v);
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

fn normalize_coarse_tool_args(tool_name: &str, value: &mut serde_json::Value) {
    if tool_name != "execute_command" {
        return;
    }
    let Some(map) = value.as_object_mut() else {
        return;
    };
    let cwd = map
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(normalize_path_like_token);
    let Some(command) = map
        .get("command")
        .and_then(|v| v.as_str())
        .map(coarse_execute_command_signature)
    else {
        return;
    };
    map.clear();
    map.insert("command".to_string(), serde_json::Value::String(command));
    if let Some(cwd) = cwd {
        map.insert("cwd".to_string(), serde_json::Value::String(cwd));
    }
}

fn coarse_execute_command_signature(command: &str) -> String {
    let mut parts = Vec::new();
    for segment in split_shell_segments_for_coarse(command) {
        if let Some(sig) = coarse_shell_segment_signature(&segment) {
            if parts.last() != Some(&sig) {
                parts.push(sig);
            }
        }
    }
    if parts.is_empty() {
        return truncate_chars(command.trim(), 160);
    }
    parts.join(" | ")
}

fn split_shell_segments_for_coarse(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    while let Some(ch) = chars.next() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if !in_single => {
                current.push(ch);
                escaped = true;
            }
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(ch);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(ch);
            }
            ';' | '|' | '&' if !in_single && !in_double => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    segments.push(trimmed.to_string());
                }
                current.clear();
                if matches!(ch, '|' | '&') && chars.peek() == Some(&ch) {
                    chars.next();
                }
            }
            _ => current.push(ch),
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        segments.push(trimmed.to_string());
    }
    segments
}

fn tokenize_shell_words_for_coarse(command: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut token_started = false;

    for ch in command.chars() {
        if escaped {
            current.push(ch);
            token_started = true;
            escaped = false;
            continue;
        }
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                current.push(ch);
            }
            token_started = true;
            continue;
        }
        if in_double {
            match ch {
                '"' => in_double = false,
                '\\' => escaped = true,
                _ => current.push(ch),
            }
            token_started = true;
            continue;
        }

        if ch.is_whitespace() {
            if token_started {
                tokens.push(std::mem::take(&mut current));
                token_started = false;
            }
            continue;
        }

        match ch {
            '\'' => {
                in_single = true;
                token_started = true;
            }
            '"' => {
                in_double = true;
                token_started = true;
            }
            '\\' => {
                escaped = true;
                token_started = true;
            }
            _ => {
                current.push(ch);
                token_started = true;
            }
        }
    }

    if escaped {
        current.push('\\');
    }
    if token_started {
        tokens.push(current);
    }
    tokens
}

fn coarse_shell_segment_signature(segment: &str) -> Option<String> {
    let tokens = tokenize_shell_words_for_coarse(segment);
    let program = tokens.first()?.to_ascii_lowercase();
    if is_window_only_shell_segment(&program, &tokens) {
        return None;
    }
    match program.as_str() {
        "ls" => Some(normalize_ls_segment(&tokens)),
        "grep" | "rg" => Some(normalize_search_segment(&program, &tokens)),
        _ => Some(normalize_generic_shell_segment(&program, &tokens)),
    }
}

fn is_window_only_shell_segment(program: &str, tokens: &[String]) -> bool {
    match program {
        "head" | "tail" => tokens[1..]
            .iter()
            .all(|token| token.starts_with('-') || token.chars().all(|ch| ch.is_ascii_digit())),
        "wc" => tokens[1..].iter().all(|token| token.starts_with('-')),
        _ => false,
    }
}

fn normalize_ls_segment(tokens: &[String]) -> String {
    let mut paths = collect_shell_target_tokens(tokens, 1, false);
    if paths.is_empty() {
        paths.push(".".to_string());
    }
    format!("ls:{}", paths.join(","))
}

fn normalize_search_segment(program: &str, tokens: &[String]) -> String {
    let mut pattern = None;
    let mut paths = Vec::new();
    let mut expect_option_value = false;
    let mut after_double_dash = false;
    for token in tokens.iter().skip(1) {
        if should_skip_shell_token(token) {
            continue;
        }
        if expect_option_value {
            if !token.chars().all(|ch| ch.is_ascii_digit()) && pattern.is_none() {
                pattern = Some(token.to_string());
            }
            expect_option_value = false;
            continue;
        }
        if !after_double_dash && token == "--" {
            after_double_dash = true;
            continue;
        }
        if !after_double_dash && token.starts_with('-') {
            if matches!(
                token.as_str(),
                "-e" | "--regexp" | "-f" | "--file" | "-g" | "--glob" | "--iglob"
            ) {
                expect_option_value = true;
            }
            continue;
        }
        if looks_like_path_token(token) {
            paths.push(normalize_path_like_token(token));
            continue;
        }
        if pattern.is_none() {
            pattern = Some(token.to_string());
        }
    }
    if paths.is_empty() {
        paths.push("<stdin>".to_string());
    }
    match pattern {
        Some(pattern) => format!("{program}:{}#{pattern}", paths.join(",")),
        None => format!("{program}:{}", paths.join(",")),
    }
}

fn normalize_generic_shell_segment(program: &str, tokens: &[String]) -> String {
    let mut paths = collect_shell_target_tokens(tokens, 1, true);
    if paths.is_empty() {
        program.to_string()
    } else {
        paths.sort();
        paths.dedup();
        format!("{program}:{}", paths.join(","))
    }
}

fn collect_shell_target_tokens(
    tokens: &[String],
    start: usize,
    keep_literals: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip_next = false;
    for token in tokens.iter().skip(start) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if should_skip_shell_token(token) {
            continue;
        }
        if token == ">" || token == ">>" || token == "<" || token == "<<" {
            skip_next = true;
            continue;
        }
        if token.starts_with('-') {
            continue;
        }
        if looks_like_path_token(token) {
            out.push(normalize_path_like_token(token));
            continue;
        }
        if keep_literals && !token.chars().all(|ch| ch.is_ascii_digit()) {
            out.push(token.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

fn should_skip_shell_token(token: &str) -> bool {
    matches!(token, "|" | ";" | "&&" | "||" | "&")
        || token.starts_with("2>")
        || token.starts_with("1>")
        || token.starts_with(">")
        || token.starts_with("<")
}

fn looks_like_path_token(token: &str) -> bool {
    token == "."
        || token == ".."
        || token.starts_with('/')
        || token.starts_with("./")
        || token.starts_with("../")
        || token.starts_with("~/")
        || token.contains('/')
}

fn normalize_path_like_token(token: &str) -> String {
    let mut out = String::with_capacity(token.len());
    let mut prev_slash = false;
    for ch in token.trim().chars() {
        if ch == '/' {
            if !prev_slash {
                out.push(ch);
            }
            prev_slash = true;
        } else {
            out.push(ch);
            prev_slash = false;
        }
    }
    while out.len() > 1 && out.ends_with('/') {
        out.pop();
    }
    out
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

fn signature_set_is_execute_command_only(sigs: &[String]) -> bool {
    !sigs.is_empty() && sigs.iter().all(|sig| sig.starts_with("execute_command::"))
}

fn detect_execute_command_coarse_loop(history: &[Vec<String>], window: usize) -> bool {
    if !detect_tool_loop(history, window) {
        return false;
    }
    let tail = &history[history.len() - window..];
    signature_set_is_execute_command_only(&tail[0])
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
    /// 近似低收益重复：同一工具反复命中同一目标资源（忽略翻页参数）。温和提示一次。
    Coarse,
    /// `execute_command` 在同一 coarse 目标上长时间空转，直接强制收敛。
    CoarseHard,
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
        if !self.hard_loop_stop_injected
            && detect_execute_command_coarse_loop(
                &self.tool_signature_history_coarse,
                TOOL_LOOP_COARSE_HARD_WINDOW,
            )
        {
            self.hard_loop_stop_injected = true;
            return ToolLoopSignal::CoarseHard;
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
        从现在起进入无工具收口模式：不要再发起任何工具调用；\n\
        请基于已有信息给出阶段总结与当前结论；若任务仍未完成，明确说明缺口、剩余工作与建议的下一步。";
    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 近似低收益重复命中：同一工具反复命中同一目标资源（仅翻页/检索参数在变）。
/// 提醒 agent 判断这些调用是否真的在推进问题；若只是碎片化翻页则收敛，
/// 若各轮服务于不同且明确的子问题则允许继续。软提示，不强制收敛。
fn inject_coarse_loop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-yield-repetition] 你最近多轮都在对同一目标调用同一工具，主要变化只是翻页/检索窗口参数。\n\
        这常常意味着低收益重复，但不一定是错误：如果这些调用分别服务于不同且明确的子问题，可以继续；\n\
        否则请优先：(a) 一次读取更大的行范围（提高 read_file 的 limit）或用检索工具一次定位；\n\
        (b) 复用已读到的内容，不要重复读同一文件同一段；(c) 若信息已足够，就直接作答。";
    messages.push(Message {
        role: "system".to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 低收益的 `execute_command` 粗粒度重复升级到 hard-stop：在同一 coarse 目标上
/// 连续多轮只改窗口/排序细节，基本可判定为无效探索。
fn inject_coarse_hard_loop_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-yield-hard-stop] 你已连续多轮对同一目标重复调用 `execute_command`，变化主要只是窗口/排序细节，判定为无效探索。\n\
        从现在起进入无工具收口模式：不要再发起任何工具调用；\n\
        请基于已有信息给出阶段总结与当前结论；若任务仍未完成，明确说明当前缺口、剩余工作与建议的下一步。";
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
    fn coarse_execute_command_signature_collapses_log_listing_window_variants() {
        let a = coarse_execute_command_signature("ls -lt /data01/dataagent_be/logs/ | head -20");
        let b = coarse_execute_command_signature("ls -la /data01/dataagent_be/logs/ | head -30");
        let c = coarse_execute_command_signature("ls /data01/dataagent_be/logs/ 2>/dev/null");
        assert_eq!(a, "ls:/data01/dataagent_be/logs");
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn coarse_execute_command_signature_keeps_search_pattern_and_path() {
        let sig = coarse_execute_command_signature(
            "grep -rl \"24394294\" /data01/dataagent_be/logs/ 2>/dev/null | head -10",
        );
        assert_eq!(sig, "grep:/data01/dataagent_be/logs#24394294");
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
            let signal = supervisor.record_tool_signatures(&messages);
            if i < TOOL_LOOP_COARSE_WINDOW - 1 {
                assert!(matches!(signal, ToolLoopSignal::None));
            } else {
                assert!(matches!(signal, ToolLoopSignal::Coarse));
            }
        }
        assert!(supervisor.coarse_loop_note_injected);
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
            signals.push(supervisor.record_tool_signatures(&messages));
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
        assert!(text.contains("不同且明确的子问题"));
        assert!(text.contains("不一定是错误"));
    }

    #[test]
    fn tool_iteration_soft_limit_clamps_to_absolute_ceiling() {
        assert_eq!(tool_iteration_soft_limit(0), 1);
        assert_eq!(tool_iteration_soft_limit(1), 1);
        assert_eq!(tool_iteration_soft_limit(10), 5);
        assert_eq!(tool_iteration_soft_limit(256), TOOL_ITERATION_SOFT_LIMIT);
        assert_eq!(tool_iteration_soft_limit(2048), TOOL_ITERATION_SOFT_LIMIT);
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
    // 每轮开始清除上一轮的打断标记，确保它只反映「本轮」是否被 Ctrl+C 打断。
    app.last_turn_interrupted = false;
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
    // 同理保存 max_tokens 自适应覆盖：零输出截断时自动降 max_tokens 重试。
    // 降级是临时的：一旦有正常输出（正常完成或正常截断）就恢复原始值，
    // 因为原始值本身是合理的（首次请求能成功）。turn 末兜底恢复。
    let saved_max_tokens_override = app.cli.max_tokens_override;
    // 当前是否处于零输出降级状态。
    let mut mt_downgraded = false;
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
            // 用服务端返回的实际 prompt_tokens 校正后续请求的 max_tokens clamp。
            // 字符估算偏保守（高估），服务端的实际值更准确，能减少不必要的钳小。
            let usage_prompt = match &execution {
                IterationExecution::Truncated(sr) | IterationExecution::FinalResponse(sr) => {
                    Some(sr.usage_prompt_tokens)
                }
                IterationExecution::ToolCall(tce) => Some(tce.stream_result.usage_prompt_tokens),
                _ => None,
            };
            if let Some(pt) = usage_prompt.filter(|&v| v > 0) {
                app.last_known_prompt_tokens = Some(pt);
            }
            // 空响应重试计数：连续 >2 次空响应则放弃，避免浪费迭代预算
            if matches!(execution, IterationExecution::EmptyResponse) {
                consecutive_empty_responses += 1;
                if consecutive_empty_responses > 5 {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ✗ 连续 {} 次空响应，停止重试",
                        consecutive_empty_responses
                    );
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

                    // 零输出截断检测：completion=0 + finish_reason=length 说明服务端
                    // 拒绝了 max_tokens 值（典型：relay/兼容层对超大 max_tokens 返回
                    // 空响应而非报错）。此时降 reasoning_effort / 禁 thinking 都无济于事
                    // ——问题不在模型输出太多，而在 max_tokens 本身被服务端拒绝。
                    // 策略：将 max_tokens 减半后重试，直到服务端接受。
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
                            "  ⚠ 零输出截断（completion=0），max_tokens {} → {} 自动降级重试",
                            current_max,
                            halved
                        );
                        app.cli.max_tokens_override = Some(halved);
                        mt_downgraded = true;
                    } else if mt_downgraded {
                        // 正常截断（有输出但被打断）：服务端接受了当前 max_tokens，
                        // 恢复原始值给后续迭代更大输出预算。
                        app.cli.max_tokens_override = saved_max_tokens_override;
                        mt_downgraded = false;
                    }

                    // 该模型降 reasoning_effort 是否真能缩短思考链。
                    // enable_thinking 布尔开关方言（GLM 等）请求体里根本不带
                    // effort，降档是空操作——必须直接关 thinking 才能腾出输出预算。
                    // dialect 分派与请求层一致：用 request_model_name + endpoint。
                    let endpoint = crate::ai::models::endpoint_for_model(&next_model, "");
                    let adapter = crate::ai::models::model_adapter(&next_model);
                    let request_model = crate::ai::models::request_model_name(&next_model);
                    let effort_helps = crate::ai::provider::reasoning_effort_reduces_thinking_for(
                        adapter,
                        &request_model,
                        &endpoint,
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
                        // effort 阶梯走到第 3 档仍截断，说明仅靠降 effort 已不足以收敛，
                        // 叠加强制关闭 thinking 作为兜底，把整个输出预算让给可见内容。
                        if consecutive_truncations >= 3 {
                            app.cli.thinking_disabled_override = true;
                        }
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
                if has_visible_text && consecutive_truncations >= 16 && !stream_result.stream_error
                {
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
                // 非截断：恢复因零输出降级的 max_tokens。
                if mt_downgraded {
                    app.cli.max_tokens_override = saved_max_tokens_override;
                    mt_downgraded = false;
                }
            }
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
                    "possible low-yield repetition detected (same target, paging only): injecting converge hint",
                );
                inject_coarse_loop_note(&mut messages);
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
                force_final_response = true;
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
                    "tool-loop hard-stop: switching to no-tool handoff",
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
            supervisor.maybe_inject_task_anchor(&mut messages, &question, "iteration-soft-limit");
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
    app.cli.max_tokens_override = saved_max_tokens_override;

    // 老化未在本 turn 使用的 explicit-enabled tool。
    // 连续 N 个 turn 闲置就 demote，避免"启用一次永久焊接"。
    crate::ai::tools::enable_tools::age_unused_explicit_tools(tools_used_this_turn.iter());

    skill_turn.restore_agent_context(app);

    let loop_result = loop_result.map_err(|e: Box<dyn std::error::Error>| e.to_string());

    match loop_result {
        Ok(Some(outcome)) => {
            app.last_turn_had_tool_calls = false;
            Ok(outcome)
        }
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
