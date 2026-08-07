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
        completion_evidence_state, completion_tool_result_succeeded,
        handle_iteration_execution_for_model, tool_call_is_successful_mutation_candidate,
    },
    types::{IterationExecution, TurnLoopStep, TurnOutcome, TurnPreparation},
};

/// 工具调用循环检测窗口：
/// - soft: 连续 4 轮调用 (tool_name, normalized_args) 完全一致，先注入反思提示
/// - hard: 收到 soft 提示后仍连续 6 轮完全一致，直接强制收敛，不再继续工具循环
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

/// 工具轮次检查点的首个固定阈值。默认 turn 硬预算是 4096；24 / 48 / 96
/// 三档检查点只调度收敛、不禁用工具，累计轮次也不会因 mutation 清零。
const TOOL_ROUND_CHECKPOINT: usize = 24;
const TOOL_ROUND_CHECKPOINT_MULTIPLIERS: [usize; 3] = [1, 2, 4];

/// 连续「流读取中断型」截断（stream_error）的重试上限。超过即放弃本 turn，
/// 避免服务端持续断流时无限重试（尤其后台任务的 max_iterations = usize::MAX）。
const MAX_STREAM_ERROR_RETRIES: usize = 16;
/// 连续「模型输出过长 / 工具调用 JSON 半截」截断的重试上限。
/// stream_error 使用独立上限，不参与该计数。
const MAX_MODEL_TRUNCATION_RETRIES: usize = 3;

/// === 长循环感知的中段压缩 ===
/// 中段压缩的软阈值按模型 token 窗口换算（flagship 256K → ~135K 字符）。对
/// 「历史体积中等、但工具迭代轮次很多」的长循环 turn，历史峰值可能长期低于该
/// 阈值 → 压缩全程不触发 → 每轮把「截至当前的完整历史 + 全部 tool schema」重发
/// 一遍，累计发送量随迭代轮次 O(n²) 膨胀，几分钟内撞破 TPM 限流（真实案例：
/// 一个 provider 重构会话单 turn 56 轮迭代，历史峰值仅 ~120K < 135K 阈值，
/// turn 内累计发送 ~2.8M token 撞破 380K TPM 约 7 倍）。
///
/// 治理：一旦单 turn 工具迭代轮次达到该阈值，即认定进入「长循环」，把中段压缩的
/// 有效软阈值下调到 [`MID_TURN_COMPRESS_SOFT_FLOOR`]（36K），让内容级去重
/// （byte-identical 重读折叠）与旧结果裁剪尽早介入，遏制 O(n²) 累积。短 turn
/// （迭代轮次未达阈值）保持原窗口比例阈值，不影响正常单轮大任务的探索空间。
const LONG_LOOP_COMPRESS_ITERATION_THRESHOLD: usize = 12;

/// === Progress Budget（信息增益进展预算）===
/// 这是叠加在 exact / coarse 循环检测之上的第三层，用于治理「参数每轮都变、
/// 但整体不推进任务」的发散型 loop——前两层按「签名重复」判定，结构上抓不到
/// 每轮都在搜新符号 / 读新文件却始终零收敛的膨胀（真实案例：一个「删除方法」
/// 的变更请求连续 60+ 轮只读取/检索、零 apply_patch）。
///
/// 核心理念：不按「动作次数」计费，按「信息增益」这一**行为信号**计费——本轮
/// 触碰到新目标资源（成功读取 / 检索到新目标），或调用了变更类工具，即算推进；
/// 失败调用（无目标）与反复取证同一目标都不算。不再从用户问题文本去猜任务意图。
/// 早期探索几乎免费；越往后，「继续但没进展」越要付出显式理由。惩罚对象是
/// 「说不出理由的无进展重复」，而非探索本身。
///
/// 免费探索轮数：达到该轮次前，即使连续无进展也完全不打扰（删代码前先定位、
/// 陌生代码库先摸索都是正常的）。
const PROGRESS_FREE_EXPLORE_ROUNDS: usize = 20;
/// 已触碰大量不同目标仍未收口，说明可能从「补关键证据」滑向「不断扩分支」。
/// 此阈值只注入一次非阻断式广度检查，不把新目标判成无进展，避免压缩大型排查
/// 任务的正当探索空间。
const READ_ONLY_BREADTH_CHECK_TARGETS: usize = 32;
/// 宽限窗口：软提示后，若模型给出了「实质不同的理由」（新目标 / reasoning 指纹
/// 变化），则在该窗口内不升级，给它继续探索的空间。
const PROGRESS_GRACE_WINDOW: usize = 6;
/// 一次低进展 episode 被真实进展打断后，至少间隔这么多轮才允许再次注入 soft。
/// 避免复杂任务在「探索 → 小进展 → 再探索」的正常节奏中反复收到同一收敛提示。
const PROGRESS_EPISODE_COOLDOWN: usize = 16;
/// 从「软提示 / 记账」升级到「硬停收口」额外需要的连续无进展轮数。
const PROGRESS_NO_PROGRESS_HARD_MARGIN: usize = 16;
/// scoped instruction preflight 使用独立预算，不消耗正常工具迭代；上限防止模型
/// 通过不断切换新目录无限延长单 turn。
const MAX_SCOPED_PREFLIGHT_GRACE_ROUNDS: usize = 8;
/// 变更类工具：调用这些动作（或产出 final text）即视为本轮有实质动作、算进展。
const MUTATION_TOOL_NAMES: &[&str] = &[
    "apply_patch",
    "write_file",
    "plan",
    "task_spawn",
    "task_spawn_batch",
    "task_wait",
    "task_cancel",
    "task_status",
    "execute_command",
];

/// 首档检查点对小预算仍按一半缩放；大预算明确固定在 24 轮，后续档位为 48 / 96。
fn initial_tool_round_checkpoint(max_iterations: usize) -> usize {
    (max_iterations / 2).max(1).min(TOOL_ROUND_CHECKPOINT)
}

fn tool_round_checkpoint_threshold(max_iterations: usize, level: usize) -> Option<usize> {
    let multiplier = *TOOL_ROUND_CHECKPOINT_MULTIPLIERS.get(level)?;
    let threshold = initial_tool_round_checkpoint(max_iterations).checked_mul(multiplier)?;
    if level > 0 && threshold >= max_iterations {
        return None;
    }
    Some(threshold)
}

/// 提取最近一轮 assistant 消息中的 (tool_name, args_json) 签名集合。
/// 任何一个签名与窗口内某轮完全一致即认为有循环倾向。
fn extract_round_tool_signatures(messages: &[crate::ai::history::Message]) -> Option<Vec<String>> {
    extract_round_tool_signatures_inner(messages, false)
}

/// 提取「粗粒度」签名：剥离 offset/limit/page 等易变翻页参数后再归一化。
/// 用于抓字节精确检测漏掉的同文件翻页 / 仅微调分页参数的重复检索。
/// 对 `execute_command` 额外折叠 shell 中的低收益变体（如 `| head -20/-30`、
/// `2>/dev/null`、`ls -la/-lt` 的细微差异，以及 git log/show/diff 取证视角的
/// 轻微切换），让同目标资源的反复试探能命中。
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
    match tool_name {
        "execute_command" => normalize_coarse_execute_command_args(value),
        "task_wait" => normalize_coarse_task_wait_args(value),
        "task_status" => normalize_coarse_task_status_args(value),
        _ => {}
    }
}

fn normalize_coarse_execute_command_args(value: &mut serde_json::Value) {
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

fn normalize_coarse_task_wait_args(value: &mut serde_json::Value) {
    let Some(map) = value.as_object_mut() else {
        return;
    };
    let task_ids = map
        .get("task_ids")
        .and_then(|v| v.as_array())
        .map(|values| {
            let mut ids = values
                .iter()
                .filter_map(|v| v.as_str().map(str::trim))
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            ids.sort();
            ids.dedup();
            ids
        });
    map.clear();
    if let Some(ids) = task_ids {
        map.insert(
            "task_ids".to_string(),
            serde_json::Value::Array(
                ids.into_iter()
                    .map(serde_json::Value::String)
                    .collect::<Vec<_>>(),
            ),
        );
    }
}

fn normalize_coarse_task_status_args(value: &mut serde_json::Value) {
    if let Some(map) = value.as_object_mut() {
        // task_status 忽略参数；不同空壳参数不应逃过 coarse 循环检测。
        map.clear();
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
        "git" => Some(normalize_git_segment(&tokens)),
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

fn normalize_git_segment(tokens: &[String]) -> String {
    let Some(subcommand_idx) = find_git_subcommand_index(tokens) else {
        return "git".to_string();
    };

    let subcommand = tokens[subcommand_idx].to_ascii_lowercase();
    // 对「为什么有两个 commit / 这两个 commit 差什么 / 当前分支状态如何」这类
    // git 取证问题，模型常在 log/show/diff/status/reflog 之间来回切视角，命令
    // 字面不同但语义上仍在围绕同一份证据打转。coarse 模式将其折叠成同一簇。
    if matches!(
        subcommand.as_str(),
        "log" | "show" | "diff" | "diff-tree" | "reflog" | "status"
    ) {
        return "git:inspect".to_string();
    }

    let mut paths = Vec::new();
    let mut revs = Vec::new();
    let mut after_double_dash = false;
    let mut skip_next = false;
    for token in tokens.iter().skip(subcommand_idx + 1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if should_skip_shell_token(token) {
            continue;
        }
        if token == "--" {
            after_double_dash = true;
            continue;
        }
        if !after_double_dash && token.starts_with('-') {
            if git_option_takes_value(token) {
                skip_next = true;
            }
            continue;
        }
        if looks_like_path_token(token) {
            paths.push(normalize_path_like_token(token));
            continue;
        }
        if looks_like_git_revision_token(token) {
            revs.push(normalize_git_revision_token(token));
        }
    }
    paths.sort();
    paths.dedup();
    revs.sort();
    revs.dedup();
    if !paths.is_empty() && !revs.is_empty() {
        format!("git:{subcommand}:{}#{}", revs.join(","), paths.join(","))
    } else if !paths.is_empty() {
        format!("git:{subcommand}:{}", paths.join(","))
    } else if !revs.is_empty() {
        format!("git:{subcommand}:{}", revs.join(","))
    } else {
        format!("git:{subcommand}")
    }
}

fn find_git_subcommand_index(tokens: &[String]) -> Option<usize> {
    let mut idx = 1;
    while idx < tokens.len() {
        let token = &tokens[idx];
        if !token.starts_with('-') {
            return Some(idx);
        }
        if git_option_takes_value(token) {
            idx += 2;
        } else {
            idx += 1;
        }
    }
    None
}

fn git_option_takes_value(token: &str) -> bool {
    matches!(
        token,
        "-C" | "-c"
            | "--git-dir"
            | "--work-tree"
            | "--format"
            | "--pretty"
            | "--grep"
            | "--author"
            | "--committer"
            | "--since"
            | "--until"
    )
}

fn looks_like_git_revision_token(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    if token.contains("..") || token.contains("...") || token.contains("@{") {
        return true;
    }
    if matches!(
        token,
        "HEAD" | "FETCH_HEAD" | "ORIG_HEAD" | "MERGE_HEAD" | "CHERRY_PICK_HEAD"
    ) {
        return true;
    }
    let trimmed = token.trim_end_matches(['^', '~']);
    let hexish = trimmed.len() >= 7
        && trimmed
            .chars()
            .all(|ch| ch.is_ascii_hexdigit() || matches!(ch, '^' | '~' | ':'));
    if hexish {
        return true;
    }
    trimmed.starts_with("refs/")
}

fn normalize_git_revision_token(token: &str) -> String {
    let normalized = token.trim().trim_matches(',');
    if normalized.contains("..") || normalized.contains("...") {
        let sep = if normalized.contains("...") {
            "..."
        } else {
            ".."
        };
        let mut parts: Vec<String> = normalized
            .split(sep)
            .filter(|part| !part.is_empty())
            .map(normalize_git_revision_token)
            .collect();
        parts.sort();
        parts.dedup();
        return parts.join(sep);
    }
    if normalized.eq_ignore_ascii_case("head") {
        return "HEAD".to_string();
    }
    if normalized.starts_with("HEAD@{") {
        return "HEAD@{}".to_string();
    }
    let hex_prefix: String = normalized
        .chars()
        .take_while(|ch| ch.is_ascii_hexdigit())
        .take(12)
        .collect();
    if hex_prefix.len() >= 7 {
        return hex_prefix;
    }
    normalized.to_string()
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
    if tail.iter().all(|sigs| sigs == first) {
        return true;
    }

    // 除 A-A-A-A 外，模型还会以 A-B-A-B 或 A-B-C-A-B-C 的方式规避逐轮
    // 去重。只识别恰好填满当前窗口的短周期，避免把正常的长任务误判成循环。
    for period in 2..=3 {
        let cycle = &tail[..period];
        if cycle.iter().any(Vec::is_empty) {
            continue;
        }
        if window % period == 0 {
            // 窗口恰好被周期整除：要求窗口内是完整周期的重复。
            if tail.chunks_exact(period).all(|chunk| chunk == cycle) {
                return true;
            }
        } else {
            // 窗口不能整除周期（如 soft 窗口 4 vs 周期 3）：退化为
            // 「若干完整周期 + 一个周期前缀」也判为循环。这补上了 3 周期在
            // soft 检查里永远不触发（4 % 3 != 0 被跳过）、导致第 6 轮被无预警
            // 直接 hard-stop 的洞：A-B-C-A-B-C 会在第 4 轮（A-B-C-A 匹配
            // 周期 [A,B,C] 的前缀）先拿到 Soft 预警，维持 soft→hard 升级不变量。
            if tail.iter().zip(cycle.iter().cycle()).all(|(a, b)| a == b) {
                return true;
            }
        }
    }
    false
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

/// 目标级重复检测：窗口内每一轮都触碰了**同一个**目标资源即命中。
///
/// 这是对整轮签名比较的补位。`detect_tool_loop` 要求整轮签名集合相等（或短周期
/// 循环），模型只要在每轮里多穿插一个不同的陪衬工具（今天读 A+搜 X、明天读 A+搜 Y、
/// 后天读 A+列目录），整轮签名就各不相等而逃逸，但真正的低收益重复是「A 被反复读」。
/// 这里改为求窗口内各轮目标集合的**交集**：只要存在一个目标在每一轮都出现，就判定
/// 为该目标被反复取证。空轮（无目标）不参与，避免误判。
fn detect_target_repeat_loop(history: &[Vec<String>], window: usize) -> bool {
    if window < 2 || history.len() < window {
        return false;
    }
    let tail = &history[history.len() - window..];
    if tail.iter().any(Vec::is_empty) {
        return false;
    }
    let mut intersection: FxHashSet<&str> = tail[0].iter().map(String::as_str).collect();
    for round in &tail[1..] {
        let round_set: FxHashSet<&str> = round.iter().map(String::as_str).collect();
        intersection.retain(|target| round_set.contains(target));
        if intersection.is_empty() {
            return false;
        }
    }
    !intersection.is_empty()
}

fn is_direct_file_mutation_tool(name: &str) -> bool {
    matches!(name, "apply_patch" | "write_file")
}

/// 判断最近一轮 assistant 是否调用了变更类工具（apply_patch/write_file）。
///
/// `execute_command` 是双关工具：`git status`/`git log`/`ls` 等只读取证命令不改变
/// 世界，却曾被无差别计为 Mutation 进展，导致模型反复刷同一批 git 检查就能不断
/// 刷新 no-progress 预算、永不收敛。因此对 execute_command 额外判定：只有**非只读**
/// 命令才算 Mutation 动作。
///
/// `task_wait` / `task_status` 也是双关工具：只有真正交付了子任务结果时才算推进。
/// 空轮询、PARKED、BUDGET-ELAPSED、already-collected 提示和无任务状态都不算实质动作。
fn round_has_mutation(messages: &[crate::ai::history::Message]) -> bool {
    let Some(last_assistant) = messages.iter().rev().find(|m| m.role == "assistant") else {
        return false;
    };
    let Some(tool_calls) = last_assistant.tool_calls.as_ref() else {
        return false;
    };
    let tool_results_by_call_id: FxHashMap<&str, &str> = messages
        .iter()
        .filter(|message| message.role == "tool")
        .filter_map(|message| {
            let call_id = message.tool_call_id.as_deref()?;
            let text = message.content.as_str().unwrap_or_default();
            Some((call_id, text))
        })
        .collect();
    tool_calls.iter().any(|tc| {
        let name = tc.function.name.as_str();
        if !MUTATION_TOOL_NAMES.contains(&name) {
            return false;
        }
        match name {
            "execute_command" => {
                // 只读取证命令不算变更进展；解析失败或非只读命令保守计为 Mutation
                // （安全方向：避免把真实改动误判为无进展而过早收口）。
                serde_json::from_str::<serde_json::Value>(tc.function.arguments.as_str())
                    .ok()
                    .and_then(|args| {
                        args.get("command")
                            .and_then(|v| v.as_str())
                            .map(|cmd| !execute_command_is_read_only(cmd))
                    })
                    .unwrap_or(true)
            }
            "task_wait" | "task_status" => tool_results_by_call_id
                .get(tc.id.as_str())
                .is_some_and(|text| task_tool_result_delivered_task_output(text)),
            "write_file" | "apply_patch" => {
                // 直接文件变更工具只有**写入成功**才算推进。失败（沙箱越界、
                // 上下文不匹配、路径错误等）不改变世界，却曾无差别计为 Mutation，
                // 每次重试都会清零 no-progress 预算，使进展预算 loop guard 永远
                // 攒不满窗口——模型可对同一个被拒路径反复 write_file / apply_patch
                // 而不被收口（见 write blocked 循环）。结果缺失（None）保守计为
                // 推进，避免把真实改动误判为无进展而过早停。
                match tool_results_by_call_id.get(tc.id.as_str()) {
                    Some(text) => matches!(
                        classify_tool_result_progress(text),
                        ToolResultProgressStatus::Success
                    ),
                    None => true,
                }
            }
            _ => true,
        }
    })
}

fn task_tool_result_delivered_task_output(text: &str) -> bool {
    text.lines()
        .any(|line| line.trim_start().starts_with("[Task: "))
}

fn current_tool_round_messages(
    messages: &[crate::ai::history::Message],
) -> Vec<crate::ai::history::Message> {
    let Some(assistant_idx) = messages.iter().rposition(|message| {
        message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .is_some_and(|tool_calls| !tool_calls.is_empty())
    }) else {
        return Vec::new();
    };
    let Some(tool_calls) = messages[assistant_idx].tool_calls.as_ref() else {
        return Vec::new();
    };
    let tool_call_ids: FxHashSet<&str> = tool_calls.iter().map(|tc| tc.id.as_str()).collect();
    let mut out = vec![messages[assistant_idx].clone()];
    let mut idx = assistant_idx + 1;
    while idx < messages.len() && messages[idx].role == "tool" {
        match messages[idx].tool_call_id.as_deref() {
            Some(id) if tool_call_ids.contains(id) => out.push(messages[idx].clone()),
            _ => break,
        }
        idx += 1;
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolRoundCheckpointPhase {
    Explore,
    ImplementedNeedsVerification,
    VerifiedNeedsFinalization,
    RecoveringFromError,
}

impl ToolRoundCheckpointPhase {
    fn recent_progress(self) -> &'static str {
        match self {
            Self::Explore => "read-only",
            Self::ImplementedNeedsVerification => "mutation",
            Self::VerifiedNeedsFinalization => "verification-success",
            Self::RecoveringFromError => "verification-failure",
        }
    }

    fn action(self) -> &'static str {
        match self {
            Self::Explore => "choose-one-next-step",
            Self::ImplementedNeedsVerification => "verify-and-wrap-up",
            Self::VerifiedNeedsFinalization => "finalize",
            Self::RecoveringFromError => "fix-current-failure",
        }
    }

    fn guidance(self) -> &'static str {
        match self {
            Self::Explore => {
                "当前仍处于只读取证阶段：总结已确认事实与唯一缺口，只选择一个最有信息增益的下一步；优先精准检索或一次足够大的读取，停止扩展无关证据面。"
            }
            Self::ImplementedNeedsVerification => {
                "最近已成功修改状态：不要重新扩展探索或继续无关修改；运行最窄且能覆盖该变更的检查/测试，必要时查看 diff/status，然后立即收尾。"
            }
            Self::VerifiedNeedsFinalization => {
                "已观察到成功验证：除非存在一个明确且会改变结论的缺口，否则不要继续调用工具；直接总结改动、验证结果与剩余风险并完成答复。"
            }
            Self::RecoveringFromError => {
                "最近的修改或验证失败：只诊断当前失败并进行一次针对性修复/重试，不要扩展到无关问题；若仍受阻，明确报告阻塞点并收尾。"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolRoundCheckpointLevel {
    Review,
    Restrict,
    Finalize,
}

impl ToolRoundCheckpointLevel {
    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Review,
            1 => Self::Restrict,
            _ => Self::Finalize,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Review => "review",
            Self::Restrict => "restrict",
            Self::Finalize => "finalize",
        }
    }

    fn guidance(self) -> &'static str {
        match self {
            Self::Review => "这是非错误、非工具失败的一次性阶段检查点。",
            Self::Restrict => {
                "这是第二级检查点：先列出剩余必要工作，只允许完成关键修复与最小验证，不再扩大任务范围。"
            }
            Self::Finalize => {
                "这是第三级检查点：基于现有证据收尾；除非当前验证失败且一次针对性修复可直接解决，否则不要继续调用工具。"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ToolRoundCheckpoint {
    level: ToolRoundCheckpointLevel,
    phase: ToolRoundCheckpointPhase,
    threshold: usize,
}

fn checkpoint_tool_call_effects(tool_call: &crate::ai::types::ToolCall) -> (bool, bool) {
    if tool_call.function.name != "execute_command" {
        return (tool_call_is_successful_mutation_candidate(tool_call), false);
    }
    let effects = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
        .ok()
        .map(|args| super::iteration::execute_command_segment_effects_for_args(&args))
        .unwrap_or_default();
    (
        effects.iter().any(|effect| effect.project_mutation),
        effects
            .iter()
            .any(|effect| effect.scope_review || effect.behavior_check),
    )
}

fn tool_round_checkpoint_phase(
    current_round: &[crate::ai::history::Message],
    turn_messages: &[crate::ai::history::Message],
) -> ToolRoundCheckpointPhase {
    let evidence = completion_evidence_state(turn_messages);
    let results: FxHashMap<&str, bool> = current_round
        .iter()
        .filter(|message| message.role == "tool")
        .filter_map(|message| {
            Some((
                message.tool_call_id.as_deref()?,
                completion_tool_result_succeeded(&message.content),
            ))
        })
        .collect();
    let mut relevant_failure = false;

    for tool_call in current_round
        .iter()
        .find(|message| message.role == "assistant")
        .and_then(|message| message.tool_calls.as_ref())
        .into_iter()
        .flatten()
    {
        let (mutation, verification) = checkpoint_tool_call_effects(tool_call);
        let Some(succeeded) = results.get(tool_call.id.as_str()).copied() else {
            continue;
        };
        if mutation || verification {
            // checkpoint 只关心最后一个相关动作的状态；早期失败若已被后续
            // mutation + verification 修复，不应继续强制进入 Recovering。
            relevant_failure = !succeeded;
        }
    }

    if relevant_failure {
        ToolRoundCheckpointPhase::RecoveringFromError
    } else if evidence.successful_mutation && evidence.successful_post_mutation_verification {
        ToolRoundCheckpointPhase::VerifiedNeedsFinalization
    } else if evidence.successful_mutation {
        ToolRoundCheckpointPhase::ImplementedNeedsVerification
    } else {
        ToolRoundCheckpointPhase::Explore
    }
}

/// 判断一条 shell 命令是否为纯只读取证（不改变世界）。解析不确定时返回 false
/// （保守：宁可把只读误判为可能变更，也不把真实变更误判为只读）。
/// 判定单个 shell 段是否只读。程序取段首 token 的 basename，`git` 再取真正的
/// 子命令（跳过 `-C <path>` / `-c k=v` 等全局选项）。
fn shell_segment_is_read_only(segment: &str) -> bool {
    let mut tokens = segment.split_whitespace();
    let Some(program) = tokens.next() else {
        return false;
    };
    let program = program.rsplit('/').next().unwrap_or(program);
    if program == "git" {
        let mut skip_next = false;
        for token in tokens {
            if skip_next {
                skip_next = false;
                continue;
            }
            if token == "-C" || token == "-c" {
                skip_next = true;
                continue;
            }
            if token.starts_with('-') {
                continue;
            }
            return GIT_READ_ONLY_SUBCOMMANDS.contains(&token);
        }
        return false;
    }
    READ_ONLY_COMMAND_PROGRAMS.contains(&program)
}

/// `cd` / `export` 只改变工作目录或环境，不写文件系统，本身无副作用。作为前导段
/// 跳过，避免 `cd X && git status` 这类「游走 + 检查」命令被误判为变更。
fn shell_segment_is_nav_or_env(segment: &str) -> bool {
    let mut tokens = segment.split_whitespace();
    let Some(program) = tokens.next() else {
        return false;
    };
    matches!(
        program.rsplit('/').next().unwrap_or(program),
        "cd" | "export"
    )
}

fn execute_command_is_read_only(command: &str) -> bool {
    // 跳过前导 `cd`/`export` 段后，要求**所有**实质段都只读才算只读；任一实质段
    // 可能变更即视为变更（安全方向：避免把真实改动误判为无进展而过早收口）。
    let mut saw_substantive = false;
    for segment in split_shell_segments_for_coarse(command) {
        if shell_segment_is_nav_or_env(&segment) {
            continue;
        }
        saw_substantive = true;
        if !shell_segment_is_read_only(&segment) {
            return false;
        }
    }
    saw_substantive
}

/// 明确只读的独立程序。刻意排除 `sed`/`awk`（可 `-i` 原地改写）与任何可能带副作用
/// 的工具，保证「误判方向」永远偏向「可能变更」。
const READ_ONLY_COMMAND_PROGRAMS: &[&str] = &[
    "ls", "cat", "grep", "rg", "find", "fd", "head", "tail", "wc", "pwd", "echo", "stat", "tree",
    "file", "which", "type", "du", "df", "ps", "date", "env", "printenv", "sort", "uniq", "cut",
    "nl", "xxd", "od", "basename", "dirname", "realpath", "readlink", "less", "more", "diff",
    "cmp", "column",
];

/// 明确只读的 git 子命令。刻意排除 `branch`/`tag`/`remote`/`config` 等可带副作用的
/// 子命令（裸列出形式虽只读，但带参可变更；无法区分时按可能变更处理）。
const GIT_READ_ONLY_SUBCOMMANDS: &[&str] = &[
    "status",
    "log",
    "diff",
    "show",
    "reflog",
    "blame",
    "describe",
    "rev-parse",
    "rev-list",
    "ls-files",
    "ls-tree",
    "cat-file",
    "shortlog",
    "whatchanged",
    "name-rev",
    "merge-base",
    "for-each-ref",
    "symbolic-ref",
    "count-objects",
    "diff-tree",
    "diff-index",
    "grep",
    "annotate",
];

#[derive(Clone, Debug, PartialEq, Eq)]
enum ToolResultProgressStatus {
    Success,
    Failure,
    DedupOnly,
    BlockedOutsideWorkspace(String),
}

fn classify_tool_result_progress(text: &str) -> ToolResultProgressStatus {
    let text = text.trim_start();
    if let Some(path) = blocked_outside_workspace_path(text) {
        return ToolResultProgressStatus::BlockedOutsideWorkspace(path);
    }
    if let Some(path) = write_blocked_outside_root_path(text) {
        return ToolResultProgressStatus::BlockedOutsideWorkspace(path);
    }
    if is_dedup_only_tool_result(text) {
        return ToolResultProgressStatus::DedupOnly;
    }
    if text.starts_with("Error:") || text.starts_with("Exit code:") {
        return ToolResultProgressStatus::Failure;
    }
    ToolResultProgressStatus::Success
}

fn is_dedup_only_tool_result(text: &str) -> bool {
    let text = text.trim_start();
    text.starts_with("[deduped:") || text.starts_with("[overlap dedup:")
}

/// 从 `write_file` / `apply_patch` 的沙箱越界拒绝消息中解析被拒的目标路径。
///
/// 消息形如 `... Write blocked: path '/abs/path' is outside the allowed write
/// directory ...`。与 `blocked_outside_workspace_path`（execute_command 的命令级
/// 拒绝）平行：把「反复写同一个被拒路径」归一成稳定目标，让 target-repeat loop
/// guard 能在少数几轮内抓到，而不是任由模型对同一路径反复重试。
fn write_blocked_outside_root_path(text: &str) -> Option<String> {
    let marker = "Write blocked: path '";
    let rest = text.split_once(marker)?.1;
    let path = rest.split_once('\'').map(|(path, _)| path)?.trim();
    (!path.is_empty()).then(|| normalize_path_like_token(path))
}

fn blocked_outside_workspace_path(text: &str) -> Option<String> {
    let marker = "Command blocked: command references path ";
    let rest = text.split_once(marker)?.1;
    if let Some((_, after_resolves)) = rest.split_once(" (resolves to ") {
        let resolved = after_resolves
            .split_once(") which is outside")
            .map(|(path, _)| path)
            .or_else(|| after_resolves.split_once(')').map(|(path, _)| path))?
            .trim();
        if !resolved.is_empty() {
            return Some(normalize_path_like_token(resolved));
        }
    }

    let original = rest
        .split_once(" which is outside")
        .map(|(path, _)| path)
        .unwrap_or(rest)
        .trim();
    (!original.is_empty()).then(|| normalize_path_like_token(original))
}

/// 提取最近一轮触碰的「目标资源」集合：文件路径 / 检索 pattern / 命令 coarse
/// target。普通失败请求（尤其是拼错路径）不能被算作信息增益，否则模型可不断生成
/// 新的无效参数来逃避收敛；但沙箱外路径拒绝会归一成稳定目标，专门用于识别
/// 反复读取同一个禁止路径的循环。
fn extract_round_targets(messages: &[crate::ai::history::Message]) -> Vec<String> {
    extract_round_targets_inner(messages, true)
}

fn extract_round_probe_targets(messages: &[crate::ai::history::Message]) -> Vec<String> {
    extract_round_targets_inner(messages, false)
}

fn extract_round_targets_inner(
    messages: &[crate::ai::history::Message],
    include_direct_file_mutations: bool,
) -> Vec<String> {
    use serde_json::Value;
    let Some(last_assistant) = messages.iter().rev().find(|m| m.role == "assistant") else {
        return Vec::new();
    };
    let Some(tool_calls) = last_assistant.tool_calls.as_ref() else {
        return Vec::new();
    };
    let results_by_call_id: FxHashMap<&str, ToolResultProgressStatus> = messages
        .iter()
        .filter(|message| message.role == "tool")
        .filter_map(|message| {
            let call_id = message.tool_call_id.as_deref()?;
            let text = message.content.as_str().unwrap_or_default();
            Some((call_id, classify_tool_result_progress(text)))
        })
        .collect();

    let mut targets = Vec::new();
    for tc in tool_calls.iter() {
        // 写被拒（沙箱越界）的直接文件变更工具：即使在排除变更工具的 probe 通道里，
        // 也要放行成一个稳定目标。否则「反复写同一个被拒路径」既不算进展、又不进入
        // target 历史，任何 loop guard 都抓不到（见 write blocked 循环）。归一路径让
        // 同一被拒目标跨轮稳定命中；成功写入仍按下方正常目标提取处理。
        if is_direct_file_mutation_tool(&tc.function.name) {
            if let Some(ToolResultProgressStatus::BlockedOutsideWorkspace(path)) =
                results_by_call_id.get(tc.id.as_str())
            {
                targets.push(format!("{}:blocked-outside-root:{path}", tc.function.name));
                continue;
            }
        }
        if !include_direct_file_mutations && is_direct_file_mutation_tool(&tc.function.name) {
            continue;
        }
        match results_by_call_id.get(tc.id.as_str()) {
            Some(ToolResultProgressStatus::Success) | None => {}
            Some(ToolResultProgressStatus::BlockedOutsideWorkspace(path))
                if tc.function.name == "execute_command" =>
            {
                targets.push(format!("execute_command:blocked-outside-workspace:{path}"));
                continue;
            }
            Some(
                ToolResultProgressStatus::BlockedOutsideWorkspace(_)
                | ToolResultProgressStatus::Failure
                | ToolResultProgressStatus::DedupOnly,
            ) => continue,
        }
        let Ok(args) = serde_json::from_str::<Value>(tc.function.arguments.as_str()) else {
            continue;
        };
        let Some(map) = args.as_object() else {
            continue;
        };
        // url/selector：浏览器工具（navigate/get_text/click/type_text 等）读写的是
        // 「当前页面」这一外部状态。它们不在 MUTATION_TOOL_NAMES 里，参数也不带
        // path/query，若不纳入目标提取，则 navigate 新 URL、读取新 selector 这类真实
        // 推进会被 assess_progress 一律判成「无进展」，导致正常的多步浏览 turn 在
        // ~41 轮被 LowProgressHard 误停。把 url/selector 视作新目标即可正确记进展。
        for key in ["path", "file_path", "pattern", "query", "url", "selector"] {
            if let Some(s) = map.get(key).and_then(|v| v.as_str()) {
                let target = if matches!(key, "path" | "file_path") {
                    normalize_path_like_token(s)
                } else {
                    s.trim().to_string()
                };
                targets.push(format!("{}:{key}:{target}", tc.function.name));
            }
        }
        if let Some(cmd) = map.get("command").and_then(|v| v.as_str()) {
            // 用 coarse 签名（而非命令前两 token）作为目标标识：`git log`/`git show`/
            // `git diff` 等围绕同一份证据来回切视角的只读取证会归并到同一个
            // `git:inspect` 目标，不再被逐条误判为「新目标 = 新进展」。否则模型只要
            // 每轮换一个 git 子命令，assess_progress 就持续判定有进展并清空循环历史，
            // 使 coarse-hard 永远攒不满窗口——这正是多样化只读命令逃逸 loop guard 的
            // 根因。coarse 归一对无法解析的命令会回退到命令原文，语义与旧行为一致。
            let target = coarse_execute_command_signature(cmd);
            targets.push(format!("{}:{}", tc.function.name, target));
        }
    }
    targets
}

/// 稳定的 64-bit 内容指纹（用于判定 reasoning / 结果是否实质变化）。
fn content_fingerprint(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    s.trim().hash(&mut hasher);
    hasher.finish()
}

/// 提取最近一轮 assistant 的 reasoning 指纹（若有）。软提示后 reasoning 指纹
/// 变化视为「给出了新理由」，触发 grace 宽限。
fn extract_round_reasoning_fingerprint(messages: &[crate::ai::history::Message]) -> Option<u64> {
    let last_assistant = messages.iter().rev().find(|m| m.role == "assistant")?;
    let reasoning = last_assistant.reasoning_content.as_ref()?;
    if reasoning.trim().is_empty() {
        return None;
    }
    Some(content_fingerprint(reasoning))
}

/// 提取本轮成功只读工具返回的内容指纹。Progress Budget 不能只看「是否换了目标」：
/// 同一文件的新分页、同一页面的新区域也可能带来真实新证据。结果内容发生变化时记为
/// 信息增益；出现新证据时会重启 exact/coarse 连续重复窗口，只有结果也不再变化时才
/// 升级。
fn extract_round_evidence_fingerprints(messages: &[crate::ai::history::Message]) -> Vec<u64> {
    use serde_json::Value;

    let Some(last_assistant) = messages.iter().rev().find(|m| m.role == "assistant") else {
        return Vec::new();
    };
    let Some(tool_calls) = last_assistant.tool_calls.as_ref() else {
        return Vec::new();
    };
    let calls_by_id: FxHashMap<&str, (&str, &str)> = tool_calls
        .iter()
        .map(|tc| {
            (
                tc.id.as_str(),
                (tc.function.name.as_str(), tc.function.arguments.as_str()),
            )
        })
        .collect();

    let mut fingerprints = Vec::new();
    for message in messages.iter().filter(|message| message.role == "tool") {
        let Some(call_id) = message.tool_call_id.as_deref() else {
            continue;
        };
        let Some((tool_name, arguments)) = calls_by_id.get(call_id).copied() else {
            continue;
        };
        let text = message.content.as_str().unwrap_or_default().trim();
        if text.is_empty()
            || classify_tool_result_progress(text) != ToolResultProgressStatus::Success
        {
            continue;
        }

        // 变更/调度工具由 round_has_mutation 单独判定。execute_command 只有明确只读时
        // 才把返回内容当作证据，避免一次成功写操作被重复记账。
        if MUTATION_TOOL_NAMES.contains(&tool_name) {
            if tool_name != "execute_command" {
                continue;
            }
            let Ok(args) = serde_json::from_str::<Value>(arguments) else {
                continue;
            };
            let Some(command) = args.get("command").and_then(|value| value.as_str()) else {
                continue;
            };
            if !execute_command_is_read_only(command) {
                continue;
            }
        }

        fingerprints.push(content_fingerprint(&format!("{tool_name}\0{text}")));
    }
    fingerprints.sort_unstable();
    fingerprints.dedup();
    fingerprints
}

/// 稳定的「无进展」软阈值。免费探索区内返回 usize::MAX（永不触发）。
///
/// 旧逻辑会在长任务后段从 5 轮递减到 3 / 2 轮，导致任务越复杂、越接近收尾，
/// 正常的同目标验证越容易被误判。真实 exact/coarse 重复已有独立 detector，因此这里
/// 保持稳定阈值，不再仅因 turn 变长而提高提示频率。
fn no_progress_soft_threshold(iteration: usize, free_explore_rounds: usize) -> usize {
    if iteration <= free_explore_rounds {
        return usize::MAX;
    }
    5
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
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
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
    next_tool_round_checkpoint_level: usize,
    iteration_limit_note_injected: bool,
    scoped_preflight_grace_rounds: usize,
    task_anchor_injected: bool,
    last_compress_iteration: usize,
    last_compress_after_chars: usize,
    /// 等待与下次 pre-request LLM 压缩结果合并输出的 mid-turn 状态。
    pending_compression_report: CompressionReport,
    tool_signature_history: Vec<Vec<String>>,
    tool_signature_history_coarse: Vec<Vec<String>>,
    /// 每轮触碰的「coarse 目标资源」集合历史（同 read_file 文件 /
    /// 同 execute_command coarse 命令，忽略翻页参数）。用于抓「整轮签名不
    /// 相等、但同一目标被混在不同工具批次里反复取证」的循环——纯整轮签名比较对
    /// 这类混合批次无能为力（每轮多一个陪衬工具即逃逸）。
    tool_target_history: Vec<Vec<String>>,
    target_repeat_note_injected: bool,
    progress: ProgressLedger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolLoopSignal {
    None,
    /// 近似低收益重复：同一工具反复命中同一目标资源（忽略翻页参数）。温和提示一次。
    Coarse,
    /// 混合工具轮里同一目标资源被反复取证：整轮签名各不相等（每轮穿插不同陪衬
    /// 工具）逃过了 exact/coarse 整轮比较，但某个 read_file 文件
    /// 在窗口每一轮都出现。温和提示一次。
    TargetRepeat,
    /// `execute_command` 在同一 coarse 目标上长时间空转，直接强制收敛。
    CoarseHard,
    Soft,
    Hard,
    /// Progress Budget 第一级：连续多轮无信息增益（既无新目标也无实质动作），
    /// 注入反思式软提示，不阻断工具。
    LowProgressSoft,
    /// Progress Budget 第二级：软提示后仍无进展，要求写下轻量决策账本
    /// （已确认事实 / 待解决问题 / 候选与已排除分支），仍不硬阻断。
    LowProgressLedger,
    /// Progress Budget 第三级：软提示 + 记账后仍连续无进展，切换无工具收口模式。
    LowProgressHard,
    /// 已覆盖大量不同目标，提醒先汇总当前证据和唯一关键缺口。
    ReadOnlyBreadth,
}

/// Progress Budget 的运行时状态。挂在 `TurnSupervisor` 上，按「信息增益」而非
/// 动作次数计费；只惩罚「说不出理由的无进展重复」。进展是**行为信号**：本轮
/// 触碰到新目标资源，或调用了变更类工具（`round_has_mutation`），即算推进；不再
/// 从用户问题文本去猜任务意图。
#[derive(Default)]
struct ProgressLedger {
    /// 累计触碰过的目标资源（「新目标 = 信息增益」判定）。
    seen_targets: FxHashSet<String>,
    /// 累计见过的成功只读工具结果。新内容即使来自同一目标，也属于新证据。
    seen_evidence_fingerprints: FxHashSet<u64>,
    /// 连续无进展轮数。任意一轮判定为 Progress 即清零。
    consecutive_no_progress: usize,
    /// 上一轮 reasoning 指纹（软提示后指纹变化 → 视为给出新理由 → grace 宽限）。
    last_reasoning_fp: Option<u64>,
    /// grace 宽限截止迭代号：在此之前不升级，给模型继续探索的空间。
    grace_until_iteration: usize,
    /// reasoning 变化每个 turn 最多换取一次 grace，避免靠逐轮改写理由无限续期。
    grace_consumed: bool,
    soft_injected: bool,
    ledger_injected: bool,
    hard_injected: bool,
    read_only_breadth_injected: bool,
    /// 新的 low-progress episode 最早允许注入 soft 的迭代号。实质进展会重置当前
    /// episode，但保留该 cooldown，防止复杂任务被同一提示反复打断。
    next_episode_iteration: usize,
}

impl ProgressLedger {
    /// 重置升级阶梯：清空无进展计数与 soft/ledger/hard/grace 等一次性状态，
    /// 让计费从零重新开始。两类场景共用：
    /// 1. 截断重试（mark_truncation_skip）：截断清空历史后，重复读取是预期行为，
    ///    与 exact/coarse 检测的 mark_truncation_skip 语义保持一致，避免截断恢复后的
    ///    新循环跳过 soft 提示直接到 hard-stop。
    /// 2. 实质进展（assess_progress 的 made_progress 分支）：软提示后模型给出真正推进
    ///    任务的动作，应视为「这一轮提醒生效了」，给予完整的新预算而非继续累加，否则
    ///    模型在长任务中只要早期发散过一次，后续每次收敛提醒都会更快滑向硬停。
    fn reset_escalation(&mut self) {
        self.consecutive_no_progress = 0;
        self.soft_injected = false;
        self.ledger_injected = false;
        self.hard_injected = false;
        self.grace_until_iteration = 0;
        self.grace_consumed = false;
    }

    /// 截断是外部约束，不应继承上一 episode 的提示冷却。
    fn reset_after_truncation(&mut self) {
        self.reset_escalation();
        self.next_episode_iteration = 0;
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

    /// 本轮实际生效的中段压缩软阈值。
    ///
    /// 长循环（工具迭代轮次 >= [`LONG_LOOP_COMPRESS_ITERATION_THRESHOLD`]）时把
    /// 阈值下调到 [`MID_TURN_COMPRESS_SOFT_FLOOR`]，让内容级去重与旧结果裁剪尽早
    /// 介入，遏制 O(n²) 累积重发；短 turn 保持按窗口换算的基准阈值，不影响正常
    /// 单轮大任务。门控与实际 [`mid_turn_compress`](crate::ai::history::mid_turn_compress)
    /// 调用必须共用本方法返回值——后者内部有 `before <= soft_threshold` 的 no-op
    /// 早退，若两处阈值不一致会「门开了却压不动」。
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

    /// 任务出现实质进展后，丢弃此前无效循环的样本并恢复 soft → hard 升级阶梯。
    /// 否则模型已经响应 soft 提示而改做有效动作时，后续一次新的重复会跳过 soft，
    /// 直接沿用旧标志进入 hard-stop。
    fn reset_tool_loop_escalation(&mut self) {
        self.tool_signature_history.clear();
        self.tool_signature_history_coarse.clear();
        self.tool_target_history.clear();
        self.hard_loop_stop_injected = false;
        self.loop_breaker_injected = false;
        self.coarse_loop_note_injected = false;
        self.target_repeat_note_injected = false;
    }

    /// 截断重试时重置工具循环检测状态：截断是外部约束（输出上限 / 模型可用性波动），
    /// 重试时重复调用相同工具属于预期行为，不应计入循环检测窗口。
    /// 截断本身已有独立的 `consecutive_truncations` 上限兜底，不需要循环检测再叠加。
    ///
    /// 清空历史 + 跳过当前迭代的签名记录，使截断重试不被误判为循环。
    /// 重置所有一次性标志：截断清空历史后，soft/coarse/hard 的完整升级阶梯
    /// 应从零重新开始，否则截断恢复后形成的新循环会跳过 soft 提示直接到 hard-stop。
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
        // 截断重试跳过：清空历史后不记录本轮签名，避免截断重试
        // 被误判为工具循环。`skip_tool_signature_rounds` 由
        // `mark_truncation_skip()` 递增，每跳过一次递减。
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
        // 目标级历史：与 coarse 签名平行维护，供混合工具轮的目标交集检测使用。
        // 与 exact/coarse 一样受 TOOL_SIGNATURE_HISTORY_LIMIT 约束。
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
            // Soft 提示已明确要求停止重复调用。清空此前用于触发 soft 的样本，
            // 让模型有完整的 hard window 来响应提示，而不是只再重复两轮就被强制
            // 收口（旧逻辑中 soft=4、hard=6，实际恢复窗口只有两轮）。
            self.tool_signature_history.clear();
            // 设计洞补位：soft 只清了 exact 样本，但 coarse/target 历史未清，且它们
            // 的一次性门（coarse_loop_note_injected / target_repeat_note_injected）
            // 被下方 `!loop_breaker_injected` 永久挡死——模型 soft 后换一批参数继续
            // 翻同一目标时，coarse/target 再也无法触发，直到迭代上限才被收口。这里把
            // 两个门重新武装并清空对应历史，让「soft 后换姿势」的翻页 / 混合轮循环仍
            // 能被后续 coarse/target 捕获（提示仍是一次性、soft 优先级，不额外加压）。
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
        // 字节精确的 soft/hard 均未命中时，再看粗粒度：同一目标资源反复翻页/微调
        // 检索参数的膨胀会在这里被抓到。仅提示一次，且让位于精确检测。
        if !self.coarse_loop_note_injected
            && detect_tool_loop(&self.tool_signature_history_coarse, TOOL_LOOP_COARSE_WINDOW)
        {
            self.coarse_loop_note_injected = true;
            return ToolLoopSignal::Coarse;
        }
        // 整轮签名（exact/coarse）都要求整轮集合相等，抓不到「同一目标混在不同工具
        // 批次里反复取证」的混合轮循环。这里用目标交集补位：窗口内每轮都触碰同一
        // 目标即命中。让位于上面所有整轮检测，且与 coarse 一样只提示一次。
        if !self.target_repeat_note_injected
            && !self.coarse_loop_note_injected
            && detect_target_repeat_loop(&self.tool_target_history, TOOL_LOOP_COARSE_WINDOW)
        {
            self.target_repeat_note_injected = true;
            return ToolLoopSignal::TargetRepeat;
        }
        // exact/coarse 均未命中「签名重复」型循环时，交给 Progress Budget 补位：
        // 抓「参数每轮都变、但整体不推进任务」的发散型 loop。
        self.assess_progress(messages, progress_messages, free_explore_rounds)
    }

    /// Progress Budget 判定：按「信息增益」而非动作次数计费。只在 exact/coarse
    /// 签名检测未命中时补位调用。进展是纯**行为信号**，不再从问题文本猜意图：
    ///
    /// - 本轮触碰到新目标资源（`extract_round_targets` 首次出现）→ 信息增益，算进展；
    /// - 成功只读工具返回了此前未见的新内容 → 新证据，算进展；
    /// - 或本轮调用了变更类工具（`round_has_mutation`）→ 实质动作，算进展。
    ///
    /// 免费探索区（iteration <= free_explore_rounds）内完全不计费；退出后按稳定
    /// 阈值升级：软提示 → 固定响应窗口 → 记账 → 硬停。soft episode 之间还有 turn 内
    /// cooldown，避免复杂任务在正常的探索/推进节奏中反复收到同一提示。
    fn assess_progress(
        &mut self,
        messages: &[crate::ai::history::Message],
        progress_messages: &[crate::ai::history::Message],
        free_explore_rounds: usize,
    ) -> ToolLoopSignal {
        // 三类进展信号分开保留：ReadOnlyBreadth 只由新目标触发；调用模式历史只在
        // 新目标/变更动作后清空。同一目标返回新内容时只重置 Progress Budget，保留
        // exact/coarse detector 对重复翻页模式的一次性提醒能力。
        let round_had_mutation = round_has_mutation(progress_messages);
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
            // 任意真实进展都结束当前 low-progress episode；next_episode_iteration 保留，
            // 因而短暂推进后不会立刻再次注入 soft。seen targets/evidence 也跨轮累积。
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
            // 只有**结构性进展**（新目标 / 变更动作）且此前已提示过 loop 时，才清空
            // exact/coarse/target 重试窗口。新的只读结果内容已计入上面的 made_progress
            // （Progress Budget 不会因高效翻页误升级），但**绝不**清空签名/目标历史：
            // 否则同文件反复翻页会因每页内容不同而永远填不满 coarse 窗口，绕过循环刹车
            // ——这正是「进展哈希须忽略 offset/limit 以防预算逃逸」不变量要防的情况。
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

        // 免费探索区：探索完全免费，不计费也不升级（删代码前先定位、陌生代码库
        // 先摸索都属正常）。仅更新 reasoning 指纹基线。
        if self.iteration <= free_explore_rounds {
            self.progress.last_reasoning_fp = reasoning_fp;
            return ToolLoopSignal::None;
        }

        self.progress.consecutive_no_progress += 1;

        // 每次 soft 后都给固定响应窗口。旧逻辑只有暴露 reasoning_content 且指纹变化
        // 的模型才有 grace；不输出 reasoning 的模型会在下一轮立刻收到 ledger。基础
        // 窗口结束后，新理由仍可额外延长一次，但不得滚动续期。
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

        // 升级阶梯严格按 软提示 → 记账 → 硬停 推进，每级一次性。硬停额外要求
        // 连续无进展达到 soft_threshold + margin，避免越过软层直接收口。
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

    /// 分级工具轮次检查点：保留累计轮次，在 24 / 48 / 96（小预算按首档缩放）
    /// 根据当前工作阶段注入不同调度提示。
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

/// 工具循环检测命中后，向 messages 注入一条 internal_note 让 agent 自我反思
/// （而非直接 force_final，给 agent 一个跳出循环的机会）。
fn inject_loop_breaker_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[loop-detected] 你最近 4 轮都在用相同参数调用相同工具；此前的工具结果仍在上下文中，重复调用不会产生新信息。\n\
        不要再次调用这一组相同参数。先基于已有证据决定下一步：\n\
        (a) 信息已足够时，直接执行实质动作或回答用户；\n\
        (b) 信息不足时，只能选择一个不同且具体的动作（例如读取未覆盖的行范围、搜索新的符号/目标，或修改文件）；\n\
        (c) 确实无法继续时，说明缺少的唯一关键信息及原因。";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

fn inject_hard_loop_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[loop-hard-stop] 你在收到重复调用提示后，仍连续 6 轮用相同参数调用相同工具，判定为无效循环。\n\
        从现在起进入无工具收口模式：不要再发起任何工具调用；\n\
        请基于已有信息给出阶段总结与当前结论；若任务仍未完成，明确说明缺口、剩余工作与建议的下一步。";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

const TOOL_STOP_REASON_PREFIX: &str = "[runtime-tool-stop]";

/// 将进入无工具收口模式的首个根因仅写入当前 request context。
pub(super) fn record_force_final_reason(
    messages: &mut Vec<crate::ai::history::Message>,
    reason: &str,
    iteration: usize,
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

    let event = Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!(
            "{TOOL_STOP_REASON_PREFIX} reason={reason}, iteration={iteration}, action=no_tool_handoff"
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    // 运行时停止原因只属于本次请求投影；若写入 canonical turn_messages，下一轮会把
    // 过期的 no-tool-handoff 控制状态重新提升为 system 并永久重放。
    messages.push(event);
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
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 混合工具轮的目标级重复提示：同一目标被穿插在不同工具批次里反复取证。
/// 与 coarse 提示同级（温和、不阻断），但措辞强调「换工具查同一个东西」这一
/// 特定反模式，引导模型复用已读结果而非再换个工具重查一遍。
fn inject_target_repeat_loop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-yield-repetition] 你最近多轮一直在对同一个目标（同一文件 / 同一检索目标）反复取证，\n\
        只是每轮换了不同的工具或搭配了不同的陪衬调用来绕过重复——但你并没有得到新信息。\n\
        请停下来做一件事：直接复用你已经读到 / 搜到的关于该目标的内容，不要再换一个工具去查同一个东西。\n\
        然后二选一：(a) 若信息已足够，立即执行下一步实质动作或直接作答；\n\
        (b) 若确需继续，请明确写下你还缺哪一条『关于该目标的新信息』、以及为什么换工具能拿到它。";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
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
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget 第一级（软反思）：连续多轮无可测信息增益。
/// 收敛类提示在“工具仍可继续”的阶段（soft / breadth / ledger）要求模型写下的
/// 账本 / 归纳属于**内部自省**，必须落在隐藏的 `<meta:self_note>…</meta:self_note>`
/// 通道里：流层 [`push_text_with_hidden_meta`](crate::ai::stream) 会把它从可见输出
/// 剥离、持久化成 internal_note，下一轮模型仍能读到。若不约束落点，模型会把这段
/// 中途反思写进面向用户的正文，被立即流式呈现成「预结论」，并与真正的最终答复
/// 重复（本次事故的直接成因）。硬停 / 迭代上限等 force-final 提示不套用本约束——
/// 那时正文就是最终答复。
const SELF_NOTE_REFLECTION_CHANNEL_HINT: &str = "\n\
    重要（落点约束）：上面要求你写下的账本 / 归纳属于内部自省，必须整段写在 \
    `<meta:self_note>` 与 `</meta:self_note>` 之间；这段内容不会展示给用户、但会保留在你的后续上下文里。\n\
    面向用户的正文本轮应保持为空或仅承载「继续执行的下一步」，只有在你确实可以收尾时才写真正的最终结论。";

/// 反思式提示，不阻断工具——给模型解释「为什么还要继续同方向」和继续探索的权利。
fn inject_low_progress_soft_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[low-progress-review] 运行时最近没有观察到新的目标、成功状态变更或新的工具结果内容。\n\
        这是启发式检查，不代表同一目标上的工作一定无效，也不要仅因本提示放弃必要步骤。\n\
        继续调用工具前，请确认：下一次调用会补哪条尚缺证据，以及什么结果会结束该分支。\n\
        若现有证据已足够，就完成最窄验证并作答；若不足，可按上述明确缺口继续。\
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

/// ReadOnly 广度检查：新目标仍算信息增益；这里只在目标面过宽时提醒先归纳，
/// 不阻断工具，避免把大型排查任务误判为低进展。
fn inject_read_only_breadth_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[read-only-breadth-check] 你已在只读分析中覆盖了大量不同目标资源，\n\
        这可能是必要的广泛排查，也可能已经从『补关键证据』滑向『不断扩分支』。\n\
        工具仍然可用；但在继续前，请先用不超过 6 行写下：\n\
        1) 已确认事实（最多 3 条）；2) 当前结论或最可能解释；\n\
        3) 仍缺的唯一关键证据；4) 下一步唯一工具动作。\n\
        如果已经足够回答，请直接给出结论，不要为了再次确认而继续扩展搜索面。{SELF_NOTE_REFLECTION_CHANNEL_HINT}"
    );
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget 第二级（记账）：软提示后仍无进展，要求写下轻量决策账本，
/// 让模型显式说明继续探索的依据。仍不硬阻断工具。
fn inject_progress_ledger_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[low-progress-ledger] 在上一阶段检查后的响应窗口内，运行时仍未观察到新的目标、\n\
        成功状态变更或新的工具结果内容。若要继续，请先用不超过 6 行写出决策账本：\n\
        1) 已确认事实（bullet，最多 3 条）\n\
        2) 仍待解决的唯一关键问题\n\
        3) 候选分支 A / B 及你现在选哪个、为什么\n\
        4) 基于所选分支的下一步唯一动作\n\
        若缺口明确，可继续执行该动作；若无法表述缺口，就基于现有证据收尾。\
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

/// Progress Budget 第三级（硬停）：软提示 + 记账后仍连续无进展，切换无工具收口。
fn inject_low_progress_hard_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-progress-hard-stop] 经软提示、响应窗口与记账后，运行时仍未观察到可测进展。\n\
        为避免继续消耗预算，现在进入无工具收口模式：不要再发起任何工具调用；\n\
        请基于已收集到的信息给出阶段结论：已确认了什么、还差什么、\n\
        以及若要完成任务建议的下一步（若是变更类任务，直接说明应改哪些文件、怎么改）。";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 分级、阶段感知的工具轮次检查点；它调度下一步，但不把刚完成的工具标成失败。
fn inject_tool_round_checkpoint_note(
    messages: &mut Vec<crate::ai::history::Message>,
    iteration: usize,
    checkpoint: ToolRoundCheckpoint,
) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = format!(
        "[tool-round-checkpoint] level={} phase={} round={iteration} threshold={}。\n\
        {}\n\
        {}\n\
        checkpoint 不改变委派标准：不要因上下文或迭代压力转交当前分支，只委派原本就独立、有界且值得委派的子任务。",
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
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 同步等待快到硬超时时，请子代理停止扩展新分支，优先交付可验证结论。
fn inject_subagent_pre_timeout_wrap_up_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;

    let note = "[subagent-pre-timeout-wrap-up] 当前同步子任务的前台等待时间即将耗尽。\n\
        现在进入无工具收口模式：不要再发起新的工具调用或扩展新的审计分支。\n\
        请立即基于已收集的证据输出最终答复：先列出已验证的结论；\n\
        将尚未验证的风险单独标注，绝不可猜测。";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subagent_pre_timeout_wrap_up_note_requires_immediate_final_answer() {
        let mut messages = Vec::new();
        inject_subagent_pre_timeout_wrap_up_note(&mut messages);

        let note = messages
            .last()
            .and_then(|message| message.content.as_str())
            .expect("wrap-up note should be textual");
        assert!(note.contains("无工具收口模式"));
        assert!(note.contains("最终答复"));
        assert!(!note.contains("`/audit`"));
    }

    #[test]
    fn force_final_reason_is_request_only_and_deduplicated() {
        let mut messages = Vec::new();

        record_force_final_reason(&mut messages, "iteration_limit", 24);
        record_force_final_reason(&mut messages, "tool_loop_exact", 25);

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
            messages.push(pb_read_msg("src/other.rs", &format!("retry-{i}")));
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
                    arguments: "{\"path\":\"src/main.rs\"}".to_string(),
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
        assert!(text.contains("不同且明确的子问题"));
        assert!(text.contains("不一定是错误"));
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
                            arguments: "{\"path\":\"src/bin/ai/mod.rs\"}".to_string(),
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
        assert!(text.contains("同一个目标"));
        assert!(text.contains("换一个工具去查同一个东西"));
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
    // `/audit` 是用户直接请求的同步子代理调用。必须在父 DRIVER_CTX 已建立、
    // 子 agent 尚未进入递归 turn 前处理，才能复用 task 的隔离与证据生命周期。
    if crate::ai::driver::runtime_ctx::current_subagent_depth() == 0 {
        if let Some(command) = crate::ai::driver::commands::audit::parse_audit_command(&question) {
            return Ok(execute_audit_command(command, should_quit));
        }
    }
    // 把 (session_id, turn_id) 注入 task_local，让下游工具调用与反馈
    // 写入路径能拿到正确身份。turn_id 由 session SQLite 原子分配，包含普通、
    // resume 和 internal turn，跨重启/多进程也不会重复。
    let session_id = app.session_id.clone();
    let turn_index = history::reserve_turn_index(&app.session_history_file)?;
    let turn_id = turn_index;
    // 仅前台主 turn 抬起「turn 活动」标志：子 agent（sync / background）持有私有
    // 信号标志，且都通过 SUBAGENT_RESULT_SLOT 作用域执行，据此排除。该标志让
    // prepare / 思考 / 阶段切换 / mid-turn 压缩等 streaming=false 的空窗里的
    // Ctrl+C 也走「取消本轮」而非「退出会话」。guard 随本 future drop 自动落下。
    let _foreground_turn_guard = (!crate::ai::driver::runtime_ctx::has_subagent_result_slot())
        .then(crate::ai::driver::signal::ForegroundTurnGuard::enter);
    crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .scope((session_id, turn_id), async {
            // enable_tools 的 per-turn 状态必须跟随整个 future，而不能只依赖
            // run_turn_body 的 happy-path 尾部清理；abort / early return 也会 Drop。
            let _enable_turn_guard = crate::ai::tools::enable_tools::EnableTurnStateGuard::enter();
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
}

fn execute_audit_command(
    command: crate::ai::driver::commands::audit::AuditCommand,
    should_quit: bool,
) -> TurnOutcome {
    match command {
        crate::ai::driver::commands::audit::AuditCommand::Usage => {
            println!("Usage: /audit <instruction>");
        }
        crate::ai::driver::commands::audit::AuditCommand::Run(instruction) => {
            // 默认只继承 cwd/skills，避免把无关的父对话和 memory 带入审计任务。
            // 但子代理完全看不到父对话，必须显式告知 main agent 当前改了什么：
            // 经常多个需求并行改动，子代理只有看到当前工作区 diff 才能判断哪些属于本次审计。
            let prompt = crate::ai::driver::commands::audit::build_audit_prompt(&instruction);
            let description = format!("/audit {instruction}");
            let args = serde_json::json!({
                "description": description,
                "prompt": prompt,
                "agent": "audit",
            });
            match crate::ai::driver::tools::execute_direct_subagent_task(
                "slash-audit",
                &args,
                crate::ai::driver::commands::audit::AUDIT_SUBAGENT_HARD_TIMEOUT,
                Some(crate::ai::driver::commands::audit::AUDIT_SUBAGENT_WRAP_UP_LEAD_TIME),
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
    // 每轮开始清除上一轮的打断标记，确保它只反映「本轮」是否被 Ctrl+C 打断。
    app.last_turn_interrupted = false;
    // 请求用户输入是工具层到 driver 的 turn 级侧信道；先清除遗留状态，避免异常退出
    // 的上一轮把续接误带入当前 turn。
    crate::ai::tools::skill_tools::clear_pending_user_input_request();
    // reasoning items 侧信道是 turn 级内存态：每轮开始清空，避免上一轮的
    // encrypted reasoning 泄漏到本轮请求（call_id 也不会再匹配）。
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

    let mut supervisor = TurnSupervisor::default();
    let mut force_final_response = false;
    let mut pre_timeout_wrap_up_requested = false;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut final_response_model = None::<String>;
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
    // VL 图片摘要（429 TPM 缓解）状态：pending_digest_source 保存上一轮 tool-call
    // 响应的未截断原文（含 reasoning），作为“搭车”解析摘要的来源；
    // image_digest_resolved 一旦置位，表示图片已被摘要替换、或两条路径都拿不到
    // 摘要而按用户决定保留原图——两种情况都不再每轮重试，避免反复发兜底请求。
    let mut pending_digest_source: Option<String> = None;
    let mut image_digest_resolved = false;
    // preflight 拒绝的 mutation 目标必须在下一次 prompt 重建时优先获得 scoped
    // 指令预算；不能再与本轮所有历史读取目标竞争，否则会稳定重复暂停。
    let mut pending_scoped_project_targets = Vec::new();
    let loop_result = 'turn: loop {
        let iteration = supervisor.next_iteration();
        let effective_max_iterations = supervisor.effective_max_iterations(max_iterations);
        // 从第二轮起，把请求投影里用户消息内联的图片换成上一轮产出的文字摘要，
        // 避免在工具循环里反复重放 base64 触发 Doubao/Ark 侧 429 TPM 限流。
        // 优先搭车解析上一轮响应；拿不到再发一次禁用工具的一次性 VL 请求兜底；
        // 两者都失败则保留原图。只改请求投影 messages，canonical turn_messages 不动。
        // 放在 mcp 锁块之前：兜底请求含 .await，绝不能跨 std::Mutex 持锁。
        if !image_digest_resolved && iteration >= 2 {
            // 只处理「当前 turn」的用户图片：用 rposition 取最后一条含图 user 消息。
            // messages 头部可能载入历史里更早 turn 的图片（若未被压缩外溢），但我们
            // 采集/生成的摘要只描述当前 turn 的图片（指令注入在当前 user 消息、兜底用
            // app.attached_image_files=当前 turn），拿它去替换旧图会张冠李戴。旧 turn
            // 图片本就由压缩的 spill 外溢到文件，当前 turn 图片才是被 spill 豁免、每轮
            // 重放的那张，正是要换掉的目标。
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
                }
                image_digest_resolved = true;
            } else {
                image_digest_resolved = true;
            }
        }
        {
            let mc = mcp_client.lock().unwrap();
            let required_project_targets = std::mem::take(&mut pending_scoped_project_targets);
            refresh_skill_turn_for_iteration(
                app,
                &mc,
                skill_manifests,
                &question,
                iteration,
                &mut skill_turn,
                &required_project_targets,
                &mut messages,
            );
        }
        if crate::ai::driver::runtime_ctx::take_subagent_wrap_up_request() {
            pre_timeout_wrap_up_requested = true;
            record_force_final_reason(&mut messages, "subagent_pre_timeout_wrap_up", iteration);
            force_final_response = true;
            inject_subagent_pre_timeout_wrap_up_note(&mut messages);
        }
        let active_skill_name = skill_turn.matched_skill_name().map(str::to_string);
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
        // 预超时收口信号在模型请求中途触发：放弃当前请求，立即进入强制收口迭代，
        // 而不是等当前（可能很长的）迭代自然结束。消费信号，避免下一轮迭代顶部重复注入。
        if matches!(&execution, IterationExecution::WrapUpFinal) {
            let _ = crate::ai::driver::runtime_ctx::take_subagent_wrap_up_request();
            pre_timeout_wrap_up_requested = true;
            record_force_final_reason(&mut messages, "subagent_pre_timeout_wrap_up", iteration);
            force_final_response = true;
            inject_subagent_pre_timeout_wrap_up_note(&mut messages);
            continue 'turn;
        }
        let was_final_response = matches!(&execution, IterationExecution::FinalResponse(_));
        let had_tool_call_execution = matches!(&execution, IterationExecution::ToolCall(_));
        // 搭车采集：图片摘要尚未解决时，缓存本轮 tool-call 响应的未截断原文
        // （assistant_text + reasoning_text）。下一轮循环开头据此尝试 parse 出摘要，
        // 命中即可省掉一次兜底请求。必须用 stream_result 原文——写回 messages 的
        // assistant narration 会被截断到 800 字符，可能截掉摘要尾部哨兵。
        if !image_digest_resolved && let IterationExecution::ToolCall(tce) = &execution {
            let sr = &tce.stream_result;
            pending_digest_source = Some(format!("{}\n{}", sr.assistant_text, sr.reasoning_text));
        }
        {
            let mc = mcp_client.lock().unwrap().routing_snapshot();
            // 用服务端返回的实际 prompt_tokens 校正后续请求的 max_tokens clamp。
            // 字符估算偏保守（高估），服务端的实际值更准确，能减少不必要的钳小。
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
                        "  ⚠ 响应流读取中断（连续第 {consecutive_stream_errors} 次）；正在自动重试，连续超过 {MAX_STREAM_ERROR_RETRIES} 次时停止…"
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

                    // 该模型降 reasoning_effort 是否真能缩短思考链。模型级 wire
                    // 声明优先于 provider 默认：例如 DashScope DeepSeek 虽使用
                    // enable_thinking 开关，但推理强度仍由顶层 reasoning_effort 控制。
                    // 未声明有效 effort wire 的布尔开关方言才需要直接关闭 thinking。
                    let effort_helps =
                        crate::ai::models::reasoning_effort_reduces_thinking(&next_model);

                    if effort_helps {
                        // 渐进式 reasoning effort 降档，把输出预算从 reasoning 让给实际内容。
                        // resolve_reasoning_effort 每次迭代实时读该字段，改了立即对下一次生效。
                        //
                        // 1 次截断 → Low（减半推理开销）
                        // 2 次截断 → None（显式最低档，真正的推理下限）
                        // 3 次以上 → 完全禁用 reasoning（不下发 effort 字段）+ 关 thinking
                        //
                        // 用显式 `None`（下发 `reasoning_effort: "none"`）而非省略字段：
                        // 省略字段会让服务端回退到自身默认档（gpt-5.x 默认 medium），
                        // 反而把推理预算调高，破坏阶梯单调性。`None` 是各 gpt-5.x 版本
                        // 都支持的真正下限，取代已被 gpt-5.6 系列移除、会触发 400 的
                        // `Minimal`。
                        app.cli.reasoning_effort_override = Some(match consecutive_truncations {
                            1 => Some(crate::ai::provider::ReasoningEffort::Low),
                            2 => Some(crate::ai::provider::ReasoningEffort::None),
                            _ => None, // Some(None) = 禁用 reasoning，不下发 effort 字段
                        });
                        // effort 阶梯走到第 3 档仍截断，说明仅靠降 effort 已不足以收敛，
                        // 叠加强制关闭 thinking 作为兜底，把整个输出预算让给可见内容。
                        if consecutive_truncations >= MAX_MODEL_TRUNCATION_RETRIES {
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
                if has_visible_text
                    && consecutive_truncations >= MAX_MODEL_TRUNCATION_RETRIES
                    && !stream_result.stream_error
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
                if consecutive_truncations >= MAX_MODEL_TRUNCATION_RETRIES
                    && !stream_result.stream_error
                {
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
                skill_turn.matched_skill_name().is_none(),
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
                        pending_scoped_project_targets = targets;
                        // 该轮没有执行 mutation，也不应计入 progress/loop 统计。
                        continue 'turn;
                    }
                    // 独立 preflight 预算耗尽后保持安全拒绝并收口，避免通过不断
                    // 切换目录无限扩张迭代预算。
                    record_force_final_reason(
                        &mut messages,
                        "scoped_preflight_budget_exhausted",
                        iteration,
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
                TurnLoopStep::Break => {
                    if was_final_response {
                        final_response_model.clone_from(&response_model);
                    }
                    break 'turn Ok(None);
                }
                TurnLoopStep::Return(outcome) => break 'turn Ok(Some(outcome)),
            }
        }
        // ↓↓↓ Continue 分支的后续处理（已离开 mc 锁，可以安全 await）↓↓↓
        let progress_messages = if had_tool_call_execution {
            current_tool_round_messages(&messages)
        } else {
            Vec::new()
        };

        // === Mid-turn 渐进式压缩 ===
        // 每轮 tool 执行完毕后检查 messages 总字符；超过软阈值时
        // 复用跨 turn 压缩管线，避免长链工具调用把上下文撑爆。
        // 节流：①冷却 N 轮 ②增量小于 DELTA 时跳过，避免 no-op 反复压缩。
        // 阈值按 history_max_chars 动态计算（floor 兜底），避免用户调整
        // history_max_chars 后 mid-turn 阈值依旧死锁在 36K/80K。
        let history_max_chars = app.config.history_max_chars;
        let mid_turn_soft_base = mid_turn_compress_soft_threshold(&next_model, history_max_chars);
        // 长循环时把软阈值下调到 SOFT_FLOOR，遏制 O(n²) 累积重发（详见
        // [`LONG_LOOP_COMPRESS_ITERATION_THRESHOLD`]）。门控与下面的实际
        // mid_turn_compress 调用共用同一 `mid_turn_soft`，避免「门开了却 no-op」。
        let mid_turn_soft = supervisor.effective_mid_turn_soft_threshold(mid_turn_soft_base);
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
            let mut compression_report = CompressionReport::default();
            if after < before {
                compression_report.record("mid-turn", before, after);
            }
            // 硬阈值：无损 + 弱损管线之后仍超额，调用 LLM 摘要兜底，
            // 把早期对话压成单条 internal_note，并将各压缩阶段合并为一条 status line。
            if after > mid_turn_hard
                && should_try_llm_summary(&app.session_id, after, mid_turn_hard)
            {
                let drained: Vec<crate::ai::history::Message> = std::mem::take(&mut messages);
                let (after_msgs, llm_before, llm_after, was_effective) =
                    crate::ai::history::mid_turn_llm_summarize(
                        app,
                        drained,
                        MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
                        MID_TURN_LLM_SUMMARY_MAX_CHARS,
                        history_max_chars,
                    )
                    .await;
                messages = after_msgs;
                record_llm_summary_attempt_chars(&app.session_id, llm_after);
                compression_report.record_llm_summary_attempt(
                    format!("mid-turn LLM (limit {mid_turn_hard})"),
                    llm_before,
                    llm_after,
                    was_effective,
                );
                compression_report.emit();
            } else {
                supervisor.pending_compression_report = compression_report;
            }
        }

        // === 工具循环检测 ===
        // 若 execute 层已决定进入最终响应，就只保留 iteration-limit 调度，不再叠加
        // loop/progress/checkpoint prompt。反过来，health hard-stop 也不应冒充 iteration limit。
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
                record_force_final_reason(&mut messages, "low_yield_repetition", iteration);
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
                record_force_final_reason(&mut messages, "tool_loop", iteration);
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
                inject_read_only_breadth_note(&mut messages);
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
                record_force_final_reason(&mut messages, "progress_no_progress", iteration);
                force_final_response = true;
            }
        }

        // === 分级、阶段感知的工具轮次检查点 ===
        // 使用 pre-compression 的当前工具轮判断阶段，累计轮次保持不变；检查点只调度
        // 下一步，不把刚完成的 mutation 误报为失败。同一轮已有更具体的 health signal
        // 时跳过 checkpoint，避免多个收敛 prompt 叠加。
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

        // === Iteration limit 自反思 ===
        // execute.rs 在 iteration >= max_iterations 时会把
        // force_final_response 置 true。此时除原有的 "Tool limit reached"
        // system prompt 外，再额外补一条更具体的反思 prompt
        // （只注入一次，避免重复刷屏）。
        supervisor.maybe_inject_iteration_limit_note(
            &mut messages,
            effective_max_iterations,
            force_final_before_health && !pre_timeout_wrap_up_requested,
        );
        if force_final_before_health && !pre_timeout_wrap_up_requested {
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

    let loop_result = loop_result.map_err(|e: Box<dyn std::error::Error>| e.to_string());

    // 只有 active skill 明确通过工具请求用户输入、且本轮正常结束时才建立一次性续接。
    // 这避免了按自然语言问号/关键词猜测，外部 skill 也无需修改自身 manifest。
    let requested_user_input = crate::ai::tools::skill_tools::take_pending_user_input_request();
    if requested_user_input && loop_result.is_ok() && !app.last_turn_interrupted {
        if let Some(skill_name) = skill_turn.matched_skill_name().map(str::to_owned) {
            app.pending_skill_continuation =
                Some(crate::ai::types::PendingSkillContinuation { skill_name });
        }
    }

    let final_skill_name = skill_turn.matched_skill_name().map(str::to_owned);
    skill_turn.restore_agent_context(app);

    match loop_result {
        Ok(Some(outcome)) => {
            app.last_turn_had_tool_calls = false;
            Ok(outcome)
        }
        Ok(_) => {
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
