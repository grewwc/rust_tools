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

use crate::ai::{mcp::SharedMcpClient, types::App};

use super::{
    MID_TURN_COMPRESS_COOLDOWN_ITERATIONS, MID_TURN_COMPRESS_DELTA_THRESHOLD,
    MID_TURN_COMPRESS_SOFT_FLOOR, MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
    MID_TURN_LLM_SUMMARY_MAX_CHARS,
    finalize::{
        finalize_turn, maybe_generate_session_title, should_generate_session_title_in_background,
    },
    iteration::{execute_turn_iteration, refresh_skill_turn_for_iteration},
    mid_turn_compress_hard_threshold, mid_turn_compress_soft_threshold,
    persistence::persist_pending_turn_messages,
    record_llm_summary_attempt_chars, should_try_llm_summary,
    prepare::prepare_turn,
    tool_result::handle_iteration_execution,
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

/// 中段断路器绝对上限：单轮工具迭代达到该值即注入一次收敛提示，远早于
/// max_iterations 硬上限，用于治理「合法但低效」的工具刷屏。
/// 默认 turn 预算是 1024；软提示应足够早，给模型留出整理证据并主动收口的空间，
/// 但它只提醒、不禁用工具，不压缩复杂任务的硬预算。
const TOOL_ITERATION_SOFT_LIMIT: usize = 24;

/// 连续「流读取中断型」截断（stream_error）的重试上限。超过即放弃本 turn，
/// 避免服务端持续断流时无限重试（尤其后台任务的 max_iterations = usize::MAX）。
const MAX_STREAM_ERROR_RETRIES: usize = 16;

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
/// 从「软提示 / 记账」升级到「硬停收口」额外需要的连续无进展轮数。
const PROGRESS_NO_PROGRESS_HARD_MARGIN: usize = 16;
/// 变更类工具：调用这些动作（或产出 final text）即视为本轮有实质动作、算进展。
const MUTATION_TOOL_NAMES: &[&str] = &[
    "apply_patch",
    "write_file",
    "delete_path",
    "plan",
    "task_spawn",
    "task_wait",
    "task_cancel",
    "task_status",
    "execute_command",
];

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
        if window % period != 0 {
            continue;
        }
        let cycle = &tail[..period];
        if cycle.iter().any(Vec::is_empty) {
            continue;
        }
        if tail.chunks_exact(period).all(|chunk| chunk == cycle) {
            return true;
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
    matches!(name, "apply_patch" | "write_file" | "delete_path")
}

/// 判断最近一轮 assistant 是否调用了变更类工具（apply_patch/write_file/delete_path）。
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

/// 判断一条 shell 命令是否为纯只读取证（不改变世界）。解析不确定时返回 false
/// （保守：宁可把只读误判为可能变更，也不把真实变更误判为只读）。
fn execute_command_is_read_only(command: &str) -> bool {
    let Some(first_segment) = split_shell_segments_for_coarse(command).into_iter().next() else {
        return false;
    };
    let mut tokens = first_segment.split_whitespace();
    let Some(program) = tokens.next() else {
        return false;
    };
    let program = program.rsplit('/').next().unwrap_or(program);
    if program == "git" {
        // 跳过 `-C <path>` / `-c k=v` 等全局选项，取真正的子命令。
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
        if !include_direct_file_mutations && is_direct_file_mutation_tool(&tc.function.name) {
            continue;
        }
        match results_by_call_id.get(tc.id.as_str()) {
            Some(ToolResultProgressStatus::Success) | None => {}
            Some(ToolResultProgressStatus::BlockedOutsideWorkspace(path))
                if tc.function.name == "execute_command" =>
            {
                targets.push(format!(
                    "execute_command:blocked-outside-workspace:{path}"
                ));
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
        for key in ["path", "file_path", "pattern", "query"] {
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

/// 递减的「无进展」软阈值：越往后越严。免费探索区内返回 usize::MAX（永不触发）。
fn no_progress_soft_threshold(iteration: usize, free_explore_rounds: usize) -> usize {
    if iteration <= free_explore_rounds {
        return usize::MAX;
    }
    let over = iteration - free_explore_rounds;
    match over {
        0..=20 => 5,
        21..=40 => 3,
        _ => 2,
    }
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
    iteration_soft_limit_note_injected: bool,
    iteration_limit_note_injected: bool,
    task_anchor_injected: bool,
    last_compress_iteration: usize,
    last_compress_after_chars: usize,
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

#[derive(Debug)]
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
        self.progress.reset_escalation();
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
        // 整轮签名（exact/coarse）都要求整轮集合相等，抓不到「同一目标混在不同工具
        // 批次里反复取证」的混合轮循环。这里用目标交集补位：窗口内每轮都触碰同一
        // 目标即命中。让位于上面所有整轮检测，且与 coarse 一样只提示一次。
        if !self.target_repeat_note_injected
            && !self.coarse_loop_note_injected
            && !self.loop_breaker_injected
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
    /// - 或本轮调用了变更类工具（`round_has_mutation`）→ 实质动作，算进展。
    ///
    /// 免费探索区（iteration <= free_explore_rounds）内完全不计费；退出后按递减
    /// 阈值升级：软提示 → 记账 → 硬停。软提示后若模型给出「实质不同的理由」
    /// （reasoning 指纹变化）则进入 grace 宽限，不升级。
    fn assess_progress(
        &mut self,
        messages: &[crate::ai::history::Message],
        progress_messages: &[crate::ai::history::Message],
        free_explore_rounds: usize,
    ) -> ToolLoopSignal {
        // 本轮是否推进了任务：触碰新目标（信息增益）或调用变更类工具（实质动作）。
        // 两类信号要分开保留：ReadOnlyBreadth 只能由“继续扩展只读证据面”触发；
        // 一旦本轮已有 apply_patch/write_file/delete_path 等 mutation，就不应再给模型
        // 注入“先收敛证据”的 prompt。
        let round_had_mutation = round_has_mutation(progress_messages);
        let mut made_progress = round_had_mutation;
        let mut added_new_target = false;
        for t in extract_round_targets(progress_messages) {
            if self.progress.seen_targets.insert(t) {
                added_new_target = true;
                made_progress = true;
            }
        }

        let reasoning_fp = extract_round_reasoning_fingerprint(progress_messages)
            .or_else(|| extract_round_reasoning_fingerprint(messages));
        if made_progress {
            // 实质进展：清零无进展计数，并重置已注入的升级阶梯标志（soft / grace /
            // hard / grace_until），让模型在「被提醒收敛 -> 真正推进」后获得完整的新
            // 预算。否则 soft 注入后即使模型真的推进了任务，下一轮 consecutive 再次达
            // 阈值也会因 soft_injected 仍为 true 直接跳到 ledger/hard，长任务被误杀。
            // seen_targets / last_reasoning_fp 不动，保持跨轮累积（last_reasoning_fp
            // 随后用本轮指纹覆写，作为后续 grace 比较的新基线）。
            if self.progress.soft_injected
                || self.progress.ledger_injected
                || self.progress.hard_injected
                || self.progress.grace_consumed
                || self.progress.grace_until_iteration > 0
            {
                self.progress.reset_escalation();
            }
            if self.hard_loop_stop_injected
                || self.loop_breaker_injected
                || self.coarse_loop_note_injected
                || self.target_repeat_note_injected
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

        // grace 出口：软提示后，若模型首次给出「实质不同的理由」（reasoning 指纹
        // 变化），给予一次固定宽限。后续 reasoning 变化不得滚动续期。
        let reasoning_changed =
            reasoning_fp.is_some() && reasoning_fp != self.progress.last_reasoning_fp;
        self.progress.last_reasoning_fp = reasoning_fp;
        if self.progress.soft_injected && reasoning_changed && !self.progress.grace_consumed {
            self.progress.grace_until_iteration = self.iteration + PROGRESS_GRACE_WINDOW;
            self.progress.grace_consumed = true;
        }
        if self.iteration < self.progress.grace_until_iteration {
            return ToolLoopSignal::None;
        }

        let soft_threshold = no_progress_soft_threshold(self.iteration, free_explore_rounds);
        if self.progress.consecutive_no_progress < soft_threshold {
            return ToolLoopSignal::None;
        }

        // 升级阶梯严格按 软提示 → 记账 → 硬停 推进，每级一次性。硬停额外要求
        // 连续无进展达到 soft_threshold + margin，避免越过软层直接收口。
        if !self.progress.soft_injected {
            self.progress.soft_injected = true;
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

/// Progress Budget 第一级（软反思）：连续多轮无信息增益（无新目标也无实质动作）。
/// 反思式提示，不阻断工具——给模型解释「为什么还要继续同方向」和继续探索的权利。
fn inject_low_progress_soft_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-progress] 你已连续多轮调用工具，但任务没有可见的推进：\n\
        - 若这是「修改/删除/新增代码」类任务，你一直在读取/检索却还没提交任何 apply_patch/write_file/delete_path；\n\
        - 若是探索类任务，最近几轮没有触及新的目标资源，也没有排除掉候选分支。\n\
        请先停下来回答自己：上一步得到了什么『新信息』？如果说不出，就不要再沿同一方向重复。\n\
        然后二选一：(a) 若信息已足够，立即执行下一步实质动作（提交修改 / 给出结论）；\n\
        (b) 若确需继续探索，请显式写下你正在权衡的 A/B 候选分支及当前倾向——这不是惩罚，\n\
        而是帮你把『无意识的打转』变成『有方向的探索』。";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
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
    let note = "[read-only-breadth-check] 你已在只读分析中覆盖了大量不同目标资源，\n\
        这可能是必要的广泛排查，也可能已经从『补关键证据』滑向『不断扩分支』。\n\
        工具仍然可用；但在继续前，请先用不超过 6 行写下：\n\
        1) 已确认事实（最多 3 条）；2) 当前结论或最可能解释；\n\
        3) 仍缺的唯一关键证据；4) 下一步唯一工具动作。\n\
        如果已经足够回答，请直接给出结论，不要为了再次确认而继续扩展搜索面。";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget 第二级（记账）：软提示后仍无进展，要求写下轻量决策账本，
/// 逼模型意识到自己在摇摆。仍不硬阻断工具。
fn inject_progress_ledger_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-progress-ledger] 软提示后你仍在原地打转。现在请在继续任何工具调用之前，\n\
        先用不超过 6 行写出一份决策账本，强制自己收敛：\n\
        1) 已确认事实（bullet，最多 3 条）\n\
        2) 仍待解决的唯一关键问题\n\
        3) 候选分支 A / B 及你现在选哪个、为什么\n\
        4) 基于所选分支的下一步唯一动作\n\
        写完后，直接执行该动作；不要再做与该动作无关的探索性读取/检索。";
    messages.push(Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Progress Budget 第三级（硬停）：软提示 + 记账后仍连续无进展，切换无工具收口。
fn inject_low_progress_hard_stop_note(messages: &mut Vec<crate::ai::history::Message>) {
    use crate::ai::history::Message;
    use serde_json::Value;
    let note = "[low-progress-hard-stop] 经软提示与记账后你仍未取得任何可见进展，判定为无效循环。\n\
        从现在起进入无工具收口模式：不要再发起任何工具调用；\n\
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
    fn detect_tool_loop_triggers_for_short_periodic_cycles() {
        let a = vec!["list_directory::{\"path\":\"src\"}".to_string()];
        let b = vec!["read_file::{\"path\":\"src/bin/a.rs\"}".to_string()];
        let c = vec!["list_directory::{\"path\":\"src/bin\"}".to_string()];

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
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
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
            let signal = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
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
            let signal = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
            if i == TOOL_LOOP_SOFT_WINDOW - 1 {
                assert!(matches!(signal, ToolLoopSignal::Soft));
            } else {
                assert!(matches!(signal, ToolLoopSignal::None));
            }
        }
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
            let signal =
                supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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
            let signal =
                supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
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
            let signal =
                supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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
            let signal =
                supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
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
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
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
        assert_eq!(TOOL_ITERATION_SOFT_LIMIT, 24);
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
    fn pb_failed_read_round(
        messages: &mut Vec<crate::ai::history::Message>,
        path: &str,
        id: &str,
    ) {
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

    fn pb_dedup_read_round(
        messages: &mut Vec<crate::ai::history::Message>,
        path: &str,
        id: &str,
    ) {
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
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
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
    fn progress_budget_readonly_novel_targets_warn_once_after_breadth_threshold() {
        let mut supervisor = TurnSupervisor::default();
        let mut messages = Vec::new();
        let rounds = READ_ONLY_BREADTH_CHECK_TARGETS.max(PROGRESS_FREE_EXPLORE_ROUNDS) + 2;
        let mut breadth_warnings = 0;
        for i in 1..=rounds {
            supervisor.next_iteration();
            messages.push(pb_read_msg(&format!("src/f{i}.rs"), &format!("tc-{i}")));
            let signal = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
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
            let signal = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
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
        let signal =
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);

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
            pb_failed_read_round(&mut messages, &format!("src/missing-{i}.rs"), &format!("failed-{i}"));
            last = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
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
            pb_dedup_read_round(&mut messages, &format!("src/repeated-{i}.rs"), &format!("dedup-{i}"));
            last = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
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
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
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
    /// list_directory，使整轮 exact/coarse 签名各不相等而逃过 detect_tool_loop；此时
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
                            name: "list_directory".to_string(),
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
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
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
            matches!(signals[TOOL_LOOP_COARSE_WINDOW - 1], ToolLoopSignal::TargetRepeat),
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
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
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
            let signal = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
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
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
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
            let signal = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
            assert!(matches!(signal, ToolLoopSignal::None));
        }
        assert_eq!(supervisor.progress.consecutive_no_progress, 4);
        // 一次真正的变更动作：无进展计数清零。
        messages.push(pb_apply_patch_msg("patch-1"));
        let signal =
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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

        pb_failed_read_round(&mut compressed_messages, "src/missing.rs", "read-after-compress");

        let mut current_round = Vec::new();
        current_round.push(pb_apply_patch_msg("patch-current"));
        current_round.push(pb_tool_result("patch-current", "Patch applied successfully."));

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
        let delivered = "[Task: inspect driver via explorer @ sonnet] SUCCESS after 0.1s\nConfirmed result.";
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
            ("task_status", serde_json::json!({}), status_delivered.as_str()),
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
            let signal =
                supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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
        let signal =
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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
            let signal =
                supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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
        let signal =
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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
    fn progress_budget_real_progress_after_soft_resets_escalation_ladder() {
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
            last = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
        }
        assert!(matches!(last, ToolLoopSignal::LowProgressSoft));
        assert!(supervisor.progress.soft_injected);

        // 阶段二：一次真正的变更动作（apply_patch）-> 实质进展，重置整个升级阶梯。
        messages.push(pb_apply_patch_msg("patch-1"));
        let signal =
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(matches!(signal, ToolLoopSignal::None));
        assert!(!supervisor.progress.soft_injected);
        assert!(!supervisor.progress.ledger_injected);
        assert!(!supervisor.progress.hard_injected);
        assert_eq!(supervisor.progress.consecutive_no_progress, 0);

        // 阶段三：再次连续 5 轮无信息增益 -> 应重新触发软提示，而非跳级到 ledger/hard。
        // 若 soft_injected 未被重置，第 5 轮会因 soft_injected 仍为 true 直接走 ledger。
        for i in 0..5 {
            pb_failed_read_round(&mut messages, &format!("src/g{i}.rs"), &format!("r2-{i}"));
            last = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
        }
        assert!(
            matches!(last, ToolLoopSignal::LowProgressSoft),
            "软提示后真正推进任务应重置升级阶梯，使下一轮无进展重新从 soft 开始"
        );
        assert!(supervisor.progress.soft_injected);
        assert!(!supervisor.progress.ledger_injected);
    }

    #[test]
    fn progress_budget_escalates_soft_then_ledger_then_hard() {
        let mut supervisor = TurnSupervisor::default();
        let mut messages = Vec::new();
        // 固定 iteration=50 → over=30 → soft_threshold=3；硬停阈值由额外
        // margin 决定，确保修改 margin 后测试仍覆盖完整的升级阶梯。
        supervisor.iteration = 50;
        let mut signals = Vec::new();
        for i in 0..(3 + PROGRESS_NO_PROGRESS_HARD_MARGIN) {
            pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
            signals.push(supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            ));
        }
        // consecutive 递增：达到 soft_threshold 时依次触发 Soft 与 Ledger，
        // 到达 soft_threshold + margin 时才触发 Hard。
        assert!(matches!(signals[0], ToolLoopSignal::None));
        assert!(matches!(signals[1], ToolLoopSignal::None));
        assert!(matches!(signals[2], ToolLoopSignal::LowProgressSoft));
        assert!(matches!(signals[3], ToolLoopSignal::LowProgressLedger));
        assert!(
            signals[4..signals.len() - 1]
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
        // 固定 iteration=30 → soft_threshold=5。
        supervisor.iteration = 30;
        let mut last = ToolLoopSignal::None;
        for i in 0..5 {
            pb_failed_read_round(&mut messages, &format!("src/f{i}.rs"), &format!("r-{i}"));
            last = supervisor.record_tool_signatures(
                &messages,
                PROGRESS_FREE_EXPLORE_ROUNDS,
            );
        }
        assert!(matches!(last, ToolLoopSignal::LowProgressSoft));
        assert!(supervisor.progress.soft_injected);

        // 第 6 轮给出「实质不同的理由」（reasoning 指纹变化）→ 进入 grace 宽限，
        // 不立即升级到 ledger，给模型继续探索的空间。
        pb_failed_read_round_reasoning(
            &mut messages,
            "src/g.rs",
            "r-grace",
            Some("换一个思路：先看调用方再决定删除策略"),
        );
        let signal =
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
        assert!(
            matches!(signal, ToolLoopSignal::None),
            "new reasoning should buy a grace window instead of immediate ledger"
        );
        assert!(!supervisor.progress.ledger_injected);
        assert!(supervisor.progress.grace_until_iteration > supervisor.iteration);
        assert!(supervisor.progress.grace_consumed);

        // grace 到期后即使 reasoning 再变化，也不能继续滚动续期。
        let grace_until = supervisor.progress.grace_until_iteration;
        supervisor.iteration = grace_until;
        pb_failed_read_round_reasoning(
            &mut messages,
            "src/h.rs",
            "r-after-grace",
            Some("再换一个思路：检查配置加载顺序"),
        );
        let signal =
            supervisor.record_tool_signatures(&messages, PROGRESS_FREE_EXPLORE_ROUNDS);
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

    // 第一条用户消息一落盘就启动标题生成，不再等到完整回答结束。
    // PromptEditor 会在输入框 TUI 中轮询 session title 文件；后台任务完成后，
    // 下一次 500ms 刷新即可把新标题展示给终端前端。
    persist_pending_turn_messages(
        app,
        one_shot_mode,
        &turn_messages,
        &mut persisted_turn_messages,
    );
    if should_generate_session_title_in_background(one_shot_mode, should_quit) {
        maybe_generate_session_title(app, true).await;
    }

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
        let had_tool_call_execution = matches!(&execution, IterationExecution::ToolCall(_));
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
            if after < before {
                print_compress_status("mid-turn", before, after);
            }
            // 硬阈值：无损 + 弱损管线之后仍超额，调用 LLM 摘要兜底，
            // 把早期对话压成单条 internal_note，并在终端打 status line。
            if after > mid_turn_hard
                && should_try_llm_summary(&app.session_id, after, mid_turn_hard)
            {
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
                record_llm_summary_attempt_chars(&app.session_id, llm_after);
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
        match supervisor.record_tool_signatures_for_progress(
            &messages,
            if progress_messages.is_empty() {
                &messages
            } else {
                &progress_messages
            },
            PROGRESS_FREE_EXPLORE_ROUNDS,
        )
        {
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
            ToolLoopSignal::LowProgressSoft => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "progress-budget: no measurable progress, injecting self-reflect prompt",
                );
                inject_low_progress_soft_note(&mut messages);
            }
            ToolLoopSignal::LowProgressLedger => {
                crate::ai::driver::print::print_tool_note_line(
                    "agent-health",
                    "progress-budget: still no progress, requesting explicit decision ledger",
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
