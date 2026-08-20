// =============================================================================
// Tool-Loop Detection
// =============================================================================
// Extracted from orchestrator.rs during a logic-preserving split.
// Tool-round signature extraction, coarse shell-command signatures and the loop / target-rescan detectors used by the turn supervisor.
// =============================================================================

use super::*;

/// 变更类工具：调用这些动作（或产出 final text）即视为本轮有实质动作、算进展。
pub(super) const MUTATION_TOOL_NAMES: &[&str] = &[
    "apply_patch",
    "write_file",
    "plan",
    "plan_update",
    "task_spawn",
    "task_spawn_batch",
    "task_wait",
    "task_cancel",
    "task_status",
    "execute_command",
];


/// 提取最近一轮 assistant 消息中的 (tool_name, args_json) 签名集合。
/// 任何一个签名与窗口内某轮完全一致即认为有循环倾向。
pub(super) fn extract_round_tool_signatures(messages: &[crate::ai::history::Message]) -> Option<Vec<String>> {
    extract_round_tool_signatures_inner(messages, false)
}

/// 提取「粗粒度」签名：剥离 offset/limit/page 等易变翻页参数后再归一化。
/// 用于抓字节精确检测漏掉的同文件翻页 / 仅微调分页参数的重复检索。
/// 对 `execute_command` 额外折叠 shell 中的低收益变体（如 `| head -20/-30`、
/// `2>/dev/null`、`ls -la/-lt` 的细微差异，以及 git log/show/diff 取证视角的
/// 轻微切换），让同目标资源的反复试探能命中。
pub(super) fn extract_round_tool_signatures_coarse(
    messages: &[crate::ai::history::Message],
) -> Option<Vec<String>> {
    extract_round_tool_signatures_inner(messages, true)
}

pub(super) fn extract_round_tool_signatures_inner(
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
pub(super) fn strip_volatile_args(value: &mut serde_json::Value) {
    if let Some(map) = value.as_object_mut() {
        for key in VOLATILE_ARG_KEYS {
            map.remove(*key);
        }
    }
}

pub(super) fn normalize_coarse_tool_args(tool_name: &str, value: &mut serde_json::Value) {
    match tool_name {
        "execute_command" => normalize_coarse_execute_command_args(value),
        "task_wait" => normalize_coarse_task_wait_args(value),
        "task_status" => normalize_coarse_task_status_args(value),
        _ => {}
    }
}

pub(super) fn normalize_coarse_execute_command_args(value: &mut serde_json::Value) {
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

pub(super) fn normalize_coarse_task_wait_args(value: &mut serde_json::Value) {
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

pub(super) fn normalize_coarse_task_status_args(value: &mut serde_json::Value) {
    if let Some(map) = value.as_object_mut() {
        // task_status 忽略参数；不同空壳参数不应逃过 coarse 循环检测。
        map.clear();
    }
}

pub(super) fn coarse_execute_command_signature(command: &str) -> String {
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

pub(super) fn split_shell_segments_for_coarse(command: &str) -> Vec<String> {
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

pub(super) fn tokenize_shell_words_for_coarse(command: &str) -> Vec<String> {
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

pub(super) fn coarse_shell_segment_signature(segment: &str) -> Option<String> {
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

pub(super) fn is_window_only_shell_segment(program: &str, tokens: &[String]) -> bool {
    match program {
        "head" | "tail" => tokens[1..]
            .iter()
            .all(|token| token.starts_with('-') || token.chars().all(|ch| ch.is_ascii_digit())),
        "wc" => tokens[1..].iter().all(|token| token.starts_with('-')),
        _ => false,
    }
}

pub(super) fn normalize_ls_segment(tokens: &[String]) -> String {
    let mut paths = collect_shell_target_tokens(tokens, 1, false);
    if paths.is_empty() {
        paths.push(".".to_string());
    }
    format!("ls:{}", paths.join(","))
}

pub(super) fn normalize_search_segment(program: &str, tokens: &[String]) -> String {
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

pub(super) fn normalize_git_segment(tokens: &[String]) -> String {
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

pub(super) fn find_git_subcommand_index(tokens: &[String]) -> Option<usize> {
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

pub(super) fn git_option_takes_value(token: &str) -> bool {
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

pub(super) fn looks_like_git_revision_token(token: &str) -> bool {
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

pub(super) fn normalize_git_revision_token(token: &str) -> String {
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

pub(super) fn normalize_generic_shell_segment(program: &str, tokens: &[String]) -> String {
    let mut paths = collect_shell_target_tokens(tokens, 1, true);
    if paths.is_empty() {
        program.to_string()
    } else {
        paths.sort();
        paths.dedup();
        format!("{program}:{}", paths.join(","))
    }
}

/// 判断 shell 参数 token 是否是「翻页/窗口」类数字字面量：纯数字、+N/−N 字节
/// 偏移、sed/awk 行号区间（1,40p / 40,80 / 5p / 1,$p）。这类字面量只描述读哪个
/// 窗口，不是目标资源本身。coarse 签名里保留它们会让「同一文件换窗口翻页」的
/// 每轮签名互不相同，使 coarse / target-repeat 两道检测都抓不到翻页循环——这与
/// read_file 的 offset/limit 被 strip_volatile_args 剥离同理。剥掉后，
/// `tail -c +2401 f | head -c 2400` 与 `tail -c +2402 f | head -c 1400`
/// 折叠为同一签名 `tail:f`，翻页循环即可被粗粒度检测命中。其余非数字字面量
/// （version2、s/foo/bar/g 等）不受影响。
pub(super) fn is_numeric_window_literal(token: &str) -> bool {
    if token.is_empty() {
        return true;
    }
    let body = token.trim_start_matches(['+', '-']);
    if !body.is_empty() && body.chars().all(|ch| ch.is_ascii_digit()) {
        return true;
    }
    // sed/awk 行号区间或命令尾字母：仅当 token 由「数字 + 可选逗号/$ + 少数命令
    // 尾字母」构成时才算窗口字面量，其余（如 version2）保留。
    let mut saw_digit = false;
    for (idx, ch) in token.chars().enumerate() {
        match ch {
            '0'..='9' => saw_digit = true,
            ',' | '$' => {}
            'p' | 'd' | 'q' | 's' | 'g' | 'w' | 'a' | 'i' | 'c' if idx > 0 && saw_digit => {}
            _ => return false,
        }
    }
    saw_digit
}

pub(super) fn collect_shell_target_tokens(
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
        if keep_literals && !is_numeric_window_literal(token) {
            out.push(token.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

pub(super) fn should_skip_shell_token(token: &str) -> bool {
    matches!(token, "|" | ";" | "&&" | "||" | "&")
        || token.starts_with("2>")
        || token.starts_with("1>")
        || token.starts_with(">")
        || token.starts_with("<")
}

pub(super) fn looks_like_path_token(token: &str) -> bool {
    token == "."
        || token == ".."
        || token.starts_with('/')
        || token.starts_with("./")
        || token.starts_with("../")
        || token.starts_with("~/")
        || token.contains('/')
}

pub(super) fn normalize_path_like_token(token: &str) -> String {
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

pub(super) fn detect_tool_loop(history: &[Vec<String>], window: usize) -> bool {
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

pub(super) fn signature_set_is_execute_command_only(sigs: &[String]) -> bool {
    !sigs.is_empty() && sigs.iter().all(|sig| sig.starts_with("execute_command::"))
}

pub(super) fn detect_execute_command_coarse_loop(history: &[Vec<String>], window: usize) -> bool {
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
pub(super) fn detect_target_repeat_loop(history: &[Vec<String>], window: usize) -> bool {
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

pub(super) fn is_direct_file_mutation_tool(name: &str) -> bool {
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
pub(super) fn round_has_mutation(messages: &[crate::ai::history::Message]) -> bool {
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
                // 推进，避免把真实改动误判为无进展而过早停。temp write / dry-run
                // 只改 session 临时态，不得让纯只读广度刹车永久失效。
                if !tool_call_is_successful_mutation_candidate(tc) {
                    return false;
                }
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

/// 纯只读广度硬停只应被真实项目修改关闭；plan / task 生命周期等虽然算任务进展，
/// 但不能把后续无限串行扫描伪装成「读+改」实现任务。
pub(super) fn round_has_project_mutation(messages: &[crate::ai::history::Message]) -> bool {
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
            Some((
                message.tool_call_id.as_deref()?,
                message.content.as_str().unwrap_or_default(),
            ))
        })
        .collect();
    tool_calls.iter().any(|tool_call| {
        let (project_mutation, _) = checkpoint_tool_call_effects(tool_call);
        if !project_mutation {
            return false;
        }
        match tool_call.function.name.as_str() {
            "write_file" | "apply_patch" => tool_results_by_call_id
                .get(tool_call.id.as_str())
                .is_none_or(|text| {
                    matches!(
                        classify_tool_result_progress(text),
                        ToolResultProgressStatus::Success
                    )
                }),
            "execute_command" => true,
            _ => false,
        }
    })
}

pub(super) fn task_tool_result_delivered_task_output(text: &str) -> bool {
    text.lines()
        .any(|line| line.trim_start().starts_with("[Task: "))
}

pub(super) fn current_tool_round_messages(
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
