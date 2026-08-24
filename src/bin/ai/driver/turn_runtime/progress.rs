// =============================================================================
// Progress Budget
// =============================================================================
// Extracted from orchestrator.rs during a logic-preserving split.
// Progress-budget helpers: tool-result progress classification, round target extraction and content fingerprinting.
// =============================================================================

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ToolResultProgressStatus {
    Success,
    Failure,
    DedupOnly,
    BlockedOutsideWorkspace(String),
}

pub(super) fn classify_tool_result_progress(text: &str) -> ToolResultProgressStatus {
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

pub(super) fn is_dedup_only_tool_result(text: &str) -> bool {
    let text = text.trim_start();
    text.starts_with("[deduped:") || text.starts_with("[overlap dedup:")
}

/// 从 `write_file` / `apply_patch` 的沙箱越界拒绝消息中解析被拒的目标路径。
///
/// 消息形如 `... Write blocked: path '/abs/path' is outside the allowed write
/// directory ...`。与 `blocked_outside_workspace_path`（execute_command 的命令级
/// 拒绝）平行：把「反复写同一个被拒路径」归一成稳定目标，让 target-repeat loop
/// guard 能在少数几轮内抓到，而不是任由模型对同一路径反复重试。
pub(super) fn write_blocked_outside_root_path(text: &str) -> Option<String> {
    let marker = "Write blocked: path '";
    let rest = text.split_once(marker)?.1;
    let path = rest.split_once('\'').map(|(path, _)| path)?.trim();
    (!path.is_empty()).then(|| normalize_path_like_token(path))
}

pub(super) fn blocked_outside_workspace_path(text: &str) -> Option<String> {
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
pub(super) fn extract_round_targets(messages: &[crate::ai::history::Message]) -> Vec<String> {
    extract_round_targets_inner(messages, true)
}

pub(super) fn extract_round_probe_targets(messages: &[crate::ai::history::Message]) -> Vec<String> {
    extract_round_targets_inner(messages, false)
}

pub(super) fn extract_round_targets_inner(
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
pub(super) fn content_fingerprint(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    s.trim().hash(&mut hasher);
    hasher.finish()
}

/// 提取最近一轮 assistant 的 reasoning 指纹（若有）。软提示后 reasoning 指纹
/// 变化视为「给出了新理由」，触发 grace 宽限。
pub(super) fn extract_round_reasoning_fingerprint(messages: &[crate::ai::history::Message]) -> Option<u64> {
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
pub(super) fn extract_round_evidence_fingerprints(messages: &[crate::ai::history::Message]) -> Vec<u64> {
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
        let mut read_only_command: Option<String> = None;
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
            // 无信息量探针（纯 `echo` 回显）不计入新证据：其输出恒等于回显字面量，
            // 模型每换一个 echo 字符串就产生一个"新证据"刷新 no-progress 预算，使进展
            // 刹车永远攒不满窗口（muse-spark 死循环的逃逸通道）。判据最窄——只有全 echo
            // 段才跳过，含任何真实只读段（cat/grep/…）仍照常记账，不误伤正当探查。
            if command_is_low_information_probe(command) {
                continue;
            }
            read_only_command = Some(command.to_string());
        }

        // cargo 验证命令（check/test/build 等）输出含易变编译进度/时长行：同一源码
        // 下重复运行的结果只有时长抖动，若按原始文本指纹会被误判为「新证据」而刷新
        // no-progress 预算（f08171fc 循环逃逸的同款通道）。归一化后再指纹，
        // 使相同验证结果 → 相同指纹。
        let fingerprint_text = match read_only_command.as_deref() {
            Some(command) if command_is_cargo_verify(command) => normalize_verify_output(text),
            _ => text.to_string(),
        };
        fingerprints.push(content_fingerprint(&format!("{tool_name}\0{fingerprint_text}")));
    }
    fingerprints.sort_unstable();
    fingerprints.dedup();
    fingerprints
}

/// 归一化验证类命令输出中的易变部分：丢弃编译进度/运行行（Compiling/Checking/
/// Finished/Running/Blocking 等），并抹平 `finished in 0.01s` 时长后缀，使
/// 「相同源码、相同验证结果」产出相同指纹。保留 test result 等稳定信息行。
pub(super) fn normalize_verify_output(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !(trimmed.starts_with("Compiling ")
                || trimmed.starts_with("Checking ")
                || trimmed.starts_with("Building ")
                || trimmed.starts_with("Finished ")
                || trimmed.starts_with("Running ")
                || trimmed.starts_with("Doc-tests ")
                || trimmed.starts_with("Blocking ")
                || trimmed.starts_with("Updating ")
                || trimmed.starts_with("Downloading ")
                || trimmed.starts_with("Removing "))
        })
        .map(|line| {
            if let Some(idx) = line.rfind(" finished in ") {
                let tail = &line[idx..];
                if tail.trim_end().ends_with('s') {
                    return line[..idx].trim_end().to_string();
                }
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// 归一化重扫检测的目标路径：去掉 `./` 前缀后与目标提取共用同一归一路径。
pub(crate) fn normalize_rescan_path(path: &str) -> String {
    normalize_path_like_token(path.strip_prefix("./").unwrap_or(path))
}

/// 命令是否「从文件头开始读」（整体读取，或从第 1 行 / 第 1 字节开始的分页）。
pub(super) fn command_reads_from_top(cmd: &str) -> bool {
    let cmd = cmd.trim_start();
    for prefix in ["cat ", "head ", "less ", "more ", "view ", "nl "] {
        if cmd.starts_with(prefix) {
            return true;
        }
    }
    // tail -c +1 / tail -n +1：从开头分页；tail -c +2401 等从中间开始，不算。
    if let Some(rest) = cmd.strip_prefix("tail ") {
        let mut tokens = rest.split_whitespace();
        if matches!(tokens.next(), Some("-c") | Some("-n")) && tokens.next() == Some("+1") {
            return true;
        }
    }
    // sed -n '1,40p' / sed -n "1,40p" / sed -n 1,40p：从第 1 行开始。
    if let Some(rest) = cmd.strip_prefix("sed ") {
        if let Some(expr) = rest.trim_start().strip_prefix("-n ") {
            let expr = expr.trim_start();
            let expr = expr
                .strip_prefix('\'')
                .or_else(|| expr.strip_prefix('"'))
                .unwrap_or(expr);
            return expr.starts_with("1,") || expr.starts_with("1;");
        }
    }
    false
}

/// 命令中第一个路径 token（用于确定「从头重读」的目标文件）。
pub(super) fn first_path_token(cmd: &str) -> Option<String> {
    cmd.split_whitespace()
        .find(|token| looks_like_path_token(token) || token.contains('.'))
        .map(|token| token.trim_matches(['\'', '"', '`']).to_string())
}

/// 提取最近一轮「从文件头开始读」的目标路径：read_file 的 offset 缺省/0/1；
/// execute_command 中 cat/head/less/more/view/nl/tail -c +1/sed -n 1, 等。
pub(super) fn extract_round_from_top_read_targets(messages: &[crate::ai::history::Message]) -> Vec<String> {
    use serde_json::Value;
    let Some(last_assistant) = messages.iter().rev().find(|m| m.role == "assistant") else {
        return Vec::new();
    };
    let Some(tool_calls) = last_assistant.tool_calls.as_ref() else {
        return Vec::new();
    };
    // 只统计「实际读到内容」的从头读：失败或被沙箱拦截的 execute_command/read_file
    // 没有产出任何内容，不应计入重扫计数——否则被拒的 blocked 命令也会累计，触发
    // 与真实翻页循环无关的 TargetRescan。与 extract_round_targets 共用同一结果分类。
    let results_by_call_id: FxHashMap<&str, ToolResultProgressStatus> = messages
        .iter()
        .filter(|message| message.role == "tool")
        .filter_map(|message| {
            let call_id = message.tool_call_id.as_deref()?;
            let text = message.content.as_str().unwrap_or_default();
            Some((call_id, classify_tool_result_progress(text)))
        })
        .collect();
    let mut out = Vec::new();
    for call in tool_calls.iter() {
        if matches!(
            results_by_call_id.get(call.id.as_str()),
            Some(ToolResultProgressStatus::Failure)
                | Some(ToolResultProgressStatus::BlockedOutsideWorkspace(_))
        ) {
            continue;
        }
        let Ok(args) = serde_json::from_str::<Value>(call.function.arguments.as_str()) else {
            continue;
        };
        let Some(map) = args.as_object() else {
            continue;
        };
        match call.function.name.as_str() {
            "read_file" => {
                let Some(path) = ["file_path", "path"]
                    .iter()
                    .find_map(|key| map.get(*key).and_then(|v| v.as_str()))
                else {
                    continue;
                };
                let offset = map.get("offset").and_then(|v| v.as_i64());
                if offset.is_none() || offset == Some(0) || offset == Some(1) {
                    out.push(normalize_rescan_path(path));
                }
            }
            "execute_command" => {
                let Some(command) = map.get("command").and_then(|v| v.as_str()) else {
                    continue;
                };
                if command_reads_from_top(command) {
                    if let Some(path) = first_path_token(command) {
                        out.push(normalize_rescan_path(&path));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// 提取最近一轮被 write_file/apply_patch 直接修改的目标路径（修改后从头重读
/// 属于合法验证，应清零重扫计数）。
pub(super) fn extract_round_mutated_targets(messages: &[crate::ai::history::Message]) -> Vec<String> {
    use serde_json::Value;
    let Some(last_assistant) = messages.iter().rev().find(|m| m.role == "assistant") else {
        return Vec::new();
    };
    let Some(tool_calls) = last_assistant.tool_calls.as_ref() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for call in tool_calls.iter() {
        if !is_direct_file_mutation_tool(&call.function.name) {
            continue;
        }
        let Ok(args) = serde_json::from_str::<Value>(call.function.arguments.as_str()) else {
            continue;
        };
        let Some(map) = args.as_object() else {
            continue;
        };
        for key in ["file_path", "path"] {
            if let Some(path) = map.get(key).and_then(|v| v.as_str()) {
                out.push(normalize_rescan_path(path));
            }
        }
    }
    out
}

/// 稳定的「无进展」软阈值。免费探索区内返回 usize::MAX（永不触发）。
///
/// 旧逻辑会在长任务后段从 5 轮递减到 3 / 2 轮，导致任务越复杂、越接近收尾，
/// 正常的同目标验证越容易被误判。真实 exact/coarse 重复已有独立 detector，因此这里
/// 保持稳定阈值，不再仅因 turn 变长而提高提示频率。
pub(super) fn no_progress_soft_threshold(iteration: usize, free_explore_rounds: usize) -> usize {
    if iteration <= free_explore_rounds {
        return usize::MAX;
    }
    5
}

pub(super) fn truncate_chars(s: &str, max_chars: usize) -> String {
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
