// =============================================================================
// Tool-Round Checkpoint & Read-Only Classification
// =============================================================================
// Extracted from orchestrator.rs during a logic-preserving split.
// Tool-round checkpoint phase/level computation and read-only shell command classification.
// =============================================================================

use super::*;

pub(super) fn initial_tool_round_checkpoint(max_iterations: usize) -> usize {
    (max_iterations / 2).max(1).min(TOOL_ROUND_CHECKPOINT)
}

pub(super) fn tool_round_checkpoint_threshold(
    max_iterations: usize,
    level: usize,
) -> Option<usize> {
    let multiplier = *TOOL_ROUND_CHECKPOINT_MULTIPLIERS.get(level)?;
    let threshold = initial_tool_round_checkpoint(max_iterations).checked_mul(multiplier)?;
    if level > 0 && threshold >= max_iterations {
        return None;
    }
    Some(threshold)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolRoundCheckpointPhase {
    Explore,
    ImplementedNeedsVerification,
    VerifiedNeedsFinalization,
    RecoveringFromError,
}

impl ToolRoundCheckpointPhase {
    pub(super) fn recent_progress(self) -> &'static str {
        match self {
            Self::Explore => "read-only",
            Self::ImplementedNeedsVerification => "mutation",
            Self::VerifiedNeedsFinalization => "verification-success",
            Self::RecoveringFromError => "verification-failure",
        }
    }

    pub(super) fn action(self) -> &'static str {
        match self {
            Self::Explore => "choose-one-next-step",
            Self::ImplementedNeedsVerification => "verify-and-wrap-up",
            Self::VerifiedNeedsFinalization => "finalize",
            Self::RecoveringFromError => "fix-current-failure",
        }
    }

    pub(super) fn guidance(self) -> &'static str {
        match self {
            Self::Explore => {
                "Still in the read-only evidence-gathering phase: summarize confirmed facts and the single remaining gap, then pick only the one next step with the highest information gain; prefer a precise search or one sufficiently large read, and stop expanding the unrelated evidence surface."
            }
            Self::ImplementedNeedsVerification => {
                "State has recently been modified successfully: do not resume exploration or continue unrelated changes; run the narrowest check/test that covers the change, check diff/status if needed, then wrap up immediately."
            }
            Self::VerifiedNeedsFinalization => {
                "Successful verification observed: unless there is a clear gap that would change the conclusion, do not call any more tools; summarize the changes, verification results, and remaining risks, and complete the answer."
            }
            Self::RecoveringFromError => {
                "A recent change or verification failed: diagnose only the current failure and make one targeted fix/retry without expanding into unrelated issues; if still blocked, clearly report the blocker and wrap up."
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolRoundCheckpointLevel {
    Review,
    Restrict,
    Finalize,
}

impl ToolRoundCheckpointLevel {
    pub(super) fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Review,
            1 => Self::Restrict,
            _ => Self::Finalize,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Review => "review",
            Self::Restrict => "restrict",
            Self::Finalize => "finalize",
        }
    }

    pub(super) fn guidance(self) -> &'static str {
        match self {
            Self::Review => {
                "This is a one-time phase checkpoint that is not an error or a tool failure."
            }
            Self::Restrict => {
                "This is the second-level checkpoint: first list the remaining necessary work, then only complete critical fixes and minimal verification; do not expand the task scope."
            }
            Self::Finalize => {
                "This is the third-level checkpoint: wrap up based on existing evidence; do not call further tools unless the current verification failed and one targeted fix can resolve it directly."
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ToolRoundCheckpoint {
    pub(super) level: ToolRoundCheckpointLevel,
    pub(super) phase: ToolRoundCheckpointPhase,
    pub(super) threshold: usize,
}

pub(super) fn checkpoint_tool_call_effects(tool_call: &crate::ai::types::ToolCall) -> (bool, bool) {
    if tool_call.function.name != "execute_command" {
        return (tool_call_is_successful_mutation_candidate(tool_call), false);
    }
    let effects = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
        .ok()
        .map(|args| super::super::iteration::execute_command_segment_effects_for_args(&args))
        .unwrap_or_default();
    (
        effects.iter().any(|effect| effect.project_mutation),
        effects
            .iter()
            .any(|effect| effect.scope_review || effect.behavior_check),
    )
}

pub(super) fn tool_round_checkpoint_phase(
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
pub(super) fn shell_segment_is_read_only(segment: &str) -> bool {
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
    // `cargo check/test/build` 等验证命令不改源码：重复运行同一验证命令（相同源码、
    // 相同结果）不构成实质进展，不应再被 round_has_mutation 计为 Mutation 而刷新
    // no-progress 预算（f08171fc 循环正是靠反复 `cargo test` 逃逸升级）。
    if program == "cargo" {
        let (subcommand, fix_flag, check_flag) = parse_cargo_tokens(tokens);
        return match subcommand {
            Some("check" | "test" | "build" | "bench" | "doc" | "metadata" | "tree") => true,
            // `clippy --fix` 会改写源码 → 非只读；`fmt` 仅 `--check` 时只读。
            Some("clippy") => !fix_flag,
            Some("fmt") => check_flag,
            _ => false,
        };
    }
    READ_ONLY_COMMAND_PROGRAMS.contains(&program)
}

/// 解析 `cargo` 子命令与 `--fix`/`--check` 标志，跳过 `--config`/`--manifest-path`
/// 等取值选项。返回 `(子命令, 是否 --fix, 是否 --check)`。
fn parse_cargo_tokens<'a>(tokens: impl Iterator<Item = &'a str>) -> (Option<&'a str>, bool, bool) {
    let mut subcommand = None;
    let mut skip_next = false;
    let mut fix_flag = false;
    let mut check_flag = false;
    for token in tokens {
        if skip_next {
            skip_next = false;
            continue;
        }
        match token {
            "-C" | "-c" | "--config" | "--manifest-path" | "--target-dir" | "--lockfile-path" => {
                skip_next = true;
            }
            "--fix" => fix_flag = true,
            "--check" => check_flag = true,
            _ if token.starts_with('-') => {}
            _ if subcommand.is_none() => subcommand = Some(token),
            _ => {}
        }
    }
    (subcommand, fix_flag, check_flag)
}

/// cargo 验证类子命令：输出含易变编译进度/时长行，evidence 指纹需归一化。
/// 集合含 `clippy`/`fmt` 属过包含（`clippy --fix`/裸 `fmt` 不满足只读判定，
/// 输出根本不会进入 evidence 归一化分支）——过包含因只读门控而无害。
const CARGO_VERIFY_SUBCOMMANDS: &[&str] = &[
    "check", "test", "build", "bench", "doc", "metadata", "tree", "clippy", "fmt",
];

/// 命令（可带 `cd`/`export` 前导段）是否为 cargo 验证类子命令。供 evidence 指纹
/// 归一化复用与 shell_segment_is_read_only 同一套子命令解析。
pub(crate) fn command_is_cargo_verify(command: &str) -> bool {
    for segment in split_shell_segments_for_coarse(command) {
        if shell_segment_is_nav_or_env(&segment) {
            continue;
        }
        let mut tokens = segment.split_whitespace();
        let Some(program) = tokens.next() else {
            continue;
        };
        if program.rsplit('/').next().unwrap_or(program) != "cargo" {
            // 非 cargo 段跳过：`echo hi && cargo check` 这类多段命令只要含任一
            // cargo 验证段即视为构建校验，必须排除出同轮重放。
            continue;
        }
        let (subcommand, _, _) = parse_cargo_tokens(tokens);
        if subcommand.is_some_and(|sub| CARGO_VERIFY_SUBCOMMANDS.contains(&sub)) {
            return true;
        }
    }
    false
}

/// 命令是否为"无信息量探针"：所有实质段都是 `echo`。
///
/// 背景：`echo "integrate"` / `echo "x"` 这类命令只读、必成功、输出等于回显字面量，
/// 本身不携带对任务的新信息。但 evidence 指纹按输出内容取哈希，模型每换一个 echo
/// 字符串就产生一个"新证据"、刷新 no-progress 预算，使进展刹车（soft→ledger→hard）
/// 永远攒不满窗口——这正是 muse-spark 死循环里 `echo` 反复刷新预算的逃逸通道。
///
/// 判据刻意最窄：要求**每个**实质段（跳过 `cd`/`export` 前导）都是裸 `echo`。
/// 只要含任一非 echo 段（如 `echo hi && cargo check`、`cat f`）即返回 false，
/// 交回常规证据记账——避免把真实只读探查（`cat version.txt`→内容）误判为无信息。
pub(crate) fn command_is_low_information_probe(command: &str) -> bool {
    let mut saw_substantive = false;
    for segment in split_shell_segments_for_coarse(command) {
        if shell_segment_is_nav_or_env(&segment) {
            continue;
        }
        saw_substantive = true;
        let mut tokens = segment.split_whitespace();
        let Some(program) = tokens.next() else {
            return false;
        };
        if program.rsplit('/').next().unwrap_or(program) != "echo" {
            return false;
        }
    }
    saw_substantive
}

/// `cd` / `export` 只改变工作目录或环境，不写文件系统，本身无副作用。作为前导段
/// 跳过，避免 `cd X && git status` 这类「游走 + 检查」命令被误判为变更。
pub(super) fn shell_segment_is_nav_or_env(segment: &str) -> bool {
    let mut tokens = segment.split_whitespace();
    let Some(program) = tokens.next() else {
        return false;
    };
    matches!(
        program.rsplit('/').next().unwrap_or(program),
        "cd" | "export"
    )
}

pub(crate) fn execute_command_is_read_only(command: &str) -> bool {
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
pub(super) const READ_ONLY_COMMAND_PROGRAMS: &[&str] = &[
    "ls", "cat", "grep", "rg", "find", "fd", "head", "tail", "wc", "pwd", "echo", "stat", "tree",
    "file", "which", "type", "du", "df", "ps", "date", "env", "printenv", "sort", "uniq", "cut",
    "nl", "xxd", "od", "basename", "dirname", "realpath", "readlink", "less", "more", "diff",
    "cmp", "column",
];

/// 明确只读的 git 子命令。刻意排除 `branch`/`tag`/`remote`/`config` 等可带副作用的
/// 子命令（裸列出形式虽只读，但带参可变更；无法区分时按可能变更处理）。
pub(super) const GIT_READ_ONLY_SUBCOMMANDS: &[&str] = &[
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
