// Tool-round checkpoint phases and shell-command progress classification.

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
            Self::ImplementedNeedsVerification => "verify-and-continue",
            Self::VerifiedNeedsFinalization => "checkpoint-and-continue",
            Self::RecoveringFromError => "fix-current-failure",
        }
    }

    pub(super) fn guidance(self) -> &'static str {
        match self {
            Self::Explore => {
                "Still in the read-only evidence-gathering phase: summarize confirmed facts. If they already settle the request, write the final answer now, because the evidence suffices rather than because of this checkpoint. Otherwise name the single remaining gap and pick only the one next step with the highest information gain; prefer a precise search or one sufficiently large read, and stop expanding the unrelated evidence surface. Then execute that step and continue the loop; do not stop after merely naming it."
            }
            Self::ImplementedNeedsVerification => {
                "State has recently been modified successfully: do not resume exploration or continue unrelated changes. If the change affects compilation or runtime behavior and no check covers it yet, run the narrowest check/test that covers it; otherwise (for example docs-only or comment-only edits) review the diff/status instead of running a build or test. Then continue with the next planned step. Wrap up only when the task itself is complete or you are genuinely blocked."
            }
            Self::VerifiedNeedsFinalization => {
                "Successful verification observed: record the verified work into plan notes so it survives across turns, then continue with the next planned step. Only wrap up when the task itself is complete or you are genuinely blocked; a checkpoint alone is never a reason to stop calling tools."
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
                "This is a one-time phase checkpoint that is not an error or a tool failure; it is a pacing reminder only. Do not end the turn or write a final answer because of this checkpoint: choose the next step, execute it, and keep working until the task is complete or you are genuinely blocked."
            }
            Self::Restrict => {
                "This is the second-level checkpoint: first list the remaining necessary work, then focus on the critical path and defer nice-to-haves; do not expand the task scope. You may keep calling tools: this checkpoint paces the turn but never ends it."
            }
            Self::Finalize => {
                "This is the third-level checkpoint: this turn has run long, so consolidate verified progress into plan notes or context checkpoints first so nothing is lost across turns. Then continue the critical path, or hand a clean continuation to the next turn if the remaining work fits better there. This checkpoint never forbids tool calls and never requires ending the turn: long tasks are expected to continue past it."
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

/// Classify a shell segment by program name and selected Git/Cargo options.
/// Unknown commands return false. This is a progress heuristic, not a guarantee
/// that arbitrary arguments, redirections, or subprocesses have no side effects.
/// The program basename is used; Git's `-C` and `-c` values are skipped.
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
    // Cargo verification is classified as read-only for progress accounting even
    // though it can write build artifacts; repeated checks are not new source mutations.
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

/// Identify commands whose substantive segments all use an `echo` program.
///
/// Ignoring changing echo output keeps literal probes from refreshing the
/// no-progress budget. Any non-echo segment uses normal evidence accounting.
/// This only checks program names; it does not inspect echo arguments or expansions.
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

/// Ignore `cd` and `export` segments when classifying substantive command work.
/// They alter the command's shell context rather than representing task progress.
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
    // Every substantive segment must pass the read-only heuristic. Unknown work
    // is treated as a possible mutation so it cannot cause premature convergence.
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

/// Programs treated as read-only by the progress heuristic.
/// Argument-level side effects are not checked; `sed` and `awk` are excluded
/// because common invocations can rewrite files.
pub(super) const READ_ONLY_COMMAND_PROGRAMS: &[&str] = &[
    "ls", "cat", "grep", "rg", "find", "fd", "head", "tail", "wc", "pwd", "echo", "stat", "tree",
    "file", "which", "type", "du", "df", "ps", "date", "env", "printenv", "sort", "uniq", "cut",
    "nl", "xxd", "od", "basename", "dirname", "realpath", "readlink", "less", "more", "diff",
    "cmp", "column",
];

/// Git subcommands treated as read-only by the progress heuristic.
/// Common mutating families such as `branch`, `tag`, `remote`, and `config`
/// are excluded; argument-dependent effects of listed subcommands are not checked.
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
