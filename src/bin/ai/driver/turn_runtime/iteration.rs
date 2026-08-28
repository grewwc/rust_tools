use colored::Colorize;
use rust_tools::commonw::FastSet;
use serde_json::Value;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;

use crate::ai::{
    driver::{drain_response, input, skill_runtime},
    history::{
        Message, ROLE_INTERNAL_NOTE, compress::llm_prune, last_real_user_index,
    },
    mcp::McpClient,
    middleware::request::build_llm_client_chain,
    ports::{DefaultLlmClient, LlmClient, LlmRequest},
    request,
    stream,
    tools::task_tools,
    types::{App, StreamOutcome, StreamResult},
};

use super::{
    CompressionReport, MID_TURN_COMPRESS_SOFT_FLOOR, MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
    MID_TURN_LLM_SUMMARY_MAX_CHARS, TurnOutcome, context_budget,
    persistence::persist_pending_turn_messages,
    pre_request_llm_summary_threshold, record_llm_summary_attempt_chars, should_try_llm_summary,
    types::{IterationExecution, ToolCallExecution},
};

struct StreamingFlagGuard {
    flag: Arc<AtomicBool>,
}

impl StreamingFlagGuard {
    fn new(flag: &Arc<AtomicBool>) -> Self {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        Self {
            flag: Arc::clone(flag),
        }
    }
}

impl Drop for StreamingFlagGuard {
    fn drop(&mut self) {
        self.flag.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

fn request_visible_tool_names(app: &App) -> FastSet<String> {
    app.agent_context
        .as_ref()
        .map(|ctx| {
            ctx.tools
                .iter()
                .map(|tool| tool.function.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

fn push_project_target(targets: &mut Vec<PathBuf>, seen: &mut FastSet<String>, raw_path: &str) {
    let path = raw_path.trim().trim_matches(|ch| matches!(ch, '"' | '\''));
    if path.is_empty() || !seen.insert(path.to_string()) {
        return;
    }
    targets.push(PathBuf::from(path));
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ExecuteCommandSegmentEffect {
    pub(super) mutation: bool,
    pub(super) project_mutation: bool,
    pub(super) scope_review: bool,
    pub(super) behavior_check: bool,
    pub(super) success_guaranteed: bool,
}

#[derive(Default)]
struct ExecuteCommandAnalysis {
    effects: Vec<ExecuteCommandSegmentEffect>,
    mutation_targets: Vec<PathBuf>,
    unknown_mutation_bases: Vec<PathBuf>,
}

fn shell_command_tokens(segment: &str) -> Vec<String> {
    crate::ai::tools::service::audit::effective_command_tokens(segment)
}

fn command_subcommand(tokens: &[String]) -> Option<&str> {
    crate::ai::tools::service::audit::command_subcommand_index(tokens)
        .and_then(|index| tokens.get(index))
        .map(String::as_str)
}

fn command_program(tokens: &[String]) -> Option<&str> {
    tokens
        .first()
        .and_then(|token| Path::new(token).file_name().and_then(|name| name.to_str()))
}

fn segment_verification_effect(tokens: &[String]) -> (bool, bool) {
    let Some(program) = command_program(tokens) else {
        return (false, false);
    };
    let subcommand = command_subcommand(tokens).unwrap_or_default();
    let scope_review = program == "git" && matches!(subcommand, "diff" | "status");
    let behavior_check = match program {
        "cargo" => matches!(subcommand, "check" | "test" | "clippy" | "build"),
        "pytest" => true,
        "go" => subcommand == "test",
        "npm" | "pnpm" | "yarn" => matches!(subcommand, "test" | "check"),
        "make" => matches!(subcommand, "test" | "check"),
        _ => false,
    };
    (scope_review, behavior_check)
}

/// Write-file intent classification for single-segment commands.
///
/// The read-only/non-read-only two-state semantics are folded into this three-state enum; non-read-only mixes two meanings:
/// - `WriteIntended`: a program known to explicitly write local files (`sed -i`, `bytedcli --output-dir`,
///   `git commit`, `cargo build`, etc.), or a known mutator;
/// - `Unknown`: an unrecognized program (interpreters/scripts like `python3`/`node`/`perl -e`) with
///   neither read-only nor write knowledge — it may or may not write.
///
/// An unknown program running from the project cwd has no write evidence, so it must not be judged as "the project was changed";
/// otherwise read-only python/node checks would be misjudged as project changes, polluting downstream consumers such as
/// checkpoint-phase prompts. The completion evidence gate only credits provable tool-level changes (apply_patch /
/// write_file) and does not depend on this classification; this classification still serves checkpoint-phase prompts and other downstream consumers.
/// Interpreters cannot be exhaustively enumerated (no whitelist possible), so classification must rely on write evidence rather than the program name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentWriteKind {
    ReadOnly,
    WriteIntended,
    Unknown,
}

fn segment_write_kind(tokens: &[String]) -> SegmentWriteKind {
    let Some(program) = command_program(tokens) else {
        return SegmentWriteKind::ReadOnly;
    };
    let subcommand = command_subcommand(tokens).unwrap_or_default();
    match program {
        "cd" | "pwd" | "ls" | "cat" | "rg" | "grep" | "head" | "tail" | "wc" | "stat" | "file"
        | "which" | "type" | "echo" | "printf" | "sleep" | "true" | "false" | "test" | "sort"
        | "uniq" | "cut" | "find" | "comm" | "tr" => SegmentWriteKind::ReadOnly,
        "sed" => {
            if tokens
                .iter()
                .any(|token| token == "-i" || token.starts_with("-i"))
            {
                SegmentWriteKind::WriteIntended
            } else {
                SegmentWriteKind::ReadOnly
            }
        }
        "git" => {
            // The subcommand may not be at index 1: `git -C <repo> tag -l` / `git --git-dir=... tag` etc.
            // put global options first. The `tag`/`worktree` branches need "the first argument after the subcommand"
            // (`subcommand_index + 1`); hardcoding `tokens.get(2)` would treat a global option's value
            // (such as the `-C` path) as the tag/worktree argument, judging a read-only query as writing refs.
            let subcommand_index =
                crate::ai::tools::service::audit::command_subcommand_index(tokens).unwrap_or(1);
            let tag_worktree_arg = tokens.get(subcommand_index + 1).map(|s| s.as_str());
            if matches!(
                subcommand,
                "diff"
                    | "status"
                    | "log"
                    | "show"
                    | "reflog"
                    | "branch"
                    | "rev-parse"
                    | "ls-files"
                    | "grep"
                    | "blame"
                    | "ls-remote"
                    | "remote"
                    | "fetch"
            ) {
                SegmentWriteKind::ReadOnly
            } else if subcommand == "tag" {
                // `git tag` (no args) and `git tag -l/--list/-n/...` are read-only queries;
                // the other forms (`git tag <name>`, `-a/-d/-m/-s/-f`, etc.) write refs and count as writes.
                // `-l/-n/--list/--sort/--format` accept attached-value forms (`-n1`, `-ln`,
                // `--sort=...`, `--format=...`), also read-only, matched by prefix; these prefixes do not
                // collide with git tag's write flags (`-a/-d/-f/-m/-s/-u/-F/-e`).
                let read_only_tag_arg = |arg: &str| {
                    arg == "-l"
                        || arg == "--list"
                        || arg == "-n"
                        || arg == "--format"
                        || arg == "--sort"
                        || arg == "-v"
                        || arg == "--verify"
                        || arg == "--contains"
                        || arg == "--merged"
                        || arg == "--no-merged"
                        || arg == "--points-at"
                        || arg.starts_with("-l")
                        || arg.starts_with("-n")
                        || arg.starts_with("--list")
                        || arg.starts_with("--sort")
                        || arg.starts_with("--format")
                };
                match tag_worktree_arg {
                    None => SegmentWriteKind::ReadOnly,
                    Some(arg) if read_only_tag_arg(arg) => SegmentWriteKind::ReadOnly,
                    Some(_) => SegmentWriteKind::WriteIntended,
                }
            } else if subcommand == "worktree" {
                // `git worktree list` is read-only; `add/remove/prune/move/repair/lock/unlock` write.
                match tag_worktree_arg {
                    None | Some("list") => SegmentWriteKind::ReadOnly,
                    Some(_) => SegmentWriteKind::WriteIntended,
                }
            } else {
                SegmentWriteKind::WriteIntended
            }
        }
        "cargo" => {
            if matches!(subcommand, "check" | "test" | "clippy" | "build" | "metadata") {
                SegmentWriteKind::ReadOnly
            } else {
                SegmentWriteKind::WriteIntended
            }
        }
        "go" => {
            if subcommand == "test" {
                SegmentWriteKind::ReadOnly
            } else {
                SegmentWriteKind::WriteIntended
            }
        }
        "npm" | "pnpm" | "yarn" => {
            if matches!(subcommand, "test" | "check") {
                SegmentWriteKind::ReadOnly
            } else {
                SegmentWriteKind::WriteIntended
            }
        }
        "make" => {
            if matches!(subcommand, "test" | "check") {
                SegmentWriteKind::ReadOnly
            } else {
                SegmentWriteKind::WriteIntended
            }
        }
        "pytest" => SegmentWriteKind::ReadOnly,
        // `bytedcli` is the ByteDance internal platform CLI (codebase/db/faas/log subcommands
        // are all remote API queries/operations); it does not write local project files by default, so it is classified read-only;
        // but `--output`/`--output-dir`/`--manifest` explicitly write local files
        // (e.g. `codebase mr artifacts download --output-dir ...`) and must count as a change.
        "bytedcli" => {
            if tokens.iter().any(|token| {
                token == "--output"
                    || token == "--output-dir"
                    || token == "--manifest"
                    || token.starts_with("--output=")
                    || token.starts_with("--output-dir=")
                    || token.starts_with("--manifest=")
            }) {
                SegmentWriteKind::WriteIntended
            } else {
                SegmentWriteKind::ReadOnly
            }
        }
        _ => SegmentWriteKind::Unknown,
    }
}

fn positional_tokens(tokens: &[String], skip: usize) -> Vec<String> {
    tokens
        .iter()
        .skip(skip)
        .filter(|token| {
            !token.is_empty()
                && !token.starts_with('-')
                && !matches!(token.as_str(), ">" | ">>" | "<" | "<<")
                && !token.contains('=')
        })
        .cloned()
        .collect()
}

fn redirection_targets(segment: &str) -> Vec<String> {
    let mut targets = Vec::new();
    let chars: Vec<char> = segment.chars().collect();
    let mut index = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    while index < chars.len() {
        let ch = chars[index];
        if escaped {
            escaped = false;
            index += 1;
            continue;
        }
        if ch == '\\' && !in_single {
            escaped = true;
            index += 1;
            continue;
        }
        if ch == '\'' && !in_double {
            in_single = !in_single;
            index += 1;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
            index += 1;
            continue;
        }
        if ch != '>' || in_single || in_double {
            index += 1;
            continue;
        }
        index += 1;
        if index < chars.len() && chars[index] == '>' {
            index += 1;
        }
        while index < chars.len() && chars[index].is_whitespace() {
            index += 1;
        }
        let start = index;
        while index < chars.len()
            && !chars[index].is_whitespace()
            && !matches!(chars[index], ';' | '|' | '&')
        {
            index += 1;
        }
        if start < index {
            targets.push(chars[start..index].iter().collect());
        }
    }
    targets
}

fn mutation_target_tokens(tokens: &[String]) -> (Vec<String>, bool) {
    let Some(program) = command_program(tokens) else {
        return (Vec::new(), false);
    };
    let subcommand_index =
        crate::ai::tools::service::audit::command_subcommand_index(tokens).unwrap_or(1);
    let mut targets = match program {
        "touch" | "mkdir" | "rm" | "rmdir" | "truncate" | "tee" | "ln" => {
            positional_tokens(tokens, 1)
        }
        "cp" | "mv" | "install" => positional_tokens(tokens, 1),
        "chmod" | "chown" | "chgrp" => positional_tokens(tokens, 2),
        "sed"
            if tokens
                .iter()
                .any(|token| token == "-i" || token.starts_with("-i")) =>
        {
            positional_tokens(tokens, 1).into_iter().skip(1).collect()
        }
        "perl"
            if tokens.iter().any(|token| {
                token.starts_with("-pi") || token.starts_with("-ip") || token.starts_with("-i")
            }) =>
        {
            positional_tokens(tokens, 1)
        }
        "git"
            if matches!(
                command_subcommand(tokens),
                Some("add" | "checkout" | "restore" | "rm" | "mv")
            ) =>
        {
            positional_tokens(tokens, subcommand_index + 1)
        }
        _ => Vec::new(),
    };
    targets.retain(|target| !target.chars().all(|ch| ch.is_ascii_digit()));
    // `perl` is an interpreter; only the `-p/-i` (in-place edit) forms "write files by design";
    // a bare `perl -e '...'` has no write evidence, just like python3/node.
    let known_mutator = matches!(
        program,
        "touch"
            | "mkdir"
            | "rm"
            | "rmdir"
            | "truncate"
            | "tee"
            | "ln"
            | "cp"
            | "mv"
            | "install"
            | "chmod"
            | "chown"
            | "chgrp"
            | "sed"
            | "git"
            | "cargo"
            | "npm"
            | "pnpm"
            | "yarn"
            | "make"
    ) || (program == "perl"
        && tokens.iter().any(|token| {
            token.starts_with("-pi") || token.starts_with("-ip") || token.starts_with("-i")
        }));
    (targets, known_mutator)
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn resolve_command_path(base: &Path, raw: &str) -> Option<PathBuf> {
    let raw = raw.trim().trim_matches(|ch| matches!(ch, '\'' | '"'));
    if raw.is_empty() || raw.starts_with('$') || raw.contains(['*', '?', '[', ']']) {
        return None;
    }
    let path = Path::new(raw);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    Some(normalize_lexical(&resolved))
}

fn analyze_execute_command(command: &str, initial_cwd: &Path) -> ExecuteCommandAnalysis {
    let mut analysis = ExecuteCommandAnalysis::default();
    let mut cwd = normalize_lexical(initial_cwd);
    let mut seen = FastSet::default();
    let project_root = project_root_dir();
    let segments = crate::ai::tools::service::audit::split_unquoted_command_segments(command);
    let success_guaranteed = segments.iter().all(|segment| {
        matches!(
            segment.join,
            crate::ai::tools::service::audit::ShellJoin::Start
                | crate::ai::tools::service::audit::ShellJoin::And
        )
    });
    for segment in segments {
        let tokens = shell_command_tokens(&segment.command);
        if tokens.is_empty() {
            continue;
        }
        if command_program(&tokens) == Some("cd") {
            if let Some(path) = tokens
                .get(1)
                .and_then(|path| resolve_command_path(&cwd, path))
            {
                cwd = path;
            }
            analysis.effects.push(ExecuteCommandSegmentEffect {
                success_guaranteed,
                ..Default::default()
            });
            continue;
        }
        let (scope_review, behavior_check) = segment_verification_effect(&tokens);
        let mut raw_targets = redirection_targets(&segment.command);
        let has_redirection = !raw_targets.is_empty();
        let write_kind = segment_write_kind(&tokens);
        let read_only = matches!(write_kind, SegmentWriteKind::ReadOnly);
        let (mut command_targets, known_mutator) = mutation_target_tokens(&tokens);
        raw_targets.append(&mut command_targets);
        // Only programs "known to write local files" (WriteIntended or a known mutator) may be judged
        // as a project change from the project cwd alone; unknown programs (python3/node/...) have no write
        // evidence, and the cwd fallback would misjudge read-only checks as changes, tripping the completion evidence gate
        // (`successful_post_mutation_verification` gets reset) and making the model repeat its conclusion.
        let known_writer =
            matches!(write_kind, SegmentWriteKind::WriteIntended) || known_mutator;
        let mutation = has_redirection || !read_only;
        if !mutation {
            analysis.effects.push(ExecuteCommandSegmentEffect {
                scope_review,
                behavior_check,
                success_guaranteed,
                ..Default::default()
            });
            continue;
        }
        let mut resolved_any = false;
        let mut project_mutation = false;
        for raw_target in raw_targets {
            if let Some(target) = resolve_command_path(&cwd, &raw_target) {
                project_mutation |= path_is_in_project(&target, &project_root);
                let key = target.to_string_lossy().into_owned();
                if seen.insert(key) {
                    analysis.mutation_targets.push(target);
                }
                resolved_any = true;
            } else {
                if known_writer {
                    project_mutation |= path_is_in_project(&cwd, &project_root);
                }
                if !analysis.unknown_mutation_bases.contains(&cwd) {
                    analysis.unknown_mutation_bases.push(cwd.clone());
                }
            }
        }
        if !read_only && (!known_mutator || !resolved_any) {
            if known_writer {
                project_mutation |= path_is_in_project(&cwd, &project_root);
            }
            if !analysis.unknown_mutation_bases.contains(&cwd) {
                analysis.unknown_mutation_bases.push(cwd.clone());
            }
        }
        analysis.effects.push(ExecuteCommandSegmentEffect {
            mutation,
            project_mutation,
            scope_review,
            behavior_check,
            success_guaranteed,
        });
    }
    analysis
}

pub(super) fn execute_command_segment_effects(command: &str) -> Vec<ExecuteCommandSegmentEffect> {
    analyze_execute_command(command, Path::new(".")).effects
}

pub(super) fn execute_command_segment_effects_for_args(
    args: &Value,
) -> Vec<ExecuteCommandSegmentEffect> {
    let Some(command) = args.get("command").and_then(Value::as_str) else {
        return Vec::new();
    };
    analyze_execute_command(command, &command_base_dir(args)).effects
}

pub(super) fn execute_command_may_mutate(command: &str) -> bool {
    execute_command_segment_effects(command)
        .iter()
        .any(|effect| effect.mutation)
}

fn command_base_dir(args: &Value) -> PathBuf {
    let effective_cwd = project_root_dir();
    args.get("cwd")
        .and_then(Value::as_str)
        .filter(|cwd| !cwd.trim().is_empty())
        .map(Path::new)
        .map(|cwd| {
            let resolved = if cwd.is_absolute() {
                cwd.to_path_buf()
            } else {
                effective_cwd.join(cwd)
            };
            normalize_lexical(&resolved)
        })
        .unwrap_or_else(|| normalize_lexical(&effective_cwd))
}

fn project_root_dir() -> PathBuf {
    normalize_lexical(
        &crate::ai::driver::runtime_ctx::effective_cwd().unwrap_or_else(|_| PathBuf::from(".")),
    )
}

fn path_is_in_project(path: &Path, project_root: &Path) -> bool {
    path.starts_with(project_root)
}

pub(super) fn project_instruction_target_paths_from_tool_calls(
    tool_calls: &[crate::ai::types::ToolCall],
    include_read_only: bool,
) -> Vec<PathBuf> {
    let mut targets = Vec::new();
    let mut seen = FastSet::default();
    for tool_call in tool_calls {
        let supported = matches!(
            tool_call.function.name.as_str(),
            "write_file" | "apply_patch"
        ) || tool_call.function.name == "execute_command"
            || (include_read_only && tool_call.function.name == "read_file");
        if !supported {
            continue;
        }
        let Ok(args) = serde_json::from_str::<Value>(&tool_call.function.arguments) else {
            continue;
        };
        if let Some(path) = args
            .get("file_path")
            .or_else(|| args.get("path"))
            .and_then(Value::as_str)
        {
            push_project_target(&mut targets, &mut seen, path);
        }
        if tool_call.function.name == "execute_command" {
            let base = command_base_dir(&args);
            let project_root = project_root_dir();
            if let Some(command) = args.get("command").and_then(Value::as_str) {
                let analysis = analyze_execute_command(command, &base);
                for path in analysis
                    .mutation_targets
                    .into_iter()
                    .chain(analysis.unknown_mutation_bases)
                    .filter(|path| path_is_in_project(path, &project_root))
                {
                    push_project_target(&mut targets, &mut seen, &path.to_string_lossy());
                }
            }
        }
        if tool_call.function.name == "apply_patch"
            && let Some(patch) = args.get("patch").and_then(Value::as_str)
        {
            for path in crate::ai::tools::apply_patch_target_paths_from_patch(patch) {
                push_project_target(&mut targets, &mut seen, &path.to_string_lossy());
            }
        }
    }
    targets
}

fn project_instruction_target_paths(messages: &[Message]) -> Vec<PathBuf> {
    // Start the turn at the last **real** user message: runtime-synthetic user messages
    // (evidence handoff, image followup) do not form turn boundaries, otherwise the scoped-instruction
    // target would wrongly start after the synthetic message and the target directory's AGENTS.md would drop out of the system prompt.
    let current_turn_start = last_real_user_index(messages).unwrap_or(0);
    let mut targets = Vec::new();
    let mut seen = FastSet::default();
    for message in messages.iter().skip(current_turn_start) {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for target in project_instruction_target_paths_from_tool_calls(tool_calls, true) {
            push_project_target(&mut targets, &mut seen, &target.to_string_lossy());
        }
    }
    targets
}

pub(super) fn refresh_skill_turn_for_iteration(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    question: &str,
    iteration: usize,
    skill_turn: &mut super::super::skill_runtime::SkillTurnGuard,
    required_project_targets: &[PathBuf],
    messages: &mut [Message],
) -> bool {
    if iteration <= 1 {
        return required_project_targets.is_empty();
    }

    let prev_skills = skill_turn.matched_skill_names().to_vec();

    // Explicit change requests from the model via the activate_skill / deactivate_skill tools take priority:
    // force-activate by name and skip automatic routing scores. The tool side already validates the name (it must really exist);
    // here skill_manifests is checked once more, falling back to automatic routing on a miss.
    // Multiple skills: a pending action can be Add (append) or Remove (drop); multiple calls within the
    // same turn are all applied in order (queue semantics, no last-write-wins) against the current active set.
    use crate::ai::tools::skill_tools::PendingSkillAction;
    let actions = crate::ai::tools::skill_tools::take_pending_skill_action();
    let mut current_names = prev_skills.clone();
    for action in &actions {
        match action {
            PendingSkillAction::Add(name) => {
                if !current_names.iter().any(|n| n == name) {
                    current_names.push(name.clone());
                }
            }
            PendingSkillAction::Remove(name) => {
                current_names.retain(|n| n != name);
            }
        }
    }
    // When nothing changed, reuse the existing guard (skip the full rebuild): no pending action, skill set
    // untouched, and the scoped project instructions required by preflight plus those for files touched this turn are all in place. A rebuild costs more
    // than re-rendering: it re-pulls the MCP toolset, re-reads the SQLite activation history, and swaps out ctx.tools wholesale, so
    // the upstream prompt cache invalidates repeatedly across a long turn. If an observed target added a file carrying instructions
    // that are not yet in the current prompt, the rebuild path still runs to pick them up.
    if actions.is_empty() && current_names == prev_skills {
        let project_targets = project_instruction_target_paths(messages);
        if !skill_runtime::scoped_project_instructions_missing(
            skill_turn.system_prompt(),
            required_project_targets,
        ) && !skill_runtime::scoped_project_instructions_missing(
            skill_turn.system_prompt(),
            &project_targets,
        ) {
            return true;
        }
    }
    let inherited_restore = skill_turn.take_restore_agent_context();
    let mut new_skill_turn = if !actions.is_empty() {
        skill_runtime::force_activate_named_skill(
            app,
            mcp_client,
            skill_manifests,
            question,
            &current_names,
        )
        .unwrap_or_else(|| {
            skill_runtime::rebuild_skill_turn_with_existing_selection(
                app,
                mcp_client,
                skill_manifests,
                question,
                &current_names,
            )
        })
    } else {
        skill_runtime::rebuild_skill_turn_with_existing_selection(
            app,
            mcp_client,
            skill_manifests,
            question,
            &current_names,
        )
    };
    let project_targets = project_instruction_target_paths(messages);
    let scoped_project_instructions_ready = new_skill_turn
        .push_scoped_project_instructions(required_project_targets, &project_targets);
    if inherited_restore.is_some() {
        new_skill_turn.set_restore_agent_context(inherited_restore);
    }
    let next_skills = new_skill_turn.matched_skill_names().to_vec();

    if prev_skills != next_skills {
        if next_skills.is_empty() {
            println!("[skill switched: <none>]");
        } else {
            println!("[skill switched: {}]", next_skills.join(", ").cyan());
        }
    }

    *skill_turn = new_skill_turn;
    if let Some(system_message) = messages.first_mut() {
        // Overwrite only when the new system prompt text differs from the old.
        // Overwriting with the same string is not just useless: it repeatedly invalidates the upstream
        // prompt cache (e.g. anthropic cache_control hits, or the driver's internal string hash reuse),
        // silently wasting tokens in long multi-iteration turns.
        let next_prompt = skill_turn.system_prompt();
        let same = matches!(&system_message.content, Value::String(s) if s == next_prompt);
        if !same {
            system_message.content = Value::String(next_prompt.to_string());
        }
    }
    scoped_project_instructions_ready
}

fn continue_or_quit(should_quit: bool) -> TurnOutcome {
    if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    }
}

fn interrupted_iteration_execution(
    app: &mut App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
    should_quit: bool,
) -> IterationExecution {
    IterationExecution::Exit(finish_interrupted_turn(
        app,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
        should_quit,
    ))
}

fn shutdown_iteration_execution(
    app: &App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> IterationExecution {
    IterationExecution::Exit(finish_shutdown_turn(
        app,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
    ))
}

fn finish_interrupted_turn(
    app: &mut App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
    should_quit: bool,
) -> TurnOutcome {
    app.streaming
        .store(false, std::sync::atomic::Ordering::Relaxed);
    // Consume only interrupts triggered by cancel_stream for this turn, avoiding accidental clearing of
    // global interrupt bits from other sources (e.g. shutdown / request-level interrupt).
    let _ = crate::ai::types::take_stream_cancelled(app);
    app.ignore_next_prompt_interrupt = true;
    // Mark this turn as interrupted: run_loop's goal-continuation logic uses this to tell "interrupted"
    // apart from "finished naturally"; on interruption goal_mode is kept and we fall back to waiting for user input instead of falsely reporting "Goal achieved".
    app.last_turn_interrupted = true;
    persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);
    println!("\nInterrupted.");
    continue_or_quit(should_quit)
}

fn finish_shutdown_turn(
    app: &App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> TurnOutcome {
    persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);
    println!();
    TurnOutcome::Quit
}

fn handle_request_error(
    app: &App,
    err: request::RequestError,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) -> String {
    app.streaming
        .store(false, std::sync::atomic::Ordering::Relaxed);
    persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);
    let err_text = err.to_string();
    if request::is_transient_error(&err) {
        eprintln!("[Warning] {}", err_text);
    } else {
        eprintln!("[Error] {}", err_text);
    }
    if err_text.contains("function.arguments") && err_text.contains("must be in JSON format") {
        eprintln!("[Info] Model returned invalid tool arguments; skipped this round and continued to the next round.");
    } else {
        eprintln!("[Info] Request failed this round; session kept alive, you can keep asking.");
    }
    "[Request failed this round; please retry or rephrase]".to_string()
}

fn request_interrupt_pending(shutdown: &AtomicBool, cancel_stream: &AtomicBool) -> bool {
    shutdown.load(std::sync::atomic::Ordering::Relaxed)
        || cancel_stream.load(std::sync::atomic::Ordering::Relaxed)
}

fn request_interrupt_futex_ready() -> bool {
    crate::ai::driver::signal::request_interrupt_ready()
}

/// Why a request wait was interrupted.
enum RequestInterruptKind {
    /// User-initiated cancel (Ctrl+C / shutdown / cancel_stream).
    User,
    /// Parent agent pre-timeout wrap-up signal: abandon the current request and enter a forced wrap-up iteration immediately.
    WrapUp,
}

async fn wait_for_request_interrupt(
    shutdown: Arc<AtomicBool>,
    cancel_stream: Arc<AtomicBool>,
) -> RequestInterruptKind {
    let notify = crate::ai::driver::signal::request_interrupt_notify();
    loop {
        if request_interrupt_pending(shutdown.as_ref(), cancel_stream.as_ref())
            || request_interrupt_futex_ready()
        {
            return RequestInterruptKind::User;
        }
        // Pre-timeout wrap-up signal (task-local; does not touch global interrupt state and does not disturb parallel background turns).
        if crate::ai::driver::runtime_ctx::has_subagent_wrap_up_pending() {
            return RequestInterruptKind::WrapUp;
        }
        // Re-check after registering the wait future to avoid a race between signal_request_interrupt and registration.
        let notified = notify.notified();
        if request_interrupt_pending(shutdown.as_ref(), cancel_stream.as_ref())
            || request_interrupt_futex_ready()
        {
            return RequestInterruptKind::User;
        }
        if crate::ai::driver::runtime_ctx::has_subagent_wrap_up_pending() {
            return RequestInterruptKind::WrapUp;
        }
        // 50ms fallback to accommodate external futex wakeups (not going through the Notify channel).
        tokio::select! {
            _ = notified => {}
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
    }
}

pub(in crate::ai::driver::turn_runtime) fn no_tool_handoff_note() -> &'static str {
    "Enter no-tool wrap-up mode: do not call any tools for the rest of this turn.\n\
Produce a wrap-up/handoff reply based on the information already gathered:\n\
1. First summarize confirmed facts and the current conclusion;\n\
2. Directly answer the parts of the user's question you can answer;\n\
3. If the task is not yet complete, clearly state the remaining work, blockers, and suggested next steps;\n\
4. Do not dress up an unfinished task as completed.\n\
5. Wrap-up does not authorize fabrication: this note only asks you to stop calling tools; it never authorizes guessing or making things up.\
Identifiers, paths, command output, line numbers, or quotes not confirmed by a tool or the source code must not be stated as fact;\
for uncertain parts, label them honestly as \"unverified\" and give the next verification step — an honest \"unverified\" is always better than a fabricated complete answer."
}

fn clear_outstanding_task_anchor(messages: &mut Vec<Message>) {
    let prefix = task_tools::outstanding_task_anchor_prefix();
    messages.retain(|message| {
        !(message.role == ROLE_INTERNAL_NOTE
            && matches!(&message.content, Value::String(text) if text.starts_with(prefix)))
    });
}

fn refresh_outstanding_task_anchor(messages: &mut Vec<Message>, session_id: &str) {
    clear_outstanding_task_anchor(messages);
    let Ok(Some(note)) = task_tools::build_outstanding_task_anchor(session_id) else {
        return;
    };
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// Reactive context shrink: only called after the provider rejects a request
/// for exceeding the context window.
///
/// Proactive compression (`apply_pre_request_context_budget` + LLM summary)
/// already tries to bring the context near the soft threshold before the
/// request, but a char estimate is not an authoritative judge of tokens — a
/// request heavy in English text, images, or tool-schema overhead can still be
/// judged over-limit by the provider's tokenizer. Instead of failing locally
/// on a char threshold (which would kill legitimate requests), send the
/// request out and shrink only after a real rejection.
///
/// Each call cuts the target budget by another 25% from the previous one (a
/// floor keeps it from reaching 0), reusing the cross-turn compression
/// pipeline [`mid_turn_compress`](crate::ai::history::mid_turn_compress)
/// (including Path C emergency truncation) to force convergence. Returns the
/// char count after compression. Compression policies never truncate user
/// messages, so once they hit their floor the last resort is offloading the
/// middle of the current user message to the overflow archive
/// ([`truncate_last_real_user_message_to_fit`](crate::ai::history::truncate_last_real_user_message_to_fit));
/// if even that makes no progress, the returned count does not drop and the
/// caller stops retrying.
fn reactive_shrink_context_after_overflow(
    app: &App,
    messages: &mut Vec<Message>,
    target_chars: usize,
) -> usize {
    let overflow_dir = {
        use crate::ai::history::SessionStore;
        let store = SessionStore::new(app.config.history_file.as_path());
        store.session_assets_dir(&app.session_id)
    };
    let drained = std::mem::take(messages);
    let (compressed, before, after) =
        crate::ai::history::mid_turn_compress(
            drained,
            target_chars,
            Some(overflow_dir.as_path()),
            crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
        );
    *messages = compressed;
    // A dead end here usually means the oversized body is the current user
    // message itself: compression policies never truncate user messages. As a
    // last resort, offload its middle to the overflow archive so the retry can
    // converge instead of failing the turn outright. The rescue also refuses
    // when the archive write fails — an unarchived preview of the current
    // instruction would be unrecoverable — so the unshrunk total falls through
    // to the caller's give-up path and the provider error surfaces.
    if after >= before
        && crate::ai::history::truncate_last_real_user_message_to_fit(
            messages,
            target_chars,
            Some(overflow_dir.as_path()),
        )
    {
        return crate::ai::history::messages_total_chars_pub(messages);
    }
    after
}

/// Update the temporary context projection at every model request boundary without touching canonical `turn_messages`.
fn apply_model_guided_pruning_before_request(app: &App, messages: &mut Vec<Message>) {
    let overflow_dir = {
        use crate::ai::history::SessionStore;
        let store = SessionStore::new(app.config.history_file.as_path());
        store.session_assets_dir(&app.session_id)
    };
    let candidate_count = llm_prune::active_prunable_tool_ids(messages).len();
    let had_protocol = messages
        .iter()
        .any(|message| message.content.as_str() == Some(llm_prune::PRUNE_PROTOCOL_PROMPT));
    let report = llm_prune::prepare_request_projection(
        messages,
        &app.prune_marks,
        Some(overflow_dir.as_path()),
    );
    let protocol_injected = !had_protocol
        && messages
            .iter()
            .any(|message| message.content.as_str() == Some(llm_prune::PRUNE_PROTOCOL_PROMPT));
    if protocol_injected && crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        crate::ai::driver::print::print_tool_note_line(
            "context-prune",
            &format!("model pruning enabled for {candidate_count} old tool result(s)"),
        );
    }
    if report.pruned_count == 0 {
        return;
    }

    let tools = if report.tools.is_empty() {
        String::new()
    } else {
        format!(" [{}]", report.tools.join(", "))
    };
    crate::ai::driver::print::print_tool_note_line(
        "context-pruned",
        &format!(
            "{} tool result(s){}, ~{} chars freed",
            report.pruned_count, tools, report.freed_chars
        ),
    );
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "B",
    "turn_runtime::run_turn:do_request_messages",
    "[DEBUG] sending model request",
    "[DEBUG] model request finished",
    {
        "iteration": _iteration,
        "message_count": messages.len(),
        "model": next_model,
    },
    {
        "iteration": _iteration,
        "ok": __agent_hang_result.is_ok(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
/// Build the LLM client chain used for real requests: an empty chain by default = go straight to `DefaultLlmClient` (zero behavior change);
/// with `RequestMiddleware` registered, wrap in registration order (retries/short-circuits/auditing take effect on production requests).
fn build_llm_request_client(app: &App) -> Box<dyn LlmClient> {
    build_llm_client_chain(app.llm_middlewares.clone(), Box::new(DefaultLlmClient))
}

/// Send one LLM request through the client chain and normalize errors back to `RequestError`:
/// the default chain passes through unchanged (downcast succeeds; fallback/over-limit classification behavior is unchanged);
/// non-`RequestError` errors produced by custom middleware count as local policy failures and do not trigger model fallback.
async fn send_llm_request(
    client: &dyn LlmClient,
    app: &mut App,
    model: &str,
    messages: &mut Vec<Message>,
    tools_enabled: bool,
) -> Result<reqwest::Response, request::RequestError> {
    let request = LlmRequest {
        model: model.to_string(),
        messages: messages.clone(),
        stream: true,
        tools_enabled,
    };
    match client.send(app, request).await {
        Ok(response) => Ok(response.response),
        Err(err) => match err.downcast::<request::RequestError>() {
            Ok(inner) => Err(*inner),
            Err(opaque) => Err(request::RequestError::status(
                reqwest::StatusCode::UNPROCESSABLE_ENTITY,
                opaque.to_string(),
            )),
        },
    }
}

async fn request_model_response(
    app: &mut App,
    next_model: &str,
    messages: &mut Vec<Message>,
    force_final_response: bool,
    _iteration: usize,
    mut compression_report: CompressionReport,
) -> Result<(reqwest::Response, String), request::RequestError> {
    // Pre-request-build hooks (on_before_request → BuildRequest.before), fired before any app state mutation.
    // The request messages being built are passed in so hooks can inspect/rewrite them.
    app.fire_before_request_hooks(messages);
    if crate::ai::driver::runtime_ctx::take_subagent_checkpoint_due_reminder() {
        messages.push(Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(
                "[runtime checkpoint due] This subagent has run for at least one checkpoint interval without publishing durable progress. Before the next long operation, emit a concise <context_checkpoint> containing verified progress, evidence, and the next decision-relevant step."
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    if force_final_response {
        clear_outstanding_task_anchor(messages);
    } else {
        refresh_outstanding_task_anchor(messages, &app.session_id);
    }
    if force_final_response {
        messages.push(Message {
            role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(no_tool_handoff_note().to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    // Process before every request, not just at turn initialization, so later tool rounds within the same
    // turn can also consume prune markers that just crossed the threshold and offload losslessly before context compression.
    apply_model_guided_pruning_before_request(app, messages);

    let budget_report = context_budget::apply_pre_request_context_budget(app, next_model, messages);
    if let Some(reason) = budget_report.rollback_reason {
        crate::ai::driver::print::print_tool_note_line("context-budget", reason.note());
    } else if budget_report.changed {
        crate::ai::driver::print::print_tool_note_line(
            "context-budget",
            &format!(
                "compressed {} -> {} chars",
                budget_report.before_chars, budget_report.after_chars,
            ),
        );
    }

    // === Pre-request LLM summary fallback ===
    // When the context still far exceeds the threshold after lossless + lossy compression, call the LLM to squeeze the early conversation into a summary.
    // This is the last line of defense before sending the request, preventing oversized context from causing model 4xx or quality degradation
    // (the user-reported "295K compressed to 294K and then stalled" problem).
    // The threshold is history_max_chars * 2 (default 180K), more aggressive than the orchestrator's
    // hard threshold (*3.5 = 315K) — that one only fires between tool calls, while this covers the
    // final check before every request.
    // Growth guard: mid-turn and pre-request share the same LLM summary attempt cursor.
    // If the same context batch was just attempted with no effective growth, do not request a summary again.
    let llm_threshold = pre_request_llm_summary_threshold(next_model, app.config.history_max_chars);
    let session_id = app.session_id.clone();
    if should_try_llm_summary(&session_id, budget_report.after_chars, llm_threshold) {
        // Cancel safety: pass a **clone** of messages instead of `mem::take`. If this summary await
        // is interrupted by Ctrl+C, the request future is dropped while `messages` keeps its original
        // full content, so it cannot degrade into an empty Vec and send an empty context / lose message state on later requests.
        let (after_msgs, llm_before, llm_after, was_effective, llm_summary_inserted) =
            crate::ai::history::mid_turn_llm_summarize(
                app,
                messages.clone(),
                MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
                MID_TURN_LLM_SUMMARY_MAX_CHARS,
                app.config.history_max_chars,
                crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
            )
            .await;
        *messages = after_msgs;
        compression_report.record_llm_summary_attempt(
            format!("pre-request LLM (limit {llm_threshold})"),
            llm_before,
            llm_after,
            was_effective,
            llm_summary_inserted,
        );
        record_llm_summary_attempt_chars(&session_id, llm_after);
    }
    // The summary pipeline keeps system messages in principle; this idempotent safeguard still runs to guarantee the request-boundary protocol exists.
    llm_prune::ensure_prune_protocol_prompt(messages);
    compression_report.emit();

    let auto_model_fallback_spec = crate::ai::driver::runtime_ctx::auto_model_fallback_spec();
    if crate::ai::driver::runtime_ctx::terminal_output_enabled()
        && auto_model_fallback_spec.is_some()
        && crate::ai::models::subagent_model_needs_probe(next_model)
    {
        eprintln!(
            "[model] validating auto-selected model '{}' with the first real subagent request",
            next_model
        );
    }

    // Reactive context-over-limit retry: proactive compression has done its best to squeeze the context near the soft threshold, but character
    // estimation is not the authoritative judge — the provider tokenizer is. If the request is still judged over the context limit, shrink in place and retry,
    // instead of locally raising 413 and killing a legitimate request or shoving the oversized payload at the provider and giving up.
    // Over-limit errors do not trigger model fallback (`should_try_model_fallback` already excludes 400/413);
    // the two are mutually exclusive.
    const MAX_CONTEXT_OVERFLOW_RETRIES: usize = 4;
    let mut overflow_retries = 0usize;
    let llm_client = build_llm_request_client(app);
    loop {
        let mut actual_model = next_model.to_string();
        let mut request_result = if force_final_response {
            send_llm_request(&*llm_client, app, next_model, messages, false).await
        } else {
            send_llm_request(&*llm_client, app, next_model, messages, true).await
        };
        if let Err(err) = &request_result
            && let Some(fallback_spec) = auto_model_fallback_spec
            && request::should_try_model_fallback(err)
        {
            if request::should_temporarily_disable_auto_selected_model(err) {
                crate::ai::models::mark_model_temporarily_unavailable(next_model, &err.to_string());
            }
            if let Some(fallback_model) =
                crate::ai::models::fallback_subagent_model_after_failure(next_model, fallback_spec)
            {
                if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                    eprintln!(
                        "[model] auto-selected model '{}' failed; retrying subagent with '{}'",
                        next_model, fallback_model
                    );
                }
                actual_model = fallback_model.clone();
                request_result = if force_final_response {
                    send_llm_request(&*llm_client, app, &fallback_model, messages, false).await
                } else {
                    send_llm_request(&*llm_client, app, &fallback_model, messages, true).await
                };
                if let Err(fallback_err) = &request_result
                    && request::should_temporarily_disable_auto_selected_model(fallback_err)
                {
                    crate::ai::models::mark_model_temporarily_unavailable(
                        &fallback_model,
                        &fallback_err.to_string(),
                    );
                }
            }
        }

        if let Err(err) = &request_result
            && request::is_context_overflow_error(err)
            && overflow_retries < MAX_CONTEXT_OVERFLOW_RETRIES
        {
            let before = crate::ai::history::messages_total_chars_pub(messages);
            // Cut the target by another 25% of the current size; the floor
            // keeps it away from 0 so the shrink never becomes a no-op.
            let target = before
                .saturating_mul(3)
                .saturating_div(4)
                .max(MID_TURN_COMPRESS_SOFT_FLOOR);
            let after = reactive_shrink_context_after_overflow(app, messages, target);
            overflow_retries += 1;
            if after < before {
                crate::ai::driver::print::print_tool_note_line(
                    "context-overflow",
                    &format!(
                        "provider rejected oversized context; compressed {before} → {after} chars, retrying"
                    ),
                );
                continue;
            }
            // No progress even after the current-user-message rescue: retrying
            // would only be rejected the same way again.
            crate::ai::driver::print::print_tool_note_line(
                "context-overflow",
                "provider rejected context but it cannot be compressed further",
            );
        }

        return request_result.map(|response| {
            if auto_model_fallback_spec.is_some() {
                crate::ai::models::mark_subagent_model_verified(&actual_model);
            }
            (response, actual_model)
        });
    }
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "B",
    "turn_runtime::run_turn:stream_response",
    "[DEBUG] streaming response started",
    "[DEBUG] streaming response finished",
    {
        "iteration": _iteration,
    },
    {
        "iteration": _iteration,
        "ok": __agent_hang_result.is_ok(),
        "outcome": format!("{:?}", __agent_hang_result.as_ref().ok().map(|r| r.outcome)),
        "assistant_chars": __agent_hang_result.as_ref().map(|r| r.assistant_text.chars().count()).unwrap_or(0),
        "tool_calls": __agent_hang_result.as_ref().map(|r| r.tool_calls.len()).unwrap_or(0),
        "history_chars": current_history.chars().count(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
async fn stream_model_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
    terminal_dedupe_candidate: Option<&str>,
    _active_skill_name: Option<&str>,
    _iteration: usize,
) -> Result<StreamResult, String> {
    match stream::stream_response(app, response, current_history, terminal_dedupe_candidate).await {
        Ok(result) => Ok(result),
        Err(err) => {
            app.streaming
                .store(false, std::sync::atomic::Ordering::Relaxed);
            Err(err.to_string())
        }
    }
}

async fn finalize_stream_interaction(
    app: &mut App,
    response: &mut reqwest::Response,
    stream_result: StreamResult,
    turn_messages: &[Message],
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    should_quit: bool,
    _force_final_response: bool,
) -> Result<IterationExecution, Box<dyn std::error::Error>> {
    input::clear_stdin_buffer();

    if stream_result.outcome == StreamOutcome::Cancelled {
        return Ok(interrupted_iteration_execution(
            app,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
            should_quit,
        ));
    }
    if app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(shutdown_iteration_execution(
            app,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
        ));
    }

    if !stream_result.skip_response_drain {
        // Parse-error fallback may still leave bytes buffered. Keep this bounded
        // so unusual provider behavior cannot hang the turn.
        match tokio::time::timeout(Duration::from_millis(200), drain_response(response)).await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(err),
            Err(_) => {
                if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                    app.streaming
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    return Ok(match stream_result.outcome {
                        StreamOutcome::ToolCall => {
                            IterationExecution::ToolCall(ToolCallExecution {
                                stream_result,
                                allowed_tool_names: request_visible_tool_names(app),
                            })
                        }
                        StreamOutcome::EmptyResponse => IterationExecution::EmptyResponse,
                        StreamOutcome::Truncated => IterationExecution::Truncated(stream_result),
                        _ => IterationExecution::FinalResponse(stream_result),
                    });
                }
                eprintln!("[Warning] 响应流收尾 drain 超时，已跳过剩余字节读取以避免会话卡住。");
            }
        }
    }
    app.streaming
        .store(false, std::sync::atomic::Ordering::Relaxed);

    Ok(match stream_result.outcome {
        StreamOutcome::ToolCall => IterationExecution::ToolCall(ToolCallExecution {
            stream_result,
            allowed_tool_names: request_visible_tool_names(app),
        }),
        StreamOutcome::EmptyResponse => IterationExecution::EmptyResponse,
        StreamOutcome::Truncated => IterationExecution::Truncated(stream_result),
        _ => IterationExecution::FinalResponse(stream_result),
    })
}

pub(super) async fn execute_turn_iteration(
    app: &mut App,
    next_model: &str,
    response_model: &mut Option<String>,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    should_quit: bool,
    force_final_response: bool,
    terminal_dedupe_candidate: Option<&str>,
    active_skill_name: Option<&str>,
    iteration: usize,
    compression_report: CompressionReport,
) -> Result<IterationExecution, Box<dyn std::error::Error>> {
    let mut current_history = String::new();
    request::clear_stale_request_interrupt_before_request(app);
    let _streaming_guard = StreamingFlagGuard::new(&app.streaming);
    crate::ai::driver::runtime_ctx::publish_subagent_phase("calling model");

    let shutdown = app.shutdown.clone();
    let cancel_stream = app.cancel_stream.clone();
    let request_result = tokio::select! {
        response = request_model_response(
            app,
            next_model,
            messages,
            force_final_response,
            iteration,
            compression_report,
        ) => response,
        interrupt_kind = wait_for_request_interrupt(shutdown.clone(), cancel_stream.clone()) => {
            match interrupt_kind {
                RequestInterruptKind::User => {
                    return Ok(interrupted_iteration_execution(
                        app,
                        one_shot_mode,
                        turn_messages,
                        persisted_turn_messages,
                        should_quit,
                    ));
                }
                // Pre-timeout wrap-up: abandon the current request; the orchestrator enters a forced wrap-up iteration immediately.
                RequestInterruptKind::WrapUp => {
                    return Ok(IterationExecution::WrapUpFinal);
                }
            }
        }
    };

    let (mut response, actual_model) = match request_result {
        Ok(response) => response,
        Err(err) => {
            let err_text = err.to_string();
            if crate::ai::driver::runtime_ctx::has_subagent_result_slot() {
                return Err(err_text.into());
            }
            return Ok(IterationExecution::RequestFailed(handle_request_error(
                app,
                err,
                one_shot_mode,
                turn_messages,
                persisted_turn_messages,
            )));
        }
    };
    // After automatic fallback, the model that actually completed this response must still be passed to
    // the message projection and canonical persistence layers; the pre-routing next_model / app.current_model cannot be reused.
    *response_model = Some(actual_model.clone());

    if app
        .cancel_stream
        .swap(false, std::sync::atomic::Ordering::Relaxed)
    {
        return Ok(interrupted_iteration_execution(
            app,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
            should_quit,
        ));
    }

    // A streaming response can break mid-stream on transient backend errors (e.g. "Cancelled by backend");
    // retry the whole request+stream for such retryable errors instead of abandoning the entire turn.
    const MAX_STREAM_RETRIES: usize = 16;
    let mut stream_attempt = 0usize;
    loop {
        if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
            request::print_info(app, &actual_model);
        }
        match stream_model_response(
            app,
            &mut response,
            &mut current_history,
            terminal_dedupe_candidate,
            active_skill_name,
            iteration,
        )
        .await
        {
            Ok(stream_result) => {
                // Wrap-up hook on the successful stream-parse path (on_after_stream → ParseStream.after).
                app.fire_after_stream_hooks();
                return finalize_stream_interaction(
                    app,
                    &mut response,
                    stream_result,
                    turn_messages,
                    one_shot_mode,
                    persisted_turn_messages,
                    should_quit,
                    force_final_response,
                )
                .await;
            }
            Err(err_msg) => {
                if stream_attempt < MAX_STREAM_RETRIES
                    && request::is_retryable_stream_error(&err_msg)
                {
                    stream_attempt += 1;
                    if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                        eprintln!(
                            "\n[Info] 流式响应中断（{}），第 {}/{} 次重试...",
                            err_msg, stream_attempt, MAX_STREAM_RETRIES
                        );
                    }
                    current_history.clear();
                    app.streaming
                        .store(true, std::sync::atomic::Ordering::Relaxed);

                    if request::should_abort_retry_wait(app) {
                        return Ok(interrupted_iteration_execution(
                            app,
                            one_shot_mode,
                            turn_messages,
                            persisted_turn_messages,
                            should_quit,
                        ));
                    }
                    if request::sleep_with_cancel(app, request::retry_delay(stream_attempt)).await {
                        return Ok(interrupted_iteration_execution(
                            app,
                            one_shot_mode,
                            turn_messages,
                            persisted_turn_messages,
                            should_quit,
                        ));
                    }

                    request::clear_stale_request_interrupt_before_request(app);
        let llm_client = build_llm_request_client(app);
                    let retry_request = if force_final_response {
            send_llm_request(&*llm_client, app, &actual_model, messages, false).await
                    } else {
            send_llm_request(&*llm_client, app, &actual_model, messages, true).await
                    };
                    match retry_request {
                        Ok(new_response) => {
                            response = new_response;
                        }
                        Err(retry_err) => {
                            let err_text = retry_err.to_string();
                            if crate::ai::driver::runtime_ctx::has_subagent_result_slot() {
                                return Err(err_text.into());
                            }
                            return Ok(IterationExecution::RequestFailed(handle_request_error(
                                app,
                                retry_err,
                                one_shot_mode,
                                turn_messages,
                                persisted_turn_messages,
                            )));
                        }
                    }
                    continue;
                }

                // Not retryable or retries exhausted — fall back to the old behavior and keep the conversation going
                if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                    eprintln!("\n[Error] 流式响应处理失败：{}", err_msg);
                    eprintln!("[Info] 尝试继续对话...");
                }
                let stream_result = StreamResult {
                    outcome: StreamOutcome::Completed,
                    tool_calls: Vec::new(),
                    assistant_text: "[Response parsing failed; please retry]".to_string(),
                    hidden_meta: String::new(),
                    reasoning_text: String::new(),
                    reasoning_items: Vec::new(),
                    skip_response_drain: false,
                    truncated_by_length: false,
                    stream_error: false,
                    finish_reason_value: None,
                    usage_prompt_tokens: 0,
                    usage_cached_prompt_tokens: 0,
                    usage_completion_tokens: 0,
                    usage_reasoning_tokens: 0,
                };
                return finalize_stream_interaction(
                    app,
                    &mut response,
                    stream_result,
                    turn_messages,
                    one_shot_mode,
                    persisted_turn_messages,
                    should_quit,
                    force_final_response,
                )
                .await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{record_llm_summary_attempt_chars, should_try_llm_summary};
    use super::{
        App, LlmClient, LlmRequest, StreamingFlagGuard, build_llm_request_client,
        no_tool_handoff_note, project_instruction_target_paths, refresh_outstanding_task_anchor,
        request, request_interrupt_pending, send_llm_request,
    };
    use crate::ai::history::{Message, ROLE_INTERNAL_NOTE};
    use crate::ai::middleware::RequestMiddleware;
    use crate::ai::ports::LlmResponse;
    use crate::ai::types::{FunctionCall, ToolCall};
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, atomic::Ordering};

    // ------------------------------------------------------------------
    // The RequestMiddleware chain is wired into the real request path (regression guard for the P2 fix):
    // middleware registered in `app.llm_middlewares` must take effect when production requests build the chain
    // (build_llm_request_client) and send (send_llm_request).
    // ------------------------------------------------------------------

    struct CountingRequestMiddleware {
        calls: Arc<AtomicUsize>,
    }

    impl RequestMiddleware for CountingRequestMiddleware {
        fn name(&self) -> &'static str {
            "counting"
        }
        fn wrap(&self, inner: Box<dyn LlmClient>) -> Box<dyn LlmClient> {
            struct Wrapper {
                inner: Box<dyn LlmClient>,
                calls: Arc<AtomicUsize>,
            }
            impl LlmClient for Wrapper {
                fn send<'a>(
                    &'a self,
                    app: &'a mut App,
                    req: LlmRequest,
                ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>
                {
                    self.calls.fetch_add(1, Ordering::SeqCst);
                    self.inner.send(app, req)
                }
            }
            Box::new(Wrapper {
                inner,
                calls: Arc::clone(&self.calls),
            })
        }
    }

    struct ShortCircuitRequestMiddleware;

    impl RequestMiddleware for ShortCircuitRequestMiddleware {
        fn name(&self) -> &'static str {
            "short-circuit"
        }
        fn wrap(&self, inner: Box<dyn LlmClient>) -> Box<dyn LlmClient> {
            struct Wrapper {
                _inner: Box<dyn LlmClient>,
            }
            impl LlmClient for Wrapper {
                fn send<'a>(
                    &'a self,
                    _app: &'a mut App,
                    _req: LlmRequest,
                ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>
                {
                    Box::pin(async {
                        Err(Box::new(request::RequestError::cancelled(
                            "short-circuited by test middleware".to_string(),
                        )) as Box<dyn std::error::Error + Send + Sync>)
                    })
                }
            }
            Box::new(Wrapper { _inner: inner })
        }
    }

    #[tokio::test]
    async fn llm_middlewares_chain_used_by_real_request_path() {
        let mut app = crate::ai::middleware::test_util::test_app();
        let calls = Arc::new(AtomicUsize::new(0));
        app.llm_middlewares.push(Arc::new(CountingRequestMiddleware {
            calls: Arc::clone(&calls),
        }));

        // The chain must come from app.llm_middlewares; with an empty endpoint the inner DefaultLlmClient
        // request is guaranteed to fail, but the middleware send is definitely invoked (proving the production path really goes through the chain).
        let client = build_llm_request_client(&app);
        let mut messages = Vec::new();
        let _ = send_llm_request(&*client, &mut app, "gpt-4o", &mut messages, true).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn llm_middleware_short_circuit_error_surfaces_as_request_error() {
        let mut app = crate::ai::middleware::test_util::test_app();
        app.llm_middlewares.push(Arc::new(ShortCircuitRequestMiddleware));

        let client = build_llm_request_client(&app);
        let mut messages = Vec::new();
        let err = send_llm_request(&*client, &mut app, "gpt-4o", &mut messages, true)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("short-circuited by test middleware"));
    }

    #[test]
    fn streaming_flag_guard_resets_on_drop() {
        let streaming = Arc::new(AtomicBool::new(false));
        {
            let _guard = StreamingFlagGuard::new(&streaming);
            assert!(streaming.load(Ordering::Relaxed));
        }
        assert!(!streaming.load(Ordering::Relaxed));
    }

    #[test]
    fn request_interrupt_pending_tracks_shutdown_or_stream_cancel() {
        let shutdown = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(false);
        assert!(!request_interrupt_pending(&shutdown, &cancel_stream));

        cancel_stream.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(request_interrupt_pending(&shutdown, &cancel_stream));

        cancel_stream.store(false, std::sync::atomic::Ordering::Relaxed);
        shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(request_interrupt_pending(&shutdown, &cancel_stream));
    }

    #[test]
    fn no_tool_handoff_note_requires_summary_and_next_steps() {
        let note = no_tool_handoff_note();
        assert!(note.contains("do not call any tools"));
        assert!(note.contains("summarize confirmed facts and the current conclusion"));
        assert!(note.contains("remaining work, blockers, and suggested next steps"));
        assert!(note.contains("Do not dress up an unfinished task as completed"));
        assert!(note.contains("does not authorize fabrication"));
    }

    #[test]
    fn refresh_outstanding_task_anchor_replaces_stale_anchor_note() {
        let mut messages = vec![Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!(
                "{}\nstale",
                crate::ai::tools::task_tools::outstanding_task_anchor_prefix()
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        refresh_outstanding_task_anchor(&mut messages, "session-without-tasks");

        assert!(messages.is_empty());
    }

    #[test]
    fn project_instruction_targets_follow_current_turn_file_tools() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Value::String("old turn".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: "old".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: r#"{"file_path":"old.rs"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("current turn".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![
                    ToolCall {
                        id: "read".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "read_file".to_string(),
                            arguments: r#"{"file_path":"src/bin/ai/driver/mod.rs"}"#.to_string(),
                        },
                    },
                    ToolCall {
                        id: "patch".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "apply_patch".to_string(),
                            arguments: serde_json::json!({
                                "patch": "*** Begin Patch\n*** Update File: src/bin/ai/agents.rs\n@@\n-old\n+new\n*** Add File: src/bin/ai/new.rs\n+new\n*** End Patch"
                            })
                            .to_string(),
                        },
                    },
                    ToolCall {
                        id: "git-header-patch".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "apply_patch".to_string(),
                            arguments: serde_json::json!({
                                "patch": "diff --git a/src/bin/ai/quoted-old.rs b/src/bin/ai/quoted-new.rs\n@@ -1 +1 @@\n-old\n+new\n"
                            })
                            .to_string(),
                        },
                    },
                ]),
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        assert_eq!(
            project_instruction_target_paths(&messages),
            vec![
                std::path::PathBuf::from("src/bin/ai/driver/mod.rs"),
                std::path::PathBuf::from("src/bin/ai/agents.rs"),
                std::path::PathBuf::from("src/bin/ai/new.rs"),
                std::path::PathBuf::from("src/bin/ai/quoted-new.rs"),
            ]
        );
    }

    #[test]
    fn execute_command_mutations_infer_project_instruction_targets() {
        let mut call = ToolCall {
            id: "command".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "execute_command".to_string(),
                arguments: r#"{"command":"python scripts/update.py","pty":false}"#.to_string(),
            },
        };
        let project_root = crate::ai::driver::runtime_ctx::effective_cwd().unwrap();
        assert_eq!(
            super::project_instruction_target_paths_from_tool_calls(
                std::slice::from_ref(&call),
                false
            ),
            vec![project_root.clone()]
        );

        call.function.arguments = serde_json::json!({
            "command": "touch src/bin/ai/driver/new.rs",
            "pty": false
        })
        .to_string();
        assert_eq!(
            super::project_instruction_target_paths_from_tool_calls(
                std::slice::from_ref(&call),
                false
            ),
            vec![project_root.join("src/bin/ai/driver/new.rs")]
        );

        call.function.arguments = serde_json::json!({
            "command": "cd src/bin/ai && python update.py",
            "pty": false
        })
        .to_string();
        assert_eq!(
            super::project_instruction_target_paths_from_tool_calls(
                std::slice::from_ref(&call),
                false
            ),
            vec![project_root.join("src/bin/ai")]
        );

        call.function.arguments = serde_json::json!({
            "command": "sed -i '' -e 's/old/new/' src/bin/ai/driver/mod.rs",
            "pty": false
        })
        .to_string();
        assert_eq!(
            super::project_instruction_target_paths_from_tool_calls(
                std::slice::from_ref(&call),
                false
            ),
            vec![project_root.join("src/bin/ai/driver/mod.rs")]
        );

        call.function.arguments = serde_json::json!({
            "command": "python update.py",
            "cwd": "src/bin/ai/driver",
            "pty": false
        })
        .to_string();
        assert_eq!(
            super::project_instruction_target_paths_from_tool_calls(
                std::slice::from_ref(&call),
                false
            ),
            vec![project_root.join("src/bin/ai/driver")]
        );

        call.function.arguments =
            r#"{"command":"printf ready > /tmp/agent-ready.log","pty":false}"#.to_string();
        assert!(
            super::project_instruction_target_paths_from_tool_calls(
                std::slice::from_ref(&call),
                false
            )
            .is_empty()
        );

        let args = serde_json::from_str::<Value>(&call.function.arguments).unwrap();
        let effects = super::execute_command_segment_effects_for_args(&args);
        assert!(effects.iter().any(|effect| effect.mutation));
        assert!(
            effects.iter().all(|effect| !effect.project_mutation),
            "写入系统临时目录不应被视为项目变更"
        );

        call.function.arguments =
            r#"{"command":"printf ready > .agent-ready.log","pty":false}"#.to_string();
        let args = serde_json::from_str::<Value>(&call.function.arguments).unwrap();
        assert!(
            super::execute_command_segment_effects_for_args(&args)
                .iter()
                .any(|effect| effect.project_mutation),
            "相对路径重定向仍应被视为项目变更"
        );

        for command in [
            "git -C /tmp/repo status",
            "cargo --manifest-path Cargo.toml check",
        ] {
            call.function.arguments =
                serde_json::json!({"command": command, "pty": false}).to_string();
            assert!(
                super::project_instruction_target_paths_from_tool_calls(
                    std::slice::from_ref(&call),
                    false
                )
                .is_empty(),
                "read-only command must not infer mutation targets: {command}"
            );
        }

        call.function.arguments =
            serde_json::json!({"command": "npm install", "pty": false}).to_string();
        assert_eq!(
            super::project_instruction_target_paths_from_tool_calls(
                std::slice::from_ref(&call),
                false
            ),
            vec![project_root]
        );

        call.function.arguments =
            r#"{"command":"git -c core.pager=cat --no-pager diff -- src","pty":false}"#.to_string();
        assert!(super::project_instruction_target_paths_from_tool_calls(&[call], false).is_empty());
    }

    #[test]
    fn bytedcli_remote_queries_are_not_project_mutations() {
        // bytedcli only performs remote API queries/operations, so it must not trip the completion evidence
        // gate's project-change judgment, nor derive project-instruction targets.
        for command in [
            "bytedcli codebase mr get -R \"byteapi/bytedcli\"",
            "bytedcli --json codebase commit list -R \"byteapi/bytedcli\" --revision master",
            "bytedcli codebase mr diff 42 -R \"byteapi/bytedcli\"",
        ] {
            let args = serde_json::json!({"command": command, "pty": false});
            let effects = super::execute_command_segment_effects_for_args(&args);
            assert!(!effects.is_empty(), "{command}");
            assert!(
                effects
                    .iter()
                    .all(|effect| !effect.mutation && !effect.project_mutation),
                "bytedcli 远端查询不应被视为项目变更: {command}"
            );
        }

        // `mr artifacts download` explicitly writes local files (--output-dir),
        // and must be recognized as a project change even without a shell redirection.
        let args = serde_json::json!({
            "command": "bytedcli codebase mr artifacts download -R \"byteapi/bytedcli\" 42 --output-dir ./artifacts",
            "pty": false
        });
        let effects = super::execute_command_segment_effects_for_args(&args);
        assert!(effects.iter().any(|effect| effect.mutation));
        assert!(
            effects.iter().any(|effect| effect.project_mutation),
            "bytedcli 显式输出到项目目录仍应视为项目变更"
        );
    }

    #[test]
    fn git_readonly_listing_queries_are_not_project_mutations() {
        // Regression: git read-only query subcommands (tag -l / worktree list / ls-remote / remote -v /
        // fetch) used to be treated as WriteIntended (git subcommands outside the whitelist were all treated as writes),
        // tripping the cwd fallback into a project_mutation misjudgment from the project cwd. Within the same turn this resets
        // successful_post_mutation_verification: e.g. in `git status --short &&
        // git worktree list && git tag -l`, the verification evidence from git status is reset by the
        // misjudged worktree list / tag -l afterwards, making the completion evidence gate falsely report "no post-change
        // verification observed" (session a48935d1 messages 169/170 got Warned for exactly this, forcing the conclusion to carry a bogus unverified warning).
        for command in [
            "git tag -l \"V2.78*\"",
            "git tag | grep -c foo",
            "git worktree list",
            "git ls-remote --tags origin",
            "git remote -v",
            "git fetch --tags",
            "git status --short && git worktree list && git tag -l",
            // With global options first, the subcommand is not at index 1; `tag`/`worktree` must take
            // the first argument after the subcommand instead of a hardcoded third token (the `-C` value is not an argument).
            "git -C . tag -l \"V2.78*\"",
            "git -C . worktree list",
            // Attached-value forms of the read-only query flags (`-n1`/`-ln`/`--sort=`/`--format=`).
            "git tag -n1",
            "git tag -ln \"v*\"",
            "git tag --sort=-creatordate",
            "git tag --format='%(refname:short)'",
        ] {
            let args = serde_json::json!({"command": command, "pty": false});
            let effects = super::execute_command_segment_effects_for_args(&args);
            assert!(!effects.is_empty(), "{command}");
            assert!(
                effects.iter().all(|effect| !effect.project_mutation),
                "git 只读查询不应被视为项目变更: {command}"
            );
        }

        // git forms that really write refs / the worktree must still be judged a project change.
        for command in [
            "git worktree add /tmp/wt HEAD",
            "git worktree remove /tmp/wt",
            "git tag -a v1.0 -m \"release\"",
            "git tag -d old-tag",
            "git commit -am \"x\"",
        ] {
            let args = serde_json::json!({"command": command, "pty": false});
            let effects = super::execute_command_segment_effects_for_args(&args);
            assert!(
                effects.iter().any(|effect| effect.project_mutation),
                "git 写操作应视为项目变更: {command}"
            );
        }
    }

    #[test]
    fn unknown_interpreter_readonly_checks_are_not_project_mutations() {
        // Regression: "unknown programs" such as python3/node/perl -e running read-only checks from the
        // project cwd used to be misjudged as project_mutation by the cwd fallback, tripping the completion evidence gate
        // (resetting successful_post_mutation_verification) and forcing the model to repeat its
        // conclusion (session 9cec82e3's last three turns 277/310/329 all tripped on this).
        // Fix: only programs "known to write local files" (WriteIntended or a known mutator)
        // may be judged a project change from the cwd alone; unknown programs require write evidence
        // (redirection / a resolvable in-project target) and cannot be classified by enumerating interpreter names (impossible to be exhaustive).
        for command in [
            "python3 -c \"import json; print('ok')\"",
            "node -e \"console.log('ok')\"",
            "perl -e 'print qq{ok\\n}'",
        ] {
            let args = serde_json::json!({"command": command, "pty": false});
            let effects = super::execute_command_segment_effects_for_args(&args);
            assert!(!effects.is_empty(), "{command}");
            assert!(
                effects.iter().all(|effect| !effect.project_mutation),
                "未知解释器的只读校验不应被视为项目变更: {command}"
            );
        }

        // Unknown interpreter + shell redirection writing a project file: still judged a project change based on write evidence.
        let args = serde_json::json!({
            "command": "python3 -c \"print('x')\" > generated.json",
            "pty": false
        });
        let effects = super::execute_command_segment_effects_for_args(&args);
        assert!(
            effects.iter().any(|effect| effect.project_mutation),
            "未知解释器通过重定向写入项目文件仍应视为项目变更"
        );

        // For perl only the -p/-i (in-place edit) forms are known writers; once the target file resolves, treat as a project change.
        for command in [
            "perl -pi -e 's/foo/bar/' src/main.rs",
            "perl -i -pe 's/foo/bar/' src/main.rs",
        ] {
            let args = serde_json::json!({"command": command, "pty": false});
            let effects = super::execute_command_segment_effects_for_args(&args);
            assert!(
                effects.iter().any(|effect| effect.project_mutation),
                "perl 就地编辑项目文件应视为项目变更: {command}"
            );
        }

        // The cwd fallback for known writers (npm install) stays unchanged.
        let args = serde_json::json!({"command": "npm install", "pty": false});
        let effects = super::execute_command_segment_effects_for_args(&args);
        assert!(
            effects.iter().any(|effect| effect.project_mutation),
            "npm install 在项目 cwd 下仍应视为项目变更"
        );
    }

    #[test]
    fn pre_request_llm_summary_cursor_backoff_after_attempt() {
        let sid = "test-session-cursor-backoff";
        record_llm_summary_attempt_chars(sid, 0);
        let threshold = 240_000;
        let after_chars = 240_457;

        assert!(should_try_llm_summary(sid, after_chars, threshold));

        // The caller writes the cursor after every attempt (success or not). After simulating a failure/no-op,
        // write the post-attempt size so the next same-sized request is blocked by the growth guard,
        // preventing idle retries every turn when the context structurally cannot shrink.
        record_llm_summary_attempt_chars(sid, after_chars);
        assert!(!should_try_llm_summary(sid, after_chars, threshold));
        // Re-trigger only after growth ≥ MIN_GROWTH(20K)
        assert!(should_try_llm_summary(sid, after_chars + 20_000, threshold));

        record_llm_summary_attempt_chars(sid, 230_000);
        assert!(!should_try_llm_summary(sid, after_chars, threshold));
        assert!(should_try_llm_summary(sid, 251_000, threshold));

        // No cross-session interference: another session's cursor stays 0 and should trigger independently.
        let other = "test-session-cursor-isolation";
        assert!(should_try_llm_summary(other, after_chars, threshold));

        record_llm_summary_attempt_chars(sid, 0);
    }
}
