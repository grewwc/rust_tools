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
    request::{self, do_request_messages, do_request_messages_without_tools},
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

fn is_read_only_segment(tokens: &[String]) -> bool {
    let Some(program) = command_program(tokens) else {
        return true;
    };
    let subcommand = command_subcommand(tokens).unwrap_or_default();
    match program {
        "cd" | "pwd" | "ls" | "cat" | "rg" | "grep" | "head" | "tail" | "wc" | "stat" | "file"
        | "which" | "type" | "echo" | "printf" | "sleep" | "true" | "false" | "test" | "sort"
        | "uniq" | "cut" | "find" | "comm" | "tr" => true,
        "sed" => !tokens
            .iter()
            .any(|token| token == "-i" || token.starts_with("-i")),
        "git" => matches!(
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
        ),
        "cargo" => matches!(
            subcommand,
            "check" | "test" | "clippy" | "build" | "metadata"
        ),
        "go" => subcommand == "test",
        "npm" | "pnpm" | "yarn" => matches!(subcommand, "test" | "check"),
        "make" => matches!(subcommand, "test" | "check"),
        "pytest" => true,
        // `bytedcli` 是 ByteDance 内部平台 CLI（codebase/db/faas/log 等子命令
        // 都是远端 API 查询/操作），默认不写本地项目文件，故按只读归类；
        // 但 `--output`/`--output-dir`/`--manifest` 会显式写本地文件
        // （如 `codebase mr artifacts download --output-dir ...`），必须视为变更。
        "bytedcli" => {
            !tokens.iter().any(|token| {
                token == "--output"
                    || token == "--output-dir"
                    || token == "--manifest"
                    || token.starts_with("--output=")
                    || token.starts_with("--output-dir=")
                    || token.starts_with("--manifest=")
            })
        }
        _ => false,
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
            if tokens
                .iter()
                .any(|token| token.starts_with("-pi") || token.starts_with("-ip")) =>
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
            | "perl"
            | "git"
            | "cargo"
            | "npm"
            | "pnpm"
            | "yarn"
            | "make"
    );
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
        let read_only = is_read_only_segment(&tokens);
        let (mut command_targets, known_mutator) = mutation_target_tokens(&tokens);
        raw_targets.append(&mut command_targets);
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
                project_mutation |= path_is_in_project(&cwd, &project_root);
                if !analysis.unknown_mutation_bases.contains(&cwd) {
                    analysis.unknown_mutation_bases.push(cwd.clone());
                }
            }
        }
        if !read_only && (!known_mutator || !resolved_any) {
            project_mutation |= path_is_in_project(&cwd, &project_root);
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
    // 以最后一个**真实** user 消息为本轮起点：运行时合成的 user 消息
    // （证据交接、图片 followup）不构成轮次边界，否则 scoped 指令目标
    // 会被错误地界定在合成消息之后，导致目标目录的 AGENTS.md 从系统提示中消失。
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
) {
    if iteration <= 1 {
        return;
    }

    let prev_skill = skill_turn.matched_skill_name().map(|s| s.to_string());
    let inherited_restore = skill_turn.take_restore_agent_context();

    // 模型通过 activate_skill 工具显式请求激活某个 skill 时优先采纳：直接按名字
    // 强制激活，跳过自动路由打分。名字校验在工具侧已做（必须真实存在），这里
    // 用 skill_manifests 再兜一次，未命中则回退到自动路由。
    let requested = crate::ai::tools::skill_tools::take_pending_skill_activation();
    let mut new_skill_turn = requested
        .as_deref()
        .and_then(|name| {
            skill_runtime::force_activate_named_skill(
                app,
                mcp_client,
                skill_manifests,
                question,
                name,
            )
        })
        .unwrap_or_else(|| {
            skill_runtime::rebuild_skill_turn_with_existing_selection(
                app,
                mcp_client,
                skill_manifests,
                question,
                prev_skill.as_deref(),
            )
        });
    let project_targets = project_instruction_target_paths(messages);
    new_skill_turn.push_scoped_project_instructions(required_project_targets, &project_targets);
    if inherited_restore.is_some() {
        new_skill_turn.set_restore_agent_context(inherited_restore);
    }
    let next_skill = new_skill_turn.matched_skill_name().map(|s| s.to_string());

    if prev_skill != next_skill {
        match next_skill.as_deref() {
            Some(name) => println!("[skill switched: {}]", name.cyan()),
            None => println!("[skill switched: <none>]"),
        }
    }

    *skill_turn = new_skill_turn;
    if let Some(system_message) = messages.first_mut() {
        // 仅当新旧 system prompt 文本不同才覆写。
        // 同一段字符串的覆写不仅没用，还会让上游 prompt cache（例如 anthropic
        // 的 cache_control 命中、或者 driver 内部的字符串 hash 复用）连续失效，
        // 在长 turn 多 iteration 场景里是无声的 token 浪费。
        let next_prompt = skill_turn.system_prompt();
        let same = matches!(&system_message.content, Value::String(s) if s == next_prompt);
        if !same {
            system_message.content = Value::String(next_prompt.to_string());
        }
    }
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
    // 仅消费“本轮由 cancel_stream 触发”的中断，避免误清其它来源
    // （例如 shutdown/request-level interrupt）的全局中断位。
    let _ = crate::ai::types::take_stream_cancelled(app);
    app.ignore_next_prompt_interrupt = true;
    // 标记本轮被打断：run_loop 的 goal 续推逻辑据此区分「打断」与「自然完成」，
    // 打断时保留 goal_mode 并回落到等待用户输入，不误报「Goal achieved」。
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

/// 请求等待期间被打断的原因。
enum RequestInterruptKind {
    /// 用户主动取消（Ctrl+C / shutdown / cancel_stream）。
    User,
    /// 父代理预超时收口信号：放弃当前请求，立即进入强制收口迭代。
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
        // 预超时收口信号（task-local，不影响全局中断状态，也不会误伤并行后台 turn）。
        if crate::ai::driver::runtime_ctx::has_subagent_wrap_up_pending() {
            return RequestInterruptKind::WrapUp;
        }
        // 注册等待 future 后再次检查，避免 signal_request_interrupt 与注册之间的 race。
        let notified = notify.notified();
        if request_interrupt_pending(shutdown.as_ref(), cancel_stream.as_ref())
            || request_interrupt_futex_ready()
        {
            return RequestInterruptKind::User;
        }
        if crate::ai::driver::runtime_ctx::has_subagent_wrap_up_pending() {
            return RequestInterruptKind::WrapUp;
        }
        // 50ms 兜底兼容外部 futex 唤醒（不经 Notify 通道）。
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

/// Reactive 上下文收缩：仅在 provider 因上下文超限拒绝请求后调用。
///
/// 主动压缩（`apply_pre_request_context_budget` + LLM 摘要）已在请求前尽力把
/// 上下文压到软阈值附近，但字符估算不是 token 的权威裁判——一次英文占比高、
/// 或图片/工具 schema 额外开销大的请求，仍可能被 provider 的 tokenizer 判超限。
/// 与其在本地用字符阈值主动 413（会误杀合法请求），不如把请求发出去，只在真正
/// 被拒后再收缩重试。
///
/// 每次调用把目标预算在上次基础上再砍 25%（floor 兜底防止砍到 0），复用跨 turn
/// 压缩管线 [`mid_turn_compress`](crate::ai::history::mid_turn_compress)（含
/// Path C emergency 截断）强制收敛。返回压缩后的字符数；若无法再压缩（已触及
/// system/current-user 不可裁下限），返回的字符数不会下降，调用方据此终止重试。
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
    let (compressed, _before, after) =
        crate::ai::history::mid_turn_compress(drained, target_chars, Some(overflow_dir.as_path()));
    *messages = compressed;
    after
}

/// 在每个模型请求边界更新临时上下文投影，不触碰 canonical `turn_messages`。
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
async fn request_model_response(
    app: &mut App,
    next_model: &str,
    messages: &mut Vec<Message>,
    force_final_response: bool,
    _iteration: usize,
    mut compression_report: CompressionReport,
) -> Result<(reqwest::Response, String), request::RequestError> {
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

    // 每次请求前都处理，而不是只在 turn 初始化时处理。这样同一 turn 内后续工具轮
    // 也能消费刚累计到阈值的 prune 标记，并在上下文压缩前先做无损卸载。
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

    // === Pre-request LLM 摘要兜底 ===
    // 无损+弱损压缩后仍远超阈值时，调用 LLM 把早期对话压成摘要。
    // 这是发送请求前的最后一道防线，避免超大上下文导致模型 4xx 或质量退化
    // （用户报告的 "295K 压到 294K 就停了" 问题）。
    // 阈值取 history_max_chars * 2（默认 180K），比 orchestrator 的 hard
    // threshold（*3.5 = 315K）更积极——后者只在工具调用间隙触发，此处覆盖
    // 每次请求前的最后检查。
    // 增长量守卫：mid-turn 和 pre-request 共享同一个 LLM summary 尝试游标。
    // 同一批上下文刚尝试过且无有效增量时，不再重复请求 summary。
    let llm_threshold = pre_request_llm_summary_threshold(next_model, app.config.history_max_chars);
    let session_id = app.session_id.clone();
    if should_try_llm_summary(&session_id, budget_report.after_chars, llm_threshold) {
        // 取消安全：传入 messages 的 **clone** 而非 `mem::take`。若本次摘要 await
        // 期间被 Ctrl+C 中断，请求 future 被 drop，`messages` 仍保有原始完整内容，
        // 不会退化成空 Vec 导致后续请求发出空上下文 / 丢失消息状态。
        let (after_msgs, llm_before, llm_after, was_effective) =
            crate::ai::history::mid_turn_llm_summarize(
                app,
                messages.clone(),
                MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS,
                MID_TURN_LLM_SUMMARY_MAX_CHARS,
                app.config.history_max_chars,
            )
            .await;
        *messages = after_msgs;
        compression_report.record_llm_summary_attempt(
            format!("pre-request LLM (limit {llm_threshold})"),
            llm_before,
            llm_after,
            was_effective,
        );
        record_llm_summary_attempt_chars(&session_id, llm_after);
    }
    // 摘要管线原则上保留 system 消息；这里仍做一次幂等兜底，确保请求边界协议存在。
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

    // Reactive 上下文超限重试：主动压缩已尽力把上下文压到软阈值附近，但字符估算
    // 不是 provider tokenizer 的权威裁判。若请求仍被判上下文超限，就地收缩后重试，
    // 而不是本地主动 413 误杀合法请求，也不是把超限包硬塞给 provider 后直接放弃。
    // 超限错误不触发模型 fallback（`should_try_model_fallback` 已排除 400/413），
    // 二者互斥。
    const MAX_CONTEXT_OVERFLOW_RETRIES: usize = 4;
    let mut overflow_retries = 0usize;
    loop {
        let mut actual_model = next_model.to_string();
        let mut request_result = if force_final_response {
            do_request_messages_without_tools(app, next_model, messages, true).await
        } else {
            do_request_messages(app, next_model, messages, true).await
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
                    do_request_messages_without_tools(app, &fallback_model, messages, true).await
                } else {
                    do_request_messages(app, &fallback_model, messages, true).await
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
            // 目标在当前实际大小基础上再砍 25%，floor 兜底防止砍到 0 触发 no-op。
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
            // 压不动了（system / current-user 已是不可裁下限）：再重试只会被同样拒绝。
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
                // 预超时收口：放弃当前请求，由 orchestrator 立即进入强制收口迭代。
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
    // 自动 fallback 后仍需把真正完成本次响应的模型传到消息投影与 canonical
    // 持久化层，不能继续使用路由前的 next_model / app.current_model。
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

    // 流式响应中途可能因后端瞬态错误（如 "Cancelled by backend"）中断，
    // 对这类可重试错误重试整条请求+流，避免直接放弃整轮对话。
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
                    let retry_request = if force_final_response {
                        do_request_messages_without_tools(app, &actual_model, messages, true).await
                    } else {
                        do_request_messages(app, &actual_model, messages, true).await
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

                // 不可重试或已用完重试次数——回退到旧行为，继续对话
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
        StreamingFlagGuard, no_tool_handoff_note, project_instruction_target_paths,
        refresh_outstanding_task_anchor, request_interrupt_pending,
    };
    use crate::ai::history::{Message, ROLE_INTERNAL_NOTE};
    use crate::ai::types::{FunctionCall, ToolCall};
    use serde_json::Value;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, atomic::Ordering};

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
        // bytedcli 只做远端 API 查询/操作，不应触发 completion 证据门禁的
        // 项目变更判定，也不应推导出项目指令目标。
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

        // `mr artifacts download` 显式写本地文件（--output-dir），
        // 即使不经 shell 重定向也必须被识别为项目变更。
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
    fn pre_request_llm_summary_cursor_backoff_after_attempt() {
        let sid = "test-session-cursor-backoff";
        record_llm_summary_attempt_chars(sid, 0);
        let threshold = 240_000;
        let after_chars = 240_457;

        assert!(should_try_llm_summary(sid, after_chars, threshold));

        // 调用方在每次尝试后（无论成功与否）都写入游标。模拟失败/no-op 后
        // 写入实际尝试后大小，确保下一次同样大小的请求被 growth 守卫挡掉，
        // 避免结构上无法压缩时每轮空转重试。
        record_llm_summary_attempt_chars(sid, after_chars);
        assert!(!should_try_llm_summary(sid, after_chars, threshold));
        // 增长 ≥ MIN_GROWTH(20K) 后才再次触发
        assert!(should_try_llm_summary(sid, after_chars + 20_000, threshold));

        record_llm_summary_attempt_chars(sid, 230_000);
        assert!(!should_try_llm_summary(sid, after_chars, threshold));
        assert!(should_try_llm_summary(sid, 251_000, threshold));

        // 不同 session 之间互不串扰：另一个 session 游标仍为 0，应独立触发。
        let other = "test-session-cursor-isolation";
        assert!(should_try_llm_summary(other, after_chars, threshold));

        record_llm_summary_attempt_chars(sid, 0);
    }
}
