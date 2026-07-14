use std::error::Error;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use regex::RegexBuilder;

use crate::ai::history;
use crate::ai::theme::{ACCENT_MUTED, ACCENT_SUCCESS, RESET};
use crate::ai::types::{App, QuestionContext};
use crate::clipboardw::string_content;

use crate::ai::{files, prompt::trim_trailing_newline};
use crate::pdfw::{PdfParseOptions, parse_pdf};

const HISTORY_PREVIEW_DEFAULT_COUNT: usize = 6;
const HISTORY_PREVIEW_FULL_COUNT: usize = 20;
const HISTORY_PREVIEW_MAX_COUNT: usize = 20;
const HISTORY_PREVIEW_MAX_CHARS: usize = 160;
const HISTORY_PREVIEW_FULL_MAX_CHARS: usize = 320;
const HISTORY_GREP_HIGHLIGHT_START: &str = "\x1b[1;93m";
const HISTORY_GREP_HIGHLIGHT_END: &str = "\x1b[0m";

fn print_history_help() {
    println!(
        "/history usage:\n  /history [N]           Show last N messages (default: {})\n  /history full          Show full messages\n  /history user/assistant/tool/system\n                         Filter by role\n  /history grep <keyword>  Search messages\n  /history rewind u<N>   Remove user message u<N> and everything after it\n  /history rewind last   Remove latest user message and everything after it\n  /history rewind grep <keyword>\n                         Rewind the only user message matching keyword\n  /history export [path] Export to file\n  /history copy          Copy to clipboard\n  /history last          Replay the last assistant message with markdown rendering\n  /history replay        Replay the last turn's assistant conclusion (text only)\n  /history help            Show this help",
        HISTORY_PREVIEW_DEFAULT_COUNT
    );
}

/// Clear any pending input from stdin to prevent stray Enter keys
/// from interrupting the next input prompt.
pub(crate) fn clear_stdin_buffer() {
    use std::io::IsTerminal;
    if !io::stdin().is_terminal() {
        return;
    }

    #[cfg(unix)]
    {
        use libc::{F_GETFL, F_SETFL, O_NONBLOCK, fcntl};
        use std::os::unix::io::AsRawFd;

        let fd = io::stdin().as_raw_fd();
        unsafe {
            // Get current flags
            let flags = fcntl(fd, F_GETFL, 0);
            if flags >= 0 {
                // Set non-blocking mode
                let _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK);

                // Read and discard any pending input
                let mut buf = [0u8; 1024];
                while libc::read(fd, buf.as_mut_ptr() as *mut _, buf.len()) > 0 {
                    // Discard
                }

                // Restore blocking mode
                let _ = fcntl(fd, F_SETFL, flags);
            }
        }
    }
}

pub(crate) fn next_question(app: &mut App) -> Result<Option<QuestionContext>, Box<dyn Error>> {
    if !app.cli.args.is_empty() {
        let base_question = app.cli.args.join(" ");
        app.cli.args.clear();
        if handle_local_command(app, &base_question)? {
            crate::ai::driver::signal::request_shutdown(app.shutdown.as_ref());
            return Ok(None);
        }
        let ctx = finalize_question(app, base_question, 0)?;
        return Ok(Some(ctx));
    }

    let question = loop {
        match prompt_user(app) {
            Ok(v) => {
                app.ignore_next_prompt_interrupt = false;
                let Some(input) = v else {
                    break None;
                };
                if handle_local_command(app, &input)? {
                    continue;
                }
                break Some(input);
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                if app.ignore_next_prompt_interrupt {
                    app.ignore_next_prompt_interrupt = false;
                    clear_stdin_buffer();
                    continue;
                }
                println!("Exit.");
                crate::ai::driver::signal::request_shutdown(app.shutdown.as_ref());
                return Ok(None);
            }
            Err(_) if app.shutdown.load(Ordering::Relaxed) => {
                return Ok(None);
            }
            Err(err) => return Err(err.into()),
        }
    };
    let Some(question) = question else {
        app.ignore_next_prompt_interrupt = false;
        crate::ai::driver::signal::request_shutdown(app.shutdown.as_ref());
        return Ok(None);
    };
    let ctx = finalize_question(app, question, 0)?;
    Ok(Some(ctx))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LocalCommand {
    ShowHistory(HistoryPreviewOptions),
    HelpHistory,
    ExportHistory(HistoryPreviewOptions, Option<PathBuf>),
    CopyHistory(HistoryPreviewOptions),
    RewindHistory(HistoryRewindTarget),
    RenderLastHistoryMessage,
    ReplayHistory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistoryRoleFilter {
    All,
    User,
    Assistant,
    Tool,
    System,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HistoryPreviewOptions {
    count: usize,
    role_filter: HistoryRoleFilter,
    full: bool,
    grep: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HistoryRewindTarget {
    UserOrdinal(usize),
    LatestUser,
    Grep(String),
}

#[derive(Debug, Clone)]
struct HistoryPreviewItem {
    message: history::Message,
    user_ordinal: Option<usize>,
}

fn handle_local_command(app: &mut App, input: &str) -> io::Result<bool> {
    match handle_local_command_inner(app, input) {
        Ok(v) => Ok(v),
        Err(e) => {
            eprintln!("local command error: {e}");
            Ok(true)
        }
    }
}

fn handle_local_command_inner(app: &mut App, input: &str) -> Result<bool, Box<dyn Error>> {
    let Some(command) = parse_local_command(input)? else {
        return Ok(false);
    };

    match command {
        LocalCommand::ShowHistory(options) => {
            println!("{}", render_history_preview(app, options)?);
        }
        LocalCommand::RenderLastHistoryMessage => {
            render_last_history_message(app)?;
        }
        LocalCommand::ReplayHistory => {
            println!("{}", render_history_replay(app)?);
        }
        LocalCommand::HelpHistory => {
            print_history_help();
        }
        LocalCommand::ExportHistory(options, output_path) => {
            let path = output_path.unwrap_or_else(|| default_history_export_path(app));
            let rendered = render_history_preview(app, options)?;
            fs::write(&path, rendered)?;
            println!("[history] Exported to {}", path.display());
        }
        LocalCommand::CopyHistory(options) => {
            let rendered = render_history_preview(app, options)?;
            string_content::set_clipboard_content(&rendered)?;
            println!("[history] Copied preview to clipboard.");
        }
        LocalCommand::RewindHistory(target) => {
            let plan = plan_history_rewind(app, target)?;
            if plan.removed_messages == 0 {
                println!("[history] Nothing to rewind.");
                return Ok(true);
            }
            let confirm = crate::commonw::prompt::prompt_yes_or_no_interruptible(&format!(
                "Rewind from user input u{} and remove {} message(s)? (y/n): ",
                plan.user_ordinal, plan.removed_messages
            ));
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }
            apply_history_rewind(app, plan)?;
        }
    }
    Ok(true)
}

fn parse_local_command(input: &str) -> Result<Option<LocalCommand>, Box<dyn Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(None);
    };
    let mut parts = normalized.split_whitespace();
    let Some(command) = parts.next() else {
        return Ok(None);
    };
    match command {
        "history" => parse_history_local_command(parts.collect::<Vec<_>>().as_slice()),
        _ => Ok(None),
    }
}

fn parse_history_local_command(args: &[&str]) -> Result<Option<LocalCommand>, Box<dyn Error>> {
    if args.first().copied() == Some("rewind") || args.first().copied() == Some("undo") {
        return parse_history_rewind_command(&args[1..])
            .map(|target| Some(LocalCommand::RewindHistory(target)));
    }
    // `/history last`：按 markdown 渲染回放最后一条 assistant 结论消息。
    if args.first().copied() == Some("last") {
        if args.len() > 1 {
            return Err("`/history last` takes no arguments".into());
        }
        return Ok(Some(LocalCommand::RenderLastHistoryMessage));
    }
    // `/history replay`：回放最后一轮模型的结论文本（不含 tool/thinking）。
    if args.first().copied() == Some("replay") {
        if args.len() > 1 {
            return Err("`/history replay` takes no arguments".into());
        }
        return Ok(Some(LocalCommand::ReplayHistory));
    }
    let (options, action) = parse_history_preview_options(args)?;
    Ok(Some(match action {
        HistoryAction::Show => LocalCommand::ShowHistory(options),
        HistoryAction::Help => LocalCommand::HelpHistory,
        HistoryAction::Export(path) => {
            LocalCommand::ExportHistory(options, path.map(PathBuf::from))
        }
        HistoryAction::Copy => LocalCommand::CopyHistory(options),
    }))
}

fn parse_history_rewind_command(args: &[&str]) -> Result<HistoryRewindTarget, Box<dyn Error>> {
    let Some(first) = args.first().copied() else {
        return Err("missing rewind target. try: /history rewind u3".into());
    };
    if first == "last" || first == "latest" {
        return Ok(HistoryRewindTarget::LatestUser);
    }
    if first == "grep" {
        let keyword = args[1..].join(" ");
        if keyword.trim().is_empty() {
            return Err("`/history rewind grep` requires a keyword".into());
        }
        return Ok(HistoryRewindTarget::Grep(keyword));
    }
    let raw = first
        .strip_prefix('u')
        .or_else(|| first.strip_prefix('U'))
        .unwrap_or(first);
    let ordinal = raw
        .parse::<usize>()
        .map_err(|_| format!("invalid rewind target: {first}. try: /history rewind u3"))?;
    if ordinal == 0 {
        return Err("user ordinal must be >= 1".into());
    }
    Ok(HistoryRewindTarget::UserOrdinal(ordinal))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HistoryAction {
    Show,
    Help,
    Export(Option<String>),
    Copy,
}

fn parse_history_preview_options(
    args: &[&str],
) -> Result<(HistoryPreviewOptions, HistoryAction), Box<dyn Error>> {
    let mut options = HistoryPreviewOptions {
        count: HISTORY_PREVIEW_DEFAULT_COUNT,
        role_filter: HistoryRoleFilter::All,
        full: false,
        grep: None,
    };
    let mut action = HistoryAction::Show;
    let mut idx = 0usize;

    while idx < args.len() {
        match args[idx] {
            "full" => {
                options.full = true;
                options.count = HISTORY_PREVIEW_FULL_COUNT;
                idx += 1;
            }
            "user" => {
                options.role_filter = HistoryRoleFilter::User;
                idx += 1;
            }
            "assistant" => {
                options.role_filter = HistoryRoleFilter::Assistant;
                idx += 1;
            }
            "tool" => {
                options.role_filter = HistoryRoleFilter::Tool;
                idx += 1;
            }
            "system" => {
                options.role_filter = HistoryRoleFilter::System;
                idx += 1;
            }
            "grep" => {
                let keyword = args[idx + 1..].join(" ");
                if keyword.trim().is_empty() {
                    return Err("`/history grep` requires a keyword".into());
                }
                options.grep = Some(keyword);
                break;
            }
            "export" => {
                action = HistoryAction::Export(args.get(idx + 1).map(|s| (*s).to_string()));
                break;
            }
            "copy" => {
                action = HistoryAction::Copy;
                break;
            }
            "help" => {
                action = HistoryAction::Help;
                break;
            }
            raw => {
                options.count = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid /history argument: {raw}"))?
                    .clamp(1, HISTORY_PREVIEW_MAX_COUNT);
                idx += 1;
                continue;
            }
        }
    }

    Ok((options, action))
}

fn render_history_preview(
    app: &App,
    options: HistoryPreviewOptions,
) -> Result<String, Box<dyn Error>> {
    let grep = options.grep.clone();
    let label = match options.role_filter {
        HistoryRoleFilter::All => "message(s)",
        HistoryRoleFilter::User => "user message(s)",
        HistoryRoleFilter::Assistant => "assistant message(s)",
        HistoryRoleFilter::Tool => "tool message(s)",
        HistoryRoleFilter::System => "system message(s)",
    };
    let max_chars = if options.full {
        HISTORY_PREVIEW_FULL_MAX_CHARS
    } else {
        HISTORY_PREVIEW_MAX_CHARS
    };
    let grep_suffix = options
        .grep
        .as_deref()
        .map(|grep| format!(" matching \"{}\"", grep))
        .unwrap_or_default();
    let shown = collect_history_messages(app, options)?;
    if shown.is_empty() {
        return Ok("[history] No recent messages.".to_string());
    }

    let total = shown.len();
    let mut out = format!(
        "[history] Showing {} recent {}{}:\n",
        total, label, grep_suffix
    );
    for (idx, item) in shown.iter().enumerate() {
        let content = summarize_history_content(&item.message.content, max_chars, grep.as_deref());
        let marker = item
            .user_ordinal
            .map(|ordinal| format!(" (u{ordinal})"))
            .unwrap_or_default();
        out.push_str(&format!(
            "{}. [{}] {}{}\n",
            idx + 1,
            item.message.role,
            content,
            marker
        ));
    }
    Ok(out.trim_end().to_string())
}

fn last_assistant_conclusion_text(app: &App) -> Result<Option<String>, Box<dyn Error>> {
    let history_file = active_history_path(app);
    let messages = history::build_message_arr(usize::MAX, &history_file)?;
    Ok(messages.iter().rev().find_map(|message| {
        if message.role != "assistant" {
            return None;
        }
        let has_tool_calls = message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty());
        if has_tool_calls {
            return None;
        }
        let text = searchable_history_content(&message.content);
        if text.trim().is_empty() {
            return None;
        }
        Some(text)
    }))
}

/// `/history last`：完整回放最后一条 assistant 结论消息，并走终端 markdown 渲染。
fn render_last_history_message(app: &App) -> Result<(), Box<dyn Error>> {
    match last_assistant_conclusion_text(app)? {
        Some(text) => {
            crate::ai::stream::render_markdown_block(&text)?;
            Ok(())
        }
        None => {
            println!("[history] No assistant conclusion found in recent history.");
            Ok(())
        }
    }
}

/// 渲染 `/history replay` 的输出：只取最后一轮 assistant 的结论文本，
/// 跳过 tool_calls（工具调用步骤）与 reasoning_content（thinking）。
fn render_history_replay(app: &App) -> Result<String, Box<dyn Error>> {
    Ok(match last_assistant_conclusion_text(app)? {
        Some(text) => text,
        None => "[history] No assistant conclusion found in recent history.".to_string(),
    })
}

fn collect_history_messages(
    app: &App,
    options: HistoryPreviewOptions,
) -> Result<Vec<HistoryPreviewItem>, Box<dyn Error>> {
    let history_file = active_history_path(app);
    let messages = history::build_message_arr(usize::MAX, &history_file)?;
    let mut user_ordinal = 0usize;
    let filtered = messages
        .into_iter()
        .filter_map(|message| {
            let ordinal = if message.role == "user" {
                user_ordinal += 1;
                Some(user_ordinal)
            } else {
                None
            };
            let role_matches = match options.role_filter {
                HistoryRoleFilter::All => true,
                HistoryRoleFilter::User => message.role == "user",
                HistoryRoleFilter::Assistant => message.role == "assistant",
                HistoryRoleFilter::Tool => message.role == "tool",
                HistoryRoleFilter::System => crate::ai::history::is_system_like_role(&message.role),
            };
            role_matches.then_some(HistoryPreviewItem {
                message,
                user_ordinal: ordinal,
            })
        })
        .filter(|item| {
            options.grep.as_deref().is_none_or(|needle| {
                let haystack =
                    searchable_history_content(&item.message.content).to_ascii_lowercase();
                haystack.contains(&needle.to_ascii_lowercase())
            })
        })
        .collect::<Vec<_>>();
    let shown = if filtered.len() > options.count {
        filtered[filtered.len() - options.count..].to_vec()
    } else {
        filtered
    };
    Ok(shown)
}

#[derive(Debug, Clone)]
struct HistoryRewindPlan {
    user_ordinal: usize,
    keep_messages: Vec<history::Message>,
    removed_messages: usize,
    preview: String,
}

fn plan_history_rewind(
    app: &App,
    target: HistoryRewindTarget,
) -> Result<HistoryRewindPlan, Box<dyn Error>> {
    let history_file = active_history_path(app);
    let messages = history::build_message_arr(usize::MAX, &history_file)?;
    let target_ordinal = resolve_rewind_target(&messages, target)?;
    let Some((target_index, preview)) = find_user_message_by_ordinal(&messages, target_ordinal)
    else {
        return Err(format!("user input u{target_ordinal} not found").into());
    };
    let keep_messages = messages[..target_index].to_vec();
    Ok(HistoryRewindPlan {
        user_ordinal: target_ordinal,
        removed_messages: messages.len().saturating_sub(target_index),
        keep_messages,
        preview,
    })
}

fn apply_history_rewind(app: &mut App, plan: HistoryRewindPlan) -> Result<(), Box<dyn Error>> {
    let history_file = active_history_path(app);
    history::replace_history_messages(&history_file, &plan.keep_messages)?;
    history::invalidate_context_history_cache_for(&history_file);
    crate::ai::driver::commands::session::clear_session_local_runtime_state(app);
    println!(
        "[history] Rewound from u{}: {}",
        plan.user_ordinal, plan.preview
    );
    println!(
        "[history] Removed {} message(s); kept {} message(s).",
        plan.removed_messages,
        plan.keep_messages.len()
    );
    Ok(())
}

fn resolve_rewind_target(
    messages: &[history::Message],
    target: HistoryRewindTarget,
) -> Result<usize, Box<dyn Error>> {
    match target {
        HistoryRewindTarget::UserOrdinal(ordinal) => Ok(ordinal),
        HistoryRewindTarget::LatestUser => {
            let count = messages
                .iter()
                .filter(|message| message.role == "user")
                .count();
            if count == 0 {
                Err("no user input found in history".into())
            } else {
                Ok(count)
            }
        }
        HistoryRewindTarget::Grep(keyword) => {
            let needle = keyword.to_ascii_lowercase();
            let mut matches = Vec::new();
            let mut ordinal = 0usize;
            for message in messages {
                if message.role != "user" {
                    continue;
                }
                ordinal += 1;
                let text = searchable_history_content(&message.content);
                if text.to_ascii_lowercase().contains(&needle) {
                    matches.push((ordinal, text));
                }
            }
            match matches.len() {
                0 => Err(format!("no user input matching {:?}", keyword).into()),
                1 => Ok(matches[0].0),
                _ => {
                    let mut msg = format!(
                        "{} user inputs match {:?}; use /history rewind u<N>:\n",
                        matches.len(),
                        keyword
                    );
                    for (ordinal, text) in matches.into_iter().take(8) {
                        msg.push_str(&format!(
                            "  u{} {}\n",
                            ordinal,
                            truncate_for_terminal(
                                &text.split_whitespace().collect::<Vec<_>>().join(" "),
                                120
                            )
                        ));
                    }
                    Err(msg.trim_end().to_string().into())
                }
            }
        }
    }
}

fn find_user_message_by_ordinal(
    messages: &[history::Message],
    target_ordinal: usize,
) -> Option<(usize, String)> {
    let mut ordinal = 0usize;
    for (idx, message) in messages.iter().enumerate() {
        if message.role != "user" {
            continue;
        }
        ordinal += 1;
        if ordinal == target_ordinal {
            let preview = truncate_for_terminal(
                &searchable_history_content(&message.content)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" "),
                120,
            );
            return Some((idx, preview));
        }
    }
    None
}

fn active_history_path(app: &App) -> std::path::PathBuf {
    if !app.session_history_file.as_os_str().is_empty() {
        app.session_history_file.clone()
    } else {
        app.config.history_file.clone()
    }
}

fn summarize_history_content(
    value: &serde_json::Value,
    max_chars: usize,
    grep: Option<&str>,
) -> String {
    let raw = searchable_history_content(value);
    let single_line = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let truncated = truncate_for_terminal(&single_line, max_chars);
    highlight_history_keyword(&truncated, grep)
}

fn searchable_history_content(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => {
            serde_json::to_string(other).unwrap_or_else(|_| "<non-string content>".to_string())
        }
    }
}

fn truncate_for_terminal(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = String::new();
    for (idx, ch) in value.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn highlight_history_keyword(value: &str, grep: Option<&str>) -> String {
    let Some(needle) = grep.map(str::trim).filter(|needle| !needle.is_empty()) else {
        return value.to_string();
    };
    let Ok(pattern) = RegexBuilder::new(&regex::escape(needle))
        .case_insensitive(true)
        .build()
    else {
        return value.to_string();
    };

    pattern
        .replace_all(value, |caps: &regex::Captures| {
            format!(
                "{}{}{}",
                HISTORY_GREP_HIGHLIGHT_START, &caps[0], HISTORY_GREP_HIGHLIGHT_END
            )
        })
        .into_owned()
}

fn default_history_export_path(app: &App) -> PathBuf {
    let file_name = if app.session_id.trim().is_empty() {
        "history-export.txt".to_string()
    } else {
        format!("history-{}.txt", app.session_id)
    };
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(file_name)
}

const IMAGE_PLACEHOLDER_PREFIX: &str = "[[image:";
const IMAGE_PLACEHOLDER_SUFFIX: &str = "]]";

fn extract_at_file_references(question: &mut String) -> crate::ai::types::FileParseResult {
    let mut parsed = crate::ai::types::FileParseResult::default();
    let mut rewritten = String::with_capacity(question.len());
    let chars: Vec<char> = question.chars().collect();
    let mut i = 0usize;

    while i < chars.len() {
        if chars[i] != '@' || !at_ref_can_start(&chars, i) {
            rewritten.push(chars[i]);
            i += 1;
            continue;
        }

        let Some((next_index, raw_path)) = parse_at_ref_candidate(&chars, i) else {
            rewritten.push(chars[i]);
            i += 1;
            continue;
        };

        let Some(path) = normalize_existing_ref_path(&raw_path) else {
            rewritten.push(chars[i]);
            i += 1;
            continue;
        };

        files::classify_file_reference(&mut parsed, &path);
        if !parsed.text_files.iter().any(|candidate| candidate == &path)
            && !parsed
                .image_files
                .iter()
                .any(|candidate| candidate == &path)
            && !parsed
                .binary_files
                .iter()
                .any(|candidate| candidate == &path)
        {
            rewritten.push(chars[i]);
            i += 1;
            continue;
        }

        i = next_index;
    }

    *question = rewritten;
    parsed
}

/// 从输入中提取并移除 `@skills:<name>` / `@skill:<name>` 引用（用户通过补全选择的
/// 强制 skill）。返回最后一个命中的 skill 名（多次出现以最后一个为准）。被提取的
/// token 从 `question` 中删除，避免污染发送给模型的文本。
fn extract_forced_skill_reference(question: &mut String) -> Option<String> {
    let chars: Vec<char> = question.chars().collect();
    let mut rewritten = String::with_capacity(question.len());
    let mut i = 0usize;
    let mut selected: Option<String> = None;

    while i < chars.len() {
        if chars[i] != '@' || !at_ref_can_start(&chars, i) {
            rewritten.push(chars[i]);
            i += 1;
            continue;
        }

        // 读取 `@` 之后的非空白 token。
        let mut idx = i + 1;
        let mut token = String::new();
        while idx < chars.len() && !chars[idx].is_whitespace() {
            token.push(chars[idx]);
            idx += 1;
        }

        let lower = token.to_ascii_lowercase();
        let name = lower
            .strip_prefix("skills:")
            .or_else(|| lower.strip_prefix("skill:"));
        let prefix_len = if lower.starts_with("skills:") {
            "skills:".len()
        } else {
            "skill:".len()
        };
        if name.is_some_and(|n| !n.is_empty()) {
            // 用原始大小写截取 skill 名（保持与 manifest 匹配时的展示一致）。
            let raw_name: String = token.chars().skip(prefix_len).collect();
            selected = Some(raw_name);
            i = idx;
            continue;
        }

        rewritten.push(chars[i]);
        i += 1;
    }

    if selected.is_some() {
        // 清理因移除 token 可能残留的多余空白。
        *question = rewritten.trim().to_string();
    }
    selected
}

fn at_ref_can_start(chars: &[char], index: usize) -> bool {
    if index == 0 {
        return true;
    }
    let prev = chars[index - 1];
    prev.is_whitespace() || matches!(prev, '(' | '[' | '{' | '"' | '\'')
}

fn parse_at_ref_candidate(chars: &[char], at_index: usize) -> Option<(usize, String)> {
    let start = at_index + 1;
    if start >= chars.len() {
        return None;
    }

    let quote = chars[start];
    if quote == '"' || quote == '\'' {
        let mut idx = start + 1;
        let mut value = String::new();
        while idx < chars.len() && chars[idx] != quote {
            value.push(chars[idx]);
            idx += 1;
        }
        if idx >= chars.len() || value.trim().is_empty() {
            return None;
        }
        return Some((idx + 1, value));
    }

    let mut idx = start;
    let mut value = String::new();
    while idx < chars.len() && !chars[idx].is_whitespace() {
        value.push(chars[idx]);
        idx += 1;
    }
    if value.is_empty() {
        return None;
    }
    Some((idx, value))
}

fn normalize_existing_ref_path(raw: &str) -> Option<String> {
    let mut candidate = raw.trim().to_string();
    while !candidate.is_empty() {
        let expanded = crate::commonw::utils::expanduser(&candidate).to_string();
        if fs::metadata(&expanded).is_ok() {
            return Some(expanded);
        }
        let Some(last) = candidate.chars().last() else {
            break;
        };
        if !matches!(last, ',' | '.' | ';' | ':' | '!' | '?' | ')' | ']' | '}') {
            break;
        }
        candidate.pop();
    }
    None
}

fn extract_inline_image_paths(question: &mut String) -> Vec<String> {
    let mut images = Vec::new();
    while let Some(start) = question.find(IMAGE_PLACEHOLDER_PREFIX) {
        let search_start = start + IMAGE_PLACEHOLDER_PREFIX.len();
        let Some(end_rel) = question[search_start..].find(IMAGE_PLACEHOLDER_SUFFIX) else {
            break;
        };
        let end = search_start + end_rel;
        let path = question[search_start..end].trim().to_string();
        if !path.is_empty() {
            images.push(path);
        }
        let remove_end = end + IMAGE_PLACEHOLDER_SUFFIX.len();
        question.replace_range(start..remove_end, "");
    }
    images
}

/// 粘贴图片的占位符只保留文件名（`paste-xxxx.png`，便于在编辑器中阅读），
/// 实际文件保存在 `<session>.assets/` 目录下。这里把这类裸文件名回挂到
/// 会话 assets 目录，得到可读取的绝对路径；已是存在的绝对/相对路径则原样保留。
fn resolve_inline_image_path(raw: &str, assets_dir: &Path) -> String {
    if Path::new(raw).is_file() {
        return raw.to_string();
    }
    // 仅对"纯文件名"（不含路径分隔符）做 assets 目录回挂，避免误改用户显式给出的路径。
    if !raw.contains('/') && !raw.contains('\\') {
        let candidate = assets_dir.join(raw);
        if candidate.is_file() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    raw.to_string()
}

fn apply_text_files_prefix(
    attachments: &mut String,
    text_files: &[String],
) -> Result<(), Box<dyn Error>> {
    if text_files.is_empty() {
        return Ok(());
    }
    let prefix = files::text_file_contents(text_files)?;
    if !prefix.is_empty() {
        if !attachments.is_empty() && !attachments.ends_with('\n') {
            attachments.push('\n');
        }
        attachments.push_str(&prefix);
    }
    Ok(())
}

fn is_pdf_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

fn split_pdf_files(files: Vec<String>) -> (Vec<String>, Vec<String>) {
    let mut pdfs = Vec::new();
    let mut unsupported = Vec::new();
    for file in files {
        if is_pdf_path(&file) {
            pdfs.push(file);
        } else {
            unsupported.push(file);
        }
    }
    (pdfs, unsupported)
}

fn build_pdf_text_prefix(pdfs: &[String]) -> String {
    let mut prefix = String::new();
    for path in pdfs {
        let display_name = Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        prefix.push_str("File: ");
        prefix.push_str(display_name);
        prefix.push('\n');

        let parsed = parse_pdf(path, PdfParseOptions::default()).ok();
        let Some(parsed) = parsed else {
            continue;
        };
        let Some(text) = parsed.text else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        prefix.push_str(text);
        prefix.push('\n');
        prefix.push('\n');
    }
    prefix
}

fn handle_binary_files(
    attachments: &mut String,
    binary_files: Vec<String>,
) -> Result<(), Box<dyn Error>> {
    if binary_files.is_empty() {
        return Ok(());
    }

    let (pdfs, unsupported) = split_pdf_files(binary_files);
    if !pdfs.is_empty() {
        let prefix = build_pdf_text_prefix(&pdfs);
        if !prefix.trim().is_empty() {
            if !attachments.is_empty() && !attachments.ends_with('\n') {
                attachments.push('\n');
            }
            attachments.push_str(&prefix);
        }
    }

    if !unsupported.is_empty() {
        return Err(format!("unsupported binary files: {}", unsupported.join(", ")).into());
    }
    Ok(())
}

fn finalize_question(
    app: &mut App,
    mut question: String,
    history_count: usize,
) -> Result<QuestionContext, Box<dyn Error>> {
    // 先提取 `@skills:<name>`（用户经补全显式选择、仅本轮强制注入的 skill），
    // 再做普通 `@file` 引用提取，避免 skill 引用被误当作文件路径处理。
    if let Some(name) = extract_forced_skill_reference(&mut question) {
        app.forced_skill = Some(name);
    }
    let inline_files = extract_at_file_references(&mut question);
    let mut inline_images = extract_inline_image_paths(&mut question);
    let mut attachments_text = String::new();
    apply_text_files_prefix(&mut attachments_text, &inline_files.text_files)?;
    if !inline_files.image_files.is_empty() {
        inline_images.extend(inline_files.image_files);
    }
    handle_binary_files(&mut attachments_text, inline_files.binary_files)?;
    if let Some(files) = app.pending_files.take() {
        let parsed = files::parse_files(&files);
        apply_text_files_prefix(&mut attachments_text, &parsed.text_files)?;
        if !parsed.image_files.is_empty() {
            inline_images.extend(parsed.image_files);
        }
        handle_binary_files(&mut attachments_text, parsed.binary_files)?;
    }
    // 图片是 per-turn 状态：每个 turn 重置，避免上一轮图片被无声地拼到
    // 后续 turn 的 user message（重复 token + 重复 OCR + 把 model 锁在 VL）。
    // 内联图片占位符仅存文件名，这里回挂到 session assets 目录解析出真实路径。
    let assets_dir = {
        use crate::ai::history::SessionStore;
        let store = SessionStore::new(app.config.history_file.as_path());
        store.session_assets_dir(&app.session_id)
    };
    app.attached_image_files = inline_images
        .into_iter()
        .map(|raw| resolve_inline_image_path(&raw, assets_dir.as_path()))
        .collect();

    Ok(QuestionContext {
        question,
        attachments_text,
        history_count,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        HistoryAction, HistoryPreviewOptions, HistoryRewindTarget, HistoryRoleFilter, LocalCommand,
        apply_history_rewind, extract_at_file_references, extract_forced_skill_reference,
        finalize_question, highlight_history_keyword, last_assistant_conclusion_text,
        parse_history_preview_options, parse_local_command, plan_history_rewind,
        render_history_preview, render_history_replay, resolve_inline_image_path,
        searchable_history_content, summarize_history_content, truncate_for_terminal,
    };
    use crate::ai::{
        history::{self, Message, append_history_messages},
        types::{
            AgentContext, App, AppConfig, FunctionDefinition, SkillBiasMemory, ToolDefinition,
        },
    };
    use aios_kernel::primitives::{DaemonKind, DaemonState};
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};
    use uuid::Uuid;

    #[test]
    fn resolve_inline_image_remaps_bare_filename_to_assets_dir() {
        let dir = std::env::temp_dir().join(format!("ai-img-resolve-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file_name = "paste-abc.png";
        let full = dir.join(file_name);
        std::fs::write(&full, b"x").unwrap();

        // 裸文件名应回挂到 assets 目录的真实路径。
        let resolved = resolve_inline_image_path(file_name, &dir);
        assert_eq!(resolved, full.to_string_lossy());

        // 不存在的文件名保持原样（不臆造路径）。
        let missing = resolve_inline_image_path("paste-missing.png", &dir);
        assert_eq!(missing, "paste-missing.png");

        // 已是存在的绝对路径原样保留。
        let abs = resolve_inline_image_path(&full.to_string_lossy(), &dir);
        assert_eq!(abs, full.to_string_lossy());

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn any_model_name() -> String {
        crate::ai::model_names::all()
            .first()
            .map(|m| m.name.clone())
            .expect("models.json is empty")
    }

    #[test]
    fn extract_forced_skill_reference_strips_token_and_returns_name() {
        let mut q = "请帮我 @skills:code-review 看看这段代码".to_string();
        let name = extract_forced_skill_reference(&mut q);
        assert_eq!(name.as_deref(), Some("code-review"));
        assert!(!q.contains("@skills:"));
        assert!(q.contains("请帮我"));
        assert!(q.contains("看看这段代码"));
    }

    #[test]
    fn extract_forced_skill_reference_supports_singular_and_keeps_case() {
        let mut q = "@skill:MySkill do it".to_string();
        let name = extract_forced_skill_reference(&mut q);
        assert_eq!(name.as_deref(), Some("MySkill"));
        assert_eq!(q, "do it");
    }

    #[test]
    fn extract_forced_skill_reference_ignores_midword_and_bare() {
        // 词中间的 `@` 不是边界，不应被当作技能引用。
        let mut q = "email@skills:foo".to_string();
        assert!(extract_forced_skill_reference(&mut q).is_none());
        // 裸 `@skills`（无 `:name`）不构成显式选择。
        let mut q2 = "@skills".to_string();
        assert!(extract_forced_skill_reference(&mut q2).is_none());
    }

    fn any_vl_model_name() -> String {
        crate::ai::model_names::all()
            .iter()
            .find(|m| m.is_vl)
            .map(|m| m.name.clone())
            .unwrap_or_else(any_model_name)
    }

    fn test_app() -> App {
        let client = reqwest::Client::builder().build().unwrap();
        App {
            cli: crate::ai::cli::ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                base_history_file: PathBuf::new(),
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: any_vl_model_name(),
                history_max_chars: 12000,
                history_keep_last: 8,
                history_summary_max_chars: 4000,
                intent_model: None,
                agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client,
            current_model: any_model_name(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skill: None,
            forced_question: None,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            prompt_editor: None,
            agent_context: None,
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(
                crate::ai::driver::thinking::ThinkingOrchestrator::new(),
            )],
            last_known_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
            last_turn_interrupted: false,
            prune_marks: Default::default(),
        }
    }

    fn test_message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn at_image_reference_is_attached_and_removed_from_question() {
        let path = std::env::temp_dir().join(format!("ai-image-{}.png", Uuid::new_v4()));
        std::fs::write(&path, b"fake").unwrap();

        let mut app = test_app();
        let question = format!("Please inspect @{} now", path.display());
        let ctx = finalize_question(&mut app, question, 6).unwrap();

        assert!(!ctx.question.contains(path.to_string_lossy().as_ref()));
        assert_eq!(
            app.attached_image_files,
            vec![path.to_string_lossy().to_string()]
        );
    }

    #[test]
    fn at_agent_mention_is_not_treated_as_file_reference() {
        let mut question = "@explore check this module".to_string();
        let parsed = extract_at_file_references(&mut question);

        assert!(parsed.text_files.is_empty());
        assert!(parsed.image_files.is_empty());
        assert!(parsed.binary_files.is_empty());
        assert_eq!(question, "@explore check this module");
    }

    #[test]
    fn quoted_at_text_file_reference_is_inlined() {
        let path = std::env::temp_dir().join(format!("ai-note-{}.txt", Uuid::new_v4()));
        std::fs::write(&path, "hello from file").unwrap();

        let mut app = test_app();
        let question = format!("Summarize @\"{}\"", path.display());
        let ctx = finalize_question(&mut app, question, 6).unwrap();

        assert!(
            ctx.attachments_text
                .contains(&format!("[Attached text file: {}]", path.display()))
        );
        assert!(ctx.attachments_text.contains("hello from file"));
        assert!(!ctx.question.contains("hello from file"));
        assert!(!ctx.question.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn large_text_attachment_is_truncated_with_followup_hint() {
        let path = std::env::temp_dir().join(format!("ai-large-note-{}.rs", Uuid::new_v4()));
        let content = (1..=400)
            .map(|idx| format!("fn item_{idx}() {{}}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, content).unwrap();

        let mut app = test_app();
        let question = format!("Summarize @\"{}\"", path.display());
        let ctx = finalize_question(&mut app, question, 6).unwrap();

        assert!(ctx.attachments_text.contains("Attachment preview only"));
        assert!(ctx.attachments_text.contains("read_file_lines"));
        assert!(ctx.attachments_text.contains("Symbol outline"));
        assert!(!ctx.question.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn pending_files_image_is_attached_for_dash_f_flow() {
        let path = std::env::temp_dir().join(format!("ai-pending-image-{}.png", Uuid::new_v4()));
        std::fs::write(&path, b"fake").unwrap();

        let mut app = test_app();
        app.pending_files = Some(path.to_string_lossy().to_string());
        let ctx = finalize_question(&mut app, "describe this".to_string(), 6).unwrap();

        assert_eq!(ctx.question, "describe this");
        assert_eq!(
            app.attached_image_files,
            vec![path.to_string_lossy().to_string()]
        );
    }

    #[test]
    fn parse_local_command_supports_history_default_and_custom_count() {
        assert_eq!(
            parse_local_command("/history").unwrap(),
            Some(LocalCommand::ShowHistory(HistoryPreviewOptions {
                count: 6,
                role_filter: HistoryRoleFilter::All,
                full: false,
                grep: None,
            }))
        );
        assert_eq!(
            parse_local_command("/history 3").unwrap(),
            Some(LocalCommand::ShowHistory(HistoryPreviewOptions {
                count: 3,
                role_filter: HistoryRoleFilter::All,
                full: false,
                grep: None,
            }))
        );
        assert_eq!(
            parse_local_command("/history 999").unwrap(),
            Some(LocalCommand::ShowHistory(HistoryPreviewOptions {
                count: 20,
                role_filter: HistoryRoleFilter::All,
                full: false,
                grep: None,
            }))
        );
        assert_eq!(
            parse_local_command(":history user 4").unwrap(),
            Some(LocalCommand::ShowHistory(HistoryPreviewOptions {
                count: 4,
                role_filter: HistoryRoleFilter::User,
                full: false,
                grep: None,
            }))
        );
        assert_eq!(
            parse_local_command("/history grep panic").unwrap(),
            Some(LocalCommand::ShowHistory(HistoryPreviewOptions {
                count: 6,
                role_filter: HistoryRoleFilter::All,
                full: false,
                grep: Some("panic".to_string()),
            }))
        );
        assert_eq!(
            parse_local_command("/history export dump.txt").unwrap(),
            Some(LocalCommand::ExportHistory(
                HistoryPreviewOptions {
                    count: 6,
                    role_filter: HistoryRoleFilter::All,
                    full: false,
                    grep: None,
                },
                Some(PathBuf::from("dump.txt")),
            ))
        );
        assert_eq!(
            parse_local_command("/history last").unwrap(),
            Some(LocalCommand::RenderLastHistoryMessage)
        );
        assert_eq!(
            parse_local_command("/history copy").unwrap(),
            Some(LocalCommand::CopyHistory(HistoryPreviewOptions {
                count: 6,
                role_filter: HistoryRoleFilter::All,
                full: false,
                grep: None,
            }))
        );
        assert_eq!(
            parse_local_command("/history rewind u3").unwrap(),
            Some(LocalCommand::RewindHistory(
                HistoryRewindTarget::UserOrdinal(3)
            ))
        );
        assert_eq!(
            parse_local_command("/history rewind last").unwrap(),
            Some(LocalCommand::RewindHistory(HistoryRewindTarget::LatestUser))
        );
        assert_eq!(
            parse_local_command("/history rewind grep wrong turn").unwrap(),
            Some(LocalCommand::RewindHistory(HistoryRewindTarget::Grep(
                "wrong turn".to_string()
            )))
        );
        assert_eq!(parse_local_command("hello").unwrap(), None);
    }

    #[test]
    fn parse_history_preview_options_supports_full_and_role_filters() {
        assert_eq!(
            parse_history_preview_options(&["full"]).unwrap(),
            (
                HistoryPreviewOptions {
                    count: 20,
                    role_filter: HistoryRoleFilter::All,
                    full: true,
                    grep: None,
                },
                HistoryAction::Show,
            )
        );
        assert_eq!(
            parse_history_preview_options(&["assistant", "8"]).unwrap(),
            (
                HistoryPreviewOptions {
                    count: 8,
                    role_filter: HistoryRoleFilter::Assistant,
                    full: false,
                    grep: None,
                },
                HistoryAction::Show,
            )
        );
    }

    #[test]
    fn summarize_history_content_flattens_and_truncates() {
        let content = Value::String("hello\nfrom\tterminal history".to_string());
        let summarized = summarize_history_content(&content, 12, None);
        assert_eq!(summarized, "hello from t...");
        assert_eq!(truncate_for_terminal("short", 10), "short");
    }

    #[test]
    fn highlight_history_keyword_marks_matches_with_ansi() {
        let highlighted = highlight_history_keyword("panic in fetch_chat_messages", Some("panic"));
        assert!(highlighted.contains("\x1b[1;93mpanic\x1b[0m"));
    }

    #[test]
    fn highlight_history_keyword_marks_all_matches() {
        let highlighted = highlight_history_keyword("panic then Panic again", Some("panic"));
        assert_eq!(highlighted.matches("\x1b[1;93m").count(), 2);
    }

    #[test]
    fn render_history_preview_reads_recent_session_messages() {
        let history_path =
            std::env::temp_dir().join(format!("ai-history-preview-{}.sqlite", Uuid::new_v4()));
        let mut app = test_app();
        app.session_history_file = history_path.clone();

        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Value::String("first question".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("first answer".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("second question".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        append_history_messages(&history_path, &messages).unwrap();

        let rendered = render_history_preview(
            &app,
            HistoryPreviewOptions {
                count: 2,
                role_filter: HistoryRoleFilter::All,
                full: false,
                grep: None,
            },
        )
        .unwrap();
        assert!(rendered.contains("[history] Showing 2 recent message(s):"));
        assert!(rendered.contains("[assistant] first answer"));
        assert!(rendered.contains("[user] second question"));
        assert!(!rendered.contains("first question"));

        let _ = std::fs::remove_file(history_path);
    }

    #[test]
    fn render_history_preview_filters_user_messages() {
        let history_path =
            std::env::temp_dir().join(format!("ai-history-filter-{}.sqlite", Uuid::new_v4()));
        let mut app = test_app();
        app.session_history_file = history_path.clone();

        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Value::String("first user".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("assistant reply".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("second user".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        append_history_messages(&history_path, &messages).unwrap();

        let rendered = render_history_preview(
            &app,
            HistoryPreviewOptions {
                count: 5,
                role_filter: HistoryRoleFilter::User,
                full: false,
                grep: None,
            },
        )
        .unwrap();
        assert!(rendered.contains("recent user message(s)"));
        assert!(rendered.contains("[user] first user"));
        assert!(rendered.contains("(u1)"));
        assert!(rendered.contains("[user] second user"));
        assert!(rendered.contains("(u2)"));
        assert!(!rendered.contains("assistant reply"));

        let _ = std::fs::remove_file(history_path);
    }

    #[test]
    fn render_history_replay_returns_last_assistant_conclusion() {
        let history_path =
            std::env::temp_dir().join(format!("ai-history-replay-{}.sqlite", Uuid::new_v4()));
        let mut app = test_app();
        app.session_history_file = history_path.clone();

        // 第一轮：工具调用步骤（带 tool_calls、空 content）+ 最终结论。
        let tool_step = Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: Some("thinking about the file".to_string()),
        };
        let messages = vec![
            test_message("user", "please check the file"),
            tool_step,
            test_message("tool", "file contents here"),
            test_message("assistant", "the final answer is 42"),
            test_message("user", "follow up question"),
            test_message("assistant", "and the follow-up answer"),
        ];
        append_history_messages(&history_path, &messages).unwrap();

        let replayed = render_history_replay(&app).unwrap();
        assert_eq!(replayed, "and the follow-up answer");

        let _ = std::fs::remove_file(history_path);
    }

    #[test]
    fn last_assistant_conclusion_text_preserves_full_markdown_body() {
        let history_path =
            std::env::temp_dir().join(format!("ai-history-last-{}.sqlite", Uuid::new_v4()));
        let mut app = test_app();
        app.session_history_file = history_path.clone();

        let markdown = "# Title\n\n- first\n- second\n\n```rust\nfn main() {}\n```".to_string();
        append_history_messages(
            &history_path,
            &[
                test_message("user", "show me the last answer"),
                test_message("assistant", &markdown),
            ],
        )
        .unwrap();

        let replayed = last_assistant_conclusion_text(&app).unwrap();
        assert_eq!(replayed.as_deref(), Some(markdown.as_str()));

        let _ = std::fs::remove_file(history_path);
    }

    #[test]
    fn render_history_replay_skips_tool_call_steps_and_picks_prior_conclusion() {
        let history_path =
            std::env::temp_dir().join(format!("ai-history-replay-prior-{}.sqlite", Uuid::new_v4()));
        let mut app = test_app();
        app.session_history_file = history_path.clone();

        // 最后一轮只到工具调用步骤（尚未产出结论），replay 应回退到上一轮结论。
        let tool_step = Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        };
        let messages = vec![
            test_message("user", "first question"),
            test_message("assistant", "first conclusion"),
            test_message("user", "second question"),
            tool_step,
        ];
        append_history_messages(&history_path, &messages).unwrap();

        let replayed = render_history_replay(&app).unwrap();
        assert_eq!(replayed, "first conclusion");

        let _ = std::fs::remove_file(history_path);
    }

    #[test]
    fn history_rewind_removes_target_user_and_following_messages() {
        let history_path =
            std::env::temp_dir().join(format!("ai-history-rewind-{}.sqlite", Uuid::new_v4()));
        let mut app = test_app();
        app.session_history_file = history_path.clone();

        let messages = vec![
            test_message("user", "first user"),
            test_message("assistant", "first answer"),
            test_message("user", "bad middle user"),
            test_message("assistant", "bad answer"),
            test_message("user", "later user"),
        ];
        append_history_messages(&history_path, &messages).unwrap();

        let before_context =
            history::build_context_history(usize::MAX, history_path.as_path(), 0, 8, 4000, None)
                .unwrap();
        assert!(
            before_context
                .iter()
                .any(|message| searchable_history_content(&message.content) == "bad middle user")
        );

        let plan = plan_history_rewind(&app, HistoryRewindTarget::UserOrdinal(2)).unwrap();
        assert_eq!(plan.user_ordinal, 2);
        assert_eq!(plan.removed_messages, 3);
        apply_history_rewind(&mut app, plan).unwrap();

        let remaining = history::build_message_arr(usize::MAX, history_path.as_path()).unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(
            searchable_history_content(&remaining[0].content),
            "first user"
        );
        assert_eq!(
            searchable_history_content(&remaining[1].content),
            "first answer"
        );
        let after_context =
            history::build_context_history(usize::MAX, history_path.as_path(), 0, 8, 4000, None)
                .unwrap();
        assert_eq!(after_context.len(), 2);
        assert!(!after_context.iter().any(|message| {
            let content = searchable_history_content(&message.content);
            content.contains("bad middle") || content.contains("later user")
        }));

        let _ = std::fs::remove_file(history_path);
    }

    #[test]
    fn history_rewind_removes_all_following_messages_across_roles() {
        let history_path =
            std::env::temp_dir().join(format!("ai-history-rewind-roles-{}.sqlite", Uuid::new_v4()));
        let mut app = test_app();
        app.session_history_file = history_path.clone();

        let messages = vec![
            test_message("user", "keep user"),
            test_message("assistant", "keep assistant"),
            test_message("user", "rewind me"),
            test_message("assistant", "assistant residue"),
            test_message("tool", "tool residue"),
            test_message("system", "system residue"),
            test_message("assistant", "final residue"),
        ];
        append_history_messages(&history_path, &messages).unwrap();

        let plan = plan_history_rewind(&app, HistoryRewindTarget::UserOrdinal(2)).unwrap();
        assert_eq!(plan.removed_messages, 5);
        apply_history_rewind(&mut app, plan).unwrap();

        let remaining = history::build_message_arr(usize::MAX, history_path.as_path()).unwrap();
        assert_eq!(
            remaining
                .iter()
                .map(|message| (
                    message.role.as_str(),
                    searchable_history_content(&message.content)
                ))
                .collect::<Vec<_>>(),
            vec![
                ("user", "keep user".to_string()),
                ("assistant", "keep assistant".to_string()),
            ]
        );

        let _ = std::fs::remove_file(history_path);
    }

    #[test]
    fn history_rewind_cancels_reflection_daemons_and_clears_runtime_state() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let history_path = std::env::temp_dir().join(format!(
            "ai-history-rewind-runtime-{}.sqlite",
            Uuid::new_v4()
        ));
        let mut app = test_app();
        app.session_history_file = history_path.clone();
        app.session_id = "sess-rewind".to_string();
        app.forced_skill = Some("demo-skill".to_string());
        app.forced_question = Some("follow this branch".to_string());
        app.attached_image_files = vec!["/tmp/demo.png".to_string()];
        app.last_skill_bias = Some(SkillBiasMemory {
            skill_name: "demo-skill".to_string(),
            question: "follow this branch".to_string(),
        });
        app.agent_context = Some(AgentContext {
            tools: vec![ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "read_file".to_string(),
                    description: String::new(),
                    parameters: serde_json::json!({}),
                },
            }],
            ..AgentContext::default()
        });
        crate::ai::tools::enable_tools::set_explicit_enabled_tool_names(vec![
            "mcp_feishu_doc_create_from_markdown".to_string(),
        ]);

        append_history_messages(
            &history_path,
            &[
                test_message("user", "keep user"),
                test_message("assistant", "keep assistant"),
                test_message("user", "rewind user"),
                test_message("assistant", "assistant residue"),
            ],
        )
        .unwrap();

        let daemon_handle = {
            let mut os = app.os.lock().unwrap();
            let current_pid = os.current_process_id();
            let (handle, _token) = os.daemon_register(
                "self_reflection:sess-rewind".to_string(),
                DaemonKind::Reflection,
                current_pid,
            );
            handle
        };

        let plan = plan_history_rewind(&app, HistoryRewindTarget::UserOrdinal(2)).unwrap();
        apply_history_rewind(&mut app, plan).unwrap();

        let daemon_state = {
            let os = app.os.lock().unwrap();
            os.daemon_status(daemon_handle)
                .map(|entry| entry.state)
                .expect("daemon should still be registered")
        };
        assert_eq!(daemon_state, DaemonState::Cancelled);
        assert!(app.forced_skill.is_none());
        assert!(app.forced_question.is_none());
        assert!(app.attached_image_files.is_empty());
        assert!(app.last_skill_bias.is_none());
        assert!(
            app.agent_context
                .as_ref()
                .is_some_and(|ctx| ctx.tools.is_empty())
        );
        assert!(crate::ai::tools::enable_tools::explicit_enabled_tool_names().is_empty());

        let remaining = history::build_message_arr(usize::MAX, history_path.as_path()).unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(
            remaining
                .iter()
                .map(|message| searchable_history_content(&message.content))
                .collect::<Vec<_>>(),
            vec!["keep user".to_string(), "keep assistant".to_string()]
        );

        let _ = std::fs::remove_file(history_path);
    }

    #[test]
    fn history_rewind_grep_requires_unique_user_match() {
        let history_path =
            std::env::temp_dir().join(format!("ai-history-rewind-grep-{}.sqlite", Uuid::new_v4()));
        let mut app = test_app();
        app.session_history_file = history_path.clone();

        let messages = vec![
            test_message("user", "fix login panic"),
            test_message("assistant", "ok"),
            test_message("user", "fix cache panic"),
        ];
        append_history_messages(&history_path, &messages).unwrap();

        let err = plan_history_rewind(&app, HistoryRewindTarget::Grep("panic".to_string()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("user inputs match"));
        assert!(err.contains("u1"));
        assert!(err.contains("u2"));

        let plan =
            plan_history_rewind(&app, HistoryRewindTarget::Grep("cache".to_string())).unwrap();
        assert_eq!(plan.user_ordinal, 2);
        assert_eq!(plan.removed_messages, 1);

        let _ = std::fs::remove_file(history_path);
    }

    #[test]
    fn render_history_preview_supports_grep_and_tool_role() {
        let history_path =
            std::env::temp_dir().join(format!("ai-history-grep-{}.sqlite", Uuid::new_v4()));
        let mut app = test_app();
        app.session_history_file = history_path.clone();

        let messages = vec![
            Message {
                role: "tool".to_string(),
                content: Value::String("panic in fetch_messages".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "system".to_string(),
                content: Value::String("stable note".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        append_history_messages(&history_path, &messages).unwrap();

        let rendered = render_history_preview(
            &app,
            HistoryPreviewOptions {
                count: 5,
                role_filter: HistoryRoleFilter::Tool,
                full: false,
                grep: Some("panic".to_string()),
            },
        )
        .unwrap();
        assert!(rendered.contains("recent tool message(s) matching \"panic\""));
        assert!(rendered.contains("[tool] \x1b[1;93mpanic\x1b[0m in fetch_messages"));
        assert!(!rendered.contains("stable note"));

        let _ = std::fs::remove_file(history_path);
    }
}

fn prompt_user(app: &mut App) -> io::Result<Option<String>> {
    if let Some(editor) = app.prompt_editor.as_mut() {
        crate::ai::prompt::completion::CommandCompleter::set_current_model_hint(&app.current_model);
        editor.set_current_model_label(crate::ai::models::model_display_label(&app.current_model));
        // 设置 session 主题：从当前 session 的首条用户消息生成概括性标题。
        // 若 session 尚无用户消息（新 session），显示 "new session"。
        // SessionStore 需要 session 文件所在目录（而非 history 文件本身）。
        let sessions_dir = app
            .session_history_file
            .parent()
            .unwrap_or_else(|| app.session_history_file.as_path());
        let store = crate::ai::history::SessionStore::new(sessions_dir);
        // 优先使用 LLM 生成的标题，fallback 到首条消息摘要
        let topic = match store.read_session_title(&app.session_id).ok().flatten() {
            Some(t) => Some(t),
            None => store
                .first_user_prompt(&app.session_id)
                .ok()
                .flatten()
                .as_deref()
                .map(crate::ai::history::generate_session_summary),
        };
        editor.set_session_topic(topic);
        return editor.read_multi_line();
    }

    let stdin = io::stdin();
    let mut stdin = stdin.lock();

    // 无 TUI 编辑器时，在输入提示前打印一行简短的模型提示。
    println!(
        "  {ACCENT_MUTED}[{ACCENT_SUCCESS}{}{ACCENT_MUTED}]{RESET}",
        crate::ai::models::model_display_label(&app.current_model),
    );

    let mut lines = Vec::new();
    loop {
        print!("  ");
        io::stdout().flush()?;
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => lines.push(trim_trailing_newline(line)),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                println!("Exit.");
                return Ok(None);
            }
            Err(err) => return Err(err),
        }
    }
    if lines.is_empty() {
        Ok(None)
    } else {
        Ok(Some(lines.join("\n")))
    }
}
