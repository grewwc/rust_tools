use std::error::Error;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use regex::RegexBuilder;

use super::params::parse_loop_overrides;
use crate::ai::history;
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
        "/history usage:\n  /history [N]           Show last N messages (default: {})\n  /history full          Show full messages\n  /history user/assistant/tool/system\n                         Filter by role\n  /history grep <keyword>  Search messages\n  /history export [path] Export to file\n  /history copy          Copy to clipboard\n  /history help            Show this help",
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
        let (question, overrides) = parse_loop_overrides(&base_question);
        let history_count = overrides.history_count.unwrap_or(app.cli.history);
        let ctx = finalize_question(app, question, history_count, overrides.short_output)?;
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
                app.shutdown.store(true, Ordering::Relaxed);
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
        app.shutdown.store(true, Ordering::Relaxed);
        return Ok(None);
    };
    let (question, overrides) = parse_loop_overrides(&question);
    let history_count = overrides.history_count.unwrap_or(app.cli.history);
    let ctx = finalize_question(app, question, history_count, overrides.short_output)?;
    Ok(Some(ctx))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LocalCommand {
    ShowHistory(HistoryPreviewOptions),
    HelpHistory,
    ExportHistory(HistoryPreviewOptions, Option<PathBuf>),
    CopyHistory(HistoryPreviewOptions),
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

fn handle_local_command(app: &App, input: &str) -> io::Result<bool> {
    match handle_local_command_inner(app, input) {
        Ok(v) => Ok(v),
        Err(e) => {
            eprintln!("local command error: {e}");
            Ok(false)
        }
    }
}

fn handle_local_command_inner(app: &App, input: &str) -> Result<bool, Box<dyn Error>> {
    let Some(command) = parse_local_command(input)? else {
        return Ok(false);
    };

    match command {
        LocalCommand::ShowHistory(options) => {
            println!("{}", render_history_preview(app, options)?);
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
    let (options, action) = parse_history_preview_options(args)?;
    Ok(Some(match action {
        HistoryAction::Show => LocalCommand::ShowHistory(options),
        HistoryAction::Help => LocalCommand::HelpHistory,
        HistoryAction::Export(path) => LocalCommand::ExportHistory(options, path.map(PathBuf::from)),
        HistoryAction::Copy => LocalCommand::CopyHistory(options),
    }))
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
    let mut out = format!("[history] Showing {} recent {}{}:\n", total, label, grep_suffix);
    for (idx, message) in shown.iter().enumerate() {
        let content = summarize_history_content(&message.content, max_chars, grep.as_deref());
        out.push_str(&format!(
            "{}. [{}] {}\n",
            idx + 1,
            message.role,
            content
        ));
    }
    Ok(out.trim_end().to_string())
}

fn collect_history_messages(
    app: &App,
    options: HistoryPreviewOptions,
) -> Result<Vec<history::Message>, Box<dyn Error>> {
    let history_file = active_history_path(app);
    let messages =
        history::build_message_arr(HISTORY_PREVIEW_MAX_COUNT.max(options.count), &history_file)?;
    let filtered = messages
        .into_iter()
        .filter(|message| match options.role_filter {
            HistoryRoleFilter::All => true,
            HistoryRoleFilter::User => message.role == "user",
            HistoryRoleFilter::Assistant => message.role == "assistant",
            HistoryRoleFilter::Tool => message.role == "tool",
            HistoryRoleFilter::System => crate::ai::history::is_system_like_role(&message.role),
        })
        .filter(|message| {
            options.grep.as_deref().is_none_or(|needle| {
                let haystack = searchable_history_content(&message.content).to_ascii_lowercase();
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
        other => serde_json::to_string(other).unwrap_or_else(|_| "<non-string content>".to_string()),
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
                HISTORY_GREP_HIGHLIGHT_START,
                &caps[0],
                HISTORY_GREP_HIGHLIGHT_END
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
            && !parsed.image_files.iter().any(|candidate| candidate == &path)
            && !parsed.binary_files.iter().any(|candidate| candidate == &path)
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

fn apply_text_files_prefix(
    question: &mut String,
    text_files: &[String],
) -> Result<(), Box<dyn Error>> {
    if text_files.is_empty() {
        return Ok(());
    }
    let prefix = files::text_file_contents(text_files)?;
    if !prefix.is_empty() {
        *question = format!("{prefix}\n{question}");
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
    question: &mut String,
    binary_files: Vec<String>,
) -> Result<(), Box<dyn Error>> {
    if binary_files.is_empty() {
        return Ok(());
    }

    let (pdfs, unsupported) = split_pdf_files(binary_files);
    if !pdfs.is_empty() {
        let prefix = build_pdf_text_prefix(&pdfs);
        if !prefix.trim().is_empty() {
            *question = format!("{prefix}\n{question}");
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
    loop_short_output: bool,
) -> Result<QuestionContext, Box<dyn Error>> {
    let inline_files = extract_at_file_references(&mut question);
    let mut inline_images = extract_inline_image_paths(&mut question);
    apply_text_files_prefix(&mut question, &inline_files.text_files)?;
    if !inline_files.image_files.is_empty() {
        inline_images.extend(inline_files.image_files);
    }
    handle_binary_files(&mut question, inline_files.binary_files)?;
    if let Some(files) = app.pending_files.take() {
        let parsed = files::parse_files(&files);
        apply_text_files_prefix(&mut question, &parsed.text_files)?;
        if !parsed.image_files.is_empty() {
            inline_images.extend(parsed.image_files);
        }
        handle_binary_files(&mut question, parsed.binary_files)?;
    }
    if !inline_images.is_empty() {
        app.attached_image_files = inline_images;
    }

    if app.pending_short_output || loop_short_output {
        if !question.ends_with('\n') {
            question.push('\n');
        }
        question.push_str("Be Concise.");
        app.pending_short_output = false;
    }

    Ok(QuestionContext {
        question,
        history_count,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        extract_at_file_references, finalize_question, parse_history_preview_options,
        highlight_history_keyword, parse_local_command, render_history_preview,
        summarize_history_content, truncate_for_terminal, HistoryAction,
        HistoryPreviewOptions, HistoryRoleFilter, LocalCommand,
    };
    use crate::ai::{
        history::{Message, append_history_messages},
        types::{App, AppConfig},
    };
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};
    use uuid::Uuid;

    fn any_model_name() -> String {
        crate::ai::model_names::all()
            .first()
            .map(|m| m.name.clone())
            .expect("models.json is empty")
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
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: any_vl_model_name(),
                history_max_chars: 12000,
                history_keep_last: 8,
                history_summary_max_chars: 4000,
                intent_model: None,
                intent_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("config/intent/intent_model.json"),
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            client,
            current_model: any_model_name(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            pending_short_output: false,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            writer: None,
            prompt_editor: None,
            agent_context: None,
            last_skill_bias: None,
            agent_reload_counter: None,
        }
    }

    #[test]
    fn at_image_reference_is_attached_and_removed_from_question() {
        let path = std::env::temp_dir().join(format!("ai-image-{}.png", Uuid::new_v4()));
        std::fs::write(&path, b"fake").unwrap();

        let mut app = test_app();
        let question = format!("Please inspect @{} now", path.display());
        let ctx = finalize_question(&mut app, question, 6, false).unwrap();

        assert!(!ctx.question.contains(path.to_string_lossy().as_ref()));
        assert_eq!(app.attached_image_files, vec![path.to_string_lossy().to_string()]);
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
        let ctx = finalize_question(&mut app, question, 6, false).unwrap();

        assert!(ctx.question.contains("hello from file"));
        assert!(!ctx.question.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn pending_files_image_is_attached_for_dash_f_flow() {
        let path = std::env::temp_dir().join(format!("ai-pending-image-{}.png", Uuid::new_v4()));
        std::fs::write(&path, b"fake").unwrap();

        let mut app = test_app();
        app.pending_files = Some(path.to_string_lossy().to_string());
        let ctx = finalize_question(&mut app, "describe this".to_string(), 6, false).unwrap();

        assert_eq!(ctx.question, "describe this");
        assert_eq!(app.attached_image_files, vec![path.to_string_lossy().to_string()]);
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
            parse_local_command("/history copy").unwrap(),
            Some(LocalCommand::CopyHistory(HistoryPreviewOptions {
                count: 6,
                role_filter: HistoryRoleFilter::All,
                full: false,
                grep: None,
            }))
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
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("first answer".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("second question".to_string()),
                tool_calls: None,
                tool_call_id: None,
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
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("assistant reply".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("second user".to_string()),
                tool_calls: None,
                tool_call_id: None,
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
        assert!(rendered.contains("[user] second user"));
        assert!(!rendered.contains("assistant reply"));

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
            },
            Message {
                role: "system".to_string(),
                content: Value::String("stable note".to_string()),
                tool_calls: None,
                tool_call_id: None,
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
        return editor.read_multi_line();
    }

    let stdin = io::stdin();
    let mut stdin = stdin.lock();

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
