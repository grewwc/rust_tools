use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use clap::Parser;
use reqwest::blocking::Response;

use crate::{clipboard::string_content, strw::split::split_space_keep_symbol};

use super::{
    cli::{Cli, normalize_single_dash_long_opts},
    config, files,
    history::{COLON, NEWLINE, append_history},
    models,
    prompt::{PromptEditor, trim_trailing_newline},
    request, stream,
    types::{App, LoopOverrides, QuestionContext, StreamOutcome},
};
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse_from(normalize_single_dash_long_opts(std::env::args()));
    let config = config::load_config()?;

    if cli.clear {
        config::clear_history_file(&config.history_file);
        println!("History cleared.");
        return Ok(());
    }

    let shutdown = Arc::new(AtomicBool::new(false));
    let streaming = Arc::new(AtomicBool::new(false));
    let cancel_stream = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&shutdown);
    let streaming_flag = Arc::clone(&streaming);
    let cancel_stream_flag = Arc::clone(&cancel_stream);
    ctrlc::set_handler(move || {
        handle_sigint(
            signal_flag.as_ref(),
            streaming_flag.as_ref(),
            cancel_stream_flag.as_ref(),
        );
    })?;

    let writer = config::open_output_writer(cli.out.as_deref())?;
    let current_model = models::initial_model(&cli);
    let client = reqwest::blocking::Client::builder().build()?;
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt_editor = if cli.args.is_empty() {
        Some(PromptEditor::new())
    } else {
        None
    };

    let mut app = App {
        pending_files: if cli.files.trim().is_empty() {
            None
        } else {
            Some(cli.files.clone())
        },
        pending_clipboard: cli.clipboard,
        pending_short_output: cli.short_output,
        current_model,
        raw_args,
        cli,
        config,
        client,
        attached_image_files: Vec::new(),
        attached_binary_files: Vec::new(),
        uploaded_file_ids: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        writer,
        prompt_editor,
    };

    run_loop(&mut app)
}

pub(super) fn handle_sigint(
    shutdown: &AtomicBool,
    streaming: &AtomicBool,
    cancel_stream: &AtomicBool,
) {
    if streaming.load(Ordering::SeqCst) {
        cancel_stream.store(true, Ordering::SeqCst);
        return;
    }

    shutdown.store(true, Ordering::SeqCst);
}

fn run_loop(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    let mut should_quit = !app.cli.args.is_empty();
    let mut replied_once = false;
    loop {
        if app.shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        let Some(ctx) = next_question(app)? else {
            return Ok(());
        };
        if ctx.question.trim().is_empty() {
            should_quit = false;
            continue;
        }

        let mut question = ctx.question;
        let next_model = resolve_model_for_input(app, &mut question);
        app.current_model = next_model.clone();

        app.cancel_stream.store(false, Ordering::SeqCst);
        let mut current_history = format!("user{COLON}{question}{NEWLINE}assistant{COLON}");
        let mut response = request::do_request(app, &next_model, &question, ctx.history_count)?;
        request::print_info(&next_model);
        app.streaming.store(true, Ordering::SeqCst);
        let outcome = match stream::stream_response(app, &mut response, &mut current_history) {
            Ok(outcome) => outcome,
            Err(err) => {
                app.streaming.store(false, Ordering::SeqCst);
                return Err(err);
            }
        };
        app.streaming.store(false, Ordering::SeqCst);
        if outcome == StreamOutcome::Cancelled {
            println!("\nInterrupted.");
            if should_quit {
                return Ok(());
            }
            continue;
        }
        if app.shutdown.load(Ordering::SeqCst) {
            println!();
            return Ok(());
        }
        drain_response(&mut response)?;
        current_history.push(NEWLINE);
        append_history(&app.config.history_file, &current_history)?;
        println!();

        if should_quit {
            return Ok(());
        }
        if let Some(writer) = app.writer.as_mut() {
            writer.write_all(b"\n---\n")?;
            writer.flush()?;
        }

        if should_pause_after_reply(app, replied_once) {
            wait_for_next_input(app)?;
        }
        replied_once = true;
    }
}

fn should_pause_after_reply(app: &App, replied_once: bool) -> bool {
    app.cli.multi_line && app.prompt_editor.is_some() && replied_once
}

fn wait_for_next_input(app: &mut App) -> io::Result<()> {
    use std::io::{IsTerminal, Write};

    use crossterm::{
        event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
        terminal::{disable_raw_mode, enable_raw_mode},
    };

    if !io::stdout().is_terminal() {
        return Ok(());
    }

    print!("\x1b[2m(按 Enter 开始下一次输入)\x1b[0m");
    io::stdout().flush()?;

    enable_raw_mode()?;
    let outcome: io::Result<()> = (|| loop {
        match event::read().map_err(|e| io::Error::other(e.to_string()))? {
            Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
                match (key.code, key.modifiers) {
                    (KeyCode::Enter, _) => break Ok(()),
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                        app.shutdown.store(true, Ordering::SeqCst);
                        break Ok(());
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    })();
    let _ = disable_raw_mode();
    println!();
    outcome
}

fn drain_response(response: &mut Response) -> Result<(), Box<dyn std::error::Error>> {
    response.copy_to(&mut io::sink())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::AppConfig;
    use std::path::PathBuf;

    #[test]
    fn pause_after_reply_requires_multiline_interactive_and_not_first() {
        let cli = Cli {
            history: 0,
            model: String::new(),
            multi_line: true,
            code: false,
            deepseek: false,
            clear: false,
            clipboard: false,
            no_history: false,
            files: String::new(),
            out: None,
            raw: false,
            thinking: false,
            short_output: false,
            model_0: false,
            model_1: false,
            model_2: false,
            model_3: false,
            model_4: false,
            model_5: false,
            model_6: false,
            args: Vec::new(),
        };

        let mut app = App {
            pending_files: None,
            pending_clipboard: false,
            pending_short_output: false,
            current_model: String::new(),
            raw_args: String::new(),
            cli,
            config: AppConfig {
                api_key: String::new(),
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
            },
            client: reqwest::blocking::Client::builder().build().unwrap(),
            attached_image_files: Vec::new(),
            attached_binary_files: Vec::new(),
            uploaded_file_ids: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            writer: None,
            prompt_editor: Some(PromptEditor::new()),
        };

        assert!(!should_pause_after_reply(&app, false));
        assert!(should_pause_after_reply(&app, true));
        app.cli.multi_line = false;
        assert!(!should_pause_after_reply(&app, true));
        app.cli.multi_line = true;
        app.prompt_editor = None;
        assert!(!should_pause_after_reply(&app, true));
    }
}

fn next_question(app: &mut App) -> Result<Option<QuestionContext>, Box<dyn std::error::Error>> {
    if !app.cli.args.is_empty() {
        let base_question = if app.cli.raw {
            app.raw_args.clone()
        } else {
            let question = app.cli.args.join(" ");
            app.cli.args.clear();
            question
        };
        app.cli.args.clear();
        let ctx = finalize_question(
            app,
            base_question,
            base_history_count(app.cli.history, app.cli.no_history),
            false,
        )?;
        return Ok(Some(ctx));
    }

    let Some(question) = prompt_user(app)? else {
        return Ok(None);
    };
    let overrides = loop_overrides(&question);
    let history_count = overrides
        .history_count
        .unwrap_or_else(|| base_history_count(app.cli.history, app.cli.no_history));
    let ctx = finalize_question(app, question, history_count, overrides.short_output)?;
    Ok(Some(ctx))
}

fn base_history_count(history: usize, no_history: bool) -> usize {
    if no_history { 0 } else { history }
}

fn finalize_question(
    app: &mut App,
    mut question: String,
    history_count: usize,
    loop_short_output: bool,
) -> Result<QuestionContext, Box<dyn std::error::Error>> {
    if let Some(files) = app.pending_files.take() {
        let parsed = files::parse_files(&files);
        if !parsed.text_files.is_empty() {
            let prefix = files::text_file_contents(&parsed.text_files)?;
            if !prefix.is_empty() {
                question = format!("{prefix}\n{question}");
            }
        }
        if !parsed.image_files.is_empty() {
            app.attached_image_files = parsed.image_files;
        }
        if !parsed.binary_files.is_empty() {
            app.attached_binary_files = parsed.binary_files;
        }
    }

    if app.pending_clipboard {
        let clipboard = string_content::get_clipboard_content();
        question = format!("{clipboard}{question}");
        app.pending_clipboard = false;
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

fn prompt_user(app: &mut App) -> io::Result<Option<String>> {
    if let Some(editor) = app.prompt_editor.as_mut() {
        if app.cli.multi_line {
            return editor.read_multi_line();
        }
        return editor.read_single_line();
    }

    let multiline = app.cli.multi_line;
    let stdin = io::stdin();
    let mut stdin = stdin.lock();

    if !multiline {
        print!("> ");
        io::stdout().flush()?;
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(trim_trailing_newline(line))),
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                println!("Exit.");
                Ok(None)
            }
            Err(err) => Err(err),
        }
    } else {
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
}

pub(super) fn loop_overrides(question: &str) -> LoopOverrides {
    let tokens = split_space_keep_symbol(question, "\"'").collect::<Vec<_>>();
    let short_output = tokens.iter().any(|token| *token == "-s");

    if tokens.iter().any(|token| *token == "-x") {
        return LoopOverrides {
            short_output,
            history_count: Some(0),
        };
    }

    let mut history_count = None;
    let mut idx = 0usize;
    while idx < tokens.len() {
        if tokens[idx] == "--history" {
            if let Some(next) = tokens.get(idx + 1)
                && let Ok(value) = next.parse::<usize>()
            {
                history_count = Some(value);
                break;
            }
        }
        idx += 1;
    }

    LoopOverrides {
        short_output,
        history_count,
    }
}

fn resolve_model_for_input(app: &App, question: &mut String) -> String {
    if let Some(model) = attachment_forced_model(
        &app.current_model,
        !app.attached_image_files.is_empty(),
        !app.attached_binary_files.is_empty(),
        &app.config.vl_default_model,
    ) {
        return model;
    }

    let trimmed = question.trim_end().to_string();
    if let Some(stripped) = trimmed.strip_suffix(" -code") {
        *question = stripped.to_string();
        return models::qwen_coder_plus_latest().to_string();
    }
    if let Some(stripped) = trimmed.strip_suffix(" -d") {
        *question = stripped.to_string();
        return models::deepseek_v3().to_string();
    }
    if let Some(selector) = trailing_model_selector(&trimmed) {
        if trimmed.len() >= 3 {
            *question = trimmed[..trimmed.len() - 3].trim_end().to_string();
        }
        return models::model_from_selector(selector, app.cli.thinking)
            .as_str()
            .to_string();
    }
    app.current_model.clone()
}

pub(super) fn attachment_forced_model(
    current_model: &str,
    has_image_files: bool,
    has_binary_files: bool,
    vl_default_model: &str,
) -> Option<String> {
    if current_model == models::qwen_long() {
        return Some(models::qwen_long().to_string());
    }
    if has_binary_files {
        return Some(models::qwen_long().to_string());
    }
    if has_image_files && !models::is_vl_model(current_model) {
        return Some(models::determine_vl_model(vl_default_model));
    }
    None
}

pub(super) fn trailing_model_selector(input: &str) -> Option<u8> {
    let bytes = input.as_bytes();
    if bytes.len() < 3 {
        return None;
    }
    let dash_idx = bytes.len() - 2;
    if bytes[dash_idx] != b'-' || !bytes[dash_idx + 1].is_ascii_digit() {
        return None;
    }
    if dash_idx == 0 || bytes[dash_idx - 1] != b' ' {
        return None;
    }
    Some(bytes[dash_idx + 1] - b'0')
}
