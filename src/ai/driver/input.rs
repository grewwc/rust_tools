use std::error::Error;
use std::io::{self, BufRead, Write};
use std::sync::atomic::Ordering;

use super::params::parse_loop_overrides;
use crate::ai::types::{App, LoopOverrides, QuestionContext};

use crate::ai::{files, prompt::trim_trailing_newline};
use crate::clipboard::string_content;

pub fn next_question(app: &mut App) -> Result<Option<QuestionContext>, Box<dyn Error>> {
    if !app.cli.args.is_empty() {
        let base_question = if app.cli.raw {
            app.raw_args.clone()
        } else {
            let question = app.cli.args.join(" ");
            app.cli.args.clear();
            question
        };
        app.cli.args.clear();
        let (question, overrides) = parse_loop_overrides(&base_question);
        let history_count = overrides
            .history_count
            .unwrap_or_else(|| base_history_count(app.cli.history, app.cli.no_history));
        let ctx = finalize_question(app, question, history_count, overrides.short_output)?;
        return Ok(Some(ctx));
    }

    let question = match prompt_user(app) {
        Ok(v) => v,
        Err(_) if app.shutdown.load(Ordering::Acquire) => {
            return Ok(None);
        }
        Err(err) => return Err(err.into()),
    };
    let Some(question) = question else {
        app.shutdown.store(true, Ordering::Release);
        return Ok(None);
    };
    let (question, overrides) = parse_loop_overrides(&question);
    let history_count = overrides
        .history_count
        .unwrap_or_else(|| base_history_count(app.cli.history, app.cli.no_history));
    let ctx = finalize_question(app, question, history_count, overrides.short_output)?;
    Ok(Some(ctx))
}

pub fn base_history_count(history: usize, no_history: bool) -> usize {
    if no_history { 0 } else { history }
}

pub fn finalize_question(
    app: &mut App,
    mut question: String,
    history_count: usize,
    loop_short_output: bool,
) -> Result<QuestionContext, Box<dyn Error>> {
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

pub fn prompt_user(app: &mut App) -> io::Result<Option<String>> {
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
