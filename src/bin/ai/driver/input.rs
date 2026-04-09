use std::error::Error;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::atomic::Ordering;

use super::params::parse_loop_overrides;
use crate::ai::types::{App, QuestionContext};

use crate::ai::{files, prompt::trim_trailing_newline};
use crate::pdfw::{PdfParseOptions, parse_pdf};

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
                break v;
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
    use super::{extract_at_file_references, finalize_question};
    use crate::ai::types::{App, AppConfig};
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
            Ok(_) => {
                let trimmed = trim_trailing_newline(line);
                // If we get an empty line, it might be a stray Enter from during streaming.
                // Return it as-is and let the caller decide what to do.
                Ok(Some(trimmed))
            }
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
