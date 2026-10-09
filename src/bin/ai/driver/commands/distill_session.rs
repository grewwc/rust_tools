//! Interactive entry point for the same verified distillation used by the CLI.

use std::future::Future;

use crate::ai::{history::SessionStore, tools::service::session_distill, types::App};

const USAGE: &str = "Usage: /distill-session [session-id|archive.zip] [--dry-run] [--limit N]\n\
    Omit the source to distill the current session.\n\
    Save verified conclusions to the current persona's knowledge base.\n\
    --dry-run previews without saving; --limit must be 1..100. Quote paths containing spaces.";

pub(crate) const FLAGS: &[&str] = &["--dry-run", "--limit", "-h", "--help"];

#[derive(Debug, PartialEq, Eq)]
struct Options {
    source: Option<String>,
    limit: usize,
    dry_run: bool,
}

fn command_args(input: &str) -> Option<&str> {
    let input = input.trim();
    let rest = input
        .strip_prefix('/')
        .or_else(|| input.strip_prefix(':'))?;
    let (name, args) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    (name == "distill-session").then_some(args.trim())
}

/// Parse quotes and escaped whitespace without shell expansion or execution.
fn words(args: &str) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let mut word = String::new();
    let mut quote = None;
    let mut started = false;
    let mut chars = args.chars().peekable();
    while let Some(ch) = chars.next() {
        match (quote, ch) {
            (Some(q), c) if q == c => quote = None,
            (None, '\'' | '"') => {
                quote = Some(ch);
                started = true;
            }
            (None | Some('"'), '\\') => {
                let next = chars.next().ok_or("Trailing escape in command arguments")?;
                if quote.is_some() && !matches!(next, '"' | '\\') {
                    word.push('\\');
                }
                word.push(next);
                started = true;
            }
            (None, c) if c.is_whitespace() => {
                if started {
                    result.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            _ => {
                word.push(ch);
                started = true;
            }
        }
    }
    if quote.is_some() {
        return Err("Unclosed quote in command arguments".into());
    }
    if started {
        result.push(word);
    }
    Ok(result)
}

fn parse_options(args: &str, default_limit: usize) -> Result<Option<Options>, String> {
    let words = words(args)?;
    let mut iter = words.iter();
    let mut source = None;
    let mut limit = None;
    let mut dry_run = false;
    let mut positional_only = false;
    while let Some(word) = iter.next() {
        if !positional_only {
            match word.as_str() {
                "--" => {
                    positional_only = true;
                    continue;
                }
                "-h" | "--help" => return Ok(None),
                "--dry-run" => {
                    dry_run = true;
                    continue;
                }
                "--limit" => {
                    if limit.is_some() {
                        return Err("Duplicate --limit".into());
                    }
                    limit = Some(parse_limit(
                        iter.next().ok_or("Missing value for --limit")?,
                    )?);
                    continue;
                }
                _ => {}
            }
            if let Some(value) = word.strip_prefix("--limit=") {
                if limit.is_some() {
                    return Err("Duplicate --limit".into());
                }
                limit = Some(parse_limit(value)?);
                continue;
            }
            if word.starts_with('-') {
                return Err(format!("Unknown option: {word}"));
            }
        }
        if word.is_empty() {
            return Err("Source must not be empty".into());
        }
        if source.replace(word.clone()).is_some() {
            return Err("Expected exactly one session ID or archive path".into());
        }
    }
    Ok(Some(Options {
        source,
        limit: limit.unwrap_or(default_limit),
        dry_run,
    }))
}

fn parse_limit(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|n| (1..=100).contains(n))
        .ok_or_else(|| "--limit must be an integer in 1..100".into())
}

/// Keep recognition and validation ahead of any model request or storage access.
async fn dispatch<F, Fut>(
    input: &str,
    default_limit: usize,
    run: F,
) -> Option<Result<String, String>>
where
    F: FnOnce(Options) -> Fut,
    Fut: Future<Output = Result<String, String>>,
{
    let args = command_args(input)?;
    Some(match parse_options(args, default_limit) {
        Ok(Some(options)) => run(options).await,
        Ok(None) => Ok(USAGE.into()),
        Err(error) => Err(format!("{error}\n{USAGE}")),
    })
}

/// Run inside the foreground turn's persona/session scope, before conversation
/// persistence. A malformed command or failed distillation remains a local
/// command rather than falling through to the ordinary assistant turn.
pub(crate) async fn try_handle_distill_session_command(app: &App, input: &str) -> bool {
    let result = dispatch(input, app.cli.distill_limit, |options| async move {
        let source = match options.source.as_deref() {
            Some(value) => {
                let store = SessionStore::new(app.config.history_file.as_path());
                let cwd =
                    crate::ai::driver::runtime_ctx::effective_cwd().map_err(|e| e.to_string())?;
                session_distill::DistillInput::resolve(value, &cwd, &store)?
            }
            // Use the active binding directly: an ID passed through the explicit
            // source resolver could select a same-named archive in the cwd.
            None => session_distill::DistillInput::Session {
                id: app.session_id.clone(),
                path: app.session_history_file.clone(),
            },
        };
        println!(
            "Distilling {}{}...",
            source.path().display(),
            if options.dry_run {
                " (preview; no writes)"
            } else {
                ""
            }
        );
        let report = session_distill::run_distill_source_command(
            app,
            &source,
            options.limit,
            options.dry_run,
        )
        .await?;
        Ok(session_distill::format_report(&report, source.path()))
    })
    .await;
    match result {
        None => false,
        Some(Ok(report)) => {
            println!("{report}");
            true
        }
        Some(Err(error)) => {
            eprintln!("Distillation failed: {error}");
            true
        }
    }
}

#[cfg(test)]
#[path = "distill_session_tests.rs"]
mod tests;
