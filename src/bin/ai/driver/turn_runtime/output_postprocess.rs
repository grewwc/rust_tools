//! Display-only post-processing of the final assistant body before it is
//! painted to the terminal.
//!
//! When `ai.output.postprocess_command` is set, the completed body text is
//! piped through that shell command (stdin -> stdout) right before
//! `render_markdown_block`. This lets users fix cosmetic issues in the
//! terminal echo without touching canonical history (e.g.
//! `scripts/postprocess_terminal.py`, which converts Chinese punctuation
//! inside code / file-location contexts to ASCII).
//!
//! The filter is strictly best-effort: any failure (empty command, missing
//! python, non-zero exit, non-UTF8 output) falls back to the original text so
//! the turn can never be blocked or corrupted by post-processing.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ai::config_schema::AiConfig;

/// Timeout for the post-processing command (mirrors the lifecycle-hook
/// default; post-processing must never stall the turn end).
const POSTPROCESS_TIMEOUT_SECS: u64 = 30;

/// Monotonic counter used to make temp file names unique within the process.
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Post-process `text` for the terminal via the configured command.
///
/// Returns the transformed text on success, or the original `text` unchanged
/// when the feature is disabled or the command fails.
pub fn postprocess_terminal_text(text: String) -> String {
    let command = crate::commonw::configw::get_all_config()
        .get(AiConfig::OUTPUT_POSTPROCESS_COMMAND, "");
    postprocess_with_command(command.trim(), text)
}

/// Core implementation, split out so tests can exercise the command path
/// without depending on the ambient user configuration.
fn postprocess_with_command(command: &str, text: String) -> String {
    if text.is_empty() || command.is_empty() {
        return text;
    }

    let in_path = temp_file_path("in");
    let out_path = temp_file_path("out");
    let written = || -> std::io::Result<()> {
        let mut file = std::fs::File::create(&in_path)?;
        file.write_all(text.as_bytes())?;
        Ok(())
    };
    if written().is_err() {
        return text;
    }

    // Redirect stdin/stdout through temp files: the shared command runner
    // does not expose a stdin pipe, but its shell path honors `<`/`>`.
    let full_command = format!(
        "{} < {} > {}",
        command,
        shell_quote(&in_path),
        shell_quote(&out_path)
    );
    let result = crate::ai::tools::storage::command_runner::run_command(
        &full_command,
        None,
        POSTPROCESS_TIMEOUT_SECS,
    );

    let transformed = match result {
        Ok(output) if output.status.success() => {
            std::fs::read(&out_path)
                .ok()
                .and_then(|bytes| String::from_utf8(bytes).ok())
        }
        _ => None,
    };

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);

    match transformed {
        Some(out) if !out.is_empty() => out,
        _ => {
            eprintln!(
                "[output] postprocess command failed or produced no output; showing original text"
            );
            text
        }
    }
}

/// Build a unique temp path under the system temp dir.
fn temp_file_path(kind: &str) -> std::path::PathBuf {
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "a_postprocess_{}_{}_{}.txt",
        std::process::id(),
        counter,
        kind
    ))
}

/// Wrap a path in single quotes for safe use in a shell command line.
fn shell_quote(path: &std::path::Path) -> String {
    let raw = path.to_string_lossy();
    format!("'{}'", raw.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_command_returns_text_unchanged() {
        // With an empty command the function must be a no-op with zero side
        // effects (no subprocess, no temp files).
        let text = "正文：`println！（\"hi\"）` 在 src/main，rs".to_string();
        assert_eq!(postprocess_with_command("", text.clone()), text);
    }

    #[test]
    fn failing_command_falls_back_to_original() {
        // A command that cannot be spawned must fall back to the original
        // text so the turn is never blocked by post-processing.
        let text = "原样保留：main。rs".to_string();
        assert_eq!(
            postprocess_with_command("definitely_not_a_real_command_xyz", text.clone()),
            text
        );
    }

    #[test]
    fn shell_quote_escapes_quotes() {
        assert_eq!(shell_quote(std::path::Path::new("/tmp/a b.txt")), "'/tmp/a b.txt'");
        assert_eq!(shell_quote(std::path::Path::new("/tmp/it's.txt")), "'/tmp/it'\\''s.txt'");
    }
}
