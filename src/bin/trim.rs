//! Trim whitespace in the clipboard and write the result back.
//!
//! The clipboard text is read, the whitespace is trimmed according to the
//! selected mode, and the trimmed text replaces the clipboard content.

use std::io::{self, Write};

use clap::{Parser, ValueEnum};
use rust_tools::{
    clipboardw::string_content::{get_clipboard_content, set_clipboard_content},
    strw::trim::{collapse_whitespace, trim_all_whitespace},
};

/// How the clipboard text is trimmed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Remove every whitespace character, wherever it appears.
    All,
    /// Remove whitespace at both ends only.
    Ends,
    /// Collapse whitespace runs into a single space and drop the ends.
    Collapse,
}

#[derive(Parser)]
#[command(about = "Trim whitespace in the clipboard and write the result back to it.")]
struct Cli {
    #[arg(
        short,
        long,
        value_enum,
        default_value_t = Mode::All,
        help = "trim mode: all (remove every whitespace char), ends (both ends only), collapse (runs into one space)"
    )]
    mode: Mode,

    #[arg(short, long, help = "print the trimmed text to stdout")]
    print: bool,

    #[arg(
        short,
        long,
        help = "do not write back to the clipboard, only report what would change"
    )]
    dry_run: bool,
}

/// Apply the trim mode to the text.
fn apply_mode(mode: Mode, text: &str) -> String {
    match mode {
        Mode::All => trim_all_whitespace(text).into_owned(),
        Mode::Ends => text.trim().to_string(),
        Mode::Collapse => collapse_whitespace(text).into_owned(),
    }
}

fn run(cli: &Cli) -> Result<(), String> {
    let original = get_clipboard_content();
    if original.is_empty() {
        return Err("clipboard holds no text content".to_string());
    }

    let trimmed = apply_mode(cli.mode, &original);

    if cli.print {
        let mut stdout = io::stdout();
        stdout
            .write_all(trimmed.as_bytes())
            .map_err(|e| format!("write stdout: {e}"))?;
        if !trimmed.ends_with('\n') {
            stdout.write_all(b"\n").map_err(|e| format!("write stdout: {e}"))?;
        }
        stdout.flush().map_err(|e| format!("flush stdout: {e}"))?;
    }

    if cli.dry_run {
        eprintln!(
            "clipboard unchanged (dry run): {} bytes -> {} bytes",
            original.len(),
            trimmed.len()
        );
        return Ok(());
    }

    set_clipboard_content(&trimmed).map_err(|e| format!("write clipboard: {e}"))?;
    eprintln!(
        "clipboard trimmed: {} bytes -> {} bytes",
        original.len(),
        trimmed.len()
    );

    Ok(())
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(&cli) {
        eprintln!("trim failed: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{Mode, apply_mode};

    #[test]
    fn mode_all_removes_every_whitespace_char() {
        assert_eq!(apply_mode(Mode::All, " a b\tc\n"), "abc");
    }

    #[test]
    fn mode_ends_keeps_inner_whitespace() {
        assert_eq!(apply_mode(Mode::Ends, "  a b  "), "a b");
    }

    #[test]
    fn mode_collapse_reduces_runs_to_single_space() {
        assert_eq!(apply_mode(Mode::Collapse, " a \t b "), "a b");
    }
}
