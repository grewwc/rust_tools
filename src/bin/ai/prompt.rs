use std::{
    fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
};

use rustyline::{
    CompletionType, Config, Editor,
    error::ReadlineError,
    history::DefaultHistory,
};

use super::history::SessionStore;
use crate::commonw::utils::expanduser;

mod completion;
mod multiline;

use completion::{CommandCompleter, LineEditor};
#[allow(unused_imports)]
pub(super) use multiline::MultilineHistoryState;

const LINE_REPL_HISTORY_FILE: &str = "~/.liner_history";
const MAX_INPUT_CHARS: usize = 4000;

pub(super) struct PromptEditor {
    editor: Option<LineEditor>,
    pub(super) history_path: PathBuf,
    session_image_dir: PathBuf,
}

impl PromptEditor {
    pub(super) fn new(session_id: &str, history_file: &Path) -> Self {
        let config = Config::builder()
            .completion_type(CompletionType::List)
            .build();
        let mut editor = Editor::<CommandCompleter, DefaultHistory>::with_config(config).ok();
        if let Some(editor) = editor.as_mut() {
            editor.set_helper(Some(CommandCompleter));
        }
        let history_path = PathBuf::from(expanduser(LINE_REPL_HISTORY_FILE).as_ref());
        if history_path.exists()
            && let Some(editor) = editor.as_mut()
        {
            let _ = editor.load_history(&history_path);
        }
        let session_image_dir = SessionStore::new(history_file).session_assets_dir(session_id);
        Self {
            editor,
            history_path,
            session_image_dir,
        }
    }

    pub(super) fn read_single_line(&mut self) -> io::Result<Option<String>> {
        use std::io::IsTerminal;

        if !io::stdout().is_terminal() {
            print!("> ");
            io::stdout().flush()?;
            let mut line = String::new();
            match io::stdin().read_line(&mut line) {
                Ok(0) => return Ok(None),
                Ok(_) => return Ok(Some(trim_trailing_newline(line))),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                    return interrupted_error();
                }
                Err(err) => return Err(err),
            }
        }

        let Some(editor) = self.editor.as_mut() else {
            print!("> ");
            io::stdout().flush()?;
            let mut line = String::new();
            match io::stdin().read_line(&mut line) {
                Ok(0) => return Ok(None),
                Ok(_) => return Ok(Some(trim_trailing_newline(line))),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                    return interrupted_error();
                }
                Err(err) => return Err(err),
            }
        };

        match editor.readline("> ") {
            Ok(line) => {
                self.save_history_entry(&line);
                Ok(Some(line))
            }
            Err(ReadlineError::Eof) => Ok(None),
            Err(ReadlineError::Interrupted) => interrupted_error(),
            Err(err) => Err(io::Error::other(err.to_string())),
        }
    }

    pub(super) fn read_multi_line(&mut self) -> io::Result<Option<String>> {
        use std::io::IsTerminal;
        if !io::stdout().is_terminal() {
            return self.read_multi_line_no_tty();
        }
        self.read_multi_line_tui()
    }

    fn read_multi_line_no_tty(&mut self) -> io::Result<Option<String>> {
        let stdin = io::stdin();
        let mut lines = Vec::new();
        for line in stdin.lock().lines() {
            lines.push(line?);
        }
        if lines.is_empty() {
            return Ok(None);
        }
        let content = lines.join("\n");
        self.save_history_entry(&content);
        Ok(Some(content))
    }

    fn multiline_history_entries(&self) -> Vec<String> {
        self.editor
            .as_ref()
            .map(|editor| {
                editor
                    .history()
                    .iter()
                    .filter(|entry| !entry.trim().is_empty())
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn save_history_entry(&mut self, entry: &str) {
        if entry.trim().is_empty() {
            return;
        }
        let Some(editor) = self.editor.as_mut() else {
            return;
        };

        let _ = editor.add_history_entry(entry);
        if let Some(parent) = self.history_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = editor.save_history(&self.history_path);
    }
}

pub(super) fn trim_trailing_newline(mut line: String) -> String {
    while matches!(line.chars().last(), Some('\n' | '\r')) {
        line.pop();
    }
    line
}

pub(super) fn interrupted_error() -> io::Result<Option<String>> {
    Err(io::Error::new(io::ErrorKind::Interrupted, "Ctrl+C"))
}
