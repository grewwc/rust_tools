use std::{
    fs,
    io::{self, BufRead},
    path::{Path, PathBuf},
};

use rustyline::{CompletionType, Config, Editor, history::DefaultHistory};

use super::history::SessionStore;
use crate::commonw::utils::expanduser;

pub(super) mod completion;
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
    /// 下一次 `read_multi_line` 的预填文本（用于编辑已有内容场景，读取后清空）。
    pending_prefill: Option<String>,
    /// 当前模型显示名，用于在输入框顶部展示模型提示（输入时即可看到将使用的模型）。
    current_model_label: String,
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
            pending_prefill: None,
            current_model_label: String::new(),
        }
    }

    /// 设置下一次多行输入的预填内容（编辑已有 memo 时用，读取一次后自动清空）。
    pub(super) fn set_prefill(&mut self, text: impl Into<String>) {
        self.pending_prefill = Some(text.into());
    }

    /// 设置当前模型显示名，下一次 `read_multi_line` 会在输入框顶部展示。
    pub(super) fn set_current_model_label(&mut self, label: impl Into<String>) {
        self.current_model_label = label.into();
    }

    pub(super) fn read_multi_line(&mut self) -> io::Result<Option<String>> {
        use std::io::IsTerminal;
        if !io::stdout().is_terminal() || !io::stdin().is_terminal() {
            return self.read_multi_line_no_tty();
        }
        self.read_multi_line_tui()
    }

    fn read_multi_line_no_tty(&mut self) -> io::Result<Option<String>> {
        // 非 TTY 无法交互编辑：有预填且无管道输入时，直接返回预填原文。
        let prefill = self.pending_prefill.take();
        let stdin = io::stdin();
        let mut lines = Vec::new();
        for line in stdin.lock().lines() {
            lines.push(line?);
        }
        if lines.is_empty() {
            return Ok(prefill);
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
