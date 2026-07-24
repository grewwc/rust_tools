use std::{
    fs,
    io::{self, BufRead},
    path::{Path, PathBuf},
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
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

/// 后台任务只发布标题变更；终端重绘仍由前台输入循环独占。
#[derive(Clone)]
struct SessionTitleUpdate {
    session_id: String,
    title: String,
}

static SESSION_TITLE_UPDATE_SUBSCRIBERS: LazyLock<Mutex<Vec<(u64, Sender<SessionTitleUpdate>)>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static NEXT_SESSION_TITLE_UPDATE_SUBSCRIBER_ID: AtomicU64 = AtomicU64::new(1);

fn subscribe_session_title_updates() -> (u64, Receiver<SessionTitleUpdate>) {
    let (sender, receiver) = mpsc::channel();
    let subscriber_id = NEXT_SESSION_TITLE_UPDATE_SUBSCRIBER_ID.fetch_add(1, Ordering::Relaxed);
    SESSION_TITLE_UPDATE_SUBSCRIBERS
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .push((subscriber_id, sender));
    (subscriber_id, receiver)
}

/// 将已持久化的标题变更转交给当前前台编辑器。
pub(in crate::ai) fn notify_session_title_updated(session_id: &str, title: &str) {
    let update = SessionTitleUpdate {
        session_id: session_id.to_string(),
        title: title.to_string(),
    };
    SESSION_TITLE_UPDATE_SUBSCRIBERS
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .retain(|(_, sender)| sender.send(update.clone()).is_ok());
}

pub(super) struct PromptEditor {
    editor: Option<LineEditor>,
    pub(super) history_path: PathBuf,
    session_id: String,
    session_store: SessionStore,
    session_image_dir: PathBuf,
    /// 下一次 `read_multi_line` 的预填文本（用于编辑已有内容场景，读取后清空）。
    pending_prefill: Option<String>,
    /// 下一次 `read_multi_line` 初始展示的状态消息（读取后清空）。
    pending_status_msg: Option<String>,
    /// 当前模型显示名，用于在输入框顶部展示模型提示（输入时即可看到将使用的模型）。
    current_model_label: String,
    /// 当前 session 主题，用于在输入框顶部与模型提示同行展示。
    session_topic: Option<String>,
    /// 当前前台编辑器的后台标题变更订阅。
    session_title_update_subscription: u64,
    session_title_updates: Mutex<Receiver<SessionTitleUpdate>>,
    /// 首帧绘制完成后的单次通知。启动期的后台初始化可据此避开终端首屏渲染。
    first_render_notifier: Option<Sender<()>>,
}

impl Drop for PromptEditor {
    fn drop(&mut self) {
        SESSION_TITLE_UPDATE_SUBSCRIBERS
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .retain(|(subscriber_id, _)| *subscriber_id != self.session_title_update_subscription);
    }
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
        let session_store = SessionStore::new(history_file);
        let session_image_dir = session_store.session_assets_dir(session_id);
        let (session_title_update_subscription, session_title_updates) =
            subscribe_session_title_updates();
        Self {
            editor,
            history_path,
            session_id: session_id.to_string(),
            session_store,
            session_image_dir,
            pending_prefill: None,
            pending_status_msg: None,
            current_model_label: String::new(),
            session_topic: None,
            session_title_update_subscription,
            session_title_updates: Mutex::new(session_title_updates),
            first_render_notifier: None,
        }
    }

    /// 设置下一次多行输入的预填内容（编辑已有 memo 时用，读取一次后自动清空）。
    pub(super) fn set_prefill(&mut self, text: impl Into<String>) {
        self.pending_prefill = Some(text.into());
    }

    /// 设置下一次多行输入初始展示的状态消息，避免在 TUI 外直接打印打乱输入框。
    pub(super) fn set_status_message(&mut self, message: impl Into<String>) {
        self.pending_status_msg = Some(message.into());
    }

    /// 设置当前模型显示名，下一次 `read_multi_line` 会在输入框顶部展示。
    pub(super) fn set_current_model_label(&mut self, label: impl Into<String>) {
        self.current_model_label = label.into();
    }

    /// 更新当前绑定的 session。`PromptEditor` 生命周期跨 `/session` 切换，
    /// 进入输入框前需要同步到 app 的当前 session。
    pub(super) fn set_session_id(&mut self, session_id: impl Into<String>) {
        self.session_id = session_id.into();
    }

    /// 设置当前 session 主题，下一次 `read_multi_line` 会在模型提示同行展示。
    pub(super) fn set_session_topic(&mut self, topic: Option<String>) {
        self.session_topic = topic;
    }

    /// 设置下一次输入框首帧完成后的通知。
    pub(in crate::ai) fn set_first_render_notifier(&mut self, notifier: Sender<()>) {
        self.first_render_notifier = Some(notifier);
    }

    /// 通知一次即可；发送端被消费后，后续重绘不会产生额外事件。
    pub(in crate::ai::prompt) fn notify_first_render(&mut self) {
        if let Some(notifier) = self.first_render_notifier.take() {
            let _ = notifier.send(());
        }
    }

    /// 在前台安全点应用后台标题更新，避免后台任务直接操作终端。
    fn apply_pending_session_title_updates(&mut self) -> bool {
        let updates = {
            let receiver = self
                .session_title_updates
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            receiver.try_iter().collect::<Vec<_>>()
        };
        let mut changed = false;
        for update in updates {
            if update.session_id != self.session_id {
                continue;
            }
            let title = crate::ai::history::normalize_generated_session_title(&update.title);
            if title.trim().is_empty() || self.session_topic.as_deref() == Some(title.as_str()) {
                continue;
            }
            self.session_topic = Some(title);
            changed = true;
        }
        changed
    }

    pub(super) fn read_multi_line(&mut self) -> io::Result<Option<String>> {
        use std::io::IsTerminal;
        if !io::stdout().is_terminal() || !io::stdin().is_terminal() {
            return self.read_multi_line_no_tty();
        }
        match self.read_multi_line_tui() {
            Ok(input) => Ok(input),
            Err(err) if Self::is_cursor_position_timeout(&err) => self.read_multi_line_no_tty(),
            Err(err) => Err(err),
        }
    }

    fn is_cursor_position_timeout(err: &io::Error) -> bool {
        let msg = err.to_string();
        msg.contains("cursor position")
            || msg.contains("The cursor position could not be read within a normal duration")
    }

    fn read_multi_line_no_tty(&mut self) -> io::Result<Option<String>> {
        self.notify_first_render();
        // 非 TTY 无法交互编辑：有预填且无管道输入时，直接返回预填原文。
        let prefill = self.pending_prefill.take();
        let _ = self.pending_status_msg.take();
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

#[cfg(test)]
#[path = "prompt_tests.rs"]
mod tests;

pub(super) fn trim_trailing_newline(mut line: String) -> String {
    while matches!(line.chars().last(), Some('\n' | '\r')) {
        line.pop();
    }
    line
}

pub(super) fn interrupted_error() -> io::Result<Option<String>> {
    Err(io::Error::new(io::ErrorKind::Interrupted, "Ctrl+C"))
}
