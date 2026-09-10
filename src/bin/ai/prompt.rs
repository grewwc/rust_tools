use std::{
    collections::VecDeque,
    fs,
    io::{self, BufRead},
    path::{Path, PathBuf},
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    time::Instant,
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

/// Background tasks only publish title changes; terminal redraw stays exclusive
/// to the foreground input loop.
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

/// Forward a persisted title change to the current foreground editor.
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
    /// Prefill text for the next `read_multi_line` (for editing existing content;
    /// cleared after being read).
    pending_prefill: Option<String>,
    /// Initial status message shown by the next `read_multi_line` (cleared after being read).
    pending_status_msg: Option<String>,
    /// Current model display name, shown as a model hint above the input box
    /// (so the model in use is visible while typing).
    current_model_label: String,
    /// Reasoning effort active for the current request, shown on the same line
    /// as the model hint above the input box.
    current_reasoning_effort_label: String,
    /// Current session topic, shown on the same line as the model hint above the input box.
    session_topic: Option<String>,
    /// Background title-change subscription for the current foreground editor.
    session_title_update_subscription: u64,
    session_title_updates: Mutex<Receiver<SessionTitleUpdate>>,
    /// One-shot notification after the first frame is drawn. Startup background
    /// initialization can use it to avoid terminal first-screen rendering.
    first_render_notifier: Option<Sender<()>>,
    /// Once a DSR query times out, later untagged replies cannot be matched to
    /// new queries. Subsequent prompts stay on the query-free editing screen
    /// until a probe round-trip succeeds (see the editor's recovery probe).
    /// Only the fact of degradation is carried: the probe backoff restarts with
    /// every prompt, so a new prompt retries the inline screen immediately
    /// instead of waiting out the backoff the previous prompt accumulated.
    cursor_position_queries_disabled: bool,
    /// Send time of every DSR query whose reply is still unaccounted for, oldest
    /// first. A reply carries no request id, so while a *recent* one is
    /// outstanding a new inline query could read a stale answer and anchor the box
    /// on the wrong row. The queue also drives the orphan-Escape guard: a
    /// timed-out query can leave one orphan ESC byte in crossterm's queue (the
    /// leading 0x1B arrived alone while the CPR poll was filtering for cursor
    /// events, so crossterm parsed it as an Escape key and the `[row;colR` tail
    /// follows as ordinary char events). Each entry makes the event parser defer
    /// the next standalone Escape briefly, so the orphan is swallowed instead of
    /// being typed into the box, while a real Escape still submits after that
    /// grace. Entries expire ten seconds after their query: a reply that late was
    /// lost with the connection, and deferring for it would tax every later
    /// submission. Persists across prompt rounds because the orphan can be
    /// delivered after a non-Escape submission (F2 / Alt+Enter).
    pending_cpr_replies: VecDeque<Instant>,
    /// Preserve non-CPR lookahead even when Escape submits the current prompt.
    pending_terminal_events: VecDeque<crossterm::event::Event>,
    /// Tail of the most recent model output, mirrored above the input box when a
    /// timed-out cursor query forces the editor onto the alternate screen (where
    /// the emulator keeps the real transcript out of reach). Filled by the turn
    /// finalizer right after the answer is rendered.
    alternate_tail_lines: Vec<String>,
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
            current_reasoning_effort_label: String::new(),
            session_topic: None,
            session_title_update_subscription,
            session_title_updates: Mutex::new(session_title_updates),
            first_render_notifier: None,
            cursor_position_queries_disabled: false,
            pending_cpr_replies: VecDeque::new(),
            pending_terminal_events: VecDeque::new(),
            alternate_tail_lines: Vec::new(),
        }
    }

    /// Set the prefill content for the next multi-line input (used when editing
    /// an existing memo; auto-cleared after one read).
    pub(super) fn set_prefill(&mut self, text: impl Into<String>) {
        self.pending_prefill = Some(text.into());
    }

    /// Set the status message shown initially by the next multi-line input,
    /// avoiding direct prints outside the TUI that would disturb the input box.
    pub(super) fn set_status_message(&mut self, message: impl Into<String>) {
        self.pending_status_msg = Some(message.into());
    }

    /// Mirror the tail of a just-rendered model answer. The alternate-screen
    /// fallback (cursor-position query timed out) repaints these rows above the
    /// box, so an unstable link still shows what the model just said instead of
    /// an empty screen.
    pub(super) fn set_alternate_screen_tail(&mut self, text: &str) {
        self.alternate_tail_lines = alternate_tail_source_lines(text);
    }

    /// Set the current model display name; the next `read_multi_line` shows it
    /// above the input box.
    pub(super) fn set_current_model_label(&mut self, label: impl Into<String>) {
        self.current_model_label = label.into();
    }

    /// Set the reasoning effort active for the current request; the next
    /// `read_multi_line` shows it on the same line as the model hint.
    pub(super) fn set_current_reasoning_effort_label(&mut self, label: impl Into<String>) {
        self.current_reasoning_effort_label = label.into();
    }

    /// Update the bound session. `PromptEditor` outlives `/session` switches,
    /// so it must be synced to the app's current session before entering the input box.
    pub(super) fn set_session_id(&mut self, session_id: impl Into<String>) {
        self.session_id = session_id.into();
    }

    /// Set the current session topic; the next `read_multi_line` shows it on the
    /// same line as the model hint.
    pub(super) fn set_session_topic(&mut self, topic: Option<String>) {
        self.session_topic = topic;
    }

    /// Set the notification fired once the next input box's first frame is drawn.
    pub(in crate::ai) fn set_first_render_notifier(&mut self, notifier: Sender<()>) {
        self.first_render_notifier = Some(notifier);
    }

    /// Notify only once; once the sender is consumed, later redraws produce no extra events.
    pub(in crate::ai::prompt) fn notify_first_render(&mut self) {
        if let Some(notifier) = self.first_render_notifier.take() {
            let _ = notifier.send(());
        }
    }

    /// Apply pending background title updates at a foreground safe point, so
    /// background tasks never touch the terminal directly.
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
        // A cursor-query timeout is handled inside the raw-mode editor. Never
        // flush typeahead or switch to cooked input while a reply is in flight.
        self.read_multi_line_tui()
    }

    fn is_cursor_position_timeout(err: &io::Error) -> bool {
        let msg = err.to_string();
        msg.contains("cursor position")
            || msg.contains("The cursor position could not be read within a normal duration")
    }

    fn read_multi_line_no_tty(&mut self) -> io::Result<Option<String>> {
        use std::io::IsTerminal;
        self.notify_first_render();
        // Without a TTY there is no interactive editing: if a prefill exists and
        // there is no piped input, return the prefill text as-is.
        let prefill = self.pending_prefill.take();
        let _ = self.pending_status_msg.take();
        let stdin = io::stdin();
        let mut lines = Vec::new();
        if stdin.is_terminal() {
            // With redirected stdout, stdin may still be a terminal. Read one
            // line rather than waiting for EOF from an interactive user.
            if let Some(line) = stdin.lock().lines().next() {
                lines.push(line?);
            }
        } else {
            // Consume all piped or redirected input through EOF.
            for line in stdin.lock().lines() {
                lines.push(line?);
            }
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

/// Source lines retained for the alternate-screen mirror. The mirror itself
/// shows at most a screenful, so this only bounds what the editor keeps alive.
const ALTERNATE_TAIL_SOURCE_LINES: usize = 40;

/// Extract the tail lines mirrored by `PromptEditor::set_alternate_screen_tail`.
///
/// Blank lines are dropped: the mirror is a compact reminder of how the answer
/// ended, and blank rows would waste the few rows available above the box.
fn alternate_tail_source_lines(text: &str) -> Vec<String> {
    let mut lines: Vec<String> = text
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect();
    if lines.len() > ALTERNATE_TAIL_SOURCE_LINES {
        lines.drain(..lines.len() - ALTERNATE_TAIL_SOURCE_LINES);
    }
    lines
}

#[cfg(test)]
#[path = "prompt_tests.rs"]
mod tests;

#[cfg(all(test, unix))]
#[path = "prompt/terminal_input_tests.rs"]
mod terminal_input_tests;

pub(super) fn trim_trailing_newline(mut line: String) -> String {
    while matches!(line.chars().last(), Some('\n' | '\r')) {
        line.pop();
    }
    line
}

pub(super) fn interrupted_error() -> io::Result<Option<String>> {
    Err(io::Error::new(io::ErrorKind::Interrupted, "Ctrl+C"))
}
