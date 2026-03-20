use std::{
    fs::File,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use reqwest::blocking::Client;

use super::{cli::Cli, prompt::PromptEditor};

#[derive(Clone)]
pub(super) struct AppConfig {
    pub(super) api_key: String,
    pub(super) history_file: PathBuf,
    pub(super) endpoint: String,
    pub(super) vl_default_model: String,
}

pub(super) struct App {
    pub(super) cli: Cli,
    pub(super) config: AppConfig,
    pub(super) client: Client,
    pub(super) current_model: String,
    pub(super) pending_files: Option<String>,
    pub(super) pending_clipboard: bool,
    pub(super) pending_short_output: bool,
    pub(super) attached_image_files: Vec<String>,
    pub(super) attached_binary_files: Vec<String>,
    pub(super) uploaded_file_ids: Vec<String>,
    pub(super) shutdown: Arc<AtomicBool>,
    pub(super) streaming: Arc<AtomicBool>,
    pub(super) cancel_stream: Arc<AtomicBool>,
    pub(super) raw_args: String,
    pub(super) writer: Option<File>,
    pub(super) prompt_editor: Option<PromptEditor>,
}

pub(super) fn take_stream_cancelled(app: &App) -> bool {
    app.cancel_stream.swap(false, Ordering::SeqCst)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamOutcome {
    Completed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QuestionContext {
    pub(super) question: String,
    pub(super) history_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct LoopOverrides {
    pub(super) short_output: bool,
    pub(super) history_count: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct FileParseResult {
    pub(super) text_files: Vec<String>,
    pub(super) image_files: Vec<String>,
    pub(super) binary_files: Vec<String>,
}
