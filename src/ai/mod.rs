use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use base64::Engine as _;
use clap::{ArgAction, Parser};
use colored::Colorize;
use reqwest::blocking::{Client, Response, multipart};
use rustyline::{DefaultEditor, error::ReadlineError};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    clipboard::string_content,
    common::{configw, utils::expanduser},
    strw::split::{split_by_str_keep_quotes, split_space_keep_symbol},
};

const MAX_HISTORY_LINES: usize = 100;
const DEFAULT_NUM_HISTORY: usize = 4;
const COLON: char = '\0';
const NEWLINE: char = '\x01';
const LINE_REPL_HISTORY_FILE: &str = "~/.liner_histroy";

const DEEPSEEK_V3: &str = "deepseek-v3.1";
const DEEPSEEK_R1: &str = "deepseek-r1";
const QWEN_MAX_LATEST: &str = "qwen-max-latest";
const QWEN_PLUS_LATEST: &str = "qwen3.5-plus";
const QWEN_MAX: &str = "qwen-max";
const QWEN_CODER_PLUS_LATEST: &str = "qwen3-coder-plus";
const QWEN_LONG: &str = "qwen-long";
const QWQ: &str = "qwq-plus-latest";
const QWEN_FLASH: &str = "qwen-flash";
const QWEN3_MAX: &str = "qwen3-max";

const QWEN_VL_FLASH: &str = "qwen3-vl-flash";
const QWEN_VL_MAX: &str = "qwen3-vl-plus";
const QWEN_VL_OCR: &str = "qwen-vl-ocr-latest";

const QWEN_ENDPOINT: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
const FILES_ENDPOINT: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1/files";

const ALL_MODELS: &[&str] = &[
    DEEPSEEK_V3,
    DEEPSEEK_R1,
    QWEN_MAX_LATEST,
    QWEN_PLUS_LATEST,
    QWEN_MAX,
    QWEN_CODER_PLUS_LATEST,
    QWEN_LONG,
    QWQ,
    QWEN_FLASH,
    QWEN3_MAX,
    QWEN_VL_FLASH,
    QWEN_VL_MAX,
    QWEN_VL_OCR,
];

const ENABLE_SEARCH_MODELS: &[&str] = &[
    QWEN_MAX,
    QWEN_MAX_LATEST,
    QWEN_PLUS_LATEST,
    QWEN_FLASH,
    DEEPSEEK_V3,
    QWEN3_MAX,
];

const VL_MODELS: &[&str] = &[QWEN_VL_FLASH, QWEN_VL_MAX, QWEN_VL_OCR];

#[derive(Parser, Debug)]
#[command(about = "AI CLI compatible with go_tools executable/ai/a.go")]
struct Cli {
    #[arg(long, default_value_t = DEFAULT_NUM_HISTORY, help = "number of history")]
    history: usize,

    #[arg(
        short = 'm',
        long = "model",
        default_value = "",
        help = "model name. qwq-plus-latest[0], qwen3.5-plus[1], qwen-max[2], qwen3-max[3], qwen3-coder-plus[4], deepseek-v3.1[5], qwen-flash[6]"
    )]
    model: String,

    #[arg(long = "multi-line", visible_alias = "mul", action = ArgAction::SetTrue, help = "input with multline")]
    multi_line: bool,

    #[arg(long, action = ArgAction::SetTrue, help = "use code model (qwen3-coder-plus)")]
    code: bool,

    #[arg(short = 'd', action = ArgAction::SetTrue, help = "deepseek model")]
    deepseek: bool,

    #[arg(long, action = ArgAction::SetTrue, help = "clear history")]
    clear: bool,

    #[arg(short = 'c', action = ArgAction::SetTrue, help = "prepend content in clipboard")]
    clipboard: bool,

    #[arg(short = 'x', action = ArgAction::SetTrue, help = "ask without history")]
    no_history: bool,

    #[arg(short = 'f', default_value = "", help = "input file names. seprated by comma.")]
    files: String,

    #[arg(short = 'o', long = "out", num_args = 0..=1, default_missing_value = "output.md", help = "write output to file. default is output.md")]
    out: Option<String>,

    #[arg(long, action = ArgAction::SetTrue, help = "raw mode: don't use parser to get positional arguments, use raw inputs instead.")]
    raw: bool,

    #[arg(short = 't', action = ArgAction::SetTrue, help = "use thinking model. default: false.")]
    thinking: bool,

    #[arg(short = 's', action = ArgAction::SetTrue, help = "short output")]
    short_output: bool,

    #[arg(short = '0', action = ArgAction::SetTrue, help = "select qwq-plus-latest")]
    model_0: bool,

    #[arg(short = '1', action = ArgAction::SetTrue, help = "select qwen3.5-plus")]
    model_1: bool,

    #[arg(short = '2', action = ArgAction::SetTrue, help = "select qwen-max")]
    model_2: bool,

    #[arg(short = '3', action = ArgAction::SetTrue, help = "select qwen3-max")]
    model_3: bool,

    #[arg(short = '4', action = ArgAction::SetTrue, help = "select qwen3-coder-plus")]
    model_4: bool,

    #[arg(short = '5', action = ArgAction::SetTrue, help = "select deepseek-v3.1 / deepseek-r1")]
    model_5: bool,

    #[arg(short = '6', action = ArgAction::SetTrue, help = "select qwen-flash")]
    model_6: bool,

    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

#[derive(Clone)]
struct AppConfig {
    api_key: String,
    history_file: PathBuf,
    endpoint: String,
    vl_default_model: String,
}

struct App {
    cli: Cli,
    config: AppConfig,
    client: Client,
    current_model: String,
    pending_files: Option<String>,
    pending_clipboard: bool,
    pending_short_output: bool,
    attached_image_files: Vec<String>,
    attached_binary_files: Vec<String>,
    uploaded_file_ids: Vec<String>,
    shutdown: Arc<AtomicBool>,
    streaming: Arc<AtomicBool>,
    cancel_stream: Arc<AtomicBool>,
    raw_args: String,
    writer: Option<File>,
    prompt_editor: Option<PromptEditor>,
}

struct PromptEditor {
    editor: Option<DefaultEditor>,
    history_path: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct MultilineHistoryState {
    entries: Vec<String>,
    index: Option<usize>,
    draft: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuestionContext {
    question: String,
    history_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct LoopOverrides {
    short_output: bool,
    history_count: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct FileParseResult {
    text_files: Vec<String>,
    image_files: Vec<String>,
    binary_files: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamOutcome {
    Completed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Message {
    role: String,
    content: Value,
}

#[derive(Debug, Serialize)]
struct RequestBody {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    enable_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_search: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Default, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
}

#[derive(Debug, Default, Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: String,
    #[serde(default)]
    reasoning_content: String,
}

#[derive(Debug, Deserialize)]
struct UploadResponse {
    #[serde(default)]
    id: String,
}

fn normalize_single_dash_long_opts(args: impl Iterator<Item = String>) -> Vec<String> {
    args.map(|arg| {
        let bytes = arg.as_bytes();
        if bytes.len() > 2
            && bytes[0] == b'-'
            && bytes[1] != b'-'
            && bytes[1].is_ascii_alphabetic()
        {
            format!("-{arg}")
        } else {
            arg
        }
    })
    .collect()
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse_from(normalize_single_dash_long_opts(std::env::args()));
    let config = load_config()?;

    if cli.clear {
        let _ = fs::remove_file(&config.history_file);
        println!("History cleared.");
        return Ok(());
    }

    let shutdown = Arc::new(AtomicBool::new(false));
    let streaming = Arc::new(AtomicBool::new(false));
    let cancel_stream = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&shutdown);
    let streaming_flag = Arc::clone(&streaming);
    let cancel_stream_flag = Arc::clone(&cancel_stream);
    ctrlc::set_handler(move || {
        handle_sigint(
            signal_flag.as_ref(),
            streaming_flag.as_ref(),
            cancel_stream_flag.as_ref(),
        );
    })?;

    let writer = open_output_writer(cli.out.as_deref())?;
    let current_model = initial_model(&cli);
    let client = Client::builder().build()?;
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt_editor = if cli.args.is_empty() {
        Some(PromptEditor::new())
    } else {
        None
    };

    let mut app = App {
        pending_files: if cli.files.trim().is_empty() {
            None
        } else {
            Some(cli.files.clone())
        },
        pending_clipboard: cli.clipboard,
        pending_short_output: cli.short_output,
        current_model,
        raw_args,
        cli,
        config,
        client,
        attached_image_files: Vec::new(),
        attached_binary_files: Vec::new(),
        uploaded_file_ids: Vec::new(),
        shutdown,
        streaming,
        cancel_stream,
        writer,
        prompt_editor,
    };

    run_loop(&mut app)
}

fn load_config() -> Result<AppConfig, Box<dyn std::error::Error>> {
    let cfg = configw::get_all_config();
    let api_key = cfg.get_opt("api_key").unwrap_or_default();
    if api_key.trim().is_empty() {
        println!("set api_key in ~/.configW");
        std::process::exit(0);
    }
    let history_file = cfg
        .get_opt("history_file")
        .unwrap_or_else(|| "~/.history_file.txt".to_string());
    let endpoint = cfg
        .get_opt("ai.model.endpoint")
        .unwrap_or_else(|| QWEN_ENDPOINT.to_string());
    let vl_default_model = determine_vl_model(
        &cfg.get_opt("ai.model.vl_default").unwrap_or_default(),
    );
    Ok(AppConfig {
        api_key,
        history_file: PathBuf::from(expanduser(&history_file).as_ref()),
        endpoint,
        vl_default_model,
    })
}

fn open_output_writer(path: Option<&str>) -> io::Result<Option<File>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let mut options = OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o644);
    }
    options.open(path).map(Some)
}

impl PromptEditor {
    fn new() -> Self {
        let mut editor = DefaultEditor::new().ok();
        let history_path = PathBuf::from(expanduser(LINE_REPL_HISTORY_FILE).as_ref());
        if history_path.exists()
            && let Some(editor) = editor.as_mut()
        {
            let _ = editor.load_history(&history_path);
        }
        Self {
            editor,
            history_path,
        }
    }

    fn read_single_line(&mut self) -> io::Result<Option<String>> {
        let Some(editor) = self.editor.as_mut() else {
            print!("> ");
            io::stdout().flush()?;
            let mut line = String::new();
            match io::stdin().read_line(&mut line) {
                Ok(0) => return Ok(None),
                Ok(_) => return Ok(Some(trim_trailing_newline(line))),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                    println!("Exit.");
                    return Ok(None);
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
            Err(ReadlineError::Interrupted) => {
                println!("Exit.");
                Ok(None)
            }
            Err(err) => Err(io::Error::other(err.to_string())),
        }
    }

    fn read_multi_line(&mut self) -> io::Result<Option<String>> {
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

    fn read_multi_line_tui(&mut self) -> io::Result<Option<String>> {
        use crossterm::{
            event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
            terminal::{disable_raw_mode, enable_raw_mode},
        };
        use ratatui::{
            Terminal,
            backend::CrosstermBackend,
            layout::{Constraint, Direction, Layout},
            style::{Color, Style},
            text::{Line, Span},
            widgets::{Block, Borders, Clear, Paragraph},
        };
        use tui_textarea::{CursorMove, Input, TextArea};

        // Let the user read the previous output before opening the compose viewport.
        {
            use crossterm::{
                event::{self, Event, KeyCode, KeyEventKind},
                terminal::{disable_raw_mode, enable_raw_mode},
            };
            print!("\x1b[2m[Press Enter to compose next message, Ctrl+D/Ctrl+C to quit]\x1b[0m ");
            io::stdout().flush()?;
            enable_raw_mode()?;
            loop {
                match event::read() {
                    Ok(Event::Key(key))
                        if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                    {
                        match key.code {
                            KeyCode::Enter => break,
                            KeyCode::Char('d')
                                if key.modifiers
                                    == crossterm::event::KeyModifiers::CONTROL =>
                            {
                                let _ = disable_raw_mode();
                                println!();
                                return Ok(None);
                            }
                            KeyCode::Char('c')
                                if key.modifiers
                                    == crossterm::event::KeyModifiers::CONTROL =>
                            {
                                let _ = disable_raw_mode();
                                println!();
                                return Ok(None);
                            }
                            _ => {}
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let _ = disable_raw_mode();
                        return Err(io::Error::other(e.to_string()));
                    }
                }
            }
            let _ = disable_raw_mode();
            println!();
        }

        enable_raw_mode()?;

        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = match Terminal::with_options(
            backend,
            ratatui::TerminalOptions {
                viewport: ratatui::Viewport::Inline(18),
            },
        ) {
            Ok(terminal) => terminal,
            Err(err) => {
                let _ = disable_raw_mode();
                return Err(io::Error::other(err.to_string()));
            }
        };

        let result: io::Result<Option<String>> = (|| {
            let mut textarea: TextArea = TextArea::default();
            let mut history = MultilineHistoryState::new(self.multiline_history_entries());

            let outcome = loop {
                terminal
                    .draw(|f| {
                        let area = f.area();
                        let popup_height = area.height.saturating_sub(2).min(18).max(6).min(area.height);
                        let popup_width = area.width.saturating_sub(4).min(110).max(40).min(area.width);
                        let popup = centered_rect(area, popup_width, popup_height);

                        let popup_block = Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::DarkGray))
                            .title(" Compose ");
                        let inner = popup_block.inner(popup);
                        let chunks = Layout::default()
                            .direction(Direction::Vertical)
                            .constraints([Constraint::Min(3), Constraint::Length(2)])
                            .split(inner);

                        f.render_widget(Clear, popup);
                        f.render_widget(popup_block, popup);

                        let n = textarea.lines().len();
                        textarea.set_block(
                            Block::default()
                                .borders(Borders::ALL)
                                .border_style(Style::default().fg(Color::Cyan))
                                .title(format!(
                                    " Message ({n} line{}) ",
                                    if n == 1 { "" } else { "s" }
                                )),
                        );
                        f.render_widget(&textarea, chunks[0]);
                        f.render_widget(
                            Paragraph::new(vec![
                                Line::from(vec![
                                    Span::raw("  "),
                                    Span::styled("Enter", Style::default().fg(Color::Blue)),
                                    Span::raw(" newline  ·  "),
                                    Span::styled("Ctrl+D", Style::default().fg(Color::Green)),
                                    Span::raw(" send  ·  "),
                                    Span::styled("Ctrl+C/Esc", Style::default().fg(Color::Yellow)),
                                    Span::raw(" cancel"),
                                ]),
                                Line::from(vec![
                                    Span::raw("  "),
                                    Span::styled("Up/Down edge", Style::default().fg(Color::Blue)),
                                    Span::raw(" or "),
                                    Span::styled("Ctrl+P/N", Style::default().fg(Color::Blue)),
                                    Span::raw(" history  ·  "),
                                    Span::styled("Backspace", Style::default().fg(Color::Blue)),
                                    Span::raw(" edits previous lines"),
                                ]),
                            ]),
                            chunks[1],
                        );
                    })
                    .map_err(|e| io::Error::other(e.to_string()))?;

                match event::read().map_err(|e| io::Error::other(e.to_string()))? {
                    Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
                        match (key.code, key.modifiers) {
                            (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                                let content = textarea_content(&textarea);
                                let trimmed = content.trim_end_matches('\n').to_string();
                                break Ok(if trimmed.trim().is_empty() {
                                    None
                                } else {
                                    Some(trimmed)
                                });
                            }
                            (KeyCode::Esc, _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                break Ok(None);
                            }
                            _ => {
                                let handled = match (key.code, key.modifiers) {
                                    (KeyCode::Up, modifiers)
                                        if modifiers.is_empty() && textarea.cursor().0 == 0 =>
                                    {
                                        if let Some(content) = history.previous(&textarea_content(&textarea)) {
                                            replace_textarea_content(&mut textarea, &content);
                                            true
                                        } else {
                                            false
                                        }
                                    }
                                    (KeyCode::Down, modifiers)
                                        if modifiers.is_empty()
                                            && textarea.cursor().0 + 1 >= textarea.lines().len() =>
                                    {
                                        if let Some(content) = history.next() {
                                            replace_textarea_content(&mut textarea, &content);
                                            true
                                        } else {
                                            false
                                        }
                                    }
                                    (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                                        if let Some(content) = history.previous(&textarea_content(&textarea)) {
                                            replace_textarea_content(&mut textarea, &content);
                                            true
                                        } else {
                                            false
                                        }
                                    }
                                    (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
                                        if let Some(content) = history.next() {
                                            replace_textarea_content(&mut textarea, &content);
                                            true
                                        } else {
                                            false
                                        }
                                    }
                                    _ => false,
                                };

                                if handled {
                                    textarea.move_cursor(CursorMove::Bottom);
                                    textarea.move_cursor(CursorMove::End);
                                    continue;
                                }

                                textarea.input(Input::from(key));
                            }
                        }
                    }
                    Event::Key(_) => {}
                    _ => {}
                }
            };
            outcome
        })();

        let _ = terminal.clear();
        let _ = terminal.show_cursor();
        let _ = disable_raw_mode();

        let result = result?;
        if let Some(content) = &result {
            self.save_history_entry(content);
            let mut lines = content.lines();
            if let Some(first) = lines.next() {
                println!("\x1b[2m> {first}\x1b[0m");
            }
            for line in lines {
                println!("\x1b[2m  {line}\x1b[0m");
            }
        }
        Ok(result)
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

impl MultilineHistoryState {
    fn new(entries: Vec<String>) -> Self {
        Self {
            entries,
            index: None,
            draft: None,
        }
    }

    fn previous(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }

        let next_index = match self.index {
            Some(0) => return None,
            Some(index) => index - 1,
            None => {
                self.draft = Some(current.to_string());
                self.entries.len() - 1
            }
        };
        self.index = Some(next_index);
        self.entries.get(next_index).cloned()
    }

    fn next(&mut self) -> Option<String> {
        let index = self.index?;
        if index + 1 < self.entries.len() {
            self.index = Some(index + 1);
            return self.entries.get(index + 1).cloned();
        }

        self.index = None;
        Some(self.draft.take().unwrap_or_default())
    }
}

fn textarea_content(textarea: &tui_textarea::TextArea<'_>) -> String {
    textarea.lines().join("\n")
}

fn centered_rect(area: ratatui::layout::Rect, width: u16, height: u16) -> ratatui::layout::Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    ratatui::layout::Rect::new(x, y, width, height)
}

fn replace_textarea_content(textarea: &mut tui_textarea::TextArea<'_>, content: &str) {
    let lines = content.split('\n').map(|line| line.to_string()).collect();
    *textarea = tui_textarea::TextArea::new(lines);
}

fn handle_sigint(shutdown: &AtomicBool, streaming: &AtomicBool, cancel_stream: &AtomicBool) {
    if streaming.load(Ordering::SeqCst) {
        cancel_stream.store(true, Ordering::SeqCst);
        return;
    }

    shutdown.store(true, Ordering::SeqCst);
}

fn initial_model(cli: &Cli) -> String {
    if cli.code {
        return QWEN_CODER_PLUS_LATEST.to_string();
    }
    if cli.deepseek {
        return if cli.thinking {
            DEEPSEEK_R1.to_string()
        } else {
            DEEPSEEK_V3.to_string()
        };
    }
    if let Some(selector) = selected_model_number(cli) {
        return model_from_selector(selector, cli.thinking).to_string();
    }
    if !cli.model.trim().is_empty() {
        return determine_model(&cli.model);
    }
    let cfg = configw::get_all_config();
    cfg.get_opt("ai.model.default")
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| QWEN3_MAX.to_string())
}

fn selected_model_number(cli: &Cli) -> Option<u8> {
    [
        cli.model_0,
        cli.model_1,
        cli.model_2,
        cli.model_3,
        cli.model_4,
        cli.model_5,
        cli.model_6,
    ]
    .into_iter()
    .enumerate()
    .find_map(|(idx, enabled)| enabled.then_some(idx as u8))
}

fn model_from_selector(selector: u8, thinking_mode: bool) -> &'static str {
    match selector {
        0 => QWQ,
        1 => QWEN_PLUS_LATEST,
        2 => QWEN_MAX,
        3 => QWEN3_MAX,
        4 => QWEN_CODER_PLUS_LATEST,
        5 => {
            if thinking_mode {
                DEEPSEEK_R1
            } else {
                DEEPSEEK_V3
            }
        }
        6 => QWEN_FLASH,
        _ => QWEN3_MAX,
    }
}

fn determine_model(model: &str) -> String {
    let model = model.trim().to_lowercase();
    if model.is_empty() {
        return QWEN3_MAX.to_string();
    }
    let mut best = QWEN3_MAX;
    let mut best_dist = f32::MAX;
    for candidate in ALL_MODELS {
        let dist = levenshtein(model.as_bytes(), candidate.as_bytes()) as f32
            / (model.len() + candidate.len()) as f32;
        if dist < best_dist {
            best_dist = dist;
            best = candidate;
        }
    }
    best.to_string()
}

fn determine_vl_model(model: &str) -> String {
    let model = model.trim().to_lowercase();
    if model.is_empty() {
        return QWEN_VL_FLASH.to_string();
    }

    match model.as_str() {
        "0" => return QWEN_VL_FLASH.to_string(),
        "1" => return QWEN_VL_MAX.to_string(),
        "2" => return QWEN_VL_OCR.to_string(),
        _ => {}
    }

    if is_vl_model(&model) {
        return model;
    }

    let mut best = QWEN_VL_FLASH;
    let mut best_dist = f32::MAX;
    for candidate in VL_MODELS {
        let dist = levenshtein(model.as_bytes(), candidate.as_bytes()) as f32
            / (model.len() + candidate.len()) as f32;
        if dist < best_dist {
            best_dist = dist;
            best = candidate;
        }
    }
    best.to_string()
}

fn run_loop(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    let mut should_quit = !app.cli.args.is_empty();
    loop {
        if app.shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        let Some(ctx) = next_question(app)? else {
            return Ok(());
        };
        if ctx.question.trim().is_empty() {
            should_quit = false;
            continue;
        }

        let mut question = ctx.question;
        let next_model = resolve_model_for_input(app, &mut question);
        app.current_model = next_model.clone();

        app.cancel_stream.store(false, Ordering::SeqCst);
        let mut current_history = format!("user{COLON}{question}{NEWLINE}assistant{COLON}");
        let mut response = do_request(app, &next_model, &question, ctx.history_count)?;
        print_info(&next_model);
        app.streaming.store(true, Ordering::SeqCst);
        let outcome = match stream_response(app, &mut response, &mut current_history) {
            Ok(outcome) => outcome,
            Err(err) => {
                app.streaming.store(false, Ordering::SeqCst);
                return Err(err);
            }
        };
        app.streaming.store(false, Ordering::SeqCst);
        if outcome == StreamOutcome::Cancelled {
            println!("\nInterrupted.");
            if should_quit {
                return Ok(());
            }
            continue;
        }
        if app.shutdown.load(Ordering::SeqCst) {
            println!();
            return Ok(());
        }
        response.copy_to(&mut io::sink())?;
        current_history.push(NEWLINE);
        append_history(&app.config.history_file, &current_history)?;
        println!();

        if should_quit {
            return Ok(());
        }
        if let Some(writer) = app.writer.as_mut() {
            writer.write_all(b"\n---\n")?;
            writer.flush()?;
        }
    }
}

fn next_question(app: &mut App) -> Result<Option<QuestionContext>, Box<dyn std::error::Error>> {
    if !app.cli.args.is_empty() {
        let base_question = if app.cli.raw {
            app.raw_args.clone()
        } else {
            let question = app.cli.args.join(" ");
            app.cli.args.clear();
            question
        };
        app.cli.args.clear();
        let ctx = finalize_question(
            app,
            base_question,
            base_history_count(app.cli.history, app.cli.no_history),
            false,
        )?;
        return Ok(Some(ctx));
    }

    let Some(question) = prompt_user(app)? else {
        return Ok(None);
    };
    let overrides = loop_overrides(&question);
    let history_count = overrides
        .history_count
        .unwrap_or_else(|| base_history_count(app.cli.history, app.cli.no_history));
    let ctx = finalize_question(app, question, history_count, overrides.short_output)?;
    Ok(Some(ctx))
}

fn base_history_count(history: usize, no_history: bool) -> usize {
    if no_history { 0 } else { history }
}

fn finalize_question(
    app: &mut App,
    mut question: String,
    history_count: usize,
    loop_short_output: bool,
) -> Result<QuestionContext, Box<dyn std::error::Error>> {
    if let Some(files) = app.pending_files.take() {
        let parsed = parse_files(&files);
        if !parsed.text_files.is_empty() {
            let prefix = text_file_contents(&parsed.text_files)?;
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

fn trim_trailing_newline(mut line: String) -> String {
    while matches!(line.chars().last(), Some('\n' | '\r')) {
        line.pop();
    }
    line
}

fn loop_overrides(question: &str) -> LoopOverrides {
    let tokens = split_space_keep_symbol(question, "\"'").collect::<Vec<_>>();
    let short_output = tokens.iter().any(|token| *token == "-s");

    if tokens.iter().any(|token| *token == "-x") {
        return LoopOverrides {
            short_output,
            history_count: Some(0),
        };
    }

    let mut history_count = None;
    let mut idx = 0usize;
    while idx < tokens.len() {
        if tokens[idx] == "--history" {
            if let Some(next) = tokens.get(idx + 1)
                && let Ok(value) = next.parse::<usize>()
            {
                history_count = Some(value);
                break;
            }
        }
        idx += 1;
    }

    LoopOverrides {
        short_output,
        history_count,
    }
}

fn parse_files(content: &str) -> FileParseResult {
    let files = split_by_str_keep_quotes(content, ",", "\"", false);
    let mut parsed = FileParseResult::default();
    for file in files {
        let file = expanduser(file.trim()).to_string();
        if file.is_empty() {
            continue;
        }
        if fs::read_to_string(&file).is_ok() {
            parsed.text_files.push(file);
        } else if is_image_path(&file) {
            parsed.image_files.push(file);
        } else {
            parsed.binary_files.push(file);
        }
    }
    parsed
}

fn is_image_path(path: &str) -> bool {
    let Some(ext) = Path::new(path).extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "png" | "gif" | "webp" | "bmp" | "tif" | "tiff" | "ico" | "qoi" | "avif"
    )
}

fn image_mime_type(path: &str) -> &'static str {
    let Some(ext) = Path::new(path).extension().and_then(|ext| ext.to_str()) else {
        return "image/jpeg";
    };
    match ext.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "ico" => "image/x-icon",
        "qoi" => "image/qoi",
        "avif" => "image/avif",
        _ => "image/jpeg",
    }
}

fn text_file_contents(files: &[String]) -> io::Result<String> {
    let mut content = String::new();
    for file in files {
        content.push_str(&fs::read_to_string(file)?);
        content.push('\n');
    }
    Ok(content)
}

fn resolve_model_for_input(app: &App, question: &mut String) -> String {
    if let Some(model) = attachment_forced_model(
        &app.current_model,
        !app.attached_image_files.is_empty(),
        !app.attached_binary_files.is_empty(),
        &app.config.vl_default_model,
    ) {
        return model;
    }

    let trimmed = question.trim_end().to_string();
    if let Some(stripped) = trimmed.strip_suffix(" -code") {
        *question = stripped.to_string();
        return QWEN_CODER_PLUS_LATEST.to_string();
    }
    if let Some(stripped) = trimmed.strip_suffix(" -d") {
        *question = stripped.to_string();
        return DEEPSEEK_V3.to_string();
    }
    if let Some(selector) = trailing_model_selector(&trimmed) {
        if trimmed.len() >= 3 {
            *question = trimmed[..trimmed.len() - 3].trim_end().to_string();
        }
        return model_from_selector(selector, app.cli.thinking).to_string();
    }
    app.current_model.clone()
}

fn attachment_forced_model(
    current_model: &str,
    has_image_files: bool,
    has_binary_files: bool,
    vl_default_model: &str,
) -> Option<String> {
    if current_model == QWEN_LONG {
        return Some(QWEN_LONG.to_string());
    }
    if has_binary_files {
        return Some(QWEN_LONG.to_string());
    }
    if has_image_files && !is_vl_model(current_model) {
        return Some(determine_vl_model(vl_default_model));
    }
    None
}

fn trailing_model_selector(input: &str) -> Option<u8> {
    let bytes = input.as_bytes();
    if bytes.len() < 3 {
        return None;
    }
    let dash_idx = bytes.len() - 2;
    if bytes[dash_idx] != b'-' || !bytes[dash_idx + 1].is_ascii_digit() {
        return None;
    }
    if dash_idx == 0 || bytes[dash_idx - 1] != b' ' {
        return None;
    }
    Some((bytes[dash_idx + 1] - b'0') as u8)
}

fn do_request(
    app: &mut App,
    model: &str,
    question: &str,
    history_count: usize,
) -> Result<Response, Box<dyn std::error::Error>> {
    let mut request_body = RequestBody {
        model: model.to_string(),
        messages: vec![Message {
            role: "system".to_string(),
            content: Value::String("You are a helpful assistant.".to_string()),
        }],
        stream: true,
        enable_thinking: app.cli.thinking,
        enable_search: search_enabled(model).then_some(true),
    };

    let should_use_long_upload =
        (!app.attached_binary_files.is_empty() || !app.uploaded_file_ids.is_empty())
            && !is_vl_model(model);

    if should_use_long_upload {
        request_body.model = QWEN_LONG.to_string();

        let mut files_to_upload = app.attached_binary_files.clone();
        if !app.attached_image_files.is_empty() {
            files_to_upload.extend(app.attached_image_files.iter().cloned());
        }

        if !files_to_upload.is_empty() {
            app.uploaded_file_ids = upload_qwen_long_files(
                &app.client,
                &app.config.api_key,
                &files_to_upload,
            )?;
            app.attached_binary_files.clear();
            app.attached_image_files.clear();
        }

        let file_ids = app.uploaded_file_ids.join(",");
        request_body.messages.push(Message {
            role: "system".to_string(),
            content: Value::String(format!("fileid://{file_ids}")),
        });
    } else if !is_vl_model(model) {
        request_body
            .messages
            .extend(build_message_arr(history_count, &app.config.history_file)?);
    }

    request_body.messages.push(Message {
        role: "user".to_string(),
        content: build_content(&request_body.model, question, &app.attached_image_files)?,
    });

    let response = app
        .client
        .post(&app.config.endpoint)
        .bearer_auth(&app.config.api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("request failed: {status} {body}").into());
    }

    if is_vl_model(&request_body.model) {
        app.attached_image_files.clear();
    }

    Ok(response)
}

fn build_message_arr(
    history_count: usize,
    history_file: &PathBuf,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let history = match fs::read_to_string(history_file) {
        Ok(history) => history,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };

    let newline = NEWLINE.to_string();
    let lines = split_by_str_keep_quotes(&history, &newline, "\"", false);
    let mut messages = Vec::new();

    for line in &lines {
        if line.is_empty() {
            continue;
        }
        let Some(last_colon) = line.rfind(COLON) else {
            continue;
        };
        if last_colon == 0 || last_colon + COLON.len_utf8() >= line.len() {
            continue;
        }
        let role = &line[..last_colon];
        if role != "user" && role != "assistant" {
            continue;
        }
        let content = &line[last_colon + COLON.len_utf8()..];
        messages.push(Message {
            role: role.to_string(),
            content: Value::String(content.to_string()),
        });
    }

    if lines.len() > MAX_HISTORY_LINES {
        let start = lines.len() - MAX_HISTORY_LINES;
        let trimmed = lines[start..].join(&newline);
        fs::write(history_file, trimmed)?;
    }

    if history_count >= messages.len() {
        return Ok(messages);
    }
    Ok(messages[messages.len() - history_count..].to_vec())
}

fn build_content(
    model: &str,
    question: &str,
    image_files: &[String],
) -> Result<Value, Box<dyn std::error::Error>> {
    if !is_vl_model(model) || image_files.is_empty() {
        return Ok(Value::String(question.to_string()));
    }

    let mut parts = Vec::new();
    for file in image_files {
        let bytes = fs::read(file)?;
        let mime = image_mime_type(file);
        let image = base64::engine::general_purpose::STANDARD.encode(bytes);
        parts.push(json!({
            "type": "image_url",
            "image_url": format!("data:{mime};base64,{image}"),
        }));
    }
    parts.push(json!({
        "type": "text",
        "text": question,
    }));
    Ok(Value::Array(parts))
}

fn stream_response(
    app: &mut App,
    response: &mut Response,
    current_history: &mut String,
) -> Result<StreamOutcome, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(response);
    let thinking_tag = "<thinking>".yellow().to_string();
    let end_thinking_tag = "<end thinking>".yellow().to_string();
    let mut thinking_open = false;
    let mut markdown = MarkdownStreamRenderer::new();
    let mut line = String::new();

    while !app.shutdown.load(Ordering::SeqCst) {
        if take_stream_cancelled(app) {
            return Ok(StreamOutcome::Cancelled);
        }
        line.clear();
        let n = match reader.read_line(&mut line) {
            Ok(n) => n,
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                ) =>
            {
                if take_stream_cancelled(app) {
                    return Ok(StreamOutcome::Cancelled);
                }
                if app.shutdown.load(Ordering::SeqCst) {
                    return Ok(StreamOutcome::Cancelled);
                }
                continue;
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                if take_stream_cancelled(app) {
                    return Ok(StreamOutcome::Cancelled);
                }
                continue;
            }
            Err(err) => return Err(err.into()),
        };
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if !trimmed.starts_with("data:") {
            continue;
        }
        let payload = trimmed.trim_start_matches("data:").trim();
        if payload.is_empty() {
            continue;
        }
        if payload == "[DONE]" {
            break;
        }

        let chunk: StreamChunk = match serde_json::from_str(payload) {
            Ok(chunk) => chunk,
            Err(err) => {
                eprintln!("handleResponse error {err}");
                eprintln!("======> response: ");
                eprintln!("{payload}");
                eprintln!("<======");
                continue;
            }
        };
        let content = extract_chunk_text(&chunk, &thinking_tag, &end_thinking_tag, &mut thinking_open);
        if content.is_empty() {
            continue;
        }
        write_stream_content(content.as_str(), app.writer.as_mut(), &mut markdown)?;
        if thinking_open {
            continue;
        }
        let text = content.replace(&end_thinking_tag, "");
        let text = text.trim_matches('\n');
        current_history.push_str(text);
    }

    if take_stream_cancelled(app) {
        return Ok(StreamOutcome::Cancelled);
    }

    Ok(StreamOutcome::Completed)
}

fn take_stream_cancelled(app: &App) -> bool {
    app.cancel_stream.swap(false, Ordering::SeqCst)
}

fn extract_chunk_text(
    chunk: &StreamChunk,
    thinking_tag: &str,
    end_thinking_tag: &str,
    thinking_open: &mut bool,
) -> String {
    let Some(choice) = chunk.choices.first() else {
        return String::new();
    };
    let delta = &choice.delta;

    if delta.content.is_empty() && !delta.reasoning_content.is_empty() {
        if !*thinking_open {
            *thinking_open = true;
            return format!("\n{thinking_tag}\n{}", delta.reasoning_content);
        }
        return delta.reasoning_content.clone();
    }

    if *thinking_open {
        *thinking_open = false;
        return format!("\n{end_thinking_tag}\n{}", delta.content);
    }
    delta.content.clone()
}

fn write_stream_content(
    content: &str,
    mut writer: Option<&mut File>,
    markdown: &mut MarkdownStreamRenderer,
) -> io::Result<()> {
    if let Some(file) = writer.as_mut() {
        file.write_all(content.as_bytes())?;
        file.flush()?;
    }

    if markdown.should_render(content) {
        markdown.write_chunk(content)?;
    } else {
        print!("{content}");
    }
    io::stdout().flush()
}

struct MarkdownStreamRenderer {
    tty: bool,
    enabled: bool,
    in_code_block: bool,
    line_buf: String,
}

impl MarkdownStreamRenderer {
    fn new() -> Self {
        use std::io::IsTerminal;
        Self::new_with_tty(io::stdout().is_terminal())
    }

    fn new_with_tty(tty: bool) -> Self {
        Self {
            tty,
            enabled: true,
            in_code_block: false,
            line_buf: String::new(),
        }
    }

    fn should_render(&mut self, chunk: &str) -> bool {
        if !self.tty {
            return false;
        }
        if chunk.contains("\x1b[") {
            return false;
        }
        self.enabled = true;
        true
    }

    fn write_chunk(&mut self, chunk: &str) -> io::Result<()> {
        let mut out = io::stdout();
        for ch in chunk.chars() {
            if ch == '\n' {
                let line = std::mem::take(&mut self.line_buf);
                let rendered = self.render_line(&line);
                out.write_all(rendered.as_bytes())?;
                continue;
            }
            self.line_buf.push(ch);
        }
        Ok(())
    }

    fn render_line(&mut self, line: &str) -> String {
        let (indent, rest) = split_indent(line);
        let trimmed = rest.trim_start_matches([' ', '\t']);

        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            self.in_code_block = !self.in_code_block;
            return format!("{indent}\x1b[2m{trimmed}\x1b[0m\n");
        }

        if self.in_code_block {
            if line.is_empty() {
                return "\n".to_string();
            }
            return format!("\x1b[90m{line}\x1b[0m\n");
        }

        if let Some((level, title)) = parse_heading(trimmed) {
            let (base, underline_char) = match level {
                1 => ("\x1b[1m\x1b[35m", Some('═')),
                2 => ("\x1b[1m\x1b[36m", Some('─')),
                3 => ("\x1b[1m\x1b[34m", None),
                _ => ("\x1b[1m\x1b[36m", None),
            };
            let mut out = String::new();
            out.push_str(indent);
            out.push_str(base);
            out.push_str(&render_inline_md(title, base));
            out.push_str("\x1b[0m\n");

            if let Some(ch) = underline_char {
                let len = title.chars().count().max(3).min(80);
                out.push_str(indent);
                out.push_str("\x1b[2m\x1b[36m");
                out.push_str(&std::iter::repeat(ch).take(len).collect::<String>());
                out.push_str("\x1b[0m\n");
            }
            return out;
        }

        if let Some((p_indent, prefix, body)) = split_list_prefix(line) {
            let mut out = String::new();
            out.push_str(p_indent);
            out.push_str("\x1b[36m");
            out.push_str(prefix);
            out.push_str("\x1b[0m");
            out.push_str(&render_inline_md(body, ""));
            out.push('\n');
            return out;
        }

        if line.is_empty() {
            return "\n".to_string();
        }
        format!("{}{}\n", indent, render_inline_md(rest, ""))
    }
}

fn split_indent(s: &str) -> (&str, &str) {
    let mut idx = 0usize;
    for (i, ch) in s.char_indices() {
        if ch == ' ' || ch == '\t' {
            idx = i + ch.len_utf8();
            continue;
        }
        idx = i;
        break;
    }
    if s.chars().all(|c| c == ' ' || c == '\t') {
        return (s, "");
    }
    s.split_at(idx)
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i] == b'#' {
        i += 1;
    }
    if i == 0 || i > 6 {
        return None;
    }
    if i >= bytes.len() || bytes[i] != b' ' {
        return None;
    }
    Some((i, line[i + 1..].trim_end()))
}

fn split_list_prefix(line: &str) -> Option<(&str, &str, &str)> {
    let (indent, rest) = split_indent(line);
    let rest = rest.trim_end();
    if rest.starts_with("- ") || rest.starts_with("* ") || rest.starts_with("+ ") {
        return Some((indent, &rest[..2], &rest[2..]));
    }
    let bytes = rest.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        if i > 4 {
            break;
        }
    }
    if i == 0 || i + 1 >= bytes.len() {
        return None;
    }
    if bytes[i] == b'.' && bytes[i + 1] == b' ' {
        return Some((indent, &rest[..i + 2], &rest[i + 2..]));
    }
    None
}

fn render_inline_md(s: &str, base: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    let mut bold = false;
    let mut code = false;

    while i < bytes.len() {
        if bytes[i] == b'`' {
            code = !code;
            out.push_str("\x1b[0m");
            out.push_str(base);
            if bold {
                out.push_str("\x1b[1m");
            }
            if code {
                out.push_str("\x1b[7m");
            }
            i += 1;
            continue;
        }

        if !code && bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            bold = !bold;
            out.push_str("\x1b[0m");
            out.push_str(base);
            if bold {
                out.push_str("\x1b[1m");
            }
            if code {
                out.push_str("\x1b[7m");
            }
            i += 2;
            continue;
        }

        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    out.push_str("\x1b[0m");
    out
}

fn append_history(path: &PathBuf, content: &str) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.create(true).append(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o664);
    }
    let mut file = options.open(path)?;
    file.write_all(content.as_bytes())
}

fn print_info(model: &str) {
    let search = if search_enabled(model) { "true" } else { "false" };
    print!("[{} (search: {})] ", model.green(), search.red());
    let _ = io::stdout().flush();
}

fn upload_qwen_long_files(
    client: &Client,
    api_key: &str,
    files: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut ids = Vec::with_capacity(files.len());
    for file in files {
        ids.push(upload_single_qwen_long_file_with_retry(client, api_key, file, 5)?);
    }
    Ok(ids)
}

fn upload_single_qwen_long_file_with_retry(
    client: &Client,
    api_key: &str,
    filename: &str,
    retry: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut last_err: Option<Box<dyn std::error::Error>> = None;
    for _ in 0..retry {
        match upload_single_qwen_long_file(client, api_key, filename) {
            Ok(id) if !id.is_empty() => return Ok(id),
            Ok(_) => last_err = Some("empty file id".into()),
            Err(err) => last_err = Some(err),
        }
    }
    Err(last_err.unwrap_or_else(|| "upload failed".into()))
}

fn upload_single_qwen_long_file(
    client: &Client,
    api_key: &str,
    filename: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = PathBuf::from(filename);
    let display_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(filename);
    println!("Uploading file: {display_name}");

    let bytes = fs::read(filename)?;
    let part = multipart::Part::bytes(bytes).file_name(display_name.to_string());
    let form = multipart::Form::new()
        .part("file", part)
        .text("purpose", "file-extract");

    let response = client
        .post(FILES_ENDPOINT)
        .bearer_auth(api_key)
        .multipart(form)
        .send()?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("upload failed: {status} {body}").into());
    }
    let body: UploadResponse = response.json()?;
    println!("Finished upload. Fileid: {}", body.id);
    Ok(body.id)
}

fn search_enabled(model: &str) -> bool {
    ENABLE_SEARCH_MODELS.iter().any(|candidate| *candidate == model)
}

fn is_vl_model(model: &str) -> bool {
    VL_MODELS.iter().any(|candidate| *candidate == model)
}

fn levenshtein(left: &[u8], right: &[u8]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let mut prev: Vec<usize> = (0..=right.len()).collect();
    let mut curr = vec![0usize; right.len() + 1];
    for (i, left_byte) in left.iter().enumerate() {
        curr[0] = i + 1;
        for (j, right_byte) in right.iter().enumerate() {
            let cost = usize::from(left_byte != right_byte);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn selector_mapping_matches_go() {
        assert_eq!(model_from_selector(0, false), QWQ);
        assert_eq!(model_from_selector(1, false), QWEN_PLUS_LATEST);
        assert_eq!(model_from_selector(2, false), QWEN_MAX);
        assert_eq!(model_from_selector(3, false), QWEN3_MAX);
        assert_eq!(model_from_selector(4, false), QWEN_CODER_PLUS_LATEST);
        assert_eq!(model_from_selector(5, false), DEEPSEEK_V3);
        assert_eq!(model_from_selector(5, true), DEEPSEEK_R1);
        assert_eq!(model_from_selector(6, false), QWEN_FLASH);
    }

    #[test]
    fn loop_overrides_preserve_question_but_change_history() {
        let overrides = loop_overrides("hello world -s --history 2");
        assert!(overrides.short_output);
        assert_eq!(overrides.history_count, Some(2));
    }

    #[test]
    fn trailing_selector_is_detected() {
        assert_eq!(trailing_model_selector("hello -3"), Some(3));
        assert_eq!(trailing_model_selector("hello-3"), None);
        assert_eq!(trailing_model_selector("hello -33"), None);
    }

    #[test]
    fn image_files_auto_route_to_vl() {
        let model = attachment_forced_model(QWEN_FLASH, true, false, QWEN_VL_FLASH);
        assert_eq!(model, Some(QWEN_VL_FLASH.to_string()));
    }

    #[test]
    fn binary_files_auto_route_to_long() {
        let model = attachment_forced_model(QWEN_FLASH, true, true, QWEN_VL_FLASH);
        assert_eq!(model, Some(QWEN_LONG.to_string()));
    }

    #[test]
    fn configured_vl_model_is_used_for_images() {
        let model = attachment_forced_model(QWEN_FLASH, true, false, QWEN_VL_MAX);
        assert_eq!(model, Some(QWEN_VL_MAX.to_string()));
    }

    #[test]
    fn determine_vl_model_supports_selector_and_fuzzy_name() {
        assert_eq!(determine_vl_model(""), QWEN_VL_FLASH);
        assert_eq!(determine_vl_model("1"), QWEN_VL_MAX);
        assert_eq!(determine_vl_model("qwen-vl-ocr-latst"), QWEN_VL_OCR);
    }

    #[test]
    fn image_file_detection_by_suffix() {
        assert!(is_image_path("/tmp/hello.png"));
        assert!(is_image_path("/tmp/hello.JPEG"));
        assert!(!is_image_path("/tmp/hello.pdf"));
    }

    #[test]
    fn image_mime_type_matches_suffix() {
        assert_eq!(image_mime_type("a.png"), "image/png");
        assert_eq!(image_mime_type("a.jpg"), "image/jpeg");
        assert_eq!(image_mime_type("a.unknown"), "image/jpeg");
    }

    #[test]
    fn history_file_parsing_matches_go_format() {
        let path = std::env::temp_dir().join(format!("ai-history-{}.txt", uuid::Uuid::new_v4()));
        fs::write(
            &path,
            format!("user{COLON}hi{NEWLINE}assistant{COLON}hello{NEWLINE}"),
        )
        .unwrap();

        let messages = build_message_arr(4, &path).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, Value::String("hi".to_string()));
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, Value::String("hello".to_string()));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn thinking_chunks_are_wrapped_once() {
        let chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: String::new(),
                    reasoning_content: "step one".to_string(),
                },
            }],
        };
        let mut thinking_open = false;
        let text = extract_chunk_text(&chunk, "<thinking>", "<end thinking>", &mut thinking_open);
        assert_eq!(text, "\n<thinking>\nstep one");
        assert!(thinking_open);

        let chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: "final".to_string(),
                    reasoning_content: String::new(),
                },
            }],
        };
        let text = extract_chunk_text(&chunk, "<thinking>", "<end thinking>", &mut thinking_open);
        assert_eq!(text, "\n<end thinking>\nfinal");
        assert!(!thinking_open);
    }

    #[test]
    fn multiline_history_navigation_restores_draft() {
        let mut history = MultilineHistoryState::new(vec![
            "first".to_string(),
            "second\nline".to_string(),
        ]);

        assert_eq!(history.previous("draft"), Some("second\nline".to_string()));
        assert_eq!(history.previous("ignored"), Some("first".to_string()));
        assert_eq!(history.previous("ignored"), None);
        assert_eq!(history.next(), Some("second\nline".to_string()));
        assert_eq!(history.next(), Some("draft".to_string()));
        assert_eq!(history.next(), None);
    }

    #[test]
    fn sigint_during_stream_only_cancels_current_reply() {
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(true);
        let cancel_stream = AtomicBool::new(false);

        handle_sigint(&shutdown, &streaming, &cancel_stream);

        assert!(!shutdown.load(Ordering::SeqCst));
        assert!(cancel_stream.load(Ordering::SeqCst));
    }

    #[test]
    fn sigint_while_idle_requests_shutdown() {
        let shutdown = AtomicBool::new(false);
        let streaming = AtomicBool::new(false);
        let cancel_stream = AtomicBool::new(false);

        handle_sigint(&shutdown, &streaming, &cancel_stream);

        assert!(shutdown.load(Ordering::SeqCst));
        assert!(!cancel_stream.load(Ordering::SeqCst));
    }
}
