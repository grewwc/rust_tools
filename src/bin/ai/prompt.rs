use std::{
    fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
};

use rustyline::{DefaultEditor, error::ReadlineError};
use uuid::Uuid;

use super::history::SessionStore;
use crate::clipboard::image_content;
use crate::clipboard::string_content;
use crate::common::utils::expanduser;

const LINE_REPL_HISTORY_FILE: &str = "~/.liner_histroy";

pub(super) struct PromptEditor {
    pub(super) editor: Option<DefaultEditor>,
    pub(super) history_path: PathBuf,
    session_image_dir: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct MultilineHistoryState {
    entries: Vec<String>,
    index: Option<usize>,
    draft: Option<String>,
}

impl PromptEditor {
    pub(super) fn new(session_id: &str, history_file: &Path) -> Self {
        let mut editor = DefaultEditor::new().ok();
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
        let Some(editor) = self.editor.as_mut() else {
            print!("> ");
            io::stdout().flush()?;
            let mut line = String::new();
            match io::stdin().read_line(&mut line) {
                Ok(0) => return Ok(None),
                Ok(_) => return Ok(Some(trim_trailing_newline(line))),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                    return exit_on_interrupt();
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
            Err(ReadlineError::Interrupted) => exit_on_interrupt(),
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

    fn read_multi_line_tui(&mut self) -> io::Result<Option<String>> {
        use crossterm::{
            event::{
                self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind,
                KeyModifiers,
            },
            execute,
            terminal::{disable_raw_mode, enable_raw_mode, size as terminal_size},
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

        enable_raw_mode()?;
        let _ = execute!(io::stdout(), EnableBracketedPaste);

        let viewport_height = terminal_size()
            .map(|(_, h)| h.saturating_sub(10).clamp(12, 24))
            .unwrap_or(18);

        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = match Terminal::with_options(
            backend,
            ratatui::TerminalOptions {
                viewport: ratatui::Viewport::Inline(viewport_height),
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
            let mut accept_release = false;

            loop {
                terminal
                    .draw(|f| {
                        let area = f.area();
                        let popup_height = area.height.saturating_sub(4).clamp(10, 18);
                        let popup_width =
                            area.width.saturating_sub(2).clamp(60, 180).min(area.width);
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
                                    Span::styled("Esc/Ctrl+D", Style::default().fg(Color::Green)),
                                    Span::raw(" send  ·  "),
                                    Span::styled("Ctrl+C", Style::default().fg(Color::Yellow)),
                                    Span::raw(" cancel"),
                                ]),
                                Line::from(vec![
                                    Span::raw("  "),
                                    Span::styled("Up/Down edge", Style::default().fg(Color::Blue)),
                                    Span::raw(" or "),
                                    Span::styled("Ctrl+P/N", Style::default().fg(Color::Blue)),
                                    Span::raw(" history  ·  "),
                                    Span::styled("Backspace", Style::default().fg(Color::Blue)),
                                    Span::raw(" edits previous lines  ·  "),
                                    Span::styled("Paste", Style::default().fg(Color::Blue)),
                                    Span::raw(" image"),
                                ]),
                            ]),
                            chunks[1],
                        );
                    })
                    .map_err(|e| io::Error::other(e.to_string()))?;

                match event::read().map_err(|e| io::Error::other(e.to_string()))? {
                    Event::Paste(pasted) => {
                        // 优先尝试从剪贴板保存图片（模仿 a -f 行为）
                        match save_clipboard_images(&self.session_image_dir) {
                            Ok(paths) if !paths.is_empty() => {
                                // 保存图片成功，插入所有图片的文件路径
                                for path in paths {
                                    let placeholder = image_placeholder(&path);
                                    insert_text(&mut textarea, &placeholder);
                                }
                            }
                            _ => {
                                // 保存失败或没有图片，插入原始文本
                                insert_text(&mut textarea, &pasted);
                            }
                        }
                    }
                    Event::Key(mut key) => {
                        if key.kind == KeyEventKind::Release {
                            accept_release = true;
                            key.kind = KeyEventKind::Press;
                        } else if key.kind != KeyEventKind::Press
                            && key.kind != KeyEventKind::Repeat
                            && !accept_release
                        {
                            continue;
                        }
                        match (key.code, key.modifiers) {
                            (code, modifiers) if is_submit_key(code, modifiers) => {
                                let content = textarea_content(&textarea);
                                let trimmed = content.trim_end_matches('\n').to_string();
                                break Ok(if trimmed.trim().is_empty() {
                                    None
                                } else {
                                    Some(trimmed)
                                });
                            }
                            // 检测 ctrl+v (CONTROL + v) 粘贴图片
                            (KeyCode::Char('v'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                                // 优先尝试从剪贴板保存图片（模仿 a -f 行为）
                                match save_clipboard_images(&self.session_image_dir) {
                                    Ok(paths) if !paths.is_empty() => {
                                        // 保存图片成功，插入所有图片的文件路径
                                        for path in paths {
                                            let placeholder = image_placeholder(&path);
                                            insert_text(&mut textarea, &placeholder);
                                        }
                                        continue; // 跳过后续处理
                                    }
                                    _ => {
                                        // 保存失败或没有图片，尝试粘贴文本
                                        let clipboard_text = crate::clipboard::string_content::get_clipboard_content();
                                        if !clipboard_text.is_empty() {
                                            insert_text(&mut textarea, &clipboard_text);
                                            continue;
                                        }
                                    }
                                }
                            }
                            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                break Err(io::Error::new(io::ErrorKind::Interrupted, "Ctrl+C"));
                            }
                            (KeyCode::Char('a'), modifiers)
                                if modifiers.contains(KeyModifiers::SUPER)
                                    || modifiers.contains(KeyModifiers::CONTROL) =>
                            {
                                let content = textarea_content(&textarea);
                                let _ = string_content::set_clipboard_content(&content);
                            }
                            (KeyCode::Backspace | KeyCode::Delete, modifiers)
                                if modifiers.contains(KeyModifiers::SUPER) =>
                            {
                                delete_current_line(&mut textarea);
                            }
                            (KeyCode::Char('u'), modifiers)
                                if modifiers.contains(KeyModifiers::CONTROL)
                                    || modifiers.contains(KeyModifiers::SUPER) =>
                            {
                                delete_current_line(&mut textarea);
                            }
                            _ => {
                                let handled = match (key.code, key.modifiers) {
                                    (KeyCode::Up, modifiers)
                                        if modifiers.is_empty() && textarea.cursor().0 == 0 =>
                                    {
                                        if let Some(content) =
                                            history.previous(&textarea_content(&textarea))
                                        {
                                            replace_textarea_content(&mut textarea, &content);
                                            true
                                        } else {
                                            false
                                        }
                                    }
                                    (KeyCode::Down, modifiers)
                                        if modifiers.is_empty()
                                            && textarea.cursor().0 + 1
                                                >= textarea.lines().len() =>
                                    {
                                        if let Some(content) = history.next() {
                                            replace_textarea_content(&mut textarea, &content);
                                            true
                                        } else {
                                            false
                                        }
                                    }
                                    (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                                        if let Some(content) =
                                            history.previous(&textarea_content(&textarea))
                                        {
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
                    _ => {}
                }
            }
        })();

        let _ = terminal.clear();
        let _ = terminal.show_cursor();
        let _ = execute!(io::stdout(), DisableBracketedPaste);
        let _ = disable_raw_mode();

        let result = match result {
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                return exit_on_interrupt();
            }
            Err(err) => return Err(err),
            Ok(result) => result,
        };
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
    pub(super) fn new(entries: Vec<String>) -> Self {
        Self {
            entries,
            index: None,
            draft: None,
        }
    }

    pub(super) fn previous(&mut self, current: &str) -> Option<String> {
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

    pub(super) fn next(&mut self) -> Option<String> {
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

fn exit_on_interrupt() -> io::Result<Option<String>> {
    println!("Exit.");
    #[cfg(not(test))]
    std::process::exit(130);
    #[cfg(test)]
    {
        Ok(None)
    }
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

fn delete_current_line(textarea: &mut tui_textarea::TextArea<'_>) {
    let (row, col) = textarea.cursor();
    let mut lines = textarea.lines().to_vec();
    if lines.len() <= 1 {
        textarea.set_lines(vec![String::new()], (0, 0));
        return;
    }
    let remove_at = row.min(lines.len() - 1);
    lines.remove(remove_at);
    let new_row = remove_at.min(lines.len() - 1);
    let new_col = col.min(lines[new_row].len());
    textarea.set_lines(lines, (new_row, new_col));
}

fn insert_text(textarea: &mut tui_textarea::TextArea<'_>, text: &str) {
    if text.is_empty() {
        return;
    }
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    if !normalized.contains('\n') {
        textarea.insert_str(&normalized);
        return;
    }

    let lines: Vec<&str> = normalized.split('\n').collect();
    for (idx, line) in lines.iter().enumerate() {
        if !line.is_empty() {
            textarea.insert_str(line);
        }
        if idx + 1 < lines.len() {
            let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
            textarea.input(tui_textarea::Input::from(enter));
        }
    }
}

fn image_placeholder(path: &Path) -> String {
    format!("[[image:{}]]", path.display())
}

/// 从剪贴板保存图片（可能多个），返回保存的文件路径列表
fn save_clipboard_images(dir: &Path) -> io::Result<Vec<PathBuf>> {
    fs::create_dir_all(dir)?;
    let mut paths = Vec::new();
    
    // 尝试保存一张图片
    let path = dir.join(format!("paste-{}.png", Uuid::new_v4()));
    match image_content::save_to_file(path.to_string_lossy().as_ref()) {
        Ok(()) => {
            paths.push(path);
        }
        Err(e) => {
            return Ok(paths);
        }
    }
    
    // 注意：arboard 目前只支持获取单张图片，所以这里只保存一张
    // 如果未来需要支持多张图片，可以在这里添加循环逻辑
    
    Ok(paths)
}

fn is_submit_key(
    code: crossterm::event::KeyCode,
    modifiers: crossterm::event::KeyModifiers,
) -> bool {
    use crossterm::event::{KeyCode, KeyModifiers};

    matches!(
        (code, modifiers),
        (KeyCode::Char('d'), KeyModifiers::CONTROL) | (KeyCode::Esc, KeyModifiers::NONE)
    )
}

pub(super) fn trim_trailing_newline(mut line: String) -> String {
    while matches!(line.chars().last(), Some('\n' | '\r')) {
        line.pop();
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

    #[test]
    fn submit_key_recognizes_ctrl_d() {
        assert!(is_submit_key(KeyCode::Char('d'), KeyModifiers::CONTROL));
        assert!(!is_submit_key(KeyCode::Char('d'), KeyModifiers::NONE));
    }

    #[test]
    fn submit_key_recognizes_esc() {
        assert!(is_submit_key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!is_submit_key(KeyCode::Esc, KeyModifiers::SHIFT));
    }
}
