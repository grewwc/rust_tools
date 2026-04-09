use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tui_textarea::{CursorMove, Input, TextArea};
use uuid::Uuid;

use super::{
    MultilineHistoryState,
    completion_panel::{
        CompletionPanel, PendingTabCompletion, apply_multiline_completion,
        confirm_completion_selection, dismiss_completion_panel, move_completion_selection,
    },
};
use crate::{
    clipboardw::{image_content, string_content},
};

pub(in crate::ai::prompt::multiline) enum EventLoopAction {
    Continue,
    Submit(Option<String>),
}

pub(in crate::ai::prompt::multiline) fn handle_multiline_event(
    event: Event,
    textarea: &mut TextArea<'_>,
    history: &mut MultilineHistoryState,
    accept_release: &mut bool,
    status_msg: &mut Option<String>,
    pending_tab_completion: &mut Option<PendingTabCompletion>,
    completion_panel: &mut Option<CompletionPanel>,
    session_image_dir: &Path,
) -> io::Result<EventLoopAction> {
    match event {
        Event::Paste(pasted) => {
            dismiss_completion_panel(pending_tab_completion, completion_panel);
            match save_clipboard_images(session_image_dir) {
                Ok(paths) if !paths.is_empty() => {
                    for path in paths {
                        let placeholder = image_placeholder(&path);
                        insert_text(textarea, &placeholder);
                    }
                }
                _ => {
                    insert_text(textarea, &pasted);
                }
            }
            Ok(EventLoopAction::Continue)
        }
        Event::Key(mut key) => {
            *status_msg = None;
            if key.kind == KeyEventKind::Release {
                *accept_release = true;
                key.kind = KeyEventKind::Press;
            } else if key.kind != KeyEventKind::Press
                && key.kind != KeyEventKind::Repeat
                && !*accept_release
            {
                return Ok(EventLoopAction::Continue);
            }

            if !matches!((key.code, key.modifiers), (KeyCode::Tab, modifiers) if modifiers.is_empty())
                && !matches!(
                    (key.code, key.modifiers),
                    (KeyCode::Up, modifiers)
                        | (KeyCode::Down, modifiers)
                        | (KeyCode::Enter, modifiers)
                        | (KeyCode::Esc, modifiers)
                        if modifiers.is_empty() && completion_panel.is_some()
                )
            {
                dismiss_completion_panel(pending_tab_completion, completion_panel);
            }

            match (key.code, key.modifiers) {
                (KeyCode::Up, modifiers) if modifiers.is_empty() && completion_panel.is_some() => {
                    move_completion_selection(completion_panel, -1);
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::Down, modifiers)
                    if modifiers.is_empty() && completion_panel.is_some() =>
                {
                    move_completion_selection(completion_panel, 1);
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::Enter, modifiers)
                    if modifiers.is_empty() && completion_panel.is_some() =>
                {
                    *status_msg =
                        confirm_completion_selection(textarea, pending_tab_completion, completion_panel);
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::Esc, modifiers) if modifiers.is_empty() && completion_panel.is_some() => {
                    dismiss_completion_panel(pending_tab_completion, completion_panel);
                    Ok(EventLoopAction::Continue)
                }
                (code, modifiers) if is_submit_key(code, modifiers) => {
                    let content = textarea_content(textarea);
                    let trimmed = content.trim_end_matches('\n').to_string();
                    Ok(EventLoopAction::Submit(if trimmed.trim().is_empty() {
                        None
                    } else {
                        Some(trimmed)
                    }))
                }
                (KeyCode::Char('v'), modifiers)
                    if modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    match save_clipboard_images(session_image_dir) {
                        Ok(paths) if !paths.is_empty() => {
                            for path in paths {
                                let placeholder = image_placeholder(&path);
                                insert_text(textarea, &placeholder);
                            }
                            return Ok(EventLoopAction::Continue);
                        }
                        _ => {
                            let clipboard_text = string_content::get_clipboard_content();
                            if !clipboard_text.is_empty() {
                                insert_text(textarea, &clipboard_text);
                                return Ok(EventLoopAction::Continue);
                            }
                        }
                    }
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                    Err(io::Error::new(io::ErrorKind::Interrupted, "Ctrl+C"))
                }
                (KeyCode::Tab, modifiers) if modifiers.is_empty() => {
                    *status_msg = apply_multiline_completion(
                        textarea,
                        pending_tab_completion,
                        completion_panel,
                    );
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::F(9), _) => {
                    let lines = textarea.lines();
                    let mut last_answer_start = None;
                    for (i, line) in lines.iter().enumerate() {
                        if line.starts_with("assistant:")
                            || line.starts_with("AI:")
                            || line.starts_with("Assistant:")
                        {
                            last_answer_start = Some(i);
                        }
                    }
                    let answer = if let Some(start) = last_answer_start {
                        lines[start..].join("\n")
                    } else {
                        lines.join("\n")
                    };
                    let _ = string_content::set_clipboard_content(&answer);
                    *status_msg = Some("? ???????????".to_string());
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::F(10), _) => {
                    let content = textarea_content(textarea);
                    let _ = string_content::set_clipboard_content(&content);
                    *status_msg = Some("? ?????????".to_string());
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::Backspace | KeyCode::Delete, modifiers)
                    if modifiers.contains(KeyModifiers::SUPER) =>
                {
                    delete_current_line(textarea);
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::Char('u'), modifiers)
                    if modifiers.contains(KeyModifiers::CONTROL)
                        || modifiers.contains(KeyModifiers::SUPER) =>
                {
                    delete_current_line(textarea);
                    Ok(EventLoopAction::Continue)
                }
                _ => {
                    let handled = match (key.code, key.modifiers) {
                        (KeyCode::Up, modifiers)
                            if modifiers.is_empty() && textarea.cursor().0 == 0 =>
                        {
                            if let Some(content) = history.previous(&textarea_content(textarea)) {
                                replace_textarea_content(textarea, &content);
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
                                replace_textarea_content(textarea, &content);
                                true
                            } else {
                                false
                            }
                        }
                        (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                            if let Some(content) = history.previous(&textarea_content(textarea)) {
                                replace_textarea_content(textarea, &content);
                                true
                            } else {
                                false
                            }
                        }
                        (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
                            if let Some(content) = history.next() {
                                replace_textarea_content(textarea, &content);
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
                        return Ok(EventLoopAction::Continue);
                    }

                    textarea.input(Input::from(key));
                    Ok(EventLoopAction::Continue)
                }
            }
        }
        Event::Resize(_, _) => {
            *completion_panel = None;
            Ok(EventLoopAction::Continue)
        }
        _ => Ok(EventLoopAction::Continue),
    }
}

fn textarea_content(textarea: &TextArea<'_>) -> String {
    textarea.lines().join("\n")
}

fn replace_textarea_content(textarea: &mut TextArea<'_>, content: &str) {
    let lines = content.split('\n').map(|line| line.to_string()).collect();
    *textarea = TextArea::new(lines);
}

fn delete_current_line(textarea: &mut TextArea<'_>) {
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

fn insert_text(textarea: &mut TextArea<'_>, text: &str) {
    if text.is_empty() {
        return;
    }

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
            textarea.input(Input::from(enter));
        }
    }
}

fn image_placeholder(path: &Path) -> String {
    format!("[[image:{}]]", path.display())
}

fn save_clipboard_images(dir: &Path) -> io::Result<Vec<PathBuf>> {
    fs::create_dir_all(dir)?;
    let mut paths = Vec::new();

    let path = dir.join(format!("paste-{}.png", Uuid::new_v4()));
    match image_content::save_to_file(path.to_string_lossy().as_ref()) {
        Ok(()) => {
            paths.push(path);
        }
        Err(_e) => {
            return Ok(paths);
        }
    }

    Ok(paths)
}

fn is_submit_key(
    code: crossterm::event::KeyCode,
    modifiers: crossterm::event::KeyModifiers,
) -> bool {
    matches!(
        (code, modifiers),
        (KeyCode::Esc, KeyModifiers::NONE)
            | (KeyCode::Enter, KeyModifiers::ALT)
            | (KeyCode::F(2), KeyModifiers::NONE)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_key_recognizes_ctrl_d() {
        assert!(!is_submit_key(KeyCode::Char('d'), KeyModifiers::CONTROL));
        assert!(!is_submit_key(KeyCode::Char('d'), KeyModifiers::NONE));
    }

    #[test]
    fn submit_key_recognizes_esc() {
        assert!(is_submit_key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!is_submit_key(KeyCode::Esc, KeyModifiers::SHIFT));
    }
}
