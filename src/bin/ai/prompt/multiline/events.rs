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
use crate::clipboardw::{image_content, string_content};

pub(in crate::ai::prompt::multiline) enum EventLoopAction {
    Continue,
    Submit(Option<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::prompt::multiline) enum RecentTextInputSource {
    KeyPress,
    Paste,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai::prompt::multiline) struct RecentTextInput {
    text: String,
    source: RecentTextInputSource,
}

pub(in crate::ai::prompt::multiline) fn handle_multiline_event(
    event: Event,
    textarea: &mut TextArea<'_>,
    history: &mut MultilineHistoryState,
    status_msg: &mut Option<String>,
    pending_tab_completion: &mut Option<PendingTabCompletion>,
    completion_panel: &mut Option<CompletionPanel>,
    recent_text_input: &mut Option<RecentTextInput>,
    session_image_dir: &Path,
) -> io::Result<EventLoopAction> {
    match event {
        Event::Paste(pasted) => {
            if should_skip_duplicate_text_input(
                recent_text_input,
                &pasted,
                RecentTextInputSource::Paste,
            ) {
                return Ok(EventLoopAction::Continue);
            }
            dismiss_completion_panel(pending_tab_completion, completion_panel);
            match save_clipboard_images(session_image_dir) {
                Ok(paths) if !paths.is_empty() => {
                    recent_text_input.take();
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
        Event::Key(key) => {
            *status_msg = None;
            if matches!(key.kind, KeyEventKind::Release) {
                return Ok(EventLoopAction::Continue);
            }
            if should_skip_non_ascii_repeat_or_duplicate_press(recent_text_input, key) {
                return Ok(EventLoopAction::Continue);
            }
            if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
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
                    let selection = confirm_completion_selection(
                        textarea,
                        pending_tab_completion,
                        completion_panel,
                    );
                    *status_msg = selection.status;
                    if let Some(submit) = selection.submit {
                        Ok(EventLoopAction::Submit(Some(submit)))
                    } else {
                        Ok(EventLoopAction::Continue)
                    }
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
                (KeyCode::Char('v'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
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
                (KeyCode::F(8), _) => {
                    // F8：一键清空输入框中的所有内容。
                    replace_textarea_content(textarea, "");
                    recent_text_input.take();
                    *status_msg = Some("Cleared all input.".to_string());
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
                    *status_msg = Some("Copied to clipboard!".to_string());
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::F(10), _) => {
                    let content = textarea_content(textarea);
                    let _ = string_content::set_clipboard_content(&content);
                    *status_msg = Some("Copied to clipboard!".to_string());
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::Backspace | KeyCode::Delete, modifiers)
                    if modifiers.contains(KeyModifiers::SUPER) =>
                {
                    delete_current_line(textarea);
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::Backspace, modifiers) if modifiers.is_empty() => {
                    // 显式处理 Backspace：按字符（而非字节）删除前一个字符，
                    // 确保 CJK 等多字节 Unicode 字符能正确删除。
                    // tui-textarea-2 v0.10.2 在某些终端/IME 组合下对中文退格可能失效。
                    backspace_delete_char(textarea);
                    Ok(EventLoopAction::Continue)
                }
                (KeyCode::Delete, modifiers) if modifiers.is_empty() => {
                    // 显式处理 Delete 键：按字符删除光标后一个字符。
                    delete_char_forward(textarea);
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

                    remember_trackable_key_press(recent_text_input, key);
                    textarea.input(Input::from(key));
                    Ok(EventLoopAction::Continue)
                }
            }
        }
        Event::Resize(_, _) => {
            recent_text_input.take();
            *completion_panel = None;
            Ok(EventLoopAction::Continue)
        }
        _ => {
            recent_text_input.take();
            Ok(EventLoopAction::Continue)
        }
    }
}

fn normalize_trackable_text_input(text: &str) -> Option<String> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let char_count = normalized.chars().count();
    if normalized.is_empty()
        || normalized.contains('\n')
        || char_count > 16
        || !normalized.chars().any(|ch| !ch.is_ascii())
    {
        return None;
    }
    Some(normalized)
}

fn should_skip_duplicate_text_input(
    recent_text_input: &mut Option<RecentTextInput>,
    text: &str,
    source: RecentTextInputSource,
) -> bool {
    let Some(normalized) = normalize_trackable_text_input(text) else {
        recent_text_input.take();
        return false;
    };

    if recent_text_input
        .as_ref()
        .map(|last| last.text == normalized && last.source != source)
        .unwrap_or(false)
    {
        recent_text_input.take();
        return true;
    }

    *recent_text_input = Some(RecentTextInput {
        text: normalized,
        source,
    });
    false
}

fn should_skip_non_ascii_repeat_or_duplicate_press(
    recent_text_input: &mut Option<RecentTextInput>,
    key: KeyEvent,
) -> bool {
    let KeyCode::Char(ch) = key.code else {
        recent_text_input.take();
        return false;
    };
    let Some(normalized) = normalize_trackable_text_input(&ch.to_string()) else {
        recent_text_input.take();
        return false;
    };

    if matches!(key.kind, KeyEventKind::Repeat) {
        return true;
    }

    if matches!(key.kind, KeyEventKind::Press)
        && recent_text_input
            .as_ref()
            .map(|last| last.text == normalized && last.source == RecentTextInputSource::Paste)
            .unwrap_or(false)
    {
        recent_text_input.take();
        return true;
    }

    false
}

fn remember_trackable_key_press(recent_text_input: &mut Option<RecentTextInput>, key: KeyEvent) {
    let KeyCode::Char(ch) = key.code else {
        recent_text_input.take();
        return;
    };
    if !matches!(key.kind, KeyEventKind::Press) {
        return;
    }
    let Some(normalized) = normalize_trackable_text_input(&ch.to_string()) else {
        recent_text_input.take();
        return;
    };
    *recent_text_input = Some(RecentTextInput {
        text: normalized,
        source: RecentTextInputSource::KeyPress,
    });
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

/// 显式处理 Backspace：按字符（而非字节）删除光标前一个字符。
/// 使用 chars() 操作确保 CJK 等多字节 Unicode 字符正确删除，
/// 绕过 tui-textarea-2 v0.10.2 在某些终端/IME 组合下对中文退格失效的问题。
fn backspace_delete_char(textarea: &mut TextArea<'_>) {
    let (row, col) = textarea.cursor();
    let mut lines = textarea.lines().to_vec();

    if col > 0 {
        // 删除当前行中光标前的一个字符（按字符索引，非字节）
        let line = &mut lines[row];
        let mut chars: Vec<char> = line.chars().collect();
        if col <= chars.len() {
            chars.remove(col - 1);
            *line = chars.into_iter().collect();
            textarea.set_lines(lines, (row, col - 1));
        }
    } else if row > 0 {
        // 光标在行首，合并到上一行末尾
        // 注意：textarea 的 cursor 列是字符索引，不是字节索引。
        // 必须用 chars().count() 而非 len()（后者返回字节数）。
        let prev_line_char_count = lines[row - 1].chars().count();
        let merged = format!("{}{}", lines[row - 1], lines[row]);
        lines[row - 1] = merged;
        lines.remove(row);
        let new_row = row - 1;
        let new_col = prev_line_char_count;
        textarea.set_lines(lines, (new_row, new_col));
    }
}

/// 显式处理 Delete 键：按字符删除光标后一个字符。
fn delete_char_forward(textarea: &mut TextArea<'_>) {
    let (row, col) = textarea.cursor();
    let mut lines = textarea.lines().to_vec();
    let line_len = lines[row].chars().count();

    if col < line_len {
        // 删除光标后的一个字符
        let line = &mut lines[row];
        let mut chars: Vec<char> = line.chars().collect();
        chars.remove(col);
        *line = chars.into_iter().collect();
        textarea.set_lines(lines, (row, col));
    } else if row + 1 < lines.len() {
        // 光标在行尾，合并下一行
        let next_line = lines.remove(row + 1);
        lines[row].push_str(&next_line);
        textarea.set_lines(lines, (row, col));
    }
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

/// Generate a compact but clear placeholder for pasted images.
/// Shows only the filename (not full path) to keep it readable in the editor.
fn image_placeholder(path: &Path) -> String {
    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "image.png".to_string());
    // Format: [[image:paste-xxxx.png]] — short enough to read at a glance,
    // still uniquely identifies the saved asset.
    format!("[[image:{}]]", filename)
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
    use std::path::Path;

    fn apply_event(
        textarea: &mut TextArea<'_>,
        recent_text_input: &mut Option<RecentTextInput>,
        event: Event,
    ) {
        let mut history = MultilineHistoryState::new(Vec::new());
        let mut status_msg = None;
        let mut pending_tab_completion = None;
        let mut completion_panel = None;
        let action = handle_multiline_event(
            event,
            textarea,
            &mut history,
            &mut status_msg,
            &mut pending_tab_completion,
            &mut completion_panel,
            recent_text_input,
            Path::new("/tmp"),
        )
        .unwrap();
        assert!(matches!(action, EventLoopAction::Continue));
    }

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

    #[test]
    fn chinese_ime_press_release_does_not_insert_twice() {
        let mut textarea = TextArea::default();
        let mut recent_text_input = None;

        apply_event(
            &mut textarea,
            &mut recent_text_input,
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('中'),
                KeyModifiers::NONE,
                KeyEventKind::Press,
            )),
        );
        apply_event(
            &mut textarea,
            &mut recent_text_input,
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('中'),
                KeyModifiers::NONE,
                KeyEventKind::Release,
            )),
        );

        assert_eq!(textarea.lines(), &["中".to_string()]);
    }

    #[test]
    fn non_ascii_key_repeat_is_ignored_to_avoid_ime_duplicates() {
        let mut textarea = TextArea::default();
        let mut recent_text_input = None;

        apply_event(
            &mut textarea,
            &mut recent_text_input,
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('哈'),
                KeyModifiers::NONE,
                KeyEventKind::Press,
            )),
        );
        apply_event(
            &mut textarea,
            &mut recent_text_input,
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('哈'),
                KeyModifiers::NONE,
                KeyEventKind::Repeat,
            )),
        );

        assert_eq!(textarea.lines(), &["哈".to_string()]);
    }

    #[test]
    fn ascii_key_repeat_still_inserts_repeated_character() {
        let mut textarea = TextArea::default();
        let mut recent_text_input = None;

        apply_event(
            &mut textarea,
            &mut recent_text_input,
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('a'),
                KeyModifiers::NONE,
                KeyEventKind::Press,
            )),
        );
        apply_event(
            &mut textarea,
            &mut recent_text_input,
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('a'),
                KeyModifiers::NONE,
                KeyEventKind::Repeat,
            )),
        );

        assert_eq!(textarea.lines(), &["aa".to_string()]);
    }

    #[test]
    fn chinese_paste_then_key_press_is_deduped() {
        let mut textarea = TextArea::default();
        let mut recent_text_input = None;

        assert!(!should_skip_duplicate_text_input(
            &mut recent_text_input,
            "中",
            RecentTextInputSource::Paste,
        ));
        insert_text(&mut textarea, "中");
        apply_event(
            &mut textarea,
            &mut recent_text_input,
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('中'),
                KeyModifiers::NONE,
                KeyEventKind::Press,
            )),
        );

        assert_eq!(textarea.lines(), &["中".to_string()]);
    }

    #[test]
    fn chinese_key_press_then_paste_is_deduped() {
        let mut textarea = TextArea::default();
        let mut recent_text_input = None;

        apply_event(
            &mut textarea,
            &mut recent_text_input,
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('中'),
                KeyModifiers::NONE,
                KeyEventKind::Press,
            )),
        );
        assert!(should_skip_duplicate_text_input(
            &mut recent_text_input,
            "中",
            RecentTextInputSource::Paste,
        ));

        assert_eq!(textarea.lines(), &["中".to_string()]);
    }

    #[test]
    fn backspace_deletes_cjk_char_correctly() {
        let mut textarea = TextArea::from(vec!["你好世界".to_string()]);
        textarea.set_lines(vec!["你好世界".to_string()], (0, 2)); // 光标在 "好" 和 "世" 之间

        backspace_delete_char(&mut textarea);

        assert_eq!(textarea.lines(), &["你世界".to_string()]);
        assert_eq!(textarea.cursor(), (0, 1));
    }

    #[test]
    fn backspace_at_line_start_merges_with_previous_line() {
        let mut textarea = TextArea::from(vec!["第一行".to_string(), "第二行".to_string()]);
        textarea.set_lines(vec!["第一行".to_string(), "第二行".to_string()], (1, 0)); // 光标在第二行行首

        backspace_delete_char(&mut textarea);

        assert_eq!(textarea.lines(), &["第一行第二行".to_string()]);
        assert_eq!(textarea.cursor(), (0, 3));
    }

    #[test]
    fn delete_forward_removes_cjk_char_after_cursor() {
        let mut textarea = TextArea::from(vec!["你好世界".to_string()]);
        textarea.set_lines(vec!["你好世界".to_string()], (0, 1)); // 光标在 "你" 和 "好" 之间

        delete_char_forward(&mut textarea);

        assert_eq!(textarea.lines(), &["你世界".to_string()]);
        assert_eq!(textarea.cursor(), (0, 1));
    }

    #[test]
    fn backspace_event_deletes_mixed_text() {
        let mut textarea = TextArea::from(vec!["hello你好world".to_string()]);
        textarea.set_lines(vec!["hello你好world".to_string()], (0, 7)); // 光标在 "好" 和 "w" 之间

        backspace_delete_char(&mut textarea);

        assert_eq!(textarea.lines(), &["hello你world".to_string()]);
        assert_eq!(textarea.cursor(), (0, 6));
    }
}
