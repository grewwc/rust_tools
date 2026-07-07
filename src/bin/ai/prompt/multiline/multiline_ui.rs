use std::io;
use std::time::Duration;

use crossterm::{
    event::{self, DisableBracketedPaste, EnableBracketedPaste, Event},
    execute,
    terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size as terminal_size},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use tui_textarea::TextArea;

use super::{
    MultilineHistoryState,
    completion_panel::{CompletionPanel, PendingTabCompletion},
    events::{EventLoopAction, RecentTextInput, handle_multiline_event},
    render::render_multiline_popup,
};
use crate::ai::prompt::{PromptEditor, interrupted_error};

const COMPACT_VIEWPORT_HEIGHT: u16 = 8;
const MAX_VIEWPORT_HEIGHT: u16 = 20;
const VIEWPORT_CHROME_LINES: u16 = 5; // top margin + model line + spacer + help(2)
const MIN_TEXTAREA_LINES: u16 = 3;
const MAX_PREFILL_TEXTAREA_LINES: u16 = 12;

/// `Terminal::clear()` 在 inline viewport 下会先把真实终端 cursor 挪到 viewport 顶部
/// 再执行 `FromCursorDown`。如果 resize 突发期间又立刻来一轮 autoresize，ratatui 会
/// 以这个被挪到顶部的 cursor 重新计算 viewport 锚点，导致输入框/光标跳到 terminal
/// 顶部。这里清屏后把 cursor 放回清屏前的位置，只让它承担“清屏”职责，不污染下一轮
/// viewport 定位基准。
fn clear_inline_viewport_preserving_cursor<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
) {
    let cursor_before_clear = terminal.get_cursor_position().ok();
    let _ = terminal.clear();
    if let Some(cursor_before_clear) = cursor_before_clear {
        let _ = terminal.set_cursor_position(cursor_before_clear);
    }
}

fn multiline_viewport_height(terminal_rows: u16, prefill: Option<&str>) -> u16 {
    let available_rows = terminal_rows.saturating_sub(2).max(1);
    if prefill.is_none_or(str::is_empty) {
        // inline viewport 高度在创建时一次性固定，之后不随打字增长。空输入若给太小
        // （如 4 行），一旦开始输入，完整编辑器布局需要 8 行 chrome，textarea 会被挤没，
        // 导致“看不到输入文本”。因此空输入也必须给足能容纳编辑器的高度。
        return COMPACT_VIEWPORT_HEIGHT.min(available_rows);
    }
    let prefill_rows = prefill
        .map(|text| text.lines().count().max(1))
        .unwrap_or(1)
        .min(u16::MAX as usize) as u16;
    let prefill_viewport_height = prefill_rows
        .clamp(MIN_TEXTAREA_LINES, MAX_PREFILL_TEXTAREA_LINES)
        .saturating_add(VIEWPORT_CHROME_LINES);
    let desired = COMPACT_VIEWPORT_HEIGHT.max(prefill_viewport_height);
    desired.min(MAX_VIEWPORT_HEIGHT).min(available_rows)
}

/// 处理 `Event::Resize` 连续触发的情况：终端窗口拖动时 Resize 事件会快速连发，
/// 如果每次都重绘 inline viewport，会导致"画面撕裂"或闪烁。
/// 本函数先 drain 掉后续堆积的 Resize 事件，再做一次 full redraw。
fn handle_resize_burst<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>) -> io::Result<()> {
    // 1) drain 掉所有后续的 Resize 事件，避免重复触发重绘
    while event::poll(Duration::ZERO).unwrap_or(false) {
        match event::read() {
            Ok(Event::Resize(_, _)) => continue,
            // 其他 case：resize 突发期间夹杂的非 Resize 事件直接丢弃，
            // 回到 UI 主循环重新处理
            Ok(_) | Err(_) => break,
        }
    }

    // 2) 通知 ratatui 终端尺寸已变化（宽度变化影响文本折行），
    //    然后清除 inline viewport。不再手动 MoveUp + Clear：
    //    viewport_height 是启动时的快照，resize 后手动 MoveUp 的行数
    //    与 ratatui 内部光标状态不一致，导致 clear() 清错区域，旧
    //    viewport 内容（包括 cursor 块 ▌）残留在 scrollback 中反复堆叠。
    let _ = terminal.hide_cursor();
    let _ = terminal.autoresize();
    clear_inline_viewport_preserving_cursor(terminal);
    Ok(())
}

impl PromptEditor {
    pub(in crate::ai::prompt) fn read_multi_line_tui(&mut self) -> io::Result<Option<String>> {
        enable_raw_mode()?;
        let _ = execute!(io::stdout(), EnableBracketedPaste);

        // Inline viewport 初始化会通过 append_lines() 真实撑开终端区域。空输入默认保持
        // 紧凑，避免每轮回答后和下一轮光标之间出现大段空白；编辑已有内容时再按预填行数
        // 放大，给 textarea 保留足够空间。
        let viewport_height = terminal_size()
            .map(|(_, h)| multiline_viewport_height(h, self.pending_prefill.as_deref()))
            .unwrap_or(COMPACT_VIEWPORT_HEIGHT);

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
            // 预填内容（编辑已有 memo 场景）：按行载入 textarea，读取后清空。
            let mut textarea: TextArea = match self.pending_prefill.take() {
                Some(text) => TextArea::from(text.lines().map(|l| l.to_string())),
                None => TextArea::default(),
            };
            let mut history = MultilineHistoryState::new(self.multiline_history_entries());
            let mut status_msg: Option<String> = None;
            let mut pending_tab_completion: Option<PendingTabCompletion> = None;
            let mut completion_panel: Option<CompletionPanel> = None;
            let mut recent_text_input: Option<RecentTextInput> = None;

            loop {
                terminal
                    .draw(|f| {
                        render_multiline_popup(
                            f,
                            &mut textarea,
                            status_msg.as_deref(),
                            completion_panel.as_ref(),
                            &self.current_model_label,
                            self.session_topic.as_deref(),
                        );
                    })
                    .map_err(|e| io::Error::other(e.to_string()))?;

                let event = event::read().map_err(|e| io::Error::other(e.to_string()))?;
                if let Event::Resize(_, _) = &event {
                    handle_resize_burst(&mut terminal)?;
                }

                match handle_multiline_event(
                    event,
                    &mut textarea,
                    &mut history,
                    &mut status_msg,
                    &mut pending_tab_completion,
                    &mut completion_panel,
                    &mut recent_text_input,
                    &self.session_image_dir,
                )? {
                    EventLoopAction::Continue => {}
                    EventLoopAction::Submit(result) => break Ok(result),
                }
            }
        })();

        // 退出 TUI：清除 ratatui 的 inline viewport 残留内容，
        // `FromCursorDown` 确保光标以下的区域干净，不污染 scrollback。
        let _ = terminal.clear();
        let _ = execute!(io::stdout(), Clear(ClearType::FromCursorDown));
        let _ = terminal.show_cursor();
        let _ = execute!(io::stdout(), DisableBracketedPaste);
        let _ = disable_raw_mode();

        let result = match result {
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                return interrupted_error();
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
}

#[cfg(test)]
mod tests {
    use super::{clear_inline_viewport_preserving_cursor, multiline_viewport_height};
    use ratatui::{
        Terminal, TerminalOptions, Viewport,
        backend::TestBackend,
        layout::{Position, Rect},
        widgets::{Paragraph, Widget},
    };

    #[test]
    fn clear_inline_viewport_preserves_existing_cursor_anchor() {
        let backend = TestBackend::new(24, 8);
        let mut terminal = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(3),
            },
        )
        .unwrap();

        terminal
            .insert_before(2, |buf| {
                Paragraph::new(vec!["line 1".into(), "line 2".into()]).render(buf.area, buf);
            })
            .unwrap();

        let mut viewport_area = Rect::ZERO;
        terminal
            .draw(|f| {
                viewport_area = f.area();
                f.render_widget(Paragraph::new("prompt"), f.area());
                f.set_cursor_position((viewport_area.x + 4, viewport_area.y + 1));
            })
            .unwrap();

        let expected = Position::new(viewport_area.x + 4, viewport_area.y + 1);
        terminal.backend_mut().assert_cursor_position(expected);

        clear_inline_viewport_preserving_cursor(&mut terminal);

        terminal.backend_mut().assert_cursor_position(expected);
    }

    #[test]
    fn multiline_viewport_height_stays_compact_for_empty_prompt() {
        assert_eq!(multiline_viewport_height(30, None), 8);
        assert_eq!(multiline_viewport_height(30, Some("")), 8);
        assert_eq!(multiline_viewport_height(30, Some("one line")), 8);
    }

    #[test]
    fn multiline_viewport_height_expands_for_prefill_but_caps_to_available_rows() {
        let prefill = (0..20)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(multiline_viewport_height(40, Some(&prefill)), 17);
        assert_eq!(multiline_viewport_height(10, Some(&prefill)), 8);
        assert_eq!(multiline_viewport_height(4, Some(&prefill)), 2);
        assert_eq!(multiline_viewport_height(4, None), 2);
    }
}
