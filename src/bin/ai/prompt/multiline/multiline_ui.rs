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
    events::{EventLoopAction, handle_multiline_event},
    render::render_multiline_popup,
};
use crate::ai::prompt::{PromptEditor, interrupted_error};

/// 处理 `Event::Resize` 连续触发的情况：终端窗口拖动时 Resize 事件会快速连发，
/// 如果每次都重绘 inline viewport，会导致"画面撕裂"或闪烁。
/// 本函数先 drain 掉后续堆积的 Resize 事件，再做一次 full redraw。
///
/// 注意：某些 VS Code 终端 / 终端模拟器对 ratatui inline viewport
/// 支持不完善，`Borders::TOP/BOTTOM` 可能污染 scrollback 历史记录，
/// 这是已知的终端兼容性问题。
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

    // 2) 清除整个 viewport 区域。ratatui inline viewport 在 resize 时
    //    旧的 Borders::TOP 可能残留在光标上方，仅 FromCursorDown 不够。
    //    使用实际终端高度而非 viewport 高度来上移光标，确保行回绕后仍然
    //    能覆盖到旧边框位置。
    let clear_rows = terminal_size().map(|(_, h)| h).unwrap_or(40);
    let _ = execute!(
        io::stdout(),
        crossterm::cursor::MoveUp(clear_rows),
        crossterm::cursor::MoveToColumn(0),
        Clear(ClearType::FromCursorDown)
    );
    let _ = terminal.clear();
    Ok(())
}

impl PromptEditor {
    pub(in crate::ai::prompt) fn read_multi_line_tui(&mut self) -> io::Result<Option<String>> {
        enable_raw_mode()?;
        let _ = execute!(io::stdout(), EnableBracketedPaste);

        // viewport 高度 = 1 行顶部间距 + textarea + 2 行帮助行。
        // 设为 20 行让输入框有足够的编辑空间，补全面板弹出时 textarea 会自动退让到 1 行。
        // 高度过大会导致 append_lines() 在终端底部产生大量换行，
        // 不仅造成光标与上次输出之间有大段空白，还可能在终端底部覆盖掉
        // 更多的 assistant 输出历史行。
        let viewport_height = terminal_size()
            .map(|(_, h)| h.saturating_sub(2).clamp(12, 20))
            .unwrap_or(20);

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
            let mut accept_release = false;
            let mut status_msg: Option<String> = None;
            let mut pending_tab_completion: Option<PendingTabCompletion> = None;
            let mut completion_panel: Option<CompletionPanel> = None;

            loop {
                terminal
                    .draw(|f| {
                        render_multiline_popup(
                            f,
                            &mut textarea,
                            status_msg.as_deref(),
                            completion_panel.as_ref(),
                            &self.current_model_label,
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
                    &mut accept_release,
                    &mut status_msg,
                    &mut pending_tab_completion,
                    &mut completion_panel,
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
