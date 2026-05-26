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

/// ?? `Event::Resize`????????? Resize ????????
/// ?????? inline viewport???????"??????"??
/// ???? draw ??????????? full redraw?
///
/// ?????? VS Code ???? / ????? ratatui inline viewport
/// ?????? `Borders::TOP/BOTTOM` ???? scrollback?????
/// ??????????
fn handle_resize_burst<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
) -> io::Result<()> {
    // 1) drain????????? Resize ???????
    while event::poll(Duration::ZERO).unwrap_or(false) {
        match event::read() {
            Ok(Event::Resize(_, _)) => continue,
            // ???? case?resize ?????????? Resize ???
            // ?? UI ???????
            Ok(_) | Err(_) => break,
        }
    }

    // 2) ?????????????????????????
    //    ratatui ? terminal.clear() ? Inline ???????? buffer?
    //    ???? crossterm `FromCursorDown`????????
    //    "???????"?????????? draw ????????
    //    ?????????
    let _ = terminal.clear();
    let _ = execute!(io::stdout(), Clear(ClearType::FromCursorDown));
    Ok(())
}

impl PromptEditor {
    pub(in crate::ai::prompt) fn read_multi_line_tui(&mut self) -> io::Result<Option<String>> {
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

        // ?? TUI?? ratatui ? inline viewport ?????????
        // `FromCursorDown` ????????????????? scrollback?
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
