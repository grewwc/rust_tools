use std::io;

use crossterm::{
    event::{self, DisableBracketedPaste, EnableBracketedPaste, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, size as terminal_size},
};
use ratatui::{Terminal, backend::CrosstermBackend, layout::Rect};
use tui_textarea::TextArea;

use super::{
    MultilineHistoryState,
    completion_panel::{CompletionPanel, PendingTabCompletion},
    events::{EventLoopAction, handle_multiline_event},
    render::render_multiline_popup,
};
use crate::ai::prompt::{PromptEditor, interrupted_error};

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
                if let Event::Resize(_, _) = &event
                    && let Ok((w, h)) = terminal_size()
                {
                    let _ = terminal.resize(Rect::new(0, 0, w, h));
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

        let _ = terminal.clear();
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
                println!("[2m> {first}[0m");
            }
            for line in lines {
                println!("[2m  {line}[0m");
            }
        }
        Ok(result)
    }
}
