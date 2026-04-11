use ratatui::{
    layout::Alignment,
    layout::Rect,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};
use tui_textarea::TextArea;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::completion_panel::CompletionPanel;
use crate::ai::prompt::MAX_INPUT_CHARS;

pub(in crate::ai::prompt::multiline) fn render_multiline_popup(
    f: &mut ratatui::Frame<'_>,
    textarea: &mut TextArea<'_>,
    status_msg: Option<&str>,
    completion_panel: Option<&CompletionPanel>,
) {
    let area = f.area();
    let popup_height = area.height.saturating_sub(4).clamp(10, 18);
    let popup_width = area.width.saturating_sub(2).clamp(60, 180).min(area.width);
    let popup = centered_rect(area, popup_width, popup_height);

    let popup_block = Block::default()
        .borders(Borders::ALL)
        .border_style(
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .title(
            Span::styled(
                " 📝 Compose ",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            ),
        );
    let inner = popup_block.inner(popup);
    let panel_height =
        completion_panel.map_or(0, |panel| (panel.items.len().min(5) as u16).saturating_add(2));
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(panel_height),
            Constraint::Length(2),
        ])
        .split(inner);

    f.render_widget(Clear, popup);
    f.render_widget(popup_block, popup);

    let n = textarea.lines().len();
    let current_content = textarea.lines().join("\n");
    let char_count = current_content.chars().count();

    let title = if char_count > MAX_INPUT_CHARS {
        format!(
            " Lines: {} | Chars: {} | ⚠ Exceeded (max: {}) ",
            n,
            char_count,
            MAX_INPUT_CHARS
        )
    } else {
        let warning_threshold = MAX_INPUT_CHARS * 90 / 100;
        if char_count > warning_threshold {
            format!(
                " Lines: {} | Chars: {} | ⚠ Approaching limit ",
                n,
                char_count
            )
        } else {
            format!(
                " Lines: {} | Chars: {} ",
                n,
                char_count
            )
        }
    };

    textarea.set_block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .title(title),
    );
    f.render_widget(&*textarea, chunks[0]);

    if let Some(panel) = completion_panel {
        let window_size = 5usize;
        let start = panel
            .selected_index
            .saturating_sub(window_size.saturating_sub(1))
            .min(panel.items.len().saturating_sub(window_size));
        let items: Vec<Line> = panel
            .items
            .iter()
            .enumerate()
            .skip(start)
            .take(window_size)
            .map(|(idx, item)| {
                let selected = idx == panel.selected_index;
                let selected_style = Style::default()
                    .fg(Color::Black)
                    .bg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD);
                Line::from(vec![
                    Span::styled(
                        if selected { "▶ " } else { "  " },
                        if selected {
                            selected_style
                        } else {
                            Style::default().fg(Color::DarkGray)
                        },
                    ),
                    Span::styled(
                        item,
                        if selected {
                            selected_style
                        } else {
                            Style::default().fg(Color::Gray)
                        },
                    ),
                ])
            })
            .collect();
        let panel_block = Block::default()
            .borders(Borders::ALL)
            .border_style(
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .title(
                Span::styled(
                    " Completions ",
                    Style::default()
                        .fg(Color::Gray)
                        .add_modifier(Modifier::BOLD),
                ),
            );
        let panel_para = Paragraph::new(items).block(panel_block);
        f.render_widget(panel_para, chunks[1]);
    }

    let help_lines = if completion_panel.is_some() {
        vec![
            Line::from(vec![
                Span::styled("↵", Style::default().fg(Color::Blue)),
                Span::raw(" select "),
                Span::styled("Esc", Style::default().fg(Color::Green)),
                Span::raw(" close "),
                Span::styled("⌨C+C", Style::default().fg(Color::Yellow)),
                Span::raw(" cancel "),
            ]),
            Line::from(vec![
                Span::styled("↑↓", Style::default().fg(Color::Blue)),
                Span::raw(" move "),
                Span::styled("Tab", Style::default().fg(Color::Blue)),
                Span::raw(" refresh "),
                Span::styled("↵+Alt/F2", Style::default().fg(Color::Green)),
                Span::raw(" send "),
            ]),
        ]
    } else {
        vec![
            Line::from(vec![
                Span::styled("↵", Style::default().fg(Color::Blue)),
                Span::raw(" newline "),
                Span::styled("Alt+↵/F2/Esc", Style::default().fg(Color::Green)),
                Span::raw(" send "),
                Span::styled("⌨C+C", Style::default().fg(Color::Yellow)),
                Span::raw(" cancel "),
            ]),
            Line::from(vec![
                Span::styled("↑↓/⌨P/N", Style::default().fg(Color::Blue)),
                Span::raw(" hist "),
                Span::styled("⌫", Style::default().fg(Color::Blue)),
                Span::raw(" edit "),
                Span::styled("⌨V", Style::default().fg(Color::Blue)),
                Span::raw(" paste "),
                Span::styled("F9", Style::default().fg(Color::Blue)),
                Span::raw(" last "),
                Span::styled("F10", Style::default().fg(Color::Blue)),
                Span::raw(" full "),
            ]),
        ]
    };
    f.render_widget(Paragraph::new(help_lines), chunks[2]);

    if let Some(msg) = status_msg {
        let status_width = chunks[2].width.saturating_sub(1) as usize;
        let status_text = truncate_display_width(msg, status_width);
        let status_para = Paragraph::new(Line::from(Span::styled(
            status_text,
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center);
        let status_area = Rect::new(chunks[2].x, chunks[2].y + 1, chunks[2].width, 1);
        f.render_widget(Clear, status_area);
        f.render_widget(status_para, status_area);
    }
}

fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    Rect::new(x, y, width, height)
}

fn truncate_display_width(text: &str, max_width: usize) -> String {
    if max_width == 0 || UnicodeWidthStr::width(text) <= max_width {
        return text.to_string();
    }

    let ellipsis = "...";
    if max_width <= ellipsis.len() {
        return ".".repeat(max_width);
    }

    let target_width = max_width - ellipsis.len();
    let mut out = String::new();
    let mut width = 0usize;
    for ch in text.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + ch_width > target_width {
            break;
        }
        out.push(ch);
        width += ch_width;
    }
    out.push_str(ellipsis);
    out
}

#[cfg(test)]
mod tests {
    use super::truncate_display_width;

    #[test]
    fn truncates_mixed_width_status_text_without_breaking_cells() {
        assert_eq!(truncate_display_width("已补全为 /agent", 10), "已补全...");
    }

    #[test]
    fn keeps_short_status_text_intact() {
        assert_eq!(truncate_display_width("Copied!", 20), "Copied!");
    }
}
